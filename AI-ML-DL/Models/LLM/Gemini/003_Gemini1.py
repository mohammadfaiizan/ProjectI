"""
Gemini 1.0 (Google DeepMind, 2023) -- toy demonstration of the ONE
architectural claim Google explicitly and repeatedly makes for this model:
NATIVE multimodality, i.e. text, image (and, by extension, audio/video
sampled as frames) are projected into a SHARED embedding space and consumed
by ONE joint transformer stack from the very first pretraining step --
as opposed to a "bolt-on adapter" design, where a separately pretrained
vision encoder's output is bridged into a frozen/lightly-tuned text-only
backbone after the fact.

This file implements BOTH designs side by side so the structural
difference is concrete and comparable:

  1. NativeMultimodalEncoder (what Gemini claims): token ids and
     patch/frame embeddings are projected by modality-specific *linear*
     input projections into ONE shared d_model space, concatenated into a
     single sequence, and fed through a single shared transformer stack
     end to end. Every transformer layer's parameters receive gradient
     signal from both modalities on every training step, by construction.

  2. BoltOnAdapterEncoder (the contrasting baseline it's positioned
     against): a frozen "pretrained" text backbone processes only text; a
     separately pretrained vision tower produces image features; a small
     adapter (a low-rank bridge) projects those features into the frozen
     backbone's space and injects them via a narrow cross-attention
     module. Gradient signal to the core text backbone is blocked
     (simulated here via `requires_grad_(False)` on the "frozen" backbone),
     so only the adapter and vision tower learn from image data.

Not implemented: Gemini's actual tokenizer, TPU serving stack, or any
Google-confirmed layer/head/dimension numbers (none are public -- see
003_Gemini1.md, Section 11).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerStack(nn.Module):
    """A minimal shared decoder-only transformer stack, used by both designs
    below so the comparison isolates the multimodal-fusion question rather
    than differing in unrelated architectural details."""

    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "attn": nn.MultiheadAttention(
                            d_model, num_heads, batch_first=True
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_ff),
                            nn.GELU(),
                            nn.Linear(d_ff, d_model),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        causal_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1
        )
        for layer in self.layers:
            y = layer["norm1"](x)
            attn_out, _ = layer["attn"](y, y, y, attn_mask=causal_mask, need_weights=False)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm2"](x))
        return x


class NativeMultimodalEncoder(nn.Module):
    """'Native' multimodal design: separate lightweight, LINEAR input
    projections per modality feed into ONE shared transformer stack that is
    trained jointly on all modalities from step one. This is the structural
    property Google's "trained natively across text, image, audio, and
    video from the start of pretraining" claim implies -- no modality has
    its own deep sub-network or frozen backbone; every layer downstream of
    the input projections is shared and jointly optimized.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        patch_dim: int,
    ):
        super().__init__()
        # Modality-specific projections are shallow ON PURPOSE: the point of
        # "native" fusion is that depth/reasoning capacity lives in the
        # SHARED stack, not in per-modality towers.
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.image_patch_proj = nn.Linear(patch_dim, d_model)  # e.g. flattened image/video patches
        self.modality_type_embed = nn.Embedding(2, d_model)  # 0=text, 1=visual

        self.shared_stack = TransformerStack(d_model, num_layers, num_heads, d_ff)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, text_len) int64
        patches: (batch, num_patches, patch_dim) float -- image or sampled
                 video-frame patches, already flattened.
        Returns the shared-stack output over the full interleaved sequence
        (text tokens followed by visual tokens, for simplicity).
        """
        text_tok = self.text_embed(token_ids)
        text_tok = text_tok + self.modality_type_embed(
            torch.zeros(token_ids.shape[:2], dtype=torch.long, device=token_ids.device)
        )

        visual_tok = self.image_patch_proj(patches)
        visual_tok = visual_tok + self.modality_type_embed(
            torch.ones(patches.shape[:2], dtype=torch.long, device=patches.device)
        )

        # ONE shared embedding space, ONE joint sequence, ONE transformer.
        joint_seq = torch.cat([text_tok, visual_tok], dim=1)
        return self.final_norm(self.shared_stack(joint_seq))


class BoltOnAdapterEncoder(nn.Module):
    """Contrasting baseline: a FROZEN text-only backbone (simulating "already
    pretrained on text alone, not touched by image gradients") plus a
    separately-parameterized vision tower and a narrow adapter bridge that
    injects visual features into the backbone via cross-attention. Only the
    adapter (and vision tower) receive gradients from image data; the core
    reasoning stack never does.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        patch_dim: int,
        adapter_dim: int = 64,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.frozen_backbone = TransformerStack(d_model, num_layers, num_heads, d_ff)
        for p in self.frozen_backbone.parameters():
            p.requires_grad_(False)  # pretrained text-only; not shaped by image data

        # Separate vision tower (its own depth, its own parameters).
        self.vision_tower = nn.Sequential(
            nn.Linear(patch_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        # Narrow bridge into the frozen backbone's space + one cross-attn.
        self.adapter_proj = nn.Linear(adapter_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        text_tok = self.text_embed(token_ids)
        text_repr = self.frozen_backbone(text_tok)  # never sees image gradients

        visual_feat = self.vision_tower(patches)
        visual_tok = self.adapter_proj(visual_feat)

        # Text representations attend to visual tokens through a narrow,
        # separately-trained bridge -- this is where (and only where) the
        # two modalities meet.
        bridged, _ = self.cross_attn(text_repr, visual_tok, visual_tok, need_weights=False)
        return self.final_norm(text_repr + bridged)


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size, d_model, num_layers, num_heads, d_ff = 8000, 256, 6, 8, 1024
    patch_dim = 768  # e.g. a flattened 16x16x3 image patch after linear patchify
    batch, text_len, num_patches = 2, 20, 16

    token_ids = torch.randint(0, vocab_size, (batch, text_len))
    patches = torch.randn(batch, num_patches, patch_dim)

    native = NativeMultimodalEncoder(
        vocab_size, d_model, num_layers, num_heads, d_ff, patch_dim
    )
    bolt_on = BoltOnAdapterEncoder(
        vocab_size, d_model, num_layers, num_heads, d_ff, patch_dim
    )

    native_out = native(token_ids, patches)
    bolt_on_out = bolt_on(token_ids, patches)

    print("=== Native multimodal encoder (Gemini-style claim) ===")
    print(f"Joint sequence output shape : {tuple(native_out.shape)}  "
          f"(text_len + num_patches = {text_len + num_patches})")
    print(f"Total params                : {count_total(native):,}")
    print(f"Trainable params            : {count_trainable(native):,}  "
          f"(== total: every layer sees gradients from BOTH modalities)\n")

    print("=== Bolt-on adapter encoder (contrasting baseline) ===")
    print(f"Text-length output shape    : {tuple(bolt_on_out.shape)}")
    print(f"Total params                : {count_total(bolt_on):,}")
    print(f"Trainable params            : {count_trainable(bolt_on):,}  "
          f"(< total: the frozen backbone is excluded)")
    frozen = count_total(bolt_on) - count_trainable(bolt_on)
    print(f"Frozen (text-only) params   : {frozen:,}  "
          f"-- these weights are NEVER shaped by image gradients")
