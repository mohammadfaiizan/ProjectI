"""
006_GPT4.py -- Educational reconstruction of GPT-4 (OpenAI, 2023) mechanisms.

IMPORTANT / SCOPE: OpenAI has never disclosed GPT-4's architecture, parameter
count, or training data. Nothing in this file is a reproduction of the real
GPT-4. This is a from-scratch, illustrative implementation of two things that
are discussable about GPT-4's public story:

  1. CONFIRMED capability: multimodal (text + image) input into a decoder-only
     causal transformer. This file implements a minimal vision-patch encoder
     whose output embeddings are projected into the same space as text token
     embeddings and spliced into a single causal sequence -- a standard,
     reasonable design pattern from the contemporaneous multimodal-LM
     literature (Flamingo/PaLI/LLaVA-era approaches), NOT a claim about how
     OpenAI actually built GPT-4's vision pathway (undisclosed).

  2. UNCONFIRMED RUMOR, reconstructed for pedagogy only: the SemiAnalysis /
     George Hotz claim that GPT-4 is a sparse mixture-of-experts model
     (8 experts, ~220B params each, ~1.8T total, ~2 experts active per token).
     This file implements a toy top-2-routed sparse MoE feed-forward block so
     that the *mechanism* being rumored (sparse expert routing, load balancing
     loss, all-to-all-style dispatch) is concrete and inspectable. This is
     clearly labeled everywhere below as reconstructing a RUMOR, not a fact.

Every class below is toy-scale (tens of thousands to low millions of
parameters) purely to make the forward pass runnable and legible on a laptop
CPU. It has no relationship to GPT-4's actual (undisclosed) scale.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Multimodal input stitching (CONFIRMED capability, illustrative design)
# ---------------------------------------------------------------------------

class VisionPatchEncoder(nn.Module):
    """Toy ViT-style patch encoder: splits an image into fixed patches,
    linearly embeds each patch, and adds learned positional embeddings.

    This models the *idea* that GPT-4 accepts image input, which is
    confirmed. The specific encoder design here (linear patch projection,
    no conv stem, no pretraining) is a simplification chosen for clarity,
    not a claim about GPT-4's real (undisclosed) vision encoder.
    """

    def __init__(self, image_size=32, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, images):
        # images: (batch, channels, H, W) -> non-overlapping patches
        b, c, h, w = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)  # (b, c, h/p, w/p, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(b, -1, c * p * p)
        x = self.patch_proj(patches)
        x = x + self.pos_embed
        return x  # (batch, num_patches, embed_dim)


class MultimodalTokenFuser(nn.Module):
    """Projects vision-encoder output into the LLM's token embedding space
    and concatenates it with text token embeddings to form one causal
    sequence: [image_tokens, text_tokens]. This is the "bolt-on adapter"
    pattern -- a separate vision tower feeding a shared decoder -- contrasted
    in 007_GPT4o_Multimodal.py with a more genuinely joint approach.
    """

    def __init__(self, vision_dim, text_dim):
        super().__init__()
        self.vision_to_text = nn.Linear(vision_dim, text_dim)

    def forward(self, vision_embeds, text_embeds):
        projected_vision = self.vision_to_text(vision_embeds)
        return torch.cat([projected_vision, text_embeds], dim=1)


# ---------------------------------------------------------------------------
# 2. Sparse Mixture-of-Experts block (RECONSTRUCTS AN UNCONFIRMED RUMOR)
# ---------------------------------------------------------------------------

class Top2MoEFeedForward(nn.Module):
    """Toy top-2-routed sparse MoE feed-forward layer.

    This reconstructs the *mechanism* alleged by the SemiAnalysis/Hotz GPT-4
    rumor (8 experts, top-2 routing) at toy scale. OpenAI has never confirmed
    GPT-4 uses MoE at all. Real large-scale MoE systems (e.g. Switch
    Transformer, GShard, Mixtral) additionally require expert-parallel
    placement across devices and all-to-all dispatch/combine communication;
    this single-process implementation keeps everything local and simply
    demonstrates routing + load-balancing loss, which is the part relevant
    to a research-interview discussion of *why* MoE at this scale is hard.
    """

    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        )

    def forward(self, x):
        # x: (batch, seq, d_model)
        b, s, d = x.shape
        flat = x.reshape(-1, d)  # (b*s, d)

        logits = self.gate(flat)  # (b*s, num_experts)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (b*s, top_k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renormalize

        # Load-balancing auxiliary loss (Switch-Transformer-style): encourages
        # uniform routing across experts so a handful of experts don't absorb
        # most tokens -- one of the genuinely hard systems problems in real
        # MoE training that the GPT-4 rumor implicitly raises.
        density = probs.mean(dim=0)  # avg gate probability per expert
        importance = torch.zeros(self.num_experts, device=x.device)
        importance.scatter_add_(0, top_idx.reshape(-1), torch.ones_like(top_idx.reshape(-1), dtype=x.dtype))
        importance = importance / importance.sum().clamp(min=1)
        load_balance_loss = self.num_experts * (density * importance).sum()

        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            expert_ids = top_idx[:, slot]
            gate_weight = top_probs[:, slot].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = expert_ids == e
                if mask.any():
                    out[mask] += gate_weight[mask] * self.experts[e](flat[mask])

        return out.view(b, s, d), load_balance_loss


# ---------------------------------------------------------------------------
# 3. Minimal decoder-only transformer block, with optional MoE FFN
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (b, heads, s, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, d)
        return self.out_proj(ctx)


class GPT4StyleBlock(nn.Module):
    """One decoder block. `use_moe=True` swaps the dense FFN for the toy
    Top2MoEFeedForward above, to make the rumor's mechanism inspectable
    end-to-end within a full block."""

    def __init__(self, d_model, n_heads, d_ff, num_experts=8, top_k=2, use_moe=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.use_moe = use_moe
        if use_moe:
            self.ffn = Top2MoEFeedForward(d_model, d_ff, num_experts, top_k)
        else:
            self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        normed = self.ln2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed)
            x = x + ffn_out
            return x, aux_loss
        else:
            x = x + self.ffn(normed)
            return x, torch.tensor(0.0, device=x.device)


class GPT4StyleModel(nn.Module):
    """Toy decoder-only LM: multimodal input fusion + a stack of blocks
    (optionally sparse-MoE-routed) + LM head. See module docstring: this is
    an educational reconstruction, not a description of the real GPT-4."""

    def __init__(self, vocab_size, d_model=64, n_layers=4, n_heads=4, d_ff=128,
                 max_seq_len=256, num_experts=8, top_k=2, use_moe=True,
                 image_size=32, patch_size=8):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.vision_encoder = VisionPatchEncoder(image_size, patch_size, embed_dim=d_model)
        self.fuser = MultimodalTokenFuser(vision_dim=d_model, text_dim=d_model)
        self.blocks = nn.ModuleList(
            GPT4StyleBlock(d_model, n_heads, d_ff, num_experts, top_k, use_moe)
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, text_ids, images=None):
        text_embeds = self.token_embed(text_ids)
        text_embeds = text_embeds + self.pos_embed[:, : text_embeds.size(1), :]

        if images is not None:
            vision_embeds = self.vision_encoder(images)
            x = self.fuser(vision_embeds, text_embeds)
        else:
            x = text_embeds

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 1000
    batch_size = 2
    text_len = 12
    model = GPT4StyleModel(
        vocab_size=vocab_size, d_model=64, n_layers=4, n_heads=4, d_ff=128,
        max_seq_len=256, num_experts=8, top_k=2, use_moe=True,
        image_size=32, patch_size=8,
    )

    text_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    images = torch.randn(batch_size, 3, 32, 32)

    logits, aux_loss = model(text_ids, images=images)
    print("=== GPT-4-style toy model (multimodal fusion + rumored MoE FFN) ===")
    print(f"input text shape:   {tuple(text_ids.shape)}")
    print(f"input image shape:  {tuple(images.shape)}")
    print(f"num image patches:  {model.vision_encoder.num_patches}")
    print(f"output logits shape:{tuple(logits.shape)}  (image_tokens + text_tokens, vocab)")
    print(f"MoE load-balance aux loss: {aux_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    moe_params = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    print(f"total toy params:  {total_params:,}")
    print(f"MoE FFN params:    {moe_params:,} ({moe_params / total_params:.1%} of total)")

    # Text-only forward pass (no image attached), for comparison.
    logits_text_only, aux_loss_text_only = model(text_ids, images=None)
    print(f"\ntext-only logits shape: {tuple(logits_text_only.shape)}")
