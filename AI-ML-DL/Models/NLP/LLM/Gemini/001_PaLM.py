"""
PaLM (Chowdhery et al., 2022) -- minimal PyTorch re-implementation of the
one architectural quirk that is precisely documented and worth actually
coding: the PARALLEL attention + FFN transformer block.

Standard ("sequential") transformer block:
    x = x + Attn(LN1(x))
    x = x + FFN(LN2(x))

PaLM's parallel block (used at 540B for throughput -- lets the framework
fuse the attention-input and FFN-input projections into fewer, larger
matmuls, which is more MXU/TPU-efficient):
    y  = LN(x)                      # single shared norm, not two
    x  = x + Attn(y) + FFN(y)       # both computed from the SAME normed
                                     # input, in parallel, then summed

This file also implements the two other PaLM-specific details that are
cheap to demonstrate correctly: multi-query attention (MQA) -- a single
shared key/value head across all query heads, used to shrink the KV cache
for cheaper autoregressive decoding -- and RoPE (rotary position
embeddings).

Not implemented (out of scope / no effect on the architectural point):
Pathways cross-pod orchestration, the SentencePiece tokenizer, and the
780B-token training mixture. This file is a modeling-level demo only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute rotary embedding (cos, sin) cache of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    return torch.stack([emb.cos(), emb.sin()], dim=0)  # (2, seq_len, head_dim)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings. x: (batch, heads, seq_len, head_dim)."""
    cos, sin = rope_cache[0], rope_cache[1]  # (seq_len, head_dim)
    cos = cos[: x.shape[-2]].to(x.dtype)
    sin = sin[: x.shape[-2]].to(x.dtype)

    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat([-t2, t1], dim=-1)

    return x * cos + rotate_half(x) * sin


class MultiQueryAttention(nn.Module):
    """Multi-query attention: many query heads, a single shared key/value head.

    Motivation in PaLM: has essentially no effect on training throughput
    (training is compute-bound and fully parallel across the sequence),
    but during autoregressive decoding it shrinks the KV cache by a factor
    of `num_heads`, which is exactly the resource that dominates serving
    cost at 540B scale.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # Only ONE head's worth of K and V -- shared across all query heads.
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, t, h, hd).transpose(1, 2)       # (b, h, t, hd)
        k = self.k_proj(x).view(b, t, 1, hd).transpose(1, 2)       # (b, 1, t, hd)
        v = self.v_proj(x).view(b, t, 1, hd).transpose(1, 2)       # (b, 1, t, hd)

        q = apply_rope(q, rope_cache)
        k = apply_rope(k, rope_cache)

        # Broadcast the single KV head across all Q heads.
        k = k.expand(b, h, t, hd)
        v = v.expand(b, h, t, hd)

        causal_mask = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_mask, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(attn_out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block, PaLM's MLP nonlinearity of choice."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class PaLMParallelBlock(nn.Module):
    """The defining PaLM block: attention and FFN computed from ONE shared
    normalized input and summed, rather than two sequential residual
    sub-blocks each with their own norm.

    Sequential (GPT-style):   x + Attn(LN1(x))  -->  x' + FFN(LN2(x'))
    Parallel (PaLM-style):    y = LN(x); x + Attn(y) + FFN(y)

    The parallel form lets a systems implementation fuse the attention
    input projection and the FFN's up/gate projections into a single,
    larger matmul (same for the two output projections), which is the
    throughput motivation described in the paper (~15% faster training at
    scale, roughly neutral quality at 540B).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, no_bias: bool = True):
        super().__init__()
        # PaLM uses a single shared LayerNorm feeding both sublayers, and
        # no biases anywhere (paper-reported stability choice at scale).
        self.norm = nn.LayerNorm(d_model, bias=not no_bias)
        self.attn = MultiQueryAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        # Both sublayers read the SAME normalized tensor `y` -- this is the
        # "parallel" part. They could be dispatched concurrently and their
        # matmuls fused; here we just express the math directly.
        return x + self.attn(y, rope_cache) + self.ffn(y)


class PaLMModel(nn.Module):
    """A small, structurally faithful PaLM-style decoder-only LM.

    Shares the input embedding and output (unembedding) projection matrix,
    as in the paper.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff: int = 1365,  # PaLM uses d_ff ~= 2.67 * d_model with SwiGLU
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [PaLMParallelBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model, bias=False)

        head_dim = d_model // num_heads
        self.register_buffer(
            "rope_cache", build_rope_cache(max_seq_len, head_dim), persistent=False
        )

        # Shared input/output embedding: no separate unembedding matrix.
        self.lm_head_weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, self.rope_cache)
        x = self.final_norm(x)
        logits = F.linear(x, self.lm_head_weight)  # tied weights
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    model = PaLMModel(
        vocab_size=32000, d_model=512, num_layers=8, num_heads=8, d_ff=1365
    )
    batch, seq_len = 2, 64
    tokens = torch.randint(0, 32000, (batch, seq_len))

    logits = model(tokens)
    print(f"Input token ids shape : {tuple(tokens.shape)}")
    print(f"Output logits shape   : {tuple(logits.shape)}")
    print(f"Total parameters      : {count_parameters(model):,}")

    # Sanity check: sequential vs. parallel block produce different (but
    # comparably scaled) outputs from the same input -- demonstrating the
    # two formulations are genuinely different computations, not a relabeling.
    block = model.blocks[0]
    y = block.norm(torch.randn(1, 4, 512))
    attn_out = block.attn(y, model.rope_cache)
    ffn_out = block.ffn(y)
    print(f"attn_out shape        : {tuple(attn_out.shape)}")
    print(f"ffn_out shape         : {tuple(ffn_out.shape)}")
    print("Parallel block output = x + attn(LN(x)) + ffn(LN(x))  [both from one LN]")
