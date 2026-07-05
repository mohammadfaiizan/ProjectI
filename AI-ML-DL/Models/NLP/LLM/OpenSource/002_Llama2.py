"""
Llama 2 (Touvron et al., 2023) -- "Llama 2: Open Foundation and Fine-Tuned Chat Models"

Demonstrates Grouped-Query Attention (GQA, Ainslie et al. 2023) with a configurable
n_kv_heads, the single architectural change Llama 2 makes relative to LLaMA 1's block
(RMSNorm + RoPE + SwiGLU are otherwise unchanged -- see 001_Llama1.py for those).

Real released configuration:
    - Llama-2-7B / 13B:  n_heads == n_kv_heads (standard MHA), as in LLaMA 1.
    - Llama-2-70B:       n_heads = 64, n_kv_heads = 8 -- an 8x reduction in KV
                         projections relative to full MHA, adopted specifically at
                         the largest/most inference-expensive scale to cut KV-cache
                         memory and decode-time memory-bandwidth cost.

GQA partitions the n_heads query heads into n_kv_heads groups; every query head in a
group reads from the *same* shared K/V head. Setting n_kv_heads = n_heads recovers
standard MHA; setting n_kv_heads = 1 recovers MQA (Shazeer, 2019). This file
implements the general case so the same module class serves 7B/13B (MHA) and
70B (GQA) configs, exactly as in the real released checkpoints.

Context length is also doubled in Llama 2 (4096 vs. LLaMA 1's 2048) -- reflected here
only in the default config value, since the RoPE mechanics are identical to LLaMA 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Llama2Config:
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 8       # == n_heads for 7B/13B (MHA); e.g. n_heads/8 for 70B-style GQA
    vocab_size: int = 32_000  # unchanged from LLaMA 1
    max_seq_len: int = 4096   # doubled from LLaMA 1's 2048
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """x: (batch, seq_len, n_heads, head_dim)."""
    *prefix, seq_len, n_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*prefix, seq_len, n_heads, head_dim // 2, 2))
    freqs = rope_freqs[:seq_len].view(1, seq_len, 1, head_dim // 2)
    x_out = torch.view_as_real(x_complex * freqs).reshape(*prefix, seq_len, n_heads, head_dim)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    Causal self-attention with n_heads query heads and n_kv_heads (<= n_heads) shared
    key/value heads. n_heads must be divisible by n_kv_heads; each group of
    (n_heads // n_kv_heads) query heads reads from one shared K/V head.

    n_kv_heads == n_heads  -> standard MHA (Llama-2-7B/13B).
    n_kv_heads == 1        -> MQA.
    1 < n_kv_heads < n_heads -> GQA (Llama-2-70B uses n_kv_heads = 8 with n_heads = 64).
    """

    def __init__(self, cfg: Llama2Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # query heads sharing each KV head
        self.head_dim = cfg.dim // cfg.n_heads

        # Query projection produces the full n_heads; K/V projections produce only
        # n_kv_heads worth of dimensions -- this is precisely where GQA's parameter
        # and KV-cache savings come from relative to full MHA.
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Expand (bsz, n_kv_heads, seq_len, head_dim) -> (bsz, n_kv_heads * n_rep, seq_len, head_dim)
        by repeating each KV head n_rep times so it lines up with its group of query heads.
        This materializes the repetition for clarity; a production kernel (e.g. FlashAttention
        with GQA support) does this implicitly without the memory duplication.
        """
        if n_rep == 1:
            return x
        bsz, n_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        q = q.transpose(1, 2)                      # (bsz, n_heads, seq_len, head_dim)
        k = self._repeat_kv(k.transpose(1, 2), self.n_rep)  # (bsz, n_heads, seq_len, head_dim)
        v = self._repeat_kv(v.transpose(1, 2), self.n_rep)  # (bsz, n_heads, seq_len, head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """
        KV cache cost per token per layer, in bytes, for K and V combined:
        2 (K and V) * n_kv_heads * head_dim * dtype_bytes.
        This is the quantity GQA directly reduces relative to MHA (n_kv_heads = n_heads).
        """
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes


class SwiGLU(nn.Module):
    def __init__(self, cfg: Llama2Config):
        super().__init__()
        hidden_dim = int(2 * (4 * cfg.dim) / 3)
        hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama2Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama2Model(nn.Module):
    def __init__(self, cfg: Llama2Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        head_dim = cfg.dim // cfg.n_heads
        rope_freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x, self.rope_freqs)
        x = self.norm(x)
        return self.output(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len = 2, 128
    tokens = None

    print("=" * 80)
    print("Llama-2-7B/13B-style config: standard MHA (n_kv_heads == n_heads)")
    print("=" * 80)
    mha_cfg = Llama2Config(dim=512, n_layers=4, n_heads=8, n_kv_heads=8, max_seq_len=4096)
    mha_model = Llama2Model(mha_cfg)
    tokens = torch.randint(0, mha_cfg.vocab_size, (batch_size, seq_len))
    mha_logits = mha_model(tokens)
    mha_kv_bytes = mha_model.layers[0].attn.kv_cache_bytes_per_token()
    print(f"n_heads={mha_cfg.n_heads}, n_kv_heads={mha_cfg.n_kv_heads}")
    print(f"Output logits shape: {tuple(mha_logits.shape)}")
    print(f"Total parameters:    {mha_model.num_params():,}")
    print(f"KV cache bytes/token/layer (fp16): {mha_kv_bytes}")

    print()
    print("=" * 80)
    print("Llama-2-70B-style config: GQA (n_heads=8 groups sharing n_kv_heads=2, ratio 4:1)")
    print("=" * 80)
    # Ratio mirrors the real 70B config (64 query heads : 8 kv heads = 8:1); scaled down
    # here to n_heads=8 : n_kv_heads=2 (4:1) purely to keep the illustrative dims small.
    gqa_cfg = Llama2Config(dim=512, n_layers=4, n_heads=8, n_kv_heads=2, max_seq_len=4096)
    gqa_model = Llama2Model(gqa_cfg)
    gqa_logits = gqa_model(tokens)
    gqa_kv_bytes = gqa_model.layers[0].attn.kv_cache_bytes_per_token()
    print(f"n_heads={gqa_cfg.n_heads}, n_kv_heads={gqa_cfg.n_kv_heads} "
          f"(each KV head shared by {gqa_cfg.n_heads // gqa_cfg.n_kv_heads} query heads)")
    print(f"Output logits shape: {tuple(gqa_logits.shape)}")
    print(f"Total parameters:    {gqa_model.num_params():,}")
    print(f"KV cache bytes/token/layer (fp16): {gqa_kv_bytes}")

    reduction = mha_kv_bytes / gqa_kv_bytes
    print()
    print(f"KV-cache memory reduction from GQA vs. MHA at matched n_heads: {reduction:.1f}x")
    print("(Real Llama-2-70B: n_heads=64, n_kv_heads=8 -> 8x KV-cache reduction vs. full MHA.)")
