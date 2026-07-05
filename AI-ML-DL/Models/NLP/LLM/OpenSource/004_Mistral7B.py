"""
Mistral 7B (Jiang et al., 2023)

Demonstrates the two mechanisms that are architecturally distinctive about Mistral 7B:

    1. Sliding window attention (SWA): each position i attends only to positions in
       [i - W + 1, i] rather than the full causal history [0, i]. Implemented here as
       an explicit banded boolean mask so the restriction is verifiable by construction
       (not just "trust the causal-mask helper").
    2. The rolling buffer KV cache: a fixed-size, circularly-indexed cache of exactly W
       slots per sequence, where the K/V for position i is written to slot (i mod W),
       safely overwriting the entry for position (i - W) -- which is provably safe
       because no future position j > i - 1 + W can ever have (j - (i - W)) < W, i.e.
       once a position falls more than W steps behind, it permanently leaves every
       future token's attention window and can never be read again.

The backbone (RMSNorm, RoPE, SwiGLU, GQA with n_kv_heads < n_heads) is the same
Llama-style block used throughout this series -- see 001_Llama1.py / 002_Llama2.py.
Mistral 7B's real released config: dim=4096, n_layers=32, n_heads=32, n_kv_heads=8,
sliding window W=4096, vocab=32,000. This file uses small illustrative dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MistralConfig:
    dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    n_kv_heads: int = 2         # GQA, same mechanism as Llama 2/3
    vocab_size: int = 32_000
    sliding_window: int = 16    # real Mistral 7B: 4096; small here for a legible demo
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


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
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


def sliding_window_causal_mask(seq_len: int, window: int) -> torch.Tensor:
    """
    Boolean attention mask of shape (seq_len, seq_len); True where query position i
    is allowed to attend to key position j.

    Causal:  j <= i
    Windowed: i - j < window   (equivalently j > i - window)

    So the allowed band is j in (i - window, i], a diagonal strip of width `window`
    instead of the full lower triangle a plain causal mask would give.
    """
    i = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    j = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    causal = j <= i
    windowed = (i - j) < window
    return causal & windowed


class SlidingWindowAttention(nn.Module):
    """Full-sequence (prefill-style) GQA attention restricted to a sliding window."""

    def __init__(self, cfg: MistralConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.window = cfg.sliding_window

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
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

        q = q.transpose(1, 2)
        k = self._repeat_kv(k.transpose(1, 2), self.n_rep)
        v = self._repeat_kv(v.transpose(1, 2), self.n_rep)

        mask = sliding_window_causal_mask(seq_len, self.window).to(x.device)  # (seq_len, seq_len) bool
        # SDPA expects True == "allowed to attend" when passed a bool attn_mask.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class RollingBufferKVCache:
    """
    Fixed-size circular KV cache for autoregressive decoding under a sliding window
    of size W. Holds exactly W positions' worth of K/V per (batch, kv_head) regardless
    of how many tokens have actually been generated -- memory does not grow with
    sequence length, unlike a standard unbounded KV cache.

    Slot for absolute position `pos` is `pos % W`. Writing position `pos` overwrites
    whatever was at slot `pos % W`, which necessarily held position `pos - W` (the
    only earlier position that maps to the same slot). This is safe because position
    `pos - W` is, by construction, exactly W steps behind `pos` and therefore already
    outside the attention window of position `pos` and every later position -- it will
    never be attended to again by any future query.
    """

    def __init__(self, batch_size: int, n_kv_heads: int, window: int, head_dim: int, dtype=torch.float32):
        self.window = window
        self.k_cache = torch.zeros(batch_size, n_kv_heads, window, head_dim, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, n_kv_heads, window, head_dim, dtype=dtype)
        self.filled = torch.zeros(window, dtype=torch.bool)  # tracks which slots hold valid data
        self.slot_position = -torch.ones(window, dtype=torch.long)  # absolute position stored in each slot

    def update(self, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: (batch, n_kv_heads, 1, head_dim) for the single new position `pos`."""
        slot = pos % self.window
        self.k_cache[:, :, slot, :] = k[:, :, 0, :]
        self.v_cache[:, :, slot, :] = v[:, :, 0, :]
        self.filled[slot] = True
        self.slot_position[slot] = pos

    def get_window(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (k, v, positions) for all currently valid slots, sorted by absolute
        position, ready to be used as the key/value set for the next query."""
        valid = self.filled.nonzero(as_tuple=True)[0]
        order = torch.argsort(self.slot_position[valid])
        valid = valid[order]
        return self.k_cache[:, :, valid, :], self.v_cache[:, :, valid, :], self.slot_position[valid]

    def memory_bytes(self) -> int:
        return self.k_cache.element_size() * self.k_cache.nelement() * 2  # K + V


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = MistralConfig(dim=512, n_layers=2, n_heads=8, n_kv_heads=2, sliding_window=16)

    print("=" * 80)
    print("Sliding window causal mask")
    print("=" * 80)
    seq_len = 24
    mask = sliding_window_causal_mask(seq_len, cfg.sliding_window)
    print(f"seq_len={seq_len}, window={cfg.sliding_window}")
    print(f"Mask shape: {tuple(mask.shape)}")
    print(f"Row 20 (query position 20) attends to key positions: "
          f"{mask[20].nonzero(as_tuple=True)[0].tolist()}")
    print("(Should be exactly [5..20] -- 16 positions ending at the query itself, "
          "not [0..20] as a plain causal mask would give.)")
    full_causal_edges = seq_len * (seq_len + 1) // 2
    windowed_edges = mask.sum().item()
    print(f"Full causal mask edge count: {full_causal_edges}, sliding-window mask edge count: "
          f"{windowed_edges} ({100 * windowed_edges / full_causal_edges:.0f}% of full causal)")

    print()
    print("=" * 80)
    print("Forward pass through a sliding-window GQA attention layer")
    print("=" * 80)
    attn = SlidingWindowAttention(cfg)
    head_dim = cfg.dim // cfg.n_heads
    rope_freqs = precompute_rope_freqs(head_dim, seq_len, cfg.rope_theta)
    x = torch.randn(2, seq_len, cfg.dim)
    out = attn(x, rope_freqs)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    print()
    print("=" * 80)
    print("Rolling buffer KV cache: memory stays bounded past window size")
    print("=" * 80)
    batch_size = 2
    cache = RollingBufferKVCache(batch_size, cfg.n_kv_heads, cfg.sliding_window, head_dim)
    print(f"Cache memory footprint (fixed, independent of generated length): "
          f"{cache.memory_bytes()} bytes")

    n_generated = 50  # far beyond window=16, to show the cache never grows
    for pos in range(n_generated):
        k = torch.randn(batch_size, cfg.n_kv_heads, 1, head_dim)
        v = torch.randn(batch_size, cfg.n_kv_heads, 1, head_dim)
        cache.update(pos, k, v)

    k_win, v_win, positions = cache.get_window()
    print(f"After generating {n_generated} tokens with window={cfg.sliding_window}:")
    print(f"  Cache still holds only {k_win.shape[2]} positions: {positions.tolist()}")
    print(f"  Cache memory footprint after {n_generated} tokens: {cache.memory_bytes()} bytes "
          f"(unchanged -- a standard unbounded cache would instead hold {n_generated} positions)")

    unbounded_equivalent_bytes = cache.memory_bytes() * n_generated / cfg.sliding_window
    print(f"  An unbounded cache at this point would use ~{unbounded_equivalent_bytes:.0f} bytes: "
          f"{unbounded_equivalent_bytes / cache.memory_bytes():.1f}x more than the rolling buffer.")
