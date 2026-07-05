"""
002_Claude2.py

Model: Claude 2 (Anthropic, July 2023; Claude 2.1, November 2023)
What this file demonstrates: a KV-cache memory calculator and a small live
PyTorch demonstration of KV-cache growth, motivated by Claude 2's headline
capability -- a 100,000-token context window (200,000 with Claude 2.1), which
was unusually large for its era.

IMPORTANT: Anthropic has never disclosed Claude 2's parameter count, layer
count, hidden size, number of attention heads, whether it uses multi-head,
multi-query, or grouped-query attention, or its positional encoding scheme.
Nothing below reconstructs Claude 2's real architecture. Instead, this file:

  1. Implements the standard KV-cache memory formula that applies to *any*
     decoder-only Transformer, parameterized by architecture knobs that are
     labeled explicitly as illustrative/plausible assumptions, not confirmed
     Claude 2 facts.
  2. Sweeps those knobs (context length, attention-head configuration,
     precision) to make concrete, in real gigabytes, why a 100K context
     window is a nontrivial serving-infrastructure commitment even though
     KV-cache memory scales *linearly* (not quadratically) in sequence
     length -- a distinction from attention *compute*, which does scale
     quadratically.
  3. Runs a small real PyTorch decoder and measures its actual KV-cache
     tensor sizes at increasing context lengths, so the arithmetic isn't
     just a spreadsheet -- it's backed by real tensors with real byte sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Part 1: pure-arithmetic KV-cache calculator (no torch needed for this part
# conceptually, but we keep it in the same file per the "self-contained"
# requirement and because Part 3 cross-checks it against real tensors).
# ---------------------------------------------------------------------------

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


@dataclass
class ModelShape:
    """Illustrative architecture assumptions ONLY -- none of these numbers are
    confirmed for Claude 2. They are round, plausible figures in the range
    frontier dense decoder-only models of the 2023 era occupied, used purely
    to make the KV-cache scaling argument concrete in real units."""
    name: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int          # == num_query_heads for plain MHA; smaller for GQA/MQA
    head_dim: int

    @property
    def hidden_size(self) -> int:
        return self.num_query_heads * self.head_dim


def kv_cache_bytes(shape: ModelShape, seq_len: int, batch_size: int = 1,
                    dtype: str = "fp16") -> int:
    """KV-cache memory, in bytes, for one forward/decode session:

        2 (K and V) * num_layers * seq_len * num_kv_heads * head_dim
            * bytes_per_element * batch_size

    Linear in seq_len, batch_size, and num_layers; the num_kv_heads term is
    where multi-query (num_kv_heads=1) or grouped-query attention (num_kv_heads
    << num_query_heads) buys the largest reduction relative to plain
    multi-head attention (num_kv_heads == num_query_heads).
    """
    bpe = BYTES_PER_DTYPE[dtype]
    return 2 * shape.num_layers * seq_len * shape.num_kv_heads * shape.head_dim * bpe * batch_size


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def kv_cache_report(shapes: List[ModelShape], seq_lens: List[int],
                     batch_size: int = 1, dtype: str = "fp16") -> None:
    print(f"\nKV-cache memory (batch_size={batch_size}, dtype={dtype})")
    header = f"{'model shape':<22}" + "".join(f"{f'{s:,} tok':>16}" for s in seq_lens)
    print(header)
    print("-" * len(header))
    for shape in shapes:
        row = f"{shape.name:<22}"
        for seq_len in seq_lens:
            nbytes = kv_cache_bytes(shape, seq_len, batch_size, dtype)
            row += f"{human_bytes(nbytes):>16}"
        print(row)


# ---------------------------------------------------------------------------
# Part 2: a tiny real decoder-only Transformer with an explicit, growable
# KV-cache, so we can measure real tensor byte sizes rather than only
# computing them analytically.
# ---------------------------------------------------------------------------

class KVCacheAttention(nn.Module):
    """Causal self-attention with an explicit KV-cache, supporting a
    num_kv_heads that can be smaller than num_query_heads (grouped-query /
    multi-query attention), exactly the lever discussed in the .md file's
    Section 9."""

    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(num_query_heads * self.head_dim, d_model)

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_query_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None and kv_cache.get("k") is not None:
            past_len = kv_cache["k"].shape[2]
            k = torch.cat([kv_cache["k"], k_new], dim=2)
            v = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k, v = k_new, v_new
        if kv_cache is not None:
            kv_cache["k"], kv_cache["v"] = k, v

        # Expand KV heads to match query heads for grouped-query attention.
        k_exp = k.repeat_interleave(self.group_size, dim=1)
        v_exp = v.repeat_interleave(self.group_size, dim=1)

        att = (q @ k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        Tk = k_exp.shape[2]
        # Absolute-position causal mask: new query token at absolute position
        # (past_len + i) may attend to any key at absolute position <= that,
        # correct regardless of how many tokens are cached vs. newly added.
        q_pos = torch.arange(past_len, past_len + T, device=x.device).unsqueeze(1)  # (T, 1)
        k_pos = torch.arange(0, Tk, device=x.device).unsqueeze(0)                   # (1, Tk)
        causal_mask = (k_pos <= q_pos)
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v_exp
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_query_heads * self.head_dim)
        return self.out_proj(out), kv_cache

    def cache_bytes(self, kv_cache: dict) -> int:
        if kv_cache is None or kv_cache.get("k") is None:
            return 0
        return kv_cache["k"].element_size() * kv_cache["k"].nelement() + \
               kv_cache["v"].element_size() * kv_cache["v"].nelement()


def measure_real_kv_cache_growth(d_model: int = 512, num_query_heads: int = 8,
                                  num_kv_heads: int = 2, num_layers: int = 4,
                                  seq_lens: List[int] = (1000, 10000, 100000)) -> None:
    """Builds a small real multi-layer attention stack, feeds it growing
    sequences (chunked, to avoid allocating a 100k x 100k dense attention
    matrix on a CPU demo -- KV-cache growth doesn't require materializing that
    at once, but naive dense attention *compute* does, which is exactly the
    O(n^2) point made in the .md file), and reports the real, measured
    KV-cache tensor size at each checkpoint length using PyTorch's own
    element_size()/nelement() accounting -- i.e., this cross-checks the
    analytic formula in Part 1 against actual tensors, not just arithmetic.
    """
    print(f"\nReal PyTorch KV-cache measurement "
          f"(d_model={d_model}, query_heads={num_query_heads}, kv_heads={num_kv_heads}, layers={num_layers})")
    layers = [KVCacheAttention(d_model, num_query_heads, num_kv_heads) for _ in range(num_layers)]
    caches = [dict() for _ in layers]

    chunk = 500  # feed the sequence in manageable chunks to keep this a CPU-friendly demo
    prev_len = 0
    for target_len in seq_lens:
        x = torch.randn(1, chunk, d_model)
        while prev_len < target_len:
            step = min(chunk, target_len - prev_len)
            x_step = x[:, :step, :]
            for i, layer in enumerate(layers):
                _, caches[i] = layer(x_step, caches[i])
            prev_len += step
        total_bytes = sum(layer.cache_bytes(cache) for layer, cache in zip(layers, caches))
        print(f"  seq_len={prev_len:>7,}  measured total KV-cache across "
              f"{num_layers} layers = {human_bytes(total_bytes)}")


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("Claude 2 (2023): 100K-token context window -- KV-cache memory analysis")
    print("Anthropic has NOT disclosed Claude 2's real architecture. The model")
    print("shapes below are illustrative, round, plausible 2023-era figures used")
    print("only to make the SCALING ARGUMENT concrete in real gigabytes.")
    print("=" * 78)

    illustrative_shapes = [
        ModelShape("MHA (heads=32, no GQA)", num_layers=48, num_query_heads=32, num_kv_heads=32, head_dim=128),
        ModelShape("GQA (32q / 8kv heads)", num_layers=48, num_query_heads=32, num_kv_heads=8, head_dim=128),
        ModelShape("MQA (32q / 1kv head)", num_layers=48, num_query_heads=32, num_kv_heads=1, head_dim=128),
    ]
    context_lengths = [9_000, 100_000, 200_000]

    kv_cache_report(illustrative_shapes, context_lengths, batch_size=1, dtype="fp16")

    print(
        "\nReading the table: going from Claude 1's 9,000-token window to Claude 2's"
        "\n100,000-token window is an ~11x increase in KV-cache memory per request --"
        "\nlinear in sequence length, as the formula predicts -- but attention COMPUTE"
        "\n(not modeled in this byte table) scales with seq_len^2, an ~123x increase,"
        "\nwhich is the sharper reason long-context serving needs algorithmic"
        "\nmitigation (efficient attention kernels, sparsity) rather than only more memory."
    )

    print(
        "\nReading across rows at a fixed context length: moving from plain"
        "\nmulti-head attention to grouped-query (4x fewer KV heads) or multi-query"
        "\n(32x fewer KV heads) attention shrinks the SAME context length's KV-cache"
        "\nproportionally -- this is the single largest undisclosed lever that would"
        "\ndetermine whether serving Claude 2's 100K window is memory-cheap or"
        "\nmemory-expensive, and Anthropic has not stated which regime it uses."
    )

    print("\n" + "=" * 78)
    print("Cross-checking the arithmetic against real PyTorch tensors")
    print("=" * 78)
    measure_real_kv_cache_growth(
        d_model=512, num_query_heads=8, num_kv_heads=2, num_layers=4,
        seq_lens=[1_000, 5_000, 10_000],
    )
    analytic = ModelShape("demo shape", num_layers=4, num_query_heads=8, num_kv_heads=2, head_dim=64)
    print("\nAnalytic formula at the same shape, for comparison:")
    for seq_len in (1_000, 5_000, 10_000):
        print(f"  seq_len={seq_len:>7,}  analytic KV-cache = "
              f"{human_bytes(kv_cache_bytes(analytic, seq_len, batch_size=1, dtype='fp32'))}")

    print(
        "\n(Real measurement uses fp32 by default since that's PyTorch's default"
        "\n dtype on CPU without explicit casting -- the two should match closely,"
        "\n confirming the closed-form formula against actual allocated tensors.)"
    )
