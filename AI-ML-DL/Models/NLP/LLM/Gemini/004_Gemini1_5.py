"""
Gemini 1.5 (Google DeepMind, 2024) -- KV-cache memory calculator.

The headline Gemini 1.5 feature (a 1-million-token production context
window) creates a SERVING problem before any modeling question is even
relevant: naive self-attention is O(L^2) in sequence length L, and the
KV cache needed for autoregressive decoding is O(L) but with a large
constant factor (num_layers * num_kv_heads * head_dim * 2 * bytes_per_elem),
which becomes enormous in absolute terms at L = 1,000,000.

This file computes, for a range of context lengths and a few representative
model configurations, exactly how large the KV cache gets, and shows the
size of the concrete effect of two of the standard mitigations discussed in
004_Gemini1_5.md, Section 9:
  - multi-query / grouped-query attention (fewer KV heads than query heads)
  - lower-precision KV cache storage (fp16/int8 instead of fp32)

None of the exact numbers here are Gemini 1.5's actual configuration
(undisclosed); this is a generic calculator applied to illustrative
configurations, to make the ORDER OF MAGNITUDE of the problem concrete.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int  # == num_query_heads for standard MHA; 1 for MQA; small for GQA
    head_dim: int
    bytes_per_element: int  # 4 = fp32, 2 = fp16/bf16, 1 = int8


def kv_cache_bytes(cfg: ModelConfig, seq_len: int, batch_size: int = 1) -> int:
    """Total KV-cache size in bytes for a given context length.

    Formula: batch * seq_len * num_layers * num_kv_heads * head_dim
             * 2 (K and V)  * bytes_per_element
    """
    return (
        batch_size
        * seq_len
        * cfg.num_layers
        * cfg.num_kv_heads
        * cfg.head_dim
        * 2  # one tensor for K, one for V
        * cfg.bytes_per_element
    )


def attention_score_flops(seq_len: int, num_layers: int, num_heads: int, head_dim: int) -> float:
    """Rough FLOPs for computing the QK^T score matrix across all layers/heads
    at a given sequence length (prefill, full quadratic attention, no causal
    masking discount) -- illustrates the O(L^2) compute scaling, independent
    of the KV-cache memory question."""
    # QK^T: for each layer/head, (L x d) @ (d x L) -> L*L*d multiply-adds, *2 FLOPs each.
    return 2.0 * num_layers * num_heads * (seq_len ** 2) * head_dim


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} EB"


if __name__ == "__main__":
    context_lengths = [4_096, 32_768, 128_000, 1_000_000, 10_000_000]

    configs = [
        ModelConfig("Standard MHA, fp16 (illustrative ~70B-class dense model)",
                    num_layers=80, num_query_heads=64, num_kv_heads=64, head_dim=128,
                    bytes_per_element=2),
        ModelConfig("Multi-Query Attention, fp16 (PaLM/Gemini-style MQA)",
                    num_layers=80, num_query_heads=64, num_kv_heads=1, head_dim=128,
                    bytes_per_element=2),
        ModelConfig("Grouped-Query Attention (8 KV heads), fp16",
                    num_layers=80, num_query_heads=64, num_kv_heads=8, head_dim=128,
                    bytes_per_element=2),
        ModelConfig("Multi-Query Attention, int8 KV cache",
                    num_layers=80, num_query_heads=64, num_kv_heads=1, head_dim=128,
                    bytes_per_element=1),
    ]

    print("=== KV-cache size vs. context length (batch_size=1) ===\n")
    header = f"{'Config':<52}" + "".join(f"{L:>14,}" for L in context_lengths)
    print(header)
    for cfg in configs:
        row = f"{cfg.name:<52}"
        for L in context_lengths:
            row += f"{human_bytes(kv_cache_bytes(cfg, L)):>14}"
        print(row)

    print("\n=== Why this matters: standard MHA at 1M tokens vs. accelerator HBM ===")
    mha_1m = kv_cache_bytes(configs[0], 1_000_000)
    mqa_1m = kv_cache_bytes(configs[1], 1_000_000)
    print(f"Standard MHA KV cache @ 1M tokens : {human_bytes(mha_1m)}  "
          f"(a single high-end accelerator typically has ~80-192 GB of HBM)")
    print(f"MQA KV cache @ 1M tokens          : {human_bytes(mqa_1m)}  "
          f"({mha_1m / mqa_1m:.0f}x smaller than standard MHA, same context length)")

    print("\n=== O(L^2) attention score compute (prefill, full quadratic attention) ===")
    for L in context_lengths:
        flops = attention_score_flops(L, num_layers=80, num_heads=64, head_dim=128)
        print(f"L={L:>10,}  QK^T FLOPs (all layers/heads) ~= {flops:.3e}")
    ratio = attention_score_flops(1_000_000, 80, 64, 128) / attention_score_flops(
        4_096, 80, 64, 128
    )
    print(f"\nGoing from 4,096 -> 1,000,000 tokens (a ~244x increase in length) "
          f"increases attention FLOPs by ~{ratio:,.0f}x, confirming the quadratic "
          f"(not linear) scaling.")

    print("\n=== Demonstration with actual tensors (small scale, for shape/behavior) ===")
    # A tiny torch-side sanity check: materializing the full attention score
    # matrix at a toy sequence length, to show WHERE the O(L^2) memory blowup
    # actually lives in a real forward pass (the (L, L) score tensor).
    L_small, d = 2048, 64
    q = torch.randn(1, 8, L_small, d)
    k = torch.randn(1, 8, L_small, d)
    scores = q @ k.transpose(-2, -1)  # (1, 8, L_small, L_small) -- the O(L^2) tensor
    print(f"Toy QK^T score tensor shape at L={L_small}: {tuple(scores.shape)}, "
          f"{scores.numel():,} elements, {human_bytes(scores.numel() * 4)} in fp32.")
    print("Scaling L_small by 488x to reach 1,000,000 would scale this tensor's "
          "element count by 488^2 ~= 238,000x -- this is precisely why naive, "
          "fully-materialized attention is intractable at 1M-token context, and "
          "why fused/tiled kernels (never materializing the full score matrix) "
          "are close to a necessary condition for serving at this length.")
