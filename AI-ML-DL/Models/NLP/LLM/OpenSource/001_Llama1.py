"""
LLaMA 1 (Touvron et al., 2023) -- "LLaMA: Open and Efficient Foundation Language Models"

Demonstrates the three architectural deviations from the vanilla post-LN transformer
that LLaMA canonicalized into the de facto "Llama-style block" adopted by nearly every
subsequent open decoder-only LLM:

    1. RMSNorm (Zhang & Sennrich, 2019) applied pre-normalization (input to each
       sub-layer is normalized, not its output), with no mean-centering and no bias.
    2. Rotary Position Embeddings -- RoPE (Su et al., 2021) -- injected directly into
       the attention Q/K dot product rather than added to the token embedding, giving
       relative-position-dependent attention scores with no learned position table.
    3. SwiGLU feed-forward network (Shazeer, 2020) replacing the ReLU-MLP, using three
       projection matrices (gate, up, down) with a reduced hidden dimension so that
       total FFN parameter count stays comparable to a ReLU-MLP of hidden size 4*dim.

Attention here is standard multi-head self-attention: n_kv_heads == n_heads. LLaMA 1
predates GQA entirely (introduced only for the 70B in Llama 2, see 002_Llama2.py).

This file is self-contained and illustrative -- dimensions are kept small so the
forward pass and parameter counts are easy to inspect on a CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    vocab_size: int = 32_000  # LLaMA 1's actual SentencePiece BPE vocab size
    max_seq_len: int = 2048   # LLaMA 1's actual fixed training context length
    multiple_of: int = 256    # SwiGLU hidden-dim rounding granularity, per the paper
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0  # base frequency used by LLaMA 1 / Llama 2


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = (x / RMS(x)) * gamma,  RMS(x) = sqrt(mean(x^2) + eps)

    No mean subtraction (unlike LayerNorm) and no bias term -- only a learned
    per-channel gain. Cheaper than LayerNorm (one fewer reduction) and empirically
    just as effective for pre-normalizing transformer residual streams.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for the reduction to avoid fp16/bf16 precision issues,
        # then cast back -- standard practice in production RMSNorm implementations.
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the complex rotation factors e^{i * m * theta_k} for every position m
    and every frequency index k, where theta_k = theta^(-2k/head_dim).

    Returns a complex tensor of shape (max_seq_len, head_dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(angles), angles)  # unit-magnitude complex rotations


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to a query or key tensor.

    x: (batch, seq_len, n_heads, head_dim)
    rope_freqs: (seq_len, head_dim // 2) complex rotation factors

    We view consecutive pairs of the head dimension as the real/imaginary parts of a
    complex number, multiply by the precomputed unit rotation e^{i*m*theta_k}, and
    convert back. This rotates each 2D subspace of the head vector by an angle
    proportional to absolute position m, so that for any two positions m, n the dot
    product <RoPE(q, m), RoPE(k, n)> depends only on (m - n) -- relative position is
    injected directly into the attention score, not into the token representation.
    """
    *prefix, seq_len, n_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*prefix, seq_len, n_heads, head_dim // 2, 2))
    freqs = rope_freqs[:seq_len].view(1, seq_len, 1, head_dim // 2)
    x_rotated = x_complex * freqs
    x_out = torch.view_as_real(x_rotated).reshape(*prefix, seq_len, n_heads, head_dim)
    return x_out.type_as(x)


class Attention(nn.Module):
    """Standard causal multi-head self-attention -- n_kv_heads == n_heads (no GQA/MQA)."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        q = q.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal, memory-efficient scaled-dot-product attention. In the original
        # paper this is the xformers memory-efficient-attention kernel (Rabe & Staats,
        # 2021); torch's SDPA with is_causal=True is the modern equivalent.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """
    FFN(x) = (Swish(x @ W1) * (x @ W3)) @ W2,  Swish(z) = z * sigmoid(z)

    Three matrices instead of a ReLU-MLP's two, so the hidden dimension is scaled to
    ~(2/3)*4*dim (rounded up to a multiple of `multiple_of`) to keep total FFN
    parameter count roughly comparable to a ReLU-MLP FFN with hidden size 4*dim.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * cfg.dim) / 3)
        if cfg.ffn_dim_multiplier is not None:
            hidden_dim = int(cfg.ffn_dim_multiplier * hidden_dim)
        hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)

        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # gate projection
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)  # up projection
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x = x + Attn(RMSNorm(x)); x = x + FFN(RMSNorm(x))."""

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    """Minimal LLaMA-1-style decoder-only transformer: embed -> N blocks -> RMSNorm -> LM head."""

    def __init__(self, cfg: LlamaConfig):
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

    # Small illustrative config -- not a real released LLaMA size, just enough
    # dimensionality to exercise every mechanism above.
    cfg = LlamaConfig(dim=512, n_layers=4, n_heads=8, vocab_size=32_000, max_seq_len=2048)
    model = LlamaModel(cfg)

    batch_size, seq_len = 2, 128
    tokens = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits = model(tokens)

    print(f"Config: dim={cfg.dim}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, "
          f"head_dim={cfg.dim // cfg.n_heads}, vocab_size={cfg.vocab_size}")
    print(f"Input tokens shape:  {tuple(tokens.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Total parameters:    {model.num_params():,}")

    # Sanity check: RoPE makes attention scores a function of relative position only.
    # Shifting both q and k positions by the same offset should not change the score.
    head_dim = cfg.dim // cfg.n_heads
    rope_freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
    q = torch.randn(1, 10, 1, head_dim)
    k = torch.randn(1, 10, 1, head_dim)
    q_rot = apply_rope(q, rope_freqs)
    k_rot = apply_rope(k, rope_freqs)
    score_direct = (q_rot[0, 3, 0] * k_rot[0, 7, 0]).sum()  # positions (3, 7), delta=4

    offset = 50
    q_shifted = apply_rope(q, rope_freqs[offset:])
    k_shifted = apply_rope(k, rope_freqs[offset:])
    score_shifted = (q_shifted[0, 3, 0] * k_shifted[0, 7, 0]).sum()  # positions (53, 57), delta=4

    print(f"RoPE relative-position check -- score at (3,7): {score_direct.item():.6f}, "
          f"score at (53,57): {score_shifted.item():.6f} (should match: same relative offset)")
