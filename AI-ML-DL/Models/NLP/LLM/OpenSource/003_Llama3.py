"""
Llama 3 / 3.1 (Dubey et al., 2024) -- "The Llama 3 Herd of Models"

The transformer block is architecturally identical to Llama 2's GQA block (RMSNorm +
RoPE + SwiGLU + grouped-query attention -- see 002_Llama2.py). Llama 3's changes are:

    1. GQA is now UNIVERSAL: every released size (8B, 70B, 405B) uses n_kv_heads = 8,
       not just the largest size as in Llama 2 (which used GQA only for 70B).
    2. Vocabulary expands 4x, from Llama 2's 32,000 tokens to 128,256 tokens. This
       file demonstrates the concrete parameter and compute consequence of that
       change: the embedding + LM head matrices scale directly with vocab_size, so
       a 4x vocab increase materially grows those two matrices' parameter share,
       especially at smaller model width -- while simultaneously *reducing* the
       number of tokens needed to represent any given piece of text (better
       compression), which is the actual point of the larger vocabulary.
    3. RoPE base frequency is raised to 500,000 (from 10,000) to support the 128K
       context extension in Llama 3.1 -- stretching low-frequency rotation
       wavelengths so they remain well-behaved at much larger relative distances.

This file reuses the GQA attention mechanism from 002_Llama2.py (reimplemented here
to keep this file self-contained) and focuses the demonstration on the vocab-size /
RoPE-base comparison against a Llama-2-style config.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Llama3Config:
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2       # universal GQA -- applied even at the smallest size
    vocab_size: int = 128_256  # Llama 3's actual tokenizer vocab size (vs. Llama 2's 32,000)
    max_seq_len: int = 8192    # illustrative; real Llama 3.1 extends to 128K
    multiple_of: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 500_000.0  # raised from 10,000 (Llama 1/2) to support long context


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
    *prefix, seq_len, n_heads, head_dim = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(*prefix, seq_len, n_heads, head_dim // 2, 2))
    freqs = rope_freqs[:seq_len].view(1, seq_len, 1, head_dim // 2)
    x_out = torch.view_as_real(x_complex * freqs).reshape(*prefix, seq_len, n_heads, head_dim)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """Identical mechanism to Llama 2's GQA (002_Llama2.py) -- applied here at every
    model size by convention, which is Llama 3's actual design decision, not a change
    to the mechanism itself."""

    def __init__(self, cfg: Llama3Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads

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

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        hidden_dim = int(2 * (4 * cfg.dim) / 3)
        hidden_dim = cfg.multiple_of * ((hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w1 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Llama3Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama3Model(nn.Module):
    def __init__(self, cfg: Llama3Config):
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

    def embedding_and_head_params(self) -> int:
        return self.tok_embeddings.weight.numel() + self.output.weight.numel()


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len = 2, 128

    print("=" * 80)
    print("Llama-3-style config: universal GQA (n_kv_heads=2 at every size, incl. smallest)")
    print("128,256-token vocab (vs. Llama 2's 32,000) -- 4x larger tokenizer")
    print("=" * 80)
    llama3_cfg = Llama3Config(dim=512, n_layers=4, n_heads=8, n_kv_heads=2, vocab_size=128_256)
    llama3_model = Llama3Model(llama3_cfg)
    tokens = torch.randint(0, llama3_cfg.vocab_size, (batch_size, seq_len))
    logits = llama3_model(tokens)

    total_params = llama3_model.num_params()
    embed_head_params = llama3_model.embedding_and_head_params()
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Total parameters:              {total_params:,}")
    print(f"Embedding + LM head parameters: {embed_head_params:,} "
          f"({100 * embed_head_params / total_params:.1f}% of total)")

    print()
    print("=" * 80)
    print("Same model width, Llama-2-style 32,000-token vocab, for comparison")
    print("=" * 80)
    llama2_vocab_cfg = Llama3Config(dim=512, n_layers=4, n_heads=8, n_kv_heads=2, vocab_size=32_000)
    llama2_vocab_model = Llama3Model(llama2_vocab_cfg)
    total_params_32k = llama2_vocab_model.num_params()
    embed_head_params_32k = llama2_vocab_model.embedding_and_head_params()
    print(f"Total parameters:              {total_params_32k:,}")
    print(f"Embedding + LM head parameters: {embed_head_params_32k:,} "
          f"({100 * embed_head_params_32k / total_params_32k:.1f}% of total)")

    print()
    extra_params = embed_head_params - embed_head_params_32k
    print(f"Extra parameters purely from the 4x larger vocabulary: {extra_params:,}")
    print("This parameter cost buys a shorter token sequence for any given text -- fewer")
    print("prefill positions and fewer autoregressive decode steps for the same content,")
    print("which is the actual point of Llama 3's tokenizer expansion.")

    # RoPE base-frequency comparison: higher base -> slower-rotating low frequency
    # components -> better-behaved dot products at very large relative offsets.
    head_dim = llama3_cfg.dim // llama3_cfg.n_heads
    llama2_style_freqs = precompute_rope_freqs(head_dim, 16, theta=10_000.0)
    llama3_style_freqs = precompute_rope_freqs(head_dim, 16, theta=500_000.0)
    print()
    print(f"RoPE lowest-frequency rotation angle at position 15, theta=10,000:  "
          f"{torch.angle(llama2_style_freqs[15, -1]).item():.6f} rad")
    print(f"RoPE lowest-frequency rotation angle at position 15, theta=500,000: "
          f"{torch.angle(llama3_style_freqs[15, -1]).item():.6f} rad "
          f"(slower rotation -> supports much longer context before aliasing)")
