"""
Qwen2.5 (Alibaba, 2024) -- reference implementation of the credibly-published
architecture family: RoPE + RMSNorm + SwiGLU + GQA, in the standard Llama-style
decoder block shape.

Unlike DeepSeek-V2/V3 (Multi-head Latent Attention, fine-grained MoE with
bias-based balancing) or DeepSeek-R1 (GRPO), Qwen2.5's openly-published
architecture does NOT introduce a novel attention or MoE mechanism -- its
documented deltas from the "standard Llama recipe" are:
  1. Bias terms on the Q/K/V projections (Alibaba states this explicitly;
     Llama's projections are bias-free). Implemented below via `qkv_bias=True`.
  2. A large (~151K-token) multilingual/code-heavy BBPE tokenizer -- not
     modeled here since tokenization is orthogonal to the transformer block.
  3. GQA in the larger dense variants, with per-size head/group counts
     disclosed in Alibaba's config tables.
  4. YaRN-style RoPE scaling for long-context variants (128K) -- omitted here
     for simplicity; see the comment in `RotaryEmbedding` for where it slots in.

Everything else in this file (RMSNorm, SwiGLU FFN, GQA attention, causal
masking, pre-norm residual block structure) is the common 2023+ open-model
recipe, not a Qwen-specific claim -- see 009_Qwen2_5.md Section 2 for the
precise confirmed-vs-conventional breakdown.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen25Config:
    vocab_size: int = 32000          # real Qwen2.5 uses ~151,646
    d_model: int = 512               # hidden size (toy scale)
    n_layers: int = 4
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA: shared K/V heads (n_heads must be divisible by n_kv_heads)
    d_ff: int = 1376                 # SwiGLU intermediate size (~2.67x d_model, Qwen/Llama convention)
    max_seq_len: int = 2048
    rope_theta: float = 1000000.0    # Qwen2.5 uses a large RoPE base for long-context variants
    qkv_bias: bool = True            # Qwen-specific: bias terms on Q/K/V (Llama has none)
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = False     # True for the smallest Qwen2.5 sizes (0.5B/1.5B)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """
    Standard RoPE. Real Qwen2.5 long-context variants apply YaRN-style scaling
    (interpolating/extrapolating the frequency spectrum) on top of this base
    implementation to extend from a shorter native pretraining context out to
    128K -- that scaling would be inserted here, adjusting `inv_freq` and/or
    the effective position indices; omitted in this toy version.
    """

    def __init__(self, dim: int, base: float = 1000000.0, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", torch.cat([freqs.cos(), freqs.cos()], dim=-1), persistent=False)
        self.register_buffer("sin", torch.cat([freqs.sin(), freqs.sin()], dim=-1), persistent=False)

    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [batch, n_heads, seq_len, head_dim]; cos/sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention: n_heads query heads share n_kv_heads key/value
    heads (n_heads / n_kv_heads query heads per KV head). KV cache per token
    scales with n_kv_heads, not n_heads -- the standard GQA cache-reduction
    mechanism, contrasted in the .md against DeepSeek's MLA (which compresses
    to a shared LATENT rather than sharing literal per-head K/V tensors).
    """

    def __init__(self, cfg: Qwen25Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # query heads per KV head group

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)  # output proj: no bias (Qwen keeps this bias-free too)

        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_theta, max_seq_len=cfg.max_seq_len)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, kv_cache=None, use_cache: bool = False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.cfg.n_heads, self.head_dim).transpose(1, 2)      # [b, n_heads, t, hd]
        k = self.k_proj(x).view(b, t, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)    # [b, n_kv_heads, t, hd]
        v = self.v_proj(x).view(b, t, self.cfg.n_kv_heads, self.head_dim).transpose(1, 2)    # [b, n_kv_heads, t, hd]

        cos, sin = self.rope(t)
        cos, sin = cos.to(x.device), sin.to(x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # expand KV heads to match query head count (GQA: repeat each KV head n_rep times)
        k_exp = k.repeat_interleave(self.n_rep, dim=1)   # [b, n_heads, t_kv, hd]
        v_exp = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale     # [b, n_heads, t, t_kv]
        t_kv = k_exp.shape[2]
        causal_mask = torch.triu(
            torch.ones(t, t_kv, device=x.device, dtype=torch.bool), diagonal=t_kv - t + 1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_exp)                                    # [b, n_heads, t, hd]

        out = out.transpose(1, 2).reshape(b, t, self.cfg.n_heads * self.head_dim)
        return self.o_proj(out), new_cache

    def kv_cache_elements_per_token(self) -> int:
        return 2 * self.cfg.n_kv_heads * self.head_dim  # K and V, only n_kv_heads worth


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen25Block(nn.Module):
    """Standard pre-norm decoder block: attention and FFN each wrapped in a residual + RMSNorm."""

    def __init__(self, cfg: Qwen25Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, kv_cache=None, use_cache=False):
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class Qwen25Model(nn.Module):
    def __init__(self, cfg: Qwen25Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Qwen25Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight  # Qwen2.5 0.5B/1.5B tie embeddings

    def forward(self, input_ids: torch.Tensor):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = Qwen25Config()
    model = Qwen25Model(cfg)

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))

    print("=== Qwen2.5-style block: RoPE + RMSNorm + SwiGLU + GQA ===")
    print(f"n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads} (GQA group size = {cfg.n_heads // cfg.n_kv_heads}), "
          f"qkv_bias={cfg.qkv_bias}")

    logits = model(input_ids)
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape:    {tuple(logits.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\ntotal parameter count (toy scale): {total_params:,}")

    attn = model.layers[0].attn
    gqa_cache = attn.kv_cache_elements_per_token()
    mha_equivalent_cache = 2 * cfg.n_heads * attn.head_dim
    print(f"\nKV cache per token per layer (elements):")
    print(f"  GQA (n_kv_heads={cfg.n_kv_heads}): {gqa_cache}")
    print(f"  MHA-equivalent (n_heads={cfg.n_heads}):    {mha_equivalent_cache}")
    print(f"  GQA reduction factor: {mha_equivalent_cache / gqa_cache:.1f}x")
    print(
        "  (Contrast with DeepSeek-V2's MLA, which compresses to a shared low-rank LATENT "
        "independent of head/group count entirely -- see 006_DeepSeek_V2.py.)"
    )

    # KV-cache incremental decoding demo
    print("\n=== Incremental decoding with KV cache (single layer, single head-group) ===")
    x = torch.randn(1, 5, cfg.d_model)
    out1, cache1 = attn(x, use_cache=True)
    next_token = torch.randn(1, 1, cfg.d_model)
    out2, cache2 = attn(next_token, kv_cache=cache1, use_cache=True)
    print(f"prefill output shape: {tuple(out1.shape)}, cached K shape: {tuple(cache1[0].shape)}")
    print(f"decode-step output shape: {tuple(out2.shape)}, cached K shape after append: {tuple(cache2[0].shape)}")

    print(
        "\n(Real Qwen2.5 sizes: 0.5B/1.5B/3B/7B/14B/32B/72B dense, vocab ~151,646, "
        "pretrained on ~18T tokens, context up to 128K via YaRN scaling on select variants.)"
    )
