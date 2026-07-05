"""
GPT-2 (Radford, Wu, Child, Luan, Amodei, Sutskever -- OpenAI, 2019)
"Language Models are Unsupervised Multitask Learners"

Self-contained PyTorch re-implementation of the GPT-2 decoder-only
transformer, written for study / interview-prep purposes. No external
project modules are imported -- everything needed is defined in this file.

Architectural deltas vs GPT-1 implemented and commented on below:
  1. PRE-NORM transformer blocks. LayerNorm is applied to the INPUT of
     each sub-block (attention, MLP), not to the sub-block's output.
     GPT-1 was post-norm (LN after the residual add). See `Block`.
  2. An EXTRA final LayerNorm (`ln_f`) after the last transformer block
     and before the LM head. See `GPT2.forward`.
  3. Residual-stream output projections -- the attention output
     projection (`attn.c_proj`) and the MLP output projection
     (`mlp.c_proj`) -- are initialized with an extra 1/sqrt(2*n_layer)
     scale on top of the usual std=0.02, to control variance growth of
     the residual stream through depth. See `GPT2._init_weights`.
  4. GELU activation in a 4x-expansion MLP (unchanged from GPT-1).
  5. Learned absolute positional embeddings (unchanged in kind from
     GPT-1; only the size changed: context length 1024 vs 512).
  6. Weight tying between the token embedding (`wte`) and the LM head.

This file implements the MODEL only -- no byte-level BPE tokenizer, no
training loop, no data pipeline. Config presets are provided for all
four sizes GPT-2 was released in (117M / 345M / 762M / 1.5B), but only
a tiny illustrative config is actually instantiated with real weights
in `__main__` (building a real 1.5B-parameter model in fp32 would need
~6GB just for parameters, which is unnecessary for this demo). Parameter
counts for the four real presets are computed analytically by
`count_params_analytic`, and that analytic formula is verified against
an actually-constructed module's `.parameters()` count for the tiny
config before being trusted for the large ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPT2Config:
    vocab_size: int = 50257     # byte-level BPE vocab (50000 merges + 256 byte tokens + 1 special)
    block_size: int = 1024      # context length ("n_ctx" in the paper)
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768          # "n_embd" in the paper / GPT-2 codebase
    dropout: float = 0.1
    bias: bool = True           # GPT-2 uses biases in every Linear and LayerNorm

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head


# The four sizes GPT-2 was actually released in. Names/labels follow the
# paper's own terminology ("117M", "345M", "762M", "1.5B"); note that the
# *exact* parameter count computed from these configs does not match the
# paper's labels precisely -- see the writeup (002_GPT2.md, Section 2) and
# the printout at the bottom of this file for the discrepancy (the labels
# were rounded/approximate at publication time).
GPT2_PRESETS: dict[str, GPT2Config] = {
    "gpt2-small (117M)":  GPT2Config(n_layer=12, n_head=12, d_model=768),
    "gpt2-medium (345M)": GPT2Config(n_layer=24, n_head=16, d_model=1024),
    "gpt2-large (762M)":  GPT2Config(n_layer=36, n_head=20, d_model=1280),
    "gpt2-xl (1.5B)":     GPT2Config(n_layer=48, n_head=25, d_model=1600),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    QKV projections are fused into a single Linear (`c_attn`), mirroring
    GPT-2's use of a single Conv1D for this -- functionally identical to
    three separate Linears, just one fused matmul. Output is recombined
    across heads and passed through `c_proj`.

    `c_proj` is one of the two per-block Linears (the other being
    `MLP.c_proj`) that write directly back into the residual stream, and
    is therefore one of the two layers GPT2._init_weights scales by
    1/sqrt(2*n_layer) at initialization.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True  # marker read by GPT2._init_weights

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask, precomputed once and cached as a (non-persistent)
        # buffer -- not saved in state_dict, just moved with .to(device).
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        qkv = self.c_attn(x)                          # [B, T, 3*C]
        q, k, v = qkv.split(self.d_model, dim=2)       # each [B, T, C]

        # [B, T, C] -> [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot-product attention, computed explicitly for clarity.
        # (A production implementation would use
        #  F.scaled_dot_product_attention(q, k, v, is_causal=True) to get
        #  a fused / flash-attention kernel instead of materializing the
        #  full [B, H, T, T] attention matrix.)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))   # [B, H, T, T]
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                     # [B, H, T, d_head]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # recombine heads -> [B, T, C]

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network: d_model -> 4*d_model -> d_model, GELU."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.c_proj.RESIDUAL_SCALE_INIT = True  # second residual-stream writer, see above
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Pre-norm transformer block -- GPT-2's central architectural delta vs GPT-1:

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    GPT-1 was POST-norm, i.e. equivalent to:

        x = ln_1(x + attn(x))
        x = ln_2(x + mlp(x))

    In pre-norm, the residual path (the running sum `x`) is never itself
    passed through a LayerNorm -- normalization only happens on the branch
    input feeding each sub-block. This keeps the residual stream's scale
    monotonically accumulating rather than being repeatedly renormalized,
    which in practice gives cleaner gradient flow through very deep stacks
    (the paper's analogy is to pre-activation ResNets).
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 decoder-only causal transformer LM."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)   # token embedding
        self.wpe = nn.Embedding(config.block_size, config.d_model)   # learned positional embedding
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.d_model)   # GPT-2's extra final LayerNorm, pre-LM-head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: the output/unembedding matrix IS the input token
        # embedding matrix (transposed use, same underlying Parameter).
        # This is a real GPT-2 (and original Transformer / GPT-1) detail,
        # not an optimization added later -- it roughly halves the
        # embedding-related parameter count and ties input/output token
        # representations.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 initialization scheme:
          - Linear and Embedding weights ~ N(0, 0.02^2).
          - Linear biases = 0.
          - LayerNorm left at nn.LayerNorm's default (weight=1, bias=0),
            which already matches the paper.
          - The two per-block Linears that write directly into the
            residual stream -- attn.c_proj and mlp.c_proj -- get an
            EXTRA 1/sqrt(2*n_layer) multiplicative scale on top of 0.02.

        Why the extra residual scale: every block adds two independent
        contributions to the residual stream (one from attention, one
        from the MLP), so after L blocks the stream has accumulated 2*L
        roughly-independent additive terms. If each term had the same
        fixed variance, the residual stream's variance would grow
        proportionally to depth (2*L * sigma^2), which pushes activations
        into a regime where deep networks become harder to train (LN
        keeps the *input* to each sub-block normalized, but the residual
        stream itself is unnormalized in pre-norm and would otherwise grow
        unboundedly with depth). Shrinking each contribution's projection
        by 1/sqrt(2*n_layer) keeps the variance ADDED per block roughly
        constant in aggregate, so the residual stream's growth stays
        controlled regardless of how many layers are stacked.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE_INIT", False):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds this model's block_size {self.config.block_size} "
            f"(GPT-2's learned positional embedding table has no entries beyond block_size)"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # [T]
        tok_emb = self.wte(idx)   # [B, T, C]
        pos_emb = self.wpe(pos)   # [T, C], broadcasts over batch
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Naive autoregressive sampling loop.

        This recomputes the FULL forward pass over the growing sequence at
        every step -- O(T) work per generated token, O(T^2) total. A real
        serving stack caches each layer's K/V projections for previously
        seen positions (a "KV cache") so each new token costs O(1)
        additional attention work; see 002_GPT2.md Section 9. That
        optimization is an inference-serving detail, not part of the
        original GPT-2 paper, so it is deliberately omitted here to keep
        this file's forward pass the single source of truth.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self, exclude_pos_emb: bool = False) -> int:
        """
        Total parameter count.

        Because `lm_head.weight` IS `wte.weight` (same Parameter object,
        tied at construction), `nn.Module.parameters()` de-duplicates it
        automatically -- PyTorch's parameter/module iteration tracks
        already-visited tensors by identity, so a tied weight is yielded
        exactly once. This matches how GPT-2's officially reported
        parameter counts treat tied embeddings (counted once, not twice).
        """
        n = sum(p.numel() for p in self.parameters())
        if exclude_pos_emb:
            n -= self.wpe.weight.numel()
        return n


# ---------------------------------------------------------------------------
# Analytic parameter counting (no module construction needed)
# ---------------------------------------------------------------------------


def count_params_analytic(config: GPT2Config) -> int:
    """
    Closed-form parameter count for a GPT2Config, derived directly from
    the module definitions above (used so we can report counts for the
    four full-size presets, including the 1.5B one, without paying to
    allocate ~6GB of fp32 tensors just to call .numel() on them).

    Verified in __main__ against an actually-constructed tiny model's
    real .parameters() count before being trusted for the large presets.
    """
    V, C, L, ctx = config.vocab_size, config.d_model, config.n_layer, config.block_size
    b = 1 if config.bias else 0

    def linear(in_f: int, out_f: int) -> int:
        return in_f * out_f + b * out_f

    per_layernorm = 2 * C  # weight [C] + bias [C]

    per_block = (
        2 * per_layernorm            # ln_1, ln_2
        + linear(C, 3 * C)            # attn.c_attn  (fused QKV)
        + linear(C, C)                 # attn.c_proj
        + linear(C, 4 * C)             # mlp.c_fc
        + linear(4 * C, C)             # mlp.c_proj
    )

    total = L * per_block
    total += V * C          # wte
    total += ctx * C        # wpe
    total += per_layernorm  # ln_f
    # lm_head is weight-tied to wte -> contributes 0 additional parameters
    return total


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Tiny illustrative model (NOT a real GPT-2 size -- just enough to
    #     exercise every code path cheaply: attention, MLP, pre-norm
    #     residuals, weight tying, and the loss computation). ---
    tiny_config = GPT2Config(
        vocab_size=1000,
        block_size=32,
        n_layer=2,
        n_head=4,
        d_model=64,
        dropout=0.1,
    )
    model = GPT2(tiny_config)

    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))

    logits, loss = model(idx, targets)

    print("=== Tiny illustrative GPT-2 (config below is NOT a real preset) ===")
    print(tiny_config)
    print(f"input idx shape:  {tuple(idx.shape)}")
    print(f"logits shape:     {tuple(logits.shape)}   (expected: ({B}, {T}, {tiny_config.vocab_size}))")
    print(f"loss:             {loss.item():.4f}")

    generated = model.generate(idx[:, :4], max_new_tokens=6, top_k=50)
    print(f"generate() output shape: {tuple(generated.shape)}   (expected: ({B}, {4 + 6}))")

    real_count = model.num_params()
    analytic_count = count_params_analytic(tiny_config)
    print(f"\nparam count via model.parameters(): {real_count:,}")
    print(f"param count via analytic formula:   {analytic_count:,}")
    assert real_count == analytic_count, "analytic formula does not match the real module's parameter count!"
    print("(analytic formula verified against the real module -- trusting it for the large presets below)\n")

    # --- Real GPT-2 sizes: parameter counts only, computed analytically so
    #     we never need to materialize a 1.5B-parameter model's weights. ---
    print("=== GPT-2 released sizes (exact counts from this file's own module definitions) ===")
    for name, cfg in GPT2_PRESETS.items():
        n = count_params_analytic(cfg)
        n_notied_extra = 0  # tying already reflected (no separate lm_head term above)
        print(
            f"{name:22s} n_layer={cfg.n_layer:2d}  d_model={cfg.d_model:5d}  "
            f"n_head={cfg.n_head:3d}  ctx={cfg.block_size:5d}  "
            f"-> {n:,} params  (~{n / 1e6:.1f}M)"
        )
    print(
        "\nNote: these exact counts run slightly ABOVE the paper's own rounded labels "
        "(e.g. the '117M' config computes to ~124.4M here) -- see 002_GPT2.md Section 2 "
        "for the reconciliation; this is a known, widely-noted discrepancy and not a bug "
        "in this file."
    )
