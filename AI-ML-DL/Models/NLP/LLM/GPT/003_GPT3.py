"""
GPT-3 (Brown et al., 2020, "Language Models are Few-Shot Learners") -- illustrative PyTorch implementation.

This file implements a clean, self-contained decoder-only causal transformer whose
distinctive mechanism -- relative to a plain GPT-2 -- is the ALTERNATING DENSE /
LOCALLY-BANDED-SPARSE causal self-attention pattern described in the GPT-3 paper:

    "we use alternating dense and locally banded sparse attention patterns in the
     layers of the transformer, similar to the Sparse Transformer [Child et al., 2019]"

Everything else (pre-norm transformer block, GELU FFN with 4x expansion, learned
absolute position embeddings, weight-tied embedding/unembedding) matches the
GPT-2/GPT-3 architectural lineage.

IMPORTANT SCALE NOTE:
    The real GPT-3 175B config is d_model=12288, n_layers=96, n_heads=96 (128 dim/head),
    context (block_size)=2048, FFN inner dim=49152 (4x). That config is far too large to
    instantiate, let alone forward-pass, in an illustrative script. The __main__ block
    below instead instantiates a tiny toy config to demonstrate the alternating
    dense/sparse-local attention mechanism end-to-end.

This implementation favors clarity and correctness over throughput: the sparse-local
attention below is built by masking a full (n x n) score matrix, exactly like dense
attention, rather than avoiding materializing the full score matrix. A real production
implementation of banded/local attention at long context lengths would instead:
    - block the sequence into chunks of size ~W and only compute QK^T for chunk pairs
      that fall within the band (i.e., never materialize the full n x n matrix at all),
    - or use a fused/blocksparse attention kernel (e.g., Triton block-sparse attention,
      or FlashAttention variants with local/sliding-window masking built into the kernel)
      that skips FLOPs and memory for out-of-band blocks entirely rather than computing
      then masking them.
The masking approach here is deliberately the "textbook" version for pedagogical
clarity: it is correct and easy to verify, but it does not save any FLOPs relative to
dense attention (it still computes the full n x n score matrix and then masks it) --
the compute/memory savings it is meant to illustrate only materialize with a
block-sparse-aware kernel.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class GPT3Config:
    """Configuration for the GPT-3-style model.

    The true GPT-3 175B ("davinci") config, per the paper, is:
        vocab_size  = 50257   (GPT-2-style BPE vocab)
        d_model     = 12288
        n_layers    = 96
        n_heads     = 96      (128 dim/head)
        block_size  = 2048    (context window)
        d_ff        = 49152   (4x d_model)

    That configuration is far too large to instantiate in this illustrative file
    (350GB+ of fp16 weights alone). The __main__ block below uses a tiny toy config
    instead, purely to demonstrate the alternating dense/sparse-local attention
    mechanism with a runnable forward pass.
    """
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    block_size: int = 2048
    d_ff: int = None  # defaults to 4 * d_model
    local_window: int = 128  # W: backward-looking local window size for sparse layers
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


# ----------------------------------------------------------------------------
# Attention variants
# ----------------------------------------------------------------------------

class DenseCausalSelfAttention(nn.Module):
    """Standard full causal multi-head self-attention: each query position i
    attends to every key position j <= i. Cost is O(n^2 * d) in both compute
    (QK^T and the subsequent weighted sum over V) and memory (the materialized
    n x n attention-score matrix) -- this is the same attention used in GPT-1/2,
    and is the "dense" half of GPT-3's alternating pattern.
    """

    def __init__(self, config: GPT3Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute a full lower-triangular causal mask once; reused every forward.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        # (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        causal = self.causal_mask[:T, :T]  # (T, T) bool, True where attention is allowed
        att_scores = att_scores.masked_fill(~causal, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v  # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class LocalBandedSparseCausalSelfAttention(nn.Module):
    """Locally-banded sparse causal multi-head self-attention, as described (at a
    high level) in the GPT-3 paper as similar to the Sparse Transformer (Child et
    al., 2019): each query position i attends only to keys within a backward-looking
    local window of size W, in addition to the causal constraint j <= i. That is, the
    allowed set of keys for query i is:

        { j : max(0, i - W + 1) <= j <= i }

    instead of the full { j : 0 <= j <= i } used by dense attention.

    This reduces the number of (query, key) pairs actually attended to from O(n) per
    query (dense, causal) to O(W) per query, so a full layer's attention cost drops
    from O(n^2 * d) to O(n * W * d). For GPT-3's context of n=2048, a local window W
    that is a small fraction of n yields a substantial reduction in the dominant
    quadratic term for these layers.

    Note: Child et al. 2019's Sparse Transformer combines a *local* component like
    this with a *strided* component (attending to every k-th earlier position, to
    preserve some long-range signal). This class implements only the local/banded
    component for clarity, as called for by the task; a fuller reproduction would
    OR in a strided mask term as well.

    Implementation note on efficiency: as written, this module still builds a full
    (T, T) score matrix and masks out everything outside the band -- i.e., it pays
    the full O(n^2) compute and memory cost of dense attention and then throws most
    of it away. It is correct (the masked-out positions get exactly zero attention
    weight, matching a "true" sparse implementation's output) but not efficient. A
    real implementation would instead tile queries and keys into blocks of size ~W
    and only compute QK^T for the (O(n/W)) block-pairs that fall inside the band,
    e.g. via a block-sparse attention kernel, so that FLOPs and memory scale with
    O(n * W) rather than O(n^2).
    """

    def __init__(self, config: GPT3Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.local_window = config.local_window

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Banded mask = causal AND within local window: allowed(i, j) iff
        # 0 <= i - j < W  (i.e., j <= i and j > i - W).
        idx = torch.arange(config.block_size)
        rel = idx.view(-1, 1) - idx.view(1, -1)  # rel[i, j] = i - j
        band_mask = (rel >= 0) & (rel < config.local_window)  # (block_size, block_size) bool
        self.register_buffer("band_mask", band_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        band = self.band_mask[:T, :T]  # (T, T) bool: True where in-band AND causal
        att_scores = att_scores.masked_fill(~band, float("-inf"))

        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        out = att_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


# ----------------------------------------------------------------------------
# FFN
# ----------------------------------------------------------------------------

class GELUFeedForward(nn.Module):
    """Standard GPT-2/3-style position-wise FFN: d_model -> 4*d_model -> d_model,
    with GELU activation on the hidden layer.
    """

    def __init__(self, config: GPT3Config):
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.fc_out = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(x))))


# ----------------------------------------------------------------------------
# Transformer block (configurable dense vs. sparse-local attention)
# ----------------------------------------------------------------------------

class GPT3Block(nn.Module):
    """A single pre-norm transformer decoder block:
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    where Attn is either DenseCausalSelfAttention or LocalBandedSparseCausalSelfAttention,
    selected at construction time via `use_sparse_local`. This is what lets the full
    model alternate attention patterns layer by layer.
    """

    def __init__(self, config: GPT3Config, use_sparse_local: bool):
        super().__init__()
        self.use_sparse_local = use_sparse_local
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = (
            LocalBandedSparseCausalSelfAttention(config)
            if use_sparse_local
            else DenseCausalSelfAttention(config)
        )
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = GELUFeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------

class GPT3Model(nn.Module):
    """Decoder-only causal transformer LM in the GPT-3 style: learned absolute
    position embeddings, pre-norm blocks alternating dense and locally-banded
    sparse causal self-attention layer-by-layer (even layers dense, odd layers
    sparse-local -- matching the paper's description of an alternating pattern),
    GELU FFN with 4x expansion, weight-tied embedding/unembedding.
    """

    def __init__(self, config: GPT3Config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Alternate: even layer index -> dense, odd layer index -> sparse-local.
        self.blocks = nn.ModuleList([
            GPT3Block(config, use_sparse_local=(layer_idx % 2 == 1))
            for layer_idx in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight-tie the unembedding to the token embedding, as in GPT-2/3.
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def layer_attention_types(self):
        """Returns a list of 'dense' / 'sparse-local' strings, one per layer, in order."""
        return ["sparse-local" if b.use_sparse_local else "dense" for b in self.blocks]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (B, T, d_model)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            # token_emb and lm_head are weight-tied, so this only subtracts the
            # shared embedding matrix once, plus the (untied) positional embedding.
            n -= self.token_emb.weight.numel()
            n -= self.pos_emb.weight.numel()
        return n


# ----------------------------------------------------------------------------
# Illustrative run
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Tiny illustrative config -- NOT the real 175B GPT-3 config (see GPT3Config
    # docstring above: real config is d_model=12288, n_layers=96, n_heads=96,
    # block_size=2048). This toy config is chosen purely to make the forward pass
    # run instantly and to make the dense/sparse-local layer alternation visible
    # (n_layers=4 gives layers [dense, sparse-local, dense, sparse-local]).
    config = GPT3Config(
        vocab_size=1000,
        d_model=64,
        n_layers=4,
        n_heads=4,
        block_size=32,
        local_window=8,
        dropout=0.0,
    )

    model = GPT3Model(config)

    batch_size, seq_len = 2, 32
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(dummy_input)

    print("=== GPT3Model illustrative forward pass ===")
    print(f"config: {config}")
    print(f"input shape:  {tuple(dummy_input.shape)}  (batch_size, seq_len)")
    print(f"output shape: {tuple(logits.shape)}  (batch_size, seq_len, vocab_size)")
    print()

    total_params = model.num_parameters()
    total_params_excl_emb = model.num_parameters(exclude_embeddings=True)
    print(f"total parameters:                {total_params:,}")
    print(f"total parameters (excl. embeds): {total_params_excl_emb:,}")
    print()

    print("per-layer attention pattern (as described in the GPT-3 paper, alternating):")
    for i, kind in enumerate(model.layer_attention_types()):
        print(f"  layer {i}: {kind}")

    print()
    print("sanity check -- sparse-local mask bandwidth for layer 1:")
    sparse_block = model.blocks[1]
    assert sparse_block.use_sparse_local
    band = sparse_block.attn.band_mask[:seq_len, :seq_len]
    row = seq_len - 1  # last query position sees the largest allowed window
    n_allowed = band[row].sum().item()
    print(f"  query position {row} attends to {n_allowed} key positions "
          f"(expected <= local_window={config.local_window})")

    dense_block = model.blocks[0]
    assert not dense_block.use_sparse_local
    causal = dense_block.attn.causal_mask[:seq_len, :seq_len]
    n_allowed_dense = causal[row].sum().item()
    print(f"  (contrast) dense layer: query position {row} attends to "
          f"{n_allowed_dense} key positions (expected = {row + 1}, full causal history)")
