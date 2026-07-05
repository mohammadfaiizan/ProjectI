"""
GPT-1 (Radford et al., "Improving Language Understanding by Generative
Pre-Training", OpenAI, 2018) -- reference PyTorch implementation.

What this file demonstrates
----------------------------
1. A decoder-only, causally-masked multi-head self-attention Transformer
   stack (NOT the full Vaswani et al. 2017 decoder -- there is no
   encoder-decoder cross-attention sub-layer, because GPT-1 has no encoder).
2. Learned absolute positional embeddings (a fixed-size lookup table added
   to token embeddings), as opposed to sinusoidal encodings (Vaswani et al.)
   or rotary embeddings (RoPE, which postdates this model by several years).
3. Post-LayerNorm residual blocks ("LayerNorm after the residual add"),
   matching the original 2018 GPT-1 paper's description and the original
   Transformer convention. NOTE: GPT-2 switches to pre-LayerNorm ("LayerNorm
   before the sub-layer, inside the residual branch") because it is more
   stable for deep stacks without careful LR warmup. This file intentionally
   implements POST-norm to be historically faithful to GPT-1; the docstring
   on GPTBlock below repeats this choice so it isn't missed.
4. GELU activations in the feed-forward network (GPT-1 deliberately departs
   from the original Transformer's ReLU).
5. The paper's distinctive fine-tuning methodology: a task-specific input
   is fed through the SAME pretrained backbone, a linear+softmax
   classification head reads out the hidden state at the final ("extract")
   token position, and the fine-tuning loss is a weighted sum of the
   supervised task loss and an auxiliary language-modeling loss:

        L_3 = L_task(classification head) + lambda * L_lm(next-token pred)

   This auxiliary-LM-loss-during-fine-tuning pattern is GPT-1's most
   distinctive methodological contribution (as opposed to "decoder block",
   which GPT-2/GPT-3 reference files will also cover) and is implemented
   here via GPT1ForClassification.forward(..., compute_lm_loss=True).

This file is self-contained: only `torch`, `math`, and `dataclasses` are
used. It is meant as clear, correct, illustrative code -- not an optimized
or production training script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPT1Config:
    """GPT-1-scale-able hyperparameters.

    Defaults below match the paper's full-size model:
        12 layers, d_model=768, 12 heads (64 dim/head), FFN inner=3072 (4x),
        block_size=512, vocab_size~=40000 (BPE), dropout=0.1.
    The __main__ block instantiates a much smaller config for a fast,
    illustrative forward pass.
    """

    vocab_size: int = 40_000
    block_size: int = 512        # max context length (learned pos-embed table size)
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072             # 4x d_model
    dropout: float = 0.1
    init_std: float = 0.02       # N(0, 0.02) weight init, per the paper

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal mask.

    No cross-attention sub-layer exists anywhere in this file: GPT-1 is
    decoder-only in the sense of "masked self-attention blocks stacked",
    with the encoder-decoder cross-attention of the original Transformer
    decoder removed entirely (there is no encoder to attend to).
    """

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        # Combined QKV projection (equivalent to three separate d_model x d_model
        # matrices, computed as one matmul for efficiency).
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask, precomputed for the maximum context length and sliced
        # at forward time for whatever sequence length is actually used.
        causal_mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        b, t, c = x.shape

        qkv = self.qkv_proj(x)  # (b, t, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to (b, n_heads, t, d_head) for per-head attention.
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention scores: (b, n_heads, t, t)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply the causal mask: position j may only attend to k <= j.
        mask = self.causal_mask[:t, :t]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (b, n_heads, t, d_head)
        out = out.transpose(1, 2).contiguous().view(b, t, c)  # (b, t, d_model)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise FFN with GELU activation (GPT-1 uses GELU, not ReLU)."""

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class GPT1Block(nn.Module):
    """One transformer block, implemented POST-norm to match GPT-1 (2018).

    Historical note (deliberate design choice, not an oversight): the
    original Transformer (Vaswani et al., 2017) and GPT-1 both apply
    LayerNorm AFTER the residual addition:

        x = LayerNorm(x + SubLayer(x))

    GPT-2 (2019) switches to PRE-norm:

        x = x + SubLayer(LayerNorm(x))

    Pre-norm keeps an unnormalized residual/gradient highway through the
    full depth of the network, which is materially more stable for deep
    stacks and is why GPT-2/3 and virtually all modern transformers use it.
    GPT-1's 12-layer post-norm stack instead leans on a careful learning
    rate warmup (see the paper: warmup over the first 2000 updates) to
    control early-training instability. This implementation intentionally
    keeps post-norm to be historically faithful to the 2018 paper.
    """

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ffn = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))   # residual add, THEN normalize (post-norm)
        x = self.ln2(x + self.ffn(x))    # residual add, THEN normalize (post-norm)
        return x


# ---------------------------------------------------------------------------
# Backbone: token + learned position embeddings + transformer stack
# ---------------------------------------------------------------------------

class GPT1Backbone(nn.Module):
    """The pretrained backbone: embeddings + N GPT1Block layers + final LN.

    Positional information comes from a LEARNED embedding table of shape
    (block_size, d_model), summed elementwise with token embeddings -- not
    a sinusoidal encoding and not RoPE. This ties the model to a fixed
    maximum context length equal to `block_size` (512 in the full-scale
    paper config).
    """

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([GPT1Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # N(0, 0.02) initialization, per the paper.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len) of token indices.
        b, t = input_ids.shape
        assert t <= self.cfg.block_size, (
            f"sequence length {t} exceeds block_size {self.cfg.block_size}"
        )

        positions = torch.arange(t, device=input_ids.device).unsqueeze(0)  # (1, t)
        x = self.token_embed(input_ids) + self.pos_embed(positions)       # (b, t, d_model)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        return x  # (b, t, d_model) -- final hidden states


# ---------------------------------------------------------------------------
# Pretraining head: tied LM head for next-token prediction
# ---------------------------------------------------------------------------

class GPT1LMHead(nn.Module):
    """Language-modeling head with weight tying to the token embedding.

    Tying the output projection to the input embedding matrix (rather than
    learning a separate vocab_size x d_model matrix) is standard practice
    and is how the paper's ~117M parameter count is reached without an
    extra ~30M untied output-projection parameters.
    """

    def __init__(self, backbone: GPT1Backbone):
        super().__init__()
        self.backbone = backbone
        # No separate weight matrix: reuse token_embed.weight (tied).

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids)                      # (b, t, d_model)
        logits = hidden @ self.backbone.token_embed.weight.T   # (b, t, vocab_size)
        return logits

    def lm_loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Standard next-token cross-entropy LM loss (the pretraining objective)."""
        logits = self(input_ids)
        # Shift so that position i predicts token i+1.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )


# ---------------------------------------------------------------------------
# Fine-tuning head: task classification + auxiliary LM loss (paper Section 3)
# ---------------------------------------------------------------------------

class GPT1ForClassification(nn.Module):
    """Reproduces GPT-1's fine-tuning recipe.

    Per the paper, downstream tasks are handled by:
      1. Reformatting the task input into a single token sequence using
         delimiter tokens (e.g. [start] premise [delim] hypothesis [extract]),
         so the SAME pretrained backbone architecture is reused unchanged
         (only the input format changes -- this is "task-specific input
         transformation" as opposed to "task-specific architecture").
      2. Extracting the backbone's hidden state at the final ("extract")
         token position of the formatted sequence.
      3. Feeding that hidden state into a newly-initialized linear+softmax
         classification head: P(y | x) = softmax(h_last @ W_y + b_y).
      4. Fine-tuning with a joint loss combining the supervised task loss
         and an auxiliary language-modeling loss on the same inputs:

             L_3 = L_task + lambda_lm * L_lm

         The auxiliary LM term regularizes against catastrophic forgetting
         of the pretrained representation while adapting to the task, and
         the paper reports it improves generalization and speeds
         convergence.

    This class assumes the caller has already appended an "extract" token
    as the LAST token of each input sequence (the paper's convention);
    `extract_token_pos` here is simply "the last real position per example"
    (we assume right-padding is not used / all sequences in a batch share
    length, for simplicity -- a production implementation would take an
    explicit per-example index or an attention mask).
    """

    def __init__(self, backbone: GPT1Backbone, num_classes: int, lambda_lm: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.lambda_lm = lambda_lm
        self.classifier = nn.Linear(backbone.cfg.d_model, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=backbone.cfg.init_std)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        compute_lm_loss: bool = True,
    ) -> dict:
        """
        input_ids: (batch, seq_len) -- last position of each sequence is the
                   task's "extract" token.
        labels:    (batch,) integer class labels, or None at pure-inference time.
        Returns a dict with 'logits' (classification logits), and, if labels
        are given, 'loss' = L_task + lambda_lm * L_lm (when compute_lm_loss=True)
        else 'loss' = L_task alone.
        """
        hidden = self.backbone(input_ids)         # (b, t, d_model)
        extract_hidden = hidden[:, -1, :]         # (b, d_model) -- final token's state
        class_logits = self.classifier(extract_hidden)  # (b, num_classes)

        out = {"logits": class_logits}
        if labels is not None:
            task_loss = F.cross_entropy(class_logits, labels)

            if compute_lm_loss:
                # Auxiliary LM loss on the same formatted input sequence,
                # using the tied embedding matrix as the output projection
                # (no separate LM head module needed).
                lm_logits = hidden @ self.backbone.token_embed.weight.T  # (b, t, vocab)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                out["loss"] = task_loss + self.lambda_lm * lm_loss
                out["task_loss"] = task_loss
                out["lm_loss"] = lm_loss
            else:
                out["loss"] = task_loss

        return out


# ---------------------------------------------------------------------------
# Parameter counting helper
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Illustrative smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Small illustrative config (NOT the full 117M-parameter paper config --
    # this is deliberately tiny so the forward pass is fast to run/inspect).
    small_cfg = GPT1Config(
        vocab_size=1000,
        block_size=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=4 * 64,
        dropout=0.1,
    )

    print("=== GPT-1 illustrative smoke test (small config) ===")
    print(small_cfg)

    backbone = GPT1Backbone(small_cfg)
    lm_model = GPT1LMHead(backbone)

    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, small_cfg.vocab_size, (batch_size, seq_len))

    # --- 1. Pretraining-style forward pass: next-token LM loss ---
    logits = lm_model(input_ids)
    print(f"\n[Pretraining] input_ids shape:  {tuple(input_ids.shape)}")
    print(f"[Pretraining] LM logits shape:  {tuple(logits.shape)}  "
          f"(batch, seq_len, vocab_size)")

    lm_labels = input_ids.clone()  # standard LM setup: labels == shifted input_ids
    loss = lm_model.lm_loss(input_ids, lm_labels)
    print(f"[Pretraining] LM cross-entropy loss: {loss.item():.4f}")

    # --- 2. Fine-tuning-style forward pass: classification + auxiliary LM loss ---
    num_classes = 3  # e.g. entailment / contradiction / neutral
    clf_model = GPT1ForClassification(backbone, num_classes=num_classes, lambda_lm=0.5)

    task_labels = torch.randint(0, num_classes, (batch_size,))
    out = clf_model(input_ids, labels=task_labels, compute_lm_loss=True)

    print(f"\n[Fine-tuning] classification logits shape: {tuple(out['logits'].shape)}  "
          f"(batch, num_classes)")
    print(f"[Fine-tuning] task_loss = {out['task_loss'].item():.4f}, "
          f"lm_loss = {out['lm_loss'].item():.4f}, "
          f"combined loss (L_task + 0.5 * L_lm) = {out['loss'].item():.4f}")

    # --- 3. Parameter counts ---
    print("\n=== Parameter counts (small illustrative config) ===")
    print(f"Backbone params:                 {count_parameters(backbone):,}")
    print(f"LM head (tied, +0 extra params): {count_parameters(lm_model):,}")
    print(f"Classification head extra params:"
          f" {count_parameters(clf_model.classifier):,}")
    print(f"Full fine-tuning model params:   {count_parameters(clf_model):,}")

    # --- 4. For reference: parameter count at the FULL paper-scale config ---
    full_cfg = GPT1Config()  # defaults = paper's 117M-parameter configuration
    full_backbone = GPT1Backbone(full_cfg)
    print("\n=== Parameter count at full GPT-1 paper scale ===")
    print(full_cfg)
    print(f"Full-scale backbone params (incl. tied embeddings): "
          f"{count_parameters(full_backbone):,}  "
          f"(paper reports ~117,000,000 total)")
