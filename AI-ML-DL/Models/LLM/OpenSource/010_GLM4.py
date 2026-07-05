"""
GLM / GLM-130B (Zhipu AI + Tsinghua KEG, 2021/2022) -- reference implementation
of the GLM autoregressive blank-infilling pretraining objective and its 2D
positional encoding scheme, the one genuinely distinctive research idea behind
the whole GLM -> ChatGLM -> GLM-4 lineage (see 010_GLM4.md Section 2 for the
full prose derivation).

What this file demonstrates end-to-end:
  1. `sample_spans` -- samples non-overlapping spans to mask from a token
     sequence, either several short scattered spans (the "[MASK]" regime) or
     a single long trailing span (the "[gMASK]" regime), matching the two
     masking regimes described in the GLM-130B paper.
  2. `build_glm_example` -- constructs the actual GLM training example from a
     raw token sequence and a set of sampled spans:
       - Part A: the corrupted sequence, each span collapsed to one mask
         token, retained in original order.
       - Part B: the masked spans' content, concatenated in a *randomly
         permuted* order, each span teacher-forced with a leading mask token.
       - A prefix-LM attention mask: Part A attends to Part A bidirectionally;
         Part B attends causally to itself and to all of Part A; Part A never
         attends to Part B.
       - 2D position ids: position-id-1 = position in the ORIGINAL sequence
         (shared by every token of a given span); position-id-2 = offset
         WITHIN the span currently being generated (0 for Part A).
  3. `GLMModel` -- a small transformer that embeds tokens plus the two
     positional-id streams (two learned embedding tables, summed), applies the
     prefix-LM attention mask via `F.scaled_dot_product_attention`, and
     computes the span-infilling cross-entropy loss over Part B only.

This is a simplified but mechanically faithful implementation: real
GLM-130B additionally uses DeepNorm, GeGLU, and rotary embeddings layered on
top of the 2D scheme (see the .md for those engineering specifics); this file
isolates the *objective* itself, which is the part worth understanding cold.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GLMConfig:
    vocab_size: int = 64          # includes a reserved [MASK] id (see mask_token_id)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_position: int = 64        # max value either position-id stream can take
    mask_token_id: int = 0        # reserved vocab slot for the mask placeholder
    span_lambda: float = 3.0      # Poisson(lambda) span-length sampling, per the GLM paper
    mask_frac: float = 0.15       # target fraction of the sequence covered by masked spans


# ---------------------------------------------------------------------------
# 1. Span sampling: [MASK] (several short spans) vs [gMASK] (one trailing span)
# ---------------------------------------------------------------------------

def sample_spans(seq_len: int, cfg: GLMConfig, gmask_mode: bool = False) -> list[tuple[int, int]]:
    """
    Returns a list of non-overlapping (start, length) spans to mask.

    gmask_mode=False: the "[MASK]" regime -- several short spans, lengths drawn
      from Poisson(cfg.span_lambda) (clamped to >=1), sampled until roughly
      cfg.mask_frac of the sequence is covered. This favors understanding /
      short-infilling behavior.
    gmask_mode=True: the "[gMASK]" regime -- a single long span running from a
      random cut point to the end of the sequence. This favors long-form
      generation behavior, since "predict everything after this point" is
      exactly open-ended continuation.
    """
    if gmask_mode:
        cut = random.randint(max(1, seq_len // 3), seq_len - 1)
        return [(cut, seq_len - cut)]

    target_masked = max(1, int(seq_len * cfg.mask_frac))
    spans: list[tuple[int, int]] = []
    covered = torch.zeros(seq_len, dtype=torch.bool)
    total_masked = 0
    attempts = 0
    while total_masked < target_masked and attempts < 50:
        attempts += 1
        length = max(1, int(torch.poisson(torch.tensor(cfg.span_lambda)).item()))
        length = min(length, seq_len // 2)
        start = random.randint(0, seq_len - length)
        region = covered[start:start + length]
        if region.any():
            continue  # overlaps an already-sampled span; resample
        covered[start:start + length] = True
        spans.append((start, length))
        total_masked += length
    spans.sort()
    return spans


# ---------------------------------------------------------------------------
# 2. Build the corrupted input (Part A + permuted Part B), 2D position ids,
#    the prefix-LM attention mask, and the infilling targets.
# ---------------------------------------------------------------------------

@dataclass
class GLMExample:
    input_ids: torch.Tensor       # [total_len] Part A followed by Part B
    pos_id_1: torch.Tensor        # [total_len] position in the ORIGINAL sequence
    pos_id_2: torch.Tensor        # [total_len] offset within the current span (0 for Part A)
    attn_mask: torch.Tensor       # [total_len, total_len] bool, True = "may attend to"
    part_b_start: int             # index where Part B begins
    targets: torch.Tensor         # [len(Part B)] the token GLM must predict at each Part-B position


def build_glm_example(tokens: torch.Tensor, spans: list[tuple[int, int]], cfg: GLMConfig) -> GLMExample:
    """
    tokens: [seq_len] the original, uncorrupted token sequence.
    spans: non-overlapping (start, length) spans from `sample_spans`.
    """
    seq_len = tokens.shape[0]
    mask_id = cfg.mask_token_id

    # --- Part A: original sequence with each span collapsed to one mask token.
    part_a_ids: list[int] = []
    part_a_pos1: list[int] = []
    mask_slot_original_pos: list[int] = []  # original position of each span's mask placeholder
    span_content: list[torch.Tensor] = []

    covered = torch.zeros(seq_len, dtype=torch.bool)
    for start, length in spans:
        covered[start:start + length] = True
        span_content.append(tokens[start:start + length])

    i = 0
    span_iter = iter(spans)
    next_span = next(span_iter, None)
    while i < seq_len:
        if next_span is not None and i == next_span[0]:
            part_a_ids.append(mask_id)
            part_a_pos1.append(i)  # the mask placeholder's position IS its original position
            mask_slot_original_pos.append(i)
            i += next_span[1]
            next_span = next(span_iter, None)
        else:
            part_a_ids.append(int(tokens[i].item()))
            part_a_pos1.append(i)
            i += 1

    part_a_len = len(part_a_ids)

    # --- Part B: span contents, in a RANDOMLY PERMUTED order, each teacher-forced
    #     with a leading mask token. This permutation is exactly why 2D position
    #     ids are necessary: a token's literal sequential position in Part B no
    #     longer reveals which original slot it belongs to.
    order = list(range(len(spans)))
    random.shuffle(order)

    part_b_ids: list[int] = []
    part_b_pos1: list[int] = []
    part_b_pos2: list[int] = []
    targets: list[int] = []

    for span_idx in order:
        content = span_content[span_idx]              # [span_len]
        original_pos = mask_slot_original_pos[span_idx]
        span_len = content.shape[0]

        # Teacher-forced input for this span: [MASK] w1 w2 ... w_{L-1}
        # Target for this span:                 w1 w2 ... w_L
        span_input = [mask_id] + content[:-1].tolist()
        span_target = content.tolist()

        part_b_ids.extend(span_input)
        targets.extend(span_target)
        part_b_pos1.extend([original_pos] * span_len)      # all tokens of a span share position-id-1
        part_b_pos2.extend(list(range(1, span_len + 1)))    # position-id-2 = offset within the span

    total_len = part_a_len + len(part_b_ids)
    input_ids = torch.tensor(part_a_ids + part_b_ids, dtype=torch.long)
    pos_id_1 = torch.tensor(part_a_pos1 + part_b_pos1, dtype=torch.long)
    pos_id_2 = torch.tensor([0] * part_a_len + part_b_pos2, dtype=torch.long)
    targets_t = torch.tensor(targets, dtype=torch.long)

    # --- Prefix-LM attention mask: Part A <-> Part A bidirectional; Part B is
    #     causal over itself and can see all of Part A; Part A cannot see Part B.
    attn_mask = torch.zeros(total_len, total_len, dtype=torch.bool)
    attn_mask[:part_a_len, :part_a_len] = True                    # A sees A, bidirectionally
    attn_mask[part_a_len:, :part_a_len] = True                    # B sees all of A
    b_causal = torch.tril(torch.ones(total_len - part_a_len, total_len - part_a_len, dtype=torch.bool))
    attn_mask[part_a_len:, part_a_len:] = b_causal                # B sees B causally (including itself)
    # Part A rows beyond column part_a_len remain False: A never sees B.

    return GLMExample(input_ids, pos_id_1, pos_id_2, attn_mask, part_a_len, targets_t)


# ---------------------------------------------------------------------------
# 3. A small transformer using the 2D positional scheme + prefix-LM mask.
# ---------------------------------------------------------------------------

class TwoDPositionalEmbedding(nn.Module):
    """
    Two independent learned embedding tables, one per position-id axis, summed
    together (and summed into the token embedding). Real GLM/GLM-130B combine
    this idea with rotary embeddings in later layers; this module isolates the
    core "a position is a PAIR of ids, each independently embedded" mechanic.
    """

    def __init__(self, cfg: GLMConfig):
        super().__init__()
        self.pos1_embed = nn.Embedding(cfg.max_position, cfg.d_model)
        self.pos2_embed = nn.Embedding(cfg.max_position, cfg.d_model)

    def forward(self, pos_id_1: torch.Tensor, pos_id_2: torch.Tensor) -> torch.Tensor:
        return self.pos1_embed(pos_id_1) + self.pos2_embed(pos_id_2)


class GLMAttention(nn.Module):
    """Standard multi-head attention, but driven by an explicit prefix-LM mask
    rather than a fixed causal mask -- this is the only structural change from
    a plain transformer needed to implement GLM's attention pattern."""

    def __init__(self, cfg: GLMConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        # attn_mask: [t, t] bool, True = "may attend to". SDPA expects an additive
        # float mask or a bool mask where True means "keep" -- broadcast over batch/heads.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.proj(out)


class GLMBlock(nn.Module):
    def __init__(self, cfg: GLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = GLMAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),  # stands in for GLM-130B's GeGLU; plain GELU FFN here for simplicity
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class GLMModel(nn.Module):
    def __init__(self, cfg: GLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = TwoDPositionalEmbedding(cfg)
        self.blocks = nn.ModuleList(GLMBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ex: GLMExample) -> torch.Tensor:
        """Returns logits over the FULL sequence; the caller slices out Part B
        for the infilling loss (Part A tokens are context only, never targets)."""
        x = self.tok_embed(ex.input_ids) + self.pos_embed(ex.pos_id_1, ex.pos_id_2)
        x = x.unsqueeze(0)  # add batch dim of 1
        mask = ex.attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, t, t] broadcast over batch/heads
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.head(x).squeeze(0)  # [total_len, vocab_size]

    def infilling_loss(self, ex: GLMExample) -> torch.Tensor:
        logits = self.forward(ex)
        part_b_logits = logits[ex.part_b_start:]  # only Part B carries a training signal
        return F.cross_entropy(part_b_logits, ex.targets)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    cfg = GLMConfig()
    model = GLMModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    seq_len = 20
    # keep tokens away from the reserved mask id (0)
    tokens = torch.randint(1, cfg.vocab_size, (seq_len,))

    print("=== GLM autoregressive blank-infilling objective ===")
    print(f"original sequence length: {seq_len}, vocab_size: {cfg.vocab_size}, "
          f"total params (toy scale): {total_params:,}")

    print("\n--- [MASK] regime: several short scattered spans ---")
    spans = sample_spans(seq_len, cfg, gmask_mode=False)
    print(f"sampled spans (start, length): {spans}")
    example = build_glm_example(tokens, spans, cfg)
    print(f"Part A length: {example.part_b_start}, Part B length: {example.input_ids.shape[0] - example.part_b_start}")
    print(f"position-id-1 (original-sequence position): {example.pos_id_1.tolist()}")
    print(f"position-id-2 (offset within current span):  {example.pos_id_2.tolist()}")
    loss = model.infilling_loss(example)
    print(f"infilling cross-entropy loss (random init, [MASK] regime): {loss.item():.4f}")

    print("\n--- [gMASK] regime: one long trailing span (generation-style) ---")
    gspans = sample_spans(seq_len, cfg, gmask_mode=True)
    print(f"sampled span (start, length): {gspans}")
    gexample = build_glm_example(tokens, gspans, cfg)
    print(f"Part A length: {gexample.part_b_start}, Part B length: {gexample.input_ids.shape[0] - gexample.part_b_start}")
    gloss = model.infilling_loss(gexample)
    print(f"infilling cross-entropy loss (random init, [gMASK] regime): {gloss.item():.4f}")

    print("\n--- Attention mask sanity check (prefix-LM pattern) ---")
    a_len = example.part_b_start
    print(f"Part A block is fully bidirectional (all True)?  "
          f"{bool(example.attn_mask[:a_len, :a_len].all())}")
    print(f"Part A never attends into Part B (all False)?    "
          f"{not bool(example.attn_mask[:a_len, a_len:].any())}")
    print(f"Part B attends to all of Part A (all True)?      "
          f"{bool(example.attn_mask[a_len:, :a_len].all())}")
    b_block = example.attn_mask[a_len:, a_len:]
    is_causal = torch.equal(b_block, torch.tril(torch.ones_like(b_block)))
    print(f"Part B block is exactly lower-triangular (causal)? {is_causal}")

    print(
        "\n(Real GLM-130B: 130B dense params, bilingual EN/ZH, ~400B training tokens, "
        "DeepNorm + GeGLU + RoPE layered on top of this same 2D positional scheme, "
        "trained on 96 DGX-A100 servers, later INT4-quantized for single-consumer-GPU inference.)"
    )
