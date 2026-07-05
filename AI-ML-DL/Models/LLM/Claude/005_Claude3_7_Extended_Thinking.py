"""
005_Claude3_7_Extended_Thinking.py

Demonstrates the *behavioral* mechanism popularized by Claude 3.7 Sonnet's
"extended thinking" mode: a single model that can operate in a fast
non-reasoning mode OR spend a caller-controlled token budget generating a
VISIBLE intermediate reasoning trace before producing a final answer.

This is a toy, fully self-contained PyTorch re-implementation of the *shape*
of the mechanism, not a reproduction of Claude's (undisclosed) architecture
or training recipe. It illustrates:

  1. A single decoder-only transformer (one set of weights) used for both
     modes -- there is no separate "reasoner" network.
  2. A `budget_tokens` parameter that upper-bounds how many reasoning tokens
     the model may emit before being forced to answer.
  3. An explicit, learned "stop thinking" decision (rather than always
     padding out to the budget), echoing the documented behavior that
     Claude's thinking mode terminates early once it has converged.
  4. Full exposure of the reasoning trace to the caller (contrast with a
     hidden-CoT design, sketched at the bottom for comparison), including a
     toy analogue of "redacted" thinking spans for content a safety filter
     wants to withhold from the visible trace while still keeping it in the
     model's own context for continuity.

None of the numbers, dimensions, or training procedure below reflect any
disclosed fact about Claude 3.7 Sonnet -- Anthropic has not published
architecture or training details for any Claude model. This file is a
pedagogical model of the *control-flow idea* only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# A minimal decoder-only transformer (shared weights for "fast" and
# "thinking" modes -- the whole point of the hybrid-reasoning design).
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    vocab_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_seq_len: int = 512
    dropout: float = 0.0

    # Special token ids (toy vocabulary convention used throughout this file)
    bos_id: int = 1
    eos_id: int = 2
    think_start_id: int = 3      # <thinking>
    think_end_id: int = 4        # </thinking>
    answer_start_id: int = 5     # <answer>
    stop_thinking_id: int = 6    # emitted by the model when it judges itself done


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class HybridReasoningLM(nn.Module):
    """One model, one set of weights, used for both fast answers and
    extended-thinking answers -- the architectural claim being illustrated
    is that "thinking mode" need not be a different network."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    @torch.no_grad()
    def next_token(self, idx: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self(idx)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# --------------------------------------------------------------------------- #
# The thinking-budget wrapper: the actual "extended thinking" mechanism.
# --------------------------------------------------------------------------- #


@dataclass
class ThinkingTrace:
    """Container exposing the full reasoning trace to the caller -- the
    defining, deliberate design choice this file demonstrates: nothing here
    is hidden from the API consumer, unlike a hidden-CoT design."""

    thinking_tokens: List[int] = field(default_factory=list)
    redacted_spans: List[List[int]] = field(default_factory=list)  # withheld from display
    answer_tokens: List[int] = field(default_factory=list)
    budget_tokens: int = 0
    tokens_used_thinking: int = 0
    stopped_early: bool = False

    def render(self, id_to_str) -> str:
        thinking_str = " ".join(id_to_str(t) for t in self.thinking_tokens)
        answer_str = " ".join(id_to_str(t) for t in self.answer_tokens)
        redaction_note = (
            f" [{len(self.redacted_spans)} redacted span(s) withheld from display]"
            if self.redacted_spans
            else ""
        )
        early_note = "stopped early" if self.stopped_early else "used full budget"
        return (
            f"<thinking budget={self.budget_tokens} used={self.tokens_used_thinking} "
            f"{early_note}{redaction_note}>\n{thinking_str}\n</thinking>\n"
            f"<answer>{answer_str}</answer>"
        )


class ExtendedThinkingController:
    """
    Wraps a HybridReasoningLM to expose a `budget_tokens`-style API, mirroring
    the documented Claude 3.7 Sonnet mechanism:

      - The caller sets an upper bound on reasoning tokens (`budget_tokens`).
      - The model generates into a <thinking>...</thinking> region and may
        emit a learned "stop thinking" signal before exhausting the budget
        (modeled here as sampling `stop_thinking_id`, or a cheap heuristic
        fallback so the toy model actually terminates early sometimes).
      - Once thinking ends (by budget exhaustion or self-termination), the
        model is forced into <answer>...</answer> and generates the final
        response conditioned on its own visible reasoning.
      - A toy "safety filter" (`redact_fn`) can mark spans of the thinking
        trace as sensitive; those tokens stay in the model's own context
        (so subsequent reasoning/answer generation can still condition on
        them) but are excluded from what `ThinkingTrace.render` shows the
        caller -- the toy analogue of Claude's `redacted_thinking` blocks.
    """

    def __init__(self, model: HybridReasoningLM, cfg: ModelConfig):
        self.model = model
        self.cfg = cfg

    def generate(
        self,
        prompt_ids: List[int],
        budget_tokens: int,
        max_answer_tokens: int = 40,
        temperature: float = 0.9,
        redact_fn: Optional[callable] = None,
        device: str = "cpu",
    ) -> ThinkingTrace:
        cfg = self.cfg
        trace = ThinkingTrace(budget_tokens=budget_tokens)

        seq = list(prompt_ids) + [cfg.think_start_id]
        idx = torch.tensor([seq], dtype=torch.long, device=device)

        # --- Thinking phase: bounded by budget_tokens, may stop early. ---
        for step in range(budget_tokens):
            nxt = self.model.next_token(idx, temperature=temperature).item()

            if nxt == cfg.stop_thinking_id:
                trace.stopped_early = True
                break

            trace.thinking_tokens.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
            trace.tokens_used_thinking += 1

            if redact_fn is not None and redact_fn(step, nxt):
                trace.redacted_spans.append([nxt])
        else:
            trace.stopped_early = False  # exhausted the full budget

        # --- Transition: force the model into the answer region. ---
        idx = torch.cat(
            [idx, torch.tensor([[cfg.think_end_id, cfg.answer_start_id]], device=device)],
            dim=1,
        )

        # --- Answer phase: generation conditioned on the visible trace. ---
        for _ in range(max_answer_tokens):
            nxt = self.model.next_token(idx, temperature=temperature).item()
            if nxt == cfg.eos_id:
                break
            trace.answer_tokens.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)

        return trace


class HiddenReasoningController(ExtendedThinkingController):
    """
    Contrast case, included purely for pedagogy: an OpenAI o1/o3-style hidden
    chain-of-thought design using the *same* underlying model. Reasoning
    tokens are still generated (compute is still spent), but `generate`
    returns only a caller-facing *summary* placeholder instead of the raw
    trace -- the caller never sees the actual reasoning tokens, which is
    exactly the property Claude 3.7 Sonnet's design deliberately avoids.
    """

    def generate(self, *args, **kwargs) -> ThinkingTrace:  # type: ignore[override]
        trace = super().generate(*args, **kwargs)
        # Simulate "hiding" the trace: replace it with an opaque summary and
        # discard the actual tokens from what would be returned to a caller.
        n_thought = len(trace.thinking_tokens)
        trace.thinking_tokens = []  # not returned to the caller in this mode
        trace.redacted_spans = [[]] * n_thought  # entire trace is withheld
        return trace


# --------------------------------------------------------------------------- #
# Toy vocabulary + demo
# --------------------------------------------------------------------------- #


def build_toy_vocab(cfg: ModelConfig):
    words = [f"tok{i}" for i in range(cfg.vocab_size)]
    words[cfg.bos_id] = "<bos>"
    words[cfg.eos_id] = "<eos>"
    words[cfg.think_start_id] = "<thinking>"
    words[cfg.think_end_id] = "</thinking>"
    words[cfg.answer_start_id] = "<answer>"
    words[cfg.stop_thinking_id] = "<stop_thinking>"

    def id_to_str(i: int) -> str:
        return words[i] if 0 <= i < len(words) else f"<unk:{i}>"

    return id_to_str


def demo_flag_sensitive_reasoning(step: int, token_id: int) -> bool:
    """Toy 'safety classifier': flags every 5th reasoning token as sensitive,
    purely to exercise the redaction path."""
    return (step + 1) % 5 == 0


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ModelConfig()
    model = HybridReasoningLM(cfg)
    model.eval()

    id_to_str = build_toy_vocab(cfg)
    prompt = [cfg.bos_id, 42, 17, 88]  # toy tokenized "question"

    visible_controller = ExtendedThinkingController(model, cfg)
    hidden_controller = HiddenReasoningController(model, cfg)

    print("=== Visible extended thinking (Claude 3.7 Sonnet-style) ===")
    for budget in (0, 8, 32, 128):
        # budget=0 approximates "thinking disabled" -- the fast, non-reasoning mode.
        trace = visible_controller.generate(
            prompt,
            budget_tokens=max(budget, 0),
            redact_fn=demo_flag_sensitive_reasoning,
        )
        mode = "fast (no thinking)" if budget == 0 else f"extended thinking (budget={budget})"
        print(f"\n--- mode: {mode} ---")
        print(trace.render(id_to_str))

    print("\n=== Hidden reasoning (o1/o3-style contrast, same underlying model) ===")
    hidden_trace = hidden_controller.generate(prompt, budget_tokens=32)
    print(hidden_trace.render(id_to_str))
    print(
        "\nNote: compute for reasoning was still spent in the hidden case; "
        "only the caller-visible surface differs. This is the crux of the "
        "005 transparency discussion: visibility is a design/serving choice "
        "layered on top of test-time compute, not an inherent property of it."
    )
