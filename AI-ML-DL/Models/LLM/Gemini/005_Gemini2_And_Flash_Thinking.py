"""
Gemini 2.0/2.5 and "Flash Thinking" (Google DeepMind, 2024-2025) -- toy
dual-mode module demonstrating the core product idea this generation is
built around: ONE model that can run in a "fast" mode (a single forward
pass, minimal latency/cost) or a "thinking" mode (iterative reasoning
steps, up to a configurable token/step budget), with a single knob
(`thinking_budget`) trading inference cost against answer quality --
rather than requiring two separate models (a fast one and a reasoning
one).

This is a structural/mechanistic toy, not a claim about Gemini's actual
training or internals (undisclosed -- see the .md file, Section 11). The
mechanism modeled here: each "thinking step" re-reads the running hidden
state through a shared reasoning block and refines it; the final answer
head reads whatever hidden state is available when the budget runs out
(including a budget of zero, which just skips straight to the answer
head on the raw encoded input -- the "fast path").
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModeReasoner(nn.Module):
    """A toy model with a shared encoder, a shared iterative "thinking" block,
    and an answer head, where `thinking_budget` (steps, analogous to a
    thinking-token budget) controls how many refinement iterations run
    before the answer head is invoked.

    thinking_budget = 0  -> "fast" mode: encode once, answer immediately.
    thinking_budget = k  -> "thinking" mode: run the shared reasoning block
                            for k iterations, refining the hidden state,
                            before answering.

    The SAME weights are used regardless of budget -- this is the point:
    it is one model with a configurable inference-time cost/quality knob,
    not a routing decision between two differently-trained models.
    """

    def __init__(self, vocab_size: int, d_model: int, num_classes: int, max_budget: int = 8):
        super().__init__()
        self.max_budget = max_budget
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        # Shared "thinking" block: applied repeatedly, up to `thinking_budget`
        # times, each time refining the running hidden state. Weight-shared
        # across iterations (an iterative refinement operator, not a stack
        # of distinct layers), analogous to spending more reasoning tokens
        # with the same underlying policy rather than a deeper network.
        self.think_block = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        # A learned per-step "should I stop early" confidence signal --
        # illustrates that a well-trained reasoning policy should be able to
        # spend LESS than its full budget on easy inputs, not always burn
        # the whole budget regardless of difficulty.
        self.halt_head = nn.Linear(d_model, 1)
        self.answer_head = nn.Linear(d_model, num_classes)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids).mean(dim=1)  # simple pooled representation
        return self.encoder(x)

    def forward(
        self, token_ids: torch.Tensor, thinking_budget: int = 0, halt_threshold: float = 0.9
    ) -> dict:
        assert 0 <= thinking_budget <= self.max_budget, (
            f"thinking_budget must be in [0, {self.max_budget}]"
        )
        h = self.encode(token_ids)

        if thinking_budget == 0:
            # FAST MODE: single forward pass, answer immediately.
            logits = self.answer_head(h)
            return {"logits": logits, "steps_used": 0, "mode": "fast"}

        # THINKING MODE: iteratively refine h, up to `thinking_budget` steps,
        # but allow early exit if the halt signal is confident enough --
        # this is the mechanism that lets an easy input spend LESS than the
        # full configured budget, which matters because always spending the
        # full budget regardless of difficulty is exactly the inefficiency
        # a per-request thinking-budget control is meant to avoid.
        steps_used = 0
        for step in range(thinking_budget):
            h = h + self.think_block(h)  # residual refinement step
            steps_used += 1
            halt_prob = torch.sigmoid(self.halt_head(h)).mean()
            if halt_prob.item() > halt_threshold:
                break

        logits = self.answer_head(h)
        return {"logits": logits, "steps_used": steps_used, "mode": "thinking"}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size, d_model, num_classes = 4000, 128, 10
    model = DualModeReasoner(vocab_size, d_model, num_classes, max_budget=8)
    print(f"Total parameters (shared across BOTH modes): {count_parameters(model):,}\n")

    batch, seq_len = 4, 16
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    print("=== Fast mode (thinking_budget=0): single forward pass ===")
    out_fast = model(tokens, thinking_budget=0)
    print(f"logits shape={tuple(out_fast['logits'].shape)}  "
          f"steps_used={out_fast['steps_used']}  mode={out_fast['mode']}\n")

    print("=== Thinking mode with increasing budgets ===")
    for budget in [1, 2, 4, 8]:
        out = model(tokens, thinking_budget=budget, halt_threshold=0.999)  # force full budget
        print(f"thinking_budget={budget:<2} -> steps_used={out['steps_used']:<2} "
              f"logits shape={tuple(out['logits'].shape)}  mode={out['mode']}")

    print("\n=== Early-exit behavior: a lenient halt threshold can stop before "
          "the full budget is spent (analogous to not burning reasoning tokens "
          "on an 'easy' input) ===")
    out_early = model(tokens, thinking_budget=8, halt_threshold=0.5)
    print(f"thinking_budget=8, halt_threshold=0.5 -> steps_used={out_early['steps_used']} "
          f"(<= 8, demonstrating budget != guaranteed cost)")

    print("\nCost/quality trade-off knob: thinking_budget is set PER CALL, on the "
          "SAME weights, at inference time -- this is the mechanism this file "
          "demonstrates as the toy analog of Gemini 2.5's developer-facing "
          "thinking-budget API parameter.")
