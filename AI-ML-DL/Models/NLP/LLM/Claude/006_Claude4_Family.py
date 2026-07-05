"""
006_Claude4_Family.py

Demonstrates the *behavioral* mechanism associated with the Claude 4 family
(Opus 4 / Sonnet 4): INTERLEAVED THINKING WITH TOOL USE. The model reasons,
calls a tool, observes the real tool result, and continues the SAME
extended-reasoning episode incorporating that result -- as opposed to a
single upfront reasoning block followed by one action (the file-005 shape).

This is a toy, self-contained PyTorch simulation of the control-flow idea,
not a reproduction of Claude 4's (undisclosed) architecture or training.
It also sketches, as a second class, the "long-horizon session" idea: a
multi-step agentic loop that must persist state across many interleaved
reasoning/tool/observation cycles, recover from a tool that fails, and
eventually terminate -- illustrating why "long-horizon agentic reliability"
is a distinct property from single-step reasoning quality (see markdown
Section 8: error recovery, drift resistance, context management).

Nothing below reflects any disclosed fact about Claude 4's internals.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Reuse a minimal decoder-only transformer -- same "one model, many modes"
# philosophy as file 005: no separate "agent network," just a control loop
# built around a single autoregressive model.
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 384
    max_seq_len: int = 1024
    dropout: float = 0.0

    bos_id: int = 1
    eos_id: int = 2
    think_id: int = 3          # marks a reasoning span
    tool_call_id: int = 4      # model requests a tool call
    tool_result_id: int = 5    # environment injects a tool result
    final_answer_id: int = 6   # model signals it is ready to answer


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AgenticLM(nn.Module):
    """Single model used across reasoning, tool-call emission, and final
    answer generation -- the same weights handle every phase of the loop."""

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
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def next_token(self, idx: torch.Tensor, temperature: float = 1.0) -> int:
        logits = self(idx)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# --------------------------------------------------------------------------- #
# Tool abstraction: a real (toy) side-effecting function the model can call
# mid-reasoning, whose actual result is fed back into context.
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    name: str
    args: Dict


@dataclass
class ToolResult:
    ok: bool
    value: str


class ToolRegistry:
    """Toy tool registry: a calculator and a flaky 'run_tests' tool that can
    fail, so the interleaved loop has something real to recover from --
    modeling the file-006 Section 8 point that long-horizon reliability
    requires graceful handling of failed/unexpected tool results, not just
    successful ones."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._attempts = 0

    def call(self, tool: ToolCall) -> ToolResult:
        if tool.name == "calculator":
            try:
                result = eval(tool.args["expr"], {"__builtins__": {}})  # toy/demo only
                return ToolResult(ok=True, value=str(result))
            except Exception as exc:  # noqa: BLE001
                return ToolResult(ok=False, value=f"error: {exc}")

        if tool.name == "run_tests":
            self._attempts += 1
            # Fails on the first attempt to force an error-recovery step,
            # then succeeds -- a toy stand-in for "fix the bug, rerun tests."
            if self._attempts == 1:
                return ToolResult(ok=False, value="2 tests failed: test_edge_case, test_overflow")
            return ToolResult(ok=True, value="all tests passed")

        return ToolResult(ok=False, value=f"unknown tool: {tool.name}")


# --------------------------------------------------------------------------- #
# The interleaved reasoning <-> tool-use <-> observation loop.
# --------------------------------------------------------------------------- #


@dataclass
class Episode:
    """Full transcript of one long-horizon agentic episode, exposed in full
    (mirrors the file-005 transparency stance: nothing here is hidden)."""

    steps: List[str] = field(default_factory=list)
    tool_calls_made: int = 0
    recovered_from_failure: bool = False
    final_answer: Optional[str] = None

    def log(self, entry: str) -> None:
        self.steps.append(entry)

    def render(self) -> str:
        return "\n".join(self.steps)


class InterleavedThinkingController:
    """
    Drives the Claude-4-style loop: THINK -> (optionally) CALL TOOL ->
    OBSERVE -> THINK AGAIN incorporating the observation -> ... -> ANSWER.

    Unlike file 005's single think-block-then-answer shape, this controller
    can re-enter the thinking phase arbitrarily many times, each time
    conditioned on newly observed, real tool output -- and it must decide,
    each cycle, whether to call another tool, keep thinking without a tool,
    or finalize an answer. A toy decision policy (rather than a trained
    stopping head) is used here for clarity; a stub is provided to show
    where a learned policy would plug in.
    """

    def __init__(self, model: AgenticLM, cfg: ModelConfig, tools: ToolRegistry):
        self.model = model
        self.cfg = cfg
        self.tools = tools

    def _decide_next_action(
        self,
        last_observation: Optional[ToolResult],
        pending_retry: bool,
        plan_exhausted: bool,
    ) -> str:
        """Toy policy standing in for a learned "what should I do next"
        decision the real model would make via sampling. Returns one of:
        'call_tool', 'keep_thinking', 'finalize'."""
        if last_observation is not None and not last_observation.ok:
            return "keep_thinking"  # error recovery: reason about the failure first
        if pending_retry or not plan_exhausted:
            return "call_tool"
        return "finalize"

    def run_episode(
        self,
        task_description: str,
        tool_plan: List[ToolCall],
        max_cycles: int = 8,
    ) -> Episode:
        episode = Episode()
        episode.log(f"[TASK] {task_description}")

        last_observation: Optional[ToolResult] = None
        pending_retry = False
        plan_ptr = 0

        for cycle in range(max_cycles):
            action = self._decide_next_action(
                last_observation, pending_retry, plan_exhausted=plan_ptr >= len(tool_plan)
            )

            if action == "keep_thinking":
                episode.log(
                    f"[THINK #{cycle}] Observed a failure; re-planning before next tool call: "
                    f"'{last_observation.value}'"
                )
                episode.recovered_from_failure = True
                # Re-attempt the same tool call that just failed rather than
                # advancing the plan pointer -- a toy stand-in for "fix and rerun."
                if plan_ptr > 0:
                    plan_ptr -= 1
                pending_retry = True
                # The failure has now been reasoned about; clear it so the next
                # cycle proceeds to retry the tool rather than looping forever
                # re-observing the same stale failure.
                last_observation = None

            elif action == "call_tool":
                if plan_ptr >= len(tool_plan):
                    episode.log(f"[THINK #{cycle}] No more planned tool calls; finalizing.")
                    break
                tool = tool_plan[plan_ptr]
                episode.log(f"[THINK #{cycle}] Deciding to call tool `{tool.name}`({tool.args})")
                result = self.tools.call(tool)
                episode.tool_calls_made += 1
                episode.log(
                    f"[TOOL_RESULT #{cycle}] ok={result.ok} value={result.value!r}"
                )
                last_observation = result
                pending_retry = False
                plan_ptr += 1
                continue

            elif action == "finalize":
                episode.log(f"[THINK #{cycle}] All required signals gathered; composing final answer.")
                episode.final_answer = self._compose_answer(episode, last_observation)
                episode.log(f"[ANSWER] {episode.final_answer}")
                break

        if episode.final_answer is None:
            episode.log("[ANSWER] (episode ended without explicit finalize; truncated)")

        return episode

    def _compose_answer(self, episode: Episode, last_observation: Optional[ToolResult]) -> str:
        # Toy "answer synthesis": a real model would autoregressively
        # generate text conditioned on the full interleaved transcript.
        # We exercise the shared model's forward pass to make the point
        # concrete: the same weights that "reasoned" are used to "answer."
        cfg = self.cfg
        dummy_context = torch.tensor([[cfg.bos_id, cfg.think_id, cfg.final_answer_id]])
        _ = self.model.next_token(dummy_context)  # exercise the model; output unused in this toy
        recovery_note = " after recovering from an initial tool failure" if episode.recovered_from_failure else ""
        return (
            f"Task complete{recovery_note}. Made {episode.tool_calls_made} tool call(s); "
            f"last observation: {last_observation.value if last_observation else 'n/a'}."
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ModelConfig()
    model = AgenticLM(cfg)
    model.eval()

    tools = ToolRegistry(seed=1)
    controller = InterleavedThinkingController(model, cfg, tools)

    plan = [
        ToolCall(name="calculator", args={"expr": "12 * 7 + 3"}),
        ToolCall(name="run_tests", args={}),   # fails first attempt -> forces error recovery
        ToolCall(name="run_tests", args={}),   # retried -> succeeds
    ]

    print("=== Interleaved thinking + tool use (Claude 4-style) ===\n")
    episode = controller.run_episode(
        task_description="Compute a value and make the test suite pass.",
        tool_plan=plan,
    )
    print(episode.render())

    print("\n--- Episode summary ---")
    print(f"Tool calls made:        {episode.tool_calls_made}")
    print(f"Recovered from failure: {episode.recovered_from_failure}")
    print(f"Final answer:           {episode.final_answer}")

    print(
        "\nContrast with file 005: here, reasoning re-enters *multiple* times, "
        "each time conditioned on a genuinely new, previously-unknown tool "
        "result (including a failure), rather than reasoning occurring once "
        "upfront before a single action."
    )
