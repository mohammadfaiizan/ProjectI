"""
007_Claude_Opus_4_Latest.py

Demonstrates a toy TIERED-ROUTING / CAPABILITY-BUDGET abstraction consistent
with the flagship positioning described in 007's markdown companion: Opus
as the highest-capability/highest-cost/highest-latency tier, reserved for
tasks where difficulty dominates the decision, versus Sonnet- and
Haiku-analogue tiers for balanced and high-throughput/low-latency workloads
respectively.

This extends the file-005 "thinking budget" idea (a per-call token dial) in
two directions, both consistent with the markdown's Section 6/7 discussion
of a generalized "effort" control:

  1. A coarse-grained EFFORT level (low/medium/high) that governs not just
     reasoning-token budget but also how many agentic sub-steps (think ->
     act -> verify cycles) the model is willing to spend on a task -- a toy
     analogue of "effort" as a dial over an entire episode, not just over
     raw thinking tokens.
  2. A TIER ROUTER that picks which capability tier (opus/sonnet/haiku
     analogues -- here just three configurations of the same underlying toy
     model, at different simulated cost/latency/capability points) should
     handle a given request, based on a declared task-difficulty signal --
     illustrating the product-engineering problem of routing across a
     capability ladder rather than always calling the flagship.

As with files 005 and 006, none of the concrete numbers here are disclosed
Anthropic facts; this is a pedagogical model of the *design pattern*, built
on the same minimal decoder-only transformer used in file 005, reused here
at three different configured "tiers" to keep the file self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Minimal decoder-only transformer (same shape as files 005/006; duplicated
# here so this file stays fully self-contained per the assignment's
# requirements).
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 256
    bos_id: int = 1
    eos_id: int = 2


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


class ToyLM(nn.Module):
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
# Effort levels: a coarse-grained dial over an entire agentic episode, not
# just over raw reasoning-token count (the generalization the 007 markdown
# discusses relative to file 005's narrower thinking-budget parameter).
# --------------------------------------------------------------------------- #


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


EFFORT_PROFILE: Dict[Effort, Dict[str, int]] = {
    # thinking_tokens: reasoning-token allowance per sub-step (file-005 style)
    # max_subcycles:   how many think->act->verify cycles the episode may run
    # verification_passes: how many times the model double-checks its answer
    Effort.LOW:    {"thinking_tokens": 16,  "max_subcycles": 1, "verification_passes": 0},
    Effort.MEDIUM: {"thinking_tokens": 64,  "max_subcycles": 3, "verification_passes": 1},
    Effort.HIGH:   {"thinking_tokens": 256, "max_subcycles": 6, "verification_passes": 2},
}


# --------------------------------------------------------------------------- #
# Capability tiers: Opus/Sonnet/Haiku analogues. Same underlying toy
# architecture (to keep this self-contained), differing only in configured
# size/cost/latency -- illustrating the point-in-time engineering claim that
# tiering is a capability/cost/latency ladder, not three unrelated products.
# --------------------------------------------------------------------------- #


@dataclass
class Tier:
    name: str
    relative_cost_per_token: float   # illustrative, unitless
    relative_latency: float          # illustrative, unitless
    max_effort: Effort               # ceiling on effort this tier will honor
    model: ToyLM


def build_tier(name: str, d_model: int, n_layers: int, cost: float, latency: float, max_effort: Effort) -> Tier:
    cfg = ModelConfig(d_model=d_model, n_layers=n_layers, n_heads=4, d_ff=d_model * 4, max_seq_len=2048)
    model = ToyLM(cfg)
    model.eval()
    return Tier(name=name, relative_cost_per_token=cost, relative_latency=latency, max_effort=max_effort, model=model)


def build_default_tiers() -> Dict[str, Tier]:
    return {
        "opus":   build_tier("opus",   d_model=128, n_layers=4, cost=15.0, latency=3.0, max_effort=Effort.HIGH),
        "sonnet": build_tier("sonnet", d_model=64,  n_layers=2, cost=3.0,  latency=1.0, max_effort=Effort.MEDIUM),
        "haiku":  build_tier("haiku",  d_model=32,  n_layers=1, cost=0.8,  latency=0.3, max_effort=Effort.LOW),
    }


# --------------------------------------------------------------------------- #
# Router: picks a tier from a declared difficulty signal, then runs a
# capability-budgeted episode on that tier honoring the requested effort
# (capped by what the chosen tier is willing to support).
# --------------------------------------------------------------------------- #


@dataclass
class RoutingDecision:
    tier_name: str
    effective_effort: Effort
    reason: str


class TierRouter:
    """
    Toy router mapping a declared task-difficulty label to a tier + effort,
    modeling the production question the 007 markdown raises: not every
    request should go to the flagship, and the flagship's premium should be
    reserved for requests where difficulty/long-horizon reliability actually
    dominate the decision.
    """

    DIFFICULTY_TO_TIER = {
        "trivial": ("haiku", Effort.LOW),
        "moderate": ("sonnet", Effort.MEDIUM),
        "hard_long_horizon": ("opus", Effort.HIGH),
    }

    def __init__(self, tiers: Dict[str, Tier]):
        self.tiers = tiers

    def route(self, difficulty: str, requested_effort: Effort = None) -> RoutingDecision:
        if difficulty not in self.DIFFICULTY_TO_TIER:
            raise ValueError(f"unknown difficulty label: {difficulty}")

        tier_name, default_effort = self.DIFFICULTY_TO_TIER[difficulty]
        tier = self.tiers[tier_name]

        effort = requested_effort or default_effort
        # A tier will not honor an effort level above its configured ceiling
        # (e.g., asking the haiku-analogue tier for HIGH effort does not make
        # it Opus; it is capped at what that tier supports).
        order = [Effort.LOW, Effort.MEDIUM, Effort.HIGH]
        if order.index(effort) > order.index(tier.max_effort):
            effective = tier.max_effort
            reason = (
                f"requested effort '{effort.value}' exceeds tier '{tier_name}' ceiling; "
                f"capped to '{effective.value}'"
            )
        else:
            effective = effort
            reason = f"difficulty='{difficulty}' routed to tier='{tier_name}' at effort='{effective.value}'"

        return RoutingDecision(tier_name=tier_name, effective_effort=effective, reason=reason)


# --------------------------------------------------------------------------- #
# Capability-budgeted episode runner: extends file 005's single thinking
# budget into a multi-sub-cycle episode whose length and per-cycle thinking
# depth are both governed by the effective effort level.
# --------------------------------------------------------------------------- #


def run_capability_budgeted_episode(
    tier: Tier, effort: Effort, prompt_ids: List[int], device: str = "cpu"
) -> Dict:
    profile = EFFORT_PROFILE[effort]
    cfg = tier.model.cfg
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    total_thinking_tokens = 0
    subcycles_run = 0

    for cycle in range(profile["max_subcycles"]):
        for _ in range(profile["thinking_tokens"]):
            nxt = tier.model.next_token(idx, temperature=0.9)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
            if idx.shape[1] >= cfg.max_seq_len - 1:
                break
            total_thinking_tokens += 1
        subcycles_run += 1
        if idx.shape[1] >= cfg.max_seq_len - 1:
            break

    verification_notes = []
    for v in range(profile["verification_passes"]):
        _ = tier.model.next_token(idx, temperature=0.5)  # toy "self-check" pass
        verification_notes.append(f"verification pass {v + 1}: self-consistency check run")

    estimated_cost = total_thinking_tokens * tier.relative_cost_per_token
    estimated_latency = subcycles_run * tier.relative_latency

    return {
        "tier": tier.name,
        "effort": effort.value,
        "subcycles_run": subcycles_run,
        "thinking_tokens_used": total_thinking_tokens,
        "verification_passes": len(verification_notes),
        "estimated_relative_cost": round(estimated_cost, 2),
        "estimated_relative_latency": round(estimated_latency, 2),
    }


if __name__ == "__main__":
    torch.manual_seed(0)

    tiers = build_default_tiers()
    router = TierRouter(tiers)
    prompt = [1, 10, 20, 30]  # toy tokenized "request"

    print("=== Tiered routing + capability-budgeted episodes (Opus 4.x-style positioning) ===\n")

    requests = [
        ("trivial", "Classify the sentiment of a short sentence."),
        ("moderate", "Refactor a medium-sized function with a couple of edge cases."),
        ("hard_long_horizon", "Diagnose and fix a flaky failure across a large multi-file codebase."),
    ]

    for difficulty, description in requests:
        decision = router.route(difficulty)
        tier = tiers[decision.tier_name]
        stats = run_capability_budgeted_episode(tier, decision.effective_effort, prompt)

        print(f"Request: {description}")
        print(f"  Routing: {decision.reason}")
        print(f"  Episode stats: {stats}\n")

    print("--- Demonstrating effort-ceiling capping on a low-tier route ---")
    forced = router.route("trivial", requested_effort=Effort.HIGH)
    print(f"  {forced.reason}")
    stats = run_capability_budgeted_episode(tiers[forced.tier_name], forced.effective_effort, prompt)
    print(f"  Episode stats: {stats}")

    print(
        "\nNote: cost/latency figures are illustrative relative units on a toy "
        "model, not real pricing or benchmark data. The point being modeled is "
        "structural: effort is a dial over an entire episode (thinking depth, "
        "sub-cycle count, and verification passes), and routing decides which "
        "capability tier is even allowed to spend that effort, consistent with "
        "Opus's disclosed positioning as the tier reserved for the hardest, "
        "longest-horizon requests rather than the default for all traffic."
    )
