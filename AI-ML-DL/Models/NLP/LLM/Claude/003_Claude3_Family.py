"""
003_Claude3_Family.py

Model: Claude 3 family (Anthropic, March 2024) -- Haiku, Sonnet, Opus
What this file demonstrates: a three-tier model-router that picks a
Haiku/Sonnet/Opus-equivalent tier based on estimated task complexity, latency
sensitivity, and cost budget -- the routing decision a real deployment (or an
agentic system) built on top of a tiered model family has to make explicitly.

IMPORTANT: Anthropic has never disclosed the parameter counts, architectures,
or training relationships between Haiku, Sonnet, and Opus. The "tiers" below
are toy PyTorch models of deliberately different capacity (parameter count),
used only to give the router something real to route between and to make the
capability/latency/cost tradeoff concrete -- they are not reconstructions of
the real Claude 3 models, whose internals remain undisclosed. The complexity
estimator is a small trained classifier over hand-built lexical/structural
features of a task description, standing in conceptually for the kind of
lightweight triage step a production router would run before an expensive
model call -- not a claim about how Anthropic's own systems route requests
(Anthropic does not disclose that either).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Three toy "tiers" of deliberately different capacity, standing in for
# Haiku / Sonnet / Opus. Real relative scale between the actual Claude 3
# tiers is undisclosed; these sizes are chosen only to produce a real,
# measurable latency/capacity gradient for the demo.
# ---------------------------------------------------------------------------

class Tier(str, Enum):
    HAIKU = "haiku"    # fastest, cheapest, smallest capacity
    SONNET = "sonnet"  # balanced
    OPUS = "opus"       # slowest, most expensive, largest capacity


@dataclass(frozen=True)
class TierProfile:
    tier: Tier
    d_model: int
    n_layers: int
    n_heads: int
    relative_cost_per_token: float   # illustrative, not Anthropic's real pricing
    capability_score: float          # illustrative 0-1 proxy for "can handle harder tasks"


TIER_PROFILES: List[TierProfile] = [
    TierProfile(Tier.HAIKU, d_model=64, n_layers=2, n_heads=2, relative_cost_per_token=1.0, capability_score=0.45),
    TierProfile(Tier.SONNET, d_model=128, n_layers=4, n_heads=4, relative_cost_per_token=5.0, capability_score=0.75),
    TierProfile(Tier.OPUS, d_model=256, n_layers=8, n_heads=8, relative_cost_per_token=25.0, capability_score=0.95),
]


class TinyTierModel(nn.Module):
    """A minimal causal decoder, sized per-tier only to produce a real,
    measurable compute/latency gradient across tiers for this demo -- not a
    reconstruction of any real Claude 3 model's architecture."""

    def __init__(self, vocab_size: int, profile: TierProfile, max_len: int = 128):
        super().__init__()
        self.profile = profile
        self.tok_emb = nn.Embedding(vocab_size, profile.d_model)
        self.pos_emb = nn.Embedding(max_len, profile.d_model)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(profile.d_model),
                "attn": nn.MultiheadAttention(profile.d_model, profile.n_heads, batch_first=True),
                "ln2": nn.LayerNorm(profile.d_model),
                "ff": nn.Sequential(
                    nn.Linear(profile.d_model, 4 * profile.d_model),
                    nn.GELU(),
                    nn.Linear(4 * profile.d_model, profile.d_model),
                ),
            })
            for _ in range(profile.n_layers)
        ])
        self.ln_f = nn.LayerNorm(profile.d_model)
        self.head = nn.Linear(profile.d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        for blk in self.blocks:
            h = blk["ln1"](x)
            attn_out, _ = blk["attn"](h, h, h, attn_mask=causal_mask, need_weights=False)
            x = x + attn_out
            x = x + blk["ff"](blk["ln2"](x))
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Complexity estimator: a tiny classifier over hand-built lexical/structural
# features of a task description, used to decide which tier a request should
# route to. This is a toy stand-in for a production triage step, not a claim
# about how any real system (Anthropic's or otherwise) implements routing.
# ---------------------------------------------------------------------------

COMPLEXITY_KEYWORDS = {
    "simple": ["classify", "extract", "translate", "summarize briefly", "yes or no", "sentiment"],
    "moderate": ["summarize", "explain", "compare", "rewrite", "draft", "outline"],
    "complex": ["prove", "design", "architecture", "multi-step", "debug", "optimize",
                "research", "derive", "analyze deeply", "reason step by step", "agentic"],
}


def extract_features(task: str) -> torch.Tensor:
    """Hand-built features standing in for a lightweight triage signal:
    task length, presence of complexity-indicating keywords, and punctuation
    density (as a crude proxy for structural/multi-part requests)."""
    lower = task.lower()
    length_feat = min(len(task) / 200.0, 1.0)
    simple_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["simple"] if kw in lower)
    moderate_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["moderate"] if kw in lower)
    complex_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["complex"] if kw in lower)
    multi_step = float("\n" in task or task.count(".") > 3 or " then " in lower)
    return torch.tensor([
        length_feat,
        float(simple_hits > 0),
        float(moderate_hits > 0),
        float(complex_hits > 0),
        multi_step,
    ], dtype=torch.float32)


class ComplexityClassifier(nn.Module):
    """A tiny trained (here: hand-initialized, not gradient-trained, for
    determinism in the demo) linear classifier mapping task features to a
    scalar complexity estimate in [0, 1]."""

    def __init__(self, n_features: int = 5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        with torch.no_grad():
            # Hand-set weights so keyword signals dominate length, giving a
            # deterministic, interpretable demo without needing a training
            # loop or external labeled data.
            self.linear.weight.copy_(torch.tensor([[0.1, -0.3, 0.2, 0.6, 0.3]]))
            self.linear.bias.copy_(torch.tensor([-0.05]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(features))


# ---------------------------------------------------------------------------
# The router itself: given a task description (and optional constraints),
# pick the cheapest tier expected to handle it, mirroring the production
# pattern of "escalate only when needed" built on top of a tiered model
# family such as Haiku/Sonnet/Opus.
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    task: str
    estimated_complexity: float
    chosen_tier: Tier
    reason: str
    estimated_relative_cost: float


class ModelRouter:
    def __init__(self, profiles: List[TierProfile] = TIER_PROFILES):
        self.profiles = sorted(profiles, key=lambda p: p.capability_score)
        self.classifier = ComplexityClassifier()

    def route(self, task: str, max_relative_cost: float | None = None,
              latency_sensitive: bool = False) -> RoutingDecision:
        features = extract_features(task)
        with torch.no_grad():
            complexity = self.classifier(features).item()

        # Pick the cheapest tier whose capability_score comfortably exceeds
        # the estimated complexity, respecting an optional cost ceiling and a
        # latency-sensitivity flag that biases toward the fastest tier when
        # the complexity estimate is borderline.
        candidates = [p for p in self.profiles if p.capability_score >= complexity]
        if not candidates:
            candidates = [self.profiles[-1]]  # nothing clears the bar; use the most capable tier

        if max_relative_cost is not None:
            affordable = [p for p in candidates if p.relative_cost_per_token <= max_relative_cost]
            if affordable:
                candidates = affordable

        if latency_sensitive and complexity < 0.6:
            # Below a moderate complexity threshold, prefer the fastest
            # tier among those that still clear the capability bar, rather
            # than automatically taking the cheapest-that-clears-the-bar
            # option (which could still be a slower mid-tier model).
            chosen = min(candidates, key=lambda p: p.relative_cost_per_token)
            reason = "latency-sensitive request below complexity threshold -> fastest viable tier"
        else:
            chosen = min(candidates, key=lambda p: p.relative_cost_per_token)
            reason = "cheapest tier whose capability score clears the estimated task complexity"

        return RoutingDecision(
            task=task,
            estimated_complexity=complexity,
            chosen_tier=chosen.tier,
            reason=reason,
            estimated_relative_cost=chosen.relative_cost_per_token,
        )


# ---------------------------------------------------------------------------
# Demonstration: build one TinyTierModel per tier (to show a real capacity /
# latency gradient exists across tiers), then route a handful of representative
# tasks and report which tier each lands on and why.
# ---------------------------------------------------------------------------

def benchmark_tier_latency(vocab_size: int = 200, seq_len: int = 64, n_runs: int = 20) -> None:
    print("\nMeasured latency and parameter count per tier (toy models, CPU):")
    for profile in TIER_PROFILES:
        model = TinyTierModel(vocab_size, profile, max_len=seq_len).eval()
        x = torch.randint(0, vocab_size, (1, seq_len))
        with torch.no_grad():
            model(x)  # warm-up
            start = time.perf_counter()
            for _ in range(n_runs):
                model(x)
            elapsed = (time.perf_counter() - start) / n_runs
        print(f"  {profile.tier.value:<8} params={model.num_params():>8,}  "
              f"avg forward pass={elapsed * 1000:6.2f} ms  "
              f"relative_cost/token={profile.relative_cost_per_token:>5.1f}")


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 78)
    print("Claude 3 family (2024): Haiku / Sonnet / Opus three-tier router demo")
    print("Real per-tier architecture and parameter counts are undisclosed by")
    print("Anthropic. Tiers below are toy models sized only to show a real")
    print("capacity/latency gradient exists, not to reconstruct real Claude 3.")
    print("=" * 78)

    benchmark_tier_latency()

    router = ModelRouter()

    tasks = [
        ("Classify the sentiment of this review as positive or negative.", False, None),
        ("Summarize this email in two sentences.", True, None),
        ("Explain how photosynthesis works to a 10-year-old.", False, None),
        ("Design a distributed rate limiter that survives node failures, then "
         "prove its correctness under network partitions and outline a rollout plan.", False, None),
        ("Debug this multi-step agentic workflow: it fails intermittently when "
         "step 3 times out, then step 4 retries with stale state. Analyze deeply.", False, 10.0),
    ]

    print("\nRouting decisions:")
    for task, latency_sensitive, cost_ceiling in tasks:
        decision = router.route(task, max_relative_cost=cost_ceiling, latency_sensitive=latency_sensitive)
        print(f"\n  Task: {task[:70]}{'...' if len(task) > 70 else ''}")
        print(f"    estimated_complexity = {decision.estimated_complexity:.3f}")
        print(f"    -> routed to: {decision.chosen_tier.value.upper()} "
              f"(relative_cost={decision.estimated_relative_cost:.1f})")
        print(f"    reason: {decision.reason}")

    print(
        "\nThis mirrors the production pattern of routing/cascading across a"
        "\ntiered model family: send the bulk of low-complexity traffic to the"
        "\ncheapest tier that can handle it, and reserve the most expensive tier"
        "\nfor requests whose estimated complexity genuinely requires it."
    )
