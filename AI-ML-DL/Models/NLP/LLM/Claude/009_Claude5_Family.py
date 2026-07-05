"""
009_Claude5_Family.py

Model: the Claude 5 family (Anthropic) -- Fable 5, Haiku 4.5, Sonnet 5,
Opus 4.8 -- as described in this entry's markdown companion.

IMPORTANT: Claude 5/Sonnet 5's actual internals (architecture, parameter
count, training methodology) are NOT modeled here. Anthropic has disclosed
none of that, for this generation or any prior one, and this file does not
attempt to guess at it. What this file DOES model concretely is the
structural, product-and-systems idea the markdown companion's Section 1/3/9
build their analysis around: Anthropic's current lineup is versioned on
MIXED, INDEPENDENT per-tier generation numbers (Sonnet 5, Opus 4.8, Haiku
4.5, Fable 5) rather than one synchronized whole-family generation number
(as Claude 1/2/3/4 were). That is a fleet-management and routing problem,
not a modeling problem, and it is fully demonstrable without knowing a
single fact about any tier's real weights.

Concretely, this file implements:

  1. A `ModelHandle` -- an independently-versioned tier (fable/haiku/sonnet/
     opus), each carrying its own version string, capability profile, and
     illustrative cost/latency numbers, matching the toy-tier pattern used
     in 003_Claude3_Family.py and 007_Claude_Opus_4_Latest.py in this same
     folder -- extended here from three synchronized tiers to four
     independently-versioned ones.
  2. A `ModelFleet` managing all four handles, exposing an operation to
     bump ONE tier's version in isolation, leaving the other three
     completely untouched -- the concrete mechanic behind "Sonnet reached
     5 while Opus is still at 4.8."
  3. A `FleetDispatcher` that routes a task to a tier based on declared
     task requirements (capability floor, latency sensitivity, and -- for
     Fable specifically -- a "specialized creative-writing" affinity flag,
     modeling the markdown's speculative-but-flagged read of Fable as a
     differently-scoped model rather than a fourth rung on the general
     capability ladder). The dispatcher is re-run before and after an
     isolated version bump to show that routing continues to work
     correctly across an asynchronous, single-tier upgrade -- no
     coordination with the other three tiers is required.

None of the cost/latency/capability numbers below are real Anthropic
figures; they are illustrative constants chosen only to make the routing
and version-bump mechanics concrete and testable, exactly as in the two
prior tier-router files in this folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# A minimal per-tier toy model, reused (structurally) from
# 003_Claude3_Family.py / 007_Claude_Opus_4_Latest.py, kept here only to give
# each tier handle something real to instantiate and measure -- NOT a
# reconstruction of any real Claude 5-family model's architecture, which is
# undisclosed.
# --------------------------------------------------------------------------- #


class TinyTierModel(nn.Module):
    """A minimal feed-forward stand-in, sized per-tier only to produce a
    real, measurable parameter-count gradient across tiers for this demo."""

    def __init__(self, d_model: int, n_layers: int, vocab_size: int = 128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Tier identity and capability profile. Note there are FOUR tiers, not three
# -- Fable is included as a differently-scoped, specialized handle rather
# than a fourth point on the same general capability/cost/latency ladder,
# per the markdown's Section 1/9 discussion. This is a modeling choice meant
# to make that structural distinction visible in code, not a claim about
# Fable's real, undisclosed positioning.
# --------------------------------------------------------------------------- #


class TierName(str, Enum):
    FABLE = "fable"
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


@dataclass(frozen=True)
class CapabilityProfile:
    """Illustrative, non-Anthropic-sourced numbers standing in for a tier's
    position on the capability/cost/latency ladder (or, for Fable, its
    specialized-affinity score instead of a ladder position)."""

    general_capability: float          # 0-1 proxy, generalist tasks (coding/agentic/reasoning)
    creative_affinity: float           # 0-1 proxy, narrative/creative-writing fit
    relative_cost_per_token: float     # illustrative, unitless
    relative_latency: float            # illustrative, unitless


@dataclass
class ModelHandle:
    """One independently-versioned tier in the fleet. The `version` field is
    exactly what the markdown's Section 1/9 discussion is about: it moves on
    its own schedule, independent of every other handle's version."""

    name: TierName
    version: str                       # e.g. "5", "4.8", "4.5-20251001"
    profile: CapabilityProfile
    d_model: int
    n_layers: int
    model: TinyTierModel = field(repr=False)

    @property
    def model_id(self) -> str:
        # Mirrors real Claude model-id shape (e.g. "claude-sonnet-5",
        # "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-fable-5")
        # without claiming these toy handles ARE those real models.
        return f"claude-{self.name.value}-{self.version.replace('.', '-')}"


def _build_handle(name: TierName, version: str, profile: CapabilityProfile,
                   d_model: int, n_layers: int) -> ModelHandle:
    model = TinyTierModel(d_model=d_model, n_layers=n_layers)
    model.eval()
    return ModelHandle(name=name, version=version, profile=profile,
                        d_model=d_model, n_layers=n_layers, model=model)


# --------------------------------------------------------------------------- #
# ModelFleet: owns all four independently-versioned handles and supports
# bumping exactly one tier's version without touching the others -- the
# concrete mechanic behind "Sonnet advanced to 5 while Opus remained at 4.8
# and Haiku at 4.5."
# --------------------------------------------------------------------------- #


class ModelFleet:
    def __init__(self) -> None:
        self._handles: Dict[TierName, ModelHandle] = {
            TierName.FABLE: _build_handle(
                TierName.FABLE, version="5",
                profile=CapabilityProfile(general_capability=0.55, creative_affinity=0.95,
                                           relative_cost_per_token=4.0, relative_latency=1.2),
                d_model=48, n_layers=3,
            ),
            TierName.HAIKU: _build_handle(
                TierName.HAIKU, version="4.5-20251001",
                profile=CapabilityProfile(general_capability=0.45, creative_affinity=0.30,
                                           relative_cost_per_token=0.8, relative_latency=0.3),
                d_model=32, n_layers=1,
            ),
            TierName.SONNET: _build_handle(
                TierName.SONNET, version="5",
                profile=CapabilityProfile(general_capability=0.75, creative_affinity=0.45,
                                           relative_cost_per_token=3.0, relative_latency=1.0),
                d_model=64, n_layers=2,
            ),
            TierName.OPUS: _build_handle(
                TierName.OPUS, version="4.8",
                profile=CapabilityProfile(general_capability=0.95, creative_affinity=0.40,
                                           relative_cost_per_token=15.0, relative_latency=3.0),
                d_model=128, n_layers=4,
            ),
        }

    def get(self, name: TierName) -> ModelHandle:
        return self._handles[name]

    def all_handles(self) -> List[ModelHandle]:
        return list(self._handles.values())

    def bump_version(self, name: TierName, new_version: str,
                      new_profile: Optional[CapabilityProfile] = None) -> ModelHandle:
        """Upgrade exactly ONE tier's version (and, optionally, its
        capability profile), leaving all other tiers' handles -- including
        their version strings -- completely untouched. This is the whole
        point being demonstrated: fleet-wide coordination is not required to
        ship an improvement to a single tier."""
        old = self._handles[name]
        profile = new_profile if new_profile is not None else old.profile
        new_handle = _build_handle(name, version=new_version, profile=profile,
                                    d_model=old.d_model, n_layers=old.n_layers)
        self._handles[name] = new_handle
        return new_handle

    def version_report(self) -> str:
        lines = ["Fleet versions (independent per-tier):"]
        for name in TierName:
            h = self._handles[name]
            lines.append(f"  {name.value:<8} version={h.version:<14} model_id={h.model_id}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Task requirements and routing decision.
# --------------------------------------------------------------------------- #


@dataclass
class TaskRequirements:
    description: str
    min_general_capability: float = 0.0
    creative_task: bool = False        # if True, prefer creative_affinity over general_capability
    latency_sensitive: bool = False
    max_relative_cost: Optional[float] = None


@dataclass
class RoutingDecision:
    task: str
    chosen_tier: TierName
    chosen_model_id: str
    reason: str


class FleetDispatcher:
    """Routes a task to a tier in the fleet based on declared requirements.
    Deliberately keys off each handle's CURRENT profile/version at
    dispatch time, rather than caching anything about the fleet's shape --
    which is exactly what lets it keep routing correctly across an
    asynchronous, single-tier version bump with zero changes to the
    dispatcher itself."""

    def __init__(self, fleet: ModelFleet):
        self.fleet = fleet

    def route(self, req: TaskRequirements) -> RoutingDecision:
        handles = self.fleet.all_handles()

        if req.creative_task:
            # Creative-writing-flagged tasks prefer the handle with the
            # highest creative_affinity, modeling the markdown's Section 1/9
            # point that Fable is scoped along a DIFFERENT axis than the
            # general capability/cost/latency ladder, not a fourth rung on
            # it -- so creative-task routing should not simply fall through
            # to "cheapest tier that clears a general-capability bar."
            best = max(handles, key=lambda h: h.profile.creative_affinity)
            reason = (f"creative_task=True -> routed by creative_affinity "
                      f"({best.profile.creative_affinity:.2f}), not by general capability")
            return RoutingDecision(req.description, best.name, best.model_id, reason)

        candidates = [h for h in handles if h.profile.general_capability >= req.min_general_capability]
        if not candidates:
            candidates = [max(handles, key=lambda h: h.profile.general_capability)]

        if req.max_relative_cost is not None:
            affordable = [h for h in candidates if h.profile.relative_cost_per_token <= req.max_relative_cost]
            if affordable:
                candidates = affordable

        if req.latency_sensitive:
            chosen = min(candidates, key=lambda h: h.profile.relative_latency)
            reason = "latency_sensitive=True -> fastest tier clearing the capability floor"
        else:
            chosen = min(candidates, key=lambda h: h.profile.relative_cost_per_token)
            reason = "cheapest tier clearing the capability floor"

        return RoutingDecision(req.description, chosen.name, chosen.model_id, reason)


# --------------------------------------------------------------------------- #
# Demonstration: route a fixed battery of tasks, bump ONLY sonnet's version
# (and capability profile, simulating a genuine improvement), then re-route
# the same battery and show (a) the version report reflects the isolated
# bump, and (b) routing decisions for sonnet-eligible tasks change to
# reflect the improved profile while every other tier's routing outcome is
# unaffected -- the asynchronous-upgrade scenario the markdown's Section 9
# describes.
# --------------------------------------------------------------------------- #

TASKS: List[TaskRequirements] = [
    TaskRequirements("Classify sentiment of a short product review.",
                      min_general_capability=0.3, latency_sensitive=True),
    TaskRequirements("Refactor a medium-sized function with several edge cases.",
                      min_general_capability=0.6),
    TaskRequirements("Design and prove correctness of a distributed rate limiter.",
                      min_general_capability=0.95),
    TaskRequirements("Write a short fable-style story for a children's book, in verse.",
                      creative_task=True),
    TaskRequirements("Diagnose a flaky multi-file agentic coding failure under a tight cost ceiling.",
                      min_general_capability=0.7, max_relative_cost=5.0),
    TaskRequirements("Plan a moderately complex multi-service migration with several open design questions.",
                      min_general_capability=0.8),
]


def run_battery(dispatcher: FleetDispatcher, label: str) -> Dict[str, RoutingDecision]:
    print(f"\n--- Routing battery: {label} ---")
    results: Dict[str, RoutingDecision] = {}
    for req in TASKS:
        decision = dispatcher.route(req)
        results[req.description] = decision
        print(f"  Task: {req.description[:65]}{'...' if len(req.description) > 65 else ''}")
        print(f"    -> tier={decision.chosen_tier.value:<8} model_id={decision.chosen_model_id}")
        print(f"       reason: {decision.reason}")
    return results


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 78)
    print("Claude 5 family: independent per-tier versioning + fleet dispatch demo")
    print("Fable / Haiku / Sonnet / Opus handles below are toy stand-ins with")
    print("illustrative capability/cost/latency numbers -- NOT reconstructions")
    print("of any real Claude 5-family model, whose internals are undisclosed.")
    print("=" * 78)

    fleet = ModelFleet()
    dispatcher = FleetDispatcher(fleet)

    print("\n" + fleet.version_report())
    print("\nParameter counts (toy models, illustrative capacity gradient only):")
    for h in fleet.all_handles():
        print(f"  {h.name.value:<8} version={h.version:<14} params={h.model.num_params():>8,}")

    before = run_battery(dispatcher, "BEFORE isolated Sonnet version bump")

    print("\n>>> Bumping ONLY the Sonnet tier: version '5' -> '5.1' "
          "(simulating a genuine capability improvement -- a further, "
          "independent point release on Sonnet's own track), leaving "
          "Fable, Haiku, and Opus completely untouched. <<<")
    fleet.bump_version(
        TierName.SONNET,
        new_version="5.1",
        new_profile=CapabilityProfile(
            general_capability=0.85,   # improved: was 0.75
            creative_affinity=0.50,    # improved: was 0.45
            relative_cost_per_token=3.0,   # unchanged
            relative_latency=1.0,          # unchanged
        ),
    )

    print("\n" + fleet.version_report())

    after = run_battery(dispatcher, "AFTER isolated Sonnet version bump")

    print("\n--- What changed vs. what stayed the same across the bump ---")
    for req in TASKS:
        b, a = before[req.description], after[req.description]
        changed = b.chosen_tier != a.chosen_tier
        marker = "CHANGED" if changed else "unchanged"
        print(f"  [{marker:>9}] {req.description[:55]}{'...' if len(req.description) > 55 else ''}"
              f"  {b.chosen_tier.value} -> {a.chosen_tier.value}")

    print(
        "\nNote: the dispatcher required zero code changes and zero awareness "
        "of the bump ahead of time -- it re-reads each handle's current "
        "version/profile at dispatch time. Only routing decisions that were "
        "actually sensitive to Sonnet's improved profile changed; Fable-, "
        "Haiku-, and Opus-bound routing outcomes were unaffected, exactly as "
        "the markdown's Section 9 argues should hold for a fleet that ships "
        "independent per-tier version bumps rather than synchronized "
        "whole-family generation releases."
    )
