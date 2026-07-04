# CI/CD and Versioning for LLM Applications

## Why This Isn't Just "CI/CD, But With Prompts"

Traditional CI/CD is built on an assumption that mostly holds for conventional software: given the same input and the same code, the output is deterministic, so a test either passes or fails, and a passing test suite is strong evidence the system behaves correctly. LLM applications break that assumption at the foundation. The same prompt sent twice to the same model can produce two different (both individually reasonable) outputs, a model provider can silently update a model version behind a fixed API name and shift behavior without you changing a single line of your own code, and "correctness" for a generative task is often a graded, subjective quality rather than a binary pass/fail. None of this means rigor is impossible — it means the rigor has to be built differently, around statistical evaluation over a representative sample rather than exact-match assertions, and around treating prompts, model versions, and evaluation datasets as versioned artifacts with the same seriousness as application code.

The practical consequence is that an LLM application's CI/CD pipeline has an extra dimension that a typical backend service's pipeline doesn't need: it has to answer not just "does the code compile and pass unit tests" but "did this change make the system's actual output quality better, worse, or unchanged, on a scale where 'worse' might mean 2% more hallucinations rather than a crash." Building a pipeline that can answer that question reliably, cheaply, and on every pull request is the core engineering problem this chapter is about.

## Prompts Are Code, and Need the Same Discipline

The first mental shift is to stop treating prompts as configuration strings scattered through application code and start treating them as versioned artifacts with their own lifecycle — reviewed in pull requests, diffed, rolled back, and tied to specific evaluation results, exactly like a model-serving config or a feature flag would be. A prompt change is a behavior change to the system just as much as a code change is, and it deserves the same change-management rigor: it should never go to production without having been evaluated, and it should always be possible to answer "what exact prompt version produced this output" after the fact.

```python
import hashlib
import json
from datetime import datetime

class PromptRegistry:
    """Stores versioned prompts with metadata, backed by any persistent store
    (a database table, a git-backed file store, or a dedicated prompt-management
    platform). The key invariant: every prompt version is immutable once created,
    and every generation call records which version produced it."""

    def __init__(self, storage):
        self.storage = storage

    def save_version(self, name, template, metadata=None):
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:12]
        existing = self.storage.get_latest(name)
        if existing and existing["content_hash"] == content_hash:
            return existing["version"]  # no-op if content is identical to latest

        version = (existing["version"] + 1) if existing else 1
        record = {
            "name": name,
            "version": version,
            "template": template,
            "content_hash": content_hash,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "status": "draft",  # draft -> staged -> active -> retired
        }
        self.storage.put(record)
        return version

    def promote(self, name, version, status):
        record = self.storage.get(name, version)
        record["status"] = status
        self.storage.put(record)

    def get_active(self, name):
        return self.storage.query_one({"name": name, "status": "active"})

    def diff(self, name, v1, v2):
        a = self.storage.get(name, v1)["template"]
        b = self.storage.get(name, v2)["template"]
        import difflib
        return "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=""))
```

The status field matters more than it looks: modeling a prompt's lifecycle explicitly as draft/staged/active/retired (rather than a single boolean "is this live") is what lets a staged rollout (discussed below) exist as a first-class state rather than an ad hoc branch in application code. And recording the content hash alongside every generation call — logging exactly which prompt version, which model version, and which parameters produced a given output — is the single most valuable piece of observability for debugging a quality regression after the fact, since "the model got worse" is meaningless without knowing precisely what combination of prompt, model, and params was active when the bad output was produced.

## Golden Evaluation Sets: The Regression Test Suite for Non-Determinism

If unit tests are the regression net for traditional code, the golden evaluation set is the regression net for prompts and agent behavior. It's a curated, versioned collection of representative input cases — ideally sourced from real production traffic, augmented with known edge cases and adversarial examples — each paired with either a reference answer, a rubric, or a set of properties the output must satisfy. The critical design principle is that a golden set is not a single all-purpose collection; it should be segmented by capability or failure mode (factual accuracy cases, tone/format compliance cases, tool-calling correctness cases, safety/refusal cases, previously-reported bug cases) so that a regression in one narrow capability doesn't get diluted into a single aggregate score that hides it.

```python
from dataclasses import dataclass, field

@dataclass
class EvalCase:
    id: str
    input: dict
    category: str                 # e.g. "factual_accuracy", "tool_calling", "regression_bug_142"
    reference: str = None         # exact/near-exact expected answer, if applicable
    rubric: str = None            # criteria for an LLM-judge to score against, if no single right answer
    must_contain: list = field(default_factory=list)
    must_not_contain: list = field(default_factory=list)


def score_case(case: EvalCase, actual_output: str, judge_fn) -> dict:
    checks = {}

    if case.must_contain:
        checks["contains_required"] = all(s.lower() in actual_output.lower() for s in case.must_contain)
    if case.must_not_contain:
        checks["avoids_forbidden"] = all(s.lower() not in actual_output.lower() for s in case.must_not_contain)
    if case.rubric:
        judge_result = judge_fn(case.input, actual_output, case.rubric)
        checks["judge_score"] = judge_result["score"]        # e.g. 1-5 scale
        checks["judge_rationale"] = judge_result["rationale"]

    checks["passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks
```

Using an LLM as a judge for the rubric-scored cases is standard practice, but it needs the same care as any other measurement instrument: the judge should be a separate, ideally stronger, model than the one being evaluated (to avoid a model favorably grading its own outputs), the rubric needs to be specific and behaviorally grounded rather than vague ("does the response correctly cite the retrieved document's actual numbers, not just plausible-sounding numbers" rather than "is the response good"), and the judge's scores should themselves be periodically validated against human ratings on a sample, since an uncalibrated judge model gives you false confidence rather than real signal.

## The Regression Test Harness

With a golden set and a scoring function in hand, the CI equivalent of "run the test suite" is: run every case in the golden set (or a representative, cost-bounded sample of it) through the candidate prompt/model, score every result, aggregate by category, and compare against the last known-good baseline — failing the build if any category regresses beyond a tolerance band, not just if the aggregate score drops.

```python
import statistics

def run_regression_suite(cases, candidate_generate_fn, judge_fn, baseline_scores, tolerance=0.03):
    results_by_category = {}
    for case in cases:
        output = candidate_generate_fn(case.input)
        result = score_case(case, output, judge_fn)
        results_by_category.setdefault(case.category, []).append(result)

    report = {}
    regressions = []
    for category, results in results_by_category.items():
        pass_rate = sum(r["passed"] for r in results) / len(results)
        report[category] = pass_rate

        baseline = baseline_scores.get(category)
        if baseline is not None and pass_rate < baseline - tolerance:
            regressions.append({
                "category": category,
                "baseline": baseline,
                "current": pass_rate,
                "delta": pass_rate - baseline,
            })

    return {
        "report": report,
        "regressions": regressions,
        "passed": len(regressions) == 0,
    }
```

A few choices in this harness are load-bearing rather than incidental. Comparing per-category pass rates against a tolerance band, rather than requiring every single case to pass, is what makes the suite usable at all given real model non-determinism — a single flaky case failing once shouldn't block every deploy, but a category's pass rate dropping from 94% to 78% is a real signal worth blocking on. Running evaluation at a fixed low (often zero) temperature, or running each case multiple times and aggregating, reduces (but doesn't eliminate) the noise floor from sampling variance, which matters because a regression test that's noisier than the actual effect size you're trying to detect is worse than useless — it produces both false alarms and false confidence. And storing baseline scores as a versioned artifact alongside the golden set itself, updated deliberately (as a reviewed change, not an automatic overwrite) whenever a genuine improvement is accepted, is what keeps the baseline meaningful over time instead of drifting to match whatever the current build happens to produce.

```python
def build_ci_gate(regression_result, min_overall_pass_rate=0.90):
    if not regression_result["passed"]:
        categories = ", ".join(r["category"] for r in regression_result["regressions"])
        raise SystemExit(f"BLOCKED: regression detected in categories: {categories}")

    overall = statistics.mean(regression_result["report"].values())
    if overall < min_overall_pass_rate:
        raise SystemExit(f"BLOCKED: overall pass rate {overall:.2%} below gate of {min_overall_pass_rate:.0%}")

    print(f"PASSED: overall pass rate {overall:.2%}, no category regressions")
```

## Staged Rollouts: Deploying When You Can't Fully Trust a Test Suite

Even a well-built golden set is a sample, not a guarantee — it can't cover the full diversity of real production traffic, and subtle regressions (a slightly worse tone, a slightly higher rate of unnecessary tool calls) often only surface at real-traffic scale. This is why LLM application deploys lean more heavily on staged, traffic-gated rollouts than a typical backend deploy does: the test suite earns you the right to expose a change to a small slice of real traffic, and production behavior on that slice earns the right to expand further.

A typical staged rollout for a prompt or model change moves through: shadow mode (the new version runs on a copy of live traffic but its output is never shown to users, only logged and compared against the current production version's output); canary (a small percentage, often 1-5%, of real users see the new version's actual output, with quality and safety metrics monitored closely); progressive rollout (traffic percentage increases in steps — 5%, 25%, 50%, 100% — with an automatic or manual pause-and-rollback gate at each step); and full rollout, at which point the new version becomes the baseline that the next change will be compared against.

```python
class StagedRollout:
    def __init__(self, prompt_registry, metrics_client):
        self.registry = prompt_registry
        self.metrics = metrics_client
        self.stages = [0.0, 0.05, 0.25, 0.50, 1.0]   # shadow, canary, then progressive

    def current_traffic_pct(self, rollout_name):
        return self.registry.storage.get_rollout_state(rollout_name)["traffic_pct"]

    def route(self, rollout_name, request_id):
        import random
        pct = self.current_traffic_pct(rollout_name)
        # Deterministic hashing on request/user id keeps a given user on a consistent
        # variant across their session, rather than flapping between versions per request.
        bucket = int(hashlib.sha256(f"{rollout_name}:{request_id}".encode()).hexdigest(), 16) % 10000
        return "candidate" if bucket < pct * 10000 else "baseline"

    def evaluate_gate(self, rollout_name):
        candidate_metrics = self.metrics.get_recent("candidate", rollout_name)
        baseline_metrics = self.metrics.get_recent("baseline", rollout_name)

        error_rate_ok = candidate_metrics["error_rate"] <= baseline_metrics["error_rate"] * 1.2
        latency_ok = candidate_metrics["p95_latency_ms"] <= baseline_metrics["p95_latency_ms"] * 1.3
        quality_ok = candidate_metrics["judge_score_avg"] >= baseline_metrics["judge_score_avg"] - 0.1

        return error_rate_ok and latency_ok and quality_ok

    def advance_or_rollback(self, rollout_name):
        state = self.registry.storage.get_rollout_state(rollout_name)
        if self.evaluate_gate(rollout_name):
            next_idx = min(self.stages.index(state["traffic_pct"]) + 1, len(self.stages) - 1)
            state["traffic_pct"] = self.stages[next_idx]
        else:
            state["traffic_pct"] = 0.0   # rollback to baseline entirely
            self.metrics.alert(f"rollout {rollout_name} rolled back on gate failure")
        self.registry.storage.put_rollout_state(rollout_name, state)
```

Shadow mode deserves special mention because it's underused relative to how much signal it provides for free: since the candidate's output is never actually served to a user, there's no user-facing risk at all, which means you can run shadow traffic at a much higher volume (even 100% of production traffic) than you'd ever dare for a live canary, giving you a much larger and more representative sample for comparing output distributions, cost, and latency before any real user is exposed to the change.

## What a Full Pipeline Looks Like

Stitching this together, a realistic CI/CD pipeline for an LLM application runs, in order: static checks (lint, type-check, unit tests on the deterministic parts of the system — prompt template rendering, tool-call parsing, retry logic — which are ordinary code and should be tested with ordinary deterministic unit tests); the golden-set regression suite from above, gated on category-level tolerance bands; safety-specific evaluation (a golden set specifically of adversarial/jailbreak/PII-leak cases, gated more strictly than general quality since safety regressions are less tolerable than quality regressions); a staging deployment where the change runs against a synthetic or replayed-traffic load test to validate latency and cost at scale; and finally the staged production rollout with automated gates as described above. The distinguishing feature versus traditional CI/CD isn't really the pipeline shape — that part is familiar — it's that two of these stages (the regression suite and the safety suite) require actual LLM calls to execute, which makes them slower and more expensive to run than a typical unit test suite, and pushes teams toward running the full suite on every merge to main while running a smaller, cheaper smoke-test subset on every pull request to keep iteration speed reasonable.

## Versioning the Full System, Not Just the Prompt

The last piece worth being explicit about: reproducibility for an LLM system requires versioning more than the prompt text. The tuple that fully determines behavior is (prompt version, model identifier and version, generation parameters, and — if retrieval is involved — the retrieval index/corpus version), and a change to any one of these four is a behavior change deserving the same regression-tested rollout as a prompt edit. Provider-side silent model updates are the trickiest of these to control, since a fixed model name in your code can resolve to a different underlying checkpoint over time at the provider's discretion; the practical mitigation is pinning to a dated model snapshot where the provider offers one, and treating any provider-forced migration off a deprecated snapshot as a full re-run of the regression suite before accepting the new default, exactly as you would for a deliberate model upgrade you chose yourself.

## Handling Non-Determinism Without Giving Up on Rigor

Flaky tests are a nuisance in traditional CI; in LLM CI they're a structural fact of life that needs a designed-for solution rather than a workaround. Three techniques handle this in combination. The first is pinning generation to the lowest useful temperature for anything used as a regression gate — evaluation is not the place to showcase creative sampling, and a temperature of 0 (or as close to fully deterministic as a given provider allows) removes a large fraction of run-to-run variance for free, even though it doesn't remove all of it, since even greedy decoding can vary slightly across hardware/batching configurations on some serving stacks. The second is running each case multiple times (3-5 repetitions is common) and aggregating — either majority vote for a discrete pass/fail check, or a mean and confidence interval for a continuous judge score — so that a single unlucky sample doesn't flip a gate on its own. The third, for cases that are inherently more subjective, is widening the pass criterion from an exact match to a tolerance band and being explicit in the eval design about what magnitude of difference actually matters for the product, rather than chasing an unrealistic standard of bit-for-bit reproducibility that generative models were never going to offer.

```python
def robust_case_result(case, candidate_generate_fn, judge_fn, repetitions=3):
    """Run a case multiple times and aggregate, rather than trusting a single sample --
    reduces (does not eliminate) the chance that sampling variance alone flips a CI gate."""
    outputs = [candidate_generate_fn(case.input) for _ in range(repetitions)]
    scored = [score_case(case, out, judge_fn) for out in outputs]

    if case.rubric:
        judge_scores = [s["judge_score"] for s in scored]
        avg_score = sum(judge_scores) / len(judge_scores)
        return {"passed": avg_score >= 3.5, "avg_judge_score": avg_score, "variance_across_reps": max(judge_scores) - min(judge_scores)}

    pass_votes = sum(1 for s in scored if s["passed"])
    return {"passed": pass_votes > repetitions / 2, "pass_votes": pass_votes, "repetitions": repetitions}
```

The `variance_across_reps` figure returned above is itself worth tracking over time as its own signal: a case whose repeated outputs vary wildly in judge score isn't just noisy for CI purposes, it's telling you the underlying prompt is under-specified or the task is genuinely ambiguous for the model — high output variance on a specific eval case is often the first sign that a prompt needs tightening before it becomes a live user-facing quality complaint.

## Dataset and Corpus Versioning

The regression suite's usefulness depends entirely on the golden set staying representative and correctly labeled, which means the golden set itself needs the same version-control discipline as the prompts it's evaluating — every change to it (adding a newly discovered failure case, updating a reference answer, retiring a stale case) should go through review, be tied to a commit, and be tagged with which baseline scores were measured against which version of the set, so a jump in pass rate can always be attributed correctly to either a genuine model improvement or simply an easier evaluation set. The same discipline applies to the retrieval corpus for any RAG-backed system: if the underlying document set an agent retrieves from changes (new documents added, stale ones removed, chunking strategy altered), that's a versioned event that can shift behavior just as much as a prompt or model change, and it deserves its own entry in whatever change log or dashboard tracks "what changed and when" for the system as a whole — otherwise a quality regression investigation can burn hours chasing a prompt or model change that was never the actual cause.

## Tooling Landscape

A range of dedicated LLMOps platforms (LangSmith, PromptLayer, Braintrust, and similar tools, alongside general observability platforms adding LLM-specific features) exist specifically to provide the prompt registry, eval-harness, and trace-logging pieces described in this chapter as an off-the-shelf product rather than something you build from scratch on top of Redis and Postgres. The build-vs-buy calculus here tracks the same logic as anywhere else in infrastructure: a small team validating product-market fit is usually better served adopting one of these platforms and getting a working prompt-versioning and eval pipeline in days rather than building an in-house equivalent, while a team with unusual scale, security, or workflow requirements not well served by an off-the-shelf tool may eventually justify building custom tooling — but that decision should come after outgrowing an existing platform's limits, not as a default first move driven by a general preference for owning infrastructure.
