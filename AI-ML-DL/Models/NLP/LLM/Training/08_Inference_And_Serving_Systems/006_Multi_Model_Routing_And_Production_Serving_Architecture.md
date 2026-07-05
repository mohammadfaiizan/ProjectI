## Multi-Model Routing and Production Serving Architecture

### 1. Why a production LLM product is almost never one model

Every technique in files 001-005 is about serving *one* model as efficiently as possible. A real production LLM product — a chat assistant, an API platform, an agentic coding tool — is almost never built on a single model, for reasons that are economic and product-shaped as much as technical:

- **Cost-tiering by task difficulty.** Most real traffic is easy: short factual questions, simple rewrites, boilerplate completions, classification-shaped tasks. Sending every one of these to the largest, most expensive, highest-latency model available is a direct waste of money and latency budget with no corresponding quality benefit — the largest model's extra capability simply isn't needed to get the right answer to an easy query, and the user (or the product's cost structure) pays for capability that goes unused. Conversely, a small fraction of traffic is genuinely hard (multi-step reasoning, subtle code, ambiguous or high-stakes requests) where the largest model's extra capability is precisely what separates a right answer from a wrong one, and routing that traffic to a cheaper/smaller model risks a silent quality failure. This is the same core insight underlying every frontier lab's practice of shipping a flagship model alongside smaller "mini"/"nano"-class siblings, and, one level further, the GPT-5-style router that dispatches between a fast and a deep-reasoning model on a per-query basis rather than requiring a fixed model choice per deployment (see `..\GPT\010_GPT5_Series.md` Section 2 for the fully worked treatment of that specific design).
- **Latency-tiering.** Independent of raw capability, some product surfaces have hard latency budgets (autocomplete-style suggestions, a voice assistant's turn-taking loop) that only the fastest, smallest models can meet at all, regardless of whether a larger model would have given a marginally better answer — for these surfaces, "fits the latency budget" is a harder constraint than "is the best possible answer," and the model selection is effectively latency-first, quality-second within that constraint.
- **Specialized models for specific task types.** A general-purpose chat model is rarely the best tool for every sub-task inside a larger product: embedding models for retrieval, a dedicated content-moderation/safety classifier for guardrails, a code-specialized model for a coding-assistant surface, a small model fine-tuned specifically for a narrow, high-volume classification task (intent detection, routing itself) where a general model would be both slower and no more accurate. A mature LLM product is typically a *pipeline* of several purpose-built models plus at least one large general-purpose model, not a single endpoint.

The strategic throughline: cost, latency, and quality pull in different directions, and no single model sits at the optimum for all three simultaneously across a product's full traffic distribution — so the economically and technically correct design is a *system* of models with a dispatch layer choosing which one handles which request, not a single model handling everything.

### 2. Routing-model design

A router's job: given a query (and, often, conversation context, tool-use requirements, and explicit user/developer signals), decide which model or tier should handle it, before that model has actually produced an answer. This is a genuinely hard prediction problem, and the design space spans a spectrum of sophistication:

- **Heuristic/rule-based routing.** Cheap, fast, fully interpretable: route based on query length, presence of specific markers (code fences, explicit phrases like "think step by step" or "prove that," detected tool-call requirements, an explicit user-selected mode), or simple keyword/regex matching. Cheap to build and reason about, but brittle — it captures only the signals a human thought to encode, and misses any subtler correlation between query characteristics and actual required capability.
- **Learned classifier routing.** A small, fast model (or even a lightweight non-neural classifier over hand-engineered features) trained to predict which tier a query needs, typically using either human-graded labels ("did this query actually need the expensive tier to get a good answer") or model-graded labels (comparing the cheap and expensive tier's actual outputs on a query and using a judge — often the expensive model itself, or a separate grading model — to label which tier's answer was better, then training the router on those labels). This is more accurate than pure heuristics but introduces its own hard problem: **building the training-label ground truth** requires actually running both tiers on a labeled sample and grading the difference, which is exactly the "oracle labeling" problem discussed in `..\GPT\010_GPT5_Series.md` Section 8/12 for the GPT-5 router — the "correct" tier for a query is not an intrinsic property of the query text, it's a function of the acceptable quality bar for the specific downstream use case, so what counts as ground truth has to be defined relative to a chosen quality threshold, not treated as an objective label.
- **Cascade / confidence-based escalation.** Rather than deciding up front, run the cheap model first, and escalate to the expensive model only if the cheap model's own output signals low confidence (e.g., a low-probability/high-entropy generation, an explicit self-reported uncertainty, or a downstream verifier rejecting the cheap model's answer). This avoids needing a separate, independently-trained router model at all, but pays the cheap model's latency/cost on every request even when escalation was inevitable, and requires a reliable confidence signal from the cheap model, which is itself nontrivial to calibrate well (models are frequently miscalibrated — confidently wrong on exactly the queries where escalation was most needed).
- **Router embedded in the fast model itself**, as speculated (explicitly, and clearly flagged as speculation) for GPT-5 in `..\GPT\010_GPT5_Series.md` Section 2: the fast model performs a partial forward pass and emits an escalation signal, rather than a wholly separate router model doing a full independent inference. This amortizes the routing decision's cost into work the system was doing anyway, at the cost of coupling the router's behavior to the fast model's own training and making the two harder to update or evaluate independently (a documented staff-level concern in that same doc's Section 8: router drift when either component changes without the other being re-validated).

**The router's own latency budget is a hard constraint, independent of which design is chosen**: whatever the router costs, it must be small relative to the fast tier's own latency, or the fast path's entire reason for existing is undermined — a router that itself takes a meaningful fraction of a second to decide "use the fast model" has partly defeated the purpose of having a fast model at all. This pushes real designs toward cheap-to-evaluate features or a design where the fast path begins responding concurrently with the routing decision, rather than a heavyweight, fully separate routing model evaluated serially before any user-facing work begins.

**The asymmetric cost structure of mis-routing** is the central tuning difficulty, and it deserves to be understood precisely rather than folded into a single "routing accuracy" metric: routing an easy query to the expensive tier wastes cost and latency — a visible, measurable, infrastructure-level failure. Routing a hard query to the cheap tier degrades answer quality, often **silently** — the user has no way of knowing their query warranted more capability, and the failure only shows up (if at all) in downstream task success metrics, which are noisier and slower to observe than a cost/latency dashboard. Any real router-tuning program has to treat these as different currencies to be traded off deliberately (e.g., biasing toward the expensive tier for contexts where a wrong answer is costly — enterprise, agentic tool-use with real-world side effects — and biasing toward the cheap tier for high-volume, low-stakes conversational traffic) rather than optimizing one blended accuracy number that implicitly assumes both error types are equally costly, which they demonstrably are not.

### 2b. Building the oracle-labeling pipeline concretely

Section 2's learned-classifier bullet named the oracle-labeling problem; it's worth walking through what actually building one looks like, since "get ground truth" is easy to say and genuinely hard to execute well.

```python
from dataclasses import dataclass

@dataclass
class OracleLabel:
    query: str
    cheap_output: str
    expensive_output: str
    cheap_meets_bar: bool     # graded independently against a fixed quality bar
    expensive_meets_bar: bool
    escalation_necessary: bool  # True iff cheap failed the bar AND expensive passed it

def label_query(query: str, cheap_model, expensive_model, judge) -> OracleLabel:
    cheap_out = cheap_model.generate(query)
    expensive_out = expensive_model.generate(query)

    # `judge` grades each output independently against a FIXED quality bar --
    # not a pairwise "which is better" comparison, which would conflate "expensive
    # is somewhat better" (often true, rarely decision-relevant) with "cheap is
    # actually inadequate" (the only thing that should drive an escalation label).
    cheap_ok = judge.meets_bar(query, cheap_out)
    expensive_ok = judge.meets_bar(query, expensive_out)

    return OracleLabel(
        query=query, cheap_output=cheap_out, expensive_output=expensive_out,
        cheap_meets_bar=cheap_ok, expensive_meets_bar=expensive_ok,
        escalation_necessary=(not cheap_ok) and expensive_ok,
    )

def router_precision_recall(labels: list[OracleLabel], router_decisions: list[str]):
    """router_decisions[i] in {"cheap", "expensive"} -- the router's actual choice
    for labels[i].query. Returns escalation precision/recall against the oracle."""
    tp = fp = fn = tn = 0
    for label, decision in zip(labels, router_decisions):
        escalated = decision == "expensive"
        if escalated and label.escalation_necessary:
            tp += 1
        elif escalated and not label.escalation_necessary:
            fp += 1
        elif not escalated and label.escalation_necessary:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
```

The deliberate design choice here — grading each tier's output *independently* against a fixed bar, rather than asking a judge "which output is better" — matters more than it looks. A pairwise judge will very often say the expensive tier's answer is "better" even on queries where the cheap tier's answer was already perfectly adequate (bigger models tend to produce marginally more thorough, more polished answers almost everywhere), which would mislabel nearly every query as needing escalation and produce a router trained to escalate far more aggressively than actually necessary. Grading against a fixed, independent bar — "is this answer good enough," not "is this answer better than the other one" — is what correctly isolates the queries where escalation was *load-bearing* rather than merely "would have helped a little," and is the single most important methodological detail in building this pipeline correctly. Once labeled this way, `router_precision_recall` gives the separated, non-blended reporting Section 2's asymmetric-cost argument calls for: false positives (unnecessary escalations) and false negatives (missed necessary escalations) are tracked separately, exactly because a business should weight them differently rather than optimizing one merged accuracy number.

### 3. What a realistic end-to-end serving architecture looks like

Zooming out from any one model's serving stack to the full production system a company operating a major LLM product runs:

```
                          ┌─────────────────┐
   client request  ───►   │   API gateway /   │
                          │  auth / rate-limit │
                          └─────────┬─────────┘
                                    ▼
                          ┌─────────────────┐
                          │  Router / model   │──── explicit tier override (API param)
                          │     selector       │──── heuristic / learned classifier
                          └─────────┬─────────┘
                     ┌──────────────┼──────────────┐
                     ▼              ▼              ▼
              ┌───────────┐  ┌───────────┐  ┌───────────┐
              │ Cheap-tier│  │ Mid-tier   │  │ Frontier/  │
              │  replica   │  │  replica   │  │ reasoning  │
              │   pool     │  │   pool     │  │   pool     │
              │ (load-     │  │ (load-     │  │ (load-     │
              │ balanced,  │  │ balanced)  │  │ balanced)  │
              │ autoscaled)│  │            │  │            │
              └───────────┘  └───────────┘  └───────────┘
                     │              │              │
                     └──────────────┴──────────────┘
                                    ▼
                          ┌─────────────────┐
                          │  Response post-   │──── safety/moderation filter
                          │   processing       │──── streaming assembly
                          └─────────┬─────────┘
                                    ▼
                             client response

        (cutting across all pools: monitoring/alerting, canary traffic
         splitting, fallback/retry logic -- described below)
```

**Load balancing across replicas.** Each model tier runs as a pool of replicas (identical model weights loaded on multiple GPU nodes), with a load balancer distributing incoming requests — typically weighted by each replica's current queue depth or estimated available KV-cache capacity (file 001/003) rather than pure round-robin, since replicas can have very different current load depending on how many long-context requests they happen to be holding at any moment. Autoscaling adds or removes replicas based on aggregate queue depth/latency signals, on a timescale that has to account for how slowly a new GPU replica can actually come online (model weights loading into HBM is not instantaneous, especially for large models — autoscaling for LLM serving has a materially slower reaction time than typical stateless web-service autoscaling, and capacity planning has to budget for that lag rather than assuming near-instant elasticity).

**Multi-provider routing.** Beyond routing across a single company's own model tiers, many products (especially those built *on top of* foundation model APIs, rather than a frontier lab's own first-party product) route across *multiple providers* entirely — calling different vendors' APIs for different task types, or maintaining a fallback path to a second provider if the primary provider's API degrades or becomes unavailable. This is architecturally the same routing problem as Section 2, with an added dimension: different providers' APIs have different latency/cost/rate-limit characteristics that themselves need to be monitored and factored into the routing decision, and a robust system typically wants the ability to fail over to an alternate provider under a primary-provider outage without that failover itself becoming a visible product-quality regression (a fallback model that behaves noticeably differently from the primary is a consistency problem, not just an availability fix).

**Fallback and retry logic.** Any single inference request can fail for reasons unrelated to the model's actual capability: a transient infrastructure error, a timeout under load, a malformed/unparseable structured-output response that fails a downstream schema check. A production system needs an explicit policy for what happens next — retry against the same tier (appropriate for transient infra failures), retry against a different replica (appropriate if a specific replica seems unhealthy), escalate to a different tier (appropriate if the failure looks capability-related, e.g. a structured-output validation failure that a stronger model might not have made), or fail the request back to the client with a clear error (appropriate once retries are exhausted, to avoid an unbounded retry loop consuming capacity during an outage). Retries need backoff and a cap — naive unlimited retry-on-failure under a real outage is a classic way to turn a partial capacity problem into a full cascading outage, as the retry traffic itself further overloads already-struggling capacity.

**Canary deployment of a new model version.** Before fully cutting traffic over to a new model checkpoint (a new fine-tune, a new quantization scheme, or a genuinely new model version), production systems route a small, deliberately limited fraction of live traffic to the new version while the bulk of traffic continues to the previous, validated version, comparing quality/latency/error-rate signals between the two before progressively increasing the new version's traffic share (a standard staged-rollout pattern, not unique to LLM serving, but with LLM-specific wrinkles: "quality regression" is harder to detect automatically than a typical service's error rate, since a fluent, well-formatted, *wrong* answer produces no explicit error signal at all — canary evaluation for an LLM version change generally needs either human-graded comparison samples, an automated LLM-judge comparison against the previous version's outputs on the same canary traffic, or task-specific downstream-success metrics, layered on top of the standard infra-level canary signals of latency/error-rate/throughput).

### 3b. Canary traffic splitting and staged rollout, made concrete

```python
import hashlib

class CanaryRouter:
    def __init__(self, canary_fraction: float = 0.01):
        self.canary_fraction = canary_fraction   # start small; ramp explicitly (Section 3c)
        self.metrics = {"stable": [], "canary": []}

    def _bucket(self, request_id: str) -> float:
        """Deterministic hash-based bucketing: the SAME request_id always lands in the
        same bucket, which matters for reproducibility when debugging a specific
        complaint (Q18 in this module's interview questions relies on being able to
        replay exactly what a given request saw)."""
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def route(self, request_id: str) -> str:
        return "canary" if self._bucket(request_id) < self.canary_fraction else "stable"

    def record(self, version: str, latency_ms: float, quality_score: float):
        self.metrics[version].append({"latency_ms": latency_ms, "quality_score": quality_score})

    def ramp_if_healthy(self, min_samples: int = 500, latency_regression_tol: float = 1.10,
                         quality_regression_tol: float = 0.98, ramp_step: float = 2.0,
                         max_fraction: float = 1.0) -> bool:
        """Very simplified staged-rollout gate: only increase canary traffic once enough
        samples exist AND neither latency nor quality has regressed beyond tolerance."""
        stable, canary = self.metrics["stable"], self.metrics["canary"]
        if len(canary) < min_samples or len(stable) < min_samples:
            return False
        avg = lambda xs, k: sum(x[k] for x in xs) / len(xs)
        latency_ratio = avg(canary, "latency_ms") / avg(stable, "latency_ms")
        quality_ratio = avg(canary, "quality_score") / avg(stable, "quality_score")
        if latency_ratio > latency_regression_tol or quality_ratio < quality_regression_tol:
            return False   # do NOT ramp -- a real system would alert and consider rollback
        self.canary_fraction = min(max_fraction, self.canary_fraction * ramp_step)
        return True
```

This sketch deliberately keeps the health-check gate simple (a ratio threshold on two aggregate metrics) to make the *shape* of the decision legible; a production gate would additionally require statistical significance testing rather than a raw ratio (to avoid ramping or halting based on noise from a small sample), would track several quality signals rather than one scalar `quality_score` (Section 4's monitoring checklist), and would have an explicit, separate rollback path triggered by the same health check rather than only ever pausing the ramp. The detail worth internalizing regardless of implementation sophistication: **deterministic bucketing by request/user ID**, not per-request random sampling, is what makes a canary reproducible and debuggable — without it, a specific user's bad experience can never be reliably replayed against the canary path to investigate, which directly undermines the diagnostic process this module's Q18 (file 008) walks through.

### 3c. Multi-provider architecture: circuit breakers and rate-limit-aware routing

Section 3's multi-provider discussion named the basic idea; the concrete mechanism worth knowing is the **circuit breaker** pattern, borrowed from general distributed-systems practice and directly applicable here. A circuit breaker tracks a provider's recent error/timeout rate and, once it crosses a threshold, "opens" — stops sending traffic to that provider entirely for a cooldown period, rather than continuing to send requests (and accumulate latency and failures) to a provider that's already struggling, which would otherwise both waste the failing requests' latency budget and pile additional load onto a provider that's already unhealthy.

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold: float = 0.5, window_sec: float = 30.0,
                 cooldown_sec: float = 60.0, min_requests: int = 20):
        self.failure_threshold = failure_threshold
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec
        self.min_requests = min_requests
        self.events: list[tuple[float, bool]] = []   # (timestamp, was_success)
        self.opened_at: float | None = None

    def record(self, success: bool):
        now = time.time()
        self.events.append((now, success))
        self.events = [(t, s) for t, s in self.events if now - t <= self.window_sec]
        if not success and self._current_failure_rate() >= self.failure_threshold:
            self.opened_at = now

    def _current_failure_rate(self) -> float:
        if len(self.events) < self.min_requests:
            return 0.0
        failures = sum(1 for _, s in self.events if not s)
        return failures / len(self.events)

    def is_available(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at >= self.cooldown_sec:
            self.opened_at = None   # half-open: allow traffic again, re-evaluate
            return True
        return False
```

Layered on top of per-provider circuit breakers, a multi-provider router also has to be **rate-limit-aware** in a way a single-provider router doesn't: each provider imposes its own request-per-minute and token-per-minute ceilings, and a router that ignores this and simply load-balances by latency or cost risks driving a provider into its own rate-limit throttling, which looks to the router exactly like a reliability degradation (elevated error rate) even though the underlying model is healthy — the fix, in the provider-facing dimension, is to track each provider's consumed-quota-versus-limit ratio as its own first-class routing signal, feeding proactively into the load-balancing weight rather than waiting for rate-limit errors to show up reactively as failures the circuit breaker then has to absorb.

### 4. Monitoring signals that would flag a serving-layer regression

A staff engineer on call for a serving system needs a mental checklist of signals, organized by what kind of regression each one is most diagnostic of:

- **TTFT and TPOT/ITL, tracked at p50 *and* p99 separately** (file 005 Section 3) — a p99 TTFT spike with a stable p50 usually points to queueing/admission-control or tail contention (a burst of long prompts, or KV-cache exhaustion triggering preemption, file 003 Section 7) rather than a systemic capacity shortfall; a shift in the whole distribution points to genuine capacity or hardware degradation.
- **GPU utilization and HBM occupancy per replica** — a drop in utilization alongside rising latency is a strong signal of a scheduling or batching regression (something is preventing the batch from filling up, file 003) rather than raw capacity exhaustion; rising HBM occupancy toward the ceiling alongside rising latency points at KV-cache pressure (file 001) directly.
- **Queue depth / admitted-vs-waiting request counts per pool** — a growing waiting queue with stable per-replica latency indicates an under-provisioned pool (need more replicas), not a per-request performance problem.
- **Token throughput per GPU-second, tracked against cost-per-token** (file 005 Section 4) — a drop here at constant traffic volume is the direct financial signal that something (a batching regression, a quantization rollback, a bad autoscaling decision) has made serving materially less efficient, independent of whether any user-facing latency SLA has actually been breached yet.
- **Error rate and retry rate, broken out by failure type** — infra errors (timeouts, OOM) versus output-validation failures (schema/parse failures on structured output) are different regressions with different owners and different fixes; conflating them into one aggregate "error rate" hides which one is actually happening.
- **Routing distribution drift** — the fraction of traffic landing on each tier, tracked over time; an unexplained shift (suddenly far more traffic escalating to the expensive tier, or far less) is often the earliest available signal of either a genuine shift in traffic composition or a router regression (file 006 Section 2's router-drift concern, sharpened: if the fast tier itself changed and the router wasn't re-validated against the new fast-tier capability gap, routing behavior that used to be well-calibrated can silently become miscalibrated).
- **Online quality/eval signals** — since a serving-layer regression can be entirely invisible to every infra metric above while still degrading answer quality (a model serving perfectly fast, well-formed, *wrong* answers), mature systems layer in continuous, automated quality signals: sampled human grading, LLM-judge comparison against a reference/previous-version baseline, or task-specific downstream success metrics (e.g., did an agentic tool-use trace actually complete its task) — these are slower and noisier than infra metrics but are the only signals that catch this failure mode at all, and are the natural complement to the canary-comparison methodology in Section 3.

The unifying principle across all of this: a production LLM serving architecture's job is not merely "run the model fast," it is to make deliberate, monitored, reversible decisions about *which* model handles *which* request, under a cost/latency/quality trade-off that shifts with traffic composition and with every model or infrastructure change — and the monitoring stack's job is to make every one of those decisions' consequences observable quickly enough to catch a regression before it becomes a sustained quality or cost problem, given that some of the most damaging failure modes (silent quality degradation, router miscalibration) are specifically the ones that don't show up as a clean infra alert.

### 5. A minimal alerting-rule sketch

To close the gap between "here are the signals" (Section 4) and an actually actionable monitoring system, here is the shape a simple rule-evaluation layer over those signals takes — deliberately minimal, to make the structure legible rather than to be a production-ready implementation:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class AlertRule:
    name: str
    check: Callable[[dict], bool]   # takes a metrics snapshot, returns True if breached
    severity: str                    # "page" vs "ticket", i.e. wake someone up or not

def evaluate_rules(rules: list[AlertRule], metrics_snapshot: dict) -> list[str]:
    return [rule.name for rule in rules if rule.check(metrics_snapshot)]

rules = [
    AlertRule("ttft_p99_regression",
              lambda m: m["ttft_p99_ms"] > m["ttft_p99_baseline_ms"] * 1.5,
              severity="page"),
    AlertRule("gpu_util_drop_with_latency_up",
              lambda m: m["gpu_utilization"] < 0.5 and m["ttft_p50_ms"] > m["ttft_p50_baseline_ms"] * 1.2,
              severity="page"),
    AlertRule("kv_cache_near_ceiling",
              lambda m: m["kv_cache_occupancy_fraction"] > 0.95,
              severity="page"),
    AlertRule("routing_distribution_drift",
              lambda m: abs(m["expensive_tier_fraction"] - m["expensive_tier_fraction_baseline"]) > 0.10,
              severity="ticket"),
    AlertRule("cost_per_token_regression",
              lambda m: m["cost_per_million_tokens"] > m["cost_per_million_tokens_baseline"] * 1.25,
              severity="ticket"),
]
```

Each rule here is a direct, mechanical encoding of one diagnostic from Section 4's checklist — the `gpu_util_drop_with_latency_up` rule specifically encodes the "utilization down + latency up = scheduling regression, not capacity shortfall" pattern from Section 4's second bullet, rather than alerting on either signal in isolation, which is exactly the kind of *compound* condition that turns a list of individually-useful metrics into an actually diagnostic alerting system. The `severity` field matters as much as the check itself: routing-distribution drift and cost regressions are real signals worth investigating but rarely justify waking someone up at 3am the way a p99 latency or KV-cache-ceiling breach does — deliberately distinguishing "page" from "ticket" severity is what keeps an on-call rotation sustainable rather than alert-fatigued into ignoring everything.

### 6. Summary checklist for a staff-level routing-and-architecture discussion

- State the three independent reasons for multi-model serving (cost-tiering, latency-tiering, task specialization, Section 1) rather than collapsing them into one vague "cost savings" argument — an interviewer will often probe which of the three actually applies to a specific scenario they pose.
- Be able to name and compare the router design points on the sophistication spectrum (heuristic, learned classifier, cascade, embedded-in-model, Section 2) and to state the router's own latency-budget constraint as a hard requirement, not an afterthought.
- Treat mis-routing's two failure directions (wasted cost vs. silent quality loss) as separate currencies, and be able to sketch an oracle-labeling methodology (Section 2b) that avoids conflating component capability with routing quality.
- Be able to draw the end-to-end architecture (Section 3) from memory — gateway, router, per-tier replica pools, response post-processing — and name where load balancing, fallback/retry, circuit breakers, and canary logic each sit within it.
- Be able to list monitoring signals organized by what failure mode each is diagnostic of (Section 4), not as an undifferentiated list, and be explicit that the most dangerous regressions (silent quality degradation, router miscalibration) are exactly the ones invisible to pure infra metrics.
