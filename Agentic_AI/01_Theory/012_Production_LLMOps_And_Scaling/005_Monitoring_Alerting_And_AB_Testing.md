# Monitoring, Alerting, and A/B Testing

## The Extra Dimension: Monitoring Quality, Not Just Health

Every production service needs the standard observability triad — is it up, is it fast, is it erroring — and LLM applications need all of that too. What makes LLM observability a distinct discipline is a fourth dimension that doesn't exist for a typical CRUD service: is the *content* of what the system is producing still good. A service can be up, fast, and returning 200s while quietly generating worse answers than it did last month, because nothing about a successful HTTP response tells you whether the text inside it is accurate, on-tone, or safe. Building monitoring for an LLM system means instrumenting for both halves — the operational health that any service needs, and a quality signal that has to be actively constructed, because unlike latency or error rate, "quality" isn't something the infrastructure emits for free.

## Operational Metrics: Latency, Cost, and Errors

Latency for an LLM service needs finer granularity than a single end-to-end number, because the user-perceived experience of "slow" is shaped by more than total time. Time-to-first-token (TTFT) — how long a user waits before anything at all starts streaming back — dominates perceived responsiveness for chat interfaces, since a fast-starting stream feels responsive even if total generation takes several seconds, while a slow start feels broken even if the total time is identical. Time-per-output-token (TPOT, sometimes called inter-token latency) captures how smoothly the stream continues once it starts, and it's the metric most directly affected by serving-layer contention — a GPU fleet running near saturation will show rising TPOT well before it shows outright errors, making it an effective early-warning signal for capacity problems. End-to-end latency still matters for anything that isn't streamed (structured extraction, background agent steps), and all three should be tracked as full latency distributions (P50/P95/P99), not averages, since LLM latency distributions are heavily right-skewed — a handful of very slow requests can hide behind a perfectly reasonable average.

```python
from prometheus_client import Histogram, Counter

TTFT = Histogram("llm_ttft_seconds", "Time to first token", ["model"],
                  buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
TPOT = Histogram("llm_tpot_seconds", "Per-token latency after first token", ["model"],
                  buckets=[0.01, 0.02, 0.05, 0.1, 0.2])
REQUEST_COST = Histogram("llm_request_cost_dollars", "Cost per request", ["model", "feature"],
                          buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
ERROR_COUNT = Counter("llm_errors_total", "Errors by type", ["model", "error_type"])
REFUSAL_COUNT = Counter("llm_refusals_total", "Model-issued refusals", ["model", "feature"])

def record_generation_metrics(model, feature, ttft_s, tpot_s, cost, was_refusal, error_type=None):
    TTFT.labels(model=model).observe(ttft_s)
    if tpot_s is not None:
        TPOT.labels(model=model).observe(tpot_s)
    REQUEST_COST.labels(model=model, feature=feature).observe(cost)
    if was_refusal:
        REFUSAL_COUNT.labels(model=model, feature=feature).inc()
    if error_type:
        ERROR_COUNT.labels(model=model, error_type=error_type).inc()
```

Cost per request deserves its own histogram rather than a single running total, for the same reason latency does: a shift in the *shape* of the cost distribution (more requests drifting toward the expensive tail, for instance, because users are pasting in longer documents than they used to) is an actionable signal that a single "total spend today" number hides until the bill arrives. Error rate should be broken down by error type, not reported as one aggregate percentage, because a spike in provider rate-limit errors, a spike in timeout errors, and a spike in output-validation failures point at three completely different root causes and three different people to page.

Refusal rate is worth calling out specifically because it's an error mode unique to LLM systems: the model returns a perfectly well-formed, non-erroring response that is nonetheless a refusal to help ("I can't assist with that"), which looks completely healthy to standard HTTP monitoring while being a failed interaction from the user's perspective. Tracking refusal rate as its own first-class metric — and alerting on unexpected increases — catches an entire category of degradation (an overly cautious system prompt change, a safety-filter misconfiguration, a model version upgrade that shifted its refusal calibration) that would otherwise be invisible until support tickets pile up.

## Output Quality Drift

Quality drift is the hardest of these signals to build because there's no ground truth arriving automatically the way there is for latency or errors — nothing in the HTTP response tells you a generated answer was factually wrong or subtly off-tone. Production quality monitoring has to manufacture that signal, and it does so through some combination of three complementary techniques, layered because each has a different cost and a different blind spot.

The cheapest and most scalable is automated proxy signals: track things correlated with quality that can be computed without any additional model call — output length distribution (a sudden collapse toward very short answers, or a runaway toward very long ones, often precedes a more serious quality issue), the rate of outputs failing structural validation (malformed JSON from a function-calling flow, missing required fields), and the rate of users immediately re-asking a rephrased version of the same question (a strong implicit signal that the first answer didn't satisfy them). None of these require an extra LLM call, so they can run on 100% of traffic continuously.

```python
def compute_drift_signals(window_logs):
    lengths = [len(r["output"]) for r in window_logs]
    structural_failures = sum(1 for r in window_logs if r.get("validation_failed"))
    immediate_rephrasing = sum(1 for r in window_logs if r.get("followed_by_rephrase_within_30s"))

    return {
        "avg_output_length": sum(lengths) / len(lengths) if lengths else 0,
        "length_p10": sorted(lengths)[len(lengths) // 10] if lengths else 0,
        "structural_failure_rate": structural_failures / len(window_logs) if window_logs else 0,
        "rephrase_rate": immediate_rephrasing / len(window_logs) if window_logs else 0,
    }
```

The second layer is LLM-as-judge scoring on a continuously sampled subset of live traffic — not every request (that would double your inference cost), but a statistically meaningful random sample (1-5% is typical), scored against the same rubric-based approach used in the offline golden-set evaluation, so that the production quality trend line is directly comparable to the pre-deploy evaluation baseline. The third and most expensive, but most trustworthy, layer is periodic human review of a smaller sample, which exists specifically to catch the failure modes an automated judge model shares blind spots with the model being judged on (both being LLMs, they can be fooled by the same kinds of fluent-but-wrong output), and to periodically recalibrate confidence in the judge model itself.

```python
import random

def sample_for_quality_review(request_log, sample_rate=0.03):
    return [r for r in request_log if random.random() < sample_rate]

def score_sampled_traffic(sampled_requests, judge_fn, rubric):
    scores = []
    for r in sampled_requests:
        result = judge_fn(r["input"], r["output"], rubric)
        scores.append({"request_id": r["id"], "score": result["score"], "timestamp": r["timestamp"]})
    return scores

def detect_quality_drift(recent_scores, historical_baseline_mean, historical_baseline_std, z_threshold=2.0):
    recent_mean = sum(s["score"] for s in recent_scores) / len(recent_scores)
    z = (recent_mean - historical_baseline_mean) / historical_baseline_std if historical_baseline_std else 0
    return {"recent_mean": recent_mean, "z_score": z, "drifted": z < -z_threshold}
```

## Setting Alert Thresholds That Don't Cry Wolf

The naive approach to alerting — pick a fixed threshold, alert when a metric crosses it — works reasonably for infrastructure metrics with stable, well-understood normal ranges (CPU, memory, uptime), but LLM system metrics are noisier and more workload-dependent than that, and fixed thresholds either fire constantly on ordinary variance (alert fatigue, which trains people to ignore pages) or sit silent through a real degradation because the threshold was set for a different traffic mix than the one you actually have today. The fix is to baseline dynamically: alert on deviation from the metric's own recent historical distribution rather than an absolute number, using something as simple as a rolling mean and standard deviation, or a percentile-based comparison against the trailing week at the same time-of-day (to account for legitimate daily/weekly traffic patterns rather than treating Monday-morning traffic as an anomaly relative to Sunday-night traffic).

```python
class DynamicThresholdAlert:
    def __init__(self, metric_name, z_threshold=3.0, min_history=30):
        self.metric_name = metric_name
        self.z_threshold = z_threshold
        self.min_history = min_history
        self.history = []

    def observe(self, value):
        is_anomaly = False
        if len(self.history) >= self.min_history:
            mean = sum(self.history) / len(self.history)
            variance = sum((x - mean) ** 2 for x in self.history) / len(self.history)
            std = variance ** 0.5
            if std > 0:
                z = (value - mean) / std
                is_anomaly = abs(z) > self.z_threshold

        self.history.append(value)
        if len(self.history) > 500:
            self.history.pop(0)   # bounded rolling window

        return is_anomaly
```

Beyond dynamic baselining, three practical rules keep an alerting setup usable rather than a source of dread. First, tier severity deliberately: not every anomaly deserves to page someone at 3 a.m. — a slow upward drift in cost per request is a next-business-day investigation, while a spike in error rate past 10% or a safety-guardrail bypass is an immediate page, and conflating the two in one undifferentiated alert channel is how real pages get missed in the noise. Second, always pair an alert with a cooldown so that a sustained anomaly doesn't retrigger the same page every minute it remains true — one notification per incident, with a resolution notification when the metric recovers, not a running barrage. Third, alert on symptoms that map to user impact (refusal rate, error rate, P95 latency, judge-score drift) more than on raw internal signals (GPU utilization, queue depth) for anything that pages a human immediately — internal signals are excellent for dashboards and for the on-call engineer's investigation once paged, but paging on them directly tends to alert on things that haven't yet, and might never, affect a real user.

```python
class AlertRouter:
    SEVERITY_CHANNELS = {"critical": ["pagerduty", "slack"], "warning": ["slack"], "info": ["dashboard"]}

    def __init__(self, notifiers):
        self.notifiers = notifiers
        self.open_incidents = {}

    def fire(self, alert_name, severity, metric_value, cooldown_minutes=15):
        import time
        now = time.time()
        last = self.open_incidents.get(alert_name)
        if last and now - last < cooldown_minutes * 60:
            return  # already notified, suppress until cooldown or resolution
        self.open_incidents[alert_name] = now
        for channel in self.SEVERITY_CHANNELS.get(severity, ["dashboard"]):
            self.notifiers[channel].send(f"[{severity.upper()}] {alert_name}: {metric_value}")

    def resolve(self, alert_name):
        if alert_name in self.open_incidents:
            del self.open_incidents[alert_name]
```

## A/B Testing Prompts and Models Under Uncertainty

A/B testing an LLM change is structurally the same idea as any product experiment — split traffic, measure an outcome metric, decide if the difference is real — but LLM output variance makes the statistics less forgiving than a typical conversion-rate test. Individual LLM responses to the same prompt vary even at fixed settings, which means the *within-variant* noise floor is higher than for, say, a button-color test where a single user's click is a clean binary outcome. That higher noise floor means you need either a larger sample size to detect a given effect size with confidence, or you need to accept detecting only larger effect sizes reliably — pretending you can reliably detect a 1% quality difference from a few hundred samples per variant is a common and costly mistake.

The other trap specific to this domain is "peeking": checking results continuously and stopping the test the moment it looks significant. Standard significance tests assume you looked once, at a pre-committed sample size; checking repeatedly and stopping at the first favorable-looking p-value inflates your false-positive rate dramatically, sometimes to 20-30% even when nominally testing at 5% significance, because you're implicitly running many tests and taking the best one. The fix is either committing to a fixed sample size computed in advance from a target effect size and power, or using a sequential testing method (like a sequential probability ratio test or an always-valid confidence sequence) explicitly designed to allow continuous monitoring without inflating the false-positive rate.

```python
import math
from scipy import stats

def required_sample_size(baseline_rate, minimum_detectable_effect, alpha=0.05, power=0.8):
    """Sample size per variant for detecting a difference in a proportion metric
    (e.g. task-success rate, refusal rate) -- use this before launching a test,
    not after eyeballing early results."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)

print(required_sample_size(baseline_rate=0.85, minimum_detectable_effect=0.03))


def analyze_ab_test(variant_a_scores, variant_b_scores, alpha=0.05):
    """Two-sample test on a continuous quality metric (e.g. judge scores 1-5).
    Run this once, at the pre-committed sample size -- not repeatedly during collection."""
    t_stat, p_value = stats.ttest_ind(variant_a_scores, variant_b_scores, equal_var=False)
    mean_diff = (sum(variant_b_scores) / len(variant_b_scores)) - (sum(variant_a_scores) / len(variant_a_scores))
    return {
        "mean_diff": mean_diff,
        "p_value": p_value,
        "significant": p_value < alpha,
        "n_a": len(variant_a_scores),
        "n_b": len(variant_b_scores),
    }
```

Guardrail metrics are the other essential piece of a responsible LLM A/B test: you're rarely testing quality in isolation, and a variant that wins on the primary metric while quietly regressing cost, latency, or safety is not actually a win. A sound test design declares guardrail bounds up front (for instance, "the candidate prompt must not increase P95 latency by more than 20% or cost per request by more than 15%, regardless of quality gains") and treats a guardrail breach as an automatic fail independent of what the primary metric shows.

```python
def evaluate_ab_result(primary_result, candidate_metrics, baseline_metrics,
                        max_latency_increase_pct=20, max_cost_increase_pct=15):
    guardrails_ok = (
        candidate_metrics["p95_latency_ms"] <= baseline_metrics["p95_latency_ms"] * (1 + max_latency_increase_pct / 100)
        and candidate_metrics["avg_cost"] <= baseline_metrics["avg_cost"] * (1 + max_cost_increase_pct / 100)
        and candidate_metrics["error_rate"] <= baseline_metrics["error_rate"] * 1.1
    )

    if not guardrails_ok:
        return "reject: guardrail breach, regardless of primary metric result"
    if primary_result["significant"] and primary_result["mean_diff"] > 0:
        return "ship: statistically significant improvement, guardrails intact"
    return "no decision: insufficient evidence of improvement"
```

## Shadow Testing Before You Ever Split Live Traffic

For higher-stakes changes, it's worth running a shadow comparison before committing any real user to an A/B split at all: feed the candidate variant a copy of recent production traffic, generate its outputs without ever showing them to a user, and score those outputs (via the same judge-model and proxy-signal machinery used for live drift monitoring) against what the current production variant actually produced for the same inputs. This catches large, obvious regressions cheaply and without any user-facing risk, and it's the natural warm-up step before a live A/B test — you'd rather discover a candidate prompt has a 15% higher refusal rate from a shadow run than from the first few percent of real users hitting it in a canary.

## Bringing It Together

A mature monitoring setup for an LLM application, then, has three layers working continuously and feeding each other: real-time operational dashboards (latency, cost, error and refusal rates) that any on-call engineer can read at a glance and that page on dynamically-baselined anomalies; a sampled quality-drift pipeline (proxy signals on all traffic, judge scoring on a percentage sample, human review on a smaller sample still) that answers the question operational metrics can't — is the content still good; and an experimentation framework (shadow tests, then properly powered A/B tests with pre-declared guardrails) that governs how any change, whether a new prompt, a new model version, or a new routing policy, is allowed to reach full production traffic in the first place. None of the three layers is optional in isolation: operational metrics without quality monitoring will let a system quietly get worse while every dashboard stays green, and neither is worth much without a disciplined experimentation process to actually decide, with real statistical confidence, whether a proposed change should ship.

## Tracing: Connecting a Bad Output Back to Its Cause

Aggregate metrics tell you *that* something is wrong; they rarely tell you *why* on their own, especially in agentic systems where a single user-facing response might be the product of several chained LLM calls, tool invocations, and retrieval steps. Distributed tracing — recording every step of a request's execution as a span with timing, inputs, and outputs, tied together under one trace ID — is what lets an engineer go from "judge scores dropped 8% starting Tuesday" to "the drop is concentrated in requests where the web-search tool returned zero results and the model tried to answer from stale internal knowledge instead of saying so," which is an actionable finding, whereas the aggregate metric alone is not.

```python
class RequestTrace:
    def __init__(self, request_id):
        self.request_id = request_id
        self.spans = []

    def record_span(self, name, duration_ms, metadata=None):
        self.spans.append({"name": name, "duration_ms": duration_ms, "metadata": metadata or {}})

    def to_log_entry(self, final_output, judge_score=None):
        return {
            "request_id": self.request_id,
            "spans": self.spans,
            "final_output_preview": final_output[:200],
            "judge_score": judge_score,
            "total_duration_ms": sum(s["duration_ms"] for s in self.spans),
        }


def find_common_pattern_in_low_scoring_traces(traces, judge_scores, threshold=2.5, top_n=5):
    """A simple but effective investigation technique: pull traces below a quality
    threshold and look for a shared span pattern (a specific tool failing, a specific
    retrieval returning empty, an unusually long span) rather than reading transcripts
    one at a time with no starting hypothesis."""
    low_scoring = [t for t, s in zip(traces, judge_scores) if s < threshold]
    span_name_counts = {}
    for trace in low_scoring:
        for span in trace.spans:
            key = (span["name"], span["metadata"].get("error"))
            span_name_counts[key] = span_name_counts.get(key, 0) + 1
    return sorted(span_name_counts.items(), key=lambda kv: -kv[1])[:top_n]
```

## Human Feedback as a Continuous Signal

Explicit user feedback — thumbs up/down, a star rating, an "edit and resubmit" action, or simply whether a user abandoned a conversation — is a cheaper and more directly meaningful quality signal than any judge model, precisely because it comes from the actual person the system is trying to satisfy rather than a proxy for them, but it comes with its own biases that need correcting for before treating it as ground truth: feedback is voluntary and therefore skewed toward users who are unusually satisfied or unusually frustrated, response rates are typically low single-digit percentages, and a raw thumbs-down rate is not directly comparable across features or user segments with different baseline feedback propensities. Treated correctly — as one more sampled signal to correlate against judge scores and proxy metrics, rather than as the single source of truth — user feedback is invaluable for catching real-world failure patterns that a curated golden set, however well built, didn't anticipate, and feeding confirmed failure cases from production feedback back into the golden set is one of the highest-value habitual practices a team can build into its evaluation loop.

## A Minimal Incident Runbook

Because LLM system incidents span both the operational and quality dimensions, a runbook needs branches for both. For an operational page (error rate or latency breach), the standard playbook applies largely unchanged from any backend service: check provider status pages first (a meaningful fraction of LLM incidents are the provider's outage, not yours), fail over to a backup provider or model if a fallback chain is configured, and roll back the most recent prompt/model/config deploy if the timing lines up. For a quality-drift page (judge score or refusal rate anomaly), the first diagnostic step is almost always tracing: pull a sample of recent low-scoring traces (as above), check whether the drop correlates with a specific tool, a specific retrieval source, a specific user segment, or a specific recent deploy, and if a deploy correlates, treat rollback of that deploy as the default first response rather than attempting to root-cause and fix forward under incident pressure — the same instinct that applies to any production incident, non-deterministic or not: stop the bleeding first, understand it fully afterward with time and without a page pressuring you.

## The Cost of Monitoring Itself

One closing practical note: quality monitoring is not free, since judge-model scoring is itself a metered LLM cost, and it's possible to over-invest in evaluation infrastructure to the point that it becomes a meaningful fraction of total system spend. The sampling rates discussed earlier (1-5% for automated judge scoring, a smaller fraction still for human review) exist specifically to keep this cost bounded, and the right sampling rate is itself a decision to revisit periodically — a mature, stable feature can usually be monitored at a lower sampling rate than a newly launched one still being actively iterated on, and shifting sampling budget toward whatever part of the product is currently least proven is a more efficient use of the monitoring budget than applying a flat rate everywhere indefinitely.
