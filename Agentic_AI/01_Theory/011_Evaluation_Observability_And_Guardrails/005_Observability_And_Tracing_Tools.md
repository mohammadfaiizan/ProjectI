# Observability and Tracing Tools

## Why LLM Observability Is a Different Problem Than Traditional APM

Traditional application performance monitoring was built for a world of deterministic, structurally simple requests: a request comes in, hits a handful of well-defined code paths, touches a database, returns a response, and the same input reliably produces the same output (or close enough that variance is treated as a bug, not an expected property). The observability tooling built for that world — Datadog, New Relic, Prometheus/Grafana, distributed tracing via OpenTelemetry — is optimized around metrics like latency percentiles, error rates, and request throughput, and around traces that are structurally shallow and predictable: a handful of nested spans representing service calls, each with a clear input/output contract.

LLM and agent systems violate nearly every one of those assumptions, and each violation demands something traditional APM wasn't built to capture. **Non-determinism** means the same input can legitimately produce different outputs on different runs (even at temperature zero, subtle numerical non-determinism in GPU inference and provider-side changes mean you can't assume byte-identical reproducibility), so "did this request behave correctly" can't be answered by comparing against a fixed expected value the way a traditional integration test would — you need the actual generated content, not just a pass/fail status code, to judge whether a given run was good. **Rich payloads carry the actual signal**: for a traditional API call, the request/response bodies are usually secondary to the metrics (you care that the call took 200ms and returned 200 OK, less about the exact bytes); for an LLM call, the prompt and completion text *are* the primary thing you need to debug quality issues, and a trace that logs latency and status code but not the actual prompt and response is close to useless for figuring out why an agent gave a bad answer. **Multi-step, branching traces**: a single user-facing agent request can fan out into a dozen or more LLM calls, tool invocations, and retrieval lookups, often with conditional branching based on intermediate results (the agent decides which tool to call next based on what the previous tool returned), producing a trace shape that's deep, wide, and data-dependent in a way a typical microservice call graph, which is comparatively fixed and topology-stable across requests, is not. **Cost as a first-class metric**: token usage translates directly and immediately to dollars in a way that traditional compute metrics (CPU-seconds, memory) usually don't map to at the same per-request granularity that matters to a product owner, which means cost needs to be tracked at the same fidelity as latency — per call, per step, per trace, aggregable by user/feature/prompt-version — rather than as an infrastructure-team concern several layers removed from the request path.

The consequence of all this is that LLM observability tools had to be built as a new category rather than bolted onto existing APM, because the unit of interest (a full multi-step, prompt-and-completion-carrying, cost-attributed, non-deterministic trace) doesn't fit the schema traditional tracing systems were designed around.

## Anatomy of a Good Agent Trace

A trace worth having is one that lets you reconstruct, after the fact, exactly what happened and why — without needing to reproduce the run. That requires capturing several layers of information that are each individually easy to skip under time pressure but collectively make the difference between a debuggable system and a black box.

At the top level, every trace needs a **session/conversation identifier** tying together all the turns of a multi-turn interaction, and a **trace identifier** for the specific request being traced, because a single user session can span many independent agent invocations and you need to be able to zoom from "show me everything this user did today" down to "show me this one problematic response." Within a trace, every individual **span** — an LLM call, a tool invocation, a retrieval lookup — needs its own identifier, a parent-span link establishing the call tree, start/end timestamps, and status (success, error, or a specific error type).

For LLM call spans specifically, the trace needs the full prompt actually sent to the model (after all templating, retrieval injection, and history truncation — not the template before those transformations, which is a common and costly mistake, since the whole point of debugging is seeing what the model actually received), the full completion returned, the model and parameters used (name, temperature, max tokens — because behavior differences across model versions or parameter changes are one of the most common causes of a quality regression that's otherwise invisible), and token counts and cost for that specific call. For tool-call spans, the trace needs the tool name, the exact arguments passed, the raw result returned, latency, and any error. For retrieval spans, the trace needs the query used, the documents/chunks retrieved with their relevance scores, and which of those retrieved chunks actually made it into the final prompt (a subtlety: retrieval can return ten documents but the prompt-assembly step might truncate to the top three, and if you only log retrieval, not final-prompt-assembly, you can't tell which case you're debugging).

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import uuid


@dataclass
class Span:
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    span_type: str = "llm_call"  # "llm_call" | "tool_call" | "retrieval" | "agent_step"
    name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "in_progress"  # "success" | "error" | "in_progress"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)  # model, temperature, tokens, cost
    error: Optional[str] = None


class Tracer:
    """Minimal illustrative tracer -- production tools like LangSmith or
    Langfuse provide this plus storage, querying, and UI on top."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.trace_id = str(uuid.uuid4())
        self.spans: list[Span] = []
        self._stack: list[str] = []  # active span ids, for auto-parenting

    def start_span(self, span_type: str, name: str, inputs: dict) -> Span:
        parent_id = self._stack[-1] if self._stack else None
        span = Span(span_type=span_type, name=name, inputs=inputs, parent_span_id=parent_id)
        self.spans.append(span)
        self._stack.append(span.span_id)
        return span

    def end_span(self, span: Span, outputs: dict = None, error: str = None, metadata: dict = None):
        span.end_time = datetime.utcnow()
        span.status = "error" if error else "success"
        span.outputs = outputs or {}
        span.error = error
        if metadata:
            span.metadata.update(metadata)
        if self._stack and self._stack[-1] == span.span_id:
            self._stack.pop()

    def log_llm_call(self, prompt: str, completion: str, model: str,
                      temperature: float, prompt_tokens: int, completion_tokens: int,
                      cost_usd: float, error: str = None) -> Span:
        span = self.start_span("llm_call", name=f"llm:{model}", inputs={"prompt": prompt})
        self.end_span(
            span,
            outputs={"completion": completion},
            error=error,
            metadata={
                "model": model, "temperature": temperature,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "cost_usd": cost_usd,
            },
        )
        return span

    def export(self) -> dict:
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "spans": [vars(s) for s in self.spans],
        }
```

A subtlety that separates a genuinely useful trace from a checkbox-compliance one: the trace needs to capture **why** the agent made each decision, not just **what** it did. If your agent framework exposes a reasoning/thought field (as in a ReAct-style loop), that text belongs in the span metadata even though it's never shown to the user, because it's frequently the single most useful piece of information when debugging a wrong tool choice after the fact — without it you're left guessing at the model's rationale from the tool call alone. Similarly, capturing the *system prompt version* and *retrieval index version* active at the time of each trace matters enormously for debugging regressions days or weeks later, when "we changed the prompt on Tuesday" needs to be correlated against "quality dropped starting Tuesday," and that correlation is impossible if traces don't record which prompt version actually produced them.

## What to Do With Traces: From Debugging to Systematic Signal

Traces are useful individually for debugging a specific bad response a user reported, but their larger value comes from aggregation. Once you have traces flowing for a meaningful volume of production traffic, several classes of analysis become possible that a single trace can't give you. **Latency and cost breakdown by span type** tells you where a slow or expensive agent run is actually spending its time and money — commonly it's not the "obvious" main LLM call but an expensive retrieval step, a redundant tool call, or a long tail of retried calls after transient errors, and this is invisible without span-level granularity. **Error rate by tool or by model** lets you catch a specific tool integration degrading (an external API starting to time out more) or a model version regression (a provider-side model update silently changing behavior) well before it shows up as a broad quality complaint. **Automatic quality scoring on sampled traces**, wiring the evaluation techniques from earlier chapters (LLM-as-judge scoring, trajectory evaluation, hallucination/faithfulness checks) directly into the trace pipeline so that a percentage of live production traffic gets scored continuously rather than only at pre-release eval time, is what turns observability from a purely reactive debugging tool into a proactive quality-monitoring system that can alert on a metric drop before a large volume of users notice and complain.

```python
class TraceAnalyzer:
    def __init__(self, traces: list[dict]):
        self.traces = traces

    def latency_by_span_type(self) -> dict:
        buckets = {}
        for trace in self.traces:
            for span in trace["spans"]:
                if span["end_time"] is None:
                    continue
                duration_ms = (span["end_time"] - span["start_time"]).total_seconds() * 1000
                buckets.setdefault(span["span_type"], []).append(duration_ms)
        return {
            span_type: {
                "p50": sorted(durations)[len(durations) // 2],
                "p95": sorted(durations)[int(len(durations) * 0.95)],
                "count": len(durations),
            }
            for span_type, durations in buckets.items()
        }

    def cost_by_model(self) -> dict:
        costs = {}
        for trace in self.traces:
            for span in trace["spans"]:
                if span["span_type"] != "llm_call":
                    continue
                model = span["metadata"].get("model", "unknown")
                costs[model] = costs.get(model, 0) + span["metadata"].get("cost_usd", 0)
        return costs

    def error_rate_by_tool(self) -> dict:
        counts = {}
        for trace in self.traces:
            for span in trace["spans"]:
                if span["span_type"] != "tool_call":
                    continue
                name = span["name"]
                counts.setdefault(name, {"total": 0, "errors": 0})
                counts[name]["total"] += 1
                if span["status"] == "error":
                    counts[name]["errors"] += 1
        return {
            name: c["errors"] / c["total"] for name, c in counts.items() if c["total"] > 0
        }
```

A sampled-scoring pipeline built on top of this typically routes a fixed percentage of production traces (plus 100% of traces with unusually high latency, cost, or error status, since those are disproportionately likely to be interesting) through the same LLM-judge and trajectory-evaluator machinery covered in earlier chapters, storing the resulting scores alongside the trace so that a quality dashboard can plot pass rate over time the same way an APM dashboard plots error rate over time — which is precisely the point: LLM observability succeeds when it makes quality trackable with the same rigor and immediacy that traditional APM gives to uptime and latency.

## A Worked Debugging Example

It's worth walking through what this actually looks like when something goes wrong, because the abstract description of "trace everything" doesn't convey why it matters until you've traced through a real failure. Say a support-agent product's user-satisfaction score drops noticeably starting on a Tuesday, with no corresponding code deploy that week. Without traces, this is close to undiagnosable — you have an aggregate metric moving in the wrong direction and no way to localize the cause. With traces, the investigation becomes mechanical: pull all traces from the affected window, break down average trajectory length and tool error rate by day, and you might find that `error_rate_by_tool` for a specific `lookup_order` tool jumped from 2% to 30% precisely on Tuesday — pointing at an upstream API change or outage rather than anything in the agent or prompt at all. Alternatively, you might find tool error rates flat but the LLM-judge faithfulness score on RAG-grounded answers dropped, which — cross-referenced against the retrieval-index-version metadata captured on each span — reveals that a routine reindex on Monday night changed chunking parameters and degraded retrieval quality, even though nothing about the agent's code or prompts changed. Neither diagnosis is reachable from an aggregate satisfaction score alone; both are mechanical once span-level metadata (tool identity, error status, prompt/index version) is queryable across the affected time window. This is the concrete payoff of the trace schema described above: a metric regression becomes a groupable, filterable query instead of a mystery requiring speculative rollbacks to isolate.

```python
def diagnose_regression(traces_before: list[dict], traces_after: list[dict]) -> dict:
    """Compare two time windows across every dimension the trace schema
    captures, to localize a regression instead of guessing at root cause."""
    before = TraceAnalyzer(traces_before)
    after = TraceAnalyzer(traces_after)

    return {
        "error_rate_delta": {
            tool: after.error_rate_by_tool().get(tool, 0) - rate
            for tool, rate in before.error_rate_by_tool().items()
        },
        "latency_delta_p95": {
            span_type: after.latency_by_span_type().get(span_type, {}).get("p95", 0)
                       - stats.get("p95", 0)
            for span_type, stats in before.latency_by_span_type().items()
        },
        "cost_delta_by_model": {
            model: after.cost_by_model().get(model, 0) - cost
            for model, cost in before.cost_by_model().items()
        },
    }
```

## Real-Time Alerting on Trace Anomalies

Dashboards answer questions someone remembers to ask; alerting catches regressions nobody was actively watching for, and it's the difference between finding out about a quality drop from a metrics review versus finding out from an angry customer. The alerting rules worth setting up mirror the aggregation dimensions covered above rather than being generic: a spike in error rate for a specific tool (suggesting an upstream dependency issue), a drop in the rolling sampled LLM-judge or faithfulness score for a specific agent flow (suggesting a prompt, model, or retrieval regression), a spike in average trajectory length or step count for a given task type (suggesting the agent has started looping or struggling on a class of input it previously handled efficiently), and a spike in per-request cost (which can indicate anything from a runaway retry loop to a model routing bug sending traffic to a more expensive model than intended).

```python
class TraceAlertMonitor:
    def __init__(self, thresholds: dict):
        self.thresholds = thresholds  # e.g. {"tool_error_rate": 0.1, "faithfulness_drop": 0.15}
        self.baseline = {}

    def set_baseline(self, analyzer: "TraceAnalyzer"):
        self.baseline = {
            "error_rate_by_tool": analyzer.error_rate_by_tool(),
            "cost_by_model": analyzer.cost_by_model(),
        }

    def check(self, analyzer: "TraceAnalyzer") -> list[dict]:
        alerts = []
        current_errors = analyzer.error_rate_by_tool()
        for tool, rate in current_errors.items():
            baseline_rate = self.baseline.get("error_rate_by_tool", {}).get(tool, 0)
            if rate - baseline_rate > self.thresholds.get("tool_error_rate", 0.1):
                alerts.append({
                    "type": "tool_error_spike",
                    "tool": tool,
                    "baseline": baseline_rate,
                    "current": rate,
                })

        current_cost = analyzer.cost_by_model()
        for model, cost in current_cost.items():
            baseline_cost = self.baseline.get("cost_by_model", {}).get(model, 0)
            if baseline_cost and (cost - baseline_cost) / baseline_cost > 0.5:
                alerts.append({
                    "type": "cost_spike",
                    "model": model,
                    "baseline": baseline_cost,
                    "current": cost,
                })

        return alerts
```

Both LangSmith and Langfuse support this pattern natively (threshold-based alerts wired to scoring runs and to raw span metrics), and Phoenix's OpenTelemetry foundation means the same alerting can be built on top of whatever metrics/alerting stack (Grafana, Datadog) a platform team already operates, by exporting span metrics through the standard OTel pipeline rather than requiring a separate bespoke alerting path just for LLM traffic.

## Tool Landscape: LangSmith, Langfuse, and Arize/Phoenix

**LangSmith**, built by the LangChain team, is the most tightly integrated option if you're already building on LangChain or LangGraph, since tracing is close to automatic — spans are emitted natively as your chains and graphs execute, with prompt/completion capture, token/cost accounting, and the full nested call tree requiring little to no manual instrumentation. Beyond tracing, LangSmith ships dataset management (curate examples from production traces directly into an eval set, closing the loop described in the task-specific-eval-set section of the first chapter) and built-in evaluators, including trajectory evaluators that operate directly on the run tree rather than requiring you to reconstruct trajectory objects yourself. Its main trade-off is that its value is highest when your stack is LangChain-native; integrating it with a fully custom agent loop is possible via its SDK but requires more manual span instrumentation, and it's a commercial hosted product (with usage-based pricing) rather than something you self-host by default.

**Langfuse** is the strongest open-source-first alternative, offering a comparable tracing and evaluation feature set (nested traces, prompt management with versioning, dataset-based evals, cost tracking) while being self-hostable, which matters a great deal for teams with data-residency or compliance constraints that make sending prompt/completion payloads to a third-party SaaS a non-starter. It's framework-agnostic by design — it doesn't assume LangChain — and integrates via a straightforward SDK or via OpenTelemetry-compatible instrumentation, which makes it a natural fit for custom or non-LangChain agent stacks. Its prompt-management feature (versioned prompts fetched at runtime rather than hardcoded, with the version used automatically linked to every trace) directly addresses the "which prompt version produced this trace" problem raised earlier.

**Arize Phoenix** (the open-source counterpart to Arize's commercial ML observability platform) approaches the problem with a stronger lineage in traditional ML monitoring — drift detection, embedding-space visualization of retrieval quality, and evaluation frameworks originally built for classical ML models extended to LLM-specific use cases like RAG evaluation (it has particularly strong tooling for visualizing retrieval quality: plotting query and document embeddings together to visually inspect whether retrieval is pulling genuinely relevant chunks). It's built on OpenTelemetry (via OpenInference, Arize's semantic-convention extension for LLM spans), which means it interoperates cleanly with broader observability infrastructure a platform team might already run, rather than requiring a dedicated LLM-only pipeline.

There are also strong adjacent options worth knowing by name even without deep coverage here: **Weights & Biases Weave** extends W&B's long-standing ML experiment tracking into LLM tracing and evaluation, appealing to teams already using W&B for model training. **Helicone** and **Portkey** position themselves more as an LLM gateway/proxy with observability as a byproduct of sitting in the request path, which gets you tracing with near-zero code change (point your API calls at the proxy) at the cost of less deep native evaluation tooling than LangSmith/Langfuse/Phoenix. **OpenTelemetry with the GenAI semantic conventions** (an emerging standard for representing LLM spans) is the vendor-neutral foundation increasingly underlying several of these tools, and betting on OTel-compatible instrumentation reduces vendor lock-in risk if you expect to switch observability backends later.

## What to Look for When Choosing a Tool

Reduce the decision to a small number of concrete questions rather than a feature checklist, because most of these tools' feature lists overlap heavily on paper and the real differentiators show up in day-to-day use. **Does it integrate with your actual stack with minimal instrumentation burden?** A tool that requires hand-wrapping every LLM and tool call with manual span calls will accrue engineering cost and instrumentation drift (someone adds a new tool call and forgets to instrument it) faster than one with native framework integration or automatic instrumentation via a proxy/SDK wrapper. **Can it capture full prompt/completion payloads, and does your compliance posture allow sending those payloads to a third-party SaaS, or does it need to be self-hosted?** This single question eliminates otherwise-strong candidates for teams in regulated industries or with strict data-residency requirements, and it's worth resolving before investing engineering time in an integration. **Does it support the evaluation workflows you actually need** — dataset curation from production traces, LLM-as-judge scoring wired directly into traces, trajectory-level evaluation for agents — rather than just being a pretty trace viewer, since a trace viewer without a path to systematic scoring only helps you debug incidents one at a time rather than tracking quality as a trend. **How does cost tracking work at scale**, specifically whether it correctly attributes cost across multi-step agent runs and across different models/providers if you're multi-sourcing, since a tool that only tracks cost per top-level request (rather than per span) will hide exactly the "which step is burning the budget" signal that matters for optimization. **What's the query and alerting story** — can you set up an alert when p95 latency on a specific tool spikes, or when the sampled quality score on a specific agent flow drops below a threshold, without exporting data to a separate system to build that alert yourself; a tool that's purely a passive log viewer pushes that operational work back onto your team.

None of these tools substitute for the evaluation techniques covered in the earlier chapters of this section — LLM-as-judge scoring, trajectory evaluation, hallucination/faithfulness checking, and guardrail pass/fail logging. What a good observability platform does is give those techniques a place to live inside the actual request path, attached to real production traffic, with the trace context (exact prompt, exact retrieved documents, exact tool arguments) needed to make the resulting scores actionable rather than abstract. The chapters in this section build the evaluation logic; this chapter's tooling is what turns that logic from a one-off offline script into a continuously running production quality signal.
