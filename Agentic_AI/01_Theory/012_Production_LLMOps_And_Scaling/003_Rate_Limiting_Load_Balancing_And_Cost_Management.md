# Rate Limiting, Load Balancing, and Cost Management

## Why This Is a Distinct Discipline for LLM Systems

Traditional backend systems treat rate limiting and load balancing as largely separate concerns: rate limiting protects your service from being overwhelmed by callers, and load balancing spreads your own traffic across your own fleet. LLM applications collapse these into a single, harder problem, because the thing you're rate-limited by and load-balancing across is usually a third party you don't control — a model provider with its own quotas, its own outages, and its own pricing that can change your unit economics overnight. On top of that, "cost" in an LLM system isn't a fixed infrastructure line item you can forecast from historical server counts; it's a variable, usage-driven, per-token number that scales directly with how many users show up and how verbosely both they and the model respond. Getting this right is as much a cost-engineering problem as it is a reliability problem, and the two are more tightly coupled here than almost anywhere else in software.

## Provider Rate Limits and Multi-Key Pooling

Every major LLM provider enforces quotas along at least two axes simultaneously: requests per minute (RPM) and tokens per minute (TPM), sometimes with a third limit on concurrent requests. These limits exist because token generation consumes real, scarce GPU capacity on the provider's side, and they are enforced per API key (or per organization/project, depending on the provider), which means the most direct lever available to you as a customer is having more than one key to spread load across, if your provider agreement allows it and your volume justifies the operational overhead.

A key pool needs to do more than round-robin, though, because keys can be in different states at different times — one might be near its TPM ceiling for the current minute while another is fresh. A reasonably sophisticated pool tracks recent usage per key and routes new requests to whichever key has the most remaining headroom, rather than blindly cycling through keys and hoping for the best.

```python
import time
from collections import deque

class ApiKeyPool:
    def __init__(self, keys, rpm_limit, tpm_limit, window_seconds=60):
        self.keys = keys
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window = window_seconds
        # Per key: rolling log of (timestamp, tokens) for requests in the current window
        self.usage = {k: deque() for k in keys}

    def _prune(self, key):
        cutoff = time.time() - self.window
        log = self.usage[key]
        while log and log[0][0] < cutoff:
            log.popleft()

    def _headroom(self, key):
        self._prune(key)
        log = self.usage[key]
        requests_used = len(log)
        tokens_used = sum(t for _, t in log)
        return (self.rpm_limit - requests_used, self.tpm_limit - tokens_used)

    def select_key(self, estimated_tokens):
        candidates = []
        for key in self.keys:
            req_headroom, tok_headroom = self._headroom(key)
            if req_headroom > 0 and tok_headroom >= estimated_tokens:
                candidates.append((key, tok_headroom))
        if not candidates:
            return None  # every key is saturated, caller must queue or backoff
        # Prefer the key with the most token headroom, spreading load evenly
        return max(candidates, key=lambda kv: kv[1])[0]

    def record_usage(self, key, tokens_used):
        self.usage[key].append((time.time(), tokens_used))
```

Estimating `estimated_tokens` ahead of the call (a rough tokenizer count on the input, plus a conservative assumption about output length based on `max_tokens`) is what makes this proactive rather than reactive — you want to avoid sending a request that you can already tell will blow a key's TPM ceiling, rather than discovering that after the provider rejects it. That said, no amount of proactive estimation eliminates 429s entirely (usage estimates are approximate, and limits can be tightened without notice during provider-side incidents), so a pool always needs to be paired with proper backoff handling.

## Handling 429s: Backoff Done Correctly

The naive response to a rate-limit error is to retry immediately — which is exactly the wrong move, since immediate retries from every failed caller synchronize into a thundering herd that keeps hitting the same ceiling. The standard fix is exponential backoff with jitter: each retry waits roughly double the previous wait, with some randomness mixed in so that many clients backing off from the same event don't all retry at the same instant.

```python
import random
import asyncio

async def call_with_backoff(fn, *args, max_retries=5, base_delay=1.0, max_delay=30.0, **kwargs):
    for attempt in range(max_retries):
        try:
            return await fn(*args, **kwargs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Respect an explicit Retry-After header if the provider sends one --
            # it's more accurate than any backoff formula you could guess.
            retry_after = getattr(e, "retry_after_seconds", None)
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(max_delay, base_delay * (2 ** attempt))
            delay += random.uniform(0, delay * 0.25)   # jitter
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")
```

The one detail that separates a correct implementation from a merely plausible one is checking for a provider-supplied `Retry-After` (or equivalent) header before falling back to a computed backoff — providers that return this are telling you exactly how long their rate limit window has left, and guessing with exponential backoff when you have that information is strictly worse than using it directly.

## Load Balancing Across Models and Providers

Rate limits are one reason to spread traffic across more than one provider; reliability and cost are two more, and a mature system treats all three as inputs to the same routing decision rather than three separate mechanisms bolted together. If OpenAI has an outage, a system that can fail over to Anthropic or a self-hosted model keeps serving users; if one provider's price for equivalent quality on a given task is materially higher, routing volume to the cheaper option saves money without a quality trade-off; and if you've already built the fallback chain for reliability, extending it to include the cost signal is nearly free.

```python
class ProviderRouter:
    def __init__(self, providers):
        # providers: list of dicts with name, client, cost_per_1k_tokens,
        # recent_latency_ms, recent_error_rate
        self.providers = providers

    def healthy(self, p):
        return p["recent_error_rate"] < 0.05

    def select(self, strategy="cost"):
        candidates = [p for p in self.providers if self.healthy(p)]
        if not candidates:
            raise RuntimeError("all providers unhealthy")

        if strategy == "cost":
            return min(candidates, key=lambda p: p["cost_per_1k_tokens"])
        if strategy == "latency":
            return min(candidates, key=lambda p: p["recent_latency_ms"])
        if strategy == "weighted_round_robin":
            weights = [1.0 / p["cost_per_1k_tokens"] for p in candidates]
            total = sum(weights)
            r = random.uniform(0, total)
            upto = 0
            for p, w in zip(candidates, weights):
                upto += w
                if upto >= r:
                    return p
        return candidates[0]

    async def generate(self, messages, strategy="cost"):
        tried = set()
        while len(tried) < len(self.providers):
            provider = self.select(strategy)
            if provider["name"] in tried:
                continue
            tried.add(provider["name"])
            try:
                return await provider["client"].generate(messages)
            except Exception:
                provider["recent_error_rate"] = min(1.0, provider["recent_error_rate"] + 0.1)
                continue
        raise RuntimeError("all providers failed")
```

A subtlety worth flagging explicitly: cost-based and latency-based routing across *different model families* is not a free substitution the way it is across identical replicas of the same model in a normal load balancer, because different models have genuinely different quality profiles. Routing a request to a cheaper provider purely on cost, with no regard for whether that provider's model can actually do the task well, trades a cost win for a quality loss that your users will notice even if your dashboards don't — which is why cost/latency-based provider routing should generally only rotate between models you've already validated as roughly quality-equivalent for the task at hand (via your eval suite), not between models of meaningfully different capability tiers. That second kind of routing — deliberately picking a *weaker* model because the task doesn't need a strong one — is a different, valuable technique, but it should be an explicit difficulty-based decision, not an accidental side effect of a cost-optimizing load balancer.

## Token Budgets and Spend Forecasting

Cost control in LLM systems starts with the recognition that "cost per request" is not a constant — it varies with input length, output length, and which model handled the request, so aggregate spend is a distribution, not a fixed number, and forecasting it means forecasting that distribution's shape as volume grows, not just multiplying a flat unit cost by a request count. The most direct lever is a hard or soft token budget enforced at the point of the request — capping `max_tokens` per call, capping tokens per user session, or capping tokens per organization per billing period — because unbounded output length is the single most common source of runaway cost in agentic systems specifically, where a model stuck in a reasoning loop or a tool-calling cycle can generate far more tokens than any single "answer" should ever require.

```python
class TokenBudgetManager:
    def __init__(self, per_request_max=2000, per_session_max=50_000, per_org_daily_max=5_000_000):
        self.per_request_max = per_request_max
        self.per_session_max = per_session_max
        self.per_org_daily_max = per_org_daily_max
        self.session_usage = {}     # session_id -> tokens used
        self.org_daily_usage = {}   # (org_id, date) -> tokens used

    def check_and_reserve(self, org_id, session_id, date_str, estimated_tokens):
        if estimated_tokens > self.per_request_max:
            raise BudgetExceeded("single request exceeds per-request token cap")

        session_used = self.session_usage.get(session_id, 0)
        if session_used + estimated_tokens > self.per_session_max:
            raise BudgetExceeded("session token budget exhausted")

        org_key = (org_id, date_str)
        org_used = self.org_daily_usage.get(org_key, 0)
        if org_used + estimated_tokens > self.per_org_daily_max:
            raise BudgetExceeded("organization daily token budget exhausted")

        self.session_usage[session_id] = session_used + estimated_tokens
        self.org_daily_usage[org_key] = org_used + estimated_tokens
```

Forecasting spend at the organization level is then a matter of projecting this same per-request cost distribution forward against expected volume growth, ideally broken down by the same dimensions you'd want visibility into for debugging a spend spike later: by model, by endpoint/feature, and by customer tier if you have usage-based pricing tiers of your own. The output of a good forecast isn't a single number but a budget curve with confidence bands, because LLM usage is bursty (a single customer running a large batch job can dwarf steady-state traffic for a day), and a forecast that only reports the mean will consistently under-provision for the tail.

## Model Routing and Tiering by Task Difficulty

The single highest-leverage cost optimization available in most LLM products is recognizing that not every request needs the most capable (and most expensive) model. A classification task, a simple FAQ lookup, or a well-structured data extraction job can often be handled by a small, cheap model at quality parity with a frontier model, while genuinely hard multi-step reasoning or open-ended generation tasks need the frontier model's capability to avoid a visible quality drop. Model routing formalizes this: classify the incoming request's difficulty (or task type) cheaply, and dispatch to the cheapest model tier that can handle it, reserving the expensive model for the requests that actually need it.

```python
class DifficultyRouter:
    def __init__(self, cheap_model, mid_model, strong_model, classifier):
        self.tiers = {"simple": cheap_model, "moderate": mid_model, "complex": strong_model}
        self.classifier = classifier   # a fast, cheap model or a lightweight rule-based heuristic

    def classify(self, request):
        # Cheap heuristics first -- avoid spending an LLM call just to route another LLM call
        # wherever a rule can do the job (length, presence of code, known intent categories).
        if request.get("known_intent") in {"faq_lookup", "simple_classification"}:
            return "simple"
        if len(request["message"]) > 2000 or request.get("requires_multi_step_reasoning"):
            return "complex"
        # Fall back to a cheap classifier model only when heuristics are inconclusive
        label = self.classifier.classify(request["message"])
        return label

    def route(self, request):
        difficulty = self.classify(request)
        return self.tiers[difficulty]


def handle_request(request, router):
    model = router.route(request)
    return model.generate(request["message"])
```

Two things make this pattern work reliably in production rather than just look good in a design doc. First, the classification step itself must be cheap relative to the savings it enables — if you burn a full LLM call on a strong model just to decide whether to route to a cheap model, you've eaten most of the savings before the "real" call even happens, which is why rule-based heuristics (message length, presence of code blocks, known intent taxonomies from your product's own logs) should catch as many cases as possible before falling back to a lightweight classifier model. Second, the router needs an escape hatch: a monitoring loop that samples "simple" requests routed to the cheap tier and checks (via spot-check human review or an LLM-judge pass) whether the cheap model's answers are actually holding up in quality, because difficulty classification drifts as your product's request mix changes, and a router tuned against last quarter's traffic can silently start sending genuinely hard requests to a model that can't handle them.

## Batch APIs for Non-Interactive Workloads

Not every LLM call is on the critical path of a user waiting for a response. Nightly data-enrichment jobs, bulk classification of historical records, or offline eval-suite runs can tolerate latency measured in hours rather than seconds, and most major providers offer a batch API specifically for this case, at a substantial discount (commonly 50% off standard pricing) in exchange for asynchronous, best-effort-by-a-deadline processing rather than immediate response. Any workload that doesn't have a human waiting in the loop is a candidate for the batch API by default, and treating "does a human need this synchronously" as an explicit design question — rather than defaulting every call to the synchronous API out of habit — is one of the easiest wins available in a cost review.

## Putting It Together: A Full Request Pipeline

A production system composes all of these into one pipeline: a request first passes through the difficulty router to pick a model tier, then through the token budget manager to confirm the org/session hasn't exceeded its allotment, then to the provider router to pick a healthy, cost-appropriate provider for that model tier, and finally through the key pool and backoff logic to actually execute the call reliably.

```python
async def production_generate(request, difficulty_router, budget_manager,
                                provider_router, key_pool):
    model_tier = difficulty_router.route(request)

    estimated_tokens = estimate_tokens(request["message"]) + request.get("max_tokens", 500)
    budget_manager.check_and_reserve(
        request["org_id"], request["session_id"], today_str(), estimated_tokens
    )

    key = key_pool.select_key(estimated_tokens)
    if key is None:
        raise RuntimeError("no API key headroom available, request must queue")

    async def do_call():
        return await provider_router.generate(request["message"], strategy="cost")

    response = await call_with_backoff(do_call)
    key_pool.record_usage(key, response.usage.total_tokens)
    return response
```

Every stage in this pipeline exists to answer one of the same three underlying questions: are we about to exceed a quota, are we spending appropriately for the difficulty of the task, and are we resilient to any single provider having a bad day. Keeping those three concerns explicit and composable, rather than tangled into one ad hoc retry loop, is what makes the system debuggable when spend or reliability drifts — you can look at each stage's metrics independently and know immediately whether a problem is a quota issue, a routing issue, or a provider issue.
