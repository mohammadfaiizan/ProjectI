# Scalability, Latency, and Cost Trade-offs

## Why Agentic Systems Are Different From a Single LLM Call

Reasoning about latency and cost for a single prompt-response API call is simple: one network round
trip, one model's time-to-first-token plus generation time, one line item on the bill. An agentic
system breaks that simplicity in a specific way — it chains multiple LLM calls and tool invocations
together, often sequentially, sometimes conditionally, and the end-to-end latency and cost are the
*sum* (or worse, in pathological cases the product) of everything in that chain, not a single number
you can look up in a pricing table. A five-step agent loop where each step takes 2 seconds isn't a
2-second system; it's a 10-second system, and if any step occasionally retries, it's a system with a
long tail that can hit 30+ seconds. This chapter is about developing the muscle to reason about that
accumulation quantitatively, and about the concrete techniques that pull latency and cost back down
without gutting capability.

This is consistently one of the more differentiating parts of a system-design interview for agentic
AI, because it's easy to design a plausible-looking architecture and much harder to reason correctly
about where the actual time and money go once it's running at scale. Interviewers will often ask
"how would this scale to 10x traffic" or "walk me through the latency budget for one request"
specifically to see if you can do this kind of accounting rather than gesture at it.

## Where Latency Accumulates in an Agent Loop

Break down a typical multi-step agent turn into its components and it becomes clear there are
several independent places where time is spent, each with different characteristics and different
levers to pull.

**Time-to-first-token (TTFT) per LLM call.** Every call to a large model has a fixed-ish startup
cost before the first output token appears — this depends on input length (longer prompts take
longer to process before generation starts, because the model has to run the full context through
its layers before it can begin producing tokens) and on provider-side queuing under load. For a
frontier model with a long context (tens of thousands of tokens), TTFT alone can be 500ms-2s, before
a single output token is streamed.

**Generation time (tokens/sec).** Once generation starts, output tokens stream at a roughly constant
rate for a given model and load level — commonly in the range of 30-100+ tokens/sec for large
models, though this varies significantly by provider, model size, and current load. A 500-token
response at 50 tokens/sec is another 10 seconds if you wait for it to complete rather than streaming
it to the user incrementally.

**Tool call latency.** Any external call the agent makes — a database query, a search API, a code
sandbox execution, an internal microservice — has its own latency distribution, and agent loops
typically treat tool calls as *blocking*: the agent waits for the tool result before it can decide
what to do next. A tool that's fast on average (200ms) but has a heavy tail (occasional 5-second
timeouts against a flaky downstream service) will dominate your p99 latency even though it looks
fine on average.

**Sequential chaining.** This is the multiplier that makes agent latency qualitatively different
from single-call latency. If a task requires "retrieve context, then plan, then call three tools one
after another, then synthesize a final answer," and each of those steps is itself an LLM call plus a
tool call, you're chaining 5-8 round trips end to end. At roughly 1-3 seconds per round trip, that's
5-25 seconds of latency for a single user-facing action — before accounting for any retries.

**Retries and re-planning.** When a tool call fails transiently, or the model produces an output
that fails validation and has to be re-prompted, the loop doesn't just add the retry's latency — it
adds the *original* attempt's latency too, since that time was already spent before the failure was
detected. A step with a 5% failure rate and a 2-second cost per attempt adds, on average, a small
but non-trivial tax to every single request that passes through it, and a much larger tax to the
unlucky 5%.

The mental model worth internalizing: latency in an agent system is not one number, it's a chain of
distributions, and the end-to-end latency distribution is shaped much more by the slowest, most
variable step in the chain and by how many steps are strictly sequential than by the average speed
of any individual component.

## Technique 1: Parallelize Independent Tool Calls

The most direct lever against sequential chaining is recognizing when steps don't actually depend on
each other and executing them concurrently instead of one after another. If an agent needs to check
inventory, look up shipping cost, and verify a coupon code to answer "can I get this shipped by
Friday with my discount," those three lookups have no data dependency on each other — there's no
reason to run them sequentially just because the agent "thought of them" in that order during
planning.

```python
import asyncio

async def gather_tool_results(tool_calls: list[ToolCall]) -> list[ToolResult]:
    # tool_calls that the planner has already determined are independent
    tasks = [execute_tool_async(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def execute_tool_async(tc: ToolCall) -> ToolResult:
    try:
        return await asyncio.wait_for(tc.run(), timeout=tc.timeout_s)
    except asyncio.TimeoutError:
        return ToolResult.failure(tc.id, reason="timeout")
```

The catch is that this requires the planning step to explicitly identify which calls are independent
— most modern LLM APIs support returning multiple tool calls from a single model turn, which pushes
the dependency analysis onto the model (it lists the calls it wants made together), while the
orchestrator's job is simply to execute whatever the model batches together in parallel and feed all
the results back in the next turn. When calls genuinely do depend on each other (you need the order
ID from lookup A before you can call lookup B), no amount of engineering removes that sequential
dependency — the fix there, if it matters enough, is redesigning the tools themselves (e.g., a
combined endpoint) rather than trying to force parallelism where a real dependency exists.

## Technique 2: Model Cascading and Routing

Not every step in an agent loop needs the same model. A classification step ("is this message a
refund request or a shipping question"), a validation step ("does this output match the expected
JSON schema"), or a simple extraction step ("pull the order number out of this message") can run on
a small, fast model — or sometimes no LLM at all, just a regex or a lightweight classifier — while
the step that actually requires multi-step reasoning or nuanced judgment justifies the larger,
slower, more expensive model.

```python
def route_step(step_type: str, payload: dict):
    if step_type in {"intent_classification", "schema_validation", "entity_extraction"}:
        return small_fast_model.call(payload)          # ~100-300ms, cheap
    if step_type in {"final_synthesis", "complex_reasoning", "multi_tool_planning"}:
        return large_model.call(payload)                # ~1-3s, expensive
    return small_fast_model.call(payload)               # default to cheap; escalate if it fails
```

A more sophisticated version of this is **cascading with escalation**: try the cheap path first, and
only fall back to the expensive model if the cheap model's output fails a confidence check or a
validation step. This gets you the latency and cost of the cheap model for the (often large)
fraction of cases it handles correctly, while preserving correctness for the harder tail by
escalating — the trade-off being that the escalated cases now pay for *both* calls, so this only
wins if the cheap model has a high enough success rate that the savings on the easy majority
outweigh the double-cost on the escalated minority. Whether that trade is worth it is an empirical
question you'd want to validate with actual traffic data, not assume.

## Technique 3: Caching at Multiple Layers

Caching in agentic systems operates at several distinct layers, and it's worth naming them
separately because they have different hit rates and different implementation mechanics.

**Prompt/prefix caching** exploits the fact that agent loops re-send a large, unchanging prefix
(system prompt, tool definitions, retrieved context) on every turn of a multi-turn conversation,
with only the new turn's content changing. Most major model providers now offer some form of prompt
caching that skips reprocessing the unchanged prefix, cutting both cost and TTFT substantially for
exactly this pattern — and agent loops, which resend a growing but largely-stable context on every
iteration, are close to the ideal use case for it. Structuring prompts so the stable parts (system
instructions, tool schemas, static context) come first and the variable parts (the latest turn) come
last is what makes this caching effective; interleaving stable and variable content defeats it.

**Semantic response caching** stores past query-response pairs (embedded for similarity search) and
serves a cached answer when a new query is semantically close enough to one already answered —
useful for high-volume, repetitive queries like FAQ-style support questions, much less useful for
agentic tasks where the "same" surface question can require a genuinely different answer depending
on account-specific state.

**Tool result caching** caches the output of expensive or rate-limited tool calls (a slow database
aggregation, a third-party API with a per-call cost) with an appropriate TTL based on how quickly
that data goes stale — a product catalog lookup might be cacheable for hours, while an account
balance should not be cached at all, or only for seconds.

```python
class ToolResultCache:
    def __init__(self, backend, default_ttl_s=300):
        self.backend = backend
        self.default_ttl_s = default_ttl_s

    def get_or_call(self, tool_name: str, args: dict, ttl_s: int = None):
        key = f"{tool_name}:{stable_hash(args)}"
        cached = self.backend.get(key)
        if cached is not None:
            return cached
        result = call_tool(tool_name, args)
        self.backend.set(key, result, ttl=ttl_s or self.default_ttl_s)
        return result
```

The general principle across all three cache types: cache aggressively where staleness is cheap and
correctness-tolerant, and refuse to cache (or cache for seconds, not hours) where staleness is
expensive or where it could cause the agent to act on wrong information — a refund decided against a
stale account balance is a much worse outcome than a slightly slow cache-miss lookup.

## Technique 4: Speculative Execution

Speculative execution in an agent context means starting work on a likely next step before you're
certain it's needed, and discarding it if a different path is taken. A concrete example: while the
model is still generating its plan for a multi-step task, you can speculatively kick off the
retrieval call for the most likely next tool (based on the pattern of similar past tasks) in
parallel with waiting for the plan to finish generating — if the plan confirms that tool is needed,
the result is already available; if not, you discard it at the cost of some wasted compute. This
trades a modest amount of wasted work, paid on the cases where the speculation is wrong, for a
latency win on the cases where it's right, and it's most worth doing for the *first* tool call after
a planning step, since that's often the single largest sequential latency hop in the whole loop
(plan-then-act is inherently serial otherwise). It's a technique to reach for once the simpler wins
— parallelizing genuinely independent calls, routing to smaller models, caching — are already in
place, since it adds real implementation complexity (you need cheap, reliable prediction of the
likely next step, and clean cancellation of discarded speculative work) for a comparatively narrow
latency win.

## Technique 5: Streaming as a Latency-Perception Tool

Streaming doesn't reduce the total time a model takes to generate a response, but it changes what
the user experiences: instead of waiting for the full answer before seeing anything, tokens appear
as they're produced, and perceived latency becomes dominated by time-to-first-token rather than
total generation time. For agent loops that end in a user-facing synthesis step, always stream that
final step. For intermediate steps the user doesn't see directly (an internal planning call, a
tool-selection call), streaming doesn't help perception and isn't worth the added complexity —
reserve it for the parts of the loop actually rendered to a human.

## The Cost Math: Tokens x Steps x Price

Cost in an agentic system is best reasoned about as a simple product, decomposed per step and then
summed across the loop: for each step, cost equals (input tokens + output tokens) times that model's
per-token price, and total request cost is the sum of that across every step in the loop, including
retried steps.

Walk a concrete example. Suppose a task takes 6 steps: a planning call (15K input tokens including
context, 500 output tokens, on a frontier model), three tool-selection/argument calls (2K input, 100
output each, on a small model), and a final synthesis call (5K input, 800 output, on the frontier
model). At illustrative pricing of $3/$15 per million input/output tokens for the frontier model and
$0.15/$0.60 per million for the small model:

```
Planning:    15,000 * $3/1e6  +   500 * $15/1e6  = $0.045 + $0.0075 = $0.0525
Tool calls:  (2,000 * $0.15/1e6 + 100 * $0.60/1e6) * 3
           = ($0.0003 + $0.00006) * 3            = $0.00108
Synthesis:    5,000 * $3/1e6  +   800 * $15/1e6  = $0.015  + $0.012  = $0.027
                                                    -----------------
                                            Total  ≈ $0.081 per request
```

That looks negligible per request — and it is, at low volume. The number that matters is what
happens when you multiply it by realistic production traffic: at 100,000 requests a day,
$0.081/request is roughly $8,100/day, about $243K/month, before accounting for retries, failed steps
that still consumed tokens, or growth in traffic. This is exactly the calculation an interviewer
wants to see you do unprompted, because it's the difference between an architecture that looks fine
on a whiteboard and one that survives a finance review. It's also why the routing and caching
techniques above aren't just latency optimizations — cutting the planning step from a frontier model
to a mid-tier model, or caching the tool-selection step for repeated query patterns, can plausibly
cut that $243K/month by half or more, which is a very different kind of impact than a purely
architectural cleanup would produce.

## The Latency-Cost-Quality Triangle

Every technique in this chapter is ultimately trading between three things: how fast the system
responds, how much it costs to run, and how good the output is — and you generally cannot improve
all three simultaneously with the same lever. A smaller, cheaper model is faster and cheaper but may
need more retries or produce lower-quality output on hard tasks, potentially erasing the savings.
More parallel tool calls reduce latency but can increase cost (you're now always paying for the
"just in case" call) and, if not carefully bounded, can increase the chance of the agent acting on
stale or now-irrelevant partial results. More caching reduces both latency and cost but trades away
freshness, which is a quality risk if the cached content can go stale in a way that matters. A good
system-design answer names this triangle explicitly and argues for where the specific product should
sit on it, rather than presenting one technique as a universal win — the right cache TTL for a
product catalog lookup is not the right cache TTL for an account balance, and the right model tier
for a lint-fix codemod is not the right model tier for a multi-file refactor, precisely because the
acceptable trade-off between latency, cost, and quality is different in each case.

## Reasoning About Scale in an Interview

When asked "how does this scale to 10x," the strongest answers separate the question into distinct
sub-questions rather than answering it as one thing: does inference capacity scale (are you
rate-limited by a model provider, and does 10x traffic need reserved throughput or a multi-provider
fallback strategy)? Does the retrieval/data layer scale (a vector index or database that's fine at
current QPS may need sharding or read replicas at 10x)? Does cost scale linearly, worse than
linearly, or can it scale sub-linearly (caching hit rates typically improve with volume, since more
requests means more repeated patterns to cache against, which is one of the few effects that gets
*better* with scale rather than worse)? And does the human-in-the-loop or escalation layer scale (if
2% of requests need human review, 10x traffic means 10x the review volume, and human review capacity
essentially never scales for free) — the last point is often the one candidates forget, and it's
usually the one that most concretely bounds the system's real-world scaling ceiling.

