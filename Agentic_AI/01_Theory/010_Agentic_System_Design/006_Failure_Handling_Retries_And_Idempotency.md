# Failure Handling, Retries, and Idempotency

## Why This Matters More for Agents Than for Typical Web Services

Every distributed system has to deal with transient failures — a request times out, a downstream
service returns a 503, a network blip drops a connection partway through. Standard practice for
handling this is retries, and most engineers have internalized "retry with backoff" as a reflex.
Agentic systems make this harder in a specific way: an agent doesn't just make read requests, it
takes actions with real-world side effects — charging a card, sending an email, creating a support
ticket, modifying a file, deploying code — and an LLM in the decision loop introduces a second,
independent source of unreliability on top of ordinary infrastructure flakiness (the model can time
out, return malformed output, or occasionally make a bad tool-call decision that needs to be treated
as a "failure" and retried differently than a network error would be). When you retry a
side-effecting action without care, you risk doing it twice: charging the card twice, sending the
email twice, opening two duplicate tickets. This is why idempotency isn't a nice-to-have for agent
systems, it's foundational — you cannot safely retry anything with side effects unless retrying it
twice is provably harmless.

## Idempotency Keys: Making Retries Safe

An idempotency key is a unique identifier attached to a specific *intended* action, generated once
when the action is first decided upon, and passed along with every attempt (including retries) to
execute that action. The receiving system (whether that's your own backend or a third-party API)
checks whether it has already processed an action with that key, and if so, returns the previous
result instead of executing the action again.

The key design decision is *where the key is generated and what it's derived from*. If the agent
generates a fresh random key every time it calls a tool, retries defeat the entire mechanism — a
genuine retry needs to carry the *same* key as the original attempt, meaning the key has to be
generated once, at the point where the *decision* to take this action is made, and then threaded
through every retry of that same decision. A common and robust way to do this is to derive the key
deterministically from the action's semantic content (e.g., a hash of the order ID, action type, and
amount, possibly combined with a request-scoped identifier like a conversation turn ID) rather than
a random UUID — this way, even if the calling code itself crashes and restarts and ends up
re-deciding "yes, refund this order" from scratch, it derives the *same* key it would have generated
the first time, and the downstream system correctly recognizes it as a duplicate rather than a new
action.

```python
import hashlib
import json

def make_idempotency_key(action_type: str, args: dict, scope_id: str) -> str:
    """Deterministic key: same logical action always produces the same key,
    whether it's a retry of the same call or a re-decision after a crash."""
    canonical = json.dumps(args, sort_keys=True)
    raw = f"{action_type}:{scope_id}:{canonical}"
    return hashlib.sha256(raw.encode()).hexdigest()


class IdempotentActionExecutor:
    def __init__(self, action_log_store):
        self.action_log_store = action_log_store

    def execute(self, action_type: str, args: dict, scope_id: str, fn):
        key = make_idempotency_key(action_type, args, scope_id)
        existing = self.action_log_store.get(key)

        if existing is not None:
            if existing.status == "completed":
                return existing.result          # already done -- return cached result
            if existing.status == "in_progress":
                raise ConcurrentExecutionError(key)  # another attempt is live right now

        self.action_log_store.mark_in_progress(key, action_type, args)
        try:
            result = fn(**args)
            self.action_log_store.mark_completed(key, result)
            return result
        except Exception as e:
            self.action_log_store.mark_failed(key, str(e))
            raise
```

Two things about this pattern are easy to get wrong in practice. First, the idempotency check and
the "mark in progress" write need to be atomic (a single conditional write, or a database
transaction), otherwise two concurrent retries can both pass the "does this exist" check before
either has written its own record, and you're back to a duplicate execution — this is the same class
of race condition as a distributed lock, and it needs the same care. Second, when calling
third-party APIs (a payment processor, an email provider), check whether the API itself supports an
idempotency key parameter natively — most mature payment and messaging APIs do — and pass your own
derived key through to it, so you get idempotency guarantees enforced at the source of truth rather
than relying solely on your own bookkeeping, which protects you even in scenarios where your own
action log and the downstream system's state could otherwise drift apart.

## Retry Strategies for Transient Failures

Not all failures should be retried, and not all retries should look the same. The first decision
point is classifying whether a failure is transient (worth retrying) or permanent (retrying is
pointless and can even make things worse). A request that failed because of a network timeout, a
`503 Service Unavailable`, or a rate limit response is transient — the same request will likely
succeed shortly. A request that failed because of a `400 Bad Request` (malformed arguments), a `403
Forbidden` (a real authorization problem), or a business-logic rejection (insufficient account
balance) is permanent from the retry's perspective — retrying the identical request will fail
identically, and the fix is to change the request (or escalate to a human), not to resend it.

```python
class RetryableError(Exception):
    pass

class PermanentError(Exception):
    pass

def classify_error(exc, response=None) -> type:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return RetryableError
    if response is not None:
        if response.status_code in (429, 500, 502, 503, 504):
            return RetryableError
        if response.status_code in (400, 401, 403, 404, 422):
            return PermanentError
    return PermanentError  # default to not retrying unknown failures blindly
```

For the transient case, **exponential backoff with jitter** is the standard approach, and both parts
of that phrase matter. Exponential growth in the delay between attempts (1s, 2s, 4s, 8s...) gives a
struggling downstream dependency room to recover rather than being hit with retries at a constant,
undiminished rate. Jitter — adding randomness to each delay rather than using the exact computed
value — prevents the "thundering herd" problem where many clients, having failed at the same moment
(say, because of a brief provider-wide outage), all retry at exactly the same computed intervals and
hit the recovering service with synchronized bursts of load, which can prevent it from ever actually
recovering.

```python
import random
import time

def retry_with_backoff(fn, max_attempts=5, base_delay=1.0, max_delay=30.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            error_class = classify_error(e)
            if error_class is PermanentError or attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay_with_jitter = random.uniform(delay * 0.5, delay)
            time.sleep(delay_with_jitter)
```

Beyond backoff mechanics, agent systems need a **retry budget**, not just a per-call retry limit — a
cap on total retries across an entire agent run, not just per individual tool call, because a task
that hits five different flaky dependencies and retries each one three times has effectively
multiplied its latency and cost by a factor that a per-call limit alone doesn't bound. It's also
worth distinguishing **LLM-call failures from tool-call failures** for retry purposes: an LLM call
that times out or returns a malformed/unparseable response is usually safe to retry immediately (the
LLM call itself is stateless and read-only from the world's perspective, even though it costs
tokens), whereas a tool call with side effects needs the full idempotency treatment above before
it's safe to retry at all — conflating these two categories, and applying tool-call caution to LLM
calls or LLM-call looseness to tool calls, is a common source of either wasted cost or, worse,
duplicated side effects.

## Circuit Breakers for Degraded Dependencies

Retrying a failing dependency is a reasonable response to an occasional blip; retrying it
relentlessly while it's in a sustained outage is actively harmful — it wastes your own resources,
adds load to a system that's already struggling to recover, and gives every task that depends on it
a slow, painful failure instead of a fast, clear one. The circuit breaker pattern addresses this by
tracking the failure rate of a dependency and, once it crosses a threshold, "opening" the circuit —
failing fast without even attempting the call — for a cooldown period, then allowing a small number
of trial requests through to check whether the dependency has recovered before fully closing the
circuit again.

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_s=30, half_open_trial_count=2):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.half_open_trial_count = half_open_trial_count
        self.state = "closed"       # closed -> open -> half_open -> closed
        self.failure_count = 0
        self.opened_at = None
        self.trial_successes = 0

    def call(self, fn):
        if self.state == "open":
            if time.time() - self.opened_at < self.cooldown_s:
                raise CircuitOpenError("dependency marked unhealthy; failing fast")
            self.state = "half_open"
            self.trial_successes = 0

        try:
            result = fn()
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self):
        self.failure_count += 1
        if self.state == "half_open" or self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.time()

    def _record_success(self):
        if self.state == "half_open":
            self.trial_successes += 1
            if self.trial_successes >= self.half_open_trial_count:
                self.state = "closed"
                self.failure_count = 0
        else:
            self.failure_count = 0
```

In an agent system, circuit breakers should be scoped per-dependency (a broken order-lookup API
shouldn't trip a breaker that also gates an unrelated shipping API) and the agent's higher-level
logic needs an explicit fallback path for what to do when a breaker is open — not just an exception
bubbling up to a generic failure. For a support agent, that fallback might be "tell the user this
lookup is temporarily unavailable and offer to notify them when it's back" or "escalate to a human
who has access to an internal admin tool that doesn't depend on the broken API" — the point being
that a well-designed system treats a known-degraded dependency as a distinct, plannable scenario
rather than an undifferentiated error.

## Partial Failure in Multi-Step Agent Workflows

The hardest failure-handling problem in agentic systems isn't a single failed call, it's a
multi-step workflow that fails partway through, after some steps have already produced real,
external side effects that can't simply be discarded. Consider a workflow that (1) reserves
inventory, (2) charges a payment method, and (3) creates a shipping label — if step 3 fails after
steps 1 and 2 have already succeeded, you cannot just "retry the whole workflow," because retrying
step 1 again would double-reserve inventory, and simply giving up leaves the customer charged with
nothing shipped.

This is the classic distributed-transaction problem, and the standard pattern for it is the **saga
pattern**: model the multi-step workflow as a sequence of steps, each with a corresponding
**compensating action** that can undo it if a later step fails. If step 3 fails, the saga runner
doesn't retry from step 1 — it runs the compensating actions for steps 2 and 1, in reverse order, to
unwind the partial work, then either surfaces the failure or retries step 3 in isolation depending
on whether it's classified as transient.

```python
@dataclass
class SagaStep:
    name: str
    action: callable
    compensation: callable

def run_saga(steps: list[SagaStep], context: dict):
    completed = []
    try:
        for step in steps:
            result = step.action(context)
            context[step.name] = result
            completed.append(step)
        return context
    except Exception as e:
        for step in reversed(completed):
            try:
                step.compensation(context)
            except Exception as comp_error:
                # a failed compensation is a page-a-human event, not a silent log line --
                # the system is now in a state it doesn't know how to safely unwind
                alert_oncall(f"Compensation failed for {step.name}: {comp_error}")
        raise WorkflowFailedError(str(e), completed_steps=[s.name for s in completed])
```

Compensating actions aren't always a perfect mirror of the original action — you can cancel a
payment authorization cleanly, but you generally can't "un-send" an email, only send a follow-up
correction. Designing the saga means being honest about which steps are truly reversible, which are
only approximately reversible (a follow-up communication, a corrective credit), and which are
genuinely irreversible — and for the irreversible ones, either reordering the workflow so they
happen last (do the reversible, risky-to-duplicate steps first, and the truly irreversible step only
once everything else has succeeded) or wrapping them in the strictest possible approval gate from
the human-in-the-loop patterns discussed in the previous chapter, since a saga can't compensate for
a mistake it's structurally unable to undo.

**Checkpointing the workflow's progress**, independent of the saga's own step tracking, is what
makes recovery possible after a process crash rather than just after an in-process exception —
persisting, after each step, which steps have completed and what their results were, so that a
crashed and restarted worker can resume the saga (or run its compensations) from a durable record
rather than from whatever was left in a dead process's memory. This connects directly to the
checkpointing discussion in the async execution chapter: the same durable-state infrastructure that
lets a long-running task survive a worker restart is what lets a partially-failed saga be correctly
unwound rather than left in an ambiguous, half-completed state that nobody can reason about later.

## Bringing It Together: A Layered Failure-Handling Strategy

A production-grade agent system layers these mechanisms rather than picking one: idempotency keys
make individual retries safe at the level of a single action; exponential backoff with jitter and a
bounded retry budget determine when and how often to actually retry a transient failure; circuit
breakers prevent a sustained outage in one dependency from being hammered by, or from cascading
into, the rest of the system; and the saga pattern with compensating actions handles the case where
failure happens after multiple side-effecting steps have already partially completed. None of these
substitute for the others — idempotency without backoff strategy means you retry too aggressively or
not at all; backoff without idempotency means your careful retries are unsafe the moment they touch
anything with a side effect; circuit breakers without a defined fallback just convert a slow failure
into a fast, equally unhandled one; and sagas without durable checkpointing can't survive the
process crash they were designed to protect against in the first place. The strength of an interview
answer on this topic usually comes from being able to say, for a specific proposed action in the
system under discussion, exactly which of these mechanisms applies and why — "this tool call needs
an idempotency key because it charges a payment method, needs backoff because the payment provider
occasionally rate-limits, and needs a saga-level compensation because it's step two of a three-step
booking flow" is a concrete, defensible design; a generic assertion that "we'll add retries and
idempotency" is not.

