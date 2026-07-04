# Blackboard And Shared State Systems

## The Original Idea: A Shared Workspace Instead Of A Chain Of Command

Every pattern covered earlier in this chapter — supervisor/worker, hierarchical, even peer-to-peer messaging — assumes some explicit routing: someone decides who acts next and what they're given to work with. The blackboard architecture, one of the oldest ideas in multi-agent AI, takes a different stance entirely. Instead of routing decisions being made by a coordinator, all agents (historically called "knowledge sources") share access to one common workspace — the blackboard — which holds the current state of the evolving problem. Each agent independently watches the blackboard, and whenever it notices that the current state contains something it knows how to act on, it contributes: it reads what it needs, does its work, and writes its contribution back, which in turn may trigger other agents to act on the new state. There is no fixed sequence of who goes next — control is opportunistic, driven by which knowledge source currently has something useful to contribute given the blackboard's state, not by a predetermined plan.

This architecture originated with Hearsay-II, a 1970s speech understanding system, precisely because speech recognition doesn't decompose cleanly into a fixed pipeline: acoustic-phonetic analysis, word-level hypotheses, syntactic parsing, and semantic interpretation all inform each other in a way that's hard to force into a strict "step 1, then step 2" sequence — a syntactic constraint might rule out an acoustic hypothesis, which changes what word-level hypotheses look plausible, which changes what the semantic layer can conclude, and any of these can happen in any order depending on what evidence becomes available first. The blackboard's genius was letting many independent specialists collaborate on a shared, evolving hypothesis without hard-coding the order in which they'd need to interact — a design goal that maps remarkably well onto today's multi-agent LLM systems, most of which have exactly the same property: a research agent's finding might change what a fact-checker needs to verify, which might change what the writer should emphasize, and forcing this into a strict pipeline discards flexibility that the underlying problem actually has.

## The Classic Three-Part Structure

A textbook blackboard system has three conceptually distinct pieces, and it's worth keeping them distinct even in a modern LLM implementation because conflating them is a common source of the concurrency bugs discussed later in this chapter.

The **blackboard** itself is the shared data structure — typically organized into named regions or levels representing different kinds of partial solutions (in Hearsay-II: phrases, words, syllables, segments; in a modern research-and-writing system: findings, outline, draft, citations, open-questions). It is passive — it doesn't do anything on its own, it just holds state and, ideally, a history of how that state changed.

The **knowledge sources** are the independent specialist agents — each one knows how to recognize a specific kind of opportunity ("there's a new finding but no outline yet" or "the draft references a fact with no citation") and knows how to act on it (write an outline, look up a citation). Crucially, a knowledge source in the classic architecture doesn't know about other knowledge sources at all — it only knows how to read the blackboard, decide whether it has something useful to contribute, and write its contribution back. This is what gives the pattern its openness: you can add a new knowledge source to the system without touching any of the existing ones, because none of them have any coupling to each other, only to the shared blackboard.

The **control component** decides, at each step, which of the (possibly several) knowledge sources that currently believe they have something useful to contribute actually gets to act next — because if multiple specialists all think they can contribute right now, something has to arbitrate, or you risk simultaneous conflicting writes. In the original systems this was typically a scheduler that scored each knowledge source's proposed contribution by estimated value and picked the best one to run next, one at a time.

```python
class KnowledgeSource:
    """A specialist that watches the blackboard and contributes when it
    sees a relevant opportunity. It never talks to other knowledge
    sources directly — only through blackboard state."""

    def __init__(self, name, llm, trigger_check, contribute_fn):
        self.name = name
        self.llm = llm
        self._trigger_check = trigger_check   # (blackboard) -> bool
        self._contribute_fn = contribute_fn   # (blackboard, llm) -> None

    def can_contribute(self, blackboard) -> bool:
        return self._trigger_check(blackboard)

    def contribute(self, blackboard):
        self._contribute_fn(blackboard, self.llm)


class BlackboardController:
    """The control component: on each cycle, ask every knowledge source
    whether it wants to act, then run exactly one (or run all that are
    ready, if you want more throughput at the cost of more coordination
    care)."""

    def __init__(self, blackboard, knowledge_sources, max_cycles=20):
        self.blackboard = blackboard
        self.knowledge_sources = knowledge_sources
        self.max_cycles = max_cycles

    def run(self):
        for _ in range(self.max_cycles):
            ready = [ks for ks in self.knowledge_sources if ks.can_contribute(self.blackboard)]
            if not ready:
                break  # nobody has anything left to contribute: solution is stable
            # A real scheduler would rank `ready` by expected contribution
            # value; here we just take the first as a simple default.
            ready[0].contribute(self.blackboard)
```

### Scheduling: Choosing Among Several Ready Knowledge Sources

The `BlackboardController.run` shown above picks `ready[0]` whenever multiple knowledge sources are simultaneously able to contribute, which is a placeholder for what the classical literature treats as a genuinely important design decision: the scheduler's job is to estimate, for each ready knowledge source, how valuable its proposed contribution is likely to be, and run the best one first, because running a low-value contribution first can waste a full cycle (and, for LLM-backed knowledge sources, real tokens and latency) on a contribution that a better-scheduled run wouldn't have needed at all.

```python
class ScoredController(BlackboardController):
    def __init__(self, blackboard, knowledge_sources, scorer, max_cycles=20):
        super().__init__(blackboard, knowledge_sources, max_cycles)
        self.scorer = scorer  # (knowledge_source, blackboard) -> float

    def run(self):
        for _ in range(self.max_cycles):
            ready = [ks for ks in self.knowledge_sources if ks.can_contribute(self.blackboard)]
            if not ready:
                break
            best = max(ready, key=lambda ks: self.scorer(ks, self.blackboard))
            best.contribute(self.blackboard)


def confidence_scorer(knowledge_source, blackboard) -> float:
    # A cheap heuristic: prefer knowledge sources whose trigger condition
    # is most clearly and specifically satisfied right now, rather than
    # ones that could plausibly contribute but with low expected value.
    return knowledge_source.estimate_contribution_value(blackboard)
```

A simple and often sufficient heuristic in LLM-backed blackboard systems is to prioritize knowledge sources that unblock the largest number of other, currently-stalled knowledge sources — the equivalent of running the critical-path step first in a dependency graph — since a contribution that only one other agent is waiting on is less valuable, cycle for cycle, than one that several others need before they can proceed.

## Mapping This Onto Modern Multi-Agent Frameworks

The blackboard's shared-workspace idea shows up, usually without the name, all over current LLM orchestration tooling, and recognizing the mapping helps you reason about frameworks you didn't build yourself using vocabulary that's decades more mature than "agent framework X's state object."

LangGraph's shared `TypedDict` state, updated by reducer functions (`operator.add` for lists, custom merge functions for dicts), is a blackboard in fairly literal terms: every node (agent) reads the shared state, decides what to contribute based on what's there, and writes its update back through a reducer, and the graph's edges (rather than a value-scoring scheduler) determine which knowledge source runs next. The main difference from the classical architecture is that LangGraph's control flow is usually specified as an explicit graph rather than fully opportunistic scheduling — you get blackboard-style shared state with supervisor-style explicit routing layered on top, which is a very common and pragmatic hybrid in production systems.

AutoGen's `GroupChat` shared message transcript is a restricted form of blackboard where the "blackboard" is append-only and organized as a single linear log rather than structured named regions — every agent reads the whole transcript and decides whether to speak next, and the `GroupChatManager` plays the role of the control component, deciding (often via its own LLM call) which agent should act next given the current transcript state.

CrewAI's shared task context, where a downstream task can be given the outputs of upstream tasks as context, is closer to a constrained, pipeline-shaped blackboard — the "regions" are individual task outputs, and reads are explicitly wired rather than opportunistically scanned, trading some of the classical architecture's flexibility for a simpler mental model and easier debugging.

A direct, minimal Python implementation, closer to the classic blackboard's structured regions than a flat transcript, looks like this:

```python
from datetime import datetime


class Blackboard:
    """A shared, versioned workspace multiple agents read from and
    write to. Every write is recorded with an author and a version
    number so later components (and the concurrency-control code in
    the next section) can detect stale writes."""

    def __init__(self):
        self._data: dict[str, dict] = {}
        self.history: list[dict] = []

    def write(self, key: str, value, author: str):
        current_version = self._data.get(key, {}).get("version", 0)
        self._data[key] = {
            "value": value,
            "author": author,
            "version": current_version + 1,
            "timestamp": datetime.now(),
        }
        self.history.append({"key": key, "author": author, "version": current_version + 1})

    def read(self, key: str):
        entry = self._data.get(key)
        return entry["value"] if entry else None

    def version_of(self, key: str) -> int:
        return self._data.get(key, {}).get("version", 0)

    def query(self, prefix: str) -> dict:
        return {k: v["value"] for k, v in self._data.items() if k.startswith(prefix)}
```

## Why Shared Mutable State Between Agents Is Dangerous

The blackboard's flexibility comes from every agent being able to read and write a shared structure without going through a central arbiter for every single interaction, and that is exactly the property that introduces race conditions once more than one agent can act concurrently rather than one at a time. Classical blackboard systems mostly sidestepped this by having the control component run knowledge sources strictly one at a time — pick the single best one, let it finish, then re-evaluate. Modern LLM multi-agent systems are frequently tempted to run agents concurrently for latency reasons (exactly the parallelism benefit discussed in the orchestration patterns chapter), and that's precisely where shared-state bugs creep in.

The clearest failure mode is a **lost update**: two agents both read the same key, both compute a new value based on what they read, and both write back — the second write silently overwrites the first, and the first agent's contribution is gone with no error raised anywhere. Concretely: a "risk-flags" list on the blackboard is read by both a security-review agent and a compliance-review agent at version 3; the security agent appends its finding and writes back version 4; the compliance agent, still holding its own copy read at version 3, appends its own finding to *that* copy and writes back version 4 as well — the security agent's finding is silently gone, and nothing in the system noticed, because both writes "succeeded."

```python
# The race, made explicit:
security_flags = blackboard.read("risk_flags")     # both agents read [] at version 3
compliance_flags = blackboard.read("risk_flags")    # ...

security_flags = security_flags + ["exposed credentials"]
compliance_flags = compliance_flags + ["missing consent record"]

blackboard.write("risk_flags", security_flags, author="security_agent")     # -> version 4
blackboard.write("risk_flags", compliance_flags, author="compliance_agent")  # -> version 4, overwrites!
# Final state: only the compliance finding survives. The security finding vanished.
```

A second failure mode is a **stale read leading to an inconsistent decision**: an agent reads a value, spends a long time "thinking" (an LLM call can easily take several seconds), and acts on a decision that assumed the value it read is still current — but by the time it writes its own contribution, some other agent has already changed the underlying state in a way that invalidates the assumption. A writer agent might draft a whole section based on a "finding" that a fact-checking agent has since retracted and overwritten, and the writer has no way of knowing its input is now stale unless something explicitly checks.

A third, subtler failure mode is **ordering nondeterminism producing irreproducible bugs**: because control in a blackboard-style system is opportunistic rather than fixed, the same initial state can legitimately produce different final outcomes depending on which agent happens to act first when several are simultaneously "ready" — this is a feature for exploring solution space in some classical AI applications, but in a production LLM pipeline it means the same input can silently produce different outputs on different runs, which makes bugs hard to reproduce and makes evaluation noisy in a way that's easy to misattribute to model variance rather than to the orchestration layer itself.

## Concurrency Control Strategies That Actually Work

The most robust fix, and the one worth reaching for first, is the **single-writer principle**: design the blackboard's schema so that each key has exactly one agent (or one pipeline stage) that is ever allowed to write it, even if multiple agents read it. In the example above, this means the security agent owns a `security_flags` key and the compliance agent owns a `compliance_flags` key, and a separate aggregation step merges them — rather than both agents contending for one shared `risk_flags` key. This eliminates the lost-update race entirely by construction, at the cost of a small amount of schema design discipline up front, and it should be the default whenever the data in question naturally partitions by owner, which is more often true than it first appears.

When multiple writers to the same key are genuinely unavoidable — for instance, several agents incrementally building one shared list where none of them individually "owns" the whole list — **optimistic concurrency control** with version checks is the standard fix: a write only succeeds if the version the writer last read still matches the current version in the blackboard; if not, the write is rejected and the agent must re-read the current state, re-apply its change on top of the fresh value, and retry.

```python
class VersionedBlackboard(Blackboard):
    def compare_and_write(self, key: str, expected_version: int, value, author: str) -> bool:
        current_version = self.version_of(key)
        if current_version != expected_version:
            return False  # stale write rejected; caller must re-read and retry
        self.write(key, value, author)
        return True


def append_with_retry(blackboard: VersionedBlackboard, key: str, item, author: str, max_attempts=5):
    for _ in range(max_attempts):
        current = blackboard.read(key) or []
        expected_version = blackboard.version_of(key)
        updated = current + [item]
        if blackboard.compare_and_write(key, expected_version, updated, author):
            return True
    return False  # gave up after repeated contention; caller decides how to handle this
```

This pattern — read, compute, compare-and-swap, retry on conflict — is exactly the optimistic-locking approach used in traditional databases and distributed systems, and it generalizes cleanly to LLM agents because the "compute" step (an LLM call) is the expensive part and the "compare-and-swap" step is cheap, so retrying just the cheap merge step on conflict (rather than re-running the whole expensive agent call) is usually feasible if you separate "decide what to contribute" from "attempt to write it" as distinct steps.

A related but different approach, useful when writes are naturally combinable rather than requiring conflict rejection, is to make the blackboard **append-only** instead of mutable — agents never overwrite a key's value, they only append new entries to an immutable log, and any "current state" is derived by folding over the log (last-write-wins, or a domain-specific merge) rather than stored as a single mutable cell. This sidesteps lost updates entirely, since nothing is ever overwritten, at the cost of needing a well-defined fold/merge function and slightly more bookkeeping to reconstruct "current state" on demand — but it also gives you, for free, the full history of how the shared state evolved, which is valuable for debugging exactly the kind of ordering-nondeterminism issues described above.

Finally, for the cases where genuine mutual exclusion is required — an agent needs to read-modify-write a value with no other agent touching it in between, and neither the single-writer principle nor optimistic retries fit cleanly — an explicit lock around the specific key (not the whole blackboard, which would kill your parallelism) is the last resort, used sparingly because locks reintroduce sequential bottlenecks and the possibility of deadlock if two agents ever need to hold locks on two keys in opposite orders.

```python
import threading


class LockedBlackboard(Blackboard):
    def __init__(self):
        super().__init__()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        return self._locks.setdefault(key, threading.Lock())

    def read_modify_write(self, key: str, modify_fn, author: str):
        with self._lock_for(key):
            current = self.read(key)
            new_value = modify_fn(current)
            self.write(key, new_value, author)
            return new_value
```

The practical guidance is to reach for these in order of preference: design the schema for single ownership wherever the data allows it, since that avoids the problem entirely; use optimistic versioning for the genuinely shared, combinable values that remain; consider an append-only log when you want the state's evolution history as a first-class artifact and can define a sensible merge; and reserve hard locks for the rare case where none of the above fit, keeping their scope as narrow as possible. Whichever you choose, treat the blackboard's write history (the `history` list in the examples above) as a required piece of observability, not an optional nicety — when an opportunistic, concurrently-written shared-state system produces a wrong or surprising final answer, that history is usually the only way to reconstruct which agent wrote what, in what order, and whether a race condition rather than a reasoning error is the actual root cause.

## Merge Functions For Genuinely Concurrent Contributions

Sometimes rejecting a stale write and forcing a retry, as the optimistic-concurrency pattern does, is more disruptive than necessary, because two concurrent contributions might not actually conflict in substance even though they touch the same key — two agents appending different, non-overlapping findings to the same list is a case where the "right" behavior is to combine both contributions, not to make one of them retry from scratch. This is the same problem distributed systems solve with CRDTs (conflict-free replicated data types): define a merge function for a given data type such that concurrent updates can always be combined deterministically, regardless of the order they're observed in, without needing a central lock or a reject-and-retry cycle.

```python
class MergeableBlackboard(Blackboard):
    """Associates each key with a merge function so concurrent writes
    combine instead of clobbering or requiring a retry loop."""

    def __init__(self):
        super().__init__()
        self._merge_fns: dict[str, callable] = {}

    def set_merge_fn(self, key_prefix: str, merge_fn):
        self._merge_fns[key_prefix] = merge_fn

    def merge_write(self, key: str, contribution, author: str):
        current = self.read(key)
        merge_fn = next(
            (fn for prefix, fn in self._merge_fns.items() if key.startswith(prefix)),
            lambda old, new: new,  # default: last write wins if no merge fn registered
        )
        merged = merge_fn(current, contribution)
        self.write(key, merged, author)


# Registering a set-union merge for a "findings" region: concurrent
# contributions from different agents combine automatically.
blackboard = MergeableBlackboard()
blackboard.set_merge_fn("findings", lambda old, new: list(set((old or []) + [new])))
```

Merge functions only work when the data type genuinely supports a commutative, associative combination — sets, counters, and append-only lists merge naturally; a single scalar "current best answer" field usually does not, because there's no principled way to "merge" two different proposed answers into one without a judgment call, which is exactly why the single-writer and optimistic-versioning approaches remain the right default for that kind of data. Use merge functions specifically for the subset of your blackboard schema — usually collections, not scalars — where combining concurrent contributions is well-defined, and fall back to single ownership or versioning everywhere else.

## Monitoring A Blackboard System In Production

Because control in a blackboard system is opportunistic rather than explicit, the two things that go wrong most often in production are agents contributing when they shouldn't (an ill-specified trigger condition fires on state it wasn't meant to react to, producing noisy or redundant contributions) and the system never reaching a stable state at all (a poorly designed pair of knowledge sources keep re-triggering each other — agent A's contribution makes agent B's trigger fire, whose contribution makes agent A's trigger fire again, forever). Both are best caught with a small set of standing metrics rather than by reading the write history after the fact: total contributions per knowledge source per run (a knowledge source contributing far more than its peers on a stable, well-behaved run is a signal its trigger condition is too broad), cycles-to-stability (how many control-loop iterations elapse before no knowledge source has anything left to contribute, tracked over time to catch a regression that turns a normally 5-cycle run into a 40-cycle one), and a hard cycle ceiling (the `max_cycles` parameter on `BlackboardController` is not just a performance safeguard, it's the mechanism that turns an infinite oscillation bug into a bounded, detectable failure rather than a silently runaway job). Alerting on "hit the max_cycles ceiling" as a distinct, named failure mode — rather than lumping it in with generic task failures — makes it much faster to recognize when a new knowledge source you've just added has introduced an oscillation, since that failure signature is specific enough to point directly at recently changed trigger conditions rather than requiring a full trace replay to diagnose.
