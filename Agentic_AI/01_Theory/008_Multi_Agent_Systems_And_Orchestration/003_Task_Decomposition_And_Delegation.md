# Task Decomposition And Delegation

## Decomposition Is Where Orchestration Quality Actually Gets Decided

Of everything a supervisor agent does, the decomposition step deserves the most scrutiny, because every downstream failure mode in a multi-agent system traces back to it. If a task is split badly — subtasks that overlap, subtasks that are missing a piece of the original goal, subtasks handed to a worker that isn't actually equipped for them — no amount of clever delegation, retry logic, or synthesis can fully recover, because the workers are, from their own point of view, correctly solving the wrong sub-problems. This is the multi-agent analogue of a well-known truth in human project management: a bad work-breakdown structure guarantees a bad project regardless of how good the individual contributors are. It's worth treating decomposition as a first-class step with its own validation, rather than a throwaway prompt that produces a JSON list you immediately trust.

A top-level agent breaking a goal into subtasks has to answer three questions, usually all with one LLM call plus some deterministic post-processing: what are the independent units of work, what does each unit depend on (can it start immediately, or does it need another unit's output first), and which available worker is best suited for each unit. Getting the dependency structure right is what determines whether the system can actually exploit parallelism — a decomposition that declares three subtasks independent when one secretly needs another's output will produce workers reasoning over missing information, and looks like a quality failure downstream when it is really a decomposition failure upstream.

## Decomposition Strategies

The three practical decomposition shapes are sequential, parallel, and hierarchical, and picking the right one for a given task is mostly about correctly identifying the true dependency graph rather than reflexively defaulting to one shape.

**Sequential decomposition** applies when subtasks form a pipeline: each stage genuinely needs the previous stage's output before it can meaningfully start. Research before writing, writing before editing, a database migration before the code that depends on the new schema. Forcing a sequential task into parallel execution doesn't save time — it just means each worker gets an empty or garbage input for the fields it depended on — so the decomposition step needs to be conservative about declaring independence.

**Parallel decomposition** applies when subtasks are genuinely self-contained: summarizing five unrelated documents, checking three independent files for style violations, gathering pricing from four different vendors. The payoff is real wall-clock time savings, but only if the independence claim is actually true; a common bug is an LLM-generated decomposition that optimistically marks subtasks as parallel because the prompt asked for parallel subtasks, when a careful read shows subtask B actually needs a fact from subtask A.

**Hierarchical decomposition** applies recursively when a subtask is itself complex enough to need its own breakdown — a "build the backend" subtask that itself splits into "design the schema," "write the API," and "write the tests," each of which might decompose further. This connects directly to the hierarchical orchestration topology: hierarchical decomposition is what feeds a hierarchical topology's manager-of-managers structure, whereas flat sequential/parallel decomposition feeds a simple supervisor/worker topology.

```python
import json


class TaskDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task: str, available_agents: list[dict]) -> dict:
        """Ask the model to classify dependency structure explicitly,
        rather than assuming a shape up front. Returns a DAG, not a
        flat list, so both sequential and parallel execution fall out
        of the same representation."""
        response = self.llm.generate(f"""
        Task: {task}
        Available agents (name, capability description):
        {json.dumps(available_agents, indent=2)}

        Break this into subtasks. For each, list which OTHER subtask ids
        (if any) must complete first. Be conservative: only mark a
        dependency as absent if you are confident the subtask needs
        nothing from any other subtask's output.

        Return JSON:
        {{"subtasks": [
            {{"id": "t1", "description": "...", "agent": "...", "depends_on": []}},
            {{"id": "t2", "description": "...", "agent": "...", "depends_on": ["t1"]}}
        ]}}
        """)
        return json.loads(response)

    def execution_order(self, subtasks: list[dict]) -> list[list[dict]]:
        """Topologically sort the DAG into batches; every subtask in a
        batch can run in parallel because its dependencies are all in
        earlier batches."""
        remaining = {t["id"]: t for t in subtasks}
        done = set()
        batches = []

        while remaining:
            ready = [t for t in remaining.values() if set(t["depends_on"]).issubset(done)]
            if not ready:
                raise ValueError("Cycle detected in subtask dependencies")
            batches.append(ready)
            for t in ready:
                done.add(t["id"])
                del remaining[t["id"]]

        return batches
```

Turning the decomposition into an explicit dependency graph rather than a flat "sequential" or "parallel" list, and then topologically sorting it into execution batches, is a small amount of extra deterministic code that buys a lot of robustness — the orchestrator doesn't need to trust an LLM's blanket claim about the whole task's shape, only its per-subtask dependency claims, which are easier for the model to get right and easier for you to spot-check.

### Validating A Decomposition Before You Trust It

Because a bad decomposition poisons everything downstream, it deserves the same defensive treatment you'd give any other untrusted LLM output before it's allowed to drive expensive downstream work. Three checks catch the majority of practical decomposition failures cheaply, before a single worker is invoked: coverage (does the union of subtasks plausibly address the whole original task, or did the model drop a requirement mentioned in the goal), assignment validity (is every `assigned_worker` actually a worker that exists and has the required capability, rather than a hallucinated name), and structural soundness (does the dependency graph parse as a DAG at all, or did the model produce a cycle, which the `execution_order` topological sort above will correctly refuse to run rather than silently mis-scheduling).

```python
def validate_decomposition(decomposition: dict, available_agents: list[dict], original_task: str, llm) -> list[str]:
    problems = []
    known_agents = {a["name"] for a in available_agents}
    known_ids = {t["id"] for t in decomposition["subtasks"]}

    for subtask in decomposition["subtasks"]:
        if subtask["agent"] not in known_agents:
            problems.append(f"{subtask['id']}: assigned to unknown agent '{subtask['agent']}'")
        for dep in subtask["depends_on"]:
            if dep not in known_ids:
                problems.append(f"{subtask['id']}: depends on unknown subtask '{dep}'")

    # A cheap, cheap LLM-as-checker pass for coverage, since this is a
    # judgment call rather than something purely structural.
    coverage_check = llm.generate(f"""
    Original task: {original_task}
    Proposed subtasks: {[t['description'] for t in decomposition['subtasks']]}
    Does this list of subtasks, taken together, cover everything the
    original task asked for? Answer "YES" or list what's missing.
    """)
    if "YES" not in coverage_check:
        problems.append(f"coverage gap: {coverage_check}")

    return problems
```

Treat a non-empty `problems` list as a signal to re-run decomposition with the problems fed back in as feedback (the same refine-and-retry shape used for worker failures later in this chapter) rather than either blindly proceeding or hard-failing the whole task — a decomposition that's missing one minor sub-requirement is usually fixable with one more LLM call, and that's far cheaper than discovering the gap only after several workers have already run.

## Delegation Strategies

Once you know *what* the subtasks are, the next problem is *who* does each one. The simplest approach — a static mapping from subtask type to a fixed worker — is fine for small, stable systems, but two more adaptive strategies matter once the worker pool grows or needs to handle variable load: capability-based routing and market-style bidding.

**Capability-based routing** treats each worker as advertising a set of capabilities, indexes workers by capability, and scores the intersection of "who can do this" against secondary factors like current load and historical success rate, so that routing degrades gracefully to "best available match" rather than failing outright when no worker is a perfect fit.

```python
class CapabilityRouter:
    def __init__(self, agents):
        self.agents = agents
        self.index: dict[str, list] = {}
        for agent in agents:
            for cap in agent.capabilities:
                self.index.setdefault(cap, []).append(agent)

    def route(self, required_capabilities: list[str]):
        candidates = set(self.agents)
        for cap in required_capabilities:
            candidates &= set(self.index.get(cap, []))

        if not candidates:
            # No perfect match: fall back to best partial match rather
            # than failing the whole subtask outright.
            candidates = self.agents

        scored = [(self._score(a, required_capabilities), a) for a in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _score(self, agent, required_capabilities):
        match = sum(1 for c in required_capabilities if c in agent.capabilities)
        match_ratio = match / max(len(required_capabilities), 1)
        load_headroom = 1.0 - (agent.current_tasks / max(agent.max_tasks, 1))
        return 0.5 * match_ratio + 0.25 * load_headroom + 0.25 * agent.success_rate
```

**Bidding (contract-net style) delegation** flips the direction: instead of the orchestrator unilaterally deciding who's best suited, it announces the subtask to a pool of candidate workers and lets each one estimate its own cost, time, and confidence, then picks the best bid. This is more expensive (every candidate has to be consulted before the winner is chosen) but is valuable when workers have real, variable, self-known differences in suitability that the orchestrator can't easily judge from the outside — for example, when workers represent different underlying models with different per-token costs and latencies, and the "best" choice genuinely depends on current conditions rather than a fixed capability list.

```python
class ContractNet:
    def __init__(self, contractors):
        self.contractors = contractors

    def award(self, task: str):
        bids = []
        for contractor in self.contractors:
            bid = contractor.estimate(task)
            if bid["willing"]:
                bids.append({"agent": contractor, **bid})

        if not bids:
            return None

        for bid in bids:
            bid["score"] = (
                0.4 * bid["confidence"]
                + 0.3 * (1 / max(bid["estimated_cost"], 0.01))
                + 0.3 * (1 / max(bid["estimated_time"], 0.01))
            )
        winner = max(bids, key=lambda b: b["score"])
        return winner["agent"].execute(task)
```

Use capability-based routing as the default — it's cheap, deterministic, and good enough when workers' suitability is well captured by a static capability list. Reach for bidding only when the "who's best for this" question genuinely varies at runtime in ways a static index can't capture.

### Budgeting Time And Cost Per Delegated Subtask

A subtask handed to a worker needs an explicit resource budget, not just a description — without one, a single worker that gets stuck in an unproductive tool-use loop (repeatedly retrying a failing search query, for instance) can silently consume the majority of a task's total time or cost budget while every other subtask waits or the orchestrator's own timeout eventually fires with no useful diagnostic signal about which worker was responsible. Attaching a budget at delegation time, and enforcing it at the worker boundary rather than hoping the worker self-regulates, keeps one runaway subtask from degrading the whole task.

```python
import time


class BudgetedWorker:
    def __init__(self, worker, max_seconds: float, max_tool_calls: int):
        self.worker = worker
        self.max_seconds = max_seconds
        self.max_tool_calls = max_tool_calls

    def execute(self, subtask: str):
        start = time.monotonic()
        tool_calls_used = 0
        original_tool_call = self.worker.call_tool

        def _tracked_tool_call(*args, **kwargs):
            nonlocal tool_calls_used
            tool_calls_used += 1
            if tool_calls_used > self.max_tool_calls:
                raise RuntimeError(f"{self.worker.name} exceeded tool-call budget")
            if time.monotonic() - start > self.max_seconds:
                raise TimeoutError(f"{self.worker.name} exceeded time budget")
            return original_tool_call(*args, **kwargs)

        self.worker.call_tool = _tracked_tool_call
        try:
            return self.worker.execute(subtask)
        finally:
            self.worker.call_tool = original_tool_call
```

Setting these budgets per subtask (rather than one global budget for the whole task) also gives you a much better diagnostic signal when something goes wrong: a timeout attributed to a specific worker and subtask is actionable, while a single global "the whole task took too long" timeout tells you nothing about where the time actually went.

## Handling Partial Failures

A multi-agent system that assumes every worker always succeeds is not production-ready, because LLM calls fail for all the ordinary reasons any network call fails (timeouts, rate limits, transient errors) plus a class of failures unique to agents: a worker returns output in the wrong format, a worker "succeeds" but produces something that doesn't actually satisfy the subtask, or a worker gets stuck in a tool-use loop and burns its budget without producing anything. The orchestrator needs an explicit policy for each of these, because the naive behavior — letting one failed subtask crash the whole run — throws away all the work the other, successful subtasks did.

The first layer of defense is validation plus bounded retry: check that a worker's output actually matches the expected shape before accepting it, and if it doesn't, or the call raised an exception, retry with a refined prompt that includes the failure reason, up to a small retry budget.

```python
class RecoveringExecutor:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def execute(self, worker, subtask: str) -> dict:
        current_task = subtask
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = worker.execute(current_task)
            except Exception as exc:
                last_error = str(exc)
                current_task = f"{subtask}\n\n(Previous attempt failed: {last_error})"
                continue

            if self._is_valid(result):
                return {"status": "ok", "output": result, "attempts": attempt + 1}

            last_error = "output failed validation"
            current_task = f"{subtask}\n\n(Previous attempt was invalid: {result[:200]})"

        # Exhausted retries: degrade instead of crashing the whole run.
        return {"status": "failed", "output": None, "error": last_error}

    def _is_valid(self, result) -> bool:
        return bool(result) and len(result.strip()) > 0
```

The second layer is deciding what the orchestrator does with a subtask that never recovers. There are three reasonable policies, and the right one depends on whether the failed subtask is load-bearing for the final answer: **fail the whole task** (appropriate when the failed subtask is essential and there's no reasonable way to proceed without it — e.g., a missing security review before deploying code); **degrade and proceed** (appropriate when the failed subtask is a nice-to-have — e.g., a "check competitor pricing" subtask failing shouldn't block a report that's otherwise complete, it should just be noted as a gap); and **fall back to an alternative worker or a cheaper/simpler approach** (appropriate when there's a lower-quality but workable substitute — e.g., falling back to a simpler summarization if a specialized analysis agent times out). Building this decision as an explicit policy attached to each subtask at decomposition time (a `criticality` field: `required` vs `optional`) is far more robust than hard-coding "always fail" or "always proceed," because it lets the same executor code handle both a report-writing pipeline (where most subtasks are optional) and a deployment pipeline (where a failed security check must halt everything).

## Aggregating Results Into a Coherent Answer

The final step — turning a set of independent (and possibly partially failed) subtask results into one coherent deliverable — is deceptively easy to get wrong, because it's tempting to just concatenate the pieces and call it done. Concatenation works when subtasks produce genuinely independent sections (e.g., five separate document summaries that just need to be listed), but it fails as soon as subtasks might overlap, contradict each other, or need to be woven together into a single voice and narrative structure, which is the common case for anything more than a simple fan-out report.

The more robust approach uses an LLM synthesis pass whose prompt explicitly acknowledges partial failure and gives instructions for handling both overlap and contradiction, rather than assuming every result is complete and mutually consistent.

```python
def aggregate(llm, original_task: str, results: dict) -> str:
    completed = {k: v["output"] for k, v in results.items() if v["status"] == "ok"}
    failed = [k for k, v in results.items() if v["status"] != "ok"]

    return llm.generate(f"""
    Original task: {original_task}

    Completed subtask results:
    {json.dumps(completed, indent=2)}

    Subtasks that failed and produced no usable output: {failed}

    Synthesize the completed results into one coherent final answer.
    Rules:
    - If two results disagree on a fact, prefer the more specific/recent
      one and note the discrepancy rather than silently picking one.
    - If a failed subtask left a gap that materially affects the answer,
      state the limitation explicitly instead of inventing content to
      fill it.
    - Do not simply concatenate; integrate into a single coherent voice.
    """)
```

Two details matter here in production. First, conflict resolution needs an explicit instruction, because left unguided, an LLM synthesizer will often silently pick one of two contradictory subtask outputs without flagging the contradiction, which hides a real problem (two research agents disagreeing about a fact is a signal worth surfacing, not smoothing over). Second, honesty about gaps from failed subtasks matters more than most teams initially budget for: an aggregator that isn't told which subtasks failed will happily "fill in" a plausible-sounding but fabricated section to cover the gap, because from its point of view it's just being asked to write a complete-looking answer — passing the failure list into the synthesis prompt, as above, is what prevents this. For high-stakes aggregation (financial reports, medical or legal summaries), it's worth going a step further than a single synthesis LLM call and adding a dedicated validation pass afterward that checks the synthesized answer against the original completed subtask outputs for unsupported claims — effectively a lightweight fact-check of your own aggregation step before it reaches the end user.

### Structural Aggregation As An Alternative To LLM Synthesis

Not every aggregation step needs an LLM call, and it's worth defaulting to deterministic merging whenever the subtask outputs are already structured data rather than free text — it's cheaper, faster, and, critically, it removes an entire class of synthesis-introduced errors (a paraphrase that subtly changes a number, a summary that drops a caveat) that a natural-language synthesis pass can introduce even when every individual subtask succeeded correctly.

```python
def aggregate_structured(results: dict) -> dict:
    """When subtasks return structured JSON rather than prose, merge
    deterministically instead of asking an LLM to reassemble it —
    faster, cheaper, and immune to paraphrase-introduced errors."""
    merged = {"sections": [], "sources": [], "warnings": []}
    for subtask_id, result in results.items():
        if result["status"] != "ok":
            merged["warnings"].append(f"{subtask_id} failed: {result.get('error')}")
            continue
        output = result["output"]
        merged["sections"].append({"id": subtask_id, "content": output["content"]})
        merged["sources"].extend(output.get("sources", []))
    merged["sources"] = list(dict.fromkeys(merged["sources"]))  # de-dupe, preserve order
    return merged
```

A practical rule: use structural, deterministic aggregation whenever subtask outputs are naturally sectioned or independently addressable (a list of document summaries, a set of per-file lint results), and reserve LLM-based synthesis for the cases where the subtask outputs genuinely need to be rewoven into one coherent narrative voice, cross-referenced against each other, or reconciled where they conflict — using an expensive tool only where the task actually requires the judgment that tool provides.
