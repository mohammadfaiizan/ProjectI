# Plan-and-Execute Architectures

## The Motivation: Decoupling "What to Do" From "Doing It"

ReAct's strength — replanning after every single tool call — is also a tax that gets paid whether or not it's needed. If a task's structure is largely predictable once you've thought about it for a moment (e.g., "to write this report I need to: pull last quarter's numbers, pull this quarter's numbers, compute the deltas, then draft the narrative"), interleaving a full reasoning pass between every one of those steps is wasted computation: the model is re-deriving a plan it could have committed to in one shot at the very beginning. Worse, because each ReAct step only sees the immediate local context, a purely reactive agent can lose sight of the big picture over a long trajectory — it optimizes step N without a strong representation of how step N fits into the overall goal, which can lead to redundant tool calls, drift away from the original intent, or premature stopping once *a* plausible-looking answer appears, even if it isn't the one the plan actually called for.

Plan-and-Execute architectures respond to this by making the decomposition of the task into an explicit, separate, upfront artifact — a plan — and then executing that plan with a simpler, more mechanical process that doesn't need to re-derive the overall strategy at every step. This idea traces back to classical AI planning (STRIPS-style symbolic planners from the 1970s onward) and was reintroduced for LLM agents in systems like BabyAGI and the "Plan-and-Solve" prompting line of research, which showed that separating "devise a plan" from "carry out a step" measurably reduced errors on multi-step arithmetic and reasoning tasks relative to a single undifferentiated CoT pass.

## The Two (or Three) Phases

The architecture has a clean phase structure:

1. **Planning phase**: a single call (or a small, bounded number of calls) to the LLM, given the full goal and a description of available tools, produces a structured plan — typically a numbered list of subtasks, ideally with their dependencies made explicit so independent steps can be identified and potentially parallelized.
2. **Execution phase**: each subtask in the plan is carried out, usually by a much simpler executor (which might itself be a small ReAct agent scoped to just that one subtask, or even a plain function call if the subtask is deterministic). Execution proceeds through the plan's steps in dependency order, accumulating results.
3. **Replanning phase (optional but usually present in production)**: if a step fails, or its result reveals that the plan's assumptions were wrong, the system does not simply retry blindly — it goes back to the LLM with the original goal, the plan, and what has been learned so far, and asks for a *revised* plan rather than a resumed one.

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor, max_replans: int = 2):
        self.planner_llm = planner_llm
        self.executor = executor
        self.max_replans = max_replans

    def run(self, goal: str) -> str:
        plan = self._make_plan(goal)
        history = []

        for attempt in range(self.max_replans + 1):
            outcome = self._execute_plan(plan, history)
            if outcome["success"]:
                return outcome["final_result"]

            history.append(outcome)
            if attempt == self.max_replans:
                return f"Failed after {self.max_replans} replans: {outcome['error']}"

            plan = self._replan(goal, plan, history)

        return "Unreachable"

    def _make_plan(self, goal: str) -> list[dict]:
        prompt = f"""Decompose this goal into an ordered list of subtasks.
For each subtask specify: id, description, depends_on (list of ids).

Goal: {goal}

Respond as a JSON list of objects with keys: id, description, depends_on.
"""
        return self.planner_llm.generate_json(prompt)

    def _execute_plan(self, plan: list[dict], history: list) -> dict:
        results = {}
        completed = set()

        while len(completed) < len(plan):
            ready = [s for s in plan
                     if s["id"] not in completed
                     and all(d in completed for d in s["depends_on"])]
            if not ready:
                return {"success": False, "error": "circular or unresolved dependency"}

            for step in ready:
                try:
                    context = {dep: results[dep] for dep in step["depends_on"]}
                    results[step["id"]] = self.executor.execute(step["description"], context)
                    completed.add(step["id"])
                except Exception as exc:
                    return {
                        "success": False,
                        "error": str(exc),
                        "failed_step": step["id"],
                        "partial_results": results,
                    }

        return {"success": True, "final_result": self._synthesize(plan, results)}

    def _replan(self, goal, old_plan, history) -> list[dict]:
        failure = history[-1]
        prompt = f"""Goal: {goal}

Previous plan: {old_plan}
It failed at step '{failure['failed_step']}' with error: {failure['error']}
Partial results so far: {failure.get('partial_results', {})}

Produce a revised plan that avoids this failure. You may reuse completed
results rather than redoing them. Respond in the same JSON format.
"""
        return self.planner_llm.generate_json(prompt)

    def _synthesize(self, plan, results) -> str:
        ordered = [results[s["id"]] for s in plan]
        return "\n".join(str(r) for r in ordered)
```

The dependency-graph structure in `_execute_plan` is doing real work beyond bookkeeping: because subtasks declare `depends_on` explicitly, steps with no dependency relationship to each other can be identified as parallelizable, which is something a purely sequential ReAct trajectory cannot offer without additional machinery — ReAct emits one action at a time by construction, whereas an explicit plan is a static artifact you can analyze and schedule before executing any of it.

## The Trade-offs Against ReAct

### Latency

Plan-and-Execute typically wins on latency for tasks with genuine parallelism, because independent subtasks can run concurrently once the plan has identified them as independent — something ReAct's strictly sequential Thought→Action→Observation loop cannot do without an additional orchestration layer on top of it. But it can lose on latency for tasks that are inherently sequential and where the planning call itself is expensive relative to the savings: you pay for a full planning pass before any execution work starts, and if that plan turns out to need revision, you pay for a replanning pass on top of the wasted execution. ReAct, by contrast, never "wastes" a full plan — it commits to only one step at a time, so a single bad turn costs one step, not the value of an entire discarded plan.

### Steerability

An explicit plan is a legible artifact: it can be shown to a user or another system for review *before* any tool with side effects is invoked. This is a meaningfully different risk profile from ReAct, where the first the outside world learns of an intended dangerous action is often the moment it is about to be (or has been) executed, because reasoning and acting are interleaved rather than front-loaded. Systems that need a human-in-the-loop approval gate — "here is my plan to refund this customer, delete these records, and email this vendor; confirm before I proceed" — are naturally suited to Plan-and-Execute, because there is a clean seam between "decide" and "do" at which to insert a human checkpoint. Bolting an equivalent gate onto ReAct means pausing after every single action, which is far more disruptive to the interaction.

### Replanning on failure

This is the subtlest trade-off. ReAct replans implicitly and constantly — every Thought is, in effect, a fresh (if locally scoped) re-evaluation of what to do given everything observed so far, so it adapts to a failed action almost for free, at the granularity of one step. Plan-and-Execute must replan *explicitly and deliberately*: it needs a mechanism to detect that the current plan is no longer valid (a step failed, or its result contradicts what the plan assumed), and it needs to decide how much of the plan to discard versus keep. Get this wrong — replan too eagerly — and you lose the efficiency gains that were the whole point of planning upfront, converging toward ReAct's step-by-step cost profile with worse latency. Replan too conservatively, or without noticing a stale assumption, and the agent will doggedly execute a plan that no longer makes sense given what it has since learned, silently producing garbage.

| Dimension | ReAct | Plan-and-Execute |
|---|---|---|
| When the strategy is decided | Continuously, one step at a time | Upfront, then periodically on failure |
| Parallel execution of independent subtasks | Not natural | Natural, since dependencies are explicit |
| Cost of a single wrong turn | One wasted step | Potentially a wasted plan (or sub-branch of it) |
| Human review point | Awkward — must interrupt mid-loop | Natural — review the plan before execution |
| Adaptiveness to surprises | High, automatic | Requires an explicit replanning trigger |
| Best suited to | Exploratory, unpredictable tasks | Decomposable tasks with identifiable structure |

## When a Fixed Upfront Plan Is Better — and When It Isn't

A fixed plan is the right call when the task's decomposition is genuinely stable with respect to what the agent is likely to discover along the way. "Research three competitor products and write a comparison" decomposes cleanly into "research product A," "research product B," "research product C," "write comparison" regardless of what the research actually turns up — the *structure* of the task doesn't depend on intermediate results, even though the *content* does. Tasks like this benefit from planning once, executing largely in parallel, and only replanning on outright failures (a source is unreachable, a tool errors out).

A fixed plan is the wrong call when the task's structure itself is contingent on what earlier steps reveal — not just their content, but whether subsequent steps are even needed at all. A diagnostic or troubleshooting task ("why is this service returning errors") is the clearest example: what you check second depends entirely on what the first check found, in a way that cannot be usefully pre-enumerated. Writing an upfront plan like "1. check logs, 2. check database, 3. check network" for such a task is a false economy — steps 2 and 3 might be entirely unnecessary or entirely insufficient depending on what step 1 reveals, and any plan written before running step 1 is guessing. This is precisely the regime where ReAct's continuous replanning is not overhead but the actual point.

A useful heuristic in practice: ask whether an experienced human doing this task would sketch an outline before starting, or would insist "I can't know what to do next until I see what this first step turns up." If they'd sketch an outline, Plan-and-Execute (or a hybrid, see below) is appropriate. If they'd insist on taking it one step at a time, ReAct is the better match — and forcing a plan onto that kind of task usually results in the "replan" branch firing so often that you have paid for the machinery of planning without getting any of its benefit.

## Task Decomposition Strategies

The quality of a Plan-and-Execute system is overwhelmingly determined by the quality of the decomposition produced in the planning phase — a good executor cannot rescue a bad plan, since it has no mandate to question the plan's structure, only to carry out what it's given. Three decomposition strategies show up repeatedly, and picking the right one for the shape of the task materially affects how well the plan holds up.

**Hierarchical decomposition** breaks the task into major phases first, then recursively breaks each phase into finer steps only as needed — "write the report" becomes "gather data / analyze / draft / revise," and only "gather data" gets expanded further into concrete sub-steps once the planner reaches it. This is the natural fit for large, multi-phase tasks where committing to a fully detailed plan for phase 3 before phase 1 has even started would mean planning against information you don't have yet — the classic case being research or writing tasks where what phase 2 needs depends on what phase 1 actually found.

**Dependency-based decomposition** starts from the opposite direction: identify what depends on what, first, and let the step list fall out of the dependency graph. This is the strategy embedded in the `_execute_plan` code above (steps with `depends_on` fields), and it's the right default when the main value you want from planning is *parallelism* — the point of front-loading the dependency analysis is precisely to discover, before execution starts, which steps have no dependency relationship to each other and can therefore run concurrently.

**Goal-oriented decomposition** works backward from the desired end state to the sub-goals that would produce it, rather than forward from the starting state. This is often the more natural framing for planning LLM prompts, because it maps directly onto how the planning prompt is phrased ("what needs to be true for the final goal to be satisfied, and what would make each of those true in turn") and tends to produce plans that stay anchored to the actual objective rather than drifting into "obviously useful sounding" steps that don't actually serve the stated goal — a real failure mode when a planning LLM is simply asked to "list steps" without being anchored back to the goal at each level of decomposition.

In practice, production planners often combine all three: hierarchical for the top-level phase structure, dependency-based within each phase to extract parallelism, and a goal-oriented check as a final validation pass ("does completing all of these steps actually satisfy the stated goal, or have I planned a plausible-sounding sequence that misses something") before handing the plan to the executor.

## A Fuller Worked Example: Trip Planning

To see the phase structure and the replanning path exercised together, consider the goal "Plan a 3-day trip to Lisbon next month, staying under $1200 including flights, with at least one day free for museums."

The planning phase might produce:

```json
[
  {"id": "flights", "description": "Find round-trip flights to Lisbon under $500", "depends_on": []},
  {"id": "hotel", "description": "Find 3 nights of lodging under $450 total", "depends_on": []},
  {"id": "museums", "description": "Identify top-rated museums open on a weekday", "depends_on": []},
  {"id": "itinerary", "description": "Assemble day-by-day itinerary with one full museum day", "depends_on": ["flights", "hotel", "museums"]},
  {"id": "budget_check", "description": "Verify total cost stays under $1200", "depends_on": ["flights", "hotel"]}
]
```

Notice `flights`, `hotel`, and `museums` share no dependencies and can execute in parallel — three tool-using sub-agents running concurrently, each scoped narrowly enough to be simple. Suppose execution finds flights at $520, five dollars over the assumed sub-budget. The `budget_check` step fails not because a tool errored, but because the *result* violates a constraint the plan assumed would hold. This is the deviation-based replanning case rather than the failure-driven case: nothing crashed, but the plan's premise (flights would cost under $500) turned out false, so replanning needs to renegotiate the budget split — perhaps directing the hotel search to target $380 instead of $450 to keep the $1200 total intact — rather than simply retrying the flights step with the same parameters, which would just return the same $520 result again.

This distinction between **failure-driven replanning** (a step errored outright — a tool timed out, an API returned 404) and **deviation-based replanning** (every step "succeeded" but the aggregate result no longer satisfies the original goal's constraints) is one that simplistic Plan-and-Execute implementations often miss, because it's tempting to only wire up replanning to trigger on exceptions. A production system needs an explicit check, like the `budget_check` step above, whose entire job is to evaluate the *plan's actual outcome against the original goal* — not just whether each individual step technically completed without throwing.

## Plan Validation Before Execution

Because the plan is a static, inspectable artifact before any execution happens, it can — and in production systems, should — be validated *before* a single tool call is made. This is one of Plan-and-Execute's underused advantages relative to ReAct, where there is no equivalent moment to "check the whole strategy" before committing to the first action.

```python
def validate_plan(plan: list[dict]) -> list[str]:
    problems = []
    ids = {step["id"] for step in plan}

    for step in plan:
        for dep in step["depends_on"]:
            if dep not in ids:
                problems.append(f"Step '{step['id']}' depends on unknown step '{dep}'")

    if _has_cycle(plan):
        problems.append("Plan contains a circular dependency")

    if len(plan) > 25:
        problems.append("Plan has an unusually large number of steps — likely over-decomposed")

    return problems
```

Cheap static checks like dependency-reference validation and cycle detection catch a class of planner errors (the LLM referencing a step ID that doesn't exist, or creating an accidental cycle by having step 2 depend on step 4 which depends on step 2) before they turn into a confusing runtime failure deep inside execution — precisely the kind of defect that is trivial to catch statically and much more expensive to debug after three parallel tool calls have already fired.

## Cost and Latency Accounting for Planning Itself

It's worth being explicit that the planning phase is not free, and its cost has to be weighed against the savings it's meant to produce. A planning call over a complex goal with many tools can itself consume a large prompt (full tool catalog, goal, any relevant context) and produce a lengthy structured output — for a genuinely simple task, the planning call alone can cost more than the entire task would have cost as a two-step ReAct trajectory. This is why production systems commonly add a cheap upfront triage step — a small classifier or a short LLM call that estimates whether a request needs full planning at all — before committing to the Plan-and-Execute machinery, reserving it for requests above some complexity threshold and routing simpler requests straight to a lighter-weight ReAct loop or even a single tool call.

## Replanning Strategies Compared

Three distinct triggers for replanning show up across production systems, and they are not interchangeable — each catches a different class of problem, and a mature system typically implements more than one.

**Failure-driven replanning** fires when a step raises an outright error — a tool times out, an API returns an error status, a required input is missing. This is the easiest to implement (it's a straightforward exception handler) and the least likely to be skipped by accident, because the failure is loud and unambiguous.

**Deviation-based replanning** fires when every step technically "succeeds" but the accumulated result no longer satisfies a constraint from the original goal — the budget-check example earlier in this chapter is exactly this case. This is harder to implement well because it requires the plan to include explicit checks against the original goal's constraints as first-class steps, not just an assumption that individual step success implies overall success; a system that only wires up failure-driven replanning will silently deliver a technically-completed but goal-violating result in exactly this scenario.

**Periodic replanning** fires on a fixed cadence or after a fixed number of steps, independent of whether anything has failed or deviated yet, and is used less to catch errors than to incorporate new information that has become available since the plan was made — useful for long-running plans where the world can change meaningfully mid-execution (a multi-hour data-gathering plan where prices, availability, or external conditions may shift while it runs). This is the most expensive of the three to run indiscriminately, since it triggers LLM replanning calls even when nothing is actually wrong, and is typically reserved for plans with unusually long execution windows.

| Trigger | Catches | Typical cost | Common failure mode if omitted |
|---|---|---|---|
| Failure-driven | Hard errors (timeouts, exceptions, missing data) | Low — only fires on actual failures | Agent crashes or silently drops a failed sub-task |
| Deviation-based | Constraint violations despite "successful" steps | Medium — requires explicit constraint-check steps | Plan completes but silently violates the original goal |
| Periodic | Staleness from a changing world during long executions | High — fires regardless of whether anything is wrong | Plan executes against stale assumptions from its start time |

## Production Case Study: A Report-Generation Pipeline

A concrete production pattern worth internalizing: a financial-analysis agent tasked with producing a quarterly summary report was originally built as a single long ReAct trajectory — one continuous loop reasoning about which data to pull next, one step at a time. It worked, but averaged 40+ sequential tool calls and correspondingly high latency, because the agent had no way to know upfront that "pull revenue by region" and "pull headcount by department" were entirely independent lookups that didn't need to happen in sequence.

Restructuring it as Plan-and-Execute — an upfront plan identifying roughly a dozen independent data-pull steps followed by a single synthesis step — cut wall-clock latency by more than half, since the independent pulls could run concurrently, and reduced total token cost because the growing-transcript cost driver described in the previous chapter (resending the whole history on every step) no longer applied across the parallelized pulls, each of which only needed its own small, scoped context rather than the entire trajectory's history. The one piece of ReAct-style adaptiveness that was preserved: each individual data-pull step was itself implemented as a small, bounded ReAct sub-loop (in case a specific query needed a retry with adjusted parameters), which is exactly the hybrid design described above — planning for the stable, parallelizable outer structure, and local reactive loops for the genuinely uncertain inner steps.

## Hybrid Designs Used in Production

Few production systems are purely one or the other. A common and effective hybrid is to plan at a coarse grain and execute each coarse step with a ReAct-style sub-agent: the top-level plan says "gather pricing data for competitor X," and a small ReAct loop is free to interleave reasoning and tool calls (searching, following links, retrying malformed queries) *within* that one subtask without needing to re-derive the overall report structure. This captures Plan-and-Execute's benefits at the level the task structure is genuinely stable (the overall shape of "gather → gather → gather → synthesize") while retaining ReAct's adaptiveness at the level where local uncertainty is highest (exactly how to find a given competitor's pricing page). LangGraph-style graph architectures, covered later in this series, are frequently used to implement exactly this kind of nested structure, where a top-level graph encodes the plan's dependency structure and individual nodes internally run their own bounded reasoning loops.
