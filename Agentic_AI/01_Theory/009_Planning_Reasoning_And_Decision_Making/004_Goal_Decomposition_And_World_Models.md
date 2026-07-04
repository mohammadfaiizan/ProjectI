# Goal Decomposition and World Models

## Table of Contents

1. Why a Goal Isn't a Plan
2. Decomposing a Goal into Subgoals
3. Decomposition Strategies: Top-Down vs. Bottom-Up
4. Dependency Ordering and Parallelism
5. What a World Model Is, Concretely
6. Building and Maintaining a Belief State
7. Detecting Plan Invalidation
8. Re-Planning When the World Model Was Wrong
9. Hierarchical Plans and Partial Re-Planning
10. Production Guidance and Interview Framing

---

## 1. Why a Goal Isn't a Plan

"Write a report summarizing Q3 sales performance and email it to the leadership team" is a goal, not a plan — it describes a desired end state without specifying the sequence of actions that gets you there, and an agent handed only this sentence has real work to do before it can act. It needs to figure out that this actually requires several distinct pieces of work (pulling sales data, computing the relevant aggregates, drafting prose, formatting it presentably, identifying who "the leadership team" refers to, composing and sending an email), that some of those pieces depend on others being done first (you can't draft the report before you have the numbers), and that some can happen independently of each other. This translation from a single, high-level goal statement into an ordered, actionable set of subgoals is goal decomposition, and it is the planning-side counterpart to everything covered in the earlier chapters of this section — chain/tree/graph-of-thought structure how the agent reasons, self-critique checks the quality of what it produces, search-based planning compares candidate actions before committing, and goal decomposition is what turns a vague objective into the concrete units of work that all of those other techniques then operate on.

The second half of this chapter addresses a problem that only shows up once an agent starts acting in the world rather than just reasoning about it in the abstract: the agent's understanding of the current state of the world — what data exists, what's already been done, what the environment currently looks like — is itself just a belief, built from what the agent has observed so far, and that belief can be incomplete or simply wrong. A plan built on a wrong belief about the world can look perfectly sound on paper and still fail, or worse, silently produce an incorrect result, when the world turns out not to match the assumptions baked into the plan. Maintaining an explicit model of "what I currently believe to be true about the world," and having a disciplined way to notice and react when that belief turns out to be false, is what separates an agent that gracefully recovers from surprises from one that barrels ahead executing a plan built on assumptions that stopped being true two steps ago.

## 2. Decomposing a Goal into Subgoals

Effective decomposition rests on a small number of principles worth internalizing before looking at any specific algorithm. A subgoal should be independently verifiable — you should be able to look at it in isolation and determine whether it's been achieved, without needing to re-examine the whole original goal. "Pull Q3 sales figures from the data warehouse" is independently verifiable (the data either got pulled correctly or it didn't); "make the report good" is not, because "good" isn't checkable against anything concrete. This matters because verifiable subgoals are what make progress trackable and what make partial failure detectable early rather than only at the very end when the final output is checked against the original, much vaguer goal.

A subgoal should also be actionable at the right granularity for whatever is going to execute it — a subgoal decomposed for a system with a SQL-query tool available should bottom out at something like "run this query against the sales table," while a subgoal decomposed for a system without direct data access might need to bottom out one level higher, at "request the Q3 figures from the analytics team's shared dashboard." This is why decomposition can't be done purely top-down from the goal without reference to what capabilities actually exist — a purely goal-driven decomposition can produce beautifully logical subgoals that don't correspond to anything the agent can actually execute, which is the specific failure top-down decomposition is prone to and which the bottom-up approach in the next section is designed to avoid.

Finally, subgoals need explicit dependency information, not just a flat list — "draft the report" depends on "get the sales data," and knowing this is what allows correct sequencing, safe parallelization, and, critically for the second half of this chapter, targeted re-planning: if a fact underlying "get the sales data" turns out to be wrong, you need to know precisely which downstream subgoals are now suspect and need re-doing, rather than being forced to guess or redo everything.

## 3. Decomposition Strategies: Top-Down vs. Bottom-Up

Top-down decomposition starts from the goal and recursively breaks it into smaller pieces, stopping when a piece is simple enough to execute directly. This mirrors how a human might naturally think about a big task — start broad, keep subdividing until each piece is small enough to just do. Its strength is that it stays anchored to the actual goal throughout, so the resulting subgoals are unlikely to drift from what was actually asked for. Its weakness, as noted above, is that it can produce a logically clean decomposition that has no relationship to the tools or capabilities actually available, because nothing in the process forces it to check against what's executable until the very end.

```python
import json

class TopDownDecomposer:
    def __init__(self, llm, max_depth=3):
        self.llm = llm
        self.max_depth = max_depth

    def decompose(self, goal: str, depth: int = 0) -> dict:
        if depth >= self.max_depth:
            return {"task": goal, "leaf": True, "children": []}

        response = self.llm.generate(f"""
        Task: {goal}

        Break this into 2-5 subtasks that together fully achieve it.
        For each subtask, state whether it is simple enough to execute
        directly (leaf) or still needs further breakdown, and what it
        depends on among its siblings.

        Return JSON:
        [{{"task": "...", "is_leaf": true/false, "depends_on_siblings": ["..."]}}]
        """)
        subtasks = json.loads(response)

        children = []
        for st in subtasks:
            if st["is_leaf"]:
                children.append({"task": st["task"], "leaf": True,
                                  "depends_on": st["depends_on_siblings"], "children": []})
            else:
                sub = self.decompose(st["task"], depth + 1)
                sub["depends_on"] = st["depends_on_siblings"]
                children.append(sub)

        return {"task": goal, "leaf": False, "children": children}
```

Bottom-up decomposition inverts this: it starts from the concrete set of actions/tools actually available and asks the model to compose a sequence of them that achieves the goal, rather than starting from the goal and hoping the resulting pieces happen to be executable. Its strength is the mirror image of top-down's weakness — every subgoal it produces is, by construction, grounded in something the agent can actually do. Its weakness is the mirror image of top-down's strength: because it's anchored to available capabilities rather than to the goal's natural structure, it can produce a plan that technically uses valid tools in a valid order but subtly misses part of what the goal actually required, especially when the goal has an implicit requirement that doesn't map cleanly onto any single available tool.

```python
class BottomUpPlanner:
    def __init__(self, llm, available_tools):
        self.llm = llm
        self.tools = available_tools

    def plan(self, goal: str) -> list[dict]:
        tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in self.tools)

        response = self.llm.generate(f"""
        Goal: {goal}

        Available tools (use ONLY these):
        {tool_descriptions}

        Compose a sequence of tool calls that achieves the goal. Order
        steps by dependency. If some part of the goal cannot be achieved
        with the available tools, say so explicitly rather than omitting
        it silently.

        Return JSON:
        [{{"step": 1, "tool": "...", "params": {{}}, "purpose": "...", "depends_on": []}}]
        or {{"achievable": false, "gap": "..."}}
        """)
        return json.loads(response)
```

The practical answer to "which one should I use" is usually both, in sequence: decompose top-down to get a goal-faithful breakdown, then validate each resulting leaf subgoal against the actual available tool set, folding any leaf that doesn't map cleanly onto an available capability back into a bottom-up composition step. This hybrid catches top-down's blind spot (subgoals that sound right but aren't executable) without inheriting bottom-up's blind spot (a plan that's executable but silently incomplete relative to the goal), at the cost of an extra validation pass.

## 4. Dependency Ordering and Parallelism

Once subgoals and their dependencies are known, a topological sort turns the dependency structure into a valid execution order, and grouping subgoals by "everything whose dependencies are already satisfied" turns that same structure into a set of parallel execution batches — steps within a batch have no dependency relationship to each other and can be executed concurrently, which matters directly for latency in any agent that's willing to make multiple tool calls at once.

```python
class DependencyGraph:
    def __init__(self):
        self.tasks: dict[str, dict] = {}
        self.dependencies: dict[str, set[str]] = {}

    def add_task(self, task_id: str, data: dict):
        self.tasks[task_id] = data
        self.dependencies.setdefault(task_id, set())

    def add_dependency(self, task_id: str, depends_on: str):
        self.dependencies[task_id].add(depends_on)

    def execution_order(self) -> list[str]:
        visited, order, in_progress = set(), [], set()

        def visit(node):
            if node in in_progress:
                raise ValueError(f"Circular dependency detected at {node}")
            if node in visited:
                return
            in_progress.add(node)
            for dep in self.dependencies.get(node, set()):
                visit(dep)
            in_progress.discard(node)
            visited.add(node)
            order.append(node)

        for task_id in self.tasks:
            if task_id not in visited:
                visit(task_id)
        return order

    def parallel_batches(self) -> list[list[str]]:
        order = self.execution_order()
        batches, done = [], set()
        while len(done) < len(order):
            ready = [t for t in order if t not in done and self.dependencies[t].issubset(done)]
            if not ready:
                break  # shouldn't happen if execution_order succeeded
            batches.append(ready)
            done.update(ready)
        return batches
```

The circular-dependency check is not a defensive nicety — LLM-generated decompositions do occasionally produce cyclic dependency claims (subgoal A depends on B, and the model also states B depends on A, usually from generating dependency lists per-subtask without a global consistency check), and silently proceeding with a cyclic graph produces either an infinite loop or an arbitrary, unexplained execution order, neither of which fails loudly enough to be caught quickly without an explicit check like this one.

## 5. What a World Model Is, Concretely

A "world model" in the context of an LLM agent is not a physics simulator or anything exotic — it's simply an explicit, structured record of what the agent currently believes to be true about the state of the environment it's operating in, separate from the conversational history that produced those beliefs. This distinction matters more than it sounds like it should. An agent that only has its raw conversation and tool-call transcript has to re-derive "what do I currently believe about the world" by re-reading that transcript every time it needs to know, which is both expensive (long transcripts) and unreliable (a fact stated early in a long transcript can effectively get lost or contradicted by something stated later, with no clear resolution of which one is currently true). A world model, by contrast, is a maintained, current snapshot — a dictionary or object representing the agent's best current understanding of specific, named facts about the environment — that gets explicitly updated as new observations come in, rather than being re-inferred from scratch each time.

Concretely, for the report-writing example from Section 1, a world model might track: which data sources have been queried and what they returned, which recipients have been confirmed to exist and have valid email addresses, which sections of the report have been drafted and at what quality/completeness level, and what the current best estimate is of remaining work. Each of these is a belief with an associated confidence and provenance (where did this belief come from, and when was it last confirmed), which is exactly what makes it possible to reason later about which beliefs might be stale or wrong.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class Belief:
    value: object
    confidence: float           # 0.0-1.0, how sure the agent is this is still true
    source: str                 # what observation/tool call produced this belief
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stale: bool = False         # explicitly flagged when something calls it into question


class WorldModel:
    """A structured, queryable belief state, separate from raw conversation history."""

    def __init__(self):
        self.beliefs: dict[str, Belief] = {}

    def update(self, key: str, value, confidence: float, source: str):
        self.beliefs[key] = Belief(value=value, confidence=confidence, source=source)

    def get(self, key: str) -> Belief | None:
        return self.beliefs.get(key)

    def mark_stale(self, key: str):
        if key in self.beliefs:
            self.beliefs[key].stale = True

    def stale_beliefs(self) -> list[str]:
        return [k for k, b in self.beliefs.items() if b.stale]

    def low_confidence_beliefs(self, threshold: float = 0.6) -> list[str]:
        return [k for k, b in self.beliefs.items() if b.confidence < threshold]

    def snapshot(self) -> dict:
        """A compact, current summary suitable for injecting into a planning prompt --
        much cheaper and more reliable than re-reading a full transcript."""
        return {
            k: {"value": b.value, "confidence": b.confidence, "stale": b.stale}
            for k, b in self.beliefs.items()
        }
```

## 6. Building and Maintaining a Belief State

A world model is only useful if it's actually kept current, which means every observation the agent makes — every tool result, every piece of retrieved information, every user message that reveals something about the environment — needs to flow through an update step rather than just being appended to a transcript and left there. The update step has two jobs: extracting which specific beliefs a new observation bears on, and deciding how that observation should change the existing belief (confirm it, contradict it, refine it, or introduce a wholly new belief that didn't exist before).

```python
class BeliefUpdater:
    def __init__(self, llm, world_model: WorldModel):
        self.llm = llm
        self.world_model = world_model

    def process_observation(self, observation: str, source: str):
        import json
        current_snapshot = json.dumps(self.world_model.snapshot(), default=str)

        response = self.llm.generate(f"""
        Current beliefs about the world: {current_snapshot}
        New observation (from {source}): {observation}

        For each existing belief this observation bears on, state whether
        it CONFIRMS, CONTRADICTS, or REFINES that belief. Also list any
        genuinely new belief this observation introduces.

        Return JSON:
        {{
          "updates": [{{"key": "...", "relation": "confirms|contradicts|refines",
                         "new_value": "...", "confidence": 0.0-1.0}}],
          "new_beliefs": [{{"key": "...", "value": "...", "confidence": 0.0-1.0}}]
        }}
        """)
        result = json.loads(response)

        for u in result["updates"]:
            if u["relation"] == "contradicts":
                # Don't silently overwrite -- flag the old belief as stale so
                # downstream planning knows to double-check anything built on it.
                self.world_model.mark_stale(u["key"])
            self.world_model.update(u["key"], u["new_value"], u["confidence"], source)

        for nb in result["new_beliefs"]:
            self.world_model.update(nb["key"], nb["value"], nb["confidence"], source)
```

Note the asymmetric treatment of "contradicts" versus "confirms" or "refines": a contradiction gets flagged as stale in addition to being updated, which is what triggers the plan-invalidation logic in the next section. This is a deliberate design choice — silently overwriting a belief the moment new information disagrees with it loses the signal that something the agent previously acted on has just been called into question, which is exactly the signal that needs to propagate to whatever plan was built on top of that belief.

## 7. Detecting Plan Invalidation

A plan is invalidated when one of its steps depended on a belief about the world that has since turned out to be wrong, or when a step that was assumed to succeed didn't, in either case meaning the remaining plan can no longer be trusted to achieve the goal without re-examination. Detecting this requires two things working together: the dependency structure from Section 2-4, which tells you which subgoals relied on which beliefs, and the staleness tracking from Section 6, which tells you when a belief has changed in a way that calls its prior use into question. Combining them lets you identify precisely which downstream subgoals are now suspect, rather than either ignoring the contradiction (and executing a plan built on a false premise) or overreacting by scrapping the entire plan every time any single belief shifts even slightly.

```python
class PlanInvalidationChecker:
    def __init__(self, world_model: WorldModel, dependency_graph: DependencyGraph,
                 subgoal_belief_map: dict[str, list[str]]):
        self.world_model = world_model
        self.graph = dependency_graph
        # subgoal_belief_map: which belief keys each subgoal's plan depended on
        self.subgoal_belief_map = subgoal_belief_map

    def find_invalidated_subgoals(self, completed_subgoals: set[str]) -> set[str]:
        stale_keys = set(self.world_model.stale_beliefs())
        if not stale_keys:
            return set()

        directly_invalidated = {
            sg for sg, keys in self.subgoal_belief_map.items()
            if set(keys) & stale_keys and sg in completed_subgoals
        }

        # Anything downstream of a directly invalidated subgoal is suspect too,
        # even if its own beliefs weren't directly touched -- it may have been
        # built on the invalidated subgoal's now-questionable output.
        transitively_invalidated = set(directly_invalidated)
        changed = True
        while changed:
            changed = False
            for sg, deps in self.graph.dependencies.items():
                if sg in completed_subgoals and sg not in transitively_invalidated:
                    if deps & transitively_invalidated:
                        transitively_invalidated.add(sg)
                        changed = True

        return transitively_invalidated
```

The transitive-closure step matters in practice: if "get sales data" is invalidated because the world model now believes the query hit the wrong fiscal quarter, then "draft report section on Q3 revenue," which consumed that data, is invalidated too, even though nothing about the drafting subgoal's own beliefs changed — its output was built on now-suspect input. Skipping this transitive step is a common shortcut that produces agents which correctly notice a fact changed but then continue trusting stale downstream work that was built on top of it.

## 8. Re-Planning When the World Model Was Wrong

Once invalidated subgoals are identified, re-planning should be scoped to exactly that invalidated subset wherever possible, rather than discarding the entire plan and starting over — this is both a cost optimization (redoing only what's actually suspect is cheaper than redoing everything) and a correctness one (subgoals unaffected by the belief change are still valid and re-deriving them from scratch risks introducing new inconsistencies for no benefit). The re-planning step needs the updated world model, the set of subgoals to redo, and the parts of the original plan that remain trustworthy, so the model has the full context needed to produce a coherent partial replacement rather than a plan that's internally inconsistent with the parts that were kept.

```python
class Replanner:
    def __init__(self, llm, world_model: WorldModel):
        self.llm = llm
        self.world_model = world_model

    def replan(self, goal: str, invalidated_subgoals: set[str],
               original_plan: dict, retained_results: dict) -> dict:
        import json

        retained_summary = {
            sg: result for sg, result in retained_results.items()
            if sg not in invalidated_subgoals
        }

        response = self.llm.generate(f"""
        Overall goal: {goal}

        Current, updated beliefs about the world:
        {json.dumps(self.world_model.snapshot(), default=str)}

        These subgoals from the original plan are still valid and their
        results can be reused as-is:
        {json.dumps(retained_summary, default=str)}

        These subgoals were invalidated because they relied on beliefs
        that turned out to be wrong, and must be redone:
        {json.dumps(list(invalidated_subgoals))}

        Produce an updated plan ONLY for the invalidated subgoals, taking
        into account the updated beliefs and the retained results above so
        the new work is consistent with what's being kept.

        Return JSON in the same subgoal format as the original plan.
        """)
        return json.loads(response)
```

A subtlety worth calling out: re-planning prompts need to explicitly state which prior results are being kept and treated as ground truth for this pass, not just which beliefs changed. Without this, it's easy for a re-planning call to inadvertently also revise a retained subgoal's approach in a way that's now inconsistent with the parts of the plan that weren't touched — for example, changing the report's structure while redoing the data-pull step, leaving the untouched drafting subgoals referencing a structure that no longer matches. Making the retained work an explicit, fixed input to the re-planning prompt, rather than just background context, is what keeps the two halves of the plan (retained and redone) coherent with each other.

## 9. Hierarchical Plans and Partial Re-Planning

Everything in Sections 7-8 becomes both more important and more tractable when the original plan is hierarchical — organized into a small number of high-level phases, each of which decomposes further into detailed steps — rather than a single flat list. A flat plan of thirty steps offers no natural boundary for how much to redo when one belief turns out to be wrong; a hierarchical plan does, because invalidation can usually be contained within a phase, letting the re-planner work at the phase's internal level of detail without needing to reconsider the strategic structure above it, or, in more severe cases, letting the invalidation be recognized as invalidating an entire phase's *strategy*, in which case re-planning legitimately needs to happen at the higher level too.

```python
class HierarchicalReplanner:
    def __init__(self, llm, world_model: WorldModel):
        self.llm = llm
        self.world_model = world_model

    def assess_invalidation_scope(self, invalidated_subgoals: set[str],
                                    phase_map: dict[str, str]) -> dict[str, str]:
        """Decide, per affected phase, whether the damage is contained to
        tactical steps or requires revisiting the phase's strategy itself."""
        import json
        affected_phases = {phase_map[sg] for sg in invalidated_subgoals if sg in phase_map}
        scope = {}

        for phase in affected_phases:
            phase_subgoals = [sg for sg, p in phase_map.items() if p == phase]
            fraction_invalidated = len(
                [sg for sg in phase_subgoals if sg in invalidated_subgoals]
            ) / len(phase_subgoals)

            response = self.llm.generate(f"""
            Phase: {phase}
            Fraction of this phase's subgoals invalidated: {fraction_invalidated:.2f}
            Updated world beliefs: {json.dumps(self.world_model.snapshot(), default=str)}

            Does the phase's overall STRATEGY still make sense given these
            updated beliefs, or does the strategy itself need to change
            (not just the tactical steps within it)?

            Return JSON: {{"scope": "tactical_only" | "strategy_revision"}}
            """)
            scope[phase] = json.loads(response)["scope"]

        return scope
```

This escalation check — tactical re-planning within a phase versus revisiting the phase's strategy — is what prevents two opposite failure modes: over-reacting by re-deriving the entire high-level strategy every time a single low-level fact changes (expensive and unnecessary when the strategy doesn't actually depend on that fact), and under-reacting by only ever patching tactical steps even when the accumulated belief changes have actually undermined the phase's whole approach (which produces a plan that's locally patched but globally incoherent). A concrete example: if the world model update reveals that a particular data source is simply unavailable, and the phase's strategy was built entirely around using that source, patching individual steps that reference it won't fix the underlying problem — the phase needs a new strategy (a different data source, a different approach to estimation), and recognizing that this is a strategy-level, not tactical-level, invalidation is exactly what this check is for.

## 10. Production Guidance and Interview Framing

A few points are worth being able to state precisely if this comes up in an interview. First, goal decomposition is not just an organizational nicety — the explicit dependency structure it produces is the mechanism that makes targeted re-planning possible at all; without it, "something changed, what do we redo" has no principled answer better than "everything" or "guess." Second, a world model doesn't need to be exotic machine-learning infrastructure; in the vast majority of LLM agent systems it's a maintained, structured belief store with per-belief confidence and provenance, updated by an explicit LLM-driven or rule-driven process every time a new observation comes in, kept separate from raw conversation history precisely because raw history is expensive to re-derive beliefs from on demand and doesn't cleanly represent "what do I currently believe" when earlier and later statements disagree. Third, the single most common production bug in agents that lack this discipline is silent staleness: the agent keeps acting on a belief that was true when observed but has since become false, with no mechanism that would have caused it to notice — this is precisely what the stale-flagging and invalidation-propagation logic in Sections 6-7 is designed to prevent, and being able to describe that specific mechanism, rather than gesturing vaguely at "the agent should adapt to changes," is what demonstrates real experience building these systems. Fourth, re-planning should default to scoped, partial re-planning rather than wholesale re-planning, both for cost reasons and because unnecessarily redoing valid work risks introducing fresh inconsistencies; the hierarchical escalation check in Section 9 is what decides, in a principled rather than ad hoc way, when a belief change is severe enough to warrant revisiting strategy rather than just tactics. Taken together, goal decomposition and world-model maintenance are what let an agent's plan degrade gracefully under real-world uncertainty — noticing what it got wrong, containing the blast radius of that mistake, and fixing exactly what needs fixing — rather than either blindly executing a plan that's silently gone stale or panicking into a full restart every time reality diverges even slightly from what was assumed at planning time.
