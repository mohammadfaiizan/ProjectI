# Agent Trajectory Evaluation

## Why Final-Answer Evaluation Isn't Enough for Agents

Everything in the previous chapter treated the system under evaluation as a function that maps an input to an output — you ask a question, the model produces text, you score the text. Agents break that model, because an agent isn't a single generation, it's a sequence of decisions: which tool to call, what arguments to pass, how to interpret the tool's result, whether to call another tool or stop and answer, and how to recover when something goes wrong along the way. That sequence of decisions is called the **trajectory**, and it is entirely possible — in fact, common — for an agent to reach a correct final answer via a badly broken trajectory, or to reach a wrong final answer via a trajectory that made every locally reasonable decision.

Consider a customer-support agent asked to refund an order. If the agent guesses the refund amount instead of calling the `get_order_details` tool, and by coincidence guesses correctly, final-answer evaluation says "pass." That's a false negative in your risk assessment — the same behavior on a different order will produce a wrong refund, and you had no way to see it coming because you were only looking at the output. Conversely, an agent that calls the right tools in the right order, correctly interprets an ambiguous tool result, and then makes one small formatting mistake in its final message has a trajectory that was almost entirely correct, and treating that as a full failure (as outcome-only scoring would) wastes the diagnostic signal that would tell you exactly where to intervene. Trajectory evaluation exists to close both gaps: it lets you audit *how* an answer was reached, not just whether the final string looks right, which is essential for debugging, for building trust in an autonomous system, and for catching bugs that only manifest statistically across many runs rather than in any single output.

## Anatomy of a Trajectory

A trajectory is the ordered log of everything an agent did while working a task: every intermediate reasoning step (if the agent exposes any, e.g. via a "thought" field in a ReAct-style loop), every tool call with its arguments, every tool result the agent observed, and the final response. Structurally it's usually represented as a list of steps, and almost every agent framework (LangGraph, custom ReAct loops, OpenAI's function-calling loop) produces something isomorphic to this shape even if the field names differ.

```python
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrajectoryStep:
    step_index: int
    thought: Optional[str]          # the agent's stated reasoning, if exposed
    tool_name: Optional[str]        # None if this step is a direct answer
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class Trajectory:
    task_id: str
    task_input: str
    steps: list[TrajectoryStep]
    final_answer: str
    total_latency_ms: float
    total_cost_usd: float
```

Having the trajectory in this structured form is what makes systematic evaluation possible at all — if all you persist is the final answer, you have thrown away the evidence needed to diagnose *why* a run succeeded or failed, and every debugging session degrades into re-running the agent and hoping to reproduce the issue. This is the single biggest practical argument for investing in structured tracing (covered in depth in the observability chapter) before you invest heavily in trajectory-level scoring: you cannot evaluate what you didn't record.

## Step-Level vs. Outcome-Level Evaluation

The two ends of the evaluation spectrum are outcome-level evaluation, which only looks at the final answer against some notion of ground truth, and step-level evaluation, which grades each individual decision the agent made along the way. Neither is sufficient alone, and understanding when to lean on each is a core part of designing an agent evaluation strategy.

**Outcome-level evaluation** answers "did the agent accomplish the task." For a well-defined task this can be surprisingly objective: did the calendar event actually get created with the right time, did the refund get issued for the right amount, does the generated code pass the test suite, does the final database state match the expected state after the agent's actions. This objectivity is a major advantage — when you can define outcome success as an executable check against real side effects, you get a metric that is immune to LLM-judge bias entirely, and you should prefer this whenever the task produces a checkable side effect. The blind spot is diagnostic: outcome-level evaluation tells you *that* something went wrong on a failing run, but not *where*, and on a passing run it can hide a fragile, lucky, or expensive path to success that will fail differently on the next input.

**Step-level (also called process-level or trajectory-level) evaluation** answers a different, more granular question at each decision point: given everything the agent knew at that point, was this the right tool to call, were the arguments correct, was the interpretation of the previous result sound, was stopping (or not stopping) the right call. This requires either a reference trajectory to compare against, or a judge capable of assessing each step's reasonableness given the context available to the agent at that point — not given the full task in hindsight, which is an important and easy-to-get-wrong distinction, since a step can be a perfectly reasonable decision given incomplete information even if it later turns out to be a wrong turn.

```python
def evaluate_step(step: TrajectoryStep, context_so_far: list[TrajectoryStep], judge_llm) -> dict:
    """Grade a single step given only the information the agent had at
    that point in the trajectory -- not the full trajectory or final outcome,
    to avoid hindsight bias in the judgment."""
    history_summary = "\n".join(
        f"Step {s.step_index}: called {s.tool_name}({s.tool_args}) -> {s.tool_result}"
        for s in context_so_far
    )
    prompt = f"""Given the agent's history so far:
{history_summary}

The agent then took this action:
Thought: {step.thought}
Tool called: {step.tool_name}
Arguments: {step.tool_args}

Was this a reasonable next action given only the information available at
this point? Consider: was the right tool chosen, were arguments correct
and complete, was this action necessary (not redundant with an earlier step).

Return JSON: {{"reasonable": true/false, "issue": "<description or null>"}}
"""
    return judge_llm.generate_json(prompt)
```

In practice the two levels are complementary and are usually run together: outcome-level checks act as a fast, cheap pass/fail gate across a large eval set, and step-level evaluation is run selectively — on all failing trajectories always, and on a sample of passing trajectories to catch the "right answer, wrong path" failure mode before it compounds into a real incident.

## Dimensions of Trajectory Quality

When scoring a trajectory, it helps to break "was this a good run" into independent, separately measurable dimensions rather than one fuzzy composite score, because each dimension points at a different part of the system to fix.

**Tool selection correctness** asks whether the agent called the right tool for the situation, as opposed to a plausible-sounding but wrong one, or no tool at all when one was needed (hallucinating an answer instead of looking it up). This is usually the highest-leverage thing to check first, because a wrong tool call invalidates everything downstream of it.

**Argument correctness** asks whether the arguments passed to a correctly-chosen tool were themselves right — correct types, correct values, no missing required fields, no hallucinated values invented to fill a slot the agent didn't actually have information for. This failure mode is sneaky because it doesn't show up as an error; the tool call succeeds, it just succeeds with wrong inputs, silently poisoning the rest of the run.

**Ordering and dependency correctness** asks whether steps happened in a valid sequence — did the agent check inventory before promising a shipping date, did it authenticate before calling an authenticated endpoint, did it avoid calling a tool that depends on the output of a tool it hasn't called yet. Some tasks have a strict required ordering; others have multiple valid orderings, which matters when you design a reference trajectory to compare against (a rigid step-by-step diff will falsely penalize a valid reordering).

**Efficiency** asks whether the agent took a reasonable number of steps, or wandered — calling the same tool redundantly, retrying without changing strategy, or taking five steps to do what two steps could accomplish. Efficiency matters commercially (every step is tokens and latency) as much as it matters for quality, and unbounded inefficiency is often the first visible symptom of a deeper reasoning problem, like the agent not correctly incorporating a tool result into its next decision.

**Error recovery** asks what the agent does when a tool call fails or returns an unexpected result — does it retry sensibly, fall back to an alternative approach, ask the user for clarification, or does it hallucinate a plausible-sounding result to paper over the failure. This last behavior (fabricating success when the underlying action failed) is one of the most dangerous agent failure modes because it's invisible from the final answer alone; only trajectory-level inspection of the tool result versus what the agent claimed happened can catch it.

**Groundedness of reasoning** asks whether the agent's stated "thought" at each step is actually consistent with, and justified by, the tool results it has seen — as opposed to reasoning that ignores available evidence or asserts something the tools never returned. This overlaps with hallucination detection (covered in the next chapter) but applied specifically to intermediate reasoning rather than just the final answer.

```python
class TrajectoryEvaluator:
    def __init__(self, judge_llm):
        self.judge_llm = judge_llm

    def evaluate(self, trajectory: Trajectory, expected_tools: list[str] = None) -> dict:
        return {
            "tool_selection": self._score_tool_selection(trajectory, expected_tools),
            "argument_correctness": self._score_arguments(trajectory),
            "efficiency": self._score_efficiency(trajectory),
            "error_recovery": self._score_error_recovery(trajectory),
            "groundedness": self._score_groundedness(trajectory),
        }

    def _score_tool_selection(self, trajectory: Trajectory, expected_tools) -> dict:
        called = [s.tool_name for s in trajectory.steps if s.tool_name]
        if expected_tools is None:
            return {"score": None, "note": "no reference tool sequence provided"}
        # Order-insensitive set comparison first, then flag ordering issues separately
        missing = [t for t in expected_tools if t not in called]
        extra = [t for t in called if t not in expected_tools]
        return {
            "score": 1.0 if not missing and not extra else 0.0,
            "missing_tools": missing,
            "unnecessary_tools": extra,
        }

    def _score_arguments(self, trajectory: Trajectory) -> dict:
        issues = []
        for step in trajectory.steps:
            if step.tool_name and not step.tool_args and step.tool_name != "no_op":
                issues.append(f"Step {step.step_index}: {step.tool_name} called with no arguments")
        return {"score": 1.0 if not issues else 0.0, "issues": issues}

    def _score_efficiency(self, trajectory: Trajectory) -> dict:
        tool_calls = [s.tool_name for s in trajectory.steps if s.tool_name]
        duplicate_calls = len(tool_calls) - len(set(tool_calls))
        # Not all duplicates are bad (pagination, retries after real changes),
        # but a high raw duplicate count is a strong efficiency smell.
        return {
            "total_steps": len(trajectory.steps),
            "duplicate_tool_calls": duplicate_calls,
            "score": max(0.0, 1.0 - 0.2 * duplicate_calls),
        }

    def _score_error_recovery(self, trajectory: Trajectory) -> dict:
        recoveries = []
        for i, step in enumerate(trajectory.steps):
            if step.error and i + 1 < len(trajectory.steps):
                next_step = trajectory.steps[i + 1]
                recovered = next_step.tool_name is not None  # agent tried something else
                recoveries.append(recovered)
        if not recoveries:
            return {"score": None, "note": "no errors occurred in this trajectory"}
        return {"score": sum(recoveries) / len(recoveries), "recovery_attempts": len(recoveries)}

    def _score_groundedness(self, trajectory: Trajectory) -> dict:
        ungrounded_steps = []
        for step in trajectory.steps:
            if not step.thought:
                continue
            verdict = self.judge_llm.generate_json(f"""
            Prior tool result: {step.tool_result}
            Agent's stated reasoning: {step.thought}

            Does the reasoning rely only on information actually present in
            the tool result (or general world knowledge), with no invented
            facts? Return JSON: {{"grounded": true/false, "issue": "..."}}
            """)
            if not verdict.get("grounded", True):
                ungrounded_steps.append({"step": step.step_index, "issue": verdict.get("issue")})
        return {
            "score": 1.0 - len(ungrounded_steps) / max(len(trajectory.steps), 1),
            "ungrounded_steps": ungrounded_steps,
        }
```

## Reference-Based vs. Reference-Free Trajectory Scoring

Just as with final-answer evaluation, trajectory scoring splits into approaches that compare against a known-good reference and approaches that judge quality without one, and the trade-offs mirror each other closely.

**Reference-based scoring** requires a "golden trajectory" — a human-authored or human-approved sequence of expected tool calls for a given input — and computes some form of similarity or edit distance between the actual and golden trajectories. This is precise and cheap to compute once you have the reference, but authoring golden trajectories is expensive (a human has to actually work through the task and decide the correct tool sequence) and brittle whenever a task legitimately has multiple valid paths to the same outcome — a naive sequence-equality check will fail a trajectory that solved the problem correctly but in a different, equally valid order. A common middle ground is scoring against a golden *set* of required tool calls (order-insensitive) combined with an explicit list of hard ordering constraints only where ordering genuinely matters (must-authenticate-before-write, must-check-inventory-before-promising-shipment), rather than demanding an exact sequence match.

```python
def trajectory_edit_distance(actual: list[str], golden: list[str]) -> int:
    """Levenshtein distance over the tool-call sequence -- a cheap way to
    quantify 'how far off' a trajectory was, useful for regression tracking
    even when you don't need to explain *why* it diverged."""
    n, m = len(actual), len(golden)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if actual[i - 1] == golden[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]
```

**Reference-free scoring** uses an LLM judge to assess the trajectory's soundness step by step without needing a pre-authored golden path, similar to the `_score_groundedness` method above extended across every dimension. This scales much better — you don't need a human to author a reference for every eval case — but it inherits every LLM-as-judge reliability concern from the previous chapter (position bias, verbosity bias, the need to validate the judge against human-labeled trajectories before trusting it) and it's noticeably harder to write a reliable rubric for "was this a reasonable sequence of nine tool calls" than for "is this one paragraph a good summary," simply because there's more surface area and more ways for a judge to miss a subtle issue buried in step six of nine. The practical answer most production teams converge on is a hybrid: use executable outcome checks wherever the task allows it (did the right database rows actually change), reference-based tool-set checks for the small number of high-value or safety-critical flows worth hand-authoring golden trajectories for, and reference-free LLM judging as the broad-coverage fallback for everything else.

## Common Frameworks and Approaches

Several open-source projects and vendor products formalize trajectory evaluation, and being able to name and briefly characterize them signals real familiarity with the space.

**RAGAS**, despite its name suggesting RAG-only use, includes trajectory-adjacent metrics (context precision/recall, faithfulness) that generalize to any agent step that retrieves information before acting — you can apply its faithfulness-style checking to "was this tool call's interpretation faithful to the tool's actual return value." **LangSmith** (LangChain's observability and eval product) supports trajectory evaluators natively: you can register a Python function or an LLM-judge prompt that receives the full run tree (every step, tool call, and intermediate output) and returns a score, and it integrates directly with the tracing data LangChain/LangGraph agents already emit, which removes the "you can't evaluate what you didn't record" problem almost by construction. **Langfuse** offers a similar trace-native evaluation story with a stronger open-source/self-hosted angle. Google's `AgentEvals`-style libraries and Microsoft's agent evaluation tooling in Azure AI Studio both provide pre-built trajectory metrics (exact-match tool sequence, any-order tool-set match, and LLM-judged trajectory quality) as ready-made building blocks rather than requiring you to hand-roll the comparison logic shown above from scratch — in production you'd typically reach for one of these rather than reimplementing trajectory diffing yourself, though understanding the mechanics (as covered here) is what lets you configure and debug them correctly rather than treating them as a black box.

The other pattern worth knowing is **simulation-based trajectory evaluation**, used heavily for evaluating conversational and multi-turn agents: instead of replaying a fixed input, you run the agent against a second LLM playing the role of a user (or an adversarial user, for red-teaming) with a defined persona and goal, log the resulting multi-turn trajectory, and then apply the same step-level and outcome-level scoring to the simulated conversation. This is essential for agents whose real behavior only emerges over multiple turns — a single-turn eval set can't exercise the agent's ability to handle correction, follow-up, or changing user intent mid-task, all of which are exactly the trajectory-level failure modes that matter most once an agent handles anything beyond one-shot Q&A.

## Building a Trajectory Eval Pipeline in Practice

Pulling this together into something you'd actually run: collect a set of representative tasks, run the agent against each while capturing the full structured trajectory (not just the final answer — this is the tracing infrastructure investment mentioned earlier), apply outcome-level checks as a first filter, and route to step-level evaluation for anything that fails outcome checks plus a sample of everything that passes. Aggregate scores per dimension (tool selection, argument correctness, efficiency, error recovery, groundedness) rather than into a single number, and track each dimension's trend over time and per agent/prompt version, because a regression that only shows up in "efficiency" (more retries, more redundant calls) after a prompt change is a very different, and usually cheaper, problem than a regression in "tool selection correctness" — collapsing both into one composite score would make them indistinguishable and delay diagnosis.

```python
class TrajectoryEvalPipeline:
    def __init__(self, agent, evaluator: TrajectoryEvaluator):
        self.agent = agent
        self.evaluator = evaluator

    def run_suite(self, tasks: list[dict]) -> dict:
        all_results = []
        for task in tasks:
            trajectory = self.agent.run_and_trace(task["input"])
            outcome_passed = self._check_outcome(trajectory, task.get("expected_outcome"))

            step_scores = None
            if not outcome_passed or task.get("always_deep_eval"):
                step_scores = self.evaluator.evaluate(trajectory, task.get("expected_tools"))

            all_results.append({
                "task_id": task["id"],
                "outcome_passed": outcome_passed,
                "step_scores": step_scores,
                "steps_taken": len(trajectory.steps),
                "cost_usd": trajectory.total_cost_usd,
            })

        return self._aggregate(all_results)

    def _check_outcome(self, trajectory: Trajectory, expected) -> bool:
        if expected is None:
            return True
        if callable(expected):
            return expected(trajectory.final_answer)
        return trajectory.final_answer.strip() == expected.strip()

    def _aggregate(self, results: list[dict]) -> dict:
        outcome_pass_rate = sum(r["outcome_passed"] for r in results) / len(results)
        dimension_scores = {}
        scored = [r for r in results if r["step_scores"]]
        for dim in ["tool_selection", "argument_correctness", "efficiency", "error_recovery", "groundedness"]:
            values = [
                r["step_scores"][dim]["score"] for r in scored
                if r["step_scores"][dim].get("score") is not None
            ]
            if values:
                dimension_scores[dim] = sum(values) / len(values)

        return {
            "outcome_pass_rate": outcome_pass_rate,
            "avg_steps": sum(r["steps_taken"] for r in results) / len(results),
            "avg_cost_usd": sum(r["cost_usd"] for r in results) / len(results),
            "dimension_scores": dimension_scores,
        }
```

The final point worth internalizing is that trajectory evaluation is not a nice-to-have layered on top of final-answer evaluation — for any agent with real-world side effects (moving money, sending messages, modifying data, calling external APIs) it is the primary defense against a specific and costly class of incident: the agent that looks fine in aggregate metrics while quietly taking a wrong or unsafe path some fraction of the time. Final-answer evaluation alone cannot see that fraction; only the trajectory can.
