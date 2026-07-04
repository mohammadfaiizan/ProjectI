# Reflexion and Self-Correction Patterns

## The Intuition: Learning From a Mistake Without Retraining Weights

Every pattern covered so far — ReAct, Plan-and-Execute — improves an agent's behavior *within* a single task attempt. Reflexion, introduced by Shinn et al. (2023) in "Reflexion: Language Agents with Verbal Reinforcement Learning," addresses a different question: what should an agent do differently on its *next* attempt at a similar task, given that it failed (or underperformed) on a previous one, without the enormous cost and latency of actually updating the model's weights through fine-tuning or RL?

The answer the paper proposes is disarmingly simple in retrospect: after a failed attempt, ask the model to produce a natural-language self-critique of what went wrong and why — a "verbal" reward signal, in place of the scalar reward signal that traditional reinforcement learning would use to update a policy network — and store that critique in a persistent memory. On the next attempt at a related task, retrieve the relevant past critiques and inject them into the prompt as additional context. The model's weights never change; what changes is the information available to it at inference time. This is why the paper calls it "verbal reinforcement learning" — it captures the spirit of learning from a scored trial-and-error episode, but the medium of learning is text in a memory store rather than a gradient update.

This is a meaningfully different mechanism from a within-episode reflection loop (generate, critique, refine, repeat until acceptable), even though the two are often confused. A within-episode reflection loop improves a *single output* through iteration before that output is ever shown to anyone. Reflexion, properly understood, improves behavior *across episodes* — task attempt 1 fails, the critique of that failure is banked, and task attempt 2 (potentially a different task instance, or a retry of the same one) benefits from it. Production systems often use both, and it's worth keeping them conceptually separate because they solve different problems and have different costs.

## Within-Episode Self-Critique: Generate, Reflect, Refine

The simpler and more commonly deployed pattern is a bounded loop where an agent produces an output, critiques it against some criteria, and revises — all before returning anything to the user or committing to an action.

```python
class ReflectionLoop:
    def __init__(self, llm, max_iterations: int = 3):
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, task: str) -> dict:
        output = self._generate(task)

        for i in range(self.max_iterations):
            critique = self._critique(task, output)
            if critique["acceptable"]:
                return {"output": output, "iterations": i, "critique": critique}
            output = self._refine(task, output, critique)

        return {"output": output, "iterations": self.max_iterations, "note": "max iterations reached"}

    def _generate(self, task: str) -> str:
        return self.llm.generate(f"Complete this task:\n{task}")

    def _critique(self, task: str, output: str) -> dict:
        prompt = f"""Task: {task}
Output: {output}

Critique this output against the task. Check specifically for:
1. Factual errors you can verify from what's stated in the task itself
2. Missed requirements from the task
3. Internal inconsistencies (does the output contradict itself?)

Respond as JSON: {{"acceptable": bool, "issues": [list of specific issues]}}
"""
        return self.llm.generate_json(prompt)

    def _refine(self, task: str, output: str, critique: dict) -> str:
        issues = "\n".join(f"- {i}" for i in critique["issues"])
        prompt = f"""Task: {task}
Previous output: {output}
Issues identified: {issues}

Produce a revised output that fixes these specific issues.
"""
        return self.llm.generate(prompt)
```

This loop is useful and cheap relative to full Reflexion, but it has a specific, well-documented weakness that the next section addresses head-on: the critique step in `_critique` is being performed by the *same model*, with the *same knowledge and the same blind spots*, that produced the output in `_generate`. If the model didn't know something was wrong when it generated the output, there's no strong reason to expect it will suddenly recognize the error when asked to critique it a moment later, especially for confident, coherent-sounding falsehoods rather than sloppy ones.

## Full Reflexion: Episodic Memory Across Attempts

The full Reflexion architecture assumes the agent has some way of knowing whether a task attempt succeeded or failed — this is the part that most tutorials gloss over, and it matters enormously (more on this below). Given that signal, the loop is: attempt the task, receive an outcome signal, generate a verbal reflection on *why* that outcome occurred, store the reflection in an episodic memory keyed by task similarity, and retrieve relevant past reflections before the next attempt.

```python
class ReflexionAgent:
    def __init__(self, llm, executor, memory, max_attempts: int = 3):
        self.llm = llm
        self.executor = executor
        self.memory = memory              # episodic memory store, keyed by task embedding
        self.max_attempts = max_attempts

    def run(self, task: str) -> dict:
        past_reflections = self.memory.retrieve_similar(task, k=3)

        for attempt in range(self.max_attempts):
            strategy = self._build_strategy(task, past_reflections)
            result = self.executor.execute(task, strategy=strategy)

            outcome = self._evaluate(task, result)   # external signal — see below
            if outcome["success"]:
                return {"result": result, "attempts": attempt + 1}

            reflection = self._reflect(task, strategy, result, outcome)
            self.memory.store(task, reflection)
            past_reflections.append(reflection)

        return {"result": result, "attempts": self.max_attempts, "note": "did not succeed"}

    def _build_strategy(self, task: str, reflections: list[str]) -> str:
        if not reflections:
            return "No prior attempts. Use your best judgment."
        history = "\n".join(f"- {r}" for r in reflections)
        prompt = f"""Task: {task}

Reflections from previous failed attempts on similar tasks:
{history}

Based on these, propose a strategy for this attempt that avoids the
mistakes described above. Be specific about what to do differently.
"""
        return self.llm.generate(prompt)

    def _evaluate(self, task: str, result) -> dict:
        # This MUST come from something other than "ask the model if it's happy."
        # See the section on external signals below.
        return self.executor.check_against_ground_truth(task, result)

    def _reflect(self, task, strategy, result, outcome) -> str:
        prompt = f"""Task: {task}
Strategy attempted: {strategy}
Result: {result}
Outcome: FAILED — {outcome['reason']}

Write a concise reflection (2-3 sentences) explaining what likely caused
this failure and what should be done differently next time. Be specific
enough that this reflection would actually change behavior on a retry.
"""
        return self.llm.generate(prompt)
```

The `memory.retrieve_similar` call is doing something worth naming explicitly: reflections are stored and retrieved by *task similarity*, typically via embedding search over a description of the task, not simply appended to a linear log. This means a reflection generated while debugging a database connection issue can surface again months later on an unrelated task that happens to hit a similar failure mode, which is the entire point of treating this as a memory system rather than a per-conversation scratchpad. It is worth connecting this pattern to the broader memory architectures covered elsewhere in this series — Reflexion's episodic store is a specific, narrow application of long-term agent memory, scoped to "past mistakes and what was learned from them" rather than general facts or conversation history.

## The Central Practical Limit: Models Are Bad at Judging Themselves

The uncomfortable finding, replicated across a fair amount of subsequent research since the original Reflexion paper, is that **self-critique without an external signal is a weak corrective mechanism**, and can even actively hurt in some settings. There are a few distinct reasons this happens, and it's worth separating them because they call for different fixes.

**Shared blind spots.** If a model's training and in-context reasoning lead it to a wrong belief, asking it to check that belief invokes the same reasoning process that produced the error in the first place. A model that doesn't know a particular API was deprecated last month will confidently generate code using it, and will just as confidently approve that code when asked to "critique this for errors" — the critique pass has no more access to the missing fact than the generation pass did.

**Confident, fluent wrongness is hard to distinguish from confident, fluent correctness.** LLM self-evaluation tends to correlate more with the *stylistic* qualities of an output (is it well-organized, does it sound authoritative, is it internally consistent) than with its actual correctness against ground truth. A wrong answer stated fluently and consistently often scores as "acceptable" under self-critique, while a correct but hedged or awkwardly-phrased answer can get flagged as needing revision — the critique step is measuring something correlated with, but distinct from, the thing that actually matters.

**Sycophantic drift under repeated self-editing.** In multi-round self-refinement without an external anchor, models can drift — each revision optimizes against the model's own prior critique rather than against the actual task requirements, and small biases in what the critique step tends to flag compound over iterations. Some studies of iterative self-refinement have found outputs getting *more* confident and more verbose without becoming more correct, because there is no external ground truth pulling the process back toward accuracy.

**No genuine outcome signal in many tasks.** The `_evaluate` method in the code above is a placeholder for something that is often the hardest part of the entire system to build honestly: for a creative writing task, "did this succeed" has no crisp ground truth at all. For a coding task, it might be "did the unit tests pass" — a real external signal. For a customer support resolution, it might be "did the customer re-open the ticket within 48 hours" — an external, if delayed and noisy, signal. The quality of a Reflexion system is bounded by the quality of this evaluation signal, and a system that fakes this step by asking the same model "did you succeed?" is not doing Reflexion in the sense the original paper intended — it has quietly degraded into the weaker within-episode self-critique pattern from the first section, minus even that pattern's benefit of at least seeing the current output while critiquing it.

## What Actually Works: External Verification as the Anchor

The practical takeaway that production systems have converged on is that self-correction is genuinely valuable, but only when it is anchored to a signal that does not originate from the same generative process being corrected. Concretely, this means preferring:

- **Executable verification** over self-judgment wherever the task allows it: run the generated code and check whether tests pass or whether it raises an exception, rather than asking the model whether the code looks correct. This is why code-generation agents that incorporate a test-execution step in their loop reliably outperform ones that rely on the model reviewing its own code in prose.
- **Structured, rule-based checks** for well-defined constraints: does the output satisfy a schema, does a generated SQL query parse and reference columns that actually exist in the schema, does a citation actually appear in the source document it claims to be quoting. These are cheap, deterministic, and immune to the model's own blind spots by construction.
- **A different, specialized model or a retrieval step** as the critic, rather than the same model in a different role — an approach sometimes called a "generator-critic" split, where the critic is either a separately fine-tuned model, a smaller model prompted very narrowly for verification only, or is grounded by retrieval against an authoritative source rather than reasoning from parametric memory alone.
- **Human feedback** at the highest-stakes decision points, treated as the outcome signal that populates the episodic memory — this is where Reflexion-style memory and human-in-the-loop workflows (covered in the graph-based agents chapter) meet in practice: a human's correction of an agent's mistake is exactly the kind of high-quality "reflection" that's worth banking for future retrieval.

None of this means self-critique prompting is useless — it does catch a real, if limited, category of errors: sloppy omissions, format violations, and inconsistencies that are genuinely visible on a careful re-read even without new information. The point is narrower and more useful for engineering purposes: budget for self-critique as a cheap first pass that catches "did I forget something obviously required by the prompt," and budget separately, with a real external verification mechanism, for "is this actually correct" — because those are different questions, and only one of them can be reliably answered by the same model marking its own homework.

## A Worked Example: Coding Agent With a Real Outcome Signal

The clearest illustration of the difference between anchored and unanchored reflection is a coding task, because it's one of the few domains where a genuinely external, cheap, and objective outcome signal is trivially available: does the code run, and do the tests pass.

```python
class CodingReflexionAgent:
    def __init__(self, llm, sandbox, memory, max_attempts: int = 3):
        self.llm = llm
        self.sandbox = sandbox      # executes code in an isolated environment
        self.memory = memory
        self.max_attempts = max_attempts

    def solve(self, spec: str, test_code: str) -> dict:
        reflections = self.memory.retrieve_similar(spec, k=3)

        for attempt in range(self.max_attempts):
            code = self._generate_code(spec, reflections)

            # EXTERNAL SIGNAL: actually run the tests, don't ask the model
            run_result = self.sandbox.run(code=code, tests=test_code)

            if run_result["all_passed"]:
                return {"code": code, "attempts": attempt + 1}

            # The reflection is grounded in a real stack trace / failing
            # assertion, not in the model's opinion of its own code.
            reflection = self._reflect(spec, code, run_result)
            self.memory.store(spec, reflection)
            reflections.append(reflection)

        return {"code": code, "attempts": self.max_attempts, "note": "tests still failing"}

    def _generate_code(self, spec: str, reflections: list[str]) -> str:
        context = "\n".join(f"- {r}" for r in reflections) or "No prior attempts."
        prompt = f"""Spec: {spec}

Lessons from previous failed attempts:
{context}

Write code that satisfies the spec, taking the above lessons into account.
"""
        return self.llm.generate_code(prompt)

    def _reflect(self, spec: str, code: str, run_result: dict) -> str:
        prompt = f"""Spec: {spec}
Code:
{code}

Test failure (actual output from running the tests, not a self-assessment):
{run_result['failure_output']}

In 2-3 sentences, explain the specific bug that caused this failure and
what should be changed to fix it. Be concrete about the line or logic at fault.
"""
        return self.llm.generate(prompt)
```

The load-bearing line is `run_result = self.sandbox.run(...)` — the reflection prompt in `_reflect` is built entirely from `run_result["failure_output"]`, which is a real stack trace or failing assertion message produced by actually executing the code, not a description the model invented about its own output. This is precisely the anchoring this chapter has been arguing for: the model is still the one producing the *reflection text* (turning a raw stack trace into an actionable lesson is a genuinely useful thing to ask an LLM to do), but the *fact of failure and its specifics* comes from outside the model entirely. Contrast this with a version where `_reflect` was instead asked "do you think this code has bugs?" with no test execution — that version would be exposed to exactly the shared-blind-spot and confident-wrongness failure modes described above, since a model that didn't realize its off-by-one error while writing the code has no strong reason to suddenly notice it when asked to look again without new information.

## Reflection Prompt Design: What Makes a Reflection Useful on Retrieval

Not every critique text is equally useful once stored and retrieved weeks later for an unrelated task. A reflection like "the code was wrong, please be more careful" is technically a reflection but carries almost no actionable signal — it will not change behavior on a future attempt because it doesn't specify *what* to do differently. A useful reflection names the specific failure mode in terms general enough to transfer to a similar-but-not-identical future task: "off-by-one error when the loop bound was derived from `len(list) - 1` instead of `len(list)`; when iterating to include the last element, double-check whether the bound should be inclusive" is retrievable and actionable in a way that a generic "be more careful" is not.

This suggests a concrete prompt-engineering guideline: explicitly instruct the reflection-generation prompt to (1) name the specific mechanism of failure, not just its symptom, (2) state the general principle that would have avoided it, phrased so it applies beyond just the exact failing case, and (3) keep it short enough that several reflections can be included in a future prompt's context without dominating it. Reflections that read like a senior engineer's terse code-review comment ("watch the off-by-one on inclusive loop bounds") transfer far better than reflections that read like a restatement of the error message.

## Multi-Agent Reflection: Separating the Generator From the Critic

A structural way to reduce the shared-blind-spot problem, short of finding a hard external signal, is to split the generator and the critic into genuinely different models or genuinely different prompting contexts rather than having one model play both roles in sequence within the same context. This "actor-critic" split for LLM agents can take a few forms in practice: using a different model family or checkpoint for the critic than for the generator (so systematic blind spots are less likely to be perfectly shared), giving the critic a narrower, specialized prompt focused purely on adversarial fault-finding rather than general quality assessment (a critic explicitly instructed to "assume this output contains at least one error and find it" behaves differently than one asked "is this good"), or grounding the critic in retrieval — giving it access to a source document, a schema, or a specification the generator did not have front-of-mind, so the critic's judgment is anchored in something the generator's output can be checked *against* rather than purely re-examined in isolation.

This is more expensive than single-model self-critique (it's effectively a second inference pass at minimum, sometimes with a second model) and it is not a substitute for a true external signal where one is available — running the tests is still better evidence than a second model's opinion. But where no executable check exists (open-ended writing, subjective quality judgments, nuanced policy compliance), an independently-prompted or independently-sourced critic is a meaningfully stronger anchor than asking the same generation context to grade itself, precisely because it breaks the tight coupling between "the reasoning that produced the error" and "the reasoning that's supposed to catch it."

## Practical Guidance for Deciding Whether to Build Reflexion at All

Given the cost of an extra loop (each retry is a full additional generation, and Reflexion's memory retrieval and storage add further overhead), it's worth checking, before building it, whether the investment is likely to pay off for a given system. Reflexion earns its cost when three conditions hold together: the task recurs in similar form often enough that lessons learned on one instance are likely to transfer to future instances (a one-off task has no "next attempt" for the reflection to improve); a genuine external outcome signal exists or can be cheaply constructed (test execution, schema validation, retrieval-grounded fact-checking, or reliable human feedback); and the cost of a failed attempt is low enough to tolerate the retries this pattern requires (a coding agent retrying against a test suite is cheap; an agent that sends a real email to a real customer on every attempt is not a good candidate for naive multi-attempt Reflexion without a staging/sandbox step). When any of these three is absent — the task is one-off, no external signal exists, or failures are expensive and irreversible — the engineering effort is usually better spent elsewhere: a single stronger generation pass, better tools, tighter upfront specifications, or routing directly to a human rather than iterating blindly against the agent's own uncertain judgment of its own output.

## Comparing the Variants at a Glance

It's easy to conflate the different flavors of self-improvement covered in this chapter, so it's worth laying them out side by side against the axis that actually matters — where the corrective signal comes from.

| Pattern | Scope | Corrective signal | Typical cost | Main risk if misapplied |
|---|---|---|---|---|
| Within-episode self-critique | Single output, before it's returned | The same model, same context | One extra LLM call | Model approves its own blind spots |
| Full Reflexion (episodic memory) | Across task attempts / instances | Requires an explicit outcome signal | Multiple attempts + memory overhead | Degrades to self-critique if the signal is faked |
| Anchored reflection (test execution, retrieval, schema) | Either scope | Genuinely external to the generator | Cost of running the check (often cheap) | None if the check is truly independent |
| Multi-agent generator-critic | Either scope | A separately prompted or sourced model | At least 2x inference cost | Weaker if critic shares the generator's blind spots |
| Human-in-the-loop correction | Highest-stakes decisions | A person | Human time/latency | Doesn't scale to high volume |

The pattern to notice across every row: the more clearly independent the corrective signal is from the process that produced the original output, the more the corrective step is actually worth its cost. This is the single idea the whole chapter has been building toward, and it's the detail worth having ready in an interview setting — not the mechanics of any one variant, but the general principle that self-correction is only as good as the independence of what's checking the work.

## Interview-Relevant Summary

If asked to explain the practical limits of self-correction, the key claims to have ready are: reflection improves output quality when it's anchored to a signal the generating process didn't have access to and couldn't fabricate on its own (tests, schemas, retrieval, humans); unanchored self-critique — the same model re-reading its own output with no new information — catches shallow, format-level issues but is a weak defense against confidently-stated factual errors, because the same reasoning that produced the error is being asked to detect it; and Reflexion's specific contribution beyond simple reflection is *persisting* the lesson from a failure into a retrievable memory, so it can influence a *different, later* task attempt rather than only the current one being refined in place.
