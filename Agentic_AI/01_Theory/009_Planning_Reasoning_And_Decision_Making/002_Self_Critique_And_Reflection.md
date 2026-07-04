# Self-Critique and Reflection

## Table of Contents

1. The Appeal of "Just Ask It to Check Its Own Work"
2. Why Self-Critique Sometimes Works
3. Why Self-Critique Often Fails
4. A Basic Self-Reflection Loop
5. Reflexion: Reflection as Persistent Memory Across Trials
6. The Separate-Critic Pattern
7. Tool-Based and Programmatic Verification
8. Combining Critics, Tools, and Reflection
9. Knowing When to Stop Revising
10. Production Guidance and Interview Framing

---

## 1. The Appeal of "Just Ask It to Check Its Own Work"

Once you've built an agent that produces some output — an answer, a piece of code, a plan, a draft email — the obvious next idea is to have the model look at its own output and ask "is this actually good, and if not, fix it." This is appealing for a very simple reason: it costs one more LLM call, requires no additional infrastructure, and intuitively mirrors something humans do constantly — write a first draft, reread it, and fix the obvious problems. It also has genuine empirical support: across a range of tasks, prompting a model to critique and then revise its own output measurably improves quality compared to taking the first-pass output as final, and the pattern has been formalized under several names (self-refine, reflexion, self-critique, critique-and-revise) that all amount to the same core loop of generate, critique, revise, repeat.

But the same intuitive appeal that makes this pattern attractive also makes it easy to over-trust. The honest, and slightly uncomfortable, finding from research and from production experience is that a model's ability to critique its own output is bounded by the same knowledge and the same blind spots that produced the output in the first place. If a model doesn't know that a particular fact is wrong, asking it to check its own work for factual errors will not reliably surface that specific error, because the "checking" pass draws on the same underlying weights and the same underlying gaps as the "generating" pass. This chapter is about drawing a clear, practical line between the cases where self-critique genuinely helps and the cases where it produces an illusion of rigor — confident-sounding critique text that doesn't actually catch the errors that matter — and about the alternative patterns (external critic models, tool-based verification) that hold up better when correctness genuinely matters.

## 2. Why Self-Critique Sometimes Works

Self-critique earns its keep on a specific, identifiable class of problems: those where evaluating an answer is easier than generating it, and where the errors involved are not "I don't know this fact" errors but "I made a preventable mistake in execution" errors. This asymmetry between generation difficulty and evaluation difficulty is the same principle behind why P vs NP is interesting in computer science — verifying a proposed solution to a hard problem is often dramatically cheaper than finding that solution from scratch. When you ask a model to write a paragraph and then ask it to check whether that paragraph actually answers the question that was asked, "did I answer the question" is a much simpler judgment than "compose a good answer" was, and models are noticeably better at simple judgments than at complex generation. Similarly, when a model writes code and then rereads it looking for an off-by-one error or an unhandled edge case it named as a requirement, it's checking its output against an explicit, textually available specification, which is a bounded, mechanical-ish task rather than one requiring new knowledge.

Self-critique also works reasonably well against instructions and constraints the model can literally reread: "did I stay under the 200-word limit," "did I address all three sub-questions," "did I use the requested output format." These are checks against the prompt itself, which is right there in context, rather than checks against the model's own world knowledge, which is what it's trying to introspect on and often cannot do reliably. A second re-pass over the model's own draft, specifically framed as "does this literally satisfy these explicit constraints," reliably catches a meaningful fraction of compliance failures, because it's fundamentally a text-matching task dressed up as reasoning.

Finally, self-critique works better when the critique step is given a different vantage point than the generation step, even if it's the same underlying model. Asking the model to "criticize this" cold tends to produce shallow, generic feedback ("this could be more detailed"), whereas asking it to adopt a specific adversarial role — "you are a skeptical reviewer whose job is to find the single most likely error in this code, focusing on off-by-one errors and null handling" — narrows the search and tends to surface sharper, more actionable critique. This is not a fundamentally different mechanism from plain self-critique, but the framing measurably changes output quality because it constrains what the model is looking for rather than leaving it to free-associate about "quality" in the abstract.

## 3. Why Self-Critique Often Fails

The central limitation is that self-critique cannot detect errors the model doesn't know are errors. If a model states an incorrect fact — a wrong API signature, a wrong historical date, a wrong claim about how a library behaves — because that's genuinely what the model believes to be true, then asking the same model "check this for factual errors" will, in the typical case, have it reread the false statement, find it consistent with its own (also wrong) internal knowledge, and confidently declare it correct. This is not a prompting problem you can engineer around by asking more forcefully; it's a structural consequence of the critique pass drawing on the same weights, and therefore the same knowledge and the same errors, as the generation pass. This is the single most important thing to internalize about self-critique: it is much better at catching "I said something inconsistent with my own stated reasoning" than at catching "I said something inconsistent with reality."

A second, related failure mode is sycophantic self-agreement — a bias, well documented in RLHF-tuned models, toward agreeing with whatever framing or content is already on the table, including content the model itself just produced. Ask a model to critique an answer it just gave, and there's a real tendency to generate a critique that's superficially thorough (a few bullet points about tone or clarity) while avoiding the harder work of actually challenging the substance, because substantively disagreeing with oneself, having just committed to an answer, is a less statistically typical continuation than mild self-affirmation followed by cosmetic suggestions. This effect is model- and prompt-dependent and improves somewhat with more capable models and more adversarial framing, but it does not disappear, and it's a big part of why measured gains from naive "critique yourself" loops are often smaller in practice than the intuitive appeal of the technique would suggest.

A third failure mode is specific to iterative revision loops: without a hard stopping criterion, a model asked to keep revising its own output can enter an unproductive cycle, alternating between two or three variants of similar quality, each critique finding something to "fix" even when nothing is substantively wrong, essentially generating busywork that looks like rigor. This is exacerbated when the critique prompt asks an open-ended "how could this be improved" rather than a bounded "does this meet requirements X, Y, Z" — open-ended critique prompts almost always find something to say, because there's always some stylistic tweak available, whereas bounded compliance checks can genuinely terminate with "yes, this is fine."

Concretely, this means self-critique is a poor substitute for verification wherever ground truth exists and is checkable — a test suite that either passes or fails, a calculation that a calculator can confirm, a citation that either does or doesn't appear in a retrieved document. In those cases, routing to a deterministic check (Section 7) instead of asking the model to eyeball its own work is both cheaper and more reliable. Self-critique earns its place specifically where no such ground truth is mechanically available and a "second look" is the best tool on hand — which is a real and common situation, but a narrower one than the pattern's popularity suggests.

## 4. A Basic Self-Reflection Loop

Despite its limits, a bounded self-reflection loop is worth having as a default quality pass, provided you scope its critique criteria narrowly (compliance with explicit, checkable requirements) rather than open-endedly (general "quality"), and provided you cap the number of revision rounds so it can't spin indefinitely.

```python
import json

class Self_Reflector:
    def __init__(self, llm, max_revisions: int = 2, quality_threshold: int = 8):
        self.llm = llm
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold

    def reflect(self, task: str, output: str) -> dict:
        """Bounded, checklist-style critique -- not open-ended 'is this good'."""
        response = self.llm.generate(f"""
        Task: {task}
        Output: {output}

        Check this output against the task's EXPLICIT requirements only.
        Do not invent new requirements. For each check, answer yes/no with
        a one-line reason.

        1. Does it address every part of the task as stated?
        2. Does it follow any explicit format/length constraints given?
        3. Is there an internal contradiction (the output disagrees with
           itself somewhere)?
        4. Does it make a claim that contradicts information given in the
           task/context itself (not general world knowledge)?

        Return JSON:
        {{
          "quality_score": 1-10,
          "failed_checks": ["..."],
          "needs_revision": true/false
        }}
        """)
        return json.loads(response)

    def reflect_and_revise(self, task: str, output: str) -> dict:
        current = output
        for revision in range(self.max_revisions):
            verdict = self.reflect(task, current)

            if not verdict["needs_revision"] or verdict["quality_score"] >= self.quality_threshold:
                return {
                    "final_output": current,
                    "revisions": revision,
                    "final_score": verdict["quality_score"],
                }

            current = self.llm.generate(f"""
            Task: {task}
            Current output: {current}

            Fix ONLY these specific issues, without otherwise rewriting
            the output:
            {json.dumps(verdict["failed_checks"])}
            """)

        return {
            "final_output": current,
            "revisions": self.max_revisions,
            "note": "Max revisions reached without passing all checks",
        }
```

The instruction to "fix only these specific issues, without otherwise rewriting" is not cosmetic — a common failure of naive revise loops is that each revision pass rewrites the whole output from scratch, which both wastes tokens and risks reintroducing a problem that an earlier round already fixed, since nothing enforces that fixes are cumulative. Constraining revision to be targeted, and re-running the same checklist afterward, keeps the loop converging rather than oscillating.

## 5. Reflexion: Reflection as Persistent Memory Across Trials

The pattern above reflects within a single task attempt. Reflexion (Shinn et al., 2023) extends this across attempts: when an agent fails a task outright — not just "could be polished" but "did not succeed" — it generates a natural-language reflection on what went wrong, stores that reflection in a persistent memory, and includes it as context on the next attempt at the same or a similar task. This is meaningfully different from within-attempt revision because it's designed for settings with a clear, externally-checkable success/failure signal (a coding task that either passes tests or doesn't, a game that's either won or lost, a tool-use task that either achieves the goal or doesn't), and because the reflection is retained and reused rather than discarded after one round.

The reason Reflexion is more trustworthy than open-ended self-critique is that it's anchored to an external, non-self-reported outcome. The model isn't being asked "do you think this was good" — a question vulnerable to the sycophancy and blind-spot problems from Section 3 — it's being told "this failed, an external check confirmed it," and asked to reason about why, which is a narrower and more grounded task. The model is still doing the reflecting, so it can still misdiagnose the cause of failure, but the trigger for reflection is external ground truth rather than self-assessment, which removes one whole layer of the unreliability.

```python
import json

class Reflexion_Agent:
    def __init__(self, llm, tools, external_evaluator, max_trials: int = 3):
        self.llm = llm
        self.tools = tools
        self.evaluate = external_evaluator  # must be a non-self-reported check
        self.max_trials = max_trials
        self.reflections: list[dict] = []

    def solve(self, task: str) -> dict:
        for trial in range(self.max_trials):
            reflection_context = "\n".join(
                f"Attempt {r['trial']} failed because: {r['reflection']}"
                for r in self.reflections
            )

            plan = self.llm.generate(f"""
            Task: {task}
            {"Lessons from earlier failed attempts:\n" + reflection_context if self.reflections else ""}

            Produce a plan and execute it.
            """)

            result = self._execute(plan)
            evaluation = self.evaluate(task, result)  # external, not self-reported

            if evaluation["success"]:
                return {"success": True, "result": result, "trials": trial + 1}

            reflection = self.llm.generate(f"""
            Task: {task}
            Plan attempted: {plan}
            Result: {result}
            External evaluation: {json.dumps(evaluation)}

            The external evaluator confirmed this attempt failed. Diagnose,
            in concrete and specific terms (not generic advice), what
            about THIS plan caused THIS failure, and what to try instead.
            """)

            self.reflections.append({"trial": trial + 1, "reflection": reflection})

        return {"success": False, "trials": self.max_trials, "reflections": self.reflections}

    def _execute(self, plan):
        # Tool-calling execution loop omitted for brevity.
        raise NotImplementedError
```

The `external_evaluator` argument is doing the load-bearing work in this design — it must not be "ask the same LLM if it thinks it succeeded." A weak implementation of Reflexion that uses self-reported success as the trigger inherits every problem from Section 3 while adding the extra risk of the agent convincing itself it succeeded when it didn't, and then never reflecting or retrying at all.

## 6. The Separate-Critic Pattern

If self-critique's core weakness is that the critic shares the generator's blind spots, the direct fix is to make the critic a genuinely different model, or at least a genuinely different context, from the generator. This can take a few concrete forms in practice. The cheapest version uses a different model family or size for critique than for generation — for instance, generating a draft with a strong, expensive model but critiquing it with a different vendor's model, or with a model specifically trained or prompted to be adversarial. Because different model families are trained on different data with different architectures and different RLHF processes, their errors are less correlated than two calls to the same model, so a genuine factual slip is more likely to be caught by a model that wasn't the one that made it. This isn't a guarantee — different models share plenty of common blind spots since much of their training data overlaps — but the error correlation is measurably lower than same-model self-critique.

A more structured version of the same idea is a generator-critic-arbiter setup, where a generator produces candidates, an independent critic model scores or critiques them against explicit criteria, and — if you want extra robustness — a third pass (which could even be the original generator) resolves disagreements between the critic's feedback and the generator's defense of its own output. This adds latency and cost, so it's typically reserved for higher-stakes generations: a customer-facing summary that will be published, a piece of code that will be merged without a human review gate, a plan that will trigger real-world side effects.

```python
class Critic_Model_Reviewer:
    """Generator and critic backed by intentionally different models/configs."""

    def __init__(self, generator_llm, critic_llm):
        self.generator = generator_llm
        self.critic = critic_llm

    def generate_and_review(self, task: str) -> dict:
        draft = self.generator.generate(f"Complete this task: {task}")

        critique = self.critic.generate(f"""
        You did not write the following output -- you are reviewing someone
        else's work with a skeptical, adversarial eye. Your job is to find
        real problems, not to be agreeable.

        Task: {task}
        Output to review: {draft}

        List concrete, specific issues only. If you find none after careful
        scrutiny, say so explicitly rather than inventing minor stylistic
        notes.

        Return JSON: {{"issues": ["..."], "verdict": "pass" | "needs_revision"}}
        """)

        import json
        result = json.loads(critique)

        if result["verdict"] == "needs_revision":
            draft = self.generator.generate(f"""
            Task: {task}
            Your draft: {draft}
            An independent reviewer found these issues: {json.dumps(result['issues'])}

            Produce a corrected version addressing each issue.
            """)

        return {"final_output": draft, "critic_verdict": result}
```

The instruction telling the critic explicitly that it "did not write" the output is a small but meaningful detail: even with a genuinely different model instance, framing the review as evaluating someone else's work rather than "your own" work reinforces the adversarial stance the pattern depends on, and measurably reduces the reflexive-agreement behavior that creeps in even across model boundaries when a critique prompt is phrased ambiguously about ownership.

## 7. Tool-Based and Programmatic Verification

The most reliable form of "critique" is not a language model judgment at all — it's a deterministic or semi-deterministic check that doesn't depend on any model's opinion. Wherever a task has a mechanically checkable property, prefer checking that property directly over asking any model, generator or critic, to eyeball it. Code that's supposed to work should be run against a test suite or at minimum a syntax/type check; a claimed calculation should be verified with an actual calculator or code execution, not by asking a model to "double check the arithmetic," since arithmetic verification is exactly the kind of task language models are unreliable at even when explicitly asked to be careful; a factual claim grounded in retrieved documents should be checked by confirming the claim's text or a close paraphrase actually appears in the cited source, which is a retrieval/matching operation, not a generation-quality judgment; a generated API call should be validated against the tool's actual schema before execution, catching malformed arguments deterministically rather than hoping the model "notices" it made a mistake.

```python
import ast
import subprocess

class Tool_Based_Verifier:
    """Deterministic checks, used wherever the task allows one, in place of
    or as a filter before any LLM-based critique."""

    def verify_python_syntax(self, code: str) -> dict:
        try:
            ast.parse(code)
            return {"valid": True}
        except SyntaxError as e:
            return {"valid": False, "error": str(e), "line": e.lineno}

    def run_tests(self, test_command: list[str]) -> dict:
        result = subprocess.run(test_command, capture_output=True, text=True, timeout=60)
        return {
            "passed": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def verify_arithmetic(self, expression: str, claimed_result: str) -> dict:
        try:
            actual = eval(expression, {"__builtins__": {}})  # sandboxed eval
        except Exception as e:
            return {"valid": False, "error": str(e)}
        matches = str(actual) == claimed_result.strip()
        return {"valid": matches, "actual": actual, "claimed": claimed_result}

    def verify_citation(self, claim: str, source_documents: list[str]) -> dict:
        # A production version would use embedding similarity or NLI,
        # not substring match -- this sketch shows the principle.
        found = any(claim.lower() in doc.lower() for doc in source_documents)
        return {"supported": found}

    def verify_tool_call_schema(self, call: dict, schema: dict) -> dict:
        missing = [
            field for field in schema.get("required", [])
            if field not in call.get("params", {})
        ]
        return {"valid": not missing, "missing_fields": missing}
```

The practical rule of thumb is to build a checklist, for any given task, of what fraction of its correctness criteria are mechanically checkable, and route only the remainder — genuinely subjective quality, tone, completeness against a fuzzy spec — to LLM-based critique. In most production agent tasks this fraction is larger than teams initially assume: schema validity, test pass/fail, retrieval grounding, and numeric correctness are all checkable, and reserving LLM critique for what's left (does this read naturally, did it miss an implicit but unstated user need) both cuts cost and, more importantly, removes exactly the failure modes described in Section 3 from the parts of the task where they'd be most damaging.

## 8. Combining Critics, Tools, and Reflection

The strongest production designs layer these techniques rather than picking one. A typical pipeline runs deterministic tool-based checks first, since they're cheapest and most reliable and can short-circuit the rest of the pipeline on a hard failure (invalid syntax, failed schema validation) without ever needing an LLM's opinion. If the deterministic checks pass, an independent critic model (or the same model under an adversarial, differently-framed prompt, as a weaker but cheaper substitute) reviews what's left — the subjective, non-mechanically-checkable dimensions. If the critic flags issues, a bounded self-revision loop (Section 4) applies targeted fixes, re-running the deterministic checks after each revision since a fix for one issue can silently break something that previously passed. Reflexion-style persistent memory (Section 5) sits above all of this at the level of task attempts, not individual revisions, capturing lessons when an entire attempt fails an external, ground-truth check, so future attempts at similar tasks start with accumulated experience rather than from scratch.

```python
class Layered_Verification_Pipeline:
    def __init__(self, generator_llm, critic_llm, verifier, max_revisions=2):
        self.generator = generator_llm
        self.critic = critic_llm
        self.verifier = verifier
        self.max_revisions = max_revisions

    def run(self, task: str, deterministic_checks: list) -> dict:
        output = self.generator.generate(f"Complete this task: {task}")

        for revision in range(self.max_revisions):
            # 1. Cheapest, most reliable layer first.
            hard_failures = [
                check(output) for check in deterministic_checks
                if not check(output).get("valid", True)
            ]
            if hard_failures:
                output = self._repair(task, output, hard_failures)
                continue  # re-run deterministic checks before trusting the critic

            # 2. Independent critic for what tools can't check.
            critique = self._critic_review(task, output)
            if critique["verdict"] == "pass":
                return {"output": output, "revisions": revision, "status": "verified"}

            output = self._repair(task, output, critique["issues"])

        return {"output": output, "revisions": self.max_revisions, "status": "max_revisions_reached"}

    def _repair(self, task, output, issues):
        return self.generator.generate(f"""
        Task: {task}
        Current output: {output}
        Fix exactly these issues, minimally: {issues}
        """)

    def _critic_review(self, task, output):
        import json
        response = self.critic.generate(f"""
        Review this output against the task. Be skeptical; you did not write it.
        Task: {task}
        Output: {output}
        Return JSON: {{"verdict": "pass"|"needs_revision", "issues": ["..."]}}
        """)
        return json.loads(response)
```

## 9. Knowing When to Stop Revising

Every revision loop needs a termination condition beyond "keep going until it's perfect," because "perfect" is not a state an LLM-based judge can reliably certify, and open-ended critique prompts will, as noted in Section 3, tend to keep finding something to flag indefinitely. There are three practical stopping conditions worth combining rather than relying on any single one. The first is a hard cap on revision rounds — a fixed maximum regardless of critique output — which bounds worst-case cost and prevents infinite loops outright. The second is a convergence check: if the current revision is nearly identical to the previous one (by a text-similarity or diff-size measure), the loop has plateaued and further rounds are unlikely to help, so stop and return the current output rather than continuing to burn calls chasing marginal or illusory improvements. The third, and most important where it's available, is switching the stopping condition from "the critic is satisfied" to "the deterministic check passes" — for any task with a mechanical pass/fail criterion (tests pass, schema validates, citation confirmed), stop the moment that criterion is met, and treat any further LLM-based critique afterward as optional polish rather than a gating condition, since gating on subjective critique that can always find something to say is precisely how these loops fail to terminate cleanly.

## 10. Production Guidance and Interview Framing

A few things are worth being able to state crisply if this topic comes up in an interview, because it's a place where surface-level familiarity ("yes, I use self-reflection loops") is easy to distinguish from real experience. First, self-critique is not free rigor; it's a specific tool with a specific failure mode — it cannot catch what the model doesn't know it doesn't know — and using it as if it were a substitute for actual verification, especially on tasks with checkable ground truth, is a known anti-pattern, not a best practice. Second, the single highest-leverage change teams make when a naive self-critique loop underperforms is not better prompting of the critique step, it's separating the critic from the generator — different model, different framing, or ideally a deterministic tool — because same-model self-critique is fighting the same blind spots twice. Third, reflection that persists across attempts (Reflexion-style) is only as trustworthy as the signal that triggers it; if that signal is itself self-reported success rather than an external check, you've built a system that can confidently loop without ever correcting course. Fourth, and this generalizes past this specific chapter, verification budget should be spent where it changes outcomes: cheap deterministic checks everywhere they're available, model-based critique reserved for genuinely subjective residual quality, and the most expensive patterns (separate critic models, multi-round Reflexion-style memory) reserved for the highest-stakes generations, mirroring the escalation-by-difficulty pattern that also governs when tree- or graph-structured reasoning (previous chapter) or search-based planning (next chapter) is worth invoking at all.
