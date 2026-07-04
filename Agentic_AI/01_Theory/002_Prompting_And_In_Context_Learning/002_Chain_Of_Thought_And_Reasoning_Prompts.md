# Chain-of-Thought and Reasoning Prompts

## Why Writing Out Steps Actually Helps

To understand why chain-of-thought prompting works, it helps to remember what a transformer actually
does at inference time: for each output token, it runs a fixed amount of computation — one forward
pass through a fixed number of layers — conditioned on everything generated so far. If you ask a
model to jump straight to a final numeric answer for a multi-step arithmetic or logic problem, you
are asking it to compress an arbitrarily complex chain of reasoning into that single fixed-depth
forward pass, with no intermediate scratch space. For easy problems this is fine — the "reasoning"
is shallow enough to fit. For harder problems, it is genuinely too little computation, and the
model's only option is to pattern-match toward a plausible-looking answer, which frequently means
guessing wrong with full confidence.

Chain-of-thought (CoT) prompting works around this constraint by asking the model to generate its
intermediate reasoning as text before it commits to a final answer. Because each new token gets its
own fresh forward pass that conditions on all the previously generated reasoning tokens, writing out
the steps effectively gives the model more total computation for the problem — every intermediate
token is a chance to build on the last one, refine a sub-result, or catch an inconsistency, none of
which is available if the model must go straight from question to answer. This is sometimes
summarized as "CoT trades tokens for compute": you are spending inference-time tokens (and therefore
latency and cost) to buy the model more effective reasoning depth. It is not magic and it is not
"the model actually reasoning" in a human sense, but empirically, letting the model externalize
intermediate steps measurably improves accuracy on arithmetic, symbolic manipulation, multi-hop
question answering, and planning-style tasks — precisely the tasks that require chaining several
dependent inferences together.

## Few-Shot Chain-of-Thought

The original demonstration of this effect (Wei et al., 2022) used few-shot examples where each
demonstration's "answer" was not just the final result but a full worked-out reasoning trace ending
in the result. The model, conditioned on this pattern, then produces its own reasoning trace for the
new problem rather than jumping to an answer.

```python
FEW_SHOT_COT = """Q: A store had 23 apples. They sold 8 and then received a shipment of 15 more. How many apples do they have now?
A: The store started with 23 apples. After selling 8, they had 23 - 8 = 15 apples.
Then they received 15 more, so they had 15 + 15 = 30 apples.
The answer is 30.

Q: A train travels 60 miles in 1.5 hours. At the same speed, how far does it travel in 4 hours?
A: The train's speed is 60 miles / 1.5 hours = 40 miles per hour.
In 4 hours at 40 miles per hour, it travels 40 * 4 = 160 miles.
The answer is 160.

Q: {question}
A:"""

prompt = FEW_SHOT_COT.format(
    question="A bakery makes 144 cookies and packs them into boxes of 12. "
              "If 3 boxes are damaged and discarded, how many boxes are sold?"
)
```

The key design detail is that the demonstrations model *the reasoning style itself*, not just the
input-output mapping. This is why few-shot CoT examples should be written the way you actually want
the model to reason — same granularity of steps, same way of flagging intermediate quantities —
because the model will imitate the *form* of the demonstrated reasoning just as faithfully as it
imitates the final answer format.

## Zero-Shot Chain-of-Thought

A simpler and, for many tasks, nearly as effective technique requires no curated examples at all:
append a phrase like "Let's think step by step" (Kojima et al., 2022) to the prompt, and the
instruction-tuned model, having seen enormous amounts of step-by-step explanatory text during
pretraining and fine-tuning, will produce its own reasoning trace without ever being shown a
demonstration.

```python
def zero_shot_cot_prompt(question: str) -> str:
    return f"{question}\n\nLet's think step by step."

prompt = zero_shot_cot_prompt(
    "If a shirt originally costs $40 and is discounted by 25%, then an additional 10% "
    "discount is applied to the new price, what is the final price?"
)
```

Zero-shot CoT is attractive precisely because it costs nothing to try — no example bank to build or
maintain — and it is a reasonable first thing to attempt whenever a task involves more than one
logical hop. Its ceiling is generally a bit lower than a well-crafted few-shot CoT prompt on tasks
where the exact reasoning pattern is unusual or domain-specific, since the model has to infer the
right granularity of steps on its own rather than being shown it, but for general arithmetic, logic,
and everyday multi-step reasoning it captures most of the benefit few-shot CoT provides.

## Self-Consistency: Sampling Multiple Reasoning Paths

Chain-of-thought reasoning is not deterministic in its quality — at nonzero temperature, sampling
the same CoT prompt twice can produce two different reasoning traces, and one might make an
arithmetic slip that the other avoids. Self-consistency (Wang et al., 2022) exploits this by
sampling several independent reasoning paths for the same question at a moderate-to-high
temperature, extracting the final answer from each, and taking a majority vote across them,
discarding the intermediate reasoning traces themselves. The intuition is that wrong reasoning paths
tend to fail in different, uncorrelated ways, while a correct reasoning path is a stable attractor
that multiple independent samples converge back to — so the mode of the answer distribution is a
better estimator of correctness than any single sample.

```python
from collections import Counter
import re

def extract_final_answer(text: str) -> str:
    """Very simple extraction; in production, use a stricter format instruction
    or a structured-output pass to pull out the final answer reliably."""
    match = re.search(r"answer is\s*([^\.\n]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else text.strip().splitlines()[-1]

def self_consistency_solve(client, question: str, n_samples: int = 8, model: str = "gpt-4") -> dict:
    prompt = f"{question}\n\nLet's think step by step."
    answers = []
    traces = []

    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,   # nonzero temperature is essential — this only works with diversity
            max_tokens=400,
        )
        trace = response.choices[0].message.content
        traces.append(trace)
        answers.append(extract_final_answer(trace))

    vote_counts = Counter(answers)
    winning_answer, votes = vote_counts.most_common(1)[0]

    return {
        "answer": winning_answer,
        "confidence": votes / n_samples,   # fraction of samples that agreed
        "all_answers": answers,
        "traces": traces,
    }
```

Two practical points make or break a self-consistency implementation. First, **temperature must be
nonzero** — at temperature 0 every sample is identical and voting buys you nothing beyond the cost
of n redundant calls. Second, **answer extraction must be robust**; if your extraction regex or
parser fails to cleanly pull a comparable final answer out of free-form reasoning text, your "votes"
will be noise. This is a strong argument for combining self-consistency with a structured
final-answer format (e.g., "end your response with `Answer: <value>`" or a JSON-mode final field) so
that the extraction step is a simple, reliable parse rather than another source of error. The cost
model here is straightforward and important to communicate to stakeholders: self-consistency with n
samples costs roughly n times the tokens and, if run sequentially, n times the latency of a single
call, though independent samples can be parallelized to trade cost for wall-clock time. It is best
reserved for high-value, error-sensitive decisions rather than applied uniformly across all traffic.

## Least-to-Most Prompting

Chain-of-thought, whether few-shot or zero-shot, asks the model to produce its entire reasoning
trace in one continuous generation. Least-to-most prompting (Zhou et al., 2022) takes a different
structural approach: it explicitly decomposes a complex problem into an ordered sequence of simpler
subproblems, and then solves them one at a time in separate prompts, feeding the answer to each
subproblem into the prompt for the next one. This is closer to how a person might work through a
problem with a notepad, checking off sub-results and only using confirmed sub-results to attempt the
next stage, rather than trying to hold the entire dependency chain in their head in one pass.

```python
def least_to_most_solve(client, problem: str, model: str = "gpt-4") -> dict:
    # Stage 1: decompose the problem into an ordered list of subquestions
    decompose_prompt = f"""Break the following problem down into a minimal ordered list of
simpler subquestions that must be answered in sequence to solve it. Do not solve them yet,
just list the subquestions, one per line, in the order they must be tackled.

Problem: {problem}"""

    decomposition = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": decompose_prompt}],
        temperature=0,
    ).choices[0].message.content

    subquestions = [line.strip("- ").strip() for line in decomposition.splitlines() if line.strip()]

    # Stage 2: solve each subquestion in order, carrying forward prior answers as context
    solved_so_far = []
    for sub_q in subquestions:
        context = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in solved_so_far
        )
        stage_prompt = f"""{context}

Using the answers above as established facts, answer the next subquestion.
Q: {sub_q}
A:"""
        answer = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": stage_prompt}],
            temperature=0,
        ).choices[0].message.content
        solved_so_far.append((sub_q, answer))

    final_answer = solved_so_far[-1][1] if solved_so_far else None
    return {"subquestions": subquestions, "steps": solved_so_far, "final_answer": final_answer}
```

Least-to-most prompting tends to outperform plain CoT specifically on problems where the difficulty
comes from **compositional generalization** — problems structurally longer or more deeply nested
than anything in the model's few-shot examples or common training patterns, such as multi-hop
questions that chain through several facts, or instructions that require applying a rule iteratively
more times than any single demonstration showed. By explicitly separating "figure out the subgoals"
from "solve each subgoal," it prevents the model from having to simultaneously plan and execute,
which is where single-pass CoT most often breaks down on genuinely novel problem structures. The
cost is added latency and complexity: it requires at least two round trips (decomposition, then
solving) and often more, since each subquestion may be its own call, so it is best reserved for
problems where plain or self-consistent CoT has already been shown to be insufficient.

## How These Interact With Modern Reasoning Models

Everything above assumes you are prompting a standard instruction-tuned chat model that, left to its
own devices, tends to jump straight to an answer unless nudged into showing its work. Since around
2024, a distinct class of "reasoning models" (OpenAI's o-series, Claude's extended thinking mode,
DeepSeek-R1, and similar) has changed this default. These models are trained — typically with
reinforcement learning against verifiable reward signals such as unit tests or checked math answers
— to *already* generate long internal chains of reasoning before producing a final answer, without
needing to be asked. The reasoning trace is often a first-class part of the model's behavior rather
than an emergent side effect of prompting, and in many APIs it is even returned or billed as a
separate "reasoning tokens" channel from the final visible answer.

This changes the practical playbook for prompting reasoning models in a few concrete ways:

**Explicit "let's think step by step" is often redundant, and sometimes counterproductive.** The
model is already going to reason internally regardless of the surface phrasing of your prompt;
appending a CoT trigger phrase designed for non-reasoning models adds tokens without adding the
effect it was designed to produce, and in a few documented cases can even confuse formatting
expectations if the model interprets the extra instruction as a request to *also* show reasoning in
the visible answer channel when you actually wanted a terse final response.

**Few-shot CoT demonstrations are frequently unnecessary and can hurt.** Because the model already
knows how to decompose problems internally, showing it a rigid worked example can anchor it to a
shallower or more formulaic reasoning pattern than the one it would have produced natively,
effectively downgrading its own trained reasoning behavior to imitate your (likely less thorough)
example.

**Self-consistency still helps, but the calculus shifts.** Reasoning models already spend a large,
variable amount of inference-time compute internally per call, so sampling n independent completions
for a self-consistency vote multiplies an already larger per-call cost — the technique is still
valid and still improves reliability on genuinely hard, high-stakes problems, but the cost-benefit
threshold for when it's worth doing moves higher compared to a cheap non-reasoning model.

**The controllable lever becomes "how much" reasoning, not "whether."** Many reasoning-model APIs
expose an explicit effort or thinking-budget parameter rather than relying on prompt phrasing to
control depth. Prompt engineering for these models shifts from *inducing* reasoning to *scoping* it
— telling the model what to focus on, what's out of scope, and how much depth is warranted for this
particular query — and to clearly specifying the shape of the *final* answer, since the internal
reasoning is no longer the primary lever you control through prompt text.

**Least-to-most style decomposition still has a role**, but increasingly for orchestration rather
than in a single prompt: with a genuinely hard, multi-stage problem, it can still be worth
explicitly breaking it into separately verified subgoals across multiple calls to a reasoning model,
especially when intermediate results can be checked against ground truth (running code, checking a
calculation) before being trusted as input to the next stage — this hybrid of external decomposition
plus internal per-step reasoning is a common pattern in production agent pipelines.

It is worth being honest about a caveat that applies to every technique in this chapter, reasoning
models included: a chain-of-thought trace, whether elicited by prompting or produced natively by a
reasoning-trained model, is not a guaranteed faithful account of *why* the model reached its answer.
There is good evidence that models sometimes produce reasoning that looks coherent and leads to the
stated answer but does not reflect the actual computation that determined that answer — a kind of
post-hoc rationalization. This matters for interpretability and for high-stakes deployments: showing
a user or an auditor a plausible-looking chain of reasoning is not the same as proving the answer is
correct, and the practical mitigation is the same one used throughout this chapter — verify the
final answer against ground truth or an independent check whenever the cost of being wrong is high,
rather than trusting a nice-looking reasoning trace as evidence of correctness on its own.

## Practical Guidance for Choosing a Technique

For everyday tasks with a standard chat model, start with zero-shot CoT — it is free to try and
captures most of the available benefit for typical multi-step questions. Move to few-shot CoT when
the task has a domain-specific reasoning style that the model doesn't produce well on its own, such
as a particular way of citing evidence or a specific analytical framework your organization uses.
Reach for self-consistency only on high-value or error-sensitive single answers where you can afford
several times the token cost and where you have a reliable way to extract and compare final answers
across samples. Use least-to-most decomposition when you have evidence that the model is failing
specifically because a problem is structurally longer or more deeply compositional than anything in
its typical training distribution, not as a default for every multi-step task. And when you are
working with a reasoning-tuned model, default to trusting its native reasoning process, keep prompts
focused on clearly specifying the desired final output and any hard constraints, and reserve
explicit step-by-step scaffolding for cases where you have empirically observed the model's native
reasoning falling short on your specific task.
