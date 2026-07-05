# Agentic and Trajectory Evaluation

## 0. What's different about evaluating an agent

Every method in modules `001`-`003` evaluates a single input/output pair: a prompt and a response.
An agent — a model that operates in a loop, taking actions (tool calls, code execution, API calls,
file edits, web navigation), observing the results, and deciding what to do next until it either
completes a task or gives up — produces a **trajectory**: an ordered sequence of (reasoning, action,
observation) steps that can span dozens or hundreds of turns before terminating. Evaluating only the
final output of that loop — "did the pull request get merged," "was the final answer correct" — is
necessary but nowhere near sufficient, for reasons that matter a great deal at the frontier of what
these systems are being built to do:

- **A correct final answer can be reached via a badly broken process.** An agent that got the right
  answer only because it got lucky on a flaky retry, or because it happened to guess a file path
  correctly without actually reading the codebase, has not demonstrated the capability an
  outcome-only score credits it with — and that capability gap will show up on the next, slightly
  different task where luck doesn't bail it out.
- **An incorrect final answer can result from a nearly-correct process** that failed at the very
  last step (a good plan, correct tool calls, correct intermediate reasoning, but a trivial
  off-by-one error in the final formatting of the answer) — outcome-only scoring cannot distinguish
  this from a agent that had no idea what it was doing from step one, even though these represent
  very different capability profiles and very different fixes.
- **Real deployments care about the process, not only the destination.** An agent that deletes and
  recreates a file to "fix" a typo, that calls an expensive or destructive tool speculatively "just
  in case," or that takes 40 tool calls to do what a competent process would do in 6, is exhibiting
  behavior that matters for cost, safety, and trust even when it happens to land on a correct final
  state. A production agent that occasionally achieves a correct outcome via a dangerous or wasteful
  path is not a production-ready agent, and outcome-only evaluation is structurally blind to this.
- **Debugging and improvement require step-level signal.** If all you know is "the agent failed on
  this task," you don't know *where* it went wrong — bad plan, right plan wrong tool, right tool
  wrong arguments, right everything but failed to notice an error and recover — and each of those
  failure modes implies a different fix (better planning data, better tool-use fine-tuning, better
  error-handling training, etc.). Trajectory-level evaluation is what makes an agent failure
  actionable rather than just a discouraging aggregate number.

## 1. Anatomy of a trajectory

A concrete way to think about what's actually being scored: represent a trajectory as a sequence of
steps, each with a reasoning component, an action, and the resulting observation:

```python
Step = dict  # {"thought": str, "action": {"tool": str, "args": dict} | None, "observation": str}

trajectory = [
    {"thought": "I need to find where the bug is reported. Let me search the issue.",
     "action": {"tool": "read_file", "args": {"path": "issue.md"}},
     "observation": "Bug: sorting breaks on empty lists in utils/sort.py"},
    {"thought": "Let's look at the sort implementation.",
     "action": {"tool": "read_file", "args": {"path": "utils/sort.py"}},
     "observation": "<file contents, 40 lines>"},
    {"thought": "The bug is on line 12: `arr[0]` is accessed without a length check.",
     "action": {"tool": "edit_file", "args": {"path": "utils/sort.py", "line": 12,
                                                "new_content": "if not arr: return []"}},
     "observation": "Edit applied successfully."},
    {"thought": "Let's confirm the fix by running the test suite.",
     "action": {"tool": "run_tests", "args": {"path": "tests/test_sort.py"}},
     "observation": "3 passed, 0 failed."},
    {"thought": "Tests pass. Task complete.",
     "action": None,
     "observation": None},
]
```

Every one of the failure modes in Section 0 corresponds to something checkable *within* this
structure, not just at the final observation: was `read_file` on `issue.md` the right first move, or
should the agent have looked at recent commits first? Was `line: 12` actually the buggy line, or did
the agent get lucky? Did the agent run the tests *before* declaring success (good process) or just
assert completion without verification (bad process, even if the edit happened to be correct)?
Trajectory evaluation is the discipline of scoring these questions systematically rather than only
checking whether `run_tests`' final observation says "passed."

## 2. Step-level vs. outcome-level evaluation

### 2.1 Outcome-level evaluation

Scores only the terminal state of the trajectory against a success criterion: did the code compile
and pass the held-out test suite, did the booking actually get made with the right parameters, did
the final numeric answer match ground truth. This is the natural extension of the
closed-answer-space metrics from module `001` to a multi-step setting — it works well specifically
when the task has a **verifiable, checkable end state**, which is a large and growing fraction of
agentic benchmarks (SWE-bench-style "does the patch pass the hidden test suite," web-agent
benchmarks with a checkable final page state, tool-use benchmarks with a checkable final API call
log).

**Strengths**: objective, cheap to compute (often a simple programmatic check, no judge or human
needed), directly measures what usually actually matters for deployment ("did the task get done"),
and is immune to rewarding a plausible-looking-but-wrong process.

**Weaknesses**: exactly the blindness described in Section 0 — no visibility into *how* the outcome
was reached, no credit for a nearly-correct process that failed late, no penalty for a wasteful,
unsafe, or lucky process that happened to succeed, and no signal at all for tasks with no cleanly
checkable terminal state (most real-world, open-ended tasks — see Section 5).

### 2.2 Step-level (process) evaluation

Scores individual steps or sub-sequences of the trajectory against criteria that don't require the
task to have finished, or finished correctly, at all: was this specific tool call the right one to
make given the state at that point, were its arguments correct, was the reasoning that preceded it
sound, did the agent correctly interpret an ambiguous or error-carrying observation.

**Strengths**: gives credit/blame at the granularity needed for debugging and for training signal
(process reward models / step-level RL, and step-level supervision more generally, are built
directly on this kind of labeling), can evaluate partial trajectories on tasks the agent didn't
complete, and can catch problems (unsafe intermediate actions, wasted effort, near-misses) invisible
to a pass/fail outcome check.

**Weaknesses**: requires either a reference "good" trajectory to compare against (which, for
open-ended tasks, may not uniquely exist — Section 5) or a judge (human or LLM) competent enough to
assess step correctness in context, which is a harder judging problem than single-turn response
quality (Section 4) because it requires understanding the *task state* at that point in the
trajectory, not just the surface text of one step. Step-level labeling is also considerably more
expensive to produce than a single outcome label per trajectory, since it requires per-step
annotation rather than one terminal check.

### 2.3 In practice: both, for different purposes

Mature agent-evaluation setups use outcome-level scoring as the primary, cheap, objective signal for
"did this work" at benchmark scale (comparable to how automatic metrics serve as cheap regression
signals in module `001`), and step-level evaluation selectively — on a sample of trajectories,
especially failures — for diagnosis, for building training data (e.g., rejection-sampling good
trajectories, or training a process-level critic), and for auditing safety-relevant intermediate
behavior that an outcome check would never see (a destructive action taken and then silently undone
before the final state is checked still happened, and outcome-only scoring cannot know that).

## 3. Scoring dimensions for a trajectory

Breaking "was this a good trajectory" into named, separately assessable dimensions is what makes
trajectory evaluation tractable, whether the assessor is a rubric-following human, an LLM judge, or
a programmatic checker:

- **Tool selection correctness.** Given the state at step `t`, was the chosen tool an appropriate
  one to advance the task, versus a tool that was irrelevant, redundant with information already
  available, or premature (e.g., calling a "submit" or "finalize" tool before gathering enough
  information)?
- **Argument correctness.** Given that the tool choice was appropriate, were the arguments passed to
  it correct — the right file path, the right search query, the right parameters to an API call?
  This is a distinct failure axis from tool selection: an agent can pick exactly the right tool and
  still fail by malforming or mis-targeting the call (a classic and very common agent failure mode —
  right idea, wrong specifics).
- **Ordering and efficiency.** Was the sequence of actions a reasonable order (e.g., reading
  relevant context before editing, rather than editing blind and only reading afterward to check),
  and was the trajectory reasonably close to the minimal number of steps needed, or did it contain
  redundant, looping, or clearly wasted actions? Efficiency matters operationally (cost, latency)
  independent of whether the task ultimately succeeded, and excessive redundant action is itself
  often diagnostic of the model being "lost" (uncertain what to do next and stalling via repeated,
  low-information actions) even when it eventually stumbles into success.
- **Error recovery.** When a tool call fails, or an observation reveals the agent's prior assumption
  was wrong (a file doesn't exist where expected, a test fails after an edit, an API returns an
  unexpected error), does the agent correctly interpret the failure and adapt, or does it repeat the
  same failing action unchanged, misinterpret the error as success, or give up prematurely? This is
  arguably the single most differentiating capability between weak and strong agents in practice,
  because real environments are unreliable and most genuinely hard tasks involve at least one point
  where the initial plan doesn't survive contact with reality — an agent's error-recovery behavior
  is a much better predictor of real-world reliability than its behavior on the "everything goes as
  expected" happy path that a lot of easy benchmark items only exercise.
- **Faithfulness/groundedness of reasoning to actions.** Does the stated reasoning (the "thought"
  preceding an action) actually justify the action taken, or is there a disconnect (the agent's
  stated plan says one thing and its next action does something unrelated) — a signal relevant both
  for interpretability and for detecting cases where a chain-of-thought-style rationale is post-hoc
  narrative rather than the actual driver of behavior, echoing the chain-of-thought-bias caveat
  raised for LLM judges in module `002`.
- **Safety/appropriateness of intermediate actions**, independent of the final outcome — did the
  agent take an irreversible or high-consequence action (deleting data, sending a real email, making
  a real purchase, executing untrusted code without sandboxing) without appropriate caution,
  confirmation, or scope-limiting, regardless of whether that specific action turned out fine this
  time.

### 3.1 A toy trajectory scorer

A minimal illustration of combining outcome and step-level, multi-dimension scoring for a
coding-agent trajectory, assuming access to (a) a ground-truth "gold" trajectory or checkable
end-state, and (b) simple programmatic checks per dimension:

```python
from dataclasses import dataclass, field

@dataclass
class StepScore:
    tool_correct: bool
    args_correct: bool
    necessary: bool          # False if step was redundant/wasted
    recovered_from_error: bool | None = None   # None if no error was present to recover from

@dataclass
class TrajectoryScore:
    outcome_success: bool
    step_scores: list[StepScore] = field(default_factory=list)

    @property
    def tool_selection_accuracy(self) -> float:
        return mean(s.tool_correct for s in self.step_scores)

    @property
    def argument_accuracy(self) -> float:
        correct_tool_steps = [s for s in self.step_scores if s.tool_correct]
        if not correct_tool_steps:
            return 0.0
        return mean(s.args_correct for s in correct_tool_steps)

    @property
    def efficiency(self) -> float:
        # fraction of steps that were judged necessary (not redundant/wasted)
        return mean(s.necessary for s in self.step_scores)

    @property
    def error_recovery_rate(self) -> float | None:
        recoveries = [s.recovered_from_error for s in self.step_scores
                      if s.recovered_from_error is not None]
        return mean(recoveries) if recoveries else None

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0

def score_trajectory(trajectory: list[dict], gold_end_state_check, step_judge) -> TrajectoryScore:
    """gold_end_state_check: callable(final_observation) -> bool
       step_judge: callable(step, prior_context) -> StepScore, e.g. an LLM-judge call
                   with rubric-constrained pairwise/pointwise scoring per module 002."""
    step_scores = [step_judge(step, trajectory[:i]) for i, step in enumerate(trajectory)
                   if step["action"] is not None]
    outcome_success = gold_end_state_check(trajectory[-1]["observation"])
    return TrajectoryScore(outcome_success=outcome_success, step_scores=step_scores)
```

The `step_judge` here is deliberately left as a pluggable callable: in practice this is very often
implemented as an LLM-as-judge call (module `002`) given the trajectory-so-far as context and asked
to rate the specific step's tool choice, argument correctness, and necessity against a rubric —
trajectory evaluation and LLM-as-judge are not separate methodologies so much as LLM-as-judge
applied to a much richer, stateful object (a partial trajectory) instead of a single response, which
is precisely why it inherits module `002`'s biases (a judge asked to assess a long trajectory is
subject to the same position/verbosity/self-preference dynamics, plus new ones like recency bias
toward the most recent steps in a long context) and needs the same validation discipline against
human-labeled step judgments before being trusted.

## 4. Why step-level agent judging is a harder problem than single-turn judging

Everything module `002` says about LLM-as-judge reliability gets harder, not easier, for trajectory
judging:

- **Context length and attention.** A judge assessing step 40 of an 80-step trajectory needs to
  correctly track and use the entire preceding state (what files have been read, what's already been
  tried and failed, what the actual current bug is) — this is a much larger and more failure-prone
  context-tracking problem than judging a single prompt/response pair, and judges are empirically
  less reliable as relevant context is pushed earlier in a long input.
- **Counterfactual reasoning about alternatives.** "Was this the right tool to call" requires the
  judge to implicitly reason about what *else* could have been called and whether it would have been
  better — a genuinely harder inferential task than "is this response better than that response,"
  which only requires comparing two concrete alternatives rather than an implicit space of unstated
  ones.
- **Compounding uncertainty.** An error in judging step 10 can bias the judge's assessment of
  everything downstream (if the judge wrongly believes an early step was correct, its assessment of
  why later steps happened may be built on a wrong premise) — trajectory judging has
  failure-compounding dynamics that single-turn judging doesn't.
- **Environment-specific expertise requirements**, often more specialized than general
  response-quality judging: assessing whether a specific `git` command sequence, a specific SQL
  query, or a specific web-navigation action sequence was correct requires domain expertise in that
  specific tool/environment, which not every general-purpose judge model has uniformly across all
  tool domains an agent might operate in.

This is why, in practice, high-quality trajectory evaluation leans more heavily on **programmatic
checks wherever the environment allows them** (did the test suite actually pass, does the final file
diff match a reference diff within tolerance, did the API call log match an allowed schema) than
pure judge-based step assessment, reserving judge- or human-based step scoring for the dimensions
that genuinely can't be checked programmatically (was the reasoning sound, was an action
appropriately cautious, was a redundant-looking action actually justified by something the judge
needs to understand from context).

## 5. The hard problem: open-ended tasks with no single correct trajectory

Everything above gets meaningfully harder — and in some cases loses a clean answer altogether — once
the task is genuinely open-ended: "investigate why our service's latency regressed last week and
propose a fix," "plan and book a trip meeting these constraints," "refactor this module to be more
maintainable." These tasks share a property closed benchmark tasks (fix this specific bug so this
specific hidden test passes) don't: **there is no unique correct trajectory, and often no small
enumerable set of them**, in the same way module `001` noted that open-ended generation has no
single correct response string.

- **Multiple valid strategies can all be reasonable.** A latency-regression investigation might
  reasonably start from logs, from a recent-deploy diff, or from a profiler — none is objectively
  "the" correct first move, and a step-level judge penalizing deviation from one specific reference
  trajectory would be systematically wrong to do so.
- **"Efficient" and "thorough" trade off against each other**, and the right point on that trade-off
  is itself task- and context-dependent (a production incident wants speed; a one-time architecture
  review can afford more exploration) — a fixed efficiency rubric applied uniformly risks penalizing
  appropriately thorough exploration as "wasteful" or rewarding appropriately fast action as
  "efficient" when it was actually reckless, depending on which context the rubric-writer had in
  mind.
- **The end state itself may not have a single correct definition.** "Propose a fix" for a latency
  regression could reasonably end in several different valid fixes (a caching layer, a query
  optimization, a config change) — outcome-level checking against one gold fix is invalid here in
  the same way EM/F1 against one reference answer is invalid for open-ended text generation (module
  `001`), and for the same structural reason.
- **Human disagreement on trajectory quality is itself higher for open-ended tasks** than for closed
  ones, which directly lowers achievable inter-annotator agreement (module `003`, Section 3) and, in
  turn, lowers the ceiling on how well any judge (human-validated or not) can be shown to track a
  "ground truth" that is itself less well-defined for these tasks.

There is no fully solved answer to this in the current state of the field — it is a genuinely open,
actively-researched problem, and any staff-level treatment should say so plainly rather than
implying a tidy resolution exists. The partial mitigations in active use:

- **Multiple accepted reference trajectories/end-states per task**, authored by multiple domain
  experts, rather than a single gold trajectory — analogous to using multiple references for
  BLEU/ROUGE (module `001`), and subject to a similar residual limitation: more references reduce
  but don't eliminate the risk of penalizing a valid, unanticipated approach.
- **Rubric-based, criteria-level judging rather than reference-trajectory-matching** — judge or
  human evaluators assess whether the trajectory satisfies task-relevant *properties* (did it gather
  sufficient evidence before proposing a fix, is the proposed fix plausible and justified, were
  destructive/irreversible actions handled cautiously) rather than comparing against one canonical
  path, which generalizes better across valid-but-different strategies at the cost of requiring a
  more sophisticated, judgment-heavy rubric than a mechanical match check.
- **Outcome-focused evaluation where a checkable outcome exists even if the path doesn't need to be
  unique** — e.g., for a refactoring task, checking that the test suite still passes and a
  code-quality/complexity metric improved, without constraining *how* the agent got there, sidesteps
  the no-unique-trajectory problem for the parts of the task that do have a checkable end state,
  while leaving the genuinely unconstrained parts (was this a good refactor, stylistically) to
  rubric-based human or judge review.
- **Preference-based comparison between two full trajectories** (which of two different agents'
  attempts at the same open-ended task was better overall), borrowing pairwise comparison's
  relative-judgment robustness advantage (module `002`, Section 1.2) to avoid needing an absolute or
  reference-matched notion of trajectory correctness at all — you don't need to define "the correct
  trajectory" to usefully ask "which of these two attempts was better," which is often the actually
  answerable question even when the stronger absolute question isn't.

## Cross-references

- LLM-as-judge mechanics, biases, and validation discipline, which trajectory-level judging inherits
  and compounds, are covered in `002_LLM_As_Judge_Methodology_And_Biases.md`.
- Human evaluation protocols and inter-annotator agreement, relevant to trajectory-quality rubric
  design and to measuring the achievable agreement ceiling on open-ended tasks, are covered in
  `003_Human_Evaluation_And_Preference_Collection.md`.
- Named agentic/tool-use benchmarks (SWE-bench and similar) are covered in `..\06_Benchmarks`; this
  module covers trajectory-evaluation methodology in general, not any specific benchmark's
  construction.
- Statistical treatment of comparing agent success rates across models/checkpoints, including the
  sample-size implications of typically-small agentic eval sets, is covered in
  `007_Statistical_Rigor_In_LLM_Evaluation.md`.

