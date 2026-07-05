# LLM-as-Judge Methodology and Biases

## 0. Why LLM-as-judge exists

Module `001` established that reference-comparison metrics (EM/F1, BLEU, ROUGE, BERTScore) and
distributional-fit metrics (perplexity) cannot evaluate the thing that actually matters for a
general-purpose instruction-following model: whether a specific response, to a specific prompt, is
correct, helpful, safe, and well-formed, when there is no fixed reference string to compare against.
Human evaluation (module `003`) can answer that question directly but is slow, expensive, and hard
to scale to the iteration speed of model development — you cannot gate every training checkpoint or
every hyperparameter sweep on a multi-day human-annotation cycle.

LLM-as-judge fills that gap: use a strong LLM, prompted to assess quality directly (instead of
comparing strings), as a scalable proxy for human judgment. It is cheap enough to run on every
checkpoint, every A/B comparison, every regression test, and every large-scale preference dataset
construction pass, while being far closer to "did a competent evaluator think this was good" than
any metric in module `001`. The entire methodology of this module is about doing that substitution
carefully: it only works to the extent the judge's answers are shown to track real human judgment
(Section 4), and it fails in specific, well-documented, systematic ways (Section 2) that must be
actively corrected for rather than assumed away.

## 1. Two judge protocols: pointwise scoring vs. pairwise comparison

### 1.1 Pointwise (absolute) scoring

The judge receives a single prompt/response pair (optionally with a rubric or reference answer) and
outputs an absolute score, typically on a small ordinal scale (1-5, 1-10) or a structured rubric
breakdown (e.g., separate sub-scores for correctness, helpfulness, and safety that are then
combined).

```python
POINTWISE_PROMPT = """You are evaluating the quality of an AI assistant's response.

[Instruction]
{instruction}

[Response]
{response}

[Rubric]
Score the response from 1 to 5 on the following criteria:
- Correctness: is the factual/technical content accurate?
- Helpfulness: does it directly address the instruction?
- Clarity: is it well-organized and easy to follow?

Provide a single integer score from 1 (very poor) to 5 (excellent), then a one-sentence
justification. Output as JSON: {{"score": <int>, "justification": "<string>"}}
"""
```

**Strengths**: scores individual outputs independently, so they can be tracked over time (this
checkpoint's average score today vs. last week), aggregated across arbitrary subsets, and used to
build absolute quality dashboards without needing a fixed comparison set. Cost scales linearly in
the number of responses to score (one judge call per response), not quadratically.

**Weaknesses**: absolute scales are where LLM judges (and, for that matter, human raters) are least
reliable. A judge's notion of what a "4" versus a "5" means is not well anchored, drifts across
contexts, and different judges (or the same judge on different days / different surrounding context)
apply different implicit thresholds — this is the well-known **calibration problem** with absolute
Likert-style scoring, discussed further from the human-rater side in module `003`. Pointwise scores
from two separate judge calls on two different responses are also not directly comparable in the way
a head-to-head comparison is, because each score is anchored only to that call's own context window,
not to a shared frame of reference.

### 1.2 Pairwise comparison

The judge receives two responses to the same prompt (from two different models, or two checkpoints
of the same model, or a model and a human/reference) and outputs which one is better, optionally
with a margin ("A is much better," "roughly tied," "B is slightly better") and a justification.

```python
PAIRWISE_PROMPT = """You are comparing two AI assistant responses to the same instruction.

[Instruction]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

Which response better satisfies the instruction? Consider correctness, helpfulness,
and adherence to any explicit constraints in the instruction. Respond with exactly one
of: "A", "B", or "Tie", followed by a one-sentence justification.
"""
```

**Strengths**: comparative judgments are empirically more reliable and reproducible than absolute
ratings, for judges and humans alike — "which of these two is better" is a much easier and
better-anchored question than "assign an absolute number to this one," because the judge has a
direct point of contrast instead of an implicit, drifting internal scale. Pairwise comparisons
compose naturally into a ranking: run enough pairwise comparisons across a model pool and fit a
**Bradley-Terry model** (the same underlying model used by chess Elo and by RLHF reward-model
training from pairwise preferences) to derive a global strength score per model from the win/loss
matrix, which is exactly how leaderboards like Chatbot Arena convert pairwise human votes into a
single ranking number.

**Weaknesses**: cost is quadratic-ish in the number of systems being compared if you want a full
round-robin (though in practice you can sample pairs rather than running all `C(n,2)` comparisons,
and Bradley-Terry fitting is fairly sample-efficient with active/adaptive pair sampling). Pairwise
judgments only tell you relative ordering, not absolute quality — two responses can both be bad, or
both be excellent, and a pairwise judge will still confidently pick a "winner," so pairwise
comparisons alone cannot answer "is this good enough to ship," only "is this better than the
alternative."

### 1.3 Which to use when

Pairwise is the default choice for **model-vs-model comparison** (did this training change help?),
which is the single most common LLM-eval question in practice, precisely because comparative
judgments are the more reliable signal. Pointwise is preferable when you need **absolute, trackable
quality over time against a fixed bar** (e.g., a production safety/quality gate that must answer "is
this response acceptable to ship," independent of what any other model did) or when you need
per-response diagnostic scores rather than a relative ranking. Many mature eval stacks run both:
pointwise rubric scoring for absolute quality gates and regression detection, pairwise comparison
for head-to-head model selection and leaderboard-style ranking.

## 2. Documented judge biases

LLM judges are LLMs, and they inherit systematic, reproducible biases that have nothing to do with
response quality. These are not edge cases — they are large enough effects to flip the outcome of a
comparison, and every serious LLM-as-judge deployment has to actively measure and mitigate them.

### 2.1 Position bias

In pairwise comparison, judges show a measurable preference for whichever response is placed in a
particular slot (commonly, though not universally across models, a preference for the *first*-shown
response), independent of content. This has been directly measured by swapping the order of the same
two responses and observing that a nontrivial fraction of judgments flip — Zheng et al. 2023
("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena") reports order-flip disagreement rates in
a range large enough (double-digit percentages for some judge models) to materially distort a
leaderboard if left uncorrected. The direction and magnitude of the bias is judge-model-specific and
even prompt-specific — it is not a fixed universal constant you can just subtract off — which is
exactly why the mitigation (Section 3.1) is *symmetric measurement*, not a fixed correction term.

### 2.2 Verbosity bias (length bias)

Judges systematically favor longer responses, holding content quality roughly fixed. This has been
measured directly: constructing pairs where one response is a padded, more verbose version of the
other with no new correct information, and observing the judge still prefers the longer one at a
rate well above chance. This is a serious contamination risk for any pipeline that uses judge
preferences as a training signal (e.g., building an RLHF/DPO preference dataset with an LLM judge,
or using judge scores as a reward): a policy optimized against a verbosity-biased judge will learn
to pad its outputs, which is exactly the well-documented "reward hacking toward length" failure mode
seen in several RLHF and judge-based RL pipelines. Verbosity bias also confounds naive
interpretation of leaderboard results — a model family that was tuned (deliberately or incidentally)
toward longer, more elaborated responses can out-rank a more concise, equally-correct model purely
on this axis.

### 2.3 Self-preference bias (self-enhancement bias)

A judge model tends to rate outputs produced by its own model family more favorably than outputs of
equal or better quality from a different family — plausibly because the judge's own stylistic and
reasoning patterns are more "familiar" and score as more fluent/coherent to itself, or because
shared training data/RLHF choices make same-family outputs align better with the judge's own
implicit preferences. This has been directly measured (again in the Zheng et al. line of work and in
several follow-ups) by comparing a judge's ranking of its own family's outputs against a
human-judged reference ranking of the same outputs, and finding a systematic gap in the judge's own
family's favor. This bias is the single strongest argument for the "different family as judge"
mitigation in Section 3.4 — it is specifically a same-family artifact, not a general judge-quality
problem, so it is directly addressable by an organizational choice about which model does the
judging.

### 2.4 Other documented biases worth naming explicitly

- **Authority/format bias**: judges can be swayed by superficial markers of authority or effort —
  confident tone, citations (even fabricated ones), structured formatting (bullet points, headers) —
  independent of underlying correctness. A confidently wrong answer with clean formatting can
  out-score a correct but plainly-formatted one.
- **Chain-of-thought / self-generated-rationale bias**: asking a judge to "think step by step"
  before scoring generally improves judge-human agreement (it forces the judge to engage with
  content rather than pattern-match superficial cues), but the judge's own rationale can also anchor
  and rationalize a conclusion it reached quickly, especially if the rationale is generated *after*
  an implicit judgment rather than genuinely driving it — this is a subtler failure mode worth being
  aware of when interpreting a judge's stated justification as if it were the actual causal reason
  for the score.
- **Sycophancy toward the prompt's apparent framing**: if the evaluation prompt or accompanying
  metadata hints at which response "should" be better (e.g., ordering conventions, labels like
  "baseline" vs. "new model"), judges can be swayed by that framing rather than judging blind — this
  is a pipeline-hygiene issue (never leak system identity or expected-outcome hints into the judge
  prompt) as much as a model-inherent bias.

## 3. Mitigation techniques

### 3.1 Randomize (and average over) position

The direct fix for position bias: for every pairwise comparison, run the judge twice, once with (A,
B) and once with (B, A), and only count the comparison as a genuine preference if the judge picks
the same underlying response both times; treat position-inconsistent judgments as a tie (or discard
them, depending on how conservative the pipeline needs to be).

```python
def debiased_pairwise(judge_fn, instruction: str, resp_a: str, resp_b: str) -> str:
    """Returns 'A', 'B', or 'Tie', robust to position bias."""
    verdict_1 = judge_fn(instruction, resp_a, resp_b)   # A shown first
    verdict_2 = judge_fn(instruction, resp_b, resp_a)   # B shown first (swapped)

    # Map verdict_2 back to the original A/B labeling
    verdict_2_remapped = {"A": "B", "B": "A", "Tie": "Tie"}[verdict_2]

    if verdict_1 == verdict_2_remapped:
        return verdict_1          # consistent regardless of position -> trust it
    return "Tie"                  # position-sensitive -> treat as inconclusive
```

This doubles judge-call cost per comparison but directly measures, rather than assumes, whether
position bias affected a given verdict — and gives you, as a byproduct, an empirical position-bias
rate for that judge/prompt combination (what fraction of comparisons flip on swap), which is itself
a useful judge-quality diagnostic to track over time.

### 3.2 Use a rubric and/or a reference answer

Providing the judge with an explicit scoring rubric (what specifically to weigh — correctness,
completeness, adherence to constraints, safety — and how to weigh them relative to each other)
measurably improves judge-human agreement over an unconstrained "which is better" prompt, because it
narrows the judge's implicit decision criteria to the ones the evaluation designer actually cares
about, rather than letting the judge fall back on superficial correlates like length or tone. Where
a ground-truth or reference answer exists (math, code, factual QA), including it in the judge prompt
and instructing the judge to check the candidate against it directly is one of the most effective
single mitigations available, because it converts an open-ended quality judgment into something
closer to the closed-answer-space checking that automatic metrics handle reliably (module `001`,
Section 1) — the judge's job becomes "did this response reach the same conclusion as the reference
and via valid reasoning," which is a much better-anchored question than "is this response good in
the abstract."

### 3.3 Ensemble multiple judge calls

Run the same comparison multiple times (with temperature > 0 for genuine sampling diversity, or
across multiple distinct judge prompts/rubrics, or across multiple distinct judge models) and
aggregate — majority vote for categorical A/B/Tie verdicts, mean or median for numeric scores. This
reduces variance from any single stochastic judge call and, when ensembling across different judge
*models* rather than just resampling one model, directly dilutes any single model's idiosyncratic
biases (a self-preference bias specific to judge model X is diluted if judge models Y and Z are also
in the ensemble and don't share it). The cost is roughly linear in ensemble size, which is usually
an acceptable trade given how cheap a single judge call is relative to a human annotation.

### 3.4 Use a different model family as judge than the model being evaluated

The direct, structural fix for self-preference bias (Section 2.3): never use model X (or a model
from the same training lineage/family as X) as the primary judge when X is one of the systems under
evaluation. In practice this means labs evaluating their own frontier models lean on either (a) a
deliberately different model family as judge, (b) a panel of multiple judge families whose biases
are unlikely to point in the same direction, or (c) human evaluation as the final arbiter
specifically for head-to-head claims against a close competitor, precisely because a same-family
judge's verdict is not independent evidence in that comparison — it's expected to be biased in a
known direction, which makes it unpersuasive as the sole basis for a competitive claim regardless of
the actual number it produces.

### 3.5 Combine mitigations, and be honest about residual risk

None of the above singly eliminates judge bias; they are risk-reduction, not risk-elimination. A
production-grade judge pipeline typically stacks several of the above (position-randomized pairwise
+ rubric-constrained + ensembled across at least one non-same-family judge) and still validates the
resulting pipeline against human judgment (Section 4) rather than trusting the stacked mitigations
to be sufficient by construction.

## 4. Validating a judge against human judgment before trusting it operationally

A judge is only useful to the extent its verdicts track what a competent human evaluator would say,
and that has to be measured empirically, not assumed from a judge's benchmark scores on unrelated
tasks. The standard validation protocol:

1. **Collect a human-judged reference set.** Sample a set of prompts and response pairs (ideally
   spanning the difficulty and topic distribution the judge will actually be used on in production —
   a judge validated only on easy cases will look more reliable than it is) and collect genuine
   human pairwise preferences or pointwise scores under the protocols described in module `003`,
   with enough annotators per item to also measure human-human agreement as a ceiling reference
   (Section 4.3 below).
2. **Run the candidate judge pipeline on the exact same items**, under the exact prompt/protocol
   intended for production use (not a simplified or idealized version — validate what you will
   actually deploy).
3. **Compute agreement between judge and human majority/aggregate verdict.** For pairwise A/B/Tie
   verdicts, this is typically reported as raw agreement rate and/or Cohen's kappa against the human
   majority vote (kappa corrects for the agreement expected by chance, which matters here because a
   3-way A/B/Tie label space has a nontrivial chance-agreement floor — see module `003`, Section 3,
   for the kappa mechanics). For pointwise numeric scores, report Pearson or Spearman correlation
   between judge score and human mean score, and consider Krippendorff's alpha if you want a single
   agreement statistic that handles both nominal and ordinal data consistently across the human and
   judge raters as if they were just more raters in the pool.
4. **Compare judge-human agreement against human-human agreement.** This is the step most pipelines
   skip and shouldn't: human annotators disagree with each other a nontrivial amount on genuinely
   subjective quality judgments (module `003` covers typical inter-annotator kappa ranges), so "the
   judge only agrees with humans 75% of the time" is uninformative in isolation — the right
   comparison is whether 75% is close to, equal to, or below the rate at which two independent human
   annotators agree with *each other* on the same items. A judge that matches human-human agreement
   levels is doing about as well as another human rater would; a judge that falls well short is not
   yet trustworthy as a human proxy, regardless of how good 75% sounds in isolation.
5. **Slice the agreement analysis, don't just report an aggregate.** Aggregate agreement can hide
   systematic failure on a subpopulation that matters — check agreement separately across task
   categories (code, creative writing, factual QA, safety-sensitive prompts), across response-length
   buckets (does agreement degrade specifically when responses are very long or very short — a
   direct probe for residual verbosity bias even after mitigation), and across cases where the two
   responses are close in quality vs. clearly different (judges, like humans, are least reliable on
   close calls, and an aggregate number computed mostly over easy, clearly-differentiated pairs will
   overstate reliability on the harder, closer comparisons that actually determine most real
   model-selection decisions).
6. **Re-validate periodically and after any judge-model change.** A judge's bias profile is a
   property of the specific judge model, prompt, and protocol version in use; upgrading the judge
   model, changing its prompt template, or changing the population of systems it evaluates (e.g.,
   adding a new model family into the comparison pool) can shift its bias profile, so validation is
   not a one-time gate but a recurring check, particularly whenever the judge pipeline's role
   expands (e.g., moving from "monitoring dashboard" to "gating a release" or "generating an RL
   training signal" raises the cost of an undetected bias and should raise the validation bar
   accordingly).

The operating principle underlying all of this: an LLM judge is a fast, cheap, *approximate*
human-preference model, and every claim it produces should carry an explicit, measured confidence
derived from step 3-4 above, not an implicit assumption of correctness because "it's an LLM and LLMs
are smart." Treat a judge exactly the way you would treat any other learned proxy model deployed as
a substitute for an expensive ground-truth label: validate it against the ground truth it's a proxy
for, on the actual distribution you'll use it on, before wiring it into anything consequential (a
leaderboard claim, a training reward, a release gate).

## Cross-references

- Human evaluation protocols, inter-annotator agreement mechanics (Cohen's/Fleiss' kappa), and the
  cost/scale trade-offs that motivate LLM-as-judge in the first place are covered in
  `003_Human_Evaluation_And_Preference_Collection.md`.
- Automatic metrics and precisely why they cannot substitute for judge-style evaluation are covered
  in `001_Automatic_Metrics_And_Their_Limits.md`.
- Judge-based scoring of multi-step agent trajectories (a materially harder judging problem than
  single-turn response comparison) is covered in `005_Agentic_And_Trajectory_Evaluation.md`.
- Statistical treatment of judge-derived win rates and scores (confidence intervals on a
  Bradley-Terry fit, significance testing between two models' judge-scored win rates) is covered in
  `007_Statistical_Rigor_In_LLM_Evaluation.md`.

