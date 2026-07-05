# Evaluation Methods — Interview Questions (Part 2)

## Q1: Exact match/F1 work reasonably well for SQuAD-style extractive QA but fail for open-domain, free-form QA. Why specifically, and what property of the task is doing the work in each case?

The property doing the work in extractive QA is that the answer is defined as a contiguous substring
of a given passage, which makes the space of valid answers small, low-cardinality, and largely
enumerable by human annotators ahead of time — a handful of acceptable spans per question
essentially cover the correctness criterion. Token-level F1 (bag-of-tokens precision/recall against
the best-matching reference) is a good proxy for "did you extract the right substring" precisely
because that really is what's being asked, and the metric's bag-of-tokens forgiveness handles minor
span-boundary variation (e.g., "Obama" vs. "Barack Obama") without needing to handle full semantic
equivalence.

Open-domain, free-form QA breaks this in two independent ways. First, the model is composing an
answer from its own knowledge rather than extracting a span from a fixed passage, so there's no
small enumerable answer space to check against — a correct answer to "why does ice float on water"
has effectively unlimited valid phrasings, most of which won't share tokens with any hand-written
reference. Second, and more subtly, free-form correctness often depends on content the reference
didn't happen to include or phrase the same way — a correct answer that adds accurate context, or
reaches the same conclusion via different but valid reasoning, gets penalized on precision by
EM/F1's bag-of-tokens comparison even though nothing about it is wrong. The task moved from "select
the right substring from a constrained set" to "generate any one of an unbounded number of valid
free-text answers," and EM/F1's entire validity was contingent on the former structure — it has no
mechanism to handle the latter, which is exactly why open-domain QA evaluation has largely moved to
LLM-as-judge or human evaluation with an explicit rubric checking whether the *claims* in the answer
are correct, rather than string-matching the answer itself.

## Q2: Explain the mechanism by which optimizing for ROUGE historically pushed automatic summarization systems toward extractive (copy-heavy) behavior rather than genuine abstraction.

ROUGE's canonical variants (ROUGE-N recall, ROUGE-L's LCS-based recall) are fundamentally asking
"how much of the reference summary's content did the candidate manage to reproduce," measured via
n-gram or longest-common-subsequence overlap. A candidate that copies phrases, or even whole
sentences, directly from the source document — especially when the reference summary itself was
constructed by lightly editing or selecting from the source, which is common for many summarization
datasets — will have systematically higher token/subsequence overlap with the reference than a
candidate that expresses the same content in genuinely different words, because copied spans are, by
definition, identical strings to whatever the reference happened to also copy or closely paraphrase
from the same source.

This creates a direct optimization pressure: any system tuned (via architecture choice, decoding
strategy, or explicit RL against ROUGE as a reward) to maximize ROUGE will find that extractive or
near-extractive strategies — selecting and lightly stitching together source sentences — reliably
scores well, while genuinely abstractive rewriting that captures the same meaning in different words
is a strictly riskier strategy under this metric, since any deviation from the reference's specific
phrasing is pure downside (lower overlap) with no corresponding credit for the rewrite being equally
or more informative. The mechanism is exactly analogous to BLEU's failure mode (module `001`,
Section 2.2) but manifests as a specific, well-documented behavioral bias in the systems trained
against it — historically, this is a large part of why early neural summarization systems optimized
against ROUGE tended toward safer, more extractive outputs, and why factual-consistency and
information-content metrics (checking candidate against the *source*, not just the reference)
emerged as necessary complements once genuinely abstractive summarization became a goal in its own
right.

## Q3 (coding): Implement ROUGE-L F1 (LCS-based) from scratch, including the precision/recall combination, and explain what property of ROUGE-L differs from ROUGE-N.

```python
def lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> dict:
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    lcs = lcs_length(cand_tokens, ref_tokens)

    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (1 + beta**2) * precision * recall / (recall + beta**2 * precision)

    return {"precision": precision, "recall": recall, "f1": f1}
```

The property that differs from ROUGE-N: ROUGE-N counts *contiguous* n-gram matches of a fixed order
`n` (a candidate 3-gram either matches a reference 3-gram exactly and contiguously, or it doesn't),
while the LCS underlying ROUGE-L allows matched tokens to be **non-contiguous** on both sides — it
credits in-order overlap even when other, non-matching tokens are interspersed between the matched
tokens in either sequence. This makes ROUGE-L more forgiving of word-order variation and insertions
than any fixed-order ROUGE-N (a candidate that reorders or lightly rephrases around a shared
subsequence of key content words still gets full LCS credit for that subsequence, where a strict
n-gram match would only credit tokens that stayed exactly adjacent). It's the same generalization
relationship that BLEU-family n-gram matching has to a subsequence-based alternative, and it's why
ROUGE-L is often reported as a complement to ROUGE-1/ROUGE-2 rather than a replacement — the two
variants are sensitive to different kinds of surface divergence between candidate and reference.

## Q4: How would you design an experiment to measure whether a specific LLM judge exhibits verbosity bias, distinct from just noticing that it "seems to like longer answers"?

The key design requirement is holding content quality fixed while varying length, so any observed
preference shift can only be attributed to length rather than to a genuine quality difference that
happens to correlate with length in your sample. The standard construction: take a set of prompts
with a known-good response for each, then create a padded variant of each response that adds
verbose, redundant, or filler content — restating points, adding unnecessary caveats or preambles,
elaborating without adding new correct information — while deliberately not adding any new correct
claims, not fixing any errors the original had, and not improving clarity (if anything, padding
should be constructed to be neutral-to-slightly-negative for genuine readability, so any judge
preference for it is attributable only to length). Run the judge (pairwise, position-swapped per
module `002`'s position-bias mitigation, so position bias doesn't confound the length measurement)
on original-vs-padded pairs across a reasonably large item set, and measure the padded-response win
rate. A verbosity-bias-free judge should show a win rate close to 50% (no reliable preference either
way, since the two responses are equivalent in actual informational content); a win rate reliably
above 50% in the padded response's favor, especially if it holds up with a tight confidence interval
(module `007`'s bootstrap CI machinery applied to this specific win-rate statistic) rather than
being explainable as sampling noise, is direct, quantified evidence of verbosity bias rather than an
impression.

A more diagnostic follow-up: vary the *degree* of padding (add a little vs. a lot of filler) and
check whether win rate for the longer response increases monotonically with padding amount — a
dose-response relationship between length and judge preference, controlling for content, is stronger
evidence of a genuine length-driven bias mechanism than a single length-doubling comparison, and
also gives you a rough sense of the bias's magnitude as a function of length delta, which is useful
for deciding how aggressively to correct for it (e.g., via a length-normalization term in a reward
model, or via explicit rubric instructions telling the judge to penalize unnecessary length) in a
downstream training pipeline.

## Q5: What's the strongest evidence that self-preference bias is a real, distinct phenomenon rather than "the judge model's own family is just genuinely better and the judge is correctly detecting that"?

The distinguishing evidence has to come from comparing the judge's verdict on same-family outputs
against an *independent* ground truth — specifically, human judgment on the exact same output pairs.
The methodology: take a set of response pairs where one response is from the judge's own model
family and the other is from a different family, collect genuine human preference judgments on those
same pairs (blinded to source model, following module `003`'s protocol), and separately collect the
LLM judge's verdicts on the same pairs. If the judge model's own family is simply, genuinely better,
its favorable verdicts toward its own family's outputs should be corroborated by the independent
human judgments at a similar rate. Self-preference bias is demonstrated specifically when there's a
*gap* — the judge favors its own family's outputs at a measurably higher rate than the human-judged
ground truth on the same items supports, meaning the judge's verdict is not simply tracking real
quality more accurately than the alternative, it's systematically over-crediting outputs from its
own lineage beyond what an independent assessor agrees with.

This is exactly the design used in the studies that first documented this bias (e.g., the line of
work following Zheng et al. 2023): the finding isn't "judge X always prefers family X's outputs,"
which could just mean family X is good — it's "judge X prefers family X's outputs at a rate that
exceeds what human raters, looking at the same specific outputs, agree with," which isolates the
bias as a property of the *judging process* rather than a property of genuine output quality. This
is also precisely why the practical mitigation (using a different-family judge, or a multi-family
ensemble) is targeted and justified: it's not a generic "improve the judge" fix, it's a structural
fix for a specifically identified, independently-verified same-family artifact.

## Q6 (coding): Implement Fleiss' kappa from a rating-count matrix, and explain what breaks if you naively apply Cohen's kappa formula to more than two raters instead.

```python
import numpy as np

def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """rating_matrix[i, j] = number of the n raters who assigned category j to item i.
    Shape: [N items, k categories]; each row sums to n (constant raters per item)."""
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)[0]

    # overall category marginal rates, pooled across all items and raters
    p_j = rating_matrix.sum(axis=0) / (N * n)
    P_e_bar = (p_j ** 2).sum()

    # per-item observed agreement rate, then averaged across items
    P_i = (rating_matrix * (rating_matrix - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = P_i.mean()

    return (P_bar - P_e_bar) / (1 - P_e_bar) if P_e_bar != 1 else 1.0
```

Naively applying the two-rater Cohen's kappa formula to more than two raters breaks down at the very
first step of the computation: Cohen's kappa is defined over a single pair's joint label
distribution (an item-by-item comparison of *one specific* rater against *another specific* rater),
and its chance-agreement term is computed from exactly two marginal distributions multiplied
together. With more than two raters per item, there is no single well-defined "the two raters' joint
distribution" to compute this from — you'd either have to arbitrarily pick one pair per item
(discarding the other raters' information and making the result depend on an arbitrary pairing
choice), or average pairwise kappa over all `C(n,2)` rater pairs per item (which is a defensible
alternative but is a different statistic with different properties, not actually "Cohen's kappa
extended"). Fleiss' kappa avoids this by never referring to specific individual raters at all — it
only uses the *counts* of how many raters (out of the panel size `n` for that item) chose each
category, which is exactly what lets it handle a rotating panel of interchangeable raters (a
different specific set of `n` raters per item, which is the common real-world crowd-annotation
design) rather than requiring a fixed, identity-tracked pair or full panel across every item.

## Q7 (scenario): You need to decide between pointwise and pairwise LLM-judge scoring for a production release-gating pipeline (does this checkpoint meet the bar to ship). Walk through the decision.

The core question a release gate needs answered is absolute, not relative: "does this specific
checkpoint meet our quality/safety bar," not "is this checkpoint better than some other specific
checkpoint" — which on its face argues for pointwise scoring against a fixed rubric. But I'd be
cautious about relying on pointwise scoring alone, given its known calibration weakness (an absolute
1-5 scale's meaning drifts across contexts and doesn't have a natural shared anchor the way a
head-to-head comparison does) — a gate built purely on "average pointwise score >= 4.2" is
vulnerable to that scale silently drifting (e.g., if the judge model version changes, or if the
prompt distribution shifts) without anyone noticing until scores that used to reliably signal "good"
no longer do.

My actual design would combine both: pointwise, rubric-constrained scoring (separate sub-scores for
correctness, helpfulness, safety) as the primary absolute gate, since that's the question being
asked, but I would also run a pairwise comparison of the candidate checkpoint against the current
production checkpoint (the most recent one that already passed the gate and is live) as a required
companion signal — this gives me a stable, slowly-evolving reference point rather than an anchorless
absolute scale, and lets me catch a subtle but important failure mode: a checkpoint that passes the
absolute pointwise bar but is actually a regression relative to what's currently in production
(possible if the judge's implicit scale itself drifted, or if the new checkpoint traded one
dimension for another in a way the aggregate pointwise score doesn't clearly reveal but a direct
head-to-head comparison would).

Before trusting either signal for a real go/no-go decision, I'd validate both the pointwise rubric
scorer and the pairwise comparator against human judgment on a held-out sample (module `002`,
Section 4), specifically checking agreement on cases near the gate threshold (where miscalibration
matters most for a binary ship/no-ship decision) rather than only on an aggregate sample that's
mostly easy, clearly-good-or-bad cases. And for the highest-stakes releases, I would not let the
automated gate be the sole decision-maker — human evaluation on a smaller, carefully chosen sample
as the final check before an irreversible release decision, consistent with module `003`'s point
that the most consequential decisions still warrant the ground truth, not the cheaper proxy, however
well-validated that proxy has tested historically.

## Q8: Why does safety-policy annotation (e.g., "is this response a policy violation") typically require a different annotator pool and training process than general helpfulness rating, even though both are "human evaluation"?

General helpfulness/preference rating is trying to capture something close to average, intuitive
user judgment — "would a typical person find this response good" — which is exactly why a broad,
demographically diverse crowd of general annotators is the *right* population to sample from: their
intuitive, varied judgments are the target quantity, not noise to be trained away. Safety-policy
annotation is trying to capture something different and much more specific: consistent application
of a particular, often carefully negotiated, written policy standard (what exactly counts as
disallowed content in category X, how to handle a specific ambiguous edge case the policy team has
already thought through and has a documented answer for) — here, annotator intuition is explicitly
*not* the target; the target is faithful, consistent execution of a specific external standard that
the annotator did not personally author and may not personally agree with in every edge case.

That difference drives every downstream design choice: safety annotators need explicit training on
the actual written policy (not just calibration examples of a fuzzy quality scale), need vetting
given the sensitive/disturbing content some categories involve (which also raises duty-of-care and
annotator-wellbeing considerations that general helpfulness rating usually doesn't), and need a much
heavier adjudication process for disagreement (escalation to a senior policy-trained reviewer with
authority to render the "correct per-policy" answer) because the goal is convergence on one
correct-per-policy answer, not an average of legitimately varied personal opinions the way pairwise
helpfulness preference is. Using general crowd annotators for policy-sensitive labeling risks
exactly the failure mode this distinction predicts: inconsistent application of nuanced policy edge
cases by people who were never trained on the policy and are, reasonably, applying their own
intuitive sense of "harmful" instead — which is a fine population sample for helpfulness rating and
a poor one for policy-standard adjudication.

## Q9: A private, never-published eval set sounds like a clean solution to contamination — what are its real limitations, and why can't a lab just rely on private evals exclusively?

Three separate limitations, each independent of the others. First, external comparability: a score
on a set nobody outside the organization can inspect isn't independently verifiable, so it can't
serve the function public benchmarks serve — supporting a claim the broader field or the public can
audit and trust, or enabling a fair, reproducible comparison against a competitor's model evaluated
on the same, openly available items. A lab that only ever reports private-eval numbers is asking to
be trusted on faith for every capability claim it makes, which is a real cost in a competitive and
safety-scrutinized field.

Second, item diversity and quality assurance at scale: large public benchmarks benefit from broad
community usage and scrutiny over time — ambiguous items get flagged and fixed, edge cases get
debated, coverage gaps get identified by many different users approaching the benchmark from
different angles — for free, essentially as a byproduct of wide adoption. A small, closely-held
private set gets none of that free quality assurance; its quality ceiling is bounded by whatever
internal team built and maintains it, and errors or blind spots in a private set can persist
unnoticed far longer than they would in a heavily-used public one.

Third, and most important for the contamination framing specifically: "private" is a property that
has to be actively maintained under real organizational pressure, and it degrades incrementally and
often irreversibly — a single internal leak (an employee discussing an item externally, a slide at a
talk showing "an example item," a misconfigured access control, a vendor handling the data without
adequate agreements) can compromise part of the set permanently, and the larger and more widely-used
inside the org a "private" set becomes, the more it starts to mean "known to hundreds of people"
rather than genuinely private, which is a meaningfully weaker guarantee than the label suggests.
Given all three, the right architecture is layered, not private-only: public benchmarks for
external, reproducible claims (with training-time decontamination applied and contamination risk
explicitly acknowledged as a limitation of that specific number), and private evals specifically for
the internal ground-truth tracking that doesn't need external verifiability — neither one replaces
the other's function.

## Q10: You're told to design a rotation policy for a private evaluation set. What specific parameters would you set, and what's the core tension you're balancing?

The core tension is between exposure risk (older items have had more cumulative opportunity to leak,
whether through an internal incident or gradual erosion of access control) and cost plus measurement
continuity (authoring new, well-calibrated, expert-verified items is expensive and time-consuming,
and replacing too much of the set too often makes it hard to tell whether a score change between
versions reflects real model improvement or just a difficulty change in the new item pool).

Concretely, I'd set: an explicit version identifier for every eval-set snapshot, so every reported
score is unambiguously tied to a specific version and never silently compared across versions as if
they measured the same thing; a scheduled partial-retirement cadence (e.g., replace roughly a third
of the item pool every quarter, rather than a full wholesale replacement or, at the other extreme,
only ever appending new items and never removing old ones) so that the average item's
time-in-the-wild exposure stays bounded without constantly resetting the set's difficulty
calibration all at once; deliberate overlap between consecutive versions (keep some fraction of
items unchanged across a rotation cycle) specifically to measure whether the newly rotated-in items
are calibrated to a similar difficulty as what they replaced, which lets me distinguish "the model
actually got better" from "the new item batch happened to be easier" when interpreting a score
change across a rotation boundary; and an explicit out-of-cycle trigger — any suspected or confirmed
partial disclosure of specific items (an internal leak, a vendor incident, an item surfacing
somewhere unexpected in an audit) immediately retires those specific items rather than waiting for
the next scheduled cycle, since a known-exposed item sitting in active use until the next quarterly
rotation defeats the purpose of having a rotation policy at all.

## Q11: Explain the difference between verbatim contamination and indirect/paraphrased contamination, and why the second is meaningfully harder to defend against even with a rigorous decontamination pipeline.

Verbatim contamination is when the exact eval question and/or answer text appears in a training
corpus, essentially unmodified — a benchmark's file mirrored on GitHub, a blog post quoting test
items directly, a paper appendix reproducing benchmark examples. This is mechanically detectable
with reasonably high recall by n-gram overlap search or embedding-based near-duplicate detection
against the known eval content (the Datasets module's core decontamination technique), precisely
because the defining feature of verbatim leakage is that the *string itself* is preserved closely
enough for a similarity search to find it.

Indirect or paraphrased contamination is when the underlying content — the specific fact pattern,
the solution methodology, the answer — appears in training data without matching the eval item's
exact text: a forum thread discussing how to approach a particular class of problem the benchmark
tests, a translated or reworded version of a benchmark item, a blog post explaining "the trick" to a
well-known tricky question without quoting the question verbatim. This is much harder to defend
against mechanically because there is no fixed string to search for — the leakage is at the level of
ideas and solution patterns rather than surface text, and a similarity-search-based decontamination
pipeline, however well-tuned, is fundamentally built around detecting textual overlap, which this
kind of leakage may not produce at all. The practical consequence is that even a rigorously
executed, well-recalled decontamination pass against known verbatim benchmark text provides no real
guarantee against the softer, harder-to-detect version of the same underlying problem — a model can
have "seen the trick" via a paraphrased discussion without ever having seen anything an n-gram
search would flag — which is precisely the gap that motivates treating contamination as a permanent,
only-partially-mitigable background risk (this module, Section 1) rather than a problem a good
enough decontamination pipeline fully closes.

## Q12: In trajectory evaluation, why is it important to separate "tool selection correctness" from "argument correctness" as distinct scoring dimensions, rather than just scoring each step as right/wrong overall?

Because they're different failure modes with different causes, different downstream implications,
and different fixes, and collapsing them into a single right/wrong step label destroys exactly the
diagnostic information that makes trajectory evaluation useful in the first place. Tool selection
failure means the agent misjudged *what kind of action* the current state calls for — it called a
search tool when it should have read a specific file it already knew about, or it tried to
finalize/submit before gathering enough information. This is a planning/decision-making failure, and
the fix is typically better training data or reinforcement signal around task decomposition and
state-appropriate action choice.

Argument correctness failure, by contrast, means the agent correctly identified *what kind* of
action to take but got the *specifics* wrong — the right tool, wrong file path; the right API, wrong
parameter value; the right general approach, malformed input. This is a much more common real-world
agent failure mode than outright wrong-tool selection in many domains, and it implies a different
fix entirely — better grounding/parsing of the current context into precise arguments, rather than
better high-level planning. An agent that reliably picks the right tool but frequently malforms the
arguments has a narrower, more mechanical problem than an agent that's generally lost about what to
do next, and conflating the two into one "step was wrong" label would make both look like the same
kind of failure when they require different interventions to fix. This distinction is directly
analogous to why module `001` insists on separating precision from recall rather than reporting one
blended number — the two failure directions are diagnostically different even when a single combined
score might look similar for two agents that are actually failing for entirely different reasons.

## Q13 (coding): Sketch an LLM-judge-based step scorer for a partial agent trajectory that combines a programmatic outcome check (where available) with judge-based reasoning-quality assessment (where it isn't), and explain why you'd prefer this hybrid over a pure end-to-end judge call on the whole transcript.

```python
from dataclasses import dataclass

@dataclass
class StepJudgment:
    step_index: int
    tool_appropriate: bool | None      # None if no programmatic check available
    reasoning_sound: bool | None       # from judge, when programmatic check can't answer this

def score_step(step: dict, trajectory_so_far: list[dict],
                programmatic_checker, llm_judge) -> StepJudgment:
    """programmatic_checker(step, trajectory_so_far) -> bool | None
           returns a definitive right/wrong verdict when mechanically checkable
           (e.g., 'did this git diff apply cleanly', 'did this API call match an allowed schema'),
           or None if this step's correctness isn't mechanically checkable.
       llm_judge(step, trajectory_so_far) -> bool
           rubric-constrained judgment of reasoning soundness / tool appropriateness,
           used only as a fallback where programmatic_checker returns None."""
    prog_result = programmatic_checker(step, trajectory_so_far)

    if prog_result is not None:
        tool_appropriate = prog_result
        reasoning_sound = None          # not needed; outcome is mechanically settled
    else:
        tool_appropriate = None
        reasoning_sound = llm_judge(step, trajectory_so_far)

    return StepJudgment(step_index=len(trajectory_so_far),
                         tool_appropriate=tool_appropriate,
                         reasoning_sound=reasoning_sound)

def score_full_trajectory(trajectory: list[dict], programmatic_checker, llm_judge) -> list[StepJudgment]:
    return [score_step(step, trajectory[:i], programmatic_checker, llm_judge)
            for i, step in enumerate(trajectory) if step.get("action") is not None]
```

I'd prefer this hybrid over one end-to-end judge call on the whole transcript for two compounding
reasons. First, wherever a step's correctness genuinely is mechanically checkable (did the test
suite pass after this edit, did the API call's schema validate), a programmatic check is strictly
more reliable than a judge's holistic assessment of the same fact — there's no reason to introduce
judge variance and cost into a question that has a deterministic answer. Second, restricting judge
calls to only the steps that actually need subjective/contextual reasoning assessment (was this a
sensible thing to try given the ambiguity in the current state) keeps the judge's job narrower and
better-scoped than asking it to holistically evaluate an entire long transcript at once, which runs
directly into the context-tracking and compounding-error problems described in module `005`, Section
4 — a judge given one specific step plus its preceding context to assess is a meaningfully easier
and more validated task than a judge given an 80-step transcript and asked to render one aggregate
verdict, and errors in judging individual steps don't compound into a single fragile end-to-end
judgment.

## Q14: Compare the trade-offs of internal vs. external red-teaming specifically along the axis of "what kind of blind spot each one has," not just cost/speed.

Internal red-teaming's blind spot is bounded by the collective imagination and background of the
people doing it — an internal team, however skilled, shares the organization's own cultural
assumptions, technical mental models, and (often unconsciously) some awareness of what the org
already considers "handled," which biases them toward re-testing known categories more thoroughly
than discovering genuinely unanticipated ones. There's also a subtler organizational-incentive blind
spot: internal teams operate inside the same reporting structure and release-timeline pressure as
everyone else, which can (even without any explicit pressure to under-report) shape what gets
prioritized for deep investigation versus what gets a lighter pass under time constraints.

External and third-party red-teaming's blind spot is different: it's bounded by information access
and context. External testers typically don't have access to the training data composition,
known-weakness history from prior model versions, or the specifics of already-deployed mitigations
that an internal team has by default — which means an external red-team can spend significant effort
rediscovering an issue the internal team already knows about and has a planned fix for, or can miss
a subtlety that only makes sense with internal context (e.g., testing against a mitigation that's
already scheduled to be replaced). External engagements are also inherently episodic (a scoped
engagement with a start and end date) rather than continuously integrated into the development loop,
so they structurally can't provide the fast, iterate-every-checkpoint coverage internal red-teaming
can.

The reason mature programs run both rather than picking one is that these are complementary, not
overlapping, blind spots: internal red-teaming's weakness (bounded imagination, shared assumptions)
is exactly what external diversity of background and incentive structure corrects for, and external
red-teaming's weakness (no deep internal context, episodic rather than continuous) is exactly what
internal red-teaming's standing access and iteration speed corrects for. Relying on only one leaves
a specific, predictable category of vulnerability under-covered, not just "less coverage" in a
generic sense.

## Q15: A red-team exercise finds a specific jailbreak prompt that reliably elicits a harmful response. The team patches the model so that exact prompt now gets refused. Why might this "fix" be insufficient, and what would a more robust response look like?

Patching the model to refuse the exact reported string is very likely fixing the symptom rather than
the underlying vulnerability, and this is a well-documented failure pattern in practice. A specific
successful jailbreak prompt is almost always one instance of a broader attack *category* (a
particular kind of fictional-framing setup, a particular multi-turn escalation pattern, a particular
encoding trick) — training the model to specifically refuse that literal string (whether via a
narrow rule or narrow fine-tuning example) can produce a model that passes the regression test for
that exact reported case while remaining just as vulnerable to trivial variants: paraphrasing the
same request, changing a surface detail of the fictional frame, translating it to another language,
or applying the same underlying strategy to a different harm topic. If the "fix" is validated only
by re-testing the literal reported string, it will look successful and be substantively hollow.

A more robust response starts by treating the reported instance as a signal to characterize, not a
string to patch: use the automated adversarial-generation techniques from this module
(mutation/variant search seeded from the reported case, or an attacker-LLM prompted to generate
structurally similar attacks) to map out the boundary of the underlying category — which phrasings,
topics, and framing variants also succeed, and which don't — so the training fix (new SFT/RLHF
examples, or a more general safety-behavior adjustment) targets the category the mapping revealed,
not just the one instance. After the fix, red-team the *category*, not just the original string, to
confirm the fix generalized rather than just memorized a refusal for the specific reported case. And
regardless of how confident the team is that this specific fix worked, add the finding and a
representative sample of its variants to a standing regression suite that gets re-run against every
future checkpoint — because a fix that holds today can silently erode in a later, unrelated training
pass (e.g., a subsequent capability-focused fine-tuning run inadvertently loosening a
previously-fixed safety behavior), and without persistent regression coverage that kind of
regression can go undetected until it's rediscovered the hard way, externally.

## Q16 (coding): Implement a permutation test for whether two systems' judge-derived pairwise win rates differ significantly, as an alternative to the bootstrap approach, and explain when you'd prefer one over the other.

```python
import numpy as np

def permutation_test_win_rate_diff(wins_system_a: np.ndarray, wins_system_b: np.ndarray,
                                     n_perm: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """
    wins_system_a, wins_system_b: boolean arrays, same length, per-item indicator of
        whether system A (resp. B) won its pairwise comparison on that item.
        (Assumes a shared item set where both systems' win/loss outcomes are recorded.)
    Returns (observed_diff, p_value) under the null that A and B have the same win rate,
    via label permutation.
    """
    rng = np.random.default_rng(seed)
    n = len(wins_system_a)
    observed_diff = wins_system_a.mean() - wins_system_b.mean()

    # Pool both systems' outcomes per item and randomly reassign the A/B label per item,
    # simulating the null hypothesis that which system "wins" doesn't depend on system identity.
    pooled = np.stack([wins_system_a, wins_system_b], axis=1)   # [n, 2]
    perm_diffs = np.empty(n_perm)
    for p in range(n_perm):
        swap_mask = rng.integers(0, 2, size=n).astype(bool)
        a_perm = np.where(swap_mask, pooled[:, 1], pooled[:, 0])
        b_perm = np.where(swap_mask, pooled[:, 0], pooled[:, 1])
        perm_diffs[p] = a_perm.mean() - b_perm.mean()

    p_value = (np.abs(perm_diffs) >= abs(observed_diff)).mean()
    return observed_diff, p_value
```

The permutation test directly simulates the null hypothesis (system identity doesn't matter —
swapping which outcome we call "A" and which we call "B" on each item should be indistinguishable
from the real assignment if there's truly no difference) by explicit relabeling, and its p-value has
a very direct, assumption-light interpretation: it's literally the fraction of relabelings that
produce a difference at least as extreme as what was observed. I'd prefer the permutation test over
the bootstrap specifically when the question is a clean hypothesis test ("is there a difference at
all," producing a p-value under an explicit null) and the pairing/exchangeability structure is
exactly "these two labels are exchangeable under the null" — which is a very natural fit for this
win/loss comparison setup. I'd prefer the paired bootstrap (module `007`, Section 3.3) when I
additionally want a confidence interval on the *magnitude* of the difference, not just a yes/no
significance verdict, since the bootstrap directly produces a distribution of plausible difference
magnitudes rather than only a p-value; in practice, running both isn't costly and gives
complementary information — the permutation test as a clean, assumption-light significance check,
and the bootstrap CI for effect-size reporting alongside it.

## Q17: A lab reports "our model achieves 71.2% on Benchmark Z" and a competitor reports "68.9%" for their model on the same benchmark, in two different papers. Why might this comparison be invalid even before any statistical test is applied?

Because "the same benchmark" doesn't guarantee "the same measurement" — everything upstream of the
score itself can differ silently between two papers' self-reported numbers. The prompt template used
to elicit the answer (exact instruction wording, few-shot example formatting and count,
delimiter/answer-format conventions) can shift scores by several points on its own, and there's no
guarantee both papers used the same template; each lab typically tunes its own harness, at least
implicitly, to work well for its own model. The evaluation harness's answer-extraction and scoring
logic — how a free-form response gets parsed into a multiple-choice letter or a normalized numeric
answer, how forgiving that normalization is — also varies across harnesses (the benchmark's own
reference implementation vs. a popular open-source harness vs. an in-house one) and has been shown
to produce materially different scores for the identical model checkpoint depending purely on this
choice. Decoding configuration (temperature, number of samples per item, whether either number is
secretly a best-of-N figure) may also differ and may not even be disclosed in enough detail to
check.

The methodologically valid comparison would run both models through one held-fixed harness, prompt
template, and decoding configuration, on the same item set, and only then apply a paired
significance test to the resulting scores. Quoting two papers' self-reported numbers side by side
and treating the 2.3-point gap as a real head-to-head result is comparing two different experiments
and presenting it as one — a very common pattern in competitive benchmark reporting, and one a
careful reviewer should flag before even getting to the question of whether 2.3 points clears a
reasonable noise threshold on whatever sample size was used.

## Q18: Walk through the arithmetic of how a 95% confidence-interval half-width shrinks as eval-set size grows, and use it to explain why doubling an eval set's size does not halve its uncertainty.

For a binomial-style accuracy metric near the conservative worst case `p = 0.5`, the
normal-approximation 95% CI half-width is `1.96 * sqrt(0.25/n)`, which scales as `1/sqrt(n)`, not
`1/n`. Concretely: at `n = 100`, half-width `≈ 1.96 * sqrt(0.0025) ≈ 9.8` percentage points; at `n =
400` (4x the items), half-width `≈ 1.96 * sqrt(0.000625) ≈ 4.9` points — exactly half, because
`sqrt(4) = 2`; at `n = 1600` (16x the original), half-width `≈ 2.45` points — a quarter of the
original, because `sqrt(16) = 4`. So doubling `n` from 100 to 200 only shrinks the half-width by a
factor of `sqrt(2) ≈ 1.41`, not 2 — you'd need to *quadruple* the eval set size to halve the
uncertainty, and to get from a 9.8-point half-width down to, say, a tight 1-point half-width, you'd
need roughly a 100x increase in `n` (since `9.8/1 ≈ 9.8`, and `9.8^2 ≈ 96`).

The practical implication for someone building an eval program: precision is expensive to buy at the
margin, and the returns to adding more items diminish rapidly relative to the cost of producing each
additional (especially expert-graded or human-annotated) item — a benchmark that already has a few
thousand items and needs its CI half-width cut in half again is a much larger, more expensive
undertaking than the same relative improvement would have been starting from a few hundred items.
This is also the direct mathematical reason small, expensive, high-quality expert evals (a few
hundred items, common for domains requiring specialized annotator expertise) inherently carry wide
confidence intervals that no amount of statistical sophistication after the fact can fix — the only
lever that actually tightens the interval is more items (or, separately, reducing per-item score
variance itself, e.g., via better-defined rubrics that make individual judgments less noisy, which
shrinks the effective variance term in the CI formula rather than the `n` term).

## Q19 (scenario): You're asked to validate a new LLM judge before it's allowed to gate model releases. Design the validation study end to end.

I'd start by defining the validation population deliberately, not conveniently: sample the
prompt/response pairs to validate on from the actual distribution the judge will be used on in
production — spanning the task categories, difficulty levels, and (importantly)
closeness-of-comparison the release-gating pipeline will actually encounter, including deliberately
over-sampling genuinely close/hard comparisons rather than letting the validation set be dominated
by easy, obviously-one-sided cases, since judges (like humans) are least reliable exactly on the
close calls that matter most for a real gating decision.

I'd collect independent human judgments on this validation set under a proper protocol (module
`003`): multiple raters per item (enough to compute Fleiss' kappa as a human-human agreement
baseline, not just a single-rater label treated as ground truth), position-randomized and
source-blinded pairwise comparisons if that's the judge's protocol, with trained/qualified
annotators and gold-item quality checks built into the collection itself.

I'd then run the candidate judge, under the exact production prompt/protocol (not a simplified
validation-only version), on the identical items, and compute: raw and kappa-corrected agreement
between the judge's verdict and the human majority/aggregate verdict; and, critically, the same
agreement statistic computed between independent human raters on the same items, as the ceiling
reference — the judge's agreement number is uninterpretable in isolation and only means something
relative to that human-human ceiling.

I would not stop at an aggregate agreement number — I'd slice by task category, by
response-length-difference bucket (a targeted probe for residual verbosity bias even after any
mitigation is applied), by whether either response came from the judge's own model family (a
targeted probe for self-preference bias), and by closeness-of-comparison (agreement on clear-cut
cases vs. near-ties), because an aggregate number computed mostly over easy cases will overstate
reliability on the harder cases that actually decide most real release calls.

Only if the judge's sliced agreement is close to the human-human agreement ceiling across these
slices — not just in aggregate — would I approve it for gating, and even then I'd set an explicit
re-validation trigger: any change to the judge model version, its prompt template, or a material
shift in the population of systems it's evaluating (e.g., a new model family entering the comparison
pool) requires re-running this validation before continuing to trust the judge operationally, rather
than treating this as a one-time certification.

## Q20: Summarize, at a staff level, why no single evaluation method covered in this module is sufficient on its own for a frontier model release decision, and what "sufficient" evaluation actually looks like as a system.

Every method here has a validity domain it doesn't extend beyond, and the domains are complementary
rather than overlapping. Automatic metrics (module `001`) are cheap, deterministic, and correctly
scoped to closed-answer-space tasks and regression monitoring, but structurally cannot judge
correctness, helpfulness, or instruction adherence for open-ended generation. LLM-as-judge (module
`002`) fills that gap at scale but inherits real, measured biases (position, verbosity,
self-preference) and is only as trustworthy as its most recent validation against human judgment —
it's a fast proxy for a ground truth it doesn't itself constitute. Human evaluation (module `003`)
is that ground truth, but is slow, expensive, and scale-limited, which is exactly why it can't be
the primary tool for the iteration-speed feedback loop model development actually runs on.
Contamination-aware design (module `004`) doesn't make any benchmark's score more meaningful on its
own — it protects the evaluation program's overall conclusions from a background risk that no single
number's face value reveals. Trajectory evaluation (module `005`) is necessary the moment the system
under test is an agent rather than a single-turn responder, and has its own open, unsolved edge
(open-ended tasks with no unique correct trajectory) that no method here fully resolves. Red-teaming
(module `006`) answers "what's the worst case," a question capability eval isn't designed to ask and
would systematically undersample if it tried. And statistical rigor (module `007`) is the discipline
that keeps every one of the above numbers honest about how much confidence it actually supports,
which none of the other six methods provide for themselves.

A release decision that leans on only one or two of these — say, a strong automatic-benchmark
leaderboard position plus an unvalidated LLM-judge win rate — is missing coverage on process
quality, worst-case behavior, statistical robustness, and the possibility that some of its headline
numbers are quietly contamination-inflated. "Sufficient" evaluation, at a staff level, isn't a
single more-rigorous method substituting for all the others; it's a layered system where cheap,
high-frequency signals (automatic metrics, validated LLM judges) handle iteration-speed decisions,
human evaluation and red-teaming (internal and external) handle the highest-stakes and worst-case
questions those cheaper proxies structurally can't answer, trajectory-level scoring is added
wherever the system under test operates in a loop rather than single-turn, contamination-aware
design protects the program's numbers from a permanent background risk, and every comparative claim
that actually drives a decision is checked against the statistical-rigor discipline before anyone
treats it as real. The honest failure mode to be able to name in an interview is not "we used the
wrong metric" — it's "we let one convenient, cheap signal answer a question it was never designed to
answer, because it was the number we already had."

