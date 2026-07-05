# Human Evaluation and Preference Collection

## 0. Why human eval is still the ground truth

Everything in modules `001` and `002` is, ultimately, trying to approximate what a human would think
of a model's output — automatic metrics approximate it through string/embedding overlap with a
reference, LLM judges approximate it by asking another model to render a verdict. Human evaluation
is not an approximation; a human rating a response for helpfulness, correctness, and safety *is* the
target quantity, not a proxy for it, at least insofar as the evaluation goal is "will actual users
find this good." That is exactly why human eval remains the final validation step for LLM-as-judge
pipelines (module `002`, Section 4) and the standard a lab reaches for when a claim is consequential
enough (a competitive benchmark claim, a safety release decision) that an approximate proxy isn't
good enough. The cost of that ground-truth status is covered in Section 4: human eval is slow,
expensive, and hard to scale, which is the entire reason modules `001` and `002` exist as cheaper
substitutes for the parts of the evaluation loop that don't need ground-truth precision every time.

## 1. Study design: what you're actually asking annotators to do

### 1.1 Pairwise preference collection

The annotator sees a prompt and two responses (commonly unlabeled as to source model, and in
randomized left/right position — see Section 1.4) and picks which is better, typically with a small
number of graded options rather than a strict binary, e.g.:

```
A is much better | A is slightly better | Tie | B is slightly better | B is much better
```

This is the human-eval analogue of LLM-as-judge pairwise comparison (module `002`, Section 1.2), and
for the same underlying reason: comparative judgments are cognitively easier and more reproducible
for human raters than absolute scoring, because the rater has a direct point of contrast rather than
an internal, drifting notion of "what does a 4 mean." Pairwise data is also what feeds directly into
Bradley-Terry-style leaderboard fitting (module `002`, Section 1.2) and into RLHF/DPO
preference-pair training data — pairwise human preference collection is, in fact, the foundational
data-collection format behind RLHF as a technique (Christiano et al. 2017; Ouyang et al. 2022's
InstructGPT reward-model training data is exactly this format at scale).

### 1.2 Likert-scale absolute rating

The annotator sees a single prompt/response and assigns a score on a fixed ordinal scale (commonly
1-5 or 1-7), often against an explicit rubric ("1 = harmful or completely unhelpful ... 5 =
excellent, fully addresses the request with no issues"). This is necessary whenever you need an
absolute quality signal rather than a relative one — e.g., tracking whether a production system's
average helpfulness rating is trending up or down over time, independent of any specific comparison
— and shares the calibration weaknesses discussed for pointwise LLM-judge scoring (module `002`,
Section 1.1): different annotators (and the same annotator on different days) apply different
implicit thresholds to the scale, so raw Likert scores need either heavy
annotator-training/calibration investment (Section 2) or explicit statistical correction (e.g.,
z-scoring each annotator's ratings against their own historical mean/variance before pooling across
annotators) to be safely comparable.

### 1.3 Side-by-side comparison with rich annotation

A richer variant of pairwise comparison used for detailed diagnostic evaluation rather than just a
winner label: the annotator sees both responses side by side and, in addition to (or instead of) an
overall preference, annotates specific spans or dimensions — e.g., highlighting factually incorrect
spans, marking which response better follows a specific constraint in the instruction, or filling in
a structured rubric with sub-scores per dimension (correctness, completeness, tone, safety) for each
response before giving an overall verdict. This costs more annotator time per item than a bare
preference click but produces much more actionable signal for debugging *why* one model
underperforms, rather than just *that* it does — valuable when the study's purpose is model
improvement, not just leaderboard ranking.

### 1.4 UX and protocol details that materially affect data quality

- **Randomize left/right position of A and B** for every item, independently, to prevent position
  bias in human raters (an effect documented in human-eval literature analogous to LLM-judge
  position bias, module `002` Section 2.1, though generally smaller in magnitude for careful,
  engaged human raters than for some LLM judges).
- **Blind the source model identity.** Annotators should not know which system produced which
  response — knowledge of "this is our new model" vs. "this is the baseline" reliably introduces
  conscious or unconscious bias toward the expected/desired outcome.
- **Present the actual instruction/context, not just the response**, and require the annotator to
  re-read it for each judgment — raters fatigue and start pattern-matching on response surface
  features alone if the instruction is easy to skip past, which reintroduces exactly the
  length/fluency biases human eval was meant to avoid relative to automatic metrics.
- **Collect a free-text rationale alongside the categorical verdict** where feasible; rationales are
  useful for auditing rater quality (does the stated reason logically support the verdict?), for
  downstream error analysis, and for detecting straight-lining (raters who click through without
  engaging).
- **Cap session length and monitor per-item time.** Annotator judgment quality degrades with
  fatigue; items completed implausibly fast are a standard data-quality red flag warranting
  exclusion or re-review.

## 2. Annotator selection and training

### 2.1 Who should annotate

The right annotator pool depends on what's being evaluated:

- **General helpfulness/preference on everyday tasks**: a broad, demographically diverse crowdworker
  pool (via platforms like Amazon Mechanical Turk, Prolific, or a vendor-managed crowd) is
  appropriate and arguably desirable — general-purpose assistants are used by a broad population,
  and a narrow annotator pool (e.g., only ML researchers) can systematically misjudge what a typical
  user finds helpful or clear.
- **Domain-specialized correctness (code, math, medical/legal content, multilingual quality)**:
  requires annotators with the relevant expertise or credentials; a crowdworker with no coding
  background cannot reliably judge whether a code review response caught a real bug, and using
  non-expert raters for expert-level content is a common, avoidable source of noisy or
  systematically wrong ground truth.
- **Safety and policy-sensitive evaluation**: typically requires trained, vetted annotators working
  under explicit written policy guidelines (what counts as a policy violation, edge-case
  adjudication rules) rather than general crowd raters applying intuitive judgment, precisely
  because "is this response harmful" needs consistent application of a specific, often
  legally/ethically consequential standard rather than individual intuition — this overlaps with
  red-teaming methodology (module `006`) but is a distinct evaluation activity (rating pre-existing
  outputs against a policy rather than actively trying to elicit violations).

### 2.2 Training and calibration

- **Written guidelines with worked examples.** Every rating protocol needs an explicit rubric
  document with concrete example items at each scale point / each preference-strength level, because
  "rate helpfulness 1-5" without anchoring examples produces wildly inconsistent internal scales
  across annotators.
- **Qualification tasks / gold-standard items.** Before an annotator is allowed to contribute
  production data, have them rate a held-out set of items with known, expert-agreed answers, and
  require a minimum agreement rate with that gold standard to qualify. This filters out annotators
  who are not engaging seriously or not understanding the rubric, and is standard crowd-annotation
  practice.
- **Ongoing gold-item injection.** Continue seeding a small fraction of gold-standard items (with
  known answers, invisible to the annotator as "special") into the regular annotation stream during
  production data collection, to monitor whether a qualified annotator's quality drifts over time
  (fatigue, disengagement) and to catch this before it contaminates a large batch of real data.
- **Calibration sessions and adjudication for disagreement.** For high-stakes annotation (safety
  labeling, expert correctness judgments), a common design has multiple annotators independently
  label the same item, with automatic escalation to a senior/expert adjudicator when annotators
  disagree beyond some threshold — this simultaneously produces cleaner final labels and generates
  the disagreement data needed to refine guidelines that are ambiguous in practice.
- **Feedback loops.** Periodically reviewing a sample of each annotator's judgments against expert
  review and providing corrective feedback measurably improves consistency over a purely "guidelines
  and gold items, no further contact" design, at the cost of added program-management overhead — a
  standard trade-off decision in any serious annotation program.

## 3. Measuring inter-annotator agreement

If a rating protocol is well-specified and annotators are well-trained, independent annotators
should agree with each other well above chance on the same items. Measuring *how much* above chance
is the entire point of agreement statistics — raw agreement percentage is a poor and misleading
standalone metric because it does not account for the agreement that random-chance labeling would
already produce, which can be substantial when the label distribution is skewed (e.g., if 90% of
responses are rated "good," two annotators randomly guessing "good" most of the time will already
agree ~81% of the time with each other, giving a misleadingly high raw-agreement number that
reflects nothing about actual rater consistency).

### 3.1 Cohen's kappa (two annotators)

Cohen's kappa corrects observed agreement `p_o` for the agreement `p_e` expected by chance, given
each annotator's own marginal label distribution:

```
kappa = (p_o - p_e) / (1 - p_e)
```

- `p_o` = fraction of items where the two annotators gave the same label (raw agreement).
- `p_e` = sum over each label category of (fraction of items annotator 1 assigned that label) ×
  (fraction of items annotator 2 assigned that label) — i.e., the agreement rate you'd expect if
  both annotators assigned labels independently at their own observed base rates, with no actual
  relationship between their judgments.

```python
def cohens_kappa(labels_1: list, labels_2: list) -> float:
    assert len(labels_1) == len(labels_2)
    n = len(labels_1)
    categories = set(labels_1) | set(labels_2)

    p_o = sum(a == b for a, b in zip(labels_1, labels_2)) / n

    p_e = 0.0
    for c in categories:
        p1_c = sum(a == c for a in labels_1) / n
        p2_c = sum(b == c for b in labels_2) / n
        p_e += p1_c * p2_c

    return (p_o - p_e) / (1 - p_e) if p_e != 1 else 1.0
```

`kappa = 1` means perfect agreement beyond chance; `kappa = 0` means agreement is entirely explained
by chance (the annotators' judgments carry no information about each other, given their marginals);
`kappa < 0` means agreement is *worse* than chance (systematic disagreement, e.g., the two
annotators have inversely related judgment patterns). The commonly cited (Landis & Koch 1977)
informal bands — below 0 poor, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial,
0.8-1.0 almost perfect — are a widely used rule of thumb, not a rigorous statistical threshold, and
should be applied with the caveat that "acceptable" kappa is genuinely task-dependent: subjective
quality/preference tasks routinely land in the 0.3-0.6 range even for well-trained annotators
(people legitimately differ on what they find more helpful or better-written), while factual
correctness checks on unambiguous ground truth should be pushed toward 0.7+ before the data is
trusted, and a low kappa there is a signal to fix the guidelines rather than an acceptable property
of the task.

### 3.2 Fleiss' kappa (more than two annotators)

When more than two annotators rate each item (common in practice — 3-5 raters per item is a standard
design to also enable majority-vote aggregation and adjudication), Fleiss' kappa generalizes the
same chance-correction logic:

```
kappa = (P_bar - P_bar_e) / (1 - P_bar_e)
```

where, for `N` items each rated by `n` annotators into `k` categories:

- `P_bar` = the mean, over items, of the observed pairwise agreement rate *within* each item (for
  item `i` with `n_ij` annotators assigning category `j`, the item's own agreement rate is `(sum_j
  n_ij*(n_ij - 1)) / (n*(n-1))`, then averaged across all `N` items).
- `P_bar_e` = sum over categories `j` of `(p_j)^2`, where `p_j` is the overall fraction of all
  annotations (across all items and annotators) that fell into category `j` — the chance-agreement
  rate implied by the pooled label marginals.

```python
def fleiss_kappa(rating_matrix: np.ndarray) -> float:
    """rating_matrix[i, j] = number of annotators who assigned category j to item i.
    Shape: [N items, k categories]. Each row sums to n (annotators per item)."""
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)[0]   # assumes constant annotators/item

    p_j = rating_matrix.sum(axis=0) / (N * n)          # marginal category rates
    P_e_bar = (p_j ** 2).sum()

    P_i = (rating_matrix * (rating_matrix - 1)).sum(axis=1) / (n * (n - 1))
    P_bar = P_i.mean()

    return (P_bar - P_e_bar) / (1 - P_e_bar) if P_e_bar != 1 else 1.0
```

Fleiss' kappa is the right tool whenever the annotation design pools an interchangeable panel of
raters per item (not the same fixed pair every time); it is the standard statistic reported for
crowd-annotated LLM-eval datasets with 3+ raters/item. For ordinal scales (Likert ratings) where a
disagreement of 1 point should count as "less wrong" than a disagreement of 4 points, **weighted
kappa** (Cohen's kappa with a penalty matrix that credits near-misses) or **Krippendorff's alpha**
(which generalizes cleanly to ordinal/interval data and to missing data — not every annotator rates
every item — which is common in real crowd-sourcing pipelines with variable coverage) are more
appropriate than unweighted kappa's strict-match-or-nothing treatment; unweighted kappa on an
ordinal scale can understate agreement by treating a 4-vs-5 near-miss identically to a 1-vs-5
outright contradiction.

### 3.3 Using agreement statistics operationally

Inter-annotator agreement is not just a data-quality report card — it directly bounds what
conclusions the collected data can support. If Fleiss' kappa on a pairwise preference task is, say,
0.35 (fair-to-moderate), that tells you individual human raters disagree with each other often
enough that a *single* rater's verdict on a *single* item is a fairly noisy signal of "true"
preference, which has direct consequences: you need multiple raters per item and majority/aggregate
voting rather than trusting single-rater labels, and you need correspondingly larger sample sizes
before a preference-rate difference between two models is distinguishable from noise (this connects
directly to the sample-size and confidence-interval treatment in module `007`). Low agreement is
also frequently a signal that the task itself is genuinely ambiguous or the guidelines are
underspecified — a common and productive response to low kappa is to inspect the specific
disagreement items, refine the rubric to resolve the ambiguity that's causing them, and re-measure,
rather than treating the kappa number as a fixed property of the task to just report and move past.

## 4. Cost, scale, and latency trade-offs

This is the section that explains why LLM-as-judge (module `002`) exists at all, despite human eval
being the ground truth:

- **Cost.** Crowd-sourced pairwise comparisons commonly run in the range of roughly $0.05-$1+ per
  judgment depending on task complexity, annotator expertise required, and quality-control overhead
  (qualification, gold-item auditing, multiple raters per item) — cheap relative to, say, a clinical
  trial, but expensive relative to an LLM judge call, which can be several orders of magnitude
  cheaper per judgment and only gets cheaper as inference costs fall. Expert-annotator studies
  (domain specialists, safety-trained raters) cost substantially more per judgment than general
  crowd work, sometimes 10-100x, because the labor market for that expertise is smaller and the
  hourly rate is higher.
- **Latency.** A human-eval study — recruiting/qualifying annotators, running the annotation batch,
  quality-checking and adjudicating disagreements, aggregating results — typically takes days to
  weeks end to end, even for a moderately sized study, and expert-annotator or safety-sensitive
  studies often take longer due to smaller available annotator pools and heavier review processes.
  This is fundamentally incompatible with the iteration cadence of model development, where a
  researcher may want a quality signal on every training checkpoint, every day, or even every few
  hours during a hyperparameter search.
- **Scale ceiling.** There is a practical ceiling on how many human judgments you can collect per
  unit time even with a large crowd budget, set by annotator pool size and quality-control
  throughput; LLM judges have no comparable ceiling — you can score an arbitrarily large eval set or
  an arbitrarily large stream of production traffic, limited only by inference compute budget.
- **Consistency over time is harder for humans.** A crowd-annotation pool's composition,
  calibration, and even guideline interpretation can drift across data-collection rounds run weeks
  or months apart, making longitudinal comparisons (is quality improving release over release)
  noisier unless the same annotator pool and protocol are held rigorously constant — an LLM judge,
  held at a fixed model version and prompt, gives more mechanically reproducible scoring across
  time, at the cost of the biases in module `002`.

### 4.1 Why these trade-offs motivate, but don't fully justify, replacing human eval with LLM-as-judge

The trade-offs above are real and are exactly why virtually every LLM development pipeline uses
LLM-as-judge (and automatic metrics) for the bulk of day-to-day, high-frequency evaluation. But
"cheaper and faster" is not the same claim as "equally valid," and three things specifically limit
how far the substitution can go:

1. **An LLM judge's validity is itself established by comparison to human judgment** (module `002`,
   Section 4) — human eval is not just one more evaluation option alongside LLM-as-judge, it is the
   calibration reference LLM-as-judge is validated against, which makes eliminating human eval
   entirely a form of removing your own ground truth. A judge pipeline that has never been checked
   against real human preferences on the relevant task distribution is an unvalidated proxy, not a
   cheaper equivalent.
2. **Novel or shifting task distributions need re-grounding.** As a model's capability profile
   changes (new tools, new domains, new failure modes), the judge's own reliability on that new
   distribution is unknown until re-validated against fresh human judgments — you cannot assume a
   judge validated on last year's task mix remains valid on this year's task mix, especially for
   capabilities that are qualitatively new (module `005`'s agentic evaluation is a good example:
   judges validated on single-turn chat quality are not automatically valid judges of multi-step
   tool-use trajectories).
3. **The most consequential decisions still warrant the ground truth, not the proxy.** For claims
   with high stakes — a headline competitive benchmark claim, a safety-relevant release decision, a
   claim likely to be publicly scrutinized or disputed — the cost of a human study is small relative
   to the cost of being wrong, which is exactly the calculus that keeps human eval as the standard
   for final validation even in organizations that use LLM-as-judge pervasively for internal
   iteration.

The practical operating model most mature eval organizations converge on: automatic metrics for
cheap, high-frequency regression signals; LLM-as-judge for the bulk of iteration-speed model
comparison and large-scale preference-data generation; human eval reserved for judge validation,
final release decisions, and any claim significant enough that an unvalidated or partially-validated
proxy is not an acceptable basis for the decision.

## Cross-references

- LLM-as-judge methodology, its documented biases, and how it is validated against the human
  judgments this module describes collecting are covered in
  `002_LLM_As_Judge_Methodology_And_Biases.md`.
- Automatic metrics, which sit even further along the cost/speed-vs-validity trade-off than
  LLM-as-judge, are covered in `001_Automatic_Metrics_And_Their_Limits.md`.
- Sample-size and confidence-interval treatment for preference rates and agreement-limited data is
  covered in `007_Statistical_Rigor_In_LLM_Evaluation.md`.
- Specialized human-annotation protocols for safety policy adjudication overlap with, but are
  distinct from, red-teaming methodology, covered in
  `006_Red_Teaming_And_Adversarial_Evaluation.md`.

