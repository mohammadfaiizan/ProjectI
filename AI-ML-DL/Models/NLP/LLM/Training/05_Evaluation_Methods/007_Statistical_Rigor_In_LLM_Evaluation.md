# Statistical Rigor in LLM Evaluation

## 0. The problem: point estimates get treated as ground truth

A leaderboard entry reading "GPT-X: 84.3% on Benchmark Y" looks precise. It is, almost always, a
single number computed from a single run — one sampling temperature, one prompt template, one random
seed for whatever stochasticity exists in decoding, evaluated on a fixed (often modest-sized) set of
items — reported with no uncertainty interval and often compared to a competitor's
equally-single-run number to declare a winner. Every part of that sentence is a place where real
variance exists and gets silently discarded. This module is about naming those variance sources
explicitly, quantifying them with the standard tools (bootstrap confidence intervals, significance
tests), and using that quantification to answer the practical questions staff-level evaluation work
actually needs answered: is a reported score difference between two models real or noise, and how
large does an eval set need to be before its score is a stable estimate at all.

## 1. Sources of variance in a benchmark number

### 1.1 Sampling variance from the finite eval set itself

Even with a perfectly deterministic model and decoding procedure, a benchmark score computed on `n`
items is a sample estimate of the model's "true" accuracy on the full (often conceptually infinite)
population of possible items the benchmark is meant to represent. Two different random `n`-item
subsets of that population would give two different scores purely from sampling variance — this is
the baseline, irreducible-without-more-data source of uncertainty, quantified in Section 3.

### 1.2 Random seed / decoding stochasticity

If the model is sampled at temperature `> 0` (common even for "evaluation," especially for
generation tasks scored by a judge or by execution rather than exact match), the same prompt on the
same model produces different outputs on different runs, and the resulting benchmark score is itself
a random variable across seeds, independent of eval-set sampling variance. Even greedy (temperature
0) decoding is not always perfectly deterministic across hardware/batching configurations for some
inference stacks (numerical non-determinism from batching, kernel selection, or floating-point
non-associativity in parallel reductions can produce small output differences run to run) — a
subtlety worth knowing but usually a much smaller effect than genuine sampling-temperature variance.

### 1.3 Prompt-template sensitivity

Benchmark scores for the same model on the same items can shift by several points — sometimes
double-digit points on smaller or less robustly instruction-tuned models — purely from changing
surface details of the prompt template: the exact instruction wording, few-shot example formatting,
whether the answer format is requested as "Answer: X" vs. "The answer is X," delimiter choice, and
even whitespace/casing conventions. This has been documented repeatedly in the literature (e.g.,
prompt-sensitivity studies accompanying various benchmark evaluation harnesses) and means that two
labs reporting "the same benchmark" score for comparable models are not necessarily measuring the
same thing at all if their harnesses use different prompt templates — a huge, underappreciated
source of apparent-but-not-real performance differences across papers.

### 1.4 Evaluation-harness and scoring-function implementation differences

Beyond the prompt template, the exact answer-extraction and scoring logic (how do you parse a
free-form response to decide whether it matches the expected multiple-choice letter, how forgiving
is the numeric-answer normalization for a math benchmark) varies across evaluation harnesses (e.g.,
differences between an official benchmark release's reference implementation, the widely used
`lm-evaluation-harness` defaults, and a lab's in-house harness), and has been shown to produce
meaningfully different scores for the *same* model checkpoint on the *same* nominal benchmark
depending purely on harness choice. This is a distinct variance source from 1.1-1.3: it's not
randomness at all, it's a silent methodological difference that looks like a score difference.

### 1.5 Why this matters practically

None of this is a pedantic statistical nitpick — the practical consequence is that a headline claim
like "our model beats the competitor by 1.5 points on Benchmark Y" is frequently not distinguishable
from noise once these variance sources are accounted for, and treating it as a real, meaningful
capability difference (which then drives real decisions — which model to ship, which training run to
trust, which paper's claims to believe) without checking is a common and consequential eval error.
The rest of this module is the toolkit for checking.

## 2. Confidence intervals for a single benchmark score

### 2.1 The binomial case (accuracy-style metrics)

For a metric that is a simple accuracy over `n` i.i.d.-ish items (each item scored right/wrong), the
score `p_hat = k/n` (k correct out of n) is a sample proportion, and its uncertainty is standard
binomial-proportion territory. The naive normal approximation (Wald interval), `p_hat +/- z *
sqrt(p_hat*(1-p_hat)/n)`, is the most commonly seen form in practice but is known to perform poorly
(systematically too narrow, and can even produce nonsensical intervals below 0 or above 1)
especially for small `n` or `p_hat` near 0 or 1 — exactly the regime a lot of hard benchmarks sit in
(many-item benchmarks scored near ceiling, or small, expensive expert-graded evals). The **Wilson
score interval** corrects for this and is the better default:

```python
import math

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion. z=1.96 -> ~95% CI."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))
```

### 2.2 The general case: bootstrap confidence intervals

Many LLM-eval metrics are not simple binomial accuracies (pass@k, an LLM-judge win rate that's
itself aggregated from noisy per-item judgments, a mean numeric score like a Likert average,
ROUGE/BERTScore F1 averaged over items) — for these, the **nonparametric bootstrap** is the
standard, general-purpose tool, because it makes no distributional assumption about the metric and
works identically regardless of what the per-item score function actually is:

```python
import numpy as np

def bootstrap_ci(per_item_scores: np.ndarray, n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Nonparametric bootstrap CI for the mean of an arbitrary per-item score array.
    per_item_scores: shape [n_items], one score per eval example (0/1, or any real-valued score).
    Returns (point_estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    n = len(per_item_scores)
    point_estimate = per_item_scores.mean()

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        resample_idx = rng.integers(0, n, size=n)      # sample n items WITH replacement
        boot_means[b] = per_item_scores[resample_idx].mean()

    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper
```

The mechanism: resampling the observed items *with replacement* `n_boot` times and recomputing the
metric on each resample simulates "what score would I have gotten from a different random `n`-item
draw from the same underlying population," using the empirical distribution of observed scores as a
stand-in for the true (unknown) population distribution. The spread of the resulting `n_boot`
bootstrap means directly gives an empirical confidence interval, with no assumption that the
per-item score is binomial, normal, or any other parametric family — this generality is exactly why
the bootstrap is the default tool for judge-derived win rates, LLM-judge Likert averages, and any
other non-trivial aggregate LLM-eval statistic. For pass@k-style metrics specifically, note that the
correct per-item quantity to bootstrap is the *unbiased pass@k estimator* per item (Chen et al.
2021's combinatorial formula from multiple samples per item, not a naive single-sample binary hit),
computed once per item before bootstrapping over items — bootstrapping over the raw
multiple-samples-per-item data directly, without first collapsing to the right per-item statistic,
is a subtle and easy mistake.

### 2.3 A practical rule of thumb for required sample size

For a target confidence-interval half-width `E` at a proportion near `p` (worst case `p = 0.5`,
which maximizes variance and gives the most conservative/largest required `n`), the
normal-approximation sample-size formula is:

```
n >= z^2 * p * (1 - p) / E^2
```

For a 95% CI (`z ≈ 1.96`) and a desired half-width of `E = 0.02` (i.e., +/- 2 percentage points) at
the conservative `p = 0.5`: `n >= 1.96^2 * 0.25 / 0.02^2 ≈ 2401`. Halving the desired margin to `E =
0.01` quadruples the required `n` to roughly 9604 — the `1/E^2` scaling means tightening precision
is expensive, and this is precisely why small, expensive, expert-graded evals (a few hundred items,
sometimes fewer) inherently carry wide confidence intervals — a 300-item benchmark at `p ≈ 0.5` has
a 95% CI half-width around `1.96 * sqrt(0.25/300) ≈ 5.7` percentage points, meaning score
differences smaller than roughly that margin between two models are not distinguishable from noise
on that eval set size alone, no matter how many decimal places the leaderboard reports.

## 3. Significance testing for comparing two models

### 3.1 Why "compare two independent proportions" is usually the wrong test here

A naive approach — treat model A's score and model B's score as two independent binomial proportions
and run a two-proportion z-test — is a common mistake when the two models were evaluated **on the
same set of items** (the standard case: you run both models on the identical benchmark). Treating
paired data as independent throws away the pairing information and gives an overly conservative
(wider than necessary) test, because it ignores item-level correlation — some items are just harder
for every model, and both models' errors on those items are correlated, which the
independent-proportions test doesn't exploit.

### 3.2 McNemar's test for paired binary outcomes

When both models are scored right/wrong on the same `n` items, McNemar's test is the standard paired
test for whether their disagreement pattern is asymmetric (one model tends to get right what the
other gets wrong, more often in one direction than the other) — it only uses the *discordant* pairs
(items where the two models disagree) and tests whether that disagreement is symmetric:

```python
from scipy.stats import chi2

def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """a_correct, b_correct: boolean arrays, same length, aligned by item.
    Returns p-value for the null hypothesis that A and B have equal accuracy."""
    n01 = np.sum(~a_correct & b_correct)   # A wrong, B right
    n10 = np.sum(a_correct & ~b_correct)   # A right, B wrong
    n_discordant = n01 + n10
    if n_discordant == 0:
        return 1.0
    # continuity-corrected chi-square statistic (standard for small-to-moderate n_discordant)
    stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    return 1 - chi2.cdf(stat, df=1)
```

McNemar's test is appropriate specifically for binary, paired, same-item outcomes — the common case
for accuracy-style benchmarks. It says nothing about the *magnitude* of the difference (that's what
the paired bootstrap in 3.3 or the CI in Section 2 is for) — it only tests whether an observed
asymmetry in who-gets-what-right is unlikely under the null of "no real difference."

### 3.3 Paired bootstrap for the difference in scores

For non-binary metrics (mean judge scores, ROUGE, any continuous per-item score), or when you want a
confidence interval on the *magnitude* of the difference rather than just a p-value, the paired
bootstrap is the general tool: resample item *indices* (not each model's scores independently) so
the pairing is preserved, and look at the distribution of the resulting score-difference:

```python
def paired_bootstrap_diff(scores_a: np.ndarray, scores_b: np.ndarray,
                           n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """scores_a, scores_b: per-item scores for models A and B, aligned by item index."""
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    observed_diff = scores_a.mean() - scores_b.mean()

    boot_diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # same resampled indices for BOTH models
        boot_diffs[b] = scores_a[idx].mean() - scores_b[idx].mean()

    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    # two-sided p-value: fraction of bootstrap diffs crossing zero, doubled
    p_value = 2 * min((boot_diffs >= 0).mean(), (boot_diffs <= 0).mean())
    return observed_diff, lower, upper, p_value
```

Resampling the *same* item indices for both models on each bootstrap replicate is the critical
detail — this is what preserves the pairing structure (each resample represents "a different draw of
which items we happened to test both models on," not "an independent different draw for each model
separately"), giving correctly-sized (not overly wide) confidence intervals and correctly-calibrated
significance, analogous to how a paired t-test is more powerful than an unpaired one for the same
underlying reason.

### 3.4 Multiple-comparison correction when testing many benchmarks or many model pairs at once

Running significance tests across many benchmarks simultaneously (a common reporting pattern — "we
test on 15 benchmarks and are significantly better on 9 of them") inflates the false-positive rate
unless corrected: at a 5%-per-test significance threshold, testing 15 independent hypotheses gives a
non-trivial chance of at least one spurious "significant" result even if there's no true difference
anywhere (`1 - 0.95^15 ≈ 54%` chance of at least one false positive under the global null). Standard
corrections — Bonferroni (divide the significance threshold by the number of tests, simple but
conservative) or the less conservative Benjamini-Hochberg false-discovery-rate procedure — should be
applied whenever a paper or report claims significance across a multi-benchmark suite, and their
absence in a "significantly better on 9 of 15 benchmarks"-style claim is a common, checkable red
flag.

## 4. Common statistical pitfalls in published LLM eval claims

- **Reporting a single run with no variance information at all.** The most basic and most common
  pitfall: a benchmark table with bare point estimates and no confidence interval, no seed-variance
  report, and no indication of how many runs the number is averaged over. Absent that information, a
  reader cannot tell whether a reported gap between two models reflects a real capability difference
  or falls well within the noise floor implied by Section 2's sample-size arithmetic.
- **Best-of-N reported as if it were single-sample performance, without disclosing N.** Reporting
  "pass@1: 94%" when the underlying number was actually computed by taking the best of, say, 100
  sampled attempts per problem (and calling the best-of-100 result "pass@1" loosely, or reporting a
  best-of-N number in a headline figure while only disclosing N deep in an appendix or not at all)
  dramatically overstates the model's actual single-attempt reliability. Best-of-N and pass@1
  measure genuinely different things (an agent that succeeds 1 time in 100 tries via best-of-N is
  not "94% reliable" in any deployment-relevant sense if you can't afford 100 tries per real task),
  and conflating them — deliberately or through imprecise reporting — is one of the most
  consequential and, unfortunately, recurring reporting failures in the field. The fix, as a reader:
  always check what `k` (or `N`) an at-k / best-of-N metric actually used, and treat any pass@k
  number with undisclosed or buried `k` with real suspicion.
- **Cherry-picked prompts or qualitative examples presented as representative.** Demo-style
  qualitative examples ("look at this impressive completion") are the weakest possible form of
  evidence precisely because they are, by construction, selected by the presenter — they carry zero
  information about the rate at which the model produces outputs of that quality on the broader task
  distribution, and should never be treated as evidence of a claim about typical or aggregate
  performance, only as an existence proof that a capability is reachable under some circumstances.
- **Cherry-picked prompt templates or few-shot examples that happen to maximize the reported model's
  score.** Related to Section 1.3's prompt-sensitivity variance: if a paper's own prompt template
  was tuned (even implicitly, through iterative trial and error while writing the paper) to work
  well specifically for the paper's proposed model, and the same tuning effort was not applied to
  baseline comparisons, the resulting benchmark gap partly reflects prompt-engineering effort
  asymmetry, not model-capability asymmetry — a subtle version of cherry-picking that doesn't
  require any single cherry-picked example, just an asymmetric amount of engineering effort applied
  on one side of the comparison.
- **Comparing scores computed under different evaluation harnesses or prompt formats as if they were
  the same measurement** (Section 1.3-1.4) — e.g., quoting a competitor's self-reported number
  (produced under the competitor's own harness/prompt choices) next to your own number (produced
  under your harness) as a head-to-head comparison, when neither number was produced under a shared,
  controlled protocol. The methodologically correct comparison re-runs every compared system under
  one held-fixed harness and prompt template; anything less is comparing two different experiments
  and calling it one.
- **Ignoring contamination as a confound when interpreting an unusually high score** (module `004`)
  — a model that scores implausibly well on a widely public benchmark relative to its performance on
  structurally similar private/held-out evals is a pattern worth investigating as a possible
  contamination artifact before accepting the public number as a clean capability measurement.
- **Treating a benchmark at or near ceiling as still discriminative.** If most compared models score
  above, say, 95% on a benchmark, the remaining few points of "difference" are frequently dominated
  by noise (Section 2's small-margin problem is exacerbated near a hard ceiling, where remaining
  errors may be concentrated in a handful of genuinely ambiguous or mislabeled items) — treating a
  top-of-leaderboard ranking among near-ceiling scores as a meaningful capability ordering, without
  checking whether the gaps clear a reasonable confidence-interval bar, is a frequent source of
  overconfident claims (this compounds with benchmark-saturation issues covered from the
  benchmark-design side in `..\06_Benchmarks`).

## 5. A practical checklist for reporting (or reading) an LLM eval number responsibly

1. State `n` (eval set size) and, ideally, a confidence interval or bootstrap-derived error bar, not
   just a point estimate.
2. Disclose the decoding configuration (temperature, number of samples per item if pass@k/best-of-N
   is involved, and the exact `k`/`N`).
3. Disclose or hold fixed the prompt template and scoring/answer-extraction logic across every
   system being compared.
4. When claiming model A beats model B, report a paired significance test (McNemar's for binary
   outcomes, paired bootstrap otherwise) or at minimum a CI on the difference, not just two separate
   point estimates with non-overlapping-looking bars eyeballed as "different."
5. Apply multiple-comparison correction when claiming significance across a suite of benchmarks or
   comparisons, not per-test thresholds applied independently.
6. Cross-check an unusually strong result against a private/held-out eval or a contamination check
   before treating it as a clean capability claim (module `004`).
7. Treat any qualitative example as an existence proof, never as evidence of a rate or an average.

## Cross-references

- Automatic-metric mechanics whose scores this module's statistical tools are typically applied to
  are covered in `001_Automatic_Metrics_And_Their_Limits.md`.
- LLM-judge-derived win rates and Bradley-Terry model fitting, a common target for the
  confidence-interval and significance-testing machinery in this module, are covered in
  `002_LLM_As_Judge_Methodology_And_Biases.md`.
- Inter-annotator agreement statistics (Cohen's/Fleiss' kappa), a related but distinct
  measurement-reliability question from the significance-testing question this module focuses on,
  are covered in `003_Human_Evaluation_And_Preference_Collection.md`.
- Contamination as a confound in an unusually high or otherwise anomalous benchmark score is covered
  in `004_Contamination_Aware_Evaluation_Design.md`.
- Named benchmarks' typical sizes, known saturation points, and reporting conventions are covered in
  `..\06_Benchmarks`.

