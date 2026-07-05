# Evaluation Methods — Interview Questions (Part 1)

## Q1: Mechanistically, why does BLEU correlate poorly with human judgment for open-ended generation, and why does this problem get worse as a task becomes more open-ended?

BLEU is a geometric mean of clipped n-gram precisions (n=1..4) between a candidate and one or a few
references, multiplied by a brevity penalty. Every component of that computation only sees discrete
token identity — it has no notion of synonymy, paraphrase, or semantic equivalence. Two answers that
convey the same meaning in different words share few or no n-grams and score close to zero against
each other, identical to the score two answers would get if one were simply wrong. BLEU therefore
cannot distinguish "correct but phrased differently than the reference" from "incorrect," which are
the two most common ways a fluent LLM response actually differs from a single reference string.

This gets structurally worse as a task becomes more open-ended because the number of equally-valid
surface realizations of a correct answer grows with the task's openness. Machine translation (BLEU's
original target) has comparatively low surface-form freedom — there are only so many reasonable ways
to render a given sentence in another language, and professional reference translations cover a
meaningful fraction of that space. "Explain why the sky is blue" or "write a product description"
have an effectively unbounded space of correct phrasings, so any fixed small reference set captures
a vanishingly small fraction of valid answers, and BLEU's score becomes dominated by which arbitrary
phrasing the reference-writer happened to choose rather than by whether the candidate is actually
correct. This is also why BLEU has been shown in dialogue-evaluation studies to sometimes correlate
weakly or even negatively with human quality judgments — generic, safe responses ("I'm not sure,"
"that's interesting") have decent odds of overlapping with a diverse reference pool purely by using
common words, while specific, high-quality, informative responses diverge lexically from any one
reference and score worse.

## Q2: Why can an instruction-tuned, RLHF'd model have higher perplexity on a held-out slice of raw web text than its own base model, while still being strongly preferred by human raters?

Perplexity measures how well a model's predicted next-token distribution matches a specific
reference distribution under teacher forcing — it's a distributional-fit statistic, not a quality
statistic. A base model is trained purely to imitate the statistics of its pretraining corpus (raw
web text, forums, books), so it will, by construction, assign relatively high probability to the
kind of token sequences that occur in more raw web text — that's what minimizing pretraining
cross-entropy loss optimizes for.

RLHF and instruction tuning deliberately shift the model's output distribution away from raw-web
statistics: toward being concise rather than rambling, toward directly answering rather than
mimicking whatever a web page happened to do next, toward refusing certain requests, toward a
specific assistant "voice." That shift is the entire point of alignment training — and it
necessarily moves probability mass away from the token sequences a raw-web held-out set contains,
which mechanically increases cross-entropy (and therefore perplexity) on that held-out set, even
though the aligned model produces responses humans overwhelmingly prefer in actual use. Perplexity
on a fixed reference corpus and human preference are simply different axes: one asks "does this
match a training-adjacent text distribution," the other asks "is this a good response to this
specific prompt for this specific user," and post-training explicitly, correctly optimizes for the
second at some cost to the first. This is the textbook demonstration that perplexity is a
training-time health metric, not a deployment-quality metric.

## Q3 (coding): Implement BERTScore's precision/recall/F1 from a candidate and reference embedding matrix, and explain what the greedy matching step is actually doing.

```python
import numpy as np

def bertscore(cand_embs: np.ndarray, ref_embs: np.ndarray,
              cand_idf: np.ndarray | None = None,
              ref_idf: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    cand_embs: [n_cand, d] L2-normalized contextual token embeddings for the candidate
    ref_embs:  [n_ref, d]  L2-normalized contextual token embeddings for the reference
    cand_idf, ref_idf: optional [n_cand], [n_ref] IDF weights for each token
    """
    sim = cand_embs @ ref_embs.T                      # [n_cand, n_ref] cosine similarities

    best_match_for_cand = sim.max(axis=1)             # each candidate token's best reference match
    best_match_for_ref = sim.max(axis=0)              # each reference token's best candidate match

    if cand_idf is not None:
        precision = np.average(best_match_for_cand, weights=cand_idf)
    else:
        precision = best_match_for_cand.mean()

    if ref_idf is not None:
        recall = np.average(best_match_for_ref, weights=ref_idf)
    else:
        recall = best_match_for_ref.mean()

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
```

The greedy matching step (`sim.max(axis=1)` / `sim.max(axis=0)`) is computing, for every token on
one side, its single best (most similar) counterpart on the other side, with no constraint that the
matching be one-to-one or order-preserving — it's an alignment-free soft-overlap measure. Precision
asks "for every candidate token, how similar is its best available reference match" (are we saying
things that resemble what's in the reference), and recall asks the mirror question from the
reference's side (did we cover what the reference token set contains). This is a direct continuous
generalization of BLEU's discrete "does this n-gram appear in the reference at all" check and
ROUGE's discrete LCS-based recall — same precision/recall structure, but matching happens by
embedding proximity instead of string identity, which is exactly what lets it register "feline" as a
near-match for "cat" even though they share no characters. The IDF weighting, when used,
down-weights matches on high-frequency function words ("the," "is") relative to rarer, more
content-bearing tokens, so the score isn't dominated by trivially-matched stopwords.

## Q4: When would you use pointwise LLM-judge scoring instead of pairwise comparison, given that pairwise comparisons are generally more reliable?

Pairwise comparison is the better-anchored, more reproducible judgment for a human or an LLM judge
because it gives a direct point of contrast rather than requiring an internal, drifting absolute
scale — this is well-documented for both human raters and LLM judges. But pairwise comparisons only
ever answer a relative question: "is A better than B," never "is A good enough." You reach for
pointwise scoring specifically when the question that actually needs answering is absolute rather
than relative:

- **A fixed quality/safety gate that a response must clear regardless of what any other system
  produced** — e.g., "is this response acceptable to ship at all," where there is no natural "B" to
  compare against, or where the bar is meant to be stable over time rather than relative to a
  shifting competitor pool.
- **Tracking a metric longitudinally against a constant bar** — e.g., a production dashboard
  tracking mean helpfulness score checkpoint over checkpoint; a pairwise-only setup would require
  re-running comparisons against some fixed anchor every time, which is more awkward than a directly
  trackable absolute score, provided the calibration risk (Section on pointwise scoring in module
  002) is managed via a fixed, well-anchored rubric.
- **Per-response diagnostic sub-scoring** — when you want a structured rubric breakdown (separate
  correctness, helpfulness, and safety sub-scores per response) rather than a single relative winner
  label, because you need to know *which dimension* is weak, not just which of two responses is
  better overall.
- **When there is no natural comparator available at all** — evaluating a single production response
  in isolation, with no second system's output to compare it against in that moment.

In practice, mature pipelines run both: pointwise for absolute gating/tracking, pairwise for
head-to-head model-selection and leaderboard-style decisions, because they answer genuinely
different questions and neither subsumes the other.

## Q5: Explain the mechanism behind LLM-judge position bias, and why a single fixed "always show the stronger model second" correction is not a valid fix.

Position bias is the empirically observed tendency of an LLM judge, in a pairwise comparison, to
favor whichever response occupies a particular slot in the prompt (commonly, though not universally,
the first-shown one), independent of the actual content of the two responses. It's measured directly
by taking the same pair of responses, running the judge twice with the order swapped, and checking
whether the verdict flips — a nontrivial fraction of comparisons do flip purely from order, which is
the operational definition of the bias, and studies evaluating judge reliability (e.g., the
MT-Bench/Chatbot Arena judge-evaluation work) report this at rates large enough to distort a
leaderboard if uncorrected.

A fixed correction ("always put the model we think is stronger in slot 2") is invalid for two
separate reasons. First, it requires already knowing which model is stronger — the entire point of
running the comparison is to find that out, so any correction that presupposes the answer is
circular. Second, and more fundamentally, the bias's direction and magnitude are not a fixed
universal constant — they vary by judge model, by prompt template, and even by the specific pair of
responses being compared (a bias measured on one judge/prompt combination doesn't reliably transfer
to another) — so there is no single correction offset you could bake in even if you wanted to. The
valid fix is symmetric measurement, not a static correction: run every comparison in both orders and
only trust the verdict when it's consistent across the swap (treating position-sensitive comparisons
as ties or inconclusive), which measures and neutralizes the bias's effect on each specific
comparison rather than assuming a fixed direction and magnitude that would need to be re-derived for
every new judge/prompt setup anyway.

## Q6 (coding): Implement a pairwise LLM-judge wrapper that mitigates position bias via order-swapping, and returns both a debiased verdict and an empirical position-bias-rate diagnostic across a batch of comparisons.

```python
from dataclasses import dataclass

@dataclass
class JudgeResult:
    verdict: str          # "A", "B", or "Tie"
    position_sensitive: bool

def debiased_pairwise_judge(judge_fn, instruction: str, resp_a: str, resp_b: str) -> JudgeResult:
    """judge_fn(instruction, first, second) -> 'A' | 'B' | 'Tie', judging 'first' vs 'second'."""
    v1 = judge_fn(instruction, resp_a, resp_b)                  # A shown first
    v2_raw = judge_fn(instruction, resp_b, resp_a)               # B shown first (swapped)
    v2 = {"A": "B", "B": "A", "Tie": "Tie"}[v2_raw]              # remap back to original A/B labels

    if v1 == v2:
        return JudgeResult(verdict=v1, position_sensitive=False)
    return JudgeResult(verdict="Tie", position_sensitive=True)

def batch_position_bias_rate(judge_fn, items: list[tuple[str, str, str]]) -> float:
    """items: list of (instruction, resp_a, resp_b). Returns fraction of comparisons
    whose verdict flipped under order swap -- an empirical position-bias diagnostic."""
    results = [debiased_pairwise_judge(judge_fn, *item) for item in items]
    return sum(r.position_sensitive for r in results) / len(results) if results else 0.0
```

Two things worth calling out to an interviewer: this doubles judge-call cost per comparison, which
is a deliberate and usually worthwhile trade against silently trusting a position-biased verdict;
and `batch_position_bias_rate` is not just a byproduct — it's an actionable judge-quality metric you
should track per judge-model/prompt-template combination over time, since a rising position-bias
rate after, say, a judge-prompt change is a concrete regression signal.

## Q7 (scenario): Your LLM-judge disagrees with human raters on 30% of examples. How do you debug this?

First, get the baseline right before treating 30% as alarming in isolation: measure human-human
agreement on the same item set. If two independent human raters disagree with each other, say, 25%
of the time on this genuinely subjective task, a judge disagreeing with the human majority 30% of
the time is close to "performing about as well as another human rater would" — not a crisis. If
human-human agreement is much higher (say 10% disagreement) and the judge is at 30%, that's a real
gap worth chasing.

Second, don't look at the aggregate number — slice it. Break disagreement down by: (a) task category
(is disagreement concentrated in code/math, where judge competence might be weaker, vs. general
chat, where it's fine), (b) response-length buckets (elevated disagreement specifically when one
response is much longer is a direct signature of residual verbosity bias), (c) closeness of the two
responses (judges and humans are both least reliable on genuinely close calls — if most disagreement
is on near-ties, that's a very different finding than if it's on cases where humans confidently
preferred one response and the judge confidently, wrongly, preferred the other), and (d) which
specific model produced which response (a same-family self-preference pattern would show up here
directly).

Third, read the actual disagreement cases, not just the statistics — pull a sample of the 30% and
read the judge's stated rationale against the human raters' comments. Look specifically for:
position bias (rerun swapped and see if it's a position-sensitive case), verbosity bias (is the
judge-preferred response reliably the longer one with no added substance), self-preference bias (is
the judge-preferred response reliably from the judge's own model family), and rubric mismatch (is
the judge weighing a dimension — e.g., stylistic fluency — that the human rubric explicitly didn't
ask about, or vice versa — sometimes disagreement reflects the judge and the human protocol
genuinely optimizing for different, both-legitimate criteria, which is a specification problem, not
a judge-competence problem).

Fourth, check whether the human protocol itself is the noisy side — if the human-eval task has weak
guidelines or undertrained annotators (module `003`'s calibration discussion), some of the "judge is
wrong" cases may actually be "the human label is unreliable," discoverable by adjudicating a
subsample with a senior/expert reviewer as tie-breaker and seeing which side the expert agrees with
more often.

Fifth, act on what you find: apply the specific mitigation the pattern points to (rubric tightening,
ensembling across judge models to dilute a single-model bias, switching to a different-family judge
if self-preference is implicated) and re-measure agreement on a fresh sample — don't declare the fix
successful based on the same sample used to diagnose the problem, since that risks overfitting the
fix to the specific disagreement cases you happened to inspect.

## Q8: Explain the difference between Cohen's kappa and Fleiss' kappa, and why raw percent agreement is a misleading standalone statistic.

Raw agreement (the fraction of items where raters gave the same label) ignores the agreement that
pure chance would already produce given each rater's own label-frequency habits. If 90% of items in
a dataset are legitimately rated "good" by the underlying quality distribution, two raters who are
each independently guessing "good" 90% of the time and paying no real attention to content would
already agree roughly 81%+ of the time by chance alone — a raw agreement number in that range looks
reassuring but carries almost no information about whether the raters are actually applying a
shared, meaningful standard. Both kappa statistics fix this by subtracting out the chance-expected
agreement and normalizing: `kappa = (observed_agreement - chance_agreement) / (1 -
chance_agreement)`, so `kappa = 0` means the observed agreement is exactly what chance alone (given
each rater's own marginal label distribution) would predict, and `kappa = 1` means perfect agreement
beyond that baseline.

Cohen's kappa is defined for exactly two raters, comparing one specific pair's label sequences and
computing chance agreement from that specific pair's two marginal distributions. Fleiss' kappa
generalizes this to a *panel* of more than two, typically interchangeable, raters per item (common
in production annotation — 3-5 raters per item, with the specific set of raters varying across items
rather than a single fixed pair labeling everything) — instead of one pair's marginals, it uses the
pooled label-frequency distribution across all raters and all items to compute the chance-agreement
baseline, and averages a per-item agreement rate (computed from how many of that item's raters
agreed with each other) across all items. Use Cohen's kappa when you have a fixed pair of raters
labeling the same item set; use Fleiss' kappa when you have a rotating or larger panel of raters per
item, which is the more common real-world crowd-annotation design. Neither statistic natively
distinguishes "close" disagreements from "far" ones on an ordinal scale — for Likert-type data,
weighted kappa or Krippendorff's alpha are the more appropriate tools, since unweighted kappa treats
a 4-vs-5 near-miss identically to a 1-vs-5 outright contradiction.

## Q9 (coding): Implement weighted Cohen's kappa for an ordinal rating scale (e.g., 1-5 Likert), where near-miss disagreements should be penalized less than far disagreements.

```python
import numpy as np

def weighted_cohens_kappa(ratings_1: list[int], ratings_2: list[int],
                            categories: list[int], weight_type: str = "quadratic") -> float:
    """ratings_1, ratings_2: aligned per-item ordinal ratings (e.g., 1..5).
    weight_type: 'linear' or 'quadratic' disagreement penalty, standard for ordinal kappa."""
    k = len(categories)
    cat_index = {c: i for i, c in enumerate(categories)}
    n = len(ratings_1)

    # weight matrix: 0 penalty on the diagonal, growing penalty with distance between categories
    W = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dist = abs(i - j)
            W[i, j] = dist if weight_type == "linear" else dist ** 2

    # observed joint distribution
    O = np.zeros((k, k))
    for r1, r2 in zip(ratings_1, ratings_2):
        O[cat_index[r1], cat_index[r2]] += 1
    O /= n

    # expected joint distribution under independence, from each rater's own marginals
    p1 = O.sum(axis=1)
    p2 = O.sum(axis=0)
    E = np.outer(p1, p2)

    observed_disagreement = (W * O).sum()
    expected_disagreement = (W * E).sum()

    return 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else 1.0
```

The quadratic-weight variant is the standard default for Likert-style scales because it penalizes
large disagreements disproportionately more than small ones (a 1-vs-5 disagreement counts 16x as
much as a 1-vs-2 disagreement under quadratic weights, vs. only 4x under linear weights) — this
generally matches the intuition that a 4-point gap represents a qualitatively more serious rater
disagreement than a 1-point gap, which unweighted kappa (implicitly weight = 1 for any mismatch, 0
for exact match) cannot express at all.

## Q10: What's the practical difference between "detecting contamination" (a Datasets-module concern) and "contamination-aware evaluation design" (this module's concern), and why do you need both?

Contamination detection is a training-data-pipeline problem: given a corpus you're about to train on
(or already trained on), search it for overlap with known benchmark test items — n-gram overlap
search, embedding-based near-duplicate detection — and remove or flag what you find, so the
benchmark score you eventually report is less likely to be inflated by memorization rather than
genuine capability. It's inherently retrospective and best-effort: you can only search for overlap
with benchmarks you know about and can search for, and it can never certify a zero contamination
rate, especially against paraphrased or indirect leakage (a forum thread discussing a benchmark
problem's solution approach without quoting it verbatim) that doesn't produce a detectable string
match.

Contamination-aware evaluation design accepts that residual, undetectable contamination risk as a
permanent background condition and asks a different, forward-looking question: how do I structure my
*evaluation program* — which eval sets I maintain, how I protect them, how often I refresh them — so
that a nonzero, unknown contamination rate on any given public benchmark doesn't silently invalidate
the capability conclusions I'm actually relying on. That's why it reaches for measures
decontamination alone can't provide: private eval sets that were never public in the first place (so
there's nothing to detect-and-remove, because leakage risk is structurally near-zero by
construction), rotation (bounding how long any given eval item has been exposed and exploitable),
and canary strings (a proactive tripwire for detecting if your own benchmark leaked, rather than a
retrospective search through someone else's corpus). You need both because they cover different
failure surfaces: decontamination protects your training runs from being credited with fake gains on
benchmarks you know about; contamination-aware eval design protects your evaluation conclusions from
a leak you may never fully detect, on benchmarks (including your own) that are exposed to the open
web over time.

## Q11: How do canary strings work as a contamination-detection mechanism, and what is the single biggest limitation that keeps them from being a complete solution?

A canary string is a unique, high-entropy marker (often with a recognizable fixed prefix like
"canary GUID <random-string>") that a benchmark's creators embed directly in the benchmark's
published files at release time, accompanied by an explicit public request that anyone building a
web-scale training corpus search for and exclude any document containing it. This serves two
purposes simultaneously: it's a voluntary-compliance courtesy signal that lets good-faith data
curators filter the benchmark out of their pretraining corpora, and it's a detection tripwire for
the benchmark's own maintainers — they can later search public web crawls for the canary string to
see whether the benchmark propagated into corpora that likely fed pretraining, or probe a trained
model by prompting around canary-adjacent context and checking for verbatim regurgitation, which, if
it succeeds, is direct evidence the model's training data contained that text.

The single biggest limitation is that canary-string protection only covers the *specific file* the
canary is embedded in — it does nothing to protect against the much more common leak vector of
people discussing, quoting, or paraphrasing benchmark content elsewhere without carrying the canary
along. A forum post that says "the answer to that tricky MMLU chemistry question about X is Y" leaks
the actual answer information into anything that scrapes that forum, with zero canary string
attached to catch it. Canary strings are best understood as a detection aid for one specific leak
pathway (redistribution of the original file) layered on top of, never as a substitute for, the
harder-to-solve general problem of indirect/paraphrased leakage that this module's private-eval-set
and rotation practices exist to manage instead.

## Q12: Why is scoring only the final output insufficient for evaluating an LLM agent, even when the final output is objectively checkable (e.g., "did the tests pass")?

Because the final state and the process that produced it are not the same object, and each can be
misleading about the other in both directions. A correct final state can result from a badly broken
process — an agent that got lucky on a flaky retry, that guessed a file path correctly without ever
actually reading the relevant code, or that took a destructive, unsafe, or wildly inefficient path
to get there — and crediting that trajectory with full marks overstates the underlying capability
and misses real operational risk (cost, safety, reliability on the next slightly-different task
where the same luck won't repeat) that would be visible in the trajectory but invisible in the
outcome check alone. Conversely, an incorrect final state can result from an almost entirely correct
process that failed only at the very last step — outcome-only scoring cannot distinguish that from a
trajectory that was lost from step one, even though these represent very different capability gaps
and imply very different fixes (a formatting/verification fix vs. a fundamental planning or tool-use
retraining need).

There's also a purely practical argument: outcome-only evaluation gives you a pass/fail label with
zero diagnostic value when the agent fails, which is most of the information you actually need to
improve the system. Step-level or trajectory-level scoring — was the tool selection correct, were
the arguments right, was the ordering sensible, did the agent recover from an error along the way —
is what turns "the agent failed on this task" into an actionable finding about *which* capability
(planning, tool argument construction, error recovery) needs work. Outcome checking remains the
right primary signal for benchmark-scale, cheap, objective pass/fail measurement precisely where a
checkable end state exists, but it should be paired with step-level review, especially on failures,
rather than treated as sufficient on its own.

## Q13 (coding): Implement a scorer that separately measures tool-selection accuracy, argument correctness, and step efficiency for a toy agent trajectory, given a reference "gold" action sequence.

```python
from dataclasses import dataclass

@dataclass
class Action:
    tool: str
    args: dict

def tool_selection_accuracy(pred_actions: list[Action], gold_actions: list[Action]) -> float:
    """Fraction of predicted actions whose tool matches the gold action at the same aligned step."""
    n = min(len(pred_actions), len(gold_actions))
    if n == 0:
        return 0.0
    matches = sum(pred_actions[i].tool == gold_actions[i].tool for i in range(n))
    return matches / n

def argument_correctness(pred_actions: list[Action], gold_actions: list[Action]) -> float:
    """Among steps with a correctly-selected tool, fraction whose args also match the gold args."""
    n = min(len(pred_actions), len(gold_actions))
    correct_tool_steps = [i for i in range(n) if pred_actions[i].tool == gold_actions[i].tool]
    if not correct_tool_steps:
        return 0.0
    correct_args = sum(pred_actions[i].args == gold_actions[i].args for i in correct_tool_steps)
    return correct_args / len(correct_tool_steps)

def step_efficiency(pred_actions: list[Action], gold_actions: list[Action]) -> float:
    """Penalizes trajectories that use far more steps than the gold trajectory needed.
    1.0 if pred used exactly as many (or fewer, capped) steps as gold; degrades otherwise."""
    if not gold_actions:
        return 1.0 if not pred_actions else 0.0
    return min(1.0, len(gold_actions) / max(len(pred_actions), 1))

def score_trajectory(pred_actions: list[Action], gold_actions: list[Action]) -> dict:
    return {
        "tool_selection_accuracy": tool_selection_accuracy(pred_actions, gold_actions),
        "argument_correctness": argument_correctness(pred_actions, gold_actions),
        "step_efficiency": step_efficiency(pred_actions, gold_actions),
    }
```

The important design point to raise unprompted: this alignment-by-position approach (comparing step
`i` to gold step `i`) is a simplification that only works when there's a single canonical gold
trajectory and the predicted trajectory follows a comparable structure — it breaks down immediately
for open-ended tasks with multiple valid strategies (module `005`, Section 5), where
position-aligned comparison against one fixed gold sequence would wrongly penalize an equally-valid
alternative approach that legitimately does things in a different order. A more robust real
implementation would use set-based or DAG-based matching (did the necessary actions happen in a
valid dependency order, regardless of exact position) or fall back to rubric-based LLM-judge scoring
of the whole trajectory for tasks where no single gold sequence is defensible.

## Q14: What makes red-teaming a fundamentally different evaluation activity from standard capability benchmarking, beyond just "it's about safety"?

The core difference is in what each activity samples and what a result means. Capability
benchmarking samples the *expected* task distribution — realistic prompts a typical user would
actually give — and reports an aggregate or average performance number across that distribution; a
low score means "the model isn't reliably good at this yet," and the finding's importance scales
roughly with how much of the expected traffic it affects. Red-teaming deliberately samples the
*adversarial tail* — prompts specifically engineered to be maximally likely to break the model's
safety behavior, which a typical user would essentially never write (unusual roleplay framing,
multi-turn context-building, encoded/obfuscated requests, prompts targeting a specific known
weakness) — and a single successful, reproducible elicitation of a serious harm matters on its own,
independent of how rare that exact prompt pattern is in real traffic, because the harm model is "can
this be made to happen at all," not "how often does this happen on average."

This changes the evaluation's entire design logic: capability eval wants representative coverage and
cares about statistical aggregate performance (which is exactly why module `007`'s
confidence-interval and sample-size machinery matters for it); red-teaming wants adversarial,
non-representative coverage and treats a rare-but-severe worst case as more important than the
average case. It also changes the activity's temporal nature — a capability benchmark score is a
reasonably stable snapshot, while red-teaming is adaptive and adversarial by construction (it's
explicitly trying to find whatever current defenses don't cover), so a red-team finding has a shelf
life and the practice has to keep evolving as defenses evolve, rather than being a one-time
measurement exercise the way a benchmark run mostly is.

## Q15: Name three distinct automated adversarial-prompt-generation techniques and explain what each is actually optimizing for.

**Model-generated adversarial prompts (LLM-red-teams-LLM)**: a separate attacker LLM is prompted or
fine-tuned specifically to produce prompts designed to elicit a target harm category, optionally
iterating based on the target model's observed responses. This optimizes for *scalable exploration
within a known attack-strategy space* — it's essentially automating what a human red-teamer would
manually try, at far higher throughput, and can be run in an iterative refine-based-on-response loop
that a static prompt list can't.

**Gradient-/search-based token optimization (e.g., GCG-style greedy coordinate gradient search)**:
for models with gradient access, this directly searches over adversarial token sequences (typically
a suffix appended to an otherwise-refused request) to maximize the log-probability of a compliant,
harmful continuation. This optimizes for *finding attacks a human would never think to write* — the
resulting strings are often nonsensical-looking token sequences with no semantic content a human
red-teamer would generate by reasoning about the task, discovered purely by gradient-guided search
over the model's own loss landscape, and are notable for frequently transferring across different
target models.

**Genetic/evolutionary mutation-and-selection search**: starting from a seed set of known attack
strategies, automated mutation operators generate variants, and only variants that increase attack
success rate against the target are kept for the next generation. This optimizes for *mapping the
boundary and combinatorial space around a known attack category* — it's the right tool once you
already know roughly what kind of attack works (a fictional-framing jailbreak, say) and want to
exhaustively characterize which specific phrasings, topics, and parameter combinations within that
category succeed, which is directly useful input to a category-level (not just instance-level)
training fix. None of the three substitutes for human red-teamers discovering genuinely novel attack
*categories* in the first place — automation multiplies coverage within and around a category,
humans have historically been the primary source of new categories.

## Q16 (coding): Implement a nonparametric bootstrap confidence interval for a benchmark accuracy score, and use it to determine whether a 500-item eval set is large enough to distinguish a 2-point score difference.

```python
import numpy as np

def bootstrap_ci(per_item_correct: np.ndarray, n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_item_correct)
    point_estimate = per_item_correct.mean()
    boot_means = np.array([
        per_item_correct[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)
    ])
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return point_estimate, lower, upper

def required_n_for_margin(p: float, margin: float, z: float = 1.96) -> int:
    """Normal-approximation sample size for a desired 95%-CI half-width ('margin') at proportion p."""
    return int(np.ceil((z**2) * p * (1 - p) / margin**2))
```

Plugging in numbers: at `p = 0.5` (the conservative, maximum-variance case) and a desired half-width
of 1 percentage point (`margin = 0.01`), `required_n_for_margin` returns roughly 9,604 items — far
more than the 500-item set in question. At `n = 500` and `p = 0.5`, the actual 95% CI half-width is
`1.96 * sqrt(0.25/500) ≈ 4.4` percentage points, meaning any two models whose scores differ by less
than roughly that margin on this eval set are not reliably distinguishable from noise on sample-size
grounds alone — a reported 2-point gap on a 500-item benchmark should be treated with real
skepticism until backed by a proper paired significance test (McNemar's, since this is same-item
binary outcome data) rather than accepted at face value just because 2 points sounds like a real, if
modest, difference.

## Q17: Explain why you should use McNemar's test rather than a two-sample (independent) proportion test when comparing two models' accuracy on the same benchmark items, and what specifically goes wrong if you don't.

A two-sample proportion test (e.g., a two-proportion z-test) assumes the two accuracy estimates come
from independent samples — it treats "model A got 84% right on its set of items" and "model B got
86% right on its set of items" as if they were measured on two unrelated draws from the population.
When both models are actually evaluated on the exact same fixed set of items (the standard setup for
any head-to-head benchmark comparison), that independence assumption is simply false, and using the
independent-samples test throws away real information: item-level difficulty is correlated across
models — some items are hard for essentially every model and some are easy for essentially every
model — so the two models' error patterns on the same item set are correlated, not independent.

Ignoring that correlation and using an independent-samples test produces an overly conservative
(too-wide) estimate of the variance of the score difference, understating statistical power — you
can end up concluding "not significant" for a difference that a correctly paired test would have
identified as significant, because the independent test's implicit variance estimate includes
item-difficulty variance that a paired test correctly cancels out by only looking at where the two
models actually disagree. McNemar's test fixes this by conditioning only on the discordant pairs
(items where the two models' correctness disagrees) and testing whether that disagreement is
symmetric between "A right, B wrong" and "A wrong, B right" — it directly uses the pairing structure
that the independent test discards, which is both the more statistically correct model of how the
data was actually generated and, in practice, the more powerful test for detecting a real difference
at a given sample size.

## Q18 (scenario): A benchmark table shows Model A scoring 84.3% and Model B scoring 83.1% on a 500-item benchmark, and the paper claims Model A is better. How do you evaluate this claim?

First, treat the raw 1.2-point gap as a hypothesis to test, not a settled fact — go straight to the
sample-size arithmetic: at `p ≈ 0.83-0.84` and `n = 500`, the 95% CI half-width on either individual
score alone (using the Wilson or normal approximation) is roughly `1.96 * sqrt(0.84*0.16/500) ≈ 3.2`
percentage points, which already tells you a 1.2-point gap is well within the noise band of a single
score's own uncertainty, before even doing a proper paired comparison.

Second, ask whether the two models were actually evaluated on the *same* 500 items under the *same*
prompt template, decoding settings, and scoring harness — if the paper computed Model B's number
from a different source (a self-reported number from Model B's own paper, produced under a different
harness or prompt template), this isn't a valid head-to-head comparison at all regardless of
statistical significance, since different harnesses/templates alone can move scores by several
points (this module, Section 1.3-1.4) — the comparison needs both models run through one controlled,
shared evaluation pipeline before a difference means anything.

Third, assuming it is a genuine same-item, same-harness comparison, run McNemar's test on the paired
right/wrong outcomes (or a paired bootstrap if a non-binary score is involved) rather than
eyeballing the two point estimates — this is the correctly-powered test for exactly this
paired-binary-outcome situation, and it's entirely possible for a 1.2-point gap on 500 items to come
back non-significant, in which case the paper's "Model A is better" claim is not supported by this
benchmark at this sample size, regardless of the direction of the point estimate.

Fourth, check whether this is one comparison among many in the paper (a multi-benchmark table) — if
so, apply a multiple-comparisons correction before treating any individual benchmark's "win" as
meaningful, since testing significance independently across many benchmarks inflates the chance of
at least one spurious "significant" result.

Fifth, sanity-check for contamination and reporting practices — is either model's score suspiciously
close to ceiling or suspiciously inconsistent with its performance on a related private/held-out
eval (module `004`), and does the paper disclose decoding temperature, number of samples per item,
and whether either number is a best-of-N figure being presented without disclosing N (a common,
serious reporting pitfall). Only after all of this would a staff-level reviewer be willing to say
whether "Model A is better" is a supported claim or an artifact of noise, harness mismatch, or
reporting ambiguity.

## Q19: Why is conflating a best-of-N (or pass@k) number with single-attempt (pass@1) performance one of the more consequential statistical pitfalls in published LLM evaluation, and how would you catch it as a reviewer?

Best-of-N and pass@1 measure genuinely different capabilities and have very different deployment
implications. Pass@1 asks "if the model gets exactly one attempt, what fraction of the time does it
succeed" — the relevant number for most real deployment settings, where a user or downstream system
typically can't afford, or doesn't have the infrastructure for, dozens of parallel attempts with a
verifier to pick the best one. Best-of-N asks "if the model gets N independent attempts and we have
some way to identify (or are told) which one is correct, what fraction of the time is at least one
of the N correct" — a much easier bar to clear, since it only requires the model to be capable of
producing a correct answer at least once in N tries, not reliably on a single try, and N can be
pushed arbitrarily high to push the number arbitrarily close to 100% for tasks with any nonzero
per-attempt success probability at all (if per-attempt success probability is `p`, the probability
at least one of N independent attempts succeeds is `1 - (1-p)^N`, which approaches 1 as N grows even
for small `p`).

Reporting a best-of-100 number as "94% accuracy" in a headline table, without prominently disclosing
that N=100 samples were used and some oracle/verifier selected the best one, creates a badly
misleading impression of the model's actual single-shot reliability — a reader who assumes this is
pass@1 performance will substantially overestimate how the model will behave in a setting where only
one attempt is affordable. As a reviewer, the concrete checks are: does the methodology section
state the exact decoding procedure and number of samples per item used to produce every headline
number; if a pass@k metric is reported, is `k` stated next to every instance of the number, not just
once in a methods footnote; and does the paper report pass@1 *alongside* any best-of-N figure it
wants to headline, so a reader can see the actual gap between single-shot and best-of-N reliability
rather than only seeing the more flattering number. The absence of a clearly disclosed N next to any
at-k metric is itself the red flag worth flagging, independent of what the actual number turns out
to be.

## Q20 (scenario): You're asked to design the evaluation program for a new coding-agent product from scratch, before it ships. Walk through your approach across the methodology areas in this module.

I'd start by separating the evaluation program into layers rather than one monolithic "eval,"
because different questions need different tools. For fast, high-frequency iteration signal during
development, I'd rely on cheap automatic/programmatic checks wherever the environment supports them
— does the generated patch make the target hidden test suite pass, does the code execute without
error, does a static-analysis check flag an issue — because these are objective, deterministic, and
cheap enough to run on every checkpoint (module `001`'s point that automatic metrics remain the
right tool for regression-testing even though they can't judge holistic quality).

For anything the programmatic checks can't see — was the *process* reasonable, not just the outcome;
was an unsafe or destructive action taken along the way; was a failed attempt close to correct or
wildly off — I'd build trajectory-level evaluation (module `005`): step-level scoring of tool
selection, argument correctness, ordering/efficiency, and error recovery, initially via careful
human review on a sample of trajectories (especially failures) to establish what "good" actually
looks like for this specific agent's task domain, then layering in an LLM-judge-based step scorer
once I have a human-labeled reference set to validate that judge against (module `002`, Section 4) —
I would not trust a trajectory judge in this domain out of the box, since step-level agent judging
is a harder, less-validated problem than single-turn response judging.

For model-vs-model and version-vs-version comparison decisions (should we ship this new checkpoint),
I'd use pairwise comparison — either human preference collection (module `003`) for pre-launch,
high-stakes go/no-go decisions, or a validated LLM judge for faster internal iteration — with
position-bias mitigation and, importantly, a judge from a different model family than the agent
itself if the underlying model is from our own lineage, to avoid self-preference bias contaminating
the comparison.

Before trusting any of this operationally, I'd validate the LLM-judge and trajectory-scoring
pipelines against real human judgments on a held-out sample (module `002`, Section 4), specifically
checking agreement broken down by task category and failure type, not just an aggregate number, and
I'd compare that agreement against a measured human-human agreement baseline (module `003`) rather
than judging it against an arbitrary absolute threshold.

For eval-set integrity, I'd maintain a private, never-published eval set of originally-authored
coding tasks (module `004`) precisely because coding benchmarks are exactly the kind of content that
leaks onto GitHub/StackOverflow/forums easily, rotate a meaningful fraction of it periodically, and
cross-check any suspiciously strong public-benchmark result against this private set before trusting
it.

I'd run a red-teaming pass (module `006`) specifically targeting this product's unique risk surface
— an agent that can execute code and call tools has a materially different and more severe risk
profile than a pure chat model (arbitrary code execution, destructive file operations, unauthorized
external actions), so I'd want both internal red-teaming integrated into the dev loop and, given the
severity of what a coding agent can actually do in a real environment, at least one external review
before a general release, with findings feeding back into both training data and, for anything
severe and unmitigated, real authority to delay the ship date.

Finally, every comparative claim this program produces — "the new version's task success rate
improved" — would go through the statistical rigor discipline in module `007`: paired significance
testing against the same task set, confidence intervals reported alongside point estimates, and
explicit sample-size justification for the eval sets involved, given that trajectory-level,
human-reviewed evals in particular tend to be small and expensive, and a small eval set's apparent
improvement is exactly the kind of number most likely to be noise if I don't check.

