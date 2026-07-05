# Automatic Metrics and Their Limits

## 0. Framing: what "automatic metric" means here

An automatic metric is a function that takes model output (and usually a reference or a small set of
references) and returns a number, with no human or model-based judgment in the loop. This module is
about the specific family of automatic metrics that dominated NLP evaluation before general-purpose
instruction-following LLMs existed: exact match / F1, BLEU, ROUGE, embedding-overlap metrics like
BERTScore, and perplexity. All five are still in active use — they are cheap, deterministic,
reproducible, and require no additional model calls — but every one of them was designed for a
narrower evaluation problem than "is this a good response from a general-purpose chat model," and
understanding *precisely* where each one breaks is what motivates the rest of this module
(LLM-as-judge and human evaluation).

The unifying failure mode to watch for across all five: each metric operationalizes "quality" as a
specific, computable proxy — token overlap with a reference, embedding similarity to a reference, or
likelihood under a training distribution — and each proxy was a reasonable stand-in for quality only
within the narrow task distribution it was built for (single-span QA, single-reference MT,
single-reference summarization, next-token prediction on a fixed corpus). Open-ended generation —
"write an email," "explain this proof," "refactor this function" — has no single correct string,
frequently no small enumerable set of correct strings, and a notion of quality (helpfulness,
correctness, tone, safety) that is not a function of string overlap with anything at all. That
mismatch is not a minor calibration issue; it is why every metric in this module has documented
cases of confidently ranking a worse response above a better one.

## 1. Exact Match and F1 (extractive QA)

### 1.1 Mechanics

Exact Match (EM) and token-level F1 were popularized by SQuAD (Rajpurkar et al., 2016) for
extractive question answering, where the model's job is to output a contiguous span of the input
passage that answers the question. Because the answer is defined as *a substring of a given
passage*, the space of valid answers is small and mostly enumerable (a handful of human-annotated
acceptable spans per question).

- **EM**: binary, 1 if the model's predicted string is identical (after a fixed normalization —
  lowercasing, stripping punctuation and articles) to one of the reference answers, else 0. Averaged
  over the eval set.
- **F1**: treats the predicted answer and each reference answer as bags of tokens (after the same
  normalization) and computes the harmonic mean of token-level precision and recall against the
  best-matching reference:

```python
def token_f1(pred: str, ref: str) -> float:
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()
    common = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def squad_f1(pred: str, refs: list[str]) -> float:
    return max(token_f1(pred, ref) for ref in refs)  # max over references
```

F1 is strictly more forgiving than EM: "Barack Obama" vs. reference "Obama" scores EM = 0 but F1 >
0, because "Obama" is a token subset. Taking the max over multiple human references handles some
legitimate answer variation (e.g., "1945" vs. "in 1945").

### 1.2 Why this is a reasonable metric for its original task, and why it stops being one

EM/F1 work well specifically because extractive QA has a **closed, low-cardinality answer space
anchored to a fixed passage** — the task designer can enumerate essentially all correct answers
ahead of time, and the correctness criterion really is "did you extract the right substring," which
token overlap approximates well. The metric is measuring almost exactly what the task is defined to
be measuring.

The failure mode appears as soon as the task shifts to **free-form or abstractive** answer
generation, which is the norm for modern LLM-based QA (open-domain QA where the model composes an
answer from its own knowledge, no passage constraint):

- **Paraphrase blindness in the strict sense is less severe than for BLEU/ROUGE** (F1's
  bag-of-tokens recall does credit partial lexical overlap), but it still fails whenever a correct
  answer uses none of the reference's tokens — "he passed away in 1945" vs. reference "died in 1945"
  loses credit for the correct token "died."
- **No handling of open-ended correct answers with many phrasings**: EM/F1 assume a small reference
  set. Ask an LLM "explain why the sky is blue" and there are effectively infinite correct
  phrasings; you cannot enumerate references, so span-overlap metrics simply do not apply to most of
  what a modern LLM is asked to do.
- **Reward for verbosity is absent but reward for precise phrasing is brittle**: a technically
  correct answer that adds context ("Obama, the 44th U.S. president") gets penalized on precision
  even though it is not wrong.

This is why EM/F1 survive today almost exclusively for benchmarks that retain the original
closed-answer-space structure: span-extraction QA, multiple-choice-style tasks recast as string
matching, and code/math tasks where the "answer" is a canonical short string (a number, an
identifier) susceptible to exact-match or a light normalization. See `..\06_Benchmarks` for how
specific benchmarks (e.g., GSM8K's final-numeric-answer matching) use EM in ways that remain valid
because the benchmark was designed around a canonical answer.

## 2. BLEU: n-gram precision for generation

### 2.1 Mechanics

BLEU (Papineni et al., 2002) was designed to automate machine-translation evaluation by scoring how
much a candidate translation's n-grams overlap with one or more human reference translations. It is
fundamentally a **precision-oriented** metric: what fraction of the n-grams the model produced also
appear in the reference(s)?

Step by step:

1. **Modified (clipped) n-gram precision.** For each n-gram order `n` (BLEU-4 uses `n = 1..4`),
   count how many n-grams in the candidate also occur in the reference, but **clip** each n-gram's
   count at the maximum number of times it occurs in any single reference. Clipping exists
   specifically to stop a candidate from gaming the score by repeating a matched n-gram ("the the
   the the" would otherwise get free "the" credit four times against a reference containing "the"
   once).

```python
def modified_precision(candidate: list[str], references: list[list[str]], n: int) -> float:
    cand_ngrams = ngram_counts(candidate, n)
    max_ref_ngrams = collections.Counter()
    for ref in references:
        ref_ngrams = ngram_counts(ref, n)
        for ng, c in ref_ngrams.items():
            max_ref_ngrams[ng] = max(max_ref_ngrams[ng], c)
    clipped = sum(min(c, max_ref_ngrams[ng]) for ng, c in cand_ngrams.items())
    total = sum(cand_ngrams.values())
    return clipped / total if total else 0.0
```

2. **Geometric mean across n-gram orders.** `p_n` for `n = 1, 2, 3, 4` are combined as a geometric
   mean, typically with uniform weights `w_n = 1/4`:

```
BLEU = BP * exp( sum_{n=1}^{4} w_n * log(p_n) )
```

3. **Brevity penalty (BP).** Because precision alone rewards short outputs (a one-word candidate
   that happens to match is "100% precise"), BLEU multiplies by a penalty that punishes candidates
   shorter than the reference:

```
BP = 1                                   if c > r
BP = exp(1 - r/c)                        if c <= r
```

where `c` is candidate length and `r` is (effective) reference length.

BLEU is typically computed **corpus-level** (aggregate n-gram counts over the whole test set before
taking ratios), not by averaging per-sentence BLEU, because per-sentence BLEU is degenerate for
short sentences (a single 4-gram miss can zero out the geometric mean via `log(0)`, usually patched
with smoothing in practice).

### 2.2 Why BLEU correlates poorly with human judgment for open-ended generation

BLEU's design assumptions hold reasonably well for the task it was built for — short-sentence
machine translation with a handful of professionally produced reference translations, where "close
to the reference" and "correct" are nearly synonymous because translation has comparatively low
surface-form freedom. Every one of those assumptions breaks for open-ended LLM generation:

- **No semantic credit, only surface-string credit.** "The cat sat on the mat" and "A feline rested
  on the rug" share almost no n-grams and get a near-zero BLEU score against each other despite
  being near-paraphrases. BLEU cannot distinguish "different words, same meaning" from "different
  words, different (wrong) meaning" — it penalizes both identically. This is the single largest
  reason BLEU fails on open-ended generation: a correct, fluent, well-reasoned answer phrased
  differently than the reference scores the same as an equally-different but *wrong* answer.
- **Reference scarcity makes the problem worse as task openness increases.** Translation typically
  has a handful of valid renderings; open-ended tasks (summarize this, explain this, write this) can
  have effectively unbounded valid outputs. BLEU against one or a few references systematically
  underscores every correct answer that isn't lexically close to *those particular* references, and
  the metric has no way to know it's doing so.
- **No notion of factual correctness, logical validity, instruction adherence, or safety.** A
  fluent, reference-overlapping continuation that is factually wrong, or that ignores an explicit
  instruction in the prompt (wrong output format, wrong language, ignored constraint) can score well
  on BLEU as long as it borrows the reference's vocabulary. BLEU literally cannot detect failures
  that are orthogonal to n-gram overlap, which is most of what actually matters for judging an LLM
  response.
- **Gameable and unstable at the sentence level.** Because clipping only prevents *exact* repetition
  abuse, models optimized directly against BLEU-like signals learn to produce generic,
  high-frequency n-grams that have a good chance of appearing in *some* reference, rather than
  specific, informative content — a well-documented failure mode from the machine-translation and,
  later, RL-against-automatic-metric literature.
- **Length sensitivity via the brevity penalty is a blunt, task-specific fix** that does not
  generalize; for tasks where a shorter, more precise answer is genuinely better (e.g., a direct
  factual answer vs. a padded, hedge-y one), BLEU's own brevity penalty pushes the wrong direction.

The empirical literature on this (e.g., correlation studies run alongside every WMT metrics shared
task, and later NLG-specific studies such as Liu et al. 2016 "How NOT to Evaluate Your Dialogue
System") consistently finds BLEU's sentence-level correlation with human quality judgments to be
weak-to-moderate for translation and close to uninformative for open-ended dialogue/generation — in
some dialogue studies, BLEU-style metrics show near-zero or even *negative* correlation with human
ratings, because generic, safe, low-information responses ("I don't know," "That's interesting")
tend to have decent n-gram overlap with a diverse reference set purely by being common, while
genuinely good, specific responses diverge lexically from any one reference and score worse.

## 3. ROUGE: recall-oriented overlap for summarization

### 3.1 Mechanics

ROUGE (Lin, 2004) is BLEU's sibling for summarization, differing mainly in orientation: where BLEU
asks "how much of what the candidate produced is validated by the reference" (precision), ROUGE's
canonical variants ask "how much of the reference did the candidate manage to reproduce" (recall) —
appropriate because a good summary is expected to *cover* the reference's content, not merely avoid
saying anything unsupported.

- **ROUGE-N**: n-gram recall. For n-gram order `n`:

```
ROUGE-N = ( sum over reference n-grams that also appear in candidate, clipped )
          / ( total n-grams in reference )
```

structurally the mirror image of BLEU's modified precision, with candidate and reference roles
swapped and denominator based on reference length instead of candidate length. In practice most
modern reporting uses the **F-measure** of ROUGE precision and recall (both directions computed,
then harmonic-meaned), not pure recall, but the metric's design center of gravity remains
recall-oriented — that's what "ROUGE" (Recall-Oriented Understudy for Gisting Evaluation) names.

- **ROUGE-L**: based on the **longest common subsequence (LCS)** between candidate and reference
  token sequences (not necessarily contiguous, unlike n-grams), rewarding in-order overlap without
  requiring exact adjacency:

```python
def lcs_length(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l_f1(candidate: list[str], reference: list[str], beta: float = 1.2) -> float:
    lcs = lcs_length(candidate, reference)
    r = lcs / len(reference) if reference else 0.0
    p = lcs / len(candidate) if candidate else 0.0
    if p == 0 and r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (r + beta**2 * p)
```

- **ROUGE-S / ROUGE-SU**: skip-bigram variants allowing gaps between the two words of a "bigram,"
  giving partial credit for word pairs that appear in the right relative order but not adjacently.

### 3.2 Why ROUGE fails for the same underlying reason as BLEU, with its own specific twists

- **Same core defect as BLEU**: it is a surface n-gram / subsequence overlap statistic, so it
  inherits every paraphrase-blindness and factuality-blindness problem described in Section 2.2. A
  summary that correctly captures the source's meaning in different words scores low; a summary that
  copies reference-adjacent phrasing while subtly misrepresenting a number or a causal relationship
  scores fine.
- **Recall orientation creates a length-inflation incentive.** Because ROUGE-N/L reward covering
  more of the reference's tokens, longer candidates that simply include more content have a
  structural advantage — a well-known ROUGE gaming pattern is that near-extractive,
  longer-than-necessary summaries with padding tend to score better than tight, well-compressed
  ones, which is close to the opposite of what a good summary should optimize for. This is precisely
  why summarization systems tuned to maximize ROUGE historically converged on **extractive or
  near-extractive** behavior (copying source sentences nearly verbatim) rather than genuinely
  abstractive rewriting — copied spans guarantee high n-gram/LCS overlap with a reference that
  itself was often lightly derived from the source.
- **Single or few references, unbounded valid summaries.** As with BLEU, a good abstractive summary
  has many valid phrasings and even many valid *content selections* (which facts to include) when
  the source is long; ROUGE against 1-4 references cannot distinguish "omitted a fact the reference
  happened to include" from "omitted a fact that didn't matter," and cannot reward including a
  correct, salient fact that the reference simply didn't mention.
- **No factual-consistency signal.** This is the most consequential gap for LLM-generated summaries
  specifically: ROUGE has no mechanism to detect hallucinated content that is *not* in the source,
  because it only compares candidate to reference, never candidate to source. A summary can score
  well on ROUGE while introducing a fabricated claim, as long as enough of its other tokens overlap
  with the reference — factual-consistency evaluation for summarization became its own subfield
  (e.g., QA-based and NLI-based factuality metrics) precisely because ROUGE cannot see this failure
  mode at all.

## 4. Embedding-similarity metrics: BERTScore and friends

### 4.1 Mechanics

BERTScore (Zhang et al., 2020) and similar embedding-based metrics were built to fix BLEU/ROUGE's
most obvious defect — paraphrase blindness — by replacing discrete token-identity matching with
**continuous similarity in a contextual embedding space**.

1. Encode both candidate and reference token sequences with a pretrained contextual encoder
   (originally BERT; the mechanism generalizes to any encoder), producing one embedding vector per
   token, contextualized by the surrounding sentence.
2. For each candidate token, find its **greedy best match** in the reference by cosine similarity
   (and symmetrically for each reference token against the candidate) — this is a soft, continuous
   analogue of the discrete "does this token appear in the reference" test that BLEU/ROUGE use.

```python
def bertscore(cand_embs: np.ndarray, ref_embs: np.ndarray) -> tuple[float, float, float]:
    # cand_embs: [n_cand, d], ref_embs: [n_ref, d], both L2-normalized
    sim = cand_embs @ ref_embs.T          # [n_cand, n_ref] cosine similarities
    precision = sim.max(axis=1).mean()    # each candidate token's best reference match
    recall = sim.max(axis=0).mean()       # each reference token's best candidate match
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
```

3. Optionally weight each token's contribution by **inverse document frequency (IDF)** computed over
   a large corpus, so that matching a rare, informative token counts for more than matching "the" or
   "is" — a soft analogue of content-word emphasis.

Because matching happens in embedding space rather than by string identity, "cat" and "feline," or
"purchase" and "buy," can register as near-matches even though they share no characters — this is
the specific defect BERTScore targets and does meaningfully improve on relative to BLEU/ROUGE.
Empirically, BERTScore correlates better with human judgment than BLEU/ROUGE on several benchmark
correlation studies, particularly for tasks with high legitimate lexical diversity (paraphrase-heavy
generation, some MT settings).

### 4.2 Why embedding-similarity metrics still fail for modern LLM evaluation

Fixing paraphrase-blindness does not fix the deeper problem, and BERTScore-style metrics inherit or
introduce their own failure modes:

- **Still fundamentally reference-anchored overlap, just in a softer space.** The metric still asks
  "how similar is this to a specific reference string," which remains the wrong question whenever
  there are many valid, mutually dissimilar-in-embedding-space correct answers (different but
  equally valid arguments, different but equally valid code implementations, different but equally
  valid creative writing). Embedding similarity narrows the paraphrase gap but does not remove the
  "no single reference captures the correct-answer set" gap.
- **No factuality or logical-validity check.** Two sentences can be highly similar in embedding
  space while one is true and the other is false ("the trial succeeded" vs. "the trial failed" are
  contextually similar surface constructions and can embed close together depending on the encoder,
  despite being opposite claims). Embedding similarity captures topical/semantic *relatedness*, not
  truth-preservation or logical correctness — a model that produces a fluent, on-topic,
  embedding-similar but factually wrong answer is not reliably penalized.
- **Sensitive to the choice of underlying encoder**, in ways that are opaque to a metric consumer:
  BERTScore computed with different backbone models gives different absolute numbers and can rank
  the same pair of candidates differently, which undermines cross-paper/cross-study comparability
  unless the encoder is held fixed and disclosed.
- **Gameable by fluent, generic, on-topic text.** Because the metric rewards embedding proximity
  rather than informativeness, a bland, hedge-y, topically-adjacent response can score deceptively
  well against a reference, similar to BLEU's "generic response" gaming pattern but in embedding
  space rather than n-gram space — safe, vague completions tend to sit centrally in embedding space
  near many possible references.
- **No instruction-following, formatting, or task-adherence signal.** As with BLEU/ROUGE, none of
  these metrics see the prompt — they compare output to reference, not output to the task the prompt
  actually specified. A response that is semantically on-topic but ignores an explicit constraint in
  the instruction (wrong format, wrong length, answered a different question than the one asked) is
  invisible to a reference-comparison metric of any kind, embedding-based or not.
- **Cannot evaluate open-ended tasks with no natural reference at all** — most agentic and
  long-horizon tasks (see `005_Agentic_And_Trajectory_Evaluation.md`) have no single "reference
  trajectory" to embed and compare against in the first place.

The practical upshot: BERTScore-family metrics are a genuine improvement over n-gram overlap for
tasks where paraphrase variance is the dominant source of false negatives (e.g., some MT and
semantic-similarity settings), but they do not close — and were never designed to close — the gap
that matters most for evaluating modern instruction-following LLMs: judging correctness,
helpfulness, and task adherence rather than reference resemblance.

## 5. Perplexity: a training-time proxy that does not transfer to quality

### 5.1 Mechanics

Perplexity is a direct transform of the training loss, not an independent metric. For an
autoregressive LM assigning probability `P(x_t | x_{<t})` to each token, the cross-entropy loss over
a sequence of length `T` is:

```
H = -(1/T) * sum_{t=1}^{T} log2 P(x_t | x_<t)      (bits per token, base-2 log)
```

Perplexity is the exponentiated cross-entropy — informally, "the effective branching factor" the
model faces at each step, i.e., the size of a uniform distribution that would have produced the same
average log-loss:

```
PPL = 2^H   =  exp( -(1/T) * sum_t ln P(x_t | x_<t) )     (using natural log)
```

A perplexity of 20 means the model is, on average, about as uncertain at each step as if it had to
choose uniformly among 20 equally likely tokens; lower perplexity means the model assigns higher
probability, on average, to the tokens that actually occurred in the held-out text. Perplexity is
computed on a fixed held-out corpus under teacher forcing (the model is fed the true previous tokens
at every step, never its own generations), which makes it a measure of **one-step next-token
predictive fit to a reference distribution**, not a measure of generation quality under the model's
own autoregressive sampling.

### 5.2 Why perplexity does not reliably predict downstream task quality or human preference

Perplexity is genuinely useful for what it was built for: tracking optimization progress and
convergence during pretraining, comparing checkpoints of the *same* model/tokenizer over training
time, and diagnosing training instabilities (loss spikes, divergence). It systematically fails as a
proxy for quality once you move past that narrow use:

- **Tokenizer-dependence makes cross-model perplexity comparisons invalid without care.** Perplexity
  is measured in "surprise per token," and different models use different tokenizers with different
  average bytes-per-token. A model with a larger, more efficient vocabulary can have lower
  perplexity purely because each of its tokens covers more text, with no difference in actual
  predictive quality per unit of text. Meaningful cross-model comparison requires normalizing to a
  common unit (e.g., bits per byte/character) — comparing raw per-token perplexity across models
  with different tokenizers is a well-known and easy-to-make error.
- **It measures fit to a *reference text distribution*, and that distribution is not "quality."** A
  model minimizes perplexity by matching the statistics of its training/eval corpus as closely as
  possible — including that corpus's noise, verbosity, repetition, and stylistic idiosyncrasies. A
  model that has been instruction-tuned and RLHF'd to be concise, direct, and refuse harmful
  requests will typically show *higher* perplexity on a held-out slice of raw internet text than a
  base model trained purely to imitate that same text, precisely because the aligned model's output
  distribution has been deliberately shifted away from raw-web statistics — yet the aligned model is
  the one humans overwhelmingly prefer in practice (this is the empirically documented tension
  behind results like InstructGPT, where alignment training measurably degrades some
  likelihood-based metrics on pretraining-like held-out data while dramatically increasing human
  preference win-rate). Perplexity on a static corpus and human-judged quality are simply different
  axes; nothing in the definition of perplexity ties it to helpfulness, correctness, or preference,
  and post-training explicitly optimizes for the latter at some cost to the former.
- **Teacher-forced one-step prediction does not measure free-running generation quality.**
  Perplexity evaluates the model's probability of the *correct* next token given a *correct*
  history. It says nothing about what happens when the model must condition on its own previous
  (possibly imperfect) generations over a long sequence — exposure bias, compounding errors, and
  degenerate repetition loops in free-running sampling are invisible to a teacher-forced metric.
- **No task-structure awareness.** Two models with identical held-out perplexity can differ
  substantially in downstream task accuracy (instruction following, reasoning, code correctness),
  because perplexity averages predictive quality uniformly over every token in a corpus, while task
  performance often depends disproportionately on getting a small number of high-stakes tokens right
  (the final numeric answer in a math problem, the correct function name in code, the
  polarity-flipping word in a sentiment judgment). A model can have excellent perplexity overall
  while being reliably wrong on exactly the tokens that determine task correctness.
- **Says nothing about safety, factuality, or preference ordering between two candidate
  generations**, which is the actual object of interest for comparing modern instruction-tuned
  models against each other.

Perplexity therefore remains the right tool for a narrow, specific job — monitoring pretraining
health and comparing checkpoints within one fixed model/tokenizer/data setup — and the wrong tool
for the question "is this a better model to deploy," which is exactly the question the rest of this
module's methods exist to answer.

## 6. The common failure pattern, and the gap it created

Lay all four families side by side and the shared structure is clear: EM/F1, BLEU, ROUGE, and
BERTScore are all **reference-comparison** metrics (they need a "correct" string or small set of
strings to compare against), differing only in how forgiving the comparison function is (exact
string identity, clipped n-gram overlap, LCS, embedding cosine similarity). Perplexity is not
reference-comparison but is a **distributional fit** metric with the same underlying limitation one
level up: it measures agreement with a reference *distribution* instead of a reference *string*, but
still never asks "was this response actually good."

None of the five metrics in this module can, even in principle, evaluate:

- Correctness/factuality independent of surface resemblance to a reference.
- Instruction-following and constraint satisfaction (the prompt is not part of the computation at
  all for BLEU/ROUGE/BERTScore/EM-F1).
- Reasoning validity, multi-step coherence, or task-appropriate tone.
- Safety, appropriateness, or preference between two responses that are both plausible surface
  strings.
- Anything in tasks with no well-defined reference at all — most agentic, creative, and long-horizon
  tasks (Section 5 of `005_Agentic_And_Trajectory_Evaluation.md`).

This is precisely the gap that made **LLM-as-judge** (`002_LLM_As_Judge_Methodology_And_Biases.md`)
and **human evaluation** (`003_Human_Evaluation_And_Preference_Collection.md`) necessary as the
primary evaluation paradigms for modern general-purpose LLMs: both replace "compare to a fixed
reference string via a mechanical similarity function" with "have a competent judge (human or model)
assess the response directly, against the actual instruction, using holistic judgment that can
integrate correctness, helpfulness, safety, and style simultaneously." That flexibility is exactly
what buys LLM-as-judge and human eval their validity — and exactly what introduces the very
different failure modes (inconsistency, cost, and judge bias) that the next two modules cover in
depth.

## 7. When these metrics are still the right tool

A staff-level treatment of this topic should not conclude "these metrics are useless" — that
overstates the case and is not how they are actually used in a mature eval stack:

- **Regression testing and CI-style checks during training.** Automatic metrics are cheap and
  deterministic, so they are the right tool for high-frequency, low-stakes signals: did this
  checkpoint's ROUGE on a fixed validation summarization set drop by 5 points relative to
  yesterday's run? That is a useful, fast tripwire even though ROUGE cannot tell you the model is
  *good*, only that something changed. Perplexity curves are the standard, correct tool for
  monitoring optimization health during pretraining.
- **Closed-answer-space tasks retain valid EM/F1 use** — extractive QA, canonical-answer math/code
  benchmarks, multiple-choice-as-string-matching — precisely because those tasks were designed so
  that string-overlap correctness tracking really is correctness tracking.
- **As one signal among several in a broader eval suite**, not as the sole arbiter — e.g., reporting
  ROUGE alongside a factual-consistency check and a small human-eval sample for a summarization
  system is a reasonable, cost-aware design, where automatic metrics catch gross regressions cheaply
  and the more expensive human/LLM-judge signal is reserved for final quality claims.
- **As components inside LLM-as-judge pipelines themselves.** A judge prompt can legitimately use
  exact-match against a known ground truth as a sub-check for parts of a response that do have a
  canonical answer (e.g., "does the final numeric answer match X"), combining the
  cheapness/determinism of automatic metrics with the holistic judgment LLM-as-judge provides for
  everything else.

The discipline this module is trying to instill: know exactly which assumption a metric depends on
(closed answer space, single-reference sufficiency, distributional-fit-equals-quality) before
trusting a leaderboard number that reports it, and be able to name concretely — not just gesture at
— the kind of error the metric is structurally blind to.

## Cross-references

- Named benchmarks that use these metrics operationally (SQuAD's EM/F1, WMT's BLEU, CNN/DailyMail's
  ROUGE, GSM8K's exact-match) are covered in `..\06_Benchmarks`; this module covers the measurement
  methodology, not the specific benchmark.
- The judge-based paradigm that fills the gap identified in Section 6 is covered in
  `002_LLM_As_Judge_Methodology_And_Biases.md`.
- Human evaluation, the other side of that gap, is covered in
  `003_Human_Evaluation_And_Preference_Collection.md`.
- Statistical treatment of any of these metrics' scores (confidence intervals, sample-size
  requirements) is covered in `007_Statistical_Rigor_In_LLM_Evaluation.md`.

