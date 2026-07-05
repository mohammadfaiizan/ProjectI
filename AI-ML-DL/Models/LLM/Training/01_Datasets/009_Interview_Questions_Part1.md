# Interview Questions — Part 1

Twenty staff-level questions covering pretraining data sources, cleaning/deduplication, mixture weighting, tokenizer construction, and contamination detection, drawn from this module's content files (`001` through `005`). Questions mix conceptual explanation, design/debugging scenarios, and coding exercises. See `010_Interview_Questions_Part2.md` for the remaining twenty, covering synthetic data, post-training datasets, and licensing/governance, plus additional questions on the topics covered here.

## Q1: Every disclosed pretraining mixture — GPT-3, Llama 1, Llama 3, DeepSeek-V3 — is majority web text by raw token count, even though every one of these labs explicitly says curated sources like Wikipedia and books are higher quality per token. Why doesn't any of them just train on the curated sources alone?

The short answer is that there simply isn't enough curated text to fill a modern training budget, and the gap isn't a close call — it's multiple orders of magnitude. English Wikipedia, fully stripped of markup, is on the order of a few billion tokens (GPT-3's own Table 2.2 puts its Wikipedia slice at roughly 3 billion tokens). arXiv's full text after LaTeX is stripped is in the tens of billions. StackExchange, even across its largest constituent sites, tops out in the tens of billions. Digitized books are the largest curated category — GPT-3's Books1 and Books2 slices are 12 billion and 55 billion tokens respectively — but book supply is capped by how many books exist and are digitized and accessible at all, a ceiling that doesn't move as fast as compute budgets do. Sum every curated source generously and you land somewhere around a few hundred billion tokens at the absolute most.

Modern pretraining runs consume multiple trillions of tokens: Llama 1 used 1.0–1.4 trillion, Llama 3's initial release used over 15 trillion, DeepSeek-V3 used 14.8 trillion. There is no way to cover that gap by just repeating the curated pool more times, either — data-constrained scaling-law work (Muennighoff et al.) shows that returns from repeating a fixed corpus decay noticeably beyond roughly four epochs, and past that point you're better off, compute for compute, mixing in additional unique data even if it's noisier than continuing to repeat clean data. So "just repeat the good stuff" isn't a real substitute for scale.

The deeper point, though, is that this isn't just a scale-filler problem — web text contains registers that curated sources structurally cannot contain at all: casual conversational writing, product reviews, forum arguments, informal multilingual text, slang, opinion. Wikipedia's encyclopedic register, arXiv's technical register, and a book corpus's narrative register never cover any of that. So the trade-off isn't "noisy-and-big versus clean-and-small, pick one" — it's that the two source types are complementary along genuinely different axes (scale and register diversity on one side, density and reliability on the other), and every lab's answer has converged on "use both, and tune the mixing ratio" rather than either extreme. That convergence, across labs with very different overall strategies, is itself informative.

## Q2: You're comparing GPT-3's, Llama 1's, Llama 3's, and DeepSeek-V3's public data disclosures and you notice DeepSeek-V3 gives almost no per-source breakdown of its 14.8T-token corpus, while its report is unusually detailed about training cost and FP8 infrastructure. How would you interpret that asymmetry, and would you treat it as a red flag?

I'd resist the temptation to read too much into it in either direction, but I also wouldn't ignore the pattern, because it recurs across the field, not just at DeepSeek. Line up the disclosures: GPT-3 published a full five-source table with weights (Table 2.2). Llama 1 published a full seven-source percentage table with per-source filtering methodology. Llama 3, on ten times Llama 1's token budget, dropped to a qualitative description — more code, more multilingual, an annealing phase — with no itemized table. DeepSeek-V3 and Qwen2.5 disclose token counts and a one-sentence characterization of composition, and nothing more granular. So the trend, across four labs and roughly five years, is toward less granular data disclosure even as token budgets grew by more than an order of magnitude — which is the opposite of what you'd naively expect from a "field maturing" narrative.

What makes this not simply damning is that the same labs disclosing the least about data are, in DeepSeek's case specifically, disclosing unusually more about infrastructure and cost than most competitors — DeepSeek's report breaks out GPU-hours by training phase and gives a dollar-cost estimate few other labs have matched. That tells me disclosure isn't a single dial a lab turns up or down uniformly; it's a set of independent decisions, and data composition seems to be treated as more sensitive, or genuinely harder to summarize cleanly at this scale and diversity, than compute infrastructure is. Plausible reasons include competitive sensitivity (mixture recipes are hard-won IP), the genuine complexity of a multi-stage, many-source pipeline that no longer reduces to a clean table, and — impossible to rule out without more information — legal caution around exactly which sources are used given the active copyright litigation landscape.

I would not call the absence of a table a smoking gun on its own. I would treat it as exactly what it is: an absence of information, which means any specific claim about DeepSeek-V3's composition beyond "large, multilingual, math/code-enriched" is invented, not reported. The right move as an evaluator is to be explicit about that boundary rather than filling the silence with a plausible-sounding guess, and to weight benchmark results accordingly — a lab that won't show its data can't be given the benefit of the doubt on contamination or memorization concerns the way a lab with a fully itemized, auditable mixture can.

## Q3: Walk through why a quality-filtering pipeline typically runs both a fastText-style classifier and a perplexity-based filter rather than just picking the more accurate of the two.

They're not two implementations of the same idea with one better than the other — they measure genuinely different things and fail in different, non-overlapping ways, which is exactly why running both catches more bad content than either alone.

The classifier is a distributional/stylistic signal. It's trained as a binary classifier where the positive class is curated reference text (Wikipedia-linked pages, WebText-style Reddit-curated text) and the negative class is raw, unlabeled Common Crawl, and the shallow architecture — hashed n-gram features averaged into one vector, one linear layer, softmax — is chosen deliberately for speed at web scale rather than accuracy ceiling. What it's actually learning is which surface n-gram patterns correlate with "resembles curated text," which is a proxy, not a direct quality signal. It can be fooled by a stylistically polished but content-free page that mimics encyclopedic phrasing, and it has essentially no mechanism for catching degenerate repetition, since a repeated boilerplate phrase can easily contain n-grams that look encyclopedic on their own.

Perplexity, by contrast, has no notion at all of "does this resemble the kind of text a human curator selected." It scores a document under a small reference LM trained on clean text and measures how surprised that model is by the token sequence. This catches a different failure mode entirely — gibberish, OCR garbage, HTML remnants serialized as text — because those produce high perplexity. But it also has its own blind spot in the opposite direction: an unusually low perplexity can signal pathological repetition rather than quality, since a reference LM assigns near-zero surprise to a sequence that's trivially predictable from repeating itself, and a classifier-only or perplexity-only-upper-bound pipeline would let that straight through as some of the "most confident" text in the corpus.

So the practical shape is a funnel: cheap heuristics first (Section 2 of the cleaning file — boilerplate detection, line-length, symbol ratio, language ID) to remove the obviously bad majority cheaply, then the classifier and perplexity filter run as complementary signals — sometimes independently, sometimes combined — over what survives, before the fuzzy-dedup pass. Using both is how you cover "resembles curated register" and "is locally fluent and non-degenerate" as two separate axes, neither of which subsumes the other.

## Q4: Implement a MinHash signature function and an estimator for Jaccard similarity between two documents' shingle sets, and explain why the estimator is unbiased.

```python
import numpy as np

def shingle(text: str, k: int = 5) -> set:
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}

def make_hash_functions(num_hashes: int, seed: int = 0):
    """Universal hash family h(x) = (a*x + b) mod p, used to approximate
    independent random hash functions cheaply."""
    rng = np.random.default_rng(seed)
    p = (1 << 61) - 1  # Mersenne prime
    a = rng.integers(1, p, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, p, size=num_hashes, dtype=np.int64)
    return a, b, p

def minhash_signature(shingles: set, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    if not shingles:
        return np.full(len(a), p, dtype=np.int64)
    shingle_ints = np.array([hash(s) & p for s in shingles], dtype=np.int64)
    hashed = (a[None, :] * shingle_ints[:, None] + b[None, :]) % p
    return hashed.min(axis=0)  # one min per hash function

def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))

# Usage
a, b, p = make_hash_functions(num_hashes=128)
doc1 = shingle("the quick brown fox jumps over the lazy dog", k=3)
doc2 = shingle("the quick brown fox leaps over a lazy dog", k=3)
sig1 = minhash_signature(doc1, a, b, p)
sig2 = minhash_signature(doc2, a, b, p)
print(estimate_jaccard(sig1, sig2))  # approximates true Jaccard(doc1, doc2)
```

The unbiasedness argument is the important part to be able to state, not just the code. Think of a single hash function `h` as inducing a uniformly random ordering — a random permutation — over the union of the two shingle sets `A ∪ B`. `minhash_h(A)` is, by definition, whichever element of `A` happens to come first in that ordering restricted to `A`; same for `B`. The two minima can only be equal if a single element that lies in both sets — an element of `A ∩ B` — is the overall-first element of the entire union under this random ordering, because if the true first element of `A ∪ B` belonged only to `A`, then `A`'s minimum would be that element while `B`'s minimum would be something later and different, so they couldn't match, and symmetrically for `B`. Since the ordering is uniform over all `|A ∪ B|` elements, each element is equally likely to be first, so the probability the first element specifically falls in `A ∩ B` is exactly `|A ∩ B| / |A ∪ B|` — the Jaccard similarity. That makes a single hash function's match indicator an unbiased Bernoulli estimator of `J(A,B)`. A single trial is noisy (variance `J(1-J)`), which is exactly why the signature uses many independent hash functions and averages the agreement fraction — by the standard variance-of-an-average argument, that reduces variance by a factor of the number of hash functions, trading compute for precision.

## Q5: Given MinHash signatures for a large document collection, comparing every pair directly is O(n²) and infeasible at corpus scale. Implement LSH banding to generate candidate near-duplicate pairs sub-quadratically, and explain how to choose the number of bands and rows per band.

```python
from collections import defaultdict

def lsh_band_signature(signature, num_bands: int) -> list:
    rows_per_band = len(signature) // num_bands
    return [
        hash(tuple(signature[i * rows_per_band:(i + 1) * rows_per_band].tolist()))
        for i in range(num_bands)
    ]

def build_lsh_candidates(doc_ids, signatures, num_bands: int) -> set:
    buckets = defaultdict(list)
    for doc_id, sig in zip(doc_ids, signatures):
        for band_idx, band_hash in enumerate(lsh_band_signature(sig, num_bands)):
            buckets[(band_idx, band_hash)].append(doc_id)

    candidates = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                candidates.add(tuple(sorted((bucket_docs[i], bucket_docs[j]))))
    return candidates

def candidate_probability(s: float, b: int, r: int) -> float:
    """P(two docs with true Jaccard similarity s collide in >=1 band)."""
    return 1 - (1 - s ** r) ** b
```

This only produces candidates — pairs that share at least one band bucket — not final answers; the actual near-duplicate decision still runs the exact (or MinHash-estimated) Jaccard check only on this much smaller candidate set, which is where the sub-quadratic saving actually comes from: as long as true near-duplicates are a small fraction of all pairs (the normal regime for a web corpus), bucket sizes stay small and the total candidate count and hashing work scale far better than `n²`.

Choosing `b` (bands) and `r` (rows per band), with total hash count `H = b·r` fixed by your compute/memory budget, is about placing the S-curve `P(candidate | s) = 1 - (1 - s^r)^b` so its steep transition sits near your target similarity threshold. Increasing `r` makes each band's collision probability `s^r` fall off faster as `s` drops, sharpening the low end and pushing the knee's threshold higher — fewer false positives, but you need a higher true similarity to reliably collide. Increasing `b` gives more independent chances to collide, pushing the threshold lower and making the transition sharper from the other direction — better recall, more candidates to verify. You pick a factorization of `H` that puts the knee near, say, 0.8–0.9 similarity, which is a commonly cited operating point for web-corpus near-dedup, then verify empirically that your false-positive/false-negative rates at that setting match your tolerance, since `H` is usually fixed by budget and the s-curve math only tells you where the knee lands for a given factorization, not which factorization is "correct" in the abstract.

## Q6: A teammate built a document-level fuzzy-dedup pipeline (whole-document MinHash/LSH) and reports the corpus is "fully deduplicated," but the trained model still shows a surprisingly high rate of memorizing a specific boilerplate legal disclaimer verbatim. What's going wrong, and how would you fix it?

Document-level fuzzy dedup only flags a pair if the *aggregate* similarity of two entire documents crosses threshold. A boilerplate disclaimer embedded in thousands of otherwise-distinct documents — different articles, different sites, different surrounding content — never triggers that check, because each containing document, taken as a whole, is unique; the shared 200-word paragraph is diluted by everything around it into a low overall Jaccard similarity. So "fully deduplicated" is true and also not the relevant claim here: document-level dedup was never designed to catch sub-document, cross-document repetition of a fixed passage, and reporting it as "fully deduplicated" without that caveat is exactly the kind of overclaim that misleads a downstream memorization investigation.

The fix is to add a lower-granularity pass targeted specifically at this failure mode, not to try to push the existing document-level threshold down (which would start incorrectly flagging documents that only share a short common phrase for entirely benign reasons). The standard approach — used, for instance, in CCNet-style pipelines behind the Llama data lineage — is paragraph-level (or line-level) exact or near-duplicate detection: hash each paragraph independently, and if a specific paragraph recurs verbatim above some frequency threshold across the corpus, strip just that paragraph out of every containing document rather than discarding the whole document. A frequency threshold matters here specifically — you don't want to strip every recurrence of a legitimately-common short phrase (a section header, a standard code-license header, a frequently-quoted line) on a first-seen-wins basis, only content that recurs often enough to be genuine boilerplate rather than coincidental legitimate repetition.

I'd also want to directly verify the fix worked rather than assume it: rerun the memorization probe (prompting the model with a short prefix of the disclaimer and checking for verbatim continuation) after the paragraph-level pass, and separately check the disclaimer's duplicate count in the corpus before and after, since the actual causal claim — "this specific duplication caused this specific memorization" — is falsifiable and worth actually falsifying rather than accepting on the strength of a plausible mechanism alone.

## Q7: Derive precisely what happens to temperature-based mixture sampling weights as T→1 and as T→∞, starting from q_i(T) = p_i^(1/T) / Σ_j p_j^(1/T).

At `T=1`, the exponent `1/T` is exactly 1, so `q_i(1) = p_i^1 / Σ_j p_j^1 = p_i / Σ_j p_j = p_i`, since the raw proportions already sum to 1. This recovers exactly proportional sampling — no reweighting at all — which is the sanity-check case any correct implementation needs to reproduce.

For `T → ∞`, the clean way to see the limit is to rewrite the formula in log-space: `q_i(T) = exp((1/T) log p_i) / Σ_j exp((1/T) log p_j)`, which is a softmax over the log-proportions with inverse temperature `1/T`. As `T → ∞`, `1/T → 0`, so every exponent `(1/T) log p_i → 0` — and this holds regardless of how negative `log p_i` is, i.e., regardless of how small `p_i` is, because you're multiplying a fixed finite number by something going to zero. Since every numerator term converges to `exp(0) = 1`, and all `k` terms converge to the same value, the softmax necessarily converges to the uniform distribution `1/k` for every domain. The important nuance to state precisely: it's not that the domains "become similar in size" — it's that the log-proportions, which is the quantity the exponent actually operates on, get scaled toward zero uniformly, erasing the very information that differentiated the domains in the first place. A domain that was 0.1% of the raw corpus and a domain that was 70% both end up at exactly `1/k` in the limit, with no residual trace of the original 700x gap.

Worth flagging as a common error: some sources write this as `q_i ∝ p_i^alpha` with `alpha = 1/T` directly as the exponent, in which case `alpha < 1` is the flattening-toward-uniform direction and `alpha > 1` is the sharpening direction — the opposite-labeled regimes from the `T` convention used here. There's no universally agreed convention across the literature, so the only defensible practice is to state your exponent convention explicitly every time and never assume a paper's abstract uses the same one you do — silently assuming the wrong direction describes the exact opposite of the intended effect, which is a real, not merely pedantic, failure mode in this area.

## Q8: Implement a numerically stable temperature_sample_weights function, and explain why you shouldn't just compute p_i ** (1/T) directly in floating point.

```python
import math

def temperature_sample_weights(raw_proportions: dict, T: float) -> dict:
    """
    q_i(T) = p_i^(1/T) / sum_j p_j^(1/T)
    T=1 -> proportional sampling. T->inf -> uniform sampling. T<1 -> sharpens
    toward the already-largest domain(s).
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    total = sum(raw_proportions.values())
    if total <= 0:
        raise ValueError("proportions/counts must sum to a positive value")
    p = {k: v / total for k, v in raw_proportions.items()}

    inv_T = 1.0 / T
    log_terms = {k: inv_T * math.log(v) for k, v in p.items() if v > 0}
    max_log = max(log_terms.values())          # softmax stability trick
    exp_terms = {k: math.exp(lt - max_log) for k, lt in log_terms.items()}
    denom = sum(exp_terms.values())
    return {k: e / denom for k, e in exp_terms.items()}

raw = {"web": 0.700, "books": 0.200, "code": 0.080, "math": 0.020}
for T in (1.0, 2.0, 4.0, 10.0, 50.0):
    print(T, temperature_sample_weights(raw, T))
```

The reason to avoid computing `p_i ** (1/T)` directly is a straightforward floating-point underflow problem, and it bites exactly in the regime this function is meant to be useful in. For a domain with a genuinely small raw proportion (say `p_i = 0.0002`) and a large `1/T` (a strongly sharpening temperature), `p_i ** (1/T)` can underflow to exactly `0.0` in double precision well before the renormalization step, silently zeroing out that domain's weight entirely rather than giving it a small-but-nonzero share — which is a real correctness bug, not just a precision nuisance, since a mixture-weighting pipeline that silently drops a domain to exactly zero is functionally deciding to never sample it again, a much stronger and probably unintended claim than "give it a small share." Working in log-space — `(1/T) * log(p_i)` — and applying the standard softmax max-shift trick before exponentiating keeps every computation in a numerically well-behaved range regardless of how small `p_i` or how extreme `1/T` gets, and it's the version you'd actually want in a real data pipeline rather than the mathematically-equivalent-but-fragile direct-exponentiation version.

## Q9: The mixture-weighting literature reports that including code in the pretraining mix improves performance on non-code reasoning tasks. Give the leading mechanistic hypotheses for this, and explain why you should present this as an open question rather than a settled fact.

There are three structural hypotheses commonly offered, plus one important confound worth taking seriously rather than dismissing. First, code enforces unusually strict, unambiguous, long-range-dependent structure — a variable bound on one line must be correctly resolved possibly hundreds of tokens later, function signatures must match call sites, control flow must nest correctly — and learning to predict code plausibly requires something like symbolic state-tracking that could share representational machinery with multi-step natural-language reasoning, which also requires tracking intermediate quantities and dependencies across a chain. Second, code corpora are unusually rich in natural-language-paired-with-formal-structure — comments and docstrings describing, at varying formality, what an adjacent verifiable structure is doing — which is a comparatively rare pattern outside code (proofs-with-exposition are a partial analogue at much smaller scale) and plausibly trains a useful "informal reasoning grounded in checkable structure" mode. Third, code has an almost total absence of the ambiguity and rhetorical indirection endemic to web prose, which might nudge the model's default prediction mode toward something more precise and literal, useful wherever precision matters more than fluent approximation.

The confound worth taking seriously: code-heavy corpora are sourced from places like GitHub and Stack Overflow that are, independent of the content being code, more curated — passed through some community quality filter like upvotes or code review or working tested examples — than the median random web page. If code sources are simply higher average quality by whatever metric matters for downstream reasoning, for reasons that have nothing intrinsically to do with code's formal properties, some or all of the reported effect could be a quality confound rather than anything code-specific. Disentangling this cleanly would need a genuinely controlled comparison — code versus similarly-curated non-code text from the same platforms, or real code versus surface-scrambled fake code that preserves statistics but breaks compositional consistency — and to my knowledge no published study has run that full disambiguating design at frontier scale with public results. The right calibrated position: the empirical pattern (code-inclusive mixtures beating code-free ones at matched token count on some reasoning benchmarks) is reasonably well replicated and close to consensus practice; the causal mechanism is genuinely open, and presenting any single explanation as settled fact overstates what's actually known.

## Q10: A colleague proposes an elaborate easy-to-hard curriculum for an upcoming pretraining run, citing classical curriculum-learning results. How do you respond, and what would you actually recommend instead?

I'd separate two claims that get conflated under the single word "curriculum," because the evidence for them is very different. The first is fine-grained ordering within an otherwise stationary, fixed mixture — deciding the sequence in which examples from a fixed set of domain proportions are shown. The second is deliberately making the mixture itself non-stationary over time, most concretely a late-training shift to a smaller, more curated mixture with a decaying learning rate — commonly called an annealing or cooldown phase.

For the first claim, there's a real structural argument for why order shouldn't matter much, and it's worth being able to state precisely rather than just asserting "it probably doesn't matter." Each training step draws an approximately i.i.d. mini-batch from the stationary mixture distribution. Provided the mixture proportions don't change, permuting the order in which specific documents are drawn doesn't change the marginal distribution of what's been seen by any given point in training — the expected composition at step t is the same regardless of the specific sequence. What changes is the exact sequence of gradient updates, but SGD is already built to be robust to considerable per-step noise; that's the entire premise of mini-batching in the first place. There isn't strong evidence that the optimizer's eventual trajectory is highly sensitive to fine-grained order of an otherwise i.i.d. stream, as opposed to being primarily sensitive to the marginal distribution and the learning-rate schedule.

For the second claim, the evidence is genuinely different and considerably stronger — a distinguishable final annealing/cooldown phase combining a decaying learning rate with an upweighted curated mixture shows up, in varying detail, across several independently published pretraining recipes, and the mechanism (a decaying learning rate makes late-training updates harder to overwrite by later, smaller updates, so deliberately controlling what occupies that high-leverage window is a real lever) is coherent and distinct from the classical curriculum-learning story.

So my actual recommendation would be: don't invest engineering effort in an elaborate example-by-example easy-to-hard ordering across the bulk of training — the evidentiary case for it at this scale and objective is thin, and the SGD argument gives a principled reason to expect it wouldn't pay off. Do invest in a well-designed annealing phase near the end of the run, since that's the one curriculum-adjacent practice with real, broad adoption evidence, and it's a fundamentally different claim (deliberate non-stationarity) than "ordering matters."

## Q11: Explain, mechanically, why byte-level BPE tokenizers cannot produce an out-of-vocabulary token, and why this matters relative to word-level or character-level tokenization.

The base vocabulary in byte-level BPE is the 256 possible byte values, and this is populated before any merge is ever learned — vocabulary index 0 through 255 exist by construction, covering every possible byte. Any input text, in any language, script, or encoding — including malformed Unicode, emoji, mixed scripts within one line, or genuinely binary-looking sequences — is first decomposed into its UTF-8 byte sequence, and every single one of those bytes is already a valid vocabulary entry. The learned BPE merges then greedily combine frequent adjacent symbols into longer units, but the merge table is purely an efficiency layer on top of a base representation that is already total — there is no input that fails to decompose into some sequence of already-known tokens, in the absolute worst case a long run of single-byte tokens.

Contrast this with a word-level or character-level tokenizer built from a fixed, finite vocabulary decided at training time: any token not in that vocabulary at encoding time has no representation and must be mapped to a special `<UNK>` token. This creates two real problems. First, it's a genuine information loss — the model literally cannot distinguish two different unknown inputs that both collapse to the same `<UNK>` token, so whatever those inputs actually were is gone by the time the model sees them. Second, it creates a train/inference distribution mismatch if the frequency or nature of unknown tokens differs between training and deployment — a model that saw relatively few `<UNK>`s during training but encounters them more often on live traffic (new proper nouns, code-mixed text, a script underrepresented in training) is operating outside its trained distribution in a way that's specifically hard to characterize, since `<UNK>` is a single symbol standing in for an unbounded and shifting set of actual inputs.

Byte-level BPE trades this correctness problem for a pure compression-efficiency problem: a rare or novel sequence still gets represented completely and losslessly, just less efficiently — as more, shorter tokens rather than fewer, longer ones. That's a real cost (more tokens per unit of content, discussed at length in the vocabulary-size trade-off), but it's a degradation in efficiency, not a representational failure, which is a categorically better failure mode to have at the tokenizer layer of a production system.

## Q12: The naive BPE training loop recomputes pair frequencies over the entire corpus after every single merge, which is O(vocab_size × corpus_size) — infeasible at trillion-token scale. Complete the missing piece: after popping the highest-count pair from a max-heap and performing the merge, what specifically needs to be updated, and why is it cheap?

```python
import heapq
from collections import Counter, defaultdict

def train_bpe_incremental(word_freqs: dict, vocab_size: int):
    corpus = {tuple(word): freq for word, freq in word_freqs.items()}
    vocab = set(sym for word in corpus for sym in word)

    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []

    while len(vocab) < vocab_size and heap:
        neg_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg_count or pair_counts[pair] <= 0:
            continue  # stale entry -- lazy deletion, just discard and keep popping

        merged_symbol = pair[0] + pair[1]
        vocab.add(merged_symbol)
        merges.append(pair)

        # --- The piece to fill in: only touch words containing this pair ---
        for word in list(pair_to_words.get(pair, ())):
            freq = corpus.pop(word, None)
            if freq is None:
                continue
            # Remove this word's old pair-count contributions.
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] -= freq
                pair_to_words[p].discard(word)
            # Rebuild the word with the merge applied.
            new_word, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(merged_symbol)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            corpus[new_word] = corpus.get(new_word, 0) + freq
            # Add this word's new pair-count contributions, and push updates.
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(new_word)
                heapq.heappush(heap, (-pair_counts[p], p))

    return vocab, merges
```

The key insight is that a merge is a local edit, so its effects on pair counts are local too. Only pairs adjacent to tokens that were just merged change count as a result of that specific merge — every pair elsewhere in the corpus, in words that never contained the merged pair, is completely unaffected and doesn't need to be touched at all. So the update step iterates only over `pair_to_words[pair]` — the set of distinct words actually containing the just-merged pair — decrements the old pair-count contributions those words made, rewrites each word with the merge applied, and adds the new pair-count contributions from the rewritten word, pushing each updated count back onto the heap. This converts the total cost from re-scanning the whole corpus on every merge to a cost proportional to the (typically small, roughly constant-size) neighborhood actually touched by each merge, plus one unavoidable full-corpus pass at the very start to build the initial counts. The heap can accumulate stale entries whose true count changed since they were pushed — handled here via lazy deletion, verifying the popped count against the authoritative `pair_counts` value and discarding and re-popping if it doesn't match, rather than trying to eagerly keep the heap perfectly synchronized on every single update, which would itself be expensive.

## Q13: A model with d_model=4096 moves from a 32K to a 128K vocabulary. Quantify the actual parameter and compute cost of this change, and explain why the calculus differs for an 8B model versus a 405B model.

The embedding table (and, if untied, the separate unembedding/LM-head matrix) is a `V × d_model` matrix, so parameter count scales linearly in `V`. At `d_model = 4096`: 32K vocab gives `32,000 × 4096 ≈ 131M` parameters for one such matrix; 128K gives `128,256 × 4096 ≈ 525M` — roughly a 4x increase, tracking the roughly 4x vocabulary increase, as expected. If the unembedding is untied from the embedding, that's two such matrices, so the total goes from roughly 262M to roughly 1.05B parameters.

Against an 8B-parameter model, 1.05B is a bit over 13% of total parameters — noticeable, though the 32 transformer layers' worth of attention and FFN weights still dominate. Scale the same comparison to a 405B model at `d_model = 16,384` (Llama 3.1's dimension at that size): the same 128,256 × 16,384 unembedding matrix is about 2.1B parameters against a 405B total — roughly 0.5%. The embedding/unembedding cost is fixed in absolute terms (depends only on `V` and `d_model`, not on depth or training volume), while the transformer body's parameter count grows with model size — so as models get larger, the same absolute vocabulary cost becomes a shrinking relative cost.

There's also a distinct, recurring per-token compute cost that doesn't shrink the same way: the final projection costs roughly `2 × d_model × V` FLOPs per token (a standard matmul FLOPs count), which at `d_model=4096, V=128,256` is about 1.05 GFLOPs per token, against a rough `2 × N_params ≈ 16` GFLOPs total forward-pass estimate for an 8B model — the unembedding alone is roughly 6–7% of per-token compute at that scale, and critically this is a fixed tax paid identically on every token regardless of how many transformer layers sit in front of it, so it doesn't shrink in relative terms as depth grows the way the parameter-count fraction does.

Working against both of these costs is the compensating benefit that a larger vocabulary shortens the tokenized sequence for the same underlying text — more common words and phrases collapse into single tokens — which cuts attention compute roughly quadratically in the resulting sequence-length reduction during training and cuts sequential decode steps linearly during inference, and that benefit compounds across every token of every future training run and every inference request. That compounding is exactly why the field's vocabulary sizes have grown steadily larger release over release even though the raw embedding-cost arithmetic was just as true in 2018 as it is now — what changed is the volume on the other side of the trade, not the arithmetic itself. The one place this reverses is small or edge-deployed models, where a 128K-vocabulary embedding pair at modest `d_model` can be a genuinely non-trivial fraction of total parameters, making a smaller vocabulary a rational choice for that deployment target specifically.

## Q14: How would you measure the multilingual tokenizer fairness problem rigorously, and what levers would you use to reduce it, without changing what languages the model is actually pretrained on?

The measurement is fully offline and doesn't require running the model at all, which is exactly what makes it a rigorous, reproducible claim rather than an impressionistic one: take a fixed, comparable-content corpus across many languages — ideally a professionally-translated parallel corpus, so content is held constant by construction — tokenize each language's text with the tokenizer under evaluation, and report bytes-per-token (or tokens-per-fixed-unit-of-content) broken out by language. Languages the tokenizer's fitting corpus represented well cluster at high bytes-per-token (efficient); underrepresented languages cluster low (inefficient, more tokens for the same content). This is exactly the benchmark that produces the commonly cited 2–4x fragmentation-ratio figures between well- and poorly-represented languages, and it's also the right tool to validate any mitigation — rerun the same measurement after a change and check whether the gap between the best- and worst-served languages actually shrank, not just whether some aggregate average improved while the tail stayed flat.

For mitigation without touching the pretraining language mixture itself, the key structural fact to exploit is that fitting a tokenizer requires only a comparatively modest, frequency-stabilized sample, not a full pass over the pretraining corpus — BPE merge decisions are driven by pair-frequency statistics that are heavy-tailed and stabilize quickly, a fundamentally lower sample-complexity task than what the language model itself needs from data. That means the tokenizer-fitting corpus's language mixture can be decoupled entirely from the pretraining corpus's language mixture: deliberately construct a more linguistically-balanced corpus specifically for tokenizer fitting, upweighting underrepresented scripts well beyond their natural web-text share, without having to make the same trade-off in the actual pretraining data (which is governed by a separate question — how much capability the model should acquire in each language). A second lever is an explicit minimum merge-budget allocation per language or script, layered on top of frequency-driven selection as a fairness constraint rather than trusting pure frequency ranking to produce an equitable outcome even on a rebalanced corpus. A third, blunter lever is simply choosing a larger overall vocabulary, since merge-slot competition is zero-sum only up to the vocabulary's total size — more total slots give both high- and low-resource languages room to be adequately covered without competing as directly, which is part of why Llama 3's jump to 128,256 tokens is framed partly in multilingual-coverage terms and not purely English-compression terms.

## Q15: Why is the standard n-gram window for contamination detection around 8–13 tokens rather than something much shorter or much longer?

It's a bias-variance trade-off over what counts as "the same passage," and both directions of getting it wrong have real costs. Too short — matching on individual words or 3–4-word phrases — and the false-positive rate becomes enormous, because extremely common short phrases ("the results show that," "according to the study") recur constantly across completely unrelated documents purely from limited local entropy in natural language. A check at that granularity would flag a large fraction of the entire corpus as contaminated against almost any benchmark, which destroys the signal's usefulness — you can't afford to discard that much otherwise-clean data on a signal carrying almost no information about actual leakage.

Too long — matching on whole paragraphs or documents — and the detector becomes too strict in the other direction. A training document that only scraped part of a benchmark item (a forum post quoting just the question stem, a solutions page reproducing only the answer while paraphrasing the question) shares no single long exact span with the original, so the check silently passes it through as clean even though real leakage occurred.

A window around 8–13 tokens — 13 is the figure most often cited — sits in the useful middle: long enough that an exact match happening twice by pure chance in unrelated text is vanishingly unlikely (treating each token position as roughly an independent draw from a large effective vocabulary, the probability of an exact 13-token coincidental match is on the order of `(1/V)^13`, astronomically small even for a conservative small V), so a single match is strong evidence of actual copying rather than coincidence and you don't need to raise the bar to multiple matches to stay reliable; but short enough to be satisfied by a small fragment of a much longer benchmark item, so a document that copied even one sentence verbatim still triggers a flag without requiring the entire item to have been reproduced. The exact value is a tuned engineering choice, not something derivable from first principles, and different labs' reports — where they disclose it at all — don't always converge on exactly the same number.

## Q16: Implement an n-gram-based contamination checker for benchmark decontamination, and explain how you'd make the training-corpus side of the check tractable at multi-trillion-token scale.

```python
def get_ngrams(tokens: list, n: int = 13) -> set:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

def build_benchmark_index(examples: list, n: int = 13) -> set:
    """Index both question and answer text -- either can leak independently."""
    index = set()
    for ex in examples:
        index |= get_ngrams(ex["question"].split(), n)
        index |= get_ngrams(ex["answer"].split(), n)
    return index

def is_contaminated(document: str, benchmark_index: set, n: int = 13, min_overlap: int = 1) -> bool:
    doc_ngrams = get_ngrams(document.split(), n)
    return len(doc_ngrams & benchmark_index) >= min_overlap
```

This is correct and exactly the shape of check any n-gram decontamination pipeline runs, but it's unworkable as written at real scale: it stores every n-gram as a tuple of strings in an ordinary Python set, and a multi-trillion-token corpus has on the order of trillions of overlapping 13-grams, whose string-tuple representation with Python object overhead would require far more memory than any reasonable machine or cluster can hold — well before you even get to running the check against dozens of benchmarks simultaneously.

The production fix has two parts. First, hash every n-gram to a fixed-size integer (a fast non-cryptographic hash like xxHash, or a rolling hash computed incrementally as the window slides) rather than storing the tuple itself — this turns a variable-length, string-heavy object into a small fixed-width value, with a collision probability that's astronomically small relative to corpus size for a well-chosen 64-bit hash (on the order of `N²/2^65` by the birthday bound) and negligible compared to other approximations already in the pipeline. Second, and more importantly for the tens-of-billions-of-distinct-n-grams regime real corpora sit in, use a Bloom filter rather than an exact hash set for the membership test itself: a fixed-size bit array plus k hash functions, where inserting sets k bits and querying checks whether all k are set. Any 0 bit means definite absence; all bits set means present-or-false-positive at a small, tunable rate governed by `(1 - e^{-kn/m})^k` for m bits, k hash functions, n inserted items.

The reason this one-sided error property is specifically the right trade for this application, not just a way to save memory, is that the two error directions are not symmetric in cost. A false negative — reporting "no overlap" when real contamination exists — silently leaves contaminated data in the corpus with no downstream symptom anywhere in the pipeline; nothing tells you it happened, and it directly corrupts the eventual benchmark measurement. A false positive merely causes an extra, otherwise-clean document (or benchmark example, when checking from that side) to be discarded at some small rate, which for a multi-trillion-token corpus and a benchmark suite of a few thousand examples is a rounding error, not a correctness problem. A Bloom filter trades a small, controllable, and safely-directed error rate for typically an order of magnitude or more reduction in memory versus an exact hash set at comparable tolerance — precisely the trade that makes checking a multi-trillion-token corpus against dozens of benchmarks computationally tractable.

## Q17: A frontier model reports a suspiciously high score on GSM8K relative to its performance on other, less famous grade-school-math benchmarks that probe the same underlying skill. You have no access to the training corpus. How would you actually investigate this?

I'd start by being honest about what I can and can't establish. Without corpus access, I cannot get a ground-truth answer to "was GSM8K actually clean at training time" — that's the structural asymmetry of this whole problem: the lab has ground truth about its own data, and I fundamentally don't. What I can do is gather probabilistic, indirect evidence and be disciplined about not overstating what it proves.

First, I'd treat the cross-benchmark asymmetry itself as a starting signal rather than a conclusion — benchmarks legitimately differ in difficulty and format even when nominally testing "the same skill," so a gap alone is circumstantial, but a large and consistent gap across several comparable benchmark pairs raises the prior meaningfully. Second, I'd run a paraphrase/perturbation test, which is the most informative and most reproducible check available with nothing more than API access: construct deliberately reworded versions of GSM8K questions — altered surface numbers, reordered structure, translated-and-back-translated phrasing — that require identical underlying reasoning, and re-evaluate. If the score holds up under paraphrase, that's real evidence the capability is robust and not keyed to memorized canonical phrasing; if it drops sharply while the required reasoning is unchanged, that's strong evidence part of the original score depended on recognizing the specific published wording. Third, I'd run a direct memorization probe: feed a short verbatim prefix of a known GSM8K item with no further context and check whether the model completes it with the exact published continuation — a model that reliably reproduces long, low-probability-to-guess continuations of specific published wording from a short prefix is behaving consistently with memorization, since that doesn't happen by chance at length. Fourth, if I had access to per-token output log-probabilities, I'd consider a likelihood-based membership-inference probe (Min-K%-Prob-style): checking whether the model assigns anomalously high likelihood specifically to the low-probability, "surprising" tokens in a passage relative to a calibrated clean reference, which can pick up softer memorization that neither of the other two techniques would catch.

None of these four signals, individually or combined, amounts to proof. I'd present the finding as a probabilistic, evidence-weighted judgment — "the paraphrase-sensitivity result and the cross-benchmark asymmetry together make contamination a substantially more plausible explanation than the raw score alone suggested" — rather than a verdict, and I'd explicitly flag that I cannot rule out soft contamination (paraphrased discussion of GSM8K problems circulating on the web, which n-gram matching inside the lab's own pipeline wouldn't even catch) as a mechanism the lab itself might not have been able to fully screen against, which is a different and harder problem than exact leakage.

## Q18: Self-instruct pipelines bootstrap instruction-tuning data by having a strong model generate new instructions and responses from a small human-written seed set. This sounds circular — how can a model generate useful new training data from itself? Explain why it actually works, and where its ceiling is.

The resolution is that self-instruct isn't attempting to create new information — it's solving a data-format problem, and those are genuinely different things. The generator model, by virtue of its pretraining (and any prior instruction-tuning), already has broad instruction-following-adjacent capability baked into its weights: pretraining on a huge corpus of human-written text already exposed it to countless examples of competent summarization, classification, and creative writing, just never packaged as explicit (instruction, response) supervision pairs. What self-instruct does is repackage that latent capability into the specific paired format a supervised fine-tuning objective actually needs — converting "the model can already do this if prompted the right way" into "here is a labeled example of the model doing this, indexed under an explicit instruction, usable as a gradient-descent target."

The analogy I find clarifying: a domain expert who sits down and writes structured labels distilling their own tacit knowledge is doing genuinely valuable labeling work — the resulting dataset is useful and didn't exist before — even though the expert isn't learning any new domain facts in the process of writing the labels down. Self-instruct is the LLM analogue of that act. Mechanically, the pipeline is iterative: start from a small, diverse human-written seed set; few-shot-prompt a strong generator to produce new instructions similar in spirit but not near-duplicates of the seeds; generate matching inputs and responses for those new instructions; filter aggressively (near-duplicate removal, a diversity filter that rejects instructions too similar to anything already in the pool even if not exact duplicates, heuristic quality filters for malformed generations); and feed the survivors back in as an enlarged seed set for the next round.

The ceiling follows directly from the "repackaging, not creation" framing: self-instruct can extract and reformat capability that's already latent in the generator, but it cannot inject capability the generator doesn't have. If the generator is bad at multi-step arithmetic, a self-instruct pipeline built on top of it produces instruction-response pairs that are also bad at multi-step arithmetic, just now wrapped in an instruction — the technique amplifies coverage and format-alignment, not the raw ceiling of the underlying model's competence. This is also why it matters whether the generator and the fine-tuning target are the same model (the original Self-Instruct paper's setup, self-referential) or different models (Alpaca's setup, using a stronger proprietary generator to tune a separate base model) — in the cross-model case you're specifically trying to transfer some of the stronger generator's latent capability into the weaker target, which is a meaningfully different bet than a model bootstrapping purely from its own outputs.

## Q19: Why do RLHF pipelines collect pairwise or k-wise preference rankings instead of just asking labelers to rate each response on an absolute 1–10 scale?

The reason is a well-established finding from psychometrics and preference-elicitation research generally, not something specific to language models: humans are demonstrably more reliable and internally consistent at relative judgments — "is A better than B?" — than at absolute judgments — "how good is A, on its own, on a fixed numeric scale?" Absolute scales are prone to calibration drift in a way that isn't fixable by writing a more detailed rubric. A labeler's internal sense of what "a 7" means isn't a portable, fixed unit: it shifts over the course of a single session as they see more examples, it shifts across different labelers who never explicitly synchronized their internal scales with each other, and it shifts over time as a labeling program runs for months and both the labeler population and the prompt distribution evolve. Two careful, good-faith labelers can look at the identical response and assign a 6 and a 9 not because either is being careless, but because "9" denotes a genuinely different absolute standard in each person's head — and the same labeler's own "9" from week one may not mean the same thing as their "9" from week six.

A pairwise or k-wise comparison sidesteps this almost entirely because it only requires a locally consistent ordinal judgment: given these two specific responses to this specific prompt, sitting side by side right now, which do I prefer? That judgment doesn't depend on any labeler's notion of what a universal "9" means, doesn't need to be stable across sessions or across people in absolute terms, and is empirically far more reproducible — labelers shown the same pair tend to agree with each other, and with their own past selves, on "which is better" far more often than they would on "what absolute score is this."

There's also a real efficiency angle: a k-wise ranking over K completions (InstructGPT used K=4 to 9) is more label-efficient than collecting independent pairwise judgments, because a single full ordering implies a preference for every pair within it — C(K,2) pairwise comparisons extracted from one labeling pass rather than requiring C(K,2) separate interactions. The one subtlety worth flagging is that those C(K,2) derived pairs are not independent data points — they all condition on the same prompt and largely the same handful of completions, so a prompt with K=9 contributes 36 highly correlated pairs while K=4 contributes only 6, and treating each pair as an i.i.d. training example both overweights prompts that happened to get a larger K and risks overfitting, since the effective number of independent judgments per prompt is closer to K than to C(K,2) — which is why InstructGPT normalizes the loss per prompt by its comparison count rather than treating every derived pair as an equal, independent unit.

## Q20: Explain the fair-use argument for and against training on copyrighted web text, and describe why Llama 1's "publicly available data only" constraint doesn't fully resolve the legal question even for Llama 1 itself.

Fair use, under US copyright law, is a case-by-case, four-factor balancing test — the purpose and character of the use (commercial versus transformative), the nature of the copyrighted work, the amount used relative to the whole, and the effect on the market for the original — none of which is individually dispositive, which is exactly why reasonable people can disagree about how it applies to a genuinely novel use case. The strongest pro-fair-use argument centers on transformative use: in the ordinary case, training doesn't cause a model to store or reproduce a work in a form that competes with the original — the work is one signal among an enormous number contributing to learned parameters representing general patterns, analogized to how a widely-read human is free to write new material informed by that reading without each new sentence infringing every book read, with some support from prior technology cases (search-engine indexing, thumbnail display, TDM-style full-text-index cases) where transformative, non-expressive uses of copyrighted material were found to be fair use. The strongest counter-argument contests both the framing and the conclusion: the input stage still requires copying the full work regardless of what the output looks like, done at industrial scale for commercial purposes often without a license sought; and the fourth factor can cut against fair use if the resulting model can produce output that competes in the market for the original works — closely related to, though analytically distinct from, the verbatim-memorization concern, since a model that can be prompted to reproduce a long near-exact passage of a specific work is not obviously just extracting generalized statistical patterns from it.

I'd be explicit that this is US-specific framing — the EU's DSM Directive has an explicit statutory TDM exception with a rightsholder opt-out mechanism, a structurally different approach from open-ended after-the-fact balancing, and Japan's TDM exception is generally described as comparatively permissive — so "is AI training fair use" fragments into a separate legal analysis per jurisdiction rather than being one global question, and it's actively being litigated in multiple real lawsuits with mixed and evolving early signals, not resolved in either direction.

On Llama 1 specifically: the paper's "publicly available and compatible with open sourcing" framing is a real, deliberate, disclosed sourcing constraint, and it plausibly serves reproducibility (every source is auditable by outside researchers, unlike GPT-3's or PaLM's undisclosed proprietary mixtures) and some risk reduction relative to using known-proprietary or pirated data. But "publicly available" is not the same legal category as "public domain," "openly licensed," or "cleared for commercial training use" — a document being technically obtainable by the public says nothing by itself about whether the rightsholder authorized this specific use. This becomes concrete rather than academic in Llama 1's own Books component: Gutenberg is unambiguously public domain, but Books3 (from EleutherAI's Pile) is a shadow-library-derived corpus of largely still-copyrighted books, and both were, in the literal sense the paper used, "publicly available" at the time — which is exactly why the constraint, real as it is, doesn't by itself settle the underlying copyright question even for the paper that adopted it most deliberately.
