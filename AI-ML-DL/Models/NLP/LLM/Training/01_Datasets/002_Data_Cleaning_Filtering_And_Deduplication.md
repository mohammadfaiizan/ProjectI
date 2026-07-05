# Data Cleaning, Filtering, and Deduplication

## 0. Why This Stage Exists

A raw Common Crawl snapshot is not a training corpus; it is a dump of essentially everything a web crawler could fetch, and the overwhelming majority of it is useless or actively harmful to train on: navigation chrome, cookie-consent banners, auto-generated product listings, spam, adult-content link farms, non-prose boilerplate, and the same paragraph of legal disclaimer text copy-pasted across millions of unrelated pages. Between the raw crawl and the tokens that actually get fed into a pretraining run sits a data pipeline whose job is exactly what this file's title says: clean it, filter it, and deduplicate it. This is not a peripheral concern relative to architecture or optimizer choice — for a compute-optimal pretraining run of the kind discussed in `..\..\GPT\003_GPT3.md` (Section 7's Chinchilla discussion), every token in the training budget is a scarce resource, and a token spent on a duplicated boilerplate string or a garbled OCR artifact is a token not spent on something that actually teaches the model language, facts, or reasoning. Getting this pipeline right is one of the highest-leverage, least-glamorous parts of building a frontier LLM, and it is exactly the kind of topic where staff-level interviews probe for hands-on mechanical understanding rather than a name-check of "we deduplicate the data."

The pipeline described in this file is conventionally organized as a funnel: cheap, high-throughput heuristics run first over the entire raw corpus to throw out the obviously bad majority of documents; progressively more expensive stages (a trained classifier, a language model for perplexity scoring, then fuzzy near-duplicate detection) run only over what survives the cheaper stages. This ordering is not arbitrary — it is a direct consequence of cost asymmetry. Running a small transformer or n-gram language model over trillions of raw tokens before removing the visually obvious junk would waste enormous compute scoring documents that a two-line regex would have rejected; running MinHash/LSH deduplication (Section 4) before removing garbage would waste hashing and bucketing effort on documents that should never have entered the corpus in the first place. Every open account of a large-scale web-text pipeline — CCNet (the pipeline underlying Llama's dataset construction lineage), the GPT-2/GPT-3 WebText/CC pipeline, C4, RefinedWeb, FineWeb — follows this same funnel shape, even though the exact heuristics and thresholds at each stage differ and are, in most cases, not fully disclosed by the labs that use them.

## 1. Quality Classifiers

### 1.1 What a fastText Classifier Actually Is

Before discussing how quality classifiers are trained for corpus filtering, it is worth being precise about the model class typically used, because "fastText classifier" is thrown around in papers as if it were self-explanatory. fastText (Bojanowski et al., Facebook AI Research, 2016-2017) is not a deep neural network in the modern sense; it is a deliberately shallow linear model built for extreme training and inference speed at massive scale, which is exactly why it recurs throughout LLM data pipelines despite far more accurate deep classifiers (BERT-style encoders, etc.) being available.

The mechanism: each input document is tokenized into words, and additionally into character n-grams and/or word n-grams (bigrams, trigrams) to capture some local word-order and subword information without needing a recurrent or attention mechanism. Every one of these n-gram features is mapped to an embedding vector via a hashing trick — rather than maintaining an explicit vocabulary-to-index dictionary (which would be memory-prohibitive if you tried to enumerate every possible n-gram over a huge corpus), each n-gram string is hashed into a fixed-size embedding table of, say, one or a few million buckets, with hash collisions simply treated as acceptable noise. The document's representation is the average of the embedding vectors of all its n-gram features. That single averaged vector is then fed through one linear layer into a softmax over the output classes. There is no hidden nonlinear transformation of consequence, no attention, no recurrence — the entire "deep learning" content of the model is a learned embedding table plus one linear projection.

```python
import numpy as np

def fasttext_hash(token: str, num_buckets: int) -> int:
    # Simplified version of the FNV-style hashing fastText uses to map
    # arbitrary n-gram strings into a fixed-size embedding table.
    h = 2166136261  # FNV offset basis
    for ch in token.encode("utf-8"):
        h = (h ^ ch) * 16777619 & 0xFFFFFFFF
    return h % num_buckets

def document_features(text: str, n_min=1, n_max=2):
    words = text.lower().split()
    feats = list(words)
    for n in range(n_min, n_max + 1):
        for i in range(len(words) - n + 1):
            feats.append(" ".join(words[i:i + n]))  # word n-gram
    return feats

def fasttext_forward(text: str, embed_table: np.ndarray, W: np.ndarray, b: np.ndarray):
    num_buckets, dim = embed_table.shape
    feats = document_features(text)
    if not feats:
        return None
    idxs = [fasttext_hash(f, num_buckets) for f in feats]
    doc_vec = embed_table[idxs].mean(axis=0)          # bag-of-hashed-n-grams average
    logits = doc_vec @ W + b                          # single linear layer
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()                            # softmax over classes
```

This architecture is a deliberate speed/accuracy trade-off, not an accident of history. Filtering a multi-trillion-token web corpus means scoring on the order of billions of documents; a fastText-style classifier can score tens of thousands of documents per second per CPU core with no GPU involved, whereas even a small transformer-based classifier would require GPU inference infrastructure at a scale that could rival the cost of the pretraining run itself just to filter the data feeding it. The lost accuracy relative to a deep classifier is judged, in practice, to be an acceptable price for making corpus-scale filtering computationally tractable at all — this is a recurring theme in data pipeline design: the filtering stage's own compute budget has to remain a small fraction of the pretraining compute budget it is meant to protect, or the pipeline defeats its own purpose.

### 1.2 Training the CC Quality Classifier: Positive/Negative Class Construction

The GPT-2 and GPT-3-era approach to Common Crawl quality filtering (and its many descendants) frames document quality as a binary classification problem where the two classes are not "good text" and "bad text" in any labeled, human-annotated sense — nobody hand-labels billions of documents — but rather a proxy contrast between curated reference text and raw, unfiltered web text.

The positive class is built from text that is already known, by construction, to have passed some human curation filter. GPT-2's WebText corpus took this literally: it scraped only the text of outbound links from Reddit submissions that had received at least 3 karma, using upvote count as a crude but scalable proxy for "at least one human decided this was worth sharing." GPT-3's Common Crawl quality classifier is trained analogously, using curated corpora — WebText-derived text and Wikipedia-linked pages are the canonical examples — as the positive class exemplars. The underlying assumption is that text a human curator selected, cited, or upvoted correlates with prose quality, factual coherence, and lower spam/boilerplate density relative to the average web page, even though this is a proxy and not a direct quality label, and it inherits whatever demographic and topical biases are present in who uses Reddit or who gets cited on Wikipedia.

The negative class is simply a random sample of raw, unfiltered Common Crawl documents — no curation signal at all, representative of "whatever a crawler happened to fetch." The classifier is then trained as ordinary supervised binary classification: positive-class documents get label 1, background CC documents get label 0, and the fastText model (Section 1.1) learns a linear decision boundary in n-gram-hashed feature space that separates the two.

```python
# Illustrative training-set construction, not a literal reproduction of any
# lab's exact sampling procedure (undisclosed in detail by any of them).
positive_examples = load_documents("wikipedia_outbound_linked_pages.jsonl")
positive_examples += load_documents("webtext_reddit_karma3plus.jsonl")

negative_examples = sample_random_documents(
    "common_crawl_raw_snapshot.warc", n=len(positive_examples)
)

training_set = (
    [(text, 1) for text in positive_examples] +
    [(text, 0) for text in negative_examples]
)
# fastText training itself is then standard: minimize cross-entropy of the
# softmax output against these two labels via SGD over hashed n-gram features.
```

It is worth being explicit about what this classifier is and is not measuring. It is not a factuality classifier, not a toxicity classifier, and not a coherence classifier in any semantic sense — it is a stylistic/distributional classifier that has learned which surface n-gram patterns correlate with "the kind of text that gets curated/cited/upvoted" versus "the kind of text a crawler indiscriminately fetches." A well-written, factually accurate forum post that happens not to resemble Wikipedia's register can score low; a stylistically polished but content-free page that happens to mimic encyclopedic phrasing can score higher than it deserves. This is a known, accepted limitation of the approach, not a defect that later pipelines pretend does not exist — it is one reason the heuristic pre-filters in Section 2 and the perplexity-based filtering in Section 1.3 are run as complementary signals rather than relying on the classifier alone.

### 1.3 From Classifier Score to Filtering Decision: Hard Threshold vs. Pareto/Stochastic Retention

Once every document in the corpus has a classifier score (the softmax probability of the positive class), there are two broad ways to turn that score into a keep/discard decision. The first is a hard threshold: keep every document whose score exceeds some cutoff `t`, discard everything else. This is simple and deterministic, but it has a structural weakness — it draws a sharp line through a continuous, noisy score distribution, and documents just below the cutoff are discarded with certainty even though the classifier's score near the decision boundary carries relatively little information (a document scoring 0.49 is not meaningfully worse than one scoring 0.51; the classifier's calibration is not that precise, especially for a shallow linear model over hashed features).

The alternative, used in GPT-3's pipeline and described in the literature as Pareto or stochastic retention, keeps documents probabilistically, with retention probability increasing monotonically with classifier score rather than jumping discontinuously at a threshold. A common parametric family for this is a Pareto-distribution-shaped acceptance function, where the probability of keeping a document is a smooth, monotonic function of its score, calibrated so that the aggregate retention rate across the whole corpus produces the desired overall filtering fraction. GPT-3's paper describes sampling documents according to their classifier score raised to a power via a Pareto distribution rather than a fixed cutoff, specifically framed as a way to include some lower-scoring documents rather than strictly excluding everything below an arbitrary line.

```python
import numpy as np

def pareto_keep_probability(score: float, alpha: float = 9.0) -> float:
    """
    Illustrative stochastic-retention rule in the spirit of GPT-3's approach:
    monotonic in score, but not a hard cutoff. A document with a middling
    score still has a real, if reduced, chance of being retained, preserving
    some tail diversity rather than sharply amputating it.
    """
    # score in [0, 1]; alpha controls how aggressively low scores are downweighted.
    u = np.random.random()
    threshold = 1.0 - (1.0 - score) ** alpha
    return u < threshold

def filter_corpus_stochastic(documents_with_scores, alpha=9.0, seed=0):
    rng = np.random.default_rng(seed)
    kept = []
    for doc, score in documents_with_scores:
        if rng.random() < 1.0 - (1.0 - score) ** alpha:
            kept.append(doc)
    return kept
```

The reason a soft/stochastic rule is often preferred to a hard threshold comes down to diversity preservation and avoiding a brittle, arbitrarily precise decision boundary. Real-world text quality is not bimodal — there is a continuum from clearly spam-like to clearly encyclopedic, and a large fraction of legitimately useful web text (informal blogs, non-native-English writing that is nonetheless coherent and informative, niche technical forums, minority-dialect content) sits in the middle of that continuum and would be systematically excised by a hard cutoff tuned to maximize precision on obviously-good versus obviously-bad exemplars. A stochastic rule instead says: the higher your score, the more likely you survive, but nothing is guaranteed to survive and nothing is guaranteed to be excluded purely by being on the wrong side of one number. This directly trades off against the over-filtering risk raised at the end of this file (Section 7) — a hard threshold aggressively pursued in the name of "quality" risks silently narrowing the register and demographic diversity of the training distribution, and stochastic retention is one mitigation, though not a complete one, since the underlying classifier's biases (Section 1.2) are still baked into the score that drives the keep probability.

### 1.4 Perplexity-Based Filtering

A second, mechanically distinct approach to document-quality filtering scores each candidate document under a small language model trained on curated, high-quality reference text (Wikipedia is the canonical choice, sometimes augmented with other clean sources), and computes the document's perplexity under that reference model — the exponentiated average negative log-likelihood the reference LM assigns to the document's tokens, `perplexity = exp(-1/N * sum_t log P(x_t | x_<t))`. Intuitively, perplexity measures how surprised the reference model is by the text: text that reads like fluent, well-formed prose similar to what the reference LM was trained on will receive a low perplexity, while text that is gibberish, boilerplate, OCR garbage, or structurally alien to prose (dense HTML remnants, tables serialized as text, keyword-stuffed SEO spam) will receive a high perplexity because the reference model's learned language patterns simply do not predict those token sequences well.

The filtering rule is then to accept documents whose perplexity falls within an accepted range — not just below an upper bound, but, more subtly, sometimes also above a lower bound. The upper-bound side is the intuitive one: very high perplexity flags exactly the garbled/non-prose content described above. The lower-bound side is less obvious and worth deriving carefully: an unusually low perplexity, well below what typical fluent prose achieves, can signal that a document consists of highly repetitive or degenerate text — the same phrase or sentence repeated many times, template text with only slot values changing, or other pathologically predictable sequences. A reference LM assigns extremely low perplexity to a token sequence that is trivially predictable from its own preceding context, and heavy repetition is exactly that: after the first repetition, "predict the next token" becomes nearly deterministic, so the average per-token negative log-likelihood collapses toward zero. A naive filter that only ever removes high-perplexity documents would let this kind of degenerate, low-information repetition straight through, since by the perplexity metric it looks like the most "confident," most "well-formed" text in the entire corpus.

```python
import math

def document_perplexity(tokens, reference_lm):
    """
    reference_lm.log_prob_next(prefix, token) returns log P(token | prefix)
    under a small LM trained on curated reference text (e.g., Wikipedia).
    """
    total_log_prob = 0.0
    for i in range(1, len(tokens)):
        total_log_prob += reference_lm.log_prob_next(tokens[:i], tokens[i])
    avg_neg_log_prob = -total_log_prob / max(1, len(tokens) - 1)
    return math.exp(avg_neg_log_prob)

def perplexity_filter(tokens, reference_lm, low=10.0, high=1000.0):
    ppl = document_perplexity(tokens, reference_lm)
    return low <= ppl <= high
```

The classifier and perplexity approaches are complementary precisely because they fail differently. The fastText-style classifier is a distributional/stylistic signal learned from a specific positive/negative contrast (Section 1.2); it can be fooled by text that superficially mimics the register of the positive class without actually being informative, and it has no direct mechanism for catching degenerate repetition, since a repeated boilerplate phrase can easily contain n-grams that look encyclopedic. Perplexity, conversely, has no notion at all of "is this the kind of text a human curator would select" — a perfectly fluent, grammatically pristine advertisement or an SEO-optimized but content-free article can achieve low perplexity under a reference LM while being exactly the kind of low-value content the quality classifier is designed to catch. Perplexity is sensitive to fluency and local predictability; the classifier is sensitive to topical/stylistic resemblance to curated sources. Using both, typically as independent filters or combined signals feeding into the funnel, catches a wider range of failure modes than either alone, and this combination — heuristics, then a trained classifier, then perplexity-style scoring, then fuzzy dedup — is the standard shape of a modern web-text pretraining pipeline.

## 2. Heuristic Filters: The Cheap First Pass

Before any classifier or reference LM is run — and this ordering matters for the cost reasons discussed in Section 0 — large-scale pipelines apply a battery of cheap, rule-based heuristics designed to catch obviously bad documents at a fraction of the compute cost. None of these individually is sophisticated; their value is in removing the bulk of clearly-junk documents so fast, expensive stages only have to process what remains.

**Boilerplate and template detection.** Web pages routinely embed large blocks of text that are identical, or nearly identical, across every page on a site or across many unrelated sites using the same content-management template: navigation menus, "subscribe to our newsletter" prompts, cookie-consent banners, copyright footers. A simple and effective heuristic is line-level repetition detection within a document or across a small window of documents from the same crawl batch: if a given line (or a short sequence of lines) recurs verbatim an abnormal number of times, it is very likely boilerplate rather than content, and can be stripped or the containing document flagged.

**Lorem ipsum and placeholder text.** Many crawled pages are unfinished templates, test pages, or content-management-system defaults that were never replaced with real content, and contain literal placeholder Latin ("lorem ipsum dolor sit amet...") or other obviously synthetic filler. Detecting the presence of known placeholder-text fragments is a near-zero-cost exact-match check that removes documents containing no real information content at all.

**Line-length heuristics.** Genuine prose has a fairly stable statistical signature of average line length (measured in words or characters) once line breaks correspond to paragraph or sentence boundaries. Pages that are mostly navigation links, tag clouds, product-listing grids, or menus tend to have a very different signature — many very short lines, since each "line" is a single link or label rather than a sentence. Flagging documents whose average line length falls well below what prose typically exhibits is a cheap proxy for "this is structurally not an article."

**Symbol-to-word ratio.** Legitimate prose has a fairly bounded ratio of punctuation/symbol characters to alphabetic word characters. Spam, keyword-stuffed SEO pages, and certain kinds of scraped structured data (price listings, code fragments misidentified as prose, character-corrupted encodings) tend to have anomalously high symbol density. A simple ratio-based filter catches a lot of this cheaply.

```python
import re

def symbol_to_word_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z]+", text)
    symbols = re.findall(r"[^\w\s]", text)   # punctuation/non-alphanumeric symbols
    if not words:
        return float("inf")
    return len(symbols) / len(words)

def is_low_quality_by_symbol_ratio(text: str, max_ratio: float = 0.5) -> bool:
    return symbol_to_word_ratio(text) > max_ratio


def has_excessive_line_repetition(text: str, max_repeat_fraction: float = 0.3) -> bool:
    """
    Cheap boilerplate detector: if any single line accounts for more than
    max_repeat_fraction of all non-empty lines in the document, treat it as
    template/boilerplate-dominated rather than prose. Real per-document use
    typically also checks repetition *across* documents in a crawl batch
    (shared nav/footer text), not just within one document, but this
    within-document version illustrates the mechanism cheaply.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    counts = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    most_common_count = max(counts.values())
    return (most_common_count / len(lines)) > max_repeat_fraction
```

**Language identification.** A fastText-style language-ID model — architecturally the same shallow bag-of-hashed-n-gram classifier described in Section 1.1, just trained to distinguish languages rather than quality — is run over every document to predict its language with a confidence score. This serves two distinct purposes simultaneously: routing documents into the correct per-language bucket for pipelines that build separate or weighted per-language pretraining subsets (a document confidently identified as French goes into the French bucket, contributing to whatever sampling weight that language receives in the final mixture), and filtering out documents in languages the pipeline has no intention of training on, or documents whose language cannot be identified with sufficient confidence at all (often a symptom of corrupted encoding, character-mangled text, or content that is not actually natural language, such as raw binary or minified code misclassified as a webpage).

The unifying theme across all of Section 2 is that each individual heuristic is deliberately simple, deliberately cheap, and deliberately imprecise — false positives and false negatives at this stage are an accepted cost because the heuristics exist to shrink the corpus fast, not to make the final quality call. The classifier and perplexity stages downstream (Section 1) are where more nuanced judgment is applied, at correspondingly higher per-document cost, over a corpus that Section 2's funnel has already shrunk substantially.

## 3. Exact Deduplication via Hashing

Exact deduplication answers a narrower question than the fuzzy/near-duplicate problem covered in Section 4: is this document (or line, or sentence — the unit is a design choice) byte-for-byte, or at least token-for-token after normalization, identical to one already seen? The mechanism is a direct application of hash sets. Normalize the text (lowercase, strip whitespace variance, sometimes strip punctuation depending on how strict the notion of "exact" is meant to be), compute a hash of the normalized text using either a cryptographic hash function (SHA-256, favored when hash-collision resistance needs to be provably strong, at the cost of being computationally heavier) or a fast non-cryptographic hash (xxHash, MurmurHash — favored at corpus scale for raw throughput, since deduplication does not need cryptographic collision resistance against an adversary, only extremely low collision probability against an enormous but non-adversarial corpus), and check that hash against a hash set of everything seen so far. If the hash is already present, discard the document (or line) as a duplicate; if not, add it to the set and keep the document.

```python
import hashlib
from typing import Iterator, Set

def normalize(text: str) -> str:
    # Minimal normalization: collapse whitespace, lowercase. Real pipelines
    # often also strip punctuation and normalize unicode (NFKC) before hashing,
    # since "exact" duplicates on the web frequently differ only in whitespace
    # or invisible-character noise, not in substantive content.
    return " ".join(text.lower().split())

def hash_text(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()

def dedup_documents(documents: Iterator[str]) -> Iterator[str]:
    seen: Set[bytes] = set()
    for doc in documents:
        h = hash_text(normalize(doc))
        if h in seen:
            continue
        seen.add(h)
        yield doc

def dedup_lines_within_and_across_documents(documents: Iterator[str]) -> Iterator[str]:
    """
    Line-level exact dedup: rebuilds each surviving document from only its
    non-duplicate lines. Catches content (e.g., a legal disclaimer, a
    boilerplate paragraph) repeated verbatim across many otherwise-distinct
    documents, which document-level hashing would never flag, since the
    surrounding document text differs from page to page.
    """
    seen_lines: Set[bytes] = set()
    for doc in documents:
        kept_lines = []
        for line in doc.split("\n"):
            norm = normalize(line)
            if not norm:
                kept_lines.append(line)
                continue
            h = hash_text(norm)
            if h in seen_lines:
                continue
            seen_lines.add(h)
            kept_lines.append(line)
        yield "\n".join(kept_lines)
```

The choice between document-level and line-level exact dedup is not merely a granularity preference; the two catch structurally different redundancy. Document-level exact dedup only removes a document if the *entire* normalized document string matches something already seen — this catches true mirror pages (the same article syndicated verbatim at multiple URLs) but does nothing about a legal disclaimer paragraph, a cookie-notice sentence, or a standard code-license header that appears, verbatim, embedded inside thousands of otherwise-distinct documents, since each containing document as a whole is unique even though a substantial substring of it is repeated everywhere. Line-level exact dedup (or the closely related n-gram/sentence-level variant) hashes each line independently and can strip that repeated disclaimer out of every document it appears in, without discarding the surrounding unique content. This is exactly the mechanism CCNet-style pipelines apply — Wenzek et al.'s CCNet, which underlies the data-processing lineage used for Llama-family pretraining corpora, performs deduplication at the paragraph level for precisely this reason: the redundancy that matters most for a web-scale corpus is very often sub-document (a repeated boilerplate paragraph), not whole-document.

The corresponding cost is that line-level exact dedup produces vastly more hashing/lookup operations than document-level dedup (a corpus with an average of, say, 50 lines per document generates roughly 50x as many hash-set operations), and it introduces its own failure mode: some short lines or sentences are legitimately supposed to recur many times across genuinely distinct, useful documents (a common section heading like "References", a standard code license header, a frequently-repeated idiom or Bible verse quoted in unrelated contexts) — stripping every recurrence of these indiscriminately can remove content that is not actually redundant in the sense that matters. Production pipelines typically address this with frequency thresholds (only strip a line/paragraph as boilerplate if it recurs more than some minimum number of times across the corpus, rather than deduplicating on a first-seen-wins basis for every single line) rather than naive line-level hash-set deduplication as shown in the illustrative snippet above.

## 4. Near/Fuzzy Deduplication — MinHash and LSH

Exact hashing (Section 3) only catches documents that are byte-identical after normalization. The overwhelming majority of near-duplicate content on the web is not byte-identical: the same news article gets syndicated with a different byline or a slightly edited lede; the same product description appears on dozens of e-commerce mirror sites with different surrounding navigation chrome; a forum thread gets scraped by multiple archival crawlers with minor formatting differences. None of this survives exact hashing's collision check, because a single character difference produces a completely different hash. Catching this class of redundancy requires a *similarity* measure that degrades gracefully with small edits, and an algorithm to find similar-but-not-identical pairs without comparing every pair of documents in the corpus — which is where MinHash and Locality-Sensitive Hashing (LSH) come in.

### 4.1 Shingling and Jaccard Similarity

The first step is to convert each document into a set representation that is insensitive to where in the document a passage of text sits, and only sensitive to what substrings it contains. This is done via shingling: a document is represented as the set of all overlapping k-grams (contiguous windows of k consecutive tokens — words are common, though character k-grams are also used, particularly for shorter or more structured text) that appear anywhere in it.

```python
def shingle(text: str, k: int = 5) -> set:
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}
```

Two documents' similarity is then measured as the Jaccard similarity of their shingle sets: `J(A, B) = |A ∩ B| / |A ∪ B|`, the fraction of the combined vocabulary of k-grams that the two documents share. This is a natural similarity measure for near-duplicate detection because it is robust to reordering of unrelated content and degrades roughly proportionally to the fraction of text that differs between two documents — two documents that share 95% of their k-grams and differ only in a byline or a few edited sentences will have a Jaccard similarity close to 1, while two unrelated documents will have a similarity close to 0.

### 4.2 MinHash: An Unbiased Estimator of Jaccard Similarity

Computing exact Jaccard similarity requires materializing and intersecting full shingle sets, which is expensive to do for every pair of documents in a corpus containing potentially trillions of shingles in aggregate. MinHash (Broder, 1997) sidesteps this by compressing each document's shingle set into a small, fixed-size signature that lets you *estimate* Jaccard similarity between two documents by comparing their signatures directly, without ever looking at the original shingle sets again.

The construction: choose a hash function `h` that maps shingles (or, in practice, their pre-hashed integer representations) uniformly at random over some large integer range. For a document represented by shingle set `A`, define `minhash_h(A) = min_{x in A} h(x)` — apply `h` to every shingle in `A` and keep the minimum resulting value.

The key claim, which is the mathematical heart of the whole method, is this: for two shingle sets `A` and `B`, and a hash function `h` drawn uniformly at random from a family of hash functions that behaves like a random permutation of the universe of possible shingle values,

```
P( minhash_h(A) = minhash_h(B) ) = |A ∩ B| / |A ∪ B| = J(A, B)
```

The derivation is a clean combinatorial argument and worth being able to reproduce precisely. Consider the union `A ∪ B`, and think of `h` as inducing a uniformly random ordering (a random permutation) over the elements of `A ∪ B` — this is the defining property of an idealized random hash function over a finite universe. The minimum-hash element of `A` under `h` is, by definition, whichever element of `A` happens to be *first* in this random ordering restricted to `A`; likewise for `B`. Now ask: what does it take for `minhash_h(A) = minhash_h(B)`? Since a minimum is always achieved by some specific element, the two minima can only be equal if that minimizing element is a *single* element that lies in both `A` and `B` — i.e., an element of `A ∩ B` — and moreover that element must be the very first element of the entire union `A ∪ B` in the random ordering induced by `h` (if the overall-first element of `A ∪ B` belonged only to `A` and not `B`, then `A`'s minimum would be that element while `B`'s minimum would be some later, different element, so the two minima could not coincide; symmetric reasoning applies if the first element belonged only to `B`). So the event `minhash_h(A) = minhash_h(B)` is exactly the event "the first element of `A ∪ B`, under a uniformly random ordering, happens to land in the subset `A ∩ B`." Since the ordering is uniformly random over all `|A ∪ B|` elements, each element is equally likely to be the one that comes first, so the probability that the first element specifically falls in `A ∩ B` is simply `|A ∩ B| / |A ∪ B|` — exactly the Jaccard similarity. This gives a single hash function's minhash-agreement indicator as an unbiased Bernoulli estimator of `J(A, B)`: it equals 1 with probability `J(A, B)` and 0 otherwise.

A single Bernoulli trial is a very noisy estimator on its own (its variance is `J(A,B)(1-J(A,B))`, which is not small). The MinHash *signature* fixes this by repeating the procedure with `h` independent hash functions `h_1, ..., h_h` (unfortunately-overloaded notation in the literature — call the count of hash functions `H` to avoid confusion with the hash function symbol), producing a signature vector `(minhash_{h_1}(A), ..., minhash_{h_H}(A))` for each document, and estimating Jaccard similarity as the *fraction* of the `H` positions at which the two documents' signatures agree:

```
J_hat(A, B) = (1/H) * sum_{i=1}^{H} [minhash_{h_i}(A) = minhash_{h_i}(B)]
```

Because each of the `H` indicator terms is an independent (assuming independent hash functions) unbiased Bernoulli estimator of the same quantity `J(A, B)`, their average is still unbiased, and by the standard variance-of-an-average argument its variance shrinks by a factor of `H` relative to a single trial: `Var(J_hat) = J(A,B)(1-J(A,B)) / H`. This is the entire statistical justification for using many hash functions rather than one — it is a direct application of the law of large numbers to reduce estimator variance, trading compute (`H` hash function evaluations per shingle) for estimate precision.

```python
import numpy as np

def make_hash_functions(num_hashes: int, seed: int = 0):
    """
    Returns num_hashes independent-ish hash functions of the form
    h(x) = (a * x + b) mod p, the standard universal-hashing family used to
    approximate independent random hash functions cheaply for MinHash.
    """
    rng = np.random.default_rng(seed)
    p = (1 << 61) - 1  # a large prime, Mersenne prime 2^61 - 1
    a = rng.integers(1, p, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, p, size=num_hashes, dtype=np.int64)
    return a, b, p

def minhash_signature(shingles: set, a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    if not shingles:
        return np.full(len(a), p, dtype=np.int64)
    # Map each shingle string to an integer via a fast non-cryptographic hash first.
    shingle_ints = np.array([hash(s) & ((1 << 61) - 1) for s in shingles], dtype=np.int64)
    # Broadcast: (num_shingles, 1) against (1, num_hashes) -> (num_shingles, num_hashes)
    hashed = (a[None, :] * shingle_ints[:, None] + b[None, :]) % p
    return hashed.min(axis=0)  # the MinHash signature: min over shingles, per hash function

def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))
```

### 4.3 LSH Banding: Making Candidate Generation Sublinear

Even with a compact MinHash signature per document, finding all near-duplicate *pairs* in a corpus of `n` documents by comparing every pair of signatures is an `O(n^2)` operation. For a corpus with hundreds of millions to billions of documents, `n^2` pairwise comparisons is not merely slow, it is categorically infeasible — at even a conservative billion documents, `n^2` is on the order of `10^18` comparisons.

Locality-Sensitive Hashing (LSH) banding solves this by converting the "compare every pair" problem into a "bucket documents, only compare within a bucket" problem, at the cost of an approximate (probabilistic) guarantee rather than an exact one. The MinHash signature, a vector of `H` hash values, is partitioned into `b` contiguous bands, each containing `r` rows (hash values), so `H = b * r`. For each band, all `r` values within that band are hashed together into a single band-hash (e.g., by concatenating the `r` values and hashing the concatenation); two documents are placed into the same hash bucket for that band if and only if their band-hash matches, which requires all `r` values within that band to match exactly. A document is emitted as a *candidate* near-duplicate pair with another document if the two collide in at least one band's bucket, across any of the `b` bands.

```python
from collections import defaultdict

def lsh_band_signature(signature: np.ndarray, num_bands: int) -> list:
    rows_per_band = len(signature) // num_bands
    bands = []
    for i in range(num_bands):
        band = signature[i * rows_per_band: (i + 1) * rows_per_band]
        bands.append(hash(tuple(band.tolist())))
    return bands  # one hashable value per band

def build_lsh_index(doc_ids, signatures, num_bands: int):
    """
    Returns candidate pairs: documents that share at least one band bucket.
    This is the sublinear step — instead of O(n^2) signature comparisons,
    we do O(n * b) hashing work plus cheap bucket lookups.
    """
    buckets = defaultdict(list)  # (band_index, band_hash) -> [doc_ids]
    for doc_id, sig in zip(doc_ids, signatures):
        for band_idx, band_hash in enumerate(lsh_band_signature(sig, num_bands)):
            buckets[(band_idx, band_hash)].append(doc_id)

    candidate_pairs = set()
    for bucket_docs in buckets.values():
        if len(bucket_docs) < 2:
            continue
        for i in range(len(bucket_docs)):
            for j in range(i + 1, len(bucket_docs)):
                pair = tuple(sorted((bucket_docs[i], bucket_docs[j])))
                candidate_pairs.add(pair)
    return candidate_pairs
```

Crucially, LSH banding is only a *candidate generation* step — it produces a set of pairs that are plausibly similar and worth checking, dramatically pruned down from all `n^2` pairs, and the actual, final similarity (either the estimated MinHash Jaccard, or the true Jaccard on the original shingle sets for extra precision) is still computed only on that much smaller candidate set. The pruning is where the sublinear-in-practice behavior comes from: as long as buckets remain small relative to the corpus (which holds when true near-duplicates are a small fraction of all pairs, the expected regime for a web corpus), the total candidate-pair count and the total hashing work scale far better than quadratically with `n`.

### 4.4 The S-Curve: Choosing `b` and `r`

The probability that two documents with true Jaccard similarity `s` become a candidate pair (collide in at least one band) has a clean closed form, and understanding its shape is the key design tool for choosing `b` and `r`. Within a single band of `r` rows, the probability that all `r` MinHash values match — recall each individual value matches with probability `s` under the MinHash estimator — is `s^r` (independence across the `r` rows within the band, by construction, since they come from independent hash functions). So the probability that this one band does *not* produce a collision is `1 - s^r`. The probability that *none* of the `b` independent bands collide is `(1 - s^r)^b` (independence across bands, again by construction). Therefore the probability that at least one band collides — i.e., the two documents become a candidate pair — is:

```
P(candidate pair | true similarity s) = 1 - (1 - s^r)^b
```

This function of `s`, for fixed `b` and `r`, has the shape of an S-curve: it stays close to 0 for `s` below some knee, rises steeply through the knee, and approaches 1 for `s` above it. The location and steepness of that knee are what `b` and `r` control, and this is the entire art of LSH parameter selection: you want a sharp transition centered near your target similarity threshold, so that document pairs above the threshold are very likely to be flagged as candidates (few false negatives — true near-duplicates slipping through unflagged) and document pairs below the threshold are very unlikely to be flagged (few false positives — wasted comparison work, or worse, non-duplicates incorrectly discarded downstream). Increasing `r` (more rows required to match within a band) makes each individual band's collision probability `s^r` drop faster as `s` decreases, sharpening the low end of the curve and pushing the knee's threshold higher; increasing `b` (more independent bands, each an independent chance to collide) raises the overall collision probability at any given per-band probability, pushing the knee's threshold lower and making the transition sharper by giving more chances to catch a collision. In practice, `H` (total hash functions, `H = b*r`) is fixed by a compute/memory budget, and `b` and `r` are chosen as a factorization of `H` that places the S-curve's knee at approximately the similarity threshold the pipeline wants to treat as "near-duplicate" — a threshold of `s ≈ 0.8` or `s ≈ 0.9` (interpreted as "80% or 90% of shingles shared") is a commonly cited operating point in web-corpus dedup pipelines, though the exact value is a tunable design choice rather than a universal constant, and is not fully disclosed by every lab that uses this technique.

```python
def theoretical_candidate_probability(s: float, b: int, r: int) -> float:
    return 1 - (1 - s ** r) ** b

# Example: with H = 200 hashes split as b=20 bands of r=10 rows each,
# the S-curve's knee sits close to s ~ 0.7-0.8 -- pairs much below that
# true similarity are very unlikely to ever collide in any band.
for s in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    print(s, theoretical_candidate_probability(s, b=20, r=10))
```

## 5. Document-Level vs. Sentence/Paragraph-Level Fuzzy Dedup

Everything in Section 4 was described at the document level — shingle a whole document, MinHash the whole document's shingle set, LSH-bucket whole documents. This is the cheapest granularity at which to apply fuzzy dedup, and it catches the majority of what large-scale pipelines care most about: mirrored pages, syndicated news articles republished with minor edits across outlets, and boilerplate-heavy pages that are near-copies of each other because they were generated from the same content-management template. It is also the granularity used by the large majority of published web-corpus pipelines (CCNet/Llama-lineage corpora, C4, RefinedWeb) as their primary fuzzy dedup pass, precisely because whole-document MinHash/LSH scales to the size of a full web crawl in a way that finer-grained alternatives strain against.

The blind spot of document-level fuzzy dedup is partial overlap: a document that consists of, say, 40% verbatim-copied content (a long quoted excerpt, a block of copied boilerplate, a substantial plagiarized section) embedded within otherwise unique surrounding prose will often still have overall Jaccard similarity to the source document well below any reasonable near-duplicate threshold, because the unique 60% dilutes the shingle-set overlap. Document-level dedup, by design, only flags pairs whose *aggregate* similarity crosses the threshold — it has no visibility into whether a smaller but still substantively duplicated chunk exists inside two otherwise-different documents.

Sentence- or paragraph-level fuzzy dedup addresses this by shingling and MinHashing much smaller units, catching partial overlap that document-level comparison would dilute away. The cost is a large multiplicative increase in the number of units that must be hashed, signed, and bucketed — a corpus with an average of tens of paragraphs per document turns into tens of times as many MinHash computations and LSH lookups, and the corresponding candidate-pair generation and downstream verification step scales accordingly. There is also a distinct precision risk at this finer granularity: many short passages recur across genuinely unrelated, legitimately useful documents for reasons that have nothing to do with redundant training signal — common idiomatic phrases, standard section headers, boilerplate legal or licensing language that is fine to see repeated a bounded number of times (a software project's MIT license header, a standard "Terms of Service" clause), or frequently-quoted text (proverbs, scripture, famous quotations). Treating every such short-unit match as a duplicate to be removed risks stripping legitimate, recurring-by-nature language rather than genuinely redundant content, which is a different failure mode from anything document-level dedup exhibits, since document-level Jaccard similarity is far less likely to be driven above threshold by a single shared short phrase.

In practice, large-scale pipelines draw the line closer to the document-level end of this spectrum for the primary dedup pass, and treat finer-grained (paragraph/sentence, or n-gram-frequency-based) deduplication as a secondary, more targeted mechanism — often folded into the boilerplate-stripping heuristics of Section 2/3 rather than run as a full independent MinHash/LSH pipeline over every sentence in the corpus. CCNet's paragraph-level deduplication is a documented example of pushing one level below whole-document granularity as a deliberate middle ground: fine enough to catch a lot of the repeated-boilerplate-paragraph problem described in Section 3, without going all the way to full sentence-level MinHash/LSH over the entire corpus, which very few publicly described pipelines report doing exhaustively at trillion-token scale — the compute cost is judged, implicitly by the field's practice, not to be worth the incremental redundancy caught beyond what document- and paragraph-level dedup already remove. The exact granularity choices and thresholds used by frontier labs for their most recent training runs are, in most cases, not disclosed in detail.

## 6. Why Deduplication Matters

It is easy to treat deduplication as a hygiene step that is obviously good to do and move on; the staff-level version of this topic is being able to state concretely *why* it matters, along three separable axes.

**Memorization and verbatim regurgitation.** The empirical pattern that duplicated training sequences are disproportionately likely to be memorized and reproduced near-verbatim by a trained language model at generation time is well established in the LLM training literature — models are measurably more likely to regurgitate a passage seen many times during training than one seen once, and the relationship is not linear: a sequence duplicated a handful of extra times can see its memorization likelihood increase sharply relative to a singleton sequence. This is stated here as a well-documented empirical pattern broadly reported across LLM memorization studies, rather than attributed to one specific paper, since the exact quantitative relationship (how memorization probability scales with duplicate count, model size, and sequence length) has been characterized across several distinct studies with somewhat different setups rather than settled by a single canonical source. The practical consequence is direct: deduplication is one of the most effective, cheapest available levers for reducing a deployed model's propensity to output memorized training text verbatim — a concern that spans privacy (regurgitating personal information that appeared in training data), copyright (regurgitating substantial verbatim passages of copyrighted text), and simple quality (a model that has memorized rather than generalized from a passage is not demonstrating the capability the field actually wants from it).

**Benchmark contamination.** If a benchmark's evaluation documents, or close paraphrases of them, exist elsewhere on the open web and get scraped into the pretraining corpus — which happens more often than one might assume, since many benchmarks are themselves derived from or discussed on public web pages, forums, and repositories — and if the corpus's deduplication pipeline fails to identify and remove that overlap, benchmark scores become inflated in a way that reflects memorization of the specific eval set rather than the underlying capability the benchmark is meant to measure. This is a distinct problem from generic training-data redundancy: it does not merely waste compute, it actively corrupts the credibility of the reported evaluation numbers, and deduplication against known benchmark corpora is one of the direct mitigations. This topic — how contamination is detected and how it is scrubbed from a training corpus, as distinct from generic dedup against the corpus itself — is covered in depth in `005_Contamination_Detection_And_Decontamination.md`; the point to carry forward here is simply that fuzzy near-dedup machinery (Section 4) is a prerequisite building block for that later, more targeted contamination-scrubbing problem, not a separate technique.

**Training efficiency and compute-optimal token budgets.** Every duplicated token that survives into the training corpus is a token of gradient-descent compute spent re-learning something the model has, in expectation, already been exposed to, at the direct opportunity cost of a token of genuinely new information that could have occupied that same slot in the training budget. This connects concretely to the compute-optimal framing discussed in `..\..\GPT\003_GPT3.md`'s Section 7 treatment of Chinchilla-style token budgeting: a compute-optimal training run targets a specific token count for a given model size and compute budget, and that token count is implicitly assumed to consist of *useful, non-redundant* tokens. A corpus with substantial undetected duplication effectively has a smaller true information content than its raw token count suggests, meaning a model trained for a nominally compute-optimal number of tokens against a duplication-heavy corpus is, in the dimension that actually matters (unique information exposure per unit of compute), undertrained relative to what the same token budget would have achieved against a cleanly deduplicated corpus of equivalent raw size. Deduplication is, from this angle, not merely a data-quality nicety but a direct lever on how efficiently a fixed compute budget is converted into model capability.

## 7. The Staff-Level Judgment Call

Every technique in this file — the quality classifier, perplexity filtering, the heuristic funnel, exact and fuzzy dedup — is, underneath its mechanical description, an instance of the same underlying tension: how aggressively should the pipeline discard data in the name of quality and non-redundancy, versus how much noise, redundancy, and stylistic diversity should it simply tolerate for the sake of scale and representativeness. Every one of these filters is calibrated against a proxy for quality — resemblance to Wikipedia or curated reference text, low perplexity under a reference LM, absence of boilerplate signatures, absence of near-duplicate matches — and every proxy has a systematic bias baked into what it rewards and what it inadvertently penalizes. A classifier trained to recognize Wikipedia-like prose will, by construction, downweight registers that are legitimately informative but stylistically distant from Wikipedia: informal writing, non-native English, code-mixed multilingual text, minority-dialect speech, niche technical jargon that a shallow n-gram classifier has never been positively trained to recognize as high-quality. Aggressive filtering tuned purely to maximize a proxy-quality score risks systematically narrowing the training distribution's representational diversity in ways that surface later as disparate model performance across dialects, registers, and populations — a cost that does not show up cleanly in a filtering pipeline's own internal validation metrics, since those metrics are typically defined in terms of the same proxy the filter was built to optimize. At the same time, tolerating too much noise for the sake of preserving scale and diversity dilutes the training signal with material that measurably degrades downstream quality, wastes compute on redundant or content-free tokens (Section 6), and increases memorization and contamination risk. There is no purely mechanical answer to where the right operating point sits — it is a judgment call that depends on the model's intended use, the compute budget available, and an organization's tolerance for the concrete, if hard-to-quantify, fairness costs of representational narrowing versus the concrete, more measurable costs of training on noisier data, and being able to articulate that tension explicitly, rather than presenting any single filtering technique as a solved, judgment-free step, is exactly the signal a staff-level conversation on this topic is probing for.
