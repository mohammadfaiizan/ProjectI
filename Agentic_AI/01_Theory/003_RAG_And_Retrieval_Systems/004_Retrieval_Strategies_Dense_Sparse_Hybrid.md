# Retrieval Strategies: Dense, Sparse, and Hybrid Search

## Why Retrieval Strategy Is Its Own Design Decision

Chapter 3 covered how bi-encoders and cross-encoders turn text into comparable representations. This chapter is about a different, equally consequential decision: given that you can produce embeddings, should retrieval actually run on embeddings at all, or should it run on the raw lexical content of the documents, or both at once? This is not a settled question with one obviously correct answer — it is one of the most common whiteboard discussions in senior RAG interviews, precisely because production systems that ship with "just use vector search" quietly fail on a predictable, recurring class of query, and the fix is architectural rather than a matter of picking a better embedding model.

The three strategies in play are sparse retrieval (keyword and term-overlap based, with BM25 as the dominant modern algorithm), dense retrieval (embedding-based nearest-neighbor search, built on exactly the bi-encoders from Chapter 3), and hybrid retrieval, which runs both and fuses the results. Understanding why hybrid retrieval is now the default recommendation in almost every serious production RAG system — rather than a nice-to-have — requires understanding what each of the two underlying approaches is structurally good and bad at, which is where this chapter starts.

## Sparse Retrieval Fundamentals

### TF-IDF as the Starting Point

Sparse retrieval represents documents and queries as high-dimensional vectors where each dimension corresponds to a distinct term in the vocabulary, and almost every dimension is zero for any given document (hence "sparse" — a document only has nonzero values for the handful of terms it actually contains, out of a vocabulary that might span hundreds of thousands of terms). The classic scoring scheme for this representation is TF-IDF: term frequency multiplied by inverse document frequency.

Term frequency (TF) is simply how often a term appears in a document — the intuition being that a document mentioning "kubernetes" five times is more likely to be about Kubernetes than a document mentioning it once. Inverse document frequency (IDF) downweights terms that appear in most documents in the corpus (like "the," "system," or "data" in a technical corpus) and upweights terms that appear in only a few documents, because rare terms carry more discriminative signal about what makes a specific document distinctive. The classic IDF formula is `log(N / df_t)`, where `N` is the total number of documents and `df_t` is the number of documents containing term `t`. Multiply TF and IDF together, sum across the terms shared between query and document, and you have a basic relevance score.

TF-IDF is worth knowing conceptually because it's the ancestor of everything that follows, but it has a well-known flaw that BM25 was specifically designed to fix: raw term frequency scales the score linearly and without bound. A document that mentions the query term 100 times scores ten times higher than a document that mentions it 10 times, even though in practice the relevance difference between "mentions it 10 times" and "mentions it 100 times" is nowhere near a 10x difference in how relevant a human would judge the document — after the first several mentions, additional repetitions tell you very little more about aboutness, and past some point can even signal keyword-stuffed, low-quality content rather than genuine relevance.

### BM25: Term Frequency Saturation and Length Normalization

BM25 (Best Matching 25, from the Okapi information retrieval system where it originated) is the industry-standard sparse retrieval algorithm, and it fixes TF-IDF's linear-scaling problem with two specific, tunable mechanisms: term frequency saturation and document length normalization. It remains the default lexical scorer inside Elasticsearch, OpenSearch, and virtually every hybrid search product on the market, which is why "explain BM25" is such a recurring interview question — it's not legacy trivia, it's live production infrastructure.

The BM25 score for a document `D` given a query `Q` with terms `q_1 ... q_n` is:

```
score(D, Q) = sum over each query term q_i of:
    IDF(q_i) * ( f(q_i, D) * (k1 + 1) ) / ( f(q_i, D) + k1 * (1 - b + b * |D| / avgdl) )
```

where `f(q_i, D)` is how many times term `q_i` appears in document `D`, `|D|` is the length of document `D` in terms, and `avgdl` is the average document length across the corpus. The two parameters that make this formula behave so differently from TF-IDF are `k1` and `b`, and understanding exactly what each one controls is the crux of the BM25 interview question.

**`k1` controls term frequency saturation.** It governs how quickly the score's sensitivity to additional term occurrences flattens out. As `f(q_i, D)` grows large, the fraction `f / (f + k1)` asymptotically approaches 1, meaning additional occurrences of the term contribute rapidly diminishing marginal score — the tenth occurrence of a term barely moves the needle compared to the first or second. This directly encodes the intuition from the TF-IDF discussion above: relevance from term frequency should have diminishing returns, not linear returns. `k1` is typically set between 1.2 and 2.0; a lower `k1` saturates faster (occurrences 3 through 10 matter almost as little as occurrence 2), while a higher `k1` behaves closer to raw, unsaturated term frequency. A `k1` of 0 would make the formula ignore term frequency entirely beyond binary presence/absence.

**`b` controls document length normalization**, and it addresses a separate problem: long documents naturally accumulate more term matches simply by containing more words, not necessarily because they are more relevant. A 10,000-word document has vastly more opportunities to mention a query term at least once than a 200-word document does, purely as a function of length, independent of topical focus. The `|D| / avgdl` ratio compares a document's length to the corpus average — documents longer than average get their term frequency scores dampened, and documents shorter than average get relatively less dampening. `b` ranges from 0 (no length normalization at all — long documents get no penalty) to 1 (full length normalization), and 0.75 is the widely used default that BM25's original authors settled on and that most search engines ship with out of the box. Setting `b` too high over-penalizes legitimately long, comprehensive documents; setting it too low lets verbose documents win purely on volume.

The combination of these two mechanisms is what makes BM25 qualitatively better calibrated than raw TF-IDF: it rewards genuine term relevance while resisting both the "repeat the keyword 50 times" exploit and the "just write a longer document" exploit, without needing any learned parameters or training data — it's a closed-form statistical formula computable directly from corpus statistics.

A small worked example makes the saturation effect concrete. Suppose `k1 = 1.5`, a document is exactly average length (so the length-normalization factor `1 - b + b*|D|/avgdl` reduces to 1), and IDF for the query term is held fixed. At `f = 1` occurrence, the term-frequency component `f*(k1+1) / (f+k1)` evaluates to `1*2.5/2.5 = 1.0`. At `f = 2`, it's `2*2.5/3.5 ≈ 1.43`. At `f = 10`, it's `10*2.5/11.5 ≈ 2.17`. At `f = 100`, it's `100*2.5/101.5 ≈ 2.46`. Going from 1 to 2 occurrences nearly doubles the contribution, but going from 10 to 100 occurrences — a 10x increase in raw term frequency — barely moves the score from 2.17 to 2.46, a roughly 13% increase. That flattening curve, bounded above by `k1 + 1 = 2.5` no matter how many times the term appears, is term frequency saturation made numeric.

### BM25 From Scratch

Seeing the formula implemented directly makes the interaction between term frequency, IDF, and length normalization concrete in a way the equation alone doesn't:

```python
import math
from collections import Counter
from typing import List


class BM25:
    """A from-scratch BM25 scorer, built directly from the standard formula.
    k1 controls term-frequency saturation, b controls length normalization."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(corpus)
        self.doc_freqs = [Counter(doc) for doc in corpus]  # term counts per doc
        self.n_docs = len(corpus)
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict:
        # df_t = number of documents containing term t at least once
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1

        idf = {}
        for term, freq in df.items():
            # +1 smoothing in numerator/denominator avoids negative IDF
            # for terms that appear in more than half the corpus
            idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )
        return idf

    def score(self, query_terms: List[str], doc_index: int) -> float:
        doc_freqs = self.doc_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0

        for term in query_terms:
            if term not in doc_freqs:
                continue
            f = doc_freqs[term]
            idf = self.idf.get(term, 0.0)

            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            score += idf * (numerator / denominator)

        return score

    def search(self, query: List[str], top_k: int = 5) -> List[tuple]:
        scores = [
            (i, self.score(query, i)) for i in range(self.n_docs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# Example corpus, already tokenized (in practice: lowercase, strip
# punctuation, and run a real tokenizer rather than str.split())
corpus = [
    "the api returns error 429 rate limit exceeded".split(),
    "our service throttles requests when you exceed quota".split(),
    "kubernetes pods restart automatically after a crash".split(),
    "rate limiting protects backend services from overload".split(),
]

bm25 = BM25(corpus, k1=1.5, b=0.75)
results = bm25.search("rate limit error".split(), top_k=4)
for idx, score in results:
    print(f"doc={idx}  score={score:.4f}  text={' '.join(corpus[idx])}")
```

For anything beyond a learning exercise, use a maintained library rather than the from-scratch version above — the popular `rank_bm25` package implements the same formula (with a couple of variant flavors: `BM25Okapi`, `BM25L`, `BM25Plus`) and is what most prototypes reach for before graduating to a full search engine like Elasticsearch or OpenSearch for production scale:

```python
from rank_bm25 import BM25Okapi

tokenized_corpus = [doc.split() for doc in [
    "the api returns error 429 rate limit exceeded",
    "our service throttles requests when you exceed quota",
    "kubernetes pods restart automatically after a crash",
    "rate limiting protects backend services from overload",
]]

bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
scores = bm25.get_scores("rate limit error".split())
print(scores)  # array of per-document scores, same formula as above
```

## Dense Retrieval

### How It Works

Dense retrieval builds directly on the bi-encoder architecture from Chapter 3: a single encoder (or a matched pair of query/document encoders) maps both the query and every document in the corpus into the same fixed-dimensional vector space, ahead of time for documents and at query time for the query. Retrieval then becomes a nearest-neighbor search — find the document vectors closest to the query vector by cosine similarity or dot product, exactly the metrics covered in the embeddings chapter. Because the corpus is embedded once and stored (typically in a vector index such as HNSW), and only the query needs to be embedded live, the online cost of a dense retrieval query is a single forward pass through the query encoder plus a nearest-neighbor lookup — cheap and highly parallelizable, which is part of why dense retrieval scales well to large corpora.

The property that makes dense retrieval valuable is exactly the property sparse retrieval structurally lacks: dense vectors encode meaning rather than surface tokens, so a query and a document can be scored as highly relevant even when they share almost no literal vocabulary. A query embedding for "how do I stop my API from getting overloaded" and a document embedding for "rate limiting protects backend services from overload" can end up close together in embedding space because the encoder was trained (via contrastive learning on query-passage pairs) to associate their meanings, even though the only shared token between them is "overload." This is precisely the class of match that pure term-overlap scoring like BM25 cannot make at all — if there's no shared vocabulary, BM25's score contribution from those terms is exactly zero, because BM25 has no mechanism for recognizing synonymy or paraphrase; it only counts shared tokens weighted by frequency and rarity. Dense retrieval also naturally handles cross-lingual matching (a well-trained multilingual encoder can map a French query near an English passage discussing the same concept) and conceptual relationships that don't reduce to any single shared term at all, such as a query about "reducing cloud spend" retrieving a passage about "right-sizing EC2 instances" purely because the encoder learned that these concepts co-occur in relevant training pairs.

### A Working Dense Retriever

```python
import numpy as np
from typing import List, Tuple


class DenseRetriever:
    """Embeds a corpus once, then answers queries via cosine similarity
    nearest-neighbor search. In production this embedding step is done
    by a real bi-encoder (e.g., a sentence-transformers or OpenAI model)
    and the nearest-neighbor search is delegated to an ANN index (HNSW,
    IVF) rather than the brute-force loop shown here."""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.doc_texts: List[str] = []
        self.doc_vectors: np.ndarray = None

    def index(self, documents: List[str]) -> None:
        self.doc_texts = documents
        vectors = np.array([self.embed_fn(doc) for doc in documents])
        # Normalize once at index time so cosine similarity reduces to
        # a plain dot product at query time (see Chapter 3 discussion
        # of normalize-once, dot-product-everywhere).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = vectors / norms

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_vec = np.array(self.embed_fn(query))
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity against every normalized doc vector, in one
        # matrix-vector product -- this is the brute-force equivalent
        # of what an ANN index approximates efficiently at scale.
        scores = self.doc_vectors @ q_vec
        ranked_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in ranked_idx]


# Stand-in embedding function -- swap for a real sentence-transformers
# or API-based embedding model in production.
def fake_embed(text: str, dim: int = 64) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=dim)

documents = [
    "our service throttles requests when you exceed quota",
    "kubernetes pods restart automatically after a crash",
    "rate limiting protects backend services from overload",
    "the api returns error 429 rate limit exceeded",
]

retriever = DenseRetriever(embed_fn=fake_embed)
retriever.index(documents)
for idx, score in retriever.search("why do I keep getting throttled", top_k=4):
    print(f"doc={idx}  cosine={score:.4f}  text={documents[idx]}")
```

(The `fake_embed` stand-in is used only so this snippet runs standalone without a model download; it produces deterministic pseudo-random vectors per string and does not capture real semantics. In a real system this line is replaced by a call to an actual bi-encoder, and the resulting similarity ranking would reflect genuine semantic closeness rather than hash noise.)

## Hybrid Search: Combining Sparse and Dense

### Why Hybrid Outperforms Either Alone

Sparse and dense retrieval fail in complementary, non-overlapping ways, which is exactly why combining them tends to produce a system that is more robust than either individually rather than merely "averaging" their strengths and weaknesses. BM25 fails when the relevant document uses different words than the query, no matter how conceptually close the meanings are. Dense retrieval fails — often silently and confidently — when a query depends on exact tokens the embedding model has no strong reason to treat as distinctive: identifiers, codes, numbers, acronyms, and rare proper nouns are frequently compressed by the encoder into similar regions of the vector space as other, unrelated but superficially similar-looking tokens, because the encoder was trained on natural language semantics, not on preserving exact-match precision for arbitrary strings. Because these failure modes are largely uncorrelated — the queries that break BM25 are generally not the same queries that break dense retrieval — running both and merging the results recovers relevant documents that either retriever alone would have missed or ranked too low to matter. This is a well-replicated empirical finding across retrieval benchmarks (BEIR, MS MARCO, and numerous production case studies), not just a theoretical argument: hybrid retrieval consistently beats the stronger of the two individual retrievers on recall and downstream answer quality, even when the dense retriever alone is already a strong, well-tuned model.

The practical challenge hybrid search has to solve is fusion: sparse and dense retrieval each produce a ranked list of candidate documents with their own scores, and those two score distributions are not directly comparable. There are two standard ways to reconcile them.

### Fusion Strategy 1: Weighted Score Combination

The first approach normalizes each retriever's raw scores onto a common scale — typically min-max normalization within the candidate set — and then combines them with a weighted sum: `final_score = alpha * normalized_dense_score + (1 - alpha) * normalized_sparse_score`, where `alpha` is a tunable hyperparameter controlling how much weight dense retrieval gets relative to sparse. This is intuitive and lets you bias the system toward one retriever if you have evidence (from evaluation data) that one is more reliable for your domain, but it has a real weakness: BM25 scores are unbounded and depend heavily on corpus statistics (vocabulary size, average document length, term rarity distribution), while cosine similarities are bounded and shaped by the anisotropy properties discussed in Chapter 3. Min-max normalizing two differently-shaped distributions doesn't make them statistically comparable, just numerically bounded — a BM25 score distribution that's heavily right-skewed with a long tail behaves very differently under min-max scaling than a cosine similarity distribution that's tightly clustered, and the resulting weighted sum can be dominated by whichever retriever happens to have a "spikier" score distribution on a given query, independent of which retriever is actually more trustworthy for that query.

### Fusion Strategy 2: Reciprocal Rank Fusion (RRF)

The second, and now more commonly deployed, approach sidesteps the score-comparability problem entirely by ignoring raw scores and fusing on rank position instead. Reciprocal Rank Fusion computes a fused score for each document as:

```
RRF_score(d) = sum over each ranked list i of:  1 / (k + rank_i(d))
```

where `rank_i(d)` is the position of document `d` in ranked list `i` (1-indexed, so the top result has rank 1), and `k` is a constant, typically set to 60, that dampens the influence of very high ranks and keeps the score from blowing up when `rank` is small. A document that appears near the top of both the sparse and dense rankings accumulates a high combined score; a document that appears in only one list still contributes a smaller but nonzero score from that list alone, so it isn't automatically excluded from consideration.

The reason RRF has become the default fusion mechanism in production hybrid search — used in Elasticsearch's hybrid retriever, OpenSearch, Weaviate, and Azure AI Search, among others — is exactly the robustness argument above: rank is a scale-free quantity. It doesn't matter whether BM25 produced a score of 4.2 or 42, or whether the dense retriever's cosine similarity was 0.31 or 0.91 — all that matters for RRF is where each document landed in its own list, and "landed at position 3" means the same thing regardless of which underlying scoring function produced that ordering. This makes RRF immune to the score-scale mismatch problem that plagues naive weighted-sum fusion, at the cost of throwing away magnitude information — RRF cannot distinguish "barely made rank 1" from "overwhelmingly, obviously rank 1," since both map to the same `1/(k+1)` contribution. In practice this tradeoff is usually worth it, because magnitude comparability across dense and sparse scores was never reliable to begin with. This chapter introduces RRF only functionally, as the fusion mechanism hybrid search needs to operate; the internals, alternative fusion formulas, and tuning of `k` and other parameters are covered in depth in Chapter 6 alongside reranking.

### A Working Hybrid Scorer

```python
from collections import defaultdict
from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]], k: int = 60
) -> Dict[int, float]:
    """Fuse multiple ranked lists of document IDs into one score per
    document using Reciprocal Rank Fusion. Each input list is assumed
    to already be sorted best-first. See Chapter 6 for RRF internals
    and tuning guidance -- this is the minimal functional version
    hybrid search needs to combine sparse and dense rankings."""
    fused_scores: Dict[int, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1.0 / (k + rank)
    return fused_scores


class HybridRetriever:
    """Runs BM25 and dense retrieval independently over the same
    corpus, over-fetches candidates from each, and fuses with RRF."""

    def __init__(self, bm25: BM25, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(
        self, query: str, top_k: int = 5, fetch_k: int = 50
    ) -> List[Tuple[int, float]]:
        # Run both retrievers independently, each over-fetching a
        # larger candidate pool than the final top_k requires.
        sparse_hits = self.bm25.search(query.split(), top_k=fetch_k)
        dense_hits = self.dense.search(query, top_k=fetch_k)

        sparse_ranked_ids = [doc_id for doc_id, _ in sparse_hits]
        dense_ranked_ids = [doc_id for doc_id, _ in dense_hits]

        fused = reciprocal_rank_fusion(
            [sparse_ranked_ids, dense_ranked_ids], k=self.rrf_k
        )

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


documents = [
    "the api returns error 429 rate limit exceeded",
    "our service throttles requests when you exceed quota",
    "kubernetes pods restart automatically after a crash",
    "rate limiting protects backend services from overload",
]

bm25 = BM25([doc.split() for doc in documents])
dense = DenseRetriever(embed_fn=fake_embed)
dense.index(documents)

hybrid = HybridRetriever(bm25, dense, rrf_k=60)
for doc_id, score in hybrid.search("why do I keep getting throttled", top_k=4, fetch_k=4):
    print(f"doc={doc_id}  rrf_score={score:.4f}  text={documents[doc_id]}")
```

## When Sparse Wins, and When Dense Wins

The two retrieval modes are not interchangeable options where one is generally "better" — they win on structurally different query types, and knowing which is which, with concrete examples, is exactly what a hybrid architecture is designed to hedge against.

Sparse retrieval wins decisively — often dramatically so — on exact keyword lookups: product SKUs, order IDs, error codes, API status strings, acronyms, and other rare, precise tokens where the literal string match itself is the entire signal. A user searching a support knowledge base for `ERR_429_RATE_LIMIT` needs the document that contains that exact string, and BM25 will find it immediately: the term is rare in the corpus (high IDF), so any document containing it scores very highly on that term alone. Dense retrieval is comparatively unreliable here, because an embedding model has no strong training signal that `ERR_429_RATE_LIMIT` is meaningfully different from `ERR_500_INTERNAL` or `ERR_403_FORBIDDEN` — to a tokenizer and encoder that was trained on natural language rather than on distinguishing your specific error-code taxonomy, these can embed close enough together that the wrong error code's documentation gets retrieved instead of the right one. The same failure pattern shows up for part numbers, legal citation formats, proper nouns the model saw rarely during pretraining, and any identifier where "close in embedding space" does not imply "the same entity."

Dense retrieval wins on the mirror-image case: queries phrased conversationally, with little to no literal vocabulary overlap with the relevant passage. A user asking "why do I keep getting throttled" over the same support corpus shares essentially no tokens with a document titled "rate limiting protects backend services from overload" — no shared word except perhaps a stray function word — yet the two are obviously about the same underlying concept to a human reader, and a well-trained dense encoder captures that association because it was trained on exactly this kind of paraphrase relationship. The same advantage shows up for synonym substitution ("laptop won't turn on" retrieving a document about "device fails to power up"), cross-lingual queries against a corpus in a different language, and loosely worded natural-language questions where the user doesn't know or use the corpus's specific terminology at all.

Put together as a decision heuristic: if a query's information need hinges on an exact, rare token, sparse retrieval is doing the real work and dense retrieval is a coin flip at best; if a query's information need hinges on a concept that could be phrased many different ways, dense retrieval is doing the real work and sparse retrieval will often return nothing useful because there's no lexical overlap to score. Running both in parallel and fusing means the system doesn't have to correctly predict, ahead of time, which category a given query falls into — it lets both scoring mechanisms compete for every query and lets fusion surface whichever one actually found the right answer.

## Practical Hybrid Architecture in Production

Three architectural details separate a hybrid search implementation that works well in production from one that merely runs both retrievers and technically qualifies as "hybrid."

**Run both retrievers in parallel, not sequentially.** Sparse and dense retrieval are independent computations against independent indexes — there's no data dependency between them, so issuing them as sequential calls only adds their latencies together for no benefit. In practice this means firing both requests concurrently (async calls, a thread pool, or parallel requests to a search engine that supports hybrid queries natively) and waiting on both before fusion, so total retrieval latency is close to `max(sparse_latency, dense_latency)` rather than their sum:

```python
import asyncio


async def hybrid_search_parallel(bm25, dense, query: str, fetch_k: int = 50):
    """Fire the sparse and dense lookups concurrently instead of one
    after the other. In a real system bm25_search_async and
    dense_search_async wrap network calls to a search engine and a
    vector database respectively; here they just wrap the synchronous
    calls in a thread so the illustration runs without extra services."""
    loop = asyncio.get_event_loop()

    sparse_task = loop.run_in_executor(
        None, lambda: bm25.search(query.split(), top_k=fetch_k)
    )
    dense_task = loop.run_in_executor(
        None, lambda: dense.search(query, top_k=fetch_k)
    )

    # Total latency is ~max(sparse, dense), not sparse + dense.
    sparse_hits, dense_hits = await asyncio.gather(sparse_task, dense_task)

    sparse_ids = [doc_id for doc_id, _ in sparse_hits]
    dense_ids = [doc_id for doc_id, _ in dense_hits]
    return reciprocal_rank_fusion([sparse_ids, dense_ids], k=60)


# asyncio.run(hybrid_search_parallel(bm25, dense, "why do I keep getting throttled"))
```

**Over-fetch a larger candidate pool from each retriever before fusion.** If the final answer only needs the top 5 or top 10 documents, each individual retriever should still return something like its top 50 candidates (the `fetch_k` parameter in the `HybridRetriever` example above) rather than only its own top 5. This matters because RRF and weighted fusion are reranking mechanisms over the union of both candidate sets — if you truncate each retriever's output to the final `k` before fusing, you've already thrown away exactly the documents that one retriever ranked outside its own top-k but the other ranked highly, which defeats the entire purpose of hybrid retrieval. A document that BM25 ranks 30th and dense retrieval ranks 2nd should still have a real chance of making the final top 5 after fusion; that's only possible if BM25's candidate pool extended to rank 30 in the first place.

**Apply metadata and keyword filters as a pre-filter on both retrieval paths, not as a post-filter after fusion.** When a query needs to be scoped to a tenant, a date range, a document type, or an access-control boundary, that filter should be pushed down into both the sparse and dense retrieval calls themselves (most vector databases support filtered ANN search natively, and inverted indexes support filtered term queries just as naturally) rather than applied to the final fused, truncated result list. Filtering after truncation risks a scenario where the initial top-k candidate pool from one or both retrievers happens to be dominated by documents that get filtered out afterward — for example, if the top 50 dense hits are mostly from the wrong tenant, and the filter removes them post-hoc, the caller is left with far fewer than `k` usable results, having wasted the entire retrieval budget on candidates that were never eligible to answer the query. Pushing the filter down ensures the top-k budget from each retriever is spent entirely on documents that could actually be returned, which is both more efficient and produces measurably better final recall.
