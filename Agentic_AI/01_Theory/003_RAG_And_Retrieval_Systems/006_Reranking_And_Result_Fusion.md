# Reranking and Result Fusion

## 1. Reranking as a funnel: why a second stage exists at all

Chapters 3 and 4 established the tools of first-stage retrieval: bi-encoders that embed queries and documents independently into a shared vector space and compare them with a cheap dot product, sparse lexical methods like BM25 that score exact and near-exact term overlap, and hybrid approaches that combine both. Every one of those methods is built around the same non-negotiable constraint: it has to run over the *entire corpus*, which might be hundreds of thousands or hundreds of millions of chunks, and it has to do it in single-digit milliseconds. That constraint forces an architectural compromise. A bi-encoder can only compare a query vector against a document vector *after* the document has already been embedded in isolation, with no knowledge of what query it would eventually be compared against — that's precisely what makes it fast enough to index into an ANN structure and search in logarithmic or sub-linear time, but it's also what caps its precision. Two chunks that are topically related but subtly different in what they actually assert can end up with very similar embeddings, because the embedding was computed without ever looking at the specific question being asked. BM25 has a symmetric limitation from the opposite direction: it is excellent at exact lexical overlap but has no notion of meaning at all, so it can rank a document highly because it repeats a query term many times even when that document doesn't actually answer the question, and it can miss a document that answers the question perfectly using different words.

Cross-encoders, introduced structurally in Chapter 3, don't have this limitation, because they concatenate the query and a candidate document and run the pair jointly through a single transformer with full cross-attention between every query token and every document token. This lets the model directly reason about whether *this specific document* answers *this specific question*, rather than comparing two vectors that were computed independently. The catch is cost: a cross-encoder pass is not a single embedding lookup, it's a full forward pass of a transformer for every query-document pair, and that cost scales linearly with the number of pairs scored. Running a cross-encoder over an entire multi-million-document corpus for every query is computationally infeasible — it would turn a few-millisecond retrieval into a search that takes minutes to hours, per query.

This is exactly the mismatch that reranking, as a distinct pipeline stage, resolves. Instead of choosing one method that has to be either fast-over-everything or accurate-over-a-few, you use both, in sequence, exploiting the fact that each one's weakness is irrelevant in the regime where you actually deploy it. Think of the whole thing as a funnel:

```
Millions of chunks in the corpus
        │
        ▼  Stage 1: broad and cheap (bi-encoder ANN search, BM25, or hybrid — Ch. 3-4)
        │  Optimized for speed and recall over the entire corpus.
        │  Sacrifices some precision to stay fast at that scale.
        ▼
Top 20-100 candidates
        │
        ▼  Stage 2: narrow and expensive (cross-encoder reranker, or a fusion of
        │  several first-stage lists — this chapter)
        │  Now affordable precisely because the candidate set is tiny.
        ▼
Top 3-5 passages that actually go into the LLM's prompt
```

The key insight that makes this funnel work economically is that the expensive model's cost is now paid against dozens of candidates instead of millions, which is a difference of many orders of magnitude. Scoring 50 query-document pairs with a cross-encoder on a GPU is a task measured in tens of milliseconds; scoring 50 million pairs the same way would take hours. First-stage retrieval's entire job, in this framing, is not to find the *perfect* top result — it's to guarantee that the truly relevant documents survive somewhere in a shortlist small enough for the expensive stage to examine exhaustively. Reranking's job is then to take that shortlist, where recall is already good but precision (i.e., the *ordering*) may be mediocre, and fix the ordering using a model that can actually reason about relevance rather than approximate it geometrically. This division of labor — cheap-and-broad finds the right neighborhood, expensive-and-narrow picks the right house — is one of the most consistently effective patterns in applied information retrieval, and it long predates modern RAG; the same "retrieve-then-rerank" structure was standard practice in web search and enterprise search well before LLMs made it relevant to prompt construction.

It's also worth being explicit about why this matters more for RAG specifically than it did for classic search. In a search engine, a slightly worse ranking in position 4 versus position 2 is a UX annoyance — the user scans past it. In RAG, only the top handful of chunks (often 3-5) are physically included in the prompt sent to the LLM; anything ranked outside that cutoff simply never exists as far as the model is concerned, no matter how relevant it actually was. Reranking is therefore not a nice-to-have polish step — it is the mechanism that decides, with much higher fidelity than first-stage retrieval alone, exactly which few pieces of evidence the model will ever see.

## 2. Cross-encoder rerankers in depth

A cross-encoder reranker is a transformer, typically a fine-tuned BERT-family or similar encoder model, that takes a query and a single candidate document as one joint input — usually formatted as `[CLS] query [SEP] document [SEP]` — and outputs a single scalar relevance score. Because the query and document tokens attend to each other directly inside the same set of transformer layers, the model can pick up on fine-grained interactions that two independently-computed embeddings simply cannot represent: negation ("does NOT support"), numeric constraints ("under $50" versus "over $50"), entity disambiguation (two documents about different people named "Cook"), or the difference between a document that mentions a topic in passing versus one that actually answers the question about it.

**Training.** Cross-encoder rerankers are trained as relevance classifiers or rankers on labeled query-document pairs. The canonical public training set is MS MARCO, which provides real search-engine queries paired with human relevance judgments over passages, and this is the source of the widely-used `cross-encoder/ms-marco-*` model family. Training typically uses one of two loss formulations. A pairwise ranking loss (for example, a margin-based loss like RankNet or a simple binary cross-entropy over "is this document relevant to this query") teaches the model, given a relevant and a non-relevant document for the same query, to score the relevant one higher. A listwise loss goes further, optimizing an entire ranked list at once against a ranking-quality metric such as NDCG, which tends to produce better-calibrated orderings when there are many graded levels of relevance rather than a binary relevant/non-relevant label. Many of the strongest current open rerankers, such as the BGE reranker family, are trained with knowledge distillation on top of this: a larger, more expensive teacher model (or an LLM) is used to generate soft relevance labels or preference judgments over a huge number of query-document pairs, and the smaller cross-encoder is trained to match that teacher's judgments, which is far cheaper to scale than sourcing that much human annotation directly.

**Popular open models**, roughly ordered from lightweight to strong:

- `cross-encoder/ms-marco-MiniLM-L-6-v2` — a distilled, 6-layer MiniLM cross-encoder trained on MS MARCO. It is fast enough to run comfortably on CPU for small candidate sets and is a common default when latency budget is tight or GPU infrastructure isn't available; quality is solid but noticeably behind larger models on harder or more domain-specific queries.
- `BAAI/bge-reranker-large` — a larger, stronger open cross-encoder from the BGE family, trained with more extensive multi-task and distillation objectives; a common choice when quality matters more than raw latency and a GPU is available.
- `BAAI/bge-reranker-v2-m3` — a newer, multilingual-capable BGE reranker built on the M3 backbone, competitive with or ahead of many hosted rerankers on public benchmarks and a strong choice for teams that want state-of-the-art open-source reranking without relying on an external API.
- Jina Reranker (e.g., `jina-reranker-v2-base-multilingual`) — another strong open/commercially-licensed option, notable for long-context support (useful when candidate "documents" are actually large chunks) and multilingual coverage.

**Working implementation.** `sentence-transformers` wraps cross-encoder models behind a simple `CrossEncoder` interface that takes a list of `(query, document)` pairs and returns scores, so reranking a candidate list is a small amount of code on top of whatever first-stage retrieval already produced.

```python
"""
cross_encoder_rerank.py

Rerank a first-stage candidate list with a cross-encoder.

Dependencies: sentence-transformers, torch
    pip install sentence-transformers torch
"""

from dataclasses import dataclass
from typing import List
from sentence_transformers import CrossEncoder

# A lightweight model for low-latency use; swap for "BAAI/bge-reranker-large"
# or "BAAI/bge-reranker-v2-m3" when quality matters more than raw speed.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL, max_length=512)


@dataclass
class Candidate:
    doc_id: str
    text: str
    first_stage_score: float  # bi-encoder cosine sim or BM25 score, for reference


def rerank(query: str, candidates: List[Candidate], top_n: int = 5) -> List[Candidate]:
    """
    Score every (query, candidate) pair jointly through the cross-encoder and
    return the top_n candidates sorted by that score, descending.

    Note: this scores every candidate independently and exhaustively, which
    is exactly why reranking only works on a small shortlist (tens to low
    hundreds of candidates) rather than the whole corpus.
    """
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)  # one scalar relevance logit per pair

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked = []
    for candidate, score in scored[:top_n]:
        # Attach the reranker score for downstream logging/inspection without
        # mutating the original first-stage score.
        candidate.rerank_score = float(score)
        reranked.append(candidate)
    return reranked


if __name__ == "__main__":
    query = "What is the refund window for a physical product?"

    # Pretend these came out of Chapter 4's hybrid first-stage retrieval,
    # already narrowed from millions of chunks down to a shortlist.
    candidates = [
        Candidate("c1", "Digital products are non-refundable once downloaded.", 0.71),
        Candidate("c2", "Physical products can be refunded in full within 30 days of purchase.", 0.68),
        Candidate("c3", "Store credit from returns is valid for 12 months.", 0.74),
        Candidate("c4", "Shipping normally takes 3 to 5 business days domestically.", 0.60),
    ]

    top = rerank(query, candidates, top_n=2)
    for c in top:
        print(f"{c.doc_id}  rerank_score={c.rerank_score:.3f}  first_stage={c.first_stage_score:.3f}")
```

Note the outcome this example is designed to illustrate: `c3` had the highest first-stage cosine similarity (0.74) because it shares vocabulary like "refund" and sits near the query in embedding space, but it doesn't actually answer *this* question about the refund window for physical products — `c2` does, explicitly. A cross-encoder, seeing the query and each candidate jointly, is far more likely to correctly promote `c2` above `c3` than a bi-encoder ever could, because the bi-encoder committed to `c3`'s embedding before it ever saw this query.

## 3. Hosted rerank APIs

Self-hosting a cross-encoder means owning a GPU-backed inference service, keeping the model and its serving stack (batching, autoscaling, monitoring) up to date, and absorbing the engineering cost of retraining or swapping models as better ones are released. For many teams, especially those without dedicated ML infra, this is a disproportionate amount of operational overhead for what is conceptually a single scoring function. Hosted rerank APIs exist to remove exactly that burden: you send a query and a list of candidate documents, optionally cap the number of results with `top_n`, and get back the same documents sorted by relevance with scores attached — no model to host, no GPU to provision, no batching logic to write.

Cohere's `rerank-v3.5` is the most commonly referenced example of this category (Jina and Voyage AI offer comparable hosted rerank endpoints with the same basic contract). The operational appeal is threefold: it is a single HTTP call rather than a model deployment; the provider can iterate on and upgrade the underlying model transparently, so callers benefit from quality improvements without doing any retraining themselves; and it typically supports many languages and long documents out of the box, which would otherwise require deliberately sourcing a multilingual or long-context cross-encoder.

```python
"""
hosted_rerank_example.py

Call a Cohere-style hosted rerank endpoint to reorder a first-stage
candidate list. Requires a COHERE_API_KEY environment variable.

Dependencies: cohere
    pip install cohere
"""

import os
import cohere

co = cohere.Client(os.environ["COHERE_API_KEY"])


def hosted_rerank(query: str, documents: list[str], top_n: int = 5) -> list[dict]:
    """
    Send a query and a candidate document list to the hosted reranker and
    return the top_n results sorted by relevance, with the provider's
    relevance score and the original candidate index attached.
    """
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n,
    )

    return [
        {
            "index": result.index,          # position in the original `documents` list
            "text": documents[result.index],
            "relevance_score": result.relevance_score,
        }
        for result in response.results
    ]


if __name__ == "__main__":
    query = "What is the refund window for a physical product?"
    documents = [
        "Digital products are non-refundable once downloaded.",
        "Physical products can be refunded in full within 30 days of purchase.",
        "Store credit from returns is valid for 12 months.",
        "Shipping normally takes 3 to 5 business days domestically.",
    ]

    for r in hosted_rerank(query, documents, top_n=2):
        print(f"[{r['index']}] score={r['relevance_score']:.4f}  {r['text']}")
```

The trade-off against self-hosting is straightforward and worth stating precisely rather than treating one option as universally better. A hosted call adds a network round trip on top of the model's own inference time, which for a latency-sensitive query path is a real cost — self-hosted cross-encoders on local GPU hardware can often respond faster simply by avoiding that network hop, and self-hosting also avoids per-call pricing, which matters at high query volume. A hosted API also means sending your query text (and possibly document content) to a third party, which can be a non-starter under some data-residency or confidentiality requirements. Against that, hosted rerankers remove GPU provisioning, batching, and model-upgrade work entirely, and providers competing in this space have strong incentive to keep their models at or near the state of the art — which is a level of ongoing quality investment that most application teams cannot realistically replicate by fine-tuning and maintaining their own reranker in house.

| Dimension | Self-hosted cross-encoder | Hosted rerank API |
|---|---|---|
| Latency | Model inference only, no network hop | Model inference plus network round trip |
| Cost model | Fixed GPU/infra cost, scales with your own capacity planning | Per-call or per-document pricing, scales automatically with traffic |
| Ops burden | You own provisioning, batching, scaling, monitoring | None — a single API call |
| Model currency | You decide when (and whether) to retrain or upgrade | Provider upgrades the model transparently |
| Data residency | Documents never leave your infrastructure | Query and document text sent to a third party |
| Best fit | High query volume, strict data locality, existing GPU infra | Low-to-moderate volume, latency-tolerant, no hard residency constraint |

In practice the right choice tracks query volume and data sensitivity far more than raw quality, since the strongest open models (`bge-reranker-v2-m3` in particular) are already competitive with hosted offerings on public benchmarks.

## 4. Maximal Marginal Relevance for diversity

Reranking so far has optimized purely for relevance to the query, and that is usually the right primary objective — but pure relevance ranking has a failure mode of its own. If a corpus contains five documents that all restate essentially the same fact (a common outcome when multiple versions of a policy document, or several near-duplicate FAQ entries, exist across a knowledge base), a relevance-only ranker will happily put all five near the top, because they are all, individually, highly relevant. The result is a top-k list that is redundant rather than informative: instead of using the LLM's limited context budget to show it five different facets of the answer, you've spent it re-showing the same fact five times in slightly different words. This is doubly harmful — it wastes context tokens that could have carried genuinely new information, and it can create a false sense of confidence or consensus in the generated answer (five sources "agreeing" because they are actually the same source) when what the model actually needed was broader coverage of the topic.

Maximal Marginal Relevance (MMR), introduced by Carbonell and Goldstein (1998) in the context of document summarization and search-result diversification, addresses this directly by making result selection a function of two competing signals: how relevant a candidate is to the query, and how *different* it is from what has already been selected. Its formula, applied greedily one selection at a time, is:

```
MMR = argmax over d in Remaining [ λ * Sim(d, query) − (1 − λ) * max Sim(d, s) for s in Selected ]
```

At each step, you don't just pick the single highest-relevance remaining candidate; you pick the candidate that maximizes relevance to the query *minus* a penalty proportional to how similar it is to the most similar document already chosen. The `λ` parameter tunes the balance: `λ = 1` collapses MMR back to plain relevance ranking with no diversity penalty at all, while `λ = 0` ignores relevance entirely and just picks whatever is most different from what's already selected (a pure diversity objective, rarely useful on its own since it can surface irrelevant results). Values in the middle, commonly somewhere around 0.5 to 0.7 in practice, favor relevance but actively suppress near-duplicates once a fact has already been represented in the selected set.

```python
"""
mmr_selection.py

Greedy Maximal Marginal Relevance selection for diversifying a
relevance-ranked candidate list before it is handed to an LLM.

Dependencies: numpy
    pip install numpy
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ScoredCandidate:
    doc_id: str
    text: str
    embedding: np.ndarray   # assumed already embedded (Chapter 3), unit-normalized
    query_similarity: float  # Sim(doc, query), e.g. cosine similarity or rerank score


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def mmr_select(
    candidates: List[ScoredCandidate],
    k: int,
    lambda_param: float = 0.6,
) -> List[ScoredCandidate]:
    """
    Greedily select k candidates using Maximal Marginal Relevance.

    lambda_param close to 1.0 -> behaves like plain relevance ranking.
    lambda_param close to 0.0 -> behaves like pure diversity selection.
    """
    remaining = list(candidates)
    selected: List[ScoredCandidate] = []

    while remaining and len(selected) < k:
        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:
            relevance_term = lambda_param * candidate.query_similarity

            if selected:
                # Penalize similarity to whichever already-selected doc is
                # closest to this candidate -- that's the redundancy signal.
                redundancy = max(
                    cosine_sim(candidate.embedding, s.embedding) for s in selected
                )
            else:
                redundancy = 0.0  # nothing selected yet, so no penalty

            diversity_term = (1 - lambda_param) * redundancy
            mmr_score = relevance_term - diversity_term

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


if __name__ == "__main__":
    # Three of these candidates are near-duplicate restatements of the same
    # refund fact; one is genuinely different (shipping). Pure relevance
    # ranking would put all three refund variants ahead of the shipping one.
    def fake_embed(text: str) -> np.ndarray:
        # Toy stand-in for a real embedding model, deterministic per string.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=16)

    candidates = [
        ScoredCandidate("c1", "Full refunds are available within 30 days of purchase.", fake_embed("refund-a"), 0.91),
        ScoredCandidate("c2", "Customers can get a complete refund inside a 30-day window.", fake_embed("refund-a") + 0.01, 0.90),
        ScoredCandidate("c3", "Refunds in full are honored up to 30 days after buying.", fake_embed("refund-a") + 0.02, 0.89),
        ScoredCandidate("c4", "International shipments take 7 to 21 business days.", fake_embed("shipping-b"), 0.72),
    ]

    diversified = mmr_select(candidates, k=3, lambda_param=0.6)
    for c in diversified:
        print(c.doc_id, "-", c.text)
```

In this toy example, `c1`, `c2`, and `c3` are near-duplicate embeddings by construction (they're all perturbations of the same base vector), so after `c1` is selected first on relevance, the redundancy penalty pushes `c2` and `c3` down and lets `c4` — lower relevance but genuinely new information — into the final set ahead of one of them, despite `c4` having the lowest raw query similarity of any candidate. That is exactly the outcome MMR is designed to produce: broader, less redundant coverage within a fixed context budget, at the cost of not always picking the single highest-relevance items. MMR is typically applied as a final diversity pass after relevance ranking or cross-encoder reranking has already narrowed and ordered the candidate set — it answers "which of these already-good candidates should I actually show the model together," not "which candidates are relevant in the first place."

## 5. Reciprocal Rank Fusion, in depth

Chapter 4 introduced Reciprocal Rank Fusion briefly, in the context of combining a dense retriever's ranked list with BM25's ranked list into one hybrid result. Here is the full picture, because RRF is one of the most quietly important techniques in a modern retrieval stack, and it shows up not just for dense+sparse fusion but for combining any number of ranked lists — including a reranker's own output as one more list to fuse.

**The formula.** For a document `d` that appears in one or more ranked lists, RRF's fused score is:

```
score(d) = Σ_i  1 / (k + rank_i(d))
```

summed over every ranked list `i` in which `d` appears, where `rank_i(d)` is `d`'s position in list `i` (1-indexed: the top result has rank 1), and `k` is a small constant, conventionally 60. If a document doesn't appear in a given list at all, it simply contributes zero from that list rather than being penalized further — recall from Chapter 4 that this is exactly what makes RRF an easy way to combine BM25's top results with a dense retriever's top results even when they overlap only partially.

**Why the constant `k` matters.** Without `k` (i.e., using plain `1/rank`), a document ranked #1 in one list contributes a full point of score, while a document ranked #2 in that same list contributes only 0.5 — a massive relative drop for moving down a single position. That means a single list's rank-1 result can dominate the fused ranking almost regardless of what any other list says, which defeats the purpose of fusing multiple opinions in the first place: you wanted a consensus across retrievers, not a system where whichever retriever happens to rank something #1 wins by default. Adding `k = 60` in the denominator flattens this curve: `1/(60+1) ≈ 0.0164` versus `1/(60+2) ≈ 0.0161`, a difference of under 2% between rank 1 and rank 2, instead of the 50% swing you'd see without `k`. This damping is precisely the point — it means the difference between being ranked #1 and #2 in a single list barely matters to the fused score, but *consistently* ranking in the top handful across multiple independent lists compounds into a clearly higher fused score than ranking well in only one list. `k = 60` was the value used in the original RRF paper (Cormack, Clarke, and Buettcher, 2009) and has proven robust enough across domains that it is treated as a sensible default rather than something that typically needs tuning per deployment; smaller `k` sharpens the influence of top ranks, larger `k` flattens it further.

**Why rank-based, not score-based.** The alternative to RRF would be a weighted linear combination of each retriever's raw scores — for example, `final_score = 0.5 * bm25_score + 0.5 * cosine_similarity`. This runs into an immediate problem: BM25 scores are unbounded and depend on corpus statistics (term frequencies, document lengths, the specific IDF weighting variant used), cosine similarity is bounded in `[-1, 1]` and depends on the embedding model's geometry, and a cross-encoder's output might be an unbounded logit or a sigmoid-squashed probability depending on how it was trained. These numbers are not on comparable scales, do not have comparable distributions, and are not even guaranteed to be monotonically related to "true relevance" in the same way across retrievers. Averaging or weighting them directly is combining apples and oranges — a BM25 score of 12.4 and a cosine similarity of 0.83 have no shared numerical meaning, and naively summing them lets whichever retriever happens to produce larger-magnitude numbers dominate the combination regardless of actual quality. RRF sidesteps this entirely by discarding scores and working only with each document's *position* in each list. Rank is universally comparable across any retriever, no matter how it computed its internal score, which is exactly why RRF can fuse BM25, a dense bi-encoder, and a cross-encoder reranker's output in one formula without ever needing to know or calibrate their underlying score distributions.

| | Weighted score combination | Reciprocal Rank Fusion |
|---|---|---|
| Input | Raw scores from each retriever | Only each document's rank position |
| Requires comparable score scales | Yes — scores must be normalized or calibrated first | No — ranks are inherently comparable |
| Free parameters to tune | A weight per retriever, re-validated as retrievers change | Only `k`, conventionally fixed at 60 |
| Preserves magnitude of "how much better" | Yes, if scores are meaningfully calibrated | No — a landslide win and a near-tie both collapse to "rank 1" |
| Typical use | When scores are already on a shared, calibrated scale (e.g., two rerankers with the same training objective) | Default choice for fusing heterogeneous retrievers (BM25, dense, reranker) |

**Why it needs no tuned weights.** A weighted-score combination requires deciding, and periodically re-validating, how much to trust each retriever relative to the others — should dense retrieval count for 40% or 60% of the final score? That weight is a hyperparameter that needs tuning data, and it can silently go stale as either retriever is upgraded. RRF has no equivalent free parameter beyond `k` (which, as discussed, rarely needs tuning) — every list contributes to the fused score in the same functional form, so there is no per-retriever weight to lose track of or overfit. This is a large part of why RRF is treated as the sensible default for combining retrieval signals: it produces reasonable fused rankings out of the box, with no calibration step, across a wide range of retriever combinations.

**Limitations.** The same property that makes RRF robust — discarding raw scores in favor of rank — is also its central limitation: it throws away *magnitude* information. A document that a dense retriever scored with overwhelming confidence (a cosine similarity of 0.95 versus a runner-up at 0.40, a landslide) and a document that another retriever's top result barely edged out its runner-up (0.51 versus 0.49, a virtual tie) are both simply "rank 1" as far as RRF is concerned, and are treated completely identically in the fusion. If the underlying score distribution actually carried useful signal — the landslide winner really is far more confidently relevant than a near-tied one — RRF has no way to express that, because everything is flattened through the rank transform before fusion happens. This is a real information loss, and it's the reason some production pipelines don't use RRF alone: they use RRF as a first, robust fusion step to get a reasonable candidate ordering with no calibration burden, and then either apply a lightweight score-calibration technique (e.g., normalizing each retriever's scores to a common scale via min-max or z-score normalization over that query's result set before a weighted combination) or, more commonly in modern RAG stacks, simply feed the RRF-fused shortlist into a cross-encoder reranker (Section 2), which resolves the magnitude question anyway by directly re-scoring each candidate with a much more accurate joint model rather than trying to reconcile disparate first-stage scores at all.

```python
"""
reciprocal_rank_fusion.py

Fuse three ranked lists (BM25, dense retrieval, and a reranker's own
output) into a single consensus ranking using Reciprocal Rank Fusion.

No external dependencies beyond the standard library.
"""

from collections import defaultdict
from typing import Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[tuple]:
    """
    ranked_lists: a list of ranked lists, each a list of document IDs
        ordered best-to-worst (index 0 = rank 1). Lists may have different
        lengths and need not contain the same set of documents.
    k: RRF damping constant; 60 is the conventional default.

    Returns a list of (doc_id, fused_score) tuples sorted by fused_score
    descending.
    """
    fused_scores: Dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for position, doc_id in enumerate(ranked_list):
            rank = position + 1  # ranks are 1-indexed, not 0-indexed
            fused_scores[doc_id] += 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    # Three independent ranked lists over the same candidate pool, from
    # Chapter 4's hybrid retrieval plus this chapter's reranker.
    bm25_ranking = ["docA", "docC", "docB", "docE"]
    dense_ranking = ["docB", "docA", "docD", "docC"]
    reranker_ranking = ["docA", "docB", "docD", "docC", "docE"]

    fused = reciprocal_rank_fusion(
        [bm25_ranking, dense_ranking, reranker_ranking],
        k=60,
    )

    for doc_id, score in fused:
        print(f"{doc_id}: {score:.5f}")

    # docA ranks in the top 2 of every list and wins the fused ranking
    # despite never being the literal #1 result in two of the three lists --
    # exactly the "consistently good across multiple opinions beats a single
    # list's top pick" behavior RRF is designed to produce.
```

Running this, `docA` (ranked 1st, 2nd, and 1st across the three lists) comes out on top of the fusion even though it isn't the single highest-ranked document in every individual list, while `docE` (present in only two lists, and near the bottom of both) ends up at the bottom of the fused ranking. That consensus-seeking behavior — rewarding documents that multiple independent retrieval signals agree are good, rather than blindly trusting whichever single list ranked something first — is the entire value proposition of RRF in one worked example.

## 6. Where reranking sits in the latency and cost budget

Reranking occupies a specific, narrow slot in the request path: after first-stage retrieval has already returned a candidate list, and before that list (or a further-diversified version of it, per Section 4) is assembled into a prompt and sent to the LLM for generation. Concretely, a cross-encoder pass over roughly 50 candidates on a GPU typically costs somewhere in the tens to low hundreds of milliseconds, depending on model size, sequence length, and batching — small enough to comfortably fit inside an interactive request budget alongside first-stage retrieval and generation. A hosted rerank API call covers the same scoring work but adds a network round trip on top of the provider's own inference time, which is usually still acceptable for most interactive applications but is a real, measurable addition to the latency budget that self-hosting avoids.

The one parameter every team has to decide explicitly is how many candidates to feed into the reranking stage — the width of the neck of the funnel right before it narrows to a final few. This is a genuine trade-off in both directions, not a "bigger is always safer" knob. Feeding the reranker too few candidates, say only the first-stage top 10, risks a specific and easy-to-miss failure: if the true best passage happened to be ranked 14th by the bi-encoder or BM25 — entirely possible, since first-stage retrieval is approximate by design — it never makes it into the reranker's input at all, and no amount of reranking accuracy downstream can recover a document the reranker was never shown. Feeding it too many candidates, say the first-stage top 200, adds real, avoidable latency and cost (every extra candidate is another forward pass through the cross-encoder, or another line item in a hosted API's per-document pricing) for a shrinking quality return, because first-stage recall at very deep cutoffs is usually already high enough that additional candidates beyond some point are overwhelmingly irrelevant noise rather than hidden gems. In practice, teams commonly settle on feeding somewhere in the range of the first-stage top 20 to top 100 candidates into reranking, then keeping only the top 3-5 of the reranked output for the prompt — wide enough to make it very unlikely the best passage was excluded before reranking got a chance to see it, narrow enough that the expensive stage stays cheap and fast. The right number for a given deployment is something to actually measure against a retrieval-quality eval (Chapter 8) rather than assume, since it depends on how good first-stage recall already is at different depths for that specific corpus and query distribution.

It's worth closing on why reranking, specifically, tends to be the highest-leverage single change available to a team debugging an underperforming RAG pipeline. Almost every other retrieval improvement — a better embedding model, a different chunking strategy, a rebuilt index — requires re-embedding and re-indexing the entire corpus, which is expensive, slow, and risky to roll out. Adding a reranking stage requires none of that: the corpus, the index, and first-stage retrieval stay exactly as they are, and the only change is inserting one additional scoring step into the query-time serving path between "retrieve candidates" and "build the prompt." Because it operates purely at query time on whatever candidates first-stage retrieval already produced, it can be added, evaluated, tuned, or even A/B tested and rolled back, without touching a single document in the underlying store — which makes it one of the few retrieval-quality levers a team can pull with a same-day change rather than a multi-day reindexing project.

Whether a given reranking configuration is actually worth its added latency is an empirical question, not something to settle by intuition alone. The standard way to check is to run the same evaluation set used to judge first-stage retrieval quality (Chapter 8 covers this machinery in depth — metrics like NDCG@k, MRR, and recall@k against a labeled or LLM-judged relevance set) once with first-stage ordering alone and once with the reranked ordering, and compare. A useful diagnostic beyond the aggregate metric is to look specifically at queries where the reranker changed the top result relative to first-stage retrieval: if those swaps consistently move a more relevant document into the top slot, the reranking stage is earning its latency cost; if the swaps are mostly noise among near-equally-relevant candidates, a cheaper or smaller reranker (or a larger first-stage top-k with no reranking at all) may be just as good for a fraction of the cost. This is also the right lens for deciding candidate-pool width empirically rather than by rule of thumb — sweep the first-stage top-k fed into the reranker (say 10, 20, 50, 100) against the same eval set and look for the point where the quality curve flattens, since that point is exactly where additional candidates stop paying for their added latency.
