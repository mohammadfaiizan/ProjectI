# Hybrid Search and Scaling

## Why Pure Vector Search Falls Short

Dense vector search is excellent at capturing semantic similarity — it finds documents that mean the same thing even when they don't share vocabulary. But it has a well-known, systematic weakness: it's often mediocre at exact lexical matching, precisely the cases where a keyword search engine excels. If a user searches for a specific error code (`ORA-00942`), a product SKU, a person's name, or an acronym that rarely appears in the embedding model's training distribution, a dense embedding may not place that query especially close to the one document that contains the literal exact string, because the embedding model is optimized for paraphrase-level semantic similarity, not for surfacing rare or out-of-distribution tokens. Keyword search algorithms like BM25, by contrast, are extremely good at exactly this — they explicitly reward exact term overlap, weighted by how rare and how discriminative a term is across the corpus (its inverse document frequency), which is precisely the signal dense embeddings tend to blur.

The reverse failure mode is just as real: keyword search alone misses genuine paraphrases and synonymy entirely — a query for "how do I cancel my membership" won't match a document that only says "terminating your subscription" if there's no meaningful token overlap, even though the two are semantically identical. This complementary weakness pattern — dense search misses exact/rare terms, sparse search misses paraphrase/semantic matches — is exactly why hybrid search, combining both signals, has become close to a default expectation in serious production retrieval systems rather than a nice-to-have.

## BM25: The Keyword Half

BM25 (Best Matching 25) is a scoring function that ranks documents against a query based on term frequency, inverse document frequency, and document length normalization. Understanding its shape matters because it explains what dense embeddings are compensating for. For a query term `t` and document `d`:

```
BM25(d, q) = sum over t in q of:
    IDF(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))
```

Here `f(t, d)` is how many times term `t` appears in document `d`, `IDF(t)` rewards terms that are rare across the whole corpus (a term appearing in every document contributes almost nothing to the score, since it doesn't discriminate between documents), `|d| / avgdl` normalizes for document length so that long documents don't win purely by containing more words, and `k1`/`b` are tunable saturation and length-normalization parameters. The key intuitive property is **term saturation**: the first occurrence of a matching term boosts the score a lot, but the fifth occurrence of the same term barely adds anything further — this prevents keyword stuffing from dominating relevance, and it's a deliberately different shape from a linear term-count score.

```python
import math
from collections import Counter

def bm25_score(query_terms, doc_terms, corpus_doc_freqs, n_docs, avgdl, k1=1.5, b=0.75):
    doc_len = len(doc_terms)
    term_counts = Counter(doc_terms)
    score = 0.0
    for term in query_terms:
        f = term_counts.get(term, 0)
        if f == 0:
            continue
        df = corpus_doc_freqs.get(term, 0)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * (numerator / denominator)
    return score
```

Most modern vector databases with hybrid support (Weaviate, Qdrant, Elasticsearch/OpenSearch with vector plugins, Vespa) implement BM25 or a close variant as the sparse half of hybrid search, running it via an inverted index — the same fundamental data structure traditional search engines have used for decades, now living alongside a dense ANN index in the same system.

## Fusing Dense and Sparse Scores

The hard part of hybrid search isn't computing two separate rankings — it's combining them into one, given that BM25 scores and cosine similarity scores live on completely different, incomparable scales (BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded in a predictable range). Naively adding raw scores together is close to meaningless, since whichever score happens to have larger typical magnitude will dominate regardless of which one is actually more relevant for a given query.

**Reciprocal Rank Fusion (RRF)** sidesteps the scale-incompatibility problem entirely by ignoring raw scores and working with rank positions instead. For each result, RRF computes `1 / (k + rank)` in each ranked list it appears in (where `rank` is its position, 1-indexed, and `k` is a small constant, commonly 60, that dampens the influence of very top-ranked results and prevents any single list from completely dominating), then sums those contributions across all the lists being fused. A document that ranks highly in *both* the dense and sparse lists gets a strong combined score; a document that ranks highly in only one list still gets a moderate boost, proportional to how high it ranked there.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """ranked_lists: list of ranked-id-lists, e.g. [dense_results_ids, bm25_results_ids]
    Returns a combined ranking with fused scores."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

dense_ranking = ["doc_7", "doc_3", "doc_9", "doc_1"]
bm25_ranking = ["doc_3", "doc_1", "doc_15", "doc_7"]
fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
# doc_3 and doc_7 both appear near the top of both lists -> highest fused scores
```

RRF's popularity comes from robustness — it needs no score calibration or normalization, no tuning of relative weight between dense and sparse (beyond the fairly insensitive `k` constant), and it degrades gracefully. The alternative is **weighted linear fusion**, `combined_score = alpha * normalize(dense_score) + (1 - alpha) * normalize(sparse_score)`, which requires normalizing both score distributions onto a comparable scale (min-max normalization within the candidate set is a common approach) and choosing `alpha` — a knob that can be tuned per use case or per query type if you have labeled relevance data to optimize against, but is genuinely fragile without it. Weaviate's native hybrid search API exposes exactly this `alpha` parameter, letting you lean the fusion toward pure vector (`alpha=1`) or pure keyword (`alpha=0`) per query.

```python
def weighted_fusion(dense_results, sparse_results, alpha=0.5):
    """dense_results, sparse_results: dict of doc_id -> raw_score"""
    def minmax_normalize(d):
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    dense_norm = minmax_normalize(dense_results)
    sparse_norm = minmax_normalize(sparse_results)
    all_ids = set(dense_norm) | set(sparse_norm)
    combined = {
        doc_id: alpha * dense_norm.get(doc_id, 0) + (1 - alpha) * sparse_norm.get(doc_id, 0)
        for doc_id in all_ids
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

In practice, teams that care most about squeezing out the last bit of quality run hybrid retrieval to produce a generous candidate set (say, top 50-100 via RRF or weighted fusion), then apply a cross-encoder reranker on that shortlist as a final precision pass — the reranker sees the actual query and document text jointly rather than relying on any embedding geometry or lexical overlap heuristic, and it consistently outperforms both dense and sparse signals alone on final ranking quality, at the cost of being too slow to run against the full corpus, which is exactly why it's reserved for the shortlist stage.

## Sharding Strategies

Once a vector collection outgrows what a single machine can hold in memory or serve at acceptable query throughput, the collection needs to be sharded — split across multiple nodes, each holding a subset of the vectors and serving queries against its own slice. The dominant sharding strategy for vector databases is **random/hash-based sharding**: assign each vector to a shard via a hash of its ID (or round-robin), so vectors are distributed roughly evenly across shards without regard to their content or embedding position. This is the default in Milvus, Qdrant's distributed mode, and most others, precisely because vector similarity doesn't have an obvious, cheap-to-compute "range" the way, say, a timestamp or a numeric ID does — you can't shard "by embedding value" the way you'd shard a relational table by a sortable key, since there's no natural ordering over a high-dimensional space that preserves locality cheaply at write time.

A query against a hash-sharded collection has to be **scattered to every shard** (a query "fans out" to all nodes), each shard runs its own local ANN search and returns its own local top-k, and a coordinator node merges the per-shard results into a single global top-k by re-sorting the union. This scatter-gather pattern is simple and correct, but it means query cost scales with the number of shards you fan out to, and tail latency is governed by the slowest responding shard (the classic "tail latency amplification" problem common to any scatter-gather architecture) — a single slow or overloaded shard drags down every query that touches it, even if the other shards responded instantly.

```python
def scatter_gather_search(query_vec, shards, top_k=10):
    """Fan out to all shards, then merge. Latency is bounded by the slowest shard."""
    import concurrent.futures

    def query_shard(shard):
        return shard.search(query_vec, top_k=top_k)  # local top-k per shard

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [executor.submit(query_shard, s) for s in shards]
        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())

    all_results.sort(key=lambda r: r.distance)
    return all_results[:top_k]
```

Some systems support **metadata-aware sharding** as an alternative or complement — routing vectors to specific shards based on a known partition key (most commonly tenant ID), so that a query scoped to one tenant only needs to hit the shard(s) holding that tenant's data rather than scattering everywhere. This is the same idea as the tenant-based physical partitioning discussed in the indexing chapter, applied at the sharding layer: when your access pattern is predictably partitioned by some field, routing on that field avoids the fan-out cost entirely for the common case, at the cost of potential load imbalance if some partitions (tenants) are much larger or hotter than others, which then needs its own mitigation (further splitting an oversized tenant's shard, for instance).

## Replication and Consistency

Replication in vector databases serves the same two purposes it serves everywhere else — availability (surviving node failure) and read throughput scaling (spreading query load across replicas of the same shard) — but has a distinctive wrinkle because of how ANN indexes are built. Unlike a relational database where a replica can apply a stream of row-level writes and stay byte-for-byte consistent, replicating an HNSW graph or IVF structure exactly, in real time, across nodes is expensive, because graph construction involves randomized choices (which layer a node lands on, the exact neighbor set chosen during greedy construction) that make two independently-built indexes over the same data non-identical, even if functionally equivalent in search quality. Most systems handle this by replicating at the level of the underlying data plus a synchronized index-build process (or by shipping index segments directly, similar to how Lucene-based search engines replicate index segments) rather than trying to replicate individual graph mutations operation-by-operation.

This has direct consistency implications: most vector databases default to **eventual consistency** for newly written or updated vectors — a write acknowledged by the primary may take some (usually short, but non-zero) time to become visible on replicas or to be reflected in the ANN index at all, since index insertion itself can lag behind raw data ingestion, especially under IVF-style batch-oriented indexing. For most retrieval and RAG use cases this is an acceptable trade-off — a document being searchable a few hundred milliseconds to a few seconds after being written rarely matters — but it's a real, sometimes-surprising design constraint for any use case that expects "write then immediately read your own write" semantics, and it's worth confirming explicitly with a given vector database's documentation and, ideally, with your own testing under load, rather than assuming synchronous consistency by default.

## Geo-Distribution and Cross-Region Latency

A distinct scaling axis from raw vector count is geographic distribution: serving a global user base with acceptable latency from every region, while keeping a single logically consistent (or acceptably-eventually-consistent) index. The naive approach — one central deployment, all queries routed to it — is simple but imposes a network round-trip tax on every user far from that region; a user in Singapore querying a vector database hosted only in `us-east-1` pays 150-200ms of pure network latency before the ANN search itself even begins, which can dwarf the actual search time for a well-tuned HNSW index.

The two realistic mitigations mirror strategies from general distributed systems. **Full regional replication** — a complete, independently-servable copy of the index in each major region, kept in sync via the same eventual-consistency replication mechanisms discussed above — gives every region local, low-latency read access, at the cost of multiplying storage cost by the number of regions and needing a defined strategy for write propagation delay (a document written in one region takes some time to appear in others' local copies, and that lag needs to be an accepted product behavior, not a surprise). **Read-through caching or regional query routing to the nearest replica with graceful staleness tolerance** is a lighter-weight middle ground used when full replication cost isn't justified — accept that some regions serve slightly stale or lower-priority traffic against a single primary, while your highest-traffic regions get full local replicas. Neither approach eliminates the CAP-theorem-style trade-off inherent to any geo-distributed system: at internet scale, you are always choosing some combination of consistency, availability, and latency, and vector databases are not exempt from that trade-off just because the workload is similarity search rather than transactional writes.

## Sizing and Cost Modeling

A useful habit before committing to a scaling architecture is doing rough back-of-envelope sizing, because the numbers involved are large enough that intuition alone is unreliable. A simple model: `total_memory ≈ N * d * bytes_per_dim * index_overhead_factor`, where `bytes_per_dim` is 4 for float32 or 1 for int8 scalar-quantized storage, and `index_overhead_factor` captures the graph or partition structure's added cost on top of raw vectors — commonly estimated at 1.3-1.6x for a well-configured HNSW graph (accounting for edge lists across all layers), close to 1.0x for IVF-PQ where the compressed codes themselves are the dominant cost.

```python
def estimate_index_memory_gb(n_vectors, dims, bytes_per_dim=4, overhead_factor=1.4):
    raw_bytes = n_vectors * dims * bytes_per_dim
    total_bytes = raw_bytes * overhead_factor
    return total_bytes / (1024 ** 3)

print(estimate_index_memory_gb(100_000_000, 768))                       # float32 HNSW
print(estimate_index_memory_gb(100_000_000, 768, bytes_per_dim=1))      # int8 scalar-quantized
```

Running this for 100 million 768-dimensional vectors gives roughly 400 GB for a float32 HNSW deployment versus roughly 100 GB for an int8 scalar-quantized equivalent — a difference that, translated into cloud memory-optimized instance pricing, is easily a difference of tens of thousands of dollars a month at this scale. This kind of estimate, done before architecture is locked in, is what actually drives the decision between "just use HNSW everywhere" and "invest in quantization and a compressed index" far more reliably than abstract advice about when compression "matters" — the answer is almost always "compute the number for your actual scale and let the cost delta decide."

## Scaling to Hundreds of Millions or Billions of Vectors

At this scale, several forces that were background considerations at smaller scale become primary architectural drivers. Memory cost dominates the conversation: a pure in-memory HNSW deployment over a billion 768-dimensional float32 vectors, before any graph overhead, is already on the order of 3 TB of raw vector data alone, and the graph structure typically adds another substantial fraction on top. This is exactly why the compression techniques from the ANN chapter — product quantization, scalar quantization, and IVF-PQ's partition-plus-compress approach — shift from "an interesting optimization" to "a load-bearing requirement" at billion-vector scale; very few organizations can or should pay for enough RAM to hold an uncompressed HNSW graph at that size, and hybrid designs (compressed vectors resident in RAM or fast SSD, exact vectors for reranking kept in slower, cheaper storage and fetched only for the small final candidate set) become the norm.

Disk-based and disk-augmented ANN indexes become relevant at this scale for exactly this reason. DiskANN-style approaches (used in Milvus, and available as a Microsoft research lineage adopted more broadly) are specifically designed so that the bulk of the index lives on fast NVMe SSD rather than requiring full RAM residency, using algorithmic techniques (careful graph layout to minimize random disk seeks, combined with in-memory compressed vectors for a fast first pass) to keep query latency reasonable despite most of the data living on comparatively slow storage. This trades some latency and some engineering complexity for a large reduction in the RAM budget needed to serve a billion-plus-vector index, which is often the difference between a workable and an unworkable cost structure at that scale.

Ingestion throughput also becomes a first-class concern rather than an afterthought: continuously embedding and indexing content at the rate a large, actively growing corpus demands (think a content platform ingesting millions of new items per day) requires the embedding computation itself (often the more expensive step, especially if using a hosted embedding API with rate limits) and the index insertion step to both be pipelined and horizontally scalable, and it requires deciding — explicitly, as an architectural choice, not by default — whether index updates are synchronous with ingestion (new content searchable near-immediately, at the cost of more complex, higher-overhead per-item indexing) or batched (periodic bulk index updates, simpler and more throughput-efficient, at the cost of a longer delay before new content becomes searchable).

Finally, operational practices that are optional at smaller scale become mandatory at billion-vector scale: careful capacity planning with headroom for tombstone accumulation and compaction (discussed in the previous chapter, and considerably more consequential when compaction of a billion-vector graph is itself a heavyweight, hours-long operation), monitoring recall on a continuously refreshed validation set rather than trusting that tuning parameters chosen at smaller scale still deliver the same recall/latency trade-off as the data distribution and volume grow, and a genuine multi-region or multi-availability-zone replication strategy if the service has global latency or regional-failure-tolerance requirements, since at this scale a single-region outage affecting your primary vector store is a business-critical incident rather than a minor blip.
