# Approximate Nearest Neighbor Algorithms

## Why Exact Search Doesn't Scale

The naive way to find the nearest vectors to a query is brute force: compute the distance from the query to every single vector in the collection, then sort. This is called a flat or exhaustive scan, and it is exact — it always returns the true top-k nearest neighbors. It is also `O(N*d)` per query, where `N` is the number of stored vectors and `d` is the dimensionality. For a small collection (tens of thousands of vectors), this is genuinely fine, and many teams over-engineer their first vector search deployment by reaching for a complex ANN index when a flat scan would have met their latency budget with a fraction of the operational complexity. FAISS's `IndexFlatL2` and pgvector's exact scan mode exist precisely for this regime.

The problem appears once `N` grows into the millions or the query rate grows past a handful of queries per second. At 50 million 768-dimensional vectors, one brute-force query means roughly 50 million times 768 floating point multiply-adds, per query, single query latency on the order of hundreds of milliseconds to seconds depending on hardware — and that's before you've served a second concurrent query. Real production systems need sub-100ms p99 latency at hundreds or thousands of queries per second, against corpora that keep growing. Brute force simply cannot deliver that combination, and no amount of hardware scaling changes the fundamental `O(N)` per-query cost — you need an algorithmic change, not just more machines.

This is where approximate nearest neighbor (ANN) algorithms come in. They accept a small, tunable probability of missing the true nearest neighbor (or returning it slightly out of rank order) in exchange for query times that are sublinear in `N` — often `O(log N)` or better. The entire vector database industry is built on a small number of these algorithms, and understanding how they work — not just that they exist — is what separates someone who has used Pinecone from someone who can reason about why a specific index is slow or how to tune it.

## The Recall/Latency/Memory Triangle

Before getting into specific algorithms, it's worth naming the three-way trade-off that every ANN algorithm and every tuning knob within it is ultimately trading between: **recall** (how often the approximate result matches what exact search would have returned), **latency** (query time), and **memory/index size**. You cannot independently maximize all three — every architectural decision and every hyperparameter in HNSW, IVF, and product quantization is a dial that moves you along this trade-off surface. When an interviewer asks "how would you make this index faster," the expected reasoning pattern is "here's the knob I'd turn, and here's what I'd expect to lose in exchange" — not just naming a knob in isolation.

## HNSW: Hierarchical Navigable Small World Graphs

HNSW is the dominant ANN algorithm in modern vector databases — it backs the default index in Pinecone, Weaviate, Qdrant, Milvus, Chroma, and pgvector's `hnsw` index type. Understanding it well is probably the single highest-leverage piece of ANN knowledge for an interview.

### The Intuition

Imagine you're trying to find a specific person in a large social network, and the only tool you have is "ask someone if they know anyone closer to the target than they are, and hop to that person." If everyone only knew their immediate neighbors, you'd need many hops to cross the network — that's a plain nearest-neighbor graph with only local edges, and search on it is slow because each hop makes only small progress. But if a few people in the network happen to have long-range friendships (a small-world property, the same phenomenon behind "six degrees of separation"), you can use those long-range connections to jump across large distances quickly, then switch to local hops once you're in the right neighborhood to refine your answer. HNSW builds exactly this kind of graph on purpose, and it does so at multiple "zoom levels" simultaneously, which is the "hierarchical" part of the name.

### Construction

HNSW builds a multi-layer graph where each layer is a proximity graph (nodes connected to their approximate nearest neighbors) and higher layers contain exponentially fewer nodes, each layer being a random sparse subset of the layer below. Layer 0 (the bottom) contains every vector in the index. Layer 1 contains a random subset, typically around 1/e of layer 0's nodes based on the standard construction's exponentially decaying level-assignment probability. Layer 2 contains a subset of layer 1, and so on, up to some small number of layers at the top — often just a handful of nodes in the topmost layer for a large index.

When a new vector is inserted, the algorithm randomly assigns it a "top layer" it will appear in (higher layers become exponentially less likely, giving the structure its logarithmic search behavior), then greedily searches from the current entry point downward through the layers, at each layer finding the closest existing nodes to connect the new node to (this connection count is the tunable parameter `M`, typically 16-64 — higher `M` means denser graphs, better recall, more memory, and slower construction). This is why HNSW is described as an incremental, graph-building algorithm: unlike some other index types, you don't need the full dataset up front — vectors can be added one at a time, which is why HNSW supports upserts far more gracefully than cluster-based approaches like IVF.

```python
import random
import numpy as np

class ToyHNSWNode:
    def __init__(self, vec_id, vector, level):
        self.id = vec_id
        self.vector = vector
        self.level = level
        self.neighbors = {l: [] for l in range(level + 1)}  # per-layer adjacency

def assign_level(m_L=1.0):
    """Exponentially decaying layer assignment -- the source of the hierarchy.
    Most nodes get level 0; very few reach high levels."""
    return int(-np.log(random.random()) * m_L)

# Simulate: most inserts land at level 0, a handful climb higher
levels = [assign_level(m_L=1.0) for _ in range(10000)]
from collections import Counter
print(Counter(levels))
# Something like {0: 6300, 1: 2300, 2: 850, 3: 320, 4: 130, 5: 60, ...}
```

### Search

Search starts at a fixed entry point in the topmost, sparsest layer. At each layer, it performs a greedy walk: look at the current node's neighbors in this layer, move to whichever neighbor is closest to the query, and repeat until no neighbor improves on the current position (a local minimum at this layer). Then it drops down one layer, using the current position as the new starting point, and repeats the greedy walk with the (now denser) neighbor set at that layer. This continues until it reaches layer 0, where a final, more thorough greedy search (expanded by the `ef_search` parameter, discussed below) produces the final candidate list.

```python
def greedy_search_layer(query_vec, entry_node, layer, get_vector_fn, max_steps=100):
    """Sketch of HNSW's core greedy routine at a single layer."""
    current = entry_node
    current_dist = np.linalg.norm(query_vec - get_vector_fn(current))

    for _ in range(max_steps):
        improved = False
        for neighbor in current.neighbors[layer]:
            d = np.linalg.norm(query_vec - get_vector_fn(neighbor))
            if d < current_dist:
                current, current_dist = neighbor, d
                improved = True
        if not improved:
            break  # local minimum at this layer -- drop down one level
    return current

def hnsw_search(query_vec, entry_point, top_layer, get_vector_fn):
    current = entry_point
    for layer in range(top_layer, 0, -1):
        current = greedy_search_layer(query_vec, current, layer, get_vector_fn)
    # At layer 0, do a wider beam search (ef_search candidates) for final results
    return greedy_search_layer(query_vec, current, 0, get_vector_fn, max_steps=200)
```

The reason this is fast is that the top layers let the search take huge geometric "strides" across the space with very few hops (because those layers are sparse, each hop covers a lot of ground), and only once the search has narrowed down to roughly the right neighborhood does it drop into the dense bottom layer to refine. This is directly analogous to a skip list in classical data structures, or to how you'd navigate a country by first picking the right state, then the right city, then the right street — you don't do a linear walk down every street in the country.

### Tuning Knobs

HNSW exposes three parameters that map directly onto the recall/latency/memory triangle. `M` (typically 16-64) controls how many edges each node keeps per layer; higher `M` improves recall and search speed slightly but increases memory usage (more edges to store) and construction time significantly, since each insert does more work maintaining connectivity. `ef_construction` controls how thorough the search is *during index building* when looking for each new node's neighbors — higher values produce a better-quality graph (more accurate neighbor selection during construction) at the cost of slower ingestion, but this only affects build time, not query time. `ef_search` controls how many candidates the greedy search keeps in its priority queue at query time (the search "beam width") — this is the knob you tune live, per query, without touching the index. Lower `ef_search` gives faster, lower-recall queries; higher `ef_search` gives slower, higher-recall queries, and this is usually exposed directly in the client API (e.g., Qdrant and Weaviate let you pass `ef` per search request), making it the primary lever for a service to trade off latency against quality dynamically — for instance, dropping `ef_search` under load to protect p99 latency at the cost of some recall.

The main downside of HNSW is memory: the full graph (all layers' adjacency lists) plus the raw vectors typically need to live in RAM for good performance, since graph traversal involves essentially random access patterns that are brutal on cold storage or disk-based lookups. This is why HNSW-heavy vector databases are often memory-bound in their cost structure — you're paying for RAM to hold billions of edges, not just the raw vector bytes.

## IVF: Inverted File Index

IVF takes a fundamentally different, partition-based approach rather than a graph. The idea: instead of searching every vector or navigating a graph, first partition the entire vector space into `nlist` clusters (via k-means run once over a representative sample of the data), giving each cluster a centroid. Every vector in the dataset gets assigned to its nearest centroid's "bucket" or "inverted list," analogous to how an inverted index in text search maps a term to the documents containing it — here, a centroid ID maps to the vectors assigned to it.

At query time, rather than scanning every vector, you first find the `nprobe` closest centroids to the query (a cheap operation since there are far fewer centroids than vectors), then only brute-force search within those `nprobe` clusters' vectors. This shrinks the effective search space from all `N` vectors down to roughly `N * (nprobe / nlist)` vectors, a large speedup when `nlist` is much bigger than `nprobe`.

```python
def build_ivf_index(vectors, n_clusters):
    """Simplified IVF construction using k-means for cluster assignment."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=3).fit(vectors)
    centroids = kmeans.cluster_centers_
    inverted_lists = {i: [] for i in range(n_clusters)}
    for idx, cluster_id in enumerate(kmeans.labels_):
        inverted_lists[cluster_id].append(idx)
    return centroids, inverted_lists

def ivf_search(query_vec, centroids, inverted_lists, vectors, nprobe=8, top_k=10):
    """Search only the nprobe closest clusters instead of the whole dataset."""
    centroid_dists = np.linalg.norm(centroids - query_vec, axis=1)
    closest_clusters = np.argsort(centroid_dists)[:nprobe]

    candidates = []
    for cluster_id in closest_clusters:
        for idx in inverted_lists[cluster_id]:
            d = np.linalg.norm(vectors[idx] - query_vec)
            candidates.append((d, idx))

    candidates.sort(key=lambda x: x[0])
    return candidates[:top_k]
```

`nlist` and `nprobe` are IVF's tuning knobs, mirroring HNSW's `M`/`ef_search` role: more clusters (`nlist`) means finer partitioning and, generally, faster queries at fixed `nprobe`, but coarser partitioning of the space is more prone to the classic "edge of cluster" problem — a query vector sitting near a cluster boundary might have its true nearest neighbor sitting just across the border in a cluster that wasn't among the `nprobe` selected, hurting recall. Increasing `nprobe` searches more clusters, recovering recall at the cost of latency (in the limit, `nprobe = nlist` degenerates into a brute-force scan, guaranteeing exact-ish recall at full cost). Choosing `nlist` is itself a rule-of-thumb exercise — a common heuristic is `nlist ≈ sqrt(N)`, balancing cluster count against per-cluster size.

IVF's big practical advantage over HNSW is memory efficiency and, notably, that it composes cleanly with quantization (below) since each cluster's vectors can be compressed independently and the centroids act as a small, cheap-to-scan routing layer. Its disadvantage is that it needs a representative training sample up front to build good centroids, doesn't adapt as gracefully to a highly dynamic, rapidly growing dataset (new data can be assigned to existing clusters, but if the data distribution drifts significantly, the original centroids stop being a good partitioning and the index needs to be rebuilt), and typically has lower recall than a well-tuned HNSW graph at comparable latency for pure in-memory workloads. In practice IVF and its quantized variants shine most when the priority is fitting a huge collection into a memory or cost budget that a full HNSW graph couldn't reach.

## Product Quantization (PQ) and IVF-PQ

Product quantization is a compression technique, not a search algorithm on its own — it's almost always paired with IVF (as "IVF-PQ") or, less commonly, layered onto graph indexes, specifically to shrink the memory footprint of the stored vectors themselves, as opposed to the routing structure around them.

The idea: split each `d`-dimensional vector into `m` sub-vectors (say, a 768-dim vector split into 8 sub-vectors of 96 dimensions each). For each of the `m` sub-vector positions, run k-means independently across the whole dataset's sub-vectors at that position, producing a small codebook (typically 256 centroids, so each centroid ID fits in a single byte). Every original vector is then re-encoded, position by position, as the ID of its nearest codebook centroid at each sub-vector slot. A 768-dimensional float32 vector (3072 bytes) becomes 8 single-byte codes (8 bytes) — a compression ratio in the hundreds.

```python
def train_pq_codebooks(vectors, n_subvectors, n_centroids=256):
    """Train one small k-means codebook per sub-vector slot."""
    from sklearn.cluster import KMeans
    d = vectors.shape[1]
    sub_dim = d // n_subvectors
    codebooks = []
    for i in range(n_subvectors):
        sub_vecs = vectors[:, i * sub_dim:(i + 1) * sub_dim]
        km = KMeans(n_clusters=n_centroids, n_init=3).fit(sub_vecs)
        codebooks.append(km)
    return codebooks

def pq_encode(vector, codebooks, n_subvectors):
    """Encode one vector into m single-byte codes."""
    d = len(vector)
    sub_dim = d // n_subvectors
    codes = []
    for i, km in enumerate(codebooks):
        sub_vec = vector[i * sub_dim:(i + 1) * sub_dim].reshape(1, -1)
        code = km.predict(sub_vec)[0]
        codes.append(code)
    return codes  # e.g. [17, 203, 4, 91, 128, 6, 250, 33] -- 8 bytes total

def pq_approximate_distance(query_vec, codes, codebooks, n_subvectors):
    """Distance is computed against reconstructed (quantized) centroids,
    not the original vector -- this is the source of PQ's approximation error."""
    d = len(query_vec)
    sub_dim = d // n_subvectors
    total = 0.0
    for i, code in enumerate(codes):
        centroid = codebooks[i].cluster_centers_[code]
        q_sub = query_vec[i * sub_dim:(i + 1) * sub_dim]
        total += np.sum((q_sub - centroid) ** 2)
    return np.sqrt(total)
```

The magic that makes PQ fast rather than just small is that distance computations against quantized codes can be precomputed and turned into table lookups: for a given query, you precompute the distance from each of its own sub-vectors to every centroid in the corresponding codebook (a small `m * 256` table), and then computing the approximate distance to any stored vector becomes `m` table lookups and additions rather than `d` multiplications — this is often called "asymmetric distance computation" (ADC) since the query stays full-precision while the stored vectors are quantized. Combined with IVF's cluster routing (searching only `nprobe` clusters) and PQ's compressed, fast-to-scan-in-bulk representation within each cluster, IVF-PQ can index billions of vectors in a fraction of the RAM that flat storage or HNSW would need — this combination is exactly what FAISS's `IVFxxx,PQyy` index factory strings build, and what powers extreme-scale deployments where raw vector storage would otherwise be cost-prohibitive.

The cost of this compression is approximation error at the vector level itself, layered on top of the approximation error IVF already introduces from cluster routing — PQ-compressed distances are reconstructions from quantized centroids, not the true distances, so recall drops further compared to uncompressed IVF or HNSW at the same `nprobe`/`ef_search` settings. Production systems compensate with a **rerank step**: retrieve a larger candidate set using cheap, approximate PQ-based scoring, then recompute exact (or higher-fidelity) distances against the original, uncompressed vectors for just that smaller candidate set — getting most of PQ's memory savings while recovering most of the recall loss, since exact scoring is only paid for a small final candidate list rather than the whole corpus.

## Measuring Recall in Practice

All the tuning knobs discussed above are meaningless without a way to actually measure recall, and it's worth being concrete about how that measurement is done, since "recall" is thrown around casually but has a precise operational definition here: **recall@k is the fraction of the true top-k nearest neighbors (as found by exhaustive brute-force search) that an approximate index actually returns in its own top-k results.** Computing it requires maintaining a small "ground truth" set — brute-force nearest neighbors computed once for a representative sample of queries — against which any approximate index's output can be checked.

```python
def compute_recall_at_k(approx_results: list, exact_results: list, k: int) -> float:
    """approx_results, exact_results: lists of result-id-lists, one per query,
    both already truncated/sorted to their own top-k."""
    total_recall = 0.0
    for approx_ids, exact_ids in zip(approx_results, exact_results):
        exact_set = set(exact_ids[:k])
        approx_set = set(approx_ids[:k])
        total_recall += len(exact_set & approx_set) / len(exact_set)
    return total_recall / len(exact_results)

def brute_force_top_k(query_vec, vectors, k=10):
    dists = np.linalg.norm(vectors - query_vec, axis=1)
    return list(np.argsort(dists)[:k])

# A minimal tuning loop: sweep ef_search (or nprobe) and record the recall/latency curve
def sweep_recall_vs_latency(index, queries, ground_truth, param_values, param_name, k=10):
    import time
    results = []
    for value in param_values:
        setattr(index, param_name, value)
        start = time.perf_counter()
        approx = [index.search(q, top_k=k) for q in queries]
        elapsed = (time.perf_counter() - start) / len(queries)
        recall = compute_recall_at_k(approx, ground_truth, k)
        results.append({"param": value, "recall": recall, "avg_latency_s": elapsed})
    return results
```

Running this sweep and plotting recall against latency is the standard, honest way to choose `ef_search` or `nprobe` for a production deployment — rather than picking a default value from documentation and hoping it fits your data distribution, since the right operating point is highly dependent on your specific corpus's geometry (how tightly clustered or spread out your embeddings actually are) and cannot be reliably predicted from a generic rule of thumb alone. It's worth re-running this sweep periodically as a corpus grows, since a parameter setting that hit 98% recall at 1 million vectors is not guaranteed to hit the same recall at 50 million, particularly for IVF-family indexes where cluster quality degrades if `nlist` wasn't scaled up alongside the data.

## Comparing the Trade-offs

| Algorithm | Query speed | Memory | Build cost | Update-friendliness | Best fit |
|---|---|---|---|---|---|
| Flat / brute-force | Slow (`O(N)`) | Low (raw vectors only) | None | Trivial | Small corpora (<~100K), or ground-truth eval |
| HNSW | Very fast (`O(log N)`-ish) | High (graph + vectors, mostly in RAM) | Moderate-high | Good (incremental insert) | Latency-critical, RAM-available, moderate scale |
| IVF | Fast, tunable via `nprobe` | Moderate | Needs training pass | Fair (needs periodic re-clustering) | Large scale with a cost/memory ceiling |
| IVF-PQ | Fast, tunable | Very low (compressed codes) | Needs training pass | Fair, same caveats as IVF | Billion-scale, cost-constrained |

The honest interview answer to "which one should I use" is that it depends on where your bottleneck actually is. If you have RAM to spare and want the best recall/latency combination at moderate scale (up to tens or low hundreds of millions of vectors), HNSW is the default choice and is what most managed vector databases use under the hood without asking you to choose. If your corpus is large enough that even compressed graph memory becomes the dominant cost, or you're running on cheaper, disk-heavy infrastructure, IVF-PQ trades some recall for an order-of-magnitude reduction in memory footprint. Many production systems, including FAISS-based ones, don't pick purely one or the other — they layer PQ-style compression underneath an IVF-style coarse routing structure and add an exact rerank pass on the shortlist, capturing most of the speed and memory benefits of both while keeping end-to-end recall acceptable.
