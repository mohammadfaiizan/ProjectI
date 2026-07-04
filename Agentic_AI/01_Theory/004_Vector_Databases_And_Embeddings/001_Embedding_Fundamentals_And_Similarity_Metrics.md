# Embedding Fundamentals and Similarity Metrics

## Why Embeddings Exist

Every modern retrieval, recommendation, and semantic search system is built on a single trick: turning meaning into geometry. Words, sentences, images, and even user behavior get mapped into a high-dimensional vector space such that "similar" things end up near each other and "dissimilar" things end up far apart. Once that mapping exists, the hard, fuzzy problem of "does this document answer this question" becomes the comparatively easy, well-studied problem of "which points in space are close to this other point." This is the entire reason vector databases exist as a category of infrastructure — they are specialized systems for storing points in high-dimensional space and answering nearest-neighbor queries against them efficiently.

It's worth being precise about what an embedding actually is, because interview-level understanding requires more than "it's a vector representation." An embedding is the output of a learned function `f(x) -> R^d` where `x` is some piece of content (text, image, audio) and `d` is a fixed dimensionality chosen by the model architecture (384, 768, 1536, 3072 are common values). The function `f` is trained so that the geometric relationships between output vectors reflect semantic relationships between inputs. Critically, no one hand-designs these axes — you cannot point to dimension 47 and say "that's the sentiment axis." The space is learned end-to-end, usually via a contrastive or masked-language-modeling objective, and the resulting geometry is a side effect of optimizing for a proxy task (predicting masked tokens, distinguishing positive pairs from negatives, etc.).

This matters practically because it means embeddings from different models are never comparable to each other. A vector produced by `text-embedding-3-small` and a vector produced by a Cohere or open-source BGE model live in unrelated coordinate systems, even if both happen to be 1536-dimensional. You cannot mix embeddings from different models in the same index, and re-embedding your entire corpus is mandatory whenever you switch embedding models. This is a common production trap: teams upgrade an embedding model for better quality and forget that every previously indexed vector is now geometrically meaningless relative to new query vectors.

## The Geometric Interpretation

Think of the embedding space as encoding meaning through direction and, sometimes, magnitude. Two well-trained sentence embeddings for "the cat sat on the mat" and "a feline rested on the rug" should point in nearly the same direction, because the training objective pushed semantically equivalent pairs together. Embeddings for "the cat sat on the mat" and "quarterly revenue increased by 12 percent" should point in very different directions, because nothing in training ever associated those concepts. The angle between vectors, not necessarily their raw coordinate values, is usually what carries the semantic signal — which is precisely why cosine similarity dominates this field rather than raw dot products or coordinate-wise comparisons.

A useful mental model is that the embedding space has learned local neighborhoods that correspond to topics, sometimes with some amount of interpretable substructure (the classic word2vec example: king - man + woman ≈ queen was an early and slightly overhyped demonstration that arithmetic on embeddings can track semantic relationships). Modern sentence and passage embeddings from transformer encoders are less cleanly "linear algebra friendly" than word2vec's simpler co-occurrence-based vectors, but the core intuition holds: distance and angle in the space are a (noisy) proxy for semantic distance in the real world.

## Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, ignoring their magnitudes entirely:

```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```

This yields a value in `[-1, 1]` for real-valued vectors (in practice, most sentence embedding models produce vectors that cluster in a narrower positive range, often `[0, 1]` in effective usage, because the training objective rarely produces strongly anti-correlated pairs). A value of 1 means the vectors point in exactly the same direction; 0 means orthogonal (unrelated); -1 means opposite.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity from first principles."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

a = np.array([0.2, 0.8, 0.1])
b = np.array([0.1, 0.4, 0.05])   # same direction, smaller magnitude
c = np.array([-0.2, -0.8, -0.1]) # opposite direction

print(cosine_similarity(a, b))  # ~1.0 -- direction matters, not length
print(cosine_similarity(a, c))  # -1.0
```

The reason cosine dominates text embedding use cases is that magnitude in these spaces often correlates with something incidental — like the length of the input text, or the number of tokens the encoder pooled over — rather than with semantic importance. Two sentences that mean the same thing but differ in length can produce vectors of different magnitude even though the direction is nearly identical. Cosine similarity is invariant to that noise; it isolates the part of the signal (direction) that the training objective actually optimized for.

## Dot Product

The raw dot product is:

```
dot_product(A, B) = sum(A_i * B_i for i in range(d))
```

Unlike cosine similarity, the dot product is sensitive to both angle and magnitude. If vector `A` has a large norm, its dot product with anything tends to be larger, regardless of how well-aligned the direction is. This sounds like a strict downside, but it is exactly the right metric in two important situations.

First, if your embeddings are already L2-normalized (every vector forced to unit length before storage), then dot product and cosine similarity become mathematically identical — dividing by `||A|| * ||B||` is a no-op when both norms are 1. Because of this equivalence, most production vector databases actually implement "cosine similarity" internally as a normalize-then-dot-product operation, since dot product is cheaper to compute (one multiply-accumulate pass, no square roots) and can exploit fast BLAS/SIMD kernels more directly.

Second, dot product is the correct metric — not just a cheap substitute — when magnitude is meaningful to the model you're serving. Some retrieval models (particularly those trained specifically with a dot-product training objective, like certain dense passage retrieval models, or matrix-factorization-based recommender embeddings) are trained so that a document's vector norm encodes something like "general popularity" or "quality prior," independent of query relevance. In those cases, forcing normalization would throw away signal the model was explicitly trained to produce. The rule of thumb: use whatever metric matches the loss function the embedding model was trained with. If you don't know, check the model card — OpenAI, Cohere, and most sentence-transformers models explicitly document cosine as the intended metric, and pgvector/Milvus/Pinecone let you configure the metric per index accordingly.

```python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

# Once vectors are normalized, dot product IS cosine similarity
a_n, b_n = normalize(a), normalize(b)
assert np.isclose(dot_product(a_n, b_n), cosine_similarity(a, b))
```

## Euclidean (L2) Distance

Euclidean distance measures straight-line distance in the vector space:

```
euclidean_distance(A, B) = sqrt(sum((A_i - B_i)^2 for i in range(d)))
```

Unlike the other two metrics, this is a *distance* (smaller is more similar) rather than a *similarity* (larger is more similar), so be careful when wiring it into ranking code — a naive `sort by score descending` will silently invert your results if the underlying index returns L2 distances.

L2 distance is sensitive to both direction and magnitude, similarly to dot product, but it penalizes magnitude differences additively rather than multiplicatively. There's a neat algebraic identity worth memorizing because it explains why L2 and cosine often produce identical *rankings* (though different absolute scores) on normalized vectors:

```
||A - B||^2 = ||A||^2 + ||B||^2 - 2*(A · B)
```

If `A` and `B` are both unit-normalized, `||A||^2 = ||B||^2 = 1`, so this simplifies to `2 - 2*cosine_similarity(A, B)`. That means on normalized vectors, minimizing Euclidean distance and maximizing cosine similarity produce exactly the same nearest-neighbor ordering — they're monotonic transforms of each other. This is why FAISS, for example, lets you pick `IndexFlatL2` or `IndexFlatIP` (inner product) somewhat interchangeably once you've committed to normalizing your vectors at ingestion time.

Where L2 genuinely differs from cosine is on *unnormalized* vectors, or in domains where embeddings aren't primarily text — image embeddings from CNNs, audio embeddings, and certain learned metric-embedding spaces (e.g., face recognition embeddings trained with a triplet loss directly on L2 distance) are often designed with Euclidean distance as the native metric because the training loss directly minimizes L2 distance between positive pairs. In those cases, forcing cosine similarity would be applying a metric the model was never optimized for.

```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

print(euclidean_distance(a, b))   # some positive distance
print(euclidean_distance(a_n, b_n))  # equals sqrt(2 - 2*cosine_similarity(a, b))
```

## Choosing the Right Metric in Practice

The decision tree senior engineers should internalize is short: check what the embedding model was trained with (almost always documented), and default to cosine similarity when in doubt because it is the most forgiving of magnitude artifacts introduced by tokenization length, pooling strategy, or truncation. If your vector database charges extra latency or storage for one metric versus another (some ANN index types are natively built around one metric and simulate the others), normalize your vectors once at write time and standardize on dot product / inner product internally — you get cosine's semantic correctness with dot product's computational cheapness. This "normalize once, use inner product everywhere" pattern is exactly what most production RAG pipelines do, and it's a detail worth mentioning explicitly in an interview because it demonstrates you understand the difference between the *conceptual* metric and the *implemented* metric.

One subtlety that trips people up: normalizing vectors at write time doesn't help if new query vectors aren't normalized the same way at read time. The normalization step has to be a deterministic, versioned part of both the ingestion and query pipeline; if you ever change how you compute or truncate embeddings before normalization, you need to reprocess the whole index, not just new writes.

## Normalization in Depth

L2 normalization rescales a vector to unit length: `v_normalized = v / ||v||_2`. Beyond making cosine and dot product equivalent, normalization has a second, less obvious benefit for ANN index quality: many approximate nearest neighbor algorithms (particularly graph-based ones like HNSW) build their navigation structure using distance comparisons, and those comparisons are numerically better behaved when all vectors live on the unit hypersphere rather than having wildly varying magnitudes. A handful of outlier vectors with unusually large norms can distort greedy graph traversal, causing search to get "pulled" toward those high-magnitude points even when they aren't the true semantic nearest neighbors. Normalizing removes this failure mode entirely.

There is also a subtler statistical phenomenon worth knowing: embedding vectors from transformer models often exhibit "anisotropy" — instead of pointing in all directions roughly uniformly, they tend to cluster within a narrow cone of the full space, and the average cosine similarity between two *random*, *unrelated* sentences is not 0 as you might naively expect but often a surprisingly high value like 0.3–0.6, depending on the model. This means raw cosine similarity scores are not directly interpretable as calibrated probabilities across all models; a score of 0.75 might mean "extremely relevant" for one embedding model and "barely more related than noise" for another. This is why production relevance thresholds should always be calibrated empirically per model (e.g., by looking at the score distribution on a labeled validation set) rather than copied from documentation or another project's config, and it's also part of the motivation for reranking with a cross-encoder after initial vector retrieval — cross-encoders don't suffer the same anisotropy problem since they score query-document pairs jointly rather than via cosine geometry.

```python
def calibrate_threshold(labeled_pairs, embed_fn, target_precision=0.9):
    """Empirically find a similarity cutoff rather than guessing 0.7/0.8."""
    scores_and_labels = []
    for query, doc, is_relevant in labeled_pairs:
        q_vec, d_vec = normalize(embed_fn(query)), normalize(embed_fn(doc))
        score = dot_product(q_vec, d_vec)
        scores_and_labels.append((score, is_relevant))

    scores_and_labels.sort(key=lambda x: x[0], reverse=True)
    best_threshold, best_precision = 0.0, 0.0
    for i in range(len(scores_and_labels)):
        threshold = scores_and_labels[i][0]
        selected = [lbl for s, lbl in scores_and_labels if s >= threshold]
        precision = sum(selected) / len(selected) if selected else 0
        if precision >= target_precision:
            best_threshold, best_precision = threshold, precision
    return best_threshold, best_precision
```

## Dimensionality Trade-offs

Embedding dimensionality is a knob with real, quantifiable trade-offs on three axes: storage cost, query latency, and retrieval quality. A 1536-dimensional float32 vector consumes 6 KB; at 100 million vectors, that's 600 GB just for the raw vector data, before index overhead (HNSW graphs typically add 20-50% on top for edge lists, and metadata adds more). Doubling dimensionality roughly doubles storage and roughly doubles the CPU cost of every distance computation, since distance calculations are `O(d)`. This isn't a minor implementation detail — it's often the dominant cost driver in a vector database bill, more than the number of vectors themselves.

Quality does improve with dimensionality, but with strongly diminishing returns and eventual reversal. Higher dimensional spaces can encode more independent semantic "concepts" without them interfering with each other (more orthogonal directions available), which is why frontier embedding models moved from 384/768 dimensions (early sentence-transformers, BERT-based) to 1536/3072 (OpenAI's `text-embedding-3-large`) and captured measurable quality gains on retrieval benchmarks like MTEB. But past a certain point, adding dimensions mostly adds noise and redundant capacity without adding real signal, while continuing to cost storage and latency linearly.

This tension is exactly why the industry converged on **Matryoshka Representation Learning (MRL)**, used in models like OpenAI's `text-embedding-3` family and Nomic's embeddings. These models are trained so that truncating the vector to its first `k` dimensions (256, 512, 1024, whatever you choose) still produces a valid, well-formed embedding with graceful quality degradation, rather than a corrupted mess. This gives you a runtime dial: store a shorter, cheaper vector for the bulk of your corpus and only use the full-dimensional vector where quality matters most (e.g., a two-stage system that does fast approximate retrieval with 256-dim truncated vectors, then reranks the top candidates with full 3072-dim vectors or a cross-encoder).

```python
def matryoshka_truncate(embedding: np.ndarray, target_dim: int) -> np.ndarray:
    """Truncate and re-normalize an MRL-trained embedding.
    Only valid for models explicitly trained with Matryoshka loss --
    truncating an arbitrary embedding model's output this way produces garbage."""
    truncated = embedding[:target_dim]
    return normalize(truncated)

full_vec = np.random.randn(3072)  # stand-in for a real MRL embedding
small_vec = matryoshka_truncate(full_vec, 256)
print(small_vec.shape)  # (256,) -- 12x smaller, usable for coarse filtering
```

The practical guidance for a production system: don't default to the largest embedding model "because it's presumably best." Benchmark retrieval quality (recall@k on a representative eval set) against dimensionality and cost, and pick the smallest dimensionality that meets your quality bar — the cost curve is much steeper than the quality curve past the point of diminishing returns, especially at scale.

## Scalar and Binary Quantization

Matryoshka truncation shrinks a vector by dropping dimensions; **scalar quantization** and **binary quantization** shrink it a different way, by reducing the precision used to represent each dimension that's kept. Scalar quantization maps each float32 component of a vector onto a much smaller integer range — typically int8, meaning each dimension goes from 4 bytes down to 1 — by computing a per-dimension (or per-vector) min/max range from a representative sample and linearly rescaling values into that range. This is a 4x memory reduction with a comparatively small, well-understood quality cost, and it's supported as a built-in index option in Qdrant, Milvus, and OpenSearch, usually as a toggle rather than something you'd hand-implement.

```python
def scalar_quantize_int8(vectors: np.ndarray):
    """Per-dimension min/max quantization to int8, plus the reconstruction params
    needed to decode back to an approximate float value."""
    v_min = vectors.min(axis=0)
    v_max = vectors.max(axis=0)
    scale = (v_max - v_min) / 255.0
    scale[scale == 0] = 1.0  # avoid divide-by-zero on constant dimensions
    quantized = np.round((vectors - v_min) / scale).astype(np.uint8)
    return quantized, v_min, scale

def dequantize_int8(quantized: np.ndarray, v_min: np.ndarray, scale: np.ndarray):
    return quantized.astype(np.float32) * scale + v_min
```

Binary quantization is the extreme end of this spectrum: each dimension is collapsed to a single bit (typically, sign of the value — positive becomes 1, negative becomes 0), giving a 32x reduction versus float32. Distance between binary vectors is computed via Hamming distance (a simple, extremely fast XOR-and-popcount operation that modern CPUs execute in a handful of cycles), which is why binary-quantized indexes can be dramatically faster to scan than their float equivalents even before considering the memory savings. The quality cost is much steeper than scalar quantization, and binary quantization is generally only usable well when paired with a rerank step — retrieve a large, cheap candidate set using Hamming distance over binary codes, then rerank that shortlist using the original full-precision vectors, recovering most of the accuracy while paying the expensive full-precision distance computation only for a small final set rather than the whole corpus. This retrieve-cheap-then-rerank-exact pattern recurs constantly in this field precisely because it lets you apply aggressive compression where it's cheap (bulk candidate generation) while reserving full fidelity for where it actually affects the final ranking users see.

## Multi-Vector Representations

Everything discussed so far assumes one embedding per document — a single point summarizing an entire passage's meaning. This is a real simplification: a long passage might be relevant to a query because of one specific sentence buried in the middle, and squashing the whole passage into one vector via pooling (mean-pooling or a CLS-token-style single output) can dilute that sentence's signal with everything else in the passage. **Multi-vector models**, most notably the ColBERT family, address this by keeping one embedding per token (or per small span) rather than pooling down to a single vector per document, and computing relevance via a "late interaction" mechanism — for each query token embedding, find its single closest document token embedding (a `MaxSim` operation), then sum those per-token maximum similarities across all query tokens to get the final document score.

```python
def maxsim_late_interaction(query_token_embs: np.ndarray, doc_token_embs: np.ndarray) -> float:
    """ColBERT-style late interaction scoring.
    query_token_embs: (n_query_tokens, d), doc_token_embs: (n_doc_tokens, d)"""
    # similarity matrix: every query token against every doc token
    sims = query_token_embs @ doc_token_embs.T          # (n_query_tokens, n_doc_tokens)
    max_sim_per_query_token = sims.max(axis=1)          # best matching doc token, per query token
    return float(max_sim_per_query_token.sum())
```

This captures fine-grained, token-level relevance that single-vector pooling structurally cannot, and it tends to outperform single-vector dense retrieval on benchmarks requiring precise matching. The cost is substantial: storing and indexing dozens of vectors per document instead of one multiplies storage requirements accordingly, and the `MaxSim` scoring pattern doesn't reduce to a simple ANN nearest-neighbor lookup the way single-vector cosine similarity does, requiring specialized serving infrastructure (or a two-stage design where cheap single-vector ANN search produces a candidate shortlist that multi-vector late interaction then reranks) to be practical at scale. This is exactly the same retrieve-cheap-then-rerank-exact pattern seen with quantization, applied one layer up the stack — a recurring shape worth recognizing across this entire field rather than a coincidence specific to any one technique.

## The Curse of Dimensionality in Vector Search

The curse of dimensionality is a set of related phenomena that make geometric intuition from 2D/3D break down as dimensionality grows, and it has direct, uncomfortable consequences for nearest-neighbor search specifically. The core result, sometimes called "distance concentration," is that as dimensionality increases, the ratio between the distance to the nearest point and the distance to the farthest point in a dataset tends toward 1. In plain terms: in very high-dimensional space, *everything is roughly equidistant from everything else*. Nearest neighbor becomes a less meaningful concept because the gap between "near" and "far" shrinks relative to the absolute distances involved.

```python
import numpy as np

def distance_concentration_demo(n_points=1000, dims_to_test=(2, 10, 50, 200, 1000)):
    """Show how the near/far distance ratio approaches 1 as dimensionality grows."""
    query = None
    for d in dims_to_test:
        points = np.random.randn(n_points, d)
        query = np.random.randn(d)
        dists = np.linalg.norm(points - query, axis=1)
        ratio = dists.min() / dists.max()
        print(f"dim={d:5d}  min/max distance ratio={ratio:.4f}")

distance_concentration_demo()
# Typical output shows ratio climbing from ~0.3 at dim=2 toward ~0.9+ at dim=1000
```

This is precisely why exact brute-force nearest neighbor search, while always *correct*, becomes both computationally expensive (each query is `O(N*d)` against `N` vectors) and, in some naive high-dimensional-index designs, qualitatively less useful, because "nearest" and "10th nearest" can be nearly indistinguishable in raw distance terms. It's also the theoretical justification for why approximate nearest neighbor (ANN) algorithms are not just an optimization but often the *only* reasonable approach at scale: if exact and near-exact neighbors are barely distinguishable in high dimensions anyway, spending enormous compute to guarantee finding the mathematically exact top-1 rather than a very-likely-equivalent approximate top-1 is rarely worth it. This sets up the next chapter's discussion of HNSW, IVF, and product quantization — algorithms explicitly designed to trade a small, controllable amount of recall for large, predictable gains in speed, which is a good trade precisely because the "exactness" being sacrificed is often not that meaningful to begin with in high-dimensional embedding spaces.

It's also worth noting a countervailing real-world effect: real embeddings are not uniformly random high-dimensional points (unlike the synthetic demo above) — they lie on a much lower-dimensional manifold within the ambient space, because the training process concentrates semantically related content into localized clusters and regions. This is why ANN search works as well as it does in practice despite the theoretical curse of dimensionality; the *effective* dimensionality of a well-trained embedding manifold is usually far lower than its *ambient* dimensionality (e.g., 1536 stored dimensions might have an effective intrinsic dimensionality in the dozens), and ANN algorithms exploit that structure even though the curse-of-dimensionality math above assumes no such structure exists.
