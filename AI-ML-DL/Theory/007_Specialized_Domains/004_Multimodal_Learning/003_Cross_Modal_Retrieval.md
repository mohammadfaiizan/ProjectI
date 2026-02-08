# Cross-Modal Retrieval

## Table of Contents

1. [Introduction to Cross-Modal Retrieval](#introduction-to-cross-modal-retrieval)
2. [Cross-Modal Similarity Learning](#cross-modal-similarity-learning)
3. [Joint Embedding Spaces](#joint-embedding-spaces)
4. [Triplet Loss for Retrieval](#triplet-loss-for-retrieval)
5. [Image-Text Retrieval](#image-text-retrieval)
6. [Zero-Shot Retrieval](#zero-shot-retrieval)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Advanced Retrieval Methods](#advanced-retrieval-methods)
9. [Scalability and Efficiency](#scalability-and-efficiency)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Cross-Modal Retrieval

Cross-modal retrieval enables searching across different modalities, such as finding images from text queries or text from image queries.

### Problem Formulation

**Query Modality**: $\mathbf{q} \in \mathcal{Q}$ (e.g., text)

**Target Modality**: $\mathbf{t} \in \mathcal{T}$ (e.g., images)

**Goal**: Retrieve relevant targets $\mathbf{t}^*$ for query $\mathbf{q}$

### Retrieval Tasks

**Image-to-Text**: Given image, find relevant text descriptions

**Text-to-Image**: Given text, find relevant images

**Video-to-Text**: Given video, find relevant text

**Audio-to-Text**: Given audio, find relevant text

### Challenges

1. **Modality Gap**: Different representations (pixels vs words)
2. **Semantic Alignment**: Match semantic meaning across modalities
3. **Scale**: Large-scale retrieval requires efficient methods
4. **Fine-Grained**: Distinguish similar items

### Applications

- **Search Engines**: Multimodal search
- **E-commerce**: Find products from descriptions
- **Content Recommendation**: Recommend based on cross-modal queries
- **Accessibility**: Find content across modalities

---

## Cross-Modal Similarity Learning

Cross-modal similarity learning aims to learn similarity functions that work across modalities.

### Similarity Functions

**Cosine Similarity**:

$$s(\mathbf{q}, \mathbf{t}) = \frac{\mathbf{q}^T \mathbf{t}}{\|\mathbf{q}\| \|\mathbf{t}\|}$$

**Euclidean Distance**:

$$d(\mathbf{q}, \mathbf{t}) = \|\mathbf{q} - \mathbf{t}\|_2$$

**Learned Similarity**: 

$$s(\mathbf{q}, \mathbf{t}) = f_\theta(\mathbf{q}, \mathbf{t})$$

where $f_\theta$ is a neural network.

### Embedding Learning

**Separate Encoders**: 

$$\mathbf{e}_q = f_q(\mathbf{q}), \quad \mathbf{e}_t = f_t(\mathbf{t})$$

**Joint Space**: Map to shared embedding space

**Similarity**: Compute similarity in embedding space

### Contrastive Learning

**Positive Pairs**: Matching query-target pairs $(\mathbf{q}, \mathbf{t}^+)$

**Negative Pairs**: Non-matching pairs $(\mathbf{q}, \mathbf{t}^-)$

**Objective**: Maximize similarity for positives, minimize for negatives

### Hard Negative Mining

**Random Negatives**: Sample random non-matching pairs

**Hard Negatives**: Select difficult negatives:

$$\mathbf{t}^- = \arg\max_{\mathbf{t} \neq \mathbf{t}^+} s(\mathbf{q}, \mathbf{t})$$

**Semi-Hard Negatives**: Select negatives within margin

---

## Joint Embedding Spaces

Joint embedding spaces map different modalities to a common space where similarity can be computed.

### Shared Embedding Space

**Image Encoder**: $f_I(\mathbf{x}_I) \in \mathbb{R}^d$

**Text Encoder**: $f_T(\mathbf{x}_T) \in \mathbb{R}^d$

**Same Dimension**: Both map to $d$-dimensional space

### Learning Objectives

**Contrastive Loss**: 

$$\mathcal{L} = -\log \frac{\exp(s(\mathbf{q}, \mathbf{t}^+) / \tau)}{\exp(s(\mathbf{q}, \mathbf{t}^+) / \tau) + \sum_{i=1}^{N} \exp(s(\mathbf{q}, \mathbf{t}_i^-) / \tau)}$$

**Triplet Loss**: 

$$\mathcal{L} = \max(0, d(\mathbf{q}, \mathbf{t}^+) - d(\mathbf{q}, \mathbf{t}^-) + m)$$

**Ranking Loss**: 

$$\mathcal{L} = \sum_{i=1}^{N} \max(0, s(\mathbf{q}, \mathbf{t}_i^-) - s(\mathbf{q}, \mathbf{t}^+) + m)$$

### Normalization

**L2 Normalization**: Normalize embeddings to unit sphere:

$$\mathbf{e} \leftarrow \frac{\mathbf{e}}{\|\mathbf{e}\|_2}$$

**Benefits**: 
- Bounds similarity values
- Improves training stability
- Enables efficient similarity computation

### Multi-Modal Embeddings

**Concatenation**: $[\mathbf{e}_I; \mathbf{e}_T]$

**Addition**: $\mathbf{e}_I + \mathbf{e}_T$

**Multiplication**: $\mathbf{e}_I \odot \mathbf{e}_T$

**Bilinear**: $\mathbf{e}_I^T W \mathbf{e}_T$

---

## Triplet Loss for Retrieval

Triplet loss is widely used for learning cross-modal embeddings for retrieval.

### Triplet Formulation

**Anchor**: Query $\mathbf{q}$

**Positive**: Matching target $\mathbf{t}^+$

**Negative**: Non-matching target $\mathbf{t}^-$

**Loss**:

$$\mathcal{L} = \max(0, d(\mathbf{q}, \mathbf{t}^+) - d(\mathbf{q}, \mathbf{t}^-) + m)$$

where $m$ is margin.

### Triplet Mining

**Random Triplets**: Sample random triplets

**Hard Negative Mining**: Select hard negatives:

$$\mathbf{t}^- = \arg\max_{\mathbf{t} \neq \mathbf{t}^+} s(\mathbf{q}, \mathbf{t})$$

**Semi-Hard Negatives**: Select negatives within margin

**Hard Positive Mining**: Select hard positives (less common)

### Batch Hard Mining

**Within Batch**: Use hardest negatives within batch:

$$\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \max(0, d(\mathbf{q}_i, \mathbf{t}_i^+) - \min_{\mathbf{t}_j^- \neq \mathbf{t}_i^+} d(\mathbf{q}_i, \mathbf{t}_j^-) + m)$$

**Benefits**: More efficient, better learning signal

### Multi-Triplet Loss

**Multiple Negatives**: Compare against multiple negatives:

$$\mathcal{L} = \sum_{i=1}^{N} \max(0, d(\mathbf{q}, \mathbf{t}^+) - d(\mathbf{q}, \mathbf{t}_i^-) + m)$$

### Margin Selection

**Fixed Margin**: $m = 0.2$ or $m = 1.0$

**Adaptive Margin**: Learn margin per sample

**Soft Margin**: Use smooth approximation

---

## Image-Text Retrieval

Image-text retrieval is a fundamental cross-modal retrieval task with wide applications.

### Problem Setup

**Image Corpus**: $\mathcal{I} = \{\mathbf{x}_I^{(i)}\}_{i=1}^{N}$

**Text Corpus**: $\mathcal{T} = \{\mathbf{x}_T^{(j)}\}_{i=1}^{M}$

**Task**: Given image, retrieve relevant texts or vice versa

### CLIP for Retrieval

**Embeddings**: 

$$\mathbf{e}_I = f_I(\mathbf{x}_I), \quad \mathbf{e}_T = f_T(\mathbf{x}_T)$$

**Similarity**: 

$$s(\mathbf{x}_I, \mathbf{x}_T) = \frac{\mathbf{e}_I^T \mathbf{e}_T}{\|\mathbf{e}_I\| \|\mathbf{e}_T\|}$$

**Retrieval**: Rank by similarity

### Fine-Grained Retrieval

**Region-Word Alignment**: Align image regions to words

**Phrase-Region Matching**: Match phrases to regions

**Attention Maps**: Visualize which regions correspond to text

### Hierarchical Retrieval

**Coarse-to-Fine**: First retrieve coarse matches, then refine

**Multi-Scale**: Use multiple image/text resolutions

**Ensemble**: Combine multiple retrieval methods

### Cross-Modal Attention

**Image-to-Text**: Image features attend to text:

$$\mathbf{t}' = \sum_{i} \alpha_i \mathbf{v}_i$$

**Text-to-Image**: Text features attend to image:

$$\mathbf{v}' = \sum_{j} \beta_j \mathbf{t}_j$$

---

## Zero-Shot Retrieval

Zero-shot retrieval enables retrieving items from unseen classes or domains.

### Zero-Shot Setup

**Training Classes**: $\mathcal{C}_{\text{train}}$

**Test Classes**: $\mathcal{C}_{\text{test}}$ where $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$

**Goal**: Retrieve items from test classes

### Attribute-Based Retrieval

**Attributes**: Learn attribute representations

**Composition**: Compose attributes for unseen classes

**Retrieval**: Match query to attribute composition

### Semantic Embeddings

**Word Embeddings**: Use word2vec, GloVe, BERT

**Class Embeddings**: Embed class names or descriptions

**Retrieval**: Match query to class embeddings

### Prompt-Based Retrieval

**Text Prompts**: "a photo of a {class}"

**Embedding**: Embed prompts

**Retrieval**: Match query to prompt embeddings

### Transfer Learning

**Pre-training**: Large-scale pre-training on diverse data

**Fine-tuning**: Adapt to target domain

**Domain Adaptation**: Adapt across domains

---

## Evaluation Metrics

Evaluation metrics measure retrieval performance across different aspects.

### Recall@K

**Definition**: Percentage of relevant items in top-$K$ results

$$\text{Recall}@K = \frac{|\{\text{relevant items in top-}K\}|}{|\{\text{all relevant items}\}|}$$

**Common**: Recall@1, Recall@5, Recall@10

### Mean Reciprocal Rank (MRR)

**Definition**: Average reciprocal rank of first relevant item

$$\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is rank of first relevant item for query $q$.

### Normalized Discounted Cumulative Gain (NDCG)

**Definition**: Measures ranking quality with position discounting

$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

where DCG discounts lower positions and IDCG is ideal DCG.

### Mean Average Precision (mAP)

**Definition**: Average precision across all queries

$$\text{AP} = \frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} P@k \cdot \text{rel}(k)$$

where $\mathcal{R}$ is ranked results and $\text{rel}(k)$ indicates relevance.

### Median Rank

**Definition**: Median rank of first relevant item

**Robust**: Less sensitive to outliers than mean rank

### R-Precision

**Definition**: Precision at rank equal to number of relevant items

$$\text{R-Precision} = \frac{|\{\text{relevant in top-}R\}|}{R}$$

where $R$ is number of relevant items.

---

## Advanced Retrieval Methods

### Re-Ranking

**Two-Stage**: Coarse retrieval + fine re-ranking

**Cross-Attention**: Use attention for re-ranking

**Interaction Models**: Model query-target interactions

### Dense Retrieval

**Dense Embeddings**: Learn dense vector representations

**ANN Search**: Approximate nearest neighbor search

**Scalability**: Handle millions of items efficiently

### Sparse Retrieval

**Sparse Representations**: Use sparse vectors (e.g., BM25)

**Hybrid**: Combine dense and sparse retrieval

**Interpretability**: Sparse methods are more interpretable

### Multi-Modal Hashing

**Binary Codes**: Learn binary hash codes

**Hamming Distance**: Fast similarity computation

**Scalability**: Very efficient for large-scale retrieval

### Cross-Modal Graph

**Graph Construction**: Build graph connecting modalities

**Graph Neural Networks**: Use GNNs for retrieval

**Propagation**: Propagate information across graph

---

## Scalability and Efficiency

Large-scale cross-modal retrieval requires efficient methods.

### Approximate Nearest Neighbor (ANN)

**Exact Search**: $O(N)$ for $N$ items (too slow)

**ANN Methods**:
- **LSH**: Locality-Sensitive Hashing
- **IVF**: Inverted File Index
- **HNSW**: Hierarchical Navigable Small World
- **FAISS**: Facebook AI Similarity Search

### Quantization

**Product Quantization**: Compress embeddings

**Scalar Quantization**: Quantize to fewer bits

**Binary Quantization**: Binary codes for extreme efficiency

### Indexing

**Inverted Index**: Index by features

**Multi-Index**: Multiple indices for different aspects

**Hierarchical Index**: Tree-based indexing

### Distributed Retrieval

**Sharding**: Partition data across machines

**Parallel Search**: Search in parallel

**Aggregation**: Combine results from shards

### Caching

**Query Cache**: Cache frequent queries

**Result Cache**: Cache retrieval results

**Embedding Cache**: Cache computed embeddings

---

## Key Takeaways

1. **Cross-Modal Retrieval**: Enables searching across modalities (image-text, video-text) by learning joint embeddings where similarity can be computed, supporting applications in search, e-commerce, and accessibility.

2. **Similarity Learning**: Learns cross-modal similarity functions (cosine, Euclidean, learned) through contrastive learning, mapping different modalities to shared embedding spaces for comparison.

3. **Joint Embedding Spaces**: Maps different modalities to common $d$-dimensional space using separate encoders, enabling similarity computation through normalized embeddings and contrastive/triplet losses.

4. **Triplet Loss**: Widely used for retrieval, comparing anchor-positive-negative triplets with margin, using hard negative mining and batch hard mining for efficient and effective learning.

5. **Image-Text Retrieval**: Fundamental task using CLIP-style embeddings for similarity-based ranking, with fine-grained methods (region-word alignment) and hierarchical retrieval for improved performance.

6. **Zero-Shot Retrieval**: Enables retrieving unseen classes through attribute-based composition, semantic embeddings (word/class embeddings), prompt-based methods, and transfer learning from large-scale pre-training.

7. **Evaluation Metrics**: Recall@K, MRR, NDCG, mAP measure different aspects of retrieval quality, with Recall@K most common for reporting results.

8. **Advanced Methods**: Re-ranking (two-stage, cross-attention), dense/sparse retrieval, multi-modal hashing, and graph-based methods improve accuracy and efficiency for large-scale retrieval.

9. **Scalability**: ANN methods (LSH, HNSW, FAISS), quantization (product, binary), indexing (inverted, hierarchical), distributed retrieval, and caching enable efficient retrieval at scale.

10. **Challenges**: Modality gap, semantic alignment, fine-grained distinctions, and scalability remain active research areas, with contrastive learning and joint embeddings providing strong foundations for cross-modal retrieval systems.
