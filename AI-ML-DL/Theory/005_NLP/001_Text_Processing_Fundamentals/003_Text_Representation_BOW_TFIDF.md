# Text Representation: Bag of Words and TF-IDF

## Table of Contents

1. [Introduction](#introduction)
2. [Vector Space Model](#vector-space-model)
3. [Bag of Words Representation](#bag-of-words-representation)
4. [Term Frequency-Inverse Document Frequency](#term-frequency-inverse-document-frequency)
5. [Document-Term Matrix](#document-term-matrix)
6. [One-Hot Encoding](#one-hot-encoding)
7. [Document Similarity](#document-similarity)
8. [Word Embeddings Introduction](#word-embeddings-introduction)
9. [Dimensionality and Sparsity](#dimensionality-and-sparsity)
10. [Key Takeaways](#key-takeaways)

## Introduction

Text representation transforms unstructured text into numerical vectors suitable for machine learning algorithms. The choice of representation fundamentally affects model performance and interpretability.

Text is inherently discrete and high-dimensional. Each document can be viewed as a point in a high-dimensional space where dimensions correspond to vocabulary terms. This geometric perspective enables mathematical operations on text.

The bag of words (BoW) model is the simplest text representation, treating documents as unordered collections of words. Despite its simplicity, BoW and its extensions like TF-IDF remain widely used and provide strong baselines for many NLP tasks.

## Vector Space Model

The vector space model (VSM) represents documents as vectors in a high-dimensional space, enabling geometric operations on text.

### Vector Space Assumptions

The VSM makes several assumptions:

**Independence**: Words are independent (bag of words assumption)
**Positional invariance**: Word order doesn't matter
**Sparsity**: Most documents use only a small subset of vocabulary
**Geometric similarity**: Similar documents are close in vector space

These assumptions are simplifications but enable efficient computation and often yield good results.

### Vector Space Geometry

In the vector space:
- **Documents**: Points or vectors
- **Terms**: Dimensions or axes
- **Similarity**: Measured by distance or angle between vectors
- **Queries**: Treated as documents, enabling information retrieval

The dimensionality equals vocabulary size $V$, typically thousands to millions of dimensions.

### Advantages and Limitations

**Advantages**:
- Simple and interpretable
- Enables mathematical operations (addition, scaling, dot products)
- Foundation for many NLP algorithms
- Computationally efficient

**Limitations**:
- Loses word order information
- Ignores semantic relationships
- High dimensionality and sparsity
- Cannot capture context

## Bag of Words Representation

Bag of words represents documents as multisets (bags) of words, ignoring order and structure.

### BoW Construction

Given a vocabulary $V = \{w_1, w_2, \ldots, w_{|V|}\}$, a document $d$ is represented as:

$$\mathbf{d} = [c_1, c_2, \ldots, c_{|V|}]$$

where $c_i$ is the count of word $w_i$ in document $d$.

### Term Frequency

The simplest BoW uses raw term frequencies:

$$tf(w_i, d) = \text{count of } w_i \text{ in } d$$

This gives equal weight to all occurrences, regardless of word importance.

### Binary BoW

Binary BoW indicates presence/absence:

$$bow_i(d) = \begin{cases}
1 & \text{if } w_i \in d \\
0 & \text{otherwise}
\end{cases}$$

Useful when word presence matters more than frequency.

### Normalized BoW

Normalization addresses document length variation:

**L2 normalization**:
$$\mathbf{d}_{norm} = \frac{\mathbf{d}}{||\mathbf{d}||_2}$$

**L1 normalization**:
$$\mathbf{d}_{norm} = \frac{\mathbf{d}}{||\mathbf{d}||_1}$$

Normalization enables fair comparison across documents of different lengths.

### BoW Properties

**Sparsity**: Most entries are zero (documents use small vocabulary subsets)
**High dimensionality**: Dimension equals vocabulary size
**Order independence**: "cat dog" and "dog cat" have identical representations
**Additivity**: Combining documents corresponds to vector addition

## Term Frequency-Inverse Document Frequency

TF-IDF weights terms by their importance, balancing term frequency with inverse document frequency.

### Term Frequency Component

Term frequency measures how often a word appears in a document:

$$tf(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total words in } d}$$

Alternative formulations:
- **Raw count**: $tf(t, d) = \text{count}(t, d)$
- **Log normalization**: $tf(t, d) = 1 + \log(\text{count}(t, d))$
- **Boolean**: $tf(t, d) = 1$ if $t \in d$, else $0$

### Inverse Document Frequency

IDF measures how rare a term is across the corpus:

$$idf(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}$$

where $|D|$ is the number of documents and the denominator is documents containing $t$.

IDF is high for rare terms and low for common terms. The logarithm dampens the effect.

### TF-IDF Formula

TF-IDF combines both components:

$$tf\text{-}idf(t, d, D) = tf(t, d) \times idf(t, D)$$

This gives high weight to:
- Terms frequent in the document (high TF)
- Terms rare across the corpus (high IDF)

Common terms like "the" have low IDF, reducing their influence.

### TF-IDF Variants

Different TF-IDF formulations exist:

**Standard TF-IDF**:
$$tf\text{-}idf(t, d, D) = tf(t, d) \times \log \frac{|D|}{df(t)}$$

**Smooth IDF** (avoids division by zero):
$$idf(t, D) = \log \frac{1 + |D|}{1 + df(t)}$$

**Probabilistic IDF**:
$$idf(t, D) = \log \frac{|D| - df(t)}{df(t)}$$

### TF-IDF Interpretation

TF-IDF values indicate term importance:
- **High TF-IDF**: Term is characteristic of the document
- **Low TF-IDF**: Term is common or absent
- **Zero**: Term doesn't appear in document

TF-IDF effectively identifies distinctive terms that differentiate documents.

## Document-Term Matrix

The document-term matrix (DTM) organizes BoW representations into a matrix for efficient computation.

### Matrix Structure

Given $n$ documents and vocabulary size $V$, the DTM is:

$$\mathbf{X} \in \mathbb{R}^{n \times V}$$

where $X_{ij}$ is the count (or TF-IDF) of term $j$ in document $i$.

### Matrix Properties

**Sparsity**: Most entries are zero (typically 99%+ sparse)
**Storage**: Sparse matrix formats (CSR, CSC) save memory
**Operations**: Matrix multiplication enables batch processing
**Dimensionality**: Columns correspond to vocabulary terms

### Sparse Matrix Formats

Efficient storage for sparse matrices:

**CSR (Compressed Sparse Row)**: Row-oriented, efficient row operations
**CSC (Compressed Sparse Column)**: Column-oriented, efficient column operations
**COO (Coordinate)**: List of (row, col, value) tuples

Sparse formats reduce storage from $O(nV)$ to $O(nnz)$ where $nnz$ is non-zero entries.

### Matrix Operations

Common operations on DTMs:

**Document similarity**: $\mathbf{X}\mathbf{X}^T$ computes pairwise similarities
**Term co-occurrence**: $\mathbf{X}^T\mathbf{X}$ computes term-term relationships
**Dimensionality reduction**: SVD, PCA on $\mathbf{X}$
**Clustering**: Apply clustering algorithms to rows (documents)

## One-Hot Encoding

One-hot encoding represents categorical variables as binary vectors with a single 1.

### One-Hot Vectors

For vocabulary $V$ with $|V|$ words, each word $w_i$ has a one-hot vector:

$$\mathbf{e}_i = [0, 0, \ldots, 1, \ldots, 0]$$

where the 1 is at position $i$.

### Properties

**Orthogonality**: One-hot vectors are orthogonal ($\mathbf{e}_i \cdot \mathbf{e}_j = 0$ for $i \neq j$)
**Unit norm**: Each vector has L2 norm of 1
**High dimensionality**: Dimension equals vocabulary size
**No similarity**: All words are equally distant

### Limitations

One-hot encoding has significant limitations:

**No semantic similarity**: "cat" and "dog" are as distant as "cat" and "philosophy"
**Curse of dimensionality**: Very high-dimensional sparse vectors
**No context**: Cannot capture word relationships
**Inefficient**: Most dimensions are zero

These limitations motivate distributed representations (word embeddings).

### Applications

Despite limitations, one-hot encoding is used for:
- **Input to neural networks**: First layer learns embeddings
- **Categorical features**: In feature engineering
- **Baseline representations**: Simple starting point

## Document Similarity

Similarity measures quantify how similar two documents are in vector space.

### Cosine Similarity

Cosine similarity measures the angle between vectors:

$$\text{cosine}(\mathbf{d}_1, \mathbf{d}_2) = \frac{\mathbf{d}_1 \cdot \mathbf{d}_2}{||\mathbf{d}_1||_2 ||\mathbf{d}_2||_2}$$

Range: $[-1, 1]$ (typically $[0, 1]$ for non-negative vectors)

**Properties**:
- Length-invariant: Normalizes for document length
- Efficient: Computable via dot product
- Interpretable: 1 = identical, 0 = orthogonal

### Euclidean Distance

Euclidean distance measures straight-line distance:

$$d(\mathbf{d}_1, \mathbf{d}_2) = ||\mathbf{d}_1 - \mathbf{d}_2||_2 = \sqrt{\sum_i (d_{1i} - d_{2i})^2}$$

**Properties**:
- Sensitive to document length
- Lower values indicate greater similarity
- Not normalized

### Jaccard Similarity

Jaccard similarity for sets (binary BoW):

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Measures overlap relative to union, range $[0, 1]$.

### Dot Product

Simple dot product (unnormalized):

$$\mathbf{d}_1 \cdot \mathbf{d}_2 = \sum_i d_{1i} d_{2i}$$

Affected by document length and term frequencies.

### Similarity Choice

Choice depends on application:

**Cosine**: Best for length-normalized representations (TF-IDF)
**Euclidean**: When absolute differences matter
**Jaccard**: For set-based representations
**Dot product**: When magnitude matters (e.g., relevance scoring)

## Word Embeddings Introduction

Word embeddings map words to dense, low-dimensional vectors that capture semantic relationships.

### Distributed Representations

Unlike one-hot encoding, embeddings use distributed representations:
- **Dense**: Few zero entries
- **Low-dimensional**: Typically 50-300 dimensions
- **Semantic**: Similar words have similar vectors
- **Learned**: From data, not hand-crafted

### Embedding Properties

Good embeddings capture:
- **Semantic similarity**: Related words are close
- **Analogical relationships**: "king" - "man" + "woman" ≈ "queen"
- **Context**: Words in similar contexts have similar vectors
- **Compositionality**: Word combinations can be computed

### Embedding Methods

Major approaches:
- **Word2Vec**: Predicts context (CBOW) or word (Skip-gram)
- **GloVe**: Global matrix factorization
- **FastText**: Character n-gram embeddings
- **Contextual**: BERT, ELMo (context-dependent)

### From BoW to Embeddings

Embeddings address BoW limitations:
- **Semantic similarity**: "cat" and "dog" are close
- **Dimensionality**: Much lower than vocabulary size
- **Generalization**: Can handle unseen words (subword models)
- **Composition**: Can combine word vectors

However, embeddings lose interpretability compared to BoW.

## Dimensionality and Sparsity

High dimensionality and sparsity are fundamental challenges in text representation.

### Curse of Dimensionality

As dimensionality increases:
- **Distance metrics**: All points become equidistant
- **Volume**: Most volume is near the boundary
- **Sparsity**: Data becomes sparse
- **Overfitting**: Models overfit easily

For text, $V$ can be $10^5$ to $10^6$, making dimensionality reduction crucial.

### Sparsity in Text

Text vectors are extremely sparse:
- Documents use $\ll V$ unique words
- Typical sparsity: 99%+ zeros
- Sparse storage: Essential for efficiency
- Sparse operations: Specialized algorithms needed

### Dimensionality Reduction

Techniques to reduce dimensionality:

**Feature selection**: Choose important terms
**Feature extraction**: Learn lower-dimensional representations
**Matrix factorization**: SVD, NMF, LSA
**Embeddings**: Learn dense low-dimensional vectors

### Trade-offs

Dimensionality reduction involves trade-offs:
- **Information loss**: Lower dimensions lose information
- **Computational cost**: Reduction itself is expensive
- **Interpretability**: Lower dimensions may be less interpretable
- **Generalization**: Can improve generalization

## Key Takeaways

1. **Vector space model enables geometric operations**: Representing text as vectors allows mathematical operations like similarity computation and clustering.

2. **Bag of words is simple but effective**: Despite ignoring word order, BoW provides strong baselines and remains widely used.

3. **TF-IDF balances frequency and rarity**: By weighting terms by both document frequency and corpus rarity, TF-IDF identifies distinctive terms effectively.

4. **Sparsity is fundamental**: Text representations are extremely sparse, requiring specialized storage and algorithms for efficiency.

5. **Document-term matrices enable batch processing**: Organizing BoW vectors into matrices enables efficient computation on large corpora.

6. **One-hot encoding has severe limitations**: No semantic similarity, high dimensionality, and sparsity make one-hot encoding inadequate for many tasks.

7. **Similarity measures depend on representation**: Cosine similarity works well with TF-IDF, while other measures suit different representations.

8. **Word embeddings address BoW limitations**: Dense, low-dimensional embeddings capture semantic relationships that BoW cannot, motivating modern NLP approaches.
