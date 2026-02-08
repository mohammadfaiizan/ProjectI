# Word Embeddings: Word2Vec and GloVe

## Table of Contents

1. [Introduction](#introduction)
2. [Distributed Representations](#distributed-representations)
3. [Word2Vec Architecture](#word2vec-architecture)
4. [Continuous Bag of Words](#continuous-bag-of-words)
5. [Skip-Gram Model](#skip-gram-model)
6. [GloVe: Global Vectors](#glove-global-vectors)
7. [FastText and Subword Embeddings](#fasttext-and-subword-embeddings)
8. [Embedding Evaluation](#embedding-evaluation)
9. [Properties and Applications](#properties-and-applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Word embeddings map words to dense, low-dimensional vectors that capture semantic and syntactic relationships. Word2Vec and GloVe revolutionized NLP by learning high-quality word representations from large unlabeled corpora.

Traditional representations (one-hot encoding) suffer from:
- **No semantic similarity**: All words equally distant
- **High dimensionality**: Vocabulary-sized vectors
- **Sparsity**: Mostly zeros

Word embeddings address these by learning dense vectors where similar words are close in vector space, enabling transfer learning and improved performance on downstream tasks.

## Distributed Representations

Distributed representations encode meaning across multiple dimensions, enabling rich semantic relationships.

### One-Hot Limitations

One-hot encoding represents word $w_i$ as:

$$\mathbf{e}_i = [0, \ldots, 1, \ldots, 0]$$

**Problems**:
- **No similarity**: $\mathbf{e}_i \cdot \mathbf{e}_j = 0$ for $i \neq j$
- **High dimensionality**: $V$ dimensions for vocabulary size $V$
- **No generalization**: Cannot handle unseen words

### Distributed Representations

Word embeddings use dense vectors:

$$\mathbf{w} \in \mathbb{R}^d$$

where $d \ll V$ (typically 50-300 dimensions).

**Advantages**:
- **Semantic similarity**: Similar words have similar vectors
- **Low dimensionality**: Much smaller than vocabulary
- **Compositionality**: Can combine word vectors
- **Generalization**: Captures regularities

### Distributional Hypothesis

**Distributional hypothesis**: Words that appear in similar contexts have similar meanings.

This hypothesis underlies most embedding methods: learn representations from co-occurrence patterns.

## Word2Vec Architecture

Word2Vec learns embeddings by predicting words from their contexts or contexts from words.

### Two Architectures

**CBOW (Continuous Bag of Words)**: Predict target word from context
**Skip-gram**: Predict context words from target word

Both use shallow neural networks with one hidden layer.

### Training Objective

Learn embeddings that maximize:

$$\sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where $c$ is context window size and $T$ is corpus length.

### Negative Sampling

Instead of softmax over entire vocabulary (expensive), use negative sampling:

**Positive examples**: $(w_t, w_{t+j})$ pairs from corpus
**Negative examples**: Random word pairs $(w_t, w_k)$ where $w_k$ doesn't appear in context

**Objective**:
$$\log \sigma(\mathbf{w}_{t+j}^T \mathbf{w}_t) + \sum_{i=1}^{k} \mathbb{E}_{w_n \sim P_n} [\log \sigma(-\mathbf{w}_n^T \mathbf{w}_t)]$$

where $k$ is number of negative samples and $P_n$ is noise distribution.

## Continuous Bag of Words

CBOW predicts the center word from surrounding context words.

### CBOW Architecture

**Input**: Context words $w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}$
**Hidden layer**: Average of input word embeddings
**Output**: Probability distribution over vocabulary

**Forward pass**:
$$\mathbf{h} = \frac{1}{2c} \sum_{-c \leq j \leq c, j \neq 0} \mathbf{W}_{input} \mathbf{w}_{t+j}$$

$$P(w_t | \text{context}) = \text{softmax}(\mathbf{W}_{output} \mathbf{h})$$

### CBOW Training

**Loss function**: Cross-entropy
$$L = -\log P(w_t | \text{context})$$

**Gradient descent**: Update embeddings via backpropagation

**Efficiency**: Use hierarchical softmax or negative sampling to avoid $O(V)$ computation.

### CBOW Properties

**Advantages**:
- Faster training (averages context)
- Works well with frequent words
- Smooth representations

**Disadvantages**:
- Averages context (loses information)
- Less effective for rare words

## Skip-Gram Model

Skip-gram predicts context words from the center word, often performing better than CBOW.

### Skip-Gram Architecture

**Input**: Center word $w_t$
**Output**: Probability distributions for each context position

**Forward pass**:
$$\mathbf{h} = \mathbf{W}_{input} \mathbf{w}_t$$

$$P(w_{t+j} | w_t) = \text{softmax}(\mathbf{W}_{output} \mathbf{h})$$

for each context position $j \in \{-c, \ldots, -1, 1, \ldots, c\}$.

### Skip-Gram Training

**Objective**: Maximize probability of context words:

$$L = -\sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

**Negative sampling**: Sample $k$ negative words per context:

$$L = -\log \sigma(\mathbf{w}_{t+j}^T \mathbf{w}_t) - \sum_{i=1}^{k} \log \sigma(-\mathbf{w}_{n_i}^T \mathbf{w}_t)$$

### Skip-Gram Properties

**Advantages**:
- Better for rare words
- More training examples per word pair
- Often outperforms CBOW

**Disadvantages**:
- Slower training
- More parameters to update

### Subsampling

Subsample frequent words to balance training:

$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

where $t$ is threshold and $f(w_i)$ is word frequency.

Discards very frequent words (e.g., "the", "a") to focus on informative words.

## GloVe: Global Vectors

GloVe combines global matrix factorization with local context window methods.

### GloVe Motivation

**Word2Vec limitations**:
- Local context windows
- Doesn't use global co-occurrence statistics
- Inefficient use of training data

**GloVe insight**: Use global word-word co-occurrence counts.

### Co-Occurrence Matrix

Build matrix $\mathbf{X}$ where $X_{ij}$ is count of word $j$ appearing in context of word $i$:

$$X_{ij} = \sum_{t=1}^{T} \mathbb{1}(w_t = i, w_{t+j} = j)$$

**Weighting**: Use distance-weighted counts (closer words weighted more).

### GloVe Objective

Learn embeddings such that:

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$$

where $\mathbf{w}_i$ and $\tilde{\mathbf{w}}_j$ are word and context embeddings, $b_i$ and $\tilde{b}_j$ are biases.

**Loss function**:
$$J = \sum_{i,j=1}^{V} f(X_{ij}) (\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

where $f(X_{ij})$ is weighting function:

$$f(x) = \begin{cases}
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}$$

with $\alpha = 0.75$ and $x_{max} = 100$.

### GloVe Training

**Stochastic gradient descent**: Sample co-occurrence pairs
**Final embeddings**: Use $\mathbf{w}_i + \tilde{\mathbf{w}}_i$ (sum of word and context embeddings)

**Advantages**:
- Uses global statistics
- Efficient training
- Good performance

## FastText and Subword Embeddings

FastText extends Word2Vec to handle subword units, enabling handling of rare and unseen words.

### FastText Architecture

FastText represents words as sum of character n-gram embeddings:

$$\mathbf{w} = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$

where $\mathcal{G}_w$ is set of character n-grams in word $w$ and $\mathbf{z}_g$ are n-gram embeddings.

### Character N-Grams

**Example**: "where" with $n=3$:
- `<wh`, `whe`, `her`, `ere`, `re>`
- `<where>` (word itself)

**Benefits**:
- Handles OOV words
- Captures morphology
- Shares representations across words

### FastText Training

Same as Word2Vec (CBOW or Skip-gram) but:
- Input/output uses n-gram sums
- Learns both word and n-gram embeddings

**Advantages**:
- Handles rare words
- Morphological awareness
- Multilingual (character-level)

## Embedding Evaluation

Evaluating word embeddings measures how well they capture semantic and syntactic relationships.

### Intrinsic Evaluation

**Word similarity**: Correlate embedding similarity with human similarity judgments
- **Datasets**: WordSim-353, SimLex-999
- **Metric**: Spearman correlation

**Word analogy**: Solve analogies like "man:woman :: king:?"
- **Method**: Find word maximizing $\cos(\mathbf{w}_{woman} - \mathbf{w}_{man} + \mathbf{w}_{king}, \mathbf{w}_?)$
- **Datasets**: Google analogy dataset

**Categorization**: Cluster words into semantic categories

### Extrinsic Evaluation

Evaluate on downstream tasks:
- **Named entity recognition**: Use embeddings as features
- **Sentiment analysis**: Classification performance
- **Machine translation**: Translation quality
- **Question answering**: QA accuracy

Extrinsic evaluation is ultimate test but confounded by other factors.

### Embedding Properties

Good embeddings capture:
- **Semantic similarity**: Related words are close
- **Analogical relationships**: "king - man + woman ≈ queen"
- **Compositionality**: Can combine word vectors
- **Clustering**: Words group by meaning

## Properties and Applications

Word embeddings enable various NLP applications and exhibit interesting properties.

### Semantic Relationships

Embeddings capture:
- **Synonymy**: Similar meanings → similar vectors
- **Antonymy**: Opposites may be close (same domain)
- **Hypernymy**: Hierarchical relationships
- **Analogy**: Linear relationships (king - man + woman ≈ queen)

### Arithmetic Properties

Word embeddings support vector arithmetic:

$$\mathbf{w}_{queen} \approx \mathbf{w}_{king} - \mathbf{w}_{man} + \mathbf{w}_{woman}$$

This property emerges from training but isn't explicitly enforced.

### Applications

**Initialization**: Initialize neural network input layers
**Feature extraction**: Use as features for traditional ML
**Similarity computation**: Find similar words/documents
**Clustering**: Group words by meaning
**Visualization**: t-SNE, PCA for exploration

### Transfer Learning

Pre-trained embeddings enable:
- **Few-shot learning**: Learn from few examples
- **Domain adaptation**: Transfer to new domains
- **Multilingual**: Cross-lingual transfer

## Key Takeaways

1. **Word embeddings capture semantic relationships**: Dense vector representations enable similarity computation and transfer learning, addressing limitations of sparse one-hot encoding.

2. **Word2Vec learns from local context**: CBOW and Skip-gram models predict words from contexts or contexts from words, learning embeddings that capture distributional semantics.

3. **Skip-gram often outperforms CBOW**: Predicting multiple context words from center word provides more training signal and better handles rare words.

4. **GloVe combines global and local information**: Using global co-occurrence statistics while maintaining local context window benefits enables efficient and effective embedding learning.

5. **FastText handles morphology and OOV**: Character n-gram embeddings enable handling of rare and unseen words while capturing morphological structure.

6. **Negative sampling enables efficient training**: Approximating softmax with negative sampling makes training feasible for large vocabularies.

7. **Embedding evaluation is multifaceted**: Intrinsic (similarity, analogy) and extrinsic (downstream tasks) evaluation capture different aspects of embedding quality.

8. **Pre-trained embeddings are powerful**: Transfer learning with pre-trained embeddings improves performance across many NLP tasks with minimal additional training.
