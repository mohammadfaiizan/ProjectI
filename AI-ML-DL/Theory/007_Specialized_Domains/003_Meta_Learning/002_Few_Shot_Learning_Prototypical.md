# Few-Shot Learning and Prototypical Networks

## Table of Contents

1. [Introduction to Few-Shot Learning](#introduction-to-few-shot-learning)
2. [Prototypical Networks](#prototypical-networks)
3. [Matching Networks](#matching-networks)
4. [Relation Networks](#relation-networks)
5. [Siamese Networks](#siamese-networks)
6. [Metric Learning for Few-Shot](#metric-learning-for-few-shot)
7. [Episode Training Protocol](#episode-training-protocol)
8. [N-Way K-Shot Setup](#n-way-k-shot-setup)
9. [Advanced Few-Shot Methods](#advanced-few-shot-methods)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Few-Shot Learning

Few-shot learning aims to learn from very few examples, typically 1-5 examples per class, by leveraging prior knowledge from related tasks.

### Problem Formulation

**Standard Supervised Learning**: Large labeled dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ with $N \gg 1$.

**Few-Shot Learning**: Very few labeled examples per class:
- **1-shot**: 1 example per class
- **5-shot**: 5 examples per class
- **K-shot**: K examples per class

### N-Way K-Shot Classification

**N-way**: $N$ classes to distinguish
**K-shot**: $K$ examples per class for training

**Support Set**: $\mathcal{S} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{NK}$ training examples

**Query Set**: $\mathcal{Q} = \{(\mathbf{x}_j, y_j)\}_{j=1}^{M}$ test examples

**Goal**: Classify query examples using support set.

### Challenges

1. **Limited Data**: Too few examples for standard training
2. **Overfitting**: Model memorizes training examples
3. **Generalization**: Must generalize to new classes
4. **Transfer**: Leverage knowledge from base classes

### Approaches

1. **Metric Learning**: Learn distance metrics (Prototypical, Matching, Relation networks)
2. **Meta-Learning**: Learn to learn (MAML, Reptile)
3. **Data Augmentation**: Generate more examples
4. **Transfer Learning**: Fine-tune pre-trained models

---

## Prototypical Networks

Prototypical networks learn a metric space where classification is performed by computing distances to prototype representations of each class.

### Prototype Computation

For each class $k$, compute prototype:

$$\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum_{(\mathbf{x}_i, y_i) \in \mathcal{S}_k} f_\phi(\mathbf{x}_i)$$

where:
- $\mathcal{S}_k$ is support set for class $k$
- $f_\phi$ is embedding function (neural network)
- $\mathbf{c}_k$ is prototype (class center)

### Classification

For query example $\mathbf{x}$, compute distances to prototypes:

$$d(\mathbf{x}, \mathbf{c}_k) = \|f_\phi(\mathbf{x}) - \mathbf{c}_k\|_2^2$$

**Probability Distribution**:

$$p(y=k | \mathbf{x}) = \frac{\exp(-d(\mathbf{x}, \mathbf{c}_k))}{\sum_{j=1}^{N} \exp(-d(\mathbf{x}, \mathbf{c}_j))}$$

Uses softmax over negative distances (Euclidean distance).

### Training Objective

**Negative Log-Likelihood**:

$$\mathcal{L} = -\sum_{(\mathbf{x}, y) \in \mathcal{Q}} \log p(y | \mathbf{x})$$

**Episodic Training**: Sample episodes with support and query sets.

### Distance Metrics

**Euclidean Distance**: $\|f_\phi(\mathbf{x}) - \mathbf{c}_k\|_2$

**Cosine Distance**: $1 - \frac{f_\phi(\mathbf{x})^T \mathbf{c}_k}{\|f_\phi(\mathbf{x})\| \|\mathbf{c}_k\|}$

**Mahalanobis Distance**: $(f_\phi(\mathbf{x}) - \mathbf{c}_k)^T M (f_\phi(\mathbf{x}) - \mathbf{c}_k)$

### Advantages

- **Simple**: Easy to implement and understand
- **Efficient**: Fast inference, no need to store all examples
- **Interpretable**: Prototypes represent class centers
- **Effective**: Strong performance on few-shot tasks

### Limitations

- **Single Prototype**: Assumes unimodal class distributions
- **Distance Choice**: Euclidean distance may not be optimal
- **Embedding Quality**: Depends on learned embeddings

---

## Matching Networks

Matching networks use attention mechanisms to match query examples to support examples, enabling flexible few-shot learning.

### Architecture

**Embedding Functions**:
- **Support Embedder**: $g(\mathbf{x}_i)$ for support examples
- **Query Embedder**: $f(\mathbf{x})$ for query examples

**Attention Mechanism**: Match query to support:

$$a(\mathbf{x}, \mathbf{x}_i) = \frac{\exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_i)))}{\sum_{j=1}^{|\mathcal{S}|} \exp(\text{cosine}(f(\mathbf{x}), g(\mathbf{x}_j)))}$$

### Prediction

**Weighted Sum**:

$$p(y | \mathbf{x}, \mathcal{S}) = \sum_{i=1}^{|\mathcal{S}|} a(\mathbf{x}, \mathbf{x}_i) y_i$$

where $y_i$ is one-hot encoding of support label.

### Full Context Embeddings

**Bidirectional LSTM**: Encode support set with context:

$$\mathbf{h}_i = \text{LSTM}(g(\mathbf{x}_i), \{\mathbf{h}_j\}_{j \neq i})$$

**Attention LSTM**: Encode query with attention to support:

$$\mathbf{h}_q = \text{LSTM}(f(\mathbf{x}), \{\mathbf{h}_i\}_{i=1}^{|\mathcal{S}|})$$

### Training

**Episodic Training**: Sample episodes, minimize:

$$\mathcal{L} = -\mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{Q}}[\log p(y | \mathbf{x}, \mathcal{S})]$$

### Advantages

- **Flexible**: Can handle variable support set sizes
- **Attention**: Focuses on relevant support examples
- **Context**: Uses full support set context

### Comparison with Prototypical

| Aspect | Matching | Prototypical |
|--------|----------|--------------|
| Representation | Individual examples | Class prototypes |
| Attention | Yes | No |
| Complexity | Higher | Lower |
| Performance | Similar | Similar |

---

## Relation Networks

Relation networks learn a relation function to compare query-support pairs, providing more flexible similarity learning.

### Architecture

**Embedding**: $f_\phi(\mathbf{x})$ for both query and support

**Relation Module**: $g_\theta(\mathbf{x}_q, \mathbf{x}_s)$ computes relation score

**Concatenation**: Combine query and support embeddings:

$$\mathbf{r}_{qs} = g_\theta([f_\phi(\mathbf{x}_q), f_\phi(\mathbf{x}_s)])$$

### Relation Score

**Binary Classification**: Relation score indicates same class:

$$r(\mathbf{x}_q, \mathbf{x}_s) = \sigma(g_\theta([f_\phi(\mathbf{x}_q), f_\phi(\mathbf{x}_s)]))$$

where $\sigma$ is sigmoid.

### Prediction

**Sum over Support**: Aggregate relation scores:

$$p(y=k | \mathbf{x}_q) = \frac{\sum_{(\mathbf{x}_s, y_s) \in \mathcal{S}_k} r(\mathbf{x}_q, \mathbf{x}_s)}{\sum_{j=1}^{N} \sum_{(\mathbf{x}_s, y_s) \in \mathcal{S}_j} r(\mathbf{x}_q, \mathbf{x}_s)}$$

### Training

**Binary Cross-Entropy**:

$$\mathcal{L} = -\sum_{(\mathbf{x}_q, y_q) \in \mathcal{Q}} \sum_{(\mathbf{x}_s, y_s) \in \mathcal{S}} [y_q == y_s] \log r(\mathbf{x}_q, \mathbf{x}_s) + [y_q \neq y_s] \log(1-r(\mathbf{x}_q, \mathbf{x}_s))$$

### Advantages

- **Learned Similarity**: Learns optimal similarity function
- **Flexible**: Can capture complex relationships
- **Pairwise**: Considers query-support pairs explicitly

### Limitations

- **Computational Cost**: Requires comparing with all support examples
- **Memory**: Stores all support embeddings

---

## Siamese Networks

Siamese networks learn similarity functions by comparing pairs of examples, enabling few-shot learning through learned distance metrics.

### Architecture

**Twin Networks**: Two identical networks share weights:

$$f_\phi(\mathbf{x}_1), f_\phi(\mathbf{x}_2)$$

**Distance**: Compute distance in embedding space:

$$d(\mathbf{x}_1, \mathbf{x}_2) = \|f_\phi(\mathbf{x}_1) - f_\phi(\mathbf{x}_2)\|_2$$

### Training

**Contrastive Loss**: 

$$\mathcal{L} = (1-y) \cdot d^2 + y \cdot \max(0, m-d)^2$$

where:
- $y=1$ if same class, $y=0$ otherwise
- $m$ is margin

**Triplet Loss**:

$$\mathcal{L} = \max(0, d(\mathbf{x}_a, \mathbf{x}_p) - d(\mathbf{x}_a, \mathbf{x}_n) + m)$$

where $\mathbf{x}_a$ is anchor, $\mathbf{x}_p$ is positive, $\mathbf{x}_n$ is negative.

### Few-Shot Classification

**Nearest Neighbor**: Classify query by nearest support example:

$$\hat{y} = \arg\min_{(\mathbf{x}_s, y_s) \in \mathcal{S}} d(\mathbf{x}_q, \mathbf{x}_s)$$

### Advantages

- **Simple**: Easy to understand and implement
- **Efficient**: Fast inference
- **Pairwise**: Learns from pairs of examples

### Limitations

- **No Prototypes**: Doesn't aggregate class information
- **Sensitive**: Performance depends on support examples

---

## Metric Learning for Few-Shot

Metric learning aims to learn distance functions that are optimal for few-shot classification.

### Mahalanobis Distance

**Learned Metric**:

$$d_M(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i - \mathbf{x}_j)^T M (\mathbf{x}_i - \mathbf{x}_j)$$

where $M$ is positive semi-definite matrix.

**Learning**: Optimize $M$ to minimize classification error.

### Deep Metric Learning

**Embedding Function**: $f_\phi: \mathcal{X} \rightarrow \mathbb{R}^d$

**Distance**: $d(f_\phi(\mathbf{x}_i), f_\phi(\mathbf{x}_j))$

**Objective**: Minimize distance for same class, maximize for different classes.

### Contrastive Learning

**Positive Pairs**: Same class, minimize distance

**Negative Pairs**: Different classes, maximize distance (with margin)

**Loss**:

$$\mathcal{L} = \sum_{(i,j) \in \mathcal{P}} d^2(\mathbf{x}_i, \mathbf{x}_j) + \sum_{(i,j) \in \mathcal{N}} \max(0, m-d(\mathbf{x}_i, \mathbf{x}_j))^2$$

### Triplet Learning

**Triplets**: $(\mathbf{x}_a, \mathbf{x}_p, \mathbf{x}_n)$ where $y_a = y_p \neq y_n$

**Loss**:

$$\mathcal{L} = \max(0, d(\mathbf{x}_a, \mathbf{x}_p) - d(\mathbf{x}_a, \mathbf{x}_n) + m)$$

**Mining**: Select hard triplets for better learning.

### N-Pair Loss

**Multiple Negatives**: Compare anchor-positive against multiple negatives:

$$\mathcal{L} = -\log \frac{\exp(\mathbf{x}_a^T \mathbf{x}_p)}{\exp(\mathbf{x}_a^T \mathbf{x}_p) + \sum_{i=1}^{N-1} \exp(\mathbf{x}_a^T \mathbf{x}_{n_i})}$$

---

## Episode Training Protocol

Episode training is crucial for few-shot learning, simulating the few-shot scenario during training.

### Episode Construction

**Sample Task**: Randomly sample $N$ classes from training set

**Sample Support**: For each class, sample $K$ examples

**Sample Query**: Sample $M$ query examples per class

**Episode**: $(\mathcal{S}, \mathcal{Q})$ where $\mathcal{S}$ is support and $\mathcal{Q}$ is query

### Training Procedure

**For each episode**:

1. Sample $N$ classes: $\mathcal{C} = \{c_1, \ldots, c_N\}$
2. Sample support: $\mathcal{S} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{NK}$
3. Sample query: $\mathcal{Q} = \{(\mathbf{x}_j, y_j)\}_{j=1}^{NM}$
4. Forward pass: Compute predictions $p(y | \mathbf{x}, \mathcal{S})$ for $\mathbf{x} \in \mathcal{Q}$
5. Compute loss: $\mathcal{L} = -\sum_{(\mathbf{x}, y) \in \mathcal{Q}} \log p(y | \mathbf{x}, \mathcal{S})$
6. Backward pass: Update parameters

### Benefits

- **Matches Test**: Training matches test scenario
- **Generalization**: Encourages generalization to new classes
- **Efficient**: Can train on large datasets by sampling episodes

### Variations

**Fixed Episodes**: Use fixed set of episodes

**Dynamic Episodes**: Sample new episodes each iteration

**Curriculum**: Start with easy episodes, increase difficulty

---

## N-Way K-Shot Setup

The N-way K-shot setup standardizes evaluation of few-shot learning methods.

### Definitions

**N-way**: Number of classes in episode

**K-shot**: Number of support examples per class

**Common Settings**:
- **5-way 1-shot**: 5 classes, 1 example per class
- **5-way 5-shot**: 5 classes, 5 examples per class
- **20-way 1-shot**: 20 classes, 1 example per class

### Evaluation Protocol

**Meta-Training**: Train on base classes $\mathcal{C}_{\text{train}}$

**Meta-Testing**: Evaluate on novel classes $\mathcal{C}_{\text{test}}$ where $\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset$

**Episodes**: Sample multiple episodes from test classes

**Metrics**: Accuracy averaged over episodes

### Standard Datasets

**MiniImagenet**: 
- 100 classes, 600 examples per class
- 64 train, 16 validation, 20 test classes
- 84×84 RGB images

**Omniglot**:
- 1623 characters, 20 examples per character
- 1200 train, 300 test characters
- 28×28 grayscale images

**TieredImagenet**:
- 608 classes, hierarchical structure
- 351 train, 97 validation, 160 test classes

### Performance Benchmarks

**5-way 1-shot**:
- Random: 20%
- Prototypical: ~50%
- Matching: ~50%
- MAML: ~48%

**5-way 5-shot**:
- Random: 20%
- Prototypical: ~70%
- Matching: ~65%
- MAML: ~63%

---

## Advanced Few-Shot Methods

### TADAM: Task-Dependent Adaptive Metric

Learns task-specific metrics:

$$d_k(\mathbf{x}, \mathbf{c}_k) = \|f_\phi(\mathbf{x}) - \mathbf{c}_k\|_{M_k}$$

where $M_k$ is class-specific metric.

### Dynamic Few-Shot Learning

Adapts to task difficulty:

- Easy tasks: Use simple classifier
- Hard tasks: Use more complex model

### Meta-Learning with Memory

**Memory-Augmented Networks**: Store and retrieve examples:

$$p(y | \mathbf{x}) = \sum_{i=1}^{M} a(\mathbf{x}, \mathbf{m}_i) y_i$$

where $\mathbf{m}_i$ are memory slots.

### Cross-Domain Few-Shot

**Domain Shift**: Train and test on different domains

**Domain Adaptation**: Adapt embeddings across domains

### Self-Supervised Pre-Training

**Pre-training**: Learn representations on large unlabeled data

**Fine-tuning**: Adapt to few-shot tasks

**Benefits**: Better initialization, improved performance

### Transductive Few-Shot

**Inductive**: Classify each query independently

**Transductive**: Use all queries together:

$$p(\mathbf{y}_{\mathcal{Q}} | \mathbf{X}_{\mathcal{Q}}, \mathcal{S}) = \prod_{i=1}^{|\mathcal{Q}|} p(y_i | \mathbf{x}_i, \mathcal{S}, \mathbf{X}_{\mathcal{Q}})$$

---

## Key Takeaways

1. **Few-Shot Learning**: Learns from very few examples (1-5 per class) by leveraging prior knowledge, with N-way K-shot setup standardizing evaluation (N classes, K examples per class).

2. **Prototypical Networks**: Compute class prototypes $\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum f_\phi(\mathbf{x}_i)$ and classify by distance $d(\mathbf{x}, \mathbf{c}_k)$, providing simple and effective few-shot learning.

3. **Matching Networks**: Use attention to match queries to support examples $p(y|\mathbf{x}) = \sum a(\mathbf{x}, \mathbf{x}_i) y_i$, enabling flexible few-shot learning with full context.

4. **Relation Networks**: Learn relation function $r(\mathbf{x}_q, \mathbf{x}_s) = g_\theta([f_\phi(\mathbf{x}_q), f_\phi(\mathbf{x}_s)])$ to compare query-support pairs, providing learned similarity.

5. **Siamese Networks**: Learn similarity through twin networks with contrastive or triplet loss, enabling pairwise metric learning for few-shot tasks.

6. **Metric Learning**: Learns optimal distance functions (Mahalanobis, deep embeddings) through contrastive learning, triplet learning, or N-pair loss to improve few-shot performance.

7. **Episode Training**: Samples episodes with support and query sets during training to match test scenario, encouraging generalization to new classes.

8. **N-Way K-Shot**: Standard evaluation protocol (e.g., 5-way 1-shot, 5-way 5-shot) on datasets like MiniImagenet and Omniglot, with clear train/test class separation.

9. **Advanced Methods**: TADAM learns task-specific metrics, dynamic few-shot adapts to difficulty, memory-augmented networks store examples, cross-domain handles domain shift, self-supervised pre-training improves initialization.

10. **Challenges**: Limited data, overfitting, generalization to truly novel classes, and domain shift remain active research areas, with metric learning and meta-learning providing complementary approaches.
