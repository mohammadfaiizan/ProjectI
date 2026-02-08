# Topic Modeling and Latent Dirichlet Allocation

## Table of Contents

1. [Introduction](#introduction)
2. [Topic Modeling Overview](#topic-modeling-overview)
3. [Latent Semantic Analysis](#latent-semantic-analysis)
4. [Latent Dirichlet Allocation](#latent-dirichlet-allocation)
5. [Generative Process](#generative-process)
6. [Inference Algorithms](#inference-algorithms)
7. [Gibbs Sampling for LDA](#gibbs-sampling-for-lda)
8. [Topic Coherence and Evaluation](#topic-coherence-and-evaluation)
9. [Model Selection and Extensions](#model-selection-and-extensions)
10. [Key Takeaways](#key-takeaways)

## Introduction

Topic modeling discovers latent thematic structure in document collections, identifying topics as distributions over words and documents as mixtures of topics. Latent Dirichlet Allocation (LDA) is the most widely used topic model, providing a principled probabilistic framework for uncovering semantic patterns.

Topic modeling addresses:
- **Document organization**: Group similar documents
- **Dimensionality reduction**: Represent documents in low-dimensional topic space
- **Exploratory analysis**: Discover themes in large corpora
- **Feature extraction**: Topics as features for downstream tasks

LDA models documents as probabilistic mixtures of topics, where topics are distributions over words, enabling interpretable and flexible topic discovery.

## Topic Modeling Overview

Topic modeling assumes documents exhibit multiple topics simultaneously, with each topic characterized by a distribution over words.

### Topic Definition

A **topic** is a probability distribution over vocabulary:

$$\boldsymbol{\phi}_k = [P(w_1 | z=k), P(w_2 | z=k), \ldots, P(w_V | z=k)]$$

where $\sum_{i=1}^{V} P(w_i | z=k) = 1$ and $V$ is vocabulary size.

Topics are discovered from data, not predefined, making topic modeling an unsupervised learning problem.

### Document-Topic Distribution

Each document has a distribution over topics:

$$\boldsymbol{\theta}_d = [P(z=1 | d), P(z=2 | d), \ldots, P(z=K | d)]$$

where $\sum_{k=1}^{K} P(z=k | d) = 1$ and $K$ is the number of topics.

Documents are mixtures of topics, enabling multi-thematic representation.

### Applications

Topic modeling enables:
- **Exploratory analysis**: Discover themes in document collections
- **Dimensionality reduction**: Represent documents in $K$-dimensional topic space
- **Document similarity**: Compare documents via topic distributions
- **Feature extraction**: Topics as features for classification/clustering
- **Trend analysis**: Track topic prevalence over time

## Latent Semantic Analysis

Latent Semantic Analysis (LSA) is a precursor to LDA that uses matrix factorization to discover latent semantic structure.

### LSA Formulation

LSA applies Singular Value Decomposition (SVD) to the document-term matrix:

$$\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times V}$: Document-term matrix (TF-IDF)
- $\mathbf{U} \in \mathbb{R}^{n \times k}$: Document-topic matrix
- $\boldsymbol{\Sigma} \in \mathbb{R}^{k \times k}$: Singular values
- $\mathbf{V} \in \mathbb{R}^{V \times k}$: Term-topic matrix

### LSA Interpretation

**Document representation**: Rows of $\mathbf{U}$ represent documents in topic space
**Topic representation**: Columns of $\mathbf{V}$ represent topics as word vectors
**Dimensionality**: $k$ is the number of latent dimensions (topics)

### LSA Limitations

**No probabilistic interpretation**: SVD doesn't provide probability distributions
**Negative values**: Topics can have negative weights (hard to interpret)
**No generative model**: Cannot generate new documents
**Orthogonality constraint**: Topics are orthogonal (may not match reality)

These limitations motivated probabilistic topic models like LDA.

## Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is a generative probabilistic model that addresses LSA's limitations.

### LDA Assumptions

LDA makes key assumptions:
- **Bag of words**: Word order doesn't matter
- **Fixed vocabulary**: Vocabulary is known and fixed
- **Fixed topics**: Number of topics $K$ is fixed
- **Dirichlet priors**: Topic and word distributions use Dirichlet priors

### LDA Plate Notation

LDA can be represented as:

```
For each topic k:
  φ_k ~ Dirichlet(β)

For each document d:
  θ_d ~ Dirichlet(α)
  For each word w_{d,i}:
    z_{d,i} ~ Multinomial(θ_d)
    w_{d,i} ~ Multinomial(φ_{z_{d,i}})
```

where:
- $\alpha$: Hyperparameter for document-topic distributions
- $\beta$: Hyperparameter for topic-word distributions
- $z_{d,i}$: Topic assignment for word $i$ in document $d$

### LDA Parameters

**Global parameters**:
- $\boldsymbol{\Phi} = \{\boldsymbol{\phi}_1, \ldots, \boldsymbol{\phi}_K\}$: Topic-word distributions
- $\boldsymbol{\Theta} = \{\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_D\}$: Document-topic distributions

**Local variables**:
- $\mathbf{Z} = \{z_{d,i}\}$: Topic assignments for each word

**Hyperparameters**:
- $\alpha$: Prior for document-topic distributions
- $\beta$: Prior for topic-word distributions

## Generative Process

LDA defines a generative process for creating documents.

### Step-by-Step Generation

For each document $d$:

1. **Sample topic distribution**: $\boldsymbol{\theta}_d \sim \text{Dirichlet}(\alpha)$
2. **For each word** $w_{d,i}$:
   - Sample topic: $z_{d,i} \sim \text{Multinomial}(\boldsymbol{\theta}_d)$
   - Sample word: $w_{d,i} \sim \text{Multinomial}(\boldsymbol{\phi}_{z_{d,i}})$

### Dirichlet Distribution

Dirichlet distribution is a distribution over probability vectors:

$$\text{Dirichlet}(\mathbf{x} | \boldsymbol{\alpha}) = \frac{\Gamma(\sum_{i=1}^{K} \alpha_i)}{\prod_{i=1}^{K} \Gamma(\alpha_i)} \prod_{i=1}^{K} x_i^{\alpha_i - 1}$$

where $\mathbf{x}$ is a $K$-dimensional probability vector and $\boldsymbol{\alpha}$ are concentration parameters.

**Properties**:
- **Conjugate prior**: Dirichlet is conjugate to multinomial
- **Sparsity control**: Smaller $\alpha$ values encourage sparser distributions
- **Symmetric**: $\alpha = [\alpha, \ldots, \alpha]$ treats all components equally

### Joint Probability

The joint probability of words and topic assignments:

$$P(\mathbf{W}, \mathbf{Z} | \boldsymbol{\Theta}, \boldsymbol{\Phi}, \alpha, \beta) = \prod_{d=1}^{D} P(\boldsymbol{\theta}_d | \alpha) \prod_{i=1}^{N_d} P(z_{d,i} | \boldsymbol{\theta}_d) P(w_{d,i} | \boldsymbol{\phi}_{z_{d,i}})$$

where $N_d$ is the number of words in document $d$.

## Inference Algorithms

Inference in LDA estimates posterior distributions over latent variables given observed documents.

### Inference Problem

Given observed words $\mathbf{W}$, infer:
- Topic-word distributions: $P(\boldsymbol{\Phi} | \mathbf{W})$
- Document-topic distributions: $P(\boldsymbol{\Theta} | \mathbf{W})$
- Topic assignments: $P(\mathbf{Z} | \mathbf{W})$

The posterior is intractable, requiring approximate inference.

### Variational Inference

Variational inference approximates the posterior with a simpler distribution:

**Variational distribution**: $q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi} | \boldsymbol{\lambda})$

**Optimization**: Minimize KL divergence:

$$\text{KL}(q || p) = \int q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi}) \log \frac{q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi})}{p(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi} | \mathbf{W})} d\mathbf{Z} d\boldsymbol{\Theta} d\boldsymbol{\Phi}$$

**Mean-field assumption**: Factorize variational distribution:

$$q(\mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi}) = q(\mathbf{Z}) q(\boldsymbol{\Theta}) q(\boldsymbol{\Phi})$$

### Expectation-Maximization

EM algorithm alternates:
- **E-step**: Update variational parameters given model parameters
- **M-step**: Update model parameters given variational parameters

Converges to local optimum.

## Gibbs Sampling for LDA

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method for LDA inference.

### Collapsed Gibbs Sampling

Integrate out $\boldsymbol{\Theta}$ and $\boldsymbol{\Phi}$, sample only $\mathbf{Z}$:

$$P(z_{d,i} = k | \mathbf{Z}_{-(d,i)}, \mathbf{W}, \alpha, \beta) \propto \frac{n_{d,k}^{-(d,i)} + \alpha_k}{\sum_{k'=1}^{K} (n_{d,k'}^{-(d,i)} + \alpha_{k'})} \times \frac{n_{k,w_{d,i}}^{-(d,i)} + \beta_{w_{d,i}}}{\sum_{v=1}^{V} (n_{k,v}^{-(d,i)} + \beta_v)}$$

where:
- $n_{d,k}^{-(d,i)}$: Count of topic $k$ in document $d$ excluding word $i$
- $n_{k,w}^{-(d,i)}$: Count of word $w$ in topic $k$ excluding word $i$ in document $d$

### Sampling Procedure

1. **Initialize**: Randomly assign topics to words
2. **Iterate**: For each word, sample new topic from conditional distribution
3. **Burn-in**: Discard initial samples
4. **Collect samples**: Average over samples to estimate distributions

### Parameter Estimation

After sampling, estimate parameters:

**Topic-word distributions**:
$$\hat{\phi}_{k,w} = \frac{n_{k,w} + \beta_w}{\sum_{v=1}^{V} (n_{k,v} + \beta_v)}$$

**Document-topic distributions**:
$$\hat{\theta}_{d,k} = \frac{n_{d,k} + \alpha_k}{\sum_{k'=1}^{K} (n_{d,k'} + \alpha_{k'})}$$

### Advantages of Gibbs Sampling

**Simple**: Easy to implement
**Exact**: Asymptotically samples from true posterior
**Flexible**: Can incorporate additional structure
**Interpretable**: Clear probabilistic interpretation

## Topic Coherence and Evaluation

Evaluating topic models is challenging due to unsupervised nature.

### Perplexity

Perplexity measures predictive performance:

$$\text{Perplexity}(\mathbf{W}_{\text{test}}) = \exp\left(-\frac{\sum_{d=1}^{D} \log P(\mathbf{w}_d)}{\sum_{d=1}^{D} N_d}\right)$$

Lower perplexity indicates better fit, but may not correlate with topic quality.

### Topic Coherence

Topic coherence measures semantic consistency of top words:

**UCI coherence**: Pointwise Mutual Information (PMI) based:

$$\text{Coherence}(k) = \sum_{i=2}^{M} \sum_{j=1}^{i-1} \log \frac{P(w_i, w_j) + \epsilon}{P(w_i) P(w_j)}$$

where $w_1, \ldots, w_M$ are top-$M$ words in topic $k$.

**UMass coherence**: Document co-occurrence based:

$$\text{Coherence}(k) = \sum_{i=2}^{M} \sum_{j=1}^{i-1} \log \frac{D(w_i, w_j) + \epsilon}{D(w_j)}$$

where $D(w_i, w_j)$ is documents containing both words.

### Human Evaluation

**Topic interpretability**: Human judges rate topic quality
**Word intrusion**: Detect intruder words in topic lists
**Topic labeling**: Assign labels to discovered topics

Human evaluation is gold standard but expensive.

### Application-Based Evaluation

Evaluate topics via downstream tasks:
- **Document classification**: Topics as features
- **Information retrieval**: Topic-based retrieval
- **Document clustering**: Compare to known categories

## Model Selection and Extensions

### Choosing Number of Topics

**Cross-validation**: Hold out documents, maximize likelihood
**Perplexity**: Minimize perplexity on held-out data
**Topic coherence**: Maximize average coherence
**Domain knowledge**: Use prior knowledge about corpus

### Hyperparameter Tuning

**$\alpha$ (document-topic prior)**:
- Smaller $\alpha$: Documents focus on fewer topics
- Larger $\alpha$: Documents use more topics uniformly

**$\beta$ (topic-word prior)**:
- Smaller $\beta$: Topics focus on fewer words
- Larger $\beta$: Topics use words more uniformly

Typical values: $\alpha = 50/K$, $\beta = 0.01$

### LDA Extensions

**Correlated Topic Model (CTM)**: Models topic correlations
**Dynamic Topic Model (DTM)**: Topics evolve over time
**Author-Topic Model**: Topics associated with authors
**Supervised LDA**: Incorporate document labels
**Hierarchical LDA**: Hierarchical topic structure

### Scalable LDA

For large corpora:
- **Online LDA**: Process documents in batches
- **Distributed LDA**: Parallelize across machines
- **Sparse LDA**: Exploit sparsity for efficiency
- **LightLDA**: Fast sampling algorithm

## Key Takeaways

1. **Topic modeling discovers latent structure**: LDA uncovers thematic patterns in document collections without supervision, enabling exploratory analysis and dimensionality reduction.

2. **LDA is a generative probabilistic model**: The generative process provides interpretable topic-word and document-topic distributions, addressing limitations of matrix factorization approaches like LSA.

3. **Dirichlet priors control sparsity**: Hyperparameters $\alpha$ and $\beta$ control how concentrated topic and word distributions are, affecting model behavior and interpretability.

4. **Inference requires approximation**: The posterior distribution is intractable, requiring variational inference or MCMC methods like Gibbs sampling for parameter estimation.

5. **Gibbs sampling is practical**: Collapsed Gibbs sampling integrates out parameters and samples only topic assignments, providing an efficient and interpretable inference algorithm.

6. **Evaluation is multifaceted**: Perplexity, topic coherence, human evaluation, and application-based metrics capture different aspects of topic model quality.

7. **Model selection matters**: Choosing the number of topics and hyperparameters significantly affects discovered topics, requiring careful tuning and validation.

8. **LDA has many extensions**: Correlated topics, temporal evolution, supervision, and hierarchical structure extend LDA to diverse applications and data types.
