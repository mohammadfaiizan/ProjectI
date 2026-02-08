# Language Models and N-Grams

## Table of Contents

1. [Introduction](#introduction)
2. [Probabilistic Language Models](#probabilistic-language-models)
3. [N-Gram Models](#n-gram-models)
4. [Markov Assumption](#markov-assumption)
5. [Smoothing Techniques](#smoothing-techniques)
6. [Perplexity and Evaluation](#perplexity-and-evaluation)
7. [Backoff and Interpolation](#backoff-and-interpolation)
8. [Practical Considerations](#practical-considerations)
9. [Extensions and Limitations](#extensions-and-limitations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Language models assign probabilities to sequences of words, enabling prediction of likely continuations and evaluation of text fluency. N-gram models, despite their simplicity, provide foundational understanding of language modeling and remain relevant for many applications.

A language model estimates the probability distribution over word sequences:

$$P(w_1, w_2, \ldots, w_n)$$

This probability can be used for:
- **Text generation**: Sample likely word sequences
- **Speech recognition**: Score candidate transcriptions
- **Machine translation**: Evaluate translation quality
- **Spell checking**: Identify unlikely word sequences

N-gram models approximate this distribution using limited context, making them computationally tractable while capturing local dependencies in language.

## Probabilistic Language Models

A probabilistic language model defines a probability distribution over strings in a language. The fundamental challenge is estimating these probabilities from finite training data.

### Chain Rule of Probability

The chain rule decomposes the joint probability:

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

This exact formulation requires conditioning on all previous words, which is computationally and statistically infeasible for long sequences.

### Maximum Likelihood Estimation

Given a training corpus, maximum likelihood estimation (MLE) estimates probabilities by counting:

$$P_{MLE}(w_i | w_1, \ldots, w_{i-1}) = \frac{C(w_1, \ldots, w_{i-1}, w_i)}{C(w_1, \ldots, w_{i-1})}$$

where $C(\cdot)$ denotes the count in the training corpus.

MLE suffers from the sparse data problem: most word sequences never appear in training data, leading to zero probabilities.

### Vocabulary and Unknown Words

Language models must handle out-of-vocabulary (OOV) words. Common approaches:

**Closed vocabulary**: Assume all words are known (unrealistic)
**Open vocabulary**: Include an `<UNK>` token for unknown words
**Subword modeling**: Model subword units instead of words

The vocabulary size $V$ significantly impacts model complexity and sparsity.

## N-Gram Models

N-gram models approximate the full history with only the previous $n-1$ words, dramatically reducing the number of parameters.

### N-Gram Definition

An n-gram is a sequence of $n$ consecutive words. Common n-gram orders:

- **Unigram** ($n=1$): Single words, no context
- **Bigram** ($n=2$): Word pairs
- **Trigram** ($n=3$): Word triplets
- **4-gram, 5-gram**: Higher-order models

### N-Gram Probability Estimation

For an n-gram model, the probability is:

$$P(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \frac{C(w_{i-n+1}, \ldots, w_{i-1}, w_i)}{C(w_{i-n+1}, \ldots, w_{i-1})}$$

The bigram case:

$$P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

### N-Gram Counts

Building an n-gram model requires counting all n-grams in the training corpus:

**Unigram counts**: $C(w)$ for each word $w$
**Bigram counts**: $C(w_{i-1}, w_i)$ for each word pair
**Higher-order counts**: Extend to longer sequences

The number of possible n-grams grows exponentially: $V^n$ for vocabulary size $V$ and order $n$.

### Start and End Tokens

N-gram models use special tokens:

**Start tokens**: `<s>` or `<BOS>` (beginning of sentence) mark sentence starts
**End tokens**: `</s>` or `<EOS>` (end of sentence) mark sentence ends

For a bigram model:
$$P(w_1, w_2, \ldots, w_n) = P(w_1 | <s>) \prod_{i=2}^{n} P(w_i | w_{i-1}) P(</s> | w_n)$$

## Markov Assumption

The Markov assumption states that the future depends only on the recent past, not the entire history.

### Markov Property

For an n-gram model, the Markov assumption is:

$$P(w_i | w_1, \ldots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \ldots, w_{i-1})$$

This is a $k$-th order Markov process where $k = n-1$.

### Independence Assumptions

The Markov assumption introduces independence:

- Words beyond the n-gram window are conditionally independent
- Long-range dependencies are ignored
- Local patterns are captured effectively

### Validity of the Assumption

The Markov assumption is approximately valid because:
- Local word order strongly predicts next words
- Long-range dependencies are less frequent
- Computational benefits outweigh accuracy loss for many applications

However, it fails for:
- Long-distance agreement (subject-verb agreement across clauses)
- Discourse-level coherence
- Semantic dependencies spanning sentences

## Smoothing Techniques

Smoothing addresses the zero probability problem by redistributing probability mass from seen to unseen events.

### Add-One (Laplace) Smoothing

Add-one smoothing adds 1 to all counts:

$$P_{Laplace}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

where $V$ is the vocabulary size.

**Problems**:
- Too much probability mass goes to unseen events
- Doesn't account for count reliability
- Performs poorly in practice

### Add-K Smoothing

Generalization of add-one:

$$P_{Add-k}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$$

$k$ is a hyperparameter, typically chosen via held-out validation.

### Good-Turing Smoothing

Good-Turing estimates the probability of unseen events using the frequency of once-seen events:

$$P_{GT}(w_i | w_{i-1}) = \frac{C^*(w_{i-1}, w_i)}{C(w_{i-1})}$$

where $C^*$ is the adjusted count:

$$C^*(w_{i-1}, w_i) = \frac{(C(w_{i-1}, w_i) + 1) N_{C(w_{i-1}, w_i) + 1}}{N_{C(w_{i-1}, w_i)}}$$

$N_r$ is the number of n-grams with count $r$.

Good-Turing is theoretically principled but requires careful handling of high-count n-grams.

### Kneser-Ney Smoothing

Kneser-Ney is a sophisticated smoothing method that uses absolute discounting and continuation probabilities.

**Absolute discounting**: Subtract a constant $d$ from non-zero counts:

$$P_{KN}(w_i | w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P_{continuation}(w_i)$$

**Continuation probability**: Probability that $w_i$ follows any word:

$$P_{continuation}(w_i) = \frac{|\{w : C(w, w_i) > 0\}|}{|\{(w, w') : C(w, w') > 0\}|}$$

**Interpolation weight**:
$$\lambda(w_{i-1}) = \frac{d |\{w : C(w_{i-1}, w) > 0\}|}{C(w_{i-1})}$$

Kneser-Ney typically uses $d = 0.75$ and performs excellently in practice.

### Modified Kneser-Ney

Modified Kneser-Ney uses different discount values $d_1, d_2, d_3$ for counts of 1, 2, and 3+:

$$d_1 = 1 - 2Y \frac{N_2}{N_1}$$
$$d_2 = 2 - 3Y \frac{N_3}{N_2}$$
$$d_{3+} = 3 - 4Y \frac{N_4}{N_3}$$

where $Y = \frac{N_1}{N_1 + 2N_2}$.

This refinement improves performance slightly over standard Kneser-Ney.

## Perplexity and Evaluation

Perplexity measures how well a language model predicts a test corpus.

### Perplexity Definition

Perplexity is the exponentiated average negative log-likelihood:

$$PP(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i | w_1, \ldots, w_{i-1})}}$$

In log space:

$$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, \ldots, w_{i-1})\right)$$

### Interpretation

Perplexity can be interpreted as:
- **Effective vocabulary size**: The number of equally likely choices the model faces
- **Uncertainty measure**: Lower perplexity means more confident predictions
- **Cross-entropy**: Perplexity = $2^{H(W)}$ where $H(W)$ is cross-entropy

### Perplexity for N-Grams

For a bigram model:

$$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{i-1})\right)$$

Perplexity is typically computed on a held-out test set, not the training set.

### Typical Perplexity Values

- **Random word selection**: $V$ (vocabulary size)
- **Unigram model**: ~100-1000 depending on corpus
- **Bigram model**: ~50-200
- **Trigram model**: ~20-100
- **Neural language models**: ~10-50

Lower is better, but perplexity should be interpreted relative to the task and corpus.

## Backoff and Interpolation

When higher-order n-grams are unavailable, backoff and interpolation combine information from lower-order models.

### Backoff

Backoff uses higher-order n-grams when available, otherwise falls back to lower-order:

$$P_{backoff}(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \begin{cases}
P^*(w_i | w_{i-n+1}, \ldots, w_{i-1}) & \text{if } C(w_{i-n+1}, \ldots, w_i) > 0 \\
\alpha(w_{i-n+1}, \ldots, w_{i-1}) P_{backoff}(w_i | w_{i-n+2}, \ldots, w_{i-1}) & \text{otherwise}
\end{cases}$$

$\alpha$ is a backoff weight ensuring probabilities sum to 1.

### Interpolation

Interpolation always combines all orders:

$$P_{interp}(w_i | w_{i-n+1}, \ldots, w_{i-1}) = \lambda_1 P(w_i | w_{i-n+1}, \ldots, w_{i-1}) + \lambda_2 P(w_i | w_{i-n+2}, \ldots, w_{i-1}) + \ldots + \lambda_n P(w_i)$$

where $\sum \lambda_i = 1$.

### Linear Interpolation

Linear interpolation uses fixed weights, often learned via EM algorithm or held-out validation:

$$P_{linear}(w_i | w_{i-1}) = \lambda_1 P(w_i | w_{i-1}) + \lambda_2 P(w_i)$$

### Weight Estimation

Weights can be:
- **Uniform**: $\lambda_i = 1/n$ for all $i$
- **Learned**: Optimize on held-out data
- **Count-based**: Weight by n-gram counts

## Practical Considerations

### Storage and Efficiency

N-gram models require efficient storage:

**Hash tables**: Fast lookup for n-gram counts
**Trie structures**: Memory-efficient for prefix queries
**Bloom filters**: Approximate membership testing
**Quantization**: Reduce precision of counts/probabilities

### Memory Requirements

Storage grows with:
- Vocabulary size $V$
- N-gram order $n$
- Corpus size

A 5-gram model with $V=50,000$ has $50,000^5 \approx 3 \times 10^{24}$ possible n-grams, though most are never seen.

### Pruning

Pruning reduces model size:

**Count cutoff**: Remove n-grams with count below threshold
**Entropy pruning**: Remove n-grams contributing little to perplexity
**Weighted difference**: Remove n-grams with small probability difference from backoff

### Handling Unknown Words

Strategies for OOV words:

**Unknown token**: Replace rare words with `<UNK>`
**Minimum count threshold**: Words below threshold become `<UNK>`
**Open vocabulary**: Use subword units or character-level modeling

## Extensions and Limitations

### Class-Based N-Grams

Class-based models group words into classes:

$$P(w_i | w_{i-1}) = P(c_i | c_{i-1}) P(w_i | c_i)$$

where $c_i$ is the class of word $w_i$. This reduces sparsity but loses word-specific information.

### Caching and Adaptation

**Cache models**: Boost probability of recently seen words
**Topic adaptation**: Adjust probabilities based on document topic
**Domain adaptation**: Adapt to target domain using in-domain data

### Limitations of N-Grams

N-gram models have fundamental limitations:

**Sparsity**: Most n-grams never appear in training
**Context window**: Limited to $n-1$ words
**Independence**: Cannot capture long-range dependencies
**Generalization**: Poor generalization to new domains

These limitations motivated the development of neural language models.

## Key Takeaways

1. **N-grams approximate language**: The Markov assumption enables tractable language modeling by limiting context, trading long-range dependencies for computational feasibility.

2. **Smoothing is essential**: Zero probabilities from sparse data require smoothing. Kneser-Ney smoothing provides state-of-the-art performance for n-gram models.

3. **Perplexity measures model quality**: Lower perplexity indicates better predictions, but values must be interpreted relative to the corpus and task.

4. **Backoff and interpolation handle sparsity**: Combining multiple n-gram orders improves robustness when higher-order n-grams are missing.

5. **Storage efficiency matters**: N-gram models can be large. Pruning and efficient data structures are crucial for practical deployment.

6. **Unknown words require special handling**: OOV words are inevitable. Strategies include unknown tokens, subword modeling, and minimum count thresholds.

7. **N-grams have fundamental limitations**: Fixed context windows and sparsity problems limit n-gram models, motivating neural approaches that can learn distributed representations.

8. **N-grams remain relevant**: Despite limitations, n-gram models are fast, interpretable, and effective for many applications, providing a baseline for more sophisticated methods.
