# Text Classification and Sentiment Analysis

## Table of Contents

1. [Introduction](#introduction)
2. [Text Classification Problem](#text-classification-problem)
3. [Feature Engineering for Text](#feature-engineering-for-text)
4. [Naive Bayes for Text Classification](#naive-bayes-for-text-classification)
5. [Support Vector Machines for Text](#support-vector-machines-for-text)
6. [Sentiment Analysis](#sentiment-analysis)
7. [Document Classification](#document-classification)
8. [Evaluation and Metrics](#evaluation-and-metrics)
9. [Multi-Class and Multi-Label Classification](#multi-class-and-multi-label-classification)
10. [Key Takeaways](#key-takeaways)

## Introduction

Text classification assigns predefined categories to text documents, enabling automated organization, filtering, and analysis. Sentiment analysis, a specialized form of text classification, identifies emotional tone or opinion polarity.

Text classification applications:
- **Spam detection**: Classify emails as spam/ham
- **Topic categorization**: News articles, web pages
- **Sentiment analysis**: Product reviews, social media
- **Language identification**: Detect document language
- **Author attribution**: Identify document author

The fundamental challenge: Learn a function $f: \mathcal{D} \to \mathcal{C}$ mapping documents $d \in \mathcal{D}$ to classes $c \in \mathcal{C}$.

## Text Classification Problem

Text classification is a supervised learning problem requiring labeled training data.

### Problem Formulation

Given:
- **Training set**: $\{(d_1, y_1), \ldots, (d_n, y_n)\}$ where $d_i$ are documents and $y_i \in \mathcal{C}$ are labels
- **Document representation**: $\mathbf{x}_i = \phi(d_i)$ where $\phi$ maps documents to feature vectors
- **Classifier**: $f: \mathcal{X} \to \mathcal{C}$ learned from training data

### Classification Types

**Binary classification**: Two classes (e.g., spam/not spam)
**Multi-class classification**: Multiple mutually exclusive classes
**Multi-label classification**: Multiple non-exclusive labels per document

### Challenges

**High dimensionality**: Vocabulary size can be $10^5$ to $10^6$
**Sparsity**: Documents use small vocabulary subsets
**Class imbalance**: Some classes may be rare
**Domain adaptation**: Performance drops on new domains
**Interpretability**: Understanding why documents are classified

## Feature Engineering for Text

Feature engineering transforms raw text into numerical representations suitable for machine learning.

### Bag of Words Features

**Word counts**: Raw term frequencies
**Binary features**: Presence/absence indicators
**TF-IDF**: Term frequency-inverse document frequency weights

Document represented as:
$$\mathbf{x} = [x_1, x_2, \ldots, x_V]$$

where $x_i$ is the feature value for term $i$ and $V$ is vocabulary size.

### N-Gram Features

Beyond unigrams:
- **Bigrams**: Word pairs ("machine learning")
- **Trigrams**: Word triplets
- **Character n-grams**: Subword units

N-grams capture:
- **Phrases**: Multi-word expressions
- **Context**: Local word order
- **Morphology**: Word structure (character n-grams)

### Linguistic Features

**Part-of-speech tags**: Grammatical categories
**Named entities**: People, organizations, locations
**Syntactic features**: Parse tree features
**Semantic features**: WordNet synsets, semantic roles

### Statistical Features

**Document length**: Number of words, characters
**Term statistics**: Average word length, punctuation count
**Readability scores**: Flesch-Kincaid, SMOG index
**Lexical diversity**: Type-token ratio, vocabulary richness

### Feature Selection

Reduce dimensionality:

**Frequency filtering**: Remove very rare or very common terms
**Information gain**: Select terms with high information gain
**Chi-square**: Statistical significance testing
**Mutual information**: Measure term-class association

Feature selection improves:
- **Efficiency**: Faster training and prediction
- **Generalization**: Reduces overfitting
- **Interpretability**: Focuses on important terms

### Feature Normalization

Normalize features for algorithms sensitive to scale:

**L2 normalization**: Unit length vectors
**L1 normalization**: Sum to 1
**Z-score normalization**: Zero mean, unit variance
**Min-max normalization**: Scale to [0, 1]

## Naive Bayes for Text Classification

Naive Bayes is a probabilistic classifier that works well for text despite its independence assumption.

### Bayes' Theorem

Bayes' theorem for classification:

$$P(c | d) = \frac{P(d | c) P(c)}{P(d)} \propto P(d | c) P(c)$$

Choose class maximizing posterior probability:

$$\hat{c} = \arg\max_{c \in \mathcal{C}} P(c | d) = \arg\max_{c \in \mathcal{C}} P(d | c) P(c)$$

### Naive Independence Assumption

Naive Bayes assumes word independence:

$$P(d | c) = P(w_1, w_2, \ldots, w_n | c) = \prod_{i=1}^{n} P(w_i | c)$$

This assumption is clearly false (words are dependent) but works well in practice.

### Multinomial Naive Bayes

Multinomial Naive Bayes models word counts:

$$P(d | c) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_{i=1}^{V} P(w_i | c)^{x_i}$$

where $x_i$ is count of word $w_i$ in document $d$.

**Parameter estimation**:
$$P(w_i | c) = \frac{\text{count}(w_i, c) + \alpha}{\sum_{j=1}^{V} \text{count}(w_j, c) + \alpha V}$$

where $\alpha$ is smoothing parameter (Laplace smoothing when $\alpha=1$).

**Prior probability**:
$$P(c) = \frac{\text{count}(c)}{N}$$

where $N$ is total number of documents.

### Bernoulli Naive Bayes

Bernoulli Naive Bayes models word presence/absence:

$$P(d | c) = \prod_{i=1}^{V} P(w_i | c)^{x_i} (1 - P(w_i | c))^{1-x_i}$$

where $x_i \in \{0, 1\}$ indicates presence.

**Parameter estimation**:
$$P(w_i | c) = \frac{\text{documents in } c \text{ containing } w_i + \alpha}{\text{documents in } c + 2\alpha}$$

### Naive Bayes Advantages

**Simple**: Easy to implement and understand
**Fast**: Linear time complexity
**Interpretable**: Feature importance via $P(w_i | c)$
**Works well**: Despite independence assumption, performs competitively
**Probabilistic**: Provides probability estimates, not just labels

### Naive Bayes Limitations

**Independence assumption**: Words are clearly dependent
**Sparse data**: Poor estimates for rare words
**Feature engineering**: Requires careful preprocessing
**Class imbalance**: Sensitive to imbalanced classes

## Support Vector Machines for Text

Support Vector Machines (SVMs) find optimal separating hyperplanes, performing excellently for text classification.

### Linear SVM

Linear SVM finds hyperplane maximizing margin:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

**Optimization problem**:
$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{n} \xi_i$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

where $\xi_i$ are slack variables and $C$ controls regularization.

### Kernel SVMs

Kernel functions enable non-linear decision boundaries:

**Polynomial kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + 1)^d$
**RBF kernel**: $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2)$

For text, linear kernels often perform best due to high dimensionality.

### SVM for Text

**Advantages**:
- **High performance**: State-of-the-art for many text tasks
- **Sparse data**: Handles high-dimensional sparse vectors well
- **Margin maximization**: Good generalization
- **Kernel flexibility**: Can use non-linear kernels

**Disadvantages**:
- **Slow training**: Quadratic complexity in worst case
- **Parameter tuning**: $C$ and kernel parameters need tuning
- **Interpretability**: Less interpretable than probabilistic models
- **Memory**: Stores support vectors

### Multi-Class SVMs

**One-vs-rest**: Train $|\mathcal{C}|$ binary classifiers
**One-vs-one**: Train $\binom{|\mathcal{C}|}{2}$ binary classifiers
**Multi-class SVM**: Single optimization problem

## Sentiment Analysis

Sentiment analysis identifies emotional tone, opinion polarity, or attitude in text.

### Sentiment Classification

**Binary**: Positive vs negative
**Three-way**: Positive, negative, neutral
**Fine-grained**: 1-5 star ratings, emotion categories

### Lexicon-Based Approaches

Use sentiment lexicons (word lists with sentiment scores):

**AFINN**: Word scores from -5 to +5
**VADER**: Valence Aware Dictionary and sEntiment Reasoner
**SentiWordNet**: WordNet synsets with sentiment scores

**Scoring**: Aggregate word-level scores:
$$\text{sentiment}(d) = \sum_{w \in d} \text{score}(w)$$

### Challenges in Sentiment Analysis

**Negation**: "not good" vs "good"
**Sarcasm**: "Great, another meeting" (negative sentiment)
**Context**: "This phone is small" (positive for portability, negative for screen)
**Domain dependence**: "sick" means cool (positive) in some contexts
**Comparatives**: "better than X" requires comparison

### Feature Engineering for Sentiment

**Lexical features**: Sentiment word counts, ratios
**N-gram features**: Capture phrases ("not good", "very happy")
**Negation handling**: Mark words following negation
**Intensifiers**: "very", "extremely" modify sentiment
**Emoticons**: 😊, 😢 provide sentiment signals

### Aspect-Based Sentiment

Identify sentiment toward specific aspects:

**Aspect extraction**: Find aspects (e.g., "battery", "screen")
**Aspect sentiment**: Sentiment for each aspect
**Example**: "Great battery but poor screen" → battery: positive, screen: negative

## Document Classification

Document classification assigns topics or categories to entire documents.

### Topic Classification

Classify documents into predefined topics:
- **News categories**: Sports, politics, technology
- **Web page classification**: Product pages, blog posts
- **Email categorization**: Work, personal, spam

### Hierarchical Classification

Organize classes hierarchically:
- **Top level**: Broad categories
- **Subcategories**: Fine-grained classes

Enables:
- **Coarse-to-fine**: Predict at multiple levels
- **Transfer learning**: Use coarse labels for fine-grained
- **Efficiency**: Early stopping at coarse level

### Domain Adaptation

Adapt classifiers to new domains:

**Fine-tuning**: Retrain on target domain data
**Domain adaptation**: Use source domain to help target
**Multi-task learning**: Learn shared representations
**Transfer learning**: Pre-trained models adapted to new tasks

## Evaluation and Metrics

Evaluation measures classification performance.

### Accuracy

Fraction of correct predictions:

$$\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}$$

**Limitations**: Misleading for imbalanced classes.

### Precision, Recall, F1

**Precision**: Fraction of positive predictions that are correct

$$P = \frac{TP}{TP + FP}$$

**Recall**: Fraction of positives correctly identified

$$R = \frac{TP}{TP + FN}$$

**F1-score**: Harmonic mean

$$F_1 = \frac{2PR}{P + R}$$

### Confusion Matrix

Confusion matrix shows classification breakdown:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

Enables detailed error analysis.

### Per-Class Metrics

For multi-class, compute metrics per class:

**Macro-averaged**: Average across classes
**Micro-averaged**: Aggregate all predictions
**Weighted**: Weight by class frequency

### Cross-Validation

**K-fold cross-validation**: Split data into $k$ folds, train on $k-1$, test on 1
**Stratified**: Maintain class distribution in folds
**Leave-one-out**: Extreme case ($k=n$)

## Multi-Class and Multi-Label Classification

### Multi-Class Classification

Multiple mutually exclusive classes:

**One-vs-rest**: Binary classifiers for each class
**One-vs-one**: Pairwise classifiers
**Softmax**: Single model with softmax output

**Evaluation**: Accuracy, per-class precision/recall, confusion matrix

### Multi-Label Classification

Multiple non-exclusive labels per document:

**Binary relevance**: Independent binary classifiers
**Classifier chains**: Chain classifiers, use previous predictions
**Label powerset**: Treat label combinations as classes

**Evaluation**:
- **Hamming loss**: Fraction of incorrect labels
- **Subset accuracy**: Exact match fraction
- **F1-micro**: Micro-averaged F1 across labels
- **F1-macro**: Macro-averaged F1

### Hierarchical Multi-Label

Labels organized hierarchically:
- **Taxonomy**: Tree structure
- **DAG**: Directed acyclic graph

Enforces consistency: if child label predicted, parent must be too.

## Key Takeaways

1. **Text classification enables automated organization**: Assigning categories to documents enables filtering, routing, and analysis at scale.

2. **Feature engineering is crucial**: Bag of words, n-grams, and linguistic features transform text into numerical representations suitable for machine learning.

3. **Naive Bayes works despite independence assumption**: The naive independence assumption is clearly false but Naive Bayes performs competitively and provides interpretable, probabilistic predictions.

4. **SVMs excel for text classification**: Linear SVMs handle high-dimensional sparse text vectors effectively, achieving state-of-the-art performance for many tasks.

5. **Sentiment analysis requires special handling**: Negation, sarcasm, context, and domain dependence make sentiment analysis challenging, requiring specialized features and approaches.

6. **Evaluation requires multiple metrics**: Accuracy, precision, recall, and F1 capture different aspects of performance, especially important for imbalanced classes.

7. **Multi-label classification extends single-label**: Documents can have multiple labels, requiring different approaches and evaluation metrics than standard classification.

8. **Domain adaptation is essential**: Classifiers trained on one domain often perform poorly on others, requiring adaptation techniques for real-world deployment.
