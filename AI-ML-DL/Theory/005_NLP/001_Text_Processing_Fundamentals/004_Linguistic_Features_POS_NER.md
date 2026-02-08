# Linguistic Features: POS Tagging and Named Entity Recognition

## Table of Contents

1. [Introduction](#introduction)
2. [Part-of-Speech Tagging](#part-of-speech-tagging)
3. [Hidden Markov Models for POS](#hidden-markov-models-for-pos)
4. [Conditional Random Fields for POS](#conditional-random-fields-for-pos)
5. [Named Entity Recognition](#named-entity-recognition)
6. [Dependency Parsing](#dependency-parsing)
7. [Constituency Parsing](#constituency-parsing)
8. [Semantic Role Labeling](#semantic-role-labeling)
9. [Feature Engineering for Linguistic Tasks](#feature-engineering-for-linguistic-tasks)
10. [Key Takeaways](#key-takeaways)

## Introduction

Linguistic annotation adds structured information to text, enabling deeper understanding of language structure and meaning. These annotations serve as features for downstream NLP tasks and provide interpretable linguistic analysis.

Linguistic features capture different levels of language structure:
- **Morphological**: Word forms and parts of speech
- **Syntactic**: Phrase structure and dependencies
- **Semantic**: Meaning, roles, and entities

Part-of-speech (POS) tagging and named entity recognition (NER) are fundamental annotation tasks that provide essential features for many NLP applications.

## Part-of-Speech Tagging

Part-of-speech tagging assigns grammatical categories (noun, verb, adjective, etc.) to each word in a sentence.

### POS Tag Sets

Different tag sets provide varying granularity:

**Penn Treebank**: 45 tags (NN, VB, JJ, etc.)
**Universal Dependencies**: 17 universal tags
**Brown Corpus**: 87 tags (more fine-grained)

Tag categories include:
- **Open class**: Nouns, verbs, adjectives, adverbs (content words)
- **Closed class**: Prepositions, determiners, conjunctions (function words)

### POS Tagging Challenges

**Ambiguity**: Words can have multiple POS tags ("bank" as noun or verb)
**Unknown words**: OOV words require handling
**Context dependence**: Correct tag depends on context
**Tag set differences**: Different tag sets for different languages

### Applications of POS Tagging

POS tags are used for:
- **Syntactic parsing**: Guide phrase structure analysis
- **Information extraction**: Identify noun phrases
- **Machine translation**: Preserve grammatical structure
- **Speech recognition**: Constrain word hypotheses
- **Text-to-speech**: Determine pronunciation

### Evaluation Metrics

POS tagging is evaluated by:

**Accuracy**: Percentage of correctly tagged words
$$\text{Accuracy} = \frac{\text{correct tags}}{\text{total words}}$$

**Per-tag metrics**: Precision, recall, F1 for each tag
**Sentence accuracy**: Percentage of fully correct sentences

State-of-the-art taggers achieve 97%+ accuracy on English.

## Hidden Markov Models for POS

Hidden Markov Models (HMMs) are probabilistic sequence models that model POS tagging as a sequence labeling problem.

### HMM Formulation

An HMM consists of:
- **States**: POS tags $T = \{t_1, t_2, \ldots, t_n\}$
- **Observations**: Words $W = \{w_1, w_2, \ldots, w_m\}$
- **Transition probabilities**: $P(t_i | t_{i-1})$
- **Emission probabilities**: $P(w_j | t_i)$

### Markov Assumption

HMMs assume:
- **State transitions**: Current tag depends only on previous tag
- **Observations**: Word depends only on current tag

$$P(t_1, \ldots, t_n, w_1, \ldots, w_n) = P(t_1) \prod_{i=2}^{n} P(t_i | t_{i-1}) \prod_{i=1}^{n} P(w_i | t_i)$$

### Parameter Estimation

HMM parameters are estimated from labeled data:

**Transition probabilities**:
$$P(t_i | t_j) = \frac{C(t_j, t_i)}{C(t_j)}$$

**Emission probabilities**:
$$P(w | t) = \frac{C(t, w)}{C(t)}$$

**Initial probabilities**:
$$P(t_1) = \frac{C(\text{<s>}, t_1)}{C(\text{<s>})}$$

### Viterbi Algorithm

Viterbi finds the most likely tag sequence:

$$\hat{\mathbf{t}} = \arg\max_{\mathbf{t}} P(\mathbf{t} | \mathbf{w}) = \arg\max_{\mathbf{t}} P(\mathbf{t}, \mathbf{w})$$

Dynamic programming computes:

$$v_t(i) = \max_{t_{i-1}} v_{t_{i-1}}(i-1) \times P(t_i | t_{i-1}) \times P(w_i | t_i)$$

where $v_t(i)$ is the probability of best path ending at tag $t$ for word $i$.

### Smoothing for HMMs

Smoothing addresses sparse data:

**Add-one smoothing**: Add 1 to all counts
**Backoff**: Use unigram probabilities when bigrams missing
**Interpolation**: Combine multiple orders

Smoothing is crucial for handling rare tag sequences and unknown words.

### Unknown Word Handling

Unknown words pose challenges:

**Uniform distribution**: Assign equal probability to all tags
**Morphological features**: Use word endings, capitalization
**Default tag**: Assign most common tag (often noun)
**Subword modeling**: Use character-level features

## Conditional Random Fields for POS

Conditional Random Fields (CRFs) are discriminative sequence models that directly model $P(\mathbf{t} | \mathbf{w})$.

### CRF Formulation

CRFs model the conditional probability:

$$P(\mathbf{t} | \mathbf{w}) = \frac{1}{Z(\mathbf{w})} \exp\left(\sum_{i=1}^{n} \sum_{k=1}^{K} \lambda_k f_k(t_{i-1}, t_i, \mathbf{w}, i)\right)$$

where:
- $f_k$ are feature functions
- $\lambda_k$ are learned weights
- $Z(\mathbf{w})$ is the partition function (normalization)

### Feature Functions

Feature functions capture patterns:

**Transition features**: $f(t_{i-1}, t_i)$ - tag bigrams
**Emission features**: $f(t_i, w_i)$ - word-tag pairs
**Context features**: $f(t_i, w_{i-1}, w_{i+1})$ - surrounding words
**Morphological features**: $f(t_i, \text{prefix}(w_i))$ - word prefixes/suffixes

### CRF vs HMM

**CRF advantages**:
- Discriminative: Directly models $P(t|w)$
- Flexible features: Can use arbitrary features
- No independence assumptions: Can model long-range dependencies
- Better performance: Typically outperforms HMMs

**HMM advantages**:
- Generative: Can generate text
- Interpretable: Clear probabilistic interpretation
- Faster training: Simpler optimization

### CRF Training

CRF training maximizes conditional likelihood:

$$\mathcal{L} = \sum_{i=1}^{N} \log P(\mathbf{t}^{(i)} | \mathbf{w}^{(i)})$$

Optimization uses:
- **Gradient descent**: Compute gradients w.r.t. $\lambda_k$
- **L-BFGS**: Quasi-Newton method
- **Stochastic gradient**: For large datasets

### Inference in CRFs

Inference finds the most likely tag sequence:

**Viterbi decoding**: Same algorithm as HMMs, adapted for CRF features
**Forward-backward**: For marginal probabilities $P(t_i | \mathbf{w})$

## Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies named entities: people, organizations, locations, dates, etc.

### Entity Types

Common entity types:
- **PER**: Person names
- **ORG**: Organizations
- **LOC**: Locations
- **MISC**: Miscellaneous (events, products, etc.)
- **DATE, TIME, MONEY**: Temporal and numerical entities

### NER Challenges

**Boundary detection**: Where entities start/end
**Type classification**: Which type of entity
**Ambiguity**: "Apple" as company or fruit
**Nested entities**: Entities within entities
**Cross-lingual**: Different conventions across languages

### BIO Tagging Scheme

BIO scheme tags each word:
- **B-X**: Beginning of entity type X
- **I-X**: Inside entity type X
- **O**: Outside any entity

Example: "Barack Obama visited France"
- Barack: B-PER
- Obama: I-PER
- visited: O
- France: B-LOC

### NER Approaches

**Rule-based**: Hand-crafted patterns and gazetteers
**Feature-based**: CRFs with hand-engineered features
**Neural**: BiLSTM-CRF, transformer-based models
**Transfer learning**: Pre-trained models fine-tuned on NER

### Feature Engineering for NER

Effective features include:
- **Word features**: Current word, capitalization, prefixes/suffixes
- **Context features**: Surrounding words
- **Linguistic features**: POS tags, chunk tags
- **Gazetteer features**: Membership in lists (cities, names)
- **Orthographic features**: Contains digits, punctuation patterns

## Dependency Parsing

Dependency parsing identifies syntactic relationships between words as head-dependent pairs.

### Dependency Structure

A dependency tree consists of:
- **Nodes**: Words in the sentence
- **Edges**: Directed arcs from head to dependent
- **Labels**: Relationship types (subject, object, modifier)

Example: "The cat sat on the mat"
- sat → cat (nsubj: subject)
- sat → on (prep: preposition)
- on → mat (pobj: object of preposition)

### Dependency Parsing Algorithms

**Graph-based**: Find maximum spanning tree
**Transition-based**: Greedy sequence of actions
**Neural**: End-to-end learned parsers

### Transition-Based Parsing

Uses a stack and buffer:

**Actions**:
- **SHIFT**: Move word from buffer to stack
- **LEFT-ARC**: Create arc from top stack word to second
- **RIGHT-ARC**: Create arc from second stack word to top
- **REDUCE**: Remove word from stack

Greedy classifier predicts actions, building the tree incrementally.

### Dependency Labels

Universal Dependencies provides standard labels:
- **nsubj**: Nominal subject
- **dobj**: Direct object
- **amod**: Adjectival modifier
- **det**: Determiner
- **prep**: Prepositional modifier

## Constituency Parsing

Constituency parsing identifies hierarchical phrase structure using context-free grammars.

### Phrase Structure Trees

Constituency trees group words into phrases:

```
(S (NP (DT The) (NN cat))
   (VP (VBD sat)
       (PP (IN on)
           (NP (DT the) (NN mat)))))
```

### Context-Free Grammars

CFGs define phrase structure:

$S \rightarrow NP\ VP$
$NP \rightarrow DT\ NN$
$VP \rightarrow VBD\ PP$

### CYK Algorithm

CYK (Cocke-Younger-Kasami) parses using dynamic programming:

Fills table $T[i,j]$ with non-terminals spanning words $i$ to $j$.

**Base case**: $T[i,i+1]$ from grammar rules
**Recursion**: $T[i,j]$ from combinations of $T[i,k]$ and $T[k,j]$

### Probabilistic CFGs

PCFGs add probabilities to rules:

$$P(S \rightarrow NP\ VP) = 0.8$$

Enables finding most likely parse:

$$\hat{T} = \arg\max_T P(T | \mathbf{w})$$

## Semantic Role Labeling

Semantic Role Labeling (SRL) identifies who did what to whom, when, where, etc.

### Semantic Roles

Common roles (PropBank/FrameNet):
- **ARG0**: Agent (doer)
- **ARG1**: Patient/Theme (undergoer)
- **ARG2**: Instrument, beneficiary
- **ARGM-TMP**: Temporal modifier
- **ARGM-LOC**: Location modifier

### SRL Process

1. **Predicate identification**: Find verbs/predicates
2. **Argument identification**: Find argument spans
3. **Role classification**: Assign roles to arguments

### SRL Approaches

**Feature-based**: CRFs with syntactic features
**Neural**: BiLSTM with attention
**End-to-end**: Joint predicate and argument identification

## Feature Engineering for Linguistic Tasks

Effective feature engineering is crucial for traditional NLP approaches.

### Word Features

**Surface form**: The word itself
**Lowercase**: Normalized case
**Prefixes/suffixes**: Character n-grams
**Capitalization**: All caps, title case, etc.
**Digit patterns**: Contains numbers, patterns

### Context Features

**Window**: Surrounding words in $k$-word window
**Position**: Distance from sentence start/end
**Sentence position**: First, middle, last sentence

### Linguistic Features

**POS tags**: From POS tagger
**Chunk tags**: Noun phrases, verb phrases
**Dependency features**: Head word, dependency label
**Morphological**: Lemma, morphological tags

### Gazetteer Features

**Named entity lists**: Cities, countries, names
**Domain lexicons**: Technical terms, abbreviations
**WordNet**: Hypernyms, synonyms

### Feature Combinations

Combining features captures interactions:
- **Word + POS**: "bank/NN" vs "bank/VB"
- **Word + context**: "New York" (location indicator)
- **Capitalization + position**: Capitalized word at sentence start

## Key Takeaways

1. **POS tagging provides grammatical structure**: Identifying word categories enables syntactic analysis and serves as features for downstream tasks.

2. **HMMs model POS as sequence labeling**: The Markov assumption enables efficient Viterbi decoding, though smoothing is crucial for handling sparsity.

3. **CRFs outperform HMMs**: Discriminative modeling and flexible features make CRFs the preferred approach for sequence labeling tasks.

4. **NER identifies entities**: Recognizing named entities is essential for information extraction, question answering, and many applications.

5. **Dependency parsing captures relationships**: Head-dependent relationships provide rich syntactic structure useful for semantic analysis.

6. **Constituency parsing reveals hierarchy**: Phrase structure trees show how words group into larger units, important for understanding sentence structure.

7. **SRL extracts semantic roles**: Identifying who did what enables deeper semantic understanding beyond syntax.

8. **Feature engineering matters**: Carefully designed features significantly impact performance of traditional NLP models, though neural approaches can learn features automatically.
