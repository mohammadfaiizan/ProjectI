# Statistical Machine Translation

## Table of Contents

1. [Introduction](#introduction)
2. [Translation Models](#translation-models)
3. [IBM Models 1-5](#ibm-models-1-5)
4. [Word Alignment](#word-alignment)
5. [Phrase-Based Translation](#phrase-based-translation)
6. [Phrase Tables](#phrase-tables)
7. [Decoding Algorithms](#decoding-algorithms)
8. [BLEU Score and Evaluation](#bleu-score-and-evaluation)
9. [Limitations and Challenges](#limitations-and-challenges)
10. [Key Takeaways](#key-takeaways)

## Introduction

Statistical Machine Translation (SMT) models translation as a probabilistic process, learning translation patterns from parallel corpora. SMT dominated machine translation before neural approaches, providing the foundation for modern translation systems.

The fundamental SMT problem: given source sentence $\mathbf{f} = f_1, \ldots, f_m$ in language $F$, find target sentence $\mathbf{e} = e_1, \ldots, e_n$ in language $E$ that maximizes:

$$\hat{\mathbf{e}} = \arg\max_{\mathbf{e}} P(\mathbf{e} | \mathbf{f})$$

Using Bayes' rule:

$$P(\mathbf{e} | \mathbf{f}) = \frac{P(\mathbf{f} | \mathbf{e}) P(\mathbf{e})}{P(\mathbf{f})} \propto P(\mathbf{f} | \mathbf{e}) P(\mathbf{e})$$

This decomposition separates:
- **Translation model** $P(\mathbf{f} | \mathbf{e})$: How source words relate to target words
- **Language model** $P(\mathbf{e})$: Fluency of target sentence

## Translation Models

Translation models estimate how source words correspond to target words, handling word order differences and many-to-many alignments.

### Alignment

An alignment $a$ maps source positions to target positions:

$$a: \{1, \ldots, m\} \rightarrow \{0, 1, \ldots, n\}$$

where $a(j) = i$ means source word $f_j$ aligns to target word $e_i$, and $a(j) = 0$ means $f_j$ has no alignment (null alignment).

### Translation Model Formulation

The translation model with alignment:

$$P(\mathbf{f} | \mathbf{e}) = \sum_{\mathbf{a}} P(\mathbf{f}, \mathbf{a} | \mathbf{e})$$

where the sum is over all possible alignments $\mathbf{a}$.

For efficiency, often approximate with the best alignment:

$$P(\mathbf{f} | \mathbf{e}) \approx \max_{\mathbf{a}} P(\mathbf{f}, \mathbf{a} | \mathbf{e})$$

### Word Translation Probabilities

The core component is the word translation probability $t(f | e)$, the probability that target word $e$ translates to source word $f$.

These probabilities are learned from parallel corpora using the Expectation-Maximization (EM) algorithm.

## IBM Models 1-5

The IBM models progressively add complexity to handle translation phenomena.

### IBM Model 1

Model 1 makes strong simplifying assumptions:

**Uniform alignment**: All alignments equally likely
**Word independence**: Words translate independently
**No word order**: Alignment doesn't depend on position

$$P(\mathbf{f}, \mathbf{a} | \mathbf{e}) = \frac{\epsilon}{(l+1)^m} \prod_{j=1}^{m} t(f_j | e_{a(j)})$$

where $l$ is target length, $m$ is source length, and $\epsilon$ is a constant.

**Training**: EM algorithm estimates $t(f|e)$ from parallel sentences.

**Limitations**: 
- Ignores word order
- Uniform alignment unrealistic
- No fertility (one-to-many alignments)

### IBM Model 2

Model 2 adds alignment probabilities:

$$P(\mathbf{f}, \mathbf{a} | \mathbf{e}) = \epsilon \prod_{j=1}^{m} t(f_j | e_{a(j)}) a(a(j) | j, m, l)$$

where $a(i | j, m, l)$ is the probability that source position $j$ aligns to target position $i$ given sentence lengths.

**Alignment model**: Captures position preferences (e.g., first words often align)

**Training**: Jointly estimate translation and alignment probabilities via EM.

### IBM Model 3

Model 3 adds fertility: how many source words a target word produces.

**Fertility**: $\phi_i$ = number of source words aligned to target word $e_i$

$$P(\mathbf{f}, \mathbf{a} | \mathbf{e}) = \prod_{i=1}^{l} \phi_i! n(\phi_i | e_i) \prod_{j=1}^{m} t(f_j | e_{a(j)})$$

where $n(\phi | e)$ is the fertility probability.

**Features**:
- Handles one-to-many alignments
- Models null insertions
- More realistic than Models 1-2

### IBM Model 4

Model 4 adds relative position modeling for better word order handling.

**Distortion**: Models where aligned words appear relative to their positions:

$$d(i | \mathcal{A}_{i-1}(e_{i-1}), \mathcal{C}_{\text{fertility}})$$

where $\mathcal{A}_{i-1}$ is the set of source positions aligned to previous target words.

**Cept**: A target word and its aligned source words form a "cept".

Model 4 handles:
- Word order differences
- Phrase-like alignments
- Relative positioning

### IBM Model 5

Model 5 fixes a deficiency in Model 4: it ensures no position is used twice (one-to-one constraint at the position level).

**Placement model**: More sophisticated distortion that prevents position conflicts.

Model 5 is computationally expensive but provides the best alignments among IBM models.

### Model Comparison

| Model | Features | Complexity |
|-------|----------|------------|
| Model 1 | Uniform alignment, word translation | Low |
| Model 2 | + Position-based alignment | Medium |
| Model 3 | + Fertility | Medium-High |
| Model 4 | + Relative distortion | High |
| Model 5 | + Position constraints | Very High |

## Word Alignment

Word alignment identifies correspondences between source and target words, crucial for learning translation patterns.

### Alignment Types

**One-to-one**: Each word aligns to exactly one word
**One-to-many**: One word aligns to multiple words
**Many-to-one**: Multiple words align to one word
**Many-to-many**: Phrase alignments

### Symmetrization

IBM models are directional. Symmetrization combines alignments from both directions:

**Intersection**: Keep alignments present in both directions (high precision)
**Union**: Keep alignments in either direction (high recall)
**Grow-diag-final**: Heuristic that grows alignment from intersection

### Alignment Quality

Alignment quality measured by:
- **AER (Alignment Error Rate)**: Comparison to gold alignments
- **Translation quality**: Impact on final translation
- **Consistency**: Agreement between directions

Good alignments are essential for phrase extraction.

## Phrase-Based Translation

Phrase-based translation uses multi-word phrases instead of single words, better handling idiomatic expressions and local reordering.

### Phrase Extraction

Phrases extracted from word-aligned parallel sentences:

**Consistent phrase pair**: Phrase pair $(\bar{f}, \bar{e})$ where:
- All words in $\bar{f}$ align only to words in $\bar{e}$
- All words in $\bar{e}$ align only to words in $\bar{f}$
- Contains at least one alignment

**Extraction algorithm**: Enumerate all consistent phrase pairs up to maximum length.

### Phrase Translation Probability

Phrase translation probability estimated from counts:

$$\phi(\bar{f} | \bar{e}) = \frac{\text{count}(\bar{f}, \bar{e})}{\text{count}(\bar{e})}$$

Multiple features often used:
- **Phrase translation**: $\phi(\bar{f} | \bar{e})$ and $\phi(\bar{e} | \bar{f})$
- **Lexical weighting**: Word-level translation probabilities
- **Phrase penalty**: Prefer shorter/longer phrases

### Phrase Reordering

Phrase-based models handle reordering through:

**Distortion model**: Penalty for jumping positions
$$d(\text{start}_i - \text{end}_{i-1} - 1)$$

**Lexicalized reordering**: Learn reordering patterns from data

**Limit distortion**: Maximum jump distance to control search space

## Phrase Tables

Phrase tables store extracted phrase pairs with their translation probabilities and features.

### Table Structure

Each entry contains:
- **Source phrase**: $\bar{f}$
- **Target phrase**: $\bar{e}$
- **Translation probabilities**: $\phi(\bar{f} | \bar{e})$, $\phi(\bar{e} | \bar{f})$
- **Lexical weights**: Word-level scores
- **Additional features**: Phrase count, alignment information

### Phrase Table Filtering

Phrase tables can be huge. Filtering strategies:

**Count threshold**: Remove low-frequency phrases
**Length limit**: Maximum phrase length (typically 7-10 words)
**Probability threshold**: Remove low-probability translations
**Target vocabulary**: Only keep phrases with in-vocabulary words

### Storage and Lookup

Efficient storage crucial:
- **Hash tables**: Fast phrase lookup
- **Trie structures**: Prefix matching
- **Compression**: Reduce memory footprint
- **Disk-based**: For very large tables

## Decoding Algorithms

Decoding finds the best target sentence given source sentence and translation model.

### Search Problem

Decoding is a search problem over:
- **Phrase selection**: Which phrases to use
- **Phrase ordering**: How to order target phrases
- **Coverage**: Ensure all source words covered

### Beam Search

Beam search maintains top-$k$ hypotheses:

**Hypothesis**: Partial translation with coverage vector
**Scoring**: Combination of translation model, language model, distortion
**Pruning**: Keep only best $k$ hypotheses per coverage state

**Complexity**: $O(k \times |\text{phrases}| \times \text{length})$

### Stack Decoding

Stack decoding organizes hypotheses by number of source words covered:

**Stacks**: $S_0, S_1, \ldots, S_m$ where $S_i$ contains hypotheses covering $i$ words

**Algorithm**:
1. Start with empty hypothesis in $S_0$
2. For each stack, extend hypotheses with phrases
3. Prune each stack to top-$k$
4. Final answer in $S_m$

### Cube Pruning

Cube pruning efficiently explores phrase combinations:

**Lattice**: Organize phrase options
**Pruning**: Early elimination of poor combinations
**Efficiency**: Reduces search space significantly

## BLEU Score and Evaluation

BLEU (Bilingual Evaluation Understudy) is the standard automatic evaluation metric for machine translation.

### BLEU Formulation

BLEU measures n-gram precision:

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$: Precision of $n$-grams
- $w_n$: Weights (typically uniform: $w_n = 1/N$)
- $BP$: Brevity penalty

### Precision Calculation

N-gram precision counts matches:

$$p_n = \frac{\sum_{\text{n-grams} \in \text{candidate}} \text{Count}_{clip}(\text{n-gram})}{\sum_{\text{n-grams} \in \text{candidate}} \text{Count}(\text{n-gram})}$$

**Clipped count**: Maximum count in any reference, prevents over-counting.

### Brevity Penalty

Penalizes translations shorter than references:

$$BP = \begin{cases}
1 & \text{if } c > r \\
e^{1-r/c} & \text{if } c \leq r
\end{cases}$$

where $c$ is candidate length, $r$ is effective reference length.

### BLEU Properties

**Range**: 0 to 1 (often reported as percentage)
**Interpretation**: Higher is better, but absolute values depend on corpus
**Limitations**: 
- Doesn't measure fluency directly
- Requires multiple references for reliability
- May not correlate with human judgment for some phenomena

### Other Metrics

**METEOR**: Considers synonyms and stemming
**TER**: Translation Error Rate, edit distance based
**chrF**: Character-level F-score
**Human evaluation**: Most reliable but expensive

## Limitations and Challenges

SMT faces several fundamental challenges that limit its effectiveness.

### Long-Range Dependencies

SMT struggles with:
- Long-distance word order differences
- Discontinuous phrases
- Complex syntactic reordering

Phrase-based models have limited context window.

### Rare Words

**OOV problem**: Out-of-vocabulary words cannot be translated
**Low-frequency phrases**: Poor probability estimates
**Domain adaptation**: Performance drops on new domains

### Language Pair Specificity

SMT requires:
- Parallel corpora for each language pair
- Language-specific tuning
- Hand-crafted features for some languages

### Computational Complexity

**Decoding**: NP-hard search problem
**Training**: Expensive for large corpora
**Storage**: Large phrase tables

These limitations motivated the shift to neural machine translation.

## Key Takeaways

1. **SMT models translation probabilistically**: The noisy channel model decomposes translation into translation and language models, enabling data-driven learning.

2. **IBM models progressively add complexity**: From simple word translation (Model 1) to sophisticated alignment and distortion (Model 5), each model addresses limitations of previous ones.

3. **Word alignment is fundamental**: High-quality alignments enable phrase extraction and are crucial for learning translation patterns from parallel data.

4. **Phrase-based translation handles local reordering**: Using multi-word phrases better captures idiomatic expressions and local word order differences than word-based models.

5. **Decoding is a complex search problem**: Finding the best translation requires efficient search algorithms (beam search, stack decoding) over large hypothesis spaces.

6. **BLEU provides automatic evaluation**: While imperfect, BLEU enables rapid iteration and comparison of translation systems without expensive human evaluation.

7. **SMT has fundamental limitations**: Long-range dependencies, rare words, and language pair specificity limit SMT performance, motivating neural approaches.

8. **SMT provides foundation for modern MT**: Concepts from SMT (alignment, phrase extraction, decoding) inform neural machine translation systems.
