# Word Sense Disambiguation

## Table of Contents

1. [Introduction](#introduction)
2. [Word Sense Disambiguation Problem](#word-sense-disambiguation-problem)
3. [Knowledge-Based Approaches](#knowledge-based-approaches)
4. [Supervised Approaches](#supervised-approaches)
5. [Unsupervised Approaches](#unsupervised-approaches)
6. [The Lesk Algorithm](#the-lesk-algorithm)
7. [WordNet and Sense Inventories](#wordnet-and-sense-inventories)
8. [Sense Embeddings](#sense-embeddings)
9. [Evaluation and Datasets](#evaluation-and-datasets)
10. [Key Takeaways](#key-takeaways)

## Introduction

Word Sense Disambiguation (WSD) identifies which sense of a word is intended in a given context. Words often have multiple meanings (polysemy), and WSD resolves this ambiguity to enable accurate language understanding.

WSD is fundamental for:
- **Machine translation**: Correct sense selection affects translation quality
- **Information retrieval**: Matching queries to documents requires sense understanding
- **Question answering**: Understanding question meaning
- **Text understanding**: Semantic analysis depends on correct sense identification

The challenge: Given word $w$ in context $c$, determine sense $s \in S(w)$ where $S(w)$ is the set of possible senses for $w$.

## Word Sense Disambiguation Problem

WSD requires sense inventories, context representation, and disambiguation algorithms.

### Sense Inventories

**Fine-grained**: Many senses per word (WordNet: 3-10+ senses)
**Coarse-grained**: Fewer senses (domain labels, topic categories)

**Sense granularity trade-off**:
- Fine-grained: More precise but harder to distinguish
- Coarse-grained: Easier but less informative

### WSD Approaches

**Knowledge-based**: Use external resources (dictionaries, thesauri)
**Supervised**: Learn from sense-annotated data
**Unsupervised**: Discover senses from unlabeled data
**Hybrid**: Combine multiple approaches

### Context Representation

Context features for WSD:
- **Surrounding words**: Local context window
- **Syntactic features**: POS tags, parse structure
- **Semantic features**: Topic, domain
- **Document features**: Document topic, genre

## Knowledge-Based Approaches

Knowledge-based WSD uses external lexical resources without training data.

### Dictionary-Based Methods

Use sense definitions from dictionaries:

**Overlap method**: Count word overlap between context and sense definitions
**Similarity method**: Measure semantic similarity between context and definitions

**Advantages**:
- No training data needed
- Interpretable
- Works for any language with dictionary

**Disadvantages**:
- Dictionary coverage limitations
- Definition quality varies
- May not capture usage patterns

### Thesaurus-Based Methods

Use semantic relationships from thesauri:

**Hypernym paths**: Distance in taxonomy
**Synonym sets**: Overlap with related words
**Semantic fields**: Domain associations

### Knowledge Sources

**WordNet**: English lexical database with synsets
**BabelNet**: Multilingual semantic network
**FrameNet**: Frame-semantic analysis
**Wikipedia**: Encyclopedic knowledge

## Supervised Approaches

Supervised WSD learns classifiers from sense-annotated examples.

### Problem Formulation

Given training examples $\{(c_1, s_1), \ldots, (c_n, s_n)\}$ where $c_i$ is context and $s_i$ is sense label, learn classifier:

$$f: \mathcal{C} \to \mathcal{S}$$

### Feature Engineering

**Local context**: Words in window around target
**Syntactic context**: POS tags, dependency relations
**Collocations**: Fixed phrases containing target
**Morphological**: Word form, lemma

### Classification Algorithms

**Naive Bayes**: Probabilistic classifier
$$P(s | c) \propto P(s) \prod_{w \in c} P(w | s)$$

**SVM**: Maximum margin classifier
**Decision trees**: Interpretable rule-based
**Neural networks**: Learned representations

### Context Representation

**Bag of words**: Simple but effective
**N-grams**: Capture phrases
**Syntactic features**: Parse tree features
**Embeddings**: Distributed representations

## Unsupervised Approaches

Unsupervised WSD discovers senses from unlabeled data.

### Clustering-Based Methods

**Context clustering**: Cluster word contexts, each cluster = sense

**Steps**:
1. Extract contexts for target word
2. Represent contexts as vectors
3. Cluster contexts
4. Assign sense labels to clusters

**Challenges**:
- Determining number of senses
- Interpreting clusters
- Evaluation without gold labels

### Topic Modeling

Use topic models to discover senses:

**LDA for WSD**: Each topic corresponds to a sense
**Context as documents**: Word contexts as documents
**Topic assignment**: Assign sense based on topic

### Distributional Methods

**Distributional hypothesis**: Words with similar distributions have similar meanings

**Context vectors**: Represent word by its contexts
**Clustering**: Group similar context vectors
**Sense identification**: Clusters represent senses

## The Lesk Algorithm

The Lesk algorithm is a classic knowledge-based WSD method using dictionary definitions.

### Basic Lesk Algorithm

For each sense $s$ of target word $w$:
1. Get definition $D(s)$
2. Count word overlap between context $c$ and $D(s)$
3. Choose sense with maximum overlap

**Score**:
$$\text{score}(s) = |\text{words}(c) \cap \text{words}(D(s))|$$

### Simplified Lesk

Simplified version counts exact word matches:

```python
def lesk(context, word, senses):
    best_sense = None
    max_overlap = 0
    context_words = set(context.split())
    
    for sense in senses:
        definition_words = set(get_definition(sense).split())
        overlap = len(context_words & definition_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense
```

### Extended Lesk

Extended Lesk includes:
- **Related words**: Synonyms, hypernyms
- **Weighted overlap**: Weight important words
- **Stemming**: Match word stems

**Score**:
$$\text{score}(s) = \sum_{w \in c \cap D(s)} \text{weight}(w)$$

### Limitations

**Sparse overlap**: Definitions may not overlap with context
**Short definitions**: Limited information
**Polysemy in definitions**: Definitions may contain ambiguous words

## WordNet and Sense Inventories

WordNet is the primary sense inventory for English WSD.

### WordNet Structure

**Synsets**: Sets of synonymous words (senses)
**Relations**: Hypernymy, hyponymy, meronymy, etc.
**Hierarchy**: Tree structure for nouns and verbs

**Example**: "bank" has synsets:
- Financial institution
- River edge
- Storage container
- etc.

### Sense Identification

**Sense keys**: Unique identifiers (e.g., "bank%1:17:00::")
**Sense numbers**: Ordered by frequency
**Glosses**: Definitions and examples

### Using WordNet for WSD

**Definition overlap**: Lesk algorithm
**Path similarity**: Distance in WordNet hierarchy
**Information content**: Probability-based similarity
**Extended glosses**: Include related synsets

### Multilingual WordNets

**BabelNet**: Multilingual semantic network
**EuroWordNet**: European languages
**Open Multilingual Wordnet**: Many languages

## Sense Embeddings

Sense embeddings represent word senses as dense vectors, enabling similarity computation.

### Word Sense Embeddings

Learn separate embeddings for each sense:

$$\mathbf{e}_{w,s} \in \mathbb{R}^d$$

where $w$ is word and $s$ is sense.

**Training**: From sense-annotated corpora or automatic sense induction.

### Contextualized Embeddings

Modern approach: Context determines sense implicitly:

**ELMo**: Contextual word representations
**BERT**: Bidirectional encoder representations
**Context vectors**: Embeddings depend on context

### Sense Induction

Discover senses automatically:

**Clustering**: Cluster word contexts
**Topic modeling**: Topics as senses
**Neural methods**: Learn sense representations

### Using Sense Embeddings

**Similarity**: Compare sense embeddings to context
**Classification**: Train classifier on embeddings
**Retrieval**: Find similar senses

## Evaluation and Datasets

WSD evaluation requires sense-annotated datasets and appropriate metrics.

### Evaluation Metrics

**Accuracy**: Fraction of correctly disambiguated words
**Precision/Recall**: Per-sense metrics
**F1-score**: Harmonic mean

**Macro-averaged**: Average across words
**Micro-averaged**: Overall accuracy

### Standard Datasets

**Senseval/SemEval**: Standard WSD evaluation campaigns
- **All-words**: Disambiguate all content words
- **Lexical sample**: Specific target words

**Datasets**:
- SemCor: Manually annotated corpus
- OntoNotes: Multi-layer annotations
- MASC: Manually annotated subcorpus

### Evaluation Challenges

**Sense granularity**: Fine vs coarse-grained evaluation
**Inter-annotator agreement**: Human disagreement on senses
**Domain adaptation**: Performance on new domains
**Coverage**: Handling words/senses not in inventory

### Baselines

**Most frequent sense**: Always choose most common sense
**Random**: Random sense selection
**First sense**: Choose first sense in inventory

Strong baselines are important for comparison.

## Key Takeaways

1. **WSD resolves lexical ambiguity**: Identifying correct word senses is crucial for accurate language understanding and downstream NLP tasks.

2. **Knowledge-based methods use external resources**: Dictionary and thesaurus-based approaches work without training data but depend on resource quality and coverage.

3. **Supervised methods learn from annotations**: Training on sense-labeled examples enables accurate disambiguation but requires expensive annotation.

4. **Unsupervised methods discover senses**: Clustering and topic modeling can identify senses automatically but face evaluation and interpretation challenges.

5. **Lesk algorithm is a classic approach**: Overlap-based methods provide simple, interpretable baselines for knowledge-based WSD.

6. **WordNet is the standard sense inventory**: Rich semantic relationships in WordNet enable various WSD approaches, though coverage and granularity are limitations.

7. **Sense embeddings enable similarity-based WSD**: Dense vector representations of senses allow geometric similarity computation and integration with neural methods.

8. **Evaluation requires careful design**: Sense granularity, inter-annotator agreement, and domain adaptation significantly affect WSD evaluation and system comparison.
