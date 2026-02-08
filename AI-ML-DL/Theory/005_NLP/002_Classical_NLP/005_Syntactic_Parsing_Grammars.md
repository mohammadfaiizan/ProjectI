# Syntactic Parsing and Grammars

## Table of Contents

1. [Introduction](#introduction)
2. [Context-Free Grammars](#context-free-grammars)
3. [CYK Algorithm](#cyk-algorithm)
4. [Probabilistic Context-Free Grammars](#probabilistic-context-free-grammars)
5. [Dependency Parsing](#dependency-parsing)
6. [Transition-Based Dependency Parsing](#transition-based-dependency-parsing)
7. [Graph-Based Dependency Parsing](#graph-based-dependency-parsing)
8. [Parsing Evaluation](#parsing-evaluation)
9. [Grammar Formalisms](#grammar-formalisms)
10. [Key Takeaways](#key-takeaways)

## Introduction

Syntactic parsing identifies the grammatical structure of sentences, revealing how words group into phrases and how phrases relate to each other. Parsing enables deeper language understanding and provides features for downstream NLP tasks.

Parsing approaches:
- **Constituency parsing**: Identifies hierarchical phrase structure
- **Dependency parsing**: Identifies head-dependent relationships
- **Hybrid approaches**: Combine both representations

Parsing applications:
- **Grammar checking**: Identify ungrammatical sentences
- **Information extraction**: Extract structured information
- **Machine translation**: Preserve syntactic structure
- **Question answering**: Understand sentence structure
- **Semantic analysis**: Foundation for semantic parsing

## Context-Free Grammars

Context-Free Grammars (CFGs) define phrase structure through rewrite rules.

### CFG Definition

A CFG is a 4-tuple $G = (N, \Sigma, R, S)$:
- **$N$**: Non-terminal symbols (phrases: NP, VP, S)
- **$\Sigma$**: Terminal symbols (words)
- **$R$**: Production rules $A \rightarrow \beta$ where $A \in N$, $\beta \in (N \cup \Sigma)^*$
- **$S$**: Start symbol (sentence)

### Production Rules

Rules specify how phrases decompose:

$S \rightarrow NP\ VP$
$NP \rightarrow DT\ NN$
$VP \rightarrow VBD\ NP$
$DT \rightarrow \text{the}$
$NN \rightarrow \text{cat}$

### Parse Trees

Parse trees represent sentence structure:

```
        S
       / \
      NP  VP
     / \  / \
    DT NN VBD NP
    |  |  |  / \
   the cat sat / \
             DT  NN
              |  |
             the mat
```

### Grammar Ambiguity

Sentences can have multiple valid parses:

**Structural ambiguity**: "I saw the man with binoculars"
- [I saw [the man] [with binoculars]] (binoculars = instrument)
- [I saw [the man with binoculars]] (binoculars = modifier)

**Attachment ambiguity**: Prepositional phrase attachment

### Grammar Coverage

**Hand-crafted grammars**: Linguists write rules (comprehensive but expensive)
**Treebank grammars**: Extract rules from annotated trees (data-driven)
**Lexicalized grammars**: Include word-specific information

## CYK Algorithm

The CYK (Cocke-Younger-Kasami) algorithm parses sentences using dynamic programming.

### CYK Assumptions

CYK requires:
- **CNF form**: Grammar in Chomsky Normal Form
- **CNF rules**: $A \rightarrow BC$ or $A \rightarrow w$ (binary or terminal)

Any CFG can be converted to CNF.

### CYK Table

Fill table $T[i,j]$ with non-terminals spanning words $i$ to $j$:

**Base case**: $T[i,i+1]$ contains non-terminals that generate word $i$:
$$T[i,i+1] = \{A : A \rightarrow w_i \in R\}$$

**Recursion**: $T[i,j]$ from combinations:
$$T[i,j] = \bigcup_{k=i+1}^{j-1} \{A : A \rightarrow BC \in R, B \in T[i,k], C \in T[k,j]\}$$

### CYK Algorithm

```
for length = 2 to n:
    for i = 1 to n - length + 1:
        j = i + length - 1
        for k = i to j - 1:
            for each rule A -> B C:
                if B in T[i,k] and C in T[k,j]:
                    add A to T[i,j]
```

**Complexity**: $O(n^3 |R|)$ where $n$ is sentence length, $|R|$ is number of rules.

### Recovering Parse Trees

After filling table, recover parse tree:
- Start from $S \in T[1,n]$
- Trace back through table entries
- Build tree bottom-up

## Probabilistic Context-Free Grammars

Probabilistic CFGs (PCFGs) add probabilities to grammar rules, enabling disambiguation and learning from data.

### PCFG Definition

PCFG assigns probabilities to rules:

$$P(A \rightarrow \beta | A)$$

with constraint:
$$\sum_{\beta} P(A \rightarrow \beta | A) = 1$$

### Parse Tree Probability

Probability of parse tree $T$:

$$P(T) = \prod_{r \in T} P(r)$$

where $r$ are rules used in tree $T$.

### Most Likely Parse

Find parse maximizing probability:

$$\hat{T} = \arg\max_T P(T | \mathbf{w}) = \arg\max_T P(T, \mathbf{w}) = \arg\max_T P(T)$$

since $P(\mathbf{w} | T) = 1$ for valid parses.

### Inside-Outside Algorithm

Inside-Outside algorithm estimates rule probabilities from unlabeled trees:

**Inside probability**: $\alpha(i,j,A)$ = probability $A$ generates words $i$ to $j$

**Outside probability**: $\beta(i,j,A)$ = probability of generating sentence with $A$ spanning $i$ to $j$

**EM algorithm**: Alternates E-step (compute probabilities) and M-step (update rule probabilities)

### Lexicalized PCFGs

Lexicalized PCFGs include head words:

$VP(\text{saw}) \rightarrow VBD(\text{saw})\ NP(\text{man})$

**Head propagation**: Head word percolates up tree
**Lexical dependencies**: Capture word-specific preferences

## Dependency Parsing

Dependency parsing identifies head-dependent relationships between words.

### Dependency Structure

Dependency tree:
- **Nodes**: Words in sentence
- **Edges**: Directed arcs from head to dependent
- **Labels**: Relationship types (nsubj, dobj, etc.)

Example: "The cat sat on the mat"
```
sat (root)
├─ nsubj → cat
│  └─ det → The
└─ prep → on
   └─ pobj → mat
      └─ det → the
```

### Dependency Properties

**Single head**: Each word has exactly one head (except root)
**Connected**: Tree spans all words
**Acyclic**: No cycles
**Projective**: No crossing arcs (for some languages)

### Projectivity

**Projective**: Arc $(i,j)$ doesn't cross other arcs
**Non-projective**: Allows crossing arcs (needed for some languages)

Projective trees can be parsed with simpler algorithms.

## Transition-Based Dependency Parsing

Transition-based parsing uses a sequence of actions to build dependency trees.

### Parser State

State consists of:
- **Stack**: Partially processed words
- **Buffer**: Remaining input words
- **Arcs**: Dependency arcs built so far

### Transition Actions

**SHIFT**: Move word from buffer to stack
**LEFT-ARC(l)**: Create arc from top stack word to second, label $l$
**RIGHT-ARC(l)**: Create arc from second stack word to top, label $l$
**REDUCE**: Remove word from stack (all dependents processed)

### Arc-Standard Algorithm

**LEFT-ARC**: If top stack word is head of second:
```
[..., w2, w1] | [buffer] → [..., w2] | [buffer]
Arc: w2 → w1
```

**RIGHT-ARC**: If second stack word is head of top:
```
[..., w2, w1] | [buffer] → [..., w1] | [buffer]
Arc: w1 → w2
```

### Arc-Eager Algorithm

**LEFT-ARC**: Create left arc, don't reduce
**RIGHT-ARC**: Create right arc, reduce dependent
**SHIFT**: Always available
**REDUCE**: Remove word with all arcs

Arc-eager processes left dependents before right.

### Greedy Classifier

Greedy classifier predicts next action:

$$a^* = \arg\max_a P(a | \text{state})$$

**Features**:
- Stack top words
- Buffer front words
- Existing arcs
- Part-of-speech tags

**Training**: Learn from gold parse trees (oracle actions)

## Graph-Based Dependency Parsing

Graph-based parsing finds maximum spanning tree over word pairs.

### Scoring Function

Score dependency tree:

$$\text{score}(T) = \sum_{(h,d) \in T} s(h,d)$$

where $s(h,d)$ is score of arc from head $h$ to dependent $d$.

### Maximum Spanning Tree

Find tree maximizing score:

$$\hat{T} = \arg\max_T \sum_{(h,d) \in T} s(h,d)$$

**Chu-Liu/Edmonds algorithm**: Finds MST in directed graphs
**Complexity**: $O(n^2)$ for projective, $O(n^2 \log n)$ for non-projective

### Arc Scoring

Score arcs using features:

$$s(h,d) = \mathbf{w}^T \boldsymbol{\phi}(h,d,\mathbf{w})$$

**Features**:
- Word forms: $w_h$, $w_d$
- POS tags: $t_h$, $t_d$
- Distance: $|h-d|$
- Context: Surrounding words
- Morphological: Word features

### Training

Learn weights $\mathbf{w}$:

**Structured perceptron**: Update on mistakes
**MIRA**: Margin-infused relaxed algorithm
**Global linear models**: Optimize global objective

## Parsing Evaluation

Parsing evaluation measures how well predicted parses match gold standard.

### Constituency Parsing Metrics

**Labeled precision**: Fraction of predicted constituents that are correct (label and span)
**Labeled recall**: Fraction of gold constituents that are predicted
**F1**: Harmonic mean of precision and recall

**Parseval metrics**: Standard evaluation for constituency parsing

### Dependency Parsing Metrics

**Unlabeled Attachment Score (UAS)**: Fraction of words with correct head
**Labeled Attachment Score (LAS)**: Fraction of words with correct head and label

**Root accuracy**: Fraction of sentences with correct root

### Evaluation Challenges

**Annotation differences**: Different treebanks may have different conventions
**Partial credit**: Some errors worse than others
**Domain mismatch**: Performance drops on new domains

## Grammar Formalisms

Various grammar formalisms extend CFGs for natural language.

### Tree-Adjoining Grammar (TAG)

TAG uses tree operations:
- **Substitution**: Replace leaf with tree
- **Adjunction**: Insert tree into another

More expressive than CFG, handles long-distance dependencies.

### Combinatory Categorial Grammar (CCG)

CCG uses categories and combinators:
- **Categories**: $S$, $NP$, $S\backslash NP$ (functions)
- **Combinators**: Application, composition

Provides semantic interpretation alongside syntax.

### Head-Driven Phrase Structure Grammar (HPSG)

HPSG uses feature structures:
- **Phrase structure**: Head features percolate
- **Unification**: Feature compatibility

Rich linguistic representation.

### Lexical Functional Grammar (LFG)

LFG separates:
- **C-structure**: Constituency
- **F-structure**: Functional relations

Handles complex linguistic phenomena.

## Key Takeaways

1. **Syntactic parsing reveals sentence structure**: Identifying how words group into phrases and how phrases relate enables deeper language understanding.

2. **CFGs define phrase structure**: Context-free grammars provide formal framework for constituency parsing, though ambiguity and coverage are challenges.

3. **CYK enables efficient parsing**: Dynamic programming algorithm parses sentences in cubic time, though requires CNF grammar form.

4. **PCFGs handle ambiguity**: Probabilistic grammars disambiguate multiple parses by choosing most likely, and can be learned from treebanks.

5. **Dependency parsing captures relationships**: Head-dependent relationships provide alternative to phrase structure, often more suitable for some languages and tasks.

6. **Transition-based parsing is efficient**: Greedy sequence of actions builds dependency trees incrementally, enabling fast parsing with learned classifiers.

7. **Graph-based parsing optimizes globally**: Finding maximum spanning tree considers all arcs simultaneously, potentially more accurate than greedy approaches.

8. **Grammar formalisms extend CFGs**: TAG, CCG, HPSG, and LFG handle linguistic phenomena beyond CFG capabilities, trading expressiveness for complexity.
