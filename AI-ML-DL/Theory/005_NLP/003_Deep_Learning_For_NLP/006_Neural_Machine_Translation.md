# Neural Machine Translation

## Table of Contents

1. [Introduction](#introduction)
2. [Sequence-to-Sequence Architecture](#sequence-to-sequence-architecture)
3. [Attention in Neural MT](#attention-in-neural-mt)
4. [Beam Search Decoding](#beam-search-decoding)
5. [Subword Segmentation](#subword-segmentation)
6. [Multilingual Neural MT](#multilingual-neural-mt)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Challenges and Solutions](#challenges-and-solutions)
9. [Recent Advances](#recent-advances)
10. [Key Takeaways](#key-takeaways)

## Introduction

Neural Machine Translation (NMT) uses neural networks to translate between languages, replacing phrase-based statistical methods. NMT models learn end-to-end mappings from source to target sentences, achieving state-of-the-art translation quality.

Key advantages over statistical MT:
- **End-to-end learning**: Single model learns translation directly
- **Context awareness**: Handles long-range dependencies
- **Fewer hand-crafted features**: Learns representations automatically
- **Better fluency**: Generates more natural translations

NMT has become the dominant approach, powering commercial translation systems and enabling new capabilities like zero-shot multilingual translation.

## Sequence-to-Sequence Architecture

NMT uses encoder-decoder architecture to map source sequences to target sequences.

### Basic Architecture

**Encoder**: Processes source sentence $\mathbf{x} = (x_1, \ldots, x_m)$ into hidden states
**Decoder**: Generates target sentence $\mathbf{y} = (y_1, \ldots, y_n)$ from encoder representation

**RNN-based**:
- Encoder: Bidirectional RNN
- Decoder: Unidirectional RNN with attention

**Transformer-based**: Full Transformer encoder-decoder (current standard)

### Encoder

**RNN encoder**:
$$\mathbf{h}_t = \text{RNN}(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

**Transformer encoder**: Stack of self-attention and feedforward layers

**Output**: Sequence of hidden states $\mathbf{H} = (\mathbf{h}_1, \ldots, \mathbf{h}_m)$

### Decoder

**RNN decoder** (with attention):
$$\mathbf{s}_t = \text{RNN}(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$

where $\mathbf{c}_t$ is attention context vector.

**Transformer decoder**: Masked self-attention + encoder-decoder attention

### Training Objective

Maximize conditional likelihood:

$$L = \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \log P(\mathbf{y} | \mathbf{x}) = \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \sum_{t=1}^{n} \log P(y_t | y_{<t}, \mathbf{x})$$

**Teacher forcing**: Use ground truth $y_{<t}$ during training (not predictions).

## Attention in Neural MT

Attention mechanisms enable decoders to focus on relevant source positions.

### Attention Mechanism

At each decoding step $t$:

**Attention scores**:
$$e_{t,i} = \text{score}(\mathbf{s}_{t-1}, \mathbf{h}_i)$$

**Attention weights**:
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{m} \exp(e_{t,j})}$$

**Context vector**:
$$\mathbf{c}_t = \sum_{i=1}^{m} \alpha_{t,i} \mathbf{h}_i$$

**Decoder**:
$$\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$

### Attention Benefits

**Alignment**: Attention weights show word alignments
**Long-range dependencies**: Can attend to distant source words
**Selective focus**: Focus on relevant source parts
**Interpretability**: Visualize what model focuses on

### Multi-Head Attention

Transformer uses multi-head attention:

**Multiple attention heads**: Attend to different aspects
**Encoder-decoder attention**: Decoder attends to encoder output
**Self-attention**: Within encoder and decoder

## Beam Search Decoding

Beam search finds high-probability translations by maintaining multiple hypotheses.

### Greedy Decoding

**Greedy**: Always choose highest probability token:
$$\hat{y}_t = \arg\max_{y_t} P(y_t | y_{<t}, \mathbf{x})$$

**Problem**: Local optimal choices may not lead to globally best translation.

### Beam Search Algorithm

**Beam width $k$**: Maintain top-$k$ hypotheses

**At each step**:
1. Extend each hypothesis with all possible next tokens
2. Score all extensions
3. Keep top-$k$ hypotheses

**Termination**: When all hypotheses generate `<EOS>` token

### Beam Search Formulation

**Score**: Log probability (or length-normalized):

$$\text{score}(\mathbf{y}) = \sum_{t=1}^{n} \log P(y_t | y_{<t}, \mathbf{x})$$

**Length normalization**:
$$\text{score}(\mathbf{y}) = \frac{1}{|\mathbf{y}|^\alpha} \sum_{t=1}^{n} \log P(y_t | y_{<t}, \mathbf{x})$$

where $\alpha$ controls length penalty (typically 0.6-0.7).

### Beam Search Trade-offs

**Larger beam**: Better translations but slower
**Smaller beam**: Faster but may miss good translations
**Typical beam size**: 4-10

## Subword Segmentation

Subword segmentation handles rare words and enables open vocabulary.

### Word-Level Problems

**Out-of-vocabulary**: Rare words not in vocabulary
**Morphology**: Complex word formation
**Vocabulary size**: Large vocabularies increase parameters

### Byte Pair Encoding (BPE)

**Algorithm**:
1. Start with character-level vocabulary
2. Count all adjacent symbol pairs
3. Merge most frequent pair
4. Repeat until desired vocabulary size

**Example**: "low", "lower", "newest" → "low", "low er", "new est"

### SentencePiece

**Language-agnostic**: Works for any language
**Reversible**: Can reconstruct original text
**Subword sampling**: Multiple segmentations for regularization

### WordPiece

Similar to BPE but uses likelihood-based merging (used in BERT).

### Benefits

**OOV handling**: Can represent any word
**Morphological awareness**: Captures word structure
**Efficiency**: Smaller vocabulary than word-level
**Consistency**: Same segmentation across training and inference

## Multilingual Neural MT

Multilingual models translate between multiple language pairs.

### Many-to-Many Translation

**Single model**: Handles all language pairs
**Language tags**: Prepend source/target language IDs
**Shared representations**: Learn cross-lingual representations

**Example input**: `<2en> <2de> Hello world` (translate to German)

### Zero-Shot Translation

**Training**: Train on some language pairs
**Zero-shot**: Translate between pairs not seen during training

**Example**: Train on EN↔FR and EN↔DE, can translate FR↔DE

### Benefits

**Efficiency**: Single model for multiple languages
**Low-resource**: Transfer from high-resource to low-resource languages
**Cross-lingual**: Learns shared representations

### Challenges

**Language imbalance**: Some languages have more data
**Interference**: Languages may interfere with each other
**Quality**: May be worse than language-specific models

## Evaluation Metrics

Translation quality evaluation uses automatic and human metrics.

### BLEU Score

**BLEU**: N-gram precision with brevity penalty

$$\text{BLEU} = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$: N-gram precision
- $BP$: Brevity penalty
- $w_n$: Weights (typically uniform)

**Range**: 0 to 1 (often reported as percentage)

### Other Metrics

**METEOR**: Considers synonyms and stemming
**TER**: Translation Error Rate (edit distance)
**chrF**: Character-level F-score
**BERTScore**: Semantic similarity using BERT

### Human Evaluation

**Adequacy**: How much meaning is preserved
**Fluency**: How natural the translation is
**Preference**: Compare multiple translations

Human evaluation is gold standard but expensive.

## Challenges and Solutions

NMT faces several challenges requiring specialized solutions.

### Rare Words

**Problem**: OOV words cannot be translated
**Solutions**:
- Subword segmentation
- Copy mechanism
- Back-translation for rare word pairs

### Long Sentences

**Problem**: Performance degrades for long sentences
**Solutions**:
- Attention mechanisms
- Hierarchical encoding
- Sentence splitting

### Domain Adaptation

**Problem**: Performance drops on new domains
**Solutions**:
- Fine-tuning on domain data
- Multi-domain training
- Domain tags

### Low-Resource Languages

**Problem**: Limited parallel data
**Solutions**:
- Multilingual training
- Back-translation
- Unsupervised methods

## Recent Advances

Recent developments improve NMT quality and capabilities.

### Large Language Models

**GPT-3, PaLM**: Can translate with few examples
**Few-shot translation**: Translate with minimal parallel data
**Prompting**: Natural language instructions

### Non-Autoregressive Translation

**Parallel decoding**: Generate all tokens simultaneously
**Faster**: No sequential dependencies
**Quality**: Approaching autoregressive models

### Retrieval-Augmented Translation

**Retrieve similar examples**: Use similar translations from memory
**Better rare words**: Copy from retrieved examples
**Domain adaptation**: Retrieve domain-specific examples

## Key Takeaways

1. **NMT learns end-to-end translation**: Encoder-decoder architectures learn direct mappings from source to target, eliminating need for hand-crafted features.

2. **Attention enables alignment**: Attention mechanisms show word alignments and enable focusing on relevant source positions during translation.

3. **Beam search finds good translations**: Maintaining multiple hypotheses enables finding translations better than greedy decoding.

4. **Subword segmentation handles OOV**: BPE, SentencePiece, and WordPiece enable representing any word and handling rare words effectively.

5. **Multilingual models enable transfer**: Single models can handle multiple language pairs and enable zero-shot translation between unseen pairs.

6. **BLEU provides automatic evaluation**: While imperfect, BLEU enables rapid iteration and comparison of translation systems.

7. **Domain adaptation is crucial**: Fine-tuning on domain-specific data significantly improves translation quality for specialized domains.

8. **NMT continues evolving**: Large language models, non-autoregressive methods, and retrieval augmentation push translation quality and capabilities forward.
