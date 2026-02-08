# Transformer Architecture and BERT

## Table of Contents

1. [Introduction](#introduction)
2. [Transformer Architecture Overview](#transformer-architecture-overview)
3. [Self-Attention Mechanism](#self-attention-mechanism)
4. [Multi-Head Attention](#multi-head-attention)
5. [Positional Encoding](#positional-encoding)
6. [Encoder and Decoder Stacks](#encoder-and-decoder-stacks)
7. [BERT: Bidirectional Encoder Representations](#bert-bidirectional-encoder-representations)
8. [Masked Language Modeling](#masked-language-modeling)
9. [Next Sentence Prediction](#next-sentence-prediction)
10. [Key Takeaways](#key-takeaways)

## Introduction

The Transformer architecture, introduced in "Attention Is All You Need," replaces recurrent layers with self-attention mechanisms, enabling parallel processing and capturing long-range dependencies. BERT adapts the Transformer encoder for bidirectional language understanding through masked language modeling.

Key innovations:
- **Self-attention**: Captures relationships between all positions
- **Parallelization**: No sequential dependencies enable parallel training
- **Long-range dependencies**: Attention spans entire sequence
- **Bidirectional context**: BERT processes both directions simultaneously

Transformers and BERT revolutionized NLP, achieving state-of-the-art on many tasks and enabling large-scale pre-training.

## Transformer Architecture Overview

The Transformer uses stacked encoder and decoder layers with self-attention and feedforward networks.

### Overall Architecture

**Encoder**: $N$ identical layers, each with:
- Multi-head self-attention
- Position-wise feedforward network
- Residual connections and layer normalization

**Decoder**: $N$ identical layers, each with:
- Masked multi-head self-attention
- Multi-head encoder-decoder attention
- Position-wise feedforward network
- Residual connections and layer normalization

### Key Components

**Self-attention**: Relationships within sequence
**Positional encoding**: Injects position information
**Feedforward networks**: Position-wise transformations
**Residual connections**: Enable deep networks
**Layer normalization**: Stabilizes training

## Self-Attention Mechanism

Self-attention computes relationships between all positions in a sequence.

### Attention Formulation

Given input $\mathbf{X} \in \mathbb{R}^{n \times d}$, compute:

**Query, Key, Value**:
$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

**Attention scores**:
$$\mathbf{A} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$$

**Attention weights**:
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}(\mathbf{A})\mathbf{V}$$

### Scaled Dot-Product Attention

**Scaling factor** $\sqrt{d_k}$ prevents dot products from growing large:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Why scaling**: Dot products have variance $d_k$, scaling by $\sqrt{d_k}$ normalizes variance.

### Self-Attention Properties

**Parallelizable**: All positions computed simultaneously
**Long-range**: Direct connections between distant positions
**Interpretable**: Attention weights show relationships
**Flexible**: Can attend to any position

## Multi-Head Attention

Multi-head attention attends to multiple representation subspaces simultaneously.

### Multi-Head Formulation

**Multiple heads**:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

**Concatenation**:
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

where $h$ is number of heads and $\mathbf{W}^O$ projects to output dimension.

### Why Multiple Heads

**Different relationships**: Each head can focus on different types
- Syntactic relationships
- Semantic relationships
- Long-range dependencies
- Local patterns

**Representation capacity**: Increases model expressiveness

### Typical Configuration

- **Number of heads**: $h = 8$ or $16$
- **Head dimension**: $d_k = d_{model}/h$
- **Total parameters**: Same as single-head with $d_{model}$ dimension

## Positional Encoding

Since self-attention is permutation-invariant, positional encodings inject position information.

### Sinusoidal Positional Encoding

Add positional encoding to input embeddings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

where $pos$ is position and $i$ is dimension.

**Properties**:
- **Relative positions**: Encodes relative position relationships
- **Extrapolation**: Can handle longer sequences than training
- **Deterministic**: No learned parameters

### Learned Positional Embeddings

Alternative: Learn positional embeddings as parameters.

**Advantages**: Can learn optimal position representations
**Disadvantages**: Fixed maximum length, no extrapolation

### Positional Encoding in Practice

Both approaches work well. Sinusoidal is more common in original Transformer, learned embeddings common in BERT and GPT.

## Encoder and Decoder Stacks

The Transformer stacks multiple encoder and decoder layers.

### Encoder Layer

Each encoder layer:

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$

**Components**:
- **Self-attention**: Captures relationships within input
- **Feedforward**: Position-wise transformation
- **Residual**: Enables gradient flow
- **LayerNorm**: Normalizes activations

### Decoder Layer

Each decoder layer:

**Masked self-attention**: Prevents attending to future positions
**Encoder-decoder attention**: Attends to encoder output
**Feedforward**: Position-wise transformation

**Masking**: Set attention scores to $-\infty$ for future positions before softmax.

### Feedforward Networks

Position-wise feedforward:

$$\text{FFN}(x) = \max(0, x\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Applied independently to each position.

**Typical size**: $d_{ff} = 4 \times d_{model}$

## BERT: Bidirectional Encoder Representations

BERT uses Transformer encoder for bidirectional language understanding.

### BERT Architecture

**Base model**: 12 layers, 12 attention heads, 768 hidden size
**Large model**: 24 layers, 16 attention heads, 1024 hidden size

**Input representation**:
- **Token embeddings**: WordPiece tokenization
- **Segment embeddings**: Distinguish sentence pairs
- **Position embeddings**: Learned positional encodings

### Bidirectional Context

Unlike autoregressive models (GPT), BERT processes both directions:
- **Left context**: Words before target
- **Right context**: Words after target

Enables richer representations for understanding tasks.

### Input Format

**Single sentence**: `[CLS] token1 token2 ... [SEP]`
**Sentence pair**: `[CLS] sentence1 [SEP] sentence2 [SEP]`

**Special tokens**:
- `[CLS]`: Classification token (used for sentence-level tasks)
- `[SEP]`: Separator token

## Masked Language Modeling

BERT is pre-trained using Masked Language Modeling (MLM).

### MLM Objective

Randomly mask 15% of tokens:
- **80%**: Replace with `[MASK]`
- **10%**: Replace with random token
- **10%**: Keep original

**Prediction**: Predict original token from context:

$$L_{MLM} = -\sum_{i \in \text{masked}} \log P(x_i | \mathbf{x}_{\backslash i})$$

### Why Masking

**Bidirectional context**: Can use both directions (unlike left-to-right models)
**Denoising**: Learns to recover masked information
**Robustness**: Handles missing/corrupted tokens

### MLM Variants

**Whole word masking**: Mask entire words (not subwords)
**Span masking**: Mask contiguous spans
**Dynamic masking**: Different masks each epoch

## Next Sentence Prediction

NSP helps BERT understand sentence relationships.

### NSP Task

Given sentence pair $(A, B)$:
- **50%**: $B$ follows $A$ (positive)
- **50%**: $B$ is random sentence (negative)

**Prediction**: Binary classification using `[CLS]` token:

$$L_{NSP} = -\log P(\text{IsNext} | \mathbf{x})$$

### NSP Effectiveness

**Debated**: Some studies show NSP helps, others show minimal benefit
**Removed**: RoBERTa removes NSP, still performs well
**Alternative**: Sentence order prediction (ALBERT)

### Combined Pre-training

BERT optimizes:

$$L = L_{MLM} + L_{NSP}$$

Both objectives trained jointly.

## Key Takeaways

1. **Transformer replaces recurrence with self-attention**: Enabling parallel processing and capturing long-range dependencies without sequential computation.

2. **Self-attention captures all pairwise relationships**: Computing attention between all positions enables rich modeling of sequence structure.

3. **Multi-head attention increases expressiveness**: Attending to multiple representation subspaces enables capturing different types of relationships simultaneously.

4. **Positional encoding injects order information**: Since self-attention is permutation-invariant, positional encodings are essential for sequence modeling.

5. **BERT enables bidirectional understanding**: Unlike autoregressive models, BERT processes both directions simultaneously, improving understanding tasks.

6. **Masked language modeling learns rich representations**: Predicting masked tokens from bidirectional context enables learning powerful language representations.

7. **Pre-training enables transfer learning**: BERT's pre-trained representations transfer effectively to downstream tasks with minimal task-specific architecture.

8. **Transformer architecture is foundational**: The Transformer architecture underlies most modern NLP models and enables scaling to very large models.
