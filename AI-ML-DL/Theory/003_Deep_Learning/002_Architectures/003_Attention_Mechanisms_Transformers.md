# Attention Mechanisms and Transformers

## Table of Contents

1. [Introduction](#introduction)
2. [Attention Mechanism Fundamentals](#attention-mechanism-fundamentals)
3. [Self-Attention](#self-attention)
4. [Multi-Head Attention](#multi-head-attention)
5. [Positional Encoding](#positional-encoding)
6. [Transformer Architecture](#transformer-architecture)
7. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
8. [Variants and Improvements](#variants-and-improvements)
9. [Applications and Impact](#applications-and-impact)
10. [Key Takeaways](#key-takeaways)

## Introduction

Attention mechanisms revolutionized sequence modeling by allowing models to focus on relevant parts of the input when making predictions. The Transformer architecture, built entirely on attention mechanisms, replaced RNNs in many applications and became the foundation for modern language models like BERT, GPT, and their successors.

This chapter covers the mathematical foundations of attention, the Transformer architecture, and how these innovations enable effective sequence modeling without recurrence.

## Attention Mechanism Fundamentals

### Basic Attention

Attention computes a weighted combination of values based on query-key similarity:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

where:
- $\mathbf{Q}$: Queries (what we're looking for)
- $\mathbf{K}$: Keys (what we're matching against)
- $\mathbf{V}$: Values (what we retrieve)
- $d_k$: Dimension of keys (for scaling)

### Intuition

1. **Query**: "What am I looking for?"
2. **Key**: "What does each position offer?"
3. **Value**: "What information does each position contain?"

Attention computes similarity between queries and keys, then uses these similarities to weight the values.

### Scaled Dot-Product Attention

The scaling factor $\sqrt{d_k}$ prevents dot products from growing too large:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**Why Scaling**: Without scaling, dot products have variance $d_k$, causing softmax to saturate. Scaling by $\sqrt{d_k}$ normalizes variance to 1.

### Attention Weights

The attention weights $\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)$ represent:

- **Interpretability**: Which positions the model attends to
- **Visualization**: Can be visualized as heatmaps
- **Understanding**: Reveals what the model focuses on

## Self-Attention

Self-attention uses the same sequence for queries, keys, and values.

### Formulation

For input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

$$\text{SelfAttention}(\mathbf{X}) = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$$

### Properties

1. **Permutation Equivariant**: Order of input affects output order
2. **No Recurrence**: Can be computed in parallel
3. **Long-Range Dependencies**: Direct connections between all positions
4. **Quadratic Complexity**: $O(n^2)$ in sequence length

### Comparison with RNNs

| Property | RNN | Self-Attention |
|----------|-----|----------------|
| Parallelization | Sequential | Parallel |
| Long-range | Hard | Easy |
| Complexity | $O(n)$ | $O(n^2)$ |
| Memory | $O(n)$ | $O(n^2)$ |

### Computational Complexity

For sequence length $n$ and dimension $d$:
- **Time**: $O(n^2 \cdot d)$ for attention matrix
- **Space**: $O(n^2)$ for attention weights

This quadratic complexity limits application to very long sequences.

## Multi-Head Attention

Multi-head attention applies attention multiple times in parallel with different learned projections.

### Formulation

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

where each head is:

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

and:
- $h$: Number of heads
- $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V$: Learned projections for head $i$
- $\mathbf{W}^O$: Output projection

### Why Multiple Heads

1. **Different Representations**: Each head can attend to different aspects
2. **Specialization**: Heads may specialize in different patterns
3. **Capacity**: Increases model capacity
4. **Interpretability**: Different heads may capture different relationships

### Typical Configuration

- **Number of Heads**: 8, 12, or 16
- **Head Dimension**: $d_k = d_{\text{model}} / h$ (typically 64)
- **Total Parameters**: Similar to single-head with same $d_{\text{model}}$

## Positional Encoding

Since attention is permutation-equivariant, positional information must be added explicitly.

### Sinusoidal Positional Encoding

Adds positional encodings to input embeddings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where:
- $pos$: Position in sequence
- $i$: Dimension index
- $d_{\text{model}}$: Model dimension

### Properties

1. **Deterministic**: Fixed, not learned
2. **Extrapolation**: Can extend to longer sequences
3. **Relative Positions**: Encodes relative positions through trigonometric relationships

### Learned Positional Embeddings

Alternative: Learn positional embeddings as parameters

$$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{n_{\max} \times d_{\text{model}}}$$

**Advantages**: Can learn optimal positions
**Disadvantages**: Fixed maximum length, no extrapolation

### Relative Position Encoding

Encodes relative positions instead of absolute:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T + \mathbf{R}}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{R}$ encodes relative positions.

## Transformer Architecture

The Transformer consists of encoder and decoder stacks built entirely from attention and feedforward layers.

### Encoder Block

Each encoder block contains:

1. **Multi-Head Self-Attention**
2. **Residual Connection & Layer Norm**
3. **Feedforward Network**
4. **Residual Connection & Layer Norm**

**Mathematical Formulation**:

$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$

### Decoder Block

Decoder blocks include:

1. **Masked Multi-Head Self-Attention** (causal masking)
2. **Multi-Head Cross-Attention** (encoder-decoder attention)
3. **Feedforward Network**
4. **Residual Connections & Layer Norm**

**Masked Self-Attention**: Prevents attending to future positions:

$$A_{ij} = \begin{cases}
\frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{l=1}^i \exp(q_i^T k_l / \sqrt{d_k})} & \text{if } j \leq i \\
0 & \text{if } j > i
\end{cases}$$

### Feedforward Network

Two linear transformations with ReLU:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

Typically: $d_{\text{ff}} = 4 \times d_{\text{model}}$

### Layer Normalization

Normalizes across features:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d} \sum_{i=1}^d x_i$, $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$

### Full Architecture

**Encoder Stack**:
- Input embeddings + positional encoding
- $N$ encoder blocks
- Output: Contextualized representations

**Decoder Stack**:
- Output embeddings + positional encoding
- $N$ decoder blocks
- Output: Generated sequence

## Encoder-Decoder Architecture

The Transformer uses an encoder-decoder architecture for sequence-to-sequence tasks.

### Encoder

Processes input sequence:

$$\mathbf{H}_{\text{enc}} = \text{Encoder}(\mathbf{X}_{\text{src}})$$

Produces contextualized representations of source sequence.

### Decoder

Generates target sequence autoregressively:

$$P(y_t | y_{<t}, \mathbf{X}_{\text{src}}) = \text{softmax}(\text{Decoder}(\mathbf{Y}_{<t}, \mathbf{H}_{\text{enc}}))$$

### Cross-Attention

Decoder attends to encoder outputs:

$$\text{CrossAttention}(\mathbf{Q}_{\text{dec}}, \mathbf{K}_{\text{enc}}, \mathbf{V}_{\text{enc}})$$

This allows decoder to focus on relevant parts of source.

### Training

**Teacher Forcing**: During training, decoder receives ground truth tokens:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* | y_{<t}^*, \mathbf{X}_{\text{src}})$$

**Inference**: Autoregressive generation:

$$y_t \sim P(y_t | y_{<t}, \mathbf{X}_{\text{src}})$$

## Variants and Improvements

### Efficient Attention

**Sparse Attention**: Only attend to subset of positions
- **Local Attention**: Attend to local window
- **Strided Attention**: Attend to every $k$-th position
- **Global Attention**: Attend to few global positions

**Linear Attention**: Approximate attention with linear complexity:
- **Performer**: Random feature maps
- **Linformer**: Low-rank approximation
- **Linear Transformer**: Kernel-based linearization

### Architecture Variants

**Pre-LayerNorm vs. Post-LayerNorm**:
- Pre-LN: LayerNorm before sublayer (more stable)
- Post-LN: LayerNorm after sublayer (original)

**GLU Variants**: Gated Linear Units in feedforward:
$$\text{GLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1) \odot \sigma(\mathbf{x}\mathbf{W}_2 + \mathbf{b}_2)$$

**Depth Scaling**: Vary depth across layers

### Positional Encoding Improvements

**Rotary Position Embedding (RoPE)**: Rotates query-key pairs:
- Better extrapolation
- Relative position encoding
- Used in LLaMA, PaLM

**ALiBi**: Attention with Linear Biases:
- Adds learned bias to attention scores
- No positional embeddings needed
- Better extrapolation

## Applications and Impact

### Language Models

**GPT**: Decoder-only, autoregressive language modeling

**BERT**: Encoder-only, bidirectional pretraining

**T5**: Encoder-decoder, text-to-text transfer

### Machine Translation

Transformer achieved state-of-the-art in WMT benchmarks, replacing RNN-based models.

### Vision Transformers (ViT)

Applied transformers to images:
- Split image into patches
- Treat patches as sequence
- Self-attention over patches

### Multimodal Models

CLIP, DALL-E combine vision and language using transformer architectures.

### Scaling Laws

Transformers scale predictably:
- Performance improves with model size
- Data and compute requirements scale
- Emergent abilities at scale

## Key Takeaways

1. **Attention Mechanism**: Computes weighted combinations based on query-key similarity, enabling models to focus on relevant information.

2. **Self-Attention**: Allows each position to attend to all other positions, capturing long-range dependencies without recurrence.

3. **Multi-Head Attention**: Applies attention multiple times in parallel with different projections, increasing model capacity and enabling specialization.

4. **Positional Encoding**: Adds positional information since attention is permutation-equivariant, with sinusoidal encodings providing extrapolation capabilities.

5. **Transformer Architecture**: Built entirely on attention and feedforward layers, enabling parallel processing and effective sequence modeling.

6. **Encoder-Decoder**: Encoder processes input, decoder generates output with cross-attention connecting them, enabling sequence-to-sequence tasks.

7. **Quadratic Complexity**: Self-attention has $O(n^2)$ complexity, limiting application to very long sequences and motivating efficient variants.

8. **Variants**: Sparse attention, linear attention, and architectural improvements address limitations and improve efficiency.

9. **Impact**: Transformers revolutionized NLP and enabled large language models, becoming the foundation for modern AI systems.

10. **Practical Considerations**: Layer normalization, residual connections, and proper initialization are crucial for training stable transformers.
