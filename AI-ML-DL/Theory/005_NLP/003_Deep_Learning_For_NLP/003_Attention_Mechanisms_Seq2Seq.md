# Attention Mechanisms and Sequence-to-Sequence Models

## Table of Contents

1. [Introduction](#introduction)
2. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
3. [Attention Mechanism](#attention-mechanism)
4. [Bahdanau Attention](#bahdanau-attention)
5. [Luong Attention](#luong-attention)
6. [Pointer Networks and Copy Mechanism](#pointer-networks-and-copy-mechanism)
7. [Attention Variants](#attention-variants)
8. [Applications](#applications)
9. [Visualization and Interpretation](#visualization-and-interpretation)
10. [Key Takeaways](#key-takeaways)

## Introduction

Attention mechanisms enable sequence-to-sequence models to focus on relevant parts of the input when generating each output token. This addresses the bottleneck of compressing entire input sequences into fixed-size vectors, dramatically improving performance on tasks like machine translation.

The key insight: Instead of encoding all information into a single vector, allow the decoder to attend to different parts of the encoder output at each decoding step. This enables:
- **Long-range dependencies**: Handle long input sequences
- **Selective focus**: Attend to relevant input parts
- **Interpretability**: Visualize what the model focuses on

Attention revolutionized neural machine translation and became fundamental to transformer architectures.

## Encoder-Decoder Architecture

Encoder-decoder (seq2seq) models map variable-length input sequences to variable-length output sequences.

### Basic Architecture

**Encoder**: Processes input sequence $\mathbf{x} = (x_1, \ldots, x_m)$ into hidden states $\mathbf{h} = (h_1, \ldots, h_m)$

**Decoder**: Generates output sequence $\mathbf{y} = (y_1, \ldots, y_n)$ from encoder representation

**Bottleneck**: Encoder output $\mathbf{c}$ (often just final hidden state) must encode all input information.

### Encoder

RNN encoder processes input:

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)$$

**Output**: Sequence of hidden states $\mathbf{h} = (\mathbf{h}_1, \ldots, \mathbf{h}_m)$

**Context vector**: Often just $\mathbf{c} = \mathbf{h}_m$ (final hidden state)

### Decoder

RNN decoder generates output:

$$\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c})$$

$$P(y_t | y_{<t}, \mathbf{x}) = g(\mathbf{s}_t, y_{t-1}, \mathbf{c})$$

where $\mathbf{s}_t$ is decoder hidden state and $\mathbf{c}$ is context from encoder.

### Limitations

**Fixed-size bottleneck**: Single vector $\mathbf{c}$ must encode entire input
**Information loss**: Long sequences lose information
**No selective focus**: Cannot focus on relevant parts

Attention addresses these limitations.

## Attention Mechanism

Attention allows the decoder to dynamically focus on different parts of the encoder output.

### Attention Intuition

At each decoding step, compute:
1. **Attention scores**: How relevant each encoder position is
2. **Attention weights**: Normalized scores (probabilities)
3. **Context vector**: Weighted sum of encoder hidden states

### Attention Formulation

**Attention scores**:
$$e_{t,i} = \text{score}(\mathbf{s}_{t-1}, \mathbf{h}_i)$$

**Attention weights**:
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{m} \exp(e_{t,j})}$$

**Context vector**:
$$\mathbf{c}_t = \sum_{i=1}^{m} \alpha_{t,i} \mathbf{h}_i$$

**Decoder**:
$$\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$

### Attention Properties

**Dynamic**: Different attention for each output position
**Selective**: Focuses on relevant input parts
**Interpretable**: Attention weights show what model focuses on
**Flexible**: Can attend to multiple positions simultaneously

## Bahdanau Attention

Bahdanau (additive) attention uses a learned alignment model.

### Scoring Function

**Additive attention**:
$$e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{s}_{t-1} + \mathbf{W}_2 \mathbf{h}_i)$$

where $\mathbf{v}$, $\mathbf{W}_1$, $\mathbf{W}_2$ are learned parameters.

**Intuition**: Learn how well decoder state $\mathbf{s}_{t-1}$ aligns with encoder state $\mathbf{h}_i$.

### Architecture

**Encoder**: Bidirectional RNN produces $\mathbf{h}_i = [\overrightarrow{\mathbf{h}}_i; \overleftarrow{\mathbf{h}}_i]$

**Decoder**: 
$$\mathbf{s}_t = \text{GRU}(\mathbf{s}_{t-1}, [y_{t-1}; \mathbf{c}_t])$$

where $\mathbf{c}_t$ is attention context vector.

**Output**:
$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o [\mathbf{s}_t; \mathbf{c}_t])$$

### Training

Jointly train encoder, decoder, and attention parameters via backpropagation.

**Advantages**:
- Handles variable-length sequences
- Learns alignment automatically
- Improves translation quality

## Luong Attention

Luong (multiplicative) attention uses simpler dot-product scoring.

### Scoring Functions

**Dot product**:
$$e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$$

**General**:
$$e_{t,i} = \mathbf{s}_t^T \mathbf{W} \mathbf{h}_i$$

**Concat**:
$$e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W} [\mathbf{s}_t; \mathbf{h}_i])$$

### Architecture Differences

**Luong**: Compute attention after decoder state $\mathbf{s}_t$
**Bahdanau**: Compute attention before decoder state (uses $\mathbf{s}_{t-1}$)

**Luong variants**:
- **Global**: Attend to all encoder positions
- **Local**: Attend to subset (local-p or monotonic)

### Comparison

**Bahdanau**:
- More parameters
- Often better for long sequences
- Computationally more expensive

**Luong**:
- Simpler (dot product)
- Faster computation
- Often comparable performance

## Pointer Networks and Copy Mechanism

Pointer networks and copy mechanisms enable copying words directly from input.

### Copy Mechanism

Some words should be copied from source (e.g., names, numbers):

**Generate mode**: Sample from vocabulary
**Copy mode**: Copy from input

**Probability**:
$$P(y_t) = p_{gen} P_{vocab}(y_t) + (1-p_{gen}) \sum_{i: x_i = y_t} \alpha_{t,i}$$

where $p_{gen}$ is generation probability.

### Pointer Networks

Pointer networks output positions instead of vocabulary tokens:

**Attention as pointer**: $\alpha_{t,i}$ directly indicates input position to copy

**Applications**: 
- Summarization (copy phrases)
- Question answering (copy spans)
- Code generation (copy identifiers)

### Applications

**Summarization**: Copy important phrases
**Dialogue**: Copy entity names
**Code generation**: Copy variable names, strings

## Attention Variants

Various attention mechanisms address different needs.

### Multi-Head Attention

Attend to multiple representation subspaces simultaneously:

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

Enables attending to different types of information.

### Self-Attention

Attention within same sequence:

$$\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$$

Captures relationships between all positions.

### Hierarchical Attention

**Word-level**: Attend to words in sentences
**Sentence-level**: Attend to sentences in documents

Useful for document-level tasks.

### Sparse Attention

**Local attention**: Attend only to nearby positions
**Strided attention**: Attend to every $k$-th position
**Random attention**: Attend to random subset

Reduces computational cost for long sequences.

## Applications

Attention improves many sequence-to-sequence tasks.

### Machine Translation

**Encoder**: Process source sentence
**Decoder**: Generate target with attention to source
**Alignment**: Attention weights show word alignments

### Summarization

**Encoder**: Process source document
**Decoder**: Generate summary
**Attention**: Focus on important sentences/phrases

### Question Answering

**Encoder**: Process question and context
**Decoder**: Generate answer
**Attention**: Focus on relevant context spans

### Image Captioning

**Encoder**: CNN processes image (spatial features)
**Decoder**: RNN generates caption
**Attention**: Focus on image regions

## Visualization and Interpretation

Attention weights provide interpretability.

### Attention Heatmaps

Visualize attention weights as heatmaps:
- Rows: Output positions
- Columns: Input positions
- Colors: Attention strength

**Interpretation**: Shows alignment between input and output.

### Alignment Analysis

Compare attention to:
- **Gold alignments**: From parallel corpora
- **Linguistic structure**: Phrase boundaries, dependencies

Attention often learns linguistically meaningful alignments.

### Attention Patterns

Common patterns:
- **Diagonal**: Monotonic alignment (left-to-right)
- **Block**: Phrase-level alignment
- **Scattered**: Complex reordering

## Key Takeaways

1. **Attention solves the bottleneck problem**: Allowing decoders to attend to encoder outputs eliminates the need to compress entire sequences into fixed-size vectors.

2. **Attention enables selective focus**: Models can dynamically focus on relevant input parts at each decoding step, improving handling of long sequences.

3. **Bahdanau attention uses learned alignment**: Additive attention learns how to align decoder and encoder states, often performing well on complex tasks.

4. **Luong attention is simpler**: Dot-product attention is computationally efficient and often achieves comparable performance to additive attention.

5. **Copy mechanisms handle OOV**: Enabling models to copy words from input addresses out-of-vocabulary problems and improves performance on tasks requiring exact copying.

6. **Multi-head attention captures multiple relationships**: Attending to different representation subspaces enables richer modeling of input-output relationships.

7. **Attention provides interpretability**: Visualization of attention weights reveals what models focus on, enabling debugging and analysis.

8. **Attention is fundamental to modern NLP**: The attention mechanism underlies transformer architectures and has become essential for state-of-the-art NLP systems.
