# Sequence Models: RNN and LSTM

## Table of Contents

1. [Introduction](#introduction)
2. [Recurrent Neural Networks](#recurrent-neural-networks)
3. [RNN Architecture](#rnn-architecture)
4. [Backpropagation Through Time](#backpropagation-through-time)
5. [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
6. [Long Short-Term Memory](#long-short-term-memory)
7. [Gated Recurrent Units](#gated-recurrent-units)
8. [Bidirectional RNNs](#bidirectional-rnns)
9. [Applications in NLP](#applications-in-nlp)
10. [Key Takeaways](#key-takeaways)

## Introduction

Recurrent Neural Networks (RNNs) process sequences by maintaining hidden states that capture information from previous time steps. This enables modeling sequential dependencies in text, making RNNs fundamental for language modeling, machine translation, and other sequence-to-sequence tasks.

Unlike feedforward networks, RNNs can:
- **Handle variable-length sequences**: Process inputs of different lengths
- **Capture temporal dependencies**: Use information from previous time steps
- **Share parameters**: Same weights across time steps

However, standard RNNs struggle with long-range dependencies due to vanishing gradients, motivating Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU).

## Recurrent Neural Networks

RNNs extend feedforward networks with recurrent connections that allow information to persist across time steps.

### RNN Motivation

Feedforward networks process fixed-size inputs independently. For sequences:
- **Variable length**: Sentences have different lengths
- **Dependencies**: Later words depend on earlier words
- **Parameter efficiency**: Share parameters across positions

RNNs address these by processing sequences step-by-step while maintaining hidden state.

### RNN Formulation

At each time step $t$:

$$\mathbf{h}_t = f(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = g(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

where:
- $\mathbf{x}_t$: Input at time $t$
- $\mathbf{h}_t$: Hidden state at time $t$
- $\mathbf{y}_t$: Output at time $t$
- $\mathbf{W}_{*}$: Weight matrices
- $f, g$: Activation functions

### Unfolding RNNs

RNNs can be "unfolded" to show computation across time:

```
h_0 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T
      x_1         x_2              x_T
```

This reveals the sequential nature and enables backpropagation.

## RNN Architecture

RNN architectures vary in how inputs, outputs, and hidden states are used.

### Many-to-Many (Sequence-to-Sequence)

**Input**: Sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T$
**Output**: Sequence $\mathbf{y}_1, \ldots, \mathbf{y}_T$

Used for: Sequence labeling, machine translation

### Many-to-One

**Input**: Sequence $\mathbf{x}_1, \ldots, \mathbf{x}_T$
**Output**: Single vector $\mathbf{y}$

Used for: Sentiment analysis, document classification

### One-to-Many

**Input**: Single vector $\mathbf{x}$
**Output**: Sequence $\mathbf{y}_1, \ldots, \mathbf{y}_T$

Used for: Text generation, image captioning

### Encoder-Decoder

**Encoder**: Many-to-one RNN processes input
**Decoder**: One-to-many RNN generates output

Used for: Machine translation, summarization

## Backpropagation Through Time

Backpropagation Through Time (BPTT) extends backpropagation to RNNs by unrolling the network across time steps.

### Forward Pass

Compute hidden states and outputs:

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

### Backward Pass

Compute gradients w.r.t. loss $L$:

$$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L}{\partial \mathbf{y}_t} \frac{\partial \mathbf{y}_t}{\partial \mathbf{h}_t} + \frac{\partial L}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$$

The second term shows how gradients flow backward through time.

### Gradient Computation

For weight matrix $\mathbf{W}_{hh}$:

$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}$$

Gradients accumulate across time steps.

### Truncated BPTT

For long sequences, truncate:
- Process sequences in chunks
- Reset gradients periodically
- Reduces memory and computation

## Vanishing and Exploding Gradients

RNNs suffer from gradient problems that prevent learning long-range dependencies.

### Vanishing Gradients

Gradient w.r.t. $\mathbf{h}_0$:

$$\frac{\partial L}{\partial \mathbf{h}_0} = \prod_{t=1}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} \frac{\partial L}{\partial \mathbf{h}_T}$$

If $|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}| < 1$, gradient vanishes exponentially.

**Effect**: Early time steps receive negligible gradients, cannot learn long dependencies.

### Exploding Gradients

If $|\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}| > 1$, gradient explodes.

**Solution**: Gradient clipping:
$$\text{grad} = \min(1, \frac{\theta}{||\text{grad}||}) \text{grad}$$

### Why Vanishing Gradients Occur

For $\mathbf{h}_t = \tanh(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{b})$:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \mathbf{W}^T \text{diag}(1 - \tanh^2(\mathbf{W} \mathbf{h}_{t-1} + \mathbf{b}))$$

The derivative of $\tanh$ is at most 1, and repeated multiplication causes vanishing.

### Solutions

**Better initialization**: Initialize weights carefully
**Gradient clipping**: Prevent explosion
**Better architectures**: LSTM, GRU (address vanishing)
**Skip connections**: Residual connections help gradients flow

## Long Short-Term Memory

LSTM addresses vanishing gradients through gating mechanisms that control information flow.

### LSTM Architecture

LSTM maintains:
- **Hidden state** $\mathbf{h}_t$: Short-term memory
- **Cell state** $\mathbf{c}_t$: Long-term memory

**Gates**:
- **Forget gate**: What to forget from cell state
- **Input gate**: What new information to store
- **Output gate**: What parts of cell state to output

### LSTM Equations

**Forget gate**:
$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input gate**:
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate values**:
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell state update**:
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output gate**:
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden state**:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\odot$ is element-wise multiplication and $\sigma$ is sigmoid.

### Why LSTM Works

**Constant error carousel**: Cell state gradient can flow unchanged through forget gate (if $\mathbf{f}_t \approx 1$)

**Gating**: Selective information flow prevents irrelevant information from interfering

**Additive updates**: Cell state updates are additive, not multiplicative, helping gradients flow

### LSTM Variants

**Peephole connections**: Gates see cell state directly
**Coupled gates**: Combine forget and input gates
**GRU**: Simplified version (discussed next)

## Gated Recurrent Units

GRU simplifies LSTM by combining gates while maintaining effectiveness.

### GRU Architecture

GRU has two gates:
- **Update gate** $\mathbf{z}_t$: How much to update hidden state
- **Reset gate** $\mathbf{r}_t$: How much to forget previous hidden state

### GRU Equations

**Update gate**:
$$\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Reset gate**:
$$\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Candidate hidden state**:
$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden state update**:
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### GRU vs LSTM

**GRU advantages**:
- Fewer parameters (faster training)
- Simpler architecture
- Often comparable performance

**LSTM advantages**:
- Separate cell state (explicit long-term memory)
- More expressive
- Better for very long sequences

## Bidirectional RNNs

Bidirectional RNNs process sequences in both directions, capturing context from past and future.

### Bidirectional Architecture

Two RNNs:
- **Forward RNN**: Processes sequence left-to-right
- **Backward RNN**: Processes sequence right-to-left

**Combined representation**:
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

Concatenation of forward and backward hidden states.

### Applications

**Sequence labeling**: POS tagging, NER (future context helps)
**Language modeling**: Not applicable (no future context at test time)
**Encoding**: Create rich representations for downstream tasks

### Bidirectional LSTM/GRU

Apply bidirectionality to LSTM or GRU:
- Forward and backward LSTMs/GRUs
- Concatenate or combine outputs
- Common in modern NLP architectures

## Applications in NLP

RNNs and LSTMs enable various NLP applications.

### Language Modeling

Predict next word given previous words:

$$P(w_t | w_1, \ldots, w_{t-1}) = \text{softmax}(\mathbf{W} \mathbf{h}_t + \mathbf{b})$$

**Evaluation**: Perplexity on test set

### Text Generation

Sample from language model:
1. Start with seed text
2. Predict next word distribution
3. Sample word
4. Repeat

**Applications**: Story generation, dialogue systems

### Sequence Labeling

**POS tagging**: Tag each word
**NER**: Identify named entities
**Chunking**: Identify phrase boundaries

**Architecture**: Many-to-many RNN with CRF layer

### Machine Translation

**Encoder-decoder**:
- Encoder: Process source sentence
- Decoder: Generate target sentence

**Attention**: Focus on relevant source words (discussed in next file)

### Sentiment Analysis

**Many-to-one**: Process entire document, output sentiment

**Hierarchical**: Sentence-level then document-level

## Key Takeaways

1. **RNNs process sequences sequentially**: Maintaining hidden states enables modeling temporal dependencies in text and other sequential data.

2. **BPTT enables RNN training**: Unfolding RNNs across time allows backpropagation, though gradients can vanish or explode over long sequences.

3. **Vanishing gradients prevent long-range learning**: Standard RNNs struggle with long dependencies because gradients diminish exponentially through time.

4. **LSTM addresses vanishing gradients**: Gating mechanisms (forget, input, output gates) control information flow, enabling learning of long-range dependencies.

5. **GRU simplifies LSTM**: Fewer gates and parameters while maintaining effectiveness, making GRU a popular alternative to LSTM.

6. **Bidirectional RNNs capture full context**: Processing sequences in both directions provides richer representations for tasks where future context is available.

7. **RNNs enable sequence-to-sequence tasks**: Language modeling, translation, generation, and labeling all benefit from RNN's sequential processing capabilities.

8. **LSTMs remain relevant**: Despite attention mechanisms, LSTMs are still used in many applications and provide interpretable sequential processing.
