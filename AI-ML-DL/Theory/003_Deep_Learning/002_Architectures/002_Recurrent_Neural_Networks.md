# Recurrent Neural Networks

## Table of Contents

1. [Introduction](#introduction)
2. [Vanilla RNN Architecture](#vanilla-rnn-architecture)
3. [Backpropagation Through Time](#backpropagation-through-time)
4. [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
5. [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
6. [Bidirectional RNNs](#bidirectional-rnns)
7. [Sequence Modeling Applications](#sequence-modeling-applications)
8. [Training Challenges](#training-challenges)
9. [Modern RNN Variants](#modern-rnn-variants)
10. [Key Takeaways](#key-takeaways)

## Introduction

Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining hidden states that capture information from previous time steps. Unlike feedforward networks, RNNs have connections that form cycles, allowing them to exhibit dynamic temporal behavior and model dependencies across time.

This chapter covers the architecture and training of RNNs, from basic vanilla RNNs to sophisticated gated architectures like LSTM and GRU, examining how they handle sequential patterns and the challenges of training recurrent networks.

## Vanilla RNN Architecture

### Basic Structure

A vanilla RNN processes sequences by maintaining a hidden state:

$$\mathbf{h}_t = \phi(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$$

$$\mathbf{y}_t = W_y \mathbf{h}_t + \mathbf{b}_y$$

where:
- $\mathbf{x}_t$ is input at time $t$
- $\mathbf{h}_t$ is hidden state at time $t$
- $\mathbf{y}_t$ is output at time $t$
- $\phi$ is activation function (typically tanh)

### Unfolding Through Time

An RNN can be unfolded into a feedforward network:

```
h_0 → [RNN] → h_1 → [RNN] → h_2 → ... → h_T
       ↑       ↑       ↑              ↑
      x_1     x_2     x_3           x_T
```

This reveals the temporal dependencies and enables backpropagation.

### Mathematical Formulation

For a sequence of length $T$:

**Forward Pass**:
- $\mathbf{h}_0 = \mathbf{0}$ (initial hidden state)
- For $t = 1, \ldots, T$:
  - $\mathbf{h}_t = \tanh(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$
  - $\mathbf{y}_t = W_y \mathbf{h}_t + \mathbf{b}_y$

**Parameters**:
- $W_h$: Hidden-to-hidden weights
- $W_x$: Input-to-hidden weights
- $W_y$: Hidden-to-output weights
- $\mathbf{b}, \mathbf{b}_y$: Bias vectors

### Computational Graph

The computational graph shows dependencies:
- Each time step depends on previous hidden state
- Gradients flow backward through time
- Parameters are shared across time steps

## Backpropagation Through Time

Backpropagation Through Time (BPTT) extends backpropagation to RNNs by unrolling the network through time.

### Gradient Computation

The gradient with respect to parameters involves summing over all time steps:

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial W_h}$$

### Error Signal Propagation

Error signals propagate backward through time:

$$\boldsymbol{\delta}_t = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = W_y^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}_t} + W_h^T \boldsymbol{\delta}_{t+1} \odot \phi'(\mathbf{z}_t)$$

where $\mathbf{z}_t = W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b}$.

### Vanishing Gradient Problem

In RNNs, gradients are multiplied repeatedly:

$$\boldsymbol{\delta}_t = W_h^T \boldsymbol{\delta}_{t+1} \odot \phi'(\mathbf{z}_t)$$

If $||W_h|| < 1$ and $\phi'(\mathbf{z}_t) < 1$, gradients vanish exponentially:

$$||\boldsymbol{\delta}_t|| \approx ||W_h||^{T-t} ||\boldsymbol{\delta}_T||$$

This makes it difficult to learn long-term dependencies.

### Truncated BPTT

For long sequences, BPTT is truncated:
- Process sequence in chunks
- Backpropagate only within chunk
- Reduces memory and computation
- Still enables learning dependencies

### Gradient Clipping

Essential for RNNs to prevent exploding gradients:

$$\mathbf{g}_{\text{clipped}} = \begin{cases}
\mathbf{g} & \text{if } ||\mathbf{g}|| \leq \text{max\_norm} \\
\mathbf{g} \cdot \frac{\text{max\_norm}}{||\mathbf{g}||} & \text{otherwise}
\end{cases}$$

## Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem through gating mechanisms.

### LSTM Cell Structure

An LSTM cell maintains:
- **Cell State** $\mathbf{c}_t$: Long-term memory
- **Hidden State** $\mathbf{h}_t$: Short-term memory/output

### Gates

**Forget Gate**: Decides what to forget from cell state

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate**: Decides what new information to store

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Output Gate**: Decides what parts of cell state to output

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

### Cell State Update

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Hidden State**:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

### Why LSTM Works

1. **Cell State Highway**: Cell state provides direct gradient path
2. **Gating**: Gates control information flow
3. **Additive Updates**: Cell state uses addition, not multiplication
4. **Gradient Flow**: Gradients can flow through cell state without vanishing

### Variants

**Peephole Connections**: Gates can see cell state

**Coupled Gates**: Combine forget and input gates

**GRU**: Simplified version (see next section)

## Gated Recurrent Unit (GRU)

GRU is a simplified LSTM with fewer parameters.

### GRU Structure

**Reset Gate**: Controls how much past information to forget

$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

**Update Gate**: Controls how much past information to keep

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

**Candidate Activation**:

$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden State Update**:

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### Comparison with LSTM

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell State | Yes | No |
| Parameters | More | Fewer |
| Performance | Often better | Similar, sometimes better |
| Speed | Slower | Faster |

### When to Use

- **LSTM**: When long-term dependencies are critical
- **GRU**: When speed/parameters matter, or when performance is similar

## Bidirectional RNNs

Bidirectional RNNs process sequences in both directions.

### Architecture

Two RNNs:
- **Forward RNN**: Processes sequence left-to-right
- **Backward RNN**: Processes sequence right-to-left

**Hidden States**:

$$\overrightarrow{\mathbf{h}}_t = \text{RNN}_{\text{forward}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = \text{RNN}_{\text{backward}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

**Combined**:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

### Applications

- **Sequence Labeling**: POS tagging, NER
- **Machine Translation**: Encoder-decoder architectures
- **Speech Recognition**: Context from both directions

### Limitations

- Requires full sequence (not online)
- More parameters and computation
- Cannot be used for generation (causal)

## Sequence Modeling Applications

### Language Modeling

Predict next word given previous words:

$$P(w_t | w_{t-1}, w_{t-2}, \ldots, w_1)$$

**Training**: Maximize likelihood of training sequences

**Evaluation**: Perplexity

$$PP = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{<t})\right)$$

### Machine Translation

Encoder-decoder architecture:
- **Encoder RNN**: Encodes source sentence
- **Decoder RNN**: Generates target sentence

**Attention**: Allows decoder to attend to encoder states

### Sequence Classification

Classify entire sequence:
- Process sequence with RNN
- Use final hidden state for classification
- Or use attention over all states

### Sequence Tagging

Label each element:
- Each time step produces a label
- Examples: POS tagging, NER, chunking

## Training Challenges

### Vanishing Gradients

**Problem**: Gradients vanish through time

**Solutions**:
- LSTM/GRU gates
- Gradient clipping
- Proper initialization
- Skip connections

### Exploding Gradients

**Problem**: Gradients explode

**Solutions**:
- Gradient clipping (essential)
- Proper initialization
- Smaller learning rates

### Long Sequences

**Challenges**:
- Memory: Storing all activations
- Computation: BPTT over long sequences
- Dependencies: Learning long-term patterns

**Solutions**:
- Truncated BPTT
- Hierarchical RNNs
- Attention mechanisms

### Overfitting

**Problem**: RNNs can overfit to training sequences

**Solutions**:
- Dropout (variational dropout for RNNs)
- Weight regularization
- Early stopping
- Data augmentation

## Modern RNN Variants

### Attention Mechanisms

Attention allows focusing on relevant parts of input:

$$\alpha_t = \text{softmax}(\text{score}(\mathbf{h}_t, \mathbf{s}))$$

$$\mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{h}_i$$

Used in encoder-decoder architectures.

### Transformer Architecture

Replaces RNNs entirely with self-attention:
- No recurrence
- Parallel processing
- Better long-range dependencies

### Neural Turing Machines

RNNs with external memory:
- Read/write operations
- Differentiable memory access
- Can learn algorithms

### Differentiable Neural Computers

Extension of NTM with improved memory mechanisms.

## Key Takeaways

1. **Recurrent Architecture**: RNNs process sequences by maintaining hidden states that capture temporal dependencies, enabling modeling of sequential patterns.

2. **Backpropagation Through Time**: Extends backpropagation to RNNs by unrolling through time, but suffers from vanishing/exploding gradients in long sequences.

3. **Vanishing Gradients**: The fundamental challenge of RNNs, where gradients vanish exponentially through time, making it difficult to learn long-term dependencies.

4. **LSTM**: Addresses vanishing gradients through gating mechanisms (forget, input, output gates) and a cell state that provides a direct gradient path.

5. **GRU**: Simplified LSTM with fewer parameters, combining forget and input gates into an update gate, often achieving similar performance with better efficiency.

6. **Bidirectional RNNs**: Process sequences in both directions, providing richer context but requiring full sequences and more computation.

7. **Sequence Applications**: RNNs excel at language modeling, machine translation, sequence classification, and sequence tagging tasks.

8. **Training Challenges**: Vanishing/exploding gradients, long sequences, and overfitting require careful techniques like gradient clipping, truncated BPTT, and regularization.

9. **Modern Variants**: Attention mechanisms and Transformer architectures have largely replaced RNNs in many applications, though RNNs remain useful for certain tasks.

10. **Practical Considerations**: Gradient clipping is essential, proper initialization matters, and architectural choices (LSTM vs. GRU) depend on specific requirements and empirical performance.
