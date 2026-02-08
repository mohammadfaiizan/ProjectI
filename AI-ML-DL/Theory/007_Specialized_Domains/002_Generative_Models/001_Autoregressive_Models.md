# Autoregressive Models

## Table of Contents

1. [Introduction to Autoregressive Modeling](#introduction-to-autoregressive-modeling)
2. [PixelRNN and PixelCNN](#pixelrnn-and-pixelcnn)
3. [WaveNet Architecture](#wavenet-architecture)
4. [Autoregressive Language Models](#autoregressive-language-models)
5. [Masked Autoregressive Flows](#masked-autoregressive-flows)
6. [Tractable Density Estimation](#tractable-density-estimation)
7. [Training and Optimization](#training-and-optimization)
8. [Applications and Extensions](#applications-and-extensions)
9. [Limitations and Challenges](#limitations-and-challenges)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Autoregressive Modeling

Autoregressive models generate data by modeling the conditional distribution of each element given all previous elements. This approach provides tractable likelihood estimation and enables high-quality generation through sequential sampling.

### Autoregressive Principle

For a sequence $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, the joint distribution factorizes as:

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i | x_{<i})$$

where $x_{<i} = (x_1, \ldots, x_{i-1})$ denotes all previous elements.

### Advantages

1. **Tractable Likelihood**: Exact likelihood computation enables training via maximum likelihood
2. **No Latent Variables**: Direct modeling avoids inference complexity
3. **High Quality**: Can generate high-fidelity samples
4. **Flexible**: Applicable to images, audio, text, and other sequential data

### Disadvantages

1. **Sequential Generation**: Cannot parallelize generation (slow)
2. **Ordering Sensitivity**: Generation order affects quality
3. **Long Dependencies**: Difficulty capturing long-range dependencies

### Applications

- **Image Generation**: PixelRNN, PixelCNN
- **Audio Generation**: WaveNet, SampleRNN
- **Language Modeling**: GPT, Transformer-XL
- **Density Estimation**: Normalizing flows, autoregressive flows

---

## PixelRNN and PixelCNN

PixelRNN and PixelCNN generate images pixel by pixel, modeling the conditional distribution of each pixel given previous pixels.

### PixelRNN

PixelRNN uses LSTM or GRU to model dependencies:

$$p(x_{i,j} | x_{<i,j}) = \text{softmax}(\text{LSTM}(h_{i,j-1}, x_{i,j-1}))$$

where $h_{i,j-1}$ is the hidden state and generation proceeds row by row.

**Row LSTM**: Processes row by row:
$$h_{i,j} = \text{LSTM}(h_{i,j-1}, x_{i,j-1})$$

**Diagonal BiLSTM**: Processes along diagonals for better context.

### PixelCNN

PixelCNN uses masked convolutions to ensure causal dependencies:

**Masked Convolution**: Zero out future pixels:

$$M_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{otherwise}
\end{cases}$$

**Standard Mask**: For RGB images, mask current pixel's green and blue channels:

$$M = \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}$$

### PixelCNN Architecture

**Residual Blocks**: Use residual connections:

$$h^{(l+1)} = h^{(l)} + \sigma(\text{MaskedConv}(h^{(l)}))$$

**Gated Activation**: Use gated activation units:

$$h = \tanh(W_{f} * x) \odot \sigma(W_{g} * x)$$

where $\odot$ is element-wise multiplication.

### PixelCNN++ Improvements

1. **Discretized Logistic Mixture**: Model pixel values as mixture of logistics
2. **Shortcut Connections**: Add skip connections
3. **Dropout**: Regularize with dropout
4. **Multi-Scale**: Use multiple resolutions

### Conditional PixelCNN

Condition on class labels or other information:

$$p(x_{i,j} | x_{<i,j}, y) = \text{softmax}(\text{MaskedConv}(x_{<i,j}, y))$$

---

## WaveNet Architecture

WaveNet generates raw audio waveforms by modeling the conditional distribution of each audio sample.

### Autoregressive Audio Generation

For audio waveform $\mathbf{x} = (x_1, \ldots, x_T)$:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t | x_{<t})$$

### Dilated Causal Convolutions

WaveNet uses dilated convolutions to capture long-range dependencies:

**Causal Convolution**: Only depends on past samples:
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

**Dilated Convolution**: Skip samples to increase receptive field:
$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-d \cdot k}$$

where $d$ is the dilation rate.

**Exponential Dilation**: Use dilation rates $1, 2, 4, 8, \ldots$ to exponentially increase receptive field.

### WaveNet Block

Each block consists of:

1. **Dilated Causal Convolution**: Extract features
2. **Gated Activation**: 
   $$z = \tanh(W_{f} * x) \odot \sigma(W_{g} * x)$$
3. **Residual Connection**: $h = h + z$
4. **Skip Connection**: Accumulate for final output

### Multi-Scale Architecture

Stack multiple blocks with increasing dilation:

```
Block 1: dilation = 1, 2, 4, ..., 512
Block 2: dilation = 1, 2, 4, ..., 512
Block 3: dilation = 1, 2, 4, ..., 512
```

**Receptive Field**: With $L$ blocks and $K$ filters per block:
$$\text{Receptive Field} = 2^L \cdot K$$

### Quantized Output

Model audio as categorical distribution over quantized values:

$$p(x_t | x_{<t}) = \text{softmax}(\text{WaveNet}(x_{<t}))$$

Use $\mu$-law companding for quantization:
$$\mu(x) = \text{sign}(x) \frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)}$$

### Conditional WaveNet

Condition on speaker identity, text, or other features:

$$p(x_t | x_{<t}, c) = \text{softmax}(\text{WaveNet}(x_{<t}, c))$$

---

## Autoregressive Language Models

Autoregressive language models generate text by predicting the next token given previous tokens.

### GPT Architecture

GPT (Generative Pre-trained Transformer) uses Transformer decoder:

$$p(x_t | x_{<t}) = \text{softmax}(W \cdot \text{Transformer}(x_{<t}))$$

**Masked Self-Attention**: Prevent attending to future tokens:

$$A_{ij} = \begin{cases}
\frac{Q_i K_j^T}{\sqrt{d_k}} & \text{if } j \leq i \\
-\infty & \text{otherwise}
\end{cases}$$

**Positional Encoding**: Add positional embeddings to token embeddings.

### GPT Training

**Pre-training**: Maximize likelihood on large corpus:

$$\mathcal{L} = -\sum_{t=1}^{T} \log p(x_t | x_{<t})$$

**Fine-tuning**: Adapt to downstream tasks with task-specific heads.

### GPT-2 and GPT-3

**GPT-2**: Larger model (1.5B parameters), better generation quality.

**GPT-3**: Massive scale (175B parameters), few-shot learning capabilities.

### Transformer-XL

Addresses fixed context length limitation:

**Segment-Level Recurrence**: Reuse hidden states from previous segments:

$$h_{\tau+1} = f(h_\tau, s_{\tau+1})$$

**Relative Positional Encoding**: Encode relative positions instead of absolute.

### XLNet

Permutation-based autoregressive training:

**Permutation Language Modeling**: Train on all permutations:

$$\max_\theta \mathbb{E}_{\mathbf{z} \sim \mathcal{Z}_T} \left[\sum_{t=1}^{T} \log p_\theta(x_{z_t} | \mathbf{x}_{\mathbf{z}_{<t}})\right]$$

**Two-Stream Attention**: Query stream and content stream for proper conditioning.

---

## Masked Autoregressive Flows

Masked Autoregressive Flows (MAF) combine autoregressive models with normalizing flows for flexible density estimation.

### Autoregressive Flow

Transform a simple distribution $p_Z(z)$ to complex $p_X(x)$:

$$x_i = \tau(z_i; \mathbf{x}_{<i})$$

where $\tau$ is an invertible transformation parameterized by previous values.

### Masked Autoregressive Flow

Use masked neural networks to ensure autoregressive structure:

**Coupling Layer**: Split dimensions and transform:

$$x_{1:d} = z_{1:d}$$
$$x_{d+1:D} = \tau(z_{d+1:D}; \text{NN}(z_{1:d}))$$

**Autoregressive Masking**: Mask connections to ensure $x_i$ only depends on $x_{<i}$:

$$M_{ij} = \begin{cases}
1 & \text{if } j < i \\
0 & \text{otherwise}
\end{cases}$$

### MAF Architecture

**MADE (Masked Autoencoder for Distribution Estimation)**: Use masked autoencoders:

$$h^{(l+1)} = \sigma(M^{(l)} \cdot (W^{(l)} h^{(l)} + b^{(l)}))$$

where $M^{(l)}$ ensures autoregressive structure.

**MAF Layer**: Combine MADE with affine transformation:

$$x_i = z_i \cdot \exp(s_i(\mathbf{x}_{<i})) + t_i(\mathbf{x}_{<i})$$

where $s_i$ and $t_i$ are outputs of MADE.

### Inverse Autoregressive Flow (IAF)

Reverse the transformation for faster sampling:

$$z_i = \frac{x_i - t_i(\mathbf{z}_{<i})}{\exp(s_i(\mathbf{z}_{<i}))}$$

IAF enables fast sampling but slow density evaluation.

### Real NVP

Use coupling layers with checkerboard or channel-wise masking:

$$x_{1:d} = z_{1:d}$$
$$x_{d+1:D} = z_{d+1:D} \odot \exp(s(z_{1:d})) + t(z_{1:d})$$

---

## Tractable Density Estimation

Autoregressive models provide exact likelihood computation, enabling tractable density estimation.

### Likelihood Computation

For autoregressive model:

$$p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i | x_{<i})$$

**Log-Likelihood**:
$$\log p(\mathbf{x}) = \sum_{i=1}^{n} \log p(x_i | x_{<i})$$

### Continuous Variables

For continuous variables, model as conditional density:

$$p(x_i | x_{<i}) = \mathcal{N}(x_i; \mu_i(x_{<i}), \sigma_i^2(x_{<i}))$$

or use mixture models:

$$p(x_i | x_{<i}) = \sum_{k=1}^{K} \pi_k(x_{<i}) \mathcal{N}(x_i; \mu_{ik}(x_{<i}), \sigma_{ik}^2(x_{<i}))$$

### Discrete Variables

For discrete variables (e.g., pixels, tokens):

$$p(x_i | x_{<i}) = \text{Categorical}(x_i; \pi(x_{<i}))$$

where $\pi(x_{<i})$ is output of softmax.

### Evaluation Metrics

**Bits per Dimension (BPD)**:
$$\text{BPD} = -\frac{1}{n} \sum_{i=1}^{n} \log_2 p(x_i | x_{<i})$$

**Perplexity** (for language models):
$$\text{Perplexity} = \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log p(x_i | x_{<i})\right)$$

### Comparison with Other Models

| Model | Likelihood | Sampling Speed | Quality |
|-------|------------|----------------|---------|
| Autoregressive | Exact | Slow (sequential) | High |
| VAE | Lower bound | Fast | Medium |
| GAN | None | Fast | High |
| Flow | Exact | Fast (parallel) | High |

---

## Training and Optimization

### Maximum Likelihood Training

Maximize log-likelihood:

$$\max_\theta \sum_{i=1}^{N} \log p_\theta(\mathbf{x}^{(i)}) = \sum_{i=1}^{N} \sum_{j=1}^{n} \log p_\theta(x_j^{(i)} | x_{<j}^{(i)})$$

### Teacher Forcing

During training, use ground truth previous tokens:

$$x_t \sim p_\theta(\cdot | x_{1:t-1}^{\text{true}})$$

During inference, use generated tokens:

$$x_t \sim p_\theta(\cdot | x_{1:t-1}^{\text{generated}})$$

### Scheduled Sampling

Gradually transition from teacher forcing to using generated tokens:

$$x_t \sim \begin{cases}
p_\theta(\cdot | x_{1:t-1}^{\text{true}}) & \text{with probability } \epsilon \\
p_\theta(\cdot | x_{1:t-1}^{\text{generated}}) & \text{with probability } 1-\epsilon
\end{cases}$$

where $\epsilon$ decays during training.

### Mixed Precision Training

Use FP16 for faster training while maintaining stability:

- Store master weights in FP32
- Compute gradients in FP16
- Update in FP32

### Gradient Clipping

Prevent exploding gradients:

$$g \leftarrow \min\left(1, \frac{\tau}{\|g\|}\right) \cdot g$$

### Learning Rate Scheduling

**Warmup**: Gradually increase learning rate:
$$\eta_t = \min\left(\frac{t}{T_{\text{warmup}}}, 1\right) \cdot \eta_{\text{max}}$$

**Cosine Annealing**: Decrease learning rate:
$$\eta_t = \eta_{\text{min}} + (\eta_{\text{max}} - \eta_{\text{min}}) \cdot \frac{1 + \cos(\pi t / T)}{2}$$

---

## Applications and Extensions

### Image Generation

**PixelRNN/PixelCNN**: Generate high-quality images pixel by pixel.

**ImageGPT**: Apply GPT architecture to images by treating pixels as tokens.

### Audio Generation

**WaveNet**: Generate realistic speech and music.

**SampleRNN**: Hierarchical RNN for audio generation.

**Jukebox**: Large-scale music generation with VQ-VAE.

### Text Generation

**GPT Models**: Language modeling and text generation.

**Code Generation**: Generate code from natural language.

**Dialogue Systems**: Conversational AI with autoregressive models.

### Conditional Generation

**Class-Conditional**: Generate samples from specific classes.

**Text-to-Image**: Generate images from text descriptions.

**Text-to-Speech**: Synthesize speech from text.

### Few-Shot Learning

**In-Context Learning**: GPT-3 demonstrates few-shot capabilities through prompting.

**Meta-Learning**: Learn to generate with few examples.

---

## Limitations and Challenges

### Sequential Generation

**Slow Sampling**: Cannot parallelize generation, limiting real-time applications.

**Solutions**: 
- Use parallel decoding (non-autoregressive models)
- Optimize with caching and quantization
- Use distillation to smaller models

### Ordering Sensitivity

**Dependency on Order**: Generation quality depends on chosen ordering.

**Solutions**:
- Try multiple orderings
- Use learned orderings
- Use permutation-based training (XLNet)

### Long-Range Dependencies

**Limited Context**: Difficulty capturing very long dependencies.

**Solutions**:
- Use attention mechanisms (Transformer)
- Use recurrence (Transformer-XL)
- Increase model capacity

### Evaluation Challenges

**Likelihood vs Quality**: High likelihood doesn't always mean high quality.

**Diversity**: Autoregressive models may generate repetitive samples.

**Solutions**:
- Use multiple metrics (FID, IS, human evaluation)
- Use diverse decoding strategies (nucleus sampling, top-k)
- Regularize to encourage diversity

### Computational Cost

**Large Models**: Training large autoregressive models is expensive.

**Solutions**:
- Model parallelism
- Gradient checkpointing
- Mixed precision training
- Efficient attention (sparse, linear)

---

## Key Takeaways

1. **Autoregressive Principle**: Model joint distribution as product of conditionals $p(\mathbf{x}) = \prod_i p(x_i | x_{<i})$, enabling tractable likelihood computation.

2. **PixelRNN/PixelCNN**: Generate images pixel by pixel using RNNs or masked convolutions, achieving high-quality image generation with exact likelihood.

3. **WaveNet**: Uses dilated causal convolutions to generate raw audio waveforms, capturing long-range dependencies through exponential dilation.

4. **GPT Architecture**: Transformer-based autoregressive language models achieve state-of-the-art text generation through masked self-attention and large-scale pre-training.

5. **Masked Autoregressive Flows**: Combine autoregressive structure with normalizing flows for flexible density estimation, using masked neural networks to ensure causality.

6. **Tractable Likelihood**: Autoregressive models provide exact likelihood computation, enabling density estimation, anomaly detection, and likelihood-based evaluation.

7. **Training Challenges**: Teacher forcing, scheduled sampling, and gradient clipping address training-inference mismatch and optimization difficulties.

8. **Applications**: Autoregressive models excel at image generation (PixelCNN), audio synthesis (WaveNet), and language modeling (GPT), with extensions to conditional and few-shot generation.

9. **Limitations**: Sequential generation is slow, ordering sensitivity affects quality, and long-range dependencies remain challenging despite attention mechanisms.

10. **Future Directions**: Parallel decoding, learned orderings, efficient attention, and better evaluation metrics continue to advance autoregressive modeling capabilities.
