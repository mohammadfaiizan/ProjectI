# Diffusion Models and Score Matching

## Table of Contents

1. [Introduction](#introduction)
2. [Denoising Diffusion Probabilistic Models](#denoising-diffusion-probabilistic-models)
3. [Forward and Reverse Diffusion Processes](#forward-and-reverse-diffusion-processes)
4. [Noise Scheduling Strategies](#noise-scheduling-strategies)
5. [Score Matching and Langevin Dynamics](#score-matching-and-langevin-dynamics)
6. [Classifier Guidance and Classifier-Free Guidance](#classifier-guidance-and-classifier-free-guidance)
7. [DDIM: Denoising Diffusion Implicit Models](#ddim-denoising-diffusion-implicit-models)
8. [Latent Diffusion Models](#latent-diffusion-models)
9. [Training and Sampling Procedures](#training-and-sampling-procedures)
10. [Key Takeaways](#key-takeaways)

## Introduction

Diffusion models have emerged as a powerful class of generative models that achieve state-of-the-art results in image generation, audio synthesis, and other domains. Unlike Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs), diffusion models work by learning to reverse a gradual noising process, transforming random noise into structured data.

**Key advantages:**
- **Stable training**: No adversarial training required
- **High quality**: Produce high-fidelity samples
- **Flexible**: Can be conditioned on various inputs
- **Theoretically grounded**: Well-understood probabilistic framework

**Core idea:**
Learn to reverse a forward diffusion process that gradually adds noise to data until it becomes pure noise. The reverse process then generates new samples by iteratively denoising.

## Denoising Diffusion Probabilistic Models

Denoising Diffusion Probabilistic Models (DDPM) provide a probabilistic framework for diffusion-based generation.

### Forward Diffusion Process

Given data distribution $q(x_0)$, the forward process gradually adds Gaussian noise:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

where $\beta_t$ is a noise schedule with $0 < \beta_t < 1$.

**Properties:**
- $x_t$ becomes more noisy as $t$ increases
- $x_T \approx \mathcal{N}(0, I)$ for large $T$
- Process is Markovian: $q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$

### Reparameterization

Using reparameterization, we can sample $x_t$ directly from $x_0$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
- $\epsilon \sim \mathcal{N}(0, I)$

This allows efficient sampling at any timestep $t$ without iterating through all previous steps.

### Reverse Diffusion Process

The reverse process learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

where $\mu_\theta$ and $\Sigma_\theta$ are neural networks.

**Goal:** Learn $p_\theta(x_{t-1} | x_t)$ to approximate true reverse $q(x_{t-1} | x_t)$.

### Training Objective

The training loss is derived from variational lower bound:

$$L = \mathbb{E}_q \left[ -\log p_\theta(x_0 | x_1) + \sum_{t=2}^T D_{KL}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t)) + D_{KL}(q(x_T | x_0) \| p(x_T)) \right]$$

**Simplified form:**
After simplification, the loss becomes:

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

where:
- $t \sim \text{Uniform}(1, T)$
- $x_0 \sim q(x_0)$
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
- $\epsilon_\theta(x_t, t)$ predicts the noise $\epsilon$

**Key insight:** Instead of predicting $x_{t-1}$ directly, predict the noise $\epsilon$ added to $x_0$ to get $x_t$.

### Sampling Procedure

```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1, else z = 0
    ε_t = ε_θ(x_t, t)
    x_{t-1} = (1/√(α_t)) (x_t - (β_t/√(1-ᾱ_t)) ε_t) + √(β_t) z
return x_0
```

## Forward and Reverse Diffusion Processes

### Forward Process Details

The forward process transforms data $x_0$ into noise $x_T$:

$$q(x_{1:T} | x_0) = \prod_{t=1}^T q(x_t | x_{t-1})$$

where:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

**Marginal distribution:**
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)$$

**Variance schedule:**
- Early steps: Small $\beta_t$, preserve data structure
- Later steps: Larger $\beta_t$, more noise
- Final step: $x_T \approx \mathcal{N}(0, I)$

### Reverse Process Details

The true reverse process (if we knew $q(x_0)$) would be:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

where:
$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

**Learned reverse process:**
We approximate this with:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

**Parameterization choices:**
- **Mean**: $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$
- **Variance**: Often fixed to $\Sigma_\theta(x_t, t) = \sigma_t^2 I$ where $\sigma_t^2 = \beta_t$ or $\tilde{\beta}_t$

### Connection to Score Matching

The score function is:
$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

Predicting $\epsilon$ is equivalent to predicting the score (up to scaling).

## Noise Scheduling Strategies

The noise schedule $\{\beta_t\}_{t=1}^T$ determines how noise is added over time.

### Linear Schedule

$$\beta_t = \frac{t}{T} \beta_{\max} + \left(1 - \frac{t}{T}\right) \beta_{\min}$$

**Properties:**
- Simple and interpretable
- Linear increase in noise
- May be suboptimal for some domains

### Cosine Schedule

$$\bar{\alpha}_t = \frac{\cos(\pi t / 2T + s)}{1 + s}$$

where $s$ is a small offset (e.g., $s = 0.008$).

**Properties:**
- Slower noise addition at beginning and end
- Faster in middle
- Often performs better than linear

### Quadratic Schedule

$$\beta_t = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \left(\frac{t}{T}\right)^2$$

**Properties:**
- Accelerating noise addition
- Useful for certain data types

### Learned Schedule

Learn $\{\beta_t\}$ as parameters:
$$\beta_t = \text{softplus}(\beta_t^{\text{raw}})$$

Optimize jointly with model parameters.

### Schedule Selection

**Considerations:**
- **Data type**: Images vs audio vs text may need different schedules
- **Number of steps**: More steps allow finer control
- **Computational budget**: Affects training and sampling time

**Typical values:**
- $T = 1000$ steps
- $\beta_{\min} = 0.0001$, $\beta_{\max} = 0.02$ (linear)
- Or cosine schedule with $s = 0.008$

## Score Matching and Langevin Dynamics

Score matching provides an alternative perspective on diffusion models.

### Score Function

The score function is the gradient of the log-density:
$$s_\theta(x) = \nabla_x \log p_\theta(x)$$

**Properties:**
- Points toward higher density regions
- Unnormalized (doesn't require partition function)
- Useful for sampling via Langevin dynamics

### Score Matching Objective

**Denoising score matching:**
$$L = \mathbb{E}_{q_\sigma(\tilde{x} | x) q(x)} \left[ \| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} | x) \|^2 \right]$$

where $q_\sigma(\tilde{x} | x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$ adds noise.

**Sliced score matching:**
Uses random projections to reduce computational cost.

### Langevin Dynamics

Langevin dynamics uses the score function for sampling:

$$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t$$

where $z_t \sim \mathcal{N}(0, I)$ and $\epsilon$ is step size.

**Properties:**
- Converges to $p(x)$ under mild conditions
- Requires many steps
- Sensitive to step size

### Connection to Diffusion

Diffusion models can be viewed as:
1. Learning score function at multiple noise levels
2. Using annealed Langevin dynamics for sampling

**Noise-conditioned score network:**
$$s_\theta(x, \sigma) = \nabla_x \log p_\sigma(x)$$

where $\sigma$ indexes noise level.

**Sampling:**
Use Langevin dynamics with decreasing noise levels (annealing).

## Classifier Guidance and Classifier-Free Guidance

Conditional generation enables control over generated samples.

### Classifier Guidance

**Setup:**
- Train unconditional diffusion model $p_\theta(x)$
- Train classifier $p_\phi(y | x)$ on noisy images
- Use classifier gradient to guide generation

**Modified reverse process:**
$$\tilde{\epsilon}_\theta(x_t, t) = \epsilon_\theta(x_t, t) - s \cdot \sigma_t \nabla_{x_t} \log p_\phi(y | x_t)$$

where $s$ is guidance scale.

**Intuition:**
- Gradient points toward higher $p(y | x)$
- Stronger guidance ($s > 1$) increases conditioning strength
- May reduce sample diversity

### Classifier-Free Guidance

**Setup:**
- Train conditional model $p_\theta(x | y)$
- Train unconditional model $p_\theta(x)$ (or use $p_\theta(x | y=\emptyset)$)
- Combine during sampling

**Guided prediction:**
$$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$

**Properties:**
- No separate classifier needed
- Single model handles both conditional and unconditional
- Often better quality than classifier guidance
- $s = 1$: Standard conditional generation
- $s > 1$: Stronger conditioning
- $s < 1$: Weaker conditioning

### Training for Classifier-Free Guidance

**Random conditioning:**
During training, randomly set $y = \emptyset$ with probability $p_{\text{uncond}}$ (e.g., 0.1-0.2):

```python
if random() < p_uncond:
    y = None  # Unconditional
else:
    y = actual_condition  # Conditional
```

Model learns to generate both conditional and unconditional samples.

## DDIM: Denoising Diffusion Implicit Models

DDIM enables deterministic sampling and faster generation.

### Motivation

DDPM sampling is stochastic and requires many steps. DDIM provides:
- **Deterministic sampling**: Same noise → same sample
- **Fewer steps**: Can use fewer than $T$ steps
- **Inversion**: Can encode images to noise

### DDIM Formulation

**Deterministic reverse process:**
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0(x_t) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)$$

where:
$$\hat{x}_0(x_t) = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Key difference from DDPM:**
- No stochastic noise term (except at $t=1$)
- Deterministic mapping from $x_t$ to $x_{t-1}$

### Subsequence Sampling

Can use subsequence $\tau = \{\tau_1, \ldots, \tau_S\}$ where $S < T$:

```
x_{τ_S} ~ N(0, I)  # or encode from x_0
for i = S-1, ..., 1:
    x_{τ_i} = deterministic_update(x_{τ_{i+1}}, τ_{i+1})
```

**Speed-quality trade-off:**
- Fewer steps: Faster but potentially lower quality
- More steps: Slower but higher quality
- Typical: 50-200 steps instead of 1000

### Image Inversion

DDIM can encode images to noise:

```
x_0 = input_image
for t = 1, ..., T:
    ε_t = ε_θ(x_{t-1}, t-1)
    x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε_t
return x_T
```

Enables:
- Image editing via manipulation in noise space
- Interpolation between images
- Style transfer

## Latent Diffusion Models

Latent diffusion models operate in a learned latent space, reducing computational cost.

### Motivation

**Challenges with pixel-space diffusion:**
- High-dimensional (e.g., $256 \times 256 \times 3 = 196,608$ dimensions)
- Computationally expensive
- Slow sampling

**Solution:**
- Encode images to lower-dimensional latent space
- Apply diffusion in latent space
- Decode to pixel space

### Architecture

**Components:**

1. **Encoder**: $E: \mathcal{X} \to \mathcal{Z}$
   - Encodes image $x$ to latent $z = E(x)$
   - Typically VAE encoder or VQ-VAE

2. **Diffusion in latent space:**
   - Forward: $q(z_t | z_{t-1})$
   - Reverse: $p_\theta(z_{t-1} | z_t)$

3. **Decoder**: $D: \mathcal{Z} \to \mathcal{X}$
   - Decodes latent $z$ to image $\hat{x} = D(z)$

### Stable Diffusion

Stable Diffusion is a prominent latent diffusion model:

**Architecture:**
- **VAE**: Encoder/decoder for images
- **U-Net**: Denoising network in latent space
- **Text encoder**: CLIP text encoder for conditioning

**Latent space:**
- $64 \times 64 \times 4$ latents (for $512 \times 512$ images)
- $64^2 \times 4 = 16,384$ dimensions vs $512^2 \times 3 = 786,432$ pixels
- ~48× reduction in dimensionality

**Training:**
1. Train VAE encoder/decoder
2. Train diffusion model in latent space with text conditioning
3. Use classifier-free guidance for text-to-image

### Advantages

**Computational efficiency:**
- Faster training and sampling
- Lower memory requirements
- Enables higher resolution generation

**Quality:**
- Latent space may capture semantic structure better
- VAE ensures high-fidelity reconstruction

**Flexibility:**
- Can condition on various modalities (text, images, etc.)
- Enables various downstream tasks

## Training and Sampling Procedures

### Training Procedure

```
Initialize diffusion model ε_θ
for iteration = 1 to N:
    # Sample data
    x_0 ~ q(x_0)
    
    # Sample timestep
    t ~ Uniform(1, T)
    
    # Sample noise
    ε ~ N(0, I)
    
    # Compute noisy sample
    x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
    
    # Predict noise
    ε_pred = ε_θ(x_t, t)
    
    # Compute loss
    L = ||ε - ε_pred||²
    
    # Update parameters
    θ ← θ - α ∇_θ L
```

**Key points:**
- Random timestep sampling
- Mean squared error loss
- Can use various optimizers (Adam, AdamW)

### Sampling Procedure (DDPM)

```
# Start from noise
x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # Predict noise
    ε_t = ε_θ(x_t, t)
    
    # Predict x_0 (for variance computation)
    x_0_pred = (x_t - √(1-ᾱ_t) ε_t) / √(ᾱ_t)
    
    # Compute posterior mean
    μ_t = (1/√(α_t)) (x_t - (β_t/√(1-ᾱ_t)) ε_t)
    
    # Sample (add noise if t > 1)
    if t > 1:
        z ~ N(0, I)
        x_{t-1} = μ_t + √(β_t) z
    else:
        x_0 = μ_t

return x_0
```

### Sampling Procedure (DDIM)

```
# Start from noise (or encode image)
x_T ~ N(0, I)  # or encode x_0

# Define subsequence (optional)
τ = {τ_1, ..., τ_S}  # e.g., every 10th step

for i = S, S-1, ..., 1:
    t = τ_i
    
    # Predict noise
    ε_t = ε_θ(x_t, t)
    
    # Predict x_0
    x_0_pred = (x_t - √(1-ᾱ_t) ε_t) / √(ᾱ_t)
    
    # Deterministic update
    if i > 1:
        t_prev = τ_{i-1}
        x_{t_prev} = √(ᾱ_{t_prev}) x_0_pred + √(1-ᾱ_{t_prev}) ε_t
    else:
        x_0 = x_0_pred

return x_0
```

### Practical Considerations

**Number of steps:**
- Training: $T = 1000$ typical
- Sampling: Can use fewer with DDIM (50-200)

**Guidance scale:**
- Classifier-free: $s = 1.5$ to $7.5$ typical
- Higher: Stronger conditioning, less diversity
- Lower: Weaker conditioning, more diversity

**Conditional generation:**
- Text: CLIP embeddings
- Images: Encoded features
- Other: Domain-specific encoders

## Key Takeaways

1. **Diffusion models learn reverse denoising**: They transform noise into data by learning to reverse a forward noising process.

2. **DDPM provides probabilistic framework**: Forward and reverse processes are well-defined, enabling principled training.

3. **Noise prediction is key**: Predicting the added noise (rather than $x_{t-1}$ directly) simplifies training and improves stability.

4. **Noise schedule matters**: Linear, cosine, and learned schedules affect training dynamics and sample quality.

5. **Score matching connection**: Diffusion models are related to score matching, providing theoretical grounding.

6. **Conditional generation enables control**: Classifier guidance and classifier-free guidance allow controlling generated samples.

7. **DDIM enables efficient sampling**: Deterministic sampling and subsequence steps reduce computational cost while maintaining quality.

8. **Latent diffusion improves efficiency**: Operating in learned latent spaces reduces dimensionality and computational requirements.

9. **Training is straightforward**: Simple MSE loss on noise prediction, though requires many diffusion steps.

10. **State-of-the-art quality**: Diffusion models achieve excellent results in image generation, audio synthesis, and other domains, often surpassing GANs.
