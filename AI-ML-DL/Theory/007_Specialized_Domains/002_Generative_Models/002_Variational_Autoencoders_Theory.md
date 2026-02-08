# Variational Autoencoders Theory

## Table of Contents

1. [Introduction to Variational Inference](#introduction-to-variational-inference)
2. [VAE Derivation and ELBO](#vae-derivation-and-elbo)
3. [Reparameterization Trick](#reparameterization-trick)
4. [KL Divergence and Regularization](#kl-divergence-and-regularization)
5. [Beta-VAE and Disentanglement](#beta-vae-and-disentanglement)
6. [VQ-VAE: Vector Quantization](#vq-vae-vector-quantization)
7. [Posterior Collapse Problem](#posterior-collapse-problem)
8. [Advanced VAE Variants](#advanced-vae-variants)
9. [Training and Optimization](#training-and-optimization)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Variational Inference

Variational inference provides a framework for approximating intractable posterior distributions, forming the foundation for Variational Autoencoders (VAEs).

### Problem Setup

Given observed data $\mathbf{x}$ and latent variables $\mathbf{z}$, we want to infer the posterior:

$$p(\mathbf{z} | \mathbf{x}) = \frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}$$

The marginal likelihood $p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ is typically intractable.

### Variational Approximation

Approximate $p(\mathbf{z} | \mathbf{x})$ with a simpler distribution $q_\phi(\mathbf{z} | \mathbf{x})$:

$$q_\phi(\mathbf{z} | \mathbf{x}) \approx p(\mathbf{z} | \mathbf{x})$$

Choose $q_\phi$ from a tractable family (e.g., Gaussian) parameterized by $\phi$.

### KL Divergence Objective

Minimize KL divergence between approximate and true posterior:

$$\text{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z} | \mathbf{x})) = \mathbb{E}_{q_\phi}[\log q_\phi(\mathbf{z} | \mathbf{x})] - \mathbb{E}_{q_\phi}[\log p(\mathbf{z} | \mathbf{x})]$$

### Evidence Lower Bound (ELBO)

The ELBO provides a tractable lower bound on the log-likelihood:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}[\log p_\theta(\mathbf{x} | \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

Maximizing ELBO simultaneously:
1. Maximizes data likelihood (reconstruction)
2. Minimizes KL divergence (regularization)

---

## VAE Derivation and ELBO

The Variational Autoencoder combines variational inference with neural networks for generative modeling.

### VAE Architecture

**Encoder**: $q_\phi(\mathbf{z} | \mathbf{x})$ maps data to latent distribution
**Decoder**: $p_\theta(\mathbf{x} | \mathbf{z})$ maps latent to data distribution
**Prior**: $p(\mathbf{z})$ is typically standard Gaussian $\mathcal{N}(0, I)$

### ELBO Derivation

Starting from log-likelihood:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

Introduce variational distribution:

$$\log p(\mathbf{x}) = \log \int q_\phi(\mathbf{z} | \mathbf{x}) \frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} d\mathbf{z}$$

Apply Jensen's inequality:

$$\log p(\mathbf{x}) \geq \int q_\phi(\mathbf{z} | \mathbf{x}) \log \frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} d\mathbf{z}$$

Expanding:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}[\log p_\theta(\mathbf{x} | \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

### ELBO Components

**Reconstruction Term**: $\mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})}[\log p_\theta(\mathbf{x} | \mathbf{z})]$
- Encourages decoder to reconstruct data
- Measures how well latent codes explain data

**Regularization Term**: $-\text{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$
- Encourages posterior to match prior
- Prevents overfitting to training data
- Enables sampling from prior for generation

### VAE Loss Function

For dataset $\mathcal{D} = \{\mathbf{x}^{(i)}\}_{i=1}^{N}$:

$$\mathcal{L}_{\text{VAE}} = -\sum_{i=1}^{N} \left[\mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x}^{(i)})}[\log p_\theta(\mathbf{x}^{(i)} | \mathbf{z})] - \text{KL}(q_\phi(\mathbf{z} | \mathbf{x}^{(i)}) \| p(\mathbf{z}))\right]$$

### Gaussian VAE

For Gaussian encoder and decoder:

**Encoder**: $q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}))$

**Decoder**: $p_\theta(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \boldsymbol{\sigma}_\theta^2(\mathbf{z}))$

**Prior**: $p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, I)$

---

## Reparameterization Trick

The reparameterization trick enables backpropagation through stochastic sampling.

### Problem

Sampling $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ is non-differentiable:

$$\mathbf{z} = \text{sample}(q_\phi(\mathbf{z} | \mathbf{x}))$$

Cannot backpropagate through random sampling.

### Solution

Reparameterize sampling using a deterministic function and auxiliary noise:

$$\mathbf{z} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) \quad \text{where } \boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$$

### Gaussian Reparameterization

For Gaussian $q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}))$:

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$ and $\odot$ is element-wise multiplication.

### Gradient Estimation

Now gradients flow through deterministic function:

$$\nabla_\phi \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} | \mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})}[\nabla_\phi \log p_\theta(\mathbf{x} | g_\phi(\boldsymbol{\epsilon}, \mathbf{x}))]$$

### Other Distributions

**Beta Distribution**: Use inverse CDF:
$$z = F^{-1}(\epsilon; \alpha, \beta)$$

**Gamma Distribution**: Use rejection sampling or approximations.

**Discrete Variables**: Use Gumbel-Softmax or straight-through estimator.

---

## KL Divergence and Regularization

The KL divergence term regularizes the latent space and enables generation.

### KL Divergence for Gaussians

For $q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$ and $p(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$:

$$\text{KL}(q \| p) = \frac{1}{2}\left[\text{tr}(\boldsymbol{\Sigma}_p^{-1} \boldsymbol{\Sigma}_q) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^T \boldsymbol{\Sigma}_p^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q) - d + \log \frac{|\boldsymbol{\Sigma}_p|}{|\boldsymbol{\Sigma}_q|}\right]$$

For standard Gaussian prior $p(\mathbf{z}) = \mathcal{N}(0, I)$:

$$\text{KL}(q_\phi(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z})) = \frac{1}{2}\sum_{i=1}^{d}[\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2]$$

### Regularization Effect

The KL term encourages:
1. **Latent codes near origin**: $\mu_i \approx 0$
2. **Unit variance**: $\sigma_i \approx 1$
3. **Smooth latent space**: Enables interpolation

### Trade-off

Balancing reconstruction and regularization:

- **High KL weight**: Better regularization, worse reconstruction
- **Low KL weight**: Better reconstruction, worse regularization

### KL Annealing

Gradually increase KL weight during training:

$$\mathcal{L} = -\mathbb{E}[\log p(\mathbf{x} | \mathbf{z})] + \beta(t) \cdot \text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

where $\beta(t)$ increases from 0 to 1.

---

## Beta-VAE and Disentanglement

Beta-VAE introduces a hyperparameter to control the trade-off between reconstruction and disentanglement.

### Beta-VAE Objective

$$\mathcal{L}_{\beta\text{-VAE}} = -\mathbb{E}[\log p(\mathbf{x} | \mathbf{z})] + \beta \cdot \text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$$

where $\beta > 1$ increases regularization strength.

### Disentangled Representations

A disentangled representation separates independent factors of variation:
- Each dimension captures one factor
- Factors are independent
- Interpretable and controllable

### Disentanglement Metrics

**Beta-VAE Metric**: Train classifier to predict factors from latent codes.

**Factor-VAE Metric**: Measure total correlation:
$$\text{TC}(\mathbf{z}) = \text{KL}(q(\mathbf{z}) \| \prod_{i} q(z_i))$$

**MIG (Mutual Information Gap)**: Measure mutual information between factors and latents.

### Why Beta-VAE Works

Stronger KL regularization encourages:
1. **Factorized posterior**: $q(\mathbf{z} | \mathbf{x}) \approx \prod_i q(z_i | \mathbf{x})$
2. **Independent latents**: Reduces correlation between dimensions
3. **Sparse representations**: Fewer active dimensions

### Limitations

- **Reconstruction quality**: Higher $\beta$ degrades reconstruction
- **No guarantee**: Disentanglement not guaranteed
- **Task-dependent**: Optimal $\beta$ varies by task

---

## VQ-VAE: Vector Quantization

VQ-VAE uses vector quantization to learn discrete latent representations.

### Vector Quantization

Quantize continuous latents to discrete codes:

$$\mathbf{z}_q = \text{VQ}(\mathbf{z}_e) = \mathbf{e}_k \quad \text{where } k = \arg\min_j \|\mathbf{z}_e - \mathbf{e}_j\|_2$$

where $\mathbf{e}_j$ are learnable codebook vectors.

### VQ-VAE Architecture

**Encoder**: $q(\mathbf{z}_e | \mathbf{x})$ outputs continuous latents
**Quantizer**: $\mathbf{z}_q = \text{VQ}(\mathbf{z}_e)$ quantizes to codebook
**Decoder**: $p(\mathbf{x} | \mathbf{z}_q)$ reconstructs from quantized codes

### Training Objective

$$\mathcal{L}_{\text{VQ-VAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|_2^2 + \beta \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|_2^2$$

where $\text{sg}[\cdot]$ stops gradients.

**Terms**:
1. Reconstruction loss
2. Codebook loss (update codebook)
3. Commitment loss (commit encoder to codebook)

### Straight-Through Estimator

Gradients pass through quantizer using straight-through:

$$\nabla_{\mathbf{z}_e} \mathcal{L} = \nabla_{\mathbf{z}_q} \mathcal{L}$$

### Advantages

- **Discrete latents**: Natural for discrete data (text, code)
- **Hierarchical modeling**: Can stack multiple VQ-VAEs
- **Prior learning**: Learn powerful prior over discrete codes

### VQ-VAE-2

Hierarchical VQ-VAE with multiple levels:
- Bottom level: High-resolution details
- Top level: Low-resolution structure

---

## Posterior Collapse Problem

Posterior collapse occurs when the posterior matches the prior, making the latent code uninformative.

### Symptoms

- $\text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z})) \approx 0$ for all $\mathbf{x}$
- Latent codes are nearly identical: $q(\mathbf{z} | \mathbf{x}) \approx p(\mathbf{z})$
- Decoder ignores latent: $p(\mathbf{x} | \mathbf{z}) \approx p(\mathbf{x})$

### Causes

1. **Powerful decoder**: Can reconstruct without using latent
2. **Weak encoder**: Cannot encode useful information
3. **KL weight too high**: Over-regularization

### Solutions

**KL Annealing**: Gradually increase KL weight:
$$\beta(t) = \min(1, t / T_{\text{anneal}})$$

**Free Bits**: Ensure minimum KL per dimension:
$$\mathcal{L} = -\mathbb{E}[\log p(\mathbf{x} | \mathbf{z})] + \sum_i \max(\lambda, \text{KL}_i)$$

**Cyclical Annealing**: Cycle KL weight to explore latent space.

**Decoder Regularization**: Weaken decoder (e.g., dropout, weight decay).

**Aggressive Encoder**: Use powerful encoder to force information flow.

### Detection

Monitor during training:
- Average KL divergence
- Reconstruction quality
- Latent code diversity

---

## Advanced VAE Variants

### Conditional VAE (CVAE)

Condition on additional information (class labels, text):

$$q_\phi(\mathbf{z} | \mathbf{x}, \mathbf{c}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}, \mathbf{c}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}, \mathbf{c}))$$

$$p_\theta(\mathbf{x} | \mathbf{z}, \mathbf{c}) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{z}, \mathbf{c}), \boldsymbol{\sigma}_\theta^2(\mathbf{z}, \mathbf{c}))$$

### Adversarial Autoencoder (AAE)

Replace KL term with adversarial training:

$$\min_G \max_D \mathbb{E}[\log D(\mathbf{z})] + \mathbb{E}[\log(1-D(G(\mathbf{x})))]$$

where $G$ is encoder and $D$ discriminates between prior and posterior.

### Wasserstein Autoencoder (WAE)

Use Wasserstein distance instead of KL:

$$\mathcal{L}_{\text{WAE}} = \mathbb{E}[\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2] + \lambda \cdot \mathcal{D}(q(\mathbf{z}), p(\mathbf{z}))$$

### Hierarchical VAE

Multiple levels of latents:

$$p(\mathbf{x}, \mathbf{z}_1, \mathbf{z}_2) = p(\mathbf{x} | \mathbf{z}_1) p(\mathbf{z}_1 | \mathbf{z}_2) p(\mathbf{z}_2)$$

### NVAE (Normalizing Flow VAE)

Use normalizing flows for flexible posterior:

$$q(\mathbf{z} | \mathbf{x}) = q_0(\mathbf{z}_0 | \mathbf{x}) \prod_{i=1}^{K} \left|\det \frac{\partial f_i}{\partial \mathbf{z}_{i-1}}\right|^{-1}$$

---

## Training and Optimization

### Optimization Challenges

1. **Gradient variance**: High variance in Monte Carlo gradient estimates
2. **KL vanishing**: Posterior collapse
3. **Training instability**: Sensitive to hyperparameters

### Gradient Estimation

**Single sample**: Use one sample for Monte Carlo:
$$\nabla_\phi \mathcal{L} \approx \nabla_\phi [-\log p(\mathbf{x} | \mathbf{z}) + \text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))]$$

**Multiple samples**: Average over multiple samples for lower variance.

### Variance Reduction

**Control variates**: Use baseline to reduce variance.

**Importance sampling**: Weight samples by importance.

**Rao-Blackwellization**: Analytically compute some expectations.

### Learning Rate Scheduling

- Start with higher learning rate
- Decay when loss plateaus
- Use warmup for KL annealing

### Regularization

**Weight decay**: L2 regularization on parameters.

**Dropout**: Regularize encoder/decoder.

**Spectral normalization**: Stabilize training.

---

## Key Takeaways

1. **Variational Inference**: VAE uses variational inference to approximate intractable posterior, maximizing ELBO as lower bound on log-likelihood.

2. **ELBO Decomposition**: $\text{ELBO} = \mathbb{E}[\log p(\mathbf{x} | \mathbf{z})] - \text{KL}(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}))$ balances reconstruction and regularization.

3. **Reparameterization Trick**: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ enables backpropagation through stochastic sampling by making sampling differentiable.

4. **KL Regularization**: KL term encourages latent codes to match prior, enabling generation and preventing overfitting, but requires careful balancing.

5. **Beta-VAE**: $\beta > 1$ increases KL weight to encourage disentangled representations, trading reconstruction quality for interpretability.

6. **VQ-VAE**: Vector quantization creates discrete latent codes, natural for discrete data and enabling powerful hierarchical priors.

7. **Posterior Collapse**: When posterior matches prior, latent becomes uninformative. Solutions include KL annealing, free bits, and decoder regularization.

8. **Advanced Variants**: CVAE conditions on additional info, AAE uses adversarial training, WAE uses Wasserstein distance, hierarchical VAE uses multiple latent levels.

9. **Training Challenges**: High gradient variance, KL vanishing, and instability require careful optimization, variance reduction, and regularization strategies.

10. **Applications**: VAEs enable generation, representation learning, anomaly detection, and controllable generation through learned latent spaces.
