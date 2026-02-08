# Autoencoders and Variational Autoencoders

## Table of Contents

1. [Introduction](#introduction)
2. [Standard Autoencoders](#standard-autoencoders)
3. [Denoising Autoencoders](#denoising-autoencoders)
4. [Variational Autoencoders: Motivation](#variational-autoencoders-motivation)
5. [VAE Mathematical Foundation](#vae-mathematical-foundation)
6. [Reparameterization Trick](#reparameterization-trick)
7. [Beta-VAE and Variants](#beta-vae-and-variants)
8. [Training and Implementation](#training-and-implementation)
9. [Applications](#applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Autoencoders are neural networks trained to reconstruct their input, learning efficient representations in the process. Variational Autoencoders (VAEs) extend this idea by learning a probabilistic latent space, enabling generation of new samples. These architectures are fundamental to unsupervised learning, representation learning, and generative modeling.

This chapter covers the theory and implementation of autoencoders and VAEs, from basic reconstruction objectives to advanced probabilistic formulations and their applications in generative modeling.

## Standard Autoencoders

### Architecture

An autoencoder consists of:

1. **Encoder**: Maps input $\mathbf{x}$ to latent code $\mathbf{z}$
   $$\mathbf{z} = f_{\text{enc}}(\mathbf{x})$$

2. **Decoder**: Maps latent code $\mathbf{z}$ to reconstruction $\hat{\mathbf{x}}$
   $$\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z})$$

3. **Objective**: Minimize reconstruction error
   $$\mathcal{L} = ||\mathbf{x} - \hat{\mathbf{x}}||^2$$

### Bottleneck Principle

The latent space $\mathbf{z}$ has lower dimensionality than input, forcing the network to learn compressed representations.

**Dimensions**:
- Input: $\mathbf{x} \in \mathbb{R}^d$
- Latent: $\mathbf{z} \in \mathbb{R}^k$ where $k < d$
- Output: $\hat{\mathbf{x}} \in \mathbb{R}^d$

### Loss Function

Common reconstruction losses:

**Mean Squared Error**:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} ||\mathbf{x}_i - \hat{\mathbf{x}}_i||^2$$

**Binary Cross-Entropy** (for binary inputs):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} [\mathbf{x}_i \log \hat{\mathbf{x}}_i + (1-\mathbf{x}_i) \log(1-\hat{\mathbf{x}}_i)]$$

### Training

Standard backpropagation minimizes reconstruction error:

$$\theta^* = \arg\min_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\mathcal{L}(\mathbf{x}, f_{\text{dec}}(f_{\text{enc}}(\mathbf{x})))]$$

### Limitations

1. **No Probabilistic Model**: Cannot sample new data
2. **Discontinuous Latent Space**: Similar codes may map to very different outputs
3. **No Regularization**: May memorize training data
4. **Limited Generalization**: Poor interpolation in latent space

## Denoising Autoencoders

Denoising autoencoders learn robust representations by reconstructing clean inputs from corrupted versions.

### Training Procedure

1. **Corrupt Input**: $\tilde{\mathbf{x}} \sim q(\tilde{\mathbf{x}} | \mathbf{x})$
2. **Encode**: $\mathbf{z} = f_{\text{enc}}(\tilde{\mathbf{x}})$
3. **Decode**: $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z})$
4. **Reconstruct**: Minimize $||\mathbf{x} - \hat{\mathbf{x}}||^2$

### Corruption Process

Common corruption methods:

**Gaussian Noise**:
$$\tilde{\mathbf{x}} = \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$

**Masking Noise**:
$$\tilde{\mathbf{x}}_i = \begin{cases}
0 & \text{with probability } p \\
\mathbf{x}_i & \text{otherwise}
\end{cases}$$

**Salt and Pepper**: Randomly set pixels to 0 or 1

### Benefits

1. **Robustness**: Learns to handle noise
2. **Better Representations**: Must extract meaningful features
3. **Regularization**: Prevents overfitting
4. **Generalization**: Better interpolation properties

### Variational Bound

Denoising can be viewed as maximizing a variational lower bound:

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q(\tilde{\mathbf{x}}|\mathbf{x})}[\log p(\mathbf{x}|\tilde{\mathbf{x}})] - \text{KL}(q(\tilde{\mathbf{x}}|\mathbf{x}) || p(\tilde{\mathbf{x}}))$$

## Variational Autoencoders: Motivation

VAEs address limitations of standard autoencoders by learning a probabilistic latent space.

### Key Ideas

1. **Probabilistic Encoder**: $q_{\phi}(\mathbf{z}|\mathbf{x})$ instead of deterministic $f_{\text{enc}}(\mathbf{x})$
2. **Probabilistic Decoder**: $p_{\theta}(\mathbf{x}|\mathbf{z})$ instead of deterministic $f_{\text{dec}}(\mathbf{z})$
3. **Prior Distribution**: $p(\mathbf{z})$ over latent space
4. **Regularized Latent Space**: Encourages smooth, continuous representations

### Generative Model

VAE defines a generative process:

1. Sample latent: $\mathbf{z} \sim p(\mathbf{z})$
2. Generate data: $\mathbf{x} \sim p_{\theta}(\mathbf{x}|\mathbf{z})$

This enables sampling new data points.

### Inference

Given data $\mathbf{x}$, infer latent $\mathbf{z}$:

$$p(\mathbf{z}|\mathbf{x}) = \frac{p_{\theta}(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})}$$

But $p(\mathbf{x}) = \int p_{\theta}(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$ is intractable.

**Solution**: Approximate $p(\mathbf{z}|\mathbf{x})$ with $q_{\phi}(\mathbf{z}|\mathbf{x})$

## VAE Mathematical Foundation

### Variational Lower Bound

Maximize log-likelihood:

$$\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

Introduce variational distribution $q_{\phi}(\mathbf{z}|\mathbf{x})$:

$$\log p_{\theta}(\mathbf{x}) = \log \int q_{\phi}(\mathbf{z}|\mathbf{x}) \frac{p_{\theta}(\mathbf{x}|\mathbf{z}) p(\mathbf{z})}{q_{\phi}(\mathbf{z}|\mathbf{x})} d\mathbf{z}$$

Using Jensen's inequality:

$$\log p_{\theta}(\mathbf{x}) \geq \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \text{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

### ELBO Decomposition

The Evidence Lower BOund (ELBO) consists of:

1. **Reconstruction Term**: $\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]$
   - Encourages accurate reconstruction
   - Maximizes likelihood of data given latent

2. **Regularization Term**: $-\text{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$
   - Encourages posterior to match prior
   - Regularizes latent space

### Standard VAE Formulation

**Encoder**: $q_{\phi}(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_{\phi}(\mathbf{x}), \boldsymbol{\sigma}_{\phi}^2(\mathbf{x})\mathbf{I})$

**Decoder**: $p_{\theta}(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{\theta}(\mathbf{z}), \boldsymbol{\sigma}_{\theta}^2(\mathbf{z})\mathbf{I})$ or Bernoulli

**Prior**: $p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})$

### KL Divergence

For Gaussian distributions:

$$\text{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\sigma}_1^2) || \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\sigma}_2^2)) = \frac{1}{2}\left[\log\frac{\boldsymbol{\sigma}_2^2}{\boldsymbol{\sigma}_1^2} + \frac{\boldsymbol{\sigma}_1^2 + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^2}{\boldsymbol{\sigma}_2^2} - 1\right]$$

For standard VAE with unit Gaussian prior:

$$\text{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) = \frac{1}{2}\sum_{i=1}^{d} [\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2]$$

## Reparameterization Trick

The reparameterization trick enables backpropagation through stochastic sampling.

### Problem

Sampling $\mathbf{z} \sim q_{\phi}(\mathbf{z}|\mathbf{x})$ is not differentiable.

### Solution

Reparameterize sampling:

$$\mathbf{z} = \boldsymbol{\mu}_{\phi}(\mathbf{x}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}) \odot \boldsymbol{\epsilon}$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### Benefits

1. **Differentiable**: Gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$
2. **Stochastic**: Still samples from correct distribution
3. **Efficient**: Single sample sufficient for gradient estimate

### Implementation

```python
def reparameterize(mu, logvar):
    """Reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### Gradient Flow

Gradients flow through:
- $\boldsymbol{\mu}_{\phi}(\mathbf{x})$: Direct
- $\boldsymbol{\sigma}_{\phi}(\mathbf{x})$: Through $\boldsymbol{\epsilon}$
- $\boldsymbol{\epsilon}$: Treated as constant (no gradient)

## Beta-VAE and Variants

### Beta-VAE

Beta-VAE introduces a hyperparameter $\beta$ to control the trade-off:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \beta \cdot \text{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

**Effects**:
- $\beta > 1$: Stronger regularization, more disentangled representations
- $\beta < 1$: Weaker regularization, better reconstruction
- $\beta = 1$: Standard VAE

### Disentangled Representations

Beta-VAE encourages disentanglement:
- Each dimension of $\mathbf{z}$ controls independent factors
- Better interpretability
- Better generalization

### VAE-GAN

Combines VAE with GAN discriminator:

$$\mathcal{L} = \mathcal{L}_{\text{VAE}} + \lambda \mathcal{L}_{\text{GAN}}$$

Uses discriminator to improve sample quality.

### Conditional VAE (CVAE)

Conditions on additional information:

$$q_{\phi}(\mathbf{z}|\mathbf{x}, \mathbf{c}), \quad p_{\theta}(\mathbf{x}|\mathbf{z}, \mathbf{c})$$

Enables controlled generation.

### Vector Quantized VAE (VQ-VAE)

Uses discrete latent codes:

$$\mathbf{z}_q = \text{Quantize}(\mathbf{z}_e) = \arg\min_{\mathbf{e}_k \in \mathcal{E}} ||\mathbf{z}_e - \mathbf{e}_k||$$

where $\mathcal{E}$ is a codebook of embeddings.

## Training and Implementation

### Loss Function

For continuous data (Gaussian decoder):

$$\mathcal{L} = \frac{1}{2\sigma^2} ||\mathbf{x} - \hat{\mathbf{x}}||^2 + \frac{1}{2}\sum_{i} [\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2]$$

For binary data (Bernoulli decoder):

$$\mathcal{L} = -\sum_{i} [x_i \log \hat{x}_i + (1-x_i) \log(1-\hat{x}_i)] + \frac{1}{2}\sum_{i} [\sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2]$$

### Training Procedure

1. Forward pass: Encode $\mathbf{x} \to \mathbf{z}$
2. Reparameterize: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$
3. Decode: $\mathbf{z} \to \hat{\mathbf{x}}$
4. Compute loss: Reconstruction + KL
5. Backpropagate: Update $\phi$ and $\theta$

### Implementation Example

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss function."""
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss
```

### Challenges

1. **Posterior Collapse**: $q_{\phi}(\mathbf{z}|\mathbf{x})$ collapses to prior
2. **Blurry Samples**: Gaussian decoder produces smooth outputs
3. **KL Vanishing**: KL term becomes too small
4. **Training Instability**: Sensitive to hyperparameters

## Applications

### Data Generation

VAEs can generate new samples by sampling from prior:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \to \mathbf{x} \sim p_{\theta}(\mathbf{x}|\mathbf{z})$$

### Representation Learning

Learned latent codes $\mathbf{z}$ can be used for:
- Downstream tasks
- Clustering
- Visualization
- Interpolation

### Anomaly Detection

Reconstruction error indicates anomalies:

$$\text{Anomaly Score} = ||\mathbf{x} - \hat{\mathbf{x}}||^2$$

### Image Editing

Interpolate in latent space:

$$\mathbf{z}_{\text{interp}} = (1-\alpha)\mathbf{z}_1 + \alpha\mathbf{z}_2$$

### Denoising

Reconstruct clean version from noisy input.

## Key Takeaways

1. **Autoencoders**: Learn compressed representations by reconstructing input, with bottleneck forcing efficient encoding.

2. **Denoising Autoencoders**: Learn robust representations by reconstructing clean inputs from corrupted versions, providing implicit regularization.

3. **VAE Motivation**: Addresses autoencoder limitations by learning probabilistic latent space, enabling generation and better interpolation.

4. **Variational Lower Bound**: ELBO consists of reconstruction term (data likelihood) and regularization term (KL divergence), balancing reconstruction and latent space structure.

5. **Reparameterization Trick**: Enables backpropagation through stochastic sampling by expressing samples as deterministic function of parameters plus noise.

6. **Beta-VAE**: Introduces $\beta$ hyperparameter to control trade-off between reconstruction and regularization, with $\beta > 1$ encouraging disentangled representations.

7. **Training**: Combines reconstruction loss (MSE or BCE) with KL divergence, trained end-to-end via reparameterization trick.

8. **Applications**: Data generation, representation learning, anomaly detection, and image editing through latent space manipulation.

9. **Limitations**: Can produce blurry samples, suffer from posterior collapse, and require careful hyperparameter tuning.

10. **Modern Extensions**: VQ-VAE, VAE-GAN, and conditional VAEs extend capabilities for specific applications and improved sample quality.
