# Generative Adversarial Networks

## Table of Contents

1. [Introduction](#introduction)
2. [GAN Framework](#gan-framework)
3. [Discriminator and Generator](#discriminator-and-generator)
4. [Training Dynamics](#training-dynamics)
5. [DCGAN Architecture](#dcgan-architecture)
6. [Wasserstein GAN](#wasserstein-gan)
7. [Mode Collapse](#mode-collapse)
8. [Training Stability](#training-stability)
9. [Advanced GAN Variants](#advanced-gan-variants)
10. [Key Takeaways](#key-takeaways)

## Introduction

Generative Adversarial Networks (GANs) introduced a novel framework for training generative models through adversarial competition between a generator and discriminator. This adversarial training enables learning complex data distributions and generating high-quality samples, revolutionizing generative modeling.

This chapter covers the mathematical foundations of GANs, training dynamics, challenges like mode collapse, and advanced variants that address stability and quality issues.

## GAN Framework

### Two-Player Minimax Game

GANs formulate generation as a two-player game:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1-D(G(\mathbf{z})))]$$

where:
- $G$: Generator, maps noise $\mathbf{z}$ to data $\mathbf{x}$
- $D$: Discriminator, classifies real vs. fake
- $p_{\text{data}}$: Real data distribution
- $p(\mathbf{z})$: Prior distribution (typically $\mathcal{N}(\mathbf{0}, \mathbf{I})$)

### Optimal Discriminator

For fixed generator $G$, optimal discriminator is:

$$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

where $p_g$ is the generator's distribution.

### Optimal Generator

When discriminator is optimal, minimizing generator loss is equivalent to minimizing:

$$\text{JS}(p_{\text{data}} || p_g) - 2\log 2$$

where JS is Jensen-Shannon divergence.

### Nash Equilibrium

At equilibrium:
- Discriminator cannot distinguish real from fake: $D^*(\mathbf{x}) = \frac{1}{2}$
- Generator matches data distribution: $p_g = p_{\text{data}}$

## Discriminator and Generator

### Discriminator

The discriminator $D(\mathbf{x})$ outputs probability that $\mathbf{x}$ is real:

$$D(\mathbf{x}) = \sigma(f_D(\mathbf{x}))$$

where $f_D$ is a neural network and $\sigma$ is sigmoid.

**Objective**: Maximize

$$\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1-D(G(\mathbf{z})))]$$

**Interpretation**: 
- Maximize $\log D(\mathbf{x})$ for real data
- Maximize $\log(1-D(G(\mathbf{z})))$ for fake data (minimize $D(G(\mathbf{z}))$)

### Generator

The generator $G(\mathbf{z})$ maps noise to data:

$$\mathbf{x} = G(\mathbf{z}), \quad \mathbf{z} \sim p(\mathbf{z})$$

**Objective**: Minimize

$$\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1-D(G(\mathbf{z})))]$$

**Alternative**: Maximize (non-saturating loss)

$$\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log D(G(\mathbf{z}))]$$

### Training Procedure

**Alternating Updates**:

1. **Update Discriminator** (k steps):
   - Sample real batch: $\{\mathbf{x}_i\}_{i=1}^{m} \sim p_{\text{data}}$
   - Sample noise: $\{\mathbf{z}_i\}_{i=1}^{m} \sim p(\mathbf{z})$
   - Generate fakes: $\{\tilde{\mathbf{x}}_i = G(\mathbf{z}_i)\}_{i=1}^{m}$
   - Update $D$ to maximize: $\frac{1}{m}\sum_{i=1}^{m}[\log D(\mathbf{x}_i) + \log(1-D(\tilde{\mathbf{x}}_i))]$

2. **Update Generator** (1 step):
   - Sample noise: $\{\mathbf{z}_i\}_{i=1}^{m} \sim p(\mathbf{z})$
   - Update $G$ to minimize: $\frac{1}{m}\sum_{i=1}^{m}\log(1-D(G(\mathbf{z}_i)))$

## Training Dynamics

### Convergence Issues

GAN training is unstable due to:

1. **Non-Convex Game**: No guarantee of convergence
2. **Simultaneous Optimization**: Generator and discriminator compete
3. **Gradient Issues**: Vanishing gradients when discriminator is too good
4. **Mode Collapse**: Generator produces limited diversity

### Discriminator Too Good

When discriminator perfectly distinguishes real from fake:
- $D(G(\mathbf{z})) \approx 0$ for all $\mathbf{z}$
- Generator gradient: $\nabla_G \log(1-D(G(\mathbf{z}))) \approx 0$
- **Vanishing Gradients**: Generator cannot learn

**Solution**: Use non-saturating loss for generator:

$$\max_G \mathbb{E}_{\mathbf{z}}[\log D(G(\mathbf{z}))]$$

### Discriminator Too Weak

When discriminator cannot distinguish:
- Provides poor signal to generator
- Generator may not learn meaningful patterns

**Solution**: Train discriminator for multiple steps per generator update.

### Optimal Training Balance

- Discriminator should be good but not perfect
- Generator should improve gradually
- Balance maintained through learning rates and update frequencies

## DCGAN Architecture

Deep Convolutional GAN (DCGAN) applies CNNs to GANs with architectural guidelines.

### Generator Architecture

**Guidelines**:
1. Replace pooling with strided convolutions
2. Use batch normalization (except output layer)
3. Remove fully connected hidden layers
4. Use ReLU for hidden layers, Tanh for output
5. Use transposed convolutions for upsampling

**Structure**:
- Input: Noise vector $\mathbf{z} \in \mathbb{R}^{100}$
- Fully connected → Reshape to feature map
- Transposed convolutions (upsampling)
- Batch normalization + ReLU
- Final layer: Tanh activation

### Discriminator Architecture

**Guidelines**:
1. Use strided convolutions instead of pooling
2. Use batch normalization (except input layer)
3. Remove fully connected hidden layers
4. Use LeakyReLU instead of ReLU

**Structure**:
- Input: Image
- Convolutions (downsampling)
- Batch normalization + LeakyReLU
- Final layer: Sigmoid for binary classification

### Transposed Convolution

Also called deconvolution or fractionally strided convolution:

$$\mathbf{Y} = \mathbf{X} \star \mathbf{W}^T$$

Upsamples input by inserting zeros and convolving.

### Implementation Example

```python
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)
```

## Wasserstein GAN

Wasserstein GAN (WGAN) addresses training instability by using Wasserstein distance.

### Wasserstein Distance

Also called Earth Mover's Distance:

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[||x-y||]$$

where $\Pi$ is the set of couplings.

### Kantorovich-Rubinstein Duality

$$W(p_{\text{data}}, p_g) = \sup_{||f||_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$$

where $||f||_L \leq 1$ means $f$ is 1-Lipschitz.

### WGAN Formulation

Replace discriminator with critic (no sigmoid):

$$\min_G \max_{||D||_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[D(G(\mathbf{z}))]$$

### Weight Clipping

Enforce Lipschitz constraint by clipping weights:

$$w \leftarrow \text{clip}(w, -c, c)$$

where $c$ is typically $0.01$.

### WGAN-GP

WGAN with Gradient Penalty avoids weight clipping:

$$\mathcal{L}_{\text{GP}} = \lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}[(||\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})||_2 - 1)^2]$$

where $\hat{\mathbf{x}} = \epsilon \mathbf{x} + (1-\epsilon) G(\mathbf{z})$ with $\epsilon \sim \mathcal{U}(0,1)$.

**Benefits**:
- More stable training
- Better sample quality
- No mode collapse
- Meaningful loss metric

## Mode Collapse

Mode collapse occurs when generator produces limited diversity.

### Symptoms

- Generator produces same or few samples
- Low diversity in generated data
- Discriminator may still be fooled

### Causes

1. **Generator Optimization**: Finds single mode that fools discriminator
2. **Discriminator Weakness**: Cannot penalize lack of diversity
3. **Training Imbalance**: Generator updates too frequent

### Solutions

1. **Unrolled GANs**: Unroll discriminator updates
2. **Mini-batch Discrimination**: Encourage diversity within batch
3. **Feature Matching**: Match feature statistics instead of raw output
4. **Wasserstein Distance**: Better distance metric (WGAN)
5. **Spectral Normalization**: Stabilize discriminator

### Mini-Batch Discrimination

Add term to discriminator that measures similarity within batch:

$$f(\mathbf{x}_i) = [\mathbf{x}_i, \sum_{j \neq i} c(\mathbf{x}_i, \mathbf{x}_j)]$$

Encourages generator to produce diverse samples.

## Training Stability

### Common Issues

1. **Non-Convergence**: Training oscillates
2. **Mode Collapse**: Limited diversity
3. **Vanishing Gradients**: Discriminator too good
4. **Exploding Gradients**: Unstable updates

### Best Practices

1. **Learning Rates**: 
   - Discriminator: $2 \times 10^{-4}$
   - Generator: $2 \times 10^{-4}$ (or lower)

2. **Update Frequency**: 
   - Train discriminator more (e.g., 5:1 ratio)
   - Or train until optimal

3. **Architecture**:
   - Use batch normalization
   - Avoid fully connected layers
   - Use LeakyReLU in discriminator

4. **Initialization**:
   - Proper weight initialization
   - Avoid saturation

5. **Loss Function**:
   - Use non-saturating loss for generator
   - Or use Wasserstein distance

### Monitoring

- **Loss Values**: Should not diverge
- **Sample Quality**: Visual inspection
- **Inception Score**: Quantitative metric
- **FID Score**: Fréchet Inception Distance

## Advanced GAN Variants

### Progressive GAN

Grows generator and discriminator progressively:
- Start with low resolution
- Add layers gradually
- Enables high-resolution generation

### StyleGAN

Separates style and content:
- Mapping network maps noise to style
- Synthesis network generates image
- Enables style mixing and interpolation

### BigGAN

Scales GANs to large models:
- Large batch sizes
- Class conditioning
- Orthogonal regularization
- Truncation trick for sampling

### Conditional GANs

Condition on additional information:

$$\min_G \max_D V(D, G) = \mathbb{E}[\log D(\mathbf{x}|\mathbf{c})] + \mathbb{E}[\log(1-D(G(\mathbf{z}|\mathbf{c})|\mathbf{c}))]$$

Enables controlled generation.

### CycleGAN

Unpaired image-to-image translation:
- Two generators and discriminators
- Cycle consistency loss
- No paired training data needed

## Key Takeaways

1. **Adversarial Framework**: GANs train generator and discriminator in a minimax game, where generator learns to fool discriminator.

2. **Optimal Equilibrium**: At Nash equilibrium, generator matches data distribution and discriminator cannot distinguish real from fake.

3. **Training Challenges**: GAN training is unstable due to non-convex game, vanishing gradients, and mode collapse.

4. **DCGAN Guidelines**: Architectural guidelines (batch norm, strided convolutions, LeakyReLU) enable stable CNN-based GANs.

5. **Wasserstein GAN**: Uses Wasserstein distance instead of JS divergence, providing more stable training and meaningful loss metric.

6. **Mode Collapse**: Generator may produce limited diversity; solutions include mini-batch discrimination, unrolled GANs, and Wasserstein distance.

7. **Training Stability**: Requires careful balance of learning rates, update frequencies, architecture choices, and loss functions.

8. **Advanced Variants**: Progressive GAN, StyleGAN, BigGAN, and conditional GANs extend capabilities for specific applications and improved quality.

9. **Evaluation Metrics**: Inception Score and FID provide quantitative measures of sample quality and diversity.

10. **Practical Considerations**: Monitoring training, using proper initialization, and selecting appropriate architectures are crucial for successful GAN training.
