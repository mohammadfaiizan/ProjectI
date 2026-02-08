# Generative Adversarial Networks Theory

## Table of Contents

1. [Introduction to GANs](#introduction-to-gans)
2. [GAN Training Dynamics](#gan-training-dynamics)
3. [Nash Equilibrium and Game Theory](#nash-equilibrium-and-game-theory)
4. [Mode Collapse Problem](#mode-collapse-problem)
5. [Wasserstein Distance and WGAN](#wasserstein-distance-and-wgan)
6. [Spectral Normalization](#spectral-normalization)
7. [Progressive Training and StyleGAN](#progressive-training-and-stylegan)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Advanced GAN Architectures](#advanced-gan-architectures)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to GANs

Generative Adversarial Networks (GANs) train two neural networks in an adversarial game: a generator creates fake samples, and a discriminator distinguishes real from fake.

### Adversarial Framework

**Generator** $G$: Maps noise $\mathbf{z} \sim p_z(\mathbf{z})$ to data space:
$$\mathbf{x}_{fake} = G(\mathbf{z})$$

**Discriminator** $D$: Classifies samples as real or fake:
$$D(\mathbf{x}) = \text{probability that } \mathbf{x} \text{ is real}$$

### Objective Function

The minimax game:

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1-D(G(\mathbf{z})))]$$

**Discriminator**: Maximizes $V(D, G)$ to correctly classify real and fake.

**Generator**: Minimizes $V(D, G)$ to fool the discriminator.

### Optimal Discriminator

For fixed generator, optimal discriminator:

$$D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}$$

where $p_g$ is the generator's distribution.

### Optimal Generator

When $D = D^*$, generator minimizes:

$$C(G) = -\log(4) + 2 \cdot \text{JSD}(p_{data} \| p_g)$$

where JSD is Jensen-Shannon divergence. Optimal when $p_g = p_{data}$.

---

## GAN Training Dynamics

Understanding training dynamics is crucial for stable GAN training.

### Training Procedure

**Alternating Updates**:

1. **Update Discriminator** (k steps):
   $$\theta_D \leftarrow \theta_D + \alpha_D \nabla_{\theta_D} V(D, G)$$

2. **Update Generator** (1 step):
   $$\theta_G \leftarrow \theta_G - \alpha_G \nabla_{\theta_G} V(D, G)$$

### Non-Saturating Loss

Original generator loss $\mathbb{E}[\log(1-D(G(\mathbf{z})))]$ saturates when discriminator is confident.

**Non-saturating alternative**:
$$\mathcal{L}_G = -\mathbb{E}[\log D(G(\mathbf{z}))]$$

Maximizes probability of fake being classified as real.

### Training Challenges

1. **Instability**: Generator and discriminator must be balanced
2. **Mode collapse**: Generator produces limited diversity
3. **Vanishing gradients**: When discriminator is too confident
4. **Non-convergence**: May not reach Nash equilibrium

### Convergence Analysis

**Theorem**: If $G$ and $D$ have enough capacity and training reaches optimal $D$ for each $G$, then $p_g$ converges to $p_{data}$.

In practice, convergence is not guaranteed due to:
- Limited capacity
- Imperfect optimization
- Non-zero-sum game dynamics

---

## Nash Equilibrium and Game Theory

GAN training can be viewed as finding Nash equilibrium in a two-player game.

### Game Theory Setup

**Players**: Generator $G$ and Discriminator $D$

**Strategies**: Parameters $\theta_G$ and $\theta_D$

**Payoffs**: 
- $U_G(\theta_G, \theta_D) = -V(D, G)$
- $U_D(\theta_G, \theta_D) = V(D, G)$

### Nash Equilibrium

Nash equilibrium $(\theta_G^*, \theta_D^*)$ satisfies:

$$U_G(\theta_G^*, \theta_D^*) \geq U_G(\theta_G, \theta_D^*) \quad \forall \theta_G$$
$$U_D(\theta_G^*, \theta_D^*) \geq U_D(\theta_G^*, \theta_D) \quad \forall \theta_D$$

Neither player can improve by unilaterally changing strategy.

### Existence

**Nash's Theorem**: Finite games have mixed-strategy Nash equilibria.

For continuous parameter spaces, existence depends on:
- Convexity of strategy sets
- Continuity of payoffs
- Compactness assumptions

### Finding Equilibrium

Standard gradient descent may not converge to Nash equilibrium:

- **Simultaneous updates**: May oscillate
- **Alternating updates**: May not converge
- **Best response**: Update to best response may diverge

### Unrolled GANs

Unroll discriminator updates to stabilize generator:

$$\mathcal{L}_G = -\mathbb{E}[\log D^{(K)}(G(\mathbf{z}))]$$

where $D^{(K)}$ is discriminator after $K$ unrolled steps.

---

## Mode Collapse Problem

Mode collapse occurs when the generator produces limited diversity, focusing on a few modes of the data distribution.

### Definition

**Mode Collapse**: Generator maps different $\mathbf{z}$ to similar $\mathbf{x}$, reducing diversity.

**Partial Mode Collapse**: Generator covers some but not all modes.

**Complete Mode Collapse**: Generator produces essentially one output.

### Causes

1. **Discriminator too strong**: Quickly learns to distinguish, generator gives up
2. **Generator too weak**: Cannot learn full distribution
3. **Training imbalance**: Discriminator updates too frequently
4. **Loss function**: May encourage mode collapse

### Detection

Monitor during training:
- **Inception Score (IS)**: Measures quality and diversity
- **Fréchet Inception Distance (FID)**: Compares distributions
- **Visual inspection**: Check generated samples
- **Latent space interpolation**: Should produce smooth transitions

### Solutions

**Unrolled GANs**: Prevent discriminator from overfitting to current generator.

**Mini-batch Discrimination**: Discriminator sees multiple samples, encourages diversity:
$$f(\mathbf{x}) = [f_1(\mathbf{x}), \ldots, f_k(\mathbf{x})]$$
$$c_b(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|_1)$$

**Feature Matching**: Match intermediate features instead of final output:
$$\mathcal{L}_G = \|\mathbb{E}[\mathbf{f}(\mathbf{x}_{real})] - \mathbb{E}[\mathbf{f}(G(\mathbf{z}))]\|_2^2$$

**Spectral Normalization**: Stabilize training to prevent mode collapse.

**Diversity Regularization**: Explicitly encourage diversity:
$$\mathcal{L}_G = \mathcal{L}_{adv} + \lambda \cdot \text{diversity}(G)$$

---

## Wasserstein Distance and WGAN

Wasserstein GAN (WGAN) uses Wasserstein distance instead of Jensen-Shannon divergence, providing better training stability.

### Wasserstein Distance

**Definition**: For distributions $P_r$ and $P_g$:

$$W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x-y\|]$$

where $\Pi(P_r, P_g)$ is the set of couplings.

**Intuition**: Minimum cost to transport mass from $P_r$ to $P_g$.

### Kantorovich-Rubinstein Duality

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

where $\|f\|_L \leq 1$ means $f$ is 1-Lipschitz.

### WGAN Objective

Replace discriminator with critic (no sigmoid):

$$\min_G \max_{\|D\|_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{data}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z} \sim p_z}[D(G(\mathbf{z}))]$$

**Critic** $D$: Real-valued function (not a classifier).

**Lipschitz Constraint**: Enforce $\|D\|_L \leq 1$ through weight clipping or spectral normalization.

### Weight Clipping

Clip critic weights to $[-c, c]$:

$$w \leftarrow \text{clip}(w, -c, c)$$

**Issues**:
- May limit capacity
- Can cause vanishing/exploding gradients
- Not optimal way to enforce Lipschitz

### WGAN-GP (Gradient Penalty)

Penalize gradient norm instead of clipping:

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2]$$

where $\hat{\mathbf{x}}$ is sampled along lines between real and fake samples.

### Advantages

1. **Stable training**: More stable than standard GAN
2. **Meaningful loss**: Wasserstein distance correlates with sample quality
3. **No mode collapse**: Better diversity than standard GAN
4. **Progressive training**: Can train to convergence

---

## Spectral Normalization

Spectral normalization stabilizes GAN training by constraining the Lipschitz constant of the discriminator.

### Spectral Norm

For matrix $W$, spectral norm:

$$\sigma(W) = \max_{\|\mathbf{h}\| \neq 0} \frac{\|W\mathbf{h}\|}{\|\mathbf{h}\|} = \max_{\|\mathbf{h}\|=1} \|W\mathbf{h}\|$$

Equals the largest singular value of $W$.

### Spectral Normalization

Normalize weight matrix:

$$\bar{W} = \frac{W}{\sigma(W)}$$

Ensures $\sigma(\bar{W}) = 1$.

### Computing Spectral Norm

**Power Iteration**: Approximate largest singular value:

1. Initialize $\mathbf{u}_0$, $\mathbf{v}_0$
2. For $t = 1, \ldots, T$:
   $$\mathbf{v}_t = \frac{W^T \mathbf{u}_{t-1}}{\|W^T \mathbf{u}_{t-1}\|}$$
   $$\mathbf{u}_t = \frac{W \mathbf{v}_t}{\|W \mathbf{v}_t\|}$$
3. $\sigma(W) \approx \mathbf{u}_T^T W \mathbf{v}_T$

One iteration is often sufficient during training.

### Lipschitz Constant

For neural network $f$ with layers $W_1, \ldots, W_L$:

$$\|f\|_L \leq \prod_{i=1}^{L} \sigma(W_i)$$

Spectral normalization bounds each layer's spectral norm to 1.

### Application to GANs

Apply spectral normalization to discriminator:

$$D(\mathbf{x}) = \text{SN}(W_L) \cdot \sigma(\cdots \text{SN}(W_2) \cdot \sigma(\text{SN}(W_1) \mathbf{x}))$$

**Benefits**:
- Stabilizes training
- Prevents discriminator from becoming too strong
- Enables deeper networks
- Works with various architectures

---

## Progressive Training and StyleGAN

Progressive GAN and StyleGAN train generators by gradually increasing resolution, improving stability and quality.

### Progressive GAN

**Progressive Growing**:
1. Start with low resolution (4×4)
2. Train generator and discriminator
3. Gradually add layers for higher resolution
4. Fade in new layers smoothly

**Fade-in**: Linearly interpolate between old and new layers:

$$\text{output} = \alpha \cdot \text{new} + (1-\alpha) \cdot \text{old}$$

where $\alpha$ increases from 0 to 1.

### Advantages

1. **Stability**: Easier to train at low resolution
2. **Quality**: Higher resolution details build on stable base
3. **Efficiency**: Faster training at early stages

### StyleGAN Architecture

StyleGAN separates style (latent code) from content (noise):

**Mapping Network**: Maps $\mathbf{z}$ to intermediate latent $\mathbf{w}$:
$$\mathbf{w} = f(\mathbf{z})$$

**Synthesis Network**: Generates image from $\mathbf{w}$ and noise:
$$\mathbf{x} = G(\mathbf{w}, \mathbf{n})$$

### Adaptive Instance Normalization (AdaIN)

Style information injected via AdaIN:

$$\text{AdaIN}(\mathbf{x}, \mathbf{y}) = \sigma(\mathbf{y}) \cdot \frac{\mathbf{x} - \mu(\mathbf{x})}{\sigma(\mathbf{x})} + \mu(\mathbf{y})$$

where $\mathbf{y}$ encodes style information.

### Style Mixing

Interpolate between different $\mathbf{w}$ at different resolutions:

$$\mathbf{w}_{mixed} = [\mathbf{w}_1^{(1:k)}, \mathbf{w}_2^{(k+1:L)}]$$

Enables control over different levels of detail.

### StyleGAN2 Improvements

1. **Removed progressive growing**: Use residual connections instead
2. **Lazy regularization**: Compute regularization less frequently
3. **Path length regularization**: Encourage smooth latent space
4. **Noise injection**: Learned per-channel scaling

---

## Evaluation Metrics

Evaluating GANs is challenging due to the lack of explicit likelihood. Multiple metrics provide complementary insights.

### Inception Score (IS)

**Definition**:
$$\text{IS} = \exp(\mathbb{E}[\text{KL}(p(y|\mathbf{x}) \| p(y))])$$

where $p(y|\mathbf{x})$ is Inception network's class distribution.

**Interpretation**:
- High IS: Generated images are clear and diverse
- Requires: Pre-trained classifier (Inception)

**Limitations**:
- Only measures ImageNet classes
- Doesn't detect mode collapse well
- Sensitive to implementation details

### Fréchet Inception Distance (FID)

**Definition**:
$$\text{FID} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|_2^2 + \text{tr}(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2})$$

where $(\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r)$ and $(\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g)$ are statistics of real and generated features.

**Advantages**:
- More robust than IS
- Detects mode collapse
- Correlates with human judgment

**Limitations**:
- Requires feature extractor
- May not capture all aspects of quality

### Precision and Recall

**Precision**: Quality of generated samples
$$\text{Precision} = \frac{|\{\mathbf{x}_g : \min_{\mathbf{x}_r} d(\mathbf{x}_g, \mathbf{x}_r) < \tau\}|}{|\{\mathbf{x}_g\}|}$$

**Recall**: Coverage of data distribution
$$\text{Recall} = \frac{|\{\mathbf{x}_r : \min_{\mathbf{x}_g} d(\mathbf{x}_r, \mathbf{x}_g) < \tau\}|}{|\{\mathbf{x}_r\}|}$$

### Kernel Inception Distance (KID)

Uses polynomial kernel instead of Gaussian:

$$\text{KID} = \mathbb{E}[k(\mathbf{x}_r, \mathbf{x}_r')] + \mathbb{E}[k(\mathbf{x}_g, \mathbf{x}_g')] - 2\mathbb{E}[k(\mathbf{x}_r, \mathbf{x}_g)]$$

More robust for small sample sizes.

### Human Evaluation

**Mean Opinion Score (MOS)**: Human raters score quality.

**Two-Alternative Forced Choice (2AFC)**: Choose better of two samples.

**Advantages**: Most reliable measure

**Disadvantages**: Expensive, time-consuming, subjective

---

## Advanced GAN Architectures

### Conditional GAN (cGAN)

Condition generator and discriminator on additional information:

$$\min_G \max_D \mathbb{E}[\log D(\mathbf{x}, \mathbf{c})] + \mathbb{E}[\log(1-D(G(\mathbf{z}, \mathbf{c}), \mathbf{c}))]$$

Enables controlled generation.

### InfoGAN

Maximize mutual information between latent code and generated samples:

$$\mathcal{L} = V(D, G) - \lambda I(\mathbf{c}; G(\mathbf{z}, \mathbf{c}))$$

Learns interpretable latent codes.

### BigGAN

Large-scale GAN with:
- Large batch size (2048)
- Truncated normal prior
- Orthogonal regularization
- Class-conditional batch normalization

### Self-Attention GAN (SAGAN)

Adds self-attention to generator and discriminator:

$$\mathbf{y}_j = \sum_{i} \alpha_{ij} \mathbf{x}_i$$

where $\alpha_{ij} = \text{softmax}(f(\mathbf{x}_i)^T g(\mathbf{x}_j))$.

Captures long-range dependencies.

### CycleGAN

Unpaired image-to-image translation:

$$\mathcal{L} = \mathcal{L}_{GAN}(G, D_Y) + \mathcal{L}_{GAN}(F, D_X) + \lambda \mathcal{L}_{cycle}(G, F)$$

where $\mathcal{L}_{cycle}$ enforces cycle consistency: $F(G(\mathbf{x})) \approx \mathbf{x}$.

---

## Key Takeaways

1. **Adversarial Framework**: GANs train generator and discriminator in minimax game, where generator learns to fool discriminator and discriminator learns to distinguish real from fake.

2. **Training Dynamics**: Alternating updates between generator and discriminator require careful balancing. Non-saturating loss prevents gradient vanishing when discriminator is confident.

3. **Nash Equilibrium**: GAN training seeks Nash equilibrium in two-player game, but standard gradient descent may not converge due to non-convexity and non-zero-sum dynamics.

4. **Mode Collapse**: Generator may produce limited diversity, focusing on few modes. Solutions include unrolled GANs, mini-batch discrimination, and feature matching.

5. **Wasserstein GAN**: WGAN uses Wasserstein distance instead of JS divergence, providing more stable training, meaningful loss, and better mode coverage through Lipschitz-constrained critic.

6. **Spectral Normalization**: Constraining spectral norm of discriminator weights stabilizes training, prevents discriminator from becoming too strong, and enables deeper networks.

7. **Progressive Training**: Progressive GAN and StyleGAN gradually increase resolution, improving stability. StyleGAN separates style from content using mapping network and AdaIN.

8. **Evaluation Metrics**: IS measures quality and diversity, FID compares distributions more robustly, precision/recall measure quality and coverage. Human evaluation remains gold standard.

9. **Advanced Architectures**: Conditional GANs enable controlled generation, InfoGAN learns interpretable latents, BigGAN scales to high resolution, SAGAN captures long-range dependencies.

10. **Challenges**: Training instability, mode collapse, and evaluation difficulties remain active research areas. Careful architecture design, regularization, and monitoring are essential for successful GAN training.
