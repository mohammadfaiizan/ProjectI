# Privacy, Security, and Data Protection in Machine Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Differential Privacy](#differential-privacy)
3. [Federated Learning](#federated-learning)
4. [Homomorphic Encryption](#homomorphic-encryption)
5. [k-Anonymity, l-Diversity, and t-Closeness](#k-anonymity-l-diversity-and-t-closeness)
6. [GDPR Compliance and Legal Frameworks](#gdpr-compliance-and-legal-frameworks)
7. [Privacy-Preserving Machine Learning Techniques](#privacy-preserving-machine-learning-techniques)
8. [Attacks and Defenses](#attacks-and-defenses)
9. [Implementation Considerations](#implementation-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Privacy and security are fundamental concerns in machine learning systems that process sensitive data. As ML models become more powerful and data-driven, protecting individual privacy while maintaining model utility becomes increasingly important.

**Key challenges:**
- **Data privacy**: Protecting sensitive information in training data
- **Model privacy**: Preventing extraction of training data from models
- **Inference privacy**: Protecting data during model inference
- **Regulatory compliance**: Meeting legal requirements (GDPR, CCPA, etc.)

**Threats:**
- **Membership inference**: Determining if a specific sample was in training set
- **Model inversion**: Reconstructing training data from model outputs
- **Attribute inference**: Inferring sensitive attributes from model predictions
- **Model stealing**: Extracting model parameters or functionality

This chapter covers techniques for privacy-preserving ML, including differential privacy, federated learning, homomorphic encryption, and anonymization methods.

## Differential Privacy

Differential privacy provides a rigorous mathematical framework for privacy protection.

### Definition

A randomized mechanism $\mathcal{M}$ satisfies $(\epsilon, \delta)$-differential privacy if for all neighboring datasets $D$ and $D'$ (differing in one record) and all subsets $S \subseteq \text{Range}(\mathcal{M})$:

$$P(\mathcal{M}(D) \in S) \leq e^\epsilon P(\mathcal{M}(D') \in S) + \delta$$

**Parameters:**
- **$\epsilon$ (epsilon)**: Privacy budget, smaller = more private
  - $\epsilon = 0$: Perfect privacy (but useless)
  - $\epsilon < 1$: Strong privacy
  - $\epsilon > 10$: Weak privacy
- **$\delta$ (delta)**: Failure probability, typically $\delta \ll 1/n$ where $n$ is dataset size
  - $\delta = 0$: Pure differential privacy
  - $\delta > 0$: Approximate differential privacy

### Privacy Loss

**Privacy loss random variable:**
$$L^{(o)}_{\mathcal{M}, D, D'} = \log \frac{P(\mathcal{M}(D) = o)}{P(\mathcal{M}(D') = o)}$$

**Composition:**
- **Sequential composition**: $k$ queries with $\epsilon_i$ each → total $\epsilon = \sum_i \epsilon_i$
- **Parallel composition**: Queries on disjoint data → max $\epsilon_i$
- **Advanced composition**: Tighter bounds using $\delta > 0$

### Mechanisms

**Laplace mechanism:**
For function $f: D \to \mathbb{R}^d$ with sensitivity $\Delta f = \max_{D, D'} \|f(D) - f(D')\|_1$:

$$\mathcal{M}(D) = f(D) + \text{Lap}\left(\frac{\Delta f}{\epsilon}\right)$$

Satisfies $\epsilon$-differential privacy.

**Gaussian mechanism:**
$$\mathcal{M}(D) = f(D) + \mathcal{N}\left(0, \sigma^2 I\right)$$

where $\sigma \geq \frac{\Delta f \sqrt{2\ln(1.25/\delta)}}{\epsilon}$.

Satisfies $(\epsilon, \delta)$-differential privacy.

**Exponential mechanism:**
For quality function $q: D \times \mathcal{R} \to \mathbb{R}$:

$$P(\mathcal{M}(D) = r) \propto \exp\left(\frac{\epsilon q(D, r)}{2 \Delta q}\right)$$

### Sensitivity

**L1 sensitivity:**
$$\Delta_1 f = \max_{D, D'} \|f(D) - f(D')\|_1$$

**L2 sensitivity:**
$$\Delta_2 f = \max_{D, D'} \|f(D) - f(D')\|_2$$

**Sensitivity analysis:**
- Count queries: $\Delta = 1$
- Sum queries: $\Delta = \max |x_i|$
- Mean queries: Depends on dataset size

### Private Machine Learning

**Private SGD:**
Add noise to gradients:

$$g_t = \nabla_\theta L(\theta_t, x_i) + \mathcal{N}(0, \sigma^2 I)$$

Clip gradients to bound sensitivity:
$$g_t = \text{clip}(\nabla_\theta L(\theta_t, x_i), C) + \mathcal{N}(0, \sigma^2 I)$$

**DP-SGD algorithm:**
```
for each batch B:
    for each sample (x_i, y_i) in B:
        g_i = ∇_θ L(θ, x_i, y_i)
        g_i = clip(g_i, C)  # Clip to bound sensitivity
    
    g = (1/|B|) Σ_i g_i
    g = g + N(0, (σC)^2 I)  # Add noise
    
    θ = θ - α g
```

**Privacy accounting:**
Track privacy budget using composition:
- Moments accountant (Rényi differential privacy)
- Privacy filters
- Privacy odometers

### Rényi Differential Privacy

Generalization of differential privacy:

**Rényi divergence:**
$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \mathbb{E}_{x \sim Q} \left[\left(\frac{P(x)}{Q(x)}\right)^\alpha\right]$$

**$(\alpha, \epsilon)$-RDP:**
$$D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \leq \epsilon$$

**Conversion to DP:**
$(\alpha, \epsilon)$-RDP implies $\left(\epsilon + \frac{\log(1/\delta)}{\alpha - 1}, \delta\right)$-DP.

**Advantages:**
- Tighter composition bounds
- Better privacy-utility trade-offs
- Used in DP-SGD implementations

## Federated Learning

Federated learning enables training models on decentralized data without sharing raw data.

### Problem Setup

**Centralized learning:**
- All data at central server
- Privacy concerns
- Data cannot leave devices

**Federated learning:**
- Data stays on devices (clients)
- Only model updates shared
- Privacy-preserving

**Participants:**
- **Clients**: Devices with local data (e.g., phones, hospitals)
- **Server**: Coordinates training, aggregates updates

### Federated Averaging (FedAvg)

**Algorithm:**
```
Server initializes θ_0
for round t = 1 to T:
    # Select subset of clients
    S_t = random subset of clients
    
    # Local training
    for each client k in S_t:
        θ^k_{t+1} = LocalUpdate(θ_t, D_k)
    
    # Aggregation
    θ_{t+1} = Σ_k (n_k / n) θ^k_{t+1}
    # where n_k = |D_k|, n = Σ_k n_k
```

**Local update:**
```
θ^k = θ_t
for epoch = 1 to E:
    for batch B in D_k:
        θ^k = θ^k - α ∇_θ L(θ^k, B)
return θ^k
```

### Challenges

**Statistical heterogeneity:**
- Non-IID data across clients
- Different data distributions
- Performance degradation

**System heterogeneity:**
- Different computational resources
- Varying network speeds
- Device availability

**Communication efficiency:**
- Limited bandwidth
- Many communication rounds needed
- Compression techniques

**Privacy concerns:**
- Model updates may leak information
- Need additional privacy guarantees

### Differential Privacy in Federated Learning

**Local differential privacy:**
Each client adds noise before sending updates:

$$\tilde{g}_k = g_k + \mathcal{N}(0, \sigma^2 I)$$

where $g_k$ is client $k$'s gradient.

**Privacy amplification:**
Subsampling (selecting subset of clients) amplifies privacy:
$$\epsilon_{\text{effective}} = \epsilon \cdot q$$

where $q$ is sampling probability.

**Secure aggregation:**
Use cryptographic protocols to aggregate without server seeing individual updates.

### Advanced Techniques

**FedProx:**
Adds proximal term to handle heterogeneity:

$$L_k(\theta) = L_k(\theta) + \frac{\mu}{2} \|\theta - \theta_t\|^2$$

**SCAFFOLD:**
Uses control variates to correct for client drift.

**FedOpt:**
Server-side optimization (Adam, etc.) instead of simple averaging.

## Homomorphic Encryption

Homomorphic encryption allows computation on encrypted data without decryption.

### Definition

**Homomorphic property:**
For encryption scheme $(E, D)$ and operations $\oplus, \otimes$:

$$E(x) \otimes E(y) = E(x \oplus y)$$

Can compute on ciphertexts and decrypt to get result of computation on plaintexts.

### Types

**Partially homomorphic:**
- Supports one operation (addition or multiplication)
- Examples: Paillier (additive), RSA (multiplicative)

**Somewhat homomorphic:**
- Supports limited operations
- Bounded depth circuits

**Fully homomorphic (FHE):**
- Supports arbitrary computations
- Examples: BGV, BFV, CKKS, TFHE

### Applications in ML

**Private inference:**
- Client encrypts input: $c_x = E(x)$
- Server computes on ciphertext: $c_y = f(c_x)$
- Client decrypts: $y = D(c_y)$

**Private training:**
- Clients encrypt gradients: $c_g = E(g)$
- Server aggregates: $c_{\text{sum}} = \sum_i c_{g_i}$
- Decrypt aggregated gradient

### CKKS Scheme

**Plaintext space:** $\mathbb{C}^{N/2}$ (complex numbers)

**Encoding:**
Real vector → polynomial in $\mathbb{Z}[X]/(X^N + 1)$

**Operations:**
- Addition: Component-wise
- Multiplication: Polynomial multiplication
- Rotation: Cyclic shifts

**Approximate arithmetic:**
- Some noise/error introduced
- Suitable for ML (tolerant to small errors)

### Performance Considerations

**Computational cost:**
- 100-1000× slower than plaintext
- Significant overhead

**Ciphertext expansion:**
- Encrypted data much larger
- Storage and communication overhead

**Practical limitations:**
- Deep networks challenging
- Often combined with other techniques
- Active research area

## k-Anonymity, l-Diversity, and t-Closeness

These are privacy models for anonymized datasets.

### k-Anonymity

**Definition:**
A dataset satisfies $k$-anonymity if each record is indistinguishable from at least $k-1$ other records on quasi-identifiers.

**Quasi-identifiers:**
Attributes that can identify individuals (e.g., ZIP code, age, gender).

**Example:**
| ZIP | Age | Gender | Disease |
|-----|-----|--------|---------|
| 12345 | 30-40 | M | Flu |
| 12345 | 30-40 | M | Flu |
| 12345 | 30-40 | M | Cold |

This satisfies 3-anonymity for quasi-identifiers (ZIP, Age, Gender).

**Achieving k-anonymity:**
- **Generalization**: Replace specific values with ranges
- **Suppression**: Remove identifying information
- **Anatomy**: Separate identifying and sensitive attributes

**Limitations:**
- Vulnerable to homogeneity attacks
- Does not protect against background knowledge
- May lose utility

### l-Diversity

**Definition:**
A $k$-anonymous dataset satisfies $l$-diversity if each equivalence class (group of $k$ identical records) has at least $l$ distinct values for sensitive attributes.

**Example:**
| ZIP | Age | Gender | Disease |
|-----|-----|--------|---------|
| 12345 | 30-40 | M | Flu |
| 12345 | 30-40 | M | Cold |
| 12345 | 30-40 | M | Cancer |

This satisfies 3-anonymity and 3-diversity.

**Types:**
- **Distinct l-diversity**: $l$ distinct values
- **Entropy l-diversity**: Entropy of sensitive attribute ≥ $\log l$
- **Recursive (c, l)-diversity**: Most frequent value appears ≤ $c$ times

**Advantages:**
- Protects against homogeneity attacks
- Better than k-anonymity alone

**Limitations:**
- Still vulnerable to skewness attacks
- Does not consider attribute semantics

### t-Closeness

**Definition:**
A dataset satisfies $t$-closeness if the distribution of sensitive attributes in each equivalence class is within distance $t$ of the overall distribution.

**Earth mover's distance (EMD):**
$$EMD(P, Q) = \inf_{\gamma} \int \int d(x, y) d\gamma(x, y)$$

where $\gamma$ is a coupling of $P$ and $Q$.

**Example:**
If overall disease distribution is uniform, each equivalence class should have similar distribution (within $t$).

**Advantages:**
- Protects against attribute disclosure
- Considers global distribution
- Stronger than l-diversity

**Challenges:**
- Computationally expensive
- May require significant generalization
- Utility loss

## GDPR Compliance and Legal Frameworks

### General Data Protection Regulation (GDPR)

**Key principles:**
- **Lawfulness**: Legal basis for processing
- **Purpose limitation**: Process only for specified purposes
- **Data minimization**: Collect only necessary data
- **Accuracy**: Keep data accurate and up-to-date
- **Storage limitation**: Retain only as long as necessary
- **Integrity and confidentiality**: Secure processing
- **Accountability**: Demonstrate compliance

### Rights of Data Subjects

**Right to access:**
Individuals can request their data.

**Right to rectification:**
Correct inaccurate data.

**Right to erasure ("right to be forgotten"):**
Delete personal data.

**Right to restrict processing:**
Limit how data is used.

**Right to data portability:**
Receive data in machine-readable format.

**Right to object:**
Object to processing.

**Rights related to automated decision-making:**
Including profiling.

### Implications for ML

**Training data:**
- Legal basis for collection
- Consent where required
- Data minimization

**Model training:**
- Purpose limitation
- Transparency
- Right to explanation

**Model deployment:**
- Automated decision-making regulations
- Bias and discrimination
- Human oversight

**Data deletion:**
- Right to erasure
- Machine unlearning
- Removing data from trained models

### Machine Unlearning

**Problem:**
Remove effect of specific data points from trained model.

**Approaches:**
- **Retraining**: Train from scratch without deleted data (expensive)
- **Influence functions**: Approximate effect of removal
- **Differential privacy**: Models trained with DP naturally support unlearning
- **Data sharding**: Train on subsets, remove affected shard

### Other Regulations

**CCPA (California Consumer Privacy Act):**
- Similar to GDPR
- Applies to California residents
- Right to know, delete, opt-out

**HIPAA (Health Insurance Portability and Accountability Act):**
- Healthcare data protection
- PHI (Protected Health Information)
- Strict requirements

**Sector-specific regulations:**
- Financial services
- Education
- Children's privacy (COPPA)

## Privacy-Preserving Machine Learning Techniques

### Secure Multi-Party Computation (MPC)

**Goal:**
Multiple parties compute function on combined data without revealing individual inputs.

**Example:**
Two hospitals compute model on combined patient data without sharing raw data.

**Protocols:**
- Secret sharing
- Garbled circuits
- Oblivious transfer

**Applications:**
- Federated learning with stronger guarantees
- Private aggregation
- Secure model training

### Trusted Execution Environments (TEEs)

**Hardware-based security:**
- Intel SGX, ARM TrustZone
- Isolated execution environment
- Attestation of code execution

**Use cases:**
- Secure model inference
- Privacy-preserving training
- Protecting model parameters

**Limitations:**
- Side-channel attacks
- Performance overhead
- Hardware dependency

### Synthetic Data Generation

**Goal:**
Generate synthetic data that preserves statistical properties but protects privacy.

**Methods:**
- **Differential privacy**: Add noise to statistics
- **Generative models**: GANs, VAEs trained with privacy
- **Bayesian networks**: Learn and sample from distribution

**Evaluation:**
- Utility: Similar statistics to real data
- Privacy: Hard to distinguish from real data
- Downstream performance: Models trained on synthetic data perform well

### Secure Aggregation

**Cryptographic protocols:**
- Homomorphic encryption
- Secret sharing
- Secure aggregation protocols

**Federated learning:**
- Clients encrypt updates
- Server aggregates without decryption
- Decrypt only final aggregate

## Attacks and Defenses

### Membership Inference Attacks

**Goal:**
Determine if specific data point was in training set.

**Attack methods:**
- **Shadow models**: Train models on similar data, learn membership patterns
- **Threshold attack**: Compare confidence scores
- **Gradient-based**: Analyze gradients

**Defenses:**
- Differential privacy
- Regularization
- Membership privacy auditing

### Model Inversion Attacks

**Goal:**
Reconstruct training data from model outputs.

**Attack:**
$$\hat{x} = \arg\max_x P(y | x) \text{ for target } y$$

Optimize input to maximize probability of target output.

**Defenses:**
- Differential privacy
- Output perturbation
- Model compression

### Attribute Inference Attacks

**Goal:**
Infer sensitive attributes from model predictions.

**Example:**
Predict gender from purchase history model.

**Defenses:**
- Remove sensitive attributes from training
- Fairness constraints
- Differential privacy

### Model Stealing

**Goal:**
Extract model parameters or functionality via queries.

**Attack:**
- Query model with many inputs
- Train substitute model on input-output pairs

**Defenses:**
- Rate limiting
- Query obfuscation
- Watermarking

## Implementation Considerations

### Privacy Budget Management

**Tracking:**
- Monitor $\epsilon$ consumption
- Set budgets per user/query
- Enforce limits

**Composition:**
- Use advanced composition theorems
- Optimize budget allocation
- Consider long-term privacy

### Utility-Privacy Trade-offs

**Metrics:**
- Model accuracy
- Privacy loss ($\epsilon$)
- Data utility

**Optimization:**
- Tune noise levels
- Adjust privacy parameters
- Evaluate trade-offs

### Scalability

**Challenges:**
- Large datasets
- Deep models
- Real-time inference

**Solutions:**
- Efficient algorithms
- Approximations
- Hardware acceleration

### Auditing and Verification

**Privacy audits:**
- Verify differential privacy guarantees
- Test for vulnerabilities
- Compliance checks

**Tools:**
- Privacy accounting libraries
- Attack simulators
- Compliance frameworks

## Key Takeaways

1. **Differential privacy provides rigorous guarantees**: Mathematical framework for privacy protection with quantifiable guarantees.

2. **Privacy-utility trade-off is fundamental**: Stronger privacy typically reduces utility; must balance based on use case.

3. **Federated learning enables decentralized training**: Data stays on devices, only model updates shared, improving privacy.

4. **Homomorphic encryption allows computation on encrypted data**: Powerful but computationally expensive, often combined with other techniques.

5. **k-anonymity, l-diversity, t-closeness protect anonymized data**: Progressive privacy models with different strengths and limitations.

6. **GDPR and regulations require compliance**: Legal frameworks impose requirements on data collection, processing, and deletion.

7. **Multiple attacks threaten privacy**: Membership inference, model inversion, attribute inference require defenses.

8. **Privacy-preserving ML is an active area**: New techniques and attacks continuously developed.

9. **Implementation requires careful design**: Privacy budgets, composition, auditing all need attention.

10. **No single solution fits all**: Different techniques appropriate for different scenarios; often combine multiple approaches.
