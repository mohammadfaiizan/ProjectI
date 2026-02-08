# Federated Learning and Distributed Machine Learning

## Table of Contents

1. [Introduction](#introduction)
2. [Federated Learning Fundamentals](#federated-learning-fundamentals)
3. [Federated Averaging (FedAvg)](#federated-averaging-fedavg)
4. [Communication Efficiency](#communication-efficiency)
5. [Non-IID Data Challenges](#non-iid-data-challenges)
6. [Privacy Guarantees and Differential Privacy](#privacy-guarantees-and-differential-privacy)
7. [Secure Aggregation](#secure-aggregation)
8. [Vertical and Horizontal Federated Learning](#vertical-and-horizontal-federated-learning)
9. [Split Learning](#split-learning)
10. [Key Takeaways](#key-takeaways)

## Introduction

Federated learning enables training machine learning models across decentralized data without centralizing the data. Instead of sending data to a central server, models are trained locally on client devices, and only model updates are shared. This paradigm addresses privacy concerns, reduces communication costs, and enables learning from data that cannot be centralized.

Federated learning is particularly relevant for mobile devices, IoT sensors, healthcare systems, and other scenarios where data privacy is critical or data cannot leave local devices. The field addresses challenges including statistical heterogeneity, communication efficiency, privacy, and security.

Key research questions:
- How to aggregate updates from heterogeneous clients?
- How to reduce communication costs?
- How to handle non-IID data distributions?
- How to guarantee privacy and security?

## Federated Learning Fundamentals

Federated learning differs from traditional distributed learning in several key ways.

### Key Characteristics

**Data distribution**: Data remains on client devices
**Model distribution**: Model or updates are communicated
**Heterogeneity**: Clients have different data distributions, compute, and network
**Privacy**: Raw data never leaves client devices

### System Architecture

**Server**: Coordinates training, aggregates updates
**Clients**: Hold local data, train models locally
**Communication**: Exchange model parameters or gradients

### Workflow

1. **Initialization**: Server initializes global model
2. **Selection**: Server selects subset of clients
3. **Distribution**: Server sends current model to clients
4. **Local training**: Clients train on local data
5. **Aggregation**: Clients send updates to server
6. **Update**: Server aggregates updates to form new global model
7. **Repeat**: Iterate until convergence

### Challenges

**Statistical heterogeneity**: Non-IID data across clients
**System heterogeneity**: Different compute and network capabilities
**Privacy**: Need to protect client data
**Communication**: Minimize communication rounds and data transferred

## Federated Averaging (FedAvg)

FedAvg is the foundational algorithm for federated learning, aggregating client updates through weighted averaging.

### Algorithm

**Notation**:
- $K$: Total number of clients
- $n_k$: Number of samples on client $k$
- $n = \sum_k n_k$: Total samples
- $w_t$: Global model at round $t$
- $w_t^k$: Local model on client $k$ at round $t$

**FedAvg**:
1. Initialize $w_0$
2. For each round $t = 1, 2, ...$:
   a. Sample subset $S_t$ of clients
   b. For each client $k \in S_t$:
      - $w_{t+1}^k \leftarrow$ Train($w_t$, local data, $E$ epochs)
   c. Aggregate: $w_{t+1} \leftarrow \sum_{k \in S_t} \frac{n_k}{n_{S_t}} w_{t+1}^k$

where $n_{S_t} = \sum_{k \in S_t} n_k$.

### Weighted Averaging

**Aggregation**:
$$w_{t+1} = \sum_{k \in S_t} \frac{n_k}{n_{S_t}} w_{t+1}^k$$

**Rationale**: Weight by number of samples (clients with more data contribute more)

### Local Updates

**Multiple epochs**: Clients perform $E$ epochs of local training
**Stochastic gradient descent**: Use SGD or variants
**Batch size**: Can vary across clients

### Convergence

**Conditions**: 
- IID data: Converges under standard assumptions
- Non-IID: May diverge or converge slowly
- Partial participation: Works with subset of clients

### Advantages

**Simple**: Easy to implement
**Effective**: Works well in practice
**Flexible**: Can adapt to different scenarios

### Limitations

**Non-IID**: Performance degrades with non-IID data
**Communication**: Still requires many rounds
**Privacy**: No formal privacy guarantees

## Communication Efficiency

Communication is often the bottleneck in federated learning, motivating methods to reduce communication costs.

### Communication Costs

**Rounds**: Number of communication rounds
**Data per round**: Size of model parameters/gradients
**Total cost**: Rounds × data per round

### Strategies

**Local updates**: Multiple local epochs before communication
**Compression**: Compress model updates
**Sparsification**: Send only important updates
**Quantization**: Reduce precision of updates

### Gradient Compression

**Top-k sparsification**: Send only top-$k$ gradients by magnitude
**Random sparsification**: Randomly select gradients
**Error accumulation**: Accumulate errors for next round

**Update**:
$$\Delta w_{compressed} = \text{TopK}(\Delta w, k)$$

**Decompression**: Clients use compressed updates

### Quantization

**Low precision**: Use fewer bits (e.g., 8-bit instead of 32-bit)
**Stochastic quantization**: Probabilistic rounding
**Gradient quantization**: Quantize gradients before sending

**Example**: 8-bit quantization reduces communication by 4x

### Structured Updates

**Low-rank**: Represent updates as low-rank matrices
**Sketching**: Use sketching techniques
**Subsampling**: Send subset of parameters

### Adaptive Methods

**Adaptive selection**: Select clients based on update importance
**Adaptive compression**: Vary compression based on round
**Adaptive frequency**: Vary communication frequency

## Non-IID Data Challenges

Non-IID (non-independent and identically distributed) data is common in federated learning and poses significant challenges.

### Types of Non-IID

**Label distribution shift**: Different label distributions across clients
**Feature distribution shift**: Different feature distributions
**Quantity imbalance**: Different amounts of data
**Temporal shift**: Data from different time periods

### Impact

**Divergence**: Local models may diverge
**Slow convergence**: Slower convergence or non-convergence
**Poor performance**: Lower final accuracy
**Bias**: Model biased toward clients with more data

### Solutions

**Regularization**: Add regularization to prevent divergence
**Client clustering**: Group similar clients
**Personalization**: Allow client-specific models
**Data augmentation**: Augment local data

### FedProx

**Proximal term**: Add proximal term to local objective
$$\min_w \mathcal{L}_k(w) + \frac{\mu}{2} ||w - w_t||^2$$

where $\mu$ is a hyperparameter.

**Effect**: Prevents local models from diverging too far from global model

### SCAFFOLD

**Control variates**: Use control variates to correct for client drift
**Client and server variates**: Maintain variates for each client and server
**Correction**: Correct local updates using variates

**Update**:
$$w_{t+1}^k = w_t^k - \eta (\nabla \mathcal{L}_k(w_t^k) - c^k + c)$$

where $c^k$ is client variate and $c$ is server variate.

### Clustered Federated Learning

**Clustering**: Group clients with similar data distributions
**Per-cluster models**: Train separate model for each cluster
**Assignment**: Assign clients to clusters

**Advantages**: Better handling of heterogeneity
**Challenges**: How to cluster without seeing data?

## Privacy Guarantees and Differential Privacy

Privacy is a key concern in federated learning, motivating formal privacy guarantees.

### Threat Models

**Honest-but-curious server**: Server follows protocol but tries to learn about clients
**Malicious server**: Server may deviate from protocol
**Malicious clients**: Some clients may be adversarial
**External attacker**: Attacker intercepts communications

### Privacy Leakage

**Model inversion**: Reconstruct training data from model
**Membership inference**: Determine if specific example was in training set
**Property inference**: Infer properties of training data
**Gradient leakage**: Extract information from gradients

### Differential Privacy

**Definition**: Algorithm $M$ is $(\epsilon, \delta)$-differentially private if:
$$P(M(D) \in S) \leq e^\epsilon P(M(D') \in S) + \delta$$

for all datasets $D$, $D'$ differing in one example, and all sets $S$.

**Parameters**:
- $\epsilon$: Privacy budget (smaller = more private)
- $\delta$: Failure probability (typically very small)

### DP-FedAvg

**Gaussian noise**: Add Gaussian noise to aggregated updates
$$w_{t+1} = \sum_k \frac{n_k}{n} w_{t+1}^k + \mathcal{N}(0, \sigma^2 I)$$

**Privacy**: Provides $(\epsilon, \delta)$-DP guarantee
**Trade-off**: More noise → more privacy but lower accuracy

### Local Differential Privacy

**Client-side**: Clients add noise before sending updates
**Stronger privacy**: No trust in server
**Higher noise**: Typically requires more noise

### Privacy Accounting

**Composition**: Track privacy budget across rounds
**Rényi DP**: Tighter privacy accounting
**Moments accountant**: Advanced accounting method

## Secure Aggregation

Secure aggregation enables the server to compute the sum of client updates without seeing individual updates.

### Goal

**Privacy**: Server learns only aggregate, not individual updates
**Correctness**: Aggregate is correct sum of updates
**Efficiency**: Low communication and computation overhead

### Secret Sharing

**Shamir's secret sharing**: Split secret into shares
**Threshold**: Need threshold shares to reconstruct
**Application**: Clients share secret shares of updates

### Homomorphic Encryption

**Encryption**: Encrypt updates before sending
**Computation**: Server computes on encrypted data
**Decryption**: Decrypt only final aggregate

**Challenges**: High computational cost

### Secure Multi-Party Computation

**Protocol**: Multiple parties compute function without revealing inputs
**Application**: Clients jointly compute aggregate
**Challenges**: Communication overhead

### Practical Methods

**Masking**: Clients add random masks that cancel in aggregate
**Pairwise masks**: Pairs of clients share masks
**Dropout resilience**: Handle client dropout

**Algorithm**:
1. Clients generate pairwise masks
2. Clients add masks to updates
3. Server aggregates (masks cancel)
4. Server learns only aggregate

## Vertical and Horizontal Federated Learning

Federated learning scenarios differ based on how data is partitioned.

### Horizontal Federated Learning

**Same features, different samples**: Each client has different examples but same features
**Example**: Multiple hospitals with same patient features
**Typical**: Most common scenario

**Aggregation**: Standard FedAvg works well

### Vertical Federated Learning

**Same samples, different features**: Clients have same examples but different features
**Example**: Bank and e-commerce company with same customers but different features
**Challenge**: Need to align samples without revealing identities

**Approaches**:
- **Secure entity alignment**: Find common samples securely
- **Feature alignment**: Align features across clients
- **Split learning**: Split model across clients

### Federated Transfer Learning

**Different samples and features**: Most general case
**Challenge**: Most difficult scenario
**Approaches**: Transfer learning techniques

## Split Learning

Split learning partitions the model across clients and server, with intermediate activations communicated.

### Architecture

**Client**: Holds first layers of model and data
**Server**: Holds later layers
**Communication**: Activations and gradients

**Forward pass**:
1. Client computes activations up to cut layer
2. Client sends activations to server
3. Server computes rest of forward pass

**Backward pass**:
1. Server computes gradients up to cut layer
2. Server sends gradients to client
3. Client computes gradients for its layers

### Advantages

**Privacy**: Raw data never leaves client
**Efficiency**: Client only runs part of model
**Flexibility**: Can split at different layers

### Challenges

**Communication**: Activations may be large
**Privacy**: Activations may leak information
**Synchronization**: Need to coordinate forward/backward passes

### Variants

**U-shaped split**: Client-server-client (for encoder-decoder)
**Vertically split**: Different clients hold different parts
**Multi-split**: Multiple splits for multiple clients

## Key Takeaways

1. **Federated learning** enables training models across decentralized data without centralizing data, addressing privacy and enabling learning from distributed sources.

2. **Federated Averaging (FedAvg)** aggregates client updates through weighted averaging, serving as the foundational algorithm for federated learning.

3. **Communication efficiency** is critical, with methods including local updates, compression, quantization, and sparsification to reduce communication costs.

4. **Non-IID data** poses significant challenges, with solutions including regularization (FedProx), control variates (SCAFFOLD), and client clustering.

5. **Differential privacy** provides formal privacy guarantees by adding noise to updates, with a trade-off between privacy and accuracy.

6. **Secure aggregation** enables computing aggregates without revealing individual updates, using techniques like secret sharing and masking.

7. **Vertical federated learning** handles scenarios where clients have same samples but different features, requiring secure entity alignment.

8. **Split learning** partitions models across clients and server, with intermediate activations communicated instead of raw data.

9. **Challenges** include statistical and system heterogeneity, privacy-utility trade-offs, communication efficiency, and scalability.

10. **Future directions** include improving handling of non-IID data, developing better privacy-utility trade-offs, reducing communication costs, and expanding to more scenarios.

## References

- McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017
- Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." Foundations and Trends in Machine Learning 14, 1-210
- Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys 2020
- Karimireddy, S. P., et al. (2020). "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020
- Geyer, R. C., et al. (2017). "Differentially Private Federated Learning: A Client Level Perspective." arXiv:1712.07557
- Bonawitz, K., et al. (2017). "Practical Secure Aggregation for Privacy-Preserving Machine Learning." CCS 2017
- Yang, Q., et al. (2019). "Federated Machine Learning: Concept and Applications." ACM TIST 10, 1-19
- Vepakomma, P., et al. (2018). "Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data." arXiv:1812.00564
- Zhao, Y., et al. (2018). "Federated Learning with Non-IID Data." arXiv:1806.00582
- Wang, H., et al. (2020). "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing Magazine 37, 50-60
