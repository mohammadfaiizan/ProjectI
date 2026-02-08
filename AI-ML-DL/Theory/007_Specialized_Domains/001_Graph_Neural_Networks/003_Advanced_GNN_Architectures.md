# Advanced GNN Architectures

## Table of Contents

1. [Graph Attention Networks](#graph-attention-networks)
2. [Graph Transformers](#graph-transformers)
3. [Spectral vs Spatial Methods](#spectral-vs-spatial-methods)
4. [Heterogeneous Graph Neural Networks](#heterogeneous-graph-neural-networks)
5. [Dynamic and Temporal Graphs](#dynamic-and-temporal-graphs)
6. [Graph Generation Models](#graph-generation-models)
7. [Expressive Power and Theoretical Limits](#expressive-power-and-theoretical-limits)
8. [Scalability and Efficiency](#scalability-and-efficiency)
9. [Recent Advances](#recent-advances)
10. [Key Takeaways](#key-takeaways)

---

## Graph Attention Networks

Graph Attention Networks (GAT) introduce attention mechanisms to GNNs, enabling nodes to adaptively weight contributions from different neighbors.

### Attention Mechanism

The attention coefficient $\alpha_{ij}$ measures the importance of node $j$'s features to node $i$:

$$e_{ij} = \text{LeakyReLU}(a^T [W h_i \| W h_j])$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$$

where $a \in \mathbb{R}^{2d'}$ is a learnable attention vector, $W \in \mathbb{R}^{d' \times d}$ is a weight matrix, and $\|$ denotes concatenation.

### GAT Layer

The node update combines attention-weighted neighbor features:

$$h_i^{(l+1)} = \sigma\left(\sum_{j \in N(i)} \alpha_{ij} W^{(l)} h_j^{(l)}\right)$$

**Properties**:
- **Adaptive**: Attention weights adapt to different neighbors
- **Implicitly specifying different weights**: No need for explicit edge weights
- **Computationally efficient**: $O(|V| + |E|)$ complexity

### Multi-Head Attention

Multi-head attention aggregates information from multiple attention mechanisms:

$$h_i^{(l+1)} = \|_{k=1}^{K} \sigma\left(\sum_{j \in N(i)} \alpha_{ij}^{(k)} W^{(k)} h_j^{(l)}\right)$$

where $K$ is the number of attention heads and $\|$ denotes concatenation.

For the final layer, averaging is used instead of concatenation:

$$h_i^{(L)} = \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in N(i)} \alpha_{ij}^{(k)} W^{(k)} h_j^{(L-1)}$$

### Advantages

1. **Interpretability**: Attention weights reveal which neighbors are important
2. **Flexibility**: Can handle different types of relationships
3. **Performance**: Often outperforms GCN on node classification tasks

### Limitations

1. **Over-smoothing**: Still suffers from over-smoothing in deep networks
2. **Computational cost**: Higher than GCN due to attention computation
3. **Sparse attention**: Only attends to immediate neighbors, not global structure

---

## Graph Transformers

Graph Transformers extend the Transformer architecture to graphs, enabling global attention and long-range dependencies.

### Self-Attention on Graphs

Standard self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

For graphs, we mask attention to respect graph structure:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

where $M_{ij} = -\infty$ if $(i,j) \notin E$ and $M_{ij} = 0$ otherwise.

### Graph Transformer Architecture

A Graph Transformer layer consists of:

1. **Multi-head self-attention**:
$$H' = \text{LayerNorm}(H + \text{MHA}(H, H, H))$$

2. **Feed-forward network**:
$$H^{(l+1)} = \text{LayerNorm}(H' + \text{FFN}(H'))$$

where $\text{MHA}$ is multi-head attention and $\text{FFN}$ is a feed-forward network.

### Positional Encoding

Unlike sequences, graphs lack natural ordering. Graph Transformers use:

**Spatial Encoding**: Distance-based encoding:
$$PE_{ij} = f(d(i,j))$$

where $d(i,j)$ is the shortest path distance.

**Laplacian Eigenvectors**: Use eigenvectors of the graph Laplacian:
$$PE_i = [u_1(i), u_2(i), \ldots, u_k(i)]$$

**Random Walk Encoding**: Encode random walk probabilities:
$$PE_i = [RW_{i1}, RW_{i2}, \ldots, RW_{ik}]$$

where $RW_{ij}$ is the probability of reaching node $j$ from $i$ via random walk.

### Graphormer

Graphormer introduces three structural encodings:

1. **Centrality Encoding**: Node degree as positional encoding
2. **Spatial Encoding**: Shortest path distance between nodes
3. **Edge Encoding**: Edge features along shortest paths

The attention becomes:

$$A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(i,j)} + c_{ij}$$

where $b_{\phi(i,j)}$ is the spatial encoding and $c_{ij}$ is the edge encoding.

### Advantages

1. **Global attention**: Can attend to distant nodes
2. **Long-range dependencies**: Captures relationships beyond local neighborhoods
3. **Flexibility**: Can handle variable graph structures

### Challenges

1. **Quadratic complexity**: $O(|V|^2)$ attention computation
2. **Memory requirements**: Large graphs require significant memory
3. **Positional encoding**: Designing effective encodings is non-trivial

---

## Spectral vs Spatial Methods

Understanding the relationship between spectral and spatial approaches provides insights into GNN design.

### Spectral Graph Convolution

Spectral methods operate in the frequency domain using the graph Fourier transform:

$$(f *_\mathcal{G} g)(v) = U \text{diag}(\hat{g}(\lambda_1), \ldots, \hat{g}(\lambda_n)) U^T f(v)$$

where $U$ contains Laplacian eigenvectors and $\hat{g}$ is a spectral filter.

**ChebNet**: Approximates spectral filters using Chebyshev polynomials:

$$g_\theta(L) = \sum_{k=0}^{K} \theta_k T_k(\tilde{L})$$

where $\tilde{L} = \frac{2L}{\lambda_{max}} - I$ and $T_k$ are Chebyshev polynomials.

**GCN**: First-order approximation of ChebNet with $K=1$ and $\lambda_{max} \approx 2$:

$$g_\theta(L) \approx \theta_0 I + \theta_1 (L - I) = \theta(I - D^{-1/2} A D^{-1/2})$$

### Spatial Graph Convolution

Spatial methods define convolution directly on graph structure:

$$(f * g)(v) = \sum_{u \in N(v)} f(u) g(u,v)$$

This is more intuitive and computationally efficient.

### Unified Framework

Modern GNNs can be viewed through both lenses:

- **GCN**: Spatial (local aggregation) or spectral (first-order ChebNet)
- **GAT**: Spatial (attention-weighted aggregation) with implicit spectral filtering
- **Graph Transformer**: Spatial (masked self-attention) with global receptive field

### Trade-offs

| Aspect | Spectral | Spatial |
|--------|----------|---------|
| Interpretability | Frequency domain | Local neighborhoods |
| Computational cost | Eigendecomposition expensive | Efficient for sparse graphs |
| Transferability | Graph-specific | More transferable |
| Expressiveness | Can design filters | Limited by aggregation |

---

## Heterogeneous Graph Neural Networks

Heterogeneous graphs contain multiple node types and edge types, requiring specialized architectures.

### Heterogeneous Graph Definition

A heterogeneous graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with node type mapping $\phi: \mathcal{V} \rightarrow \mathcal{A}$ and edge type mapping $\psi: \mathcal{E} \rightarrow \mathcal{R}$.

### Relational Graph Convolutional Network (R-GCN)

R-GCN handles multiple relation types:

$$h_i^{(l+1)} = \sigma\left(\sum_{r \in \mathcal{R}} \sum_{j \in N_r(i)} \frac{1}{|N_r(i)|} W_r^{(l)} h_j^{(l)} + W_0^{(l)} h_i^{(l)}\right)$$

where $N_r(i)$ are neighbors via relation $r$, and $W_r^{(l)}$ are relation-specific weights.

### Heterogeneous Graph Transformer (HGT)

HGT extends transformers to heterogeneous graphs:

**Relation-aware attention**:
$$e_{ij}^{(r)} = \frac{(h_i W_Q^{(r)})(h_j W_K^{(r)})^T}{\sqrt{d}}$$

$$\alpha_{ij}^{(r)} = \frac{\exp(e_{ij}^{(r)})}{\sum_{r' \in \mathcal{R}} \sum_{k \in N_{r'}(i)} \exp(e_{ik}^{(r')})}$$

**Type-specific transformation**:
$$h_i^{(l+1)} = \|_{r \in \mathcal{R}} \sum_{j \in N_r(i)} \alpha_{ij}^{(r)} W_V^{(r)} h_j^{(l)}$$

### Meta-Path Based Methods

Meta-paths define composite relationships:

**HAN (Heterogeneous Attention Network)**: Uses meta-path based attention:

$$h_i^{(l+1)} = \|_{p \in \mathcal{P}} \sigma\left(\sum_{j \in N_p(i)} \alpha_{ij}^{(p)} h_j^{(l)}\right)$$

where $\mathcal{P}$ is a set of meta-paths and $N_p(i)$ are neighbors via meta-path $p$.

### Challenges

1. **Relation imbalance**: Different relation types have varying frequencies
2. **Meta-path selection**: Choosing relevant meta-paths requires domain knowledge
3. **Scalability**: Multiple relation types increase computational cost

---

## Dynamic and Temporal Graphs

Dynamic graphs evolve over time, requiring models that capture temporal dependencies.

### Temporal Graph Definition

A temporal graph $\mathcal{G}(t) = (\mathcal{V}(t), \mathcal{E}(t))$ where nodes and edges can appear/disappear over time $t$.

### Temporal GCN (T-GCN)

T-GCN combines GCN with LSTM:

$$H^{(l)}(t) = \text{GCN}(A(t), H^{(l)}(t))$$
$$H^{(l+1)}(t) = \text{LSTM}(H^{(l)}(t), H^{(l+1)}(t-1))$$

### EvolveGCN

EvolveGCN adapts GCN parameters over time:

$$W^{(l)}(t) = \text{GRU}(W^{(l)}(t-1), H^{(l)}(t))$$
$$H^{(l+1)}(t) = \text{GCN}(A(t), H^{(l)}(t), W^{(l)}(t))$$

### Temporal Graph Attention

Extend GAT to temporal graphs:

$$h_i^{(l+1)}(t) = \sigma\left(\sum_{j \in N(i,t)} \alpha_{ij}(t) W^{(l)} h_j^{(l)}(t) + U^{(l)} h_i^{(l)}(t-1)\right)$$

where $\alpha_{ij}(t)$ is time-dependent attention.

### Continuous-Time Models

Model graphs as continuous-time processes:

**Neural ODE for Graphs**: Use neural ordinary differential equations:

$$\frac{dH(t)}{dt} = f_\theta(H(t), A(t), t)$$

**Graph Neural ODE**: Combine GNN with ODE:

$$\frac{dh_i(t)}{dt} = \sum_{j \in N(i)} \text{MSG}(h_i(t), h_j(t), t)$$

### Applications

- **Social networks**: Friend connections over time
- **Citation networks**: Paper citations evolving
- **Traffic networks**: Road conditions changing
- **Financial networks**: Transaction patterns

---

## Graph Generation Models

Graph generation models learn to generate realistic graph structures, enabling drug discovery, molecule design, and network synthesis.

### Autoregressive Generation

Generate graphs node by node:

**GraphRNN**: Models graph generation as a sequence:

$$p(G) = \prod_{i=1}^{n} p(S_i | S_{<i})$$

where $S_i$ is the adjacency vector for node $i$.

**GraphRNN Architecture**:
1. Node-level RNN: Generates nodes sequentially
2. Edge-level RNN: For each node, generates edges to previous nodes

### Variational Graph Autoencoders

Extend VAE to graphs:

**Encoder**: $q(Z | G) = \prod_{i=1}^{n} q(z_i | G)$

**Decoder**: $p(G | Z) = \prod_{i,j} p(A_{ij} | z_i, z_j)$

**Loss**:
$$\mathcal{L} = \mathbb{E}_{q(Z|G)}[\log p(G|Z)] - \text{KL}(q(Z|G) \| p(Z))$$

### Graph Generative Adversarial Networks

**GraphGAN**: Adapts GAN to graphs:

**Generator**: $G(z)$ generates adjacency matrix $\hat{A}$

**Discriminator**: $D(A)$ distinguishes real from fake graphs

**Loss**:
$$\min_G \max_D \mathbb{E}[\log D(A)] + \mathbb{E}[\log(1-D(G(z)))]$$

### Diffusion Models for Graphs

Extend diffusion models to graph generation:

**Forward process**: Gradually add noise to graph structure

**Reverse process**: Learn to denoise and generate graphs

**Graph Diffusion Model**:
$$q(A_t | A_{t-1}) = \mathcal{N}(A_t; \sqrt{1-\beta_t} A_{t-1}, \beta_t I)$$

$$p_\theta(A_{t-1} | A_t) = \mathcal{N}(A_{t-1}; \mu_\theta(A_t, t), \Sigma_\theta(A_t, t))$$

### Evaluation Metrics

- **Degree distribution**: Compare generated vs real degree distributions
- **Clustering coefficient**: Measure local structure preservation
- **Orbit counts**: Count graphlet frequencies
- **MMD (Maximum Mean Discrepancy)**: Statistical distance between distributions

---

## Expressive Power and Theoretical Limits

Understanding the theoretical limits of GNNs guides architecture design.

### Weisfeiler-Lehman Test

The WL test provides a hierarchy for GNN expressiveness:

**1-WL (Color Refinement)**: 
$$c^{(l+1)}(v) = \text{hash}(c^{(l)}(v), \{c^{(l)}(u) : u \in N(v)\})$$

**k-WL**: More powerful, considers k-tuples of nodes

### GNN Expressiveness

**Theorem**: Message-passing GNNs are at most as expressive as 1-WL test.

**Corollary**: GNNs cannot distinguish non-isomorphic graphs that 1-WL cannot distinguish.

### Higher-Order GNNs

**k-GNNs**: Use k-tuples instead of nodes:

$$h_S^{(l+1)} = \text{UPD}(h_S^{(l)}, \text{AGG}(\{h_{S'}^{(l)} : S' \in N_k(S)\}))$$

where $S$ is a k-tuple and $N_k(S)$ are k-tuple neighbors.

**k-GNNs are as expressive as k-WL test**.

### Invariant Graph Networks

**IGN (Invariant Graph Networks)**: Use tensor representations:

$$H^{(l+1)} = \sigma(B^{(l)} H^{(l)} W^{(l)})$$

where $B^{(l)}$ is a basis of permutation-equivariant linear maps.

**IGNs can approximate any permutation-invariant function**.

### Limitations

1. **Local structure**: Standard GNNs only see local neighborhoods
2. **Counting substructures**: Limited ability to count triangles, cycles
3. **Long-range dependencies**: Difficulty capturing distant relationships

---

## Scalability and Efficiency

Scaling GNNs to large graphs requires efficient algorithms and architectures.

### Sampling Strategies

**Node Sampling**: Sample nodes and their neighborhoods

**Layer Sampling**: Sample different neighbors at each layer

**Subgraph Sampling**: Sample connected subgraphs

**FastGCN**: Samples nodes independently at each layer:

$$h_i^{(l+1)} = \sigma\left(\frac{|V|}{|S^{(l)}|} \sum_{j \in S^{(l)}} \frac{A_{ij}}{\sqrt{d_i d_j}} W^{(l)} h_j^{(l)}\right)$$

### Graph Coarsening

Reduce graph size while preserving structure:

**Graclus**: Greedy clustering-based coarsening

**Diffusion Wavelets**: Spectral coarsening

**Learnable Coarsening**: Learn optimal coarsening strategy

### Distributed Training

**Graph Partitioning**: Partition graph across devices

**Communication**: Minimize inter-device communication

**Synchronization**: Synchronize gradients across partitions

### Approximate Methods

**Nyström Method**: Approximate kernel matrices

**Random Features**: Use random Fourier features

**Sketching**: Use sketching techniques for large matrices

### Hardware Acceleration

- **GPU**: Parallelize neighborhood aggregation
- **TPU**: Optimize for matrix operations
- **Specialized hardware**: Graph processing units (GPUs)

---

## Recent Advances

### Graph Structure Learning

Learn optimal graph structure:

$$\min_{A, \theta} \mathcal{L}(f_\theta(A, X), Y) + \lambda R(A)$$

where $R(A)$ regularizes graph structure.

### Self-Supervised Learning

**Pretext Tasks**:
- Node masking: Predict masked node features
- Edge prediction: Predict missing edges
- Contrastive learning: Contrast positive vs negative pairs

**Graph Contrastive Learning**:
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{j} \exp(\text{sim}(z_i, z_j)/\tau)}$$

### Pre-training and Transfer Learning

**Pre-training**: Train on large unlabeled graphs

**Fine-tuning**: Adapt to downstream tasks

**Transfer Learning**: Transfer knowledge across domains

### Explainability

**GNNExplainer**: Identify important subgraphs:

$$\max_{G_S} MI(Y, G_S) = H(Y) - H(Y | G_S)$$

**Attention Visualization**: Visualize attention weights

**Gradient-based Methods**: Use gradients to identify important nodes/edges

---

## Key Takeaways

1. **Graph Attention Networks**: GAT introduces adaptive neighbor weighting through attention mechanisms, providing interpretability and often improved performance over GCN.

2. **Graph Transformers**: Extend Transformers to graphs with masked self-attention and positional encodings, enabling global attention and long-range dependencies at quadratic cost.

3. **Spectral vs Spatial**: Spectral methods operate in frequency domain while spatial methods work directly on graph structure. Modern GNNs blur this distinction.

4. **Heterogeneous Graphs**: R-GCN and HGT handle multiple node and edge types through relation-specific transformations and attention mechanisms.

5. **Dynamic Graphs**: T-GCN and EvolveGCN combine GNNs with RNNs to model temporal evolution, while continuous-time models use neural ODEs.

6. **Graph Generation**: Autoregressive (GraphRNN), VAE-based, GAN-based, and diffusion models enable generation of realistic graph structures for drug discovery and network synthesis.

7. **Theoretical Limits**: Message-passing GNNs are limited by 1-WL expressiveness. Higher-order GNNs (k-GNNs) and IGNs achieve greater expressiveness at higher cost.

8. **Scalability**: Sampling strategies (node, layer, subgraph), graph coarsening, distributed training, and approximate methods enable scaling to large graphs.

9. **Self-Supervised Learning**: Pre-training on large unlabeled graphs and contrastive learning enable transfer learning and improved performance on downstream tasks.

10. **Future Directions**: Graph structure learning, explainability, and hardware acceleration continue to advance the field, enabling applications to increasingly complex and large-scale problems.
