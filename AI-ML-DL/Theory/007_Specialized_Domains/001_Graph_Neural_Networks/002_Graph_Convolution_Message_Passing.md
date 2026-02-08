# Graph Convolution and Message Passing

## Table of Contents

1. [Introduction to Graph Neural Networks](#introduction-to-graph-neural-networks)
2. [Graph Convolutional Networks (GCN)](#graph-convolutional-networks-gcn)
3. [Message Passing Framework](#message-passing-framework)
4. [GraphSAGE: Sampling and Aggregation](#graphsage-sampling-and-aggregation)
5. [Neighborhood Aggregation Strategies](#neighborhood-aggregation-strategies)
6. [Over-Smoothing Problem](#over-smoothing-problem)
7. [Spatial vs Spectral Approaches](#spatial-vs-spectral-approaches)
8. [Training and Optimization](#training-and-optimization)
9. [Applications and Extensions](#applications-and-extensions)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Graph Neural Networks

Graph Neural Networks (GNNs) extend deep learning to graph-structured data, enabling learning of node, edge, and graph-level representations. Unlike convolutional neural networks for images or recurrent networks for sequences, GNNs must handle variable-sized neighborhoods and irregular graph topology.

### Motivation

Traditional neural networks assume Euclidean structure (grids, sequences), but many real-world problems involve relational data:
- Social networks: users and friendships
- Molecular graphs: atoms and bonds
- Knowledge graphs: entities and relations
- Citation networks: papers and citations

GNNs learn to aggregate information from local neighborhoods, enabling nodes to incorporate contextual information from their graph structure.

### Core Principle

The fundamental principle of GNNs is that a node's representation should depend on:
1. Its own features: $h_v^{(0)} = x_v$
2. Features of neighboring nodes: $\{x_u : u \in N(v)\}$
3. Graph structure: connectivity patterns

This is achieved through iterative message passing, where each layer refines node representations by aggregating information from neighbors.

---

## Graph Convolutional Networks (GCN)

The Graph Convolutional Network (GCN) by Kipf & Welling (2017) provides a simple yet effective framework for semi-supervised node classification.

### GCN Layer

A GCN layer performs the following operation:

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

where:
- $\tilde{A} = A + I$ is the adjacency matrix with self-loops
- $\tilde{D}$ is the degree matrix of $\tilde{A}$: $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- $H^{(l)} \in \mathbb{R}^{|V| \times d_l}$ contains node embeddings at layer $l$
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ is a learnable weight matrix
- $\sigma$ is an activation function (typically ReLU)

### Normalization

The normalization $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ ensures:
1. **Symmetric normalization**: Preserves symmetry of the adjacency matrix
2. **Degree normalization**: Prevents exploding activations in high-degree nodes
3. **Spectral connection**: Related to the normalized Laplacian $L_{sym} = I - \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$

### Node-Level Update

For a single node $v$, the GCN update can be written as:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in N(v) \cup \{v\}} \frac{1}{\sqrt{\deg(v) \deg(u)}} h_u^{(l)} W^{(l)}\right)$$

This aggregates normalized features from neighbors and the node itself.

### Multi-Layer GCN

A $L$-layer GCN computes:

$$H^{(0)} = X$$
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}) \quad \text{for } l = 0, \ldots, L-1$$
$$Z = H^{(L)}$$

The final embeddings $Z$ can be used for node classification, link prediction, or graph classification.

### Loss Function

For semi-supervised node classification:

$$\mathcal{L} = -\sum_{v \in V_{train}} \sum_{c=1}^{C} y_{vc} \log(\text{softmax}(z_v)_c)$$

where $V_{train}$ is the set of labeled nodes, $C$ is the number of classes, and $y_{vc}$ is the one-hot label.

---

## Message Passing Framework

The message passing framework provides a unified view of GNNs, formalizing how information flows through the graph.

### General Message Passing

At each layer $l$, nodes:
1. **Receive messages** from neighbors: $m_u^{(l)} = \text{MSG}(h_u^{(l)})$
2. **Aggregate messages**: $M_v^{(l)} = \text{AGG}(\{m_u^{(l)} : u \in N(v)\})$
3. **Update representation**: $h_v^{(l+1)} = \text{UPD}(h_v^{(l)}, M_v^{(l)})$

### Formal Definition

**Message Function**: $\text{MSG}^{(l)}: \mathbb{R}^{d_l} \rightarrow \mathbb{R}^{d_m}$ transforms node features into messages.

**Aggregation Function**: $\text{AGG}^{(l)}: 2^{\mathbb{R}^{d_m}} \rightarrow \mathbb{R}^{d_a}$ combines messages from neighbors.

**Update Function**: $\text{UPD}^{(l)}: \mathbb{R}^{d_l} \times \mathbb{R}^{d_a} \rightarrow \mathbb{R}^{d_{l+1}}$ combines current state and aggregated messages.

### Common Aggregation Functions

| Aggregation | Formula | Properties |
|------------|---------|------------|
| Mean | $M_v = \frac{1}{|N(v)|} \sum_{u \in N(v)} m_u$ | Permutation invariant, smooth |
| Sum | $M_v = \sum_{u \in N(v)} m_u$ | Permutation invariant, sensitive to neighborhood size |
| Max | $M_v = \max_{u \in N(v)} m_u$ | Permutation invariant, non-smooth |
| Attention | $M_v = \sum_{u \in N(v)} \alpha_{uv} m_u$ | Permutation invariant, adaptive weights |

### Message Passing Variants

**GCN**: 
- MSG: $m_u = h_u W$
- AGG: Mean with normalization
- UPD: $h_v = \sigma(M_v)$

**GraphSAGE**:
- MSG: $m_u = h_u$
- AGG: Learned aggregation (mean/max/LSTM)
- UPD: $h_v = \sigma([h_v \| M_v] W)$

**GAT (Graph Attention Network)**:
- MSG: $m_u = h_u W$
- AGG: Attention-weighted sum
- UPD: $h_v = \sigma(M_v)$

---

## GraphSAGE: Sampling and Aggregation

GraphSAGE (Hamilton et al., 2017) addresses scalability by sampling fixed-size neighborhoods and learning aggregation functions.

### Inductive Learning

Unlike GCN which requires the full graph, GraphSAGE learns inductive node embeddings that generalize to unseen nodes:

$$h_v^{(l+1)} = \sigma(W^{(l)} \cdot [h_v^{(l)} \| \text{AGG}(\{h_u^{(l)} : u \in S(N(v))\})])$$

where $S(\cdot)$ samples a fixed-size subset of neighbors.

### Neighborhood Sampling

For each node $v$ at layer $l$, sample $k$ neighbors:

$$S^{(l)}(N(v)) = \text{Sample}(N(v), k)$$

This enables:
- **Fixed computation**: $O(k^L)$ neighbors per node for $L$ layers
- **Batch training**: Process mini-batches of nodes
- **Scalability**: Handle large graphs with millions of nodes

### Aggregation Functions

GraphSAGE supports multiple aggregation strategies:

**Mean Aggregator**:
$$\text{AGG}_{mean} = \frac{1}{|S(N(v))|} \sum_{u \in S(N(v))} h_u^{(l)}$$

**Max-Pooling Aggregator**:
$$\text{AGG}_{max} = \max(\{\sigma(W_{pool} h_u^{(l)} + b) : u \in S(N(v))\})$$

**LSTM Aggregator**:
$$\text{AGG}_{LSTM} = \text{LSTM}([h_{u_1}^{(l)}, \ldots, h_{u_k}^{(l)}])$$

Note: LSTM aggregator is not permutation invariant but can be made so by randomizing neighbor order.

### Training Procedure

1. Sample a batch of nodes $B$
2. For each node $v \in B$, sample $L$-hop neighborhoods
3. Forward pass through $L$ layers
4. Compute loss using negative sampling:

$$\mathcal{L} = -\log(\sigma(z_v^T z_u)) - Q \cdot \mathbb{E}_{v_n \sim P_n}[\log(\sigma(-z_v^T z_{v_n}))]$$

where $u$ is a positive neighbor and $v_n$ are $Q$ negative samples.

---

## Neighborhood Aggregation Strategies

Different aggregation strategies capture different aspects of neighborhood structure.

### Mean Aggregation

Mean aggregation computes the average of neighbor features:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \left(h_v^{(l)} + \frac{1}{|N(v)|} \sum_{u \in N(v)} h_u^{(l)}\right)\right)$$

**Properties**:
- Smooth and stable
- Treats all neighbors equally
- Sensitive to neighborhood size

### Max Aggregation

Max aggregation selects the maximum element-wise:

$$h_v^{(l+1)} = \sigma\left(W^{(l)} \left(h_v^{(l)} + \max_{u \in N(v)} h_u^{(l)}\right)\right)$$

**Properties**:
- Captures dominant features
- Less sensitive to outliers
- May lose information from multiple neighbors

### Attention-Based Aggregation

Attention mechanisms learn adaptive weights:

$$\alpha_{uv} = \frac{\exp(\text{LeakyReLU}(a^T [W h_v \| W h_u]))}{\sum_{w \in N(v)} \exp(\text{LeakyReLU}(a^T [W h_v \| W h_w]))}$$

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in N(v)} \alpha_{uv} W h_u^{(l)}\right)$$

**Properties**:
- Adaptive to different neighbors
- More expressive than fixed aggregations
- Higher computational cost

### Set Aggregation

Using Deep Sets framework, aggregation should be:
- **Permutation invariant**: Order of neighbors doesn't matter
- **Expressive**: Can represent complex functions

Universal approximation for permutation-invariant functions:

$$f(X) = \rho\left(\sum_{x \in X} \phi(x)\right)$$

where $\phi$ and $\rho$ are MLPs.

---

## Over-Smoothing Problem

Over-smoothing is a critical issue in deep GNNs where node representations become indistinguishable after many layers.

### Definition

After $L$ layers, nodes within $L$ hops receive similar information, causing their representations to converge:

$$\lim_{L \rightarrow \infty} h_v^{(L)} \approx h_u^{(L)} \quad \text{for } d(v,u) \leq L$$

### Analysis

Consider the GCN update:

$$H^{(l+1)} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)}$$

The normalized adjacency matrix $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ has largest eigenvalue 1. Repeated application causes convergence to the principal eigenvector (constant vector for connected graphs).

### Solutions

**1. Residual Connections**:
$$H^{(l+1)} = H^{(l)} + \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

**2. Dense Connections**:
$$H^{(l+1)} = \text{CONCAT}(H^{(0)}, H^{(1)}, \ldots, H^{(l)}) W^{(l)}$$

**3. Jumping Knowledge Networks**:
$$Z = \text{CONCAT}(H^{(1)}, H^{(2)}, \ldots, H^{(L)}) W$$

**4. DropEdge**: Randomly remove edges during training to prevent over-smoothing.

**5. PairNorm**: Normalize node pairs to maintain diversity:

$$\hat{H}^{(l)} = H^{(l)} - \frac{1}{|V|} \mathbf{1} \mathbf{1}^T H^{(l)}$$
$$H^{(l+1)} = s \cdot \frac{\hat{H}^{(l)}}{\|\hat{H}^{(l)}\|_F} \sqrt{|V|}$$

### Theoretical Bounds

For a $d$-regular graph, the mixing time is $O(\log |V|)$. After $O(\log |V|)$ layers, nodes become indistinguishable, limiting GNN depth.

---

## Spatial vs Spectral Approaches

GNNs can be categorized into spatial (local) and spectral (global) approaches.

### Spatial Convolutions

Spatial methods define convolution directly on the graph structure:

$$(f * g)(v) = \sum_{u \in N(v)} f(u) g(u,v)$$

**Advantages**:
- Intuitive and interpretable
- Works on any graph structure
- Computationally efficient

**Examples**: GCN, GraphSAGE, GAT

### Spectral Convolutions

Spectral methods use the graph Fourier transform:

$$(f *_\mathcal{G} g)(v) = \sum_{i=1}^{|V|} \hat{f}(\lambda_i) \hat{g}(\lambda_i) u_i(v)$$

where $u_i$ are Laplacian eigenvectors and $\lambda_i$ are eigenvalues.

**ChebNet**: Uses Chebyshev polynomials to approximate spectral filters:

$$g_\theta(L) = \sum_{k=0}^{K} \theta_k T_k(\tilde{L})$$

where $\tilde{L} = \frac{2L}{\lambda_{max}} - I$ and $T_k$ are Chebyshev polynomials.

**Advantages**:
- Theoretically grounded
- Can design filters in frequency domain

**Disadvantages**:
- Requires eigendecomposition (expensive)
- Not directly transferable to new graphs

### Unified View

Modern GNNs blur the distinction:
- GCN can be viewed as a first-order approximation of ChebNet
- Spatial methods implicitly perform spectral filtering
- Both approaches converge to similar architectures

---

## Training and Optimization

### Batch Training

For large graphs, full-batch training is memory-intensive. Mini-batch training samples subgraphs:

1. Sample a batch of nodes $B$
2. Construct $L$-hop subgraph $G_B$ around $B$
3. Forward pass on $G_B$
4. Compute loss only on $B$

### Negative Sampling

For unsupervised learning, use negative sampling:

$$\mathcal{L} = -\sum_{(u,v) \in E} \log \sigma(z_u^T z_v) - \sum_{(u,v') \notin E} \log \sigma(-z_u^T z_{v'})$$

Sample negative pairs $(u,v')$ where no edge exists.

### Regularization

**Dropout**: Apply dropout to node features or edges:
- Feature dropout: $h_v = \text{dropout}(h_v)$
- Edge dropout: Randomly remove edges during training

**Weight Decay**: L2 regularization on parameters:
$$\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_{l} \|W^{(l)}\|_F^2$$

**Early Stopping**: Monitor validation performance and stop when overfitting occurs.

### Optimization

**Adam Optimizer**: Typically used with learning rate $10^{-3}$ to $10^{-2}$.

**Learning Rate Scheduling**: Reduce learning rate when validation loss plateaus.

**Gradient Clipping**: Prevent exploding gradients:
$$g \leftarrow \min(1, \frac{\tau}{\|g\|}) \cdot g$$

---

## Applications and Extensions

### Node Classification

Predict labels for nodes using semi-supervised learning:

$$\hat{y}_v = \text{softmax}(W_{cls} z_v + b)$$

### Link Prediction

Predict missing edges:

$$p((u,v) \in E) = \sigma(z_u^T z_v)$$

### Graph Classification

Pool node embeddings to graph-level:

$$z_G = \text{READOUT}(\{z_v : v \in V\})$$

Common readout functions:
- Mean: $z_G = \frac{1}{|V|} \sum_{v} z_v$
- Max: $z_G = \max_v z_v$
- Sum: $z_G = \sum_{v} z_v$
- Attention: $z_G = \sum_{v} \alpha_v z_v$

### Heterogeneous Graphs

Handle multiple node and edge types:

$$h_v^{(l+1)} = \sigma\left(\sum_{r \in R} \sum_{u \in N_r(v)} W_r^{(l)} h_u^{(l)}\right)$$

where $R$ is the set of relation types and $N_r(v)$ are neighbors via relation $r$.

### Dynamic Graphs

Handle time-varying graphs:

$$H^{(l+1)}(t) = \sigma(\tilde{A}(t) H^{(l)}(t) W^{(l)} + H^{(l)}(t-1) U^{(l)})$$

---

## Key Takeaways

1. **GCN Architecture**: The GCN layer $H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$ provides a simple yet effective framework for learning node representations through normalized neighborhood aggregation.

2. **Message Passing Framework**: GNNs operate through three steps: message generation, aggregation, and update. Different choices for each step yield different architectures (GCN, GraphSAGE, GAT).

3. **GraphSAGE Scalability**: By sampling fixed-size neighborhoods, GraphSAGE enables inductive learning and batch training on large graphs, addressing the scalability limitations of full-graph methods.

4. **Aggregation Strategies**: Mean, max, and attention-based aggregations capture different aspects of neighborhood structure. Permutation invariance is crucial for graph learning.

5. **Over-Smoothing**: Deep GNNs suffer from over-smoothing where node representations become indistinguishable. Solutions include residual connections, dense connections, and normalization techniques.

6. **Spatial vs Spectral**: Spatial methods (GCN, GraphSAGE) define convolution directly on graph structure, while spectral methods use graph Fourier transform. Modern GNNs blur this distinction.

7. **Training Strategies**: Mini-batch training with neighborhood sampling enables scalable training. Negative sampling is crucial for unsupervised learning tasks.

8. **Applications**: GNNs excel at node classification, link prediction, and graph classification. Extensions handle heterogeneous and dynamic graphs.

9. **Theoretical Limits**: Message-passing GNNs are limited by the Weisfeiler-Lehman test, establishing fundamental expressiveness bounds.

10. **Practical Considerations**: Regularization (dropout, weight decay), optimization (Adam, learning rate scheduling), and architecture choices (depth, aggregation) significantly impact performance.
