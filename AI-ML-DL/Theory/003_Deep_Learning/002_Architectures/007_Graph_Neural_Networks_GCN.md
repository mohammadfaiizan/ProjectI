# Graph Neural Networks and Graph Convolutional Networks

## Table of Contents

1. [Introduction](#introduction)
2. [Graph Representation](#graph-representation)
3. [Spectral Graph Convolution](#spectral-graph-convolution)
4. [Spatial Graph Convolution](#spatial-graph-convolution)
5. [Graph Convolutional Networks (GCN)](#graph-convolutional-networks-gcn)
6. [Message Passing Framework](#message-passing-framework)
7. [GraphSAGE](#graphsage)
8. [Graph Attention Networks (GAT)](#graph-attention-networks-gat)
9. [Applications](#applications)
10. [Key Takeaways](#key-takeaways)

## Introduction

Graph Neural Networks (GNNs) extend neural networks to operate on graph-structured data, enabling learning from relational information. Graph Convolutional Networks (GCNs) are a fundamental class of GNNs that perform convolution-like operations on graphs, learning node representations by aggregating information from neighbors.

This chapter covers the mathematical foundations of GNNs, from spectral and spatial graph convolution to modern architectures like GCN, GraphSAGE, and GAT, examining how they learn representations from graph-structured data.

## Graph Representation

### Graph Definition

A graph $G = (V, E)$ consists of:
- **Vertices/Nodes**: $V = \{v_1, \ldots, v_n\}$
- **Edges**: $E \subseteq V \times V$

### Adjacency Matrix

Binary matrix representing edges:

$$A_{ij} = \begin{cases}
1 & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

### Node Features

Each node has feature vector:

$$\mathbf{X} \in \mathbb{R}^{n \times d}$$

where $n$ is number of nodes and $d$ is feature dimension.

### Degree Matrix

Diagonal matrix of node degrees:

$$D_{ii} = \sum_j A_{ij}$$

### Normalized Adjacency

**Symmetric Normalization**:

$$\tilde{A} = D^{-1/2} A D^{-1/2}$$

**Row Normalization**:

$$\tilde{A} = D^{-1} A$$

## Spectral Graph Convolution

Spectral methods operate in Fourier domain of graphs.

### Graph Laplacian

**Unnormalized**:

$$L = D - A$$

**Normalized**:

$$L = I - D^{-1/2} A D^{-1/2}$$

### Eigendecomposition

$$L = U \Lambda U^T$$

where:
- $U$: Eigenvectors (graph Fourier modes)
- $\Lambda$: Eigenvalues (frequencies)

### Graph Fourier Transform

Signal on graph: $\mathbf{f} \in \mathbb{R}^n$

**Forward Transform**:

$$\hat{\mathbf{f}} = U^T \mathbf{f}$$

**Inverse Transform**:

$$\mathbf{f} = U \hat{\mathbf{f}}$$

### Spectral Convolution

Convolution in spectral domain:

$$\mathbf{g} * \mathbf{f} = U((U^T \mathbf{g}) \odot (U^T \mathbf{f})) = U \text{diag}(\hat{g}) U^T \mathbf{f}$$

where $\hat{g}$ are learnable spectral coefficients.

### Chebyshev Approximation

Approximate spectral filter with Chebyshev polynomials:

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

where $T_k$ are Chebyshev polynomials and $\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{\max}} - I$.

## Spatial Graph Convolution

Spatial methods aggregate information from neighbors directly.

### Neighbor Aggregation

For node $v$, aggregate features from neighbors $\mathcal{N}(v)$:

$$\mathbf{h}_v^{(l+1)} = \text{AGG}(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\})$$

### Aggregation Functions

**Mean**:

$$\mathbf{h}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$$

**Sum**:

$$\mathbf{h}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$$

**Max**:

$$\mathbf{h}_v = \max_{u \in \mathcal{N}(v)} \mathbf{h}_u$$

**Attention**:

$$\mathbf{h}_v = \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{h}_u$$

## Graph Convolutional Networks (GCN)

GCN simplifies spectral convolution with first-order approximation.

### GCN Layer

$$\mathbf{H}^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$$

where:
- $\tilde{A} = A + I$ (add self-loops)
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- $\mathbf{W}^{(l)}$: Learnable weight matrix
- $\sigma$: Activation function

### Node-Wise Formulation

For node $v$:

$$\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{d_v d_u}} \mathbf{h}_u^{(l)}\right)$$

where $d_v$ is degree of node $v$.

### Properties

1. **Localized**: Only uses 1-hop neighbors
2. **Efficient**: Linear in number of edges
3. **Spectral Approximation**: First-order Chebyshev approximation

### Multi-Layer GCN

Stack multiple GCN layers:

$$\mathbf{H}^{(0)} = \mathbf{X}$$

$$\mathbf{H}^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)})$$

Each layer aggregates information from $l+1$ hops away.

## Message Passing Framework

Message passing provides unified framework for GNNs.

### Message Function

Compute message from neighbor:

$$\mathbf{m}_{uv}^{(l)} = M^{(l)}(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}, \mathbf{e}_{uv})$$

where $\mathbf{e}_{uv}$ are edge features.

### Aggregation Function

Aggregate messages from neighbors:

$$\mathbf{a}_v^{(l)} = \text{AGG}(\{\mathbf{m}_{uv}^{(l)} : u \in \mathcal{N}(v)\})$$

### Update Function

Update node representation:

$$\mathbf{h}_v^{(l+1)} = U^{(l)}(\mathbf{h}_v^{(l)}, \mathbf{a}_v^{(l)})$$

### General Framework

$$\mathbf{h}_v^{(l+1)} = U^{(l)}\left(\mathbf{h}_v^{(l)}, \text{AGG}\left(\{M^{(l)}(\mathbf{h}_u^{(l)}, \mathbf{h}_v^{(l)}) : u \in \mathcal{N}(v)\}\right)\right)$$

## GraphSAGE

GraphSAGE (Sample and Aggregate) samples neighbors and aggregates.

### Sampling

Instead of using all neighbors, sample fixed-size set:

$$\mathcal{N}_S(v) = \text{Sample}(\mathcal{N}(v), S)$$

where $S$ is sample size.

### Aggregation

**Mean Aggregator**:

$$\mathbf{h}_v^{(l+1)} = \sigma(\mathbf{W}^{(l)} \cdot \text{MEAN}(\{\mathbf{h}_v^{(l)}\} \cup \{\mathbf{h}_u^{(l)} : u \in \mathcal{N}_S(v)\}))$$

**LSTM Aggregator**: Use LSTM to process sampled neighbors

**Pooling Aggregator**: 

$$\mathbf{h}_v^{(l+1)} = \sigma(\mathbf{W}^{(l)} \cdot [\mathbf{h}_v^{(l)} || \text{MAX}(\{\sigma(\mathbf{W}_{\text{pool}} \mathbf{h}_u^{(l)} + \mathbf{b}) : u \in \mathcal{N}_S(v)\})])$$

### Benefits

1. **Inductive**: Can generalize to unseen nodes
2. **Scalable**: Works with large graphs
3. **Flexible**: Various aggregation functions

## Graph Attention Networks (GAT)

GAT uses attention to weight neighbor contributions.

### Attention Mechanism

Compute attention coefficients:

$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j])$$

Normalize with softmax:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

### GAT Layer

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} \mathbf{W}^{(l)} \mathbf{h}_j^{(l)}\right)$$

### Multi-Head Attention

$$\mathbf{h}_i^{(l+1)} = ||_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l,k)} \mathbf{W}^{(l,k)} \mathbf{h}_j^{(l)}\right)$$

where $||$ denotes concatenation and $K$ is number of heads.

### Properties

1. **Adaptive**: Attention weights adapt to data
2. **Interpretable**: Attention weights show importance
3. **Efficient**: Computes in parallel

## Applications

### Node Classification

Predict labels for nodes:
- Use node representations from final layer
- Apply classifier

### Graph Classification

Predict label for entire graph:
- Aggregate node representations
- Use graph-level representation

**Pooling Methods**:
- Mean pooling
- Max pooling
- Attention pooling
- Set2Set

### Link Prediction

Predict missing edges:
- Use node representations
- Score edge $(u,v)$: $s(u,v) = f(\mathbf{h}_u, \mathbf{h}_v)$

### Recommendation Systems

Users and items as nodes:
- Learn user and item embeddings
- Predict user-item interactions

### Molecular Property Prediction

Molecules as graphs:
- Atoms as nodes
- Bonds as edges
- Predict molecular properties

### Social Networks

Users as nodes, relationships as edges:
- Community detection
- Influence prediction
- Fake news detection

## Key Takeaways

1. **Graph Structure**: GNNs operate on graph-structured data, learning from relational information through node and edge features.

2. **Spectral Convolution**: Operates in Fourier domain using graph Laplacian eigendecomposition, with Chebyshev approximation enabling efficient computation.

3. **Spatial Convolution**: Aggregates information directly from neighbors, providing intuitive and efficient approach to graph convolution.

4. **GCN**: Simplifies spectral convolution with first-order approximation, using normalized adjacency matrix for efficient neighbor aggregation.

5. **Message Passing**: Unified framework where nodes send messages, aggregate them, and update representations, generalizing many GNN architectures.

6. **GraphSAGE**: Samples neighbors and aggregates, enabling inductive learning and scalability to large graphs.

7. **GAT**: Uses attention to weight neighbor contributions adaptively, providing interpretability and improved performance.

8. **Applications**: GNNs excel at node classification, graph classification, link prediction, and various domain-specific tasks like molecular property prediction.

9. **Scalability**: Techniques like sampling (GraphSAGE) and efficient aggregation enable application to large-scale graphs.

10. **Design Choices**: Aggregation functions, message functions, and update functions determine GNN behavior, with different choices suited to different tasks and graph types.
