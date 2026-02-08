# Coding Theory and Compression

## Table of Contents

1. [Introduction](#introduction)
2. [Source Coding](#source-coding)
3. [Huffman Coding](#huffman-coding)
4. [Arithmetic Coding](#arithmetic-coding)
5. [Source Coding Theorem](#source-coding-theorem)
6. [Lossless Compression](#lossless-compression)
7. [Lossy Compression](#lossy-compression)
8. [Rate-Distortion Theory](#rate-distortion-theory)
9. [Model Compression](#model-compression)
10. [Machine Learning Applications](#machine-learning-applications)
11. [Key Takeaways](#key-takeaways)

## Introduction

Coding theory addresses efficient representation and transmission of information. Source coding (compression) reduces redundancy to represent data with fewer bits, while channel coding adds redundancy for error correction. In machine learning, compression appears in model compression (reducing model size), data compression (efficient storage), and understanding information bottlenecks. This document covers Huffman coding, arithmetic coding, source coding theorem, and their applications in ML.

## Source Coding

### Code Definition

**Source code**: Mapping $C: \mathcal{X} \to \{0,1\}^*$ from source alphabet to binary strings.

**Code length**: $l(x)$ = length of codeword $C(x)$.

**Average code length**: 

$$L(C) = \sum_{x \in \mathcal{X}} p(x) l(x) = \mathbb{E}[l(X)]$$

**Goal**: Minimize $L(C)$ subject to decodability constraints.

### Prefix Codes

**Prefix code**: No codeword is prefix of another.

**Example**: $\{0, 10, 110, 111\}$ is prefix code.

**Non-example**: $\{0, 01, 011\}$ is not prefix (0 is prefix of 01).

**Advantage**: Can decode without lookahead (instantaneous decoding).

**Kraft inequality**: Prefix code exists with lengths $\{l_1, \ldots, l_n\}$ iff:

$$\sum_{i=1}^n 2^{-l_i} \leq 1$$

**Necessary condition**: Any uniquely decodable code satisfies Kraft inequality.

### Optimal Code Length

**Lower bound**: For any code:

$$L(C) \geq H(X)$$

Equality achieved when $l(x) = -\log p(x)$ (if integer lengths possible).

**Shannon code**: $l(x) = \lceil -\log p(x) \rceil$ satisfies:

$$H(X) \leq L(C) < H(X) + 1$$

## Huffman Coding

### Algorithm

**Huffman coding** constructs optimal prefix code:

1. Start with leaves for each symbol with probability $p(x)$
2. Repeatedly merge two nodes with smallest probabilities
3. Assign 0/1 to left/right branches
4. Codeword is path from root to leaf

**Example**: Symbols $\{a, b, c, d\}$ with probabilities $\{0.4, 0.3, 0.2, 0.1\}$:

```
Merge: d(0.1) + c(0.2) = 0.3
Merge: b(0.3) + (d+c)(0.3) = 0.6  
Merge: a(0.4) + (b+d+c)(0.6) = 1.0

Tree:
       1.0
      /   \
   0.4(a)  0.6
          /   \
       0.3(b) 0.3
             /   \
         0.1(d) 0.2(c)

Codes: a=0, b=10, c=111, d=110
```

### Optimality

**Theorem**: Huffman code minimizes average code length among all prefix codes.

**Proof**: By induction on number of symbols. Key: Two least probable symbols have longest codewords differing only in last bit.

**Average length**: $L = 0.4 \cdot 1 + 0.3 \cdot 2 + 0.2 \cdot 3 + 0.1 \cdot 3 = 1.9$ bits

**Entropy**: $H(X) = -0.4\log 0.4 - 0.3\log 0.3 - 0.2\log 0.2 - 0.1\log 0.1 \approx 1.846$ bits

**Efficiency**: $H(X)/L \approx 97\%$

## Arithmetic Coding

### Principle

**Arithmetic coding**: Encodes entire sequence as single number in $[0,1)$.

**Interval subdivision**: 
- Start with $[0,1)$
- Subdivide based on symbol probabilities
- Final interval encodes sequence

**Example**: Encode "aba" with $p(a)=0.6$, $p(b)=0.4$:

1. 'a': $[0, 0.6)$
2. 'b': Subdivide $[0, 0.6)$: $[0.36, 0.6)$ (since $p(b)=0.4$ of interval)
3. 'a': Subdivide $[0.36, 0.6)$: $[0.36, 0.504)$ (since $p(a)=0.6$)

Any number in $[0.36, 0.504)$ (e.g., 0.4) encodes "aba".

### Decoding

**Decoding**: Given number $r \in [0,1)$:

1. Determine which symbol's interval contains $r$
2. Output that symbol
3. Rescale interval and repeat

**Example**: Decode 0.4 with same probabilities:

1. $0.4 \in [0, 0.6)$ → output 'a', new interval $[0, 0.6)$
2. Rescale: $r' = 0.4/0.6 = 0.667$, $0.667 \in [0.6, 1.0)$ → output 'b'
3. Continue...

### Advantages

**Near-optimal**: Achieves average length within 2 bits of entropy per symbol.

**Adaptive**: Can update probabilities during encoding.

**Handles any distribution**: Not limited to integer code lengths.

## Source Coding Theorem

### Statement

**Source coding theorem** (Shannon's first theorem):

For i.i.d. source $X_1, \ldots, X_n$ with entropy $H(X)$:

- **Achievability**: There exists code with average length per symbol $\leq H(X) + \epsilon$ for any $\epsilon > 0$ (for large $n$)
- **Converse**: No code can achieve average length $< H(X)$

**Interpretation**: Entropy is fundamental limit for lossless compression.

### Proof Sketch

**Achievability**: Use typical sequences. For large $n$, most probability mass is on typical set $A_\epsilon^{(n)}$ with size $\approx 2^{nH(X)}$. Encode typical sequences with $nH(X) + n\epsilon$ bits, others arbitrarily.

**Converse**: Any uniquely decodable code satisfies Kraft inequality, so:

$$nH(X) = H(X^n) \leq L(C)$$

by source coding inequality.

### Asymptotic Equipartition Property

**AEP**: For i.i.d. $X_1, \ldots, X_n$:

$$-\frac{1}{n}\log p(X_1, \ldots, X_n) \xrightarrow{p} H(X)$$

**Typical set**: 

$$A_\epsilon^{(n)} = \left\{(x_1, \ldots, x_n) : \left|-\frac{1}{n}\log p(x_1, \ldots, x_n) - H(X)\right| \leq \epsilon\right\}$$

**Properties**:
- $P(A_\epsilon^{(n)}) \to 1$ as $n \to \infty$
- $|A_\epsilon^{(n)}| \leq 2^{n(H(X)+\epsilon)}$

## Lossless Compression

### Dictionary Methods

**LZ77/LZ78**: Build dictionary of previously seen phrases, encode references.

**LZW**: Extends LZ78, used in GIF, Unix compress.

**Example**: "ababab" → encode "ab" once, then reference.

### Run-Length Encoding

**RLE**: Encode runs of repeated symbols.

**Example**: "aaaabbbcc" → "4a3b2c".

**Effective**: For data with many runs (e.g., binary images).

### Burrows-Wheeler Transform

**BWT**: Rearranges text to group similar characters, then applies move-to-front + RLE.

**Used in**: bzip2 compression.

## Lossy Compression

### Quantization

**Quantization**: Map continuous values to discrete set.

**Example**: Round to nearest integer, or use non-uniform quantization levels.

**Tradeoff**: Rate (bits) vs. distortion (error).

### Transform Coding

**Principle**: Transform to domain where energy is concentrated, quantize, encode.

**Example**: 
- **DCT** (Discrete Cosine Transform) in JPEG
- **Wavelet transform** in JPEG2000
- **PCA** for dimensionality reduction

**Process**:
1. Transform: $\mathbf{y} = \mathbf{T}\mathbf{x}$
2. Quantize: $\hat{\mathbf{y}} = Q(\mathbf{y})$
3. Encode: Use entropy coding on $\hat{\mathbf{y}}$
4. Decode and inverse transform: $\hat{\mathbf{x}} = \mathbf{T}^{-1}\hat{\mathbf{y}}$

## Rate-Distortion Theory

### Problem Formulation

**Rate-distortion function**: Minimum rate needed to achieve distortion $\leq D$:

$$R(D) = \min_{p(\hat{x}|x) : \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

where $d(x, \hat{x})$ is distortion measure.

**Interpretation**: Tradeoff between compression (rate) and quality (distortion).

### Distortion Measures

**Squared error**: $d(x, \hat{x}) = (x - \hat{x})^2$

**Hamming distance**: $d(x, \hat{x}) = \mathbf{1}[x \neq \hat{x}]$ (for discrete)

**Perceptual**: Domain-specific measures (e.g., SSIM for images).

### Gaussian Source

**Gaussian source** $X \sim \mathcal{N}(0, \sigma^2)$ with squared error:

$$R(D) = \begin{cases} \frac{1}{2}\log\frac{\sigma^2}{D} & \text{if } D < \sigma^2 \\ 0 & \text{if } D \geq \sigma^2 \end{cases}$$

**Water-filling**: Allocate bits to components with variance above threshold.

## Model Compression

### Pruning

**Magnitude pruning**: Remove weights with small magnitude.

**Structured pruning**: Remove entire neurons/channels.

**Information-theoretic view**: Remove parameters with low mutual information $I(\theta_i; \mathcal{D})$.

### Quantization

**Weight quantization**: Represent weights with fewer bits (e.g., 8-bit instead of 32-bit float).

**Quantization-aware training**: Train with quantization in the loop.

**Rate**: $\sum_i \log |\mathcal{Q}_i|$ where $\mathcal{Q}_i$ is quantization set for parameter $i$.

### Knowledge Distillation

**Teacher-student**: Train small student to mimic large teacher.

**Information**: Student learns compressed representation preserving teacher's knowledge.

**Objective**: Minimize $D_{\text{KL}}(p_{\text{teacher}} \| p_{\text{student}})$.

### Low-Rank Factorization

**Matrix factorization**: Approximate $\mathbf{W} \approx \mathbf{U}\mathbf{V}^T$ with smaller matrices.

**SVD**: $\mathbf{W} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$, keep top $k$ singular values.

**Compression ratio**: $mn / (k(m+n+1))$ for $m \times n$ matrix.

## Machine Learning Applications

### Neural Network Compression

**Goal**: Reduce model size while maintaining accuracy.

**Methods**:
- **Pruning**: Remove redundant connections
- **Quantization**: Reduce precision
- **Distillation**: Train smaller model
- **Factorization**: Low-rank approximations

**Information bottleneck**: Compress representation while preserving task-relevant information.

### Data Compression

**Storage**: Compress datasets for efficient storage.

**Transmission**: Compress models/data for deployment.

**Example**: Compress image datasets, model checkpoints.

### Information Bottleneck Method

**Objective**: 

$$\min_{p(z|x)} I(X; Z) - \beta I(Z; Y)$$

**Interpretation**: 
- Minimize $I(X; Z)$: Compress representation
- Maximize $I(Z; Y)$: Preserve task information
- $\beta$: Tradeoff parameter

**Application**: Understanding what neural networks learn.

### Variational Compression

**VAE as compression**: Encoder compresses $\mathbf{x}$ to $\mathbf{z}$, decoder reconstructs.

**Rate**: $I(\mathbf{X}; \mathbf{Z})$ (mutual information between data and latent)

**Distortion**: Reconstruction error $\mathbb{E}[\|\mathbf{X} - \hat{\mathbf{X}}\|^2]$

**Tradeoff**: Controlled by $\beta$ in $\beta$-VAE.

### Federated Learning

**Compression**: Compress model updates for communication efficiency.

**Methods**: Gradient quantization, sparsification.

**Rate**: Bits per communication round.

**Distortion**: Impact on model accuracy.

### Continual Learning

**Information**: Store compressed representation of previous tasks.

**Catastrophic forgetting**: Prevent by maintaining information about old tasks.

**Elastic Weight Consolidation**: Uses Fisher information to identify important parameters.

## Key Takeaways

1. **Source coding** aims to represent data with minimum average bits, fundamental limit is entropy.

2. **Huffman coding** constructs optimal prefix code, achieving average length close to entropy.

3. **Arithmetic coding** achieves near-optimal compression by encoding sequences as intervals.

4. **Source coding theorem** establishes entropy as fundamental limit for lossless compression.

5. **Lossy compression** trades off rate (bits) and distortion (error) via rate-distortion theory.

6. **Model compression** reduces neural network size via pruning, quantization, distillation, factorization.

7. **Information bottleneck** provides framework for understanding compression in representation learning.

8. **VAEs** implement compression with rate-distortion tradeoff controlled by $\beta$.

9. **Compression** is crucial for efficient ML: smaller models, faster inference, lower storage/transmission costs.

10. **Coding theory** provides mathematical foundation for understanding and optimizing compression in ML systems.
