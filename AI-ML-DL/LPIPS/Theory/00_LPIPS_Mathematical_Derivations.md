# LPIPS Mathematical Derivations and Proofs

## Table of Contents
1. [Core Mathematical Framework](#core-mathematical-framework)
2. [Feature Space Analysis](#feature-space-analysis)
3. [Loss Function Derivations](#loss-function-derivations)
4. [Optimization Theory](#optimization-theory)
5. [Statistical Analysis](#statistical-analysis)

---

## 1. Core Mathematical Framework

### 1.1 Fundamental Distance Metric

**Definition**: For images x₀, x₁ ∈ ℝ^(H×W×C), the LPIPS distance is:

```
d_LPIPS(x₀, x₁) = Σₗ αₗ · d_l(x₀, x₁)
```

Where:
- l ∈ {1, 2, ..., L} indexes the network layers
- αₗ: layer-wise importance weights
- d_l: distance in layer l feature space

### 1.2 Layer-wise Distance Computation

**Detailed Formulation**:

```
d_l(x₀, x₁) = (1/(H_l × W_l)) Σᵢ₌₁^(H_l) Σⱼ₌₁^(W_l) ||w_l ⊙ (φ_l^(i,j)(x₀) - φ_l^(i,j)(x₁))||₂²
```

Where:
- φ_l^(i,j)(x): Feature vector at layer l, spatial position (i,j)
- w_l ∈ ℝ^(C_l): Learned channel weights for layer l
- ⊙: Hadamard (element-wise) product
- H_l, W_l: Spatial dimensions of layer l
- C_l: Number of channels in layer l

### 1.3 Feature Normalization

**Unit L2 Normalization**:

```
φ̂_l^(i,j)(x) = φ_l^(i,j)(x) / ||φ_l^(i,j)(x)||₂
```

**Justification**: Removes magnitude bias, focuses on feature direction.

---

## 2. Feature Space Analysis

### 2.1 Feature Space Geometry

**Assumption**: Deep features form a metric space (F, d) where:
- F: Feature space ⊆ ℝ^D
- d: Distance function satisfying metric properties

**Metric Properties** (verification needed):
1. **Non-negativity**: d(f₁, f₂) ≥ 0
2. **Identity**: d(f₁, f₂) = 0 ⟺ f₁ = f₂
3. **Symmetry**: d(f₁, f₂) = d(f₂, f₁)
4. **Triangle inequality**: d(f₁, f₃) ≤ d(f₁, f₂) + d(f₂, f₃)

### 2.2 Learned Weight Constraints

**Non-negativity Constraint**:
```
w_l ≥ 0 ∀l
```

**Projection Operator**:
```
Π₊(w) = max(0, w)
```

**Mathematical Justification**: Negative weights would imply that larger feature differences make images more similar, which violates perceptual intuition.

### 2.3 Multi-layer Aggregation Theory

**Weighted Sum Formulation**:
```
d_total = Σₗ βₗ · d_l
```

**Optimal Weight Learning**: Find β* that minimizes:
```
β* = argmin_β Σᵢ L(h_i, f_β(d_l^(i)))
```

Where:
- h_i: Human judgment for pair i
- f_β: Learned aggregation function
- L: Cross-entropy loss

---

## 3. Loss Function Derivations

### 3.1 2AFC Loss Function

**Setup**: Given triplet (x, x₀, x₁) with human judgment h ∈ {0, 1}

**Distance Computation**:
```
d₀ = d_LPIPS(x, x₀)
d₁ = d_LPIPS(x, x₁)
```

**Probability Model**:
```
P(human chooses x₀) = σ(G(d₀, d₁))
```

Where σ is sigmoid function and G is learned comparison network.

**Log-Likelihood**:
```
L(x, x₀, x₁, h) = h · log(σ(G(d₀, d₁))) + (1-h) · log(1 - σ(G(d₀, d₁)))
```

### 3.2 Gradient Computation

**Chain Rule Application**:
```
∂L/∂w_l = ∂L/∂G · ∂G/∂d₀ · ∂d₀/∂w_l + ∂L/∂G · ∂G/∂d₁ · ∂d₁/∂w_l
```

**Layer Weight Gradients**:
```
∂d_l/∂w_l = (1/(H_l × W_l)) Σᵢⱼ (φ_l^(i,j)(x₀) - φ_l^(i,j)(x₁))²
```

### 3.3 Alternative Loss Formulations

**Ranking Loss**:
```
L_rank = max(0, margin + d₀ - d₁) if h = 1
L_rank = max(0, margin + d₁ - d₀) if h = 0
```

**Regression Loss** (for continuous similarity scores):
```
L_reg = ||d_pred - d_human||₂²
```

---

## 4. Optimization Theory

### 4.1 Constrained Optimization

**Problem Formulation**:
```
minimize: L(w)
subject to: w_l ≥ 0 ∀l
```

**Projected Gradient Descent**:
```
w^(t+1) = Π₊(w^(t) - η∇L(w^(t)))
```

### 4.2 Convergence Analysis

**Assumptions**:
1. L is convex in w (under fixed features)
2. Gradient is Lipschitz continuous: ||∇L(w₁) - ∇L(w₂)|| ≤ L||w₁ - w₂||

**Convergence Rate**:
```
L(w^(T)) - L(w*) ≤ O(1/T)
```

For projected gradient descent with step size η = 1/L.

### 4.3 Local Optima

**Non-convexity**: When fine-tuning backbone features, loss becomes non-convex.

**Initialization Strategy**: Use pre-trained features as initialization to avoid poor local minima.

---

## 5. Statistical Analysis

### 5.1 Human Agreement Modeling

**Inter-annotator Agreement**:
```
κ = (p_o - p_e) / (1 - p_e)
```

Where:
- p_o: Observed agreement
- p_e: Expected agreement by chance

**Model Performance Upper Bound**: Human-human agreement rate (~82.6% in LPIPS paper)

### 5.2 Confidence Intervals

**Bootstrap Estimation**:
```
CI = [μ̂ - z_{α/2}·σ̂/√n, μ̂ + z_{α/2}·σ̂/√n]
```

For 2AFC accuracy with:
- μ̂: Sample mean accuracy
- σ̂: Sample standard deviation
- n: Number of test pairs

### 5.3 Statistical Significance Testing

**McNemar's Test** for paired accuracy comparison:
```
χ² = (|b - c| - 1)² / (b + c)
```

Where:
- b: Method A correct, Method B incorrect
- c: Method A incorrect, Method B correct

**Null Hypothesis**: Both methods have equal accuracy.

### 5.4 Effect Size Measurement

**Cohen's d** for comparing metric performances:
```
d = (μ₁ - μ₂) / √((σ₁² + σ₂²)/2)
```

**Interpretation**:
- d = 0.2: Small effect
- d = 0.5: Medium effect  
- d = 0.8: Large effect

---

## 6. Theoretical Bounds

### 6.1 Generalization Bounds

**PAC-Bayes Bound**: With probability 1-δ:
```
R(w) ≤ R̂(w) + √((KL(Q||P) + log(2√m/δ)) / (2m))
```

Where:
- R(w): True risk
- R̂(w): Empirical risk
- Q, P: Posterior and prior distributions
- m: Sample size

### 6.2 Sample Complexity

**Required samples** for ε-accurate learning:
```
m ≥ O((d log(d) + log(1/δ)) / ε²)
```

Where d is the effective dimension of the weight space.

### 6.3 Approximation Error

**Universal Approximation**: Deep networks can approximate any continuous perceptual function f:
```
∃ network N: ||N(x) - f(x)||∞ < ε
```

For any ε > 0, with sufficient width/depth.

---

## 7. Information Theoretic Analysis

### 7.1 Mutual Information

**Feature-Perception Mutual Information**:
```
I(F; P) = ∫∫ p(f,p) log(p(f,p)/(p(f)p(p))) df dp
```

Where:
- F: Feature representation
- P: Perceptual similarity

**Hypothesis**: Higher I(F; P) correlates with better LPIPS performance.

### 7.2 Information Bottleneck

**Compression-Prediction Tradeoff**:
```
L_IB = -I(F; P) + βI(X; F)
```

Optimal features balance:
- Prediction: High I(F; P)
- Compression: Low I(X; F)

---

## Conclusion

The mathematical framework of LPIPS provides a principled approach to learning perceptual similarity metrics. The key theoretical insights include:

1. **Feature Space Geometry**: Proper normalization and metric properties
2. **Multi-layer Aggregation**: Optimal weighting of hierarchical features
3. **Optimization Theory**: Constrained learning with convergence guarantees
4. **Statistical Foundations**: Proper evaluation and significance testing

These mathematical foundations enable both theoretical understanding and practical implementation of perceptual similarity metrics.

---

*Mathematical notation follows standard conventions in machine learning and optimization theory.*