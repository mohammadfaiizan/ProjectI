# Hyperparameter Tuning and Optimization

## Table of Contents

1. [Introduction](#introduction)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Bayesian Optimization](#bayesian-optimization)
5. [Tree-Structured Parzen Estimator](#tree-structured-parzen-estimator)
6. [Early Stopping](#early-stopping)
7. [Hyperband and Successive Halving](#hyperband-and-successive-halving)
8. [BOHB: Best of Both Worlds](#bohb-best-of-both-worlds)
9. [Optuna Framework](#optuna-framework)
10. [Key Takeaways](#key-takeaways)

## Introduction

Hyperparameter tuning is the process of selecting optimal hyperparameters for machine learning models. Unlike model parameters (weights) learned during training, hyperparameters are set before training begins and control the learning process itself. Examples include learning rates, regularization coefficients, network architectures, batch sizes, and dropout rates.

The hyperparameter optimization problem is:

$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \in \Lambda} \mathcal{L}(\boldsymbol{\lambda})$$

where $\mathcal{L}(\boldsymbol{\lambda})$ is the validation loss achieved by training with hyperparameters $\boldsymbol{\lambda}$, and $\Lambda$ is the hyperparameter search space.

This is challenging because:
- Each evaluation requires training a model (expensive)
- The objective function is non-convex and noisy
- No gradients are available
- The search space can be high-dimensional and mixed (continuous, discrete, categorical)

## Grid Search

Grid search exhaustively evaluates all combinations of hyperparameters from predefined grids:

$$\Lambda_{\text{grid}} = \{\lambda_1^{(1)}, \lambda_1^{(2)}, \ldots\} \times \{\lambda_2^{(1)}, \lambda_2^{(2)}, \ldots\} \times \cdots$$

### Algorithm

```
1. Define grids for each hyperparameter
2. For each combination (λ_1, λ_2, ..., λ_d):
   a. Train model with hyperparameters
   b. Evaluate on validation set
   c. Record performance
3. Return best hyperparameters
```

### Properties

**Advantages**:
- Simple and straightforward
- Guaranteed to explore entire grid
- Parallelizable (each evaluation independent)
- No assumptions about objective function

**Disadvantages**:
- Exponential growth: $|\Lambda_{\text{grid}}| = \prod_{i=1}^{d} |\Lambda_i|$
- Wastes budget on poor regions
- Requires manual grid specification
- Doesn't adapt based on results

### Example

For learning rate $\alpha \in \{0.001, 0.01, 0.1\}$ and regularization $\lambda \in \{0.1, 1.0, 10.0\}$:

| $\alpha$ | $\lambda$ | Validation Loss |
|----------|-----------|-----------------|
| 0.001    | 0.1       | 0.85            |
| 0.001    | 1.0       | 0.82            |
| 0.001    | 10.0      | 0.88            |
| 0.01     | 0.1       | 0.78            |
| 0.01     | 1.0       | 0.75            |
| 0.01     | 10.0      | 0.81            |
| 0.1      | 0.1       | 0.92            |
| 0.1      | 1.0       | 0.89            |
| 0.1      | 10.0      | 0.95            |

Best: $\alpha = 0.01, \lambda = 1.0$ with loss 0.75.

## Random Search

Random search samples hyperparameters uniformly at random from the search space:

$$\boldsymbol{\lambda}_i \sim \text{Uniform}(\Lambda)$$

### Algorithm

```
1. Define search space Λ
2. For i = 1 to N:
   a. Sample λ_i ~ Uniform(Λ)
   b. Train model with λ_i
   c. Evaluate on validation set
   d. Record performance
3. Return best hyperparameters
```

### Properties

**Advantages**:
- Simple implementation
- Better than grid search when few hyperparameters matter
- Easy to parallelize
- No grid specification needed

**Disadvantages**:
- No guarantee of finding optimum
- Doesn't use information from previous evaluations
- May sample poor regions repeatedly

### Theoretical Justification

Bergstra and Bengio (2012) showed random search outperforms grid search when:
- Only a few hyperparameters significantly affect performance
- The effective dimensionality is lower than the nominal dimensionality

For $d$ hyperparameters where only $k < d$ matter, random search explores the $k$-dimensional effective space more efficiently than grid search's $d$-dimensional grid.

### Example

```python
import numpy as np
from scipy.stats import uniform, loguniform

# Continuous hyperparameters
learning_rate = loguniform(1e-4, 1e-1)  # log-uniform
batch_size = [32, 64, 128, 256]  # discrete
dropout = uniform(0.0, 0.5)  # uniform

# Sample N configurations
for i in range(N):
    lr = learning_rate.rvs()
    bs = np.random.choice(batch_size)
    do = dropout.rvs()
    # Train and evaluate
```

## Bayesian Optimization

Bayesian optimization uses a probabilistic model (surrogate) of the objective function to guide search efficiently. It balances **exploration** (uncertain regions) and **exploitation** (promising regions).

### Framework

1. **Surrogate model**: $p(f(\boldsymbol{\lambda}) | \mathcal{D})$ where $\mathcal{D} = \{(\boldsymbol{\lambda}_i, y_i)\}$ is observed data
2. **Acquisition function**: $\alpha(\boldsymbol{\lambda}; \mathcal{D})$ quantifying utility of evaluating $\boldsymbol{\lambda}$
3. **Optimization**: $\boldsymbol{\lambda}_{\text{next}} = \arg\max_{\boldsymbol{\lambda}} \alpha(\boldsymbol{\lambda}; \mathcal{D})$

### Gaussian Process Surrogate

A Gaussian Process (GP) models the objective as:

$$f(\boldsymbol{\lambda}) \sim \mathcal{GP}(\mu(\boldsymbol{\lambda}), k(\boldsymbol{\lambda}, \boldsymbol{\lambda}'))$$

where $\mu(\cdot)$ is the mean function and $k(\cdot, \cdot)$ is the covariance (kernel) function.

Given observations $\mathcal{D} = \{(\boldsymbol{\lambda}_i, y_i)\}_{i=1}^{n}$, the posterior is:

$$p(f(\boldsymbol{\lambda}_*) | \mathcal{D}) = \mathcal{N}(\mu_n(\boldsymbol{\lambda}_*), \sigma_n^2(\boldsymbol{\lambda}_*))$$

where:
$$\mu_n(\boldsymbol{\lambda}_*) = \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{y}$$

$$\sigma_n^2(\boldsymbol{\lambda}_*) = k(\boldsymbol{\lambda}_*, \boldsymbol{\lambda}_*) - \mathbf{k}_*^T(\mathbf{K} + \sigma^2\mathbf{I})^{-1}\mathbf{k}_*$$

Here, $\mathbf{K}_{ij} = k(\boldsymbol{\lambda}_i, \boldsymbol{\lambda}_j)$, $\mathbf{k}_* = [k(\boldsymbol{\lambda}_*, \boldsymbol{\lambda}_1), \ldots]^T$, and $\sigma^2$ is observation noise.

### Acquisition Functions

**Expected Improvement (EI)**:
$$\text{EI}(\boldsymbol{\lambda}) = \mathbb{E}[\max(0, f_{\min} - f(\boldsymbol{\lambda}))]$$

where $f_{\min}$ is the best observed value. For GP:

$$\text{EI}(\boldsymbol{\lambda}) = \sigma_n(\boldsymbol{\lambda})[\Phi(Z) + Z\phi(Z)]$$

where $Z = \frac{f_{\min} - \mu_n(\boldsymbol{\lambda})}{\sigma_n(\boldsymbol{\lambda})}$ and $\Phi, \phi$ are CDF and PDF of standard normal.

**Upper Confidence Bound (UCB)**:
$$\text{UCB}(\boldsymbol{\lambda}) = \mu_n(\boldsymbol{\lambda}) + \beta \sigma_n(\boldsymbol{\lambda})$$

where $\beta$ controls exploration-exploitation tradeoff.

**Probability of Improvement (PI)**:
$$\text{PI}(\boldsymbol{\lambda}) = \Phi\left(\frac{f_{\min} - \mu_n(\boldsymbol{\lambda})}{\sigma_n(\boldsymbol{\lambda})}\right)$$

### Algorithm

```
1. Initialize: Evaluate random points, build GP
2. For t = 1 to T:
   a. Optimize acquisition function: λ_t = argmax α(λ; D)
   b. Evaluate objective: y_t = L(λ_t)
   c. Update GP with (λ_t, y_t)
3. Return best hyperparameters
```

### Advantages

- **Sample efficient**: Uses information from all previous evaluations
- **Handles noise**: GP models observation uncertainty
- **Theoretically grounded**: Convergence guarantees exist
- **Adaptive**: Focuses search on promising regions

### Limitations

- **Computational cost**: GP inference is $O(n^3)$ for $n$ observations
- **Scalability**: Challenging for high-dimensional spaces
- **Discrete/categorical**: Requires special kernels or transformations

## Tree-Structured Parzen Estimator

Tree-Structured Parzen Estimator (TPE) is a sequential model-based optimization method that models $p(\boldsymbol{\lambda} | y)$ instead of $p(y | \boldsymbol{\lambda})$.

### Algorithm

TPE maintains two distributions:
- **Good distribution**: $l(\boldsymbol{\lambda}) = p(\boldsymbol{\lambda} | y < \gamma)$ for top $\gamma$ fraction
- **Bad distribution**: $g(\boldsymbol{\lambda}) = p(\boldsymbol{\lambda} | y \geq \gamma)$ for bottom $(1-\gamma)$ fraction

The acquisition function is:

$$\text{EI}(\boldsymbol{\lambda}) \propto \frac{l(\boldsymbol{\lambda})}{g(\boldsymbol{\lambda})}$$

Maximizing this ratio selects points likely under $l$ but unlikely under $g$.

### Implementation

```
1. Initialize: Evaluate random points
2. For t = 1 to T:
   a. Sort observations by y, set threshold γ
   b. Fit l(λ) using top γ fraction
   c. Fit g(λ) using bottom (1-γ) fraction
   d. Sample candidates from l(λ)
   e. Evaluate candidate maximizing l(λ)/g(λ)
3. Return best hyperparameters
```

### Properties

- **Efficient**: Avoids expensive GP inference
- **Handles mixed spaces**: Works with continuous, discrete, categorical
- **Adaptive**: Focuses on promising regions
- **Used in Hyperopt**: Popular implementation

## Early Stopping

Early stopping terminates training when validation performance stops improving, preventing overfitting and saving computational resources.

### Algorithm

```
1. Train model with patience P
2. Monitor validation loss
3. If validation loss doesn't improve for P epochs:
   a. Stop training
   b. Restore weights from best validation epoch
```

### Implementation

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True  # Stop training
        return False
```

### Regularization Interpretation

Early stopping acts as implicit regularization. For gradient descent on quadratic loss:

$$\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)^T\mathbf{H}(\boldsymbol{\theta} - \boldsymbol{\theta}^*)$$

Early stopping finds $\boldsymbol{\theta}$ with smaller norm than unconstrained minimum, similar to weight decay.

### Hyperparameter Considerations

- **Patience**: How many epochs to wait (typically 10-50)
- **Min delta**: Minimum change to qualify as improvement
- **Monitor metric**: Validation loss, accuracy, or custom metric
- **Restore best**: Whether to restore weights from best epoch

## Hyperband and Successive Halving

Hyperband addresses the **exploration vs exploitation** tradeoff in hyperparameter optimization by adaptively allocating budget across configurations.

### Successive Halving

Successive halving allocates budget non-uniformly:

```
1. Sample N configurations
2. For r = r_min, 2*r_min, 4*r_min, ..., r_max:
   a. Train each config for r resources
   b. Keep top 1/η configurations
   c. Continue with remaining configs
```

Here, $\eta$ is the elimination rate (typically 3).

### Example

With $N=81$ configurations, $\eta=3$, $r_{\max}=81$:

| Stage | Configs | Resources Each | Total Budget |
|-------|---------|----------------|--------------|
| 1     | 81      | 1              | 81           |
| 2     | 27      | 3              | 81           |
| 3     | 9       | 9              | 81           |
| 4     | 3       | 27             | 81           |
| 5     | 1       | 81             | 81           |

Total: $5 \times 81 = 405$ resources (vs $81 \times 81 = 6561$ for uniform).

### Hyperband

Hyperband runs multiple Successive Halving brackets with different $N$ values:

```
For s in [s_max, s_max-1, ..., 0]:
    n = ceil((s_max+1)/(s+1) * η^s)
    r = r_max * η^(-s)
    Run Successive Halving with (n, r)
```

This explores different tradeoffs between exploration (many configs, few resources) and exploitation (few configs, many resources).

### Advantages

- **Efficient**: Quickly eliminates poor configurations
- **Adaptive**: Allocates more budget to promising configs
- **Theoretical guarantees**: Near-optimal under certain conditions
- **Simple**: Easy to implement and parallelize

## BOHB: Best of Both Worlds

BOHB (Bayesian Optimization and Hyperband) combines Bayesian optimization's sample efficiency with Hyperband's resource efficiency.

### Algorithm

BOHB uses TPE as the surrogate model within Hyperband brackets:

```
1. Initialize: Random configurations
2. For each Hyperband bracket:
   a. Sample configurations using TPE
   b. Run Successive Halving
   c. Update TPE model with results
3. Return best configuration
```

### Key Innovation

Instead of random sampling in Hyperband, BOHB uses TPE to sample promising configurations based on all previous observations, improving sample efficiency.

### Implementation Details

- **TPE model**: Maintains separate models per bracket or shared model
- **Sampling**: Uses TPE's $l(\boldsymbol{\lambda})/g(\boldsymbol{\lambda})$ ratio
- **Budget**: Adapts to available computational resources
- **Parallelization**: Supports parallel evaluation of configurations

### Performance

BOHB typically outperforms:
- Random search: Better sample efficiency
- Pure Bayesian optimization: Better resource efficiency
- Hyperband: Better initial sampling

## Optuna Framework

Optuna is a hyperparameter optimization framework that implements multiple algorithms and provides an easy-to-use API.

### Features

- **Multiple samplers**: TPE, CMA-ES, random search, grid search
- **Pruning**: Early stopping integration (MedianPruner, PercentilePruner)
- **Multi-objective**: Optimize multiple objectives simultaneously
- **Visualization**: Tools for analyzing optimization history
- **Distributed**: Supports parallel and distributed optimization

### Basic Usage

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    
    # Train model
    model = train_model(lr, batch_size, dropout)
    
    # Evaluate
    val_loss = evaluate(model)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
```

### Advanced Features

**Pruning**:
```python
def objective(trial):
    for epoch in range(100):
        train_epoch()
        val_loss = evaluate()
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss
```

**Multi-objective**:
```python
study = optuna.create_study(directions=['minimize', 'maximize'])
study.optimize(objective, n_trials=100)
# Returns Pareto-optimal solutions
```

**Samplers**:
- `TPESampler`: Tree-structured Parzen estimator
- `CmaEsSampler`: Covariance Matrix Adaptation Evolution Strategy
- `RandomSampler`: Random search
- `GridSampler`: Grid search

## Key Takeaways

1. **Grid search** exhaustively evaluates all combinations but suffers from exponential growth and doesn't adapt to results.

2. **Random search** samples uniformly and often outperforms grid search when few hyperparameters matter, but doesn't use information from previous evaluations.

3. **Bayesian optimization** uses probabilistic models (e.g., Gaussian Processes) to guide search efficiently, balancing exploration and exploitation through acquisition functions like Expected Improvement.

4. **Tree-Structured Parzen Estimator (TPE)** models $p(\boldsymbol{\lambda} | y)$ instead of $p(y | \boldsymbol{\lambda})$, avoiding expensive GP inference while maintaining efficiency.

5. **Early stopping** prevents overfitting by terminating training when validation performance plateaus, acting as implicit regularization.

6. **Successive Halving** allocates budget non-uniformly, quickly eliminating poor configurations and focusing resources on promising ones.

7. **Hyperband** runs multiple Successive Halving brackets with different exploration-exploitation tradeoffs, providing robust performance across scenarios.

8. **BOHB** combines Bayesian optimization (TPE) with Hyperband, achieving both sample efficiency and resource efficiency.

9. **Optuna** provides a unified framework implementing multiple algorithms with features like pruning, multi-objective optimization, and visualization.

10. **Hyperparameter tuning** is crucial for model performance; choosing the right method depends on budget, search space complexity, and computational constraints.
