# Bias Mitigation Techniques in Machine Learning

## Table of Contents

1. [Introduction to Bias Mitigation](#introduction-to-bias-mitigation)
2. [Pre-processing Techniques](#pre-processing-techniques)
3. [In-processing Techniques](#in-processing-techniques)
4. [Post-processing Techniques](#post-processing-techniques)
5. [Fair Representation Learning](#fair-representation-learning)
6. [Adversarial Debiasing](#adversarial-debiasing)
7. [Constrained Optimization](#constrained-optimization)
8. [Fairness Regularization](#fairness-regularization)
9. [Toolkits and Frameworks](#toolkits-and-frameworks)
10. [Evaluation and Validation](#evaluation-and-validation)
11. [Key Takeaways](#key-takeaways)

## Introduction to Bias Mitigation

Bias mitigation techniques aim to reduce unfairness in machine learning systems by modifying data, algorithms, or predictions to achieve fairer outcomes. These techniques can be applied at different stages of the ML pipeline: before training (pre-processing), during training (in-processing), or after training (post-processing).

### Classification of Mitigation Techniques

**Pre-processing**: Modify training data to remove bias before model training:
- Resampling techniques
- Reweighting examples
- Learning fair representations
- Data augmentation

**In-processing**: Modify the learning algorithm to incorporate fairness constraints:
- Fairness-aware loss functions
- Constrained optimization
- Adversarial training
- Regularization for fairness

**Post-processing**: Adjust model predictions after training to achieve fairness:
- Threshold optimization
- Calibration adjustments
- Outcome modification
- Reject option classification

### Choosing Mitigation Approaches

**Considerations**:
- **Stage of pipeline**: Which stage offers most leverage?
- **Fairness metric**: Which metric must be satisfied?
- **Data access**: Can training data be modified?
- **Model access**: Can training algorithm be modified?
- **Performance impact**: What accuracy trade-offs are acceptable?
- **Regulatory requirements**: What constraints must be satisfied?

**Hybrid Approaches**: Often combine multiple techniques:
- Pre-processing + in-processing
- In-processing + post-processing
- Multiple techniques at same stage

### Mathematical Framework

We formalize bias mitigation as an optimization problem:

$$\min_f L(f, D) + \lambda \cdot \text{FairnessViolation}(f, D)$$

where:
- $L(f, D)$ is the loss function
- $\text{FairnessViolation}(f, D)$ measures deviation from fairness
- $\lambda$ controls the trade-off

Alternatively, as constrained optimization:

$$\min_f L(f, D) \quad \text{s.t.} \quad \text{FairnessViolation}(f, D) \leq \epsilon$$

## Pre-processing Techniques

Pre-processing techniques modify training data to reduce bias before model training, addressing bias at its source.

### Resampling Techniques

**Oversampling**: Increase representation of underrepresented groups:
- **Random oversampling**: Duplicate examples from minority groups
- **SMOTE**: Synthetic Minority Oversampling Technique generates synthetic examples
- **ADASYN**: Adaptive Synthetic Sampling adjusts for class imbalance

**Undersampling**: Reduce representation of overrepresented groups:
- **Random undersampling**: Remove examples from majority groups
- **Cluster-based**: Remove examples from dense clusters
- **Tomek links**: Remove borderline examples

**Combined Sampling**: Combine oversampling and undersampling:
- **SMOTE + Tomek**: Oversample minority, remove borderline examples
- **SMOTE + ENN**: Oversample minority, remove misclassified examples

**Stratified Sampling**: Ensure proportional representation in train/test splits:
- Maintain group proportions across splits
- Prevent leakage of group information

**Mathematical Formulation**:

For demographic parity, resample to achieve:

$$P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = a') \quad \forall a, a'$$

by adjusting group sizes in training data.

### Reweighting

**Instance Reweighting**: Assign different weights to training examples based on group membership:

$$w_i = \begin{cases}
\frac{|D|}{|D_a| \cdot |\mathcal{A}|} & \text{if } x_i \in G_a \\
1 & \text{otherwise}
\end{cases}$$

where $D_a$ is data from group $a$.

**Cost-Sensitive Learning**: Assign different misclassification costs:
- Higher cost for misclassifying minority groups
- Group-specific cost matrices
- Adapt costs to achieve fairness

**Fair Reweighting**: Weight examples to achieve fairness objectives:

$$w_i = \frac{P(A = a_i)}{P(A = a_i | Y = y_i)} \cdot \frac{P(Y = y_i)}{P(Y = y_i | A = a_i)}$$

This reweights to remove correlation between protected attributes and labels.

**Optimization-Based Reweighting**: Learn weights to optimize fairness:

$$\min_w \sum_i w_i \ell(f(x_i), y_i) + \lambda \cdot \text{FairnessViolation}(f)$$

subject to constraints on weights.

### Data Transformation

**Feature Removal**: Remove protected attributes and proxies:
- Direct removal of $A$
- Identify and remove correlated features
- Challenge: May not remove all bias

**Feature Transformation**: Transform features to remove bias:
- Principal Component Analysis (PCA) removing protected attribute information
- Fair dimensionality reduction
- Orthogonalization with respect to protected attributes

**Label Flipping**: Modify labels to reduce bias:
- Flip labels for examples contributing to bias
- Probabilistic label flipping
- Targeted label modification

### Fair Data Generation

**Synthetic Data**: Generate synthetic examples with fair distributions:
- Use generative models (GANs, VAEs) conditioned on fairness
- Ensure synthetic data maintains utility while achieving fairness
- Balance realism and fairness

**Data Augmentation**: Augment data to improve representation:
- Domain-specific augmentation
- Adversarial data generation for fairness
- Transfer learning from fair datasets

## In-processing Techniques

In-processing techniques modify the learning algorithm itself to incorporate fairness constraints during training.

### Fairness-Aware Loss Functions

**Group-Weighted Loss**: Weight loss by group to achieve fairness:

$$L(f, D) = \sum_{a \in \mathcal{A}} w_a \sum_{x_i \in G_a} \ell(f(x_i), y_i)$$

where $w_a$ are group-specific weights chosen to achieve fairness.

**Fairness Penalty**: Add fairness violation as penalty term:

$$L_{\text{total}} = L_{\text{prediction}} + \lambda \cdot L_{\text{fairness}}$$

where $L_{\text{fairness}}$ measures fairness violation.

**Example - Demographic Parity Penalty**:

$$L_{\text{fairness}} = \left(\sum_{a} P(\hat{Y} = 1 | A = a) - \bar{P}\right)^2$$

where $\bar{P}$ is the target prediction rate.

### Constrained Optimization

**Fairness Constraints**: Add fairness constraints to optimization:

$$\min_f L(f, D) \quad \text{s.t.} \quad |P(\hat{Y} = 1 | A = a) - P(\hat{Y} = 1 | A = a')| \leq \epsilon$$

**Lagrangian Method**: Convert to unconstrained optimization:

$$\min_f \max_{\lambda \geq 0} L(f, D) + \lambda \cdot (\text{FairnessViolation}(f) - \epsilon)$$

**Projected Gradient Descent**: Project gradients to satisfy constraints:
1. Compute gradient: $\nabla_f L(f, D)$
2. Project onto feasible set satisfying fairness constraints
3. Update parameters

**Frank-Wolfe Algorithm**: Iteratively solve linear approximations:
- Useful for convex constraints
- Maintains feasibility at each step
- Converges to optimal solution

### Adversarial Debiasing

**Concept**: Train model to make predictions while preventing adversary from inferring protected attributes.

**Architecture**: 
- **Predictor**: $f: X \rightarrow \hat{Y}$ (main prediction task)
- **Adversary**: $g: \hat{Y} \rightarrow \hat{A}$ (predicts protected attribute)

**Objective**: Minimize prediction loss while maximizing adversary loss:

$$\min_f \max_g L_{\text{pred}}(f) - \lambda \cdot L_{\text{adv}}(g, f)$$

where $L_{\text{adv}}$ measures adversary's ability to predict $A$ from $\hat{Y}$.

**Training Procedure**:
1. Train predictor to minimize prediction loss
2. Train adversary to predict protected attribute from predictions
3. Train predictor to fool adversary
4. Alternate until convergence

**Mathematical Formulation**:

For equalized odds, adversary tries to predict $A$ from $(\hat{Y}, Y)$:

$$\min_f \max_g \mathbb{E}[\ell_{\text{pred}}(f(X), Y)] - \lambda \cdot \mathbb{E}[\ell_{\text{adv}}(g(\hat{Y}, Y), A)]$$

**Advantages**:
- Doesn't require explicit fairness metrics
- Learns fair representations automatically
- Flexible framework

**Challenges**:
- Training instability
- Hyperparameter tuning ($\lambda$)
- May reduce accuracy significantly

### Fair Representation Learning

**Goal**: Learn representations $Z = h(X)$ that:
- Preserve information for prediction task
- Remove information about protected attributes
- Enable fair predictions

**Autoencoder Approach**: 
- Encoder: $Z = \text{Enc}(X)$
- Decoder: $\hat{X} = \text{Dec}(Z)$
- Predictor: $\hat{Y} = f(Z)$
- Adversary: $\hat{A} = g(Z)$

**Objective**:

$$\min_{\text{Enc}, \text{Dec}, f} \max_g L_{\text{recon}} + L_{\text{pred}} - \lambda \cdot L_{\text{adv}}$$

**Variational Fair Autoencoder**: Use VAE framework with fairness constraints:

$$\min \text{KL}(q(Z|X) || p(Z)) - \mathbb{E}[\log p(X|Z)] + L_{\text{pred}} - \lambda \cdot L_{\text{adv}}$$

**Invariant Risk Minimization**: Learn representations invariant across groups:

$$\min \sum_a L_a(f, D_a) + \lambda \cdot \text{Variance}_a(L_a(f, D_a))$$

where variance encourages similar performance across groups.

## Post-processing Techniques

Post-processing techniques adjust model predictions after training to achieve fairness, without modifying the model itself.

### Threshold Optimization

**Concept**: Use different decision thresholds for different groups to achieve fairness.

**For Equalized Odds**: Find group-specific thresholds $t_a$ such that:

$$P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = a')$$
$$P(\hat{Y} = 1 | Y = 0, A = a) = P(\hat{Y} = 1 | Y = 0, A = a')$$

**Optimization Problem**:

$$\min_{t_a} \sum_a |TPR_a(t_a) - \bar{TPR}| + |FPR_a(t_a) - \bar{FPR}|$$

where $\bar{TPR}$ and $\bar{FPR}$ are target rates.

**For Equal Opportunity**: Optimize only TPR:

$$\min_{t_a} \sum_a |TPR_a(t_a) - \bar{TPR}|$$

**For Demographic Parity**: Optimize prediction rates:

$$\min_{t_a} \sum_a |P(\hat{Y} = 1 | A = a, t_a) - \bar{P}|$$

**Advantages**:
- Simple to implement
- Doesn't require retraining
- Can be applied to any classifier

**Limitations**:
- May reduce overall accuracy
- Requires group membership at prediction time
- May be seen as "reverse discrimination"

### Calibration Adjustment

**Platt Scaling**: Learn calibration function separately for each group:

$$P(Y = 1 | \hat{P} = p, A = a) = \sigma(\alpha_a \cdot p + \beta_a)$$

where $\sigma$ is sigmoid function, $\alpha_a, \beta_a$ are group-specific parameters.

**Isotonic Regression**: Non-parametric calibration per group:
- Learn monotonic function mapping predictions to calibrated probabilities
- Group-specific isotonic regression
- Ensures calibration within groups

**Temperature Scaling**: Adjust temperature parameter per group:

$$P_{\text{calibrated}} = \sigma(\hat{P} / T_a)$$

where $T_a$ is group-specific temperature.

### Outcome Modification

**Randomized Prediction**: Randomly modify predictions to achieve fairness:

For demographic parity:
- If group $a$ has too many positive predictions, randomly flip some to negative
- If group $a$ has too few positive predictions, randomly flip some to positive

**Probability Mass Redistribution**: Redistribute probability mass across groups:

$$P_{\text{adjusted}}(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = a) + \delta_a$$

where $\delta_a$ is chosen to achieve fairness.

**Reject Option Classification**: Reject uncertain predictions and handle them specially:
- Identify predictions near decision boundary
- Reject these predictions
- Handle rejects through alternative process (human review, default decision)

### Post-hoc Calibration

**Group-Specific Calibration**: Calibrate predictions separately for each group:
- Learn calibration function per group
- Apply group-specific calibration
- Ensures calibration within groups

**Fair Calibration**: Calibrate to achieve both calibration and other fairness metrics:
- Multi-objective optimization
- Balance calibration and fairness
- May require trade-offs

## Fair Representation Learning

Fair representation learning aims to learn representations that preserve task-relevant information while removing information about protected attributes.

### Information-Theoretic Approach

**Goal**: Learn representation $Z$ that:
- Maximizes mutual information with $Y$: $I(Z; Y)$
- Minimizes mutual information with $A$: $I(Z; A)$

**Objective**:

$$\max_Z I(Z; Y) - \lambda \cdot I(Z; A)$$

**Variational Bound**: Use variational lower bound:

$$I(Z; Y) \geq \mathbb{E}[\log q(Y|Z)] - H(Y)$$

where $q(Y|Z)$ is variational approximation.

### Adversarial Representation Learning

**Architecture**:
- Encoder: $Z = \text{Enc}(X)$
- Task predictor: $\hat{Y} = f(Z)$
- Adversary: $\hat{A} = g(Z)$

**Training**: Minimize task loss while maximizing adversary loss:

$$\min_{\text{Enc}, f} \max_g L_{\text{task}}(f(\text{Enc}(X)), Y) - \lambda \cdot L_{\text{adv}}(g(\text{Enc}(X)), A)$$

**Gradient Reversal**: Reverse gradient from adversary during backpropagation:
- Forward pass: normal computation
- Backward pass: multiply adversary gradients by $-\lambda$
- Allows end-to-end training

### Disentangled Representations

**Goal**: Learn representations with separate components:
- Task-relevant: $Z_{\text{task}}$
- Protected attribute: $Z_{\text{protected}}$
- Independent: $Z_{\text{task}} \perp Z_{\text{protected}}$

**Beta-VAE**: Use $\beta$-VAE with fairness constraints:

$$\min \text{KL}(q(Z|X) || p(Z)) - \beta \cdot \mathbb{E}[\log p(X|Z)] + L_{\text{task}} + \lambda \cdot L_{\text{fairness}}$$

**Factorized Representations**: Explicitly factorize:

$$Z = [Z_{\text{task}}, Z_{\text{protected}}, Z_{\text{noise}}]$$

with constraints ensuring independence.

## Adversarial Debiasing

Adversarial debiasing uses adversarial training to learn fair representations and predictions.

### Basic Framework

**Two-Player Game**:
- **Learner**: Wants to predict $Y$ accurately
- **Adversary**: Wants to predict $A$ from predictions

**Equilibrium**: Nash equilibrium where:
- Learner makes accurate predictions that don't reveal $A$
- Adversary cannot predict $A$ from predictions

**Mathematical Formulation**:

$$\min_f \max_g \mathbb{E}[\ell_Y(f(X), Y)] - \lambda \cdot \mathbb{E}[\ell_A(g(f(X)), A)]$$

### Variants

**For Equalized Odds**: Adversary predicts $A$ from $(\hat{Y}, Y)$:

$$\min_f \max_g \mathbb{E}[\ell_Y(f(X), Y)] - \lambda \cdot \mathbb{E}[\ell_A(g(\hat{Y}, Y), A)]$$

**For Demographic Parity**: Adversary predicts $A$ from $\hat{Y}$:

$$\min_f \max_g \mathbb{E}[\ell_Y(f(X), Y)] - \lambda \cdot \mathbb{E}[\ell_A(g(\hat{Y}), A)]$$

**For Calibration**: Adversary predicts $A$ from $(\hat{P}, Y)$:

$$\min_f \max_g \mathbb{E}[\ell_Y(f(X), Y)] - \lambda \cdot \mathbb{E}[\ell_A(g(\hat{P}, Y), A)]$$

### Training Challenges

**Instability**: Adversarial training can be unstable:
- Use gradient penalty
- Spectral normalization
- Careful learning rate scheduling

**Hyperparameter Tuning**: $\lambda$ controls fairness-accuracy trade-off:
- Large $\lambda$: More fairness, less accuracy
- Small $\lambda$: More accuracy, less fairness
- Grid search or validation on fairness metrics

**Convergence**: May not converge to desired equilibrium:
- Monitor both objectives
- Early stopping criteria
- Multiple random initializations

## Constrained Optimization

Constrained optimization directly incorporates fairness constraints into the learning objective.

### Formulation

**General Form**:

$$\min_f L(f, D) \quad \text{s.t.} \quad c_i(f, D) \leq 0, \quad i = 1, \ldots, m$$

where $c_i$ are fairness constraints.

**Example Constraints**:

Demographic parity:
$$|P(\hat{Y} = 1 | A = a) - P(\hat{Y} = 1 | A = a')| \leq \epsilon$$

Equalized odds:
$$|TPR_a - TPR_{a'}| \leq \epsilon, \quad |FPR_a - FPR_{a'}| \leq \epsilon$$

### Solution Methods

**Lagrangian Method**: Convert to unconstrained:

$$\min_f \max_{\lambda \geq 0} L(f, D) + \sum_i \lambda_i c_i(f, D)$$

**Dual Ascent**: Iteratively:
1. Update $f$: $\min_f L(f, D) + \sum_i \lambda_i c_i(f, D)$
2. Update $\lambda$: $\lambda_i \leftarrow \max(0, \lambda_i + \alpha \cdot c_i(f, D))$

**Projected Gradient Descent**: Project onto feasible set:
1. Compute gradient: $\nabla_f L(f, D)$
2. Project: $f \leftarrow \text{Proj}_{\mathcal{C}}(f - \alpha \nabla_f L)$
where $\mathcal{C}$ is feasible set.

**Frank-Wolfe**: For convex constraints:
1. Solve linear approximation
2. Update along line to feasible point
3. Converges to optimal solution

### Challenges

**Non-Convexity**: Fairness constraints are often non-convex:
- Local minima
- Multiple solutions
- Need for good initialization

**Computational Cost**: Constrained optimization can be expensive:
- Projection operations
- Constraint evaluation
- May require specialized solvers

**Constraint Satisfaction**: Ensuring constraints are satisfied:
- Approximation errors
- Constraint violations
- Need for validation

## Fairness Regularization

Fairness regularization adds penalty terms to the loss function to encourage fairness.

### Regularization Terms

**Demographic Parity Regularization**:

$$R_{\text{DP}} = \left(\sum_a P(\hat{Y} = 1 | A = a) - \bar{P}\right)^2$$

**Equalized Odds Regularization**:

$$R_{\text{EO}} = \sum_a (TPR_a - \bar{TPR})^2 + (FPR_a - \bar{FPR})^2$$

**Individual Fairness Regularization**:

$$R_{\text{IF}} = \sum_{i,j} \max(0, |f(x_i) - f(x_j)| - d(x_i, x_j))^2$$

### Combined Objective

**Total Loss**:

$$L_{\text{total}} = L_{\text{prediction}} + \lambda_{\text{DP}} \cdot R_{\text{DP}} + \lambda_{\text{EO}} \cdot R_{\text{EO}} + \lambda_{\text{IF}} \cdot R_{\text{IF}}$$

**Hyperparameter Tuning**: Choose $\lambda$ values:
- Cross-validation on fairness metrics
- Pareto frontier analysis
- Stakeholder input

### Advantages

**Flexibility**: Can combine multiple fairness objectives:
- Weight different metrics
- Balance trade-offs
- Adapt to context

**Simplicity**: Easy to implement:
- Add regularization terms
- Standard optimization
- No special algorithms needed

**Interpretability**: Clear relationship between regularization and fairness:
- Larger $\lambda$: More emphasis on fairness
- Can analyze gradient contributions
- Understand model behavior

## Toolkits and Frameworks

Several toolkits provide implementations of bias mitigation techniques.

### AIF360 (IBM)

**Features**:
- Pre-processing algorithms (reweighing, disparate impact remover)
- In-processing algorithms (adversarial debiasing, prejudice remover)
- Post-processing algorithms (equalized odds post-processing, calibrated equalized odds)
- Fairness metrics (demographic parity, equalized odds, etc.)

**Usage**:
```python
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric

# Load dataset
dataset = load_dataset()

# Apply reweighing
RW = Reweighing(unprivileged_groups, privileged_groups)
dataset_transformed = RW.fit_transform(dataset)

# Evaluate fairness
metric = BinaryLabelDatasetMetric(dataset_transformed, 
                                  unprivileged_groups, 
                                  privileged_groups)
print(metric.statistical_parity_difference())
```

### Fairlearn (Microsoft)

**Features**:
- Post-processing (threshold optimization)
- In-processing (reduction algorithms)
- Fairness metrics
- Visualization tools

**Usage**:
```python
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference

# Train base model
model = train_model(X_train, y_train)

# Apply threshold optimization
postprocess = ThresholdOptimizer(estimator=model,
                                 constraints="demographic_parity")
postprocess.fit(X_train, y_train, sensitive_features=A_train)

# Evaluate
predictions = postprocess.predict(X_test, sensitive_features=A_test)
print(demographic_parity_difference(y_test, predictions, 
                                     sensitive_features=A_test))
```

### Fairness Indicators (Google)

**Features**:
- Comprehensive fairness metrics
- Visualization dashboards
- Integration with TensorFlow
- Slicing analysis

### Other Toolkits

**Themis-ML**: Fairness-aware machine learning library

**Fairness Comparison**: Benchmarking framework for fairness algorithms

**ML-Fairness-Gym**: Reinforcement learning environments for studying fairness

## Evaluation and Validation

Proper evaluation is crucial for assessing bias mitigation effectiveness.

### Evaluation Metrics

**Fairness Metrics**: Measure fairness after mitigation:
- Demographic parity gap
- Equalized odds difference
- Calibration error
- Individual fairness violation

**Performance Metrics**: Measure accuracy impact:
- Overall accuracy
- Group-specific accuracy
- Precision and recall
- AUC-ROC

**Trade-off Analysis**: Analyze fairness-accuracy trade-offs:
- Pareto frontier
- Fairness-accuracy curves
- Cost-benefit analysis

### Validation Procedures

**Cross-Validation**: Use stratified cross-validation:
- Maintain group proportions
- Evaluate fairness on held-out data
- Avoid data leakage

**Temporal Validation**: Test on future data:
- Train on historical data
- Test on recent data
- Detect distribution shift

**External Validation**: Validate on external datasets:
- Independent test sets
- Real-world deployment data
- Continuous monitoring

### Best Practices

**Multiple Metrics**: Evaluate multiple fairness metrics:
- No single metric captures all concerns
- Understand trade-offs
- Context-dependent importance

**Stakeholder Validation**: Involve stakeholders in evaluation:
- Affected communities
- Domain experts
- Legal and compliance

**Documentation**: Document evaluation procedures and results:
- Methods used
- Metrics computed
- Trade-offs made
- Limitations identified

**Continuous Monitoring**: Monitor fairness in deployment:
- Track metrics over time
- Detect drift
- Update mitigation as needed

## Key Takeaways

1. **Multiple approaches exist**: Pre-processing, in-processing, and post-processing techniques each have advantages and can be combined.

2. **Trade-offs are inevitable**: Bias mitigation typically involves accuracy-fairness trade-offs that must be explicitly managed.

3. **Context matters**: The appropriate technique depends on application context, data availability, and fairness requirements.

4. **No one-size-fits-all**: Different techniques work better for different fairness metrics and scenarios.

5. **Toolkits facilitate implementation**: Frameworks like AIF360 and Fairlearn provide ready-to-use implementations.

6. **Evaluation is critical**: Proper evaluation using multiple metrics is essential for assessing mitigation effectiveness.

7. **Hybrid approaches often best**: Combining multiple techniques can achieve better results than any single approach.

8. **Continuous process**: Bias mitigation is not one-time but requires ongoing monitoring and adjustment.

9. **Stakeholder involvement**: Engaging affected stakeholders in technique selection and evaluation improves outcomes.

10. **Documentation essential**: Documenting mitigation approaches, trade-offs, and results enables accountability and learning.
