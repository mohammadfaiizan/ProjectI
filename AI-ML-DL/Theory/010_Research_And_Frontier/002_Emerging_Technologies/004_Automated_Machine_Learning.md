# Automated Machine Learning

## Table of Contents

1. [Introduction](#introduction)
2. [AutoML Pipeline Overview](#automl-pipeline-overview)
3. [Hyperparameter Optimization](#hyperparameter-optimization)
4. [Neural Architecture Search Integration](#neural-architecture-search-integration)
5. [Feature Engineering Automation](#feature-engineering-automation)
6. [Auto-sklearn and AutoGluon](#auto-sklearn-and-autogluon)
7. [Model Selection and Ensembling](#model-selection-and-ensembling)
8. [Meta-Learning for AutoML](#meta-learning-for-automl)
9. [End-to-End AutoML Systems](#end-to-end-automl-systems)
10. [Key Takeaways](#key-takeaways)

## Introduction

Automated Machine Learning (AutoML) aims to automate the end-to-end process of applying machine learning to real-world problems. This includes automated data preprocessing, feature engineering, algorithm selection, hyperparameter optimization, and model evaluation.

AutoML democratizes machine learning by making it accessible to non-experts while also helping experts save time on repetitive tasks. The field has seen rapid progress, with systems that can automatically discover high-performing models for diverse tasks.

Key research directions:
- How to automate each stage of the ML pipeline?
- How to efficiently search large configuration spaces?
- How to transfer knowledge across tasks?
- How to balance automation with user control?

## AutoML Pipeline Overview

The AutoML pipeline encompasses all stages from raw data to deployed model.

### Stages

1. **Data preparation**: Loading, cleaning, validation
2. **Feature engineering**: Creating, selecting, transforming features
3. **Model selection**: Choosing algorithm(s)
4. **Hyperparameter optimization**: Tuning hyperparameters
5. **Model training**: Training selected models
6. **Model evaluation**: Assessing performance
7. **Ensemble construction**: Combining models
8. **Model deployment**: Deploying best model

### Automation Levels

**Full automation**: All stages automated
**Partial automation**: Some stages automated
**Assisted automation**: Human-in-the-loop

### Challenges

**Search space**: Extremely large configuration space
**Evaluation cost**: Training models is expensive
**Multi-objective**: Balance accuracy, latency, interpretability
**Transfer**: Leverage knowledge from previous tasks

## Hyperparameter Optimization

Hyperparameter optimization is a core component of AutoML, finding optimal hyperparameters for machine learning algorithms.

### Problem Formulation

**Given**: Algorithm $A$ with hyperparameters $\lambda \in \Lambda$
**Dataset**: $\mathcal{D} = \{(x_i, y_i)\}$
**Objective**: Find $\lambda^*$ that minimizes validation loss:

$$\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{L}_{val}(A_\lambda, \mathcal{D})$$

where $A_\lambda$ is algorithm with hyperparameters $\lambda$.

### Grid Search

**Method**: Exhaustively search over grid of values
**Advantages**: Simple, guaranteed to find best in grid
**Limitations**: Curse of dimensionality, inefficient

**Example**: For learning rate $\alpha \in \{0.001, 0.01, 0.1\}$ and batch size $b \in \{32, 64, 128\}$, evaluate all 9 combinations.

### Random Search

**Method**: Randomly sample hyperparameters
**Advantages**: More efficient than grid search, better coverage
**Limitations**: No guidance, may miss good regions

**Theoretical**: Often more efficient than grid search in high dimensions

### Bayesian Optimization

**Key idea**: Build probabilistic model of objective function, use it to guide search

**Gaussian Process**: Model $f(\lambda)$ as GP:
$$f(\lambda) \sim \mathcal{GP}(\mu(\lambda), k(\lambda, \lambda'))$$

**Acquisition function**: Choose next point to evaluate
- **Expected Improvement (EI)**: $EI(\lambda) = \mathbb{E}[\max(0, f^* - f(\lambda))]$
- **Upper Confidence Bound (UCB)**: $UCB(\lambda) = \mu(\lambda) + \beta \sigma(\lambda)$
- **Probability of Improvement**: Probability that $f(\lambda) < f^*$

**Algorithm**:
1. Initialize with random points
2. Fit GP to observed data
3. Optimize acquisition function to get next point
4. Evaluate objective at new point
5. Update GP and repeat

**Advantages**: Efficient, good sample efficiency
**Limitations**: Assumes smooth objective, can be slow for high dimensions

### Tree-Structured Parzen Estimator (TPE)

**Method**: Model $p(x|y)$ instead of $p(y|x)$
**Split**: Divide observations into "good" and "bad"
**Models**: Fit separate distributions to each group
**Sampling**: Sample from "good" distribution

**Advantages**: Handles conditional dependencies, efficient
**Limitations**: Less principled than GP-based methods

### Hyperband

**Key idea**: Early stopping of poor configurations

**Successive halving**: 
1. Start with $n$ configurations
2. Run for budget $B$
3. Keep best $n/2$, double budget
4. Repeat until one configuration remains

**Hyperband**: Multiple brackets with different $n$ and $B$

**Advantages**: Very efficient, handles large search spaces
**Limitations**: May eliminate good configurations early

### Multi-Fidelity Optimization

**Fidelities**: Different evaluation budgets (e.g., epochs, data subset)
**Low fidelity**: Fast but inaccurate
**High fidelity**: Slow but accurate
**Strategy**: Use low fidelity to filter, high fidelity for final evaluation

**Methods**: Hyperband, BOHB (Bayesian Optimization Hyperband)

## Neural Architecture Search Integration

NAS can be integrated into AutoML pipelines to automatically discover architectures.

### Integration Points

**End-to-end**: NAS as part of full AutoML pipeline
**Component**: NAS for specific components (e.g., feature extractors)
**Hybrid**: Combine hand-designed and searched architectures

### Search Spaces

**Macro**: Overall architecture structure
**Micro**: Cell structures
**Operations**: Choice of operations

### Efficiency Considerations

**Weight sharing**: Share weights across architectures
**Early stopping**: Stop training poor architectures early
**Proxy tasks**: Evaluate on smaller tasks

### AutoML-NAS Integration

**Unified search**: Search architecture and hyperparameters together
**Hierarchical**: Search architecture first, then hyperparameters
**Iterative**: Alternate between architecture and hyperparameter search

## Feature Engineering Automation

Automated feature engineering creates, selects, and transforms features without manual intervention.

### Feature Creation

**Transformations**: Mathematical transformations (log, sqrt, etc.)
**Interactions**: Products, ratios of features
**Binning**: Discretization of continuous features
**Encoding**: Categorical encoding (one-hot, target encoding)

### Automated Methods

**Genetic programming**: Evolve feature transformations
**Deep feature synthesis**: Hierarchical feature construction
**Neural architecture search**: Learn feature transformations

### Feature Selection

**Filter methods**: Statistical tests (correlation, mutual information)
**Wrapper methods**: Evaluate subsets with model
**Embedded methods**: Feature selection during training (L1 regularization)

### Automated Selection

**Auto-sklearn**: Uses various selection strategies
**TPOT**: Genetic programming for feature pipelines
**Featuretools**: Deep feature synthesis

### Challenges

**Search space**: Extremely large space of possible features
**Overfitting**: Risk of overfitting to validation set
**Interpretability**: Generated features may be hard to interpret

## Auto-sklearn and AutoGluon

Auto-sklearn and AutoGluon are popular AutoML frameworks.

### Auto-sklearn

**Components**:
- **Meta-learning**: Use performance on similar datasets
- **Bayesian optimization**: For hyperparameter tuning
- **Ensemble construction**: Automatically build ensembles

**Pipeline**:
1. Preprocessing: Imputation, encoding, scaling
2. Feature preprocessing: Feature selection, PCA
3. Classifier/Regressor: Various scikit-learn algorithms
4. Ensemble: Combine best models

**Meta-learning**: 
- Database of dataset characteristics and best configurations
- Find similar datasets, warm-start search

**Advantages**: Strong performance, easy to use
**Limitations**: Limited to scikit-learn algorithms, can be slow

### AutoGluon

**Components**:
- **Neural architecture search**: For deep learning
- **Hyperparameter optimization**: For all models
- **Ensemble**: Stacking and weighted ensembles

**Models**:
- LightGBM, CatBoost, XGBoost
- Neural networks (with NAS)
- Tabular, image, text models

**Advantages**: Fast, supports deep learning, good performance
**Limitations**: Less customizable than Auto-sklearn

### Comparison

| Framework | Models | Meta-learning | Speed | Use Case |
|-----------|--------|---------------|-------|----------|
| Auto-sklearn | Scikit-learn | Yes | Medium | Tabular data |
| AutoGluon | DL + Tree | Limited | Fast | Tabular, images, text |
| TPOT | Genetic | No | Slow | Feature engineering |

## Model Selection and Ensembling

AutoML systems must select and combine models effectively.

### Model Selection

**Algorithms**: Choose from set of algorithms
**Criteria**: Accuracy, speed, interpretability
**Multi-objective**: Balance multiple objectives

### Ensembling Strategies

**Voting**: Majority or weighted voting
**Stacking**: Train meta-learner on predictions
**Blending**: Simple weighted average
**Bagging**: Bootstrap aggregating
**Boosting**: Sequential ensemble

### Automated Ensembling

**Selection**: Choose which models to include
**Weighting**: Determine weights for combination
**Validation**: Use validation set to optimize ensemble

**Methods**:
- **Caruana et al.**: Greedy selection
- **Auto-sklearn**: Ensemble selection with replacement
- **AutoGluon**: Stacking with cross-validation

### Ensemble Construction

**Diversity**: Ensure ensemble diversity
**Performance**: Include high-performing models
**Size**: Balance performance and complexity

## Meta-Learning for AutoML

Meta-learning uses experience from previous tasks to improve AutoML.

### Concept

**Learning to learn**: Use meta-knowledge to improve learning
**Transfer**: Transfer knowledge across tasks
**Warm-start**: Initialize search with good configurations

### Meta-Features

**Dataset characteristics**: Number of samples, features, classes
**Statistical properties**: Skewness, kurtosis, correlations
**Performance**: Best configurations on similar datasets

### Applications

**Hyperparameter initialization**: Start search near good configurations
**Architecture search**: Use architectures that worked on similar tasks
**Pipeline construction**: Build pipelines based on task type

### Methods

**Landmarking**: Train simple models to predict best algorithm
**Surrogate models**: Model performance as function of meta-features
**Neural architecture search**: Use meta-learned architectures

### Challenges

**Meta-feature extraction**: How to characterize datasets?
**Similarity**: How to measure dataset similarity?
**Generalization**: Will meta-knowledge transfer to new tasks?

## End-to-End AutoML Systems

End-to-end AutoML systems automate the entire pipeline.

### Google Cloud AutoML

**Components**: Automated data preparation, feature engineering, model selection, deployment
**Models**: Tables, vision, language, translation
**Advantages**: Easy to use, scalable
**Limitations**: Black box, limited customization

### H2O AutoML

**Components**: Data preprocessing, algorithm selection, hyperparameter tuning, ensembling
**Models**: GLM, random forest, GBM, deep learning
**Advantages**: Open source, good performance
**Limitations**: Requires more expertise than cloud solutions

### Microsoft Azure AutoML

**Components**: Automated feature engineering, algorithm selection, hyperparameter tuning
**Models**: Various algorithms
**Advantages**: Integrated with Azure, good documentation
**Limitations**: Vendor lock-in

### Design Principles

**Usability**: Easy for non-experts
**Performance**: Competitive with manual tuning
**Efficiency**: Reasonable time and compute
**Interpretability**: Provide insights into choices
**Flexibility**: Allow expert customization

### Evaluation

**Benchmarks**: OpenML, Kaggle competitions
**Metrics**: Accuracy, time to solution, robustness
**Comparison**: vs manual tuning, vs other AutoML systems

## Key Takeaways

1. **Automated Machine Learning** automates the end-to-end ML pipeline, from data preparation to model deployment, democratizing ML and saving expert time.

2. **Hyperparameter optimization** is core to AutoML, with Bayesian optimization, Hyperband, and TPE being key methods for efficient search.

3. **Neural Architecture Search** can be integrated into AutoML to automatically discover architectures, though efficiency remains a challenge.

4. **Feature engineering automation** creates and selects features automatically, though the search space is extremely large and overfitting is a risk.

5. **Auto-sklearn** uses meta-learning and Bayesian optimization for scikit-learn algorithms, while **AutoGluon** supports deep learning and is faster.

6. **Model selection and ensembling** automatically choose and combine models, with diversity and performance being key considerations.

7. **Meta-learning** uses experience from previous tasks to improve AutoML through warm-starting and transfer learning.

8. **End-to-end AutoML systems** (Google Cloud AutoML, H2O AutoML, Azure AutoML) automate entire pipelines with varying levels of customization.

9. **Challenges** include large search spaces, evaluation costs, multi-objective optimization, and balancing automation with user control.

10. **Future directions** include improving efficiency, better meta-learning, handling more data types, and increasing interpretability and user control.

## References

- Hutter, F., et al. (2019). "Automated Machine Learning: Methods, Systems, Challenges." Springer
- Feurer, M., et al. (2015). "Efficient and Robust Automated Machine Learning." NeurIPS 2015
- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." JMLR 13, 281-305
- Snoek, J., et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." NeurIPS 2012
- Li, L., et al. (2017). "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." JMLR 18, 1-52
- Falkner, S., et al. (2018). "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." ICML 2018
- Olson, R. S., & Moore, J. H. (2016). "TPOT: A Tree-Based Pipeline Optimization Tool for Automating Machine Learning." ICML AutoML Workshop
- Erickson, N., et al. (2020). "AutoGluon-Tabular: Robust and Accurate AutoML for Tabular Data." arXiv:2003.06505
- Vanschoren, J. (2018). "Meta-Learning: A Survey." arXiv:1810.03548
- Zöller, M.-A., & Huber, M. F. (2021). "Benchmark and Survey of Automated Machine Learning Frameworks." JMLR 22, 1-61
