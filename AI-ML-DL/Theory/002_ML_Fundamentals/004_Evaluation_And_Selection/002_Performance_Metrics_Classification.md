# Performance Metrics Classification

## Table of Contents

1. [Introduction to Classification Metrics](#introduction-to-classification-metrics)
2. [Confusion Matrix](#confusion-matrix)
3. [Accuracy and Error Rate](#accuracy-and-error-rate)
4. [Precision and Recall](#precision-and-recall)
5. [F1-Score and Harmonic Mean](#f1-score-and-harmonic-mean)
6. [ROC Curve and AUC](#roc-curve-and-auc)
7. [Precision-Recall Curve](#precision-recall-curve)
8. [Multi-Class Metrics](#multi-class-metrics)
9. [Class Imbalance Handling](#class-imbalance-handling)
10. [Key Takeaways](#key-takeaways)

## Introduction to Classification Metrics

Classification metrics quantify how well a classifier predicts class labels, essential for model evaluation and comparison.

### Why Metrics Matter

- **Model Evaluation**: Assess classifier performance
- **Model Selection**: Compare different models
- **Hyperparameter Tuning**: Optimize model parameters
- **Business Decisions**: Translate performance to business value

### Types of Metrics

- **Threshold-Dependent**: Require classification threshold (accuracy, precision, recall)
- **Threshold-Independent**: Work across all thresholds (AUC, PR-AUC)
- **Per-Class**: Evaluate each class separately
- **Aggregate**: Summarize overall performance

### Choosing Metrics

Consider:
- **Problem Type**: Binary vs. multi-class
- **Class Balance**: Balanced vs. imbalanced
- **Business Context**: Cost of different error types
- **Interpretability**: Need for explainable metrics

## Confusion Matrix

Confusion matrix provides detailed breakdown of classification performance.

### Binary Classification

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN (True Negative) | FP (False Positive) |
| **Actual Positive** | FN (False Negative) | TP (True Positive) |

**Definitions**:
- **TP**: Correctly predicted positive
- **TN**: Correctly predicted negative
- **FP**: Incorrectly predicted positive (Type I error)
- **FN**: Incorrectly predicted negative (Type II error)

### Multi-Class Confusion Matrix

$k \times k$ matrix where entry $(i,j)$ is count of class $i$ predicted as class $j$.

**Diagonal**: Correct predictions
**Off-Diagonal**: Misclassifications

### Information from Confusion Matrix

- **Overall Performance**: Sum of diagonal / total
- **Per-Class Performance**: Row/column analysis
- **Error Patterns**: Which classes are confused
- **Class-Specific Metrics**: Precision, recall per class

## Accuracy and Error Rate

Simplest metrics measuring overall correctness.

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Range**: $[0, 1]$, higher is better

**Interpretation**: Fraction of correctly classified instances

### Error Rate

$$\text{Error Rate} = 1 - \text{Accuracy} = \frac{FP + FN}{TP + TN + FP + FN}$$

**Range**: $[0, 1]$, lower is better

### Advantages

- Simple and intuitive
- Easy to compute
- Standard baseline metric

### Limitations

- **Misleading for Imbalanced Data**: 
  - Example: 99% class A, 1% class B
  - Always predict A → 99% accuracy
  - But fails to detect any class B instances

- **Equal Weight**: Treats all errors equally
- **No Class-Specific Info**: Doesn't distinguish error types

### When to Use

- Balanced classes
- All errors equally costly
- Quick baseline assessment
- Combined with other metrics

## Precision and Recall

Precision and recall provide class-specific performance measures.

### Precision

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{Predicted Positives}}$$

**Interpretation**: Of instances predicted as positive, what fraction are actually positive?

**Also Called**: Positive Predictive Value (PPV)

**Range**: $[0, 1]$, higher is better

**Focus**: Minimize false positives

### Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{Actual Positives}}$$

**Interpretation**: Of actual positive instances, what fraction are correctly identified?

**Also Called**: Sensitivity, True Positive Rate (TPR)

**Range**: $[0, 1]$, higher is better

**Focus**: Minimize false negatives

### Tradeoff

**Precision-Recall Tradeoff**:
- **High Threshold**: More conservative → Higher precision, lower recall
- **Low Threshold**: More aggressive → Lower precision, higher recall

**Example**: Medical diagnosis
- **High Precision**: Few false alarms, but miss some cases
- **High Recall**: Catch all cases, but many false alarms

### Specificity

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{\text{True Negatives}}{\text{Actual Negatives}}$$

**Interpretation**: Of actual negative instances, what fraction are correctly identified?

**Also Called**: True Negative Rate (TNR)

**Complement**: False Positive Rate (FPR) = $1 - \text{Specificity}$

## F1-Score and Harmonic Mean

F1-score balances precision and recall.

### F1-Score

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**Harmonic Mean**: More conservative than arithmetic mean

**Properties**:
- $0 \leq \text{F1} \leq 1$
- $\text{F1} = 1$ only if precision = recall = 1
- Penalizes imbalance between precision and recall

### Why Harmonic Mean?

**Arithmetic Mean**: $\frac{P + R}{2}$ treats precision and recall equally

**Harmonic Mean**: $\frac{2PR}{P+R}$ penalizes when one is low

**Example**: 
- Precision = 1.0, Recall = 0.1
- Arithmetic: 0.55 (misleadingly high)
- Harmonic: 0.18 (reflects poor performance)

### Fβ-Score

Generalized F-score with parameter $\beta$:

$$\text{F}_\beta = (1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}$$

**$\beta$ Values**:
- $\beta = 1$: F1-score (balanced)
- $\beta > 1$: Emphasizes recall (e.g., $\beta = 2$)
- $\beta < 1$: Emphasizes precision (e.g., $\beta = 0.5$)

**Use Cases**:
- **$\beta > 1$**: Recall more important (e.g., disease detection)
- **$\beta < 1$**: Precision more important (e.g., spam filtering)

## ROC Curve and AUC

ROC curve visualizes performance across all classification thresholds.

### ROC Curve

**X-axis**: False Positive Rate (FPR) = $\frac{FP}{FP + TN} = 1 - \text{Specificity}$

**Y-axis**: True Positive Rate (TPR) = $\frac{TP}{TP + FN} = \text{Recall}$

**Plot**: TPR vs. FPR for different thresholds

### Interpretation

- **Point (0,0)**: Threshold = 1 (predict all negative)
- **Point (1,1)**: Threshold = 0 (predict all positive)
- **Point (0,1)**: Perfect classifier
- **Diagonal Line**: Random classifier

**Above Diagonal**: Better than random
**Below Diagonal**: Worse than random (flip predictions)

### Area Under ROC Curve (AUC-ROC)

**AUC**: Area under ROC curve

**Range**: $[0, 1]$
- **AUC = 1**: Perfect classifier
- **AUC = 0.5**: Random classifier
- **AUC < 0.5**: Worse than random (flip predictions)

### Interpretation of AUC

**Probabilistic Interpretation**: 
AUC = Probability that classifier ranks random positive higher than random negative

**Ranking Quality**: Measures ability to rank instances by probability

### Advantages

- **Threshold-Independent**: Evaluates across all thresholds
- **Scale-Invariant**: Not affected by class distribution (in balanced case)
- **Interpretable**: Clear probabilistic meaning

### Limitations

- **Misleading for Imbalanced Data**: 
  - Can have high AUC but poor precision
  - Example: 99% negatives, 1% positives
  - High AUC possible even if recall is low

- **Focuses on Ranking**: Not directly on classification performance

### When to Use

- Balanced classes
- Need threshold-independent evaluation
- Ranking quality matters
- Comparing models across different thresholds

## Precision-Recall Curve

PR curve focuses on positive class performance.

### PR Curve

**X-axis**: Recall

**Y-axis**: Precision

**Plot**: Precision vs. Recall for different thresholds

### Interpretation

- **Point (1,1)**: Perfect classifier
- **Baseline**: Horizontal line at $\frac{\text{Positives}}{\text{Total}}$ (precision of random classifier)

**Above Baseline**: Better than random
**Below Baseline**: Worse than random

### Area Under PR Curve (AUC-PR)

**AUC-PR**: Area under precision-recall curve

**Range**: $[0, 1]$
- **AUC-PR = 1**: Perfect classifier
- **AUC-PR = Baseline**: Random classifier (baseline = positive class proportion)

### Advantages over ROC

- **Better for Imbalanced Data**: Focuses on positive class
- **More Informative**: Directly shows precision-recall tradeoff
- **Sensitive to Performance**: Changes more noticeably with performance

### When to Use

- Imbalanced classes
- Positive class is minority
- Precision and recall both important
- Need detailed positive class analysis

## Multi-Class Metrics

Extend binary metrics to multiple classes.

### Macro-Averaging

Average metric across all classes:

$$\text{Macro-Precision} = \frac{1}{k}\sum_{i=1}^k \text{Precision}_i$$

$$\text{Macro-Recall} = \frac{1}{k}\sum_{i=1}^k \text{Recall}_i$$

$$\text{Macro-F1} = \frac{1}{k}\sum_{i=1}^k \text{F1}_i$$

**Properties**: Treats all classes equally, regardless of size

### Micro-Averaging

Aggregate TP, FP, FN across all classes:

$$\text{Micro-Precision} = \frac{\sum_{i=1}^k TP_i}{\sum_{i=1}^k TP_i + \sum_{i=1}^k FP_i}$$

$$\text{Micro-Recall} = \frac{\sum_{i=1}^k TP_i}{\sum_{i=1}^k TP_i + \sum_{i=1}^k FN_i}$$

**Properties**: 
- Micro-Precision = Micro-Recall = Accuracy
- Weighted by class frequency

### Weighted-Averaging

Weight by class frequency:

$$\text{Weighted-F1} = \sum_{i=1}^k w_i \text{F1}_i$$

where $w_i = \frac{n_i}{n}$ is proportion of class $i$.

### Per-Class Metrics

Report metrics for each class separately:
- **Precision per Class**: $\text{Precision}_i = \frac{TP_i}{TP_i + FP_i}$
- **Recall per Class**: $\text{Recall}_i = \frac{TP_i}{TP_i + FN_i}$
- **F1 per Class**: Harmonic mean of precision and recall

### Which to Use?

- **Macro**: All classes equally important
- **Micro**: Overall performance (dominated by majority class)
- **Weighted**: Account for class imbalance
- **Per-Class**: Detailed analysis, identify problematic classes

## Class Imbalance Handling

Imbalanced classes require special consideration in metric selection.

### Problem

**Example**: 99% class A, 1% class B
- Accuracy: 99% (misleading)
- Always predict A: 99% accuracy, 0% recall for B

### Solutions

**Use Appropriate Metrics**:
- Precision, Recall, F1 (not just accuracy)
- AUC-PR (better than AUC-ROC for imbalanced data)
- Per-class metrics

**Cost-Sensitive Evaluation**:
- Assign different costs to FP and FN
- Weighted metrics

**Threshold Tuning**:
- Optimize threshold for F1 or business metric
- Not default 0.5

### Matthews Correlation Coefficient (MCC)

Balanced metric for binary classification:

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**Range**: $[-1, 1]$
- **+1**: Perfect prediction
- **0**: Random prediction
- **-1**: Perfect inverse prediction

**Advantages**: 
- Works well with imbalanced data
- Considers all four confusion matrix entries

### Cohen's Kappa

Agreement metric accounting for chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where:
- $p_o$: Observed agreement (accuracy)
- $p_e$: Expected agreement by chance

**Interpretation**:
- $\kappa = 1$: Perfect agreement
- $\kappa = 0$: Agreement by chance
- $\kappa < 0$: Worse than chance

**Advantages**: Accounts for class imbalance

## Key Takeaways

1. **Classification Metrics** quantify classifier performance, with threshold-dependent metrics (accuracy, precision, recall) and threshold-independent metrics (AUC) serving different purposes.

2. **Confusion Matrix** provides detailed breakdown (TP, TN, FP, FN) enabling computation of all metrics and analysis of error patterns per class.

3. **Accuracy** measures overall correctness $\frac{TP+TN}{Total}$ but is misleading for imbalanced data, where always predicting majority class yields high accuracy despite poor performance.

4. **Precision** $\frac{TP}{TP+FP}$ measures correctness of positive predictions, while **Recall** $\frac{TP}{TP+FN}$ measures coverage of actual positives, with inherent tradeoff controlled by threshold.

5. **F1-Score** $\frac{2PR}{P+R}$ balances precision and recall using harmonic mean, penalizing imbalance, with Fβ-score allowing emphasis on precision ($\beta<1$) or recall ($\beta>1$).

6. **ROC Curve** plots TPR vs. FPR across thresholds, with **AUC-ROC** measuring ranking quality (probability positive ranked higher than negative), threshold-independent but can be misleading for imbalanced data.

7. **Precision-Recall Curve** plots precision vs. recall, with **AUC-PR** better for imbalanced data by focusing on positive class, more sensitive to performance changes than ROC.

8. **Multi-Class Metrics** include macro-averaging (equal weight per class), micro-averaging (aggregate TP/FP/FN, equals accuracy), weighted-averaging (weight by frequency), and per-class metrics for detailed analysis.

9. **Class Imbalance** requires metrics beyond accuracy: use precision/recall/F1, AUC-PR, MCC (Matthews Correlation Coefficient), or Cohen's Kappa, with threshold tuning for optimal performance.

10. **Metric Selection** depends on problem (binary vs. multi-class), class balance (balanced vs. imbalanced), business context (cost of errors), and interpretability needs, with F1 and AUC-PR recommended for imbalanced data, and ROC-AUC for balanced threshold-independent evaluation.
