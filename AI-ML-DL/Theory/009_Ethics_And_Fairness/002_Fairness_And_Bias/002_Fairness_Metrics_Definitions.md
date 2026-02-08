# Fairness Metrics and Definitions

## Table of Contents

1. [Introduction to Fairness Metrics](#introduction-to-fairness-metrics)
2. [Demographic Parity](#demographic-parity)
3. [Equalized Odds](#equalized-odds)
4. [Equal Opportunity](#equal-opportunity)
5. [Calibration](#calibration)
6. [Predictive Parity](#predictive-parity)
7. [Individual Fairness](#individual-fairness)
8. [Counterfactual Fairness](#counterfactual-fairness)
9. [Group vs. Individual Fairness](#group-vs-individual-fairness)
10. [Impossibility Theorems](#impossibility-theorems)
11. [Metric Trade-offs](#metric-trade-offs)
12. [Key Takeaways](#key-takeaways)

## Introduction to Fairness Metrics

Fairness metrics provide quantitative measures for evaluating whether AI systems treat different groups fairly. These metrics operationalize abstract notions of fairness into testable mathematical conditions, enabling systematic evaluation and comparison of algorithmic fairness.

### The Need for Fairness Metrics

Fairness is a complex, multi-faceted concept that cannot be captured by a single metric. Different fairness metrics formalize different intuitions about what constitutes fair treatment:

- **Procedural fairness**: Fairness in the decision-making process
- **Distributive fairness**: Fairness in the distribution of outcomes
- **Individual fairness**: Similar individuals treated similarly
- **Group fairness**: Equal treatment across groups

### Mathematical Framework

We formalize fairness evaluation in the following setting:

- **Input space**: $\mathcal{X}$ (features)
- **Output space**: $\mathcal{Y}$ (predictions/decisions)
- **Protected attributes**: $A \in \mathcal{A}$ (e.g., race, gender)
- **True labels**: $Y \in \{0, 1\}$ (binary classification)
- **Predictions**: $\hat{Y} = f(X)$
- **Groups**: $G_a = \{x : A = a\}$ for each protected attribute value $a$

### Types of Fairness Metrics

**Group Fairness Metrics**: Compare statistics across groups:
- Demographic parity
- Equalized odds
- Equal opportunity
- Calibration

**Individual Fairness Metrics**: Compare treatment of similar individuals:
- Individual fairness
- Counterfactual fairness

**Hybrid Metrics**: Combine group and individual considerations:
- Fairness through awareness
- Fairness through unawareness (with constraints)

### Evaluation Context

Fairness metrics must be interpreted in context:

- **Base rates**: Different groups may have different base rates of positive outcomes
- **Costs**: Different errors may have different costs for different groups
- **Rights**: Some applications may have rights-based requirements
- **Utility**: Fairness may trade off with overall utility

## Demographic Parity

Demographic parity (also called statistical parity or group fairness) requires that the proportion of positive predictions is equal across protected groups.

### Formal Definition

A classifier $f$ satisfies demographic parity if:

$$P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = a') \quad \forall a, a' \in \mathcal{A}$$

In other words, the probability of a positive prediction is the same regardless of protected attribute value.

### Mathematical Formulation

For binary classification with groups $G_0$ and $G_1$:

$$\text{Demographic Parity} = |P(\hat{Y} = 1 | A = 0) - P(\hat{Y} = 1 | A = 1)| = 0$$

We can measure deviation from demographic parity as:

$$\text{Demographic Parity Gap} = \max_{a, a'} |P(\hat{Y} = 1 | A = a) - P(\hat{Y} = 1 | A = a')|$$

### Interpretation

**Intuition**: Each group receives positive predictions at the same rate, regardless of actual outcomes.

**When Appropriate**:
- When we want equal representation in positive predictions
- When historical discrimination has led to unequal representation
- When the goal is proportional representation

**Limitations**:
- Ignores actual outcomes (true labels)
- May require different accuracy for different groups
- May not be appropriate when base rates differ legitimately

### Example

Consider a hiring system:
- Group A: 100 applicants, 50 qualified (base rate 50%)
- Group B: 100 applicants, 20 qualified (base rate 20%)

Demographic parity requires hiring equal proportions from both groups (e.g., 30% from each), even though Group A has more qualified candidates.

### Relaxed Versions

**$\epsilon$-Demographic Parity**: Allow small differences:

$$|P(\hat{Y} = 1 | A = a) - P(\hat{Y} = 1 | A = a')| \leq \epsilon$$

**Proportional Representation**: Match representation to population proportions rather than requiring equality.

## Equalized Odds

Equalized odds (also called conditional procedure accuracy equality) requires that true positive rates (TPR) and false positive rates (FPR) are equal across groups.

### Formal Definition

A classifier $f$ satisfies equalized odds if:

$$P(\hat{Y} = 1 | Y = y, A = a) = P(\hat{Y} = 1 | Y = y, A = a') \quad \forall y \in \{0, 1\}, \forall a, a' \in \mathcal{A}$$

This means:
- **Equal TPR**: $P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = a')$
- **Equal FPR**: $P(\hat{Y} = 1 | Y = 0, A = a) = P(\hat{Y} = 1 | Y = 0, A = a')$

### Mathematical Formulation

For groups $G_0$ and $G_1$:

$$\text{Equalized Odds} = \max\{|TPR_0 - TPR_1|, |FPR_0 - FPR_1|\} = 0$$

where:
- $TPR_a = P(\hat{Y} = 1 | Y = 1, A = a)$
- $FPR_a = P(\hat{Y} = 1 | Y = 0, A = a)$

### Interpretation

**Intuition**: The classifier has equal accuracy for positive and negative cases across groups. Mistakes are made at the same rate for each group.

**When Appropriate**:
- When we want equal accuracy across groups
- When both types of errors matter equally
- When base rates may legitimately differ

**Advantages over Demographic Parity**:
- Considers actual outcomes
- Allows different prediction rates if justified by different base rates
- More aligned with accuracy goals

### Example

Consider a medical diagnosis system:
- Group A: 100 patients, 50 have disease (TPR = 80%, FPR = 10%)
- Group B: 100 patients, 20 have disease (TPR = 60%, FPR = 20%)

Equalized odds requires equalizing both TPR (currently 80% vs 60%) and FPR (currently 10% vs 20%).

### Relaxed Versions

**$\epsilon$-Equalized Odds**: Allow small differences:

$$\max\{|TPR_a - TPR_{a'}|, |FPR_a - FPR_{a'}|\} \leq \epsilon$$

**Cost-Weighted Equalized Odds**: Weight TPR and FPR by their costs, which may differ across groups.

## Equal Opportunity

Equal opportunity is a relaxation of equalized odds that requires only equal true positive rates (TPR) across groups, allowing false positive rates to differ.

### Formal Definition

A classifier $f$ satisfies equal opportunity if:

$$P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = a') \quad \forall a, a' \in \mathcal{A}$$

In other words, qualified individuals (those with $Y = 1$) have equal probability of receiving positive predictions regardless of protected attribute.

### Mathematical Formulation

$$\text{Equal Opportunity Gap} = |TPR_0 - TPR_1| = 0$$

where $TPR_a = P(\hat{Y} = 1 | Y = 1, A = a)$.

### Interpretation

**Intuition**: Qualified individuals from all groups have equal opportunity to receive positive predictions. This focuses on "benefits" (positive predictions for qualified individuals) rather than "harms" (false positives).

**When Appropriate**:
- When positive predictions represent opportunities (jobs, loans, admissions)
- When false positives are less concerning than false negatives
- When the goal is ensuring qualified individuals aren't disadvantaged

**Advantages**:
- Less restrictive than equalized odds
- Focuses on the most critical fairness concern (qualified individuals)
- Allows flexibility in handling false positives

### Example

Consider a college admissions system:
- Group A: 1000 applicants, 200 qualified, admit 150 (TPR = 75%)
- Group B: 1000 applicants, 200 qualified, admit 100 (TPR = 50%)

Equal opportunity requires equalizing TPR, so Group B's qualified applicants should also have 75% admission rate.

### Relationship to Other Metrics

**Weaker than Equalized Odds**: Equal opportunity is satisfied whenever equalized odds is satisfied, but not vice versa.

**Stronger than Demographic Parity**: Equal opportunity considers actual qualifications, while demographic parity does not.

## Calibration

Calibration requires that predicted probabilities accurately reflect actual probabilities across groups. For individuals with predicted probability $p$, the actual proportion with positive outcome should be approximately $p$.

### Formal Definition

A classifier $f$ with probability predictions $\hat{P}(Y = 1 | X)$ is calibrated if:

$$P(Y = 1 | \hat{P}(Y = 1 | X) = p, A = a) = p \quad \forall p \in [0, 1], \forall a \in \mathcal{A}$$

Within each group, individuals with predicted probability $p$ should have actual positive rate $p$.

### Group Calibration

Group calibration requires calibration within each group:

$$P(Y = 1 | \hat{P}(Y = 1 | X) = p, A = a) = p \quad \forall a \in \mathcal{A}$$

### Mathematical Formulation

**Calibration Error**: Measure deviation from perfect calibration:

$$\text{Calibration Error}_a = \mathbb{E}_{p \sim \hat{P}}[|P(Y = 1 | \hat{P} = p, A = a) - p|]$$

**Expected Calibration Error (ECE)**: Bin predictions and measure calibration within bins:

$$\text{ECE}_a = \sum_{b=1}^{B} \frac{|B_b|}{n} |\text{acc}(B_b) - \text{conf}(B_b)|$$

where $B_b$ are bins, $\text{acc}(B_b)$ is accuracy in bin $b$, and $\text{conf}(B_b)$ is average confidence in bin $b$.

### Interpretation

**Intuition**: When the model predicts probability $p$, it should be correct $p$ of the time, and this should hold within each group.

**When Appropriate**:
- When probability estimates are used for decision-making
- When risk assessment requires accurate probabilities
- When decisions depend on confidence levels

**Advantages**:
- Ensures probability estimates are meaningful
- Important for applications using probability thresholds
- Enables informed decision-making

### Example

Consider a credit risk model:
- Group A: 100 applicants with predicted risk 0.2, actual default rate 0.18 (well-calibrated)
- Group B: 100 applicants with predicted risk 0.2, actual default rate 0.25 (poorly calibrated)

Calibration requires that Group B's actual default rate matches predicted risk (0.2).

### Relationship to Other Metrics

**Independent of Other Metrics**: Calibration can be satisfied independently of demographic parity, equalized odds, etc.

**Compatibility**: Calibration can be combined with other fairness metrics, though trade-offs may exist.

## Predictive Parity

Predictive parity (also called sufficiency) requires that positive predictive value (PPV) and negative predictive value (NPV) are equal across groups.

### Formal Definition

A classifier $f$ satisfies predictive parity if:

$$P(Y = 1 | \hat{Y} = 1, A = a) = P(Y = 1 | \hat{Y} = 1, A = a') \quad \forall a, a' \in \mathcal{A}$$

and

$$P(Y = 0 | \hat{Y} = 0, A = a) = P(Y = 0 | \hat{Y} = 0, A = a') \quad \forall a, a' \in \mathcal{A}$$

This means:
- **Equal PPV**: $PPV_a = P(Y = 1 | \hat{Y} = 1, A = a)$ is equal across groups
- **Equal NPV**: $NPV_a = P(Y = 0 | \hat{Y} = 0, A = a)$ is equal across groups

### Mathematical Formulation

$$\text{Predictive Parity Gap} = \max\{|PPV_0 - PPV_1|, |NPV_0 - NPV_1|\} = 0$$

### Interpretation

**Intuition**: Among those who receive positive (or negative) predictions, the proportion who actually have positive (or negative) outcomes should be the same across groups.

**When Appropriate**:
- When predictions are used as evidence or signals
- When we want predictions to have the same meaning across groups
- When post-prediction decisions depend on prediction reliability

**Example Context**: In criminal justice risk assessment, predictive parity would mean that "high risk" predictions have the same meaning (same actual recidivism rate) regardless of group.

### Relationship to Other Metrics

**Different Focus**: Predictive parity focuses on prediction reliability, while equalized odds focuses on error rates.

**Incompatibility**: Predictive parity and equalized odds cannot both be satisfied (except in special cases) when base rates differ, as shown by impossibility theorems.

## Individual Fairness

Individual fairness requires that similar individuals receive similar predictions, regardless of group membership.

### Formal Definition

A classifier $f$ satisfies individual fairness with respect to a distance metric $d$ if:

$$|f(x) - f(x')| \leq d(x, x') \quad \forall x, x' \in \mathcal{X}$$

where $d(x, x')$ measures similarity between individuals $x$ and $x'$.

### Mathematical Formulation

**Lipschitz Condition**: Individual fairness can be formalized as a Lipschitz condition:

$$|f(x) - f(x')| \leq L \cdot d(x, x')$$

where $L$ is the Lipschitz constant.

**Fairness Violation**: Measure violations:

$$\text{Individual Fairness Violation} = \max_{x, x'} \frac{|f(x) - f(x')|}{d(x, x')}$$

### Distance Metrics

**Task-Specific**: Distance metrics should reflect task-relevant similarity:
- For hiring: similarity in qualifications, experience, skills
- For lending: similarity in creditworthiness factors
- For healthcare: similarity in medical conditions, risk factors

**Protected Attributes**: Distance metrics should not directly use protected attributes, but may consider their effects through other features.

### Interpretation

**Intuition**: Two individuals who are similar in all relevant ways should receive similar predictions, regardless of their protected attributes.

**Advantages**:
- Protects individuals, not just groups
- Can handle intersectional identities
- Aligns with intuitive notions of fairness

**Challenges**:
- Defining appropriate distance metrics
- Computational complexity
- May conflict with group fairness

### Example

Consider a loan application system:
- Applicant A: Income $50k, credit score 700, debt $10k
- Applicant B: Income $51k, credit score 701, debt $9k
- Applicant C: Income $50k, credit score 700, debt $10k (different race than A)

Individual fairness requires that A and C receive similar predictions (they're identical except for protected attribute), and A and B receive similar predictions (they're very similar).

### Relaxed Versions

**$\epsilon$-Individual Fairness**: Allow small differences:

$$|f(x) - f(x')| \leq d(x, x') + \epsilon$$

**Probabilistic Individual Fairness**: Require fairness in expectation over randomness in the classifier.

## Counterfactual Fairness

Counterfactual fairness requires that predictions would be the same if an individual's protected attribute were changed, holding all else constant.

### Formal Definition

A classifier $f$ satisfies counterfactual fairness if:

$$P(\hat{Y}_{A \leftarrow a}(U) = y | X = x, A = a) = P(\hat{Y}_{A \leftarrow a'}(U) = y | X = x, A = a)$$

where $\hat{Y}_{A \leftarrow a}(U)$ is the counterfactual prediction if protected attribute were set to $a$, and $U$ represents unobserved variables.

### Causal Framework

Counterfactual fairness uses causal models:
- **Structural Causal Models (SCMs)**: Represent causal relationships
- **Interventions**: Change protected attribute values
- **Counterfactuals**: Predictions under counterfactual conditions

### Mathematical Formulation

Using causal graphs and do-calculus:

$$P(\hat{Y} | do(A = a), X = x) = P(\hat{Y} | do(A = a'), X = x)$$

where $do(A = a)$ represents an intervention setting $A$ to $a$.

### Interpretation

**Intuition**: An individual's prediction should not depend on their protected attribute, in the sense that changing only the protected attribute (and causally affected variables) should not change the prediction.

**When Appropriate**:
- When we want to remove causal influence of protected attributes
- When we can model causal relationships
- When we want to ensure protected attributes don't causally affect predictions

**Advantages**:
- Addresses causal discrimination
- Can handle complex causal structures
- Aligns with legal notions of discrimination

**Challenges**:
- Requires causal models
- Identifying causal structure is difficult
- May require strong assumptions

### Example

Consider a hiring system where:
- Gender affects education (due to historical discrimination)
- Education affects hiring
- Gender also directly affects hiring (direct discrimination)

Counterfactual fairness requires that if we change an individual's gender (and thus their education), their hiring probability should remain the same, removing both direct and indirect discrimination.

## Group vs. Individual Fairness

Group fairness and individual fairness represent different approaches to fairness, each with advantages and limitations.

### Group Fairness

**Focus**: Equal treatment across groups defined by protected attributes.

**Metrics**: Demographic parity, equalized odds, equal opportunity, calibration.

**Advantages**:
- Addresses historical group-based discrimination
- Easier to measure and enforce
- Aligns with legal frameworks
- Addresses systemic inequality

**Limitations**:
- May allow unfair treatment of individuals
- Doesn't handle intersectionality well
- May conflict with individual fairness
- Ignores within-group variation

### Individual Fairness

**Focus**: Similar individuals receive similar treatment.

**Metrics**: Individual fairness, counterfactual fairness.

**Advantages**:
- Protects individuals directly
- Handles intersectionality naturally
- Aligns with intuitive fairness
- Doesn't require group definitions

**Limitations**:
- Difficult to define similarity metrics
- Computationally expensive
- May conflict with group fairness
- May not address systemic inequality

### Tensions

**Incompatibility**: Group fairness and individual fairness can conflict:
- Group fairness may require different treatment of similar individuals from different groups
- Individual fairness may require different group-level outcomes

**Example**: To achieve demographic parity, we may need to treat similar individuals differently based on group membership, violating individual fairness.

### Hybrid Approaches

**Fairness Through Awareness**: Combine group and individual considerations:
- Use group membership to inform individual fairness
- Define similarity metrics that account for group-specific factors
- Balance group and individual fairness objectives

**Multi-Accountability**: Satisfy multiple fairness criteria simultaneously when possible, accept trade-offs when necessary.

## Impossibility Theorems

Impossibility theorems show that certain combinations of fairness criteria cannot be simultaneously satisfied under realistic conditions.

### Kleinberg's Impossibility Theorem

**Statement**: Under mild conditions, it is impossible to simultaneously satisfy:
1. **Calibration**: $P(Y = 1 | \hat{P} = p, A = a) = p$ for all groups
2. **Balance for Positive Class**: Equal expected predictions for positive cases across groups
3. **Balance for Negative Class**: Equal expected predictions for negative cases across groups

**Conditions**: Requires that base rates differ across groups and predictions are not perfectly accurate.

**Implication**: When base rates differ, we cannot have both well-calibrated probabilities and equal average predictions for positive and negative cases.

### Chouldechova's Impossibility Result

**Statement**: When base rates differ across groups, it is impossible to simultaneously satisfy:
1. **Calibration within Groups**: Well-calibrated predictions within each group
2. **Equal Positive Predictive Value**: Equal PPV across groups
3. **Equal False Positive Rate**: Equal FPR across groups

**Implication**: Predictive parity and equalized odds (specifically equal FPR) cannot both be satisfied when base rates differ and predictions are calibrated.

### Mathematical Proof Sketch

For groups with different base rates $P(Y = 1 | A = a)$:

If calibration holds: $P(Y = 1 | \hat{P} = p, A = a) = p$

If PPV is equal: $P(Y = 1 | \hat{Y} = 1, A = a) = P(Y = 1 | \hat{Y} = 1, A = a')$

If FPR is equal: $P(\hat{Y} = 1 | Y = 0, A = a) = P(\hat{Y} = 1 | Y = 0, A = a')$

These conditions lead to contradictions when base rates differ.

### Implications

**Trade-offs Required**: We must choose which fairness criteria to prioritize:
- Cannot satisfy all desirable criteria simultaneously
- Must make explicit trade-offs
- Context determines which criteria matter most

**Base Rate Considerations**: When base rates legitimately differ, some fairness criteria may be inappropriate:
- Demographic parity may be unfair if base rates differ
- Equalized odds may be more appropriate
- Calibration may conflict with other criteria

**Practical Guidance**: 
- Identify which fairness criteria are most important for the application
- Accept that perfect fairness on all dimensions is impossible
- Make trade-offs explicit and justified

## Metric Trade-offs

Different fairness metrics often conflict, requiring explicit trade-offs in practice.

### Common Trade-offs

**Accuracy vs. Fairness**: Improving fairness often reduces overall accuracy:
- Constraining predictions to satisfy fairness may reduce accuracy
- Balancing accuracy and fairness requires careful optimization

**Different Fairness Metrics**: Satisfying one fairness metric may violate another:
- Demographic parity vs. equalized odds
- Calibration vs. equalized odds
- Group fairness vs. individual fairness

**Different Groups**: Improving fairness for one group may harm another:
- Intersectional groups may have conflicting requirements
- Optimizing for one protected attribute may affect others

### Mathematical Formulation

**Pareto Frontier**: The set of achievable (accuracy, fairness) pairs:

$$\mathcal{F} = \{(A(f), F(f)) : f \in \mathcal{H}\}$$

where $A(f)$ is accuracy and $F(f)$ is fairness (measured by some metric).

**Optimization**: Find models on the Pareto frontier:

$$\min_f L(f) + \lambda \cdot \text{FairnessViolation}(f)$$

where $\lambda$ controls the trade-off.

### Decision Framework

**1. Identify Relevant Metrics**: Determine which fairness metrics matter for the application:
- Legal requirements
- Ethical considerations
- Stakeholder concerns
- Application context

**2. Assess Trade-offs**: Understand conflicts between metrics:
- Mathematical incompatibilities
- Empirical trade-offs
- Cost-benefit analysis

**3. Prioritize**: Decide which metrics to optimize:
- Primary objectives
- Constraints (must satisfy)
- Secondary objectives (nice to have)

**4. Optimize**: Find solutions balancing objectives:
- Multi-objective optimization
- Constrained optimization
- Pareto-optimal solutions

**5. Validate**: Ensure trade-offs are acceptable:
- Stakeholder approval
- Legal compliance
- Ethical review

### Examples

**Hiring System**:
- Trade-off: Demographic parity vs. accuracy
- Decision: May accept lower accuracy to achieve demographic parity if addressing historical discrimination is priority

**Medical Diagnosis**:
- Trade-off: Equalized odds vs. calibration
- Decision: Prioritize equalized odds to ensure equal accuracy, accept some calibration differences

**Credit Scoring**:
- Trade-off: Predictive parity vs. equal opportunity
- Decision: Prioritize predictive parity to ensure predictions have same meaning, accept some opportunity differences

### Best Practices

**Transparency**: Make trade-offs explicit and documented:
- Which metrics were prioritized
- Why certain trade-offs were made
- What alternatives were considered

**Stakeholder Engagement**: Involve stakeholders in trade-off decisions:
- Affected communities
- Domain experts
- Legal and compliance
- Business stakeholders

**Regular Review**: Reassess trade-offs as context evolves:
- Changing legal requirements
- New understanding of impacts
- Evolving stakeholder concerns
- Performance monitoring

## Key Takeaways

1. **Multiple metrics exist**: Different fairness metrics capture different intuitions about fairness, and no single metric is universally appropriate.

2. **Context matters**: The appropriate fairness metric depends on the application context, legal requirements, and stakeholder values.

3. **Mathematical precision**: Fairness metrics provide precise, testable definitions that enable systematic evaluation and comparison.

4. **Trade-offs are inevitable**: Different fairness metrics often conflict, and fairness may trade off with accuracy, requiring explicit decision-making.

5. **Impossibility results**: Certain combinations of fairness criteria cannot be simultaneously satisfied when base rates differ, requiring prioritization.

6. **Group vs. individual**: Group fairness and individual fairness represent different approaches, each with advantages and limitations.

7. **Calibration is important**: Well-calibrated probability estimates are crucial for informed decision-making, though they may conflict with other fairness criteria.

8. **Causal considerations**: Counterfactual fairness addresses causal discrimination but requires causal models and strong assumptions.

9. **Measurement challenges**: Defining and measuring fairness requires careful consideration of distance metrics, similarity definitions, and evaluation procedures.

10. **Ongoing process**: Fairness evaluation is not one-time but requires continuous monitoring and reassessment as systems and contexts evolve.
