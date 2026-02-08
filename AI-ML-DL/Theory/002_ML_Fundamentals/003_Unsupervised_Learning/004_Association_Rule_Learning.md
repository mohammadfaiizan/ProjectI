# Association Rule Learning

## Table of Contents

1. [Introduction to Association Rules](#introduction-to-association-rules)
2. [Market Basket Analysis](#market-basket-analysis)
3. [Support, Confidence, and Lift](#support-confidence-and-lift)
4. [Apriori Algorithm](#apriori-algorithm)
5. [FP-Growth Algorithm](#fp-growth-algorithm)
6. [Rule Generation and Pruning](#rule-generation-and-pruning)
7. [Advanced Metrics](#advanced-metrics)
8. [Applications Beyond Market Baskets](#applications-beyond-market-baskets)
9. [Challenges and Limitations](#challenges-and-limitations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Association Rules

Association rule learning discovers interesting relationships between variables in large datasets, commonly used for market basket analysis.

### What are Association Rules?

An association rule is an implication of the form:

$$X \Rightarrow Y$$

where $X$ and $Y$ are sets of items (itemsets), meaning "if $X$ occurs, then $Y$ is likely to occur."

**Example**: $\{\text{bread}, \text{butter}\} \Rightarrow \{\text{milk}\}$

### Components

- **Antecedent (LHS)**: Left-hand side, condition $X$
- **Consequent (RHS)**: Right-hand side, result $Y$
- **Itemset**: Set of items (e.g., $\{\text{bread}, \text{milk}\}$)
- **Transaction**: Set of items purchased together

### Problem Formulation

Given:
- Set of items $\mathcal{I} = \{i_1, i_2, \ldots, i_m\}$
- Database of transactions $\mathcal{D} = \{T_1, T_2, \ldots, T_n\}$
- Each transaction $T_j \subseteq \mathcal{I}$

Find: Association rules that satisfy minimum support and confidence thresholds.

### Types of Patterns

- **Frequent Itemsets**: Itemsets appearing frequently together
- **Association Rules**: Implications between itemsets
- **Sequential Patterns**: Temporal ordering of itemsets
- **Closed Itemsets**: Maximal frequent itemsets

## Market Basket Analysis

Market basket analysis is the classic application of association rule learning.

### Example

**Transactions**:
- $T_1$: $\{\text{bread}, \text{milk}\}$
- $T_2$: $\{\text{bread}, \text{butter}, \text{milk}\}$
- $T_3$: $\{\text{bread}, \text{butter}\}$
- $T_4$: $\{\text{milk}, \text{cheese}\}$
- $T_5$: $\{\text{bread}, \text{milk}, \text{cheese}\}$

**Possible Rules**:
- $\{\text{bread}\} \Rightarrow \{\text{milk}\}$ (bread buyers often buy milk)
- $\{\text{butter}\} \Rightarrow \{\text{bread}\}$ (butter buyers often buy bread)

### Business Applications

- **Product Placement**: Place related items near each other
- **Cross-Selling**: Recommend complementary products
- **Promotions**: Bundle related items
- **Inventory Management**: Stock items that sell together
- **Customer Segmentation**: Group customers by purchase patterns

## Support, Confidence, and Lift

Key metrics evaluate the quality and interestingness of association rules.

### Support

Support measures how frequently an itemset appears:

$$\text{Support}(X) = \frac{|\{T : X \subseteq T\}|}{|\mathcal{D}|} = P(X)$$

**Support of Rule** $X \Rightarrow Y$:
$$\text{Support}(X \Rightarrow Y) = \frac{|\{T : X \cup Y \subseteq T\}|}{|\mathcal{D}|} = P(X \cup Y)$$

**Interpretation**: Fraction of transactions containing both $X$ and $Y$.

### Confidence

Confidence measures how often $Y$ appears when $X$ appears:

$$\text{Confidence}(X \Rightarrow Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)} = P(Y | X)$$

**Interpretation**: Probability that $Y$ occurs given $X$ occurs.

**Range**: $[0, 1]$, higher is better.

### Lift

Lift measures how much more likely $Y$ is when $X$ occurs compared to baseline:

$$\text{Lift}(X \Rightarrow Y) = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)} = \frac{P(Y | X)}{P(Y)} = \frac{P(X \cup Y)}{P(X)P(Y)}$$

**Interpretation**:
- $\text{Lift} = 1$: $X$ and $Y$ are independent
- $\text{Lift} > 1$: Positive association (more likely together)
- $\text{Lift} < 1$: Negative association (less likely together)

### Example Calculation

Given:
- $\text{Support}(\{\text{bread}\}) = 0.6$
- $\text{Support}(\{\text{milk}\}) = 0.8$
- $\text{Support}(\{\text{bread}, \text{milk}\}) = 0.5$

**Rule**: $\{\text{bread}\} \Rightarrow \{\text{milk}\}$

- **Support**: $0.5$ (50% of transactions contain both)
- **Confidence**: $0.5 / 0.6 = 0.833$ (83.3% of bread buyers also buy milk)
- **Lift**: $0.833 / 0.8 = 1.04$ (slightly positive association)

### Minimum Thresholds

- **Minimum Support**: $\text{min\_sup}$ (e.g., 0.1 or 10%)
- **Minimum Confidence**: $\text{min\_conf}$ (e.g., 0.5 or 50%)

Rules must satisfy both thresholds to be considered interesting.

## Apriori Algorithm

Apriori is the classic algorithm for finding frequent itemsets efficiently.

### Apriori Principle

**Key Insight**: If an itemset is frequent, all its subsets are frequent.

**Contrapositive**: If an itemset is infrequent, all its supersets are infrequent.

**Example**: If $\{\text{bread}, \text{milk}\}$ is infrequent, then $\{\text{bread}, \text{milk}, \text{butter}\}$ is also infrequent.

### Algorithm

**Step 1: Find Frequent 1-Itemsets** ($L_1$)
- Scan database, count support of each item
- Keep items with $\text{Support} \geq \text{min\_sup}$

**Step 2: Generate Candidates** ($C_k$)
- For $k = 2, 3, \ldots$:
  - Generate $C_k$ by joining $L_{k-1}$ with itself
  - Prune candidates with infrequent $(k-1)$-subsets

**Step 3: Count Support**
- Scan database, count support of candidates in $C_k$

**Step 4: Filter Frequent Itemsets** ($L_k$)
- Keep candidates with $\text{Support} \geq \text{min\_sup}$

**Repeat** Steps 2-4 until no more frequent itemsets found.

### Candidate Generation

**Join Step**: 
- For itemsets $l_1, l_2 \in L_{k-1}$:
  - If first $k-2$ items are same and last items differ:
    - Join: $l_1 \cup l_2$

**Prune Step**:
- Remove candidate if any $(k-1)$-subset is not in $L_{k-1}$

### Example

**Database**: 
- $T_1$: $\{A, B, C\}$
- $T_2$: $\{A, B\}$
- $T_3$: $\{B, C\}$
- $T_4$: $\{A, C\}$

**min_sup = 0.5**

**$L_1$**: $\{A\}$ (3), $\{B\}$ (3), $\{C\}$ (3)

**$C_2$**: $\{A, B\}$, $\{A, C\}$, $\{B, C\}$

**$L_2$**: $\{A, B\}$ (2), $\{A, C\}$ (2), $\{B, C\}$ (2)

**$C_3$**: $\{A, B, C\}$ (from join of $\{A, B\}$ and $\{A, C\}$)

**$L_3$**: $\{A, B, C\}$ (1) - but support $1/4 = 0.25 < 0.5$, so empty

### Complexity

- **Time**: $O(2^m)$ worst case, but Apriori principle reduces search space
- **Space**: Stores candidates and frequent itemsets
- **Database Scans**: One per itemset size ($k$)

### Optimizations

- **Hash Trees**: Efficient candidate counting
- **Transaction Reduction**: Remove transactions not containing frequent items
- **Partitioning**: Divide database, find local frequent itemsets, combine
- **Sampling**: Use sample, verify on full database

## FP-Growth Algorithm

FP-Growth uses a compact data structure (FP-tree) to avoid candidate generation.

### FP-Tree (Frequent Pattern Tree)

**Structure**:
- Root node (null)
- Internal nodes: Items with counts
- Links: Connect nodes with same item
- Header table: Links to first occurrence of each item

**Properties**:
- Compact representation
- Preserves frequency information
- Enables efficient mining

### Algorithm

**Step 1: Build FP-Tree**
1. Scan database, count item frequencies
2. Sort items by frequency (descending)
3. For each transaction:
   - Sort items by frequency
   - Insert into FP-tree (increment counts, create nodes if needed)

**Step 2: Mine FP-Tree**
1. Start from bottom of header table (least frequent items)
2. For each item:
   - Find all paths containing item (conditional pattern base)
   - Build conditional FP-tree
   - Recursively mine conditional FP-tree
   - Generate frequent itemsets

### Advantages over Apriori

- **Fewer Database Scans**: Two scans total (vs. one per itemset size)
- **No Candidate Generation**: Directly mines from FP-tree
- **Compact Representation**: FP-tree often smaller than database
- **Faster**: Typically 10-100x faster than Apriori

### Example

**Database**:
- $T_1$: $\{f, a, c, d, g, i, m, p\}$
- $T_2$: $\{a, b, c, f, l, m, o\}$
- $T_3$: $\{b, f, h, j, o\}$
- $T_4$: $\{b, c, k, s, p\}$
- $T_5$: $\{a, f, c, e, l, p, m, n\}$

**Item Frequencies**: $f:4, c:4, a:3, b:3, m:3, p:3$

**FP-Tree**: Compact tree structure with paths sharing common prefixes.

## Rule Generation and Pruning

After finding frequent itemsets, generate and evaluate association rules.

### Rule Generation

For each frequent itemset $L$:
1. Generate all non-empty subsets $X \subset L$
2. For each subset $X$:
   - Rule: $X \Rightarrow (L \setminus X)$
   - Compute confidence
   - Keep if $\text{Confidence} \geq \text{min\_conf}$

### Confidence-Based Pruning

**Property**: If rule $X \Rightarrow Y$ has low confidence, then $X' \Rightarrow Y$ (where $X' \subset X$) also has low confidence.

**Pruning**: Don't generate rules from subsets if parent rule has low confidence.

### Example

**Frequent Itemset**: $\{A, B, C\}$ with support 0.3

**Possible Rules**:
- $\{A\} \Rightarrow \{B, C\}$: Confidence = $0.3 / 0.5 = 0.6$
- $\{B\} \Rightarrow \{A, C\}$: Confidence = $0.3 / 0.6 = 0.5$
- $\{C\} \Rightarrow \{A, B\}$: Confidence = $0.3 / 0.7 = 0.43$
- $\{A, B\} \Rightarrow \{C\}$: Confidence = $0.3 / 0.4 = 0.75$
- $\{A, C\} \Rightarrow \{B\}$: Confidence = $0.3 / 0.45 = 0.67$
- $\{B, C\} \Rightarrow \{A\}$: Confidence = $0.3 / 0.35 = 0.86$

If $\text{min\_conf} = 0.6$, keep rules with confidence $\geq 0.6$.

## Advanced Metrics

Beyond support, confidence, and lift, other metrics evaluate rule quality.

### Conviction

Measures how much more often $X$ would have to occur without $Y$ if they were independent:

$$\text{Conviction}(X \Rightarrow Y) = \frac{1 - \text{Support}(Y)}{1 - \text{Confidence}(X \Rightarrow Y)} = \frac{P(X)P(\neg Y)}{P(X \cup \neg Y)}$$

- $\text{Conviction} = 1$: Independence
- $\text{Conviction} > 1$: Positive association
- Higher conviction indicates stronger rule

### Cosine Similarity

Measures similarity between $X$ and $Y$:

$$\text{Cosine}(X, Y) = \frac{\text{Support}(X \cup Y)}{\sqrt{\text{Support}(X) \times \text{Support}(Y)}} = \frac{P(X \cup Y)}{\sqrt{P(X)P(Y)}}$$

Range: $[0, 1]$, higher indicates stronger association.

### Jaccard Coefficient

Measures overlap:

$$\text{Jaccard}(X, Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X) + \text{Support}(Y) - \text{Support}(X \cup Y)}$$

Range: $[0, 1]$, higher indicates more overlap.

### All-Confidence

Minimum confidence of all rules from itemset:

$$\text{AllConf}(X) = \min_{Y \subset X} \text{Confidence}(Y \Rightarrow (X \setminus Y))$$

Measures how cohesive an itemset is.

## Applications Beyond Market Baskets

Association rules apply to various domains.

### Web Usage Mining

- **Page Co-occurrence**: Pages visited together
- **Session Patterns**: User navigation patterns
- **Recommendations**: Suggest related pages

### Bioinformatics

- **Gene Co-expression**: Genes expressed together
- **Protein Interactions**: Proteins that interact
- **Disease Patterns**: Symptoms that co-occur

### Text Mining

- **Term Co-occurrence**: Words appearing together
- **Topic Discovery**: Themes in documents
- **Document Clustering**: Group similar documents

### Healthcare

- **Symptom Patterns**: Symptoms that co-occur
- **Drug Interactions**: Medications taken together
- **Treatment Patterns**: Treatments used together

### Cybersecurity

- **Attack Patterns**: Events that co-occur in attacks
- **Anomaly Detection**: Unusual co-occurrences
- **Threat Intelligence**: Correlate security events

## Challenges and Limitations

### Computational Complexity

- **Exponential Search Space**: $2^m$ possible itemsets for $m$ items
- **Large Databases**: Millions of transactions
- **Many Items**: Thousands of unique items

**Solutions**: Efficient algorithms (FP-Growth), sampling, parallelization

### Quality of Rules

- **Spurious Rules**: Rules that appear by chance
- **Redundant Rules**: Multiple rules expressing same pattern
- **Trivial Rules**: Obvious or uninteresting rules

**Solutions**: Statistical significance tests, redundancy removal, interestingness measures

### Parameter Selection

- **Minimum Support**: Too high misses interesting patterns, too low generates too many rules
- **Minimum Confidence**: Balance between precision and coverage

**Solutions**: Domain knowledge, iterative refinement, visualization

### Interpretability

- **Too Many Rules**: Thousands of rules hard to interpret
- **Complex Rules**: Rules with many items hard to understand

**Solutions**: Rule summarization, visualization, focusing on top rules

### Scalability

- **Memory**: Storing FP-tree or candidates
- **I/O**: Multiple database scans
- **Distributed**: Parallelizing across machines

**Solutions**: Distributed algorithms, streaming methods, approximate algorithms

## Key Takeaways

1. **Association Rules** discover relationships $X \Rightarrow Y$ between itemsets, measuring how often $Y$ occurs when $X$ occurs, commonly used for market basket analysis.

2. **Support** measures frequency: $\text{Support}(X) = P(X)$, **Confidence** measures conditional probability: $\text{Confidence}(X \Rightarrow Y) = P(Y|X)$, and **Lift** measures association strength: $\text{Lift} = \frac{P(Y|X)}{P(Y)}$.

3. **Apriori Principle** states that if itemset is frequent, all subsets are frequent (and contrapositive), enabling efficient pruning of candidate itemsets.

4. **Apriori Algorithm** finds frequent itemsets by generating candidates level-wise, pruning using Apriori principle, and counting support, requiring multiple database scans.

5. **FP-Growth Algorithm** uses FP-tree (compact tree structure) to avoid candidate generation, requiring only two database scans and typically 10-100x faster than Apriori.

6. **Rule Generation** creates rules $X \Rightarrow (L \setminus X)$ from frequent itemset $L$, filtering by minimum confidence and pruning based on confidence properties.

7. **Advanced Metrics** include conviction (independence deviation), cosine similarity (association strength), Jaccard coefficient (overlap), and all-confidence (itemset cohesiveness).

8. **Applications** extend beyond retail to web usage mining, bioinformatics (gene co-expression), text mining (term co-occurrence), healthcare (symptom patterns), and cybersecurity (attack patterns).

9. **Challenges** include computational complexity (exponential search space), rule quality (spurious/redundant rules), parameter selection (support/confidence thresholds), interpretability (too many rules), and scalability (memory/I/O).

10. **Best Practices** involve choosing appropriate thresholds based on domain knowledge, using efficient algorithms (FP-Growth), evaluating rules with multiple metrics, removing redundancy, and focusing on actionable, interpretable rules for business value.
