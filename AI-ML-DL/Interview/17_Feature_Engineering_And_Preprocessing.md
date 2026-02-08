# Feature Engineering and Preprocessing

## Q1: What is feature scaling and why is it important?

**A1:** Feature scaling standardizes the range of independent features to a common scale, typically [0,1] or mean 0, variance 1. Many machine learning algorithms are sensitive to feature scales, including distance-based methods (k-NN, SVM), gradient descent-based algorithms, and neural networks. Without scaling, features with larger magnitudes dominate the learning process. Scaling ensures all features contribute proportionally to model training. It accelerates convergence in optimization algorithms and prevents numerical instability.

## Q2: Explain normalization, standardization, and robust scaling.

**A2:** Normalization (min-max scaling) transforms features to [0,1] range: (x - min) / (max - min). Standardization (z-score) centers features around mean 0 and variance 1: (x - μ) / σ. Robust scaling uses median and IQR: (x - median) / IQR, making it resistant to outliers. Normalization preserves original distribution shape but is sensitive to outliers. Standardization assumes normal distribution and handles outliers better. Robust scaling is best when outliers are present, using robust statistics.

## Q3: What are common strategies for handling missing data?

**A3:** Missing data strategies include deletion (listwise or pairwise), mean/median/mode imputation for numerical/categorical features, forward/backward fill for time series, interpolation, and advanced methods like k-NN imputation or model-based imputation. The choice depends on missingness mechanism (MCAR, MAR, MNAR), proportion of missing data, and feature importance. Deletion is simple but loses information. Imputation preserves sample size but may introduce bias. Advanced methods model relationships but increase complexity.

## Q4: Explain different categorical encoding methods.

**A4:** One-hot encoding creates binary columns for each category, expanding dimensionality but preserving no-ordinal relationships. Label encoding assigns integers to categories, suitable for tree-based models but implies ordering. Target encoding replaces categories with mean target values, capturing predictive power but risking overfitting. Ordinal encoding assigns ordered integers based on domain knowledge. Frequency encoding uses category frequencies. The choice depends on cardinality, model type, and whether categories have inherent ordering.

## Q5: What are feature selection methods and their categories?

**A5:** Feature selection reduces dimensionality by selecting relevant features, improving model interpretability and reducing overfitting. Filter methods use statistical measures (correlation, mutual information, chi-square) independent of the model. Wrapper methods evaluate feature subsets using model performance (forward/backward selection, recursive feature elimination). Embedded methods perform selection during model training (L1 regularization, tree-based importance). Filter methods are fast but may miss interactions. Wrapper methods are computationally expensive but consider feature interactions.

## Q6: What is feature extraction and how does it differ from selection?

**A6:** Feature extraction creates new features from existing ones through transformations, while feature selection chooses a subset of original features. Extraction methods include PCA, autoencoders, and polynomial features, reducing dimensionality while preserving information. Selection maintains original feature meanings, aiding interpretability. Extraction creates abstract representations that may be harder to interpret. Both reduce dimensionality, but extraction can capture non-linear relationships and interactions that selection might miss.

## Q7: Explain polynomial features and interaction features.

**A7:** Polynomial features generate higher-order terms (x², x³) and interactions (x₁x₂) from original features, enabling models to capture non-linear relationships. They expand feature space significantly: n features become O(n²) or O(nᵈ) for degree d. Interaction features specifically capture relationships between features, useful when features' combined effect differs from their sum. While powerful, they increase dimensionality and risk overfitting. Regularization helps control complexity. Tree-based models naturally capture interactions without explicit creation.

## Q8: What are log and power transforms and when to use them?

**A8:** Log transform (log(x+1)) compresses large values and expands small ones, useful for right-skewed distributions and multiplicative relationships. Power transforms (Box-Cox, Yeo-Johnson) generalize log transforms, finding optimal power parameter λ. Square root transform is milder than log. These transforms stabilize variance, normalize distributions, and linearize relationships. They're particularly useful for count data, monetary values, and when variance increases with mean. Inverse transforms are needed for predictions.

## Q9: How do you handle outliers in feature engineering?

**A9:** Outlier handling strategies include detection (IQR method, z-score, isolation forest), capping/clipping to percentiles, transformation (log, robust scaling), removal if clearly erroneous, and robust methods (median instead of mean). The approach depends on whether outliers are errors or genuine extreme values. Capping preserves information while reducing impact. Transformation makes distributions more normal. Removal risks losing important information. Domain knowledge helps distinguish errors from valid extremes.

## Q10: Explain binning and discretization techniques.

**A10:** Binning converts continuous features into discrete bins, reducing noise and handling non-linear relationships. Methods include equal-width binning (fixed interval size), equal-frequency binning (fixed samples per bin), and domain-based binning. Binning can improve model performance for some algorithms and handle outliers. However, it loses information and may create artificial boundaries. Decision trees naturally perform binning, so explicit binning may be redundant. It's useful for interpretability and when relationships are non-monotonic.

## Q11: What is TF-IDF and how does it work for text features?

**A11:** TF-IDF (Term Frequency-Inverse Document Frequency) weights words by frequency in document (TF) and rarity across corpus (IDF). TF = count(word) / total words in document. IDF = log(total documents / documents containing word). TF-IDF = TF × IDF. This emphasizes words frequent in a document but rare overall, filtering common words like "the". It's superior to bag-of-words for capturing important terms. TF-IDF vectors are sparse and high-dimensional, requiring dimensionality reduction for many models.

## Q12: Explain bag of words representation.

**A12:** Bag of words creates a vocabulary from all unique words in the corpus, then represents each document as a vector counting word occurrences. It ignores word order and grammar, treating documents as unordered word collections. The representation is sparse and high-dimensional, with dimensionality equal to vocabulary size. It's simple but loses semantic and syntactic information. Variations include binary bag-of-words (presence/absence) and n-grams (word sequences). Despite limitations, it's effective for many text classification tasks.

## Q13: How do you engineer date and time features?

**A13:** Date/time features extract temporal patterns: cyclical encoding (sin/cos) for periodic patterns (hour, day of week, month), time since events, time differences, and domain-specific features (business days, holidays). Cyclical encoding preserves periodicity (23:59 close to 00:00). Extracting components (year, month, day) loses cyclical nature. Time deltas capture durations. Features like "is_weekend" or "is_holiday" capture domain patterns. The choice depends on whether temporal patterns are cyclical, trending, or event-based.

## Q14: What techniques handle skewed distributions?

**A14:** Skewed distribution handling includes log transform (for right skew), square root transform (milder), Box-Cox transform (optimal power), quantile transformation (maps to uniform/normal), and robust scaling. Right-skewed data benefits from log transforms, while left-skewed may need square or power transforms. Quantile transformation non-parametrically maps to target distribution. These transforms improve model assumptions (normality) and reduce impact of extreme values. Some models (tree-based) are less sensitive to skewness.

## Q15: Explain target encoding and its risks.

**A15:** Target encoding replaces categorical values with mean target values for that category, creating a single numerical feature. It captures predictive power of categories efficiently, especially for high-cardinality features. However, it risks overfitting and data leakage if not done carefully. Proper implementation uses out-of-fold or cross-validation encoding, computing means only on training folds. Smoothing with global mean reduces overfitting. Target encoding is powerful but requires careful validation to prevent leakage.

## Q16: What is feature hashing and when is it useful?

**A16:** Feature hashing (hashing trick) maps features to fixed-size vectors using hash functions, avoiding the need to maintain a vocabulary. It's memory-efficient for high-cardinality features and streaming data. Hash collisions can occur but rarely hurt performance significantly. The hash space size balances collision rate and memory. Feature hashing is useful for text features with large vocabularies, categorical features with many categories, and online learning scenarios. It's a form of dimensionality reduction.

## Q17: How do you detect and handle multicollinearity?

**A17:** Multicollinearity detection uses correlation matrices, variance inflation factor (VIF), and condition indices. VIF > 10 or correlation > 0.8-0.9 indicates multicollinearity. Solutions include removing highly correlated features, combining correlated features, using dimensionality reduction (PCA), or regularization (L2). Multicollinearity doesn't affect prediction but makes coefficient interpretation unreliable. Tree-based models handle it naturally, while linear models suffer. Feature selection and domain knowledge help identify redundant features.

## Q18: What is data leakage and how do you prevent it?

**A18:** Data leakage occurs when training data contains information unavailable at prediction time, causing unrealistically high performance. Types include target leakage (using future information), train-test contamination (preprocessing before split), and feature leakage (using features derived from targets). Prevention: proper train-test splits before any preprocessing, time-based splits for temporal data, careful feature engineering avoiding target information, and cross-validation with preprocessing inside folds. Leakage detection involves suspiciously high performance and domain knowledge review.

## Q19: Explain techniques for handling imbalanced datasets.

**A19:** Imbalanced dataset techniques include resampling (oversampling minority class, undersampling majority class), SMOTE (synthetic minority oversampling creating synthetic examples), class weights (penalizing misclassifying minority more), threshold tuning (adjusting classification threshold), ensemble methods (balanced bagging), and evaluation metrics (precision, recall, F1, AUC-ROC instead of accuracy). SMOTE generates synthetic examples in feature space between minority examples. Class weights are simple and effective. The choice depends on dataset size, imbalance ratio, and computational resources.

## Q20: What are data pipeline best practices?

**A20:** Data pipeline best practices include: versioning data and preprocessing code, automating pipelines to prevent manual errors, validating data quality (missing values, distributions, schema), separating train/test preprocessing, caching intermediate results, documenting transformations, monitoring for drift, testing on edge cases, using idempotent transformations, and maintaining reproducibility. Pipelines should be modular, testable, and maintainable. Version control ensures reproducibility. Automation reduces human error. Monitoring detects data quality issues early.
