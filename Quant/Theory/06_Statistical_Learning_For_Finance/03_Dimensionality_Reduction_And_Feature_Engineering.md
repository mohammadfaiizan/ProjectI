# Dimensionality Reduction and Feature Engineering

## Principal Component Analysis (PCA)

PCA finds directions of maximum variance in high-dimensional data.

### Mathematical Foundation

For data matrix $\mathbf{X}$ ($n \times p$, $n$ observations, $p$ features):

1. **Center data:** $\tilde{\mathbf{X}} = \mathbf{X} - \bar{\mathbf{X}}$
2. **Covariance matrix:** $\boldsymbol{\Sigma} = \frac{1}{n-1}\tilde{\mathbf{X}}^T\tilde{\mathbf{X}}$
3. **Eigen decomposition:** $\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$
4. **Principal components:** $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{V}$

**Properties:**
- Columns of $\mathbf{V}$ are eigenvectors (principal directions)
- Diagonal of $\boldsymbol{\Lambda}$ contains eigenvalues (variances)
- PCs are orthogonal and uncorrelated

### Variance Explained

**Variance of component $j$:**
$$\text{Var}(Z_j) = \lambda_j$$

**Proportion of variance:**
$$\frac{\lambda_j}{\sum_{k=1}^{p}\lambda_k}$$

**Cumulative proportion:**
$$\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{p}\lambda_j}$$

### Scree Plot

Plot eigenvalues $\lambda_j$ vs component number.

**Elbow method:** Choose number of components at "elbow" where eigenvalues drop sharply.

**Kaiser criterion:** Keep components with $\lambda_j > 1$ (for standardized data).

**Variance threshold:** Keep components explaining >80-90% of variance.

### Reconstruction

**Using $k$ components:**
$$\hat{\mathbf{X}} = \mathbf{Z}_k \mathbf{V}_k^T + \bar{\mathbf{X}}$$

where $\mathbf{Z}_k$ contains first $k$ PCs and $\mathbf{V}_k$ contains first $k$ eigenvectors.

**Reconstruction error:**
$$\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2 = \sum_{j=k+1}^{p}\lambda_j$$

### Standardization

**Important:** PCA is sensitive to scale. Standardize features:
$$X_{ij}^{std} = \frac{X_{ij} - \bar{X}_j}{s_j}$$

where $s_j$ is standard deviation of feature $j$.

**Without standardization:** Features with larger scales dominate.

### Applications: Factor Extraction

**Statistical factors:** Use PCA to extract factors from return data.

**Factor model:**
$$r_{it} = \alpha_i + \sum_{j=1}^{K}\beta_{ij}PC_{jt} + \epsilon_{it}$$

where $PC_{jt}$ are principal components.

**Interpretation:** First few PCs often capture common risk factors.

## Factor Analysis vs PCA

### Factor Analysis Model

$$\mathbf{X} = \boldsymbol{\Lambda}\mathbf{F} + \boldsymbol{\Psi}$$

where:
- $\mathbf{F}$: $k \times 1$ vector of common factors
- $\boldsymbol{\Lambda}$: $p \times k$ matrix of factor loadings
- $\boldsymbol{\Psi}$: $p \times 1$ vector of unique factors (errors)

**Assumptions:**
- $\mathbb{E}[\mathbf{F}] = 0$, $\text{Cov}(\mathbf{F}) = \mathbf{I}$
- $\mathbb{E}[\boldsymbol{\Psi}] = 0$, $\text{Cov}(\boldsymbol{\Psi}) = \boldsymbol{\Phi}$ (diagonal)
- $\mathbf{F}$ and $\boldsymbol{\Psi}$ uncorrelated

### Differences

**PCA:**
- Data reduction technique
- No probabilistic model
- Components are linear combinations of all variables
- No uniqueness assumption

**Factor Analysis:**
- Probabilistic model
- Factors are latent variables
- Separates common and unique variance
- Identifiability issues (rotation needed)

### Rotation

Factor loadings are not unique. Apply rotation to improve interpretability:

**Varimax:** Maximizes variance of squared loadings (simpler structure)

**Promax:** Oblique rotation (allows correlated factors)

**Interpretation:** Rotated factors may be more interpretable.

## t-SNE

t-SNE (t-distributed Stochastic Neighbor Embedding) reduces dimensionality for visualization.

### Algorithm

**Step 1:** Compute pairwise similarities in high dimension:
$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

$$p_{ij} = \frac{p_{i|j} + p_{j|i}}{2n}$$

**Step 2:** Compute similarities in low dimension (using t-distribution):
$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**Step 3:** Minimize KL divergence:
$$KL(P||Q) = \sum_{i,j}p_{ij}\ln\frac{p_{ij}}{q_{ij}}$$

### Properties

- **Non-linear:** Captures non-linear structure
- **Local structure:** Preserves local neighborhoods
- **Global structure:** May distort global distances
- **Stochastic:** Results vary with initialization

### Applications: Visualization of Financial Regimes

**Use:** Visualize high-dimensional return data in 2D/3D to identify:
- Market regimes
- Clusters of similar assets
- Anomalies

**Limitation:** Not for feature extraction (only visualization).

## UMAP

UMAP (Uniform Manifold Approximation and Projection) is similar to t-SNE but faster and preserves more global structure.

### Key Differences from t-SNE

- **Faster:** More efficient algorithm
- **Global structure:** Better preserves global distances
- **Theoretical foundation:** Based on manifold learning
- **Hyperparameters:** More tunable

### Applications

Similar to t-SNE: visualization of high-dimensional financial data.

## Feature Engineering for Financial Data

### Momentum Features

**Simple momentum:**
$$MOM_t^{(k)} = \frac{P_t}{P_{t-k}} - 1$$

**Rate of change:**
$$ROC_t^{(k)} = \frac{P_t - P_{t-k}}{P_{t-k}}$$

**Moving average crossover:**
$$MA_{short} - MA_{long}$$

**Relative strength index (RSI):**
$$RSI = 100 - \frac{100}{1 + RS}$$

where $RS = \frac{\text{Average gain}}{\text{Average loss}}$ over $n$ periods.

### Mean-Reversion Features

**Z-score:**
$$z_t = \frac{P_t - \mu_t}{\sigma_t}$$

where $\mu_t$ and $\sigma_t$ are rolling mean and standard deviation.

**Bollinger Bands:**
- Upper: $\mu_t + 2\sigma_t$
- Lower: $\mu_t - 2\sigma_t$
- Feature: Distance from bands

**Hurst exponent:** Measures long-term memory (mean-reverting if $H < 0.5$).

### Volatility Features

**Realized volatility:**
$$RV_t = \sqrt{\sum_{i=1}^{n}r_{i,t}^2}$$

**Parkinson estimator:** Using high-low range

**GARCH volatility:** Forecast from GARCH model

**Volatility of volatility:** Measure of vol clustering

### Technical Indicators

**Moving averages:** SMA, EMA, WMA

**MACD:** Moving Average Convergence Divergence

**Stochastic oscillator:** Momentum indicator

**ADX:** Average Directional Index (trend strength)

**Volume indicators:** On-balance volume, volume-weighted average price

### Cross-Asset Features

**Correlation:** Rolling correlation with market, sectors

**Beta:** Rolling beta with market

**Spread:** Price difference between related assets

**Ratio:** Price ratio (e.g., gold/silver)

### Macro Features

**Interest rates:** Yield curve, spreads

**Economic indicators:** GDP growth, inflation, unemployment

**Sentiment:** VIX, put/call ratio

**Currency:** Exchange rates, carry

### Time-Based Features

**Day of week:** Monday effect, Friday effect

**Month:** January effect, tax-loss selling

**Time to expiration:** For options

**Holiday effects:** Pre/post holiday returns

## Feature Selection

### Filter Methods

**Correlation:** Remove highly correlated features

**Mutual information:** Measure dependence:
$$I(X;Y) = \sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}$$

**Chi-square test:** For categorical features

**F-test:** For continuous features

### Wrapper Methods

**Forward selection:**
1. Start with no features
2. Add feature that improves model most
3. Repeat until no improvement

**Backward elimination:**
1. Start with all features
2. Remove feature that hurts model least
3. Repeat until no improvement

**Advantages:** Considers feature interactions
**Disadvantages:** Computationally expensive

### Embedded Methods

**LASSO:** Automatically selects features (sets coefficients to zero)

**Tree-based:** Feature importance from random forest, XGBoost

**Ridge:** Shrinks but doesn't eliminate (use for multicollinearity)

### Mutual Information

Measures dependence between features and target:

$$I(X_j; Y) = \sum_{x_j, y}p(x_j, y)\ln\frac{p(x_j, y)}{p(x_j)p(y)}$$

**Use:** Rank features by mutual information with target.

**Advantage:** Captures non-linear relationships.

## Handling Non-Stationarity in Features

### Problem

Financial features are often non-stationary:
- Trends
- Structural breaks
- Regime changes

**Impact:** Model performance degrades over time.

### Solutions

**Differencing:** Use $\Delta X_t = X_t - X_{t-1}$ instead of $X_t$

**Rolling windows:** Re-estimate model on recent data

**Exponential weighting:** Give more weight to recent observations

**Regime-switching:** Allow parameters to change with regime

**Adaptive models:** Online learning, update as new data arrives

### Stationarity Tests

**ADF test:** Test if feature has unit root

**KPSS test:** Test for stationarity

**Structural break tests:** Detect parameter changes

### Feature Stability

**Rolling correlation:** Check if feature-target relationship is stable

**Time-varying coefficients:** Allow feature importance to vary

**Regime-dependent features:** Use different features in different regimes

## Practical Considerations

### Feature Scaling

**Standardization:** $(X - \mu)/\sigma$ (mean 0, std 1)

**Normalization:** $(X - \min)/(\max - \min)$ (range [0,1])

**Robust scaling:** Use median and IQR (robust to outliers)

**When to scale:**
- Distance-based methods (k-means, SVM)
- Regularized regression (Ridge, Lasso)
- Neural networks

**When not needed:**
- Tree-based methods (invariant to monotonic transformations)

### Feature Interactions

**Polynomial features:** $X_1^2$, $X_1 X_2$, etc.

**Domain knowledge:** Create meaningful interactions (e.g., P/E × growth)

**Automatic:** Some methods (trees, neural networks) learn interactions

### Missing Data

**Imputation:**
- Mean/median/mode
- Forward fill (for time series)
- Model-based (regression, k-NN)

**Indicator variables:** Add binary variable for missingness

**Tree methods:** Handle missing data naturally

### Outliers

**Detection:**
- Z-score: $|z| > 3$
- IQR method: Outside $Q1 - 1.5IQR$ or $Q3 + 1.5IQR$
- Isolation Forest

**Treatment:**
- Remove
- Winsorize (cap at percentiles)
- Transform (log, Box-Cox)
- Robust methods (median, quantile regression)

### Feature Importance

**Permutation importance:** Shuffle feature, measure performance drop

**SHAP values:** Explain individual predictions

**Partial dependence:** Visualize feature effects

**Use:** Understand model, select features, ensure interpretability
