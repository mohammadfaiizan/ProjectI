# Dimensionality Reduction

## Q1: What is the curse of dimensionality?

**A1:** The curse of dimensionality refers to problems arising in high-dimensional spaces: data becomes sparse, distances become similar, and volume concentrates in corners. In high dimensions, nearest neighbors are almost as far as farthest points, making distance-based methods ineffective. Sample size requirements grow exponentially with dimensions. Overfitting becomes easier as model complexity increases. Dimensionality reduction mitigates these issues by finding lower-dimensional representations preserving essential information. Many real-world datasets have intrinsic dimensionality much lower than ambient dimensionality.

## Q2: Explain Principal Component Analysis (PCA) intuition.

**A2:** PCA finds orthogonal directions of maximum variance in data, projecting onto these principal components. The first PC captures most variance, subsequent PCs capture remaining variance while being orthogonal. PCA decorrelates features and reduces dimensionality by keeping top-k components. It's a linear transformation that rotates data to align with variance directions. The transformation is reversible (for k=d) but lossy (for k<d). PCA assumes linear relationships and is sensitive to feature scaling, requiring standardization.

## Q3: How do eigenvalues relate to PCA?

**A3:** Eigenvalues of the covariance matrix represent variances along principal components. Larger eigenvalues correspond to components explaining more variance. The proportion of variance explained by component i is λᵢ / Σλⱼ. Eigenvectors define principal component directions. Choosing components with largest eigenvalues preserves maximum variance. The cumulative variance explained helps determine how many components to keep. Typically, components explaining 80-95% cumulative variance are retained, balancing dimensionality reduction with information preservation.

## Q4: How do you choose the number of components in PCA?

**A4:** Component selection methods include: cumulative variance explained (e.g., 95% threshold), scree plot (elbow method), Kaiser criterion (eigenvalues > 1 for standardized data), and cross-validation on downstream task. Variance-based methods are common but don't guarantee better performance. Cross-validation evaluates actual impact on model performance. The elbow method looks for sharp drops in eigenvalues. Domain knowledge and interpretability also guide selection. There's no universal rule; it depends on the application and acceptable information loss.

## Q5: What is kernel PCA and when is it useful?

**A5:** Kernel PCA applies PCA in a high-dimensional feature space using the kernel trick, enabling non-linear dimensionality reduction. It maps data to feature space via kernel function, then performs PCA there without explicit mapping. Common kernels include polynomial and RBF. Kernel PCA captures non-linear structure that linear PCA misses, useful when data lies on non-linear manifolds. However, it's computationally more expensive (O(n²) or O(n³)) and requires kernel selection. It doesn't provide explicit feature transformation.

## Q6: Explain incremental PCA.

**A6:** Incremental PCA processes data in batches, updating components incrementally without storing all data in memory. It uses mini-batch updates to the covariance matrix, making PCA feasible for large datasets that don't fit in memory. The algorithm maintains an approximation that converges to batch PCA. Incremental PCA trades some accuracy for memory efficiency. It's useful for streaming data and datasets larger than RAM. The batch size affects both memory usage and approximation quality.

## Q7: What is t-SNE and how does it work?

**A7:** t-SNE (t-distributed Stochastic Neighbor Embedding) creates low-dimensional embeddings preserving local neighborhood structure. It models similarities in high and low dimensions using probability distributions, minimizing KL divergence between them. High-dimensional similarities use Gaussian distributions; low-dimensional uses t-distribution (heavier tails). t-SNE emphasizes local structure, clustering similar points together. Key parameters: perplexity (balance local/global structure, typically 5-50) and learning rate. It's excellent for visualization but computationally expensive and non-deterministic.

## Q8: Explain perplexity in t-SNE.

**A8:** Perplexity in t-SNE controls the effective number of neighbors considered for each point, balancing local versus global structure preservation. Low perplexity focuses on local neighborhoods, creating tight clusters but potentially missing global structure. High perplexity considers more neighbors, preserving global structure but potentially losing local detail. Typical values range from 5 to 50, with 30 being common default. Perplexity should be less than number of samples. It's roughly equivalent to k in k-nearest neighbors, but as a smooth, probabilistic measure.

## Q9: What is the crowding problem in t-SNE?

**A9:** The crowding problem occurs when mapping high-dimensional data to low dimensions: there isn't enough space to preserve all pairwise distances. Points get crowded together in the center of the embedding. t-SNE addresses this using t-distribution (instead of Gaussian) for low-dimensional similarities, which has heavier tails and creates more space between clusters. The t-distribution allows points to be moderately far apart without high probability, preventing crowding. This is why t-SNE creates well-separated clusters in visualizations.

## Q10: What is UMAP and how does it compare to t-SNE?

**A10:** UMAP (Uniform Manifold Approximation and Projection) preserves both local and global structure using Riemannian geometry and algebraic topology. It constructs fuzzy simplicial sets representing local structure, then optimizes low-dimensional representation. UMAP is faster than t-SNE (especially for large datasets), preserves global structure better, and has deterministic initialization options. It's more scalable and often produces better global structure preservation. However, t-SNE may create tighter clusters for visualization. UMAP works well for both visualization and dimensionality reduction.

## Q11: Explain Linear Discriminant Analysis (LDA).

**A11:** LDA finds linear combinations of features maximizing separation between classes while minimizing within-class variance. Unlike PCA (unsupervised, maximizes variance), LDA is supervised and maximizes class separability. It projects data onto directions maximizing between-class scatter relative to within-class scatter. LDA assumes Gaussian classes with equal covariance matrices. It reduces to at most (C-1) dimensions for C classes. LDA is useful for classification tasks and can serve as dimensionality reduction preserving class-discriminative information.

## Q12: What is factor analysis?

**A12:** Factor analysis models observed variables as linear combinations of unobserved latent factors plus noise. It assumes factors explain correlations among variables. The model is X = ΛF + ε, where Λ is factor loadings, F are factors, ε is noise. Factor analysis differs from PCA: factors are latent constructs explaining covariance, while PCs explain variance. Factor analysis has probabilistic interpretation and handles missing data. It's used in psychology, social sciences, and when interpretable latent factors are desired. Requires assumptions about factor structure.

## Q13: Explain random projection for dimensionality reduction.

**A13:** Random projection projects data onto lower-dimensional space using random matrices, preserving pairwise distances approximately (Johnson-Lindenstrauss lemma). The projection matrix has random entries (often Gaussian or sparse). Despite randomness, distances are preserved with high probability if dimension is sufficiently large. Random projection is computationally cheap (O(ndk) for n samples, d dimensions, k projections) and data-independent. It's useful as preprocessing step or when exact structure is unknown. Less interpretable than PCA but faster and requires no training.

## Q14: What is the difference between feature selection and feature extraction?

**A14:** Feature selection chooses a subset of original features, maintaining interpretability and original meanings. Feature extraction creates new features from original ones through transformations, potentially losing interpretability. Selection preserves original feature space; extraction creates new space. Selection is simpler but limited to existing features. Extraction can capture interactions and non-linear relationships. Selection is better when features have clear meanings; extraction when relationships are complex. Both reduce dimensionality, but extraction typically achieves greater reduction while preserving more information.

## Q15: How do autoencoders perform dimensionality reduction?

**A15:** Autoencoders are neural networks trained to reconstruct input through a bottleneck layer, learning compressed representations. The encoder maps input to latent code (lower dimension), decoder reconstructs from code. Training minimizes reconstruction error, forcing the bottleneck to capture essential information. The latent representation serves as dimensionality-reduced features. Autoencoders can learn non-linear transformations, unlike PCA. Variational autoencoders add probabilistic interpretation. They're flexible but require more data and computation than linear methods.

## Q16: Explain Variational Autoencoders (VAEs) for representation learning.

**A16:** VAEs learn probabilistic latent representations by modeling data distribution p(x|z) and latent prior p(z). The encoder outputs parameters of latent distribution q(z|x), decoder generates data from samples. Training maximizes evidence lower bound (ELBO), balancing reconstruction and regularization (KL divergence to prior). VAEs enable sampling and interpolation in latent space. The latent space is regularized to be smooth and continuous. VAEs provide uncertainty estimates and can generate new samples, useful beyond dimensionality reduction.

## Q17: What is manifold learning?

**A17:** Manifold learning assumes high-dimensional data lies on lower-dimensional manifolds embedded in high-dimensional space. It finds these intrinsic manifolds and embeds them in lower dimensions. Examples include Isomap, LLE, and t-SNE. Manifold learning preserves local geometry and neighborhood structure. It's useful when data has non-linear structure that linear methods miss. The manifold assumption may not always hold. Manifold learning methods are often computationally expensive and sensitive to parameters. They excel at visualization and discovering intrinsic structure.

## Q18: Explain Isomap and Locally Linear Embedding (LLE).

**A18:** Isomap preserves geodesic distances (shortest paths on manifold) rather than Euclidean distances. It constructs neighborhood graph, computes shortest paths, then uses multidimensional scaling on geodesic distances. Isomap captures non-linear structure but is sensitive to neighborhood size and noise. LLE preserves local linear relationships: each point is reconstructed as weighted combination of neighbors, then finds low-dimensional embedding maintaining these weights. LLE is good for smooth manifolds but struggles with holes and non-uniform sampling. Both are non-linear and preserve local structure.

## Q19: What is Multidimensional Scaling (MDS)?

**A19:** MDS finds low-dimensional embedding preserving pairwise distances (or dissimilarities) from high-dimensional space. Classical MDS minimizes stress function measuring difference between original and embedded distances. It can use Euclidean or other distance metrics. MDS is useful when only distances are available, not original coordinates. It's related to PCA (for Euclidean distances, classical MDS equals PCA). Metric MDS preserves distances; non-metric MDS preserves rank order. MDS is computationally expensive (O(n²)) and sensitive to distance metric choice.

## Q20: Explain Independent Component Analysis (ICA).

**A20:** ICA separates mixed signals into independent components, assuming sources are statistically independent and non-Gaussian. It finds linear transformation making components as independent as possible. Unlike PCA (uncorrelated, Gaussian), ICA finds independent, non-Gaussian components. ICA is used in blind source separation, signal processing, and feature extraction. It requires non-Gaussianity assumption and doesn't order components by importance. ICA can separate mixed audio signals, extract features from images, and analyze brain signals. The independence assumption is stronger than uncorrelatedness.
