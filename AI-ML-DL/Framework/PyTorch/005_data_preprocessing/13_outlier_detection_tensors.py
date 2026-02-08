#!/usr/bin/env python3
"""PyTorch Outlier Detection - Outlier detection methods"""

import torch
import math

print("=== Outlier Detection Overview ===")

print("Outlier detection methods:")
print("1. Statistical methods (Z-score, IQR)")
print("2. Distance-based methods (isolation)")
print("3. Density-based methods")
print("4. Robust statistical methods")
print("5. Machine learning approaches")
print("6. Multivariate outlier detection")

print("\n=== Statistical Outlier Detection ===")

def zscore_outliers(tensor, threshold=3.0):
    """Z-score based outlier detection"""
    mean = tensor.mean()
    std = tensor.std()
    
    if std == 0:
        return torch.zeros_like(tensor, dtype=torch.bool)
    
    z_scores = torch.abs((tensor - mean) / std)
    outliers = z_scores > threshold
    
    return outliers, z_scores

def modified_zscore_outliers(tensor, threshold=3.5):
    """Modified Z-score using median absolute deviation"""
    median = tensor.median()
    mad = torch.median(torch.abs(tensor - median))
    
    if mad == 0:
        return torch.zeros_like(tensor, dtype=torch.bool)
    
    modified_z_scores = 0.6745 * (tensor - median) / mad
    outliers = torch.abs(modified_z_scores) > threshold
    
    return outliers, modified_z_scores

def iqr_outliers(tensor, multiplier=1.5):
    """IQR-based outlier detection"""
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = (tensor < lower_bound) | (tensor > upper_bound)
    
    return outliers, (lower_bound, upper_bound)

# Test statistical methods
data = torch.cat([
    torch.randn(1000),  # Normal data
    torch.tensor([5., -6., 7., -8.])  # Outliers
])

z_outliers, z_scores = zscore_outliers(data, threshold=3.0)
mod_z_outliers, mod_z_scores = modified_zscore_outliers(data, threshold=3.5)
iqr_outliers_mask, iqr_bounds = iqr_outliers(data, multiplier=1.5)

print(f"Data shape: {data.shape}")
print(f"Z-score outliers: {z_outliers.sum()} ({z_outliers.sum()/len(data)*100:.2f}%)")
print(f"Modified Z-score outliers: {mod_z_outliers.sum()} ({mod_z_outliers.sum()/len(data)*100:.2f}%)")
print(f"IQR outliers: {iqr_outliers_mask.sum()} ({iqr_outliers_mask.sum()/len(data)*100:.2f}%)")

print(f"IQR bounds: [{iqr_bounds[0]:.3f}, {iqr_bounds[1]:.3f}]")

print("\n=== Distance-based Outlier Detection ===")

def euclidean_distance_outliers(data, k=5, threshold_percentile=95):
    """Distance to k-th nearest neighbor outlier detection"""
    n_samples = data.shape[0]
    distances = torch.cdist(data, data, p=2)
    
    # Set diagonal to infinity to exclude self-distance
    distances.fill_diagonal_(float('inf'))
    
    # Find k-th nearest neighbor distance for each point
    knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
    outlier_scores = knn_distances[:, -1]  # Distance to k-th neighbor
    
    # Determine threshold
    threshold = torch.quantile(outlier_scores, threshold_percentile / 100.0)
    outliers = outlier_scores > threshold
    
    return outliers, outlier_scores

def mahalanobis_distance_outliers(data, threshold_percentile=95):
    """Mahalanobis distance outlier detection"""
    mean = data.mean(dim=0)
    centered_data = data - mean
    
    # Covariance matrix
    cov_matrix = torch.mm(centered_data.T, centered_data) / (data.shape[0] - 1)
    
    # Regularize covariance matrix
    cov_matrix += torch.eye(cov_matrix.shape[0]) * 1e-6
    
    try:
        cov_inv = torch.inverse(cov_matrix)
    except:
        # Use pseudo-inverse if matrix is singular
        cov_inv = torch.pinverse(cov_matrix)
    
    # Compute Mahalanobis distances
    distances = []
    for i in range(data.shape[0]):
        diff = centered_data[i:i+1]
        dist = torch.sqrt(torch.mm(torch.mm(diff, cov_inv), diff.T))
        distances.append(dist.item())
    
    distances = torch.tensor(distances)
    threshold = torch.quantile(distances, threshold_percentile / 100.0)
    outliers = distances > threshold
    
    return outliers, distances

# Test distance-based methods
multivar_data = torch.randn(200, 3)
# Add some outliers
multivar_data = torch.cat([
    multivar_data,
    torch.tensor([[5., 5., 5.], [-5., -5., -5.], [0., 8., 0.]])
])

knn_outliers, knn_scores = euclidean_distance_outliers(multivar_data, k=5)
maha_outliers, maha_distances = mahalanobis_distance_outliers(multivar_data)

print(f"Multivariate data shape: {multivar_data.shape}")
print(f"KNN outliers: {knn_outliers.sum()} ({knn_outliers.sum()/len(multivar_data)*100:.2f}%)")
print(f"Mahalanobis outliers: {maha_outliers.sum()} ({maha_outliers.sum()/len(multivar_data)*100:.2f}%)")

print("\n=== Isolation Forest (Simplified) ===")

class SimpleIsolationTree:
    """Simplified isolation tree implementation"""
    
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.size = 0
        self.depth = 0
    
    def fit(self, data, depth=0):
        """Build isolation tree"""
        self.size = data.shape[0]
        self.depth = depth
        
        if depth >= self.max_depth or data.shape[0] <= 1:
            return self
        
        # Random feature and split value
        n_features = data.shape[1]
        self.split_feature = torch.randint(0, n_features, (1,)).item()
        
        feature_values = data[:, self.split_feature]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return self
        
        self.split_value = torch.rand(1) * (max_val - min_val) + min_val
        
        # Split data
        left_mask = feature_values < self.split_value
        right_mask = ~left_mask
        
        if left_mask.any():
            self.left = SimpleIsolationTree(self.max_depth)
            self.left.fit(data[left_mask], depth + 1)
        
        if right_mask.any():
            self.right = SimpleIsolationTree(self.max_depth)
            self.right.fit(data[right_mask], depth + 1)
        
        return self
    
    def path_length(self, sample):
        """Compute path length for a sample"""
        if self.split_feature is None or self.size <= 1:
            return self.depth + self._average_path_length(self.size)
        
        if sample[self.split_feature] < self.split_value:
            if self.left is not None:
                return self.left.path_length(sample)
            else:
                return self.depth + self._average_path_length(self.size)
        else:
            if self.right is not None:
                return self.right.path_length(sample)
            else:
                return self.depth + self._average_path_length(self.size)
    
    def _average_path_length(self, n):
        """Average path length of unsuccessful search in BST"""
        if n <= 1:
            return 0
        return 2 * (torch.log(torch.tensor(n - 1.0)) + 0.5772156649) - 2 * (n - 1) / n

def isolation_forest_outliers(data, n_trees=100, threshold_percentile=10):
    """Simplified isolation forest outlier detection"""
    trees = []
    n_samples = data.shape[0]
    subsample_size = min(256, n_samples)
    
    # Build trees
    for _ in range(n_trees):
        # Subsample data
        indices = torch.randperm(n_samples)[:subsample_size]
        subsample = data[indices]
        
        # Build tree
        tree = SimpleIsolationTree(max_depth=int(torch.log2(torch.tensor(subsample_size)).item()))
        tree.fit(subsample)
        trees.append(tree)
    
    # Compute anomaly scores
    scores = []
    for i in range(n_samples):
        sample = data[i]
        path_lengths = [tree.path_length(sample) for tree in trees]
        avg_path_length = sum(path_lengths) / len(path_lengths)
        scores.append(avg_path_length)
    
    scores = torch.tensor(scores)
    
    # Compute anomaly score (shorter paths = more anomalous)
    c = 2 * (torch.log(torch.tensor(subsample_size - 1.0)) + 0.5772156649) - 2 * (subsample_size - 1) / subsample_size
    anomaly_scores = torch.pow(2, -scores / c)
    
    # Determine outliers
    threshold = torch.quantile(anomaly_scores, 1 - threshold_percentile / 100.0)
    outliers = anomaly_scores > threshold
    
    return outliers, anomaly_scores

# Test isolation forest
iso_outliers, iso_scores = isolation_forest_outliers(multivar_data, n_trees=50)
print(f"Isolation forest outliers: {iso_outliers.sum()} ({iso_outliers.sum()/len(multivar_data)*100:.2f}%)")

print("\n=== Local Outlier Factor (Simplified) ===")

def local_outlier_factor(data, k=5):
    """Simplified Local Outlier Factor"""
    n_samples = data.shape[0]
    distances = torch.cdist(data, data, p=2)
    distances.fill_diagonal_(float('inf'))
    
    # Find k-nearest neighbors
    knn_distances, knn_indices = torch.topk(distances, k, largest=False, dim=1)
    
    # Compute reachability distances
    reach_distances = torch.zeros(n_samples, n_samples)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # k-distance of j
                k_dist_j = knn_distances[j, -1]
                reach_distances[i, j] = max(distances[i, j], k_dist_j)
    
    # Compute local reachability density
    lrd = torch.zeros(n_samples)
    for i in range(n_samples):
        neighbors = knn_indices[i]
        avg_reach_dist = reach_distances[i, neighbors].mean()
        lrd[i] = 1.0 / (avg_reach_dist + 1e-10)
    
    # Compute LOF scores
    lof_scores = torch.zeros(n_samples)
    for i in range(n_samples):
        neighbors = knn_indices[i]
        neighbor_lrds = lrd[neighbors]
        lof_scores[i] = neighbor_lrds.mean() / (lrd[i] + 1e-10)
    
    return lof_scores

# Test LOF
lof_scores = local_outlier_factor(multivar_data, k=5)
lof_threshold = torch.quantile(lof_scores, 0.9)
lof_outliers = lof_scores > lof_threshold

print(f"LOF outliers: {lof_outliers.sum()} ({lof_outliers.sum()/len(multivar_data)*100:.2f}%)")
print(f"LOF score range: [{lof_scores.min():.3f}, {lof_scores.max():.3f}]")

print("\n=== Robust Outlier Detection ===")

def robust_covariance_outliers(data, threshold_percentile=95):
    """Outlier detection using robust covariance estimation"""
    # Minimum Covariance Determinant (simplified version)
    n_samples, n_features = data.shape
    h = (n_samples + n_features + 1) // 2  # Number of samples to use
    
    best_det = float('inf')
    best_subset = None
    
    # Try multiple random subsets
    for _ in range(100):
        # Random subset
        indices = torch.randperm(n_samples)[:h]
        subset = data[indices]
        
        # Compute covariance
        mean_subset = subset.mean(dim=0)
        centered_subset = subset - mean_subset
        cov_subset = torch.mm(centered_subset.T, centered_subset) / (h - 1)
        
        # Add regularization
        cov_subset += torch.eye(n_features) * 1e-6
        
        # Compute determinant
        try:
            det = torch.det(cov_subset)
            if det > 0 and det < best_det:
                best_det = det
                best_subset = subset
        except:
            continue
    
    if best_subset is None:
        return torch.zeros(n_samples, dtype=torch.bool), torch.zeros(n_samples)
    
    # Compute robust mean and covariance
    robust_mean = best_subset.mean(dim=0)
    robust_centered = best_subset - robust_mean
    robust_cov = torch.mm(robust_centered.T, robust_centered) / (h - 1)
    robust_cov += torch.eye(n_features) * 1e-6
    
    try:
        robust_cov_inv = torch.inverse(robust_cov)
    except:
        robust_cov_inv = torch.pinverse(robust_cov)
    
    # Compute robust Mahalanobis distances
    centered_data = data - robust_mean
    distances = torch.zeros(n_samples)
    for i in range(n_samples):
        diff = centered_data[i:i+1]
        dist = torch.sqrt(torch.mm(torch.mm(diff, robust_cov_inv), diff.T))
        distances[i] = dist.item()
    
    threshold = torch.quantile(distances, threshold_percentile / 100.0)
    outliers = distances > threshold
    
    return outliers, distances

# Test robust covariance
robust_outliers, robust_distances = robust_covariance_outliers(multivar_data)
print(f"Robust covariance outliers: {robust_outliers.sum()} ({robust_outliers.sum()/len(multivar_data)*100:.2f}%)")

print("\n=== Ensemble Outlier Detection ===")

class OutlierEnsemble:
    """Ensemble of outlier detection methods"""
    
    def __init__(self, methods=None, voting='majority'):
        self.methods = methods or ['zscore', 'iqr', 'isolation', 'lof']
        self.voting = voting
        self.outlier_scores = {}
    
    def detect_outliers(self, data, **kwargs):
        """Detect outliers using ensemble of methods"""
        outlier_masks = {}
        
        # Z-score method
        if 'zscore' in self.methods:
            threshold = kwargs.get('zscore_threshold', 3.0)
            mask, scores = zscore_outliers(data.flatten() if data.dim() > 1 else data, threshold)
            if data.dim() > 1:
                mask = mask.view(data.shape[0], -1).any(dim=1)
            outlier_masks['zscore'] = mask
            self.outlier_scores['zscore'] = scores
        
        # IQR method
        if 'iqr' in self.methods:
            multiplier = kwargs.get('iqr_multiplier', 1.5)
            mask, bounds = iqr_outliers(data.flatten() if data.dim() > 1 else data, multiplier)
            if data.dim() > 1:
                mask = mask.view(data.shape[0], -1).any(dim=1)
            outlier_masks['iqr'] = mask
        
        # Distance-based methods (for multivariate data)
        if data.dim() > 1:
            if 'mahalanobis' in self.methods:
                mask, distances = mahalanobis_distance_outliers(data)
                outlier_masks['mahalanobis'] = mask
                self.outlier_scores['mahalanobis'] = distances
            
            if 'isolation' in self.methods:
                n_trees = kwargs.get('n_trees', 50)
                mask, scores = isolation_forest_outliers(data, n_trees=n_trees)
                outlier_masks['isolation'] = mask
                self.outlier_scores['isolation'] = scores
        
        # Combine results
        if self.voting == 'majority':
            votes = torch.stack(list(outlier_masks.values()), dim=0)
            ensemble_outliers = votes.float().mean(dim=0) > 0.5
        elif self.voting == 'unanimous':
            votes = torch.stack(list(outlier_masks.values()), dim=0)
            ensemble_outliers = votes.all(dim=0)
        elif self.voting == 'any':
            votes = torch.stack(list(outlier_masks.values()), dim=0)
            ensemble_outliers = votes.any(dim=0)
        else:
            raise ValueError(f"Unknown voting method: {self.voting}")
        
        return ensemble_outliers, outlier_masks
    
    def get_method_agreement(self, outlier_masks):
        """Compute agreement between methods"""
        methods = list(outlier_masks.keys())
        n_methods = len(methods)
        agreement_matrix = torch.zeros(n_methods, n_methods)
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                mask1 = outlier_masks[method1]
                mask2 = outlier_masks[method2]
                agreement = (mask1 == mask2).float().mean()
                agreement_matrix[i, j] = agreement
        
        return agreement_matrix, methods

# Test ensemble detection
ensemble = OutlierEnsemble(methods=['zscore', 'iqr', 'mahalanobis'], voting='majority')
ensemble_outliers, method_masks = ensemble.detect_outliers(multivar_data)

print(f"Ensemble outliers (majority vote): {ensemble_outliers.sum()} ({ensemble_outliers.sum()/len(multivar_data)*100:.2f}%)")

# Method agreement
agreement_matrix, method_names = ensemble.get_method_agreement(method_masks)
print("\nMethod agreement matrix:")
for i, method in enumerate(method_names):
    print(f"{method}: {agreement_matrix[i].tolist()}")

print("\n=== Time Series Outlier Detection ===")

def time_series_outliers(ts_data, window_size=10, threshold=3.0):
    """Detect outliers in time series using rolling statistics"""
    n_points = len(ts_data)
    outliers = torch.zeros(n_points, dtype=torch.bool)
    scores = torch.zeros(n_points)
    
    for i in range(window_size, n_points):
        # Rolling window
        window = ts_data[i-window_size:i]
        window_mean = window.mean()
        window_std = window.std()
        
        if window_std > 0:
            z_score = abs((ts_data[i] - window_mean) / window_std)
            scores[i] = z_score
            outliers[i] = z_score > threshold
    
    return outliers, scores

def seasonal_decomposition_outliers(ts_data, period=24, threshold=3.0):
    """Detect outliers after seasonal decomposition (simplified)"""
    n_points = len(ts_data)
    
    # Simple seasonal component (average of same position in period)
    seasonal = torch.zeros_like(ts_data)
    for i in range(period):
        indices = torch.arange(i, n_points, period)
        if len(indices) > 1:
            seasonal_value = ts_data[indices].mean()
            seasonal[indices] = seasonal_value
    
    # Trend component (simple moving average)
    trend = torch.zeros_like(ts_data)
    window = min(period, 5)
    for i in range(window//2, n_points - window//2):
        trend[i] = ts_data[i-window//2:i+window//2+1].mean()
    
    # Residual component
    residual = ts_data - seasonal - trend
    
    # Detect outliers in residual
    outliers, _ = zscore_outliers(residual, threshold)
    
    return outliers, {'seasonal': seasonal, 'trend': trend, 'residual': residual}

# Test time series outlier detection
time_series = torch.sin(torch.linspace(0, 4*math.pi, 100)) + 0.1 * torch.randn(100)
# Add some outliers
time_series[20] += 2.0
time_series[50] -= 1.5
time_series[80] += 1.8

ts_outliers, ts_scores = time_series_outliers(time_series, window_size=10)
seasonal_outliers, components = seasonal_decomposition_outliers(time_series, period=10)

print(f"Time series outliers (rolling): {ts_outliers.sum()}")
print(f"Time series outliers (seasonal): {seasonal_outliers.sum()}")

print("\n=== Outlier Detection Best Practices ===")

print("Outlier Detection Guidelines:")
print("1. Choose method appropriate for your data distribution")
print("2. Consider domain knowledge when setting thresholds")
print("3. Use multiple methods and ensemble approaches")
print("4. Validate outliers don't represent important patterns")
print("5. Consider temporal context for time series data")
print("6. Distinguish between global and local outliers")
print("7. Handle high-dimensional data with dimensionality reduction")

print("\nMethod Selection Guide:")
print("- Z-score/IQR: Simple univariate outliers, normal distributions")
print("- Mahalanobis: Multivariate outliers, elliptical distributions")
print("- Isolation Forest: High-dimensional data, no distribution assumptions")
print("- LOF: Local density-based outliers")
print("- Robust methods: Data with existing outliers")
print("- Ensemble: When unsure about single method")

print("\nValidation Strategies:")
print("- Visual inspection of detected outliers")
print("- Domain expert review")
print("- Sensitivity analysis with different thresholds")
print("- Cross-validation with different methods")
print("- Impact analysis on downstream tasks")

print("\n=== Outlier Detection Complete ===") 