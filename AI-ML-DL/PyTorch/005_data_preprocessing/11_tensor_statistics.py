#!/usr/bin/env python3
"""PyTorch Tensor Statistics - Computing statistics on tensors"""

import torch
import math

print("=== Tensor Statistics Overview ===")

print("Statistical measures:")
print("1. Central tendency (mean, median, mode)")
print("2. Dispersion (variance, std, range, IQR)")
print("3. Shape statistics (skewness, kurtosis)")
print("4. Distribution properties")
print("5. Correlation and covariance")
print("6. Order statistics and quantiles")

print("\n=== Basic Descriptive Statistics ===")

# Sample data
data = torch.randn(1000)
print(f"Sample data shape: {data.shape}")

# Central tendency
mean_val = data.mean()
median_val = data.median()
mode_result = data.mode()
mode_val = mode_result.values

print(f"Mean: {mean_val:.6f}")
print(f"Median: {median_val:.6f}")
print(f"Mode: {mode_val:.6f}")

# Dispersion measures
variance = data.var()
std_dev = data.std()
min_val = data.min()
max_val = data.max()
range_val = max_val - min_val

print(f"Variance: {variance:.6f}")
print(f"Standard deviation: {std_dev:.6f}")
print(f"Range: {range_val:.6f} (min: {min_val:.3f}, max: {max_val:.3f})")

# Quantiles and percentiles
quantiles = torch.quantile(data, torch.tensor([0.25, 0.5, 0.75]))
q1, q2, q3 = quantiles
iqr = q3 - q1

print(f"Q1 (25th percentile): {q1:.6f}")
print(f"Q2 (50th percentile): {q2:.6f}")
print(f"Q3 (75th percentile): {q3:.6f}")
print(f"IQR (Interquartile Range): {iqr:.6f}")

print("\n=== Multi-dimensional Statistics ===")

# Multi-dimensional data
matrix_data = torch.randn(100, 5)
print(f"Matrix data shape: {matrix_data.shape}")

# Statistics along different dimensions
row_means = matrix_data.mean(dim=1)  # Mean of each row
col_means = matrix_data.mean(dim=0)  # Mean of each column
global_mean = matrix_data.mean()     # Global mean

print(f"Row means shape: {row_means.shape}")
print(f"Column means: {col_means}")
print(f"Global mean: {global_mean:.6f}")

# Standard deviations
row_stds = matrix_data.std(dim=1)
col_stds = matrix_data.std(dim=0)
global_std = matrix_data.std()

print(f"Column standard deviations: {col_stds}")
print(f"Global standard deviation: {global_std:.6f}")

# Min/Max along dimensions
col_mins, col_min_indices = matrix_data.min(dim=0)
col_maxs, col_max_indices = matrix_data.max(dim=0)

print(f"Column minimums: {col_mins}")
print(f"Column maximums: {col_maxs}")
print(f"Indices of column minimums: {col_min_indices}")

print("\n=== Advanced Statistical Measures ===")

def compute_moments(tensor, center=True):
    """Compute statistical moments"""
    if center:
        centered = tensor - tensor.mean()
    else:
        centered = tensor
    
    # Raw moments
    moment_1 = centered.mean()
    moment_2 = (centered ** 2).mean()
    moment_3 = (centered ** 3).mean()
    moment_4 = (centered ** 4).mean()
    
    return moment_1, moment_2, moment_3, moment_4

def compute_skewness(tensor):
    """Compute skewness (measure of asymmetry)"""
    mean_val = tensor.mean()
    std_val = tensor.std()
    centered = tensor - mean_val
    skewness = ((centered / std_val) ** 3).mean()
    return skewness

def compute_kurtosis(tensor, fisher=True):
    """Compute kurtosis (measure of tail heaviness)"""
    mean_val = tensor.mean()
    std_val = tensor.std()
    centered = tensor - mean_val
    kurtosis = ((centered / std_val) ** 4).mean()
    
    if fisher:
        kurtosis = kurtosis - 3  # Excess kurtosis (normal distribution has kurtosis=0)
    
    return kurtosis

# Test advanced measures
moments = compute_moments(data)
skewness = compute_skewness(data)
kurtosis_excess = compute_kurtosis(data, fisher=True)
kurtosis_raw = compute_kurtosis(data, fisher=False)

print(f"Statistical moments:")
print(f"  1st moment (mean): {moments[0]:.6f}")
print(f"  2nd moment (variance): {moments[1]:.6f}")
print(f"  3rd moment: {moments[2]:.6f}")
print(f"  4th moment: {moments[3]:.6f}")
print(f"Skewness: {skewness:.6f}")
print(f"Kurtosis (excess): {kurtosis_excess:.6f}")
print(f"Kurtosis (raw): {kurtosis_raw:.6f}")

print("\n=== Correlation and Covariance ===")

def compute_covariance_matrix(matrix):
    """Compute covariance matrix"""
    # Center the data
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    
    # Covariance matrix
    cov_matrix = torch.mm(centered.T, centered) / (matrix.shape[0] - 1)
    
    return cov_matrix

def compute_correlation_matrix(matrix):
    """Compute Pearson correlation matrix"""
    # Center and standardize
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    standardized = centered / matrix.std(dim=0, keepdim=True)
    
    # Correlation matrix
    corr_matrix = torch.mm(standardized.T, standardized) / (matrix.shape[0] - 1)
    
    return corr_matrix

def pearson_correlation(x, y):
    """Compute Pearson correlation between two vectors"""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    
    return numerator / denominator

# Test correlation and covariance
sample_matrix = torch.randn(200, 4)

# Add some correlation
sample_matrix[:, 1] = sample_matrix[:, 0] * 0.7 + torch.randn(200) * 0.3
sample_matrix[:, 2] = sample_matrix[:, 0] * -0.5 + torch.randn(200) * 0.5

cov_matrix = compute_covariance_matrix(sample_matrix)
corr_matrix = compute_correlation_matrix(sample_matrix)

print(f"Covariance matrix:")
print(cov_matrix)
print(f"\nCorrelation matrix:")
print(corr_matrix)

# Test pairwise correlation
corr_01 = pearson_correlation(sample_matrix[:, 0], sample_matrix[:, 1])
corr_02 = pearson_correlation(sample_matrix[:, 0], sample_matrix[:, 2])

print(f"\nPairwise correlations:")
print(f"Feature 0 & 1: {corr_01:.6f}")
print(f"Feature 0 & 2: {corr_02:.6f}")

print("\n=== Distribution Analysis ===")

def histogram_counts(tensor, bins=10, range_vals=None):
    """Compute histogram counts"""
    if range_vals is None:
        range_vals = (tensor.min().item(), tensor.max().item())
    
    min_val, max_val = range_vals
    bin_width = (max_val - min_val) / bins
    
    # Create bin edges
    bin_edges = torch.linspace(min_val, max_val, bins + 1)
    
    # Count values in each bin
    counts = torch.zeros(bins)
    for i in range(bins):
        if i == bins - 1:  # Last bin includes max value
            mask = (tensor >= bin_edges[i]) & (tensor <= bin_edges[i + 1])
        else:
            mask = (tensor >= bin_edges[i]) & (tensor < bin_edges[i + 1])
        counts[i] = mask.sum()
    
    return counts, bin_edges

def empirical_cdf(tensor, x_values=None):
    """Compute empirical cumulative distribution function"""
    if x_values is None:
        x_values = torch.linspace(tensor.min(), tensor.max(), 100)
    
    cdf_values = torch.zeros_like(x_values)
    n = len(tensor)
    
    for i, x in enumerate(x_values):
        cdf_values[i] = (tensor <= x).float().sum() / n
    
    return x_values, cdf_values

# Test distribution analysis
dist_data = torch.randn(10000)

# Histogram
hist_counts, bin_edges = histogram_counts(dist_data, bins=20)
print(f"Histogram (20 bins):")
for i in range(len(hist_counts)):
    print(f"  Bin [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}): {hist_counts[i].int()} samples")

# Empirical CDF
x_vals, cdf_vals = empirical_cdf(dist_data)
print(f"\nEmpirical CDF computed for {len(x_vals)} points")
print(f"CDF at median: {cdf_vals[len(cdf_vals)//2]:.3f}")

print("\n=== Order Statistics ===")

def compute_order_statistics(tensor, k_values=None):
    """Compute order statistics (k-th smallest values)"""
    sorted_tensor, _ = torch.sort(tensor)
    n = len(tensor)
    
    if k_values is None:
        k_values = [1, n//4, n//2, 3*n//4, n]  # Min, Q1, median, Q3, max
    
    order_stats = {}
    for k in k_values:
        if 1 <= k <= n:
            order_stats[f"{k}-th smallest"] = sorted_tensor[k-1]
    
    return order_stats

def compute_extreme_values(tensor, k=5):
    """Get k smallest and k largest values"""
    sorted_vals, sorted_indices = torch.sort(tensor)
    
    smallest_vals = sorted_vals[:k]
    largest_vals = sorted_vals[-k:]
    smallest_indices = sorted_indices[:k]
    largest_indices = sorted_indices[-k:]
    
    return {
        'smallest_values': smallest_vals,
        'largest_values': largest_vals,
        'smallest_indices': smallest_indices,
        'largest_indices': largest_indices
    }

# Test order statistics
order_stats = compute_order_statistics(data)
print("Order statistics:")
for stat_name, value in order_stats.items():
    print(f"  {stat_name}: {value:.6f}")

extremes = compute_extreme_values(data, k=3)
print(f"\n3 smallest values: {extremes['smallest_values']}")
print(f"3 largest values: {extremes['largest_values']}")

print("\n=== Robust Statistics ===")

def mad_statistic(tensor):
    """Median Absolute Deviation"""
    median_val = tensor.median()
    mad = torch.median(torch.abs(tensor - median_val))
    return mad

def trimmed_mean(tensor, trim_fraction=0.1):
    """Trimmed mean (exclude extreme values)"""
    n = len(tensor)
    trim_count = int(n * trim_fraction)
    
    sorted_tensor, _ = torch.sort(tensor)
    trimmed = sorted_tensor[trim_count:n-trim_count]
    
    return trimmed.mean()

def winsorized_mean(tensor, winsor_fraction=0.05):
    """Winsorized mean (replace extreme values)"""
    n = len(tensor)
    winsor_count = int(n * winsor_fraction)
    
    sorted_tensor, _ = torch.sort(tensor)
    
    # Replace extreme values
    winsorized = tensor.clone()
    lower_threshold = sorted_tensor[winsor_count]
    upper_threshold = sorted_tensor[n - winsor_count - 1]
    
    winsorized[winsorized < lower_threshold] = lower_threshold
    winsorized[winsorized > upper_threshold] = upper_threshold
    
    return winsorized.mean()

# Test robust statistics
mad_val = mad_statistic(data)
trimmed_mean_val = trimmed_mean(data, trim_fraction=0.1)
winsorized_mean_val = winsorized_mean(data, winsor_fraction=0.05)

print(f"Median Absolute Deviation: {mad_val:.6f}")
print(f"Trimmed mean (10%): {trimmed_mean_val:.6f}")
print(f"Winsorized mean (5%): {winsorized_mean_val:.6f}")
print(f"Regular mean: {data.mean():.6f}")

print("\n=== Batch Statistics ===")

def batch_statistics(batch_tensor, dim=0):
    """Compute statistics across batch dimension"""
    stats = {}
    
    # Basic statistics
    stats['mean'] = batch_tensor.mean(dim=dim)
    stats['std'] = batch_tensor.std(dim=dim)
    stats['var'] = batch_tensor.var(dim=dim)
    stats['min'] = batch_tensor.min(dim=dim)[0]
    stats['max'] = batch_tensor.max(dim=dim)[0]
    
    # Quantiles
    if dim == 0:
        stats['median'] = batch_tensor.median(dim=dim)[0]
        stats['q25'] = torch.quantile(batch_tensor, 0.25, dim=dim)
        stats['q75'] = torch.quantile(batch_tensor, 0.75, dim=dim)
    
    return stats

def running_statistics():
    """Compute running statistics for streaming data"""
    class RunningStats:
        def __init__(self):
            self.count = 0
            self.mean = 0.0
            self.m2 = 0.0  # Sum of squared differences
        
        def update(self, value):
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
        
        def variance(self):
            return self.m2 / (self.count - 1) if self.count > 1 else 0.0
        
        def std(self):
            return math.sqrt(self.variance())
    
    return RunningStats()

# Test batch statistics
batch_data = torch.randn(64, 128, 32)  # Batch of feature maps
batch_stats = batch_statistics(batch_data, dim=0)

print(f"Batch statistics (computed over batch dimension):")
print(f"Mean shape: {batch_stats['mean'].shape}")
print(f"Std shape: {batch_stats['std'].shape}")
print(f"Global mean: {batch_stats['mean'].mean():.6f}")
print(f"Global std: {batch_stats['std'].mean():.6f}")

# Test running statistics
running_stats = running_statistics()
stream_data = torch.randn(1000)

for value in stream_data[:100]:  # Process first 100 values
    running_stats.update(value.item())

print(f"\nRunning statistics (first 100 values):")
print(f"Count: {running_stats.count}")
print(f"Mean: {running_stats.mean:.6f}")
print(f"Std: {running_stats.std():.6f}")

# Compare with batch computation
batch_mean = stream_data[:100].mean()
batch_std = stream_data[:100].std()
print(f"Batch mean: {batch_mean:.6f}")
print(f"Batch std: {batch_std:.6f}")

print("\n=== Statistical Testing ===")

def two_sample_t_test(sample1, sample2):
    """Simple two-sample t-test (equal variances assumed)"""
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sample1.mean(), sample2.mean()
    var1, var2 = sample1.var(unbiased=True), sample2.var(unbiased=True)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # Standard error
    se = torch.sqrt(pooled_var * (1/n1 + 1/n2))
    
    # t-statistic
    t_stat = (mean1 - mean2) / se
    
    # Degrees of freedom
    df = n1 + n2 - 2
    
    return t_stat, df

def kolmogorov_smirnov_test(sample1, sample2):
    """Simple Kolmogorov-Smirnov test for distribution comparison"""
    # Combine and sort samples
    combined = torch.cat([sample1, sample2])
    combined_sorted, _ = torch.sort(combined)
    
    # Compute empirical CDFs
    n1, n2 = len(sample1), len(sample2)
    
    max_diff = 0.0
    for value in combined_sorted:
        cdf1 = (sample1 <= value).float().sum() / n1
        cdf2 = (sample2 <= value).float().sum() / n2
        diff = torch.abs(cdf1 - cdf2)
        max_diff = max(max_diff, diff.item())
    
    return max_diff

# Test statistical tests
sample_a = torch.randn(100)
sample_b = torch.randn(100) + 0.5  # Shifted distribution

t_stat, df = two_sample_t_test(sample_a, sample_b)
ks_stat = kolmogorov_smirnov_test(sample_a, sample_b)

print(f"Two-sample t-test:")
print(f"  t-statistic: {t_stat:.6f}")
print(f"  degrees of freedom: {df}")

print(f"Kolmogorov-Smirnov test:")
print(f"  KS statistic: {ks_stat:.6f}")

print("\n=== Statistics Best Practices ===")

print("Statistical Analysis Guidelines:")
print("1. Always check data quality before computing statistics")
print("2. Use appropriate statistics for your data distribution")
print("3. Consider robust statistics for data with outliers")
print("4. Compute confidence intervals when reporting means")
print("5. Use appropriate statistical tests for comparisons")
print("6. Be aware of multiple comparison problems")
print("7. Visualize distributions before relying on summary statistics")

print("\nChoosing Statistics:")
print("- Mean: Symmetric distributions without outliers")
print("- Median: Skewed distributions or with outliers")
print("- Mode: Categorical or discrete data")
print("- Standard deviation: Normal distributions")
print("- MAD/IQR: Robust alternatives for spread")
print("- Pearson correlation: Linear relationships")
print("- Spearman correlation: Monotonic relationships")

print("\n=== Tensor Statistics Complete ===") 