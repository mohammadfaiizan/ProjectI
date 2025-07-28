#!/usr/bin/env python3
"""PyTorch Feature Scaling Methods - Various scaling methods"""

import torch
import torch.nn.functional as F
import math

print("=== Feature Scaling Overview ===")

print("Scaling methods:")
print("1. Standard scaling (Z-score normalization)")
print("2. Min-Max scaling")
print("3. Robust scaling")
print("4. Unit vector scaling")
print("5. Quantile uniform scaling")
print("6. Power transformations")
print("7. Feature-wise scaling")

print("\n=== Standard Scaling (Z-score) ===")

class StandardScaler:
    """Standard scaler (zero mean, unit variance)"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler on training data"""
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True, unbiased=False)
        # Avoid division by zero
        self.std = torch.where(self.std < 1e-8, torch.ones_like(self.std), self.std)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        return X_scaled * self.std + self.mean

# Test StandardScaler
data = torch.randn(100, 5) * 10 + 50  # Mean around 50, std around 10

standard_scaler = StandardScaler()
scaled_data = standard_scaler.fit_transform(data)
recovered_data = standard_scaler.inverse_transform(scaled_data)

print(f"Original data - Mean: {data.mean(dim=0)}, Std: {data.std(dim=0, unbiased=False)}")
print(f"Scaled data - Mean: {scaled_data.mean(dim=0)}, Std: {scaled_data.std(dim=0, unbiased=False)}")
print(f"Recovery error: {torch.max(torch.abs(data - recovered_data)):.8f}")

print("\n=== Min-Max Scaling ===")

class MinMaxScaler:
    """Min-Max scaler to [0, 1] or custom range"""
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.scale_range = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler on training data"""
        self.min_vals = X.min(dim=0, keepdim=True)[0]
        self.max_vals = X.max(dim=0, keepdim=True)[0]
        self.scale_range = self.max_vals - self.min_vals
        # Avoid division by zero
        self.scale_range = torch.where(self.scale_range < 1e-8, torch.ones_like(self.scale_range), self.scale_range)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data to specified range"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Scale to [0, 1]
        scaled = (X - self.min_vals) / self.scale_range
        
        # Scale to desired range
        min_target, max_target = self.feature_range
        scaled = scaled * (max_target - min_target) + min_target
        
        return scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        min_target, max_target = self.feature_range
        # Scale back to [0, 1]
        normalized = (X_scaled - min_target) / (max_target - min_target)
        # Scale back to original range
        return normalized * self.scale_range + self.min_vals

# Test MinMaxScaler
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
minmax_scaled = minmax_scaler.fit_transform(data)
minmax_recovered = minmax_scaler.inverse_transform(minmax_scaled)

print(f"Original range: [{data.min():.3f}, {data.max():.3f}]")
print(f"MinMax scaled range: [{minmax_scaled.min():.3f}, {minmax_scaled.max():.3f}]")
print(f"MinMax recovery error: {torch.max(torch.abs(data - minmax_recovered)):.8f}")

# Custom range scaling
custom_scaler = MinMaxScaler(feature_range=(-1, 1))
custom_scaled = custom_scaler.fit_transform(data)
print(f"Custom range [-1, 1]: [{custom_scaled.min():.3f}, {custom_scaled.max():.3f}]")

print("\n=== Robust Scaling ===")

class RobustScaler:
    """Robust scaler using median and IQR"""
    
    def __init__(self):
        self.median = None
        self.iqr = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler on training data"""
        self.median = X.median(dim=0, keepdim=True)[0]
        q25 = torch.quantile(X, 0.25, dim=0, keepdim=True)
        q75 = torch.quantile(X, 0.75, dim=0, keepdim=True)
        self.iqr = q75 - q25
        # Avoid division by zero
        self.iqr = torch.where(self.iqr < 1e-8, torch.ones_like(self.iqr), self.iqr)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using median and IQR"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.median) / self.iqr
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        return X_scaled * self.iqr + self.median

# Test RobustScaler with outliers
data_with_outliers = data.clone()
data_with_outliers[0, 0] = 1000  # Add outlier

robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(data_with_outliers)

# Compare with standard scaler
standard_scaler_outliers = StandardScaler()
standard_scaled_outliers = standard_scaler_outliers.fit_transform(data_with_outliers)

print(f"Data with outliers - Max: {data_with_outliers.max():.1f}")
print(f"Robust scaled - Max: {robust_scaled.max():.3f}")
print(f"Standard scaled - Max: {standard_scaled_outliers.max():.3f}")
print("Robust scaling is less affected by outliers")

print("\n=== Unit Vector Scaling ===")

class UnitVectorScaler:
    """Scale each sample to unit norm"""
    
    def __init__(self, norm='l2'):
        self.norm = norm
    
    def fit(self, X):
        """No fitting required for unit vector scaling"""
        return self
    
    def transform(self, X):
        """Transform each sample to unit norm"""
        if self.norm == 'l2':
            return F.normalize(X, p=2, dim=1)
        elif self.norm == 'l1':
            return F.normalize(X, p=1, dim=1)
        elif self.norm == 'max':
            return X / X.abs().max(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown norm: {self.norm}")
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.transform(X)

# Test UnitVectorScaler
unit_scaler_l2 = UnitVectorScaler(norm='l2')
unit_scaled_l2 = unit_scaler_l2.fit_transform(data)

unit_scaler_l1 = UnitVectorScaler(norm='l1')
unit_scaled_l1 = unit_scaler_l1.fit_transform(data)

print(f"Original sample norms (L2): {torch.norm(data[:5], p=2, dim=1)}")
print(f"L2 unit scaled norms: {torch.norm(unit_scaled_l2[:5], p=2, dim=1)}")
print(f"L1 unit scaled norms: {torch.norm(unit_scaled_l1[:5], p=1, dim=1)}")

print("\n=== Quantile Uniform Scaling ===")

class QuantileUniformScaler:
    """Scale features to uniform distribution using quantiles"""
    
    def __init__(self, n_quantiles=1000, output_distribution='uniform'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles = None
        self.fitted = False
    
    def fit(self, X):
        """Fit scaler by computing quantiles"""
        n_samples, n_features = X.shape
        self.quantiles = []
        
        for feature_idx in range(n_features):
            feature_data = X[:, feature_idx]
            # Compute quantiles
            quantile_levels = torch.linspace(0, 1, self.n_quantiles)
            feature_quantiles = torch.quantile(feature_data, quantile_levels)
            self.quantiles.append(feature_quantiles)
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using quantile mapping"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        X_transformed = torch.zeros_like(X)
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            feature_quantiles = self.quantiles[feature_idx]
            
            # Find quantile ranks
            ranks = torch.searchsorted(feature_quantiles, feature_data, right=True)
            ranks = torch.clamp(ranks, 0, len(feature_quantiles) - 1)
            
            # Convert to uniform [0, 1]
            uniform_values = ranks.float() / (len(feature_quantiles) - 1)
            
            if self.output_distribution == 'uniform':
                X_transformed[:, feature_idx] = uniform_values
            elif self.output_distribution == 'normal':
                # Convert to normal using inverse CDF
                # Clamp to avoid infinite values
                uniform_clamped = torch.clamp(uniform_values, 1e-7, 1-1e-7)
                normal_values = torch.erfinv(2 * uniform_clamped - 1) * math.sqrt(2)
                X_transformed[:, feature_idx] = normal_values
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Test QuantileUniformScaler
# Create data with different distributions
skewed_data = torch.cat([
    torch.exponential(torch.ones(50, 1)),  # Exponential distribution
    torch.randn(50, 1) ** 2,               # Chi-squared-like
    torch.randn(50, 1),                    # Normal
    torch.uniform(-2, 2, (50, 1))          # Uniform
], dim=1)

quantile_scaler = QuantileUniformScaler(n_quantiles=100, output_distribution='uniform')
quantile_scaled = quantile_scaler.fit_transform(skewed_data)

print(f"Original data statistics per feature:")
for i in range(skewed_data.shape[1]):
    mean = skewed_data[:, i].mean()
    std = skewed_data[:, i].std()
    print(f"  Feature {i}: mean={mean:.3f}, std={std:.3f}")

print(f"Quantile scaled statistics (should be ~uniform):")
for i in range(quantile_scaled.shape[1]):
    mean = quantile_scaled[:, i].mean()
    std = quantile_scaled[:, i].std()
    print(f"  Feature {i}: mean={mean:.3f}, std={std:.3f}")

print("\n=== Power Transformations ===")

class PowerTransformer:
    """Power transformations (Box-Cox, Yeo-Johnson)"""
    
    def __init__(self, method='yeo-johnson', standardize=True):
        self.method = method
        self.standardize = standardize
        self.lambdas = None
        self.scaler = None
        self.fitted = False
    
    def _box_cox_transform(self, X, lmbda):
        """Box-Cox transformation (requires positive values)"""
        if lmbda == 0:
            return torch.log(X)
        else:
            return (torch.pow(X, lmbda) - 1) / lmbda
    
    def _yeo_johnson_transform(self, X, lmbda):
        """Yeo-Johnson transformation (handles negative values)"""
        result = torch.zeros_like(X)
        
        # Case 1: x >= 0 and lambda != 0
        mask1 = (X >= 0) & (lmbda != 0)
        result[mask1] = (torch.pow(X[mask1] + 1, lmbda) - 1) / lmbda
        
        # Case 2: x >= 0 and lambda == 0
        mask2 = (X >= 0) & (lmbda == 0)
        result[mask2] = torch.log(X[mask2] + 1)
        
        # Case 3: x < 0 and lambda != 2
        mask3 = (X < 0) & (lmbda != 2)
        result[mask3] = -(torch.pow(-X[mask3] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        
        # Case 4: x < 0 and lambda == 2
        mask4 = (X < 0) & (lmbda == 2)
        result[mask4] = -torch.log(-X[mask4] + 1)
        
        return result
    
    def _find_optimal_lambda(self, x):
        """Find optimal lambda parameter (simplified)"""
        # In practice, you'd use maximum likelihood estimation
        # Here we use a simple grid search
        lambdas = torch.linspace(-2, 2, 21)
        best_lambda = 1.0
        best_score = float('inf')
        
        for lmbda in lambdas:
            try:
                if self.method == 'box-cox':
                    if (x <= 0).any():
                        continue  # Box-Cox requires positive values
                    transformed = self._box_cox_transform(x, lmbda)
                else:
                    transformed = self._yeo_johnson_transform(x, lmbda)
                
                # Use variance as a simple normality measure
                score = transformed.var()
                if score < best_score:
                    best_score = score
                    best_lambda = lmbda.item()
            except:
                continue
        
        return best_lambda
    
    def fit(self, X):
        """Fit power transformer"""
        self.lambdas = []
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            optimal_lambda = self._find_optimal_lambda(feature_data)
            self.lambdas.append(optimal_lambda)
        
        if self.standardize:
            # Fit a standard scaler on transformed data
            X_transformed = self.transform(X)
            self.scaler = StandardScaler().fit(X_transformed)
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        X_transformed = torch.zeros_like(X)
        
        for feature_idx in range(X.shape[1]):
            feature_data = X[:, feature_idx]
            lmbda = self.lambdas[feature_idx]
            
            if self.method == 'box-cox':
                X_transformed[:, feature_idx] = self._box_cox_transform(feature_data, lmbda)
            else:
                X_transformed[:, feature_idx] = self._yeo_johnson_transform(feature_data, lmbda)
        
        if self.standardize and self.scaler:
            X_transformed = self.scaler.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Test PowerTransformer
# Create data with different distributions that benefit from power transformation
exponential_data = torch.exponential(torch.ones(100, 1))
squared_normal = torch.randn(100, 1) ** 2
mixed_data = torch.cat([exponential_data, squared_normal], dim=1)

power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
power_transformed = power_transformer.fit_transform(mixed_data)

print(f"Original data skewness (approximate):")
for i in range(mixed_data.shape[1]):
    mean = mixed_data[:, i].mean()
    std = mixed_data[:, i].std()
    skewness = ((mixed_data[:, i] - mean) ** 3).mean() / (std ** 3)
    print(f"  Feature {i}: skewness={skewness:.3f}")

print(f"Power transformed skewness:")
for i in range(power_transformed.shape[1]):
    mean = power_transformed[:, i].mean()
    std = power_transformed[:, i].std()
    skewness = ((power_transformed[:, i] - mean) ** 3).mean() / (std ** 3)
    print(f"  Feature {i}: skewness={skewness:.3f}")

print(f"Optimal lambdas: {power_transformer.lambdas}")

print("\n=== Feature-wise Scaling Pipeline ===")

class FeatureScalingPipeline:
    """Pipeline for applying different scalers to different features"""
    
    def __init__(self, scalers_config):
        """
        scalers_config: dict mapping feature indices to scaler instances
        Example: {(0, 2): StandardScaler(), (1, 3): MinMaxScaler()}
        """
        self.scalers_config = scalers_config
        self.fitted = False
    
    def fit(self, X):
        """Fit all scalers on their respective features"""
        for feature_indices, scaler in self.scalers_config.items():
            if isinstance(feature_indices, int):
                feature_indices = [feature_indices]
            feature_data = X[:, feature_indices]
            scaler.fit(feature_data)
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform features using their respective scalers"""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X_transformed = X.clone()
        
        for feature_indices, scaler in self.scalers_config.items():
            if isinstance(feature_indices, int):
                feature_indices = [feature_indices]
            
            feature_data = X[:, feature_indices]
            transformed_features = scaler.transform(feature_data)
            X_transformed[:, feature_indices] = transformed_features
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# Test feature-wise scaling
mixed_feature_data = torch.cat([
    torch.randn(100, 2) * 100 + 500,    # Features 0,1: large scale, normal
    torch.exponential(torch.ones(100, 1)),  # Feature 2: exponential
    torch.rand(100, 1) * 1000            # Feature 3: uniform, large scale
], dim=1)

# Configure different scalers for different features
pipeline_config = {
    (0, 1): StandardScaler(),     # Standard scaling for normal features
    2: PowerTransformer(),        # Power transform for exponential
    3: MinMaxScaler()            # Min-max for uniform
}

scaling_pipeline = FeatureScalingPipeline(pipeline_config)
pipeline_scaled = scaling_pipeline.fit_transform(mixed_feature_data)

print(f"Original feature statistics:")
for i in range(mixed_feature_data.shape[1]):
    mean = mixed_feature_data[:, i].mean()
    std = mixed_feature_data[:, i].std()
    min_val = mixed_feature_data[:, i].min()
    max_val = mixed_feature_data[:, i].max()
    print(f"  Feature {i}: mean={mean:.1f}, std={std:.1f}, range=[{min_val:.1f}, {max_val:.1f}]")

print(f"Pipeline scaled statistics:")
for i in range(pipeline_scaled.shape[1]):
    mean = pipeline_scaled[:, i].mean()
    std = pipeline_scaled[:, i].std()
    min_val = pipeline_scaled[:, i].min()
    max_val = pipeline_scaled[:, i].max()
    print(f"  Feature {i}: mean={mean:.3f}, std={std:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")

print("\n=== Scaling Best Practices ===")

print("Feature Scaling Guidelines:")
print("1. Always fit scaler on training data only")
print("2. Apply same transformation to validation and test data")
print("3. Choose scaling method based on data distribution")
print("4. Consider the algorithm's sensitivity to feature scales")
print("5. Save scaler parameters for inference")
print("6. Monitor for feature drift in production")
print("7. Consider feature interactions when scaling")

print("\nScaling Method Selection:")
print("- StandardScaler: Normal distributions, most algorithms")
print("- MinMaxScaler: Bounded features, neural networks")
print("- RobustScaler: Data with outliers")
print("- UnitVectorScaler: Similarity/distance-based algorithms")
print("- QuantileUniformScaler: Highly skewed distributions")
print("- PowerTransformer: Make distributions more normal")

print("\nCommon Pitfalls:")
print("- Fitting scaler on entire dataset (data leakage)")
print("- Different scaling for train/test data")
print("- Not handling categorical features appropriately")
print("- Scaling target variables unnecessarily")
print("- Not considering feature interpretability after scaling")

print("\n=== Feature Scaling Complete ===") 