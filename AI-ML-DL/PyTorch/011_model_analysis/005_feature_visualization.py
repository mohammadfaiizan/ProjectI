import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

# Sample Models for Feature Visualization
class FeatureExtractorCNN(nn.Module):
    """CNN designed for feature extraction and visualization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction layers with different scales
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.classifier(x4)
        return x
    
    def extract_features(self, x, layer_name='layer4'):
        """Extract features from specified layer"""
        x1 = self.layer1(x)
        if layer_name == 'layer1':
            return x1
        
        x2 = self.layer2(x1)
        if layer_name == 'layer2':
            return x2
        
        x3 = self.layer3(x2)
        if layer_name == 'layer3':
            return x3
        
        x4 = self.layer4(x3)
        if layer_name == 'layer4':
            return x4
        
        return x4  # Default return

class MultiScaleFeatureExtractor(nn.Module):
    """Model for multi-scale feature extraction"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Shared early layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale branches
        self.scale1 = nn.Sequential(
            nn.Conv2d(64, 128, 1),  # 1x1 conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # 3x3 conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),  # 5x5 conv
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion and final layers
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, 1),  # 3 * 128 = 384
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Shared features
        shared = self.shared_conv(x)
        
        # Multi-scale features
        s1 = self.scale1(shared)
        s2 = self.scale2(shared)
        s3 = self.scale3(shared)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([s1, s2, s3], dim=1)
        
        # Fusion and classification
        fused = self.fusion(multi_scale)
        output = self.classifier(fused)
        
        return output
    
    def extract_multiscale_features(self, x):
        """Extract features from all scales"""
        shared = self.shared_conv(x)
        
        s1 = self.scale1(shared)
        s2 = self.scale2(shared)
        s3 = self.scale3(shared)
        
        return {
            'shared': shared,
            'scale1': s1,
            'scale2': s2, 
            'scale3': s3,
            'concatenated': torch.cat([s1, s2, s3], dim=1)
        }

# Feature Visualization Tools
class FeatureMapVisualizer:
    """Comprehensive feature map visualization"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.feature_maps = {}
        self.hooks = []
    
    def register_feature_hooks(self, layer_names: List[str]):
        """Register hooks to capture feature maps"""
        
        def make_feature_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        layer_dict = dict(self.model.named_modules())
        for name in layer_names:
            if name in layer_dict:
                handle = layer_dict[name].register_forward_hook(make_feature_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove feature hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_feature_maps_grid(self, input_tensor: torch.Tensor, layer_name: str,
                                   num_features: int = 16, figsize: Tuple[int, int] = (16, 12)):
        """Visualize feature maps in a grid layout"""
        
        # Clear previous feature maps
        self.feature_maps = {}
        
        # Forward pass to capture features
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if layer_name not in self.feature_maps:
            print(f"No feature maps captured for layer: {layer_name}")
            return
        
        feature_maps = self.feature_maps[layer_name][0].cpu()  # First sample
        num_channels = feature_maps.size(0)
        num_features = min(num_features, num_channels)
        
        # Create grid
        cols = int(np.ceil(np.sqrt(num_features)))
        rows = int(np.ceil(num_features / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_features):
            row, col = i // cols, i % cols
            
            feature_map = feature_maps[i]
            im = axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Channel {i}', fontsize=10)
            axes[row, col].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_features, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'feature_maps_grid_{layer_name.replace(".", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_statistics(self, input_tensor: torch.Tensor, layer_names: List[str]):
        """Visualize statistics of feature maps across layers"""
        
        # Clear and capture features
        self.feature_maps = {}
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Calculate statistics
        layer_stats = {}
        
        for layer_name in layer_names:
            if layer_name in self.feature_maps:
                features = self.feature_maps[layer_name]
                
                # Calculate per-channel statistics
                channel_means = features.mean(dim=[0, 2, 3])  # Average over batch and spatial dims
                channel_stds = features.std(dim=[0, 2, 3])
                channel_maxs = features.amax(dim=[0, 2, 3])
                
                layer_stats[layer_name] = {
                    'means': channel_means.cpu().numpy(),
                    'stds': channel_stds.cpu().numpy(),
                    'maxs': channel_maxs.cpu().numpy(),
                    'sparsity': (features == 0).float().mean().item()
                }
        
        # Plot statistics
        num_layers = len(layer_stats)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mean activations per layer
        layer_names_clean = [name.split('.')[-1] for name in layer_names if name in layer_stats]
        mean_activations = [np.mean(layer_stats[name]['means']) for name in layer_names if name in layer_stats]
        
        axes[0, 0].bar(range(len(mean_activations)), mean_activations)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].set_title('Average Activation per Layer')
        axes[0, 0].set_xticks(range(len(layer_names_clean)))
        axes[0, 0].set_xticklabels(layer_names_clean, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Activation variability
        std_activations = [np.mean(layer_stats[name]['stds']) for name in layer_names if name in layer_stats]
        
        axes[0, 1].bar(range(len(std_activations)), std_activations)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Std Activation')
        axes[0, 1].set_title('Activation Variability per Layer')
        axes[0, 1].set_xticks(range(len(layer_names_clean)))
        axes[0, 1].set_xticklabels(layer_names_clean, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sparsity
        sparsities = [layer_stats[name]['sparsity'] for name in layer_names if name in layer_stats]
        
        axes[1, 0].bar(range(len(sparsities)), sparsities)
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Sparsity')
        axes[1, 0].set_title('Feature Sparsity per Layer')
        axes[1, 0].set_xticks(range(len(layer_names_clean)))
        axes[1, 0].set_xticklabels(layer_names_clean, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution example (first layer)
        if layer_names and layer_names[0] in layer_stats:
            first_layer_means = layer_stats[layer_names[0]]['means']
            axes[1, 1].hist(first_layer_means, bins=20, alpha=0.7, density=True)
            axes[1, 1].set_xlabel('Channel Mean Activation')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title(f'Channel Activation Distribution ({layer_names_clean[0]})')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_statistics.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return layer_stats
    
    def create_feature_correlation_matrix(self, input_tensor: torch.Tensor, layer_name: str):
        """Create correlation matrix between feature channels"""
        
        # Clear and capture features
        self.feature_maps = {}
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        if layer_name not in self.feature_maps:
            print(f"No feature maps captured for layer: {layer_name}")
            return
        
        features = self.feature_maps[layer_name]
        batch_size, num_channels, height, width = features.shape
        
        # Flatten spatial dimensions
        features_flat = features.view(batch_size, num_channels, -1)
        
        # Calculate correlation matrix
        features_mean = features_flat.mean(dim=2, keepdim=True)
        features_centered = features_flat - features_mean
        
        # Compute correlation across spatial locations and batch
        correlation_matrix = torch.zeros(num_channels, num_channels)
        
        for i in range(num_channels):
            for j in range(num_channels):
                if i <= j:
                    # Flatten across batch and spatial dimensions
                    feat_i = features_centered[:, i, :].flatten()
                    feat_j = features_centered[:, j, :].flatten()
                    
                    # Calculate correlation coefficient
                    correlation = torch.corrcoef(torch.stack([feat_i, feat_j]))[0, 1]
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation  # Symmetric
        
        # Replace NaN values with 0
        correlation_matrix[torch.isnan(correlation_matrix)] = 0
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        
        # Use subset if too many channels
        max_channels = 64
        if num_channels > max_channels:
            indices = np.linspace(0, num_channels-1, max_channels, dtype=int)
            correlation_subset = correlation_matrix[indices][:, indices]
            title_suffix = f' (subset: {max_channels}/{num_channels} channels)'
        else:
            correlation_subset = correlation_matrix
            title_suffix = f' (all {num_channels} channels)'
        
        sns.heatmap(correlation_subset.cpu().numpy(), 
                   cmap='RdBu_r', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        
        plt.title(f'Feature Channel Correlations - {layer_name}{title_suffix}')
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        plt.tight_layout()
        plt.savefig(f'feature_correlations_{layer_name.replace(".", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix

class FeatureDimensionalityAnalyzer:
    """Analyze feature dimensionality and manifold structure"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def extract_features_for_analysis(self, data_loader, layer_name: str, 
                                    max_samples: int = 1000):
        """Extract features from specified layer for multiple samples"""
        
        if hasattr(self.model, 'extract_features'):
            extract_fn = lambda x: self.model.extract_features(x, layer_name)
        else:
            # Use hooks for general models
            features_list = []
            
            def feature_hook(module, input, output):
                features_list.append(output.detach())
            
            # Register hook
            layer_dict = dict(self.model.named_modules())
            if layer_name not in layer_dict:
                print(f"Layer {layer_name} not found")
                return None, None
            
            handle = layer_dict[layer_name].register_forward_hook(feature_hook)
            
            all_features = []
            all_labels = []
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(data_loader):
                    if len(all_features) * data.size(0) >= max_samples:
                        break
                    
                    data = data.to(self.device)
                    
                    features_list = []
                    _ = self.model(data)
                    
                    if features_list:
                        features = features_list[0]
                        
                        # Flatten spatial dimensions if conv layer
                        if len(features.shape) == 4:
                            features = features.mean(dim=[2, 3])  # Global average pooling
                        
                        all_features.append(features.cpu())
                        all_labels.append(labels)
            
            handle.remove()
            
            if all_features:
                features_tensor = torch.cat(all_features, dim=0)
                labels_tensor = torch.cat(all_labels, dim=0)
                return features_tensor, labels_tensor
            else:
                return None, None
    
    def perform_pca_analysis(self, features: torch.Tensor, labels: torch.Tensor, 
                           n_components: int = 50):
        """Perform PCA analysis on features"""
        
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_np)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Explained variance ratio
        axes[0, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                       pca.explained_variance_ratio_, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].set_title('PCA Explained Variance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[0, 1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'r-o', markersize=4)
        axes[0, 1].axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95%')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D projection (first two components)
        unique_labels = np.unique(labels_np)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            axes[1, 0].scatter(features_pca[mask, 0], features_pca[mask, 1], 
                             c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        
        axes[1, 0].set_xlabel('First Principal Component')
        axes[1, 0].set_ylabel('Second Principal Component')
        axes[1, 0].set_title('PCA Projection (2D)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3D projection (first three components) - shown as 2D projection of PC2 vs PC3
        axes[1, 1].scatter(features_pca[:, 1], features_pca[:, 2], 
                          c=labels_np, cmap='tab10', alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Second Principal Component')
        axes[1, 1].set_ylabel('Third Principal Component')
        axes[1, 1].set_title('PCA Projection (PC2 vs PC3)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"PCA Analysis Results:")
        print(f"Original feature dimension: {features.shape[1]}")
        print(f"Number of components: {n_components}")
        print(f"Variance explained by first component: {pca.explained_variance_ratio_[0]:.3f}")
        print(f"Variance explained by first 5 components: {cumsum_var[4]:.3f}")
        print(f"Components needed for 95% variance: {np.argmax(cumsum_var >= 0.95) + 1}")
        
        return pca, features_pca
    
    def perform_tsne_analysis(self, features: torch.Tensor, labels: torch.Tensor,
                            perplexity: int = 30, n_iter: int = 1000):
        """Perform t-SNE analysis on features"""
        
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Reduce dimensionality first if features are high-dimensional
        if features_np.shape[1] > 50:
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features_np)
            print(f"Reduced dimensionality from {features_np.shape[1]} to 50 using PCA")
        else:
            features_reduced = features_np
        
        # Perform t-SNE
        print("Performing t-SNE analysis...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        features_tsne = tsne.fit_transform(features_reduced)
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        # Color by class
        plt.subplot(1, 2, 1)
        unique_labels = np.unique(labels_np)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization (colored by class)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Color by density
        plt.subplot(1, 2, 2)
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                   c=labels_np, cmap='tab10', alpha=0.6, s=20)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization (density plot)')
        plt.colorbar(label='Class')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tsne_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return tsne, features_tsne

class MultiScaleFeatureAnalyzer:
    """Analyze features across multiple scales"""
    
    def __init__(self, model: MultiScaleFeatureExtractor, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def visualize_multiscale_features(self, input_tensor: torch.Tensor):
        """Visualize features from different scales"""
        
        self.model.eval()
        with torch.no_grad():
            multiscale_features = self.model.extract_multiscale_features(input_tensor)
        
        # Plot features from different scales
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        scale_names = ['shared', 'scale1', 'scale2', 'scale3']
        
        for i, scale_name in enumerate(scale_names):
            if i < 4:  # First 4 subplots
                row, col = i // 3, i % 3
                
                features = multiscale_features[scale_name][0]  # First sample
                
                # Show average across channels
                if len(features.shape) == 3:  # CHW format
                    avg_feature = features.mean(dim=0)
                    im = axes[row, col].imshow(avg_feature.cpu(), cmap='viridis')
                    axes[row, col].set_title(f'{scale_name.capitalize()} Features\n(avg across {features.shape[0]} channels)')
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Show concatenated features
        concat_features = multiscale_features['concatenated'][0]
        avg_concat = concat_features.mean(dim=0)
        im = axes[1, 1].imshow(avg_concat.cpu(), cmap='viridis')
        axes[1, 1].set_title(f'Concatenated Features\n(avg across {concat_features.shape[0]} channels)')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Show feature statistics comparison
        axes[1, 2].axis('off')
        stats_text = "Multi-scale Statistics:\n\n"
        
        for scale_name, features in multiscale_features.items():
            if scale_name != 'concatenated':
                mean_act = features.mean().item()
                std_act = features.std().item()
                stats_text += f"{scale_name}:\n"
                stats_text += f"  Mean: {mean_act:.3f}\n"
                stats_text += f"  Std: {std_act:.3f}\n\n"
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('multiscale_features.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return multiscale_features
    
    def analyze_scale_importance(self, data_loader, num_batches: int = 10):
        """Analyze the importance of different scales"""
        
        scale_activations = defaultdict(list)
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                if batch_idx >= num_batches:
                    break
                
                data = data.to(self.device)
                multiscale_features = self.model.extract_multiscale_features(data)
                
                # Calculate activation statistics for each scale
                for scale_name, features in multiscale_features.items():
                    if scale_name != 'concatenated':
                        mean_activation = features.mean().item()
                        scale_activations[scale_name].append(mean_activation)
        
        # Plot scale importance
        plt.figure(figsize=(10, 6))
        
        scales = list(scale_activations.keys())
        avg_activations = [np.mean(scale_activations[scale]) for scale in scales]
        std_activations = [np.std(scale_activations[scale]) for scale in scales]
        
        plt.bar(scales, avg_activations, yerr=std_activations, capsize=5, alpha=0.7)
        plt.xlabel('Scale')
        plt.ylabel('Average Activation')
        plt.title('Feature Activation by Scale')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (scale, avg_act) in enumerate(zip(scales, avg_activations)):
            plt.text(i, avg_act + std_activations[i], f'{avg_act:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('scale_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return scale_activations

if __name__ == "__main__":
    print("Feature Visualization")
    print("=" * 22)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    feature_cnn = FeatureExtractorCNN(num_classes=10).to(device)
    multiscale_model = MultiScaleFeatureExtractor(num_classes=10).to(device)
    
    # Create sample data
    sample_input = torch.randn(8, 3, 32, 32).to(device)
    
    print("\n1. Feature Map Grid Visualization")
    print("-" * 40)
    
    # Feature map visualization
    visualizer = FeatureMapVisualizer(feature_cnn, device)
    
    # Register hooks for different layers
    layers_to_visualize = ['layer1', 'layer2', 'layer3', 'layer4']
    visualizer.register_feature_hooks(layers_to_visualize)
    
    # Visualize feature maps for each layer
    for layer_name in layers_to_visualize:
        print(f"Visualizing {layer_name} feature maps...")
        visualizer.visualize_feature_maps_grid(sample_input, layer_name, num_features=16)
    
    visualizer.remove_hooks()
    
    print("\n2. Feature Statistics Analysis")
    print("-" * 35)
    
    # Feature statistics
    visualizer.register_feature_hooks(layers_to_visualize)
    layer_stats = visualizer.visualize_feature_statistics(sample_input, layers_to_visualize)
    
    # Print detailed statistics
    print("\nDetailed Feature Statistics:")
    for layer_name, stats in layer_stats.items():
        print(f"\n{layer_name}:")
        print(f"  Average channel mean: {np.mean(stats['means']):.4f}")
        print(f"  Average channel std: {np.mean(stats['stds']):.4f}")
        print(f"  Sparsity: {stats['sparsity']:.4f}")
        print(f"  Max activation: {np.max(stats['maxs']):.4f}")
    
    visualizer.remove_hooks()
    
    print("\n3. Feature Correlation Analysis")
    print("-" * 35)
    
    # Feature correlation analysis
    visualizer.register_feature_hooks(['layer2'])
    correlation_matrix = visualizer.create_feature_correlation_matrix(sample_input, 'layer2')
    
    if correlation_matrix is not None:
        avg_correlation = correlation_matrix.mean().item()
        max_correlation = correlation_matrix.max().item()
        min_correlation = correlation_matrix.min().item()
        
        print(f"Correlation Analysis for layer2:")
        print(f"  Average correlation: {avg_correlation:.4f}")
        print(f"  Max correlation: {max_correlation:.4f}")
        print(f"  Min correlation: {min_correlation:.4f}")
    
    visualizer.remove_hooks()
    
    print("\n4. Dimensionality Analysis")
    print("-" * 30)
    
    # Create sample dataset for dimensionality analysis
    num_samples = 500
    sample_data = torch.randn(num_samples, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (num_samples,))
    sample_dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=32, shuffle=False)
    
    # Dimensionality analysis
    dim_analyzer = FeatureDimensionalityAnalyzer(feature_cnn, device)
    
    # Extract features from layer3
    print("Extracting features for dimensionality analysis...")
    features, labels = dim_analyzer.extract_features_for_analysis(
        sample_loader, 'layer3', max_samples=400
    )
    
    if features is not None and labels is not None:
        print(f"Extracted features shape: {features.shape}")
        
        # PCA Analysis
        print("\nPerforming PCA analysis...")
        pca, features_pca = dim_analyzer.perform_pca_analysis(features, labels, n_components=30)
        
        # t-SNE Analysis
        print("\nPerforming t-SNE analysis...")
        tsne, features_tsne = dim_analyzer.perform_tsne_analysis(
            features, labels, perplexity=20, n_iter=500
        )
    else:
        print("Failed to extract features for dimensionality analysis")
    
    print("\n5. Multi-Scale Feature Analysis")
    print("-" * 40)
    
    # Multi-scale feature analysis
    multiscale_analyzer = MultiScaleFeatureAnalyzer(multiscale_model, device)
    
    print("Visualizing multi-scale features...")
    multiscale_features = multiscale_analyzer.visualize_multiscale_features(sample_input)
    
    # Create dataset for scale importance analysis
    multiscale_data = torch.randn(160, 3, 32, 32)
    multiscale_labels = torch.randint(0, 10, (160,))
    multiscale_dataset = torch.utils.data.TensorDataset(multiscale_data, multiscale_labels)
    multiscale_loader = torch.utils.data.DataLoader(multiscale_dataset, batch_size=16, shuffle=False)
    
    print("Analyzing scale importance...")
    scale_activations = multiscale_analyzer.analyze_scale_importance(multiscale_loader, num_batches=8)
    
    # Print scale analysis results
    print("\nScale Importance Analysis:")
    for scale_name, activations in scale_activations.items():
        avg_activation = np.mean(activations)
        std_activation = np.std(activations)
        print(f"  {scale_name}: {avg_activation:.4f} ± {std_activation:.4f}")
    
    print("\n6. Feature Evolution Analysis")
    print("-" * 35)
    
    # Analyze how features change with different inputs
    input_types = {
        'Random Noise': torch.randn(1, 3, 32, 32).to(device),
        'Zeros': torch.zeros(1, 3, 32, 32).to(device),
        'Ones': torch.ones(1, 3, 32, 32).to(device),
        'Checkerboard': torch.zeros(1, 3, 32, 32).to(device)
    }
    
    # Create checkerboard pattern
    for i in range(32):
        for j in range(32):
            if (i + j) % 2 == 0:
                input_types['Checkerboard'][0, :, i, j] = 1
    
    print("Analyzing feature responses to different input patterns...")
    
    # Analyze each input type
    visualizer.register_feature_hooks(['layer2'])
    
    pattern_responses = {}
    
    for pattern_name, pattern_input in input_types.items():
        visualizer.feature_maps = {}
        
        feature_cnn.eval()
        with torch.no_grad():
            _ = feature_cnn(pattern_input)
        
        if 'layer2' in visualizer.feature_maps:
            features = visualizer.feature_maps['layer2'][0]  # First (and only) sample
            
            pattern_responses[pattern_name] = {
                'mean': features.mean().item(),
                'std': features.std().item(),
                'max': features.max().item(),
                'sparsity': (features == 0).float().mean().item()
            }
    
    visualizer.remove_hooks()
    
    # Plot pattern responses
    patterns = list(pattern_responses.keys())
    metrics = ['mean', 'std', 'max', 'sparsity']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [pattern_responses[pattern][metric] for pattern in patterns]
        
        axes[i].bar(patterns, values, alpha=0.7)
        axes[i].set_title(f'Feature {metric.capitalize()} by Input Pattern')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pattern_responses.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPattern Response Analysis:")
    for pattern_name, response in pattern_responses.items():
        print(f"  {pattern_name}:")
        for metric, value in response.items():
            print(f"    {metric}: {value:.4f}")
    
    print("\n7. Feature Sensitivity Analysis")
    print("-" * 35)
    
    # Analyze feature sensitivity to input perturbations
    base_input = torch.randn(1, 3, 32, 32).to(device)
    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    print("Analyzing feature sensitivity to input noise...")
    
    visualizer.register_feature_hooks(['layer2'])
    
    sensitivity_results = []
    
    for noise_level in noise_levels:
        # Add noise to input
        noise = torch.randn_like(base_input) * noise_level
        noisy_input = base_input + noise
        
        visualizer.feature_maps = {}
        
        feature_cnn.eval()
        with torch.no_grad():
            _ = feature_cnn(noisy_input)
        
        if 'layer2' in visualizer.feature_maps:
            features = visualizer.feature_maps['layer2'][0]
            
            sensitivity_results.append({
                'noise_level': noise_level,
                'mean_activation': features.mean().item(),
                'std_activation': features.std().item(),
                'max_activation': features.max().item()
            })
    
    visualizer.remove_hooks()
    
    # Plot sensitivity results
    noise_levels_plot = [r['noise_level'] for r in sensitivity_results]
    mean_activations = [r['mean_activation'] for r in sensitivity_results]
    std_activations = [r['std_activation'] for r in sensitivity_results]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(noise_levels_plot, mean_activations, 'b-o', label='Mean Activation')
    plt.plot(noise_levels_plot, std_activations, 'r-o', label='Std Activation')
    plt.xlabel('Noise Level')
    plt.ylabel('Activation Value')
    plt.title('Feature Sensitivity to Input Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Calculate relative change from baseline (noise_level=0)
    baseline_mean = mean_activations[0]
    relative_changes = [(m - baseline_mean) / baseline_mean * 100 for m in mean_activations]
    
    plt.plot(noise_levels_plot, relative_changes, 'g-o')
    plt.xlabel('Noise Level')
    plt.ylabel('Relative Change in Mean Activation (%)')
    plt.title('Feature Sensitivity (Relative Change)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nSensitivity Analysis Results:")
    for result in sensitivity_results:
        print(f"  Noise level {result['noise_level']:.1f}: "
              f"mean={result['mean_activation']:.4f}, "
              f"std={result['std_activation']:.4f}")
    
    print("\nFeature visualization completed!")
    print("Generated files:")
    print("  - feature_maps_grid_*.png (feature map grids)")
    print("  - feature_statistics.png (layer statistics)")
    print("  - feature_correlations_*.png (correlation matrices)")
    print("  - pca_analysis.png (PCA analysis)")
    print("  - tsne_analysis.png (t-SNE analysis)")
    print("  - multiscale_features.png (multi-scale visualization)")
    print("  - scale_importance.png (scale importance analysis)")
    print("  - pattern_responses.png (pattern response analysis)")
    print("  - feature_sensitivity.png (sensitivity analysis)")