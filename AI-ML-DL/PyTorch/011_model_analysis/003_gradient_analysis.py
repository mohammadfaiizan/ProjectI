import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, OrderedDict
import warnings

# Sample Models for Gradient Analysis
class TestModel(nn.Module):
    """Simple model for gradient analysis"""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class DeepCNN(nn.Module):
    """Deep CNN for gradient flow analysis"""
    
    def __init__(self, num_classes=10, num_layers=12):
        super().__init__()
        
        layers = []
        in_channels = 3
        base_channels = 64
        
        for i in range(num_layers):
            out_channels = min(base_channels * (2 ** (i // 3)), 512)
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            
            if i % 3 == 2:  # Downsample every 3 layers
                layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for gradient flow testing"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TestResNet(nn.Module):
    """ResNet for gradient analysis"""
    
    def __init__(self, num_classes=10, num_blocks_per_layer=2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, num_blocks_per_layer, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks_per_layer, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks_per_layer, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks_per_layer, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Gradient Analysis Tools
class GradientAnalyzer:
    """Comprehensive gradient analysis utilities"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = {}
        self.gradient_norms = {}
        self.hooks = []
    
    def register_gradient_hooks(self):
        """Register hooks to capture gradients"""
        
        def make_gradient_hook(name):
            def hook(grad):
                if grad is not None:
                    self.gradients[name] = grad.clone()
                    self.gradient_norms[name] = grad.norm().item()
                return grad
            return hook
        
        # Register hooks for all parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(make_gradient_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all gradient hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for captured gradients"""
        stats = {}
        
        for name, grad in self.gradients.items():
            if grad is not None:
                grad_flat = grad.view(-1)
                stats[name] = {
                    'mean': grad_flat.mean().item(),
                    'std': grad_flat.std().item(),
                    'norm': grad.norm().item(),
                    'max': grad_flat.max().item(),
                    'min': grad_flat.min().item(),
                    'num_zeros': (grad_flat == 0).sum().item(),
                    'num_elements': grad_flat.numel(),
                    'sparsity': (grad_flat == 0).float().mean().item()
                }
        
        return stats
    
    def detect_vanishing_gradients(self, threshold: float = 1e-7) -> Dict[str, bool]:
        """Detect vanishing gradients"""
        vanishing = {}
        
        for name, norm in self.gradient_norms.items():
            vanishing[name] = norm < threshold
        
        return vanishing
    
    def detect_exploding_gradients(self, threshold: float = 10.0) -> Dict[str, bool]:
        """Detect exploding gradients"""
        exploding = {}
        
        for name, norm in self.gradient_norms.items():
            exploding[name] = norm > threshold
        
        return exploding
    
    def print_gradient_summary(self):
        """Print gradient analysis summary"""
        stats = self.compute_gradient_stats()
        vanishing = self.detect_vanishing_gradients()
        exploding = self.detect_exploding_gradients()
        
        print("\nGradient Analysis Summary")
        print("=" * 50)
        print(f"{'Layer':<25} {'Norm':<12} {'Mean':<12} {'Std':<12} {'Status':<15}")
        print("-" * 80)
        
        for name in stats:
            stat = stats[name]
            status = ""
            if vanishing.get(name, False):
                status = "VANISHING"
            elif exploding.get(name, False):
                status = "EXPLODING"
            else:
                status = "NORMAL"
            
            print(f"{name[:23]:<25} {stat['norm']:<12.6f} {stat['mean']:<12.6f} "
                  f"{stat['std']:<12.6f} {status:<15}")

class GradientFlowAnalyzer:
    """Analyze gradient flow through the network"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_gradients = []
        self.layer_names = []
        self.hooks = []
    
    def register_flow_hooks(self):
        """Register hooks to track gradient flow"""
        
        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    avg_grad = grad_output[0].abs().mean().item()
                    self.layer_gradients.append(avg_grad)
                    self.layer_names.append(name)
            return hook
        
        # Register hooks for key layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                handle = module.register_backward_hook(make_backward_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove flow hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_gradient_flow(self, data_loader, criterion, num_batches: int = 5):
        """Analyze gradient flow over multiple batches"""
        self.register_flow_hooks()
        
        all_gradients = []
        
        self.model.train()
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            self.layer_gradients = []
            self.layer_names = []
            
            # Forward and backward pass
            self.model.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Store gradients for this batch
            if len(self.layer_gradients) > 0:
                all_gradients.append(self.layer_gradients.copy())
        
        self.remove_hooks()
        
        return all_gradients
    
    def plot_gradient_flow(self, gradient_data: List[List[float]], title="Gradient Flow"):
        """Plot gradient flow visualization"""
        if not gradient_data:
            print("No gradient data to plot")
            return
        
        # Convert to numpy array
        gradient_array = np.array(gradient_data)
        
        # Average across batches
        avg_gradients = gradient_array.mean(axis=0)
        std_gradients = gradient_array.std(axis=0)
        
        # Get unique layer names
        unique_names = []
        for name in self.layer_names:
            if name not in unique_names:
                unique_names.append(name)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        x_pos = range(len(avg_gradients))
        plt.bar(x_pos, avg_gradients, yerr=std_gradients, capsize=5, alpha=0.7)
        
        plt.xlabel('Layer (from output to input)')
        plt.ylabel('Average Gradient Magnitude')
        plt.title(title)
        plt.yscale('log')
        
        # Set x-axis labels
        if len(unique_names) == len(avg_gradients):
            plt.xticks(x_pos, [name.split('.')[-1][:10] for name in unique_names], rotation=45)
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
        plt.show()

class GradientHistogramAnalyzer:
    """Analyze gradient distributions using histograms"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = defaultdict(list)
    
    def collect_gradients(self, data_loader, criterion, num_epochs: int = 3):
        """Collect gradients over multiple epochs"""
        
        self.model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit batches per epoch
                    break
                
                self.model.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Collect gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.gradient_history[name].append(param.grad.clone().detach().cpu())
    
    def plot_gradient_histograms(self, layer_names: Optional[List[str]] = None, 
                                num_bins: int = 50):
        """Plot gradient histograms for specified layers"""
        
        if layer_names is None:
            layer_names = list(self.gradient_history.keys())[:6]  # First 6 layers
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, name in enumerate(layer_names[:6]):
            if name in self.gradient_history and i < 6:
                # Combine all gradients for this layer
                all_grads = torch.cat([grad.view(-1) for grad in self.gradient_history[name]])
                
                # Plot histogram
                axes[i].hist(all_grads.numpy(), bins=num_bins, alpha=0.7, density=True)
                axes[i].set_title(f'{name.split(".")[-1][:15]}', fontsize=10)
                axes[i].set_xlabel('Gradient Value')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_grad = all_grads.mean().item()
                std_grad = all_grads.std().item()
                axes[i].axvline(mean_grad, color='red', linestyle='--', alpha=0.8, 
                              label=f'μ={mean_grad:.2e}')
                axes[i].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('gradient_histograms.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_gradient_evolution(self, layer_name: str):
        """Plot how gradients evolve over training"""
        
        if layer_name not in self.gradient_history:
            print(f"No gradient data for layer: {layer_name}")
            return
        
        # Calculate statistics over time
        means = []
        stds = []
        norms = []
        
        for grad in self.gradient_history[layer_name]:
            grad_flat = grad.view(-1)
            means.append(grad_flat.mean().item())
            stds.append(grad_flat.std().item())
            norms.append(grad.norm().item())
        
        # Plot evolution
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        iterations = range(len(means))
        
        ax1.plot(iterations, means)
        ax1.set_ylabel('Mean Gradient')
        ax1.set_title(f'Gradient Evolution: {layer_name}')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations, stds)
        ax2.set_ylabel('Gradient Std')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(iterations, norms)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Norm')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'gradient_evolution_{layer_name.replace(".", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

class GradientClippingAnalyzer:
    """Analyze effects of gradient clipping"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_norms_before = []
        self.gradient_norms_after = []
        self.clipping_ratios = []
    
    def analyze_clipping_effects(self, data_loader, criterion, 
                               clip_value: float = 1.0, num_batches: int = 20):
        """Analyze gradient clipping effects"""
        
        self.model.train()
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            self.model.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Calculate gradient norm before clipping
            total_norm_before = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm_before += param_norm.item() ** 2
            total_norm_before = total_norm_before ** (1. / 2)
            
            # Apply clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            
            # Calculate gradient norm after clipping
            total_norm_after = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm_after += param_norm.item() ** 2
            total_norm_after = total_norm_after ** (1. / 2)
            
            # Store results
            self.gradient_norms_before.append(total_norm_before)
            self.gradient_norms_after.append(total_norm_after)
            
            clipping_ratio = total_norm_after / max(total_norm_before, 1e-6)
            self.clipping_ratios.append(clipping_ratio)
    
    def plot_clipping_analysis(self):
        """Plot gradient clipping analysis"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        iterations = range(len(self.gradient_norms_before))
        
        # Gradient norms before and after clipping
        ax1.plot(iterations, self.gradient_norms_before, label='Before Clipping', alpha=0.7)
        ax1.plot(iterations, self.gradient_norms_after, label='After Clipping', alpha=0.7)
        ax1.set_ylabel('Gradient Norm')
        ax1.set_title('Gradient Norms: Before vs After Clipping')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Clipping ratios
        ax2.plot(iterations, self.clipping_ratios, 'r-', alpha=0.7)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Clipping Ratio')
        ax2.set_title('Gradient Clipping Ratios (1.0 = no clipping)')
        ax2.grid(True, alpha=0.3)
        
        # Distribution of clipping ratios
        ax3.hist(self.clipping_ratios, bins=20, alpha=0.7, density=True)
        ax3.axvline(x=1.0, color='r', linestyle='--', alpha=0.8, label='No clipping')
        ax3.set_xlabel('Clipping Ratio')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Clipping Ratios')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_clipping_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        clipped_batches = sum(1 for ratio in self.clipping_ratios if ratio < 0.99)
        print(f"\nClipping Statistics:")
        print(f"Batches clipped: {clipped_batches}/{len(self.clipping_ratios)} "
              f"({clipped_batches/len(self.clipping_ratios)*100:.1f}%)")
        print(f"Average clipping ratio: {np.mean(self.clipping_ratios):.3f}")
        print(f"Min clipping ratio: {np.min(self.clipping_ratios):.3f}")

class LayerWiseGradientAnalyzer:
    """Analyze gradients layer by layer"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_stats = defaultdict(list)
    
    def analyze_layers(self, data_loader, criterion, num_batches: int = 10):
        """Analyze gradients for each layer"""
        
        self.model.train()
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            self.model.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Analyze each layer
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    
                    stats = {
                        'norm': grad.norm().item(),
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'sparsity': (grad == 0).float().mean().item()
                    }
                    
                    self.layer_stats[name].append(stats)
    
    def create_gradient_heatmap(self):
        """Create heatmap of gradient statistics across layers"""
        
        # Prepare data for heatmap
        layer_names = list(self.layer_stats.keys())
        metrics = ['norm', 'mean', 'std', 'sparsity']
        
        # Calculate average statistics
        heatmap_data = []
        for metric in metrics:
            metric_values = []
            for layer_name in layer_names:
                values = [stats[metric] for stats in self.layer_stats[layer_name]]
                avg_value = np.mean(values)
                metric_values.append(avg_value)
            heatmap_data.append(metric_values)
        
        heatmap_data = np.array(heatmap_data)
        
        # Normalize each metric to [0, 1] for better visualization
        for i in range(len(metrics)):
            metric_data = heatmap_data[i]
            if metric_data.max() > metric_data.min():
                heatmap_data[i] = (metric_data - metric_data.min()) / (metric_data.max() - metric_data.min())
        
        # Create heatmap
        plt.figure(figsize=(max(12, len(layer_names) * 0.5), 6))
        
        sns.heatmap(heatmap_data, 
                   xticklabels=[name.split('.')[-1][:15] for name in layer_names],
                   yticklabels=metrics,
                   annot=False,
                   cmap='viridis',
                   cbar_kws={'label': 'Normalized Value'})
        
        plt.title('Layer-wise Gradient Statistics Heatmap')
        plt.xlabel('Layer')
        plt.ylabel('Metric')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('gradient_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_layer_comparison(self):
        """Print comparison of gradient statistics across layers"""
        
        print("\nLayer-wise Gradient Comparison")
        print("=" * 80)
        print(f"{'Layer':<25} {'Avg Norm':<12} {'Avg Mean':<12} {'Avg Std':<12} {'Sparsity':<10}")
        print("-" * 80)
        
        for name, stats_list in self.layer_stats.items():
            avg_norm = np.mean([s['norm'] for s in stats_list])
            avg_mean = np.mean([s['mean'] for s in stats_list])
            avg_std = np.mean([s['std'] for s in stats_list])
            avg_sparsity = np.mean([s['sparsity'] for s in stats_list])
            
            print(f"{name[:23]:<25} {avg_norm:<12.2e} {avg_mean:<12.2e} "
                  f"{avg_std:<12.2e} {avg_sparsity:<10.3f}")

if __name__ == "__main__":
    print("Gradient Analysis")
    print("=" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test models
    simple_model = TestModel().to(device)
    deep_cnn = DeepCNN(num_layers=8).to(device)
    resnet_model = TestResNet(num_blocks_per_layer=2).to(device)
    
    # Create sample data loaders
    def create_sample_loader(input_shape, batch_size=32, num_samples=320):
        if len(input_shape) == 1:  # For simple model
            data = torch.randn(num_samples, *input_shape)
        else:  # For CNN models
            data = torch.randn(num_samples, *input_shape)
        
        targets = torch.randint(0, 10, (num_samples,))
        dataset = torch.utils.data.TensorDataset(data, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Data loaders
    simple_loader = create_sample_loader((784,), batch_size=32)
    cnn_loader = create_sample_loader((3, 32, 32), batch_size=16)
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n1. Basic Gradient Analysis")
    print("-" * 30)
    
    # Basic gradient analysis
    analyzer = GradientAnalyzer(simple_model)
    analyzer.register_gradient_hooks()
    
    # Perform one training step
    data, targets = next(iter(simple_loader))
    data, targets = data.to(device), targets.to(device)
    
    simple_model.zero_grad()
    outputs = simple_model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Analyze gradients
    analyzer.print_gradient_summary()
    
    # Check for gradient problems
    vanishing = analyzer.detect_vanishing_gradients(threshold=1e-6)
    exploding = analyzer.detect_exploding_gradients(threshold=5.0)
    
    print(f"\nVanishing gradients detected: {sum(vanishing.values())} layers")
    print(f"Exploding gradients detected: {sum(exploding.values())} layers")
    
    analyzer.remove_hooks()
    
    print("\n2. Gradient Flow Analysis")
    print("-" * 30)
    
    # Gradient flow analysis
    flow_analyzer = GradientFlowAnalyzer(deep_cnn)
    gradient_flow_data = flow_analyzer.analyze_gradient_flow(cnn_loader, criterion, num_batches=5)
    
    if gradient_flow_data:
        flow_analyzer.plot_gradient_flow(gradient_flow_data, "Deep CNN Gradient Flow")
        
        # Calculate flow statistics
        avg_flow = np.array(gradient_flow_data).mean(axis=0)
        flow_ratio = avg_flow[-1] / avg_flow[0] if avg_flow[0] > 0 else 0
        print(f"Gradient flow ratio (output/input): {flow_ratio:.6f}")
        
        if flow_ratio < 0.1:
            print("WARNING: Significant gradient decay detected!")
        elif flow_ratio > 10:
            print("WARNING: Gradient amplification detected!")
    
    print("\n3. Gradient Histogram Analysis")
    print("-" * 35)
    
    # Gradient histogram analysis
    hist_analyzer = GradientHistogramAnalyzer(simple_model)
    print("Collecting gradients for histogram analysis...")
    hist_analyzer.collect_gradients(simple_loader, criterion, num_epochs=2)
    
    # Plot histograms for first few layers
    layer_names = list(hist_analyzer.gradient_history.keys())[:6]
    hist_analyzer.plot_gradient_histograms(layer_names)
    
    # Plot evolution for one layer
    if layer_names:
        print(f"Plotting gradient evolution for: {layer_names[0]}")
        hist_analyzer.plot_gradient_evolution(layer_names[0])
    
    print("\n4. Gradient Clipping Analysis")
    print("-" * 35)
    
    # Gradient clipping analysis
    clip_analyzer = GradientClippingAnalyzer(deep_cnn)
    print("Analyzing gradient clipping effects...")
    clip_analyzer.analyze_clipping_effects(cnn_loader, criterion, clip_value=1.0, num_batches=15)
    clip_analyzer.plot_clipping_analysis()
    
    print("\n5. Layer-wise Gradient Analysis")
    print("-" * 40)
    
    # Layer-wise analysis
    layer_analyzer = LayerWiseGradientAnalyzer(resnet_model)
    print("Analyzing gradients layer by layer...")
    layer_analyzer.analyze_layers(cnn_loader, criterion, num_batches=8)
    
    layer_analyzer.print_layer_comparison()
    layer_analyzer.create_gradient_heatmap()
    
    print("\n6. Gradient Problems Diagnosis")
    print("-" * 35)
    
    # Test different models for gradient problems
    models_to_test = {
        'Simple Model': (simple_model, simple_loader),
        'Deep CNN': (deep_cnn, cnn_loader),
        'ResNet': (resnet_model, cnn_loader)
    }
    
    print("Diagnosing gradient problems across models...")
    
    for model_name, (model, loader) in models_to_test.items():
        print(f"\n{model_name}:")
        
        # Quick gradient analysis
        quick_analyzer = GradientAnalyzer(model)
        quick_analyzer.register_gradient_hooks()
        
        data, targets = next(iter(loader))
        data, targets = data.to(device), targets.to(device)
        
        model.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        stats = quick_analyzer.compute_gradient_stats()
        vanishing = quick_analyzer.detect_vanishing_gradients()
        exploding = quick_analyzer.detect_exploding_gradients()
        
        # Calculate overall statistics
        all_norms = [stat['norm'] for stat in stats.values()]
        avg_norm = np.mean(all_norms)
        min_norm = np.min(all_norms)
        max_norm = np.max(all_norms)
        
        print(f"  Average gradient norm: {avg_norm:.2e}")
        print(f"  Min/Max gradient norm: {min_norm:.2e} / {max_norm:.2e}")
        print(f"  Layers with vanishing gradients: {sum(vanishing.values())}")
        print(f"  Layers with exploding gradients: {sum(exploding.values())}")
        
        # Diagnosis
        if sum(vanishing.values()) > len(stats) * 0.3:
            print("  ⚠️  WARNING: Significant vanishing gradient problem!")
        if sum(exploding.values()) > 0:
            print("  ⚠️  WARNING: Exploding gradient problem detected!")
        if max_norm / min_norm > 1000:
            print("  ⚠️  WARNING: Large gradient magnitude variation!")
        
        quick_analyzer.remove_hooks()
    
    print("\n7. Gradient Optimization Recommendations")
    print("-" * 45)
    
    print("Based on the analysis, here are recommendations:")
    print("\n• For vanishing gradients:")
    print("  - Use residual connections (ResNet-style)")
    print("  - Apply gradient clipping")
    print("  - Use better weight initialization (Xavier/He)")
    print("  - Consider LSTM/GRU for sequence models")
    print("  - Use batch normalization")
    
    print("\n• For exploding gradients:")
    print("  - Apply gradient clipping")
    print("  - Reduce learning rate")
    print("  - Use gradient accumulation")
    print("  - Check weight initialization")
    
    print("\n• For training stability:")
    print("  - Monitor gradient norms during training")
    print("  - Use learning rate scheduling")
    print("  - Apply regularization (dropout, weight decay)")
    print("  - Consider mixed precision training")
    
    print("\nGradient analysis completed!")
    print("Generated files:")
    print("  - gradient_flow.png")
    print("  - gradient_histograms.png")
    print("  - gradient_evolution_*.png")
    print("  - gradient_clipping_analysis.png")
    print("  - gradient_heatmap.png")