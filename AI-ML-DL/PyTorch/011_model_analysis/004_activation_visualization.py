import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, OrderedDict
import cv2
from PIL import Image

# Sample Models for Activation Visualization
class VisualizableCNN(nn.Module):
    """CNN with hooks for activation visualization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.classifier(x)
        return x

class DeepVisualizableModel(nn.Module):
    """Deeper model for advanced visualization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction layers
        self.features = nn.ModuleList([
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        ])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.classifier(x)
        return x

# Activation Visualization Tools
class ActivationVisualizer:
    """Comprehensive activation visualization utilities"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.hooks = []
    
    def register_activation_hooks(self, layer_names: Optional[List[str]] = None):
        """Register hooks to capture activations"""
        
        def make_activation_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks for specified layers or all conv/linear layers
        if layer_names is None:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
                    handle = module.register_forward_hook(make_activation_hook(name))
                    self.hooks.append(handle)
        else:
            layer_dict = dict(self.model.named_modules())
            for name in layer_names:
                if name in layer_dict:
                    handle = layer_dict[name].register_forward_hook(make_activation_hook(name))
                    self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all activation hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def visualize_conv_filters(self, layer_name: str, num_filters: int = 16, 
                              figsize: Tuple[int, int] = (12, 8)):
        """Visualize convolutional filters"""
        
        # Get the layer
        layer_dict = dict(self.model.named_modules())
        if layer_name not in layer_dict:
            print(f"Layer {layer_name} not found")
            return
        
        layer = layer_dict[layer_name]
        if not isinstance(layer, nn.Conv2d):
            print(f"Layer {layer_name} is not a Conv2d layer")
            return
        
        # Get weights
        weights = layer.weight.data.cpu()
        
        # Normalize weights for visualization
        weights_min = weights.min()
        weights_max = weights.max()
        weights_norm = (weights - weights_min) / (weights_max - weights_min)
        
        # Plot filters
        num_filters = min(num_filters, weights.size(0))
        cols = int(np.ceil(np.sqrt(num_filters)))
        rows = int(np.ceil(num_filters / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_filters):
            row, col = i // cols, i % cols
            
            # Get filter weights
            filter_weights = weights_norm[i]
            
            # If multiple input channels, show RGB or average
            if filter_weights.size(0) == 3:  # RGB
                filter_img = filter_weights.permute(1, 2, 0)
            else:
                filter_img = filter_weights.mean(0)  # Average across channels
            
            axes[row, col].imshow(filter_img, cmap='viridis')
            axes[row, col].set_title(f'Filter {i}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_filters, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Filters from {layer_name}')
        plt.tight_layout()
        plt.savefig(f'filters_{layer_name.replace(".", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_feature_maps(self, input_tensor: torch.Tensor, layer_name: str, 
                              num_maps: int = 16, figsize: Tuple[int, int] = (15, 10)):
        """Visualize feature maps for a given input"""
        
        # Clear previous activations
        self.activations = {}
        
        # Forward pass to capture activations
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Get activation for specified layer
        if layer_name not in self.activations:
            print(f"No activation captured for layer: {layer_name}")
            return
        
        activation = self.activations[layer_name]
        
        # Use first sample in batch
        if len(activation.shape) == 4:  # Conv layer (B, C, H, W)
            feature_maps = activation[0].cpu()  # First sample
            num_maps = min(num_maps, feature_maps.size(0))
            
            cols = int(np.ceil(np.sqrt(num_maps)))
            rows = int(np.ceil(num_maps / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if rows == 1:
                axes = axes.reshape(1, -1)
            if cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(num_maps):
                row, col = i // cols, i % cols
                
                feature_map = feature_maps[i]
                im = axes[row, col].imshow(feature_map, cmap='viridis')
                axes[row, col].set_title(f'Channel {i}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(num_maps, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
            
            plt.suptitle(f'Feature Maps from {layer_name}')
            plt.tight_layout()
            plt.savefig(f'feature_maps_{layer_name.replace(".", "_")}.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
        
        elif len(activation.shape) == 2:  # Linear layer (B, features)
            features = activation[0].cpu()
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(features.numpy())
            plt.title(f'Feature Activations from {layer_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Activation Value')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(features.numpy(), bins=50, alpha=0.7)
            plt.title(f'Activation Distribution')
            plt.xlabel('Activation Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'linear_activations_{layer_name.replace(".", "_")}.png', 
                       dpi=150, bbox_inches='tight')
            plt.show()
    
    def create_activation_heatmap(self, input_tensor: torch.Tensor):
        """Create heatmap showing activation statistics across layers"""
        
        # Clear and capture activations
        self.activations = {}
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Calculate statistics for each layer
        layer_stats = {}
        
        for name, activation in self.activations.items():
            if len(activation.shape) >= 2:
                # Flatten spatial dimensions for conv layers
                if len(activation.shape) == 4:  # Conv: (B, C, H, W)
                    act_flat = activation.view(activation.size(0), activation.size(1), -1).mean(dim=2)
                else:  # Linear: (B, features)
                    act_flat = activation
                
                # Calculate statistics
                mean_act = act_flat.mean().item()
                std_act = act_flat.std().item()
                max_act = act_flat.max().item()
                sparsity = (act_flat == 0).float().mean().item()
                
                layer_stats[name] = {
                    'mean': mean_act,
                    'std': std_act,
                    'max': max_act,
                    'sparsity': sparsity
                }
        
        if not layer_stats:
            print("No activation statistics to visualize")
            return
        
        # Create heatmap
        layer_names = list(layer_stats.keys())
        metrics = ['mean', 'std', 'max', 'sparsity']
        
        heatmap_data = []
        for metric in metrics:
            metric_values = [layer_stats[name][metric] for name in layer_names]
            heatmap_data.append(metric_values)
        
        heatmap_data = np.array(heatmap_data)
        
        # Normalize each metric for better visualization
        for i in range(len(metrics)):
            metric_data = heatmap_data[i]
            if metric_data.max() > metric_data.min():
                heatmap_data[i] = (metric_data - metric_data.min()) / (metric_data.max() - metric_data.min())
        
        plt.figure(figsize=(max(12, len(layer_names) * 0.5), 6))
        
        sns.heatmap(heatmap_data,
                   xticklabels=[name.split('.')[-1][:15] for name in layer_names],
                   yticklabels=metrics,
                   annot=False,
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Normalized Value'})
        
        plt.title('Activation Statistics Heatmap')
        plt.xlabel('Layer')
        plt.ylabel('Metric')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('activation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return layer_stats

class ActivationEvolutionAnalyzer:
    """Analyze how activations evolve during training"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.activation_history = defaultdict(list)
        self.hooks = []
    
    def register_tracking_hooks(self, layer_names: List[str]):
        """Register hooks to track activation evolution"""
        
        def make_tracking_hook(name):
            def hook(module, input, output):
                # Store activation statistics
                if len(output.shape) == 4:  # Conv layer
                    stats = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'sparsity': (output == 0).float().mean().item(),
                        'max': output.max().item()
                    }
                else:  # Linear layer
                    stats = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'sparsity': (output == 0).float().mean().item(),
                        'max': output.max().item()
                    }
                
                self.activation_history[name].append(stats)
            return hook
        
        layer_dict = dict(self.model.named_modules())
        for name in layer_names:
            if name in layer_dict:
                handle = layer_dict[name].register_forward_hook(make_tracking_hook(name))
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove tracking hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def track_training_evolution(self, data_loader, criterion, optimizer, 
                               num_epochs: int = 5, batches_per_epoch: int = 10):
        """Track activation evolution during training"""
        
        self.model.train()
        
        for epoch in range(num_epochs):
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= batches_per_epoch:
                    break
                
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def plot_activation_evolution(self, layer_names: Optional[List[str]] = None):
        """Plot how activations evolve over training"""
        
        if layer_names is None:
            layer_names = list(self.activation_history.keys())[:4]  # First 4 layers
        
        metrics = ['mean', 'std', 'sparsity']
        
        fig, axes = plt.subplots(len(metrics), len(layer_names), 
                                figsize=(4 * len(layer_names), 3 * len(metrics)))
        
        if len(layer_names) == 1:
            axes = axes.reshape(-1, 1)
        if len(metrics) == 1:
            axes = axes.reshape(1, -1)
        
        for j, layer_name in enumerate(layer_names):
            if layer_name in self.activation_history:
                history = self.activation_history[layer_name]
                iterations = range(len(history))
                
                for i, metric in enumerate(metrics):
                    values = [step[metric] for step in history]
                    axes[i, j].plot(iterations, values)
                    axes[i, j].set_title(f'{layer_name.split(".")[-1]} - {metric}')
                    axes[i, j].grid(True, alpha=0.3)
                    
                    if i == len(metrics) - 1:
                        axes[i, j].set_xlabel('Training Step')
        
        plt.tight_layout()
        plt.savefig('activation_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()

class GradCAMVisualizer:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: str, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
    
    def register_gradcam_hooks(self):
        """Register hooks for Grad-CAM"""
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Target layer '{self.target_layer}' not found")
        
        # Register hooks
        h1 = target_module.register_forward_hook(forward_hook)
        h2 = target_module.register_backward_hook(backward_hook)
        self.hooks.extend([h1, h2])
    
    def remove_hooks(self):
        """Remove Grad-CAM hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_gradcam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None):
        """Generate Grad-CAM heatmap"""
        
        self.register_gradcam_hooks()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Calculate weights
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            
            # Weighted combination of activation maps
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam / cam.max()
            
            self.remove_hooks()
            return cam.squeeze().detach().cpu().numpy()
        
        self.remove_hooks()
        return None
    
    def visualize_gradcam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None,
                         alpha: float = 0.4):
        """Visualize Grad-CAM overlay on input image"""
        
        # Generate CAM
        cam = self.generate_gradcam(input_tensor, class_idx)
        
        if cam is None:
            print("Failed to generate Grad-CAM")
            return
        
        # Get original image
        img = input_tensor[0].cpu()
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        img_np = img.permute(1, 2, 0).numpy()
        
        # Resize CAM to match input size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]
        
        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('Grad-CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gradcam_class_{class_idx}.png', dpi=150, bbox_inches='tight')
        plt.show()

class ActivationMaximizer:
    """Find inputs that maximize specific activations"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.target_activation = None
        self.hooks = []
    
    def register_target_hook(self, layer_name: str, channel_idx: int):
        """Register hook for target activation"""
        
        def activation_hook(module, input, output):
            if len(output.shape) == 4:  # Conv layer
                self.target_activation = output[:, channel_idx].mean()
            else:  # Linear layer
                self.target_activation = output[:, channel_idx]
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(activation_hook)
                self.hooks.append(handle)
                break
    
    def remove_hooks(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def maximize_activation(self, input_shape: Tuple[int, ...], 
                          layer_name: str, channel_idx: int,
                          num_iterations: int = 200, lr: float = 0.1):
        """Find input that maximizes target activation"""
        
        self.register_target_hook(layer_name, channel_idx)
        
        # Initialize random input
        input_tensor = torch.randn(1, *input_shape, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.Adam([input_tensor], lr=lr)
        
        activations = []
        
        self.model.eval()
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            _ = self.model(input_tensor)
            
            # Maximize activation (minimize negative activation)
            loss = -self.target_activation
            
            loss.backward()
            optimizer.step()
            
            # Record activation
            activations.append(-loss.item())
            
            if i % 50 == 0:
                print(f"Iteration {i}, Activation: {-loss.item():.4f}")
        
        self.remove_hooks()
        
        # Normalize result for visualization
        result = input_tensor.detach()
        result = (result - result.min()) / (result.max() - result.min())
        
        return result, activations
    
    def visualize_maximizing_input(self, input_shape: Tuple[int, ...],
                                  layer_name: str, channel_idx: int):
        """Visualize input that maximizes activation"""
        
        maximizing_input, activations = self.maximize_activation(
            input_shape, layer_name, channel_idx
        )
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot maximizing input
        if len(input_shape) == 3 and input_shape[0] == 3:  # RGB image
            img = maximizing_input[0].cpu().permute(1, 2, 0)
            ax1.imshow(img)
        else:  # Grayscale or other
            img = maximizing_input[0, 0].cpu()
            ax1.imshow(img, cmap='viridis')
        
        ax1.set_title(f'Input maximizing {layer_name}[{channel_idx}]')
        ax1.axis('off')
        
        # Plot activation evolution
        ax2.plot(activations)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Activation Value')
        ax2.set_title('Activation Maximization Progress')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'activation_maximization_{layer_name}_{channel_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("Activation Visualization")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    simple_cnn = VisualizableCNN(num_classes=10).to(device)
    deep_model = DeepVisualizableModel(num_classes=10).to(device)
    
    # Create sample data
    sample_input = torch.randn(4, 3, 32, 32).to(device)
    
    print("\n1. Filter Visualization")
    print("-" * 25)
    
    # Visualize convolutional filters
    visualizer = ActivationVisualizer(simple_cnn, device)
    
    print("Visualizing conv1 filters...")
    visualizer.visualize_conv_filters('conv1', num_filters=16)
    
    print("Visualizing conv3 filters...")
    visualizer.visualize_conv_filters('conv3', num_filters=16)
    
    print("\n2. Feature Map Visualization")
    print("-" * 35)
    
    # Register hooks and visualize feature maps
    visualizer.register_activation_hooks()
    
    print("Visualizing conv1 feature maps...")
    visualizer.visualize_feature_maps(sample_input, 'conv1', num_maps=16)
    
    print("Visualizing conv2 feature maps...")
    visualizer.visualize_feature_maps(sample_input, 'conv2', num_maps=16)
    
    print("Visualizing classifier.3 (linear layer) activations...")
    visualizer.visualize_feature_maps(sample_input, 'classifier.3')
    
    visualizer.remove_hooks()
    
    print("\n3. Activation Statistics Heatmap")
    print("-" * 40)
    
    # Create activation heatmap
    visualizer.register_activation_hooks()
    stats = visualizer.create_activation_heatmap(sample_input)
    
    # Print some statistics
    print("\nActivation Statistics Summary:")
    for layer_name, layer_stats in list(stats.items())[:5]:
        print(f"{layer_name}: mean={layer_stats['mean']:.3f}, "
              f"sparsity={layer_stats['sparsity']:.3f}")
    
    visualizer.remove_hooks()
    
    print("\n4. Activation Evolution Analysis")
    print("-" * 40)
    
    # Track activation evolution during training
    evolution_analyzer = ActivationEvolutionAnalyzer(simple_cnn, device)
    
    # Track specific layers
    layers_to_track = ['conv1', 'conv2', 'conv3']
    evolution_analyzer.register_tracking_hooks(layers_to_track)
    
    # Create sample training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_cnn.parameters(), lr=0.001)
    
    # Create sample data loader
    train_data = torch.randn(160, 3, 32, 32)
    train_targets = torch.randint(0, 10, (160,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    print("Tracking activation evolution during training...")
    evolution_analyzer.track_training_evolution(
        train_loader, criterion, optimizer, num_epochs=3, batches_per_epoch=8
    )
    
    # Plot evolution
    evolution_analyzer.plot_activation_evolution(layers_to_track)
    evolution_analyzer.remove_hooks()
    
    print("\n5. Grad-CAM Visualization")
    print("-" * 30)
    
    # Grad-CAM visualization
    target_layer = 'conv4'  # Use last conv layer
    gradcam = GradCAMVisualizer(simple_cnn, target_layer, device)
    
    print(f"Generating Grad-CAM for layer: {target_layer}")
    
    # Use single image for Grad-CAM
    single_input = sample_input[:1]  # First image only
    
    try:
        gradcam.visualize_gradcam(single_input, class_idx=None)
        print("Grad-CAM visualization completed")
    except Exception as e:
        print(f"Grad-CAM visualization failed: {e}")
    
    print("\n6. Activation Maximization")
    print("-" * 35)
    
    # Activation maximization
    maximizer = ActivationMaximizer(simple_cnn, device)
    
    print("Finding input that maximizes conv2 channel 10...")
    try:
        maximizer.visualize_maximizing_input(
            input_shape=(3, 32, 32),
            layer_name='conv2',
            channel_idx=10
        )
        print("Activation maximization completed")
    except Exception as e:
        print(f"Activation maximization failed: {e}")
    
    print("\n7. Layer-wise Activation Analysis")
    print("-" * 40)
    
    # Comprehensive layer analysis
    analyzer = ActivationVisualizer(deep_model, device)
    analyzer.register_activation_hooks()
    
    # Analyze different inputs
    test_inputs = {
        'Random Noise': torch.randn(1, 3, 32, 32).to(device),
        'Zeros': torch.zeros(1, 3, 32, 32).to(device),
        'Ones': torch.ones(1, 3, 32, 32).to(device),
    }
    
    print("Analyzing activations for different input types:")
    
    for input_name, test_input in test_inputs.items():
        print(f"\n{input_name}:")
        
        # Get activation statistics
        stats = analyzer.create_activation_heatmap(test_input)
        
        # Calculate summary statistics
        all_means = [s['mean'] for s in stats.values()]
        all_sparsities = [s['sparsity'] for s in stats.values()]
        
        print(f"  Average activation mean: {np.mean(all_means):.3f}")
        print(f"  Average sparsity: {np.mean(all_sparsities):.3f}")
        print(f"  Active layers: {sum(1 for m in all_means if m > 0.01)}/{len(all_means)}")
    
    analyzer.remove_hooks()
    
    print("\n8. Activation Pattern Analysis")
    print("-" * 35)
    
    # Analyze activation patterns across different classes
    def analyze_class_activations(model, data_loader, num_classes=10):
        """Analyze how activations differ across classes"""
        
        model.eval()
        class_activations = defaultdict(list)
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Get predictions
                outputs = model(data)
                predictions = outputs.argmax(dim=1)
                
                # Store activations by predicted class
                for i, pred_class in enumerate(predictions):
                    # Use final layer activations before classifier
                    features = model.features(data[i:i+1])
                    features_flat = features.view(-1)
                    class_activations[pred_class.item()].append(features_flat.cpu())
        
        return class_activations
    
    # Create class-specific test data
    test_data = torch.randn(100, 3, 32, 32)
    test_targets = torch.randint(0, 10, (100,))
    test_dataset = torch.utils.data.TensorDataset(test_data, test_targets)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    print("Analyzing activation patterns across predicted classes...")
    class_activations = analyze_class_activations(deep_model, test_loader)
    
    # Plot class-specific activation statistics
    if class_activations:
        class_means = {}
        class_stds = {}
        
        for class_idx, activations in class_activations.items():
            if activations:  # Check if class has samples
                all_activations = torch.cat(activations)
                class_means[class_idx] = all_activations.mean().item()
                class_stds[class_idx] = all_activations.std().item()
        
        if class_means:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            classes = list(class_means.keys())
            means = [class_means[c] for c in classes]
            plt.bar(classes, means)
            plt.xlabel('Predicted Class')
            plt.ylabel('Mean Activation')
            plt.title('Average Activation by Class')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            stds = [class_stds[c] for c in classes]
            plt.bar(classes, stds)
            plt.xlabel('Predicted Class')
            plt.ylabel('Activation Std')
            plt.title('Activation Variability by Class')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('class_activation_patterns.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        print(f"Analyzed {len(class_activations)} classes")
        for class_idx, activations in class_activations.items():
            print(f"  Class {class_idx}: {len(activations)} samples")
    
    print("\nActivation visualization completed!")
    print("Generated files:")
    print("  - filters_*.png (filter visualizations)")
    print("  - feature_maps_*.png (feature map visualizations)")
    print("  - linear_activations_*.png (linear layer activations)")
    print("  - activation_heatmap.png (activation statistics)")
    print("  - activation_evolution.png (training evolution)")
    print("  - gradcam_*.png (Grad-CAM visualizations)")
    print("  - activation_maximization_*.png (maximizing inputs)")
    print("  - class_activation_patterns.png (class-specific patterns)")