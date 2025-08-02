import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

# Sample Models for Saliency Analysis
class SaliencyTestCNN(nn.Module):
    """CNN model for saliency map generation"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
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
        x = self.features(x)
        x = self.classifier(x)
        return x

class AttentionCNN(nn.Module):
    """CNN with attention mechanism for attention map visualization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Store attention maps
        self.attention_maps = None
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Attention mechanism
        attention_weights = self.attention(features)
        self.attention_maps = attention_weights.detach()
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Classification
        pooled = self.global_pool(attended_features)
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_maps(self):
        """Get the last computed attention maps"""
        return self.attention_maps

# Saliency Map Generators
class VanillaSaliency:
    """Vanilla gradient saliency maps"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_saliency(self, input_tensor: torch.Tensor, 
                         target_class: Optional[int] = None) -> torch.Tensor:
        """Generate vanilla saliency map"""
        
        # Ensure input requires gradient
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        score = output[:, target_class]
        score.backward()
        
        # Get gradients
        saliency = input_tensor.grad.data.abs()
        
        # Take maximum across color channels
        saliency, _ = torch.max(saliency, dim=1)
        
        return saliency.squeeze().cpu()
    
    def visualize_saliency(self, input_tensor: torch.Tensor, 
                          target_class: Optional[int] = None,
                          alpha: float = 0.5):
        """Visualize saliency map overlayed on input"""
        
        saliency = self.generate_saliency(input_tensor, target_class)
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy
        img_np = img_denorm.permute(1, 2, 0).numpy()
        saliency_np = saliency.numpy()
        
        # Normalize saliency
        saliency_norm = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min())
        
        # Create heatmap
        heatmap = plt.cm.jet(saliency_norm)[:, :, :3]
        
        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(saliency_norm, cmap='hot')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('Saliency Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'vanilla_saliency_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class GuidedBackprop:
    """Guided backpropagation saliency"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.hooks = []
        self.gradients = []
    
    def register_hooks(self):
        """Register hooks for guided backprop"""
        
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hook for all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                handle = module.register_backward_hook(relu_hook_function)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_guided_saliency(self, input_tensor: torch.Tensor,
                                target_class: Optional[int] = None) -> torch.Tensor:
        """Generate guided backprop saliency map"""
        
        self.register_hooks()
        
        # Ensure input requires gradient
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        score = output[:, target_class]
        score.backward()
        
        # Get gradients
        guided_gradients = input_tensor.grad.data
        
        self.remove_hooks()
        
        return guided_gradients.squeeze().cpu()
    
    def visualize_guided_saliency(self, input_tensor: torch.Tensor,
                                 target_class: Optional[int] = None):
        """Visualize guided backprop saliency"""
        
        guided_grads = self.generate_guided_saliency(input_tensor, target_class)
        
        # Convert to numpy and process
        guided_grads_np = guided_grads.permute(1, 2, 0).numpy()
        
        # Normalize
        guided_grads_norm = guided_grads_np - guided_grads_np.min()
        guided_grads_norm /= guided_grads_norm.max()
        
        # Create grayscale version
        guided_grads_gray = np.mean(guided_grads_norm, axis=2)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        axes[0].imshow(guided_grads_norm)
        axes[0].set_title('Guided Backprop (Color)')
        axes[0].axis('off')
        
        axes[1].imshow(guided_grads_gray, cmap='gray')
        axes[1].set_title('Guided Backprop (Grayscale)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'guided_backprop_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class IntegratedGradients:
    """Integrated Gradients saliency method"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_integrated_gradients(self, input_tensor: torch.Tensor,
                                    target_class: Optional[int] = None,
                                    baseline: Optional[torch.Tensor] = None,
                                    steps: int = 50) -> torch.Tensor:
        """Generate integrated gradients"""
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        
        # Forward pass to get target class
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)
        
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolated input
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_input)
            
            # Backward pass
            self.model.zero_grad()
            score = output[:, target_class]
            score.backward()
            
            # Accumulate gradients
            integrated_grads += interpolated_input.grad.data
        
        # Average and scale by input difference
        integrated_grads /= steps
        integrated_grads *= (input_tensor - baseline)
        
        return integrated_grads.squeeze().cpu()
    
    def visualize_integrated_gradients(self, input_tensor: torch.Tensor,
                                     target_class: Optional[int] = None,
                                     alpha: float = 0.5):
        """Visualize integrated gradients"""
        
        ig = self.generate_integrated_gradients(input_tensor, target_class)
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Process integrated gradients
        ig_np = ig.permute(1, 2, 0).numpy()
        
        # Take magnitude and convert to grayscale
        ig_magnitude = np.sqrt(np.sum(ig_np ** 2, axis=2))
        
        # Normalize
        ig_norm = (ig_magnitude - ig_magnitude.min()) / (ig_magnitude.max() - ig_magnitude.min())
        
        # Create heatmap
        heatmap = plt.cm.jet(ig_norm)[:, :, :3]
        
        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(ig_norm, cmap='hot')
        axes[1].set_title('Integrated Gradients')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('IG Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'integrated_gradients_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model: nn.Module, target_layer: str, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
    
    def register_hooks(self):
        """Register hooks for GradCAM"""
        
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
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_gradcam(self, input_tensor: torch.Tensor,
                        target_class: Optional[int] = None) -> torch.Tensor:
        """Generate GradCAM heatmap"""
        
        self.register_hooks()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()
        
        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Calculate weights (global average pooling of gradients)
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            
            # Weighted combination of activation maps
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            
            # Apply ReLU
            cam = F.relu(cam)
            
            # Normalize
            cam = cam / cam.max()
            
            self.remove_hooks()
            return cam.squeeze().detach().cpu()
        
        self.remove_hooks()
        return None
    
    def visualize_gradcam(self, input_tensor: torch.Tensor,
                         target_class: Optional[int] = None,
                         alpha: float = 0.4):
        """Visualize GradCAM overlay"""
        
        cam = self.generate_gradcam(input_tensor, target_class)
        
        if cam is None:
            print("Failed to generate GradCAM")
            return
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy
        img_np = img_denorm.permute(1, 2, 0).numpy()
        cam_np = cam.numpy()
        
        # Resize CAM to match input size
        cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
        
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
        axes[1].set_title('GradCAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gradcam_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class SmoothGrad:
    """SmoothGrad: removing noise by adding noise"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_smooth_grad(self, input_tensor: torch.Tensor,
                           target_class: Optional[int] = None,
                           noise_level: float = 0.15,
                           n_samples: int = 50) -> torch.Tensor:
        """Generate SmoothGrad saliency map"""
        
        # Forward pass to get target class
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Accumulate gradients
        total_gradients = torch.zeros_like(input_tensor)
        
        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            
            # Backward pass
            self.model.zero_grad()
            score = output[:, target_class]
            score.backward()
            
            # Accumulate gradients
            total_gradients += noisy_input.grad.data
        
        # Average gradients
        smooth_grad = total_gradients / n_samples
        
        return smooth_grad.squeeze().cpu()
    
    def visualize_smooth_grad(self, input_tensor: torch.Tensor,
                             target_class: Optional[int] = None,
                             alpha: float = 0.5):
        """Visualize SmoothGrad"""
        
        smooth_grad = self.generate_smooth_grad(input_tensor, target_class)
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Process smooth gradients
        smooth_grad_np = smooth_grad.permute(1, 2, 0).numpy()
        
        # Take magnitude
        smooth_grad_magnitude = np.sqrt(np.sum(smooth_grad_np ** 2, axis=2))
        
        # Normalize
        smooth_grad_norm = (smooth_grad_magnitude - smooth_grad_magnitude.min()) / \
                          (smooth_grad_magnitude.max() - smooth_grad_magnitude.min())
        
        # Create heatmap
        heatmap = plt.cm.jet(smooth_grad_norm)[:, :, :3]
        
        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(smooth_grad_norm, cmap='hot')
        axes[1].set_title('SmoothGrad')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('SmoothGrad Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'smooth_grad_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class AttentionVisualizer:
    """Visualize attention maps from attention-based models"""
    
    def __init__(self, model: AttentionCNN, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def visualize_attention_maps(self, input_tensor: torch.Tensor):
        """Visualize learned attention maps"""
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            attention_maps = self.model.get_attention_maps()
        
        if attention_maps is None:
            print("No attention maps available")
            return
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        
        # Convert to numpy
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Get attention map for first sample
        attention_map = attention_maps[0, 0].cpu().numpy()  # First sample, first (and only) channel
        
        # Resize attention map to match input size
        attention_resized = cv2.resize(attention_map, (img_np.shape[1], img_np.shape[0]))
        
        # Create heatmap
        heatmap = plt.cm.jet(attention_resized)[:, :, :3]
        
        # Overlay
        alpha = 0.4
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attention_resized, cmap='jet')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlayed)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return attention_map

class SaliencyComparison:
    """Compare different saliency methods"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def compare_saliency_methods(self, input_tensor: torch.Tensor,
                               target_class: Optional[int] = None,
                               target_layer: str = 'features.9'):
        """Compare multiple saliency methods"""
        
        # Initialize saliency methods
        vanilla = VanillaSaliency(self.model, self.device)
        guided = GuidedBackprop(self.model, self.device)
        integrated = IntegratedGradients(self.model, self.device)
        gradcam = GradCAM(self.model, target_layer, self.device)
        smooth = SmoothGrad(self.model, self.device)
        
        # Generate saliency maps
        print("Generating saliency maps...")
        
        vanilla_sal = vanilla.generate_saliency(input_tensor, target_class)
        guided_sal = guided.generate_guided_saliency(input_tensor, target_class)
        ig_sal = integrated.generate_integrated_gradients(input_tensor, target_class)
        gradcam_sal = gradcam.generate_gradcam(input_tensor, target_class)
        smooth_sal = smooth.generate_smooth_grad(input_tensor, target_class)
        
        # Process for visualization
        def process_saliency(sal_map, is_gradcam=False):
            if is_gradcam:
                # GradCAM is already 2D
                sal_np = sal_map.numpy()
                # Resize to match input
                sal_resized = cv2.resize(sal_np, (input_tensor.shape[3], input_tensor.shape[2]))
            else:
                # Other methods are 3D, take magnitude
                sal_np = sal_map.permute(1, 2, 0).numpy()
                sal_resized = np.sqrt(np.sum(sal_np ** 2, axis=2))
            
            # Normalize
            sal_norm = (sal_resized - sal_resized.min()) / (sal_resized.max() - sal_resized.min())
            return sal_norm
        
        # Process all saliency maps
        vanilla_proc = process_saliency(vanilla_sal.abs())  # Take absolute value for vanilla
        guided_proc = process_saliency(guided_sal.abs())
        ig_proc = process_saliency(ig_sal.abs())
        gradcam_proc = process_saliency(gradcam_sal, is_gradcam=True)
        smooth_proc = process_saliency(smooth_sal.abs())
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Saliency methods
        methods = [
            ('Vanilla Gradient', vanilla_proc),
            ('Guided Backprop', guided_proc),
            ('Integrated Gradients', ig_proc),
            ('GradCAM', gradcam_proc),
            ('SmoothGrad', smooth_proc)
        ]
        
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for (method_name, sal_map), (row, col) in zip(methods, positions):
            im = axes[row, col].imshow(sal_map, cmap='hot')
            axes[row, col].set_title(method_name)
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'saliency_comparison_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate correlation between methods
        saliency_maps = {
            'Vanilla': vanilla_proc.flatten(),
            'Guided': guided_proc.flatten(),
            'Integrated': ig_proc.flatten(),
            'GradCAM': gradcam_proc.flatten(),
            'SmoothGrad': smooth_proc.flatten()
        }
        
        # Correlation matrix
        method_names = list(saliency_maps.keys())
        n_methods = len(method_names)
        correlation_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                correlation = np.corrcoef(saliency_maps[method1], saliency_maps[method2])[0, 1]
                correlation_matrix[i, j] = correlation
        
        # Plot correlation matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(n_methods), method_names, rotation=45)
        plt.yticks(range(n_methods), method_names)
        plt.title('Correlation Between Saliency Methods')
        
        # Add correlation values as text
        for i in range(n_methods):
            for j in range(n_methods):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                        ha='center', va='center', 
                        color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        plt.savefig('saliency_correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'vanilla': vanilla_sal,
            'guided': guided_sal,
            'integrated': ig_sal,
            'gradcam': gradcam_sal,
            'smooth': smooth_sal,
            'correlation_matrix': correlation_matrix
        }

if __name__ == "__main__":
    print("Saliency Maps and Attention Visualization")
    print("=" * 45)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    test_model = SaliencyTestCNN(num_classes=10).to(device)
    attention_model = AttentionCNN(num_classes=10).to(device)
    
    # Create sample data
    sample_input = torch.randn(1, 3, 64, 64).to(device)  # Single image for saliency
    
    print("\n1. Vanilla Gradient Saliency")
    print("-" * 35)
    
    # Vanilla saliency
    vanilla_saliency = VanillaSaliency(test_model, device)
    print("Generating vanilla gradient saliency...")
    vanilla_saliency.visualize_saliency(sample_input, target_class=None)
    
    print("\n2. Guided Backpropagation")
    print("-" * 30)
    
    # Guided backprop
    guided_backprop = GuidedBackprop(test_model, device)
    print("Generating guided backpropagation saliency...")
    guided_backprop.visualize_guided_saliency(sample_input, target_class=None)
    
    print("\n3. Integrated Gradients")
    print("-" * 25)
    
    # Integrated gradients
    integrated_grads = IntegratedGradients(test_model, device)
    print("Generating integrated gradients...")
    integrated_grads.visualize_integrated_gradients(sample_input, target_class=None)
    
    print("\n4. GradCAM Visualization")
    print("-" * 30)
    
    # GradCAM
    target_layer = 'features.9'  # Last conv layer
    gradcam = GradCAM(test_model, target_layer, device)
    print(f"Generating GradCAM for layer: {target_layer}")
    gradcam.visualize_gradcam(sample_input, target_class=None)
    
    print("\n5. SmoothGrad")
    print("-" * 15)
    
    # SmoothGrad
    smooth_grad = SmoothGrad(test_model, device)
    print("Generating SmoothGrad...")
    smooth_grad.visualize_smooth_grad(sample_input, target_class=None)
    
    print("\n6. Attention Visualization")
    print("-" * 30)
    
    # Attention visualization
    attention_vis = AttentionVisualizer(attention_model, device)
    print("Visualizing learned attention maps...")
    attention_map = attention_vis.visualize_attention_maps(sample_input)
    
    if attention_map is not None:
        print(f"Attention map shape: {attention_map.shape}")
        print(f"Attention range: [{attention_map.min():.3f}, {attention_map.max():.3f}]")
        print(f"Attention mean: {attention_map.mean():.3f}")
    
    print("\n7. Saliency Method Comparison")
    print("-" * 35)
    
    # Compare all saliency methods
    comparison = SaliencyComparison(test_model, device)
    print("Comparing all saliency methods...")
    
    results = comparison.compare_saliency_methods(
        sample_input, 
        target_class=None,
        target_layer='features.9'
    )
    
    # Print correlation analysis
    print("\nSaliency Method Correlation Analysis:")
    correlation_matrix = results['correlation_matrix']
    method_names = ['Vanilla', 'Guided', 'Integrated', 'GradCAM', 'SmoothGrad']
    
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i < j:  # Only print upper triangle
                corr = correlation_matrix[i, j]
                print(f"  {method1} vs {method2}: {corr:.3f}")
    
    print("\n8. Sensitivity Analysis")
    print("-" * 25)
    
    # Analyze sensitivity to different input modifications
    base_input = sample_input.clone()
    
    # Test different input perturbations
    perturbations = {
        'Original': base_input,
        'Gaussian Noise': base_input + 0.1 * torch.randn_like(base_input),
        'Salt & Pepper': base_input.clone(),
        'Blur': base_input.clone()
    }
    
    # Add salt and pepper noise
    salt_pepper = perturbations['Salt & Pepper']
    noise_mask = torch.rand_like(salt_pepper) < 0.05
    salt_pepper[noise_mask] = torch.randint(0, 2, salt_pepper[noise_mask].shape).float().to(device)
    
    # Simple blur (average with neighbors)
    blur_input = perturbations['Blur']
    kernel = torch.ones(1, 1, 3, 3).to(device) / 9
    for c in range(3):
        blur_input[0, c:c+1] = F.conv2d(blur_input[0, c:c+1].unsqueeze(0), kernel, padding=1)[0]
    
    print("Analyzing saliency sensitivity to input perturbations...")
    
    # Generate vanilla saliency for each perturbation
    sensitivity_results = {}
    
    for pert_name, pert_input in perturbations.items():
        try:
            sal_map = vanilla_saliency.generate_saliency(pert_input, target_class=0)
            sal_magnitude = sal_map.abs().mean().item()
            sal_std = sal_map.abs().std().item()
            
            sensitivity_results[pert_name] = {
                'magnitude': sal_magnitude,
                'std': sal_std
            }
        except Exception as e:
            print(f"Error processing {pert_name}: {e}")
            sensitivity_results[pert_name] = {'magnitude': 0, 'std': 0}
    
    # Plot sensitivity results
    pert_names = list(sensitivity_results.keys())
    magnitudes = [sensitivity_results[name]['magnitude'] for name in pert_names]
    stds = [sensitivity_results[name]['std'] for name in pert_names]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(pert_names, magnitudes, alpha=0.7)
    plt.title('Saliency Magnitude by Input Type')
    plt.ylabel('Average Saliency Magnitude')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(pert_names, stds, alpha=0.7, color='orange')
    plt.title('Saliency Variability by Input Type')
    plt.ylabel('Saliency Standard Deviation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('saliency_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nSensitivity Analysis Results:")
    for name, results in sensitivity_results.items():
        print(f"  {name}: magnitude={results['magnitude']:.4f}, std={results['std']:.4f}")
    
    print("\n9. Class-specific Saliency Analysis")
    print("-" * 40)
    
    # Analyze saliency for different target classes
    target_classes = [0, 1, 2, 3, 4]  # First 5 classes
    
    print("Generating saliency maps for different target classes...")
    
    class_saliencies = {}
    
    for target_class in target_classes:
        sal_map = vanilla_saliency.generate_saliency(sample_input, target_class=target_class)
        sal_magnitude = sal_map.abs().mean().item()
        
        class_saliencies[target_class] = {
            'saliency_map': sal_map,
            'magnitude': sal_magnitude
        }
    
    # Plot class-specific saliency comparison
    fig, axes = plt.subplots(1, len(target_classes), figsize=(4*len(target_classes), 4))
    
    for i, target_class in enumerate(target_classes):
        sal_map = class_saliencies[target_class]['saliency_map']
        sal_magnitude = class_saliencies[target_class]['magnitude']
        
        # Take max across channels and normalize
        sal_display = sal_map.max(dim=0)[0]
        sal_display = (sal_display - sal_display.min()) / (sal_display.max() - sal_display.min())
        
        im = axes[i].imshow(sal_display, cmap='hot')
        axes[i].set_title(f'Class {target_class}\nMag: {sal_magnitude:.3f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Class-specific Saliency Maps')
    plt.tight_layout()
    plt.savefig('class_specific_saliency.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nClass-specific Saliency Results:")
    for target_class, results in class_saliencies.items():
        print(f"  Class {target_class}: magnitude={results['magnitude']:.4f}")
    
    print("\nSaliency visualization completed!")
    print("Generated files:")
    print("  - vanilla_saliency_class_*.png")
    print("  - guided_backprop_class_*.png") 
    print("  - integrated_gradients_class_*.png")
    print("  - gradcam_class_*.png")
    print("  - smooth_grad_class_*.png")
    print("  - attention_visualization.png")
    print("  - saliency_comparison_class_*.png")
    print("  - saliency_correlation_matrix.png")
    print("  - saliency_sensitivity.png")
    print("  - class_specific_saliency.png")