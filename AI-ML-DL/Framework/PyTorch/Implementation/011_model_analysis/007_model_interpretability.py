import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import itertools

# Mock LIME and SHAP implementations (simplified versions)
# In practice, you would use: pip install lime shap

class LIMEImageExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) for images"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_perturbations(self, image: torch.Tensor, 
                             num_samples: int = 1000,
                             num_features: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate perturbed versions of the image"""
        
        batch_size, channels, height, width = image.shape
        
        # Create superpixel-like segments (simplified as grid patches)
        patch_h = height // int(np.sqrt(num_features))
        patch_w = width // int(np.sqrt(num_features))
        
        # Generate binary masks for each sample
        perturbations = []
        masks = []
        
        for _ in range(num_samples):
            # Random binary mask for each patch
            mask = torch.randint(0, 2, (int(np.sqrt(num_features)), int(np.sqrt(num_features))))
            
            # Resize mask to image size
            mask_resized = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode='nearest'
            ).squeeze()
            
            # Apply mask to image (zero out masked regions)
            perturbed_image = image.clone()
            for c in range(channels):
                perturbed_image[0, c] *= mask_resized
            
            perturbations.append(perturbed_image)
            masks.append(mask.flatten())
        
        return torch.stack(perturbations), torch.stack(masks)
    
    def explain_instance(self, image: torch.Tensor, 
                        target_class: Optional[int] = None,
                        num_samples: int = 1000,
                        num_features: int = 100) -> Dict[str, Any]:
        """Explain a single instance using LIME"""
        
        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(image)
            if target_class is None:
                target_class = original_output.argmax(dim=1).item()
            original_prob = F.softmax(original_output, dim=1)[0, target_class].item()
        
        # Generate perturbations
        perturbations, masks = self.generate_perturbations(image, num_samples, num_features)
        
        # Get predictions for all perturbations
        predictions = []
        
        with torch.no_grad():
            for pert in perturbations:
                output = self.model(pert.unsqueeze(0))
                prob = F.softmax(output, dim=1)[0, target_class].item()
                predictions.append(prob)
        
        predictions = np.array(predictions)
        masks_np = masks.numpy()
        
        # Fit linear model
        lr = LinearRegression()
        lr.fit(masks_np, predictions)
        
        # Get feature importance
        feature_importance = lr.coef_
        
        # Reshape to 2D for visualization
        importance_2d = feature_importance.reshape(int(np.sqrt(num_features)), int(np.sqrt(num_features)))
        
        return {
            'feature_importance': feature_importance,
            'importance_2d': importance_2d,
            'original_prob': original_prob,
            'target_class': target_class,
            'r2_score': lr.score(masks_np, predictions)
        }
    
    def visualize_explanation(self, image: torch.Tensor, explanation: Dict[str, Any]):
        """Visualize LIME explanation"""
        
        # Get original image
        img = image[0].cpu().detach()
        
        # Denormalize if needed
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Get importance map
        importance_2d = explanation['importance_2d']
        
        # Resize importance to match image size
        height, width = img_np.shape[:2]
        importance_resized = cv2.resize(importance_2d, (width, height))
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Importance map
        im1 = axes[1].imshow(importance_resized, cmap='RdBu_r', 
                           vmin=importance_resized.min(), vmax=importance_resized.max())
        axes[1].set_title(f'LIME Explanation\nClass {explanation["target_class"]} '
                         f'(prob: {explanation["original_prob"]:.3f})')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        # Normalize importance for overlay
        importance_norm = (importance_resized - importance_resized.min()) / \
                         (importance_resized.max() - importance_resized.min())
        
        # Create heatmap overlay
        alpha = 0.4
        heatmap = plt.cm.RdBu_r(importance_norm)[:, :, :3]
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        axes[2].imshow(overlayed)
        axes[2].set_title(f'LIME Overlay\nR² Score: {explanation["r2_score"]:.3f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'lime_explanation_class_{explanation["target_class"]}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

class SHAPDeepExplainer:
    """Simplified SHAP Deep Explainer for PyTorch models"""
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.background = background_data.to(device)
    
    def shap_values(self, input_tensor: torch.Tensor, 
                   target_class: Optional[int] = None,
                   n_samples: int = 100) -> torch.Tensor:
        """Compute SHAP values using sampling approximation"""
        
        # Get baseline prediction
        self.model.eval()
        with torch.no_grad():
            baseline_output = self.model(self.background.mean(dim=0, keepdim=True))
            input_output = self.model(input_tensor)
            
            if target_class is None:
                target_class = input_output.argmax(dim=1).item()
            
            baseline_score = baseline_output[0, target_class].item()
            input_score = input_output[0, target_class].item()
        
        # Compute SHAP values using sampling
        batch_size, channels, height, width = input_tensor.shape
        total_pixels = channels * height * width
        
        shap_values = torch.zeros_like(input_tensor)
        
        # Sample random coalitions
        for _ in range(n_samples):
            # Random subset of pixels
            coalition_size = np.random.randint(1, total_pixels)
            coalition = np.random.choice(total_pixels, coalition_size, replace=False)
            
            # Create two versions: with and without current pixel
            for pixel_idx in range(total_pixels):
                c = pixel_idx // (height * width)
                h = (pixel_idx % (height * width)) // width
                w = pixel_idx % width
                
                # Coalition without current pixel
                coalition_without = coalition[coalition != pixel_idx]
                
                # Coalition with current pixel
                coalition_with = np.append(coalition_without, pixel_idx)
                
                # Create masked inputs
                input_without = self.background.mean(dim=0, keepdim=True).clone()
                input_with = self.background.mean(dim=0, keepdim=True).clone()
                
                # Set coalition pixels to input values
                for idx in coalition_without:
                    c_idx = idx // (height * width)
                    h_idx = (idx % (height * width)) // width
                    w_idx = idx % width
                    input_without[0, c_idx, h_idx, w_idx] = input_tensor[0, c_idx, h_idx, w_idx]
                    input_with[0, c_idx, h_idx, w_idx] = input_tensor[0, c_idx, h_idx, w_idx]
                
                # Add current pixel to "with" version
                input_with[0, c, h, w] = input_tensor[0, c, h, w]
                
                # Get predictions
                with torch.no_grad():
                    output_without = self.model(input_without)
                    output_with = self.model(input_with)
                    
                    score_without = output_without[0, target_class].item()
                    score_with = output_with[0, target_class].item()
                
                # Marginal contribution
                marginal_contribution = score_with - score_without
                shap_values[0, c, h, w] += marginal_contribution
        
        # Average over samples
        shap_values /= n_samples
        
        return shap_values.cpu()
    
    def visualize_shap(self, input_tensor: torch.Tensor, shap_values: torch.Tensor,
                      target_class: int):
        """Visualize SHAP values"""
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Process SHAP values
        shap_np = shap_values[0].numpy()
        
        # Sum across channels for visualization
        shap_sum = np.sum(shap_np, axis=0)
        
        # Normalize
        shap_norm = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min())
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # SHAP values
        im1 = axes[1].imshow(shap_sum, cmap='RdBu_r')
        axes[1].set_title(f'SHAP Values\nClass {target_class}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        alpha = 0.4
        heatmap = plt.cm.RdBu_r(shap_norm)[:, :, :3]
        overlayed = (1 - alpha) * img_np + alpha * heatmap
        
        axes[2].imshow(overlayed)
        axes[2].set_title('SHAP Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'shap_explanation_class_{target_class}.png', dpi=150, bbox_inches='tight')
        plt.show()

class PermutationImportance:
    """Permutation importance for feature importance analysis"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def compute_importance(self, data_loader, 
                         metric_fn: Callable = None,
                         n_repeats: int = 5) -> Dict[str, float]:
        """Compute permutation importance for model layers"""
        
        if metric_fn is None:
            metric_fn = self._accuracy_metric
        
        # Get baseline performance
        baseline_score = self._evaluate_model(data_loader, metric_fn)
        
        # Track activations from different layers
        layer_names = []
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for conv and linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layer_names.append(name)
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)
        
        layer_importance = {}
        
        # For each layer, compute importance by permutation
        for layer_name in layer_names[:5]:  # Limit to first 5 layers for efficiency
            print(f"Computing importance for layer: {layer_name}")
            
            importance_scores = []
            
            for repeat in range(n_repeats):
                # Reset activations
                activations = {}
                
                total_correct = 0
                total_samples = 0
                
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(data_loader):
                        if batch_idx >= 10:  # Limit batches for efficiency
                            break
                        
                        data, targets = data.to(self.device), targets.to(self.device)
                        
                        # Forward pass to get activations
                        outputs = self.model(data)
                        
                        # Permute activations for target layer
                        if layer_name in activations:
                            # Create permuted version
                            permuted_activations = activations[layer_name].clone()
                            
                            # Permute across batch dimension
                            perm_indices = torch.randperm(permuted_activations.size(0))
                            permuted_activations = permuted_activations[perm_indices]
                            
                            # Replace activations and continue forward pass
                            # Note: This is a simplified version - real implementation would need
                            # to modify the forward pass or use hooks more carefully
                            
                            # For now, just compute degradation based on activation statistics
                            original_mean = activations[layer_name].mean().item()
                            permuted_mean = permuted_activations.mean().item()
                            degradation = abs(original_mean - permuted_mean) / (abs(original_mean) + 1e-8)
                            
                            importance_scores.append(degradation)
                        else:
                            importance_scores.append(0.0)
            
            layer_importance[layer_name] = np.mean(importance_scores)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_importance
    
    def _accuracy_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Simple accuracy metric"""
        predictions = outputs.argmax(dim=1)
        correct = (predictions == targets).float().mean().item()
        return correct
    
    def _evaluate_model(self, data_loader, metric_fn: Callable) -> float:
        """Evaluate model performance"""
        total_score = 0.0
        total_batches = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                if batch_idx >= 10:  # Limit for efficiency
                    break
                
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                
                score = metric_fn(outputs, targets)
                total_score += score
                total_batches += 1
        
        return total_score / total_batches if total_batches > 0 else 0.0
    
    def visualize_importance(self, importance_scores: Dict[str, float]):
        """Visualize permutation importance scores"""
        
        layers = list(importance_scores.keys())
        scores = list(importance_scores.values())
        
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(range(len(layers)), scores, alpha=0.7)
        plt.xlabel('Layer')
        plt.ylabel('Permutation Importance')
        plt.title('Layer Importance via Permutation')
        plt.xticks(range(len(layers)), [layer.split('.')[-1] for layer in layers], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('permutation_importance.png', dpi=150, bbox_inches='tight')
        plt.show()

class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def analyze_input_importance(self, input_tensor: torch.Tensor,
                                target_class: Optional[int] = None,
                                methods: List[str] = ['gradient', 'integrated_gradient', 'occlusion']):
        """Analyze input feature importance using multiple methods"""
        
        results = {}
        
        # Get target class
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
        
        # Gradient-based importance
        if 'gradient' in methods:
            results['gradient'] = self._gradient_importance(input_tensor, target_class)
        
        # Integrated gradients
        if 'integrated_gradient' in methods:
            results['integrated_gradient'] = self._integrated_gradient_importance(input_tensor, target_class)
        
        # Occlusion-based importance
        if 'occlusion' in methods:
            results['occlusion'] = self._occlusion_importance(input_tensor, target_class)
        
        return results, target_class
    
    def _gradient_importance(self, input_tensor: torch.Tensor, target_class: int):
        """Gradient-based feature importance"""
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        
        score = output[:, target_class]
        score.backward()
        
        importance = input_tensor.grad.abs().sum(dim=1).squeeze().cpu()
        return importance
    
    def _integrated_gradient_importance(self, input_tensor: torch.Tensor, 
                                      target_class: int, steps: int = 20):
        """Integrated gradients importance"""
        baseline = torch.zeros_like(input_tensor)
        
        alphas = torch.linspace(0, 1, steps).to(self.device)
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            self.model.zero_grad()
            
            score = output[:, target_class]
            score.backward()
            
            integrated_grads += interpolated.grad.data
        
        integrated_grads /= steps
        integrated_grads *= (input_tensor - baseline)
        
        importance = integrated_grads.abs().sum(dim=1).squeeze().cpu()
        return importance
    
    def _occlusion_importance(self, input_tensor: torch.Tensor, target_class: int,
                            window_size: int = 8):
        """Occlusion-based feature importance"""
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            baseline_score = baseline_output[0, target_class].item()
        
        _, _, height, width = input_tensor.shape
        importance = torch.zeros(height, width)
        
        # Slide occlusion window
        for i in range(0, height - window_size + 1, window_size // 2):
            for j in range(0, width - window_size + 1, window_size // 2):
                # Create occluded version
                occluded = input_tensor.clone()
                occluded[:, :, i:i+window_size, j:j+window_size] = 0
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(occluded)
                    score = output[0, target_class].item()
                
                # Importance is the drop in confidence
                drop = baseline_score - score
                importance[i:i+window_size, j:j+window_size] += drop
        
        return importance
    
    def visualize_feature_importance(self, input_tensor: torch.Tensor,
                                   results: Dict[str, torch.Tensor],
                                   target_class: int):
        """Visualize feature importance results"""
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Create subplot for each method
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Plot each method
        for idx, (method_name, importance) in enumerate(results.items()):
            ax = axes[idx + 1]
            
            # Normalize importance
            importance_np = importance.numpy()
            importance_norm = (importance_np - importance_np.min()) / \
                             (importance_np.max() - importance_np.min())
            
            im = ax.imshow(importance_norm, cmap='hot')
            ax.set_title(f'{method_name.replace("_", " ").title()}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Feature Importance Analysis - Class {target_class}')
        plt.tight_layout()
        plt.savefig(f'feature_importance_analysis_class_{target_class}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

class ModelInterpretabilityComparison:
    """Compare different interpretability methods"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def comprehensive_analysis(self, input_tensor: torch.Tensor,
                             background_data: torch.Tensor,
                             target_class: Optional[int] = None):
        """Perform comprehensive interpretability analysis"""
        
        # Initialize explainers
        lime_explainer = LIMEImageExplainer(self.model, self.device)
        shap_explainer = SHAPDeepExplainer(self.model, background_data[:5], self.device)  # Use subset
        feature_analyzer = FeatureImportanceAnalyzer(self.model, self.device)
        
        results = {}
        
        # Get target class
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        print(f"Analyzing prediction: Class {target_class} with confidence {confidence:.3f}")
        
        # LIME explanation
        print("Generating LIME explanation...")
        lime_result = lime_explainer.explain_instance(input_tensor, target_class, num_samples=500)
        results['lime'] = lime_result
        
        # SHAP explanation (simplified)
        print("Generating SHAP explanation...")
        shap_values = shap_explainer.shap_values(input_tensor, target_class, n_samples=20)
        results['shap'] = shap_values
        
        # Feature importance analysis
        print("Computing feature importance...")
        feature_results, _ = feature_analyzer.analyze_input_importance(
            input_tensor, target_class, 
            methods=['gradient', 'occlusion']  # Skip integrated gradients for speed
        )
        results['feature_importance'] = feature_results
        
        return results, target_class, confidence
    
    def visualize_comparison(self, input_tensor: torch.Tensor, results: Dict[str, Any],
                           target_class: int, confidence: float):
        """Visualize comparison of all methods"""
        
        # Get original image
        img = input_tensor[0].cpu().detach()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f'Original Image\nClass {target_class} (conf: {confidence:.3f})')
        axes[0, 0].axis('off')
        
        # LIME
        if 'lime' in results:
            lime_importance = results['lime']['importance_2d']
            height, width = img_np.shape[:2]
            lime_resized = cv2.resize(lime_importance, (width, height))
            
            im1 = axes[0, 1].imshow(lime_resized, cmap='RdBu_r')
            axes[0, 1].set_title(f'LIME\nR² = {results["lime"]["r2_score"]:.3f}')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # SHAP
        if 'shap' in results:
            shap_values = results['shap'][0].numpy()
            shap_sum = np.sum(shap_values, axis=0)
            
            im2 = axes[0, 2].imshow(shap_sum, cmap='RdBu_r')
            axes[0, 2].set_title('SHAP Values')
            axes[0, 2].axis('off')
            plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Feature importance methods
        if 'feature_importance' in results:
            feature_results = results['feature_importance']
            
            # Gradient-based
            if 'gradient' in feature_results:
                grad_importance = feature_results['gradient'].numpy()
                im3 = axes[1, 0].imshow(grad_importance, cmap='hot')
                axes[1, 0].set_title('Gradient Importance')
                axes[1, 0].axis('off')
                plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Occlusion-based
            if 'occlusion' in feature_results:
                occ_importance = feature_results['occlusion'].numpy()
                im4 = axes[1, 1].imshow(occ_importance, cmap='hot')
                axes[1, 1].set_title('Occlusion Importance')
                axes[1, 1].axis('off')
                plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Summary statistics
        axes[1, 2].axis('off')
        summary_text = "Method Comparison:\n\n"
        
        if 'lime' in results:
            summary_text += f"LIME R² Score: {results['lime']['r2_score']:.3f}\n"
            summary_text += f"LIME Features: {len(results['lime']['feature_importance'])}\n\n"
        
        if 'shap' in results:
            shap_values = results['shap'][0]
            shap_mean = shap_values.abs().mean().item()
            summary_text += f"SHAP Mean |Value|: {shap_mean:.4f}\n\n"
        
        if 'feature_importance' in results:
            for method, importance in results['feature_importance'].items():
                mean_importance = importance.mean().item()
                summary_text += f"{method.title()}: {mean_importance:.4f}\n"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'interpretability_comparison_class_{target_class}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

# Sample Model for Testing
class InterpretableTestCNN(nn.Module):
    """Simple CNN for interpretability testing"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("Model Interpretability Analysis")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and sample data
    model = InterpretableTestCNN(num_classes=10).to(device)
    
    # Sample input image
    input_image = torch.randn(1, 3, 64, 64).to(device)
    
    # Background data for SHAP
    background_data = torch.randn(20, 3, 64, 64).to(device)
    
    print("\n1. LIME Explanation")
    print("-" * 22)
    
    # LIME analysis
    lime_explainer = LIMEImageExplainer(model, device)
    print("Generating LIME explanation...")
    
    lime_explanation = lime_explainer.explain_instance(
        input_image, target_class=None, num_samples=500, num_features=64
    )
    
    print(f"LIME Results:")
    print(f"  Target class: {lime_explanation['target_class']}")
    print(f"  Original probability: {lime_explanation['original_prob']:.4f}")
    print(f"  R² score: {lime_explanation['r2_score']:.4f}")
    print(f"  Number of features: {len(lime_explanation['feature_importance'])}")
    
    lime_explainer.visualize_explanation(input_image, lime_explanation)
    
    print("\n2. SHAP Analysis")
    print("-" * 18)
    
    # SHAP analysis
    shap_explainer = SHAPDeepExplainer(model, background_data[:5], device)
    print("Computing SHAP values...")
    
    shap_values = shap_explainer.shap_values(
        input_image, target_class=lime_explanation['target_class'], n_samples=30
    )
    
    print(f"SHAP Results:")
    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Mean absolute SHAP value: {shap_values.abs().mean():.6f}")
    print(f"  SHAP value range: [{shap_values.min():.6f}, {shap_values.max():.6f}]")
    
    shap_explainer.visualize_shap(input_image, shap_values, lime_explanation['target_class'])
    
    print("\n3. Feature Importance Analysis")
    print("-" * 35)
    
    # Feature importance analysis
    feature_analyzer = FeatureImportanceAnalyzer(model, device)
    print("Computing feature importance using multiple methods...")
    
    importance_results, target_class = feature_analyzer.analyze_input_importance(
        input_image, 
        target_class=lime_explanation['target_class'],
        methods=['gradient', 'integrated_gradient', 'occlusion']
    )
    
    print(f"Feature Importance Results:")
    for method, importance in importance_results.items():
        print(f"  {method}:")
        print(f"    Shape: {importance.shape}")
        print(f"    Mean: {importance.mean():.6f}")
        print(f"    Std: {importance.std():.6f}")
        print(f"    Range: [{importance.min():.6f}, {importance.max():.6f}]")
    
    feature_analyzer.visualize_feature_importance(input_image, importance_results, target_class)
    
    print("\n4. Permutation Importance")
    print("-" * 30)
    
    # Create a simple dataset for permutation importance
    sample_data = torch.randn(100, 3, 64, 64)
    sample_labels = torch.randint(0, 10, (100,))
    sample_dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=16, shuffle=False)
    
    perm_importance = PermutationImportance(model, device)
    print("Computing permutation importance...")
    
    layer_importance = perm_importance.compute_importance(sample_loader, n_repeats=3)
    
    print("Layer Importance Scores:")
    for layer, score in layer_importance.items():
        print(f"  {layer}: {score:.6f}")
    
    perm_importance.visualize_importance(layer_importance)
    
    print("\n5. Comprehensive Interpretability Analysis")
    print("-" * 45)
    
    # Comprehensive analysis
    comparison = ModelInterpretabilityComparison(model, device)
    print("Performing comprehensive interpretability analysis...")
    
    comprehensive_results, final_target_class, final_confidence = comparison.comprehensive_analysis(
        input_image, background_data, target_class=None
    )
    
    print(f"\nComprehensive Analysis Results:")
    print(f"  Target class: {final_target_class}")
    print(f"  Model confidence: {final_confidence:.4f}")
    
    # Print method-specific results
    if 'lime' in comprehensive_results:
        lime_r2 = comprehensive_results['lime']['r2_score']
        print(f"  LIME R² score: {lime_r2:.4f}")
    
    if 'shap' in comprehensive_results:
        shap_values = comprehensive_results['shap']
        shap_mean = shap_values.abs().mean().item()
        print(f"  SHAP mean |value|: {shap_mean:.6f}")
    
    if 'feature_importance' in comprehensive_results:
        feature_results = comprehensive_results['feature_importance']
        for method, importance in feature_results.items():
            mean_imp = importance.mean().item()
            print(f"  {method} mean importance: {mean_imp:.6f}")
    
    # Visualize comprehensive comparison
    comparison.visualize_comparison(
        input_image, comprehensive_results, final_target_class, final_confidence
    )
    
    print("\n6. Method Reliability Analysis")
    print("-" * 35)
    
    # Test interpretability methods on multiple samples
    print("Testing method consistency across multiple samples...")
    
    # Generate multiple samples
    test_samples = torch.randn(5, 3, 64, 64).to(device)
    
    method_consistency = {
        'lime_r2_scores': [],
        'shap_means': [],
        'gradient_means': []
    }
    
    for i, test_sample in enumerate(test_samples):
        print(f"  Processing sample {i+1}/5...")
        
        # Quick LIME analysis
        lime_result = lime_explainer.explain_instance(
            test_sample.unsqueeze(0), num_samples=200, num_features=36
        )
        method_consistency['lime_r2_scores'].append(lime_result['r2_score'])
        
        # Quick gradient analysis
        test_sample_grad = test_sample.unsqueeze(0).clone().requires_grad_(True)
        output = model(test_sample_grad)
        target_class = output.argmax(dim=1).item()
        
        model.zero_grad()
        score = output[:, target_class]
        score.backward()
        
        grad_importance = test_sample_grad.grad.abs().sum(dim=1).squeeze().cpu()
        method_consistency['gradient_means'].append(grad_importance.mean().item())
        
        # Quick SHAP analysis (simplified)
        try:
            shap_vals = shap_explainer.shap_values(
                test_sample.unsqueeze(0), target_class, n_samples=10
            )
            method_consistency['shap_means'].append(shap_vals.abs().mean().item())
        except:
            method_consistency['shap_means'].append(0.0)
    
    # Analyze consistency
    print("\nMethod Consistency Analysis:")
    for method, values in method_consistency.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else float('inf')
            print(f"  {method}: mean={mean_val:.4f}, std={std_val:.4f}, CV={cv:.4f}")
    
    print("\nModel interpretability analysis completed!")
    print("Generated files:")
    print("  - lime_explanation_class_*.png")
    print("  - shap_explanation_class_*.png")
    print("  - feature_importance_analysis_class_*.png")
    print("  - permutation_importance.png")
    print("  - interpretability_comparison_class_*.png")