import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import cv2

# Sample Models for Adversarial Testing
class AdversarialTestCNN(nn.Module):
    """CNN model for adversarial example testing"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
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
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class RobustCNN(nn.Module):
    """CNN with defensive mechanisms"""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Adversarial Attack Methods
class FGSM:
    """Fast Gradient Sign Method"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor, 
               epsilon: float = 0.03) -> torch.Tensor:
        """Generate FGSM adversarial examples"""
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Require gradients
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Get gradients
        data_grad = images.grad.data
        
        # Generate adversarial examples
        sign_data_grad = data_grad.sign()
        perturbed_images = images + epsilon * sign_data_grad
        
        # Clamp to valid image range
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def targeted_attack(self, images: torch.Tensor, target_labels: torch.Tensor,
                       epsilon: float = 0.03) -> torch.Tensor:
        """Generate targeted FGSM adversarial examples"""
        
        images = images.clone().detach().to(self.device)
        target_labels = target_labels.clone().detach().to(self.device)
        
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, target_labels)
        
        self.model.zero_grad()
        loss.backward()
        
        data_grad = images.grad.data
        
        # For targeted attack, subtract the gradient (minimize loss for target)
        sign_data_grad = data_grad.sign()
        perturbed_images = images - epsilon * sign_data_grad
        
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()

class PGD:
    """Projected Gradient Descent Attack"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               epsilon: float = 0.03, alpha: float = 0.01, 
               num_iter: int = 10) -> torch.Tensor:
        """Generate PGD adversarial examples"""
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize perturbation
        delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for i in range(num_iter):
            # Forward pass
            outputs = self.model(images + delta)
            
            # Calculate loss
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta.data = delta.data + alpha * grad.sign()
            
            # Project to epsilon ball
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            # Clamp to valid image range
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            
            # Zero gradients for next iteration
            delta.grad.zero_()
        
        return (images + delta).detach()
    
    def targeted_attack(self, images: torch.Tensor, target_labels: torch.Tensor,
                       epsilon: float = 0.03, alpha: float = 0.01,
                       num_iter: int = 10) -> torch.Tensor:
        """Generate targeted PGD adversarial examples"""
        
        images = images.clone().detach().to(self.device)
        target_labels = target_labels.clone().detach().to(self.device)
        
        delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        for i in range(num_iter):
            outputs = self.model(images + delta)
            loss = F.cross_entropy(outputs, target_labels)
            
            loss.backward()
            
            grad = delta.grad.detach()
            # For targeted attack, subtract gradient
            delta.data = delta.data - alpha * grad.sign()
            
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(images + delta.data, 0, 1) - images
            
            delta.grad.zero_()
        
        return (images + delta).detach()

class CarliniWagner:
    """Carlini & Wagner (C&W) Attack"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def attack(self, images: torch.Tensor, labels: torch.Tensor,
               c: float = 1.0, kappa: float = 0, num_iter: int = 100,
               learning_rate: float = 0.01) -> torch.Tensor:
        """Generate C&W adversarial examples"""
        
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # Initialize perturbation in tanh space
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        
        for i in range(num_iter):
            # Convert to valid image range using tanh
            perturbed_images = 0.5 * (torch.tanh(w) + 1)
            
            # Forward pass
            outputs = self.model(perturbed_images)
            
            # L2 distance loss
            l2_loss = torch.norm(perturbed_images - images, p=2, dim=[1, 2, 3]).sum()
            
            # Classification loss
            real_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze()
            other_logits = outputs.clone()
            other_logits.scatter_(1, labels.unsqueeze(1), -float('inf'))
            max_other_logits = other_logits.max(1)[0]
            
            # C&W loss: maximize (max_other - real + kappa)
            f_loss = torch.clamp(max_other_logits - real_logits + kappa, min=0).sum()
            
            # Total loss
            total_loss = l2_loss + c * f_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Final perturbed images
        with torch.no_grad():
            perturbed_images = 0.5 * (torch.tanh(w) + 1)
        
        return perturbed_images.detach()

class DeepFool:
    """DeepFool Attack (Simplified Implementation)"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def attack(self, image: torch.Tensor, num_classes: int = 10,
               max_iter: int = 50, overshoot: float = 0.02) -> torch.Tensor:
        """Generate DeepFool adversarial example for a single image"""
        
        image = image.clone().detach().to(self.device)
        image.requires_grad = True
        
        # Get original prediction
        output = self.model(image)
        original_class = output.argmax(dim=1).item()
        
        perturbed_image = image.clone()
        r_total = torch.zeros_like(image)
        
        for i in range(max_iter):
            # Get current prediction
            perturbed_image.requires_grad = True
            output = self.model(perturbed_image)
            current_class = output.argmax(dim=1).item()
            
            # If misclassified, stop
            if current_class != original_class:
                break
            
            # Calculate gradients for all classes
            gradients = []
            for k in range(num_classes):
                if k == original_class:
                    continue
                
                self.model.zero_grad()
                output[0, k].backward(retain_graph=True)
                grad = perturbed_image.grad.data.clone()
                gradients.append(grad)
                perturbed_image.grad.zero_()
            
            # Calculate gradient for original class
            self.model.zero_grad()
            output[0, original_class].backward(retain_graph=True)
            grad_orig = perturbed_image.grad.data.clone()
            
            # Find minimum perturbation
            min_norm = float('inf')
            min_perturbation = None
            
            for k, grad_k in enumerate(gradients):
                w_k = grad_k - grad_orig
                f_k = output[0, k] - output[0, original_class]
                
                norm_w_k = torch.norm(w_k.flatten())
                if norm_w_k > 0:
                    perturbation = abs(f_k) / (norm_w_k ** 2) * w_k
                    norm_perturbation = torch.norm(perturbation.flatten())
                    
                    if norm_perturbation < min_norm:
                        min_norm = norm_perturbation
                        min_perturbation = perturbation
            
            # Apply minimum perturbation with overshoot
            if min_perturbation is not None:
                r_total += (1 + overshoot) * min_perturbation
                perturbed_image = image + r_total
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
            else:
                break
        
        return perturbed_image.detach()

# Adversarial Defense Methods
class AdversarialTraining:
    """Adversarial training for robust models"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.fgsm = FGSM(model, device)
        self.pgd = PGD(model, device)
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   attack_type: str = 'fgsm', epsilon: float = 0.03,
                   alpha_adv: float = 0.5) -> Dict[str, float]:
        """Single adversarial training step"""
        
        self.model.train()
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            adv_images = self.fgsm.attack(images, labels, epsilon)
        elif attack_type == 'pgd':
            adv_images = self.pgd.attack(images, labels, epsilon)
        else:
            adv_images = images  # No attack
        
        # Combine clean and adversarial examples
        combined_images = torch.cat([images, adv_images], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        
        # Forward pass
        outputs = self.model(combined_images)
        loss = criterion(outputs, combined_labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracies
        with torch.no_grad():
            clean_outputs = self.model(images)
            adv_outputs = self.model(adv_images)
            
            clean_acc = (clean_outputs.argmax(1) == labels).float().mean().item()
            adv_acc = (adv_outputs.argmax(1) == labels).float().mean().item()
        
        return {
            'loss': loss.item(),
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc
        }

class InputTransformDefense:
    """Input transformation defense methods"""
    
    def __init__(self):
        pass
    
    def gaussian_noise_defense(self, images: torch.Tensor, 
                              noise_std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise as defense"""
        noise = torch.randn_like(images) * noise_std
        return torch.clamp(images + noise, 0, 1)
    
    def jpeg_compression_defense(self, images: torch.Tensor,
                                quality: int = 75) -> torch.Tensor:
        """JPEG compression defense (simplified)"""
        # This is a simplified version - actual JPEG compression would be more complex
        compressed = images.clone()
        
        # Add compression artifacts (simplified as noise)
        compression_noise = torch.randn_like(images) * 0.02
        compressed = torch.clamp(compressed + compression_noise, 0, 1)
        
        return compressed
    
    def bit_depth_reduction(self, images: torch.Tensor,
                           bits: int = 4) -> torch.Tensor:
        """Reduce bit depth as defense"""
        levels = 2 ** bits
        quantized = torch.round(images * (levels - 1)) / (levels - 1)
        return quantized

# Adversarial Example Analysis
class AdversarialAnalyzer:
    """Analyze adversarial examples and model robustness"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Initialize attack methods
        self.fgsm = FGSM(model, device)
        self.pgd = PGD(model, device)
        self.cw = CarliniWagner(model, device)
        self.deepfool = DeepFool(model, device)
    
    def evaluate_robustness(self, data_loader, epsilons: List[float] = None,
                           attacks: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate model robustness against multiple attacks"""
        
        if epsilons is None:
            epsilons = [0.01, 0.03, 0.05, 0.1]
        
        if attacks is None:
            attacks = ['fgsm', 'pgd']
        
        results = {}
        
        for attack_name in attacks:
            results[attack_name] = {}
            
            for epsilon in epsilons:
                print(f"Evaluating {attack_name} with epsilon={epsilon}")
                
                total_correct = 0
                total_samples = 0
                
                self.model.eval()
                with torch.no_grad():
                    for batch_idx, (data, targets) in enumerate(data_loader):
                        if batch_idx >= 10:  # Limit for efficiency
                            break
                        
                        data, targets = data.to(self.device), targets.to(self.device)
                        
                        # Generate adversarial examples
                        if attack_name == 'fgsm':
                            adv_data = self.fgsm.attack(data, targets, epsilon)
                        elif attack_name == 'pgd':
                            adv_data = self.pgd.attack(data, targets, epsilon)
                        else:
                            continue
                        
                        # Evaluate on adversarial examples
                        outputs = self.model(adv_data)
                        predictions = outputs.argmax(dim=1)
                        
                        correct = (predictions == targets).sum().item()
                        total_correct += correct
                        total_samples += targets.size(0)
                
                accuracy = total_correct / total_samples if total_samples > 0 else 0.0
                results[attack_name][epsilon] = accuracy
        
        return results
    
    def visualize_adversarial_examples(self, images: torch.Tensor, labels: torch.Tensor,
                                     epsilon: float = 0.03):
        """Visualize adversarial examples from different attacks"""
        
        # Take first image
        image = images[0:1]
        label = labels[0:1]
        
        # Generate adversarial examples
        fgsm_adv = self.fgsm.attack(image, label, epsilon)
        pgd_adv = self.pgd.attack(image, label, epsilon)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            orig_pred = self.model(image).argmax(dim=1).item()
            fgsm_pred = self.model(fgsm_adv).argmax(dim=1).item()
            pgd_pred = self.model(pgd_adv).argmax(dim=1).item()
        
        # Prepare images for visualization
        def prep_image(img_tensor):
            img = img_tensor[0].cpu().detach()
            # Denormalize if needed
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            return img_denorm.permute(1, 2, 0).numpy()
        
        orig_img = prep_image(image)
        fgsm_img = prep_image(fgsm_adv)
        pgd_img = prep_image(pgd_adv)
        
        # Calculate perturbations
        fgsm_pert = np.abs(fgsm_img - orig_img)
        pgd_pert = np.abs(pgd_img - orig_img)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original images
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title(f'Original\nPred: {orig_pred}, True: {label.item()}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fgsm_img)
        axes[0, 1].set_title(f'FGSM (ε={epsilon})\nPred: {fgsm_pred}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pgd_img)
        axes[0, 2].set_title(f'PGD (ε={epsilon})\nPred: {pgd_pred}')
        axes[0, 2].axis('off')
        
        # Perturbations
        axes[1, 0].axis('off')
        
        im1 = axes[1, 1].imshow(fgsm_pert, cmap='hot')
        axes[1, 1].set_title(f'FGSM Perturbation\nL2: {np.linalg.norm(fgsm_pert):.3f}')
        axes[1, 1].axis('off')
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        im2 = axes[1, 2].imshow(pgd_pert, cmap='hot')
        axes[1, 2].set_title(f'PGD Perturbation\nL2: {np.linalg.norm(pgd_pert):.3f}')
        axes[1, 2].axis('off')
        plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'adversarial_examples_eps_{epsilon}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return {
            'original': {'image': orig_img, 'prediction': orig_pred},
            'fgsm': {'image': fgsm_img, 'prediction': fgsm_pred, 'perturbation': fgsm_pert},
            'pgd': {'image': pgd_img, 'prediction': pgd_pred, 'perturbation': pgd_pert}
        }
    
    def plot_robustness_curve(self, robustness_results: Dict[str, Dict[str, float]]):
        """Plot robustness curves for different attacks"""
        
        plt.figure(figsize=(10, 6))
        
        for attack_name, results in robustness_results.items():
            epsilons = list(results.keys())
            accuracies = list(results.values())
            
            plt.plot(epsilons, accuracies, 'o-', label=attack_name.upper(), linewidth=2, markersize=6)
        
        plt.xlabel('Perturbation Budget (ε)')
        plt.ylabel('Accuracy')
        plt.title('Model Robustness vs Perturbation Budget')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('robustness_curve.png', dpi=150, bbox_inches='tight')
        plt.show()

class AdversarialDetector:
    """Detect adversarial examples"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def prediction_confidence_detector(self, images: torch.Tensor,
                                     threshold: float = 0.9) -> torch.Tensor:
        """Detect adversarial examples based on prediction confidence"""
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs = probs.max(dim=1)[0]
            
            # Images with low confidence are flagged as adversarial
            is_adversarial = max_probs < threshold
        
        return is_adversarial
    
    def reconstruction_detector(self, images: torch.Tensor,
                              noise_std: float = 0.1,
                              threshold: float = 0.1) -> torch.Tensor:
        """Detect adversarial examples using reconstruction error"""
        
        # Add noise and denoise (simplified)
        noisy_images = images + torch.randn_like(images) * noise_std
        noisy_images = torch.clamp(noisy_images, 0, 1)
        
        # Calculate reconstruction error
        reconstruction_error = torch.norm(noisy_images - images, p=2, dim=[1, 2, 3])
        
        # Normalize by image size
        reconstruction_error /= np.prod(images.shape[1:])
        
        # Flag high reconstruction error as adversarial
        is_adversarial = reconstruction_error > threshold
        
        return is_adversarial

if __name__ == "__main__":
    print("Adversarial Examples Analysis")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    model = AdversarialTestCNN(num_classes=10).to(device)
    robust_model = RobustCNN(num_classes=10).to(device)
    
    # Sample data
    sample_images = torch.randn(4, 3, 64, 64).to(device)
    sample_labels = torch.randint(0, 10, (4,)).to(device)
    
    print("\n1. FGSM Attack")
    print("-" * 18)
    
    # FGSM Attack
    fgsm_attacker = FGSM(model, device)
    
    print("Generating FGSM adversarial examples...")
    fgsm_adv = fgsm_attacker.attack(sample_images, sample_labels, epsilon=0.03)
    
    # Evaluate attack success
    model.eval()
    with torch.no_grad():
        orig_outputs = model(sample_images)
        adv_outputs = model(fgsm_adv)
        
        orig_preds = orig_outputs.argmax(dim=1)
        adv_preds = adv_outputs.argmax(dim=1)
        
        success_rate = (orig_preds != adv_preds).float().mean().item()
    
    print(f"FGSM Attack Results:")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Original predictions: {orig_preds.cpu().tolist()}")
    print(f"  Adversarial predictions: {adv_preds.cpu().tolist()}")
    print(f"  True labels: {sample_labels.cpu().tolist()}")
    
    # Calculate perturbation statistics
    perturbation = (fgsm_adv - sample_images).abs()
    print(f"  L∞ perturbation: {perturbation.max().item():.4f}")
    print(f"  L2 perturbation: {perturbation.norm(p=2).item():.4f}")
    
    print("\n2. PGD Attack")
    print("-" * 16)
    
    # PGD Attack
    pgd_attacker = PGD(model, device)
    
    print("Generating PGD adversarial examples...")
    pgd_adv = pgd_attacker.attack(sample_images, sample_labels, epsilon=0.03, num_iter=10)
    
    with torch.no_grad():
        pgd_outputs = model(pgd_adv)
        pgd_preds = pgd_outputs.argmax(dim=1)
        pgd_success_rate = (orig_preds != pgd_preds).float().mean().item()
    
    print(f"PGD Attack Results:")
    print(f"  Success rate: {pgd_success_rate:.2%}")
    print(f"  Adversarial predictions: {pgd_preds.cpu().tolist()}")
    
    pgd_perturbation = (pgd_adv - sample_images).abs()
    print(f"  L∞ perturbation: {pgd_perturbation.max().item():.4f}")
    print(f"  L2 perturbation: {pgd_perturbation.norm(p=2).item():.4f}")
    
    print("\n3. Carlini & Wagner Attack")
    print("-" * 32)
    
    # C&W Attack
    cw_attacker = CarliniWagner(model, device)
    
    print("Generating C&W adversarial examples...")
    cw_adv = cw_attacker.attack(sample_images[:2], sample_labels[:2], num_iter=50)  # Use fewer samples
    
    with torch.no_grad():
        cw_outputs = model(cw_adv)
        cw_preds = cw_outputs.argmax(dim=1)
        cw_success_rate = (orig_preds[:2] != cw_preds).float().mean().item()
    
    print(f"C&W Attack Results:")
    print(f"  Success rate: {cw_success_rate:.2%}")
    print(f"  Adversarial predictions: {cw_preds.cpu().tolist()}")
    
    cw_perturbation = (cw_adv - sample_images[:2]).abs()
    print(f"  L∞ perturbation: {cw_perturbation.max().item():.4f}")
    print(f"  L2 perturbation: {cw_perturbation.norm(p=2).item():.4f}")
    
    print("\n4. DeepFool Attack")
    print("-" * 20)
    
    # DeepFool Attack
    deepfool_attacker = DeepFool(model, device)
    
    print("Generating DeepFool adversarial example...")
    single_image = sample_images[0:1]
    single_label = sample_labels[0:1]
    
    deepfool_adv = deepfool_attacker.attack(single_image, num_classes=10, max_iter=20)
    
    with torch.no_grad():
        orig_output = model(single_image)
        deepfool_output = model(deepfool_adv)
        
        orig_pred = orig_output.argmax(dim=1).item()
        deepfool_pred = deepfool_output.argmax(dim=1).item()
    
    print(f"DeepFool Attack Results:")
    print(f"  Original prediction: {orig_pred}")
    print(f"  Adversarial prediction: {deepfool_pred}")
    print(f"  Attack successful: {orig_pred != deepfool_pred}")
    
    deepfool_perturbation = (deepfool_adv - single_image).abs()
    print(f"  L∞ perturbation: {deepfool_perturbation.max().item():.4f}")
    print(f"  L2 perturbation: {deepfool_perturbation.norm(p=2).item():.4f}")
    
    print("\n5. Adversarial Training")
    print("-" * 25)
    
    # Adversarial Training
    adv_trainer = AdversarialTraining(robust_model, device)
    
    # Create simple optimizer and criterion
    optimizer = torch.optim.Adam(robust_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Demonstrating adversarial training step...")
    
    training_results = adv_trainer.train_step(
        sample_images, sample_labels, optimizer, criterion,
        attack_type='fgsm', epsilon=0.03
    )
    
    print(f"Adversarial Training Results:")
    print(f"  Training loss: {training_results['loss']:.4f}")
    print(f"  Clean accuracy: {training_results['clean_accuracy']:.2%}")
    print(f"  Adversarial accuracy: {training_results['adversarial_accuracy']:.2%}")
    
    print("\n6. Defense Mechanisms")
    print("-" * 25)
    
    # Input Transformation Defenses
    defense = InputTransformDefense()
    
    print("Testing input transformation defenses...")
    
    # Test defenses against FGSM
    defended_images = {
        'gaussian_noise': defense.gaussian_noise_defense(fgsm_adv, noise_std=0.1),
        'bit_reduction': defense.bit_depth_reduction(fgsm_adv, bits=4),
        'jpeg_compression': defense.jpeg_compression_defense(fgsm_adv, quality=75)
    }
    
    print("Defense effectiveness against FGSM:")
    
    with torch.no_grad():
        for defense_name, defended_imgs in defended_images.items():
            defended_outputs = model(defended_imgs)
            defended_preds = defended_outputs.argmax(dim=1)
            
            # Check if defense restored correct predictions
            restoration_rate = (defended_preds == sample_labels).float().mean().item()
            print(f"  {defense_name}: {restoration_rate:.2%} restoration rate")
    
    print("\n7. Robustness Evaluation")
    print("-" * 30)
    
    # Create dataset for robustness evaluation
    eval_data = torch.randn(80, 3, 64, 64)
    eval_labels = torch.randint(0, 10, (80,))
    eval_dataset = torch.utils.data.TensorDataset(eval_data, eval_labels)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=False)
    
    # Analyze robustness
    analyzer = AdversarialAnalyzer(model, device)
    
    print("Evaluating model robustness...")
    robustness_results = analyzer.evaluate_robustness(
        eval_loader, 
        epsilons=[0.01, 0.03, 0.05, 0.1],
        attacks=['fgsm', 'pgd']
    )
    
    print("Robustness Results:")
    for attack, results in robustness_results.items():
        print(f"  {attack.upper()}:")
        for epsilon, accuracy in results.items():
            print(f"    ε={epsilon}: {accuracy:.2%} accuracy")
    
    # Plot robustness curve
    analyzer.plot_robustness_curve(robustness_results)
    
    print("\n8. Adversarial Example Visualization")
    print("-" * 40)
    
    # Visualize adversarial examples
    print("Creating adversarial example visualizations...")
    visualization_results = analyzer.visualize_adversarial_examples(
        sample_images, sample_labels, epsilon=0.05
    )
    
    print("Visualization Results:")
    for method, result in visualization_results.items():
        if 'prediction' in result:
            print(f"  {method}: prediction = {result['prediction']}")
        if 'perturbation' in result:
            pert_norm = np.linalg.norm(result['perturbation'])
            print(f"    perturbation L2 norm: {pert_norm:.4f}")
    
    print("\n9. Adversarial Detection")
    print("-" * 28)
    
    # Adversarial Detection
    detector = AdversarialDetector(model, device)
    
    print("Testing adversarial detection methods...")
    
    # Combine clean and adversarial images
    combined_images = torch.cat([sample_images, fgsm_adv], dim=0)
    true_labels = torch.cat([
        torch.zeros(len(sample_images)),  # Clean images
        torch.ones(len(fgsm_adv))         # Adversarial images
    ])
    
    # Test confidence-based detection
    confidence_detection = detector.prediction_confidence_detector(
        combined_images, threshold=0.8
    )
    
    # Test reconstruction-based detection
    reconstruction_detection = detector.reconstruction_detector(
        combined_images, threshold=0.05
    )
    
    print("Detection Results:")
    print(f"  Confidence-based detector:")
    conf_accuracy = (confidence_detection.float() == true_labels).float().mean()
    print(f"    Accuracy: {conf_accuracy:.2%}")
    
    print(f"  Reconstruction-based detector:")
    recon_accuracy = (reconstruction_detection.float() == true_labels).float().mean()
    print(f"    Accuracy: {recon_accuracy:.2%}")
    
    print("\n10. Attack Transferability")
    print("-" * 30)
    
    # Test attack transferability between models
    print("Testing attack transferability...")
    
    # Generate adversarial examples on original model
    transfer_adv = fgsm_attacker.attack(sample_images, sample_labels, epsilon=0.05)
    
    # Test on robust model
    with torch.no_grad():
        orig_preds_robust = robust_model(sample_images).argmax(dim=1)
        transfer_preds_robust = robust_model(transfer_adv).argmax(dim=1)
        
        transfer_success = (orig_preds_robust != transfer_preds_robust).float().mean().item()
    
    print(f"Transfer Attack Results:")
    print(f"  Success rate on target model: {transfer_success:.2%}")
    print(f"  Original model predictions: {orig_preds.cpu().tolist()}")
    print(f"  Robust model predictions (clean): {orig_preds_robust.cpu().tolist()}")
    print(f"  Robust model predictions (adversarial): {transfer_preds_robust.cpu().tolist()}")
    
    print("\nAdversarial examples analysis completed!")
    print("Generated files:")
    print("  - adversarial_examples_eps_*.png")
    print("  - robustness_curve.png")