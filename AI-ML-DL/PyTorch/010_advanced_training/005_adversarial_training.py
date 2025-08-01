import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Tuple, Dict
import time

# Adversarial Attack Methods
class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) attack"""
    
    def __init__(self, epsilon=0.3):
        self.epsilon = epsilon
    
    def generate(self, model, data, targets, device='cuda'):
        """Generate FGSM adversarial examples"""
        data = data.to(device).requires_grad_(True)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = F.cross_entropy(outputs, targets)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = data.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = data + self.epsilon * sign_data_grad
        
        # Clamp to valid range [0, 1]
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        return perturbed_data.detach()

class PGDAttack:
    """Projected Gradient Descent (PGD) attack"""
    
    def __init__(self, epsilon=0.3, alpha=0.01, num_steps=40):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def generate(self, model, data, targets, device='cuda'):
        """Generate PGD adversarial examples"""
        data = data.to(device)
        targets = targets.to(device)
        
        # Start with random perturbation
        perturbed_data = data + torch.empty_like(data).uniform_(-self.epsilon, self.epsilon)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        
        for _ in range(self.num_steps):
            perturbed_data.requires_grad_(True)
            
            # Forward pass
            outputs = model(perturbed_data)
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + self.alpha * data_grad.sign()
            
            # Project back to epsilon ball
            eta = torch.clamp(perturbed_data - data, -self.epsilon, self.epsilon)
            perturbed_data = torch.clamp(data + eta, 0, 1).detach()
        
        return perturbed_data

class CWAttack:
    """Carlini & Wagner (C&W) attack"""
    
    def __init__(self, confidence=0, learning_rate=0.01, max_iterations=1000):
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def generate(self, model, data, targets, device='cuda'):
        """Generate C&W adversarial examples"""
        data = data.to(device)
        targets = targets.to(device)
        batch_size = data.size(0)
        
        # Initialize perturbation
        w = torch.zeros_like(data, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        best_adv = data.clone()
        best_l2 = float('inf') * torch.ones(batch_size, device=device)
        
        for iteration in range(self.max_iterations):
            # Convert w to adversarial example
            adv_x = 0.5 * (torch.tanh(w) + 1)
            
            # Compute loss
            outputs = model(adv_x)
            
            # C&W loss function
            real = torch.sum(outputs * targets, dim=1)
            other = torch.max((1 - targets) * outputs - targets * 10000, dim=1)[0]
            
            loss1 = torch.clamp(real - other + self.confidence, min=0)
            loss2 = torch.sum((adv_x - data) ** 2, dim=(1, 2, 3))
            
            loss = loss1 + loss2
            total_loss = torch.sum(loss)
            
            # Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update best adversarial examples
            l2_dist = torch.sum((adv_x - data) ** 2, dim=(1, 2, 3))
            mask = l2_dist < best_l2
            best_l2[mask] = l2_dist[mask]
            best_adv[mask] = adv_x[mask].detach()
        
        return best_adv

# Adversarial Training Methods
class AdversarialTrainer:
    """Adversarial training with various attack methods"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 attack_method='FGSM', attack_params=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Initialize attack method
        if attack_params is None:
            attack_params = {}
        
        if attack_method == 'FGSM':
            self.attack = FGSMAttack(**attack_params)
        elif attack_method == 'PGD':
            self.attack = PGDAttack(**attack_params)
        elif attack_method == 'CW':
            self.attack = CWAttack(**attack_params)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        self.attack_method = attack_method
        
        # Training statistics
        self.stats = {
            'clean_accuracy': [],
            'adv_accuracy': [],
            'training_loss': []
        }
    
    def train_step(self, data, targets, adv_ratio=0.5):
        """Single adversarial training step"""
        data, targets = data.to(self.device), targets.to(self.device)
        batch_size = data.size(0)
        
        # Split batch into clean and adversarial examples
        adv_size = int(batch_size * adv_ratio)
        clean_size = batch_size - adv_size
        
        # Generate adversarial examples
        if adv_size > 0:
            adv_data = self.attack.generate(self.model, data[:adv_size], targets[:adv_size], self.device)
            
            # Combine clean and adversarial data
            combined_data = torch.cat([data[adv_size:], adv_data], dim=0)
            combined_targets = targets
        else:
            combined_data = data
            combined_targets = targets
        
        # Training step
        self.optimizer.zero_grad()
        outputs = self.model(combined_data)
        loss = self.criterion(outputs, combined_targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate_robustness(self, test_loader, attack_epsilon=0.3):
        """Evaluate model robustness against adversarial attacks"""
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        # Create attack for evaluation
        eval_attack = FGSMAttack(epsilon=attack_epsilon)
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Clean accuracy
                clean_outputs = self.model(data)
                _, clean_pred = torch.max(clean_outputs, 1)
                clean_correct += (clean_pred == targets).sum().item()
                
                # Adversarial accuracy
                adv_data = eval_attack.generate(self.model, data, targets, self.device)
                adv_outputs = self.model(adv_data)
                _, adv_pred = torch.max(adv_outputs, 1)
                adv_correct += (adv_pred == targets).sum().item()
                
                total += targets.size(0)
        
        clean_acc = 100. * clean_correct / total
        adv_acc = 100. * adv_correct / total
        
        return clean_acc, adv_acc
    
    def train_epoch(self, dataloader, adv_ratio=0.5):
        """Train for one epoch with adversarial training"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            loss = self.train_step(data, targets, adv_ratio)
            total_loss += loss
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss:.4f}')
        
        avg_loss = total_loss / len(dataloader)
        self.stats['training_loss'].append(avg_loss)
        
        return avg_loss

# TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
class TRADESTrainer:
    """TRADES adversarial training method"""
    
    def __init__(self, model, optimizer, device='cuda', beta=6.0, epsilon=0.3, step_size=0.007, num_steps=10):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.beta = beta  # Regularization parameter
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
    
    def trades_loss(self, data, targets):
        """Compute TRADES loss"""
        # Natural loss
        logits = self.model(data)
        natural_loss = F.cross_entropy(logits, targets)
        
        # Generate adversarial examples using PGD
        adv_data = self._generate_adversarial(data)
        
        # Robust loss (KL divergence between natural and adversarial predictions)
        logits_adv = self.model(adv_data)
        
        robust_loss = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits, dim=1),
            reduction='batchmean'
        )
        
        # Combined loss
        total_loss = natural_loss + self.beta * robust_loss
        
        return total_loss, natural_loss, robust_loss
    
    def _generate_adversarial(self, data):
        """Generate adversarial examples for TRADES"""
        # Initialize perturbation
        delta = torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon)
        delta = torch.clamp(delta, 0 - data, 1 - data)
        delta.requires_grad_(True)
        
        for _ in range(self.num_steps):
            adv_data = data + delta
            
            # Forward pass
            logits_clean = self.model(data)
            logits_adv = self.model(adv_data)
            
            # KL divergence loss for adversarial generation
            loss = F.kl_div(
                F.log_softmax(logits_adv, dim=1),
                F.softmax(logits_clean, dim=1),
                reduction='batchmean'
            )
            
            # Compute gradients
            loss.backward()
            
            # Update perturbation
            grad = delta.grad.detach()
            delta.data = delta.data + self.step_size * grad.sign()
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            delta.data = torch.clamp(data + delta.data, 0, 1) - data
            
            delta.grad.zero_()
        
        return (data + delta).detach()
    
    def train_step(self, data, targets):
        """Single TRADES training step"""
        data, targets = data.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Compute TRADES loss
        total_loss, natural_loss, robust_loss = self.trades_loss(data, targets)
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), natural_loss.item(), robust_loss.item()

# Adversarial Regularization
class AdversarialRegularization:
    """Adversarial regularization techniques"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def virtual_adversarial_loss(self, data, epsilon=1.0, xi=1e-6, ip=1):
        """Virtual Adversarial Training (VAT) loss"""
        # Generate virtual adversarial perturbation
        d = torch.randn_like(data)
        d = self._l2_normalize(d)
        
        for _ in range(ip):
            d.requires_grad_(True)
            
            # Predict with perturbed input
            logits_p = self.model(data + xi * d)
            logits = self.model(data)
            
            # KL divergence
            kl_loss = F.kl_div(
                F.log_softmax(logits_p, dim=1),
                F.softmax(logits.detach(), dim=1),
                reduction='batchmean'
            )
            
            # Compute gradient
            kl_loss.backward()
            d = d.grad.detach()
            d = self._l2_normalize(d)
        
        # Final VAT loss
        r_vadv = epsilon * d
        logits_p = self.model(data + r_vadv)
        
        vat_loss = F.kl_div(
            F.log_softmax(logits_p, dim=1),
            F.softmax(logits.detach(), dim=1),
            reduction='batchmean'
        )
        
        return vat_loss
    
    def _l2_normalize(self, d):
        """L2 normalize perturbation"""
        d_reshaped = d.view(d.size(0), -1)
        d_norm = torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        d_reshaped = d_reshaped / d_norm
        return d_reshaped.view_as(d)

# Sample Models
class RobustCNN(nn.Module):
    """CNN designed for adversarial robustness"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SampleDataset(Dataset):
    """Sample dataset for adversarial training"""
    
    def __init__(self, size=1000, input_shape=(1, 28, 28), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Evaluation Functions
def evaluate_attack_success_rate(model, dataloader, attack, device='cuda'):
    """Evaluate attack success rate"""
    model.eval()
    total = 0
    successful_attacks = 0
    
    for data, targets in dataloader:
        data, targets = data.to(device), targets.to(device)
        
        # Generate adversarial examples
        adv_data = attack.generate(model, data, targets, device)
        
        # Check if attack succeeded (prediction changed)
        with torch.no_grad():
            clean_pred = torch.argmax(model(data), dim=1)
            adv_pred = torch.argmax(model(adv_data), dim=1)
            
            successful_attacks += (clean_pred != adv_pred).sum().item()
            total += data.size(0)
    
    success_rate = 100. * successful_attacks / total
    return success_rate

def compare_training_methods(dataset, model_class, methods, epochs=5):
    """Compare different adversarial training methods"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for method_name, method_config in methods.items():
        print(f"\nTraining with {method_name}...")
        
        # Create model and data
        model = model_class().to(device)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        if method_name == 'Standard':
            # Standard training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                for data, targets in dataloader:
                    data, targets = data.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
        
        elif method_name == 'FGSM':
            # FGSM adversarial training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = AdversarialTrainer(model, optimizer, nn.CrossEntropyLoss(), device, 'FGSM')
            
            for epoch in range(epochs):
                trainer.train_epoch(dataloader, adv_ratio=0.5)
        
        elif method_name == 'TRADES':
            # TRADES training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = TRADESTrainer(model, optimizer, device)
            
            for epoch in range(epochs):
                for data, targets in dataloader:
                    trainer.train_step(data, targets)
        
        # Evaluate robustness
        attack = FGSMAttack(epsilon=0.3)
        success_rate = evaluate_attack_success_rate(model, dataloader, attack, device)
        
        results[method_name] = {
            'attack_success_rate': success_rate,
            'model': model
        }
        
        print(f"{method_name} - Attack Success Rate: {success_rate:.2f}%")
    
    return results

if __name__ == "__main__":
    print("Adversarial Training Techniques")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    dataset = SampleDataset(size=200, input_shape=(1, 28, 28), num_classes=10)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Test different attacks
    print("\n1. Testing Attack Methods")
    print("-" * 30)
    
    model = RobustCNN(num_classes=10).to(device)
    
    # Test FGSM
    fgsm = FGSMAttack(epsilon=0.3)
    pgd = PGDAttack(epsilon=0.3, alpha=0.01, num_steps=10)
    
    # Get a sample batch
    data, targets = next(iter(dataloader))
    
    # Generate adversarial examples
    print("Generating adversarial examples...")
    adv_fgsm = fgsm.generate(model, data, targets, device)
    adv_pgd = pgd.generate(model, data, targets, device)
    
    print(f"FGSM perturbation norm: {torch.norm(adv_fgsm - data.to(device)).item():.4f}")
    print(f"PGD perturbation norm: {torch.norm(adv_pgd - data.to(device)).item():.4f}")
    
    # Test adversarial training
    print("\n2. Testing Adversarial Training")
    print("-" * 30)
    
    model = RobustCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = AdversarialTrainer(model, optimizer, nn.CrossEntropyLoss(), device, 'FGSM')
    
    # Train for a few epochs
    for epoch in range(3):
        avg_loss = trainer.train_epoch(dataloader, adv_ratio=0.5)
        print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}")
    
    # Test TRADES
    print("\n3. Testing TRADES Training")
    print("-" * 30)
    
    model = RobustCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trades_trainer = TRADESTrainer(model, optimizer, device, beta=6.0)
    
    # Train for a few steps
    for batch_idx, (data, targets) in enumerate(dataloader):
        if batch_idx >= 3:
            break
        
        total_loss, natural_loss, robust_loss = trades_trainer.train_step(data, targets)
        print(f"Batch {batch_idx + 1}: Total={total_loss:.4f}, Natural={natural_loss:.4f}, Robust={robust_loss:.4f}")
    
    # Test Virtual Adversarial Training
    print("\n4. Testing Virtual Adversarial Training")
    print("-" * 30)
    
    model = RobustCNN(num_classes=10).to(device)
    adv_reg = AdversarialRegularization(model, device)
    
    # Compute VAT loss on sample
    data, targets = next(iter(dataloader))
    data = data.to(device)
    
    vat_loss = adv_reg.virtual_adversarial_loss(data, epsilon=1.0)
    print(f"VAT Loss: {vat_loss.item():.4f}")
    
    # Compare training methods
    print("\n5. Comparing Training Methods")
    print("-" * 30)
    
    methods = {
        'Standard': {},
        'FGSM': {},
        'TRADES': {}
    }
    
    small_dataset = SampleDataset(size=50, input_shape=(1, 28, 28), num_classes=10)
    
    comparison_results = compare_training_methods(
        small_dataset, 
        lambda: RobustCNN(num_classes=10), 
        methods, 
        epochs=2
    )
    
    print("\nTraining Method Comparison:")
    for method, results in comparison_results.items():
        print(f"{method}: Attack Success Rate = {results['attack_success_rate']:.2f}%")
    
    print("\nAdversarial training demonstrations completed!") 