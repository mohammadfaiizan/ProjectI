import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import math
from typing import List, Tuple, Callable, Optional, Dict
import random

# Curriculum Learning Dataset Wrapper
class CurriculumDataset(Dataset):
    """Dataset wrapper that supports curriculum learning"""
    
    def __init__(self, base_dataset, difficulty_scores=None, curriculum_strategy='linear'):
        self.base_dataset = base_dataset
        self.curriculum_strategy = curriculum_strategy
        
        # Calculate or use provided difficulty scores
        if difficulty_scores is None:
            self.difficulty_scores = self._calculate_difficulty_scores()
        else:
            self.difficulty_scores = difficulty_scores
        
        # Sort indices by difficulty
        self.sorted_indices = np.argsort(self.difficulty_scores)
        
        # Current curriculum parameters
        self.current_epoch = 0
        self.max_epochs = 100  # Will be set by trainer
        self.current_data_ratio = 0.1  # Start with 10% of easiest data
        
    def _calculate_difficulty_scores(self):
        """Calculate difficulty scores for each sample"""
        # Simple heuristic: use target class as difficulty (higher class = harder)
        # In practice, this would be more sophisticated
        difficulty_scores = []
        
        for i in range(len(self.base_dataset)):
            _, target = self.base_dataset[i]
            
            # Convert target to difficulty score
            if isinstance(target, torch.Tensor):
                difficulty = float(target)
            else:
                difficulty = float(target)
            
            difficulty_scores.append(difficulty)
        
        return np.array(difficulty_scores)
    
    def update_curriculum(self, epoch, total_epochs):
        """Update curriculum based on training progress"""
        self.current_epoch = epoch
        self.max_epochs = total_epochs
        
        # Update data ratio based on curriculum strategy
        progress = epoch / total_epochs
        
        if self.curriculum_strategy == 'linear':
            self.current_data_ratio = min(0.1 + 0.9 * progress, 1.0)
        elif self.curriculum_strategy == 'exponential':
            self.current_data_ratio = min(0.1 + 0.9 * (progress ** 2), 1.0)
        elif self.curriculum_strategy == 'root':
            self.current_data_ratio = min(0.1 + 0.9 * (progress ** 0.5), 1.0)
        elif self.curriculum_strategy == 'step':
            # Step function: 25% at 1/4, 50% at 1/2, 75% at 3/4, 100% at end
            if progress < 0.25:
                self.current_data_ratio = 0.25
            elif progress < 0.5:
                self.current_data_ratio = 0.5
            elif progress < 0.75:
                self.current_data_ratio = 0.75
            else:
                self.current_data_ratio = 1.0
        
        print(f"Epoch {epoch}: Using {self.current_data_ratio:.2%} of data")
    
    def get_current_indices(self):
        """Get indices of samples to use in current curriculum stage"""
        num_samples = int(len(self.base_dataset) * self.current_data_ratio)
        return self.sorted_indices[:num_samples]
    
    def __len__(self):
        return int(len(self.base_dataset) * self.current_data_ratio)
    
    def __getitem__(self, idx):
        # Map curriculum index to actual dataset index
        current_indices = self.get_current_indices()
        actual_idx = current_indices[idx]
        return self.base_dataset[actual_idx]

# Curriculum Sampler
class CurriculumSampler(Sampler):
    """Sampler that implements curriculum learning"""
    
    def __init__(self, dataset, curriculum_type='easy_first', batch_size=32):
        self.dataset = dataset
        self.curriculum_type = curriculum_type
        self.batch_size = batch_size
        
        # Get difficulty scores and indices
        if hasattr(dataset, 'difficulty_scores'):
            self.difficulty_scores = dataset.difficulty_scores
        else:
            self.difficulty_scores = self._estimate_difficulty()
        
        self.indices = list(range(len(dataset)))
        self.epoch = 0
    
    def _estimate_difficulty(self):
        """Estimate difficulty when not provided"""
        # Simple estimation based on data variance or other heuristics
        difficulty_scores = []
        
        for i in range(len(self.dataset)):
            data, target = self.dataset[i]
            
            # Use data variance as difficulty proxy
            if isinstance(data, torch.Tensor):
                difficulty = float(torch.var(data))
            else:
                difficulty = np.var(data)
            
            difficulty_scores.append(difficulty)
        
        return np.array(difficulty_scores)
    
    def set_epoch(self, epoch):
        """Set current epoch for curriculum progression"""
        self.epoch = epoch
    
    def __iter__(self):
        # Sort indices by difficulty
        sorted_pairs = sorted(zip(self.indices, self.difficulty_scores), 
                            key=lambda x: x[1])
        
        if self.curriculum_type == 'easy_first':
            # Start with easiest samples
            progress = min(self.epoch / 50.0, 1.0)  # Full curriculum by epoch 50
            num_samples = max(len(self.indices) // 10, 
                            int(len(self.indices) * progress))
            selected_indices = [pair[0] for pair in sorted_pairs[:num_samples]]
        
        elif self.curriculum_type == 'hard_first':
            # Start with hardest samples
            progress = min(self.epoch / 50.0, 1.0)
            num_samples = max(len(self.indices) // 10, 
                            int(len(self.indices) * progress))
            selected_indices = [pair[0] for pair in sorted_pairs[-num_samples:]]
        
        elif self.curriculum_type == 'mixed':
            # Mix easy and hard samples
            easy_ratio = max(0.7 - self.epoch * 0.01, 0.3)
            num_easy = int(len(self.indices) * easy_ratio)
            num_hard = len(self.indices) - num_easy
            
            easy_indices = [pair[0] for pair in sorted_pairs[:num_easy]]
            hard_indices = [pair[0] for pair in sorted_pairs[-num_hard:]]
            selected_indices = easy_indices + hard_indices
        
        else:  # random
            selected_indices = self.indices
        
        # Shuffle selected indices
        random.shuffle(selected_indices)
        return iter(selected_indices)
    
    def __len__(self):
        return len(self.indices)

# Self-Paced Learning
class SelfPacedLearner:
    """Self-paced learning implementation"""
    
    def __init__(self, model, criterion, device='cuda', lambda_sp=1.0):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.lambda_sp = lambda_sp  # Self-paced regularization parameter
        
        # Track sample weights
        self.sample_weights = None
        
    def calculate_sample_weights(self, losses):
        """Calculate sample weights based on self-paced learning"""
        # Weights are 1 for losses <= lambda, 0 otherwise
        weights = (losses <= self.lambda_sp).float()
        return weights
    
    def self_paced_loss(self, outputs, targets):
        """Compute self-paced loss"""
        batch_size = outputs.size(0)
        
        # Compute individual sample losses
        individual_losses = F.cross_entropy(outputs, targets, reduction='none')
        
        # Calculate sample weights
        weights = self.calculate_sample_weights(individual_losses)
        
        # Weighted loss
        weighted_loss = torch.sum(weights * individual_losses) / batch_size
        
        # Self-paced regularization term
        regularization = -self.lambda_sp * torch.sum(weights) / batch_size
        
        total_loss = weighted_loss + regularization
        
        # Store weights for analysis
        self.sample_weights = weights.detach().cpu().numpy()
        
        return total_loss, individual_losses.detach().cpu().numpy()
    
    def update_lambda(self, epoch, total_epochs, initial_lambda=0.1, final_lambda=2.0):
        """Update lambda parameter over training"""
        progress = epoch / total_epochs
        self.lambda_sp = initial_lambda + (final_lambda - initial_lambda) * progress
        
        print(f"Updated lambda to {self.lambda_sp:.3f}")

# Competence-Based Curriculum
class CompetenceBasedCurriculum:
    """Curriculum based on model competence"""
    
    def __init__(self, model, difficulty_estimator=None):
        self.model = model
        self.difficulty_estimator = difficulty_estimator
        
        # Track model competence over time
        self.competence_history = []
        self.current_competence = 0.0
        
    def estimate_competence(self, dataloader):
        """Estimate current model competence"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        competence = correct / total
        self.competence_history.append(competence)
        self.current_competence = competence
        
        return competence
    
    def select_samples(self, dataset, target_difficulty=None):
        """Select samples based on current competence"""
        if target_difficulty is None:
            # Set target difficulty based on competence
            target_difficulty = min(self.current_competence + 0.1, 1.0)
        
        selected_indices = []
        
        for i in range(len(dataset)):
            if hasattr(dataset, 'difficulty_scores'):
                sample_difficulty = dataset.difficulty_scores[i] / max(dataset.difficulty_scores)
            else:
                # Estimate difficulty on the fly
                sample_difficulty = random.random()  # Placeholder
            
            # Include sample if difficulty is appropriate
            if sample_difficulty <= target_difficulty:
                selected_indices.append(i)
        
        return selected_indices

# Curriculum Learning Trainer
class CurriculumTrainer:
    """Trainer with curriculum learning support"""
    
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 curriculum_type='linear', use_self_paced=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.curriculum_type = curriculum_type
        self.use_self_paced = use_self_paced
        
        # Initialize self-paced learner if needed
        if use_self_paced:
            self.self_paced = SelfPacedLearner(model, criterion, device)
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'epoch_accuracies': [],
            'curriculum_ratios': []
        }
    
    def train_epoch(self, curriculum_dataset, epoch, total_epochs):
        """Train for one epoch with curriculum learning"""
        self.model.train()
        
        # Update curriculum
        curriculum_dataset.update_curriculum(epoch, total_epochs)
        
        # Create dataloader for current curriculum
        dataloader = DataLoader(curriculum_dataset, batch_size=32, shuffle=True)
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            
            if self.use_self_paced:
                # Use self-paced learning
                loss, individual_losses = self.self_paced.self_paced_loss(outputs, targets)
                
                # Update lambda
                self.self_paced.update_lambda(epoch, total_epochs)
            else:
                # Standard loss
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total_samples
        
        # Store statistics
        self.training_stats['epoch_losses'].append(avg_loss)
        self.training_stats['epoch_accuracies'].append(accuracy)
        self.training_stats['curriculum_ratios'].append(curriculum_dataset.current_data_ratio)
        
        print(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# Sample Models and Data
class SimpleCNN(nn.Module):
    """Simple CNN for curriculum learning experiments"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SyntheticDataset(Dataset):
    """Synthetic dataset with controllable difficulty"""
    
    def __init__(self, size=1000, num_classes=10, difficulty_distribution='uniform'):
        self.size = size
        self.num_classes = num_classes
        
        # Generate data with different difficulty levels
        self.data = []
        self.targets = []
        self.difficulty_scores = []
        
        for i in range(size):
            # Generate difficulty score
            if difficulty_distribution == 'uniform':
                difficulty = random.random()
            elif difficulty_distribution == 'easy_bias':
                difficulty = random.random() ** 2  # Bias towards easier samples
            elif difficulty_distribution == 'hard_bias':
                difficulty = 1 - (random.random() ** 2)  # Bias towards harder samples
            else:
                difficulty = random.random()
            
            # Generate synthetic image (28x28)
            noise_level = difficulty * 0.5  # More noise = harder
            base_pattern = torch.randn(1, 28, 28) * 0.1
            noise = torch.randn(1, 28, 28) * noise_level
            image = base_pattern + noise
            
            # Generate target (higher targets are "harder")
            target = int(difficulty * (num_classes - 1))
            
            self.data.append(image)
            self.targets.append(target)
            self.difficulty_scores.append(difficulty)
        
        self.difficulty_scores = np.array(self.difficulty_scores)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Curriculum Analysis Tools
def analyze_curriculum_progress(trainer):
    """Analyze curriculum learning progress"""
    stats = trainer.training_stats
    
    print("\nCurriculum Learning Analysis")
    print("=" * 40)
    
    print(f"Final accuracy: {stats['epoch_accuracies'][-1]:.2f}%")
    print(f"Best accuracy: {max(stats['epoch_accuracies']):.2f}%")
    print(f"Final curriculum ratio: {stats['curriculum_ratios'][-1]:.2%}")
    
    # Calculate convergence metrics
    accuracy_improvement = stats['epoch_accuracies'][-1] - stats['epoch_accuracies'][0]
    print(f"Accuracy improvement: {accuracy_improvement:.2f}%")
    
    return stats

def compare_curriculum_strategies(dataset, model_class, strategies, epochs=10):
    """Compare different curriculum learning strategies"""
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for strategy in strategies:
        print(f"\nTraining with {strategy} curriculum...")
        
        # Create model and trainer
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Create curriculum dataset
        curriculum_dataset = CurriculumDataset(dataset, curriculum_strategy=strategy)
        
        # Train
        trainer = CurriculumTrainer(model, optimizer, criterion, device)
        
        for epoch in range(epochs):
            trainer.train_epoch(curriculum_dataset, epoch, epochs)
        
        # Store results
        final_accuracy = trainer.training_stats['epoch_accuracies'][-1]
        results[strategy] = {
            'final_accuracy': final_accuracy,
            'training_stats': trainer.training_stats
        }
        
        print(f"{strategy} final accuracy: {final_accuracy:.2f}%")
    
    return results

if __name__ == "__main__":
    print("Curriculum Learning Techniques")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset with difficulty labels
    print("\n1. Creating Synthetic Dataset")
    print("-" * 30)
    
    dataset = SyntheticDataset(size=500, num_classes=5, difficulty_distribution='uniform')
    print(f"Dataset size: {len(dataset)}")
    print(f"Difficulty score range: {dataset.difficulty_scores.min():.3f} - {dataset.difficulty_scores.max():.3f}")
    
    # Test curriculum dataset
    print("\n2. Testing Curriculum Dataset")
    print("-" * 30)
    
    curriculum_dataset = CurriculumDataset(dataset, curriculum_strategy='linear')
    
    # Simulate curriculum progression
    for epoch in [0, 10, 20, 30, 40]:
        curriculum_dataset.update_curriculum(epoch, 50)
        print(f"Epoch {epoch}: Dataset size = {len(curriculum_dataset)}")
    
    # Test curriculum sampler
    print("\n3. Testing Curriculum Sampler")
    print("-" * 30)
    
    sampler = CurriculumSampler(dataset, curriculum_type='easy_first')
    
    for epoch in [0, 10, 20]:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        difficulties = [dataset.difficulty_scores[i] for i in indices[:10]]
        avg_difficulty = np.mean(difficulties)
        print(f"Epoch {epoch}: Average difficulty of first 10 samples = {avg_difficulty:.3f}")
    
    # Test self-paced learning
    print("\n4. Testing Self-Paced Learning")
    print("-" * 30)
    
    model = SimpleCNN(num_classes=5).to(device)
    self_paced = SelfPacedLearner(model, nn.CrossEntropyLoss(), device)
    
    # Simulate some losses
    fake_outputs = torch.randn(8, 5).to(device)
    fake_targets = torch.randint(0, 5, (8,)).to(device)
    
    loss, individual_losses = self_paced.self_paced_loss(fake_outputs, fake_targets)
    print(f"Self-paced loss: {loss.item():.4f}")
    print(f"Sample weights: {self_paced.sample_weights}")
    
    # Test curriculum trainer
    print("\n5. Testing Curriculum Trainer")
    print("-" * 30)
    
    model = SimpleCNN(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = CurriculumTrainer(model, optimizer, criterion, device, curriculum_type='linear')
    
    # Train for a few epochs
    curriculum_dataset = CurriculumDataset(dataset, curriculum_strategy='linear')
    
    for epoch in range(3):
        loss, accuracy = trainer.train_epoch(curriculum_dataset, epoch, 10)
    
    # Analyze progress
    analyze_curriculum_progress(trainer)
    
    # Compare strategies (shortened for demo)
    print("\n6. Comparing Curriculum Strategies")
    print("-" * 30)
    
    strategies = ['linear', 'exponential']
    small_dataset = SyntheticDataset(size=100, num_classes=3)
    
    comparison_results = compare_curriculum_strategies(
        small_dataset, lambda: SimpleCNN(num_classes=3), strategies, epochs=3
    )
    
    print("\nStrategy Comparison Results:")
    for strategy, results in comparison_results.items():
        print(f"{strategy}: {results['final_accuracy']:.2f}%")
    
    print("\nCurriculum learning demonstrations completed!") 