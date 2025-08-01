import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import random

# Active Learning Query Strategies
class UncertaintySampling:
    """Uncertainty-based sampling strategies"""
    
    def __init__(self, strategy='entropy'):
        self.strategy = strategy
    
    def query(self, model, unlabeled_loader, n_samples, device='cuda'):
        """Query samples based on uncertainty"""
        model.eval()
        uncertainties = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(device)
                outputs = model(data)
                probs = F.softmax(outputs, dim=1)
                
                if self.strategy == 'entropy':
                    # Entropy-based uncertainty
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    uncertainties.extend(entropy.cpu().tolist())
                
                elif self.strategy == 'least_confidence':
                    # Least confidence
                    max_probs, _ = torch.max(probs, dim=1)
                    uncertainty = 1 - max_probs
                    uncertainties.extend(uncertainty.cpu().tolist())
                
                elif self.strategy == 'margin':
                    # Margin sampling
                    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
                    uncertainty = 1 - margins
                    uncertainties.extend(uncertainty.cpu().tolist())
                
                # Track original indices
                batch_size = data.size(0)
                start_idx = batch_idx * unlabeled_loader.batch_size
                indices.extend(range(start_idx, start_idx + batch_size))
        
        # Select most uncertain samples
        uncertainty_scores = list(zip(indices, uncertainties))
        uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = [idx for idx, _ in uncertainty_scores[:n_samples]]
        return selected_indices

class QueryByCommittee:
    """Query by Committee sampling strategy"""
    
    def __init__(self, committee_models, strategy='vote_entropy'):
        self.committee_models = committee_models
        self.strategy = strategy
    
    def query(self, unlabeled_loader, n_samples, device='cuda'):
        """Query samples using committee disagreement"""
        for model in self.committee_models:
            model.eval()
        
        disagreements = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(device)
                
                # Get predictions from all committee members
                committee_outputs = []
                for model in self.committee_models:
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)
                    committee_outputs.append(probs)
                
                committee_outputs = torch.stack(committee_outputs)  # [n_models, batch_size, n_classes]
                
                if self.strategy == 'vote_entropy':
                    # Vote entropy
                    avg_probs = committee_outputs.mean(dim=0)
                    vote_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-8), dim=1)
                    disagreements.extend(vote_entropy.cpu().tolist())
                
                elif self.strategy == 'consensus_entropy':
                    # Consensus entropy
                    consensus_entropy = 0
                    for model_probs in committee_outputs:
                        consensus_entropy += -torch.sum(model_probs * torch.log(model_probs + 1e-8), dim=1)
                    consensus_entropy /= len(self.committee_models)
                    disagreements.extend(consensus_entropy.cpu().tolist())
                
                elif self.strategy == 'kl_divergence':
                    # Average KL divergence between committee members
                    avg_probs = committee_outputs.mean(dim=0)
                    kl_divs = []
                    for model_probs in committee_outputs:
                        kl_div = F.kl_div(torch.log(model_probs + 1e-8), avg_probs, reduction='none').sum(dim=1)
                        kl_divs.append(kl_div)
                    avg_kl_div = torch.stack(kl_divs).mean(dim=0)
                    disagreements.extend(avg_kl_div.cpu().tolist())
                
                # Track indices
                batch_size = data.size(0)
                start_idx = batch_idx * unlabeled_loader.batch_size
                indices.extend(range(start_idx, start_idx + batch_size))
        
        # Select samples with highest disagreement
        disagreement_scores = list(zip(indices, disagreements))
        disagreement_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_indices = [idx for idx, _ in disagreement_scores[:n_samples]]
        return selected_indices

class DiversitySampling:
    """Diversity-based sampling strategies"""
    
    def __init__(self, strategy='k_means'):
        self.strategy = strategy
    
    def query(self, model, unlabeled_loader, n_samples, device='cuda'):
        """Query diverse samples"""
        model.eval()
        features = []
        indices = []
        
        # Extract features
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(unlabeled_loader):
                data = data.to(device)
                
                # Get features from model (assume model has feature extraction)
                if hasattr(model, 'features'):
                    feat = model.features(data)
                else:
                    # Use penultimate layer
                    feat = model(data)
                
                feat = feat.view(feat.size(0), -1)
                features.append(feat.cpu())
                
                # Track indices
                batch_size = data.size(0)
                start_idx = batch_idx * unlabeled_loader.batch_size
                indices.extend(range(start_idx, start_idx + batch_size))
        
        features = torch.cat(features, dim=0).numpy()
        
        if self.strategy == 'k_means':
            # K-means clustering for diversity
            kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
            cluster_centers = kmeans.fit(features).cluster_centers_
            
            # Find closest samples to cluster centers
            selected_indices = []
            for center in cluster_centers:
                distances = np.sum((features - center) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(indices[closest_idx])
        
        elif self.strategy == 'farthest_first':
            # Farthest-first traversal
            selected_indices = []
            remaining_indices = list(range(len(features)))
            
            # Start with random sample
            first_idx = random.choice(remaining_indices)
            selected_indices.append(indices[first_idx])
            remaining_indices.remove(first_idx)
            
            # Iteratively select farthest samples
            for _ in range(n_samples - 1):
                if not remaining_indices:
                    break
                
                max_min_distance = -1
                farthest_idx = None
                
                for idx in remaining_indices:
                    # Compute minimum distance to selected samples
                    min_distance = float('inf')
                    for selected_idx in selected_indices:
                        selected_feat_idx = indices.index(selected_idx)
                        distance = np.sum((features[idx] - features[selected_feat_idx]) ** 2)
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        farthest_idx = idx
                
                if farthest_idx is not None:
                    selected_indices.append(indices[farthest_idx])
                    remaining_indices.remove(farthest_idx)
        
        return selected_indices

# Active Learning Trainer
class ActiveLearningTrainer:
    """Active Learning training loop"""
    
    def __init__(self, model, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def active_learning_loop(self, labeled_dataset, unlabeled_dataset, test_dataset,
                           query_strategy, n_queries=100, n_rounds=5, batch_size=32):
        """Complete active learning training loop"""
        
        labeled_indices = list(range(len(labeled_dataset)))
        unlabeled_indices = list(range(len(unlabeled_dataset)))
        
        results = {'round': [], 'labeled_size': [], 'test_accuracy': []}
        
        for round_num in range(n_rounds):
            print(f"\n=== Active Learning Round {round_num + 1} ===")
            
            # Train on current labeled data
            if labeled_indices:
                labeled_subset = Subset(labeled_dataset, labeled_indices)
                labeled_loader = DataLoader(labeled_subset, batch_size=batch_size, shuffle=True)
                
                # Train for a few epochs
                for epoch in range(5):
                    loss, acc = self.train_epoch(labeled_loader)
                    if epoch % 2 == 0:
                        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%")
            
            # Evaluate on test set
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_accuracy = self.evaluate(test_loader)
            
            print(f"Test Accuracy: {test_accuracy:.2f}%")
            print(f"Labeled samples: {len(labeled_indices)}")
            
            # Store results
            results['round'].append(round_num + 1)
            results['labeled_size'].append(len(labeled_indices))
            results['test_accuracy'].append(test_accuracy)
            
            # Query new samples (if not last round)
            if round_num < n_rounds - 1 and unlabeled_indices:
                # Create unlabeled subset
                unlabeled_subset = Subset(unlabeled_dataset, unlabeled_indices)
                unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False)
                
                # Query samples
                if isinstance(query_strategy, UncertaintySampling):
                    queried_indices = query_strategy.query(self.model, unlabeled_loader, n_queries, self.device)
                elif isinstance(query_strategy, QueryByCommittee):
                    queried_indices = query_strategy.query(unlabeled_loader, n_queries, self.device)
                elif isinstance(query_strategy, DiversitySampling):
                    queried_indices = query_strategy.query(self.model, unlabeled_loader, n_queries, self.device)
                
                # Convert subset indices to original dataset indices
                actual_queried_indices = [unlabeled_indices[idx] for idx in queried_indices if idx < len(unlabeled_indices)]
                
                print(f"Queried {len(actual_queried_indices)} new samples")
                
                # Move queried samples to labeled set
                labeled_indices.extend(actual_queried_indices)
                for idx in actual_queried_indices:
                    if idx in unlabeled_indices:
                        unlabeled_indices.remove(idx)
        
        return results

# Simple CNN Model
class SimpleCNN(nn.Module):
    """Simple CNN for active learning experiments"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
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
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

# Sample Datasets
class SampleDataset(Dataset):
    """Sample dataset for active learning"""
    
    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Comparison Functions
def compare_active_learning_strategies(labeled_dataset, unlabeled_dataset, test_dataset, 
                                     strategies, n_rounds=3, device='cuda'):
    """Compare different active learning strategies"""
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\n{'='*50}")
        print(f"Testing {strategy_name}")
        print(f"{'='*50}")
        
        # Create fresh model for each strategy
        model = SimpleCNN(num_classes=10)
        trainer = ActiveLearningTrainer(model, device=device)
        
        # Run active learning
        strategy_results = trainer.active_learning_loop(
            labeled_dataset, unlabeled_dataset, test_dataset,
            strategy, n_queries=50, n_rounds=n_rounds
        )
        
        results[strategy_name] = strategy_results
    
    return results

def random_sampling_baseline(labeled_dataset, unlabeled_dataset, test_dataset, 
                            n_rounds=3, n_queries=50, device='cuda'):
    """Random sampling baseline"""
    
    print(f"\n{'='*50}")
    print(f"Testing Random Sampling Baseline")
    print(f"{'='*50}")
    
    model = SimpleCNN(num_classes=10)
    trainer = ActiveLearningTrainer(model, device=device)
    
    labeled_indices = list(range(len(labeled_dataset)))
    unlabeled_indices = list(range(len(unlabeled_dataset)))
    
    results = {'round': [], 'labeled_size': [], 'test_accuracy': []}
    
    for round_num in range(n_rounds):
        print(f"\n=== Random Sampling Round {round_num + 1} ===")
        
        # Train on current labeled data
        if labeled_indices:
            labeled_subset = Subset(labeled_dataset, labeled_indices)
            labeled_loader = DataLoader(labeled_subset, batch_size=32, shuffle=True)
            
            for epoch in range(5):
                loss, acc = trainer.train_epoch(labeled_loader)
                if epoch % 2 == 0:
                    print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%")
        
        # Evaluate
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_accuracy = trainer.evaluate(test_loader)
        
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Labeled samples: {len(labeled_indices)}")
        
        results['round'].append(round_num + 1)
        results['labeled_size'].append(len(labeled_indices))
        results['test_accuracy'].append(test_accuracy)
        
        # Random sampling
        if round_num < n_rounds - 1 and unlabeled_indices:
            n_to_sample = min(n_queries, len(unlabeled_indices))
            random_indices = random.sample(unlabeled_indices, n_to_sample)
            
            labeled_indices.extend(random_indices)
            for idx in random_indices:
                unlabeled_indices.remove(idx)
            
            print(f"Randomly sampled {len(random_indices)} new samples")
    
    return results

if __name__ == "__main__":
    print("Active Learning")
    print("=" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    total_size = 800
    initial_labeled_size = 100
    test_size = 200
    
    # Split data
    full_dataset = SampleDataset(size=total_size, num_classes=5)
    
    # Initial labeled set (small)
    labeled_indices = list(range(initial_labeled_size))
    labeled_dataset = Subset(full_dataset, labeled_indices)
    
    # Unlabeled pool
    unlabeled_indices = list(range(initial_labeled_size, total_size))
    unlabeled_dataset = Subset(full_dataset, unlabeled_indices)
    
    # Test set
    test_dataset = SampleDataset(size=test_size, num_classes=5)
    
    print(f"Initial labeled set: {len(labeled_dataset)}")
    print(f"Unlabeled pool: {len(unlabeled_dataset)}")
    print(f"Test set: {len(test_dataset)}")
    
    # Test individual strategies
    print("\n1. Testing Uncertainty Sampling")
    print("-" * 35)
    
    uncertainty_strategy = UncertaintySampling(strategy='entropy')
    model = SimpleCNN(num_classes=5)
    trainer = ActiveLearningTrainer(model, device=device)
    
    uncertainty_results = trainer.active_learning_loop(
        labeled_dataset, unlabeled_dataset, test_dataset,
        uncertainty_strategy, n_queries=30, n_rounds=3
    )
    
    print("\n2. Testing Query by Committee")
    print("-" * 30)
    
    # Create committee of models
    committee_models = [SimpleCNN(num_classes=5) for _ in range(3)]
    
    # Pre-train committee on initial labeled data
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True)
    
    for i, committee_model in enumerate(committee_models):
        committee_model = committee_model.to(device)
        committee_trainer = ActiveLearningTrainer(committee_model, device=device)
        
        print(f"Pre-training committee model {i+1}")
        for epoch in range(3):
            loss, acc = committee_trainer.train_epoch(labeled_loader)
        
        committee_models[i] = committee_model
    
    qbc_strategy = QueryByCommittee(committee_models, strategy='vote_entropy')
    
    # Note: For demonstration, we'll use a smaller example
    small_unlabeled = Subset(unlabeled_dataset, list(range(min(100, len(unlabeled_dataset)))))
    
    model = SimpleCNN(num_classes=5)
    trainer = ActiveLearningTrainer(model, device=device)
    
    qbc_results = trainer.active_learning_loop(
        labeled_dataset, small_unlabeled, test_dataset,
        qbc_strategy, n_queries=20, n_rounds=2
    )
    
    print("\n3. Testing Diversity Sampling")
    print("-" * 30)
    
    diversity_strategy = DiversitySampling(strategy='k_means')
    model = SimpleCNN(num_classes=5)
    trainer = ActiveLearningTrainer(model, device=device)
    
    diversity_results = trainer.active_learning_loop(
        labeled_dataset, small_unlabeled, test_dataset,
        diversity_strategy, n_queries=20, n_rounds=2
    )
    
    print("\n4. Random Sampling Baseline")
    print("-" * 30)
    
    random_results = random_sampling_baseline(
        labeled_dataset, small_unlabeled, test_dataset,
        n_rounds=3, n_queries=20, device=device
    )
    
    # Compare strategies
    print("\n5. Strategy Comparison")
    print("-" * 25)
    
    strategies = {
        'Uncertainty (Entropy)': UncertaintySampling('entropy'),
        'Uncertainty (Margin)': UncertaintySampling('margin'),
        'Diversity (K-means)': DiversitySampling('k_means')
    }
    
    # Create smaller datasets for comparison
    small_labeled = Subset(labeled_dataset, list(range(50)))
    small_unlabeled = Subset(unlabeled_dataset, list(range(150)))
    small_test = Subset(test_dataset, list(range(100)))
    
    comparison_results = compare_active_learning_strategies(
        small_labeled, small_unlabeled, small_test,
        strategies, n_rounds=2, device=device
    )
    
    # Print comparison summary
    print("\n6. Results Summary")
    print("-" * 20)
    
    print("Strategy Performance (Final Round):")
    for strategy_name, results in comparison_results.items():
        final_accuracy = results['test_accuracy'][-1]
        print(f"  {strategy_name}: {final_accuracy:.2f}%")
    
    # Random baseline
    final_random_accuracy = random_results['test_accuracy'][-1]
    print(f"  Random Baseline: {final_random_accuracy:.2f}%")
    
    print("\nActive learning demonstrations completed!") 