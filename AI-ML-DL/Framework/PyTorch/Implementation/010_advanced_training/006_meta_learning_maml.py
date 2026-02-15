import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import copy

# MAML (Model-Agnostic Meta-Learning) Implementation
class MAML:
    """Model-Agnostic Meta-Learning implementation"""
    
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, inner_steps=5, 
                 first_order=False, device='cuda'):
        self.model = model.to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order  # First-order MAML (FOMAML)
        self.device = device
        
        # Meta optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
        # Task-specific loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def inner_update(self, support_data, support_labels, model_params=None):
        """Perform inner loop update on support set"""
        if model_params is None:
            model_params = list(self.model.parameters())
        
        # Create functional model with current parameters
        updated_params = []
        
        for step in range(self.inner_steps):
            # Forward pass through support set
            support_predictions = self._forward_with_params(support_data, model_params)
            support_loss = self.criterion(support_predictions, support_labels)
            
            # Compute gradients with respect to model parameters
            grads = torch.autograd.grad(support_loss, model_params, 
                                      create_graph=not self.first_order, 
                                      retain_graph=True, allow_unused=True)
            
            # Update parameters using gradient descent
            updated_params = []
            for param, grad in zip(model_params, grads):
                if grad is not None:
                    updated_param = param - self.inner_lr * grad
                else:
                    updated_param = param
                updated_params.append(updated_param)
            
            model_params = updated_params
        
        return updated_params
    
    def _forward_with_params(self, x, params):
        """Forward pass using specific parameters"""
        # This is a simplified version - in practice, you'd need to handle
        # the forward pass with custom parameters for each layer type
        
        # For demonstration, we'll use a simple approach
        # Save original parameters
        original_params = list(self.model.parameters())
        
        # Set new parameters
        for orig_param, new_param in zip(original_params, params):
            orig_param.data = new_param.data
        
        # Forward pass
        output = self.model(x)
        
        # Restore original parameters
        for orig_param, new_param in zip(original_params, params):
            orig_param.data = new_param.data
        
        return output
    
    def meta_update(self, tasks_batch):
        """Perform meta update using batch of tasks"""
        meta_loss = 0.0
        meta_gradients = []
        
        for task in tasks_batch:
            support_data, support_labels, query_data, query_labels = task
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Inner loop: adapt to support set
            adapted_params = self.inner_update(support_data, support_labels)
            
            # Compute loss on query set with adapted parameters
            query_predictions = self._forward_with_params(query_data, adapted_params)
            query_loss = self.criterion(query_predictions, query_labels)
            
            meta_loss += query_loss
        
        # Average meta loss
        meta_loss = meta_loss / len(tasks_batch)
        
        # Meta optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_task(self, support_data, support_labels):
        """Adapt model to a new task using support set"""
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Fine-tune on support set
        adapted_model.train()
        for step in range(self.inner_steps):
            adapted_optimizer.zero_grad()
            
            predictions = adapted_model(support_data)
            loss = self.criterion(predictions, support_labels)
            
            loss.backward()
            adapted_optimizer.step()
        
        return adapted_model

# First-Order MAML (FOMAML)
class FOMAML(MAML):
    """First-Order MAML for computational efficiency"""
    
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, inner_steps=5, device='cuda'):
        super().__init__(model, meta_lr, inner_lr, inner_steps, first_order=True, device=device)
    
    def meta_update(self, tasks_batch):
        """Simplified meta update without second-order gradients"""
        total_meta_loss = 0.0
        
        # Store original parameters
        original_params = [p.clone() for p in self.model.parameters()]
        
        task_gradients = []
        
        for task in tasks_batch:
            support_data, support_labels, query_data, query_labels = task
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Reset model to original parameters
            for orig_param, model_param in zip(original_params, self.model.parameters()):
                model_param.data = orig_param.data
            
            # Inner loop adaptation
            inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
            
            for _ in range(self.inner_steps):
                inner_optimizer.zero_grad()
                support_pred = self.model(support_data)
                support_loss = self.criterion(support_pred, support_labels)
                support_loss.backward()
                inner_optimizer.step()
            
            # Compute query loss with adapted parameters
            query_pred = self.model(query_data)
            query_loss = self.criterion(query_pred, query_labels)
            
            # Compute gradients for meta update
            meta_grads = torch.autograd.grad(query_loss, self.model.parameters())
            task_gradients.append([g.clone() for g in meta_grads])
            
            total_meta_loss += query_loss.item()
        
        # Average gradients across tasks
        avg_gradients = []
        for i in range(len(self.model.parameters())):
            avg_grad = torch.stack([task_grads[i] for task_grads in task_gradients]).mean(dim=0)
            avg_gradients.append(avg_grad)
        
        # Restore original parameters and apply meta update
        for orig_param, model_param, meta_grad in zip(original_params, self.model.parameters(), avg_gradients):
            model_param.data = orig_param.data - self.meta_lr * meta_grad
        
        return total_meta_loss / len(tasks_batch)

# Task Generator for Meta-Learning
class TaskGenerator:
    """Generate tasks for meta-learning"""
    
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        self.dataset = dataset
        self.n_way = n_way  # Number of classes per task
        self.k_shot = k_shot  # Number of support examples per class
        self.q_query = q_query  # Number of query examples per class
        
        # Organize dataset by class
        self.class_to_indices = self._organize_by_class()
        self.classes = list(self.class_to_indices.keys())
    
    def _organize_by_class(self):
        """Organize dataset indices by class"""
        class_to_indices = {}
        
        for idx, (_, label) in enumerate(self.dataset):
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        return class_to_indices
    
    def sample_task(self):
        """Sample a single task (support and query sets)"""
        # Randomly select n_way classes
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for class_idx, class_label in enumerate(selected_classes):
            # Get available indices for this class
            available_indices = self.class_to_indices[class_label]
            
            # Ensure we have enough samples
            if len(available_indices) < self.k_shot + self.q_query:
                # Sample with replacement if needed
                sampled_indices = random.choices(available_indices, k=self.k_shot + self.q_query)
            else:
                sampled_indices = random.sample(available_indices, self.k_shot + self.q_query)
            
            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:self.k_shot + self.q_query]
            
            # Collect support examples
            for idx in support_indices:
                data, _ = self.dataset[idx]
                support_data.append(data)
                support_labels.append(class_idx)  # Use 0-indexed class labels
            
            # Collect query examples
            for idx in query_indices:
                data, _ = self.dataset[idx]
                query_data.append(data)
                query_labels.append(class_idx)
        
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)
        
        return support_data, support_labels, query_data, query_labels
    
    def sample_batch(self, batch_size):
        """Sample a batch of tasks"""
        return [self.sample_task() for _ in range(batch_size)]

# Reptile Meta-Learning Algorithm
class Reptile:
    """Reptile meta-learning algorithm"""
    
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, inner_steps=5, device='cuda'):
        self.model = model.to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
    
    def meta_update(self, task):
        """Reptile meta update using a single task"""
        support_data, support_labels, _, _ = task
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Inner loop: fine-tune on support set
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        self.model.train()
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            
            predictions = self.model(support_data)
            loss = self.criterion(predictions, support_labels)
            
            loss.backward()
            inner_optimizer.step()
        
        # Meta update: move towards fine-tuned parameters
        for initial_param, current_param in zip(initial_params, self.model.parameters()):
            initial_param.data += self.meta_lr * (current_param.data - initial_param.data)
            current_param.data = initial_param.data

# Sample Models for Meta-Learning
class SimpleMetaModel(nn.Module):
    """Simple model for meta-learning experiments"""
    
    def __init__(self, input_size=784, hidden_size=64, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ConvMetaModel(nn.Module):
    """Convolutional model for meta-learning"""
    
    def __init__(self, input_channels=1, num_classes=5, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Synthetic Dataset for Meta-Learning
class SyntheticMetaDataset(Dataset):
    """Synthetic dataset for meta-learning experiments"""
    
    def __init__(self, num_classes=20, samples_per_class=100, input_shape=(1, 28, 28)):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_shape = input_shape
        
        # Generate synthetic data
        self.data = []
        self.labels = []
        
        for class_id in range(num_classes):
            # Create class-specific pattern
            class_mean = torch.randn(input_shape) * 0.5
            
            for _ in range(samples_per_class):
                # Add noise to class pattern
                sample = class_mean + torch.randn(input_shape) * 0.3
                sample = torch.clamp(sample, 0, 1)
                
                self.data.append(sample)
                self.labels.append(class_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Evaluation Functions
def evaluate_few_shot_performance(model, task_generator, num_tasks=100, device='cuda'):
    """Evaluate few-shot learning performance"""
    model.eval()
    
    total_accuracy = 0.0
    
    with torch.no_grad():
        for _ in range(num_tasks):
            # Sample a task
            support_data, support_labels, query_data, query_labels = task_generator.sample_task()
            
            # Adapt model to task (simplified - just use support for reference)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            # Evaluate on query set
            predictions = model(query_data)
            _, predicted = torch.max(predictions, 1)
            
            accuracy = (predicted == query_labels).float().mean().item()
            total_accuracy += accuracy
    
    return total_accuracy / num_tasks

def compare_meta_algorithms(dataset, algorithms, num_meta_iterations=50):
    """Compare different meta-learning algorithms"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_generator = TaskGenerator(dataset, n_way=3, k_shot=1, q_query=10)
    
    results = {}
    
    for alg_name, alg_config in algorithms.items():
        print(f"\nTraining {alg_name}...")
        
        # Create model and algorithm
        model = alg_config['model']().to(device)
        
        if alg_name == 'MAML':
            algorithm = MAML(model, device=device)
        elif alg_name == 'FOMAML':
            algorithm = FOMAML(model, device=device)
        elif alg_name == 'Reptile':
            algorithm = Reptile(model, device=device)
        
        # Meta-training
        for iteration in range(num_meta_iterations):
            if alg_name in ['MAML', 'FOMAML']:
                # Sample batch of tasks
                task_batch = task_generator.sample_batch(4)
                meta_loss = algorithm.meta_update(task_batch)
            else:  # Reptile
                # Sample single task
                task = task_generator.sample_task()
                algorithm.meta_update(task)
                meta_loss = 0.0  # Reptile doesn't return loss
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}")
        
        # Evaluate
        accuracy = evaluate_few_shot_performance(model, task_generator, num_tasks=20, device=device)
        results[alg_name] = accuracy
        
        print(f"{alg_name} Few-shot Accuracy: {accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    print("Meta-Learning with MAML")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    dataset = SyntheticMetaDataset(num_classes=10, samples_per_class=50, input_shape=(1, 28, 28))
    print(f"Dataset size: {len(dataset)}")
    
    # Test task generation
    print("\n1. Testing Task Generation")
    print("-" * 25)
    
    task_generator = TaskGenerator(dataset, n_way=3, k_shot=2, q_query=5)
    
    # Sample a task
    support_data, support_labels, query_data, query_labels = task_generator.sample_task()
    print(f"Support set: {support_data.shape}, {support_labels.shape}")
    print(f"Query set: {query_data.shape}, {query_labels.shape}")
    print(f"Support labels: {support_labels}")
    print(f"Query labels: {query_labels}")
    
    # Test MAML
    print("\n2. Testing MAML")
    print("-" * 25)
    
    model = SimpleMetaModel(input_size=784, output_size=3).to(device)
    maml = MAML(model, meta_lr=0.001, inner_lr=0.01, inner_steps=3, device=device)
    
    # Sample batch of tasks
    task_batch = task_generator.sample_batch(2)
    
    # Perform meta update
    meta_loss = maml.meta_update(task_batch)
    print(f"Meta loss: {meta_loss:.4f}")
    
    # Test task adaptation
    print("\n3. Testing Task Adaptation")
    print("-" * 25)
    
    support_data, support_labels, query_data, query_labels = task_generator.sample_task()
    adapted_model = maml.adapt_to_task(support_data.to(device), support_labels.to(device))
    
    # Evaluate adapted model
    with torch.no_grad():
        query_pred = adapted_model(query_data.to(device))
        _, predicted = torch.max(query_pred, 1)
        accuracy = (predicted == query_labels.to(device)).float().mean().item()
        print(f"Adapted model accuracy: {accuracy:.4f}")
    
    # Test FOMAML
    print("\n4. Testing FOMAML")
    print("-" * 25)
    
    model = SimpleMetaModel(input_size=784, output_size=3).to(device)
    fomaml = FOMAML(model, device=device)
    
    # Meta training for a few iterations
    for i in range(3):
        task_batch = task_generator.sample_batch(2)
        meta_loss = fomaml.meta_update(task_batch)
        print(f"FOMAML iteration {i+1}: Meta loss = {meta_loss:.4f}")
    
    # Test Reptile
    print("\n5. Testing Reptile")
    print("-" * 25)
    
    model = SimpleMetaModel(input_size=784, output_size=3).to(device)
    reptile = Reptile(model, device=device)
    
    # Meta training for a few iterations
    for i in range(3):
        task = task_generator.sample_task()
        reptile.meta_update(task)
        print(f"Reptile iteration {i+1} completed")
    
    # Compare algorithms (simplified)
    print("\n6. Comparing Meta-Learning Algorithms")
    print("-" * 25)
    
    algorithms = {
        'MAML': {'model': lambda: SimpleMetaModel(input_size=784, output_size=3)},
        'FOMAML': {'model': lambda: SimpleMetaModel(input_size=784, output_size=3)},
        'Reptile': {'model': lambda: SimpleMetaModel(input_size=784, output_size=3)}
    }
    
    small_dataset = SyntheticMetaDataset(num_classes=6, samples_per_class=20, input_shape=(1, 28, 28))
    
    comparison_results = compare_meta_algorithms(small_dataset, algorithms, num_meta_iterations=10)
    
    print("\nAlgorithm Comparison Results:")
    for alg_name, accuracy in comparison_results.items():
        print(f"{alg_name}: {accuracy:.4f}")
    
    print("\nMAML and meta-learning demonstrations completed!") 