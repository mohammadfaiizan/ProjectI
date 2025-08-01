import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List

# Multi-Task Learning Model
class MultiTaskModel(nn.Module):
    """Multi-task learning model with shared backbone and task-specific heads"""
    
    def __init__(self, input_dim, shared_dim=128, task_configs=None):
        super().__init__()
        
        if task_configs is None:
            task_configs = {
                'classification': {'type': 'classification', 'num_classes': 10},
                'regression': {'type': 'regression', 'output_dim': 1}
            }
        
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                self.task_heads[task_name] = nn.Linear(shared_dim, config['num_classes'])
            elif config['type'] == 'regression':
                self.task_heads[task_name] = nn.Linear(shared_dim, config['output_dim'])
            else:
                raise ValueError(f"Unknown task type: {config['type']}")
    
    def forward(self, x, task_name=None):
        """Forward pass through shared backbone and specific task head"""
        shared_features = self.shared_backbone(x)
        
        if task_name is not None:
            # Single task prediction
            return self.task_heads[task_name](shared_features)
        else:
            # All task predictions
            outputs = {}
            for task in self.task_names:
                outputs[task] = self.task_heads[task](shared_features)
            return outputs

# Gradient Balancing Strategies
class GradNorm:
    """Gradient Normalization for multi-task learning"""
    
    def __init__(self, model, task_names, alpha=1.5, lr=0.025):
        self.model = model
        self.task_names = task_names
        self.alpha = alpha
        self.lr = lr
        
        # Initialize task weights
        self.task_weights = nn.Parameter(torch.ones(len(task_names)))
        self.weight_optimizer = torch.optim.Adam([self.task_weights], lr=lr)
        
        # Track initial task losses
        self.initial_losses = {}
        self.current_losses = {}
    
    def compute_gradnorm_loss(self, losses, shared_parameters):
        """Compute GradNorm loss for task weight optimization"""
        # Compute gradient norms for each task
        grad_norms = []
        
        for i, (task_name, loss) in enumerate(losses.items()):
            # Compute gradients w.r.t. shared parameters
            grads = torch.autograd.grad(
                loss, shared_parameters, retain_graph=True, create_graph=True
            )
            
            # Compute gradient norm
            grad_norm = 0
            for grad in grads:
                grad_norm += grad.norm() ** 2
            grad_norm = grad_norm ** 0.5
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # Compute relative inverse training rates
        if len(self.initial_losses) == len(losses):
            loss_ratios = []
            for task_name, loss in losses.items():
                ratio = loss.item() / self.initial_losses[task_name]
                loss_ratios.append(ratio)
            
            loss_ratios = torch.tensor(loss_ratios, device=grad_norms.device)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            
            # Target gradient norms
            mean_grad_norm = grad_norms.mean()
            target_grad_norms = mean_grad_norm * (inverse_train_rates ** self.alpha)
            
            # GradNorm loss
            gradnorm_loss = F.l1_loss(grad_norms, target_grad_norms)
            
            return gradnorm_loss
        
        return torch.tensor(0.0, device=grad_norms.device)
    
    def update_weights(self, losses, shared_parameters):
        """Update task weights using GradNorm"""
        if not self.initial_losses:
            # Store initial losses
            for task_name, loss in losses.items():
                self.initial_losses[task_name] = loss.item()
        
        # Compute GradNorm loss
        gradnorm_loss = self.compute_gradnorm_loss(losses, shared_parameters)
        
        # Update task weights
        self.weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        self.weight_optimizer.step()
        
        # Renormalize weights
        with torch.no_grad():
            self.task_weights.data = len(self.task_names) * F.softmax(self.task_weights, dim=0)
        
        return gradnorm_loss.item()

# Multi-Task Loss Functions
class MultiTaskLoss(nn.Module):
    """Multi-task loss with various weighting strategies"""
    
    def __init__(self, task_configs, weighting_strategy='equal', uncertainty_weights=False):
        super().__init__()
        self.task_configs = task_configs
        self.weighting_strategy = weighting_strategy
        self.uncertainty_weights = uncertainty_weights
        
        # Initialize loss functions
        self.loss_functions = {}
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                self.loss_functions[task_name] = nn.CrossEntropyLoss()
            elif config['type'] == 'regression':
                self.loss_functions[task_name] = nn.MSELoss()
        
        # Initialize task weights
        if weighting_strategy == 'learned':
            self.task_weights = nn.Parameter(torch.ones(len(task_configs)))
        else:
            self.register_buffer('task_weights', torch.ones(len(task_configs)))
        
        # Uncertainty-based weighting (homoscedastic uncertainty)
        if uncertainty_weights:
            self.log_vars = nn.Parameter(torch.zeros(len(task_configs)))
    
    def forward(self, predictions, targets):
        """Compute multi-task loss"""
        losses = {}
        weighted_losses = {}
        
        for i, (task_name, pred) in enumerate(predictions.items()):
            if task_name in targets:
                target = targets[task_name]
                loss = self.loss_functions[task_name](pred, target)
                losses[task_name] = loss
                
                if self.uncertainty_weights:
                    # Uncertainty-based weighting
                    precision = torch.exp(-self.log_vars[i])
                    weighted_loss = precision * loss + self.log_vars[i]
                else:
                    # Standard weighting
                    weight = self.task_weights[i]
                    weighted_loss = weight * loss
                
                weighted_losses[task_name] = weighted_loss
        
        total_loss = sum(weighted_losses.values())
        
        return total_loss, losses, weighted_losses

# Multi-Task Trainer
class MultiTaskTrainer:
    """Trainer for multi-task learning"""
    
    def __init__(self, model, task_configs, device='cuda', lr=1e-3, 
                 use_gradnorm=False, weighting_strategy='equal'):
        self.model = model.to(device)
        self.device = device
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Loss function
        self.criterion = MultiTaskLoss(
            task_configs, 
            weighting_strategy=weighting_strategy,
            uncertainty_weights=(weighting_strategy == 'uncertainty')
        ).to(device)
        
        # GradNorm (optional)
        self.use_gradnorm = use_gradnorm
        if use_gradnorm:
            shared_params = list(model.shared_backbone.parameters())
            self.gradnorm = GradNorm(model, self.task_names)
            self.shared_parameters = shared_params
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {task: 0.0 for task in self.task_names}
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Unpack batch data
            inputs = batch_data['inputs'].to(self.device)
            targets = {task: batch_data[task].to(self.device) 
                      for task in self.task_names if task in batch_data}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(inputs)
            
            # Compute losses
            combined_loss, individual_losses, weighted_losses = self.criterion(predictions, targets)
            
            # Backward pass
            combined_loss.backward(retain_graph=self.use_gradnorm)
            
            # GradNorm weight update
            if self.use_gradnorm and batch_idx % 10 == 0:  # Update every 10 batches
                gradnorm_loss = self.gradnorm.update_weights(individual_losses, self.shared_parameters)
                
                # Update task weights in criterion
                self.criterion.task_weights.data = self.gradnorm.task_weights.data
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += combined_loss.item()
            for task_name, loss in individual_losses.items():
                total_losses[task_name] += loss.item()
            
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Total Loss = {combined_loss.item():.4f}')
                for task_name, loss in individual_losses.items():
                    print(f'  {task_name}: {loss.item():.4f}')
        
        # Average losses
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in total_losses.items()}
        
        return avg_total_loss, avg_task_losses
    
    def evaluate(self, dataloader):
        """Evaluate model on all tasks"""
        self.model.eval()
        
        task_metrics = {}
        for task_name, config in self.task_configs.items():
            if config['type'] == 'classification':
                task_metrics[task_name] = {'correct': 0, 'total': 0}
            elif config['type'] == 'regression':
                task_metrics[task_name] = {'mse': 0.0, 'total': 0}
        
        with torch.no_grad():
            for batch_data in dataloader:
                inputs = batch_data['inputs'].to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Compute metrics for each task
                for task_name in self.task_names:
                    if task_name in batch_data:
                        targets = batch_data[task_name].to(self.device)
                        pred = predictions[task_name]
                        
                        if self.task_configs[task_name]['type'] == 'classification':
                            _, predicted = torch.max(pred, 1)
                            task_metrics[task_name]['correct'] += (predicted == targets).sum().item()
                            task_metrics[task_name]['total'] += targets.size(0)
                        
                        elif self.task_configs[task_name]['type'] == 'regression':
                            mse = F.mse_loss(pred.squeeze(), targets.squeeze())
                            task_metrics[task_name]['mse'] += mse.item() * targets.size(0)
                            task_metrics[task_name]['total'] += targets.size(0)
        
        # Compute final metrics
        results = {}
        for task_name, metrics in task_metrics.items():
            if self.task_configs[task_name]['type'] == 'classification':
                results[task_name] = 100. * metrics['correct'] / metrics['total']
            elif self.task_configs[task_name]['type'] == 'regression':
                results[task_name] = metrics['mse'] / metrics['total']
        
        return results

# Multi-Task Dataset
class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning"""
    
    def __init__(self, size=1000, input_dim=100, task_configs=None):
        if task_configs is None:
            task_configs = {
                'classification': {'type': 'classification', 'num_classes': 5},
                'regression': {'type': 'regression', 'output_dim': 1}
            }
        
        self.size = size
        self.input_dim = input_dim
        self.task_configs = task_configs
        
        # Generate synthetic data
        self.inputs = torch.randn(size, input_dim)
        
        # Generate task-specific targets
        self.targets = {}
        
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                # Classification targets
                self.targets[task_name] = torch.randint(0, config['num_classes'], (size,))
            elif config['type'] == 'regression':
                # Regression targets (correlated with input)
                weights = torch.randn(input_dim, config['output_dim'])
                self.targets[task_name] = torch.mm(self.inputs, weights).squeeze()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = {'inputs': self.inputs[idx]}
        
        for task_name in self.targets:
            sample[task_name] = self.targets[task_name][idx]
        
        return sample

# Comparison with Single-Task Learning
def compare_with_single_task(multi_task_model, task_configs, train_dataset, test_dataset, device='cuda'):
    """Compare multi-task learning with single-task baselines"""
    
    print("\nTraining single-task baselines...")
    
    single_task_results = {}
    
    for task_name, config in task_configs.items():
        print(f"\nTraining single-task model for {task_name}")
        
        # Create single-task model
        if config['type'] == 'classification':
            single_model = nn.Sequential(
                nn.Linear(train_dataset.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, config['num_classes'])
            ).to(device)
            
            criterion = nn.CrossEntropyLoss()
        
        elif config['type'] == 'regression':
            single_model = nn.Sequential(
                nn.Linear(train_dataset.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, config['output_dim'])
            ).to(device)
            
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(single_model.parameters(), lr=1e-3)
        
        # Train single-task model
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        single_model.train()
        for epoch in range(10):  # Quick training
            for batch_data in train_loader:
                inputs = batch_data['inputs'].to(device)
                targets = batch_data[task_name].to(device)
                
                optimizer.zero_grad()
                outputs = single_model(inputs)
                
                if config['type'] == 'classification':
                    loss = criterion(outputs, targets)
                else:
                    loss = criterion(outputs.squeeze(), targets)
                
                loss.backward()
                optimizer.step()
        
        # Evaluate single-task model
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        single_model.eval()
        
        if config['type'] == 'classification':
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data in test_loader:
                    inputs = batch_data['inputs'].to(device)
                    targets = batch_data[task_name].to(device)
                    outputs = single_model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            single_task_results[task_name] = 100. * correct / total
        
        elif config['type'] == 'regression':
            total_mse = 0.0
            total_samples = 0
            with torch.no_grad():
                for batch_data in test_loader:
                    inputs = batch_data['inputs'].to(device)
                    targets = batch_data[task_name].to(device)
                    outputs = single_model(inputs)
                    mse = F.mse_loss(outputs.squeeze(), targets)
                    total_mse += mse.item() * targets.size(0)
                    total_samples += targets.size(0)
            
            single_task_results[task_name] = total_mse / total_samples
    
    return single_task_results

if __name__ == "__main__":
    print("Multi-Task Learning")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define task configurations
    task_configs = {
        'classification': {'type': 'classification', 'num_classes': 5},
        'regression': {'type': 'regression', 'output_dim': 1}
    }
    
    # Create datasets
    train_dataset = MultiTaskDataset(size=800, input_dim=50, task_configs=task_configs)
    test_dataset = MultiTaskDataset(size=200, input_dim=50, task_configs=task_configs)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test equal weighting
    print("\n1. Multi-Task Learning with Equal Weighting")
    print("-" * 45)
    
    model1 = MultiTaskModel(input_dim=50, shared_dim=128, task_configs=task_configs)
    trainer1 = MultiTaskTrainer(model1, task_configs, device, weighting_strategy='equal')
    
    for epoch in range(5):
        avg_loss, task_losses = trainer1.train_epoch(train_loader)
        print(f'Epoch {epoch + 1}: Total Loss = {avg_loss:.4f}')
        for task, loss in task_losses.items():
            print(f'  {task}: {loss:.4f}')
    
    # Evaluate
    mt_results_equal = trainer1.evaluate(test_loader)
    print("Multi-task results (equal weighting):")
    for task, result in mt_results_equal.items():
        if task_configs[task]['type'] == 'classification':
            print(f"  {task} accuracy: {result:.2f}%")
        else:
            print(f"  {task} MSE: {result:.4f}")
    
    # Test uncertainty-based weighting
    print("\n2. Multi-Task Learning with Uncertainty Weighting")
    print("-" * 50)
    
    model2 = MultiTaskModel(input_dim=50, shared_dim=128, task_configs=task_configs)
    trainer2 = MultiTaskTrainer(model2, task_configs, device, weighting_strategy='uncertainty')
    
    for epoch in range(5):
        avg_loss, task_losses = trainer2.train_epoch(train_loader)
        print(f'Epoch {epoch + 1}: Total Loss = {avg_loss:.4f}')
        
        # Print uncertainty weights
        if hasattr(trainer2.criterion, 'log_vars'):
            uncertainties = torch.exp(-trainer2.criterion.log_vars)
            print(f'  Task weights: {uncertainties.detach().cpu().numpy()}')
    
    # Evaluate
    mt_results_uncertainty = trainer2.evaluate(test_loader)
    print("Multi-task results (uncertainty weighting):")
    for task, result in mt_results_uncertainty.items():
        if task_configs[task]['type'] == 'classification':
            print(f"  {task} accuracy: {result:.2f}%")
        else:
            print(f"  {task} MSE: {result:.4f}")
    
    # Test GradNorm
    print("\n3. Multi-Task Learning with GradNorm")
    print("-" * 40)
    
    model3 = MultiTaskModel(input_dim=50, shared_dim=128, task_configs=task_configs)
    trainer3 = MultiTaskTrainer(model3, task_configs, device, use_gradnorm=True)
    
    for epoch in range(3):  # Fewer epochs for GradNorm demo
        avg_loss, task_losses = trainer3.train_epoch(train_loader)
        print(f'Epoch {epoch + 1}: Total Loss = {avg_loss:.4f}')
        
        # Print task weights
        if trainer3.use_gradnorm:
            weights = trainer3.gradnorm.task_weights.detach().cpu().numpy()
            print(f'  GradNorm weights: {weights}')
    
    # Evaluate
    mt_results_gradnorm = trainer3.evaluate(test_loader)
    print("Multi-task results (GradNorm):")
    for task, result in mt_results_gradnorm.items():
        if task_configs[task]['type'] == 'classification':
            print(f"  {task} accuracy: {result:.2f}%")
        else:
            print(f"  {task} MSE: {result:.4f}")
    
    # Compare with single-task baselines
    print("\n4. Comparison with Single-Task Baselines")
    print("-" * 45)
    
    single_task_results = compare_with_single_task(
        model1, task_configs, train_dataset, test_dataset, device
    )
    
    print("\nSingle-task baseline results:")
    for task, result in single_task_results.items():
        if task_configs[task]['type'] == 'classification':
            print(f"  {task} accuracy: {result:.2f}%")
        else:
            print(f"  {task} MSE: {result:.4f}")
    
    # Summary comparison
    print("\n5. Summary Comparison")
    print("-" * 25)
    
    for task in task_configs.keys():
        print(f"\n{task.capitalize()} Task:")
        print(f"  Single-task: {single_task_results[task]:.4f}")
        print(f"  Multi-task (equal): {mt_results_equal[task]:.4f}")
        print(f"  Multi-task (uncertainty): {mt_results_uncertainty[task]:.4f}")
        print(f"  Multi-task (GradNorm): {mt_results_gradnorm[task]:.4f}")
    
    print("\nMulti-task learning demonstrations completed!") 