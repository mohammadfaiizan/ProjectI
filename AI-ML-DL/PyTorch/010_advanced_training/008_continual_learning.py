import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
from typing import Dict, List, Optional

# Elastic Weight Consolidation (EWC)
class EWC:
    """Elastic Weight Consolidation for continual learning"""
    
    def __init__(self, model, dataset_loader, device='cuda', importance=1000):
        self.model = model
        self.device = device
        self.importance = importance
        
        # Store important parameters and Fisher information
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher_information(dataset_loader)
    
    def _compute_fisher_information(self, dataloader):
        """Compute Fisher Information Matrix"""
        fisher = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        self.model.eval()
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(data)
            loss = F.log_softmax(outputs, dim=1)[range(targets.size(0)), targets].sum()
            
            # Compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        
        # Normalize by dataset size
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)
        
        return fisher
    
    def penalty(self):
        """Compute EWC penalty"""
        loss = 0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        
        return self.importance * loss

class EWCTrainer:
    """Trainer with EWC regularization"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.ewc_loss = None
        self.tasks_seen = 0
    
    def train_task(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """Train on a new task"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                ce_loss = criterion(outputs, targets)
                
                # Add EWC penalty for previous tasks
                total_loss = ce_loss
                if self.ewc_loss is not None:
                    ewc_penalty = self.ewc_loss.penalty()
                    total_loss += ewc_penalty
                
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    print(f'Task {self.tasks_seen + 1}, Epoch {epoch + 1}, '
                          f'Batch {batch_idx}, Loss: {total_loss.item():.4f}')
        
        # Update EWC for this task
        self.ewc_loss = EWC(self.model, train_loader, self.device)
        self.tasks_seen += 1

# Learning without Forgetting (LwF)
class LwFTrainer:
    """Learning without Forgetting trainer"""
    
    def __init__(self, model, device='cuda', alpha=1.0, T=2.0):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha  # Weight for distillation loss
        self.T = T  # Temperature for distillation
        self.old_model = None
        
    def train_task(self, train_loader, epochs=10, lr=0.001):
        """Train on new task while preserving old knowledge"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            if self.old_model is not None:
                self.old_model.eval()
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                ce_loss = criterion(outputs, targets)
                
                total_loss = ce_loss
                
                # Knowledge distillation loss
                if self.old_model is not None:
                    with torch.no_grad():
                        old_outputs = self.old_model(data)
                    
                    # Distillation loss using KL divergence
                    distillation_loss = F.kl_div(
                        F.log_softmax(outputs / self.T, dim=1),
                        F.softmax(old_outputs / self.T, dim=1),
                        reduction='batchmean'
                    ) * (self.T ** 2)
                    
                    total_loss += self.alpha * distillation_loss
                
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
        
        # Save current model as old model for next task
        self.old_model = copy.deepcopy(self.model)

# Progressive Neural Networks
class ProgressiveColumn(nn.Module):
    """Single column in progressive neural network"""
    
    def __init__(self, input_size, hidden_sizes, output_size, prev_columns=None):
        super().__init__()
        
        self.prev_columns = prev_columns or []
        
        # Build layers
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        
        for i in range(len(sizes) - 1):
            layer_input_size = sizes[i]
            
            # Add lateral connections from previous columns
            if i > 0:  # Not input layer
                for prev_col in self.prev_columns:
                    if i - 1 < len(prev_col.layers):
                        layer_input_size += hidden_sizes[i-1]
            
            self.layers.append(nn.Linear(layer_input_size, sizes[i + 1]))
    
    def forward(self, x, prev_activations=None):
        """Forward pass with lateral connections"""
        activations = []
        current = x
        
        for i, layer in enumerate(self.layers):
            # Add lateral connections
            if i > 0 and prev_activations:
                lateral_inputs = []
                for j, prev_acts in enumerate(prev_activations):
                    if i - 1 < len(prev_acts):
                        lateral_inputs.append(prev_acts[i - 1])
                
                if lateral_inputs:
                    current = torch.cat([current] + lateral_inputs, dim=1)
            
            current = layer(current)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                current = F.relu(current)
            
            activations.append(current)
        
        return current, activations

class ProgressiveNeuralNetwork(nn.Module):
    """Progressive Neural Network for continual learning"""
    
    def __init__(self, input_size, hidden_sizes, output_sizes):
        super().__init__()
        self.columns = nn.ModuleList()
        self.output_sizes = output_sizes
        self.current_task = 0
        
        # Add first column
        self.add_column(input_size, hidden_sizes, output_sizes[0])
    
    def add_column(self, input_size, hidden_sizes, output_size):
        """Add new column for new task"""
        prev_columns = list(self.columns)
        new_column = ProgressiveColumn(input_size, hidden_sizes, output_size, prev_columns)
        self.columns.append(new_column)
    
    def forward(self, x, task_id=None):
        """Forward pass through specific task column"""
        if task_id is None:
            task_id = self.current_task
        
        # Get activations from previous columns
        prev_activations = []
        for i in range(task_id):
            _, activations = self.columns[i](x, prev_activations)
            prev_activations.append(activations)
        
        # Forward through target column
        output, _ = self.columns[task_id](x, prev_activations)
        return output

# Memory-based Continual Learning
class ExperienceReplay:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, memory_size=1000):
        self.memory_size = memory_size
        self.memory = []
        self.position = 0
    
    def add(self, data, targets, task_id):
        """Add experience to memory"""
        experience = (data.cpu(), targets.cpu(), task_id)
        
        if len(self.memory) < self.memory_size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            self.position = (self.position + 1) % self.memory_size
    
    def sample(self, batch_size, device='cuda'):
        """Sample batch from memory"""
        if len(self.memory) == 0:
            return None
        
        batch_size = min(batch_size, len(self.memory))
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        batch_data = []
        batch_targets = []
        batch_tasks = []
        
        for idx in indices:
            data, targets, task_id = self.memory[idx]
            batch_data.append(data)
            batch_targets.append(targets)
            batch_tasks.append(task_id)
        
        return (torch.stack(batch_data).to(device),
                torch.stack(batch_targets).to(device),
                batch_tasks)

class ERTrainer:
    """Experience Replay trainer"""
    
    def __init__(self, model, memory_size=1000, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.memory = ExperienceReplay(memory_size)
        self.current_task = 0
    
    def train_task(self, train_loader, epochs=10, lr=0.001, replay_ratio=0.5):
        """Train with experience replay"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Add current batch to memory
                self.memory.add(data, targets, self.current_task)
                
                optimizer.zero_grad()
                
                # Train on current batch
                outputs = self.model(data)
                current_loss = criterion(outputs, targets)
                
                total_loss = current_loss
                
                # Train on replayed experiences
                if np.random.random() < replay_ratio and len(self.memory.memory) > 0:
                    replay_data, replay_targets, _ = self.memory.sample(data.size(0), self.device)
                    replay_outputs = self.model(replay_data)
                    replay_loss = criterion(replay_outputs, replay_targets)
                    total_loss += replay_loss
                
                total_loss.backward()
                optimizer.step()
                
                if batch_idx % 50 == 0:
                    print(f'Task {self.current_task + 1}, Epoch {epoch + 1}, '
                          f'Batch {batch_idx}, Loss: {total_loss.item():.4f}')
        
        self.current_task += 1

# Sample Models and Data
class SimpleCNN(nn.Module):
    """Simple CNN for continual learning experiments"""
    
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
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

class TaskDataset(Dataset):
    """Dataset for continual learning tasks"""
    
    def __init__(self, task_id, size=500, input_shape=(1, 28, 28), num_classes=5):
        self.task_id = task_id
        self.size = size
        
        # Generate task-specific data
        np.random.seed(task_id * 42)  # Different seed per task
        torch.manual_seed(task_id * 42)
        
        # Create task-specific pattern
        base_pattern = torch.randn(input_shape) * 0.3
        
        self.data = []
        self.targets = []
        
        for i in range(size):
            # Add task-specific noise and pattern
            sample = base_pattern + torch.randn(input_shape) * 0.5
            sample = torch.clamp(sample, 0, 1)
            
            target = i % num_classes
            
            self.data.append(sample)
            self.targets.append(target)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Evaluation Functions
def evaluate_on_all_tasks(model, task_datasets, device='cuda'):
    """Evaluate model on all seen tasks"""
    model.eval()
    task_accuracies = []
    
    for task_id, dataset in enumerate(task_datasets):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        task_accuracies.append(accuracy)
        print(f'Task {task_id + 1} accuracy: {accuracy:.2f}%')
    
    return task_accuracies

def compute_forgetting_metric(accuracies_matrix):
    """Compute average forgetting metric"""
    if len(accuracies_matrix) < 2:
        return 0.0
    
    forgetting = 0.0
    count = 0
    
    for i in range(len(accuracies_matrix) - 1):
        for j in range(i + 1, len(accuracies_matrix)):
            # Forgetting = max accuracy - current accuracy
            max_acc = max(accuracies_matrix[k][i] for k in range(i, j + 1))
            current_acc = accuracies_matrix[j][i]
            forgetting += max(0, max_acc - current_acc)
            count += 1
    
    return forgetting / count if count > 0 else 0.0

if __name__ == "__main__":
    print("Continual Learning Techniques")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create task datasets
    num_tasks = 3
    task_datasets = []
    
    for task_id in range(num_tasks):
        dataset = TaskDataset(task_id, size=200, num_classes=5)
        task_datasets.append(dataset)
    
    print(f"Created {num_tasks} tasks with {len(task_datasets[0])} samples each")
    
    # Test EWC
    print("\n1. Testing Elastic Weight Consolidation (EWC)")
    print("-" * 45)
    
    model = SimpleCNN(input_channels=1, num_classes=5)
    ewc_trainer = EWCTrainer(model, device)
    
    all_accuracies = []
    
    for task_id, dataset in enumerate(task_datasets):
        print(f"\nTraining on Task {task_id + 1}")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train on current task
        ewc_trainer.train_task(dataloader, epochs=3, lr=0.001)
        
        # Evaluate on all tasks seen so far
        task_accuracies = evaluate_on_all_tasks(model, task_datasets[:task_id + 1], device)
        all_accuracies.append(task_accuracies)
    
    ewc_forgetting = compute_forgetting_metric(all_accuracies)
    print(f"\nEWC Average Forgetting: {ewc_forgetting:.2f}%")
    
    # Test Learning without Forgetting
    print("\n2. Testing Learning without Forgetting (LwF)")
    print("-" * 40)
    
    model = SimpleCNN(input_channels=1, num_classes=5)
    lwf_trainer = LwFTrainer(model, device, alpha=1.0, T=2.0)
    
    lwf_accuracies = []
    
    for task_id, dataset in enumerate(task_datasets):
        print(f"\nTraining on Task {task_id + 1}")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train on current task
        lwf_trainer.train_task(dataloader, epochs=3, lr=0.001)
        
        # Evaluate on all tasks
        task_accuracies = evaluate_on_all_tasks(model, task_datasets[:task_id + 1], device)
        lwf_accuracies.append(task_accuracies)
    
    lwf_forgetting = compute_forgetting_metric(lwf_accuracies)
    print(f"\nLwF Average Forgetting: {lwf_forgetting:.2f}%")
    
    # Test Experience Replay
    print("\n3. Testing Experience Replay")
    print("-" * 30)
    
    model = SimpleCNN(input_channels=1, num_classes=5)
    er_trainer = ERTrainer(model, memory_size=500, device=device)
    
    er_accuracies = []
    
    for task_id, dataset in enumerate(task_datasets):
        print(f"\nTraining on Task {task_id + 1}")
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train with experience replay
        er_trainer.train_task(dataloader, epochs=3, lr=0.001, replay_ratio=0.5)
        
        # Evaluate on all tasks
        task_accuracies = evaluate_on_all_tasks(model, task_datasets[:task_id + 1], device)
        er_accuracies.append(task_accuracies)
    
    er_forgetting = compute_forgetting_metric(er_accuracies)
    print(f"\nER Average Forgetting: {er_forgetting:.2f}%")
    
    # Test Progressive Neural Networks
    print("\n4. Testing Progressive Neural Networks")
    print("-" * 35)
    
    # Create progressive network
    input_size = 28 * 28
    hidden_sizes = [128, 64]
    output_sizes = [5] * num_tasks
    
    prog_net = ProgressiveNeuralNetwork(input_size, hidden_sizes, output_sizes).to(device)
    
    prog_accuracies = []
    
    for task_id, dataset in enumerate(task_datasets):
        print(f"\nTraining on Task {task_id + 1}")
        
        # Add new column for new task (except first)
        if task_id > 0:
            prog_net.add_column(input_size, hidden_sizes, output_sizes[task_id])
        
        prog_net.current_task = task_id
        
        # Train current column
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(prog_net.columns[task_id].parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(3):
            for batch_idx, (data, targets) in enumerate(dataloader):
                data = data.view(data.size(0), -1).to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = prog_net(data, task_id)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate on all tasks
        task_accuracies = []
        for eval_task_id in range(task_id + 1):
            eval_dataset = task_datasets[eval_task_id]
            eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in eval_loader:
                    data = data.view(data.size(0), -1).to(device)
                    targets = targets.to(device)
                    outputs = prog_net(data, eval_task_id)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = 100. * correct / total
            task_accuracies.append(accuracy)
            print(f'Task {eval_task_id + 1} accuracy: {accuracy:.2f}%')
        
        prog_accuracies.append(task_accuracies)
    
    prog_forgetting = compute_forgetting_metric(prog_accuracies)
    print(f"\nProgressive NN Average Forgetting: {prog_forgetting:.2f}%")
    
    # Summary comparison
    print("\n5. Method Comparison")
    print("-" * 20)
    print(f"EWC Forgetting: {ewc_forgetting:.2f}%")
    print(f"LwF Forgetting: {lwf_forgetting:.2f}%")
    print(f"Experience Replay Forgetting: {er_forgetting:.2f}%")
    print(f"Progressive NN Forgetting: {prog_forgetting:.2f}%")
    
    print("\nContinual learning demonstrations completed!") 