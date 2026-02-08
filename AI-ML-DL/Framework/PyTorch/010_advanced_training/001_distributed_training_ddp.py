import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import os
import tempfile
from typing import Tuple, Optional
import time
import argparse

# Distributed Training Setup
def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    print(f"Distributed training initialized: rank {rank}/{world_size}")

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

# Sample Model for Distributed Training
class DistributedModel(nn.Module):
    """Sample CNN model for distributed training"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Sample Dataset
class SyntheticDataset(Dataset):
    """Synthetic dataset for distributed training demonstration"""
    
    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.size = size
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Distributed Trainer
class DistributedTrainer:
    """Trainer class for distributed training with DDP"""
    
    def __init__(self, model, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Move model to device
        self.model = model.to(device)
        
        # Wrap model with DDP
        self.ddp_model = DDP(self.model, device_ids=[rank])
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
    def create_dataloader(self, dataset, batch_size=32, shuffle=True):
        """Create distributed dataloader"""
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.rank,
            shuffle=shuffle
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader, sampler
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.ddp_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.ddp_model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient synchronization happens automatically with DDP
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0 and self.rank == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Synchronize metrics across processes
        total_loss_tensor = torch.tensor(total_loss).to(self.device)
        correct_tensor = torch.tensor(correct).to(self.device)
        total_tensor = torch.tensor(total).to(self.device)
        
        # All-reduce to get global statistics
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_tensor.item() / (len(dataloader) * self.world_size)
        accuracy = 100. * correct_tensor.item() / total_tensor.item()
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validation loop"""
        self.ddp_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.ddp_model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Synchronize validation metrics
        total_loss_tensor = torch.tensor(total_loss).to(self.device)
        correct_tensor = torch.tensor(correct).to(self.device)
        total_tensor = torch.tensor(total).to(self.device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_tensor.item() / (len(dataloader) * self.world_size)
        accuracy = 100. * correct_tensor.item() / total_tensor.item()
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, filepath):
        """Save model checkpoint (only from rank 0)"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.ddp_model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ddp_model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

# Distributed Training Function
def train_distributed(rank, world_size, epochs=5):
    """Main distributed training function"""
    
    # Setup distributed training
    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    try:
        # Create model
        model = DistributedModel(num_classes=10)
        
        # Create trainer
        trainer = DistributedTrainer(model, rank, world_size, device)
        
        # Create datasets
        train_dataset = SyntheticDataset(size=2000)
        val_dataset = SyntheticDataset(size=500)
        
        # Create dataloaders
        train_loader, train_sampler = trainer.create_dataloader(train_dataset, batch_size=32)
        val_loader, _ = trainer.create_dataloader(val_dataset, batch_size=32, shuffle=False)
        
        # Training loop
        for epoch in range(epochs):
            # Set epoch for distributed sampler (important for shuffling)
            train_sampler.set_epoch(epoch)
            
            # Train
            train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = trainer.validate(val_loader)
            
            # Step scheduler
            trainer.scheduler.step()
            
            # Print metrics (only from rank 0)
            if rank == 0:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print('-' * 50)
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
            
            # Synchronize all processes
            dist.barrier()
    
    finally:
        # Cleanup
        cleanup_distributed()

# Advanced DDP Features
class AdvancedDDPTrainer:
    """Advanced DDP trainer with additional features"""
    
    def __init__(self, model, rank, world_size, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        
        # Model setup
        self.model = model.to(device)
        
        # Advanced DDP configuration
        self.ddp_model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,  # For models with dynamic graphs
            gradient_as_bucket_view=True,  # Memory optimization
            broadcast_buffers=False  # Only broadcast parameters, not buffers
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=0.001)
    
    def train_with_gradient_sync_control(self, dataloader):
        """Training with gradient synchronization control"""
        self.ddp_model.train()
        
        accumulation_steps = 4  # Accumulate gradients over 4 steps
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Disable gradient synchronization for accumulation steps
            if (batch_idx + 1) % accumulation_steps != 0:
                with self.ddp_model.no_sync():
                    outputs = self.ddp_model(data)
                    loss = self.criterion(outputs, targets) / accumulation_steps
                    loss.backward()
            else:
                # Enable synchronization on the last accumulation step
                outputs = self.ddp_model(data)
                loss = self.criterion(outputs, targets) / accumulation_steps
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
    
    def broadcast_object(self, obj):
        """Broadcast Python object to all processes"""
        if self.rank == 0:
            # Convert object to tensor for broadcasting
            obj_list = [obj]
        else:
            obj_list = [None]
        
        # Broadcast from rank 0 to all other ranks
        dist.broadcast_object_list(obj_list, src=0)
        return obj_list[0]
    
    def all_gather_metrics(self, local_metric):
        """Gather metrics from all processes"""
        # Convert to tensor
        local_tensor = torch.tensor(local_metric).to(self.device)
        
        # Gather from all processes
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, local_tensor)
        
        # Convert back to list
        return [tensor.item() for tensor in gathered_tensors]

# Utility Functions for Distributed Training
def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def print_distributed_info():
    """Print distributed training information"""
    if dist.is_available():
        print(f"Distributed training available: {dist.is_available()}")
        
        if dist.is_initialized():
            print(f"World size: {dist.get_world_size()}")
            print(f"Current rank: {dist.get_rank()}")
            print(f"Backend: {dist.get_backend()}")
    else:
        print("Distributed training not available")

def launch_distributed_training():
    """Launch distributed training using torch.multiprocessing"""
    world_size = torch.cuda.device_count()
    print(f"Launching distributed training on {world_size} GPUs")
    
    if world_size > 1:
        mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Single GPU training")
        train_distributed(0, 1)

# FSDP (Fully Sharded Data Parallel) Example
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    
    class FSDPTrainer:
        """Trainer using Fully Sharded Data Parallel"""
        
        def __init__(self, model, rank, world_size, device):
            self.rank = rank
            self.world_size = world_size
            self.device = device
            
            # FSDP auto wrap policy
            auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1000)
            
            # Wrap model with FSDP
            self.model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=None,  # Can enable mixed precision
                sharding_strategy=None,  # Default sharding strategy
                device_id=rank,
            )
            
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        def train_step(self, data, targets):
            """Single training step with FSDP"""
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
    
    FSDP_AVAILABLE = True
    
except ImportError:
    FSDP_AVAILABLE = False
    print("FSDP not available in this PyTorch version")

if __name__ == "__main__":
    print("Distributed Training with PyTorch DDP")
    print("=" * 50)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. Distributed training requires GPUs.")
        exit(1)
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Print distributed info
    print_distributed_info()
    
    # Calculate model size
    sample_model = DistributedModel()
    model_size = calculate_model_size(sample_model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Test single GPU training first
    if torch.cuda.device_count() == 1:
        print("\nRunning single GPU training...")
        device = torch.device('cuda:0')
        
        # Create simple trainer for single GPU
        model = DistributedModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create data
        dataset = SyntheticDataset(size=100)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(2):
            total_loss = 0
            for batch_idx, (data, targets) in enumerate(dataloader):
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    else:
        print("\nLaunching distributed training...")
        # Uncomment the line below to run actual distributed training
        # launch_distributed_training()
        print("Distributed training launch disabled for demonstration")
    
    print("\nDistributed training setup completed!") 