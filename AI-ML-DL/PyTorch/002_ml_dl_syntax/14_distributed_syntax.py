#!/usr/bin/env python3
"""PyTorch Distributed Training Syntax - Basic distributed training setup"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os

print("=== Distributed Training Concepts ===")

# Check distributed availability
print(f"Distributed available: {torch.distributed.is_available()}")
print(f"NCCL backend available: {torch.distributed.is_nccl_available()}")
print(f"Gloo backend available: {torch.distributed.is_gloo_available()}")

# Environment variables for distributed training
print("\nKey environment variables for distributed training:")
print("- MASTER_ADDR: Address of rank 0 process")
print("- MASTER_PORT: Port of rank 0 process")
print("- WORLD_SIZE: Total number of processes")
print("- RANK: Process rank (0 to WORLD_SIZE-1)")
print("- LOCAL_RANK: Local rank within node")

print("\n=== Data Parallel (Single Machine, Multiple GPUs) ===")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    # DataParallel usage
    model = SimpleModel()
    model = nn.DataParallel(model)
    model = model.cuda()
    
    print(f"DataParallel model device: {next(model.parameters()).device}")
    print(f"DataParallel module: {model.module}")
    
    # Test with multi-GPU batch
    batch_size = 32 * torch.cuda.device_count()  # Scale batch size
    input_data = torch.randn(batch_size, 784, device='cuda')
    output = model(input_data)
    
    print(f"Multi-GPU input: {input_data.shape}")
    print(f"Multi-GPU output: {output.shape}")

else:
    print("Single GPU or CPU - DataParallel not demonstrated")

print("\n=== DistributedDataParallel Setup ===")

def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

# Example distributed training function
def distributed_train_example(rank, world_size):
    """Example distributed training function"""
    print(f"Running on rank {rank} of {world_size}")
    
    # Setup
    setup_distributed(rank, world_size)
    
    # Create model
    model = SimpleModel()
    
    if torch.cuda.is_available():
        model = model.cuda(rank)
        # Wrap with DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[rank]
        )
    
    # Create synthetic data
    data = torch.randn(64, 784)
    target = torch.randint(0, 10, (64,))
    
    if torch.cuda.is_available():
        data = data.cuda(rank)
        target = target.cuda(rank)
    
    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"Rank {rank} - Loss: {loss.item():.4f}")
    
    # Cleanup
    cleanup_distributed()

# Note: Actual multiprocessing would use mp.spawn() in real application
print("Distributed training function defined (requires multiprocessing to run)")

print("\n=== Distributed Sampler ===")

# DistributedSampler for data loading
class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 784)
        self.targets = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

dataset = FakeDataset(1000)

# Create distributed sampler (simulated)
def create_distributed_sampler(dataset, num_replicas=2, rank=0):
    """Create distributed sampler (demonstration)"""
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True
    )
    return sampler

# Example usage
sampler_rank0 = create_distributed_sampler(dataset, num_replicas=2, rank=0)
sampler_rank1 = create_distributed_sampler(dataset, num_replicas=2, rank=1)

print(f"Distributed sampler created for 2 ranks")
print(f"Rank 0 sampler length: {len(sampler_rank0)}")
print(f"Rank 1 sampler length: {len(sampler_rank1)}")

print("\n=== Communication Primitives ===")

# Simulated communication operations (would work in distributed setting)
def communication_examples():
    """Examples of distributed communication operations"""
    
    print("Communication operations (requires distributed setup):")
    print("1. all_reduce - Sum/average tensors across all processes")
    print("2. all_gather - Gather tensors from all processes")
    print("3. broadcast - Send tensor from one process to all")
    print("4. reduce - Reduce tensors to one process")
    print("5. scatter - Scatter tensor chunks to processes")
    
    # Example tensor operations
    if dist.is_initialized():
        # All-reduce example
        tensor = torch.ones(2, 2) * dist.get_rank()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"After all_reduce: {tensor}")
        
        # All-gather example
        tensor_list = [torch.zeros(2, 2) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, tensor)
        print(f"All-gather result: {tensor_list}")
    
    else:
        print("Distributed not initialized - showing syntax only")

communication_examples()

print("\n=== Gradient Synchronization ===")

class DistributedTrainingLoop:
    """Example distributed training loop structure"""
    
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train_step(self, data, target):
        """Single training step with gradient synchronization"""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Backward pass (gradients automatically synchronized in DDP)
        loss.backward()
        
        # Optional: gradient clipping before sync
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            loss = self.train_step(data, target)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss

print("Distributed training loop structure defined")

print("\n=== Model Saving in Distributed Setting ===")

def save_distributed_checkpoint(model, optimizer, epoch, rank, filename):
    """Save checkpoint in distributed setting"""
    # Only save from rank 0 to avoid conflicts
    if rank == 0:
        # For DDP, access the underlying module
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }
        
        torch.save(checkpoint, filename)
        print(f"Rank {rank}: Checkpoint saved to {filename}")
    
    # Synchronize all processes
    if dist.is_initialized():
        dist.barrier()

def load_distributed_checkpoint(model, optimizer, filename):
    """Load checkpoint in distributed setting"""
    # Load checkpoint
    checkpoint = torch.load(filename, map_location='cpu')
    
    # Load model state
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch

print("Distributed checkpoint functions defined")

print("\n=== Mixed Precision with Distributed Training ===")

def distributed_mixed_precision_example():
    """Example of mixed precision with distributed training"""
    from torch.cuda.amp import autocast, GradScaler
    
    class DistributedAMPTrainer:
        def __init__(self, model, optimizer):
            self.model = model
            self.optimizer = optimizer
            self.scaler = GradScaler()
            self.criterion = nn.CrossEntropyLoss()
        
        def train_step(self, data, target):
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Scaled backward pass
            self.scaler.scale(loss).backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            return loss.item()
    
    print("Mixed precision distributed trainer defined")
    return DistributedAMPTrainer

DistributedAMPTrainer = distributed_mixed_precision_example()

print("\n=== Launch Script Example ===")

launch_script = '''
#!/bin/bash

# Example launch script for distributed training

# Single node, multiple GPUs
python -m torch.distributed.launch \\
    --nproc_per_node=4 \\
    --nnodes=1 \\
    --node_rank=0 \\
    --master_addr="localhost" \\
    --master_port=12355 \\
    train_distributed.py

# Multiple nodes example
# Node 0:
# python -m torch.distributed.launch \\
#     --nproc_per_node=4 \\
#     --nnodes=2 \\
#     --node_rank=0 \\
#     --master_addr="192.168.1.1" \\
#     --master_port=12355 \\
#     train_distributed.py

# Node 1:
# python -m torch.distributed.launch \\
#     --nproc_per_node=4 \\
#     --nnodes=2 \\
#     --node_rank=1 \\
#     --master_addr="192.168.1.1" \\
#     --master_port=12355 \\
#     train_distributed.py
'''

print("Launch script example:")
print(launch_script)

print("\n=== Environment Setup ===")

env_setup = '''
# Environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=0  # Different for each process

# For NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# For multi-node training
export NCCL_SOCKET_IFNAME=eth0  # Network interface
export NCCL_IB_DISABLE=1        # Disable InfiniBand if not available
'''

print("Environment setup:")
print(env_setup)

print("\n=== Distributed Training Best Practices ===")

print("Distributed Training Guidelines:")
print("1. Use DistributedDataParallel (DDP) over DataParallel")
print("2. Set different random seeds per process")
print("3. Use DistributedSampler for data loading")
print("4. Save checkpoints only from rank 0")
print("5. Use barriers for synchronization when needed")
print("6. Scale learning rate with number of GPUs")
print("7. Use gradient accumulation for large effective batch sizes")

print("\nPerformance Optimization:")
print("- Use NCCL backend for GPU communication")
print("- Set find_unused_parameters=False in DDP when possible")
print("- Use torch.cuda.amp for mixed precision")
print("- Overlap communication with computation")
print("- Tune batch size per GPU for optimal throughput")

print("\nCommon Issues:")
print("- Hanging processes due to uneven communication")
print("- Different random states across processes")
print("- Model saving conflicts from multiple processes")
print("- Network configuration for multi-node training")
print("- NCCL errors due to environment misconfiguration")

print("\nDebugging Tips:")
print("- Set NCCL_DEBUG=INFO for communication debugging")
print("- Use torch.distributed.barrier() for synchronization")
print("- Check process ranks and world size")
print("- Verify network connectivity between nodes")
print("- Test with single node before scaling to multiple nodes")

print("\n=== Distributed Syntax Complete ===") 