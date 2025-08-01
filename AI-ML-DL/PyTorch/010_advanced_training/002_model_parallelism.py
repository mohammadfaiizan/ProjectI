import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from typing import List, Dict, Optional, Tuple
import time
import math

# Basic Model Parallelism
class ModelParallelCNN(nn.Module):
    """CNN with layers split across multiple GPUs"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # First part on GPU 0
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1).to('cuda:0')
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1).to('cuda:0')
        self.pool1 = nn.MaxPool2d(2, 2).to('cuda:0')
        
        # Second part on GPU 1 (if available)
        device1 = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1).to(device1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1).to(device1)
        self.pool2 = nn.MaxPool2d(2, 2).to(device1)
        
        # Final layers on GPU 1
        self.fc1 = nn.Linear(512 * 8 * 8, 1024).to(device1)
        self.fc2 = nn.Linear(1024, num_classes).to(device1)
        
        self.device1 = device1
    
    def forward(self, x):
        # Forward through first part (GPU 0)
        x = x.to('cuda:0')
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Move to second GPU
        x = x.to(self.device1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Pipeline Parallelism
class PipelineStage(nn.Module):
    """Base class for pipeline stages"""
    
    def __init__(self, stage_id: int, device: str):
        super().__init__()
        self.stage_id = stage_id
        self.device = device
    
    def forward(self, x):
        raise NotImplementedError

class ConvStage(PipelineStage):
    """Convolutional stage for pipeline"""
    
    def __init__(self, stage_id: int, device: str, in_channels: int, out_channels: int):
        super().__init__(stage_id, device)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1).to(device)
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.bn1 = nn.BatchNorm2d(out_channels).to(device)
        self.bn2 = nn.BatchNorm2d(out_channels).to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class ClassifierStage(PipelineStage):
    """Classification stage for pipeline"""
    
    def __init__(self, stage_id: int, device: str, input_size: int, num_classes: int):
        super().__init__(stage_id, device)
        
        self.fc1 = nn.Linear(input_size, 512).to(device)
        self.fc2 = nn.Linear(512, num_classes).to(device)
        self.dropout = nn.Dropout(0.5).to(device)
    
    def forward(self, x):
        x = x.to(self.device)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PipelineParallelModel(nn.Module):
    """Pipeline parallel model with multiple stages"""
    
    def __init__(self, stages: List[PipelineStage]):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.num_stages = len(stages)
    
    def forward(self, x):
        # Sequential forward through all stages
        for stage in self.stages:
            x = stage(x)
        return x
    
    def pipeline_forward(self, batches: List[torch.Tensor]):
        """Pipeline forward with multiple micro-batches"""
        # Number of micro-batches
        num_microbatches = len(batches)
        
        # Results storage
        stage_outputs = [[] for _ in range(self.num_stages)]
        
        # Pipeline execution
        for step in range(num_microbatches + self.num_stages - 1):
            for stage_idx in range(self.num_stages):
                microbatch_idx = step - stage_idx
                
                if 0 <= microbatch_idx < num_microbatches:
                    if stage_idx == 0:
                        # First stage processes input
                        input_data = batches[microbatch_idx]
                        output = self.stages[stage_idx](input_data)
                    else:
                        # Subsequent stages process output from previous stage
                        if len(stage_outputs[stage_idx - 1]) > microbatch_idx:
                            input_data = stage_outputs[stage_idx - 1][microbatch_idx]
                            output = self.stages[stage_idx](input_data)
                        else:
                            continue
                    
                    stage_outputs[stage_idx].append(output)
        
        # Return final outputs
        return stage_outputs[-1]

# Tensor Parallelism for Transformers
class ParallelLinear(nn.Module):
    """Linear layer with tensor parallelism"""
    
    def __init__(self, input_size: int, output_size: int, world_size: int, rank: int):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Each rank gets a slice of the output dimension
        self.output_size_per_rank = output_size // world_size
        
        self.linear = nn.Linear(input_size, self.output_size_per_rank)
    
    def forward(self, x):
        # Each rank computes its portion
        output = self.linear(x)
        
        # All-gather outputs from all ranks
        if dist.is_initialized():
            gathered_outputs = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(gathered_outputs, output)
            # Concatenate along the feature dimension
            output = torch.cat(gathered_outputs, dim=-1)
        
        return output

class ParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism"""
    
    def __init__(self, d_model: int, num_heads: int, world_size: int, rank: int):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % world_size == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        
        # Each rank handles a subset of heads
        self.num_heads_per_rank = num_heads // world_size
        self.d_k = d_model // num_heads
        
        # Linear projections (only for assigned heads)
        self.query = nn.Linear(d_model, self.num_heads_per_rank * self.d_k)
        self.key = nn.Linear(d_model, self.num_heads_per_rank * self.d_k)
        self.value = nn.Linear(d_model, self.num_heads_per_rank * self.d_k)
        
        # Output projection
        self.output = nn.Linear(self.num_heads_per_rank * self.d_k, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads_per_rank, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads_per_rank, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads_per_rank, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads_per_rank * self.d_k
        )
        
        output = self.output(attended)
        
        # All-reduce across ranks to combine outputs
        if dist.is_initialized():
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output

# Large Model Sharding
class ShardedLinear(nn.Module):
    """Linear layer sharded across multiple devices"""
    
    def __init__(self, input_size: int, output_size: int, devices: List[str]):
        super().__init__()
        self.devices = devices
        self.num_shards = len(devices)
        
        # Shard the weight matrix
        self.output_size_per_shard = output_size // self.num_shards
        
        self.shards = nn.ModuleList([
            nn.Linear(input_size, self.output_size_per_shard).to(device)
            for device in devices
        ])
    
    def forward(self, x):
        # Compute each shard on its respective device
        shard_outputs = []
        
        for i, shard in enumerate(self.shards):
            x_shard = x.to(self.devices[i])
            output_shard = shard(x_shard)
            shard_outputs.append(output_shard)
        
        # Move all outputs to the first device and concatenate
        final_device = self.devices[0]
        shard_outputs = [out.to(final_device) for out in shard_outputs]
        output = torch.cat(shard_outputs, dim=-1)
        
        return output

class ShardedTransformerBlock(nn.Module):
    """Transformer block with sharded parameters"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, devices: List[str]):
        super().__init__()
        self.devices = devices
        self.d_model = d_model
        
        # Layer normalization on first device
        self.ln1 = nn.LayerNorm(d_model).to(devices[0])
        self.ln2 = nn.LayerNorm(d_model).to(devices[0])
        
        # Sharded attention
        self.attention = ShardedLinear(d_model, d_model * 3, devices)  # Q, K, V
        self.attention_output = ShardedLinear(d_model, d_model, devices)
        
        # Sharded feed-forward
        self.ff1 = ShardedLinear(d_model, d_ff, devices)
        self.ff2 = ShardedLinear(d_ff, d_model, devices)
    
    def forward(self, x):
        # Move input to first device for layer norm
        x = x.to(self.devices[0])
        
        # Self-attention
        residual = x
        x = self.ln1(x)
        
        # Simplified attention (for demonstration)
        qkv = self.attention(x)
        attention_out = self.attention_output(qkv)
        
        # Move back to first device for residual connection
        attention_out = attention_out.to(self.devices[0])
        x = residual + attention_out
        
        # Feed-forward
        residual = x
        x = self.ln2(x)
        
        x = self.ff1(x)
        x = F.relu(x)
        x = self.ff2(x)
        
        # Move back to first device for final residual
        x = x.to(self.devices[0])
        x = residual + x
        
        return x

# Model Parallel Trainer
class ModelParallelTrainer:
    """Trainer for model parallel models"""
    
    def __init__(self, model, loss_device='cuda:0'):
        self.model = model
        self.loss_device = loss_device
        self.criterion = nn.CrossEntropyLoss().to(loss_device)
    
    def setup_optimizer(self, lr=0.001):
        """Setup optimizer for model parallel model"""
        # Collect parameters from all devices
        all_params = []
        for name, param in self.model.named_parameters():
            all_params.append(param)
        
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
    
    def train_step(self, data, targets):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(data)
        
        # Move outputs and targets to loss device
        outputs = outputs.to(self.loss_device)
        targets = targets.to(self.loss_device)
        
        # Compute loss
        loss = self.criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            loss = self.train_step(data, targets)
            total_loss += loss
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss:.4f}')
        
        return total_loss / len(dataloader)

# Gradient Checkpointing for Memory Efficiency
class CheckpointedLinear(nn.Module):
    """Linear layer with gradient checkpointing"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.linear, x)

class MemoryEfficientModel(nn.Module):
    """Model with gradient checkpointing for memory efficiency"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            layers.append(CheckpointedLinear(in_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# Utility Functions
def measure_memory_usage():
    """Measure GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            print(f"GPU {i}: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

# Sample Dataset
class SampleDataset(Dataset):
    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

if __name__ == "__main__":
    print("Model Parallelism with PyTorch")
    print("=" * 40)
    
    # Check available devices
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available. Using CPU for demonstration.")
        device_list = ['cpu']
    else:
        device_list = [f'cuda:{i}' for i in range(min(num_gpus, 2))]
    
    print(f"Using devices: {device_list}")
    
    # Test basic model parallelism
    print("\n1. Basic Model Parallelism")
    print("-" * 30)
    
    if num_gpus >= 2:
        # Test model parallel CNN
        model_parallel_cnn = ModelParallelCNN(num_classes=10)
        
        # Count parameters
        total_params, _ = count_parameters(model_parallel_cnn)
        
        # Test forward pass
        sample_input = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            output = model_parallel_cnn(sample_input)
            print(f"Model parallel CNN output shape: {output.shape}")
        
        print("Basic model parallelism test completed")
    else:
        print("Skipping basic model parallelism (requires 2+ GPUs)")
    
    # Test pipeline parallelism
    print("\n2. Pipeline Parallelism")
    print("-" * 30)
    
    # Create pipeline stages
    stages = [
        ConvStage(0, device_list[0], 3, 64),
        ConvStage(1, device_list[-1], 64, 128),
        ClassifierStage(2, device_list[-1], 128 * 8 * 8, 10)
    ]
    
    pipeline_model = PipelineParallelModel(stages)
    
    # Test forward pass
    sample_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = pipeline_model(sample_input)
        print(f"Pipeline model output shape: {output.shape}")
    
    # Test pipeline forward with micro-batches
    micro_batches = [torch.randn(1, 3, 32, 32) for _ in range(4)]
    with torch.no_grad():
        pipeline_outputs = pipeline_model.pipeline_forward(micro_batches)
        print(f"Pipeline outputs: {len(pipeline_outputs)} micro-batches")
    
    # Test sharded linear layer
    print("\n3. Sharded Linear Layer")
    print("-" * 30)
    
    sharded_linear = ShardedLinear(256, 512, device_list)
    sample_input = torch.randn(4, 256)
    
    with torch.no_grad():
        output = sharded_linear(sample_input)
        print(f"Sharded linear output shape: {output.shape}")
    
    # Test memory-efficient model
    print("\n4. Memory-Efficient Model")
    print("-" * 30)
    
    memory_model = MemoryEfficientModel(784, 512, 5, 10).to(device_list[0])
    count_parameters(memory_model)
    
    # Test forward pass
    sample_input = torch.randn(8, 784).to(device_list[0])
    output = memory_model(sample_input)
    print(f"Memory-efficient model output shape: {output.shape}")
    
    # Test training
    print("\n5. Model Parallel Training")
    print("-" * 30)
    
    # Create simple dataset
    dataset = SampleDataset(size=100)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Test training with pipeline model
    trainer = ModelParallelTrainer(pipeline_model, loss_device=device_list[-1])
    trainer.setup_optimizer(lr=0.001)
    
    # Train for a few steps
    pipeline_model.train()
    for batch_idx, (data, targets) in enumerate(dataloader):
        if batch_idx >= 3:  # Only a few batches for demo
            break
        
        loss = trainer.train_step(data, targets)
        print(f"Training step {batch_idx + 1}, Loss: {loss:.4f}")
    
    # Memory usage
    print("\n6. Memory Usage")
    print("-" * 30)
    measure_memory_usage()
    
    print("\nModel parallelism demonstrations completed!") 