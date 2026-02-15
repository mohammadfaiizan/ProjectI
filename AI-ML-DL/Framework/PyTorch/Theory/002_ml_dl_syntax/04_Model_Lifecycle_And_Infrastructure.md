# Model Lifecycle and Infrastructure

## Table of Contents
1. [Model Parameters](#model-parameters)
2. [Forward and Backward Pass](#forward-and-backward-pass)
3. [Training and Evaluation Modes](#training-and-evaluation-modes)
4. [Checkpointing and State Dict](#checkpointing-and-state-dict)
5. [Device Movement](#device-movement)
6. [Distributed Training](#distributed-training)
7. [Mixed Precision Training](#mixed-precision-training)

---

## Model Parameters

### Accessing Parameters

Every `nn.Module` tracks its parameters (learnable weights) and buffers (non-learnable state).

```python
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### nn.Parameter vs register_buffer

| | nn.Parameter | register_buffer |
|---|---|---|
| In `parameters()` | Yes | No |
| In `state_dict()` | Yes | Yes |
| `requires_grad` | True (default) | False (default) |
| Moved by `.to(device)` | Yes | Yes |
| Use case | Learnable weights | Running stats, fixed tensors |

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 5))
        self.register_buffer('running_mean', torch.zeros(5))
```

### Freezing Parameters

```python
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
```

### Unfreezing (Fine-Tuning)

```python
for param in model.backbone.parameters():
    param.requires_grad = True
```

### Parameter Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')

model.apply(init_weights)
```

---

## Forward and Backward Pass

### Standard Training Step

```python
model.train()
optimizer.zero_grad()

output = model(input_data)
loss = criterion(output, targets)
loss.backward()
optimizer.step()
```

### Step Breakdown

| Step | What Happens |
|------|-------------|
| `model(input)` | Calls `forward()`, builds computation graph |
| `criterion(output, target)` | Computes scalar loss, extends graph |
| `loss.backward()` | Traverses graph in reverse, computes `.grad` for all leaf tensors |
| `optimizer.step()` | Updates parameters using `.grad` values |
| `optimizer.zero_grad()` | Clears `.grad` to prevent accumulation |

### Multiple Losses

```python
loss_cls = criterion_cls(output_cls, targets_cls)
loss_reg = criterion_reg(output_reg, targets_reg)
total_loss = loss_cls + 0.5 * loss_reg
total_loss.backward()
```

### Gradient-Free Inference

```python
model.eval()
with torch.no_grad():
    predictions = model(test_input)
```

---

## Training and Evaluation Modes

### model.train() vs model.eval()

| Behavior | train() | eval() |
|----------|---------|--------|
| Dropout | Active (random zeroing) | Disabled (identity) |
| BatchNorm | Uses batch statistics, updates running stats | Uses running statistics |
| Other layers | No change | No change |

```python
model.train()
train_output = model(train_input)

model.eval()
with torch.no_grad():
    val_output = model(val_input)
```

### Checking Mode

```python
print(model.training)    # True or False
```

### Common Mistake

Forgetting to switch back to `train()` after validation:

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        train_step(model, batch)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            validate_step(model, batch)
```

---

## Checkpointing and State Dict

### state_dict

A Python dictionary mapping parameter/buffer names to tensors. This is the recommended way to save models.

```python
model.state_dict()
# OrderedDict([('linear.weight', tensor(...)), ('linear.bias', tensor(...))])
```

### Save and Load Model

```python
torch.save(model.state_dict(), 'model.pth')

model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Full Training Checkpoint

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'train_loss_history': train_losses,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
```

### Resuming Training

```python
checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Partial Loading (Transfer Learning)

```python
pretrained = torch.load('pretrained.pth')
model_dict = model.state_dict()

pretrained_filtered = {k: v for k, v in pretrained.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(pretrained_filtered)
model.load_state_dict(model_dict)
```

### strict Parameter

```python
model.load_state_dict(state_dict, strict=False)
```

`strict=False` ignores missing and unexpected keys. Useful when architectures differ slightly.

---

## Device Movement

### Device Selection

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda:0')
device = torch.device('cuda:1')
device = torch.device('mps')        # Apple Silicon
```

### Moving Models and Data

```python
model = model.to(device)
input_data = input_data.to(device)
targets = targets.to(device)
```

### Device-Agnostic Code Pattern

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)

for data, target in dataloader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
```

### Pinned Memory for Faster Transfers

```python
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)

for data, target in dataloader:
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
```

### Loading Checkpoints Across Devices

```python
model_state = torch.load('model.pth', map_location='cpu')
model_state = torch.load('model.pth', map_location='cuda:0')
model_state = torch.load('model.pth', map_location=device)
```

### Multi-GPU with DataParallel

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
```

Access the underlying model:

```python
actual_model = model.module if isinstance(model, nn.DataParallel) else model
```

---

## Distributed Training

### DistributedDataParallel (DDP)

DDP is the recommended approach for multi-GPU training. Each process runs on one GPU with its own model replica.

### Setup

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
```

### Model Wrapping

```python
model = MyModel().to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

### DistributedSampler

```python
from torch.utils.data import DistributedSampler

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for data, target in dataloader:
        train_step(ddp_model, data, target)
```

### Backend Options

| Backend | GPU | CPU | Notes |
|---------|-----|-----|-------|
| `nccl` | Yes | No | Best for GPU, NVIDIA only |
| `gloo` | Yes | Yes | Cross-platform |
| `mpi` | Yes | Yes | Requires MPI installation |

### Launch Command

```bash
torchrun --nproc_per_node=4 train.py
```

---

## Mixed Precision Training

Mixed precision uses `float16` for forward/backward computation and `float32` for parameter updates. This reduces memory usage and increases throughput on GPUs with Tensor Cores.

### Automatic Mixed Precision (AMP)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda'):
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### How It Works

| Component | Role |
|-----------|------|
| `autocast` | Automatically casts operations to float16 where safe |
| `GradScaler` | Scales loss to prevent underflow in float16 gradients |
| `scaler.scale(loss)` | Multiplies loss by scale factor |
| `scaler.step(optimizer)` | Unscales gradients, skips step if NaN/Inf |
| `scaler.update()` | Adjusts scale factor for next iteration |

### Operations Under autocast

- **float16**: matmul, conv, linear, BMM
- **float32**: reductions (sum, softmax, loss functions), norms, exp, log

### Gradient Clipping with AMP

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### BFloat16

```python
with autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(data)
```

`bfloat16` has the same exponent range as float32 (avoiding overflow) with reduced precision. Supported on Ampere+ GPUs and TPUs.

### CPU Mixed Precision

```python
with autocast(device_type='cpu', dtype=torch.bfloat16):
    output = model(data)
```
