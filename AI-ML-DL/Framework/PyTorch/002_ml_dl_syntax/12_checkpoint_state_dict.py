#!/usr/bin/env python3
"""PyTorch Checkpoint and State Dict - Model saving, loading, state management"""

import torch
import torch.nn as nn
import os
import tempfile

print("=== Basic State Dict Operations ===")

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = SimpleModel()
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Get state dict
state_dict = model.state_dict()
print(f"State dict keys: {len(state_dict.keys())}")
for key in list(state_dict.keys())[:5]:  # Show first 5
    print(f"  {key}: {state_dict[key].shape}")

print("\n=== Saving and Loading State Dict ===")

# Save state dict
torch.save(state_dict, 'model_state_dict.pth')
print("State dict saved to 'model_state_dict.pth'")

# Load state dict
loaded_state_dict = torch.load('model_state_dict.pth')
print(f"Loaded state dict with {len(loaded_state_dict)} keys")

# Create new model and load state
new_model = SimpleModel()
new_model.load_state_dict(loaded_state_dict)
print("State dict loaded into new model")

# Verify models are identical
with torch.no_grad():
    test_input = torch.randn(1, 3, 32, 32)
    output1 = model(test_input)
    output2 = new_model(test_input)
    models_identical = torch.allclose(output1, output2)
    print(f"Models produce identical outputs: {models_identical}")

print("\n=== Complete Model Saving ===")

# Save entire model (architecture + weights)
torch.save(model, 'complete_model.pth')
print("Complete model saved")

# Load complete model
loaded_model = torch.load('complete_model.pth')
print(f"Loaded model type: {type(loaded_model)}")

# Test loaded model
with torch.no_grad():
    output_loaded = loaded_model(test_input)
    models_match = torch.allclose(output1, output_loaded)
    print(f"Loaded model matches original: {models_match}")

print("\n=== Checkpoint with Training State ===")

# Create optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Simulate some training
for epoch in range(5):
    # Simulate training step
    dummy_input = torch.randn(16, 3, 32, 32)
    dummy_target = torch.randint(0, 10, (16,))
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_target)
    loss.backward()
    optimizer.step()
    scheduler.step()

print(f"Training completed, final LR: {scheduler.get_last_lr()[0]:.6f}")

# Create comprehensive checkpoint
checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss.item(),
    'learning_rate': scheduler.get_last_lr()[0]
}

# Save checkpoint
torch.save(checkpoint, 'training_checkpoint.pth')
print("Training checkpoint saved")

print("\n=== Loading Training Checkpoint ===")

# Create fresh model, optimizer, scheduler
resume_model = SimpleModel()
resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=0.001)
resume_scheduler = torch.optim.lr_scheduler.StepLR(resume_optimizer, step_size=10, gamma=0.1)

# Load checkpoint
checkpoint = torch.load('training_checkpoint.pth')

# Restore states
resume_model.load_state_dict(checkpoint['model_state_dict'])
resume_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
resume_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch']
last_loss = checkpoint['loss']

print(f"Resumed from epoch {start_epoch}, loss: {last_loss:.4f}")
print(f"Resumed LR: {resume_scheduler.get_last_lr()[0]:.6f}")

# Verify model state is restored
with torch.no_grad():
    resumed_output = resume_model(test_input)
    state_restored = torch.allclose(output1, resumed_output)
    print(f"Model state correctly restored: {state_restored}")

print("\n=== Partial State Loading ===")

# Load only specific layers
partial_state = {k: v for k, v in loaded_state_dict.items() if 'conv' in k}
print(f"Partial state has {len(partial_state)} keys")

# Create model and load partial state
partial_model = SimpleModel()
partial_model.load_state_dict(partial_state, strict=False)
print("Partial state loaded (strict=False)")

# Check which parameters were loaded
missing_keys, unexpected_keys = partial_model.load_state_dict(partial_state, strict=False)
print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")

print("\n=== State Dict Manipulation ===")

# Modify state dict keys (useful for transfer learning)
def modify_state_dict_keys(state_dict, prefix_to_remove="", prefix_to_add=""):
    """Modify state dict keys"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if prefix_to_remove and key.startswith(prefix_to_remove):
            new_key = key[len(prefix_to_remove):]
        if prefix_to_add:
            new_key = prefix_to_add + new_key
        new_state_dict[new_key] = value
    return new_state_dict

# Example: add prefix to all keys
modified_state = modify_state_dict_keys(state_dict, prefix_to_add="backbone.")
print(f"Modified state dict sample key: {list(modified_state.keys())[0]}")

# Filter state dict for specific layers
def filter_state_dict(state_dict, include_patterns=None, exclude_patterns=None):
    """Filter state dict based on patterns"""
    filtered = {}
    for key, value in state_dict.items():
        include = True
        
        if include_patterns:
            include = any(pattern in key for pattern in include_patterns)
        
        if exclude_patterns and include:
            include = not any(pattern in key for pattern in exclude_patterns)
        
        if include:
            filtered[key] = value
    
    return filtered

# Filter to only conv layers
conv_only_state = filter_state_dict(state_dict, include_patterns=['conv'])
print(f"Conv only state has {len(conv_only_state)} keys")

# Filter excluding batch norm
no_bn_state = filter_state_dict(state_dict, exclude_patterns=['bn'])
print(f"No batch norm state has {len(no_bn_state)} keys")

print("\n=== Model Versioning ===")

# Version-aware model saving
def save_model_with_version(model, optimizer, epoch, version="1.0", metadata=None):
    """Save model with version information"""
    save_dict = {
        'version': version,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metadata': metadata or {}
    }
    
    filename = f'model_v{version}_epoch_{epoch}.pth'
    torch.save(save_dict, filename)
    return filename

# Save versioned model
metadata = {
    'architecture': 'SimpleModel',
    'dataset': 'CIFAR-10',
    'notes': 'Baseline model with dropout'
}

filename = save_model_with_version(
    model, optimizer, epoch=5, version="1.0", metadata=metadata
)
print(f"Versioned model saved as: {filename}")

# Load with version checking
def load_model_with_version_check(filename, expected_version=None):
    """Load model with version checking"""
    checkpoint = torch.load(filename)
    
    version = checkpoint.get('version', 'unknown')
    print(f"Loading model version: {version}")
    
    if expected_version and version != expected_version:
        print(f"Warning: Expected version {expected_version}, got {version}")
    
    metadata = checkpoint.get('metadata', {})
    print(f"Model metadata: {metadata}")
    
    return checkpoint

# Load versioned model
loaded_checkpoint = load_model_with_version_check(filename, expected_version="1.0")

print("\n=== Transfer Learning State Management ===")

# Simulate pre-trained model
class PretrainedModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create and save pretrained model
pretrained_model = PretrainedModel(num_classes=1000)
torch.save(pretrained_model.state_dict(), 'pretrained_weights.pth')

# Load for transfer learning
def transfer_learning_load(pretrained_path, model, freeze_features=True):
    """Load pretrained weights for transfer learning"""
    # Load pretrained weights
    pretrained_state = torch.load(pretrained_path)
    
    # Filter compatible weights
    model_state = model.state_dict()
    compatible_state = {k: v for k, v in pretrained_state.items() 
                       if k in model_state and model_state[k].shape == v.shape}
    
    # Load compatible weights
    model.load_state_dict(compatible_state, strict=False)
    print(f"Loaded {len(compatible_state)} compatible layers")
    
    # Freeze feature layers
    if freeze_features:
        for name, param in model.named_parameters():
            if 'features' in name:
                param.requires_grad = False
        print("Feature layers frozen")

# Create target model for transfer learning
class TargetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Different number of classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

target_model = TargetModel(num_classes=10)
transfer_learning_load('pretrained_weights.pth', target_model, freeze_features=True)

# Check which parameters are trainable
trainable_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in target_model.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params}")

print("\n=== Best Practices for Checkpointing ===")

class CheckpointManager:
    """Advanced checkpoint management"""
    
    def __init__(self, checkpoint_dir='checkpoints', max_checkpoints=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_files = []
    
    def save_checkpoint(self, model, optimizer, epoch, loss, metrics=None):
        """Save checkpoint with automatic cleanup"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {}
        }
        
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, filename)
        
        # Add to list and manage max checkpoints
        self.checkpoint_files.append(filename)
        
        if len(self.checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_files.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        print(f"Checkpoint saved: {filename}")
        return filename
    
    def load_latest_checkpoint(self, model, optimizer):
        """Load the most recent checkpoint"""
        if not self.checkpoint_files:
            print("No checkpoints found")
            return None
        
        latest_checkpoint = self.checkpoint_files[-1]
        checkpoint = torch.load(latest_checkpoint)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint: {latest_checkpoint}")
        return checkpoint
    
    def save_best_model(self, model, optimizer, epoch, loss, is_best):
        """Save best model separately"""
        if is_best:
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(best_checkpoint, best_path)
            print(f"Best model saved: {best_path}")

# Test checkpoint manager
manager = CheckpointManager(max_checkpoints=3)

# Simulate training with checkpoints
best_loss = float('inf')
for epoch in range(7):
    # Simulate training
    current_loss = 1.0 - (epoch * 0.1)  # Decreasing loss
    
    # Save regular checkpoint
    manager.save_checkpoint(model, optimizer, epoch, current_loss)
    
    # Save best model
    is_best = current_loss < best_loss
    if is_best:
        best_loss = current_loss
    manager.save_best_model(model, optimizer, epoch, current_loss, is_best)

print(f"Checkpoint files maintained: {len(manager.checkpoint_files)}")

print("\n=== Error Handling and Validation ===")

def safe_load_checkpoint(checkpoint_path, model, optimizer=None):
    """Safely load checkpoint with validation"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing required key: {key}")
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
        
        # Load optimizer if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded successfully")
        
        return checkpoint
        
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
    except KeyError as e:
        print(f"Invalid checkpoint format: {e}")
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

# Test safe loading
test_checkpoint = safe_load_checkpoint('training_checkpoint.pth', new_model, optimizer)
if test_checkpoint:
    print(f"Successfully loaded checkpoint from epoch {test_checkpoint.get('epoch', 'unknown')}")

# Cleanup temporary files
temp_files = [
    'model_state_dict.pth', 'complete_model.pth', 'training_checkpoint.pth',
    'pretrained_weights.pth', filename
]

for file in temp_files:
    if os.path.exists(file):
        os.remove(file)

# Cleanup checkpoint directory
import shutil
if os.path.exists('checkpoints'):
    shutil.rmtree('checkpoints')

print("\n=== Checkpoint Best Practices ===")

print("Model Saving Guidelines:")
print("1. Save state_dict (not entire model) for portability")
print("2. Include optimizer state for resuming training")
print("3. Save learning rate scheduler state")
print("4. Include epoch number and loss for tracking")
print("5. Add metadata for model versioning")
print("6. Use meaningful checkpoint naming")
print("7. Implement checkpoint rotation to save space")

print("\nTransfer Learning:")
print("- Save feature extractor and classifier separately")
print("- Use strict=False when loading partial states")
print("- Filter incompatible layers based on shape")
print("- Freeze layers appropriately for transfer learning")

print("\nError Prevention:")
print("- Always use map_location when loading")
print("- Validate checkpoint structure before loading")
print("- Handle missing/unexpected keys gracefully")
print("- Save multiple checkpoints for backup")
print("- Test checkpoint loading in your training loop")

print("\n=== Checkpoint and State Dict Complete ===") 