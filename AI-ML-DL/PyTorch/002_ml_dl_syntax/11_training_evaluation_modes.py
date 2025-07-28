#!/usr/bin/env python3
"""PyTorch Training and Evaluation Modes - train(), eval(), no_grad() contexts"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Basic Training and Evaluation Modes ===")

# Create a simple model with dropout and batch normalization
class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

model = ExampleModel()
print(f"Model training mode: {model.training}")

# Sample input
input_tensor = torch.randn(4, 3, 32, 32)

print("\n=== Training Mode Behavior ===")

# Set model to training mode
model.train()
print(f"Model in training mode: {model.training}")

# Check individual module states
for name, module in model.named_modules():
    if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
        print(f"{name:>12} training: {module.training}")

# Forward pass in training mode
output_train = model(input_tensor)
print(f"Training mode output shape: {output_train.shape}")

# Run multiple times to see dropout randomness
outputs_train = []
for i in range(3):
    with torch.no_grad():
        output = model(input_tensor)
        outputs_train.append(output)

# Check if outputs are different (due to dropout)
diff_01 = torch.allclose(outputs_train[0], outputs_train[1])
diff_12 = torch.allclose(outputs_train[1], outputs_train[2])
print(f"Training outputs identical (should be False): {diff_01 and diff_12}")

print("\n=== Evaluation Mode Behavior ===")

# Set model to evaluation mode
model.eval()
print(f"Model in evaluation mode: {model.training}")

# Check individual module states
for name, module in model.named_modules():
    if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
        print(f"{name:>12} training: {module.training}")

# Forward pass in evaluation mode
output_eval = model(input_tensor)
print(f"Evaluation mode output shape: {output_eval.shape}")

# Run multiple times to see deterministic behavior
outputs_eval = []
for i in range(3):
    with torch.no_grad():
        output = model(input_tensor)
        outputs_eval.append(output)

# Check if outputs are identical (no dropout)
diff_eval_01 = torch.allclose(outputs_eval[0], outputs_eval[1])
diff_eval_12 = torch.allclose(outputs_eval[1], outputs_eval[2])
print(f"Evaluation outputs identical (should be True): {diff_eval_01 and diff_eval_12}")

print("\n=== Context Managers for Modes ===")

# Using context managers for temporary mode changes
model.train()  # Set to training mode

print(f"Before context: model.training = {model.training}")

# Temporarily switch to eval mode
with torch.no_grad():
    model.eval()
    print(f"Inside eval context: model.training = {model.training}")
    eval_output = model(input_tensor)

print(f"After context: model.training = {model.training}")

# Custom context manager for mode switching
class EvalMode:
    def __init__(self, model):
        self.model = model
        self.training_mode = None
    
    def __enter__(self):
        self.training_mode = self.model.training
        self.model.eval()
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.training_mode)

# Test custom context manager
model.train()
print(f"Before custom context: {model.training}")

with EvalMode(model) as eval_model:
    print(f"Inside custom context: {eval_model.training}")
    context_output = eval_model(input_tensor)

print(f"After custom context: {model.training}")

print("\n=== torch.no_grad() Context ===")

# Memory and computation benefits of no_grad
model.train()
input_with_grad = torch.randn(4, 3, 32, 32, requires_grad=True)

# Forward pass with gradient tracking
output_with_grad = model(input_with_grad)
print(f"With grad - output requires_grad: {output_with_grad.requires_grad}")

# Forward pass without gradient tracking
with torch.no_grad():
    output_no_grad = model(input_with_grad)
    print(f"No grad - output requires_grad: {output_no_grad.requires_grad}")

# torch.inference_mode() (even more efficient than no_grad)
with torch.inference_mode():
    output_inference = model(input_with_grad)
    print(f"Inference mode - output requires_grad: {output_inference.requires_grad}")

print("\n=== Gradient Enabling and Disabling ===")

# torch.enable_grad() context
with torch.no_grad():
    print("Inside no_grad context")
    
    # Re-enable gradients within no_grad
    with torch.enable_grad():
        x = torch.randn(4, 3, 32, 32, requires_grad=True)
        y = model(x)
        print(f"Enable grad within no_grad: {y.requires_grad}")

# torch.set_grad_enabled() for conditional gradient computation
def forward_pass(model, input_tensor, compute_gradients=True):
    with torch.set_grad_enabled(compute_gradients):
        return model(input_tensor)

# Test conditional gradient computation
model.train()
grad_output = forward_pass(model, input_tensor, compute_gradients=True)
no_grad_output = forward_pass(model, input_tensor, compute_gradients=False)

print(f"Conditional grad enabled: {grad_output.requires_grad}")
print(f"Conditional grad disabled: {no_grad_output.requires_grad}")

print("\n=== Module-specific Mode Control ===")

# Control specific modules independently
model.train()

# Set only dropout to eval mode
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.eval()

print("Dropout modules set to eval while model in train mode:")
for name, module in model.named_modules():
    if isinstance(module, (nn.Dropout, nn.BatchNorm2d)):
        print(f"{name:>12} training: {module.training}")

# Reset all to training mode
model.train()

print("\n=== Batch Normalization Behavior ===")

# Detailed look at BatchNorm behavior
bn_test = nn.BatchNorm2d(64)
test_input = torch.randn(16, 64, 32, 32)

# Training mode - updates running statistics
bn_test.train()
running_mean_before = bn_test.running_mean.clone()

with torch.no_grad():
    output_bn_train = bn_test(test_input)

running_mean_after = bn_test.running_mean.clone()
mean_changed = not torch.allclose(running_mean_before, running_mean_after)

print(f"BatchNorm training mode - running mean updated: {mean_changed}")

# Evaluation mode - uses running statistics
bn_test.eval()
running_mean_eval_before = bn_test.running_mean.clone()

with torch.no_grad():
    output_bn_eval = bn_test(test_input)

running_mean_eval_after = bn_test.running_mean.clone()
mean_eval_changed = not torch.allclose(running_mean_eval_before, running_mean_eval_after)

print(f"BatchNorm eval mode - running mean updated: {mean_eval_changed}")

print("\n=== Training Loop Integration ===")

# Complete training and evaluation loop example
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()  # Set to training mode
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    
    return avg_loss, accuracy

# Simulate data loaders
class FakeDataLoader:
    def __init__(self, batch_size, num_batches):
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            data = torch.randn(self.batch_size, 3, 32, 32)
            target = torch.randint(0, 10, (self.batch_size,))
            yield data, target

# Test training and evaluation
train_loader = FakeDataLoader(16, 5)
val_loader = FakeDataLoader(16, 3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Single epoch
train_loss = train_epoch(model, train_loader, optimizer, criterion)
val_loss, val_accuracy = evaluate(model, val_loader, criterion)

print(f"Training loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.2f}%")

print("\n=== Mode State Debugging ===")

def debug_model_modes(model):
    """Debug the training/eval state of all modules"""
    mode_info = {
        'model_training': model.training,
        'modules': {}
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'training'):
            mode_info['modules'][name] = {
                'type': type(module).__name__,
                'training': module.training
            }
    
    return mode_info

# Debug model state
model.train()
debug_info = debug_model_modes(model)

print(f"Model training state: {debug_info['model_training']}")
print("Module training states:")
for name, info in list(debug_info['modules'].items())[:5]:  # Show first 5
    print(f"  {name:>15} ({info['type']:>12}): {info['training']}")

print("\n=== Advanced Mode Management ===")

# Save and restore training states
class TrainingStateManager:
    def __init__(self):
        self.saved_states = {}
    
    def save_state(self, model, name):
        """Save training state of all modules"""
        state = {}
        for module_name, module in model.named_modules():
            if hasattr(module, 'training'):
                state[module_name] = module.training
        self.saved_states[name] = state
    
    def restore_state(self, model, name):
        """Restore training state of all modules"""
        if name in self.saved_states:
            state = self.saved_states[name]
            for module_name, module in model.named_modules():
                if module_name in state:
                    module.train(state[module_name])

# Test state manager
state_manager = TrainingStateManager()

# Save current state
model.train()
state_manager.save_state(model, 'training_state')

# Change to eval mode
model.eval()
print(f"After eval(): {model.training}")

# Restore training state
state_manager.restore_state(model, 'training_state')
print(f"After restore: {model.training}")

print("\n=== Performance Implications ===")

import time

# Measure inference speed difference
model.eval()
input_batch = torch.randn(32, 3, 32, 32)

# Warm up
for _ in range(10):
    with torch.no_grad():
        _ = model(input_batch)

# Time with gradients
model.train()
start_time = time.time()
for _ in range(100):
    output = model(input_batch)
train_time = time.time() - start_time

# Time without gradients
model.eval()
start_time = time.time()
for _ in range(100):
    with torch.no_grad():
        output = model(input_batch)
eval_time = time.time() - start_time

print(f"Training mode time: {train_time:.4f}s")
print(f"Eval mode (no_grad) time: {eval_time:.4f}s")
print(f"Speedup: {train_time / eval_time:.2f}x")

print("\n=== Best Practices ===")

print("Training/Evaluation Mode Guidelines:")
print("1. Always call model.train() before training")
print("2. Always call model.eval() before evaluation/inference")
print("3. Use torch.no_grad() during evaluation to save memory")
print("4. Be aware of BatchNorm and Dropout behavior differences")
print("5. Use context managers for temporary mode switches")
print("6. Consider torch.inference_mode() for pure inference")
print("7. Debug mode states when getting unexpected results")

print("\nCommon Pitfalls:")
print("- Forgetting to switch modes between train/eval")
print("- Not using no_grad() during evaluation (memory waste)")
print("- Assuming dropout/batchnorm behave the same in both modes")
print("- Not considering running statistics in BatchNorm")
print("- Memory issues from gradient accumulation during eval")

print("\nPerformance Tips:")
print("- Use torch.inference_mode() for production inference")
print("- Batch evaluation samples for better efficiency")
print("- Consider model.half() for faster inference")
print("- Use torch.jit.script() for optimized evaluation")
print("- Profile memory usage during long evaluation loops")

print("\n=== Training/Evaluation Modes Complete ===") 