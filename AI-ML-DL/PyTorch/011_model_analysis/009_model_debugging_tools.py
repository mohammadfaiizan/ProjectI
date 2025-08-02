import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from collections import defaultdict, OrderedDict
import traceback
import sys
import gc

# Sample Models for Debugging
class ProblematicModel(nn.Module):
    """Model with common issues for debugging demonstration"""
    
    def __init__(self, num_classes=10, add_problems=True):
        super().__init__()
        
        # Potentially problematic layer dimensions
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Batch norm layers (potential issues with eval/train mode)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Potentially problematic classifier with wrong dimensions
        if add_problems:
            # This will cause dimension mismatch issues
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 512),  # Wrong size for 32x32 input
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # This will fail with wrong classifier dimensions
        x = self.classifier(x)
        return x

class DebugModel(nn.Module):
    """Model instrumented for debugging"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Debug flags
        self.debug_mode = False
        self.layer_outputs = {}
    
    def forward(self, x):
        if self.debug_mode:
            self.layer_outputs['input'] = x.clone().detach()
        
        x = F.relu(self.bn1(self.conv1(x)))
        if self.debug_mode:
            self.layer_outputs['after_conv1'] = x.clone().detach()
        
        x = self.pool(x)
        if self.debug_mode:
            self.layer_outputs['after_pool1'] = x.clone().detach()
        
        x = F.relu(self.bn2(self.conv2(x)))
        if self.debug_mode:
            self.layer_outputs['after_conv2'] = x.clone().detach()
        
        x = self.pool(x)
        if self.debug_mode:
            self.layer_outputs['after_pool2'] = x.clone().detach()
        
        x = F.relu(self.bn3(self.conv3(x)))
        if self.debug_mode:
            self.layer_outputs['after_conv3'] = x.clone().detach()
        
        x = self.adaptive_pool(x)
        if self.debug_mode:
            self.layer_outputs['after_adaptive_pool'] = x.clone().detach()
        
        x = self.classifier(x)
        if self.debug_mode:
            self.layer_outputs['output'] = x.clone().detach()
        
        return x

# Model Debugging Tools
class ModelDebugger:
    """Comprehensive model debugging utilities"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.debug_info = {}
    
    def check_model_architecture(self) -> Dict[str, Any]:
        """Check model architecture for common issues"""
        
        issues = []
        info = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info['total_parameters'] = total_params
        info['trainable_parameters'] = trainable_params
        info['non_trainable_parameters'] = total_params - trainable_params
        
        # Check for very large or very small models
        if total_params > 50_000_000:
            issues.append(f"Very large model ({total_params:,} parameters) - consider model compression")
        elif total_params < 1000:
            issues.append(f"Very small model ({total_params:,} parameters) - might be underfitted")
        
        # Check layer types and potential issues
        layer_counts = defaultdict(int)
        for name, module in self.model.named_modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] += 1
            
            # Check for potential issues
            if isinstance(module, nn.BatchNorm2d):
                if module.num_features == 0:
                    issues.append(f"BatchNorm layer {name} has 0 features")
            
            elif isinstance(module, nn.Linear):
                if module.in_features != module.weight.shape[1]:
                    issues.append(f"Linear layer {name} has mismatched dimensions")
            
            elif isinstance(module, nn.Conv2d):
                if module.kernel_size[0] > 11 or module.kernel_size[1] > 11:
                    issues.append(f"Conv layer {name} has very large kernel size: {module.kernel_size}")
        
        info['layer_counts'] = dict(layer_counts)
        info['issues'] = issues
        
        return info
    
    def check_gradients(self, data_loader, num_batches: int = 3) -> Dict[str, Any]:
        """Check gradient flow and potential gradient issues"""
        
        gradient_info = {}
        gradient_norms = defaultdict(list)
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (data, targets) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Collect gradient information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_norms[name].append(grad_norm)
                else:
                    gradient_norms[name].append(0.0)
        
        # Analyze gradients
        gradient_stats = {}
        issues = []
        
        for name, norms in gradient_norms.items():
            avg_norm = np.mean(norms)
            std_norm = np.std(norms)
            max_norm = np.max(norms)
            min_norm = np.min(norms)
            
            gradient_stats[name] = {
                'mean': avg_norm,
                'std': std_norm,
                'max': max_norm,
                'min': min_norm
            }
            
            # Check for gradient issues
            if avg_norm < 1e-7:
                issues.append(f"Very small gradients in {name} (vanishing gradients)")
            elif avg_norm > 10:
                issues.append(f"Very large gradients in {name} (exploding gradients)")
            elif min_norm == 0:
                issues.append(f"Zero gradients detected in {name}")
        
        gradient_info['statistics'] = gradient_stats
        gradient_info['issues'] = issues
        
        return gradient_info
    
    def check_activations(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Check activation statistics for potential issues"""
        
        activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().cpu()
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
                handle = module.register_forward_hook(make_hook(name))
                hooks.append(handle)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze activations
        activation_stats = {}
        issues = []
        
        for name, activation in activations.items():
            if activation.numel() > 0:
                stats = {
                    'mean': activation.mean().item(),
                    'std': activation.std().item(),
                    'max': activation.max().item(),
                    'min': activation.min().item(),
                    'zeros_percentage': (activation == 0).float().mean().item(),
                    'shape': list(activation.shape)
                }
                
                activation_stats[name] = stats
                
                # Check for issues
                if stats['zeros_percentage'] > 0.9:
                    issues.append(f"High sparsity ({stats['zeros_percentage']:.1%}) in {name}")
                
                if stats['std'] < 1e-6:
                    issues.append(f"Very low activation variance in {name}")
                
                if abs(stats['mean']) > 100:
                    issues.append(f"Very large activation magnitudes in {name}")
        
        return {
            'statistics': activation_stats,
            'issues': issues
        }
    
    def test_input_output_shapes(self, input_shapes: List[Tuple[int, ...]]) -> Dict[str, Any]:
        """Test various input shapes to check for dimension issues"""
        
        results = {}
        
        for input_shape in input_shapes:
            try:
                test_input = torch.randn(1, *input_shape).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    output = self.model(test_input)
                
                results[str(input_shape)] = {
                    'success': True,
                    'output_shape': list(output.shape),
                    'error': None
                }
                
            except Exception as e:
                results[str(input_shape)] = {
                    'success': False,
                    'output_shape': None,
                    'error': str(e)
                }
        
        return results
    
    def check_batch_norm_issues(self) -> Dict[str, Any]:
        """Check for batch normalization related issues"""
        
        bn_info = {}
        issues = []
        
        # Find all BatchNorm layers
        bn_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_layers[name] = module
        
        if not bn_layers:
            return {'has_batch_norm': False, 'issues': []}
        
        bn_info['has_batch_norm'] = True
        bn_info['num_bn_layers'] = len(bn_layers)
        
        # Check for common BN issues
        for name, bn_layer in bn_layers.items():
            # Check if running stats are reasonable
            if hasattr(bn_layer, 'running_mean') and bn_layer.running_mean is not None:
                mean_magnitude = bn_layer.running_mean.abs().mean().item()
                var_magnitude = bn_layer.running_var.mean().item()
                
                if mean_magnitude > 10:
                    issues.append(f"Large running mean in {name}: {mean_magnitude:.2f}")
                
                if var_magnitude > 100:
                    issues.append(f"Large running variance in {name}: {var_magnitude:.2f}")
                
                if var_magnitude < 1e-6:
                    issues.append(f"Very small running variance in {name}: {var_magnitude:.2e}")
        
        bn_info['issues'] = issues
        return bn_info

class TrainingDebugger:
    """Debug training process and identify common training issues"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'learning_rates': [],
            'gradient_norms': []
        }
    
    def debug_training_step(self, data_loader, optimizer, criterion, 
                           num_steps: int = 10) -> Dict[str, Any]:
        """Debug a few training steps"""
        
        debug_info = {}
        step_info = []
        
        self.model.train()
        
        for step, (data, targets) in enumerate(data_loader):
            if step >= num_steps:
                break
            
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Calculate gradient norm before optimizer step
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Optimizer step
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == targets).float().mean().item()
            
            # Store step information
            step_data = {
                'step': step,
                'loss': loss.item(),
                'accuracy': accuracy,
                'gradient_norm': total_norm,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            step_info.append(step_data)
            
            # Update history
            self.training_history['losses'].append(loss.item())
            self.training_history['accuracies'].append(accuracy)
            self.training_history['gradient_norms'].append(total_norm)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Analyze training steps
        debug_info['steps'] = step_info
        debug_info['issues'] = self._analyze_training_issues(step_info)
        
        return debug_info
    
    def _analyze_training_issues(self, step_info: List[Dict]) -> List[str]:
        """Analyze training steps for common issues"""
        
        issues = []
        
        if len(step_info) < 2:
            return issues
        
        losses = [step['loss'] for step in step_info]
        accuracies = [step['accuracy'] for step in step_info]
        gradient_norms = [step['gradient_norm'] for step in step_info]
        
        # Check for loss issues
        if losses[0] == losses[-1]:
            issues.append("Loss not changing - possible learning rate or gradient issues")
        
        if any(np.isnan(loss) for loss in losses):
            issues.append("NaN loss detected - possible numerical instability")
        
        if any(loss > 1000 for loss in losses):
            issues.append("Very high loss values - possible learning rate too high")
        
        # Check for gradient issues
        avg_grad_norm = np.mean(gradient_norms)
        if avg_grad_norm < 1e-7:
            issues.append("Very small gradients - possible vanishing gradient problem")
        elif avg_grad_norm > 100:
            issues.append("Very large gradients - possible exploding gradient problem")
        
        # Check accuracy progression
        if len(accuracies) > 5:
            accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
            if accuracy_trend < 0:
                issues.append("Accuracy decreasing during training")
        
        return issues
    
    def plot_training_debug(self):
        """Plot training debugging information"""
        
        if not self.training_history['losses']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['accuracies'])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norm plot
        axes[1, 0].plot(self.training_history['gradient_norms'])
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 1].plot(self.training_history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_debug.png', dpi=150, bbox_inches='tight')
        plt.show()

class MemoryDebugger:
    """Debug memory usage issues"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def check_memory_leaks(self, model: nn.Module, data_loader, 
                          num_iterations: int = 10) -> Dict[str, Any]:
        """Check for memory leaks during training"""
        
        memory_usage = []
        
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        for i, (data, targets) in enumerate(data_loader):
            if i >= num_iterations:
                break
            
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Record memory before forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_before = 0
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Record memory after backward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                memory_after = 0
            
            memory_usage.append({
                'iteration': i,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_increase': memory_after - memory_before
            })
            
            # Clear gradients (important for memory)
            model.zero_grad()
            
            # Force garbage collection
            if i % 3 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Analyze memory usage
        memory_increases = [usage['memory_increase'] for usage in memory_usage]
        avg_increase = np.mean(memory_increases)
        total_increase = memory_usage[-1]['memory_after'] - memory_usage[0]['memory_before']
        
        issues = []
        if avg_increase > 10:  # More than 10MB per iteration
            issues.append(f"High memory increase per iteration: {avg_increase:.1f} MB")
        
        if total_increase > 100:  # More than 100MB total increase
            issues.append(f"Significant total memory increase: {total_increase:.1f} MB")
        
        return {
            'memory_usage': memory_usage,
            'average_increase_per_iteration': avg_increase,
            'total_memory_increase': total_increase,
            'issues': issues
        }
    
    def profile_model_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Profile memory usage of model components"""
        
        # Get baseline memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            baseline_memory = 0
        
        # Model parameters memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        
        # Test forward pass memory
        test_input = torch.randn(1, *input_shape).to(self.device)
        
        model.eval()
        with torch.no_grad():
            _ = model(test_input)
        
        if torch.cuda.is_available():
            forward_memory = torch.cuda.max_memory_allocated() / 1024**2
            activation_memory = forward_memory - baseline_memory - param_memory
        else:
            forward_memory = 0
            activation_memory = 0
        
        return {
            'baseline_memory_mb': baseline_memory,
            'parameter_memory_mb': param_memory,
            'forward_pass_memory_mb': forward_memory,
            'activation_memory_mb': max(0, activation_memory),
            'total_memory_mb': forward_memory
        }

class ErrorHandler:
    """Handle and diagnose common PyTorch errors"""
    
    @staticmethod
    def diagnose_error(error: Exception, model: nn.Module = None, 
                      input_data: torch.Tensor = None) -> Dict[str, str]:
        """Diagnose common PyTorch errors and provide solutions"""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        diagnosis = {
            'error_type': error_type,
            'error_message': error_message,
            'possible_cause': 'Unknown',
            'suggested_solution': 'Check PyTorch documentation'
        }
        
        # Dimension mismatch errors
        if 'size mismatch' in error_message.lower() or 'dimension' in error_message.lower():
            diagnosis['possible_cause'] = 'Tensor dimension mismatch'
            diagnosis['suggested_solution'] = (
                "Check tensor shapes in forward pass. Use tensor.shape to debug. "
                "Common issues: wrong input size, incorrect layer dimensions."
            )
        
        # CUDA errors
        elif 'cuda' in error_message.lower():
            if 'out of memory' in error_message.lower():
                diagnosis['possible_cause'] = 'GPU out of memory'
                diagnosis['suggested_solution'] = (
                    "Reduce batch size, use gradient accumulation, "
                    "or use torch.cuda.empty_cache(). Consider mixed precision training."
                )
            else:
                diagnosis['possible_cause'] = 'CUDA device error'
                diagnosis['suggested_solution'] = (
                    "Check device placement of tensors and model. "
                    "Ensure all tensors are on the same device."
                )
        
        # Gradient computation errors
        elif 'backward' in error_message.lower() or 'gradient' in error_message.lower():
            diagnosis['possible_cause'] = 'Gradient computation error'
            diagnosis['suggested_solution'] = (
                "Check if model is in training mode. "
                "Ensure loss tensor requires gradients. "
                "Check for in-place operations that break gradient computation."
            )
        
        # DataLoader errors
        elif 'dataloader' in error_message.lower() or 'batch' in error_message.lower():
            diagnosis['possible_cause'] = 'Data loading issue'
            diagnosis['suggested_solution'] = (
                "Check dataset implementation, batch size, "
                "and data preprocessing. Verify __getitem__ and __len__ methods."
            )
        
        # NaN/Inf errors
        elif 'nan' in error_message.lower() or 'inf' in error_message.lower():
            diagnosis['possible_cause'] = 'Numerical instability (NaN/Inf values)'
            diagnosis['suggested_solution'] = (
                "Check learning rate (may be too high), "
                "add gradient clipping, check for division by zero, "
                "use torch.isnan() and torch.isinf() to debug."
            )
        
        # Add model-specific diagnosis if available
        if model is not None:
            try:
                # Check model state
                model_info = []
                if hasattr(model, 'training'):
                    model_info.append(f"Model training mode: {model.training}")
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                model_info.append(f"Total parameters: {total_params:,}")
                
                diagnosis['model_info'] = '; '.join(model_info)
            except:
                pass
        
        return diagnosis

def run_comprehensive_debug(model: nn.Module, data_loader, device='cuda') -> Dict[str, Any]:
    """Run comprehensive debugging on a model"""
    
    print("Running comprehensive model debugging...")
    
    debug_results = {}
    
    # 1. Model Architecture Check
    print("1. Checking model architecture...")
    debugger = ModelDebugger(model, device)
    arch_info = debugger.check_model_architecture()
    debug_results['architecture'] = arch_info
    
    # 2. Gradient Check
    print("2. Checking gradients...")
    gradient_info = debugger.check_gradients(data_loader)
    debug_results['gradients'] = gradient_info
    
    # 3. Activation Check
    print("3. Checking activations...")
    sample_data, _ = next(iter(data_loader))
    sample_data = sample_data.to(device)
    activation_info = debugger.check_activations(sample_data[:1])
    debug_results['activations'] = activation_info
    
    # 4. Input/Output Shape Testing
    print("4. Testing input/output shapes...")
    test_shapes = [(3, 32, 32), (3, 64, 64), (3, 28, 28)]
    shape_results = debugger.test_input_output_shapes(test_shapes)
    debug_results['shape_tests'] = shape_results
    
    # 5. Batch Norm Check
    print("5. Checking batch normalization...")
    bn_info = debugger.check_batch_norm_issues()
    debug_results['batch_norm'] = bn_info
    
    # 6. Training Debug
    print("6. Debugging training process...")
    training_debugger = TrainingDebugger(model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    training_info = training_debugger.debug_training_step(data_loader, optimizer, criterion)
    debug_results['training'] = training_info
    
    # 7. Memory Check
    print("7. Checking memory usage...")
    memory_debugger = MemoryDebugger(device)
    memory_info = memory_debugger.check_memory_leaks(model, data_loader)
    debug_results['memory'] = memory_info
    
    return debug_results

if __name__ == "__main__":
    print("Model Debugging Tools")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    sample_data = torch.randn(100, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (100,))
    sample_dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=16, shuffle=True)
    
    print("\n1. Testing Problematic Model")
    print("-" * 35)
    
    # Test problematic model
    try:
        problematic_model = ProblematicModel(add_problems=True).to(device)
        test_input = torch.randn(2, 3, 32, 32).to(device)
        
        output = problematic_model(test_input)
        print("Problematic model worked unexpectedly!")
        
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}")
        
        # Diagnose the error
        error_handler = ErrorHandler()
        diagnosis = error_handler.diagnose_error(e, problematic_model, test_input)
        
        print("Error Diagnosis:")
        print(f"  Error Type: {diagnosis['error_type']}")
        print(f"  Possible Cause: {diagnosis['possible_cause']}")
        print(f"  Suggested Solution: {diagnosis['suggested_solution']}")
    
    print("\n2. Testing Working Model")
    print("-" * 30)
    
    # Test working model
    working_model = ProblematicModel(add_problems=False).to(device)
    
    try:
        test_input = torch.randn(2, 3, 32, 32).to(device)
        output = working_model(test_input)
        print(f"Working model output shape: {output.shape}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    print("\n3. Comprehensive Model Debugging")
    print("-" * 40)
    
    # Run comprehensive debugging on working model
    debug_results = run_comprehensive_debug(working_model, sample_loader, device)
    
    # Print summary of results
    print("\nDebugging Summary:")
    print("=" * 20)
    
    # Architecture issues
    arch_issues = debug_results['architecture']['issues']
    print(f"Architecture Issues: {len(arch_issues)}")
    for issue in arch_issues:
        print(f"  - {issue}")
    
    # Gradient issues
    grad_issues = debug_results['gradients']['issues']
    print(f"Gradient Issues: {len(grad_issues)}")
    for issue in grad_issues:
        print(f"  - {issue}")
    
    # Activation issues
    act_issues = debug_results['activations']['issues']
    print(f"Activation Issues: {len(act_issues)}")
    for issue in act_issues:
        print(f"  - {issue}")
    
    # Training issues
    train_issues = debug_results['training']['issues']
    print(f"Training Issues: {len(train_issues)}")
    for issue in train_issues:
        print(f"  - {issue}")
    
    # Memory issues
    mem_issues = debug_results['memory']['issues']
    print(f"Memory Issues: {len(mem_issues)}")
    for issue in mem_issues:
        print(f"  - {issue}")
    
    print("\n4. Debug Model with Instrumentation")
    print("-" * 42)
    
    # Test debug model
    debug_model = DebugModel().to(device)
    debug_model.debug_mode = True
    
    test_input = torch.randn(1, 3, 32, 32).to(device)
    output = debug_model(test_input)
    
    print("Debug Model Layer Outputs:")
    for layer_name, layer_output in debug_model.layer_outputs.items():
        print(f"  {layer_name}: {layer_output.shape}")
    
    print("\n5. Training Debugging Visualization")
    print("-" * 40)
    
    # Training debugging with visualization
    training_debugger = TrainingDebugger(debug_model, device)
    optimizer = torch.optim.Adam(debug_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Run more training steps for better visualization
    training_info = training_debugger.debug_training_step(
        sample_loader, optimizer, criterion, num_steps=20
    )
    
    print("Training Debug Results:")
    print(f"  Steps analyzed: {len(training_info['steps'])}")
    print(f"  Issues found: {len(training_info['issues'])}")
    
    for issue in training_info['issues']:
        print(f"    - {issue}")
    
    # Plot training debug information
    training_debugger.plot_training_debug()
    
    print("\n6. Memory Profiling")
    print("-" * 22)
    
    # Memory profiling
    memory_debugger = MemoryDebugger(device)
    memory_profile = memory_debugger.profile_model_memory(debug_model, (3, 32, 32))
    
    print("Memory Profile:")
    for key, value in memory_profile.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n7. Error Handling Examples")
    print("-" * 32)
    
    # Demonstrate error handling for common issues
    error_handler = ErrorHandler()
    
    # Simulate different types of errors
    test_errors = [
        RuntimeError("size mismatch, m1: [16 x 256], m2: [128 x 64]"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn"),
        ValueError("Expected input batch_size (32) to match target batch_size (16)"),
    ]
    
    print("Error Diagnosis Examples:")
    for i, error in enumerate(test_errors, 1):
        diagnosis = error_handler.diagnose_error(error)
        print(f"\n{i}. {diagnosis['error_type']}")
        print(f"   Cause: {diagnosis['possible_cause']}")
        print(f"   Solution: {diagnosis['suggested_solution'][:80]}...")
    
    print("\n8. Best Practices Summary")
    print("-" * 32)
    
    best_practices = [
        "Always check tensor shapes and device placement",
        "Use model.train() and model.eval() appropriately",
        "Monitor gradient norms to detect vanishing/exploding gradients",
        "Check for memory leaks in training loops",
        "Use proper error handling and debugging tools",
        "Validate model architecture before training",
        "Monitor activation statistics for dead neurons",
        "Use gradient clipping for stability",
        "Implement proper data loading and preprocessing",
        "Test models with different input sizes"
    ]
    
    print("PyTorch Debugging Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\nModel debugging completed!")
    print("Generated files:")
    print("  - training_debug.png")