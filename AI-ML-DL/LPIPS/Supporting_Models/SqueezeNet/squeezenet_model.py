"""
SqueezeNet Implementation for LPIPS Supporting Model
===================================================

Complete SqueezeNet implementation with ImageNet training, evaluation, and parameter analysis.
Based on the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
by Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer.

This implementation includes:
- SqueezeNet 1.0 and 1.1 architectures
- Fire module implementation
- ImageNet training pipeline
- Efficiency analysis and optimization
- Parameter analysis for LPIPS feature extraction
- Performance benchmarking

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import OrderedDict
import json


class Fire(nn.Module):
    """
    Fire module implementation for SqueezeNet
    
    The Fire module consists of:
    1. Squeeze layer: 1x1 convolution to reduce input channels
    2. Expand layer: combination of 1x1 and 3x3 convolutions
    
    This design significantly reduces parameters while maintaining expressivity.
    """
    
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        
        self.inplanes = inplanes
        self.squeeze_planes = squeeze_planes
        self.expand1x1_planes = expand1x1_planes
        self.expand3x3_planes = expand3x3_planes
        
        # Squeeze layer (1x1 convolution)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Squeeze phase
        x = self.squeeze_activation(self.squeeze(x))
        
        # Expand phase
        expand1x1_out = self.expand1x1_activation(self.expand1x1(x))
        expand3x3_out = self.expand3x3_activation(self.expand3x3(x))
        
        # Concatenate expand outputs
        return torch.cat([expand1x1_out, expand3x3_out], 1)
    
    def get_parameter_count(self):
        """Get parameter count for this Fire module"""
        total_params = sum(p.numel() for p in self.parameters())
        
        squeeze_params = sum(p.numel() for p in self.squeeze.parameters())
        expand1x1_params = sum(p.numel() for p in self.expand1x1.parameters())
        expand3x3_params = sum(p.numel() for p in self.expand3x3.parameters())
        
        return {
            'total': total_params,
            'squeeze': squeeze_params,
            'expand1x1': expand1x1_params,
            'expand3x3': expand3x3_params
        }


class SqueezeNet(nn.Module):
    """
    SqueezeNet architecture implementation
    
    Original paper: https://arxiv.org/abs/1602.07360
    
    Key design principles:
    1. Replace 3x3 filters with 1x1 filters (Strategy 1)
    2. Decrease number of input channels to 3x3 filters (Strategy 2)
    3. Downsample late in the network (Strategy 3)
    
    Two versions available:
    - SqueezeNet 1.0: Original version
    - SqueezeNet 1.1: Improved version with better accuracy/efficiency trade-off
    """
    
    def __init__(self, version='1_0', num_classes=1000, dropout=0.5):
        super(SqueezeNet, self).__init__()
        
        self.version = version
        self.num_classes = num_classes
        self.dropout_prob = dropout
        
        if version == '1_0':
            self._build_squeezenet_1_0()
        elif version == '1_1':
            self._build_squeezenet_1_1()
        else:
            raise ValueError(f"Unsupported SqueezeNet version: {version}")
        
        # Initialize weights
        self._initialize_weights()
        
        # Store layer information for LPIPS feature extraction
        self.lpips_layers = self._get_lpips_layers()
    
    def _build_squeezenet_1_0(self):
        """Build SqueezeNet 1.0 architecture"""
        self.features = nn.Sequential(
            # Initial convolution
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire modules
            Fire(96, 16, 64, 64),      # fire2
            Fire(128, 16, 64, 64),     # fire3
            Fire(128, 32, 128, 128),   # fire4
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(256, 32, 128, 128),   # fire5
            Fire(256, 48, 192, 192),   # fire6
            Fire(384, 48, 192, 192),   # fire7
            Fire(384, 64, 256, 256),   # fire8
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(512, 64, 256, 256),   # fire9
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _build_squeezenet_1_1(self):
        """Build SqueezeNet 1.1 architecture (more efficient)"""
        self.features = nn.Sequential(
            # Initial convolution (smaller kernel size for efficiency)
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            # Fire modules
            Fire(64, 16, 64, 64),      # fire2
            Fire(128, 16, 64, 64),     # fire3
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(128, 32, 128, 128),   # fire4
            Fire(256, 32, 128, 128),   # fire5
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            
            Fire(256, 48, 192, 192),   # fire6
            Fire(384, 48, 192, 192),   # fire7
            Fire(384, 64, 256, 256),   # fire8
            Fire(512, 64, 256, 256),   # fire9
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _get_lpips_layers(self):
        """Get layer indices for LPIPS feature extraction"""
        if self.version == '1_0':
            # Layers: conv1, fire2, fire4, fire6, fire8
            return [0, 3, 5, 7, 9]  # Indices in features
        else:  # version '1_1'
            # Layers: conv1, fire2, fire4, fire6, fire8
            return [0, 3, 6, 9, 11]  # Indices in features
    
    def _initialize_weights(self):
        """Initialize weights according to the paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.classifier[1]:  # Final conv layer
                    # Initialize final layer with normal distribution
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    # Initialize other conv layers with Xavier
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through SqueezeNet"""
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
    
    def forward_features(self, x):
        """
        Forward pass with intermediate feature extraction for LPIPS
        
        Returns:
            dict: Dictionary containing features from LPIPS-relevant layers
        """
        features = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.lpips_layers:
                features[f'features.{i}'] = x.clone()
        
        return features
    
    def get_fire_module_analysis(self):
        """Analyze Fire modules in the network"""
        fire_analysis = {}
        fire_count = 0
        
        for name, module in self.named_modules():
            if isinstance(module, Fire):
                fire_count += 1
                fire_params = module.get_parameter_count()
                
                fire_analysis[f'fire{fire_count + 1}'] = {
                    'module_name': name,
                    'inplanes': module.inplanes,
                    'squeeze_planes': module.squeeze_planes,
                    'expand1x1_planes': module.expand1x1_planes,
                    'expand3x3_planes': module.expand3x3_planes,
                    'total_output_channels': module.expand1x1_planes + module.expand3x3_planes,
                    'compression_ratio': module.inplanes / module.squeeze_planes,
                    'parameter_count': fire_params
                }
        
        return fire_analysis
    
    def get_parameter_count(self):
        """Get detailed parameter count analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Separate feature and classifier parameters
        feature_params = sum(p.numel() for p in self.features.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        # Fire module parameters
        fire_params = 0
        conv_params = 0
        
        for module in self.modules():
            if isinstance(module, Fire):
                fire_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Conv2d) and not isinstance(module.parent(), Fire):
                conv_params += sum(p.numel() for p in module.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_parameters': feature_params,
            'classifier_parameters': classifier_params,
            'fire_module_parameters': fire_params,
            'conv_layer_parameters': conv_params,
            'parameter_efficiency': total_params / 1e6  # Parameters in millions
        }
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def compare_with_alexnet(self):
        """Compare parameter efficiency with AlexNet"""
        squeezenet_params = self.get_parameter_count()['total_parameters']
        
        # AlexNet has approximately 61 million parameters
        alexnet_params = 61_000_000
        
        parameter_reduction = alexnet_params / squeezenet_params
        size_reduction = 240 / self.get_model_size()  # AlexNet ~240MB
        
        return {
            'squeezenet_parameters': squeezenet_params,
            'alexnet_parameters': alexnet_params,
            'parameter_reduction_factor': parameter_reduction,
            'size_reduction_factor': size_reduction,
            'squeezenet_size_mb': self.get_model_size(),
            'alexnet_size_mb': 240
        }


class SqueezeNetTrainer:
    """
    Comprehensive training framework for SqueezeNet on ImageNet
    """
    
    def __init__(self, model, device='cuda', data_dir='./data'):
        self.model = model.to(device)
        self.device = device
        self.data_dir = data_dir
        
        # Training hyperparameters (optimized for SqueezeNet)
        self.learning_rate = 0.04  # Higher LR due to smaller model
        self.momentum = 0.9
        self.weight_decay = 1e-4   # Lower weight decay
        self.batch_size = 512      # Larger batch size due to efficiency
        self.num_epochs = 120      # More epochs for convergence
        
        # Learning rate schedule (polynomial decay)
        self.lr_schedule_type = 'polynomial'  # or 'step'
        self.power = 1.0
        
        # Initialize optimizer and criterion
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Polynomial learning rate scheduler
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=self.num_epochs,
            power=self.power
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epoch_times': [],
            'memory_usage': []
        }
    
    def setup_data_loaders(self):
        """Setup ImageNet data loaders with SqueezeNet-specific preprocessing"""
        
        # SqueezeNet preprocessing (similar to original paper)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load ImageNet dataset
        try:
            train_dataset = torchvision.datasets.ImageNet(
                root=self.data_dir,
                split='train',
                transform=train_transform,
                download=False
            )
            
            val_dataset = torchvision.datasets.ImageNet(
                root=self.data_dir,
                split='val',
                transform=val_transform,
                download=False
            )
            
        except Exception as e:
            print(f"ImageNet dataset not found: {e}")
            print("Using CIFAR-10 for demonstration purposes...")
            
            # Fallback to CIFAR-10 for demonstration
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=train_transform,
                download=True
            )
            
            val_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=val_transform,
                download=True
            )
        
        # Create data loaders with larger batch size for efficiency
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  # More workers for larger batches
            pin_memory=True,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Training batches per epoch: {len(self.train_loader)}")
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
        return 0
    
    def train_epoch(self, epoch):
        """Train for one epoch with efficiency monitoring"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.train_history['learning_rates'].append(current_lr)
        
        # Monitor memory usage
        memory_usage = self.get_memory_usage()
        self.train_history['memory_usage'].append(memory_usage)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress (less frequent due to larger batches)
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'LR: {current_lr:.6f}, '
                      f'Memory: {memory_usage:.2f}GB')
        
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_acc)
        self.train_history['epoch_times'].append(epoch_time)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model with efficiency metrics"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        top5_correct = 0
        
        inference_times = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                # Measure inference time
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                val_loss += self.criterion(output, target).item()
                
                # Top-1 accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, 1, largest=True, sorted=True)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        top5_acc = 100. * top5_correct / total
        avg_inference_time = np.mean(inference_times)
        
        self.train_history['val_loss'].append(val_loss)
        self.train_history['val_accuracy'].append(val_acc)
        
        return val_loss, val_acc, top5_acc, avg_inference_time
    
    def train(self, num_epochs=None):
        """Complete training loop with efficiency monitoring"""
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        print(f"Starting SqueezeNet-{self.model.version} training...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial learning rate: {self.learning_rate}")
        print(f"Model parameters: {self.model.get_parameter_count()['total_parameters']:,}")
        print(f"Model size: {self.model.get_model_size():.2f} MB")
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            val_loss, val_acc, top5_acc, inference_time = self.validate()
            
            print(f'\nEpoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
            print(f'Epoch Time: {self.train_history["epoch_times"][-1]:.2f}s')
            print(f'Inference Time: {inference_time:.4f}s per batch')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'Memory Usage: {self.train_history["memory_usage"][-1]:.2f}GB')
            print('-' * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Final efficiency analysis
        self.analyze_training_efficiency()
    
    def analyze_training_efficiency(self):
        """Analyze training efficiency"""
        total_training_time = sum(self.train_history['epoch_times'])
        avg_epoch_time = np.mean(self.train_history['epoch_times'])
        samples_per_second = len(self.train_loader.dataset) / avg_epoch_time
        
        efficiency_metrics = {
            'total_training_time_hours': total_training_time / 3600,
            'average_epoch_time_minutes': avg_epoch_time / 60,
            'samples_per_second': samples_per_second,
            'final_accuracy': self.train_history['val_accuracy'][-1],
            'peak_memory_usage_gb': max(self.train_history['memory_usage']),
            'parameter_count': self.model.get_parameter_count()['total_parameters'],
            'model_size_mb': self.model.get_model_size()
        }
        
        print("\n=== Training Efficiency Analysis ===")
        for metric, value in efficiency_metrics.items():
            print(f"{metric}: {value:.3f}")
        
        return efficiency_metrics
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'version': self.model.version,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'model_config': {
                'version': self.model.version,
                'num_classes': self.model.num_classes,
                'dropout': self.model.dropout_prob
            }
        }
        
        filename = f'checkpoints/squeezenet_{self.model.version}_best.pth' if best else f'checkpoints/squeezenet_{self.model.version}_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
    
    def plot_training_history(self):
        """Plot comprehensive training history with efficiency metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss plot
        ax1.plot(self.train_history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.train_history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title(f'SqueezeNet-{self.model.version} Loss History', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_history['accuracy'], label='Train Acc', linewidth=2)
        ax2.plot(self.train_history['val_accuracy'], label='Val Acc', linewidth=2)
        ax2.set_title(f'SqueezeNet-{self.model.version} Accuracy History', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate and memory usage
        ax3_twin = ax3.twinx()
        line1, = ax3.plot(self.train_history['learning_rates'], 'b-', linewidth=2, label='Learning Rate')
        line2, = ax3_twin.plot(self.train_history['memory_usage'], 'r-', linewidth=2, label='Memory Usage (GB)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate', color='b')
        ax3_twin.set_ylabel('Memory Usage (GB)', color='r')
        ax3.set_title('Learning Rate and Memory Usage', fontsize=14)
        ax3.set_yscale('log')
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='center right')
        ax3.grid(True, alpha=0.3)
        
        # Epoch time analysis
        ax4.plot(self.train_history['epoch_times'], linewidth=2, color='green')
        ax4.set_title('Training Time per Epoch', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'squeezenet_{self.model.version}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class SqueezeNetEvaluator:
    """
    Comprehensive evaluation framework for SqueezeNet with efficiency focus
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_classification(self, data_loader, detailed=True):
        """Comprehensive classification evaluation with efficiency metrics"""
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        class_correct = {}
        class_total = {}
        predictions = []
        ground_truth = []
        confidence_scores = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(output, dim=1)
                
                # Top-1 accuracy
                _, pred = output.max(1)
                correct_predictions += pred.eq(target).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, 1, largest=True, sorted=True)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
                
                total_samples += target.size(0)
                
                if detailed:
                    # Per-class accuracy
                    for i in range(len(target)):
                        label = target[i].item()
                        if label not in class_correct:
                            class_correct[label] = 0
                            class_total[label] = 0
                        class_total[label] += 1
                        if pred[i] == label:
                            class_correct[label] += 1
                    
                    # Store predictions and confidence
                    predictions.extend(pred.cpu().numpy())
                    ground_truth.extend(target.cpu().numpy())
                    confidence_scores.extend(probabilities.max(1)[0].cpu().numpy())
        
        # Calculate metrics
        top1_accuracy = 100. * correct_predictions / total_samples
        top5_accuracy = 100. * top5_correct / total_samples
        avg_inference_time = np.mean(inference_times)
        
        result = {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'total_samples': total_samples,
            'average_inference_time': avg_inference_time,
            'inference_fps': len(data_loader.dataset) / sum(inference_times)
        }
        
        if detailed:
            # Per-class accuracy
            per_class_accuracy = {}
            for label in class_correct:
                per_class_accuracy[label] = 100. * class_correct[label] / class_total[label]
            
            result.update({
                'per_class_accuracy': per_class_accuracy,
                'predictions': predictions,
                'ground_truth': ground_truth,
                'confidence_scores': confidence_scores,
                'mean_confidence': np.mean(confidence_scores),
                'std_confidence': np.std(confidence_scores)
            })
        
        return result
    
    def analyze_efficiency_vs_accuracy(self, test_loader, batch_sizes=[1, 4, 8, 16, 32, 64]):
        """Analyze efficiency vs accuracy trade-offs"""
        efficiency_analysis = {}
        
        for batch_size in batch_sizes:
            # Create new data loader with specific batch size
            test_dataset = test_loader.dataset
            temp_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            # Evaluate with this batch size
            results = self.evaluate_classification(temp_loader, detailed=False)
            
            efficiency_analysis[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'accuracy': results['top1_accuracy'],
                'inference_time_per_batch': results['average_inference_time'],
                'inference_time_per_image': results['average_inference_time'] / batch_size,
                'fps': results['inference_fps'],
                'throughput': batch_size / results['average_inference_time']
            }
        
        return efficiency_analysis
    
    def analyze_fire_module_activations(self, sample_images):
        """Analyze Fire module activations and efficiency"""
        fire_analysis = {}
        
        with torch.no_grad():
            for img_idx, img in enumerate(sample_images):
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
                
                # Track activations through Fire modules
                x = img
                fire_count = 0
                
                for i, layer in enumerate(self.model.features):
                    x = layer(x)
                    
                    if isinstance(layer, Fire):
                        fire_count += 1
                        
                        # Analyze Fire module output
                        activation_stats = {
                            'output_shape': list(x.shape),
                            'mean_activation': x.mean().item(),
                            'std_activation': x.std().item(),
                            'sparsity': (x == 0).float().mean().item(),
                            'max_activation': x.max().item(),
                            'min_activation': x.min().item()
                        }
                        
                        fire_analysis[f'image_{img_idx}_fire_{fire_count}'] = activation_stats
        
        return fire_analysis
    
    def measure_memory_efficiency(self, input_sizes=None):
        """Measure memory efficiency across different input sizes"""
        if input_sizes is None:
            input_sizes = [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224)]
        
        memory_analysis = {}
        
        for input_size in input_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Measure memory during forward pass
            dummy_input = torch.randn(input_size).to(self.device)
            
            if torch.cuda.is_available():
                start_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - start_memory
                
                memory_analysis[f'batch_{input_size[0]}'] = {
                    'input_size': input_size,
                    'memory_used_mb': memory_used / 1024**2,
                    'memory_per_image_mb': memory_used / input_size[0] / 1024**2,
                    'peak_memory_mb': peak_memory / 1024**2
                }
            else:
                memory_analysis[f'batch_{input_size[0]}'] = {
                    'input_size': input_size,
                    'note': 'Memory analysis requires CUDA'
                }
        
        return memory_analysis
    
    def compare_efficiency_with_other_models(self, comparison_models, test_input):
        """Compare SqueezeNet efficiency with other models"""
        comparison_results = {}
        
        # Evaluate SqueezeNet
        squeezenet_results = self._evaluate_single_model(self.model, test_input, 'SqueezeNet')
        comparison_results[f'SqueezeNet-{self.model.version}'] = squeezenet_results
        
        # Evaluate comparison models
        for model_name, model in comparison_models.items():
            model.eval()
            results = self._evaluate_single_model(model, test_input, model_name)
            comparison_results[model_name] = results
        
        return comparison_results
    
    def _evaluate_single_model(self, model, test_input, model_name):
        """Evaluate a single model for comparison"""
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = param_size / 1024**2
        
        # Inference time
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(test_input)
                times.append(time.time() - start)
        
        avg_time = np.mean(times[10:])  # Skip first few for warmup
        
        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(test_input)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_memory = 0
        
        return {
            'model_name': model_name,
            'parameters': total_params,
            'model_size_mb': model_size_mb,
            'inference_time_ms': avg_time * 1000,
            'fps': 1.0 / avg_time,
            'peak_memory_mb': peak_memory,
            'efficiency_score': total_params / (avg_time * 1000)  # params per ms
        }


def create_squeezenet_for_lpips(version='1_0', pretrained=True):
    """Create SqueezeNet model specifically configured for LPIPS feature extraction"""
    model = SqueezeNet(version=version, num_classes=1000)
    
    if pretrained:
        try:
            # Load pretrained weights from torchvision
            if version == '1_0':
                pretrained_model = torchvision.models.squeezenet1_0(pretrained=True)
            elif version == '1_1':
                pretrained_model = torchvision.models.squeezenet1_1(pretrained=True)
            else:
                raise ValueError(f"Unsupported version: {version}")
            
            model.load_state_dict(pretrained_model.state_dict())
            print(f"Loaded pretrained SqueezeNet {version} weights from torchvision")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using random initialization")
    
    return model


def efficiency_comparison_study():
    """Comprehensive efficiency comparison study"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    squeezenet_1_0 = SqueezeNet(version='1_0')
    squeezenet_1_1 = SqueezeNet(version='1_1')
    
    models = {
        'SqueezeNet-1.0': squeezenet_1_0,
        'SqueezeNet-1.1': squeezenet_1_1
    }
    
    print("=== SqueezeNet Efficiency Comparison ===")
    
    for name, model in models.items():
        evaluator = SqueezeNetEvaluator(model, device)
        
        # Get model statistics
        param_count = model.get_parameter_count()
        fire_analysis = model.get_fire_module_analysis()
        alexnet_comparison = model.compare_with_alexnet()
        
        print(f"\n{name}:")
        print(f"  Parameters: {param_count['total_parameters']:,}")
        print(f"  Model size: {model.get_model_size():.2f} MB")
        print(f"  Fire modules: {len(fire_analysis)}")
        print(f"  vs AlexNet: {alexnet_comparison['parameter_reduction_factor']:.1f}x fewer params")
        print(f"  vs AlexNet: {alexnet_comparison['size_reduction_factor']:.1f}x smaller size")


def main():
    """Main function for SqueezeNet training and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create SqueezeNet 1.1 model (more efficient)
    model = SqueezeNet(version='1_1', num_classes=1000)
    
    # Model analysis
    print("\n=== SqueezeNet-1.1 Model Analysis ===")
    param_count = model.get_parameter_count()
    fire_analysis = model.get_fire_module_analysis()
    alexnet_comparison = model.compare_with_alexnet()
    
    print(f"Total parameters: {param_count['total_parameters']:,}")
    print(f"Fire module parameters: {param_count['fire_module_parameters']:,}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    print(f"Fire modules: {len(fire_analysis)}")
    print(f"Parameter reduction vs AlexNet: {alexnet_comparison['parameter_reduction_factor']:.1f}x")
    print(f"Size reduction vs AlexNet: {alexnet_comparison['size_reduction_factor']:.1f}x")
    
    # Efficiency comparison study
    print("\n=== Efficiency Comparison Study ===")
    efficiency_comparison_study()
    
    # Create trainer
    trainer = SqueezeNetTrainer(model, device)
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Train model (reduced epochs for demonstration)
    trainer.train(num_epochs=5)  # Change to 120 for full training
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    evaluator = SqueezeNetEvaluator(model, device)
    
    # Classification evaluation
    eval_results = evaluator.evaluate_classification(trainer.val_loader)
    print(f"\n=== SqueezeNet-1.1 Evaluation Results ===")
    print(f"Top-1 Accuracy: {eval_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {eval_results['top5_accuracy']:.2f}%")
    print(f"Inference FPS: {eval_results['inference_fps']:.1f}")
    print(f"Mean Confidence: {eval_results['mean_confidence']:.3f}")
    
    # Efficiency analysis
    efficiency = evaluator.analyze_efficiency_vs_accuracy(trainer.val_loader)
    print(f"\n=== Efficiency Analysis ===")
    for batch_config, metrics in efficiency.items():
        print(f"{batch_config}: {metrics['inference_time_per_image']*1000:.2f} ms/image, {metrics['fps']:.1f} FPS")
    
    # Memory efficiency
    memory_analysis = evaluator.measure_memory_efficiency()
    print(f"\n=== Memory Efficiency ===")
    for batch_config, metrics in memory_analysis.items():
        if 'memory_used_mb' in metrics:
            print(f"{batch_config}: {metrics['memory_per_image_mb']:.2f} MB/image")
    
    # Save final model
    torch.save(model.state_dict(), 'squeezenet_1_1_final.pth')
    print("\nModel saved as 'squeezenet_1_1_final.pth'")


if __name__ == "__main__":
    main()