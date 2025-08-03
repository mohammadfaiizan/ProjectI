"""
VGG Implementation for LPIPS Supporting Model
============================================

Complete VGG implementation with ImageNet training, evaluation, and parameter analysis.
Based on the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"
by Karen Simonyan and Andrew Zisserman.

This implementation includes:
- VGG-16 and VGG-19 architectures
- ImageNet training pipeline
- Evaluation and testing framework
- Parameter analysis for LPIPS feature extraction
- Performance benchmarking and comparison

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


class VGG(nn.Module):
    """
    VGG architecture implementation supporting VGG-11, VGG-13, VGG-16, and VGG-19
    
    Original paper: https://arxiv.org/abs/1409.1556
    
    Key principles:
    - Small 3x3 convolution filters throughout
    - Max pooling with 2x2 windows and stride 2
    - ReLU activation after each convolution
    - Three fully connected layers at the end
    - Batch normalization option for improved training
    """
    
    # VGG configurations
    VGG_CONFIGS = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    
    def __init__(self, architecture='vgg16', num_classes=1000, batch_norm=False, dropout=0.5):
        super(VGG, self).__init__()
        
        self.architecture = architecture
        self.batch_norm = batch_norm
        
        # Create feature extraction layers
        self.features = self._make_layers(self.VGG_CONFIGS[architecture], batch_norm)
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Store information for LPIPS feature extraction
        self.lpips_layers = self._get_lpips_layers()
    
    def _make_layers(self, config, batch_norm=False):
        """Create the feature extraction layers based on configuration"""
        layers = []
        in_channels = 3
        
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers)
    
    def _get_lpips_layers(self):
        """Get layer indices for LPIPS feature extraction (VGG-16 specific)"""
        if self.architecture == 'vgg16':
            # Corresponding to conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
            return [3, 8, 15, 22, 29]  # After ReLU activations
        elif self.architecture == 'vgg19':
            # Corresponding to conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
            return [3, 8, 17, 26, 35]
        else:
            # Default for other architectures
            layer_indices = []
            conv_count = 0
            for i, layer in enumerate(self.features):
                if isinstance(layer, nn.Conv2d):
                    conv_count += 1
                    if conv_count in [2, 4, 7, 10, 13]:  # Representative layers
                        layer_indices.append(i + 1)  # After ReLU
            return layer_indices
    
    def _initialize_weights(self):
        """Initialize weights according to original VGG paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through VGG"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
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
    
    def get_feature_maps_at_layers(self, x, layer_indices=None):
        """Get feature maps at specific layers"""
        if layer_indices is None:
            layer_indices = self.lpips_layers
        
        feature_maps = {}
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in layer_indices:
                feature_maps[f'layer_{i}'] = x.clone()
        
        return feature_maps
    
    def get_parameter_count(self):
        """Get detailed parameter count analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Separate feature and classifier parameters
        feature_params = sum(p.numel() for p in self.features.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        # Layer-wise parameter count
        layer_params = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                params = sum(p.numel() for p in module.parameters())
                layer_params[name] = params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_parameters': feature_params,
            'classifier_parameters': classifier_params,
            'layer_parameters': layer_params
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
    
    def analyze_receptive_field(self):
        """Analyze receptive field sizes for each layer"""
        receptive_fields = {}
        
        # Calculate receptive field for VGG-16
        if self.architecture == 'vgg16':
            rf_size = 1
            stride = 1
            
            layer_info = [
                ('conv1_1', 3, 1), ('conv1_2', 3, 1), ('pool1', 2, 2),
                ('conv2_1', 3, 1), ('conv2_2', 3, 1), ('pool2', 2, 2),
                ('conv3_1', 3, 1), ('conv3_2', 3, 1), ('conv3_3', 3, 1), ('pool3', 2, 2),
                ('conv4_1', 3, 1), ('conv4_2', 3, 1), ('conv4_3', 3, 1), ('pool4', 2, 2),
                ('conv5_1', 3, 1), ('conv5_2', 3, 1), ('conv5_3', 3, 1), ('pool5', 2, 2),
            ]
            
            for layer_name, kernel_size, layer_stride in layer_info:
                rf_size = rf_size + (kernel_size - 1) * stride
                stride = stride * layer_stride
                receptive_fields[layer_name] = {
                    'receptive_field': rf_size,
                    'stride': stride
                }
        
        return receptive_fields


class VGGTrainer:
    """
    Comprehensive training framework for VGG on ImageNet
    """
    
    def __init__(self, model, device='cuda', data_dir='./data'):
        self.model = model.to(device)
        self.device = device
        self.data_dir = data_dir
        
        # Training hyperparameters (from original paper)
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.batch_size = 256
        self.num_epochs = 74  # Original paper epochs
        
        # Learning rate schedule (divide by 10 every 30 epochs)
        self.lr_schedule = [30, 60]
        self.lr_decay = 0.1
        
        # Initialize optimizer and criterion
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=self.lr_schedule, 
            gamma=self.lr_decay
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epoch_times': []
        }
    
    def setup_data_loaders(self):
        """Setup ImageNet data loaders with VGG-specific preprocessing"""
        
        # VGG preprocessing (from original paper)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Color jittering as mentioned in paper
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
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.train_history['learning_rates'].append(current_lr)
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
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
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'LR: {current_lr:.6f}')
        
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_acc)
        self.train_history['epoch_times'].append(epoch_time)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        top5_correct = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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
        
        self.train_history['val_loss'].append(val_loss)
        self.train_history['val_accuracy'].append(val_acc)
        
        return val_loss, val_acc, top5_acc
    
    def train(self, num_epochs=None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        print(f"Starting VGG-{self.model.architecture} training...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial learning rate: {self.learning_rate}")
        print(f"Batch norm: {self.model.batch_norm}")
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate
            val_loss, val_acc, top5_acc = self.validate()
            
            print(f'\nEpoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Top-5 Acc: {top5_acc:.2f}%')
            print(f'Epoch Time: {self.train_history["epoch_times"][-1]:.2f}s')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, best=True)
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/3600:.2f} hours')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'architecture': self.model.architecture,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'model_config': {
                'architecture': self.model.architecture,
                'num_classes': self.model.classifier[-1].out_features,
                'batch_norm': self.model.batch_norm
            }
        }
        
        filename = f'checkpoints/vgg_{self.model.architecture}_best.pth' if best else f'checkpoints/vgg_{self.model.architecture}_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss plot
        ax1.plot(self.train_history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.train_history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title(f'VGG-{self.model.architecture} Loss History', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_history['accuracy'], label='Train Acc', linewidth=2)
        ax2.plot(self.train_history['val_accuracy'], label='Val Acc', linewidth=2)
        ax2.set_title(f'VGG-{self.model.architecture} Accuracy History', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(self.train_history['learning_rates'], linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Epoch time plot
        ax4.plot(self.train_history['epoch_times'], linewidth=2, color='orange')
        ax4.set_title('Training Time per Epoch', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'vgg_{self.model.architecture}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class VGGEvaluator:
    """
    Comprehensive evaluation framework for VGG models
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_classification(self, data_loader, detailed=True):
        """Comprehensive classification evaluation"""
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        class_correct = {}
        class_total = {}
        predictions = []
        ground_truth = []
        confidence_scores = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
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
        
        result = {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'total_samples': total_samples
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
    
    def analyze_feature_representations(self, sample_images, layer_analysis=True):
        """Comprehensive feature representation analysis"""
        feature_analysis = {}
        
        with torch.no_grad():
            for img_idx, img in enumerate(sample_images):
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
                
                # Extract features from LPIPS layers
                features = self.model.forward_features(img)
                
                img_analysis = {}
                for layer_name, feature_map in features.items():
                    # Comprehensive feature statistics
                    feature_stats = {
                        'shape': list(feature_map.shape),
                        'mean': feature_map.mean().item(),
                        'std': feature_map.std().item(),
                        'min': feature_map.min().item(),
                        'max': feature_map.max().item(),
                        'sparsity': (feature_map == 0).float().mean().item(),
                        'l1_norm': feature_map.abs().mean().item(),
                        'l2_norm': feature_map.pow(2).mean().sqrt().item()
                    }
                    
                    if layer_analysis:
                        # Channel-wise statistics
                        channel_means = feature_map.mean(dim=[2, 3]).squeeze().cpu().numpy()
                        channel_stds = feature_map.std(dim=[2, 3]).squeeze().cpu().numpy()
                        
                        feature_stats.update({
                            'channel_mean_avg': np.mean(channel_means),
                            'channel_mean_std': np.std(channel_means),
                            'channel_std_avg': np.mean(channel_stds),
                            'channel_std_std': np.std(channel_stds)
                        })
                    
                    img_analysis[layer_name] = feature_stats
                
                feature_analysis[f'image_{img_idx}'] = img_analysis
        
        return feature_analysis
    
    def measure_computational_efficiency(self, input_sizes=None, num_runs=100):
        """Measure computational efficiency across different input sizes"""
        if input_sizes is None:
            input_sizes = [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224)]
        
        efficiency_results = {}
        
        for input_size in input_sizes:
            # Warm up
            dummy_input = torch.randn(input_size).to(self.device)
            for _ in range(10):
                _ = self.model(dummy_input)
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = self.model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_runs
            avg_time_per_image = avg_time_per_batch / input_size[0]
            
            efficiency_results[f'batch_{input_size[0]}'] = {
                'input_size': input_size,
                'total_time_s': total_time,
                'avg_time_per_batch_ms': avg_time_per_batch * 1000,
                'avg_time_per_image_ms': avg_time_per_image * 1000,
                'fps': 1.0 / avg_time_per_image,
                'throughput_images_per_sec': input_size[0] / avg_time_per_batch
            }
        
        return efficiency_results
    
    def analyze_model_complexity(self):
        """Comprehensive model complexity analysis"""
        # Parameter analysis
        param_analysis = self.model.get_parameter_count()
        model_size_mb = self.model.get_model_size()
        
        # Receptive field analysis
        rf_analysis = self.model.analyze_receptive_field()
        
        # FLOPs estimation for VGG-16
        def estimate_conv_flops(in_channels, out_channels, kernel_size, input_size, padding=1):
            """Estimate FLOPs for a convolutional layer"""
            output_h = (input_size[0] + 2 * padding - kernel_size) + 1
            output_w = (input_size[1] + 2 * padding - kernel_size) + 1
            kernel_flops = kernel_size * kernel_size * in_channels
            output_elements = out_channels * output_h * output_w
            return kernel_flops * output_elements, (output_h, output_w)
        
        # Estimate total FLOPs
        flops_analysis = {}
        total_flops = 0
        current_size = [224, 224]
        
        if self.model.architecture == 'vgg16':
            layers_config = [
                ('conv1_1', 3, 64, 3), ('conv1_2', 64, 64, 3),  # Pool: 112x112
                ('conv2_1', 64, 128, 3), ('conv2_2', 128, 128, 3),  # Pool: 56x56
                ('conv3_1', 128, 256, 3), ('conv3_2', 256, 256, 3), ('conv3_3', 256, 256, 3),  # Pool: 28x28
                ('conv4_1', 256, 512, 3), ('conv4_2', 512, 512, 3), ('conv4_3', 512, 512, 3),  # Pool: 14x14
                ('conv5_1', 512, 512, 3), ('conv5_2', 512, 512, 3), ('conv5_3', 512, 512, 3),  # Pool: 7x7
            ]
            
            for layer_name, in_ch, out_ch, kernel_size in layers_config:
                layer_flops, new_size = estimate_conv_flops(in_ch, out_ch, kernel_size, current_size)
                flops_analysis[layer_name] = layer_flops
                total_flops += layer_flops
                
                # Update size after pooling layers
                if layer_name.endswith('_2') or layer_name.endswith('_3'):
                    if 'conv1_2' in layer_name or 'conv2_2' in layer_name:
                        current_size = [s // 2 for s in new_size]
                    elif layer_name in ['conv3_3', 'conv4_3', 'conv5_3']:
                        current_size = [s // 2 for s in new_size]
                else:
                    current_size = new_size
        
        # Classifier FLOPs
        classifier_flops = 512 * 7 * 7 * 4096 + 4096 * 4096 + 4096 * 1000
        total_flops += classifier_flops
        flops_analysis['classifier'] = classifier_flops
        
        return {
            'parameter_analysis': param_analysis,
            'model_size_mb': model_size_mb,
            'receptive_field_analysis': rf_analysis,
            'flops_analysis': flops_analysis,
            'total_flops': total_flops,
            'architecture': self.model.architecture,
            'batch_norm': self.model.batch_norm
        }
    
    def compare_with_other_models(self, other_models, test_loader):
        """Compare VGG with other model architectures"""
        comparison_results = {}
        
        # Evaluate current VGG model
        vgg_results = self.evaluate_classification(test_loader, detailed=False)
        vgg_complexity = self.analyze_model_complexity()
        vgg_efficiency = self.measure_computational_efficiency()
        
        comparison_results[f'VGG-{self.model.architecture}'] = {
            'accuracy': vgg_results,
            'complexity': vgg_complexity,
            'efficiency': vgg_efficiency
        }
        
        # Evaluate other models
        for model_name, model in other_models.items():
            model.eval()
            model_evaluator = VGGEvaluator(model, self.device)
            
            model_results = model_evaluator.evaluate_classification(test_loader, detailed=False)
            model_complexity = model_evaluator.analyze_model_complexity()
            model_efficiency = model_evaluator.measure_computational_efficiency()
            
            comparison_results[model_name] = {
                'accuracy': model_results,
                'complexity': model_complexity,
                'efficiency': model_efficiency
            }
        
        return comparison_results


def create_vgg_for_lpips(architecture='vgg16', pretrained=True):
    """Create VGG model specifically configured for LPIPS feature extraction"""
    model = VGG(architecture=architecture, num_classes=1000, batch_norm=False)
    
    if pretrained:
        try:
            # Load pretrained weights from torchvision
            if architecture == 'vgg16':
                pretrained_model = torchvision.models.vgg16(pretrained=True)
            elif architecture == 'vgg19':
                pretrained_model = torchvision.models.vgg19(pretrained=True)
            elif architecture == 'vgg11':
                pretrained_model = torchvision.models.vgg11(pretrained=True)
            elif architecture == 'vgg13':
                pretrained_model = torchvision.models.vgg13(pretrained=True)
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
            
            model.load_state_dict(pretrained_model.state_dict())
            print(f"Loaded pretrained {architecture.upper()} weights from torchvision")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
            print("Using random initialization")
    
    return model


def compare_vgg_architectures():
    """Compare different VGG architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    architectures = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    comparison_data = {}
    
    for arch in architectures:
        print(f"\nAnalyzing {arch.upper()}...")
        model = VGG(architecture=arch)
        evaluator = VGGEvaluator(model, device)
        
        # Get complexity analysis
        complexity = evaluator.analyze_model_complexity()
        
        comparison_data[arch] = {
            'total_parameters': complexity['parameter_analysis']['total_parameters'],
            'model_size_mb': complexity['model_size_mb'],
            'total_flops': complexity.get('total_flops', 0)
        }
        
        print(f"Parameters: {comparison_data[arch]['total_parameters']:,}")
        print(f"Model size: {comparison_data[arch]['model_size_mb']:.2f} MB")
    
    return comparison_data


def main():
    """Main function for VGG training and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create VGG-16 model
    model = VGG(architecture='vgg16', num_classes=1000, batch_norm=False)
    
    # Model analysis
    print("\n=== VGG-16 Model Analysis ===")
    param_count = model.get_parameter_count()
    print(f"Total parameters: {param_count['total_parameters']:,}")
    print(f"Feature parameters: {param_count['feature_parameters']:,}")
    print(f"Classifier parameters: {param_count['classifier_parameters']:,}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    
    # Compare architectures
    print("\n=== VGG Architecture Comparison ===")
    arch_comparison = compare_vgg_architectures()
    
    # Create trainer
    trainer = VGGTrainer(model, device)
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Train model (reduced epochs for demonstration)
    trainer.train(num_epochs=5)  # Change to 74 for full training
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    evaluator = VGGEvaluator(model, device)
    
    # Classification evaluation
    eval_results = evaluator.evaluate_classification(trainer.val_loader)
    print(f"\n=== VGG-16 Evaluation Results ===")
    print(f"Top-1 Accuracy: {eval_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {eval_results['top5_accuracy']:.2f}%")
    print(f"Mean Confidence: {eval_results['mean_confidence']:.3f}")
    
    # Efficiency analysis
    efficiency = evaluator.measure_computational_efficiency()
    print(f"\n=== Performance Analysis ===")
    for batch_size, metrics in efficiency.items():
        print(f"{batch_size}: {metrics['avg_time_per_image_ms']:.2f} ms/image, {metrics['fps']:.1f} FPS")
    
    # Complexity analysis
    complexity = evaluator.analyze_model_complexity()
    print(f"\n=== Complexity Analysis ===")
    print(f"Total FLOPs: {complexity['total_flops']:,}")
    
    # Save final model
    torch.save(model.state_dict(), 'vgg16_final.pth')
    print("\nModel saved as 'vgg16_final.pth'")


if __name__ == "__main__":
    main()