"""
AlexNet Implementation for LPIPS Supporting Model
================================================

Complete AlexNet implementation with ImageNet training, evaluation, and parameter analysis.
Based on the original 2012 paper "ImageNet Classification with Deep Convolutional Neural Networks"
by Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.

This implementation includes:
- Original AlexNet architecture
- ImageNet training pipeline
- Evaluation and testing framework
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


class AlexNet(nn.Module):
    """
    AlexNet architecture implementation
    
    Original paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    
    Architecture:
    - Input: 227x227x3 (original) or 224x224x3 (adapted)
    - Conv1: 96 filters, 11x11, stride 4, padding 2
    - MaxPool1: 3x3, stride 2
    - Conv2: 256 filters, 5x5, stride 1, padding 2
    - MaxPool2: 3x3, stride 2
    - Conv3: 384 filters, 3x3, stride 1, padding 1
    - Conv4: 384 filters, 3x3, stride 1, padding 1
    - Conv5: 256 filters, 3x3, stride 1, padding 1
    - MaxPool3: 3x3, stride 2
    - FC1: 4096 units
    - FC2: 4096 units
    - FC3: 1000 units (ImageNet classes)
    """
    
    def __init__(self, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1: Input 224x224x3 -> Output 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: Input 27x27x96 -> Output 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: Input 13x13x256 -> Output 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: Input 13x13x384 -> Output 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: Input 13x13x384 -> Output 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Store layer names for LPIPS feature extraction
        self.lpips_layers = ['features.0', 'features.3', 'features.6', 'features.8', 'features.10']
    
    def _initialize_weights(self):
        """Initialize weights according to original AlexNet paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through AlexNet"""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward_features(self, x):
        """
        Forward pass with intermediate feature extraction for LPIPS
        
        Returns:
            dict: Dictionary containing features from each layer
        """
        features = {}
        
        # Process through feature layers
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Store features from conv layers (before pooling)
            if i in [0, 3, 6, 8, 10]:  # Conv1, Conv2, Conv3, Conv4, Conv5
                layer_name = f'features.{i}'
                features[layer_name] = x.clone()
        
        return features
    
    def get_parameter_count(self):
        """Get detailed parameter count analysis"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_params = {}
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params = sum(p.numel() for p in module.parameters())
                layer_params[name] = params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
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


class AlexNetTrainer:
    """
    Comprehensive training framework for AlexNet on ImageNet
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
        self.num_epochs = 90
        
        # Learning rate schedule
        self.lr_schedule = {30: 0.1, 60: 0.01, 80: 0.001}
        
        # Initialize optimizer and criterion
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def setup_data_loaders(self):
        """Setup ImageNet data loaders with proper preprocessing"""
        
        # ImageNet preprocessing (from original paper)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
                download=False  # ImageNet must be manually downloaded
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
        
        # Adjust learning rate according to schedule
        if epoch in self.lr_schedule:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.lr_schedule[epoch]
        
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
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.train_history['val_loss'].append(val_loss)
        self.train_history['val_accuracy'].append(val_acc)
        
        return val_loss, val_acc
    
    def train(self, num_epochs=None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        print("Starting AlexNet training...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial learning rate: {self.learning_rate}")
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            print(f'\nEpoch: {epoch+1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            print('-' * 50)
            
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'model_config': {
                'num_classes': self.model.classifier[-1].out_features,
                'dropout': 0.5
            }
        }
        
        filename = 'checkpoints/alexnet_best.pth' if best else f'checkpoints/alexnet_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
        print(f'Checkpoint saved: {filename}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_history['loss'], label='Train Loss')
        ax1.plot(self.train_history['val_loss'], label='Val Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_history['accuracy'], label='Train Acc')
        ax2.plot(self.train_history['val_accuracy'], label='Val Acc')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(self.train_history['learning_rates'])
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Combined loss and accuracy
        ax4_twin = ax4.twinx()
        line1, = ax4.plot(self.train_history['loss'], 'b-', label='Train Loss')
        line2, = ax4_twin.plot(self.train_history['accuracy'], 'r-', label='Train Acc')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='b')
        ax4_twin.set_ylabel('Accuracy (%)', color='r')
        ax4.set_title('Loss vs Accuracy')
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('alexnet_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


class AlexNetEvaluator:
    """
    Comprehensive evaluation framework for AlexNet
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_classification(self, data_loader):
        """Evaluate classification performance"""
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        class_correct = {}
        class_total = {}
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Top-1 accuracy
                _, pred = output.max(1)
                correct_predictions += pred.eq(target).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = output.topk(5, 1, largest=True, sorted=True)
                top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()
                
                total_samples += target.size(0)
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    if label not in class_correct:
                        class_correct[label] = 0
                        class_total[label] = 0
                    class_total[label] += 1
                    if pred[i] == label:
                        class_correct[label] += 1
                
                predictions.extend(pred.cpu().numpy())
                ground_truth.extend(target.cpu().numpy())
        
        # Calculate metrics
        top1_accuracy = 100. * correct_predictions / total_samples
        top5_accuracy = 100. * top5_correct / total_samples
        
        # Per-class accuracy
        per_class_accuracy = {}
        for label in class_correct:
            per_class_accuracy[label] = 100. * class_correct[label] / class_total[label]
        
        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'total_samples': total_samples
        }
    
    def analyze_feature_maps(self, sample_images):
        """Analyze feature maps for LPIPS understanding"""
        feature_analysis = {}
        
        with torch.no_grad():
            for img_idx, img in enumerate(sample_images):
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
                
                # Extract features
                features = self.model.forward_features(img)
                
                img_analysis = {}
                for layer_name, feature_map in features.items():
                    # Feature map statistics
                    img_analysis[layer_name] = {
                        'shape': list(feature_map.shape),
                        'mean': feature_map.mean().item(),
                        'std': feature_map.std().item(),
                        'min': feature_map.min().item(),
                        'max': feature_map.max().item(),
                        'sparsity': (feature_map == 0).float().mean().item()
                    }
                
                feature_analysis[f'image_{img_idx}'] = img_analysis
        
        return feature_analysis
    
    def measure_inference_time(self, input_size=(1, 3, 224, 224), num_runs=100):
        """Measure inference time performance"""
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
        
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        return {
            'average_inference_time_ms': avg_inference_time,
            'fps': 1000 / avg_inference_time,
            'input_size': input_size,
            'num_runs': num_runs
        }
    
    def analyze_model_complexity(self):
        """Analyze model computational complexity"""
        # Parameter analysis
        param_analysis = self.model.get_parameter_count()
        model_size_mb = self.model.get_model_size()
        
        # FLOPs estimation (simplified)
        def estimate_conv_flops(in_channels, out_channels, kernel_size, input_size):
            kernel_flops = kernel_size * kernel_size * in_channels
            output_elements = out_channels * input_size[0] * input_size[1]
            return kernel_flops * output_elements
        
        # Estimate FLOPs for each layer (simplified)
        flops_analysis = {}
        input_size = [224, 224]
        
        # Conv1: 96 filters, 11x11, stride 4
        conv1_flops = estimate_conv_flops(3, 96, 11, [55, 55])
        flops_analysis['conv1'] = conv1_flops
        
        # Conv2: 256 filters, 5x5
        conv2_flops = estimate_conv_flops(96, 256, 5, [27, 27])
        flops_analysis['conv2'] = conv2_flops
        
        # Conv3: 384 filters, 3x3
        conv3_flops = estimate_conv_flops(256, 384, 3, [13, 13])
        flops_analysis['conv3'] = conv3_flops
        
        # Conv4: 384 filters, 3x3
        conv4_flops = estimate_conv_flops(384, 384, 3, [13, 13])
        flops_analysis['conv4'] = conv4_flops
        
        # Conv5: 256 filters, 3x3
        conv5_flops = estimate_conv_flops(384, 256, 3, [13, 13])
        flops_analysis['conv5'] = conv5_flops
        
        total_flops = sum(flops_analysis.values())
        
        return {
            'parameter_analysis': param_analysis,
            'model_size_mb': model_size_mb,
            'flops_analysis': flops_analysis,
            'total_flops': total_flops,
            'flops_per_layer': flops_analysis
        }


def create_alexnet_for_lpips():
    """Create AlexNet model specifically configured for LPIPS feature extraction"""
    model = AlexNet(num_classes=1000)
    
    # Load pretrained weights if available
    try:
        # Try to load torchvision pretrained weights
        pretrained_model = torchvision.models.alexnet(pretrained=True)
        model.load_state_dict(pretrained_model.state_dict())
        print("Loaded pretrained AlexNet weights from torchvision")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Using random initialization")
    
    return model


def main():
    """Main function for AlexNet training and evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = AlexNet(num_classes=1000)
    
    # Model analysis
    print("\n=== AlexNet Model Analysis ===")
    param_count = model.get_parameter_count()
    print(f"Total parameters: {param_count['total_parameters']:,}")
    print(f"Trainable parameters: {param_count['trainable_parameters']:,}")
    print(f"Model size: {model.get_model_size():.2f} MB")
    
    # Create trainer
    trainer = AlexNetTrainer(model, device)
    
    # Setup data loaders
    trainer.setup_data_loaders()
    
    # Train model (reduced epochs for demonstration)
    trainer.train(num_epochs=5)  # Change to 90 for full training
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    evaluator = AlexNetEvaluator(model, device)
    
    # Classification evaluation
    eval_results = evaluator.evaluate_classification(trainer.val_loader)
    print(f"\n=== Evaluation Results ===")
    print(f"Top-1 Accuracy: {eval_results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {eval_results['top5_accuracy']:.2f}%")
    
    # Performance analysis
    inference_time = evaluator.measure_inference_time()
    print(f"\n=== Performance Analysis ===")
    print(f"Average inference time: {inference_time['average_inference_time_ms']:.2f} ms")
    print(f"FPS: {inference_time['fps']:.1f}")
    
    # Complexity analysis
    complexity = evaluator.analyze_model_complexity()
    print(f"\n=== Complexity Analysis ===")
    print(f"Total FLOPs: {complexity['total_flops']:,}")
    
    # Save final model
    torch.save(model.state_dict(), 'alexnet_final.pth')
    print("\nModel saved as 'alexnet_final.pth'")


if __name__ == "__main__":
    main()