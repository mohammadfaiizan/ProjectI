"""
PyTorch Transfer Learning Syntax - Leveraging Pretrained Models
Comprehensive guide to transfer learning with pretrained models in PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import copy
import time

print("=== TRANSFER LEARNING SYNTAX ===")

# 1. LOADING PRETRAINED MODELS
print("\n1. LOADING PRETRAINED MODELS")

# Load various pretrained models
pretrained_models = {
    'resnet18': models.resnet18(pretrained=True),
    'resnet50': models.resnet50(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    'densenet121': models.densenet121(pretrained=True),
    'mobilenet_v2': models.mobilenet_v2(pretrained=True),
    'efficientnet_b0': models.efficientnet_b0(pretrained=True),
}

# Display model information
for name, model in pretrained_models.items():
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{name:15} - {total_params:>10,} parameters")

# 2. FEATURE EXTRACTION APPROACH
print("\n2. FEATURE EXTRACTION APPROACH")

class FeatureExtractor(nn.Module):
    """Use pretrained model as fixed feature extractor"""
    
    def __init__(self, backbone_name: str = 'resnet18', num_classes: int = 10):
        super(FeatureExtractor, self).__init__()
        
        # Load pretrained backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
            self.feature_size = 512
        elif backbone_name == 'vgg16':
            self.backbone = models.vgg16(pretrained=True)
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            self.feature_size = 25088  # 512 * 7 * 7
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=True)
            self.backbone.classifier = nn.Identity()
            self.feature_size = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features (frozen)
        with torch.no_grad():
            features = self.backbone(x)
            features = features.view(features.size(0), -1)
        
        # Classify (trainable)
        x = self.classifier(features)
        return x

# Test feature extractor
feature_extractor = FeatureExtractor('resnet18', num_classes=10)
test_input = torch.randn(4, 3, 224, 224)
output = feature_extractor(test_input)
print(f"FeatureExtractor output shape: {output.shape}")

# Check which parameters are trainable
trainable_params = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in feature_extractor.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# 3. FINE-TUNING APPROACH
print("\n3. FINE-TUNING APPROACH")

class FineTuner(nn.Module):
    """Fine-tune entire pretrained model"""
    
    def __init__(self, backbone_name: str = 'resnet18', num_classes: int = 10, 
                 freeze_early_layers: bool = True):
        super(FineTuner, self).__init__()
        
        if backbone_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Replace final layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        elif backbone_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # Replace classifier
            self.model.classifier[6] = nn.Linear(4096, num_classes)
            
        elif backbone_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Optionally freeze early layers
        if freeze_early_layers:
            self._freeze_early_layers(backbone_name)
            
    def _freeze_early_layers(self, backbone_name: str):
        """Freeze early layers to preserve low-level features"""
        
        if backbone_name == 'resnet18':
            # Freeze conv1, bn1, layer1, layer2
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
                    
        elif backbone_name == 'vgg16':
            # Freeze first few convolutional layers
            for i, param in enumerate(self.model.features.parameters()):
                if i < 10:  # Freeze first 10 layers
                    param.requires_grad = False
                    
        elif backbone_name == 'densenet121':
            # Freeze features.denseblock1 and features.denseblock2
            for name, param in self.model.named_parameters():
                if 'features.denseblock1' in name or 'features.denseblock2' in name:
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Test fine-tuner
fine_tuner = FineTuner('resnet18', num_classes=10, freeze_early_layers=True)
output = fine_tuner(test_input)
print(f"FineTuner output shape: {output.shape}")

# Check trainable parameters
trainable_params = sum(p.numel() for p in fine_tuner.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in fine_tuner.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# 4. PROGRESSIVE UNFREEZING
print("\n4. PROGRESSIVE UNFREEZING")

class ProgressiveUnfreezer:
    """Progressively unfreeze layers during training"""
    
    def __init__(self, model: nn.Module, unfreeze_schedule: Dict[int, List[str]]):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.current_epoch = 0
        
    def update_epoch(self, epoch: int):
        """Update epoch and unfreeze layers if scheduled"""
        self.current_epoch = epoch
        
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            self._unfreeze_layers(layers_to_unfreeze)
            print(f"Epoch {epoch}: Unfreezing layers: {layers_to_unfreeze}")
            
    def _unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specified layers"""
        for name, param in self.model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break
                    
    def get_trainable_params(self) -> int:
        """Get current number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# Example progressive unfreezing schedule
resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 10)

# Freeze all layers initially
for param in resnet_model.parameters():
    param.requires_grad = False

# Only classifier trainable
for param in resnet_model.fc.parameters():
    param.requires_grad = True

# Unfreezing schedule
schedule = {
    0: ['fc'],           # Start with only classifier
    2: ['layer4'],       # Unfreeze layer4 at epoch 2
    4: ['layer3'],       # Unfreeze layer3 at epoch 4
    6: ['layer2'],       # Unfreeze layer2 at epoch 6
    8: ['layer1']        # Unfreeze layer1 at epoch 8
}

unfreezer = ProgressiveUnfreezer(resnet_model, schedule)

# Simulate training epochs
for epoch in range(10):
    unfreezer.update_epoch(epoch)
    trainable = unfreezer.get_trainable_params()
    if epoch % 2 == 0:
        print(f"  Epoch {epoch}: {trainable:,} trainable parameters")

# 5. LAYER-WISE LEARNING RATES
print("\n5. LAYER-WISE LEARNING RATES")

def get_layer_wise_params(model: nn.Module, base_lr: float = 1e-3) -> List[Dict[str, Any]]:
    """Get parameters with different learning rates for different layers"""
    
    params = []
    
    if hasattr(model, 'backbone') or isinstance(model, models.ResNet):
        # Handle ResNet-style models
        base_model = model if isinstance(model, models.ResNet) else model.model
        
        # Early layers - lower learning rate
        early_params = []
        for name, param in base_model.named_parameters():
            if any(layer in name for layer in ['conv1', 'bn1', 'layer1']):
                early_params.append(param)
                
        if early_params:
            params.append({'params': early_params, 'lr': base_lr * 0.1})
        
        # Middle layers - medium learning rate
        middle_params = []
        for name, param in base_model.named_parameters():
            if any(layer in name for layer in ['layer2', 'layer3']):
                middle_params.append(param)
                
        if middle_params:
            params.append({'params': middle_params, 'lr': base_lr * 0.5})
        
        # Later layers - higher learning rate
        later_params = []
        for name, param in base_model.named_parameters():
            if any(layer in name for layer in ['layer4', 'fc', 'classifier']):
                later_params.append(param)
                
        if later_params:
            params.append({'params': later_params, 'lr': base_lr})
    
    else:
        # Default: all parameters with base learning rate
        params.append({'params': model.parameters(), 'lr': base_lr})
    
    return params

# Test layer-wise learning rates
layer_params = get_layer_wise_params(fine_tuner, base_lr=1e-3)
optimizer = optim.Adam(layer_params)

print(f"Created optimizer with {len(layer_params)} parameter groups:")
for i, group in enumerate(layer_params):
    param_count = sum(p.numel() for p in group['params'])
    print(f"  Group {i}: LR={group['lr']:.1e}, Parameters={param_count:,}")

# 6. DOMAIN ADAPTATION STRATEGIES
print("\n6. DOMAIN ADAPTATION STRATEGIES")

class DomainAdaptationModel(nn.Module):
    """Model for domain adaptation with different strategies"""
    
    def __init__(self, backbone_name: str = 'resnet18', num_classes: int = 10,
                 adaptation_strategy: str = 'gradual'):
        super(DomainAdaptationModel, self).__init__()
        
        self.adaptation_strategy = adaptation_strategy
        
        # Load pretrained backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        self.feature_size = feature_size
        
        # Different adaptation strategies
        if adaptation_strategy == 'replace_classifier':
            # Simply replace the classifier
            self.classifier = nn.Linear(feature_size, num_classes)
            self._freeze_backbone()
            
        elif adaptation_strategy == 'add_adapter':
            # Add adapter layers
            self.adapter = nn.Sequential(
                nn.Linear(feature_size, feature_size // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feature_size // 4, feature_size)
            )
            self.classifier = nn.Linear(feature_size, num_classes)
            self._freeze_backbone()
            
        elif adaptation_strategy == 'gradual':
            # Gradual unfreezing with smaller learning rates
            self.classifier = nn.Linear(feature_size, num_classes)
            
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        
        if self.adaptation_strategy == 'add_adapter':
            # Apply adapter
            adapted_features = features + self.adapter(features)
            return self.classifier(adapted_features)
        else:
            return self.classifier(features)

# Test domain adaptation strategies
strategies = ['replace_classifier', 'add_adapter', 'gradual']

for strategy in strategies:
    model = DomainAdaptationModel('resnet18', num_classes=5, adaptation_strategy=strategy)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"{strategy:20} - Trainable: {trainable_params:>8,} / {total_params:>8,}")

# 7. MULTI-TASK TRANSFER LEARNING
print("\n7. MULTI-TASK TRANSFER LEARNING")

class MultiTaskTransferModel(nn.Module):
    """Multi-task model with shared backbone"""
    
    def __init__(self, backbone_name: str = 'resnet18', 
                 task_configs: Dict[str, int] = None):
        super(MultiTaskTransferModel, self).__init__()
        
        if task_configs is None:
            task_configs = {'classification': 10, 'regression': 1}
            
        # Shared backbone
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            feature_size = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        for task_name, output_size in task_configs.items():
            if task_name == 'classification':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, output_size)
                )
            elif task_name == 'regression':
                self.task_heads[task_name] = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(feature_size, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, output_size)
                )
            else:
                # Generic task head
                self.task_heads[task_name] = nn.Linear(feature_size, output_size)
                
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        # Shared feature extraction
        features = self.backbone(x)
        
        # Task-specific outputs
        outputs = {}
        
        if task is None:
            # Return all task outputs
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(features)
        else:
            # Return specific task output
            if task in self.task_heads:
                outputs[task] = self.task_heads[task](features)
            else:
                raise ValueError(f"Unknown task: {task}")
                
        return outputs

# Test multi-task model
task_configs = {'classification': 10, 'regression': 1, 'detection': 20}
multi_task_model = MultiTaskTransferModel('resnet18', task_configs)

# Forward pass
test_input = torch.randn(4, 3, 224, 224)
outputs = multi_task_model(test_input)

print("Multi-task outputs:")
for task, output in outputs.items():
    print(f"  {task:15} - {output.shape}")

# Task-specific forward pass
classification_output = multi_task_model(test_input, task='classification')
print(f"Classification only: {classification_output['classification'].shape}")

# 8. TRANSFER LEARNING TRAINING UTILITIES
print("\n8. TRANSFER LEARNING TRAINING UTILITIES")

class TransferLearningTrainer:
    """Trainer specifically designed for transfer learning"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        elif hasattr(self.model, 'features'):
            for param in self.model.features.parameters():
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.model.parameters():
            param.requires_grad = True
            
    def get_optimizer(self, strategy: str = 'different_lr', 
                     backbone_lr: float = 1e-4, head_lr: float = 1e-3):
        """Get optimizer with transfer learning strategies"""
        
        if strategy == 'different_lr':
            # Different learning rates for backbone and head
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if any(keyword in name for keyword in ['backbone', 'features']):
                        backbone_params.append(param)
                    else:
                        head_params.append(param)
            
            param_groups = []
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
            if head_params:
                param_groups.append({'params': head_params, 'lr': head_lr})
                
            return optim.Adam(param_groups)
            
        elif strategy == 'single_lr':
            return optim.Adam(self.model.parameters(), lr=head_lr)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
    def train_phase(self, phase_name: str, dataloader: DataLoader, 
                   epochs: int, optimizer: optim.Optimizer, 
                   criterion: nn.Module, scheduler: Optional[Any] = None):
        """Train for a specific phase"""
        
        print(f"Starting {phase_name} phase...")
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            if scheduler:
                scheduler.step()
                
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100. * correct / total
            
            self.training_history['train_loss'].append(epoch_loss)
            self.training_history['train_acc'].append(epoch_acc)
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")

# 9. COMPLETE TRANSFER LEARNING PIPELINE
print("\n9. COMPLETE TRANSFER LEARNING PIPELINE")

def create_dummy_dataset(num_samples: int = 1000, num_classes: int = 10):
    """Create dummy dataset for demonstration"""
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, num_classes):
            self.num_samples = num_samples
            self.num_classes = num_classes
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Random image and label
            image = torch.randn(3, 224, 224)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return image, label
    
    dataset = DummyDataset(num_samples, num_classes)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def transfer_learning_pipeline():
    """Complete transfer learning pipeline demonstration"""
    
    print("Transfer Learning Pipeline Demo:")
    print("=" * 50)
    
    # 1. Create model
    model = FeatureExtractor('resnet18', num_classes=5)
    trainer = TransferLearningTrainer(model)
    
    # 2. Create dummy data
    train_loader = create_dummy_dataset(200, 5)
    
    # 3. Phase 1: Train only classifier (backbone frozen)
    print("\nPhase 1: Feature extraction (backbone frozen)")
    trainer.freeze_backbone()
    optimizer1 = trainer.get_optimizer('single_lr', head_lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer.train_phase("Feature Extraction", train_loader, epochs=2, 
                       optimizer=optimizer1, criterion=criterion)
    
    # 4. Phase 2: Fine-tune entire model
    print("\nPhase 2: Fine-tuning (backbone unfrozen)")
    trainer.unfreeze_backbone()
    optimizer2 = trainer.get_optimizer('different_lr', backbone_lr=1e-5, head_lr=1e-4)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer.train_phase("Fine-tuning", train_loader, epochs=2, 
                       optimizer=optimizer2, criterion=criterion)
    
    print("\nTransfer learning pipeline completed!")

# Run the pipeline
transfer_learning_pipeline()

print("\n=== TRANSFER LEARNING SYNTAX COMPLETE ===")
print("Key concepts covered:")
print("- Loading pretrained models")
print("- Feature extraction approach")
print("- Fine-tuning strategies")
print("- Progressive unfreezing")
print("- Layer-wise learning rates")
print("- Domain adaptation strategies")
print("- Multi-task transfer learning")
print("- Complete training pipeline")