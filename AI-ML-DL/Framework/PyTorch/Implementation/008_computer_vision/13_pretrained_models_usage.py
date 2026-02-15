import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.models import ResNet50_Weights, VGG16_Weights, EfficientNet_B0_Weights

# Loading Pretrained Models
def load_pretrained_models():
    """Load various pretrained models from torchvision"""
    
    # ResNet variants
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    
    # VGG variants
    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    # DenseNet variants
    densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    densenet169 = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
    densenet201 = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    
    # EfficientNet variants
    efficientnet_b0 = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    efficientnet_b1 = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
    efficientnet_b7 = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    
    # Vision Transformer
    vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit_l_16 = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
    
    # MobileNet variants
    mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    mobilenet_v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # RegNet variants
    regnet_y_400mf = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2)
    regnet_x_8gf = models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.IMAGENET1K_V2)
    
    # ConvNeXt
    convnext_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    convnext_base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    
    models_dict = {
        'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50,
        'resnet101': resnet101, 'resnet152': resnet152,
        'vgg11': vgg11, 'vgg16': vgg16, 'vgg19': vgg19,
        'densenet121': densenet121, 'densenet169': densenet169, 'densenet201': densenet201,
        'efficientnet_b0': efficientnet_b0, 'efficientnet_b1': efficientnet_b1, 'efficientnet_b7': efficientnet_b7,
        'vit_b_16': vit_b_16, 'vit_l_16': vit_l_16,
        'mobilenet_v2': mobilenet_v2, 'mobilenet_v3_large': mobilenet_v3_large, 'mobilenet_v3_small': mobilenet_v3_small,
        'regnet_y_400mf': regnet_y_400mf, 'regnet_x_8gf': regnet_x_8gf,
        'convnext_tiny': convnext_tiny, 'convnext_base': convnext_base
    }
    
    return models_dict

# Transfer Learning Strategies
class FeatureExtractor(nn.Module):
    """Use pretrained model as feature extractor (freeze all layers)"""
    def __init__(self, model_name='resnet50', num_classes=10):
        super().__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.classifier = nn.Linear(2048, num_classes)
        elif model_name == 'vgg16':
            self.backbone = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
            self.classifier = nn.Linear(4096, num_classes)
        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()
            self.classifier = nn.Linear(1280, num_classes)
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

class FineTuner(nn.Module):
    """Fine-tune pretrained model (unfreeze some layers)"""
    def __init__(self, model_name='resnet50', num_classes=10, freeze_layers=True):
        super().__init__()
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
            if freeze_layers:
                # Freeze early layers, unfreeze later layers
                for name, param in self.model.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False
                        
        elif model_name == 'vgg16':
            self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
            
            if freeze_layers:
                # Freeze feature extractor
                for param in self.model.features.parameters():
                    param.requires_grad = False
                    
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            
            if freeze_layers:
                # Freeze early blocks
                for name, param in self.model.named_parameters():
                    if 'features.7' not in name and 'features.8' not in name and 'classifier' not in name:
                        param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

class MultiModelEnsemble(nn.Module):
    """Ensemble of multiple pretrained models"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Load different pretrained models
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
        
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)
        
        self.weights = nn.Parameter(torch.ones(3) / 3)  # Learnable ensemble weights
        
    def forward(self, x):
        resnet_out = self.resnet(x)
        efficientnet_out = self.efficientnet(x)
        vgg_out = self.vgg(x)
        
        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        ensemble_out = (weights[0] * resnet_out + 
                       weights[1] * efficientnet_out + 
                       weights[2] * vgg_out)
        
        return ensemble_out

# Advanced Transfer Learning Techniques
class LayerWiseFineTuning(nn.Module):
    """Gradual unfreezing of layers during training"""
    def __init__(self, model_name='resnet50', num_classes=10):
        super().__init__()
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            
        self.layer_groups = self._get_layer_groups()
        self.current_unfrozen = 0
        self._freeze_all()
    
    def _get_layer_groups(self):
        """Define layer groups for gradual unfreezing"""
        return [
            ['fc'],  # Final classifier
            ['layer4'],  # Last residual block
            ['layer3'],  # Third residual block
            ['layer2'],  # Second residual block
            ['layer1'],  # First residual block
            ['conv1', 'bn1']  # Initial conv layer
        ]
    
    def _freeze_all(self):
        """Freeze all parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_next_layer_group(self):
        """Unfreeze the next layer group"""
        if self.current_unfrozen < len(self.layer_groups):
            group = self.layer_groups[self.current_unfrozen]
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in group):
                    param.requires_grad = True
            self.current_unfrozen += 1
            return True
        return False
    
    def forward(self, x):
        return self.model(x)

class AdaptiveBatchNorm(nn.Module):
    """Adaptive batch normalization for domain adaptation"""
    def __init__(self, model_name='resnet50', num_classes=10):
        super().__init__()
        
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace batch norm layers with adaptive ones
        self._replace_bn_layers()
        
        # Replace classifier
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Freeze feature layers, only train BN and classifier
        for name, param in self.backbone.named_parameters():
            if 'bn' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def _replace_bn_layers(self):
        """Replace standard BN with adaptive BN"""
        def replace_bn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    # Keep the same parameters but make them trainable
                    setattr(module, name, nn.BatchNorm2d(child.num_features))
                else:
                    replace_bn(child)
        
        replace_bn(self.backbone)
    
    def forward(self, x):
        return self.backbone(x)

# Model Comparison and Selection
def compare_models(models_dict, test_data):
    """Compare performance of different pretrained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for name, model in models_dict.items():
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_data:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        results[name] = accuracy
        print(f"{name}: {accuracy:.2f}%")
    
    return results

# Efficient Model Loading and Caching
class ModelCache:
    """Cache for pretrained models to avoid repeated downloads"""
    def __init__(self):
        self.cache = {}
    
    def get_model(self, model_name, **kwargs):
        """Get model from cache or load if not cached"""
        cache_key = f"{model_name}_{hash(str(sorted(kwargs.items())))}"
        
        if cache_key not in self.cache:
            if model_name == 'resnet50':
                self.cache[cache_key] = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, **kwargs)
            elif model_name == 'vgg16':
                self.cache[cache_key] = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1, **kwargs)
            elif model_name == 'efficientnet_b0':
                self.cache[cache_key] = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1, **kwargs)
            # Add more models as needed
        
        return self.cache[cache_key]

# Training Functions
def train_transfer_model(model, train_loader, val_loader, num_epochs=10):
    """Train a transfer learning model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_accuracy:.2f}%')
        
        scheduler.step()

# Data preprocessing for pretrained models
def get_transforms(model_type='imagenet'):
    """Get appropriate transforms for different model types"""
    if model_type == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == 'efficientnet':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    # Load and test pretrained models
    print("Loading pretrained models...")
    models_dict = load_pretrained_models()
    
    # Test model loading
    for name, model in list(models_dict.items())[:3]:  # Test first 3 models
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"{name} output shape: {output.shape}")
    
    # Test transfer learning approaches
    print("\nTesting transfer learning approaches...")
    
    # Feature extractor
    feature_extractor = FeatureExtractor('resnet50', num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = feature_extractor(x)
    print(f"Feature extractor output shape: {output.shape}")
    
    # Fine tuner
    fine_tuner = FineTuner('resnet50', num_classes=10)
    output = fine_tuner(x)
    print(f"Fine tuner output shape: {output.shape}")
    
    # Ensemble
    ensemble = MultiModelEnsemble(num_classes=10)
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")
    
    # Layer-wise fine tuning
    layer_wise = LayerWiseFineTuning('resnet50', num_classes=10)
    print(f"Layer groups: {layer_wise.layer_groups}")
    print(f"Unfroze next group: {layer_wise.unfreeze_next_layer_group()}")
    
    # Model cache
    cache = ModelCache()
    cached_model = cache.get_model('resnet50')
    print(f"Cached model type: {type(cached_model)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in fine_tuner.parameters())
    trainable_params = sum(p.numel() for p in fine_tuner.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nPretrained models usage examples completed!")