# Image Classification and Transfer Learning

## Table of Contents

- [CNN Architectures for Classification](#cnn-architectures-for-classification)
- [Building Classification Pipelines in PyTorch](#building-classification-pipelines-in-pytorch)
- [Transfer Learning](#transfer-learning)
- [Using torchvision.models](#using-torchvisionmodels)
- [Loading Pretrained Weights and Modifying Final Layers](#loading-pretrained-weights-and-modifying-final-layers)

---

## CNN Architectures for Classification

Convolutional Neural Networks form the backbone of modern image classification. Key architectural concepts include **LeNet** (early digit recognition), **AlexNet** (deep networks with ReLU and dropout), **VGG** (uniform 3x3 convolutions), **ResNet** (residual connections enabling very deep networks), and **DenseNet** (dense connectivity for feature reuse).

### LeNet and AlexNet Concepts

LeNet introduced the basic pattern: conv layers, pooling, and fully connected classifier. AlexNet scaled this with multiple GPUs, ReLU activations, and dropout. The core building block is a **ConvBlock** with convolution, batch normalization, and activation.

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))
```

### VGG and ResNet

VGG uses repeated 3x3 convolutions to build depth. ResNet introduces **residual connections** that allow gradients to flow directly, enabling training of networks with 50+ layers.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + residual)
```

### DenseNet and Inception Concepts

DenseNet connects each layer to every other layer in a feed-forward fashion. Inception uses **parallel branches** with different kernel sizes (1x1, 3x3, 5x5) and concatenates outputs.

```python
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool):
        super().__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1, padding=0)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_3x3_reduce, kernel_size=1, padding=0),
            ConvBlock(out_3x3_reduce, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, out_5x5_reduce, kernel_size=1, padding=0),
            ConvBlock(out_5x5_reduce, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, out_pool, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)
```

### Depthwise Separable Convolution

Used in **MobileNet** and **EfficientNet** for parameter efficiency. Separates spatial and channel-wise operations.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

---

## Building Classification Pipelines in PyTorch

A complete classification pipeline includes **feature extraction**, **global pooling**, **classifier head**, and **training infrastructure**.

### Simple CNN Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### Data Loading and Preprocessing

```python
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
```

### Training Loop

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    return total_loss / len(dataloader), 100. * correct / total
```

---

## Transfer Learning

Transfer learning leverages **pretrained models** on large datasets (e.g., ImageNet) and adapts them to new tasks. Two main strategies: **feature extraction** (freeze backbone, train classifier) and **fine-tuning** (unfreeze some or all layers).

### Feature Extraction vs Fine-Tuning

| Strategy | Backbone | Classifier | Use Case |
|----------|----------|------------|----------|
| Feature extraction | Frozen | Trainable | Small dataset, similar domain |
| Fine-tuning | Partially trainable | Trainable | Medium dataset, domain shift |
| Full fine-tuning | Trainable | Trainable | Large dataset, different domain |

### Freezing Layers

```python
def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def freeze_early_layers(model, num_layers=10):
    for i, param in enumerate(model.features.parameters()):
        if i < num_layers:
            param.requires_grad = False
```

### Learning Rate Strategies

Use **lower learning rates** for pretrained layers and **higher rates** for new layers. Layer-wise learning rates prevent catastrophic forgetting.

```python
def get_layer_wise_params(model, base_lr=1e-3):
    params = []
    early_params = [p for n, p in model.named_parameters() if 'layer1' in n or 'conv1' in n]
    later_params = [p for n, p in model.named_parameters() if 'layer4' in n or 'fc' in n]
    if early_params:
        params.append({'params': early_params, 'lr': base_lr * 0.1})
    if later_params:
        params.append({'params': later_params, 'lr': base_lr})
    return params
```

### Progressive Unfreezing

Unfreeze layers gradually during training to stabilize fine-tuning.

```python
class ProgressiveUnfreezer:
    def __init__(self, model, unfreeze_schedule):
        self.model = model
        self.schedule = unfreeze_schedule

    def update_epoch(self, epoch):
        if epoch in self.schedule:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in self.schedule[epoch]):
                    param.requires_grad = True
```

---

## Using torchvision.models

The **torchvision.models** module provides pretrained architectures. Use `torchvision.models.get_model` or direct constructors with `weights` parameter.

### Available Model Families

| Family | Models | Typical Use |
|--------|--------|-------------|
| ResNet | resnet18, resnet34, resnet50, resnet101, resnet152 | General purpose, good baseline |
| VGG | vgg11, vgg16, vgg19 | Feature extraction, style transfer |
| DenseNet | densenet121, densenet169, densenet201 | Dense features |
| EfficientNet | efficientnet_b0 through b7 | Accuracy-efficiency tradeoff |
| MobileNet | mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large | Mobile, edge deployment |
| ConvNeXt | convnext_tiny, convnext_base | Modern CNN |
| ViT | vit_b_16, vit_l_16 | Vision Transformer |

### Loading Models

```python
import torchvision.models as models

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
vit_b_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
```

### torchvision.models.get_model

```python
model = models.get_model('resnet50', weights=models.ResNet50_Weights.IMAGENET1K_V2)
```

---

## Loading Pretrained Weights and Modifying Final Layers

### Feature Extractor Pattern

Replace the final classification layer with `nn.Identity()` and add a custom classifier. Freeze the backbone.

```python
class FeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet18', num_classes=10):
        super().__init__()
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
            self.feature_size = 512
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()
            self.feature_size = 1280
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features.view(features.size(0), -1))
```

### Fine-Tuner Pattern

Replace only the final layer and optionally freeze early layers.

```python
class FineTuner(nn.Module):
    def __init__(self, backbone_name='resnet18', num_classes=10, freeze_early=True):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if freeze_early:
            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
```

### Different Architectures, Different Replacement Points

| Model | Replacement Point | Feature Size |
|-------|-------------------|--------------|
| ResNet | model.fc | 512 (resnet18) or 2048 (resnet50) |
| VGG | model.classifier[6] | 4096 |
| DenseNet | model.classifier | model.classifier.in_features |
| EfficientNet | model.classifier[1] | 1280 |
| MobileNet | model.classifier[1] | 1280 (v2) |

### Data Transforms for Pretrained Models

ImageNet normalization is required for models trained on ImageNet.

```python
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
