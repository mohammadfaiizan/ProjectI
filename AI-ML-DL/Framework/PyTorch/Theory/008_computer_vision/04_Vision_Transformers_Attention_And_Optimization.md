# Vision Transformers, Attention, and Optimization

## Table of Contents

- [Vision Transformer](#vision-transformer)
- [Attention Mechanisms in Vision](#attention-mechanisms-in-vision)
- [Vision Data Augmentation](#vision-data-augmentation)
- [Building Custom Vision Architectures](#building-custom-vision-architectures)
- [Model Optimization](#model-optimization)

---

## Vision Transformer

The **Vision Transformer (ViT)** applies the Transformer architecture to images by splitting them into **patches**, embedding each patch, and processing with a standard Transformer encoder. A **classification token** is prepended for the final prediction.

### Patch Embedding

Images are divided into non-overlapping patches. Each patch is flattened and projected to the **embedding dimension** via a linear layer or convolution.

```python
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
```

### Position Encoding

**Position embeddings** are added to patch embeddings so the model knows spatial layout. Can be learned or sinusoidal.

```python
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
nn.init.trunc_normal_(self.pos_embed, std=0.02)
```

### Classification Token

A learnable **cls token** is prepended to the sequence. The final hidden state of this token is used for classification.

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
x = torch.cat([cls_tokens, x], dim=1)
```

### Transformer Encoder for Images

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Complete ViT

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits
```

---

## Attention Mechanisms in Vision

Attention in CNNs can be **channel attention** (what to emphasize), **spatial attention** (where to look), or **self-attention** (long-range dependencies).

### Self-Attention

**Self-attention** computes query, key, value from the same input and applies scaled dot-product attention.

```python
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.inter_channels = in_channels // reduction_ratio
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.output_conv = nn.Conv2d(self.inter_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)
        value = self.value_conv(x).view(batch_size, self.inter_channels, -1)
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.inter_channels, height, width)
        out = self.output_conv(out)
        return self.gamma * out + x
```

### Squeeze-and-Excitation (SE)

**SE blocks** perform channel-wise recalibration. Squeeze (global pooling) -> Excitation (FC layers) -> Scale.

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

### CBAM (Convolutional Block Attention Module)

**CBAM** combines **channel attention** and **spatial attention** in sequence.

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(combined))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)
```

### Spatial vs Channel Attention

| Type | Focus | Typical Use |
|------|-------|-------------|
| Channel | Which features matter | SE, CBAM channel branch |
| Spatial | Where to look | CBAM spatial branch |
| Self-attention | Long-range dependencies | Non-local, ViT |

---

## Vision Data Augmentation

Data augmentation increases diversity and improves generalization. Key transforms: **RandomResizedCrop**, **Mixup**, **CutMix**, **RandAugment**, **AutoAugment**.

### Basic Transforms

```python
import torchvision.transforms as transforms

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

advanced_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Mixup

**Mixup** interpolates between two images and their labels.

```python
import numpy as np

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

### CutMix

**CutMix** cuts and pastes patches between images. The label is mixed by the area ratio.

```python
import numpy as np

def cutmix(x, y, beta=1.0):
    lam = np.random.beta(beta, beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    W, H = x.size(-1), x.size(-2)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[index], lam
```

### RandAugment and AutoAugment

**RandAugment** uses a fixed set of operations with random magnitudes. **AutoAugment** uses learned policies from reinforcement learning.

### Test-Time Augmentation (TTA)

Apply multiple augmentations at test time and average predictions for robustness.

```python
def tta_predict(model, image, transforms_list):
    predictions = []
    for transform in transforms_list:
        aug_img = transform(image)
        pred = model(aug_img)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

---

## Building Custom Vision Architectures

Custom architectures combine **residual blocks**, **FPN**, **multi-scale features**, and attention.

### Residual Blocks

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        hidden = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))
```

### Feature Pyramid Network (FPN)

**FPN** builds a pyramid of features with top-down and lateral connections.

```python
class FPNBlock(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [lc(f) for lc, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode='nearest')
        return [oc(l) for oc, l in zip(self.fpn_convs, laterals)]
```

### Multi-Scale Features

Extract features at multiple resolutions (e.g., from different stages of a backbone) for detection or segmentation heads.

---

## Model Optimization

Optimization techniques reduce model size and inference cost: **quantization**, **pruning**, **knowledge distillation**, **TorchScript**, **ONNX export**.

### Quantization

**Quantization** reduces precision (e.g., FP32 to INT8) for smaller models and faster inference.

```python
import torch.quantization as quant

model.qconfig = quant.get_default_qconfig('fbgemm')
prepared = quant.prepare(model, inplace=False)
for data, _ in calibration_loader:
    prepared(data)
quantized_model = quant.convert(prepared, inplace=False)

dynamic_quantized = quant.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)
```

### Pruning

**Pruning** removes weights (unstructured) or channels (structured) to create sparse models.

```python
from torch.nn.utils import prune

prune.l1_unstructured(module, name='weight', amount=0.3)
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)
prune.remove(module, 'weight')
```

### Knowledge Distillation

**Knowledge distillation** trains a small **student** to mimic a large **teacher**. Soft targets from the teacher provide richer supervision.

```python
def distillation_loss(student_logits, teacher_logits, targets, temperature=4.0, alpha=0.7):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, targets)
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

### TorchScript and ONNX Export

**TorchScript** compiles models for deployment without Python. **ONNX** enables cross-framework deployment.

```python
scripted = torch.jit.script(model)
scripted.save('model.pt')

torch.onnx.export(model, dummy_input, 'model.onnx', input_names=['input'], output_names=['output'])
```

### Optimization Summary

| Technique | Size Reduction | Speed Gain | Accuracy Impact |
|-----------|----------------|------------|-----------------|
| Quantization | 2-4x | 2-4x | Minimal |
| Pruning | Variable | Variable | Depends on sparsity |
| Distillation | 2-10x | 2-5x | Small |
| Mixed precision | 2x (memory) | 1.5-2x | None |
