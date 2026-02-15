# Detection, Segmentation, and Recognition

## Table of Contents

- [Object Detection Fundamentals](#object-detection-fundamentals)
- [Single-Shot vs Two-Stage Detectors](#single-shot-vs-two-stage-detectors)
- [Building Detection Heads in PyTorch](#building-detection-heads-in-pytorch)
- [Semantic Segmentation](#semantic-segmentation)
- [Instance Segmentation](#instance-segmentation)

---

## Object Detection Fundamentals

Object detection localizes objects with **bounding boxes** and assigns **class labels**. Core concepts include **anchor boxes**, **IoU (Intersection over Union)**, **NMS (Non-Maximum Suppression)**, and **region proposals**.

### Bounding Box Formats and IoU

Bounding boxes use either **xyxy** (x1, y1, x2, y2) or **xywh** (x, y, width, height). IoU measures overlap between predicted and ground-truth boxes.

```python
import torch
from torchvision.ops import box_iou, nms

def calculate_iou(boxes1, boxes2):
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    x_left = torch.max(x1_1.unsqueeze(1), x1_2.unsqueeze(0))
    y_top = torch.max(y1_1.unsqueeze(1), y1_2.unsqueeze(0))
    x_right = torch.min(x2_1.unsqueeze(1), x2_2.unsqueeze(0))
    y_bottom = torch.min(y2_1.unsqueeze(1), y2_2.unsqueeze(0))
    intersection = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    return intersection / (union + 1e-6)
```

### Anchor Generation

Anchors are predefined boxes at each spatial location. Generated from **sizes** and **aspect ratios** and tiled across the feature map.

```python
import math

class AnchorGenerator:
    def __init__(self, sizes=[32, 64, 128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0], stride=16):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.stride = stride

    def generate_anchors_for_feature_map(self, feature_height, feature_width, size):
        base_anchors = []
        for ratio in self.aspect_ratios:
            area = size ** 2
            width = math.sqrt(area / ratio)
            height = width * ratio
            base_anchors.append([-width/2, -height/2, width/2, height/2])
        base_anchors = torch.tensor(base_anchors)
        shifts_x = torch.arange(feature_width) * self.stride
        shifts_y = torch.arange(feature_height) * self.stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2).reshape(-1, 4)
        anchors = base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        return anchors.view(-1, 4)
```

### Non-Maximum Suppression

NMS removes overlapping detections, keeping the highest-scoring box in each cluster.

```python
keep_indices = nms(boxes, scores, iou_threshold=0.5)
filtered_boxes = boxes[keep_indices]
filtered_scores = scores[keep_indices]
```

---

## Single-Shot vs Two-Stage Detectors

| Type | Examples | Speed | Accuracy |
|------|----------|-------|----------|
| Two-stage | R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN | Slower | Higher |
| Single-shot | YOLO, SSD, RetinaNet | Faster | Good |

**Two-stage** detectors first generate region proposals (RPN), then classify and refine. **Single-shot** detectors predict boxes and classes in one pass from multiple feature levels.

---

## Building Detection Heads in PyTorch

### Multi-Scale Backbone

A detection backbone outputs features at multiple resolutions for detecting objects at different scales.

```python
class SimpleDetectionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x2, x3, x4]
```

### Detection Head

A detection head predicts **classification**, **bounding box regression**, and **objectness** per anchor.

```python
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 3, padding=1)
        self.obj_head = nn.Conv2d(256, num_anchors, 3, padding=1)

    def forward(self, x):
        shared = self.shared_conv(x)
        batch_size, _, height, width = shared.shape
        cls_pred = self.cls_head(shared).view(batch_size, self.num_anchors, self.num_classes, height, width)
        reg_pred = self.reg_head(shared).view(batch_size, self.num_anchors, 4, height, width)
        obj_pred = self.obj_head(shared)
        return cls_pred, reg_pred, obj_pred
```

### Detection Loss

Combines **classification loss** (e.g., Focal Loss), **regression loss** (Smooth L1), and **objectness loss**.

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def smooth_l1_loss(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    diff = pred[mask] - target[mask]
    return torch.where(torch.abs(diff) < 1.0, 0.5 * diff ** 2, torch.abs(diff) - 0.5).mean()
```

---

## Semantic Segmentation

Semantic segmentation assigns a **class label to each pixel**. No instance separation. Key architectures: **FCN**, **U-Net**, **DeepLab**.

### Encoder-Decoder and Pixel-Wise Classification

The **encoder** downsamples to extract features. The **decoder** upsamples to restore spatial resolution. **Skip connections** preserve fine details.

### U-Net Building Blocks

```python
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
```

### U-Net Architecture

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(DoubleConv(in_channels, features[0]))
        for i in range(len(features) - 1):
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(features[i], features[i + 1])
            ))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(UpBlock(features[i] * 2, features[i - 1]))
        self.final_conv = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder):
            x = layer(x, skip_connections[i])
        return self.final_conv(x)
```

### FCN Concepts

FCN replaces fully connected layers with **1x1 convolutions** and uses **transposed convolutions** for upsampling. FCN-8s adds skip connections from pool3 and pool4 for finer boundaries.

### Segmentation Loss Functions

```python
def dice_loss(pred, target, num_classes, smooth=1.0):
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

combined_loss = F.cross_entropy(pred, target) + dice_loss(pred, target, num_classes)
```

---

## Instance Segmentation

Instance segmentation combines **detection** (bounding boxes + classes) with **pixel-level masks** per instance. **Mask R-CNN** is the canonical approach.

### Mask R-CNN Concepts

Mask R-CNN extends Faster R-CNN with a **mask head** that predicts a binary mask for each detected object. Uses **ROI Align** (not ROI Pool) for sub-pixel accurate feature extraction.

### Region Proposal Network (RPN)

```python
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, features):
        logits, bbox_reg = [], []
        for f in features:
            x = F.relu(self.conv(f))
            logits.append(self.cls_logits(x))
            bbox_reg.append(self.bbox_pred(x))
        return logits, bbox_reg
```

### ROI Align

```python
from torchvision.ops import roi_align

def roi_align_forward(features, boxes, output_size=(7, 7), spatial_scale=1/16.0):
    return roi_align(features, boxes, output_size, spatial_scale=spatial_scale, sampling_ratio=2)
```

### Mask Head

```python
class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes, dim_reduced=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, dim_reduced, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, stride=2)
        self.predictor = nn.Conv2d(dim_reduced, num_classes, 1)

    def forward(self, roi_features):
        x = self.conv_layers(roi_features)
        x = F.relu(self.deconv(x))
        return self.predictor(x)
```

### Feature Pyramid Network (FPN)

FPN builds a **multi-scale feature pyramid** from a backbone. Top-down pathway with lateral connections.

```python
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        laterals = [lc(f) for lc, f in zip(self.lateral_convs, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i + 1], size=laterals[i].shape[2:], mode='nearest')
        return [oc(l) for oc, l in zip(self.output_convs, laterals)]
```

### Combining Detection and Segmentation

The full pipeline: backbone -> FPN -> RPN (proposals) -> ROI Align -> classification + bbox regression + mask prediction. Post-process with NMS and mask thresholding.
