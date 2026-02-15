# Detection and Segmentation

## Table of Contents

1. [Object Detection Fundamentals](#1-object-detection-fundamentals)
2. [Building Detection Heads in TensorFlow](#2-building-detection-heads-in-tensorflow)
3. [Semantic Segmentation](#3-semantic-segmentation)
4. [Instance Segmentation](#4-instance-segmentation)
5. [Combining Detection and Segmentation](#5-combining-detection-and-segmentation)

---

## 1. Object Detection Fundamentals

Object detection localizes and classifies multiple objects in an image. Unlike classification, it outputs **bounding boxes** and **class labels** for each object.

### Anchor Boxes

**Anchor boxes** are predefined reference boxes of various scales and aspect ratios. The network predicts offsets from these anchors to produce final detections.

```python
# Example: generate anchor boxes at different scales
def generate_anchors(scales=[32, 64, 128], aspect_ratios=[0.5, 1.0, 2.0]):
    anchors = []
    for scale in scales:
        for ar in aspect_ratios:
            w = scale * (ar ** 0.5)
            h = scale / (ar ** 0.5)
            anchors.append([-w/2, -h/2, w/2, h/2])  # [x1, y1, x2, y2]
    return tf.constant(anchors, dtype=tf.float32)
```

### Intersection over Union (IoU)

**IoU** measures overlap between predicted and ground-truth boxes. Used for matching predictions to targets and evaluating quality.

```python
def compute_iou(boxes1, boxes2):
    """boxes: [N, 4] format (x1, y1, x2, y2)"""
    x1 = tf.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = tf.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = tf.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = tf.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_w = tf.maximum(0.0, x2 - x1)
    inter_h = tf.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    return inter_area / (union_area + 1e-6)
```

### Non-Maximum Suppression (NMS)

**NMS** removes redundant overlapping detections. For each class, keep the highest-scoring box and suppress others with IoU above a threshold.

```python
def nms(boxes, scores, max_output_size=100, iou_threshold=0.5):
    """boxes: [N, 4], scores: [N]"""
    indices = tf.image.non_max_suppression(
        boxes, scores, max_output_size, iou_threshold
    )
    return indices
```

### Detection Pipeline Summary

| Step | Purpose |
|------|---------|
| Generate anchors | Reference boxes at each location |
| Predict offsets + classes | Refine anchors, assign labels |
| Match to ground truth | Assign positive/negative anchors |
| NMS | Remove duplicate detections |

---

## 2. Building Detection Heads in TensorFlow

A **detection head** takes feature maps from a backbone and outputs box coordinates and class scores.

### Single-Stage Detection Head

```python
from tensorflow.keras import layers, Model

def build_detection_head(feature_map, num_classes, num_anchors=9):
    """feature_map: [B, H, W, C] from backbone"""
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(feature_map)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    # Classification: num_anchors * num_classes per spatial location
    cls_output = layers.Conv2D(
        num_anchors * num_classes, 3, padding='same', activation='sigmoid'
    )(x)
    
    # Regression: 4 values (dx, dy, dw, dh) per anchor
    reg_output = layers.Conv2D(num_anchors * 4, 3, padding='same')(x)
    
    return cls_output, reg_output
```

### Multi-Scale Feature Pyramids

Detection benefits from features at multiple scales. Use a **Feature Pyramid Network (FPN)** or similar structure.

```python
def build_fpn_features(c2, c3, c4, c5, filters=256):
    """Build FPN from backbone stages c2-c5"""
    p5 = layers.Conv2D(filters, 1)(c5)
    p4 = layers.Conv2D(filters, 1)(c4) + layers.UpSampling2D()(p5)
    p3 = layers.Conv2D(filters, 1)(c3) + layers.UpSampling2D()(p4)
    p2 = layers.Conv2D(filters, 1)(c2) + layers.UpSampling2D()(p3)
    return p2, p3, p4, p5
```

### Post-Processing

```python
def decode_predictions(cls_logits, reg_outputs, anchors, score_threshold=0.5):
    """Decode raw outputs to [x1, y1, x2, y2, class_id, score]"""
    # Apply softmax/sigmoid to cls_logits
    # Decode reg_outputs to box coordinates using anchor offsets
    # Filter by score_threshold
    # Apply NMS per class
    pass
```

---

## 3. Semantic Segmentation

**Semantic segmentation** assigns a class label to each pixel. All pixels of the same object class share the same label (no instance distinction).

### U-Net Encoder-Decoder

The **U-Net** architecture uses an encoder to downsample and a decoder to upsample, with **skip connections** between encoder and decoder.

```python
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(256, 256, 3), num_classes=21):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck
    b = conv_block(p4, 1024)
    
    # Decoder with skip connections
    u4 = layers.UpSampling2D(2)(b)
    u4 = layers.concatenate([u4, c4])
    u4 = conv_block(u4, 512)
    
    u3 = layers.UpSampling2D(2)(u4)
    u3 = layers.concatenate([u3, c3])
    u3 = conv_block(u3, 256)
    
    u2 = layers.UpSampling2D(2)(u3)
    u2 = layers.concatenate([u2, c2])
    u2 = conv_block(u2, 128)
    
    u1 = layers.UpSampling2D(2)(u2)
    u1 = layers.concatenate([u1, c1])
    u1 = conv_block(u1, 64)
    
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(u1)
    return Model(inputs, outputs)
```

### Skip Connections

**Skip connections** concatenate or add encoder features to the decoder. They preserve spatial detail and improve gradient flow.

| Benefit | Description |
|---------|-------------|
| Spatial detail | Encoder has high-resolution features |
| Gradient flow | Shortcuts for backpropagation |
| Multi-scale | Combines coarse and fine information |

### Pixel-Wise Classification

The output is a tensor of shape `[B, H, W, num_classes]`. Each spatial position has a class probability distribution.

```python
# Loss: typically cross-entropy per pixel
loss = tf.keras.losses.SparseCategoricalCrossentropy()
# Or weighted for class imbalance
```

---

## 4. Instance Segmentation

**Instance segmentation** assigns a class and instance ID to each pixel. Each object instance is segmented separately.

### Mask R-CNN Concepts

**Mask R-CNN** extends Faster R-CNN with a mask head. Pipeline:

1. **Backbone**: Extract features (e.g., ResNet + FPN)
2. **Region Proposal Network (RPN)**: Propose regions
3. **RoI Align**: Extract fixed-size features per region
4. **Box head**: Refine box and classify
5. **Mask head**: Predict binary mask per class per region

### Mask Head Architecture

```python
def build_mask_head(roi_features, num_classes):
    """roi_features: [N, 14, 14, 256] from RoI align"""
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(roi_features)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(256, 2, strides=2, activation='relu')(x)
    # Per-class mask (or single mask for predicted class)
    masks = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    return masks
```

### RoI Align vs RoI Pooling

**RoI Align** uses bilinear interpolation and avoids quantization. It preserves spatial accuracy better than RoI Pooling.

---

## 5. Combining Detection and Segmentation

Many applications require both detection and segmentation.

### Two-Stage: Detect Then Segment

1. Run object detector to get boxes
2. Crop regions and run segmentation network
3. Combine results

```python
# Pseudocode
boxes, classes, scores = detector(image)
for box, cls in zip(boxes, classes):
    crop = crop_roi(image, box)
    mask = segmentation_model(crop)
    # Resize mask to original box size, place in full mask
```

### Single-Stage: Panoptic Segmentation

**Panoptic segmentation** unifies semantic and instance segmentation. Each pixel has a (category, instance_id) pair. Background uses semantic labels only.

### Architecture Patterns

| Pattern | Detection | Segmentation | Use Case |
|---------|-----------|---------------|----------|
| Separate models | Yes | Yes | Modular, independent training |
| Shared backbone | Yes | Yes | Efficient, joint features |
| Mask R-CNN | Yes | Yes | Unified, instance-level |

### Joint Training

When using a shared backbone, train detection and segmentation losses together:

```python
total_loss = (
    det_cls_loss + det_reg_loss +
    mask_loss
)
```
