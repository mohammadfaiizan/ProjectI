# Object Detection: R-CNN to YOLO

## Table of Contents

1. [Introduction](#introduction)
2. [R-CNN Architecture](#r-cnn-architecture)
3. [Fast R-CNN](#fast-r-cnn)
4. [Faster R-CNN](#faster-r-cnn)
5. [Single Shot Detector (SSD)](#single-shot-detector-ssd)
6. [YOLO Versions](#yolo-versions)
7. [Anchor-Based vs Anchor-Free](#anchor-based-vs-anchor-free)
8. [Feature Pyramid Networks](#feature-pyramid-networks)
9. [Detection Head Design](#detection-head-design)
10. [Key Takeaways](#key-takeaways)

## Introduction

Object detection localizes and classifies objects in images, requiring both spatial localization (bounding boxes) and semantic understanding (class labels). The evolution from R-CNN to modern YOLO represents a progression from multi-stage to single-stage detectors, balancing accuracy and speed.

R-CNN introduced deep learning to detection, Fast R-CNN improved efficiency, Faster R-CNN unified the pipeline, and YOLO revolutionized real-time detection. Understanding these architectures reveals fundamental design choices: region proposals vs dense prediction, two-stage vs single-stage, and anchor-based vs anchor-free approaches.

## R-CNN Architecture

R-CNN (Regions with CNN features) was the first to apply CNNs effectively to object detection.

### R-CNN Pipeline

1. **Region proposals**: Generate ~2000 candidate regions using selective search
2. **Feature extraction**: Resize each region to 227×227, extract CNN features
3. **Classification**: Classify each region using SVM
4. **Bounding box regression**: Refine bounding boxes using linear regression

### Selective Search

Selective search generates region proposals by:
- **Hierarchical segmentation**: Start with superpixels, merge similar regions
- **Diversity**: Use multiple color spaces and similarity measures
- **Efficiency**: Generate ~2000 proposals per image

Similarity measures:
- Color similarity
- Texture similarity
- Size similarity
- Shape compatibility

### Feature Extraction

For each region proposal:
1. **Warp**: Resize to fixed size (227×227)
2. **Forward pass**: Extract features using AlexNet (4096-dim)
3. **Classification**: Use class-specific SVMs
4. **Regression**: Refine boxes using class-specific regressors

### Training

**Pre-training**: Train CNN on ImageNet classification
**Fine-tuning**: Fine-tune on detection dataset with positive/negative regions
**SVM training**: Train binary SVMs per class on CNN features
**Bbox regression**: Train regressors to refine boxes

### Limitations

- **Slow**: Forward pass for each region (~2000 per image)
- **Memory**: Store features for all regions
- **Multi-stage**: Separate training for CNN, SVM, regressor
- **Fixed size**: Warping distorts aspect ratios

## Fast R-CNN

Fast R-CNN addresses R-CNN inefficiencies by sharing computation and end-to-end training.

### Architecture

1. **Feature extraction**: Single forward pass through CNN for entire image
2. **Region of Interest (RoI) pooling**: Extract fixed-size features for each proposal
3. **Classification and regression**: Shared fully connected layers

### RoI Pooling

RoI pooling converts variable-size regions to fixed-size features:

1. **Input**: Feature map $F$ and RoI $(r, c, h, w)$
2. **Divide**: Divide RoI into $H \times W$ grid (e.g., 7×7)
3. **Max pool**: Max pool each grid cell
4. **Output**: Fixed-size $H \times W$ feature map

Mathematically:
$$y_{i,j} = \max_{u \in [ih/H, (i+1)h/H], v \in [jw/W, (j+1)w/W]} F_{r+u, c+v}$$

### Multi-task Loss

Fast R-CNN uses joint training:

$$L = L_{cls}(p, u) + \lambda [u \geq 1] L_{loc}(t^u, v)$$

where:
- $L_{cls}$: Classification loss (log loss)
- $L_{loc}$: Localization loss (smooth L1)
- $u$: True class label
- $p$: Predicted probabilities
- $t^u$: Predicted box parameters
- $v$: True box parameters

### Advantages

- **Faster**: Single forward pass instead of 2000
- **End-to-end**: Joint training of all components
- **Better accuracy**: Multi-task learning improves both tasks

### Limitations

- **Still slow**: Region proposal generation is bottleneck
- **Selective search**: Not learnable, slow

## Faster R-CNN

Faster R-CNN makes region proposal generation learnable and fast through Region Proposal Network (RPN).

### Architecture

1. **Backbone**: CNN feature extractor (e.g., ResNet)
2. **RPN**: Generates region proposals
3. **RoI Head**: Classification and bounding box regression

### Region Proposal Network (RPN)

RPN slides a small network over feature map:

**Anchor generation**: At each location, generate $k$ anchors (typically $k=9$: 3 scales × 3 aspect ratios)

**RPN head**: For each anchor:
- **Objectness score**: Binary classifier (object vs background)
- **Bbox regression**: Refine anchor to object box

### Anchors

Anchors are pre-defined boxes at each location:
- **Scales**: $\{128^2, 256^2, 512^2\}$ pixels
- **Aspect ratios**: $\{1:1, 1:2, 2:1\}$
- **Total**: 9 anchors per location

Anchor coordinates:
$$x_a = x_c + s \cdot w_a \cdot \Delta x_a$$
$$y_a = y_c + s \cdot h_a \cdot \Delta y_a$$
$$w_a = w_a \cdot \exp(\Delta w_a)$$
$$h_a = h_a \cdot \exp(\Delta h_a)$$

where $(x_c, y_c)$ is center, $s$ is stride, and $\Delta$ are predicted offsets.

### RPN Training

**Positive anchors**: IoU > 0.7 with ground truth, or highest IoU per GT
**Negative anchors**: IoU < 0.3 with all ground truth
**Loss function**:
$$L_{RPN} = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* L_{reg}(t_i, t_i^*)$$

where $p_i^*$ indicates positive anchor.

### Training Strategy

**Alternating training**:
1. Train RPN
2. Train detection head with RPN proposals
3. Fine-tune RPN with detection head
4. Fine-tune detection head

**End-to-end training**: Joint optimization with shared features (preferred).

### Performance

- **Speed**: ~5 FPS (vs ~0.02 FPS for R-CNN)
- **Accuracy**: State-of-the-art on PASCAL VOC and COCO
- **Unified**: Single network for proposals and detection

## Single Shot Detector (SSD)

SSD is a single-stage detector that predicts objects directly from feature maps without region proposals.

### Architecture

1. **Backbone**: Base network (VGG, ResNet) for feature extraction
2. **Multi-scale feature maps**: Features at multiple resolutions
3. **Detection heads**: Predict boxes and classes at each scale
4. **Non-maximum suppression**: Remove duplicate detections

### Multi-Scale Detection

SSD uses features at multiple scales:
- **High resolution**: Detect small objects
- **Low resolution**: Detect large objects

Feature maps: 38×38, 19×19, 10×10, 5×5, 3×3, 1×1

### Default Boxes

Similar to anchors, but called default boxes:
- **Scales**: $s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1)$
- **Aspect ratios**: $\{1, 2, 1/2, 3, 1/3\}$ (some scales use fewer)
- **Center**: $(i+0.5)/|f_k|, (j+0.5)/|f_k|$ for feature map $f_k$

### Prediction

For each default box, predict:
- **Class scores**: $(C+1)$ scores (C classes + background)
- **Location offsets**: 4 values $(dx, dy, dw, dh)$

Total predictions: $k \times (C+1+4)$ per location.

### Loss Function

$$L = \frac{1}{N}(L_{conf} + \alpha L_{loc})$$

**Confidence loss**: Softmax over classes
$$L_{conf} = -\sum_{i \in Pos} \log(\hat{c}_i^{p_i}) - \sum_{i \in Neg} \log(\hat{c}_i^0)$$

**Localization loss**: Smooth L1
$$L_{loc} = \sum_{i \in Pos} \sum_{m \in \{x,y,w,h\}} \text{smooth}_{L1}(l_i^m - \hat{g}_i^m)$$

### Hard Negative Mining

SSD uses hard negative mining:
1. Sort negatives by confidence loss
2. Keep top $k$ (typically 3× number of positives)
3. Discard rest

Balances positive/negative ratio during training.

### Advantages

- **Fast**: Single forward pass, no region proposals
- **Multi-scale**: Handles objects of various sizes
- **End-to-end**: Fully differentiable

### Limitations

- **Small objects**: Challenging for very small objects
- **Aspect ratios**: Fixed aspect ratios may not match all objects

## YOLO Versions

YOLO (You Only Look Once) revolutionized real-time object detection with its single-stage approach.

### YOLO v1

**Architecture**:
- Single CNN processes entire image
- Divides image into $S \times S$ grid (e.g., 7×7)
- Each cell predicts $B$ bounding boxes and class probabilities

**Prediction**: For each grid cell:
- $B$ boxes: $(x, y, w, h, \text{confidence})$
- Class probabilities: $P(\text{class}_i | \text{object})$

**Final detections**: Combine boxes and classes:
$$P(\text{class}_i) = P(\text{class}_i | \text{object}) \times P(\text{object}) \times \text{IoU}$$

**Loss function**:
$$L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]$$
$$+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_i^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2$$

### YOLO v2 (YOLO9000)

**Improvements**:
- **Batch normalization**: Added to all conv layers
- **High resolution**: Fine-tune on 448×448
- **Anchor boxes**: Use anchor boxes instead of predicting coordinates directly
- **Dimension clusters**: K-means on training boxes to find good anchor sizes
- **Fine-grained features**: Passthrough layer from earlier feature map
- **Multi-scale training**: Train on multiple input sizes

**Architecture**: Darknet-19 backbone

### YOLO v3

**Key features**:
- **Darknet-53**: Deeper backbone (53 layers)
- **Multi-scale**: Predictions at 3 scales (like FPN)
- **Better backbone**: Residual connections
- **Logistic regression**: For objectness (instead of softmax)

**Architecture**:
- Backbone: Darknet-53
- Neck: Feature pyramid network
- Head: Detection at 3 scales

### YOLO v4

**Improvements**:
- **CSPDarknet53**: Cross-stage partial connections
- **PANet**: Path aggregation network
- **SPP**: Spatial pyramid pooling
- **SAM**: Spatial attention module
- **Mish activation**: Better than ReLU
- **CIoU loss**: Complete IoU loss

### YOLO v5-v8

**YOLO v5**: PyTorch implementation, focus on ease of use
**YOLO v6**: Reparameterization, efficient design
**YOLO v7**: Extended efficient layer aggregation, model scaling
**YOLO v8**: Latest version with improved architecture and training

## Anchor-Based vs Anchor-Free

Modern detectors can be categorized as anchor-based or anchor-free.

### Anchor-Based Detectors

**Examples**: Faster R-CNN, SSD, YOLO v2-v5

**Advantages**:
- Good performance with proper anchor design
- Handles aspect ratios well
- Established and well-understood

**Disadvantages**:
- Anchor design requires heuristics
- Many anchors (computational cost)
- Hyperparameter sensitive

### Anchor-Free Detectors

**Examples**: CornerNet, CenterNet, FCOS, YOLO v1

**Approaches**:
- **Corner-based**: Detect object corners/keypoints
- **Center-based**: Detect object centers
- **Dense prediction**: Predict at every location

**Advantages**:
- Simpler design (no anchors)
- Fewer hyperparameters
- Can be faster

**Disadvantages**:
- May struggle with extreme aspect ratios
- Requires post-processing for grouping

### FCOS (Fully Convolutional One-Stage)

FCOS predicts:
- **Centerness**: How close to object center
- **Classification**: Class scores
- **Regression**: Distances to box boundaries

For location $(x, y)$, predict distances $(l, t, r, b)$ to left, top, right, bottom edges.

## Feature Pyramid Networks

Feature Pyramid Networks (FPN) address multi-scale detection by combining features at different resolutions.

### FPN Architecture

1. **Bottom-up pathway**: Standard CNN (e.g., ResNet)
2. **Top-down pathway**: Upsample high-level features
3. **Lateral connections**: Combine bottom-up and top-down features

### Feature Fusion

For level $l$:
$$P_l = \text{Upsample}(P_{l+1}) + C_l$$

where:
- $C_l$: Bottom-up feature at level $l$
- $P_{l+1}$: Top-down feature at level $l+1$
- Upsample: Nearest neighbor or bilinear

After fusion, apply 3×3 conv to reduce aliasing.

### Detection with FPN

Assign objects to pyramid levels:
$$l = \lfloor l_0 + \log_2(\sqrt{wh}/224) \rfloor$$

where $l_0$ is base level (e.g., $P_4$), $w \times h$ is object size.

### Benefits

- **Multi-scale**: Single detector handles all scales
- **Rich features**: High-level semantics + low-level details
- **Efficiency**: Shared computation across scales

## Detection Head Design

Detection heads convert features to bounding boxes and classes.

### Two-Stage Heads

**RPN head**: Objectness + bbox regression
**RoI head**: Classification + bbox regression

Shared backbone, separate heads for proposals and detection.

### Single-Stage Heads

**Classification branch**: Predict class scores
**Regression branch**: Predict box coordinates
**Objectness branch**: Predict object confidence (optional)

### Head Variants

**RetinaNet head**: Two parallel branches (classification, regression)
**FCOS head**: Classification + regression + centerness
**YOLO head**: Combined predictions (classes + boxes)

### Loss Functions

**Classification**: Focal loss, cross-entropy
**Localization**: Smooth L1, IoU loss, GIoU, DIoU, CIoU
**Objectness**: Binary cross-entropy

## Key Takeaways

1. R-CNN introduced deep learning to detection but was slow due to per-region forward passes and multi-stage training.

2. Fast R-CNN improved efficiency through shared feature extraction and RoI pooling, enabling end-to-end training.

3. Faster R-CNN unified the pipeline with learnable region proposals via RPN, achieving state-of-the-art accuracy.

4. SSD is a single-stage detector using multi-scale feature maps and default boxes, balancing speed and accuracy.

5. YOLO revolutionized real-time detection with its single-pass approach, evolving through multiple versions with improved architectures and training strategies.

6. Anchor-based detectors use pre-defined anchor boxes, while anchor-free detectors predict directly, each with trade-offs in design complexity and performance.

7. Feature Pyramid Networks enable multi-scale detection by combining bottom-up and top-down pathways, improving detection of objects at various sizes.

8. Detection head design varies between two-stage (separate proposal and detection) and single-stage (unified prediction) approaches.

9. Modern detectors balance accuracy and speed through architectural innovations: better backbones, multi-scale features, efficient heads, and improved loss functions.

10. Understanding the evolution from R-CNN to modern YOLO reveals fundamental design principles: region proposals vs dense prediction, two-stage vs single-stage, and anchor-based vs anchor-free approaches.
