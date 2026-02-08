# Instance Segmentation and Mask R-CNN

## Table of Contents

1. [Introduction](#introduction)
2. [Instance vs Semantic vs Panoptic Segmentation](#instance-vs-semantic-vs-panoptic-segmentation)
3. [Mask R-CNN Architecture](#mask-r-cnn-architecture)
4. [RoI Pooling and RoI Align](#roi-pooling-and-roi-align)
5. [Mask Head Design](#mask-head-design)
6. [Panoptic Segmentation](#panoptic-segmentation)
7. [YOLACT and Real-Time Methods](#yolact-and-real-time-methods)
8. [Training and Loss Functions](#training-and-loss-functions)
9. [Applications and Extensions](#applications-and-extensions)
10. [Key Takeaways](#key-takeaways)

## Introduction

Instance segmentation combines object detection and semantic segmentation, identifying individual object instances and their pixel-level masks. Unlike semantic segmentation (which labels pixels by class) or object detection (which provides bounding boxes), instance segmentation provides both instance identity and precise boundaries.

Mask R-CNN extends Faster R-CNN with a mask prediction branch, enabling simultaneous detection and segmentation. Understanding instance segmentation reveals the relationship between detection and segmentation tasks and enables applications including autonomous driving, robotics, medical imaging, and augmented reality.

## Instance vs Semantic vs Panoptic Segmentation

Understanding the differences between segmentation types clarifies task requirements and method selection.

### Semantic Segmentation

**Goal**: Assign class label to each pixel
**Output**: Dense class map (no instance identity)
**Example**: All "person" pixels have same label

Challenges:
- Cannot distinguish between instances of same class
- No object-level understanding

### Instance Segmentation

**Goal**: Identify and segment each object instance
**Output**: Set of instance masks with class labels
**Example**: Each person has unique instance ID

Challenges:
- Must separate overlapping objects
- Requires detection + segmentation

### Panoptic Segmentation

**Goal**: Unified segmentation of both "things" (countable objects) and "stuff" (regions)
**Output**: Instance IDs for things, class labels for stuff
**Example**: Person instances + sky/street regions

Panoptic combines:
- Instance segmentation (for things)
- Semantic segmentation (for stuff)

### Comparison

| Task | Output | Counts Instances | Handles Stuff |
|------|--------|------------------|---------------|
| Semantic | Class per pixel | No | Yes |
| Instance | Instance masks | Yes | No |
| Panoptic | Unified map | Yes | Yes |

### Evaluation Metrics

**Semantic**: Pixel accuracy, mIoU
**Instance**: Average Precision (AP), mask IoU
**Panoptic**: Panoptic Quality (PQ)

## Mask R-CNN Architecture

Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch.

### Architecture Overview

Mask R-CNN consists of:
1. **Backbone**: Feature extraction (ResNet, ResNeXt)
2. **Neck**: Feature pyramid network (FPN)
3. **RPN**: Region proposal network
4. **RoI Head**: 
   - Classification branch
   - Bounding box regression branch
   - Mask prediction branch (new)

### Key Innovation

Mask R-CNN adds mask prediction as a parallel branch:
- **Classification**: Object class
- **Bbox regression**: Box refinement
- **Mask prediction**: Binary mask per class

This design enables:
- End-to-end training
- Shared features
- Efficient inference

### Architecture Details

**Backbone**: ResNet-50/101 with FPN
- Bottom-up: Standard ResNet
- Top-down: Upsampled features
- Lateral: 1×1 conv + addition

**RPN**: Same as Faster R-CNN
- Generates object proposals
- Shared features with detection

**RoI Head**: 
- RoI Align (replaces RoI Pooling)
- Classification: FC layers → class scores
- Bbox regression: FC layers → box offsets
- Mask: Small FCN → $K$ binary masks

### Mask Branch

The mask branch is a small FCN applied to each RoI:

1. **Input**: RoI feature (from RoI Align)
2. **Conv layers**: 4× 3×3 conv + ReLU
3. **Deconvolution**: 2× upsampling
4. **Output**: $K \times m \times m$ masks

where:
- $K$: Number of classes
- $m$: Mask resolution (typically 28×28)

Each mask is binary (foreground/background) for one class.

### Mask Prediction

For each RoI:
1. Predict class $c$ (from classification branch)
2. Select mask corresponding to class $c$
3. Threshold mask to get binary segmentation

This design:
- Predicts masks for all classes
- Uses only the mask for predicted class
- Enables class-agnostic mask learning

## RoI Pooling and RoI Align

RoI Align fixes quantization issues in RoI Pooling, critical for accurate mask prediction.

### RoI Pooling (Faster R-CNN)

RoI Pooling quantizes RoI coordinates:

1. **Quantize RoI**: Round to integer coordinates
2. **Divide into bins**: Split into $H \times W$ grid
3. **Max pool**: Max pool each bin
4. **Output**: Fixed-size $H \times W$ feature map

**Problem**: Quantization causes misalignment:
- RoI coordinates rounded
- Feature map coordinates rounded
- Misalignment between RoI and features

### RoI Align (Mask R-CNN)

RoI Align removes quantization:

1. **No quantization**: Use floating-point coordinates
2. **Bilinear interpolation**: Sample features at exact locations
3. **Average pool**: Average interpolated values in each bin
4. **Output**: Fixed-size feature map

### Bilinear Interpolation

For location $(x, y)$ (may be fractional), compute:

$$f(x, y) = \sum_{i,j} w_{i,j} \cdot F_{i,j}$$

where weights $w_{i,j}$ depend on distance to $(x, y)$:
$$w_{i,j} = (1 - |x-i|)(1 - |y-j|)$$

for $i, j \in \{\lfloor x \rfloor, \lceil x \rceil\} \times \{\lfloor y \rfloor, \lceil y \rceil\}$.

### Impact

RoI Align improves:
- **Mask accuracy**: Better alignment → better masks
- **Localization**: More precise boundaries
- **Small objects**: Handles small RoIs better

### Comparison

**RoI Pooling**: Fast but inaccurate (quantization)
**RoI Align**: Slightly slower but accurate (interpolation)

For mask prediction, accuracy is more important than speed.

## Mask Head Design

The mask head design balances accuracy and efficiency.

### Fully Convolutional Mask Head

Mask R-CNN uses a small FCN:

```
Input: RoI feature (14×14×256)
Conv1: 256 filters, 3×3, padding 1 → 14×14×256
Conv2: 256 filters, 3×3, padding 1 → 14×14×256
Conv3: 256 filters, 3×3, padding 1 → 14×14×256
Conv4: 256 filters, 3×3, padding 1 → 14×14×256
Deconv: 256 filters, 2×2, stride 2 → 28×28×256
Conv: K filters, 1×1 → 28×28×K
```

Output: $K$ binary masks of size 28×28.

### Design Choices

**Resolution**: 28×28 masks (balance accuracy/speed)
- Higher resolution: More accurate but slower
- Lower resolution: Faster but less accurate

**Depth**: 4 conv layers (sufficient for mask prediction)
- Deeper: More capacity but slower
- Shallower: Faster but less accurate

**Class-specific masks**: One mask per class
- Enables class-specific shape learning
- More parameters but better accuracy

### Alternative Designs

**Class-agnostic mask**: Single mask for all classes
- Fewer parameters
- Less capacity

**Higher resolution**: 56×56 or 112×112 masks
- More accurate boundaries
- Slower inference

**Refinement**: Additional refinement stages
- Iterative refinement
- More accurate but slower

## Panoptic Segmentation

Panoptic segmentation unifies instance and semantic segmentation.

### Task Definition

Panoptic segmentation requires:
- **Things**: Countable objects (person, car, bicycle)
- **Stuff**: Uncountable regions (sky, road, grass)

Output: Single segmentation map with:
- Instance IDs for things
- Class labels for stuff

### Panoptic Segmentation Methods

**Panoptic FPN**: Extends Mask R-CNN
- Instance branch: Things segmentation
- Semantic branch: Stuff segmentation
- Merging: Combine predictions

**UPSNet**: Unified panoptic segmentation network
- Shared backbone
- Separate heads for things and stuff
- Learned merging

**DETR**: Detection Transformer for panoptic
- End-to-end set prediction
- Unified architecture

### Merging Strategy

Combine instance and semantic predictions:

1. **Things**: Use instance masks
2. **Stuff**: Use semantic predictions
3. **Conflict resolution**: 
   - Things take priority over stuff
   - Resolve overlaps

### Panoptic Quality (PQ)

PQ metric combines recognition and segmentation:

$$\text{PQ} = \frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$$

where:
- $TP$: True positive segments
- $FP$: False positives
- $FN$: False negatives

PQ penalizes both false positives and false negatives.

## YOLACT and Real-Time Methods

Real-time instance segmentation enables applications requiring fast inference.

### YOLACT (You Only Look At Coefficients)

YOLACT is a single-stage instance segmentation method:

**Architecture**:
1. **Backbone**: ResNet-101 with FPN
2. **Protonet**: Generates prototype masks
3. **Prediction head**: Predicts coefficients + boxes/classes
4. **Assembly**: Combine prototypes and coefficients

**Key idea**: 
- Generate $k$ prototype masks (shared across image)
- Predict coefficients per instance
- Linear combination: $\text{mask} = \sum_i c_i \cdot \text{prototype}_i$

**Advantages**:
- Fast: Single forward pass
- Efficient: Shared prototypes
- Real-time: ~30 FPS

### YOLACT++

Improvements:
- Deformable convolutions
- Better anchor design
- Improved training

### Other Real-Time Methods

**PolarMask**: Polar representation of masks
**SOLO**: Segmenting Objects by Locations
**BlendMask**: Blending mask features

### Trade-offs

**Two-stage** (Mask R-CNN):
- Accurate but slow (~5 FPS)
- Good for accuracy-critical applications

**Single-stage** (YOLACT):
- Fast (~30 FPS) but less accurate
- Good for real-time applications

## Training and Loss Functions

Training Mask R-CNN requires careful loss design and sampling strategies.

### Multi-Task Loss

Mask R-CNN uses combined loss:

$$L = L_{cls} + L_{box} + L_{mask}$$

where:
- $L_{cls}$: Classification loss (cross-entropy)
- $L_{box}$: Bounding box regression loss (smooth L1)
- $L_{mask}$: Mask prediction loss (binary cross-entropy)

### Classification Loss

Standard cross-entropy:
$$L_{cls} = -\log(p_c)$$

where $p_c$ is probability of correct class.

### Box Regression Loss

Smooth L1 loss:
$$L_{box} = \sum_{i \in \{x,y,w,h\}} \text{smooth}_{L1}(t_i - t_i^*)$$

where $t_i$ are predicted box parameters and $t_i^*$ are targets.

### Mask Loss

Binary cross-entropy per pixel:
$$L_{mask} = -\frac{1}{m^2} \sum_{i,j} [y_{i,j} \log(\hat{m}_{i,j}) + (1-y_{i,j}) \log(1-\hat{m}_{i,j})]$$

where:
- $y_{i,j}$: Ground truth mask (0 or 1)
- $\hat{m}_{i,j}$: Predicted mask probability
- $m$: Mask resolution

**Important**: Only computed for positive RoIs (objects), not background.

### Training Strategy

**Pre-training**: Initialize backbone on ImageNet
**Fine-tuning**: Train on detection/segmentation dataset
**Multi-scale training**: Random scale augmentation
**Online hard example mining**: Focus on difficult examples

### Sampling

**RPN**: 
- 256 anchors per image
- 128 positives, 128 negatives

**RoI Head**:
- 512 RoIs per image
- 128 positives (IoU > 0.5), 384 negatives

**Mask branch**: Only positive RoIs

## Applications and Extensions

Instance segmentation enables diverse applications and research directions.

### Applications

**Autonomous driving**:
- Pedestrian/vehicle segmentation
- Scene understanding
- Path planning

**Medical imaging**:
- Organ segmentation
- Lesion detection
- Cell counting

**Robotics**:
- Object manipulation
- Scene understanding
- Grasp planning

**Augmented reality**:
- Object overlay
- Scene understanding
- Interaction

### Extensions

**3D instance segmentation**: Extend to point clouds/voxels
**Video instance segmentation**: Track instances across frames
**Panoptic segmentation**: Unified things and stuff
**Few-shot instance segmentation**: Learn from few examples

### Recent Advances

**Query-based methods**: DETR, Mask2Former
- Set prediction paradigm
- End-to-end training
- No anchors or NMS

**Transformer-based**: Vision transformers for segmentation
- Self-attention mechanisms
- Long-range dependencies
- Competitive performance

## Key Takeaways

1. Instance segmentation identifies and segments individual object instances, combining detection (which objects) and segmentation (precise boundaries).

2. Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch, enabling simultaneous detection and segmentation with shared features.

3. RoI Align fixes quantization issues in RoI Pooling through bilinear interpolation, critical for accurate mask prediction at pixel level.

4. Mask head uses a small FCN to predict class-specific binary masks, balancing accuracy (28×28 resolution) and efficiency.

5. Panoptic segmentation unifies instance segmentation (things) and semantic segmentation (stuff) into a single task with unified evaluation.

6. Real-time methods like YOLACT enable fast inference through prototype-based mask generation, trading some accuracy for speed.

7. Multi-task loss combines classification, box regression, and mask prediction, with mask loss computed only for positive RoIs.

8. Instance segmentation enables applications including autonomous driving, medical imaging, robotics, and augmented reality through precise object understanding.

9. Recent advances include query-based methods (DETR) and transformer architectures, moving toward end-to-end set prediction without anchors or NMS.

10. Understanding Mask R-CNN architecture, RoI Align, and mask head design provides foundation for instance segmentation, enabling applications requiring both object detection and precise segmentation.
