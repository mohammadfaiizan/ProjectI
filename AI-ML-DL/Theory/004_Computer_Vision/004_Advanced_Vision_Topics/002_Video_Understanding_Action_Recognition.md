# Video Understanding and Action Recognition

## Table of Contents

1. [Introduction](#introduction)
2. [Temporal Modeling Challenges](#temporal-modeling-challenges)
3. [3D CNNs for Video](#3d-cnns-for-video)
4. [Two-Stream Networks](#two-stream-networks)
5. [Video Transformers](#video-transformers)
6. [Action Detection](#action-detection)
7. [Temporal Action Localization](#temporal-action-localization)
8. [Video Representation Learning](#video-representation-learning)
9. [Multi-Modal Video Understanding](#multi-modal-video-understanding)
10. [Key Takeaways](#key-takeaways)

## Introduction

Video understanding extends image understanding to temporal sequences, requiring modeling of both spatial appearance and temporal dynamics. Action recognition identifies human actions in videos, while action detection localizes actions in both space and time. The evolution from 2D CNNs to 3D CNNs, two-stream networks, and video transformers represents advances in capturing temporal relationships.

Understanding video requires handling temporal dependencies, long-range relationships, and efficient processing of large video datasets. These methods enable applications including video search, surveillance, human-computer interaction, and autonomous systems.

## Temporal Modeling Challenges

Video understanding faces unique challenges compared to image understanding.

### Temporal Dependencies

**Short-term**: Frame-to-frame motion, optical flow
**Long-term**: Action sequences, temporal context
**Variable duration**: Actions have different lengths

### Computational Complexity

**Data volume**: Videos are much larger than images
- 1 second @ 30fps = 30 frames
- High resolution videos = GBs of data

**Processing**: Need to process multiple frames
- Temporal convolutions
- Attention mechanisms
- Recurrent networks

### Temporal Scale

**Fine-grained**: Frame-level actions (micro-expressions)
**Coarse-grained**: Clip-level actions (walking, running)
**Long-term**: Activity-level (cooking, sports)

### Motion vs Appearance

**Appearance**: Static visual features
**Motion**: Temporal changes, optical flow
**Both**: Most actions require both

## 3D CNNs for Video

3D CNNs extend 2D convolutions to temporal dimension, learning spatiotemporal features.

### 3D Convolution

3D convolution operates on video volumes:

$$y_{i,j,t} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} \sum_{u=0}^{U-1} w_{m,n,u} \cdot x_{i+m, j+n, t+u} + b$$

where:
- $x$: Input video volume $(H \times W \times T)$
- $w$: 3D filter $(M \times N \times U)$
- $y$: Output volume

### C3D Architecture

C3D uses 3×3×3 convolutions throughout:

**Architecture**:
- Conv1: 64 filters, 3×3×3
- Pool1: 1×2×2 (temporal stride 1)
- Conv2: 128 filters, 3×3×3
- Pool2: 2×2×2
- Conv3-5: 256, 512, 512 filters
- FC layers: 4096, 4096, classes

**Key**: Small 3×3×3 filters, temporal pooling.

### I3D (Inflated 3D)

I3D inflates 2D ImageNet pre-trained weights to 3D:

**Inflation**:
- 2D filter $w_{2D}$ → 3D filter $w_{3D}$
- Copy 2D filter across temporal dimension
- Average and normalize

**Benefits**:
- Transfer ImageNet knowledge
- Better than training from scratch
- Strong performance

### X3D

X3D scales network efficiently:

**Scaling dimensions**:
- **Depth**: Number of layers
- **Width**: Number of channels
- **Resolution**: Spatial and temporal
- **Bottleneck**: Channel width

**Efficient design**: Balance accuracy and efficiency.

### Limitations

- **Memory**: 3D convolutions are memory-intensive
- **Computation**: More expensive than 2D
- **Temporal range**: Limited by receptive field

## Two-Stream Networks

Two-stream networks combine spatial and temporal streams for action recognition.

### Architecture

**Spatial stream**: 
- Input: RGB frames
- Network: 2D CNN (e.g., VGG, ResNet)
- Captures: Appearance features

**Temporal stream**:
- Input: Optical flow
- Network: 2D CNN (same architecture)
- Captures: Motion features

**Fusion**: Combine predictions
- **Late fusion**: Average scores
- **Early fusion**: Concatenate features

### Optical Flow

Optical flow captures motion between frames:

**Dense flow**: Flow vector per pixel
**Sparse flow**: Flow at feature points

**Stacking**: Stack 10 frames of flow (x and y) → 20 channels

### Two-Stream ConvNets

**Spatial stream**: 
- Single frame input
- Pre-trained on ImageNet
- Fine-tuned on video

**Temporal stream**:
- 10 stacked optical flow frames
- Trained from scratch
- Captures motion

**Fusion**: Average softmax scores

### Improvements

**TSN (Temporal Segment Networks)**:
- Sample segments from video
- Process each segment
- Aggregate predictions

**TRN (Temporal Relation Networks)**:
- Model temporal relations
- Multi-scale temporal modeling

## Video Transformers

Transformers apply self-attention to video, capturing long-range temporal dependencies.

### Vision Transformer for Video

Extend ViT to video:

**Patches**: 3D patches (spatial + temporal)
**Positional encoding**: Spatial + temporal positions
**Self-attention**: Across all patches

### TimeSformer

TimeSformer uses divided space-time attention:

**Space attention**: Within each frame
**Time attention**: Across frames at same spatial location

**Efficiency**: Reduces computation from $O(N^2)$ to $O(N \times (H \times W + T))$

### ViViT

ViViT (Video Vision Transformer):

**Tubelet embedding**: Extract 3D patches
**Factorized attention**: Separate spatial and temporal
**Multi-head attention**: Standard transformer attention

### Video Swin Transformer

Swin Transformer adapted for video:

**3D shifted windows**: Extend 2D windows to 3D
**Hierarchical**: Multi-scale features
**Efficient**: Linear complexity

### Benefits

- **Long-range**: Captures long temporal dependencies
- **Flexible**: Handles variable-length videos
- **Transfer**: Pre-trained on large datasets

## Action Detection

Action detection localizes actions in both space and time.

### Spatiotemporal Detection

**Spatial**: Bounding boxes in frames
**Temporal**: Action segments in time
**Output**: Tubes (3D boxes over time)

### Methods

**Frame-level detection**: Detect in each frame, link across time
**Tube detection**: Detect action tubes directly
**3D proposal**: Generate 3D proposals

### Action Tubes

Action tubes are sequences of bounding boxes:

$$\mathcal{T} = \{b_1, b_2, \ldots, b_T\}$$

where $b_t$ is bounding box at time $t$.

**Linking**: Connect detections across frames
- **Tracking**: Use tracking algorithms
- **Temporal consistency**: Enforce smoothness

### Tubelet Detection

Detect short tubelets, then link:

1. **Generate tubelets**: Short sequences (e.g., 8 frames)
2. **Classify**: Action class per tubelet
3. **Link**: Connect tubelets into tubes
4. **Refine**: Temporal NMS, smoothing

## Temporal Action Localization

Temporal action localization finds action boundaries in time.

### Problem

Given untrimmed video, find:
- **Action segments**: Start and end times
- **Action classes**: What actions occur

### Methods

**Two-stage**:
1. **Proposal generation**: Generate candidate segments
2. **Classification**: Classify proposals

**One-stage**:
- Directly predict boundaries and classes

### Proposal Generation

**Sliding window**: Fixed-size windows
**Temporal actionness**: Score action likelihood
**Boundary detection**: Detect start/end boundaries

### Boundary Detection

**Temporal boundary detection**:
- Detect action starts and ends
- Use boundary classifiers
- Refine boundaries

### Evaluation

**mAP**: Mean average precision
- IoU threshold: 0.5, 0.75, etc.
- Average over classes

**Temporal IoU**: Overlap in time
$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$$

## Video Representation Learning

Learning good video representations is crucial for downstream tasks.

### Self-Supervised Learning

**Temporal order**: Predict frame order
**Speed**: Predict playback speed
**Rotation**: Predict rotation in time
**Contrastive**: Contrastive learning across time

### Contrastive Video Learning

**Positive pairs**: Frames from same video
**Negative pairs**: Frames from different videos

**Temporal contrast**: 
- Close frames: Similar
- Distant frames: Different

### Video-MAE

Masked autoencoding for video:
- **Mask**: Random patches in space-time
- **Reconstruct**: Predict masked patches
- **Pre-training**: Large-scale unlabeled data

### Benefits

- **Transfer**: Good representations transfer well
- **Efficiency**: Less labeled data needed
- **Generalization**: Learn general video understanding

## Multi-Modal Video Understanding

Videos contain multiple modalities: visual, audio, text.

### Modalities

**Visual**: RGB frames, optical flow
**Audio**: Sound, speech
**Text**: Subtitles, captions
**Motion**: Skeleton, pose

### Fusion Strategies

**Early fusion**: Concatenate inputs
**Late fusion**: Combine predictions
**Attention**: Learn to attend to modalities

### Audio-Visual Learning

**Synchronization**: Learn audio-visual correspondence
**Separation**: Separate audio sources
**Localization**: Localize sound sources

### Video-Language

**Video captioning**: Generate text descriptions
**Video QA**: Answer questions about video
**Video retrieval**: Retrieve videos by text query

### Multi-Modal Transformers

**Cross-attention**: Attention between modalities
**Fusion layers**: Combine modalities
**Pre-training**: Large-scale multi-modal data

## Key Takeaways

1. Video understanding requires modeling temporal dependencies at multiple scales, from frame-to-frame motion to long-term action sequences.

2. 3D CNNs extend 2D convolutions to temporal dimension, learning spatiotemporal features through 3D convolutions and pooling operations.

3. Two-stream networks combine spatial (RGB) and temporal (optical flow) streams, capturing both appearance and motion for action recognition.

4. Video transformers apply self-attention to video patches, enabling long-range temporal modeling through divided space-time attention (TimeSformer) or factorized attention (ViViT).

5. Action detection localizes actions in space and time, generating action tubes (sequences of bounding boxes) through frame-level detection and temporal linking.

6. Temporal action localization finds action boundaries in untrimmed videos, using proposal generation and classification or one-stage boundary detection.

7. Video representation learning uses self-supervision (temporal order, contrastive learning) to learn general-purpose features from unlabeled video data.

8. Multi-modal video understanding combines visual, audio, and text modalities through fusion strategies and cross-modal attention for richer understanding.

9. Efficient video processing requires balancing accuracy and computation, with methods like X3D scaling networks efficiently and video transformers reducing complexity through factorized attention.

10. Understanding video understanding and action recognition enables applications including video search, surveillance, human-computer interaction, and autonomous systems through spatiotemporal modeling and multi-modal fusion.
