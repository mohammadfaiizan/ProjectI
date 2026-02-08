# Semantic Segmentation: FCN to U-Net

## Table of Contents

1. [Introduction](#introduction)
2. [Fully Convolutional Networks (FCN)](#fully-convolutional-networks-fcn)
3. [U-Net Architecture](#u-net-architecture)
4. [DeepLab Series](#deeplab-series)
5. [Dilated and Atrous Convolutions](#dilated-and-atrous-convolutions)
6. [Encoder-Decoder Architecture](#encoder-decoder-architecture)
7. [Pixel-Wise Classification](#pixel-wise-classification)
8. [Multi-Scale Context](#multi-scale-context)
9. [Segmentation Loss Functions](#segmentation-loss-functions)
10. [Key Takeaways](#key-takeaways)

## Introduction

Semantic segmentation assigns a class label to every pixel in an image, providing dense semantic understanding. Unlike object detection (bounding boxes) or classification (image-level labels), segmentation requires pixel-level precision, making it one of the most challenging computer vision tasks.

The evolution from FCN to modern architectures like DeepLab and U-Net represents advances in handling multi-scale context, preserving spatial resolution, and efficiently processing high-resolution images. These methods enable applications including autonomous driving, medical imaging, scene understanding, and image editing.

## Fully Convolutional Networks (FCN)

FCN was the first to apply deep learning effectively to semantic segmentation by converting classification networks to dense prediction.

### From Classification to Segmentation

Classification networks (e.g., VGG, ResNet) end with fully connected layers:
- Input: Variable size
- Output: Fixed-size class scores

FCN replaces FC layers with convolutions:
- Input: Any size
- Output: Dense prediction map

### Architecture

**Base network**: Pre-trained classification network (VGG-16, ResNet)
**Conversion**: Replace FC layers with convolutions
**Upsampling**: Deconvolution (transposed convolution) to recover resolution

### Deconvolution

Deconvolution (transposed convolution) upsamples feature maps:

$$y_{i,j} = \sum_{m,n} w_{m,n} \cdot x_{i-m, j-n}$$

where indices wrap around or use zero-padding.

For stride $s$ and kernel size $k$:
- Input size: $H \times W$
- Output size: $s(H-1) + k$

### Skip Connections

FCN uses skip connections to combine multi-scale features:

**FCN-32s**: Single upsampling (coarse)
**FCN-16s**: Skip connection from pool4 (finer)
**FCN-8s**: Skip connections from pool3 and pool4 (finest)

Skip connections preserve fine details lost in downsampling.

### Architecture Details

1. **Encoder**: VGG-16 (conv + pooling layers)
2. **Decoder**: 
   - 1×1 conv to reduce channels
   - Deconvolution to upsample
   - Element-wise addition with skip connections
3. **Final layer**: Deconvolution to original resolution + softmax

### Training

**Pre-training**: Initialize with ImageNet classification weights
**Fine-tuning**: Train on segmentation datasets
**Loss**: Pixel-wise cross-entropy

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})$$

where $N$ is pixels, $C$ is classes, $y$ is ground truth, $\hat{p}$ is predictions.

### Limitations

- **Coarse predictions**: Limited by upsampling from low resolution
- **Boundary accuracy**: Struggles with fine boundaries
- **Context**: Limited receptive field

## U-Net Architecture

U-Net introduced the encoder-decoder architecture with skip connections, becoming the standard for medical image segmentation.

### Architecture Design

**Encoder (contracting path)**:
- Repeated: 3×3 conv, ReLU, 3×3 conv, ReLU
- 2×2 max pooling with stride 2
- Double channels at each downsampling

**Decoder (expansive path)**:
- 2×2 upsampling (deconvolution)
- Concatenate with corresponding encoder feature
- 3×3 conv, ReLU, 3×3 conv, ReLU
- Halve channels at each upsampling

**Skip connections**: Connect encoder and decoder at same resolution

### Symmetric Architecture

U-Net is symmetric:
- Encoder: 4 downsampling steps
- Decoder: 4 upsampling steps
- Skip connections at each level

This preserves spatial information throughout the network.

### Feature Concatenation

Skip connections use concatenation (not addition):
$$D_l = \text{Concat}[\text{Upsample}(D_{l+1}), E_l]$$

where $D_l$ is decoder feature at level $l$ and $E_l$ is encoder feature.

Benefits:
- Preserves all information from encoder
- Decoder can choose what to use
- More capacity than addition

### Training Strategy

**Data augmentation**: Critical for small datasets
- Elastic deformations
- Rotations
- Scaling
- Intensity variations

**Loss function**: Pixel-wise cross-entropy + Dice loss (for medical imaging)

**Small batches**: Often 1-2 images per batch due to memory

### Advantages

- **Precise boundaries**: Skip connections preserve fine details
- **Efficient**: Relatively small network
- **Effective**: Works well with limited data
- **Flexible**: Adaptable to various tasks

### Variants

**3D U-Net**: Extension to volumetric data
**Attention U-Net**: Adds attention gates
**Residual U-Net**: Adds residual connections
**Dense U-Net**: Dense connections in blocks

## DeepLab Series

DeepLab addresses semantic segmentation challenges through atrous convolutions, ASPP, and CRF post-processing.

### DeepLab v1

**Key innovations**:
- **Atrous convolution**: Increases receptive field without downsampling
- **CRF post-processing**: Refines boundaries using conditional random fields

**Architecture**:
- Base: VGG-16 with atrous convolutions
- Final prediction: Bilinear upsampling
- CRF: Refine boundaries

### DeepLab v2

**Improvements**:
- **ASPP**: Atrous Spatial Pyramid Pooling for multi-scale context
- **Better backbone**: ResNet-101
- **Multi-scale input**: Test-time augmentation

### Atrous Spatial Pyramid Pooling (ASPP)

ASPP applies atrous convolutions at multiple rates in parallel:

$$\text{ASPP}(x) = \text{Concat}[\text{AtrousConv}(x, r=6), \text{AtrousConv}(x, r=12), \text{AtrousConv}(x, r=18), \text{GlobalAvgPool}(x)]$$

where $r$ is the dilation rate.

Benefits:
- Captures objects at multiple scales
- No additional parameters
- Efficient parallel computation

### DeepLab v3

**Improvements**:
- **Improved ASPP**: Image-level features, batch normalization
- **Encoder-decoder**: Optional decoder for refinement
- **Better training**: Longer training, poly learning rate

**ASPP with image pooling**:
$$\text{ASPP}(x) = \text{Concat}[\text{AtrousConv}(x, r=6), \text{AtrousConv}(x, r=12), \text{AtrousConv}(x, r=18), \text{ImagePool}(x)]$$

ImagePool: Global average pooling + 1×1 conv + bilinear upsampling

### DeepLab v3+

**Architecture**:
- **Encoder**: Xception or ResNet with atrous convolutions + ASPP
- **Decoder**: Simple decoder for boundary refinement

**Decoder design**:
1. Upsample encoder output by 4×
2. Concatenate with low-level features (from encoder)
3. 3×3 conv + upsampling to final resolution

**Improvements**:
- Better boundary accuracy
- Simpler than v3
- State-of-the-art performance

## Dilated and Atrous Convolutions

Dilated (atrous) convolutions increase receptive field without downsampling or additional parameters.

### Standard Convolution

Standard convolution with kernel size $k$:
$$y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} w_{m,n} \cdot x_{i+m, j+n}$$

Receptive field: $k \times k$

### Dilated Convolution

Dilated convolution with dilation rate $r$:
$$y_{i,j} = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} w_{m,n} \cdot x_{i+rm, j+rn}$$

Effective kernel size: $(k-1)r + 1$
Receptive field: Larger without downsampling

### Benefits

- **Larger receptive field**: Capture more context
- **Same resolution**: No information loss from downsampling
- **Efficient**: Same number of parameters as standard conv

### Example

For 3×3 kernel:
- Standard: Receptive field 3×3
- Dilation rate 2: Receptive field 5×5
- Dilation rate 4: Receptive field 9×9

### Gridding Artifacts

Large dilation rates can cause gridding artifacts (checkerboard patterns). Solutions:
- Hybrid approach: Mix standard and dilated convolutions
- Gradual increase: Start with small rates, increase gradually

## Encoder-Decoder Architecture

Encoder-decoder is the standard architecture for dense prediction tasks.

### Encoder

**Purpose**: Extract high-level semantic features
**Design**: Classification network (VGG, ResNet, Xception)
**Output**: Low-resolution, high-semantic features

**Downsampling**: 
- Strided convolutions
- Pooling layers
- Typically 32× downsampling

### Decoder

**Purpose**: Recover spatial resolution
**Design**: Upsampling layers
**Output**: High-resolution predictions

**Upsampling methods**:
- **Deconvolution**: Learnable upsampling
- **Bilinear interpolation**: Fixed upsampling
- **Unpooling**: Remember max locations

### Skip Connections

Skip connections preserve fine details:

**Addition**: $D_l = \text{Upsample}(D_{l+1}) + E_l$
**Concatenation**: $D_l = \text{Concat}[\text{Upsample}(D_{l+1}), E_l]$

Concatenation preserves more information.

### Multi-Scale Features

Combine features at multiple scales:
- **Pyramid pooling**: Pool at multiple scales
- **ASPP**: Parallel atrous convolutions
- **Skip connections**: Multiple resolution levels

## Pixel-Wise Classification

Semantic segmentation is fundamentally pixel-wise classification.

### Dense Prediction

For each pixel $(i, j)$, predict class:
$$\hat{y}_{i,j} = \arg\max_c p_{i,j,c}$$

where $p_{i,j,c}$ is the probability of class $c$ at pixel $(i, j)$.

### Softmax

Final layer applies softmax:
$$p_{i,j,c} = \frac{\exp(z_{i,j,c})}{\sum_{c'=1}^{C} \exp(z_{i,j,c'})}$$

where $z_{i,j,c}$ are logits from network.

### Class Imbalance

Segmentation datasets often have severe class imbalance:
- Background pixels dominate
- Rare classes have few pixels

Solutions:
- **Weighted loss**: Weight classes by inverse frequency
- **Focal loss**: Down-weight easy examples
- **Dice loss**: Focus on overlapping regions

### Evaluation Metrics

**Pixel accuracy**: Fraction of correctly classified pixels
$$\text{PA} = \frac{\sum_{i,j} \mathbb{1}[y_{i,j} = \hat{y}_{i,j}]}{N}$$

**Mean IoU**: Average intersection over union per class
$$\text{mIoU} = \frac{1}{C} \sum_{c=1}^{C} \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$

where $P_c$ is predicted pixels for class $c$ and $G_c$ is ground truth.

## Multi-Scale Context

Capturing multi-scale context is crucial for segmentation.

### Why Multi-Scale?

- **Objects at different sizes**: Need to detect both small and large objects
- **Context matters**: Surrounding information helps classification
- **Receptive field**: Limited by network depth

### Methods

**Image pyramids**: Process at multiple scales, combine results
**Feature pyramids**: Multi-scale features within network
**Atrous convolutions**: Multiple dilation rates (ASPP)
**Pyramid pooling**: Pool at multiple scales

### Pyramid Pooling Module (PPM)

PPM pools features at multiple scales:

$$\text{PPM}(x) = \text{Concat}[\text{GlobalPool}(x), \text{Pool}_{2×2}(x), \text{Pool}_{4×4}(x), \text{Pool}_{8×8}(x)]$$

Each pooled feature is upsampled to original size before concatenation.

### Multi-Scale Inference

Test-time augmentation:
1. Process image at multiple scales
2. Average or vote predictions
3. Improves accuracy (slower)

## Segmentation Loss Functions

Loss function design is critical for segmentation performance.

### Cross-Entropy Loss

Standard pixel-wise cross-entropy:
$$L_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$$

**Weighted cross-entropy**: Weight classes by inverse frequency
$$L_{WCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} w_c y_{i,c} \log(p_{i,c})$$

where $w_c = \frac{1}{\log(1.02 + f_c)}$ and $f_c$ is class frequency.

### Dice Loss

Dice loss focuses on overlapping regions:
$$L_{Dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$$

For multi-class:
$$L_{Dice} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2\sum_i p_{i,c} y_{i,c}}{\sum_i p_{i,c} + \sum_i y_{i,c}}$$

Benefits:
- Handles class imbalance
- Focuses on correct predictions
- Good for medical imaging

### Focal Loss

Focal loss down-weights easy examples:
$$L_{Focal} = -\alpha (1-p_t)^\gamma \log(p_t)$$

where $p_t$ is probability of true class and $\gamma > 0$ is focusing parameter.

### Combined Losses

Often combine multiple losses:
$$L = \lambda_1 L_{CE} + \lambda_2 L_{Dice} + \lambda_3 L_{Boundary}$$

Boundary loss focuses on object boundaries.

## Key Takeaways

1. FCN converted classification networks to dense prediction by replacing FC layers with convolutions and using deconvolution for upsampling.

2. U-Net introduced the symmetric encoder-decoder architecture with skip connections, preserving fine details and becoming standard for medical segmentation.

3. DeepLab series addressed segmentation through atrous convolutions for larger receptive fields and ASPP for multi-scale context aggregation.

4. Dilated (atrous) convolutions increase receptive field without downsampling, enabling dense prediction at full resolution.

5. Encoder-decoder architecture extracts high-level semantics in the encoder and recovers spatial resolution in the decoder, with skip connections preserving details.

6. Semantic segmentation is pixel-wise classification, requiring handling of class imbalance and evaluation using metrics like mIoU.

7. Multi-scale context is crucial for handling objects at different sizes, achieved through image/feature pyramids, ASPP, and pyramid pooling.

8. Skip connections (concatenation or addition) preserve fine details lost during downsampling, critical for accurate boundary prediction.

9. Loss functions like weighted cross-entropy, Dice loss, and focal loss address class imbalance and focus learning on challenging examples.

10. Understanding FCN, U-Net, and DeepLab architectures provides insight into dense prediction design: preserving resolution, multi-scale context, and efficient upsampling strategies.
