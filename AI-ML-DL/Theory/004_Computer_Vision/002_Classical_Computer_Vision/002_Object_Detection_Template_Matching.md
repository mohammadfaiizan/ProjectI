# Object Detection and Template Matching

## Table of Contents

1. [Introduction](#introduction)
2. [Template Matching](#template-matching)
3. [Sliding Window Approach](#sliding-window-approach)
4. [Viola-Jones Detector](#viola-jones-detector)
5. [HOG and SVM Detection](#hog-and-svm-detection)
6. [Cascade Classifiers](#cascade-classifiers)
7. [Deformable Parts Model](#deformable-parts-model)
8. [Multi-Scale Detection](#multi-scale-detection)
9. [Detection Evaluation](#detection-evaluation)
10. [Key Takeaways](#key-takeaways)

## Introduction

Object detection is a fundamental computer vision task that identifies and localizes objects of interest within images. Unlike classification, which assigns a label to an entire image, detection must identify both what objects are present and where they are located, typically represented by bounding boxes.

Template matching represents one of the earliest approaches to object detection, searching for instances of a template pattern within an image. Modern methods combine feature extraction, machine learning classifiers, and efficient search strategies to achieve robust detection across varying scales, orientations, and appearances.

## Template Matching

Template matching finds instances of a template image $T$ within a larger image $I$ by comparing the template at every possible location.

### Correlation-Based Matching

The normalized cross-correlation measures similarity:

$$R(x, y) = \frac{\sum_{x',y'} (T(x', y') - \bar{T})(I(x+x', y+y') - \bar{I}(x, y))}{\sqrt{\sum_{x',y'} (T(x', y') - \bar{T})^2 \sum_{x',y'} (I(x+x', y+y') - \bar{I}(x, y))^2}}$$

where $\bar{T}$ is the mean of template $T$ and $\bar{I}(x, y)$ is the mean of the image patch at $(x, y)$.

Normalization makes the measure invariant to brightness changes.

### Distance Metrics

**Sum of Squared Differences (SSD)**:
$$D_{SSD}(x, y) = \sum_{x',y'} (T(x', y') - I(x+x', y+y'))^2$$

**Sum of Absolute Differences (SAD)**:
$$D_{SAD}(x, y) = \sum_{x',y'} |T(x', y') - I(x+x', y+y')|$$

**Normalized Cross-Correlation (NCC)**:
$$R_{NCC}(x, y) = \frac{\sum_{x',y'} T(x', y') I(x+x', y+y')}{\sqrt{\sum_{x',y'} T(x', y')^2 \sum_{x',y'} I(x+x', y+y')^2}}$$

### Template Matching Limitations

- **Scale sensitivity**: Template must match object scale exactly
- **Rotation sensitivity**: Template must match object orientation
- **Computational cost**: $O(MN \cdot WH)$ for $M \times N$ image and $W \times H$ template
- **Illumination sensitivity**: Brightness changes affect matching

### Efficient Template Matching

**FFT-based correlation**: Reduces complexity to $O(MN \log(MN))$:
$$I * T = \mathcal{F}^{-1}(\mathcal{F}(I) \cdot \mathcal{F}^*(T))$$

**Pyramid matching**: Match at multiple scales using image pyramids
**Coarse-to-fine**: Start with low resolution, refine at high resolution

## Sliding Window Approach

The sliding window approach systematically searches for objects by evaluating a classifier at every possible window location and scale.

### Basic Algorithm

1. **Define window size**: Typically based on expected object size
2. **Slide window**: Move window across image with step size $\Delta$
3. **Extract features**: Compute features for each window
4. **Classify**: Apply classifier to determine if object is present
5. **Non-maximum suppression**: Remove overlapping detections

### Multi-Scale Search

Objects appear at different scales, requiring search at multiple scales:

1. **Image pyramid**: Create scaled versions of image
2. **Fixed window**: Use same window size at all scales
3. **Scale space**: Search across scale dimension

For scale factor $s$:
$$I_s(x, y) = I(s \cdot x, s \cdot y)$$

### Window Size and Step Size

**Window size**: Should match expected object size
- Too small: Miss large objects
- Too large: Poor localization, high false positives

**Step size**: Trade-off between speed and coverage
- Small step: Better coverage, slower
- Large step: Faster, may miss objects

Typical step sizes: 4-8 pixels for detection, 1-2 pixels for refinement.

### Computational Complexity

For image size $M \times N$, window size $W \times H$, step size $\Delta$, and $S$ scales:

$$\text{Number of windows} = S \cdot \left\lfloor\frac{M-W}{\Delta}\right\rfloor \cdot \left\lfloor\frac{N-H}{\Delta}\right\rfloor$$

This can be millions of windows, making efficient classifiers essential.

## Viola-Jones Detector

The Viola-Jones detector was the first real-time face detector, combining several key innovations: Haar-like features, integral images, AdaBoost, and cascade classifiers.

### Haar-Like Features

Haar-like features are rectangular patterns that capture intensity differences:

**Two-rectangle features**:
- Horizontal: Difference between left and right rectangles
- Vertical: Difference between top and bottom rectangles

**Three-rectangle features**:
- Horizontal: Difference between sum of outer rectangles and middle
- Vertical: Difference between sum of outer rectangles and middle

**Four-rectangle features**:
- Difference between diagonal pairs

A feature value is computed as:
$$f = \sum_{\text{white}} I(x, y) - \sum_{\text{black}} I(x, y)$$

### Integral Images

Integral images enable constant-time computation of rectangular sums:

$$II(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)$$

The sum of any rectangle can be computed in $O(1)$:
$$\text{Sum}(x_1, y_1, x_2, y_2) = II(x_2, y_2) - II(x_1-1, y_2) - II(x_2, y_1-1) + II(x_1-1, y_1-1)$$

This makes Haar feature computation extremely fast.

### AdaBoost Learning

AdaBoost combines weak classifiers into a strong classifier:

$$H(\mathbf{x}) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(\mathbf{x})\right)$$

where:
- $h_t$: Weak classifier (threshold on single Haar feature)
- $\alpha_t$: Weight learned during training
- $T$: Number of weak classifiers

Each weak classifier selects:
1. A Haar feature
2. A threshold
3. A polarity (direction of inequality)

AdaBoost iteratively:
1. Trains weak classifier on weighted training set
2. Updates weights to focus on misclassified examples
3. Combines weak classifiers with learned weights

### Cascade Structure

The cascade structure enables fast rejection of non-objects:

1. **Stage 1**: Simple classifier rejects ~50% of windows
2. **Stage 2**: More complex classifier rejects ~50% of remaining
3. **Stage N**: Final classifier makes final decision

Each stage is trained to achieve:
- **High detection rate**: $d \geq 0.999$ (miss very few objects)
- **Moderate false positive rate**: $f \leq 0.5$ (reject half of negatives)

Overall cascade performance:
- Detection rate: $D = \prod d_i$
- False positive rate: $F = \prod f_i$

For 10 stages with $d=0.999$ and $f=0.5$:
- $D \approx 0.99$
- $F \approx 0.001$ (rejects 99.9% of negatives)

## HOG and SVM Detection

The Histogram of Oriented Gradients (HOG) descriptor combined with Support Vector Machine (SVM) classifier achieved state-of-the-art pedestrian detection.

### HOG Feature Extraction

HOG captures local shape through gradient orientation histograms:

1. **Gradient computation**: Compute $G_x$ and $G_y$ using central differences
2. **Cell histograms**: Divide window into cells (e.g., 8×8 pixels), create orientation histogram for each
3. **Block normalization**: Group cells into blocks (e.g., 2×2 cells), normalize histograms
4. **Descriptor**: Concatenate normalized histograms

For a detection window, HOG produces a high-dimensional feature vector (typically 3780 dimensions for 64×128 window).

### SVM Classifier

SVM finds the optimal separating hyperplane:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

subject to:
$$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

where:
- $\mathbf{w}$: Normal vector to hyperplane
- $b$: Bias term
- $C$: Regularization parameter
- $\xi_i$: Slack variables for soft margin

The decision function:
$$f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

Positive $f(\mathbf{x})$ indicates object presence.

### Hard Negative Mining

Hard negative mining improves detection by retraining on false positives:

1. Train initial classifier on positive and negative examples
2. Run detector on negative images (images without objects)
3. Collect false positives (hard negatives)
4. Retrain classifier including hard negatives
5. Repeat until convergence

This focuses learning on challenging negative examples.

### Detection Pipeline

1. **Multi-scale detection**: Search at multiple scales using image pyramid
2. **Sliding window**: Evaluate classifier at each window location
3. **Non-maximum suppression**: Remove overlapping detections
4. **Thresholding**: Keep detections with score above threshold

## Cascade Classifiers

Cascade classifiers enable efficient object detection by quickly rejecting negative windows.

### Cascade Architecture

A cascade consists of $N$ stages, each a classifier:

$$C(\mathbf{x}) = \begin{cases}
\text{object} & \text{if } h_N(\mathbf{x}) = 1 \text{ and } \ldots \text{ and } h_1(\mathbf{x}) = 1 \\
\text{background} & \text{otherwise}
\end{cases}$$

Windows are rejected as soon as any stage classifies them as background.

### Stage Design

Each stage $i$ is designed to achieve:
- **Detection rate**: $d_i \geq d_{target}$
- **False positive rate**: $f_i \leq f_{target}$

Typical targets: $d_{target} = 0.995$, $f_{target} = 0.5$

### Training Cascade

1. **Initialize**: Set target detection rate $d_{target}$ and false positive rate $f_{target}$
2. **For each stage**:
   - Add negative examples (initially random, later hard negatives)
   - Train classifier to meet $d_i \geq d_{target}$ and $f_i \leq f_{target}$
   - Evaluate on validation set
   - If overall false positive rate meets target, stop
3. **Hard negative mining**: Run current cascade on negative images, collect false positives
4. **Repeat**: Add hard negatives and retrain stages

### Computational Efficiency

Cascade efficiency comes from early rejection:
- Most windows are rejected by early stages
- Only promising windows reach expensive later stages
- Average computation per window is much less than full classifier

## Deformable Parts Model

The Deformable Parts Model (DPM) represents objects as collections of parts with spatial relationships, handling pose variation and partial occlusion.

### Part-Based Representation

An object is represented as:
- **Root filter**: Coarse template covering entire object
- **Part filters**: Finer templates for object parts
- **Deformation costs**: Penalties for part displacement from ideal positions

### Model Structure

A DPM consists of:
- Root filter $F_0$ at resolution $H_0 \times W_0$
- $n$ part filters $F_1, \ldots, F_n$ at resolution $H \times W$
- Deformation parameters $(a_i, b_i, c_i, d_i)$ for each part

### Detection Score

The score for placing root at $(x_0, y_0)$ and parts at $(x_i, y_i)$:

$$S = \sum_{i=0}^{n} F_i \cdot \phi(H, (x_i, y_i)) - \sum_{i=1}^{n} d_i((x_i, y_i) - (x_0, y_0) + v_i)$$

where:
- $\phi(H, (x, y))$: Feature vector at location $(x, y)$
- $v_i$: Ideal part location relative to root
- $d_i$: Deformation cost function

Deformation cost:
$$d_i(dx, dy) = a_i dx^2 + b_i dx dy + c_i dy^2 + d_i dx + e_i dy$$

### Latent SVM Training

DPM uses Latent SVM (LSVM) to learn part locations:

1. **Initialize**: Set part locations manually or using heuristics
2. **Positive examples**: For each positive example, find optimal part locations (latent variables)
3. **Negative examples**: Use hard negative mining
4. **Optimize**: Update model parameters using coordinate descent

The optimization alternates between:
- **Latent variable update**: Find best part locations for positives
- **Parameter update**: Update filter and deformation parameters

### Detection Algorithm

1. **Root filter detection**: Detect root at multiple scales
2. **Part filter detection**: Detect parts at twice the resolution
3. **Composition**: Combine root and parts, accounting for deformation costs
4. **Non-maximum suppression**: Remove overlapping detections

DPM achieves good performance on PASCAL VOC dataset but is computationally expensive.

## Multi-Scale Detection

Objects appear at various scales, requiring multi-scale search strategies.

### Image Pyramid

Create scaled versions of the image:

$$I_s(x, y) = I(s \cdot x, s \cdot y)$$

Scale factors typically: $s \in \{2^{-k/2} : k = 0, 1, 2, \ldots\}$ (octave-based)

### Feature Pyramid

Instead of scaling images, scale features:

1. Compute features at original scale
2. Downsample feature maps
3. Apply detector at each feature scale

More efficient than image pyramid for feature-based detectors.

### Scale-Invariant Features

Some features are naturally scale-invariant:
- **SIFT**: Detects features at multiple scales
- **Multi-scale HOG**: Computes HOG at multiple scales
- **Scale-space representation**: Build scale-space pyramid

### Scale Selection

Strategies for scale selection:
- **Exhaustive**: Search all scales (slow but complete)
- **Coarse-to-fine**: Start coarse, refine at promising scales
- **Scale prediction**: Predict likely scales from context
- **Scale-invariant detector**: Use scale-invariant features

## Detection Evaluation

Standard metrics evaluate detection performance.

### Intersection over Union (IoU)

IoU measures overlap between predicted and ground truth boxes:

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}$$

Typical threshold: $\text{IoU} \geq 0.5$ for a correct detection.

### Precision and Recall

**Precision**: Fraction of detections that are correct
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall**: Fraction of objects that are detected
$$\text{Recall} = \frac{TP}{TP + FN}$$

where:
- $TP$: True positives (correct detections)
- $FP$: False positives (incorrect detections)
- $FN$: False negatives (missed objects)

### Average Precision (AP)

AP summarizes precision-recall curve:

$$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$

where $P_n$ and $R_n$ are precision and recall at the $n$-th threshold.

**Mean Average Precision (mAP)**: Average AP across all object classes.

### Non-Maximum Suppression

NMS removes duplicate detections:

1. Sort detections by confidence score
2. Select highest scoring detection
3. Remove all detections with $\text{IoU} > \tau$ with selected detection
4. Repeat until no detections remain

Typical threshold: $\tau = 0.5$

## Key Takeaways

1. Template matching finds objects by correlating templates with image regions, but is sensitive to scale, rotation, and illumination changes.

2. The sliding window approach systematically searches for objects by evaluating classifiers at all locations and scales, requiring efficient feature extraction and classification.

3. Viola-Jones detector combines Haar features, integral images, AdaBoost, and cascades to achieve real-time face detection with high accuracy.

4. HOG features capture local shape through gradient orientation histograms, and when combined with SVM classifiers, achieve strong pedestrian detection performance.

5. Cascade classifiers enable efficient detection by quickly rejecting negative windows through early-stage classifiers, dramatically reducing computation.

6. Deformable Parts Models represent objects as collections of parts with spatial relationships, handling pose variation and partial occlusion through learned deformation costs.

7. Multi-scale detection is essential for handling objects at different sizes, typically implemented through image pyramids or feature pyramids.

8. Hard negative mining improves detection by retraining classifiers on challenging false positive examples, focusing learning on difficult cases.

9. Detection evaluation uses IoU, precision, recall, and average precision metrics, with non-maximum suppression to remove duplicate detections.

10. Understanding the trade-offs between accuracy, speed, and robustness enables selection of appropriate detection methods for specific applications and constraints.
