# Feature Extraction and Descriptors

## Table of Contents

1. [Introduction](#introduction)
2. [Corner Detection](#corner-detection)
3. [Scale-Invariant Feature Transform (SIFT)](#scale-invariant-feature-transform-sift)
4. [Speeded Up Robust Features (SURF)](#speeded-up-robust-features-surf)
5. [Oriented FAST and Rotated BRIEF (ORB)](#oriented-fast-and-rotated-brief-orb)
6. [Histogram of Oriented Gradients (HOG)](#histogram-of-oriented-gradients-hog)
7. [Local Binary Patterns (LBP)](#local-binary-patterns-lbp)
8. [Feature Matching](#feature-matching)
9. [Descriptor Comparison and Evaluation](#descriptor-comparison-and-evaluation)
10. [Key Takeaways](#key-takeaways)

## Introduction

Feature extraction is a fundamental task in computer vision that identifies distinctive points, regions, or patterns in images. These features serve as robust representations that can be matched across different images, enabling applications such as object recognition, image stitching, tracking, and 3D reconstruction.

A good feature should possess several properties:
- **Repeatability**: Detectable in multiple views of the same scene
- **Distinctiveness**: Uniquely identifiable among different features
- **Robustness**: Invariant to transformations (rotation, scale, illumination)
- **Efficiency**: Computable in reasonable time

Feature descriptors encode the local appearance around detected features, creating compact representations suitable for matching and recognition tasks.

## Corner Detection

Corners are points where intensity changes significantly in multiple directions, making them highly distinctive and repeatable features.

### Harris Corner Detector

The Harris corner detector measures the local autocorrelation of image gradients. For a window $W$ around point $(x, y)$, the structure tensor is:

$$M = \sum_{(u, v) \in W} w(u, v) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

where $I_x$ and $I_y$ are image gradients, and $w(u, v)$ is a Gaussian weighting function.

The corner response function is:

$$R = \det(M) - k \cdot \text{trace}(M)^2$$

where $k$ is typically 0.04-0.06. Corners are detected where $R$ exceeds a threshold and is a local maximum.

The eigenvalues $\lambda_1, \lambda_2$ of $M$ characterize the local structure:
- **Flat region**: $\lambda_1 \approx 0, \lambda_2 \approx 0$
- **Edge**: One eigenvalue large, one small
- **Corner**: Both eigenvalues large

### Shi-Tomasi Corner Detector

The Shi-Tomasi detector (also called Good Features to Track) uses a simpler criterion:

$$R = \min(\lambda_1, \lambda_2)$$

A point is considered a corner if $R > \tau$ for threshold $\tau$. This method is more stable than Harris for tracking applications.

### FAST Corner Detector

FAST (Features from Accelerated Segment Test) is a high-speed corner detector. A pixel $p$ is a corner if at least $n$ contiguous pixels in a 16-pixel circle around $p$ are all brighter or darker than $p$ by threshold $t$:

$$\text{FAST}(p) = \begin{cases}
\text{corner} & \text{if } \exists \text{ contiguous set } S \text{ with } |S| \geq n \\
& \text{and } \forall q \in S: |I(q) - I(p)| > t \\
\text{not corner} & \text{otherwise}
\end{cases}$$

FAST is extremely fast but not scale-invariant. The ORB detector extends FAST with scale and rotation invariance.

## Scale-Invariant Feature Transform (SIFT)

SIFT is one of the most influential feature detectors and descriptors, providing scale, rotation, and illumination invariance.

### Scale Space Construction

SIFT builds a scale space using Gaussian pyramids:

$$L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)$$

where $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$ is the Gaussian kernel.

Difference of Gaussians (DoG) approximates the scale-normalized Laplacian:

$$D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)$$

where $k$ is a constant factor (typically $\sqrt{2}$).

### Keypoint Detection

Keypoints are detected as local extrema in the DoG scale space. Each pixel is compared to its 26 neighbors (8 in the same scale, 9 in the scale above, 9 in the scale below). Extrema are refined using Taylor expansion:

$$D(\mathbf{x}) = D + \frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} + \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x}$$

The offset $\hat{\mathbf{x}}$ is found by setting the derivative to zero:

$$\hat{\mathbf{x}} = -\frac{\partial^2 D^{-1}}{\partial \mathbf{x}^2} \frac{\partial D}{\partial \mathbf{x}}$$

Low-contrast and edge responses are rejected to improve stability.

### Orientation Assignment

For each keypoint, dominant orientations are computed from local gradient histograms:

$$m(x, y) = \sqrt{(L(x+1, y) - L(x-1, y))^2 + (L(x, y+1) - L(x, y-1))^2}$$

$$\theta(x, y) = \arctan\left(\frac{L(x, y+1) - L(x, y-1)}{L(x+1, y) - L(x-1, y)}\right)$$

A 36-bin histogram of orientations (weighted by gradient magnitude and Gaussian window) identifies dominant directions. Peaks above 80% of the maximum create additional keypoints with different orientations.

### SIFT Descriptor

The SIFT descriptor is a 128-dimensional vector computed from gradient histograms in a 4×4 grid around the keypoint:

1. Rotate the patch to canonical orientation
2. Divide 16×16 region into 4×4 cells
3. Compute 8-bin orientation histogram for each cell
4. Concatenate 16 histograms → 128 dimensions
5. Normalize to unit length for illumination invariance
6. Threshold values > 0.2 and renormalize

The descriptor captures local gradient distribution, providing robustness to affine transformations and illumination changes.

## Speeded Up Robust Features (SURF)

SURF accelerates SIFT using integral images and approximates the Hessian matrix for blob detection.

### Integral Images

An integral image $I_\Sigma$ enables fast box filter computation:

$$I_\Sigma(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)$$

The sum of any rectangular region can be computed in constant time:

$$\sum_{x_1 \leq x \leq x_2, y_1 \leq y \leq y_2} I(x, y) = I_\Sigma(x_2, y_2) - I_\Sigma(x_1-1, y_2) - I_\Sigma(x_2, y_1-1) + I_\Sigma(x_1-1, y_1-1)$$

### Hessian-Based Detection

SURF uses the determinant of the Hessian matrix for blob detection:

$$H(x, \sigma) = \begin{bmatrix} L_{xx}(x, \sigma) & L_{xy}(x, \sigma) \\ L_{xy}(x, \sigma) & L_{yy}(x, \sigma) \end{bmatrix}$$

$$\det(H) = L_{xx} L_{yy} - (0.9 L_{xy})^2$$

where $L_{xx}$, $L_{yy}$, $L_{xy}$ are second-order derivatives approximated using box filters. The 0.9 factor compensates for the box filter approximation.

### SURF Descriptor

The SURF descriptor uses Haar wavelet responses:

1. Compute Haar wavelet responses in $x$ and $y$ directions for a 20×20 region
2. Weight responses with Gaussian centered at keypoint
3. Sum responses over 4×4 subregions to get $(\sum dx, \sum dy, \sum |dx|, \sum |dy|)$
4. Concatenate 16 subregions → 64 dimensions

SURF is 3-7 times faster than SIFT while maintaining similar performance.

## Oriented FAST and Rotated BRIEF (ORB)

ORB combines FAST keypoint detection with a rotation-aware BRIEF descriptor, providing a fast alternative to SIFT/SURF.

### FAST Keypoints with Orientation

ORB uses FAST for keypoint detection, then assigns orientation using intensity centroid:

$$m_{pq} = \sum_{x, y} x^p y^q I(x, y)$$

$$\theta = \arctan\left(\frac{m_{01}}{m_{10}}\right)$$

The orientation is computed from the intensity-weighted centroid relative to the keypoint center.

### rBRIEF Descriptor

BRIEF (Binary Robust Independent Elementary Features) creates binary descriptors by comparing pixel intensities:

$$\text{BRIEF}(p) = \sum_{1 \leq i \leq n} 2^{i-1} \tau(p; x_i, y_i)$$

where $\tau(p; x, y) = \begin{cases} 1 & \text{if } I(p+x) < I(p+y) \\ 0 & \text{otherwise} \end{cases}$

ORB uses steered BRIEF (rBRIEF) that rotates the test pattern according to keypoint orientation, providing rotation invariance. ORB also learns optimal test pairs to maximize variance and minimize correlation.

The final ORB descriptor is a 256-bit binary string, enabling fast matching using Hamming distance.

## Histogram of Oriented Gradients (HOG)

HOG captures local shape information through gradient orientation histograms, originally designed for pedestrian detection.

### HOG Computation

1. **Gradient Computation**: Compute gradients $G_x$ and $G_y$ using Sobel or central differences:
   $$G_x = I(x+1, y) - I(x-1, y)$$
   $$G_y = I(x, y+1) - I(x, y-1)$$
   $$m(x, y) = \sqrt{G_x^2 + G_y^2}$$
   $$\theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)$$

2. **Cell Histograms**: Divide image into cells (e.g., 8×8 pixels). For each cell, create orientation histogram with $n$ bins (typically 9 bins for 0-180°):
   $$H[k] = \sum_{(x, y) \in \text{cell}} m(x, y) \cdot w(\theta(x, y), k)$$
   where $w$ distributes gradient magnitude to adjacent bins using bilinear interpolation.

3. **Block Normalization**: Group cells into blocks (e.g., 2×2 cells). Normalize histograms within each block using L2-norm:
   $$v' = \frac{v}{\sqrt{\|v\|^2 + \epsilon^2}}$$
   where $\epsilon$ prevents division by zero.

4. **Descriptor Concatenation**: Concatenate normalized histograms from all blocks to form the final descriptor.

### HOG Variants

- **Rectangular HOG (R-HOG)**: Standard rectangular blocks
- **Circular HOG (C-HOG)**: Circular sampling regions
- **Opponent HOG**: Separate histograms for different color channels

## Local Binary Patterns (LBP)

LBP encodes local texture information by comparing each pixel with its neighbors.

### Basic LBP

For a pixel $p_c$ with neighbors $p_i$ at radius $R$:

$$\text{LBP}_{P,R} = \sum_{i=0}^{P-1} s(p_i - p_c) \cdot 2^i$$

where $s(x) = \begin{cases} 1 & \text{if } x \geq 0 \\ 0 & \text{if } x < 0 \end{cases}$ and $P$ is the number of neighbors.

### Uniform LBP

Uniform patterns have at most two bitwise transitions (0→1 or 1→0). These patterns represent fundamental texture primitives (edges, corners, spots). Uniform LBP reduces the number of bins from $2^P$ to $P(P-1) + 3$.

### Rotation-Invariant LBP

Rotation invariance is achieved by circularly shifting the binary code to its minimum value:

$$\text{LBP}_{P,R}^{ri} = \min\{\text{ROR}(\text{LBP}_{P,R}, i) : i = 0, 1, \ldots, P-1\}$$

where ROR is the circular bitwise right shift operation.

### LBP Histograms

LBP codes are aggregated into histograms over image regions:

$$H[k] = \sum_{x, y} \delta(\text{LBP}(x, y), k)$$

where $\delta$ is the Kronecker delta. Histograms can be computed at multiple scales and concatenated.

## Feature Matching

Feature matching establishes correspondences between features in different images.

### Distance Metrics

**Euclidean Distance** (for real-valued descriptors like SIFT):
$$d(\mathbf{f}_1, \mathbf{f}_2) = \|\mathbf{f}_1 - \mathbf{f}_2\|_2$$

**Hamming Distance** (for binary descriptors like ORB):
$$d_H(\mathbf{b}_1, \mathbf{b}_2) = \sum_i \mathbf{b}_1[i] \oplus \mathbf{b}_2[i]$$

**Cosine Similarity**:
$$s(\mathbf{f}_1, \mathbf{f}_2) = \frac{\mathbf{f}_1 \cdot \mathbf{f}_2}{\|\mathbf{f}_1\| \|\mathbf{f}_2\|}$$

### Matching Strategies

**Nearest Neighbor**: Match each feature to its closest descriptor in the other image.

**Ratio Test** (Lowe's ratio test for SIFT): Accept matches where:
$$\frac{d_1}{d_2} < \tau$$

where $d_1$ and $d_2$ are distances to the nearest and second-nearest neighbors. Typical $\tau = 0.7-0.8$.

**Cross-Check**: A feature $f_1$ matches $f_2$ only if $f_2$ also matches $f_1$ as its nearest neighbor.

### Outlier Rejection

**RANSAC** (Random Sample Consensus) robustly estimates geometric transformations while rejecting outliers:

1. Randomly sample minimal set of correspondences
2. Estimate transformation model
3. Count inliers (matches consistent with model)
4. Repeat and keep model with most inliers

For homography estimation, 4 correspondences are needed. The model is:
$$\mathbf{x}' = H\mathbf{x}$$

where $H$ is a 3×3 homography matrix.

## Descriptor Comparison and Evaluation

### Performance Metrics

**Repeatability**: Percentage of features detected in both images that correspond to the same 3D point:
$$\text{Repeatability} = \frac{\text{Number of correspondences}}{\text{Number of detected features}}$$

**Precision**: Ratio of correct matches to total matches:
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$

**Recall**: Ratio of correct matches to total possible matches:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

### Invariance Properties

Different descriptors provide varying levels of invariance:

| Descriptor | Scale | Rotation | Illumination | Affine |
|------------|-------|----------|--------------|--------|
| Harris     | No    | Partial  | Partial      | No     |
| SIFT       | Yes   | Yes      | Yes          | Partial|
| SURF       | Yes   | Yes      | Yes          | Partial|
| ORB        | Partial| Yes     | Partial      | No     |
| HOG        | Partial| Partial | Yes          | No     |
| LBP        | No    | Partial  | Yes          | No     |

### Computational Complexity

- **SIFT**: $O(n \log n)$ for $n$ keypoints, ~100ms per image
- **SURF**: $O(n)$, ~30ms per image
- **ORB**: $O(n)$, ~10ms per image
- **HOG**: $O(MN)$ for $M \times N$ image, ~50ms per image
- **LBP**: $O(MN)$, ~5ms per image

## Key Takeaways

1. Corner detectors (Harris, Shi-Tomasi, FAST) identify distinctive points where intensity changes significantly in multiple directions.

2. SIFT provides scale and rotation invariance through scale-space analysis and orientation assignment, creating robust 128-dimensional descriptors.

3. SURF accelerates SIFT using integral images and box filters while maintaining similar performance characteristics.

4. ORB combines FAST detection with rotation-aware BRIEF descriptors, offering a fast binary alternative suitable for real-time applications.

5. HOG captures shape information through gradient orientation histograms, effective for object detection tasks.

6. LBP encodes local texture patterns through binary comparisons, providing efficient texture representation.

7. Feature matching requires appropriate distance metrics (Euclidean for real-valued, Hamming for binary descriptors).

8. Ratio tests and RANSAC are essential for robust matching, rejecting outliers and establishing reliable correspondences.

9. Descriptor selection depends on application requirements: SIFT/SURF for maximum robustness, ORB for speed, HOG for shape-based detection.

10. Understanding invariance properties and computational trade-offs enables appropriate feature extraction method selection for specific computer vision tasks.
