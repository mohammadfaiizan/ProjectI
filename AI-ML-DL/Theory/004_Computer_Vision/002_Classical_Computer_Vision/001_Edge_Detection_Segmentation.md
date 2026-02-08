# Edge Detection and Segmentation

## Table of Contents

1. [Introduction](#introduction)
2. [Canny Edge Detector](#canny-edge-detector)
3. [Region Growing](#region-growing)
4. [Watershed Algorithm](#watershed-algorithm)
5. [Active Contours and Snakes](#active-contours-and-snakes)
6. [Graph Cuts](#graph-cuts)
7. [Mean Shift Segmentation](#mean-shift-segmentation)
8. [Segmentation Evaluation](#segmentation-evaluation)
9. [Hybrid Approaches](#hybrid-approaches)
10. [Key Takeaways](#key-takeaways)

## Introduction

Edge detection and image segmentation are fundamental tasks in computer vision that partition images into meaningful regions. Edge detection identifies boundaries between regions of different intensity or texture, while segmentation groups pixels into coherent regions based on similarity criteria.

Segmentation serves as a preprocessing step for higher-level vision tasks such as object recognition, tracking, and scene understanding. The choice of segmentation method depends on the application requirements, image characteristics, and computational constraints. No single method works optimally for all scenarios, making it essential to understand the strengths and limitations of different approaches.

## Canny Edge Detector

The Canny edge detector is one of the most widely used edge detection algorithms, designed to optimize three criteria: good detection, good localization, and single response.

### Canny Algorithm Steps

**Step 1: Gaussian Smoothing**
Reduce noise by convolving the image with a Gaussian kernel:
$$I_s = G_\sigma * I$$

where $G_\sigma$ is a Gaussian with standard deviation $\sigma$.

**Step 2: Gradient Computation**
Compute gradient magnitude and direction:
$$G_x = \frac{\partial I_s}{\partial x}, \quad G_y = \frac{\partial I_s}{\partial y}$$
$$|\nabla I| = \sqrt{G_x^2 + G_y^2}$$
$$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

**Step 3: Non-Maximum Suppression**
Thin edges by keeping only local maxima in the gradient direction. For each pixel, compare its gradient magnitude with neighbors along the gradient direction, keeping only if it's the maximum.

**Step 4: Double Thresholding**
Classify pixels into three categories:
- **Strong edges**: $|\nabla I| > T_{high}$
- **Weak edges**: $T_{low} < |\nabla I| \leq T_{high}$
- **Non-edges**: $|\nabla I| \leq T_{low}$

**Step 5: Edge Tracking by Hysteresis**
Connect weak edge pixels to strong edge pixels. Weak pixels are kept only if they're connected to strong pixels through a chain of weak pixels.

### Parameter Selection

The Canny detector has three main parameters:
- **$\sigma$**: Controls smoothing (typically 1-3 pixels)
- **$T_{high}$**: High threshold (typically 0.7-0.9 times maximum gradient)
- **$T_{low}$**: Low threshold (typically 0.4-0.5 times $T_{high}$)

The ratio $T_{high}/T_{low}$ is typically 2:1 or 3:1.

### Canny Variants

**Adaptive Canny**: Thresholds computed locally based on image statistics
**Multi-scale Canny**: Applied at multiple scales, edges combined
**Color Canny**: Extended to color images using gradient in color space

## Region Growing

Region growing is a simple segmentation technique that starts from seed points and grows regions by adding neighboring pixels that satisfy similarity criteria.

### Basic Algorithm

1. **Seed Selection**: Choose initial seed points (manually or automatically)
2. **Growth Criteria**: Define similarity measure (intensity difference, color distance, texture)
3. **Region Expansion**: Iteratively add pixels to regions if they satisfy criteria
4. **Termination**: Stop when no more pixels can be added

### Similarity Measures

**Intensity-based**:
$$|I(x, y) - \mu_R| < T$$

where $\mu_R$ is the mean intensity of region $R$ and $T$ is a threshold.

**Color-based**:
$$\|\mathbf{c}(x, y) - \mathbf{c}_R\| < T$$

where $\mathbf{c}$ is the color vector and $\mathbf{c}_R$ is the mean color of region $R$.

**Statistical**:
A pixel is added if it's within $k$ standard deviations:
$$|I(x, y) - \mu_R| < k\sigma_R$$

### Seed Selection Strategies

- **Manual**: User selects seed points
- **Automatic**: Use local minima/maxima, corner detectors, or regular grid
- **Multi-seed**: Multiple seeds per region for robustness

### Region Growing Variants

**Split-and-Merge**: Combine region growing with region splitting
1. Start with entire image as one region
2. Split regions that don't satisfy homogeneity
3. Merge adjacent similar regions

**Seeded Region Growing**: Uses explicit seeds and grows until boundaries are reached

**Unseeded Region Growing**: Automatically finds seeds based on local properties

## Watershed Algorithm

The watershed algorithm treats the image as a topographic surface, where intensity represents elevation. Segmentation is performed by flooding from local minima.

### Mathematical Formulation

For grayscale image $I$, the watershed transform identifies catchment basins:

$$WS(I) = \{C_1, C_2, \ldots, C_n\}$$

where each $C_i$ is a catchment basin corresponding to a local minimum $m_i$.

### Flooding Process

1. **Find Minima**: Identify all local minima
2. **Initialize**: Create a label for each minimum
3. **Flooding**: Gradually increase water level, flooding basins
4. **Merging**: When water from different basins meets, create watershed line

### Marker-Based Watershed

To avoid over-segmentation, use markers (seeds):

1. **Marker Extraction**: Identify object and background markers
2. **Modify Image**: Set non-marker pixels to maximum intensity
3. **Apply Watershed**: Watershed on modified image

Markers can be obtained from:
- User input
- Morphological operations (erosion, opening)
- Distance transform
- Other segmentation methods

### Distance Transform

The distance transform computes distance from each pixel to the nearest boundary:

$$DT(x, y) = \min_{(i, j) \in B} d((x, y), (i, j))$$

where $B$ is the set of boundary pixels and $d$ is a distance metric (Euclidean, Manhattan, chessboard).

Watershed on the distance transform separates touching objects.

## Active Contours and Snakes

Active contours (snakes) are deformable curves that evolve to fit object boundaries by minimizing an energy functional.

### Snake Energy Functional

The total energy of a snake $v(s) = (x(s), y(s))$ is:

$$E_{snake} = \int_0^1 E_{int}(v(s)) + E_{image}(v(s)) + E_{ext}(v(s)) ds$$

where:
- **Internal energy** $E_{int}$: Controls smoothness and elasticity
- **Image energy** $E_{image}$: Attracts snake to image features
- **External energy** $E_{ext}$: User-defined constraints

### Internal Energy

$$E_{int} = \alpha(s) \left|\frac{\partial v}{\partial s}\right|^2 + \beta(s) \left|\frac{\partial^2 v}{\partial s^2}\right|^2$$

- First term: Elasticity (penalizes stretching)
- Second term: Stiffness (penalizes bending)

Parameters $\alpha(s)$ and $\beta(s)$ control the relative importance.

### Image Energy

Common choices for image energy:

**Gradient-based**:
$$E_{image} = -|\nabla I(v(s))|^2$$

**Gradient magnitude of smoothed image**:
$$E_{image} = -|\nabla G_\sigma * I(v(s))|^2$$

**Line/Edge functional**:
$$E_{image} = -I(v(s)) \quad \text{(for bright lines)}$$
$$E_{image} = I(v(s)) \quad \text{(for dark lines)}$$

### Energy Minimization

The snake evolves to minimize energy using gradient descent:

$$\frac{\partial v}{\partial t} = \frac{\partial}{\partial s}\left(\alpha \frac{\partial v}{\partial s}\right) - \frac{\partial^2}{\partial s^2}\left(\beta \frac{\partial^2 v}{\partial s^2}\right) - \nabla E_{image}$$

Discretizing the snake as $N$ points, this becomes a system of equations solved iteratively.

### Active Contour Variants

**Geodesic Active Contours**: Level set formulation, handles topology changes
**Chan-Vese Model**: Region-based, doesn't require edges
**Gradient Vector Flow**: External force field improving convergence

## Graph Cuts

Graph cuts formulate segmentation as a minimum cut problem in a graph, enabling globally optimal solutions for certain energy functions.

### Graph Construction

Construct graph $G = (V, E)$ where:
- **Nodes $V$**: Image pixels plus source $s$ and sink $t$
- **Edges $E$**: 
  - **n-links**: Connect neighboring pixels
  - **t-links**: Connect pixels to source and sink

### Energy Function

The energy function has two terms:

$$E(f) = \sum_{p \in P} D_p(f_p) + \lambda \sum_{(p,q) \in N} V_{p,q}(f_p, f_q)$$

where:
- **Data term** $D_p$: Cost of assigning label $f_p$ to pixel $p$
- **Smoothness term** $V_{p,q}$: Cost of assigning different labels to neighbors
- **$\lambda$**: Weight balancing the terms

### Minimum Cut

A cut $C$ partitions nodes into sets $S$ (containing $s$) and $T$ (containing $t$). The cut cost is:

$$|C| = \sum_{(u,v) \in C} w(u,v)$$

where $w(u,v)$ is the edge weight.

The minimum cut corresponds to the optimal segmentation and can be found using max-flow algorithms (e.g., Ford-Fulkerson, push-relabel).

### Edge Weights

**t-link weights** (source/sink to pixels):
- $w(s, p) = D_p(\text{foreground})$: Cost of pixel being foreground
- $w(p, t) = D_p(\text{background})$: Cost of pixel being background

**n-link weights** (between pixels):
$$w(p, q) = \lambda \cdot V_{p,q}(f_p, f_q)$$

Common smoothness term:
$$V_{p,q} = \begin{cases}
0 & \text{if } f_p = f_q \\
\exp\left(-\frac{(I_p - I_q)^2}{2\sigma^2}\right) & \text{if } f_p \neq f_q
\end{cases}$$

### GrabCut Algorithm

GrabCut extends graph cuts with iterative refinement:

1. User provides bounding box (or trimap)
2. Initialize foreground/background models (Gaussian Mixture Models)
3. Iterate:
   - Assign pixels using graph cut
   - Update GMM parameters
   - Repeat until convergence

GrabCut provides high-quality segmentation with minimal user interaction.

## Mean Shift Segmentation

Mean shift is a non-parametric clustering technique that finds modes in the feature space, naturally handling arbitrary cluster shapes.

### Mean Shift Algorithm

For a point $\mathbf{x}$, the mean shift vector is:

$$m_h(\mathbf{x}) = \frac{\sum_{i=1}^{n} K_h(\mathbf{x}_i - \mathbf{x}) \mathbf{x}_i}{\sum_{i=1}^{n} K_h(\mathbf{x}_i - \mathbf{x})} - \mathbf{x}$$

where $K_h$ is a kernel function with bandwidth $h$.

The algorithm iteratively moves points toward modes:
$$\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} + m_h(\mathbf{x}^{(t)})$$

### Kernel Function

Common kernels:

**Epanechnikov kernel**:
$$K_E(\mathbf{x}) = \begin{cases}
c(1 - \|\mathbf{x}\|^2) & \text{if } \|\mathbf{x}\| \leq 1 \\
0 & \text{otherwise}
\end{cases}$$

**Gaussian kernel**:
$$K_G(\mathbf{x}) = c \exp\left(-\frac{\|\mathbf{x}\|^2}{2h^2}\right)$$

### Mean Shift for Segmentation

**Feature Space**: Combine spatial $(x, y)$ and color $(L, a, b)$ coordinates:
$$\mathbf{f} = (x, y, L, a, b)$$

**Bandwidth**: Separate bandwidths for spatial ($h_s$) and color ($h_r$) domains.

**Algorithm**:
1. For each pixel, perform mean shift to find mode
2. Pixels converging to the same mode belong to the same segment
3. Merge segments with similar modes

### Advantages

- No assumption about number of clusters
- Handles arbitrary cluster shapes
- Robust to outliers
- Single parameter (bandwidth) to tune

### Disadvantages

- Computationally expensive
- Bandwidth selection is critical
- May over-segment or under-segment

## Segmentation Evaluation

Evaluating segmentation quality is challenging due to the subjective nature of "correct" segmentation.

### Supervised Metrics

Given ground truth segmentation $S_{GT}$ and result $S_R$:

**Pixel Accuracy**:
$$PA = \frac{\sum_i \delta(S_{GT}(i), S_R(i))}{N}$$

where $\delta$ is the Kronecker delta and $N$ is the number of pixels.

**Intersection over Union (IoU)**:
For each region $R$:
$$\text{IoU}(R) = \frac{|R_{GT} \cap R_R|}{|R_{GT} \cup R_R|}$$

**Rand Index**:
Measures similarity of pixel pairs:
$$RI = \frac{TP + TN}{TP + TN + FP + FN}$$

where:
- $TP$: Pairs in same region in both
- $TN$: Pairs in different regions in both
- $FP$: Pairs together in result but not ground truth
- $FN$: Pairs together in ground truth but not result

### Unsupervised Metrics

**Intra-region uniformity**: Regions should be homogeneous
**Inter-region disparity**: Adjacent regions should be different
**Compactness**: Regions should be compact (low perimeter-to-area ratio)

**Normalized cuts value**: Measures how well segmentation separates the graph

## Hybrid Approaches

Modern segmentation often combines multiple techniques:

### Hierarchical Segmentation

Build segmentation hierarchy:
1. Over-segment using simple method (e.g., watershed)
2. Build region adjacency graph
3. Merge regions based on similarity
4. Create hierarchy at multiple scales

### Multi-scale Approaches

Apply segmentation at multiple scales and combine results:
- Fine scale: Captures details
- Coarse scale: Captures large structures
- Combine using voting or hierarchical merging

### Learning-Based Segmentation

Train classifiers to predict region boundaries:
- Features: Color, texture, gradients, context
- Labels: Boundary vs non-boundary
- Methods: Random forests, CNNs, CRFs

### Conditional Random Fields (CRFs)

CRFs model pixel labels considering neighborhood context:

$$P(\mathbf{y}|\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_i \phi_i(y_i, \mathbf{x}) + \sum_{i,j} \psi_{ij}(y_i, y_j, \mathbf{x})\right)$$

where:
- $\phi_i$: Unary potential (data term)
- $\psi_{ij}$: Pairwise potential (smoothness term)
- $Z$: Normalization constant

CRFs can be integrated with deep learning for end-to-end trainable segmentation.

## Key Takeaways

1. The Canny edge detector optimizes detection, localization, and single response through multi-stage processing including non-maximum suppression and hysteresis thresholding.

2. Region growing segments images by iteratively expanding regions from seed points based on similarity criteria, requiring careful seed selection and threshold tuning.

3. The watershed algorithm treats images as topographic surfaces, flooding from minima to create segmentation, with marker-based variants reducing over-segmentation.

4. Active contours (snakes) evolve curves to fit boundaries by minimizing energy functionals combining internal smoothness constraints and image-based attraction forces.

5. Graph cuts formulate segmentation as minimum cut problems, enabling globally optimal solutions for certain energy functions through efficient max-flow algorithms.

6. Mean shift segmentation finds modes in feature space through iterative shifting, naturally handling arbitrary cluster shapes without assuming cluster count.

7. Segmentation evaluation requires both supervised metrics (pixel accuracy, IoU) and unsupervised metrics (uniformity, compactness) to assess quality.

8. Hybrid approaches combining multiple techniques (hierarchical, multi-scale, learning-based) often outperform single-method approaches.

9. The choice of segmentation method depends on image characteristics, application requirements, and available computational resources.

10. Understanding the mathematical foundations and trade-offs of different segmentation methods enables selection of appropriate techniques for specific computer vision tasks.
