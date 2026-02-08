# Feature Matching and Stereo Vision

## Table of Contents

1. [Introduction](#introduction)
2. [Feature Correspondence](#feature-correspondence)
3. [RANSAC Algorithm](#ransac-algorithm)
4. [Epipolar Geometry](#epipolar-geometry)
5. [Fundamental and Essential Matrices](#fundamental-and-essential-matrices)
6. [Stereo Reconstruction](#stereo-reconstruction)
7. [Disparity Maps](#disparity-maps)
8. [Stereo Matching Algorithms](#stereo-matching-algorithms)
9. [Multi-View Stereo](#multi-view-stereo)
10. [Key Takeaways](#key-takeaways)

## Introduction

Stereo vision enables 3D reconstruction from multiple 2D images by exploiting geometric relationships between different viewpoints. The fundamental principle is triangulation: corresponding points in multiple views are back-projected to find their 3D location at the intersection of projection rays.

Feature matching establishes correspondences between images, identifying the same physical points across different views. These correspondences, combined with knowledge of camera geometry, enable computation of 3D structure. Epipolar geometry provides constraints that simplify correspondence search and enable robust estimation of camera relationships.

## Feature Correspondence

Feature correspondence identifies the same physical point across multiple images, forming the foundation for stereo vision and structure-from-motion.

### Correspondence Problem

Given feature $f_1$ in image $I_1$, find corresponding feature $f_2$ in image $I_2$ such that $f_1$ and $f_2$ represent the same 3D point.

Challenges:
- **Occlusion**: Points visible in one view may be occluded in another
- **Viewpoint changes**: Appearance changes with viewing angle
- **Illumination**: Lighting differences affect appearance
- **Scale**: Objects appear at different scales
- **Repetitive patterns**: Ambiguous matches in textureless regions

### Matching Strategies

**Descriptor Matching**: Compare feature descriptors (SIFT, SURF, ORB) using distance metrics:
- Euclidean distance for real-valued descriptors
- Hamming distance for binary descriptors

**Template Matching**: Match image patches around features using correlation or SSD.

**Geometric Constraints**: Use epipolar geometry to constrain search to epipolar lines.

### Ratio Test

Lowe's ratio test improves matching robustness:

Accept match if:
$$\frac{d_1}{d_2} < \tau$$

where $d_1$ is the distance to the nearest neighbor and $d_2$ is the distance to the second-nearest neighbor. Typical threshold $\tau = 0.7-0.8$.

The ratio test rejects ambiguous matches where multiple features are equally similar.

### Cross-Check

Cross-check ensures bidirectional consistency:
- Feature $f_1$ matches $f_2$ if $f_2$ is the nearest neighbor of $f_1$
- Feature $f_2$ matches $f_1$ if $f_1$ is the nearest neighbor of $f_2$
- Keep match only if both conditions hold

### Symmetric Matching

Symmetric matching combines forward and backward matching:
1. Match $I_1 \rightarrow I_2$ (forward)
2. Match $I_2 \rightarrow I_1$ (backward)
3. Keep matches that are consistent in both directions

## RANSAC Algorithm

RANSAC (Random Sample Consensus) robustly estimates model parameters in the presence of outliers, essential for feature matching where many incorrect correspondences exist.

### RANSAC Algorithm

Given data points $\mathcal{D}$ and model $M$ with parameters $\theta$:

1. **Sample**: Randomly select minimal subset $S$ needed to estimate $\theta$
2. **Estimate**: Compute model parameters $\theta$ from $S$
3. **Consensus**: Count inliers (points consistent with model within threshold $\tau$)
4. **Repeat**: Steps 1-3 for $N$ iterations
5. **Select**: Choose model with most inliers
6. **Refine**: Re-estimate using all inliers (least squares)

### Number of Iterations

The number of iterations $N$ needed to find a good model with probability $p$:

$$N = \frac{\log(1-p)}{\log(1-(1-\epsilon)^s)}$$

where:
- $\epsilon$: Outlier ratio
- $s$: Minimal sample size
- $p$: Desired success probability (typically 0.99)

For homography estimation ($s=4$) with 50% outliers:
$$N = \frac{\log(0.01)}{\log(1-0.5^4)} \approx 72$$

### Inlier Threshold

The inlier threshold $\tau$ depends on the application:
- **Homography**: Reprojection error in pixels (typically 1-3 pixels)
- **Fundamental matrix**: Distance to epipolar line (typically 1-2 pixels)
- **Essential matrix**: Angular error or reprojection error

### Adaptive RANSAC

Adaptive RANSAC updates the number of iterations based on current best model:
1. Start with initial estimate of outlier ratio
2. Update estimate after each iteration: $\epsilon = 1 - \frac{\text{inliers}}{\text{total}}$
3. Recompute $N$ using updated $\epsilon$
4. Stop early if sufficient inliers found

### RANSAC Variants

**MSAC (M-estimator SAmple Consensus)**: Uses robust cost function instead of binary inlier/outlier
**MLESAC (Maximum Likelihood Estimation SAmple Consensus)**: Maximizes likelihood instead of inlier count
**PROSAC (PROgressive SAmple Consensus)**: Samples from most promising correspondences first

## Epipolar Geometry

Epipolar geometry describes the geometric relationship between two views of the same scene, providing constraints that simplify correspondence search.

### Epipolar Concepts

For a 3D point $\mathbf{X}$ viewed by two cameras:

- **Epipolar plane**: Plane containing $\mathbf{X}$ and both camera centers $C_1$ and $C_2$
- **Baseline**: Line joining camera centers $C_1C_2$
- **Epipolar line**: Intersection of epipolar plane with image plane
- **Epipole**: Intersection of baseline with image plane

### Epipolar Constraint

The epipolar constraint states that corresponding points must lie on corresponding epipolar lines:

For point $\mathbf{x}_1$ in image 1, its corresponding point $\mathbf{x}_2$ in image 2 must lie on the epipolar line $l_2$:
$$\mathbf{x}_2^T l_2 = 0$$

Similarly, $\mathbf{x}_1$ lies on epipolar line $l_1$ corresponding to $\mathbf{x}_2$.

### Epipolar Line Equation

The epipolar line in image 2 corresponding to point $\mathbf{x}_1$ in image 1:
$$l_2 = F \mathbf{x}_1$$

where $F$ is the fundamental matrix.

Similarly:
$$l_1 = F^T \mathbf{x}_2$$

### Benefits of Epipolar Geometry

1. **Constrained search**: Instead of searching entire image, search along epipolar line
2. **Outlier rejection**: Matches violating epipolar constraint are likely incorrect
3. **Rectification**: After stereo rectification, epipolar lines are horizontal, simplifying matching

## Fundamental and Essential Matrices

The fundamental and essential matrices encode epipolar geometry, enabling computation of 3D structure from correspondences.

### Fundamental Matrix

The fundamental matrix $F$ is a $3 \times 3$ matrix of rank 2 that relates corresponding points in two images:

$$\mathbf{x}_2^T F \mathbf{x}_1 = 0$$

for corresponding points $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$ in homogeneous coordinates.

### Properties of Fundamental Matrix

- **Rank 2**: $\det(F) = 0$
- **7 degrees of freedom**: 9 elements minus 1 scale factor minus 1 rank constraint
- **Epipolar lines**: $l_2 = F \mathbf{x}_1$, $l_1 = F^T \mathbf{x}_2$
- **Epipoles**: $F \mathbf{e}_1 = 0$, $F^T \mathbf{e}_2 = 0$

### Essential Matrix

The essential matrix $E$ relates camera coordinates in two views:

$$\mathbf{x}_2^T E \mathbf{x}_1 = 0$$

where $\mathbf{x}_1$ and $\mathbf{x}_2$ are in normalized camera coordinates.

Relationship to fundamental matrix:
$$F = K_2^{-T} E K_1^{-1}$$

where $K_1$ and $K_2$ are camera intrinsic matrices.

### Essential Matrix Decomposition

The essential matrix can be decomposed into rotation and translation:

$$E = [\mathbf{t}]_\times R$$

where:
- $R$: Rotation matrix (3 DOF)
- $\mathbf{t}$: Translation vector (3 DOF, up to scale)
- $[\mathbf{t}]_\times$: Skew-symmetric matrix for cross product

### Estimating Fundamental Matrix

**8-Point Algorithm**: Given 8 point correspondences, solve linear system:

For each correspondence $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$:
$$[x_2, y_2, 1] \begin{bmatrix} f_{11} & f_{12} & f_{13} \\ f_{21} & f_{22} & f_{23} \\ f_{31} & f_{32} & f_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ y_1 \\ 1 \end{bmatrix} = 0$$

Expanding:
$$x_2 x_1 f_{11} + x_2 y_1 f_{12} + x_2 f_{13} + y_2 x_1 f_{21} + y_2 y_1 f_{22} + y_2 f_{23} + x_1 f_{31} + y_1 f_{32} + f_{33} = 0$$

Stacking 8 equations gives $A \mathbf{f} = 0$, solved using SVD.

### Normalized 8-Point Algorithm

Normalization improves numerical stability:

1. **Normalize points**: Translate to centroid, scale so average distance is $\sqrt{2}$
2. **Compute $F$**: Apply 8-point algorithm
3. **Enforce rank 2**: Set smallest singular value to 0
4. **Denormalize**: Transform $F$ back to original coordinates

### Robust Estimation

RANSAC is used for robust fundamental matrix estimation:

1. Sample 8 correspondences
2. Estimate $F$ using 8-point algorithm
3. Count inliers (points with distance to epipolar line $< \tau$)
4. Repeat and keep $F$ with most inliers
5. Refine using all inliers

### Recovering Pose from Essential Matrix

From essential matrix $E$, recover $R$ and $\mathbf{t}$:

1. **SVD decomposition**: $E = U \Sigma V^T$
2. **Two solutions**: 
   - $R = U R_Z(\pm \pi/2) V^T$
   - $\mathbf{t} = U [0, 0, 1]^T$ or $\mathbf{t} = -U [0, 0, 1]^T$
3. **Disambiguation**: Choose solution with points in front of both cameras

## Stereo Reconstruction

Stereo reconstruction recovers 3D structure from two calibrated cameras using triangulation.

### Stereo Setup

**Parallel cameras**: Cameras with parallel optical axes, simplifying geometry:
- Epipolar lines are horizontal
- Disparity equals horizontal shift
- Depth inversely proportional to disparity

**General stereo**: Arbitrary camera positions, requires rectification.

### Triangulation

Given corresponding points $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$ and camera matrices $P_1$ and $P_2$, find 3D point $\mathbf{X}$:

From camera 1:
$$\lambda_1 \mathbf{x}_1 = P_1 \mathbf{X}$$

From camera 2:
$$\lambda_2 \mathbf{x}_2 = P_2 \mathbf{X}$$

This gives 4 equations in 3 unknowns, solved using least squares or SVD.

### Linear Triangulation

In homogeneous coordinates:
$$\begin{bmatrix} \mathbf{x}_1 \times P_1 \\ \mathbf{x}_2 \times P_2 \end{bmatrix} \mathbf{X} = 0$$

Solved using SVD: $\mathbf{X}$ is the right singular vector corresponding to the smallest singular value.

### Optimal Triangulation

Linear triangulation doesn't account for measurement noise. Optimal triangulation minimizes reprojection error:

$$\min_{\mathbf{X}} \|\mathbf{x}_1 - P_1 \mathbf{X}\|^2 + \|\mathbf{x}_2 - P_2 \mathbf{X}\|^2$$

subject to epipolar constraint $\mathbf{x}_2^T F \mathbf{x}_1 = 0$.

### Depth from Disparity

For parallel stereo cameras with baseline $b$ and focal length $f$:

$$Z = \frac{bf}{d}$$

where $d$ is the disparity (horizontal shift between corresponding points).

Disparity:
$$d = x_1 - x_2$$

## Disparity Maps

A disparity map assigns a disparity value to each pixel, encoding depth information.

### Disparity Definition

Disparity $d(x, y)$ at pixel $(x, y)$:
$$d(x, y) = x_1 - x_2$$

where $(x_1, y)$ and $(x_2, y)$ are corresponding points in left and right images.

### Disparity Range

Disparity is bounded:
- **Minimum disparity**: $d_{min}$ (far objects)
- **Maximum disparity**: $d_{max}$ (near objects)
- **Disparity range**: $[d_{min}, d_{max}]$

Typical range: 0 to 100+ pixels depending on baseline and scene depth.

### Disparity to Depth

For rectified stereo:
$$Z = \frac{bf}{d}$$

where:
- $b$: Baseline (distance between cameras)
- $f$: Focal length
- $d$: Disparity

### Disparity Map Properties

- **Smooth regions**: Constant disparity (planar surfaces)
- **Depth discontinuities**: Disparity jumps (object boundaries)
- **Occlusions**: No valid disparity (visible in one view only)
- **Textureless regions**: Ambiguous matches

## Stereo Matching Algorithms

Stereo matching algorithms compute disparity maps from stereo image pairs.

### Local Methods

**Block Matching**: For each pixel in left image, search along epipolar line in right image:

$$d(x, y) = \arg\min_d \sum_{(i,j) \in W} |I_L(x+i, y+j) - I_R(x+i-d, y+j)|$$

where $W$ is a window around $(x, y)$.

**Normalized Cross-Correlation**: More robust to illumination:
$$d(x, y) = \arg\max_d \frac{\sum I_L I_R}{\sqrt{\sum I_L^2 \sum I_R^2}}$$

### Global Methods

**Dynamic Programming**: Optimize along scanlines:
$$E(d) = \sum_{(x,y)} C(x, y, d(x,y)) + \lambda \sum_{(x,y),(x',y')} V(d(x,y), d(x',y'))$$

where:
- $C$: Matching cost
- $V$: Smoothness term
- $\lambda$: Regularization weight

**Graph Cuts**: Formulate as energy minimization:
$$E(d) = \sum_p D_p(d_p) + \sum_{p,q} V_{p,q}(d_p, d_q)$$

Solved using max-flow/min-cut algorithms.

**Belief Propagation**: Iterative message passing to minimize energy.

### Semi-Global Matching (SGM)

SGM combines efficiency of local methods with quality of global methods:

1. **Cost computation**: Compute matching cost for all pixels and disparities
2. **Path aggregation**: Aggregate costs along multiple paths (typically 8 or 16 directions)
3. **Disparity selection**: Choose disparity minimizing aggregated cost
4. **Sub-pixel refinement**: Refine to sub-pixel accuracy

Cost aggregation along direction $r$:
$$L_r(p, d) = C(p, d) + \min\begin{cases}
L_r(p-r, d) \\
L_r(p-r, d-1) + P_1 \\
L_r(p-r, d+1) + P_1 \\
\min_k L_r(p-r, k) + P_2
\end{cases} - \min_k L_r(p-r, k)$$

where $P_1$ and $P_2$ are penalty parameters.

### Post-Processing

**Left-Right Consistency**: Check consistency between left-to-right and right-to-left matching
**Sub-pixel Refinement**: Fit parabola or use interpolation for sub-pixel accuracy
**Filtering**: Remove outliers, fill holes using median or bilateral filtering

## Multi-View Stereo

Multi-view stereo uses more than two images to improve reconstruction quality and handle occlusions.

### Advantages

- **Better coverage**: More viewpoints reduce occlusions
- **Higher accuracy**: Multiple measurements improve precision
- **Robustness**: Outlier rejection through consensus

### Volumetric Methods

**Space Carving**: Start with volume, remove voxels inconsistent with images:
1. Initialize 3D volume
2. For each image, project voxels and check photo-consistency
3. Remove inconsistent voxels
4. Repeat until convergence

**Voxel Coloring**: Similar but uses color consistency.

### Patch-Based Methods

**PatchMatch Stereo**: Efficiently finds best patches:
1. Initialize random disparities
2. Propagate good matches to neighbors
3. Random search for improvements
4. Iterate until convergence

### Depth Map Fusion

Combine depth maps from multiple views:
1. Compute depth map for each view pair
2. Transform to common coordinate system
3. Fuse using median or weighted average
4. Generate final 3D model

## Key Takeaways

1. Feature correspondence identifies the same physical points across images, enabled by robust descriptors and geometric constraints.

2. RANSAC robustly estimates model parameters in the presence of outliers, essential for handling incorrect correspondences in feature matching.

3. Epipolar geometry constrains correspondence search to epipolar lines, dramatically reducing search space and enabling outlier rejection.

4. The fundamental matrix encodes epipolar geometry between uncalibrated cameras, while the essential matrix relates calibrated camera coordinates.

5. Stereo reconstruction recovers 3D structure through triangulation, computing 3D points from corresponding 2D points in multiple views.

6. Disparity maps encode depth information as horizontal shifts between corresponding points, with depth inversely proportional to disparity.

7. Stereo matching algorithms range from local block matching to global optimization methods, with SGM providing a good balance of quality and efficiency.

8. Multi-view stereo improves reconstruction by combining information from multiple viewpoints, reducing occlusions and improving accuracy.

9. Robust estimation techniques (RANSAC, robust cost functions) are essential for handling outliers and noise in real-world stereo vision applications.

10. Understanding epipolar geometry, correspondence matching, and triangulation enables 3D reconstruction from multiple 2D images, fundamental to many computer vision applications.
