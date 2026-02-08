# Image Transformations and Geometry

## Table of Contents

1. [Introduction](#introduction)
2. [Geometric Transformations](#geometric-transformations)
3. [Affine Transformations](#affine-transformations)
4. [Projective Transformations and Homography](#projective-transformations-and-homography)
5. [Image Warping and Interpolation](#image-warping-and-interpolation)
6. [Camera Models](#camera-models)
7. [Camera Calibration](#camera-calibration)
8. [Lens Distortion Correction](#lens-distortion-correction)
9. [Multi-View Geometry](#multi-view-geometry)
10. [Key Takeaways](#key-takeaways)

## Introduction

Geometric transformations are fundamental operations in computer vision that map points from one coordinate system to another. These transformations enable image registration, panorama stitching, 3D reconstruction, and camera calibration. Understanding geometric transformations is essential for applications requiring spatial alignment or coordinate system conversion.

Transformations can be classified by their degrees of freedom and preserved properties. Linear transformations preserve lines and parallelism, while projective transformations preserve collinearity but not parallelism. The choice of transformation model depends on the imaging geometry and application requirements.

## Geometric Transformations

A geometric transformation maps points from source coordinates $(x, y)$ to target coordinates $(x', y')$:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = T\left(\begin{bmatrix} x \\ y \end{bmatrix}\right)$$

where $T$ is the transformation function.

### Transformation Hierarchy

Transformations form a hierarchy based on degrees of freedom and preserved properties:

1. **Translation** (2 DOF): Preserves orientation and shape
2. **Rigid/Euclidean** (3 DOF): Preserves distances and angles
3. **Similarity** (4 DOF): Preserves angles and ratios of distances
4. **Affine** (6 DOF): Preserves parallelism and ratios of areas
5. **Projective** (8 DOF): Preserves collinearity and cross-ratios

### Homogeneous Coordinates

Homogeneous coordinates enable linear representation of transformations using matrix multiplication:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Cartesian coordinates are recovered by:
$$x_{cart} = \frac{x'}{w'}, \quad y_{cart} = \frac{y'}{w'}$$

Homogeneous coordinates allow translation to be expressed as matrix multiplication, unifying all transformations.

## Affine Transformations

Affine transformations preserve parallelism and can be decomposed into linear transformation and translation:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ d & e \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} c \\ f \end{bmatrix}$$

In homogeneous coordinates:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

### Affine Transformation Components

An affine transformation can be decomposed into:

1. **Translation**: $\mathbf{t} = [c, f]^T$
2. **Rotation**: $R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
3. **Scaling**: $S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$
4. **Shear**: $H = \begin{bmatrix} 1 & h_x \\ h_y & 1 \end{bmatrix}$

The general form: $A = R(\theta) \cdot S \cdot H + \mathbf{t}$

### Affine Invariants

Affine transformations preserve:
- **Parallelism**: Parallel lines remain parallel
- **Ratio of areas**: Area ratios are preserved
- **Ratio of lengths along parallel lines**
- **Centroids**: Centroids map to centroids

### Affine Parameter Estimation

Given point correspondences $\{(x_i, y_i) \leftrightarrow (x'_i, y'_i)\}$, the affine parameters are estimated by solving:

$$\begin{bmatrix} x'_1 \\ y'_1 \\ x'_2 \\ y'_2 \\ \vdots \end{bmatrix} = \begin{bmatrix} x_1 & y_1 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & x_1 & y_1 & 1 \\ x_2 & y_2 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & x_2 & y_2 & 1 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \end{bmatrix} \begin{bmatrix} a \\ b \\ c \\ d \\ e \\ f \end{bmatrix}$$

At least 3 point correspondences are needed (6 equations for 6 unknowns).

## Projective Transformations and Homography

Projective transformations (homographies) are the most general linear transformations in the plane, preserving collinearity and cross-ratios but not parallelism.

### Homography Matrix

A homography is represented by a 3×3 matrix $H$:

$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Since $H$ is defined up to scale (multiplying by a scalar doesn't change the transformation), it has 8 degrees of freedom.

### Homography Estimation

Given 4 point correspondences, the homography can be estimated. For each correspondence $(x, y) \leftrightarrow (x', y')$:

$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}$$
$$y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

Rearranging:
$$x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13}$$
$$y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}$$

This gives 2 linear equations per correspondence. With 4 correspondences (8 equations), $H$ can be solved using Direct Linear Transform (DLT).

### Direct Linear Transform (DLT)

The DLT algorithm solves for $H$ by constructing:

$$Ah = 0$$

where $h$ is the vectorized homography matrix (9 elements), and $A$ is constructed from point correspondences. The solution is the right singular vector corresponding to the smallest singular value of $A$.

### Normalized DLT

Normalization improves numerical stability:
1. Translate points so centroid is at origin
2. Scale so average distance from origin is $\sqrt{2}$
3. Apply DLT
4. Denormalize the result

### Robust Homography Estimation

RANSAC is commonly used for robust homography estimation:
1. Randomly sample 4 correspondences
2. Compute homography $H$
3. Count inliers (points with reprojection error $< \tau$)
4. Repeat and keep $H$ with most inliers
5. Refine using all inliers (least squares)

Reprojection error:
$$e_i = \left\| \mathbf{x}'_i - \frac{H\mathbf{x}_i}{w_i} \right\|$$

## Image Warping and Interpolation

Image warping applies a geometric transformation to an image, requiring interpolation to determine pixel values at non-integer coordinates.

### Forward vs Backward Warping

**Forward warping**: Map source pixels to target coordinates. Problems: holes and overlaps.

**Backward warping**: For each target pixel, find corresponding source location. Preferred approach.

For backward warping with transformation $T$:
$$(x_s, y_s) = T^{-1}(x_t, y_t)$$
$$I_t(x_t, y_t) = I_s(x_s, y_s)$$

### Interpolation Methods

**Nearest Neighbor**: Use value of closest pixel
$$I(x, y) = I(\text{round}(x), \text{round}(y))$$

Fast but produces aliasing artifacts.

**Bilinear Interpolation**: Weighted average of 4 nearest neighbors
$$I(x, y) = \sum_{i=0}^{1} \sum_{j=0}^{1} I(\lfloor x \rfloor + i, \lfloor y \rfloor + j) \cdot w(i, j)$$

where weights depend on fractional parts:
$$w(0, 0) = (1-\Delta x)(1-\Delta y)$$
$$w(1, 0) = \Delta x(1-\Delta y)$$
$$w(0, 1) = (1-\Delta x)\Delta y$$
$$w(1, 1) = \Delta x \Delta y$$

**Bicubic Interpolation**: Uses 16 neighbors with cubic interpolation kernel:
$$I(x, y) = \sum_{i=-1}^{2} \sum_{j=-1}^{2} I(\lfloor x \rfloor + i, \lfloor y \rfloor + j) \cdot B(i - \Delta x) \cdot B(j - \Delta y)$$

where $B$ is the cubic B-spline or Catmull-Rom kernel. Higher quality but more expensive.

## Camera Models

Camera models describe the projection of 3D points onto 2D image planes.

### Pinhole Camera Model

The pinhole camera is the simplest camera model. A 3D point $\mathbf{X} = [X, Y, Z]^T$ projects to image point $\mathbf{x} = [x, y]^T$:

$$x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}$$

where $f$ is the focal length.

### Camera Coordinate Systems

1. **World coordinates** $\mathbf{X}_w$: 3D scene coordinates
2. **Camera coordinates** $\mathbf{X}_c$: 3D coordinates relative to camera
3. **Image coordinates** $\mathbf{x}$: 2D coordinates on image plane
4. **Pixel coordinates** $\mathbf{u}$: Integer pixel indices

### Intrinsic Parameters

Intrinsic parameters describe internal camera geometry:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where:
- $f_x, f_y$: Focal lengths in pixels (may differ due to non-square pixels)
- $c_x, c_y$: Principal point (optical center in pixel coordinates)
- $s$: Skew parameter (usually 0 for modern cameras)

The projection equation:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{bmatrix}$$

### Extrinsic Parameters

Extrinsic parameters describe camera pose (position and orientation):

$$\mathbf{X}_c = R \mathbf{X}_w + \mathbf{t}$$

where $R$ is a 3×3 rotation matrix and $\mathbf{t}$ is a 3×1 translation vector.

### Complete Projection Model

Combining intrinsic and extrinsic parameters:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

where $\lambda = Z_c$ is the depth.

## Camera Calibration

Camera calibration estimates intrinsic and extrinsic parameters from known 3D-2D correspondences.

### Calibration Target

Common calibration targets:
- **Checkerboard**: Easy to detect corners automatically
- **Circular patterns**: More accurate center detection
- **Coded targets**: Unique identification of points

### Zhang's Method

Zhang's method uses multiple views of a planar calibration target:

1. Detect corners in each image
2. For each view, estimate homography $H$ mapping calibration target to image
3. Extract constraints on intrinsic parameters from homographies
4. Solve for $K$ using constraints
5. Estimate extrinsic parameters for each view
6. Refine all parameters using non-linear optimization

### Constraints from Homography

For homography $H = [h_1, h_2, h_3]$ mapping a plane to image:

$$h_1^T K^{-T} K^{-1} h_2 = 0$$
$$h_1^T K^{-T} K^{-1} h_1 = h_2^T K^{-T} K^{-1} h_2$$

These constraints allow solving for $K$ from multiple views.

### Non-linear Refinement

Initial estimates are refined using bundle adjustment, minimizing reprojection error:

$$\min_{K, R_i, \mathbf{t}_i} \sum_{i,j} \left\| \mathbf{u}_{ij} - \pi(K, R_i, \mathbf{t}_i, \mathbf{X}_j) \right\|^2$$

where $\pi$ is the projection function and $\mathbf{u}_{ij}$ is the observed image point of 3D point $\mathbf{X}_j$ in view $i$.

## Lens Distortion Correction

Real lenses introduce geometric distortions that deviate from the pinhole model.

### Radial Distortion

Radial distortion depends on distance from optical center:

$$x_{corrected} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$
$$y_{corrected} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

where $r^2 = x^2 + y^2$ and $k_1, k_2, k_3$ are distortion coefficients.

- **Barrel distortion**: $k_1 > 0$, lines curve outward
- **Pincushion distortion**: $k_1 < 0$, lines curve inward

### Tangential Distortion

Tangential distortion (decentering) is caused by misalignment of lens elements:

$$x_{corrected} = x + [2p_1 xy + p_2(r^2 + 2x^2)]$$
$$y_{corrected} = y + [p_1(r^2 + 2y^2) + 2p_2 xy]$$

where $p_1, p_2$ are tangential distortion coefficients.

### Distortion Correction

Distortion parameters are estimated during calibration and used to correct images:

1. For each pixel $(u, v)$, convert to normalized coordinates $(x, y)$
2. Apply distortion model to get corrected $(x', y')$
3. Convert back to pixel coordinates $(u', v')$
4. Interpolate from original image at $(u', v')$

## Multi-View Geometry

Multi-view geometry studies relationships between multiple views of the same scene.

### Epipolar Geometry

Epipolar geometry describes the geometric relationship between two views. For a 3D point $\mathbf{X}$:

- **Epipolar plane**: Plane containing $\mathbf{X}$ and both camera centers
- **Epipolar line**: Intersection of epipolar plane with image plane
- **Epipole**: Intersection of baseline (line joining camera centers) with image plane

### Fundamental Matrix

The fundamental matrix $F$ relates corresponding points in two views:

$$\mathbf{x}'^T F \mathbf{x} = 0$$

For corresponding points $\mathbf{x} \leftrightarrow \mathbf{x}'$.

The fundamental matrix has rank 2 and 7 degrees of freedom. It can be estimated from 8 point correspondences using the 8-point algorithm.

### Essential Matrix

The essential matrix $E$ relates camera coordinates in two views:

$$E = [\mathbf{t}]_\times R$$

where $[\mathbf{t}]_\times$ is the skew-symmetric matrix for cross product.

The relationship to fundamental matrix:
$$F = K'^{-T} E K^{-1}$$

### Stereo Rectification

Stereo rectification transforms images so epipolar lines are horizontal:

1. Compute fundamental matrix $F$
2. Extract epipoles $\mathbf{e}, \mathbf{e}'$
3. Construct rectification homographies $H, H'$
4. Apply homographies to both images

After rectification, corresponding points lie on the same horizontal scanline, simplifying stereo matching.

## Key Takeaways

1. Geometric transformations form a hierarchy from translation (2 DOF) to projective (8 DOF), each preserving different geometric properties.

2. Homogeneous coordinates enable linear matrix representation of all transformations, unifying translation with rotation and scaling.

3. Affine transformations preserve parallelism and area ratios, suitable for orthographic or weak perspective projections.

4. Homographies (projective transformations) preserve collinearity and cross-ratios, modeling perspective projection accurately.

5. Image warping requires backward mapping with interpolation to avoid holes and ensure smooth results.

6. The pinhole camera model projects 3D points to 2D using intrinsic parameters (focal length, principal point) and extrinsic parameters (rotation, translation).

7. Camera calibration estimates camera parameters from known 3D-2D correspondences, typically using planar calibration targets and multiple views.

8. Lens distortion (radial and tangential) deviates from the pinhole model and must be corrected for accurate geometric measurements.

9. Epipolar geometry describes relationships between multiple views, enabling stereo vision and structure-from-motion algorithms.

10. Understanding geometric transformations and camera models is essential for applications requiring spatial reasoning, 3D reconstruction, and multi-view analysis.
