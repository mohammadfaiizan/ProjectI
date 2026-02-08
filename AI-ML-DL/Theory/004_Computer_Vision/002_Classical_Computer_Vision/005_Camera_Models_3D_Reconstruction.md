# Camera Models and 3D Reconstruction

## Table of Contents

1. [Introduction](#introduction)
2. [Pinhole Camera Model](#pinhole-camera-model)
3. [Intrinsic and Extrinsic Parameters](#intrinsic-and-extrinsic-parameters)
4. [Structure from Motion](#structure-from-motion)
5. [Bundle Adjustment](#bundle-adjustment)
6. [Multi-View Stereo](#multi-view-stereo)
7. [Dense Reconstruction](#dense-reconstruction)
8. [3D Representation Formats](#3d-representation-formats)
9. [Reconstruction Evaluation](#reconstruction-evaluation)
10. [Key Takeaways](#key-takeaways)

## Introduction

3D reconstruction recovers three-dimensional structure from two-dimensional images, enabling understanding of scene geometry, object shape, and spatial relationships. Camera models describe how 3D points project onto 2D image planes, forming the mathematical foundation for reconstruction.

Structure from Motion (SfM) simultaneously estimates camera poses and 3D structure from image sequences, while multi-view stereo produces dense 3D models from multiple calibrated views. These techniques enable applications including 3D modeling, augmented reality, autonomous navigation, and cultural heritage preservation.

## Pinhole Camera Model

The pinhole camera is the simplest and most widely used camera model, approximating how light rays pass through a small aperture to form an image.

### Basic Projection

A 3D point $\mathbf{X} = [X, Y, Z]^T$ projects to 2D image point $\mathbf{x} = [x, y]^T$:

$$x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}$$

where $f$ is the focal length (distance from pinhole to image plane).

### Homogeneous Coordinates

In homogeneous coordinates, projection becomes linear:

$$\lambda \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

where $\lambda = Z$ is the depth.

### Principal Point

The principal point $(c_x, c_y)$ is where the optical axis intersects the image plane:

$$x = f \frac{X}{Z} + c_x, \quad y = f \frac{Y}{Z} + c_y$$

### Non-Square Pixels

If pixels are not square, different focal lengths $f_x$ and $f_y$ are used:

$$x = f_x \frac{X}{Z} + c_x, \quad y = f_y \frac{Y}{Z} + c_y$$

### Skew Parameter

Skew parameter $s$ accounts for non-rectangular pixels (rare in modern cameras):

$$\begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X/Z \\ Y/Z \\ 1 \end{bmatrix}$$

## Intrinsic and Extrinsic Parameters

Camera parameters are divided into intrinsic (internal) and extrinsic (pose) parameters.

### Intrinsic Parameters

Intrinsic parameters describe internal camera geometry, represented by calibration matrix $K$:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Parameters:
- $f_x, f_y$: Focal lengths in pixels
- $c_x, c_y$: Principal point coordinates
- $s$: Skew parameter (usually 0)

### Extrinsic Parameters

Extrinsic parameters describe camera pose (position and orientation):

$$\mathbf{X}_c = R \mathbf{X}_w + \mathbf{t}$$

where:
- $\mathbf{X}_w$: 3D point in world coordinates
- $\mathbf{X}_c$: 3D point in camera coordinates
- $R$: 3×3 rotation matrix
- $\mathbf{t}$: 3×1 translation vector

### Complete Projection

Combining intrinsic and extrinsic parameters:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

In compact form:
$$\lambda \mathbf{x} = P \mathbf{X}$$

where $P = K[R | \mathbf{t}]$ is the $3 \times 4$ projection matrix.

### Camera Pose Representation

Rotation can be represented as:
- **Rotation matrix**: $R$ (9 parameters, 6 DOF)
- **Axis-angle**: $\mathbf{r} = \theta \mathbf{n}$ (3 parameters)
- **Quaternion**: $\mathbf{q} = (w, x, y, z)$ (4 parameters, 3 DOF)

Translation: $\mathbf{t} = [t_x, t_y, t_z]^T$ (3 parameters)

Total: 6 DOF for camera pose.

## Structure from Motion

Structure from Motion (SfM) simultaneously estimates camera poses and 3D structure from image sequences without known camera calibration.

### SfM Pipeline

1. **Feature detection and matching**: Detect and match features across images
2. **Initialization**: Select two views, estimate fundamental matrix, triangulate initial points
3. **Incremental reconstruction**: Add views one by one:
   - Estimate pose of new view
   - Triangulate new points
   - Bundle adjustment
4. **Global optimization**: Bundle adjustment on all views
5. **Dense reconstruction**: Multi-view stereo for dense model

### Two-View Reconstruction

**Step 1: Fundamental Matrix**
Estimate $F$ from correspondences using 8-point algorithm + RANSAC.

**Step 2: Camera Matrices**
From $F$, recover camera matrices:
$$P_1 = [I | 0], \quad P_2 = [[\mathbf{e}_2]_\times F | \mathbf{e}_2]$$

where $\mathbf{e}_2$ is the epipole in image 2.

**Step 3: Triangulation**
For each correspondence $\mathbf{x}_1 \leftrightarrow \mathbf{x}_2$:
$$\lambda_1 \mathbf{x}_1 = P_1 \mathbf{X}, \quad \lambda_2 \mathbf{x}_2 = P_2 \mathbf{X}$$

Solve for 3D point $\mathbf{X}$.

### Incremental SfM

Add views incrementally:

1. **Select next view**: Choose view with most matches to existing reconstruction
2. **Estimate pose**: Use PnP (Perspective-n-Point) to estimate camera pose
3. **Triangulate**: Create new 3D points from new correspondences
4. **Bundle adjustment**: Optimize all parameters

### Perspective-n-Point (PnP)

PnP estimates camera pose from $n$ 3D-2D correspondences.

**P3P**: Minimal case with 3 points, up to 4 solutions
**EPnP**: Efficient PnP for $n \geq 4$
**DLT**: Direct Linear Transform for PnP

For $n$ correspondences $\mathbf{X}_i \leftrightarrow \mathbf{x}_i$:
$$\lambda_i \mathbf{x}_i = K[R | \mathbf{t}] \mathbf{X}_i$$

Solve for $R$ and $\mathbf{t}$.

### Robust Estimation

RANSAC is used throughout SfM:
- Fundamental matrix estimation
- Homography estimation
- PnP pose estimation
- Outlier rejection

## Bundle Adjustment

Bundle adjustment is the joint optimization of camera parameters and 3D structure, minimizing reprojection error.

### Reprojection Error

For 3D point $\mathbf{X}_j$ observed in image $i$ at pixel $\mathbf{u}_{ij}$:

$$e_{ij} = \mathbf{u}_{ij} - \pi(P_i, \mathbf{X}_j)$$

where $\pi$ is the projection function:
$$\pi(P, \mathbf{X}) = \frac{1}{Z} \begin{bmatrix} f_x X + s Y + c_x Z \\ f_y Y + c_y Z \\ Z \end{bmatrix}$$

### Bundle Adjustment Objective

Minimize total reprojection error:

$$\min_{\{P_i\}, \{\mathbf{X}_j\}} \sum_{i,j} \rho(\|e_{ij}\|^2)$$

where $\rho$ is a robust cost function (e.g., Huber loss).

For squared error:
$$\min_{\{P_i\}, \{\mathbf{X}_j\}} \sum_{i,j} \|\mathbf{u}_{ij} - \pi(P_i, \mathbf{X}_j)\|^2$$

### Parameterization

Parameters to optimize:
- **Camera poses**: $R_i, \mathbf{t}_i$ for each view (6 DOF each)
- **3D points**: $\mathbf{X}_j$ for each point (3 DOF each)
- **Intrinsics**: $K$ (typically 4-5 parameters, may be shared)

Total parameters: $6m + 3n + k$ for $m$ views, $n$ points, $k$ intrinsic parameters.

### Optimization

Bundle adjustment is a large-scale non-linear least squares problem, solved using:

**Levenberg-Marquardt**: Combines Gauss-Newton and gradient descent
**Sparse solvers**: Exploit sparsity structure (each point visible in few views)
**Schur complement**: Eliminate 3D points, solve for cameras only

The normal equations:
$$J^T J \delta = -J^T \mathbf{r}$$

where $J$ is the Jacobian and $\mathbf{r}$ is the residual vector.

### Sparse Structure

The Jacobian has block structure:
$$J = \begin{bmatrix} A & B \end{bmatrix}$$

where:
- $A$: Derivatives w.r.t. camera parameters (sparse)
- $B$: Derivatives w.r.t. 3D points (sparse)

Each row corresponds to one observation, with non-zero blocks only for the camera and point involved.

## Multi-View Stereo

Multi-view stereo (MVS) produces dense 3D models from multiple calibrated views, complementing sparse SfM reconstruction.

### MVS Pipeline

1. **Depth estimation**: Compute depth maps for each view
2. **Depth map fusion**: Combine depth maps into unified model
3. **Surface reconstruction**: Generate mesh from point cloud
4. **Texture mapping**: Project images onto mesh

### Depth Map Computation

For each view, compute depth for each pixel:

**Stereo matching**: Match pixels across views
**Cost volume**: Build 3D cost volume $(x, y, d)$
**Optimization**: Find optimal depth per pixel

Cost function:
$$C(x, y, d) = \sum_{i} w_i \cdot \text{PhotoConsistency}(I_i, I_{ref}, d)$$

where photo-consistency measures similarity of projections.

### PatchMatch Stereo

PatchMatch efficiently finds best matches:

1. **Random initialization**: Random disparities
2. **Propagation**: Propagate good matches to neighbors
3. **Random search**: Random search for improvements
4. **Iterate**: Repeat until convergence

### Depth Map Fusion

Combine depth maps from multiple views:

1. **Transform to common frame**: Convert all depth maps to world coordinates
2. **Fusion**: Combine using median or weighted average
3. **Outlier removal**: Remove inconsistent depths
4. **Point cloud**: Generate 3D point cloud

### Surface Reconstruction

Convert point cloud to mesh:

**Poisson reconstruction**: Solve Poisson equation
**Marching cubes**: Extract isosurface from signed distance function
**Delaunay triangulation**: Triangulate point cloud

## Dense Reconstruction

Dense reconstruction produces complete 3D models with surface details.

### Volumetric Methods

**Space carving**: Start with volume, remove inconsistent voxels
**Voxel coloring**: Use color consistency
**Visual hull**: Intersection of visual cones

### Surface-Based Methods

**Multi-view stereo**: Compute depth maps, fuse into surface
**Variational methods**: Optimize surface energy functionals
**Deformable models**: Evolve surfaces to fit images

### Signed Distance Functions

Represent surface implicitly using signed distance function (SDF):

$$f(\mathbf{x}) = \begin{cases}
+d(\mathbf{x}, S) & \text{if } \mathbf{x} \text{ outside} \\
-d(\mathbf{x}, S) & \text{if } \mathbf{x} \text{ inside} \\
0 & \text{if } \mathbf{x} \text{ on surface}
\end{cases}$$

Surface is zero level set: $S = \{\mathbf{x} : f(\mathbf{x}) = 0\}$.

### Truncated SDF (TSDF)

TSDF truncates distance values:

$$f_{TSDF}(\mathbf{x}) = \begin{cases}
+1 & \text{if } d > \tau \\
d/\tau & \text{if } |d| \leq \tau \\
-1 & \text{if } d < -\tau
\end{cases}$$

Enables efficient fusion from multiple views.

## 3D Representation Formats

3D reconstructions can be represented in various formats.

### Point Clouds

Set of 3D points $\{\mathbf{X}_i\}$, optionally with colors/normals:
- **PLY**: Polygon file format
- **PCD**: Point Cloud Data
- **XYZ**: Simple text format

### Meshes

Triangular meshes: vertices $\{\mathbf{v}_i\}$ and faces $\{f_j\}$:
- **OBJ**: Wavefront OBJ format
- **PLY**: Polygon file format
- **STL**: Stereolithography format

### Voxel Grids

3D arrays of voxels:
- **Occupancy grids**: Binary (occupied/empty)
- **TSDF volumes**: Truncated signed distance values
- **Color volumes**: RGB values per voxel

### Implicit Representations

**Neural radiance fields (NeRF)**: Neural network $f(\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$
**Signed distance functions**: $f(\mathbf{x}) \rightarrow d$
**Occupancy networks**: $f(\mathbf{x}) \rightarrow \{0, 1\}$

## Reconstruction Evaluation

Evaluating 3D reconstruction quality requires ground truth data.

### Accuracy Metrics

**Mean distance**: Average distance from reconstructed points to ground truth:
$$\text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \min_{\mathbf{y} \in GT} \|\mathbf{x}_i - \mathbf{y}\|$$

**Completeness**: Fraction of ground truth points with nearby reconstruction:
$$\text{Completeness} = \frac{|\{\mathbf{y} \in GT : \min_i \|\mathbf{x}_i - \mathbf{y}\| < \tau\}|}{|GT|}$$

**F-score**: Harmonic mean of accuracy and completeness:
$$F = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Camera Pose Evaluation

**Rotation error**: Angular difference between estimated and ground truth rotation
**Translation error**: Euclidean distance between estimated and ground truth translation

### Visual Quality

**Photometric error**: Difference between rendered and actual images
**Geometric consistency**: Consistency across multiple views
**Texture quality**: Quality of texture mapping

## Key Takeaways

1. The pinhole camera model projects 3D points to 2D using linear transformations in homogeneous coordinates, parameterized by intrinsic (focal length, principal point) and extrinsic (rotation, translation) parameters.

2. Structure from Motion simultaneously estimates camera poses and 3D structure from image sequences, using feature matching, fundamental matrix estimation, and triangulation.

3. Bundle adjustment jointly optimizes all camera parameters and 3D points by minimizing reprojection error, using sparse solvers to handle large-scale problems efficiently.

4. Multi-view stereo produces dense 3D models by computing and fusing depth maps from multiple calibrated views, complementing sparse SfM reconstruction.

5. Depth estimation uses stereo matching, cost volumes, and optimization to compute per-pixel depth, with PatchMatch providing efficient matching.

6. Dense reconstruction methods include volumetric (space carving, TSDF) and surface-based (multi-view stereo, variational) approaches.

7. 3D models can be represented as point clouds, meshes, voxel grids, or implicit functions (NeRF, SDF), each suitable for different applications.

8. Camera pose estimation uses PnP algorithms to recover rotation and translation from 3D-2D correspondences, with RANSAC for robustness.

9. Reconstruction evaluation uses accuracy, completeness, and F-score metrics, comparing against ground truth 3D models and camera poses.

10. Understanding camera models, SfM, bundle adjustment, and multi-view stereo enables 3D reconstruction from images, fundamental to many computer vision and graphics applications.
