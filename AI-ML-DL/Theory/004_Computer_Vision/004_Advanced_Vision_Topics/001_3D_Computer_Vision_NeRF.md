# 3D Computer Vision and NeRF

## Table of Contents

1. [Introduction](#introduction)
2. [3D Representation Formats](#3d-representation-formats)
3. [3D Object Detection](#3d-object-detection)
4. [Point Cloud Processing](#point-cloud-processing)
5. [Neural Radiance Fields (NeRF)](#neural-radiance-fields-nerf)
6. [NeRF Architecture and Training](#nerf-architecture-and-training)
7. [NeRF Variants and Improvements](#nerf-variants-and-improvements)
8. [Depth Estimation](#depth-estimation)
9. [3D Reconstruction Methods](#3d-reconstruction-methods)
10. [Key Takeaways](#key-takeaways)

## Introduction

3D computer vision extends 2D image understanding to three-dimensional space, enabling applications including autonomous driving, robotics, augmented reality, and 3D content creation. Neural Radiance Fields (NeRF) revolutionized 3D scene representation by learning continuous volumetric functions from images, enabling novel view synthesis and high-quality 3D reconstruction.

Understanding 3D vision requires knowledge of representation formats (point clouds, meshes, voxels), detection methods (3D bounding boxes), and modern neural approaches (NeRF, point cloud networks). These methods enable machines to understand and interact with 3D environments.

## 3D Representation Formats

Different 3D representations suit different applications and have different properties.

### Point Clouds

Point clouds are sets of 3D points $\{\mathbf{p}_i = (x_i, y_i, z_i)\}$:
- **Sparse**: Only surface points
- **Unstructured**: No connectivity information
- **Efficient**: Compact representation

**Properties**:
- Permutation invariant
- Variable size
- No explicit topology

**Formats**: PLY, PCD, XYZ

### Meshes

Meshes consist of vertices and faces:
- **Vertices**: 3D points $\{\mathbf{v}_i\}$
- **Faces**: Triangular faces $\{f_j = (v_{j1}, v_{j2}, v_{j3})\}$

**Properties**:
- Explicit topology
- Surface representation
- Can be textured

**Formats**: OBJ, PLY, STL

### Voxel Grids

Voxel grids are 3D arrays:
- **Occupancy**: Binary (occupied/empty)
- **TSDF**: Truncated signed distance function
- **Color**: RGB values per voxel

**Properties**:
- Regular structure
- Memory intensive ($O(n^3)$)
- Easy to process with CNNs

### Implicit Representations

**Signed Distance Functions (SDF)**:
$$f(\mathbf{x}) = \begin{cases}
+d(\mathbf{x}, S) & \text{if outside} \\
-d(\mathbf{x}, S) & \text{if inside} \\
0 & \text{if on surface}
\end{cases}$$

**Neural Implicit Functions**: $f_\theta(\mathbf{x}) \rightarrow \text{value}$
- SDF: $f_\theta(\mathbf{x}) \rightarrow d$
- Occupancy: $f_\theta(\mathbf{x}) \rightarrow \{0, 1\}$
- NeRF: $f_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$

## 3D Object Detection

3D object detection localizes objects in 3D space with 3D bounding boxes.

### 3D Bounding Boxes

A 3D box is parameterized by:
- **Center**: $(x, y, z)$
- **Size**: $(w, l, h)$ (width, length, height)
- **Rotation**: $\theta$ (yaw angle around z-axis)

Total: 7 parameters per box.

### Detection from Images

**Monocular 3D detection**:
- Input: Single RGB image
- Output: 3D boxes
- Challenge: Depth ambiguity

**Stereo 3D detection**:
- Input: Stereo image pair
- Output: 3D boxes
- Benefit: Depth from stereo

**Multi-view detection**:
- Input: Multiple images
- Output: 3D boxes
- Benefit: Better coverage

### Point Cloud Detection

**VoxelNet**: Voxelize point cloud, apply 3D CNN
**PointNet/PointNet++**: Process points directly
**Point Pillars**: Convert to pillars, apply 2D CNN

### VoxelNet Architecture

1. **Voxelization**: Divide space into voxels
2. **Feature extraction**: Group points per voxel, extract features
3. **3D CNN**: Apply 3D convolutions
4. **Detection head**: Predict 3D boxes

### PointNet

PointNet processes point clouds directly:

**Architecture**:
- Input: $N \times 3$ points
- MLP: Per-point features
- Max pooling: Global features
- MLP: Classification/segmentation

**Key innovation**: Permutation invariant through max pooling.

### Point Pillars

Point Pillars converts points to pillars:

1. **Pillar encoding**: Project points to pillars
2. **Pillar features**: Extract features per pillar
3. **2D CNN**: Apply 2D convolutions on pillar map
4. **Detection**: Predict boxes

Efficient: 2D CNN instead of 3D CNN.

## Point Cloud Processing

Point cloud processing requires handling unstructured, variable-size data.

### Point Cloud Networks

**PointNet**: Permutation invariant through max pooling
**PointNet++**: Hierarchical feature learning
**DGCNN**: Dynamic graph CNN
**Point Transformer**: Self-attention for points

### PointNet Architecture

**Per-point MLP**: Extract features for each point
**Max pooling**: Aggregate to global features
**Permutation invariant**: Order doesn't matter

$$f(\{p_1, \ldots, p_n\}) = \gamma(\max_{i=1,\ldots,n} h(p_i))$$

where $\gamma$ and $h$ are MLPs.

### PointNet++

PointNet++ adds hierarchical structure:

1. **Sampling**: Farthest point sampling
2. **Grouping**: K-nearest neighbors
3. **PointNet**: Extract local features
4. **Upsampling**: Interpolate to original points

Enables multi-scale feature learning.

### Point Cloud Segmentation

**Semantic segmentation**: Class per point
**Instance segmentation**: Instance per point
**Part segmentation**: Part label per point

Methods:
- PointNet/PointNet++: Direct point processing
- Projection: Project to 2D, use 2D CNNs
- Voxelization: Convert to voxels, use 3D CNNs

## Neural Radiance Fields (NeRF)

NeRF represents scenes as continuous volumetric functions learned from images.

### NeRF Representation

NeRF models a scene as:
$$F_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$$

where:
- $\mathbf{x} = (x, y, z)$: 3D position
- $\mathbf{d} = (\theta, \phi)$: Viewing direction
- $c = (r, g, b)$: RGB color
- $\sigma$: Volume density

### Volume Rendering

Render image by integrating along rays:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$$

where:
- $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$: Ray
- $T(t) = \exp(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds)$: Transmittance

**Discrete approximation**:
$$C(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$$

where:
- $T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j \delta_j)$
- $\delta_i$: Distance between samples

### Key Properties

- **Continuous**: Implicit representation
- **View-dependent**: Color depends on viewing direction
- **Differentiable**: Enables gradient-based optimization

## NeRF Architecture and Training

NeRF uses MLPs to represent the radiance field.

### Architecture

**Positional encoding**: Encode position and direction
$$\gamma(\mathbf{x}) = (\sin(2^0 \pi \mathbf{x}), \cos(2^0 \pi \mathbf{x}), \ldots, \sin(2^{L-1} \pi \mathbf{x}), \cos(2^{L-1} \pi \mathbf{x}))$$

where $L=10$ for positions, $L=4$ for directions.

**MLP**: 8 layers for density, 1 layer for color
- Input: Encoded position
- Hidden: 256 neurons, ReLU
- Output: Density $\sigma$ + 256-dim feature
- Color head: Input feature + encoded direction → RGB

### Training

**Input**: 
- Images with known camera poses
- Camera intrinsics and extrinsics

**Process**:
1. Sample rays through each pixel
2. Sample points along rays
3. Query NeRF for color and density
4. Volume render to get pixel color
5. Compare with ground truth
6. Backpropagate and update

**Loss**: Mean squared error
$$L = \sum_{\mathbf{r} \in \mathcal{R}} \|C(\mathbf{r}) - \hat{C}(\mathbf{r})\|^2$$

where $\mathcal{R}$ is set of rays.

### Hierarchical Sampling

**Coarse network**: Sample uniformly
**Fine network**: Sample based on coarse density
- More samples in high-density regions
- Better quality with same computation

### Challenges

- **Slow training**: Many forward passes per image
- **Slow rendering**: Requires many samples per ray
- **Memory**: Store many samples

## NeRF Variants and Improvements

Many improvements address NeRF limitations.

### Instant NGP

**Hash encoding**: Multi-resolution hash tables
- Faster training and rendering
- Better quality
- Less memory

### Plenoxels

**Explicit representation**: Voxel grid instead of MLP
- Faster training
- Faster rendering
- More memory

### Mip-NeRF

**Conical frustums**: Instead of rays
- Handles aliasing better
- Better for multi-resolution

### NeRF-W

**Uncertainty**: Models uncertainty in images
- Handles varying lighting
- More robust

### NeRF in the Wild

**Appearance embedding**: Per-image appearance code
- Handles varying conditions
- Better for in-the-wild scenes

### Fast NeRF

**Plucker coordinates**: More efficient ray representation
- Faster rendering
- Similar quality

## Depth Estimation

Depth estimation recovers 3D structure from 2D images.

### Monocular Depth Estimation

**Supervised**: Train on depth maps
- **Loss**: L1/L2 between predicted and ground truth
- **Challenges**: Scale ambiguity, limited training data

**Self-supervised**: Use stereo or video
- **Stereo**: Left-right consistency
- **Video**: Temporal consistency

### Stereo Depth

**Stereo matching**: Find correspondences
**Disparity**: Horizontal shift
**Depth**: $Z = \frac{bf}{d}$

### Multi-View Depth

**MVS**: Multi-view stereo
**Cost volume**: 3D volume of matching costs
**Optimization**: Find optimal depth per pixel

### Learning-Based Depth

**Monodepth**: Self-supervised from stereo
**Depth from motion**: Use camera motion
**Depth from focus**: Use focus information

## 3D Reconstruction Methods

3D reconstruction recovers 3D models from images.

### Structure from Motion (SfM)

**Sparse reconstruction**: Feature points
**Multi-view stereo**: Dense reconstruction
**Mesh generation**: Surface from points

### Neural Implicit Surfaces

**Neural SDF**: Learn signed distance function
**Neural occupancy**: Learn occupancy function
**Differentiable rendering**: Render and compare

### Differentiable Rendering

**Rasterization**: Differentiable mesh rendering
**Ray tracing**: Differentiable ray tracing
**Neural rendering**: NeRF-style rendering

### 3D from Single Image

**Shape from shading**: Recover shape from shading
**Shape from texture**: Use texture information
**Learning-based**: Train on 3D data

## Key Takeaways

1. 3D representations include point clouds (sparse, unstructured), meshes (explicit topology), voxel grids (regular structure), and implicit functions (continuous).

2. 3D object detection localizes objects with 3D bounding boxes, using methods like VoxelNet, PointNet, and Point Pillars for point cloud data.

3. Point cloud processing requires permutation-invariant networks like PointNet (max pooling) and hierarchical methods like PointNet++ for multi-scale features.

4. Neural Radiance Fields (NeRF) represent scenes as continuous volumetric functions $F_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$, enabling novel view synthesis.

5. NeRF training uses volume rendering to integrate radiance along rays, optimizing MLPs to match observed images through gradient descent.

6. NeRF variants address limitations: Instant NGP (hash encoding), Plenoxels (explicit voxels), Mip-NeRF (conical frustums), NeRF-W (uncertainty modeling).

7. Depth estimation recovers 3D structure from 2D images, using supervised learning, self-supervision (stereo/video), or multi-view stereo.

8. 3D reconstruction methods include SfM (sparse), multi-view stereo (dense), and neural implicit surfaces (learned SDF/occupancy).

9. Differentiable rendering enables end-to-end optimization of 3D representations by comparing rendered and observed images.

10. Understanding 3D computer vision and NeRF enables applications including autonomous driving, robotics, AR/VR, and 3D content creation through novel view synthesis and 3D understanding.
