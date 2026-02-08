# Optical Flow and Motion Estimation

## Table of Contents

1. [Introduction](#introduction)
2. [Optical Flow Constraint](#optical-flow-constraint)
3. [Lucas-Kanade Method](#lucas-kanade-method)
4. [Horn-Schunck Method](#horn-schunck-method)
5. [Motion Vectors and Displacement Fields](#motion-vectors-and-displacement-fields)
6. [Feature Tracking](#feature-tracking)
7. [Dense Optical Flow](#dense-optical-flow)
8. [Multi-Scale Optical Flow](#multi-scale-optical-flow)
9. [Applications of Optical Flow](#applications-of-optical-flow)
10. [Key Takeaways](#key-takeaways)

## Introduction

Optical flow estimates the apparent motion of objects in image sequences, representing how pixel intensities move between consecutive frames. Unlike actual motion (movement in 3D space), optical flow captures 2D projection of motion onto the image plane.

Motion estimation is fundamental to video analysis, tracking, action recognition, and many other computer vision applications. The challenge lies in estimating motion from intensity changes, which can be ambiguous due to the aperture problem, occlusions, and illumination changes.

## Optical Flow Constraint

The optical flow constraint equation forms the foundation for most motion estimation methods.

### Brightness Constancy Assumption

The fundamental assumption is that pixel intensities remain constant over time:

$$I(x, y, t) = I(x + \Delta x, y + \Delta y, t + \Delta t)$$

where $I(x, y, t)$ is the image intensity at position $(x, y)$ and time $t$.

### Optical Flow Constraint Equation

Expanding using Taylor series and assuming small motion:

$$I(x + u\Delta t, y + v\Delta t, t + \Delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x}u\Delta t + \frac{\partial I}{\partial y}v\Delta t + \frac{\partial I}{\partial t}\Delta t$$

Setting the first-order terms to zero (brightness constancy):

$$I_x u + I_y v + I_t = 0$$

where:
- $I_x = \frac{\partial I}{\partial x}$: Spatial gradient in $x$
- $I_y = \frac{\partial I}{\partial y}$: Spatial gradient in $y$
- $I_t = \frac{\partial I}{\partial t}$: Temporal gradient
- $(u, v)$: Optical flow vector (velocity)

This is the **optical flow constraint equation** (OFCE).

### Aperture Problem

The OFCE provides only one constraint for two unknowns $(u, v)$, making the problem underconstrained. This is the **aperture problem**: motion along edges is ambiguous.

For an edge with normal $\mathbf{n} = (I_x, I_y)^T / \sqrt{I_x^2 + I_y^2}$:
- **Normal flow**: Component perpendicular to edge: $v_n = -\frac{I_t}{\|\nabla I\|}$
- **Tangential flow**: Component along edge is unconstrained

Additional constraints are needed to resolve the ambiguity.

### Gradient Computation

Spatial gradients computed using finite differences:

$$I_x(x, y, t) \approx \frac{I(x+1, y, t) - I(x-1, y, t)}{2}$$

$$I_y(x, y, t) \approx \frac{I(x, y+1, t) - I(x, y-1, t)}{2}$$

Temporal gradient:

$$I_t(x, y, t) \approx I(x, y, t+1) - I(x, y, t)$$

More robust gradients use Sobel or Scharr operators with smoothing.

## Lucas-Kanade Method

The Lucas-Kanade method assumes constant flow within a local neighborhood, enabling solution of the underconstrained optical flow equation.

### Local Constant Flow Assumption

Assume optical flow $(u, v)$ is constant within a window $W$ around pixel $(x, y)$:

$$I_x(p)u + I_y(p)v + I_t(p) = 0 \quad \forall p \in W$$

This gives an overdetermined system of equations.

### Least Squares Solution

Stacking equations for all pixels in window:

$$\begin{bmatrix} I_x(p_1) & I_y(p_1) \\ I_x(p_2) & I_y(p_2) \\ \vdots & \vdots \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_t(p_1) \\ I_t(p_2) \\ \vdots \end{bmatrix}$$

In matrix form: $A \mathbf{v} = \mathbf{b}$

Least squares solution:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (A^T A)^{-1} A^T \mathbf{b}$$

Expanding:

$$A^T A = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$$

$$A^T \mathbf{b} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}$$

Solution:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \frac{1}{\det(A^T A)} \begin{bmatrix} \sum I_y^2 & -\sum I_x I_y \\ -\sum I_x I_y & \sum I_x^2 \end{bmatrix} \begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}$$

### Structure Tensor

The matrix $A^T A$ is the **structure tensor** (or second-moment matrix):

$$M = \begin{bmatrix} \sum_{W} I_x^2 & \sum_{W} I_x I_y \\ \sum_{W} I_x I_y & \sum_{W} I_y^2 \end{bmatrix}$$

The eigenvalues $\lambda_1, \lambda_2$ of $M$ indicate local structure:
- **Both large**: Corner or textured region (good for tracking)
- **One large, one small**: Edge (aperture problem)
- **Both small**: Flat region (no reliable flow)

Flow is reliable when $\min(\lambda_1, \lambda_2) > \tau$ for threshold $\tau$.

### Weighted Lucas-Kanade

Weight pixels by distance from center using Gaussian weights:

$$M = \sum_{p \in W} w(p) \begin{bmatrix} I_x^2(p) & I_x(p) I_y(p) \\ I_x(p) I_y(p) & I_y^2(p) \end{bmatrix}$$

where $w(p) = \exp\left(-\frac{\|p - c\|^2}{2\sigma^2}\right)$ and $c$ is the window center.

### Iterative Refinement

Lucas-Kanade can be applied iteratively for large motions:

1. Initialize flow $(u_0, v_0) = (0, 0)$
2. Warp image $I_2$ using current flow estimate
3. Compute flow increment $(\Delta u, \Delta v)$
4. Update: $(u, v) \leftarrow (u, v) + (\Delta u, \Delta v)$
5. Repeat until convergence

This is the **iterative Lucas-Kanade** or **pyramidal Lucas-Kanade** when applied across scales.

## Horn-Schunck Method

The Horn-Schunck method formulates optical flow as a global optimization problem, assuming smooth flow across the entire image.

### Smoothness Constraint

Horn-Schunck adds a smoothness term penalizing flow variations:

$$E = \int \int (I_x u + I_y v + I_t)^2 + \lambda (u_x^2 + u_y^2 + v_x^2 + v_y^2) dx dy$$

where:
- First term: Optical flow constraint
- Second term: Smoothness constraint
- $\lambda$: Regularization parameter

The smoothness term assumes flow varies slowly, which is reasonable for most scenes.

### Euler-Lagrange Equations

Minimizing the energy functional gives Euler-Lagrange equations:

$$I_x(I_x u + I_y v + I_t) - \lambda \nabla^2 u = 0$$
$$I_y(I_x u + I_y v + I_t) - \lambda \nabla^2 v = 0$$

where $\nabla^2$ is the Laplacian operator:
$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$$

### Iterative Solution

Discretizing and solving iteratively:

$$u^{n+1} = \bar{u}^n - \frac{I_x(I_x \bar{u}^n + I_y \bar{v}^n + I_t)}{\lambda + I_x^2 + I_y^2}$$

$$v^{n+1} = \bar{v}^n - \frac{I_y(I_x \bar{u}^n + I_y \bar{v}^n + I_t)}{\lambda + I_y^2 + I_x^2}$$

where $\bar{u}^n$ and $\bar{v}^n$ are local averages of $u$ and $v$:
$$\bar{u} = \frac{1}{6}(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1}) + \frac{1}{12}(u_{i-1,j-1} + u_{i-1,j+1} + u_{i+1,j-1} + u_{i+1,j+1})$$

### Boundary Conditions

Boundary conditions are needed:
- **Neumann**: $\frac{\partial u}{\partial n} = 0$ (zero normal derivative)
- **Dirichlet**: $u = 0$ (zero flow at boundaries)

### Advantages and Limitations

**Advantages**:
- Produces dense flow fields
- Handles large regions of uniform motion
- Smooth results

**Limitations**:
- Assumes smoothness everywhere (violated at boundaries)
- Computationally expensive
- May oversmooth motion discontinuities

## Motion Vectors and Displacement Fields

Motion vectors represent the displacement of pixels between frames, directly related to optical flow.

### Displacement Field

A displacement field $\mathbf{d}(x, y) = (d_x(x, y), d_y(x, y))$ maps each pixel to its new location:

$$I_2(x + d_x, y + d_y) = I_1(x, y)$$

Relationship to optical flow:
$$\mathbf{v}(x, y) = \frac{\mathbf{d}(x, y)}{\Delta t}$$

For $\Delta t = 1$: $\mathbf{v} = \mathbf{d}$.

### Motion Models

**Translation**: $\mathbf{d}(x, y) = (t_x, t_y)$
**Affine**: $\mathbf{d}(x, y) = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$
**Perspective**: Homography transformation

### Block-Based Motion Estimation

Divide image into blocks, estimate motion per block:

1. **Block matching**: Search for best match in reference frame
2. **Motion vector**: Displacement of best match
3. **Search strategy**: Full search, three-step search, diamond search

Cost function (e.g., SAD):
$$C(d_x, d_y) = \sum_{(i,j) \in B} |I_1(x+i, y+j) - I_2(x+i+d_x, y+j+d_y)|$$

## Feature Tracking

Feature tracking follows distinctive points across frames, more robust than dense optical flow for many applications.

### KLT Tracker

The Kanade-Lucas-Tomasi (KLT) tracker extends Lucas-Kanade for feature tracking:

1. **Feature selection**: Choose corners using Shi-Tomasi criterion
2. **Tracking**: Apply Lucas-Kanade to each feature
3. **Validation**: Check tracking quality, remove lost features

### Feature Selection

Select features with large minimum eigenvalue:
$$\min(\lambda_1, \lambda_2) > \tau$$

This ensures reliable tracking.

### Tracking Algorithm

For each feature at location $(x, y)$ in frame $t$:

1. **Initialize**: Set search window around feature
2. **Compute flow**: Use Lucas-Kanade to find displacement
3. **Update location**: $(x', y') = (x, y) + (u, v)$
4. **Validate**: Check tracking quality
5. **Re-detect**: If lost, detect new features

### Tracking Quality Measures

**SSD**: Sum of squared differences between template and current patch
**NCC**: Normalized cross-correlation
**Forward-backward error**: Track forward then backward, check consistency

### Mean Shift Tracking

Mean shift tracks objects by maximizing similarity in feature space:

1. **Target model**: Represent target as histogram (e.g., color histogram)
2. **Candidate model**: Compute histogram in search window
3. **Mean shift**: Iteratively move toward maximum similarity
4. **Update**: Update target model for next frame

Similarity measure (Bhattacharyya coefficient):
$$\rho(\mathbf{q}, \mathbf{p}) = \sum_{u=1}^{m} \sqrt{q_u p_u}$$

where $\mathbf{q}$ is target histogram and $\mathbf{p}$ is candidate histogram.

### CamShift

Continuously Adaptive Mean Shift (CamShift) extends mean shift:
- Adapts window size based on object scale
- Handles scale changes
- Used for face and hand tracking

## Dense Optical Flow

Dense optical flow computes flow for every pixel, enabling detailed motion analysis.

### Variational Methods

Variational methods minimize energy functionals:

$$E(u, v) = \int \int E_{data}(u, v) + \lambda E_{smooth}(u, v) dx dy$$

**Data term**: Optical flow constraint
**Smoothness term**: Flow regularization

### TV-L1 Optical Flow

Total Variation L1 (TV-L1) uses robust L1 norm:

$$E = \int \int |I_x u + I_y v + I_t| + \lambda (|\nabla u| + |\nabla v|) dx dy$$

L1 norm is more robust to outliers than L2.

### Brox et al. Method

Brox et al. use gradient constancy in addition to brightness:

$$E = \int \int \Psi(|I_1(\mathbf{x}) - I_2(\mathbf{x}+\mathbf{u})|^2) + \gamma \Psi(|\nabla I_1(\mathbf{x}) - \nabla I_2(\mathbf{x}+\mathbf{u})|^2) + \lambda \Psi(|\nabla u|^2 + |\nabla v|^2) dx dy$$

where $\Psi(s^2) = \sqrt{s^2 + \epsilon^2}$ is a robust function.

### Deep Learning Methods

Modern methods use convolutional neural networks:

**FlowNet**: End-to-end CNN for optical flow
**FlowNet 2.0**: Improved architecture and training
**PWC-Net**: Pyramid, warping, and cost volume
**RAFT**: Recurrent All-Pairs Field Transforms

These methods learn to predict flow from image pairs, achieving state-of-the-art results.

## Multi-Scale Optical Flow

Multi-scale approaches handle large motions by processing images at multiple resolutions.

### Image Pyramids

Build Gaussian pyramid:
$$I^0 = I$$
$$I^{l+1} = \text{Downsample}(G_\sigma * I^l)$$

where $G_\sigma$ is Gaussian kernel and levels $l = 0, 1, 2, \ldots$.

### Coarse-to-Fine Strategy

1. **Coarse level**: Compute flow at low resolution (handles large motions)
2. **Warp**: Warp second image using coarse flow
3. **Fine level**: Compute residual flow at higher resolution
4. **Combine**: Add residual to upsampled coarse flow
5. **Repeat**: Continue to finest level

### Pyramidal Lucas-Kanade

Apply Lucas-Kanade at each pyramid level:

1. Start at coarsest level $L$
2. Compute flow $(u^L, v^L)$
3. Upsample to level $L-1$: $(u^{L-1}, v^{L-1}) = 2 \cdot (u^L, v^L)$
4. Warp $I_2$ using upsampled flow
5. Compute residual flow $(\Delta u, \Delta v)$
6. Update: $(u^{L-1}, v^{L-1}) \leftarrow (u^{L-1}, v^{L-1}) + (\Delta u, \Delta v)$
7. Repeat until finest level

This handles motions larger than the search window.

## Applications of Optical Flow

Optical flow enables numerous computer vision applications.

### Motion Segmentation

Separate moving objects from background:
- Compute optical flow
- Cluster flow vectors
- Identify coherent motion regions

### Action Recognition

Analyze human actions from flow:
- Compute dense optical flow
- Extract motion features (histograms, trajectories)
- Classify actions using machine learning

### Video Stabilization

Remove camera shake:
- Estimate global motion (dominant flow)
- Compensate for camera motion
- Generate stabilized video

### Object Tracking

Track objects using flow:
- Compute flow in region of interest
- Estimate object motion
- Update object location

### Ego-Motion Estimation

Estimate camera motion:
- Compute optical flow
- Estimate dominant motion (camera motion)
- Recover camera rotation and translation

### Video Compression

Motion compensation in video coding:
- Estimate motion vectors
- Predict frames using motion
- Encode residuals

## Key Takeaways

1. Optical flow estimates apparent motion from intensity changes, constrained by the brightness constancy assumption and optical flow constraint equation.

2. The aperture problem makes optical flow underconstrained, requiring additional assumptions (local constancy, smoothness) to solve.

3. Lucas-Kanade assumes constant flow within local neighborhoods, solving overdetermined systems using least squares with structure tensor analysis.

4. Horn-Schunck formulates optical flow as global optimization with smoothness constraints, producing dense flow fields but potentially oversmoothing boundaries.

5. Feature tracking (KLT, mean shift) follows distinctive points across frames, more robust than dense flow for many tracking applications.

6. Dense optical flow methods (variational, TV-L1, deep learning) compute flow for every pixel, enabling detailed motion analysis.

7. Multi-scale approaches (pyramidal Lucas-Kanade) handle large motions by processing images at multiple resolutions in coarse-to-fine fashion.

8. Motion vectors represent pixel displacements, directly related to optical flow and used in block-based motion estimation.

9. Optical flow enables applications including motion segmentation, action recognition, video stabilization, object tracking, and ego-motion estimation.

10. Understanding optical flow constraints, solution methods, and multi-scale strategies enables robust motion estimation for diverse computer vision applications.
