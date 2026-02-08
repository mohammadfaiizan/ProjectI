# Digital Image Representation

## Table of Contents

1. [Introduction](#introduction)
2. [Pixel Fundamentals](#pixel-fundamentals)
3. [Color Spaces](#color-spaces)
4. [Image Formats and Compression](#image-formats-and-compression)
5. [Spatial and Temporal Resolution](#spatial-and-temporal-resolution)
6. [Bit Depth and Dynamic Range](#bit-depth-and-dynamic-range)
7. [Image Sampling and Quantization](#image-sampling-and-quantization)
8. [Digital Image Storage](#digital-image-storage)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Digital image representation forms the foundation of computer vision and image processing. An image in the digital domain is fundamentally a discrete two-dimensional array of numerical values, where each element represents the intensity or color information at a specific spatial location. Understanding how images are represented, stored, and manipulated is crucial for developing effective computer vision algorithms.

The transition from continuous analog signals to discrete digital representations involves several critical processes: sampling in the spatial domain, quantization in the intensity domain, and encoding for efficient storage and transmission. Each of these processes introduces trade-offs between fidelity, storage requirements, and computational efficiency.

## Pixel Fundamentals

A pixel (picture element) is the fundamental unit of a digital image. Each pixel represents a small region of the continuous image and contains information about the intensity or color at that location. In mathematical terms, a digital image $I$ can be represented as:

$$I(x, y) = f(x, y)$$

where $(x, y)$ are discrete spatial coordinates, and $f(x, y)$ represents the intensity or color value at that position.

For grayscale images, each pixel typically stores a single intensity value. The intensity range depends on the bit depth:

- **8-bit images**: Values range from 0 to 255, representing 256 distinct intensity levels
- **12-bit images**: Values range from 0 to 4095
- **16-bit images**: Values range from 0 to 65535

The pixel grid forms a rectangular array with dimensions $M \times N$, where $M$ is the number of rows (height) and $N$ is the number of columns (width). The origin $(0, 0)$ is typically located at the top-left corner, with $x$ increasing downward and $y$ increasing to the right.

### Pixel Neighborhoods

Understanding pixel neighborhoods is essential for image processing operations:

- **4-connected neighborhood**: Pixels sharing an edge
- **8-connected neighborhood**: Pixels sharing an edge or corner
- **Distance metrics**: Euclidean distance $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$

## Color Spaces

Color spaces provide different ways to represent color information, each optimized for specific applications or perceptual properties.

### RGB Color Space

The RGB (Red, Green, Blue) color space is the most common representation for digital displays and image sensors. Each pixel is represented by three values:

$$I_{RGB}(x, y) = [R(x, y), G(x, y), B(x, y)]$$

where each component typically ranges from 0 to 255 for 8-bit representation. RGB is an additive color model where colors are created by combining red, green, and blue light.

The conversion from RGB to grayscale can be performed using:

$$I_{gray} = 0.299R + 0.587G + 0.114B$$

This weighted combination approximates human luminance perception.

### HSV Color Space

HSV (Hue, Saturation, Value) separates color information into perceptually meaningful components:

- **Hue (H)**: The color type, represented as an angle from 0° to 360°
- **Saturation (S)**: The intensity or purity of the color, ranging from 0 to 1
- **Value (V)**: The brightness, ranging from 0 to 1

Conversion from RGB to HSV:

$$V = \max(R, G, B)$$
$$S = \begin{cases} 
\frac{V - \min(R, G, B)}{V} & \text{if } V \neq 0 \\
0 & \text{if } V = 0
\end{cases}$$

$$H = \begin{cases}
60 \times \frac{G - B}{V - \min(R, G, B)} & \text{if } V = R \\
60 \times \left(2 + \frac{B - R}{V - \min(R, G, B)}\right) & \text{if } V = G \\
60 \times \left(4 + \frac{R - G}{V - \min(R, G, B)}\right) & \text{if } V = B
\end{cases}$$

HSV is particularly useful for color-based segmentation and image manipulation tasks where separating color from intensity is beneficial.

### YCbCr Color Space

YCbCr separates luminance (Y) from chrominance (Cb, Cr) components, making it suitable for image compression:

$$\begin{bmatrix} Y \\ Cb \\ Cr \end{bmatrix} = \begin{bmatrix}
0.299 & 0.587 & 0.114 \\
-0.168736 & -0.331264 & 0.5 \\
0.5 & -0.418688 & -0.081312
\end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 128 \\ 128 \end{bmatrix}$$

The Y component represents luminance, while Cb and Cr represent blue-difference and red-difference chroma components. This separation allows for subsampling of chroma components without significant perceptual loss, as the human visual system is more sensitive to luminance changes.

### CIELAB Color Space

The CIELAB (L*a*b*) color space is designed to be perceptually uniform, meaning equal distances in the color space correspond to equal perceived color differences:

- **L***: Lightness, ranging from 0 (black) to 100 (white)
- **a***: Green-red axis, typically ranging from -128 to 127
- **b***: Blue-yellow axis, typically ranging from -128 to 127

CIELAB is device-independent and widely used in color science, image quality assessment, and applications requiring perceptually accurate color comparisons.

## Image Formats and Compression

Digital images are stored in various file formats, each with different characteristics regarding compression, quality, and metadata support.

### Lossless Formats

**PNG (Portable Network Graphics)**:
- Supports lossless compression
- Can store images with various bit depths (8, 16, 24, 32 bits per pixel)
- Supports transparency through alpha channel
- Uses DEFLATE compression algorithm

**TIFF (Tagged Image File Format)**:
- Flexible format supporting multiple compression schemes
- Can store high bit-depth images
- Supports multiple pages and extensive metadata
- Commonly used in professional photography and scientific imaging

**BMP (Bitmap)**:
- Uncompressed format, resulting in large file sizes
- Simple structure, easy to parse
- Rarely used in modern applications due to size inefficiency

### Lossy Formats

**JPEG (Joint Photographic Experts Group)**:
- Uses discrete cosine transform (DCT) for compression
- Quality controlled by quantization parameter $Q$
- Excellent compression ratios for photographic images
- Artifacts become visible at high compression ratios

The JPEG compression process involves:
1. Color space conversion to YCbCr
2. Chroma subsampling (typically 4:2:0)
3. DCT transformation of 8×8 blocks
4. Quantization using quality-dependent tables
5. Entropy coding (Huffman or arithmetic coding)

**JPEG 2000**:
- Uses wavelet transform instead of DCT
- Better compression efficiency and quality
- Supports both lossy and lossless compression
- More computationally intensive than JPEG

### Compression Ratio

The compression ratio is defined as:

$$\text{Compression Ratio} = \frac{\text{Original Size}}{\text{Compressed Size}}$$

For an $M \times N$ image with $B$ bits per pixel, the uncompressed size is:

$$\text{Uncompressed Size} = M \times N \times B \text{ bits}$$

## Spatial and Temporal Resolution

### Spatial Resolution

Spatial resolution refers to the number of pixels used to represent an image. Higher spatial resolution provides more detail but increases storage and computational requirements.

The relationship between physical dimensions and pixel count:

$$\text{Pixel Density} = \frac{\text{Total Pixels}}{\text{Physical Area}}$$

Common resolutions include:
- **VGA**: 640 × 480 pixels
- **HD**: 1920 × 1080 pixels (Full HD)
- **4K UHD**: 3840 × 2160 pixels
- **8K UHD**: 7680 × 4320 pixels

### Temporal Resolution

For video sequences, temporal resolution (frame rate) determines how many frames are captured per second. Common frame rates include:

- **24 fps**: Cinema standard
- **30 fps**: Standard video
- **60 fps**: High frame rate video
- **120+ fps**: Slow-motion capture

The Nyquist-Shannon sampling theorem applies to temporal sampling: the frame rate must be at least twice the highest frequency component in the temporal signal to avoid aliasing.

## Bit Depth and Dynamic Range

### Bit Depth

Bit depth determines the number of distinct intensity levels that can be represented:

$$\text{Number of Levels} = 2^b$$

where $b$ is the bit depth.

Common bit depths:
- **1-bit**: Binary images (black and white only)
- **8-bit**: 256 levels per channel (standard for most applications)
- **12-bit**: 4096 levels (common in medical and scientific imaging)
- **16-bit**: 65536 levels (high dynamic range applications)

### Dynamic Range

Dynamic range represents the ratio between the maximum and minimum measurable intensity:

$$\text{Dynamic Range} = 20 \log_{10}\left(\frac{I_{max}}{I_{min}}\right) \text{ dB}$$

High dynamic range (HDR) imaging captures a wider range of luminance values than standard imaging, requiring higher bit depths or specialized encoding techniques like floating-point representation or tone mapping.

### Quantization Error

Quantization introduces error when converting continuous intensity values to discrete levels. The quantization error for uniform quantization is bounded by:

$$|e| \leq \frac{\Delta}{2}$$

where $\Delta$ is the quantization step size:

$$\Delta = \frac{I_{max} - I_{min}}{2^b}$$

## Image Sampling and Quantization

### Sampling Theorem

The Nyquist-Shannon sampling theorem states that to accurately represent a continuous signal, the sampling rate must be at least twice the highest frequency component:

$$f_s \geq 2f_{max}$$

where $f_s$ is the sampling rate and $f_{max}$ is the maximum frequency.

Violation of this theorem leads to aliasing, where high-frequency components appear as low-frequency artifacts.

### Sampling Patterns

Different sampling patterns are used for various applications:

- **Rectangular sampling**: Standard grid pattern
- **Hexagonal sampling**: More efficient packing, reduces aliasing
- **Random sampling**: Used in compressed sensing applications

### Quantization Methods

**Uniform Quantization**:
- Equal step sizes across the intensity range
- Simple implementation
- May waste levels in unused intensity ranges

**Non-uniform Quantization**:
- Adaptive step sizes based on intensity distribution
- Better utilization of available levels
- More complex implementation

**Lloyd-Max Quantization**:
- Optimal quantization minimizing mean squared error
- Requires knowledge of intensity probability distribution

## Digital Image Storage

### Memory Requirements

The memory required to store an image depends on dimensions, bit depth, and number of channels:

$$\text{Memory (bytes)} = M \times N \times \frac{B}{8} \times C$$

where:
- $M$: Height in pixels
- $N$: Width in pixels
- $B$: Bits per pixel per channel
- $C$: Number of channels

### Storage Formats

**Row-major order**: Pixels stored row by row, left to right
**Column-major order**: Pixels stored column by column, top to bottom
**Interleaved format**: Color channels interleaved (RGBRGBRGB...)
**Planar format**: Each channel stored separately (RRR...GGG...BBB...)

### Metadata

Image files often contain metadata including:
- Image dimensions and bit depth
- Color space information
- Camera settings (EXIF data)
- Timestamps and geolocation
- Compression parameters

## Practical Considerations

### Color Space Selection

The choice of color space depends on the application:

- **RGB**: Display, image editing, general processing
- **HSV**: Color-based segmentation, color manipulation
- **YCbCr**: Video compression, transmission
- **CIELAB**: Color matching, quality assessment

### Bit Depth Trade-offs

Higher bit depths provide:
- Greater precision and dynamic range
- Reduced quantization artifacts
- Increased storage and computational requirements

Lower bit depths provide:
- Reduced storage requirements
- Faster processing
- Potential loss of detail in high-contrast regions

### Format Selection Guidelines

- **JPEG**: Photographic images, web applications, acceptable quality loss
- **PNG**: Graphics with sharp edges, transparency needed, lossless required
- **TIFF**: Professional photography, scientific imaging, extensive metadata needed
- **RAW**: Maximum quality, post-processing flexibility, large file sizes

## Key Takeaways

1. Digital images are discrete representations of continuous scenes, requiring careful consideration of sampling and quantization.

2. Color spaces serve different purposes: RGB for display, HSV for segmentation, YCbCr for compression, and CIELAB for perceptual uniformity.

3. Spatial resolution determines detail level, while bit depth determines intensity precision and dynamic range.

4. Image formats balance compression efficiency, quality, and feature support based on application requirements.

5. Understanding pixel fundamentals, neighborhoods, and storage formats is essential for efficient image processing algorithm design.

6. The Nyquist-Shannon sampling theorem guides proper spatial and temporal sampling to avoid aliasing artifacts.

7. Quantization introduces error that must be managed through appropriate bit depth selection.

8. Color space conversion enables specialized processing techniques optimized for specific visual properties.

9. Metadata preservation is crucial for maintaining image provenance and enabling advanced processing workflows.

10. The choice of representation (format, color space, bit depth) significantly impacts both computational efficiency and result quality in computer vision applications.
