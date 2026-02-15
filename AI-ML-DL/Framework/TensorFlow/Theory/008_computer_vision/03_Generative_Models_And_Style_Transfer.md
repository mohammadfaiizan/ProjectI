# Generative Models and Style Transfer

## Table of Contents

1. [Generative Adversarial Networks (GANs)](#1-generative-adversarial-networks-gans)
2. [Variational Autoencoders (VAEs)](#2-variational-autoencoders-vaes)
3. [Neural Style Transfer](#3-neural-style-transfer)
4. [Image Super-Resolution](#4-image-super-resolution)
5. [Comparison and Use Cases](#5-comparison-and-use-cases)

---

## 1. Generative Adversarial Networks (GANs)

**GANs** consist of two networks: a **generator** that creates fake samples and a **discriminator** that distinguishes real from fake. They are trained adversarially.

### Architecture

| Component | Role |
|-----------|------|
| Generator | Maps noise z to image; tries to fool discriminator |
| Discriminator | Classifies real vs fake; tries to detect fakes |

### DCGAN (Deep Convolutional GAN)

- Generator: Dense -> Reshape -> Conv2DTranspose layers.
- Discriminator: Conv2D -> LeakyReLU -> Dense.
- Use **BatchNorm** in generator, **Dropout** in discriminator.
- Use **LeakyReLU** in discriminator.

```python
def build_generator(latent_dim=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, 5, padding='same', activation='tanh')
    ])
    return model

def build_discriminator(img_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Conv2D(128, 5, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
```

### Training Loop

1. Train discriminator on real + fake batches.
2. Train generator (discriminator frozen) to maximize discriminator's fake score.
3. Use **binary crossentropy** for both.

---

## 2. Variational Autoencoders (VAEs)

**VAEs** learn a latent space by encoding inputs to a distribution (mean, log_var), sampling via **reparameterization**, and decoding. They optimize reconstruction + **KL divergence** to regularize the latent space.

### Reparameterization Trick

Sample from N(mean, var) as: `z = mean + exp(0.5 * log_var) * epsilon`, where epsilon ~ N(0,1). This allows gradients to flow through the sampling step.

```python
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
```

### VAE Loss

- **Reconstruction loss**: Binary crossentropy or MSE between input and reconstruction.
- **KL loss**: -0.5 * sum(1 + log_var - mean^2 - exp(log_var)).
- Total: recon_loss + kl_loss.

```python
def vae_loss(x, x_recon, z_mean, z_log_var):
    recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return recon_loss + kl_loss
```

### Encoder-Decoder Structure

```python
# Encoder outputs (z_mean, z_log_var)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)
z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
out = decoder(z)
```

---

## 3. Neural Style Transfer

**Style transfer** combines the **content** of one image with the **style** of another using deep feature representations.

### Content and Style Loss

- **Content loss**: MSE between feature maps of content image and generated image (usually from deep layer).
- **Style loss**: MSE between **Gram matrices** of feature maps (captures texture/style).

### Gram Matrix

The Gram matrix captures correlations between filter responses (style). For feature map of shape (H, W, C), flatten to (H*W, C) and compute G = F^T F / (H*W).

```python
def gram_matrix(x):
    x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
    n = tf.cast(tf.shape(x)[1], tf.float32)
    gram = tf.matmul(x, x, transpose_a=True) / n
    return gram

def style_loss(style, target):
    g_style = gram_matrix(style)
    g_target = gram_matrix(target)
    return tf.reduce_mean(tf.square(g_style - g_target))
```

### Feature Extractor

Use a pretrained network (e.g., VGG19) and extract activations from selected layers. Content from deep layers; style from multiple layers.

```python
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
outputs = [vgg.get_layer(name).output for name in layer_names]
extractor = tf.keras.Model(vgg.input, outputs)
```

### Total Loss

`L = alpha * content_loss + beta * style_loss`, where alpha and beta control the balance.

---

## 4. Image Super-Resolution

**Super-resolution** increases image resolution, recovering high-frequency details from low-resolution input.

### SRCNN (Super-Resolution CNN)

- Three conv layers: patch extraction, non-linear mapping, reconstruction.
- Input: low-res; output: high-res (e.g., 2x or 4x).

### Sub-Pixel Convolution (ESPCN)

- Use **depth_to_space** to upsample: increase channels by scale^2, then rearrange to spatial.
- More efficient than transposed convolution for upsampling.

```python
def subpixel_upsample(x, scale=2):
    x = tf.keras.layers.Conv2D(x.shape[-1] * scale * scale, 3, padding='same')(x)
    return tf.nn.depth_to_space(x, scale)
```

### SRCNN-Style Architecture

```python
inp = tf.keras.layers.Input(shape=(32, 32, 3))
x = tf.keras.layers.Conv2D(64, 9, padding='same', activation='relu')(inp)
x = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(3 * 4, 5, padding='same')(x)
x = tf.nn.depth_to_space(x, 2)
model = tf.keras.Model(inp, x)
```

### Loss Functions

- **MSE** or **MAE** on pixel space.
- **Perceptual loss**: MSE on VGG features (better perceptual quality).

---

## 5. Comparison and Use Cases

| Model | Use Case | Pros | Cons |
|-------|----------|-----|-----|
| GAN | Image generation, augmentation | Sharp, diverse samples | Training instability, mode collapse |
| VAE | Compression, interpolation, anomaly detection | Stable, interpretable latent space | Blurry reconstructions |
| Style transfer | Artistic rendering | Single forward pass at inference | Requires content + style images |
| Super-resolution | Upscaling, restoration | Direct mapping | Limited by information in LR input |

### When to Use

- **GAN**: When you need high-quality, diverse synthetic images.
- **VAE**: When you need a continuous latent space for interpolation or anomaly detection.
- **Style transfer**: For artistic or creative applications.
- **Super-resolution**: For enhancing resolution of existing images.
