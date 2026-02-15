# Generative Models and Style Transfer

## Table of Contents

- [Generative Adversarial Networks](#generative-adversarial-networks)
- [Variational Autoencoders](#variational-autoencoders)
- [Neural Style Transfer](#neural-style-transfer)
- [Image Super-Resolution](#image-super-resolution)

---

## Generative Adversarial Networks

GANs consist of a **generator** that produces fake samples and a **discriminator** that distinguishes real from fake. Training is **adversarial**: the generator tries to fool the discriminator, which tries to correctly classify.

### Generator and Discriminator Architecture

The generator maps **latent noise** to images. The discriminator maps images to a **real/fake probability**.

```python
import torch
import torch.nn as nn

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        self.img_size = int(torch.tensor(img_shape).prod())
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), *self.img_shape)

class SimpleDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_size = int(torch.tensor(img_shape).prod())
        self.model = nn.Sequential(
            nn.Linear(self.img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))
```

### Adversarial Loss

Vanilla GAN uses **binary cross-entropy**. Discriminator maximizes correct classification; generator minimizes discriminator confidence on fakes.

```python
def vanilla_gan_loss(real_output, fake_output):
    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)
    d_loss_real = F.binary_cross_entropy(real_output, real_labels)
    d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    g_loss = F.binary_cross_entropy(fake_output, real_labels)
    return d_loss, g_loss
```

### DCGAN

**Deep Convolutional GAN** uses transposed convolutions for the generator and strided convolutions for the discriminator. Guidelines: BatchNorm (except in D output), LeakyReLU in D, no pooling (use strided conv).

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3):
        super().__init__()
        self.init_size = 4
        self.l1 = nn.Linear(latent_dim, 512 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z).view(z.size(0), 512, self.init_size, self.init_size)
        return self.conv_blocks(out)
```

### Training Loop

```python
def train_gan_step(generator, discriminator, g_optimizer, d_optimizer, real_imgs, latent_dim, device):
    batch_size = real_imgs.size(0)
    real_imgs = real_imgs.to(device)
    z = torch.randn(batch_size, latent_dim).to(device)
    fake_imgs = generator(z)

    d_optimizer.zero_grad()
    real_output = discriminator(real_imgs)
    fake_output = discriminator(fake_imgs.detach())
    d_loss, _ = vanilla_gan_loss(real_output, fake_output)
    d_loss.backward()
    d_optimizer.step()

    g_optimizer.zero_grad()
    fake_output = discriminator(fake_imgs)
    _, g_loss = vanilla_gan_loss(real_output, fake_output)
    g_loss.backward()
    g_optimizer.step()

    return d_loss.item(), g_loss.item()
```

### Mode Collapse

**Mode collapse** occurs when the generator produces limited diversity. Mitigations: spectral normalization, gradient penalty (WGAN-GP), different learning rates for G and D, label smoothing.

---

## Variational Autoencoders

VAEs learn a **latent distribution** and reconstruct inputs. The **encoder** outputs mean and log-variance; the **decoder** reconstructs from sampled latent codes. The **reparameterization trick** enables backpropagation through sampling.

### Encoder, Decoder, and Reparameterization Trick

```python
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(inplace=True)])
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### KL Divergence and ELBO

The VAE loss is **ELBO** (Evidence Lower Bound): reconstruction loss + KL divergence to the prior (standard normal).

```python
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

### Convolutional VAE

```python
class ConvVAEEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.flatten_size = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
```

### Beta-VAE

**Beta-VAE** increases the weight on the KL term (beta > 1) to encourage **disentangled** latent factors.

```python
total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, beta=4.0)
```

---

## Neural Style Transfer

Neural style transfer combines **content** from one image with **style** from another. Uses a pretrained VGG to extract features. **Content loss** is MSE between feature maps. **Style loss** uses the **Gram matrix** to capture texture statistics.

### Content Loss and Style Loss via Gram Matrix

```python
def content_loss(generated_features, content_features):
    return F.mse_loss(generated_features, content_features)

def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, -1)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (channels * height * width)

def style_loss(generated_features, style_features):
    return F.mse_loss(gram_matrix(generated_features), gram_matrix(style_features))
```

### Total Variation Loss

**Total variation loss** encourages spatial smoothness and reduces noise.

```python
def total_variation_loss(image):
    tv_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return tv_h + tv_w
```

### Gatys Optimization-Based Transfer

Optimize the generated image directly. Extract content features from content image and style features from style image. Initialize with content image and minimize weighted sum of content loss, style loss, and TV loss.

```python
def style_transfer_step(generated_image, content_features, style_features, feature_extractor,
                       content_weight=1.0, style_weight=1000.0, tv_weight=1e-4):
    gen_features = feature_extractor(generated_image)
    content_loss_val = content_loss(gen_features['conv4_2'], content_features['conv4_2'])
    style_loss_val = sum(style_loss(gen_features[k], style_features[k]) for k in style_features)
    tv_loss_val = total_variation_loss(generated_image)
    return content_weight * content_loss_val + style_weight * style_loss_val + tv_weight * tv_loss_val
```

### Fast Neural Style Transfer

Train a **transformation network** that maps content images to stylized outputs. Use perceptual loss (content + style) for training. Inference is a single forward pass.

---

## Image Super-Resolution

Super-resolution upsamples low-resolution images. Key approaches: **SRCNN**, **sub-pixel convolution**, **perceptual loss**, and **SRGAN**.

### SRCNN

**SRCNN** uses a three-layer CNN. Input is bicubic-upsampled LR image. Learns mapping to HR.

```python
class SRCNN(nn.Module):
    def __init__(self, num_channels=3, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
```

### Sub-Pixel Convolution

**Pixel shuffle** (sub-pixel convolution) learns upsampling filters. More efficient than transposed convolution.

```python
class SubPixelCNN(nn.Module):
    def __init__(self, num_channels=3, upscale_factor=2, num_features=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_features, 5, padding=2)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv3 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_up = nn.Conv2d(num_features, num_channels * (upscale_factor ** 2), 3, padding=1)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv_up(x)
        x = F.pixel_shuffle(x, self.upscale_factor)
        return x
```

### Perceptual Loss

**Perceptual loss** uses VGG features instead of pixel-wise MSE. Produces more realistic textures.

```python
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()
        vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layers = nn.ModuleDict()
        layer_indices = {'conv1_2': 3, 'conv2_2': 8, 'conv3_3': 17, 'conv4_3': 26}
        prev = 0
        for name, idx in layer_indices.items():
            self.layers[name] = nn.Sequential(*list(vgg.children())[prev:idx+1])
            prev = idx + 1
        for p in self.layers.parameters():
            p.requires_grad = False
        self.weights = layer_weights or {'conv3_3': 1.0, 'conv4_3': 1.0}

    def forward(self, sr, hr):
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        loss = 0
        for name, weight in self.weights.items():
            sr_f = self.layers[name](sr_norm)
            hr_f = self.layers[name](hr_norm)
            loss += weight * F.mse_loss(sr_f, hr_f)
        return loss
```

### SRGAN

**SRGAN** uses a GAN: generator produces SR images, discriminator judges realism. Loss combines pixel loss, perceptual loss, and adversarial loss.

### Loss Comparison

| Loss | Pros | Cons |
|------|------|------|
| MSE/L1 | Simple, stable | Blurry results |
| Charbonnier | Smooth L1 variant | Still pixel-based |
| SSIM | Structural similarity | More complex |
| Perceptual | Realistic textures | Requires VGG |
| Adversarial | Sharp, realistic | Training instability |
