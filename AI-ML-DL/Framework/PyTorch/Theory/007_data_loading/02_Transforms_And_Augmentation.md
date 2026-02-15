# Transforms and Augmentation

## Table of Contents

- [torchvision.transforms](#torchvisiontransforms)
- [Geometric and Color Transforms](#geometric-and-color-transforms)
- [Tensor Conversion and Normalization](#tensor-conversion-and-normalization)
- [NLP Transforms](#nlp-transforms)
- [Augmentation Pipelines and Strategies](#augmentation-pipelines-and-strategies)
- [Custom Transforms](#custom-transforms)

---

## torchvision.transforms

The **torchvision.transforms** module provides composable image transformations for computer vision pipelines. Transforms operate on PIL Images or tensors and are typically chained with **Compose**.

### Compose

**Compose** applies a sequence of transforms in order. Each transform receives the output of the previous one.

```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Geometric and Color Transforms

### Geometric Transforms

| Transform | Description | Key Parameters |
|-----------|-------------|-----------------|
| Resize | Resize image to given size | size |
| CenterCrop | Crop center region | size |
| RandomCrop | Random crop | size, padding |
| RandomResizedCrop | Random crop then resize | size, scale |
| Pad | Pad image | padding, fill, padding_mode |
| RandomRotation | Random rotation | degrees |
| RandomHorizontalFlip | Horizontal flip | p |
| RandomVerticalFlip | Vertical flip | p |
| RandomAffine | Affine transform | degrees, translate, scale |
| RandomPerspective | Perspective transform | distortion_scale, p |

```python
from PIL import Image

img = Image.new('RGB', (224, 224), color=(100, 150, 200))

resize = transforms.Resize((128, 128))
center_crop = transforms.CenterCrop(150)
random_crop = transforms.RandomCrop(100)
random_resized = transforms.RandomResizedCrop(128, scale=(0.8, 1.0))
pad = transforms.Pad(20, fill=0, padding_mode='constant')
random_rotation = transforms.RandomRotation(30)
random_hflip = transforms.RandomHorizontalFlip(p=1.0)
random_affine = transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))

resized_img = resize(img)
cropped_img = center_crop(img)
```

### Color and Intensity Transforms

| Transform | Description | Key Parameters |
|-----------|-------------|-----------------|
| ColorJitter | Random brightness, contrast, saturation, hue | brightness, contrast, saturation, hue |
| Grayscale | Convert to grayscale | num_output_channels |
| RandomGrayscale | Randomly convert to grayscale | p |
| GaussianBlur | Gaussian blur | kernel_size, sigma |
| RandomInvert | Invert colors | p |
| RandomPosterize | Reduce bits per channel | bits, p |
| RandomSolarize | Invert above threshold | threshold, p |
| RandomEqualize | Histogram equalization | p |
| RandomAutocontrast | Autocontrast | p |

```python
color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
grayscale = transforms.Grayscale(num_output_channels=3)
gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
random_invert = transforms.RandomInvert(p=1.0)
```

---

## Tensor Conversion and Normalization

### ToTensor and ToPILImage

**ToTensor** converts PIL Image or numpy array (H, W, C) to tensor (C, H, W) with values in [0, 1]. **ToPILImage** converts tensor back to PIL Image.

```python
to_tensor = transforms.ToTensor()
tensor_image = to_tensor(img)
print(tensor_image.shape)
print(tensor_image.min(), tensor_image.max())

to_pil = transforms.ToPILImage()
pil_image = to_pil(tensor_image)
```

### Normalize

**Normalize** applies channel-wise normalization: `output = (input - mean) / std`. Use ImageNet statistics for pretrained models.

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalized_tensor = normalize(tensor_image)
print(normalized_tensor.mean(dim=[1, 2]))
print(normalized_tensor.std(dim=[1, 2]))
```

### v2 Transforms

**torchvision.transforms.v2** provides a newer API with improved performance and consistency. v2 transforms support both PIL and tensor inputs natively.

```python
from torchvision.transforms import v2 as transforms_v2

v2_transform = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## NLP Transforms

### Text Cleaning and Normalization

Text preprocessing includes lowercase conversion, punctuation removal, URL removal, HTML tag stripping, and contraction expansion.

```python
import re
import string

def lowercase(text):
    return text.lower()

def remove_punctuation(text, keep_chars=""):
    translator = str.maketrans("", "", string.punctuation.replace(keep_chars, ""))
    return text.translate(translator)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)
```

### Tokenization Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Whitespace | Split on spaces | Simple English |
| Punctuation-aware | Preserve punctuation as tokens | Sentiment, parsing |
| Subword | BPE-like splitting | Rare words, multilingual |
| Character | Character-level | Morphology, spelling |
| N-gram | Word n-grams | Bag-of-words |

```python
def whitespace_tokenize(text):
    return text.split()

def punctuation_aware_tokenize(text):
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return [t for t in tokens if t.strip()]

def character_tokenize(text):
    return list(text)
```

### Vocabulary Building and Encoding

Build a vocabulary from corpus, assign indices, and encode text with special tokens (PAD, UNK, BOS, EOS).

```python
from collections import Counter

class VocabularyBuilder:
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.word_counts = Counter()

    def build_vocab(self, texts, tokenizer=None):
        tokenizer = tokenizer or str.split
        for text in texts:
            self.word_counts.update(tokenizer(text))
        for word, count in self.word_counts.most_common(self.max_vocab_size - 4):
            if count >= self.min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        return self

    def encode(self, text, tokenizer=None, max_length=None):
        tokenizer = tokenizer or str.split
        tokens = tokenizer(text)
        indices = [self.word_to_idx.get(t, self.word_to_idx['<UNK>']) for t in tokens]
        indices = [self.word_to_idx['<BOS>']] + indices + [self.word_to_idx['<EOS>']]
        if max_length:
            indices = indices[:max_length] if len(indices) > max_length else indices + [self.word_to_idx['<PAD>']] * (max_length - len(indices))
        return indices
```

### Padding and Sequence Transforms

Pad variable-length sequences to uniform length for batching.

```python
def pad_sequences(sequences, max_length=None, pad_value=0, truncate='post'):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) > max_length:
            seq = seq[:max_length] if truncate == 'post' else seq[-max_length:]
        else:
            seq = seq + [pad_value] * (max_length - len(seq))
        padded.append(seq)
    return torch.tensor(padded, dtype=torch.long)
```

---

## Augmentation Pipelines and Strategies

### Policy-Based Augmentation

Define light, medium, and heavy augmentation policies for different training stages or dataset sizes.

```python
class VisionAugmentationPipeline:
    def __init__(self, policy='medium', input_size=224):
        self.policy = policy
        self.input_size = input_size
        self.pipelines = {
            'light': transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'medium': transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'heavy': transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
```

### Text Augmentation

Apply synonym replacement, random insertion, random swap, and random deletion to increase text diversity.

```python
import random

class TextAugmentation:
    def __init__(self):
        self.synonyms = {
            'good': ['great', 'excellent', 'wonderful'],
            'bad': ['terrible', 'awful', 'horrible'],
            'big': ['large', 'huge', 'enormous'],
        }

    def synonym_replacement(self, text, n=1):
        words = text.split()
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()
            if word in self.synonyms:
                words[idx] = random.choice(self.synonyms[word])
        return ' '.join(words)

    def random_swap(self, text, n=1):
        words = text.split()
        for _ in range(n):
            if len(words) < 2:
                break
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return ' '.join(words)

    def random_deletion(self, text, p=0.1):
        words = [w for w in text.split() if random.random() > p]
        return ' '.join(words) if words else text
```

### Albumentations

**Albumentations** provides fast, GPU-friendly augmentations with consistent application to images and masks (e.g., for segmentation).

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

albumentations_pipeline = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.3),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

import numpy as np
image_np = np.array(img)
augmented = albumentations_pipeline(image=image_np)
augmented_image = augmented['image']
```

---

## Custom Transforms

### Callable Class Transforms

Custom transforms implement `__call__` to be used like functions. They can hold state (e.g., running statistics).

```python
class TransformBase:
    def __init__(self, **kwargs):
        self.params = kwargs

    def __call__(self, sample):
        raise NotImplementedError

class CustomNormalize(TransformBase):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__(mean=mean, std=std)
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.tensor(self.mean).view(-1, 1, 1) if isinstance(self.mean, (list, tuple)) else self.mean
        std = torch.tensor(self.std).view(-1, 1, 1) if isinstance(self.std, (list, tuple)) else self.std
        return (tensor - mean) / std

class RandomNoise(TransformBase):
    def __init__(self, noise_factor=0.1):
        super().__init__(noise_factor=noise_factor)
        self.noise_factor = noise_factor

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        noise = torch.randn_like(img) * self.noise_factor
        return torch.clamp(img + noise, 0, 1)
```

### Functional Transforms

Use **torchvision.transforms.functional** for fine-grained control and conditional application.

```python
from torchvision.transforms import functional as TF

def conditional_transform(img, apply_rotation=True, apply_flip=True):
    if isinstance(img, Image.Image):
        img = TF.to_tensor(img)
    if apply_rotation and random.random() > 0.5:
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
    if apply_flip and random.random() > 0.5:
        img = TF.hflip(img)
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    return img
```

### Probabilistic and Conditional Transforms

Apply transforms with given probabilities or based on data properties.

```python
class ProbabilisticTransform:
    def __init__(self, transform_dict):
        self.transform_dict = transform_dict

    def __call__(self, data):
        for transform, probability in self.transform_dict.values():
            if random.random() < probability:
                data = transform(data)
        return data

class ConditionalTransform:
    def __init__(self, brightness_threshold=0.5):
        self.brightness_threshold = brightness_threshold

    def __call__(self, tensor):
        brightness = tensor.mean()
        if brightness < self.brightness_threshold:
            return torch.clamp(tensor * 1.2 + 0.1, 0, 1)
        return torch.clamp((tensor - 0.5) * 1.1 + 0.5, 0, 1)
```

### Transform Composition Modes

Compose transforms in sequential, parallel (multiple outputs), or random-choice modes.

```python
class CompositeTransform:
    def __init__(self, transforms_list, mode='sequential'):
        self.transforms_list = transforms_list
        self.mode = mode

    def __call__(self, data):
        if self.mode == 'sequential':
            for t in self.transforms_list:
                data = t(data)
            return data
        elif self.mode == 'random_choice':
            return random.choice(self.transforms_list)(data)
        elif self.mode == 'parallel':
            return [t(data.clone() if hasattr(data, 'clone') else data) for t in self.transforms_list]
```

---

## Task-Specific Pipelines

### Classification

Training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize. Validation: Resize, CenterCrop, ToTensor, Normalize.

### Segmentation

Apply identical geometric transforms to image and mask. Use Albumentations for synchronized augmentation.

### Object Detection

Preserve bounding box coordinates when applying geometric transforms. Use libraries that support bbox-aware augmentation.
