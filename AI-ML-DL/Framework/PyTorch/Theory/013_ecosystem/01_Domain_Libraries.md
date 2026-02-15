# Domain Libraries

## Table of Contents

1. [Torchvision](#1-torchvision)
2. [Torchtext](#2-torchtext)
3. [Torchaudio](#3-torchaudio)
4. [PyTorch Geometric](#4-pytorch-geometric)

---

## 1. Torchvision

Torchvision is the official computer vision library for PyTorch, providing datasets, models, transforms, and utilities for image and video processing.

### 1.1 Datasets

Torchvision provides built-in datasets for common computer vision tasks:

| Dataset | Task | Key Parameters |
|---------|------|----------------|
| CIFAR10/100 | Classification | root, train, download, transform |
| ImageFolder | Classification | root, transform, loader |
| COCO | Detection/Segmentation | annFile, root, transforms |
| VOC | Detection/Segmentation | root, year, image_set |

```python
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
```

### 1.2 Models

The **model zoo** provides pretrained architectures for classification, detection, and segmentation:

```python
import torchvision.models as models

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, num_classes)
```

**Classification models**: ResNet, VGG, DenseNet, EfficientNet, MobileNet, Inception, GoogLeNet, ShuffleNet, RegNet

**Detection models**: Faster R-CNN, Mask R-CNN, RetinaNet, SSD

**Segmentation models**: FCN, DeepLabV3, LR-ASPP

### 1.3 Transforms

Transforms handle image preprocessing and augmentation:

```python
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 1.4 Utils and IO

```python
from torchvision.utils import make_grid, save_image, draw_bounding_boxes
from torchvision.io import read_image, write_image

image = read_image("path/to/image.jpg")
grid = make_grid(images, nrow=8, normalize=True, padding=2)
save_image(grid, "grid.png")
```

### 1.5 Feature Extraction

For transfer learning, freeze backbone layers and modify the classifier:

```python
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for name, param in model.named_parameters():
    if 'classifier' not in name and 'fc' not in name and 'heads' not in name:
        param.requires_grad = False

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
```

### 1.6 Object Detection Operations

```python
from torchvision.ops import nms, roi_align, roi_pool, box_iou

keep_indices = nms(boxes, scores, iou_threshold=0.5)
iou_matrix = box_iou(boxes1, boxes2)
roi_features = roi_align(features, roi_boxes, output_size=(7, 7))
```

---

## 2. Torchtext

Torchtext provides data loading utilities and text processing pipelines for NLP tasks.

### 2.1 Datasets

```python
from torchtext.datasets import IMDB, AG_NEWS

train_iter = IMDB(split='train')
```

### 2.2 Vocabulary

**build_vocab_from_iterator** creates vocabulary from tokenized text:

```python
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    min_freq=2,
    specials=['<unk>', '<pad>', '<bos>', '<eos>']
)
vocab.set_default_index(vocab['<unk>'])
```

### 2.3 Text Processing Pipelines

```python
def text_to_indices(text, vocab, tokenizer):
    tokens = tokenizer(text)
    return vocab(tokens)

def pad_sequences(sequences, max_length, padding_value=0):
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[:max_length])
        else:
            padded.append(seq + [padding_value] * (max_length - len(seq)))
    return torch.tensor(padded, dtype=torch.long)
```

### 2.4 Data Iterators

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(t) for t in texts)
    padded = pad_sequences(texts, max_len)
    return padded, torch.tensor(labels)

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 3. Torchaudio

Torchaudio provides audio loading, transforms, and speech processing utilities.

### 3.1 Audio Loading

```python
import torchaudio

waveform, sample_rate = torchaudio.load("audio.wav")

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

if sample_rate != target_rate:
    resampler = torchaudio.transforms.Resample(sample_rate, target_rate)
    waveform = resampler(waveform)
```

### 3.2 Transforms: MelSpectrogram and MFCC

```python
import torchaudio.transforms as T

mel_spec = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=80,
    f_min=0.0,
    f_max=8000.0
)

mfcc = T.MFCC(
    sample_rate=16000,
    n_mfcc=13,
    melkwargs={'n_fft': 1024, 'hop_length': 512, 'n_mels': 80}
)

spectrogram = mel_spec(waveform)
mfcc_features = mfcc(waveform)
```

### 3.3 Speech Processing

Common speech tasks include speech recognition, speaker recognition, and voice activity detection:

```python
spectrogram_transform = T.Spectrogram(n_fft=1024, hop_length=512)
time_stretch = T.TimeStretch(hop_length=512, n_freq=513)
pitch_shift = T.PitchShift(sample_rate=16000, n_steps=2)
```

### 3.4 Audio Augmentation

```python
import torchaudio.functional as F_audio

noise = torch.randn_like(waveform) * 0.1
augmented = waveform + noise

effects = [["speed", "1.2"]]
augmented = F_audio.apply_effects_tensor(waveform, sample_rate, effects)[0]
```

---

## 4. PyTorch Geometric

PyTorch Geometric (PyG) extends PyTorch for graph neural networks and geometric deep learning.

### 4.1 Graph Data Structure

The **Data** object represents a single graph:

```python
from torch_geometric.data import Data

x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
edge_attr = torch.tensor([[1.0], [1.0], [2.0], [2.0]], dtype=torch.float)
y = torch.tensor([1], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

**edge_index** format: `[2, num_edges]` where row 0 is source nodes, row 1 is target nodes.

### 4.2 GCN (Graph Convolutional Network)

```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 4.3 GAT (Graph Attention Network)

```python
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 4.4 Message Passing

Custom message passing layers extend **MessagePassing**:

```python
from torch_geometric.nn import MessagePassing

class CustomMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.linear(x_j)

    def update(self, aggr_out):
        return aggr_out
```

### 4.5 Graph Classification

Use **global pooling** to obtain graph-level representations:

```python
from torch_geometric.nn import global_mean_pool, global_max_pool

class GraphClassifier(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(num_features, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
        ])
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return F.log_softmax(self.classifier(x), dim=1)
```

### 4.6 Node Classification

```python
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

### 4.7 Batching Graphs

```python
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    out = model(batch.x, batch.edge_index, batch.batch)
```
