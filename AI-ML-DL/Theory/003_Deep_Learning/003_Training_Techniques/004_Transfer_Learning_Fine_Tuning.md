# Transfer Learning and Fine-Tuning

## Table of Contents

1. [Introduction](#introduction)
2. [Transfer Learning Concepts](#transfer-learning-concepts)
3. [Pre-Trained Models](#pre-trained-models)
4. [Feature Extraction](#feature-extraction)
5. [Fine-Tuning Strategies](#fine-tuning-strategies)
6. [Domain Adaptation](#domain-adaptation)
7. [Few-Shot Transfer Learning](#few-shot-transfer-learning)
8. [Multi-Task Learning](#multi-task-learning)
9. [Practical Considerations](#practical-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction

Transfer learning leverages knowledge from one task to improve performance on another, enabling effective learning with limited data. Fine-tuning adapts pre-trained models to new tasks, making it possible to achieve good performance quickly and with less data.

This chapter covers transfer learning strategies, from feature extraction to fine-tuning, domain adaptation, and few-shot learning, examining how to effectively leverage pre-trained models.

## Transfer Learning Concepts

### Definition

Transfer learning uses knowledge from source task to improve learning on target task:

- **Source Domain**: Original task/data
- **Target Domain**: New task/data
- **Transfer**: Knowledge transfer between domains

### Types of Transfer

1. **Inductive Transfer**: Different tasks, same domain
2. **Transductive Transfer**: Same task, different domains
3. **Unsupervised Transfer**: Source task unlabeled
4. **Multi-Task Learning**: Learn multiple tasks simultaneously

### When Transfer Learning Helps

- Limited target data
- Source and target tasks related
- Source model learned useful features
- Computational resources limited

### Transfer Learning Scenarios

**Scenario 1**: Large source dataset, small target dataset
**Scenario 2**: Source and target similar (e.g., ImageNet → medical images)
**Scenario 3**: Source and target different (e.g., images → text)

## Pre-Trained Models

### Image Classification Models

**ResNet**: Deep residual networks
- ResNet-50, ResNet-101, ResNet-152
- Trained on ImageNet
- Good feature extractors

**VGG**: Simple architecture
- VGG-16, VGG-19
- Good for feature extraction

**EfficientNet**: Efficient architectures
- EfficientNet-B0 to B7
- Good accuracy-efficiency trade-off

**Vision Transformers**: Transformer-based
- ViT-Base, ViT-Large
- State-of-the-art performance

### Language Models

**BERT**: Bidirectional encoder
- BERT-Base, BERT-Large
- Pre-trained on large text corpora
- Good for NLP tasks

**GPT**: Autoregressive language model
- GPT-2, GPT-3
- Good for generation tasks

**RoBERTa**: Improved BERT
- Better pre-training
- Strong performance

### Where to Get Pre-Trained Models

- PyTorch: `torchvision.models`, `transformers`
- TensorFlow: `tf.keras.applications`, `tensorflow_hub`
- Hugging Face: `transformers` library
- Model Zoos: Various repositories

## Feature Extraction

Feature extraction uses pre-trained model as fixed feature extractor.

### Procedure

1. **Remove Classifier**: Remove final classification layers
2. **Freeze Backbone**: Set requires_grad=False for feature extractor
3. **Add New Classifier**: Train new classifier on extracted features
4. **Train**: Only train new classifier

### Advantages

- Fast training (only classifier)
- Less memory (no gradients for backbone)
- Good when target data small
- Prevents overfitting

### Implementation

```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Train only classifier
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### When to Use

- Very small target dataset
- Target task very different
- Limited computational resources
- Quick prototyping

## Fine-Tuning Strategies

Fine-tuning updates pre-trained model parameters for target task.

### Full Fine-Tuning

Update all parameters:

```python
# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Use smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

**Advantages**:
- Maximum adaptation
- Best performance potential

**Disadvantages**:
- More parameters to train
- Risk of overfitting
- Requires more data

### Partial Fine-Tuning

Update only later layers:

```python
# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Fine-tune later layers
optimizer = torch.optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

**Advantages**:
- Less overfitting risk
- Faster training
- Preserves low-level features

### Differential Learning Rates

Use different learning rates for different layers:

```python
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

**Rationale**:
- Early layers: Small LR (preserve features)
- Later layers: Larger LR (adapt to task)

### Gradual Unfreezing

Progressively unfreeze layers:

1. Start: Freeze all, train classifier
2. Unfreeze last layer, train
3. Unfreeze more layers, train
4. Continue until desired

**Benefits**:
- Stable training
- Prevents catastrophic forgetting
- Better convergence

## Domain Adaptation

Domain adaptation transfers from source domain to different target domain.

### Problem

- **Source Domain**: Labeled data (e.g., natural images)
- **Target Domain**: Unlabeled or different distribution (e.g., medical images)
- **Goal**: Learn on target domain using source knowledge

### Approaches

**1. Domain-Adversarial Training**:
- Train domain discriminator
- Adversarially align domains
- Learn domain-invariant features

**2. Domain-Specific BatchNorm**:
- Separate BatchNorm for source/target
- Normalize per domain
- Better domain adaptation

**3. Pseudo-Labeling**:
- Predict on target domain
- Use confident predictions as labels
- Iteratively refine

**4. Domain Randomization**:
- Augment source domain
- Increase diversity
- Better generalization

### Implementation Example

```python
# Domain-adversarial approach
class DomainAdversarial(nn.Module):
    def __init__(self, feature_extractor, classifier, domain_classifier):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.domain_classifier = domain_classifier
    
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        
        # Reverse gradient for domain classifier
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)
        
        class_pred = self.classifier(features)
        return class_pred, domain_pred
```

## Few-Shot Transfer Learning

Few-shot learning adapts with very few examples per class.

### Problem Setting

- **N-way K-shot**: N classes, K examples per class
- **Support Set**: Few labeled examples
- **Query Set**: Examples to classify
- **Goal**: Learn to classify with minimal data

### Approaches

**1. Metric Learning**:
- Learn embedding space
- Compare query to support examples
- Nearest neighbor classification

**2. Meta-Learning**:
- Learn to learn quickly
- MAML, Prototypical Networks
- Adapt with few gradient steps

**3. Fine-Tuning**:
- Fine-tune on support set
- Careful regularization
- Prevent overfitting

### Prototypical Networks

Learn prototype per class:

$$\mathbf{c}_k = \frac{1}{|\mathcal{S}_k|} \sum_{(\mathbf{x}_i, y_i) \in \mathcal{S}_k} f_\phi(\mathbf{x}_i)$$

Classify by distance to prototypes:

$$p(y=k|\mathbf{x}) = \frac{\exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_k))}{\sum_{k'} \exp(-d(f_\phi(\mathbf{x}), \mathbf{c}_{k'}))}$$

## Multi-Task Learning

Learn multiple related tasks simultaneously.

### Hard Parameter Sharing

Share some layers across tasks:

```python
class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_backbone = nn.Sequential(...)
        self.task1_head = nn.Linear(512, num_classes1)
        self.task2_head = nn.Linear(512, num_classes2)
    
    def forward(self, x):
        features = self.shared_backbone(x)
        out1 = self.task1_head(features)
        out2 = self.task2_head(features)
        return out1, out2
```

### Soft Parameter Sharing

Separate models with regularization:

$$\mathcal{L}_{\text{total}} = \sum_t \mathcal{L}_t + \lambda \sum_i ||\theta_i^{(1)} - \theta_i^{(2)}||^2$$

### Benefits

- Shared representations
- Better generalization
- Data efficiency
- Transfer between tasks

## Practical Considerations

### Choosing Strategy

**Feature Extraction**:
- Very small dataset (< 1000 examples)
- Different task
- Limited resources

**Fine-Tuning**:
- Moderate dataset (1000-10000 examples)
- Similar task
- Want best performance

**Full Fine-Tuning**:
- Large dataset (> 10000 examples)
- Similar task
- Sufficient resources

### Learning Rate Selection

- **Feature Extraction**: $10^{-3}$ to $10^{-2}$
- **Fine-Tuning**: $10^{-5}$ to $10^{-4}$
- **New Layers**: $10^{-3}$ to $10^{-2}$

### Data Augmentation

Important for small datasets:
- Standard augmentations
- Domain-specific augmentations
- Mixup, CutMix

### Regularization

Prevent overfitting:
- Dropout
- Weight decay
- Early stopping

### Evaluation

- Use validation set
- Monitor overfitting
- Compare with/without transfer
- Measure improvement

## Key Takeaways

1. **Transfer Learning**: Leverages knowledge from source task to improve target task performance, especially effective with limited target data.

2. **Pre-Trained Models**: Models trained on large datasets (ImageNet, large text corpora) provide excellent starting points for transfer learning.

3. **Feature Extraction**: Uses pre-trained model as fixed feature extractor, training only new classifier, ideal for very small datasets.

4. **Fine-Tuning**: Updates pre-trained model parameters, with strategies ranging from full fine-tuning to partial fine-tuning of later layers.

5. **Differential Learning Rates**: Uses smaller learning rates for early layers and larger rates for later layers, balancing feature preservation and adaptation.

6. **Domain Adaptation**: Transfers from source domain to different target domain, using techniques like domain-adversarial training and domain-specific normalization.

7. **Few-Shot Learning**: Adapts with very few examples using metric learning, meta-learning, or careful fine-tuning with strong regularization.

8. **Multi-Task Learning**: Learns multiple tasks simultaneously with shared representations, improving generalization and data efficiency.

9. **Strategy Selection**: Choose feature extraction for very small datasets, fine-tuning for moderate datasets, with learning rates and regularization adjusted accordingly.

10. **Practical Success**: Transfer learning enables achieving good performance quickly with limited data, making it essential for many real-world applications.
