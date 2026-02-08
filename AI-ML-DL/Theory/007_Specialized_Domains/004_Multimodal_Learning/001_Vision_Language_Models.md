# Vision-Language Models

## Table of Contents

1. [Introduction to Vision-Language Learning](#introduction-to-vision-language-learning)
2. [CLIP: Contrastive Language-Image Pre-training](#clip-contrastive-language-image-pre-training)
3. [ALIGN: Large-Scale Image-Text Alignment](#align-large-scale-image-text-alignment)
4. [Flamingo: Few-Shot Learning](#flamingo-few-shot-learning)
5. [Visual Grounding](#visual-grounding)
6. [Image-Text Alignment](#image-text-alignment)
7. [Contrastive Learning for Multimodal](#contrastive-learning-for-multimodal)
8. [Zero-Shot Transfer](#zero-shot-transfer)
9. [Architectures and Training](#architectures-and-training)
10. [Key Takeaways](#key-takeaways)

---

## Introduction to Vision-Language Learning

Vision-language models learn joint representations of images and text, enabling tasks like image-text retrieval, visual question answering, and image captioning.

### Multimodal Learning Challenges

1. **Modality Gap**: Images and text have different representations
2. **Alignment**: Matching corresponding image-text pairs
3. **Scale**: Requires large-scale datasets
4. **Generalization**: Transfer to unseen combinations

### Key Tasks

**Image-Text Retrieval**: Find images matching text queries or vice versa

**Visual Question Answering (VQA)**: Answer questions about images

**Image Captioning**: Generate text descriptions of images

**Visual Grounding**: Localize objects mentioned in text

**Zero-Shot Classification**: Classify images using text descriptions

### Approaches

1. **Contrastive Learning**: Learn aligned embeddings (CLIP, ALIGN)
2. **Generative Models**: Generate text from images or vice versa
3. **Attention Mechanisms**: Cross-modal attention for alignment
4. **Pre-training**: Large-scale pre-training on image-text pairs

---

## CLIP: Contrastive Language-Image Pre-training

CLIP learns visual representations from natural language supervision using contrastive learning on large-scale image-text pairs.

### Architecture

**Image Encoder**: $f_I(\mathbf{x}_I) \in \mathbb{R}^{d}$ encodes images

**Text Encoder**: $f_T(\mathbf{x}_T) \in \mathbb{R}^{d}$ encodes text

**Shared Embedding Space**: Both encoders map to same $d$-dimensional space

### Contrastive Objective

**Positive Pairs**: Matching image-text pairs $(\mathbf{x}_I, \mathbf{x}_T)$

**Negative Pairs**: Non-matching pairs

**Similarity**: Cosine similarity in embedding space:

$$s(\mathbf{x}_I, \mathbf{x}_T) = \frac{f_I(\mathbf{x}_I)^T f_T(\mathbf{x}_T)}{\|f_I(\mathbf{x}_I)\| \|f_T(\mathbf{x}_T)\|}$$

### Loss Function

**Symmetric Contrastive Loss**:

$$\mathcal{L}_I = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s(\mathbf{x}_I^{(i)}, \mathbf{x}_T^{(i)}) / \tau)}{\sum_{j=1}^{N} \exp(s(\mathbf{x}_I^{(i)}, \mathbf{x}_T^{(j)}) / \tau)}$$

$$\mathcal{L}_T = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(s(\mathbf{x}_I^{(i)}, \mathbf{x}_T^{(i)}) / \tau)}{\sum_{j=1}^{N} \exp(s(\mathbf{x}_I^{(j)}, \mathbf{x}_T^{(i)}) / \tau)}$$

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_I + \mathcal{L}_T)$$

where $\tau$ is temperature parameter.

### Training Data

**Web-Scale**: 400M image-text pairs from internet

**Diversity**: Natural language descriptions, not curated labels

**Scale**: Key to CLIP's success

### Zero-Shot Classification

**Text Prompts**: "a photo of a {class}" for each class

**Classification**: 

$$p(y | \mathbf{x}_I) = \frac{\exp(s(\mathbf{x}_I, \mathbf{t}_y) / \tau)}{\sum_{c=1}^{C} \exp(s(\mathbf{x}_I, \mathbf{t}_c) / \tau)}$$

where $\mathbf{t}_c$ is text prompt for class $c$.

### Advantages

- **Zero-Shot**: No task-specific training
- **Flexible**: Works with any text description
- **Scalable**: Benefits from more data
- **Transferable**: Strong performance on downstream tasks

### Limitations

- **Fine-Grained**: Struggles with fine-grained classification
- **Abstract Concepts**: Difficulty with abstract reasoning
- **Bias**: Inherits biases from web data

---

## ALIGN: Large-Scale Image-Text Alignment

ALIGN scales contrastive learning to over 1 billion image-text pairs, demonstrating the importance of scale for vision-language learning.

### Scale-Up Strategy

**Data**: 1.8B image-text pairs from web

**Simple Architecture**: Similar to CLIP but larger scale

**Efficient Training**: Optimized for large-scale training

### Architecture

**Image Encoder**: EfficientNet-B7 or Vision Transformer

**Text Encoder**: BERT or Transformer

**Contrastive Loss**: Same as CLIP

### Key Differences from CLIP

1. **Scale**: 4.5× more data
2. **Architecture**: Larger models
3. **Training**: More efficient optimization

### Performance

**Image-Text Retrieval**: Strong improvements over CLIP

**Zero-Shot Classification**: Better on many benchmarks

**Transfer Learning**: Excellent performance on downstream tasks

### Lessons

- **Scale Matters**: More data improves performance
- **Simple Works**: Complex architectures not necessary
- **Efficiency**: Need efficient training for scale

---

## Flamingo: Few-Shot Learning

Flamingo enables few-shot learning by interleaving pretrained vision and language models with cross-attention.

### Architecture

**Vision Encoder**: Frozen pretrained vision model (e.g., CLIP)

**Language Model**: Frozen pretrained language model (e.g., Chinchilla)

**Gated Cross-Attention**: Connects vision and language

**Perceiver Resampler**: Processes variable number of images

### Few-Shot Learning

**In-Context Learning**: Provide examples in context:

```
Image 1: "a cat"
Image 2: "a dog"
Image 3: ?
```

Model predicts "a bird" based on examples.

### Training

**Interleaved Data**: Mix images and text in sequences

**Cross-Attention**: Vision features attend to language and vice versa

**Frozen Backbones**: Only train cross-attention layers

### Capabilities

- **Few-Shot VQA**: Answer questions with few examples
- **Image Captioning**: Generate captions with examples
- **Visual Dialog**: Multi-turn conversations about images

### Advantages

- **Efficient**: Only trains small cross-attention layers
- **Flexible**: Handles variable numbers of images
- **Few-Shot**: Learns from few examples

---

## Visual Grounding

Visual grounding localizes objects or regions in images based on text descriptions.

### Problem Formulation

**Input**: Image $\mathbf{x}_I$ and text query $\mathbf{x}_T$

**Output**: Bounding box $b = (x, y, w, h)$ or segmentation mask

### Approaches

**Two-Stage**: 
1. Generate region proposals
2. Match regions to text

**One-Stage**:
- Directly predict boxes from image-text

### Attention-Based Grounding

**Cross-Modal Attention**: 

$$\alpha_{ij} = \frac{\exp(\mathbf{v}_i^T \mathbf{t}_j)}{\sum_{k} \exp(\mathbf{v}_i^T \mathbf{t}_k)}$$

where $\mathbf{v}_i$ are visual features and $\mathbf{t}_j$ are text features.

**Attended Features**:

$$\mathbf{v}'_i = \sum_{j} \alpha_{ij} \mathbf{t}_j$$

### Phrase Grounding

**Phrase Localization**: Localize noun phrases in text

**Example**: "the red car" → bounding box around red car

### Referring Expression Comprehension

**Task**: Given referring expression, find object

**Example**: "the dog on the left" → localize specific dog

### Evaluation Metrics

**IoU**: Intersection over Union for bounding boxes

**Accuracy**: Percentage of correct localizations

**mAP**: Mean Average Precision

---

## Image-Text Alignment

Image-text alignment ensures corresponding image and text features are close in embedding space.

### Alignment Objectives

**Contrastive Loss**: Push positive pairs together, negative pairs apart

**Triplet Loss**: 

$$\mathcal{L} = \max(0, d(\mathbf{x}_I, \mathbf{x}_T^+) - d(\mathbf{x}_I, \mathbf{x}_T^-) + m)$$

**Cross-Modal Retrieval**: Retrieve images from text or vice versa

### Fine-Grained Alignment

**Region-Word Alignment**: Align image regions to words

**Attention Maps**: Visualize which regions correspond to which words

**Phrase-Region Matching**: Match phrases to image regions

### Cross-Modal Attention

**Image-to-Text**: Image features attend to text:

$$\mathbf{t}'_j = \sum_{i} \alpha_{ij} \mathbf{v}_i$$

**Text-to-Image**: Text features attend to image:

$$\mathbf{v}'_i = \sum_{j} \alpha_{ij} \mathbf{t}_j$$

### Alignment Quality

**Retrieval Performance**: How well can we retrieve matching pairs?

**Attention Visualization**: Do attention maps make sense?

**Downstream Tasks**: Does alignment help downstream tasks?

---

## Contrastive Learning for Multimodal

Contrastive learning is the foundation of modern vision-language models.

### Contrastive Framework

**Positive Pairs**: Matching image-text $(\mathbf{x}_I, \mathbf{x}_T)$

**Negative Pairs**: Non-matching pairs

**Objective**: Maximize similarity for positives, minimize for negatives

### InfoNCE Loss

**Formulation**:

$$\mathcal{L} = -\log \frac{\exp(s(\mathbf{x}_I, \mathbf{x}_T^+) / \tau)}{\exp(s(\mathbf{x}_I, \mathbf{x}_T^+) / \tau) + \sum_{j=1}^{N-1} \exp(s(\mathbf{x}_I, \mathbf{x}_T_j^-) / \tau)}$$

**Interpretation**: Maximize mutual information between image and text

### Hard Negative Mining

**Random Negatives**: Sample random non-matching pairs

**Hard Negatives**: Select difficult negatives:

$$\mathbf{x}_T^- = \arg\max_{\mathbf{x}_T \neq \mathbf{x}_T^+} s(\mathbf{x}_I, \mathbf{x}_T)$$

**Benefits**: Better learning signal

### Temperature Scaling

**Temperature $\tau$**: Controls concentration of distribution

**Low $\tau$**: Sharper distribution, harder negatives

**High $\tau$**: Softer distribution, easier negatives

**Typical**: $\tau \in [0.01, 0.1]$

### Multi-Positive Contrastive Learning

**Multiple Positives**: One image can match multiple texts

**Loss**:

$$\mathcal{L} = -\log \frac{\sum_{i \in \mathcal{P}} \exp(s(\mathbf{x}_I, \mathbf{x}_T^{(i)}) / \tau)}{\sum_{i \in \mathcal{P}} \exp(s(\mathbf{x}_I, \mathbf{x}_T^{(i)}) / \tau) + \sum_{j \in \mathcal{N}} \exp(s(\mathbf{x}_I, \mathbf{x}_T^{(j)}) / \tau)}$$

---

## Zero-Shot Transfer

Zero-shot transfer enables models to perform tasks without task-specific training.

### Zero-Shot Classification

**Text Prompts**: Create prompts for each class

**Classification**: Use similarity to prompts

**Example**: "a photo of a {class}" for ImageNet classes

### Zero-Shot Retrieval

**Image-to-Text**: Retrieve text from image query

**Text-to-Image**: Retrieve image from text query

**No Training**: Directly use learned embeddings

### Prompt Engineering

**Template Design**: "a photo of a {class}" vs "{class}"

**Ensemble**: Average multiple prompts

**Learned Prompts**: Learn optimal prompts (CoOp, CoCoOp)

### Few-Shot Adaptation

**Linear Probe**: Train linear classifier on few examples

**Prompt Tuning**: Learn prompts from few examples

**Adapter**: Add small adapter layers

### Evaluation

**Zero-Shot Accuracy**: Performance without training

**Few-Shot Accuracy**: Performance with few examples

**Transfer Learning**: Performance on downstream tasks

---

## Architectures and Training

### Image Encoders

**CNN-Based**: ResNet, EfficientNet

**Vision Transformer**: ViT, DeiT

**Hybrid**: CNN + Transformer

### Text Encoders

**BERT**: Bidirectional encoder

**GPT**: Autoregressive decoder

**T5**: Encoder-decoder

### Joint Architectures

**Dual Encoder**: Separate encoders, contrastive loss (CLIP)

**Cross-Attention**: Cross-modal attention (Flamingo)

**Fusion**: Combine features before prediction

### Training Strategies

**Pre-training**: Large-scale image-text pairs

**Fine-tuning**: Task-specific adaptation

**Prompt Tuning**: Learn prompts, freeze model

**Adapter Tuning**: Add small adapters

### Data Augmentation

**Image Augmentation**: Random crops, flips, color jitter

**Text Augmentation**: Paraphrasing, back-translation

**Cross-Modal Augmentation**: Replace images/text with similar ones

### Optimization

**Learning Rate**: Different rates for image/text encoders

**Warmup**: Gradual learning rate increase

**Gradient Clipping**: Prevent exploding gradients

**Mixed Precision**: FP16 training for efficiency

---

## Key Takeaways

1. **Vision-Language Learning**: Learns joint representations of images and text, enabling tasks like retrieval, VQA, captioning, and zero-shot classification through aligned embeddings.

2. **CLIP Framework**: Uses contrastive learning on image-text pairs with symmetric InfoNCE loss, learning aligned embeddings that enable zero-shot classification via text prompts.

3. **Scale Importance**: ALIGN demonstrates that scaling to billions of image-text pairs significantly improves performance, with simple architectures sufficient when data is abundant.

4. **Flamingo Architecture**: Enables few-shot learning by interleaving frozen vision and language models with learned cross-attention, supporting in-context learning for vision-language tasks.

5. **Visual Grounding**: Localizes objects in images based on text using cross-modal attention, enabling phrase grounding and referring expression comprehension through region-word alignment.

6. **Image-Text Alignment**: Ensures corresponding features are close in embedding space via contrastive learning, with fine-grained alignment matching regions to words through attention mechanisms.

7. **Contrastive Learning**: Foundation of vision-language models, using InfoNCE loss with hard negative mining and temperature scaling to maximize mutual information between modalities.

8. **Zero-Shot Transfer**: Enables task performance without training through text prompts, with prompt engineering (templates, ensembles, learned prompts) critical for performance.

9. **Architectures**: Dual encoders (CLIP) for contrastive learning, cross-attention (Flamingo) for few-shot, with CNN or ViT image encoders and BERT/GPT text encoders.

10. **Training**: Large-scale pre-training on web data, efficient optimization (mixed precision, gradient clipping), and flexible adaptation (fine-tuning, prompt tuning, adapters) enable strong vision-language capabilities.
