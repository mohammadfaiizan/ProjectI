# Multimodal Vision-Language

## Table of Contents

1. [Introduction](#introduction)
2. [Vision-Language Pre-training](#vision-language-pre-training)
3. [CLIP (Contrastive Language-Image Pre-training)](#clip-contrastive-language-image-pre-training)
4. [ALIGN](#align)
5. [Image Captioning](#image-captioning)
6. [Visual Question Answering (VQA)](#visual-question-answering-vqa)
7. [Visual Grounding](#visual-grounding)
8. [Text-to-Image Generation](#text-to-image-generation)
9. [Cross-Modal Retrieval](#cross-modal-retrieval)
10. [Key Takeaways](#key-takeaways)

## Introduction

Multimodal vision-language models connect visual and textual understanding, enabling applications including image captioning, visual question answering, text-to-image generation, and cross-modal retrieval. These models learn joint representations of images and text, enabling machines to understand relationships between visual and linguistic concepts.

The evolution from task-specific models to large-scale pre-trained models like CLIP and ALIGN represents a shift toward general-purpose vision-language understanding. Understanding these methods reveals how to align visual and textual representations and enables applications requiring both vision and language understanding.

## Vision-Language Pre-training

Large-scale pre-training on image-text pairs enables transfer to downstream tasks.

### Pre-training Objectives

**Contrastive learning**: Align image and text embeddings
**Masked language modeling**: Predict masked words given image
**Image-text matching**: Classify if image and text match
**Masked region modeling**: Predict masked image regions given text

### Architecture

**Dual encoders**: Separate image and text encoders
**Cross encoders**: Joint encoding of image-text pairs
**Transformer-based**: Self-attention and cross-attention

### Benefits

- **Transfer learning**: Pre-train on large datasets, fine-tune on tasks
- **Generalization**: Learn general vision-language understanding
- **Efficiency**: Less task-specific data needed

## CLIP (Contrastive Language-Image Pre-training)

CLIP learns visual representations from natural language supervision using contrastive learning.

### Architecture

**Image encoder**: Vision Transformer (ViT) or ResNet
**Text encoder**: Transformer
**Projection**: Linear projection to shared embedding space

### Training

**Data**: 400M image-text pairs from internet
**Objective**: Contrastive learning

For batch of $N$ image-text pairs:
- **Positive pairs**: $(I_i, T_i)$ match
- **Negative pairs**: $(I_i, T_j)$ for $i \neq j$ don't match

**Loss**: InfoNCE
$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

where $\text{sim}$ is cosine similarity and $\tau$ is temperature.

### Zero-Shot Transfer

CLIP enables zero-shot classification:

1. **Text prompts**: "a photo of a {class}"
2. **Encode**: Encode all prompts
3. **Match**: Find image most similar to class prompt

No training on target dataset needed.

### Applications

**Image classification**: Zero-shot on ImageNet
**Image retrieval**: Text-to-image search
**Image generation**: Guide generation with text
**Robustness**: More robust than supervised models

### Limitations

- **Fine-grained tasks**: Struggles with fine-grained classification
- **Abstract concepts**: Limited understanding of abstract concepts
- **Bias**: Inherits biases from training data

## ALIGN

ALIGN (A Large-scale ImaGe and Noisy-text) scales up contrastive pre-training.

### Key Innovation

**Noisy data**: Use alt-text from web images (noisy but abundant)
**Scale**: 1.8B image-text pairs
**Simple**: Contrastive learning only

### Architecture

Similar to CLIP:
- **Image encoder**: EfficientNet
- **Text encoder**: BERT
- **Contrastive loss**: Same as CLIP

### Benefits

- **Scale**: More data improves performance
- **Simplicity**: Single objective
- **Robustness**: Handles noisy data

### Performance

Outperforms CLIP on many tasks:
- **ImageNet**: Better zero-shot accuracy
- **Retrieval**: Better cross-modal retrieval
- **Downstream**: Better transfer learning

## Image Captioning

Image captioning generates natural language descriptions of images.

### Problem

Given image $I$, generate caption $C = \{w_1, w_2, \ldots, w_n\}$.

### Architecture

**Encoder-decoder**:
- **Encoder**: CNN (ResNet, VGG) extracts image features
- **Decoder**: RNN (LSTM) or Transformer generates caption

### Show and Tell

Early CNN-RNN approach:

**Encoder**: CNN extracts features
**Decoder**: LSTM generates words
**Attention**: Attend to image regions

### Attention-Based Captioning

**Visual attention**: 
- Attend to different image regions
- Generate words based on attended regions
- Improves accuracy

### Transformer-Based Captioning

**Image features**: Extract patches or regions
**Transformer**: Self-attention and cross-attention
**Generation**: Autoregressive generation

### Evaluation

**BLEU**: N-gram overlap with references
**METEOR**: Considers synonyms and paraphrases
**CIDEr**: Consensus-based image description evaluation
**SPICE**: Semantic propositional image caption evaluation

## Visual Question Answering (VQA)

VQA answers natural language questions about images.

### Problem

Given image $I$ and question $Q$, predict answer $A$.

### Architecture

**Multi-modal fusion**:
- **Image encoder**: CNN extracts features
- **Question encoder**: RNN or Transformer
- **Fusion**: Combine image and question features
- **Answer prediction**: Classify or generate answer

### Fusion Strategies

**Early fusion**: Concatenate features
**Late fusion**: Combine predictions
**Attention**: Learn to attend to relevant regions
**Bilinear pooling**: Outer product of features

### Attention Mechanisms

**Question-guided attention**: Attend to image based on question
**Co-attention**: Joint attention over image and question
**Stacked attention**: Multiple attention layers

### Datasets

**VQA v2**: Balanced dataset (reduces language bias)
**GQA**: Scene graph-based questions
**Visual7W**: 7 types of questions

### Challenges

**Language bias**: Models exploit question patterns
**Visual reasoning**: Requires understanding relationships
**Compositional**: Complex questions require composition

## Visual Grounding

Visual grounding localizes objects in images based on text descriptions.

### Problem

Given image $I$ and text description $T$, find bounding box $B$ of described object.

### Methods

**Two-stage**:
1. **Object detection**: Detect all objects
2. **Matching**: Match descriptions to objects

**One-stage**:
- Directly predict boxes from text
- End-to-end training

### Referring Expression Comprehension

**Input**: Image + referring expression ("the red car")
**Output**: Bounding box of referred object

**Challenges**:
- Ambiguity: Multiple objects match description
- Context: Requires understanding context
- Relationships: "car next to the tree"

### Phrase Grounding

**Input**: Image + phrases
**Output**: Boxes for each phrase

**Methods**:
- **Matching**: Match phrases to regions
- **Attention**: Attend to regions for phrases
- **Dense prediction**: Predict boxes per phrase

## Text-to-Image Generation

Text-to-image generation creates images from text descriptions.

### Problem

Given text $T$, generate image $I$ matching description.

### Methods

**GAN-based**: 
- **StackGAN**: Stacked generators
- **AttnGAN**: Attention-guided generation
- **DF-GAN**: Deep fusion GAN

**Diffusion-based**:
- **DALL-E 2**: Diffusion model with CLIP
- **Imagen**: Large language model + diffusion
- **Stable Diffusion**: Latent diffusion

**Autoregressive**:
- **DALL-E**: Autoregressive generation
- **Parti**: Pathways autoregressive text-to-image

### DALL-E

**Architecture**: 
- **Text encoder**: Transformer
- **Image tokens**: VQ-VAE tokens
- **Autoregressive**: Generate tokens sequentially

**Training**: 
- Large-scale image-text pairs
- Autoregressive language modeling objective

### DALL-E 2

**Architecture**:
- **CLIP**: Pre-trained vision-language model
- **Prior**: Generate CLIP image embedding from text
- **Decoder**: Generate image from embedding

**Process**:
1. **Text → Embedding**: CLIP text encoder
2. **Prior**: Generate image embedding
3. **Decoder**: Generate image from embedding

### Stable Diffusion

**Latent diffusion**: 
- Operate in latent space (not pixel space)
- More efficient
- Better quality

**Process**:
1. **Encode**: VAE encoder → latent
2. **Diffuse**: Denoise in latent space
3. **Decode**: VAE decoder → image

**Conditioning**: Text embeddings guide diffusion

### Evaluation

**FID**: Fréchet Inception Distance
**IS**: Inception Score
**CLIP Score**: Semantic similarity to text
**Human evaluation**: Subjective quality

## Cross-Modal Retrieval

Cross-modal retrieval finds images from text queries or vice versa.

### Image-Text Retrieval

**Image-to-text**: Given image, find relevant text
**Text-to-image**: Given text, find relevant images

### Methods

**Dual encoders**: 
- Encode images and text separately
- Compare in shared embedding space
- Efficient for large-scale retrieval

**Cross encoders**:
- Encode image-text pairs jointly
- More accurate but slower
- Used for re-ranking

### CLIP for Retrieval

CLIP enables zero-shot retrieval:

1. **Encode**: Encode images and texts
2. **Similarity**: Compute cosine similarity
3. **Rank**: Rank by similarity

**Benefits**:
- No training on retrieval dataset
- Generalizes to new domains
- Efficient

### Evaluation

**Recall@K**: Fraction of relevant items in top K
**Mean Reciprocal Rank (MRR)**: Average reciprocal rank
**Normalized Discounted Cumulative Gain (NDCG)**: Ranked relevance

## Key Takeaways

1. Vision-language pre-training learns joint representations from large-scale image-text pairs, enabling transfer to downstream tasks with less task-specific data.

2. CLIP uses contrastive learning to align image and text embeddings, enabling zero-shot transfer to tasks like image classification and retrieval without task-specific training.

3. ALIGN scales up contrastive pre-training with noisy web data, demonstrating that scale and simplicity can achieve strong performance.

4. Image captioning generates natural language descriptions using encoder-decoder architectures with attention mechanisms to focus on relevant image regions.

5. Visual Question Answering requires multi-modal fusion of image and question features, with attention mechanisms enabling question-guided visual reasoning.

6. Visual grounding localizes objects based on text descriptions, requiring understanding of referring expressions and spatial relationships.

7. Text-to-image generation creates images from text using GANs, diffusion models, or autoregressive models, with recent methods like DALL-E 2 and Stable Diffusion achieving high-quality results.

8. Cross-modal retrieval enables finding images from text or vice versa, with dual encoders providing efficient large-scale retrieval and cross encoders providing accurate re-ranking.

9. Large-scale pre-trained models like CLIP enable zero-shot transfer to many vision-language tasks, reducing the need for task-specific training data.

10. Understanding multimodal vision-language models enables applications including image search, content creation, visual reasoning, and human-computer interaction through aligned visual and textual representations.
