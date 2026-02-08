# Pretrained Models: BERT, GPT, and Variants

## Table of Contents

1. [Introduction](#introduction)
2. [BERT Architecture and Variants](#bert-architecture-and-variants)
3. [GPT and Autoregressive Models](#gpt-and-autoregressive-models)
4. [RoBERTa: Robust BERT](#roberta-robust-bert)
5. [ALBERT: A Lite BERT](#albert-a-lite-bert)
6. [DistilBERT: Distilled BERT](#distilbert-distilled-bert)
7. [T5: Text-to-Text Transfer Transformer](#t5-text-to-text-transfer-transformer)
8. [Model Comparison](#model-comparison)
9. [Fine-Tuning Strategies](#fine-tuning-strategies)
10. [Key Takeaways](#key-takeaways)

## Introduction

Pretrained language models have become the foundation of modern NLP, achieving state-of-the-art performance across diverse tasks. BERT (encoder-only), GPT (decoder-only), and T5 (encoder-decoder) represent different architectural choices, each optimized for different objectives.

Key developments:
- **BERT**: Bidirectional encoder for understanding tasks
- **GPT**: Autoregressive decoder for generation tasks
- **T5**: Encoder-decoder for text-to-text tasks
- **Variants**: Optimizations for efficiency, robustness, and performance

These models enable transfer learning: pre-train on large unlabeled corpora, then fine-tune on specific tasks with minimal labeled data.

## BERT Architecture and Variants

BERT uses Transformer encoder for bidirectional language understanding.

### BERT Base and Large

**BERT-Base**: 12 layers, 12 attention heads, 768 hidden size, 110M parameters
**BERT-Large**: 24 layers, 16 attention heads, 1024 hidden size, 340M parameters

**Pre-training**:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

**Fine-tuning**: Add task-specific layers, train on downstream tasks.

### BERT Variants

**BERTweet**: Pre-trained on Twitter data
**SciBERT**: Pre-trained on scientific text
**BioBERT**: Pre-trained on biomedical text
**Multilingual BERT**: Pre-trained on 104 languages

Domain-specific variants improve performance on specialized tasks.

### BERT Limitations

**Masked tokens**: `[MASK]` token doesn't appear at inference
**Pre-training/fine-tuning mismatch**: Different objectives
**Fixed length**: Maximum sequence length (512 tokens)
**Computational cost**: Large models require significant resources

## GPT and Autoregressive Models

GPT (Generative Pre-trained Transformer) uses decoder-only architecture for autoregressive generation.

### GPT Architecture

**Decoder-only**: Transformer decoder layers (masked self-attention)
**Autoregressive**: Generates tokens left-to-right
**Causal masking**: Prevents attending to future tokens

**Pre-training**: Language modeling objective:
$$L = -\sum_{i=1}^{n} \log P(x_i | x_{<i})$$

### GPT-1, GPT-2, GPT-3

**GPT-1**: 117M parameters, demonstrates transfer learning
**GPT-2**: 1.5B parameters, shows scaling improves performance
**GPT-3**: 175B parameters, few-shot learning capabilities

**Key insight**: Larger models and more data enable better few-shot performance.

### Autoregressive Advantages

**Natural generation**: Left-to-right generation matches human language production
**No pre-training/inference mismatch**: Same objective during pre-training and inference
**Flexible length**: Can generate sequences of any length

### Autoregressive Limitations

**Unidirectional**: Only left context available
**Slower inference**: Sequential generation
**Exposure bias**: Training uses ground truth, inference uses predictions

## RoBERTa: Robust BERT

RoBERTa improves BERT through better training procedures.

### Key Changes

**Removed NSP**: Next Sentence Prediction doesn't help
**Dynamic masking**: Different masks each epoch (not static)
**Larger batches**: 8K instead of 256
**More data**: 10x more training data
**Longer training**: More training steps
**Byte-level BPE**: Better tokenization

### Training Improvements

**Learning rate**: Tuned more carefully
**Warmup**: Longer warmup period
**Dropout**: Adjusted dropout rates

### Results

**Performance**: Outperforms BERT on GLUE benchmark
**Robustness**: More stable across hyperparameters
**Efficiency**: Similar architecture, better training

## ALBERT: A Lite BERT

ALBERT reduces parameters while maintaining performance through parameter sharing and factorization.

### Parameter Reduction Techniques

**Factorized embedding parameterization**:
- Separate embedding size from hidden size
- Project embeddings to hidden size
- Reduces parameters when $V \gg d_{hidden}$

**Cross-layer parameter sharing**:
- Share parameters across all layers
- Dramatically reduces parameters
- Slight performance drop

### Additional Improvements

**Sentence order prediction**: Predicts order of two consecutive sentences (replaces NSP)
**Inter-sentence coherence**: Better than NSP for learning sentence relationships

### ALBERT Results

**Parameters**: 18M (ALBERT-base) vs 110M (BERT-base)
**Performance**: Comparable to BERT
**Training**: Slower (parameter sharing increases computation)
**Memory**: Lower memory footprint

## DistilBERT: Distilled BERT

DistilBERT uses knowledge distillation to create smaller, faster BERT.

### Knowledge Distillation

**Teacher**: Large BERT model
**Student**: Smaller model (6 layers vs 12)

**Loss function**:
$$L = \alpha L_{CE} + (1-\alpha) L_{KL}$$

where:
- $L_{CE}$: Cross-entropy with hard labels
- $L_{KL}$: KL divergence from teacher softmax

### Distillation Process

**Pre-training**: Distill during pre-training
**Fine-tuning**: Can further distill during task-specific fine-tuning

### Results

**Size**: 60% of BERT parameters
**Speed**: 60% faster inference
**Performance**: 97% of BERT performance
**Efficiency**: Better speed/accuracy trade-off

## T5: Text-to-Text Transfer Transformer

T5 frames all NLP tasks as text-to-text problems using encoder-decoder architecture.

### Text-to-Text Framework

**Input**: Text prefix describing task
**Output**: Text response

**Examples**:
- Translation: "translate English to German: The house is wonderful"
- Summarization: "summarize: [article text]"
- Classification: "cola sentence: [sentence]" → "acceptable" or "not acceptable"

### T5 Architecture

**Encoder-decoder**: Full Transformer architecture
**Pre-training**: Span corruption (mask spans, predict them)
**Fine-tuning**: Task-specific prefixes

### T5 Variants

**T5-Small**: 60M parameters
**T5-Base**: 220M parameters
**T5-Large**: 770M parameters
**T5-3B**: 3B parameters
**T5-11B**: 11B parameters

### T5 Advantages

**Unified framework**: Same architecture for all tasks
**Task prefixes**: Natural way to specify tasks
**Flexible**: Handles generation and understanding

## Model Comparison

Different architectures suit different tasks and constraints.

### Architecture Comparison

| Model | Architecture | Direction | Best For |
|-------|--------------|-----------|----------|
| BERT | Encoder-only | Bidirectional | Understanding tasks |
| GPT | Decoder-only | Left-to-right | Generation tasks |
| T5 | Encoder-decoder | Bidirectional input | Text-to-text tasks |

### Task Suitability

**BERT**: Classification, NER, QA (understanding)
**GPT**: Text generation, completion (generation)
**T5**: Translation, summarization, QA (both)

### Efficiency Comparison

**Parameters**: ALBERT < DistilBERT < BERT < GPT-3
**Speed**: DistilBERT > ALBERT > BERT > GPT-3
**Performance**: Depends on task and model size

## Fine-Tuning Strategies

Effective fine-tuning is crucial for leveraging pretrained models.

### Full Fine-Tuning

**Update all parameters**: Fine-tune entire model
**High capacity**: Can adapt to task
**Risk of overfitting**: Especially with small datasets

### Feature Extraction

**Freeze encoder**: Keep pretrained weights fixed
**Train classifier**: Only train task-specific head
**Faster**: Less computation
**Less flexible**: Cannot adapt representations

### Layer-Wise Fine-Tuning

**Progressive unfreezing**: Unfreeze layers gradually
**Top-down**: Start with top layers, add lower layers
**Bottom-up**: Start with bottom layers, add top layers

### Hyperparameter Tuning

**Learning rate**: Lower than pre-training (typically $10^{-5}$ to $10^{-4}$)
**Batch size**: Smaller batches often work better
**Epochs**: Few epochs sufficient (1-4)
**Warmup**: Gradual learning rate increase

### Task-Specific Adaptations

**Classification**: Add classification head
**Sequence labeling**: Add CRF layer
**QA**: Add span prediction head
**Generation**: Use decoder for generation

## Key Takeaways

1. **BERT enables bidirectional understanding**: Encoder-only architecture with MLM pre-training learns rich bidirectional representations for understanding tasks.

2. **GPT excels at generation**: Decoder-only autoregressive architecture naturally suits text generation and completion tasks.

3. **T5 unifies tasks as text-to-text**: Encoder-decoder architecture with task prefixes enables handling diverse tasks in a unified framework.

4. **RoBERTa improves BERT training**: Better training procedures (dynamic masking, more data) improve performance without architecture changes.

5. **ALBERT reduces parameters**: Parameter sharing and factorization enable smaller models with comparable performance.

6. **DistilBERT balances size and performance**: Knowledge distillation creates efficient models suitable for deployment.

7. **Model choice depends on task**: Understanding tasks favor BERT, generation tasks favor GPT, flexible tasks favor T5.

8. **Fine-tuning is crucial**: Effective fine-tuning strategies maximize transfer learning benefits while avoiding overfitting.
