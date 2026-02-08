# Question Answering Systems

## Table of Contents

1. [Introduction](#introduction)
2. [Question Answering Types](#question-answering-types)
3. [Extractive Question Answering](#extractive-question-answering)
4. [SQuAD Dataset and Evaluation](#squad-dataset-and-evaluation)
5. [Generative Question Answering](#generative-question-answering)
6. [Reading Comprehension](#reading-comprehension)
7. [Retrieval-Augmented QA](#retrieval-augmented-qa)
8. [Open-Domain Question Answering](#open-domain-question-answering)
9. [Multi-Hop Reasoning](#multi-hop-reasoning)
10. [Key Takeaways](#key-takeaways)

## Introduction

Question Answering (QA) systems answer natural language questions, either by extracting spans from documents (extractive) or generating answers (generative). QA is a fundamental NLP task that requires understanding questions, finding relevant information, and producing accurate answers.

QA applications:
- **Search engines**: Answer queries directly
- **Virtual assistants**: Respond to user questions
- **Customer support**: Answer frequently asked questions
- **Education**: Tutoring systems, study aids

Modern QA systems leverage large language models and retrieval mechanisms to achieve human-level performance on many benchmarks.

## Question Answering Types

QA systems can be categorized by answer source and format.

### Extractive vs Generative

**Extractive QA**: Answer is a span from the context
- **Input**: Question + context document
- **Output**: Start and end positions in context
- **Example**: "What is the capital of France?" → "Paris" (from context)

**Generative QA**: Answer is generated text
- **Input**: Question + context (optional)
- **Output**: Generated answer text
- **Example**: "Explain quantum computing" → Generated explanation

### Open-Domain vs Closed-Domain

**Closed-domain**: Questions about specific domain or document set
**Open-domain**: Questions about anything (requires retrieval)

### Single-Hop vs Multi-Hop

**Single-hop**: Answer found in single document/passage
**Multi-hop**: Requires reasoning across multiple documents

## Extractive Question Answering

Extractive QA finds answer spans within provided context.

### Problem Formulation

Given:
- **Question**: $q = (q_1, \ldots, q_m)$
- **Context**: $c = (c_1, \ldots, c_n)$

Find: **Answer span** $(s, e)$ where answer is $c[s:e]$

### Model Architecture

**Encoder**: Encode question and context
- **BERT-based**: Concatenate `[CLS] q [SEP] c [SEP]`
- **BiLSTM**: Encode question and context separately

**Span prediction**: Predict start and end positions
$$P_{start}(i) = \text{softmax}(\mathbf{W}_s \mathbf{h}_i)$$
$$P_{end}(i) = \text{softmax}(\mathbf{W}_e \mathbf{h}_i)$$

**Answer**: Span with highest $P_{start}(s) \times P_{end}(e)$

### BERT for QA

**Input format**: `[CLS] question [SEP] context [SEP]`

**Output**:
- Start logits: $S \in \mathbb{R}^n$ (one per context token)
- End logits: $E \in \mathbb{R}^n$

**Prediction**: 
$$(s^*, e^*) = \arg\max_{s \leq e} S[s] + E[e]$$

**Constraints**: $s \leq e$ (start before end)

### Training Objective

**Loss function**:
$$L = -\log P_{start}(s_{gold}) - \log P_{end}(e_{gold})$$

Train to maximize probability of correct span boundaries.

## SQuAD Dataset and Evaluation

SQuAD (Stanford Question Answering Dataset) is the standard benchmark for extractive QA.

### SQuAD 1.1

**Format**: Question + context paragraph + answer span
**Size**: 100K+ question-answer pairs
**Source**: Wikipedia articles
**Answer**: Always extractive (span in context)

### SQuAD 2.0

**Additions**: Unanswerable questions (no answer in context)
**Challenge**: Distinguish answerable vs unanswerable
**Evaluation**: F1 and Exact Match (EM) scores

### Evaluation Metrics

**Exact Match (EM)**: Percentage of predictions exactly matching gold answer
**F1 Score**: Token-level F1 between prediction and gold answer

**Handling multiple answers**: Use best match among all valid answers

### Performance

**Human performance**: ~91% F1, ~82% EM
**State-of-the-art**: Exceeds human performance (BERT, RoBERTa, etc.)

## Generative Question Answering

Generative QA produces free-form answers not limited to context spans.

### Architecture

**Encoder-decoder**: 
- **Encoder**: Process question and context
- **Decoder**: Generate answer sequence

**Transformer-based**: T5, BART, GPT models

### Training Objective

**Language modeling**: Maximize likelihood of answer:
$$L = -\sum_{t=1}^{T} \log P(a_t | a_{<t}, q, c)$$

where $a$ is answer sequence.

### Advantages

**Flexibility**: Can generate answers not in context
**Natural language**: More natural phrasing
**Explanation**: Can provide reasoning

### Challenges

**Hallucination**: May generate incorrect information
**Consistency**: Must stay faithful to context
**Evaluation**: Harder to evaluate than extractive

## Reading Comprehension

Reading comprehension requires understanding passages and answering questions about them.

### Task Definition

**Input**: Passage + question
**Output**: Answer (extractive or generative)

**Skills required**:
- **Factual recall**: Direct information extraction
- **Inference**: Reasoning from passage
- **Synthesis**: Combining multiple facts

### Datasets

**SQuAD**: Extractive reading comprehension
**MS MARCO**: Generative answers
**Natural Questions**: Real user questions
**RACE**: Multiple choice reading comprehension

### Model Components

**Passage encoding**: Understand passage content
**Question understanding**: Parse question intent
**Answer extraction/generation**: Produce answer

## Retrieval-Augmented QA

Retrieval-augmented QA combines retrieval with generation for open-domain questions.

### Architecture

**Retriever**: Find relevant documents/passages
**Reader**: Extract or generate answer from retrieved text

**Two-stage**:
1. **Retrieval**: $D_{ret} = \text{Retrieve}(q)$
2. **Reading**: $a = \text{Read}(q, D_{ret})$

### Dense Passage Retrieval (DPR)

**Query encoder**: Encode question as vector
**Passage encoder**: Encode passages as vectors
**Similarity**: Dot product between query and passage embeddings

**Training**: Contrastive learning
$$L = -\log \frac{\exp(\text{sim}(q, p^+))}{\exp(\text{sim}(q, p^+)) + \sum_{p^-} \exp(\text{sim}(q, p^-))}$$

where $p^+$ is positive passage, $p^-$ are negative passages.

### Fusion-in-Decoder (FiD)

**Retrieve**: Multiple passages per question
**Encode**: Encode each passage separately
**Decode**: Attend to all passages during generation

Enables using multiple retrieved passages simultaneously.

## Open-Domain Question Answering

Open-domain QA answers questions without provided context, requiring retrieval.

### Challenges

**Retrieval**: Find relevant information from large corpus
**Ranking**: Rank retrieved passages by relevance
**Answering**: Extract/generate answer from passages
**Verification**: Ensure answer correctness

### Systems

**DrQA**: Wikipedia-based QA system
**RAG**: Retrieval-Augmented Generation
**REALM**: Retrieval-augmented language model pre-training

### Evaluation

**Datasets**: Natural Questions, TriviaQA, WebQuestions
**Metrics**: Exact Match, F1, Human evaluation
**Challenges**: Requires large knowledge bases, efficient retrieval

## Multi-Hop Reasoning

Multi-hop QA requires reasoning across multiple documents or passages.

### Problem

**Single-hop**: Answer in one document
**Multi-hop**: Need to combine information from multiple documents

**Example**: "Who wrote the book that the movie Inception is based on?"
- Need: Book author + Movie based on book

### Approaches

**Sequential reasoning**: Retrieve, read, retrieve again based on intermediate answer
**Graph reasoning**: Build knowledge graph, reason over graph
**Attention**: Attend to multiple passages simultaneously

### Datasets

**HotpotQA**: Multi-hop reasoning dataset
**2WikiMultihopQA**: Wikipedia-based multi-hop
**MuSiQue**: Multi-hop with single-hop distractors

### Challenges

**Reasoning chain**: Must follow correct reasoning path
**Noise**: Irrelevant passages may mislead
**Compositionality**: Combine multiple facts correctly

## Key Takeaways

1. **Extractive QA finds answer spans**: Predicting start and end positions in context enables accurate answer extraction with BERT and similar models.

2. **SQuAD is the standard benchmark**: SQuAD 1.1 and 2.0 provide evaluation standards, with state-of-the-art models exceeding human performance.

3. **Generative QA produces free-form answers**: Encoder-decoder architectures enable generating natural language answers beyond context spans.

4. **Retrieval is crucial for open-domain QA**: Dense retrieval methods like DPR enable finding relevant passages from large knowledge bases.

5. **Retrieval-augmented generation combines strengths**: Systems like RAG and FiD combine retrieval and generation for accurate open-domain answers.

6. **Multi-hop reasoning requires compositionality**: Answering complex questions requires reasoning across multiple documents and combining information correctly.

7. **Evaluation metrics matter**: Exact Match and F1 capture different aspects of QA performance, with F1 being more forgiving of minor differences.

8. **QA systems enable practical applications**: From search engines to virtual assistants, QA systems power many real-world applications requiring accurate information retrieval and generation.
