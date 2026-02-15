# NLP Tasks and Generation

## Table of Contents

- [Text Generation Methods](#text-generation-methods)
- [Named Entity Recognition](#named-entity-recognition)
- [Machine Translation](#machine-translation)
- [Question Answering](#question-answering)

---

## Text Generation Methods

### Greedy Decoding

**Greedy decoding** selects the token with the highest probability at each step. Fast and deterministic, but often produces repetitive or suboptimal sequences.

```python
def greedy_generate(model, input_ids, max_length=100):
    for _ in range(max_length):
        logits = model(input_ids)[0]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids
```

### Top-K Sampling

**Top-k sampling** restricts sampling to the k most likely tokens, renormalizing their probabilities. Reduces low-probability noise while preserving diversity.

```python
def top_k_sampling(logits, k=50):
    if k > 0:
        values, _ = torch.topk(logits, min(k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Top-P (Nucleus) Sampling

**Top-p (nucleus) sampling** selects the smallest set of tokens whose cumulative probability exceeds p. Adapts the candidate set size dynamically.

```python
def top_p_sampling(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Temperature Scaling

**Temperature** controls the sharpness of the probability distribution. Lower temperature (e.g., 0.7) makes outputs more deterministic; higher (e.g., 1.2) increases randomness.

```python
def apply_temperature(logits, temperature=1.0):
    return logits / temperature
```

### Repetition Penalty

**Repetition penalty** reduces the probability of tokens that have already appeared in the generated sequence, discouraging repetitive output.

```python
def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    for token_id in set(input_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits
```

### Generation Method Comparison

| Method | Diversity | Coherence | Speed | Use Case |
|--------|-----------|-----------|-------|----------|
| Greedy | Low | High | Fast | Short, factual output |
| Top-k | Medium | Medium | Fast | General generation |
| Top-p | Medium-High | Medium | Fast | Creative writing |
| Beam search | Low | High | Slow | Translation, summarization |
| Temperature | Tunable | Tunable | Fast | Control randomness |

---

## Named Entity Recognition

### Token Classification

**Named Entity Recognition (NER)** is a **token classification** task: each token is assigned a label (e.g., B-PER, I-PER, O). Models output logits per token; a CRF or softmax produces the final tags.

```python
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size)

    def forward(self, word_ids, tags, lengths=None, mask=None):
        embeds = self.embedding(word_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return self.crf(lstm_feats, tags, mask)

    def predict(self, word_ids, lengths=None, mask=None):
        lstm_feats = self._get_lstm_features(word_ids, lengths)
        return self.crf.decode(lstm_feats, mask)
```

### BIO Tagging

**BIO tagging** uses B- (begin), I- (inside), O (outside) to mark entity spans. B-PER starts a person, I-PER continues it; O is non-entity.

| Tag | Meaning |
|-----|---------|
| B-PER | Begin person |
| I-PER | Inside person |
| B-ORG | Begin organization |
| I-ORG | Inside organization |
| B-LOC | Begin location |
| I-LOC | Inside location |
| O | Outside any entity |

```python
def extract_entities(labels):
    entities = []
    current_entity = None
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = (i, i, label[2:])
        elif label.startswith('I-') and current_entity and label[2:] == current_entity[2]:
            current_entity = (current_entity[0], i, current_entity[2])
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities
```

### CRF Layer Concepts

**Conditional Random Field (CRF)** models transition scores between consecutive tags. It encourages valid tag sequences (e.g., I-PER after B-PER) and discourages invalid ones. **Viterbi decoding** finds the highest-scoring tag sequence.

```python
class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        self.start_tag_idx = num_tags
        self.end_tag_idx = num_tags + 1

    def forward(self, emissions, tags, mask=None):
        partition = self._forward_algorithm(emissions, mask)
        score = self._score_sentence(emissions, tags, mask)
        return (partition - score).mean()

    def decode(self, emissions, mask=None):
        return self._viterbi_decode(emissions, mask)
```

---

## Machine Translation

### Parallel Corpus Handling

**Parallel corpora** contain source and target sentence pairs. Preprocessing includes tokenization, vocabulary building for both languages, and alignment. Special tokens: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`.

```python
class TranslationDataset(Dataset):
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        src_indices = self.encode_sentence(src_sentence, self.src_vocab)
        tgt_indices = self.encode_sentence(tgt_sentence, self.tgt_vocab)
        tgt_input = tgt_indices[:-1]
        tgt_output = tgt_indices[1:]
        return {'src': src_indices, 'tgt_input': tgt_input, 'tgt_output': tgt_output}
```

### BPE for Translation

**BPE** is commonly used for machine translation. Joint BPE merges source and target vocabularies for shared subwords; separate BPE keeps language-specific vocabularies.

### Evaluation with BLEU

**BLEU (Bilingual Evaluation Understudy)** measures n-gram precision between hypothesis and reference. Includes a brevity penalty for short outputs.

```python
def compute_bleu_score(predictions, references, max_n=4):
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def precision(pred_ngrams, ref_ngrams):
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        matches = sum(min(pred_counter[ng], ref_counter.get(ng, 0)) for ng in pred_counter)
        return matches / len(pred_ngrams) if pred_ngrams else 0.0

    precisions = []
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred = 0
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            pred_counter = Counter(pred_ngrams)
            ref_counter = Counter(ref_ngrams)
            total_matches += sum(min(pred_counter[ng], ref_counter.get(ng, 0)) for ng in pred_counter)
            total_pred += len(pred_ngrams)
        precisions.append(total_matches / total_pred if total_pred > 0 else 0.0)

    log_precisions = [math.log(p + 1e-10) for p in precisions]
    avg_log = sum(log_precisions) / len(log_precisions)
    total_pred_len = sum(len(p) for p in predictions)
    total_ref_len = sum(len(r) for r in references)
    brevity_penalty = 1.0 if total_pred_len > total_ref_len else math.exp(1 - total_ref_len / total_pred_len)
    return brevity_penalty * math.exp(avg_log)
```

---

## Question Answering

### Extractive QA

**Extractive question answering** assumes the answer is a contiguous span in the context. The model predicts **start** and **end** token indices.

### Span Prediction

**Span prediction** uses two linear heads: one for start logits, one for end logits. Valid spans satisfy start <= end and fall within the context. The score for a span (s, e) is often start_logits[s] + end_logits[e].

```python
class BERTForQuestionAnswering(nn.Module):
    def __init__(self, bert_model, dropout=0.1):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(bert_model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, start_logits, end_logits
        return None, start_logits, end_logits
```

### Start and End Logits

The model outputs **start_logits** and **end_logits** of shape `[batch_size, seq_len]`. At inference, the best span is found by maximizing start_logits[s] + end_logits[e] over valid (s, e) pairs.

```python
def extract_answer(start_logits, end_logits, input_tokens, max_answer_length=30):
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    best_score = 0
    best_start, best_end = 0, 0
    seq_len = len(start_probs)
    for start in range(seq_len):
        for end in range(start, min(start + max_answer_length, seq_len)):
            score = start_probs[start] * end_probs[end]
            if score > best_score:
                best_score = score
                best_start, best_end = start, end
    answer_tokens = input_tokens[best_start:best_end + 1]
    answer = ' '.join([t for t in answer_tokens if t not in ['[PAD]', '[CLS]', '[SEP]']])
    return answer, best_start, best_end
```

### Input Format for QA

The input is typically `[CLS] question [SEP] context [SEP]`. **Token type IDs** distinguish question (0) from context (1). Answer positions are adjusted for the prepended question tokens.

```python
tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
answer_start += len(question_tokens) + 2
answer_end += len(question_tokens) + 2
token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Exact Match (EM) | Percentage of predictions that exactly match the reference |
| F1 | Token-level overlap; 2 * precision * recall / (precision + recall) |

```python
def compute_exact_match(predictions, references):
    return sum(normalize(p) == normalize(r) for p, r in zip(predictions, references)) / len(predictions)

def compute_f1_score(pred_tokens, ref_tokens):
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
```
