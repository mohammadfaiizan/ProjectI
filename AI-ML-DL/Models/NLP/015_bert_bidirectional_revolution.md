# BERT: The Bidirectional Revolution

This file (`015_bert_bidirectional_revolution.py`) implements a small, educational version of **BERT** (Bidirectional Encoder Representations from Transformers), the 2018 model that changed how the entire NLP field trains language models. It builds the embeddings, self-attention, encoder stack, pre-training heads (MLM + NSP), and a fine-tuning classifier, then trains all of it end to end on a tiny slice of WikiText-2.

Only one real, named model is covered in this file: **BERT**. But BERT actually ships two published sizes (BERT-base and BERT-large), and both are discussed below since interviewers expect you to know the concrete numbers.

---

## BERT (Devlin et al. 2018)

### 1. What Problem It Solved

Before BERT, the best pre-trained language models (like GPT-1 and ELMo) had a big blind spot: they could only look in **one direction** at a time, or they combined two directions in a shallow way.

- GPT-1 is a left-to-right (unidirectional) model. When it processes the word "bank" in "I went to the bank to deposit money," it can only use the words before "bank" ("I went to the") to build its representation — it is not allowed to peek at "to deposit money" while encoding that token, because its whole architecture is built around predicting the next word from only the past.
- ELMo tried to fix this by training a separate left-to-right LSTM and a separate right-to-left LSTM, then concatenating their outputs. That is bidirectional in a shallow sense (two independent models glued together at the end), not a deep, joint understanding of context from both sides at every layer.

This one-directional limitation hurt tasks that need full-sentence understanding, like question answering or sentence classification, where the meaning of a word genuinely depends on words that come both before and after it. BERT's core idea was: build a model that conditions on the entire sentence — left and right context together, at every layer — and pre-train it in a way that a normal left-to-right language modeling objective cannot support (because if you literally show a model the whole sentence and ask it to predict each next word, it can just "cheat" by looking ahead).

### 2. Architecture — How It Works

**Big picture:** BERT is just the encoder half of the original Transformer (no decoder, no causal masking), stacked many layers deep, fed by a token that mixes together three kinds of embeddings, and topped with two small "heads" during pre-training.

Step by step:

1. **Input construction.** Every input starts with a special `[CLS]` token (used later as a summary of the whole sequence) and uses `[SEP]` tokens to separate segments. For pre-training, an input looks like: `[CLS] sentence_A_tokens [SEP] sentence_B_tokens [SEP]`.
2. **Three embeddings are summed together** for every token position:
   - **Token (word) embedding** — what the word means, from a lookup table.
   - **Position embedding** — where the token sits in the sequence (a learned vector per position, not the sinusoidal kind).
   - **Segment (token-type) embedding** — whether this token belongs to "sentence A" or "sentence B." This is BERT-specific and is what lets the model do sentence-pair tasks.
   The three vectors are added, then passed through LayerNorm and dropout.
3. **Self-attention, but fully bidirectional.** Each encoder layer runs multi-head self-attention where every token can attend to every other token in the sequence — nothing is masked out to hide future tokens (unlike GPT's causal mask). This is the actual mechanism that makes BERT "bidirectional": there's no directionality restriction at all: attention is computed as `softmax(QK^T / sqrt(d_k)) V` over the whole sequence at once.
4. **Add & Norm, then feed-forward.** Each layer does self-attention → add residual → LayerNorm → feed-forward (Linear → GELU → Linear) → add residual → LayerNorm. This is "post-norm" (LayerNorm is applied *after* adding the residual), which is the original BERT recipe (later models like GPT-2 switched to "pre-norm").
5. **Stack this layer N times** (12 for BERT-base, 24 for BERT-large) to get the final sequence of hidden vectors.
6. **Pooler.** The hidden vector at the `[CLS]` position is passed through one more Linear + tanh to get a fixed-size "pooled" sentence representation, used for sentence-level tasks.

**Masked Language Modeling (MLM) — the trick that makes bidirectional pre-training possible.** You can't train a bidirectional model with plain "predict the next word," because the model can already see the answer. BERT's fix: randomly hide some tokens and make the model guess them using context from *both* sides.

Concretely, for each token, with 15% probability it is selected for masking. Once a token is selected, what actually happens to it splits three ways:

```
Of the tokens chosen to be "masked" (15% of all tokens):
  80% of the time  -> replace the token with the special [MASK] token
  10% of the time  -> replace the token with a random word from the vocabulary
  10% of the time  -> leave the token unchanged
```

Why not just always replace with `[MASK]`? Because during fine-tuning later, the model will never see a `[MASK]` token — that mismatch between pre-training input and fine-tuning input would hurt it. By sometimes leaving the real word in place, or swapping in a random wrong word, the model is forced to build a genuine contextual representation of *every* token (since it can't tell in advance which ones are "trustworthy"), rather than just learning "if you see `[MASK]`, guess a word." The loss is only computed on the originally-selected 15% of positions; everything else is ignored (label = -100 in the code, which `CrossEntropyLoss(ignore_index=-100)` skips).

**Next Sentence Prediction (NSP) — the second pre-training task.** Alongside MLM, BERT also trains on sentence-pair relationships. Given `[CLS] A [SEP] B [SEP]`, 50% of the time B really is the sentence that follows A in the original text (label = "IsNext"), and 50% of the time B is a random sentence pulled from elsewhere in the corpus (label = "NotNext"). The pooled `[CLS]` vector is fed into a small binary classifier to predict this. The idea was to teach the model something about sentence-level coherence, useful for tasks like question answering.

**Pre-train then fine-tune.** BERT is trained once on MLM + NSP over a huge unlabeled corpus (pre-training), and the same weights are then reused and lightly retrained ("fine-tuned") on a much smaller labeled dataset for a specific task — in this file, sentiment classification. Fine-tuning just swaps out the pre-training heads for a task-specific head (a classifier on top of the pooled `[CLS]` vector) and continues training with a small learning rate.

### 3. Model Size & Parameters

**Real, published BERT specs (Devlin et al., 2018):**

| | BERT-base | BERT-large |
|---|---|---|
| Parameters | 110 million | 340 million |
| Layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Feed-forward size | 3072 | 4096 |
| Max sequence length | 512 | 512 |
| Vocabulary (WordPiece) | ~30,522 | ~30,522 |

**What this code actually uses:** In `main()`, `BERTForPreTraining` is instantiated with `hidden_size=256`, `num_attention_heads=8`, `num_hidden_layers=4`, `intermediate_size=512`, over a vocabulary capped at 3,000 tokens (built from a 600-sentence WikiText-2 subset). That's a tiny fraction of BERT-base's already-modest 110M — the printed parameter count for this demo model will be in the low millions at most, versus 110M/340M for the real thing.

**Why the gap:** The educational version needs to train in seconds to minutes on a laptop CPU, on a few hundred sentences. Real BERT needed 12–24 layers of 768–1024-dimensional hidden states, a 30k-token WordPiece vocabulary, and days of TPU time on a corpus three billion words wide. Shrinking every dimension (layers, hidden size, heads, vocabulary, corpus size) by roughly one to two orders of magnitude each compounds into a model that's thousands of times smaller overall — enough to preserve the mechanics (MLM, NSP, bidirectional attention, pre-train/fine-tune) while making the code runnable without a GPU cluster.

### 4. Dataset & What It Was Trained On

**Real BERT training data:** BooksCorpus (about 800 million words, drawn from roughly 11,000 unpublished books) plus English Wikipedia (about 2,500 million words, text passages only, no lists/tables/headers). Combined, that's about 3.3 billion words of running text — chosen specifically because they're long, coherent documents (useful for the NSP task, which needs real consecutive sentences) rather than shuffled, sentence-level data.

**What this code uses:** WikiText-2, a much smaller public dataset of Wikipedia articles (about 2 million tokens total), and even then only a subset — the code takes the first 600 sentences for the pre-training split (train), 120 for validation, and 120 for test, after filtering to sentences with 5–25 tokens. Text is lowercased and tokenized with NLTK's `word_tokenize`, not WordPiece.

**The gap:** This is a difference of roughly six orders of magnitude in corpus size (a few hundred sentences vs. billions of words), and a difference in tokenization scheme (a closed word-level vocabulary of 3,000 tokens vs. WordPiece subword tokenization with ~30k tokens, which handles rare/unseen words far better). The demo dataset is enough to prove the training loop works and losses decrease; it is nowhere near enough data or vocabulary coverage to learn genuinely useful language representations.

### 5. Training Process

**Real objective:** Total pre-training loss = MLM loss + NSP loss, both cross-entropy. MLM loss is computed only on the masked positions (everything else ignored via `ignore_index=-100`); NSP loss is a standard 2-class cross-entropy on the `[CLS]` prediction. This code implements exactly that combination in `BERTForPreTraining.forward()`: `total_loss = mlm_loss + nsp_loss` (an unweighted sum — no extra hyperparameter balancing the two).

**Real training setup:** BERT was trained with batches of 256 sequences × 512 tokens (about 128,000 tokens per batch), for 1,000,000 steps — roughly 40 epochs over the 3.3-billion-word corpus. BERT-base took 4 days on 4 Cloud TPUs (16 TPU chips); BERT-large took 4 days on 16 Cloud TPUs (64 TPU chips). The optimizer was Adam with a learning rate warm-up over the first 10,000 steps, weight decay, and dropout of 0.1.

**What this code's training loop does:**
- **Pre-training** (`train_bert_pretraining`): AdamW optimizer, learning rate `5e-5`, weight decay `0.01`, gradient clipping at norm 1.0, batch size 8, run for 3 epochs over the 600-sentence subset. Each step feeds `input_ids`, `attention_mask`, `segment_ids`, `mlm_labels`, and `nsp_labels` through the model, sums the two losses, backpropagates, clips gradients, and steps the optimizer. Validation loss is computed the same way with gradients off.
- **Fine-tuning** (`train_bert_classification`): the pre-trained `BERTForPreTraining`'s encoder weights are reused inside `BERTForSequenceClassification`, a fresh classifier head is added on top of the pooled `[CLS]` output, and the whole thing is fine-tuned for 3 epochs at a smaller learning rate (`2e-5`), which mirrors real BERT fine-tuning practice (smaller LR than pre-training). Accuracy, not loss, is tracked as the validation metric here.
- Sentiment labels for fine-tuning are generated heuristically (counting positive/negative keyword hits per sentence, with a random label if tied) rather than coming from a real labeled dataset — this is a stand-in so the fine-tuning code path can be exercised without a labeled corpus.

### 6. Training Challenges

- **The pre-train/fine-tune mismatch with `[MASK]`.** If every masked token were literally replaced with `[MASK]` during pre-training, the model would never see `[MASK]` during fine-tuning (since fine-tuning uses real sentences with no masking) — this train/test mismatch would degrade fine-tuning quality. The 80/10/10 masking split (this file implements it in `_apply_masking`) exists specifically to reduce this mismatch by not letting the model over-rely on the presence of `[MASK]`.
- **NSP later turned out to be weak.** The NSP task's negative examples (random sentence pairs) are often trivially distinguishable just by topic mismatch, so the model could "solve" NSP without learning much fine-grained sentence coherence. RoBERTa (2019) ran ablations showing that removing NSP entirely and just using longer contiguous text sequences performed as well or better — this became one of the most cited "BERT could be improved" findings, and virtually every BERT successor (RoBERTa, ALBERT with Sentence Order Prediction instead, ELECTRA) dropped or replaced NSP.
- **Compute cost at real scale.** Even BERT-base's 4-day, multi-TPU pre-training run was expensive and slow to iterate on, which is part of why so much follow-up research (DistilBERT, ALBERT, RoBERTa) focused on getting BERT-level quality more cheaply.
- **In this demo's code specifically**, the challenge is the opposite: the corpus and vocabulary are so small that the model can easily overfit or produce noisy metrics, and heuristic sentiment labels (keyword counting) are a source of label noise that a real labeled dataset wouldn't have.

### 7. Performance & Evaluation

Real, published BERT results (from the original paper):

- **GLUE benchmark** (a suite of 9 sentence-understanding tasks): BERT-large scored 80.5 average, a 7.7-point absolute improvement over the prior best system (which averaged 72.8).
- **SQuAD v1.1** (question answering): BERT pushed F1 to 93.2, *surpassing* the reported human performance baseline of 91.2 F1 — a striking result at the time.
- **MultiNLI** (natural language inference): 86.7% accuracy, a 4.6-point absolute improvement over the previous state of the art.
- Overall, the paper reported **11 new state-of-the-art results** across different NLP tasks with a single pre-trained architecture just fine-tuned differently per task — that "one architecture, many tasks" outcome was itself part of the news.

The demo model in this file doesn't produce comparable numbers — it reports its own MLM+NSP validation loss and fine-tuning validation accuracy on a few hundred WikiText-2 sentences, which is only useful for confirming the training loop behaves sensibly (loss decreasing, accuracy above chance), not for benchmarking against the real BERT numbers above.

### 8. Impact — Why It Mattered

BERT is arguably the second most influential NLP paper after the original "Attention Is All You Need" Transformer paper. It established the **pre-train + fine-tune paradigm** as the default way to build NLP systems: pre-train one large model on unlabeled text, then cheaply fine-tune (or later, just prompt) it for any downstream task, instead of designing and training a bespoke architecture per task. This directly inspired an entire lineage of derivative models — RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa (all covered in file 017) — and its "deep bidirectional context" idea, along with the Transformer encoder block itself, became a foundational building block referenced by essentially every later encoder-based language model. It also helped popularize the idea that scaling up pre-training data and model size yields consistent quality gains, a theme that GPT-2 and GPT-3 (file 016) would push much further.

### 9. How To Explain This In An Interview

"BERT solved the problem that language models before it, like GPT-1, could only read text in one direction, which limited how well they could understand a word using context from both before and after it. BERT is just a stack of Transformer encoder layers — no causal masking — so every token can attend to every other token. To pre-train that bidirectionally without letting the model cheat, BERT uses Masked Language Modeling: it randomly picks 15% of tokens, and of those, 80% get replaced with a `[MASK]` token, 10% get replaced with a random word, and 10% are left alone, and the model has to predict the original word at each of those positions using full left-and-right context. It's trained jointly with a second task, Next Sentence Prediction, which asks whether two sentences are truly consecutive, using the pooled `[CLS]` token. BERT-base has 110 million parameters across 12 layers; BERT-large has 340 million across 24 layers; both were pre-trained on BooksCorpus plus Wikipedia, about 3.3 billion words, for roughly a million steps on TPU pods. Once pre-trained, the same weights get fine-tuned cheaply on labeled data for a specific task, like classification or QA — that pre-train-then-fine-tune pattern is BERT's biggest legacy. It delivered 11 new state-of-the-art NLP results on release, including F1 that beat the human baseline on SQuAD, and it directly set the template for RoBERTa, ALBERT, DistilBERT, and the rest of the encoder-model family that followed."
