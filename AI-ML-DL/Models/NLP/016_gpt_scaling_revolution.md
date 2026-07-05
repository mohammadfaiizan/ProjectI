# GPT Scaling Revolution: GPT-1, GPT-2, and GPT-3

This file (`016_gpt_scaling_revolution.py`) implements a small, scalable GPT-style decoder-only Transformer — causal self-attention, learned position embeddings, weight-tied output head, a text-generation function with temperature/top-k/top-p sampling, and a `get_gpt_config()` helper that provides several named sizes explicitly labeled as scaled-down stand-ins for GPT-1 and GPT-2. Only two of those configs (`tiny` and `small`) are actually trained in `main()`.

Three distinct, real, historically important released models are covered below: **GPT-1**, **GPT-2**, and **GPT-3**. All three share the same underlying architecture (a decoder-only autoregressive Transformer) implemented in this file — what changed across the three real releases was overwhelmingly scale (parameters, data, compute) and how the model was used at inference time (fine-tuning → zero-shot → few-shot).

---

## GPT-1 (Radford et al. 2018)

### 1. What Problem It Solved

Before GPT-1, most NLP systems still needed task-specific architectures: a QA model looked structurally different from a sentiment classifier, which looked different from an entailment model, and each was usually trained mostly from scratch on a labeled dataset for that one task. Labeled data is expensive and limited, so models trained this way often plateaued in quality. GPT-1's insight was: pre-train a single, general-purpose language model on lots of unlabeled text first (where language modeling itself is the training signal — no labels needed), and then reuse and fine-tune that same model for many different downstream tasks. This is the "generative pre-training" idea in GPT's name — it showed that a generic left-to-right language model could learn broadly useful representations of language that transfer to tasks it was never explicitly trained on, then get specialized cheaply.

### 2. Architecture — How It Works

**Big picture:** GPT-1 is a decoder-only Transformer — there's no encoder, and there's no bidirectionality. It reads text strictly left to right and, at every position, is only allowed to use words that came before it.

Step by step (as implemented in this file's `GPTModel`/`GPTBlock`/`GPTMultiHeadAttention`):

1. **Token + position embeddings** are looked up and summed (no segment embeddings — GPT doesn't need sentence-pair segment IDs since it isn't doing NSP-style tasks).
2. **Causal self-attention.** Inside `GPTMultiHeadAttention`, Q, K, V are all computed from a single combined linear layer (`c_attn`, projecting to `3 * d_model` at once — a small efficiency trick GPT itself used), then attention scores are computed as usual, but a **causal mask** — a lower-triangular matrix (`torch.tril`) — is applied before the softmax, setting all "look-ahead" positions to `-1e9` so they get zero probability after softmax. This is the one architectural change, relative to BERT's encoder, that defines GPT: token *i* can attend to tokens `1...i`, never to `i+1...n`.
3. **Transformer block.** Each block does LayerNorm → causal self-attention → residual add → LayerNorm → feed-forward (Linear → GELU → Linear) → residual add. (Note: this file's `GPTBlock` uses *pre-norm*, i.e., LayerNorm before attention/FFN rather than after — that's actually the GPT-2-style convention; the original 2018 GPT-1 paper used post-norm like BERT. The file's docstring flags this explicitly as "Pre-norm architecture (GPT-2 style)".)
4. **Weight tying.** The output projection (`lm_head`) shares its weight matrix with the input token embedding (`self.lm_head.weight = self.token_embedding.weight`) — a parameter-saving trick used across the GPT family.
5. **Training objective: predict the next token.** Given a sequence, the labels are just the same sequence shifted one position to the right. The loss is standard cross-entropy over vocabulary logits at every position, ignoring padding.
6. **Fine-tuning (what made GPT-1 distinctive at the time).** After pre-training as a plain language model, GPT-1 was fine-tuned per task by adding a small linear output layer and continuing training on labeled data, using a specific input format for each task type (e.g., wrapping premise/hypothesis pairs with special delimiter tokens for entailment). This is a supervised fine-tuning step — GPT-1 could not do useful zero-shot or few-shot tasks the way GPT-2/GPT-3 later could; it needed labeled examples and gradient updates to specialize.

### 3. Model Size & Parameters

**Real, published GPT-1 spec:** 117 million parameters, 12 decoder layers, hidden size (d_model) 768, 12 attention heads, context window of 512 tokens.

**What this code uses:** The `get_gpt_config('small')` entry is explicitly commented `# GPT-1 scale` and sets `d_model=256, num_heads=8, num_layers=6, d_ff=512, max_length=512`. In `main()`, this config is actually trained, but with `max_length` further capped to 64 tokens for the demo, a vocabulary capped at 3,000 tokens, batch size 4, and only 5 epochs over ~800 WikiText-2 sentences. That's a small fraction of GPT-1's 117M parameters — likely in the hundreds of thousands to low millions of parameters for this run.

**Why the gap:** The real GPT-1 needed 768-dimensional hidden states, 12 layers, and a much larger subword vocabulary (BPE) to represent general English well; this demo only needs to prove causal masking, next-token loss, and the AdamW+cosine-schedule training loop work correctly on a laptop, so every dimension is cut by roughly one to two orders of magnitude.

### 4. Dataset & What It Was Trained On

**Real GPT-1 training data:** BooksCorpus — about 7,000 unique unpublished books (novels, mostly), roughly 800 million words. Long, contiguous prose was chosen deliberately because it lets the model learn long-range dependencies across paragraphs, something sentence-shuffled corpora don't support as well.

**What this code uses:** WikiText-2, filtered to sentences between 8 and 35 tokens, further cut down to an 800-sentence training subset (160 validation, 160 test) tokenized at the word level with NLTK.

**The gap:** Roughly six orders of magnitude less text (hundreds of sentences vs. hundreds of millions of words), word-level tokenization with a 3,000-token closed vocabulary instead of BPE subword tokenization, and no long-document structure (WikiText-2 sentences here are treated independently, not as a continuous book). The demo dataset is sufficient only to exercise the causal LM training and generation code paths, not to learn real language competence.

### 5. Training Process

**Real objective:** Standard causal (autoregressive) language modeling — cross-entropy loss between predicted next-token distribution and the actual next token, summed/averaged over the sequence. GPT-1 was pre-trained this way, then fine-tuned with a task-specific supervised loss (often with the LM loss mixed in as an auxiliary term) on labeled datasets for each of the 12 tasks evaluated in the paper.

**What this code's training loop does** (`train_gpt`, shared across all configs in this file including the GPT-1-scale `small` config): AdamW optimizer with `betas=(0.9, 0.95)` and weight decay `0.1` (a training-recipe detail actually borrowed from the real GPT family), a cosine learning-rate schedule with linear warm-up over the first 10% of steps, gradient clipping at norm 1.0, learning rate `6e-4`. Labels are the input sequence shifted by one position (this shift is applied both in dataset construction and again inside `GPTModel.forward` via `shift_logits`/`shift_labels`), and loss ignores padding (`ignore_index=0`). Validation is measured as perplexity (`exp(average loss)`), which is the standard way to report language model quality.

### 6. Training Challenges

- **Limited pre-training data by later standards.** 800M words was large for 2018 but tiny compared to what followed — GPT-1's transfer-learning gains, while notable, were capped by both model size and corpus size.
- **Task-specific input formatting for fine-tuning.** Because GPT-1 has one universal architecture but needed to handle very different task shapes (single sentence, sentence pairs, multiple choice), the fine-tuning setup required carefully designed delimiter tokens and input transformations per task — this added engineering complexity that later zero-shot/few-shot GPT-2/GPT-3 approaches removed by not requiring architecture changes at all.
- **In this file's implementation**, the practical challenge is numerical/training stability with very short demo sequences and a small vocabulary — the code addresses this the standard way, with gradient clipping and a warm-up schedule, even though the real payoff of a warm-up schedule matters far more at GPT-1's actual scale.

### 7. Performance & Evaluation

GPT-1 achieved strong results for its time on 9 of 12 evaluated NLP tasks after fine-tuning, including then-state-of-the-art results on tasks like commonsense reasoning (Stories Cloze Test) and textual entailment — proving that generative pre-training followed by discriminative fine-tuning ("semi-supervised" learning) beat models trained from scratch on labeled data alone. It was, however, evaluated only in the fine-tuned setting; it was not designed or evaluated for zero-shot use the way GPT-2 later was.

### 8. Impact — Why It Mattered

GPT-1 established the recipe — pre-train a decoder-only Transformer as a language model, then fine-tune for downstream tasks — that GPT-2, GPT-3, and effectively every subsequent large language model would inherit and scale up. It also gave the field an early, concrete demonstration that a single generic architecture could beat task-specific ones once given enough unlabeled pre-training. In hindsight it's most notable as the first entry in a lineage where each successor mostly just scaled the same core idea up.

### 9. How To Explain This In An Interview

"GPT-1 was the first model to show that you could pre-train a plain decoder-only Transformer as a next-word predictor on a large unlabeled corpus, and then fine-tune that same model on much smaller labeled datasets to beat task-specific architectures. It uses causal self-attention — a triangular mask so each token can only see the tokens before it — stacked 12 layers deep, with 117 million parameters and a 768-dimensional hidden size. It was pre-trained on BooksCorpus, about 800 million words of long-form fiction, using a standard cross-entropy next-token loss, and then fine-tuned per task with a task-specific head and input formatting. Its big legacy is the pattern itself — generative pre-training plus fine-tuning — which is exactly the template GPT-2 and GPT-3 scaled up rather than replaced."

---

## GPT-2 (Radford et al. 2019)

### 1. What Problem It Solved

GPT-1 still needed labeled data and gradient-based fine-tuning to be useful on a new task. GPT-2's authors asked a different question: if you make the model and the pre-training data much bigger, does it start being able to do tasks *without* any task-specific fine-tuning at all, just by being prompted in natural language? GPT-2 was built to test — and ended up demonstrating — that scale alone could produce **zero-shot** task transfer: the same frozen, pre-trained model, with no fine-tuning and no extra parameters, could perform reading comprehension, summarization, translation, and question answering respectably just by being given a suitable text prompt.

### 2. Architecture — How It Works

GPT-2 uses the *same* decoder-only causal-attention architecture as GPT-1 (everything described in the GPT-1 section above applies unchanged: causal masked self-attention, token+position embeddings, weight tying between input embedding and output head), with two notable, real changes: it explicitly moved to **pre-norm** (LayerNorm before each sub-block rather than after — the convention this file's `GPTBlock` follows and calls out in a comment), and it scaled every dimension up substantially, releasing four sizes instead of one. Mechanically, there is nothing new to learn beyond GPT-1's causal decoder — the "innovation" in GPT-2 is almost entirely about scale (parameters and data) plus a change in *how* the model is used: as a frozen model prompted directly, not fine-tuned.

Concretely, zero-shot use works by framing a task as plain text continuation. For example, for summarization, you'd feed the model an article followed by the literal text "TL;DR:" and let it generate a continuation — no architecture change, no gradient updates, just a prompt design trick that leverages patterns already present in web text (where "TL;DR:" really does precede human-written summaries).

### 3. Model Size & Parameters

**Real, published GPT-2 specs** — GPT-2 was released as a family of four sizes:

| | Small | Medium | Large | XL ("GPT-2") |
|---|---|---|---|---|
| Parameters | 117M | 345M | 762M | 1.5B |
| Layers | 12 | 24 | 36 | 48 |
| Hidden size | 768 | 1024 | 1280 | 1600 |
| Attention heads | 12 | 16 | 20 | 25 |
| Context window | 1024 | 1024 | 1024 | 1024 |

The 1.5-billion-parameter XL version is the one usually meant by "GPT-2" in headlines.

**What this code uses:** `get_gpt_config('medium')` is explicitly commented `# GPT-2 small` (`d_model=512, num_heads=8, num_layers=8, d_ff=1024, max_length=1024`) and `get_gpt_config('large')` is commented `# GPT-2 large scale (scaled down for demo)` (`d_model=768, num_heads=12, num_layers=12, d_ff=2048, max_length=1024`). Neither of these two configs is actually trained in `main()` — only `'tiny'` and `'small'` (the GPT-1-scale config) are run. So in this file, GPT-2-scale configurations exist as named presets in `get_gpt_config()` but are not exercised by the demo training loop; they are there to illustrate the scaling comparison, not to reproduce GPT-2 training.

**Why the gap:** Even the *smallest* real GPT-2 checkpoint (117M params, coincidentally the same size as GPT-1) is far larger than anything this file trains, and the full 1.5B XL model is roughly three orders of magnitude larger than that. Training anything at GPT-2 scale on a laptop CPU with a few hundred sentences isn't feasible, so the file keeps GPT-2's configs as reference presets and only actually trains the smaller `tiny`/`small` configs.

### 4. Dataset & What It Was Trained On

**Real GPT-2 training data: WebText** — roughly 40GB of text scraped from about 8 million web documents, built by following outbound links from Reddit posts that had received at least 3 karma (used as a crude quality/human-interest filter), then deduplicated and cleaned. This was a deliberate move away from BooksCorpus toward a much larger, more diverse slice of the open internet.

**What this code uses:** The same WikiText-2 subset pipeline as elsewhere in this file — a few hundred to low-thousand filtered sentences, word-level tokenized, capped at a 3,000-token vocabulary.

**The gap:** WebText is roughly four to five orders of magnitude larger and vastly more topically diverse than the WikiText-2 subset used here; WebText's diversity (many domains, writing styles, and implicit "tasks" embedded in natural web text) is precisely what gave GPT-2 its zero-shot abilities — a small, narrow Wikipedia-derived subset cannot reproduce that effect regardless of model size.

### 5. Training Process

**Real objective:** Identical to GPT-1 — plain causal language modeling, cross-entropy on next-token prediction, no fine-tuning stage at all for the paper's headline results (the whole point was demonstrating zero-shot transfer from the pre-trained LM directly).

**Real training setup:** Trained on WebText with a BPE (byte-pair encoding) tokenizer (50,257 vocabulary), context window of 1024 tokens, using large batch sizes and many GPU/TPU-days of compute (exact hardware details were less fully disclosed than for GPT-3, but training the 1.5B model was understood to require a substantial multi-GPU cluster over an extended period).

**What this code's training loop does:** Uses the exact same `train_gpt` function described for GPT-1 above (AdamW, `betas=(0.9,0.95)`, weight decay 0.1, cosine schedule with warm-up, gradient clipping) — the code doesn't distinguish a separate "GPT-2 training procedure"; scale is the only thing that would differentiate a GPT-2-config run from a GPT-1-config run in this file, and as noted, the GPT-2-scale configs aren't actually trained in `main()`.

### 6. Training Challenges

- **Data quality at web scale.** Raw web text is extremely noisy; GPT-2's Reddit-karma filtering trick was a practical way to approximate "human-curated, reasonably high-quality" text without manual review of millions of documents — still an imperfect heuristic (it biases toward Reddit's demographic and content norms).
- **Staged, cautious release.** This is a deployment/release challenge rather than a training challenge, but it's a famous part of GPT-2's story: OpenAI initially withheld the full 1.5B model (releasing only the 117M version first in February 2019), citing concerns about misuse for generating convincing fake text/disinformation at scale, and released the full model later in November 2019 after a staged risk assessment.
- **Diminishing returns were *not* observed** — one of GPT-2's own findings was that perplexity kept improving smoothly as model size increased across all four sizes, with no sign of saturation, which is part of what motivated GPT-3's even larger scale-up.

### 7. Performance & Evaluation

GPT-2 set new state-of-the-art zero-shot results on several language modeling benchmarks (e.g., it improved perplexity on datasets like LAMBADA and others without any fine-tuning on them) and demonstrated qualitatively coherent long-form text generation that was noticeably more fluent than GPT-1's outputs. It performed zero-shot reading comprehension, summarization, and translation at levels that, while below dedicated supervised systems of the time, were surprisingly competitive for a model given no task-specific training signal at all.

### 8. Impact — Why It Mattered

GPT-2 was the moment "scale up a language model and it starts doing new things without being told how" became broadly visible outside the research community, partly because of the "too dangerous to release" media narrative. Technically, it validated that zero-shot transfer from a large enough LM was viable, which directly motivated GPT-3's even bigger bet on few-shot in-context learning, and it cemented the idea that data diversity (WebText's broad web crawl) matters as much as data volume.

### 9. How To Explain This In An Interview

"GPT-2 is architecturally the same causal decoder-only Transformer as GPT-1 — same causal self-attention, same next-token training objective — but scaled up to four sizes, up to 1.5 billion parameters, and trained on WebText, about 40GB of curated web text instead of BooksCorpus. The key result wasn't a new mechanism, it was a new usage pattern: without any fine-tuning, just by prompting the frozen pre-trained model with the right text pattern — like appending 'TL;DR:' for summarization — GPT-2 could do zero-shot reading comprehension, translation, and summarization reasonably well. That was the first strong evidence that scaling alone unlocks new capabilities beyond what you explicitly trained for, and it's why OpenAI staged the release of the full 1.5B model out of misuse concerns. It set up directly for GPT-3's much larger bet on the same idea, just pushed from zero-shot to few-shot."

---

## GPT-3 (Brown et al. 2020)

### 1. What Problem It Solved

GPT-2 showed zero-shot transfer was possible but its performance on most tasks still noticeably lagged behind supervised, fine-tuned systems. GPT-3 asked: what if you scale up again by another two orders of magnitude, and instead of relying purely on zero-shot prompting, let the model see a handful of labeled examples *directly in the prompt itself* (no gradient updates, no fine-tuning) — does performance jump close to fine-tuned-model quality? This is **few-shot in-context learning**, and it solved the practical problem that fine-tuning a giant model for every new task is expensive and slow: with GPT-3, you could instead just write a better prompt.

### 2. Architecture — How It Works

Same causal decoder-only Transformer family as GPT-1 and GPT-2 (same mechanism: causal self-attention, token+position embeddings, weight-tied LM head) — GPT-3's paper itself describes the architecture as nearly identical to GPT-2, with a minor detail (alternating dense and locally banded sparse attention patterns in some layers). The real story is scale plus a new way of using the frozen model at inference time.

**In-context / few-shot learning, worked example.** Instead of fine-tuning the model for, say, translation, you construct a single prompt that contains a few example input→output pairs followed by a new input, and let the model generate the continuation:

```
Translate English to French:

sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe en peluche
cheese =>
```

The model has never had its weights updated for translation. It is simply predicting the most likely continuation of this text, and because its pre-training corpus contained huge amounts of text with this kind of pattern (and much more general structure), it tends to complete it correctly (e.g., "fromage"). This is called "few-shot" because a handful of examples are shown in-context; GPT-3's paper also evaluates **one-shot** (a single example) and **zero-shot** (an instruction with no examples) variants of the same idea, and shows performance improves as you go from zero to one to few examples — all *without* any parameter updates. The "learning" happens purely through the forward pass attending over the examples in the prompt, not through backpropagation.

### 3. Model Size & Parameters

**Real, published GPT-3 spec:** 175 billion parameters, 96 decoder layers, hidden size (d_model) 12,288, 96 attention heads, context window of 2048 tokens. (GPT-3's paper actually trained and evaluated a family of 8 sizes ranging from 125M up to 175B; "GPT-3" almost always refers to the largest, 175B, "davinci"-class model.)

**What this code uses:** Nothing in this file trains anything close to GPT-3 scale — there is no `'xl'`/`'gpt3'` entry in `get_gpt_config()` at all. The largest config actually defined is `'large'` (commented as a scaled-down stand-in for GPT-2-large, not GPT-3), at `d_model=768, num_layers=12`, and even that config is never trained in `main()` — only `tiny` and `small` are run, each with a handful of layers and a hidden size of 128–256.

**Why the gap:** 175 billion parameters is not just "bigger," it's a fundamentally different computational regime — GPT-3 needed to be split across many GPUs/TPUs at once just to fit in memory (the model's weights alone in half precision take over 300GB), which is completely outside the scope of a single-machine educational script. This file's purpose is to demonstrate the *mechanism* (causal decoder, scaling configs, generation strategies) at a scale you can run in minutes on a laptop; reproducing GPT-3's actual scale would require a distributed training cluster, not a code change.

### 4. Dataset & What It Was Trained On

**Real GPT-3 training data mixture:** a blend of five sources, sampled non-uniformly (higher-quality sources oversampled relative to their raw size):
- Filtered Common Crawl (~410 billion tokens after quality filtering)
- WebText2 (~19 billion tokens, an expanded version of GPT-2's WebText)
- Books1 (~12 billion tokens)
- Books2 (~55 billion tokens)
- Wikipedia (~3 billion tokens)

In total, training sampled roughly 300 billion tokens across an epoch-weighted mixture of these sources (with higher-quality sources like Wikipedia and the Books corpora seen more than once relative to their share of raw bytes, and the huge Common Crawl pool downweighted despite being the largest raw source).

**What this code uses:** The same WikiText-2 subset used throughout this file (a few hundred sentences, word-level tokenized, 3,000-word vocabulary).

**The gap:** This is an almost incomparable gap — GPT-3's training mixture is roughly eight to nine orders of magnitude larger in raw token count than this demo's dataset, and spans multiple genres and quality tiers that were deliberately balanced against each other, something a single small Wikipedia-derived dataset can't approximate at any scale.

### 5. Training Process

**Real objective:** Still the same causal language modeling cross-entropy loss as GPT-1/GPT-2 — GPT-3 did not introduce a new training objective. All of its new capabilities came from scaling model size, data, and compute while keeping the training objective identical.

**Real training setup:** GPT-3's training compute was reported at approximately 3.14 × 10^23 FLOPs (often cited as about "3,640 petaflop/s-days"), using a large cluster of V100 GPUs with both model parallelism (splitting individual layers/matrices across GPUs, since 175B parameters cannot fit on one device) and data parallelism (splitting batches across groups of GPUs). Training cost has been widely estimated in the range of several million dollars for a single full training run.

**What this code's training loop does:** Uses the same shared `train_gpt` function (AdamW with `betas=(0.9,0.95)`, weight decay 0.1, cosine LR schedule with warm-up, gradient clipping) described under GPT-1 and GPT-2 above. There is no separate few-shot-training code path in this file for the `'language_modeling'` task — `main()` only exercises causal LM pre-training on the `tiny`/`small` configs. There is, however, a `_create_few_shot_sequences` method on `GPTDataset` that builds toy "Complete: ... -> ..." style examples (with a small set of hardcoded pattern pairs like "The color of grass is" → "green") to illustrate the *shape* of few-shot-style training data, though this method is never actually invoked from `main()` (the demo run uses `task='language_modeling'`, not `task='few_shot'`).

### 6. Training Challenges

- **Enormous compute and memory demands.** A 175B-parameter model cannot fit on a single accelerator's memory in full or half precision, so GPT-3 required both model (tensor/pipeline) parallelism and data parallelism working together across a large GPU cluster — a substantial distributed-systems engineering problem on top of the ML problem.
- **Cost and iteration speed.** At an estimated multi-million-dollar cost per full training run, there was very little room for trial-and-error at full scale — most hyperparameter and architecture decisions had to be validated at smaller scale first and extrapolated using scaling-law trends (a research direction that GPT-2's smooth, non-plateauing scaling curves had already suggested was reliable).
- **Data contamination risk.** With hundreds of billions of tokens crawled from the open web, avoiding leakage of benchmark test sets into the training data became a real methodological concern that the GPT-3 paper had to explicitly analyze and control for.
- **In this file**, none of these GPT-3-scale challenges apply directly since the code never attempts GPT-3 scale — the closest analogous "challenge" in the code is just making sure the shared training loop (warm-up schedule, gradient clipping) remains stable across the different config sizes it *does* support.

### 7. Performance & Evaluation

GPT-3's headline result was strong **few-shot** performance across dozens of NLP benchmarks without any fine-tuning — on some tasks (like certain cloze and completion tasks, and on-the-fly arithmetic) few-shot GPT-3 approached or matched fine-tuned state-of-the-art systems; on others (like some natural language inference benchmarks) it lagged behind dedicated fine-tuned models. It also demonstrated strong performance on tasks that hadn't been the focus of prior LM evaluation at all, such as generating code (a capability later productized as Codex/Copilot) and producing coherent long-form creative writing that human evaluators sometimes couldn't reliably distinguish from human-written text.

### 8. Impact — Why It Mattered

GPT-3 is the model that made "in-context learning" and "prompting" mainstream concepts, shifting how people interact with language models from "fine-tune it" to "just ask it well." It's also the direct technical ancestor of ChatGPT: OpenAI's InstructGPT work (and the alignment techniques behind it — supervised fine-tuning on human demonstrations, followed by reinforcement learning from human feedback, RLHF) was built by taking a GPT-3-class base model and further training it to follow instructions and be more helpful/safe, which is exactly the lineage that led to ChatGPT's public release in late 2022. In that sense, GPT-3's scale-driven capabilities were necessary but not sufficient — the alignment/instruction-tuning work on top of GPT-3-scale models is what turned "a very good text completer" into "a usable assistant," setting the stage for the current LLM-assistant era.

### 9. How To Explain This In An Interview

"GPT-3 kept the same causal decoder-only Transformer architecture as GPT-1 and GPT-2 — nothing structurally new — and instead pushed scale to 175 billion parameters, 96 layers, trained on a mixture of filtered Common Crawl, WebText2, two books corpora, and Wikipedia, totaling on the order of hundreds of billions of tokens. The big behavioral shift was few-shot in-context learning: instead of fine-tuning the model per task, you put a handful of input-output examples directly in the prompt, and the frozen model's forward pass alone is often enough to generalize to a new, similar input — no gradient updates at all. Training it required both model and data parallelism across a large GPU cluster, since the weights alone don't fit on one device, and the reported training compute was on the order of 3,640 petaflop/s-days. GPT-3's few-shot results approached fine-tuned-model quality on many benchmarks and showed surprising abilities like code generation. Its biggest downstream impact is that it became the base model that OpenAI's instruction-tuning and RLHF work built on top of to create InstructGPT and eventually ChatGPT — so GPT-3 is really the bridge between 'scaling laws produce better language models' and the current era of aligned, conversational LLM assistants."
