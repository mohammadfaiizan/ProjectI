# 005 — LSTM Breakthrough (Companion Notes)

Companion to `005_lstm_breakthrough.py`. This file implements an LSTM cell from scratch (`CustomLSTMCell`/`CustomLSTM`) to expose exactly what happens inside each gate, and also trains a standard `nn.LSTM`-based model (`PyTorchLSTM`) for comparison. It covers two things: the core LSTM gating mechanism, and — as the second required topic for this file — the Bidirectional LSTM extension, which this particular code file does not itself instantiate but which is directly relevant to explain alongside it (and which shows up later in this same repo, e.g. the encoder in `008_attention_mechanism_birth.py`).

## LSTM — Long Short-Term Memory (1997 / popularized 2010s)

### 1. What Problem It Solved

The previous file in this series showed the core failure of the vanilla RNN: backpropagation through time multiplies gradients by the recurrent weight matrix and by tanh's derivative at every single time step, so gradient signal from far in the past shrinks exponentially (vanishing gradients) or occasionally blows up (exploding gradients). In practice this meant vanilla RNNs could really only "remember" a handful of recent tokens — they were bad at tasks like "the subject of this sentence, mentioned 20 words ago, determines the verb conjugation now."

LSTM, introduced by Hochreiter and Schmidhuber in 1997 (and popularized much later, around 2013-2015, once compute and tooling caught up), fixed this not by tweaking the RNN's math but by redesigning the recurrence itself. It introduced a separate memory channel — the **cell state** — that is protected from the repeated squashing/multiplying that killed gradients in vanilla RNNs, plus a set of learned **gates** that control what gets written to, kept in, and read from that memory. The gates let the network learn, per input, whether to preserve information over long distances or let it decay — instead of every RNN update forcibly overwriting the entire hidden state through a `tanh`.

### 2. Architecture — How It Works

**Intuition first.** Picture the cell state as a conveyor belt running through the whole sequence, carrying a running memory forward. At each time step, three gates act like valves on that belt: one valve decides what old memory to throw away (**forget gate**), one decides what new information to add (**input gate**), and one decides how much of the memory to actually reveal as this step's output (**output gate**). Because the conveyor belt update is mostly additive (multiply-and-add, rather than repeatedly squashing through tanh the way the vanilla RNN's hidden state was), gradients have a much more direct path backward through time.

**The precise computation**, as implemented in `CustomLSTMCell.forward`. At each time step, given the current input `x_t` and the previous hidden state `h_{t-1}` and cell state `c_{t-1}`:

```
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_{t-1} + b_hi)   # input gate
f_t = sigmoid(W_if x_t + b_if + W_hf h_{t-1} + b_hf)   # forget gate
g_t = tanh   (W_ig x_t + b_ig + W_hg h_{t-1} + b_hg)   # candidate values ("new gate")
o_t = sigmoid(W_io x_t + b_io + W_ho h_{t-1} + b_ho)   # output gate

c_t = f_t * c_{t-1} + i_t * g_t     # update cell state
h_t = o_t * tanh(c_t)               # compute hidden state / output
```

All four gates are computed from a single fused matrix multiply in the code (`weight_ih` has shape `(4*hidden_size, input_size)` and is split into 4 chunks with `.chunk(4, 1)`) — that's just an implementation efficiency, mathematically it's the same four separate gates.

**A tiny worked example.** Suppose the cell state currently holds a value of `2.0` for some memory slot (say it's tracking "we're inside a quoted sentence"). At the next word, suppose the forget gate for that slot computes `f_t = 0.9` (meaning "mostly keep this memory"), the input gate computes `i_t = 0.1` (meaning "barely let new info in"), and the candidate value `g_t = -0.5`. Then the new cell state for that slot is:

```
c_t = 0.9 * 2.0 + 0.1 * (-0.5) = 1.8 - 0.05 = 1.75
```

The memory barely changed — the forget gate protected it. Compare that to a vanilla RNN, where the entire hidden state gets recombined through a `tanh` every step with no option to "mostly pass this value through unchanged." That option to pass values through almost untouched (`f_t` close to 1, `i_t` close to 0) is exactly what keeps gradients from vanishing over long sequences: the derivative of `c_t` with respect to `c_{t-1}` is approximately `f_t`, and if `f_t` stays close to 1 across many steps, gradients can flow backward through many time steps without shrinking to nothing.

**Why the forget-gate bias trick matters.** Both `CustomLSTMCell._init_weights` and `PyTorchLSTM._init_weights` in this code deliberately initialize the forget gate's bias to `1.0` instead of `0.0`. At the start of training, this makes `sigmoid(bias) = sigmoid(1.0) ≈ 0.73`, biasing the network toward "remember by default" rather than "forget by default" (which `sigmoid(0) = 0.5` would give, and which historically made early LSTM training slower/less stable, since a freshly initialized network with no reason yet to keep information would otherwise forget everything immediately).

### 3. Model Size & Parameters

Historically, the LSTMs that mattered most for NLP (e.g., in Google's 2016 production machine translation system) used hidden sizes in the hundreds to low thousands (512-1024 was common) stacked 2-8 layers deep, with tens to hundreds of millions of parameters once combined with large vocabularies and embeddings.

This repo's code uses, for both `CustomLSTM` and `PyTorchLSTM`: `embedding_dim=128`, `hidden_dim=256`, `num_layers=2`, `dropout=0.2`, with a vocabulary capped at 3,000 words. Working through the actual matrix shapes: the embedding table contributes about 384,000 parameters, the two stacked LSTM layers contribute roughly 920,000 parameters (each gate needs its own `(4*hidden) x input` and `(4*hidden) x hidden` weight matrices — note this is 4x the weight count of a same-sized vanilla RNN or roughly 1.33x a same-sized GRU, because LSTM has 4 gates), and the output projection back to the vocabulary contributes about 770,000 parameters. That totals to roughly **2.1 million parameters** — noticeably larger than the vanilla RNN in file 004 (about 700K-780K) at a comparable vocabulary size, which is expected: the gates are the whole point, and they cost 4x the recurrent weight matrices of a plain RNN.

This is still tiny by production standards, scaled down deliberately so the from-scratch `CustomLSTMCell` implementation trains in minutes on a laptop while still being big enough (256 hidden units, 100-token sequences) to demonstrate that the gates genuinely let gradients survive over longer sequences than the vanilla RNN could handle.

### 4. Dataset & What It Was Trained On

The original 1997 LSTM paper used small synthetic sequence-learning benchmarks designed specifically to test long-range dependency memorization. LSTM's real fame came later, in the 2010s, when it was applied to real NLP tasks like language modeling, speech recognition, and machine translation, trained on large text/speech corpora.

This code trains on **WikiText-2** (same dataset as file 004, loaded via Hugging Face `datasets`, lowercased and tokenized with NLTK). It uses a larger slice than file 004 — 20,000 training tokens, 4,000 validation tokens, 4,000 test tokens — and, notably, a longer sequence length of 100 tokens per training example (versus 50 for the vanilla RNN), specifically because handling longer sequences is the LSTM's whole selling point. Sequences here are non-overlapping (stride = sequence length, unlike file 004's overlapping windows), and the vocabulary is again capped at 3,000 words.

WikiText-2 remains a reasonable stand-in for the same reasons as file 004: it's real prose, fast to load, and lets this repo directly compare RNN vs. LSTM vs. GRU vs. attention-based models under matched conditions. It is far smaller and narrower in domain than the corpora used to train production-grade LSTMs, but that's an acceptable trade for a fast, reproducible learning exercise.

### 5. Training Process

**Objective.** Same as file 004: next-word language modeling with `nn.CrossEntropyLoss`.

**Optimization.** Adam optimizer, learning rate `0.001`, batch size 20 (smaller than the RNN's batch size of 32, to accommodate the longer 100-token sequences within memory), trained for 8 epochs. Gradient clipping is applied with a norm threshold of `1.0` after every backward pass.

**Two implementations trained side by side.** `CustomLSTM` builds the LSTM by manually looping over each of the 100 time steps in Python, calling `CustomLSTMCell` at each step for each of its 2 stacked layers — this is deliberately slow and transparent, so the gate computations are visible and inspectable. `PyTorchLSTM` uses `nn.LSTM`, which runs the same underlying math but through PyTorch's fused, optimized (often cuDNN-accelerated) implementation. Both are trained identically otherwise, which lets the code directly compare a "from scratch" understanding-focused implementation against a production-grade one — same architecture, same math, very different speed.

**Historical note.** The 1997 LSTM was originally trained with a truncated form of backpropagation for the gated parts, using tools far more limited than modern autodiff frameworks; the 2010s revival trained LSTMs the same way this file does — full BPTT with a modern gradient-based optimizer — just at vastly larger scale.

### 6. Training Challenges

The main challenge LSTM was built to solve — vanishing gradients — is mitigated but not fully eliminated. Gradients can still shrink or grow across time steps depending on how the gates behave, but the *possibility* of the forget gate holding near 1.0 gives the network a stable path for gradients to flow through when it needs one, which vanilla RNNs never had.

A practical training challenge specific to LSTMs is initialization sensitivity: the forget-gate-bias-to-1.0 trick used in this code's `_init_weights` methods is exactly the kind of detail that separates a stable LSTM from one that forgets everything at the start of training and struggles to recover. This code applies that trick to both the custom implementation and the standard PyTorch one.

Another practical challenge is simply computational cost: each LSTM cell has 4 full gate computations instead of a single RNN update, which is roughly 4x the matrix multiplications per time step. The code's `analyze_gradient_flow` function collects per-parameter gradient norm/mean/std statistics specifically to let you inspect whether gradients are staying healthy (not collapsing to near-zero) throughout training, which is the concrete, measurable version of "did the gating mechanism actually help."

### 7. Performance & Evaluation

Evaluation uses the same **perplexity** metric as file 004 (`exp` of average cross-entropy loss on held-out data), which is the standard way to score any language model — lower perplexity means the model assigns higher probability to the actual next word on average.

Historically, LSTM-based language models produced dramatic perplexity improvements over both n-gram models and vanilla RNNs once trained at scale — this was the architecture behind state-of-the-art language modeling and machine translation results throughout roughly 2014-2017, before Transformers took over. In this repo's small-scale training run, the expected (and demonstrable) result is that the LSTM should handle the longer 100-token sequences noticeably better than the vanilla RNN handled its 50-token sequences relative to model size, and its gradient statistics (from `analyze_gradient_flow`) should look healthier — less collapsed toward zero — than the vanilla RNN's did.

### 8. Impact — Why It Mattered

LSTM was the architecture that made deep sequence modeling actually work in practice. It became the default choice for machine translation, speech recognition, text generation, and any task requiring modeling of long sequences, throughout the first half of the 2010s. Google Translate's neural rewrite (2016) was built on stacked LSTMs. Crucially for the rest of this series: the LSTM's encoder/decoder building blocks are exactly what get assembled into the Seq2Seq architecture (file 007) and later wrapped with attention (file 008) — the LSTM's success at handling individual sequences is a direct prerequisite for those breakthroughs, and LSTM cells (or their close cousin GRU, file 006) remained the standard recurrent building block until self-attention (file 010) and the Transformer made recurrence unnecessary entirely.

### 9. How To Explain This In An Interview

"The vanilla RNN had a vanishing gradient problem because its hidden state gets fully rewritten through a tanh at every time step, so gradients shrink exponentially the further back you go. LSTM fixes this by adding a protected cell state and three gates — forget, input, and output — that are computed with sigmoids and combined with the cell state through mostly additive, multiplicative operations instead of repeated squashing. The forget gate can learn to hold close to 1.0, which lets gradients flow backward through many time steps nearly unchanged, giving the network a real path to learn long-range dependencies. I implemented the LSTM cell from scratch to see exactly how the four gates are computed from a single fused weight matrix, and I made sure to initialize the forget gate's bias to 1.0, which is a well-known trick to keep the network 'remembering by default' rather than forgetting everything at the start of training. I trained it as a language model on WikiText-2 with Adam, cross-entropy loss, and gradient clipping, using 100-token sequences specifically because that's the length where LSTM's advantage over vanilla RNN should show up. I also compared my from-scratch implementation against PyTorch's built-in `nn.LSTM` to confirm the math matched, and I tracked gradient statistics to verify gradients stayed healthier than they did for the vanilla RNN. This gating idea — learn what to keep and what to discard — is the direct ancestor of GRU's simplified two-gate version and shows up again conceptually in attention mechanisms, which also learn what to 'keep' (attend to) at each step."

## Bidirectional LSTM

### 1. What Problem It Solved

A standard (unidirectional) LSTM reads a sequence in one direction — left to right — so at any given position, its hidden state only knows about words that came *before* it, not after. For language modeling (predicting the next word) that's correct and necessary — you shouldn't be allowed to see the future. But for many other tasks — understanding what a word *means* in context, tagging parts of speech, or, most relevantly for this series, encoding a whole input sentence before decoding it into another sequence — a word's meaning can depend just as much on what comes after it as what comes before. A unidirectional LSTM encoding "The bank by the river" versus "The bank raised interest rates" only sees "bank" with the words before it, so it can't yet fully disambiguate the word "bank" until it reads further right, and even then, that later context never flows backward into the earlier hidden states.

Bidirectional LSTM's fix is to run two separate LSTMs over the same sequence — one reading left-to-right, one reading right-to-left — and combine their hidden states at each position. That way, the representation at every position has access to the full sentence, both what came before and what came after.

### 2. Architecture — How It Works

**Intuition first.** Imagine two readers going through the same sentence at the same time: one starts from the first word and reads forward, the other starts from the last word and reads backward. At every word position, you get each reader's summary-so-far, and you glue them together. Now each word's representation reflects context from both directions, not just one.

**The precise computation.** A Bidirectional LSTM runs the standard LSTM recurrence (the four gates described above) twice per layer, independently:

```
h_forward_t  = LSTM_forward(x_t, h_forward_{t-1})    # reads left to right
h_backward_t = LSTM_backward(x_t, h_backward_{t+1})  # reads right to left
h_t = concat(h_forward_t, h_backward_t)               # combined representation at position t
```

Each direction has its own independent set of weight matrices (its own forget/input/output gates and cell state) — the backward LSTM is not "the same LSTM run in reverse," it's a separately trained LSTM that happens to process the sequence in reverse order. The output at each position is the concatenation of the two hidden states, so if each direction has `hidden_dim` units, the combined output has `2 * hidden_dim` units.

In this repo, `PyTorchLSTM` (in `005_lstm_breakthrough.py`) explicitly sets `bidirectional=False` — this file trains only unidirectional LSTMs, which is the right choice for its task (next-word language modeling, where looking at future tokens would leak the answer). But the very next attention-focused file in this series, `008_attention_mechanism_birth.py`, does use a bidirectional LSTM (`AttentionEncoder`, with `bidirectional=True`) as its encoder — because encoding a whole input sentence before translating/transforming it doesn't have the "don't peek at the future" restriction, and giving the encoder full-sentence context measurably helps the attention mechanism build better representations to attend over.

### 3. Model Size & Parameters

Historically, bidirectional LSTMs became a near-default choice for encoders in sequence-to-sequence models and for tagging tasks (POS tagging, named entity recognition) throughout the mid-2010s, typically with hidden sizes of 256-512 per direction.

Because a bidirectional LSTM is really two independent LSTMs, its parameter count is (approximately) double that of a unidirectional LSTM of the same hidden size — you pay for a second full set of gate weight matrices. Concretely, using this repo's own dimensions elsewhere in the series (`hidden_dim=256`, `embedding_dim=128`, as used in the `AttentionEncoder` in file 008): a single-direction, 1-layer LSTM of that size costs roughly 395,000 parameters in its recurrent weights; running it bidirectionally roughly doubles that to about 790,000 parameters for the recurrent layer alone, before the embedding table or any output layers are counted. This file (005) does not pay that cost anywhere, since its `PyTorchLSTM` is explicitly unidirectional — the point here is understanding the concept and the parameter-cost trade-off, which becomes concretely relevant once you reach the encoder in file 008.

### 4. Dataset & What It Was Trained On

Bidirectional LSTMs were historically trained on the same kinds of NLP tasks as unidirectional ones, but skewed toward tasks that benefit from full-context understanding: part-of-speech tagging, named entity recognition, sentiment classification, and — most relevant here — as encoders inside sequence-to-sequence and attention-based translation systems (e.g., Bahdanau et al. 2015's original attention paper used a bidirectional RNN encoder).

Within this repo, no dataset is used for a bidirectional LSTM in file 005 specifically (since `005_lstm_breakthrough.py` only trains unidirectional models). The relevant demonstration of a bidirectional LSTM being trained on WikiText-2 sentence data happens in `008_attention_mechanism_birth.py`, where it serves as the encoder for an attention-based sequence-to-sequence model — a natural fit, since that file is not doing left-to-right language modeling but rather encoding a whole input sequence before generating an output sequence.

### 5. Training Process

The training procedure for a bidirectional LSTM is identical in spirit to a unidirectional one: forward pass, compute loss (cross-entropy for classification/generation tasks), backpropagate, update with an optimizer like Adam, clip gradients if needed. The only architectural difference during backpropagation is that gradients now need to flow backward through *two* unrolled recurrences per layer — one for each direction — which roughly doubles the backward-pass compute per layer compared to a unidirectional LSTM of the same hidden size.

One thing to get right in implementation (and something the encoder in file 008 handles explicitly) is how to combine the bidirectional encoder's final hidden state into a single hidden state usable by a unidirectional decoder — file 008's `AttentionSeq2Seq.forward` does this by summing the forward and backward final hidden states (`h[0::2] + h[1::2]`) rather than concatenating them, so the decoder's expected hidden dimension doesn't need to double.

### 6. Training Challenges

Bidirectional LSTMs inherit all of the unidirectional LSTM's training considerations (forget-gate bias initialization, gradient clipping, the general BPTT cost) but double the per-layer computation and parameter count, which means slower training and higher memory use for the same hidden size.

The more architecture-specific challenge is the one already mentioned: **bidirectional encoders cannot be used for autoregressive generation of the same sequence**, because the backward direction requires seeing future tokens that don't exist yet during generation. This is why bidirectional LSTMs are used as *encoders* (which see the whole input at once) rather than as *decoders* (which generate one token at a time and must not see the future) — mixing this up is a common conceptual mistake. It's also why file 005's language-modeling task correctly avoids bidirectionality (predicting the next word bidirectionally would trivially "cheat" by looking at the very word it's supposed to predict), while file 008's encoder correctly embraces it (the whole input sentence is already fully available before decoding starts).

### 7. Performance & Evaluation

Bidirectional LSTMs are evaluated with whatever metric fits the downstream task — perplexity for language modeling (though, as noted, bidirectionality isn't appropriate there), accuracy/F1 for tagging and classification tasks, and, when used as an encoder inside a sequence-to-sequence system, indirectly through the quality of the final generated output (BLEU for translation, or the sequence-to-sequence-style validation loss used in file 007/008 of this repo).

Historically, switching from unidirectional to bidirectional encoders in tagging tasks and in Bahdanau-style attention translation models produced measurable quality improvements, because the richer, full-context representations at each input position gave downstream components (the tagger, or the attention mechanism) much more informative vectors to work with.

### 8. Impact — Why It Mattered

Bidirectional LSTM became the standard choice for "encoder" roles throughout the sequence-to-sequence and attention era (roughly 2014-2017): whenever a model needed to build a rich representation of an already-complete input (a sentence to translate, a sentence to classify, a document to summarize), a bidirectional recurrent encoder was the default. This is directly why the attention mechanism in file 008 pairs a bidirectional LSTM encoder with a unidirectional decoder — the encoder's job (understand the whole input) benefits from bidirectionality, while the decoder's job (generate output one token at a time, left to right) forbids it. The core idea — that context can and should flow from both directions when the whole input is already available — resurfaces later in a much more powerful form in BERT (2018), which is essentially a "fully bidirectional, all-positions-at-once" idea taken to its logical extreme using self-attention instead of recurrence.

### 9. How To Explain This In An Interview

"A unidirectional LSTM only has access to context from before the current position, which is a limitation whenever you need to understand a word using both its left and right context — like when encoding a full sentence before translating it. Bidirectional LSTM solves that by running two independent LSTMs over the same sequence, one forward and one backward, and concatenating (or otherwise combining) their hidden states at each position, so every position's representation reflects the entire sequence. It roughly doubles the parameter count and backward-pass compute of a same-sized unidirectional LSTM, since you're training two full sets of gates instead of one. The key constraint is that bidirectionality only makes sense when the whole sequence is already available — it's the right choice for an encoder that sees a complete input sentence, and the wrong choice for a decoder or a next-token predictor, because the backward direction would require peeking at tokens that haven't been generated yet. That's exactly why, later in this series, the attention-based sequence-to-sequence model pairs a bidirectional LSTM encoder with a strictly unidirectional decoder, and it's also the conceptual seed that BERT later grew into a much larger, self-attention-based, fully bidirectional model."
