# 013: The Transformer From Scratch -- Companion Notes

This file documents `013_transformer_from_scratch.py`, a from-scratch PyTorch implementation of
the original Transformer architecture from Vaswani et al.'s 2017 paper "Attention Is All You
Need." This is arguably the single most important file in this entire NLP collection: every
major modern language model (BERT, GPT-2/3/4, T5, and effectively everything since) is a
descendant of this architecture. It gets the deepest treatment of any file in this series.

---

## The Transformer (2017)

### 1. What Problem It Solved

By 2017, the state of the art in sequence-to-sequence tasks (machine translation, in particular)
was RNN or LSTM encoder-decoders, usually enhanced with an attention mechanism so the decoder
could look back at relevant parts of the source sentence (see files 007-010 in this collection).
This family of models had two compounding limitations:

1. **Sequential computation blocks parallelization.** An RNN must compute hidden state `h_t`
   before it can compute `h_{t+1}`, because `h_{t+1}` depends on `h_t`. This is true whether
   you're training on a single sentence or a billion sentences -- inside each sequence, the steps
   are strictly ordered. On modern GPUs, which are designed to do huge numbers of independent
   computations at once, this sequential dependency is a major waste of hardware: most of the
   chip sits idle waiting for the previous time step to finish. Training therefore scales poorly
   with sequence length and dataset size.
2. **Long-range dependencies are hard to learn**, even with gated units like LSTM/GRU. Every time
   information from word 1 needs to influence word 50, it has to pass through 49 sequential
   transformation steps, and at each step some of that information can get diluted or distorted.
   Attention-augmented seq2seq models (file 008-010) helped the *decoder* look back at any encoder
   position directly, but the *encoder* itself was still an RNN internally, so it still had to
   propagate information about early words through many sequential steps to build a
   representation of a later word's context.

ConvS2S (file 011) attacked problem #1 with convolutions -- local, parallel operations stacked to
build up range. The Transformer went further: it removed recurrence and convolution *entirely*.
Its core claim, right there in the title of the paper, is that a mechanism the field had already
been using as a *helper* on top of RNNs -- attention -- was actually **sufficient on its own** to
both build representations of a sequence (encoder self-attention) and generate an output sequence
conditioned on it (decoder self-attention + cross-attention), with no recurrent or convolutional
backbone needed at all. Because self-attention lets every position directly look at every other
position in a single computation (rather than through a chain of sequential steps or a stack of
local convolutions), it solves both problems simultaneously: the computation for all positions in
a layer can be done in parallel (matrix multiplications), and any two positions -- no matter how
far apart -- are only ever "one hop" away from each other in terms of information flow.

### 2. Architecture -- How It Works

**Core intuition first.** Forget the equations for a second. Self-attention answers the question,
for every word in a sentence: "given everything else in this sentence, what should I actually pay
attention to in order to understand my own meaning here?" For the word "it" in "The animal didn't
cross the street because it was too tired," self-attention lets "it" directly look at every other
word and learn to assign a high weight to "animal" -- in one step, not by slowly accumulating
information through 8 sequential RNN transitions. The Transformer does this "everyone looks at
everyone" operation many times, in parallel, from different learned perspectives (that's the
"multi-head" part), and stacks several rounds of it on top of each other (encoder/decoder layers),
interleaved with simple per-position feed-forward transformations, residual connections, and
normalization to keep the whole deep stack trainable.

Now, precisely, piece by piece, in the order the model is actually assembled in the code:

#### Embeddings and positional encoding

A Transformer has no built-in notion of word order the way an RNN or a causally-padded
convolution does -- self-attention treats the input as an unordered set of vectors unless you
explicitly tell it where each word sits. So:
1. Each input token index is turned into a `d_model`-dimensional embedding vector
   (`nn.Embedding`), and this embedding is **scaled by `sqrt(d_model)`** before anything else
   happens (`x = self.embedding(src) * math.sqrt(self.d_model)`). This scaling, specified in the
   original paper, roughly balances the relative magnitude of the token embedding against the
   positional encoding that gets added next.
2. A **fixed, non-learned positional encoding** is added to every embedding, using the classic
   sine/cosine formula:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

   Here `pos` is the position in the sequence (0, 1, 2, ...) and `i` indexes pairs of dimensions
   within the embedding. Every dimension of the encoding oscillates at a different frequency --
   early dimensions oscillate fast (change a lot between adjacent positions), later dimensions
   oscillate slowly (change little between adjacent positions, more between distant positions).
   Because sine and cosine of a sum can be expressed in terms of sine and cosine of the parts
   (angle-addition identities), the *relative* position between any two tokens is, in principle,
   linearly recoverable from their positional encodings -- this is why the paper picked a
   sinusoidal, rather than arbitrary, encoding: it gives the model a mathematically consistent way
   to reason about relative offsets, and (unlike a learned embedding table) it can be evaluated at
   sequence lengths longer than anything seen in training. In `PositionalEncoding`, this table is
   precomputed once up to `max_length` positions and stored as a non-trainable buffer, then simply
   added to the token embeddings at the start of both the encoder and the decoder.

#### Scaled dot-product attention (the atomic operation)

This is the single most important formula in the whole file:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

In plain language: for every query vector, compare it (via dot product) against every key
vector to get a raw similarity score for each key; scale those scores down by `sqrt(d_k)`
(`d_k` is the dimensionality of each attention head, i.e. `d_model / num_heads`); turn the scores
into a probability distribution with softmax; then take a weighted average of the value vectors,
weighted by that probability distribution. The result is: "this position's new representation is
a blend of the value vectors of whichever other positions its query vector matched most strongly
with." The scaling by `sqrt(d_k)` matters because without it, as `d_k` grows, the dot products
(sums of `d_k` terms) can grow large in magnitude, pushing softmax into a very peaked,
near-one-hot regime where gradients vanish; dividing by `sqrt(d_k)` keeps the score distribution
in a numerically well-behaved range regardless of dimensionality.

**Small worked numeric example.** Suppose `d_k = 2` and we have two tokens, "cat" and "sat," in
the sequence. Say the query vector for "cat" is `Q = [1, 0]`, and the two keys are
`K_cat = [1, 0]` and `K_sat = [0, 1]`, with corresponding values `V_cat = [1, 2]` and
`V_sat = [3, 4]`.

1. Raw dot-product scores: `Q . K_cat = 1*1 + 0*0 = 1`, and `Q . K_sat = 1*0 + 0*1 = 0`.
2. Scale by `sqrt(d_k) = sqrt(2) ≈ 1.414`: scores become `[1/1.414, 0/1.414] = [0.707, 0]`.
3. Softmax: `exp(0.707) ≈ 2.028`, `exp(0) = 1`, sum `≈ 3.028`, giving weights
   `[2.028/3.028, 1/3.028] ≈ [0.670, 0.330]`.
4. Weighted sum of values: `0.670 * [1, 2] + 0.330 * [3, 4] = [0.670 + 0.990, 1.340 + 1.320]
   = [1.660, 2.660]`.

So the new representation for "cat" after this attention step is roughly `[1.66, 2.66]` -- a blend
that leans more heavily toward its own value (because its query matched its own key best) but
still pulls in some information from "sat."

#### Multi-head attention

Rather than doing this attention computation once with the full `d_model`-dimensional vectors,
`MultiHeadAttention` splits `d_model` into `num_heads` independent, smaller subspaces (each of
size `d_k = d_model / num_heads`, e.g. `512 / 8 = 64` in the original paper), runs scaled
dot-product attention **separately and in parallel** within each subspace, and then concatenates
all the heads' outputs back together and passes them through one more linear layer (`w_o`). Why
bother splitting into heads at all, instead of one big attention operation? Because it lets
different heads specialize: one head might learn to track subject-verb agreement, another might
track coreference (like "it" -> "animal" above), another might track adjacent-word relationships.
A single attention operation, softmaxing over the *whole* `d_model` dimensionality, can really
only express one dominant "pattern of relevance" per position; multiple smaller heads let the
model represent several different, simultaneously-useful patterns of relevance. Concretely, the
code:
1. Projects the input `query`, `key`, `value` (all `d_model`-dimensional) through separate linear
   layers `w_q`, `w_k`, `w_v`.
2. Reshapes each into `(batch, num_heads, seq_len, d_k)` so the heads become an independent batch
   dimension.
3. Runs scaled dot-product attention (including masking, described below) independently per head.
4. Concatenates the `num_heads` outputs back into a single `d_model`-dimensional vector per
   position, then applies the final output projection `w_o`.

#### Position-wise feed-forward network

After the attention sub-layer in every encoder/decoder layer, there is a small two-layer MLP
applied **independently and identically to every position**:

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

That is: a linear layer up to a larger hidden size `d_ff` (2048 in the original paper), a ReLU
nonlinearity, then a linear layer back down to `d_model`. Attention is what lets positions
exchange information with each other; the feed-forward network is where each position, now
carrying information gathered from attention, gets to do its own nonlinear processing. It is the
same two weight matrices applied at every position (so it doesn't add parameters per sequence
length), but it is a *large* part of the model's total parameter count and representational
capacity.

#### Residual connections and layer normalization

Every sub-layer (self-attention, and separately the feed-forward network) is wrapped the same way:

```
output = LayerNorm(x + Dropout(Sublayer(x)))
```

The `x + Sublayer(x)` part is the **residual (skip) connection** -- the sub-layer only has to
learn a *change* to add to the input, rather than having to reproduce the entire input from
scratch, which makes gradients flow much more easily through deep stacks (the same idea used in
ResNets, and in ConvS2S's residual convolutions in file 011). **Layer normalization** then
renormalizes the resulting vector (per position, across the feature dimension) to have stable
mean and variance, which keeps training numerically stable as you stack many layers. Note that
this code applies LayerNorm *after* adding the residual (`norm(x + dropout(sublayer(x)))`), which
is the original paper's "post-LN" formulation -- many later Transformer variants moved the
LayerNorm to *before* the sub-layer instead ("pre-LN"), because pre-LN tends to be more stable to
train at large depth, but this file faithfully reproduces the original 2017 post-LN design.

#### Full encoder stack (`TransformerEncoder` / `TransformerEncoderLayer`)

Putting it together, one encoder layer does, in order:
1. Multi-head **self-attention** over the encoder's own input (query, key, and value all come
   from the same sequence) -- this lets every source position build a representation informed by
   every other source position.
2. Residual connection + LayerNorm.
3. Position-wise feed-forward network.
4. Residual connection + LayerNorm.

The encoder stacks `num_layers` (6 in the original paper) of these identical layers on top of
each other. Each layer's self-attention can only attend around *padding* positions (masked out via
a padding mask so real tokens don't attend to `<PAD>`), but there is no causal restriction in the
encoder -- every source position can see every other source position, in both directions, since
the whole source sentence is available up front (unlike generation, which happens token by token).

#### Full decoder stack (`TransformerDecoder` / `TransformerDecoderLayer`)

One decoder layer has *three* sub-layers, not two:
1. **Masked multi-head self-attention** over the decoder's own (partially generated) sequence.
   The mask here is a **causal mask** -- a lower-triangular matrix of 1s and 0s that forces
   position `t` to only attend to positions `<= t`. This is what makes the decoder autoregressive:
   during training, even though the entire target sequence is fed in at once for parallel
   computation (teacher forcing), each position is prevented from "seeing" future tokens it
   hasn't generated yet, so the training objective matches what will actually be possible at
   inference time.
2. Residual connection + LayerNorm.
3. **Encoder-decoder cross-attention:** here is where the decoder actually looks at the source
   sentence. The *query* comes from the decoder's own (masked-self-attended) hidden state, but
   the *key* and *value* come from the **encoder's output**. This is functionally the same role
   that classic attention mechanisms (Bahdanau, Luong -- files 008/009) played bolted onto RNN
   decoders, except here it's just another instance of the exact same multi-head attention
   building block used everywhere else in the model, rather than a bespoke mechanism.
4. Residual connection + LayerNorm.
5. Position-wise feed-forward network.
6. Residual connection + LayerNorm.

The decoder also stacks `num_layers` (6 in the paper) of these three-sublayer blocks. The final
decoder output is projected through a linear layer to vocabulary-sized logits
(`output_projection`), from which a softmax gives a probability distribution over the next token.

#### Masking, end to end

`Transformer.forward` builds two kinds of masks: a **padding mask** (`create_padding_mask`,
marking which positions are real tokens vs. `<PAD>`) for both source and target, and a **causal
mask** (`create_causal_mask`, a triangular matrix) for the decoder's self-attention. The decoder's
self-attention mask is the *combination* (logical AND, `tgt_padding_mask & tgt_causal_mask`) of
"don't attend to padding" and "don't attend to the future" -- both restrictions have to hold at
once. The cross-attention mask only needs the source padding mask, expanded across all target
positions, since the decoder is allowed to look at *any* (non-padding) source position regardless
of target position.

#### How training and generation differ

During **training**, the entire target sequence (shifted by one position, via teacher forcing --
`decoder_input = tgt[:, :-1]`, `decoder_target = tgt[:, 1:]`) is fed into the decoder in a single
forward pass. Thanks to the causal mask, this is still "fair" -- position `t`'s prediction only
ever had access to positions `< t` -- but it means the entire target sequence's loss can be
computed with one matrix-multiply-heavy forward and backward pass, no sequential loop needed. This
parallel-training property is the whole reason the Transformer trains so much faster than an RNN
decoder, which would have to unroll one step at a time even during training.

During **generation** (`Transformer.generate`), there is no target sequence to feed in -- the
model has to invent it one token at a time. The code starts with just the `<SOS>` token, runs a
full decoder forward pass, takes the logits at the *last* position, greedily picks the
highest-probability next token (`argmax`), appends it to the sequence, and repeats, growing the
causal mask each step, until it produces `<EOS>` or hits `max_length`. Note that although the
function signature includes a `beam_size` parameter (suggesting beam search, which keeps several
candidate sequences alive at once and picks the overall best), the implementation only actually
performs greedy decoding regardless of the value passed for `beam_size` -- beam search is not
wired up in this code, which is a simplification worth being upfront about.

### 3. Model Size & Parameters

**Original paper -- base model:** `d_model=512`, `num_heads=8` (`d_k = d_v = 64`),
`num_encoder_layers = num_decoder_layers = 6`, `d_ff=2048`, dropout 0.1, roughly **65 million**
parameters, trained on 8 NVIDIA P100 GPUs for about 12 hours (100,000 training steps).
**Big model:** `d_model=1024`, 16 heads, `d_ff=4096`, same depth, roughly **213 million**
parameters, trained for 300,000 steps (about 3.5 days on 8 P100s).

**This code's configuration:** `d_model=256`, `num_heads=8` (so `d_k=32`),
`num_encoder_layers = num_decoder_layers = 3` (half the paper's depth, explicitly commented
"Reduced for demo" in `main()`), `d_ff=512` (a quarter of the paper's base `d_ff`), dropout 0.1,
and a vocabulary capped at 4,000 tokens (versus roughly 37,000 BPE tokens in the paper's shared
vocabulary). The code prints the exact resulting parameter count at runtime via
`count_parameters(transformer)`, which comes out to a few million parameters -- one to two orders
of magnitude smaller than the base paper model, driven mostly by the much smaller `d_model`,
`d_ff`, and vocabulary size.

**Why scaled down:** this is meant to run end-to-end on a laptop, on a tiny slice of WikiText-2,
in minutes rather than the paper's multi-GPU, multi-hour/day training run. Shrinking `d_model`,
`d_ff`, depth, and vocabulary all reduce compute and memory roughly proportionally (attention cost
scales with `d_model` and sequence length; feed-forward cost scales with `d_model * d_ff`), while
every structural component of the real architecture -- multi-head attention, positional encoding,
the full six-sublayer encoder/decoder stack pattern, masking, residuals, LayerNorm -- is still
present and functioning exactly as designed.

### 4. Dataset & What It Was Trained On

**Original paper:** the Transformer was trained and evaluated on **WMT 2014 English-German**
(about 4.5 million sentence pairs) and **WMT 2014 English-French** (about 36 million sentence
pairs), using byte-pair encoding (BPE) to build a shared source/target subword vocabulary of
roughly 32,000-37,000 tokens. These are large, professionally curated, genuinely bilingual parallel
corpora -- the gold standard for machine translation research at the time.

**This code's demo:** WikiText-2 (`load_wikitext2_dataset`), an English-only Wikipedia corpus.
The code lowercases text, splits on periods, tokenizes with NLTK, keeps sentences of length 5-30
tokens, and uses only the first 1,000 training / 200 validation / 200 test sentences, with a
4,000-word vocabulary (`build_vocabulary(..., vocab_size=4000)`). Since there is no real bilingual
data available in this demo, `TransformerDataset` manufactures a synthetic **"translation"** task
out of monolingual English sentences: for each sentence it randomly applies one of several
transformations -- reversing word order, replacing a handful of words with hand-picked synonyms,
adding a "start"/"end" marker, or upper-casing longer words -- and treats the transformed sentence
as the "target." (A `'language_modeling'` task, which would instead train the model to predict
the next word given previous words, is also implemented in `TransformerDataset._create_lm_pairs`,
but `main()` only actually runs the first entry of `tasks = ['translation', 'language_modeling']`,
so the translation-style synthetic task is what's actually trained on by default.)

**The gap:** real translation quality depends on the model learning a genuine semantic mapping
between two languages across tens of millions of sentence pairs. This demo's synthetic task
(word reversal, synonym swaps) is solvable with much less data and much less model capacity, but
it still requires the model to learn *something* about attending across positions and
reordering/substituting tokens correctly, which is enough to demonstrate that every piece of the
real architecture (encoder self-attention, causal decoder self-attention, cross-attention,
masking) is wired together correctly and can learn a non-trivial sequence transformation.

### 5. Training Process

**Objective/loss:** token-level cross-entropy loss (`nn.CrossEntropyLoss(ignore_index=0)`) between
the decoder's predicted next-token distribution at each position and the true next token, with the
padding index (0) excluded from the loss.

**Optimizer:** Adam, with `betas=(0.9, 0.98)` and `eps=1e-9` -- these exact optimizer
hyperparameters are lifted directly from the original paper (Section 5.3), which specifically
tuned Adam's beta_2 and epsilon for Transformer training.

**Learning rate -- an important simplification to flag:** the original paper does **not** use a
constant learning rate. It uses a specific **warmup-then-decay schedule**:

```
lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

with `warmup_steps = 4000`. This linearly *increases* the learning rate for the first 4,000
training steps, then *decreases* it proportionally to the inverse square root of the step number
afterward. This schedule is well known to matter a great deal for getting the original
Transformer to train stably (see Training Challenges below). **This code does not implement that
schedule at all** -- it simply passes a fixed learning rate (`learning_rate=0.0001`, i.e. 1e-4) to
Adam for the entire training run, with no warmup and no decay. This is a deliberate simplification
appropriate for a small model on a small dataset over only a handful of epochs, but it is a real
and important difference from the original paper's training recipe, and worth being explicit
about if asked.

The paper's loss also uses **label smoothing** (`eps_ls = 0.1`, softening the one-hot target
distribution so the model isn't pushed to be maximally overconfident) -- this code's
`CrossEntropyLoss` does not apply label smoothing, another simplification relative to the original
recipe.

**Batch size:** 16 (`batch_size = 16` in `main()`, explicitly commented "Smaller batch size for
Transformer").

**Gradient handling:** gradients are clipped to a maximum norm of 1.0 before each optimizer step.

**Training loop structure:** `train_transformer` runs for `epochs=6` (as called in `main()`),
looping over the training `DataLoader`, computing teacher-forced loss (`decoder_input = tgt[:,
:-1]`, `decoder_target = tgt[:, 1:]`), backpropagating, clipping, stepping Adam, and printing loss
every 20 batches; at the end of each epoch it evaluates on the validation set (`evaluate_transformer`,
model in `eval()` mode with `torch.no_grad()`) and records both train and validation loss for
plotting.

### 6. Training Challenges

- **Learning-rate warmup sensitivity.** This is the single most famous training quirk of the
  original Transformer: without the warmup phase, Adam's adaptive learning rate estimates can be
  poorly calibrated in the first few hundred steps (because its running estimates of gradient
  variance haven't stabilized yet), and combined with the Transformer's LayerNorm-heavy,
  residual-heavy architecture, this can lead to training divergence or a stuck/degenerate model
  early on. The 4,000-step linear warmup gives Adam's statistics time to stabilize before the
  learning rate reaches its peak. This code sidesteps the issue entirely by using a small,
  constant learning rate (1e-4) instead, which is more forgiving but is not how the original
  model was actually trained.
- **Quadratic cost of self-attention in sequence length.** Every self-attention layer computes a
  full `seq_len x seq_len` score matrix, so both compute and memory cost scale as O(n^2) in
  sequence length. For the short sentences used in this demo (a few dozen tokens) this is a
  non-issue, but it is the central scalability bottleneck of the Transformer architecture in
  general, and it is exactly the problem that motivated later "efficient Transformer" variants
  (see file 017 in this collection) and Transformer-XL's segment-based approach (see the next
  file, 014).
- **Deep post-LN stacks can be harder to train at scale.** Because this code (faithfully) applies
  LayerNorm *after* the residual addition (post-LN, as in the original paper), very deep stacks
  of this exact configuration are known in the broader literature to be somewhat more sensitive
  to initialization and learning rate than the pre-LN variant many later models switched to; this
  is manageable at the paper's 6-layer depth (and this code's 3-layer depth) but becomes more of
  an issue as depth grows into the dozens of layers.
- **Getting the masking logic exactly right.** There are three separate masks in play (source
  padding, target padding, target causal), and they have to be combined correctly for each of the
  three attention operations (encoder self-attention, decoder self-attention, cross-attention). A
  bug here (e.g. forgetting the causal mask, or applying the causal mask to cross-attention where
  it doesn't belong) would silently let the model "cheat" during training and then fail
  mysteriously at generation time -- a classically hard-to-debug class of error in
  Transformer implementations.

### 7. Performance & Evaluation

Historically, the Transformer is evaluated with **BLEU score** on machine translation
(the standard metric comparing generated translations against reference translations via n-gram
overlap). The original paper's base model achieved **28.4 BLEU** on WMT 2014 English-German
(more than 2 BLEU points above the best previously reported result, including ensembles) and
**41.8 BLEU** on WMT 2014 English-French, while training in a fraction of the time/cost of the
best previous single models -- this combination of *better accuracy* and *cheaper training* was
the paper's central result.

This code, since it is not doing real translation, evaluates with **validation cross-entropy
loss** per epoch on the synthetic task (there is no BLEU computation here, since there is no
reference translation to compare against in a meaningful way). `main()` also demonstrates
qualitative evaluation: it runs the trained model's `generate()` method on a few held-out test
examples and prints input/target/generated side by side, and it renders **attention heatmaps**
(`visualize_transformer_attention`, using seaborn) for a couple of encoder self-attention
layers/heads, which is a good way to visually sanity-check whether attention has learned anything
sensible (e.g. words attending to themselves or to nearby/related words) even without a formal
BLEU-style score.

### 8. Impact -- Why It Mattered

It is difficult to overstate this. The Transformer is the architectural foundation underneath
essentially all modern large language models. Its encoder became the basis for **BERT** (2018),
which pre-trains a bidirectional Transformer encoder for language understanding. Its decoder
became the basis for the **GPT series** (2018-present), which pre-trains an autoregressive,
causally-masked Transformer decoder for text generation, scaled up to hundreds of billions of
parameters. **T5** (2019) and other text-to-text models use the full encoder-decoder Transformer,
almost unchanged in its core mechanics, just scaled up and pre-trained differently. Even
architectures explicitly designed to fix the Transformer's weaknesses -- Transformer-XL (file
014, longer context), sparse/efficient attention variants (file 017, lower compute cost) -- are
modifications *of* this architecture, not replacements for it. The self-attention mechanism
introduced here has also been exported wholesale into computer vision (Vision Transformers),
speech, protein structure prediction (AlphaFold), and multimodal models. The paper's core insight
-- that a uniform, parallelizable, "everyone attends to everyone" mechanism could replace both
recurrence and convolution and still improve state-of-the-art results -- turned out to be one of
the most generative ideas in the history of machine learning, largely because it also happened to
scale extremely well with more data and more compute, which set up the entire subsequent era of
ever-larger language models.

### 9. How To Explain This In An Interview

"The Transformer solved a very specific problem with RNN-based sequence models: their sequential,
step-by-step processing prevented parallel training and made it hard to learn long-range
dependencies, since information from far-apart words had to pass through many sequential
transformations to interact. The Transformer's answer was to drop recurrence and convolution
entirely and rely purely on self-attention, where every position computes a query, key, and value
vector, and attends to every other position in a single matrix operation -- scaled dot-product
attention, softmax of QK-transpose over the square root of the head dimension, times V. It does
this with multiple attention heads in parallel so different heads can specialize in different
kinds of relationships, and stacks six of these encoder layers and six decoder layers, each
combining self-attention with a position-wise feed-forward network, wrapped in residual
connections and layer normalization so the whole deep stack trains stably. Since there's no
recurrence, you have to explicitly inject position information, which the original paper does
with a fixed sine/cosine positional encoding. The decoder adds a causal mask so it can't peek at
future tokens, plus a cross-attention step where its queries look at the encoder's keys and
values, which is functionally the same job classic attention mechanisms did for RNN decoders, just
generalized. I implemented this from scratch and trained a scaled-down version -- 256-dimensional
embeddings, 3 layers instead of 6, a 4,000-word vocabulary -- on a synthetic sequence
transformation task built from WikiText-2, since I didn't have a real translation corpus, using
Adam with the paper's specific betas and epsilon, though I used a constant learning rate rather
than the paper's well-known warmup-then-decay schedule, which in the real paper matters a lot for
stable training. The reason this architecture is such a big deal is that it isn't just one good
model -- it's the direct ancestor of BERT, the GPT family, T5, and basically every modern large
language model, because it turned out to both work extremely well and scale extremely well with
more data and compute."
