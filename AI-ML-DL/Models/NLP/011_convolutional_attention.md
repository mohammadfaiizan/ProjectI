# 011: Convolutional Sequence-to-Sequence Learning -- Companion Notes

This file documents `011_convolutional_attention.py`, which implements Facebook AI Research's
**ConvS2S** architecture: a sequence-to-sequence model built entirely out of convolutions
(no recurrence at all), with an attention mechanism connecting the decoder to the encoder.
It was published the same year as the Transformer and represents an alternative answer to the
same question: "how do we get rid of the sequential bottleneck in RNNs?"

---

## ConvS2S -- Convolutional Sequence to Sequence Learning (2017)

### 1. What Problem It Solved

Before 2017, the standard way to do sequence-to-sequence tasks (translation, summarization) was
an RNN or LSTM encoder-decoder, often with attention (see files 007-010 in this series). The
problem with RNNs is that they process a sequence one token at a time: token 2 cannot be
computed until token 1's hidden state exists, token 3 needs token 2's, and so on. This is a
**sequential dependency**. It means:

- You cannot parallelize computation across the time dimension during training. Even with a
  giant GPU, an RNN encoder processing a 50-word sentence must do 50 sequential steps.
- Training is slow, especially on long sequences, because the GPU sits partially idle waiting
  for each step to finish before starting the next.
- Gradients have to flow back through every one of those sequential steps, which contributes to
  vanishing/exploding gradient issues on long sequences (the same issue that motivated LSTM/GRU).

ConvS2S's answer: replace the recurrent connections with **stacked 1D convolutions**. A
convolution over a sequence looks at a small fixed-size window of neighboring tokens (say, 3
tokens at a time) and produces an output for every position **simultaneously**. Because every
position's output only depends on nearby inputs (not on the previous position's output), all
positions in a layer can be computed in parallel. Stacking many convolutional layers on top of
each other lets the effective "receptive field" (how far back/forward a position can see) grow
with depth, similar to how deep CNNs build up larger receptive fields in image processing.

The concrete failure mode being fixed is: **RNN sequential processing makes training slow and
hard to parallelize, especially as sequences and datasets get longer.** ConvS2S keeps some
locality-based structure (like a CNN) but adds attention so the decoder can still look at any
part of the source sentence, not just nearby convolution windows.

### 2. Architecture -- How It Works

**Core intuition:** think of the encoder as a stack of "read a few words at a time" filters,
each layer looking a bit further afield than the last, building up richer and richer
representations of each word in the context of its neighbors. The decoder does the same thing
for the sequence it is generating, but it is only allowed to look at words it has already
generated (causal/autoregressive), and at every layer it also asks "which parts of the source
sentence are relevant to what I'm generating right now?" via attention.

Step-by-step, following the actual classes in the code:

**Gated Linear Unit (`GLU`).** Every convolution in this model outputs twice as many channels
as it needs. The GLU activation splits those channels into two equal halves, `a` and `b`, and
computes:

```
GLU(x) = a * sigmoid(b)
```

`sigmoid(b)` acts as a learned "gate" between 0 and 1 for each channel, and `a` is the actual
content. This is like an LSTM's forget/input gates, but applied inside a convolution stack
instead of a recurrent cell. It gives the network a way to control how much information flows
through each layer, which helps gradients flow through very deep convolutional stacks.

**Encoder (`ConvEncoder`).**
1. Each input token is turned into a token embedding (`embed_dim=256` in this code) and a
   learned position embedding (a separate embedding table indexed by position 0, 1, 2, ...,
   because unlike an RNN, a convolution has no inherent sense of "this is the 3rd token" --
   position must be told to it explicitly, just like the Transformer needs positional encoding).
2. Token embedding + position embedding are added together and projected up to the hidden size
   (`hidden_dim=512`).
3. The sequence is fed through a stack of 1D convolutions (`num_layers=6` in the paper, reduced
   to 4 in this code's demo). Each convolutional layer: convolve (kernel size 3, "same" padding
   so the output length matches the input length) -> GLU activation -> add the residual
   (skip connection from before the convolution) -> LayerNorm -> dropout.
4. The final output is a sequence of hidden vectors, one per source position, that has "seen"
   information from nearby positions at increasing distances as you go deeper in the stack.

**Multi-step attention (`ConvAttention`), used inside the decoder.** This is ConvS2S's version
of attention, and it is applied **at every decoder layer**, not just once at the end (that's
what "multi-step" refers to -- attention happens at multiple steps/layers of decoding, letting
each layer refine its own view of the source sentence). For a given decoder position:
1. Project the decoder's current hidden state into the same space as the encoder's embeddings.
2. Compute a dot product between that projected decoder state and every encoder position's
   projected embedding to get a raw attention score per source position.
3. Mask out padding positions (using the real `input_lengths` so padding never receives
   attention), then softmax over the source positions to get attention weights.
4. Take a weighted sum of the encoder embeddings (not the encoder's convolved hidden states --
   the original token+position embeddings) to produce a context vector.
5. Concatenate that context vector with the decoder's hidden state and project back down to the
   decoder's hidden size. This becomes the "context-aware" representation for that decoder
   position at that layer.

**Decoder (`ConvDecoder`).**
1. Same idea as the encoder: token embedding + position embedding, projected to hidden size.
2. Convolutions here use **causal padding**: the code pads the left side of the sequence by
   `kernel_size - 1` and then truncates the convolution output back to the target length. This
   guarantees a decoder position can only see itself and earlier positions -- it can never
   "cheat" by seeing future tokens it hasn't generated yet, which is the same idea as the
   causal mask in a Transformer decoder, just implemented through convolution padding instead
   of a masked attention matrix.
3. After the causal convolution + GLU + residual, the code applies `ConvAttention` **at every
   time step, for every layer**, adds the resulting context vector back into the hidden state,
   then applies LayerNorm and dropout. This attention-at-every-layer pattern is why the
   mechanism is called "multi-hop" or "multi-step" attention in the ConvS2S paper -- the
   decoder gets multiple opportunities across depth to re-examine the source sentence.
4. After the final layer, a linear projection maps hidden vectors to vocabulary-sized logits.

Note an implementation detail worth knowing for an interview: the code's decoder attention loop
literally iterates over every target position one at a time (`for t in range(target_length)`)
inside `ConvDecoder.forward`, calling `ConvAttention` separately per position. This is a
readable, teaching-friendly way to write it, but it is not the fully-parallel matrix formulation
that the original ConvS2S paper uses at inference-time efficiency; it still trains in parallel
across the batch dimension and target positions are all known already (teacher forcing), so no
sequential dependency is introduced during training -- it is just a slower way to compute the
same result.

### 3. Model Size & Parameters

**Original paper (Gehring et al., 2017):** the ConvS2S models used in the paper had up to 15
convolutional layers in the encoder and 15 in the decoder for their strongest results on WMT
translation tasks, embedding and hidden dimension of 512, kernel width 3 (with some variants
using wider kernels), and trained on 8 GPUs. Their base configuration is broadly comparable in
scale to a 6-layer encoder/decoder RNN or Transformer of the same era -- tens of millions of
parameters.

**This code's configuration:** `embed_dim=256`, `hidden_dim=512`, and in `main()` the encoder
and decoder are each instantiated with `num_encoder_layers=4` and `num_decoder_layers=4`
(explicitly commented "Reduced for demo" in the code, down from the class defaults of 6). Kernel
size is 3, dropout 0.1. Vocabulary size is capped at 3,000 tokens (`build_vocabulary(...,
vocab_size=3000)`). The exact parameter count is printed at runtime via `count_parameters()`
but is on the order of a few million, not tens of millions, because of the small vocabulary,
small hidden size, and shallow stack.

**Why scaled down:** this is a teaching notebook meant to run on a laptop CPU (or a single GPU)
in minutes, on a tiny slice of WikiText-2, not on WMT's tens of millions of sentence pairs across
a GPU cluster for days. Shrinking layers, hidden size, and vocabulary keeps the whole pipeline
(data loading, forward pass, backward pass, generation) fast enough to run and inspect
end-to-end, while still containing every structural piece (GLU, residual convolutions, causal
masking, multi-step attention) that the real architecture uses.

### 4. Dataset & What It Was Trained On

**Original paper:** ConvS2S was evaluated on standard machine translation benchmarks -- WMT'14
English-French (~36M sentence pairs) and WMT'16 English-German, as well as an abstractive
summarization task. These are large, real bilingual parallel corpora.

**This code's demo:** WikiText-2, a monolingual English corpus of Wikipedia articles, loaded via
`load_dataset('wikitext', 'wikitext-2-v1')`. The code lowercases text, splits on periods into
pseudo-sentences, tokenizes with NLTK's `word_tokenize`, and keeps sentences with 6-25 tokens.
It then takes only the first 1,000 training sentences, 200 validation, and 200 test sentences.
Since WikiText-2 is not a translation dataset, the code does **not** do real translation.
Instead `ConvS2SDataset` synthesizes toy tasks from these English sentences:
- `'reverse'` (the task actually run in `main()`, since `tasks[:1]` only executes the first
  entry of `tasks = ['reverse', 'translation']`): the target sequence is just the input sequence
  reversed word-for-word. This is a controllable, easy-to-verify proxy for sequence
  transduction -- if the model can learn to reverse a sequence, it has learned to attend to and
  reorder source positions, which is the core skill translation also requires.
- `'translation'` (defined but not run by default): a fake "translation" made by substituting
  a handful of English words with French words from a small hand-written dictionary (`the` ->
  `le`, `is` -> `est`, etc.) and occasionally splitting the sentence in half and swapping the
  halves.
- `'summarization'`: picks the first word, last word, and longest word of the sentence as a toy
  extractive summary.

**The gap:** the real ConvS2S needed millions of real bilingual sentence pairs with genuine
semantic correspondence between source and target languages. This demo needs a task where
"correct" is well-defined and checkable, but it doesn't need real translation data to prove the
architecture works -- reversing or lightly transforming a few hundred English sentences is
enough to demonstrate that the convolutional encoder-decoder-with-attention plumbing is wired up
correctly and can learn a non-trivial sequence transformation.

### 5. Training Process

**Objective/loss:** standard token-level cross-entropy loss (`nn.CrossEntropyLoss(ignore_index=0)`)
between the decoder's predicted next-token distribution and the actual next token in the target
sequence, with the padding token index (0) excluded from the loss so the model isn't rewarded or
punished for predicting padding.

**Teacher forcing:** the decoder is always given the true previous target tokens as input during
training (`decoder_input = target_ids[:, :-1]`, `decoder_target = target_ids[:, 1:]`) -- this is
what makes fully-parallel training possible even though generation later has to happen one
token at a time. Because every target token is known in advance, all target positions can be
processed by the convolutions in one pass, rather than the decoder having to wait for its own
previous prediction like at inference time.

**Optimizer:** Adam, learning rate `0.001` (the `learning_rate` default parameter passed into
`train_model`), no learning-rate schedule or warmup.

**Batch size:** 32.

**Gradient handling:** gradients are clipped to a max norm of 1.0 (`clip_grad_norm_`) before the
optimizer step, which prevents any single unusually large gradient (common early in training)
from destabilizing the weights.

**Training loop structure:** a standard PyTorch loop over `epochs=8` (as called in `main()`),
where each epoch does a full pass over the training `DataLoader`, computing loss, backpropagating,
clipping, and stepping the optimizer per batch, then evaluating on the validation loader (in
`eval()` mode with `torch.no_grad()`) at the end of the epoch and printing train/validation loss.

### 6. Training Challenges

- **Depth and gradient flow in convolution stacks:** stacking many convolutional layers can
  suffer the same kind of degradation problems seen in deep CNNs for vision. ConvS2S addresses
  this the same way ResNets do -- residual (skip) connections around every convolutional block,
  plus GLU's gating, which lets some of the pre-activation signal pass through more directly and
  keeps gradients from shrinking to nothing as they propagate through many stacked layers.
- **Causal convolution correctness:** it is easy to get the padding wrong and accidentally let
  the decoder "see the future." The code handles this by padding the left side with
  `kernel_size - 1` zeros and then slicing the convolution's output back down to the original
  target length, which is a manual but reliable way to enforce causality without needing an
  explicit attention mask (unlike the Transformer, which needs a triangular mask for the same
  purpose).
- **Kernel size vs. receptive field tradeoff:** a small kernel (3, as used here) needs many
  stacked layers before a position can "see" far-away context; a bigger kernel sees further per
  layer but costs more compute and parameters per layer. The original paper tuned this carefully
  across depth and kernel width; this repo just fixes kernel size at 3 and reduces depth to 4
  layers for speed, which limits how much long-range context the encoder can capture in the demo.
- **Attention computed per-decoder-position in a loop:** as noted in section 2, this repo's
  decoder attention is computed one target position at a time inside a Python loop rather than
  as one batched matrix operation. It is correct, but it is slower than it needs to be -- a real
  challenge in translating "the math is right" into "the implementation is fast," which is a
  recurring theme across all attention-based sequence models.

### 7. Performance & Evaluation

This code tracks **validation cross-entropy loss** per epoch as its primary metric (there is no
BLEU score computed, since there's no real translation task being run) and reports final
validation loss, parameter count, training time, and memory usage per model/task combination.
The main function also runs the trained model in `generate()` on a few held-out test examples
and prints the input, the true target, and the model's greedy-decoded output side by side, so you
can visually sanity-check whether the model learned the toy transformation.

Historically, the original ConvS2S paper reported that it matched or slightly exceeded the
translation quality (BLEU score) of the best RNN-based systems of the time (including Google's
GNMT) on WMT English-French and English-German, while training substantially faster -- the paper
claimed roughly a **9x speedup** in training time on GPUs due to full parallelization across
sequence positions, which was the headline result: comparable accuracy, dramatically faster
training.

### 8. Impact -- Why It Mattered

ConvS2S was one of two major 2017 papers (alongside the Transformer) that proved you did not need
recurrence to get state-of-the-art sequence-to-sequence performance. It demonstrated that:
- Convolutions -- already dominant in computer vision -- could be a competitive backbone for
  sequence modeling, not just images.
- GLU-gated residual convolution stacks were an effective way to build deep, trainable networks
  without recurrence.
- Attention did not have to live only "on top of" an RNN decoder; it could be woven into every
  layer of a non-recurrent architecture (the "multi-step attention" idea).

It ultimately lost the architecture race to the Transformer, which offered an even more flexible
form of "look anywhere" connectivity (global self-attention, no fixed receptive field) and turned
out to scale better. But ConvS2S is historically important as proof that the RNN's sequential
bottleneck was not a law of nature -- it was a choice, and multiple independent research groups
broke from it in the same year via different mechanisms (convolution vs. attention). It also
influenced later convolutional sequence models used in speech and other domains.

### 9. How To Explain This In An Interview

"ConvS2S is a 2017 sequence-to-sequence model from Facebook AI that replaces the RNN
encoder-decoder with stacked 1D convolutions, so it can process an entire sequence in parallel
instead of one token at a time. Each convolutional block uses a Gated Linear Unit -- basically a
learned gate that controls how much signal passes through, similar in spirit to an LSTM's gates
-- plus a residual connection, which keeps gradients healthy across a deep stack. Because a
convolution only sees a small local window, the decoder also uses an attention mechanism at every
single layer, called multi-step attention, so it can still pull in relevant information from
anywhere in the source sentence, not just nearby words. In the decoder, causal padding on the
convolutions makes sure the model never peeks at future tokens, which is the convolutional
equivalent of a causal mask in a Transformer. I trained a scaled-down version -- 4 layers instead
of the paper's 15, 256/512-dimensional embeddings, a 3,000-word vocabulary -- on a subset of
WikiText-2, using a toy sequence-reversal task with Adam, cross-entropy loss, and gradient
clipping, since I didn't have a real bilingual corpus. The interesting historical point is that
ConvS2S and the Transformer both appeared in 2017 as answers to the same problem -- RNNs can't be
parallelized -- but they solved it with different mechanisms, local convolution versus global
attention, and global attention is what ended up winning out and becoming the basis for
everything that followed."
