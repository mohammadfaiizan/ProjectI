# 014: Transformer-XL -- Companion Notes

This file documents `014_transformer_xl.py`, which implements **Transformer-XL** (Dai et al.,
2019): segment-level recurrence plus relative positional encoding, designed to give Transformers
a much longer effective context than the original architecture's fixed-length window. The file's
header comment also frames Transformer-XL as part of a broader wave of "fix the Transformer's
context/efficiency limitations" research; in that spirit, this document also gives brief coverage
to three closely related variants -- the Universal Transformer, the Sparse Transformer, and the
Linear Transformer -- that solve adjacent problems (adaptive depth, long-sequence efficiency, and
linear-time attention, respectively). **Accuracy note:** only Transformer-XL is actually
implemented as code in this file (the `TransformerXL`, `RelativeMultiHeadAttention`, and
`RelativePositionalEncoding` classes); the other three are covered here for interview-readiness
and historical context, not because their code exists in this file.

---

## Transformer-XL (2019)

### 1. What Problem It Solved

The original Transformer (file 013) processes text in **fixed-length segments** -- for example,
train and evaluate on chunks of 512 tokens at a time, with no connection between one chunk and
the next. This creates two concrete problems, both named explicitly in the Transformer-XL paper:

1. **Context fragmentation.** If a sentence or a dependency happens to straddle the boundary
   between two segments, the model has no way to connect the two halves -- each segment is
   processed completely independently, with no information carried over from the previous chunk.
   Even worse, at the very start of every new segment, the model has essentially no context at
   all, no matter how much text came before it in the document; it starts "cold" every 512 tokens.
2. **Fixed maximum context length.** Because self-attention costs grow quadratically with sequence
   length, you can't just make the segment length arbitrarily long -- there's a hard ceiling on
   how far back the model can ever look, set once at training time (e.g. 512 tokens) and true for
   every position, regardless of whether a particular prediction would benefit from seeing much
   further back.

Transformer-XL's fix is to let the model **carry forward the hidden states it already computed
for the previous segment**, cached as a kind of short-term memory, so a new segment doesn't start
from nothing -- it can attend back into the previous segment's cached representations, for free
computationally (those hidden states were already computed once and don't need gradients
recomputed for them). This is where the "XL" comes from -- "extra long" context, achieved with a
**segment-level recurrence mechanism**: not recurrence at the level of individual tokens (like an
RNN), but recurrence at the level of entire segments, where segment N+1 gets to consult the cached
hidden states from segment N.

Introducing this memory mechanism creates a secondary problem, though: the original Transformer's
positional encoding is **absolute** -- position 5 always gets the exact same positional encoding
vector, regardless of what segment it's in. If segment 2's position 5 is now being attended to by
segment 3's position 3 (because of the memory mechanism), the model needs a *consistent* way to
represent "this key is 2 positions before this query" -- but absolute position 5 doesn't carry
that relative information cleanly once tokens from different segments (with different absolute
offsets) get mixed together in a single attention computation. Transformer-XL's second
innovation, **relative positional encoding**, exists specifically to solve this: encode
attention scores in terms of the *relative distance* between the query and the key, not their
absolute positions, so the same relative-distance encoding remains meaningful no matter which
segment a token originally came from.

### 2. Architecture -- How It Works

**Core intuition first.** Imagine reading a long book one page at a time. The original Transformer
is like reading each page in total isolation, throwing away your memory of the previous page
before starting the next one. Transformer-XL is like keeping a compressed set of notes from the
last page or two on your desk while you read the current page, so you can still refer back to
"what was mentioned two pages ago" even though you're only actively re-reading the current page.
And instead of your notes saying "this idea was on line 5 of page 1" (an absolute reference that
becomes confusing once you're several pages later), they say "this idea was mentioned 40 lines
ago" (a relative reference that stays meaningful no matter how many pages you've turned).

Concretely, following the classes in the code:

**Relative positional encoding (`RelativePositionalEncoding`).** Instead of a lookup table indexed
by absolute position, this module builds a lookup table indexed by **relative distance between two
positions**. For every pair of positions `(i, j)` in a sequence of a given length, it computes
`i - j` (how far apart they are, with sign indicating direction), clamps that value to a maximum
range (`max_relative_position=512`, so very large distances all map to the same "far away" bucket),
shifts it to be non-negative (adding `max_relative_position` so it can index into an embedding
table), and looks up a learned embedding vector for that specific relative distance. The result is
a `(length, length, d_model)` tensor: one full-`d_model` position vector for every `(query
position, key position)` pair. This is a slightly different mechanism than the exact formulation
in Dai et al.'s original paper (which uses a *fixed*, sinusoidal relative-position basis combined
with a "relative shift" trick to compute all pairwise relative scores efficiently in a single
matrix multiply); this code instead uses a **learned embedding table indexed by clipped relative
distance** (closer in spirit to Shaw et al.'s 2018 relative position representations). It
represents the same underlying idea -- attention scores that depend on relative, not absolute,
position -- through a simpler (if somewhat less compute-efficient) mechanism.

**Relative multi-head attention (`RelativeMultiHeadAttention`).** This is the core innovation,
combining content and position information, and combining current-segment tokens with cached
memory from the previous segment:
1. If a `memory` tensor is passed in (the cached hidden states from the previous segment), it is
   concatenated onto the front of the current segment's keys and values (`key`/`value` are
   extended, but **not** the query -- the query only ever comes from the current segment's
   tokens, since those are the positions we're actually computing new representations for).
2. Query, key, and value are linearly projected and split into heads, exactly as in standard
   multi-head attention (file 013), except these projections have `bias=False`, matching the
   paper's formulation.
3. The relative position embeddings are projected through their own linear layer, `w_r`.
4. Attention scores are computed as the sum of a **content-based score** (query, offset by a
   learned bias vector `u`, dotted with the content keys) and a **position-based score** (query,
   offset by a second learned bias vector `v`, dotted with the relative-position keys), then
   scaled by `sqrt(d_k)`. This mirrors the paper's idea of decomposing attention into
   "how much does this key's content matter" versus "how much does this key's relative position
   matter," each with its own learned global bias (`u` for content, `v` for position) that is
   shared across all query positions -- the intuition being that some attention patterns are
   driven mostly by *what* a token is, and others mostly by *how far away* it is, and the model
   should be able to learn both kinds of preference.
5. Softmax over the combined scores, apply to the (memory-extended) values, concatenate heads,
   final output projection -- same as standard multi-head attention otherwise.

**Transformer-XL layer (`TransformerXLLayer`).** Each layer is structurally similar to a
Transformer encoder layer (file 013): self-attention (here, relative + memory-augmented) with
residual connection and LayerNorm, followed by a feed-forward network with its own residual
connection and LayerNorm. There is no separate encoder/decoder split here -- Transformer-XL, like
GPT, is a **decoder-only, causally-masked** architecture used for language modeling (predicting
the next token), not a full encoder-decoder translation model.

**The full model and its memory mechanism (`TransformerXL`).**
1. `init_memory` creates an all-zero memory tensor of shape `(batch, memory_length, d_model)` for
   every layer, used as the starting memory before any segment has been processed.
2. In `forward`, the token embedding is scaled by `sqrt(d_model)` (same convention as file 013),
   relative position embeddings are computed for the combined length of the current segment plus
   whatever memory is being carried in, and a causal mask is built over `(current segment length,
   total length including memory)` so a current-segment position can attend to all of memory
   (which is, by construction, entirely in the past) plus itself and earlier current-segment
   positions, but never anything later.
3. Each layer runs `RelativeMultiHeadAttention` with that layer's own memory slice, then the
   feed-forward network, exactly as described above.
4. `update_memory` takes the hidden states just computed for the current segment, concatenates
   them onto the *old* memory, and truncates to keep only the most recent `memory_length` tokens
   -- this becomes the memory that will be handed to the *next* segment. Critically, this is
   wrapped in `.detach()`, meaning gradients are **not** backpropagated through the memory into
   the segment that originally produced it. This is a deliberate and important design choice
   (**truncated backpropagation through time**, borrowed from RNN training practice): it lets the
   model use the *information* in old hidden states as extra context, without paying the memory
   and compute cost of backpropagating gradients arbitrarily far back through segment after
   segment.
5. Output projection maps the final hidden states to vocabulary logits, exactly as in a standard
   language model head.

**Generation (`generate`)** works autoregressively, feeding the most recent `segment_length`
tokens through the model at each step, sampling the next token from a temperature-scaled softmax
(`torch.multinomial`, not greedy argmax -- this is stochastic sampling, giving varied output), and
carrying the updated memory forward from each step to the next, so generation itself also benefits
from the extended-context mechanism.

**A subtlety worth knowing for accuracy:** in `main()`, the training `DataLoader` is created with
`shuffle=True`, while memory is still carried across consecutive *batches* within an epoch
(`memories = [mem.detach() for mem in new_memories]` after each batch). Segment-level recurrence
is only meaningful if consecutive batches actually contain consecutive text -- shuffling the
training segments means the "memory" being carried into a batch is, for the most part, cached
hidden states from an *unrelated* segment elsewhere in the corpus, not the text that actually
precedes it. The validation loader, by contrast, uses `shuffle=False`, so validation's use of
memory is faithful to the real mechanism (sequential segments, memory genuinely reflects "what
came right before"). This is a small but real gap between what the training loop does and what
the technique is designed to do, and is worth flagging honestly rather than glossing over.

### 3. Model Size & Parameters

**Original paper:** Transformer-XL's WikiText-103 (word-level) base configuration used 16 layers,
`d_model=410`, 10 attention heads, `d_ff=2100`, with an adaptive softmax/embedding scheme for the
large vocabulary, totaling roughly **151 million** parameters; its larger configuration used 18
layers, `d_model=1024`, 16 heads, `d_ff=4096`, roughly **257 million** parameters. For the
character-level enwik8 benchmark, the paper used a deep 24-layer, `d_model=1024` configuration.
Memory (cache) lengths in the paper's experiments ranged from roughly 150 tokens during training up
to several hundred or more at evaluation time, since a longer cache at evaluation is "free" (no
extra gradient computation needed).

**This code's configuration:** `d_model=256`, `num_heads=8`, `num_layers=4` (versus the paper's
16-18), `d_ff=512`, `segment_length=32`, `memory_length=32`, dropout 0.1, with a 4,000-word
vocabulary. The exact parameter count is printed at runtime via `count_parameters(transformer_xl)`
and comes out to a few million parameters -- dramatically smaller than either paper configuration,
mostly because of the much smaller `d_model`, `d_ff`, layer count, and vocabulary.

**Why scaled down:** the paper's configurations were built for large word-level or
character-level corpora processed over many GPU-days; this code needs to run a full
train/validate/generate cycle in a few minutes on a small subset of WikiText-2, so every dimension
that drives compute and memory cost (depth, width, segment/memory length, vocabulary) is reduced,
while the two defining mechanisms -- segment-level memory recurrence and relative positional
attention -- are both fully implemented and exercised.

### 4. Dataset & What It Was Trained On

**Original paper:** Transformer-XL's flagship results are on **WikiText-103** (a much larger
sibling of WikiText-2, roughly 103 million tokens of Wikipedia text, word-level) and **enwik8**
(a byte/character-level compression benchmark derived from Wikipedia, commonly measured in bits
per character). These are large corpora specifically chosen because they let researchers measure
whether a model can actually exploit very long contexts -- WikiText-103's long articles and
enwik8's long raw byte streams both reward models that remember information from far earlier in
the document.

**This code's demo:** WikiText-2 (`load_wikitext2_dataset`) -- notably the *smaller* sibling
dataset, not WikiText-103. Sentences of length 8-40 tokens are kept (a wider length window than
files 011/012/013, since Transformer-XL is specifically meant to benefit from longer sequences),
using the first 800 training / 160 validation / 160 test sentences, with a 4,000-word vocabulary.
`TransformerXLDataset` concatenates all the kept sentences into one long token stream (inserting
`<EOS>` at each original sentence boundary) and then slices that stream into fixed-length,
50%-overlapping segments (`segment_length=32`, stride `segment_length // 2 = 16`) for
next-token-prediction language modeling.

**The gap:** the entire point of Transformer-XL is to shine on *long* documents where far-back
context genuinely helps; WikiText-2, used here in a small 800-sentence subset with 32-token
segments, is far too short and far too small to meaningfully showcase the benefit of segment
recurrence the way WikiText-103 or enwik8 do in the original paper. This demo is sized to prove
the mechanism is implemented correctly (memory is created, carried, truncated, and attended to
with relative positions, all without errors) rather than to reproduce the paper's actual
long-context performance gains.

### 5. Training Process

**Objective/loss:** standard next-token-prediction cross-entropy loss
(`nn.CrossEntropyLoss(ignore_index=0)`), comparing the model's predicted distribution at every
position in the segment against the true next token at that position (the dataset is built so
`target[i] = input[i+1]` throughout the segment).

**Optimizer:** Adam, with `betas=(0.9, 0.999)` (the PyTorch defaults, rather than the
Transformer-specific betas used in file 013), learning rate `0.00025` (2.5e-4).

**Gradient handling:** gradients clipped to a maximum norm of **0.25** -- notably tighter than the
1.0 clipping norm used in files 011/012/013. This matches the kind of tight gradient clipping
Transformer-XL's own training recipe uses, which makes sense given the extra instability risk
introduced by very deep stacks combined with cross-segment memory propagation.

**Batch size:** 8 -- the smallest batch size of any file in this collection, reflecting the
higher per-example memory footprint of caching hidden states per layer across segments.

**Training loop structure -- the recurrence in action:** `train_transformer_xl` initializes
`memories = None` once at the **start of each epoch** (not once per batch), then for every batch:
runs the forward pass with the *current* memory, computes loss, backpropagates, clips, steps the
optimizer, and then **updates and detaches** the memory (`memories = [mem.detach() for mem in
new_memories]`) so it is ready to be handed to the *next* batch in the same epoch. This detach step
is what implements truncated backpropagation through time -- each batch's backward pass only
computes gradients for that batch's own segment, treating the incoming memory as a fixed, constant
input (since it has no `requires_grad` history attached after `.detach()`). Runs for `epochs=6`.

**Evaluation metric -- perplexity, not raw loss:** `evaluate_transformer_xl` accumulates total
loss weighted by token count across the whole validation set, then computes
`perplexity = exp(average loss)`. This is worth noting as a difference from files 011-013, which
only ever report raw validation cross-entropy loss -- this file explicitly converts to perplexity,
which is the standard way language-modeling quality is reported in the literature (a lower,
more interpretable number roughly meaning "the model was, on average, as confused as if it had to
choose uniformly among this many words").

### 6. Training Challenges

- **The problem this whole architecture exists to fix: fixed context and context fragmentation.**
  Any Transformer trained on fixed-length chunks (like the base Transformer in file 013) cannot
  use information from outside its window, and suffers a "cold start" at the beginning of every
  new chunk. The segment-recurrence mechanism directly addresses this, at the cost of added
  bookkeeping (memory has to be initialized, threaded through every forward call, updated, and
  detached correctly every single batch -- any bug in this bookkeeping silently degrades or
  breaks the whole benefit of the technique).
- **Truncated backpropagation through time is a deliberate tradeoff, not a free lunch.** By
  detaching memory before it's handed to the next segment, the model gets the *information*
  benefit of a longer effective context without the *compute/memory* cost of backpropagating
  gradients across unboundedly many segments -- but it also means the model can never directly
  learn "how a very early segment should have been represented differently to help a much later
  prediction," only that the frozen, already-computed representation of the early segment is
  useful as extra input. This is the same fundamental tradeoff that motivated truncated BPTT in
  RNN training, applied here at the segment level instead of the per-token level.
- **Tighter gradient clipping (0.25 vs. 1.0) suggests real training instability risk.** Combining
  deep self-attention stacks with a persistent, cross-batch recurrent state is more prone to
  runaway gradients than a stateless, fully-parallel Transformer, which is presumably why the
  paper (and this code, following it) uses noticeably more aggressive clipping.
- **Consistency between training and inference use of memory.** As noted in section 2, this code's
  own training loop shuffles segments while still carrying memory across batches, which weakens
  (though does not entirely eliminate, since nearby-in-training-order segments can still land in
  the same epoch) the intended "genuinely sequential context" benefit during training, even though
  validation and generation do use memory in the intended, sequential way. Getting this right in a
  real system requires deliberately using a non-shuffled, order-preserving sampler for
  segment-recurrent training, which is a real engineering detail that's easy to get wrong.
- **Relative position representation choices matter.** This code's simplified, learned-embedding
  relative encoding is easier to implement than the paper's sinusoidal-plus-relative-shift
  formulation, but the original paper's approach was specifically designed to be efficiently
  computable via one matrix operation and to generalize to relative distances not seen during
  training (since it's based on a continuous sinusoidal function, not a finite lookup table) --
  the tradeoff between "simple to implement" and "the exact efficient/generalizing formulation
  from the paper" is a recurring theme in reproducing published architectures faithfully.

### 7. Performance & Evaluation

Historically, Transformer-XL is evaluated with **perplexity** on word-level benchmarks (WikiText-103)
and **bits-per-character (bpc)** on character-level benchmarks (enwik8, text8). The paper reported
new state-of-the-art results at the time: roughly **18.3 perplexity on WikiText-103** (down from
the previous best around 20.5), and roughly **0.99 bpc on enwik8**. The paper's headline efficiency
claim was that Transformer-XL achieved an effective context length **80% longer than a vanilla
Transformer** and was up to **1,800 times faster** than comparable RNN-based language models at
evaluation time on long sequences, because of how memory reuse avoids redundant recomputation.

This code evaluates with **validation perplexity per epoch** (computed as described in section 5),
tracked alongside training loss in `training_histories`, and plotted at the end of `main()`
alongside bar charts summarizing the architecture's key innovations and a rough illustrative
comparison of "effective context length" between a vanilla Transformer and Transformer-XL. It also
demonstrates memory-augmented text generation from a short seed phrase, and a memory-usage
visualization (`visualize_memory_usage`) that runs the model over consecutive segments of a longer
test passage and renders the resulting attention heatmaps side by side.

### 8. Impact -- Why It Mattered

Transformer-XL was the **first major architectural improvement to the original Transformer's
context-handling limitation**, and it introduced two ideas that persisted well beyond this one
paper. Segment-level recurrence -- caching and reusing past hidden states rather than recomputing
or discarding them -- became a recurring pattern in later long-context language model designs.
Relative positional encoding turned out to be broadly useful beyond just the memory-recurrence use
case: it gives a model a more natural, translation-invariant sense of position ("this token is 3
words before that one," rather than "this token is at absolute position 47"), which generalizes
better to sequence lengths not seen during training -- and variants of relative position encoding
(including later refinements like rotary position embeddings) became standard components in many
subsequent large language models, including influencing design decisions in the GPT family and
other long-context models that followed. Transformer-XL is a clear example of how the field's
progress after 2017 was largely a story of identifying specific weaknesses of the original
Transformer (context length here; compute cost in the efficient-attention variants discussed
below) and designing targeted fixes, rather than replacing the core architecture outright.

### 9. How To Explain This In An Interview

"Transformer-XL fixes a specific limitation of the original Transformer: it can only see a fixed
window of tokens, so it suffers what the paper calls context fragmentation -- any dependency that
crosses a chunk boundary is invisible to the model, and every new chunk starts with zero context
about what came before. Transformer-XL's fix is segment-level recurrence: after processing a
segment, it caches the resulting hidden states as a memory, and the next segment's self-attention
is allowed to attend back into that cached memory, in addition to attending within itself. That
memory is detached before being passed forward, so it's used as extra context, not backpropagated
through indefinitely -- that's truncated backpropagation through time, borrowed from how RNNs are
trained. Introducing that memory breaks the original Transformer's absolute positional encoding,
because a fixed absolute position doesn't mean the same thing once you're mixing tokens from the
current segment with cached tokens from the previous one, so Transformer-XL also introduces
relative positional encoding, where attention scores are computed based on the distance between
the query and key positions rather than their absolute positions, using separate learned bias
terms for content-based versus position-based attention. In my implementation I trained a
scaled-down, 4-layer, 256-dimensional version on WikiText-2 with 32-token segments and a 32-token
memory, using Adam, tight gradient clipping at 0.25 as the paper does, and I evaluated with
perplexity rather than raw loss, which is the standard metric for language models. The reason
this matters historically is that it was the first big fix to the original Transformer's context
limitation, and both its core ideas -- reusing cached representations across segments, and
representing position relatively rather than absolutely -- became recurring building blocks in the
long-context language models that came after it."

---

## Universal Transformer (2018)

### 1. What Problem It Solved

The original Transformer applies a *fixed* number of layers to every input, and every layer has
its own separate set of weights -- position 5 in layer 1 is processed completely differently from
position 5 in layer 6, with no shared parameters between them. This is unlike an RNN, which
applies the *same* transformation repeatedly over time, giving it a naturally recurrent inductive
bias that turns out to help on certain algorithmic and compositional tasks (e.g. tasks that
require applying the same simple operation a variable number of times, like copying, counting, or
simple arithmetic-style reasoning). The Transformer's fixed depth and non-shared weights meant it
sometimes underperformed RNNs on exactly these kinds of tasks, despite outperforming them on
translation. The Universal Transformer (Dehghani et al., 2018) asked: can we keep the
Transformer's parallelism and attention while getting back some of the RNN's helpful
"apply-the-same-operation-repeatedly, for a data-dependent number of steps" behavior?

### 2. Architecture -- How It Works

The core idea is **recurrence in depth with shared weights, plus adaptive computation time.**
Instead of stacking `N` distinct Transformer layers, the Universal Transformer applies the
*same* single layer (same self-attention weights, same feed-forward weights) repeatedly, feeding
each position's output back in as the input to the next application of that same layer -- this is
recurrence, but across depth/time-steps of processing rather than across sequence positions.
Crucially, it also uses **Adaptive Computation Time (ACT)**, a mechanism (from Graves, 2016)
that lets *each position independently decide how many processing steps it needs* -- easy
positions can "halt" early and stop updating, while harder positions keep being refined for more
steps, up to some maximum. This gives the model a form of dynamic, per-token depth, rather than
every token being forced through the exact same fixed number of transformations.

### 3. Model Size & Parameters

Because the same layer's weights are reused at every step instead of having separate weights per
layer, a Universal Transformer with a given per-step width has noticeably *fewer* parameters than
a standard Transformer of comparable maximum depth, since it doesn't need `N` independent copies
of the layer's weight matrices. The original paper's experiments used comparable per-layer widths
to the base Transformer (`d_model` in the low hundreds to 512-ish range) with a maximum number of
steps typically in the same rough range as the base Transformer's depth (around 6-16, depending on
the task), combined with the ACT halting mechanism so most positions used fewer steps than the
maximum in practice.

### 4. Dataset & What It Was Trained On

The original paper evaluated the Universal Transformer on a mix of tasks specifically chosen to
probe algorithmic/compositional generalization -- the **bAbI question-answering tasks** (the same
suite discussed in this collection's Memory Networks file), subject-verb agreement prediction,
learning to execute simple programs, and standard **WMT machine translation** benchmarks (English-German)
to confirm the architecture didn't sacrifice translation quality while gaining these other
abilities.

### 5. Training Process

Training follows the same general recipe as the standard Transformer -- cross-entropy loss,
Adam optimizer with a warmup-based learning rate schedule -- with the addition of an extra loss
term for the adaptive halting mechanism: ACT is typically trained with a small penalty on the
number of steps taken (encouraging the model to halt as early as it reasonably can), added to the
main task loss, so the model learns to trade off "more steps for more accuracy" against "fewer
steps for efficiency."

### 6. Training Challenges

Sharing the same weights across every step, and unrolling that shared layer a variable, adaptive
number of times per position, makes the compute graph deeper and more RNN-like in its training
dynamics than a standard fixed-depth Transformer, reintroducing some of the sequential/recurrent
training considerations (like managing gradient flow through many repeated applications of the
same weights) that the original Transformer was specifically designed to avoid. Tuning the ACT
halting penalty is also a delicate balance -- too weak a penalty and positions rarely halt early
(losing the efficiency benefit), too strong a penalty and the model halts too early to do useful
computation on genuinely hard positions.

### 7. Performance & Evaluation

The Universal Transformer was evaluated with **task accuracy on bAbI** and other algorithmic
tasks (where it notably outperformed both the standard Transformer and LSTM baselines on several
of the harder tasks requiring compositional/iterative reasoning) and with **BLEU score** on WMT
translation (where it matched or slightly exceeded the standard Transformer's performance while
using fewer parameters).

### 8. Impact -- Why It Mattered

The Universal Transformer demonstrated that the Transformer's fixed-depth design was a genuine
tradeoff, not a free improvement over RNNs in every respect -- some tasks really do benefit from a
recurrent, "same operation applied repeatedly until done" inductive bias. It kept the broader
field thinking about depth-adaptive and weight-shared variants of attention-based architectures,
and its adaptive-computation ideas resurface in later research on making transformer inference
more efficient by not spending full computation on every token uniformly.

### 9. How To Explain This In An Interview

"The Universal Transformer is a variant that reintroduces a form of recurrence into the
Transformer, but across depth rather than across sequence position -- it applies the same shared
self-attention-plus-feedforward layer repeatedly, instead of stacking separate layers with
separate weights, and it uses adaptive computation time so each token can dynamically decide how
many of these repeated steps it actually needs before halting. This addressed a real weakness of
the standard Transformer, which processes every token through exactly the same fixed number of
layers regardless of how simple or complex that token's role is, and it noticeably helped on
algorithmic and compositional-reasoning tasks like bAbI, where the RNN's 'repeat until done'
inductive bias tends to help, while still matching standard Transformer performance on
translation. It's a good example of the field exploring the design space between 'pure
recurrence' and 'pure parallel attention' rather than treating them as strictly opposed."

---

## Sparse Transformer (2019)

### 1. What Problem It Solved

Standard self-attention computes a full pairwise attention score between every pair of positions
in a sequence, which costs O(n^2) in both compute and memory as sequence length `n` grows. For
short sentences this is trivial, but for very long sequences -- high-resolution images flattened
into pixel sequences, long-form text, raw audio, or music -- this quadratic cost quickly becomes
prohibitive: doubling the sequence length quadruples the attention cost. This was a hard ceiling
on how long a sequence a standard Transformer could realistically be applied to. The Sparse
Transformer (Child et al., 2019, OpenAI) tackled this directly: can we get most of the benefit of
"attend to everything" while only actually computing a small, structured *subset* of all possible
pairwise attention scores?

### 2. Architecture -- How It Works

Instead of every position attending to every other position, the Sparse Transformer restricts
each position to attend to a carefully chosen, structured *sparse* subset of positions, using a
factorized combination of attention patterns -- most notably a **strided pattern** (attend to
every `k`-th previous position, useful for data with a periodic or grid-like structure, like image
rows/columns) and a **local/fixed pattern** (attend to a small window of nearby positions, plus a
handful of fixed "summary" positions). By combining two or more sparse patterns across different
attention heads or layers, information can still propagate across the whole sequence within a few
layers (similar in spirit to how dilated convolutions build up a large receptive field with few
layers), while the actual per-layer attention computation only costs roughly O(n * sqrt(n)) instead
of O(n^2).

### 3. Model Size & Parameters

The original paper applied Sparse Transformers at depths and widths broadly comparable to
contemporary Transformer language/image models of the time (dozens of layers, `d_model` in the
many hundreds), but the headline achievement was in **sequence length**, not parameter count: the
paper demonstrated modeling sequences of **tens of thousands of positions** (for example,
generating images at the pixel level, and modeling raw audio waveforms), lengths that would have
been computationally infeasible with full O(n^2) attention at similar model sizes.

### 4. Dataset & What It Was Trained On

The Sparse Transformer was evaluated across several very-long-sequence generative modeling
benchmarks: **CIFAR-10** and higher-resolution image datasets modeled autoregressively at the
pixel level, **enwik8** (character-level text, the same benchmark used in Transformer-XL's
evaluation), and raw, unprocessed **audio waveform (classical music) generation**, chosen
specifically because each of these domains naturally produces very long sequences that stress-test
the efficiency of the attention mechanism.

### 5. Training Process

Training follows the standard autoregressive/language-modeling recipe -- next-token (or
next-pixel, or next-audio-sample) cross-entropy loss, Adam-style optimization -- with the sparse
attention pattern determining which query-key pairs actually participate in each layer's attention
computation, implemented with custom, memory-efficient GPU kernels so that the theoretical
O(n*sqrt(n)) savings actually translate into real memory and speed savings in practice, not just
in principle.

### 6. Training Challenges

Designing a sparse attention pattern is a delicate balancing act: too sparse, and information
can't propagate between distant positions within a reasonable number of layers, hurting model
quality; not sparse enough, and you lose the computational benefit that was the whole point.
Efficiently implementing sparse attention patterns also required custom, carefully engineered GPU
kernels (rather than simply masking a full attention matrix, which would still cost O(n^2) memory
even if many of the resulting values are thrown away) -- getting the real-world speed and memory
benefits required real systems engineering, not just an architectural idea on paper.

### 7. Performance & Evaluation

The Sparse Transformer was evaluated with the standard metrics for each domain: **bits-per-byte**
on enwik8 (achieving results competitive with much larger dense models), **negative
log-likelihood/bits-per-dimension** on image modeling benchmarks (achieving state-of-the-art
results on CIFAR-10 density modeling at the time), and qualitative sample quality for audio
generation, alongside direct measurements of memory usage and achievable sequence length compared
to dense attention.

### 8. Impact -- Why It Mattered

The Sparse Transformer was an early, influential proof that the Transformer's quadratic
attention cost was not an unavoidable law -- structured sparsity could preserve most of the
architecture's benefits while making drastically longer sequences computationally tractable. It
directly influenced a large family of later "efficient attention" research (various sparse,
low-rank, and windowed attention mechanisms, several of which are covered in file 017 of this
collection) that all pursue the same underlying goal: keep Transformer-quality modeling while
breaking the O(n^2) scaling wall, which became increasingly important as the field moved toward
wanting models that could handle ever-longer documents, images, and audio.

### 9. How To Explain This In An Interview

"The Sparse Transformer addresses the Transformer's quadratic attention cost -- full self-attention
computes a score between every pair of positions, so cost grows with the square of sequence
length, which becomes a real bottleneck for very long sequences like high-resolution images or raw
audio. Instead of attending everywhere, it restricts each position to a structured sparse subset of
other positions -- for example a strided pattern plus a local window -- so that information can
still propagate across the whole sequence over a few layers, but any single layer's attention cost
drops from roughly n-squared to roughly n times the square root of n. It required custom GPU
kernels to realize the actual memory and speed savings, not just a theoretical complexity
improvement. It mattered because it was one of the first strong demonstrations that you could break
the Transformer's quadratic scaling wall without giving up its quality, which opened the door to a
whole family of later efficient-attention variants aimed at handling much longer contexts."

---

## Linear Transformer (2020)

### 1. What Problem It Solved

Even after the Sparse Transformer showed that structured sparsity could reduce attention's
big-O cost, another line of work asked a more radical question: can attention be computed in
genuinely **linear** time, O(n), with no sparsity pattern needed at all -- and, as a bonus, can
this be done in a way that also makes *autoregressive inference* (generating one token at a time)
as fast as an RNN's constant-time-per-step recurrence, instead of the standard Transformer's
inference cost, which grows with how much has been generated so far (because it has to
re-attend over the whole growing sequence at every step)? The Linear Transformer
(Katharopoulos et al., 2020, "Transformers are RNNs") targeted exactly this: both faster training
on long sequences and faster, constant-cost-per-step generation.

### 2. Architecture -- How It Works

The trick is a change to the attention formula itself. Standard attention computes
`softmax(QK^T)V`, and the softmax is what forces you to compute the full `n x n` score matrix
before you can do anything with it. The Linear Transformer replaces the softmax similarity with a
**kernel feature map**, `sim(q, k) = phi(q)^T phi(k)`, for some feature function `phi` (a common
practical choice is a positive, ELU-based feature map). Because this similarity is now expressed
as a dot product of *transformed* query and key vectors rather than an exponential/softmax
function, the associativity of matrix multiplication lets you rearrange the computation:
instead of computing `(phi(Q) phi(K)^T) V` (which still costs O(n^2)), you can compute
`phi(Q) (phi(K)^T V)` -- summing up `phi(K)^T V` once, incrementally, as you scan through the
sequence, and then just multiplying each query against that running sum. This reduces attention's
cost from O(n^2) to O(n), and -- crucially for generation -- it means the "running sum" can be
updated incrementally one token at a time, exactly like an RNN's hidden state update, which is
where the paper's title, "Transformers are RNNs," comes from: this reformulation reveals that
linear attention is mathematically equivalent to a particular kind of RNN with a specific
(linear, kernel-based) update rule.

### 3. Model Size & Parameters

The Linear Transformer is architecturally a drop-in replacement for the standard multi-head
attention block -- same `d_model`, number of heads, and layer count as a comparable standard
Transformer of the same generation. The original paper's experiments used model sizes broadly
comparable to standard Transformer-XL-scale configurations of the time (similar depth and width),
since the point of the paper was a fair head-to-head efficiency and quality comparison at matched
model capacity, not a change in how large the model is.

### 4. Dataset & What It Was Trained On

The paper evaluated the Linear Transformer on standard autoregressive language modeling
benchmarks of the era, including **WikiText-103**, alongside image generation tasks, comparing
directly against standard (softmax) Transformer and Transformer-XL baselines at matched
parameter counts, specifically to isolate the effect of the attention mechanism itself (rather
than confound it with a change in training data or model scale).

### 5. Training Process

Training uses the same next-token cross-entropy objective and Adam-style optimization as a
standard Transformer language model; the only fundamental change is swapping the attention
sub-layer's computation from softmax attention to the linear, kernel-feature-map formulation
described above, which is a drop-in replacement inside an otherwise standard Transformer training
loop.

### 6. Training Challenges

Removing the softmax changes the *inductive bias* of attention -- softmax attention naturally
produces a sharply peaked probability distribution that can strongly emphasize a few highly
relevant positions, while kernel-based linear attention tends to produce smoother, less peaked
weightings, which can occasionally cost the model some ability to sharply focus on one especially
important token. Choosing a good feature map `phi` also matters a lot for both stability and
quality (the paper had to specifically pick a feature map that stays numerically well-behaved,
e.g. always non-negative, so the resulting "attention weights" behave sensibly), and getting the
efficient recurrent-style implementation exactly right (correctly maintaining the running
cumulative sums per head, and handling causal masking within that recurrent formulation) requires
care that a naive implementation can easily get wrong.

### 7. Performance & Evaluation

The Linear Transformer was evaluated with **perplexity** on language modeling benchmarks (matching
or approaching standard softmax-attention Transformer quality at similar model sizes), and,
distinctively, with **direct wall-clock inference speed measurements** for autoregressive
generation, where it demonstrated dramatically faster generation -- because generating each new
token only requires a constant-time update to the running cumulative sum, rather than re-attending
over the entire generated-so-far sequence -- showing large speedups over standard Transformers
specifically as generated sequence length grows.

### 8. Impact -- Why It Mattered

The Linear Transformer helped establish that softmax was not a mandatory ingredient of
"attention" -- the core query/key/value mechanism could be reformulated with a linear kernel and
still work well, while unlocking both linear-time training and RNN-like constant-time-per-step
inference. Its "Transformers are RNNs" framing was influential in clarifying the deep mathematical
relationship between attention-based and recurrent sequence models, and it helped seed a broader
family of later linear-attention and state-space-model research aimed at making very-long-context
and fast-inference language models practical, an area of continued active research well beyond
2020.

### 9. How To Explain This In An Interview

"The Linear Transformer tackles the same quadratic attention cost problem as the Sparse
Transformer, but with a completely different trick: instead of restricting which positions can
attend to which, it replaces the softmax similarity function with a kernel feature map, so the
attention computation becomes a dot product of transformed queries and keys rather than an
exponential. That change lets you use the associativity of matrix multiplication to compute
attention as a running cumulative sum instead of a full pairwise score matrix, which drops the
cost from quadratic to linear in sequence length. The really elegant part is that this
reformulation is mathematically equivalent to a specific kind of RNN, which is why the paper is
titled 'Transformers are RNNs' -- it means you can generate text one token at a time with
constant cost per step, just like an RNN, instead of the standard Transformer's cost per step
growing as the generated sequence gets longer. It mattered because it showed softmax wasn't a
required ingredient of attention, and it's part of the same broader research thread as the Sparse
Transformer and Transformer-XL -- all attacking different aspects of the same core problem, which
is making attention scale to longer sequences and faster inference without giving up the quality
that made the original Transformer so effective."
