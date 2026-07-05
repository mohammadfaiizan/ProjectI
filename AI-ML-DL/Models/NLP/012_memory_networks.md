# 012: Memory Networks -- Companion Notes

This file documents `012_memory_networks.py`, which implements two distinct memory-augmented
architectures: **End-to-End Memory Networks (MemN2N)**, the differentiable successor to Weston
et al.'s original Memory Networks, and the **Dynamic Memory Network (DMN)**, a more elaborate
episodic-memory architecture built on GRUs. Both models are trained in this file on the same toy
"which sentence answers this question" retrieval task built out of WikiText-2 sentences. The
code also defines a small generic `MemoryModule` class that demonstrates the basic
read/attend/write memory operation described in the original Memory Networks paper, although it
is not wired into either of the two trained models -- it exists as a standalone illustration of
the core idea.

---

## Memory Networks and End-to-End Memory Networks (2014 / 2015)

### 1. What Problem It Solved

Standard RNNs (and LSTMs/GRUs) have to compress everything they need to remember into a single
fixed-size hidden state vector, which gets overwritten at every time step. If a question depends
on a fact mentioned 40 tokens ago, that fact has to have survived 40 rounds of being blended into
(and partially overwritten by) the hidden state. This works poorly for tasks that require
**multi-step reasoning over multiple discrete facts** -- for example: "Mary went to the kitchen.
John went to the garden. Where is the milk?" requires combining several separate facts, not just
remembering "the most recent thing that happened."

Weston et al.'s 2014 Memory Networks paper proposed making memory **explicit and external**:
store each fact (e.g., each sentence) as its own separate memory slot, instead of squashing
everything into one RNN hidden state. A separate addressing/attention mechanism then decides,
given a question, which memory slots are relevant, and can even do this in multiple passes
("hops") to chain facts together. The catch with the original 2014 formulation was that the
memory-selection step used **hard, discrete attention** (argmax over memory slots), which is not
differentiable -- so it required extra supervision telling the model exactly which memory slots
were the correct "supporting facts" for each question, which is expensive to obtain and doesn't
scale.

**End-to-End Memory Networks (MemN2N, Sukhbaatar et al., 2015)** fixed this specific limitation:
they replaced the hard argmax memory addressing with **soft attention** (a softmax over memory
slots), making the entire model differentiable end-to-end. This meant the network could be
trained purely from question-answer pairs, with no supervision about which sentences were the
"correct" supporting facts -- the model learns on its own, through gradient descent, which memory
slots to attend to.

### 2. Architecture -- How It Works

**Core intuition:** treat every sentence in the context as its own labeled "memory card." Given
a question, repeatedly ask "which cards are relevant?", read a blend of the relevant cards, fold
that information into your understanding of the question, and repeat a few times so you can
chain together facts that individually don't answer the question but together do.

Concretely, following `EndToEndMemoryNetwork` in the code:
1. **Encode each memory (context sentence) as a single vector.** Each word in a sentence is
   embedded (`input_embedding`), and the code sums the word embeddings within each sentence
   (a simple "bag of embeddings" sentence representation) to get one vector per sentence. A
   learned position encoding is then added per sentence slot so the model can tell sentence 1
   apart from sentence 2, etc.
2. **Encode the question the same way:** embed each word (`question_embedding`) and sum across
   the question's words to get a single question vector. This becomes the initial "query."
3. **Do `num_hops` rounds of reasoning** (3 hops in this code). In each hop:
   - Project the current query through a hop-specific linear layer.
   - Compute a dot-product attention score between the projected query and every memory
     (sentence) vector.
   - Softmax those scores into attention weights over the sentences.
   - Take the weighted sum of the memory vectors -- this is the information "retrieved" from
     memory for this hop.
   - Add the retrieved vector back onto the query (a residual update: `query = query +
     retrieved`), producing an updated query that now incorporates what was just read from
     memory, ready for the next hop.
4. **Answer:** after all hops, the final query vector is projected to a distribution over
   possible answers. In this code's specific setup, the "answer" is which sentence (index 0
   through 4) contains the word the question is asking about -- so `output_projection` maps to
   `max_sentences` logits, i.e., a classification over "which of the 5 context sentences answers
   this," rather than generating a free-form vocabulary word. (The class also defines a separate
   `output_embedding`, mirroring the paper's use of distinct "input" and "output" embedding
   matrices for reading vs. representing memory content, but in this simplified implementation it
   is not actually used inside `forward` -- the sentence vectors that were embedded for attention
   are reused directly as the values that get retrieved.)

The "multiple hops" mechanism is the key link to later architectures: each hop is functionally
very similar to one layer of multi-head self-attention in a Transformer, and stacking multiple
hops is analogous to stacking multiple attention layers -- MemN2N is arguably one of the direct
conceptual ancestors of the Transformer's attention stack.

### 3. Model Size & Parameters

**Original paper:** Sukhbaatar et al.'s MemN2N experiments on the bAbI tasks used small
embedding sizes (around 20-50 dimensions per bAbI task, sometimes up to a few hundred when
jointly trained across many tasks) with 1-3 hops, since bAbI stories are short and synthetic.
Their language-modeling experiments (Penn Treebank, Text8) used larger memory sizes (hundreds of
memory slots) and higher embedding dimensions (around 150-500) since those needed to store many
more, longer-range facts.

**This code's configuration:** `embed_dim=128`, `memory_dim=128`, `num_hops=3`, and
`max_sentences=5` (so the memory can hold at most 5 sentences at a time), with a vocabulary
capped at 2,000 tokens. This is comparable in spirit to the smaller bAbI-scale configurations
from the original paper -- small embedding dimension, a handful of hops, a small number of memory
slots -- because the toy WikiText-based task here is also small and synthetic, much like bAbI.

**Why scaled down:** the point of the exercise is to demonstrate multi-hop reasoning over
explicit memory slots working correctly on a small, fast, easily-inspected task, not to match a
production QA system. A 128-dimensional embedding and 5 memory slots is enough to prove the
mechanism -- attention over memory, iterated a few times -- functions correctly.

### 4. Dataset & What It Was Trained On

**Original models:** the canonical evaluation for both Memory Networks and MemN2N is
**Facebook's bAbI tasks** -- 20 synthetic reasoning tasks (e.g., "single supporting fact,"
"two supporting facts," "three supporting facts," counting, lists/sets, yes/no questions,
positional reasoning) each consisting of short generated stories followed by a question. bAbI
was purpose-built to isolate specific reasoning skills a model needs to demonstrate. MemN2N was
additionally evaluated on language modeling benchmarks (Penn Treebank and Text8) to show the
mechanism generalizes beyond QA.

**This code's demo:** WikiText-2 (`load_wikitext2_dataset`), with sentences of length 5-20
tokens kept, using only the first 800 training / 160 validation / 160 test sentences, and a
2,000-word vocabulary. Since WikiText-2 is not a reasoning benchmark, the code manufactures a
bAbI-like task from it in `_create_qa_examples`: it groups sentences into chunks of 5, picks a
random word that appears somewhere in that chunk, forms a fixed-template question ("what
sentence contains `<word>`"), and the answer label is the index (0-4) of the sentence containing
that word. This is a much simpler, single-hop-sufficient version of bAbI's "supporting fact"
tasks -- it does not require multi-step reasoning across multiple facts the way bAbI's harder
tasks do, but it exercises the exact same architecture (embed sentences into memory slots, embed
a question, attend over memory, predict an answer).

**The gap:** bAbI's harder tasks specifically require chaining 2-3 facts together (which is why
multiple hops matter at all); the WikiText-based sentence-retrieval task here can often be solved
by finding which memory slot contains a matching word, which is closer to a single attention
lookup than genuine multi-step reasoning. The multi-hop machinery is present and exercised, but
the demo task doesn't strictly require more than one hop to solve, unlike bAbI's more elaborate
tasks.

### 5. Training Process

**Objective/loss:** cross-entropy loss (`nn.CrossEntropyLoss()`, no ignore index here) between
the model's predicted sentence-index logits and the true sentence-index label.

**Optimizer:** Adam, learning rate `0.001`.

**Batch size:** 16 (explicitly kept small -- the code comments "Smaller batch size for memory
networks").

**Gradient handling:** gradients are clipped to a max norm of 1.0.

**Training loop structure:** `train_memory_network` runs for `epochs=8`, iterating batches from
a `DataLoader` that uses the custom `collate_memory_fn` to pad variable-length sentences and
questions within a batch and stack them into fixed-size tensors (padding both the number of
sentences per example up to the batch's max, and the length of each sentence/question up to the
batch's max). The loop dispatches to either `EndToEndMemoryNetwork` or `DynamicMemoryNetwork`
depending on `isinstance(model, ...)`, since both models are trained with the same shared loop
and evaluated with the same shared `evaluate_memory_network` function, which reports **accuracy**
(fraction of examples where `argmax(logits) == answer`) rather than loss.

### 6. Training Challenges

- **Differentiability was the whole point:** the historical challenge this specific model solves
  is that the earlier, hard-attention Memory Networks couldn't be trained end-to-end -- gradients
  can't flow through an argmax. MemN2N's soft-attention approach fixes this directly, at the cost
  of the model's attention being "fuzzier" (a weighted blend of memories rather than a clean pick
  of one memory).
- **Variable numbers of memory slots and variable sentence lengths per batch** are awkward to
  batch efficiently -- this repo's `collate_memory_fn` has to pad on two axes at once (number of
  sentences, and length of each sentence), which is more bookkeeping than a standard flat
  sequence batch and is a common source of subtle bugs (e.g., accidentally attending over padding
  sentences) in real memory-network implementations.
- **Choosing the number of hops:** too few hops and the model can't chain facts that require
  multi-step reasoning; too many hops adds parameters and risk of overfitting on small data.
  Sukhbaatar et al. found 2-3 hops sufficient for most bAbI tasks, which is why this code also
  uses 3.
- **Bag-of-embeddings sentence encoding loses word order** within a sentence (since the code sums
  word embeddings rather than using a sequence model to encode each sentence) -- this is a known
  simplification in the original MemN2N paper too (they call it "bag of words" positional
  encoding, and also propose a positional-encoding variant that weights words by position within
  the sentence to partially address this).

### 7. Performance & Evaluation

Historically, MemN2N was evaluated using **per-task accuracy on the 20 bAbI tasks**, and reported
strong results -- solving (greater than 95% accuracy) the large majority of the 20 tasks in the
joint-training setting, without any supervision on which sentences were the supporting facts,
which was the paper's headline result versus the original hard-attention Memory Networks that
needed that supervision. On language modeling (Penn Treebank, Text8) MemN2N achieved perplexity
competitive with, though generally a bit behind, well-tuned LSTM language models of the time.

This code evaluates with the same style of metric -- **validation accuracy** on the toy
sentence-retrieval task -- tracked and printed per epoch, along with training loss, parameter
count, training time, and memory usage, and plotted at the end of `main()`.

### 8. Impact -- Why It Mattered

Memory Networks and MemN2N together established that giving a neural network **explicit,
addressable, external memory** -- separate from the network's own weights -- was a viable and
powerful design pattern, not just a theoretical curiosity. Concretely, this file's own summary
output frames it well: memory addressing (softmax attention over memory slots) is a direct
conceptual precursor to attention over sequences in seq2seq models; multiple hops foreshadow
multi-head/multi-layer attention; and treating memory as a differentiable key-value lookup
foreshadows the query-key-value formulation used throughout the Transformer. MemN2N is also a
direct ancestor of Neural Turing Machines and other differentiable-memory architectures, and its
core idea -- "attend over a set of stored items rather than compress into one hidden state" -- is
exactly the idea the Transformer later generalized into self-attention over an entire sequence.

### 9. How To Explain This In An Interview

"Memory Networks, and especially the End-to-End Memory Network (MemN2N) from 2015, addressed a
real weakness of RNNs: an RNN has to compress everything it needs to remember into one
fixed-size hidden state that gets overwritten every step, which makes multi-fact reasoning hard.
MemN2N instead stores each fact -- in my implementation, each sentence -- as its own memory slot,
and uses softmax attention to decide which memory slots are relevant to a given question. The key
innovation over the original 2014 Memory Networks paper is that this attention is soft and fully
differentiable, so the whole thing trains end-to-end from question-answer pairs alone, with no
need to be told which sentence was the 'correct' supporting fact. It also does this in multiple
'hops' -- it reads from memory, updates its understanding of the question, and reads again,
which lets it chain several facts together for questions that need more than one piece of
information. I trained a small version -- 128-dimensional embeddings, 3 hops, memory holding 5
sentences -- on a synthetic 'which sentence contains this word' task built from WikiText-2, using
Adam and cross-entropy loss, and measured accuracy rather than loss for evaluation. The reason
this model matters historically is that its softmax-over-memory-slots mechanism is essentially a
direct ancestor of the attention mechanism that later became the core of the Transformer."

---

## Dynamic Memory Networks (2015 / 2016)

### 1. What Problem It Solved

MemN2N's memory retrieval is a fairly simple, static process: it sums word embeddings into a
sentence vector, then does the same dot-product attention math at every hop. It has no way to
build a richer, sequential understanding of each sentence (word order inside a sentence is
thrown away by the bag-of-embeddings sum), and it does not explicitly separate "which facts
matter" from "how much have I already understood, and what do I still need to look for." The
Dynamic Memory Network (DMN), introduced by Kumar et al. in 2015 ("Ask Me Anything: Dynamic
Memory Networks for Natural Language Processing"), targeted this gap: it wanted a memory
architecture that (a) encodes each input sentence with a proper sequence model instead of a bag
of words, and (b) performs **iterative, gated episodic memory updates** -- explicitly tracking
"what have I gathered so far" as its own evolving memory state across multiple reasoning
episodes, rather than just updating a query vector.

### 2. Architecture -- How It Works

**Core intuition:** DMN has four cooperating modules -- an input module that reads the story, a
question module that reads the question, an episodic memory module that repeatedly scans the
story for relevant information while keeping track of what it has already gathered, and an
answer module that produces the final answer. The "episodic" part is the key idea: each pass
over the story is called an episode, and after each episode the memory is explicitly updated,
somewhat like a person re-reading a passage multiple times, each time paying attention to
something new based on what they've already figured out.

Concretely, following `DynamicMemoryNetwork` in the code:
1. **Input module:** each context sentence is embedded word-by-word and run through a GRU
   (`input_gru`); the GRU's final hidden state becomes that sentence's representation. This is a
   real sequential encoder (unlike MemN2N's sum-of-embeddings), so word order within a sentence
   is preserved.
2. **Question module:** the question is embedded and run through its own GRU (`question_gru`);
   its final hidden state is the question representation.
3. **Episodic memory module:** the memory starts out initialized to the question representation
   itself (a reasonable starting guess: "what I'm looking for is basically defined by the
   question"). Then, for `num_episodes` iterations (3 in this code):
   - Compute an attention score for every context sentence, based on a feature vector built from
     the sentence's representation, the question representation, the current memory, and the
     element-wise product of the sentence and question representations (this last term lets the
     network directly measure how well a sentence representation "matches" the question,
     dimension by dimension). These features are fed through an `attention_gru`, and the
     resulting hidden state is summed across dimensions to get one attention logit per sentence.
   - Softmax over sentences to get episode attention weights, then take the attention-weighted
     sum of the sentence representations to get the "attended context" for this episode.
   - **Update memory:** concatenate the current memory with the attended context and pass it
     through a linear layer and a tanh nonlinearity to produce the new memory. This is the
     "dynamic" update -- the memory is not just a running attention query like in MemN2N, it is
     its own explicit hidden state that gets nonlinearly transformed each episode.
4. **Answer module:** concatenate the final memory state with the question representation, run
   it through one more GRU (`answer_gru`), and project the result to vocabulary-sized logits.

The multi-episode loop is conceptually similar to MemN2N's multi-hop loop, but richer: each
sentence has its own sequence-level (GRU) representation rather than a bag-of-embeddings sum,
and the "memory" is a separately-parameterized, nonlinearly-updated state rather than simply the
running query vector itself.

One implementation detail worth flagging for accuracy: in this code's `main()`, both
`EndToEndMemoryNetwork` and `DynamicMemoryNetwork` are trained through the exact same loop on the
exact same "which of the 5 sentences answers this" labels. DMN's `answer_projection` produces
`vocab_size`-dimensional logits (designed for predicting an actual answer *word*), but here it is
still trained against a small sentence-index label (0-4), just like MemN2N. This works
mechanically (cross-entropy only needs the true class index to be within range), but it means
this particular demo does not showcase DMN's intended strength -- free-form word-level answer
generation -- it exercises the same simplified retrieval task as MemN2N so the two architectures
can be compared side-by-side on equal footing.

### 3. Model Size & Parameters

**Original paper:** Kumar et al.'s DMN experiments on bAbI used modest hidden sizes (on the order
of 80 GRU units) and typically 1-5 episodes depending on task difficulty; their DMN also was
evaluated on Stanford Sentiment Treebank (sentiment classification) and part-of-speech tagging,
using similarly modest hidden dimensions by today's standards (tens to low hundreds of units).

**This code's configuration:** `embed_dim=128`, `hidden_dim=128`, `memory_dim=128`, and
`num_episodes=3`. This is in the same ballpark as the original paper's bAbI configuration --
compact GRUs, a handful of episodes -- scaled to fit a fast educational demo rather than a paper
benchmark suite.

**Why scaled down:** DMN is already a heavier architecture than MemN2N (it runs a GRU per
sentence, plus a question GRU, plus an attention GRU, plus an answer GRU), so keeping the hidden
size at 128 and episodes at 3 keeps runtime and memory usage manageable for a CPU-friendly demo
while still exercising every module described in the paper.

### 4. Dataset & What It Was Trained On

**Original paper:** DMN's flagship results are on the **bAbI tasks** (the same 20 synthetic
reasoning tasks used to evaluate MemN2N), where it achieved strong accuracy on most tasks,
including some of the harder multi-fact and three-supporting-fact tasks that were more
challenging for simpler architectures. The same paper also evaluated DMN on **Stanford Sentiment
Treebank** for sentiment classification and a **part-of-speech tagging** dataset, showing the
episodic memory idea could be repurposed beyond QA.

**This code's demo:** the same WikiText-2-derived, "which sentence contains this word" synthetic
QA task described above for MemN2N, reusing the identical `train_dataset`/`val_dataset` and
`train_loader`/`val_loader` objects. As with MemN2N, this is a much easier task than the harder
bAbI tasks DMN was designed to shine on, and it does not exercise DMN's word-level answer
generation capability, only its episodic attention mechanism.

### 5. Training Process

**Objective/loss:** the same `nn.CrossEntropyLoss()` used for MemN2N, comparing predicted logits
against the sentence-index label.

**Optimizer:** Adam, learning rate `0.001` -- same hyperparameters as MemN2N, since both models
are trained by the same `train_memory_network` function with the same call-site arguments
(`epochs=8`, `learning_rate=0.001`) in `main()`.

**Batch size:** 16, shared with MemN2N (the same `train_loader`/`val_loader`).

**Gradient handling:** gradients clipped to max norm 1.0, identical to MemN2N.

**Training loop structure:** identical shared loop to MemN2N (`train_memory_network`), with an
`isinstance` check that routes to `DynamicMemoryNetwork.forward` and unpacks
`(logits, episode_attentions)` instead of `(logits, attention_weights)`. Evaluation similarly
shares `evaluate_memory_network`, reporting accuracy.

### 6. Training Challenges

- **Compounding recurrence:** DMN uses four separate GRUs (input, question, attention, answer),
  each of which has its own vanishing/exploding gradient considerations from the RNN era; the
  episodic loop then also unrolls the attention computation `num_episodes` times, adding another
  layer of sequential depth to backpropagate through relative to MemN2N's simpler feed-forward
  hops, so DMN is more exposed to the classic RNN-era gradient issues than MemN2N is.
- **Per-sentence GRU encoding is more expensive** than MemN2N's sum-of-embeddings: the code
  literally runs a separate GRU forward pass for every sentence in every example
  (`for i in range(context.size(1))`), so training cost scales with the number of sentences per
  example, unlike MemN2N which can encode sentences with one embedding-and-sum operation.
- **The attention-scoring GRU is unusual:** feeding concatenated features through a GRU and then
  summing across the hidden dimension to get a scalar attention logit per sentence is a somewhat
  indirect way to compute attention scores (compared to MemN2N's simple dot product), and it adds
  more trainable parameters and more ways for the episodic attention to be miscalibrated early in
  training.
- **Reusing the sentence-index label for a vocabulary-sized output head** (noted above) is itself
  a training-time simplification/quirk worth understanding: it works, but it doesn't test DMN's
  actual designed capability of producing a word-level answer, so a strong validation accuracy
  here reflects "DMN can also solve the retrieval task," not "DMN can generate free-form answers."

### 7. Performance & Evaluation

Historically, DMN was evaluated with **per-task accuracy on bAbI** (like MemN2N), and additionally
with **classification accuracy on Stanford Sentiment Treebank** and **tagging accuracy on
part-of-speech tagging**, demonstrating the episodic memory module generalizes to tasks beyond
QA. DMN's original results were competitive with or ahead of MemN2N on several of the harder bAbI
tasks that require combining more supporting facts, which the paper attributed to the richer
GRU-based sentence encoding and the explicit episodic memory update.

This code evaluates DMN with the same **validation accuracy** metric on the toy sentence-retrieval
task, tracked per epoch alongside MemN2N in the same `training_histories` dictionary so the two
models' loss curves and accuracy curves can be plotted side by side.

### 8. Impact -- Why It Mattered

DMN pushed memory-augmented networks toward more structured, multi-module designs -- separating
"reading the input," "reading the question," "iteratively reasoning," and "answering" into
distinct sub-networks, each with a clear job. Its explicit episodic memory update (a genuinely
separate, nonlinearly-transformed memory state, rather than just an attention query being
carried forward) influenced later memory-augmented and iterative-reasoning architectures. A
follow-up paper (DMN+, Xiong et al., 2016) improved the input module with a bidirectional GRU and
introduced an attention-based GRU for the episodic memory update, further blurring the line
between "memory network" and "attention network" -- another example of memory-network research
converging toward what eventually became the attention mechanisms at the heart of the
Transformer.

### 9. How To Explain This In An Interview

"The Dynamic Memory Network takes the same 'external memory plus attention' idea as MemN2N but
makes it richer and more structured. It has four modules: an input module that runs each context
sentence through a GRU so word order is preserved, a question module that does the same for the
question, an episodic memory module that iterates a fixed number of times -- computing attention
over the sentences based on the question and the current memory state, then explicitly updating a
separate memory vector through a nonlinear transformation -- and finally an answer module that
turns the finished memory state into a prediction. The key difference from MemN2N is that DMN's
memory is its own evolving state, updated with a learned nonlinear function each episode, rather
than just the running attention query itself, and each sentence gets a real sequential encoding
instead of a bag-of-embeddings sum. In my implementation I trained DMN on the same toy
sentence-retrieval task I used for MemN2N, with Adam, cross-entropy loss, and gradient clipping,
so I could compare the two side by side on equal footing. Historically, DMN mattered because it
pushed memory networks toward the multi-module, iterative-attention design pattern, and its
episodic attention mechanism is one more thread that fed into the broader convergence toward
attention-centric architectures that culminated in the Transformer."
