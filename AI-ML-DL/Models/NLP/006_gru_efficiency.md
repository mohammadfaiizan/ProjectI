# 006 — GRU Efficiency (Companion Notes)

Companion to `006_gru_efficiency.py`. This file implements a Gated Recurrent Unit from scratch (`CustomGRUCell`/`CustomGRU`) alongside a standard `nn.GRU`-based model (`PyTorchGRU`), and directly measures training time, inference throughput, memory usage, and parameter count against the LSTM from file 005 to make the "efficiency" claim concrete rather than just asserted.

## GRU — Gated Recurrent Unit (2014)

### 1. What Problem It Solved

LSTM (file 005) solved the vanishing gradient problem, but it did so at a real cost: four separate gates (input, forget, output, and the candidate/"new gate"), each with its own full set of weight matrices, plus a separate cell state that has to be tracked alongside the hidden state. That's a lot of machinery — more parameters, more matrix multiplications per time step, more memory, and more code complexity, for what is, in practice, often only a small quality improvement over a simpler design.

Cho et al. introduced the GRU in 2014 specifically to ask: how much of LSTM's gating complexity can we remove while keeping the core benefit (better gradient flow than a vanilla RNN, ability to learn longer-range dependencies)? The GRU's answer was to collapse LSTM's three "control" gates (forget, input, output) down to two (reset, update), and to eliminate the separate cell state entirely — the hidden state itself now plays the cell state's role. The result is a noticeably smaller, faster, simpler recurrent unit that in practice performs comparably to LSTM on many tasks.

### 2. Architecture — How It Works

**Intuition first.** Where LSTM has one gate deciding what to forget, a separate gate deciding what new information to let in, and a separate output gate deciding what to reveal, GRU merges the "forget vs. let in" decision into a single **update gate**: a knob that directly interpolates between "keep the old hidden state" and "replace it with new information," so there's no need for forget and input to be tracked as two independent decisions. The second gate, the **reset gate**, controls how much of the previous hidden state gets used when computing that candidate "new information" in the first place — essentially, "should I even look at the past when forming this new candidate value, or should I mostly start fresh?"

**The precise computation**, as implemented in `CustomGRUCell.forward`:

```
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)              # reset gate
z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)              # update gate
n_t = tanh   (W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))      # candidate ("new gate")

h_t = (1 - z_t) * n_t + z_t * h_{t-1}                              # blend old and new
```

Notice there's no separate cell state `c_t` at all — `h_t` does double duty as both the thing passed to the next time step and the thing used for output/prediction, unlike LSTM which keeps `h_t` and `c_t` separate.

**Direct comparison to LSTM's gates:**
- LSTM's **forget gate** (`f_t`) and **input gate** (`i_t`) are two independent decisions about how to blend old cell state and new candidate information. GRU's **update gate** (`z_t`) makes this one decision: the weight given to old information (`z_t`) and new information (`1 - z_t`) are forced to sum to 1, so there's no way to (for example) simultaneously forget a lot AND let in only a little, the way LSTM technically can.
- GRU's **reset gate** (`r_t`) has no LSTM equivalent — it controls how much the *candidate* computation itself looks at the past, which is a different kind of control than any of LSTM's three gates.
- LSTM's **output gate** (`o_t`), which controls how much of the cell state is revealed as the hidden state, has no GRU equivalent — GRU's hidden state is always fully "revealed," there is no separate filtering step before output.

**A tiny worked example.** Suppose the update gate computes `z_t = 0.8` (meaning "mostly keep the old hidden state") for some unit where the old hidden state `h_{t-1} = 3.0` and the freshly computed candidate `n_t = -1.0`. Then:

```
h_t = (1 - 0.8) * (-1.0) + 0.8 * 3.0 = -0.2 + 2.4 = 2.2
```

The hidden state moved only a little from 3.0 toward 2.2, mostly preserving the old value — analogous to LSTM's forget gate staying close to 1. The mechanism is different (one interpolation weight instead of two independent gates), but the effect — being able to preserve information across time steps instead of being forced to overwrite it — is the same core trick that fixes vanishing gradients.

### 3. Model Size & Parameters

Historically, GRUs became popular specifically because they matched LSTM's quality on many sequence tasks (especially smaller or less data-rich ones) while training faster and using fewer parameters, which made them attractive as a default choice, especially in the mid-2010s when compute budgets were tighter than they are today. Typical hidden sizes were similar to LSTM's — 256-1024 depending on the task.

This repo's code uses, for both `CustomGRU` and `PyTorchGRU`: `embedding_dim=128`, `hidden_dim=256`, `num_layers=2`, `dropout=0.2` — identical embedding/hidden dimensions to the LSTM in file 005, which is intentional, so the parameter/speed comparison is apples-to-apples. GRU's gates need `(3*hidden) x input` and `(3*hidden) x hidden` weight matrices (3 gates instead of LSTM's 4), so the recurrent layers contribute roughly 691,000 parameters — versus roughly 921,000 for the same-sized LSTM, about **25% fewer parameters in the recurrent layers specifically**. Once you add the embedding table (~384,000) and output projection (~771,000), which are identical between the two models, the total model comes out to roughly **1.85 million parameters** for GRU versus roughly **2.1 million** for LSTM — an overall reduction of a bit over 10%, since the embedding and output layers (unaffected by the gating choice) make up a large share of the total either way. The code's own `compare_computational_efficiency` function measures this directly, alongside inference throughput and model size in megabytes.

### 4. Dataset & What It Was Trained On

The original GRU paper (Cho et al., 2014) introduced it as part of an encoder-decoder model for statistical machine translation, evaluated on translation benchmarks. GRUs subsequently became widely used across general sequence modeling tasks — language modeling, speech, translation — anywhere an LSTM might otherwise be used.

This code trains on the same **WikiText-2** setup as the LSTM file: 20,000 training tokens, 4,000 validation tokens, 4,000 test tokens, tokenized and lowercased with NLTK, vocabulary capped at 3,000 words, sequences of length 100 (non-overlapping). Using identical dataset slices and sequence lengths to file 005 is deliberate — it's what makes the head-to-head GRU-vs-LSTM comparison in this repo meaningful rather than confounded by different data.

### 5. Training Process

**Objective and optimization** are identical in structure to the LSTM file: next-word language modeling with `nn.CrossEntropyLoss`, Adam optimizer with learning rate `0.001`, batch size 20, 8 epochs, gradient clipping at norm `1.0`.

**Two implementations trained side by side**, exactly mirroring file 005's structure: `CustomGRU` manually loops over time steps and layers calling `CustomGRUCell`, making the reset/update gate math fully inspectable; `PyTorchGRU` uses `nn.GRU`, the optimized built-in version. Training both under identical hyperparameters is what lets the code's efficiency comparison isolate "does the custom implementation match the optimized one" from "does GRU differ from LSTM in speed/memory."

**Historical note.** Cho et al.'s original GRU was trained the same general way — gradient-based optimization with backpropagation through time — the main change over the years has been the optimizer (Adam largely replacing plain SGD/RMSProp-style methods) and hardware, not the fundamental training procedure.

### 6. Training Challenges

GRU still relies on BPTT and so is not immune to vanishing/exploding gradients in principle, but the same core mitigation applies: the update gate can learn to sit close to 1 (mostly preserving old hidden state) or close to 0 (mostly accepting new information), giving gradients a path that avoids the aggressive squashing a vanilla RNN forces on every step.

A subtler practical challenge, and part of why some practitioners still prefer LSTM in specific cases, is that GRU's single update gate is less expressive than LSTM's independent forget/input gates — there are hidden-state update patterns an LSTM can represent (forget a lot AND accept a lot of new information simultaneously) that a GRU structurally cannot, because `z_t` and `1 - z_t` are tied together. In practice this rarely matters for the datasets and tasks in this repo's demonstrations, but it's a real, provable difference in representational capacity, not just a stylistic one.

The main computational challenge GRU is *not* subject to (compared to LSTM) is the overhead of tracking and updating a separate cell state at every step — this is precisely the simplification that gives GRU its speed and memory advantage, and it's what the `memory_usage_analysis` and `compare_computational_efficiency` functions in this code are built to measure directly (forward/backward memory deltas, throughput in samples/second, model size in MB).

### 7. Performance & Evaluation

Evaluation again uses **perplexity** (`exp` of average cross-entropy loss), matching file 005 exactly so the two architectures can be compared on equal footing. The code additionally reports **inference throughput** (samples/second) and **memory usage** during a training step, which are the metrics that actually matter for the "efficiency" claim in this file's title — perplexity tells you about model quality, throughput and parameter count tell you about cost.

Historically, GRU's empirical finding (reinforced by many follow-up studies through the mid-2010s) was that GRU and LSTM achieve broadly comparable quality on most sequence modeling and translation tasks, with GRU training somewhat faster due to fewer parameters and simpler per-step computation — exactly the trade-off this file's comparison is designed to surface at small scale.

### 8. Impact — Why It Mattered

GRU gave practitioners a genuinely useful choice: when compute or data was limited, or when a slightly simpler, faster architecture was preferable, GRU was (and often still is) a very reasonable default that rarely loses much quality to LSTM. It became a standard building block right alongside LSTM throughout the sequence-to-sequence and attention era (2014-2017) — many machine translation and text generation systems from that period used GRU-based encoders and decoders. In the larger arc of this series, GRU doesn't change the fundamental story (gating fixes vanishing gradients; both GRU and LSTM are still constrained by sequential, step-by-step processing that can't be parallelized across time), which is exactly the limitation that the next files in this series — Seq2Seq (007), attention (008, 009), and self-attention (010) — progressively work around, culminating in the fully parallelizable, recurrence-free Transformer.

### 9. How To Explain This In An Interview

"GRU is a simplification of LSTM: instead of LSTM's three gates (forget, input, output) plus a separate cell state, GRU uses two gates — a reset gate that controls how much past information feeds into computing new candidate values, and an update gate that directly interpolates between the old hidden state and the new candidate. There's no separate cell state at all; the hidden state does that job itself. That's roughly 3 weight matrices per direction instead of LSTM's 4, which in my implementation worked out to about 25% fewer parameters in the recurrent layers specifically, and noticeably faster training and inference, while keeping the same core benefit over vanilla RNNs — the update gate can hold close to 1 and let gradients flow through many time steps largely unchanged, avoiding the vanishing gradient problem. I implemented the GRU cell from scratch, matched it against PyTorch's built-in `nn.GRU`, and trained both as language models on WikiText-2 under the exact same hyperparameters and data as the LSTM comparison, so I could measure the efficiency trade-off directly — parameter count, training time, memory, and throughput — rather than just assert it. The finding matches the historical consensus: GRU gets you comparable quality to LSTM with meaningfully less compute, which is why it became a standard alternative rather than a strict downgrade. Both GRU and LSTM still process sequences one step at a time though, which is the limitation that Seq2Seq, attention, and eventually self-attention/Transformers all chip away at next."
