# 004 — RNN Fundamentals (Companion Notes)

Companion to `004_rnn_fundamentals.py`. This file trains a plain ("vanilla") Recurrent Neural Network and a deeper 3-layer version on WikiText-2, specifically to make the vanishing-gradient problem visible, not to build a great language model.

## Vanilla RNN (2010-2015)

### 1. What Problem It Solved

Before RNNs, the standard neural network was a feedforward network: you give it a fixed-size input (like a vector of a fixed length), and it gives you a fixed-size output. That works fine for things like image classification, but language is not fixed-size. A sentence can be 3 words or 30 words. A feedforward network has no natural way to handle "I don't know how long this input will be" or "the meaning of this word depends on the ten words before it."

Two workarounds existed before RNNs: (1) use a fixed-size sliding window of the last N words (like n-gram models), which throws away everything before the window, or (2) pad/truncate everything to one length, which either wastes computation or cuts off information. Neither lets the network use an arbitrary amount of past context.

The RNN's fix was architectural: give the network a "memory" — a hidden state vector that gets updated at every time step and carries forward whatever it has seen so far. This let one single set of weights process a sequence of any length, one token at a time, while (in principle) accumulating information about everything before the current position.

### 2. Architecture — How It Works

**Intuition first.** Think of the RNN as reading a sentence word by word, and after each word, updating a small notebook (the hidden state) that summarizes everything read so far. The same "update the notebook" rule is reused at every position — that's what makes it "recurrent": the same function is applied over and over, feeding its own output back in as part of the next input.

**The precise computation.** At each time step `t`, the RNN takes the current input `x_t` (here, a word embedding) and the previous hidden state `h_{t-1}`, and produces a new hidden state:

```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

- `W_ih` is the input-to-hidden weight matrix — it decides how much the current word matters.
- `W_hh` is the hidden-to-hidden weight matrix — it decides how much of the past to carry forward and how to blend it with the new word.
- `tanh` squashes the result into the range (-1, 1), which keeps the hidden state from growing without bound.

At every time step, `h_t` is also projected through an output layer (`output_projection`, a plain linear layer) to produce a probability distribution over the vocabulary — "given everything so far, what's the next word?"

**Backpropagation Through Time (BPTT).** This is how the RNN is trained. Because the same weights `W_ih` and `W_hh` are reused at every time step, computing the gradient (how much to adjust the weights) requires "unrolling" the RNN across all time steps and applying the chain rule backward through each one. If a sequence has 50 time steps, gradients from the loss at the end have to flow backward through 50 repeated multiplications by `W_hh` (and 50 derivatives of `tanh`) to reach the weights that processed the first word.

**A tiny worked example of why that's dangerous.** Say the derivative of `tanh` at some point is roughly 0.5 (the max derivative of tanh is 1.0, and it's usually much less than that away from zero), and imagine `W_hh`'s dominant eigenvalue is also around 0.5. Every step backward through time multiplies the gradient by roughly `0.5 * 0.5 = 0.25`. After just 10 steps, the gradient signal has shrunk by a factor of `0.25^10 ≈ 0.00000095` — essentially zero. That's the **vanishing gradient problem**: the further back in time a word is, the more its gradient signal gets multiplied down toward zero, so the network effectively can't learn "this word 20 steps ago mattered." The opposite failure, **exploding gradients**, happens if that per-step factor is greater than 1 — the gradient grows exponentially instead of shrinking, and weight updates become huge and unstable (you'll see `NaN` losses or huge oscillations).

`004_rnn_fundamentals.py` demonstrates both sides of this directly: `VanillaRNN` (1 recurrent layer) and `DeepRNN` (3 stacked recurrent layers) are trained side by side, and a helper function (`analyze_gradients`) records the gradient norms of the recurrent weights every 2 epochs so you can literally watch them shrink as depth increases.

### 3. Model Size & Parameters

Historically, the original RNN work (Elman networks, and their revival for language modeling around 2010-2011 by Mikolov et al.) used very small hidden sizes by modern standards — often in the low hundreds of hidden units, because larger RNNs were both hard to train (worse vanishing gradients) and computationally expensive to run token-by-token on the hardware of the time.

This repo's code uses:
- `VanillaRNN`: `embedding_dim=100`, `hidden_dim=128`, `num_layers=1`, `dropout=0.2`
- `DeepRNN`: `embedding_dim=100`, `hidden_dim=128`, `num_layers=3`, `dropout=0.3`, with a smaller orthogonal initialization gain (`gain=0.5`) on the recurrent weights specifically to try to keep the deeper network's activations under control

With a vocabulary capped at 3,000 words, `VanillaRNN` comes out to roughly **700,000 parameters** (about 300K in the embedding table, about 30K in the single recurrent layer, and about 390K in the output projection back to vocabulary size — the embedding and output layers dominate, not the recurrence itself). `DeepRNN` is only slightly larger, around **780,000 parameters**, because stacking two more recurrent layers of the same hidden size adds relatively few extra weights compared to the embedding/output layers.

This is deliberately tiny compared to any real-world RNN language model (which might use hidden sizes of 512-2048 and vocabularies of tens of thousands of words). The point of scaling it down here isn't performance — it's to run fast on a laptop CPU/GPU in minutes while still being large enough to visibly demonstrate the vanishing gradient effect when you go from 1 layer to 3.

### 4. Dataset & What It Was Trained On

Early RNN language models (e.g., Mikolov's RNNLM) were typically trained on corpora like the Penn Treebank or similar newswire text, evaluated primarily by perplexity on held-out text.

This code trains on **WikiText-2**, a collection of verified "Good" and "Featured" Wikipedia articles, loaded via Hugging Face's `datasets` library. The text is lowercased and tokenized with NLTK's `word_tokenize`. Only the first 15,000 training tokens, 3,000 validation tokens, and 3,000 test tokens are actually used (a small slice of the full dataset), and the vocabulary is capped at the 3,000 most frequent words (with `<UNK>` for anything else and `<PAD>` for padding).

WikiText-2 is a reasonable stand-in for teaching purposes for a few reasons: it's real, grammatical English prose (unlike a synthetic toy dataset), it's small enough to download and tokenize in seconds, and it's the same dataset reused consistently across files 004-010 in this repo, which makes it possible to compare architectures apples-to-apples across the whole NLP evolution series. It is not, however, anywhere near the scale (or genre diversity) of what modern language models train on — that's a deliberate simplification for a learning exercise.

### 5. Training Process

**Objective.** Both RNNs are trained as language models: given the words seen so far, predict the next word. The loss function is `nn.CrossEntropyLoss`, which measures how far the model's predicted probability distribution over the vocabulary is from the actual next word (a one-hot target).

**Data preparation.** Token sequences of length 50 are created with a sliding window that overlaps by half (`stride = seq_length // 2 = 25`), so consecutive training examples share context — this gives more training examples out of limited text.

**Optimization.** The optimizer is Adam with a learning rate of `0.001`. Training runs for 8 epochs with a batch size of 32. After the loss is computed and `.backward()` is called, gradient clipping (`torch.nn.utils.clip_grad_norm_`) rescales the gradient if its total norm exceeds a threshold — `1.0` for `VanillaRNN` and a stricter `0.5` for `DeepRNN` (the deeper network needs a tighter clip because its gradients are more prone to instability). This is the standard fix for exploding gradients: cap the gradient's magnitude without changing its direction.

**Historical note.** The very first RNN language models were trained with plain stochastic gradient descent (not Adam, which didn't exist yet), often with manually-tuned learning rate schedules and sometimes truncated BPTT (only backpropagating a limited number of steps back) purely to make training tractable on 2010-era hardware.

### 6. Training Challenges

The headline challenge is the vanishing gradient problem described above: as the sequence gets longer or the network gets deeper, gradients from later time steps shrink exponentially by the time they reach earlier time steps or earlier layers, so the network struggles to learn dependencies that span more than a handful of tokens.

This code surfaces that challenge two ways: first, by literally comparing 1-layer vs. 3-layer RNNs and tracking their gradient norms; second, through the initialization choices — `weight_hh` (the recurrent weight matrix) is initialized orthogonally rather than randomly, because an orthogonal matrix preserves vector norms under repeated multiplication, which slows down (but does not eliminate) the exponential shrinkage/growth of the gradient signal across time steps. The `DeepRNN` additionally uses a reduced orthogonal gain (0.5) to further dampen the compounding effect across its extra layers.

Exploding gradients are the secondary challenge and are handled directly with gradient clipping, tuned tighter for the deeper model.

A more mundane challenge is that RNNs process tokens strictly one at a time — step `t` needs the result of step `t-1` — so there's no way to parallelize across the sequence dimension the way you can with, say, a convolution or (later) a Transformer. That's a training-speed problem more than a training-quality problem, but it's part of why RNNs eventually lost favor.

### 7. Performance & Evaluation

Language models like this are evaluated with **perplexity**, computed here as `exp(average cross-entropy loss)` over the validation/test set. Perplexity has a clean interpretation: it's roughly "the average number of equally-likely word choices the model is confused between at each step." Lower is better; a perfect model would have a perplexity of 1 (no uncertainty at all), and a model just guessing uniformly among a 3,000-word vocabulary would have a perplexity around 3,000.

Historically, early neural RNN language models (Mikolov, 2010) reported meaningful perplexity improvements over n-gram baselines on benchmarks like Penn Treebank, which is what made RNNs interesting in the first place — despite the gradient problems, they still beat the older statistical approaches on capturing short-to-medium range dependencies. In this repo's small-scale run, the exact perplexity numbers matter less than the relative comparison: expect the deeper RNN to not clearly outperform (and possibly underperform) the single-layer RNN on this small dataset/short training run, which is itself the intended lesson — depth without a fix for vanishing gradients doesn't reliably help.

### 8. Impact — Why It Mattered

The RNN proved the core idea that a sequence model needs: shared weights across time, with a hidden state carrying context forward, enabling variable-length input processing. That idea persists in every sequence architecture that followed. But this file's own conclusion is the important one for the story arc of this whole NLP series: vanilla RNNs made vanishing/exploding gradients obvious and painful enough that researchers went looking for an architecture that could keep gradient signal alive over long distances. That search produced LSTM (1997, revived and popularized in the mid-2010s) and later GRU (2014) — both of which are the direct subject of the next two files in this series, and both of which fix the vanishing gradient problem with gating mechanisms rather than trying to out-engineer the plain RNN's recurrence formula.

### 9. How To Explain This In An Interview

"Before RNNs, feedforward networks couldn't naturally handle variable-length sequences like sentences. The RNN's fix is to keep a hidden state that gets updated at every time step using the same shared weights, so one architecture can process a sequence of any length while carrying context forward. I trained a vanilla RNN and a 3-layer deep RNN as language models on WikiText-2 — next-word prediction, cross-entropy loss, Adam optimizer, gradient clipping — and the key thing I wanted to demonstrate wasn't state-of-the-art performance, it was the vanishing gradient problem. Because training an RNN means backpropagating through every time step (BPTT), and the same weight matrix gets multiplied in at every step, gradients from later time steps get multiplied by the derivative of tanh and by the recurrent weight repeatedly as they flow backward — if that per-step factor is less than 1, which it usually is, the gradient shrinks exponentially and the network can't learn long-range dependencies. I made this visible by tracking gradient norms across a 1-layer vs. 3-layer RNN and by using orthogonal initialization on the recurrent weights to slow the effect down. Exploding gradients are the flip side, and I handled those with gradient clipping. This vanishing gradient problem is exactly what motivated LSTM and GRU, which replace the plain recurrence with gated mechanisms that let gradients flow through a more stable path."
