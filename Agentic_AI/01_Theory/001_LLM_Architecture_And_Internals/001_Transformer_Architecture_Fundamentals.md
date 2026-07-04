# Transformer Architecture Fundamentals: The Decoder-Only Block

## Why This Block, and Not Something Else

Every modern large language model you interact with — GPT-4, Claude, Llama, Mistral, DeepSeek — is built by stacking the same basic unit dozens or hundreds of times: a decoder-only transformer block. Understanding this block cold is the single highest-leverage thing you can know for an LLM engineering interview, because everything else (tokenization, positional encoding, MoE, KV caching, quantization) is a modification or optimization layered on top of this core repeating structure.

A decoder-only block consists of two sublayers, each wrapped in a residual connection and a normalization step: a self-attention sublayer, followed by a position-wise feed-forward (MLP) sublayer. That's it. The entire model is `N` copies of this block stacked on top of an embedding layer, ending in a final normalization and a linear projection to vocabulary logits. The apparent simplicity is deliberate — the architecture's power comes almost entirely from scale (parameters, data, compute) applied to this uniform, easily-parallelizable structure, not from architectural cleverness per block. This document walks through every piece of that block from first principles, and then explains why the *decoder-only* variant specifically won out over the original encoder-decoder transformer for generative LLMs.

## Self-Attention From First Principles

### The core problem attention solves

Before attention, sequence models (RNNs, LSTMs) processed tokens one at a time, carrying forward a fixed-size hidden state. This made it hard for information from far back in the sequence to influence the current step — the signal had to survive many sequential transformations, and it also made training inherently sequential and slow. Attention solves both problems: at each position, it lets a token look directly at every other token in the sequence in a single step, weighted by how relevant each other token is, and this operation is fully parallelizable across positions during training.

The mechanism needs to answer, for every position, "which other positions should I gather information from, and how much from each?" Attention answers this by giving every token three learned vector representations of itself — a query, a key, and a value — and structuring the lookup like an information-retrieval system.

### Query, Key, Value — the intuition before the math

Think of it like a soft key-value database lookup. Every token emits a **query**: a vector describing what kind of information it is looking for. Every token also emits a **key**: a vector describing what kind of information it offers, so it can be matched against other tokens' queries. Finally every token emits a **value**: the actual content it will hand over if it is selected as relevant. The query of token *i* is compared (via dot product) against the keys of every token in the sequence to produce a similarity score per pair; those scores are turned into a probability distribution (softmax), and the output for token *i* is the weighted sum of all tokens' value vectors under that distribution. Critically, Q, K, and V are not the raw embeddings — they are three separate learned linear projections of the same underlying representation, which is what lets the model learn to look for different things than it offers, and to offer different content than what it uses for matching.

### The math

Given an input sequence represented as a matrix `X` of shape `(seq_len, d_model)`, we compute three projections using learned weight matrices:

```
Q = X @ W_Q      # (seq_len, d_k)
K = X @ W_K      # (seq_len, d_k)
V = X @ W_V      # (seq_len, d_v)
```

Scaled dot-product attention is then:

```
Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V
```

`Q @ K^T` produces a `(seq_len, seq_len)` matrix of raw similarity scores between every pair of positions. Softmax is applied row-wise so that each token's attention weights over all other tokens sum to 1. In a decoder-only (causal) model, before the softmax we also apply a causal mask that sets scores for future positions to `-inf`, so a token can never attend to positions after itself — this is what makes the architecture usable for autoregressive generation, since at inference time future tokens don't exist yet, and at training time it prevents the model from "cheating" by looking ahead.

### Why the 1/sqrt(d_k) scaling factor exists

This is one of the most commonly asked "why" questions in interviews, and the reasoning is a clean piece of statistics, not an arbitrary hyperparameter. Assume the components of `Q` and `K` are independent random variables with mean 0 and variance 1 (a reasonable approximation early in training, given typical initialization schemes). The dot product of two such `d_k`-dimensional vectors is a sum of `d_k` independent products, each with mean 0 and variance 1, so the dot product itself has variance `d_k` (variance is additive for sums of independent variables). This means as `d_k` grows, the raw dot-product scores grow in magnitude roughly proportional to `sqrt(d_k)`.

Large-magnitude inputs to softmax are a problem because softmax's gradient is proportional to `p_i * (1 - p_i)` for each output probability `p_i`. When the input scores are large, softmax saturates — it pushes almost all probability mass onto the single largest score, making the output distribution close to one-hot. Once that happens, `p_i` is close to 0 or 1 for every entry, so `p_i * (1 - p_i)` is close to zero almost everywhere, and gradients vanish. The model effectively stops learning how to distribute attention because the softmax has become nearly a hard argmax with a flat gradient landscape around it.

Dividing by `sqrt(d_k)` exactly cancels the variance growth: it rescales the dot products back down to unit variance regardless of `d_k`, keeping softmax in a regime where its gradients are informative. Without this scaling, empirically, attention training becomes unstable and slow to converge, especially in the earlier, more sensitive stages of training, and the effect gets worse as head dimension grows. This is why the "Attention is All You Need" paper introduced the term explicitly and called it "scaled" dot-product attention to distinguish it from an earlier additive-attention formulation that didn't have this problem in the same way.

### Attention implemented from scratch

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, causal_mask=True):
    """
    Q, K: (seq_len, d_k)
    V:    (seq_len, d_v)
    """
    seq_len, d_k = Q.shape
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len)

    if causal_mask:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(mask, -np.inf, scores)

    weights = softmax(scores, axis=-1)        # each row sums to 1
    output = weights @ V                      # (seq_len, d_v)
    return output, weights

# toy example: 4 tokens, d_model = 8, single head with d_k = d_v = 8
np.random.seed(0)
seq_len, d_model = 4, 8
X = np.random.randn(seq_len, d_model)

W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

Q, K, V = X @ W_Q, X @ W_K, X @ W_V
out, attn_weights = scaled_dot_product_attention(Q, K, V, causal_mask=True)
print(attn_weights.round(3))
```

Running this, you'd see that the attention-weight matrix is lower-triangular (because of the causal mask — token 0 can only attend to itself, token 1 to tokens 0-1, and so on), and each row sums to 1.

## Multi-Head Attention

### Why not just use one big attention operation

A natural question is: if a single attention operation with a large `d_k` already lets every token attend to every other token, why split it into multiple smaller "heads" instead of running one large attention function? The answer is about representational diversity, not raw capacity. A single softmax distribution per query position is fundamentally limited to expressing *one* pattern of relevance at a time — one weighted average over the sequence. But language requires attending to multiple, qualitatively different kinds of relationships simultaneously: a pronoun needs to resolve its antecedent, a verb needs to check subject-verb agreement, a token might need nearby positional context, and a word might need long-range thematic context — all at once, at the same position.

Multi-head attention runs several independent attention operations in parallel, each with its own learned `W_Q`, `W_K`, `W_V` projecting into a smaller subspace (typically `d_k = d_model / num_heads`), and then concatenates their outputs and projects back to `d_model` with a final learned matrix `W_O`. Because each head has independently-learned projections, each head is free to specialize in a different kind of relationship, and the concatenation lets the model combine several such relationships into a single output vector per token. Empirically and via interpretability studies (attention-head probing, path patching, activation patching in circuits-style analysis), researchers have found heads that reliably specialize: some heads attend almost entirely to the immediately preceding token (positional heads), some attend to the first token of the sequence (a kind of "no-op" or attention-sink behavior), some track syntactic dependencies like matching a verb to its subject, and famous "induction heads" learn to complete patterns of the form "if token A was followed by token B earlier in the context, and A appears again now, predict B" — this specific circuit is considered one of the building blocks of in-context learning.

### The math of multi-head attention

```
head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)     for i in 1..h
MultiHead(X) = Concat(head_1, ..., head_h) @ W_O
```

Total compute is roughly the same as one large attention operation over `d_model`, because splitting into `h` heads of dimension `d_model / h` each, and running `h` of them, costs about the same FLOPs as one head of the full dimension — the benefit is purely about what the heads *learn to represent*, not extra capacity in the naive parameter-count sense (although the extra `W_O` mixing matrix and the independent per-head projections do add some expressive flexibility).

```python
def multi_head_attention(X, num_heads, W_Q, W_K, W_V, W_O, causal_mask=True):
    """
    X: (seq_len, d_model)
    W_Q, W_K, W_V: (d_model, d_model)  -- combined projections for all heads
    W_O: (d_model, d_model)
    """
    seq_len, d_model = X.shape
    d_k = d_model // num_heads

    Q = (X @ W_Q).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (h, seq_len, d_k)
    K = (X @ W_K).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = (X @ W_V).reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)

    head_outputs = []
    for h in range(num_heads):
        out, _ = scaled_dot_product_attention(Q[h], K[h], V[h], causal_mask=causal_mask)
        head_outputs.append(out)

    concat = np.concatenate(head_outputs, axis=-1)   # (seq_len, d_model)
    return concat @ W_O
```

In production LLMs, plain multi-head attention (MHA) has largely been supplanted at inference-critical scale by **multi-query attention (MQA)**, where all heads share a single K/V projection (only Q stays per-head), and **grouped-query attention (GQA)**, a middle ground where a small number of K/V head groups are shared among multiple query heads. Llama 2 70B and Llama 3, Mistral, and many other production models use GQA specifically because the KV cache — the stored keys and values for every past token, which must be kept in GPU memory during autoregressive generation — scales with the number of KV heads. Reducing distinct KV heads shrinks the KV cache memory footprint and the memory-bandwidth cost of reading it at every decoding step, which is often the actual inference bottleneck, with only a small quality cost relative to full MHA.

## The Feed-Forward (MLP) Sublayer

### Structure and expansion ratio

Every transformer block's second sublayer is a position-wise feed-forward network — "position-wise" meaning the exact same MLP weights are applied independently to each token's vector, with no mixing across positions (all cross-token mixing happens in attention). The classic form is two linear layers with a nonlinearity in between:

```
FFN(x) = W_2 @ activation(W_1 @ x + b_1) + b_2
```

The inner dimension is expanded relative to `d_model`, classically by a factor of 4 (e.g., `d_model = 768` expands to `3072` in the original GPT-2 style), before being projected back down. This expansion ratio is one of the more consistent architectural choices across model families, though modern gated-activation models often use a somewhat different effective ratio (discussed below) to keep total parameter count comparable when they add a third weight matrix.

While attention is often described as the "interesting" part of the transformer, the FFN sublayers actually contain the majority of the model's parameters — for a hidden size `d`, attention projections are roughly `4d^2` parameters (Q, K, V, O), while the FFN with expansion ratio 4 is `8d^2` parameters (two matrices of `4d^2` each), so the FFN sublayers typically account for roughly two-thirds of a block's non-embedding parameters. When people say "most of an LLM's knowledge is stored in the MLP layers," this parameter distribution is part of why that claim is architecturally plausible.

### The FFN as key-value memory

A significant line of interpretability research (notably Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories," and follow-up work on locating and editing factual associations, such as the ROME/MEMIT line of work) reframes the FFN sublayer not as a generic nonlinear transformation but as an associative memory lookup, structurally analogous to attention itself. In this view, the first matrix `W_1` acts as a bank of "keys" — each row/neuron computes a dot product with the input, and after the nonlinearity, high activations indicate that the current hidden state matches a particular stored pattern (this could be a specific fact, a syntactic pattern, or a semantic category). The second matrix `W_2` then acts as a bank of "values" — each activated neuron contributes its corresponding row of `W_2`, scaled by its activation, into the output, effectively retrieving and adding in the associated value vector.

This framing explains several empirically observed phenomena: individual FFN neurons can be found that activate specifically for concepts like "countries" or "the color associated with an object," specific factual associations (like "The Eiffel Tower is located in ___") can often be localized to a small number of middle-layer FFN weights, and this localization is precisely what makes targeted model-editing techniques like ROME/MEMIT possible — they work by directly writing new key-value associations into specific FFN weight matrices rather than retraining the whole model. This "memory" interpretation is also why the FFN sublayer is sometimes described as where the model stores its factual and world knowledge, while attention is more responsible for routing and combining information contextually.

### Activation functions: ReLU, GELU, and gated variants (GLU/SwiGLU)

The choice of nonlinearity inside the FFN has evolved meaningfully across model generations, and this is a favorite interview detail because it connects theory to concrete named models.

**ReLU** (`max(0, x)`) was the original choice, used in the original transformer paper and early GPT models. It's cheap and avoids vanishing gradients for positive inputs, but it has a hard, non-smooth kink at zero and a "dead neuron" failure mode where a neuron that always receives negative input stops learning entirely, since its gradient is exactly zero there.

**GELU** (Gaussian Error Linear Unit), `x * Phi(x)` where `Phi` is the standard Gaussian CDF, smooths out that kink. Intuitively, instead of a hard gate that either passes `x` through unchanged or zeroes it, GELU weights `x` by the probability that a standard normal variable is less than `x`, producing a smooth, differentiable curve that behaves like a soft, stochastic version of ReLU. GELU became the default in BERT, GPT-2, GPT-3, and remained extremely common because the smoothness tends to help optimization, especially at the depths and scales used in large models. In practice it's often approximated with a `tanh`-based formula for computational efficiency rather than computing the exact Gaussian CDF.

**GLU (Gated Linear Unit) variants**, and specifically **SwiGLU**, are what most current-generation frontier models use (Llama 1/2/3, PaLM, Mistral, and others). A GLU-style FFN splits the up-projection into two parallel matrices instead of one, and uses one of them to *gate* the other elementwise:

```
GLU_variant(x) = (activation(x @ W_gate)) * (x @ W_up)
FFN_swiglu(x) = ( SiLU(x @ W_gate) * (x @ W_up) ) @ W_down
```

where SiLU (also called Swish), `x * sigmoid(x)`, is the activation applied to the gating branch. The intuition is that this gives the FFN an input-dependent, multiplicative gating mechanism rather than a fixed elementwise nonlinearity — the network can learn to let some feature dimensions pass through nearly unchanged while suppressing others conditionally, per input, which is a strictly more expressive computational primitive than a fixed pointwise function. The paper that popularized this for transformers (Shazeer, "GLU Variants Improve Transformer") tested several gating functions (ReGLU, GEGLU, SwiGLU) and found consistent, if modest, quality improvements over plain ReLU/GELU FFNs at equal parameter count, with SwiGLU (Swish-gated) generally performing best.

The practical cost of GLU variants is that they need *three* weight matrices instead of two (`W_gate`, `W_up`, `W_down`), so to keep total FFN parameter count comparable to a standard 4x-expansion ReLU/GELU FFN, models using SwiGLU typically shrink the hidden expansion ratio to roughly `8/3 * d_model` rather than `4 * d_model` — this is exactly what the Llama technical reports specify. This is a good concrete fact to have ready in an interview: Llama's FFN uses SwiGLU with an approximately 8/3x expansion rather than 4x, specifically to hold parameter count constant against a classical two-matrix, 4x-expansion GELU FFN.

```python
import numpy as np

def silu(x):
    return x / (1 + np.exp(-x))

def swiglu_ffn(x, W_gate, W_up, W_down):
    """
    x: (seq_len, d_model)
    W_gate, W_up: (d_model, d_ff)
    W_down: (d_ff, d_model)
    """
    gate = silu(x @ W_gate)
    up = x @ W_up
    return (gate * up) @ W_down
```

## Residual Connections

Residual (skip) connections wrap every sublayer: instead of a block computing `x_out = Sublayer(x_in)`, it computes `x_out = x_in + Sublayer(x_in)`. This single design choice is what makes it possible to stack transformer blocks tens or hundreds of layers deep at all. The reasoning traces back to the same motivation as ResNets in computer vision: in a deep composition of functions, gradients computed via the chain rule during backpropagation are products of many Jacobians, and if each sublayer's Jacobian tends to shrink (or grow) the gradient signal even slightly, that effect compounds multiplicatively across dozens of layers, producing vanishing or exploding gradients that make deep networks untrainable.

A residual connection provides an additive "identity shortcut" alongside the sublayer's transformation. Because `d(x_in + Sublayer(x_in))/d(x_in) = I + d(Sublayer)/d(x_in)`, the gradient flowing backward always has that identity term `I` added to it, guaranteeing a direct, unattenuated path for gradient to flow from the output all the way back to any earlier layer's input, regardless of how the sublayer's own Jacobian behaves. This means that even if a particular attention or FFN sublayer's gradient were to shrink toward zero, the residual path still carries a full-strength gradient signal past it. Practically, this is also why transformer blocks are often described as learning a *residual update* to a persistent "hidden state stream" — sometimes called the residual stream in interpretability literature — where each block reads from the stream, computes some contribution, and additively writes its update back in, rather than fully replacing the representation at each layer.

## LayerNorm vs RMSNorm

### What normalization is for here

Normalization layers inside the transformer block exist to control the scale and distribution of activations flowing between sublayers, which stabilizes training especially at the depths and widths used in large models — without it, activation magnitudes tend to drift across layers and training becomes highly sensitive to learning rate and initialization.

**LayerNorm**, used in the original transformer, BERT, and GPT-2/3, normalizes each token's activation vector independently (across the feature/hidden dimension, not across the batch or sequence) by subtracting the mean and dividing by the standard deviation, then applies a learned elementwise scale `gamma` and shift `beta`:

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

**RMSNorm** (Root Mean Square Normalization), used in Llama, Mistral, and most recent open models, simplifies this by dropping the mean-centering step entirely and only rescaling by the root-mean-square of the activations:

```
RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
```

Note there is no `beta` shift and no mean subtraction — RMSNorm only re-scales the vector's magnitude, it doesn't re-center it. The authors of RMSNorm (Zhang & Sennrich) argued and empirically showed that the re-centering (mean subtraction) step in LayerNorm contributes little to its benefit; what actually matters for training stability is controlling the *scale* of activations, not their mean. Because RMSNorm skips computing the mean and the corresponding gradient terms, it's computationally cheaper — fewer reduction operations and fewer parameters (no `beta` vector) — while empirically matching LayerNorm's quality in transformer training. This is a very concrete, checkable fact for interviews: Llama's technical reports explicitly cite using RMSNorm instead of LayerNorm for this efficiency reason, at negligible quality cost.

```python
import numpy as np

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms
```

## Pre-Norm vs Post-Norm

The original transformer applied normalization *after* each sublayer and after the residual addition — "post-norm": `x_out = LayerNorm(x_in + Sublayer(x_in))`. Almost every modern LLM instead uses "pre-norm": normalization is applied to the input *before* it enters the sublayer, and the sublayer's raw (unnormalized) output is what gets added to the residual stream: `x_out = x_in + Sublayer(LayerNorm(x_in))`.

The reason for this shift is training stability at depth. In the post-norm arrangement, the residual stream itself gets normalized at every layer, which means the clean identity-gradient path described above gets passed through a normalization operation's own (nontrivial) Jacobian at every single layer — diluting exactly the benefit residual connections were meant to provide, and this compounds badly as depth increases. Post-norm transformers are empirically much harder to train past a moderate depth without careful learning-rate warmup schedules, and they are prone to instability or divergence early in training.

In the pre-norm arrangement, the residual stream itself is never directly normalized — normalization only happens on a side branch that feeds into the sublayer, while the main residual path remains a clean, unimpeded sum of the original input plus every sublayer's contribution. This preserves the identity-gradient guarantee essentially intact all the way through the depth of the network, which is why pre-norm transformers can be trained substantially deeper and with less sensitive warmup schedules, and is exactly why GPT-2 onward, and essentially every subsequent large model (GPT-3, Llama, Mistral, PaLM, Claude, DeepSeek), adopted pre-norm as standard. The one downside worth knowing is that pre-norm can lead to a gradual increase in the residual stream's activation magnitude across depth (since nothing ever rescales the accumulated sum on the main path), which is one motivation behind extra tricks in some very deep or very large models, such as an additional final normalization before the output head (which all pre-norm models include), or, in a few architectures, extra normalization layers placed elsewhere (for example, some models add a normalization directly after the attention/FFN output before the residual add, a "sandwich norm" variant, specifically to control this growth at extreme scale).

## Why Decoder-Only Won Over Encoder-Decoder for Generative LLMs

The original transformer, and T5 after it, used an encoder-decoder architecture: a bidirectional encoder processes the full input with unrestricted (non-causal) self-attention, producing contextualized representations, and a separate decoder generates output tokens autoregressively while cross-attending to the encoder's representations. This is a natural fit for tasks with a clear input/output split, like translation. Yet GPT, Llama, Mistral, Claude, and essentially every frontier general-purpose LLM today are decoder-only: a single stack of causally-masked self-attention blocks that treats "input" and "output" as one continuous token sequence, with no separate encoder and no cross-attention. Several distinct arguments, architectural and empirical, explain why this became the dominant choice for large-scale generative models.

The first is **simplicity and uniformity**. A decoder-only model has one stack, one attention pattern (causal self-attention), and one training objective (next-token prediction) applied uniformly across every token in every training example. There's no architectural asymmetry between "the part that reads" and "the part that writes," no cross-attention mechanism to design and tune, and no need to decide, per task, how to split a problem into an "encoder side" and a "decoder side." This uniformity turns out to matter enormously at scale, because it means the exact same architecture and the exact same objective can be pointed at essentially unlimited raw text with no task-specific formatting, whereas encoder-decoder pretraining objectives (like T5's span-corruption) require deciding what counts as "masked span to predict" versus "context," which is a less natural fit for arbitrary, open-ended free text and generation-heavy downstream use.

The second argument is about **transfer and multi-task uniformity**. Because a decoder-only model already treats every task as "predict the next token given everything so far," any task — translation, summarization, question answering, code completion, chat — can be cast as plain text continuation simply by choosing how you format the prompt, with no architecture-level change required. This is precisely what makes prompting and in-context learning natural: the model was never trained with a rigid input/output boundary, so it has no architectural bias against, say, mixing instructions and content together, or handling multi-turn dialogue as one long growing sequence. Encoder-decoder models, by contrast, generally need the input to be clearly delineated from the target and processed through a structurally different pathway, which is a worse fit for open-ended, conversational, many-shot, or few-shot prompting scenarios that dominate real LLM usage today.

The third, and most cited in the research literature, is the empirical scaling and **in-context learning behavior** that emerged specifically from GPT-style decoder-only pretraining at scale. The GPT-3 paper demonstrated that a sufficiently large decoder-only model, trained purely on next-token prediction with no task-specific fine-tuning, could perform few-shot and even zero-shot task learning purely through prompting — a capability that was not what encoder-decoder models of similar size were primarily optimized to exhibit, since their pretraining and typical fine-tuning setup were more oriented around supervised sequence-to-sequence transfer per task (T5's own paper frames its objective explicitly around transfer learning via fine-tuning, not zero-shot prompting). Later analysis (including scaling-law-style comparisons across architectures) found that the causal, uniform, next-token objective of decoder-only models scales its downstream in-context and zero/few-shot task performance especially well as parameter count and data grow, which lines up with why essentially the entire frontier-model research community converged on decoder-only as parameter counts moved into the tens and hundreds of billions.

Finally there's a **data- and compute-efficiency** argument that is more subtle but shows up in scaling-law comparisons: because every single token in a decoder-only training sequence contributes a next-token-prediction training signal (the loss is computed at every position, not just at a masked subset), decoder-only pretraining extracts a training gradient from a larger fraction of the tokens it processes per pass than encoder-style masked or span-corruption objectives, which by construction only compute a loss on the corrupted/masked portion of each example. At the trillion-token pretraining scale used by modern LLMs, that difference in "loss-bearing tokens per FLOP" compounds into a real advantage for the decoder-only, full-sequence, causal-LM objective, which is one more concrete reason (beyond the behavioral in-context-learning argument) that decoder-only became the default recipe once compute and data budgets grew large enough for the difference to dominate other considerations. It's worth noting encoder-decoder and encoder-only models are not "wrong" or obsolete for every purpose — T5-style models and BERT-style encoders remain strong choices for certain classification, retrieval/embedding, and structured seq2seq tasks — but for the specific goal of building a single, general-purpose, promptable, in-context-learning generative model at the largest scales, the field converged decisively on decoder-only.

## Putting the Full Block Together

Combining everything above, a single modern (Llama-style) decoder-only transformer block looks like this in pseudocode:

```python
def transformer_block(x, params):
    # Pre-norm attention sublayer with residual connection
    normed = rms_norm(x, params.attn_norm_gamma)
    attn_out = multi_head_attention(
        normed, params.num_heads,
        params.W_Q, params.W_K, params.W_V, params.W_O,
        causal_mask=True,
    )
    x = x + attn_out

    # Pre-norm FFN sublayer with residual connection
    normed = rms_norm(x, params.ffn_norm_gamma)
    ffn_out = swiglu_ffn(normed, params.W_gate, params.W_up, params.W_down)
    x = x + ffn_out

    return x

def decoder_only_lm(token_ids, params):
    x = params.embedding[token_ids]           # (seq_len, d_model)
    x = x + params.positional_info(token_ids)  # RoPE is applied inside attention instead, in practice
    for block_params in params.blocks:
        x = transformer_block(x, block_params)
    x = rms_norm(x, params.final_norm_gamma)
    logits = x @ params.output_projection      # (seq_len, vocab_size)
    return logits
```

Every architectural choice discussed in this document — the `1/sqrt(d_k)` scaling, splitting attention into heads, gating the FFN with SwiGLU, using RMSNorm instead of LayerNorm, and placing normalization before rather than after each sublayer — exists because it measurably improves training stability, compute efficiency, or downstream quality at the scale modern LLMs are trained at. None of these choices were obvious in advance; they are the product of years of ablation studies across many labs, which is exactly why being able to explain the *reasoning* behind each one, not just name it, is what separates a strong answer from a memorized one in an interview setting.
