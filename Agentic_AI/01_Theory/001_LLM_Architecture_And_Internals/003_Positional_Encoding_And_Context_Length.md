# Positional Encoding and Context Length

## Why Transformers Need Explicit Positional Information

Self-attention, as described in the architecture document in this folder, computes its output as a weighted sum over value vectors, where the weights come from query-key dot products. Look closely at that operation and notice what it does *not* depend on: nowhere in `softmax(Q @ K^T / sqrt(d_k)) @ V` does the actual position of a token in the sequence appear. If you took an input sequence and permuted the order of its tokens, and permuted the corresponding rows of `Q`, `K`, and `V` the same way, the set of attention outputs produced would just be the same set of values, permuted identically — attention is a permutation-equivariant (and, ignoring the causal mask, permutation-invariant) operation over the set of input tokens. This is a direct mathematical consequence of the fact that attention treats its input as a set (technically a bag of key-value pairs looked up by content-based queries), not a sequence.

This is a real problem, because language is not a bag of words — "the dog bit the man" and "the man bit the dog" contain the exact same set of tokens but mean opposite things, and a model with no notion of token order literally cannot distinguish them (in a non-causal setting at least; the causal mask itself does inject some asymmetry, since it restricts which tokens can see which, but it does not tell the model *where* in the sequence a token sits, only a partial "before/after" structure). Every scheme discussed in this document is a different answer to the same question: how do we inject information about token position into a mechanism that is otherwise blind to it?

There are two broad families of answer. **Absolute** positional schemes give each position index (0, 1, 2, ...) its own distinct representation, added or otherwise combined with the token's content representation, so the model learns to interpret "this content, at this position" as a joint signal. **Relative** positional schemes instead directly inject information about the *distance* between a query position and a key position into the attention computation itself, without ever representing an absolute position 0, 1, 2 as such. This absolute-vs-relative distinction turns out to matter enormously for a model's ability to generalize to sequence lengths longer than it was trained on, which is the throughline connecting the historical schemes (sinusoidal, learned absolute) to the schemes that dominate modern LLMs (RoPE, ALiBi) and to the context-extension techniques used to stretch a pretrained model's usable context well past its original training length.

## Original Sinusoidal Absolute Positional Encoding

The original "Attention is All You Need" transformer added a fixed (non-learned) positional encoding vector to each token's embedding before the first layer. For position `pos` and embedding dimension index `i` (out of `d_model` total dimensions), the encoding is defined using sine and cosine functions at geometrically varying frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Each pair of dimensions `(2i, 2i+1)` oscillates at its own frequency, with frequencies spanning a wide geometric range from very fast-oscillating (low `i`) to very slow-oscillating (high `i`, close to a constant across realistic sequence lengths). The reason sinusoids specifically were chosen, rather than some other fixed function, comes down to a convenient mathematical property: for any fixed offset `k`, `PE(pos + k)` can be expressed as a linear function of `PE(pos)` (a rotation, in fact, since sine and cosine of a sum expand into linear combinations of sine and cosine of the original angle). The paper's authors argued this property would make it easy for the model to learn to attend by *relative* position, since a fixed linear transformation could in principle convert an absolute positional signal at one position into the corresponding signal at a fixed relative offset. This same rotation-based reasoning is, not coincidentally, the direct conceptual ancestor of RoPE, discussed in depth below.

```python
import numpy as np

def sinusoidal_positional_encoding(seq_len, d_model, base=10000):
    positions = np.arange(seq_len)[:, None]              # (seq_len, 1)
    dim_idx = np.arange(d_model)[None, :]                 # (1, d_model)
    angle_rates = 1.0 / (base ** ((2 * (dim_idx // 2)) / d_model))
    angles = positions * angle_rates                      # (seq_len, d_model)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

pe = sinusoidal_positional_encoding(seq_len=50, d_model=64)
```

In practice, this fixed additive scheme has a real extrapolation weakness: although the encoding function is mathematically defined for arbitrarily large `pos`, the model only ever sees a bounded range of positions during training, and it learns to interpret the additive positional signal jointly with content only within that observed range. At positions well beyond anything seen in training, the raw sinusoidal values are technically defined and bounded, but the model's learned weights for interpreting them were never exposed to that regime, so behavior degrades — empirically, transformers with sinusoidal absolute encodings do not extrapolate gracefully to sequences much longer than their training length, even though the encoding function itself doesn't hit a hard wall.

## Learned Absolute Positional Embeddings

GPT-2 and BERT took a simpler approach: instead of a fixed mathematical function, just learn a positional embedding table directly, exactly analogous to the token embedding table — a matrix of shape `(max_position, d_model)`, where row `p` is a learned vector added to every token that appears at position `p`, trained end-to-end via ordinary backpropagation just like every other parameter in the model.

This is simpler to implement and, given enough training data at the positions actually used, can fit the data at least as well as a fixed sinusoidal scheme, since it isn't constrained to any particular functional form. But it introduces a hard architectural ceiling: `max_position` is a fixed, finite hyperparameter chosen before training, and the model has no mechanism whatsoever to produce a positional representation for any position beyond that number — there simply is no row `max_position + 1` in the embedding table. This is why GPT-2's original context window (1024 tokens) and BERT's (512 tokens) were hard limits baked directly into the architecture, not just conventions: feeding a longer sequence would require indexing into rows of the positional embedding table that were never allocated or trained, and doing so (for instance, by naively extending the table with random or interpolated rows and continuing to serve traffic without further training) produces badly degraded output, because those extended positions were never seen during training at all. Extending a model with learned absolute positional embeddings to a longer context therefore strictly requires additional training exposure at the new, previously-unseen position indices — you cannot simply "ask" the model to generalize past its trained ceiling the way a relative positional scheme can, at least somewhat, be asked to.

## Rotary Position Embedding (RoPE)

### The core intuition: encoding position as rotation

RoPE, introduced in the RoFormer paper and now used by Llama (all generations), Mistral, DeepSeek, PaLM, and most other current open frontier models, takes a fundamentally different approach from both prior schemes: instead of adding a positional vector to the token embedding before attention runs, RoPE **rotates** the query and key vectors by an angle that depends on their position, and it does this rotation as part of computing attention itself, not as a preprocessing step on the embeddings.

The key mathematical trick is to treat each consecutive pair of dimensions in a query or key vector as the `(x, y)` coordinates of a 2D point, and to rotate that point by an angle `theta * pos`, where `pos` is the token's position in the sequence and `theta` is a frequency associated with that particular pair of dimensions (different dimension-pairs use different, geometrically-spaced frequencies, directly analogous to the different frequencies used across dimension-pairs in the original sinusoidal scheme). A 2D rotation by angle `phi` is a simple, well-known linear operation:

```
[x']   [cos(phi)  -sin(phi)] [x]
[y'] = [sin(phi)   cos(phi)] [y]
```

RoPE applies this rotation independently to every consecutive pair of dimensions within a query or key vector, with each pair's rotation angle scaled by that pair's assigned frequency times the token's position. The reason this achieves relative positional encoding — and this is the crucial mathematical payoff, worth being able to derive in an interview — is a property of rotations: rotating two vectors and then taking their dot product gives the same result as rotating just one of them by the *difference* in the two rotation angles. Concretely, if `q` is rotated by angle `theta * m` (query at position `m`) and `k` is rotated by angle `theta * n` (key at position `n`), then the dot product `(R(theta*m) @ q) . (R(theta*n) @ k)` depends on `m` and `n` *only through their difference* `m - n`, because rotation matrices satisfy `R(a)^T @ R(b) = R(b - a)`. This means the attention score between two tokens becomes a function of their relative distance `m - n`, even though the rotation was applied to each vector independently using only its own absolute position — no pairwise, quadratic-in-sequence-length computation is needed to inject relative position; it falls out for free from the geometry of the dot product after independent per-token rotation.

### Why RoPE is applied to Q and K but not V

RoPE's entire mechanism of action is through the query-key dot product that produces attention scores — it works precisely because rotating both `q` and `k` causes their dot product to depend on relative position. The value vectors `V` are not involved in that dot product at all; they are only ever combined via a weighted sum (the weights being the already-computed attention probabilities) to produce the attention output. There is no dot product involving `V` for a rotation to usefully cancel or combine within, so rotating `V` would serve no purpose for encoding relative position — it would only reorient the content vectors being aggregated with no corresponding benefit, and would need to be un-rotated somewhere to make the aggregated output meaningful in the original content space. RoPE is therefore applied only to `Q` and `K`, immediately before the dot-product step of attention, leaving `V` (and the final weighted-sum output) untouched by any positional transformation.

### A from-scratch RoPE implementation

```python
import numpy as np

def rope_frequencies(dim, base=10000):
    """One frequency per dimension-pair; dim must be even."""
    i = np.arange(0, dim, 2)
    return 1.0 / (base ** (i / dim))          # shape (dim/2,)

def apply_rope(x, positions, base=10000):
    """
    x: (seq_len, dim) -- a Q or K matrix for one head, dim must be even
    positions: (seq_len,) -- integer position index per token
    """
    seq_len, dim = x.shape
    freqs = rope_frequencies(dim, base)                  # (dim/2,)
    angles = positions[:, None] * freqs[None, :]          # (seq_len, dim/2)

    cos = np.cos(angles)
    sin = np.sin(angles)

    x1 = x[:, 0::2]     # "x" components of each 2D pair
    x2 = x[:, 1::2]     # "y" components of each 2D pair

    rotated = np.empty_like(x)
    rotated[:, 0::2] = x1 * cos - x2 * sin
    rotated[:, 1::2] = x1 * sin + x2 * cos
    return rotated

# demonstrate the core relative-position property:
# dot product after rotation depends only on (pos_q - pos_k)
dim = 8
np.random.seed(0)
q = np.random.randn(1, dim)
k = np.random.randn(1, dim)

for (pos_q, pos_k) in [(5, 3), (10, 8), (100, 98)]:   # all have the same offset: 2
    q_rot = apply_rope(q, np.array([pos_q]))
    k_rot = apply_rope(k, np.array([pos_k]))
    score = (q_rot @ k_rot.T).item()
    print(f"pos_q={pos_q}, pos_k={pos_k}, offset={pos_q - pos_k}, score={score:.4f}")
```

Running this, the three printed scores come out numerically identical (up to floating point precision) despite the absolute positions being completely different in each pair — 5&3, 10&8, 100&98 — because RoPE's dot product depends only on the offset (2, in every case here), which is exactly the relative-position property described above, verified numerically rather than just asserted.

### Why RoPE became dominant

Several properties combine to explain why RoPE displaced both sinusoidal and learned absolute embeddings as the default in essentially every major open model released since 2023 (Llama, Mistral, DeepSeek, Qwen, and others). First, it directly encodes relative position through the attention mechanism itself, which aligns well with the empirical and intuitive observation that what should matter for predicting a token is largely how far away context is, not its absolute index in the document — "the previous word" should mean the same thing whether it's word 5 or word 5,000. Second, unlike learned absolute embeddings, RoPE has no hard-coded maximum position baked into a fixed-size lookup table — the rotation angle formula is defined for any position index, so a RoPE model can, at least mechanically, be run on sequences longer than it was trained on, even though (as covered below) quality still degrades past the trained range without further intervention, because the *specific* rotation angles at very large positions were never seen during training even though the formula itself extends there. Third, RoPE adds no extra parameters and no extra embedding table — it's a deterministic function applied at attention time, which keeps the parameter count and memory footprint identical to not having positional embeddings at all, a meaningful efficiency advantage at the scale of billions of parameters. Finally, and most practically, essentially all of the context-length-extension research and tooling described in the next section (position interpolation, NTK-aware scaling, YaRN) exists specifically because RoPE's clean, closed-form rotation-angle formula is something you can deliberately *rescale* after the fact to change a model's effective context behavior without retraining from scratch — a lever that simply doesn't exist for learned absolute embeddings, whose behavior at new positions is undefined rather than merely "defined but out of the training distribution."

## ALiBi: Attention with Linear Biases

ALiBi takes yet another approach, arguably the most minimal of all: it doesn't modify the query or key vectors at all, and it doesn't add anything to the token embeddings. Instead, it adds a fixed, non-learned penalty directly to the raw attention scores, proportional to the distance between the query and key positions, before the softmax:

```
score(i, j) = (q_i . k_j) / sqrt(d_k)  -  m * (i - j)          for j <= i (causal)
```

where `i - j` is the distance between the query position `i` and key position `j`, and `m` is a fixed, head-specific slope (different attention heads get different, geometrically-spaced slope values, so some heads impose a steep penalty that effectively restricts them to very local context, while others impose a shallow penalty that lets them attend broadly). Because this penalty grows linearly with distance and is subtracted before the softmax, distant tokens are always penalized relative to nearby ones, biasing every head toward attending more to recent context, with the degree of that bias fixed per head rather than learned.

The reason this achieves strong length extrapolation is that the *mechanism* generating the bias is a simple, fixed linear function of relative distance that is well-defined and behaves exactly the same way for any distance, including distances far larger than anything seen in training — there is no learned table, no rotation-angle range implicitly tied to training-time position magnitudes, nothing that was fit to a particular numeric range during training. The ALiBi paper demonstrated that models trained with this scheme at a given sequence length could be evaluated at substantially longer sequence lengths at inference time with much smaller quality degradation than sinusoidal or learned absolute schemes showed under the same test. The trade-off relative to RoPE is that ALiBi's positional signal is a comparatively blunt instrument — a fixed, monotonic recency bias baked into the attention logits — rather than RoPE's richer, rotation-based encoding that in principle lets the model learn more nuanced position-dependent relationships through how Q and K project into the rotated space; in practice, RoPE has become the more widely adopted default among current frontier open models, with ALiBi more associated with earlier long-context-focused efforts like BLOOM and MPT, though both remain valid, actively-referenced design points in the literature.

## Sliding Window Attention

Sliding window attention is not a positional-encoding scheme at all, but it's closely related in purpose and frequently discussed alongside these techniques, and Mistral popularized it prominently in a production frontier model. The idea is architectural rather than mathematical: instead of allowing every token to attend to every prior token in the full context (standard causal attention, cost quadratic in sequence length), each token is restricted to attend only to the most recent `W` tokens (a fixed window size), regardless of how long the overall sequence is.

This directly bounds the compute and memory cost of attention to `O(seq_len * W)` rather than `O(seq_len^2)`, which matters enormously for long-context serving, since both the FLOPs spent computing attention scores and, critically, the size of the KV cache that must be kept resident in GPU memory during generation, no longer grow without bound as the conversation or document gets longer — beyond `W` tokens back, older key/value pairs can in principle be evicted entirely, capping memory use at a constant multiple of `W` rather than growing linearly with total sequence length. The trade-off is the obvious one: a token genuinely cannot directly attend to information more than `W` positions back within a single layer. Mistral's approach mitigates this partially through **stacking across layers** — because each layer's window lets information propagate one window's worth of distance further, information from `k` layers back can, in principle, reach up to roughly `k * W` tokens away by the time it has passed through `k` stacked windowed-attention layers, similar in spirit to how stacking convolutional layers with small kernels expands a CNN's effective receptive field over depth. Whether this stacked effective range is actually exploited as well as full unrestricted attention in practice is an empirical question that depends on the task, but the compute and memory savings are unconditional and are the primary reason for adopting the scheme.

## Extending Context Length After Pretraining

A model is pretrained at some fixed maximum sequence length (its "native" context window), and a recurring, high-value production need is to extend a model's *usable* context well beyond that native length without paying for full pretraining again. The naive approach — simply fine-tuning the existing model on longer documents while feeding position indices past what it saw during pretraining — tends to work poorly on its own, and understanding precisely *why* it fails is what motivates each of the more careful techniques below.

The core problem is that with RoPE, every dimension-pair's rotation angle is `position * frequency`, and during pretraining the model's weights (particularly in attention) become implicitly tuned to the specific *range* of angle values it actually encountered — for high-frequency dimension-pairs, that range wraps around many full rotations even at modest positions, while for low-frequency pairs, the angle stays small and slowly-varying across the entire trained context. If you fine-tune naively at, say, 4x the original context length by just extending position indices linearly, the low-frequency dimension-pairs (which barely moved across the *entire* original training range) suddenly need to represent a much wider range of angle values than the model ever saw an attention pattern trained against, and the model has comparatively little capacity or training signal to adapt to that unfamiliar range, especially without a large volume of long-context fine-tuning data (which is also generally scarcer than short-context data). This produces degraded quality, particularly on tasks that require precise use of information toward the far end of the extended window.

### Position interpolation (PI)

The simplest fix, called position interpolation, keeps every rotation-angle *formula* the same but rescales the position indices themselves before feeding them in: instead of using raw positions `0, 1, 2, ..., L_new - 1` for a new target length `L_new`, it uses `0, L_orig/L_new, 2*L_orig/L_new, ...`, i.e., it linearly compresses the new, longer position range down to fit inside the original range the model was trained on. This guarantees that every rotation angle the model ever computes at inference time falls within the range it saw during pretraining, which avoids the "unfamiliar angle range" problem directly, and empirically this requires comparatively little additional fine-tuning to work reasonably well, since the model is never asked to interpret an out-of-distribution rotation value.

The downside is that this rescaling is applied uniformly across all dimension-pairs, including the high-frequency ones, which means positions that used to be one full "period" apart under the original scale now get compressed much closer together in angle space than before. Because high-frequency dimension-pairs are precisely the ones responsible for letting the model distinguish *nearby* tokens from each other with fine resolution (a small change in position produces a large change in a high-frequency angle), uniformly compressing them reduces the model's resolution for distinguishing nearby tokens, which can hurt performance on tasks sensitive to fine-grained local position, even while it helps overall length extrapolation.

### NTK-aware scaling

NTK-aware scaling (the name references a rough analogy to Neural Tangent Kernel theory about how networks preferentially learn different frequency components) addresses precisely that weakness by changing strategy: rather than uniformly compressing every position index, it modifies the RoPE **base** parameter (the `10000` constant in the frequency formula) so that only the *low-frequency* dimension-pairs get stretched to accommodate the longer context, while the *high-frequency* dimension-pairs are left close to their original scale. Since the base constant controls the geometric spacing of frequencies across dimension-pairs, increasing it shifts the whole frequency spectrum in a way that disproportionately affects the slow-varying, long-range-sensitive dimensions while leaving the fast-varying, local-resolution dimensions comparatively undisturbed. This directly targets the specific failure mode of naive extension (the low-frequency pairs running out of trained-on angle range) without paying position interpolation's cost of blurring fine local resolution across the board, since the dimensions responsible for that local resolution are largely left alone.

### YaRN

YaRN ("Yet another RoPE extensioN") combines and refines these ideas into a more carefully engineered scheme, and is worth knowing as the current state-of-the-art reference point for RoPE extension. It applies what's called NTK-by-parts interpolation: rather than treating all dimension-pairs with one single rule (either uniformly interpolating everything, as in plain PI, or one global base adjustment, as in basic NTK-aware scaling), it explicitly partitions dimension-pairs by their wavelength relative to the original and new context lengths, and applies interpolation only to the dimension-pairs whose wavelength is long enough that they would otherwise be pushed outside the trained range, while leaving short-wavelength (local-resolution) dimension-pairs essentially untouched — a more surgical, per-dimension version of the same underlying insight NTK-aware scaling captured with a single global adjustment.

YaRN's second, distinct contribution is **attention temperature scaling**: it introduces a fixed scaling factor applied to the attention logits (or equivalently, a temperature adjustment inside the softmax) that compensates for a subtle side effect of extending the effective context — as the attended-over context grows much longer, the entropy of the attention distribution over a larger set of candidate keys tends to shift, and without compensation, the sharpness/softness of attention (how concentrated its probability mass becomes) is no longer well-calibrated to what the model's weights were trained to expect. Adding a small, empirically-tuned temperature correction to the attention scores counteracts this shift, and the YaRN authors reported this contributes a further meaningful improvement in extended-context quality on top of the NTK-by-parts interpolation alone. Together, these two adjustments allow models to be extended to substantially longer context windows (the YaRN paper demonstrated extending Llama-family models many times past their original trained length) using a comparatively small amount of additional fine-tuning steps at the target length, rather than requiring long-context data volumes and compute anywhere close to full pretraining scale.

### Why this progression matters as a whole

The throughline across position interpolation, NTK-aware scaling, and YaRN is that each technique is a progressively more precise diagnosis and fix of the exact same underlying issue: naive context extension pushes RoPE's low-frequency rotation angles outside the range the model's attention weights were actually trained to interpret, and different dimension-pairs are affected differently depending on their frequency. Plain interpolation fixes the range problem for every dimension uniformly, at the cost of losing local resolution; NTK-aware scaling targets only the frequencies that actually need it; and YaRN adds both a more surgical per-dimension version of that targeting and a separate correction for how attention's overall sharpness needs to be recalibrated once the number of keys being attended over has grown far past the original training regime. Being able to explain this progression — not just name the three techniques — is exactly the kind of depth that distinguishes a strong answer from a memorized list in a systems-level LLM interview, and it also explains, concretely, why a production model's advertised context length (e.g., a model trained natively at 8K tokens later being offered commercially at 128K) is very often the result of one of these post-hoc extension techniques applied deliberately, rather than the model having been pretrained from scratch at that full advertised length.
