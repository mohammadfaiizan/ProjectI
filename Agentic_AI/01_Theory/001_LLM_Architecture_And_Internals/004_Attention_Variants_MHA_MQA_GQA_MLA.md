# Attention Variants: MHA, MQA, GQA, and MLA (and why FlashAttention matters)

## Why this topic exists: the KV cache problem

Every modern transformer decoder generates text one token at a time, and it does so autoregressively: to produce token `t+1`, the model needs the hidden representations of every token from `1` to `t`. The naive way to do this would be to re-run the entire forward pass over the whole prefix on every single generation step, but that throws away an enormous amount of redundant computation, because the key (K) and value (V) projections for tokens `1..t` never change once they have been computed. Only the query (Q) for the newest token is new. This observation is the entire reason the **KV cache** exists: instead of recomputing K and V for the whole prefix at every step, we compute them once per token as it is first processed, and store them in GPU memory so that later steps can just look them up.

This turns autoregressive decoding from an O(n²) per-step cost into something closer to O(n) per step (attention over a growing cache), which is what makes chatbots feel responsive at all. But it comes at a direct cost: the KV cache is a piece of state that has to live in GPU memory for the entire lifetime of a generation, for every sequence being served concurrently. And critically, **this is a per-request, per-token memory cost that grows linearly with context length and linearly with batch size**, whereas the model's weights are a fixed, one-time cost paid regardless of how many requests you serve. At long context lengths and with many concurrent requests (the exact regime that production LLM serving lives in), the KV cache — not the weights — frequently becomes the dominant consumer of GPU memory, and the primary constraint on how many users you can serve at once and how long a context you can support. Almost every attention variant discussed in this file (MQA, GQA, MLA) exists specifically to attack this one bottleneck. This is worth internalizing before looking at any formula: **these architectural choices are not really about modeling quality in the first place — they exist because of an inference-serving economics problem**, and the quality trade-offs are the price paid to solve it.

### The memory arithmetic

The size of the KV cache for a single sequence, in a standard multi-head attention layer, is given by:

```
KV cache size (bytes) = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_element
```

The leading `2` accounts for the fact that we must cache both K and V (not just one of them). `num_kv_heads` is the number of distinct key/value head "streams" the model computes — in plain multi-head attention this equals the number of query heads, but as you will see below, that is exactly the number the later variants (MQA, GQA, MLA) shrink. `head_dim` is the per-head dimensionality (typically `hidden_size / num_heads`), `seq_len` is the number of tokens currently in the cache (prompt + generated so far), `batch_size` is the number of sequences being decoded concurrently, and `bytes_per_element` is 2 for fp16/bf16 or 4 for fp32 (or as low as 1 for int8-quantized KV caches, which many serving stacks now use).

Let's ground this with a concrete worked example on a 7B-class dense model with a Llama-style configuration: 32 transformer layers, 32 attention heads, head dimension 128 (so hidden size 4096), running in bf16 (2 bytes/element), with a single sequence at 4096 tokens of context.

```python
num_layers = 32
num_kv_heads = 32       # plain MHA: num_kv_heads == num_query_heads
head_dim = 128
seq_len = 4096
batch_size = 1
bytes_per_element = 2   # bf16

kv_cache_bytes = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_element
kv_cache_gb = kv_cache_bytes / (1024**3)
print(f"{kv_cache_gb:.2f} GB")   # -> 2.00 GB, for ONE sequence at 4K context
```

Two gigabytes for a single 4K-token sequence, on a model whose weights themselves only take about 14 GB in bf16. Now scale that to something realistic: a serving batch of 32 concurrent sequences at that same 4K context multiplies the cache to 64 GB — more than four times the size of the model weights themselves, and easily enough to blow through the memory budget of a single 80 GB GPU once you also need room for the weights, activations, and any framework overhead. This is precisely why long-context serving and high-throughput batched serving are memory problems as much as they are compute problems, and it is why every major model family released since 2023 has adopted one of the KV-cache-reduction techniques described below.

## Multi-Head Attention (MHA): the baseline

Standard multi-head attention, as introduced in "Attention Is All You Need" and used in GPT-2/GPT-3 and the original Llama, computes a **separate** set of K and V projections for every query head. If there are `h` heads, there are `h` independent K heads and `h` independent V heads, each of dimension `head_dim = d_model / h`. Every query head attends only to its own corresponding K/V head — there is no sharing at all.

```python
import torch
import torch.nn.functional as F

def mha_forward(x, Wq, Wk, Wv, Wo, num_heads, head_dim):
    B, T, D = x.shape
    q = (x @ Wq).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    k = (x @ Wk).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    v = (x @ Wv).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)

    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                    # (B, h, T, d)
    out = out.transpose(1, 2).reshape(B, T, num_heads * head_dim)
    return out @ Wo
```

The KV cache for MHA stores a full `(num_heads, head_dim)` pair of K and V vectors per token per layer, which is exactly the `num_kv_heads = num_heads` case in the formula above. This gives MHA the best possible representational flexibility — every head can specialize and attend to genuinely different aspects of the context — but it also gives it the worst possible KV-cache footprint. This is the baseline every other variant below tries to beat.

## Multi-Query Attention (MQA): one KV head for everyone

MQA, proposed by Noam Shazeer in 2019, makes a deliberately aggressive simplification: keep the full number of **query** heads (so the model retains its ability to compute many different attention patterns over the same content), but collapse all the **key and value** heads down to a single shared K head and a single shared V head. Every query head still has its own learned projection and therefore its own attention pattern, but all of them read from and write into the exact same K/V representation.

```python
def mqa_forward(x, Wq, Wk, Wv, Wo, num_heads, head_dim):
    B, T, D = x.shape
    q = (x @ Wq).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    k = (x @ Wk).view(B, T, 1, head_dim).transpose(1, 2)          # (B, 1, T, d)
    v = (x @ Wv).view(B, T, 1, head_dim).transpose(1, 2)          # (B, 1, T, d)

    # broadcast the single KV head across all query heads
    k = k.expand(B, num_heads, T, head_dim)
    v = v.expand(B, num_heads, T, head_dim)

    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, T, num_heads * head_dim)
    return out @ Wo
```

Notice what happened to the cache: `num_kv_heads` in the memory formula drops from `num_heads` (e.g. 32) all the way down to `1`. That is a reduction in KV-cache size proportional to the number of heads — for a 32-head model, roughly a 32x smaller cache. This is a huge win for serving throughput and long-context capability, and it was the technique behind Google's PaLM and the original Falcon models.

The cost is quality risk. Forcing every query head to read from a single shared K/V representation removes a real degree of freedom from the model: heads can no longer specialize in *what content* they attend to, only in *how* they weight that shared content via their own query projection. In practice this can measurably hurt quality, especially on tasks that benefit from diverse attention patterns, and training stability can also suffer because the single KV head becomes a much more contended, higher-stakes bottleneck in the computational graph. MQA is a real, usable technique, but its aggressiveness is exactly why the field mostly settled on the milder compromise described next.

## Grouped-Query Attention (GQA): the interpolation that won

GQA, introduced in the 2023 paper "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al.), sits deliberately between MHA and MQA. Instead of either "every query head gets its own KV head" (MHA) or "every query head shares one KV head" (MQA), GQA partitions the query heads into `g` groups, and every head within a group shares a single K/V head. If `g` equals the number of query heads, GQA degenerates exactly to MHA; if `g = 1`, it degenerates exactly to MQA. This makes GQA a tunable dial rather than a fixed architectural choice, and it is why it is usually described as an interpolation rather than a separate third technique.

```python
def gqa_forward(x, Wq, Wk, Wv, Wo, num_heads, num_kv_heads, head_dim):
    B, T, D = x.shape
    group_size = num_heads // num_kv_heads   # query heads per KV head

    q = (x @ Wq).view(B, T, num_heads, head_dim).transpose(1, 2)        # (B, h, T, d)
    k = (x @ Wk).view(B, T, num_kv_heads, head_dim).transpose(1, 2)     # (B, h_kv, T, d)
    v = (x @ Wv).view(B, T, num_kv_heads, head_dim).transpose(1, 2)     # (B, h_kv, T, d)

    # repeat each KV head `group_size` times so it lines up with its query group
    k = k.repeat_interleave(group_size, dim=1)   # (B, h, T, d)
    v = v.repeat_interleave(group_size, dim=1)   # (B, h, T, d)

    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, T, num_heads * head_dim)
    return out @ Wo
```

Crucially, the `k`/`v` tensors that actually get **written into the KV cache** are the small `(B, num_kv_heads, T, head_dim)` tensors *before* the `repeat_interleave` — the expansion to match query-head count happens on the fly at attention-computation time and is never cached. This is the whole trick: the cache shrinks by a factor of `group_size = num_heads / num_kv_heads`, while the query side of the computation still has full head diversity.

GQA became the de facto industry default very quickly because empirically it captures almost all of MHA's quality while getting most of MQA's memory savings, and because the paper also showed you can convert an already-pretrained MHA checkpoint into a GQA one cheaply, by mean-pooling groups of existing KV heads together and then briefly fine-tuning ("uptraining"), rather than needing to pretrain from scratch. Concretely: Llama 2 70B (and Llama 3's entire family, including the smaller 8B model) uses GQA, Mistral 7B uses GQA (8 KV heads with 32 query heads, i.e. group size 4), and most other modern open-weight model families followed suit. Llama 2's smaller 7B/13B variants notably still used plain MHA, which is a useful reminder that the memory pressure from KV cache becomes proportionally more painful as models (and their intended context lengths and serving batch sizes) get larger — the incentive to adopt GQA scales with model size and target context length.

Choosing the number of groups (equivalently, the number of KV heads `num_kv_heads`) is a direct trade-off dial: fewer KV heads (larger groups) means smaller cache and cheaper serving but higher quality risk from over-sharing; more KV heads means the opposite. In practice, a common rule of thumb used by several model families is to pick something like `num_kv_heads = num_heads / 4` or `num_heads / 8` as a sweet spot — small enough to meaningfully shrink the cache (4-8x), but large enough that each KV head is still shared among a modest number of query heads rather than all of them. The right value is ultimately determined empirically via ablations that measure downstream task quality against cache-size/throughput targets, but the intuition to carry into an interview is: GQA lets you buy back most of the MQA memory win while giving up only a small, tunable slice of MHA's quality, and the "how much to give up" decision is exactly what the group count controls.

## Multi-head Latent Attention (MLA): DeepSeek's rethink

MLA, introduced by DeepSeek in the DeepSeek-V2 paper and refined in DeepSeek-V3, attacks the same problem from a genuinely different angle rather than just tuning the MHA-MQA dial further. GQA and MQA reduce the cache by reducing the *number* of distinct K/V head streams — they are still caching full-width K and V vectors, just fewer copies of them. MLA instead asks: what if, instead of caching K and V directly at all, we cache a much lower-dimensional **latent** vector that can be decompressed back into K and V (or, more precisely, folded into the attention computation) only when needed?

The mechanism works by learning a down-projection that compresses the per-token hidden state into a small latent vector `c` (dimension much smaller than `num_heads * head_dim`), and up-projection matrices that reconstruct the per-head K and V from that shared latent when attention needs to be computed. Only the compressed latent `c` is written to the KV cache — the full-size K and V are reconstructed on the fly at attention time and thrown away immediately after, exactly analogous to how GQA's repeated K/V never get cached. Because the up-projection matrices are learned per-head, MLA does **not** force different heads to read literally identical content the way MQA does; each head still gets its own reconstructed K/V, computed from a shared compressed representation, which is a much less destructive form of sharing than MQA's literal broadcast. DeepSeek-V2 also adds a decoupled rotary positional embedding pathway specifically to make this compression compatible with RoPE, since naively compressing K would otherwise break the relative-position math RoPE depends on.

```python
def mla_forward(x, W_dc, W_uk, W_uv, Wq, Wo, num_heads, head_dim, latent_dim):
    """
    Illustrative, simplified MLA forward pass (omits the RoPE decoupling
    DeepSeek-V2 uses in practice, to keep the core compression idea clear).
    """
    B, T, D = x.shape

    # Compress hidden state into a small latent vector -- THIS is what gets cached.
    c = x @ W_dc                       # (B, T, latent_dim), latent_dim << num_heads*head_dim

    # Reconstruct per-head K and V from the shared latent, on the fly.
    k = (c @ W_uk).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)
    v = (c @ W_uv).view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, h, T, d)

    q = (x @ Wq).view(B, T, num_heads, head_dim).transpose(1, 2)    # (B, h, T, d)

    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, T, num_heads * head_dim)
    return out @ Wo, c   # cache `c`, not k or v
```

The payoff is that MLA's cache size is governed by `latent_dim`, a number that is completely decoupled from `num_heads` — DeepSeek-V2 reports achieving a KV cache smaller than what even GQA with very few groups would give, while retaining quality closer to full MHA than GQA typically achieves, because the per-head reconstruction preserves much more of the representational diversity that MQA/GQA sacrifice. This is why MLA is often described as achieving "MHA-level quality at MQA-level (or better) cache cost" — it breaks what looked like a fundamental trade-off curve by changing what gets cached rather than just how many times it gets duplicated. This was a significant enough efficiency innovation that it is widely credited as one of the key architectural reasons DeepSeek-V2/V3 were able to serve very large, very long-context models at dramatically lower inference cost than comparably-sized dense or GQA-based competitors, directly feeding into DeepSeek's well-publicized low API pricing.

### Comparison

| Variant | KV heads cached | Relative cache size | Quality vs. MHA | Notable adopters |
|---|---|---|---|---|
| MHA | `num_heads` (all) | 1x (baseline, largest) | Reference/best | GPT-2/3, original Llama, Llama 2 7B/13B |
| MQA | 1 | ~`1/num_heads` | Noticeable risk of degradation | PaLM, Falcon |
| GQA | `g` (tunable groups) | ~`g/num_heads` | Close to MHA, small tunable gap | Llama 2 70B, Llama 3 (all sizes), Mistral 7B |
| MLA | N/A (caches a latent, not K/V heads) | Smaller than GQA in practice | Close to/better than GQA, near-MHA | DeepSeek-V2, DeepSeek-V3 |

## FlashAttention: making exact attention IO-aware

Everything above changes *what* gets cached and *how much of it*. FlashAttention is a different kind of optimization entirely: it does not change the KV cache size or the mathematical result of attention at all — it is an **exact** algorithm, not an approximation — but it changes *how* the attention computation is carried out on the GPU so that it runs dramatically faster and uses dramatically less memory to do so. It is complementary to everything discussed above and is used alongside MHA, MQA, GQA, or MLA, not instead of them.

The key insight behind FlashAttention (Dao et al., 2022) is that naive attention implementations are **memory-bandwidth-bound, not compute-bound**, on modern GPUs. A standard implementation computes the full `Q @ K^T` score matrix of shape `(seq_len, seq_len)`, writes that entire matrix out to the GPU's high-bandwidth memory (HBM), reads it back in to apply the softmax, writes the softmax result back to HBM, reads it back in again to multiply by `V`, and so on. Each of those read/write round-trips to HBM is far slower than the GPU's actual arithmetic throughput — GPUs have become so fast at raw matrix multiplication that the bottleneck in practice is not "how many FLOPs can we do" but "how fast can we move data between HBM and the much smaller, much faster on-chip SRAM." Materializing the full `N x N` attention matrix (which also happens to be the very thing that makes attention's memory cost scale quadratically with sequence length) is precisely the expensive, avoidable HBM traffic that FlashAttention targets.

FlashAttention avoids ever writing that full `N x N` matrix to HBM at all. It does this via **tiling**: Q, K, and V are split into blocks small enough to fit in the GPU's on-chip SRAM, and the algorithm iterates over blocks of K and V, computing attention output incrementally for each block of Q, keeping all the intermediate score and softmax computation for the current block resident in fast on-chip memory rather than ever writing it back out to HBM. The one wrinkle this creates is that softmax needs the full row of scores (across every key) to normalize correctly, but with tiling you only ever see one block of keys at a time. FlashAttention solves this with the **online softmax** (softmax rescaling) trick: it maintains a running maximum and a running sum of exponentials as it processes each new block, and whenever a new block reveals a larger score, it rescales the previously-accumulated partial output and partial sum by the appropriate correction factor so that the final result is mathematically identical to having computed softmax over the whole row at once. This is the piece of numerical bookkeeping that lets the algorithm process the sequence block-by-block without ever needing the whole matrix in memory simultaneously, while still producing bit-for-bit (up to floating point associativity) the exact same output as standard attention.

```python
def online_softmax_update(m_prev, l_prev, o_prev, scores_block, v_block):
    """
    One step of the online-softmax / rescaling trick used inside FlashAttention.
    m: running max of scores seen so far (per query row)
    l: running sum of exp(scores - m) seen so far (per query row)
    o: running (unnormalized) weighted sum of V seen so far
    """
    m_block = scores_block.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m_prev, m_block)

    # rescale factors correct for the shift in the running max
    correction_prev = torch.exp(m_prev - m_new)
    p_block = torch.exp(scores_block - m_new)

    l_new = l_prev * correction_prev + p_block.sum(dim=-1, keepdim=True)
    o_new = o_prev * correction_prev + p_block @ v_block

    return m_new, l_new, o_new
    # after the final block: output = o_new / l_new
```

Because this approach never materializes the `N x N` matrix in HBM, it turns what used to be `O(N^2)` HBM traffic into something close to `O(N)` (in terms of block reads/writes), which is why FlashAttention is simultaneously faster (fewer slow memory round-trips) and dramatically more memory-efficient (peak memory no longer scales quadratically with sequence length) despite computing the exact same mathematical function as standard attention — there is no approximation, no loss of precision, no quality trade-off, which is why it was adopted essentially universally almost immediately after publication, in contrast to MQA/GQA/MLA where the memory-quality trade-off is a real design decision.

FlashAttention-2 (2023) refined the original algorithm to improve GPU occupancy and reduce non-matmul overhead — it restructured the parallelization strategy (parallelizing over sequence length in addition to batch and head dimensions, and better balancing work across warps) to better saturate modern GPUs, roughly doubling throughput over the original. FlashAttention-3 (2024) targeted Hopper-generation GPUs (H100) specifically, exploiting asynchrony between the tensor cores and the memory-copy units and adding support for low-precision (FP8) execution with techniques to control the accuracy loss that low precision would otherwise introduce, pushing hardware utilization even closer to the theoretical peak. Across all three versions, the core algorithmic idea — tile the computation, keep it in fast on-chip memory, use online softmax to avoid ever materializing the full attention matrix — remains the same; the successive versions are primarily about squeezing more of the available hardware's raw throughput out of that same IO-aware strategy.
