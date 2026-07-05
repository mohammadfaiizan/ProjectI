## KV-Cache Management and Memory Scaling

### 1. Why a KV cache exists at all

Autoregressive decoding generates one token at a time: given tokens `x_1 ... x_t`, the model produces `x_{t+1}`, appends it, and repeats. Inside every self-attention layer, computing the output at position `t` requires the query at position `t` to attend over the keys and values of *every* position `1..t`:

```
Attention(Q_t, K_{1:t}, V_{1:t}) = softmax(Q_t K_{1:t}^T / sqrt(d_h)) V_{1:t}
```

Naively, generating token `t+1` from scratch means re-running the full forward pass over the entire sequence `1..t` — recomputing `K_i` and `V_i` for every earlier position `i`, even though those keys and values never change once position `i` has been processed (they depend only on tokens up to `i`, and causal masking means later tokens cannot influence earlier ones). This is pure wasted compute: an O(t) recomputation for every one of the O(n) decode steps, giving O(n^2) total work when O(n) is achievable.

The fix is the **KV cache**: after computing `K_i, V_i` for a position once, store them in GPU memory (HBM) and simply append the newly computed `K_t, V_t` at each step. Decoding step `t+1` then only needs to compute `Q_{t+1}, K_{t+1}, V_{t+1}` from the current token, and can read `K_{1:t}, V_{1:t}` back from cache rather than recomputing them. Queries are never cached — a query is only ever used once, at the step it's produced, and discarded.

This trades compute for memory: you now hold two tensors (`K` and `V`, per layer, per head) whose size grows linearly with sequence length, for the entire lifetime of the request. Understanding exactly how that memory scales — and why it, rather than the model's parameter count, is usually the thing that runs out first — is the single most load-bearing piece of intuition for reasoning about LLM serving economics.

### 2. Deriving the memory formula from first principles

Fix a transformer with:

- `n_layers` — number of transformer blocks, each with its own independent KV cache (caches are not shared across layers).
- `n_kv_heads` — number of *key/value* heads. In plain multi-head attention (MHA) this equals the number of query heads `n_heads`; under grouped-query attention (GQA) it is smaller; under multi-query attention (MQA) it is 1.
- `head_dim` (`d_h`) — the dimensionality of each individual head's key/value vector. Typically `d_h = d_model / n_heads`.
- `seq_len` (`s`) — number of tokens currently in the cache (grows by one per decode step, or is populated all at once for a prompt's prefill).
- `batch_size` (`b`) — number of concurrent sequences being served.
- `bytes_per_value` — size of the numeric type used to store each cached scalar (2 bytes for fp16/bf16, 1 byte for int8/fp8, ~0.5 bytes for 4-bit KV quantization).

Per token, per layer, you must store one key vector and one value vector, each of size `n_kv_heads x d_h`. That gives the **factor of 2** (K and V) in the canonical formula:

```
KV_cache_bytes = 2 x n_layers x n_kv_heads x d_h x seq_len x batch_size x bytes_per_value
```

Read it left to right as a chain of independent multipliers, each contributing its own axis of scaling:

- `2` — one copy for K, one for V. This is fixed; nothing shrinks it (MLA changes what's *inside* this factor conceptually but the cache in DeepSeek's formulation is really one shared latent, not literally "K and V" — see Section 5).
- `n_layers` — every layer keeps an independent cache; a 60-layer model pays 60x what a single-layer toy model would.
- `n_kv_heads x d_h` — this product is the "KV width" per token per layer per K-or-V-tensor. This is exactly the term that MHA, GQA, MQA, and MLA differ on, and it's the primary attention-architecture-level lever for cache size (Section 5).
- `seq_len` — the number of tokens whose K/V has been computed and retained. This grows without bound as generation proceeds (up to the context window), and it is the term that makes long-context serving qualitatively different from short-context serving.
- `batch_size` — every concurrent request pays this cost independently; caches are per-request, not shared (except for prefix-sharing schemes, Section 7 and file 003).
- `bytes_per_value` — the only "free" lever: changing numeric precision scales cache size linearly with no architectural change.

Note what is *not* in this formula: `d_model`, `vocab_size`, total parameter count, MLP width, or number of attention heads used for *queries* specifically (only KV heads matter). This is the first sign that KV-cache scaling and weight-memory scaling are governed by different quantities and can diverge sharply.

### 3. Worked numeric examples

Take a concrete, LLaMA-2-70B-like dense MHA configuration: `n_layers = 80`, `n_heads = n_kv_heads = 64`, `d_h = 128` (so `d_model = 8192`), fp16 cache (`bytes_per_value = 2`).

Per-token, per-layer, per K-or-V-tensor width: `n_kv_heads x d_h = 64 x 128 = 8192` values. With the factor of 2 and 80 layers, **per-token cache cost**:

```
per_token_bytes = 2 x 80 x 8192 x 2   # (K&V) x layers x width x bytes
                = 2,621,440 bytes  ≈ 2.5 MiB / token
```

That number — roughly 2.5 MiB of cache per token held in the sequence, per request — is the atomic unit to multiply out.

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch_size, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_value

GiB = 1024 ** 3

for seq_len in (4_000, 128_000, 1_000_000):
    b = kv_cache_bytes(n_layers=80, n_kv_heads=64, head_dim=128,
                        seq_len=seq_len, batch_size=1, bytes_per_value=2)
    print(f"{seq_len:>9,} tokens -> {b/GiB:.2f} GiB per request (single sequence)")
```

Running this mentally (or literally): at `seq_len = 4,000` you get roughly `2.5 MiB x 4,000 ≈ 9.8 GiB`; at `128,000` tokens you get `≈ 313 GiB`; at `1,000,000` tokens you get `≈ 2.44 TiB` — **for a single request, at fp16, with plain MHA**. An H100 GPU has 80 GiB of HBM. A single 1M-token-context MHA request would need on the order of 30 H100-equivalents' worth of memory *just for its own KV cache*, before a single weight or activation is loaded. Even the 128K case (313 GiB) exceeds a single GPU and would need multi-GPU tensor-parallel sharding of the cache just to hold one such request.

Now compare this to weight memory: LLaMA-2-70B's parameters, at fp16 (2 bytes each), occupy `70e9 x 2 = 140 GiB` — a **fixed** cost paid once, independent of how many requests you serve or how long their contexts are. At 4K context, a single request's KV cache (9.8 GiB) is small relative to the 140 GiB of weights; the weights dominate, and you can batch many short-context requests before KV cache becomes the constraint. At 128K context, one request's cache (313 GiB) already dwarfs the weights. This crossover — where cache-per-request, multiplied by however many concurrent requests you want to serve, exceeds fixed weight memory — is the central fact of long-context serving economics, elaborated in Section 6.

### 4. The batch-size multiplication that actually matters in production

Production serving is not about one request; it's about how many requests you can hold *concurrently* in a batch, because batching is what lets you amortize memory-bandwidth cost across requests during decode (file 003 covers why decode is memory-bandwidth-bound and batching is the fix). So the practically relevant question is not "how big is the cache for one sequence" but "how many sequences of average length L can I fit in the HBM left over after weights and activations."

```python
def max_batch_size(total_hbm_bytes, weight_bytes, activation_headroom_bytes,
                    n_layers, n_kv_heads, head_dim, avg_seq_len, bytes_per_value=2):
    available_for_kv = total_hbm_bytes - weight_bytes - activation_headroom_bytes
    per_seq = kv_cache_bytes(n_layers, n_kv_heads, head_dim, avg_seq_len,
                              batch_size=1, bytes_per_value=bytes_per_value)
    return max(0, available_for_kv // per_seq)

GiB = 1024 ** 3
n = max_batch_size(
    total_hbm_bytes=8 * 80 * GiB,       # 8x H100, 640 GiB total
    weight_bytes=140 * GiB,             # LLaMA-2-70B fp16 weights
    activation_headroom_bytes=20 * GiB, # rough working-memory reservation
    n_layers=80, n_kv_heads=64, head_dim=128, avg_seq_len=4_000,
)
print(n)   # roughly (640 - 140 - 20) GiB / 9.8 GiB ≈ 48 concurrent 4K-context sequences
```

At an average context of 4K tokens on an 8xH100 node, roughly 48 concurrent sequences fit. At an average context of 32K tokens (8x longer), the same math gives roughly 6 concurrent sequences — an 8x drop in achievable batch size, and therefore roughly an 8x drop in decode throughput and serving efficiency, for the *same hardware and same model*, purely because contexts got longer. This is why context-length growth is not a "free" product improvement from a serving-cost perspective: every doubling of typical context roughly halves how many users a fixed GPU fleet can serve concurrently, all else equal.

### 5. Attention-variant comparison: MHA vs GQA vs MQA vs MLA

The `n_kv_heads x d_h` term is the only architectural lever inside the formula, and different attention variants attack it differently. (The exact per-model numbers below are worked out in depth in the architecture-specific docs; this section is about the general *pattern*, not re-deriving any one model's math.)

**MHA (multi-head attention).** `n_kv_heads = n_heads`. Every query head gets its own independent K/V head. Maximum expressivity, maximum cache cost. This is the baseline the formula in Section 2 assumes.

**MQA (multi-query attention).** `n_kv_heads = 1`. All query heads share a single K/V head. This is the extreme end of the GQA family — cache cost per token drops by a factor of `n_heads` relative to MHA, at a real (if often small) quality cost, because every head is now forced to attend using the same key/value geometry.

**GQA (grouped-query attention).** `n_kv_heads = n_groups`, with `1 < n_groups < n_heads` — query heads are partitioned into `n_groups` groups, and each group shares one K/V head. This interpolates between MHA and MQA: cache cost drops by a factor of `n_heads / n_groups` relative to MHA. GQA is the mechanism used across most modern open dense and MoE models (LLaMA-2/3's larger variants, Mistral, Qwen2.5, etc. — see the respective architecture docs in `..\OpenSource\` for each model's exact group count) precisely because it's a simple, structural, no-extra-inference-cost way to trade a modest amount of quality for a large, tunable cache reduction. The trade is *literal sharing*: the group's K/V head is identical information served to multiple query heads, so there's an information bottleneck relative to giving every head its own K/V.

**MLA (multi-head latent attention, DeepSeek-V2/V3).** Structurally different: rather than sharing K/V heads across groups, MLA compresses the *entire* per-token K/V information into one shared low-rank latent vector (`d_c = 512` in DeepSeek-V2) plus one small decoupled RoPE-carrying vector (`d_h^R = 64`), and reconstructs full-rank, full-head K/V from that latent via learned up-projection matrices at attention time. Only the latent is cached. The full derivation — the down/up-projection structure, the RoPE-decoupling fix, and the inference-time matrix-absorption trick that avoids materializing per-head K at every decode step — is in `..\OpenSource\006_DeepSeek_V2.md` Section 2 and 9; the point relevant here is purely the cache-scaling comparison: DeepSeek-V2 reports a per-token-per-layer cache cost (576 elements) smaller than even an 8-group GQA configuration (2,048 elements) at that model's head count, while empirically matching full MHA quality rather than paying GQA's typical quality tax. The mechanism is a **learned low-rank bottleneck with full-rank-effective reconstruction**, not literal head-sharing — which is why it can beat GQA on cache size *and* MHA on quality simultaneously, something structural head-sharing cannot do by construction.

**Sliding-window attention** (Mistral-style local attention layers) is a fourth, orthogonal lever: it doesn't change `n_kv_heads x d_h`, it caps the *effective* `seq_len` term by only ever attending to the most recent `W` tokens, evicting older cache entries. This bounds cache size independent of total generated length, at the cost of losing exact long-range dependency (mitigated in practice by interleaving a few full-attention layers with many windowed layers).

The table below summarizes the per-token-per-layer element count as a function of the same `n_heads = 128, d_h = 128` configuration used in `006_DeepSeek_V2.md`, purely to make the comparison concrete:

| Variant | KV width formula | Elements/token/layer (n_h=128, d_h=128) | Relative to MHA |
|---|---|---|---|
| MHA | `2 x n_heads x d_h` | 32,768 | 1x |
| GQA (8 groups) | `2 x 8 x d_h` | 2,048 | 16x smaller |
| MQA (1 group) | `2 x 1 x d_h` | 256 | 128x smaller |
| MLA (DeepSeek-V2) | `d_c + d_h^R` | 576 | ~57x smaller |

### 6. Why KV cache, not weight size, is usually the binding constraint

Weight memory is **fixed**: it's paid once when the model is loaded, independent of traffic. KV cache memory is **variable**: it scales with `batch_size x seq_len`, i.e. with exactly the two quantities that determine how much revenue-generating traffic you can serve concurrently. This has a direct operational consequence: a server that is "weight-memory-bound" has a fixed, known cost floor and can serve more traffic simply by adding more identical replicas behind a load balancer. A server that is "KV-cache-bound" has a cost floor that *depends on the traffic mix itself* (average context length, number of concurrent sessions) — the same GPU fleet serves a different effective number of users depending on whether those users are sending short chat turns or long-document / long-agentic-trace requests.

Concretely, restate the HBM budget as:

```
HBM_total = weight_bytes + activation_bytes + KV_cache_bytes(batch_size, seq_len)
```

For a fixed model on fixed hardware, `weight_bytes` and `activation_bytes` (the latter is comparatively small — activations for one forward pass, not accumulated across the sequence) are constants. Everything you can trade off at serving time lives inside `KV_cache_bytes`, and the two knobs you have — how many concurrent sequences (`batch_size`) and how long they run (`seq_len`) — are in direct tension: for fixed leftover HBM, you can serve many short-context requests or few long-context requests, and the product `batch_size x seq_len` is roughly capped by `(HBM_total - weight_bytes - activation_bytes) / (2 x n_layers x n_kv_heads x d_h x bytes_per_value)`.

This is precisely the number DeepSeek's MLA paper and every subsequent GQA/MLA architecture choice is targeting: shrinking the denominator (`n_kv_heads x d_h`, or replacing it with MLA's compressed latent width) directly multiplies out the achievable `batch_size x seq_len` product for a fixed hardware budget — i.e., it converts directly into either more concurrent users at the same context length, or longer context at the same concurrency, or (typically) some blend of both. This is why an architecture-level KV-cache reduction is reported by frontier labs as an *inference cost* claim, not merely a quality claim: it changes the denominator of the serving-economics equation for every future deployment of the model, for the life of that model in production.

A second, practically important framing: weight memory is amortizable across arbitrarily many requests (load once, serve forever), while KV cache is *not* amortizable — every additional concurrent request pays its own, non-shared, full per-token cache cost (modulo the prefix-sharing techniques in Section 7 / file 003). This asymmetry is why serving cost per token typically does **not** scale simply with parameter count between two models of similar size but different attention architectures — the KV-cache term can dominate the comparison entirely at realistic context lengths and concurrency, independent of how many parameters either model has.

### 7. Cache-size reduction levers, summarized

Beyond attention-architecture choice (Section 5), a staff engineer reasoning about a deployment has several independent levers, and they compose multiplicatively:

- **Numeric precision of the cache itself.** Storing K/V in int8 or fp8 instead of fp16 halves cache size outright (`bytes_per_value: 2 -> 1`), independent of any architecture change; some serving stacks (e.g. vLLM, TensorRT-LLM) support this as a pure runtime flag. Quality impact is typically small for K/V (unlike weight quantization, discussed in file 002) because attention is a soft, averaged operation somewhat tolerant of small per-element error, but it is not zero-cost — very long contexts can accumulate quantization error across many attended positions.
- **Sliding-window / local attention** caps effective `seq_len` (Section 5).
- **Prefix / prompt caching**: when many requests share an identical prefix (a system prompt, a few-shot template, a shared long document multiple users query against), the KV cache for that shared prefix can be computed once and reused — either literally shared in memory (PagedAttention's copy-on-write blocks, file 003) or persisted and reloaded across requests (prompt caching APIs). This doesn't change the per-token formula but changes the *effective* `batch_size` multiplier for the shared portion, since it's paid once instead of once per request.
- **Cache eviction / offloading**: evicting less-likely-to-be-reused cache to CPU RAM or NVMe and reloading on a cache hit, trading latency for HBM headroom — a technique conceptually analogous to OS page swapping, and complementary to (not a replacement for) PagedAttention's in-HBM page management (file 003).
- **KV-cache quantization at export/serve time for MoE/dense models where GQA group count and head dim are already fixed by pretraining**: since architecture is baked in post-training, precision reduction and prefix caching are the only two levers a serving engineer can apply to an already-trained checkpoint without retraining.

### 8. A complete worked calculator

Putting the whole chapter's math into one reference implementation, including a rough MLA-style latent-cache mode for comparison:

```python
def kv_cache_bytes_mha_gqa(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                            bytes_per_value=2):
    """Standard MHA/GQA/MQA cache: pass n_kv_heads = n_heads for MHA, = 1 for MQA."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_value

def kv_cache_bytes_mla(n_layers, d_latent, d_rope, seq_len, batch_size, bytes_per_value=2):
    """MLA-style: cache is one shared latent (d_latent) + one shared rope key (d_rope)
    per token per layer -- no factor of 2, no per-head multiplication."""
    return n_layers * (d_latent + d_rope) * seq_len * batch_size * bytes_per_value

GiB = 1024 ** 3

configs = {
    "MHA (n_h=128)":  dict(n_kv_heads=128, head_dim=128),
    "GQA (8 groups)": dict(n_kv_heads=8,   head_dim=128),
    "MQA (1 group)":  dict(n_kv_heads=1,   head_dim=128),
}

for name, cfg in configs.items():
    b = kv_cache_bytes_mha_gqa(n_layers=60, seq_len=128_000, batch_size=1, **cfg)
    print(f"{name:<18} 128K ctx, 1 seq: {b/GiB:6.2f} GiB")

mla = kv_cache_bytes_mla(n_layers=60, d_latent=512, d_rope=64, seq_len=128_000, batch_size=1)
print(f"{'MLA (DSv2-style)':<18} 128K ctx, 1 seq: {mla/GiB:6.2f} GiB")
```

The relative ordering this prints — MHA far larger than GQA, GQA larger than MQA, MLA competitive with or better than an aggressive GQA configuration while retaining MHA-level quality — is the concrete, numeric version of the qualitative argument in Section 5, and is the calculation a staff-level interview question on this topic is generally probing for: not just "know the formula" but "be able to plug in a model's real config and reason about what dominates."

### 9. Real-model reference points

It's worth anchoring the abstract formula against a small table of real, publicly disclosed configurations, so the numbers in Sections 3-5 don't feel like they belong only to one hypothetical 70B example. Treat the derived per-token bytes below as illustrative arithmetic from each model's disclosed head configuration, not an independently benchmarked serving number (actual deployed cache footprint also depends on the serving stack's chosen precision and any KV-cache quantization applied at serve time):

| Model (approx. config) | Attention type | n_layers | n_kv_heads | head_dim | Per-token-per-layer elements | Per-token bytes (fp16, all layers) |
|---|---|---|---|---|---|---|
| Dense 7B-class, MHA | MHA | 32 | 32 | 128 | 8,192 | 32 x 8,192 x 2 x 2 ≈ 1.05 MiB |
| Dense 70B-class, GQA(8) | GQA | 80 | 8 | 128 | 2,048 | 80 x 2,048 x 2 x 2 ≈ 655 KiB |
| Mistral-7B-class, GQA(8) + sliding window | GQA + windowed | 32 | 8 | 128 | 2,048 | bounded by window W, not total seq_len |
| DeepSeek-V2-class, MLA | MLA | 60 | n/a (latent) | n/a | 576 | 60 x 576 x 2 ≈ 67.5 KiB |

Two things jump out. First, the GQA(8) 70B-class configuration's *per-layer* element count is smaller than the 7B MHA model's despite having more than double the layers and a much larger overall model — the attention-architecture choice dominates the layer-count difference entirely, reinforcing that "bigger model" and "bigger KV cache" are not the same claim. Second, the MLA row's per-token bytes are roughly an order of magnitude below even the aggressive GQA(8) row, despite fewer layers being unable to explain that gap on its own (60 vs 80) — the latent-compression mechanism itself is doing essentially all of the work. Sliding-window architectures don't have a single "per-token bytes" number in the same sense, because their defining property is that per-token cost stops accumulating once the window fills — the right comparison for those models is "steady-state cache size" (a constant, `W`-token cache) rather than a growing function of total generated length.

### 10. Mental-math shortcuts worth having ready

A staff interview will often want an order-of-magnitude answer produced live, not a precise figure computed with a calculator. Two shortcuts make this tractable:

- **Per-token-per-layer bytes for a standard MHA/GQA model** is just `2 x n_kv_heads x head_dim x bytes_per_value`. For a typical `head_dim=128`, fp16 cache, this is `2 x n_kv_heads x 128 x 2 = 512 x n_kv_heads` bytes. At `n_kv_heads = 8` (a common GQA choice), that's 4,096 bytes/token/layer — multiply by layer count and you have per-token-total bytes without needing to separately track head_dim and the factor of 2 as distinct steps.
- **The crossover context length** where KV cache starts to rival weight memory is a genuinely useful number to be able to estimate on the spot: set `weight_bytes ≈ batch_size x seq_len x per_token_total_bytes` and solve for `seq_len` at `batch_size=1` to get the single-sequence context length at which cache alone matches the weights. For the 70B/GQA(8)/80-layer example (655 KiB/token, 140 GiB weights), that crossover is at `140 GiB / 655 KiB ≈ 224,000 tokens` for a *single* sequence — meaning at realistic multi-tenant concurrency (even just 10-20 concurrent sequences), the crossover context length where aggregate cache matches weight memory drops to the 10K-20K token range, well within ordinary long-document or long-agentic-trace use cases. This is the single most useful number to be able to derive live: it makes concrete exactly when a given deployment stops being "weight-memory-bound" and becomes "KV-cache-bound," which is the crux of nearly every serving-economics argument in this module.

### 11. Common pitfalls when reasoning about this in an interview

- **Conflating "more layers" with "more cache" without checking head configuration.** A model with more layers but a smaller `n_kv_heads x head_dim` product can have a *smaller* total cache than a shallower model with a wider, ungrouped attention configuration — always multiply out both factors rather than eyeballing depth alone.
- **Forgetting that queries are never cached.** Only K and V are cached; a query is used once and discarded. This is why MLA's query-side compression (`c_t^Q`, described in `..\OpenSource\006_DeepSeek_V2.md`) is explicitly framed as an activation-memory/training-efficiency optimization, not a caching optimization — there's no query cache to shrink.
- **Treating KV-cache quantization as free.** It's a real, if usually small, quality cost, not a pure win — accumulated error across many attended positions in a very long context is a genuine (if generally under-characterized in the open literature) risk, and shouldn't be waved away as automatically safe at every context length and every model.
- **Assuming sliding-window attention is a strict improvement.** It bounds cache size but at the cost of a genuine capability loss for anything requiring a dependency older than the window — a trade decided at pretraining time, and one a serving engineer cannot retrofit onto an already-trained dense/GQA checkpoint without retraining hybrid local/global layers in from scratch.

### 12. Interaction with tensor parallelism

Nothing in Sections 1-11 addressed how the cache is physically distributed once a model is sharded across multiple GPUs, which is the norm for any model too large to fit on one device. Under standard tensor parallelism, attention heads are typically partitioned across GPUs — each GPU owns a subset of the `n_kv_heads` (and the corresponding query heads) and computes attention locally for its shard, meaning the KV cache itself is *sharded* across the tensor-parallel group exactly the way the corresponding weight matrices are. This has a direct, favorable consequence for the memory formula: each GPU's *local* KV-cache burden is the full formula divided by the tensor-parallel degree `TP`, i.e. `n_kv_heads` in the per-GPU accounting is really `n_kv_heads / TP`. This is precisely why GQA/MLA's cache reduction and tensor-parallel sharding are complementary, not redundant, levers — sharding spreads a fixed total cache burden across more devices' HBM; GQA/MLA shrinks the total burden being spread in the first place. A model with very few KV heads (aggressive GQA, or MQA's single head) has an obvious lower bound on how finely its cache can be sharded this way: you cannot tensor-parallelize a single shared KV head across more GPUs than there are heads to distribute without an additional (and non-free) replication or all-to-all communication step, which is a genuine, if second-order, engineering complication aggressive KV-head reduction introduces at very high tensor-parallel degrees — worth flagging as a real trade-off rather than treating cache reduction as a strictly dominant strategy along every axis simultaneously.

### 13. A short worked interview-style derivation, end to end

To make the whole chapter's logic replayable as a single connected chain of reasoning (the form an interviewer is most likely to actually ask for at a whiteboard), here is the full path from "how much cache does one long request need" to "how many concurrent requests fit," using round numbers:

```
Given: 80-layer model, GQA with 8 KV heads, head_dim 128, fp16 cache, 8xH100 (640 GiB HBM),
       140 GiB of fp16 weights, 20 GiB activation reserve, requests averaging 32K tokens.

Step 1 -- per-token-per-layer bytes:      2 x 8 x 128 x 2  = 4,096 bytes
Step 2 -- per-token-total bytes:          4,096 x 80       = 327,680 bytes (~320 KiB)
Step 3 -- per-sequence bytes at 32K ctx:  320 KiB x 32,000  ≈ 10 GiB
Step 4 -- HBM available for KV cache:     640 - 140 - 20    = 480 GiB
Step 5 -- max concurrent sequences:       480 GiB / 10 GiB  = 48
```

Forty-eight concurrent 32K-context sequences on an 8xH100 node, for this configuration — and every one of the five steps above is a single multiplication or subtraction, chainable live without needing a calculator beyond basic arithmetic. Being able to reproduce this chain fluently, and to immediately re-run it with a different attention variant's head count swapped into Step 1, is the concrete skill this file is building toward.

### 14. One more subtlety: prefill also allocates cache, before generation even starts

Everything above framed cache growth as a per-decode-step accumulation, which is the right mental model for the *decode* phase, but it's worth being explicit that the *entire prompt's* KV cache is materialized in one shot during prefill (file 005 covers prefill's compute characteristics in depth) — a request with a 100K-token prompt and a 50-token expected output allocates essentially all of its lifetime cache footprint immediately, at admission time, not gradually. This matters for admission control (file 003 Section 7 and file 008 Part 2 Q6): a scheduler cannot safely admit a long-prompt request "optimistically" and hope its cache demand ramps up slowly — the demand is present in full from the first iteration that request participates in, and any capacity check has to account for the prompt length, not just an assumed steady decode-time growth rate.

### 15. Summary of the mental model

1. KV cache exists to avoid O(n^2) recomputation during autoregressive decoding, at the cost of O(n) memory that grows with sequence length.
2. Its size is `2 x n_layers x n_kv_heads x d_h x seq_len x batch_size x bytes_per_value` — linear in five independent axes, three of which (layers, heads, head_dim) are architecture-fixed at pretraining time, two of which (seq_len, batch_size) are determined at serving time by traffic.
3. Because seq_len and batch_size are the *serving-time* variables and they trade off against each other for fixed HBM, KV cache — not weight size — is typically the binding constraint on achievable concurrency at any non-trivial context length, and this is precisely why architecture-level cache-reduction techniques (GQA, MQA, MLA) are treated as first-class inference-cost contributions by frontier labs, not incidental side effects.
4. GQA/MQA reduce cache by literal, structural head-sharing (a quality/cache trade decided at architecture time); MLA reduces cache via a learned low-rank bottleneck with full-rank reconstruction, avoiding the usual quality tax — see `..\OpenSource\006_DeepSeek_V2.md` for the exact mechanism.
5. Everything else being explored in this module — continuous batching, PagedAttention, quantization, disaggregated serving — is in some sense downstream of this one constraint: they are all techniques for getting more useful work out of a fixed pool of HBM whose biggest variable consumer, at scale, is the KV cache.

**Where each subsequent file in this module picks up this thread:**

- File 002 (quantization) attacks `bytes_per_value` directly for weights, and shows that the memory freed from weights converts into more room for exactly the KV cache discussed here.
- File 003 (continuous batching / PagedAttention) attacks the *efficiency* with which a fixed KV-cache budget converts into achievable concurrent `batch_size` at a given `seq_len`.
- File 004 (speculative decoding / prefix caching) shows how the *shared* portion of a cache (a common prefix) can be paid for once rather than once per request, and how decode throughput can be raised without changing the cache formula at all.
- File 005 (serving infrastructure) connects achievable `batch_size` back to the cost-per-token and latency metrics a staff engineer is ultimately accountable for.
- File 006 (routing) is the one file in this module where KV-cache math is *not* the central constraint — it operates one level up, at the question of which model a request should even reach in the first place.

Holding this chain in mind — cache formula, cache-vs-weight crossover, the specific lever each attention variant pulls, and how every downstream serving technique either shrinks the cache, uses it more efficiently, or amortizes its shared portion — is what separates a candidate who can recite "KV cache is important" from one who can derive, on demand, exactly why and by how much.

As a final sanity check on the whole chapter: if asked to name, in one sentence, the single fact that makes every other fact in this file matter, it is this — decode-time serving cost is overwhelmingly a memory-bandwidth story, KV cache is the one memory consumer that grows with *traffic* rather than sitting fixed at *load time*, and therefore it is the lever that determines how a fixed GPU fleet's effective capacity responds to changes in context length and concurrency, which is exactly the lever every technique in the rest of this module is trying to pull in one direction or another.
