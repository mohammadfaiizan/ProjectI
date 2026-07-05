## Interview Questions — Part 1

## Q1: Derive the KV-cache memory formula from first principles. What does each term represent, and which terms are fixed at pretraining time versus determined at serving time?

Autoregressive decoding computes, at every layer, an attention operation where the query at the current position attends over the keys and values of every prior position. Those keys and values depend only on tokens already processed, so recomputing them at every future step is pure waste — the standard fix is to cache each position's key and value vectors the first time they're computed, and simply append to the cache as generation proceeds, avoiding O(n^2) recomputation across an n-token generation.

The memory cost of that cache is:

```
KV_cache_bytes = 2 x n_layers x n_kv_heads x head_dim x seq_len x batch_size x bytes_per_value
```

- `2` — one tensor for keys, one for values.
- `n_layers` — each transformer block keeps its own independent cache; caches are not shared across depth.
- `n_kv_heads` — the number of *key/value* heads (not necessarily equal to the number of query heads: this is exactly the axis GQA/MQA manipulate).
- `head_dim` — the width of each individual head's key/value vector.
- `seq_len` — number of cached token positions; grows by one per decode step (or is populated all at once during prefill).
- `batch_size` — number of concurrent sequences; each pays this cost independently (barring shared-prefix optimizations).
- `bytes_per_value` — numeric precision of the cache (2 bytes fp16, 1 byte int8/fp8, etc.).

`n_layers`, `n_kv_heads`, and `head_dim` are architecture choices fixed once the model is pretrained — nothing at serving time changes them. `seq_len` and `batch_size` are serving-time variables that trade off against each other under a fixed HBM budget, and `bytes_per_value` is the one lever a serving engineer can adjust post-hoc without retraining (KV-cache quantization). Note what's absent from the formula entirely: total parameter count, MLP width, vocabulary size — KV-cache scaling is governed by a different set of quantities than weight-memory scaling, which is precisely why the two can diverge sharply at long context (Q2).

## Q2: Why is KV-cache size, rather than raw model weight size, usually the binding constraint on how many concurrent requests a server can batch?

Weight memory is fixed: loaded once, paid once, independent of traffic. KV-cache memory is variable: it scales with `batch_size x seq_len`, which are exactly the two quantities that determine how much revenue-generating concurrent traffic a server can hold. The available-HBM budget decomposes as `HBM_total = weight_bytes + activation_bytes + KV_cache_bytes(batch_size, seq_len)`; for a fixed model on fixed hardware, the first two terms are constants, so everything a serving engineer can trade at runtime lives inside the KV-cache term, and the achievable `batch_size x seq_len` product is capped by whatever HBM is left over.

Concretely: a LLaMA-2-70B-class model's fp16 weights occupy roughly 140 GiB — fixed. Its KV cache at plain MHA, fp16, costs roughly 2.5 MiB per token per sequence (worked out in file 001), so a single 128K-context sequence alone costs on the order of 300 GiB — more than double the weights' own footprint, on one request. Once average context length grows past a few thousand tokens, the KV cache for even a modest number of concurrent sequences exceeds the weight footprint, and every further doubling of average context roughly halves the number of concurrent sequences that fit, for the *same* hardware — a direct throughput and cost hit with no change to the model or the hardware. This is why GQA, MQA, and MLA are treated by frontier labs as first-class inference-cost contributions rather than incidental architecture choices: they shrink exactly the term that becomes the bottleneck as context length and concurrency both scale.

## Q3 (Coding): Implement a KV-cache memory calculator that supports standard MHA/GQA/MQA and an MLA-style latent-cache mode, and use it to compare configurations.

```python
def kv_cache_bytes_standard(n_layers, n_kv_heads, head_dim, seq_len, batch_size,
                             bytes_per_value=2):
    """MHA (n_kv_heads = n_heads), GQA (1 < n_kv_heads < n_heads), or MQA (n_kv_heads = 1)."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_value


def kv_cache_bytes_mla(n_layers, d_latent, d_rope, seq_len, batch_size, bytes_per_value=2):
    """MLA caches one shared compressed latent (d_latent) plus one shared decoupled
    RoPE key (d_rope) per token per layer -- no per-head multiplication, no factor of 2
    (the latent implicitly encodes both K and V, reconstructed via up-projection at
    attention time; see ../OpenSource/006_DeepSeek_V2.md for the full derivation)."""
    return n_layers * (d_latent + d_rope) * seq_len * batch_size * bytes_per_value


def max_concurrent_sequences(total_hbm_bytes, weight_bytes, activation_reserve_bytes,
                              per_sequence_cache_bytes):
    available = total_hbm_bytes - weight_bytes - activation_reserve_bytes
    if available <= 0 or per_sequence_cache_bytes <= 0:
        return 0
    return int(available // per_sequence_cache_bytes)


if __name__ == "__main__":
    GiB = 1024 ** 3
    seq_len, n_layers = 128_000, 80

    mha = kv_cache_bytes_standard(n_layers, n_kv_heads=64, head_dim=128,
                                   seq_len=seq_len, batch_size=1)
    gqa = kv_cache_bytes_standard(n_layers, n_kv_heads=8, head_dim=128,
                                   seq_len=seq_len, batch_size=1)
    mla = kv_cache_bytes_mla(n_layers, d_latent=512, d_rope=64,
                              seq_len=seq_len, batch_size=1)

    print(f"MHA: {mha/GiB:7.2f} GiB/seq")
    print(f"GQA: {gqa/GiB:7.2f} GiB/seq")
    print(f"MLA: {mla/GiB:7.2f} GiB/seq")

    n = max_concurrent_sequences(total_hbm_bytes=8*80*GiB, weight_bytes=140*GiB,
                                  activation_reserve_bytes=20*GiB,
                                  per_sequence_cache_bytes=gqa)
    print(f"Concurrent 128K-context sequences fitting on 8xH100 with GQA cache: {n}")
```

This is exactly the calculation underlying the qualitative claims in files 001 and the DeepSeek-V2 architecture doc: plugging in real head counts and comparing against a fixed HBM budget turns "MLA is more memory-efficient" from a slogan into a number you can defend in front of a whiteboard.

## Q4: Compare MHA, GQA, and MLA on cache size and quality. Why can MLA beat GQA on cache size *and* MHA on quality simultaneously, when GQA cannot?

MHA gives every query head its own K/V head — maximum cache cost, maximum expressivity. GQA partitions query heads into groups and has each group share one K/V head, shrinking cache size by the ratio `n_heads / n_groups`; this is *literal, structural sharing* decided at architecture time, and it pays a real quality cost because multiple query heads are now forced to attend using an identical key/value geometry rather than their own. MQA is the extreme case, one shared K/V head for all query heads.

MLA (DeepSeek-V2) takes a structurally different approach: rather than sharing K/V heads across groups, it compresses the *entire* per-token K/V information into one shared low-rank latent vector, and reconstructs full-rank, per-head K/V from that latent via learned up-projection matrices at attention time. Only the latent (plus a small decoupled RoPE-carrying vector — RoPE and low-rank compression are in tension, requiring the decoupling fix described in `../OpenSource/006_DeepSeek_V2.md`) is cached. Because the up-projections are learned and full-rank-effective (they can, in principle, reconstruct any per-head K/V pattern the compressed latent has capacity to encode), MLA is not throwing away per-head distinctiveness the way literal head-sharing does — it's routing all the per-token information through a bottleneck and then re-expanding it, rather than forcing several heads to share one un-expanded copy. DeepSeek-V2's own reported comparison: MLA's cache cost (576 elements/token/layer at that model's head config) is smaller than even an 8-group GQA configuration (2,048 elements), while empirically matching or exceeding full MHA quality — a combination literal head-sharing cannot achieve by construction, because sharing is inherently lossy in a way a learned low-rank bottleneck with full reconstruction is not (up to the bottleneck's actual rank capacity).

## Q5 (Scenario): Your product needs to support 128K-token context windows for a large number of concurrent enterprise users, on a fixed GPU budget. Walk through the levers you'd pull, in order of how much they'd move the needle.

First, quantify the problem: at 128K context, a plain-MHA 70B-class model's KV cache is on the order of 300 GiB *per sequence* (file 001's worked example) — larger than the weights themselves. No amount of clever scheduling saves you if the per-sequence cost itself is this large; the first-order lever has to attack the per-token cache-size term directly.

1. **Attention architecture**, if you control model choice or retraining: pick (or fine-tune toward) GQA with an aggressive group count, or an MLA-based model if available — this is a multiplicative reduction (8-50x depending on configuration) applied to every request, for free at serving time, and is the single biggest lever available. If the model is fixed and can't be changed, this lever isn't available and you move to the next ones.
2. **KV-cache quantization** (fp8 or int8 for the cache specifically, independent of weight precision) — a further ~2x reduction, usually with small quality impact for attention specifically (file 002 Section 7), applicable to an already-trained checkpoint with no retraining.
3. **Weight quantization** (file 002) — doesn't shrink the KV cache directly, but frees HBM that would otherwise be occupied by weights, which converts directly into more room for KV cache under the shared HBM budget (file 002 Section 8) — often the single largest *practical* lever when the model itself can't be changed.
4. **PagedAttention-style memory management** (file 003) — doesn't shrink cache per se, but eliminates the internal/external fragmentation that a naive contiguous-buffer allocator would otherwise waste, meaning the HBM you do have converts into usable concurrent-request capacity much more efficiently.
5. **Prefix/prompt caching** (file 004 Section 6) — if enterprise users share a common system prompt or a shared long document (common in enterprise deployments — a shared knowledge base, a shared instruction template), share that portion's cache across requests via copy-on-write, paying its cost once rather than once per request.
6. **Prefill/decode disaggregation and careful batching-policy tuning** (file 005) — doesn't change memory footprint, but ensures the memory you have is being used at maximal achievable batch size without TTFT/TPOT regressions caused by scheduling interference.
7. As a last resort, **sliding-window/local-attention** variants or context-length-based tiered pricing/throttling, if the enterprise use case tolerates it — capping effective seq_len directly, at the cost of losing exact long-range attention.

The ordering matters: architecture-level changes are the only ones with multiplicative, model-wide impact; everything after that is squeezing efficiency out of whatever HBM budget the architecture leaves you with.

## Q6: Explain GPTQ's quantization mechanism conceptually. Why is it better than naive round-to-nearest quantization at low bit-widths?

Naive round-to-nearest (RTN) quantization treats every weight independently: pick a scale, round each weight to the nearest representable value on the target grid, done. This ignores that weights in a linear layer (`y = Wx`) are *not* independent with respect to their effect on the layer's output — quantizing one weight introduces an error that, propagated through the shared input `x`, could in principle be partially cancelled by nudging the *remaining* weights in a compensating direction. RTN throws this compensation opportunity away entirely.

GPTQ (building on the Optimal Brain Compression/Surgeon lineage) exploits it: it processes a weight matrix column by column, and after quantizing each column, it uses the inverse of the layer's local Hessian (computed cheaply from a small calibration set, as a least-squares proxy for `||Wx - W_q x||^2` over calibration activations, not the full training loss) to redistribute the quantization error introduced by that column into the *not-yet-quantized* columns, in the direction that best cancels the error. It then moves to the next column, now working with already error-compensated weights. This greedy-but-compensating procedure is what lets GPTQ reach 4-bit weight-only quantization with a much smaller quality gap than RTN at the same bit-width — RTN's error compounds independently across every weight with no mechanism to offset it, while GPTQ's error is actively redistributed to minimize the layer's overall output reconstruction error.

## Q7: Explain AWQ's mechanism, and contrast it with GPTQ. Why doesn't AWQ need mixed-precision kernels to protect "important" weights?

AWQ starts from the observation that a weight's importance to quantize carefully isn't intrinsic to the weight's own magnitude — it's about which *activation channel* that weight multiplies against. In `y = Wx`, a weight feeding into a systematically large-magnitude activation channel has an outsized effect on the output relative to an identically-sized weight feeding a small-magnitude channel, and empirically, trained transformers exhibit a small number of such large-magnitude "salient" activation channels.

Rather than keeping the weights feeding those channels in higher precision (mixed-precision — different bit-widths within one matrix — which is awkward for uniform, dense low-bit GPU kernels), AWQ exploits an exact mathematical identity: scaling up a channel's weights by `s_j` and scaling down the corresponding activation by `1/s_j` leaves the product `Wx` unchanged. Scaling up a salient channel's weights before quantizing gives that channel effectively more range relative to the fixed quantization grid, reducing its relative rounding error, while the compensating `1/s_j` on the activation side is folded into an adjacent operation so the correction is exact, not approximate. The calibration process picks per-channel scaling factors from observed activation magnitude statistics.

Contrast with GPTQ: GPTQ operates purely on weights, using second-order (Hessian) curvature information to compensate error across weights within a layer, and needs no notion of activation-channel saliency. AWQ operates by pre-conditioning weights based on the activation channels they interact with, using only first-order magnitude statistics — cheaper to calibrate (no per-layer matrix inversion) but attacking a different, and complementary, source of quantization error. Neither dominates the other unconditionally across all models and tasks; both are standard, widely deployed 4-bit weight-only PTQ techniques with comparable practical outcomes on most workloads.

## Q8 (Coding): Implement a per-channel symmetric int8 quantizer and dequantizer for a weight matrix, and show why per-channel granularity beats a single tensor-wide scale.

```python
import numpy as np

def quantize_per_channel_symmetric(W: np.ndarray, qmax=127):
    """W: [out_channels, in_channels]. One scale per output channel (row)."""
    max_abs = np.max(np.abs(W), axis=1, keepdims=True)          # [out_channels, 1]
    scales = np.where(max_abs > 0, max_abs / qmax, 1.0)
    W_q = np.clip(np.round(W / scales), -qmax, qmax).astype(np.int8)
    return W_q, scales.squeeze(axis=1)                          # scales: [out_channels]

def dequantize_per_channel(W_q: np.ndarray, scales: np.ndarray):
    return W_q.astype(np.float32) * scales[:, None]

def quantize_per_tensor_symmetric(W: np.ndarray, qmax=127):
    scale = np.max(np.abs(W)) / qmax
    W_q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int8)
    return W_q, scale

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    W = rng.normal(0, 1, size=(4, 256)).astype(np.float32)
    W[0] *= 50   # simulate one output channel with a much larger typical magnitude

    Wq_pt, s_pt = quantize_per_tensor_symmetric(W)
    Wq_pc, s_pc = quantize_per_channel_symmetric(W)

    err_pt = np.abs(dequantize_per_channel(Wq_pt, np.full(4, s_pt)) - W)
    err_pc = np.abs(dequantize_per_channel(Wq_pc, s_pc) - W)

    print("Per-tensor  mean abs error by channel:", err_pt.mean(axis=1))
    print("Per-channel mean abs error by channel:", err_pc.mean(axis=1))
```

With one outlier-magnitude channel (`W[0] *= 50`), a single tensor-wide scale is stretched to accommodate that channel's range, wasting most of the int8 grid's resolution on the other three channels, whose reconstruction error is visibly larger under `err_pt` than under `err_pc`. Per-channel quantization gives each row its own scale tuned to its own range, so the outlier channel doesn't degrade the others — the exact reason GPTQ, AWQ, and GGUF's k-quants all use per-channel or per-group granularity rather than a single scale for an entire matrix (file 002 Section 2).

## Q9 (Scenario): You quantized a model to int4 with a standard PTQ pipeline. Aggregate perplexity looks essentially unchanged, but a specific downstream eval — say, multi-digit arithmetic or strict JSON schema adherence — has collapsed. Diagnose, and explain why perplexity didn't catch it.

Perplexity is an aggregate, averaged signal over the whole token distribution of a broad evaluation corpus — it measures how well-calibrated the model's probabilities are *on average*, dominated by the overwhelming majority of "easy," high-frequency, low-information tokens (common words, punctuation, syntactic glue) where quantization error barely registers. A capability like multi-digit arithmetic or exact schema adherence depends on a comparatively small number of *specific, high-precision decision points* — the exact digit chosen at a particular position, the exact closing brace emitted at the exact right moment — and it's entirely possible for a quantization scheme to introduce error that is negligible in aggregate (barely moving perplexity) while being catastrophic at exactly those few decision points, because the error isn't uniformly distributed — it's concentrated wherever the model's underlying computation was already operating with a small effective margin (e.g., distinguishing between two similarly-likely-looking digit continuations, or the weights/activations responsible for tracking nested bracket depth).

Diagnosis path: first confirm this isn't a calibration-data mismatch (was the calibration set representative of code/structured-output traffic at all, or dominated by prose? A calibration set skewed away from the failing capability's typical inputs will produce a scale/zero-point tuned for the wrong distribution). Second, check whether the failure is uniform across the quantized model or concentrated in specific layers — ablate by keeping a few likely-sensitive layers (commonly attention output projections, or layers close to the output head, which the GGUF k-quant heuristics also specifically privilege, file 002 Section 5) at higher precision and re-testing; if the eval recovers, that pinpoints which layers were sensitive. Third, consider whether GPTQ/AWQ was even used versus naive RTN — RTN is exactly the scheme most likely to produce this failure mode, since it has no error-compensation mechanism at all. Fourth, and most importantly as a process fix: this is the direct argument for why quantization validation must include task-representative evals, not just perplexity (file 002 Section 7) — perplexity is a necessary but not sufficient quality gate, and any quantization rollout for a capability-sensitive deployment needs eval coverage specifically targeting the capabilities the deployment actually depends on.

## Q10: Why does naive static batching waste GPU capacity for autoregressive generation?

Static batching groups a fixed set of `N` requests, pads them to a common length, and advances every sequence in lockstep for exactly as many steps as the *longest* sequence in the batch needs, because the physical batch is one tensor and every row is processed together. This wastes capacity two ways. First, **tail idling**: requests needing few output tokens finish early but their slot can't be reclaimed — the framework must keep the batch's shape consistent until the single longest-running request in the cohort finishes, during which time a chunk of the batch's slots do no useful work while still costing the full compute/memory price of that batch size. Second, **no mid-batch admission**: a new request arriving while a batch is running cannot be inserted into it (the batch's shape and KV-cache allocations are fixed for its lifetime), so it queues until the *entire* batch retires — which could take as long as the single longest request in that cohort — directly inflating time-to-first-token for unlucky arrivals and depressing average GPU utilization measured over a full serving window, even though a mid-batch snapshot might look fully utilized. Both failure modes share one root cause: the unit of scheduling is the whole batch's entire lifetime, not the individual decode step or individual request.

## Q11: Explain how continuous batching fixes this. What is "iteration-level scheduling," and what new problem does it introduce?

Continuous batching moves the unit of scheduling from "the whole batch, for its entire lifetime" down to "one forward pass" — i.e., one decode step, evaluated across whichever requests happen to be active right now. At every iteration, the scheduler retires any request that finished on the previous step (releasing its KV-cache allocation immediately), admits new requests from a waiting queue into whatever capacity just freed up, and then runs exactly one forward pass over the resulting active set. A request that finishes at step 5 frees its slot immediately, and a new request can occupy that slot on the very next iteration — no waiting for the rest of a cohort. This eliminates both static-batching failure modes: no tail idling (slots are reused as soon as they free), and no batch-lifetime admission barrier (new requests join at iteration granularity, bounded by a single forward pass's latency rather than by the slowest member of an entire cohort).

The problem it introduces: prefill (a large, compute-bound parallel pass over a new request's whole prompt) and decode (many small, memory-bandwidth-bound single-token steps for already-running requests) are computationally very different, and naively injecting a large prefill into an iteration that would otherwise be a fast decode step spikes that iteration's latency, delaying every decode step riding along in the same batch — directly hurting inter-token latency for already-streaming users. The standard mitigation is **chunked prefill**: splitting a long prompt's prefill into smaller pieces scheduled alongside ongoing decode iterations, smoothing the cost across several iterations instead of one large spike (file 003 Section 3, file 005 Section 2).

## Q12 (Coding): Implement a simplified continuous-batching scheduler that admits, runs, and evicts requests at iteration granularity, given a fixed capacity of concurrent slots.

```python
from dataclasses import dataclass, field

@dataclass
class Request:
    request_id: int
    max_tokens: int
    tokens_generated: int = 0
    finished: bool = False

    def step(self):
        self.tokens_generated += 1
        if self.tokens_generated >= self.max_tokens:
            self.finished = True


class ContinuousBatchScheduler:
    def __init__(self, max_concurrent_slots: int):
        self.max_slots = max_concurrent_slots
        self.active: dict[int, Request] = {}
        self.waiting: list[Request] = []
        self.iteration = 0
        self.log: list[dict] = []

    def submit(self, request: Request):
        self.waiting.append(request)

    def run_iteration(self):
        # 1. Evict anything finished on the previous iteration.
        finished_ids = [rid for rid, r in self.active.items() if r.finished]
        for rid in finished_ids:
            del self.active[rid]

        # 2. Admit from the waiting queue into freed capacity (FCFS policy).
        while len(self.active) < self.max_slots and self.waiting:
            req = self.waiting.pop(0)
            self.active[req.request_id] = req

        # 3. Advance every currently active request by exactly one decode step.
        for req in self.active.values():
            req.step()

        self.log.append({
            "iteration": self.iteration,
            "active": list(self.active.keys()),
            "waiting": len(self.waiting),
            "just_finished": finished_ids,
        })
        self.iteration += 1

    def is_idle(self) -> bool:
        return not self.active and not self.waiting


if __name__ == "__main__":
    sched = ContinuousBatchScheduler(max_concurrent_slots=2)
    sched.submit(Request(request_id=1, max_tokens=3))
    sched.submit(Request(request_id=2, max_tokens=6))
    sched.submit(Request(request_id=3, max_tokens=2))   # arrives, but queued: only 2 slots

    while not sched.is_idle():
        sched.run_iteration()

    for entry in sched.log:
        print(entry)
```

Running this shows request 3 sitting in `waiting` until request 1 (the shorter of the first two) finishes and frees a slot at iteration 3 — exactly the behavior static batching cannot provide, since request 3 would otherwise have to wait for *both* requests 1 and 2 to finish before being admitted at all. A production scheduler adds KV-cache-budget-aware admission (not just a slot count, since different requests' contexts consume different amounts of cache — file 003 Section 7) and a fairness/priority policy beyond plain FCFS, but the core admit/run/evict loop is unchanged.

## Q13: Explain PagedAttention's mechanism, including how it enables copy-on-write sharing of a common prompt prefix.

Continuous batching solves scheduling but creates a new memory-allocation problem: without knowing in advance how long a request will run, a naive implementation must reserve a contiguous KV-cache buffer sized for the worst case (the full context window) per request — most requests never use most of that reservation, wasting HBM to internal fragmentation, and as requests of varying length come and go, free HBM fragments into gaps too small to satisfy a new reservation even when aggregate free space would suffice (external fragmentation).

PagedAttention fixes this by dividing the KV cache into small, fixed-size blocks (analogous to OS memory pages) that can live anywhere in HBM, and giving each sequence a block table (analogous to a page table) mapping logical token positions to whichever physical block actually holds that range. Blocks are allocated one at a time, on demand, as a sequence grows, from a global free-block pool — no request ever reserves more than it has actually used, and any free block anywhere can satisfy any sequence's next allocation, since sequences never require contiguity. This eliminates both fragmentation modes the same way OS paging does for general-purpose memory.

Because a sequence's identity is just its block table, two sequences sharing an identical token prefix (a common system prompt, or multiple sampled completions branching from one prompt) can have their block tables point at the *same* physical blocks for the shared span — no duplication, no extra memory for the shared portion beyond the one copy already computed. Each shared block carries a reference count; as long as all sharing sequences only read the block (attending over it, not writing new tokens into it), they share it freely. The moment one sequence's generation diverges and needs to write into what was a shared block, that sequence gets a private copy of just that block, decrements the original's reference count, and continues independently — exactly the fork/copy-on-write pattern from OS process memory management, applied to KV-cache blocks.

## Q14 (Scenario): Your API's p99 time-to-first-token just doubled after a traffic increase. Walk through your diagnosis.

Start by separating whether this is a **distributional shift** (the whole TTFT distribution moved, p50 included) or a genuine **tail-only** regression (p50 stable, only p99 degraded) — these point to different root causes (file 005 Section 3), so this is the first branch, not an afterthought.

If p50 moved too: this looks like a genuine capacity shortfall. Check aggregate queue depth per replica pool and GPU utilization — rising queue depth with high utilization confirms the pool is simply under-provisioned for the new traffic volume and needs more replicas (and check whether autoscaling has actually kept pace — LLM replica autoscaling reacts slowly because loading model weights into a new replica's HBM isn't instantaneous, file 006 Section 3, so a traffic *spike* can outrun autoscaling's reaction time even if the eventual steady-state capacity would be sufficient).

If only p99 moved: suspect specific unlucky-request interference rather than aggregate shortfall. Candidates, roughly in order of likelihood: (a) a shift in prompt-length distribution — if the traffic increase specifically brought more long-prompt requests, their prefill cost directly inflates TTFT for anyone queued behind them, and if prefill and decode are colocated (not disaggregated, file 005 Section 2), a burst of long prefills also delays decode iterations for already-streaming requests, which can masquerade as a TTFT problem in aggregate dashboards that don't cleanly separate the two; (b) KV-cache exhaustion triggering preemption — check whether HBM/KV-cache occupancy is near its ceiling, since a pool running close to its cache budget will start evicting/preempting under load exactly the way file 003 Section 7 describes, and preempted requests re-queue, inflating their effective TTFT; (c) a batching-policy or scheduler regression — check whether achieved batch size actually dropped (rules out "still healthy, just busier") versus GPU utilization dropping alongside rising latency (points at a scheduling bug preventing the batch from filling, not a capacity problem at all); (d) a co-deployed model-version or config change coinciding with the traffic increase — check the deployment timeline for a coincident rollout (a canary or config push landing right as traffic increased is a classic false-attribution trap: the real cause might be the deployment, not the traffic).

Conclude with the fix matched to the diagnosis: more replicas / faster autoscaling for a genuine shortfall; prefill/decode disaggregation or chunked-prefill tuning for prefill-interference; KV-cache budget increase, better eviction/preemption policy, or admission control for cache exhaustion; and a rollback plus root-cause on the scheduler or deployment for the last two.

## Q15: Explain speculative decoding's mechanics precisely, and explain — with the actual argument, not just the claim — why it preserves the target model's exact output distribution.

A small, fast draft model proposes `k` tokens ahead autoregressively, recording its own probability `q(x_i)` at each drafted position. The large target model then verifies all `k` positions in a single parallel forward pass (since each position's target distribution conditions only on already-known, drafted tokens before it, all `k` positions' distributions can be computed simultaneously, exactly like a prefill), producing `p(x_i)` at each position. Positions are then accepted left to right: at position `i`, the drafted token `x_i` is accepted with probability `min(1, p(x_i)/q(x_i))`; the first rejection stops the walk, discarding everything drafted after it (since later drafted tokens were conditioned on the now-discarded rejected token and aren't a valid continuation of the accepted prefix), and a corrected token is sampled from the residual distribution `p_residual(x) = max(0, p(x)-q(x)) / sum_x' max(0, p(x')-q(x'))`. If every drafted token is accepted, one bonus token is sampled directly from the target's own distribution at the position right after the draft — already computed for free by the verify pass.

**Why this preserves the exact distribution.** Fix a position and consider the probability the algorithm emits some specific value `v`. There are two ways: (A) the draft proposes `v` and it's accepted — probability `q(v) * min(1, p(v)/q(v)) = min(p(v), q(v))`, by direct algebra on the `min`. (B) the draft proposes something else, it's rejected, and the residual resampling lands on `v`. The total rejection probability at this step is `sum_x max(0, q(x)-p(x))`, and — because both `p` and `q` are normalized distributions summing to 1 — this quantity is exactly equal to `sum_x max(0, p(x)-q(x))` (the "excess" mass `q` has beyond `p` somewhere must equal the "deficit" mass `q` has relative to `p` elsewhere, since both integrate to 1). Multiplying the rejection probability by the residual distribution's density at `v`, `max(0, p(v)-q(v)) / sum_x' max(0, p(x')-q(x'))`, the denominator cancels the rejection-probability factor exactly, leaving path B's contribution as simply `max(0, p(v)-q(v))`.

Summing paths A and B: `min(p(v),q(v)) + max(0, p(v)-q(v))`. Case-splitting: if `p(v) >= q(v)`, this is `q(v) + (p(v)-q(v)) = p(v)`. If `p(v) < q(v)`, this is `p(v) + 0 = p(v)`. Either way, the total probability of emitting `v` is exactly `p(v)` — the target model's own distribution — regardless of what the draft distribution `q` actually was. This is the crucial, subtle result: correctness of the output distribution is completely decoupled from draft-model quality; a bad draft model only costs speed (more rejections, fewer tokens accepted per round), never correctness. That decoupling is exactly why labs deploy speculative decoding as a pure speedup with no quality asterisk, unlike quantization or using a smaller model directly.

## Q16 (Coding): Implement the accept/reject sampling step of speculative decoding correctly, given target and draft probability vectors for one drafted token, and demonstrate empirically that repeated sampling reproduces the target distribution.

```python
import numpy as np

def spec_accept_reject_step(p_target: np.ndarray, q_draft: np.ndarray,
                             drafted_token: int, rng: np.random.Generator):
    """Returns (accepted: bool, emitted_token: int).
    If rejected, emitted_token is resampled from the exact residual distribution."""
    accept_prob = min(1.0, p_target[drafted_token] / q_draft[drafted_token])
    if rng.random() <= accept_prob:
        return True, drafted_token

    residual = np.clip(p_target - q_draft, a_min=0.0, a_max=None)
    residual_sum = residual.sum()
    residual = residual / residual_sum
    corrected = rng.choice(len(residual), p=residual)
    return False, corrected


def empirical_check(p_target, q_draft, n_trials=200_000, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(p_target))
    for _ in range(n_trials):
        drafted = rng.choice(len(q_draft), p=q_draft)
        _, emitted = spec_accept_reject_step(p_target, q_draft, drafted, rng)
        counts[emitted] += 1
    empirical = counts / n_trials
    print("target distribution :", np.round(p_target, 4))
    print("empirical distribution:", np.round(empirical, 4))
    print("max abs deviation    :", np.max(np.abs(empirical - p_target)))


if __name__ == "__main__":
    p_target = np.array([0.5, 0.3, 0.15, 0.05])   # target model's true distribution
    q_draft  = np.array([0.2, 0.2, 0.3, 0.30])    # a deliberately mismatched draft model
    empirical_check(p_target, q_draft)
```

Even with a deliberately poorly-matched draft distribution, the empirical distribution of emitted tokens converges to `p_target`, not `q_draft` and not some blend of the two — the direct empirical confirmation of the Q15 proof. Note the implementation detail that actually makes this correct: the residual is computed from `p_target - q_draft` (clipped at zero), *not* from `p_target` alone and *not* conditioned on which specific token was drafted-then-rejected — using the wrong residual (e.g., resampling from `p_target` directly, ignoring `q_draft` entirely) would double-count the mass already captured by path A and bias the result.

## Q17: What determines the acceptance rate in speculative decoding, and how does that map onto realistic speedup numbers?

The acceptance probability at a position is `min(1, p(x)/q(x))` for whatever token `x` the draft proposed, so acceptance is high whenever the draft distribution closely tracks the target distribution on the tokens the draft actually proposes. Two levers dominate: draft-model quality/alignment (a draft model trained on similar data, or explicitly distilled from the target, tends to place mass on the same tokens the target would prefer in the same context, giving high `p(x)/q(x)` ratios; an unrelated generic small model diverges more often, giving lower ratios and more rejections), and task predictability (low-entropy, highly formulaic spans — boilerplate code, repeated structure, closing syntax — have the *target's own* distribution concentrated on one or two tokens, so almost any reasonable draft model agrees with it and acceptance is high regardless of draft sophistication; high-entropy spans — creative writing, the specific digit in an arithmetic result, the first word after a genuinely open prompt — spread the target's own mass across many plausible tokens, making it much more likely the draft picks a *plausible but not the specific* token the target would have generated, driving rejections up).

The resulting speedup is bounded by both the acceptance rate and the draft/target cost ratio: for average acceptance `alpha` and draft length `k`, expected tokens emitted per round is approximately `(1 - alpha^(k+1))/(1-alpha)`, and if a draft step costs a fraction `c` of a target step, wall-clock cost per round is roughly `k*c + 1` target-equivalents — so speedup is `E[tokens]/(k*c+1)`. Both terms matter: a great draft model with high `alpha` but high `c` (nearly as expensive as the target) buys little; a cheap draft model with low `alpha` wastes most of its drafted tokens on rejections. This is why reported speedups vary heavily by workload — code generation and structured output (high, predictable `alpha`) see the largest gains; open-ended creative or multi-step reasoning generation (lower, more variable `alpha`) sees smaller gains — and why any specific multiplier quoted for "speculative decoding speedup" should be read as workload-conditional, not a universal constant.

## Q18: Explain why prefill and decode have fundamentally different optimal hardware/batching characteristics, and what motivates disaggregating them onto separate hardware pools.

Prefill processes an entire prompt's tokens in one parallel pass — no position depends on another position's *output* within that pass (only on causal masking of *inputs*, which doesn't block parallel computation once the full prompt is known) — so it's a large, dense, high-arithmetic-intensity matmul: **compute-bound**, and it's the regime GPUs are naturally efficient at. Decode generates one token at a time, each step depending on the previous step's sampled output, with a batch-but-effectively-single-position matmul against the full weight set and KV cache — tiny arithmetic intensity relative to data moved: **memory-bandwidth-bound**, and the only lever to raise utilization is batching many sequences' single-token steps together.

These are close to opposite optimization targets: prefill wants a lot of parallel work briefly, and barely benefits from bigger batches once one prompt already saturates the GPU; decode wants sustained, high concurrent low-intensity work, and is largely insensitive to any individual sequence's prompt length once its cache is resident. Colocating them on the same hardware, interleaved via continuous batching, creates interference: a prefill chunk injected into an otherwise-fast decode iteration inflates that iteration's latency, degrading inter-token latency for every already-streaming request riding along in the same batch (file 005 Section 2).

Disaggregation — routing new requests to a prefill-optimized pool that computes the prompt's KV cache and first token, then transferring that cache over fast interconnect to a decode-optimized pool that takes over token-by-token generation — removes this interference entirely (different hardware, no shared iteration schedule) and lets each pool be sized and autoscaled independently to match the actual prefill-vs-decode demand mix, which need not move in lockstep (a burst of new conversations stresses prefill capacity; a large number of long, slow-streaming sessions stresses decode capacity). The cost is the added latency and engineering complexity of the KV-cache transfer step and running two differently-shaped fleets — a trade-off that becomes worthwhile at large enough scale and heterogeneous enough traffic mix, but isn't automatically the right choice for every deployment size.

## Q19 (Coding): Given hardware cost and achieved throughput, compute cost-per-million-tokens, and show how batch size affects the result through a simple decode-cost model.

```python
def cost_per_million_tokens(dollars_per_gpu_hour: float, num_gpus: int,
                             tokens_per_second: float) -> float:
    dollars_per_second = dollars_per_gpu_hour * num_gpus / 3600.0
    tokens_per_dollar = tokens_per_second / dollars_per_second
    return 1_000_000 / tokens_per_dollar


def decode_tokens_per_second(batch_size: int, per_step_latency_ms: float) -> float:
    """Simplified model: one decode step advances every sequence in the batch by one
    token; per_step_latency_ms grows sub-linearly with batch size while memory-bandwidth
    bound, then roughly linearly once compute-bound. Model that saturation crudely."""
    compute_bound_batch = 64   # illustrative point where the GPU stops being idle
    if batch_size <= compute_bound_batch:
        effective_latency_ms = per_step_latency_ms          # bandwidth-bound: ~flat
    else:
        scale = batch_size / compute_bound_batch
        effective_latency_ms = per_step_latency_ms * (scale ** 0.5)  # sub-linear growth
    tokens_per_step = batch_size
    steps_per_second = 1000.0 / effective_latency_ms
    return tokens_per_step * steps_per_second


if __name__ == "__main__":
    for batch in (1, 8, 32, 64, 128, 256):
        tps = decode_tokens_per_second(batch, per_step_latency_ms=20.0)
        cost = cost_per_million_tokens(dollars_per_gpu_hour=28.0, num_gpus=8,
                                        tokens_per_second=tps)
        print(f"batch={batch:>4} -> {tps:8.1f} tok/s -> ${cost:6.3f} / 1M tokens")
```

The output shows cost-per-token dropping sharply as batch size grows through the memory-bandwidth-bound region (throughput scales almost linearly with batch size while per-step latency stays flat), then flattening out once the batch becomes large enough to be compute-bound (per-step latency starts growing with batch size too, eating into the throughput gain) — the concrete numeric shape of exactly the TTFT/TPOT-vs-throughput trade-off surface described qualitatively in file 005 Section 3: pushing batch size further past the compute-bound knee keeps improving aggregate cost-per-token only slowly, while degrading individual-request TPOT steadily, which is why real systems pick an operating point on this curve rather than maximizing batch size unconditionally.

## Q20: Why does a production LLM product typically serve through multiple models rather than a single model, and what are the main axes along which that tiering happens?

Three largely independent pressures push toward a multi-model system rather than one endpoint. **Cost-tiering by task difficulty**: most traffic is easy (short factual questions, simple rewrites, boilerplate completions) and sending it to the largest, most expensive model wastes money and latency budget for no quality benefit, while a small fraction of traffic is genuinely hard and needs the largest model's extra capability to avoid a silent quality failure — no single model sits at the cost/quality optimum across this whole distribution. **Latency-tiering**: some product surfaces (autocomplete, voice turn-taking) have hard latency budgets that only the smallest/fastest models can meet at all, independent of whether a larger model would answer better — for these, "fits the budget" dominates "is the best possible answer." **Task specialization**: embedding models for retrieval, moderation/safety classifiers for guardrails, code-specialized models for coding surfaces, and small classifiers for narrow high-volume tasks (including routing itself) are each better suited to their specific job than a general-purpose chat model would be, both in quality and in cost.

The result is that a mature LLM product is a *pipeline* of purpose-built models coordinated by a routing layer, not a single endpoint — mirroring, at the product-system level, the same conditional-compute idea that shows up one level down inside a single model's MoE routing (token-level expert gating) and one level further down inside GPT-5's confirmed fast/reasoning router design (`../GPT/010_GPT5_Series.md` Section 2) — the same underlying idea (a gate deciding which sub-network or which whole model handles a unit of work) recurring at every granularity of the stack, from a single layer's experts, to a whole query's model tier, to a whole product's model portfolio.
