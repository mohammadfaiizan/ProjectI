## Interview Questions — Part 2

## Q1: Explain the GGUF / llama.cpp k-quant approach. When would you choose it over a GPTQ/AWQ-quantized model for a GPU serving deployment?

GGUF's k-quant formats (`Q4_K_M`, `Q5_K_S`, etc.) target CPU and consumer/edge inference rather than high-throughput multi-tenant GPU serving. Mechanically, weights are grouped into small blocks (e.g. 32 weights) each with its own scale, and blocks are further grouped into superblocks whose shared scale-of-scales is itself quantized, capturing most of the benefit of fine-grained per-block scaling without paying full fp16-per-block metadata overhead. Different tensors within one model file get different bit-widths according to empirically-tuned recipes (the `_S`/`_M`/`_L` suffixes) that allocate more bits to layers found to be more quantization-sensitive — a lighter-weight, heuristic cousin of AWQ/GPTQ's calibration-driven precision allocation, tuned by broad empirical sweeps rather than derived per-model from calibration data, and often requiring no calibration run at all.

You would *not* choose GGUF for a high-throughput multi-tenant GPU serving deployment: GPTQ/AWQ-quantized weights on a mature GPU-serving stack (vLLM, TensorRT-LLM) with continuous batching and PagedAttention will give better achieved throughput and generally a better quality-per-bit trade-off for that specific hardware and traffic pattern, because GPTQ/AWQ calibrate against real activation statistics for the exact target model rather than relying on cross-model empirical heuristics. GGUF is close to the default choice instead when the deployment target is CPU-only or unified-memory consumer hardware (a laptop, an on-device deployment, Apple Silicon), when there's no GPU-based calibration pipeline available, or when portability and single-file simplicity matter more than squeezing out the last few points of quality-per-bit for one specific serving stack.

## Q2 (Coding): Implement a simplified, GPTQ-style column-wise quantizer with greedy error compensation (a toy Optimal-Brain-Compression-lite), and compare its reconstruction error against naive round-to-nearest.

```python
import numpy as np

def rtn_quantize(W: np.ndarray, bits=4):
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W)) / qmax
    W_q = np.clip(np.round(W / scale), -qmax, qmax)
    return W_q * scale

def gptq_lite_quantize(W: np.ndarray, X: np.ndarray, bits=4, damp=1e-2):
    """W: [out, in] weight matrix. X: [n_samples, in] calibration activations.
    Quantizes columns left to right, compensating remaining unquantized columns
    using the (damped) inverse Hessian H = X^T X, approximating GPTQ's core loop."""
    out_dim, in_dim = W.shape
    qmax = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(W)) / qmax   # single tensor-wide scale for simplicity

    H = X.T @ X / X.shape[0]
    H += damp * np.eye(in_dim) * np.mean(np.diag(H))   # damping for numerical stability
    H_inv = np.linalg.inv(H)

    W_work = W.copy()
    W_q = np.zeros_like(W)

    for col in range(in_dim):
        w_col = W_work[:, col]
        q_col = np.clip(np.round(w_col / scale), -qmax, qmax) * scale
        error = w_col - q_col                       # per-output-channel quantization error
        W_q[:, col] = q_col

        # Redistribute this column's error into remaining, not-yet-quantized columns,
        # weighted by the inverse-Hessian's coupling between this column and the rest --
        # this is the "compensation" step; a simplified stand-in for GPTQ's exact update.
        if col + 1 < in_dim:
            h_inv_row = H_inv[col, col + 1:]
            denom = H_inv[col, col] if H_inv[col, col] != 0 else 1e-6
            correction = np.outer(error, h_inv_row / denom)
            W_work[:, col + 1:] -= correction

    return W_q


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    out_dim, in_dim, n_samples = 8, 64, 512
    W = rng.normal(0, 1, size=(out_dim, in_dim)).astype(np.float64)
    X = rng.normal(0, 1, size=(n_samples, in_dim)).astype(np.float64)

    W_rtn = rtn_quantize(W, bits=4)
    W_gptq = gptq_lite_quantize(W, X, bits=4)

    Y_true = X @ W.T
    err_rtn = np.linalg.norm(X @ W_rtn.T - Y_true) / np.linalg.norm(Y_true)
    err_gptq = np.linalg.norm(X @ W_gptq.T - Y_true) / np.linalg.norm(Y_true)
    print(f"RTN relative output error : {err_rtn:.4f}")
    print(f"GPTQ-lite relative output error: {err_gptq:.4f}")
```

This should show `err_gptq < err_rtn`: even this simplified compensation loop reduces the *output*-level reconstruction error relative to independently rounding every weight, because later columns are nudged to counteract the error already introduced by earlier columns' quantization, evaluated against the actual calibration activations `X` rather than against the weights in isolation. Real GPTQ additionally uses per-group scales (not the single tensor-wide scale used here for simplicity) and a numerically more careful Cholesky-based update order; the compensation *principle* — quantize, measure error, redistribute using inverse-Hessian coupling, move on — is what this toy implementation is meant to make concrete.

## Q3: Why is activation quantization harder than weight quantization, and what does FP8 buy you that INT8 doesn't?

Weights are static: their distribution can be measured once, offline, with no data dependency, and (empirically) they tend to be comparatively well-behaved — smoothly distributed, without extreme, structured outliers. Activations are the output of nonlinear operations (attention, layernorm, SwiGLU/GELU) applied to data-dependent inputs, and they exhibit far more extreme, *structured* outliers — specific channels in specific layers routinely produce values 10-100x the typical magnitude in that same tensor, a well-documented phenomenon in the LLM quantization literature. A single such outlier, under naive min/max calibration, sets the scale for the entire tensor and wastes most of the integer grid's resolution on the common-case, small-magnitude values — exactly the problem AWQ's channel-scaling trick and the related SmoothQuant technique are built to mitigate by migrating scale from problematic activation channels onto the corresponding weights via an exact rescaling identity.

FP8 (`E4M3`/`E5M2` on H100-class tensor cores) is a floating-point format, not integer — its non-uniform, exponent-driven dynamic range lets it represent a large-magnitude outlier and a small typical value with proportionally similar *relative* precision (unlike int8's uniform grid spacing, which gives every representable value the same *absolute* spacing regardless of magnitude). This makes fp8 considerably more forgiving of exactly the outlier problem that plagues int8 activation quantization, which is why fp8 (for weights, activations, and increasingly the KV cache itself) has become a preferred choice for high-throughput GPU serving on hardware that supports it natively — it captures much of int8's speed/memory benefit with meaningfully less outlier-driven accuracy risk, without requiring an AWQ/SmoothQuant-style rescaling workaround to make activation quantization viable at all.

## Q4 (Coding): Given a target quantization scheme, compute the maximum batch size at a given average context length for a specific GPU node, and show the effect of moving from fp16 to int4 weights.

```python
GiB = 1024 ** 3

def weight_bytes(num_params: float, bytes_per_param: float) -> float:
    return num_params * bytes_per_param

def kv_cache_bytes_per_seq(n_layers, n_kv_heads, head_dim, seq_len, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_value

def max_batch_size(total_hbm, weight_bytes_, activation_reserve, per_seq_kv_bytes):
    available = total_hbm - weight_bytes_ - activation_reserve
    return max(0, int(available // per_seq_kv_bytes)) if per_seq_kv_bytes > 0 else 0

if __name__ == "__main__":
    total_hbm = 8 * 80 * GiB          # 8xH100
    num_params = 70e9
    activation_reserve = 20 * GiB
    per_seq_kv = kv_cache_bytes_per_seq(n_layers=80, n_kv_heads=8, head_dim=128,
                                         seq_len=8_000)   # GQA, 8K avg context

    for name, bytes_per_param in [("fp16", 2.0), ("int8", 1.0), ("int4", 0.5)]:
        w_bytes = weight_bytes(num_params, bytes_per_param)
        batch = max_batch_size(total_hbm, w_bytes, activation_reserve, per_seq_kv)
        print(f"{name:<5} weights: {w_bytes/GiB:6.1f} GiB -> max batch size: {batch}")
```

Moving from fp16 (140 GiB of weights) to int4 (35 GiB) frees roughly 105 GiB of HBM that converts directly into additional concurrent-sequence capacity under the same per-sequence KV-cache cost — this is exactly the point made in file 002 Section 8: the *memory-freed-for-batching* effect of weight quantization is frequently larger in practice than the raw per-token-latency improvement from a smaller/faster matmul, because it multiplies through the entire batch-size-dependent throughput equation, not just a single forward pass's cost.

## Q5: Compare sliding-window (local) attention with GQA/MQA and MLA as KV-cache-reduction techniques. What do they trade off differently?

GQA/MQA and MLA both attack the `n_kv_heads x head_dim` (or, for MLA, the latent width) term in the KV-cache formula — they reduce the *per-token* cache cost, and that reduction applies uniformly regardless of how long the sequence eventually gets; a sequence at 1M tokens still pays the full `seq_len` multiplication, just against a smaller per-token base cost. Sliding-window attention instead caps the *effective* `seq_len` term directly: each layer only ever attends to the most recent `W` tokens, discarding (evicting) older cache entries once they fall outside the window, so cache size for a windowed layer is bounded by `W` regardless of how long generation continues — a fundamentally different lever, operating on the "seq_len" axis rather than the "per-token width" axis.

The trade-off is also different in kind: GQA/MLA's cost is a *quality* tax paid on every token, uniformly, in exchange for a *cache-size* benefit that scales with everything (short and long contexts alike, though it only matters once context is long enough for cache to be the bottleneck at all). Sliding-window attention's cost is a genuine *capability* loss — anything requiring a dependency older than `W` tokens back is architecturally invisible to a windowed layer, not just approximated — in exchange for a cache benefit that specifically targets very long contexts (windowed attention buys you nothing over full attention once sequences are already shorter than `W`). In practice, models combining these ideas (e.g., interleaving a few full-attention layers with many windowed layers, as in Mistral-style architectures) are making a bet that most long-range dependencies can be captured by the occasional full-attention layer while the bulk of layers get away with a bounded window — a design decision made at pretraining time, not something a serving engineer can retrofit onto an already-trained dense/GQA model without retraining.

## Q6 (Coding): Implement a KV-cache-budget-aware admission control function that decides whether a new request can be safely admitted into a running continuous-batching server, given the current pool's committed cache and the new request's expected length.

```python
class KVCacheAdmissionController:
    def __init__(self, total_kv_budget_bytes: int, bytes_per_token_per_seq: int,
                 safety_margin: float = 0.10):
        """bytes_per_token_per_seq: 2 * n_layers * n_kv_heads * head_dim * bytes_per_value
        (file 001's per-token-per-layer term already multiplied through layers)."""
        self.total_budget = total_kv_budget_bytes
        self.bytes_per_token = bytes_per_token_per_seq
        self.safety_margin = safety_margin   # reserve headroom against underestimated growth
        self.committed_bytes = 0
        self.active_requests: dict[int, int] = {}   # request_id -> reserved tokens

    def _usable_budget(self) -> int:
        return int(self.total_budget * (1 - self.safety_margin))

    def can_admit(self, expected_max_tokens: int) -> bool:
        needed = expected_max_tokens * self.bytes_per_token
        return self.committed_bytes + needed <= self._usable_budget()

    def admit(self, request_id: int, expected_max_tokens: int) -> bool:
        if not self.can_admit(expected_max_tokens):
            return False
        self.active_requests[request_id] = expected_max_tokens
        self.committed_bytes += expected_max_tokens * self.bytes_per_token
        return True

    def release(self, request_id: int):
        tokens = self.active_requests.pop(request_id, 0)
        self.committed_bytes -= tokens * self.bytes_per_token


if __name__ == "__main__":
    GiB = 1024 ** 3
    bytes_per_token = 2 * 80 * 8 * 128 * 2   # GQA(8), 80 layers, head_dim=128, fp16
    controller = KVCacheAdmissionController(total_kv_budget_bytes=100 * GiB,
                                             bytes_per_token_per_seq=bytes_per_token)

    requests = [(1, 4_000), (2, 32_000), (3, 4_000), (4, 128_000)]
    for rid, max_tok in requests:
        ok = controller.admit(rid, max_tok)
        print(f"request {rid} (max {max_tok} tok): {'admitted' if ok else 'REJECTED - queue it'}")
```

This models the real admission-control decision a continuous-batching scheduler has to make beyond a simple slot count (file 003 Section 7): admitting a request whose *worst-case* KV-cache reservation would push committed usage past the usable budget risks either an out-of-memory failure or forced preemption of already-running requests mid-generation, so a conservative admission controller reserves against the request's expected maximum length up front (with a safety margin for underestimation) rather than admitting optimistically and hoping the average case saves it — a real production trade-off between admitting more concurrency optimistically (higher throughput, risk of preemption) versus admitting conservatively (lower risk, potentially underutilized capacity when actual generation lengths turn out shorter than reserved).

## Q7: Explain chunked prefill and why mixing a large prefill into an ongoing decode batch causes a latency spike. How does chunking address it?

In continuous batching, prefill for a new request and decode steps for already-running requests can, in principle, be scheduled into the same iteration (or interleaved across iterations) since the scheduler operates at forward-pass granularity, not per-request-type granularity. The problem: a full prefill over a long prompt (thousands of tokens processed in one large parallel matmul) is a much larger unit of compute than a single decode step (a handful of sequences advancing by one token each), and if that large prefill is injected wholesale into one iteration, the *entire* iteration's latency balloons to whatever the large prefill costs — every decode step for every other request riding along in that same iteration is delayed by exactly that amount, directly spiking inter-token latency for users who were mid-stream and had nothing to do with the new request.

Chunked prefill fixes this by splitting a long prompt's prefill into smaller pieces (chunks) sized to fit comfortably alongside an ongoing decode iteration's normal cost, and scheduling one chunk per iteration (or every few iterations) rather than the whole prefill in one shot — spreading the prefill's total cost across several iterations instead of concentrating it into one large spike. This does lengthen the *new* request's own time-to-first-token slightly relative to an uncontended full prefill (since its prefill now takes several iterations instead of one), but it protects every other concurrently-streaming request's inter-token latency from that spike — a direct, deliberate trade of one request's TTFT against every other active request's TPOT, tuned via the chunk-size parameter (file 003 Section 3, file 005 Section 2).

## Q8 (Scenario): You increased the maximum batch size on your decode servers to raise throughput, and now per-user streaming feels slightly slower even though your dashboards show improved GPU utilization and lower cost-per-token. Is this expected? How would you decide whether to keep the change?

Yes, this is exactly the expected behavior, not a bug: a larger decode batch means each iteration's matmul is bigger, so each individual token step takes somewhat longer in wall-clock terms even though more tokens are produced in aggregate per step — this is precisely the TTFT/TPOT-versus-throughput trade-off described in file 005 Section 3. Aggregate throughput and cost-per-token improving while per-user inter-token latency (TPOT) degrades is not a contradiction; they are different axes of the same underlying trade-off surface, and pushing batch size further generally keeps improving one while degrading the other, at least until the batch is large enough to become compute-bound, at which point *further* increases stop buying much additional throughput while continuing to cost TPOT (Q19 of Part 1's cost model shows this saturation numerically).

Whether to keep the change is not a purely technical question — it's a product decision about where on that trade-off surface this specific product should sit. The right process: quantify the actual TPOT regression in absolute terms (milliseconds per token, not just "feels slower"), compare it against the product's committed or implicit latency SLA/UX bar (a live conversational assistant has a much tighter tolerable-TPOT bar than a bulk batch-summarization job), and weigh that against the realized cost/throughput improvement. If the product has distinct latency-sensitive and throughput-sensitive traffic (e.g., interactive chat versus bulk API batch jobs), the better fix is often not to revert the change globally but to route those two traffic types to differently-tuned pools (a smaller max batch size for latency-sensitive traffic, a larger one for throughput-sensitive traffic) — turning a single global trade-off decision into a per-traffic-class one, which is usually the higher-leverage fix once the underlying cause is correctly diagnosed as an expected trade-off rather than a defect.

## Q9: Explain prompt/prefix caching and its relationship to PagedAttention's copy-on-write mechanism.

Prompt (prefix) caching skips recomputing the KV cache for a prefix of a request that the server has already processed before — most commonly a shared system prompt, a shared few-shot template, or a shared long document multiple requests query against. Mechanically, this is exactly the block-sharing/copy-on-write machinery from PagedAttention (file 003 Section 5): if two requests share an identical token prefix, their block tables can point at the same physical KV-cache blocks for that shared span, so the shared portion's prefill compute and its cache memory are both paid exactly once, no matter how many concurrent requests share it — right up until a given request's generation diverges from the shared prefix, at which point that request gets a private copy of just the block(s) it needs to write into (copy-on-write), while every other sharer remains unaffected.

Commercial LLM APIs that advertise a "prompt caching" discount (reduced per-token cost and reduced time-to-first-token for a cache hit on a previously-seen prefix) are exposing this exact server-side mechanism as a billed product feature, typically with a time-based eviction policy for cached entries and a strict exact-match requirement on the shared prefix (a single differing token anywhere in the prefix invalidates the cache hit from that point onward, since attention is causal and every later position genuinely depends on everything before it — there is no "fuzzy" or partial-match caching possible here without changing the actual computation). This is the same underlying mechanism as speculative decoding's *speedup* target in spirit — get more useful output per unit of expensive compute — but attacking the *redundant-prefill* half of the problem rather than the *sequential-decode* half; the two compose rather than substitute for each other (file 004 Section 7).

## Q10 (Coding): Implement PagedAttention's block table with allocate, append-token, fork (for shared prefixes), and copy-on-write-on-write semantics.

```python
BLOCK_SIZE = 4   # small for a legible demo; real systems use larger blocks (e.g. 16)

class BlockAllocator:
    def __init__(self, num_physical_blocks: int):
        self.free_blocks = list(range(num_physical_blocks))
        self.ref_counts = [0] * num_physical_blocks
        self.storage: dict[int, list] = {i: [None] * BLOCK_SIZE for i in range(num_physical_blocks)}

    def allocate(self) -> int:
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free(self, block_id: int):
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.storage[block_id] = [None] * BLOCK_SIZE
            self.free_blocks.append(block_id)


class Sequence:
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_table: list[int] = []
        self.length = 0

    def append_token(self, kv_pair):
        if self.length % BLOCK_SIZE == 0:
            self.block_table.append(self.allocator.allocate())
        block_id = self.block_table[-1]
        offset = self.length % BLOCK_SIZE
        if self.allocator.ref_counts[block_id] > 1:
            block_id = self._cow_copy(len(self.block_table) - 1)
        self.allocator.storage[block_id][offset] = kv_pair
        self.length += 1

    def _cow_copy(self, logical_block_idx: int) -> int:
        old_block = self.block_table[logical_block_idx]
        new_block = self.allocator.allocate()
        self.allocator.storage[new_block] = list(self.allocator.storage[old_block])
        self.allocator.free(old_block)
        self.block_table[logical_block_idx] = new_block
        return new_block

    def fork(self) -> "Sequence":
        child = Sequence(self.allocator)
        child.block_table = list(self.block_table)
        child.length = self.length
        for block_id in child.block_table:
            self.allocator.ref_counts[block_id] += 1
        return child


if __name__ == "__main__":
    allocator = BlockAllocator(num_physical_blocks=16)
    parent = Sequence(allocator)
    for i in range(6):                       # shared system prompt: 6 tokens
        parent.append_token(kv_pair=f"tok{i}")

    child_a = parent.fork()                  # two completions branch off the same prefix
    child_b = parent.fork()

    shared_block = parent.block_table[-1]
    print("ref count on shared tail block right after fork:", allocator.ref_counts[shared_block])

    child_a.append_token(kv_pair="A-continues")   # triggers CoW on the shared tail block
    print("ref count after child_a diverges        :", allocator.ref_counts[shared_block])
    print("child_b's tail block unaffected          :", allocator.storage[child_b.block_table[-1]])
```

The reference count on the shared tail block rises to 3 right after both forks (parent + 2 children all pointing at it), and drops back down the moment `child_a` writes into it — triggering a private copy for `child_a` only, leaving `parent` and `child_b` sharing the original, unmodified block. This is the exact mechanism underneath both multi-sample generation (best-of-N, beam search) and shared-system-prompt serving.

## Q11: Give the expected-speedup formula for speculative decoding and explain how you'd pick the draft length k for a real deployment.

For average per-token acceptance probability `alpha` and draft length `k`, the expected number of tokens emitted per round (accepted tokens plus the guaranteed bonus-or-corrected final token) is approximately `E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)`. If a draft-model step costs a fraction `c` of a target-model step, the wall-clock cost of one round is roughly `k*c + 1` target-model-equivalents, giving `speedup ≈ E[tokens] / (k*c + 1)`.

Picking `k` in practice means balancing two opposing effects visible directly in this formula: increasing `k` raises the numerator's ceiling (more tokens *could* be emitted per round) but the probability of surviving all `k` draft positions without a rejection falls as `alpha^k`, so beyond some point additional drafted tokens are very likely to be discarded anyway (drafted after the point the walk would have stopped), while still costing `c` per additional drafted token in the denominator. The optimal `k` is therefore a function of `alpha` specifically: a high-acceptance regime (predictable, formulaic generation, or a very well-aligned draft model) justifies a larger `k` since long draft runs are likely to mostly survive; a low-acceptance regime (creative or highly open-ended generation) wants a small `k`, since most rounds will terminate on an early rejection regardless, and additional drafted tokens beyond that point are wasted draft-model compute. Real deployments tune `k` empirically per draft/target pair and per representative workload — often ending up in the single digits — rather than deriving it purely analytically, since `alpha` itself varies within a single deployment across different spans of generation (a request can move between predictable and open-ended text mid-generation).

## Q12 (Scenario): You've deployed speculative decoding in production and the observed acceptance rate is far lower than what you measured in offline benchmarking. What would you check?

First, check for a **distribution mismatch between the benchmark and production traffic**: if the offline benchmark's evaluation set skewed toward predictable content (code completion, structured tasks) and production traffic is more open-ended (creative writing, diverse conversational turns), lower real-world acceptance is exactly expected per Q17/Part1's task-predictability argument — this isn't a bug, it's the offline number having measured the wrong workload.

Second, check for a **draft/target version mismatch**: if either the draft model or the target model was updated independently (a new fine-tune, a new quantized version of either) without re-validating the pair together, their distributions may have drifted apart even though each individually still "works" — speculative decoding's acceptance rate is a property of the *pair*, not either model alone, and this is a real operational hazard: any model-version update needs to trigger re-validation of every draft/target pairing that depends on it, not just an isolated quality check of the updated model by itself.

Third, check whether **sampling parameters differ** between offline benchmarking and production (temperature, top-p/top-k truncation) — the accept/reject math (Q15/Part1) assumes the draft and target probabilities being compared are the actual sampling distributions used; if production applies a different temperature or truncation to the target model's distribution than the benchmark did, but the draft model's distribution wasn't adjusted correspondingly, the effective `p(x)/q(x)` ratios shift and acceptance rates change in ways the offline number didn't anticipate.

Fourth, verify the **implementation itself is exact**: a subtle bug in the accept/reject step (e.g., not applying the same tokenization/detokenization boundary handling to both models, or a numerical precision mismatch between how the draft and target compute their probabilities in production versus in the benchmark harness) can silently shift the *effective* comparison being made, not just degrade acceptance rate but potentially bias the output distribution too — worth an explicit correctness re-check (per Q16/Part1's empirical-distribution-check pattern) against the production code path specifically, not just the benchmark's isolated implementation.

## Q13: When would you deliberately choose *not* to disaggregate prefill and decode onto separate hardware pools, even knowing the interference argument for doing so?

Disaggregation adds two real costs that only pay off at sufficient scale: the KV-cache transfer step between pools (an added latency cost on every request's path, which has to stay small relative to the TTFT it might otherwise save, and an added engineering surface — a fast, reliable transfer mechanism that itself needs monitoring and failure handling) and the operational complexity of running, capacity-planning, and autoscaling two differently-shaped fleets instead of one homogeneous one (twice the fleet-management surface, and a control-plane component to hand requests off between pools that doesn't exist at all in a colocated design).

At small scale or with a traffic mix that doesn't actually produce much prefill/decode interference in practice (e.g., traffic dominated by short prompts and short generations, where neither phase's cost is large enough to meaningfully disrupt the other even colocated, or traffic with naturally low concurrency where contention for shared iteration scheduling rarely arises), the interference this technique solves may simply not be large enough to justify the added complexity — a colocated design with well-tuned chunked prefill (Q7) can often absorb the interference adequately at that scale. Disaggregation earns its complexity specifically once traffic is large and heterogeneous enough (a real mix of many long-prompt and many long-generation requests, at high concurrency) that colocated interference becomes a measurable, recurring tax on tail latency — the same "does the added complexity pay for itself at this scale" judgment call that governs most infrastructure-disaggregation decisions generally, not something automatically correct at every deployment size.

## Q14: Explain TTFT and TPOT precisely, and give a concrete example of a product decision where you'd prioritize one over the other.

TTFT (time-to-first-token) is the delay between a request arriving and the first output token being returned — dominated by queueing delay plus the prefill pass, whose cost scales with prompt length since a longer prompt is genuinely a bigger compute-bound matmul. TPOT (time-per-output-token, or inter-token latency/ITL) is the steady-state delay between successive tokens once generation is underway — governed by decode-phase cost, which depends on current batch size and KV-cache read volume, and by whatever else (a colocated prefill chunk, file 005 Section 2) is competing for the same iteration schedule.

They're optimized somewhat independently because the levers that help one often hurt the other: bigger batches raise aggregate throughput and lower cost-per-token but slow each individual decode step, hurting TPOT; prioritizing new-request admission or larger chunked-prefill slices lowers TTFT for arrivals but can steal iteration time from already-streaming requests' decode steps, hurting their TPOT. A concrete product trade-off: a voice assistant with strict turn-taking needs a very low TTFT (the user is waiting, silently, for the first word — any perceptible delay before speech starts feels broken) but can tolerate a moderately higher TPOT as long as the streamed speech-synthesis pipeline downstream can keep pace with token arrival — so that product would deliberately bias scheduling toward minimizing queueing/prefill delay for new requests even at some cost to steady-state per-token pacing. Conversely, a code-completion IDE plugin streaming a long multi-line suggestion cares enormously about TPOT (a visibly stuttering, uneven stream of tokens is jarring while reading code appear character by character) and comparatively less about TTFT beyond a basic "feels instant" threshold — that product would bias toward smooth, consistent per-token pacing even if it means slightly higher initial latency under load.

## Q15 (Coding): Implement a cascade router that tries a cheap model first and escalates to an expensive model only when a confidence signal falls below a threshold, and reason about the cost/quality trade-off the threshold controls.

```python
from dataclasses import dataclass

@dataclass
class ModelResponse:
    text: str
    confidence: float   # e.g. mean token log-prob transformed to [0, 1], or a self-reported score
    cost_units: float

class CascadeRouter:
    def __init__(self, cheap_model, expensive_model, confidence_threshold: float):
        self.cheap_model = cheap_model
        self.expensive_model = expensive_model
        self.threshold = confidence_threshold
        self.stats = {"cheap_only": 0, "escalated": 0}

    def route(self, query: str) -> tuple[ModelResponse, str]:
        cheap_resp = self.cheap_model.generate(query)
        if cheap_resp.confidence >= self.threshold:
            self.stats["cheap_only"] += 1
            return cheap_resp, "cheap"

        self.stats["escalated"] += 1
        expensive_resp = self.expensive_model.generate(query)
        return expensive_resp, "expensive"

    def escalation_rate(self) -> float:
        total = self.stats["cheap_only"] + self.stats["escalated"]
        return self.stats["escalated"] / total if total else 0.0


class _FakeModel:
    """Illustrative stand-in: real deployments call an actual model's generate endpoint
    and derive `confidence` from that call's own signal (log-probs, a verifier, etc.)."""
    def __init__(self, cost_units, confidence_fn):
        self.cost_units = cost_units
        self.confidence_fn = confidence_fn

    def generate(self, query: str) -> ModelResponse:
        conf = self.confidence_fn(query)
        return ModelResponse(text=f"response to: {query}", confidence=conf,
                              cost_units=self.cost_units)


if __name__ == "__main__":
    cheap = _FakeModel(cost_units=1, confidence_fn=lambda q: 0.9 if len(q) < 40 else 0.4)
    expensive = _FakeModel(cost_units=20, confidence_fn=lambda q: 0.95)

    router = CascadeRouter(cheap, expensive, confidence_threshold=0.7)
    queries = ["what is 2+2?", "explain the tradeoffs of disaggregated prefill/decode serving in depth"]
    for q in queries:
        resp, tier = router.route(q)
        print(f"[{tier:<9}] conf={resp.confidence:.2f} query={q!r}")
    print("escalation rate:", router.escalation_rate())
```

The threshold is the single knob controlling the cost/quality trade-off directly: raising it escalates more traffic to the expensive model (higher cost, lower risk of a silently-degraded cheap-model answer on a query that actually needed more capability), lowering it keeps more traffic on the cheap model (lower cost, higher risk of exactly the silent-quality-failure mode discussed in file 006 Section 2 — a hard query answered by the cheap model with no visible failure signal at all). Picking the threshold is not a purely statistical exercise (e.g., "maximize accuracy on a labeled validation set") because the two error types the threshold trades off — wasting cost on easy queries versus silently degrading quality on hard ones — are not equally costly to the business, and a defensible threshold choice requires an explicit, deliberate statement of that asymmetric cost, not just an accuracy-maximizing default.

## Q16: Explain the asymmetric cost of mis-routing in a production router, and sketch an evaluation methodology that avoids conflating "component capability" with "routing quality."

Mis-routing has two failure directions with very different visibility and cost. Routing an easy query to the expensive tier wastes cost and latency — a *visible*, infrastructure-measurable failure (shows up immediately in cost and latency dashboards). Routing a hard query to the cheap tier degrades answer quality — often *silently*: the user has no signal their query warranted more capability, and the failure only shows up, if at all, in downstream task-success metrics, which are noisier and slower to observe than an infra dashboard. Treating both as one blended "routing accuracy" metric implicitly assumes they're equally costly, which is essentially never true for a real product — a wrong answer on a high-stakes agentic action is far more costly than a slightly-too-expensive answer on a trivial query — so the tuning target has to keep these as separate, explicitly-weighted currencies (file 006 Section 2's framing).

An evaluation methodology that avoids conflating component capability with routing quality: rather than reporting one aggregate score for "the routed system," construct an **oracle labeling** — for a sample of real queries, run *both* tiers, grade each tier's output independently against a fixed quality bar (via human grading or a calibrated LLM-judge comparison), and label each query with whether escalation was *actually necessary* to clear that bar (i.e., did the cheap tier's answer already meet the bar, making escalation wasted cost; or did only the expensive tier's answer meet it, making escalation necessary). This decomposes cleanly into three separately reportable numbers: the cheap tier's own capability (its pass rate against the bar, measured independently of routing), the expensive tier's own capability (likewise), and the router's actual decision quality (did it escalate exactly the queries the oracle labeling says needed it, no more and no fewer) — avoiding the trap of reporting one blended benchmark score that entangles all three and gives no way to tell whether a low score reflects a weak cheap model, a weak router, or both. This is precisely the measurement gap flagged, but not resolved by any public disclosure, for GPT-5's own router in `../GPT/010_GPT5_Series.md` Section 8/12 — a real, unresolved industry-wide evaluation-methodology problem, not a solved one.

## Q17: Why is canary deployment for a new LLM version harder than for a typical microservice, and what would a rigorous canary process for a model-version change include?

A typical microservice canary compares clean, largely objective signals: error rate, latency, resource utilization — a regression usually announces itself as an explicit failure (an exception, a timeout, a 500 response). An LLM version change's most dangerous regressions are exactly the ones that produce **no explicit error at all**: a new checkpoint (or a new quantization scheme, or a new prompt template) can produce fluent, well-formatted, confidently-stated, *wrong* or subtly-worse answers, with latency and error-rate metrics looking perfectly healthy throughout. Standard infra-level canary signals are necessary but not sufficient for catching this failure mode — they'll catch a crash or a latency regression, but they are blind to a quality regression by construction, since "quality" isn't a signal those metrics measure at all.

A rigorous canary process therefore needs to layer quality-specific signals on top of the standard infra ones: (1) the standard infra-level canary comparison (latency, error rate, throughput) between the new and previous version on a small, limited traffic split, exactly as for any service; (2) an automated quality-comparison signal on sampled canary traffic — either an LLM-judge comparing the new version's outputs against the previous version's outputs on the *same* queries (a paired comparison, which is more statistically sensitive than comparing absolute scores independently), or task-specific downstream success metrics where available (did an agentic trace using the new version actually complete its task); (3) targeted eval-suite regression checks specifically covering capabilities known to be sensitive to whatever changed (e.g., if the canary is a new quantization scheme, run the task-representative evals flagged in file 002 Section 7 and Part 1's Q9, not just perplexity); (4) a staged traffic ramp (small percentage, held for long enough to accumulate a statistically meaningful sample of both infra and quality signals, before progressively increasing) rather than an all-or-nothing cutover; and (5) an explicit, pre-agreed rollback trigger tied to the quality signals specifically, not just the infra ones — since without that, a quality regression with clean infra metrics has no automatic mechanism forcing a rollback at all, and can sit live in production far longer than an infra-visible failure would.

## Q18 (Scenario): After a canary rollout, all aggregate latency/error/throughput metrics look fine, but a growing number of users are complaining that answers "got worse." How do you investigate, given that your dashboards show nothing wrong?

This is exactly the failure mode Q17 describes, so the first move is to accept that the standard dashboards are the wrong instrument for this complaint and pivot to quality-specific investigation rather than re-checking infra metrics that have already been confirmed healthy.

Concretely: pull a sample of the actual canary-version transcripts users are complaining about (or, if specific complaints are available, the exact transcripts), and do a **paired comparison** against what the previous version would have produced on the identical queries — this immediately answers whether there's a real, reproducible quality difference or whether the complaints are noise/selection bias (users who complain are not a random sample; confirm the effect exists in a controlled, paired comparison before concluding anything). If a real difference exists, characterize *where* it concentrates — is it uniform across query types, or specific to a capability (structured output, factual recall, multi-step reasoning, tone/style) — since that points at what actually changed underneath (a training-recipe change, a quantization or serving-config change bundled into the same rollout, a prompt-template or system-prompt change that shipped alongside the model update, or a routing-tier shift if this canary also touched routing behavior, file 006 Section 4's router-drift concern). Cross-check whether the canary bundled *more* than just the model checkpoint — a canary is often not a pure single-variable change (it may have shipped alongside an updated system prompt, a new quantization scheme, or a routing config change), and conflating "new model version" with "everything that shipped in this canary" is a common misdiagnosis; isolate variables if the rollout process allows it. Finally, treat the fact that this surfaced via user complaints rather than any automated signal as itself an actionable finding — it's direct evidence the canary process (Q17) lacked an adequate automated quality gate for this specific regression, and the postmortem fix should include adding whatever quality-comparison signal would have caught this *before* full rollout, not just fixing the immediate regression.

## Q19: Sketch the monitoring signals you'd wire up across a multi-tier, multi-model production serving system to catch a serving-layer regression early, and explain what each is diagnostic of.

Organize signals by what failure mode each one actually distinguishes, since a flat list without that context invites exactly the metric-conflation problems raised throughout this module. **TTFT and TPOT, tracked at both p50 and p99, per tier/pool separately** — a p99-only TTFT spike with stable p50 points at tail contention (a burst of long prompts, KV-cache-triggered preemption) rather than systemic capacity; a whole-distribution shift points at genuine under-provisioning or hardware degradation. **GPU utilization and HBM/KV-cache occupancy per replica** — falling utilization alongside rising latency indicates a scheduling/batching regression (something is preventing the batch from filling, not a capacity shortfall); rising occupancy toward the ceiling alongside rising latency points directly at KV-cache pressure. **Queue depth and admitted-vs-waiting counts per pool** — a growing queue with otherwise-healthy per-replica latency is a straightforward under-provisioning signal calling for more replicas, distinct from a per-request performance problem. **Token throughput per GPU-second against cost-per-token** — a drop at constant traffic volume is the direct financial signal of an efficiency regression (a batching bug, a quantization rollback, a bad autoscaling decision), independent of whether any latency SLA has actually been breached. **Error and retry rate broken out by failure type** (infra timeouts/OOM versus output-validation/schema failures) — conflating these hides which one is actually happening and who owns the fix. **Routing distribution drift** — an unexplained shift in which fraction of traffic lands on which tier is often the earliest available signal of either a genuine traffic-composition change or a router miscalibration (especially dangerous if one tier was updated without the router being re-validated against the new capability gap, file 006's router-drift concern). **Online quality/eval signals** — sampled human grading, paired LLM-judge comparison against a reference version, or task-specific downstream success rate — the only category of signal that catches the silent-quality-degradation failure mode (Q17/Q18) that every purely infra-level signal above is structurally blind to, and therefore the signal most worth investing in early rather than treating as optional, precisely because it's the hardest to build and the easiest to skip.

## Q20 (Coding): Implement a function that computes the expected speculative-decoding speedup given an acceptance rate, draft length, and draft/target cost ratio, and use it to find the draft length that maximizes speedup for a given acceptance rate.

```python
def expected_tokens_per_round(alpha: float, k: int) -> float:
    """E[tokens emitted per round], including the guaranteed bonus/corrected token."""
    if alpha >= 1.0:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def expected_speedup(alpha: float, k: int, cost_ratio: float) -> float:
    """cost_ratio = draft_step_cost / target_step_cost (c << 1 for a much cheaper draft)."""
    round_cost_in_target_equivalents = k * cost_ratio + 1
    return expected_tokens_per_round(alpha, k) / round_cost_in_target_equivalents

def best_draft_length(alpha: float, cost_ratio: float, k_max: int = 20) -> tuple[int, float]:
    best_k, best_speedup = 1, expected_speedup(alpha, 1, cost_ratio)
    for k in range(2, k_max + 1):
        s = expected_speedup(alpha, k, cost_ratio)
        if s > best_speedup:
            best_k, best_speedup = k, s
    return best_k, best_speedup


if __name__ == "__main__":
    for alpha in (0.5, 0.7, 0.9, 0.97):
        k_opt, speedup = best_draft_length(alpha, cost_ratio=0.05)
        print(f"alpha={alpha:.2f} -> best k={k_opt:>2}, expected speedup={speedup:.2f}x")
```

Running this shows the optimal draft length `k` growing as `alpha` rises (a highly predictable, high-acceptance workload justifies drafting further ahead before the marginal token is likely wasted) and the achievable speedup rising sharply with `alpha` — quantifying precisely why speculative decoding's real-world payoff is so workload-dependent (Q17/Part1): a deployment serving mostly high-entropy, open-ended generation should expect a materially smaller `k_opt` and smaller realized speedup than one serving mostly predictable, structured generation, even with an identical draft/target model pair and cost ratio.
