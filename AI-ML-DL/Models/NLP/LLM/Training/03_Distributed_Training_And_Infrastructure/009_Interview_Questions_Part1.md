# Distributed Training and Infrastructure: Interview Questions, Part 1

## Q1: A 175B-parameter dense model needs to be trained. Why doesn't pure data parallelism across enough GPUs solve this, no matter how many GPUs you throw at it?

Pure data parallelism replicates the *entire* model state — parameters, gradients, and optimizer state — on every participating device; it only splits the batch. Adding more DP replicas increases aggregate throughput and lets you process a larger global batch, but it does nothing to reduce the memory footprint any single device must hold, because every replica still needs a full copy of everything. `..\GPT\003_GPT3.md` works this out concretely for GPT-3: 175B params at bf16/fp16 is 350GB of weights alone, and under mixed-precision Adam (fp32 master weights + fp32 first/second moment buffers) the optimizer state adds roughly another 8–12 bytes/param, landing at a combined model-state footprint on the order of 2.5–2.8TB (see `003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`'s `16Ψ`-bytes-per-parameter accounting for the precise breakdown). No accelerator has multiple terabytes of HBM; an H100 has 80GB. Adding 10,000 more pure-DP replicas doesn't help — each one still individually needs to fit 2.5+TB in 80GB, which is impossible regardless of replica count. The actual fix is to shard the *model state itself* across devices, via tensor parallelism (splitting weight matrices within a layer), pipeline parallelism (splitting layers across devices), and/or ZeRO (sharding the DP-replicated optimizer state, gradients, and optionally parameters across the DP group instead of replicating them). Pure DP is only viable once the per-replica model already fits — which for a 175B model requires TP and/or PP (and typically ZeRO on top) to have already been applied; DP alone is structurally insufficient at this scale, not merely suboptimal.

## Q2: Derive the communication cost of ring all-reduce, and explain why this makes DP's communication overhead independent of the number of replicas (asymptotically).

Ring all-reduce decomposes into a reduce-scatter phase followed by an all-gather phase, each taking `N-1` steps for `N` participants arranged in a ring. In the reduce-scatter phase, each device's gradient buffer of `Ψ` bytes is split into `N` equal chunks; over `N-1` steps, each device sends and receives one chunk per step, ending with exactly one fully-reduced chunk. In the all-gather phase, another `N-1` steps circulate the now-finished chunks so every device ends with the complete reduced buffer. Each step moves `Ψ/N` bytes; there are `2(N-1)` steps total (both phases); so the total bytes moved per device is:

```
bytes_per_device = 2 * (N - 1) * (Ψ / N) = 2Ψ * (N-1)/N
```

As `N → ∞`, `(N-1)/N → 1`, so `bytes_per_device → 2Ψ` — a constant, independent of `N`. This is the key scaling property: doubling the number of DP replicas does not double (or even meaningfully increase) the per-device communication volume for the gradient synchronization; it approaches a fixed cost determined only by model size. Contrast this explicitly with tensor parallelism's all-reduce (`001_Parallelism_Strategies_Data_Tensor_Pipeline.md`, Section 3.4), which is paid *every layer, every micro-batch*, both forward and backward, and cannot be overlapped with unrelated compute the way DP's once-per-step, backward-overlappable gradient all-reduce can (Section 2.3 of the same file) — TP's per-device cost also asymptotes to a constant in group size, but it's paid vastly more frequently and sits directly in the compute critical path, which is the actual reason TP is restricted to fast, low-latency NVLink domains while DP tolerates the much slower inter-node fabric perfectly well.

## Q3: (Coding) Write a function that estimates per-GPU memory usage in bytes for a given ZeRO stage, given total parameter count, DP degree, TP degree, PP degree, and micro-batch/sequence config (activation memory can use the simplified Megatron formula).

```python
def estimate_per_gpu_memory_bytes(
    total_params: float,
    dp_degree: int,
    tp_degree: int,
    pp_degree: int,
    zero_stage: int,          # 0 (plain DP), 1, 2, or 3
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    micro_batch_size: int,
    seq_len: int,
    bytes_per_param_bf16: float = 2.0,
    bytes_per_param_fp32: float = 4.0,
) -> dict:
    # Step 1: parameters actually resident on this device before ZeRO sharding,
    # after TP (intra-layer sharding) and PP (depth sharding) have already reduced it.
    local_params = total_params / (tp_degree * pp_degree)

    # Step 2: model-state bytes/param under mixed-precision Adam, before ZeRO sharding:
    #   2 (bf16 params) + 2 (bf16 grad) + 4 (fp32 master) + 4 (fp32 m) + 4 (fp32 v) = 16 bytes/param
    bf16_params = bytes_per_param_bf16
    bf16_grads = bytes_per_param_bf16
    fp32_master = bytes_per_param_fp32
    adam_m = bytes_per_param_fp32
    adam_v = bytes_per_param_fp32
    optimizer_state = fp32_master + adam_m + adam_v          # 12 bytes/param

    if zero_stage == 0:
        model_state_bytes = local_params * (bf16_params + bf16_grads + optimizer_state)
    elif zero_stage == 1:
        # optimizer state sharded across DP group; params + grads still fully replicated
        model_state_bytes = local_params * (bf16_params + bf16_grads) + \
                             local_params * optimizer_state / dp_degree
    elif zero_stage == 2:
        # + gradients also sharded; only params still fully replicated
        model_state_bytes = local_params * bf16_params + \
                             local_params * (bf16_grads + optimizer_state) / dp_degree
    elif zero_stage == 3:
        # everything sharded, including params (FSDP-equivalent)
        model_state_bytes = local_params * (bf16_params + bf16_grads + optimizer_state) / dp_degree
    else:
        raise ValueError("zero_stage must be 0, 1, 2, or 3")

    # Step 3: activation memory for this device's local layer count (post-PP sharding),
    # using the Megatron-style per-layer formula (unsharded by TP for simplicity here;
    # a full TP-aware version would additionally divide the appropriate terms by tp_degree).
    local_layers = num_layers / pp_degree
    activation_bytes_per_layer = micro_batch_size * seq_len * hidden_dim * (
        34 + 5 * num_heads * seq_len / hidden_dim
    )
    activation_bytes = activation_bytes_per_layer * local_layers

    total_bytes = model_state_bytes + activation_bytes
    return {
        "model_state_bytes": model_state_bytes,
        "activation_bytes": activation_bytes,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
    }


# Example: 70B model, TP=8, PP=4, DP=16, ZeRO-2, seq_len=4096, micro_batch=2
result = estimate_per_gpu_memory_bytes(
    total_params=70e9, dp_degree=16, tp_degree=8, pp_degree=4, zero_stage=2,
    num_layers=80, hidden_dim=8192, num_heads=64, micro_batch_size=2, seq_len=4096,
)
print(result["total_gb"], "GB")   # sanity check against an 80GB H100 budget
```

This mirrors `003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`'s worked example directly — the function makes explicit that `local_params` depends on the TP/PP grid chosen in `001`, and that ZeRO stage then determines how much of *that* local footprint gets further divided by the DP degree, which is the composition point most candidates miss when reasoning about this only qualitatively.

## Q4: Walk through exactly how Megatron-style tensor parallelism shards an FFN block, and explain precisely why an all-reduce is needed exactly once per block rather than not at all, or more than once.

The FFN is `Y = f(X W1) W2` with `f` an elementwise nonlinearity (GELU/SwiGLU). Megatron shards `W1` **column-wise**: device `t` holds columns `W1_t` of shape `[d_model, d_ff/T]`. Since `X` (the block's input activation) is identical on every device (replicated), each device computes `X W1_t` fully locally — no communication needed, because slicing a matmul's *output columns* requires no cross-device interaction; each device is independently computing a disjoint subset of output columns from data it already fully has. Applying `f` elementwise preserves this locality (`f(X W1_t)` needs nothing from other devices' columns). Then `W2` is sharded **row-wise**, matched to the column split of the previous step's output: device `t` holds rows `W2_t` of shape `[d_ff/T, d_model]`, and computes `Z_t = Y_t @ W2_t`. Crucially, `Z_t` has full `d_model` width but is only a **partial sum** over `1/T` of the contraction dimension (`d_ff`) — the true output `Z = sum_t Z_t` requires summing across devices. This is the one unavoidable cross-device dependency in the block, and it's resolved with exactly one all-reduce. Why not zero: because the row-parallel matmul's contraction dimension is split, and matrix multiplication's defining operation is precisely that sum over the contraction dimension — there is no way to avoid summing partial results from every shard without changing what's being computed. Why not more than one: because the sharding was deliberately chosen (column-then-row, matched at the split boundary) so that *every* other operation in the block (the elementwise nonlinearity) commutes cleanly with the existing split and needs no communication of its own — a different sharding choice (e.g., splitting both matmuls row-wise) would force cross-device communication at additional points, which is exactly why Megatron's specific column/row pairing is the design, not an arbitrary choice among many equally good options.

## Q5: Derive the pipeline bubble fraction for GPipe-style scheduling, and explain the two-sided lever this creates between bubble overhead and activation memory.

With `P` pipeline stages and `m` micro-batches per step, under the "all-forward-then-all-backward" schedule, the pipeline needs `P-1` steps to *fill* (before the last stage can even start its first forward pass, the first `P-1` stages must each have started) and, symmetrically, `P-1` steps to *drain* at the end. During the fill and drain phases, at least one stage is idle at any given moment (early in fill, later stages have nothing yet; late in drain, earlier stages have nothing left). The total wall-clock time for one step is proportional to `m + (P-1)` "slots" (the `m` micro-batches' worth of steady-state work, plus the `P-1` slots of fill/drain overhead), while the *useful* work done is proportional to `m`. So:

```
bubble_fraction = (P - 1) / (m + P - 1)
```

Two direct consequences: **increasing `m`** (more, smaller micro-batches per step) amortizes the fixed `P-1` overhead over more useful work, shrinking the bubble fraction — this is the primary lever a systems engineer has, and it costs nothing algorithmically. **Increasing `P`** (more pipeline stages) directly increases the fixed overhead term for the same `m`, so bubble fraction grows with `P` — meaning PP degree should be chosen as the *minimum* sufficient to fit memory constraints, not maximized. The catch, and the reason this is a genuine two-sided trade rather than "just increase `m` arbitrarily": under GPipe's naive schedule, every stage must hold activations for **all** `m` in-flight micro-batches simultaneously (since no backward pass starts until every micro-batch's forward has completed), so activation memory scales linearly with `m` — pushing the bubble toward zero directly inflates activation memory. The 1F1B schedule (`001`, Section 4.2) breaks this specific coupling by interleaving forward and backward per micro-batch, bounding in-flight activation sets to roughly `P` rather than `m`, which is precisely why 1F1B (not GPipe's schedule) is the practical default: it lets `m` be pushed up to shrink the bubble without inflating activation memory in lockstep.

## Q6: Scenario — your 512-GPU training job just stalled. No crash, no error message, GPU utilization dashboards show every GPU at roughly 0% compute. Walk through how you'd diagnose this.

First, resist the instinct to assume it's a single root cause and start ruling things in/out cheaply, in order of cost. **Step 1:** check whether any process actually died — `nvidia-smi` and process-liveness checks across all 512 ranks (or a sample, if full coverage is slow) to rule out an OOM-killed or segfaulted rank whose death left the rest of the job waiting indefinitely on a collective that will never complete (`008_Debugging_Distributed_Training_Failures.md`, Section 2). If a rank is dead, the fix path is node replacement plus resume-from-checkpoint (`006`), not further hang diagnosis. **Step 2:** if every process is alive but idle, get a stack trace from every rank (`py-spy dump` or equivalent) simultaneously. The diagnostic signature to look for: most ranks blocked inside the *same* collective call at the *same* logical point in the code, while one or a few ranks are elsewhere — that divergent rank (or the small set of them) is the actual root cause; everyone else is a symptom of waiting on it. **Step 3:** check `NCCL_DEBUG=INFO` output (should be enabled as standing practice, not retrofitted after the fact) for what transport was actually selected — a silent fallback from InfiniBand to TCP sockets (misdetected NIC, container missing IB verbs) can look exactly like a near-hang given how much slower TCP fallback is, even though technically nothing is "stuck." **Step 4:** if stack traces show a genuine deadlock pattern (mismatched collective ordering — one rank took a code path with an extra or missing collective call relative to everyone else), the fix is a code-level bug hunt on whatever rank-dependent branch caused the divergence — often something like a rank-specific exception being silently caught on one rank while others don't hit it, or a data-dependent conditional that isn't actually rank-invariant despite being intended to be. **Step 5:** if none of the above localizes it, bisect physically — relaunch on half the nodes to check whether the hang reproduces on a smaller topology (implicates a specific bad node/link, findable via further bisection) versus only manifesting at full scale (implicates a scale-dependent resource exhaustion, e.g., NIC queue pairs or communication buffer limits hit only at large world size). Throughout, the organizing principle is localizing *which* rank or component is anomalous before theorizing about *why* — in a 512-way job, undirected theorizing is far more expensive than the mechanical localization steps above.

## Q7: Why is tensor parallelism confined to the NVLink domain (typically one node), while pipeline and data parallelism can span nodes over InfiniBand without the same restriction?

The determining factor is not bandwidth alone — it's the combination of bandwidth, latency, and whether the communication is *overlappable* with other useful work. TP's all-reduce happens **inline**, once after attention's output projection and once after the FFN's down-projection, in *every* transformer layer, in both forward and backward — four all-reduces per layer per micro-batch (`001`, Section 3.3). Because the very next operation in that same layer depends on the all-reduce's result, there is no unrelated compute to overlap it with; any latency or bandwidth shortfall stalls the pipeline directly, in the critical path, at very high frequency. DP's gradient all-reduce, by contrast, happens once per step, and can be overlapped with the *ongoing backward pass* of earlier layers via bucketed, asynchronous dispatch (`001`, Section 2.3) — it tolerates the higher latency and lower bandwidth of inter-node InfiniBand (`005_Cluster_Hardware_Networking_And_Interconnect.md`) far better because it isn't blocking anything on the frequent, per-layer time scale. PP's point-to-point activation handoffs between adjacent stages are comparatively infrequent (once per micro-batch at each stage boundary, not four times per layer) and, under 1F1B, are already designed around a schedule with some slack, so the added latency of crossing InfiniBand is a smaller fraction of the total step time. Quantitatively, NVLink offers roughly an order of magnitude more bandwidth and dramatically lower latency (sub-microsecond, direct GPU-to-GPU via NVSwitch) than InfiniBand (low microseconds, traversing NICs and switch hops) — the rule of thumb "TP degree ≤ GPUs per node" exists because TP is the one axis whose communication pattern is both latency-critical and non-overlappable, making it uniquely intolerant of the slower inter-node fabric.

## Q8: Explain why expert-parallel all-to-all communication is a fundamentally different systems problem than DP/TP/PP's communication, not just "another collective operation."

DP's all-reduce, TP's all-reduce, and PP's point-to-point handoff all move a **fixed, statically-known** volume of data between a **fixed, statically-known** set of ranks — the pattern is determined entirely by tensor shapes and the parallel configuration, known before the job even starts running, regardless of what data flows through the model. Expert-parallel all-to-all breaks this completely: **which device sends how many tokens to which other device is an output of the router's decision on the current batch**, which itself depends on the model's current weights and the actual input data — neither of which is known statically. Three concrete consequences follow, as developed in `002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`: (1) aggregate communication volume scales predictably with `tokens × top_k × hidden_dim`, but *per-device* volume is not fixed and can vary step to step and device to device depending on routing; (2) the dispatch all-to-all is a synchronization barrier every device must wait on — no device can begin its local expert computation until it has received *all* tokens routed to it from *every* other device, so an imbalanced router (some experts overloaded, others starved) directly stalls the entire EP group behind its single most-loaded device, exactly analogous to how a straggler stalls a synchronous collective (`008`, Section 3) but caused by data/routing rather than hardware; (3) buffer sizing for the all-to-all must accommodate a data-dependent, a priori unknown per-device token count, forcing either a fixed capacity (with token dropping on overflow) or a more complex variable-size all-to-all requiring extra metadata exchange. This is why MoE load-balancing techniques — Mixtral's differentiable auxiliary loss (`..\OpenSource\005_Mixtral8x7B.md`) and DeepSeek-V3's non-gradient bias-feedback mechanism (`..\OpenSource\007_DeepSeek_V3.md`) — are not purely quality interventions; they are directly solving a systems problem (bounding the data-dependence of the all-to-all's traffic pattern back down toward something closer to DP/TP/PP's static predictability).

## Q9: (Coding) Implement an MFU calculator that takes a model configuration and measured throughput and returns MFU, using the precise (attention-inclusive) FLOPs-per-token formula.

```python
def calculate_mfu(
    num_params: float,           # N: non-embedding parameter count
    num_layers: int,             # L
    hidden_dim: int,             # h
    seq_len: int,                # s
    global_batch_size: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float,   # MUST match the actual precision in use (bf16 vs fp8)
) -> dict:
    """
    MFU = achieved FLOPs/sec (useful model compute) / theoretical peak FLOPs/sec.
    Uses FLOPs_per_token ~= 6N + 12*L*h*s (PaLM-style formula), which adds the
    attention-specific O(s) term the naive 6N-only approximation omits.
    """
    tokens_per_step = global_batch_size * seq_len
    tokens_per_second = tokens_per_step / step_time_seconds

    flops_per_token_naive = 6 * num_params
    attention_term = 12 * num_layers * hidden_dim * seq_len
    flops_per_token_precise = flops_per_token_naive + attention_term

    achieved_flops_naive = flops_per_token_naive * tokens_per_second
    achieved_flops_precise = flops_per_token_precise * tokens_per_second
    peak_flops_total = peak_flops_per_gpu * num_gpus

    return {
        "tokens_per_second": tokens_per_second,
        "mfu_naive_6N": achieved_flops_naive / peak_flops_total,
        "mfu_precise": achieved_flops_precise / peak_flops_total,
        "attention_term_fraction_of_flops": attention_term / flops_per_token_precise,
    }


# 70B model, seq_len=4096, global batch 1536 sequences, 4.2s/step, 512 H100s bf16 (peak 9.9e14 FLOPs/s)
result = calculate_mfu(
    num_params=70e9, num_layers=80, hidden_dim=8192, seq_len=4096,
    global_batch_size=1536, step_time_seconds=4.2, num_gpus=512,
    peak_flops_per_gpu=9.9e14,
)
print(f"MFU (precise): {result['mfu_precise']:.1%}")
print(f"Naive 6N would have reported: {result['mfu_naive_6N']:.1%}")
```

The two MFU values (naive vs. precise) are reported side by side deliberately: at short-to-moderate sequence lengths the divergence is small, but as `seq_len` grows relative to `hidden_dim`, the omitted attention term becomes non-negligible, and a naive `6N`-only calculation will systematically *overstate* achieved FLOPs (and therefore MFU) — worth surfacing explicitly rather than silently picking one formula, per `007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s discussion of why MFU figures from different sources aren't always directly comparable.

## Q10: Explain precisely, in terms of exponent/mantissa bit allocation, why bf16 displaced fp16 as the default training format despite being less numerically precise.

fp16 and bf16 both use 16 total bits but split them differently: fp16 is 1 sign / 5 exponent / 10 mantissa bits; bf16 is 1 sign / 8 exponent / 7 mantissa bits. Exponent bits set dynamic range; mantissa bits set relative precision. fp16's 5-bit exponent gives a narrow range (roughly `6e-5` to `6.5e4`), which in practice is genuinely too narrow for large-transformer training: gradients — especially in deep networks where backward-pass chain-rule multiplication compounds shrinkage across many layers — routinely take on magnitudes below fp16's minimum representable value, causing **silent underflow to exactly zero** (not a crash — a valid, unremarkable-looking finite value that simply discards real gradient signal). The standard mitigation, loss scaling (multiply the loss by a scalar before backward, unscale after), works but adds real operational complexity: a dynamic scale factor that must be tuned, monitored for a rising overflow/skip rate, and that occasionally forces skipping an entire optimizer step when the scale turns out to be too aggressive in the other direction (pushing an already-large value past fp16's *maximum*, producing `inf`). bf16's 8-bit exponent exactly matches fp32's, giving it fp32's full dynamic range — the underflow/overflow problem that motivates loss scaling largely disappears structurally, because almost anything fp32 could represent without under/overflowing, bf16 can too. The cost is bf16's 7 mantissa bits versus fp16's 10 — roughly 8x worse relative precision (ULP ~7.8e-3 vs. ~9.8e-4). Empirically, large transformer training has turned out to tolerate this coarser per-value precision well, while the range problem fp16 has was a real, frequent source of NaN losses and wasted steps. The net trade bf16 makes — worse precision, in exchange for eliminating an entire operational failure mode (range-induced under/overflow) and the loss-scaling machinery built to manage it — is why it became the default the moment hardware (Ampere onward) gave it native tensor-core support, not because it is unconditionally "better" in an information-theoretic sense.

## Q11: Scenario — training has been stable for 10,000 steps with a smoothly decreasing loss, then the loss abruptly jumps to NaN over the course of one or two steps. Diagnose the likely cause and how you'd confirm it.

An abrupt jump to NaN after a long stable run, rather than a gradual drift, is the signature of an **overflow event** (`004_Mixed_Precision_Training_And_Numerical_Stability.md`, Section 4), not underflow — underflow is silent and produces plateauing or degraded quality, not NaN. The likely mechanical sequence: something (a specific batch with unusual statistics, a learning-rate schedule transition, or an emergent instability in the optimization dynamics) produced an unusually large gradient at some step; if gradient clipping either wasn't applied or was set too permissively, that large gradient produced a disproportionately large parameter update; the *next* forward pass, run through the now-perturbed weights, produced larger-than-normal activations; those larger activations produced an even larger gradient on the following backward pass — a short positive-feedback loop that terminates in an actual numeric overflow (`inf`) within a handful of steps once some tensor's magnitude exceeds the working precision's representable maximum, and that `inf` then propagates through nearly every subsequent operation, becoming `NaN` (e.g., `inf - inf`) almost immediately. To confirm this diagnosis rather than merely asserting it: check the **gradient-norm history** for the steps immediately preceding the NaN — the leading-indicator pattern to look for is one or more sharp spikes to several multiples of the recent trailing-average gradient norm shortly before the collapse (this is exactly why gradient-norm monitoring, not just loss monitoring, should be standing infrastructure — the spike is very often visible one or several steps before the loss itself shows anything unusual). If using fp16 with dynamic loss scaling, also check the scaler's recent skip-rate — a rising skip rate in the steps before the event is a second corroborating signal that the run was drifting into a less stable numeric regime. Remediation: restore from the last good checkpoint before the spike (`006_Checkpointing_Fault_Tolerance_And_Elastic_Training.md`), and address the root cause rather than merely resuming blindly — tighten gradient clipping (a lower max-norm threshold), check whether the offending step corresponded to an unusual/corrupted data batch (worth inspecting directly if identifiable), and consider whether the learning rate at that point in the schedule was too aggressive for the current training phase. If using bf16, the diagnosis shifts slightly: since bf16's range matches fp32, an outright overflow is less likely to be a pure range artifact and more likely to indicate a genuine training-dynamics instability (or, per `004`, an FP8-specific quantization issue if any part of the pipeline uses FP8) rather than a precision-format limitation — worth distinguishing explicitly, since the fix differs (LR/clipping/data investigation vs. revisiting FP8 scaling granularity).

## Q12: Explain DeepSeek-V3's fine-grained FP8 quantization scheme and why a single per-tensor scale factor would be insufficient.

FP8 (E4M3: 1 sign, 4 exponent, 3 mantissa bits) has only 3 mantissa bits — an order of magnitude coarser relative precision than even bf16's already-reduced 7. The standard way to use a narrow-range, low-precision format for a tensor whose true values may exceed that format's representable range is **tensor scaling**: multiply the tensor by a scale factor before casting down, sized so the tensor's largest element just fits the format's maximum, then divide the scale back out after the operation. The problem this runs into at FP8's precision level, more acutely than at fp16/bf16: real transformer activation and weight tensors frequently contain a small number of **outlier elements** (specific channels or positions with magnitude far larger than the tensor's typical element) — a single global scale factor sized to accommodate that outlier forces every other, much-smaller element in the same tensor down toward the bottom of FP8's tiny 3-bit mantissa range, where the relative quantization error is largest. The elements a global scale factor is supposed to protect from overflow are exactly the ones whose presence degrades every *other* element's precision. `..\OpenSource\007_DeepSeek_V3.md`'s fix is **fine-grained scaling**: rather than one scale per tensor, compute a separate scale factor per small **tile** (1×128 elements) for activations and per small **block** (128×128 elements) for weights, each sized to that specific tile/block's own local maximum rather than the whole tensor's. This localizes the "must accommodate the largest element" cost to a small, more magnitude-homogeneous group, so an outlier in one tile no longer degrades unrelated elements elsewhere in the same tensor. DeepSeek additionally keeps certain accumulations and the optimizer state in higher precision (bf16/fp32) rather than FP8 — the general principle being that a single low-precision multiply's rounding error is usually tolerable, but error accumulated across a long reduction (many summed terms, or many thousands of training steps' worth of optimizer updates) compounds, and it's specifically the *compounding* cases that get exempted from FP8 rather than a blanket judgment about which tensors matter more.

## Q13: (Coding) The following bucketed gradient-synchronization snippet for data-parallel training has a bug that causes some ranks to occasionally read stale (partially-updated) gradients before the optimizer step. Find and fix it.

```python
# BUGGY VERSION
def sync_gradients_buggy(model, process_group):
    handles = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=process_group, async_op=True)
            handles.append(handle)
    # optimizer.step() gets called right after this function returns —
    # but nothing here actually waits for the async all-reduces to finish!
    for param in model.parameters():
        if param.grad is not None:
            param.grad /= dist.get_world_size(process_group)
    return handles
```

The bug: `all_reduce` is issued asynchronously (`async_op=True`, correctly done to allow overlap with ongoing backward compute, per `001_Parallelism_Strategies_Data_Tensor_Pipeline.md` Section 2.3), but the function divides each gradient by world size and returns *without ever waiting on the returned handles* — so the division (and, worse, the subsequent optimizer step) can execute before the all-reduce has actually completed, reading a partially-summed, in-flight gradient buffer. This is exactly the kind of bug that doesn't crash — it silently corrupts the training dynamics with wrong (partially-reduced) gradients, indistinguishable from the "loss looks approximately fine but is subtly wrong" symptom class discussed in `004` and `008`.

```python
# FIXED VERSION
def sync_gradients(model, process_group):
    handles = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=process_group, async_op=True)
            handles.append((param, handle))

    world_size = dist.get_world_size(process_group)
    # Explicitly wait for EACH all-reduce to complete before touching that param's
    # gradient — this is the missing synchronization point.
    for param, handle in handles:
        handle.wait()
        param.grad /= world_size
    # Only now is it safe for the caller to invoke optimizer.step()
```

The fix makes the synchronization explicit: `handle.wait()` blocks until that specific all-reduce has actually completed before the gradient is divided and, later, consumed by the optimizer. The async dispatch loop still allows overlap with backward (issuing each parameter's all-reduce as soon as its gradient is ready, while later layers' backward computation continues), but the wait loop guarantees correctness before the optimizer step — overlap and correctness are not in tension here, but only if the wait is never skipped.

## Q14: Explain the checkpoint-frequency trade-off quantitatively, and why frontier-scale runs tend toward frequent checkpointing despite the larger absolute checkpoint size at that scale.

If checkpoints are written every `T` steps and a failure occurs uniformly at random within an interval, the **expected** number of steps of work lost per failure is `T/2` (on average, failure strikes midway through the interval since the last checkpoint), with a worst case of a full `T` steps. Set against this is the overhead of checkpointing itself: each checkpoint write costs some time `C` (a stall if synchronous, or a smaller overlapped cost if asynchronous per `006_Checkpointing_Fault_Tolerance_And_Elastic_Training.md` Section 5), and checkpoints are written `(total_steps / T)` times over the run — so total checkpointing overhead scales as `1/T` while expected lost work scales as `T` (linearly, via the `T/2` term multiplied by however many failures occur, and failure count itself is a function of wall-clock duration, which is itself weakly a function of `T` through the overhead — a genuine coupled optimization, but treating failure rate `λ` as roughly fixed for a given cluster size is a reasonable first-order approximation). The optimal `T` balances these two competing costs, and it shifts with `λ`: a cluster with a higher aggregate failure rate (more GPUs, per `006` Section 1's near-certainty arithmetic at thousands-of-GPU scale) wants a *smaller* `T` (checkpoint more often) because each unit of `T` risked costs more in expectation, even though each individual checkpoint at large model scale is itself larger (multi-terabyte, per `006` Section 2) and nominally more expensive to write. The reason this doesn't force frontier runs into paralysis is asynchronous checkpointing (`006` Section 5): decoupling "make the state safe against a GPU-only failure" (a fast GPU-to-host-RAM copy) from "get the state to durable storage" (a slower background flush that overlaps with continued training) means the *exposed* stall per checkpoint shrinks by an order of magnitude or more relative to a naive synchronous write, which is precisely what makes frequent checkpointing affordable at exactly the scale where the failure-rate arithmetic most demands it.

## Q15: Scenario — an MoE training job shows, via per-rank profiling, that some GPUs in the expert-parallel group are at ~100% utilization while others sit idle roughly 40% of the time, and this pattern is consistent step after step. Diagnose and propose fixes.

A *consistent*, step-after-step pattern (not random noise) of some devices maxed out while others idle specifically within an EP group is the direct signature of **routing load imbalance** (`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`, Section 4), not a hardware straggler (a hardware straggler would typically show consistently *slow*, not consistently *idle*, and wouldn't correlate specifically with EP-group membership) and not a generic communication bottleneck (which would tend to show up more uniformly across the group rather than a specific idle/busy split). The mechanism: the EP all-to-all's combine step is an implicit barrier — no device proceeds past the layer until every device has both received its routed tokens *and* finished processing them; if the router is sending disproportionately more tokens to the experts hosted on a subset of devices, those devices are the ones at 100% (genuinely more work), while the rest, having finished their smaller share, sit idle waiting at the barrier. **Confirming the diagnosis**: instrument per-expert (not just per-device) token counts over a representative window of steps — a skewed distribution (a small number of experts receiving disproportionately more tokens than a balanced target) directly confirms router collapse rather than any other explanation. **Fixes, in order of typically increasing intervention**: (1) if using a differentiable auxiliary load-balancing loss (Mixtral-style, `..\OpenSource\005_Mixtral8x7B.md`), check whether its weight is tuned aggressively enough — too small a weight is a common, easy-to-miss cause of exactly this symptom, though over-weighting it trades away model quality, so this is a genuine knob to tune carefully rather than maximize; (2) consider a bias-based, non-gradient load-balancing mechanism (DeepSeek-V3-style, `..\OpenSource\007_DeepSeek_V3.md`) if not already in use — its key advantage here is that the balancing lever (`γ`, the bias step size) can be tuned directly against the observed load-imbalance metric without simultaneously perturbing the LM loss's gradient, making it a cleaner systems-level fix than adjusting an auxiliary-loss weight that also affects quality; (3) as a bounding mechanism regardless of the balancing approach's tuning, introduce or tighten a capacity factor with token dropping/overflow handling, which caps the *worst-case* imbalance a poorly-tuned router can inflict on the systems side, at the cost of some dropped-token quality impact; (4) verify this isn't compounded by a genuinely poor EP-to-topology placement (`005_Cluster_Hardware_Networking_And_Interconnect.md`, Section 5) — if the overloaded devices also happen to sit behind an oversubscribed network link, both routing imbalance and topology are contributing, and fixing only one won't fully resolve the symptom.

## Q16: Explain why network bandwidth, rather than raw FLOPs, is frequently the actual bottleneck at frontier training scale, and describe the hardware trend driving this.

Across recent GPU generations, per-GPU FLOPs have grown considerably faster than per-GPU network bandwidth, both intra-node (NVLink) and especially inter-node (InfiniBand/RoCE). A100 to H100 is roughly a 3x increase in dense bf16 FLOPs (and closer to 6x for FP8-capable workloads via H100's native FP8 tensor cores), while NVLink bandwidth grew from roughly 600GB/s to roughly 900GB/s over the same span — a much smaller multiple — and inter-node bandwidth growth has been slower still in relative terms (`005_Cluster_Hardware_Networking_And_Interconnect.md`, Section 6). The direct consequence: the ratio of available compute to available communication bandwidth per GPU gets **worse** with each new hardware generation, meaning a parallel configuration that was comfortably compute-bound on one generation's hardware can become communication-bound on the very next generation, with no change to the model or parallel strategy at all — purely because the compute side raced ahead of the network side. A concrete worked check (`005`, Section 7): for a 70B model at `TP=8`, a single layer's four TP all-reduces on a `[b,s,d_model]`-shaped bf16 activation tensor land in the same rough order of magnitude of wall-clock time (roughly a millisecond) as that layer's matmul compute time at H100 peak throughput — not two orders of magnitude smaller, which is what "compute dominates, communication is a rounding error" would require to be true. This is precisely why so much systems engineering effort in frontier training (overlap-aware pipeline schedules like DualPipe, topology-aware rank placement, custom communication kernels) is aimed at **hiding** communication behind compute rather than eliminating communication volume outright: on modern hardware, there are usually enough idle compute cycles during a communication-bound stretch to do useful work concurrently, and the entire engineering problem is arranging the schedule so that idle-compute time and required-communication time coincide, rather than serializing them. A practical diagnostic implication worth stating directly: if scaling a job to more GPUs *increases* wall-clock step time (or increases it by more than a small margin), that is direct evidence communication — not compute — is the actual bottleneck for that configuration, since per-GPU compute work didn't change when GPU count grew, only the communication group size did.

## Q17: (Coding) Implement activation checkpointing manually for a simple two-layer MLP block, showing both the memory-saving forward pass and the recompute-based backward pass.

```python
import torch

class CheckpointedMLP(torch.autograd.Function):
    """
    Manual activation checkpointing for y = W2 @ gelu(W1 @ x).
    Forward: run the computation but do NOT retain gelu's input/output for backward.
    Backward: RE-RUN the forward computation (with grad tracking) to regenerate
    exactly the intermediates needed, use them immediately, then discard again.
    """

    @staticmethod
    def forward(ctx, x, W1, W2):
        ctx.save_for_backward(x, W1, W2)   # only the SEGMENT BOUNDARY input is saved
        with torch.no_grad():
            h = torch.nn.functional.linear(x, W1)
            a = torch.nn.functional.gelu(h)     # intermediate 'h' and 'a' are NOT retained
            y = torch.nn.functional.linear(a, W2)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, W1, W2 = ctx.saved_tensors
        # Recompute the forward pass, this time WITH grad tracking, to regenerate
        # the intermediates (h, a) that backward needs but forward discarded.
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            W1_ = W1.detach().requires_grad_(True)
            W2_ = W2.detach().requires_grad_(True)
            h = torch.nn.functional.linear(x, W1_)
            a = torch.nn.functional.gelu(h)
            y = torch.nn.functional.linear(a, W2_)
        grad_x, grad_W1, grad_W2 = torch.autograd.grad(y, (x, W1_, W2_), grad_outputs=grad_y)
        return grad_x, grad_W1, grad_W2


def checkpointed_mlp(x, W1, W2):
    return CheckpointedMLP.apply(x, W1, W2)


# Usage: memory profile shows only x, W1, W2 retained across the forward call for this
# block (not h or a, which would normally be kept for a standard, non-checkpointed backward) —
# at the cost of running the linear+gelu+linear sequence TWICE (once in forward, discarded;
# once again in backward, to regenerate what's needed) instead of once.
```

This is the mechanism `003_ZeRO_Optimizer_Sharding_And_Memory_Management.md` Section 7 describes in prose: the forward pass computes but does not retain `h` and `a`; only the segment's boundary input (`x`, plus the weights) survives to backward, and backward regenerates the discarded intermediates by literally re-executing the forward computation. The cost is one extra forward-equivalent pass through this specific segment (roughly the ~30% extra-compute figure commonly quoted for full-layer checkpointing, when applied uniformly across every layer of a model), in exchange for `O(1)` retained activation memory per checkpointed segment instead of `O(segment depth)`.

## Q18: Explain ZeRO-3 / FSDP's just-in-time parameter all-gather mechanism, and why its communication cost is roughly 1.5x a plain data-parallel baseline rather than being "free" the way ZeRO-1/2 are.

Under ZeRO-1 and ZeRO-2, the bf16 parameters remain fully replicated on every device at rest — only the optimizer state (Stage 1) or optimizer state plus gradients (Stage 2) are sharded — so both stages can implement their sharding as a reduce-scatter (for the gradient) plus an all-gather (to redistribute the updated weights), which together move exactly the same total volume as a standard DP all-reduce would (recall an all-reduce *is* internally implemented as reduce-scatter followed by all-gather). ZeRO-3 goes further and shards the parameters themselves, so no device holds a full weight tensor at rest at all. But matrix multiplication needs the **full** weight matrix to compute a layer's output — a `1/N_d` shard of a weight matrix cannot compute a correct partial result the way TP's row/column shards can (TP's sharding is specifically designed so partial results are meaningful and summable; ZeRO-3's sharding is an arbitrary partition purely for memory-storage purposes, with no such algebraic structure). So immediately before a layer's forward computation, ZeRO-3 must **all-gather** that layer's full parameter tensor from its shards across the DP group, use it, and then discard the non-owned portion again to free the memory. This same all-gather must happen **again** before that layer's backward computation, since backward also needs the full weight tensor (to compute the gradient with respect to the layer's input). This gives ZeRO-3 **two** additional all-gathers per layer (forward and backward) that ZeRO-1/2 simply don't need to pay, because ZeRO-1/2 already have the full weight sitting in memory at all times. Added to the reduce-scatter for gradients that all three stages pay, the ZeRO paper's own accounting puts Stage 3's total communication volume at roughly **1.5x** a plain DP all-reduce's volume — a real, measurable throughput cost. This is precisely why ZeRO-3 is reached for specifically when the still-replicated `2Ψ` bytes/param of bf16 parameters under ZeRO-1/2 is itself the binding memory constraint (very large models relative to per-device HBM, or a parallel grid leaving few devices on the DP axis to shard across), rather than applied unconditionally as a default the way ZeRO-1/2 essentially are.

## Q19: Scenario — a training configuration runs at a healthy MFU on 64 GPUs, but when scaled to 512 GPUs with the same per-GPU micro-batch size (only the DP degree increased 8x), MFU drops substantially. Diagnose.

Holding per-GPU micro-batch size (and hence per-GPU compute work) fixed while increasing only the DP degree isolates the DP axis specifically as the variable under test — per `007_Training_Efficiency_Metrics_MFU_And_Utilization.md` Section 4's Step 2 diagnostic. Since per-device compute time is unchanged, a drop in MFU under this specific experiment implicates the DP gradient-synchronization communication (or something correlated with a larger DP group) as the bottleneck, not compute-kernel efficiency or the TP/PP portions of the configuration, which didn't change. The most likely concrete causes, roughly in order of how commonly they explain this pattern: (1) **inadequate overlap of the gradient all-reduce with backward compute** — the theoretical asymptote of ring all-reduce's per-device cost approaching a constant (`Q2` above) assumes an efficient ring topology and reasonably-sized buckets; if bucket sizing or the overlap-scheduling logic wasn't tuned for the larger group size, more of the all-reduce's cost becomes exposed (not hidden behind backward) at 512 than at 64; (2) **topology-unaware placement** (`005_Cluster_Hardware_Networking_And_Interconnect.md`, Section 5) — going from 64 to 512 GPUs very likely means the DP group now spans many more nodes, and possibly crosses a fat-tree topology's spine links for the first time or more heavily than before; if the cluster's bisection bandwidth is oversubscribed, a DP group spanning more of the topology can see meaningfully degraded effective bandwidth per participant even though the *algorithmic* all-reduce cost model predicts near-constant per-device cost — the real-world topology, not the idealized ring model, is the actual limiter; (3) **checking whether the increase was purely DP-axis, or whether TP/PP groups also now span more nodes than before** — if the original 64-GPU job fit its TP group within a single node's NVLink domain and the 512-GPU job's launch script didn't preserve that placement (e.g., a naive job scheduler that doesn't guarantee TP-group locality once total GPU count grows), TP's latency-critical all-reduces may have been silently pushed onto the slower inter-node fabric, which would produce exactly this symptom and is a placement bug entirely orthogonal to the DP scaling itself. The correct next step is exactly `007`'s diagnostic sequence: profile the 512-GPU run's step-time breakdown (compute vs. communication vs. idle) and compare directly against the 64-GPU run's breakdown — a growing communication-kernel-time fraction confirms the DP/topology hypothesis; a growing idle-time fraction with flat communication-kernel time points instead toward a scheduling/overlap bug rather than raw bandwidth.

## Q20: (Coding) Write a function that computes the pipeline bubble fraction for a given (P, m) configuration, and a second function that, given a target maximum bubble fraction and a fixed P, recommends the minimum micro-batch count m needed to meet it.

```python
def pipeline_bubble_fraction(num_stages: int, num_microbatches: int) -> float:
    """
    bubble_fraction = (P - 1) / (m + P - 1), per the GPipe/1F1B fill-drain overhead model.
    Both schedules share this bubble fraction; 1F1B differs only in activation-memory
    scaling with m, not in this ratio (see 001_Parallelism_Strategies... Section 4.2).
    """
    P, m = num_stages, num_microbatches
    if m < 1 or P < 1:
        raise ValueError("num_stages and num_microbatches must be >= 1")
    return (P - 1) / (m + P - 1)


def min_microbatches_for_target_bubble(num_stages: int, target_bubble_fraction: float) -> int:
    """
    Solve (P-1)/(m+P-1) <= target for the smallest integer m >= 1.
    Rearranged: m >= (P-1)/target - (P-1) = (P-1) * (1/target - 1)
    """
    P = num_stages
    if not (0 < target_bubble_fraction < 1):
        raise ValueError("target_bubble_fraction must be strictly between 0 and 1")
    m_min_real = (P - 1) * (1.0 / target_bubble_fraction - 1.0)
    m_min = max(1, int(m_min_real) + 1)   # ceiling, and at least 1 microbatch
    # verify (guards against off-by-one from floating point)
    while pipeline_bubble_fraction(P, m_min) > target_bubble_fraction:
        m_min += 1
    return m_min


# Example: P=4 pipeline stages, want bubble fraction <= 10%
P = 4
m_needed = min_microbatches_for_target_bubble(P, target_bubble_fraction=0.10)
print(f"Need at least {m_needed} microbatches for <=10% bubble with P={P}")
achieved = pipeline_bubble_fraction(P, m_needed)
print(f"Achieved bubble fraction: {achieved:.1%}")
# -> Need at least 28 microbatches; achieved ~9.7%

# Cross-check against 001's worked example: P=4, m=16 -> bubble fraction:
print(f"P=4, m=16: {pipeline_bubble_fraction(4, 16):.1%}")   # ~15.8%, matches 001's worked example
```

This directly operationalizes the two-sided lever from Q5: given a memory-imposed ceiling on `m` (from activation memory, `003`), `pipeline_bubble_fraction` tells you what bubble cost you're paying; given a target bubble budget, `min_microbatches_for_target_bubble` tells you the minimum `m` (and hence minimum activation memory, under 1F1B's `O(P)`-in-flight bound) required to hit it — the exact reasoning `001`'s Section 5.2 worked 3D-parallelism example applies informally, made mechanical.
