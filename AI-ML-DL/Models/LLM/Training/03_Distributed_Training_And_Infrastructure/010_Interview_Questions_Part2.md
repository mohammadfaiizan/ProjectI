# Distributed Training and Infrastructure: Interview Questions, Part 2

## Q1: You're given a 300B-parameter dense model and a cluster of 2048 H100 GPUs (256 nodes x 8 GPUs/node, NVLink intra-node, NDR InfiniBand inter-node). Walk through how you'd choose a 3D parallel configuration.

Follow the same ordered reasoning as `001_Parallelism_Strategies_Data_Tensor_Pipeline.md` Section 5.2, with numbers appropriate to this larger model. **Step 1 — fix TP from the node boundary:** 8 GPUs/node with NVLink gives `TP=8` as the natural default, per the rule that TP's inline, non-overlappable all-reduce should stay off the slower inter-node fabric. **Step 2 — size PP from a memory check:** each TP shard holds `300B/8 = 37.5B` "logical" parameters. Under ZeRO-shape mixed-precision accounting (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`, ~16 bytes/param before further DP-axis sharding), that's `37.5B x 16 = 600GB` per device before any ZeRO sharding — far beyond an H100's 80GB, so either PP or ZeRO-3 (or both) must absorb this. Given ZeRO-1/2 alone shards only the optimizer/gradient state (12-14 bytes/param) and leaves ~4Ψ bytes/param of bf16 params+grads still replicated across the DP group, even a large DP degree can't shrink that residual `4 x 37.5B = 150GB` below 80GB — so at this size, pure ZeRO-1/2 plus TP=8 is insufficient regardless of DP degree, and PP is now genuinely load-bearing for memory (not just a throughput tool), or ZeRO-3 needs to be adopted for the DP axis specifically to shard that residual bf16 replica. A reasonable choice: `PP=8` (splitting depth 8-way, cutting local params to `300B/(8x8)=4.69B`, comfortably fitting even under ZeRO-1/2's ~4 bytes/param residual replica: `4.69B x 4 = 18.75GB`, leaving generous headroom for optimizer-state shards and activations). **Step 3 — fill DP:** `TP x PP = 8 x 8 = 64`; with 2048 total GPUs, `DP = 2048/64 = 32`. Configuration: **TP=8, PP=8, DP=32** (`8x8x32=2048`). **Step 4 — sanity-check the bubble:** with `PP=8` (higher than the earlier 70B example's `PP=4`), the bubble fraction is more sensitive to micro-batch count — `(8-1)/(m+7)`; hitting a 15% bubble target requires `m ≈ 40` micro-batches (versus `m≈16` sufficing at `PP=4`), meaning this configuration needs either a larger global batch or activation-checkpointing-freed memory (`003`, Section 7) to afford enough in-flight micro-batches under 1F1B's `O(P)` memory bound — worth flagging explicitly as the direct cost of the larger `PP` this model size forced. **Step 5 — reconsider whether interleaved/virtual pipeline stages (`001` Section 4.3) are worth adopting here**, since they specifically target situations exactly like this one (larger `P`, bubble fraction becoming a real tax) by shrinking the effective bubble at the cost of more frequent, smaller point-to-point transfers — a reasonable follow-up refinement once the baseline TP=8/PP=8/DP=32 grid is validated.

## Q2: From a systems perspective, why does decoupling MoE load balancing from the LM loss's gradient (DeepSeek-V3's bias mechanism) matter more than it might first appear?

An auxiliary balancing loss (Mixtral-style, `..\OpenSource\005_Mixtral8x7B.md`) and the LM cross-entropy loss share the same backward pass and therefore compete for the same gradient budget on the router's parameters — the auxiliary loss's gradient pulls toward balanced routing, the LM loss's gradient pulls toward whatever routing the model finds most useful for prediction, and the *weight* on the auxiliary loss is the single knob mediating this tension. That weight is simultaneously a **quality knob** (too high, and language-modeling quality measurably degrades because routing is being forced away from what the LM loss prefers) and a **systems knob** (too low, and load imbalance persists, directly costing cluster throughput via the all-to-all barrier mechanism in `002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md` Section 4) — and there is no way to tune it for one purpose without also affecting the other, because it's one number feeding one shared gradient computation. DeepSeek-V3's bias mechanism (`..\OpenSource\007_DeepSeek_V3.md`) removes this coupling structurally: the per-expert bias `b_i` is adjusted by a fixed step `γ` based purely on *observed load* (`b_i -= γ` if overloaded, `b_i += γ` if underloaded), entirely outside the backpropagated graph — no gradient ever flows through this update, so it cannot compete with or distort the LM loss's gradient at all. This means an infrastructure engineer observing an imbalance metric (per-device or per-expert token counts, exactly the metric from `007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s Step 5 diagnostic) can tune `γ` directly against that systems-level signal, with a guarantee that doing so has zero direct effect on the LM objective's gradient — a genuinely different, and operationally much cleaner, posture than tuning an auxiliary-loss weight, where every adjustment is simultaneously a quality experiment and a systems experiment bundled into one number, making it much harder to attribute an observed change in either loss curves or throughput to the correct cause.

## Q3: (Coding) Implement a single-process simulation of ring all-reduce over N simulated "ranks" (e.g., using a list of tensors to represent each rank's local buffer), and verify it produces the correct sum.

```python
import torch

def ring_allreduce_simulated(local_buffers: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Simulates ring all-reduce across len(local_buffers) ranks within a single process,
    for pedagogical verification of the reduce-scatter + all-gather decomposition
    described in 001_Parallelism_Strategies_Data_Tensor_Pipeline.md Section 2.2.
    Each rank's buffer must be evenly divisible into N chunks.
    """
    N = len(local_buffers)
    numel = local_buffers[0].numel()
    assert numel % N == 0, "buffer size must be divisible by world size for this simplified sim"
    chunk_size = numel // N

    # Represent each rank's buffer as a list of N chunks
    chunks = [buf.clone().view(N, chunk_size) for buf in local_buffers]

    # --- Phase 1: reduce-scatter ---
    # After this phase, chunks[r][r] holds the FULLY reduced sum for chunk index r.
    for step in range(N - 1):
        for rank in range(N):
            send_chunk_idx = (rank - step) % N
            recv_from = (rank - 1) % N
            recv_chunk_idx = (rank - step - 1) % N
            # simulate rank `rank` receiving recv_from's send_chunk_idx-matching chunk
            chunks[rank][recv_chunk_idx] += chunks[recv_from][send_chunk_idx]

    # --- Phase 2: all-gather ---
    # Circulate the now-finished chunks so every rank ends with ALL N finished chunks.
    for step in range(N - 1):
        for rank in range(N):
            send_chunk_idx = (rank - step) % N
            recv_from = (rank - 1) % N
            recv_chunk_idx = (rank - step - 1) % N
            chunks[rank][recv_chunk_idx] = chunks[recv_from][send_chunk_idx].clone()

    return [c.view(-1) for c in chunks]


# Verification against the naive/expected result
torch.manual_seed(0)
N = 4
local_buffers = [torch.randn(8) for _ in range(N)]
expected_sum = sum(local_buffers)

result = ring_allreduce_simulated(local_buffers)
for r in range(N):
    assert torch.allclose(result[r], expected_sum, atol=1e-5), f"rank {r} mismatch"
print("Ring all-reduce simulation verified: every rank holds the correct full sum.")
```

Note the implementation deliberately mirrors the two-phase structure from `001` Section 2.2 exactly (reduce-scatter, then all-gather) rather than using a shortcut sum, specifically so the chunk-indexing logic in the simulation is the same logic a real NCCL-backed implementation follows — a useful thing to be able to produce from memory, since interviewers testing this topic are often checking whether the candidate actually understands the two-phase decomposition rather than just being able to state "it's bandwidth optimal."

## Q4: What is silent data corruption in the context of distributed training, why is it uniquely hard to detect, and what would you actually build to catch it?

Silent data corruption (SDC) refers to a hardware defect — an undetected or only partially corrected bit flip in GPU memory, a marginal compute unit producing subtly wrong results on specific operations, or a corrupting interconnect link — that causes a device to produce **numerically wrong activations or gradients without crashing, erroring, or producing an obvious NaN/inf**. It is uniquely hard to detect precisely because every other failure mode discussed in this module announces itself through some observable channel: a crash is loud, a straggler shows up as measurably slower throughput, an overflow produces NaN within a step or two. SDC produces none of these — the job continues running, the affected device's local computation completes "successfully" from the process's point of view, and the loss curve can look approximately normal, especially if the corruption affects a small fraction of values or occurs at low frequency, making it easy to dismiss as ordinary training noise even by an attentive engineer watching the loss curve closely. This is not a hypothetical concern: multiple hyperscale infrastructure operators (Google and Meta, in published work on their general server fleets) have reported SDC as a real, low-probability-per-unit but operationally nonzero-in-aggregate phenomenon — the same "individually rare, collectively near-certain at scale" arithmetic from `006_Checkpointing_Fault_Tolerance_And_Elastic_Training.md` Section 1 applies here, just to subtle miscomputation instead of outright failure. **What to actually build**: (1) cross-replica statistical monitoring — log per-DP-replica loss and gradient-norm distributions over a rolling window, and alert on any replica that is a *persistent, systematic* outlier relative to its peers (not just noisier — genuinely shifted), since replicas process different data but should be statistically exchangeable on these metrics over a large enough window; (2) periodic (not every-step, since it's expensive) numerical cross-validation of a fixed batch against a known-good reference path, to catch deterministic miscomputation a purely statistical check might miss if its aggregate effect on loss happens to be small; (3) correlate any flagged anomaly against hardware-level telemetry (ECC correctable-error counters, Xid errors in driver logs) — a device with an elevated but sub-crash-threshold correctable-error rate is a legitimate leading indicator worth proactively investigating even absent a confirmed training-quality symptom yet. The honest framing for an interview answer: SDC cannot be prevented at the software layer (it's a hardware reliability problem), but its blast radius can be bounded by treating "persistent per-replica statistical anomaly" as a first-class alert condition on par with a crash, scaled in monitoring intensity to the size and cost of the run.

## Q5: Scenario — two nodes running what should be identical code produce measurably different numerical results on the same input batch (verified via a controlled single-batch test), and this has been going on since a recent cluster maintenance window. Diagnose.

"Identical code, different numerical results, correlated with a recent cluster change" is the textbook signature of a **configuration mismatch** (`008_Debugging_Distributed_Training_Failures.md`, Section 5), not a hardware SDC issue (Q4) or an algorithmic bug — the correlation with a maintenance window is the key clue pointing away from a random hardware defect and toward something that changed for a subset of nodes specifically during that window. The concrete checklist, in order of likelihood: (1) **driver/library version drift** — a maintenance window is exactly when a rolling driver, CUDA, NCCL, or cuDNN update is likely to have been applied to some but not all nodes; different cuDNN/cuBLAS versions can select different algorithms (e.g., different convolution or attention kernel implementations) for the same operation, and different algorithms can have measurably different floating-point rounding behavior even when mathematically equivalent — check `nvidia-smi`, `nccl-tests`, and library version strings on both nodes and diff them directly rather than assuming they match because the deployment process was "supposed to" keep them in sync; (2) **stale container image** — if the maintenance window involved rebuilding or repushing an image under a mutable tag rather than a pinned digest, it's entirely possible the two nodes are running genuinely different code or dependency versions despite both having pulled "the same" tag at different times; compare image digests directly, not tags; (3) **environment variable drift**, particularly `NCCL_*` variables that can alter transport selection or algorithm choice — diff the full environment on both nodes; (4) **stale local data/tokenizer cache** — if either node has a locally cached copy of training data or the tokenizer that wasn't refreshed during the maintenance window while the canonical source was updated, checksum the actual files each node is reading from, not just the paths. The fix, once identified, is almost always re-provisioning the drifted node(s) to match the canonical configuration exactly — and the durable prevention, worth recommending regardless of which specific cause is found, is a pre-flight configuration/environment hash check run automatically at every job launch (`008` Section 5, prevention item 2), so this class of bug is caught in seconds at launch time rather than requiring a multi-hour controlled-batch investigation after the fact.

## Q6: Quantify activation checkpointing's compute-memory trade-off precisely — what is the actual extra compute cost, and where does the "~33%" figure people cite actually come from?

Standard (non-checkpointed) training does one forward pass and one backward pass per step; the backward pass costs roughly twice the forward pass's FLOPs (one component to compute the gradient with respect to the layer's input, needed to propagate further backward, and one component to compute the gradient with respect to the layer's weights, needed for the optimizer update) — so, in FLOPs-per-token units, standard training costs roughly `1 (forward) + 2 (backward) = 3` units. Full activation checkpointing (checkpointing every layer, `003_ZeRO_Optimizer_Sharding_And_Memory_Management.md` Section 7) adds one additional forward-pass-equivalent computation per checkpointed segment, since backward must *re-run* that segment's forward pass to regenerate the intermediates it discarded — bringing the total to `1 (original forward) + 1 (recompute forward) + 2 (backward) = 4` units. The extra cost relative to the unchecked baseline is `(4-3)/3 ≈ 33%` — this is exactly where the commonly-cited "~33% more compute" figure comes from, and it's worth being able to derive it on the spot rather than just quoting it, since interviewers testing this often follow up by asking exactly this derivation. The memory benefit in exchange: activation memory drops from `O(L)` (every layer's intermediates retained until backward reaches them) to `O(1)` per checkpointed segment (only the segment-boundary input is retained across the whole model, with each layer's actual intermediates freed immediately after use and regenerated on demand) — or `O(sqrt(L))` under the classical Chen et al. 2016 scheme, which deliberately checkpoints only every `sqrt(L)`-th layer rather than every layer, balancing the recompute cost more finely (paying less than the full 33% tax) against a memory bound that's `O(sqrt(L))` rather than strictly `O(1)`. **Selective activation recomputation** (Megatron's refinement, referenced in `..\OpenSource\007_DeepSeek_V3.md`'s broader systems discussion) pushes this further by checkpointing *unevenly within* a layer — recomputing specifically the operations that are cheap to redo but expensive to store (attention's `O(s^2)`-memory score matrix, dominant at long context) while leaving operations with an unfavorable recompute-cost-to-memory-saved ratio uncheckpointed — targeting the ~33% tax specifically at the terms where the trade is genuinely favorable, rather than paying it uniformly everywhere regardless of whether a given operation's memory footprint actually warranted the recompute cost.

## Q7: (Coding) Write a function that estimates total per-GPU memory (model states + activations) for a training configuration, and use it to determine the maximum feasible micro-batch size for a fixed memory budget.

```python
def per_gpu_memory_gb(
    total_params: float, dp_degree: int, tp_degree: int, pp_degree: int, zero_stage: int,
    num_layers: int, hidden_dim: int, num_heads: int, micro_batch_size: int, seq_len: int,
    activation_checkpointing: bool = True,
) -> float:
    local_params = total_params / (tp_degree * pp_degree)
    per_param_bytes_unsharded = 16.0   # 2(bf16 w) + 2(bf16 g) + 4(fp32 w) + 4(m) + 4(v)

    if zero_stage == 0:
        model_state_bytes = local_params * per_param_bytes_unsharded
    elif zero_stage == 1:
        model_state_bytes = local_params * 4.0 + local_params * 12.0 / dp_degree
    elif zero_stage == 2:
        model_state_bytes = local_params * 2.0 + local_params * 14.0 / dp_degree
    elif zero_stage == 3:
        model_state_bytes = local_params * per_param_bytes_unsharded / dp_degree
    else:
        raise ValueError("zero_stage must be 0-3")

    local_layers = num_layers / pp_degree
    # Megatron-style per-layer activation formula; checkpointing collapses the L-scaling
    # to an O(1)-per-in-flight-microbatch approximation (simplified: one layer's worth,
    # scaled by a small constant for in-flight pipeline stages, rather than local_layers).
    per_layer_activation_bytes = micro_batch_size * seq_len * hidden_dim * (
        34 + 5 * num_heads * seq_len / hidden_dim
    )
    if activation_checkpointing:
        activation_bytes = per_layer_activation_bytes * min(local_layers, pp_degree)
    else:
        activation_bytes = per_layer_activation_bytes * local_layers

    return (model_state_bytes + activation_bytes) / 1e9


def max_microbatch_for_budget(
    budget_gb: float, total_params: float, dp_degree: int, tp_degree: int, pp_degree: int,
    zero_stage: int, num_layers: int, hidden_dim: int, num_heads: int, seq_len: int,
    activation_checkpointing: bool = True,
) -> int:
    mb = 1
    while True:
        usage = per_gpu_memory_gb(
            total_params, dp_degree, tp_degree, pp_degree, zero_stage,
            num_layers, hidden_dim, num_heads, mb, seq_len, activation_checkpointing,
        )
        if usage > budget_gb:
            return max(1, mb - 1)
        mb += 1
        if mb > 4096:   # safety bound against runaway loop
            return mb


# Example: 70B model, TP=8/PP=4/DP=16, ZeRO-2, 80GB H100 budget
mb_max = max_microbatch_for_budget(
    budget_gb=80.0, total_params=70e9, dp_degree=16, tp_degree=8, pp_degree=4,
    zero_stage=2, num_layers=80, hidden_dim=8192, num_heads=64, seq_len=4096,
)
print(f"Max micro-batch size within 80GB: {mb_max}")
```

This composes Q3/File-009's memory estimator with the bubble-fraction reasoning from `001` Section 5.2: the returned `mb_max` directly bounds how many in-flight micro-batches 1F1B can afford (`001` Section 4.2), which in turn bounds the achievable bubble fraction (File 009 Q20) — the three calculations are not independent, and a complete answer to "what micro-batch size should I use" has to chain through all three rather than treating memory sizing and bubble-fraction targeting as separate problems.

## Q8: When does CPU/NVMe offloading (ZeRO-Offload/ZeRO-Infinity) make sense, and when is it actively the wrong choice?

Offloading moves sharded optimizer state or parameters out of GPU HBM into host CPU RAM, or further to local NVMe SSD, pulling data back across PCIe only when actively needed for compute. The bandwidth hierarchy makes the cost explicit: HBM delivers several TB/s, NVLink several hundred GB/s to ~900GB/s, PCIe Gen4/5 roughly 32-64GB/s, and NVMe a few GB/s per drive — each tier down is roughly an order of magnitude slower (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md` Section 8; `005_Cluster_Hardware_Networking_And_Interconnect.md`'s bandwidth hierarchy). Offloading is the right call specifically when **GPU count** (and therefore aggregate HBM and aggregate NVLink/IB bandwidth to shard state across) is the scarce resource and host RAM is comparatively abundant and idle — the canonical case is fine-tuning or research iteration on a single node or a handful of GPUs, where there is no larger DP/TP/PP grid available to shard across at all, and the honest alternative to offloading is not "use more GPUs, it's already using what's available" — it's "this job does not run." Offloading is actively the wrong choice for frontier-scale pretraining on clusters with thousands of GPUs, because at that scale the aggregate HBM and NVLink/IB bandwidth across the fleet is *already* the intended resource ZeRO-3 and the 3D parallel grid (`001_Parallelism_Strategies_Data_Tensor_Pipeline.md`) are designed to shard state across at full-speed GPU-to-GPU bandwidth — deliberately routing that same state through the much slower PCIe/CPU/NVMe path instead, when ample faster GPU-to-GPU capacity sits idle nearby, trades away throughput for memory headroom the cluster didn't actually need to buy that way. The crisp framing: **offloading exchanges GPU-memory scarcity for host-bandwidth scarcity, and is worth it exactly when you are memory-constrained but not GPU-count-constrained** — which is close to the opposite of the typical frontier-pretraining regime, where GPU-count efficiency is precisely the thing a lab is trying to maximize, not the resource it has slack in.

## Q9: Scenario — a training job hangs mid-step; every rank's stack trace shows it blocked inside the same all-reduce call, at the same point in the code, with no rank appearing to have taken a different path. Diagnose, given that this rules out the classic "mismatched collective ordering" cause.

If every rank is genuinely blocked inside the *same* collective call at the *same* logical point — ruling out the code-divergence deadlock from `008_Debugging_Distributed_Training_Failures.md` Section 2 — the remaining candidates shift toward the network layer itself rather than application logic. **Check first**: `NCCL_DEBUG=INFO` output for the transport actually selected on every rank — if a subset of nodes silently fell back from InfiniBand to TCP sockets (a misdetected NIC, a driver mismatch, IB verbs unavailable inside a container after an image change), the all-reduce isn't technically deadlocked, it's just proceeding at TCP-fallback speed, which at the data volumes involved can be slow enough to be indistinguishable from a hang on any reasonable human timescale — this is the single most common "looks exactly like a hang but isn't" cause and should be ruled out before assuming a genuine deadlock. **If transport looks correct**, check for a **network partition**: a failed spine switch or cable can partition connectivity between specific node groups even while every individual node reports itself healthy and every NIC link-status check passes locally — the all-reduce genuinely cannot complete because some pair of ranks that need to communicate literally cannot reach each other, and this specific fault is invisible to any single-node health check, only detectable by testing actual reachability *between* the specific node pairs involved (a targeted `ib_write_bw` or similar point-to-point bandwidth/connectivity test between the ranks the stuck collective spans). **If reachability checks pass everywhere**, consider **resource exhaustion at scale** — a fixed-size hardware or software resource (available NIC queue pairs, GPU-side communication buffer pools sized for a smaller world size than currently deployed) can be silently exhausted only once the job scales past some threshold, producing a hang that is reproducible at the current scale but was never seen at a smaller scale during earlier testing, which is exactly the kind of failure that a bisection test (rerun on half the nodes) will reveal by *not* reproducing at smaller scale, pointing squarely at a scale-dependent resource limit rather than a specific bad node or link. The key discipline this scenario tests: don't stop investigating just because the most commonly-cited cause (code divergence) has been ruled out — the space of "everyone's stuck in the same place" causes is itself a checklist (transport fallback, partition, resource exhaustion), not a single diagnosis.

## Q10: Explain bisection bandwidth and why an oversubscribed fat-tree network topology can make otherwise-identical training jobs perform very differently depending on rank placement.

A fat-tree (or similar multi-tier folded) network topology connects groups of nodes to "leaf" switches, and leaf switches up to "spine" switches. **Bisection bandwidth** is the worst-case aggregate bandwidth available if the cluster were split into two equal halves and every node in one half needed to communicate with a node in the other half simultaneously — it's a property of how many spine-level links exist relative to how many nodes' traffic might need to cross them. A **full-bisection** (non-oversubscribed) design provisions enough spine capacity that any such worst-case traffic pattern still achieves full per-NIC line rate; an **oversubscribed** design — common because full bisection bandwidth is expensive to provision at very large node counts — provisions less spine capacity than the worst case would need, meaning nodes attached to the *same* leaf switch can communicate at full line rate with each other, while nodes that must cross the spine to reach nodes under a *different* leaf switch contend for a comparatively scarce shared resource and see markedly lower effective bandwidth. This directly determines training performance because the parallel groups from `001_Parallelism_Strategies_Data_Tensor_Pipeline.md` (TP, PP, DP, and EP from `002`) each have a specific communication pattern that either stays local to a leaf-switch's group of nodes or crosses the spine, entirely as a function of **which physical nodes got assigned which logical ranks** — a purely scheduling decision, unrelated to the model or the parallelism algorithm's correctness. Two numerically and algorithmically identical training jobs, differing only in whether their DP group's (or EP group's) members happen to sit under the same leaf switch versus scattered across many leaf switches under an oversubscribed spine, can show meaningfully different achieved throughput for reasons that have nothing to do with anything discussed in `001`-`004` — purely a topology-aware-versus-topology-unaware placement difference (`005_Cluster_Hardware_Networking_And_Interconnect.md` Section 5). This is why production training infrastructure invests in rack-aware/topology-aware schedulers that deliberately co-locate a tightly-communicating group's ranks within the same leaf-switch domain wherever the parallel configuration allows, rather than leaving rank-to-node assignment to an arbitrary or purely load-balancing-driven scheduler that has no visibility into which ranks need to talk to which others most intensely.

## Q11: (Coding) Implement a dynamic loss-scaling mechanism for fp16 training that tracks an inf/nan-triggered skip rate, and expose that skip rate as a diagnostic signal per the discussion in the mixed-precision file.

```python
import torch

class DynamicLossScaler:
    """
    Manages a dynamic scale factor for fp16 mixed-precision training (004_Mixed_Precision...
    Section 2), and additionally tracks a rolling skip rate as an early-warning diagnostic
    signal (004 Section 4): a rising skip rate can precede a visible loss-curve anomaly.
    """
    def __init__(self, init_scale=2.0**15, growth_interval=2000, growth_factor=2.0,
                 backoff_factor=0.5, skip_rate_window=200):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self._good_steps = 0
        self._skip_history = []          # rolling window of booleans: True = step was skipped
        self._skip_rate_window = skip_rate_window

    def _has_overflow(self, grads) -> bool:
        for g in grads:
            if g is not None and not torch.isfinite(g).all():
                return True
        return False

    def step(self, grads) -> bool:
        overflowed = self._has_overflow(grads)
        self._skip_history.append(overflowed)
        if len(self._skip_history) > self._skip_rate_window:
            self._skip_history.pop(0)

        if overflowed:
            self.scale = max(self.scale * self.backoff_factor, 1.0)
            self._good_steps = 0
            return False   # caller must SKIP the optimizer step this iteration
        self._good_steps += 1
        if self._good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self._good_steps = 0
        return True

    @property
    def recent_skip_rate(self) -> float:
        if not self._skip_history:
            return 0.0
        return sum(self._skip_history) / len(self._skip_history)

    def check_alert(self, threshold: float = 0.02) -> bool:
        """Returns True if the rolling skip rate exceeds `threshold` -- worth escalating
        as a leading indicator of instability BEFORE the loss curve itself shows a problem,
        per 004's discussion of loss-scaler skip-rate as a diagnostic signal."""
        return self.recent_skip_rate > threshold


# Usage sketch inside a training loop:
scaler = DynamicLossScaler()
# ... after backward() with a scaled loss ...
# ok = scaler.step([p.grad for p in model.parameters()])
# if not ok: continue  # skip optimizer.step() and zero_grad this iteration
# if scaler.check_alert():
#     log_warning(f"loss-scaler skip rate elevated: {scaler.recent_skip_rate:.1%} over last "
#                 f"{scaler._skip_rate_window} steps -- possible emerging instability")
```

The `check_alert` method operationalizes the point from `004_Mixed_Precision_Training_And_Numerical_Stability.md` Section 4 directly: a skip rate that's occasionally nonzero is normal (the scaler probing the edge of representable range as designed), but a *rising or persistently elevated* rate is itself a symptom worth surfacing to whatever monitoring/alerting watches the training run, independent of and often earlier than any visible anomaly in the loss curve.

## Q12: Why do MoE models typically show lower MFU than dense models at the same activated-parameter count, and is this an argument against using MoE?

MFU (`007_Training_Efficiency_Metrics_MFU_And_Utilization.md`) measures achieved useful FLOPs against theoretical peak; every mechanism that lowers MFU in a dense model (exposed communication, pipeline bubbles, kernel inefficiency, load imbalance) is still present in an MoE model, plus one MoE has that a dense model structurally doesn't: the expert-parallel all-to-all's dispatch/combine communication, which — per `002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md` — is **data-dependent** (its volume and destination pattern are outputs of the router's decision, not statically known) and sits as a synchronization barrier in the critical path, only partially overlappable with compute even with aggressive engineering (DualPipe-style cross-stage overlap). This is a genuine, structural additional communication cost with no dense-model equivalent, so at matched activated-parameter count, an MoE model's MFU is very often measurably lower than a comparable dense model's, purely from this added communication tax (compounded by however well or poorly the router happens to be load-balanced at any given time, per `002` Section 4). **Is this an argument against MoE?** No — and conflating "lower MFU" with "worse overall efficiency" is exactly the mistake to avoid here. MFU measures utilization of *peak hardware FLOPs*, but the entire point of MoE is that it activates far fewer FLOPs per token than a dense model of comparable *total* parameter count (DeepSeek-V3 activates ~37B of 671B total per token, `..\OpenSource\007_DeepSeek_V3.md`) while still benefiting from the larger total parameter capacity for specialization. The metric that actually matters for "was this the right architectural choice" is **quality achieved per unit of wall-clock training compute (or dollar cost)**, not MFU in isolation — a MoE model can have meaningfully lower MFU than a dense model and still be the more compute-efficient (or cost-efficient) choice overall, if its lower per-token activated-FLOPs more than compensates for its lower utilization of whatever FLOPs it does spend. The correct staff-level framing: MFU is a diagnostic for "how well is this specific configuration using the hardware it's running on," not a direct proxy for "was this the right modeling choice" — the two questions are related but distinct, and a good answer here should explicitly decouple them rather than treating "MoE has lower MFU" as a criticism of MoE as an architecture choice.

## Q13: Scenario — a gradient-norm monitoring dashboard shows a small but consistent spike in gradient norm roughly every 500 steps, correlated with the point where the dataloader crosses into a new data shard. Diagnose.

A spike that recurs at a **fixed, predictable interval correlated with a specific, identifiable event** (shard boundaries) rather than appearing randomly is a strong signal that the cause is **data-related, not a generic numerical-stability issue** (`004_Mixed_Precision_Training_And_Numerical_Stability.md`'s failure modes are typically either persistent underflow or acute, non-periodic overflow events — a clean periodicity tied to a specific pipeline event points elsewhere first). The leading hypothesis: something about how shards are constructed or transitioned between introduces a **distributional discontinuity** at shard boundaries — e.g., shards built from contiguous spans of a poorly-shuffled corpus (a shard boundary coinciding with a topic/domain/language transition in the underlying data, producing a batch of unusually out-of-distribution examples relative to the model's recent training trajectory, which can genuinely produce a larger-than-usual gradient without any hardware or precision fault at all), or a genuine data-pipeline bug (e.g., a shard-loading routine that occasionally serves a partially-corrupted or incorrectly-decoded batch specifically at the transition point, a duplicate-data artifact from an off-by-one in shard indexing, or a tokenization edge case triggered by whatever content happens to sit at these specific file boundaries). **Diagnostic steps**: (1) log and directly inspect the actual batch(es) at a few of these transition points — look for anomalies in sequence length distribution, unusual token-ID values, or content that's obviously different in kind from typical batches; (2) check whether the spike magnitude and shard-transition correlation persist if shard order is randomized differently (rules in/out something specific to *this* shard sequence versus something inherent to how shards are built generally); (3) check whether gradient clipping is already absorbing these spikes without consequence (if the clipped gradient norm, post-clip, is unremarkable and the loss curve shows no corresponding disturbance, this may be a benign, already-handled artifact rather than an active problem, and the fix is lower priority than a spike that's actually visible in the loss trajectory too). If confirmed as a genuine data-pipeline artifact rather than a benign, already-clipped effect, the fix belongs in the data pipeline (better shuffling across shard boundaries, or fixing whatever indexing/decoding bug is producing anomalous batches at those specific points), not in the training loop's numerical-stability machinery — an important distinction, since a well-intentioned but misdirected fix (e.g., tightening gradient clipping further to mask the symptom) would hide a real data-quality issue rather than resolving it.

## Q14: Explain the "universal checkpoint" / re-sharding concept in elastic training, and why it's necessary rather than merely convenient.

A checkpoint written under one parallel configuration — a specific TP/PP/DP grid — naturally stores state sharded according to that configuration: under TP, a specific device's shard holds specific rows/columns of specific weight matrices; under ZeRO, a specific device's shard holds a specific slice of optimizer state, determined by that device's position in the DP group. If a job needs to resume with a **different** parallel configuration — a smaller GPU count after losing nodes and not yet having replacements provisioned (`006_Checkpointing_Fault_Tolerance_And_Elastic_Training.md` Section 4), a deliberately different grid for a subsequent training phase, or simply because the exact original topology isn't available at resume time — naively loading the checkpoint's literal on-disk shards into a differently-shaped grid produces **incorrect** results: shard `i` of a checkpoint written under `TP=8` does not correspond to any single, meaningful shard under `TP=4`, and there is no way to correctly reconstruct model state by simply reassigning old shards to new device indices, since the actual *content* of each shard (which rows/columns of which matrices, which slice of which optimizer buffer) was determined by the writing job's specific grid dimensions. **Universal checkpointing** (DeepSpeed's term, with equivalent mechanisms elsewhere) solves this by checkpointing in a more **canonical, re-shardable representation** — either the full, unsharded logical tensors (reconstructed once at checkpoint-write time by gathering shards, then written in a form that any future grid can re-shard from scratch) or a self-describing sharded format that explicitly records enough metadata (which grid dimensions produced this shard, and how it maps to the logical tensor) that a loader targeting a *different* grid can correctly reconstruct and re-partition the full logical state before re-sharding it according to the new configuration. This is necessary, not merely convenient, because without it, elastic recovery is constrained to resuming at the *exact* original parallel configuration — defeating much of the purpose of elastic training, which is specifically to tolerate node loss (and hence a temporarily or permanently different GPU count) without a full, slow, human-mediated restart at the original topology. The engineering cost is real: gathering to (and re-sharding from) a canonical representation is itself extra communication and compute at checkpoint save/load time, a trade accepted specifically because the alternative — being unable to resume at any grid other than the exact original — is a much larger operational cost at the failure rates frontier-scale clusters actually experience (`006` Section 1).

## Q15: Explain interleaved (virtual) pipeline stages and precisely how they reduce the effective bubble fraction without adding physical devices.

Standard pipeline parallelism assigns each physical device one **contiguous** block of layers — with `P` devices and `L` layers, device `p` holds layers `[p*(L/P), (p+1)*(L/P))`. The bubble fraction under this scheme, `(P-1)/(m+P-1)` (`001_Parallelism_Strategies_Data_Tensor_Pipeline.md` Section 4.1), has its fixed fill/drain overhead term set by `P` — the number of physical devices a micro-batch must sequentially traverse. Interleaving breaks the one-device-one-contiguous-block assumption: instead, each physical device is assigned **several smaller, non-contiguous chunks** of the model — e.g., with 4 physical devices and 8 "virtual stages," device 0 might hold layers `{1-2, 17-18}`, device 1 holds `{3-4, 19-20}`, and so on, with a micro-batch now passing through device 0, then device 1, ..., then device 3, then back to device 0 (for its second chunk), then device 1 again, etc. — a micro-batch crosses *more* stage boundaries (`v` virtual stages per device instead of 1), but each individual hop moves through a smaller chunk of the model. The bubble-fraction formula becomes `(P-1)/(m*v + P - 1)` where `v` is the number of virtual stages per physical device: the fill/drain overhead term `(P-1)` (still set by the number of *physical* devices, since that's what determines how many hops are needed before the pipeline is genuinely full end-to-end) is now amortized over `m*v` effective "slots" instead of just `m`, shrinking the bubble fraction for the same `m` and `P` simply by increasing `v`. This achieves what increasing `m` alone achieves (Q5/File 009's lever) but without the corresponding activation-memory cost of holding more *whole-model-depth* micro-batches in flight — since each virtual stage's chunk is smaller, the memory cost of holding a chunk's activations is correspondingly smaller too. The cost interleaving introduces instead: point-to-point activation transfers between devices now happen `v` times more frequently (a micro-batch crosses a device boundary at every virtual-stage transition, not just once per physical stage), which increases the *fixed per-hop* communication overhead paid (each transfer's fixed latency cost, independent of data volume, is now incurred more often) — a real trade, but one that's typically favorable when the per-hop fixed cost is small relative to the bubble-fraction improvement gained, which is generally true within a single node's low-latency NVLink domain but can become less favorable if virtual-stage boundaries are allowed to cross the slower inter-node fabric without care.

## Q16: (Coding) Implement both the naive 6N FLOPs-per-token approximation and the more precise 6N+12Lhs formula, and write a function that reports the percentage divergence between them as a function of sequence length, to show where the naive approximation breaks down.

```python
def flops_per_token_naive(num_params: float) -> float:
    return 6 * num_params


def flops_per_token_precise(num_params: float, num_layers: int, hidden_dim: int, seq_len: int) -> float:
    return 6 * num_params + 12 * num_layers * hidden_dim * seq_len


def divergence_vs_seq_len(num_params: float, num_layers: int, hidden_dim: int, seq_lens: list[int]) -> None:
    """
    Reports how far the naive 6N approximation drifts from the precise formula as
    sequence length grows, per 007_Training_Efficiency_Metrics_MFU_And_Utilization.md
    Section 2's discussion of the attention term the naive approximation omits.
    """
    naive = flops_per_token_naive(num_params)
    print(f"{'seq_len':>8} | {'precise FLOPs/tok':>20} | {'naive FLOPs/tok':>18} | {'divergence':>10}")
    for s in seq_lens:
        precise = flops_per_token_precise(num_params, num_layers, hidden_dim, s)
        divergence = (precise - naive) / precise
        print(f"{s:>8} | {precise:>20.3e} | {naive:>18.3e} | {divergence:>9.2%}")


# 70B model, 80 layers, hidden_dim=8192
divergence_vs_seq_len(
    num_params=70e9, num_layers=80, hidden_dim=8192,
    seq_lens=[512, 2048, 4096, 8192, 32768, 131072],
)
# At short context (512-2048), divergence is a small single-digit percentage -- the naive
# 6N approximation is a fine shorthand. At very long context (131072, i.e. 128K), the
# 12*L*h*s attention term becomes a much larger fraction of total FLOPs/token, and a
# 6N-only MFU calculation would systematically OVERSTATE achieved FLOPs (and hence MFU)
# by omitting real hardware work the attention operation is actually doing.
```

This makes the qualitative warning in `007_Training_Efficiency_Metrics_MFU_And_Utilization.md` Section 2 checkable directly: for a fixed model, the naive approximation degrades specifically as a function of `seq_len` growing relative to `hidden_dim`, which is exactly the regime long-context training (128K-context models, per the KV-cache/context discussions in `..\OpenSource\007_DeepSeek_V3.md`) increasingly operates in — meaning MFU comparisons between a short-context and long-context run computed with the naive formula are not apples-to-apples even for the identical model, a subtlety worth surfacing explicitly rather than assuming any single FLOPs-per-token shorthand travels safely across context-length regimes.

## Q17: Scenario — a model being trained with DeepSeek-V3-style fine-grained FP8 shows a healthy, smoothly decreasing loss curve for the first 5,000 steps, then begins diverging (loss trending upward, not an abrupt NaN spike) over the following few hundred steps. Diagnose, distinguishing this from the abrupt-NaN scenario covered elsewhere.

This is deliberately a different failure signature than the abrupt-NaN case (File 009 Q11): a **gradual upward trend over hundreds of steps**, not a one-or-two-step collapse, points away from a single acute overflow event and toward **accumulating quantization error** or a **systematic bias compounding over many steps** — exactly the failure mode `004_Mixed_Precision_Training_And_Numerical_Stability.md` Section 3 flags as the reason certain operations (long reductions, optimizer state) are deliberately kept out of FP8 even in an otherwise-FP8-heavy recipe: a single low-precision multiply's rounding error is usually tolerable in isolation, but error that compounds across many summed terms or many thousands of steps' worth of updates is a qualitatively different, slower-building problem. **Diagnostic steps, specific to an FP8 recipe**: (1) check whether the tile/block-wise scaling factors (`004` Section 3) have started drifting toward degenerate values for any tensor — e.g., a tile whose local maximum has grown unusually large relative to its historical range would force that tile's scale factor to shrink, pushing every other element in that tile toward FP8's already-narrow mantissa floor; tracking the distribution of per-tile/per-block scale factors over training time (not just at initialization) can reveal whether a specific region of the network has drifted into a regime where fine-grained scaling is no longer adequately protecting precision; (2) check specifically which operations are actually running in FP8 versus which are exempted to higher precision in the current implementation, and cross-reference against DeepSeek-V3's stated recipe (`..\OpenSource\007_DeepSeek_V3.md`) — a divergence between the intended recipe (certain accumulations and optimizer state kept in bf16/fp32) and what's actually implemented is a very plausible, mundane root cause if this is a from-scratch FP8 implementation rather than a directly reused, validated one; (3) run a controlled ablation: resume from a checkpoint shortly before the divergence began, switch the same run to bf16 for a few hundred steps, and compare trajectories — if the bf16 branch remains stable while the FP8 branch continues drifting, this strongly localizes the cause to the FP8 recipe specifically rather than to the model/data/optimizer configuration in general, which would also be visible in the bf16 branch if it weren't precision-specific. Unlike the abrupt-NaN case, where restoring from a nearby checkpoint and tightening gradient clipping is often sufficient, a gradual-drift FP8 divergence usually requires revisiting the *scaling granularity or which operations are exempted*, since the problem is architectural to the precision recipe itself, not a one-off bad batch or an under-clipped gradient spike.

## Q18: Explain, at a mechanistic level, why a generic 1F1B pipeline schedule is insufficient for an MoE model in the way DualPipe is specifically designed to address.

1F1B (`001_Parallelism_Strategies_Data_Tensor_Pipeline.md` Section 4.2) is designed around a specific assumed communication pattern: point-to-point activation handoffs between adjacent pipeline stages, at predictable points (stage boundaries), with a schedule that interleaves each stage's forward and backward work to bound in-flight activation memory. This scheduling logic implicitly assumes the *only* cross-device communication a stage needs to worry about is that stage-boundary handoff. An MoE layer breaks this assumption: within a *single* pipeline stage's forward (and backward) pass, there is now an **additional**, expert-parallel all-to-all (`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`) that must complete before that stage's computation can proceed to its combine step — a second, structurally different communication event nested *inside* what 1F1B treats as an atomic unit of "this stage's forward/backward work." A generic 1F1B schedule has no mechanism for overlapping *this* communication with anything, because it wasn't designed with it in mind; the all-to-all simply becomes exposed, serial latency added on top of whatever bubble 1F1B already accounts for, and — compounding the problem — that added latency is **variable** step to step (since all-to-all volume depends on the current batch's routing decisions, `002` Section 3), making the effective bubble noisier and harder to reason about than the clean, deterministic formula 1F1B was designed around for dense models. DualPipe (`..\OpenSource\007_DeepSeek_V3.md`) is a custom, bidirectional schedule co-designed specifically to give the scheduler visibility into *both* the pipeline-stage handoff communication *and* the MoE all-to-all simultaneously, so that one pipeline stage's or micro-batch's all-to-all can be deliberately overlapped with a *different* stage's or micro-batch's compute — a strictly harder scheduling problem than 1F1B's, because it requires jointly coordinating two distinct communication patterns (point-to-point pipeline handoffs and all-to-all expert dispatch) against a single compute timeline, rather than optimizing the pipeline schedule in isolation and treating MoE communication as an orthogonal, separately-optimized concern. The general lesson worth stating explicitly: once a model has structural communication requirements beyond the ones a generic parallelism-library schedule was designed around, it can be worth co-designing the schedule itself around that specific requirement, rather than assuming off-the-shelf scheduling logic (built for a simpler, dense-model communication pattern) generalizes for free.

## Q19: What is the difference between MFU and HFU, and why does activation checkpointing create a gap between the two?

**MFU (Model FLOPs Utilization)** counts, in its numerator, only the FLOPs a *from-scratch* forward-plus-backward pass conceptually requires for the model as specified — the `6N + 12Lhs`-style formula (`007_Training_Efficiency_Metrics_MFU_And_Utilization.md` Section 2), deliberately independent of whatever the actual hardware implementation happened to do. **HFU (Hardware FLOPs Utilization)** counts, in its numerator, *every* FLOP the hardware actually executed during the step, including any additional recomputation the implementation performs for reasons unrelated to the model's mathematical definition — most notably, the extra forward-pass recomputation activation checkpointing introduces (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md` Section 7). Because checkpointing makes the hardware do genuinely more work (re-running a forward pass to regenerate discarded intermediates) than the mathematically minimal forward-plus-backward computation strictly requires, HFU's numerator is larger than MFU's for any run using checkpointing, while the denominator (hardware peak FLOPs) is identical for both — so **HFU ≥ MFU always, with equality only when no extra hardware-side recomputation (or other non-model-defined extra work) is happening**. The gap between the two is, in fact, a direct, quantifiable measurement of how much a given checkpointing configuration costs in raw compute: if HFU comes out to, say, 45% and MFU comes out to 34% for the same run, that roughly `(45-34)/45 ≈ 24%` relative gap is attributable to checkpointing's recompute overhead specifically (consistent with, though not identical to, the ~33% figure from Q6/File 010, since the exact ratio depends on how uniformly checkpointing was applied and what fraction of total FLOPs the checkpointed segments represent). The practical reason to track both rather than just one: MFU answers "how efficiently is this model's *necessary* computation using the hardware," which is the number to compare across different checkpointing strategies (a strategy with less checkpointing but requiring smaller batch sizes to fit memory might show a different MFU than one with more checkpointing and larger batches, and MFU alone tells you which is using hardware more efficiently for the model's own sake); HFU answers "how efficiently is the hardware being used given everything the current implementation is actually asking it to do," which is closer to the number that determines actual wall-clock training time and cost. Quoting only one of the two, without being explicit about which, is a common source of confusion when comparing reported efficiency numbers across papers or teams with different checkpointing configurations.

## Q20: (Coding) Write a heuristic function that, given a GPU count, nodes-per-GPU-count (NVLink domain size), a model's parameter count, and per-device memory budget, proposes a starting TP/PP/DP grid (ignoring EP for a dense-model case), following the ordered reasoning developed across this module's parallelism discussion.

```python
def propose_parallel_grid(
    total_gpus: int,
    gpus_per_node: int,
    total_params: float,
    per_gpu_memory_budget_gb: float,
    zero_stage: int = 2,
    bytes_per_param_unsharded: float = 16.0,
    activation_headroom_gb: float = 15.0,   # reserve budget for activations, not modeled precisely here
) -> dict:
    """
    Heuristic starting point only -- mirrors the ordered reasoning in
    001_Parallelism_Strategies_Data_Tensor_Pipeline.md Section 5.2:
      1. Fix TP to the NVLink domain size (never span TP across the slower inter-node fabric).
      2. Grow PP from 1 upward until per-GPU model-state memory (post TP+PP sharding,
         and post whatever the chosen ZeRO stage additionally shards across an ASSUMED
         max feasible DP degree) fits within budget minus activation headroom.
      3. Fill the remaining GPU budget with DP.
    Real deployments should follow this with an actual profiling pass -- this function
    proposes a defensible starting point, not a guaranteed-optimal configuration.
    """
    tp = min(gpus_per_node, total_gpus)
    if total_gpus % tp != 0:
        raise ValueError("total_gpus should be evenly divisible by the chosen TP degree")

    remaining_after_tp = total_gpus // tp
    usable_budget_gb = per_gpu_memory_budget_gb - activation_headroom_gb

    def model_state_gb(pp: int, dp: int) -> float:
        local_params = total_params / (tp * pp)
        if zero_stage == 0:
            bytes_per_param = bytes_per_param_unsharded
        elif zero_stage == 1:
            bytes_per_param = 4.0 + 12.0 / dp
        elif zero_stage == 2:
            bytes_per_param = 2.0 + 14.0 / dp
        elif zero_stage == 3:
            bytes_per_param = bytes_per_param_unsharded / dp
        else:
            raise ValueError("zero_stage must be 0-3")
        return local_params * bytes_per_param / 1e9

    pp = 1
    while pp <= remaining_after_tp:
        dp = remaining_after_tp // pp
        if remaining_after_tp % pp == 0 and dp >= 1:
            if model_state_gb(pp, dp) <= usable_budget_gb:
                return {
                    "tp": tp, "pp": pp, "dp": dp,
                    "total_gpus_used": tp * pp * dp,
                    "estimated_model_state_gb": round(model_state_gb(pp, dp), 2),
                    "note": "starting point only -- validate with profiling and bubble-fraction check",
                }
        pp += 1

    raise RuntimeError("No feasible (TP, PP, DP) grid found within the given memory budget; "
                        "consider ZeRO-3, activation checkpointing, or a larger memory budget.")


# Example: 300B model, 2048 GPUs, 8 GPUs/node, 80GB H100s, ZeRO-2
grid = propose_parallel_grid(
    total_gpus=2048, gpus_per_node=8, total_params=300e9,
    per_gpu_memory_budget_gb=80.0, zero_stage=2,
)
print(grid)
```

This directly encodes the five-step reasoning from `001` Section 5.2 and the follow-up worked example in File 010 Q1 into a runnable heuristic: TP is fixed first and never treated as a free variable to search over (consistent with the "TP degree ≤ GPUs per node" rule from `001`/`005`), PP is grown only as far as memory strictly requires (consistent with `001`'s point that PP should be minimized to control bubble fraction, not maximized for its own sake), and DP absorbs whatever GPU budget remains — with the function's docstring and returned `note` field deliberately flagging that this is a starting point requiring a bubble-fraction and profiling-based validation pass (per `007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s diagnostic framework), not a final answer a real deployment should trust without further measurement.
