# ZeRO, Optimizer Sharding, and Memory Management

## 1. The four memory consumers

Every byte of GPU HBM consumed during training falls into one of four categories. Getting the
arithmetic for each exactly right is a prerequisite for every other topic in this file, and it is
exactly the calculation `..\GPT\003_GPT3.md` performs informally for GPT-3's 175B — this section
makes it precise and general.

Let `Ψ` = number of parameters, and assume the now-standard mixed-precision training recipe: bf16
weights and gradients for compute, fp32 "master" weights and fp32 Adam optimizer state for the
actual update (the *why* of this split is covered in full in
`004_Mixed_Precision_Training_And_Numerical_Stability.md`; here it is simply taken as given so the
memory arithmetic is concrete).

**1. Parameters.** One bf16 copy for forward/backward compute (2 bytes/param) plus one fp32 "master"
copy that the optimizer actually updates (4 bytes/param) = **6 bytes/param**.

**2. Gradients.** One bf16 copy accumulated during backward (2 bytes/param) — some implementations
accumulate gradients in fp32 for numerical safety, which would add another 4 bytes/param here
instead; the widely-quoted "ZeRO number" below uses the bf16/fp16-gradient convention. = **2
bytes/param** (convention-dependent; see note above).

**3. Optimizer state (Adam).** First moment `m` (fp32, 4 bytes/param) and second moment `v` (fp32, 4
bytes/param) = **8 bytes/param**.

Summing under this convention: `6 (params) + 2 (grad) + 8 (optimizer) = 16 bytes/param` — the figure
the original ZeRO paper (Rajbhandari et al., 2019) quotes for **model states** under mixed-precision
Adam. This is the number to reproduce from memory in an interview: `16Ψ` bytes for model states
alone, independent of batch size or sequence length. For a 70B model: `70 × 10^9 × 16 = 1.12TB`. For
GPT-3's 175B: `175 × 10^9 × 16 = 2.8TB` — matching, up to convention differences in exactly how the
paper apportions the fp32 weight copy, the ~2.45TB figure `..\GPT\003_GPT3.md` derives independently
from first principles (that document's convention nets 14 bytes/param by not double counting a
separate fp32-weights-plus-bf16-weights split the same way; both are correct under their own stated
conventions — the point to internalize is the order of magnitude and the linear-in-`Ψ` scaling, not
a single canonical byte count).

**4. Activations.** Unlike the first three (which scale only with `Ψ`), activation memory scales
with `batch_size × sequence_length × depth × d_model` — it is a *usage-pattern-dependent* cost, not
a fixed model-state cost, and it is what Section 5 (activation checkpointing) directly targets. The
Megatron-LM paper's formula for a single transformer layer's stored activations (without any
checkpointing), in bytes, is approximately:

```
activation_bytes_per_layer ≈ b * s * h * (34 + 5 * a * s / h)
```

where `b` = micro-batch size, `s` = sequence length, `h` = hidden dimension, `a` = number of
attention heads (the `5*a*s/h` term captures the attention score matrix's `O(s^2)` memory, which
becomes dominant at long context lengths). Multiply by `L` (number of layers) for the full model's
unsharded activation memory. This term is why, at long context, activation memory — not
parameter/optimizer state — can dominate total memory, and why activation checkpointing (Section 5)
is not an optional nicety but frequently the single highest-leverage memory lever available.

## 2. What ZeRO actually is: eliminating DP's redundancy, not adding new communication

Return to pure data parallelism (`001_Parallelism_Strategies_Data_Tensor_Pipeline.md`, Section 2):
every one of the `N_d` DP replicas holds a **full, identical copy** of all `16Ψ` bytes of model
state. This redundancy is the entire target of ZeRO (Zero Redundancy Optimizer). The core idea is
simple to state and easy to underestimate: **there is no reason every replica needs its own copy of
state that every replica updates identically anyway** — after the gradient all-reduce, every replica
computes the *same* optimizer update from the *same* averaged gradient, so if instead each replica
is responsible for correctly updating only `1/N_d` of the parameters (and holds only the
corresponding `1/N_d` slice of optimizer state), the *aggregate* memory required across the DP group
drops by a factor of `N_d`, with the missing pieces reconstructed via communication exactly when
needed. ZeRO's three stages differ in *how much* of the `16Ψ` gets sharded this way, trading
progressively more communication for progressively less memory.

## 3. Stage 1: optimizer state partitioning

**What's sharded:** the fp32 Adam state (`m`, `v`) and the fp32 master-weight copy — the `8 + 4 =
12` bytes/param that only the optimizer step itself touches. Each of the `N_d` replicas owns and
updates only `1/N_d` of these buffers.

**What's not sharded:** the bf16 parameters and bf16 gradients are still fully replicated on every
device, exactly as in plain DP.

**Mechanics:** the forward and backward passes are completely unchanged from plain DP — every
replica computes gradients for the *full* parameter set using its full local bf16 weight replica.
The gradient all-reduce is also unchanged in *effect* (every replica still needs the fully-averaged
gradient for every parameter, since it still holds a full bf16 weight copy that needs a full
gradient to update) but is typically implemented as **reduce-scatter** rather than all-reduce:
reduce-scatter delivers the fully-reduced gradient for parameter shard `i` *only* to the replica
that owns shard `i`'s optimizer state, rather than to every replica. That replica then runs the
optimizer step on just its owned shard (using its owned `1/N_d` of `m`, `v`, and the fp32 master
weights), producing updated fp32 master weights for that shard, casts them down to bf16, and
**all-gathers** the updated bf16 weights so every replica ends the step with an up-to-date, full
bf16 parameter copy again for the next forward pass.

**Memory:** `2Ψ (bf16 params) + 2Ψ (bf16 grad) + 12Ψ/N_d (sharded optimizer state)` per device —
versus `16Ψ` for plain DP.

**Communication:** one reduce-scatter (gradients) + one all-gather (updated weights) per step. This
has the *same total data movement* as a plain DP all-reduce (recall from `001` that an all-reduce is
implemented internally as exactly reduce-scatter + all-gather) — **ZeRO-1 shards memory at
essentially no extra communication cost over plain DP.** This is the reason ZeRO-1 is close to a
strict Pareto improvement over plain DP whenever you have a DP group at all, and is close to a "why
would you not do this" default in modern training stacks.

## 4. Stage 2: + gradient partitioning

**What's additionally sharded:** the bf16 gradient buffer. Instead of every replica ending backward
with a full bf16 gradient for every parameter, each replica only needs to *retain* the gradient for
the parameter shard it owns (since Section 3 already established that only the owning replica's
optimizer step consumes that shard's gradient).

**Mechanics:** during backward, as each layer's gradient becomes available, it is reduce-scattered
immediately (same overlap-with-backward strategy as `001` Section 2.3) directly into the
shard-owner's buffer; non-owning replicas can **discard** their contribution to that gradient once
it's been sent, rather than retaining a full local gradient buffer for the whole model. Forward pass
and the bf16-weight all-gather are unchanged from Stage 1.

**Memory:** `2Ψ (bf16 params, still fully replicated) + (2 + 12)Ψ/N_d (sharded gradients + optimizer
state)` = `2Ψ + 14Ψ/N_d` per device.

**Communication:** identical total volume to Stage 1 (still one reduce-scatter + one all-gather per
step in aggregate) — the only change from Stage 1 is *when* memory is freed, not how much data
moves. Stage 2 is thus also "free" relative to Stage 1 in communication-volume terms, and is
generally enabled by default alongside Stage 1 in most training stacks (the combination is usually
just referred to as "ZeRO-2").

## 5. Stage 3: + parameter partitioning (the FSDP-equivalent)

**What's additionally sharded:** the bf16 parameters themselves. No device holds a full copy of the
model's weights at rest — each device owns a `1/N_d` shard of every parameter tensor, all the time.

**Mechanics:** this is the qualitative jump. Because forward and backward computation for a given
layer needs that layer's *full* weight tensor (matrix multiplication needs the whole matrix, not an
arbitrary 1/N_d slice of it), Stage 3 must **all-gather the full parameters for a layer,
just-in-time, immediately before that layer's forward computation**, use them, and then **discard
the non-owned portion immediately afterward** to free the memory back up — repeating this all-gather
again before that layer's backward computation. This is precisely what PyTorch's **FSDP (Fully
Sharded Data Parallel)** implements; "ZeRO-3" and "FSDP" describe the same underlying idea
(just-in-time parameter materialization via all-gather, shard-owned by default) from two different
software lineages, and the terms are often used interchangeably at staff level — it is a reasonable
interview move to state this equivalence explicitly.

```
# ZeRO-3 / FSDP forward pass for one layer, from a single device's perspective
full_weights = all_gather(local_weight_shard, dp_group)   # materialize full layer just-in-time
output = layer_forward(full_weights, input_activations)
del full_weights                                          # free immediately — do NOT hold full copy
# ... same pattern repeats for backward: all_gather again, compute grad, then
# reduce_scatter the grad directly into the owning shard (as in Stage 2) and discard again
```

**Memory:** `16Ψ/N_d` per device — full linear scaling of *all* model-state memory with `N_d`, the
theoretical best case for pure state-sharding (activations are a separate axis, Section 6).

**Communication cost — the real trade-off.** Stage 3 pays for that additional memory reduction with
**additional all-gathers**: one all-gather per layer in the forward pass (to materialize weights)
and one more all-gather per layer in the backward pass (weights are needed again to compute that
layer's input gradient), on top of the reduce-scatter for gradients that Stages 1/2 already pay. The
ZeRO paper's own accounting puts Stage 3's total communication volume at roughly **1.5×** a plain DP
all-reduce's volume (the extra 0.5× coming from the additional forward-pass all-gather that Stages
1/2 don't need, since they keep a full bf16 weight replica at rest specifically to avoid this). This
is a real, measurable throughput cost, not a free upgrade — which is why Stage 3 is chosen when
Stage 1/2's `2Ψ` of still-replicated bf16 parameters is itself the memory bottleneck (very large
models relative to per-device HBM, or a large TP/PP grid leaving few devices for the DP axis to
shard across), not applied unconditionally as a default the way Stage 1/2 typically are.

## 6. Summary table

| Stage | Sharded | Replicated | Memory (model states) | Extra comm vs. plain DP |
|---|---|---|---|---|
| Plain DP | nothing | params, grads, optimizer state | `16Ψ` | — (baseline) |
| ZeRO-1 | optimizer state (12 bytes/param) | bf16 params, bf16 grads | `2Ψ + 12Ψ/N_d` | ~0 (reduce-scatter+all-gather = same volume as all-reduce) |
| ZeRO-2 | + gradients (2 bytes/param) | bf16 params | `2Ψ + 14Ψ/N_d` | ~0 (same as ZeRO-1, just freed earlier) |
| ZeRO-3 / FSDP | + params (2 bytes/param) | nothing | `16Ψ/N_d` | ~1.5x plain DP (extra per-layer all-gathers) |

The practical decision rule this table encodes: **ZeRO-1/2 are essentially free wins and should be
the DP-axis default whenever a DP group exists at all; ZeRO-3 is reached for specifically when `2Ψ`
of residually-replicated bf16 state is itself the binding memory constraint**, accepting the
throughput cost documented above in exchange for the additional headroom.

## 7. Activation checkpointing (gradient checkpointing): trading compute for memory

Sections 1–6 all address the `16Ψ`-scale "model state" memory. Activation memory (Section 1's fourth
category) is an orthogonal axis, and at long sequence lengths or large micro-batch sizes, it can
dominate. **Activation checkpointing** (Chen et al. 2016, "Training Deep Nets with Sublinear Memory
Cost") is the standard technique, and the mechanism is worth stating precisely rather than
hand-wavily, since "trades compute for memory" alone is not a complete answer at staff level.

**The mechanism.** Backpropagation, in the standard (non-checkpointed) case, requires every
intermediate activation produced during the forward pass to be *retained in memory* until the
corresponding backward computation consumes it (because the backward pass for a given operation
typically needs that operation's forward-pass input or output to compute its local gradient — e.g.,
the backward of `y = GELU(x)` needs `x`). For a model with `L` layers, this means activation memory
scales as `O(L)`.

Activation checkpointing breaks this by **not** retaining most intermediate activations. Instead,
the forward pass is divided into segments (commonly, one segment per transformer layer). For each
segment, only the **input activation at the segment boundary** is retained; every intermediate
activation *inside* the segment is discarded immediately after use in the forward pass. When
backward reaches a checkpointed segment, it **re-runs that segment's forward pass** (using the
retained boundary input, which is by definition still available) to regenerate the intermediate
activations it needs, uses them immediately for the local backward computation, and discards them
again once done.

```
# Checkpointed layer, conceptually
def checkpointed_layer_forward(x, layer_fn):
    # Forward pass: run the layer, but do NOT save intermediate activations for backward.
    # Only `x` (already the caller's responsibility to keep) is needed to reconstruct later.
    with torch.no_grad():
        y = layer_fn(x)
    return y, x   # only the boundary input is retained; layer_fn's internals are freed

def checkpointed_layer_backward(x_saved, grad_y, layer_fn):
    # Backward pass: RE-RUN forward (with grad tracking on this time) to regenerate
    # the intermediates the backward computation actually needs.
    x_saved.requires_grad_(True)
    with torch.enable_grad():
        y = layer_fn(x_saved)
    grad_x = torch.autograd.grad(y, x_saved, grad_outputs=grad_y)
    return grad_x
```

**The cost.** Each checkpointed segment's forward computation is run **twice**: once during the
original forward pass (discarding intermediates), and once again during backward (to regenerate
them). For a model checkpointed at every layer, this adds roughly one extra forward pass' worth of
FLOPs on top of the standard forward+backward cost — commonly quoted as **~33% more total compute**
(since standard training is roughly `1 forward + 2 backward-equivalent` units of FLOPs, i.e., 3
units, and full checkpointing adds 1 more forward unit, making 4 units — some sources state this
more precisely per architecture, but ~30% is a reasonable order-of-magnitude default to state in an
interview). In exchange, activation memory drops from `O(L)` to `O(1)` per checkpointed segment
(only the segment-boundary inputs are retained across all `L` layers, i.e., `O(L)` boundary tensors
but each is far smaller than a full segment's internal activations — or `O(sqrt(L))` under the
classical Chen et al. scheme that checkpoints only every `sqrt(L)`-th layer and keeps the rest,
balancing recompute cost against memory more finely than "checkpoint literally everything").

**Selective activation recomputation** (Megatron-LM's refinement, referenced in
`..\OpenSource\007_DeepSeek_V3.md`'s broader systems context) applies checkpointing *unevenly*
within a layer rather than uniformly across all layers: recompute only the specific sub-operations
that are cheap to recompute but expensive to store (e.g., attention's softmax and the `s × s` score
matrix, whose memory scales with `O(s^2)` and dominates at long context per Section 1's formula)
while *not* checkpointing operations that are comparatively expensive to recompute relative to what
they'd cost to just keep in memory (e.g., some matmul outputs where the FLOPs-to-bytes-saved ratio
is unfavorable). This targets the compute-for-memory trade specifically at the terms that benefit
most, rather than paying the full ~33% recompute tax uniformly for operations where the trade is a
poor one.

## 8. CPU and NVMe offloading: a further, more extreme memory-for-speed trade

When even ZeRO-3's `16Ψ/N_d` per-device model-state memory (plus activations) doesn't fit —
typically because `N_d` is small (few GPUs, e.g., single-node fine-tuning of a large model) rather
than because the cluster genuinely lacks aggregate memory — the next lever is **offloading**: keep
the sharded state not in GPU HBM at all, but in host **CPU RAM**, or, if that is also insufficient,
on local **NVMe SSD**, moving pieces across PCIe to the GPU only when actively needed for compute
(ZeRO-Offload and ZeRO-Infinity, Rajbhandari et al., are the reference implementations of this idea
layered on top of ZeRO-3's sharding).

**Why this is a last resort, not a default, at frontier scale — the bandwidth math.** Recall from
`005_Cluster_Hardware_Networking_And_Interconnect.md` the rough bandwidth hierarchy: HBM (on-package
GPU memory) delivers on the order of several TB/s; NVLink (intra-node GPU-to-GPU) delivers hundreds
of GB/s; PCIe (GPU-to-host-CPU, the link offloading depends on) delivers on the order of tens of
GB/s (PCIe Gen4 x16 ≈ 32GB/s, Gen5 x16 ≈ 64GB/s); NVMe SSD read/write throughput is typically a few
GB/s per drive. Every step down this hierarchy is roughly an order of magnitude slower than the one
above it. Offloading optimizer states or parameters to CPU RAM means every time that state is
touched (the optimizer step, or — under parameter offloading — every layer's forward/backward), the
data must cross the PCIe bottleneck, at bandwidth roughly **1–2 orders of magnitude below** what
moving the same data over NVLink within a node would cost. NVMe offloading adds another such drop on
top.

**When it's the right call anyway.** Offloading is a genuinely good trade specifically when GPU
*count* (and hence aggregate HBM and aggregate NVLink/IB bandwidth) is the scarce resource and host
RAM is comparatively abundant and idle — the canonical case being fine-tuning or research
experimentation on a single node or a small handful of GPUs, where there simply is no larger
DP/TP/PP grid to shard across, and the alternative to offloading is not "use more GPUs," it's "this
job does not run at all." It is close to never the right call for frontier pretraining runs on
clusters with thousands of GPUs, precisely because at that scale the aggregate HBM and NVLink/IB
bandwidth across the fleet is already the intended resource to shard across (via ZeRO-3 and the 3D
grid from `001`), and deliberately routing state through the much slower PCIe/CPU/NVMe path when
ample faster GPU-to-GPU capacity exists nearby is a strictly worse trade than just using that
capacity. The staff-level framing: **offloading exchanges GPU-memory scarcity for host-bandwidth
scarcity, and is worth it exactly when you are memory-constrained but not GPU-count-constrained** —
the opposite of the typical frontier-pretraining regime, where GPU count itself is the number a lab
is fighting to make more efficient use of.

## 9. Putting it together: a worked memory estimate

Take a 70B-parameter dense model, `N_d = 16` (matching the DP degree chosen in `001`'s worked
3D-parallelism example, where TP=8, PP=4 leaves DP=16 out of 512 GPUs), bf16 mixed precision,
sequence length 4096, micro-batch size 2, ZeRO-2.

- Model states (ZeRO-2): `2Ψ + 14Ψ/N_d` where `Ψ = 70×10^9 / (TP × PP) = 70×10^9/32 ≈ 2.19×10^9`
(the *local* parameter count each TP/PP-sharded device actually holds, since ZeRO's `N_d` sharding
applies within the DP group of an already TP/PP-sharded model replica) → `2×2.19e9 + 14×2.19e9/16 ≈
4.38GB + 1.92GB ≈ 6.3GB` per device for model states.
- Activations (assume checkpointed at every layer, so roughly `O(1)` per in-flight layer rather than
`O(L)`; using the Megatron formula from Section 1 for a single layer's boundary activation at `b=2,
s=4096, h≈8192` for a 70B-scale model's per-TP-shard hidden size): a few hundred MB to low
single-digit GB per device, depending on exact checkpointing granularity and micro-batch count held
in flight under the pipeline schedule (`001`, Section 4.2's 1F1B bound of roughly `P` in-flight
micro-batches' worth).
- **Total**: comfortably within an 80GB H100's budget with wide margin left for a larger micro-batch
or less aggressive checkpointing — which is exactly the kind of arithmetic that should be run
*before* committing to a parallel configuration, not discovered via an OOM crash after a multi-hour
job launch.

This worked estimate is the connective tissue between this file and `001`: the parallel grid
determines `Ψ_local` (how many parameters, and hence how much model state, a given device is
responsible for at all), and ZeRO stage determines how much of *that* local state must additionally
be replicated versus sharded across the DP axis. Getting a training job's memory budget right
requires composing both calculations, not treating them as independent.
