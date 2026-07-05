# Parallelism Strategies: Data, Tensor, and Pipeline Parallelism

## 1. Why a single GPU is never enough, and why parallelism is not one thing

Before touching any specific technique, fix the actual constraint that forces distributed training
to exist at all: **HBM capacity**, not compute. A single H100 has 80GB of HBM. A 70B-parameter model
in bf16 already needs 140GB just for weights, before a single gradient, optimizer state, or
activation tensor is allocated. `..\GPT\003_GPT3.md` works this arithmetic out concretely for
GPT-3's 175B: 350GB of bf16/fp16 weights, plus roughly 2.45TB of mixed-precision Adam state (fp32
master weights + fp32 first and second moment buffers), before activations are even considered. No
accelerator built to date holds that in local memory. The conclusion is structural, not a matter of
optimization: **the model's state has to be split across many devices' memories, and the computation
has to be split correspondingly.**

"Parallelism" in this context is a design space with (at least) four largely independent axes, each
answering a different question about *what gets split*:

- **Data parallelism (DP):** split the *batch*. Every device holds a full copy of the model and
processes a different slice of the data.
- **Tensor parallelism (TP):** split individual *weight matrices* (and their corresponding
activations) within a layer, across devices, so that no single device holds a full copy of any one
layer.
- **Pipeline parallelism (PP):** split the *layers* (depth) of the model across devices, so each
device holds a contiguous subset of the full stack.
- **Expert parallelism (EP):** specific to Mixture-of-Experts models, splits the *experts* of an MoE
layer across devices. Covered in depth in
`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`; this file treats it as a fourth axis to
be aware of but does not develop the mechanics here.

These axes are orthogonal and composable — a frontier training run does not pick one, it picks a
*combination*, usually called **3D parallelism** (TP × PP × DP) or **4D** once expert parallelism is
added for an MoE model. The rest of this document develops each axis's mechanics precisely enough to
reason about its communication pattern and cost, then works through how to actually choose a
concrete 3D configuration for a given model size and cluster size — the kind of exercise a staff
interview will ask you to do live.

A framing that will recur throughout this module: every parallelism strategy is a trade between
**memory saved**, **communication introduced**, and **idle time (bubbles) created**. There is no
free axis. Choosing a parallel configuration is choosing which of these costs you can most afford
given your specific model shape and network topology.

## 2. Data Parallelism

### 2.1 The basic mechanism

Data parallelism is conceptually the simplest axis: replicate the entire model (weights, optimizer
state) on `N` devices. Split a global batch of size `B` into `N` local batches of size `B/N`. Each
device runs a full forward and backward pass on its local batch, producing a local gradient `g_i`
for every parameter. Before the optimizer step, the gradients must be **synchronized** — averaged
across all `N` replicas — so that every replica applies an identical update and stays bit-for-bit
consistent (this consistency is what makes DP correct: without it, replicas would drift into `N`
different models).

```
for each replica i in parallel:
    x_i = batch_shard(global_batch, i, N)
    loss_i = forward(model, x_i)
    g_i = backward(loss_i)          # local gradient, full model shape
g_avg = all_reduce_mean(g_i)        # synchronization point, ALL replicas participate
for each replica i in parallel:
    optimizer_step(model, g_avg)    # every replica ends up bit-identical
```

Mathematically, this is only correct because the loss is a sum (or mean) over examples, so the
gradient of the mean loss over the global batch equals the mean of the per-shard gradients. DP is an
exact, lossless way to increase batch size and throughput — it does not approximate anything,
provided the all-reduce truly averages (not just sums, or forgets to divide by `N`) and provided
batch normalization statistics (irrelevant for transformers with LayerNorm/RMSNorm, which normalize
per-example) don't introduce cross-replica dependence.

### 2.2 Ring all-reduce: the actual communication mechanics

The synchronization step needs an **all-reduce**: every device ends up with the sum (or mean) of a
value that started out different on every device. The naive implementation — every device sends its
gradient to a coordinator, which sums and broadcasts back — makes the coordinator a bandwidth
bottleneck that scales linearly with `N`. The standard bandwidth-optimal algorithm is **ring
all-reduce** (popularized by Baidu's 2017 paper, and the default primitive inside NCCL):

Arrange the `N` devices in a logical ring. All-reduce decomposes into two phases, each a sequence of
`N-1` steps:

1. **Reduce-scatter phase.** Split each device's gradient buffer into `N` equal chunks. In step `k`,
device `i` sends chunk `k` to its ring-neighbor `i+1` and simultaneously receives a chunk from
`i-1`, adding the received chunk into its own local accumulator for that chunk index. After `N-1`
steps, device `i` holds the *fully reduced sum* for exactly one chunk (chunk index `i`), but not the
others.
2. **All-gather phase.** Now that every device holds one fully-reduced chunk, another `N-1` steps
circulate those finished chunks around the ring so that every device ends up with all `N` finished
chunks — i.e., the complete reduced buffer.

```
def ring_allreduce(local_grad, rank, world_size, ring_comm):
    chunks = split(local_grad, world_size)
    # Phase 1: reduce-scatter
    accum = chunks[rank]
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        send(ring_comm.next, chunks[send_chunk_idx])
        received = recv(ring_comm.prev)
        chunks[recv_chunk_idx] += received
    # chunks[rank] is now the fully-reduced chunk `rank`
    # Phase 2: all-gather the finished chunks around the ring
    for step in range(world_size - 1):
        send_chunk_idx = (rank - step) % world_size
        recv_chunk_idx = (rank - step - 1) % world_size
        send(ring_comm.next, chunks[send_chunk_idx])
        chunks[recv_chunk_idx] = recv(ring_comm.prev)
    return concat(chunks)
```

**Cost model.** Let `Ψ` = number of bytes in the gradient buffer (parameter count × bytes/param).
Each device sends and receives `2(N-1)/N × Ψ` bytes total across the two phases. As `N → ∞`, this
approaches `2Ψ` bytes moved per device — i.e., **the communication volume per device is independent
of `N`** and depends only on model size. This is the crucial scaling property: doubling the number
of DP replicas does not double each replica's communication burden; it stays roughly constant
(asymptotically), which is why DP scales to large replica counts far more gracefully than TP does
(Section 3).

### 2.3 Overlapping communication with backward compute

A naive DP implementation waits for the *entire* backward pass to finish, then issues one all-reduce
over the whole gradient buffer — wasting the time when the network is idle during backward and the
compute is idle during the all-reduce. Production DDP implementations (PyTorch DDP, Megatron's DP
layer) instead **bucket** parameters into groups and kick off the all-reduce for a bucket as soon as
that bucket's gradients are fully computed, while backward continues computing gradients for earlier
layers. Because backward proceeds from the output layer to the input layer, and buckets near the
output finish first, this pipelines communication underneath ongoing compute:

```
backward pass (output → input layer order):
  layer_N grad ready → enqueue all_reduce(bucket containing layer_N) [async, non-blocking]
  layer_{N-1} grad ready → enqueue all_reduce(bucket containing layer_{N-1})   # overlaps with above
  ...
  layer_1 grad ready → enqueue all_reduce(bucket containing layer_1)
optimizer_step()   # must wait for ALL buckets' all-reduces to complete first
```

At sufficient bucket granularity and network bandwidth, this overlap can hide most or all of the
all-reduce cost behind backward compute — one of the first things a step-time profile should be
checked for (see `007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s diagnostic framework).

### 2.4 What DP does not solve

Pure DP replicates the *entire* model state on every device: full weights, full gradients, full
optimizer state. It increases achievable batch size and (up to communication limits) throughput, but
it does **nothing** for the memory-wall problem from Section 1 — every replica still needs to fit
the whole model plus optimizer state in its own HBM. This is exactly the gap ZeRO
(`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`) closes: it takes the DP axis and shards the
*state* (optimizer state, gradients, and optionally parameters) across the DP group instead of
replicating it, while keeping the same communication primitive (all-reduce, decomposed into
reduce-scatter + all-gather) but distributing which device stores which fully-reduced piece. Pure
"replicate everything" DP as described above is really only viable once the per-replica model is
already small enough to fit — which for frontier-scale models means DP is always used *in
combination with* TP and/or PP, never alone.

## 3. Tensor Parallelism

### 3.1 The problem TP solves

DP shards the batch but not the model. TP goes the other way: it shards individual weight matrices
*within* a layer across devices, so that no single device ever materializes a full weight matrix,
activation, or intermediate tensor for that layer. This is the Megatron-LM (Shoeybi et al., 2019)
approach, and it is what makes it possible to fit and compute a single transformer layer that is, by
itself, too large (in weights or in activation memory) for one device.

The key design insight is choosing *which* matrix dimension to split so that the two halves of a
computation can each proceed independently on-device, with communication needed only once — at the
point where the split-apart partial results must be recombined into a value every device needs
identically.

### 3.2 FFN sharding: column-parallel then row-parallel

A transformer FFN block is two linear layers with a nonlinearity in between: `Y = f(X W1) W2`
(ignoring bias for clarity; `f` is GELU/SwiGLU-family). Megatron shards this as:

- **`W1` is split column-wise** across `T` devices: device `t` holds `W1_t` of shape `[d_model,
d_ff/T]`. Every device computes `X W1_t` independently — no communication needed, because `X` (the
input activation) is identical (replicated) on every device at the start of the block, and slicing
the output columns of a matmul requires no interaction between devices computing different output
columns.
- Apply the nonlinearity `f` elementwise: `Y_t = f(X W1_t)`, still purely local, since GELU/SwiGLU
are elementwise (this is precisely *why* the split happens on this dimension — an elementwise
nonlinearity commutes with a column split, but would not commute cleanly with a row split of the
*input* dimension).
- **`W2` is split row-wise**, matched to `Y_t`'s column split: device `t` holds `W2_t` of shape
`[d_ff/T, d_model]`. Each device computes a **partial sum** `Z_t = Y_t W2_t`, which has full
`d_model` width but only accounts for `1/T` of the contraction — it is not yet the final output.
- **Communication point:** the true output is `Z = sum_t Z_t`, so an **all-reduce** across the `T`
TP devices sums the partial `Z_t`'s into the identical final `Z` on every device.

```
# Column-parallel W1, row-parallel W2 — one all-reduce per FFN block
Y_t = gelu(X @ W1_t)            # local, no communication (X already replicated)
Z_t = Y_t @ W2_t                # local, partial sum over d_ff/T
Z   = all_reduce_sum(Z_t)       # ONE communication point, output now replicated on all T devices
```

This pattern — split so the nonlinearity is local, and defer the one unavoidable cross-device
summation to a single all-reduce at the block's exit — is the template Megatron reuses for
attention.

### 3.3 Attention sharding: split along the head dimension

Multi-head attention is naturally amenable to TP because heads are already independent: `d_model`
splits cleanly into `n_heads × d_head`, and each head's computation (its own Q/K/V slice, its own
softmax, its own attention-weighted sum) does not need information from other heads until the final
output projection concatenates them back together.

- **Q, K, V projections are column-parallel**: device `t` computes `Q_t = X W_Q_t`, `K_t = X W_K_t`,
`V_t = X W_V_t` where each `W_*_t` produces exactly `n_heads/T` heads' worth of columns. Attention
(`softmax(Q_t K_t^T / sqrt(d_head)) V_t`) is then computed *entirely locally* per device, using only
that device's own subset of heads — no cross-device communication required for the attention
operation itself.
- **The output projection `W_O` is row-parallel**, matched to the local attention output's column
split, producing a partial sum exactly as with `W2` in the FFN case.
- **One all-reduce** sums the partial output-projection results across the `T` devices.

So each transformer block, under pure tensor parallelism, needs exactly **two all-reduces in the
forward pass** (one after attention's output projection, one after the FFN's down-projection) and,
symmetrically, **two more in the backward pass** (all-reduces are needed on the backward side
because the split dimension that required no forward communication — column-parallel inputs being
locally complete — becomes the dimension that *does* need a gradient-sum on the way back, and vice
versa). That is **4 all-reduces per transformer layer per micro-batch** under naive Megatron-style
TP — a nontrivial, fixed communication tax paid every single layer, not just once per model or once
per step.

### 3.4 Communication cost and why TP must stay intra-node

Each all-reduce here operates on an activation tensor of shape `[batch, seq_len, d_model]`, not a
gradient buffer — so its size is `b × s × d_model × bytes_per_element` (bf16 → 2 bytes), and unlike
DP's gradient all-reduce (paid once per step), this is paid **every layer, every micro-batch, both
forward and backward**. Using the ring all-reduce cost model from Section 2.2, each device moves
roughly `2(T-1)/T × (b·s·d_model·2 bytes)` per all-reduce; multiply by 4 per layer and by the number
of layers, and TP's aggregate communication volume per step is large and, critically,
**latency-sensitive**: because it happens inline in the critical path of every layer's forward and
backward, any added latency directly stalls compute waiting on it (unlike DP's gradient all-reduce,
which can be overlapped with unrelated backward compute per Section 2.3 — there is no "other work"
to overlap TP's inline all-reduce with, since the very next operation in the same layer depends on
its result).

This is precisely why **TP groups are conventionally confined to a single node's NVLink domain**
(typically 8 GPUs, matched to a DGX-class node) rather than spanning nodes over InfiniBand: NVLink
offers roughly an order of magnitude more bandwidth and dramatically lower latency than inter-node
RDMA (quantified in `005_Cluster_Hardware_Networking_And_Interconnect.md`), and TP's per-layer,
latency-critical, non-overlappable communication pattern is the one axis of the three that is least
tolerant of a slow interconnect. A common rule of thumb: **TP degree ≤ GPUs per node**.

### 3.5 Sequence parallelism (a TP extension, briefly)

Megatron's later work adds **sequence parallelism**: the regions of a transformer block that are
*not* covered by the attention/FFN tensor-parallel sharding above — LayerNorm/RMSNorm and dropout,
which operate identically and independently per-token — are instead split along the *sequence*
dimension across the same TP group, further reducing the activation memory each device must hold for
those specific operations (they no longer need the fully-replicated activation, only their sequence
shard). This trades one all-gather/reduce-scatter pair for the memory saved on the
normalization/dropout activations; it composes with the rest of TP rather than replacing it, and is
now standard in most Megatron-derived training stacks.

## 4. Pipeline Parallelism

### 4.1 The problem PP solves, and the bubble it creates

TP shards *within* a layer; pipeline parallelism instead partitions the model *across* layers:
device 0 holds layers 1–k, device 1 holds layers k+1–2k, and so on, forming `P` pipeline **stages**.
A micro-batch flows through the stages like an assembly line: stage 0 computes its forward pass and
sends the resulting activation to stage 1, which computes its forward pass and sends onward, and so
forth, with an analogous backward flow in reverse once the loss is computed at the final stage.

The naive way to run this — one micro-batch fully traverses all `P` stages forward, then fully back,
before the next micro-batch starts — leaves every stage except the one currently active **idle**,
which defeats the purpose of parallelism entirely. The fix (GPipe, Huang et al. 2018) is
**micro-batching**: split the per-step global batch into `m` micro-batches and pipeline them through
the stages, so that while stage 0 is processing micro-batch 2's forward pass, stage 1 is
simultaneously processing micro-batch 1's forward pass. This keeps more stages busy concurrently,
but it cannot eliminate idle time entirely: at the very start of a step, only stage 0 has anything
to do (stages 1..P-1 are waiting for the first micro-batch to arrive); at the very end, only the
last stage still has work (earlier stages have run out of micro-batches to forward and are waiting
for backward passes to arrive from later stages). This structural idle time is the **pipeline
bubble**.

For GPipe's schedule (all `m` micro-batches' forward passes complete before any backward pass begins
— the "all-forward-then-all-backward" schedule), the bubble fraction is:

```
bubble_fraction = (P - 1) / (m + P - 1)
```

where `P` = number of pipeline stages and `m` = number of micro-batches per step. Two consequences
follow directly from this formula, and both are staff-interview-testable:

- **Bubble fraction shrinks as `m` grows** (more micro-batches amortize the fixed `P-1` fill/drain
cost over more useful work) — so increasing micro-batch count is the single most direct lever
against bubble overhead.
- **Bubble fraction grows with `P`** — more pipeline stages means proportionally more fill/drain
overhead for the same `m`, which is exactly why PP degree is chosen conservatively (just enough
stages to fit the model in memory, not more) rather than maximized.

The trade-off against pushing `m` arbitrarily high: GPipe's schedule requires every stage to hold
**activations for all `m` in-flight micro-batches simultaneously** (since none of the backward
passes start until all forwards finish), so activation memory scales with `m` — pushing bubble
fraction toward zero directly inflates activation memory, another instance of the "no free axis"
principle from Section 1.

### 4.2 1F1B: bounding activation memory without eliminating the bubble

PipeDream's **1F1B ("one-forward-one-backward")** schedule, adopted by Megatron-LM as its default
pipeline schedule, interleaves each stage's forward and backward work as early as possible rather
than deferring all backward passes to the end: once a stage has enough in-flight micro-batches to
keep the pipeline full, it alternates one forward step with one backward step for a completed
micro-batch, freeing that micro-batch's activation memory immediately rather than holding it until
every micro-batch's forward has finished.

```
# Stage p's 1F1B schedule, m micro-batches, P stages
warmup_forwards = min(m, P - p - 1)   # steady-state: fill the pipeline before backward starts
for i in range(warmup_forwards):
    forward(microbatch[i])
for i in range(m - warmup_forwards):
    forward(microbatch[warmup_forwards + i])
    backward(microbatch[i])            # 1 forward, 1 backward, steady-state
for i in range(m - warmup_forwards, m):
    backward(microbatch[i])            # drain remaining backward passes
```

1F1B achieves the **same bubble fraction as GPipe** (the fill/drain cost is structurally unavoidable
given `P` stages) but bounds the number of activation sets any stage must hold concurrently to
roughly `P` (one per stage currently "in flight") rather than `m`, decoupling activation memory from
micro-batch count. This is why 1F1B, not GPipe's schedule, is the practical default at frontier
scale: it lets you push `m` up to shrink the bubble fraction without paying for it in activation
memory.

### 4.3 Interleaved (virtual) pipeline stages

A further refinement, also from Megatron-LM: instead of giving each physical device one contiguous
block of layers, give each device **several smaller, non-contiguous chunks** ("virtual stages")
interleaved with other devices' chunks — e.g., with 4 devices and 8 virtual stages, device 0 might
hold layers {1–2, 17–18}, device 1 holds {3–4, 19–20}, etc. This shrinks the *effective* bubble
fraction further, because the fill/drain cost scales with the number of layers per "hop" rather than
the number of physical devices — the bubble formula becomes `(P-1) / (m·v + P - 1)` where `v` is the
number of virtual stages per device, letting `v > 1` shrink the bubble without adding physical
devices. The cost is more frequent (though individually smaller) point-to-point activation transfers
between devices, since a micro-batch now crosses device boundaries `v` times more often.

### 4.4 DualPipe (DeepSeek-V3), briefly, as a case study in co-designed scheduling

`..\OpenSource\007_DeepSeek_V3.md` describes DualPipe, a custom bidirectional pipeline schedule
built specifically to overlap MoE's added all-to-all communication
(`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`) underneath pipeline compute more
aggressively than 1F1B does by default. The general lesson worth internalizing beyond this specific
example: standard schedules (GPipe, 1F1B, interleaved 1F1B) are *generic* — they assume the only
cross-stage traffic is the forward/backward activation handoff. Once a model has additional
communication baked into each layer (MoE's all-to-all, or any other structural requirement), it can
be worth co-designing the pipeline schedule itself around that traffic, rather than treating
pipeline scheduling and other communication as independent problems layered on top of each other.

## 5. Combining All Three: 3D Parallelism

### 5.1 The parallel grid

A concrete training configuration is specified by a triple `(TP, PP, DP)`, with the constraint `TP ×
PP × DP = total GPU count`. Conceptually:

- Partition the full GPU fleet into `PP` pipeline stages.
- Within each pipeline stage, partition further into `DP` data-parallel groups.
- Within each data-parallel group (i.e., within one pipeline stage of one data-parallel replica),
partition into `TP` tensor-parallel shards that jointly hold one copy of that stage's layers.

Rank layout matters, not just the three numbers: because TP's all-reduce is the most
latency-sensitive and bandwidth-hungry (Section 3.4), TP ranks are assigned to be **physically
contiguous within a single node** wherever `TP ≤ GPUs per node`. PP ranks, whose point-to-point
activation handoffs tolerate higher latency than TP's inline all-reduces, are assigned **across
nodes**. DP ranks — whose gradient all-reduce is a once-per-step, overlappable operation (Section
2.3) — are typically spread the widest across the topology, since they are the most tolerant of
higher latency.

### 5.2 Worked example: choosing a 3D configuration

Take a concrete, staff-interview-shaped problem: **a 70B-parameter dense transformer, training on a
512-GPU H100 cluster (64 nodes × 8 GPUs/node, NVLink intra-node, InfiniBand inter-node).** Walk the
choice through in order:

**Step 1 — fix TP from the node boundary.** With 8 GPUs/node and NVLink available, `TP = 8` is the
natural ceiling per Section 3.4's rule of thumb (never span TP across the slower inter-node fabric
if you can avoid it). Going below 8 (e.g., `TP=4`) would leave two TP groups per node sharing the
same node's NVLink fabric — workable, but usually only chosen if the model's per-layer size doesn't
warrant 8-way sharding (not the case at 70B). Fix `TP = 8`.

**Step 2 — decide whether PP is needed at all, from a memory check.** With `TP=8`, each TP shard
holds `70B / 8 ≈ 8.75B` "logical" parameters' worth of weight matrix per GPU (in reality, some
structures like layernorm are not shardable, but this approximation suffices for sizing). Under
ZeRO-shape mixed-precision accounting (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`;
roughly 16 bytes/param across bf16 weights+grads and fp32 master+Adam state before any ZeRO sharding
of that state) that is already `8.75B × 16 bytes ≈ 140GB` — over an H100's 80GB *before counting
activations*. Two options: (a) shard optimizer/gradient state further via ZeRO-1/2 across the DP
axis (doesn't require adding PP stages), or (b) add pipeline stages to reduce the *parameter and
optimizer-state* footprint per device directly, since PP splits which layers (and their associated
state) a device holds at all, rather than sharding the same layers' state across more owners. In
practice, for a 70B model, teams typically prefer to lean on ZeRO stage 1/2 within the DP group plus
a modest PP degree rather than a large PP degree, because pipeline bubbles (Section 4.1) are a real
throughput tax that ZeRO's memory sharding avoids incurring. A reasonable choice: `PP = 4` (giving
comfortable headroom, since 4-way pipelining plus ZeRO-1 optimizer-state sharding brings per-GPU
state well under 80GB with room for activations), rather than pushing PP higher just to save memory
that ZeRO can shard instead.

**Step 3 — fill the remainder with DP.** `TP × PP = 8 × 4 = 32`. With 512 GPUs total, `DP = 512 / 32
= 16`. So the grid is **TP=8, PP=4, DP=16**, giving `8 × 4 × 16 = 512`. Layout: each of the 64 nodes
hosts one complete TP=8 group (using its full NVLink domain); those 64 TP groups are organized into
`PP=4` pipeline stages of 16 TP-groups each; and within a pipeline stage, those 16 TP-groups
constitute the `DP=16` data-parallel replicas of that stage, whose gradients are all-reduced (or,
more precisely, reduce-scattered under ZeRO) across the InfiniBand fabric once per step.

**Step 4 — sanity-check the bubble.** With `PP=4` and, say, a global batch of 16 micro-batches per
DP replica (`m=16`), bubble fraction ≈ `(4-1)/(16+4-1) ≈ 3/19 ≈ 16%` under 1F1B — a real but
tolerable tax; increasing `m` (more, smaller micro-batches) would shrink this further at essentially
no memory cost under 1F1B (Section 4.2). If `m` were forced down to, say, 4 by activation-memory
pressure, the bubble fraction would jump to `3/7 ≈ 43%` — an unacceptable throughput loss — which is
the concrete mechanism by which "not enough memory for enough micro-batches" and "too much pipeline
bubble" are the *same underlying problem*, reachable either by adding activation checkpointing
(`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`) or by revisiting the PP degree chosen in
Step 2.

**Step 5 — where ZeRO fits into this grid.** The DP axis (16-way here) is exactly where ZeRO stage 1
or 2 (sharding optimizer state, and optionally gradients, across those 16 replicas rather than
replicating them) is applied, layered underneath this exact TP/PP structure, not as a competing
alternative to it. In current practice, this composed strategy is usually just called "3D
parallelism with ZeRO-DP" rather than treated as a fourth independent axis.

This same reasoning generalizes: fix TP to the node's NVLink domain size (or a divisor of it) as the
first decision, use a memory calculation to decide the minimum viable PP degree (leaning on ZeRO to
avoid over-provisioning PP purely for memory), fill the rest of the cluster with DP, and always
sanity-check the resulting bubble fraction against the micro-batch count activation memory can
actually afford. Frontier labs' actual configurations (e.g., DeepSeek-V3's DualPipe-scheduled setup
across 2048 H800s, described in `..\OpenSource\007_DeepSeek_V3.md`) follow this same skeleton with
model-specific refinements (MoE adding the expert-parallelism axis covered next, and a custom
schedule replacing 1F1B).

## 6. Summary Table

| Axis | Splits | Communication primitive | Frequency | Latency sensitivity | Typical placement |
|---|---|---|---|---|---|
| Data (DP) | Batch | All-reduce (or reduce-scatter+all-gather under ZeRO) on gradients | Once per step | Low (overlappable with backward) | Spread widest across cluster |
| Tensor (TP) | Weight matrices within a layer | All-reduce on activations | Every layer, forward + backward | Very high (inline, non-overlappable) | Confined to NVLink/node domain |
| Pipeline (PP) | Layers (depth) | Point-to-point activation/gradient handoff between adjacent stages | Every micro-batch, at stage boundaries | Moderate (tolerates higher latency than TP) | Across nodes |
| Expert (EP) | Experts within an MoE layer | All-to-all (dispatch + combine) | Every MoE layer, forward + backward | High, and data-dependent | See `002_...md` |

The through-line for the rest of this module: every subsequent file assumes fluency with this grid.
ZeRO (`003`) refines what the DP axis actually shards; mixed precision (`004`) determines the
byte-widths used in every cost formula above; hardware and networking (`005`) supplies the actual
bandwidth/latency numbers that make TP-must-stay-intra-node a hard constraint rather than a
preference; and MFU (`007`) is, in large part, a direct measurement of how much of the theoretical
throughput this grid's bubbles and communication overhead actually cost you in practice.
