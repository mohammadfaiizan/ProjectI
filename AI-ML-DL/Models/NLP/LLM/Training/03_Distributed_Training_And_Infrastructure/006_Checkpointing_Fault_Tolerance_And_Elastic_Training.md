# Checkpointing, Fault Tolerance, and Elastic Training

## 1. The premise: at frontier scale, failure is not an edge case

A training run spanning thousands of GPUs for weeks is, statistically, not a single long-running
computation that might occasionally fail — it is a continuous process in which **some component
failing on any given day is close to a certainty**, and the entire infrastructure design
(checkpointing cadence, elastic recovery, asynchronous I/O) has to be built around that certainty
rather than around failure as an exceptional event to merely handle gracefully if it happens.

**The arithmetic that makes this concrete.** Suppose a single GPU (or, more realistically, a single
node, since a node failure typically takes 8 GPUs down with it) has some small daily probability `p`
of failing — due to any of a long list of real causes: ECC-uncorrectable memory errors, GPU falling
off the PCIe bus, a failed power supply, a NIC or cable fault, a cooling failure causing thermal
shutdown, a software/driver crash, or a host-level kernel panic. Even a `p` as small as 0.1% per
node per day sounds negligible for any *individual* node. But the probability that **at least one**
of `M` independent nodes fails on a given day is `1 - (1-p)^M`, which grows quickly with `M`. At `M
= 1000` nodes (a modest fraction of a frontier cluster) and `p = 0.1%`, the probability of at least
one failure that day is `1 - 0.999^1000 ≈ 63%`. At `M = 4000` nodes, it's `1 - 0.999^4000 ≈ 98%`.
This is the standard "birthday-problem-shaped" reasoning: **individually rare events become
near-certain in aggregate once you have enough independent trials**, and a frontier cluster has
thousands of independent trials running continuously. Publicly reported experience from labs
training at this scale is consistent with this arithmetic in spirit — training logbooks and
technical reports from large-scale runs (e.g., Meta's OPT-175B logbook, and Llama-family technical
reports) describe encountering hardware-triggered interruptions on the order of one or more per day
at multi-thousand-GPU scale, sometimes considerably more frequently during specific problematic
stretches; treat any exact frequency figure as a reported-elsewhere data point to verify against the
primary source rather than a number to reproduce confidently from memory, but treat the qualitative
claim — frequent, routine interruptions are the normal operating condition, not an anomaly — as
solid and well corroborated across multiple independent large-scale training efforts.

**The design implication.** If failure is a near-daily certainty rather than a rare tail event,
checkpointing and recovery cannot be an afterthought bolted onto a "happy path" training loop — they
have to be treated as a first-class, continuously-exercised part of the system, engineered for the
failure-recovery path to be as fast and automatic as possible, because it *will* be exercised,
repeatedly, over the life of any sufficiently large run.

## 2. What a checkpoint actually contains, and how big it is

A correct, resumable checkpoint must capture everything needed to reconstruct training state
*exactly* (or acceptably close to exactly) as it was at the checkpointed step, not merely the model
weights:

- **Model parameters** — the bf16 (or fp8) working copy, and, separately, the fp32 master-weight
copy if the recipe uses one (`004_Mixed_Precision_Training_And_Numerical_Stability.md`).
- **Optimizer state** — Adam's `m` and `v` buffers
(`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`), which are just as essential to resume
correctly as the weights themselves: restarting Adam's moment estimates from zero after a resume is
a real, avoidable regression in optimization dynamics, not merely an inconvenience.
- **Learning-rate scheduler state** and any other stateful training-loop component (loss-scaler
state if using dynamic loss scaling, `004` Section 2).
- **RNG state** — for full reproducibility of data ordering, dropout masks, etc., though in practice
many production systems accept a minor loss of exact reproducibility here in exchange for simpler
infrastructure.
- **Dataloader position** — which shard/offset in the training corpus the run had reached, so
resuming doesn't silently re-serve already-seen data or skip data (an easy, consequential bug:
silently repeating a large chunk of data or dropping a large chunk both distort the effective
training distribution).
- **The exact parallel configuration** (TP/PP/DP/EP degrees) the checkpoint was written under —
needed by the loading logic to know how to *reshape* sharded state if the resuming job's parallel
grid differs from the one that wrote the checkpoint (Section 4).

**Size.** Using the `16Ψ`-bytes-per-parameter figure from `003` for model states (params + gradients
+ optimizer state under mixed-precision Adam; gradients are not strictly necessary to checkpoint
since they're recomputed from data, but many implementations include some gradient-adjacent state
for exact resumption), a full checkpoint of a 175B model is on the order of **2.8TB**; a
671B-parameter model (DeepSeek-V3 scale) is on the order of **~10.7TB** by the same accounting. This
is not a small file — writing and reading checkpoints of this size to persistent storage at any
meaningful frequency is itself a nontrivial systems problem, which is exactly why Sections 3 and 5
exist.

## 3. The checkpoint-frequency trade-off

**More frequent checkpoints:** less expected work lost on failure (a shorter interval since the last
checkpoint means less recomputation needed after a restart), at the cost of (a) the I/O overhead of
the checkpoint write itself repeatedly interrupting or slowing training, and (b) the storage cost of
retaining (some number of) multi-terabyte checkpoint snapshots.

**Less frequent checkpoints:** lower overhead and storage cost, at the cost of more expected lost
work per failure.

**Quantifying the trade-off.** If checkpoints are written every `T` steps, and a failure can occur
uniformly at random within that interval, the *expected* number of steps of work lost per failure is
`T/2` (on average, a failure occurs midway through the interval); the *worst case* is a full `T`
steps lost (failure occurs just before the next checkpoint would have been written). Set against
this: if a single checkpoint write costs `C` seconds of either wall-clock stall (for a fully
synchronous/blocking checkpoint) or background I/O/network contention (for an asynchronous one,
Section 5), and failures occur at rate `λ` (failures per unit wall-clock time, itself a function of
cluster size per Section 1's arithmetic), the total overhead is roughly a balance between
`(checkpoints written) × C` and `(failures) × (expected recompute cost of T/2 lost steps)`.
Shrinking `T` trades more of the first cost for less of the second; the optimal `T` is smaller for
larger, more failure-prone clusters (higher `λ`) and larger for smaller, more reliable ones — which
is exactly why frontier-scale runs, with their much higher aggregate `λ` per Section 1, tend toward
checkpointing quite frequently (commonly on the order of every 100s to low-1000s of steps, though
exact cadences are generally not disclosed with precision by labs and should be treated as reasoning
rather than a memorized constant) despite the individually larger absolute checkpoint size at that
model scale — the two effects (bigger checkpoints, but also a stronger incentive to checkpoint
often) partially offset, and asynchronous checkpointing (Section 5) is precisely the technique that
decouples "checkpoint often" from "pay a large stall cost every time," resolving much of the tension
the naive trade-off above implies.

## 4. Elastic and resumable training: surviving node loss without a full restart

**The naive failure-recovery path** — a job crashes, the entire job (all `N` GPUs) is torn down, a
human or an orchestration script identifies and replaces the failed node(s), and the job is
relaunched from the last checkpoint with the *exact same* `N` and parallel configuration — works,
but wastes the time of every *healthy* GPU in the cluster for the duration of the
diagnose-and-replace cycle, which can be minutes to hours depending on how automated the fleet's
health-checking and spare-node provisioning is.

**Elastic training** targets this waste directly: the goal is for a job to **continue running,
automatically, with a different GPU count** after losing (and, ideally, having automatically
replaced) some nodes — without a full job teardown and cold restart, and ideally without even a full
pause of the *healthy* remaining GPUs while recovery happens. This requires several pieces working
together:

- **Automatic failure detection.** A watchdog (heartbeats between ranks, or a NCCL communication
timeout, `008_Debugging_Distributed_Training_Failures.md`) needs to distinguish "a specific rank has
died or become unreachable" from "the job is merely slow," and do so within a bounded time window —
too aggressive a timeout risks false-positive restarts on a transient network blip; too lax a
timeout means healthy GPUs sit idle for longer than necessary waiting on a truly-dead rank.
- **A spare-node pool** the orchestrator can draw from to replace a failed node automatically,
without waiting on a human in the loop — standard in mature large-scale training infrastructure
(e.g., Kubernetes-based or Slurm-based schedulers configured with health-checked spare capacity, and
frameworks like PyTorch's `torchelastic`/`torchrun` supporting dynamic rank membership changes).
- **Re-sharding on load ("universal checkpoint").** If the replacement changes the effective
parallel grid even temporarily (e.g., a PP stage or a DP replica is short one member while a spare
is being provisioned, or the job deliberately resumes at a different total GPU count than it
checkpointed at — a common scenario when spare capacity of the exact original topology isn't
immediately available), the checkpoint loading logic must be able to **reshape** sharded state
(ZeRO-sharded optimizer state, TP-sharded weight matrices) from the checkpoint's original parallel
configuration into whatever new configuration the resumed job is running under. This is a nontrivial
piece of engineering in its own right — DeepSpeed's "universal checkpoint" format and similar
mechanisms in other frameworks exist specifically to decouple the checkpoint's on-disk
representation from any single fixed parallel configuration, typically by checkpointing in a more
"logical" (unsharded, or consistently-shardable) representation and re-sharding at load time
according to whatever grid the loading job specifies, rather than checkpointing the literal
in-memory shards of the writing job's specific configuration.
- **Batch-size and learning-rate implications of a changed GPU count.** If the effective DP degree
changes because of a temporary or permanent change in node count, the global batch size (and,
depending on the scaling rule in use, the learning rate) may need to adjust correspondingly to
preserve training dynamics — an elastic system that silently changes global batch size across a
node-count change without accounting for this is introducing a training-dynamics confound on top of
the infrastructure change, which is exactly the kind of subtle bug that produces a mysterious,
hard-to-attribute change in the loss curve around the time of a node-count-changing recovery event.

## 5. Asynchronous, non-blocking checkpointing

The straightforward way to write a checkpoint — pause training, have every rank write its shard of
state directly to persistent (typically networked/distributed) storage, resume training once every
rank confirms its write completed — is **synchronous** and blocks all forward progress for the
duration of the write, which, for a multi-terabyte checkpoint, can be a genuinely long stall
(bounded by the persistent storage system's write bandwidth, which is very often far below
GPU-to-GPU or even GPU-to-host bandwidth). At the checkpoint frequencies implied by Section 3's
failure-rate arithmetic, this stall recurring regularly is a real, measurable tax on total training
throughput.

**The fix: decouple "make the state safe from GPU-side loss" from "get the state to its final
durable location."** The now-standard pattern (used by DeepSpeed, PyTorch Distributed Checkpointing,
and most production training stacks at scale) is:

1. **Fast, local snapshot.** Each rank copies its shard of state from GPU HBM to **host CPU memory**
(or, in some designs, to fast local NVMe) — a transfer that, per
`005_Cluster_Hardware_Networking_And_Interconnect.md`'s bandwidth hierarchy, is far faster than a
write to networked persistent storage, and can typically be done with a comparatively short stall
(or even fully asynchronously via a non-blocking copy, if the framework supports issuing the next
training step's GPU work before the copy is confirmed complete, provided care is taken that the
GPU-side buffer being copied from isn't mutated by the next step before the copy actually reads it —
a real synchronization hazard implementations must handle correctly).
2. **Resume training immediately.** Once state is safely resident in host memory (not yet in durable
storage, but no longer solely in GPU HBM, and therefore already surviving a GPU-only failure, e.g.,
an ECC error or SXM fault that doesn't take down the host CPU/RAM), the GPUs are free to proceed
with the *next* training step. The host memory copy is not on the GPU-compute critical path any
further.
3. **Background flush to durable storage.** A separate background thread or process on each node (or
a dedicated set of I/O-focused processes) asynchronously streams the host-memory snapshot out to the
actual persistent, distributed storage system (a networked filesystem or object store), overlapping
this slower I/O with the *next* several steps' worth of GPU compute rather than blocking on it.

```
# Conceptual asynchronous checkpoint flow, per rank
def checkpoint_step(model_shard, optimizer_shard, step, async_writer):
    # 1. Fast GPU -> host copy (short stall, or fully async with correct synchronization)
    cpu_snapshot = {
        "model": {k: v.detach().to("cpu", non_blocking=True) for k, v in model_shard.items()},
        "optimizer": {k: v.detach().to("cpu", non_blocking=True) for k, v in optimizer_shard.items()},
        "step": step,
    }
    torch.cuda.synchronize()   # ensure the copy has actually landed before training mutates GPU buffers further

    # 2. Hand off to a background writer; training resumes immediately, NOT waiting on durable-storage I/O
    async_writer.submit(cpu_snapshot, dest_path=f"ckpt_step_{step}")
    # training loop continues here without blocking on the actual disk/network write
```

The net effect: the *exposed* stall (time the GPUs are actually idle, not doing useful training
work) shrinks from "however long it takes to durably persist several terabytes across the network"
down to "however long it takes to copy a rank's shard from HBM to host RAM" — often more than an
order of magnitude less, per the bandwidth hierarchy in `005`. The trade this introduces: a
checkpoint that exists only in host RAM (steps 1–2 complete, step 3 still in flight) is **not yet
safe against a whole-node failure** (a node that dies loses its host RAM along with its GPUs) — so
the fault-tolerance guarantee an asynchronous scheme actually provides is "safe against a GPU-only
failure as soon as the host-memory copy completes; safe against a whole-node failure only once the
background durable-storage flush also completes," and a rigorous design needs to be explicit about
which of these two failure classes it's protecting against at each stage, rather than treating
"checkpointed" as a single binary state once the fast path completes.

**A further refinement some systems use: in-memory replica checkpoints.** Rather than (or in
addition to) flushing to a separate durable storage tier, a rank's host-memory snapshot can be
replicated to *another* node's host memory (e.g., its DP-group peer, which already holds
numerically-equivalent — under ZeRO-1/2's still-replicated-parameters property from `003` — or at
least reconstructable state) purely in RAM, giving fast in-memory recovery from many single-node
failures without touching disk or network storage at all for the common case, falling back to the
slower durable-storage path only for less common, larger-scale failure scenarios (e.g., losing an
entire rack). This trades additional host RAM usage for a further reduction in both recovery latency
and the load placed on shared durable storage infrastructure at checkpoint time.

## 6. Verifying a checkpoint is actually correct before trusting it

Everything in Sections 2–5 addresses *writing and restoring* checkpoints efficiently; a separate,
easy-to-skip concern is *verifying* that a written checkpoint is actually correct — because a
checkpoint corrupted in a way that doesn't crash the write process is exactly as dangerous as the
silent-data-corruption failure mode developed at length in
`008_Debugging_Distributed_Training_Failures.md`, Section 4, and arguably worse: a corrupted
checkpoint that gets used as the recovery point after a later, unrelated failure can silently
resurrect bad state into an otherwise-healthy job, at a point in time that may be far removed from
whatever originally caused the corruption, making root-cause attribution significantly harder than
for an in-flight SDC event. Concrete, low-overhead verification steps worth building as standing
infrastructure: checksumming each rank's written shard immediately after the write (comparing against
a checksum computed before the GPU-to-host copy in Section 5's fast path, so the check covers the
entire write path rather than only the final storage layer); a lightweight post-write sanity load —
reading back a small, fixed subset of the just-written checkpoint and confirming it round-trips
correctly — run asynchronously so it doesn't reintroduce the stall asynchronous checkpointing was
built to avoid; and, periodically rather than on every checkpoint, a full resume-and-compare test on a
held-out portion of the cluster, confirming that a resumed run reproduces the expected loss for a few
steps before trusting that checkpoint as a viable rollback target for the main job. None of this
needs to be expensive relative to the checkpoint-write cost itself, and the alternative — discovering
during an actual incident that the checkpoint you just restored from was itself already corrupted —
is a strictly worse position to be in than paying a small, continuous verification tax throughout the
run.

## 7. Putting it together: a resilience posture for a frontier-scale run

The pieces from Sections 2–5 compose into an overall resilience posture, and articulating this
composition clearly is a good way to answer an open-ended "how would you design fault tolerance for
a 10,000-GPU training run" interview question: (1) checkpoint frequently enough that expected lost
work per failure (Section 3's `T/2` term) stays small relative to `T`'s implied overhead, calibrated
against the cluster's actual observed failure rate rather than a generic default; (2) make each
checkpoint asynchronous (Section 5) so the frequent-checkpointing choice from (1) doesn't itself
become the dominant source of lost throughput; (3) build automatic failure detection and a
spare-node pool (Section 4) so that when — not if — a node fails, the *healthy* remainder of the
cluster loses as little idle time as possible waiting for a human-mediated recovery; and (4) make
the checkpoint format parallel-configuration-agnostic (Section 4's re-sharding requirement) so that
recovery doesn't require reproducing the exact failed configuration, which is often the slowest and
least automatable part of a naive recovery path. None of these four pieces alone is sufficient — a
system with fast async checkpointing but no automatic node replacement still stalls on the
human-mediated recovery loop; a system with instant automatic node replacement but only synchronous,
infrequent checkpointing still loses large amounts of recompute per failure and pays a large stall
on every checkpoint — the resilience of the overall system is a property of all four working
together, matched to the specific failure rate implied by the cluster's actual size per Section 1's
arithmetic.
