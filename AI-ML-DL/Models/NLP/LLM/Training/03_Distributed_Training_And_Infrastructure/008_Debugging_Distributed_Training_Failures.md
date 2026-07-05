# Debugging Distributed Training Failures: A Troubleshooting Playbook

## 1. Framing: distributed failures are rarely where the error message points

A single-GPU training bug usually announces itself close to its cause: a shape mismatch throws
immediately, a NaN loss appears the step after the offending operation. Distributed training
failures routinely violate this locality. A collective operation is, by construction, a rendezvous
point where every participating rank must show up — so a bug or slowdown on rank 47 frequently
manifests as rank 0 (or all other ranks) appearing to hang, crash, or slow down, with nothing in
rank 0's own logs pointing at rank 47 at all. The first, most important mental adjustment for this
domain: **when something goes wrong in a multi-rank job, the rank reporting the symptom is often not
the rank with the root cause.** Every playbook below is built around first localizing *which* rank
is actually misbehaving before trying to explain *why*.

## 2. NCCL communication hangs

**What causes them, concretely.** A collective (all-reduce, all-to-all, broadcast, barrier) only
completes once every participating rank has called into it. A hang means at least one rank never
made that call, or made an incompatible one. The common concrete causes:

- **Mismatched collective call ordering across ranks.** NCCL collectives are matched by call order
within a communicator, not by any explicit tag — if one rank's code path executes an extra
collective (or skips one) that the others don't — e.g., a conditional branch that differs across
ranks because of a rank-dependent bug, or an exception on one rank that causes it to skip a
collective the surviving ranks still call — every rank still waiting to match calls with the
divergent rank will block forever. This is a **deadlock**, not a crash, and it is silent: nothing
errors, the job simply stops making progress.
- **A rank has died or is stuck, and other ranks are waiting on it.** An OOM kill, a segfault, or a
stuck Python-level exception on one rank leaves every other rank blocked in a collective waiting for
a partner that will never respond. Without a communication timeout configured, this can hang
indefinitely rather than surfacing as an error.
- **Network partition or switch/cable fault.** A rank whose NIC has failed, or whose path to some
subset of the topology is unreachable (`005_Cluster_Hardware_Networking_And_Interconnect.md`'s
topology discussion — a failed spine link can partition connectivity between specific node groups
even while every individual node appears healthy), causes any collective spanning that broken path
to hang.
- **Topology/environment misconfiguration causing a silent fallback to a much slower transport.** If
InfiniBand isn't correctly detected (wrong `NCCL_SOCKET_IFNAME`, a driver mismatch, or IB verbs
unavailable in a container), NCCL can silently fall back to TCP sockets over Ethernet — often not an
outright hang, but dramatically slower, which can look enough like a hang (a step that should take
seconds instead takes many minutes) to be diagnosed the same way.

**Isolation approach, in order:**

1. **Turn on `NCCL_DEBUG=INFO` (or `TRACE` for more detail)** before anything else. This surfaces
which transport NCCL actually selected (IB vs. TCP fallback — immediately answers the last bullet
above), and, on a hang, often shows which collective call each rank last entered.
2. **Get a stack trace from every rank, not just the one that looks stuck.** Attach `py-spy dump`
(for Python-level stacks) or `gdb`-based inspection to every rank's process while the job is hung.
The diagnostic signature of a classic collective-order mismatch: most ranks' stacks show them
blocked inside the *same* collective call (e.g., all sitting inside an all-reduce at the same
logical point in the training loop), while one or a few ranks' stacks show them somewhere *else
entirely* — already past that point, stuck in an earlier operation, or not there at all because they
crashed. That divergence is the smoking gun, and it directly identifies which rank(s) to investigate
further, rather than treating the whole job as an undifferentiated hang.
3. **Enable NCCL's async error handling and a communication timeout**
(`NCCL_ASYNC_ERROR_HANDLING=1`, or PyTorch's equivalent `TORCH_NCCL_ASYNC_ERROR_HANDLING`, plus a
configured `timeout` on the process group) *before* the fact, as standing infrastructure rather than
a reactive debugging step — this converts a silent, indefinite hang into a bounded wait followed by
an explicit exception naming the stuck collective and the offending rank, which is far cheaper to
diagnose than an open-ended hang discovered only because a job has been stuck for an hour with no
output.
4. **Bisect by disabling nodes** if the above doesn't immediately localize the fault — restart the
job on progressively smaller subsets of the cluster (e.g., half the nodes) to determine whether the
hang is reproducible on a smaller topology (implicating a specific bad node or link, findable by
further bisection) or only manifests at full scale (implicating a scale-dependent resource limit,
e.g., exhausting some fixed-size hardware resource like available NIC queue pairs or GPU-side
communication buffers, more common than it sounds at very large world sizes).

## 3. Straggler nodes: silently slowing the whole job

**The mechanism.** Every synchronous collective in
`001_Parallelism_Strategies_Data_Tensor_Pipeline.md` and
`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md` — DP's gradient sync, TP's per-layer
all-reduce, PP's stage handoff, EP's all-to-all — is a barrier: the operation completes only when
its *slowest* participant arrives. A single straggler device therefore sets the pace for the
*entire* group it participates in, not just for itself, and — critically — **it does not need to
fail or error to do this damage; merely running measurably slower than its peers is sufficient.**
This is what makes stragglers more insidious than outright crashes: a crash is loud and gets
investigated immediately; a straggler produces a training job that is simply, mysteriously slower
than expected, with every individual rank's code path executing "correctly."

**Common root causes of a straggler:** thermal throttling (a GPU running hotter than its peers,
e.g., due to a failing fan or poor airflow in a specific rack position, clocking itself down to stay
within thermal limits); a degrading GPU throwing correctable ECC errors at an elevated rate, which
some hardware/driver stacks handle by reducing clocks preemptively as a protective measure; a "noisy
neighbor" on the same physical network path (another job on shared infrastructure, or even another
process on the same node, contending for NIC or PCIe bandwidth); or a slow data-loading path
specific to one rank (a flaky or overloaded storage shard, a slow decompression path for that rank's
specific data shard) that has nothing to do with the GPU at all but still shows up as that rank
arriving late to the first collective of the step.

**Detection.** The direct fix for "which rank is slow" is the same instrumentation Step 5 of
`007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s diagnostic framework recommends
generally: **log per-rank step time (or, more granularly, per-rank compute time versus per-rank
collective-wait time) continuously, and look at the distribution across ranks, not just the
aggregate mean.** A healthy job shows a tight distribution of per-rank times; a straggler shows one
or a small number of ranks consistently at the tail, session after session, rather than a rank
that's randomly slow one step and fast the next (random per-step variance is normal system noise;
*consistent* per-rank tail behavior across many steps is a straggler). Canary/synthetic benchmarks —
periodically running a fixed, known-cost micro-benchmark on every node independent of the actual
training workload — are a complementary detection method that isolates hardware-level slowness from
workload-specific effects (e.g., a rank that's slow only because of *its* specific data shard would
pass a compute-only canary cleanly, correctly pointing the investigation toward the data pipeline
instead of the GPU).

**Mitigation.** Once identified, the fix is usually operational rather than algorithmic: drain and
replace the offending node (the elastic-training machinery from
`006_Checkpointing_Fault_Tolerance_And_Elastic_Training.md`, Section 4, applies directly — a
straggler is a "soft" failure that the same automatic-replacement infrastructure built for hard
failures should ideally also catch), or, if the cause is a shared/contended resource rather than a
specific bad node, address the resource contention directly (isolate the job's network traffic from
other tenants, fix the data-shard imbalance). Some frameworks support redundant/backup computation
for exactly this scenario (mirroring PP/DP work across a spare so a straggler's slow shard can be
preempted by a faster redundant copy), though this is a heavier-weight mitigation generally reserved
for cases where node replacement isn't fast enough to be worth the added complexity.

## 4. Silent data corruption: wrong gradients without a crash

**Why this is the hardest failure mode in this file.** Every failure mode discussed so far either
crashes outright (making itself known immediately, if not always diagnosed quickly) or measurably
slows the job (making itself known via a throughput metric). Silent data corruption (SDC) does
neither: a hardware defect — a bit flip in GPU memory that goes undetected or is only partially
corrected, a marginal GPU unit producing a subtly wrong result on specific matmul operations, a
faulty interconnect link corrupting a small fraction of transferred bytes without triggering a
checksum failure — can produce **gradients (or activations) that are simply numerically wrong, on
one specific device, without any error, crash, or NaN.** The training job continues running, the
loss curve may look approximately normal (especially if the corruption affects a small fraction of
parameters or occurs infrequently), and the actual damage — training progress subtly corrupted,
potentially compounding over many steps before it's ever suspected — can go undetected far longer
than any other failure mode in this document, precisely because "loss went down roughly as expected,
no crash" is exactly what a healthy job also looks like.

**This is not a hypothetical risk.** Silent data corruption at the level of individual server
components has been reported as a real, non-negligible phenomenon by multiple hyperscale
infrastructure operators (both Google and Meta have published on SDC in their general server fleets,
independent of ML-specific workloads) — the phenomenon is a known consequence of running enormous
numbers of components continuously at scale, where even a very low per-unit defect rate produces a
nonzero, operationally-relevant absolute count of affected units, exactly the "individually rare,
collectively near-certain" arithmetic from `006`'s failure-rate discussion, but applied here to
*subtle* miscomputation rather than to outright failure.

**Detection strategies**, all of which trade some overhead for the ability to catch what a
crash-only monitoring posture would miss entirely:

- **Cross-replica loss/gradient-norm comparison.** Under data parallelism, every replica processes
different data but should show *statistically* similar loss and gradient-norm distributions over a
large enough window (not identical per-step, but not systematically divergent either). A specific
replica whose gradient norm is a persistent, systematic outlier relative to its peers — not just
noisier, but shifted — is a candidate for host- or device-level corruption on that specific rank,
and is a detectable signal *before* any crash, given the per-rank logging infrastructure already
recommended in Section 3.
- **Redundant/checksum computation for especially sensitive operations.** Recomputing a cheap,
independent check on a small sample of outputs (or periodically running a known-input/known-output
validation kernel on each device) can catch a systematically miscomputing unit that would otherwise
blend into normal-looking training noise.
- **Numerical cross-validation against a reference.** Periodically (not every step — this is
expensive) re-running a fixed batch through a known-good, independently-verified path (e.g., a
different device, or a CPU reference implementation for a small sub-computation) and comparing
results catches deterministic miscomputation that a purely statistical (loss/gradient-norm) check
might not flag if the corruption's effect on aggregate loss happens to be small.
- **Hardware-level ECC/RAS telemetry.** Correlate observed anomalies with the hardware's own
error-reporting (correctable/uncorrectable ECC counters, Xid errors in NVIDIA's driver logs) — a
device throwing an elevated but sub-crash-threshold rate of correctable errors is a legitimate
leading indicator worth cross-referencing against any statistically anomalous rank identified via
the methods above, and worth flagging for proactive replacement even before it produces an
unambiguous training-quality symptom.

**The practical posture**, worth stating directly in an interview: SDC cannot be fully prevented at
the software layer — it is fundamentally a hardware reliability problem — but its *training-quality*
impact can be bounded by treating "a specific rank is a persistent statistical outlier across many
steps, on metrics that should be exchangeable across replicas" as a first-class alert condition, on
par with a crash, rather than dismissing distributional weirdness in per-rank metrics as noise. The
frequency and depth of active cross-validation checks is itself a cost/risk trade (checking
constantly is expensive; never checking risks large amounts of wasted, subtly-corrupted training
compute) that should scale with the size and cost of the run — worth checking far more aggressively
on a frontier-scale, multi-week run than on a small experiment, exactly mirroring `006`'s
checkpoint-frequency trade-off logic (more frequent, more expensive, but the downside of skipping it
scales with cluster size and run duration).

## 5. Configuration mismatches across nodes: the embarrassing, very real failure class

**Why this class deserves explicit attention.** Every failure mode above involves genuine hardware
or algorithmic subtlety. Configuration mismatches are, by contrast, usually mundane — and precisely
because they're mundane, they are extremely common in practice and can consume disproportionate
debugging time before anyone considers them, since engineers instinctively reach for a more
"interesting" explanation first.

**Common concrete instances:**

- **Different CUDA/NCCL/cuDNN driver or library versions across nodes.** A cluster provisioned
incrementally, or where a subset of nodes received a rolling update and others didn't, can have a
training job launch successfully on every node individually (no immediate crash) but produce subtly
different numerical results on different nodes — e.g., because a different cuDNN version selected a
different convolution/attention algorithm with slightly different floating-point rounding behavior —
or, worse, crash only on the specific nodes with the mismatched version once a code path that
depends on the version difference is actually exercised (which might not happen until deep into a
long run, if the triggering code path is data-dependent).
- **Stale or inconsistent container images.** If node provisioning doesn't guarantee every node
pulled the *exact* same container image digest (as opposed to a mutable tag like `:latest`, which
can point to different actual image contents at different times depending on when each node happened
to pull it), some nodes can be running genuinely different code or library versions than others,
with no explicit error indicating this at launch — the job starts, and behaves inconsistently in
ways that look like a distributed-systems bug but are actually a provisioning bug.
- **Environment variable drift.** A `NCCL_*` or framework-specific environment variable set
differently (or unset) on a subset of nodes — often because a launch script sourced a node-local
environment file rather than a centrally-controlled one — can silently alter behavior (e.g., a
subset of nodes falling back to a different, slower transport per Section 2) without producing any
explicit error naming the mismatch as the cause.
- **Filesystem/data mismatches.** A subset of nodes reading from a stale local cache of the training
data or tokenizer, while others read the current version, produces a genuinely hard-to-diagnose
failure because it looks exactly like SDC (Section 4) from the training loop's perspective — a
specific rank producing statistically different results — but the root cause and fix are completely
different (a stale cache versus a hardware defect), making correct root-cause attribution here
directly consequential for what fix to apply.

**Prevention and detection, in priority order:**

1. **Immutable, digest-pinned images and configuration**, distributed identically to every node as a
matter of standing infrastructure policy — the single highest-leverage prevention, because it
removes the possibility of drift rather than relying on catching it after the fact.
2. **A configuration/environment hash check at job launch**, run automatically on every node before
the training job's main loop starts: hash the relevant library versions, environment variables, and
data/tokenizer file checksums on each node, and have every node report its hash to a central point
(or all-gather the hashes among themselves) so a mismatch is caught and the job aborted *before* any
training compute is wasted on a misconfigured launch, rather than discovered hours into a run via a
confusing symptom.
3. **Health-check scripts run as a pre-flight step**, distinct from the training job itself, that
explicitly probe for the specific mismatches enumerated above (driver version, NCCL version,
environment variables, checksum of key data files) — treating cluster configuration verification as
its own first-class, automated, pre-job gate rather than something inferred indirectly from
training-job behavior after the fact.

## 6. A general troubleshooting sequence for "the job stalled/is behaving strangely"

Bringing Sections 2–5 together into a single ordered response to a live incident (the shape of "your
512-GPU job just stalled, walk me through your diagnosis" as an interview prompt):

1. **Is any rank actually dead or erroring?** Check process status and recent logs across all ranks
first — this is the cheapest check and rules in/out a straightforward crash before investigating
anything more subtle.
2. **If no rank crashed, get stack traces from every rank and look for divergence** (Section 2) —
this single check discriminates between "genuine hang/deadlock, findable rank" and "everything is
just slow" (Section 3), which are different investigations from this point forward.
3. **If ranks are progressing but slowly, pull per-rank step-time and collective-wait-time
distributions** (Section 3) to check for a straggler before assuming a global, cluster-wide cause.
4. **If nothing above localizes a cause, check for a configuration mismatch** (Section 5) — cheap to
check via a hash/health-check comparison across nodes, and disproportionately likely to be the
actual cause relative to how often it's considered early, especially after any recent cluster change
(a rolling driver update, a new batch of provisioned nodes, an image rebuild).
5. **If the job is running without error or unusual slowness but training quality looks subtly
wrong** (a loss curve that's technically fine but not matching expectations, or downstream eval
quietly underperforming, echoing `004_Mixed_Precision_Training_And_Numerical_Stability.md` Section
4's hardest-to-catch failure mode), escalate to the SDC-style cross-replica statistical checks in
Section 4 — the failure mode that produces no operational symptom at all beyond training quality
itself, and therefore the one most likely to be missed by any monitoring built only around crashes
and throughput.

The unifying discipline across all five steps: **localize which rank (or which specific
link/component) is actually the anomaly before theorizing about why**, because in a system with
hundreds or thousands of participants, the space of "why" is enormous and mostly wasted effort until
"which" has been narrowed down using the cheap, mechanical checks above.
