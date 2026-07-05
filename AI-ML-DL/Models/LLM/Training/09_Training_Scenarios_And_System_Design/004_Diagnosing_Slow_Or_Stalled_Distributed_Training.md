# Diagnosing Slow Or Stalled Distributed Training

## The Scenario

"Your training job's per-step throughput has degraded by roughly 30% over the last several hours. No code change went in, no config was touched. Walk me through your diagnosis."

Unlike the loss-spike scenario (`003_Debugging_A_Loss_Spike_Mid_Training.md`), which is primarily about model/data correctness, this scenario is purely a systems-performance investigation — the loss curve is presumably still fine, tokens are still being processed correctly, they're just being processed *slower*. The skill being tested is whether you have a mental model of where throughput actually goes in a distributed training job, so that you can localize the regression efficiently rather than randomly poking at the system. I'll structure this the way you'd actually run the investigation: cheap global signals first, then branch into the four main hypothesis classes, each with its own concrete monitoring signature.

## Step -1: A Quick FAQ Before Diving In

- **Is this scenario meaningfully different from a total job hang (zero throughput)?** Yes — a genuine hang (near-zero utilization, no progress at all) is much more likely a deadlock in collective-communication logic than any of the four hypotheses below, which all assume the job is still making progress, just more slowly; see `012_Interview_Questions_Part2.md`, Q7 for the hang-specific diagnostic tree.
- **Does "30% degradation" by itself suggest which hypothesis is most likely?** Not on its own — the magnitude alone is under-informative; the shape (step-function vs. gradual) and the step-time breakdown are what actually discriminate between hypotheses, not the percentage.
- **Should the on-call engineer assume this is transient and wait to see if it self-resolves?** Only briefly, and only if the historical baseline shows this level of transient variance is normal — a regression sustained over "hours," per the scenario's own framing, has already passed the point where waiting further is likely to resolve it on its own.

## Step 0: Why This Question Is a Favorite Systems-Depth Probe

Before working the diagnostic tree, it's worth naming why interviewers reach for this scenario specifically. It has no single, memorizable answer the way an algorithms question might — the correct response depends entirely on which telemetry you'd actually pull first and how you'd interpret it, which makes it very hard to answer well from memorized facts alone and very easy to distinguish a candidate who has genuinely operated distributed training infrastructure from one who has only read about it. It also rewards exactly the kind of efficient, hypothesis-narrowing investigation discipline (per Step 6) that a staff engineer is expected to bring to any live production incident, not just this specific training-throughput scenario — the structure of the answer generalizes far beyond training infrastructure specifically.

## Step 0b: A Quick FAQ Before Starting

- **Is a 30% throughput regression always worth a full investigation, or could it be normal variance?** Compare against the historical baseline's own variance first — if normal day-to-day fluctuation is routinely 5-10%, a 30% drop sustained over hours is well outside that band and warrants full investigation; a single noisy data point might not.
- **Should the on-call engineer try a quick restart before investigating?** Generally no for this scenario specifically — a restart can mask the signal needed to diagnose which hypothesis is correct (e.g., it would reset memory-allocator state, destroying the evidence for Hypothesis 5), whereas for the loss-spike scenario in `003_...` a checkpoint-based restart is the correct immediate action; the two scenarios' correct first moves are genuinely different.
- **What's the very first question to answer before branching into any hypothesis?** Whether the degradation is a step-function or a gradual drift — this single characterization reorders the entire priority of which hypothesis to check first.

## Step 1: Establish the Baseline and the Blast Radius

Before hypothesizing about cause, characterize the regression precisely:

- **Get the actual throughput time series**, not just "it feels slower." Plot tokens/second or steps/second over the relevant window and identify whether the degradation was a step-function drop at a specific point in time or a gradual decline. A step-function drop at a specific timestamp is far more useful diagnostically than a gradual decline — it lets you correlate against any discrete event (a node replaced by the cluster scheduler, a network reconfiguration, a background job starting on shared infrastructure) at that exact time.
- **Is the degradation uniform across all ranks, or concentrated?** This is the single highest-value early check, exactly as in the loss-spike scenario, and for the same reason: per-rank step-time telemetry (every well-instrumented distributed training job should log per-rank step time, not just an aggregate) immediately tells you whether you're looking for one bad component (a straggler) or a systemic one (a global resource contention or configuration issue affecting all ranks roughly equally).
- **What's the actual compute/communication split doing?** If your training framework exposes a breakdown of step time into compute (forward/backward matmul time) versus communication (all-reduce, all-to-all, pipeline-parallel send/recv) versus data-loading wait time, pull that breakdown for before-and-after the regression. This one number — which *category* of step time grew — collapses the entire diagnostic tree by roughly a factor of three immediately, because it tells you whether to look at compute (Hypothesis: memory fragmentation, Step 2 below), communication (Hypothesis: network degradation, Step 3), or data loading (Hypothesis: storage/pipeline bottleneck, Step 4) before you've touched a single other piece of telemetry.

If none of this instrumentation exists yet, building it is the first, immediate deliverable of the investigation — and it's worth saying explicitly in an interview that "we don't have per-rank, compute/comm/data-split step-time telemetry" is itself a finding, not just a blocker; a staff engineer's response to discovering this gap mid-incident should be "instrument it now, even coarsely, because the alternative is guessing," and should also flag it as a durable infrastructure investment to make standard practice for every future training run, not just this one.

## Step 1b: A Concrete Checklist for the First Five Minutes

- [ ] Pull the throughput time series and classify the shape: step-function or gradual.
- [ ] Pull per-rank step-time and confirm whether the degradation is uniform or localized.
- [ ] Pull the compute/communication/data-wait breakdown, before-and-after the regression window.
- [ ] Check the deploy/config/scheduler change log for anything at the exact regression-onset timestamp.
- [ ] If the shape is a step-function, check specifically for a concurrent cluster event (a node swap, a scheduler-triggered rebalance, a network-maintenance window).
- [ ] If the shape is gradual over hours, weight the memory-fragmentation/leak hypothesis (Step 5) higher from the start, since that's the hypothesis whose signature is specifically gradual.

## Step 2: Hypothesis — A Straggler Node

**The mechanism.** In synchronous distributed training (data-parallel gradient all-reduce, or any pipeline/tensor-parallel scheme with synchronization points), the overall step time is bounded by the *slowest* participant — every rank waits at the synchronization barrier for the last one to arrive. A single underperforming GPU, node, or link degrades the throughput of the *entire* job, not just its own local work, which is precisely why a straggler is a disproportionately costly failure mode relative to its apparent scope (one bad component out of thousands can produce a global slowdown).

**Concrete causes of a straggler appearing with no code change:**
- **Thermal throttling** — a GPU or node running hotter than normal (a failing fan, a cooling-system degradation, unusually high ambient datacenter temperature) clocks down to stay within thermal limits, silently reducing its achievable FLOP/s without crashing or erroring.
- **ECC memory errors accumulating** — HBM ECC-correctable-error rates climbing on a specific GPU is a classic leading indicator of impending hardware failure; the GPU still functions and produces correct results (that's what ECC correction is for) but at reduced effective throughput, and continuing to climb often precedes an eventual hard failure.
- **A "gray failure"** — a node that hasn't crashed but is running measurably slower for a subtler reason: a driver/firmware state issue, a NUMA-affinity misconfiguration that crept in after a node reboot (e.g., the scheduler restarted a process without correctly reapplying CPU/NUMA pinning), or a partially-failed NVLink/PCIe lane still nominally "up" but running at reduced link width/speed.
- **Contention from a co-located process** — on shared or imperfectly isolated infrastructure, another job (a monitoring agent, another tenant's workload if isolation is imperfect, or even the cluster's own health-check tooling) competing for the same node's CPU, memory bandwidth, or NIC.

**How to confirm.** Per-rank step-time telemetry (Step 1) should directly identify the offending rank(s) if this is the cause — look for one or a small number of ranks whose local compute time increased while others' didn't. Cross-reference against the cluster's hardware telemetry (GPU temperature, clock speed, ECC error counters, NVLink/PCIe link status) for exactly those ranks. If available, run an isolated microbenchmark (a standalone GEMM or NCCL all-reduce benchmark) directly on the suspect node, outside the training job, to get a clean throughput number uncontaminated by the training job's own synchronization behavior — this is the single most reliable confirmation step, because it isolates the hardware from the distributed-training context entirely.

**The fix.** Cordon/drain the offending node from the job (most modern training-orchestration setups support hot-swapping a failed or degraded node for a healthy spare without a full job restart, though the specifics are orchestration-stack-dependent), resume training with the replacement node, and file the degraded node for hardware diagnostics/repair outside the training job's critical path.

## Step 2b: Straggler Root Causes, Tabulated

| Cause | Detection signal | Fix |
|---|---|---|
| Thermal throttling | GPU clock speed down, temperature up, on affected rank only | Cool/replace node; check datacenter cooling if widespread |
| Rising ECC error rate | HBM ECC-correctable-error counters climbing on one GPU | Preemptively drain before a hard failure occurs |
| Gray failure (driver/NUMA/firmware) | Reduced local throughput with no obvious hardware fault; isolated microbenchmark confirms | Reboot/reprovision the node; verify NUMA pinning post-reboot |
| Contention from co-located process | CPU/memory-bandwidth/NIC usage on the node inconsistent with expected training-job-only load | Improve isolation; escalate to platform team if a scheduling bug |

## Step 3: Hypothesis — Network / Interconnect Degradation

**The mechanism.** Distributed training at scale is communication-bound in specific, predictable ways: data-parallel gradient synchronization (all-reduce) volume scales with model size and is largely independent of batch size per step; tensor-parallel and MoE expert-parallel communication (all-reduce within TP groups, all-to-all dispatch/combine for MoE) is latency- and bandwidth-sensitive and, for MoE specifically, scales with token count × top-k routing degree (per `..\..\OpenSource\007_DeepSeek_V3.md`, Section 4) largely independent of total expert count. A degradation anywhere in the interconnect fabric — a flaky NVLink connection within a node, a degraded InfiniBand/RoCE link between nodes, congestion from another workload sharing the same network fabric, or a routing/topology change made by the cluster's network layer without the training job's knowledge — increases the communication component of step time without touching compute at all.

**Concrete causes:**
- **A partially-failed link** running at reduced width or in a degraded/retry-heavy state rather than fully down (fully-down links usually produce hard errors that are easy to spot; partially-degraded links are the insidious case, because the job keeps running, just slower).
- **Network congestion from other tenants/jobs** sharing switches or fabric segments, if the cluster's network isolation isn't perfect — a classic "no code change on our side" cause, because the actual root cause is external to the job entirely.
- **A topology-aware collective operation silently falling back to a less efficient path** — many high-performance collective-communication libraries (NCCL and similar) choose communication algorithms/topologies based on detected hardware configuration at initialization; if a node was replaced or a link's characteristics changed, a subsequent job restart or dynamic re-topology event could cause the library to select a less optimal algorithm than before, without any explicit error.
- **Cross-node checkpoint I/O contention** — if checkpointing traffic shares the same network fabric as training communication and checkpoint frequency/size has crept up (or a checkpoint write is unusually slow due to a storage-side issue), this can manifest as intermittent communication-time bloat correlated with checkpoint events specifically.

**How to confirm.** The compute/communication split from Step 1 should already point here if communication time specifically grew. Beyond that: run a standalone NCCL (or equivalent) bandwidth/latency benchmark between the specific node pairs involved in the job's parallelism topology, compare against known-good baseline numbers for the same hardware/topology, and check cluster-level network telemetry (switch port error counters, retransmission rates, link utilization from other tenants if visible) for the affected time window. If the job uses a fixed parallelism topology (a specific TP/PP/DP/EP grid), verify that the actual runtime communication pattern matches the intended topology — a topology-selection regression (the "silent fallback" cause above) is confirmed by comparing observed collective-operation latencies against the expected latency for the intended algorithm.

**The fix.** Depending on root cause: reroute around the degraded link (drain and replace the affected node, exactly as in Step 2, if the degradation is node/link-specific), escalate to network/datacenter operations if the cause is external contention, or explicitly pin/force the collective-communication algorithm/topology if a silent suboptimal-fallback is confirmed rather than relying on auto-detection.

## Step 3b: A Quick FAQ on Network Hypotheses

- **How do you distinguish "our job's config is wrong" from "the fabric itself is degraded"?** Run the standalone communication benchmark at the exact topology/algorithm the job is configured to use; if it matches known-good historical baselines, suspect the job's own topology-selection logic; if it doesn't, suspect the fabric itself.
- **Why does a routing change made for unrelated maintenance show up as a training-job problem?** Because collective-communication performance is highly sensitive to path length and contention, and a routing change that's invisible to most workloads can be very visible to a workload issuing frequent, latency-sensitive all-reduce/all-to-all calls at the volume a large training job does.
- **Is this hypothesis more or less likely than a straggler node, a priori?** Roughly comparable in frequency across the field, but network causes are more likely to be *global* (as in Step 3's worked narrative) while straggler causes are more likely to be *localized* — Step 1's per-rank uniformity check is what actually distinguishes them, not a prior guess about which is more common.

## Step 3c: A Quick FAQ on Distinguishing the Four Hypotheses Fast

- **If you could only pull one piece of telemetry before branching, what would it be?** The compute/communication/data-wait step-time breakdown — it alone eliminates two of the four hypotheses in most real incidents within minutes.
- **What's the fastest way to rule out data-loading (Step 4) specifically?** Check GPU utilization at step boundaries — GPUs idling specifically while waiting for input is a distinctive signature that compute-bound or communication-bound hypotheses don't produce.
- **Is it ever correct to skip straight to a fix without confirming the hypothesis first?** Only when the fix is cheap and reversible (e.g., trying a node drain-and-replace when a straggler is merely suspected, not yet confirmed) — for expensive or hard-to-reverse fixes, confirming first is worth the extra minutes.

## Step 4: Hypothesis — Storage / Data-Loading Bottleneck

**The mechanism.** If GPUs are waiting on the data loader to produce the next batch (visible directly as "data wait time" in a proper step-time breakdown, or indirectly as GPU utilization dropping below expected levels at the start of each step), the bottleneck is upstream of the GPUs entirely — in storage I/O throughput, network bandwidth between storage and compute nodes, or CPU-side data-loading/preprocessing (detokenization, packing, on-the-fly augmentation if any) not keeping pace with GPU consumption rate. This is exactly the systemic risk flagged in `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`, Step 7, and in Llama 3's own infrastructure discussion of feeding a 16,000-GPU cluster fast enough (`..\..\OpenSource\003_Llama3.md`, Section 8) — and it is a bottleneck that can appear with "no code change" for reasons entirely outside the training job's own code:

- **Storage-side contention** — another job or process reading/writing to the same shared storage backend (object store, distributed filesystem) at the same time, degrading achievable throughput for everyone including this job.
- **A change in which data shards are currently being read** — if the corpus isn't uniformly distributed across storage in a way that gives consistent read throughput (e.g., some shards live on a slower storage tier, or a specific storage node/shard is itself degraded), simply progressing further into the dataset can change effective I/O throughput even with zero code changes.
- **CPU-side data-loader worker starvation** — if data-loader worker processes are competing for CPU cores with other processes on the same host (including, ironically, monitoring/logging agents that have grown more resource-hungry over the run, or a memory leak in a long-running worker process degrading its own throughput over time — see Step 5), preprocessing throughput can degrade purely from CPU contention with no change to the training code itself.
- **A slow-growing metadata or indexing overhead** — some data-loading implementations have per-epoch or per-shard-boundary bookkeeping costs that scale poorly (e.g., a shuffle-buffer or shard-index structure that grows unboundedly over a long run) — a real, previously-observed class of bug where a data loader's performance measurably degrades over the *duration* of a long training run purely due to accumulating internal state, independent of any external system change.

**How to confirm.** Data-wait time in the step-time breakdown; GPU utilization (`nvidia-smi`-style utilization percentage, or better, active-SM-time-based utilization if available) dropping specifically at step boundaries in a way consistent with GPUs idling for input rather than being busy; storage-side I/O throughput/latency metrics for the relevant storage backend during the degraded window; CPU utilization and memory usage on data-loader worker processes specifically.

**The fix.** Depending on root cause: increase data-loader prefetch depth/worker count if CPU/storage headroom allows, move hot data to faster storage tiers, isolate data-loading CPU resources from other host processes, or patch the specific data-loader bug if a growing-internal-state issue is confirmed.

## Step 5: Hypothesis — Memory Fragmentation / Leak

**The mechanism.** This is the hypothesis most specifically tied to *gradual* degradation over hours (per Step 1's shape characterization) rather than a step-function drop, and it's the one most likely to be dismissed prematurely because "no code change" makes memory issues feel implausible — but GPU memory allocators (and, more insidiously, host-side Python/CUDA allocator behavior) can degrade in throughput over a long-running process even with completely static code, for reasons including:

- **Allocator fragmentation** — as a long-running training process allocates and frees activation memory of varying sizes across many steps (particularly with dynamic shapes, e.g., variable-length sequences before packing, or activation-checkpointing patterns that allocate/free at irregular sizes), the underlying memory allocator's free-list can become fragmented over time, forcing progressively more expensive allocation-search or memory-defragmentation work per step, or forcing occasional expensive `cudaMalloc`/`cudaFree` calls to the OS-level allocator that a well-behaved caching allocator would normally avoid.
- **A slow memory leak** — a reference-cycle or cache that grows unboundedly (a logging buffer, a metrics-accumulator, a debugging hook left enabled, or a framework-level cache that isn't being evicted correctly) consuming progressively more host or device memory over the run, eventually forcing the allocator into a more expensive regime (increased swapping/paging pressure host-side, or forcing PyTorch/framework-level cache eviction and re-allocation device-side) well before it would produce an outright out-of-memory crash.
- **Increasing garbage-collection overhead** — in Python-based training loops specifically, an accumulating object graph (even without a true "leak" in the sense of unreachable-but-retained memory) can increase per-step garbage-collector work over time, a genuinely observed cause of throughput degrading gradually over many hours in long-running Python processes.

**How to confirm.** Track GPU memory allocated/reserved (not just "used," since a caching allocator's reserved-but-not-currently-allocated memory is exactly where fragmentation manifests) over the run's duration — a reserved-memory time series that climbs steadily (even while allocated memory is roughly flat) is close to a direct signature of fragmentation. Track host-side (CPU) process memory (RSS) over time for a similar signature. If the framework exposes allocator statistics (e.g., PyTorch's memory-allocator-summary tooling), check for a growing number of allocator "cache misses" or increasing fragmentation-specific counters over the degraded window.

**The fix.** A periodic, scheduled process restart (resuming cleanly from the last checkpoint) is the pragmatic immediate mitigation for both fragmentation and slow leaks — restarting the process resets the allocator's state entirely — and is a legitimate standard practice for very long-running training jobs specifically *because* this failure mode is common enough across the field that proactive periodic restarts are cheaper than chasing every possible slow-leak source. The root-cause fix (finding and eliminating the specific leaking reference, or tuning the allocator's caching/fragmentation-avoidance behavior directly) is worth pursuing in parallel but shouldn't block getting the job back to full throughput via a scheduled restart if one is due or overdue.

## Step 5b: A Quick-Reference Signature Table Across All Four Hypotheses

| Signature | Straggler (Step 2) | Network (Step 3) | Data loading (Step 4) | Memory fragmentation/leak (Step 5) |
|---|---|---|---|---|
| Shape of degradation | Often step-function (a node degrades suddenly) | Either | Either | Gradual, over hours |
| Per-rank uniformity | Localized to one/few ranks | Usually global if fabric-wide, localized if link-specific | Usually global (shared storage) | Global (every rank's process degrades similarly) |
| Step-time breakdown component | Compute time up on affected ranks | Communication time up | Data-wait time up | Compute time up (allocator overhead), sometimes with no other component visibly changing |
| Confirming check | Isolated hardware microbenchmark on suspect node | Standalone network bandwidth/latency benchmark | Storage/CPU telemetry, GPU idle-for-input pattern | GPU/host memory (allocated vs. reserved) time series |
| Typical fix | Drain and replace node | Reroute, escalate to network ops, or force topology | Increase prefetch/workers, move to faster storage | Scheduled restart, then root-cause the leak |

## Step 5c: A Worked Narrative

- **T+0 (relative to when the regression started, discovered retrospectively):** on-call notices via a dashboard alert that hourly-averaged throughput has drifted down about 30% versus the prior day's baseline, with no associated deploy or config change logged.
- **T+5 minutes:** step-time breakdown is pulled; communication time has grown from roughly 18% to 34% of total step time, while compute and data-wait are essentially unchanged — this immediately narrows the investigation to Hypothesis 3 (network) and away from Hypotheses 2, 4, and 5.
- **T+10 minutes:** per-rank communication time is checked; it's elevated broadly across most ranks, not concentrated on one or two — this argues against a single degraded link/node and toward a fabric-wide or topology-level cause.
- **T+20 minutes:** a standalone collective-communication benchmark is run between representative node pairs and compared against the last known-good baseline for this cluster; measured bandwidth is roughly 25% below baseline across the board.
- **T+30 minutes:** cluster network operations is engaged; they confirm a routing change was made fleet-wide the previous evening for unrelated maintenance reasons, inadvertently placing this job's traffic on a less optimal path.
- **T+40 minutes:** routing is corrected; throughput recovers to baseline within the next few completed steps, confirming the fix.
- **Follow-up:** a standing alert is added that specifically flags a sustained increase in the communication-time fraction of step time (not just an aggregate throughput drop), and a request is filed with network operations to notify training-job owners before any future fleet-wide routing change that could affect active jobs.

This narrative illustrates exactly why Step 1's compute/communication/data-wait breakdown is the highest-leverage single piece of telemetry in this entire diagnostic space: it collapsed a four-hypothesis investigation into a single, correctly-targeted branch within five minutes, and the specific root cause here (an external routing change with zero visibility into the training job's own logs) would have been very difficult to find quickly without that breakdown pointing directly at communication as the affected component.

## Step 5e: A Quick FAQ on Memory-Related Regressions

- **How do you distinguish a genuine leak from expected, bounded caching behavior?** Track reserved memory over a duration much longer than any single caching mechanism's expected steady-state — bounded caching plateaus; a genuine leak (or fragmentation) keeps climbing indefinitely without bound.
- **Is a scheduled restart a real fix or just a band-aid?** It's a legitimate, standard mitigation specifically because this failure class is common enough across the field that proactive restarts are cheaper in expectation than chasing every possible slow-leak source — but it should be paired with a parallel root-cause ticket, not treated as the final word.
- **Does this hypothesis apply to CPU-side data-loader processes too, or only GPU memory?** Both — host-side Python process memory (RSS) should be tracked with the same discipline as device memory, since a CPU-side leak in a long-running data-loader worker produces a structurally identical symptom.

## Step 6: Isolating the True Cause Efficiently — The Actual Investigation Order

Given all four hypotheses are plausible a priori, the efficient investigation order (rather than checking all four in parallel, which is expensive in engineer-time and easy to do sloppily) is:

1. **Pull the step-time breakdown (compute/comm/data-wait) and the shape of the degradation (step-function vs. gradual) first — this alone should point strongly at one or two of the four hypotheses** and let you skip the other two entirely in the common case.
2. **Check per-rank uniformity second** — uniform-across-all-ranks degradation rules out Branch 2 (straggler) and points toward Branch 3 (network, if global communication got globally slower) or Branch 4/5 (data/memory, if every rank's local pipeline degraded roughly in lockstep, which happens when the underlying cause is a shared resource like storage-backend contention rather than a per-node hardware issue).
3. **For the surviving hypothesis, run the specific confirming check from that section** (isolated hardware microbenchmark for straggler; standalone network benchmark for interconnect; storage/CPU telemetry for data-loading; memory time-series for fragmentation/leak) rather than a broad, unfocused sweep of every possible metric.
4. **Fix and validate**: after applying the targeted fix, confirm throughput has actually recovered to the pre-regression baseline (not just "improved somewhat") before considering the incident closed, and add whatever monitoring/alerting gap this incident exposed (per-rank step time, allocator memory time series, network link health, whichever wasn't already instrumented) as a durable follow-up, so the next occurrence of this exact failure mode is caught and localized in minutes rather than hours.

## Step 5d: A Second Worked Narrative — Memory Fragmentation

To illustrate the gradual-degradation branch specifically, since it's mechanistically the most different from the other three:

- **Day 1:** throughput is at expected baseline.
- **Day 3:** throughput has drifted down roughly 8% — small enough to be within normal noise and not yet alarming.
- **Day 5:** throughput has drifted down roughly 22%, and the trend across the three data points is now clearly monotonic rather than noisy, prompting investigation.
- **Investigation:** step-time breakdown shows the drift concentrated in compute time, not communication or data-wait — this is an unusual signature, since a hardware or network cause would typically show up as communication or produce a step-function rather than a slow monotonic drift in the compute component specifically.
- **Memory time series:** GPU reserved-memory (not allocated-memory) has grown steadily since the job started five days ago, consistent with allocator fragmentation rather than a genuine leak of a specific object (allocated memory itself is flat; only the allocator's reserved-but-fragmented pool is growing).
- **Resolution:** a scheduled process restart, resuming cleanly from the last checkpoint, restores throughput to baseline immediately — confirming the fragmentation hypothesis by the simple fact that resetting allocator state fixed it.
- **Follow-up:** a periodic, calendar-scheduled restart cadence is added for all long-running jobs on this training framework version, as a standing mitigation, while a separate ticket investigates whether a specific allocation pattern in the framework's activation-checkpointing implementation is the underlying root cause worth fixing directly rather than just working around.

## Step 6b: Common Mistakes This Scenario Is Designed to Surface

- Restarting the entire job speculatively before pulling any telemetry, hoping a restart happens to fix an undiagnosed problem.
- Checking all four hypotheses in an unstructured, parallel sweep rather than letting the step-time breakdown and per-rank uniformity checks narrow the search first.
- Assuming "no code change" means the cause must be external, and therefore skipping data-loader and memory-allocator checks that are entirely internal to the job's own long-running process state.
- Treating a gradual degradation and a step-function degradation as requiring the same investigation, rather than using the shape itself as a discriminating signal (memory/allocator issues tend to be gradual; hardware/config-change issues tend to be step-function).
- Failing to compare against a recorded historical baseline for network/storage throughput, and instead relying on gut feel for what "normal" looks like.
- Not looping in adjacent teams (network operations, storage operations) early enough when the evidence points outside the training job's own code, delaying resolution while the on-call engineer tries to fix something they don't own.

## Step 6d: A Summary Table of the Whole Investigation

| Order | Check | Time cost | Confirms/rules out |
|---|---|---|---|
| 1 | Throughput shape (step-function vs. gradual) + per-rank uniformity | Minutes | Narrows to 1-2 hypotheses immediately |
| 2 | Compute/communication/data-wait step-time breakdown | Minutes | Points to specific hypothesis category |
| 3 | Hypothesis-specific confirming check (hardware microbenchmark, network benchmark, storage/CPU telemetry, or memory time series) | 10-30 minutes | Confirms the specific root cause |
| 4 | Apply targeted fix, validate throughput recovery to baseline | Varies | Confirms resolution |
| 5 | Add or strengthen the monitoring gap this incident exposed | Follow-up | Prevents recurrence at this diagnostic cost next time |

## Step 6c: A Pre-Incident Checklist — What Should Already Exist Before This Happens

- Per-rank, compute/communication/data-wait step-time breakdown, logged continuously, not just available on request.
- A recorded historical baseline for network bandwidth/latency and storage I/O throughput on this cluster, refreshed periodically, so "current vs. baseline" is a real comparison and not a guess.
- GPU health telemetry (temperature, clock speed, ECC error counters, link status) exported to a dashboard, not just accessible via manual `nvidia-smi`-style queries during an active incident.
- Host and device memory (allocated vs. reserved) time series, logged for the duration of every long-running job.
- A known, tested procedure for draining and replacing a single node without a full job restart.
- A defined escalation path to network and storage operations teams, established before an incident, not improvised during one.

## Step 6e: Why This Question Recurs at the Staff Level

Much like `003_Debugging_A_Loss_Spike_Mid_Training.md`'s scenario, this question has no single memorizable answer — the right response is entirely a function of which telemetry you'd pull first and how you'd read it, which makes it a strong filter for genuine hands-on distributed-systems experience versus familiarity with the vocabulary alone. It also directly tests whether a candidate defaults to guessing (trying fixes speculatively) or to measurement (narrowing the hypothesis space with cheap, targeted checks before touching anything) — and the latter is the posture every other incident-response scenario in this module is built around as well.

The throughline across all four branches, and the point most worth making explicit in an interview: **step-time-breakdown telemetry (compute vs. communication vs. data-wait, per-rank) is the single piece of instrumentation that makes this entire diagnostic tree tractable**, and a staff engineer's first instinct in a real incident where that telemetry doesn't already exist should be building it immediately, not guessing without it.
