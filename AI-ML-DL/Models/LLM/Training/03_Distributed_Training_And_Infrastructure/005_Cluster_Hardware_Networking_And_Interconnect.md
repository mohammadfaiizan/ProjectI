# Cluster Hardware, Networking, and Interconnect

## 1. Why this file exists in a training-systems module

Every cost model in `001`–`004` — all-reduce volume, TP's per-layer all-reduce latency sensitivity,
ZeRO-3's extra all-gathers, EP's all-to-all — ultimately resolves to a real number of seconds only
once you plug in *actual* bandwidth and latency figures for *actual* hardware. This file supplies
that hardware context and, more importantly, the structural reasoning about *why* network topology,
not raw FLOPs, is so often the binding constraint at frontier scale. Treat the specific numbers
below as order-of-magnitude anchors for reasoning, not as guaranteed-current spec sheet values —
hardware generations move fast enough that exact figures should be verified against current vendor
datasheets when precision matters for a specific claim; what should not need re-verification is the
*relative* structure (NVLink is roughly an order of magnitude faster than inter-node fabric; per-GPU
FLOPs have grown faster than per-GPU network bandwidth across generations) because that structural
trend is the durable, interview-relevant fact.

## 2. The accelerator landscape

**NVIDIA A100 (Ampere, 2020).** 40GB or 80GB HBM2e variants; ~1.6–2TB/s memory bandwidth on the 80GB
SKU; roughly 312 TFLOPS bf16 dense tensor-core throughput (marketing figures for
"TF32"/sparsity-assisted numbers are higher and should not be confused with dense bf16 throughput —
a common source of MFU miscalculation, flagged explicitly in
`007_Training_Efficiency_Metrics_MFU_And_Utilization.md`). NVLink 3 provides roughly 600GB/s
aggregate per-GPU intra-node bandwidth. This was the workhorse generation for
GPT-3-era-and-immediately-after training at scale.

**NVIDIA H100 (Hopper, 2022–23).** 80GB HBM3, ~3.35TB/s memory bandwidth; roughly 990 TFLOPS dense
bf16 tensor-core throughput, and — the generation's headline new capability relevant to `004` —
native FP8 tensor cores at roughly double that (~1979 TFLOPS dense FP8). NVLink 4 raises intra-node
aggregate bandwidth to roughly 900GB/s per GPU. This is the generation DeepSeek-V3's FP8 recipe
(`..\OpenSource\007_DeepSeek_V3.md`) is designed around, and the generation most current
frontier-lab training runs are reported to use.

**NVIDIA H800 (Hopper, China-market variant).** Same compute die and FLOPs as H100 for both bf16 and
FP8, but with **NVLink bandwidth deliberately reduced** (roughly less than half of H100's, reported
in the ~400GB/s range rather than ~900GB/s) as a consequence of US export-control restrictions
targeting interconnect bandwidth specifically rather than raw compute.
`..\OpenSource\007_DeepSeek_V3.md` is the concrete, well-documented case study for what this
constraint forces downstream: because H800's compute is unrestricted but its intra-node
communication bandwidth is cut, any workload whose bottleneck is communication rather than compute
(per-layer TP all-reduces, and especially MoE's cross-device all-to-all,
`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md`) is disproportionately affected relative
to the same workload on unrestricted H100 hardware — which is precisely the motivation DeepSeek
gives for investing in custom, PTX-level-tuned communication kernels and the DualPipe schedule
rather than relying on off-the-shelf collective implementations tuned for H100-class bandwidth.

**B200/GB200 (Blackwell, 2024–25) and beyond.** Reported to push HBM capacity, bandwidth, and FLOPs
further, with GB200's NVL72 rack-scale design notably extending the *NVLink domain itself* beyond a
single 8-GPU node to a much larger (72-GPU) NVSwitch-connected domain — directly relevant to the "TP
must stay intra-node" rule of thumb from `001`, since a larger NVLink domain means a larger TP
degree becomes viable at NVLink-class bandwidth rather than being capped at 8. Exact production
figures and availability are moving fast enough at time of writing that specific numbers should be
treated as provisional; the structural point (rack-scale NVLink domains are the industry's direct
response to TP/EP's bandwidth appetite) is the durable takeaway.

**TPU pods (Google).** A structurally different design point worth understanding at a conceptual
level even without hands-on TPU experience: TPUs use a systolic-array matrix-multiply unit (the MXU)
rather than a general SIMT GPU core, optimized specifically for the dense matmul patterns dominant
in transformer training, and TPU pods connect many chips via a dedicated high-bandwidth, low-latency
mesh/torus interconnect (**ICI**, inter-chip interconnect) purpose-built for this workload rather
than adapted from general-purpose datacenter networking — conceptually the TPU-world analogue of
NVLink, but architected as a pod-wide fabric from the outset rather than a node-local domain
extended outward. The systems-level implication for parallelism strategy is the same regardless of
vendor: **the fundamental TP/PP/DP/EP taxonomy from `001`/`002` is hardware-agnostic; what changes
across TPU vs. GPU clusters is which specific interconnect tier a given communication pattern lands
on, and therefore which parallel-axis choices are cheap versus expensive on that specific fabric.**
Exact current-generation TPU pod sizes, ICI bandwidth, and per-chip FLOPs are not detailed here with
confidence and should be verified against Google's published specs if needed for a specific claim.

## 3. Intra-node interconnect: NVLink

NVLink is a direct GPU-to-GPU interconnect, physically distinct from (and far faster than) the PCIe
bus that also connects GPUs to the host CPU. Within a single node (canonically 8 GPUs in a DGX-class
server), an **NVSwitch** fabric connects every GPU to every other GPU at full NVLink bandwidth
simultaneously — i.e., it is not a shared bus that degrades as more GPUs communicate concurrently,
but a genuine any-to-any switched fabric, which is exactly the topology an all-reduce or all-to-all
among those 8 GPUs wants.

Two properties matter more than the specific bandwidth number: **bandwidth** (several hundred GB/s
to ~900GB/s aggregate per GPU depending on generation, as above) and **latency** (sub-microsecond,
GPU-to-GPU, with no intervening network switch hop of the kind inter-node traffic must cross). Both
matter because TP's per-layer all-reduce (`001`, Section 3.4) is *inline* in the critical path of
every layer's forward and backward — it cannot be overlapped with unrelated compute the way DP's
once-per-step gradient all-reduce can (`001`, Section 2.3) — so both the raw bandwidth (affecting
how long the data transfer itself takes) and the latency (affecting the fixed per-call overhead paid
regardless of data size) directly stall compute. This dual sensitivity to both bandwidth *and*
latency, not just bandwidth, is the precise reason TP is confined to the NVLink domain rather than
merely "preferring" it.

## 4. Inter-node interconnect: InfiniBand / RoCE

Nodes talk to each other over a separate physical network, typically **InfiniBand** (IB) in
frontier-scale GPU clusters, or **RoCE** (RDMA over Converged Ethernet) as an Ethernet-based
alternative offering similar RDMA semantics over commodity-adjacent Ethernet hardware.
Current-generation NDR InfiniBand offers roughly 400Gb/s (~50GB/s) per NIC, and rail-optimized
cluster designs commonly provision one NIC per GPU (an "8 rails" design for an 8-GPU node)
specifically so that inter-node collective operations involving all 8 GPUs of a node can each use a
dedicated NIC rather than contending for a shared one.

Compared to NVLink, InfiniBand is roughly an order of magnitude lower bandwidth per link and
meaningfully higher latency (low microseconds rather than sub-microsecond, since traffic now
traverses NICs, cables, and switch hops rather than a direct on-board fabric). This is the concrete
basis for the placement rule from `001`, Section 5.1: **PP's point-to-point activation handoffs and
DP's once-per-step, overlappable gradient all-reduce both tolerate InfiniBand's higher latency and
lower bandwidth reasonably well; TP's inline per-layer all-reduce does not**, and EP's all-to-all
(`002`) sits in between — tolerant of *some* added latency but, like TP, sitting in the critical
path with limited ability to overlap, which is exactly why cross-node EP (forced whenever an MoE
model has more experts than fit in one node, as in DeepSeek-V3's 256 routed experts) is treated as a
genuinely hard problem requiring custom kernel and schedule engineering rather than routine
off-the-shelf collective calls.

**RDMA (Remote Direct Memory Access)**, the mechanism underlying both IB and RoCE's low-latency
transfers, allows one node's NIC to write directly into another node's GPU memory (via GPUDirect
RDMA, bypassing a copy through host CPU memory) without invoking the receiving side's CPU for each
transfer — removing a CPU-mediated copy step that would otherwise add both latency and CPU-time
overhead to every inter-node communication. GPUDirect RDMA's practical relevance: whether a
cluster's NICs and PCIe topology are actually configured to support direct GPU-to-NIC data paths
(rather than falling back to a slower CPU-staged path) is a real, checkable infrastructure detail
that materially affects achieved inter-node bandwidth, and is exactly the kind of configuration
detail worth checking first when observed inter-node throughput falls well short of the NIC's rated
bandwidth (a concrete diagnostic thread picked up again in
`008_Debugging_Distributed_Training_Failures.md`).

## 5. Topology-aware placement: it matters which specific devices are talking to which

Bandwidth and latency numbers per link are necessary but not sufficient — the **topology**
connecting many nodes' NICs together, and which specific ranks get assigned to which specific
physical devices, determines whether a given collective operation actually achieves anything close
to the per-link bandwidth in aggregate.

**Fat-tree topologies and bisection bandwidth.** Large IB clusters are typically built as a fat-tree
(or similar multi-tier folded topology): groups of nodes connect to a "leaf" switch, leaf switches
connect up to "spine" switches, and so on. Bandwidth *within* a leaf switch's group of
directly-attached nodes is at full line rate; bandwidth *across* the spine (between nodes attached
to different leaf switches) is limited by however many spine-level links exist relative to the
number of nodes needing to traverse them — the cluster's **bisection bandwidth**, i.e., the
worst-case aggregate bandwidth available if the cluster were split into two halves and every node in
one half needed to talk to a node in the other half simultaneously. A well-designed
(non-oversubscribed) fat-tree targets full bisection bandwidth (any node can talk to any other node
at full NIC line rate even under worst-case traffic patterns); a cost-optimized, oversubscribed
design trades this away, meaning **some pairs of nodes communicate at full speed while others,
depending on their position in the tree, contend for shared spine links and see markedly lower
effective bandwidth** — a real, physical asymmetry that a naive rank-to-node assignment can walk
straight into.

**Why rank placement is a live engineering decision, not an afterthought.** Given such a topology,
assigning which physical GPU gets which logical rank in the TP/PP/DP/EP grid (`001`, Section 5.1)
directly determines how much of each collective's traffic stays within a cheap, high-bandwidth
region of the topology versus crossing an expensive, potentially oversubscribed one. A
**rack-aware** or **topology-aware** scheduler places a TP group's ranks on GPUs within the same
node (already established as mandatory, Section 3–4), places a PP stage's or DP group's ranks to
minimize spine-crossing traffic where the topology allows, and — critically — a naive or
topology-*unaware* job launcher that scatters a single tightly-communicating group (e.g., an EP
group spanning many experts) arbitrarily across the cluster can produce measured throughput far
below what the same GPU count would achieve under topology-aware placement, for reasons that have
nothing to do with the model, the parallel strategy's algorithmic correctness, or the software
stack's efficiency — purely a placement/scheduling problem. This is a genuinely underrated failure
mode: a training job can be algorithmically and numerically correct, well-profiled at the
single-node level, and still underperform materially at cluster scale purely because the job
scheduler placed communicating ranks far apart in the physical topology.

**GPU-NIC affinity.** Within a node, not every GPU is necessarily equidistant (in PCIe-topology
terms) from every NIC — a node's PCIe switch layout can place a given GPU "closer" to one specific
NIC than to others. Rail-optimized designs (Section 4) exploit this by pairing each GPU with its
nearest NIC for that GPU's outbound inter-node traffic; getting this pairing wrong (e.g., a GPU's
traffic routed through a NIC it must cross an extra PCIe switch hop to reach) adds latency and can
create contention on the "wrong" GPU-to-NIC path — again, a purely infrastructure-configuration
issue, invisible to anything happening inside the training code itself.

## 6. The central practical point: bandwidth, not FLOPs, is very often the actual bottleneck

Put the pieces together and a broader, durable industry trend emerges, worth being able to state
crisply and unprompted in an interview: **across recent hardware generations, per-GPU FLOPs have
grown considerably faster than per-GPU network bandwidth (both intra- and especially inter-node).**
A100 to H100 roughly tripled dense bf16 FLOPs (and, via native FP8, closer to 6x for FP8-capable
workloads); NVLink bandwidth grew from roughly 600GB/s to roughly 900GB/s over the same span (a much
smaller multiple), and inter-node IB bandwidth has grown more slowly still in relative terms. The
practical consequence is that the **ratio of compute to communication bandwidth available per GPU
gets worse with every hardware generation**, meaning that a parallel strategy and cluster
configuration tuned to keep GPUs busy (rather than waiting on network transfers) on one hardware
generation can become communication-bound on the next generation's faster GPUs *even with no change
to the model or the parallel strategy at all*, purely because the compute side raced ahead of the
network side.

This is exactly why MFU (`007_Training_Efficiency_Metrics_MFU_And_Utilization.md`) so consistently
lands well below 100% at frontier scale, and why so much of the engineering effort described across
`001`, `002`, and the DeepSeek-V3 case study cited throughout this file (DualPipe, custom PTX-level
all-to-all kernels, fine-grained overlap) is specifically about **hiding communication behind
compute** rather than reducing communication volume outright — because on modern hardware, the
compute side usually has cycles to spare while communication is happening, and the entire game is
arranging the schedule so those idle compute cycles are exactly when the unavoidable data movement
happens, rather than the two ever needing to happen in strict sequence. A concrete way to state the
diagnostic implication for a live job (developed fully in `007` and `008`): if scaling a job to more
GPUs increases wall-clock step time, or increases it by more than a modest fraction, communication —
not compute — is very likely the actual bottleneck, and the fix belongs in parallel-configuration or
topology-placement space (this file and `001`), not in kernel-level compute optimization.

## 7. A worked bandwidth-vs-compute sanity check

To make Section 6's claim concrete rather than assertion-only: take a 70B dense model, `TP=8`
(Section 3's NVLink domain), bf16. Per `001` Section 3.4, each transformer layer pays 4 TP
all-reduces on an activation tensor of shape `[b, s, d_model]`. At `b=2, s=4096, d_model=8192`
(bf16, 2 bytes/element), that tensor is `2 × 4096 × 8192 × 2 ≈ 134MB`. Using the ring all-reduce
cost model (`001` Section 2.2) at `T=8` NVLink participants and, say, 800GB/s effective per-GPU
NVLink bandwidth: each all-reduce moves roughly `2×(8-1)/8 × 134MB ≈ 235MB` per GPU, taking roughly
`235MB / 800GB/s ≈ 0.29ms`. Four such all-reduces per layer ≈ **1.2ms/layer** of TP communication.
Compare this to the compute time for that same layer's matmuls at H100's ~990 TFLOPS bf16 peak: a
70B model's per-layer FLOPs at this batch/sequence configuration is on the order of a few TFLOPs per
layer, i.e., low single-digit milliseconds at *peak* throughput (and meaningfully more at realistic,
sub-peak achieved throughput). The two numbers landing in the same rough order of magnitude — not
the compute dwarfing the communication by 100x — is precisely why TP communication is a first-order
concern worth actively overlapping and minimizing rather than a rounding error, and why the exact
NVLink bandwidth figure used in this calculation (which varies by hardware generation, per Section
2) directly changes whether a given configuration is compute-bound or communication-bound. Running
this exact style of arithmetic — pick real shapes, compute both sides, compare — is the concrete
skill this file is building toward, and it is exactly the calculation `007`'s MFU diagnostic
framework formalizes into a repeatable procedure.

## 8. NCCL and collective-library tuning as a topology-adjacent lever

The collective-communication library actually issuing the ring all-reduce, all-to-all, and
point-to-point calls discussed throughout `001` and `002` — almost universally NCCL on NVIDIA
hardware — makes its own internal decisions about ring construction, chunk sizing, and transport
selection, and those decisions interact directly with the physical topology described above rather
than being independent of it. A few concrete levers worth knowing at staff depth, precisely because
they sit at the boundary between "topology problem" and "software configuration problem" and are
frequently misdiagnosed as one when they are actually the other:

- **Ring construction versus topology.** NCCL does not necessarily construct its communication rings
in a topology-naive round-robin rank order; a topology-aware NCCL build (using NCCL's topology
detection, informed by `nvidia-smi topo -m` or an explicit topology file) constructs rings that
respect actual physical adjacency — keeping ring "neighbors" physically close wherever possible — so
that the ring all-reduce's per-step transfers ride the fastest available links rather than
unnecessarily crossing an oversubscribed spine link on every single step of the algorithm. A
misconfigured or topology-unaware NCCL setup can construct a ring that is algorithmically correct but
physically pessimal, which looks, from the training job's perspective, exactly like a "bandwidth is
lower than expected" problem — because it is, just one caused by ring construction rather than by the
underlying hardware's actual capability.
- **Environment variables that select transport and tuning behavior**, such as `NCCL_IB_HCA` (which
InfiniBand host-channel adapters to use), `NCCL_SOCKET_IFNAME` (which network interface to use for
the initial handshake and any TCP fallback), and `NCCL_ALGO`/`NCCL_PROTO` (which override NCCL's
automatic algorithm selection for a given collective and message size). Getting any of these wrong —
commonly, after a container or driver update changes what NCCL auto-detects — reproduces exactly the
"looks like a bandwidth or hang problem, is actually a configuration problem" failure class developed
in depth in `008_Debugging_Distributed_Training_Failures.md`, Sections 2 and 5.
- **Message-size-dependent algorithm switching.** NCCL automatically switches between different
underlying algorithms (e.g., ring versus a tree-based all-reduce) depending on message size and
participant count, because the ring algorithm's asymptotically-constant-per-device cost (Section 2.2
of `001`) is not actually optimal at every message size — small messages are latency-dominated, where
a tree-based approach can have a lower fixed number of latency-bound hops than a full ring's `2(N-1)`
steps. This is a further reason to profile actual achieved bandwidth for the *specific* message sizes
a given training configuration produces (Section 7's exact tensor shapes), rather than relying purely
on the idealized ring-all-reduce cost formula, which describes the common large-message case well but
is not the whole story at every scale.
