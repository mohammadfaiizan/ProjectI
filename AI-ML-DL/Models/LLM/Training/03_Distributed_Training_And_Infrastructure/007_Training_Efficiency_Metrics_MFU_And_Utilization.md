# Training Efficiency Metrics: MFU and Utilization

## 1. Definition

**Model FLOPs Utilization (MFU)** is the single most important efficiency metric in large-scale
training, and its definition is deceptively simple:

```
MFU = (achieved FLOPs/second, doing useful model computation) / (hardware's theoretical peak FLOPs/second)
```

The entire difficulty is in the numerator: computing "achieved FLOPs/second doing useful model
computation" correctly requires (a) an accurate FLOPs-per-token formula for the *model*, decoupled
from whatever the *hardware* actually did (which may include wasted recomputation, idle time, or
communication that consumes wall-clock time but performs zero FLOPs), and (b) an accurate,
generation-correct peak-FLOPs figure for the *specific* numeric precision actually in use, since
(per `004_Mixed_Precision_Training_And_Numerical_Stability.md` and
`005_Cluster_Hardware_Networking_And_Interconnect.md`) peak FLOPs differs by roughly 2x between bf16
and FP8 on the same H100 chip, and marketing figures for sparsity-assisted throughput are not the
right denominator for a dense model's MFU.

MFU should be distinguished from the related, looser metric **Hardware FLOPs Utilization (HFU)**:
HFU counts *all* FLOPs the hardware actually executed, including FLOPs spent on
activation-checkpointing recomputation (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`,
Section 7) that MFU's numerator deliberately excludes, since a recomputed forward pass is genuinely
extra hardware work but is not "useful" in the sense of being additional model computation beyond
what a from-scratch backward pass conceptually requires. HFU is always ≥ MFU for any run using
checkpointing, and the gap between the two is itself a direct, quantifiable measure of how much a
given checkpointing configuration costs in raw compute (consistent with `003`'s ~30% recompute
overhead figure for full-layer checkpointing).

## 2. Computing the numerator: FLOPs per token

The standard approximation, used across the PaLM, Chinchilla, and GPT-NeoX-scale-law literature, for
the FLOPs required per token of training (forward + backward combined) for a dense transformer is:

```
FLOPs_per_token ≈ 6 * N
```

where `N` is the number of (non-embedding) parameters. The derivation: a forward pass through a
matmul with `N` parameters costs `2N` FLOPs per token (each parameter participates in one multiply
and one add per token, for a linear layer processing one token — the standard "2 FLOPs per MAC"
convention). Backward pass costs roughly **twice** the forward pass's FLOPs (one pass to compute the
gradient with respect to the layer's input, needed to propagate the gradient further back, and one
pass to compute the gradient with respect to the layer's weights, needed for the optimizer update) —
so backward costs `4N` FLOPs per token, and forward + backward together cost `2N + 4N = 6N` FLOPs
per token. This `6N` approximation is what most public MFU figures (including PaLM's self-reported
46.2% MFU, and Chinchilla-family scaling-law compute estimates) are built on, and it is accurate
enough for FFN- and attention-projection-dominated dense transformers, but it has a known blind spot
worth stating explicitly: **it approximates attention's own compute (the `Q K^T` and
`attention-weights × V` operations) as if it scaled purely with parameter count, when in fact
attention's compute scales with `O(sequence_length^2)` independent of parameter count** — a term the
`6N` shorthand omits.

**The more precise formula** (from the PaLM paper's appendix, and standard in careful MFU reporting)
adds this attention term explicitly:

```
FLOPs_per_token ≈ 6*N + 12*L*h*s
```

where `L` = number of layers, `h` = hidden dimension (`d_model`), and `s` = sequence length — the
`12*L*h*s` term capturing attention's `O(s)`-per-token, sequence-length-dependent cost (the full
attention matrix computation is `O(s^2)` per sequence, i.e., `O(s)` per token within that sequence).
At short-to-moderate sequence lengths relative to `d_model`, this term is a small correction to the
dominant `6N`; at very long context lengths, it becomes non-negligible and eventually dominant — a
detail worth flagging precisely because a `6N`-only MFU calculation will systematically *overstate*
achieved FLOPs (and hence overstate MFU) for long-context training runs, since it's missing a real
compute cost that the hardware is actually paying.

**Implementation as a calculator**, combining both terms and the full utilization formula:

```python
def compute_mfu(
    num_params: float,          # N, non-embedding parameter count
    num_layers: int,            # L
    hidden_dim: int,            # h
    seq_len: int,               # s
    tokens_per_second: float,   # measured throughput: (batch_size * seq_len) / step_time, averaged
    peak_flops_per_gpu: float,  # hardware peak for the ACTUAL precision in use (e.g., 9.9e14 for H100 bf16 dense)
    num_gpus: int,
) -> float:
    flops_per_token = 6 * num_params + 12 * num_layers * hidden_dim * seq_len
    achieved_flops_per_second = flops_per_token * tokens_per_second
    peak_flops_total = peak_flops_per_gpu * num_gpus
    return achieved_flops_per_second / peak_flops_total


# Worked example: 70B model, 96 layers approx, h=8192, s=4096,
# measured 3,200 tokens/sec aggregate on 512 H100s (bf16, peak 9.9e14 FLOPs/s/GPU)
mfu = compute_mfu(
    num_params=70e9,
    num_layers=80,
    hidden_dim=8192,
    seq_len=4096,
    tokens_per_second=3200,
    peak_flops_per_gpu=9.9e14,
    num_gpus=512,
)
# flops_per_token = 6*70e9 + 12*80*8192*4096 ≈ 4.20e11 + 3.22e10 ≈ 4.52e11
# achieved = 4.52e11 * 3200 ≈ 1.45e15 FLOPs/s
# peak_total = 9.9e14 * 512 ≈ 5.07e17 FLOPs/s
# mfu ≈ 1.45e15 / 5.07e17 ≈ 0.0029 -> this specific throughput number is illustrative;
# a realistic well-tuned run at this scale would show tokens/sec roughly two orders of
# magnitude higher than the illustrative 3200 used here to land in the 30-55% MFU band below.
print(f"MFU: {mfu:.1%}")
```

(The illustrative numbers above are chosen for arithmetic clarity, not to represent a realistic
achieved throughput — see Section 3 for what real MFU figures and the throughput they imply actually
look like; the calculator itself, and the `6N + 12Lhs` formula, is the reusable artifact.)

## 3. Typical real-world MFU figures, and why 100% is essentially unreachable

Reported MFU figures from frontier and near-frontier training runs cluster, broadly, in the
**30–55%** range for dense transformer pretraining on GPU clusters at scale — PaLM (TPU v4, dense,
reported ~46.2% MFU) sits near the upper end of what's been publicly disclosed; many GPU-cluster
dense-model runs land in the 35–50% range depending on model size, sequence length, and how
aggressively the parallel configuration has been tuned; MoE models frequently report somewhat lower
MFU than comparably-activated-parameter dense models, for the structural reasons developed in
`002_Mixture_Of_Experts_And_Expert_Parallelism_At_Scale.md` (the additional, data-dependent,
only-partially-overlappable all-to-all communication). Treat all of these as order-of-magnitude,
publicly-reported anchors rather than a table to reproduce with high precision — exact figures vary
run-to-run and are not always disclosed with a consistent, auditable methodology across labs, and
the honest calibration is "roughly a third to a bit over half of peak is normal and considered
good," not a specific decimal.

**Why 100% is not just difficult but structurally unreachable**, worth being able to state as a list
of *distinct* mechanisms rather than a single vague "overhead" hand-wave:

- **Communication is never fully hidden.** Even with best-effort overlap (`001`'s discussion of DP
gradient-all-reduce overlap, and `002`'s discussion of all-to-all overlap engineering), some
communication remains exposed at the boundaries of a step or a pipeline schedule's fill/drain phases
(`001` Section 4.1's bubble) — there is no parallel configuration that reduces this to exactly zero
for any nontrivial (`TP>1` or `PP>1` or `EP>1`) grid.
- **Pipeline bubbles are structural**, not a bug to be fixed away: Section 4.1 of `001` derives the
bubble fraction `(P-1)/(m+P-1)` as an inherent property of any pipeline schedule with `P>1` stages,
shrinkable by increasing `m` but never reaching exactly zero for finite `m`.
- **Kernel efficiency is shape-dependent.** GPU tensor cores achieve their advertised peak FLOPs
only for matmul shapes that map efficiently onto the hardware's native tile sizes; odd or small
dimensions (e.g., a TP-sharded matrix whose per-shard width doesn't divide evenly into the tensor
core's preferred tile size, or a small micro-batch size chosen to control activation memory per
`003`) can leave the tensor cores meaningfully under-utilized even during the time nominally spent
"computing," independent of any communication or bubble concern at all.
- **Memory-bound (non-matmul) operations exist throughout the model and do not benefit from
tensor-core FLOPs at all.** LayerNorm/RMSNorm, softmax, dropout, residual adds, and activation
functions are all comparatively cheap in FLOPs but move real data through memory bandwidth-limited
paths; time spent in these operations counts toward wall-clock step time (the denominator that
determines achieved tokens/sec) without contributing matmul-scale FLOPs to the numerator, so a model
architecture or implementation with proportionally more time in these ops (relative to matmul time)
will show lower MFU purely from this effect, with nothing wrong at all in the parallelism
configuration.
- **Load imbalance** (straggler devices in DP, `008_Debugging_Distributed_Training_Failures.md`;
imbalanced routing in MoE's EP, `002`) forces faster devices to idle waiting on slower ones at every
synchronization point, directly lowering achieved throughput without changing peak hardware
capability.

## 4. A diagnostic framework: which of these is *your* bottleneck?

Given a training run showing lower-than-expected MFU, the following ordered diagnostic sequence
isolates which of Section 3's mechanisms is dominant — this is the practical skill an MFU question
in an interview is actually probing for, more than the ability to recite the formula.

**Step 1 — profile the step, don't guess.** Use a GPU profiler (PyTorch Profiler with CUDA/NCCL
tracing, or NVIDIA Nsight Systems) to get a timeline breakdown of a representative step into three
buckets: time spent in compute kernels, time spent in communication (NCCL) kernels, and time spent
genuinely idle (neither compute nor communication actively running — often a sign of a
scheduling/dependency stall, e.g., waiting on a CPU-side dataloader, or a pipeline bubble). This
single trace answers the highest-level question — is the GPU busy at all — before attempting to
attribute *why* it isn't.

**Step 2 — check whether MFU degrades as you scale GPU count, holding per-GPU work fixed.** Run the
identical configuration (same TP/PP degree, same per-GPU micro-batch size) at two different total
GPU counts (e.g., double the DP degree). If MFU drops meaningfully as GPU count increases, the
DP-axis (or whichever axis grew) communication is the leading suspect — per-GPU compute didn't
change, so a throughput drop implicates the added communication from the larger collective group. If
MFU stays roughly flat, the DP/communication axis is scaling cleanly and the bottleneck lies
elsewhere (compute-kernel efficiency or a fixed, scale-independent bubble).

**Step 3 — isolate the pipeline bubble contribution directly.** Using `001` Section 4.1's formula,
compute the *expected* bubble fraction for the run's actual `(P, m)` configuration. If the measured
idle time in Step 1's trace is well explained by this number, the pipeline schedule is behaving as
expected (not a bug), and the lever to pull is increasing `m` (more, smaller micro-batches) if
activation memory (`003`) allows it, or reducing `P` if memory otherwise permits. If measured idle
time is *larger* than the formula predicts, something beyond the expected structural bubble is
adding idle time — a candidate for Step 5's load-imbalance check, or a scheduling/implementation bug
rather than an inherent property of the chosen configuration.

**Step 4 — check kernel efficiency independent of parallelism.** Take the exact matmul shapes the
model actually uses (given the chosen TP-sharded dimensions and micro-batch size) and benchmark them
in isolation, single-GPU, against the hardware's advertised peak for those specific shapes (many
profilers report this directly as "tensor core utilization" or an achieved-vs-peak-FLOPs ratio per
kernel). If the matmul kernels themselves are running well below peak even in isolation (no
communication or pipeline involved at all), the issue is shape/kernel-level (e.g., a TP degree
producing an awkward per-shard dimension, or a batch size too small to saturate the GPU), and the
fix is in the parallel-configuration or micro-batch-size choice, not in overlap engineering.

**Step 5 — check for load imbalance.** Log per-rank step time (or per-rank compute time
specifically) across a DP group, a PP stage set, or an EP group, and look at the *spread*, not just
the mean — a training run's aggregate step time is set by its slowest participant at every
synchronization barrier (`002` Section 4's exact mechanism for EP; the analogous point holds for any
synchronous collective). A wide spread with a low mean-but-high-max pattern implicates a specific
slow device or an imbalanced router, pointing toward
`008_Debugging_Distributed_Training_Failures.md`'s straggler-diagnosis playbook rather than anything
about the parallel configuration's design being wrong in principle.

**Putting the five steps together as a decision tree:** idle time explained by the pipeline-bubble
formula alone → tune `m`/`P`; idle time beyond that, concentrated in communication kernels,
worsening with scale → communication-bound, revisit topology placement (`005`) or overlap
engineering (`001`/`002`); compute kernels themselves under peak even in isolation →
shape/kernel-level, revisit TP degree or micro-batch size; high variance across ranks at a
synchronization point → straggler or load-imbalance, not a parallelism-configuration problem at all.
A staff-level answer to "MFU is low, what do you check" should walk this tree explicitly, in this
rough order (cheapest, most informative check first), rather than jumping straight to a single
favorite hypothesis.

## 5. A note on comparing MFU across runs and papers

Because the numerator (Section 2's FLOPs-per-token formula) and the denominator (which precision's
peak FLOPs figure is used, and whether it's a dense or sparsity-assisted number) both have real
methodological choices baked in, **MFU figures from different papers/labs are not always computed
identically**, and a naive apples-to-apples comparison can be misleading. Concretely: a paper using
the simpler `6N` approximation (omitting the attention term) will report a slightly different MFU
than one using the full `6N + 12Lhs` formula for the same underlying run, especially at long
context; a paper computing "peak FLOPs" against a sparsity-assisted marketing figure rather than
dense tensor-core throughput will report an artificially *lower* MFU for identical achieved
throughput (a larger denominator, same numerator) — which is a genuinely confusable direction of
error (it looks like *worse* efficiency, when the actual efficiency is unchanged and only the
reference point moved). Before treating any two MFU numbers as comparable, check that both used the
same FLOPs-per-token convention and the same precision-matched, non-sparsity-assisted peak-FLOPs
denominator — a genuinely useful thing to say explicitly in an interview when asked to compare,
e.g., PaLM's and some other model's reported MFU.

## 6. HFU versus MFU, worked

Section 1 defines HFU (Hardware FLOPs Utilization) as the utilization metric that counts *every*
FLOP the hardware actually executed, including checkpointing's recompute overhead, in contrast with
MFU's numerator, which counts only the FLOPs a from-scratch forward-plus-backward pass mathematically
requires. Because activation checkpointing (`003_ZeRO_Optimizer_Sharding_And_Memory_Management.md`,
Section 7) adds a genuine extra forward pass per checkpointed segment, HFU's numerator is strictly
larger than MFU's whenever checkpointing is in use, while the denominator (hardware peak FLOPs) is
identical for both — so `HFU >= MFU` always, with equality only when there is no extra hardware-side
recomputation beyond the mathematically-necessary forward-plus-backward.

**Worked example.** Take the ~33% recompute overhead figure from `003` Section 7 for full-layer
checkpointing (every layer checkpointed, one extra forward-equivalent pass added on top of the
standard `1 forward + 2 backward = 3` FLOPs-per-token units, giving `4` units total). If a run's
*measured* achieved FLOPs/second (from wall-clock throughput, counting everything the hardware did,
checkpointing recompute included) corresponds to an HFU of 45% against peak, the *model-necessary*
portion of that work is only `3/4` of what was actually executed, so MFU is `45% * (3/4) ≈ 33.75%` —
a materially different number for the *same* underlying run, purely depending on which of the two
metrics is being reported. Reporting only "45%" without specifying HFU versus MFU, in a context where
the run uses activation checkpointing, is exactly the kind of ambiguity Section 5 warns about in
general and is worth calling out specifically whenever checkpointing is part of the training
configuration being discussed.

**Which metric to use for which question.** MFU is the right metric for comparing *model-level*
efficiency choices independent of how memory-constrained the implementation happens to be — e.g.,
comparing two different checkpointing strategies that both fit within the same memory budget but
recompute different amounts, where MFU tells you which is using the hardware more efficiently *for
the model's own necessary computation*. HFU is the right metric for predicting actual wall-clock
training time and cost, since it reflects everything the hardware is actually being asked to do,
checkpointing overhead included, and is therefore the more operationally relevant number when the
question is "how long will this run actually take" rather than "how efficient is this model
architecture's use of FLOPs in principle."
