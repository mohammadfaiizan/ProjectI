# Mixture-of-Experts and Expert Parallelism at Scale

## 1. Scope of this file

`..\OpenSource\005_Mixtral8x7B.md` and `..\OpenSource\007_DeepSeek_V3.md` already derive the MoE
routing math in detail: softmax/sigmoid gating, top-k selection, the Switch-Transformer-style
auxiliary load-balancing loss, and DeepSeek-V3's auxiliary-loss-free bias mechanism. This file does
not re-derive any of that. It answers a different question: **once you've decided on a routing
function, how do you actually run an MoE layer across a GPU cluster, and why is that a fundamentally
different systems problem than data, tensor, or pipeline parallelism?** The short answer, developed
below: MoE introduces a *fourth parallelism axis* — expert parallelism (EP) — whose defining
communication primitive (all-to-all) is **data-dependent** in a way none of DP/TP/PP's communication
is, and that data-dependence is the root cause of essentially every systems headache specific to
training MoE models at scale.

## 2. Expert parallelism as a fourth axis

Recall the setup from `001_Parallelism_Strategies_Data_Tensor_Pipeline.md`: TP shards weight
matrices within a layer, PP shards layers across depth, DP shards the batch. None of those three
naturally describes what to do with an MoE layer's `E` independent expert FFNs. You could, in
principle, apply TP to each expert (shard each expert's own FFN weight matrix Megatron-style) — and
this is sometimes done for very large individual experts — but the more fundamental and more common
strategy at scale is **expert parallelism**: place different experts on different devices (or device
groups), and route each token's hidden state to whichever device(s) hold the experts that token's
router selected.

Concretely, for a layer with `E` total experts distributed across `EP` devices (typically `E / EP`
experts per device, assuming even distribution), a forward pass through that MoE layer requires:

1. **Router computation** (cheap, replicated): every device computes the top-k routing decision for
its local tokens.
2. **Dispatch (all-to-all #1):** every device must send each of its local tokens' hidden-state
vectors to whichever remote device(s) hold that token's selected expert(s), and correspondingly
*receive* whatever tokens other devices have routed to the experts it hosts.
3. **Expert compute (local):** each device runs its resident experts' FFN forward pass over whatever
tokens have arrived — a purely local, standard dense-FFN computation.
4. **Combine (all-to-all #2):** the expert outputs must be sent back to the *originating* device
(where that token's other computations, e.g., the residual add, will happen), weighted by the
router's gate value and summed if top-k > 1.

```
# One MoE layer forward pass under expert parallelism, device d's perspective
local_tokens, gates, expert_ids = route(local_hidden_states)     # local, replicated router

# Dispatch: reshuffle tokens so each device receives exactly the tokens
# routed to the expert(s) it hosts. This is the defining communication step.
incoming_tokens = all_to_all(local_tokens, routed_by=expert_ids, comm=ep_group)

expert_outputs = local_expert_ffn(incoming_tokens)                # local compute only

# Combine: send results back to the token's *origin* device, and
# accumulate (weighted by gate value) if top-k selected multiple experts.
outgoing = all_to_all(expert_outputs, routed_by=origin_device, comm=ep_group)
output = weighted_sum(outgoing, gates)
```

## 3. Why this is a genuinely different systems problem

Every other collective operation discussed in `001` — DP's gradient all-reduce, TP's activation
all-reduce, PP's point-to-point handoff — has a communication pattern and volume that is **known at
compile time, independent of the actual data**. An all-reduce always moves the same number of bytes
between the same set of ranks regardless of what the input values are. This matters enormously for
systems engineering: it means communication can be scheduled, overlapped, and capacity-planned
deterministically.

Expert-parallel all-to-all breaks this assumption completely. **Which device sends how many tokens
to which other device is a function of the router's output on this specific batch of tokens for this
specific step** — i.e., it depends on the data and, for a still-training model, on weights that are
changing every step. Three concrete consequences follow, and each is a distinct engineering problem:

- **Communication volume is not fixed per device, only in aggregate.** Total volume across the whole
all-to-all scales predictably with `tokens × top_k × hidden_dim` (this is why
`..\OpenSource\007_DeepSeek_V3.md` notes that going from 160 to 256 routed experts, while keeping
top-k roughly constant, does not blow up communication proportionally to expert count — total
dispatched-token-volume is governed by token count and top-k, not by how many experts exist to
receive them). But the *per-device* volume — how many tokens land on device `d` specifically —
depends on routing and can vary step to step and device to device.
- **The all-to-all is a synchronization barrier every device must wait on.** Unlike DP's gradient
all-reduce (Section 2.3 of `001`), which can be overlapped with unrelated backward compute on other
layers, the dispatch all-to-all sits directly in the critical path: no device can start its local
expert computation until it has received *all* the tokens routed to it, from *every* other device in
the EP group. If routing is imbalanced (Section 4), some devices finish receiving quickly (few
tokens) and then idle waiting for the slowest device to finish sending/receiving its (larger) share,
before the layer can proceed to the combine step.
- **Buffer sizing must accommodate a data-dependent, a priori unknown count.** Standard
collective-communication implementations (NCCL's all-to-all primitives) generally want fixed-size
buffers. If the number of tokens actually routed to a given device varies, the system needs either a
fixed **capacity** per expert (with overflow tokens dropped or rerouted — Section 4) or a
variable-size all-to-all (more complex, often slower due to extra metadata exchange to communicate
sizes before the actual data transfer).

This is the core reason expert parallelism is treated as a genuinely separate research and systems
area rather than "TP but for experts": TP and PP's communication patterns are static and can be
reasoned about with a closed-form cost model exactly like the ones in `001`; EP's cannot, because
the pattern itself is a runtime output of the model being trained.

## 4. Load imbalance: the direct systems cost

Everything in Section 3 collapses into a single practical fact: **if the router does not distribute
tokens roughly evenly across experts, the EP all-to-all's barrier-like nature turns that imbalance
directly into idle GPU-time**, which is real, unrecoverable cluster cost — not merely a
quality/loss-curve concern.

Walk through the mechanism concretely. Suppose an EP group has 8 devices, one expert each, and top-1
routing. If token routing were perfectly uniform, each device would receive `tokens/8` tokens,
all-to-all volume is balanced, and every device finishes its local expert compute at roughly the
same time. If instead the router collapses toward a small subset of "popular" experts — a
well-documented pathological failure mode of unconstrained MoE training, discussed algorithmically
in `..\OpenSource\005_Mixtral8x7B.md`'s Section 8 and `..\OpenSource\007_DeepSeek_V3.md`'s
bias-mechanism discussion — one device might receive several times its fair share of tokens while
another receives almost none. The overloaded device's local expert FFN forward/backward pass now
takes proportionally longer; every other device in the EP group, having already finished their
(smaller) local compute, sits idle waiting at the combine step's implicit barrier for the overloaded
device to catch up. **The step time for the entire EP group is determined by its single most-loaded
device**, exactly analogous to how PP's step time is determined by its slowest stage, or how a
straggler node determines a synchronous collective's completion time
(`008_Debugging_Distributed_Training_Failures.md`). Cluster-wide, this shows up as GPUs burning
power and occupying scheduler slots while doing zero useful work — wasted cost that scales with
however many steps the imbalance persists, potentially the entire training run if nothing corrects
it.

**Capacity-based routing and token dropping** (used in the original Switch Transformer, and
available as a fallback in most production MoE implementations) is the systems-side mitigation for
this specific failure mode: assign each expert a fixed **capacity** (typically `capacity_factor ×
tokens_per_batch / num_experts`, with `capacity_factor` slightly above 1.0 to allow some slack), and
if more tokens are routed to an expert than its capacity allows, the excess tokens are either
dropped (their contribution to that layer becomes zero, often implemented via a residual
pass-through instead of a hard zero) or overflow to a designated fallback path. This bounds the
*worst-case* all-to-all volume per device — solving Section 3's buffer-sizing problem by fixing the
buffer size up front — at the cost of a quality hit whenever tokens are actually dropped, and shifts
the pressure back onto the load-balancing objective to keep drop rate low in practice.
`..\OpenSource\005_Mixtral8x7B.md`'s Section 8 confirms Mixtral relies on its differentiable
auxiliary loss alone (soft encouragement, no hard capacity-based dropping in the published recipe) —
worth noting as a design choice with the systems trade-off just described, not a universal MoE
default.

## 5. Connecting to the load-balancing techniques already covered elsewhere

The algorithmic load-balancing mechanisms in `..\OpenSource\005_Mixtral8x7B.md` (differentiable
auxiliary loss, competing with the LM loss's gradient) and `..\OpenSource\007_DeepSeek_V3.md`
(auxiliary-loss-free bias-based control loop, decoupled from any gradient) are not just
quality/loss-shape interventions — read them through the systems lens developed above and they are
directly solving the all-to-all imbalance problem from Section 4. The key systems-relevant point
already made in the DeepSeek-V3 doc, worth restating precisely here: because the bias adjustment
`b_i -= γ` (overloaded) / `b_i += γ` (underloaded) operates as a feedback loop on *observed load*
rather than as a loss term fighting for the same gradient budget as the LM objective, it gives
infrastructure engineers a **tunable, predictable lever** (`γ`, the bias step size) that can be
adjusted directly against observed communication/compute imbalance metrics from the running job,
independent of any language-modeling-quality tuning. This is a meaningfully different operational
posture than an auxiliary-loss weight, which infra engineers cannot safely tune in isolation because
it also changes model quality — a systems fix and a quality knob sharing one number is intrinsically
harder to operate than two decoupled ones.

**Expert Choice routing** (a further alternative worth knowing at staff depth, from Zhou et al.
2022, not covered in the per-model docs) inverts the routing decision's granularity to address the
same imbalance problem from a different angle: rather than each *token* choosing its top-k experts
(which can produce arbitrarily skewed per-expert load, as above), each *expert* chooses its top-k
tokens from the batch, up to a fixed capacity. This makes perfect load balance a structural
guarantee of the routing algorithm itself (every expert processes exactly its capacity, by
construction) rather than something an auxiliary loss or bias term merely encourages — trading away
the property that every token gets a guaranteed minimum number of experts (some tokens may receive
zero experts if they lose out on every relevant expert's top-k token selection) for a hard guarantee
on the systems side.

## 6. Combining EP with TP/PP/DP: a 4D parallel layout

In practice, EP is not used in isolation — it composes with the other three axes into a 4D grid, and
the placement decisions from `001` about which axis tolerates which network tier apply here with one
further wrinkle specific to MoE.

A common layout for a large MoE model (following the shape of DeepSeek-V3's setup,
`..\OpenSource\007_DeepSeek_V3.md`):

- **TP** shards the non-MoE portions of each layer (attention projections, the shared expert if
present) within a node's NVLink domain, same as a dense model.
- **EP** distributes the routed experts of each MoE FFN layer across a (typically larger) group of
devices — potentially spanning *multiple nodes*, because with hundreds of experts (DeepSeek-V3: 256
routed experts) there are simply more experts than fit in one node's GPU count, forcing EP's
all-to-all across the slower inter-node fabric. This is exactly the scenario
`..\OpenSource\007_DeepSeek_V3.md` flags as acute for H800 clusters: H800's NVLink bandwidth is
deliberately reduced relative to H100 for export-control reasons, so **the exact communication
pattern that most needs high intra-cluster bandwidth (cross-node EP all-to-all, given 256 experts
don't fit in one 8-GPU node) is forced onto the exact link that was cut** — the direct motivation
for DeepSeek's custom cross-node all-to-all kernels and for co-designing DualPipe (`001`, Section
4.4) specifically to overlap this all-to-all with pipeline compute rather than serialize it.
- **PP** splits layer depth as usual, with the added subtlety that pipeline stage boundaries and EP
group boundaries both need scheduling attention: a stage's compute time now includes its MoE layers'
all-to-all wait time, which is more variable step-to-step than a dense stage's compute time (Section
4), making pipeline bubble analysis (`001`, Section 4.1) noisier for MoE models than for dense ones.
- **DP** replicates the entire EP+TP+PP grid across data-parallel copies as usual, with gradient
synchronization for the routed-expert weights following the same reduce-scatter/all-reduce machinery
as any other parameter — the router itself, being a small dense layer, is also just DP-synchronized
normally.

The practical throughput implication, worth stating plainly for an interview answer: **MoE models
systematically show lower MFU (`007_Training_Efficiency_Metrics_MFU_And_Utilization.md`) than
comparable dense models at the same activated-parameter count**, precisely because the EP all-to-all
is an additional, data-dependent, barrier-like communication cost with no dense-model equivalent —
DualPipe-style overlap engineering exists specifically to claw back as much of that gap as possible,
not to eliminate it structurally (the underlying data-dependence from Section 3 cannot be engineered
away, only hidden more effectively behind compute).

## 7. A worked communication-volume example, and monitoring EP imbalance directly

Make Section 3's claim that aggregate all-to-all volume depends on `tokens × top_k × hidden_dim`
rather than on total expert count concrete with numbers close to DeepSeek-V3's published
configuration: 256 routed experts, top-k = 8, hidden dimension on the order of a few thousand after
MLA's compression (`..\OpenSource\007_DeepSeek_V3.md`), bf16 dispatch. For a global batch of, say,
4 million tokens per step, the dispatch all-to-all must move `4,000,000 × 8 × hidden_dim × 2 bytes`
in aggregate across the EP group — a fixed quantity determined entirely by token count and top-k,
regardless of whether the layer had 64, 160, or 256 experts to choose from. This is the arithmetic
behind the claim in Section 5: going from V2's 160 routed experts to V3's 256 while keeping top-k
roughly unchanged did not blow up communication proportionally to expert count, because expert count
only determines how finely that fixed aggregate volume gets *subdivided* across destinations, not
how large the aggregate volume itself is. What *does* change with expert count is the **granularity**
of potential imbalance: with more, narrower experts, a router that collapses onto a small subset of
them concentrates a fixed total volume onto fewer receiving devices, which is a real, separate risk
introduced by finer-grained MoE designs and is exactly why auxiliary-loss and bias-based balancing
mechanisms remain necessary even though aggregate volume itself is well-behaved.

**Monitoring EP imbalance directly, as a first-class operational metric.** Section 4 established
that the EP group's step time is set by its single most-loaded device. The direct, low-overhead way
to monitor this in a running job — the same instrumentation `007_Training_Efficiency_Metrics_MFU_And_Utilization.md`'s
diagnostic framework calls for generally — is to log per-expert (not just per-device) token counts
every step or every few steps, and track the ratio of the maximum per-expert load to the mean
per-expert load over a rolling window. A healthy, well-balanced router keeps this ratio close to 1;
a ratio that drifts upward over training, or that spikes sharply at specific points (e.g., right
after a change in data mixture, or early in training before the balancing mechanism has had time to
act), is a direct, quantitative, and *actionable* signal — actionable in the specific sense that,
under a bias-based mechanism (Section 5), the response is to check whether `γ` is large enough to
correct the observed drift within an acceptable number of steps, and under an auxiliary-loss
mechanism, whether the loss weight needs revisiting, understanding that any such revisit is also a
quality-affecting change and not a free systems-only knob.

## 8. Overlap strategies in practice

Given that the all-to-all cannot be avoided, the practical engineering lever is minimizing its
*exposed* (non-overlapped) cost:

- **Compute/communication overlap within a layer:** split the incoming token batch into sub-chunks,
and pipeline dispatch(chunk 1) → expert_compute(chunk 1) concurrently with dispatch(chunk 2), rather
than waiting for the *entire* dispatch all-to-all to finish before starting any expert compute.
GShard (Lepikhin et al. 2020) and most production MoE frameworks implement some version of this
micro-batched overlap inside a single MoE layer.
- **Cross-layer/cross-stage overlap:** DualPipe's contribution (`..\OpenSource\007_DeepSeek_V3.md`)
is overlapping one pipeline stage's all-to-all communication with a *different* stage's or
micro-batch's compute, rather than only overlapping within a single MoE layer's own forward pass — a
strictly harder scheduling problem because it requires coordinating the pipeline schedule and the EP
communication schedule jointly, rather than treating pipeline scheduling (`001`, Section 4) as
independent of MoE-specific communication.
- **Low-level kernel tuning:** because the all-to-all's per-device volume is variable and the
operation sits in the critical path, DeepSeek describes PTX-level-tuned custom communication kernels
rather than relying on off-the-shelf NCCL all-to-all — a sign of how much throughput is left on the
table by generic collective implementations that aren't aware of the specific (skewed, MoE-shaped)
traffic pattern they're serving.

## 9. Summary: the one-sentence systems distinction

If asked to state, in one breath, why expert parallelism is not "just another parallelism axis"
alongside DP/TP/PP: **DP, TP, and PP all move a fixed, statically-known volume of data between a
fixed, statically-known set of ranks, so their communication cost can be modeled and scheduled at
compile time; expert parallelism's all-to-all volume and destination pattern are outputs of the
model's own (changing, data-dependent) routing decisions, which means load imbalance is not a tuning
inconvenience but a direct, unbounded source of wasted cluster-wide compute time, and the entire
apparatus of auxiliary losses, bias-based control loops, capacity factors, and custom overlap-aware
communication kernels exists to bound that data-dependence back down to something closer to
DP/TP/PP's predictability.**
