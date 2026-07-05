# Debugging A Loss Spike Mid-Training

## Step -1: Why "40% Through" Matters to the Answer

The scenario deliberately specifies a run that's 40% complete, and that detail should change the answer's tenor, not just its framing. Early in a run (say, 5% through), the sunk cost is small enough that a conservative "just roll back further and re-verify carefully" response costs comparatively little; late in a run (80%+), the sunk cost is large enough that the cost of over-investigating (versus the cost of a wrong resume decision propagating undetected to a near-finished model) tips further toward caution as well, but for a different reason — there's very little runway left to recover from a bad decision made now. At 40%, the run is past the point where restarting is remotely sane, but with enough runway remaining that getting the resume decision wrong and discovering it 20% later is a genuinely expensive mistake, not a minor one. This is exactly why the resume-vs-investigate framework in Step 4 argues for spending real time on confidence before resuming, at this specific point in a run's lifecycle — earlier or much later than this, the calculus shifts, and a strong answer should be able to say so explicitly if asked.

## The Scenario

"You're 40% through a large pretraining run — say, a 70B-parameter model, several trillion tokens in, weeks of wall-clock time invested. The loss curve, which had been smoothly decreasing, suddenly spikes upward and does not recover over the following few hundred steps. Walk me through your diagnosis and response."

This is a classic staff-level incident-response question: it rewards a systematic diagnostic tree over guessing, and it specifically probes whether you understand that "roll back to the last checkpoint and hope" is not a diagnosis, it's a stopgap that buys time to actually diagnose. I'll build the diagnostic tree top-down (fastest, cheapest checks first), then cover the mitigation playbook and the resume-vs-investigate decision.

## Step 1: First, Characterize the Spike Before Doing Anything Else

Before branching into hypotheses, pull the actual telemetry and characterize the spike precisely, because the *shape* of the spike is itself diagnostic:

- **Is it instantaneous (one step, then gone) or sustained?** A single-step spike that immediately recovers is usually a bad batch or a transient numerical event; a sustained spike that plateaus at a higher loss and doesn't recover on its own is much more consistent with the optimizer having been knocked into a genuinely worse region of parameter space, or the model having diverged/collapsed in some way that gradient descent alone won't repair.
- **Is it global or localized to a subset of parallelism ranks?** If you have per-rank or per-data-parallel-shard loss logging (you should — this is one of the highest-value low-cost monitoring investments for exactly this failure mode), check whether the spike is visible uniformly across all ranks or concentrated on a specific rank/node. A spike visible on all ranks simultaneously points toward a global cause (a specific global batch, a scheduler/LR event, a synchronized numerical issue); a spike isolated to one or a few ranks points toward a hardware or data-shard-specific cause on that rank.
- **What else moved at the same step?** Check gradient norm (pre- and post-clipping), the learning-rate schedule value, any recent checkpoint-resume event, and — critically — whether this coincides with a data-shard or dataloader-epoch boundary. Correlating the spike's exact step number against every other time-series you're logging is the single highest-leverage five minutes of the whole investigation, and it's the step most likely to be skipped by someone eager to jump straight to a hypothesis.

## Step 1b: A Quick-Reference Signature Table

Before walking each branch in depth, it's useful to have the distinguishing signatures side by side, since this table is effectively what Step 1's characterization work is trying to populate:

| Signature | Points toward | Points away from |
|---|---|---|
| Spike concentrated on one or a few examples within a batch | Data issue (Branch A) | Numerical, optimizer |
| Gradient norm spikes one step *before* the loss spike | Numerical issue (Branch B) | Pure data issue |
| Loss-scaler scale-down event at or near the spike | Numerical issue (Branch B) | Optimizer, hardware |
| LR value at the spike step doesn't match the intended schedule | Optimizer/LR issue (Branch C) | Data, numerical |
| Spike localized to one or a small number of ranks | Hardware issue (Branch D) | Data, numerical, optimizer (all of which tend to be global) |
| Spike immediately follows a checkpoint-resume event | Optimizer-state corruption (Branch C) or resize-specific issue (`011_Interview_Questions_Part1.md`, Q20) | Ordinary data/numerical causes |
| Broad, uniform increase across the whole batch, not a few extreme examples | Optimizer or numerical (Branch B/C) | Data issue (Branch A) |

This table is not a substitute for actually confirming a branch with direct evidence (Step 2's per-branch confirmation methods) — it's a triage tool for deciding which branch to investigate first, given that investigating all four in parallel with divided attention is less efficient than following the strongest available signature to its specific confirming check.

## Step 2: The Diagnostic Tree

With the spike characterized, walk the tree in order of how cheap each hypothesis is to check and how common each cause actually is in practice — data issues and numerical issues are, empirically, far more common causes of unrecovered mid-run loss spikes than genuine optimizer pathology, so check those first.

### Step 1c: A Quick FAQ on Spike Characterization

- **What if the spike is instantaneous and self-recovers within a few steps — do you still need to investigate?** Yes, briefly — even a self-recovering spike is worth a quick check against the branches below, because a self-recovering *symptom* can still be an early, mild instance of a cause (a marginal numerical event, a borderline-bad batch) that will recur and worsen later if left uninvestigated.
- **What if per-rank logging shows the spike on literally every rank, with identical magnitude?** This is actually a stronger, not weaker, signal toward a global cause (a specific global batch, or an LR/schedule event) — perfectly uniform magnitude across ranks is unusual for a hardware-driven cause, which more often varies at least somewhat by rank even when broadly distributed.
- **How much telemetry history do you need before the spike to characterize it properly?** Enough to establish what "normal" gradient-norm and loss variance looked like in the recent past — a spike's z-score (as in `011_Interview_Questions_Part1.md`, Q13's guard implementation) is only meaningful relative to a genuine recent baseline, not an arbitrary fixed threshold.

## Branch A: Data Issue — A Bad Shard or Corrupted Batch

**What to check.** Pull the exact global batch (or batches, if the spike spans several steps) that was being trained on at the spike step, from the versioned, shard-indexed training corpus (this is exactly why data versioning, discussed in `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`, Step 7, matters operationally, not just for compliance — without a way to reconstruct exactly which documents were in a given global batch, this branch of the investigation is impossible). Look for:

- **Corrupted or malformed data** — a shard with a tokenization bug, a truncated file, or (concretely observed across the field) documents containing anomalous repeated tokens, degenerate byte sequences, or encoding errors that survived the cleaning pipeline (`002_...`, Step 2) and produce an extreme, out-of-distribution loss on a specific example.
- **A single pathological long document** — an unusually long sequence (near or exceeding the packing/context boundary) can produce anomalous attention or loss behavior if sequence packing/truncation logic has an edge-case bug.
- **Duplicate or near-duplicate content within the batch** — if a dedup pipeline bug let a large cluster of near-identical documents into one batch, gradient signal for that batch can become unusually sharp/high-magnitude in a way that a well-deduplicated corpus would never produce.
- **Distributional shift at a shard/epoch boundary** — if the data pipeline reads shards in a fixed, non-shuffled order (a genuine pipeline bug, not a hypothetical) and shards happen to be sorted by source or domain, crossing from one domain-heavy region of shards to another can look exactly like a loss spike caused by "the model suddenly seeing very different data," which is a data-pipeline sequencing bug masquerading as a training instability.

**How to confirm.** Reconstruct the exact batch, decode it back to text, and manually inspect it. If per-example loss logging is available (log loss per micro-batch or, ideally, per example within a batch, at least during suspect windows), check whether the spike is driven by one or a handful of extreme-loss examples versus a broad, uniform increase across the whole batch — a small number of extreme examples strongly implicates a data issue; a broad uniform increase points away from Branch A and toward Branch B or C.

### Step 2c: A Quick FAQ on the Branch Ordering

- **Why check data and numerical issues before optimizer and hardware?** Because they're empirically the more common causes of this specific failure signature, and they're also the cheapest to check — cheap-and-common should always be checked before expensive-and-rarer, all else equal.
- **What if the run recently had a config change (a resize, a resumed checkpoint) — does that change the ordering?** Yes — a recent, known configuration change is a strong prior that should reorder the investigation toward the branch that change most plausibly affects, exactly as argued in `011_Interview_Questions_Part1.md`, Q20.
- **Is it ever correct to check two branches in parallel rather than strictly sequentially?** Yes, if there are two engineers available and the checks don't share a bottleneck resource (e.g., one engineer decodes the suspect batch while another checks loss-scaler telemetry) — the sequential ordering matters most when investigator time is the scarce resource, not as a rigid rule regardless of available parallelism.

## Branch B: Numerical Issue — Precision Overflow / Underflow

**What to check.** This is the branch most directly informed by `..\03_Distributed_Training_And_Infrastructure\004_Mixed_Precision_Training_And_Numerical_Stability.md` — the mechanics of loss scaling, BF16/FP16/FP8 dynamic range, and where overflow/underflow actually bites live there; here the question is how to recognize this failure mode from the outside. Signals:

- **Gradient norm behavior immediately before the spike.** A gradient-norm spike that *precedes* the loss spike by one or a few steps is a strong signal of a numerical event — an activation or gradient overflowing its representable range (common in FP16 given its narrow dynamic range, less common but not impossible in BF16 given its wider exponent range but reduced mantissa, and a genuinely live risk in FP8 training given FP8's much narrower dynamic range, which is exactly why DeepSeek-V3's FP8 recipe uses fine-grained tile/block-wise scaling factors rather than a single per-tensor scale, see `..\..\OpenSource\007_DeepSeek_V3.md`, Section 3). If loss-scale telemetry is available (dynamic loss scalers typically log scale-factor reduction events), check whether the scaler reduced its scale around the spike — a scale-down event is direct evidence the training software itself detected an overflow.
- **NaN/Inf detection in activations or gradients.** Any framework-level NaN/Inf guard firing (even if it didn't halt training) is close to a smoking gun; if such guards aren't currently instrumented, that's itself a finding — "we had no NaN/Inf detection in the training loop" is a real gap this incident should close regardless of what the ultimate root cause turns out to be.
- **Layer-level or module-level anomalies.** If per-layer gradient-norm logging exists (a worthwhile investment specifically to make this branch diagnosable), check whether the spike is concentrated in a specific layer or module — a specific attention layer or the final unembedding layer producing anomalously large gradients is a classic numerical-instability signature, often tied to a specific extreme activation value propagating through a normalization layer in a way that amplifies rather than dampens it.

**How to confirm.** If infrastructure allows, replay the exact suspect batch in FP32 (or at least with loss scaling disabled/very conservative) against the pre-spike checkpoint and compare the resulting loss and gradient norms against the mixed-precision run's actual behavior at that step. A large discrepancy confirms a precision-induced artifact; a matching result rules Branch B out and pushes the investigation toward Branch A or C.

### Step 2d: A Quick FAQ on Optimizer/LR Issues

- **How common is a checkpoint-resume bug relative to the other optimizer-branch causes?** Common enough to check first specifically when a resize or resume happened recently — it's a narrow, specific, and easy-to-verify hypothesis (compare actual post-resume optimizer-state values against expected ones) relative to a genuinely subtle LR-schedule bug.
- **What's the cheapest way to verify the LR schedule is behaving as configured?** Log the actual LR value applied at every step and diff it against what the schedule config says it should be at that step — this is a five-minute check that rules out an entire class of bugs immediately.
- **Does gradient clipping being present in the config guarantee it's actually functioning?** No — verify it's actually bounding the effective gradient norm at the step in question, not just that the config flag is set; a clipping implementation bug that silently no-ops is a real, previously-observed failure mode.

## Branch C: Optimizer / Learning-Rate Issue

**What to check.** This branch is the right one when Branches A and B come back clean — no anomalous batch, no numerical overflow signature — and the spike is broad and uniform rather than driven by a handful of extreme examples. Candidate causes:

- **LR schedule discontinuity** — a bug or misconfiguration in the learning-rate schedule (an incorrect warmup/decay boundary, a checkpoint-resume that reset the schedule's step counter incorrectly, or a manually-triggered LR change mid-run) producing a genuine LR spike. This is exactly the kind of bug that's invisible unless the LR value itself is logged and checked at the suspect step — never assume the schedule is doing what the config says; log and verify the *actual* value used at every step.
- **Gradient-clipping threshold too loose (or clipping silently disabled)** — if gradient clipping is meant to bound the effective step size taken in response to any single batch's gradient and it isn't actually functioning (a config bug, or a threshold set too high to matter), an unusually sharp gradient from an otherwise-legitimate batch can push the optimizer state into a bad region that subsequent steps don't recover from cleanly, particularly for Adam-family optimizers where the second-moment estimate can be transiently distorted by one extreme step and takes many subsequent steps to "forget."
- **Optimizer state corruption from a bad resume** — if this training run has been checkpointed and resumed (nearly certain at this duration), verify that the resumed optimizer state (Adam's first/second moment buffers) was loaded correctly and matches the expected shard/rank mapping — a checkpoint-resume bug that silently mismatches optimizer-state shards across a changed parallelism configuration (e.g., resuming with a different data-parallel or ZeRO-sharding degree than the run that produced the checkpoint) is a real, previously-observed failure mode across the field that can produce exactly this kind of unrecovered instability appearing some number of steps after a resume event.

**How to confirm.** Check the LR and gradient-norm (pre-/post-clip) time series explicitly against the spike step; check the training run's checkpoint-resume history for a recent resume event that could correlate; if a resume event is implicated, verify optimizer-state shard integrity directly (checksum or spot-check specific parameter tensors' Adam moment values against what they should be, if a redundant reference is available).

### Branch D: Hardware Issue

**What to check.** This branch overlaps heavily with `..\03_Distributed_Training_And_Infrastructure\008_Debugging_Distributed_Training_Failures.md` and `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md` (this folder) — a silently-corrupting GPU (a rare but real failure mode: a specific GPU producing subtly wrong results without crashing, sometimes called "silent data corruption" in large-fleet hardware-reliability literature), a bit-flip in an all-reduce, or a NIC/interconnect issue corrupting a gradient during cross-node communication can all produce a loss spike that looks, from the training-loop's perspective, identical to a legitimate but unusual gradient. This branch is the hardest to confirm directly and is usually investigated by elimination — if Branches A, B, and C are all ruled out and the spike is reproducibly tied to a specific rank or node (per Step 1's per-rank characterization), suspect hardware and route to the health-check/diagnostic tooling covered in the cross-referenced files, which includes running standalone GPU diagnostics and communication-microbenchmarks against the suspect node in isolation, outside the training job.

## Step 2b: A Pre-Incident Checklist — What Should Already Exist Before This Happens

- Per-rank loss and gradient-norm logging, exported continuously to a dashboard, not just queryable on request.
- Loss-scaler scale-factor telemetry, if training in FP16/FP8, logged at every step.
- Per-layer gradient-norm logging, at least for a representative subset of layers, to localize numerical anomalies quickly.
- A versioned, reconstructable corpus manifest (per `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`, Step 7b) so any suspect batch can actually be decoded back to text.
- Per-example (or at least per-micro-batch) loss logging, at minimum toggleable on demand for a suspect window, to distinguish a few extreme examples from a broad uniform increase.
- A tested, fast checkpoint-restore procedure, so that "roll back and resume" is a matter of minutes, not hours, once a decision is made.
- A documented, pre-agreed decision framework for resume-vs-investigate-further, so the response to an actual incident is "follow the established process" rather than an ad hoc judgment call made under pressure for the first time.

## Step 3: The Mitigation Playbook

While diagnosis is ongoing (it can take hours), the training run itself needs an immediate decision, because every additional step trained on a corrupted state is wasted compute at this cluster scale:

1. **Pause the run, don't let it keep training through the spike hoping it self-corrects.** A spike that hasn't recovered after a few hundred steps is very unlikely to self-correct through further training on whatever state it's currently in — continuing burns GPU-hours without new information and, in the optimizer-corruption case, can compound the problem as Adam's moment estimates continue adapting to a bad trajectory.
2. **Identify the last known-good checkpoint** — the most recent checkpoint saved *before* the spike, ideally with enough logging resolution to know precisely how many steps/tokens separate that checkpoint from the spike (frequent checkpointing is a direct hedge against exactly this scenario; if checkpoint cadence is coarse, e.g., every several hours, this incident is itself the argument for tightening cadence going forward, at the acknowledged cost of more checkpointing overhead per `..\..\OpenSource\003_Llama3.md`'s discussion of checkpointing-cost tradeoffs at scale).
3. **Skip vs. roll back, and how far.** If Branch A (bad batch/shard) is confirmed or strongly suspected, the fix can often be narrower than a full rollback: roll back to the last checkpoint, patch the data pipeline to exclude/repair the offending shard, and resume from that same checkpoint with the corrected data stream — this loses only the steps since the last checkpoint, not the diagnostic investigation time. If Branch B or C is confirmed (a numerical or optimizer-state issue), the fix needs to address the underlying config (tighten loss-scaling behavior, lower the gradient-clipping threshold, fix the LR-schedule/resume bug) *before* resuming, because resuming from the same checkpoint with the same faulty config will likely just reproduce the spike at the same relative point.
4. **Adjust gradient clipping or LR defensively even if the root cause isn't fully nailed down yet.** A common, reasonable mitigation when time pressure doesn't allow full root-cause certainty before resuming: tighten the gradient-clipping norm threshold and/or apply a short LR warmup-style ramp back up from a reduced value for a few hundred to a few thousand steps after resuming, as a defensive measure that reduces the blast radius of whatever caused the spike even without full certainty about the mechanism — this is a legitimate engineering judgment call under uncertainty, not a substitute for continuing the investigation in parallel.
5. **Never resume into full-speed training on an unexamined hypothesis.** The single worst version of this response is: see the spike, assume it's "probably just a bad batch," resume without actually inspecting the batch, and discover days later (many trillion tokens and considerable cost later) that the real cause was a systemic numerical or hardware issue that has now silently recurred multiple times, each instance individually written off as "probably just a bad batch."

## Step 3b: The Mitigation Playbook, Tabulated by Confirmed Branch

| Confirmed branch | Immediate action | Before resuming | Risk if skipped |
|---|---|---|---|
| A: bad batch/shard | Roll back to last checkpoint before the spike | Patch data pipeline to exclude the offending shard | Same corruption re-enters training on next epoch pass |
| B: numerical overflow/underflow | Roll back; disable/adjust loss scaling for the suspect region | Validate fix by replaying the suspect batch under corrected precision settings | Silent, harder-to-detect recurrence later in training |
| C: optimizer/LR issue | Roll back; freeze further checkpoint-resumes until schedule/state is verified | Fix LR-schedule bug or verify optimizer-state resharding correctness | Optimizer state continues adapting to a corrupted trajectory |
| D: hardware | Drain suspect node/rank; roll back | Run isolated hardware diagnostics before re-admitting the node | Corruption recurs, possibly less visibly (silent data corruption) |

## Step 4: Deciding Resume-vs-Investigate-Further

This is the actual judgment call a staff engineer owns, and it's a cost/risk tradeoff, not a pure engineering question:

- **Resume quickly if:** Branch A is confirmed with a clear, narrow root cause (one identifiable bad shard, now excluded) and there's no evidence of numerical or hardware involvement (gradient-norm and loss-scale telemetry are clean; no NaN/Inf guard fired; the spike was cleanly localized to a small number of extreme-loss examples). In this case, the marginal value of further investigation is low relative to the cost of idling an expensive training cluster.
- **Hold and investigate further if:** any hardware-involvement signal is present (a rank-localized spike, a suspicious interconnect metric at the spike step), because resuming without confirming hardware health risks re-encountering the same corruption — possibly less visibly the second time, silently degrading model quality rather than spiking loss obviously — which is a strictly worse outcome than losing additional wall-clock time now to run isolated diagnostics on the suspect node/rank.
- **Hold and investigate further if:** the spike is unexplained after working through Branches A-C and doesn't cleanly localize to a rank (ruling out an easy hardware story either). An unexplained loss spike on a run this expensive is exactly the situation where "we don't know why, let's just resume and see" is the highest-variance, worst-expected-value choice available — a repeat, unexplained spike a few hundred steps later, after having already resumed once, is a substantially worse position (both in sunk cost and in diagnostic difficulty, since now there are two candidate spike events with possibly different causes to disentangle) than spending a few more hours confirming a specific mechanism before resuming.
- **A useful heuristic for calibrating how much investigation time is proportionate:** compare the GPU-hour cost of the investigation delay against the GPU-hour cost already sunk into the run and the GPU-hour cost of potentially having to discard and re-run a large fraction of the post-spike training if the eventual root cause turns out to require it. At 40% through a run this large, the sunk cost and the downstream risk both argue strongly for spending hours, not minutes, getting confident about the mechanism before resuming — this is exactly the point in a run where "move fast" instincts are most likely to produce the worst outcome.

## Step 4b: A Worked Narrative — Walking Through One Concrete Version of This Incident

To make the abstract diagnostic tree concrete, here is one plausible, fully worked-through incident, in the order it would actually unfold:

- **T+0:** loss jumps from a stable ~1.85 to ~4.2 in a single step, and does not recover over the next 300 steps, plateauing around 3.6.
- **T+2 minutes:** on-call engineer pulls per-rank loss; the spike is uniform across all data-parallel ranks, ruling out a single-node hardware cause immediately and pointing toward a global cause — either the specific global batch, or a scheduler/numerical event affecting every rank identically.
- **T+5 minutes:** gradient-norm plot shows a sharp spike *at the same step* as the loss spike, not one step before it — this timing (concurrent, not leading) is more consistent with a data-driven cause (an unusual batch producing genuinely large gradients) than a precision-overflow cause (which more often shows a telltale scale-factor reduction slightly before or at the spike).
- **T+10 minutes:** loss-scaler telemetry is checked and shows no scale-down event around the spike — this weakens the numerical-overflow hypothesis (Branch B) further, without fully ruling it out.
- **T+20 minutes:** the exact global batch at the spike step is reconstructed from the versioned corpus manifest and decoded back to text; one micro-batch shard contains a cluster of several hundred near-identical documents — a deduplication gap that let a large cluster of syndicated-copy news articles through unflagged.
- **T+25 minutes:** per-example loss logging (if available) confirms the spike is driven by a small subset of examples within the batch (the near-duplicate cluster), not a broad uniform increase — this is the confirming signature for Branch A (data issue) described in Step 2 above.
- **T+30 minutes:** decision made to pause the run, roll back to the last checkpoint (18 minutes of training before the spike, given a tight checkpoint cadence), patch the data pipeline to exclude the offending shard, and resume.
- **T+45 minutes:** training resumes from the rolled-back checkpoint with the corrected data stream; loss returns to its pre-incident trajectory within the first few hundred steps, confirming the fix.
- **T+1 day:** a follow-up ticket is filed against the deduplication pipeline itself (per `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`, Step 3) to understand why this specific near-duplicate cluster wasn't caught, since the whole point of finding one instance of a systemic gap is fixing the gap, not just the one occurrence.

This narrative is deliberately the *fast, clean* version of the incident — a case where the diagnostic tree resolves within 30 minutes because the telemetry was already in place and the versioned corpus manifest made batch reconstruction trivial. The value of walking through it explicitly is showing exactly how each piece of pre-built monitoring infrastructure (per-rank loss, gradient-norm and loss-scale telemetry, per-example loss logging, a versioned/reconstructable corpus) collapses what would otherwise be a multi-hour, multi-person investigation into a half-hour, single-engineer one.

## Step 4c: What This Incident Looks Like Without the Right Monitoring in Place

Contrast the above against the same incident in an environment missing some of that instrumentation, since this is the realistic failure case an interviewer may probe for:

- Without per-rank loss telemetry, the first 10-15 minutes are spent simply confirming whether this is a global or localized event, delaying every subsequent branch of the investigation.
- Without loss-scaler telemetry, ruling out Branch B (numerical overflow) requires a much more expensive manual replay of the suspect batch in higher precision, adding potentially an hour or more before that branch can be confidently excluded.
- Without a versioned, reconstructable corpus manifest, identifying the exact documents in the offending batch may not be possible at all — the investigation stalls at "we think it's probably a bad batch" without ever being able to confirm which batch or what was in it, which is a materially worse position than a confirmed, narrow root cause.
- Without per-example loss logging, distinguishing "a few extreme examples" from "a broad uniform increase" requires either an expensive offline replay with finer-grained logging enabled, or an educated guess — and an educated guess is exactly the posture Step 4's resume-vs-investigate framework argues against on a run this expensive.

The gap between these two versions of the same incident is the single strongest argument for treating monitoring infrastructure as a Phase 5 deliverable of the overall training-run plan (`008_Planning_A_Model_Training_Run_End_To_End.md`), built before the run starts, rather than something assembled reactively after the first incident forces the issue.

## Step 4d: A Postmortem Template Worth Having Ready

Once resolved, every incident of this kind should produce a short, structured postmortem — not as bureaucratic overhead, but because the pattern of *which* branch turns out to be the cause, across many incidents over a training program's lifetime, is itself valuable signal about where the pipeline/infrastructure has systemic weaknesses. A minimal, reusable template:

- **Detection:** how was the spike first noticed, and how long after it occurred (this number, tracked over time, is a direct measure of whether monitoring investment is improving detection latency).
- **Root cause:** which branch (data, numerical, optimizer, hardware), and the specific confirming evidence.
- **Blast radius:** how many steps/tokens were lost to the rollback, and what was the GPU-hour cost of the incident (both the lost training time and the investigation time).
- **Fix applied:** the specific mitigation (patch, config change, hardware replacement) and whether it was validated before resuming.
- **Systemic follow-up:** what standing gap (in monitoring, in the data pipeline, in checkpoint cadence) does this incident reveal, and what's the concrete ticket/owner to close it.

## Step 4d2: A Summary Table of the Whole Response

| Order | Action | Time cost | Purpose |
|---|---|---|---|
| 1 | Characterize the spike (shape, per-rank scope, correlated events) | Minutes | Narrows the diagnostic tree immediately |
| 2 | Check the strongest-signature branch first (per Step 1b's table) | Minutes to ~30 min | Confirms or rules out the leading hypothesis |
| 3 | Pause the run; identify last known-good checkpoint | Immediate | Stops further wasted/corrupted training |
| 4 | Apply confirmed-branch-specific fix (per Step 3b's table) | Varies | Addresses the actual root cause, not just the symptom |
| 5 | Decide resume vs. further investigation using the sunk-cost/risk tradeoff | Hours if warranted | Avoids a worse, repeat incident later in the run |
| 6 | File systemic follow-up ticket | Follow-up | Prevents recurrence of the same class of incident |

## Step 4e: Common Mistakes This Scenario Is Designed to Surface

- Jumping straight to "roll back and resume" without characterizing the spike's shape or scope first.
- Assuming a spike is "probably just a bad batch" without actually decoding and inspecting the batch.
- Treating gradient-norm and loss-scale telemetry as optional instrumentation rather than prerequisites for a fast diagnosis.
- Continuing to train through the spike, hoping it self-corrects, well past the point where that's a reasonable wait-and-see window.
- Resuming from a rolled-back checkpoint without fixing the underlying cause first, and being surprised when the same spike recurs at the same relative point.
- Over-tightening gradient clipping or dropping the learning rate defensively without any diagnostic evidence, as a substitute for actually root-causing the event.
- Treating hardware causes as vanishingly unlikely and never checking per-rank localization, even though this is one of the cheapest checks available.
- Failing to file a systemic follow-up ticket once a root cause is found, so the same class of incident recurs weeks later with no institutional memory of the first occurrence.

## Step 4f: A Quick FAQ on the Resume Decision

- **Is there a simple rule of thumb for how long to investigate before resuming?** Roughly: investigation time should scale with (sunk cost so far) plus (estimated downside if the wrong branch is assumed) minus (cluster idle-cost per hour of further delay) — at 40% through an expensive run, this arithmetic almost always favors hours of investigation over minutes.
- **What if leadership is pushing to resume immediately to avoid idle GPU cost?** This is exactly where a staff engineer needs to make the idle-cost-versus-repeat-incident-cost tradeoff explicit and quantified, rather than either capitulating to schedule pressure or refusing to explain the reasoning — a clearly stated cost comparison is far more persuasive than an appeal to caution alone.
- **Can you resume on a hypothesis without 100% certainty?** Yes — certainty isn't the bar; the bar is having ruled out the hypotheses whose recurrence risk is worst (hardware, unconfirmed numerical issues) and having a specific, falsifiable reason to believe the confirmed branch's fix actually addresses the cause.

## Step 5: What a Complete Answer Signals

## Step 5b: Why This Question Recurs at the Staff Level

This scenario is a favorite for the same reason as its throughput-regression sibling in `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md`: there is no single correct answer to memorize, only a correct *process* to demonstrate, and the specific evidence a candidate reaches for first (per-rank telemetry, gradient-norm history, the corpus manifest) reveals whether they've actually operated infrastructure at this scale or are reasoning about it from general principles alone. The resume-vs-investigate judgment call in Step 4, in particular, is a values question as much as a technical one — it's testing whether a candidate can make and defend a real cost/risk tradeoff under organizational pressure, not just recite a diagnostic checklist.

A strong response to this scenario walks the tree in the order above (data, then numerics, then optimizer, then hardware — cheapest and most common checks first), cites concrete telemetry for each branch rather than describing hypotheses in the abstract, is explicit about what confirms versus merely suggests each hypothesis, and — critically — treats the resume-vs-investigate decision as a real cost/risk tradeoff rather than a reflexive "always roll back and resume immediately" or "always fully root-cause before touching anything" answer. Both extremes are wrong in different regimes; the skill being tested is knowing which regime you're in from the evidence in front of you.
