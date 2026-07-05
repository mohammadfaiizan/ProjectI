# Estimating Training Compute And Cost

## The Scenario

An interviewer says: "We're considering training a 70B-parameter dense model on 15 trillion tokens. Before we commit a cluster to this, give me an estimate of the training compute in FLOPs, how many GPU-hours that translates to, and a rough dollar figure. Then tell me how you'd sanity-check that estimate against a real disclosed number."

This is one of the most common opening questions in a staff research-engineer loop, precisely because it is a compressed test of several things at once: do you know the compute-estimation formula and its provenance, can you reason about hardware utilization honestly rather than quoting a spec-sheet peak number, do you understand how cloud/cluster economics actually work, and — critically — can you cross-check a back-of-envelope number against a real, disclosed figure and explain the gap rather than just presenting a number with false confidence. I'll work through it in that order.

## Step 0b: A Quick FAQ Before Starting the Arithmetic

- **Does the interviewer expect an exact number or a range?** A range, always — presenting a single point estimate with no stated uncertainty is a tell that the assumptions weren't actually examined; a defensible range with named sensitivities (Step 4c) is the stronger answer.
- **Is it acceptable to ask the interviewer for the target model size and token count rather than assuming one?** Yes, and doing so is a good sign — clarifying the exact target before computing anything mirrors how this estimate would actually be requested in a real planning conversation.
- **What's the very first number to write down, before anything else?** `C = 6ND`, stated with N and D pinned to specific values — everything else in this file is downstream of getting this first line exactly right, including which parameter count (total vs. activated) belongs in it.

## Step 1: Clarify What "Compute" Means Here

Before reaching for a formula, it's worth being explicit about what quantity is actually being asked for, because "training compute" is overloaded:

- **Total training FLOPs (C)** — the total number of floating-point operations performed across the entire pretraining run, forward and backward combined. This is what scaling-law papers (Kaplan et al. 2020, Hoffmann et al. 2022) report and what the C≈6ND heuristic estimates.
- **Peak hardware FLOP/s** — a spec-sheet number for the accelerator (e.g., an H100's advertised BF16 tensor-core throughput), which is never actually achieved in practice.
- **Achieved FLOP/s (MFU-adjusted)** — the realistic sustained throughput once you account for communication overhead, pipeline bubbles, activation recomputation, and attention's non-matmul overhead. This is the number that actually determines wall-clock time and dollar cost.

The interview answer needs all three, connected: C tells you the numerator, achieved FLOP/s (derived from peak FLOP/s times a Model FLOPs Utilization, or MFU, fraction) tells you the denominator, and GPU-hours = C / achieved-FLOP/s-per-GPU. Dollar cost is GPU-hours times a price-per-GPU-hour assumption. Each of those three steps carries its own set of assumptions that need to be stated explicitly, not buried.

## Step 2: The C ≈ 6ND Formula — Where It Comes From and What It Actually Counts

The standard heuristic, from Kaplan et al. (2020) and reused throughout Hoffmann et al. (2022, "Chinchilla") and essentially every subsequent scaling-law paper, is:

```
C ≈ 6 * N * D
```

where `N` is the number of (non-embedding, though the correction is usually small at scale) model parameters and `D` is the number of training tokens. The derivation is straightforward matmul accounting:

- A forward pass through a dense transformer costs approximately `2 * N` FLOPs per token. This comes from the fact that a matrix-vector multiply of a `[d_in, d_out]` weight matrix against a `d_in`-dimensional activation costs `2 * d_in * d_out` FLOPs (one multiply and one add per element of the output, summed over the input dimension) — so summing `2 * d_in * d_out` over every weight matrix in the network gives `2N` for one token's forward pass, since `N` is defined as exactly the sum of all those `d_in * d_out` terms.
- The backward pass costs roughly **twice** the forward pass in FLOPs — one backward matmul to compute the gradient with respect to the input activations (needed to propagate the error further back) and one to compute the gradient with respect to the weights, each costing about the same as the forward matmul. So backward ≈ `4N` per token.
- Forward + backward ≈ `2N + 4N = 6N` FLOPs per token, and over `D` tokens, `C ≈ 6ND`.

Important caveats to state out loud in an interview, because glossing over them is a tell that you're pattern-matching a formula rather than understanding it:

1. **This ignores attention's quadratic term.** The `6ND` heuristic treats the transformer as if every FLOP scales linearly with parameter count and token count, which is true for all the linear projections (QKVO projections, FFN up/down projections) but not for the attention score computation and softmax, which cost `O(sequence_length^2 * d_model)` per layer. For most pretraining context lengths relative to `d_model` (e.g., 4K-8K context against a 70B model's ~8K hidden dimension) this quadratic term is a modest single-digit percentage of total FLOPs and is conventionally dropped from the headline estimate. It stops being negligible at very long context (100K+ tokens), where it should be added back in explicitly as its own term.
2. **`N` should be the count of parameters actually participating in matmuls per token** — for a dense model this is just total parameters (minus a small embedding correction that's usually ignored); for a Mixture-of-Experts model, `N` in this formula should be the **activated** parameter count per token, not total parameters, since routed experts that aren't selected for a given token don't perform any FLOPs for that token. This distinction becomes essential the moment you're estimating compute for something like DeepSeek-V3 (671B total / 37B activated) — using 671B in the formula would overstate compute by roughly 18x relative to what training actually costs, because 94.5% of parameters are inactive for any given token.
3. **This is training compute, not inference compute.** Inference forward-pass-only compute per token is `~2N`, roughly a third of the training-time per-token cost, and — for a model served at scale — the aggregate inference compute over the model's deployment lifetime can dwarf training compute. This distinction is the entire argument behind Llama 3's deliberate 8B-model overtraining strategy (see `..\..\OpenSource\003_Llama3.md`, Section 5) and is worth flagging even when the question is narrowly about training cost, because an interviewer who asks "why does this matter" is checking whether you connect training-compute estimation to the broader cost-of-ownership picture.

## Step 3: Work the 70B / 15T-Token Example End to End

**Compute.**

```
N = 70e9
D = 15e12
C = 6 * N * D = 6 * 70e9 * 15e12 = 6.3e24 FLOPs
```

That's 6.3 × 10^24 FLOPs — for scale, this sits about an order of magnitude above GPT-3's disclosed ~3.14×10^23 FLOPs (see `..\..\GPT\003_GPT3.md`, Section 3) and roughly comparable to Llama 3.1 405B's disclosed ~3.8×10^25 FLOPs divided down for the smaller parameter count and lower token count relative to that specific run — a plausible, mid-frontier-scale training run for 2024-2025.

**Hardware throughput and MFU.**

This is the step where interview candidates most often go wrong, by quoting a GPU's peak spec-sheet FLOP/s and dividing straight through, which overstates achievable throughput by 2-3x. An H100 SXM's advertised dense BF16 tensor-core peak is approximately 989 TFLOPS (~1.98 PFLOPS with structured sparsity, which is not applicable to a normal dense training workload). In practice, no real training job sustains anywhere near that peak, because:

- Communication (all-reduce for data parallelism, all-to-all or point-to-point for tensor/pipeline parallelism) is not free and does not overlap perfectly with compute.
- Attention and normalization operations are not pure matmul and run at a lower fraction of peak than the GEMMs the peak figure is quoted for.
- Pipeline bubbles, activation recomputation (trading extra forward-pass FLOPs for reduced memory), and imperfect batch-size/sequence-length packing all eat into effective throughput.

**Model FLOPs Utilization (MFU)** — the fraction of theoretical peak FLOP/s actually converted into useful model FLOPs — is the standard metric for this. Well-engineered large dense-model training runs on H100 clusters with good interconnect (NVLink within node, high-bandwidth InfiniBand/RoCE across nodes) typically report MFU in the **35-50%** range; MoE models with expert-parallel all-to-all communication often report lower MFU (20-35%) because of the added communication pattern, unless it's been carefully co-designed away (DeepSeek-V3's DualPipe schedule is precisely an engineering response to this — see `..\..\OpenSource\007_DeepSeek_V3.md`, Section 4). For this dense 70B example, a reasonable planning assumption is **40% MFU**, stated explicitly as an assumption rather than a fact:

```
peak_flops_per_gpu = 989e12          # H100 SXM BF16 dense peak
mfu = 0.40
achieved_flops_per_gpu = 989e12 * 0.40 = 395.6e12   # FLOP/s per GPU
```

**GPU-hours.**

```
total_gpu_seconds = C / achieved_flops_per_gpu
                   = 6.3e24 / 395.6e12
                   ≈ 1.593e10 seconds

gpu_hours = 1.593e10 / 3600 ≈ 4.42e6 GPU-hours
```

So roughly **4.4 million H100-hours**. Sanity-check this against wall-clock time on a plausible cluster size: on a 16,000-GPU cluster (Llama 3.1 405B's scale), 4.42e6 GPU-hours / 16,000 GPUs = 276 hours ≈ **11.5 days**. On a more modest 2,000-GPU cluster, that becomes 2,210 hours ≈ **92 days**, about three months — both numbers are plausible durations for a real frontier-adjacent training run, which is itself a useful sanity check: if your GPU-hours estimate divided by a realistic cluster size produces either "40 minutes" or "40 years," you've made an arithmetic error, not found a surprising result.

**Dollar cost.**

Cloud GPU pricing varies enormously by commitment structure — on-demand hourly pricing for H100s from major cloud providers has historically run $4-8/GPU-hour, but that on-demand rate is not what large training runs actually pay. Reserved-capacity, multi-year commitments, or self-owned/colocated clusters (amortizing capex + power + networking + operations over the hardware's useful life) land much lower, commonly cited in the **$1.5-2.5/GPU-hour** range as an all-in effective rate for frontier labs' actual training economics — and this is also the exact assumption DeepSeek used in its own cost disclosure. Using **$2/GPU-hour** as a defensible planning number:

```
cost = 4.42e6 GPU-hours * $2/GPU-hour ≈ $8.84M
```

So the headline answer: **≈6.3×10^24 FLOPs, ≈4.4 million H100-hours, ≈$8.8M** for the main pretraining run of a 70B-dense model on 15T tokens, under a 40% MFU and $2/GPU-hour assumption. It's worth explicitly stating in an interview that every number after "6.3×10^24 FLOPs" is contingent on the MFU and price assumptions, and that a candidate should be ready to redo the GPU-hours and cost lines instantly if the interviewer challenges either assumption (e.g., "what if MFU is only 25% because this is an early, not-yet-optimized cluster" — GPU-hours roughly scales inversely with MFU, so 25% MFU would push this to ~7.07M GPU-hours and ~$14.1M).

It's also worth noting explicitly what this number excludes: data pipeline compute (CPU/storage-side processing, not GPU-hours), any ablations, ramp-up runs, or failed attempts that preceded the final configuration, post-training (SFT/RLHF/RLVR) compute, and evaluation compute. Every one of these is typically a real, non-trivial additional cost that a "training compute estimate" of this narrow kind does not capture — this caveat is exactly what Section 4 below returns to when cross-checking against DeepSeek-V3's disclosed figure.

## Step 4: Sanity-Checking Against a Real Disclosed Number — DeepSeek-V3

The single best sanity check available in this space is DeepSeek-V3's technical report, because it is unusually transparent: it discloses total parameters (671B), activated parameters per token (37B), training tokens (14.8T), GPU count (2048 H800s), GPU-hours by phase (2.664M pretraining / 119K context extension / 5K post-training), and an assumed $2/GPU-hour price, yielding a headline ≈$5.576M figure (see `..\..\OpenSource\007_DeepSeek_V3.md`, Section 3). Reconstructing this from first principles is exactly the exercise a staff interviewer wants to see, because it's the difference between "I memorized a formula" and "I can independently verify a claim."

**Applying C≈6ND with the correct N.** DeepSeek-V3 is MoE, so `N` must be the **activated** parameter count (37B), not total (671B):

```
N = 37e9
D = 14.8e12
C = 6 * 37e9 * 14.8e12 ≈ 3.285e24 FLOPs
```

**Back-solving achieved throughput from the disclosed GPU-hours.**

```
total_gpu_seconds = 2.664e6 GPU-hours * 3600 = 9.590e9 seconds
achieved_flops_per_gpu = C / total_gpu_seconds = 3.285e24 / 9.590e9 ≈ 3.43e14 FLOP/s ≈ 343 TFLOPS/GPU
```

**Implied MFU.** H800 has the same compute throughput as H100 (the export-control difference is NVLink interconnect bandwidth, not FLOP/s), and V3 trains primarily in FP8. FP8 tensor-core dense peak throughput on this hardware generation is commonly cited around 1979 TFLOPS with structured sparsity and roughly half that, ~990 TFLOPS, for dense (non-sparse) FP8 matmul — this specific figure should be flagged as an approximate, not fully nailed-down number, since NVIDIA's published peak figures are frequently quoted inconsistently across sparse/dense and different documentation. Taking ~990 TFLOPS dense FP8 as the peak:

```
implied_mfu = 343 / 990 ≈ 35%
```

That is a plausible, if unremarkable, MFU for a well-run large training job — meaningfully lower than the >50% MFU narrative that sometimes circulates in casual discussion of DeepSeek-V3's efficiency, and this discrepancy is itself the useful finding, not a failure of the method. A staff-level answer should surface exactly this kind of "the arithmetic roughly closes, but here's where and why it doesn't close perfectly" reasoning rather than either (a) refusing to engage with real numbers or (b) presenting a suspiciously exact match as if the estimation method were more precise than it actually is.

**Reasons the numbers don't close exactly, worth stating explicitly:**

- The peak FP8 dense throughput figure (~990 TFLOPS) is an approximation with real uncertainty; if the true dense peak is closer to 800 or 1100 TFLOPS, implied MFU shifts by several points in either direction — this alone explains most of the "is it 35% or 45%" ambiguity.
- The 6ND heuristic ignores attention's quadratic term and MTP's (Multi-Token Prediction) auxiliary loss compute, both of which add real FLOPs beyond the base next-token-prediction estimate — MTP modules are additional small transformer blocks trained alongside the main model (see `..\..\OpenSource\007_DeepSeek_V3.md`, Section 2), so the true C is somewhat higher than the 3.285e24 estimate, which would push the *true* implied MFU *up* from 35% (more real FLOPs delivered per GPU-hour than the base-formula estimate credits).
- Expert-parallel all-to-all communication (dispatch/combine across 256 routed experts) is exactly the kind of overhead that suppresses MFU below a dense model's typical range, and DeepSeek's DualPipe schedule and custom communication kernels are specifically engineered to claw back MFU that this communication pattern would otherwise cost — so a 35% figure, in a regime where naive MoE training might land at 20-25%, is actually a *strong* result once contextualized against the counterfactual.

**On the dollar-cost debate itself.** The reconstructed $5.576M-equivalent figure (2.664M + 119K + 5K = 2.788M total GPU-hours × $2) is *arithmetically* just GPU-hours × price — there's no independent way to falsify it beyond querying whether $2/GPU-hour and 2048 H800s for the disclosed durations are themselves accurate, which outside parties cannot fully verify. The more important sanity-check question, and the one a staff interviewer is really probing for, is not "does the arithmetic multiply out correctly" (it does, trivially) but "what does this number *not* include" — data curation cost, the cost of prior ablations and discarded runs, cluster capex versus rental-equivalent pricing, and the broader organizational cost of the research program that produced the final configuration. The honest characterization, and the one worth stating unprompted in an interview: **the disclosed figure is a credible estimate of marginal, direct compute for the final successful run — not a claim about the fully-loaded cost of producing the model as an organizational effort**, and treating it as the latter (as much casual industry commentary did) is a category error the disclosure itself does not make but that its reception often did.

## Step 4b: A Second Cross-Check — Llama 3.1 405B

It's worth running the same reconstruction against a second disclosed data point, because a method that only "works" against one example hasn't really been validated. Llama 3.1 405B discloses its headline compute figure directly (~3.8×10^25 FLOPs) rather than requiring a `6ND` reconstruction, and discloses training on up to 16,000 H100s (`..\..\OpenSource\003_Llama3.md`, Section 3-4). Meta does not disclose a GPU-hour or dollar-cost figure with DeepSeek-V3's level of granularity, so the cross-check here runs in the other direction: forward-estimate wall-clock time and see whether it's consistent with what's known about the run's duration.

```
C = 3.8e25 FLOPs
peak_flops_per_gpu (H100 BF16 dense) = 989e12
mfu_assumption = 0.40                      # dense model, well-engineered 4D parallelism
achieved_flops_per_gpu = 989e12 * 0.40 = 395.6e12

total_gpu_seconds = 3.8e25 / 395.6e12 ≈ 9.61e10 seconds
gpu_hours = 9.61e10 / 3600 ≈ 2.67e7 GPU-hours

wall_clock_hours_at_16000_gpus = 2.67e7 / 16000 ≈ 1,668 hours ≈ 69.5 days
```

Roughly ten weeks at full 16,000-GPU scale and 40% MFU. This is broadly consistent with the range of durations reported informally for frontier training runs of this scale in this period (commonly discussed in the multi-month range), which is exactly the kind of "does this pass the smell test" check that should accompany any estimate before it's presented as a finding rather than a guess. If this arithmetic had instead produced "3 hours" or "40 years," that would be the signal to go back and re-examine the MFU or peak-FLOP/s assumption before trusting anything downstream of it — an estimate that survives a wall-clock sanity check against a second, independent disclosed data point is meaningfully more trustworthy than one checked against only a single example, precisely because it rules out the estimate having been tuned (even unconsciously) to fit one specific case.

## Step 4c: Sensitivity Analysis — How Much Does the Answer Actually Move?

A staff-level answer should be ready to immediately re-run the arithmetic under a challenged assumption rather than treating the headline number as fixed. The two assumptions that matter most are MFU and $/GPU-hour, and both enter the final cost linearly (cost is directly proportional to GPU-hours, and GPU-hours is inversely proportional to MFU), so the sensitivity is easy to state precisely rather than hand-waved:

| Assumption changed | Effect on GPU-hours | Effect on cost (at fixed $/hr) |
|---|---|---|
| MFU 40% → 25% (early, unoptimized cluster) | ×1.6 (4.42M → 7.07M) | ×1.6 (~$8.8M → ~$14.1M) |
| MFU 40% → 55% (mature, well-tuned dense-model cluster) | ×0.73 (4.42M → 3.22M) | ×0.73 (~$8.8M → ~$6.4M) |
| $/GPU-hr $2 → $4 (on-demand pricing, no reserved capacity) | no change | ×2 (~$8.8M → ~$17.7M) |
| $/GPU-hr $2 → $1.2 (large, long-term reserved commitment) | no change | ×0.6 (~$8.8M → ~$5.3M) |
| Token count 15T → 20T (data availability turns out better than planned) | ×1.33 (4.42M → 5.89M) | ×1.33 (~$8.8M → ~$11.8M) |

The point of building this table on the spot in an interview, rather than reciting a single point estimate, is that it demonstrates the estimate is a function of stated assumptions, each independently checkable and challengeable — which is exactly the posture a staff engineer needs when presenting this number to a budget-owning executive who will, correctly, ask "what if MFU comes in lower than planned" and expects an answer computed in seconds, not a re-derivation from scratch.

## Step 4d: A Quick Table Across Model Sizes

It's also useful to have the shape of how these numbers scale across a few plausible target sizes memorized well enough to reproduce on a whiteboard, holding tokens-per-parameter roughly fixed near a compute-optimal-ish ratio for illustration (real projects would set D via the Phase 2 reasoning in `008_Planning_A_Model_Training_Run_End_To_End.md`, not a fixed ratio):

| N (params) | D (tokens) | C = 6ND (FLOPs) | GPU-hours @ 40% MFU | Cost @ $2/hr |
|---|---|---|---|---|
| 8B | 15T (deliberately overtrained, Llama-3-style) | 7.2e23 | ~5.06e5 | ~$1.0M |
| 70B | 15T | 6.3e24 | ~4.42e6 | ~$8.8M |
| 405B | 15T | 3.65e25 | ~2.56e7 | ~$51.3M |
| 671B (MoE, 37B active) | 14.8T | 3.28e24 | ~2.30e6 | ~$4.6M |

The 671B MoE row is the one worth pausing on in an interview: despite having by far the largest *total* parameter count in the table, its estimated training compute and cost are far below the 405B dense model's, purely because the `6ND` formula (correctly) uses activated, not total, parameters — a clean, concrete illustration of why MoE's inference-cost/training-cost decoupling (discussed in `008_...`, Phase 1) shows up directly in a compute estimate, not just in serving-cost intuition.

## Step 4e: Common Mistakes This Question Is Designed to Surface

Worth naming these explicitly, because avoiding them is most of what separates a strong answer from a mediocre one on this specific question:

- **Dividing by peak spec-sheet FLOP/s instead of an MFU-adjusted figure.** This is the single most common error and typically overstates achievable throughput by 2-3x, producing a GPU-hours estimate that's too low by the same factor.
- **Using total parameters instead of activated parameters for an MoE model.** This overstates compute (and cost) by roughly `total/activated` — nearly 18x for DeepSeek-V3's specific ratio — and is the fastest way to produce a wildly wrong estimate for any MoE target.
- **Presenting the final number without stating the MFU and $/GPU-hour assumptions it depends on.** A number presented without its assumptions is unfalsifiable and, more importantly, useless to whoever has to act on it — they can't tell you how wrong it might be.
- **Treating the disclosed DeepSeek-V3 figure as if it must reconcile exactly**, and either (a) forcing a fudge factor to make the arithmetic match precisely, which manufactures false precision, or (b) concluding the whole `6ND`-based methodology is broken because it doesn't reconcile exactly, which throws away a genuinely useful estimation tool over an explicable few-percentage-point-of-MFU gap.
- **Forgetting that the estimate is training-compute only.** Presenting a pretraining-FLOPs-derived cost as "the cost of the model" without flagging that it excludes data pipeline engineering, ablations, post-training, evaluation, and inference-serving cost over the deployment lifetime is a scope error significant enough that a careful interviewer will usually probe for it directly.

## Step 4f: Extending the Estimate to Inference — Why the Interviewer Might Push Here Next

A sharp interviewer, having watched you produce a clean training-cost estimate, will often immediately ask "and what does it cost to *serve* this model to N users a day" — because the entire Llama 3 overtraining argument (`..\..\OpenSource\003_Llama3.md`, Section 5, and revisited in `008_Planning_A_Model_Training_Run_End_To_End.md`'s Phase 2) hinges on training cost and inference cost being comparable-order-of-magnitude line items over a real deployment lifetime, not on training cost being the only number that matters. The inference-side estimate uses the same building blocks with one change: a forward-pass-only token costs `~2N` FLOPs (roughly a third of the `6N` training cost per token, since there's no backward pass), so:

```
inference_flops_per_token = 2 * N
queries_per_day = Q
avg_output_tokens_per_query = T
daily_inference_flops = 2 * N * Q * T

# For a 70B model, 50M queries/day, 500 output tokens/query average:
daily_inference_flops = 2 * 70e9 * 50e6 * 500 = 3.5e21 FLOPs/day

# Inference MFU is typically lower than training MFU for a given peak
# figure once you account for decode being memory-bandwidth-bound rather
# than compute-bound at low batch sizes — but for a rough planning number,
# assume a serving-optimized stack achieves an effective 20-30% of peak
# for a well-batched workload.
achieved_inference_flops_per_gpu = 989e12 * 0.25 ≈ 247e12
daily_gpu_hours = (3.5e21 / 247e12) / 3600 ≈ 3,936 GPU-hours/day
daily_cost ≈ 3,936 * $2 ≈ $7,872/day ≈ $2.87M/year
```

At this specific (illustrative) query volume, the *annual* inference cost (~$2.87M/year) is already a meaningful fraction of the one-time ~$8.8M training cost from Step 3, and it recurs every year the model is served — over a 3-year deployment lifetime at this volume, inference cost alone (~$8.6M) is comparable to or exceeds the one-time training cost. This is precisely the arithmetic that makes Llama 3's "shrink the model, even at large extra training-token cost" strategy rational at Meta's actual query volume, and precisely the arithmetic a staff engineer should be able to produce cold, symmetrically with the training-cost estimate, rather than treating training cost as the only number "training compute estimation" is about.

## Step 5: What a Complete Answer Looks Like

Pulling this together, a strong staff-level response to the opening scenario:

1. States the C≈6ND formula and derives it (forward ≈2N, backward ≈4N per token), rather than just quoting it.
2. Flags the caveats: quadratic attention term dropped, N must be activated params for MoE, this is training-only compute.
3. Picks an explicit, stated MFU assumption (35-50% for dense, lower for MoE) rather than dividing by peak spec-sheet FLOP/s.
4. Picks an explicit, stated $/GPU-hour assumption (distinguishing on-demand from reserved/effective pricing) and shows the arithmetic.
5. Sanity-checks the resulting GPU-hours against a plausible cluster size to confirm the implied wall-clock duration is sane.
6. Cross-checks the whole method against a real disclosed number (DeepSeek-V3), shows the reconstruction, and — most importantly — explains the residual gap rather than either forcing an exact match or throwing up hands.
7. Closes with the scope caveat: this estimate is direct pretraining compute only, and a real project budget needs to add data pipeline cost, ablations, post-training, and evaluation on top, plus the inference-cost-over-deployment-lifetime consideration that determines whether training a bigger or smaller model was even the right call in the first place (per Llama 3's overtraining logic, `..\..\OpenSource\003_Llama3.md`).

That seven-point structure — formula, caveats, assumptions stated explicitly, arithmetic worked through, sanity-checked against cluster size, cross-checked against a real number with an honest gap analysis, and scoped explicitly — is what separates a "knows the formula" answer from a "has actually reasoned about what training a frontier model costs" answer, and it's exactly the bar a staff interview loop is calibrated to distinguish.

## Step 6: Presenting This to a Budget-Owning Executive — A Different Register Than an Interview Answer

It's worth closing with how this estimate actually gets *used* organizationally, because a staff engineer's job doesn't end at producing the number — it includes presenting it in a form a non-technical budget owner can act on. That means leading with the headline range (not a false-precision point estimate — "$7-14M depending on achieved MFU" is more honest and more useful than "$8.8M"), stating the two or three assumptions the range is most sensitive to (MFU and $/GPU-hour, per Step 4c's table), and explicitly flagging the scope boundary (this covers pretraining compute only; data pipeline, post-training, evaluation, and inference-serving costs are separate line items that need their own estimates, cross-referencing `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md` and `008_Planning_A_Model_Training_Run_End_To_End.md` for those). A budget presented as a single confident number invites exactly the wrong kind of follow-up conversation when actual costs land outside it; a budget presented as a range with named, checkable assumptions invites the right one — "which of these assumptions should we invest engineering effort in de-risking before we commit the full budget," which is usually the actually decision-relevant question at this stage of a real project.

## Step 7: A Pre-Flight Checklist for This Exact Question

For quick recall under interview time pressure, the estimate should always be produced in this order, checking each box explicitly rather than skipping straight to a final number:

1. Confirm N and D, and whether N should be total or activated parameters (dense vs. MoE).
2. Compute `C = 6ND`, stating the forward/backward derivation if asked.
3. State the peak hardware FLOP/s figure being used and its source (spec sheet, precision format).
4. State an explicit MFU assumption, distinguishing dense from MoE if relevant, and justify it by range (35-50% dense, lower for MoE) rather than picking an arbitrary single number with no stated basis.
5. Compute GPU-hours, then sanity-check against a plausible cluster size to confirm the implied wall-clock duration is neither absurdly short nor absurdly long.
6. State an explicit $/GPU-hour assumption, distinguishing on-demand from reserved/effective pricing, and compute dollar cost.
7. If asked to cross-check, reconstruct against a real disclosed figure (DeepSeek-V3 or Llama 3.1 405B), and explain — not paper over — any residual gap.
8. State the scope boundary explicitly: this is training-compute cost only, and note what's excluded.

Working through these eight steps in order, out loud, with the arithmetic shown at each step rather than skipped to a memorized final answer, is what this question is actually testing — not whether you can recite "$8.8M" from memory.

## Closing Note: Why This Question Recurs So Often in Staff Loops

It's worth reflecting on why some version of this exact question shows up so frequently at the top of a staff research-engineer loop rather than being treated as a warm-up throwaway. It compresses several genuinely distinct competencies into one five-minute exercise: quantitative fluency (can you actually do the arithmetic under time pressure, including unit conversions between FLOPs, seconds, and dollars, without a calculator doing the conceptual work for you), systems literacy (do you know that peak spec-sheet FLOP/s is not achieved throughput, and can you name the reasons why), intellectual honesty (do you state your assumptions instead of hiding behind false precision), and — the part most candidates under-invest in — the discipline to cross-check a number against reality rather than presenting an untested estimate with unwarranted confidence. A candidate who nails the formula but skips the cross-check, or nails the cross-check but can't explain the residual gap, has demonstrated real but incomplete competence; the full seven-to-eight-step version above is what a genuinely staff-level answer looks like end to end.
