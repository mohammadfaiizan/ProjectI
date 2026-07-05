# Interview Questions — Part 1

Scenario, design, and debugging questions in the staff-research-engineer register. Each answer is written as a full worked response, not a bullet-point summary — this is deliberate, since a real interview answer needs to demonstrate reasoning, not just conclusions.

## Q1: You're 10% into a planned 15T-token pretraining run when a new, larger data source becomes available that would materially change your mixture. Do you restart, or work it in mid-run?

The default answer should be "work it in, don't restart," but the reasoning matters more than the conclusion. Restarting discards 10% of an expensive run's compute outright — at the cost profile worked in `001_Estimating_Training_Compute_And_Cost.md`, for a frontier-scale run that's a multi-million-dollar sunk-cost write-off, and it should only be chosen if there's a specific, concrete reason the *existing* 10% of training is actively harmful to keep (not merely suboptimal) going forward.

The real question is how to work the new source in without destabilizing training. Two options: (a) blend the new source into the ongoing data stream at a modest sampling rate immediately, treating it the same as any other mixture-weight adjustment, or (b) hold it back and introduce it in a later "annealing"-style phase — deliberately up-weighting new, high-quality data in the final portion of training, exactly the technique Llama 3 uses for its own high-quality subset (`..\..\OpenSource\003_Llama3.md`, Section 5). Option (b) is generally safer for the model-quality outcome specifically if the new source is large, high-quality, and somewhat distributionally different from what's already been trained on, because introducing a significant distributional shift mid-run (rather than smoothly blended in from a low weight) risks exactly the kind of loss-curve disturbance covered in `003_Debugging_A_Loss_Spike_Mid_Training.md`'s data-issue branch — a sudden large shift in the training distribution can look, mechanically, like a bad-shard event even though nothing is actually corrupted.

The concrete decision: blend the new source in at a conservative sampling rate immediately (avoiding any restart), monitor the loss curve and any capability-benchmark checkpoint evals closely for the following several thousand steps specifically watching for the failure signature above, and reserve the option to up-weight it further in a deliberate annealing phase near the end of the run if the early blend-in shows no adverse effect and evaluation suggests the new source is genuinely valuable.

## Q2: Your gradient norm has been steadily decreasing toward near-zero over several thousand steps, and loss has plateaued well above where scaling-law extrapolation predicted it should be. What's going on?

This is the mirror image of the loss-spike scenario in `003_...` — instead of a sudden explosive event, this is a **vanishing-gradient / dead-optimization** signature, and the diagnostic tree branches differently because the failure is gradual rather than abrupt.

First, distinguish "gradients are genuinely near zero because the model has converged" from "gradients are near zero because something is preventing them from being meaningful," since these look identical on a gradient-norm plot but require completely different responses. Check whether the plateau matches where a well-fit scaling-law curve would predict convergence at this token count for this parameter count — if the observed loss is meaningfully *above* the scaling-law-predicted loss for this compute budget, genuine convergence is not the right explanation, because a properly-optimizing model at this scale shouldn't plateau there yet.

Second, check for a specific set of known causes: (a) an overly aggressive learning-rate decay schedule that decayed the LR too early or too steeply relative to the token budget, effectively starving the back half of training of enough step size to keep making progress — check the actual LR value at the plateau's onset against the schedule's intended value; (b) a numerical precision issue causing gradient underflow specifically (distinct from the overflow failure mode covered in `003_...`) — in FP16 particularly, gradients that are individually small but collectively meaningful can underflow to exactly zero in a narrow-dynamic-range format, which a healthy loss-scaling implementation should prevent, so check whether the loss scaler's scale factor has been allowed to grow appropriately or has been stuck at an overly conservative value; (c) a dead-neuron/saturated-activation problem in a specific subset of layers (common with certain activation functions or normalization misconfigurations) where a meaningful fraction of the network's units have saturated into a regime with near-zero local gradient, which would show up as the near-zero gradient norm being concentrated in specific layers rather than uniform, checkable via per-layer gradient-norm logging.

The fix depends entirely on which of these confirms: a schedule fix (extend or reshape the LR decay) for (a), a loss-scaling fix for (b), or — for (c) — potentially a more invasive intervention (re-initializing or otherwise perturbing the affected layers, though this is a more drastic and lower-confidence fix that should be a last resort after the cheaper schedule and precision checks have been exhausted).

## Q3: Write the check you'd run to determine whether a specific benchmark's headline score is contaminated, given the training corpus and the benchmark's question set.

```python
import numpy as np
from datasketch import MinHash, MinHashLSH

def build_ngram_minhash(text: str, n: int = 8, num_perm: int = 128) -> MinHash:
    """Represent a document as a MinHash over its n-gram (word-level) shingles."""
    tokens = text.lower().split()
    shingles = {" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))}
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))
    return mh

def screen_benchmark_contamination(training_corpus_shards, benchmark_items,
                                    n=8, num_perm=128, jaccard_threshold=0.8):
    """
    For each benchmark item (question + reference answer text), check for
    near-duplicate n-gram overlap against the training corpus via LSH,
    rather than exact string match, since the more common and more
    consequential leakage path is paraphrased/lightly-modified duplication,
    not verbatim copying.
    """
    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)

    # Index the training corpus (in practice: sharded, distributed, incremental)
    for doc_id, doc_text in training_corpus_shards:
        lsh.insert(doc_id, build_ngram_minhash(doc_text, n, num_perm))

    contamination_report = []
    for item_id, item_text in benchmark_items:
        query_mh = build_ngram_minhash(item_text, n, num_perm)
        matches = lsh.query(query_mh)
        contamination_report.append({
            "benchmark_item_id": item_id,
            "num_near_duplicate_matches": len(matches),
            "matched_doc_ids": matches[:5],   # sample, not exhaustive, for triage
        })

    overlap_rate = sum(1 for r in contamination_report if r["num_near_duplicate_matches"] > 0) / len(contamination_report)
    return overlap_rate, contamination_report
```

The design choices worth defending out loud: MinHash/LSH over exact hashing, because paraphrase-level leakage is the dominant real-world contamination path, not byte-identical copying; a tunable `jaccard_threshold`, because the report needs granularity (a 0.3% overlap rate is a very different finding from a 15% one, as discussed in `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`, Step 6) rather than a binary clean/dirty flag; and returning per-item match detail, not just an aggregate rate, so the eval team can specifically exclude or re-evaluate the affected subset of the benchmark rather than discarding the whole benchmark's result over a partial contamination finding.

## Q4: Your checkpointing cadence is currently every 4 hours on a 4,000-GPU cluster with an observed hardware failure rate that causes a job restart roughly every 30 hours. Is this cadence right, and how would you actually reason about it quantitatively?

This is a direct expected-value tradeoff between checkpointing overhead and expected lost work from a failure, and it should be worked numerically rather than answered with an intuition.

```python
def expected_wasted_gpu_hours_per_day(checkpoint_interval_hours, checkpoint_cost_hours,
                                       mean_time_between_failures_hours, num_gpus):
    hours_per_day = 24
    num_checkpoints_per_day = hours_per_day / checkpoint_interval_hours
    checkpoint_overhead_hours = num_checkpoints_per_day * checkpoint_cost_hours

    # Expected wall-clock lost per failure: on average, half a checkpoint
    # interval of training is lost (uniform arrival assumption for the failure
    # within the interval).
    expected_loss_per_failure_hours = checkpoint_interval_hours / 2
    failures_per_day = hours_per_day / mean_time_between_failures_hours
    expected_failure_loss_hours = failures_per_day * expected_loss_per_failure_hours

    total_wasted_wallclock_hours = checkpoint_overhead_hours + expected_failure_loss_hours
    return total_wasted_wallclock_hours * num_gpus   # convert to GPU-hours

# Sweep checkpoint_interval_hours to find the minimum of this function,
# holding checkpoint_cost_hours and MTBF fixed, rather than assuming
# "checkpoint as often as possible" or "checkpoint as rarely as possible"
# is obviously right.
```

Concretely: if a checkpoint write costs, say, 10 minutes (0.167 hours) of stalled training regardless of interval, checkpointing every 4 hours costs `6 checkpoints/day * 0.167h = 1h/day` of pure overhead, while the expected failure loss at a 30-hour MTBF is `(24/30) * (4/2) = 1.6h/day`. Checkpointing more frequently (say, every 2 hours) roughly doubles the overhead (`12 * 0.167 = 2h/day`) while roughly halving the expected failure loss (`(24/30) * 1 = 0.8h/day`) — worse total (2.8h vs. 2.6h) at this specific MTBF and checkpoint cost. The right cadence is the one that minimizes the *sum*, and — critically — it should be recomputed whenever either input changes materially: a cluster with a degrading MTBF (per `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md`'s hardware-degradation branch) or a checkpoint-cost reduction (e.g., from adopting asynchronous/non-blocking checkpoint writes) both shift the optimal interval, and treating checkpoint cadence as a fixed, set-once config value rather than a live-tuned parameter is a common, avoidable inefficiency.

## Q5: A colleague proposes serving a brand-new checkpoint to 100% of production traffic immediately after it passes your offline eval suite. What's wrong with this plan, and what would you do instead?

The core problem: an offline eval suite, however comprehensive, is a curated sample of the real production query distribution, not the distribution itself — and `009_Post_Launch_Model_Degradation_And_Incident_Response.md`'s entire premise is that regressions can appear in production that no offline eval caught, whether from a genuine capability gap in a task category underrepresented in the eval suite, a serving-stack interaction the offline eval harness doesn't exercise (different batching, different quantization path in production vs. eval), or a real-world query-distribution mismatch. Going straight to 100% traffic removes the only mechanism (a staged rollout) that would let you catch this cheaply before it affects the whole user base.

What I'd do instead: a staged canary rollout — an initial small percentage of production traffic (low enough that a regression's blast radius is limited, but large enough to get statistically meaningful signal within a reasonable time window), instrumented with the same task-success/regression-detection monitoring described in `009_...`'s Q&A-adjacent code sketch, compared in real time against the currently-serving checkpoint on the same traffic mix. Only after the canary window shows no statistically significant regression on any pre-registered product/quality metric would traffic ramp up in stages (e.g., 1% → 10% → 50% → 100%), with an explicit, pre-agreed rollback trigger and procedure at each stage — and the rollback procedure itself needs to be tested and ready *before* the rollout starts, not designed reactively if a problem appears, because designing a rollback mechanism under live incident pressure is exactly the wrong time to discover it's harder than expected.

## Q6: Halfway through a planned 60-day pretraining run, a new hardware generation with much better FP8 throughput becomes available on your cluster. Do you migrate mid-run?

Almost never, and the reasoning is worth being precise about rather than just asserting "no." Migrating to new hardware mid-run, even holding numerics fixed, reintroduces exactly the kind of confound `003_Debugging_A_Loss_Spike_Mid_Training.md`'s Branch D describes: any subsequent anomaly becomes far harder to root-cause, because you no longer have a stable hardware baseline to compare against, and a training run half-completed on one hardware generation's numerical behavior (accumulation precision, exact kernel implementations, communication-library versions tuned for that hardware) switching to a different generation risks subtle numerical discontinuities that are extremely difficult to distinguish from a genuine training instability after the fact.

The more defensible version of "take advantage of the new hardware": let the *current* run finish on its current hardware, and plan the *next* run (or a fresh ablation/validation cycle) on the new hardware, treating the migration itself as a project with its own validation requirements — confirming numerical equivalence on a smaller-scale test run, re-establishing MFU baselines (since a new hardware generation's achievable MFU for your specific model/parallelism configuration is not something you can safely assume transfers from the old generation's tuning) — before committing a full-scale production run to it. The only circumstance that would justify a genuine mid-run migration is a hard external forcing function (e.g., the current hardware is being decommissioned and there's no option to simply wait), and even then, the migration should be executed at a checkpoint boundary with an explicit validation pass (replay a recent batch on both hardware configurations and confirm matching loss/gradient behavior) before resuming full-speed training on the new hardware, exactly the same "confirm before resuming" discipline as `003_...`'s Step 4.

## Q7: Your near-duplicate deduplication job, using standard MinHash/LSH, is taking far longer than planned against a multi-trillion-token raw corpus. Walk through how you'd speed it up, and what you'd give up.

The two levers that actually move the needle at this scale, and their respective costs:

**Reduce `num_perm` (the number of hash permutations per MinHash signature).** Fewer permutations means a smaller signature per document (cheaper to compute and store) and cheaper LSH banding/querying, at the cost of a noisier Jaccard-similarity estimate — the variance of the MinHash Jaccard estimator scales roughly as `1/num_perm`, so halving `num_perm` roughly doubles the estimator's variance, which pushes more true-near-duplicate pairs into ambiguous territory near the similarity threshold, meaning your dedup pass will both miss more genuine near-duplicates and, depending on how the LSH bands/threshold interact with the noisier estimate, potentially flag more false positives. This is a real precision/recall tradeoff, not a free win, and the right response is to characterize it on a small held-out sample where ground-truth near-duplicate pairs are known (or can be manually labeled) before committing the reduced-`num_perm` configuration to the full corpus.

**Increase the LSH banding coarseness (fewer bands, more rows per band, or vice versa depending on which failure mode you're avoiding).** LSH's banding parameters directly trade recall for compute: coarser banding (fewer bands) reduces the number of candidate-pair comparisons generated (cheaper) but reduces the probability that a true near-duplicate pair actually lands in the same bucket at least once (lower recall). The standard mitigation, if compute allows, is to tune bands/rows to hit a target recall at a specific similarity threshold via the well-known S-curve formula for LSH's probability of detection as a function of `(bands, rows, similarity)`, rather than picking parameters ad hoc.

**What I'd actually give up, stated plainly:** near-perfect deduplication recall, in exchange for tractable compute — and I'd make this trade *legible*, not silent, by running the tuned configuration against a labeled validation sample and reporting the resulting estimated recall/precision explicitly as part of the corpus's versioned manifest (per `002_...`, Step 7), so that anyone later investigating a training anomaly potentially tied to residual near-duplicate content has an honest, quantified starting point rather than an unstated assumption that dedup was perfect.

## Q8: You've trained proxy models at several scales and fit a scaling-law curve. The curve suggests diminishing returns are setting in earlier than expected for your target architecture. How do you decide whether to trust this and change the plan?

First, check whether the proxy-model fit is actually well-conditioned before trusting its extrapolation: how many scale points were fit, do they span a wide enough range of compute to constrain the power-law fit's exponent with reasonable confidence intervals, and — critically — is the *architecture* of the proxy models identical to the target architecture in every respect that could plausibly change scaling behavior (same attention mechanism, same MoE-vs-dense structure, same data mixture). Chinchilla-style scaling laws are empirically fit curves, not physical laws, and a fit built on, say, three proxy points spanning a narrow compute range extrapolated confidently out to a target 100x larger than the largest fitted point is exactly the kind of overconfident extrapolation that has produced real, documented surprises in the field (both directions — sometimes actual runs beat the naive extrapolation, sometimes they undershoot it) — the confidence interval on the extrapolation, not just the point estimate, is the thing to actually examine here.

Second, if the diminishing-returns signal survives that scrutiny (well-conditioned fit, matched architecture, still shows earlier-than-expected plateauing), take it seriously as actionable evidence rather than dismissing it because it complicates the plan — this is exactly the discipline `008_Planning_A_Model_Training_Run_End_To_End.md`'s Phase 6 argues for (periodic checkpoint evals against the expected trajectory, to catch exactly this kind of signal early enough to act on it) applied one level earlier, at the planning-proxy stage rather than mid-run. The concrete action: revisit the N/D allocation decision from Phase 2 of that file — if the target architecture's actual scaling curve says the planned parameter count is past its useful marginal-return point sooner than a generic Chinchilla-style prior would suggest, the compute-optimal reallocation is toward more tokens (or a different architecture change entirely) rather than blindly executing the original N/D split the planning stage assumed. This should be a go/reconsider decision gate in its own right, made explicit in the project plan before committing full-scale compute, not an after-the-fact rationalization if the full-scale run underperforms.

## Q9: You're increasing global batch size over the course of a pretraining run (a common practice at scale). What could go wrong, and how would you validate the schedule before committing to it at full scale?

The core risk: increasing batch size without a correspondingly adjusted learning rate changes the effective noise scale of the SGD/Adam update in a way that's well-documented to interact with training stability — a batch-size increase without an LR adjustment can either slow convergence (if the LR is now too small relative to the new, larger, lower-variance gradient estimate) or, in the other direction, a batch-size *decrease* or a batch-size increase paired with an inappropriately large LR bump can push training into the "large learning rate at large batch" instability regime that shows up as elevated gradient noise or the kind of spike behavior covered in `003_Debugging_A_Loss_Spike_Mid_Training.md`. There is a reasonably well-established heuristic (roughly, scale LR with the square root of the batch-size ratio, though the exact right scaling depends on the optimizer and regime) for how to adjust LR alongside a batch-size change, but it's a heuristic, not a guarantee, and should be validated rather than assumed.

The validation approach: before committing the batch-size schedule to the full-scale run, test it at a smaller proxy scale first — run the exact planned batch-size-increase schedule (same relative timing, same LR-adjustment rule) on a smaller model/token budget, and confirm the loss curve shows no anomaly at each scheduled batch-size transition point. If a small-scale proxy run shows any instability at a transition, that's a strong signal to revisit the LR-adjustment rule before committing the full-scale run to the same schedule, rather than discovering the interaction for the first time at full scale where the cost of a resulting loss-spike incident (per `003_...`) is far higher.

## Q10: A benchmark your team has used for two years is now saturating — nearly every frontier model scores within a point or two of the maximum. How do you decide whether and when to retire it?

Saturation alone isn't sufficient grounds to retire a benchmark immediately — the right diagnostic sequence: first, determine *why* it's saturating. Is it because models have genuinely mastered the underlying capability the benchmark was designed to measure (a legitimate "this benchmark has done its job" retirement case), or is it saturating because of contamination creeping in over successive training runs (an increasingly probable outcome the longer a fixed, static benchmark has been public and available to be scraped into training corpora, exactly the risk `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`'s Step 6 exists to screen for), or because the benchmark has a narrow answer-format/surface-pattern that models have learned to exploit without the underlying capability genuinely improving (a benchmark-specific overfitting/gaming pattern, structurally similar to the reward-hacking dynamics in `006_Responding_To_A_Reward_Hacking_Incident.md` but applied to an evaluation metric rather than a training-time reward).

The way to distinguish these: construct a fresh, held-out variant of the benchmark (new items testing the same underlying capability, either newly authored or paraphrased/rewritten from the original in a way that couldn't have been in any training corpus) and compare scores on the fresh variant against the original. If scores on the fresh variant are meaningfully lower than the saturated original, that's strong evidence for contamination or surface-pattern overfitting rather than genuine capability mastery, and the benchmark shouldn't be retired so much as *replaced* with a harder or freshly-constructed successor testing the same capability. If scores on a well-constructed fresh variant are comparably high, that's better evidence the capability really has been mastered broadly, and retirement (or at least demotion from a launch-gating headline metric to an informational one, per `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md`'s tiering) is the right call — freeing up the evaluation framework's attention and launch-gating weight for a benchmark that still discriminates meaningfully between models.

## Q11: During RLHF reward-model training, the RM's training loss looks healthy, but validation accuracy on held-out preference comparisons is barely above chance. What's the likely cause, and how do you fix it?

A large gap between healthy training loss and near-chance held-out accuracy is a classic overfitting signature, and the specific mechanism worth checking first, given the Bradley-Terry pairwise training setup described in `..\..\GPT\004_InstructGPT_And_RLHF.md` Section 6: if all `C(K,2)` pairwise comparisons from the same prompt are treated as independent training examples without normalizing per-prompt (the exact overfitting risk that paper's own methodology explicitly guards against via per-prompt loss normalization and single-forward-pass batching), the RM can effectively memorize prompt-specific patterns from the handful of completions seen for each training prompt rather than learning a generalizable notion of "better response" — training loss looks great because the model has learned the training prompts' specific completions well, while held-out accuracy on genuinely new prompts stays near chance because nothing generalizable was learned.

Confirm by checking whether the per-prompt normalization is actually implemented correctly (not just present in the code, but verified to produce the intended effective weighting), and by checking whether held-out accuracy specifically on *prompts* not seen during training (not just held-out comparison pairs from prompts that *were* seen in training, which is a much weaker and more easily-gamed held-out split) is the metric being tracked — a held-out split that only holds out comparison pairs but not prompts themselves will systematically overstate generalization. If the normalization is confirmed correct and the held-out split is confirmed to be prompt-disjoint from training, the next suspects are a genuinely too-small or too-narrow training prompt/comparison dataset (not enough diversity for the RM to learn a generalizable preference signal at all) or a labeler-agreement problem (if labeler preference judgments are highly inconsistent/noisy — a real, acknowledged risk given the labeler-pool-bias and consistency issues flagged directly in `..\..\GPT\004_InstructGPT_And_RLHF.md` Section 8 — the RM has no consistent signal to learn from regardless of architecture or training setup, and the fix is a labeling-quality/agreement audit, not a training-code fix).

## Q12: You're planning a red-team engagement for an agentic coding model before launch. What specifically would you have the red team try that a standard capability benchmark wouldn't catch?

A standard capability benchmark measures whether the model can produce a correct solution to a well-specified task; a red-team engagement for an *agentic* model needs to specifically probe failure modes that only emerge from extended, tool-using, multi-step interaction, which is exactly the gap `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md`'s Layer 5 names. Concretely, I'd have the red team specifically attempt: (a) prompt-injection-style attacks embedded in tool outputs or retrieved content (a malicious instruction hidden inside a file the agent reads or a web page it fetches, testing whether the agent's behavior can be hijacked by content it wasn't directly told by the user), since this is a failure mode that literally cannot manifest in a single-turn, no-tool-use benchmark; (b) tasks deliberately designed to tempt destructive or irreversible actions under ambiguous authorization (e.g., a task framed in a way that could be read as authorizing a `rm -rf`-equivalent action, testing whether the model seeks clarification or proceeds on a risky interpretation); (c) long-horizon tasks specifically probing whether the agent recognizes when it's stuck in an unproductive loop (retrying the same failed approach repeatedly) versus escalating to ask for help or trying a genuinely different strategy; and (d) tasks that require the agent to recognize the limits of its own tool access or permissions and behave appropriately when a requested action would exceed those limits, rather than fabricating a plausible-looking success report for an action it couldn't actually perform (a particularly dangerous failure mode for an agentic system specifically, since a confidently-fabricated "I completed X" report is much harder for a downstream user to catch than an obviously-failed, visible error).

The output of this engagement should feed directly into Layer 3's severity-tiered gating structure from `005_...` — a finding in category (a) or (b) above, given the potential consequence severity, should sit in the hard-blocker or high-severity-negotiable-blocker tier, not the informational tier, regardless of how rare the red team found the specific triggering condition to be.

## Q13: Write a lightweight training-loop hook that detects a gradient-norm spike and automatically pauses training rather than continuing through it.

```python
import torch

class GradientSpikeGuard:
    """
    Tracks a running statistic of gradient norm and halts training if the
    current step's gradient norm is an extreme outlier relative to recent
    history — implementing the "pause, don't train through it" discipline
    from 003_Debugging_A_Loss_Spike_Mid_Training.md as an automated guard
    rather than relying on a human noticing the loss curve later.
    """
    def __init__(self, window: int = 200, z_threshold: float = 8.0, min_history: int = 50):
        self.window = window
        self.z_threshold = z_threshold
        self.min_history = min_history
        self.history = []

    def check(self, model: torch.nn.Module, step: int) -> dict:
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.float().norm(2).item() ** 2
        grad_norm = total_norm_sq ** 0.5

        result = {"step": step, "grad_norm": grad_norm, "flagged": False}

        if len(self.history) >= self.min_history:
            mean = sum(self.history) / len(self.history)
            var = sum((x - mean) ** 2 for x in self.history) / len(self.history)
            std = var ** 0.5
            z = (grad_norm - mean) / (std + 1e-8)
            if z > self.z_threshold:
                result["flagged"] = True
                result["z_score"] = z
                result["recent_mean"] = mean
                result["recent_std"] = std

        self.history.append(grad_norm)
        if len(self.history) > self.window:
            self.history.pop(0)

        return result

# Usage inside the training loop:
# guard = GradientSpikeGuard()
# ...
# loss.backward()
# check = guard.check(model, step)
# if check["flagged"]:
#     save_emergency_checkpoint(model, optimizer, step, tag="spike_guard_triggered")
#     raise TrainingPausedForInvestigation(check)
# optimizer.step()
```

The design decisions worth defending: a rolling-window z-score rather than a fixed absolute threshold, because "normal" gradient-norm magnitude drifts over the course of a run (different phases of training, LR schedule changes) and a fixed threshold would either be too sensitive early or too insensitive late; a `min_history` guard so the check doesn't fire spuriously before enough history has accumulated to compute a meaningful statistic; and an emergency checkpoint saved *at the moment of detection*, before raising the halt — giving the investigation from `003_...` a precise, immediately-available artifact to inspect (the exact model/optimizer state right at the flagged step) rather than having to reconstruct it from the last regular checkpoint, which could be hours of steps earlier.

## Q14: After fine-tuning a model on a narrow, high-quality domain dataset, you notice it has become measurably worse at general-purpose tasks it used to handle well. What's happening, and what are the fix options?

This is catastrophic forgetting: fine-tuning on a comparatively narrow dataset, especially for enough epochs/steps to strongly fit that dataset, can overwrite or degrade the base model's more general capabilities, since nothing in a narrow-domain SFT loss explicitly protects general capability the way pretraining's broad, diverse objective implicitly does. The severity scales with how narrow the fine-tuning data is, how many steps/epochs were run, and how high the learning rate was relative to how far the fine-tuned distribution is from the base distribution (directly connecting to the distributional-distance axis from `007_Deciding_Between_Pretraining_Fine_Tuning_And_Prompting.md`).

Fix options, in order of how invasive they are: (a) reduce the number of fine-tuning epochs/steps and lower the learning rate, accepting a smaller shift toward the target domain in exchange for less collateral damage to general capability — the cheapest fix, worth trying first; (b) mix a general-capability dataset into the fine-tuning data alongside the narrow domain data, directly analogous to PPO-ptx's mixed pretraining-log-likelihood objective in InstructGPT (`..\..\GPT\004_InstructGPT_And_RLHF.md`, Section 6) — anchoring the fine-tuning process to not drift too far from general capability, at the cost of somewhat diluting how strongly the domain-specific behavior is learned, exactly the alignment-tax-style tradeoff that technique makes explicit; (c) use a parameter-efficient fine-tuning method (LoRA or similar) instead of full fine-tuning, which constrains the update to a lower-rank subspace and empirically tends to cause less catastrophic forgetting than full-parameter fine-tuning, at some potential cost to how much domain-specific capability gain is achievable; and (d), if none of the above sufficiently resolves it and the domain requirement is genuinely large-scale, escalate to continued pretraining on a broader domain corpus (rather than a narrow fine-tuning set) followed by a lighter fine-tuning pass, per `007_...`'s escalation framework — since continued pretraining on a large enough and appropriately-mixed corpus is less prone to this specific narrow-overfitting failure mode than fine-tuning on a small, narrow dataset is.

## Q15: You have a $50K compute budget and a domain-specialization task. How do you decide between full fine-tuning and a parameter-efficient method like LoRA?

Work this as a direct application of `007_Deciding_Between_Pretraining_Fine_Tuning_And_Prompting.md`'s Axis 2 (budget) and Axis 4 (distributional distance), made concrete at this specific budget scale. At $50K, full fine-tuning of a large (70B+) model is likely affordable for a single run but leaves little room for iteration, ablation, or a mistake — and if the target task is a comparatively shallow behavioral/format shift (Axis 4's "sweet spot" case), LoRA's dramatically lower per-run compute cost (updating a small fraction of parameters via a low-rank adapter, rather than the full parameter set) buys enough headroom within the same budget to run several iterations — different rank choices, different target-module selections, different data curation passes — and empirically arrive at a better final result than a single full-fine-tuning attempt with no room for correction.

The case for spending the budget on full fine-tuning instead: if the target task requires a genuinely larger behavioral shift than LoRA's constrained low-rank update subspace can comfortably express (a real, if somewhat task-dependent, limitation — LoRA's effective capacity to shift model behavior is bounded by its rank in ways that can matter for sufficiently large distributional shifts, even short of the "needs continued pretraining" regime from Axis 4), and if the $50K budget is genuinely a one-shot allocation with no expectation of iteration budget later, a single well-executed full fine-tuning run with careful hyperparameter selection informed by smaller-scale pilot runs (spend a modest fraction of the budget on a cheap pilot at reduced scale/data before committing the rest to the full run, rather than spending the whole budget on one unvalidated full-scale attempt) is the more defensible choice. The practical recommendation at this exact budget scale, absent more specific information about task difficulty: start with LoRA specifically because it preserves iteration capacity within a fixed budget, and only escalate to full fine-tuning if a well-executed LoRA pass demonstrably plateaus below the required quality bar.

## Q16: Your MoE model's router has collapsed — a small number of experts receive the vast majority of tokens, and the rest are barely used. Diagnose and fix.

This is a load-balancing failure, and the diagnostic question is which balancing mechanism is in place and whether it's actually functioning as intended. If using a standard auxiliary-loss-based balancing scheme, check whether the auxiliary loss's weight is actually large enough to meaningfully compete with the LM loss's gradient signal — too small a weight is the single most common cause of exactly this symptom, since a negligibly-weighted balancing loss provides essentially no correction pressure against whatever the LM loss's gradient naturally prefers (which, absent balancing pressure, often collapses toward a small number of experts that happened to get an early advantage and then compound that advantage through the standard rich-get-richer dynamic of gradient-based routing, since experts that receive more tokens receive more gradient signal and become progressively better/more attractive to route to).

If using an aux-loss-free, bias-based balancing mechanism (à la DeepSeek-V3, `..\..\OpenSource\007_DeepSeek_V3.md`, Section 2), check whether the bias step size γ is large enough to meaningfully counteract the observed load imbalance within a reasonable number of steps, and whether the load-observation window the bias update is computed over is appropriately sized (too long a window means the bias reacts too slowly to a developing collapse; too short a window can make the bias update noisy and possibly unstable) — this connects directly to the tuning-surface critique raised in `010_Critiquing_Real_Published_Training_Recipes.md`'s discussion of this exact mechanism, and a real collapse incident is precisely the kind of evidence that would validate or invalidate that critique's concern in practice.

The fix, once diagnosed: increase the auxiliary-loss weight (if using that scheme) or the bias step size γ / shrink the observation window (if using the bias scheme), and — separately, as a structural mitigation worth having regardless of which balancing mechanism is used — check whether router initialization gave a meaningfully uniform starting preference across experts, since a poorly-initialized router that starts with a strong preference for a small subset of experts gives the rich-get-richer dynamic a head start that balancing pressure then has to overcome rather than merely maintain.

## Q17: Write the monitoring check you'd deploy to catch a gradual, multi-hour latency-percentile regression in a production serving system before users notice.

```python
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class LatencySample:
    timestamp: float
    p50_ms: float
    p99_ms: float
    request_count: int

class LatencyRegressionMonitor:
    """
    Watches p50/p99 latency over a rolling baseline window and flags a
    statistically meaningful, *sustained* drift — distinguishing a real
    gradual regression (per 004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md's
    "gradual vs. step-function" shape distinction, applied to serving
    rather than training) from normal minute-to-minute noise.
    """
    def __init__(self, baseline_window_minutes: int = 60,
                 sustained_window_minutes: int = 15,
                 p99_regression_threshold_pct: float = 20.0):
        self.baseline_window = baseline_window_minutes * 60
        self.sustained_window = sustained_window_minutes * 60
        self.threshold_pct = p99_regression_threshold_pct
        self.samples: deque[LatencySample] = deque()

    def ingest(self, sample: LatencySample):
        self.samples.append(sample)
        cutoff = sample.timestamp - self.baseline_window
        while self.samples and self.samples[0].timestamp < cutoff:
            self.samples.popleft()

    def check(self, now: float) -> dict | None:
        if not self.samples:
            return None

        baseline_cutoff = now - self.baseline_window
        recent_cutoff = now - self.sustained_window

        baseline = [s for s in self.samples if s.timestamp < recent_cutoff and s.timestamp >= baseline_cutoff]
        recent = [s for s in self.samples if s.timestamp >= recent_cutoff]

        if len(baseline) < 5 or len(recent) < 5:
            return None  # not enough data in either window yet

        baseline_p99 = sum(s.p99_ms for s in baseline) / len(baseline)
        recent_p99 = sum(s.p99_ms for s in recent) / len(recent)

        pct_change = (recent_p99 - baseline_p99) / baseline_p99 * 100
        if pct_change > self.threshold_pct:
            return {
                "alert": "sustained_p99_latency_regression",
                "baseline_p99_ms": baseline_p99,
                "recent_p99_ms": recent_p99,
                "pct_change": pct_change,
                # A real deployment attaches the same recent-infra-change-log
                # lookup used in 009_Post_Launch_Model_Degradation_And_Incident_Response.md
                # here automatically.
            }
        return None
```

The design point worth calling out: comparing a *recent, sustained* window against a longer *baseline* window, rather than alerting on any single latency spike, is specifically what distinguishes catching a genuine multi-hour drift from drowning in false alarms triggered by normal, transient load-driven latency variance — a monitor that alerts on every momentary p99 blip is a monitor nobody trusts and everybody eventually mutes, which is a worse outcome than not having the check at all.

## Q18: How would you empirically verify that a scaling law fit on a dense-transformer architecture still applies after a significant architecture change (e.g., switching from standard MHA to MLA)?

The honest starting position: don't assume it transfers, and don't assume it doesn't — treat this as an empirical question requiring its own small-scale validation, because a scaling law's fitted exponents are a property of the specific architecture family (and, to a lesser extent, the data mixture) it was fit on, not a universal physical constant, and a sufficiently significant architectural change (MLA's attention mechanism materially changes both the parameter allocation within the attention block and the effective information bottleneck the model routes through relative to standard MHA) is exactly the kind of change that could plausibly shift the fitted curve's exponents, not just its intercept.

The validation approach: run the same proxy-model-at-multiple-scales methodology used to fit the original scaling law, but with the new architecture (MLA) substituted in, at a range of scales small enough to be cheap but wide enough to constrain a fit, and directly compare the resulting fitted curve against the original MHA-based curve. Two useful things to check specifically: does the new architecture's curve show a similar loss-vs-compute exponent (suggesting the change primarily affects the constant/intercept — e.g., MLA being more compute/memory-efficient per parameter without changing the fundamental scaling relationship) or a genuinely different exponent (suggesting the change affects how the model's effective capacity per parameter scales, a more fundamental and consequential finding); and does the new curve's *predicted* loss at the target full scale differ meaningfully from the old curve's prediction at the same compute budget, since that delta is exactly the planning-relevant number for `008_Planning_A_Model_Training_Run_End_To_End.md`'s Phase 2 decision. Only once this small-scale validation is in hand should the full-scale training-run plan be committed using the new architecture's own fitted curve rather than inheriting assumptions from a curve fit on a structurally different architecture.

## Q19: A product team wants a large vocabulary (200K+ tokens) for a heavily multilingual model. What are the actual costs of this decision, beyond "bigger embedding matrix"?

The embedding/unembedding matrix size (`vocab_size * d_model`, counted twice for input and output unless weight-tied) is the most visible cost, but it's not the most important one to walk through in an interview, because the more consequential costs are less obvious. First: a larger vocabulary, if built primarily to improve compression for underrepresented languages, directly changes tokens-per-character across the entire training corpus — this is a genuine efficiency win in the direction Llama 3's vocabulary expansion demonstrates (`..\..\OpenSource\003_Llama3.md`, Section 2), reducing both training-token count needed for a fixed amount of "real" content and inference decode-step count for generation — but only if the added vocabulary entries are actually well-utilized across the training mixture; a 200K vocabulary built without a correspondingly large, well-balanced multilingual training corpus risks a large fraction of vocabulary entries being rarely seen during training, which under-trains their embeddings and can produce degenerate or unstable behavior specifically when those rare tokens do appear at inference time (a real, previously-observed failure pattern with large, poorly-balanced vocabularies).

Second, and less obvious: a larger vocabulary increases the compute cost of the final unembedding projection and softmax at every single training and inference step (this scales with `vocab_size * d_model` per token, applied at every position), which is a real, non-negligible fraction of total FLOPs especially for smaller models where `d_model` doesn't dominate the per-layer cost as heavily as it does at frontier scale — this is exactly why the cost is proportionally worse at 8B (per `..\..\OpenSource\003_Llama3.md`, Section 2's explicit note) than at 405B. Third, tokenizer training itself (fitting a 200K-entry BPE or similar vocabulary well across a genuinely multilingual corpus, with sensible per-language allocation) is a nontrivial data-engineering effort in its own right, connecting back to `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md`'s sequencing — the tokenizer needs to be finalized early, since changing it after training has begun means restarting, and getting the per-language token-count allocation right requires the multilingual data-mixture decisions from that file's Step 5 to already be reasonably settled before tokenizer training can be done well.

## Q20: You resume a training run from a checkpoint after a cluster resize (going from 512 to 768 GPUs, changing the data-parallel degree). A few thousand steps later, you see a loss anomaly. How does the resize specifically change your diagnostic approach from the generic loss-spike playbook?

The generic playbook from `003_Debugging_A_Loss_Spike_Mid_Training.md` still applies, but the resize adds a specific, high-prior additional hypothesis that should be checked *before* working through the generic tree from scratch: a resize changes the data-parallel degree, which changes the global batch size (if per-GPU micro-batch size is held fixed) and, depending on how the training framework's checkpoint-resume logic handles a changed parallelism configuration, creates real risk of an optimizer-state sharding mismatch — Adam's moment buffers are typically sharded across the data-parallel group in a ZeRO/FSDP-style scheme, and resuming with a different DP degree than the checkpoint was saved under requires the resume logic to correctly re-shard (or gather-and-re-shard) that state, which is exactly the kind of checkpoint-resume bug flagged as a real, previously-observed failure class in `003_...`'s Branch C.

Given the resize just happened, this hypothesis should jump to the front of the investigation queue rather than being reached only after working through data and numerical branches first (the usual "cheapest/most common first" ordering from the generic playbook is itself context-dependent — a recent, known configuration change is a strong prior that should reorder the investigation). Concretely: verify the global batch size actually changed as expected post-resize (or was deliberately kept constant via a corresponding micro-batch-size adjustment, which itself needs verifying), and directly inspect specific parameters' post-resume optimizer-state values against what they should be, to confirm the resharding logic preserved the correct moment-buffer values rather than silently corrupting or zeroing them for some subset of parameters. If this comes back clean, then proceed through the standard data/numerical/optimizer/hardware tree exactly as in the generic scenario — but skipping the resize-specific check first, given how directly it correlates with the timing of the anomaly, would be a real diagnostic inefficiency.
