# Post-Launch Model Degradation And Incident Response

## Step -1: Why "No Model Update" Is the Load-Bearing Detail

This scenario's premise — no weight change on the team's end — is doing real work, and a strong answer should say so explicitly before diving into hypotheses. It rules out the most reflexive explanation (something about the model regressed) and forces the investigation toward the surrounding system: serving infrastructure, configuration, the real world the model is being asked about, and the experimentation layer routing traffic to it. This is precisely the same discipline as `006_Responding_To_A_Reward_Hacking_Incident.md`'s Step 1 (confirm before diagnosing) and `003_Debugging_A_Loss_Spike_Mid_Training.md`'s Step 1 (characterize before branching) — in every one of this module's incident-response scenarios, the single most common way to waste the first hour of an investigation is skipping straight to a favorite hypothesis instead of using the specific details of the premise to narrow the hypothesis space first.

## Step -2: A Quick FAQ on Framing This Investigation

- **Should the investigation team include anyone beyond the model/training team?** Yes, from the start — serving infrastructure, product/experimentation, and the training team all own a piece of the hypothesis space, and routing the investigation through only one of them delays whichever hypothesis lands outside that team's visibility.
- **Is it acceptable to run a quick fix while the investigation is still ongoing?** Only if the fix is cheap and reversible and doesn't destroy diagnostic evidence — e.g., restarting a serving process speculatively could reset exactly the state (a stuck experiment-assignment cache, a memory-fragmentation signature) that Hypothesis A or D's confirmation depends on.
- **How long should the investigation run before escalating for more resources?** As soon as the cheap, high-prior checks (infra/config and routing logs) come back clean without an explanation — at that point the remaining hypothesis (query-distribution shift) is analytically more expensive, and escalating for dedicated data-analysis support early is usually the right call rather than one engineer working through it alone.

## The Scenario

"Users start reporting that a model in production seems to have gotten noticeably worse at a specific task — say, code generation, or following multi-step instructions — over the past week. Your team hasn't shipped a model update. Walk me through your incident response."

This is a deliberately disorienting scenario, because the natural first assumption ("something about the model changed") is explicitly ruled out by the premise, which forces the investigation toward causes that have nothing to do with weights — serving infrastructure, configuration drift, input-distribution shift, or an experiment-routing bug. The skill being tested is whether you can hold multiple genuinely different hypothesis classes in mind simultaneously and design an investigation that discriminates between them efficiently, rather than fixating on the first plausible-sounding story.

## Step 0: A Pre-Incident Checklist — What Should Already Exist

- A fixed, versioned, per-task-category offline regression suite, runnable on demand against production at any time, not assembled from scratch during an active incident.
- A continuously-running scheduled version of that same suite (per Step 7's monitoring sketch), alerting on sustained degradation automatically.
- A unified, cross-functional change log covering infrastructure deploys, quantization/config changes, and experiment/routing assignment changes — not three separate logs owned by three teams with no shared view.
- Input-prompt-distribution monitoring (length, structure, task-subtype composition) tracked over time, so a genuine query-distribution shift (Hypothesis C) is visible proactively rather than only inferable after the fact.
- A tested rollback procedure for both infrastructure/config changes and experiment-routing assignments, so Step 8's resolution step is fast once root-caused.

## Step 1: Establish That the Regression Is Real Before Investigating Causes

Exactly as in `006_Responding_To_A_Reward_Hacking_Incident.md`'s first step, the first move is to convert "users report it seems worse" into a measured, quantified effect, because user reports are a noisy, biased sample (people who notice and report a regression are not a random sample of all users, and recency/attention bias means a handful of vivid complaints can look like a trend that a broader measurement doesn't support):

- **Pull the actual task-success telemetry** for the specific task category being complained about, over the relevant time window, compared against the weeks before. If there's a proxy for task success in production (explicit user feedback signals, regeneration/edit rates, downstream action completion for agentic tasks), plot it — a real, statistically distinguishable drop in this metric, correlated with the reported time window, converts "vibes" into "confirmed regression, starting approximately [date]."
- **Reproduce it directly, offline.** Take a fixed, curated set of prompts representative of the complained-about task and run them against the current production system, comparing against either a saved historical output set (if available) or, if not, against a same-architecture/same-checkpoint replica running in a controlled environment isolated from whatever production infrastructure is under suspicion. A reproducible offline regression is far stronger footing for the rest of the investigation than trying to diagnose a phenomenon you can only observe indirectly through user complaints.
- **Nail down the exact regression window as precisely as possible.** Because the premise states no model update shipped, the regression window is your best lead for correlating against *everything else* that happened in that window — an infrastructure deploy, a config change, a traffic-routing change, or simply the passage of enough calendar time for the real-world query distribution to have shifted. Precision here (was it a step-function change on a specific date, or a gradual drift over the week) directly narrows which of the four hypotheses below is most likely, exactly as throughput-regression shape-characterization narrows the diagnostic tree in `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md`.

## Step 1b: Common Mistakes This Scenario Is Designed to Surface

- Assuming the model's weights must be involved simply because the complaint is about "model quality," when the premise explicitly rules out a weight change.
- Treating user complaints as directly quantitative evidence rather than converting them into a measured, reproducible regression first.
- Investigating all four hypotheses simultaneously with divided attention rather than checking the cheap log-audits (Hypotheses A, B, D) before the more analytically expensive distribution-shift investigation (Hypothesis C).
- Forgetting that Hypothesis C often has no "revert" resolution at all, and mistakenly forcing a rollback response onto a situation that actually calls for a product/training input instead.
- Not maintaining a fixed offline reproduction set, and therefore having no stable reference point to test each hypothesis against.

## Step 2: Hypothesis A — Serving-Infrastructure Regression

**The mechanism.** Nothing about the model's weights needs to change for its *effective* behavior to change if something in the serving stack shifted — a batching-configuration change that altered numerical behavior, a change in speculative-decoding draft-model or acceptance-threshold configuration, a change in the maximum-context or truncation-handling logic, a routing change that shifted traffic onto different hardware (e.g., a different GPU generation, or a different quantization/precision configuration for that hardware pool) than before, or a subtle change in how KV-cache reuse/prefix-caching interacts with request batching under a specific new load pattern. The mechanics of these serving-stack components are covered in `..\08_Inference_And_Serving_Systems\`; the incident-response question here is how to implicate or rule this hypothesis out efficiently.

**How to confirm or rule out.** Check the infrastructure change log for the exact regression window identified in Step 1 — deploys, config changes, autoscaling/routing changes, hardware-pool composition changes — independent of whether anyone flagged those changes as "model-affecting" at the time (this is exactly the kind of change that looks unrelated to model quality from the deploying team's perspective and is precisely why a cross-functional change log, not just the model-training team's own deploy history, has to be part of this investigation). If a serving-stack-only reproduction is possible — running the *same* model weights through the *previous* serving configuration versus the *current* one, on the same prompt set — a measurable difference between the two directly confirms this hypothesis with no ambiguity.

## Step 2b: What Counts as Confirming Evidence, Tabulated

| Evidence | Points toward |
|---|---|
| Infra/deploy change log shows a change at the exact regression-onset time | A |
| Serving-config diff shows a precision/quantization change | B |
| Task-targeted re-run shows a gap that an aggregate benchmark re-run doesn't | B specifically (aggregate-hiding signature) |
| Prompt-distribution comparison shows a shift in structure/source correlated with the window | C |
| Regression concentrated in one experiment arm or user segment, not uniform | D |
| Regression uniform across all traffic, all segments, all quantization configs | Push back toward A, or reconsider whether Step 1's "regression is real" confirmation was solid |

## Step 3: Hypothesis B — Silent Quantization or Configuration Change

**The mechanism.** A specific, common variant of Hypothesis A worth treating separately because its investigation is distinct: somewhere between the trained checkpoint and the production-serving instance, a precision or quantization decision may have changed — a shift from BF16 to a lower-precision serving format (INT8/INT4/FP8 post-training quantization) for cost reasons, a change in quantization calibration data or method, or a change in which layers/operations are quantized versus kept in higher precision — without an accompanying change in the *reported* model version, because from a release-versioning perspective "same weights, different serving precision" is easy to treat as a serving-team-only decision that doesn't trigger the model-quality re-evaluation a genuine weight update would. Quantization-induced quality regressions are frequently **task-specific and disproportionate** relative to their impact on aggregate benchmark scores — a quantization scheme validated against a broad benchmark average can still measurably degrade a narrower skill (precise multi-step arithmetic, exact-format code generation, long-chain-of-thought coherence) that happens to be more sensitive to the specific precision loss involved, which is exactly consistent with a complaint concentrated on "one specific task" rather than a broad, even degradation.

**How to confirm or rule out.** Check the exact serving configuration's quantization/precision settings for the regression window against before; if a quantization change occurred, re-run the offline reproduction set (Step 1) against both the pre- and post-change quantization configuration directly, specifically targeting the complained-about task category, since a broad benchmark re-run might show negligible aggregate difference while a task-targeted re-run reveals the concentrated regression.

## Step 3b: Quantization Regression — A Quick FAQ

- **Why would a quantization change pass aggregate validation but still cause this complaint?** Because aggregate benchmark averages can absorb a concentrated regression in one task category without moving much, exactly the same "aggregate hides a targeted regression" pattern discussed for safety classifiers in `012_Interview_Questions_Part2.md`, Q10.
- **Why is code generation specifically vulnerable?** Multi-step, exact-format tasks (precise syntax, exact indentation, multi-step arithmetic) are more sensitive to reduced numeric precision than free-form prose generation, where small perturbations rarely change the "correctness" of an answer.
- **How do you tell this apart from Hypothesis A (a broader infra regression)?** Hypothesis B's signature is *task-specific concentration*; a broad infra regression (Hypothesis A) more often affects a wide range of task categories roughly proportionally, not one narrow category disproportionately.
- **What's the long-term fix, not just the immediate rollback?** Add a task-targeted validation step to the quantization-change approval process itself, so future precision changes are checked against the specific task categories most likely to be sensitive before they ship, not only against an aggregate suite.

## Step 4: Hypothesis C — Real-World Query-Distribution Shift

**The mechanism.** This is the hypothesis most likely to be under-weighted because it's the least "someone made a mistake" of the four, and it's also the one most specific to the "no model update" premise: the model's weights, serving stack, and configuration are all genuinely unchanged, but the *population of queries it's being asked* has shifted — a new product surface started routing traffic to this model, a seasonal or news-driven shift changed what users are asking about, a change in an upstream product feature altered the typical prompt structure/length reaching the model, or (for a coding-assistant use case specifically) a new, more complex class of codebase or task started being routed through this endpoint. A model that was never actually worse at the task in any absolute sense can appear to have "gotten worse" purely because the task distribution it's now being evaluated against (by real users, informally) has become harder or has shifted outside the region where the model was strongest.

**How to confirm or rule out.** Compare the input-prompt distribution (length, structure, task-subtype composition if classifiable, source/product-surface tagging if available) for the regression window against the weeks before — a measurable shift in prompt characteristics correlated with the complaint window is direct evidence for this hypothesis. A clean, decisive test: take the Step 1 offline reproduction set (representative of the *historical* task distribution) and confirm it still performs at the historical baseline level on the unchanged model — if it does, while real production traffic in the same task category shows degraded outcomes, the discrepancy is strong evidence that the *task itself* (as currently being posed by real users) has shifted, not the model's capability on the task as previously characterized.

## Step 4b: A Quick FAQ on Query-Distribution Shift

- **How do you avoid this hypothesis becoming a catch-all excuse for any unexplained regression?** By requiring the same standard of evidence as every other hypothesis — a measurable, quantified shift in the input-prompt distribution correlated with the regression window, not just an absence of findings elsewhere used as a default explanation by elimination.
- **What's the right organizational response once this is confirmed?** Feed the newly-revealed harder distribution back into the training and evaluation pipelines as a concrete input (per Step 8's resolution discussion) — this is a product/data signal, not an infrastructure bug, and treating it as the latter wastes effort looking for a revert target that doesn't exist.
- **Can this hypothesis and Hypothesis A/B/D be true simultaneously?** Yes — a genuine distribution shift can coincide with an unrelated infra regression, and Step 1's careful, hypothesis-by-hypothesis confirmation discipline is exactly what prevents a real infra fix from being wrongly credited for a recovery that was actually just the distribution shifting back, or vice versa.

## Step 5: Hypothesis D — A/B Test or Experiment Mis-Routing

**The mechanism.** If any experimentation infrastructure is in play — a staged rollout of a different model version, a feature-flagged serving-configuration experiment, a routing experiment testing a different prompt-template or system-prompt variant — a bug in the traffic-splitting/assignment logic can silently route a larger-than-intended fraction of traffic (or a specific user segment, correlated with whoever happens to be reporting the complaint) into a control or experimental arm that is not the intended production configuration. This is a distinct hypothesis from Hypothesis A specifically because the *intended* production configuration may be entirely correct — the bug is in which configuration actual users are being assigned to, not in any configuration itself being wrong.

**How to confirm or rule out.** Audit the experiment/feature-flag assignment logs for the regression window, specifically checking whether the observed complaint pattern correlates with a specific experiment arm, user segment, or routing rule rather than being uniform across all production traffic — a regression concentrated in a specific segment (a specific client version, a specific geographic region tied to a specific serving cluster, a specific subset of users who happen to have been bucketed into an active experiment) is a strong, specific signature of mis-routing that a uniform, all-traffic regression would not produce. This is exactly the kind of check that Step 1's "nail down the exact regression window and its scope" groundwork makes tractable — without knowing whether the regression is uniform or segmented, this hypothesis can't be efficiently distinguished from the others.

## Step 5b: A Quick FAQ on Mis-Routing Incidents

- **How is this different from a straightforward bug report?** The distinguishing feature is that the *intended* configuration in every arm may be entirely correct — the bug is purely in which users get assigned to which arm, a routing/bucketing defect rather than a configuration-content defect, which is why it needs its own dedicated audit rather than being folded into Hypothesis A or B's change-log check.
- **What's the fastest way to confirm or rule this out?** Segment the regression by every dimension experiment/flag assignment could plausibly correlate with (client version, region, user cohort, active-experiment membership) — a clean, non-uniform segmentation pattern is close to definitive evidence, and a uniform, uncorrelated pattern rules it out quickly.
- **Once confirmed, does fixing the routing bug fully resolve the incident?** Only for future traffic — any users who received the wrong configuration during the affected window may need a separate remediation/communication decision, which is a distinct follow-up from the routing fix itself.

## Step 5c: A Quick FAQ on Investigation Efficiency

- **Why check infra/config change logs and routing logs before the more "interesting" distribution-shift hypothesis?** Because they're cheap, fast, and empirically the more common real-world cause — starting with the analytically interesting hypothesis when a five-minute log check could have resolved the incident is a classic way to waste hours.
- **What's the risk of skipping the offline-reproduction-set step and going straight to log audits?** Without a fixed reproduction set, even a confirmed log-audit finding (e.g., "yes, a config changed") can't be definitively linked to the reported regression — the reproduction set is what turns a correlated timing coincidence into an actual confirmed causal link.
- **How do you know when to stop investigating and declare a hypothesis confirmed?** When the fix, applied and validated against the offline reproduction set, actually closes the gap — a hypothesis is "confirmed" in the sense that matters only once its predicted fix demonstrably restores the expected behavior, not merely because the evidence seems consistent with it.

## Step 6: An Efficient Investigation Order, Not a Parallel Sweep of All Four

Given limited investigator time, sequence the checks by cost and prior likelihood rather than running all four simultaneously with divided attention:

1. **Check the infrastructure/deploy/config change log first** (Hypotheses A and B) — this is the cheapest check (pulling a log, no data analysis required) and, in practice, the most common actual root cause of "the model seems different but we didn't touch the model," because serving-stack changes are frequent, numerous, and not always recognized by their authors as model-quality-relevant.
2. **Check experiment/routing assignment logs second** (Hypothesis D) — similarly cheap (a log/config audit) and a common enough real-world cause (especially in orgs running many concurrent experiments) to check early, before investing in the more analytically involved distribution-shift investigation.
3. **If both come back clean, invest in the query-distribution-shift analysis** (Hypothesis C) — this requires actual data analysis (characterizing and comparing prompt distributions across time windows) and is appropriately the more expensive, later step given it's checked only once the cheaper log-audits have failed to explain the regression.
4. **Throughout, maintain the offline reproduction set from Step 1 as the constant reference point** — every hypothesis's confirmation strategy above routes back through "does re-running this fixed, curated prompt set under configuration X show the regression," which is the single most efficient tool available precisely because it holds everything except the variable under test constant.

## Step 6b: A Summary Table of the Whole Investigation

| Step | Action | Typical time cost | Rules out / confirms |
|---|---|---|---|
| 1 | Convert complaint into measured, reproducible regression | Hours | Confirms the regression is real |
| 2/3 (parallel) | Audit infra/deploy and quantization-config change logs | Minutes | Hypotheses A, B |
| 5 | Audit experiment/routing assignment logs | Minutes | Hypothesis D |
| 4 | Compare prompt distributions across time windows | Hours | Hypothesis C |
| 8 | Apply fix, re-validate against offline reproduction set | Varies by fix | Confirms resolution |

## Step 6c: A Quick FAQ on Hypothesis Prioritization

- **Does the order in which the four hypotheses are presented in this file reflect their real-world frequency?** Roughly, based on general field experience — infra/config changes (A/B) are checked first partly because they're cheap to check, but also because they are, in practice, a very common real cause of "the model seems different but nothing was touched at the model layer."
- **What if the organization has very little experimentation infrastructure (Hypothesis D essentially doesn't apply)?** Skip straight from the A/B log audits to the distribution-shift analysis — the framework's value is in its ordering logic, not in mechanically working through every hypothesis regardless of whether it's structurally possible in this specific environment.
- **Is it possible for the root cause to be a combination of two hypotheses at once?** Yes, and this is exactly why Step 1's fixed offline reproduction set matters — it lets each hypothesis be tested independently even when more than one factor is contributing simultaneously.

## Step 7: Writing the Monitoring Check That Would Have Caught This Earlier

A strong answer should close by proposing the concrete monitoring that would have caught this regression closer to when it started, rather than relying on user complaints as the detection mechanism — a staff-level response treats "we found out from user complaints a week later" as itself a finding, not just background. A sketch of the kind of standing check this argues for:

```python
# Sketch: a scheduled job comparing live production outputs against a fixed,
# versioned offline regression suite, run on a recurring cadence (e.g., hourly)
# and alerting on statistically significant degradation, per task category.

def run_regression_check(task_category: str, prod_endpoint, reference_scores: dict):
    prompts = load_versioned_eval_set(task_category)          # fixed, curated, versioned
    outputs = [prod_endpoint.generate(p) for p in prompts]
    scores = score_outputs(outputs, task_category)             # automated grader / verifier

    baseline_mean, baseline_std = reference_scores[task_category]
    current_mean = mean(scores)

    z = (current_mean - baseline_mean) / (baseline_std + 1e-6)
    if z < -SIGNIFICANCE_THRESHOLD:
        alert(
            task_category=task_category,
            current_mean=current_mean,
            baseline_mean=baseline_mean,
            z_score=z,
            # attach the concurrent infra/config/experiment change log window
            # automatically, so the on-call engineer starts Step 6 pre-populated
            recent_changes=fetch_change_log(window="last_24h"),
        )

# Run this per task-category (code generation, multi-step instruction-following,
# etc.) on a fixed cadence against production, not just at model-launch time —
# the whole point is continuous post-launch monitoring, not a one-time gate.
```

The design choices worth calling out explicitly in an interview: the check runs against a **fixed, versioned prompt set** (so a detected change is attributable to the system, not to prompt-set drift), it runs **continuously post-launch**, not just at launch-gating time (directly closing the gap this entire scenario exposes), it's **scoped per task category** rather than one aggregate score (because, per Hypothesis B, a task-specific regression can hide inside a healthy aggregate average), and it **automatically attaches the concurrent change log** to any alert, which collapses Step 6's investigation-ordering discipline into the alert itself rather than requiring an on-call engineer to remember to go pull that log manually during a live incident.

## Step 7b: A Quick-Reference Table Across the Four Hypotheses

| | A: Serving-infra regression | B: Silent quantization/config change | C: Query-distribution shift | D: A/B mis-routing |
|---|---|---|---|---|
| Is anything about the model's weights different? | No | No (same weights, different serving precision) | No | No |
| Cheapest confirming check | Infra/deploy change-log audit | Quantization-config change-log audit + task-targeted re-run | Prompt-distribution comparison across time windows | Experiment/flag assignment-log audit |
| Typical scope signature | Often broad/uniform across traffic | Concentrated on precision-sensitive tasks, hides in aggregate averages | Correlates with a new traffic source/product surface | Concentrated on a specific segment/experiment arm |
| Fix | Revert the change | Revert or re-calibrate quantization | Not a revert — treat as new product/training input | Fix routing bug; audit affected users |

## Step 7c: A Worked Narrative

- **T+0 (retrospective):** support tickets start mentioning "the coding assistant gives worse suggestions than last week" — a handful of reports over several days, easy to dismiss as anecdotal at first.
- **T+1 day:** on-call pulls task-success telemetry for the code-generation category specifically; a real, statistically distinguishable drop is confirmed starting roughly 6 days earlier — this converts the complaint into a scoped incident.
- **T+1 day, +2 hours:** the offline reproduction set (a fixed, curated code-generation prompt set) is run against current production; it reproduces the regression cleanly, ruling out "this is just a perception/reporting artifact."
- **T+1 day, +3 hours:** infra/deploy change log is checked for the 6-day window; a quantization-config change is found — a cost-driven move from BF16 to INT8 serving for this model pool, deployed 6 days ago, that was validated against an aggregate benchmark suite but not against a task-targeted code-generation check.
- **T+1 day, +4 hours:** the same offline reproduction set is re-run against both the pre-change (BF16) and post-change (INT8) configurations directly; the INT8 configuration shows a concentrated regression specifically on multi-step, exact-format code tasks, while the aggregate benchmark suite it was originally validated against shows negligible difference — confirming Hypothesis B precisely, and explaining why the original validation missed it.
- **T+1 day, +6 hours:** decision made to revert the affected model pool to BF16 for code-generation traffic specifically, while quantization work continues on a task-targeted recalibration before attempting the cost optimization again.
- **Follow-up:** the standing per-task-category latency/quality monitor (Step 7's code sketch) is deployed specifically so any future quantization or config change is caught by an automated check within hours, not by user complaints a week later.

## Step 7d: Why This Question Recurs at the Staff Level

This scenario is a favorite because the premise itself does most of the filtering work: a candidate who immediately reaches for "something's wrong with the model" despite being told explicitly that nothing changed has revealed they didn't actually process the constraint, which is a meaningfully bigger tell than getting any individual hypothesis wrong. The strongest answers treat the premise as informative rather than as a minor detail to work around, and use it to actively prune the hypothesis space from the very first sentence of the response.

## Step 8: Resolution and Rollback

Once root-caused, the fix is usually mechanical relative to the diagnosis effort: revert the offending infrastructure/config change (Hypothesis A/B), fix the routing/assignment bug and re-audit which users were affected for any necessary follow-up communication (Hypothesis D), or — for Hypothesis C — recognize that there may be no "fix" in the sense of reverting anything, because nothing regressed; the appropriate response is instead to treat the newly-revealed harder query distribution as a genuine product/training input (a candidate to add to the next post-training data mixture or the next evaluation suite's coverage, closing the loop back into `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md` and `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md` respectively) rather than an incident to be resolved by rolling anything back at all. Correctly identifying which of these two very different resolution paths applies — revert a mistake, versus recognize a genuine shift in what the product now needs to handle well — is itself part of the staff-level judgment this scenario is testing.
