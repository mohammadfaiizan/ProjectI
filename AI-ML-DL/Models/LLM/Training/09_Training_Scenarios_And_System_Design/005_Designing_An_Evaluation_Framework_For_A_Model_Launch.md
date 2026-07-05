# Designing An Evaluation Framework For A Model Launch

## Step -2: A Quick FAQ on Scope

- **Does this framework apply the same way to an incremental model update as to a brand-new frontier release?** The layers are the same, but the regression-gate comparisons in Layer 1 become the dominant signal for an incremental update (since there's an existing production baseline to compare against), while Layer 3's dangerous-capability evaluation matters most for a genuinely new capability tier.
- **Who has final authority to override a Tier 1 hard blocker?** By design, no one — this is exactly what makes it a hard blocker rather than a Tier 2 negotiable one; if an organization finds itself routinely wanting to override a specific Tier 1 gate, that's a signal the gate was mis-tiered, not that an override mechanism is needed.
- **How does this framework interact with a Responsible Scaling Policy's pre-committed capability thresholds?** Directly — Layer 3's dangerous-capability sub-layer *is* the operational implementation of an RSP's thresholds (`..\07_Safety_Alignment_And_Responsible_Scaling\001_Responsible_Scaling_Policies_And_Preparedness_Frameworks.md`), not a separate, parallel process.

## Step -1: Why This Should Be Designed Before Any Result Exists

The single most important discipline this scenario tests is sequencing: a launch-gating framework only functions as a real gate if its thresholds are set before the model's actual scores are known. A team that builds "the eval framework" by running whatever benchmarks come to mind and then, after seeing the results, deciding retroactively which ones matter enough to block on has built a rationalization process, not a gate — and this distinction, stated explicitly and early, is worth leading with before walking through the six layers below, because every one of those layers is only as trustworthy as this sequencing discipline being genuinely honored under real organizational pressure to ship on a promised date.

## The Scenario

"You're about to ship a new frontier model. Design the full evaluation framework that gates the go/no-go launch decision — not just 'what benchmarks do you run,' but how the results of those benchmarks actually translate into a launch decision."

This is a system-design question about a decision process, not a list-the-benchmarks question, and interviewers will notice immediately if you answer with a flat list of eval names rather than a layered framework with explicit gating logic. I'll build this as a set of layers, each answering a different question ("can it do the task," "will it cause harm," "do people actually like using it," "did we cheat," "does it work in a real multi-step setting"), and then show how those layers combine into an actual go/no-go decision structure, closing with how this connects to a Responsible Scaling Policy framework.

## Step 0a: The Six Layers at a Glance

Before working through each layer in depth, here is the full structure as a single reference table, since a strong answer should be able to reproduce this shape on a whiteboard before diving into any one row's mechanics:

| Layer | Question answered | Example concrete checks | Typical gate tier |
|---|---|---|---|
| 1. Capability benchmarks | Can it do the task, and is it better than what's shipped? | MMLU, GPQA, GSM8K/MATH, HumanEval/SWE-bench | 1-2 |
| 2. Contamination checks | Are the Layer 1 numbers trustworthy? | N-gram overlap audit, fresh held-out variants | 1 |
| 3. Safety / red-team | Will it cause harm? | Automated harm benchmarks, red-teaming, dangerous-capability thresholds, refusal/over-refusal balance | 1-2 |
| 4. Human preference | Do people actually prefer using it? | Blind side-by-side comparisons, staged A/B rollout, latency/cost | 2-3 |
| 5. Agentic/trajectory | Does it work across multi-step tool use? | End-to-end task-completion rate, failure-mode taxonomy, cost-per-success | 2-3 |
| 6. Go/no-go synthesis | How do all the above combine into a decision? | Tiered gate structure, documented exceptions | — |

## Step 0d: Why This Question Recurs So Often at the Staff Level

Much like the compute-estimation question in `001_Estimating_Training_Compute_And_Cost.md`, this question compresses several distinct competencies: systems-design thinking (can you structure an open-ended requirement into layers with clear responsibilities), organizational realism (do you understand why pre-registration matters and how gate-gaming actually happens under real launch-timeline pressure), and safety judgment (do you correctly treat dangerous-capability findings as categorically different from a benchmark regression, rather than flattening every finding into the same negotiable-severity bucket). A candidate who only lists benchmark names has answered a much narrower and less interesting question than the one actually being asked.

## Step 0: What Is a "Launch Gate," Mechanically?

Before designing the layers, define the mechanism precisely: a launch gate is a **predetermined, pre-registered threshold or condition on a specific evaluation result, decided before the model's actual scores are known**, such that the launch decision is a function of the evaluation outcome rather than the evaluation being interpreted after the fact to justify a launch decision already made on other grounds. This ordering — thresholds set before results are in — is the entire point of having a framework at all; a launch-gating process that lets teams see the numbers first and then decide whether those numbers are "good enough" is not actually a gate, it's a rationalization process, and a staff-level answer should name this distinction explicitly, because it's exactly the failure mode organizational incentives push toward under launch-timeline pressure.

## Layer 1: Capability Benchmarks — "Can It Do the Task?"

This layer answers the most basic launch-readiness question: is the model at least as capable as its predecessor and competitive with the frontier, on the tasks it will actually be used for. The mechanics of specific benchmark suites (MMLU, GPQA, GSM8K/MATH, HumanEval/LiveCodeBench/SWE-bench, and domain-specific suites) are covered in `..\06_Benchmarks\`; what belongs in a launch-gating design is:

- **A fixed, versioned benchmark suite decided before the run, not cherry-picked afterward.** If a model underperforms on a benchmark that was part of the pre-registered suite, that's a real data point the framework has to reckon with — not a benchmark that quietly gets dropped from the launch report because the number was disappointing. Conversely, a benchmark that wasn't pre-registered and happens to look great post-hoc is weak evidence for a launch decision precisely because of selection bias (see also `010_Critiquing_Real_Published_Training_Recipes.md`'s discussion of self-reported benchmark numbers).
- **Regression gates against the current production model, not just absolute thresholds.** A launch gate that only asks "is MMLU above 85%" misses the case where the new model is *worse* than what's currently shipped, on a task real users depend on, even while clearing an absolute bar. Both an absolute floor (the model must be competitive with the frontier) and a regression floor (the model must not meaningfully regress from the currently-shipped model on any pre-registered benchmark in the "who currently uses this for X" category) belong in the gate.
- **Distinguish "headline" benchmarks the gate actually blocks on from "informational" benchmarks that are tracked but don't block.** Not every benchmark result should have launch-blocking authority — over-gating on too many metrics either produces gate paralysis (nothing ever launches) or, worse, produces incentive to game whichever handful of benchmarks people know are gates while ignoring everything else. A well-designed framework names a small number (5-10) of headline capability gates explicitly and tracks a much broader dashboard of informational metrics that inform judgment without unilaterally blocking.

## Layer 1b: Example Headline vs. Informational Benchmarks

| Benchmark | Typical tier | Why |
|---|---|---|
| MMLU-Pro / GPQA | Headline (Tier 1-2 gate) | Broad capability signal directly relevant to most deployments |
| GSM8K/MATH | Headline if reasoning is a target capability | Directly measures a capability the model is explicitly meant to have |
| A narrow, single-domain internal benchmark | Informational (Tier 3) | Useful signal, but not broad enough to unilaterally block a launch |
| SWE-bench Verified / agentic task suites | Headline if agentic use is a target deployment | Directly measures Layer 5's concern, not just Layer 1's |
| A benchmark under active contamination investigation (Layer 2) | Demoted to informational until resolved | Untrustworthy numbers shouldn't carry gating authority |

## Layer 2: Contamination Checks — "Are These Numbers Even Real?"

This layer has to run *before* the capability numbers in Layer 1 are trusted, not after — a headline benchmark score is worthless as a launch-gating input if the eval set (or a close paraphrase of it) was present in the training corpus, and this is exactly the failure mode `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md` Step 6 is designed to prevent at the data-pipeline stage. At launch-gating time, the framework needs an independent, post-hoc contamination audit as a check on the pipeline stage having actually worked:

- **N-gram/embedding-similarity overlap analysis** between the eval suite and the training corpus, run as a dedicated audit rather than assumed-clean because the upstream pipeline claimed to screen for it — pipelines have bugs, and a launch-gating framework's job is specifically to not trust its own upstream stages blindly.
- **Canary/held-out variant testing** — where feasible, evaluate against a freshly-constructed or paraphrased variant of a benchmark that could not have been in the training corpus (because it postdates the training cutoff, or was deliberately withheld/rewritten) and compare the score against the standard benchmark's score; a large gap between the two is direct evidence of either contamination or overfitting to a specific benchmark's surface form, either of which invalidates using the standard score as a launch input.
- **Documenting a contamination report per benchmark**, not a binary clean/dirty flag — exactly as recommended for the data-pipeline stage, the launch-gating team needs the granularity to decide whether a specific benchmark's headline number should be trusted, discounted, or excluded from Layer 1's gates entirely.

## Step 1b: A Quick FAQ on Gate Design

- **Why not just use one aggregate "launch readiness score"?** Because collapsing capability, safety, and product-quality signals into one number destroys exactly the information a real decision needs — whether a low score reflects a dangerous-capability crossing (must block) or a minor benchmark regression (negotiable) is the whole point of tiering, and an aggregate score erases that distinction.
- **Who decides which tier a new finding belongs to?** This should itself be pre-registered — a defined mapping from finding-severity categories to gate tiers, decided before any specific launch's findings exist, exactly mirroring the "thresholds set before results are known" discipline from Step 0.
- **What if a genuinely new category of finding shows up that wasn't anticipated in the pre-registered tiering?** This is exactly what the Tier 2 exception process (Step 6) is for — a named owner makes an explicit, documented call, rather than either blocking automatically on an unanticipated finding or silently ignoring it.

## Layer 3: Safety and Red-Team Evaluation — "Will It Cause Harm?"

This is where the eval framework connects most directly to `..\07_Safety_Alignment_And_Responsible_Scaling\001_Responsible_Scaling_Policies_And_Preparedness_Frameworks.md` and to the broader safety-evaluation methodology in `..\05_Evaluation_Methods\`. The layer itself has several distinct sub-components that shouldn't be collapsed into one generic "safety eval" line item:

- **Automated harm-category benchmarks** — standardized tests for categories like disallowed-content generation, bias/toxicity, and jailbreak-susceptibility, run at scale and cheaply, providing broad but comparatively shallow coverage.
- **Structured red-teaming** — human (and increasingly AI-assisted) adversarial probing specifically aimed at finding failure modes the automated suite wouldn't surface, covering both known harm categories (probed more creatively/adversarially than an automated benchmark can) and open-ended discovery of genuinely novel failure modes. This needs a defined process: a red-team engagement with a scoped time window, a defined reporting structure for findings, and — critically — a defined process for what happens to a finding (does it block launch, does it require a specific mitigation to be verified before launch, does it get logged as a known limitation to disclose).
- **Dangerous-capability evaluations** — the category most directly tied to Responsible Scaling Policy / Preparedness Framework thresholds: does the model demonstrate capability uplift in specific high-consequence domains (biological, chemical, cyber, or autonomous-replication-relevant capabilities, depending on the lab's specific framework). This sub-layer is qualitatively different from the others because its gating logic is typically **binary and non-negotiable at the threshold** — crossing a defined dangerous-capability threshold triggers a mandatory mitigation requirement (enhanced safeguards, restricted deployment, or in the most severe defined tiers, a launch block until mitigations are verified) rather than being weighed against other considerations the way a capability-benchmark regression might be.
- **Refusal/over-refusal balance evaluation** — a launch-blocking safety eval framework has to check both directions: does the model refuse things it should refuse (under-refusal, a safety failure) and does it refuse things it shouldn't (over-refusal, a genuine capability/usability regression that a naive "just make it refuse more" mitigation to a red-team finding can silently introduce). A framework that only measures under-refusal will systematically reward increasingly evasive models, exactly the failure mode Constitutional AI's original evaluation explicitly tracked and reported against (`..\..\Claude\008_Constitutional_AI_And_RLAIF.md`, Section 10) — non-evasiveness under comparable harmlessness was treated as a first-class metric, not an afterthought.

## Layer 3b: A Quick FAQ on Safety-Layer Design

- **Why separate automated benchmarks, red-teaming, and dangerous-capability evaluation into distinct sub-layers instead of one "safety score"?** Because they have different cost profiles, different lead-time requirements, and — critically — different gating logic (dangerous-capability findings are non-negotiable; automated-benchmark findings are more often negotiable), and collapsing them loses exactly the information the gating structure in Step 6 depends on.
- **How do you avoid the refusal/over-refusal balance becoming just another vague "be helpful but safe" aspiration?** By measuring both directions with equally concrete, equally pre-registered benchmarks — an under-refusal benchmark and an over-refusal benchmark, both with headline-tier status, not one measured rigorously and the other assessed informally.
- **What's the actual cost of red-teaming, roughly?** Meaningfully higher in calendar time (multi-week engagement windows) than pure compute cost — the binding constraint is skilled human red-teamer time and the review/triage process for findings, not GPU-hours.

## Layer 4: Human Preference Testing — "Do People Actually Like Using It?"

Capability benchmarks and safety evals both measure something narrower than "is this a good product experience," and this layer exists specifically to close that gap:

- **Side-by-side human preference comparisons** against the current production model and, where relevant, competitor models, on a realistic distribution of actual usage prompts (not just benchmark-style prompts) — this is the same mechanism underlying reward-model training (`006_Responding_To_A_Reward_Hacking_Incident.md` in this folder covers the failure modes of over-relying on this signal during training; at launch-gating time the equivalent risk is over-relying on a narrow rater pool's preferences as a proxy for the full deployed user base's experience).
- **A/B testing in a controlled rollout** (covered operationally in `009_Post_Launch_Model_Degradation_And_Incident_Response.md`'s discussion of rollout mechanics) as a pre-full-launch gate: a staged rollout to a small traffic percentage, with real usage-pattern telemetry (task success signals, regeneration/edit rates, explicit feedback where available) compared against the currently-shipped model, is a stronger launch-readiness signal than any offline human-preference study alone, precisely because it captures the actual query distribution rather than a curated eval prompt set.
- **Latency and cost as launch-gating dimensions in their own right**, not purely capability-adjacent afterthoughts — a model that scores higher on every capability benchmark but is materially slower or more expensive to serve than its predecessor (see `..\08_Inference_And_Serving_Systems\`) can still fail a launch gate on product grounds, and a well-designed framework makes this an explicit, pre-registered gate rather than a post-hoc surprise discovered after capability sign-off has already happened.

## Layer 4b: A Quick FAQ on Human Preference and A/B Testing

- **Why is a staged A/B rollout a stronger signal than an offline preference study alone?** Because it captures the actual production query distribution rather than a curated eval prompt set, and real usage-pattern telemetry (regeneration/edit rates, task-success signals) surfaces issues an offline study's necessarily-limited prompt sample can miss entirely.
- **How small can the initial canary percentage safely be?** Small enough to bound blast radius, large enough for statistical power within a reasonable window — the specific number is a function of baseline traffic volume and the minimum effect size worth detecting, and should be computed explicitly (per `012_Interview_Questions_Part2.md`, Q6's confidence-interval discipline) rather than picked by convention.
- **What if the A/B rollout shows a regression the offline evals never flagged?** This is exactly the scenario this layer exists to catch — treat it as a genuine launch-blocking finding requiring root-cause investigation, not as noise to be waited out until it resolves on its own.

## Layer 5: Agentic / Trajectory Evaluation — "Does It Work Across Multiple Steps?"

For any model expected to be used in agentic, tool-using, or multi-turn task-completion settings (increasingly the default expectation for a frontier model, not a niche use case), single-turn benchmark performance is a necessary but insufficient signal, because failure modes specific to extended trajectories — compounding errors across steps, losing track of long-horizon context, getting stuck in unproductive loops, mishandling tool-call errors — don't show up in single-shot evaluation at all:

- **End-to-end task-completion rate on realistic multi-step benchmarks** (software-engineering task suites, browsing/research tasks, multi-tool workflows), measuring whether the *entire trajectory* succeeds, not whether individual steps look locally reasonable.
- **Failure-mode taxonomy specific to trajectories** — does the model recover from a failed tool call or does it compound the error; does it know when to stop and ask for clarification versus confidently proceeding on a wrong assumption; does performance degrade as trajectory length grows in a way that a fixed per-step accuracy number would hide entirely.
- **Cost-per-successful-task, not just success rate** — an agentic evaluation that only reports success rate can reward a model that succeeds slowly and expensively (many redundant tool calls, long deliberation) exactly as much as one that succeeds efficiently; for a launch gate that's meant to inform a real deployment decision, the cost dimension has to be part of the reported metric, not a separate afterthought analysis.

## Layer 6: Tying It Together — The Go/No-Go Decision Framework

This is the part most answers skip, and it's the part that actually answers the scenario's question. A concrete structure:

```
Gate tier 1 (hard blockers — any failure here blocks launch unconditionally):
  - Dangerous-capability threshold crossed without verified mitigation (Layer 3)
  - Contamination audit reveals the headline capability claims are not trustworthy (Layer 2)
  - Regression on a pre-registered safety benchmark below the current production model's score (Layer 3)

Gate tier 2 (negotiable blockers — failure here requires an explicit sign-off exception,
              not an automatic launch, and the exception itself must be documented and owned):
  - Regression on a pre-registered capability benchmark relative to production (Layer 1)
  - Red-team findings in a defined-severity band without a verified mitigation (Layer 3)
  - A/B rollout shows a statistically significant negative shift in a core product metric (Layer 4)

Gate tier 3 (informational — tracked, reported in the launch review, does not block by itself):
  - Non-headline capability benchmarks (Layer 1)
  - Broader human-preference margins outside the specific pre-registered comparison (Layer 4)
  - Agentic trajectory metrics outside the specific pre-registered task suite (Layer 5)
```

The mechanism that makes this a real decision framework rather than a spreadsheet of numbers: **tier 1 failures are launch-blocking by construction, with no exception path** — this is the direct implementation of a Responsible Scaling Policy's binding-commitment structure (`..\07_Safety_Alignment_And_Responsible_Scaling\001_Responsible_Scaling_Policies_And_Preparedness_Frameworks.md`), where crossing certain capability thresholds is defined, in advance, to require specific verified mitigations before launch can proceed at all, independent of how good the model is on every other axis. **Tier 2 failures require an explicit, named decision-maker to sign off on an exception**, with the exception and its rationale documented as part of the launch record — this is what prevents "the benchmark looked bad so we just didn't mention it in the launch review" from being a viable path; the finding still has to be surfaced and someone with the authority to accept the risk has to do so explicitly and on the record. **Tier 3 metrics inform judgment without mechanically blocking anything**, which is what keeps the framework from collapsing into gate paralysis over every possible metric.

## Step 6d: A Pre-Launch Checklist, Usable as an Actual Sign-Off Document

- [ ] Every Layer 1 headline capability benchmark has been run against the final candidate checkpoint and compared against both an absolute floor and the current production model.
- [ ] Layer 2's contamination audit has completed against every benchmark in the headline suite, with a per-benchmark overlap report attached, not a binary pass/fail.
- [ ] Layer 3's dangerous-capability evaluation has completed with enough lead time that any triggered mitigation has already been built and re-verified, not merely scheduled.
- [ ] Layer 3's red-team engagement has run its full planned window, and every finding has been triaged into a named gate tier with an owner.
- [ ] Layer 3's refusal/over-refusal balance has been measured explicitly, not inferred from the under-refusal numbers alone.
- [ ] Layer 4's offline human-preference comparison and staged A/B rollout have both completed, with the A/B rollout's sample size/duration meeting a pre-registered statistical-power bar (per `012_Interview_Questions_Part2.md`, Q6).
- [ ] Layer 5's agentic/trajectory evaluation has run against the same final candidate checkpoint, if the model is expected to see agentic/tool-use deployment.
- [ ] Every Tier 2 exception has a named, documented sign-off, and every Tier 1 finding has either cleared or is actively blocking launch with no exception path.

## Layer 5b: A Quick FAQ on Agentic Evaluation

- **Why is single-turn benchmark performance insufficient for an agentic model specifically?** Because failure modes like compounding errors across steps, unproductive loops, and mishandled tool-call errors structurally cannot manifest in a single-shot evaluation — they are properties of a trajectory, not of any individual step in isolation.
- **Should cost-per-successful-task be a launch-blocking metric or purely informational?** It depends on the deployment's cost sensitivity, but it should at minimum be reported alongside success rate, never omitted — a success-rate-only report can silently reward an expensive, inefficient success path over a cheap, efficient one.
- **How mature does agentic evaluation tooling need to be before it's trustworthy as a launch gate?** Mature enough that a failure-mode taxonomy (not just a pass/fail rate) is being tracked — an aggregate success-rate number alone, without breakdown by failure type, is not yet a sufficiently informative signal for a launch decision on an agentic-capable model.

## Step 7: Sequencing the Evaluation Campaign Against the Launch Timeline

A staff-level answer should also address *when* each layer runs, because running everything only in the final week before a planned launch date is a common, avoidable failure:

1. **Layers 1 (capability) and 2 (contamination)** can and should start as soon as a stable, near-final checkpoint exists — often well before post-training is fully complete, against intermediate checkpoints, specifically to catch capability or contamination surprises with enough lead time to act on them.
2. **Layer 3's automated safety benchmarks** run in parallel with Layer 1; **structured red-teaming**, being human-labor-intensive and needing real engagement time, should be scoped and scheduled with a fixed multi-week window that starts as early as a red-team-ready checkpoint exists, not compressed into the final days before launch under timeline pressure — red-team findings that surface a mitigation requirement need lead time for that mitigation to actually be built and re-verified, and a compressed red-team window structurally forecloses that lead time.
3. **Dangerous-capability evaluations**, given their binding, non-negotiable gating status, need to run early enough that a positive finding (crossing a threshold) can be acted on — building and verifying the required mitigation — well before the planned launch date, not discovered as a launch-week surprise.
4. **Layer 4's A/B rollout** is necessarily one of the last steps, since it requires a near-final model, but the offline human-preference comparisons within Layer 4 can run earlier, in parallel with Layer 1.
5. **Layer 5's agentic evaluation** should run against the same checkpoints as Layer 1, on the same cadence, rather than being treated as a separate, later-arriving evaluation track — it's easy to under-resource this layer precisely because it's newer and less standardized than Layers 1-3, and under-resourcing it is exactly how agentic-specific failure modes end up discovered in production rather than pre-launch.

## Step 6b: A Worked Go/No-Go Meeting

To make the tiered gating structure concrete, here is what an actual launch-review meeting applying it might look like, item by item:

- **Capability (Layer 1):** new model beats the current production model on 8 of 9 pre-registered headline benchmarks; regresses 1.5 points on a coding benchmark. Tier 2 (negotiable) — the coding-benchmark owner is asked whether the regression is within noise or real; a quick re-run with a larger sample resolves it as within noise, and the item is downgraded to Tier 3 (informational) for this launch.
- **Contamination (Layer 2):** audit finds 0.4% overlap on one math benchmark, traced to a handful of documents that mirror a small subset of problems. Tier 2 — benchmark owner re-runs on a decontaminated subset; the score barely moves, so the original number is accepted with a documented footnote.
- **Dangerous capability (Layer 3):** model does not cross any pre-registered threshold. No gate triggered; result logged as informational confirmation.
- **Red-team findings (Layer 3):** two medium-severity findings, one high-severity finding (a prompt-injection susceptibility in agentic mode). Tier 1 for the high-severity finding — launch-blocking until a verified mitigation ships; Tier 2 for the two medium findings, accepted with named sign-off and a committed post-launch fix timeline.
- **Human preference (Layer 4):** new model wins 54% of blind comparisons against current production, comfortably outside the confidence interval computed the way `012_Interview_Questions_Part2.md`, Q6 argues for. No gate triggered.
- **A/B rollout (Layer 4):** 1% canary shows no statistically significant regression on any core product metric over a one-week window. No gate triggered; proceed to staged ramp-up.
- **Agentic trajectory (Layer 5):** task-completion rate flat versus production; cost-per-successful-task down 12%. No gate triggered; logged as a positive informational finding.

**Outcome:** launch is blocked pending the Tier 1 prompt-injection mitigation specifically, not the whole model — everything else has either cleared cleanly or been accepted with a named, documented exception. This is the concrete difference between a real gating framework and a single up-or-down launch vote: the decision is granular, attributable to specific findings, and only as blocking as the evidence actually requires.

## Step 6c: Common Mistakes This Framework Is Designed to Prevent

- Treating every finding as equally launch-blocking, which either produces gate paralysis or trains the org to route around the gate entirely.
- Letting a Tier 2 exception get accepted informally, in a hallway conversation, rather than documented with a named owner — this is exactly the gap that makes a launch decision indefensible after the fact if the accepted risk materializes.
- Running the safety/red-team evaluation only in the final week before launch, foreclosing any real lead time to build and verify a mitigation for whatever is found.
- Only measuring under-refusal in the safety layer and missing an over-refusal regression entirely, per the discussion in `012_Interview_Questions_Part2.md`, Q10.
- Presenting an A/B rollout's early, still-underpowered results as conclusive, rather than waiting for the pre-registered sample size/duration before treating the canary as a pass.

## Closing the Loop

## Step 7b: A Final Summary Table

| Question to answer before closing | Where it's addressed |
|---|---|
| Are the thresholds set before results are known? | Step 0 |
| Which layer catches which failure mode? | Step 0a's table |
| How does a specific finding map to a specific gate tier? | Step 6, Step 6b's worked meeting |
| Is refusal measured in both directions? | Layer 3 |
| Does the evaluation campaign have enough lead time for findings to be actionable? | Step 7 |
| Is every exception documented and owned? | Step 6, Step 6d's checklist |

The full picture a staff-level answer should land on: an evaluation framework for a launch decision is not a benchmark leaderboard — it's a **pre-registered, tiered gating structure** spanning capability, contamination, safety/dangerous-capability, human preference, and agentic-trajectory evaluation, sequenced against the launch timeline with enough lead time for findings to be actionable, with an explicit, documented exception process for negotiable gates and zero exception path for the hard, Responsible-Scaling-Policy-derived blockers — and the entire design exists specifically to make the launch decision a function of evidence gathered and thresholds set in advance, not a post-hoc narrative constructed to justify a ship date decided on other grounds.
