# Planning A Model Training Run End To End

## Step -2: How to Use This File in an Actual Interview

This file is long by design, because the real question is long by design — but a live interview response shouldn't recite every phase at the same depth. The right way to use this material live: state the phase structure at a high level first (target, architecture, compute estimate, data pipeline and infra in parallel, pretraining execution with monitoring, post-training, evaluation/gating, rollout), confirm with the interviewer which phase(s) they want to go deep on, and then bring the corresponding depth from the relevant phase below rather than delivering the entire plan at uniform detail regardless of where the interviewer's actual interest lies. An interviewer who has already asked the compute-estimation question (`001_...`) separately is signaling they want Phase 3 here treated briefly and Phase 5's monitoring plan or Phase 8's gating plan treated in depth instead.

## The Scenario

"Walk me through, end to end, how you would plan and execute a new frontier model training run from a blank slate — architecture, data, compute, infrastructure, monitoring, post-training, evaluation, and rollout. Present it as a real project plan."

This is the synthesis question the whole module builds toward, and it's usually asked near the end of a staff loop specifically because it requires pulling together every other topic into one coherent narrative with correct sequencing and realistic dependencies between workstreams. I'll present this exactly as I would in a real planning document: phases, decisions made and why, what runs in parallel versus what gates what, and where each decision connects to a concrete real-model precedent or a sibling module for the mechanics I'm not re-deriving here.

## Phase -1: Common Mistakes This Plan Is Designed to Prevent

- Sequencing the data pipeline and training infrastructure workstreams one after another instead of in parallel, needlessly adding months to the critical path.
- Locking the parameter/token split (Phase 2) by unexamined default to Chinchilla-optimal without examining whether the deployment-volume forecast argues for deliberate overtraining instead.
- Building monitoring infrastructure (Phase 5) reactively, after the first mid-run incident forces the issue, rather than as a scheduled deliverable before pretraining starts.
- Treating post-training (Phase 7) and evaluation (Phase 8) as strictly sequential after pretraining completes, rather than prototyping both against intermediate checkpoints well before the final pretrained model is ready.
- Skipping or compressing the red-team and dangerous-capability evaluation windows under launch-timeline pressure, foreclosing the lead time any triggered mitigation would need.
- Treating launch as a single cutover event rather than a staged rollout with a tested rollback procedure.

## Step -3: A Quick FAQ on Using This as a Presentation

- **Should this plan be presented phase-by-phase in a single pass, or organized differently for an executive audience?** For an executive audience, lead with Phase 0's target definition, Phase 3's cost estimate, and Phase 9b's risk register — the phase-by-phase technical detail is what a technical reviewer wants, but a budget-approving executive wants the target, the cost, and the top risks first, with technical phases available on request.
- **What's the one phase most likely to get cut under budget pressure, and why is that a mistake?** Phase 5's monitoring infrastructure, because it has no directly visible "deliverable" the way a data pipeline or a trained model does — but it's exactly the investment that determines whether Phase 6's inevitable incidents cost hours or days each, making it one of the highest-leverage, least-visible line items in the entire plan.
- **How does this plan change for a much smaller organization without frontier-scale resources?** The phase structure and sequencing logic are unchanged; what shrinks is the scale of each phase's investment (a smaller cluster, a smaller data pipeline, a lighter-weight evaluation suite) — the discipline of parallelizing Phases 4/5, building monitoring before Phase 6, and gating Phase 9 on Phase 8 applies at any scale.

## Phase 0: Define the Target Before Anything Else

Every downstream decision depends on answering three questions first, and skipping this step is the most common reason "plan a training run" answers become an unfocused tour of techniques rather than a real plan:

1. **What is this model for?** A general-purpose frontier assistant, a specialized reasoning/agentic model, a cost-optimized model meant for high-volume deployment — the answer changes almost every subsequent decision (architecture, data mixture, post-training emphasis, and critically, the inference-cost-vs-training-cost tradeoff discussed in Phase 2).
2. **What is the compute and timeline budget?** This bounds the feasible parameter/token combination before any architecture discussion, exactly via the `C≈6ND` accounting in `001_Estimating_Training_Compute_And_Cost.md`.
3. **What does "done" mean — what are the launch gates?** Defining this now, not after training completes, is what makes `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md`'s gating structure actually enforceable rather than retrofitted.

## Phase 0b: Target-Definition Questions, Tabulated

| Question | Example answer | Downstream consequence |
|---|---|---|
| What is this model for? | General-purpose assistant at high query volume | Favors overtraining a smaller model (Phase 2) |
| What is the compute/timeline budget? | $10M, 4-month training window | Bounds feasible N/D combinations (Phase 3) |
| What defines "done"? | Beats current production on headline benchmarks, clears all Tier 1 safety gates | Fixes Phase 8's gate thresholds before any result exists |
| Dense or MoE? | MoE, given expected serving volume | Determines Phase 5's parallelism strategy entirely |
| Target context length? | 128K | Determines whether context parallelism is needed (Phase 5) |

## Phase 1: Architecture Decisions

With the target defined, lock the architecture early, because changing it later invalidates data-pipeline tokenization choices, infrastructure sizing, and any completed ablations. Key decisions, each informed by real precedent:

- **Dense vs. MoE.** A dense model is simpler to train and serve (no expert-parallel all-to-all communication, no load-balancing concerns) but ties inference cost directly to total parameter count. An MoE model (DeepSeek-V3's 671B total / 37B activated split, `..\..\OpenSource\007_DeepSeek_V3.md`) decouples training-time capacity from per-token inference cost, at the cost of substantially more complex distributed-training infrastructure (expert parallelism, load balancing — DeepSeek-V3's auxiliary-loss-free bias-based balancing being the current state-of-the-art answer to the load-balance/LM-quality gradient conflict) and a MoE-specific serving architecture. The decision should be driven by the target's deployment-volume expectation from Phase 0: a model expected to serve enormous query volume benefits disproportionately from MoE's inference-cost decoupling; a model trained mainly for research or lower-volume deployment may not justify the added systems complexity.
- **Attention mechanism.** Standard MHA is simplest but most KV-cache-expensive; GQA (applied universally, including at smaller sizes, per Llama 3's design choice, `..\..\OpenSource\003_Llama3.md`, Section 2) is close to a free efficiency win once long context is a design goal, since KV-cache bandwidth cost matters at every model size once context routinely extends past a few thousand tokens; MLA (DeepSeek-V2/V3) pushes further, compressing K/V into a shared latent representation for a larger cache-efficiency win at the cost of a more intricate attention implementation. Given this model's target likely includes long-context and high-volume serving, GQA at minimum, with MLA as a stretch decision contingent on engineering bandwidth to implement and validate it correctly, is the right default.
- **Context length target and how to reach it.** Decide the eventual context length (e.g., 128K) and plan a staged approach — short-context pretraining first (cheaper per token), then a dedicated continued-pretraining phase on long-document and synthetic long-context data with RoPE base-frequency rescaling, exactly as Llama 3.1 executed (`..\..\OpenSource\003_Llama3.md`, Section 2/5) — rather than attempting to train at the full target context length from step zero, which is both more expensive per token and, at extreme context lengths, requires context parallelism (splitting a single sequence's computation across GPUs) as a fourth parallelism dimension beyond data/tensor/pipeline, a real infrastructure commitment that should be scoped in Phase 3, not discovered as a surprise mid-run.
- **Tokenizer and vocabulary size.** A larger vocabulary (Llama 3's jump to 128K tokens from Llama 2's 32K) reduces tokens-per-character, which reduces both training cost per unit of "real" content and inference decode-step count for the same reason — a comparatively cheap, high-leverage decision to get right before training starts, since changing the tokenizer after any training has begun means starting over.

## Phase 1b: Architecture Decision Summary Table

| Decision | Conservative default | Aggressive option | When to choose aggressive |
|---|---|---|---|
| Dense vs. MoE | Dense | MoE (expert-parallel) | Very high expected serving volume, team has MoE infra experience |
| Attention | GQA | MLA | Team has bandwidth to validate a more intricate cache-compression implementation |
| Context length approach | Staged continued pretraining | Training-free RoPE rescaling | Use case tolerates "doesn't break" over "genuinely uses" long context |
| Vocabulary size | Match a proven prior generation | Expand for multilingual compression | Multilinguality is a first-class target, not an afterthought |

## Phase 2: The Parameter/Token/Inference-Cost Decision

This is the decision Phase 0's "what is this model for" answer feeds directly into, and it deserves its own phase because it's frequently under-examined: given a fixed training-compute budget, Chinchilla-style compute-optimal scaling (`~20 tokens/parameter`) minimizes training loss for that budget — but training-compute-optimality and total-cost-of-ownership-optimality are different objectives. If the model will be queried at very high volume over a multi-year deployment lifetime, the Llama 3 argument applies directly: deliberately **overtrain a smaller model** well past its Chinchilla-optimal token count, because inference cost (paid on every future query, scaling with parameter count) can dwarf the one-time training-compute cost saved by stopping at the Chinchilla-optimal point, and Llama 3's 8B model continuing to improve measurably at 15T+ tokens (75-100x past its ~150-200B-token Chinchilla-optimal point) is the clearest public demonstration of this logic (`..\..\OpenSource\003_Llama3.md`, Section 5/12). This project plan should state explicitly, as a Phase 0/2 joint decision: are we optimizing training FLOPs for a target loss, or total cost of ownership including projected inference volume — and size the model and token budget accordingly, rather than defaulting to Chinchilla-optimal by unexamined habit.

## Phase 3: Compute and Cost Estimation

With N (parameters) and D (tokens) pinned down by Phase 1/2, produce the estimate exactly as worked in `001_Estimating_Training_Compute_And_Cost.md`: `C≈6ND` (using activated parameters if MoE), converted to GPU-hours via an explicit, stated MFU assumption (35-50% dense, lower for MoE pending DualPipe-style communication-overlap engineering investment), converted to dollar cost via an explicit $/GPU-hour assumption. This produces the budget line the rest of the project plan is measured against, and it should be revisited (not treated as fixed) once Phase 4's infrastructure plan is concrete enough to refine the MFU assumption with real numbers rather than a planning-stage guess.

## Phase 3b: A Quick FAQ on the Compute Estimate's Role in This Plan

- **Why does the compute estimate come before the data pipeline and infra phases, rather than after?** Because it's what determines whether the org can afford the plan at all before committing engineering months to building out Phases 4 and 5 — running the estimate late would mean discovering an unaffordable plan only after substantial sunk engineering cost.
- **Should the estimate be revisited later in the project?** Yes — explicitly, once Phase 5's infrastructure is concrete enough to replace the planning-stage MFU assumption with a measured one, per `001_Estimating_Training_Compute_And_Cost.md`'s sensitivity-analysis discussion.
- **What if the estimate comes back far higher than the available budget?** This is exactly the point of running it early — it forces a Phase 0/2 reconsideration (a smaller model, a different N/D split, a longer timeline) before any downstream phase has consumed real engineering time.

## Phase 4: Data Pipeline

Run this phase in parallel with Phase 5 (infrastructure), not sequentially after it — data pipeline engineering has its own long lead time and should not wait on the training cluster being ready, since a training-ready corpus needs to exist before the cluster's expensive GPU-hours start being consumed. The full sequencing (acquisition, cleaning/filtering, deduplication, quality classification, mixture weighting, contamination screening, and the distributed-processing/storage/versioning infrastructure underneath all of it) is worked in detail in `002_Designing_A_Pretraining_Data_Pipeline_From_Scratch.md` — the project-plan-level point is: **this is the single workstream most likely to be underestimated in a naive timeline**, because it looks like "just gather some data" from the outside and is actually a multi-stage distributed-systems engineering effort with its own multi-month lead time, and contamination screening against the eval suite defined in Phase 0/8 has to complete *before* the final training corpus snapshot is declared ready, not as a post-hoc audit after training has already consumed the data.

## Phase 4b: A Quick FAQ on Parallelizing Data and Infra Workstreams

- **What's the actual risk of running Phases 4 and 5 in parallel rather than sequentially?** Very low, and it's the single highest-leverage scheduling call in the whole plan — the two workstreams have almost no hard dependency on each other's outputs (data pipeline needs the tokenizer decision from Phase 1; infra needs the parallelism-strategy decision from Phase 1; neither needs the other's output to begin).
- **What's the one place these two workstreams do need to sync?** The data-loading interface — the exact sharded, pre-tokenized format the data pipeline produces (Phase 4) needs to match what the training infrastructure's data-loading layer (Phase 5) expects to consume, and this interface contract should be settled early and explicitly, not discovered as a mismatch when the two workstreams first try to integrate.
- **Who owns resolving a Phase 4/5 integration mismatch if one appears?** This should be named explicitly in the project plan before it's needed — an unowned integration boundary between two independently-progressing workstreams is a classic place for a real-world project to lose weeks to finger-pointing.

## Phase 5: Distributed Training Infrastructure

Also run substantially in parallel with Phase 4, gated by Phase 1's architecture decisions (parallelism strategy depends directly on whether the model is dense or MoE, and on the target context length per Phase 1's context-parallelism note):

- **Parallelism grid.** For a dense model at this scale, a standard 3D (data/tensor/pipeline) parallelism grid with ZeRO/FSDP-style optimizer-state sharding, exactly the systems reasoning worked through for GPT-3-scale dense models in `..\..\GPT\003_GPT3.md`, Section 4 — plus context parallelism as a fourth dimension if the long-context target from Phase 1 requires it (Llama 3.1 405B's disclosed 4D parallelism, `..\..\OpenSource\003_Llama3.md`, Section 4). For an MoE model, expert parallelism is added on top, with a pipeline schedule (DualPipe-style) specifically engineered to overlap the MoE all-to-all communication pattern with compute, since this communication pattern is the primary MFU tax an MoE architecture pays relative to dense.
- **Precision strategy.** BF16 mixed precision is the safe, well-understood default; FP8 (DeepSeek-V3's fine-grained tile/block-quantization recipe, `..\..\OpenSource\007_DeepSeek_V3.md`, Section 3) offers a substantial MFU/memory win but requires real engineering investment in scaling-factor infrastructure and careful placement of which operations must stay in higher precision — the mechanics of this tradeoff, and the numerical-stability failure modes it introduces, are covered in `..\03_Distributed_Training_And_Infrastructure\004_Mixed_Precision_Training_And_Numerical_Stability.md`. The project plan should treat FP8 as a deliberate, resourced engineering bet with its own validation timeline, not a checkbox flipped at the last minute.
- **Fault tolerance and checkpointing.** At the GPU counts implied by Phase 3's budget, non-trivial hardware failure/interruption rates over a multi-week-to-multi-month run are a certainty, not a risk to be hand-waved — Llama 3.1 405B's paper treats automated failure-detection and fast-restart tooling as a first-class engineering deliverable, not an afterthought (`..\..\OpenSource\003_Llama3.md`, Section 4/8), and this project plan should budget explicit engineering time for this before the run starts, alongside a checkpointing-cadence decision that balances failure-recovery protection against checkpointing overhead.
- **Monitoring infrastructure, built before the run starts, not reactively during an incident.** Per-rank step-time breakdown (compute/communication/data-wait), gradient-norm and loss-scale telemetry, per-layer gradient-norm logging, GPU health telemetry (temperature, ECC error counters, NVLink/link status), and data-shard-level batch reconstruction capability — this is the exact instrumentation `003_Debugging_A_Loss_Spike_Mid_Training.md` and `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md` both depend on to diagnose mid-run incidents efficiently, and the mechanics of building it are covered in `..\03_Distributed_Training_And_Infrastructure\008_Debugging_Distributed_Training_Failures.md`. Treat this monitoring buildout as a Phase 5 deliverable with its own timeline and owner, not something assembled ad hoc after the first incident occurs — every hour spent building this before the run starts saves many more hours of blind diagnosis during an actual incident weeks into training.

## Phase 5b: A Quick FAQ on the Infrastructure Phase

- **Which parallelism decision is hardest to change after the fact?** The MoE-vs-dense and context-length-target decisions from Phase 1, since they determine the whole parallelism grid (adding expert or context parallelism mid-run is far more disruptive than adjusting a hyperparameter) — this is exactly why Phase 1 has to lock before Phase 5 can be designed in earnest.
- **Is FP8 worth adopting for a first-ever frontier training run at a given organization?** Only with real caution — per the risk-sequencing critique in `010_Critiquing_Real_Published_Training_Recipes.md`'s Critique 2, stacking an unfamiliar precision format with other novel bets in the same run compounds diagnostic difficulty if something goes wrong; a first run might reasonably defer FP8 to a subsequent generation once BF16 training at this scale is well-understood internally.
- **How early should fault-tolerance tooling be validated?** Before the full-scale run starts, via a deliberate fault-injection test (killing a node intentionally) on a smaller proxy run — discovering that automated failure-recovery doesn't actually work correctly should never happen for the first time during the real run.

## Phase 6: Execute Pretraining, With a Standing Incident-Response Plan

Once Phases 4 and 5 converge (training-ready data, validated infrastructure), pretraining begins. The project plan for this phase is less about new engineering and more about **operational discipline**: a defined on-call rotation for the run's duration, a pre-agreed decision framework for exactly the situations covered in `003_...` (loss spikes) and `004_...` (throughput regressions) so that when — not if — one of these occurs, the response is "follow the established diagnostic tree" rather than an ad hoc scramble, and a checkpoint-evaluation cadence (running the capability-benchmark subset of Phase 8's eval suite against periodic checkpoints throughout the run, not only at the very end) that gives early warning if the run is tracking below the expected scaling-law-predicted loss/capability trajectory, cheaply enough to react while there's still budget and timeline left to do so.

## Phase 6b: A Quick FAQ on Execution-Phase Discipline

- **What's the biggest difference between planning this phase and actually running it?** Planning is about having the right infrastructure and diagnostic trees ready; execution is almost entirely about *following* them under real time pressure rather than improvising a fresh response to each incident as if it were the first of its kind.
- **How much of the on-call rotation's time should be budgeted for incident response versus proactive monitoring review?** Both matter, but proactive review (checking the checkpoint-evaluation cadence's trajectory against the scaling-law prediction, per this phase) is what catches problems before they become incidents — an on-call rotation that's purely reactive is missing the cheaper, earlier intervention point.
- **What triggers escalating a mid-run finding to the Phase 2 "reconsider the plan" gate rather than just a tactical fix?** A checkpoint-eval trajectory meaningfully below the scaling-law-predicted curve (not just a transient blip) is exactly this trigger — it's a strategic signal, not an incident, and deserves the reconsideration process from `011_Interview_Questions_Part1.md`, Q8, not a diagnostic-tree response.

## Phase 7: Post-Training / Alignment

Sequenced immediately after pretraining (and, for capability-critical skills, prototyped on intermediate pretraining checkpoints well before the final pretrained model is ready, so the post-training pipeline isn't starting from zero the day pretraining finishes):

- **SFT.** Establish an instruction-following baseline via supervised fine-tuning on curated demonstrations, exactly the InstructGPT-lineage first stage (`..\..\GPT\004_InstructGPT_And_RLHF.md`, Section 6) — this is also the stage that produces the reference/anchor policy every subsequent RL stage's KL penalty will be measured against, so its quality directly bounds how well later stages can be constrained.
- **Preference-based alignment.** Whether via classic RLHF (reward model + PPO), a DPO-style direct optimization (Llama 3's shift away from PPO-centric RLHF toward SFT + rejection sampling + DPO, `..\..\OpenSource\003_Llama3.md`, Section 6, citing simpler, more reliably scalable tuning as the reason), or an AI-feedback-augmented pipeline in the Constitutional AI lineage (`..\..\Claude\008_Constitutional_AI_And_RLAIF.md`) for harmlessness specifically — the choice depends on available human-labeling budget, the org's tolerance for RL infrastructure complexity, and how much of the labeling burden needs to be relocated from human preference-comparison labor to a written-constitution-plus-AI-feedback approach.
- **Verifiable-reward RL, if the target includes strong reasoning/agentic capability.** A DeepSeek-R1-style RLVR pipeline (`..\..\OpenSource\008_DeepSeek_R1.md`) — cold-start SFT for output-style anchoring, then GRPO-driven RL against rule-based accuracy/format rewards, then rejection-sampling SFT to broaden coverage back out from the RL stage's narrow verifiable-task distribution, then a secondary RL stage blending verifiable and preference-style rewards — is the current state-of-the-art template for this, and it should be planned as a genuinely separate workstream from the preference-alignment stage above, since it has its own distinct infrastructure needs (rollout-generation worker pools, sandboxed code-execution verifiers) covered in that file's Section 4.
- **Standing reward-hacking monitoring**, built into this phase from the start rather than added reactively: periodic blinded true-preference audits against the training-time reward signal, exactly the monitoring discipline argued for in `006_Responding_To_A_Reward_Hacking_Incident.md`, so that a Goodhart's-Law divergence is caught within the same training run rather than discovered only after the model has already shipped.

## Phase 7b: A Quick FAQ on Sequencing Post-Training

- **Why prototype SFT/alignment against late-stage pretraining checkpoints rather than waiting for the final model?** Because post-training pipeline development (data collection, RM/verifier infrastructure, RL infrastructure validation) has its own long lead time, and waiting for a fully-finished pretrained model before starting any of it needlessly extends the critical path — exactly the same "start the long-lead-time workstream as early as its inputs allow" logic as Phase 4/5's parallelization.
- **Does the choice between DPO-style and full RLHF/RLVR pipelines need to be locked as early as the architecture decisions in Phase 1?** No — this choice can be made later, informed by the org's labeling-budget and infrastructure-complexity tolerance, and doesn't gate any earlier phase the way Phase 1's architecture decisions do.
- **Should reward-hacking monitoring (per `006_Responding_To_A_Reward_Hacking_Incident.md`) be built during this phase or treated as a later add-on?** Built in from the start of this phase — retrofitting it after a suspected incident has already occurred is strictly worse than having the standing audit running from RL's first step.

## Phase 8: Evaluation and Launch Gating

Run the full layered framework from `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md` — capability benchmarks (with contamination screening as a prerequisite, not an afterthought), safety/red-team evaluation (including dangerous-capability thresholds tied to the org's Responsible Scaling Policy, `..\07_Safety_Alignment_And_Responsible_Scaling\001_Responsible_Scaling_Policies_And_Preparedness_Frameworks.md`), human preference testing, and agentic/trajectory evaluation if the target model is expected to be used in multi-step tool-use settings. Sequence this against the timeline exactly as that file's Step 7 lays out: capability and contamination checks starting against intermediate checkpoints well before the final model is ready, red-teaming scoped with a real multi-week window rather than compressed into the final days, and dangerous-capability evaluation completed early enough that any required mitigation has time to be built and re-verified before the planned launch date.

## Phase 8b: A Quick FAQ on the Evaluation Phase's Placement in the Plan

- **Why does evaluation start against intermediate Phase 6 checkpoints rather than waiting for Phase 7 to finish?** Because capability and contamination checks (Layers 1-2 of `005_...`) don't require post-training to be complete, and running them early surfaces problems with enough lead time to act on — exactly the same principle driving Phase 4/5's parallelization.
- **What's the single most timeline-risky sub-component of this phase?** Red-teaming, because it's human-labor-intensive with its own multi-week minimum engagement window that can't be compressed without directly reducing finding quality — this is the sub-component most likely to be squeezed by a slipping schedule, and squeezing it is the highest-regret shortcut available at this phase.
- **Should Phase 8 and Phase 9 (rollout) ever overlap?** Only in the narrow sense that the staged A/B rollout is itself the tail end of Layer 4's evaluation — the two phases share that specific mechanism, but full production rollout should not begin before Phase 8's Tier 1 gates have cleared.

## Phase 9: Rollout

Launch is not a single cutover event for a model at this scale — plan a staged rollout: an initial limited-traffic A/B deployment (feeding Phase 8's human-preference and product-metric evaluation with real usage-distribution data, not just offline eval-set data), a defined rollback trigger and procedure if the staged rollout surfaces a regression the offline evaluation missed, and a monitoring plan for the post-launch period that watches for exactly the failure modes covered in `009_Post_Launch_Model_Degradation_And_Incident_Response.md` — serving-infrastructure regressions, quantization/config drift, real-world query-distribution shift, and A/B mis-routing — since a launch decision made well does not guarantee a deployment that stays healthy without continued monitoring.

## Phase 9b: A Risk Register — What Could Derail This Plan

A real project plan names its top risks explicitly rather than presenting an idealized happy path; a plausible risk register for this project:

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data pipeline (Phase 4) takes longer than planned, delaying pretraining start | Medium-high | High (delays entire critical path) | Start Phase 4 immediately after Phase 1 locks; treat dedup specifically as the highest-risk sub-stage per `002_...`'s own flag |
| Mid-run loss spike or throughput regression consumes significant unplanned time | High (near-certain at this scale, per `003_...`/`004_...`) | Medium (bounded if monitoring is in place; unbounded if not) | Build monitoring infra (Phase 5) before the run starts, not reactively |
| Post-training reward hacking discovered late | Medium | Medium-high | Standing true-preference audits from the start of Phase 7, not just at the end |
| Dangerous-capability threshold crossed unexpectedly | Low-medium | High (can block launch entirely if discovered late) | Run Layer 3 dangerous-capability evals against intermediate checkpoints, early, per Phase 8 |
| Scaling-law extrapolation from Phase 2 proves wrong at full scale | Low-medium | High (compute already spent) | Periodic checkpoint evals against the predicted trajectory (Phase 6), with a defined reconsideration gate |
| Contamination discovered in the final corpus snapshot | Low (if Phase 4's screening is done well) | High if discovered post-training | Contamination screening is a hard gate before training starts, not an audit after |

## Phase 9c: Team Ownership Across Phases

| Phase | Primary owner | Key dependency |
|---|---|---|
| 0-2 (target, architecture, N/D decision) | Research leadership + pretraining research team | None — this is the starting point |
| 3 (compute/cost estimate) | Infrastructure/finance-adjacent planning function | Phase 1-2 outputs |
| 4 (data pipeline) | Data infrastructure + data-quality teams | Phase 1 architecture lock (tokenizer) |
| 5 (training infra) | Distributed-training infrastructure team | Phase 1 architecture lock (parallelism strategy) |
| 6 (pretraining execution) | Pretraining research team + infra on-call | Phases 4 and 5 both complete |
| 7 (post-training) | Alignment/post-training research team | Late-stage Phase 6 checkpoints |
| 8 (evaluation/gating) | Evaluation and safety teams jointly | Intermediate Phase 6 checkpoints onward |
| 9 (rollout) | Product/serving-infrastructure team | Phase 8 sign-off |

## Phase 9d: A Quick FAQ on the Overall Plan

- **What's the single change to this plan that would most reduce total project risk?** Building Phase 5's monitoring infrastructure before Phase 6 starts — of everything in the risk register (Phase 9b), the mid-run incident risk is both the most likely and the one whose impact is most directly bounded by how much monitoring was already in place before it occurred.
- **How should this plan change if the organization has already trained one frontier model and is planning its second?** Phases 4 and 5 shrink dramatically in lead time (the pipeline and infrastructure already exist and need incremental extension, not a from-scratch build), which shifts the critical path toward Phases 1-2's architecture/scale decisions and Phase 7-8's post-training and evaluation depth as the areas most worth fresh investment.
- **What's the most common way this exact plan fails in practice, even when every phase is individually well-executed?** Underinvesting in the handoffs between phases — the Phase 4/5 data-loading interface (Phase 4b), the Phase 6-to-7 checkpoint handoff, and the Phase 8-to-9 gating decision — each of which needs an explicit owner and contract, not just well-run phases in isolation.

## Phase 10: What Runs in Parallel vs. What Gates What — The Actual Project-Management View

Pulling the above into a dependency-ordered view, because a strong answer should be able to draw this as a real project timeline, not just a list of phases in reading order:

```
Phase 0 (target definition) ─┬─> Phase 1 (architecture) ─┬─> Phase 4 (data pipeline)      ──┐
                              │                            │                                  │
                              └─> Phase 2 (N/D/inference   └─> Phase 5 (training infra)    ──┼─> Phase 6 (pretraining)
                                  cost decision)                                              │        │
                                            │                                                 │        v
                                            └─> Phase 3 (compute/cost estimate) ──────────────┘   Phase 7 (post-training)
                                                                                                         │
                                                                                                         v
                                                                                              Phase 8 (evaluation/gating)
                                                                                                         │
                                                                                                         v
                                                                                                   Phase 9 (rollout)
```

Phases 4 and 5 are the two long-lead-time workstreams that should start immediately after Phase 1's architecture decisions lock, running fully in parallel with each other — this is the single highest-leverage scheduling decision in the whole plan, since sequencing them one-after-the-other (data pipeline first, then infrastructure, or vice versa) needlessly adds months to the critical path for no technical reason. Phase 8's capability/contamination checks should start against intermediate Phase 6 checkpoints, not wait for Phase 7 to complete, and Phase 7's SFT/alignment pipeline prototyping should similarly start against late-stage Phase 6 checkpoints rather than waiting for pretraining to be fully declared "done." The discipline of identifying which workstreams can and must overlap, versus which genuinely gate each other, is exactly the project-planning judgment a staff engineer is expected to bring to this conversation — and it's the detail that separates "I know what all the pieces are" from "I could actually run this."
