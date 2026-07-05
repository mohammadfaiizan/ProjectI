# Critiquing Real Published Training Recipes

## The Scenario

"Pick real published model training recipes from what you know and critique them — not summarize them, actually critique them. What would you have done differently, and why?"

This is the question most likely to expose a candidate who has memorized model facts without actually forming an independent technical opinion about them. Summarizing DeepSeek-V3's architecture accurately is table stakes; a staff-level answer has to take a genuine position — including a position that might be uncomfortable to state about a well-regarded, widely-praised recipe — and defend it with specific technical reasoning rather than vague praise or vague criticism.

I'll do this for five real, well-documented cases, in decreasing order of how firmly the critique ultimately lands after hearing the strongest counter-argument:

- Llama 3's deliberate overtraining of the 8B and 70B models.
- DeepSeek-V3's aux-loss-free MoE balancing, stacked with its FP8 recipe in the same run.
- GPT-3's complete absence of any alignment stage, contrasted directly with InstructGPT two years later.
- DeepSeek-R1's negative result on process reward models.
- Constitutional AI's constitution-authorship legitimacy question.

For each, I state the recipe, state the critique plainly, then argue the strongest counter-case — because a critique that doesn't survive contact with the best counterargument isn't actually a critique, it's just a complaint.

## Step 0b: A Quick FAQ on Critique Construction

- **What if you genuinely can't think of a counter-case for a critique that seems obviously right?** Keep looking before presenting it as settled — a critique of well-resourced, heavily-scrutinized frontier work that has no serious counter-case at all is rare enough that its absence should itself raise suspicion that something about the critique's framing is off, not confidence that the critique is airtight.
- **Is it acceptable to critique a decision you'd have made the same way given the same constraints?** Yes, and doing so explicitly (stating "I likely would have made the same call, and here's why the call is still worth scrutinizing") is often a stronger answer than pretending you'd have obviously done better.
- **How many critiques should a candidate have ready before an interview?** Two or three, worked all the way through to a stated verdict, is more valuable than five stated only at the "here's the critique" stage with no counter-case — depth on fewer examples beats breadth without follow-through.

## Critique 1: Llama 3's Deliberate Overtraining of the 8B and 70B Models

**The recipe.**

- Llama 3's 8B and 70B models are trained on 15T+ tokens.
- That is 75-100x past the token count Chinchilla-style compute-optimal scaling would prescribe for their parameter counts (`..\..\OpenSource\003_Llama3.md`, Section 5).
- The paper's stated rationale: training-compute-optimality and inference-cost-minimization are different objectives.
- For a model served at enormous volume over a multi-year lifetime, minimizing parameters at a fixed capability bar — even at the cost of a much larger one-time training-token bill — minimizes *total* cost of ownership.

**The critique.**

- The entire cost-benefit case depends on a specific, unstated forecast of future query volume, and that forecast is never made explicit or quantitatively defended in the public paper.
- The overtraining decision is a bet: "we predict enough future inference volume at these specific sizes that the extra training-token cost pays for itself."
- The actual break-even math — at what query volume does the extra training compute for the 8B model amortize against inference savings, versus spending that same budget on a somewhat larger, compute-optimally-trained model — is never shown.
- For an organization at Meta's scale and query volume, the bet is probably comfortably justified.
- As a piece of generalizable methodology, though, the argument risks being over-applied by smaller organizations citing "Llama 3 proved overtraining is right" without running their own, much smaller-volume, break-even calculation.
- The correct generalized lesson is "work out which objective binds for *your* deployment volume," not "always overtrain small models" — the latter is a category error dressed up as a takeaway.
- A second, more technical critique: the paper reports the 8B model "had not saturated" at 15T tokens, loss still improving log-linearly when training stopped.
- This is framed as validation of the strategy, but it equally signals that the stopping point was set by budget and timeline, not by a principled quality/cost optimum.
- A more rigorous version of this research program would have published the actual empirical loss/eval-vs-tokens curve well past the point of diminishing returns, rather than a qualitative "we observed continued improvement."

**The strongest counter-case.**

- Meta's internal telemetry on Llama-integrated product query volume is almost certainly far more detailed than any outside observer's ability to second-guess it.
- Demanding a fully quantified break-even analysis in a public paper may be asking for competitively sensitive information the org has no obligation to disclose.
- The qualitative direction of the argument — inference cost dominates for widely-deployed models, and Chinchilla-optimality says nothing about it — is unambiguously correct and a genuinely important corrective to the field.
- **Verdict:** the paper's argument is directionally right and important; its generalizability beyond Meta's specific deployment scale is asserted rather than shown. The decision was almost certainly right for Meta; the *argument*, taken as general methodology, is under-specified.

## Critique 2: DeepSeek-V3's Aux-Loss-Free MoE Balancing, Stacked With Its FP8 Recipe

**The recipe.**

- DeepSeek-V3 replaces the standard MoE auxiliary-balancing-loss with a non-differentiable, bias-based feedback-control mechanism (`b_i += / -= γ` based on observed expert load).
- This removes the gradient-level conflict between load-balancing and language-modeling objectives.
- Separately, the entire 671B-parameter run trains primarily in FP8 with fine-grained tile/block quantization (`..\..\OpenSource\007_DeepSeek_V3.md`, Sections 2-3).

**The critique — the bias mechanism specifically.**

- The aux-loss-free approach is a genuinely elegant fix: it removes a structural gradient conflict rather than just reweighting it.
- But it introduces its own, less-examined tuning surface: the step size γ and the load-observation window are hyperparameters with no principled selection method disclosed.
- The mechanism's stability properties as a feedback control loop — does it oscillate under certain load patterns, how fast does it respond to a shift in token-to-expert affinity mid-training, how does it interact with the retained sequence-wise auxiliary loss — are asserted to work empirically, not analyzed with control-systems rigor.
- "We tried a specific γ and observed good balancing" is a considerably weaker claim than "this feedback loop is provably stable across the range of load patterns an MoE router can produce."
- This is a fair critique even while fully crediting the technique's real conceptual contribution: eliminating one problem via a mechanism whose own failure modes aren't yet well-characterized is progress with a new, smaller, differently-shaped risk surface — not risk elimination.

**The critique — pairing aggressive FP8 with an already load-balance-sensitive architecture.**

- Stacking two independently ambitious systems bets in the same run — a novel, not-fully-characterized balancing mechanism, *and* training the whole model primarily in a narrow-dynamic-range format for the first time at this scale — compounds risk.
- If a training anomaly had appeared, disentangling "FP8 quantization-error accumulation" from "the bias-balancing loop responding badly to a load shift" from "genuine MoE routing collapse" (per `003_Debugging_A_Loss_Spike_Mid_Training.md`) would have been materially harder than with either bet made in isolation.
- This is a critique about **risk sequencing in a research program**, not about either technique individually.
- A more conservative execution path would have staged these two bets — validate balancing at BF16 first, then separately validate FP8, then combine — rather than committing both to the same production run at once.

**The strongest counter-case.**

- DeepSeek explicitly reports the sequence-wise auxiliary loss as a low-weight safety net specifically because they anticipated the bias mechanism alone might not suffice — evidence of appropriate caution, not recklessness.
- Staging the two bets sequentially at smaller scale first is exactly what ablations are for, and there's no strong reason to assume DeepSeek didn't do exactly that before committing the full 671B run.
- The report simply doesn't disclose the ablation history in enough detail for an outside critic to know whether the staging happened.
- **Verdict:** the critique is more precisely "the public report doesn't demonstrate this risk was appropriately staged," not "it wasn't." The report's transparency is unusually high for this class of disclosure, but even an unusually transparent report leaves real gaps in exactly the places (ablation history, hyperparameter sensitivity) needed to fully evaluate the risk-sequencing concern.

## Critique 3: GPT-3's Complete Absence of Any Alignment Stage

**The recipe.**

- GPT-3 ships as a raw pretrained language model: zero instruction-tuning, zero RLHF, zero preference modeling of any kind (`..\..\GPT\003_GPT3.md`, Section 6).
- It is evaluated and shipped purely via prompting, with no training-time mechanism distinguishing "a helpful, direct answer" from "a plausible continuation of the prompt in the statistical style of internet text."

**The critique.**

- InstructGPT's own headline result two years later: labelers preferred outputs from a **1.3B-parameter aligned model over the 175B-parameter unaligned GPT-3** (`..\..\GPT\004_InstructGPT_And_RLHF.md`, Section 1).
- Shipping a 175B-parameter model, at that era's considerable training cost, with zero investment in the training stage that would turn out to matter more than two further orders of magnitude of scale, looks like a significant strategic miscalibration in hindsight.
- GPT-3's own limitations section is candid about the consequences: no mechanism for calibrated honesty, no correction mechanism, inherited web-text bias/toxicity with no post-hoc mitigation.
- The organizational response to identifying this gap was a full separate research program (InstructGPT), not a training-time fix bundled into the same release.
- A more foresighted program, informed by pre-2020 RLHF-for-LMs literature that already existed (Ziegler et al. 2019, and earlier preference-learning work outside LMs), could plausibly have shipped *some* lightweight alignment stage — even just SFT on a modest demonstration set — alongside the original release.

**The strongest counter-case — the one that should genuinely change the critique's framing.**

- This critique is much easier to make in 2026 than it would have been to act on in 2020; a staff-level answer must be honest about that asymmetry.
- In 2020, RLHF-for-general-instruction-following at GPT-3 scale was **not yet a proven recipe** — Stiennon et al.'s directly relevant precedent (RLHF for summarization specifically) was itself barely published.
- Generalizing that recipe from one narrow task to the full open-ended distribution of "whatever a user might ask" was exactly the open research question InstructGPT's paper set out to answer, not a known-safe extension anyone could have simply bundled in with confidence.
- GPT-3's actual contribution — scale alone produces qualitatively new capability (in-context learning) — is a different, and at the time genuinely uncertain, research bet than "does RLHF generalize to instruction-following," and conflating the two understates the real research uncertainty between 2020 and 2022.
- **Verdict:** GPT-3's alignment gap was a reasonable, defensible scope decision for a paper whose actual research question was about scaling, not alignment. The sharper critique reframes as: the field, including OpenAI, underinvested in exploring whether the summarization-RLHF result would generalize for longer than the eventual result suggests it should have — a critique of the field's 2020-2022 research-portfolio allocation, not of one paper's scope decision.

## Critique 4: DeepSeek-R1's Negative Result on Process Reward Models

**The recipe.**

- DeepSeek-R1's report states that naive process reward models (PRMs — scoring individual intermediate reasoning steps, not just the final answer) were prone to reward hacking and underperformed simple outcome-only verifiable rewards at scale (`..\..\OpenSource\008_DeepSeek_R1.md`, Sections 2 and 8).
- The released pipeline relies entirely on outcome-level accuracy plus a format reward.

**The critique.**

- This is reported as a clean negative result, but at summary-level detail relative to how load-bearing the conclusion is for the field's subsequent research direction.
- Many other groups have since cited it as evidence that process supervision doesn't scale, without the underlying ablation (which PRM architecture, what specific gaming behavior, at what scale, what mitigations were tried first) being disclosed.
- "Process supervision is fundamentally hard to get right at this scale" and "DeepSeek's specific PRM implementation had a fixable gap they didn't have time to fix" are very different claims with very different implications for whether other groups should keep investing in process supervision.
- The report's brevity means the field is currently generalizing from a result whose robustness is simply not verifiable from the public disclosure.

**The strongest counter-case.**

- DeepSeek explicitly frames this as a practical engineering finding under their specific scale and time budget, not a claim about the theoretical limits of process supervision as a research direction.
- Reporting a clean negative result at all, even without exhaustive ablation detail, is more useful to the field than omitting the finding or overclaiming a general impossibility result the report doesn't actually assert.
- **Verdict:** the disclosure itself is a positive contribution; the field's subsequent citation of it as broadly conclusive outruns what the specific disclosure actually supports.

## Critique 5: Constitutional AI's Legitimacy Gap — Who Actually Writes the Constitution

**The recipe.**

- Constitutional AI relocates the alignment target from an implicit, statistically-encoded human-preference dataset to an explicit, written constitution.
- It relocates human labeling effort from thousands of individual harmful-prompt comparisons to drafting and curating a comparatively small set of principles (`..\..\Claude\008_Constitutional_AI_And_RLAIF.md`, Sections 1-2).

**The critique.**

- The original paper's own Section 8 acknowledges this gap directly, which is exactly where the critique should press.
- The constitution is authored by a small group of researchers at one organization, drawing on external reference material (e.g., the UN Declaration of Human Rights) alongside internal judgment.
- This makes the alignment *target* auditable as text, but does nothing to establish that the *content* represents the range of stakeholders a deployed-at-scale model's behavior actually affects.
- A large, demographically diverse labeling workforce — for all its own well-documented problems (labeler-pool bias, per `..\..\GPT\004_InstructGPT_And_RLHF.md`, Section 8) — at least aggregates judgment across many individuals.
- A constitution drafted by a small internal team concentrates that judgment in far fewer hands.
- "The target is now auditable" elides that auditability of a document and legitimacy of its content are different properties: you can read and critique the text without any mechanism existing for affected external stakeholders to actually change it.

**The strongest counter-case.**

- RLHF's implicit target has exactly the same legitimacy gap, just in a harder-to-see form — a labeling workforce's guidelines are themselves authored by a small internal team, rarely published in full.
- CAI's transparency about *where* the values come from is arguably a strict improvement in legitimacy-*visibility*, even where it doesn't resolve legitimacy-*substance*.
- Anthropic has since discussed, at a summary level, soliciting broader public/expert input into constitution drafting for at least some experimental efforts — a real, if partial and unverifiable-from-outside, response to this exact critique.
- **Verdict:** real, explicitly acknowledged by the original authors, and symmetric across both the RLHF and CAI paradigms rather than a unique flaw of CAI specifically.

## A Rapid-Fire List of Other Candidate Critiques Worth Having Ready

An interview may ask for a critique of a model not covered in depth above; the same discipline applies directly:

- **Mixtral 8x7B's comparatively coarse-grained MoE** (8 experts, top-2) relative to DeepSeek-V2/V3's fine-grained approach (256 experts, top-8) — fair critique: routing granularity leaves specialization gains on the table; fair counter-case: Mixtral shipped before the fine-grained-MoE literature matured, and coarser routing is simpler to serve efficiently.
- **GPT-4's near-total lack of architectural disclosure** relative to the open-weight lineage — fair critique: opacity slows the field's collective ability to learn from a frontier lab's design choices; fair counter-case: competitive and safety considerations are a real, not merely convenient, reason to withhold detail.
- **Qwen2.5's broad multi-size simultaneous release** — fair critique: spreading validation effort across many sizes can dilute depth of validation per size; fair counter-case: a broad size ladder serves far more deployment constraints, and marginal cost per additional size is low once the recipe is validated once.

- **Gemini 1.5's extreme long-context claims (multi-million-token windows) without equally extensive published long-range-utilization evaluation** — fair critique: architecturally supporting a huge context window is not the same claim as genuinely using information anywhere within it (the same distinction raised in `..\..\OpenSource\003_Llama3.md`, Section 10, and in `012_Interview_Questions_Part2.md`, Q12), and headline context-length numbers can outrun the depth of published needle-in-a-haystack-style validation; fair counter-case: Google has published at least some long-context retrieval evaluation, and extreme context length is itself a genuine, hard-won engineering achievement worth crediting even if the utilization evaluation could be deeper.
- **GLM-4.5's rapid iteration cadence across closely-spaced releases** — fair critique: a very fast release cadence risks each individual release receiving a shallower evaluation and safety-review pass than a slower cadence would allow, purely as a function of calendar time available per release; fair counter-case: rapid iteration lets a lab incorporate lessons from each release into the next far faster, and a shallow-eval concern is only a real critique if it can be shown a specific release actually skipped or compressed a review step that mattered, not merely that the cadence was fast.

## A Scorecard Across All Five Critiques

Laying the five critiques side by side makes the calibration point below concrete rather than asserted:

| # | Decision critiqued | Strength of critique | Strength of counter-case | Where it lands |
|---|---|---|---|---|
| 1 | Llama 3's 8B/70B overtraining | Real gap in the *argument's* generalizability | Very strong for Meta specifically | Decision likely right; argument under-specified |
| 2 | DeepSeek-V3's balancing + FP8 stacked together | Real risk-sequencing concern | Strong (report doesn't refute or confirm it) | Open question, not a confirmed error |
| 3 | GPT-3's zero alignment stage | Weak against the 2020 paper's actual scope | Very strong (RLHF-for-instruction-following was unproven in 2020) | Reframe as field-level, not paper-level |
| 4 | DeepSeek-R1's PRM negative result | Real gap in disclosed ablation depth | Moderate (disclosure itself is valuable) | Field over-generalized an under-specified result |
| 5 | Constitutional AI's constitution authorship | Real, explicitly acknowledged by the authors | Moderate (RLHF has the same gap, less visibly) | Legitimate, unresolved, symmetric across paradigms |

The pattern across all five rows: not one lands as an uncomplicated "they were simply wrong." That's not a failure of nerve — it's what a genuinely defensible critique of well-executed, heavily-scrutinized frontier work actually looks like. A critique portfolio where every row reads "obviously wrong, should have done X" is much more likely evidence of a shallow, uncharitable reading than evidence of five independent real mistakes by five different well-resourced research teams.

## A Reusable Checklist for Constructing a Critique Under Interview Pressure

When asked this question live, with no chance to prepare the specific model in advance, the same structure generalizes into five steps:

1. **State the specific decision, not the general topic.** "DeepSeek-V3's MoE design" is a topic; "DeepSeek-V3 replaced the auxiliary balancing loss with a non-differentiable bias-update rule, with step size γ left unprincipled" is a decision.
2. **State what alternative was foreclosed**, concretely — what would the team have had to do differently, and what would that alternative have cost or risked in return.
3. **State the critique as a specific, checkable claim.** "The disclosed report doesn't demonstrate X" is checkable; "this was a bad idea" is not.
4. **Construct the strongest counter-case a genuinely well-informed defender would make** — not a strawman, and not skipped entirely.
5. **State where the critique lands after hearing the counter-case** — fully sustained, partially sustained and reframed, or substantially withdrawn. Being willing to land in any of these three, including the third, is what separates genuine critique from contrarianism-as-performance.

## Why This Question Recurs So Often at the Staff Level

A few reasons this specific format of question is disproportionately common in staff-and-above loops, worth having ready as a closing thought:

- It's very hard to fake. A candidate who has only skimmed model announcements can usually summarize architecture facts convincingly, but constructing a specific, checkable critique and then a genuine (not strawmanned) counter-case requires having actually reasoned about the tradeoffs, not just memorized the outcome.
- It's a direct proxy for the actual job. A staff research engineer spends a large fraction of their time evaluating exactly this kind of tradeoff — "should we do X the way lab Y did it, or differently, and why" — inside their own organization's research program, not just about published external work.
- It surfaces intellectual honesty under mild social pressure. Critiquing a widely-praised recipe from a well-regarded lab, in front of an interviewer who may have deep respect for that work, requires a willingness to take a real position rather than hedge into vagueness — and conversely, having the discipline to *walk back* a critique once a strong counter-case is heard, rather than defending a position purely for consistency's sake, is an equally important and equally tested signal.
- It reveals calibration, not just opinions. The scorecard in the previous section exists specifically to make this legible: a candidate whose five critiques all land at "fully sustained, they were wrong" is giving a very different, and generally weaker, signal than one whose five critiques land at five different points on the sustained-to-withdrawn spectrum, because the latter pattern is what actually happens when critique is done honestly against genuinely well-executed work.

## What Distinguishes a Real Critique From a List of Facts

Across all five cases, the pattern worth internalizing: a genuine technical critique names a specific, concrete decision — not "the model has flaws" in the abstract — states precisely what a different choice would have cost and bought, and then argues the strongest available counter-case honestly enough that the final position has actually survived scrutiny rather than being asserted unopposed. The five critiques above land at deliberately different points on the "how much does this actually stick" spectrum:

- Critique 1 (overtraining's generalizability) is a real but narrow gap in the *argument*, not the *decision*, which was almost certainly right for Meta specifically.
- Critique 2 (risk-sequencing two ambitious bets in one run) critiques what the public report fails to demonstrate, not a confident claim that DeepSeek got the sequencing wrong.
- Critique 3 (GPT-3's alignment gap) survives the counter-case at the paper-scope level but is more defensible reframed as a field-level critique than a critique of one release decision.
- Critique 4 (the PRM negative result) is a critique of the field's citation practice more than of DeepSeek's own claim.
- Critique 5 (CAI's constitution authorship) is real and acknowledged, but symmetric with RLHF's own unexamined version of the same gap.

Being able to calibrate *how hard* to press a critique — some deserve to land firmly, some should be substantially walked back once the counter-case is heard, and none of the five above should be pressed all the way to "this was simply a mistake" — is itself the signal a staff interviewer is listening for, more than which specific model gets criticized.
