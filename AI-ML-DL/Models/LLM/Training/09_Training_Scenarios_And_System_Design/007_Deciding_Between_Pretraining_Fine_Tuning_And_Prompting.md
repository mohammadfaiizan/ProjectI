# Deciding Between Pretraining, Fine-Tuning, And Prompting

## Step -2: The Question Behind the Question

A product team asking "should we fine-tune or pretrain" is often actually asking a different question they haven't fully articulated: "how do I get this model to do the specific thing I need, as cheaply and reliably as possible, in a way that doesn't become my team's permanent maintenance burden." Reframing the request this way before answering it is itself a useful first move in a real conversation, because it surfaces the actual constraints (what specifically isn't working today, what's the budget, what's the deployment target) that the rest of this framework needs as inputs, rather than accepting "fine-tune or pretrain" as the literal menu of options being requested.

## The Scenario

"A product team comes to you wanting a model specialized for their domain — say, legal contract review, or clinical documentation, or some internal enterprise codebase. Walk me through how you decide between prompting an existing model, fine-tuning one, or investing in a full pretraining/continued-pretraining approach."

This is a decision-framework question, not a "here's how fine-tuning works" mechanics question, and the interviewer is checking whether you default to the cheapest viable option and can articulate *why* you'd escalate past it, rather than defaulting to whichever technique happens to be intellectually interesting. I'll build the decision framework around four axes — data availability, budget, latency/deployment constraints, and distributional distance from the base model — and then show how they combine into an actual recommendation, plus the maintenance-burden dimension that's easy to underweight in a one-time decision but dominates total cost of ownership over time.

## Step -1b: A Quick FAQ on Scope

- **Does this framework apply the same way to a multimodal specialization request (e.g., a domain-specific vision-language task)?** The same four axes still apply, but Axis 4 (distributional distance) typically needs assessment on both the language and the visual/modality-specific distribution separately, since a request can be a knowledge gap on one modality and a pure format gap on the other.
- **What if the product team explicitly asks for pretraining because they want to "own the whole stack" for strategic reasons unrelated to capability?** This is a legitimate input, but it should be surfaced explicitly as a strategic decision requiring executive sign-off (per Step 6's fourth bullet) rather than folded into the technical recommendation as if it were capability-driven.
- **Is there a version of this framework for choosing between two already-fine-tuned candidate models rather than between techniques?** That's a different, evaluation-shaped question (per `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md`), not this file's decision — this framework is specifically about which *technique* to invest in, not which resulting artifact to ship.

## Step -1: Common Mistakes This Framework Is Designed to Prevent

- Defaulting to fine-tuning because it "feels like the real engineering solution," without first attempting and honestly evaluating prompting/RAG.
- Treating "we have a lot of data" as automatically justifying pretraining-scale investment, without checking whether that data volume is actually large enough to matter for a next-token-prediction objective (thousands of examples is fine-tuning-scale, not continued-pretraining-scale).
- Fine-tuning on a narrow dataset without budgeting for the catastrophic-forgetting risk covered in `011_Interview_Questions_Part1.md`, Q14.
- Ignoring the recurring maintenance cost of whichever approach is chosen, and being surprised a year later that the specialized model has fallen behind the general frontier.
- Assuming an on-premises/air-gapped deployment constraint by itself settles the technique choice, when it only settles which base models are eligible, not whether prompting, fine-tuning, or continued pretraining is the right escalation level on top of that eligible set.

## Step 0: The Default Should Be Prompting, and the Burden of Proof Is on Escalating Past It

Before working through the axes, state the prior explicitly, because it's the single most common thing a weak answer gets backwards: **prompting an existing frontier model (with retrieval augmentation and in-context examples where useful) is the default starting point for almost any new domain-specialization request**, not fine-tuning, and definitely not pretraining. The reasons this should be the prior, not just one option among equals:

- It requires zero training infrastructure, zero training data curation beyond a handful of examples, and can be iterated on in minutes rather than days or weeks.
- Frontier models' in-context learning capability (the core finding of GPT-3, `..\..\GPT\003_GPT3.md`, Section 7) means a well-constructed prompt with a clear task description and a handful of demonstrations often gets most of the way to acceptable performance on a genuinely new domain, especially if the domain isn't wildly outside the base model's pretraining distribution.
- It preserves the ability to swap the underlying model as better ones ship, without any retraining — a maintenance-burden advantage (Step 5) that's easy to undervalue in the initial decision but dominates the multi-year cost picture.

The question the rest of this framework answers is: **what specific evidence justifies moving past prompting**, and if you do move past it, **what specific evidence justifies moving past fine-tuning to full (continued) pretraining**. Each escalation should be justified by a concrete, checkable failure of the cheaper option, not by a vague sense that "this domain feels specialized enough to deserve fine-tuning."

## Step 0b: The Four Axes at a Glance

| Axis | Key question | Pushes toward prompting | Pushes toward fine-tuning | Pushes toward continued pretraining |
|---|---|---|---|---|
| 1. Data availability | How much, and what form? | Little to none | Moderate labeled set | Large raw domain corpus |
| 2. Budget | Orders of magnitude | Tens of $K | Hundreds to low $K-$10Ks | $10Ks-$100Ks+ |
| 3. Latency/deployment | Hosted API OK? Cost/latency floor? | Hosted API acceptable | Smaller model needed for cost/latency, or on-prem with a moderate gap | On-prem plus a large capability gap |
| 4. Distributional distance | Format gap or knowledge gap? | No real gap found | Shallow format/behavior gap | Genuine knowledge/vocabulary gap |

## Axis 1: Data Availability

**What determines the answer.** How much high-quality, task-representative data actually exists, and in what form (labeled examples, raw domain documents, human preference comparisons, nothing at all beyond a task description)?

- **Little to no domain data (a handful to a few dozen examples).** This is a strong prompting-only signal — there isn't enough data to fine-tune reliably anyway (fine-tuning on a few dozen examples risks catastrophic overfitting to those specific examples' surface patterns rather than the underlying task), and few-shot in-context examples are the appropriate mechanism for this data regime.
- **A moderate labeled dataset (hundreds to low tens of thousands of examples), well-matched to the target task format.** This is the regime where fine-tuning starts to earn its cost — enough data to meaningfully shift model behavior via supervised fine-tuning without just memorizing the training set, but nowhere near enough to justify pretraining-scale investment (pretraining-scale corpora are trillions of tokens; a fine-tuning dataset in the thousands-to-tens-of-thousands-of-examples range is many orders of magnitude smaller and simply cannot support learning genuinely new world knowledge or a genuinely new domain distribution the way pretraining/continued-pretraining on a large domain corpus can).
- **A large, domain-specific raw corpus (many billions to trillions of tokens) — e.g., a large internal codebase, a large corpus of internal clinical notes, a large legal-filing archive.** This is the regime where **continued pretraining** (further next-token-prediction training on the domain corpus, before any task-specific fine-tuning) becomes viable and potentially valuable, because there's enough raw text to actually shift the model's underlying distributional knowledge of the domain's vocabulary, style, and factual content — not just its surface-level task format, which is what fine-tuning on a smaller labeled set primarily changes.
- **No data, and the domain is intentionally novel enough that no comparably-distributed public data exists anywhere** (e.g., a genuinely novel scientific subfield, a brand-new internal formal specification language). This is the rare case that can justify data *generation* investment (synthetic data pipelines, or a deliberate human-data-collection program) before any of prompting/fine-tuning/pretraining can even be evaluated — worth naming explicitly because "we don't have the data" is sometimes the real answer to "which training approach," disguised as a training-technique question.

## Step 1b: A Quick FAQ on the Default-to-Prompting Prior

- **Doesn't fine-tuning always produce a "more real" solution?** No — this conflates engineering effort with correctness. If prompting/RAG meets the quality bar at a fraction of the cost and with far lower maintenance burden, it is the objectively better engineering decision, not a shortcut.
- **What's the fastest way to lose credibility with a product team on this question?** Recommending fine-tuning or pretraining reflexively, without first attempting and measuring a well-constructed prompting/RAG baseline — a team that later discovers prompting alone would have sufficed will reasonably question every other technical recommendation that follows.
- **When is it acceptable to skip the prompting baseline entirely?** Only when Axis 3 (a hard on-premises/air-gapped requirement with no hosted-API path) or Axis 4 (an already-diagnosed, unambiguous knowledge gap from prior experience with this exact domain) makes the outcome of that baseline attempt a foregone conclusion — and even then, it's worth stating explicitly why the baseline is being skipped, not silently omitting it.

## Axis 2: Budget

**What determines the answer.** Orders-of-magnitude matter far more than precise numbers here:

- **Prompting/RAG:** effectively the API-call cost of the existing model, plus modest engineering effort to build retrieval and prompt-construction infrastructure — commonly low tens of thousands of dollars of total engineering investment for a solid implementation, dominated by engineering time, not compute.
- **Fine-tuning (SFT, or a lightweight adapter method like LoRA):** compute cost scales with base model size and dataset size but is typically a small fraction of pretraining cost — full fine-tuning of a mid-size open model on a moderate dataset is commonly in the hundreds to low thousands of dollars of raw compute, parameter-efficient methods (LoRA and similar) often cheaper still, though real total cost includes the data-curation and evaluation engineering effort around the training run, which usually dominates the raw compute line item.
- **Continued pretraining on a large domain corpus:** cost scales with the token count processed, using the same `C≈6ND` estimation approach as `001_Estimating_Training_Compute_And_Cost.md` — continuing to train an existing 70B-class model on an additional 50-100B domain tokens is a meaningfully larger compute investment than fine-tuning (commonly tens to hundreds of thousands of dollars depending on scale) but still far below full pretraining-from-scratch cost, because you're leveraging an already-pretrained model's existing general capability rather than learning language and reasoning from zero.
- **Full pretraining from scratch:** the full cost structure worked in `001_...` — millions to tens of millions of dollars for a frontier-adjacent model. This is essentially never the right call purely for *domain specialization*; it's justified only when the target capability genuinely cannot be reached by adapting an existing base model (a fundamentally different tokenization/architecture requirement, an extreme quality bar that no available base model's pretraining distribution can be adapted to reach even after continued pretraining, or a strategic/IP requirement to own the entire model stack) — and a staff engineer's job in this conversation is often to push back on a product team's assumption that pretraining is warranted, not to plan how to execute it.

## Step 3b: A Quick FAQ on Budget Sizing

- **How precise does the budget comparison need to be to make this decision?** Not very — the decision is almost always determined by which order of magnitude the budget falls into (tens of $K vs. hundreds of $K vs. $M+), not by precise dollar figures, so a rough `6ND`-style estimate (per `001_Estimating_Training_Compute_And_Cost.md`) is sufficient input.
- **What if the budget is ambiguous or not yet fixed by the requesting team?** Push back and get an order-of-magnitude number before proceeding — the entire escalation framework depends on knowing which budget tier applies, and designing a recommendation without it risks recommending an approach the team can't actually afford.
- **Does a larger budget ever argue for a cheaper approach anyway?** Yes — if prompting/RAG already meets the quality bar, spending a larger available budget on fine-tuning anyway is not "better," it's wasted spend and unnecessary future maintenance burden; budget size should never override a clean diagnostic result showing the cheaper approach already works.

## Axis 3: Latency / Deployment Constraints

**What determines the answer.** How and where the model will actually be served changes which technique is viable independent of data/budget considerations:

- **Calling a hosted frontier-model API is acceptable** (no on-premises/air-gapped requirement, latency budget compatible with a network round-trip to a third-party API): prompting/RAG against a hosted model remains viable even at larger scale, and fine-tuning APIs offered by the same hosted providers extend this without requiring you to run your own training or serving infrastructure at all.
- **On-premises or air-gapped deployment is required** (common in regulated domains — healthcare, defense, some financial-services contexts): this immediately constrains you to open-weight models (or a privately-hosted commercial model under a specific enterprise agreement), which changes the *available base model* but not fundamentally the prompting-vs-fine-tuning-vs-pretraining decision framework itself — it's a constraint on Step 4 below (which base model), not a reason by itself to skip straight to fine-tuning or pretraining.
- **A tight latency/cost budget favors a much smaller model than the largest available frontier model** — and this is where fine-tuning's actual comparative advantage over prompting-a-bigger-model becomes concrete: a well-fine-tuned smaller model can match or exceed a much larger general-purpose model's performance on a narrow, well-specified task, at a fraction of the per-query serving cost and latency (this is structurally the same "smaller model, more targeted training, cheaper to serve" logic behind DeepSeek-R1's distillation result, where SFT-only transfer onto much smaller dense models captured most of the reasoning benefit at dramatically lower serving cost than the full 671B teacher, `..\..\OpenSource\008_DeepSeek_R1.md`, Section 9) — so a hard latency/cost constraint is one of the strongest legitimate reasons to escalate past prompting even when data and budget alone might not have forced the decision.

## Step 4c: A Quick FAQ on Deployment Constraints

- **Does an on-premises requirement always force fine-tuning over prompting?** No — it forces the choice down to open-weight models, but a well-constructed RAG pipeline against a privately-hosted open-weight model can still be the right first attempt before escalating to fine-tuning; the deployment constraint changes *which models are eligible*, not automatically *which technique tier is required*.
- **How much does latency actually matter relative to raw capability in practice?** For an interactive, user-facing product surface, latency is frequently as decision-relevant as capability — a noticeably slower but marginally more capable model is often a worse product outcome than a faster, slightly less capable one, and this tradeoff deserves the same explicit weighing as any capability comparison.
- **Is a smaller fine-tuned model ever *more* capable than prompting a larger general model, on the narrow task?** Yes, routinely — a well-fine-tuned smaller model can match or exceed a much larger general-purpose model on a narrow, well-specified task, which is exactly the "smaller, more targeted training beats bigger and general" logic underlying `..\..\OpenSource\008_DeepSeek_R1.md`'s distillation result.

## Axis 4: Distributional Distance From the Base Model's Training Distribution

**What determines the answer.** This is the axis most often skipped in a shallow answer, and it's the one that actually determines whether fine-tuning alone will *work at all*, independent of how much data or budget is available:

- **The target domain is a stylistic/format specialization of something the base model already understands reasonably well** (e.g., "write in this specific legal-memo format," "always structure output as this particular JSON schema," "adopt this house writing style") — this is squarely fine-tuning's (or even well-constructed prompting's) sweet spot, because the underlying knowledge and reasoning the task needs is already present in the base model; what's missing is a comparatively shallow behavioral/format shift, which SFT on a few thousand well-chosen examples reliably achieves.
- **The target domain uses substantially different vocabulary, notation, or factual content than anything in the base model's pretraining distribution** (e.g., a specialized scientific notation, a large proprietary codebase with idioms and internal APIs that don't resemble public code, a low-resource language or dialect underrepresented in the pretraining mixture) — this is the regime where fine-tuning alone tends to underperform, because SFT primarily reshapes *how* the model responds within its existing knowledge, and struggles to inject large amounts of genuinely new factual/distributional content efficiently; this is exactly the case for escalating to **continued pretraining** on a large domain corpus before layering task-specific fine-tuning on top, giving the model a chance to actually absorb the new distribution's vocabulary and content patterns via the same next-token-prediction objective that built its original knowledge, before fine-tuning teaches it the specific task format.
- **A practical diagnostic**, worth naming explicitly: if a well-executed prompting/RAG approach with good retrieval and strong in-context examples still fails specifically on domain *knowledge* gaps (the model doesn't know facts/terminology/conventions the domain corpus would teach it) rather than on task *format* gaps (the model knows the relevant facts but doesn't structure its answer the way the product needs), that's a distributional-distance signal pointing toward continued pretraining being necessary before fine-tuning can be expected to help — fine-tuning a model that doesn't have the underlying knowledge will, at best, teach it to sound confidently fluent in a domain it still doesn't actually know, which is a worse outcome than an honest "I don't know" from a model that hasn't been trained to fake domain fluency.

## Step 4b: Maintenance Burden at a Glance

| Approach | Recurring cost when a better base model ships | Portability |
|---|---|---|
| Prompting/RAG | Low — swap the base model, re-validate prompts | High — retrieval corpus and prompts are largely model-agnostic |
| Fine-tuning | Moderate — re-run or re-validate the fine-tune against the new base | Medium — the fine-tuning dataset transfers, the trained artifact doesn't |
| Continued pretraining | High — close to redoing the investment from scratch | Low — the investment is baked into one specific base checkpoint's weights |

## Step 4d: A Quick FAQ on the Distributional-Distance Diagnostic

- **What's the fastest way to run the format-gap-vs-knowledge-gap diagnostic in practice?** Take the failures from a well-executed prompting/RAG attempt and sort them by hand into "the model didn't know this" versus "the model knew this but presented it wrong" — this simple triage, done on even a few dozen failures, is usually enough to reveal which regime the request is in.
- **Can a domain look like a knowledge gap when it's actually a retrieval gap?** Yes, and this is a common false positive — before concluding a genuine knowledge/distributional gap exists, verify the retrieval corpus in the RAG attempt actually contained the relevant information; a retrieval-configuration problem masquerading as a model-knowledge problem would otherwise incorrectly justify an expensive continued-pretraining escalation.
- **Is distributional distance a binary (present/absent) or a continuum?** A continuum — most real requests fall somewhere between "pure format shift" and "entirely novel vocabulary and facts," and the escalation decision should track roughly where on that continuum the specific request actually falls, not force it into one of two buckets artificially.

## Step 5: The Maintenance Burden Over Time — The Dimension Most Answers Skip

This is the axis that most distinguishes a staff-level answer from a mid-level one, because it's invisible at the moment of the initial decision and dominates cost several months to years later:

- **Prompting/RAG maintenance burden:** low and largely automatic — when a new, better base model ships, you can typically swap it in with modest re-validation effort, since the domain knowledge lives in the retrieval corpus (which is maintained independently of any model) and the task specification lives in the prompt (which is model-agnostic, modulo minor re-tuning). This portability is a genuine, durable advantage that compounds every time the field's frontier model improves.
- **Fine-tuning maintenance burden:** moderate and recurring — every time a meaningfully better base model ships, the organization faces a decision to re-run the fine-tuning pipeline against the new base (re-curating or at least re-validating the fine-tuning dataset against the new model's behavior) or fall behind the frontier while keeping the fine-tuned model on an aging base. This is not a one-time cost; it's a standing commitment to a re-fine-tuning cadence, and an organization that fine-tunes without budgeting for this recurring cost commonly ends up serving an increasingly stale specialized model well after better general-purpose options have shipped.
- **Continued-pretraining maintenance burden:** the highest of the three, and the one most likely to be underestimated at decision time — a continued-pretraining investment is tied even more tightly to a specific base model checkpoint (the domain corpus was absorbed *into* that specific model's weights via a training run that took real compute and time to execute), and re-running it against each new base-model generation is close to redoing the full continued-pretraining investment from scratch, not a lightweight re-validation. An organization that commits to continued pretraining is implicitly committing to either repeating a non-trivial training investment on a recurring cadence, or accepting that its specialized model will drift further behind the general frontier over time — a tradeoff that should be made explicitly, with eyes open, rather than discovered as a surprise a year into production.

## Step 5a2: A Quick FAQ Bridging Into the Worked Examples

- **Why walk through three separate examples rather than one thoroughly analyzed case?** Because the four-axis framework's real value is in showing how the same four questions produce genuinely different recommendations depending on the specific combination of answers — a single example risks looking like a fixed recipe rather than a live decision process.
- **Do all three examples need to be memorized verbatim for an interview?** No — the method (run the four axes, characterize the gap, match to the escalation ladder) is what should be internalized; the three examples exist to demonstrate the method being applied, not to be recited as fixed case law.
- **What's the fastest way to adapt these examples if an interviewer presents a genuinely different domain?** Map the new domain onto the same four axes explicitly, out loud, before offering a recommendation — the structure transfers even when the specific domain doesn't match any of the three worked cases exactly.

## Step 5b: Three Worked Product Requests, Walked Through the Framework

**Request 1: "We want a model that reviews legal contracts for our internal team."**

- Data availability: the org has a few thousand internally-reviewed contracts with annotator comments — a moderate labeled dataset.
- Distributional distance: legal contract language and reasoning patterns are reasonably well-represented in most frontier models' pretraining data (contract law is a widely-discussed public domain) — this is closer to a format/behavior gap than a knowledge gap.
- Deployment: no hard on-premises requirement; a hosted API is acceptable.
- **Recommendation:** start with prompting/RAG against a frontier model, using the internal contract corpus as a retrieval store rather than fine-tuning data; escalate to fine-tuning only if a careful diagnostic shows a specific, recurring format/behavior gap prompting can't close.

**Request 2: "We want a model that drafts clinical documentation matching our hospital system's specific note-taking conventions and internal drug-formulary references."**

- Data availability: a large corpus of past clinical notes exists, but under strict privacy/compliance constraints on how it can be used for training.
- Distributional distance: general clinical language is present in most frontier models' training data, but the hospital's *specific* formulary references, note templates, and internal abbreviations are not — a genuine knowledge/distributional gap, not just a format gap.
- Deployment: likely requires an on-premises or tightly access-controlled deployment given health-data compliance requirements.
- **Recommendation:** fine-tune an open-weight model (deployment constraint forces this choice regardless of what a hosted option might otherwise offer), using a carefully compliance-reviewed subset of the note corpus; continued pretraining on the full internal corpus is a plausible escalation only if fine-tuning alone doesn't close the formulary-specific knowledge gap after a proper diagnostic.

**Request 3: "We want a model that understands our 15-year-old, 40-million-line internal codebase and its idiosyncratic internal frameworks."**

- Data availability: the full codebase and its history exist as a very large raw corpus (likely billions of tokens once history, documentation, and internal wikis are included).
- Distributional distance: the internal frameworks and idioms are, by construction, not represented anywhere in any public pretraining corpus — this is squarely a knowledge/distributional gap, not a format gap, and prompting/RAG alone will plateau on questions requiring deep familiarity with the internal frameworks' conventions rather than just retrieving the relevant file.
- Deployment: internal developer tooling, latency-sensitive (interactive coding assistant use case).
- **Recommendation:** continued pretraining on the codebase corpus, followed by task-specific fine-tuning (code completion/chat format) on top — the codebase's scale and its idiosyncrasy relative to any public pretraining distribution are exactly the two conditions Axis 1 and Axis 4 name as jointly justifying this heavier escalation.

## Step 5c: A Decision Flowchart, Condensed

```
Start: prompting/RAG against best available base model
  |
  v
Characterize the gap (if any): format/behavior, or knowledge/distributional?
  |
  +-- No meaningful gap --> Ship prompting/RAG. Done.
  |
  +-- Format/behavior gap, moderate labeled data available --> Fine-tune. Done
  |     (budget for recurring re-fine-tuning against future base-model upgrades)
  |
  +-- Knowledge/distributional gap, large domain corpus available
  |     --> Continued pretraining, then fine-tune on top. Done.
  |         (budget for the heaviest recurring maintenance cost of the three paths)
  |
  +-- Gap persists despite all of the above, for reasons unrelated to cost
        --> Escalate to full pretraining only with executive-level sign-off,
            per 001_Estimating_Training_Compute_And_Cost.md's cost profile.
```

## Step 5d: A Quick FAQ on the Worked Examples

- **Do these three requests cover the realistic range of what a product team might ask for?** Reasonably well across the two axes that matter most (data availability and distributional distance), but every real request should still be run through the full four-axis framework rather than pattern-matched to the nearest worked example — the worked examples illustrate the method, not a lookup table.
- **What would change Request 1's recommendation from prompting to fine-tuning?** A demonstrated, specific format/behavior gap that a well-constructed prompt genuinely can't close (e.g., a very rigid, unusual output schema the model keeps drifting from) — the recommendation is conditional on the diagnostic outcome, not fixed in advance.
- **What would change Request 3's recommendation away from continued pretraining?** A materially smaller or younger codebase, or one built on mostly-standard, publicly-known frameworks rather than idiosyncratic internal ones — it's the combination of scale and genuine novelty relative to any public pretraining distribution that justifies the heavier escalation, not codebase size alone.

## Step 6: Putting It Together — A Concrete Decision Sequence

1. **Start by attempting prompting/RAG against the best available base model**, and treat this attempt as a genuine diagnostic exercise, not a pro-forma step to get past before doing what you already wanted to do. Characterize *where specifically* it falls short — format/behavior gaps versus knowledge/distributional gaps (Axis 4's diagnostic) — because that characterization directly determines the next step.
2. **If the gap is a format/behavior gap and there's a moderate labeled dataset (or one can be constructed at reasonable cost) representative of the target task**, fine-tune. Budget for the recurring re-fine-tuning cost against future base-model upgrades from the start, not as an afterthought.
3. **If the gap is a genuine knowledge/distributional gap and there's a large domain corpus available**, escalate to continued pretraining on that corpus, followed by task-specific fine-tuning on top of the continued-pretrained checkpoint — the two are complementary, not competing (continued pretraining shifts the underlying knowledge distribution; fine-tuning then shapes the task-specific behavior on top of that shifted distribution), and this combination is the standard pattern for genuinely deep domain specialization at reasonable cost.
4. **Only consider full pretraining from scratch if none of the above is structurally viable for reasons unrelated to cost alone** (a required architecture/tokenizer change no existing open base model supports, a strategic requirement to own the full model stack independent of any available base model's license or availability) — and even then, treat this as an organizational-strategy decision requiring executive-level sign-off given the cost profile in `001_Estimating_Training_Compute_And_Cost.md`, not a default engineering recommendation.
5. **At every stage, weigh the latency/deployment constraint (Axis 3) explicitly against the data/budget/distributional-distance conclusion** — a hard on-premises or cost/latency requirement can force fine-tuning of a smaller open model even when a hosted frontier model's prompting performance would otherwise have been sufficient, and this should be named as a deployment-driven escalation, distinct from a capability-driven one, so the eventual recommendation's actual justification is legible to the product team rather than looking like an arbitrary technical preference.

## Step 6b: Why This Question Recurs at the Staff Level

This scenario is a favorite precisely because the "obviously sophisticated" answer (recommend the most technically advanced option) is usually the wrong one, and the discipline of defaulting to the cheapest viable technique and demanding specific evidence before escalating is exactly the muscle a staff engineer needs when a product team, understandably excited about a new capability, asks for more engineering investment than the actual requirement justifies. A candidate who reflexively reaches for fine-tuning or pretraining without first characterizing the gap has demonstrated enthusiasm for technique, not judgment about when technique is actually warranted — and judgment about when *not* to build something is a large part of what separates a staff engineer from an engineer who simply executes whatever is asked.

The single sentence version of this whole framework, worth having ready verbatim in an interview: **escalate from prompting to fine-tuning to continued pretraining only in proportion to a specifically diagnosed gap — format versus knowledge versus deployment constraint — and price every escalation not just at its one-time cost but at its recurring maintenance cost against a base-model landscape that keeps improving out from under whatever you build today.**
