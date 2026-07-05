## Open Problems in Scaling and Data Efficiency

*Scope note: this file is about the pretraining data and compute-scaling debate specifically. The self-improvement and synthetic-data mechanisms that partially mitigate it are covered in depth in `002_Self_Improvement_And_Synthetic_Data_Flywheels.md` and only summarized here where directly relevant.*

### 0. Framing — this file is a genuine debate, not a settled survey

Unlike the pretraining-recipe file elsewhere in this collection, which treats causal decoder-only training as settled practice, this file's every major claim carries a confidence label, and the reader should expect to come away with a sharper sense of the shape of the disagreement rather than a resolved answer.

Every other file in this collection that touches scaling laws treats Kaplan et al. (2020) and Hoffmann et al./Chinchilla (2022) as settled empirical results: given a compute budget, there is a well-characterized, reproducible relationship between model size, data size, and loss. That result is not in question here.

What this file covers is what happens *next* — whether the inputs that made those laws deliver capability gains, fresh, high-quality human-generated text chief among them, will remain available at the rate frontier training runs have come to depend on, and whether the transformer-plus-next-token-prediction recipe itself keeps paying off as favorably as it has, or needs a fundamental rethink.

Credible, technically serious people disagree on both questions, and this file's job is to present that disagreement honestly rather than to resolve it. Anyone who claims high confidence in a definitive answer to either question should be read with skepticism, including this file's own author.

The two questions are related but separable, and it is worth tracking which one a given argument actually addresses: the data-wall question is about the availability of a specific input (Sections 1-3), while the architecture-rethink question is about whether the current recipe is the right one to apply to whatever input is available (Section 4) — a lab could face a severe data constraint while the current architecture remains the best available choice, or face no meaningful data constraint at all while still benefiting from an architectural change for other reasons (efficiency, different capability profile).

### 0.1 A note on why this file avoids naming specific companies' internal decisions

Several of the arguments below reference "industry-wide pivots" and "credible voices on multiple sides" without attributing specific strategic decisions to specific companies. This is deliberate: the actual internal reasoning behind any given lab's resourcing decisions between pretraining scale-up, RLVR investment, and data-efficiency research is not public information, and attributing a specific motivation to a specific lab based only on the observable pattern of its public releases would be exactly the kind of unfounded speculation this file's epistemic standards are meant to rule out. Where a specific, named, public claim exists (Epoch AI's estimates, Muennighoff et al.'s results), it is cited by name; where the underlying reasoning is inferred from a pattern of behavior, it is flagged as inference rather than presented as a confirmed fact.

### 1. The data wall — the claim, stated precisely

The concern: total-internet-scale, filtered, deduplicated, reasonably high-quality human-generated text is a **finite stock**, not a renewable resource growing at the rate compute has grown.

Compute, and the money behind it, has scaled by roughly an order of magnitude every couple of years across the frontier-training era; the total amount of human-written text on the internet grows at nowhere near that rate, because human text production is bounded by the number of humans writing things down, which does not compound the way GPU fleets do.

If frontier training runs' *token* requirements — as dictated by compute-optimal scaling, i.e., Chinchilla-style ratios of data to parameters — keep growing roughly in step with compute, and the highest-quality readily available fraction of that stock is already substantially consumed by current-generation frontier runs, then further scaling along the pretraining-data axis specifically, as opposed to compute or parameters, runs into a supply constraint that money alone cannot instantly solve, because you cannot buy text that was never written.

**Epoch AI and related published estimates** — Villalobos et al., "Will We Run Out of Data? Limits of LLM Scaling Based on Human-Generated Data," and Epoch AI's own tracking work — attempt to quantify this: estimating the total stock of human-generated text, indexed web text, books, and other sources, at somewhere on the order of a few hundred trillion to low quadrillions of tokens depending heavily on what counts as usable, and comparing that to the token counts frontier labs have publicly disclosed or are estimated to have used for their largest training runs, which, for several 2023-2025-era frontier runs, are already estimated to be in the tens of trillions of tokens, some of it via repetition (Section 3).

The Epoch AI-style projection is that, on current growth trajectories, the highest-quality, most useful fraction of the readily-available human text stock could be substantially exhausted for training purposes sometime in the second half of this decade, with a wide uncertainty band rather than a precise date.

**Why "high quality" is doing a huge amount of load-bearing work in this claim.** The raw size of "the internet" is enormous and still growing; the actual constraint is not raw byte count but the much smaller subset that is non-duplicated, reasonably well-written, and not itself low-value spam/boilerplate/junk — a growing fraction of newly-created web content is now itself LLM-generated, which raises its own contamination and quality concerns (Section 2's synthetic-data discussion).

Estimates of "how much of the internet is actually useful training data" vary across published analyses by well over an order of magnitude depending on filtering assumptions, which is precisely why the "when do we run out" date carries such wide uncertainty bands rather than being a single confident number. Treat any specific year quoted for "running out of data" as an illustrative estimate under stated assumptions, not a forecast with tight error bars.

This estimate uncertainty compounds with the definitional ambiguity in "usable" itself, since different published analyses do not even agree on a consistent operational definition of what counts as a usable token before getting to the separate question of how many such tokens exist.

A further complication worth naming explicitly: "usable" is itself a moving target, since data-efficiency research (Section 3) can, in principle, expand the effectively-usable fraction of a fixed raw stock over time, meaning the data-wall timeline and the data-efficiency research agenda are not independent variables — progress on one directly shifts the other, which is part of why a single, static "we run out in year X" framing understates how dynamic this picture actually is.

### 2. Counterarguments and mitigants

**2.1 Multimodal data as a large, comparatively untapped reserve.** Video, audio, and image data — especially video, given the sheer volume uploaded continuously — represent an enormous data source that frontier labs have only begun to exploit at pretraining scale for general-purpose language-and-reasoning capability, as opposed to purpose-built vision/video models.

The open research question is whether training on this data transfers usefully to the kind of general reasoning and world-knowledge capability text pretraining has delivered, or whether it mainly helps multimodal-specific tasks without meaningfully relieving the text-data constraint for general reasoning capability.

This is genuinely unresolved; there is real optimism in parts of the field — multimodal grounding could, in principle, teach a model about physical causality, spatial reasoning, and real-world dynamics that text alone under-represents — alongside real skepticism: video is enormously redundant frame-to-frame relative to its raw byte count, and the "effective," non-redundant information content per token of video may be much lower than for curated text, meaning the reserve may be far smaller in *useful* terms than its raw size suggests.

**2.2 Synthetic data as a mitigant, and its own sharply bounded ceiling.** As covered in depth in `002_Self_Improvement_And_Synthetic_Data_Flywheels.md`, model-generated data can supplement or partially substitute for human-generated text, but only reliably where an external anchor — a verifiable-reward checker, most concretely — keeps the generated data correct.

Unanchored synthetic data reintroduces the model-collapse and echo-chamber risks discussed there in detail, so "just generate more data synthetically" is not a clean, unconditional solution to the data wall; it is a solution with a specific, bounded domain of applicability — verifiable domains, chiefly math and code — rather than a general one.

**2.3 Repeated-epoch training — real, quantified, but sharply diminishing value.** Muennighoff et al., "Scaling Data-Constrained Language Models" (2023), directly studies the regime this file cares about: what happens if you are data-constrained and must repeat existing data rather than acquire fresh tokens?

The paper's central, precisely quotable finding: **repeating data has real value up to roughly four epochs, after which the marginal value of additional repetition decays sharply toward zero** — a model trained on a fixed corpus repeated ~4 times performs close to, though measurably below, a model trained on an equivalent number of *fresh* unique tokens, but pushing repetition much further yields rapidly diminishing returns, and past a certain point additional compute is better spent training a smaller model than repeating data on an oversized one.

The paper also proposes a data-constrained scaling law extending Chinchilla's framework to explicitly account for a fixed data budget, giving a compute-optimal model-size recommendation that shifts, toward smaller models, deployed at higher repetition, or trained for fewer total tokens, relative to the unconstrained Chinchilla recommendation once data is the binding constraint rather than compute.

This is one of the few genuinely quantitative, peer-reviewed results in this entire file, and it directly upper-bounds how far "just repeat the data you have" can carry a lab through a data-constrained regime — it buys real headroom, not indefinite headroom.

**2.4 Test-time-compute and RLVR as a partial structural escape.** A point worth making precisely because it's easy to miss: the RLVR-driven capability gains behind o1/o3 and DeepSeek-R1 (`..\..\GPT\008_O1_O3_Reasoning_Models.md`, `..\..\OpenSource\008_DeepSeek_R1.md`) did **not** require more pretraining tokens at all — they are a post-training and inference-time-compute story layered on an unchanged pretraining corpus and base model.

To the extent that a meaningful share of near-future capability improvement comes from this axis — test-time compute, covered in `003_Test_Time_Compute_And_Inference_Scaling_Research.md` — rather than from ever-larger pretraining runs, the practical bite of the pretraining data wall is softened, not because the data wall claim is wrong, but because the field's locus of investment has partly, visibly shifted toward an axis that doesn't depend on the same finite resource.

This is one of the more credible reasons some researchers argue the data wall matters less than headline "running out of data" framing suggests: labs are not purely and exclusively racing to pour more raw tokens into ever-larger pretraining runs, and the industry's own visible 2024-2025 pivot toward RL-and-reasoning-focused post-training is itself informative evidence that at least some frontier labs judged the marginal pretraining-token investment to be less valuable than the marginal RLVR/test-time-compute investment.

It would be overclaiming, however, to say this was *purely* a data-wall-driven decision rather than also being a genuine, independently-motivated capability discovery — the two motivations are not mutually exclusive and the public record does not let you cleanly attribute the shift to one over the other.

### 3. Data efficiency research — what's validated versus what's still small-scale

**3.1 Curriculum learning.** Ordering training data by some notion of difficulty or domain progression — easy-to-hard, or general-to-specialized — has a long history in machine learning generally and mixed, scale-dependent results specifically for LLM pretraining.

Some published academic-scale studies report modest gains from careful curriculum design; other work, including practical experience reported informally across the field, suggests that at genuinely frontier training scale, with sufficiently diverse and large corpora, the benefit of an explicit curriculum washes out relative to simply training on a well-filtered, well-mixed corpus in a randomized, or near-randomized, order — plausibly because at large enough scale and enough passes implicitly re-exposing the model to material in different contexts, an explicit difficulty ordering matters less than data quality and mixture proportions do.

**This is a case where the honest answer is "we don't know how much frontier labs actually rely on curriculum ordering,"** because none have published a detailed account of it — the academic curriculum-learning literature is real and peer-reviewed, but its scale is generally far below frontier pretraining scale, and extrapolating its findings to trillion-token regimes is not validated.

**3.2 Data selection, filtering, and quality-over-quantity.** This is the data-efficiency lever with the strongest evidence of frontier-scale relevance, even though exact recipes are undisclosed.

Quality classifiers — train a classifier to distinguish high-quality reference text, e.g., Wikipedia-adjacent or curated-book text, from generic web text, then use the classifier's score to filter or upweight the pretraining corpus — deduplication, now essentially universal practice since duplicated content both wastes training compute on redundant signal and has been directly linked to memorization/regurgitation risk, and importance-resampling techniques, DSIR-style methods that select a subset of a large, noisy corpus that best matches the distribution of a smaller, high-quality target corpus, are all real, published, and understood in the abstract to matter.

But there is no public, rigorously benchmarked account of exactly how much they move the needle at frontier scale for any specific frontier lab's actual training run, since the details of pretraining-corpus construction are among the most closely guarded specifics for essentially every model covered elsewhere in this collection.

**3.3 Active-learning-style and value-targeted data selection.** A more speculative direction: rather than filtering a fixed corpus for quality, actively identify which *additional* data — from a large candidate pool, or via a targeted data-acquisition effort — would most reduce loss or most improve a specific target capability if added to the training mix, analogous to active learning in the classical ML sense, but applied at pretraining-corpus-construction scale.

This remains largely an academic-scale research direction; the practical difficulty is that estimating "which not-yet-included document would most help" is itself an expensive prediction problem, and the field lacks a validated, efficient method for doing this at trillion-token corpus scale, as opposed to small-scale demonstrations.

**3.4 The honest gap between academic results and frontier practice.** A pattern worth naming explicitly, since it recurs across all three sub-sections above: a meaningful fraction of the *published, peer-reviewed* data-efficiency literature is conducted at a scale — millions to low billions of parameters, single-digit-billions of tokens — far below frontier pretraining scale, hundreds of billions of parameters, tens of trillions of tokens, and it is not established that findings transfer cleanly across that gap.

Frontier labs almost certainly do incorporate sophisticated, in-house data curation, filtering, and mixture-optimization work — the strong, consistent improvement in reported benchmark performance per parameter across successive model generations is difficult to explain purely by architectural changes, and data quality/curation is a widely-inferred, though rarely disclosed in technical detail, contributor.

But the *specific techniques and their *quantified* contribution are not public, and a staff-level answer should distinguish "there is good reason to believe frontier labs invest heavily and effectively in data curation" from "here is the specific, validated recipe," because only the former is something you can state with real confidence.

### 3.5 A synthesis of Section 3's three sub-findings

Pulling Sections 3.1-3.4 together: the field has a peer-reviewed, if scale-limited, evidence base for curriculum learning's mixed results, strong indirect evidence that data selection and filtering matter a great deal in practice even without a public quantified account of how much, and an honest admission that active-learning-style targeted data selection remains an academic-scale aspiration rather than a frontier-deployed technique.

The common thread across all three is the academic-to-frontier transfer gap named in Section 3.4 — and a staff-level answer should treat "frontier labs probably do some version of this" as a reasonable inference while being explicit that the specific, quantified recipe is not public information, rather than either dismissing the likely existence of sophisticated internal data engineering or overclaiming specific knowledge of what it looks like.

### 4. The architecture/objective rethink debate

**4.1 The case that current scaling continues to pay off.** The strongest empirical argument for staying the course: pretraining loss-versus-compute power laws have held remarkably consistently across roughly five to six orders of magnitude of compute, from early GPT-era models through the largest disclosed frontier runs, with no clearly observed structural break in the loss curve itself.

Proponents of continuing the current recipe — transformer architecture, next-token-prediction objective, scale everything — point out that essentially every "this won't keep working" prediction made at an earlier scale checkpoint, millions of parameters, then billions, turned out to be wrong at the next order of magnitude, and treat the burden of proof as being on anyone claiming *this* particular scale checkpoint is where it finally breaks down, rather than assuming it by default.

The additional, complementary argument: even if pretraining-loss-per-compute gains slow, RLVR and test-time compute (Sections 2.4 above, and `003_Test_Time_Compute_And_Inference_Scaling_Research.md`) demonstrate that useful capability gains are available along axes that don't require the pretraining recipe itself to change at all — so even a genuine pretraining-scaling slowdown would not necessarily mean overall frontier capability progress stalls, since the field has visibly diversified its investment across multiple scaling axes rather than betting everything on one.

**4.2 The case for a rethink.** Several distinct arguments, each worth stating separately rather than lumping into one vague "scaling might not continue" claim.

*The loss-versus-capability decoupling concern.* Pretraining loss — cross-entropy on held-out text — has continued improving smoothly, but there is a genuine, if hard-to-rigorously-quantify, informal observation across the field that *perceived* downstream capability gains per unit of additional loss improvement have felt smaller at the frontier than they did at earlier scale transitions.

That is, the loss metric may be continuing to improve on a clean power law while the *capabilities people actually care about* saturate or improve more slowly than the loss curve alone would predict, because loss is an average over the entire token distribution and most of that distribution, routine, low-information text, contributes the bulk of the loss signal while contributing comparatively little to the specific capabilities — hard reasoning, long-horizon planning, genuinely novel problem-solving — that matter most for the highest-value use cases.

This is a real, discussed concern but is difficult to state as a rigorous, quantified claim rather than an impression, precisely because "downstream capability" is not a single clean metric the way pretraining loss is — treat this as a genuinely contested, hard-to-adjudicate point rather than an established finding.

*The sample-efficiency gap relative to human learning.* Human children acquire fluent language competence and basic causal/physical reasoning from a minute fraction of the token count frontier LLMs are trained on — a gap of easily several orders of magnitude by any reasonable accounting.

This is a genuinely striking and frequently cited data point, and it underlies arguments — made publicly and repeatedly by researchers including Yann LeCun, among others, in the context of advocating for self-supervised world-model objectives distinct from pure next-token text prediction — that current architectures/objectives are statistically inefficient learners relative to whatever mechanism biological cognition uses, and that this inefficiency is masked rather than solved by simply having enormously more compute and data available than a human brain's developmental window does.

The counterargument, made by defenders of the current paradigm, is that this comparison is not apples-to-apples: human learning is embodied, multimodal, actively-interactive, and benefits from millions of years of evolved inductive bias baked into brain structure before any individual learning even begins, none of which a from-scratch-initialized transformer has access to — so the comparison may say more about the difference between learning-with-strong-priors-and-rich-interactive-multimodal-input versus learning-from-scratch-on-static-text than it says about any fixable inefficiency in the transformer architecture specifically.

*Alternative directions being actively explored, framed honestly as exploratory rather than proven.* State-space models and linear-attention variants — the Mamba lineage and related work — aiming for better long-context compute scaling than quadratic self-attention; retrieval-augmented architectures that offload some of the model's factual-memorization burden to an external, updatable store rather than requiring everything to be baked into parametric weights, a design that could, in principle, make the model itself smaller and more sample-efficient for reasoning while a retrieval system handles the "remember a huge amount of specific factual content" job separately; and JEPA-style, Joint Embedding Predictive Architecture, predictive objectives that aim to learn from raw sensory data by predicting learned latent representations of missing content rather than predicting the raw content itself, video pixels, for instance, argued by proponents to be a more sample-efficient and more world-model-like learning signal than pixel- or token-level generative prediction.

None of these has demonstrated frontier-scale general-purpose-LLM-competitive results as a *replacement* for the transformer-plus-next-token-prediction recipe as of this writing; each remains a genuinely open research direction with real academic and some industry investment, not an established alternative paradigm.

**4.4 A switching-cost argument that cuts across both sides of the debate.** Independent of which side of Section 4.1/4.2 is ultimately correct, there is a real economic argument for why the field would be slow to abandon the transformer-plus-next-token-prediction recipe even if a genuinely better alternative existed on paper.

Every layer of infrastructure investment discussed elsewhere in this collection — FlashAttention-style fused kernels, KV-cache-based serving, speculative decoding, the entire distributed-training tooling ecosystem — is built against the specific computational contract causal decoder-only transformers provide (see `..\02_Pretraining_Recipes_And_Objectives\001_Pretraining_Objectives_Overview.md`, Section 6.3, for the detailed argument). A genuinely different architecture or objective would need to justify not just a capability or efficiency improvement in isolation, but an improvement large enough to offset rebuilding a substantial fraction of this infrastructure investment from scratch.

This means the bar for a successful architectural challenger is higher than "shows a promising result in an academic paper" — it needs to be good enough to justify the switching cost, which is a separate and additional hurdle beyond the underlying research question of whether the alternative is genuinely better at matched scale.

This is a genuinely different kind of argument from either Section 4.1 or Section 4.2 — it is not a claim about whether the current recipe is *technically* optimal, but about the *economics of switching away from it* being unfavorable even for a technically superior alternative, which is worth distinguishing clearly if asked to weigh in on why a promising academic architecture hasn't been adopted at frontier scale.

**4.3 Credible voices, genuinely on multiple sides.** It is worth resisting the temptation to resolve this into a single "consensus" position, because there isn't one.

Some researchers and organizations — including voices at Epoch AI, and some public statements from individuals at multiple frontier labs — have treated the data-wall concern as real enough to be a first-order factor already visibly shaping investment decisions. The industry-wide pivot toward RL/test-time-compute and toward synthetic and agentic data-collection strategies in 2024-2025 is citable as at least partially consistent with labs hedging against a pretraining-data constraint, whether or not any given lab would frame its own strategy that way publicly.

Other credible voices argue the wall is much softer or further out than headline framing suggests, pointing to multimodal data's scale, continued improvements in data efficiency techniques, and the sheer scale of *not-yet-well-curated* web and licensed-content data as reasons the binding constraint is more "we haven't finished building the curation and multimodal-integration pipeline" than "the raw resource is gone."

Both positions are held by technically serious people with access to more detailed non-public information than is available externally, and the honest, calibrated stance for an outside observer — which, for essentially every reader of this file, is the actual epistemic position — is to hold this as a genuinely open empirical question rather than adopt either side's confidence level as your own.

### 5. How you would actually investigate this, if it were your job

A staff-level answer to "is my organization approaching a data wall" should propose something concrete and measurable rather than restate the debate. A reasonable research program:

1. **Track marginal loss improvement per additional unique token, at fixed compute, across successive training-corpus refreshes.** If genuinely fresh, high-quality data is becoming scarcer, you would expect the marginal value of each additional unique token — holding compute and model size fixed, varying only how much of the added tokens are genuinely new versus increasingly-diluted-by-repetition or increasingly-lower-quality — to decline in a measurable, trackable way over successive corpus-construction cycles. This is directly measurable internally even without knowing the field-wide answer, and gives an organization-specific, empirically grounded signal rather than relying on external, highly uncertain field-wide estimates.
2. **Monitor duplication and near-duplication rates in newly-scraped web data over time**, since a rising share of "new" web content being low-quality near-duplicates of existing content, including, increasingly, LLM-generated content circulating back onto the web, is a direct, measurable early-warning signal distinct from the harder-to-measure "are we running out of good data" question in the abstract.
3. **Use held-out canary evaluations designed to detect saturation** — a fixed, hard, contamination-resistant benchmark suite tracked across successive training runs, watching specifically for the point where additional data/compute stops moving the needle on genuinely held-out, uncontaminated hard tasks even while training loss on the corpus continues to improve, which would be a direct, empirical signal of the Section 4.2 loss-versus-capability decoupling concern rather than an impression.
4. **Explicitly quantify the Muennighoff et al.-style data-constrained scaling law for your own corpus and compute regime** rather than relying on the published paper's fitted constants, since the exact repetition-value-decay curve is corpus- and domain-dependent, and a lab operating near this constraint should want its own fitted version of this law, not someone else's.
5. **Track the field's and your own organization's marginal return on RLVR/test-time-compute investment versus marginal return on additional pretraining-token investment**, as a practical, decision-relevant proxy for whether the pretraining-specific constraint is actually binding for your capability roadmap, independent of resolving the abstract "is there a data wall" debate — this is the version of the question that actually matters for a resourcing decision, and it is answerable with internal experimentation even when the field-wide question isn't.

### 5.1 Why this program is internally answerable even when the field-wide question isn't

It is worth stating explicitly why Section 5's five-item program is a reasonable research plan despite the field-wide debate (Sections 1-4) remaining unresolved: every item measures something specific to the organization's own corpus, compute budget, and roadmap, rather than requiring an answer to the harder, less tractable question of the total global stock of human-generated text. This is the same move Section 7.1's worked scenario makes more concretely — decomposing an intractable global question into a set of tractable local ones.

### 6. A worked illustration: fitting a data-constrained scaling curve internally

To make Section 5's item 4 concrete, the following is a minimal skeleton for how an internal experiment along these lines would actually be structured — not a claim about any specific lab's implementation, but a useful sketch to offer if asked to operationalize this research program.

```python
from dataclasses import dataclass

@dataclass
class TrainingRun:
    unique_tokens: int
    epochs: float
    total_tokens_seen: int    # unique_tokens * epochs
    held_out_loss: float

def fit_repetition_value_curve(runs: list[TrainingRun]) -> list[dict]:
    """Given runs at fixed compute (total_tokens_seen roughly constant) but varying
    the split between unique_tokens and epochs, estimate the marginal value of an
    additional epoch of repetition versus an equivalent amount of fresh data."""
    runs = sorted(runs, key=lambda r: r.epochs)
    results = []
    for prev, curr in zip(runs, runs[1:]):
        d_epoch = curr.epochs - prev.epochs
        d_loss = prev.held_out_loss - curr.held_out_loss
        results.append({
            "epoch_range": (prev.epochs, curr.epochs),
            "marginal_gain_per_epoch": d_loss / d_epoch if d_epoch else float("nan"),
        })
    return results
```

The organization-specific decay point — where `marginal_gain_per_epoch` drops below whatever threshold makes repetition no longer worth it relative to acquiring fresh data or reallocating compute to test-time methods — is the internal analog of Muennighoff et al.'s roughly-four-epoch finding, and there is no strong reason to assume it will match the published constant exactly for a different corpus, domain mix, or model scale.

### 6.1 A related but distinct scarcity problem: evaluation contamination

It is worth separating the training-data-wall discussion above from an adjacent, easily-conflated concern: as LLM-generated content increasingly circulates on the open web, and as benchmark questions and their answers are increasingly likely to have been discussed, restated, or directly leaked into training corpora scraped from that same web, **clean, uncontaminated evaluation data is becoming scarce in a way that is structurally similar to, but mechanistically distinct from, the training-data-wall problem.**

A model that has seen a benchmark's questions, or close paraphrases of them, during pretraining will show inflated performance on that benchmark without the corresponding capability gain generalizing elsewhhere — meaning the data wall's effects can show up first, and more insidiously, as unreliable measurement rather than as an outright capability plateau. This is precisely why held-out canary evaluations (Section 5, item 3) need to be actively protected from public release and refreshed periodically, rather than treated as a fixed, reusable asset — a benchmark's usefulness for detecting genuine capability saturation decays over time in a way that is itself a data-scarcity problem, just one about evaluation data rather than training data.

### 6.2 Summary table: mitigant, mechanism, and evidentiary strength

| Mitigant | Mechanism | Evidentiary strength |
|---|---|---|
| Multimodal data (2.1) | New modality, unclear transfer to reasoning | Speculative; real optimism and real skepticism both credible |
| Synthetic data (2.2) | Anchored generation in verifiable domains only | Strong within its narrow domain; see `002_...` |
| Repeated-epoch training (2.3) | Reuse existing tokens | Strong, quantitative, peer-reviewed (Muennighoff et al.) |
| RLVR / test-time compute (2.4) | Shift investment off the pretraining-token axis entirely | Strong as a capability lever; attribution to data-wall-avoidance specifically is speculative |
| Better data selection/filtering (3.2) | Extract more value per token | Plausible and likely used; unquantified publicly |

### 6.3 A terminology note: "data wall" versus "compute wall"

These two terms are sometimes conflated in casual discussion despite referring to entirely different constraints. A **compute wall** would mean the physical, financial, or energy constraints on building ever-larger training clusters become binding — a hardware, capital, and power-generation question, largely outside this file's scope and more a matter of semiconductor supply chains and datacenter energy economics than of ML research per se. A **data wall**, this file's actual subject, means the *raw material* compute would be applied to becomes scarce even if unlimited compute were available to throw at it. The two constraints are logically independent — it is entirely possible to hit one without the other — and conflating them leads to imprecise claims like "scaling is ending" that don't specify which resource is actually claimed to be the binding constraint.

### 6.4 Open research questions

- **Can the marginal-value-of-additional-tokens measurement proposed in Section 5 be standardized well enough to become a comparable, cross-lab metric**, analogous to how pretraining scaling laws became a shared vocabulary across the field, or is data composition and quality too organization-specific for such a comparison to mean anything across different corpora?
- **How much of the observed frontier capability improvement per model generation is attributable to better data curation versus architectural or scale improvements**, given that Section 3.4 notes this is almost entirely undisclosed — is there any indirect, externally-observable proxy (e.g., tracking how benchmark performance per unit of disclosed compute has changed across generations at fixed reported parameter/data scale) that could shed light on this without requiring lab disclosure?
- **Does the sample-efficiency gap relative to human learning (Section 4.2) close meaningfully when interactive, embodied, or multimodal data is added to a training mix**, or does it persist even with those additions, which would be a much stronger piece of evidence for a genuine architecture/objective limitation rather than a missing-input-type explanation?
- **Is there a way to empirically distinguish "the field is investing in RLVR/test-time compute because pretraining data is genuinely becoming scarce" from "the field would have invested in RLVR/test-time compute regardless, because it is an independently valuable capability lever"**, given that Section 2.4 explicitly flags this attribution question as unresolved by the public record?
- **What would falsify the "current architecture and objective will keep scaling favorably" position, and what would falsify the "a fundamental rethink is needed" position**, stated precisely enough that a specific future empirical result could actually settle the debate in Section 4 rather than each side simply reinterpreting ambiguous evidence in its own favor?
- **Does the switching-cost argument in Section 4.4 create a systematic bias toward underinvestment in architectural alternatives** relative to their true expected value, given that the infrastructure-compounding dynamic makes early-stage alternatives look worse on a pure cost basis than they would if evaluated purely on research merit — and if so, is there a way to correct for this bias in how research investment decisions get made, or is the compounding cost a genuinely rational reason to discount early-stage alternatives more heavily?

### 6.5 Practical guidance for a team facing this question today

- Run Section 5's internal measurement program before adopting either side of the Section 4.3 debate as a planning assumption — an organization's own marginal-token-value trend is a better guide to its own resourcing decisions than any externally-published, field-wide estimate, given how wide the uncertainty bands on the latter are.
- Treat data efficiency research (Section 3) as worth piloting at small scale before committing frontier compute to it, precisely because Section 3.4's academic-to-frontier transfer gap means a technique's small-scale promise is not a reliable predictor of its frontier-scale payoff.
- Reassess the Section 4.4 switching-cost calculation periodically rather than once — infrastructure investment in the current recipe compounds over time, which means the bar for a successful architectural challenger rises the longer the field continues investing in the current paradigm, a dynamic worth factoring into any long-horizon research bet on an alternative architecture.
- Protect and periodically refresh held-out evaluation sets specifically because of Section 6.1's contamination dynamic — an evaluation suite that was clean a year ago is not guaranteed to still be clean, independent of any change in the model being evaluated.
- Distinguish, in any internal or external communication about this topic, between a claim about the raw data stock (Section 1), a claim about data efficiency (Section 3), and a claim about architecture (Section 4) — these are three separate debates with three separate evidence bases, and a single "is scaling ending" framing collapses distinctions that matter for deciding what to actually do about it.
- When communicating about this topic externally or internally, default to Section 6.3's compute-wall-versus-data-wall distinction explicitly, since conflating the two constraints is one of the most common sources of imprecise claims about "the end of scaling" in both public and internal discourse.

### 7. Cross-references and what to read next

- The self-improvement and synthetic-data mechanisms referenced in Section 2.2 are covered in full, including their anchoring requirements and collapse risks, in `002_Self_Improvement_And_Synthetic_Data_Flywheels.md`.
- The test-time-compute axis referenced in Section 2.4 as a partial structural escape from the pretraining data wall is covered in full in `003_Test_Time_Compute_And_Inference_Scaling_Research.md`, including its own open question about whether it has an analogous wall of its own.
- The capstone synthesis (`006_The_Next_Frontier_What_Staff_Researchers_Are_Actually_Working_On.md`, Section 5) situates data-efficiency and data-wall research within the broader set of research thrusts, with explicit confidence labeling of what is publicly discussed as a priority versus what is this document's own speculation.
- The pretraining-objective infrastructure argument in Section 4.4 draws directly on `..\02_Pretraining_Recipes_And_Objectives\001_Pretraining_Objectives_Overview.md`, Section 6, which develops the causal-decoder-only infrastructure-compounding argument in full outside the specific context of the data-wall debate.
- The multi-agent file's Cicero discussion (`004_Multi_Agent_Training_And_Emergent_Behavior.md`, Section 3.1) is a concrete example of the retrieval-augmented/hybrid-architecture idea gestured at in Section 4.2 — pairing a learned component with a separate, non-learned module to reduce the learned component's burden, applied there to strategic planning rather than factual memorization.

### 7.1 A worked scenario connecting several of this file's threads

Consider a hypothetical mid-2020s frontier training run facing a genuine question about whether to scale its pretraining corpus another 3x for the next model generation. The team runs Section 5's marginal-token-value measurement and finds the value of the most recently added third of its corpus is measurably lower than the value of the first third was for the prior generation — a concrete, internal early-warning signal, not a field-wide estimate.

Rather than concluding scaling has simply stopped working, the team separately evaluates Section 2.3's repeated-epoch option (is there value left in additional passes over the existing high-quality subset), Section 2.1's multimodal option (does adding a large video corpus move the needle on the specific reasoning benchmarks that matter for the product roadmap), and Section 2.4's RLVR option (would the same compute budget produce a larger capability gain if redirected to post-training RL instead of additional pretraining).

This is the concrete, decomposed version of the abstract Section 4 debate — and it illustrates why the debate does not need to be resolved in the abstract for a specific organization to make a well-reasoned resourcing decision: each of the three options is independently, internally testable, and the organization's actual answer can be empirical rather than philosophical.

### 8. Staff-level synthesis

The defensible position, stated as precisely as the evidence allows: there is real, published, technically serious support for the claim that high-quality human-generated text is a finite and non-trivially-constrained resource, with credible estimates, wide uncertainty bands notwithstanding, suggesting current-generation frontier runs are already consuming a meaningful fraction of the readily available high-quality stock.

There is also real, credible, technically serious pushback on how binding this constraint actually is once multimodal data, synthetic data (within its anchored domain of applicability), repeated-epoch training (within its own sharply diminishing-returns regime), and the industry's visible pivot toward RL/test-time-compute investment are all accounted for.

Repeated-epoch training's value is the one sub-claim in this entire file with a genuinely quantitative, peer-reviewed answer — roughly four epochs before sharp decay. Nearly everything else — the exact size of the usable data stock, the transferability of multimodal data to general reasoning capability, whether current architectures need a fundamental rethink versus continued scaling — remains a live, multi-sided debate among people with more information than any outside observer has access to.

The single strongest thing you can do in an interview on this topic is resist false confidence in either direction, name the specific quantitative results that do exist — Muennighoff et al., the Epoch AI data-stock estimates, with their explicit uncertainty — and propose a concrete, internally-answerable research program (Section 5) rather than a definitive verdict on a question the field itself has not settled.

This file's Section 7.1 worked scenario is worth carrying forward as a template for how to answer any open-ended question in this space during an interview: rather than picking a side of an abstract debate, decompose the question into its independently-testable components and describe how each would actually be measured. That move — decomposition into testable sub-questions rather than a global verdict — is a more reliable signal of research judgment than any specific position on data walls or architecture rethinks, precisely because the field's most credible voices disagree on the global verdict while broadly agreeing on how each sub-question would, in principle, be tested.
