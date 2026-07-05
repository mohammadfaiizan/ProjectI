# Benchmark Saturation, Contamination, and the Real-World Gap

Files 001-006 each covered a family of named benchmarks and flagged, individually, where each one had saturated or was showing signs of contamination. This file steps back and treats those observations as instances of a small number of general, recurring dynamics.

Three threads run through this file: a benchmark lifecycle pattern that has now repeated enough times to be treated as a predictable phenomenon rather than a series of unrelated coincidences; a contamination problem that is structural to how benchmarks and pretraining data both come from the same open web; and a persistent gap between what any benchmark score can measure and what "this model is actually good to deploy" requires.

Understanding this file well is arguably more valuable in a staff-level interview than memorizing any individual benchmark's numbers, because it is the layer of understanding that lets you correctly discount or contextualize any specific number you're handed.

## The saturation lifecycle

### The pattern, stated generally

A new benchmark is introduced specifically because existing benchmarks no longer separate models of interest — either because they're saturated (top models near the ceiling) or because they don't probe the specific capability the new benchmark's authors think matters.

At introduction, there is a wide gap between the strongest available models and either the benchmark's ceiling (100%, or an estimated human-expert ceiling) or a meaningful human baseline. Over the following one to two years, that gap closes rapidly — partly through genuine model improvement, partly through prompting-technique refinement (e.g., chain-of-thought), partly through models and their training pipelines being specifically tuned against benchmarks that have become widely used success metrics.

This last mechanism is worth naming honestly: once a benchmark is a headline number every paper and product announcement cites, there is direct competitive pressure to do well on it specifically, which is a different thing from improving general capability, even though the two are often correlated. Once frontier models cluster within a few points of the ceiling and of each other, the benchmark stops being able to distinguish "which frontier model is better" — the remaining variance is dominated by label noise, prompt-format sensitivity, and evaluation-harness differences rather than by real capability gaps — and the field's attention shifts to whatever the next benchmark is that still has headroom.

This is not a criticism of any individual benchmark's design; it is closer to a law of the field's incentive structure. The entire reason a benchmark gets adopted widely in the first place is that it currently discriminates well among models people care about, and that exact property is what erodes with success.

### MMLU to MMLU-Pro to GPQA, traced concretely

MMLU (2020) initially showed a wide spread across models, and was the default "how broadly knowledgeable is this model" number for several years. By 2023-2024, GPT-4-class and subsequent frontier models were reporting scores in the mid-to-high 80s, clustered close to the paper's own estimated ~89.8% human-expert baseline and close to each other. At that point, a 1-2 point MMLU difference between two frontier models is not a reliable signal of which is more capable, especially once you account for the label-error rate MMLU-Redux documented (file 001).

MMLU-Pro (2024) directly targeted both problems at once: more distractors (10 options instead of 4, cutting the guess-floor from 25% to 10%) restored raw headroom, and its filtering-plus-CoT-sensitivity design restored discriminative power specifically among models that reason well versus models that pattern-match.

GPQA (2023, actually published slightly *before* MMLU-Pro but representing the same underlying response to MMLU's exam-style-question ceiling) took a different axis entirely — rather than making the same style of question harder, it changed *what kind* of question was being asked (graduate-level, adversarially validated against web lookup) to reopen a large human-model gap by construction.

Both are now visibly on the same trajectory MMLU was: GPQA Diamond scores for frontier reasoning models climbed from the 30s-40s at introduction into the 70s-80s within roughly two years — a faster saturation curve than MMLU's own, plausibly because the field had already built the RL-on-verifiable-reward and long-chain-of-thought machinery (see the DeepSeek-R1 discussion in `../../OpenSource/008_DeepSeek_R1.md`) that generalizes well to hard science-reasoning questions once it had been developed for math and code.

### GSM8K to MATH to AIME, traced concretely

The same shape recurs on the math side, covered in more mechanical depth in file 002. GSM8K (2021) saturated for frontier models by around 2023 (>95% for models with decent chain-of-thought prompting), at which point it stopped being informative for comparing frontier models, even though it remains useful as a floor-level sanity check for smaller or weaker models.

MATH (2021, but harder from the start, so it saturated later) followed the same arc — by 2024-2025, frontier reasoning models were reporting numbers in the 90s on the commonly used MATH-500 subset (DeepSeek-R1's self-reported ≈97.3%, flagged as approximate), again clustering near ceiling.

AIME-style evaluation is the field's response, but notably via a *different* mechanism than either MMLU-Pro's harder-question-construction or GPQA's different-question-type approach. Rather than a purpose-built benchmark at all, the field adopted an already-existing, externally-validated hard-problem source (a real competition, administered independently of any benchmark-design process) specifically because its recurring yearly cycle offers a structural (if temporary and decaying, as discussed in file 002) contamination advantage that a static, purpose-built benchmark cannot replicate.

This is worth noting as a genuinely different *kind* of response to saturation than the MMLU lineage's: escalate difficulty by importing external, naturally-occurring hard problems rather than by re-engineering the benchmark-construction methodology itself.

### The general lesson

Across both traced lineages, the specific *mechanism* of escalation differs (more distractors and re-annotation; a different question genre and validation methodology; importing an external, naturally refreshing problem source), but the *trigger* is identical in both cases: convergence of frontier models near a ceiling within roughly one to two years of a benchmark's introduction and adoption.

A reasonable rule of thumb, worth stating explicitly in an interview: if a benchmark has been the standard headline number for more than about two years, and current frontier models are reported within a few points of each other and of an estimated ceiling on it, treat that specific number as low-information for distinguishing current frontier models — regardless of how prominently it's still being cited. Prominence in citation lags the benchmark's actual loss of discriminative power, often by a wide margin, because switching the field's default headline metric takes time even after the old one has stopped being useful.

## The contamination problem, from the benchmark's perspective

This file's job is not to re-derive contamination-detection methodology — n-gram/substring overlap checking against training corpora, canary strings, decontamination filtering pipelines, and temporal train/test holdout design are covered in `../05_Evaluation_Methods/004_Contamination_Aware_Evaluation_Design.md`, and that is the right place to look for the mechanics of *detecting or preventing* contamination. What belongs here is the narrower point of how contamination specifically undermines a benchmark's validity across its lifecycle, which is a slightly different question from "how do you catch it."

### Why any sufficiently popular, static benchmark is a contamination target by default

A benchmark's value depends on being used and cited widely — that's what makes it the field's shared yardstick. But the same visibility that makes it valuable as a shared reference also guarantees it ends up reproduced, discussed, quoted, and mirrored across the open web (academic aggregator sites, dataset-hosting platforms, blog posts, discussion forums, other papers' appendices quoting example questions) at a scale that makes exclusion from any sufficiently broad web crawl increasingly difficult over time.

GPT-3's paper (see `../../GPT/003_GPT3.md`, Section 5) already flagged this as a known, imperfectly-solved problem in 2020, using its own contamination-filtering attempt as an explicit self-reported caveat rather than a solved issue. This is not a new or newly discovered problem — it is one that has been acknowledged since essentially the first paper in this document's reference lineage, and it has not gotten structurally easier since, because the volume of benchmark-discussing web content has only grown as the field's benchmark culture has grown.

### Contamination degrades validity in a way that's hard to detect from the score alone

A model that has memorized a benchmark's answer key — or, short of verbatim memorization, has been repeatedly exposed to highly similar practice material covering the same specific questions — will score well on that benchmark without the score reflecting the underlying capability the benchmark was designed to measure. But the resulting number looks identical, on its face, to a genuine capability score.

This is exactly why contamination is corrosive to a benchmark's *credibility* even when its actual prevalence is uncertain: once a benchmark is known to be a plausible contamination target, every reported score on it carries an unresolvable asterisk, and reasonable people can disagree about how much to discount any specific model's specific number without a rigorous, independently-run contamination audit — itself expensive and not always feasible for a lab without access to a model's actual training data, especially for evaluating a competitor's or another lab's model.

### Contamination risk differs systematically by benchmark shape

This connects directly to the escalation mechanisms described above:

| Benchmark shape | Contamination dynamic | Example |
|---|---|---|
| Static, perennial | Accumulates monotonically over time, no natural refresh mechanism | MMLU, MATH, HumanEval |
| Freshly-generated-per-instance | Sidesteps the risk almost entirely (regenerated with new random content each run) | RULER's synthetic task templates |
| Naturally-recurring external source | Temporary reprieve at introduction, decays as that instance gets discussed | AIME's yearly cycle |
| Agentic/trajectory-based | Strategy-overlap risk with no verbatim-text requirement, hard to detect via n-gram tools | WebArena, GAIA (file 003) |

Recognizing which of these shapes a given benchmark has is itself a useful diagnostic for how skeptically to treat its longevity.

### The self-reinforcing loop with the saturation lifecycle

Contamination and saturation are not independent phenomena. A benchmark that is contamination-prone will *look* like it is saturating (scores climbing toward ceiling) even in a world where genuine underlying capability is not actually improving at the same rate, because contamination-driven score inflation and genuine-capability-driven score improvement are observationally similar from the outside — both show up as "scores went up over time."

This means some fraction of every saturation curve traced in the previous section is plausibly attributable to contamination rather than to real capability gains, and there is no clean way to apportion credit between the two without independent, contamination-controlled re-evaluation — precisely the kind of analysis the sibling evaluation-methodology module's contamination-aware design techniques exist to make possible, and precisely why this file defers to it rather than re-deriving it.

## Why labs keep citing saturated benchmarks anyway

A reasonable question at this point is why model cards and launch announcements keep prominently featuring MMLU, GSM8K, and HumanEval numbers years after this file's argument that they've stopped being discriminative among frontier models. Several genuine, non-cynical reasons coexist with the more cynical ones:

- **Comparability with historical results.** A benchmark's main remaining value once saturated for frontier models is as a fixed point for comparing against older models, smaller models, and open-weight models that are *not* yet saturated on it — omitting it entirely would break that continuity for a large fraction of the field's actual comparison needs, even if it says little about frontier-vs-frontier comparisons specifically.
- **Audience expectations and inertia.** A model card that omits MMLU reads as evasive to an audience that has learned to expect it as a baseline sanity check, regardless of whether an expert reader would weight it heavily — this is a real communication constraint, not a methodological one.
- **Genuine remaining signal at the smaller-model end of the market.** A 3B or 7B open-weight model's MMLU score is nowhere near saturated and still meaningfully separates weak models from strong ones at that scale, so the same benchmark can be simultaneously uninformative for one comparison (frontier-vs-frontier) and informative for another (small-model-vs-small-model) — the saturation critique in this file is scoped to frontier-model comparisons specifically, not a blanket claim that MMLU is worthless everywhere.
- **Less charitably: a saturated benchmark with a model still narrowly ahead is easy, low-cost, positive-sounding marketing content**, even when the underlying gap is statistically indistinguishable from noise (see file 001's discussion of GPQA Diamond confidence intervals for the general version of this problem) — a healthy skepticism toward headline benchmark numbers in launch materials is warranted precisely because the incentive to report a favorable-looking number never fully goes away, independent of how informative that number actually is.

## A second concrete historical example of the lifecycle: HellaSwag and ARC

MMLU and GSM8K are this document's primary running examples, but the saturation lifecycle is not unique to them — it is worth being able to name at least one more instance to demonstrate the pattern is general rather than cherry-picked. HellaSwag (Zellers et al., 2019) and ARC (the AI2 Reasoning Challenge, Clark et al., 2018) were, in the years just before MMLU's introduction, among the standard commonsense-reasoning and science-QA benchmarks cited to differentiate models. Both were effectively saturated for frontier models well before MMLU itself saturated — HellaSwag in particular became well known for models reaching human-level or above-human-level reported accuracy while still visibly failing at the kind of commonsense reasoning the benchmark was nominally designed to probe, a widely-discussed early example of the exact gap between benchmark score and underlying capability that motivates this file's final section. Both benchmarks are now used more as minimum-bar sanity checks for new small/open models than as frontier differentiators, exactly the fate this file predicts for MMLU and GSM8K, and exactly the fate GPQA Diamond and AIME should be expected to eventually share once frontier models saturate them too.

## The benchmark-score-vs-deployed-usefulness gap

### Framing the question: "tops the leaderboard, users complain it's worse"

This is worth working through concretely because it forces several distinct mechanisms — rather than one vague "benchmarks aren't everything" hand-wave — into the open:

1. **Distributional mismatch between the benchmark and real usage.** MMLU is multiple-choice exam-style knowledge recall; real usage is overwhelmingly open-ended generation, multi-turn conversation, instruction-following on underspecified requests, and often domain- or organization-specific tasks the benchmark never represented in the first place. A model can genuinely be better at MMLU's specific task shape while being worse at open-ended helpfulness, instruction adherence, or conversational tone — correlated but not identical capabilities, and a training change (e.g., a data mix shift, a new RLHF pass) can move them in different directions.
2. **Single-turn/isolated-capability benchmarks miss multi-turn robustness and reliability.** As file 003's discussion of tau-bench's pass^k makes explicit, a model's single-attempt success rate on an isolated task and its reliability across repeated, structurally similar real interactions are different properties — a model could look equally or more capable on any single-shot benchmark while being less consistent in a way that shows up as user-visible flakiness only once you look at repeated real usage.
3. **Goodhart's-law-style benchmark-targeting ("benchmaxxing").** If a lab's post-training pipeline includes data that resembles benchmark-style questions — deliberately, or as an unintended side effect of using benchmark-adjacent public datasets in training — the model can improve specifically on the benchmark's exact task distribution without a proportional improvement, or even with a regression, in the broader distribution of real usage the benchmark was only ever meant to be a *proxy* for. This is the single most direct explanation for the "tops the leaderboard, users complain" scenario, and it's worth stating explicitly as a named phenomenon ("when a measure becomes a target, it ceases to be a good measure") rather than just describing the symptom.
4. **Aggregate scores hide sub-population and failure-class variance.** A macro-averaged benchmark score (MMLU across 57 subjects, BBH across 23 tasks) can improve in aggregate while regressing on a specific slice that happens to matter disproportionately to real users — a change that improves 50 of MMLU's 57 subjects while regressing on the other 7 still raises the aggregate, and the aggregate alone gives no way to know whether those 7 happen to overlap heavily with what your actual user base asks about.
5. **The benchmark doesn't measure the failure mode users are actually hitting.** Complaints like "it hallucinates more confidently," "it's worse at following formatting instructions," "it feels less careful," or "it regressed on our specific internal use case" are not things MMLU (or most static ground-truth benchmarks) were ever designed to detect — MMLU has no mechanism to penalize confident wrongness beyond the specific multiple-choice question at hand, no mechanism to check instruction-following fidelity, and no visibility whatsoever into an internal enterprise use case that never resembled any of its 57 subjects.

### The actual investigation, concretely

A staff-level response to the leaderboard-vs-complaints scenario should not stop at "benchmarks aren't everything" — it should describe a concrete diagnostic path:

1. Pull a sample of the specific user complaints and categorize the failure mode (formatting/instruction-following regression? factual confidence miscalibration? tone/verbosity change? a specific task category?) rather than treating "worse" as a monolithic complaint.
2. Check whether the new model's training pipeline changed in a way that plausibly explains a benchmark/reality divergence — a new RLHF/preference-optimization pass tuned partly against human-preference signals (see file 006's style-and-length-gaming discussion) can measurably shift output style in ways that read as "worse" to users used to the old model's style even when underlying correctness is unchanged or improved.
3. Run a benchmark or benchmark-like eval that is *closer* to the actual complaint category (an instruction-following-specific eval, a domain-specific held-out set resembling your actual product traffic, a multi-turn conversation eval) rather than relying on MMLU to adjudicate a complaint MMLU was never built to detect.
4. Check for regression on your own product's historical eval suite, if one exists, since an internal eval built from your actual user distribution is a far more direct signal than any public benchmark for a "did this get worse for our users specifically" question.
5. Consider contamination/benchmaxxing directly — if the new model's MMLU gain is suspiciously large relative to its gains elsewhere, ask what changed in its training data or post-training pipeline that could specifically target MMLU's task shape without a matching general-capability gain.

This kind of layered, hypothesis-driven investigation — not a single "run more benchmarks" reflex — is what distinguishes a staff-level answer to this class of question.

### Why this gap is structural, not a temporary embarrassment

No finite, static, automatically-gradable benchmark suite can fully specify "useful in deployment," because deployment usage is open-ended, adversarial in ways benchmarks aren't (real users occasionally try to break the model, benchmarks don't), organizationally specific (an enterprise's actual use case is not represented in any general-purpose public benchmark), and constantly evolving (what users ask for changes as products change and as users learn what the model can and can't do).

This is not a claim that benchmarks are worthless — the entire rest of this document is a case for their real, load-bearing value in tracking genuine capability progress and comparing models on well-defined axes — but it is a claim that the gap between benchmark score and deployed usefulness is a permanent structural feature of using any finite proxy metric for an open-ended real-world goal, not a bug that the next benchmark generation will finally close.

The correct posture, and the one worth articulating in an interview, is to treat benchmarks as one class of evidence to be triangulated against others — human preference signals with their own known biases (file 006), product-specific evals, live monitoring of real usage and complaint patterns, red-teaming for safety-relevant gaps (file 005) — rather than as a single authoritative scalar that settles the question of which model is "better."

## A worked example of disentangling saturation from contamination

Suppose a model shows a 12-point jump on MATH relative to its predecessor from the same lab, released eight months later. Before concluding "the model's mathematical reasoning improved by roughly 12 points," a disciplined analysis would check several things in sequence:

1. **Does the gain replicate on a benchmark with the same task shape but a different, less-contaminated instance set?** If a comparably-difficult, less-publicly-discussed problem set (e.g., a private held-out set, or the most recent AIME sitting) shows a much smaller or absent gain, that's evidence the MATH-specific jump is at least partly contamination or benchmark-targeting rather than general capability.
2. **Did the training pipeline change in a way that specifically increases exposure to MATH-shaped data?** New RLVR-style training against math-competition-style verifiable rewards (see the DeepSeek-R1 GRPO discussion) is exactly the kind of pipeline change that would produce a large, genuine, and *specifically* math-concentrated capability gain — which is a different, benign explanation for a large isolated jump, distinct from contamination.
3. **Is the gain consistent with the model's gains elsewhere?** A 12-point MATH jump alongside a comparable jump on GPQA, AIME, and BBH's arithmetic-heavy tasks is a consistent signature of a genuine underlying reasoning improvement (plausibly from exactly the kind of RL-on-verifiable-rewards training just mentioned). A 12-point MATH jump with flat or negative movement everywhere else is a signature far more consistent with contamination or narrow benchmark-targeting.

None of these checks is individually conclusive, but running all three and reporting the composite picture is the difference between a defensible claim ("this looks like genuine reasoning improvement, corroborated across three independent evals") and an indefensible one ("MATH went up, so the model reasons better now").

## Common interview framings worth preparing for

- **"You're told a new open-weight model matches GPT-4 on MMLU. Should you be impressed?"** — the calibrated answer is "less than the headline suggests, and it depends which specific number and protocol." MMLU has been saturated and public long enough that a strong MMLU number alone is weak evidence of frontier-level general capability; the more informative follow-up questions are what the same model scores on GPQA Diamond, AIME, and SWE-bench Verified — benchmarks not yet saturated and with less contamination runway — and whether those numbers tell a consistent story.
- **"A lab claims a new benchmark result that no other lab has been able to reproduce. How do you weigh that?"** — treat self-reported, unreplicated numbers with real skepticism proportional to how surprising the result is and how much incentive the lab has to report a favorable number (which is to say, always some incentive) — and explicitly ask what harness, prompt template, and number of samples were used, since these alone can produce large, benign-looking discrepancies (see file 001's and file 002's discussions of scoring-protocol sensitivity).
- **"How would you design an internal evaluation practice that doesn't fall into the saturation-and-contamination trap the public benchmarks did?"** — maintain a private, held-out eval set built from your own product's real usage distribution and refreshed on a regular cadence (so no single snapshot has time to leak or saturate), track performance on it alongside public benchmarks rather than instead of them, and treat any public-benchmark score as context for, not a substitute for, that internal signal.

## Distinguishing three lookalike explanations for the same observed pattern

A model scoring near ceiling on an old, widely-used benchmark is consistent with at least three different underlying stories, and this file's argument is essentially that you cannot tell which from the score alone:

| Story | What's actually happening | What would help distinguish it |
|---|---|---|
| Genuine capability convergence | Multiple labs have independently reached a real capability ceiling on the skill the benchmark measures | Corroborating gains on a fresh, uncontaminated benchmark measuring a similar skill |
| Contamination-driven inflation | The benchmark's content (or close paraphrases) leaked into training data across the field | A large score/behavior gap between the public benchmark and a private, held-out set of similar difficulty |
| Benchmark-targeting / benchmaxxing | Post-training data specifically resembling the benchmark's task shape became more prevalent without a matching general-capability gain | A gain on the specific benchmark that doesn't replicate on structurally similar but distinct evals, paired with a training-pipeline change that plausibly explains the mechanism |

In practice, real saturation curves are very likely a blend of all three simultaneously, in proportions that are generally not possible to cleanly separate from published information alone — which is exactly why file 007's overall recommendation is to triangulate across multiple evals and, wherever the stakes justify the cost, run independent/private verification rather than trusting any single published number to resolve the question.

## Quick-reference: the escalation ladders

| Domain | Gen 1 (saturated) | Gen 2 (saturating) | Gen 3 / current frontier | Escalation mechanism |
|---|---|---|---|---|
| Knowledge/reasoning | MMLU | MMLU-Pro | GPQA (Diamond) | More distractors + re-annotation, then a harder question genre |
| Math | GSM8K | MATH | AIME (yearly) | Harder answer-equivalence problem, then an external refreshing source |
| Code | HumanEval / MBPP | — | SWE-bench (Verified) | Task-realism escalation (function to whole-repo) |
| Long context | Needle-in-a-haystack | RULER | (open) | Task-diversity decomposition |

## How this file's three threads show up differently across the document

It is worth explicitly tracing which of files 001-006's benchmarks are most exposed to each of this file's three concerns, since the exposure is not uniform:

- **Most saturation-exposed:** MMLU, GSM8K, HumanEval/MBPP, classic single-needle NIAH — all are old enough (2020-2023) and simple enough in task shape that frontier models have had ample time to close the gap to ceiling.
- **Most contamination-exposed:** anything static and perennial that's been public for multiple years and is a standard fixture in eval tables — MMLU, MATH, HumanEval chief among them — versus AIME (temporarily protected each year, decaying fast) and RULER/agentic benchmarks (protected from verbatim-text contamination but exposed to the different strategy-leakage/trajectory-contamination risk described in file 003).
- **Most exposed to the deployment gap regardless of saturation or contamination status:** the human-preference leaderboards of file 006 (structurally, by design, only ever as representative as their voter population) and the agentic benchmarks of file 003 (structurally bottlenecked by scaffold confounds and environment narrowness) — these two families' core limitations are not primarily about saturation or contamination at all, but about a different kind of validity gap that no amount of "make the benchmark harder" or "refresh the benchmark" fixes, because the gap is about what population or environment the benchmark represents, not about how hard its questions are.

## What a good benchmark-reporting practice looks like, given all of the above

Pulling the file's argument into a positive prescription rather than only a critique: a reporting practice that takes saturation, contamination, and the deployment gap seriously would, at minimum,

1. **Report confidence intervals, not bare point estimates**, especially for small benchmarks (GPQA Diamond, AIME) where a few flipped answers materially changes the headline number.
2. **Report the evaluation protocol alongside the number** — few-shot count, prompting template, scoring method (log-likelihood vs. generation-plus-parsing, exact-match vs. symbolic-equivalence) — since, as files 001 and 002 both demonstrate, protocol differences alone can produce non-trivial score differences that have nothing to do with underlying capability.
3. **Prefer benchmarks with a known, recent contamination profile over older, more heavily-cited ones** when the goal is genuinely differentiating frontier models, and explicitly flag when a reported number comes from a benchmark that's been public for multiple years.
4. **Corroborate large or surprising jumps across multiple, ideally structurally different, evals** before treating them as evidence of a real capability change, rather than accepting a single benchmark's headline number in isolation.
5. **Pair any public-benchmark claim with an internal, product-specific signal** wherever the actual decision at stake (a launch, a model swap, a procurement choice) has real stakes tied to real deployed usage rather than to benchmark standing alone.

None of these fully solves the underlying structural problem this file describes — no reporting practice can make a finite, static proxy metric fully capture an open-ended real-world goal — but each one closes a specific, avoidable gap between what a benchmark number appears to say and what it actually supports.

## A note on timing: how far behind does citation lag actual usefulness?

It's worth being concrete about the lag this file references between "a benchmark has stopped being discriminative for frontier models" and "the field stops citing it prominently," since the size of that lag is itself informative. MMLU was arguably no longer discriminative among the very top frontier labs by sometime in 2023 (once GPT-4-class models clustered in the mid-80s), yet it remained, and largely still remains as of this writing, a standard line item in nearly every subsequent model card and technical report — a lag measured in years, not months. This is not necessarily a failure of judgment on the field's part; as the "why labs keep citing saturated benchmarks anyway" section above lays out, there are legitimate comparability and audience-expectation reasons for the lag, alongside the less charitable marketing-incentive explanation. The practical lesson for an interview setting is to expect this lag as a normal feature of the field rather than being surprised that a benchmark is still prominently cited well after this file's own analysis would suggest its frontier-discriminative power has substantially eroded.

## Synthesis

The three threads in this file — saturation, contamination, and the deployment gap — are not three unrelated caveats to append to a benchmark discussion; they compound. A benchmark that becomes popular enough to matter accumulates contamination exposure over time, which artificially extends and inflates its apparent saturation curve, which delays the field's recognition that the benchmark has stopped being informative, all while the underlying deployment-usefulness gap persists regardless of where any given benchmark happens to sit on that saturation curve.

The single most useful habit this file is trying to instill is treating every benchmark number handed to you — in a paper, in a product announcement, in an interview question — as an artifact with a specific, nameable set of failure modes (which of files 001-006's specific weaknesses could be inflating or deflating it, how long it's been public and thus how contamination-exposed it plausibly is, and what specific slice of real capability it was ever designed to represent) rather than as a settled, self-interpreting fact.
