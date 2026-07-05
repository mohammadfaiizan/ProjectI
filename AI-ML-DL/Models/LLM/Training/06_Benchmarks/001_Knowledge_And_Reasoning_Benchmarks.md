# Knowledge and Reasoning Benchmarks

This file covers the four benchmarks most commonly cited as the "general knowledge and reasoning" scorecard for a frontier LLM: MMLU, its successor MMLU-Pro, GPQA, and BIG-Bench-Hard (BBH).

All four share a multiple-choice or short-form-answer format, which matters more than it sounds. Multiple choice is cheap and deterministic to grade — no LLM-judge, no human rater needed. That is exactly why it became the default format for large aggregate benchmarks. But it also caps how much these benchmarks can tell you about a model's actual reasoning process versus its ability to recognize a plausible-looking option.

This module is about naming and critiquing the specific named benchmarks; for the general methodology of *how* to design a good eval (few-shot vs zero-shot protocol, contamination controls, LLM-as-judge, statistical significance) see `../05_Evaluation_Methods/` — this file assumes that methodology and focuses on what each named benchmark actually is, how it was built, and where it specifically breaks.

## MMLU (Massive Multitask Language Understanding)

**Citation:** Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt, "Measuring Massive Multitask Language Understanding," ICLR 2021 (arXiv 2020).

### What it measures

MMLU is an aggregation of 57 subject-area multiple-choice test sets spanning four broad categories:

- **STEM** — e.g., abstract algebra, college physics, elementary mathematics
- **Humanities** — e.g., philosophy, world religions, jurisprudence
- **Social sciences** — e.g., econometrics, sociology, high-school government
- **Other** — e.g., nutrition, marketing, professional medicine, professional accounting

Each subject contributes on the order of 100-1500 questions, for roughly 15,908 questions total across the standard test split (plus a small few-shot dev set and a validation set used for hyperparameter selection, not for the reported score).

Every question is 4-way multiple choice (A/B/C/D), sourced originally from real-world sources — practice tests for the US medical licensing exam, GRE/AP exams, professional certification exams, and textbook question banks — rather than being purpose-written for the benchmark. This is a deliberate design choice: the authors wanted to measure the kind of knowledge a broadly educated adult (or domain professional) would be tested on, using genuine exam-style questions rather than synthetic ones.

### A representative question

To make the format concrete, a typical MMLU item (professional-law subject, paraphrased in style) looks like this:

```
Question: A state statute requires all bicycles sold within the state to be
equipped with a bell. A manufacturer based in another state challenges the
statute as an undue burden on interstate commerce. Which constitutional
provision is most directly implicated?

A) The Commerce Clause
B) The Equal Protection Clause
C) The Establishment Clause
D) The Takings Clause

Answer: A
```

Nothing about this item requires multi-step derivation — it requires recognizing which of four plausible-sounding legal doctrines the fact pattern maps to. That recognition-shaped structure, repeated across all 57 subjects, is exactly what MMLU-Pro's authors argued was too easily solved by pattern-matching rather than reasoning (see below).

### Construction and scoring mechanics

The canonical evaluation protocol is 5-shot (5 worked examples from the same subject prepended in-context, no gradient updates) with the model's next-token log-probability compared across the four answer-letter tokens (A/B/C/D), and the argmax taken as the model's answer.

This detail matters in practice: MMLU in its original form is scored via log-likelihood comparison over answer tokens, not via free-form generation parsed for an answer. The "5-shot MMLU" number reported for GPT-3-era base models is not directly comparable to a chat-model number obtained by prompting the model to *generate* a letter and then parsing the response, because the two protocols can diverge. A badly instruction-tuned chat model asked to "answer with a single letter" might ramble first and get misparsed by a naive regex, even while placing high probability mass on the correct token. Most modern reports use generation-based scoring with careful answer extraction, but which protocol was used is not always stated alongside the number — a minor but real comparability problem across papers.

```python
# MMLU log-likelihood scoring (the original protocol), sketched
def score_mmlu_question(model, prompt_with_few_shot, choices):
    # choices = ["A", "B", "C", "D"]; prompt already ends right before the answer letter
    logprobs = {c: model.logprob_of_next_token(prompt_with_few_shot, c) for c in choices}
    predicted = max(logprobs, key=logprobs.get)
    return predicted

# Aggregate score: macro-average across the 57 subjects, not micro-average across
# all questions -- a subject with 100 questions counts as much as one with 1500.
# This deliberately weights *subject coverage* over raw question count, so a model
# strong on the numerous STEM-heavy subjects but weak on a handful of small
# humanities subjects will not automatically dominate.
def mmlu_score(per_subject_accuracy: dict) -> float:
    return sum(per_subject_accuracy.values()) / len(per_subject_accuracy)
```

### Known weaknesses (documented, not speculative)

**Verified ground-truth errors.** MMLU-Redux (Gema et al., 2024, "Are We Done with MMLU?") manually re-annotated a stratified sample across all 57 subjects and found that a non-trivial fraction of questions have incorrect gold labels, ambiguous phrasing, multiple defensible correct answers, or are unanswerable as written.

Reported error rates vary sharply by subject — some subjects (their headline example is virology) show error rates high enough that a meaningful share of a model's "wrong" answers on that subject are actually correct answers marked wrong by a bad label. This is a real, verified finding, not a hypothetical critique. It means any single model's reported MMLU accuracy has a nontrivial, subject-dependent noise floor baked into the *labels*, independent of the model's actual capability, and that floor differs by subject in ways that can shift a model's ranking on close comparisons.

**Answer-order / position sensitivity.** Multiple independent studies have shown that shuffling which letter position (A/B/C/D) holds the correct answer measurably changes model accuracy on the same underlying question. Some models exhibit a detectable prior bias toward particular letters (e.g., a tendency to answer "C" when uncertain) — a symptom of surface-form pattern-matching rather than the option-content driving the decision. This is a confound that log-likelihood-over-letters scoring cannot easily distinguish from genuine reasoning ability. Robustness checks that permute answer order and check for consistency are a common mitigation, but they are not part of the original standard protocol.

**Contamination.** MMLU was published in 2020 and has since been quoted, discussed, mirrored, and re-hosted across enough of the open web that direct verbatim exposure during pretraining is a documented concern for any model trained on a broad web crawl after roughly 2021. The original paper's own train/test contamination caveat (see the GPT-3 discussion of Common-Crawl contamination) applies here by extension. Because MMLU is multiple choice, contamination can manifest subtly: a model doesn't need to have memorized the answer key verbatim to benefit from having seen highly similar practice-exam material, which makes contamination on MMLU harder to detect via naive n-gram overlap than on a benchmark with unique free-form answers.

**Saturation.** MMLU is now close to ceiling for frontier models. GPT-4 reported 86.4% at launch (2023); by 2024, models including GPT-4o, Claude 3 Opus/3.5 Sonnet, and Llama 3.1 405B were clustered in the high 80s (approximate, self-reported by each lab — treat exact decimal figures as best-recollection rather than independently re-verified). The original paper's own estimate of unspecialized-human-expert performance is around 89.8%.

Once frontier models are within a couple of points of that estimated human-expert ceiling, and of each other, MMLU stops being discriminative among the models that matter most for a "which frontier model is better" comparison. The remaining gap is dominated by label noise (the MMLU-Redux finding above) rather than by real capability differences — which is precisely the motivation for MMLU-Pro.

**Multiple-choice format ceiling.** Four options means a 25% random-guess floor, and partial-knowledge elimination strategies (ruling out two obviously wrong options and guessing between the remaining two) let models look more capable than they are at free-form recall. This is a generic multiple-choice critique, but it applies directly to MMLU's format choice.

## MMLU-Pro

**Citation:** Wang, Ma, et al., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark," NeurIPS 2024 (Datasets & Benchmarks track).

### What it changed and why

MMLU-Pro is a direct, explicit response to MMLU's saturation and noise problems. It is not a superset or simple filter of MMLU; it is a reconstructed benchmark of 12,032 questions that makes four changes at once:

1. **Expands from 4 to 10 answer choices**, dropping the random-guess floor from 25% to 10% and substantially reducing the benefit of elimination-based guessing.
2. **Removes trivial and noisy questions**, identified via a filtering pipeline: questions that weaker models could answer correctly with high consistency across many prompt perturbations were flagged as too easy or too pattern-matchable, then manually reviewed.
3. **Increases the share of reasoning-intensive questions** relative to pure-recall questions, drawing additional questions from harder sources including graduate-level exams.
4. **Adds a human-verification pass** on the added/modified questions specifically to reduce the label-error rate that MMLU-Redux exposed in the original set.

### The chain-of-thought sensitivity result

This is the headline empirical finding, and the direct evidence the redesign worked. The MMLU-Pro paper reports that models lose substantially more accuracy on MMLU-Pro than on MMLU when chain-of-thought prompting is removed — i.e., when the model is forced to answer directly without reasoning through the problem first. The reported drop is on the order of 16-33 percentage points across models tested, versus a much smaller CoT-ablation effect on original MMLU.

This is the concrete evidence for the benchmark's stated goal: MMLU could largely be answered by pattern-matching against memorized facts or surface heuristics, whereas MMLU-Pro's questions are constructed so that skipping explicit multi-step reasoning costs real accuracy. It is measuring something closer to "can the model reason through this," rather than "does the model recognize this."

### Reported numbers and remaining headroom

(Approximate, self-reported by labs — flagged.) At introduction, GPT-4-class models scored in roughly the 70s (percent), meaningfully below their MMLU scores in the high 80s, restoring the discriminative headroom that MMLU had lost. By 2025, frontier reasoning models (o1-class, Claude 3.5/3.7 Sonnet, DeepSeek-R1) pushed into the 80s on MMLU-Pro.

The benchmark is already following the same saturation trajectory MMLU did, just a couple of years behind. This pattern — introduce harder benchmark, watch top models close most of the gap to ceiling within 1-2 years — recurs across essentially every benchmark in this file and is treated as its own topic in file 007.

### Weaknesses

MMLU-Pro inherits MMLU's fundamental format limitation: it is still multiple choice, just with more distractors. Its filtering pipeline — which used weaker models' consistency under prompt perturbation as a proxy for "too easy" — is itself a methodological choice that could subtly bias the resulting question set toward whatever kinds of difficulty happen to trip up the specific models used for filtering, rather than difficulty in some model-independent sense.

It is also newer and smaller than MMLU, so it has had less time for independent error-auditing at MMLU-Redux's level of scrutiny. Whether MMLU-Pro has a comparable rate of label errors once it receives that scrutiny is not yet independently confirmed as of this writing.

## GPQA (Graduate-Level Google-Proof Q&A)

**Citation:** Rein, Hou, Stickland, Petty, Pang, Dirani, Michael, Bowman, "GPQA: A Graduate-Level Google-Proof Q&A Benchmark," 2023.

### What it measures and how it validates difficulty

GPQA consists of 448 multiple-choice questions in biology, physics, and chemistry, each written by a PhD-holder or PhD student specializing in the relevant subfield, explicitly targeting graduate-level difficulty.

The defining methodological feature is the validation protocol used to earn the "Google-proof" label. Questions were tested against **skilled non-expert validators** — people with strong general research skills and unrestricted internet access, but without a graduate-level background in the specific subfield of the question. These validators were given roughly 30+ minutes per question and permitted to search the web freely.

- Skilled non-expert validators averaged around **34% accuracy** — close to the 25% random-guess floor for 4-option MC, meaning web search barely helped them.
- Domain PhD experts answering questions in their own subfield averaged in the range of **65-74%** (accuracy is intentionally not 100%, since these are hard graduate-level questions even for the right expert).

The gap between those two numbers is the entire point: it is direct empirical evidence that these questions resist being solved by "look it up," as opposed to a benchmark like MMLU where a sufficiently patient web search can often locate the answer directly.

### What a GPQA-style question looks like, and why it resists lookup

A representative GPQA-style physics item (paraphrased in style, not a real released item) illustrates the difference from an MMLU question:

```
Question: A particle in an infinite square well of width L is prepared in an
equal superposition of the ground state and the second excited state. At
what time t does the expectation value of position first return to its
initial value, in terms of the well's characteristic energy scale?

A) t = h / (E_3 - E_1)
B) t = 2h / (E_3 - E_1)
C) t = h / (E_2 - E_1)
D) t = 4h / (E_3 - E_1)
```

Answering this requires actually knowing the infinite-square-well energy spectrum, constructing the time-dependent superposition, and recognizing which energy gap sets the oscillation period of the expectation value — a web search for "infinite square well expectation value oscillation period" will surface general formulas, but applying them correctly to this specific superposition and distinguishing the four plausible-looking distractor answers requires the underlying physics fluency itself. This is the mechanical difference between a GPQA-style item and an MMLU-style recognition item: the lookup gets you the tool, not the answer.

### GPQA Diamond

The commonly reported subset in frontier-model papers is not the full 448-question set but **GPQA Diamond**, a 198-question subset selected for the highest inter-annotator agreement among the expert validators — i.e., the subset where domain experts most consistently agreed on the correct answer, used as a higher-confidence-label slice of the full set. Most headline numbers cited (e.g., DeepSeek-R1's reported 71.5% on GPQA Diamond) refer to this subset, not the full 448.

### Reported trajectory

(Approximate, self-reported — flagged.) GPT-4-class models at GPQA's introduction scored in roughly the 35-40% range on the full set — barely above the skilled-non-expert baseline and well below domain-expert accuracy, which was exactly consistent with the benchmark doing its job.

By 2024-2025, frontier reasoning-tuned models closed much of that gap: o1-class models and DeepSeek-R1 are reported in the 70-80% range on GPQA Diamond, in some reports exceeding the reported non-expert-validator baseline by a wide margin and approaching or nearing the reported domain-expert accuracy band. This is a much faster saturation trajectory than the "Google-proof" framing might have suggested at launch. It suggests that RL-driven long-chain-of-thought training closes a meaningful fraction of the gap between "recognize plausible science facts" and "work through graduate-level science reasoning," at least as GPQA operationalizes that distinction.

### Weaknesses

- **Small sample size, high variance.** 448 questions (198 for Diamond) is small enough that a handful of flipped answers moves the reported percentage by a point or more. Comparing two frontier models that differ by 1-2 points on GPQA Diamond is close to comparing noise, and confidence intervals are rarely reported alongside the headline number even though they should be.
- **Web contamination is not permanently solved.** "Google-proof at the time of validation" is not the same guarantee as "Google-proof forever" — once the questions and answer keys are published, they become findable via search or present in any web crawl that includes discussion of the paper. The original non-expert baseline measured resistance to *ordinary* web search by a non-specialist at one point in time; it says nothing about resistance to a model pretrained on a crawl that includes the answer key itself.
- **Verified ground-truth errors are a live risk.** Given the graduate-level specialization required to write and check these questions, even expert-reviewed answer keys have had errors identified post-publication in specific subfields. A comprehensive independent GPQA-Redux-style audit at MMLU-Redux's scale and rigor is not something this file can point to as confirmed at time of writing — flagged as an open question.
- **Multiple choice still caps the ceiling.** GPQA still asks the model to pick from 4 options rather than derive and state an answer from scratch, so a model could in principle exploit distractor-elimination heuristics rather than solving the underlying science problem — though the graduate-level distractor design mitigates this more than MMLU's distractors typically do.

## BIG-Bench-Hard (BBH)

**Citation:** Suzgun, Scales, Schärli, Gehrmann, Tay, Chung, Chowdhery, Le, Chi, Zhou, Wei, "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them," 2022. Built from the original BIG-Bench suite (Srivastava et al., 2022), which contained over 200 diverse tasks contributed by hundreds of authors.

### Construction methodology — a curated subset, not a new task set

BBH selects 23 tasks from BIG-Bench's 200+ using an explicit, mechanical selection criterion: tasks where the *prior state-of-the-art model performance* (using the best available models and prompting techniques at curation time) fell below the *average human rater performance* reported for that task in the original BIG-Bench paper.

In other words, BBH is not hand-picked by intuition for "seems hard" — it is the set of tasks where there was a documented, measured human-model gap at the time. The 23 tasks are algorithmic and multi-step-reasoning-heavy by construction, since that is where the original suite's human-model gaps concentrated. Examples include:

| Task (representative subset) | What it actually requires |
|---|---|
| Boolean expression evaluation | Correctly evaluate a nested Boolean formula with `and`/`or`/`not` |
| Causal judgement | Identify the intuitively "responsible" cause among several contributing factors in a short scenario |
| Date understanding | Arithmetic over relative and absolute date references embedded in a short narrative |
| Tracking shuffled objects | Track which entity ends up with which object after a sequence of pairwise swaps |
| Geometric shapes from SVG paths | Infer the shape a raw SVG path-command sequence draws |
| Logical deduction | Determine a strict ordering of entities from a set of relative-position constraints |
| Multi-step arithmetic | Chain several arithmetic operations correctly in the stated order |
| Navigation | Track a 2D position/heading after a sequence of movement instructions |
| Dyck languages | Determine whether a sequence of nested brackets is well-formed and complete it correctly |
| Word sorting | Sort a list of words alphabetically, exposing tokenization/ordering edge cases |

A deliberately synthetic/algorithmic flavor runs through most of the 23 tasks, in contrast to MMLU/GPQA's naturalistic exam-question flavor. Every one of these has a clean, mechanically checkable answer, which is exactly why BBH's exact-match scoring can be fully automated with no equivalence-checking machinery — unlike MATH's answer-equivalence problem in file 002.

### Why chain-of-thought is the paper's actual subject

The paper's core empirical result is that few-shot chain-of-thought prompting (providing worked reasoning traces as in-context examples, then letting the model generate its own reasoning before an answer) closes most or all of the human-model performance gap on 17 of the 23 BBH tasks, whereas standard few-shot prompting without explicit reasoning traces does not.

This was, at the time, some of the clearest task-level evidence that CoT prompting specifically helps on *multi-step, compositional* reasoning tasks rather than being a generically-helpful trick. A task like Dyck-language matching requires tracking state across many tokens; forcing the model to externalize intermediate steps measurably helps in a way that simply asking for a direct answer does not.

### Scoring mechanics

Each task uses its own natural answer format and matching rule — exact-match on a word/label for classification-style tasks, exact-match on a final answer extracted from free-form CoT text via a task-specific parser, or multiple-choice log-likelihood scoring for a few tasks. BBH is a *heterogeneous* benchmark: an aggregation of 23 differently-shaped tasks each with roughly 250 examples. The headline "BBH score" reported in most papers is a simple or task-normalized average across all 23, similar in spirit to MMLU's macro-average across subjects.

```python
# BBH aggregate scoring: unweighted mean across the 23 heterogeneous tasks,
# each already reduced to a 0-1 accuracy via its own task-specific exact-match rule
def bbh_score(per_task_accuracy: dict) -> float:
    assert len(per_task_accuracy) == 23
    return sum(per_task_accuracy.values()) / 23
```

### Weaknesses and saturation status

**Synthetic/algorithmic tasks are gameable in a way naturalistic tasks are not.** Several BBH tasks have a mechanically generatable structure (e.g., Dyck-language balance checking, object-tracking permutations). It is straightforward to synthesize large quantities of near-identical practice data for exactly this task shape, so strong performance can partly reflect targeted exposure — deliberate or incidental, via web text that discusses or reproduces BIG-Bench-style tasks — rather than a general compositional-reasoning capability that would transfer to a novel task of similar shape but different surface form.

**Small per-task sample size (~250 examples per task)** means task-level percentages are noisy. Since the aggregate is an unweighted mean across very different task difficulties and formats, a small number of tasks can swing the aggregate disproportionately relative to their real-world importance.

**Saturation for frontier models.** By 2023-2024, GPT-4-class and later models were reported at or near ceiling (often 90%+) on the large majority of the 23 BBH tasks under CoT prompting — the same lifecycle pattern as MMLU. The benchmark was introduced specifically to expose a gap, CoT prompting (and later, models natively trained toward longer reasoning) closed most of that gap within roughly a year or two, and BBH has correspondingly lost most of its power to discriminate among current frontier models. It remains useful as a diagnostic for smaller or non-reasoning-tuned models, and as a component in aggregate "reasoning suite" leaderboards, but is rarely the benchmark anyone points to as evidence of frontier capability anymore.

**No unified difficulty calibration across tasks.** Because BBH is a curated subset of a much larger, heterogeneously-authored suite, there was never a single consistent difficulty rubric applied across all 23 tasks the way GPQA applied one deliberate "graduate-level, expert-validated" standard. Some BBH tasks are hard because they require many sequential reasoning steps (deep but narrow); others because they require careful attention to a long, information-dense prompt (wide but shallow). The aggregate score does not distinguish between these very different failure modes.

## Why labs still report all four despite the weaknesses above

A reasonable question at this point is why frontier labs continue to report MMLU prominently in model cards and launch announcements given how thoroughly documented its saturation and label-noise problems are. Part of the answer is genuinely inertial — MMLU has been the field's shared reference point for long enough that omitting it from a model card reads as evasive, even to an audience that knows its limitations. Part of the answer is that MMLU (and BBH) still retain real diagnostic value for *non-frontier* models — a 7B open-weight model's MMLU score is far from saturated and still separates meaningfully weak models from meaningfully capable ones, even if the same statement is false for comparing two frontier labs' flagship releases. And part of the answer is that a documented weakness in a benchmark does not mean the benchmark carries zero information — it means the information it carries has to be weighted and contextualized against the specific weaknesses catalogued in this file, which is a different and more defensible position than either "MMLU proves X" or "MMLU is worthless."

## Quick-reference comparison

| Benchmark | Format | Size | Ceiling status (as of writing) | Signature weakness |
|---|---|---|---|---|
| MMLU | 4-choice MC, 57 subjects | ~15,908 Q | Saturated (frontier models high 80s) | Verified label errors, position sensitivity |
| MMLU-Pro | 10-choice MC | 12,032 Q | Approaching saturation | Filtering pipeline bias, less-audited than MMLU |
| GPQA (Diamond) | 4-choice MC | 448 (198 Diamond) | Closing fast (70-80% for reasoning models) | Small n, decaying "Google-proof" guarantee |
| BBH | Heterogeneous, 23 tasks | ~250/task | Largely saturated under CoT | Synthetic/algorithmic tasks are gameable |

## A checklist for reading any reported score from this quartet

Before treating a reported MMLU/MMLU-Pro/GPQA/BBH number as meaningful, it is worth mechanically running through a short checklist — this is precisely the kind of scrutiny a staff-level interviewer expects you to apply unprompted:

1. **Which scoring protocol?** Log-likelihood-over-letters, or generation-plus-parsing? The two are not directly comparable (see the MMLU scoring-mechanics discussion above), and a paper that doesn't specify is a paper you should discount slightly.
2. **Which subset?** For GPQA specifically, is this the full 448-question set or the 198-question Diamond subset? These are routinely conflated in casual reporting.
3. **How many shots, and what prompt template?** Zero-shot, 5-shot, and CoT-prompted numbers are different measurements on the same benchmark, and mixing them across a comparison table is a common, easy-to-miss error.
4. **Is a confidence interval given?** For GPQA Diamond in particular (n=198), a bare point estimate with no interval should be read with real skepticism for close comparisons.
5. **How long has the benchmark been public?** The longer a benchmark has been public and heavily cited, the more weight you should give to the contamination concern before accepting a large reported gain at face value.
6. **Does the gain show up elsewhere?** A jump on one specific benchmark that doesn't show up on structurally similar benchmarks is a signal worth investigating (see file 007's discussion of benchmark-targeting).

## Synthesis: what this quartet actually tells you

Read together, these four benchmarks trace a single, fairly legible research arc. MMLU established that broad exam-style knowledge could be measured cheaply and at scale, and was immediately revealed to be saturable and label-noisy. MMLU-Pro patched the format (more distractors) and the noise (re-annotation) while keeping the same underlying idea. GPQA abandoned "broad coverage of exam questions" in favor of "narrow set of adversarially-validated, genuinely hard-to-look-up graduate questions," trading breadth for depth and validated difficulty. BBH took a different axis entirely — not knowledge breadth or depth but multi-step algorithmic/compositional reasoning, curated by an explicit human-model-gap criterion from a much larger, noisier task pool.

None of the four is a reliable end-to-end proxy for deployed usefulness on its own (see file 007 for that gap in general), and all four share the multiple-choice-or-exact-match format constraint that makes them cheap to grade at the cost of being unable to directly assess open-ended generation quality, calibration, or multi-turn behavior. This is exactly why interview-level fluency in this space means being able to state, for any one reported number, which of these specific documented weaknesses could plausibly be inflating or deflating it.
