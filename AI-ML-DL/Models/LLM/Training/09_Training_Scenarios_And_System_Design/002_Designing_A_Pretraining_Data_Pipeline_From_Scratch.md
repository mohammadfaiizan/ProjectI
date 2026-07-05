# Designing A Pretraining Data Pipeline From Scratch

## The Scenario

"You're joining a new frontier lab as a staff engineer. There is no data pipeline yet — no crawled corpus, no dedup infrastructure, nothing. The mandate is to produce a multi-trillion-token, multi-epoch-ready training corpus for a model in the Llama-3/DeepSeek-V3 class within a defined timeline. Design the pipeline end to end."

This question is deliberately open-ended and system-design-shaped rather than mechanics-shaped: the interviewer is not asking you to re-derive MinHash LSH from scratch or explain exactly how a quality classifier is trained (those mechanics live in `..\01_Datasets\` and this file should point there rather than re-teach them). What's being tested is whether you can sequence a multi-stage pipeline correctly, reason about where the real engineering risk concentrates at multi-trillion-token scale, and make sane build-vs-buy and ordering decisions under a deadline. I'll walk through the pipeline stage by stage, in the order they actually need to run, and treat each stage's *mechanics* as a cross-reference rather than re-deriving it here.

## Step 0: Set the Target and Work Backward

Before touching a single crawl, pin down the target token budget and its consequences, because it determines every downstream engineering decision (storage sizing, dedup infrastructure choice, how much filtering is affordable). If the target model is roughly Llama-3-405B-class trained at ~15T+ tokens, or DeepSeek-V3-class at ~14.8T tokens, the working assumption for this design is **a final training corpus of 15-20T tokens**, sourced from a raw crawl-plus-curated-source pool that is likely 5-10x larger before filtering and deduplication remove the majority of low-quality and redundant content. This ratio — final corpus is a small, heavily-filtered fraction of raw acquired data — is the single most important sizing fact for the whole pipeline: the infrastructure has to be built to process petabyte-scale raw text to produce a terabyte-scale (in token-count terms) final corpus, and every stage's throughput requirement should be sized against the *raw* volume, not the final volume, because that's where the bytes actually flow through the system.

## Step 0b: What Failure Looks Like If This Is Designed Badly

Before walking the stages, it's worth being explicit about the failure modes a bad design produces, since they're what motivate almost every specific recommendation below: a pipeline with no versioning discipline makes any later data-related incident (per `003_Debugging_A_Loss_Spike_Mid_Training.md`) nearly impossible to root-cause; a pipeline that bakes mixture weights into an early stage makes every subsequent mixture revision as expensive as rebuilding the corpus from scratch; a pipeline that treats acquisition as a one-time batch job cannot support the iterative, multi-generation reality of an ongoing research program; and a pipeline that skips contamination screening produces benchmark numbers the evaluation team cannot actually trust, silently undermining the entire launch-gating framework in `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md` without anyone necessarily noticing until much later. Every one of the nine steps that follow is best understood as a direct, deliberate mitigation against one of these specific, foreseeable failure modes — not an arbitrary checklist assembled from general good practice.

## Step 0c: A Quick FAQ on Scope

- **Does this design change much for a smaller, non-frontier-scale training run?** The stages are the same; what changes is the infrastructure investment justified at each one — a 500B-token run can get away with a much lighter-weight distributed-processing setup than a 15T-token run, but skipping contamination screening or versioning is never justified regardless of scale.
- **Should mixture weighting be finalized before or after the first small-scale ablation results come in?** After — Step 5 explicitly argues for treating mixture weights as a late, cheaply-revisable knob precisely so ablation results can inform the final weighting rather than the weighting being locked before any evidence exists.
- **What's the single most common reason a first-time pipeline build runs over schedule?** Underestimating deduplication's engineering timeline, per Step 7c's explicit flag — it is reliably the stage most likely to be treated as simpler than it actually is.

## Step 1: Source Acquisition — What Goes Into the Pool

The source mix at frontier scale is not "just Common Crawl." A realistic 2024-2025-era acquisition plan draws from:

- **Web crawl** (Common Crawl snapshots, or a proprietary crawler if licensing/freshness demands it) — the largest-volume, lowest-average-quality source, requiring the heaviest downstream filtering.
- **Code repositories** (GitHub-scale mirrors, filtered by license) — increasingly weighted heavily in frontier mixtures given the well-established finding that code data measurably improves general reasoning performance, not just coding benchmarks (a finding threaded through nearly every frontier report from GPT-4 onward).
- **Books and long-form text** — high per-token quality and long-range-coherence signal, comparatively scarce and licensing-sensitive.
- **Academic/scientific text** (arXiv, PubMed-style corpora, if licensing allows) — dense in reasoning and technical vocabulary.
- **Curated/licensed data partnerships** — increasingly load-bearing at the frontier as easily-scraped web text becomes both more exhausted and more legally contested; this is a genuine, non-technical constraint that a staff engineer needs to flag to the org rather than treat as someone else's problem, since licensing terms directly gate what can legally enter the pool at all.
- **Synthetic data** — model-generated text (for math, code, and reasoning traces especially, in the style Llama 3's post-training pipeline uses extensively, `..\..\OpenSource\003_Llama3.md` Section 6) is increasingly used to *augment* pretraining-adjacent and mid-training data, not just post-training, particularly in math and code domains where verifiability lets you filter synthetic output for correctness before it enters the corpus.

The mechanics of harvesting each source (crawler design, license/robots.txt handling, format extraction from HTML/PDF/etc.) are covered in `..\01_Datasets\`; the system-design decision that belongs *here* is sequencing and scope: acquisition needs to run continuously and incrementally (new crawl snapshots arriving on a cadence, e.g., monthly Common Crawl dumps) rather than as a single one-time batch job, because the rest of the pipeline (cleaning, dedup, mixture weighting) will need to be re-run as new data becomes available, and building acquisition as a one-shot script rather than a maintained, scheduled ingestion service is a common early mistake that creates enormous rework later.

## Step 1b: A Quick Sizing Exercise for Acquisition Infrastructure

To make Step 1's storage/throughput requirement concrete rather than abstract: targeting a final corpus of ~15T tokens, assuming a raw-to-final filtering ratio of roughly 5-8x (Step 0's working assumption), the raw acquired pool needs to be on the order of 75-120T tokens before filtering. At a rough average of 4-5 characters per token and roughly 1 byte per character for plain text (before considering the HTML/markup overhead of raw crawl data, which can add several more times the eventual plain-text size), that's on the order of 300-600TB of raw plain-text-equivalent content, and likely 1-3PB of actual raw crawl data once HTML/markup overhead is included. This is the number that should drive the object-storage capacity planning conversation with infrastructure/platform teams months before the first byte is ingested — under-provisioning storage capacity for a target this size is exactly the kind of gap that turns into an urgent, disruptive mid-project scramble rather than a calmly planned procurement.

## Step 2: Cleaning and Filtering — The First Big Volume Cut

Raw crawled HTML/text needs boilerplate stripping, language identification, and quality filtering before it's even a candidate for deduplication (running expensive dedup against unfiltered garbage wastes compute on data that was never going to survive anyway — ordering matters). The pipeline stage here:

1. Extract main content from HTML (strip navigation, ads, boilerplate).
2. Language identification and filtering to the target language mix (a multilingual frontier model needs an explicit, deliberate per-language quality bar, not an accidental one that falls out of whatever the crawl happened to contain).
3. Heuristic quality filters — document length thresholds, symbol-to-word ratio, repeated-line/paragraph ratio, and similar cheap statistical filters that remove obviously degenerate documents (link farms, auto-generated spam, boilerplate-heavy pages) at a fraction of the cost of a learned classifier.
4. A **learned quality classifier** (Llama 3's pipeline explicitly uses earlier Llama models themselves as quality classifiers over web data, `..\..\OpenSource\003_Llama3.md` Section 5) that scores documents against a notion of "quality" typically anchored to a reference distribution (e.g., text resembling curated/high-signal sources) and filters or reweights accordingly.

The exact classifier training methodology, feature choices, and threshold-tuning tradeoffs are `..\01_Datasets\` territory. The system-design point that belongs here is **ordering cheap-to-expensive**: heuristic filters (near-free, pure string/statistics operations) should run first and remove the bulk of obviously-bad documents before the comparatively expensive learned-classifier pass runs on what remains, and the learned-classifier pass should itself run before the even-more-expensive deduplication stage (Step 3), because dedup's fuzzy-matching cost scales with corpus size and there's no reason to fuzzy-deduplicate documents that a cheap heuristic would have discarded anyway.

## Step 2b: The Bootstrap Problem for the Learned Quality Classifier

Step 2's learned quality classifier has a chicken-and-egg problem worth naming explicitly, because it trips up first-time pipeline builds: the classifier needs *some* notion of "high quality" to train against, but that notion doesn't yet exist in a labeled form before the pipeline has produced anything. The standard resolution, and the one Llama 3's own pipeline uses (`..\..\OpenSource\003_Llama3.md`, Section 5): use an *existing*, already-trained model (an earlier generation of your own model line, or a strong open model) as the initial quality-scoring signal, generating a first-pass labeled dataset by scoring a sample of documents and treating a reference distribution (curated encyclopedic/book-quality text, or text resembling known high-signal sources) as the positive class. Train the first classifier generation against this bootstrapped signal, use it to filter the first real corpus snapshot, and — once a model has actually been trained on that snapshot and evaluated — use the *resulting* model's own judgment (or human review of its outputs) to refine the classifier for the next generation. This bootstrap-and-iterate loop is normal and expected; a team that treats the first classifier generation as if it needs to be perfect before any data can be processed will stall the entire pipeline waiting for a labeled dataset that has no other source than the pipeline itself producing something to evaluate first.

## Step 3: Deduplication — Exact and Fuzzy, at Trillion-Token Scale

Deduplication is the stage most likely to be underestimated in a first-pass design, because "just hash the documents" sounds trivial and isn't, at this scale, for two reasons: (a) near-duplicates (boilerplate-wrapped copies of the same underlying content, e.g., syndicated news articles, mirrored documentation) vastly outnumber exact duplicates and require fuzzy matching (MinHash/LSH-style techniques) rather than exact hashing, and (b) fuzzy matching at trillion-document-fragment scale is itself a distributed systems problem — an all-pairs comparison is infeasible, so the actual mechanism (locality-sensitive hashing bucketing candidate near-duplicates into a tractable number of comparison groups) needs to run as a distributed job across the whole raw corpus. The mechanics of MinHash/LSH itself belong in `..\01_Datasets\`; what belongs here is the sequencing and infrastructure decision: dedup needs to run **after** cheap filtering (Step 2) has already cut volume, and it needs to run at multiple granularities — document-level (removing near-identical whole documents) and, increasingly at frontier scale, substring/n-gram-level dedup against the eval benchmark suite specifically, which is really Step 5 (contamination screening) but shares the same underlying fuzzy-matching infrastructure and is worth building once and reusing for both purposes rather than as two separate systems.

## Step 3b: Choosing Dedup Granularity — Document, Paragraph, or Substring

A design decision worth making explicit rather than defaulting to a single granularity: near-duplicate detection can run at the whole-document level (catching mirrored/syndicated full documents), the paragraph or chunk level (catching partial overlap — a document that quotes or reuses a large block from another source without being a full duplicate), or the substring/n-gram level (catching short verbatim spans, which is closer to what contamination screening against eval-item-length text actually needs, per Step 6). These are not interchangeable, and a pipeline that only implements document-level dedup will systematically miss a real and consequential category of redundancy: large partial-overlap chunks distributed across many nominally-distinct documents, which can still meaningfully inflate the effective repetition rate of specific content the document-level check never flags because no single document is a near-duplicate of any other *in its entirety*. A mature pipeline runs dedup at more than one granularity, reusing the same underlying LSH infrastructure with different shingle-construction parameters per granularity, rather than assuming one pass at one granularity is sufficient coverage.

## Step 4: Quality Classification and Domain/Topic Tagging (Beyond Step 2's Filter)

Once the corpus has been cleaned and deduplicated, a second pass of classification is typically applied — not to filter further, but to **tag** documents by domain, topic, and estimated quality tier, because this tagging is the input to Step 5's mixture-weighting decision. This is where "code," "math," "encyclopedic," "conversational web," "low-quality web," and similar buckets get assigned, often via a combination of source-level heuristics (a document from arXiv is presumptively "academic/math" without needing a classifier to tell you) and learned classifiers for less clearly source-delineated buckets (distinguishing genuinely high-quality general web text from mediocre web text within Common Crawl, where source alone doesn't tell you).

## Step 5: Mixture Weighting — The Actual Modeling Decision Hiding Inside a Data Pipeline

This is the step where "data engineering" becomes "a modeling decision with a data-engineering implementation," and it's worth naming explicitly as such in an interview, because it's easy to talk about mixture weighting as if it were purely mechanical. The decision — how much of the final training budget should come from web text versus code versus books versus math versus multilingual data — is not determined by how much raw data of each type happens to be available; it is a deliberate up/down-weighting choice, exactly as GPT-3's training mixture upsampled small high-quality sources (Wikipedia, Books) and downsampled the much larger but noisier Common Crawl (`..\..\GPT\003_GPT3.md`, Section 5), and as Llama 3's "data annealing" up-weights a small, high-quality mixture specifically in the final phase of pretraining (`..\..\OpenSource\003_Llama3.md`, Section 5).

Practically, this means the pipeline needs to expose mixture weights as an explicit, tunable configuration — a per-source or per-domain sampling rate — separate from the underlying corpus, so that researchers can run small-scale ablations (train small proxy models on candidate mixtures, compare downstream eval performance) before committing the full-scale run to a fixed mixture. Building the pipeline so that mixture weighting is a late-stage, cheaply-adjustable knob (applied at data-loading/sampling time, not baked irreversibly into an earlier processing stage) is a genuine infrastructure decision with real payoff: it lets the mixture be revised (e.g., mid-run annealing toward a smaller high-quality subset) without re-running the expensive cleaning/dedup stages that produced the underlying pool.

## Step 6: Contamination Screening Against Eval Sets

Before any training run begins, the corpus needs to be screened against the evaluation suite the model will eventually be judged on — MMLU, GSM8K/MATH, HumanEval, and whatever benchmark suite the launch-gating framework (see `005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md` in this folder, and `..\06_Benchmarks\`) will use. This has to happen *before* training, not as a post-hoc audit, because the fix for discovered contamination (removing contaminated documents, or documents near-duplicate to eval items) is a data-pipeline operation, and finding it only after a multi-million-dollar training run has already ingested the contaminated data means the finding arrives too late to act on cheaply.

Mechanically this reuses the fuzzy-matching infrastructure from Step 3 (n-gram overlap / near-duplicate detection), applied specifically between the candidate training corpus and the eval benchmark question/answer text, at whatever granularity is appropriate (exact string match catches verbatim leakage; n-gram overlap thresholds catch paraphrased or lightly-modified leakage, which is the more common and more consequential failure mode at scale, since a determined-but-lazy contamination path — an eval set mirrored onto a web page that then gets crawled — produces near-duplicate rather than byte-identical text). The output of this stage should be a documented contamination report per benchmark, not just a binary pass/fail, because a benchmark with 0.3% detected overlap is a very different finding from one with 15% overlap, and the eval team downstream needs that granularity to decide whether specific benchmark numbers should be discounted or the benchmark re-run against a decontaminated eval subset.

## Step 7: The Engineering Infrastructure — What Actually Has to Be Built

Pulling the above stages together, the infrastructure requirements at multi-trillion-token scale:

**Distributed processing framework.** Every stage above (cleaning, filtering, dedup, classification, mixture sampling) needs to run as a distributed batch/streaming job across a cluster large enough to process petabytes of raw text in a reasonable wall-clock window — this is a Spark/Ray/Dask-class distributed-compute problem, not a single-machine scripting problem, and the choice of framework matters less than the discipline of designing every stage to be embarrassingly parallel over document shards from the start, since retrofitting parallelism onto a pipeline built as sequential single-node scripts is a common and expensive mistake. A useful rule of thumb: if a stage cannot be expressed as a map (possibly followed by a shuffle/groupby for the dedup stage specifically) over independent document shards, it will not scale to trillion-token corpora regardless of how fast a single-node implementation is.

**Storage.** Raw crawl data, intermediate cleaned/filtered checkpoints (kept for reproducibility and re-processing without re-crawling), and the final tokenized training corpus all need durable, high-throughput storage — object storage (S3-class) for the bulk raw/intermediate data, with a data-loading layer (sharded, pre-tokenized, ideally already packed into fixed-length sequences with minimal padding) that can feed a training cluster's GPUs fast enough that the data pipeline is never the throughput bottleneck. This last point is explicitly called out as a first-class systems risk in Llama 3's own infrastructure discussion (`..\..\OpenSource\003_Llama3.md`, Section 8): feeding a cluster of up to 16,000 H100s a continuous, well-shuffled, deduplicated stream at the rate those GPUs can consume tokens is a genuine I/O-engineering problem, not an afterthought, and under-provisioning storage/network bandwidth for the data pipeline is one of the more common ways a well-designed model/training-infra plan still ends up GPU-idle-time-bound in practice (see also `004_Diagnosing_Slow_Or_Stalled_Distributed_Training.md` in this folder for the failure-mode side of this).

**Versioning.** The final training corpus, and ideally every major intermediate stage's output, needs to be versioned as a first-class artifact — not just "the data as of whenever training started," but a specific, reproducible snapshot with a manifest (which crawl snapshots, which filter/classifier model versions, which mixture weights, which contamination-screening pass) that can be referenced later. This matters for two concrete reasons beyond general good hygiene: reproducibility of any downstream training-quality investigation (if a loss anomaly is later traced to a specific batch or shard, per `003_Debugging_A_Loss_Spike_Mid_Training.md`, you need to be able to identify exactly what data that shard contained and how it was produced), and defensibility (being able to state precisely what filtering/classification/decontamination process a given training corpus went through, for both internal quality review and any external scrutiny of training data provenance).

## Step 7b: A Concrete Versioning Manifest — What the Artifact Actually Looks Like

"Version the corpus" is easy to say and easy to leave underspecified. A concrete manifest schema, attached to every training-ready corpus snapshot, makes the requirement checkable rather than aspirational:

```json
{
  "snapshot_id": "corpus-v14-2025-03",
  "created_at": "2025-03-02T00:00:00Z",
  "source_composition": {
    "web_crawl": {"snapshot_ids": ["CC-2025-01", "CC-2025-05"], "raw_tokens": 4.2e13, "post_filter_tokens": 6.8e12},
    "code": {"source": "internal-code-mirror-v3", "raw_tokens": 2.1e12, "post_filter_tokens": 1.4e12},
    "books": {"source": "licensed-books-pool-v2", "raw_tokens": 9.0e10, "post_filter_tokens": 8.1e10},
    "synthetic_math": {"source": "verified-synth-math-v5", "raw_tokens": 3.0e11, "post_filter_tokens": 3.0e11}
  },
  "pipeline_versions": {
    "quality_classifier": "qc-model-v7",
    "dedup_lsh_params": {"num_perm": 128, "bands": 16, "rows": 8, "jaccard_threshold": 0.8},
    "contamination_screen": "eval-suite-v2025-02, threshold 0.75 jaccard"
  },
  "mixture_weights": {"web_crawl": 0.55, "code": 0.20, "books": 0.05, "synthetic_math": 0.05, "other": 0.15},
  "final_token_count": 1.48e13,
  "contamination_report_ref": "contamination-report-corpus-v14.json"
}
```

The specific value of a schema like this over an informal "we know what's in the corpus" understanding: it makes the corpus a *queryable* artifact months later, when (per `003_Debugging_A_Loss_Spike_Mid_Training.md`) an investigation needs to reconstruct exactly what a specific batch contained, or when a downstream evaluation needs to know precisely which contamination-screening pass and threshold a given training run's corpus was checked against before its benchmark numbers can be trusted.

## Step 7c: Cost and Timeline by Stage — What a Project Plan Actually Needs

A staff engineer proposing this pipeline needs to attach rough cost/timeline figures per stage, not just a technical description, because this is what lets the rest of the organization plan around it:

| Stage | Primary cost driver | Rough lead time (new pipeline, first build) |
|---|---|---|
| Acquisition infrastructure | Storage + crawler/ingestion engineering | 4-8 weeks to a stable, scheduled pipeline |
| Cleaning / heuristic filtering | CPU compute at raw-corpus volume | 2-4 weeks engineering, then continuous |
| Learned quality classification | Classifier training + CPU/GPU inference at volume | 3-6 weeks (including bootstrap labeling) |
| Deduplication (LSH-based) | Distributed compute, the single largest compute line item | 4-8 weeks to tune parameters and validate at scale |
| Mixture weighting / ablations | Small-scale proxy-model training compute | Ongoing, should start as soon as tagged data exists |
| Contamination screening | Reuses dedup infrastructure against eval suite | 1-2 weeks once dedup infra exists |
| Tokenization + packing | CPU compute, storage I/O | 1-2 weeks |

The deduplication row deserves the flag it's given here: it is reliably the single most underestimated line item in a first-pass timeline, precisely because "hash the documents" sounds trivial and the actual distributed LSH tuning-and-validation cycle (getting band/row parameters right, validating recall against a labeled sample per the discussion in `011_Interview_Questions_Part1.md`, Q7) is not.

## Step 7d: Multilingual Data — A Cross-Cutting Concern, Not a Separate Stage

A frontier-scale corpus is rarely English-only, and multilinguality isn't a ninth pipeline stage bolted on at the end — it's a dimension that cuts across every stage above and needs its own explicit decisions at each one:

- **Acquisition (Step 1):** raw web-crawl volume by language is wildly uneven — English, Chinese, and a handful of other high-resource languages dominate any unweighted crawl, and a deliberate acquisition strategy needs to identify which additional languages the target model must support well and actively seek out (or license) additional sources for those specifically, rather than assuming a generic web crawl will naturally produce adequate volume for anything outside the highest-resource languages.
- **Cleaning/filtering (Step 2):** language identification has to run before most other filtering, since heuristic quality filters and learned classifiers tuned primarily on high-resource-language data can systematically misjudge quality for lower-resource languages with different typical document structures — a classifier trained mostly on English web text is a real risk of silently degrading a multilingual corpus's non-English quality bar.
- **Deduplication (Step 3):** near-duplicate detection via shingling is language-sensitive (shingle/n-gram construction assumes a tokenization scheme that behaves reasonably for the language in question — word-boundary-based shingling that works for space-delimited languages needs adjustment for languages without clear word boundaries in their written form).
- **Mixture weighting (Step 5):** per-language sampling rates are a genuinely separate knob from per-domain sampling rates, and the two interact — a mixture that gets domain balance right in aggregate can still badly under-serve a specific language if that language happens to be concentrated in a domain that's being down-weighted for unrelated reasons.
- **Tokenizer training (Step 8):** vocabulary allocation across languages is exactly the concern raised in `011_Interview_Questions_Part1.md`, Q19 — a vocabulary trained without deliberate per-language balance under-serves lower-resource languages with more tokens-per-character, which compounds every downstream cost (training tokens, inference decode steps) for exactly the languages the org is trying to support well.

## Step 7e: Licensing and Provenance Risk — A Non-Technical Gate That Blocks Technical Work

It's worth stating directly, as a staff engineer's responsibility rather than someone else's problem: a pipeline that acquires and processes data faster than legal/policy review can clear it for use creates a growing backlog of unusable (or risky-to-use) processed data, and a mature pipeline design treats licensing clearance as an explicit **gate between acquisition and every downstream stage**, not an afterthought applied only if someone happens to ask. Concretely, this means the acquisition manifest (Step 7b's schema) needs a `licensing_status` field per source, populated before that source is eligible to flow into cleaning/filtering, and a real project timeline (Step 7c) needs to budget calendar time for legal/policy review as a dependency for any new source category, especially licensed/partnership data where terms are actively negotiated rather than a matter of interpreting existing public licensing. Retrofitting this gate after a corpus has already been built without it — exactly the scenario in `009_Post_Launch_Model_Degradation_And_Incident_Response.md`'s sibling file, `011_Interview_Questions_Part2.md` Q1, where a licensing issue surfaces post-launch — is dramatically more expensive than building the gate in from the start, because the remediation options at that point (retraining, unlearning, or accepting the risk) are all worse than simply not having ingested the problematic data in the first place.

## Step 9: What Commonly Goes Wrong When Teams Build This for the First Time

A few recurring failure patterns worth naming directly, since a staff-level design should anticipate and design around them rather than discovering them the hard way mid-project:

- **Treating acquisition as a one-time batch job.** Per Step 1, this is a maintained, scheduled ingestion service, not a script run once — teams that build it as the latter end up needing a costly re-architecture the first time a mixture-weight revision or a newly available source requires pulling in fresh data on a cadence.
- **Running deduplication before cheap heuristic filtering.** This inverts the cost-ordering principle from Step 2/3 and wastes expensive fuzzy-matching compute on documents a near-free heuristic filter would have removed anyway — a mistake that's easy to make when dedup is built by a different team/workstream than filtering and the two aren't sequenced deliberately against each other.
- **Baking mixture weights into an early, hard-to-revise processing stage** rather than exposing them as a late-stage, cheaply-adjustable sampling configuration (Step 5's explicit recommendation) — teams that get this wrong end up needing to re-run expensive upstream stages every time a mixture ablation suggests a revision, when the fix (making mixture weighting a data-loading-time knob) is comparatively cheap to build correctly from the start.
- **Skipping or under-resourcing contamination screening under timeline pressure** (Step 6) — this is the single highest-regret shortcut in the entire pipeline, because it's cheap to run early and disproportionately expensive to discover missing only after a full-scale training run has completed and its benchmark numbers turn out to be unusable.
- **No versioning discipline until an incident forces it.** Teams frequently build the acquisition-through-tokenization pipeline without a manifest schema (Step 7b) until the first time an investigation needs to reconstruct exactly what a specific training batch contained and discovers that capability doesn't exist — at which point it has to be retrofitted under incident pressure rather than designed calmly in advance.

## Step 7f: Ownership — Who Actually Runs Each Stage

A design that names stages without naming owners tends to leave gaps exactly at the handoff boundaries between stages, so it's worth being explicit about a plausible ownership split for a team standing this up from scratch:

- **Acquisition infrastructure**: a dedicated data-infrastructure/platform team, since this is fundamentally a distributed-systems and storage-engineering problem, not a research problem.
- **Cleaning, filtering, quality classification**: a data-quality/research-adjacent team that owns the classifier training loop and the heuristic-filter rule set, working closely with whoever owns evaluation (since quality-classifier validation depends on downstream capability signal).
- **Deduplication infrastructure**: shared between data-infrastructure (the distributed LSH job itself) and data-quality (parameter tuning and recall validation) — this stage's ownership split is worth calling out explicitly because it's the stage most likely to fall into a gap between teams if ownership isn't assigned deliberately.
- **Mixture weighting and ablations**: the pretraining research team, since this is fundamentally a modeling decision (Step 5) that happens to have a data-engineering implementation, and should not be delegated entirely to a data-engineering team that lacks the context to run the small-scale proxy-model ablations the decision depends on.
- **Contamination screening**: jointly owned by data-quality and the evaluation team, since the evaluation team is the ultimate consumer of the resulting guarantee and needs visibility into exactly what was screened and at what threshold.
- **Versioning and manifest infrastructure**: data-infrastructure, but with a schema (Step 7b) co-designed with every downstream consumer (pretraining research, evaluation, and any future incident-response investigator) so the manifest actually captures what those consumers will eventually need from it.

## Step 7g: A Review Checklist Before Declaring a Corpus Snapshot "Training-Ready"

A concrete gate, usable as an actual sign-off checklist rather than an abstract description of good practice:

- [ ] Every source in the snapshot has a `licensing_status` of "cleared" in the manifest (Step 7e) — no source pending legal/policy review is included.
- [ ] Heuristic filtering and learned quality classification have both run against the full raw pool, with rejection rates by source logged and reviewed for anomalies (an unexpectedly high or low rejection rate for a specific source is often the first sign of a pipeline bug).
- [ ] Deduplication has run at the validated LSH parameters, with a recall/precision estimate against a labeled sample attached to the manifest (per the discussion in `011_Interview_Questions_Part1.md`, Q7).
- [ ] Mixture weights are finalized and attached to the manifest, with the small-scale ablation results that justified them referenced or linked.
- [ ] Contamination screening has run against the current version of every benchmark in the evaluation suite's launch-gating tier (`005_Designing_An_Evaluation_Framework_For_A_Model_Launch.md`), with a per-benchmark overlap-rate report, not just a binary pass/fail.
- [ ] Multilingual coverage (Step 7d) has been reviewed per-language, not just in aggregate, against the target model's intended language support list.
- [ ] The manifest (Step 7b's schema) is complete and has been written to durable, versioned storage before the snapshot is handed to the training team.
- [ ] Tokenization and packing have been validated against a small sample (checking for edge cases — pathologically long documents, malformed encoding — per the data-issue branch of `003_Debugging_A_Loss_Spike_Mid_Training.md`) before committing the full snapshot to the expensive full-scale training run.

Treating this as a literal, signed-off checklist — not just a mental model of "the pipeline is mature enough" — is what makes the training-readiness decision auditable after the fact, exactly the same auditability principle argued for throughout this module's evaluation and incident-response files.

## Step 8: Sequencing the Whole Thing — A Realistic Timeline View

Putting the stages in dependency order (not calendar order — several run concurrently once the pipeline is mature, but this is the order in which a *new* pipeline needs to come online):

1. Acquisition infrastructure (crawler/ingestion + storage) — must exist before anything else can start.
2. Cleaning/heuristic filtering — can begin as soon as raw data starts arriving, and should run continuously against the incoming stream rather than in one giant batch.
3. Learned quality classification — requires a first pass of heuristically-filtered data to train the classifier against (bootstrap problem: early classifier training data is itself hand-curated or heuristically-selected, later iterations can use the growing curated pool).
4. Deduplication — runs against the filtered pool; this is usually the single most compute- and engineering-intensive stage and the one most worth prototyping early on a small sample to validate the LSH parameters before running at full scale.
5. Domain/topic tagging and mixture-weight ablation — small-scale proxy-model experiments here should start well before the full corpus is finalized, because mixture weighting decisions need lead time to validate, not a rushed last-minute call right before the training run starts.
6. Contamination screening — the last gate before a corpus snapshot is declared "training-ready," reusing Step 4's fuzzy-matching infrastructure against the eval suite specifically.
7. Tokenization and packing into the final training-ready sharded format, with the versioned manifest from Step 7 attached.

A staff-level answer should be explicit that steps 2-6 are not strictly sequential in a mature pipeline — they run as a continuously-refreshed pipeline against an ever-growing raw pool, with periodic "cut a new training corpus snapshot" events that pull a consistent view through all six stages — but a *new* pipeline being built from scratch under a deadline should sequence its initial engineering investment exactly in the dependency order above, because building deduplication infrastructure before acquisition and cleaning exist to feed it is wasted early effort, and skipping contamination screening to save time is the single highest-regret shortcut on this list, since it is cheap to run early and catastrophically expensive to discover missing after a full training run has already completed.

## Step 10: A Worked Timeline for a Team Standing This Up From Zero

To make Step 8's dependency ordering concrete, here is a plausible calendar-time view for a team building this pipeline from scratch, assuming reasonable staffing (a data-infrastructure team of 4-6 engineers, a data-quality/research team of 3-4):

- **Weeks 1-6:** acquisition infrastructure comes online (crawler/ingestion pipeline, object storage, initial licensed-source partnerships negotiated in parallel by legal/policy). Cheap heuristic filters are prototyped against early incoming data during this window, even before the full acquisition pipeline is at target throughput.
- **Weeks 4-10 (overlapping):** learned quality classifier development begins as soon as a bootstrap pool of heuristically-filtered data exists; deduplication infrastructure prototyping begins in parallel on a small sample, specifically to validate LSH parameters (Step 7c flagged this as the highest-risk-of-underestimation stage, and starting it early is the direct mitigation).
- **Weeks 8-14:** full-scale deduplication runs against the growing filtered pool; domain/topic tagging and the first round of small-scale mixture-weight ablations begin, using whatever tagged data is available even before the corpus is fully finalized.
- **Weeks 12-16:** contamination-screening infrastructure is built (reusing the now-validated dedup infrastructure) and run against a first candidate corpus snapshot; findings feed back into Step 2/3 if meaningful contamination is discovered, which is exactly why this shouldn't be scheduled as the very last step with no time buffer to react to findings.
- **Weeks 14-18:** tokenizer finalized and trained against the near-final mixture; first full training-ready snapshot cut, manifest (Step 7b) attached, review checklist (Step 7g) signed off.

This is a 3.5-4 month critical path for a *first* training-ready snapshot from a completely cold start — a number worth having ready in an interview specifically because it's the kind of concrete, defensible estimate (as opposed to a vague "it takes a while") that distinguishes someone who has actually thought through the dependency structure from someone reciting stage names in a list. Subsequent snapshots, once the pipeline is mature and running continuously, are dramatically cheaper to produce — the 3.5-4 month figure is a one-time buildout cost, not a recurring cadence.

## Closing Note: The Pipeline as a Living System, Not a One-Time Deliverable

The single framing worth carrying out of this exercise: a pretraining data pipeline for a frontier lab is not a project with a completion date so much as a standing piece of infrastructure that gets re-run, re-tuned, and re-validated for every subsequent training generation — new crawl snapshots keep arriving, new licensed sources get negotiated, quality classifiers get retrained as the field's notion of "quality" evolves, and contamination screening has to be re-run against every new benchmark the evaluation team adds to the launch-gating suite. Designing the pipeline's *stages* correctly (Steps 1-6) is necessary but not sufficient; designing it so that mixture weights are a late, cheap knob (Step 5), so that every snapshot is versioned and reconstructable (Step 7b), and so that the whole thing runs as a continuously-refreshed service rather than a one-shot batch job (Step 8's closing point) is what actually determines whether this investment pays off once, or pays off every single time the organization trains its next model.
