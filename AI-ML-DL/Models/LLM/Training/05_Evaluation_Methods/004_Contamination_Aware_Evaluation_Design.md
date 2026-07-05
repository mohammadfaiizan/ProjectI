# Contamination-Aware Evaluation Design

## 0. Scope: methodology, not detection

The Datasets module's `..\01_Datasets\005_Contamination_Detection_And_Decontamination.md` covers how
you detect and remove benchmark contamination *from a training corpus* — n-gram overlap search,
embedding-similarity dedup against known eval sets, and the general problem of a pretraining crawl
accidentally ingesting a benchmark's test split. This module covers the complementary,
forward-looking discipline: designing and operating an **evaluation program** so that it remains
informative even granting that contamination is a persistent, probably-never-fully-solved background
risk. The mindset shift is from "detect and clean it out of training data" (a data-pipeline problem)
to "assume some contamination will always slip through, and structure the evaluation system so that
it still tells you the truth about model capability" (an eval-program-design problem). These are
complementary defenses, not substitutes for each other — a lab that only decontaminates training
data but publishes every eval set in full, forever, is still exposed; a lab that keeps private eval
sets but does no training-data decontamination at all is paying an unnecessary training-time cost
for benchmarks it could have cleaned.

## 1. Why contamination is structurally hard to fully solve

Public benchmarks are, definitionally, public: MMLU, GSM8K, HumanEval, and every other widely used
benchmark exist as downloadable files with known formats, and once a benchmark is popular enough to
be a meaningful signal, it is also popular enough to get mirrored, quoted in blog posts, discussed
in forums, included in other papers' appendices, and scraped into future web crawls — including,
eventually, the pretraining corpora of models trained after the benchmark's release. This is not a
hypothetical: multiple published analyses (e.g., contamination audits accompanying various model
releases and independent third-party investigations) have found measurable overlap between popular
benchmark test sets and large web-scale pretraining corpora, and the more a model improves at a
benchmark for reasons that don't generalize (memorized answer strings, memorized answer *letter* for
multiple choice, memorized surface structure of a proof) rather than reasons that do (actual
capability), the less that benchmark's score means.

Two contamination mechanisms are worth distinguishing because they call for different defenses:

- **Verbatim leakage**: the exact eval question and/or answer text appears in the training corpus (a
  benchmark's GitHub repo, a paper's appendix, a blog post quoting test items, a forum thread
  discussing answers). This is what n-gram-overlap decontamination (Datasets module) is built to
  catch, and it is catchable with reasonably high recall if the exact strings are known.
- **Indirect/near-duplicate leakage**: paraphrased versions, translated versions, or
  same-underlying-fact-pattern variants of eval items appear in training data without matching
  verbatim (e.g., a math competition problem's *solution methodology* discussed in a forum thread,
  even if the exact numbers differ from the benchmark's specific instance). This is much harder to
  detect mechanically and is the deeper reason no decontamination pipeline should be assumed
  complete — it produces a softer form of the same problem, where the model has effectively "seen
  the trick" even without having seen the literal test item.

Given that even a well-executed decontamination pipeline cannot certify zero contamination —
especially against the second, harder-to-detect mechanism — the practical response for anyone
running an evaluation program is to design the *evaluation itself* to be robust to a nonzero,
unknown contamination rate, rather than to treat decontamination as a solved precondition.

## 2. Private, held-out evaluation sets

### 2.1 The core idea

The most direct defense against contamination is simple in principle: maintain an evaluation set
that is **never published**, not in any form — not the raw items, not example items in a paper, not
a description detailed enough to reconstruct meaningfully, and ideally not even shared broadly
inside the organization beyond the team that runs the eval. If the eval set literally does not exist
anywhere a web crawler (or a human who might post about it) could find it, verbatim leakage is
structurally impossible, and even indirect leakage is far less likely because no one outside the
eval team has seen the specific items to discuss or paraphrase.

### 2.2 What this actually requires operationally

- **Original item authoring, not aggregation of existing public material.** A private eval set built
  by scraping or lightly modifying existing public benchmarks doesn't buy much — the underlying
  problems and their answer patterns may already be represented in training data even if the
  specific compiled file is new. Genuine protection requires writing new items (or, for tasks like
  code/math where problems can be programmatically generated, generating fresh instances from a
  template/generator that itself is not published in a way that reveals the exact instance
  distribution).
- **Strict access control.** Treat the eval set the way you'd treat any sensitive credential: access
  logged, restricted to the specific team/pipeline that needs it, no copies in shared docs,
  notebooks, or Slack messages that could leak outside the intended boundary, and no committing it
  to a public or even loosely-permissioned internal-but-broadly-readable code repository.
- **No item-level examples in any external communication.** This is the discipline that's hardest to
  maintain in practice: researchers naturally want to show a benchmark's format or an interesting
  failure case in a paper, blog post, or conference talk. Even "just one illustrative example" from
  a supposedly private set defeats the purpose if that example (or enough of its structure) ends up
  quoted, discussed, and eventually crawled — a private eval set's value degrades incrementally and
  irreversibly with every partial disclosure, which is why disciplined programs treat disclosure of
  even single items as a real leak, not a harmless illustration.
- **No third-party benchmark-service dependency without a data-handling agreement.** If a private
  eval is run through any external tooling, API, or contracted human-annotation vendor, the
  contractual and technical guarantees that the vendor isn't logging/retaining/reusing the eval
  content need to be explicit, because a "private" eval set that flows through a third party without
  such guarantees isn't actually private.

### 2.3 What a private eval set costs you

- **No external comparability.** A private eval set can't be used to make a publicly verifiable
  claim ("our model scores X on this benchmark") because outside parties can't inspect or reproduce
  the measurement — this is a direct tension with the scientific-transparency and
  competitive-benchmarking functions public benchmarks serve, discussed further in
  `..\06_Benchmarks`. Labs typically solve this by maintaining *both*: public benchmarks for
  external claims (accepting the contamination risk as a known limitation of that number) and
  private evals for internal ground-truth capability tracking that doesn't depend on external
  validation.
- **No community-scale item diversity and review.** Large public benchmarks benefit from broad
  community scrutiny (errors get reported and fixed, ambiguous items get flagged) that a small,
  closely-held private set does not get for free — a private set's quality is only as good as the
  internal team that built and maintains it, which is a real ongoing cost.
- **Confidence that "private" really means private is itself imperfect.** Internal leaks (an
  employee discussing eval items externally, a misconfigured access control, a vendor breach) are a
  real residual risk for any private set, especially one used broadly enough inside a large
  organization that "private" starts to mean "known to hundreds of employees" rather than "known to
  a five-person team."

## 3. Rotating and refreshing evaluation sets over time

### 3.1 The idea

Rather than (or in addition to) a single static private set, periodically retire old eval items and
replace them with newly authored ones, on a cadence set by how much the field, the model pool, and
the risk of leakage have moved since the current set was created. This directly bounds the *maximum
age* — and therefore the maximum cumulative leak exposure — of any given item still in active use,
and it also keeps the eval set's difficulty and content coverage current as models improve and
previously-hard items saturate (a benchmark where every current model scores 95%+ has stopped
discriminating between good and great models regardless of contamination, which is a separate but
related reason to refresh — see `..\06_Benchmarks` on benchmark saturation).

### 3.2 Practical rotation design

- **Version the eval set explicitly**, so every reported score is attributable to a specific version
  (e.g., "InternalReasoningEval-v7") and a historical score on an older version is never silently
  compared to a new version's score as if they were the same measurement — this is the same
  discipline needed for any longitudinal metric that changes definition over time.
- **Retire, don't just add.** If old items are never removed, the set accumulates more and more
  items with more and more cumulative time-in-the-wild exposure to leakage, especially for any item
  that was inadvertently exposed at some point (an internal demo, a slide, a partial disclosure).
  Active rotation — replacing a meaningful fraction of the item pool each cycle rather than only
  appending — keeps the average item age, and therefore the average leak exposure, bounded.
- **Overlap enough between versions to measure drift.** Retire items gradually and keep some overlap
  between consecutive eval-set versions specifically so you can measure whether the new version's
  difficulty is calibrated similarly to the old one (a naive full replacement risks silently
  changing what "70% on this eval" means release over release, confounding a real capability change
  with a difficulty-of-the-new-item-set change).
- **Tie rotation cadence to actual leak-risk events, not just a calendar.** A fixed quarterly or
  annual rotation is a reasonable default, but any known or suspected partial disclosure of the
  current set (an internal leak, a vendor incident, an item surfacing somewhere unexpected) should
  trigger an out-of-cycle rotation of at least the affected items, rather than waiting for the next
  scheduled refresh.

### 3.3 Cost of rotation

Authoring new, high-quality eval items (especially ones requiring domain expertise, careful
ground-truth verification, or difficulty calibration against the current model frontier) is a real,
recurring cost — it is not a one-time investment the way a static benchmark is. This is one of the
reasons programmatic item generation (templated math/code/logic problem generators that can produce
a fresh, non-overlapping instance distribution on demand) is an attractive complement to
hand-authored rotation for domains where it's feasible: the generator itself needs to be built once,
but can then produce arbitrarily many fresh, never-before-seen instances cheaply, sidestepping much
of the recurring authoring cost while still getting the leak-exposure benefits of never reusing the
exact same instance twice.

## 4. Canary strings and contamination-detection tripwires

### 4.1 The idea

A canary string is a unique, otherwise-meaningless marker string embedded in or alongside a
benchmark's published materials specifically so that its later appearance in a training corpus (or
its being memorized/regurgitated by a trained model) is a detectable, unambiguous signal that the
benchmark leaked and was ingested. This is a deliberate, proactive instrumentation choice made *by
the benchmark's own creators*, distinct from a downstream consumer trying to detect contamination
after the fact with n-gram search (the Datasets module's topic) — it's the benchmark-side analogue
of a "beacon" or watermark, designed in at publication time.

BIG-bench (Srivastava et al., 2022) is a well-known example of a public benchmark that shipped with
an explicit canary string convention, published alongside guidance asking anyone building web-scale
training corpora to filter out any document containing the canary string before including it in a
training set — an appeal to good-faith self-policing by the field, functioning simultaneously as (a)
a contamination-avoidance courtesy signal for careful data curators and (b) a detection tripwire for
the benchmark's own maintainers, since a model that has memorized text adjacent to the canary
string, or a training-data audit that finds the canary string present in a corpus, is direct
evidence of leakage.

### 4.2 Mechanics of using a canary string as a detection tripwire

1. **Generate a high-entropy, unique marker string** unlikely to occur by chance in any unrelated
   text (e.g., a long random alphanumeric token, often paired with a fixed, greppable prefix like
   `canary GUID <random-string> DO NOT TRAIN ON THIS DATA`) and embed it in the benchmark's
   published files (e.g., in a header/comment of the dataset file, or as metadata accompanying each
   item).
2. **Publish an explicit, machine-readable statement** asking data-curation pipelines to search for
   and exclude documents containing the canary before scraping/training — this is a
   voluntary-compliance mechanism, not an enforcement one, and its effectiveness depends on the
   broader field actually implementing the corresponding filter, which is a real limitation (Section
   4.3).
3. **Periodically audit for the canary's appearance** in places it shouldn't be: searching public
   web crawls (e.g., Common Crawl snapshots) for the canary string to see whether the benchmark's
   own published files (or copies/mirrors/forum discussions of them) have propagated into corpora
   that feed pretraining; and, more directly, probing candidate trained models by prompting for
   likely completions around the canary context and checking for verbatim regurgitation, which — if
   it succeeds — is strong direct evidence that the model's training data contained the
   canary-adjacent text.
4. **Treat a positive hit as an actionable finding**, not just a data point: it should trigger
   investigation of how the leak happened (which specific mirror/reproduction/forum-post carried
   it), an assessment of the practical severity (was it isolated to one document, or found broadly),
   and, if severity warrants, an eval-set rotation (Section 3) for the affected benchmark.

### 4.3 Limitations of canary strings

- **Compliance is voluntary and adoption is inconsistent.** A canary string only prevents leakage
  into a *given curator's* training data if that curator actually implements the corresponding
  filter; there is no mechanism forcing every organization building a web-scale pretraining corpus
  to honor the convention, and in practice adoption has been uneven across the field.
- **A canary protects the file it's embedded in, not the ideas/answers themselves.** If someone
  quotes benchmark items in a blog post or forum thread without the canary string attached (which is
  exactly the common, harder-to-prevent contamination pathway — a Reddit thread discussing "the
  answer to that tricky MMLU question about X" rarely carries the original file's embedded canary),
  the canary provides zero protection against that specific, very common leak vector.
- **Detection, not prevention, for anyone who doesn't cooperate.** For any data curator who ignores
  or is unaware of the convention, the canary's only remaining value is as a post-hoc detection
  signal (Section 4.2, step 3) rather than a leak-prevention mechanism — which is still useful
  (better to know a benchmark leaked than not to know) but is a materially weaker guarantee than the
  term "canary" might suggest.
- **A sufficiently capable model could, in principle, learn to recognize and suppress
  canary-adjacent text specifically** if canary conventions became well-known and a lab wanted to
  game a contamination audit rather than genuinely avoid training on the affected data — this is a
  somewhat exotic adversarial concern relative to the much more mundane "most contamination is
  incidental, not adversarially concealed" reality, but it's worth naming as a reason canary
  auditing shouldn't be the *only* contamination-detection layer a lab relies on
  (n-gram/embedding-based decontamination against the actual eval content, per the Datasets module,
  remains necessary regardless of canary-string use).

## 5. A worked example: deciding whether an anomalously high score is contamination

Suppose a new model scores 91% on a public reasoning benchmark, an 11-point jump over the previous
best model, while showing only a 2-point improvement on your private, never-published eval covering
a closely related skill. How do you reason about this before writing "state of the art" in a release
note?

Start with the base rates. Public-benchmark jumps of that size do happen for genuine reasons — a
targeted data-mixture change, a new synthetic-data pipeline specifically for the skill in question,
or a scale increase can plausibly move a single benchmark by double digits. So the anomaly is not
"an 11-point jump is impossible," it is "an 11-point jump on the public benchmark paired with only a
2-point jump on a closely related private benchmark is a divergence that needs an explanation,"
because absent contamination or overfitting to the public benchmark's specific format, you'd expect
the two related evals to move together at least roughly, since they are meant to measure overlapping
skill.

Next, separate the competing explanations rather than jumping to the contamination conclusion by
default. Possible explanations, in rough order of how easy they are to check: (a) the two evals
genuinely measure different things more than "closely related" suggested, and the divergence is real
and benign — check by inspecting item-level content overlap in skill and format between the two
evals; (b) the public benchmark's specific answer format or prompt convention was specifically
targeted by post-training (a narrow, format-specific overfitting that isn't full contamination but
is also not generalizable capability) — check by testing the model on reformatted/paraphrased
versions of the public benchmark's items and seeing whether the score holds; (c) genuine
contamination — check via the canary-string and n-gram-overlap tooling from Section 4 and the
Datasets module, run specifically against this benchmark's known content, and via a targeted
regurgitation probe (prompt for a benchmark item's likely continuation and check for suspiciously
exact recall of item text or answer keys the model should have no legitimate way to produce).

The operational takeaway for a staff-level evaluator: a single-metric anomaly is a prompt to
investigate, not a verdict either way, and the private/held-out eval's main practical value in
exactly this situation is giving you a second, independent data point to triangulate against before
either dismissing the public number as contamination-tainted or celebrating it as a genuine
breakthrough.

## 6. Putting it together: a contamination-aware evaluation program

A mature evaluation program layers these defenses rather than picking one:

1. **Public benchmarks** for external, reproducible capability claims and community comparability,
   with training-time decontamination (Datasets module) applied on a best-effort basis and an
   explicit acknowledgment that residual contamination risk is nonzero and grows over a benchmark's
   public lifetime.
2. **Private, held-out eval sets**, built from originally-authored items or programmatically
   generated instances, access-controlled and never partially disclosed, used as the internal
   ground-truth capability signal that is not subject to the same public-benchmark leak risk.
3. **Rotation** of both the private sets (on a cadence, and reactively on any suspected leak) and,
   where feasible, awareness that public benchmark scores should be discounted or cross-checked
   against private-set trends as a public benchmark ages and its contamination risk accumulates.
4. **Canary strings and periodic leak audits** on any benchmark a lab authors and cares about
   protecting, functioning as an early-warning system layered on top of, not instead of, the
   decontamination and privacy measures above.
5. **Cross-checking public and private results against each other** as a sanity check, following the
   worked-example logic in Section 5: if a model shows a large, otherwise-unexplained gap between
   its public-benchmark score and its private-eval score on a closely related capability,
   contamination of the public benchmark is one of the first hypotheses worth investigating,
   alongside the alternative explanation that the two evals simply measure meaningfully different
   things, which is itself worth confirming rather than assuming away.

None of this makes contamination a solved problem — the honest staff-level position is that
contamination risk is a permanent, managed background condition of evaluating models trained on
web-scale data, not a bug that gets fixed once. The goal of contamination-aware evaluation design is
to make sure that background condition doesn't silently invalidate the specific capability claims an
organization is actually relying on to make decisions.

## Cross-references

- Detecting and removing contamination from a training corpus (n-gram overlap search,
  embedding-based near-duplicate detection against known eval sets) is covered in
  `..\01_Datasets\005_Contamination_Detection_And_Decontamination.md`.
- Specific named public benchmarks, their known contamination histories, and benchmark saturation
  are covered in `..\06_Benchmarks`.
- Statistical treatment of comparing a public (potentially contaminated) score against a private
  eval score, and how much difference is meaningful vs. noise, is covered in
  `007_Statistical_Rigor_In_LLM_Evaluation.md`.

