# Red-Teaming and Adversarial Evaluation

## 0. Red-teaming is a different question than capability eval

Every other module in this directory answers some version of "how good is the model at doing what
it's asked." Red-teaming answers a categorically different question: "what can this model be made to
do that it should not do." The distinction is not stylistic — it changes almost everything about how
the evaluation is designed:

- **Capability eval samples the task distribution you expect at deployment** (representative
  prompts, typical phrasing, the instructions real users will actually give) and measures average or
  aggregate performance across it. **Red-teaming deliberately samples the adversarial tail** — the
  prompts a typical user would never write, but a motivated bad actor, a curious jailbreaker, or an
  unlucky edge case might — and cares about the *worst* case it can find, not the average case.
- **Capability eval treats a low score as "the model isn't good enough yet."** Red-teaming treats a
  single successful elicitation of a serious harm (working bioweapon-synthesis guidance, reliable
  jailbreak defeating all safety training, a prompt that reliably extracts another user's private
  data) as a finding that matters on its own, independent of how rare that prompt pattern is in the
  overall traffic distribution — one bad, reproducible failure can outweigh a thousand good
  average-case responses, because the harm model is "did anyone manage to cause this specific bad
  outcome," not "what fraction of interactions were bad."
- **Capability eval is typically run once a model/checkpoint is ready and treated as a snapshot
  measurement.** Red-teaming is adversarial and adaptive by nature — it is explicitly trying to find
  whatever the current defenses don't cover, which means it has to keep evolving as defenses evolve,
  and a red-team result has a shelf life in a way a capability benchmark score mostly doesn't (a
  jailbreak technique that works today may be patched next week, and a red-team program's job is to
  keep looking for the next one, not to report last month's finding as a stable characterization of
  the model).
- **The goal is adversarial, not representative, coverage.** A capability benchmark wants items that
  look like real usage; a red-team wants items specifically engineered to be maximally likely to
  break the model, even if such a prompt would almost never occur "naturally" — an unusual roleplay
  framing, a multi-turn context-building strategy, an encoded or obfuscated request, a prompt
  exploiting a specific known model weakness. The value of the exercise comes precisely from *not*
  restricting attention to natural-looking inputs.

This is why red-teaming sits in its own module rather than as a variant of module `002`'s
judge-based scoring or module `005`'s trajectory scoring, even though it borrows tooling from both
(an LLM judge is frequently used to score whether a red-team attempt succeeded; an agent's full
action trajectory, not just its final text, is what a red-team on an agentic system needs to
inspect) — the defining thing about red-teaming is the *adversarial intent behind prompt/scenario
construction*, not the scoring mechanism applied afterward.

## 1. Internal vs. external/third-party red-teaming

### 1.1 Internal red-teaming

Conducted by the organization's own staff — often a dedicated safety/red-team function, sometimes
augmented by researchers and engineers across the org during a scheduled red-teaming exercise before
a major release. Internal red-teamers have the advantage of deep access: they typically know the
model's training data composition, known weaknesses from prior versions, the specific mitigations
already in place (so they can specifically probe whether those mitigations actually hold, rather
than rediscovering already-known issues), and can iterate quickly with direct access to intermediate
checkpoints, un-released capabilities, and the ability to test with elevated/unfiltered access modes
not available externally.

The limitation is a structural blind-spot problem: an organization's own staff share the
organization's own assumptions, cultural context, and mental models of what an attacker would try —
the set of attack strategies an internal team thinks to test is bounded by what that specific group
of people has thought of, which is a genuinely narrower space than what a large, diverse population
of external adversarial testers with different backgrounds, incentives, languages, and cultural
contexts would collectively surface.

### 1.2 External and third-party red-teaming

Conducted by people outside the core development organization: contracted domain experts (e.g.,
biosecurity, cybersecurity, or CSAM-prevention specialists brought in specifically for high-stakes
domains that require expertise the internal team doesn't have), academic or civil-society partners,
structured public/semi-public red-teaming events (bug-bounty-style programs, or organized exercises
like the widely reported DEF CON-affiliated LLM red-teaming events), and formal pre-deployment
third-party evaluation arrangements with independent safety-evaluation organizations.

External red-teaming buys genuine diversity of adversarial perspective and, in the case of
domain-expert engagement, checks against harm categories (e.g., dual-use biological/chemical
knowledge, cyber-offense capability) that require specialized expertise the core model-development
team is unlikely to have in-house and shouldn't try to informally approximate — a generalist ML
researcher is not well positioned to judge whether a model's chemistry answer constitutes materially
uplifting synthesis guidance, and pretending otherwise is itself a red-teaming-program risk. It also
provides an independent check that isn't subject to the same organizational incentive pressure an
internal team can face (pressure, even if unstated, to find "an acceptable number" of issues rather
than exhaustively hunting for every one, especially close to a release deadline).

The trade-offs: external engagements are slower to set up (contracting, access provisioning,
information-security review of what an external party can be shown), require careful handling of any
genuinely dangerous capability uplift discovered (external testers finding a real
bioweapon-synthesis-uplift issue creates its own information-hazard-handling problem — how do you
let them report it without the finding itself becoming a distribution vector), and typically cannot
be run at the iteration speed internal red-teaming can, since they don't have standing access to
every intermediate checkpoint.

### 1.3 Why both are used together

Neither internal nor external red-teaming alone is considered sufficient practice at a frontier lab:
internal red-teaming provides fast, deep-access, continuous coverage integrated into the development
loop; external and third-party red-teaming provides the diversity-of-perspective and
domain-expertise coverage internal teams structurally cannot fully replicate, and provides an
independent check that carries more credibility for external safety claims (e.g., in a model system
card or a regulatory disclosure) than a purely self-reported internal exercise would. Frontier model
release processes (as described in various labs' published system cards and
responsible-scaling-adjacent policies) typically combine both, often with external third-party
evaluation specifically gating release for the highest-stakes capability categories.

## 2. Automated adversarial-prompt generation

Manual red-teaming — a human deliberately crafting an attack prompt — does not scale to the volume
of probing needed to have confidence a model resists a broad space of attack strategies, and it is
itself bounded by human creativity and throughput. Automated adversarial generation techniques exist
specifically to expand coverage beyond what manual red-teaming alone can achieve, and to do so
continuously and cheaply as new checkpoints are produced.

- **Model-generated adversarial prompts (LLM-red-teams-LLM).** Use a separate LLM, explicitly
  prompted or fine-tuned to generate prompts designed to elicit a target harm category from the
  model under test, then evaluate the target's response (often with an automated judge, module
  `002`, itself checking whether the response constitutes a policy violation). This can be run as a
  single generate-then-test pass or as an iterative loop, where the attacker model observes the
  target's response and refines its next attempt based on what worked or didn't — a red-teaming
  analogue of iterative jailbreak refinement, automatable at scale far beyond what a human red-team
  could manually attempt.
- **Gradient-/search-based adversarial suffix and prompt optimization.** For models where gradient
  access is available (open-weight models, or via internal access for a lab red-teaming its own
  model), techniques in the lineage of GCG (greedy coordinate gradient, Zou et al. 2023) search
  directly over token sequences to find an adversarial suffix that, appended to an otherwise-refused
  request, maximizes the probability of a compliant (harmful) response — an automated,
  optimization-driven attack-discovery process rather than a human-authored one. These attacks are
  notable for often transferring across models (a suffix optimized against one model frequently
  degrades safety behavior on a different model too), which is itself an important red-teaming
  finding about the generality of certain vulnerability classes.
- **Genetic/evolutionary and mutation-based prompt search.** Start from a seed set of known attack
  strategies (roleplay framing, hypothetical/fictional framing, encoding/obfuscation, multi-turn
  context-building) and apply automated mutation and selection — keep variants that increase attack
  success rate against the target model, discard ones that don't, iterate — to discover novel
  combinations and phrasings a human wouldn't necessarily have tried, at a throughput no manual
  process can match.
- **Persona and multi-turn escalation strategies, automated.** Many successful jailbreaks are not
  single-prompt attacks but multi-turn strategies (establishing a fictional frame over several
  turns, incrementally escalating a request, or exploiting a model's tendency to maintain in-context
  consistency with an earlier, seemingly-benign framing) — automated red-teaming pipelines that
  generate and test entire multi-turn conversation strategies, not just single prompts, are
  necessary to cover this attack class, and are a meaningfully harder engineering problem than
  single-turn adversarial generation because the search space is a full conversation tree rather
  than a single string.
- **Coverage-based fuzzing analogues.** Borrowing from software-security fuzzing methodology:
  systematically vary a known attack template's surface parameters (topic, phrasing, language,
  encoding) to map out *which* variations succeed and which don't, producing a more structured
  characterization of a vulnerability's boundary than a handful of hand-picked examples would, which
  is directly useful for the training-feedback loop in Section 3 (you want to know the shape of the
  hole, not just one point inside it).

### 2.1 Limits of automation

Automated adversarial generation is a coverage and throughput multiplier, not a full substitute for
human red-teaming: automated attackers are themselves bounded by whatever attack strategies they
were seeded with, fine-tuned on, or can be steered to explore, and genuinely novel attack
*categories* (as opposed to novel instances within a known category) have historically tended to
come from human red-teamers noticing something conceptually new, which is then the point at which it
becomes worth automating discovery of variants within that newly-identified category. The two are
complementary stages of the same pipeline: humans (internal or external) for discovering new attack
categories, automation for exhaustively mapping and stress-testing known ones at scale.

## 3. Threat modeling: categorizing harms before you go looking for them

Effective red-teaming is not "throw creative prompts at the model and see what sticks" — the
highest-leverage programs start from an explicit threat model that names the harm categories being
probed for, because the right testing strategy, the right expertise requirement, and the right
severity framework differ substantially across categories.

- **CBRN (chemical, biological, radiological, nuclear) uplift.** Probing whether the model provides
  information that meaningfully increases a non-expert's ability to cause mass-casualty harm, as
  distinct from information that is freely available in a library and confers no *additional*
  uplift. This category requires genuine subject-matter expertise to assess correctly (a generalist
  cannot reliably judge what constitutes real uplift in synthesis routes or agent selection), is
  treated as maximally severe by every major lab's published policies, and is a primary driver of
  the external/third-party expert engagement discussed in Section 1.2.
- **Cyber-offense capability.** Probing whether the model can be used to meaningfully accelerate
  real-world cyberattacks — vulnerability discovery, exploit development, malware authoring — again
  requiring domain expertise to distinguish genuine uplift from generic, already-public security
  knowledge.
- **Persuasion and manipulation.** Probing whether the model can be induced to generate highly
  persuasive disinformation, personalized manipulation, or scalable influence-operation content, a
  harm category that is comparatively harder to define crisply (the line between "persuasive
  writing" and "manipulative content" is genuinely contested) and correspondingly harder to build a
  crisp automated checker for.
- **Privacy and data exfiltration.** Probing whether the model can be induced to reveal memorized
  personal information about real individuals, or, for agentic systems, to exfiltrate data via tool
  access it shouldn't have used that way.
- **Agentic/tool-misuse harms.** Distinct from the above because the harm materializes as an
  *action* rather than as text — an agent induced to make an unauthorized purchase, delete data, or
  take an irreversible real-world step. This category requires the trajectory-level evaluation
  machinery from module `005` rather than single-turn text scoring, and is covered further in
  Section 5 below.
- **Conventional policy-violation content** (garden-variety harmful, hateful, or disallowed content
  that isn't CBRN/cyber-severity but still violates usage policy) — the highest-volume category in
  practice, generally amenable to the general-crowd-plus-trained-annotator pipeline described in
  module `003` rather than requiring specialized outside expertise.

Naming these categories explicitly up front matters for two operational reasons: it determines which
findings require which kind of reviewer (a generalist internal red-teamer should not be the sole
adjudicator of CBRN-uplift severity, but is perfectly positioned to adjudicate conventional
policy-violation severity), and it lets a red-team program report coverage honestly — "we ran N
adversarial prompts against the model" is a much weaker claim than "we ran structured campaigns
against each of these six threat categories, with category-appropriate reviewers," and only the
latter tells you anything about what wasn't tested as thoroughly as something else.

## 4. How red-teaming findings feed back into training

Red-teaming that produces a report nobody acts on is a compliance exercise, not a safety practice.
The methodologically important part of red-teaming — and the part that distinguishes it from simple
vulnerability *reporting* — is the feedback loop back into the model itself:

1. **Triage and categorize findings.** Not every successful elicitation is equally severe or equally
   actionable — a red-team program needs a severity/priority framework (how dangerous is the
   elicited content, how easy is the attack to execute, how likely is a real user to stumble into it
   vs. how much specialized attacker effort it required) to decide what needs an urgent fix, what
   needs a scheduled fix in the next training cycle, and what is a known, accepted, documented
   residual risk (a category that should exist and be explicit, rather than every finding being
   implicitly treated as fully fixed once discovered).
2. **Convert successful attacks into training signal.** The most direct feedback path: add the
   red-team-discovered prompts (and a demonstrated safe/refusing response) into the safety-relevant
   supervised fine-tuning and/or RLHF preference data used for the next training cycle, so the model
   is directly trained away from the specific discovered failure — this is exactly analogous to
   using rejection-sampled/curated trajectories as SFT data elsewhere in a training pipeline,
   applied to safety data specifically. Frontier lab safety reports (e.g., Anthropic's and OpenAI's
   published model/system cards) describe iterative rounds of red-teaming feeding directly into
   subsequent safety-training passes as a standard part of the release cycle, not a one-off audit.
3. **Fix at the right layer, not just the symptom.** A specific successful jailbreak string can
   sometimes be patched narrowly (train the model to refuse that specific phrasing) without
   addressing the underlying vulnerability class, which is a well-known failure pattern — narrow
   patching produces a model that resists the exact reported attack but remains vulnerable to
   trivial variants. Mature programs push for the automated-variant-mapping described in Section 2
   specifically so the fix targets the *category* the finding revealed, not just the literal
   reported string, and then re-run red-teaming against the fix to check whether the category, not
   just the instance, was actually closed.
4. **Add persistent regression coverage.** Every confirmed finding (or a representative cluster of
   variants from Section 2's mapping) should be added to a standing red-team regression suite that
   gets re-run against every future checkpoint, precisely so a fix that works today doesn't silently
   regress in a later training run (e.g., a subsequent capability-focused fine-tuning pass
   inadvertently eroding a previously-fixed safety behavior — a real and recurring risk pattern in
   iterative model development) — this is the safety-domain analogue of the general
   regression-testing use of automatic metrics described in module `001`, Section 7.
5. **Feed severity-weighted findings into release-gating decisions**, not only into training data —
   if a red-team exercise surfaces a sufficiently severe, sufficiently easy-to-execute, and
   sufficiently unmitigated finding close to a planned release, the finding should be able to
   actually delay or block that release, which requires red-teaming results to have real
   organizational authority over the release decision rather than being an informational report
   delivered after the ship decision has effectively already been made. This is the structural
   difference between red-teaming as genuine risk management and red-teaming as a box-checking
   exercise.
6. **Track finding recurrence and time-to-fix as program-level metrics**, not just per-finding
   severity — a red-team program that keeps rediscovering variants of the same underlying
   vulnerability class release after release is signaling a training-pipeline or architecture-level
   gap that a one-off patch isn't addressing, and that pattern is itself a more important signal
   than any single finding's severity score.

## 5. Why red-teaming is never "done": the arms-race dynamic

A capability benchmark score has a natural endpoint — once a model reaches near-ceiling performance
on a well-designed benchmark, that specific measurement has largely finished telling you what it can
tell you (module `007`'s discussion of benchmark saturation applies directly). Red-teaming has no
equivalent endpoint, and treating a clean red-team report as "the model is now safe" rather than
"the model resisted everything this specific campaign tried, at this point in time" is a common and
consequential misreading of what a red-team result means.

Three forces keep this an ongoing, never-finished activity rather than a one-time gate:

- **Attackers adapt to defenses.** Once a jailbreak technique becomes public or well-known,
  defenders patch against it, and the adversarial community (researchers, curious users, and actual
  bad actors) shifts to new techniques — the same dynamic seen in computer security more broadly,
  where "patched" and "secure" are not synonyms. A red-team program's job is structurally similar to
  a security team's: continuous monitoring for new attack techniques, not a certificate issued once.
- **Model capability changes what's testable and what's risky.** A new capability (longer context,
  tool use, multimodal input, agentic autonomy) opens new attack surfaces that didn't exist in the
  previous version and that the existing red-team regression suite was never designed to cover —
  red-teaming has to be re-scoped, not just re-run, every time the model's capability profile
  changes materially.
- **The field's understanding of harm categories itself evolves.** Persuasion/manipulation risk, for
  instance, is an area where the field's own threat models have visibly matured over time as
  capabilities and real-world usage patterns became clearer — a red-team program's threat-category
  taxonomy (Section 3) needs periodic revisiting, not a one-time definition, to stay aligned with
  current understanding of what actually matters.

The practical implication for how to talk about this in an interview: describe red-teaming as a
standing organizational capability with a cadence (recurring campaigns, a regression suite, periodic
re-scoping) rather than as a pre-release checklist item, and be ready to name the specific mechanism
(Sections 3 and 4) by which a finding today changes what gets tested and trained against tomorrow.

## 6. Relationship to other evaluation methods in this module

Red-teaming is not a separate universe from the rest of this directory's methodology — it reuses the
same scoring machinery on a differently-selected input distribution:

- **Judge-based scoring of red-team attempts** (did this response actually constitute a policy
  violation, and how severe) uses the same LLM-as-judge methodology and inherits the same biases and
  validation requirements as module `002` — a red-team judge that is itself miscalibrated (e.g.,
  systematically under-flagging violations because it was validated on typical, non-adversarial
  content) is a serious, easy-to-miss failure mode, and validating a red-team-specific judge against
  expert human safety-policy judgments (module `003`, Section 2.1's point about specialized
  annotator pools for safety-sensitive rating) is if anything more important here than for
  general-quality judging, given the cost asymmetry of a missed severe finding.
- **Agentic red-teaming** — probing whether an agent can be induced to take a harmful real-world
  action (not just say something harmful) — requires trajectory-level evaluation exactly as
  described in module `005`, since the harm of concern is often in the action taken (an unauthorized
  purchase, a destructive file operation, exfiltrating data via a tool call) rather than in the
  surface text of any single turn.
- **Contamination-aware handling of red-team eval sets** — the confirmed, reproducible jailbreaks
  and attack templates a red-team program accumulates are exactly the kind of eval material that
  should never be published in full (module `004`'s private-eval-set discipline applies directly: a
  leaked jailbreak-string dataset is worse than useless, it's an attack toolkit).
- **Statistical treatment of attack-success rates** — reporting "attack success rate dropped from
  12% to 3%" between two model versions needs the same confidence-interval and sample-size
  discipline as any other benchmark comparison (module `007`), and is arguably more prone to being
  over-interpreted from small samples than ordinary capability benchmarks, precisely because
  red-team eval sets are often smaller and more expensive to construct than general capability
  benchmarks.

## Cross-references

- LLM-as-judge scoring of red-team attempts, and the bias/validation issues that transfer directly
  to this use case, are covered in `002_LLM_As_Judge_Methodology_And_Biases.md`.
- Trajectory-level evaluation, needed for red-teaming agentic/tool-using systems rather than pure
  text generation, is covered in `005_Agentic_And_Trajectory_Evaluation.md`.
- Private/held-out eval set discipline, directly applicable to protecting a red-team program's
  accumulated attack corpus, is covered in `004_Contamination_Aware_Evaluation_Design.md`.
- Broader safety-alignment training methodology that red-team findings feed into is covered in
  `..\07_Safety_Alignment_And_Responsible_Scaling`; this module covers the evaluation/discovery
  side, not the full training methodology.
- Statistical treatment of attack-success-rate comparisons across model versions is covered in
  `007_Statistical_Rigor_In_LLM_Evaluation.md`.

