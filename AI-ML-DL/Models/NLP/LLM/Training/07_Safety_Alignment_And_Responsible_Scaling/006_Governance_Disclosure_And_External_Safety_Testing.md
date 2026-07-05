## Governance, Disclosure, and External Safety Testing

### 1. From Model Cards to System Cards: The Disclosure Mechanism Itself

Before asking what frontier labs disclose, it is worth being precise about the artifact that carries the disclosure,
because the genre has a specific origin and has been stretched well past its original design intent. "Model Cards
for Model Reporting" (Mitchell et al., 2019, a Google-authored paper) proposed a short, standardized document to
accompany a released ML model — intended use, out-of-scope uses, factors relevant to evaluation (demographic groups,
environmental conditions), the metrics used to evaluate it, and quantitative results broken out by those factors.
The original proposal was scoped to comparatively narrow, single-task models (an image classifier, a toxicity
detector) where "here is what this model was tested on and how it performs across subgroups" is a tractable, largely
complete disclosure. This is a confirmed, citable origin point, not a retrospective label applied to the practice
after the fact.

Frontier labs releasing general-purpose LLM-based systems adapted this genre into what is now usually called a
**system card** rather than a model card — a naming choice that itself signals scope creep from "a report on a
model's measured statistical behavior" to "a report on an entire deployed system," including the model weights, the
safety-training layered on top, deployment-time mitigations (classifiers, usage policies, rate limits), and
increasingly, results from the kind of dangerous-capability evaluations that file 001's RSP/Preparedness-Framework
machinery is designed to trigger on. This expansion is a real, confirmed shift in practice (both OpenAI and
Anthropic, among other labs, publish "system card"-labeled documents alongside frontier releases), not merely a
terminology change — a system card for a 2024-2025-era frontier model is a substantially longer, more heterogeneous
document than a 2019-era model card for an image classifier, because the object being described is no longer
well-characterized by a handful of subgroup-broken-out accuracy metrics.

In current practice, a frontier-lab system card typically contains, in some combination and at some level of detail
(exact contents vary lab to lab and release to release — there is no binding template):

- **Capability summary.** A qualitative and quantitative account of what the model can do, usually anchored to
  standard academic and industry benchmarks (reasoning, coding, math, multilingual performance) plus qualitative
  capability claims specific to the release.
- **Benchmark and eval results**, increasingly including a summary-level accounting of results from
  RSP/Preparedness-Framework-relevant dangerous-capability evaluations — e.g., a stated ASL/risk-category
  determination and a high-level description of the bio-, cyber-, or autonomy-uplift testing that produced it (the
  eval methodology and mechanics behind these numbers are the subject of file 002's info-hazard-constrained
  treatment; this file is concerned with what gets surfaced about them publicly, not with how they are designed).
- **Known limitations and failure modes.** Documented weaknesses — hallucination tendencies, known
  jailbreak-susceptible categories at a general level, multilingual or multimodal performance gaps, and similar.
- **Intended use and deployment guidance.** What the model is and is not licensed or recommended for, aimed at
  downstream developers and enterprise customers as much as at regulators or the public.
- **Safety-training methodology, described in general terms.** A qualitative account of the alignment pipeline used
  (RLHF-style post-training, Constitutional-AI-descended techniques, deliberative-alignment-style safety reasoning,
  or similar), without reproducible implementation detail.
- **Red-teaming summary findings**, sometimes — an aggregate account ("external red-teamers identified categories of
  concern in X, Y, Z, which were mitigated via A, B") rather than a raw findings log.

A useful way to hold the disclosure decision in mind is as an explicit filter applied to everything a lab's internal
safety and evaluation process produces, only a fraction of which survives into the published card. A rough sketch of
that filter, as a decision process (this is an analytical reconstruction for teaching purposes, not a disclosed
algorithm any lab has published in this form):

```
function decide_disclosure(finding):
    # finding: a candidate piece of internal information —
    # an eval result, a red-team discovery, a known limitation,
    # a training-data fact, a vulnerability, etc.

    if finding.reveals_exploitable_attack_path_in_detail:
        # e.g., the exact prompt sequence that elicits a dangerous
        # capability, or step-by-step jailbreak mechanics
        return WITHHOLD(reason="information hazard",
                         mitigation="publish aggregate/summary only,
                                     consider narrow disclosure to
                                     gov't safety institute or affected vendor")

    if finding.reveals_unpatched_vulnerability:
        return DELAY(reason="responsible-disclosure-style timing",
                      release_when="mitigation is shipped or a
                                     coordinated-disclosure deadline passes")

    if finding.reveals_competitively_sensitive_method:
        # training data composition/sources, architectural specifics,
        # proprietary safety-technique implementation detail
        return WITHHOLD(reason="competitive/IP, and possibly
                                 copyright-litigation caution")

    if finding.is_legally_exposed:
        return ESCALATE_TO_LEGAL(reason="litigation/liability review
                                          required before any public
                                          statement")

    if finding.is_negative_but_not_a_safety_hazard:
        # e.g., an embarrassing benchmark regression, a use case
        # where the model underperforms a competitor
        return JUDGMENT_CALL(
            considerations=["does withholding this materially impair
                             outside verification of safety claims?",
                            "is this substantively different from
                             ordinary competitive discretion?"])

    return DISCLOSE(form="system card section, at a level of
                           granularity that supports independent
                           verification without reproducing
                           the underlying attack/method")
```

The point of walking through this filter explicitly is that a system card's contents are the output of a selection
process, not a raw transcript of what the safety org internally knows — and the categories in that filter map
directly onto the omissions catalogued in the next section.

### 2. What System Cards Typically Omit, and the Specific Reasons Why

Each of the following omission categories is a general characterization of industry practice, not a claim that a
specific lab has stated its reasoning in exactly these terms for a specific document — but the categories
themselves, and the fact that these are the recurring cited justifications, are broadly confirmable from public
commentary, lab statements, and the visible pattern of what does and does not appear across published system cards.

**Exact training data composition and sources.** No major frontier lab publishes a full accounting of training
corpus provenance at the level of "these specific web domains, books, code repositories, licensed datasets, in these
specific proportions." The stated and inferred reasons are layered: competitively, the data mixture is a genuine
input to model quality that competitors would value knowing; legally, in an environment with active copyright
litigation against multiple labs over training data, granular disclosure creates discoverable, citable claims that
plausibly increase litigation exposure rather than reduce it — a lab's caution here is at least as plausibly driven
by ongoing legal risk management as by pure competitive secrecy, and the two motives are not mutually exclusive.

**Full red-team methodology and raw findings.** Labs summarize red-teaming ("external testers probed for X category
of harm and found Y level of susceptibility, mitigated via Z") far more often than they publish the actual prompts,
techniques, or step-by-step methodology that surfaced a given weakness. The primary stated rationale is an
information-hazard concern structurally identical to the one file 002 addresses for dangerous-capability evals:
publishing the exact mechanics of how a jailbreak was discovered can function as a roadmap for reproducing it,
handing capability to exactly the audience (bad-faith users) the finding was meant to help defend against. A
secondary, less publicly emphasized but real factor is litigation/liability caution — a detailed internal account of
"we knew the model could be induced to do X, in this much detail, this far in advance" is potentially discoverable
and citable in later legal proceedings, which creates an incentive toward less granular internal-methodology
disclosure independent of any pure information-hazard logic.

**Model weights and undisclosed architectural specifics.** Essentially no frontier lab publishes weights for its
most capable models (a few labs have released open-weight models at various capability tiers, which is a distinct,
separate business/strategy decision from system-card disclosure practice and not the subject of this file), and
system cards do not disclose parameter counts, exact architecture variants, or training infrastructure details
beyond what a lab has separately chosen to reveal in research publications. This is treated industry-wide as core
competitive IP rather than safety-relevant disclosure, though a fair critique (raised in Section 4) is that
architecture and scale are not obviously irrelevant to independent risk assessment either.

**Specific unpatched vulnerabilities.** Where a red-team or external tester finds an exploitable weakness that has
not yet been mitigated, disclosure is typically delayed until a fix ships — a direct import of the
coordinated-disclosure norm from cybersecurity vulnerability handling (a security researcher who finds a software
vulnerability conventionally reports it privately to the vendor and agrees to a disclosure timeline, rather than
immediately publishing exploit details). Applying this norm to model behavior rather than software code is a genuine
adaptation, not a one-to-one transplant — a "vulnerability" in a model is often a fuzzier,
harder-to-patch-definitively thing (a class of jailbreak technique rather than a single code defect), which means
the coordinated-disclosure clock is less crisply defined than in traditional security disclosure, and "the fix has
shipped" is a softer claim for a probabilistic, hard-to-fully-patch model behavior than for a software CVE.

**Precise dangerous-capability eval task content.** As covered in depth in file 002, publishing the literal task
content of a bio-, chem-, cyber-, or persuasion-uplift evaluation risks being an information hazard in itself — the
eval design can double as a curriculum for the exact capability being tested for. System cards accordingly report
eval results (a score, a risk-level classification, a qualitative statement) without reproducing the underlying task
set, and this file treats that omission as settled context rather than re-deriving the information-hazard argument.

**Granular negative findings that are not, strictly, safety hazards.** This is the least clean category and worth
being honest about. A benchmark regression relative to a competitor, an underperforming use case, an internal eval
where the model behaves oddly in a commercially embarrassing but not dangerous way — these have a plausible claim to
being ordinary competitive discretion (no company publishes a full account of everywhere its product is weaker than
a rival's), but they are also exactly the kind of finding a skeptical outside reader might reasonably want
visibility into when trying to assess whether a lab's overall safety claims are trustworthy. The honest position is
that the boundary between "legitimate information-hazard or IP protection" and "avoiding disclosure that would
simply look bad" is not cleanly verifiable from outside the organization — an external reader cannot, in general,
distinguish a withheld finding that was genuinely dangerous to publish from one that was merely unflattering,
because the same non-disclosure decision is consistent with either explanation and the lab controls the account of
its own reasoning. This ambiguity is itself a substantive critique of the current disclosure norm, not a minor
caveat: a disclosure regime whose omissions cannot be independently audited for legitimacy is, definitionally,
trust-based rather than verification-based, and that gap recurs as the throughline of this entire file.

### 3. Third-Party Red-Teaming and Pre-Deployment Government Access

**The general pattern (confirmed as a recurring practice, though specifics vary by lab and release).** Frontier labs
commonly grant external researchers or contracted specialists early, pre-release access to a not-yet-launched model
specifically to probe it adversarially before the public ever sees it — access is typically bounded by NDA,
time-limited to the pre-launch window, and scoped to specific risk categories (biosecurity domain experts probing
bio-uplift, cybersecurity specialists probing offensive-cyber capability, linguists and regional experts probing
multilingual harms, and so on) rather than being an open, unstructured bug hunt. Some of this activity has been
structured in bug-bounty-like terms — adapting the cybersecurity industry's coordinated vulnerability-disclosure
norms (report privately, get credited, sometimes get paid, disclosure timed to mitigation readiness) to model
behavior rather than to a discrete software defect. This adaptation is real and confirmed as a general pattern
across the industry, but it is a looser fit than in traditional software security: a "bug" in a model is often a
class of behavior along a spectrum of severity and reproducibility rather than a single well-defined defect with a
clean patch, so bounty-style programs for models tend to involve more judgment calls about severity and reward than
a typical software CVE bounty does.

**Government AI safety institute partnerships — the publicly discussed arrangements.** The **UK AI Safety Institute
(AISI)**, established following the UK's hosting of the Bletchley Park AI Safety Summit in November 2023, has been
publicly described as having received pre-deployment access to evaluate frontier models from multiple labs,
including OpenAI, Anthropic, and Google DeepMind, around 2024, testing for capability and safety properties ahead of
public release. This is a specific, confirmable institutional fact as of the period in question.

The US counterpart is a case study in exactly the kind of institutional instability this file is structured to warn
about, and it should be read as such rather than as a settled reference: a **US AI Safety Institute** was
established under the National Institute of Standards and Technology (NIST) in 2024, following commitments connected
to the 2023 US executive order on AI (discussed further in Section 5), and was likewise publicly described as
engaging in pre-deployment evaluation access arrangements with frontier labs including OpenAI and Anthropic around
2024. Its precise current name, institutional home, staffing, and mandate should not be treated as fixed at time of
reading — US AI-policy institutions and executive-branch AI priorities have already been revised across
administration changes in the 2024-2026 window, and an institute's structure, funding, and even its continued
existence in the form described here are exactly the kind of detail that can and does change on a timescale shorter
than this document's shelf life. Flagging this explicitly is more useful, for interview purposes, than memorizing a
specific current name: the durable fact is "a US federal government body under NIST was set up in 2024 to do
voluntary pre-deployment frontier-model evaluation, and its subsequent status is unsettled and should be checked
independently" — that sentence remains true regardless of exactly what the institute is called or how it is
organized by the time it is read.

**The limits of these arrangements, stated precisely.** As publicly described (not as a matter of binding statute,
in most of what has been publicly discussed for these specific 2024-era arrangements), these government-institute
testing relationships have generally been:

- **Voluntary, offered by the labs rather than compelled by binding law.** The labs providing pre-deployment access
  chose to do so as a matter of policy commitment (connected to the voluntary commitments discussed in file 001,
  Section 4) rather than because a statute required it, for the arrangements publicly discussed in this period.
- **Without a binding government veto over release.** Public descriptions of these arrangements have not described
  the safety institutes as holding the power to block a model's launch; their role has been characterized as
  evaluation and advisory, feeding into — but not overriding — the lab's own internal launch decision.
- **Under-specified in scope and feedback loop.** What exactly gets tested, how deeply, and precisely how the
  institute's findings feed back into the lab's own go/no-go process has not been fully publicly detailed by either
  the labs or the institutes involved, which makes it hard for an outside observer to assess how substantively these
  evaluations shape outcomes versus how much they function as an additional, informative-but-non-binding data point
  alongside the lab's own internal RSP/Preparedness-Framework evaluation process.

The right way to characterize this entire arrangement, for interview purposes, is as a **voluntary, evolving
practice actively being negotiated between labs and governments in real time** — not as a mature regulatory regime
with settled institutional roles, defined legal authority, or a fixed set of participating institutes. Anyone citing
this section should mentally timestamp it and expect that the specific institutional names, mandates, and even the
voluntary-versus-binding character of the access could have shifted by the time it is read.

### 4. The Commercial-Safety Tension, Held Honestly

Frontier labs are simultaneously safety-mission-driven research organizations and commercially competitive product
companies selling API access and consumer products in a market with substantial revenue at stake — this dual
identity is not a hidden contradiction so much as the explicit, stated self-description of these organizations, and
it is the single most important lens for understanding why disclosure practice looks the way it does.

Some labs have adopted unusual governance structures specifically intended to manage this tension, and these are
worth naming precisely because they are concrete, confirmable facts rather than mission-statement rhetoric:

- **Anthropic** is structured as a **public-benefit corporation (PBC)**, a corporate form that legally permits (and
  in some formulations, obligates) the board to weigh a stated public-benefit mission alongside shareholder
  financial return rather than being bound to pure profit-maximization, paired with a **Long-Term Benefit Trust** —
  a body of trustees, independent of commercial shareholders, holding a class of stock with the power to elect and
  remove a portion of Anthropic's board specifically to safeguard the company's stated safety mission against pure
  commercial pressure over time. These are confirmed, publicly described structural facts about Anthropic's
  corporate form.
- **OpenAI** originated as a nonprofit and restructured around a **"capped-profit" subsidiary overseen by a
  nonprofit board**, with the nonprofit's stated mission (ensuring artificial general intelligence benefits humanity
  broadly) formally positioned as taking precedence over investor returns, and with investor returns in the
  capped-profit entity structurally limited (capped) rather than open-ended. OpenAI's exact corporate structure has
  itself been the subject of public reporting and restructuring discussions across 2023-2025, and — consistent with
  this file's calibration point about fast-moving specifics — the precise current details of OpenAI's structure
  should be treated as needing independent verification rather than assumed fixed, even though the general shape
  (nonprofit-mission-oversight-over-a-capped-commercial-entity) is a confirmed starting design.

Both structures are genuine, confirmed attempts to build a legal/governance counterweight to pure commercial
incentive, and both are the subject of genuine, ongoing public debate about how effectively they function in
practice — critics on one side argue these structures are largely symbolic, insufficiently tested under real
competitive pressure, or too easily worked around by boards or leadership when the stakes are high; defenders argue
that the mere existence of a structural mechanism with legal teeth (a trust with actual board-appointment power, a
nonprofit board with actual authority over a capped-profit entity) is a meaningfully different and stronger
commitment than an ordinary corporate mission statement, even before any specific test case proves out how it
behaves under real pressure. This document takes no position on which characterization is more accurate — the point
worth holding for an interview is that these structures exist, are specific and nameable rather than vague, and that
their real-world efficacy is genuinely, not performatively, contested.

**Where the tension concretely bites in disclosure decisions.** The abstract tension becomes a specific, repeated
operational choice at exactly the disclosure boundary described in Section 2:

- Disclosing detailed red-team findings or eval methodology in full risks handing competitors a capability or
  safety-technique roadmap (a rival lab reading exactly how you elicited or mitigated a dangerous capability learns
  something about both your model's behavior and your safety-engineering approach for free), and risks handing bad
  actors a literal attack roadmap in the information-hazard sense already discussed.
- Disclosing too little undermines exactly the audience the disclosure genre is nominally for — outside researchers
  trying to independently verify safety claims, journalists trying to report accurately on model risk, and
  regulators trying to assess whether voluntary commitments are being honored all depend on the published record
  being substantive enough to check against, and a system card that is mostly capability marketing with safety
  claims asserted rather than evidenced does not support that function.

**No settled resolution exists in current industry practice.** This is not a case where one lab has found the
correct answer and others are lagging — different labs have visibly made different choices about where to draw the
disclosure line (more or less granular eval-result reporting, more or less detail on red-team findings, more or less
willingness to publish negative results), and the line each lab draws is negotiated case by case, release by
release, rather than governed by any external, binding, or even fully consistent internal standard. A staff-level
answer should be able to name this as an unresolved, actively negotiated tension — not resolve it, because it has
not been resolved by the industry itself.

### 5. The Regulatory Landscape: Mechanisms, Not Memorized Bill Statuses

This section covers ground that is, at the time of writing, unusually unstable relative to most technology policy
areas — executive-branch AI policy in the US has already changed direction more than once in the 2023-2026 window,
state legislative activity is ongoing and its outcomes are not fixed, and international frameworks are explicitly
soft-law rather than binding treaty regimes still being negotiated. Every specific claim below should be read as a
snapshot of what was publicly true at some point in this window, not as a claim about what is true when this
document is read. The framing goal for a staff-level interview answer is fluency with the underlying mechanisms and
tensions — compute-based thresholds as an imperfect but auditable proxy, voluntary versus binding commitments, and
the layering of federal, state, and international governance — rather than a memorized, inevitably-stale list of
current bill numbers and statuses.

**The EU AI Act.** The EU AI Act establishes a **risk-tiered regulatory structure**: different categories of AI
system face different obligations depending on an assessed risk tier (ranging from minimal-risk systems facing
essentially no specific obligations, up through high-risk systems facing substantial compliance requirements, with
certain uses classified as prohibited outright). Layered on top of this general risk-tier structure, the Act
includes provisions specifically for **general-purpose AI (GPAI) models** — the category frontier LLMs fall into —
with an additional, higher tier of obligation triggered for GPAI models assessed as posing **"systemic risk."** The
mechanism used to operationalize the systemic-risk trigger is worth understanding precisely because it is a
genuinely interesting piece of regulatory design: rather than trying to define "systemic risk" purely in terms of
assessed real-world capability (which regulators cannot easily audit or verify independently, and which a regulated
entity has every incentive to characterize favorably), the Act ties the systemic-risk designation in part to a
**training-compute FLOPs threshold** — a specific, large quantity of floating-point operations used in training,
used as a triggering proxy criterion for additional obligations. The regulatory logic for choosing a compute-based
proxy over a pure capability-claim standard is straightforward and worth being able to articulate: training compute
is externally measurable and auditable in a way that "is this model dangerously capable" is not — a regulator can,
in principle, verify or estimate compute expenditure from infrastructure and cost data with less reliance on the
regulated party's own characterization of its model's abilities, whereas a capability-based standard requires either
trusting the lab's self-assessment or running independent evaluations the regulator may lack the resources or access
to perform at scale. The acknowledged cost of this choice is that compute is an **imperfect proxy for actual risk**:
a model trained with less compute but using a more efficient architecture, better data, or a more effective
post-training recipe could plausibly pose more real-world risk than a larger-compute model trained less effectively,
and a pure-FLOPs threshold does not directly capture that. This is a confirmed, structural design tradeoff in the
Act's approach, not a flaw unique to any drafting error — it is the same general "measurable proxy versus true
target" problem that shows up in file 001's discussion of dangerous-capability eval thresholds, applied at the
regulatory rather than internal-governance layer. The specific FLOPs threshold number, the exact timeline for phased
obligations coming into force, and the specific enforcement mechanisms and penalties have been subject to revision
and phased implementation since the Act's adoption, and should not be treated as fixed reference numbers — the
mechanism (compute-based proxy triggering a higher obligation tier) is the durable thing to know; the specific
numbers are not.

**The United States: executive-branch volatility and the state-law patchwork.** The US illustrates a different
structural dynamic: in the absence of comprehensive federal AI legislation, governance has come from a combination
of executive-branch action (inherently reversible by a subsequent administration) and a growing patchwork of
state-level legislation (filling perceived gaps left by federal inaction, and itself uneven and shifting). A 2023
executive order on AI was issued by the Biden administration, directing federal agencies toward a range of AI
safety, security, and reporting practices, including provisions connected to the frontier-model reporting
requirements and the NIST-housed safety institute discussed in Section 3. That executive order was subsequently
revised or rescinded by a later administration — this specific sequence (an executive order issued, then
substantially altered by a change in administration) is a concrete, confirmable illustration of a general structural
point worth internalizing rather than a detail to memorize precisely: **executive-branch policy is inherently less
durable than legislation**, because it can be reversed by the stroke of a subsequent administration's pen without
requiring Congressional action, whereas a statute requires new legislation (a materially higher bar) to undo. The
exact current state of US federal executive AI policy at the time this document is read should be independently
verified rather than assumed from this description.

Alongside federal-level volatility, US states have been legislating individually, producing a patchwork rather than
a single national standard. The general pattern — states filling a perceived federal-action gap with their own
frontier-AI-safety legislation, with mixed and evolving outcomes — is illustrated by California's experience: a
frontier-AI-safety bill (widely referred to by its bill number, SB 1047) that would have imposed specific
safety-testing and reporting obligations on developers of the largest frontier models was passed by the state
legislature and then **vetoed** by the governor in 2024, followed by continued state legislative activity on similar
ground, including a subsequent bill (referred to as SB 53) taking a related but distinct approach. This sequence
should be understood as **illustrative of the pattern**, not as a current-status claim — the specific fate of any
named bill, and the broader trajectory of California's or other states' frontier-AI legislation, is exactly the kind
of fact that shifts on a timescale shorter than this document's usefulness, and a reader should verify current
status independently rather than rely on the bill-specific detail here. The durable, general point is that in the
absence of settled federal legislation, individual US states have repeatedly moved to legislate frontier-AI safety
obligations themselves, producing a fragmented compliance landscape for labs operating nationally, and that this
state-level activity has not converged into a single stable standard as of the period this document covers.

**Other jurisdictions and international coordination, briefly.** The **United Kingdom** has, at least through the
period covered by this document, generally taken a comparatively more voluntary, innovation-permissive regulatory
posture relative to the EU's more prescriptive statutory approach — favoring sector-specific guidance and voluntary
commitments (including the AISI arrangement discussed in Section 3) over a single binding cross-sector AI statute,
though this posture, like everything else in this section, is a policy stance rather than a law of nature and is
subject to change. **China** has taken a more directly state-administered approach to AI governance, with specific
regulatory requirements (including content and generative-AI-service registration/approval mechanisms) reflecting a
governance model oriented around state oversight of deployed AI services rather than the EU's risk-tiered
market-regulation model or the US's currently fragmented executive/state pattern — this is a general
characterization of a different governance philosophy, not a claim to have enumerated China's specific current
regulatory instruments exhaustively. **International coordination efforts** have so far taken the form of soft-law
and diplomatic coordination rather than binding cross-border regulation: the **Bletchley Declaration**, issued at
the UK's AI Safety Summit at Bletchley Park in November 2023, was a multi-country joint statement on frontier AI
risk, followed by the **Seoul Summit** in May 2024 (which produced the Seoul Frontier AI Safety Commitments
discussed in file 001) and subsequent successor summits. These are genuine, confirmed diplomatic and coordination
events, and they matter as evidence that governments are actively trying to coordinate on frontier-AI risk across
borders — but they should be understood precisely as **soft-law coordination mechanisms** (joint statements,
voluntary commitments, shared frameworks for discussion) rather than as binding treaties creating enforceable
cross-border legal obligations. No binding, enforceable international AI-safety treaty regime has been established
as of the period this document covers.

**On currency, stated one more time explicitly because it matters more here than almost anywhere else in this
reference set:** this regulatory landscape is unusually fast-moving relative to typical technology regulation —
comparable historical regulatory buildouts (financial services, pharmaceuticals, aviation) generally settled into
stable statutory and institutional structures over a much longer timescale than AI policy has shown across
2023-2026. Every specific claim in this section — thresholds, bill statuses, institute names, executive-order status
— is a snapshot as of the writer's knowledge and should be independently verified for currency before being relied
upon. A staff-level interview answer on this material should demonstrate fluency with the mechanisms (why a
compute-based proxy gets chosen over a capability-based one and what that tradeoff costs; the durability gap between
executive action and legislation; the federal/state/international layering problem; voluntary versus binding
commitment as a recurring fault line) and should explicitly flag, unprompted, that any specific current-status claim
needs a currency check — that flagging behavior is itself a mark of calibrated seniority in this specific subject
area, more so than being able to recite a specific bill's current status correctly by chance.

### 6. Synthesis: Disclosure as the Verification Layer Over RSPs and Evals

Pulling this file together with the two files it is most tightly coupled to: file 001 describes Responsible Scaling
Policies and Preparedness Frameworks as internal, self-administered if-then governance — a lab commits to running
specific evaluations, hitting specific thresholds, and taking specific gating actions, all decided and verified
inside the organization. File 002 (by this document's description, not independently re-derived here) covers why the
dangerous-capability evaluations feeding those thresholds are themselves partly withheld from public view, for
information-hazard reasons that are largely legitimate on their own terms. This file is the layer that sits on top
of both: **governance and disclosure practice is what determines how much of a lab's internal RSP-triggered
decisions and eval results the outside world — independent researchers, journalists, regulators, the public —
actually gets to see and independently check**, as opposed to being asked to trust a self-report.

The honest, current answer to "how much verification does the outside world actually get" is: **partial, voluntary,
and unevenly distributed across labs.** Partial, because system cards summarize rather than fully reproduce eval
methodology and red-team findings, for reasons that are a defensible mix of genuine information-hazard concern,
IP/competitive protection, and litigation caution, in a proportion no outside party can cleanly verify. Voluntary,
because neither the system-card disclosure practice itself, nor the government-safety-institute pre-deployment
access arrangements described in Section 3, nor (for the large majority of jurisdictions and time periods covered in
Section 5) the specific content of what gets tested and disclosed, is compelled by binding law rather than chosen by
the lab as a matter of policy and reputational positioning — this is changing, unevenly and incompletely, as
regulation like the EU AI Act's GPAI provisions phases in binding obligations, but it has not yet converged into a
mature, externally audited disclosure regime anywhere. Unevenly distributed, because different labs have made
visibly different choices about disclosure granularity, and no external standard currently forces convergence toward
a common floor.

This is not a cynical conclusion so much as a calibrated one, and it mirrors the same both-sides-are-true structure
file 001 lands on for RSPs themselves: publishing a system card at all, imperfect as the genre is, is a materially
stronger transparency commitment than publishing nothing, and it gives outside parties a specific, citable document
to check claims against — a strictly better starting point than a fully private internal safety process. At the same
time, "partial, voluntary, and self-selected disclosure, whose omissions cannot be independently distinguished from
convenient omissions" is a genuinely different and weaker thing than an audited, externally verified disclosure
regime, and pretending otherwise in an interview answer would be the same category of overclaim this reference set
has flagged elsewhere (treating a lab's own safety framing as settled fact rather than as one interested party's
account of its own practice). A strong staff-level answer on governance and disclosure holds both of those
statements at once, names the specific mechanisms (system cards, third-party red-teaming, government
safety-institute access, the compute-threshold regulatory proxy, the executive-versus-legislative durability gap)
precisely, and resists the temptation to resolve the underlying tension into either "the industry is transparent
enough" or "none of this disclosure means anything" — neither is the calibrated position, and the calibrated
position is the one worth being able to hold under interview pressure.
