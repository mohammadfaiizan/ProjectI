## Responsible Scaling Policies and Preparedness Frameworks

### 1. What an RSP/PF Actually Is, and Why It Is Not a PR Document

A Responsible Scaling Policy (Anthropic's term) or Preparedness Framework (OpenAI's term) — Google
DeepMind calls its version a Frontier Safety Framework — is a published, internal-governance document
that commits a lab to a specific *conditional structure*: define capability thresholds, measure against
them with concrete evaluations, and bind future training and deployment decisions to the measured result
rather than to a calendar date, a product roadmap, or a general statement of values. Anthropic's own RSP
text uses the phrase "if-then commitments" to describe this structure, and that phrase is a useful anchor
because it names precisely what the genre is trying to be: not a promise about intent, but a promise about
a mechanism. The generic form is a conditional rule, not a mission statement:

```
if eval_suite(candidate_model) crosses threshold_for_level(N+1):
    require: mitigations_for_level(N+1) are verified as in place
    if mitigations_verified:
        permit: continued training and/or deployment, gated by which
                mitigations apply (deployment-only vs. deployment+scaling)
    else:
        pause: the specific activity (training-scaling or deployment)
               that the triggered threshold governs, until mitigations
               are verified
```

This is a meaningfully different kind of artifact from a corporate AI-ethics charter or a set of deployment
principles ("we will be careful," "we value safety," "we are committed to beneficial AI"). Those documents
make no falsifiable commitment: there is no evaluation result that could be pointed to as having been
crossed, and no specific action the company is on record as owing in response to any measurable state of
the world. An RSP/PF, by contrast, is meant to be read the way an engineer reads a spec — a named
threshold, a named eval (or eval category) that operationalizes it, and a named consequence tied to
crossing it. Whether any given lab's document actually achieves this level of operational precision
throughout is itself a fair and contested question, taken up in Section 6, but the *intent and structure*
of the genre is to be an engineering-process document with consequences defined in advance of the trigger
event, not a values statement written after the fact to explain a decision already made — and this
self-description (process document, not values statement) is confirmed and explicit in both Anthropic's
and OpenAI's published policy text, not an outside characterization imposed on them.

A second structural feature worth naming up front: these documents are explicitly *scoped to catastrophic
and large-scale risk*, not to the full space of AI harms. Bias, ordinary misuse, low-stakes hallucination,
copyright, and labor-market disruption are real and heavily discussed harms in the broader AI-policy
conversation, but RSPs/PFs are deliberately narrower — they exist to govern the tail-risk case where a
frontier model's capabilities could provide meaningful uplift toward mass-casualty weapons, large-scale
cyberattacks against critical infrastructure, or loss of meaningful human control over an increasingly
autonomous system. This scoping choice is itself confirmed by both lead documents' own framing, and it is
worth being able to state precisely in an interview, since a common confusion is to treat "does this lab
have an RSP" as a proxy for "does this lab take AI safety broadly seriously" — the RSP/PF genre is
specifically about catastrophic/extreme-scale risk governance, and a lab's broader safety posture (content
policy, red-teaming for ordinary abuse, fairness work) is governed by separate internal processes not
described in these documents at all.

**The safety-case analogy, and where it holds and breaks.** People discussing this space routinely compare
RSP/PF-style thinking to safety-case regimes in nuclear power, aviation, and biosafety (BSL-1 through
BSL-4 containment levels for pathogen research). The analogy is useful for the core mechanism it shares —
a tiered system where the tier is determined by an assessed hazard property of the thing being handled, and
each tier mandates specific, escalating physical and procedural controls before you're permitted to
proceed — and BSL levels are the closest structural cousin, since ASL naming is an intentional echo of it
(Anthropic states this analogy directly in its RSP). It is important to be honest about where the analogy
breaks, and this is analytical framing rather than a claim either lab makes about itself: nuclear and
aviation safety cases are typically produced by the operator but then reviewed, licensed, and enforced by
an independent regulator with legal authority to halt operations (the NRC, the FAA); BSL containment levels
are similarly backed by institutional biosafety committees and, for the highest tiers, external inspection
regimes. Current frontier-AI RSPs/PFs have no equivalent external licensor — the lab defines the
thresholds, runs the evaluations, judges whether they've been crossed, decides what counts as adequate
mitigation, and signs off on all of it internally. The safety-case *form* has been imported; the
external-verification *institution* has not, at least not as of this writing, and this gap is the
throughline connecting most of the substantive criticism in Section 6.

### 2. Anthropic's Responsible Scaling Policy: AI Safety Levels

Anthropic's RSP (first published September 2023, revised multiple times since — most substantively in
late 2024, with further updates continuing after that) is organized around **AI Safety Levels (ASL)**,
deliberately modeled on biosafety level naming. This tiering, and the broad definitions below, are
confirmed by Anthropic's published RSP text; specifics of any given model's actual ASL assignment for a
specific release are typically disclosed in that model's system card rather than in the RSP document
itself, since the RSP defines the *general* threshold and process while the system card reports the
*specific* evaluation result for a given model.

- **ASL-1**: systems with no meaningful capacity for catastrophic misuse or autonomous harm — Anthropic's
  RSP uses this level for narrow, clearly non-frontier systems (its own published example is something like
  a chess-playing model), not for any current general-purpose Claude release. Essentially a floor category
  that exists to complete the ladder rather than to describe any system Anthropic is actively deciding
  about.
- **ASL-2**: current-generation frontier chat/assistant models as Anthropic has assessed them to date —
  models that show early, low-level dangerous-capability signals (e.g., some capacity to provide
  information relevant to hazardous topics) but not at a level assessed as materially increasing real-world
  catastrophic risk beyond what's already obtainable from existing resources like search engines, textbooks,
  or a knowledgeable human collaborator. Standard safety practices — the kind of harm-refusal training,
  red-teaming, and deployment review already normalized industry-wide — are deemed sufficient at this
  level. Confirmed: Anthropic's own public statements have placed its released Claude models in the ASL-2
  category as of the versions covered by this document's knowledge horizon.
- **ASL-3**: the first level at which the RSP mandates a qualitatively different bar, triggered by
  evaluation results showing a model could provide meaningful uplift to a non-expert actor seeking to cause
  mass-casualty harm (biological, chemical, radiological, or nuclear — the general "CBRN" framing), or
  could substantially uplift cyberattacks against critical infrastructure, or shows early, credible signs of
  the autonomous-replication and resource-acquisition capabilities that later feed into ASL-4 concerns.
  Crossing this threshold is meant to be determined by specific capability evaluations (run by Anthropic's
  internal red-teaming function, discussed below) rather than by an unstructured subjective judgment call,
  though how crisply "meaningful uplift" is operationalized into a reproducible pass/fail eval score is
  itself a live methodological question taken up in Section 6b. ASL-3 triggers two categories of required
  safeguards before a model assessed at that level can be deployed, or, per the RSP's separate
  training-scaling commitments, trained further at increased scale:
  - **Deployment safeguards**: a harm-refusal robustness standard specifically targeting the CBRN-uplift and
    similarly catastrophic misuse categories — meaning the model must be shown resistant to jailbreak
    attempts seeking that specific category of harmful output, not merely refuse the harm when asked
    directly and unadversarially. This is a materially higher bar than ordinary content-policy refusal
    robustness, because the threat model assumes a motivated adversary actively probing for a bypass rather
    than a casual user.
  - **Security safeguards**: hardening of model-weight security — access controls, exfiltration-resistance
    measures, and internal-access restrictions on the trained weights themselves — on the theory that an
    ASL-3-capable model's weights becoming available to a bad actor (via theft, insider misuse, or lax
    access control) would functionally hand that actor the dangerous capability regardless of what
    deployment-time refusal training the hosted product enforces. This dual structure — a deployment gate
    and a security/weight-protection gate, as distinct, separately-triggered requirements — is a specific,
    confirmed feature of Anthropic's published RSP and is worth being able to name precisely, since it is a
    common point of confusion to treat "ASL-3 safeguards" as a single undifferentiated bucket rather than
    two categorically different kinds of control (one about what the model will say, one about who can touch
    the weights at all).
- **ASL-4**: not yet fully operationally specified in public Anthropic documents as of this writing.
  Anthropic has stated ASL-4 standards will be published before they become necessary — i.e., before a
  model is actually assessed as approaching that level — rather than being fully pre-specified today. The
  general orientation, per Anthropic's own framing, is toward more severe autonomy/replication risk and
  larger-scale CBRN/cyber uplift, expected to require materially stronger security and deployment controls
  than ASL-3, but the concrete evaluation criteria and mitigation requirements remain to be published.
- **ASL-5**: explicitly a placeholder in Anthropic's own document — a conceptual slot reserved for systems
  with catastrophic risk potential exceeding all current human institutions' ability to counter, without
  concrete evaluation criteria yet defined at all. Anthropic is explicit that this level is aspirational and
  structural rather than operationally specified today, which is itself worth noting as an honest limitation
  acknowledged directly in the source document rather than a gap this summary is introducing on its own.

**Who decides, and with what evaluations.** Anthropic's RSP assigns capability-threshold evaluation to an
internal function generally referred to as the Frontier Red Team, alongside other internal safety and
alignment evaluation teams, whose job is to run the specific dangerous-capability elicitation evaluations —
bio/chem uplift testing, cyber-offense uplift testing, autonomous-replication testing, and similar — against
candidate models before or during scaled deployment decisions. The general shape of these evaluations
(what "elicitation" means, how you try to surface a genuine capability ceiling rather than a
refusal-shaped or capability-suppressed artifact) is the subject matter this file defers to a companion
evaluations file in this reference set rather than detailing here; this file treats the existence and role
of such evaluations as the input to the governance process, not their internal design.

The RSP also names a **Responsible Scaling Officer** — a specific, designated internal role, confirmed to
exist in Anthropic's published RSP, with responsibility for confirming ASL determinations, reviewing
whether required safeguards are actually in place before a threshold-gated action proceeds, and
maintaining the internal compliance process the RSP describes. This role sits alongside ordinary
executive- and board-level sign-off on major releases rather than replacing it — the RSP describes a
layered process in which the Responsible Scaling Officer's determination feeds into, rather than
substitutes for, the company's normal governance chain for a major release decision. The precise reporting
line, degree of independence from commercial and product leadership, and any instance of the role's
recommendation being overridden are not fully detailed in public materials, which is a fair point of
underspecification worth flagging explicitly rather than assuming resolved in either direction.

### 3. OpenAI's Preparedness Framework: Risk Categories and Scorecards

OpenAI's Preparedness Framework (first published December 2023, with a revised version published in 2025)
takes a structurally different shape from Anthropic's single ordinal ASL ladder: rather than one aggregate
risk score gating the whole model, it tracks **named risk categories independently**, each with its own
risk-level rating, and uses the *highest* category rating reached — not an average or blend — to determine
what governance action is required. This scorecard structure, separate tracked categories rather than one
blended number, is a confirmed, deliberate design choice in OpenAI's published framework, on the reasoning
that a model could be low-risk in most categories and high-risk in exactly one, and a blended average would
mask that single high-risk category rather than surface it.

The named categories have been revised across versions of the document, and this should be flagged
explicitly rather than treated as a fixed list. The original December 2023 framework named categories
including cybersecurity, biological/chemical weapons uplift, persuasion, and model autonomy. Later
revisions have adjusted this list — including, per OpenAI's own public update, narrowing or reweighting how
persuasion-related capability is tracked, and adjusting category definitions more generally as the
framework matured through practical use. An interview-grade answer should therefore name the *type* of
category (cyber-offense uplift, bio/chem uplift, persuasion/influence-operations capability,
autonomous self-improvement or replication capability) while explicitly caveating that the exact current
category list and definitions are versioned and have already changed at least once, rather than asserting a
single frozen taxonomy as a permanent fact of the framework.

Each category is rated on an ordinal risk-level scale — **low, medium, high, critical** in the original
framework's terminology — determined again by capability evaluations designed to elicit the risk-relevant
behavior: can the model meaningfully uplift a novice attempting to synthesize a biological threat, relative
to existing baseline resources like search engines and textbooks; can it autonomously carry out multi-step
cyber-intrusion tasks end to end without a human operator in the loop; can it generate persuasive content at
a scale or quality that meaningfully changes real-world influence-operation economics. The framework's
operational commitment: a model cannot be deployed if any tracked category reaches "high" risk without
adequate mitigations first being applied and verified, and OpenAI's framework further commits that a model
reaching "critical" in any category triggers restrictions on continuing to *develop and scale* it further,
not just on deploying it — the same deployment-gate-versus-scaling-gate distinction Anthropic's RSP draws
via ASL-3, expressed through OpenAI's category/level vocabulary instead of Anthropic's ASL vocabulary.

**Safety Advisory Group.** OpenAI's framework names a cross-functional internal body, the Safety Advisory
Group (SAG), whose confirmed role is to review capability-evaluation results against the framework's
category and level definitions and issue a recommendation on whether a model's assessed risk profile
permits proceeding, escalating that recommendation to OpenAI's leadership and board for the actual go/no-go
decision on higher-risk cases. As with Anthropic's Responsible Scaling Officer role, the SAG's practical
independence from commercial pressure, and the degree to which its recommendations have in practice ever
been overridden, is not something publicly documented in a way that allows an outside party to verify how
binding the mechanism actually is in a contested edge case — this is a fair, explicitly-flagged gap rather
than a resolved fact in either direction, and it is the same structural gap that shows up under Anthropic's
Responsible Scaling Officer role, just under different internal-body naming.

**A side-by-side comparison**, useful for keeping the two vocabularies from blurring together in an
interview answer:

| Dimension | Anthropic RSP | OpenAI Preparedness Framework |
|---|---|---|
| Structuring axis | Single ordinal ladder (ASL-1 to ASL-5) | Independent categories, each with its own level |
| Aggregation | One level per model | Highest category level governs, not an average |
| Current-gen models | Assessed at ASL-2 (per public statements) | Assessed per-category against low/medium/high/critical |
| Escalation threshold named | ASL-3 (deployment + security safeguards) | "High" (deployment gate); "critical" (scaling gate) |
| Named internal review body | Responsible Scaling Officer | Safety Advisory Group (SAG) |
| Placeholder for future extreme risk | ASL-5 (unspecified) | No explicit named placeholder beyond "critical" |
| First published / notable revision | Sept. 2023; substantive revision late 2024 | Dec. 2023; revised version 2025 |

This table is a compression of the confirmed structural facts in Sections 2 and 3 above; it should be
treated as a study aid, not as an independent source of new claims — everything in it is asserted with more
context and appropriate hedging in the prose above.

### 4. The Broader Landscape: Convergence Toward a Shared Norm

Google DeepMind published its own **Frontier Safety Framework** (2024, subsequently updated), structurally
similar in spirit — capability thresholds, which it terms "critical capability levels," tied to evaluation
results, with corresponding required mitigations — and its existence, alongside Anthropic's and OpenAI's
documents, is the clearest evidence that RSP-style if-then governance became a shared norm across major
frontier labs within a roughly twelve-to-eighteen-month window (2023-2024) rather than remaining one lab's
idiosyncratic practice. Other labs and organizations (including further entrants publishing their own
safety-framework-style documents) have continued to add to this landscape since, which is part of why this
document treats "which labs currently have a published framework" as a moving target rather than a fixed
list to enumerate exhaustively.

This convergence did not happen in a vacuum: it sits alongside two rounds of voluntary, government-convened
commitments. The **White House voluntary AI commitments** (July 2023) were a set of commitments made by
major AI companies including internal and external red-teaming prior to release, information-sharing on
risks across companies and with government, and investment in model-weight security — themes that map
directly onto the deployment-safeguard and security-safeguard categories described above. The **Seoul
Frontier AI Safety Commitments** (May 2024, made at the AI Seoul Summit) went a step further procedurally:
signatory companies specifically committed to publishing their own safety frameworks, including defining
thresholds at which severe risks would be deemed intolerable absent mitigations — i.e., the Seoul
commitments effectively asked companies to commit to producing something in the RSP/PF genre, rather than
merely to good general safety practice.

These government-convened commitments are confirmed historical events, and are a meaningful part of why
RSP/PF-style publication became something like an expected norm across the frontier-lab set rather than a
single company's unilateral choice. It is important to be precise, however, that the commitments themselves
are still voluntary and non-binding: they created reputational and coordination pressure toward publishing
a framework, not a legal obligation to adopt any particular threshold, evaluation methodology, or
enforcement mechanism. No signatory faces a legal penalty for publishing a weak framework, revising it
downward, or failing to follow it in a specific instance — the commitments bind attention and reputation,
not conduct, in the way a statute or treaty would.

This entire area should be treated as fast-moving and already-revised-more-than-once: Anthropic's RSP,
OpenAI's Preparedness Framework, and Google DeepMind's Frontier Safety Framework have each gone through at
least one substantive public revision since their first publication, and any claim about "the current
version" of any of these documents should be understood as a snapshot subject to further change rather than
a stable long-term reference point. This document describes the structure and mechanisms that have been
stable across revisions — the if-then form, the tiered thresholds, the deployment-versus-scaling gate
distinction, the named internal review bodies — while flagging that category lists, specific numeric or
qualitative thresholds, and specific mitigation requirements are the parts most likely to have moved by the
time this is read, and should be verified against the labs' current published text rather than assumed
frozen at whatever this document describes.

### 5. The Engineering-Process Reality: What a Trigger Decision Looks Like in Practice

Strip away lab-specific vocabulary, and the underlying decision pipeline that both Anthropic's ASL
structure and OpenAI's category/level structure implement is the same shape:

```
function evaluate_scaling_gate(candidate_model):
    results = run_capability_elicitation_evals(candidate_model)
    # e.g., bio/chem uplift probes, cyber-offense task suites,
    # autonomous-replication/agentic-task suites, persuasion probes
    # -- the general category this reference set's companion
    # evaluations file covers in depth as "dangerous capability evals"

    level = classify(results, threshold_definitions)
    # Anthropic: which ASL does this result profile correspond to
    # OpenAI: which risk level, per tracked category, does this
    #         result profile correspond to

    if level.exceeds(current_authorized_level):
        required_mitigations = lookup_mitigations(level)
        sign_off = internal_governance_review(required_mitigations)
        # Anthropic: Responsible Scaling Officer + leadership
        # OpenAI: Safety Advisory Group + leadership/board

        if sign_off.mitigations_verified_in_place:
            authorize(deployment_surface = restricted_to(level),
                      training_scaling  = permitted_if(level.allows_further_scaling))
        else:
            gate(pause_deployment = True,
                 pause_training_scaling = level.requires_scaling_pause)
    else:
        authorize(deployment_surface = standard, training_scaling = standard)
```

The evaluations that feed `run_capability_elicitation_evals` are themselves the hard, unglamorous
engineering work underlying the whole framework. This document treats them only at the level of "capability
elicitation evals exist and produce the numbers the threshold logic consumes," and defers their actual
design — how you elicit a genuine capability ceiling rather than a refusal-shaped artifact, how you
red-team against a model that may be deliberately or inadvertently underperforming on its own evals, how
you validate an eval against real-world uplift rather than mere benchmark performance — to this module's
dedicated evaluations file.

**A worked illustration of the distinction that matters most.** Suppose a candidate model, during routine
pre-release evaluation, scores well above the established baseline on a bio-uplift elicitation suite —
clearing the threshold associated with ASL-3 / a "high" bio/chem rating. Two different institutional
responses are consistent with the frameworks described above, and confusing them is a common error:

- If the lab judges that adequate deployment-time mitigations (jailbreak-robustness hardening targeted at
  the bio-uplift category, usage monitoring, staged or access-restricted release) can be verified and put in
  place, the model can still ship — later, and behind a materially different safeguard stack than an
  ASL-2/low-risk model would need, but it ships. This is a **deployment gate**: it changes the shape and
  timing of the release, not whether the lab continues building successors of comparable or greater scale.
- If the lab additionally judges that the *training-scaling* commitment tied to that threshold applies
  (which, per both frameworks, is reserved for the more severe tier — Anthropic's higher ASL bands, OpenAI's
  "critical" level rather than "high"), then work on scaling that model line further is paused until
  mitigations are verified, independent of whatever happens with the already-trained checkpoint's
  deployment. This is a **training/scaling gate**, and it is the categorically more expensive commitment,
  because it throttles the lab's own capability-frontier progress rather than just a specific product's
  release calendar.

Neither lab has, as of this writing, publicly confirmed having actually invoked the second, stronger form of
gate on a frontier model in production — which is worth stating plainly rather than implying either that it
has happened quietly or that the commitment is therefore empty; it is simply an observation that the
strongest lever described in both documents remains, publicly, untested in practice.

**What actually changes when a gate fires, concretely.** Across both frameworks, the documented menu of
operational responses includes: mandatory additional red-team re-evaluation before proceeding further;
narrowing the deployment surface — fewer customers, more usage monitoring, staged access rather than open
release, gated API access requiring attestation of legitimate use case; hardening model-weight access
controls and exfiltration defenses, up to and including restricting which employees can access raw weights
at all; raising the required jailbreak/adversarial-robustness bar a model must clear before its public
release is authorized, typically verified via dedicated red-team campaigns targeting the specific triggered
risk category; and, in the reserved, most severe case, halting further capability scaling on that model
line until the lab's own internal governance process certifies mitigations are in place. All of these are
confirmed as named categories of response in the published documents; which specific response a lab would
apply to a specific real trigger event is, by the nature of these frameworks, a judgment call made
internally at the time the trigger fires, not a fully pre-specified lookup table with one correct answer
per input.

### 6. Honest Critique: Where This Genre of Governance Is Genuinely Contested

**(a) Self-regulation without external enforcement.** Every mechanism described above — the threshold
definitions, the evaluation design, the classification of results, the sign-off, the choice of mitigation —
is decided and administered by the same organization whose commercial incentives the framework is meant to
constrain. A lab can revise its own RSP/PF, and both Anthropic and OpenAI have, in fact, revised theirs more
than once since first publication. Critics have specifically argued that some revisions — loosening
specific threshold language, narrowing category scope, adjusting what counts as adequate mitigation —
weakened the original commitment relative to what was first published, while the labs' own framing of the
same revisions is typically that they reflect refined understanding of what's actually measurable and
operationally meaningful, learned from having tried to apply the earlier version in practice, rather than a
weakening. This document takes no side on which characterization is correct for any specific revision — the
point to hold onto for an interview is that revision-direction disputes are a real, recurring feature of
this space, not a one-off controversy, and that "the RSP/PF is a living document the lab itself fully
controls, including the direction of its own revisions" is a structural fact independent of whether any
particular revision was substantively good or bad.

**(b) The threshold-definition problem.** Policy language like "could meaningfully assist a non-expert in
creating a biological weapon" is doing real conceptual work, but turning it into a crisp, reproducible eval
with a bright-line pass/fail is genuinely hard. Uplift relative to what baseline — a search engine, a
domain textbook, a knowledgeable human collaborator willing to help? Measured on whom — a red-teamer
already skilled enough to know what to ask, or a true novice with no relevant background at all, who might
ask a differently-shaped and less effective question? Validated how, given that there is no ethical way to
run a ground-truth trial of whether eval-passing uplift actually translates into real-world attack success?
This is not a criticism unique to any one lab's document — it is close to an inherent property of trying to
operationalize a catastrophic, low-base-rate risk into a repeatable benchmark, and it means that "threshold
crossed" determinations necessarily involve interpretive judgment even when the surrounding process looks
rigorous, quantitative, and procedurally careful on paper.

**(c) Competitive pressure versus unilateral gating.** A lab that actually pauses training-scaling or
restricts deployment because its own framework's threshold fired cedes capability and market ground to
competitors who either haven't adopted an equivalent framework, interpret their own thresholds more
leniently, or simply aren't yet at the same point on the capability frontier where the threshold would bind.
Critics have argued this creates a structural incentive against ever actually invoking the strongest gates,
and that credible mutual restraint plausibly requires either external coordination — binding multilateral
agreements among labs — or regulation with external enforcement, rather than relying on each lab's
unilateral willingness to eat a competitive cost that rivals don't have to bear. Labs' counter-framing is
that publishing a framework at all is intended precisely to create reputational and internal-organizational
pressure that makes reneging costly even absent external enforcement, and that a public commitment,
however unilaterally revocable in theory, is harder to quietly abandon in practice than a private one would
be. This is a genuine, unresolved argument about incentive design, not a question with a settled empirical
answer, and a strong interview answer represents both sides rather than picking one as obviously correct.

**(d) The audit and verification gap.** Outside of the still-limited pre-deployment testing arrangements
some governments have established with individual labs — covered in depth elsewhere in this reference set
rather than here — there is no standard, independent mechanism to verify that a lab's internal claims about
its own eval results, its ASL or risk-level classification of a given model, or its mitigation-verification
sign-off are accurate. The rare third-party evaluation arrangements that do exist in adjacent contexts —
external benchmark organizations given early model access, academic red-teaming partnerships — are narrow,
benchmark-specific, and not a general audit of RSP/PF compliance as such. "Trust the lab's self-report"
remains, as of this writing, closer to the operative reality than "an outside party independently checked
this," and that gap is the single most-cited structural weakness of the entire genre in external policy
commentary.

**(e) The genuine counter-arguments.** It would be inaccurate to present RSPs/PFs as valueless because they
are unenforced. A published threshold framework is a meaningfully stronger public commitment than no
framework at all, because it gives external researchers, journalists, and policymakers a specific, citable
claim to check the lab's behavior against — "you said X would trigger Y" is a more falsifiable statement
than "we value safety," even without an external enforcer standing behind it. Internally, safety and
alignment teams inside these organizations have argued — and it is a plausible mechanism even where it is
not independently verifiable from outside — that a written, leadership-adopted RSP/PF gives them
organizational leverage they would not otherwise have: a documented basis to say "this release cannot
proceed as currently planned" that exists independently of any individual safety researcher's personal
standing in a given internal debate at a given moment. And publishing specific thresholds, even imperfectly
operationalized ones, invites exactly the kind of external scrutiny and critique this section is itself
engaged in — a dynamic that a purely private, undisclosed internal risk-management process would not
generate at all, since there would be nothing on the record to hold the lab to. Both the critique and the
counter-argument are simultaneously true and in tension with each other, and holding that tension without
collapsing to either "this is meaningless" or "this solves the problem" is the calibrated position a strong
answer should land on.

### 7. What an RSP/PF Is Not, and What Staff-Level Fluency Requires

An RSP/PF is not a guarantee that a lab will not deploy or scale something dangerous — it is a conditional,
self-administered commitment to a *process*, and the process's actual bite depends on threshold definitions
and internal judgment calls that remain contestable even when followed in good faith by well-intentioned
people. It is not independently enforced — no external body currently has the standing authority to compel
compliance or verify a compliance claim the way a nuclear regulator or aviation authority can for their
respective industries, notwithstanding the structural analogy the ASL naming and the biosafety-level
framing deliberately invoke. And it is not free of interpretive judgment at any stage — from what counts as
"meaningful uplift" in a specific eval result, to what counts as a "verified" mitigation before a gate is
lifted, humans inside the organization are making calls that a purely mechanical if-then reading of the
document, of the kind sketched in the pseudocode above, would tend to obscure.

A genuinely staff-level command of this area means being able to do several things at once, on demand:

- Name the specific mechanics precisely — ASL-1 through ASL-5 and what operationally changes at ASL-3
  specifically (the split between harm-refusal robustness and weight-security hardening); OpenAI's
  category/level scorecard structure, the highest-category-governs aggregation rule, and the SAG's role —
  without hedging into vagueness or conflating the two labs' vocabularies.
- Distinguish, inline, which specific facts are confirmed by a lab's own published document versus your own
  reasonable inference about how the mechanism probably works in practice, rather than presenting inference
  with the same confidence as a sourced claim.
- Draw the deployment-gate-versus-training-scaling-gate distinction correctly and apply it to a novel
  hypothetical, as in the worked example in Section 5, rather than only being able to recite it as an
  abstract definition.
- Argue the engineering-process case — this is a real, if imperfect, advance over no public commitment at
  all, and it creates internal leverage and external scrutiny that would not otherwise exist — and the
  critique case — self-administered, hard to operationalize into bright-line evals, competitively fragile
  absent coordination, unaudited by any outside party — with equal fluency, in the same breath, without the
  answer collapsing into advocacy for or dismissal of the entire genre.

That combination — mechanistic precision plus even-handed critique, applied to a specific hypothetical
rather than only recited in the abstract — is what separates a strong answer here from either uncritical
repetition of a lab's own framing, or a cynical dismissal that ignores the real structural difference
between an if-then engineering-process framework and a values statement with no mechanism behind it at all.
