# Human-in-the-Loop Design Patterns

## Why Full Autonomy Is Usually the Wrong Default

It's tempting to treat human-in-the-loop (HITL) machinery as a temporary crutch — something you bolt
on because the model isn't good enough yet, to be removed once it improves. That framing is mostly
wrong. Even with a hypothetically perfect model, some actions are irreversible enough, or
consequential enough, that a system design shouldn't grant an autonomous agent unilateral authority
over them: deleting production data, transferring money above a threshold, sending an
externally-visible communication on a company's behalf, merging code that touches authentication.
The question isn't "is the model smart enough to be trusted," it's "what's the cost of being wrong,
and can that cost be undone" — and for a meaningful slice of real-world actions, the honest answer
is that some human accountability needs to remain in the loop regardless of model quality, both for
genuine risk management and because organizations and regulators reasonably expect a human to be
answerable for certain classes of decisions.

At the same time, the entire point of building an agent is to reduce human toil, and a design that
routes every single action through a human approval step has simply rebuilt a slow, expensive manual
process with extra steps — the agent isn't saving anyone time if a human has to review and approve
each of its twenty micro-actions per task. The craft of human-in-the-loop design is in finding the
narrow set of actions that actually need a human, keeping the review lightweight and
well-contextualized when it happens, and making sure the friction is calibrated to real risk rather
than applied uniformly out of an abundance of caution that quietly kills the automation's value.

## Pattern 1: Approve-Before-Execute for Risky Actions

The most direct pattern: the agent forms an intended action, presents it to a human, and does not
execute until approved. This is the right pattern for actions that are both high-consequence and
infrequent enough that the review doesn't become a bottleneck — a refund above a policy threshold, a
database schema migration, a production deployment.

The design details that separate a good approve-before-execute flow from an annoying one are mostly
about what the human is shown and how much friction the approval carries. Showing a human "the agent
wants to call `execute_refund(amount=450, order_id=48213)`" is technically an approval gate but a
bad one, because it forces the reviewer to reconstruct context (why this amount, why this order,
what did the customer actually ask for) before they can make a real judgment — at which point the
"efficient" automated system is slower than a human just handling the ticket directly. A
well-designed gate instead surfaces the *reasoning*, not just the raw action: what the customer
asked for, what policy justifies the amount, and what the agent already checked, so the human's job
is a quick sanity check against something legible, not an investigation.

```python
@dataclass
class ApprovalRequest:
    action: str
    args: dict
    reasoning: str          # why the agent believes this action is correct
    supporting_context: dict  # e.g. order details, policy excerpt cited
    risk_tier: str
    requested_at: datetime
    expires_at: datetime    # approvals shouldn't hang forever


def request_approval(action, args, reasoning, context) -> ApprovalRequest:
    req = ApprovalRequest(
        action=action,
        args=args,
        reasoning=reasoning,
        supporting_context=context,
        risk_tier=classify_risk(action, args),
        requested_at=now(),
        expires_at=now() + timedelta(minutes=30),
    )
    approval_queue.enqueue(req)
    return req  # caller awaits req.resolution (approve/reject/expire)
```

Note the `expires_at` field: an approval request that sits unresolved forever is a silent failure
mode of its own — either the customer is left waiting indefinitely with no feedback, or the agent's
context window moves on and it loses track of the pending action. A production design needs an
explicit expiry policy (auto-escalate to a different reviewer, auto-reject with an apologetic
message to the user, or a safe default action) rather than leaving "what happens if nobody looks at
this" undefined.

## Pattern 2: Confidence-Based Escalation

Rather than gating every instance of a specific action type, this pattern gates based on the agent's
own uncertainty about the *specific* case at hand — the same action (say, answering a policy
question) might execute autonomously in most cases and escalate in others, depending on a confidence
signal computed per-instance. This is more nuanced than a static risk classification by action type,
because it captures the reality that even a normally-safe action can be uncertain in an unusual
case, and even a normally-risky action type can sometimes be clear-cut.

Useful confidence signals include retrieval quality (how strong the match was between the query and
the supporting knowledge, discussed in more depth in the RAG-grounded support agent chapter),
self-reported model confidence (asking the model to state how sure it is — only moderately reliable
in isolation, but useful as one signal among several), agreement across multiple samples (running
the same query twice, or through two different prompts, and checking whether the answers agree —
disagreement is a strong signal of genuine ambiguity), and historical outcome data (if similar past
actions of this shape were later reversed, corrected, or complained about, that's a direct empirical
signal worth feeding back into the threshold).

```python
def confidence_score(retrieval_result, self_reported, sample_agreement) -> float:
    # weighted blend; weights tuned against historical outcome data
    return (
        0.5 * retrieval_result.top_similarity
        + 0.2 * self_reported
        + 0.3 * sample_agreement
    )

def route_by_confidence(action, confidence: float) -> str:
    if confidence >= 0.85:
        return "auto_execute"
    if confidence >= 0.55:
        return "execute_with_notification"  # act, but flag for async review
    return "require_approval"
```

The middle tier here — **execute with notification** — is worth calling out because it's underused
relative to how useful it is. Not every uncertain action needs to *block* on a human; for actions
that are moderately risky but also reasonably easy to reverse or correct after the fact, it's often
better to let the agent proceed and simultaneously flag the action for a human to review
asynchronously, catching genuine mistakes within minutes to hours rather than blocking every user in
real time waiting on a reviewer's availability. This only works, of course, for actions where
"reversible after the fact" is actually true — it's the wrong choice for anything genuinely
irreversible.

## Pattern 3: Review Queues and Sampling-Based Audit

For high-volume, lower-individual-risk actions, real-time gating (of any kind) doesn't scale — you
can't have a human look at every one of ten thousand daily auto-generated email drafts before they
go out. The pattern that fits this volume is a review queue that a human works through
asynchronously, combined with sampling: rather than reviewing every action, review a statistically
meaningful random sample (plus anything flagged by automated checks as borderline), and use the
sample's outcome rate to estimate overall system health and to catch systematic problems before they
scale.

This shifts the human's role from gatekeeper-of-every-instance to auditor-of-the-population, which
is a fundamentally different (and much more scalable) job. The design implications are that the
review queue needs good triage — surfacing the highest-value items first (recent, higher-stakes, or
already-flagged-as-borderline actions) rather than presenting a flat, chronological list — and that
the audit sampling rate itself should be a tunable, risk-weighted knob: sample a higher fraction of
a brand-new action type the system has little track record with, and taper the sampling rate down as
track record accumulates and demonstrates the action type is reliably safe, which mirrors how canary
releases work in general software deployment.

```python
def should_sample_for_review(action, historical_stats) -> bool:
    if action.risk_tier == "high":
        return True  # always review high risk, regardless of volume

    if historical_stats.sample_count(action.action_type) < MIN_SAMPLES_FOR_TRUST:
        return True  # new action type: review heavily until track record exists

    error_rate = historical_stats.error_rate(action.action_type)
    # taper sampling rate down as observed error rate stays low
    target_rate = min(0.5, max(0.02, error_rate * 5))
    return random.random() < target_rate
```

## Designing the UX/API So HITL Doesn't Kill the Automation Benefit

The single biggest risk in human-in-the-loop design isn't picking the wrong pattern above — it's
implementing any of them in a way that makes the human step so slow or so poorly contextualized that
the "automation" ends up slower than doing the task manually, at which point the whole investment
has negative value. A few concrete practices consistently make the difference.

**Batch similar approvals together.** If an agent generates twenty similar low-risk actions (say,
twenty auto-drafted responses that all cite the same policy), presenting them one at a time to a
reviewer, each requiring a full context switch, is far slower than presenting them as a batch with a
single "approve all" affordance and the ability to drill into any individual one that looks off.
Batching by similarity is a genuine UX design problem, not just an API detail — grouping needs to be
based on what actually varies between the items in a way that matters for the reviewer's judgment,
not just superficial similarity.

**Make the default path the fast path.** If 95% of approval requests get approved unchanged, the UI
and API should be optimized for one-motion approval (a single click, a single API call with sensible
defaults) rather than requiring the reviewer to fill out a form every time — reserve the
higher-friction "edit before approving" or "reject with reason" flows for the genuine minority of
cases that need them.

**Give the agent a way to keep working while waiting.** A poorly designed system blocks its entire
task on a single pending approval; a well-designed one lets the agent continue other, independent
parts of the task (or other users' tasks entirely, if it's a shared worker pool) while a specific
approval is pending, and resumes the blocked branch once the decision comes back. This connects
directly to the async execution patterns discussed in the previous chapter — a synchronous, blocking
approval step inside an otherwise-async system reintroduces exactly the problems that async
architecture was meant to solve.

**Return an explicit, structured decision, not just an implicit signal.** The approval interface
(whether it's a UI a human clicks or an API a downstream system calls) should return a decision
object — approved / rejected / modified, with an optional reason and, for the modified case, the
corrected parameters — rather than something ambiguous like "the human didn't respond." An
unresolved decision after a timeout is a distinct state (see the `expires_at` handling above) from
an explicit rejection, and the agent's downstream logic needs to treat them differently: a rejection
might mean "try a different approach," while a timeout might mean "escalate to a different reviewer"
or "fail safe."

**Track approval outcomes as training signal, not just a gate.** Every approve/reject/modify
decision is a labeled data point about where the agent's judgment diverges from a human's — feeding
this back into confidence-threshold tuning (as in the `route_by_confidence` example above), into
prompt or retrieval improvements, and into deciding which action types have earned a lower
review-sampling rate over time is how the human-in-the-loop system becomes self-improving rather
than a static tax that never gets cheaper.

## Common Anti-Patterns Worth Naming

A few recurring mistakes show up often enough in real deployments, and in interview discussions,
that it's worth naming them explicitly rather than only describing what good looks like.

**The all-or-nothing gate.** A team ships an agent with either full autonomy or a blanket
approve-everything requirement, discovers one is too risky and the other too slow, and treats this
as evidence that "human-in-the-loop doesn't work for our use case." The actual lesson is almost
always that risk wasn't tiered — the fix is going back to the action-level risk classification
discussed in the coding-agent and support-platform chapters, not abandoning HITL altogether.

**Approval requests with no expiry and no owner.** A request that can sit in a queue indefinitely,
with no single person or role responsible for picking it up, will eventually be the one that sits
for six hours while a customer waits. Every approval request needs both a timeout policy (as in the
`expires_at` field above) and a clear routing/ownership model — round-robin, skill-based, or
severity-based — so "who looks at this" is never ambiguous.

**Treating the audit log as the control.** Logging that an action happened, and logging that a human
"approved" it, is not the same as the human having exercised real judgment. A system that can
technically show a compliance auditor "every high-risk action had a human sign-off" while those
sign-offs were rubber-stamped in under a second each has satisfied the letter of an oversight
requirement without the substance — this is the approval-fatigue failure mode discussed below, and
it's worth calling out separately because it's often invisible until someone specifically audits
reviewer behavior rather than just the presence of a reviewer.

**Confidence thresholds set once and never revisited.** A threshold tuned at launch, before the
system has real production data, is a guess. Treating approval outcomes as a feedback signal (as
described above) and revisiting thresholds on a regular cadence — monthly, or whenever a new action
type is introduced — is what turns an initial guess into a genuinely calibrated system over time.

## Trade-off: Friction vs. Safety, and Approval Fatigue

There's a well-documented failure mode in human review systems generally, and it applies directly
here: if reviewers are shown too many low-value approval requests, they habituate to clicking
"approve" without really evaluating each one — a phenomenon usually called approval fatigue or
rubber-stamping. This is a genuinely dangerous outcome, because it produces the *appearance* of
human oversight (there's a person in the loop, there's an audit log showing approvals) without the
substance of it, which can be worse than not having the gate at all, since it creates false
confidence in a control that isn't actually functioning. The mitigations are the same ones already
discussed — keep the volume of true real-time approvals low by using confidence-based routing and
notification-tier handling for the merely-moderate-risk cases, make each individual review fast and
well-contextualized so genuine attention is cheap to give, and periodically audit reviewer behavior
itself (e.g., inject a small number of deliberately-flagged test cases, or measure whether rejection
rates ever move, since a reviewer with a 0% rejection rate over thousands of approvals is itself a
signal worth investigating) rather than assuming the presence of a human gate is self-verifying.

The broader trade-off to hold in mind when defending a HITL design in an interview: every additional
point of friction you add reduces both the speed benefit of automation and the rate of genuine
mistakes reaching production, and the right amount of friction is the amount that matches the actual
cost distribution of being wrong for that specific action — not a uniform policy applied everywhere
out of caution, and not zero friction everywhere in pursuit of maximum automation. Being able to
name this trade-off explicitly, and to justify a specific threshold with a specific cost argument
("a wrong $20 auto-approved discount is a rounding error; a wrong $20,000 wire transfer is not, so
the threshold sits there, not at zero and not at infinity") is usually what separates a strong
answer from a generic one.

