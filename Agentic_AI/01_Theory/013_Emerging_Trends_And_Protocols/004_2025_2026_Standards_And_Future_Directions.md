# 2025-2026 Standards and Future Directions in Agentic AI

## Table of Contents

1. Framing: What Actually Changed Between the Early Demos and Now
2. Convergence Around Interoperability Protocols
3. From "Did It Work Once" to Continuous Evaluation and Observability
4. Agent Identity, Accountability, and the Audit Trail Problem
5. Governance and Regulation Catching Up
6. From Single-Agent Demos to Production Multi-Agent Systems
7. Economic Interoperability: Agents That Transact
8. Consolidation Pressures on the Framework Layer
9. What Is Still Genuinely Unresolved
10. What a Senior Engineer Should Actually Watch
11. Summary

---

## 1. Framing: What Actually Changed Between the Early Demos and Now

It's worth being precise about the shape of the shift this chapter is describing, because "agentic
AI is maturing" is vague enough to be almost meaningless on its own. The concrete change is this: in
the early period of agentic AI (roughly 2023 into 2024), the dominant unit of progress was the
impressive single-agent demo — a model given a broad goal and a loop, shown accomplishing something
a static, single-turn model call could not. Demos like this were genuinely important for
establishing that the agent loop pattern worked at all, but they were almost universally evaluated
on "did it work this one time, on this one example," run by the same team that built the system,
with no adversarial pressure, no long-term operation, and no real accountability if it failed.

What changed through 2025 and into 2026 is that the field's hardest problems moved from "can an
agent do this at all" to "can an agent do this reliably, safely, observably, and accountably, at
scale, in production, across organizational boundaries, over an extended period of time, without a
human watching every step." That shift in the nature of the hard problem is what unifies the four
threads covered in this chapter — interoperability standards, evaluation and observability
infrastructure, accountability and governance, and the move from single agents to multi-agent
production systems — and it's the framing to reach for if asked in an interview "what's changed
recently in agentic AI" rather than reciting a list of individual product launches.

## 2. Convergence Around Interoperability Protocols

The interoperability landscape covered in depth in the first chapter of this section — MCP for
agent-to-tool communication, A2A for agent-to-agent communication — is the clearest instance of the
field moving from "everyone builds their own plugin format" toward a shared substrate, and the
trajectory from here is convergence rather than continued fragmentation. Three converging forces are
worth naming specifically. First, cross-vendor adoption of protocols that didn't originate with the
adopting vendor (every major model provider building MCP client and server support, despite MCP
originating at a competitor) signals that the industry has decided the interoperability layer is not
a place it wants to compete, the same way no cloud provider tries to differentiate on a proprietary
alternative to TCP/IP. Second, neutral governance — A2A's move to the Linux Foundation mirrors a
well-worn playbook (Kubernetes being the most cited precedent) for turning a vendor-originated
technology into genuinely shared infrastructure that competitors are comfortable co-depending on.
Third, the boundary between the two protocol families is visibly blurring at the edges, with agents
exposed over A2A being wrapped as MCP tools for simpler callers and MCP servers growing
longer-running, more stateful operation patterns that start to resemble A2A tasks; expect the next
couple of years to bring either explicit bridging standards or a gradual absorption of one
protocol's semantics into the other, rather than the current clean conceptual split persisting
indefinitely.

It would be a mistake, however, to read this convergence as "the interoperability problem is
basically solved and we should stop paying attention to it." As covered in the first chapter, the
parts that are converging are the wire-level and session-level mechanics; the parts that remain
genuinely unresolved — fine-grained authorization for autonomous action, trustworthy discovery
without a verified root of trust, and cross-organizational accountability when something goes wrong
— are not wire-protocol problems and will not be solved by protocol convergence alone. The realistic
expectation is that the *plumbing* layer of agentic interoperability finishes converging well before
the *governance* layer built around it does, and the governance layer is where the more interesting
and more difficult standards work of the next several years will actually happen.

## 3. From "Did It Work Once" to Continuous Evaluation and Observability

As the demos in Section 1 gave way to production deployments handling real user requests and real
business processes, "did it work on my test example" stopped being an adequate quality bar, for the
same reason it was never an adequate quality bar for any other piece of production software —
production traffic is more varied, more adversarial, and more consequential than any hand-picked
test set, and agentic systems compound this because a single bad decision early in a multi-step task
can cascade into a badly wrong final outcome that looks locally reasonable at every individual step.

The practical response has been the rapid maturation of observability infrastructure purpose-built
for agentic systems, distinct from both traditional application performance monitoring and from
earlier, single-call LLM evaluation tooling. The industry converged reasonably quickly on tracing as
the foundational primitive — capturing the full sequence of an agent's reasoning steps, tool calls,
and intermediate outputs as a structured trace, analogous to a distributed trace in traditional
microservice observability, rather than just logging the final input and output of a request. This
matters specifically for agents because most failures are not visible from the final output alone:
an agent that reached a correct-looking answer via a wrong or unsafe reasoning path, or that made
three failed tool call attempts before an unrelated fourth attempt happened to succeed, looks
identical to a clean, correct execution if all you log is the final response. Standardization
efforts around semantic conventions for capturing this kind of trace — extending existing
observability standards like OpenTelemetry with GenAI- and agent-specific fields for capturing
prompts, tool calls, token usage, and model identity in a vendor-neutral way — are a direct and
pragmatic response to the interoperability lesson from Section 2: teams did not want yet another
proprietary logging format tied to a specific agent framework or model vendor, for exactly the
reasons that drove them toward MCP for tool integration.

Evaluation methodology shifted in a parallel way. Where early evaluation asked "is the final answer
correct," mature agent evaluation increasingly asks "was the trajectory reasonable" as a distinct,
additional question — did the agent use an appropriate number of steps, did it call tools that were
actually necessary, did it avoid unsafe or out-of-scope actions along the way, and would a domain
expert endorse the *process*, not just the outcome. This is the same lesson computer-use evaluation
ran into directly (Section 7 of the multimodal and computer-use chapter): end-state correctness
alone is not a sufficient safety signal, because an agent can reach a correct or acceptable-looking
end state via a path that took unacceptable risks along the way. LLM-as-judge techniques, where a
separate model evaluates an agent's trajectory or output against a rubric, became a practical
necessity here simply because trajectories are too voluminous and too idiosyncratic for exhaustive
human review to scale, but production teams increasingly pair automated judging with continuous
sampling of real traffic for human review, specifically to catch the cases where an automated
judge's own blind spots would otherwise go undetected — a judge model can share correlated failure
modes with the model it's judging, so it cannot be the only check in the system.

Finally, evaluation moved from a pre-deployment gate to a continuous, always-on production activity.
Rather than treating "eval" as something done once before shipping a new prompt or model version,
mature agent operations treat it as an ongoing monitoring function — sampling live traffic, tracking
success and safety metrics over time, and catching regressions caused by upstream model updates,
changes in the tools an agent depends on, or gradual shifts in the population of tasks users
actually bring to the system, none of which a one-time pre-launch evaluation could ever detect.

## 4. Agent Identity, Accountability, and the Audit Trail Problem

As agents take more autonomous action with real consequences — spending money, modifying production
systems, communicating externally on an organization's behalf — the question "who or what did this,
and under whose authority" stops being a philosophical curiosity and becomes an operational and
legal necessity. This is driving early but concrete work on agent identity: giving an individual
agent (or agent instance, or agent acting on behalf of a specific user) a verifiable identity
distinct from both the underlying model and the human who ultimately authorized it, so that actions
can be attributed, scoped, and revoked at the right granularity. This is a meaningfully different
problem from traditional service-account identity in software systems, because an agent's authority
is often meant to be a delegated, bounded subset of a human's own authority for a specific task or
time window, not a fixed, standing set of permissions the way a typical backend service account
works — and revoking or auditing that delegation cleanly, especially across the organizational
boundaries A2A is meant to span, is still closer to an open design problem than a settled pattern.

Audit trails follow directly from this. An organization deploying an autonomous agent that can take
consequential action needs a tamper-evident record of what the agent did, what information it had
access to, and what it was told to do, sufficient to reconstruct and justify its behavior after the
fact — for internal post-incident review, for customer disputes, and increasingly for regulatory
purposes. The tracing infrastructure discussed in Section 3 is a necessary ingredient here but not
sufficient on its own; an audit trail has a different design goal than a debugging trace
(completeness and tamper-resistance rather than developer convenience), and building audit logging
that would actually satisfy a compliance or legal review, rather than just an engineer trying to
figure out why a test failed, is a distinct and still-maturing discipline.

The cross-organizational version of this problem, already flagged in the A2A discussion, is the
least solved piece: when an agent's action spans two organizations connected via a protocol like
A2A, there is currently no standard for producing a single, mutually-trusted audit record both
parties agree reflects what happened, and no standard mechanism for either party to independently
verify the other's account of a disputed interaction. This is a real, practical blocker to the more
ambitious cross-company "agent economy" vision often described alongside these protocols, and it is
likely to be solved, if it is solved, by a combination of protocol extensions, third-party
attestation services, and plain contractual practice, rather than by a purely technical fix.

## 5. Governance and Regulation Catching Up

Regulatory and organizational governance frameworks are, as usual, arriving after the technology
rather than ahead of it, but 2025 saw the beginning of governance language that explicitly
contemplates autonomous agentic systems rather than only single-turn generative outputs. Frameworks
like the EU AI Act include risk-based obligations that scale with what a system is used for and how
autonomously it operates, and enterprise AI governance practices (internal review boards, mandated
human-in-the-loop checkpoints for certain classes of decision, required documentation of an agent's
permitted scope of action) increasingly draw an explicit line between "generates a suggestion a
human acts on" and "takes action directly," treating the latter as warranting materially more
oversight regardless of how good the underlying model's accuracy metrics look in isolation.

The practical implication for engineers building these systems is that governance and compliance
requirements are becoming a design input on par with functional requirements, not a paperwork
exercise applied after the system is built. Questions like "what is this agent authorized to do
without human confirmation," "what is logged and for how long," and "how would we reconstruct and
justify a specific consequential decision six months later" are increasingly things a production
agent's architecture needs to answer by design, in the same way security and privacy requirements
became load-bearing architectural inputs for web applications over the preceding two decades rather
than a checklist applied at the end.

## 6. From Single-Agent Demos to Production Multi-Agent Systems

The shift from a single agent handling an entire task end-to-end toward multiple specialized agents
coordinating on different parts of a task is not a new idea by 2025-2026 — the architectural
patterns (orchestrator-worker, peer-to-peer collaboration, hierarchical delegation) were already
established. What changed is that these patterns moved from research demonstrations and internal
experiments into systems that handle real production load with the accountability requirements
described above attached to them, and that transition surfaced a set of problems that don't show up
in a demo.

State and memory management across a multi-agent system running continuously in production, rather
than for the duration of a single demo session, turned out to be a much harder engineering problem
than orchestrating a single conversation: which agent owns the canonical state of a long-running
task, how state is handed off cleanly when one agent delegates to another (especially across the
A2A-style organizational boundary where the receiving agent may not share the sending agent's memory
or context at all), and how failures partway through a multi-agent workflow are detected and
recovered from without silently corrupting shared state, are all questions that only become pressing
once a system runs continuously and unattended rather than for the length of one supervised demo
run.

Cost and latency accounting also becomes materially harder once a task fans out across multiple
agents, each potentially making multiple model calls and tool invocations, because the simple mental
model of "one request, one model call, one cost" no longer applies, and production operators need
per-task and per-agent cost attribution to make sensible decisions about where to invest in
efficiency, which agent in a pipeline is the bottleneck, and whether a given multi-agent
decomposition is actually worth its overhead compared to a simpler single-agent approach for the
same task — a comparison that is easy to skip in a demo (where cost is rarely measured) but
unavoidable once the same system is running thousands of times a day against a real budget.

Finally, production multi-agent systems forced a harder answer to the accountability question raised
in Section 4: when a multi-step, multi-agent pipeline produces a bad outcome, which agent's decision
was actually responsible, and is that even a well-formed question when several agents' outputs were
combined? Answering this in practice requires the kind of trajectory-level tracing described in
Section 3 applied consistently across every agent in the pipeline, not just the one that produced
the final user-facing output, which is a nontrivial requirement on system design that many earlier,
demo-stage multi-agent architectures were never built to satisfy.

## 7. Economic Interoperability: Agents That Transact

A newer and still genuinely early thread, worth knowing about even though it's less mature than
everything discussed above, is standardization work around letting agents transact economically —
making payments, purchasing services, or compensating other agents for work, autonomously, as part
of completing a task. Efforts in this space (examples include Google's Agent Payments Protocol
proposal and Coinbase's x402 protocol built around the long-dormant HTTP 402 "Payment Required"
status code) are attempting to define how an agent proves it has authorization to spend on a
principal's behalf, how a payment is requested and settled as part of an otherwise normal
agent-to-agent or agent-to-service interaction, and how spending limits and consent are enforced at
the protocol level rather than left entirely to application-level trust.

It would be overclaiming to describe this as a mature or widely adopted layer of the stack in
2025-2026 — it is closer to where MCP itself was in its first few months, an interesting proposal
with some vendor backing and no settled consensus yet on which approach, if any, becomes the
standard. It's included here because it is a logical and predictable next layer once the identity,
authorization, and audit problems from Sections 4 and 5 have even partial answers: an agent that can
be reliably identified, scoped, and audited is a prerequisite for trusting it to spend money
autonomously, so progress on economic interoperability is naturally gated by progress on the
accountability infrastructure discussed earlier in this chapter, and is worth watching as a leading
indicator of how mature that underlying infrastructure has actually become in practice.

## 8. Consolidation Pressures on the Framework Layer

A related trend worth naming: the number of agent orchestration frameworks exploded during the
early, exploratory phase of this technology, as is typical for any new category before the market
has had time to sort out which abstractions actually matter. As production requirements — the
observability, accountability, and multi-agent state management needs described above — became
clearer, the frameworks that are thriving are the ones that treated these previously-secondary
concerns as first-class citizens early, rather than the ones that optimized purely for how
impressive a five-minute demo looks. This is a reasonably standard pattern in software
infrastructure markets generally, and it suggests that evaluating an agent framework choice today
should weight its observability, permissioning, and state-management story more heavily than its raw
ease of writing a first prototype, since the latter is table stakes across essentially every option
at this point while the former is what actually determines whether a system survives contact with
production requirements.

At the protocol layer discussed in Section 2, a similar consolidation logic applies with a twist:
because MCP and A2A sit *underneath* the framework layer rather than competing with it, framework
consolidation and protocol consolidation are somewhat decoupled — a framework can lose out in the
market while the protocol it happened to support continues to thrive, because the protocol's value
comes from being framework-agnostic in the first place. This is worth being clear-eyed about when
advising a team on technology choices: betting on a specific framework carries real platform risk
given how young and fast-moving this layer still is, while betting on the interoperability protocols
underneath it carries comparatively less risk, precisely because those protocols were designed from
the outset to outlive any particular framework's popularity.

## 9. What Is Still Genuinely Unresolved

It's worth closing with an honest inventory of what remains unresolved, both because interview
conversations reward calibrated uncertainty over confident overclaiming, and because these are the
areas where real engineering and research contributions are still needed rather than areas where the
answer is already known and just needs to propagate. Fine-grained, task-scoped authorization for
autonomous agents — going beyond coarse "this agent can access this system" grants to something that
reflects the actual bounded intent of a specific delegated task — does not have a settled design
pattern yet. Trustworthy discovery of tools and agents across organizational boundaries, without a
centrally verified root of trust comparable to what domain certificate authorities provide for the
web, remains an open problem, and the registries that exist today are closer to unvetted package
indexes than to a trusted directory. Cross-organizational accountability and dispute resolution when
an autonomous multi-party interaction goes wrong has essentially no technical standard yet and is
likely to be solved, if at all, through a combination of attestation infrastructure and ordinary
contract law rather than a purely protocol-level fix. Reliable, generalizable evaluation of agent
trajectories — not just final outcomes — at the scale and cost needed for continuous production
monitoring is still an active area of methodological development, particularly for judging safety
and appropriateness of process rather than correctness of outcome. And the deeper limits on
autonomous judgment discussed in the coding agents chapter — architectural taste, understanding of
unstated organizational context, recognizing when a task requires escalation rather than independent
action — show no sign of being close to solved by scaling alone, and are likely to remain the
load-bearing argument for keeping a human in the loop at meaningful checkpoints for the foreseeable
future, regardless of how good raw task-completion metrics get.

## 10. What a Senior Engineer Should Actually Watch

Given everything above, a few concrete things are worth tracking as leading indicators rather than
trying to follow every individual product announcement. Watch whether MCP and A2A's specifications
continue to converge or formally merge some of their semantics, since that would be a strong signal
about how the interoperability layer is settling. Watch whether a genuinely trusted, curated
registry or verification layer emerges for MCP servers and A2A agent cards, since that is the single
infrastructure piece most obviously missing relative to how mature the wire protocols themselves
already are, and whoever builds it credibly will have solved a real, currently-unmet need. Watch
adoption of GenAI-specific observability semantic conventions (the OpenTelemetry extensions and
similar efforts) as a proxy for how seriously the industry is taking production-grade agent
monitoring versus treating it as an afterthought. Watch whether economic interoperability efforts
like agent payment protocols gain any real multi-party adoption, since that is a leading indicator
of how much the underlying identity and accountability infrastructure has actually matured, not just
how it's marketed. And watch, in your own organization or the organizations you interview with,
whether "human in the loop" checkpoints are placed thoughtfully at genuinely high-leverage points,
or are either absent (a red flag for an organization moving faster than its governance can support)
or present everywhere indiscriminately (a sign the team hasn't yet figured out where autonomy is
actually safe to grant) — the maturity of that judgment is one of the more reliable practical
signals of how seriously a team has engaged with the problems this chapter describes, rather than
just adopted the technology.

## 11. Summary

The period from 2025 into 2026 marks agentic AI's transition from proof-of-concept demonstrations
toward the harder, less glamorous work of making these systems reliable, observable, accountable,
and interoperable enough to run in production with real consequences attached. Interoperability
protocols (MCP and A2A) are converging at the wire-protocol layer faster than the governance
problems underneath them (fine-grained authorization, trustworthy discovery, cross-organizational
accountability) are being solved. Evaluation and observability have matured from one-off, pre-launch
checks into continuous, trajectory-aware production monitoring, borrowing structural ideas from
distributed-systems tracing but requiring genuinely new methodology for judging process and safety,
not just outcomes. Accountability infrastructure — agent identity, audit trails, governance
frameworks that explicitly contemplate autonomous action — is still early but is increasingly
treated as a first-class design requirement rather than an afterthought, and it is the necessary
foundation for the more speculative economic-interoperability efforts (agent-to-agent payments)
beginning to emerge on top of it. And the field's center of gravity has genuinely shifted from
single-agent demos toward production multi-agent systems, which surfaced real engineering problems
around state management, cost attribution, and distributed accountability that a demo never has to
confront. None of this is fully solved, and calibrated engineers should treat the remaining gaps —
authorization granularity, trust without a verified root, cross-organizational dispute resolution,
and the durable limits of autonomous judgment — as the genuinely open problems they are, rather than
assuming the field's rapid capability progress has quietly resolved them too.

