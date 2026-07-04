# MCP and A2A: The Interoperability Ecosystem

## Table of Contents

1. The Integration Problem That Standards Are Solving
2. From Proprietary Function Calling to Open Protocols
3. MCP in the Ecosystem: A Recap From 30,000 Feet
4. Agent2Agent (A2A): Standardizing Agent-to-Agent Communication
5. How MCP and A2A Relate to Each Other
6. The Adoption Landscape: Clients, Servers, and Registries
7. Enterprise Gateways and the Rise of "MCP Infrastructure"
8. What Remains Unsolved: Auth, Discovery, and Trust
9. Where a Senior Engineer Should Focus
10. Summary

---

## 1. The Integration Problem That Standards Are Solving

Every agentic system eventually runs into the same wall: an agent is only as useful as the systems
it can reach, and every new system it needs to reach requires bespoke integration work. Before any
standard existed, if you wanted an LLM-based agent to read from Jira, write to Salesforce, query a
internal data warehouse, and browse a filesystem, you wrote four different adapters, each
hand-rolled to translate that system's API into something you could describe to a model as a "tool."
Multiply this by the number of agent frameworks in your organization (maybe you have one team on
LangGraph, another building a custom loop, a third using a vendor's agent platform) and you get a
combinatorial explosion: every agent framework needs its own adapter to every backend system. This
is the classic N×M integration problem, the same shape of problem that motivated ODBC for databases,
HTTP for networked services, and USB for peripherals. The lesson from all of those precedents is
consistent: N×M problems get solved by inserting a standard interface in the middle, turning the
problem into N+M — every tool provider implements the standard once, every agent framework
implements the standard once, and the two sides no longer need to know about each other's internals.

That is the core motivation behind the current wave of agentic interoperability protocols. Model
Context Protocol (MCP), introduced by Anthropic in late 2024, standardizes the *agent-to-tool* side
of this problem: how an agent discovers, describes, and invokes external capabilities (tools,
resources, prompts) regardless of which model or framework is driving the agent. Agent2Agent (A2A),
introduced by Google shortly after and subsequently moved to the Linux Foundation for vendor-neutral
stewardship, standardizes the *agent-to-agent* side: how one autonomous agent can discover another
agent (possibly built by a different vendor, running in a different organization, backed by a
different model) and delegate work to it as a peer rather than as a tool. Together, these two
protocols are the closest thing the field has to a shared substrate for "agentic web"
interoperability, in the same way HTTP and REST became the shared substrate for the web of services.

It's worth being precise about why this matters for a senior engineer specifically, beyond the
abstract elegance of standardization. Protocol adoption changes where the leverage in the system
sits. When every integration is bespoke, the differentiator is often "who built the most
integrations," which favors platforms with the largest engineering teams. When integrations are
standardized, the differentiator shifts to "who has the best agent reasoning, the best tool
selection, the best orchestration" — the model and the agent logic — because the plumbing becomes
commoditized. This is analogous to what happened when cloud providers standardized around
S3-compatible object storage APIs: the storage layer became commoditized, and competition moved up
the stack. Recognizing this shift matters when you're deciding where to invest engineering effort:
on custom connectors that a standard might obsolete in a year, or on agent capabilities that remain
valuable regardless of which protocol wins.

## 2. From Proprietary Function Calling to Open Protocols

Before MCP, the dominant pattern for giving an LLM access to external capabilities was function
calling (also called tool calling): a vendor-specific API where you pass the model a JSON Schema
describing available functions, the model responds with a structured request to invoke one of them,
and your application code executes that function and feeds the result back into the conversation.
OpenAI popularized this pattern in mid-2023, and every other major model provider (Anthropic,
Google, Mistral, open-weight model serving stacks) converged on functionally similar but
syntactically distinct versions of it. This was a genuine advance over parsing free-text "I will now
call get_weather(city)" style outputs, but it solved only half the problem. Function calling
standardized *how a single model call describes and requests tool use*; it said nothing about *where
the definitions of those tools come from*, *how they are packaged and distributed*, *how a running
tool process is discovered and connected to*, or *how permissions and authentication are managed*
across many tools from many different providers.

In practice, before MCP, every team building an agent wrote its own glue code to take some tool
implementation (a Python function, a REST wrapper, a database query) and turn it into the JSON
Schema shape that a particular model's function-calling API expected, then wrote separate connection
and lifecycle management for however that tool was hosted. If your organization had thirty internal
tools and used three different agent frameworks, that glue code was written and re-written many
times over, and there was no way to take a tool built for one framework and drop it into another
without rewriting the adapter. The function-calling schema was a contract between an application and
a model; it was never meant to be a contract between independently-developed tool servers and
independently-developed agent clients across organizational boundaries. That is precisely the gap
MCP was designed to close: instead of each agent framework defining its own plugin interface, MCP
defines a transport-and-message-format contract that any tool provider can implement once, and any
agent client can consume, without either side depending on the other's internal architecture.

It is important not to overstate this as a replacement for function calling — it's a layer built on
top of it. When an MCP client connects to an MCP server and lists its tools, the client still
ultimately hands the model a function-calling-style schema so the model can decide when to invoke
it; MCP standardizes how that schema and the corresponding invocation logic are packaged,
discovered, and executed outside the model call itself. The protocol mechanics of MCP (the JSON-RPC
message format, the session lifecycle, resources versus tools versus prompts, the transport options)
are covered in depth in the companion file
`007_Tool_Use_Function_Calling_And_MCP/003_Model_Context_Protocol_Deep_Dive.md`; this chapter
deliberately stays above that layer and focuses on the ecosystem question — why the protocol exists,
who has adopted it, and what interoperability problems still don't have good answers.

## 3. MCP in the Ecosystem: A Recap From 30,000 Feet

At the ecosystem level, the important thing to understand about MCP is not its message format but
its role as a distribution mechanism. An MCP server is a self-contained, independently deployable
unit of capability: a small program that speaks the protocol and exposes some set of tools,
resources, or prompt templates. Because the server doesn't need to know anything about which agent
or model will eventually connect to it, a single MCP server implementation can be reused across
every agent product that speaks MCP. This is what turned MCP from "a nice interface" into "an
ecosystem": within roughly a year of its release, thousands of community-built and vendor-built MCP
servers appeared covering version control platforms, project management tools, cloud provider APIs,
databases, browser automation, and internal enterprise systems, because building one server unlocked
compatibility with every MCP-speaking client rather than with a single product.

The other ecosystem effect worth naming is what MCP did to the largest model vendors' incentives.
Anthropic created MCP, but its real test of "standard" status was whether competitors would adopt it
despite it originating from a competitor. Through 2025, OpenAI, Google DeepMind, and Microsoft all
added MCP client and/or server support to their own agent products and SDKs, and major agent
frameworks (LangChain/LangGraph, LlamaIndex, Semantic Kernel, and others) added first-class MCP
client support. That kind of cross-vendor adoption of a competitor-originated standard is unusual in
this industry and is the strongest evidence that MCP solved a real, shared pain point rather than a
proprietary one. It does not mean MCP "won" permanently or that no successor will emerge, but as of
this writing it is the closest thing to a lingua franca for the agent-to-tool boundary.

## 4. Agent2Agent (A2A): Standardizing Agent-to-Agent Communication

A2A addresses a distinct problem that MCP was never designed to solve: how does an autonomous agent
delegate a task to *another autonomous agent*, one that might be opaque, long-running, stateful, and
built by a completely different organization? The distinction matters more than it first appears. A
tool, in the MCP sense, is typically stateless or lightly stateful, synchronous or short-lived, and
fully specified: you call `search_documents(query)` and you get a result back in the same
request-response cycle, and the tool has no independent goals or planning of its own. An agent, in
the A2A sense, might take minutes or hours to complete a task, might need to ask clarifying
questions mid-task, might invoke its own sub-agents and tools internally that the caller should
never see, and critically, might not want to expose its internal reasoning, its private data
sources, or its underlying model to the calling party. Treating a partner agent as if it were just
another tool discards exactly the properties that make it an agent — autonomy, statefulness over
long horizons, and encapsulation of its own internal complexity.

A2A's design reflects this. Agents publish an "Agent Card," a machine-readable description of the
agent's identity, the skills it offers, and how to authenticate to it, roughly analogous to a
service's OpenAPI document but oriented around capabilities rather than endpoints. A calling agent
(the client in an A2A exchange) discovers a remote agent's card, then creates a "task" with that
agent, which the remote agent works on asynchronously; the protocol defines how task state is
queried, how streaming updates are delivered, how artifacts (files, structured outputs) are
returned, and how the two agents can exchange further messages if the task requires clarification.
Crucially, the remote agent's internal implementation — which model it uses, what tools it calls
internally, how it's orchestrated — is opaque to the caller. This opacity is a feature: it means a
healthcare provider's scheduling agent and a logistics company's delivery-tracking agent can
interoperate over A2A without either party needing to know, or trust, anything about the other's
internal architecture, only that it correctly implements the protocol's task lifecycle.

Google originated A2A in 2025 with backing from a large number of technology and consulting
partners, and subsequently contributed it to the Linux Foundation, mirroring the governance path
that gave protocols like Kubernetes broad, vendor-neutral credibility. That governance move is
significant for adoption: enterprises are historically wary of building critical infrastructure on a
protocol unilaterally controlled by one commercial vendor, especially a vendor that also competes in
the same market. Neutral governance lowers that barrier, though it is not a guarantee of long-term
adoption by itself — plenty of neutrally-governed standards have failed to gain traction, and it is
still early days for A2A relative to MCP.

## 5. How MCP and A2A Relate to Each Other

A natural question is whether MCP and A2A are competitors or complements, and the honest answer is
that they address different layers of the same stack, though the boundary blurs in practice. A
useful mental model: MCP standardizes the vertical relationship between an agent and the
deterministic capabilities beneath it (databases, APIs, files, search indexes), while A2A
standardizes the horizontal relationship between peer agents that each have their own reasoning
loop. A single system commonly uses both: a customer-support orchestrator agent might use A2A to
delegate a billing dispute to a specialized billing agent (a peer, opaque, potentially built by
another team or vendor), while that billing agent internally uses MCP to query the billing database
and the payments API (tools, transparent, fully specified).

In practice, the line is not always clean, and this is a legitimate source of design confusion in
2025-era systems. An agent exposed over A2A can itself be wrapped and exposed as an MCP tool to a
simpler caller that just wants a single function-call interface and doesn't care about the richer
task lifecycle A2A offers; conversely, some MCP servers have grown facilities for long-running,
asynchronous operations that start to resemble what A2A tasks provide natively. Expect continued
convergence pressure here — either through explicit bridging (gateways that translate between the
two) or through the protocols themselves absorbing pieces of each other's semantics over subsequent
versions. For system design purposes today, the pragmatic rule of thumb is: if the remote party is a
fixed capability you fully control the interface to and expect a fast, bounded response from, model
it as an MCP tool; if the remote party has its own autonomy, might take a long time, and you want to
preserve its right to keep its internals private, model it as an A2A peer.

## 6. The Adoption Landscape: Clients, Servers, and Registries

On the client side, "speaks MCP" has become close to a checkbox feature for agent products and IDEs
released since 2025: coding assistants, general-purpose chat products with agent modes, and
orchestration frameworks generally ship an MCP client, meaning they can be pointed at an MCP
server's connection details and immediately gain access to whatever tools that server exposes. This
is the visible payoff of the standard from a user's perspective — install one server configuration
and multiple different agent products can use it without separate integration work per product.

On the server side, the ecosystem split into three overlapping populations. First, official or
vendor-maintained servers published by the companies that own the underlying system — a cloud
provider publishing an MCP server for its own infrastructure APIs, a SaaS company publishing one for
its own product. Second, community-built servers, often thin wrappers around a public API, published
as open source with no formal support commitment; these vary enormously in quality, are the largest
population by count, and are also the largest source of the trust and security concerns discussed
later in this chapter. Third, internal/enterprise servers that organizations build for their own
proprietary systems and never publish externally at all — arguably the most economically significant
category even though it's the least visible, since it is what actually connects agents to a
company's own data warehouse, ticketing system, or deployment pipeline.

This proliferation created demand for discovery infrastructure, and a handful of registries and
directories emerged to catalog available servers, alongside a semi-formal community registry effort
intended to become a canonical index. None of these registries functions like a fully trusted,
centrally-curated app store yet — most are closer to a package index (comparable to npm or PyPI in
its early years) than to a vetted marketplace, which is precisely the trust gap discussed in Section
8. There is also a parallel, IDE-centric distribution channel: many coding tools bundle a curated,
small set of "recommended" MCP servers directly in their settings UI, which for most engineers is
currently a more meaningful discovery path than browsing an external registry.

## 7. Enterprise Gateways and the Rise of "MCP Infrastructure"

Once organizations started connecting many MCP servers to many internal agents, a predictable
second-order need appeared: infrastructure to manage the connections themselves, separate from the
tools they expose. This gave rise to what's often called an MCP gateway or MCP proxy pattern — a
single internal service that agents connect to, which in turn fans out to the actual downstream MCP
servers. The gateway centralizes exactly the concerns that are awkward to solve per-server:
authentication and credential injection (so individual agents never hold raw API keys for every
downstream system), audit logging of every tool call for compliance, rate limiting and cost
attribution per team or per agent, and the ability to allow-list or block specific tools or servers
organization-wide without touching every agent's configuration. Several API management vendors and
cloud providers moved quickly to offer this as a managed capability, treating it as a natural
extension of existing API gateway products rather than something wholly new, which suggests the
enterprise pattern here will likely converge on "MCP gateway" being a standard line item in an API
platform's product suite rather than a bespoke build for most companies.

A related trend is remote-hosted MCP servers reachable over HTTP (as opposed to the local,
stdio-launched servers that dominated MCP's earliest usage pattern), which matters because it's what
makes centrally-governed, multi-tenant MCP infrastructure practical in the first place — a stdio
server spawned as a local subprocess is inherently single-user and hard to govern centrally, while
an HTTP-reachable server can sit behind the same authentication, logging, and rate-limiting layers
as any other internal web service.

## 8. What Remains Unsolved: Auth, Discovery, and Trust

It's tempting, looking at the adoption numbers, to conclude the interoperability problem is
basically solved. It is not; the core message-passing mechanics are solved, but several problems
that only become visible at scale, across organizational boundaries, remain genuinely open.

**Authentication and authorization across trust boundaries.** MCP's authorization model has matured
to align with standard OAuth 2.1 patterns, which is a real improvement over the protocol's early
days when auth was largely left as an exercise for each server implementer. But OAuth answers "can
this client obtain a token to call this server," not the harder question of "what should this agent
be allowed to do with that token once it's mid-task, acting semi-autonomously, on behalf of a human
who granted broad consent once." Most real-world MCP deployments today grant tools access at a
coarse scope (this agent can read and write to this system) because building fine-grained,
task-aware, revocable permission scopes for autonomous agents is still an open design problem, not a
solved one. The A2A side has an analogous unsolved problem: an Agent Card can declare what
authentication scheme a remote agent expects, but establishing that two organizations' agents should
trust each other with a given task, at a given data sensitivity level, is closer to a legal and
business-process problem than a cryptographic one, and the protocol layer alone cannot resolve it.

**Discovery without a trusted root.** There is no equivalent yet of a domain name system or a
certificate authority hierarchy for agentic capability discovery — no single, universally trusted
way to answer "is this the real, legitimate MCP server for this company's product, or an impostor
with a similar name and a malicious tool description." The registries that exist today provide some
curation, but nothing close to the verification guarantees users unconsciously rely on when they see
a padlock icon in a browser. This is not a hypothetical risk: a malicious or compromised MCP server
can describe its tools in a way that manipulates the calling model (a form of prompt injection
embedded in tool metadata rather than in user input), and an agent that blindly trusts tool
descriptions from an unverified registry is exposed to exactly this class of attack. The same
concern applies to A2A Agent Cards — nothing in the base protocol prevents a card from
misrepresenting the skills or trustworthiness of the agent behind it.

**Cross-organizational accountability.** When an agent in Company A delegates a task to an agent in
Company B over A2A, and something goes wrong (bad data returned, an action taken that shouldn't have
been, a cost incurred that wasn't authorized), the protocols themselves are silent on liability,
dispute resolution, or even on producing a shared, tamper-evident audit trail that both parties
agree reflects what happened. This is squarely a governance and contractual problem, not a
wire-protocol problem, but it is a real blocker to using these protocols for anything beyond
low-stakes, single-organization use cases today, and it is one of the reasons most production A2A
usage in 2025-2026 remains intra-organizational (different teams' agents talking to each other
inside one company) rather than the cross-company "agent economy" vision the protocol's marketing
often gestures toward.

**Semantic interoperability, not just syntactic.** Both protocols solve the syntactic problem — the
message format, the transport, the session lifecycle — but neither solves the deeper problem that
two independently-built systems can use the same field names to mean subtly different things, or
that a tool's natural-language description can be interpreted differently by different models with
different training. Standardizing the pipe does not standardize what flows through it; a
`create_ticket` tool from one vendor and a `create_ticket` tool from another may have incompatible
assumptions about required fields, idempotency, or side effects, and an agent composing calls across
both is exposed to that mismatch with no protocol-level safety net.

**Versioning and long-term compatibility.** As MCP and A2A both continue to evolve their
specifications, the ecosystem faces the same challenge every open protocol eventually faces: how to
evolve without breaking the large number of already-deployed servers and clients. MCP has begun
addressing this through explicit protocol version negotiation during the initial handshake, but the
practical burden of keeping a large population of independently-maintained community servers
compatible with a moving specification is still shifting onto individual maintainers rather than
being solved structurally.

## 9. Where a Senior Engineer Should Focus

Given this state of affairs, a few practical implications are worth internalizing rather than
treating this as pure trivia. First, treat the choice of "build vs. adopt" for internal tool
interfaces as effectively decided in favor of adopting MCP for any new tool integration work — the
ecosystem gravity is strong enough that a bespoke internal plugin format built today is choosing to
swim against the current, with no offsetting benefit unless you have a genuinely unusual latency or
transport requirement. Second, if you're integrating third-party MCP servers you don't control,
treat every tool description as untrusted input, exactly as you would treat any other content that
reaches your model's context window — apply the same prompt-injection defenses (sanitization,
least-privilege scoping, human confirmation for high-impact actions) you would apply to any external
content. Third, for cross-organization agent collaboration, expect the interesting engineering work
for the next several years to be in the governance layer wrapped around A2A — identity, contracts,
audit — rather than in the protocol's wire format itself, which is comparatively mature already.
Finally, keep an eye on convergence: it would not be surprising if, over the next couple of major
revisions, the practical distinction between "call a tool" and "delegate to an agent" becomes
something the protocols handle more uniformly, with today's sharp MCP/A2A split remembered as an
artifact of how the ecosystem happened to bootstrap itself in 2024-2025 rather than a permanent
architectural boundary.

## 10. Summary

MCP and A2A exist because agentic systems ran headlong into the same N×M integration problem every
prior generation of distributed systems has faced, and the industry responded the way it usually
does: by standardizing the interface layer so that tool providers and agent builders can innovate
independently. MCP standardizes the agent-to-tool boundary and has achieved unusually broad
cross-vendor adoption for a protocol that originated from a single company; A2A standardizes the
agent-to-agent boundary, preserving the autonomy and opacity that make an "agent" different from a
"tool," and has taken the neutral-governance path to try to earn similarly broad trust. Both
protocols have solved the mechanical, syntactic parts of interoperability convincingly. Neither has
solved the harder problems of authentication at the granularity autonomous action requires,
trustworthy discovery in the absence of a verified root of trust, or accountability when things go
wrong across organizational lines — and those gaps, not the wire format, are where the real
engineering and governance work of the next few years will happen.

