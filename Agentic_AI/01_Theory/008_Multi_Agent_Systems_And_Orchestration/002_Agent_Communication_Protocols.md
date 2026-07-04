# Agent Communication Protocols

## Why Communication Design Is Its Own Problem

Once you've decided to split work across multiple agents, a question immediately follows that is easy to underestimate: how exactly does information move between them? It is tempting to treat this as a plumbing detail — "just pass the text along" — but the choice of communication mechanism has first-order effects on cost, quality, and debuggability, in the same way the choice between a shared database and explicit message-passing APIs shapes a distributed software system. Two agents that disagree about what the other one "knows" produce confidently wrong answers, and that disagreement is almost always traceable to a sloppy communication design rather than a reasoning failure in either individual agent.

There are two broad families of approach in use today. The first is a shared message history — all participating agents read and append to one common transcript, the way participants in a group chat see every message. The second is explicit structured messages — agents exchange discrete, addressed, typed payloads, more like function calls or API requests than a conversation. Both are legitimate, and most non-trivial systems end up using each where it fits, so it's worth understanding the mechanics and failure modes of both before looking at how the industry is now trying to standardize the explicit-message style across vendors with the Agent2Agent (A2A) protocol.

## Shared Message History

The shared-history approach is the natural extension of how a single agent already works: instead of one agent talking to itself across turns, multiple agents take turns appending to the same growing transcript, and every agent's next generation is conditioned on the entire history so far, including turns produced by other agents. This is exactly the model AutoGen's `GroupChat` and LangGraph's typical `messages` state channel use — a list that every node reads in full and appends to.

```python
from typing import TypedDict, Annotated
import operator


class SharedState(TypedDict):
    # Every agent node appends to this list; nothing is ever removed,
    # so by the time the graph finishes, `messages` is the full transcript.
    messages: Annotated[list[dict], operator.add]
    task: str


def researcher_node(state: SharedState) -> dict:
    transcript = "\n".join(f"{m['agent']}: {m['content']}" for m in state["messages"])
    finding = llm.invoke(f"Task: {state['task']}\nSo far:\n{transcript}\nAdd your research.")
    return {"messages": [{"agent": "researcher", "content": finding}]}


def writer_node(state: SharedState) -> dict:
    transcript = "\n".join(f"{m['agent']}: {m['content']}" for m in state["messages"])
    draft = llm.invoke(f"Task: {state['task']}\nSo far:\n{transcript}\nWrite a draft.")
    return {"messages": [{"agent": "writer", "content": draft}]}
```

The appeal is simplicity and transparency: every agent has full visibility into everything that has happened, there is no risk of an agent acting on stale or incomplete information because it missed a message that was sent to someone else, and debugging is just reading one linear log top to bottom. This makes shared history an excellent fit for small groups of agents engaged in genuinely collaborative reasoning — a debate, a draft/critique/revise loop, a brainstorming session — where every participant benefits from seeing the full context of what the others have said.

It has two costs that grow with scale. The first is the same context-window pressure discussed for single agents, just relocated: as the transcript grows across many turns and many agents, every subsequent LLM call has to pay to re-read (and pay for, in tokens) the entire history, even the parts irrelevant to its current turn. A five-agent, twenty-round debate can produce a transcript that dwarfs the size of the actual task. The second is ambiguity of addressing — when everyone sees everything, "who was that message actually meant for, and does it need a response from anyone in particular" becomes implicit rather than explicit, and orchestration logic often ends up doing its own ad-hoc parsing of the transcript to figure out whose turn is next or which message is actionable. Shared history works best when the group is small, the number of rounds is bounded, and every participant genuinely benefits from seeing everything.

## Explicit Structured Messages

The alternative is to treat inter-agent communication like an API: every message has an explicit sender, an explicit receiver, a typed purpose, and a payload, and an agent only ever sees the messages addressed to it, not a global transcript. This is closer to how real distributed systems and multi-agent research (FIPA-ACL performatives, contract-net protocols) have always approached agent communication, and it scales far better to larger agent populations because each agent's input is bounded by what's relevant to it, not by the size of the whole system's conversation.

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: MessageType
    content: dict
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reply_to: str | None = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class MessageBus:
    """A minimal point-to-point router: each agent only ever sees
    messages explicitly addressed to it."""

    def __init__(self):
        self.queues: dict[str, list[AgentMessage]] = {}
        self.handlers: dict[str, callable] = {}

    def register(self, agent_name: str, handler):
        self.queues[agent_name] = []
        self.handlers[agent_name] = handler

    def send(self, message: AgentMessage):
        self.queues.setdefault(message.receiver, []).append(message)

    def drain(self, agent_name: str):
        queue = self.queues.get(agent_name, [])
        while queue:
            message = queue.pop(0)
            reply = self.handlers[agent_name](message)
            if reply is not None:
                self.send(reply)
```

The benefit is precise scoping: an agent's context contains exactly the messages relevant to its own work, so context size scales with the number of messages actually addressed to that agent rather than with total system activity. It also makes intent explicit and machine-checkable — a `REQUEST` message expects a `RESPONSE`, a `PROPOSE` expects an `ACCEPT` or `REJECT`, and orchestration code can enforce these expectations programmatically instead of relying on an LLM to infer conversational structure from a wall of text. This is why explicit messaging is the right choice as agent populations grow, as message volume grows, or as agents are built and operated by different teams (or different companies) that shouldn't need to parse each other's internal reasoning to collaborate.

The cost is the infrastructure itself: you have to build (or adopt) a message bus, define message schemas, handle routing, and decide what happens to a message sent to an agent that never replies. It is also less naturally suited to freeform collaborative reasoning — forcing a genuine back-and-forth debate into discrete typed messages can feel more rigid than letting it play out in a shared transcript. The practical guidance mirrors the shared-history section's conclusion in reverse: use explicit structured messages once the agent population, message volume, or organizational boundary between agents makes a global shared transcript unwieldy or inappropriate — and this is precisely the situation the Agent2Agent protocol was designed for at industry scale.

### Delivery And Ordering Guarantees Matter More Than They Look

A detail that's easy to skip past in a toy implementation but that bites hard in production is what guarantees your message bus actually provides. The `MessageBus` above is synchronous and in-process: `send` immediately enqueues, `drain` processes messages in strict FIFO order, and nothing is lost as long as the process doesn't crash mid-drain. The moment you move to a real distributed deployment — agents running as separate services, possibly on separate machines, talking over a real queue (Kafka, RabbitMQ, SQS) — you inherit that queue's actual guarantees, and they are rarely "exactly once, in order, always delivered" by default. Most production queues offer at-least-once delivery, which means your message handlers need to be idempotent (processing the same `REQUEST` twice should not, for example, double-charge a downstream API or double-append to a shared list), and many offer no strict ordering guarantee across partitions, which means a `RESPONSE` can in principle arrive before the `REQUEST` it responds to is even considered "sent" from the sender's own bookkeeping, if you're not careful about how you track conversation state. The `conversation_id` field on `AgentMessage` exists specifically to make this tractable: instead of relying on delivery order, an agent can buffer messages by conversation ID and only act once it has everything a given step actually needs, which is a strictly safer pattern than assuming "message N always arrives before message N+1."

### A Hybrid Pattern: Structured Envelope, Shared Payload

In practice, many production systems land on a hybrid that borrows from both families: messages are explicitly addressed and typed (so routing and intent are unambiguous, as in the explicit-message pattern), but the payload of a message is allowed to be a pointer into a larger shared context object — a blackboard, a shared document store, or a vector store — rather than the full content being duplicated into every message. This keeps message envelopes small and precisely scoped while still letting agents access a rich shared context when they need to, without paying the "everyone re-reads everything" tax of a pure shared-transcript design.

```python
@dataclass
class ScopedMessage:
    sender: str
    receiver: str
    msg_type: MessageType
    summary: str          # small, always included
    context_ref: str      # a key into a shared store, fetched only if needed


class ScopedMessageBus(MessageBus):
    def __init__(self, shared_store: dict):
        super().__init__()
        self.shared_store = shared_store

    def deliver(self, message: ScopedMessage):
        # The handler decides whether it actually needs the full context;
        # cheap routing/triage can happen from `summary` alone.
        full_context = self.shared_store.get(message.context_ref)
        return self.handlers[message.receiver](message, full_context)
```

This pattern is worth reaching for once you notice that your explicit messages are ballooning in size because every message carries a full copy of some large shared artifact (a research document, a codebase diff) — split the routing/intent metadata (small, always read) from the bulk payload (large, fetched on demand), and you get the addressing clarity of explicit messages without paying to serialize the same large payload into every message that references it.

## The Agent2Agent (A2A) Protocol

Both patterns above assume you control every agent in the system, or at least that they're built on a shared internal framework. That assumption breaks down the moment you want your agent to delegate to another organization's agent — a travel-booking agent built by one vendor calling a hotel-search agent built by another, or an enterprise orchestrator calling out to a specialized legal-research agent it doesn't own and can't inspect. Two agents built on different stacks (one on LangGraph, one on a proprietary framework, one exposed only as a hosted API) have no shared message-bus code and no shared internal state format to piggyback on. This is the interoperability gap A2A (originally introduced by Google, now under the Linux Foundation as an open standard with broad industry backing) is designed to close: a vendor-neutral, transport-level protocol for one agent to discover, invoke, and hold a multi-turn task-oriented conversation with another agent, treating the other agent as an opaque black box rather than something whose internals you need to understand.

The core idea is that an A2A-compliant agent publishes an **Agent Card** — a small JSON document, typically served at a well-known URL, describing what the agent can do, how to authenticate to it, and what transport it supports.

```python
agent_card = {
    "name": "Hotel Search Agent",
    "description": "Finds and compares hotel availability and pricing.",
    "url": "https://hotels.example.com/a2a",
    "version": "1.2.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": True,
    },
    "authentication": {
        "schemes": ["bearer"],
    },
    "skills": [
        {
            "id": "search_hotels",
            "name": "Search Hotels",
            "description": "Search hotels by city, date range, and budget.",
            "inputModes": ["text", "application/json"],
            "outputModes": ["text", "application/json"],
        }
    ],
}
```

A calling agent — the "client" in A2A terms — fetches this card, learns what the remote agent can do and how to authenticate, and then submits a **task**: a unit of work identified by a task ID, which carries an ongoing lifecycle rather than being a single stateless request/response pair. A task moves through explicit states (`submitted`, `working`, `input-required`, `completed`, `failed`, `canceled`), which matters a great deal for agentic work specifically, because many real agent tasks are not instantaneous — they may need clarifying input partway through, may take minutes to complete, and may need to stream partial progress back to the caller.

```python
import requests


class A2AClient:
    """A minimal illustrative client for calling a remote A2A agent.
    Real implementations use the official A2A SDKs; this shows the
    conceptual shape of the interaction."""

    def __init__(self, agent_card_url: str, bearer_token: str):
        self.card = requests.get(agent_card_url).json()
        self.endpoint = self.card["url"]
        self.headers = {"Authorization": f"Bearer {bearer_token}"}

    def send_task(self, skill_id: str, message: str) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": "task-001",
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "skill": skill_id,
            },
        }
        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        return response.json()  # contains task status + any returned artifacts

    def poll_task(self, task_id: str) -> dict:
        payload = {"jsonrpc": "2.0", "method": "tasks/get", "params": {"id": task_id}}
        response = requests.post(self.endpoint, json=payload, headers=self.headers)
        return response.json()
```

Under the hood, A2A runs over standard HTTP with JSON-RPC 2.0 as the message envelope, supports Server-Sent Events for streaming partial results as a long-running task progresses, and supports webhook-style push notifications so a client doesn't have to poll a slow task. The payload itself is built from a small set of primitives: **messages** (turns in the conversation, made of typed **parts** — text, files, or structured data), **artifacts** (the actual output the task produces, which can itself be streamed incrementally), and the **task** object that ties a conversation's messages and artifacts together under one lifecycle. Critically, A2A deliberately does not standardize what happens *inside* the remote agent — it doesn't care whether the hotel-search agent is built on LangGraph, a hand-rolled state machine, or a single prompt-and-tool-call loop. It only standardizes the boundary: how you discover it, how you authenticate to it, how you send it a task, and how you get results back, whether that's one shot or a long-running multi-turn negotiation.

## How A2A Complements MCP Rather Than Competing With It

It's easy to conflate A2A with the Model Context Protocol (MCP) because both are JSON-RPC-based standards that showed up around the same time in the agent ecosystem, but they solve different layers of the same problem and are meant to be used together, not as alternatives. MCP standardizes the boundary between an agent and its **tools and data sources** — a database connector, a file system, a search API, a set of internal enterprise functions — exposing them as a discoverable set of tools, resources, and prompts that any MCP-compliant client can call. The relationship in MCP is fundamentally asymmetric: one side is an agent (or a host application driving an agent) and the other side is a tool or a data source with no agency of its own — it doesn't reason, plan, or hold a multi-turn negotiation, it just executes the function it's told to and returns a result.

A2A standardizes the boundary between one **agent and another agent** — both sides are autonomous, both sides can reason, plan, and potentially say "I need more information before I can finish this" or "here is partial progress, more is coming." That asymmetry (or lack of it) is the crux of the distinction: MCP is a client-to-resource protocol, A2A is a peer-to-peer protocol between two systems that are both, in principle, capable of independent decision-making. A concrete way to see this: the hotel-search agent behind the A2A endpoint above almost certainly uses MCP internally to call a hotel-inventory database and a pricing API — those are tools, so MCP is the right fit there. But when a trip-planning orchestrator wants to delegate the entire "find me hotels" sub-problem to that hotel-search agent, without knowing or caring how it's implemented internally, A2A is the right fit for that outer boundary, because the orchestrator isn't calling a single deterministic function, it's delegating a task to another reasoning system.

This layering has a direct practical consequence for how you design a multi-agent system that has to interoperate outside your own organization: use MCP to wire your own agents up to the tools, databases, and APIs they need, and use A2A (or an equivalent explicit-message, task-lifecycle protocol) at the boundary where you hand off work to an agent you don't control and shouldn't need to inspect. Inside a single team's system, either the shared-history or the explicit-message pattern discussed earlier is usually simpler than adopting a full standard, since you can rely on shared code and shared assumptions. A2A earns its complexity specifically at trust and vendor boundaries — cross-company agent marketplaces, enterprise agents built by different internal teams on different stacks, or any scenario where "I need to call an agent I don't own and can't modify" is a first-class requirement rather than a hypothetical.

### Streaming And Long-Running Tasks

A meaningful fraction of real agent-to-agent tasks are not fast — a research agent might take minutes to produce a thorough answer, and a caller shouldn't have to block on a single synchronous HTTP call for that whole duration, nor should it be left guessing whether the remote agent is still alive. A2A addresses this with Server-Sent Events: instead of a single request/response, the client opens a stream and receives incremental task-status and artifact-chunk events as the remote agent makes progress, which lets a calling orchestrator surface partial progress to its own end users (e.g., "researching..." then "drafting..." then "done") rather than showing a blank loading state for the task's full duration.

```python
def stream_task(self, skill_id: str, message: str):
    """Illustrative streaming client: yields incremental events instead
    of waiting for one final response."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/sendSubscribe",
        "params": {
            "id": "task-002",
            "message": {"role": "user", "parts": [{"type": "text", "text": message}]},
            "skill": skill_id,
        },
    }
    with requests.post(self.endpoint, json=payload, headers=self.headers, stream=True) as response:
        for line in response.iter_lines():
            if not line:
                continue
            event = json.loads(line.decode().removeprefix("data: "))
            yield event  # {"status": "working"|"input-required"|"completed", "artifact": {...}}
```

For tasks that take even longer — hours, not seconds — polling or holding a stream open is impractical, which is why A2A also supports push notifications: the client registers a webhook URL when it submits the task, and the remote agent calls that webhook when the task's state changes, rather than requiring either side to keep a connection open. Choosing between synchronous request/response, streaming, and push notifications is a real design decision that should be driven by expected task duration: sub-second-to-a-few-seconds tasks are fine synchronous, tasks in the seconds-to-low-minutes range benefit from streaming so a caller can show progress, and tasks that can run for a long time in the background are better served by push notifications so neither side has to hold a connection or poll.

### A Quick Checklist For Choosing Between The Patterns

Given a specific system you're designing, it helps to have a short, concrete checklist rather than re-deriving the trade-offs from scratch each time. Reach for **shared message history** when: the agent count is small (roughly two to five), every participant genuinely benefits from seeing everything the others have said, the interaction is bounded in rounds, and you control the framework on both sides. Reach for **explicit structured messages** when: the agent count is larger, message volume is high enough that a global transcript would dominate every agent's context budget, different agents need different, non-overlapping slices of information, or you need machine-enforceable protocol structure (a request must get a matching response) rather than relying on an LLM to infer it from prose. Reach for **A2A** (or an equivalent task-lifecycle protocol) specifically at trust or vendor boundaries — when the other agent is built, deployed, and operated by a different team or company, when you need it to remain an upgradeable black box rather than something you inspect, or when the interaction is long-running enough to need real task-state semantics (submitted, working, input-required, completed) rather than a single request/response pair.

## Production Considerations

A few things matter in practice once any of these communication mechanisms leave a notebook and enter a real system. Authentication and authorization need to be explicit at every hop — a shared message bus inside one trusted process can get away with implicit trust, but the moment you cross an A2A boundary to a third party, every task submission needs a real auth scheme (the agent card's `authentication` field exists for exactly this reason), and you should assume the remote agent might misbehave, time out, or return malformed output, so client code needs the same defensive timeout and validation handling you'd apply to any external API call. Versioning matters too: agent cards carry a version field because a remote agent's skills and input/output shapes will change over time, and a caller that hard-codes assumptions about a skill's parameters will break silently when the remote agent is upgraded — treat an agent card the way you'd treat any external API's OpenAPI spec, checked at integration time rather than assumed forever. Finally, observability across a communication boundary is harder than within one process: for shared-history and explicit-message systems you own, you can log the full transcript or message bus traffic; for A2A calls to agents you don't own, you typically only see the task lifecycle events and whatever artifacts come back, so instrument the client side (latency, failure rate, retry counts per remote agent) since you won't get visibility into what happened on the other side of the boundary.
