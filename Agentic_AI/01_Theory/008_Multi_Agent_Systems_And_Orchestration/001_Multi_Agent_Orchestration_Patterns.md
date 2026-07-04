# Multi-Agent Orchestration Patterns

## Why Split Work Across Multiple Agents At All

A single LLM agent with a big system prompt, a bag of tools, and a reasoning loop can go surprisingly far. Before reaching for a multi-agent design, it is worth being honest about why a single agent stops being sufficient, because every additional agent you introduce adds coordination cost, latency, and a new class of failure modes. The decision to go multi-agent should be a deliberate trade, not a default architectural choice made because it looks sophisticated in a diagram.

The first real limit is context. A single agent's context window has to hold the system prompt, the tool definitions, the conversation history, and any retrieved documents or intermediate scratch work. As a task grows — say, "audit this codebase for security issues, then refactor the worst offenders, then write tests, then update the docs" — the accumulated transcript from earlier phases starts crowding out the budget available for the current phase, and the model's attention gets diluted across an increasingly irrelevant history. Splitting the work across agents lets each one carry only the context relevant to its slice of the problem, discarding or summarizing the rest.

The second limit is specialization and prompt coherence. A single system prompt that tries to make one model behave as a meticulous security reviewer, a pragmatic refactoring engineer, a thorough test writer, and a concise technical writer all at once produces a mushy compromise — the model has to hold multiple, sometimes conflicting, behavioral instructions in mind simultaneously, and instruction-following quality degrades as the prompt tries to do more things. Giving each role its own agent, with a tightly scoped system prompt, tool list, and set of behavioral constraints, produces sharper, more reliable behavior in each role, in the same way a human team of specialists usually outperforms one generalist trying to do everything.

The third limit is parallelism and latency. If subtasks are genuinely independent — researching three unrelated topics, or reviewing five files for different concerns — a single sequential agent processes them one after another, and wall-clock time scales linearly with the number of subtasks. Multiple agents running concurrently collapse that into roughly the time of the slowest branch, which matters enormously in interactive or time-boxed products.

The fourth limit is fault isolation and quality through independent perspectives. When one agent does everything, a single bad reasoning step early in the trajectory can silently corrupt everything downstream, because there is no independent check. Splitting responsibilities across agents creates natural checkpoints: a reviewer agent that never saw the writer's internal reasoning is more likely to catch an error than the writer re-reading its own output, because it is not anchored to its own prior conclusions. This is the same reason human editorial workflows separate the writer from the editor.

None of this is free. Every agent boundary you introduce means an extra LLM call (cost and latency), a serialization point where state has to be packaged into text or JSON and handed off (information loss and drift risk), and a new place where things can go wrong — an agent misinterpreting its handoff, a message getting lost, an infinite loop of two agents deferring to each other. So the practical decision rule is: use a single agent with tools for anything that fits comfortably in one context window and one coherent role; reach for multiple agents when you have genuinely separable domains of expertise, when subtasks can run in parallel and latency matters, or when independent verification meaningfully improves quality on a task where mistakes are costly.

## The Three Core Topologies

Once you've decided multiple agents are justified, the next decision is how they are wired together — who talks to whom, who has authority to make final decisions, and how control flows. Three topologies cover the overwhelming majority of production systems: supervisor/orchestrator-worker, hierarchical, and peer-to-peer/swarm. Nearly every framework-specific pattern you'll encounter (LangGraph's supervisor node, AutoGen's `GroupChatManager`, CrewAI's hierarchical process) is a variation on one of these three, so understanding them at this structural level transfers across tooling.

### Supervisor / Orchestrator-Worker

This is the default starting topology for most production multi-agent systems, and for good reason: it is the easiest to reason about, debug, and constrain. A single orchestrator agent (sometimes just a piece of deterministic code, sometimes itself an LLM call) receives the overall goal, decomposes it into subtasks, assigns each subtask to a worker agent chosen for its specialization, waits for the workers to finish, and then synthesizes their outputs into a final answer. Workers do not talk to each other directly; all communication flows through the supervisor. This mirrors a manager directing a team of individual contributors who don't need to coordinate with each other because the manager owns the integration.

```python
import json


class Orchestrator:
    """A supervisor that decomposes a goal, dispatches to specialist
    workers, and synthesizes their results into one answer."""

    def __init__(self, llm, workers):
        self.llm = llm
        self.workers = {w.name: w for w in workers}

    def run(self, task: str) -> str:
        subtasks = self._decompose(task)
        results = {}

        for subtask in subtasks:
            worker = self.workers[subtask["assigned_worker"]]
            try:
                results[subtask["id"]] = {
                    "worker": worker.name,
                    "output": worker.execute(subtask["description"]),
                    "status": "ok",
                }
            except Exception as exc:
                # A failed worker does not crash the whole run; the
                # supervisor decides how to proceed with a partial result.
                results[subtask["id"]] = {
                    "worker": worker.name,
                    "output": None,
                    "status": "failed",
                    "error": str(exc),
                }

        return self._synthesize(task, results)

    def _decompose(self, task: str) -> list[dict]:
        worker_descriptions = "\n".join(
            f"- {name}: {w.description}" for name, w in self.workers.items()
        )
        response = self.llm.generate(f"""
        Break this task into subtasks, each assigned to exactly one worker.
        Task: {task}
        Workers available:
        {worker_descriptions}

        Return JSON: [{{"id": "...", "description": "...", "assigned_worker": "..."}}]
        """)
        return json.loads(response)

    def _synthesize(self, task: str, results: dict) -> str:
        return self.llm.generate(f"""
        Original task: {task}
        Subtask results (some may have failed): {json.dumps(results, indent=2)}

        Produce the best possible final answer. If a subtask failed, work
        around the gap or note the limitation explicitly rather than
        inventing a result.
        """)
```

The strength of this pattern is control. Because everything routes through one place, you get a single point where you can validate inputs before dispatch, apply budget and timeout limits per worker, log a clean linear trace of what happened, and enforce policy (e.g., "never let the writer agent call external APIs"). Debugging is comparatively pleasant: when something goes wrong, you look at the supervisor's decomposition decision and each worker's isolated input/output pair, rather than untangling a web of peer messages.

The weaknesses are equally structural. The supervisor is a single point of failure — if its decomposition step misunderstands the task, every downstream worker inherits that misunderstanding, and there is no peer to catch it. It is also a throughput bottleneck: all synthesis work funnels through one LLM call, and if that call has to reason over a large number of worker outputs, its context window becomes the new limiting resource, and the same context-dilution problem you were trying to avoid at the single-agent level reappears one level up, just with pre-digested rather than raw content. Cost-wise, this topology is efficient in the sense that the number of LLM calls scales linearly with the number of subtasks (one decomposition, N worker calls, one synthesis), which is the cheapest of the three topologies for a fixed number of workers.

### Hierarchical

Hierarchical orchestration is what you get when a single supervisor's span of control becomes too wide — too many workers for one decomposition step to route sensibly, or subtasks that themselves need further decomposition. Instead of one supervisor managing every worker directly, you introduce intermediate manager agents, each responsible for a subdomain, which in turn manage their own specialists. A director agent talks only to managers; managers talk only to their own specialists. This is structurally identical to how large human organizations scale management: no single manager needs to understand the full org chart, only their direct reports.

```python
class HierarchicalAgent:
    """A node that is either a specialist (does work directly) or a
    manager (decomposes and routes to subordinates)."""

    def __init__(self, name, role, llm, is_specialist=False):
        self.name = name
        self.role = role
        self.llm = llm
        self.is_specialist = is_specialist
        self.subordinates: list["HierarchicalAgent"] = []

    def add_subordinate(self, agent: "HierarchicalAgent"):
        self.subordinates.append(agent)

    def execute(self, task: str) -> str:
        if self.is_specialist or not self.subordinates:
            return self.llm.generate(
                f"You are {self.name}, a specialist in {self.role}. "
                f"Complete this task: {task}"
            )

        subtasks = self._decompose(task)
        results = {}
        for subtask in subtasks:
            subordinate = self._route(subtask)
            results[subordinate.name] = subordinate.execute(subtask["description"])

        return self._synthesize(task, results)

    def _decompose(self, task: str) -> list[dict]:
        roster = [(s.name, s.role) for s in self.subordinates]
        response = self.llm.generate(f"""
        As {self.name} ({self.role}), break this task into subtasks for
        your direct reports: {roster}
        Task: {task}
        Return JSON: [{{"description": "...", "best_subordinate": "..."}}]
        """)
        return json.loads(response)

    def _route(self, subtask: dict) -> "HierarchicalAgent":
        name = subtask["best_subordinate"]
        return next(s for s in self.subordinates if s.name == name)

    def _synthesize(self, task: str, results: dict) -> str:
        return self.llm.generate(f"""
        As {self.name}, integrate your team's results into one deliverable.
        Task: {task}
        Team results: {json.dumps(results, indent=2)}
        """)
```

The recursive structure has a real benefit: each manager's decomposition problem stays small and tractable, because it only has to reason about its own direct reports rather than the entire agent population. This keeps each individual LLM call's context bounded even as the total system scales to dozens of agents, which is exactly the property that a flat supervisor-worker topology loses once the worker count grows large.

The cost is depth. Every level of hierarchy adds a synthesis pass — a leaf specialist's output has to be synthesized by its manager, whose synthesized output is itself synthesized by the director above, and so on. That means end-to-end latency grows with tree depth even when work at each level is parallelized, because synthesis is inherently sequential (a manager can't summarize its team until the team has all reported back). It also means errors and quality loss compound multiplicatively: each synthesis step is a lossy compression of the layer below it, so a five-level hierarchy risks a much blurrier final answer than a two-level one, purely from repeated summarization. Use hierarchy only when the problem is genuinely wide (many specialists, naturally grouped into subdomains) — don't add levels for their own sake.

### Peer-to-Peer / Swarm

In a peer-to-peer topology, there is no designated supervisor. Agents communicate directly with each other, typically through direct messages or a shared broadcast channel, and the overall solution emerges from their interactions rather than being imposed top-down. A common pattern is a small peer group with complementary roles — a writer, an editor, and a fact-checker — that pass drafts back and forth until they jointly converge on an accepted output, or a larger swarm of many simple, largely interchangeable agents whose local interactions produce useful collective behavior without any individual agent understanding the whole task.

```python
class PeerAgent:
    def __init__(self, name, llm, capabilities):
        self.name = name
        self.llm = llm
        self.capabilities = capabilities
        self.peers: dict[str, "PeerAgent"] = {}
        self.inbox: list[dict] = []

    def register_peer(self, peer: "PeerAgent"):
        self.peers[peer.name] = peer

    def send(self, recipient_name: str, content):
        peer = self.peers.get(recipient_name)
        if peer:
            peer.inbox.append({"from": self.name, "content": content})

    def broadcast(self, content):
        for peer in self.peers.values():
            peer.inbox.append({"from": self.name, "content": content})

    def process_inbox(self) -> list[dict]:
        responses = []
        while self.inbox:
            message = self.inbox.pop(0)
            reply = self.llm.generate(f"""
            You are {self.name}. Capabilities: {self.capabilities}
            Message from {message['from']}: {message['content']}
            Respond, and say whether you accept, reject, or want a revision.
            """)
            self.send(message["from"], reply)
            responses.append(reply)
        return responses


def run_peer_round(agents: list[PeerAgent], rounds: int = 3):
    """Round-robin message processing until agents stop generating new
    traffic or a round budget is exhausted — a simple termination
    condition that avoids an unbounded back-and-forth."""
    for _ in range(rounds):
        activity = False
        for agent in agents:
            if agent.inbox:
                agent.process_inbox()
                activity = True
        if not activity:
            break
```

Peer-to-peer's chief virtue is resilience and flexibility: there is no single agent whose failure takes down the whole system, and new peers can be added without redesigning a central decomposition step. It also models genuinely collaborative tasks well — writer/editor/fact-checker loops are naturally peer relationships, not manager/subordinate ones, because none of them has authority over the others' domain expertise.

The costs are real and often underestimated. Coordination overhead grows combinatorially rather than linearly: with N peers potentially messaging each other, the number of possible communication paths grows roughly with N², and without a supervisor imposing order, you need explicit protocols to prevent infinite loops (agent A asks agent B to revise, B asks A to clarify, A asks B to revise again, forever) and to decide when the group is actually "done." Termination conditions have to be designed deliberately — a round budget, a convergence check, or an explicit "I accept this" signal — because there is no external authority to simply declare completion. Debugging is the hardest of the three topologies: instead of one clean supervisor trace, you have to reconstruct a distributed conversation across N independent message histories, and the emergent quality of a swarm is much harder to guarantee or unit-test than a supervisor's explicit decomposition-then-synthesis logic. Cost also tends to be higher for the same task, because iterative back-and-forth (draft, critique, revise, re-critique) burns more LLM calls than a single-pass supervisor pipeline.

## Modeling Cost And Latency Before You Commit To A Topology

Because every topology decision ultimately trades coordination flexibility for LLM calls, it's worth building a rough cost model before implementation rather than discovering the multiplier in a production bill. For a supervisor/worker system with one decomposition call, N worker calls, and one synthesis call, total cost scales as `1 + N + 1` calls and latency (assuming workers run in parallel) as roughly `decompose_latency + max(worker_latencies) + synthesize_latency` — the parallel fan-out is essentially free in wall-clock time as long as your infrastructure actually issues the worker calls concurrently rather than looping over them sequentially, which is a common accidental regression when someone "simplifies" async code into a `for` loop.

```python
def estimate_supervisor_worker_cost(n_workers: int, call_cost: float, call_latency: float) -> dict:
    total_calls = 1 + n_workers + 1  # decompose + workers + synthesize
    total_cost = total_calls * call_cost
    # Workers run in parallel; latency is decompose + slowest worker + synthesize.
    total_latency = 3 * call_latency
    return {"calls": total_calls, "cost": total_cost, "latency": total_latency}


def estimate_hierarchical_cost(branching_factor: int, depth: int, call_cost: float, call_latency: float) -> dict:
    # Each internal node issues one decompose + one synthesize call;
    # leaves issue one "do the work" call. Total nodes: branching^depth.
    leaf_count = branching_factor ** depth
    internal_count = sum(branching_factor ** d for d in range(depth))
    total_calls = leaf_count + 2 * internal_count
    total_cost = total_calls * call_cost
    # Latency compounds with depth because each level's synthesis waits
    # on the level below finishing, even though each level parallelizes internally.
    total_latency = depth * 3 * call_latency
    return {"calls": total_calls, "cost": total_cost, "latency": total_latency}


def estimate_peer_debate_cost(n_peers: int, rounds: int, call_cost: float, call_latency: float) -> dict:
    total_calls = n_peers * rounds
    total_cost = total_calls * call_cost
    # Rounds are inherently sequential; peers within a round parallelize.
    total_latency = rounds * call_latency
    return {"calls": total_calls, "cost": total_cost, "latency": total_latency}
```

Running these three estimators against the same nominal per-call cost and latency makes the qualitative comparison from the table below concrete: a 3-worker supervisor system and a 3-peer, 3-round debate cost roughly the same in raw call count (5 vs 9), but the debate's latency is three times higher because its calls can't all be parallelized, while the supervisor's can. A two-level, branching-factor-3 hierarchy (9 leaves) costs almost twice the calls of a single flat supervisor with 9 direct workers, purely from the extra layer of decompose/synthesize overhead — which is the concrete, numeric version of the "don't add hierarchy levels for their own sake" guidance above.

## Comparing the Trade-offs Directly

| Dimension | Supervisor/Worker | Hierarchical | Peer-to-Peer/Swarm |
|---|---|---|---|
| Coordination overhead | Low — one decomposition, N parallel calls, one synthesis | Medium — one decomposition/synthesis pair per level | High — grows combinatorially with peer count, needs explicit termination logic |
| Failure isolation | Worker failures are isolated; supervisor failure is a single point of failure | Failures isolated within a subtree; a manager failure loses its whole branch | No single point of failure, but a stuck pair can loop indefinitely without a mediator |
| Latency for independent subtasks | Good — workers run in parallel, one synthesis pass | Good within a level, but sequential across levels (bounded by tree depth) | Often worse — iterative rounds of back-and-forth are inherently sequential |
| Cost predictability | High — call count is deterministic given the decomposition | Medium — depends on tree shape and depth | Low — call count depends on how many rounds it takes to converge |
| Debuggability | High — linear trace, clear ownership of each decision | Medium — trace is a tree, still explicit | Low — distributed conversation, emergent behavior |
| Best fit | Clearly separable subtasks with one obvious integrator | Wide problems that naturally decompose into subdomains, each with their own specialists | Genuinely collaborative tasks (draft/critique loops) or workloads needing decentralized resilience |

In practice, production systems are rarely a pure instance of one topology. A common and effective hybrid is a supervisor at the top level that dispatches to a small peer group for a specific collaborative sub-step (e.g., the supervisor hands a draft to a writer/editor pair that iterate a few rounds, then returns a single finished artifact back to the supervisor as if it were an ordinary worker). This keeps the overall system easy to reason about while letting the naturally collaborative parts of the task benefit from peer interaction. The rule of thumb: default to supervisor/worker for anything with a clear decomposition and an obvious integrator; escalate to hierarchical only when a single supervisor's fan-out is unmanageable; reach for peer-to-peer only for the specific sub-problems that are genuinely dialogic in nature, and bound them tightly with round limits and explicit termination signals rather than letting them run open-ended.

## Choosing a Topology in Practice

A useful way to decide is to ask three questions about the task in front of you. First, is there a natural "owner" of the final answer — someone who has to look at everything and make the last call? If yes, that's a strong signal for supervisor/worker, because you already have your synthesizer. Second, does the problem have more sub-domains than one decomposition step can sensibly route between (rule of thumb: more than five or six distinct worker types starts to strain a flat supervisor's decomposition prompt)? If yes, group related workers under intermediate managers and go hierarchical. Third, is the task inherently a negotiation or iterative refinement between roles that both have legitimate authority over the outcome — nobody is purely "in charge" — such as a writer and a critic who both need to agree a piece is done? If yes, that sub-step belongs in a peer loop, ideally nested inside a supervisor that owns the overall workflow and bounds how many rounds the peer loop is allowed to run.

Finally, remember that the topology decision interacts with cost and latency budgets in a very concrete way. Every extra agent hop is another LLM call on the critical path (for sequential dependencies) or another parallel branch to pay for (even if latency-free, it isn't cost-free). Before committing to a five-agent hierarchical swarm, it's worth prototyping the same task with a single strong agent and good tool access — often a well-scoped single agent with retrieval and a code execution tool solves 80% of what looked like it needed orchestration, and you only pay the coordination tax for the genuinely separable, genuinely parallelizable, or genuinely dialogic remainder.

## A Worked Hybrid Example

The cleanest way to see how these topologies compose is to walk through a concrete hybrid: a content-production system where a supervisor owns the overall workflow, dispatches a research phase to parallel workers, and then hands the drafting phase to a small peer loop (writer and editor) that iterates a bounded number of rounds before returning a single finished artifact back to the supervisor, which treats that artifact exactly like any other worker's output for the purposes of final synthesis.

```python
class HybridOrchestrator:
    """Supervisor/worker at the top level, with one 'worker' that is
    internally a bounded peer loop rather than a single LLM call."""

    def __init__(self, llm, research_workers, writer, editor, max_edit_rounds=2):
        self.llm = llm
        self.research_workers = research_workers
        self.writer = writer
        self.editor = editor
        self.max_edit_rounds = max_edit_rounds

    def run(self, topic: str) -> str:
        # Phase 1: supervisor/worker fan-out, run in parallel.
        research = {w.name: w.execute(topic) for w in self.research_workers}

        # Phase 2: a bounded peer loop, exposed to the supervisor as a
        # single opaque "draft_with_review" step.
        draft = self._draft_with_review(topic, research)

        # Phase 3: supervisor-level synthesis, identical in shape to the
        # plain supervisor/worker pattern.
        return self.llm.generate(f"""
        Topic: {topic}
        Research: {research}
        Reviewed draft: {draft}
        Produce the final polished piece, incorporating any editor notes
        still outstanding in the draft.
        """)

    def _draft_with_review(self, topic: str, research: dict) -> str:
        draft = self.writer.execute(f"Topic: {topic}\nResearch: {research}\nWrite a draft.")
        for _ in range(self.max_edit_rounds):
            review = self.editor.execute(f"Review this draft:\n{draft}")
            if "APPROVED" in review:
                break
            draft = self.writer.execute(f"Revise based on this feedback:\n{review}\n\nDraft:\n{draft}")
        return draft
```

This is a useful default shape for real systems: keep the outer control flow as a supervisor (predictable, easy to trace, cheap to reason about), and only drop into a peer loop for the specific sub-steps that are genuinely dialogic, bounding those loops tightly (`max_edit_rounds` above) so a stuck writer/editor pair degrades to "best draft after N rounds" rather than looping indefinitely.

## Observability Across Topologies

Whichever topology you choose, instrument it before you need it, not after a production incident forces you to. At minimum, log three things for every agent invocation regardless of topology: which agent ran, what it was given as input (or a hash/summary of it if the input is large or sensitive), and what it produced, tagged with a shared trace ID for the overall task so the whole run can be reconstructed as a single timeline. For supervisor/worker and hierarchical systems this is straightforward because the call graph is explicit and known ahead of time; for peer-to-peer systems it's more important, not less, because the call graph is only known after the fact, and without a shared trace ID, reconstructing "what actually happened in this run" from N independent peer logs is close to impossible after the system has moved on to the next task. Set a hard per-task budget — a maximum number of total LLM calls or a wall-clock timeout — independent of any per-agent or per-round limits, because the combination of retries, hierarchy depth, and peer rounds can compound in ways that are hard to bound analytically in advance; a global circuit breaker that aborts and returns the best-available partial result is cheap insurance against a misconfigured loop silently burning an unbounded budget.
