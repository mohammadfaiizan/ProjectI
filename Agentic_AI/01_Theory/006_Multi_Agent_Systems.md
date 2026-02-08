# Multi-Agent Systems

## Table of Contents

1. Introduction to Multi-Agent Systems
2. Multi-Agent Architectures
3. Communication Protocols
4. Task Delegation and Distribution
5. Coordination and Collaboration
6. Agent Roles and Specialization
7. Multi-Agent Frameworks
8. Challenges in Multi-Agent Systems
9. Production Multi-Agent Systems
10. Case Studies

---

## 1. Introduction to Multi-Agent Systems

### Why Multiple Agents?

Single agents have inherent limitations when facing complex, multi-faceted tasks:

| Limitation | Description | Multi-Agent Solution |
|-----------|-------------|---------------------|
| Context window | Single agent cannot hold all relevant info | Distribute context across agents |
| Specialization | One agent cannot be expert in everything | Dedicated specialist agents |
| Reliability | Single point of failure | Redundancy and fault tolerance |
| Latency | Sequential processing of subtasks | Parallel execution |
| Complexity | Prompt becomes unwieldy for complex tasks | Decompose into focused agents |
| Quality | One perspective may miss issues | Multiple viewpoints improve quality |

### Single Agent vs Multi-Agent Decision

```
                    How complex is the task?
                           |
                  +--------+--------+
                  |                 |
               Simple           Complex
                  |                 |
            Single Agent     How many domains
                             are involved?
                                  |
                         +--------+--------+
                         |                 |
                        One             Multiple
                         |                 |
                   Single Agent      Can subtasks run
                   with tools       in parallel?
                                         |
                                +--------+--------+
                                |                 |
                               Yes               No
                                |                 |
                          Parallel MAS      Sequential MAS
                          (speed gain)     (quality gain)
```

### Foundational Concepts

**Agent**: An autonomous entity with its own prompt, tools, memory, and reasoning loop.

**Multi-Agent System (MAS)**: A collection of agents that interact, coordinate, and collaborate to solve problems that are beyond the capabilities of individual agents.

**Orchestrator**: A component (agent or system) that manages the flow of tasks and communication between agents.

**Swarm**: A collection of many simple agents that collectively exhibit intelligent behavior through local interactions.

---

## 2. Multi-Agent Architectures

### 2.1 Centralized / Master-Worker

A central orchestrator decomposes tasks, assigns them to worker agents, and aggregates results.

```
                    +----------------+
                    |  ORCHESTRATOR  |
                    |  (Master)      |
                    +-------+--------+
                            |
              +-------------+-------------+
              |             |             |
        +-----v----+  +----v-----+  +----v-----+
        | Worker A |  | Worker B |  | Worker C |
        | (Research|  | (Analysis|  | (Writing)|
        |  Agent)  |  |  Agent)  |  |  Agent)  |
        +----------+  +----------+  +----------+
```

```python
class Orchestrator:
    def __init__(self, llm, workers):
        self.llm = llm
        self.workers = {w.name: w for w in workers}

    def Execute(self, task):
        # Step 1: Decompose task
        subtasks = self.Decompose(task)

        # Step 2: Assign to workers
        assignments = self.Assign(subtasks)

        # Step 3: Execute (potentially in parallel)
        results = {}
        for worker_name, subtask in assignments.items():
            worker = self.workers[worker_name]
            results[worker_name] = worker.Execute(subtask)

        # Step 4: Aggregate results
        final = self.Aggregate(task, results)
        return final

    def Decompose(self, task):
        response = self.llm.generate(f"""
        Decompose this task into subtasks:
        Task: {task}

        Available workers: {list(self.workers.keys())}
        Worker capabilities:
        {self.Get_Worker_Descriptions()}

        Return a JSON list of subtasks with assigned worker.
        """)
        return json.loads(response)

    def Assign(self, subtasks):
        assignments = {}
        for st in subtasks:
            worker = st["assigned_worker"]
            assignments[worker] = st["description"]
        return assignments

    def Aggregate(self, original_task, results):
        return self.llm.generate(f"""
        Original task: {original_task}
        Worker results: {json.dumps(results, indent=2)}

        Synthesize all results into a comprehensive final output.
        """)

    def Get_Worker_Descriptions(self):
        return "\n".join(
            f"- {name}: {w.description}" for name, w in self.workers.items()
        )


class Worker_Agent:
    def __init__(self, name, description, llm, tools=None):
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []

    def Execute(self, task):
        return self.llm.generate(f"""
        You are {self.name}. {self.description}
        Task: {task}
        Available tools: {[t.name for t in self.tools]}

        Complete the task and return your results.
        """)
```

**Pros**: Clear control flow, easy to debug, simple error handling
**Cons**: Bottleneck at orchestrator, single point of failure, limited scalability

### 2.2 Peer-to-Peer / Decentralized

Agents communicate directly with each other without a central coordinator.

```
        +----------+          +----------+
        | Agent A  |<-------->| Agent B  |
        | (Writer) |          | (Editor) |
        +----+-----+          +-----+----+
             |                      |
             |    +----------+      |
             +--->| Agent C  |<-----+
                  | (Fact    |
                  | Checker) |
                  +----------+
```

```python
class Peer_Agent:
    def __init__(self, name, llm, capabilities):
        self.name = name
        self.llm = llm
        self.capabilities = capabilities
        self.peers = {}
        self.inbox = []

    def Register_Peer(self, peer):
        self.peers[peer.name] = peer

    def Send_Message(self, recipient_name, message):
        peer = self.peers.get(recipient_name)
        if peer:
            peer.Receive_Message(self.name, message)

    def Receive_Message(self, sender, message):
        self.inbox.append({"from": sender, "content": message})

    def Process_Inbox(self):
        results = []
        for msg in self.inbox:
            response = self.Handle_Message(msg)
            self.Send_Message(msg["from"], response)
            results.append(response)
        self.inbox.clear()
        return results

    def Handle_Message(self, message):
        return self.llm.generate(f"""
        You are {self.name}. Capabilities: {self.capabilities}
        Message from {message['from']}: {message['content']}

        Respond appropriately based on your capabilities.
        """)

    def Broadcast(self, message):
        for name, peer in self.peers.items():
            peer.Receive_Message(self.name, message)
```

**Pros**: No single point of failure, scalable, flexible
**Cons**: Complex coordination, potential message loops, harder to debug

### 2.3 Hierarchical

Multiple levels of management. Managers delegate to specialists, who may further delegate.

```
                    +-----------+
                    | Director  |
                    | Agent     |
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
        +-----v-----+          +-----v-----+
        | Manager A |          | Manager B |
        | (Research)|          | (Product) |
        +-----+-----+          +-----+-----+
              |                       |
        +-----+-----+          +-----+-----+
        |           |          |           |
   +----v---+ +----v---+ +----v---+ +----v---+
   |Scholar | |Analyst | |Designer| |Engineer|
   +--------+ +--------+ +--------+ +--------+
```

```python
class Hierarchical_Agent:
    def __init__(self, name, role, llm, parent=None):
        self.name = name
        self.role = role  # "director", "manager", "specialist"
        self.llm = llm
        self.parent = parent
        self.subordinates = []

    def Add_Subordinate(self, agent):
        agent.parent = self
        self.subordinates.append(agent)

    def Execute(self, task):
        if self.role == "specialist":
            return self.Do_Work(task)

        # Manager/Director: decompose and delegate
        subtasks = self.Decompose(task)
        results = {}

        for subtask in subtasks:
            best_sub = self.Route_To_Subordinate(subtask)
            results[best_sub.name] = best_sub.Execute(subtask["description"])

        return self.Synthesize(task, results)

    def Route_To_Subordinate(self, subtask):
        # Match subtask to best subordinate based on capabilities
        scores = []
        for sub in self.subordinates:
            score = self.llm.generate(f"""
            Subtask: {subtask['description']}
            Agent: {sub.name} - {sub.role}
            Rate match 0-10:
            """)
            scores.append((float(score.strip()), sub))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def Do_Work(self, task):
        return self.llm.generate(f"""
        You are {self.name}, a specialist in {self.role}.
        Complete this task: {task}
        """)

    def Decompose(self, task):
        response = self.llm.generate(f"""
        Task: {task}
        Subordinates: {[(s.name, s.role) for s in self.subordinates]}
        Decompose into subtasks for your team.
        Return JSON: [{{"description": "...", "priority": 1-5}}]
        """)
        return json.loads(response)

    def Synthesize(self, task, results):
        return self.llm.generate(f"""
        Original task: {task}
        Team results: {json.dumps(results, indent=2)}
        Synthesize into final output.
        """)
```

### 2.4 Federated

Independent agents with different owners/domains working toward shared goals. Each agent maintains sovereignty over its own data and decision-making.

```python
class Federated_Agent:
    def __init__(self, name, domain, llm, shared_registry):
        self.name = name
        self.domain = domain
        self.llm = llm
        self.registry = shared_registry  # Shared service discovery
        self.private_memory = {}

    def Request_Service(self, capability_needed, request_data):
        # Find agents with needed capability
        providers = self.registry.Find_Providers(capability_needed)

        # Send request (only share necessary data)
        sanitized = self.Sanitize_Data(request_data)

        for provider in providers:
            try:
                result = provider.Handle_Request(
                    requester=self.name,
                    capability=capability_needed,
                    data=sanitized
                )
                return result
            except Exception:
                continue

        return None

    def Handle_Request(self, requester, capability, data):
        # Validate request
        if not self.Can_Handle(capability):
            raise ValueError(f"Cannot handle: {capability}")

        # Process within own domain
        result = self.Process(capability, data)

        # Return only shareable results
        return self.Sanitize_Output(result)

    def Sanitize_Data(self, data):
        # Remove sensitive information before sharing
        return {k: v for k, v in data.items() if k not in self.private_fields}
```

### Architecture Comparison

| Architecture | Control | Scalability | Fault Tolerance | Complexity | Best For |
|-------------|---------|-------------|----------------|------------|----------|
| Centralized | High | Low-Medium | Low | Low | Simple workflows |
| Peer-to-Peer | Low | High | High | High | Collaborative tasks |
| Hierarchical | Medium | Medium | Medium | Medium | Large organizations |
| Federated | Low | High | High | High | Cross-domain tasks |
| Hybrid | Varies | High | High | High | Production systems |

---

## 3. Communication Protocols

### 3.1 Message Passing

```python
from dataclasses import dataclass, field
from enum import Enum
import uuid

class Message_Type(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    BROADCAST = "broadcast"

@dataclass
class Agent_Message:
    sender: str
    receiver: str
    msg_type: Message_Type
    content: dict
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reply_to: str = None
    priority: int = 5
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

class Message_Bus:
    def __init__(self):
        self.queues = {}  # agent_name -> [messages]
        self.handlers = {}

    def Register_Agent(self, agent_name, handler):
        self.queues[agent_name] = []
        self.handlers[agent_name] = handler

    def Send(self, message: Agent_Message):
        if message.receiver in self.queues:
            self.queues[message.receiver].append(message)
        elif message.msg_type == Message_Type.BROADCAST:
            for name, queue in self.queues.items():
                if name != message.sender:
                    queue.append(message)

    def Process(self, agent_name):
        queue = self.queues.get(agent_name, [])
        queue.sort(key=lambda m: m.priority, reverse=True)

        results = []
        while queue:
            msg = queue.pop(0)
            handler = self.handlers[agent_name]
            response = handler(msg)
            if response:
                self.Send(response)
            results.append(response)

        return results
```

### 3.2 Shared Blackboard

A central data structure that all agents can read from and write to.

```python
class Blackboard:
    def __init__(self):
        self.data = {}
        self.history = []
        self.watchers = {}  # key_pattern -> [callbacks]

    def Write(self, key, value, author):
        old_value = self.data.get(key)
        self.data[key] = {
            "value": value,
            "author": author,
            "timestamp": datetime.now(),
            "version": self.data.get(key, {}).get("version", 0) + 1,
        }
        self.history.append({
            "action": "write",
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "author": author,
        })
        self.Notify_Watchers(key, value, author)

    def Read(self, key):
        entry = self.data.get(key)
        return entry["value"] if entry else None

    def Watch(self, key_pattern, callback):
        self.watchers.setdefault(key_pattern, []).append(callback)

    def Notify_Watchers(self, key, value, author):
        for pattern, callbacks in self.watchers.items():
            if key.startswith(pattern) or pattern == "*":
                for cb in callbacks:
                    cb(key, value, author)

    def Query(self, prefix=None, author=None):
        results = {}
        for key, entry in self.data.items():
            if prefix and not key.startswith(prefix):
                continue
            if author and entry["author"] != author:
                continue
            results[key] = entry["value"]
        return results
```

### 3.3 Publish-Subscribe

```python
class Event_Bus:
    def __init__(self):
        self.subscribers = {}  # topic -> [callbacks]
        self.message_log = []

    def Subscribe(self, topic, agent_name, callback):
        self.subscribers.setdefault(topic, []).append({
            "agent": agent_name,
            "callback": callback,
        })

    def Publish(self, topic, data, publisher):
        event = {
            "topic": topic,
            "data": data,
            "publisher": publisher,
            "timestamp": datetime.now(),
        }
        self.message_log.append(event)

        for sub in self.subscribers.get(topic, []):
            if sub["agent"] != publisher:
                sub["callback"](event)

    def Unsubscribe(self, topic, agent_name):
        if topic in self.subscribers:
            self.subscribers[topic] = [
                s for s in self.subscribers[topic] if s["agent"] != agent_name
            ]

# Usage
event_bus = Event_Bus()

event_bus.Subscribe("research_complete", "writer_agent", lambda e: print(f"Writer got: {e}"))
event_bus.Subscribe("research_complete", "editor_agent", lambda e: print(f"Editor got: {e}"))

event_bus.Publish("research_complete", {"findings": "..."}, "researcher_agent")
```

### Communication Pattern Comparison

| Pattern | Coupling | Scalability | Ordering | Complexity | Use Case |
|---------|---------|-------------|----------|------------|----------|
| Direct Message | Tight | Low | Guaranteed | Low | 1-to-1 communication |
| Blackboard | Medium | Medium | None | Medium | Shared state |
| Pub-Sub | Loose | High | None | Medium | Event-driven |
| Request-Response | Medium | Medium | Per-pair | Low | Service calls |
| Broadcast | Loose | Low | None | Low | Announcements |

---

## 4. Task Delegation and Distribution

### 4.1 Task Decomposition Strategies

```python
class Task_Decomposer:
    def __init__(self, llm):
        self.llm = llm

    def Decompose_Sequential(self, task, available_agents):
        return self.llm.generate(f"""
        Task: {task}
        Available agents: {json.dumps(available_agents)}

        Decompose into sequential subtasks. Each subtask may depend on
        results of previous subtasks.

        Return JSON:
        [
          {{"id": 1, "description": "...", "agent": "...", "depends_on": []}},
          {{"id": 2, "description": "...", "agent": "...", "depends_on": [1]}}
        ]
        """)

    def Decompose_Parallel(self, task, available_agents):
        return self.llm.generate(f"""
        Task: {task}
        Available agents: {json.dumps(available_agents)}

        Decompose into independent subtasks that can run in parallel.
        Identify which subtasks are truly independent.

        Return JSON:
        {{
          "parallel_groups": [
            {{"group": 1, "tasks": [...]}},
            {{"group": 2, "tasks": [...], "depends_on_group": 1}}
          ]
        }}
        """)

    def Decompose_Hierarchical(self, task, available_agents, depth=2):
        return self.llm.generate(f"""
        Task: {task}
        Available agents: {json.dumps(available_agents)}

        Decompose hierarchically to depth {depth}:
        - Level 1: Major phases
        - Level 2: Specific tasks within each phase

        Return nested JSON structure.
        """)
```

### 4.2 Capability-Based Routing

```python
class Capability_Router:
    def __init__(self, agents):
        self.agents = agents
        self.capability_index = self.Build_Index()

    def Build_Index(self):
        index = {}
        for agent in self.agents:
            for cap in agent.capabilities:
                index.setdefault(cap, []).append(agent)
        return index

    def Route(self, task, required_capabilities):
        candidates = set(self.agents)

        for cap in required_capabilities:
            capable = set(self.capability_index.get(cap, []))
            candidates = candidates.intersection(capable)

        if not candidates:
            return self.Find_Best_Partial_Match(required_capabilities)

        # Score candidates by workload and past performance
        scored = []
        for agent in candidates:
            score = self.Score_Agent(agent, required_capabilities)
            scored.append((score, agent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def Score_Agent(self, agent, capabilities):
        capability_match = sum(
            1 for c in capabilities if c in agent.capabilities
        ) / len(capabilities)
        workload = 1.0 - (agent.current_tasks / agent.max_tasks)
        performance = agent.success_rate

        return 0.4 * capability_match + 0.3 * workload + 0.3 * performance
```

### 4.3 Load Balancing

```python
class Load_Balancer:
    def __init__(self, agents):
        self.agents = agents
        self.task_counts = {a.name: 0 for a in agents}

    def Assign_Round_Robin(self, task):
        # Simple round-robin
        min_agent = min(self.task_counts, key=self.task_counts.get)
        self.task_counts[min_agent] += 1
        agent = next(a for a in self.agents if a.name == min_agent)
        return agent

    def Assign_Weighted(self, task):
        # Weight by agent capacity and current load
        best = None
        best_score = -1

        for agent in self.agents:
            load = self.task_counts[agent.name] / agent.max_concurrent
            available_capacity = 1.0 - load
            score = available_capacity * agent.speed_factor

            if score > best_score:
                best_score = score
                best = agent

        if best:
            self.task_counts[best.name] += 1
        return best

    def Release(self, agent_name):
        self.task_counts[agent_name] = max(0, self.task_counts[agent_name] - 1)
```

---

## 5. Coordination and Collaboration

### 5.1 Consensus Building

```python
class Consensus_Manager:
    def __init__(self, agents, threshold=0.6):
        self.agents = agents
        self.threshold = threshold

    def Reach_Consensus(self, question, max_rounds=3):
        for round_num in range(max_rounds):
            # Collect opinions
            opinions = {}
            for agent in self.agents:
                opinion = agent.Provide_Opinion(question)
                opinions[agent.name] = opinion

            # Check agreement
            agreement = self.Check_Agreement(opinions)

            if agreement["score"] >= self.threshold:
                return {
                    "consensus": True,
                    "result": agreement["majority_opinion"],
                    "confidence": agreement["score"],
                    "rounds": round_num + 1,
                }

            # Share opinions and ask to reconsider
            question = self.Build_Reconsideration_Prompt(question, opinions)

        return {
            "consensus": False,
            "opinions": opinions,
            "best_effort": self.Get_Majority(opinions),
        }

    def Check_Agreement(self, opinions):
        values = list(opinions.values())
        unique = set(values)
        if len(unique) == 1:
            return {"score": 1.0, "majority_opinion": values[0]}

        # Count agreement
        from collections import Counter
        counts = Counter(values)
        majority_opinion, majority_count = counts.most_common(1)[0]
        score = majority_count / len(values)

        return {"score": score, "majority_opinion": majority_opinion}
```

### 5.2 Voting Mechanisms

```python
class Voting_System:
    def __init__(self, agents):
        self.agents = agents

    def Majority_Vote(self, question):
        votes = {}
        for agent in self.agents:
            vote = agent.Vote(question)
            votes[agent.name] = vote

        from collections import Counter
        counts = Counter(votes.values())
        winner = counts.most_common(1)[0]
        return {
            "winner": winner[0],
            "votes": winner[1],
            "total": len(votes),
            "all_votes": votes,
        }

    def Ranked_Vote(self, question, options):
        rankings = {}
        for agent in self.agents:
            ranking = agent.Rank_Options(question, options)
            rankings[agent.name] = ranking

        # Borda count
        scores = {opt: 0 for opt in options}
        for agent_ranking in rankings.values():
            for rank, option in enumerate(agent_ranking):
                scores[option] += len(options) - rank

        winner = max(scores, key=scores.get)
        return {"winner": winner, "scores": scores, "rankings": rankings}

    def Weighted_Vote(self, question, agent_weights=None):
        if agent_weights is None:
            agent_weights = {a.name: 1.0 for a in self.agents}

        votes = {}
        for agent in self.agents:
            vote = agent.Vote(question)
            weight = agent_weights.get(agent.name, 1.0)
            votes[agent.name] = {"vote": vote, "weight": weight}

        weighted_counts = {}
        for v in votes.values():
            weighted_counts[v["vote"]] = (
                weighted_counts.get(v["vote"], 0) + v["weight"]
            )

        winner = max(weighted_counts, key=weighted_counts.get)
        return {"winner": winner, "weighted_scores": weighted_counts}
```

### 5.3 Contract Net Protocol

```python
class Contract_Net:
    def __init__(self, manager, contractors):
        self.manager = manager
        self.contractors = contractors

    def Execute(self, task):
        # Step 1: Announce task
        announcement = self.manager.Announce_Task(task)

        # Step 2: Collect bids
        bids = []
        for contractor in self.contractors:
            bid = contractor.Submit_Bid(announcement)
            if bid["willing"]:
                bids.append({
                    "agent": contractor,
                    "cost": bid["estimated_cost"],
                    "time": bid["estimated_time"],
                    "confidence": bid["confidence"],
                })

        if not bids:
            return {"success": False, "reason": "No bids received"}

        # Step 3: Evaluate bids and select winner
        winner = self.Evaluate_Bids(bids)

        # Step 4: Award contract
        result = winner["agent"].Execute_Contract(task)

        return {"success": True, "agent": winner["agent"].name, "result": result}

    def Evaluate_Bids(self, bids):
        # Score bids (lower cost, lower time, higher confidence = better)
        for bid in bids:
            bid["score"] = (
                0.3 * (1 / bid["cost"]) +
                0.3 * (1 / bid["time"]) +
                0.4 * bid["confidence"]
            )
        return max(bids, key=lambda b: b["score"])
```

---

## 6. Agent Roles and Specialization

### Common Agent Roles

| Role | Description | Skills |
|------|-------------|--------|
| Researcher | Gathers and synthesizes information | Web search, document analysis, summarization |
| Writer | Creates content and documentation | Writing, formatting, tone adaptation |
| Analyst | Analyzes data and provides insights | Data processing, statistics, visualization |
| Coder | Writes and debugs code | Code generation, debugging, testing |
| Reviewer | Reviews and critiques outputs | Quality assessment, error detection |
| Planner | Creates plans and strategies | Task decomposition, scheduling, prioritization |
| Executor | Carries out specific actions | Tool use, API calls, file operations |
| Coordinator | Manages workflow between agents | Routing, load balancing, conflict resolution |

### Role Definition Pattern

```python
class Agent_Role:
    def __init__(self, name, system_prompt, capabilities, tools, constraints=None):
        self.name = name
        self.system_prompt = system_prompt
        self.capabilities = capabilities
        self.tools = tools
        self.constraints = constraints or []

RESEARCHER_ROLE = Agent_Role(
    name="Researcher",
    system_prompt="""You are a research specialist. Your job is to:
    1. Search for relevant information using available tools
    2. Evaluate source credibility
    3. Synthesize findings into structured summaries
    4. Cite all sources
    Always verify claims from multiple sources.""",
    capabilities=["web_search", "document_analysis", "summarization"],
    tools=["search_web", "read_document", "summarize_text"],
    constraints=[
        "Must cite sources for all claims",
        "Must evaluate source credibility",
        "Must flag uncertain information",
    ],
)

WRITER_ROLE = Agent_Role(
    name="Writer",
    system_prompt="""You are a writing specialist. Your job is to:
    1. Transform research findings into polished content
    2. Adapt tone and style to the target audience
    3. Ensure clarity and coherence
    4. Follow the provided outline or structure""",
    capabilities=["content_creation", "editing", "formatting"],
    tools=["text_editor", "grammar_check", "format_document"],
    constraints=[
        "Must follow the provided outline",
        "Must maintain consistent tone",
        "Must keep content factually accurate",
    ],
)

REVIEWER_ROLE = Agent_Role(
    name="Reviewer",
    system_prompt="""You are a quality reviewer. Your job is to:
    1. Check content for factual accuracy
    2. Identify logical inconsistencies
    3. Evaluate completeness
    4. Provide specific, actionable feedback
    Be thorough but constructive.""",
    capabilities=["quality_assessment", "fact_checking", "feedback"],
    tools=["search_web", "compare_documents"],
    constraints=[
        "Must provide specific feedback, not vague criticism",
        "Must suggest improvements, not just identify problems",
        "Must prioritize critical issues over minor ones",
    ],
)
```

### Dynamic Role Assignment

```python
class Dynamic_Role_Manager:
    def __init__(self, available_roles, llm):
        self.roles = {r.name: r for r in available_roles}
        self.llm = llm
        self.assignments = {}

    def Assign_Role(self, agent, task_context):
        best_role = self.llm.generate(f"""
        Task context: {task_context}
        Available roles: {list(self.roles.keys())}
        Agent capabilities: {agent.base_capabilities}

        Which role best fits this agent for the current task?
        Return the role name only.
        """).strip()

        role = self.roles.get(best_role)
        if role:
            agent.Apply_Role(role)
            self.assignments[agent.name] = role.name

        return role

    def Reassign_If_Needed(self, agent, performance_data):
        if performance_data["success_rate"] < 0.5:
            # Agent is underperforming in current role
            current_role = self.assignments.get(agent.name)
            alternative = self.Find_Better_Role(agent, current_role, performance_data)
            if alternative:
                self.Assign_Role(agent, alternative)
```

---

## 7. Multi-Agent Frameworks

### 7.1 CrewAI

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Research and analyze the given topic thoroughly",
    backstory="Expert researcher with deep analytical skills",
    tools=[search_tool, scrape_tool],
    verbose=True,
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging and informative content",
    backstory="Experienced writer who transforms research into compelling narratives",
    tools=[text_editor],
    verbose=True,
)

research_task = Task(
    description="Research the impact of AI agents on software development in 2025",
    expected_output="Comprehensive research report with key findings and statistics",
    agent=researcher,
)

writing_task = Task(
    description="Write a blog post based on the research findings",
    expected_output="2000-word blog post with introduction, body, and conclusion",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
```

### 7.2 AutoGen

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

assistant = AssistantAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"model": "gpt-4"},
)

coder = AssistantAgent(
    name="Coder",
    system_message="You write Python code to solve problems. Return code in ```python blocks.",
    llm_config={"model": "gpt-4"},
)

reviewer = AssistantAgent(
    name="Reviewer",
    system_message="You review code for bugs, security issues, and best practices.",
    llm_config={"model": "gpt-4"},
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "output"},
)

group_chat = GroupChat(
    agents=[user_proxy, assistant, coder, reviewer],
    messages=[],
    max_round=10,
)

manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4"})
user_proxy.initiate_chat(manager, message="Create a REST API for a todo app")
```

### 7.3 LangGraph Multi-Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class Multi_Agent_State(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    task: str
    results: dict
    iteration: int

def Researcher_Node(state):
    task = state["task"]
    research = llm.invoke(f"Research: {task}")
    return {
        "messages": [{"agent": "researcher", "content": research}],
        "results": {"research": research},
        "current_agent": "writer",
    }

def Writer_Node(state):
    research = state["results"].get("research", "")
    draft = llm.invoke(f"Write based on: {research}")
    return {
        "messages": [{"agent": "writer", "content": draft}],
        "results": {**state["results"], "draft": draft},
        "current_agent": "reviewer",
    }

def Reviewer_Node(state):
    draft = state["results"].get("draft", "")
    review = llm.invoke(f"Review: {draft}")
    needs_revision = "APPROVED" not in review
    return {
        "messages": [{"agent": "reviewer", "content": review}],
        "results": {**state["results"], "review": review},
        "current_agent": "writer" if needs_revision else "done",
        "iteration": state["iteration"] + 1,
    }

def Route(state):
    if state["current_agent"] == "done" or state["iteration"] > 3:
        return END
    return state["current_agent"]

graph = StateGraph(Multi_Agent_State)
graph.add_node("researcher", Researcher_Node)
graph.add_node("writer", Writer_Node)
graph.add_node("reviewer", Reviewer_Node)

graph.set_entry_point("researcher")
graph.add_conditional_edges("researcher", Route)
graph.add_conditional_edges("writer", Route)
graph.add_conditional_edges("reviewer", Route)

app = graph.compile()
```

### Framework Comparison

| Feature | CrewAI | AutoGen | LangGraph | Custom |
|---------|--------|---------|-----------|--------|
| Learning curve | Low | Medium | High | High |
| Flexibility | Medium | High | Very High | Very High |
| Built-in patterns | Many | Several | Flexible | None |
| Multi-agent chat | No | Yes | No | Build it |
| Graph workflows | No | No | Yes | Build it |
| Code execution | Via tools | Built-in | Via tools | Build it |
| Human-in-loop | Yes | Yes | Yes | Build it |
| Production ready | Growing | Growing | Yes | Depends |
| Community | Large | Large | Large | N/A |

---

## 8. Challenges in Multi-Agent Systems

### 8.1 Communication Overhead

Each message between agents adds latency and cost.

```
Single Agent:     1 LLM call = ~2 seconds, ~$0.05
3-Agent Pipeline: 5 LLM calls = ~10 seconds, ~$0.25
5-Agent Debate:   15 LLM calls = ~30 seconds, ~$0.75
```

**Mitigation strategies:**
- Minimize inter-agent messages
- Use structured, concise message formats
- Batch communications where possible
- Use cheaper models for routine coordination

### 8.2 Error Propagation

An error in one agent can cascade through the system.

```python
class Error_Handler:
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def Execute_With_Recovery(self, agent, task):
        for attempt in range(self.max_retries):
            try:
                result = agent.Execute(task)
                if self.Validate_Result(result):
                    return result
                else:
                    task = self.Refine_Task(task, result, "Invalid result format")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return self.Fallback(agent, task, str(e))
                task = self.Refine_Task(task, None, str(e))

    def Fallback(self, agent, task, error):
        return {
            "success": False,
            "agent": agent.name,
            "task": task,
            "error": error,
            "action": "escalate_to_human",
        }
```

### 8.3 Debugging

```python
class Multi_Agent_Debugger:
    def __init__(self):
        self.trace = []

    def Log_Event(self, event_type, agent, data):
        self.trace.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "agent": agent,
            "data": data,
        })

    def Print_Trace(self, filter_agent=None):
        for event in self.trace:
            if filter_agent and event["agent"] != filter_agent:
                continue
            print(f"[{event['timestamp']}] {event['type']} - {event['agent']}")
            print(f"  Data: {json.dumps(event['data'], indent=2)[:200]}")

    def Find_Errors(self):
        return [e for e in self.trace if e["type"] == "error"]

    def Get_Agent_Timeline(self, agent_name):
        return [e for e in self.trace if e["agent"] == agent_name]

    def Export_Trace(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.trace, f, indent=2)
```

### 8.4 Cost Management

| Strategy | Description | Savings |
|----------|-------------|---------|
| Model tiering | Use cheaper models for simple agents | 30-60% |
| Caching | Cache repeated queries/results | 20-40% |
| Early termination | Stop when quality threshold met | 10-30% |
| Agent pooling | Reuse warm agent instances | 5-15% |
| Batch processing | Group similar tasks together | 15-25% |

---

## 9. Production Multi-Agent Systems

### Monitoring Architecture

```
+------------------------------------------------------------------+
|                 PRODUCTION MULTI-AGENT SYSTEM                    |
|                                                                   |
|  +------------------+     +------------------+                    |
|  | Load Balancer    |---->| Agent Pool       |                    |
|  |                  |     | (Auto-scaling)   |                    |
|  +------------------+     +--------+---------+                    |
|                                    |                              |
|  +------------------+     +--------v---------+                    |
|  | Message Queue    |<--->| Agent Runtime    |                    |
|  | (RabbitMQ/Kafka) |     | (Orchestrator)   |                    |
|  +------------------+     +--------+---------+                    |
|                                    |                              |
|  +------------------+     +--------v---------+                    |
|  | Observability    |     | State Store      |                    |
|  | (Traces, Logs,   |     | (Redis/Postgres) |                    |
|  |  Metrics)        |     |                  |                    |
|  +------------------+     +------------------+                    |
+------------------------------------------------------------------+
```

### Key Production Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| Task completion rate | % tasks completed successfully | < 90% |
| Avg task latency | Mean time to complete tasks | > 60 seconds |
| Agent error rate | % of agent errors per hour | > 5% |
| Inter-agent msg count | Messages per task | > 20 |
| Cost per task | LLM spend per completed task | > budget |
| Queue depth | Pending tasks in queue | > 100 |
| Agent utilization | % time agents are busy | < 30% or > 90% |

---

## 10. Case Studies

### Case Study 1: Content Production Pipeline

**Setup**: 5-agent pipeline for blog content creation

```
User Request --> Researcher --> Outliner --> Writer --> Editor --> Publisher
```

**Agents**:
- Researcher: Gathers sources, stats, quotes
- Outliner: Creates detailed article structure
- Writer: Produces first draft from outline + research
- Editor: Reviews for quality, accuracy, tone
- Publisher: Formats, adds metadata, prepares for CMS

**Results**: 80% reduction in content production time. Quality improved 15% (measured by engagement metrics) due to consistent editorial review.

### Case Study 2: Customer Support Triage

**Setup**: 3-agent system for support ticket handling

```
Ticket --> Classifier --> [Router] --> Specialist Agents
                                       |-- Billing Agent
                                       |-- Technical Agent
                                       |-- General Agent
```

**Results**: 65% of tickets resolved without human intervention. Average resolution time dropped from 4 hours to 12 minutes.

### Case Study 3: Code Review System

**Setup**: 4-agent code review team

```
PR Submitted --> Security Reviewer
             --> Performance Reviewer
             --> Style Reviewer
             --> Synthesizer --> Final Review Report
```

**Results**: Caught 40% more issues than single-agent review. Reduced false positives by 25% through multi-perspective analysis.

---

## Summary

Multi-Agent Systems extend the capabilities of individual agents by enabling specialization, parallelism, and collaborative problem-solving. Key principles:

1. **Choose the right architecture**: Centralized for simplicity, decentralized for resilience, hierarchical for large teams
2. **Design clear communication protocols**: Structured messages, defined schemas, error handling
3. **Specialize agents**: Give each agent a focused role with specific tools and prompts
4. **Handle failures gracefully**: Retries, fallbacks, error propagation control
5. **Monitor everything**: Trace inter-agent communication, measure costs, track quality
6. **Start simple**: Begin with 2-3 agents and scale up as needed
7. **Manage costs**: Use model tiering, caching, and early termination
8. **Test thoroughly**: Test individual agents, communication paths, and end-to-end workflows
