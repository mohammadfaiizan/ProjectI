# Memory and State Management for AI Agents

## Table of Contents

1. Why Memory Matters for Agents
2. Types of Memory
3. Conversation History Management
4. Context Window Management
5. State Management
6. Memory Storage Backends
7. Memory Consolidation and Pruning
8. Advanced Memory Patterns
9. Implementation Examples
10. Production Memory Architecture

---

## 1. Why Memory Matters for Agents

### Stateless vs Stateful Agents

A stateless agent treats every interaction as independent. It has no recollection of past conversations, decisions, or outcomes. A stateful agent maintains context across interactions, enabling continuity, personalization, and learning.

```
STATELESS AGENT:
User: "My name is Alice"      --> Agent: "Hello Alice!"
User: "What's my name?"       --> Agent: "I don't know your name."

STATEFUL AGENT:
User: "My name is Alice"      --> Agent: "Hello Alice!" [stores: name=Alice]
User: "What's my name?"       --> Agent: "Your name is Alice."
```

### Memory as a Competitive Advantage

| Capability | Without Memory | With Memory |
|-----------|---------------|-------------|
| Personalization | Generic responses | Tailored to user preferences |
| Context continuity | Repeats questions | Remembers past discussions |
| Learning from errors | Same mistakes | Avoids repeated failures |
| Complex tasks | Single-turn only | Multi-session projects |
| Relationship building | None | Deepening understanding |
| Efficiency | Redundant work | Builds on prior results |

### Human Memory Analogy

```
+------------------------------------------------------------------+
|                    HUMAN MEMORY MODEL                             |
|                                                                   |
|  Sensory Memory     Short-Term Memory     Long-Term Memory       |
|  (milliseconds)     (seconds-minutes)     (days-lifetime)        |
|                                                                   |
|  +----------+       +-------------+       +-----------------+    |
|  | Visual,  | ----> | Working     | ----> | Declarative     |    |
|  | Auditory |       | Memory      |       | (Facts, Events) |    |
|  | Input    |       | (7 +/- 2   |       |                 |    |
|  |          |       |  items)     |       | Procedural      |    |
|  +----------+       +-------------+       | (Skills, Habits)|    |
|                           |               +-----------------+    |
|                           v                                       |
|                     Forgotten if                                  |
|                     not rehearsed                                  |
+------------------------------------------------------------------+

+------------------------------------------------------------------+
|                    AGENT MEMORY MODEL                             |
|                                                                   |
|  Input Buffer       Working Context       Persistent Store       |
|  (current turn)     (conversation)        (cross-session)        |
|                                                                   |
|  +----------+       +-------------+       +-----------------+    |
|  | User     | ----> | Conversation| ----> | Semantic Memory |    |
|  | Message, |       | History,    |       | (Vector Store)  |    |
|  | Tool     |       | Current     |       |                 |    |
|  | Results  |       | Plan/State  |       | Episodic Memory |    |
|  +----------+       +-------------+       | (Event Log)     |    |
|                           |               |                 |    |
|                           v               | Procedural Mem  |    |
|                     Truncated when        | (Learned Procs) |    |
|                     context full          +-----------------+    |
+------------------------------------------------------------------+
```

---

## 2. Types of Memory

### 2.1 Short-Term / Working Memory

Holds information relevant to the current task or conversation. Typically stored in the LLM's context window.

```python
class Short_Term_Memory:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def Add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.Trim_If_Needed()

    def Trim_If_Needed(self):
        while self.Count_Tokens() > self.max_tokens:
            # Remove oldest messages (keep system prompt)
            if len(self.messages) > 1:
                self.messages.pop(1)  # index 0 is system prompt

    def Count_Tokens(self):
        total = 0
        for msg in self.messages:
            total += len(msg["content"]) // 4  # rough estimate
        return total

    def Get_Messages(self):
        return self.messages.copy()

    def Clear(self):
        system = self.messages[0] if self.messages else None
        self.messages = [system] if system else []
```

### 2.2 Long-Term Memory

Persists information across sessions. Stored in external databases.

```python
import json
import sqlite3
from datetime import datetime

class Long_Term_Memory:
    def __init__(self, db_path="agent_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.Create_Tables()

    def Create_Tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def Store(self, key, value, category="general", importance=0.5):
        self.conn.execute(
            "INSERT INTO memories (key, value, category, importance) VALUES (?, ?, ?, ?)",
            (key, json.dumps(value), category, importance)
        )
        self.conn.commit()

    def Retrieve(self, key=None, category=None, limit=10):
        query = "SELECT key, value, importance FROM memories WHERE 1=1"
        params = []

        if key:
            query += " AND key LIKE ?"
            params.append(f"%{key}%")
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [{"key": r[0], "value": json.loads(r[1]), "importance": r[2]} for r in rows]

    def Update_Access(self, memory_id):
        self.conn.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (datetime.now(), memory_id)
        )
        self.conn.commit()
```

### 2.3 Episodic Memory

Records specific past experiences and events. Enables the agent to recall "what happened" in previous interactions.

```python
class Episodic_Memory:
    def __init__(self):
        self.episodes = []

    def Record_Episode(self, task, actions, outcome, feedback=None):
        episode = {
            "id": len(self.episodes),
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": outcome.get("success", False),
            "feedback": feedback,
            "lessons_learned": self.Extract_Lessons(actions, outcome),
        }
        self.episodes.append(episode)
        return episode

    def Recall_Similar(self, current_task, top_k=3):
        # In production, use vector similarity search
        scored = []
        for ep in self.episodes:
            similarity = self.Compute_Similarity(current_task, ep["task"])
            scored.append((similarity, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def Recall_Failures(self, task_type=None):
        return [
            ep for ep in self.episodes
            if not ep["success"]
            and (task_type is None or task_type in ep["task"])
        ]

    def Extract_Lessons(self, actions, outcome):
        # Analyze what worked and what did not
        lessons = []
        if outcome.get("success"):
            lessons.append(f"Successful approach: {actions[-1]}")
        else:
            lessons.append(f"Failed approach to avoid: {actions[-1]}")
        return lessons
```

### 2.4 Semantic Memory

Stores factual knowledge, entity relationships, and learned concepts. Often backed by vector databases for semantic search.

```python
class Semantic_Memory:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def Store_Fact(self, fact, metadata=None):
        embedding = self.embedding_model.embed(fact)
        self.vector_store.add(
            embedding=embedding,
            document=fact,
            metadata=metadata or {}
        )

    def Recall(self, query, top_k=5, min_similarity=0.7):
        query_embedding = self.embedding_model.embed(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
        return [r for r in results if r["similarity"] >= min_similarity]

    def Store_Entity(self, entity_name, attributes):
        fact = f"Entity: {entity_name}. Attributes: {json.dumps(attributes)}"
        self.Store_Fact(fact, metadata={"type": "entity", "name": entity_name})

    def Recall_Entity(self, entity_name):
        return self.Recall(f"Entity: {entity_name}", top_k=1)
```

### 2.5 Procedural Memory

Stores learned workflows, successful strategies, and reusable procedures.

```python
class Procedural_Memory:
    def __init__(self):
        self.procedures = {}
        self.success_rates = {}

    def Learn_Procedure(self, task_type, steps, success):
        if task_type not in self.procedures:
            self.procedures[task_type] = []
            self.success_rates[task_type] = {"attempts": 0, "successes": 0}

        self.procedures[task_type].append({
            "steps": steps,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

        self.success_rates[task_type]["attempts"] += 1
        if success:
            self.success_rates[task_type]["successes"] += 1

    def Get_Best_Procedure(self, task_type):
        if task_type not in self.procedures:
            return None

        successful = [
            p for p in self.procedures[task_type] if p["success"]
        ]
        if not successful:
            return None

        # Return most recent successful procedure
        return successful[-1]["steps"]

    def Get_Success_Rate(self, task_type):
        if task_type not in self.success_rates:
            return 0.0
        stats = self.success_rates[task_type]
        if stats["attempts"] == 0:
            return 0.0
        return stats["successes"] / stats["attempts"]
```

### Memory Type Comparison

| Memory Type | Duration | Storage | Access Pattern | Example |
|-------------|----------|---------|----------------|---------|
| Short-Term | Current session | LLM context | Sequential | Current conversation |
| Long-Term | Permanent | Database | Key/query based | User preferences |
| Episodic | Permanent | Database/JSON | Similarity search | Past task outcomes |
| Semantic | Permanent | Vector store | Semantic search | Facts and knowledge |
| Procedural | Permanent | Database | Task-type lookup | Learned workflows |

---

## 3. Conversation History Management

### 3.1 Full History

Store and send the entire conversation history to the LLM. Simple but limited by context window.

```python
class Full_History_Memory:
    def __init__(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]

    def Add_User_Message(self, content):
        self.messages.append({"role": "user", "content": content})

    def Add_Assistant_Message(self, content):
        self.messages.append({"role": "assistant", "content": content})

    def Get_Messages(self):
        return self.messages
```

**Pros**: Complete context, no information loss
**Cons**: Hits context window limits quickly, expensive

### 3.2 Sliding Window

Keep only the last N messages.

```python
class Sliding_Window_Memory:
    def __init__(self, system_prompt, window_size=20):
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.messages = []
        self.window_size = window_size

    def Add_Message(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def Get_Messages(self):
        return [self.system_prompt] + self.messages
```

**Pros**: Bounded memory usage, predictable cost
**Cons**: Loses older context abruptly

### 3.3 Summary Memory

Periodically summarize older messages and replace them with the summary.

```python
class Summary_Memory:
    def __init__(self, llm, system_prompt, summary_threshold=10):
        self.llm = llm
        self.system_prompt = system_prompt
        self.summary = ""
        self.recent_messages = []
        self.summary_threshold = summary_threshold

    def Add_Message(self, role, content):
        self.recent_messages.append({"role": role, "content": content})

        if len(self.recent_messages) >= self.summary_threshold:
            self.Consolidate()

    def Consolidate(self):
        messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.recent_messages
        )
        new_summary = self.llm.generate(f"""
        Existing Summary:
        {self.summary}

        New Messages:
        {messages_text}

        Create an updated summary that captures all important information,
        decisions, and context from both the existing summary and new messages.
        Be concise but preserve key details like names, numbers, and decisions.
        """)
        self.summary = new_summary
        self.recent_messages = []

    def Get_Messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation Summary:\n{self.summary}"
            })
        messages.extend(self.recent_messages)
        return messages
```

### 3.4 Token-Aware Truncation

Intelligently truncate based on token counts and message importance.

```python
import tiktoken

class Token_Aware_Memory:
    def __init__(self, system_prompt, max_tokens=3000, model="gpt-4"):
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.messages = []
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model)

    def Count_Message_Tokens(self, message):
        return len(self.encoder.encode(message["content"])) + 4

    def Add_Message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def Get_Messages(self):
        system_tokens = self.Count_Message_Tokens(self.system_prompt)
        budget = self.max_tokens - system_tokens
        selected = []

        # Always include most recent messages first
        for msg in reversed(self.messages):
            msg_tokens = self.Count_Message_Tokens(msg)
            if budget >= msg_tokens:
                selected.insert(0, msg)
                budget -= msg_tokens
            else:
                break

        return [self.system_prompt] + selected
```

### 3.5 Hybrid: Sliding Window + Summary

The most practical approach for production agents.

```python
class Hybrid_Memory:
    def __init__(self, llm, system_prompt, window_size=10, max_tokens=4000):
        self.llm = llm
        self.system_prompt = system_prompt
        self.summary = ""
        self.window = []
        self.window_size = window_size
        self.max_tokens = max_tokens

    def Add_Message(self, role, content):
        self.window.append({"role": role, "content": content})

        if len(self.window) > self.window_size:
            overflow = self.window[: len(self.window) - self.window_size]
            self.window = self.window[-self.window_size:]
            self.Update_Summary(overflow)

    def Update_Summary(self, messages):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        self.summary = self.llm.generate(
            f"Previous summary:\n{self.summary}\n\n"
            f"New messages:\n{text}\n\n"
            f"Create updated summary preserving all key information."
        )

    def Get_Messages(self):
        result = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            result.append({
                "role": "system",
                "content": f"Prior context summary:\n{self.summary}"
            })
        result.extend(self.window)
        return result
```

### Comparison of History Strategies

| Strategy | Token Usage | Info Loss | Cost | Complexity | Best For |
|----------|------------|-----------|------|------------|----------|
| Full History | Grows unbounded | None | High | Low | Short conversations |
| Sliding Window | Fixed | High (older) | Low | Low | Simple chatbots |
| Summary | Low | Medium | Medium | Medium | Long conversations |
| Token-Aware | Fixed | Medium | Low | Medium | Cost-sensitive apps |
| Hybrid | Bounded | Low | Medium | High | Production agents |

---

## 4. Context Window Management

### Token Budgeting

Allocate the context window across different information sources.

```python
class Context_Budget:
    def __init__(self, total_tokens=8000):
        self.total = total_tokens
        self.allocations = {
            "system_prompt": 500,       # 6%
            "memory_summary": 1000,     # 12.5%
            "retrieved_context": 2000,  # 25%
            "conversation": 2500,       # 31%
            "tool_results": 1000,       # 12.5%
            "output_reserve": 1000,     # 12.5%
        }

    def Get_Budget(self, section):
        return self.allocations.get(section, 0)

    def Remaining(self):
        used = sum(self.allocations.values())
        return self.total - used

    def Adjust(self, section, new_value):
        old = self.allocations.get(section, 0)
        self.allocations[section] = new_value
        # Reduce other sections proportionally if over budget
        if sum(self.allocations.values()) > self.total:
            excess = sum(self.allocations.values()) - self.total
            other_sections = [s for s in self.allocations if s != section]
            for s in other_sections:
                reduction = excess * (self.allocations[s] / sum(
                    self.allocations[o] for o in other_sections
                ))
                self.allocations[s] = max(100, self.allocations[s] - reduction)
```

### Priority-Based Context Packing

```python
class Context_Assembler:
    def __init__(self, budget, encoder):
        self.budget = budget
        self.encoder = encoder

    def Assemble(self, system_prompt, memory, rag_results, conversation, tool_results):
        context_parts = []
        remaining = self.budget.total

        # Priority 1: System prompt (always included)
        sys_tokens = self.Count_Tokens(system_prompt)
        context_parts.append(("system", system_prompt, sys_tokens))
        remaining -= sys_tokens

        # Priority 2: Output reserve
        remaining -= self.budget.Get_Budget("output_reserve")

        # Priority 3: Recent conversation (most recent first)
        conv_budget = min(remaining, self.budget.Get_Budget("conversation"))
        conv_text, conv_tokens = self.Fit_To_Budget(conversation, conv_budget)
        context_parts.append(("conversation", conv_text, conv_tokens))
        remaining -= conv_tokens

        # Priority 4: Tool results (if any pending)
        if tool_results:
            tool_budget = min(remaining, self.budget.Get_Budget("tool_results"))
            tool_text, tool_tokens = self.Fit_To_Budget(tool_results, tool_budget)
            context_parts.append(("tools", tool_text, tool_tokens))
            remaining -= tool_tokens

        # Priority 5: RAG context
        if rag_results:
            rag_budget = min(remaining, self.budget.Get_Budget("retrieved_context"))
            rag_text, rag_tokens = self.Fit_To_Budget(rag_results, rag_budget)
            context_parts.append(("rag", rag_text, rag_tokens))
            remaining -= rag_tokens

        # Priority 6: Memory summary
        if memory:
            mem_budget = min(remaining, self.budget.Get_Budget("memory_summary"))
            mem_text, mem_tokens = self.Fit_To_Budget(memory, mem_budget)
            context_parts.append(("memory", mem_text, mem_tokens))

        return context_parts

    def Count_Tokens(self, text):
        return len(self.encoder.encode(text))

    def Fit_To_Budget(self, text, budget):
        tokens = self.Count_Tokens(text)
        if tokens <= budget:
            return text, tokens
        # Truncate to fit
        encoded = self.encoder.encode(text)[:budget]
        return self.encoder.decode(encoded), budget
```

### Dynamic Context Assembly Pattern

```
+-------------------------------------------------------------------+
| CONTEXT WINDOW (e.g., 8,192 tokens)                              |
|                                                                   |
| +------------------+ +----------------+ +-----------+             |
| | System Prompt    | | Memory Summary | | RAG Docs  |             |
| | (500 tokens)     | | (800 tokens)   | | (2000 tok)|             |
| +------------------+ +----------------+ +-----------+             |
|                                                                   |
| +-------------------------------+ +------------+ +----------+    |
| | Conversation History          | | Tool       | | Output   |    |
| | (3000 tokens)                 | | Results    | | Reserve  |    |
| |                               | | (900 tok)  | | (992 tok)|    |
| +-------------------------------+ +------------+ +----------+    |
+-------------------------------------------------------------------+
```

---

## 5. State Management

### 5.1 Agent State Machine

```python
from enum import Enum

class Agent_State(Enum):
    IDLE = "idle"
    RECEIVING = "receiving"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    EVALUATING = "evaluating"
    RESPONDING = "responding"
    ERROR = "error"

class State_Machine:
    def __init__(self):
        self.current_state = Agent_State.IDLE
        self.state_data = {}
        self.transitions = {
            Agent_State.IDLE: [Agent_State.RECEIVING],
            Agent_State.RECEIVING: [Agent_State.PLANNING, Agent_State.RESPONDING],
            Agent_State.PLANNING: [Agent_State.EXECUTING, Agent_State.ERROR],
            Agent_State.EXECUTING: [
                Agent_State.WAITING_FOR_TOOL,
                Agent_State.EVALUATING,
                Agent_State.ERROR,
            ],
            Agent_State.WAITING_FOR_TOOL: [Agent_State.EXECUTING, Agent_State.ERROR],
            Agent_State.EVALUATING: [
                Agent_State.RESPONDING,
                Agent_State.PLANNING,  # re-plan if not satisfactory
            ],
            Agent_State.RESPONDING: [Agent_State.IDLE],
            Agent_State.ERROR: [Agent_State.IDLE, Agent_State.PLANNING],
        }

    def Transition(self, new_state, data=None):
        if new_state not in self.transitions.get(self.current_state, []):
            raise ValueError(
                f"Invalid transition: {self.current_state} -> {new_state}"
            )
        self.current_state = new_state
        if data:
            self.state_data.update(data)

    def Get_State(self):
        return self.current_state

    def Get_Data(self):
        return self.state_data.copy()
```

### 5.2 Checkpointing and Resumption

```python
import json
from datetime import datetime

class Checkpoint_Manager:
    def __init__(self, storage_path="checkpoints"):
        self.storage_path = storage_path

    def Save_Checkpoint(self, agent_id, state):
        checkpoint = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "state": state.Get_State().value,
            "state_data": state.Get_Data(),
            "memory_snapshot": self.Serialize_Memory(state),
            "current_plan": state.Get_Data().get("plan"),
            "completed_steps": state.Get_Data().get("completed_steps", []),
        }

        filepath = f"{self.storage_path}/{agent_id}_{checkpoint['timestamp']}.json"
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

        return filepath

    def Load_Checkpoint(self, filepath):
        with open(filepath, "r") as f:
            checkpoint = json.load(f)
        return checkpoint

    def Resume_From_Checkpoint(self, agent, checkpoint):
        agent.state.current_state = Agent_State(checkpoint["state"])
        agent.state.state_data = checkpoint["state_data"]
        agent.memory = self.Deserialize_Memory(checkpoint["memory_snapshot"])
        return agent

    def Serialize_Memory(self, state):
        return state.Get_Data().get("memory", {})

    def Deserialize_Memory(self, snapshot):
        return snapshot
```

### 5.3 LangGraph-Style State Management

```python
from typing import TypedDict, Annotated
import operator

class Agent_Graph_State(TypedDict):
    messages: Annotated[list, operator.add]
    current_step: str
    plan: list
    results: Annotated[list, operator.add]
    iteration_count: int
    error: str

def Plan_Node(state: Agent_Graph_State) -> Agent_Graph_State:
    messages = state["messages"]
    plan = llm.generate_plan(messages)
    return {
        "plan": plan,
        "current_step": "execute",
        "iteration_count": state["iteration_count"] + 1,
    }

def Execute_Node(state: Agent_Graph_State) -> Agent_Graph_State:
    plan = state["plan"]
    current = plan[0] if plan else None

    if current is None:
        return {"current_step": "respond"}

    result = execute_tool(current)
    remaining = plan[1:]

    return {
        "plan": remaining,
        "results": [result],
        "current_step": "execute" if remaining else "evaluate",
    }

def Evaluate_Node(state: Agent_Graph_State) -> Agent_Graph_State:
    results = state["results"]
    quality = llm.evaluate(results)

    if quality["score"] >= 0.8:
        return {"current_step": "respond"}
    else:
        return {"current_step": "plan"}  # re-plan

def Route(state: Agent_Graph_State) -> str:
    return state["current_step"]
```

### 5.4 Persistent State with Redis

```python
import redis
import json

class Redis_State_Store:
    def __init__(self, host="localhost", port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def Save_State(self, session_id, state, ttl=3600):
        key = f"agent:state:{session_id}"
        self.client.setex(key, ttl, json.dumps(state))

    def Load_State(self, session_id):
        key = f"agent:state:{session_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def Update_State(self, session_id, updates):
        state = self.Load_State(session_id) or {}
        state.update(updates)
        self.Save_State(session_id, state)
        return state

    def Delete_State(self, session_id):
        key = f"agent:state:{session_id}"
        self.client.delete(key)

    def List_Active_Sessions(self, pattern="agent:state:*"):
        keys = self.client.keys(pattern)
        return [k.decode().split(":")[-1] for k in keys]
```

---

## 6. Memory Storage Backends

### Backend Comparison

| Backend | Latency | Scalability | Query Types | Cost | Best For |
|---------|---------|-------------|-------------|------|----------|
| In-memory (dict) | ~0.001ms | Low | Key lookup | Free | Development |
| JSON files | ~1-5ms | Low | Full scan | Free | Prototypes |
| SQLite | ~1-10ms | Medium | SQL queries | Free | Single-user agents |
| PostgreSQL | ~5-20ms | High | SQL + JSON | Medium | Production |
| Redis | ~0.5-2ms | High | Key-value | Medium | Session state |
| ChromaDB | ~10-50ms | Medium | Vector search | Free | Semantic memory |
| Pinecone | ~20-100ms | Very High | Vector search | High | Production semantic |
| Neo4j | ~10-50ms | High | Graph queries | High | Relational memory |

### In-Memory Storage

```python
class In_Memory_Store:
    def __init__(self):
        self.store = {}

    def Set(self, key, value):
        self.store[key] = {
            "value": value,
            "created": datetime.now(),
            "updated": datetime.now(),
        }

    def Get(self, key, default=None):
        entry = self.store.get(key)
        if entry:
            entry["updated"] = datetime.now()
            return entry["value"]
        return default

    def Delete(self, key):
        return self.store.pop(key, None)

    def Search(self, prefix):
        return {k: v["value"] for k, v in self.store.items() if k.startswith(prefix)}
```

### SQLite Storage

```python
class SQLite_Memory_Store:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT,
                importance REAL DEFAULT 0.5,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def Store(self, key, value, category=None, importance=0.5):
        now = datetime.now().timestamp()
        self.conn.execute(
            """INSERT OR REPLACE INTO memory
               (key, value, category, importance, created_at, accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)""",
            (key, json.dumps(value), category, importance, now, now)
        )
        self.conn.commit()

    def Retrieve(self, key):
        row = self.conn.execute(
            "SELECT value FROM memory WHERE key = ?", (key,)
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE memory SET accessed_at = ?, access_count = access_count + 1 WHERE key = ?",
                (datetime.now().timestamp(), key)
            )
            self.conn.commit()
            return json.loads(row[0])
        return None

    def Search_By_Category(self, category, limit=10):
        rows = self.conn.execute(
            "SELECT key, value FROM memory WHERE category = ? ORDER BY importance DESC LIMIT ?",
            (category, limit)
        ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}
```

### Vector Store for Semantic Memory

```python
import chromadb

class Vector_Memory_Store:
    def __init__(self, collection_name="agent_memory"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.counter = 0

    def Store(self, text, metadata=None):
        self.counter += 1
        self.collection.add(
            documents=[text],
            ids=[f"mem_{self.counter}"],
            metadatas=[metadata or {}]
        )

    def Search(self, query, top_k=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def Delete(self, memory_id):
        self.collection.delete(ids=[memory_id])
```

---

## 7. Memory Consolidation and Pruning

### 7.1 Importance Scoring

```python
class Importance_Scorer:
    def __init__(self, llm):
        self.llm = llm

    def Score(self, memory_entry):
        prompt = f"""
        Rate the importance of this memory on a scale of 0.0 to 1.0:
        Memory: {memory_entry}

        Consider:
        - Is it a key fact or decision? (high importance)
        - Is it routine/trivial? (low importance)
        - Would losing this information affect future tasks? (high if yes)
        - Is it time-sensitive? (lower importance over time)

        Return only a number between 0.0 and 1.0.
        """
        score = float(self.llm.generate(prompt).strip())
        return max(0.0, min(1.0, score))

    def Batch_Score(self, memories):
        return [(m, self.Score(m)) for m in memories]
```

### 7.2 Memory Pruning

```python
class Memory_Pruner:
    def __init__(self, max_memories=1000, prune_threshold=0.3):
        self.max_memories = max_memories
        self.prune_threshold = prune_threshold

    def Prune(self, memory_store):
        all_memories = memory_store.Get_All()

        if len(all_memories) <= self.max_memories:
            return 0

        scored = []
        for mem in all_memories:
            score = self.Calculate_Retention_Score(mem)
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0])

        to_remove = len(all_memories) - self.max_memories
        removed = 0
        for score, mem in scored:
            if score < self.prune_threshold and removed < to_remove:
                memory_store.Delete(mem["id"])
                removed += 1

        return removed

    def Calculate_Retention_Score(self, memory):
        recency = self.Recency_Score(memory["last_accessed"])
        frequency = min(1.0, memory["access_count"] / 10)
        importance = memory.get("importance", 0.5)

        # Weighted combination
        return 0.3 * recency + 0.3 * frequency + 0.4 * importance

    def Recency_Score(self, last_accessed):
        days_ago = (datetime.now() - last_accessed).days
        return max(0, 1 - (days_ago / 30))  # decays over 30 days
```

### 7.3 Memory Compaction

Merge related memories into consolidated entries.

```python
class Memory_Compactor:
    def __init__(self, llm):
        self.llm = llm

    def Compact(self, memories):
        # Group related memories
        groups = self.Cluster_Memories(memories)

        compacted = []
        for group in groups:
            if len(group) > 1:
                merged = self.Merge_Group(group)
                compacted.append(merged)
            else:
                compacted.append(group[0])

        return compacted

    def Merge_Group(self, group):
        texts = "\n".join([m["text"] for m in group])
        merged_text = self.llm.generate(
            f"Merge these related memories into a single comprehensive entry:\n{texts}"
        )
        return {
            "text": merged_text,
            "importance": max(m["importance"] for m in group),
            "source_count": len(group),
        }

    def Cluster_Memories(self, memories):
        # In production use embedding similarity clustering
        # Simplified: group by category
        groups = {}
        for m in memories:
            cat = m.get("category", "general")
            groups.setdefault(cat, []).append(m)
        return list(groups.values())
```

---

## 8. Advanced Memory Patterns

### 8.1 Hierarchical Memory (L1/L2/L3)

```
+-------------------------------------------+
|         HIERARCHICAL MEMORY               |
|                                           |
|  L1: HOT CACHE (In-Memory)               |
|  - Current conversation                   |
|  - Recent tool results                    |
|  - Active plan                            |
|  - Access time: < 1ms                     |
|  +---------------------------------------+|
|                   |                        |
|                   v (overflow)             |
|  L2: WARM STORE (Redis/SQLite)            |
|  - Session history                        |
|  - Recent episodic memories               |
|  - Entity cache                           |
|  - Access time: 1-10ms                    |
|  +---------------------------------------+|
|                   |                        |
|                   v (archive)              |
|  L3: COLD ARCHIVE (Vector DB/PostgreSQL)  |
|  - All historical memories                |
|  - Full conversation logs                 |
|  - Learned procedures                     |
|  - Access time: 10-100ms                  |
+-------------------------------------------+
```

```python
class Hierarchical_Memory:
    def __init__(self):
        self.l1_cache = {}         # In-memory dict
        self.l2_store = None       # Redis or SQLite
        self.l3_archive = None     # Vector DB or PostgreSQL

    def Get(self, key):
        # Check L1 first
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Check L2
        value = self.l2_store.Get(key)
        if value:
            self.l1_cache[key] = value  # promote to L1
            return value

        # Check L3
        value = self.l3_archive.Get(key)
        if value:
            self.l2_store.Set(key, value)  # promote to L2
            self.l1_cache[key] = value     # promote to L1
            return value

        return None

    def Set(self, key, value, tier="l1"):
        self.l1_cache[key] = value
        if tier in ("l2", "l3"):
            self.l2_store.Set(key, value)
        if tier == "l3":
            self.l3_archive.Set(key, value)

    def Evict_L1(self, max_size=100):
        if len(self.l1_cache) > max_size:
            # Move oldest to L2
            oldest = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].get("accessed", 0)
            )
            for key, value in oldest[:len(self.l1_cache) - max_size]:
                self.l2_store.Set(key, value)
                del self.l1_cache[key]
```

### 8.2 Shared Memory in Multi-Agent Systems

```python
class Shared_Memory:
    def __init__(self):
        self.store = {}
        self.locks = {}
        self.subscribers = {}

    def Write(self, agent_id, key, value):
        self.store[key] = {
            "value": value,
            "author": agent_id,
            "timestamp": datetime.now(),
        }
        self.Notify_Subscribers(key, value, agent_id)

    def Read(self, key):
        entry = self.store.get(key)
        return entry["value"] if entry else None

    def Subscribe(self, agent_id, key_pattern, callback):
        self.subscribers.setdefault(key_pattern, []).append(
            {"agent_id": agent_id, "callback": callback}
        )

    def Notify_Subscribers(self, key, value, author):
        for pattern, subs in self.subscribers.items():
            if key.startswith(pattern):
                for sub in subs:
                    if sub["agent_id"] != author:
                        sub["callback"](key, value, author)
```

### 8.3 Memory Reflection

Agent reviews its own memories to extract insights and improve behavior.

```python
class Memory_Reflector:
    def __init__(self, llm, episodic_memory):
        self.llm = llm
        self.episodic = episodic_memory

    def Reflect(self, recent_episodes=10):
        episodes = self.episodic.Get_Recent(recent_episodes)

        reflection = self.llm.generate(f"""
        Review these recent agent experiences:
        {json.dumps(episodes, indent=2)}

        Provide reflections on:
        1. Patterns in successful vs failed tasks
        2. Common mistakes to avoid
        3. Strategies that worked well
        4. Areas for improvement
        5. Key insights to remember

        Format as a list of actionable insights.
        """)

        return {
            "insights": reflection,
            "timestamp": datetime.now().isoformat(),
            "episode_count": len(episodes),
        }
```

### 8.4 Temporal Memory

Time-aware recall with decay.

```python
class Temporal_Memory:
    def __init__(self, decay_rate=0.1):
        self.memories = []
        self.decay_rate = decay_rate

    def Store(self, content, importance=0.5):
        self.memories.append({
            "content": content,
            "importance": importance,
            "timestamp": datetime.now(),
            "base_strength": importance,
        })

    def Recall(self, query=None, top_k=5):
        now = datetime.now()
        scored = []
        for mem in self.memories:
            hours_ago = (now - mem["timestamp"]).total_seconds() / 3600
            decay = mem["base_strength"] * (2.718 ** (-self.decay_rate * hours_ago))
            effective_strength = decay * mem["importance"]
            scored.append((effective_strength, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, m["content"]) for s, m in scored[:top_k]]
```

---

## 9. Implementation Examples

### Complete Memory System

```python
class Complete_Agent_Memory:
    def __init__(self, llm, config=None):
        config = config or {}
        self.llm = llm

        # Short-term: conversation context
        self.short_term = Hybrid_Memory(
            llm=llm,
            system_prompt=config.get("system_prompt", "You are a helpful agent."),
            window_size=config.get("window_size", 10),
        )

        # Long-term: persistent facts
        self.long_term = SQLite_Memory_Store(
            db_path=config.get("db_path", "agent_memory.db")
        )

        # Semantic: vector-based recall
        self.semantic = Vector_Memory_Store(
            collection_name=config.get("collection", "agent_semantic")
        )

        # Episodic: past experiences
        self.episodic = Episodic_Memory()

        # Procedural: learned workflows
        self.procedural = Procedural_Memory()

    def Remember(self, content, memory_type="short_term", **kwargs):
        if memory_type == "short_term":
            self.short_term.Add_Message(kwargs.get("role", "user"), content)
        elif memory_type == "long_term":
            self.long_term.Store(kwargs.get("key", content[:50]), content)
        elif memory_type == "semantic":
            self.semantic.Store(content, kwargs.get("metadata"))
        elif memory_type == "episodic":
            self.episodic.Record_Episode(
                task=content,
                actions=kwargs.get("actions", []),
                outcome=kwargs.get("outcome", {}),
            )

    def Recall(self, query, memory_types=None):
        memory_types = memory_types or ["short_term", "semantic", "episodic"]
        results = {}

        if "short_term" in memory_types:
            results["conversation"] = self.short_term.Get_Messages()

        if "semantic" in memory_types:
            results["knowledge"] = self.semantic.Search(query, top_k=5)

        if "episodic" in memory_types:
            results["past_experiences"] = self.episodic.Recall_Similar(query, top_k=3)

        if "procedural" in memory_types:
            results["procedures"] = self.procedural.Get_Best_Procedure(query)

        return results

    def Build_Context(self, query):
        memories = self.Recall(query)

        context = "Relevant Context:\n\n"

        if memories.get("knowledge"):
            context += "Knowledge:\n"
            for item in memories["knowledge"]:
                context += f"- {item['text']}\n"
            context += "\n"

        if memories.get("past_experiences"):
            context += "Past Experiences:\n"
            for ep in memories["past_experiences"]:
                context += f"- Task: {ep['task']}, Outcome: {ep['outcome']}\n"
            context += "\n"

        if memories.get("procedures"):
            context += f"Recommended Procedure:\n{memories['procedures']}\n\n"

        return context
```

---

## 10. Production Memory Architecture

### Architecture Diagram

```
+------------------------------------------------------------------+
|                    PRODUCTION MEMORY SYSTEM                       |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | API Gateway      |--->| Agent Runtime    |                     |
|  | (User requests)  |    | (LLM + Logic)   |                     |
|  +------------------+    +--------+---------+                     |
|                                   |                               |
|                    +--------------+---------------+               |
|                    |              |               |               |
|            +-------v------+ +----v-------+ +-----v------+       |
|            | Redis        | | PostgreSQL | | Vector DB  |       |
|            | (Session     | | (Long-term | | (Semantic  |       |
|            |  state,      | |  memory,   | |  memory,   |       |
|            |  L1 cache)   | |  episodes, | |  knowledge)|       |
|            |              | |  procedures)| |            |       |
|            +--------------+ +------------+ +------------+       |
|                                                                   |
|  +------------------+    +------------------+                     |
|  | Background Jobs  |    | Monitoring       |                     |
|  | - Consolidation  |    | - Memory usage   |                     |
|  | - Pruning        |    | - Hit rates      |                     |
|  | - Compaction     |    | - Latency        |                     |
|  +------------------+    +------------------+                     |
+------------------------------------------------------------------+
```

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| Memory hit rate | % of queries that find relevant memories | > 80% |
| Retrieval latency | Time to fetch memories | < 100ms |
| Storage growth | Rate of new memories per day | Bounded |
| Consolidation rate | Memories consolidated per cycle | 10-20% |
| Context utilization | % of context window used effectively | 60-85% |
| Memory accuracy | Relevance of recalled memories | > 90% |

### Best Practices

1. **Layer your memory**: Use hierarchical storage (hot/warm/cold) for optimal performance
2. **Set retention policies**: Not all memories need to last forever
3. **Score importance**: Prioritize high-value memories in context assembly
4. **Consolidate regularly**: Merge related memories to reduce redundancy
5. **Monitor usage**: Track hit rates and retrieval quality
6. **Test memory**: Validate that the right memories surface for given queries
7. **Handle conflicts**: Define strategies for contradictory memories
8. **Secure sensitive data**: Encrypt memories containing PII or sensitive information
9. **Version your schemas**: Memory format will evolve; plan for migrations
10. **Benchmark context assembly**: Measure how different memory strategies affect agent quality
