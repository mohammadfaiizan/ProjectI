# State Persistence and Checkpointing

## Distinguishing State from Memory

It's easy to conflate "memory" and "state" because both involve carrying information forward in time, but they answer different questions and, in a well-designed agent, are handled by different mechanisms. Memory (the subject of the previous three chapters) answers "what does the agent know" — facts, past episodes, learned procedures, conversational context. State answers "where is the agent right now in the middle of doing something" — which step of a multi-step plan it has completed, which tool calls are still pending, what intermediate results it's accumulated for the task in progress, and what it needs to do next if execution is interrupted this instant.

The distinction matters because state has a requirement memory usually doesn't: it needs to support exact, faithful resumption. If an agent's memory of a fact is slightly stale, the consequence is a slightly worse answer. If an agent's *state* is lost or corrupted mid-execution — say, three tool calls into a five-call chain that includes a non-idempotent payment charge — the consequence can be a duplicated charge, an orphaned resource, or a task that silently never completes. State persistence is fundamentally an engineering-for-reliability problem, closer to how you'd think about durability in a workflow engine or a distributed job scheduler than to how you'd think about a chatbot's memory of your name.

## Why Agents Need Durable State

Three realities of running agents in production make in-memory-only state untenable. The first is that agent tasks are increasingly long-running: a research agent might spend twenty minutes iterating through searches and syntheses; a coding agent might work through a multi-file refactor over several minutes of tool calls; an agent that waits on a human approval step, or on an external system to finish an async job, might be "in progress" for hours or days. None of these can survive purely in a process's memory, because processes restart — deployments happen, containers get rescheduled, machines crash — and a task that was 90% complete when the process died should not have to start over from zero.

The second reality is that resilience requires you to assume failure will happen mid-step, not just between steps. A crash can occur after a tool call has already had its real-world side effect (an email sent, a record written, a payment charged) but before the agent recorded that the step succeeded. Durable state, checkpointed at the right granularity, is what allows a resumed agent to know precisely what already happened and avoid redoing side-effecting work — this is the same problem distributed systems call exactly-once (or more realistically, effectively-once) execution.

The third reality is operational: durable, inspectable state is what makes an agent debuggable and auditable in production. When something goes wrong, you want to be able to look at exactly what state the agent was in at each step, not just log lines — that's the difference between guessing what happened and replaying it.

## The Checkpoint Model

A checkpoint is a durable, point-in-time snapshot of everything needed to resume an agent's execution from that exact point, as if nothing had been interrupted. At minimum, a checkpoint needs: the current values of the agent's working state (its plan, accumulated results, loop counters — whatever the orchestration logic uses to decide what to do next), a marker for where in the execution graph the agent currently is (which node or step it just completed or is about to run), and enough identifying information (a thread or session id, a timestamp, a checkpoint id) to locate and order checkpoints later.

```python
import json
import uuid
from datetime import datetime

class Checkpoint:
    def __init__(self, thread_id: str, step_name: str, state: dict, checkpoint_id: str = None):
        self.checkpoint_id = checkpoint_id or str(uuid.uuid4())
        self.thread_id = thread_id          # identifies the logical run / conversation
        self.step_name = step_name          # where execution was when this was saved
        self.state = state                  # the actual data needed to resume
        self.created_at = datetime.now().isoformat()

    def to_json(self):
        return json.dumps({
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "step_name": self.step_name,
            "state": self.state,
            "created_at": self.created_at,
        })

    @classmethod
    def from_json(cls, raw: str):
        data = json.loads(raw)
        cp = cls(data["thread_id"], data["step_name"], data["state"], data["checkpoint_id"])
        cp.created_at = data["created_at"]
        return cp
```

A key decision is checkpoint granularity: how often do you snapshot? Checkpointing after every single tool call maximizes resumability (you can never lose more than one step of work) but adds overhead and storage volume; checkpointing only at coarse milestones (say, after each major phase of a plan) is cheaper but means a crash mid-phase loses more progress and has to redo more (potentially non-idempotent) work. Most graph-based agent frameworks resolve this by checkpointing after every discrete unit of execution — commonly called a "superstep," borrowing the term from graph-processing systems — which in practice means one checkpoint per node execution in the agent's control-flow graph. This gives fine-grained resumability without requiring the developer to manually decide where checkpoints belong.

```python
class Checkpoint_Store:
    """A minimal durable store; swap the backend (SQLite/Postgres/Redis) in production."""

    def __init__(self, conn):
        self.conn = conn
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save(self, checkpoint: Checkpoint):
        self.conn.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (checkpoint.checkpoint_id, checkpoint.thread_id, checkpoint.step_name,
             json.dumps(checkpoint.state), checkpoint.created_at),
        )
        self.conn.commit()

    def latest_for_thread(self, thread_id: str):
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if not row:
            return None
        return Checkpoint(row[1], row[2], json.loads(row[3]), row[0])

    def history_for_thread(self, thread_id: str):
        rows = self.conn.execute(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()
        return [Checkpoint(r[1], r[2], json.loads(r[3]), r[0]) for r in rows]
```

## The LangGraph-Style Checkpointer Model

LangGraph's approach to state persistence is a useful reference model because it cleanly separates *what the agent computes* from *how that computation is made durable*, and that separation is a good pattern to copy even if you're not using LangGraph itself. In this model, you define your agent as a graph of nodes (functions that take the current state and return updates to it) connected by edges that determine what runs next. Separately, you attach a **checkpointer** — a pluggable persistence backend (an in-memory saver for development, a SQLite- or Postgres-backed saver for production) — to the graph. The graph runtime, not your node functions, is responsible for calling the checkpointer to persist state after every superstep.

```python
from typing import TypedDict

class Agent_State(TypedDict):
    messages: list
    plan: list
    completed_steps: list
    result: str | None

def plan_node(state: Agent_State) -> dict:
    # returns only the *updates* to state, not the whole state
    return {"plan": ["search", "summarize", "respond"]}

def execute_node(state: Agent_State) -> dict:
    step = state["plan"][0]
    remaining = state["plan"][1:]
    output = run_tool(step)   # side-effecting work happens here
    return {
        "plan": remaining,
        "completed_steps": state["completed_steps"] + [step],
        "messages": state["messages"] + [{"role": "tool", "content": output}],
    }

class Graph_Runtime:
    """A simplified stand-in for what LangGraph's executor does under the hood."""

    def __init__(self, nodes: dict, edges: dict, checkpointer):
        self.nodes = nodes            # name -> function(state) -> partial state update
        self.edges = edges            # name -> next node name (or a routing function)
        self.checkpointer = checkpointer

    def run(self, thread_id: str, initial_state: Agent_State, start_node: str = "plan"):
        checkpoint = self.checkpointer.latest_for_thread(thread_id)
        if checkpoint:
            state, current_node = checkpoint.state, checkpoint.step_name
        else:
            state, current_node = dict(initial_state), start_node

        while current_node != "END":
            update = self.nodes[current_node](state)
            state = {**state, **update}
            next_node = self.edges[current_node](state) if callable(self.edges[current_node]) else self.edges[current_node]

            # persist AFTER the node runs, BEFORE moving to the next one --
            # this is the superstep checkpoint
            self.checkpointer.save(Checkpoint(thread_id, next_node, state))
            current_node = next_node

        return state
```

The `thread_id` is the crucial concept: it's the durable key that ties together every checkpoint belonging to one logical run (one conversation, one task), and it's what lets a completely new process — after a redeploy, a crash, or simply a new incoming request hours later — pick up exactly where a previous process left off, just by passing the same `thread_id` and letting the runtime load the latest checkpoint before doing anything else.

This model directly enables two capabilities that are otherwise hard to bolt on after the fact. The first is **human-in-the-loop interruption**: a node can be marked to pause execution before or after it runs, persist the current checkpoint, and return control to the caller — the run simply stops advancing until something explicitly resumes it (for example, a human approving a pending action). Because the state was already checkpointed, resumption is not a special code path; it's the exact same "load latest checkpoint for this thread and continue" logic used for crash recovery.

```python
def resume_after_human_approval(runtime: Graph_Runtime, thread_id: str, approved: bool):
    checkpoint = runtime.checkpointer.latest_for_thread(thread_id)
    state = checkpoint.state
    state["human_approved"] = approved
    # continue running the graph from the node the checkpoint left off at
    return runtime.run(thread_id, state, start_node=checkpoint.step_name)
```

The second capability is **time travel**: because every superstep is checkpointed rather than just the latest state, you can fetch the full history for a thread, pick an earlier checkpoint, and fork a new run from that point — useful for debugging ("replay from right before the failure and see what happens with different tool output") and for exploring alternative paths without discarding the original run.

```python
def fork_from_checkpoint(runtime: Graph_Runtime, thread_id: str, checkpoint_id: str, new_thread_id: str):
    history = runtime.checkpointer.history_for_thread(thread_id)
    target = next(c for c in history if c.checkpoint_id == checkpoint_id)
    forked = Checkpoint(new_thread_id, target.step_name, dict(target.state))
    runtime.checkpointer.save(forked)
    return runtime.run(new_thread_id, forked.state, start_node=forked.step_name)
```

## Resuming Interrupted Runs Safely

Loading the last checkpoint and continuing is the easy half of resumption; the hard half is making sure the step that was interrupted mid-execution doesn't get incorrectly re-run in a way that causes harm. The core issue is that a checkpoint is saved *after* a node completes, which means if the process crashes *during* a node's execution — after a side effect has already happened (an API call went through) but before the checkpoint recording that fact was written — a naive resume will re-enter that same node and repeat the side effect.

The standard fix is to make side-effecting steps idempotent, typically by attaching a stable, deterministic idempotency key to any external action, so that re-executing the same logical step is safe even if it happens twice.

```python
class Idempotent_Tool_Executor:
    def __init__(self, tool_fn, dedup_store):
        self.tool_fn = tool_fn
        self.dedup_store = dedup_store    # tracks {idempotency_key: result}

    def execute(self, idempotency_key: str, **kwargs):
        cached = self.dedup_store.get(idempotency_key)
        if cached is not None:
            return cached   # already happened; return the recorded result, don't repeat it

        result = self.tool_fn(**kwargs)
        self.dedup_store.set(idempotency_key, result)
        return result
```

The idempotency key is usually derived from the thread id plus the step name plus a content hash of the inputs, so that the same logical action, attempted twice because of a crash-and-resume, is recognized as a repeat rather than executed again. Many external APIs that support side effects (payment processors, email providers, cloud infrastructure APIs) accept an idempotency key parameter directly for exactly this reason — always pass one when it's available rather than relying solely on your own dedup layer.

On the resumption path itself, the logic is straightforward once idempotency is handled: look up whether a checkpoint exists for the incoming thread id; if none exists, this is a new run and starts from the beginning; if one exists, load it and resume from the step recorded in it, trusting the idempotency layer to make any repeated side effect safe.

```python
def start_or_resume(runtime: Graph_Runtime, thread_id: str, fresh_initial_state: Agent_State):
    existing = runtime.checkpointer.latest_for_thread(thread_id)
    if existing is None:
        return runtime.run(thread_id, fresh_initial_state, start_node="plan")
    return runtime.run(thread_id, existing.state, start_node=existing.step_name)
```

## Versioning State Across Code Changes

A subtlety that's easy to miss until it bites you in production: checkpoints are data, but the code that reads them — your state schema, your node functions, your graph's edges — keeps evolving. A checkpoint saved last month, under last month's version of `Agent_State`, needs to still load correctly (or fail predictably) against this month's code, which might have added a field, renamed a key, or removed a node the old checkpoint was paused inside of.

The most robust mitigation is to tag every checkpoint with an explicit schema version at write time, and maintain a small chain of migration functions that upgrade older checkpoints to the current schema before they're used.

```python
SCHEMA_VERSION = 3

MIGRATIONS = {
    1: lambda state: {**state, "completed_steps": state.get("completed_steps", [])},  # v1 -> v2 added this field
    2: lambda state: {**state, "result": state.get("result")},                        # v2 -> v3 added this field
}

def load_and_migrate(raw_checkpoint: dict) -> dict:
    state = raw_checkpoint["state"]
    version = raw_checkpoint.get("schema_version", 1)

    while version < SCHEMA_VERSION:
        state = MIGRATIONS[version](state)
        version += 1

    return state
```

A few practices make this sustainable rather than a growing pile of special cases. Prefer additive, backward-compatible schema changes (new fields with sensible defaults) over renames or removals whenever possible, since additive changes usually don't need a migration function at all — old checkpoints simply read as if the new field were absent, and code should already handle that with `.get(key, default)` rather than direct indexing. When a node is removed or renamed entirely, keep a stub entry for it in the graph definition for some deprecation window, so that a checkpoint left paused at that exact node still has somewhere valid to resume to, even if the stub's only job is to log a warning and route to whatever replaced it. And it's worth deciding, as a policy, how long checkpoints are trusted before they're considered too stale to safely resume (a run paused for a human approval that never came, sitting for six months across a dozen schema versions, is often better terminated and re-started cleanly than force-migrated through every intermediate version).

## Production Concerns Beyond the Happy Path

A few additional considerations separate a toy checkpointing setup from one that survives real traffic. Storage backend choice should track environment: an in-memory saver is fine for local development and tests, but production needs a backend that survives process restarts independently of the agent itself — SQLite for a single-node deployment, Postgres or a managed equivalent for anything horizontally scaled, since multiple agent worker processes need to see the same checkpoint state regardless of which one picks up a given thread next.

Checkpoint volume needs active management. If you checkpoint after every superstep (the recommended default for resumability), long-running or high-throughput agents will accumulate a large number of checkpoint rows per thread; most of that history is only useful for debugging and time-travel, not for normal operation, so a retention policy — keep the last N checkpoints per thread, or prune anything older than X days once a thread reaches a terminal state — keeps storage bounded without sacrificing the ability to resume active runs.

Checkpoint *size* matters independently of count: it's tempting to stuff large blobs (full tool outputs, entire retrieved documents, long conversation histories) directly into the state object because it's convenient, but that makes every checkpoint write and read proportionally expensive. A better pattern is to store large content in your object store or memory backend from Chapters 1-3, and keep only references (ids, keys) in the checkpointed state itself — the checkpoint records *where* to find the data, not the data.

Finally, treat persisted state as a security-sensitive asset, not an implementation detail. Checkpoints frequently contain the same conversational content, tool arguments, and intermediate results that make up the rest of an agent's memory — including anything sensitive a user typed or any credentials passed through tool calls — and inherit the same obligations around encryption at rest, access control, and retention limits that apply to any other store of user data. A checkpointing system built purely for resumability, without that lens applied from the start, is one of the more common gaps that shows up in a production security review of agent infrastructure.
