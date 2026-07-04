# Types of Agent Memory

## The Core Problem: LLMs Are Stateless

It's worth starting from an uncomfortable fact: the large language model at the heart of an agent has no memory of its own. Every call to the model is a pure function — the same input tokens produce the same distribution over next tokens, every single time, with no side channel carrying information from one call to the next. Whatever feels like "the agent remembering something" is not a property of the model; it is a property of the *system you built around* the model. Someone (you, or the framework you're using) captured some piece of information, decided it was worth keeping, stored it somewhere, and later retrieved it and stuffed it back into the prompt as text. That's it. That's the entire mechanism behind every memory feature you've ever seen in an agent product.

This reframing matters because it turns "memory" from a mysterious capability into an engineering problem with three concrete sub-problems: **what to capture**, **where to store it**, and **when and how to bring it back into context**. Everything in this chapter, and the three that follow, is really just different answers to those three questions. The taxonomy of memory types that follows is a way of organizing the *what* — different categories of information call for different capture strategies, different storage backends, and different retrieval triggers.

Two independent axes are useful for organizing agent memory. The first axis is **duration**: how long does this information need to live — for the rest of this single turn, for the rest of this conversation, or forever, across sessions and even across users? This gives us the short-term vs. long-term distinction. The second axis is **content type**: is this information a specific experience that happened, a general fact about the world, or a skill for doing something? This gives us the episodic/semantic/procedural taxonomy borrowed from cognitive psychology. The two axes are orthogonal — long-term memory can be episodic, semantic, or procedural, and in practice a production agent needs all three flavors of long-term memory plus a well-managed short-term buffer.

## Short-Term (Working / Context) Memory

Short-term memory is whatever the agent is actively "thinking with" right now — the information available to the model at the moment it generates its next token, without needing to fetch anything from an external system. In practice, this is the content of the context window: the system prompt, the messages exchanged so far in the current conversation, the results of any tools the agent has called during this task, and any scratchpad reasoning (chain-of-thought, a running plan, intermediate variables) the agent has produced along the way.

The defining property of short-term memory is that it is *free to access* (no retrieval step, no network call, no similarity search — it's already sitting in the prompt) but *severely bounded* (the context window has a hard token ceiling, and every token in it costs money and, past a certain point, degrades the model's attention quality — more on that in the next chapter). This is the direct analog of human working memory, classically described as holding "seven plus or minus two" chunks of information at once, available for immediate manipulation but easily bumped out by new incoming information.

Concretely, in a ReAct-style agent loop, short-term memory is the ever-growing list of `{thought, action, observation}` triples for the current task:

```python
class Working_Memory:
    """Holds everything the model needs for the CURRENT task, nothing more."""

    def __init__(self, system_prompt: str, max_tokens: int = 8000):
        self.system_prompt = system_prompt
        self.turns = []          # list of {"role": ..., "content": ...}
        self.scratchpad = []     # intermediate thoughts / tool observations
        self.max_tokens = max_tokens

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def add_observation(self, tool_name: str, result: str):
        # Tool output is working memory too — it did not exist before this task
        self.scratchpad.append(f"[{tool_name}] -> {result}")

    def to_prompt_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.scratchpad:
            messages.append({
                "role": "system",
                "content": "Scratchpad:\n" + "\n".join(self.scratchpad),
            })
        messages.extend(self.turns)
        return messages
```

Two things are easy to miss about short-term memory. First, it is not limited to *conversational* text. In graph-based agent frameworks such as LangGraph, the "state" object passed between nodes — a plan, a counter, a partial result, a list of pending sub-tasks — is working memory in the broader engineering sense even though most of it never gets serialized into the LLM's prompt directly; it only surfaces in the prompt when a node decides to format part of it as text for the model to read. Second, short-term memory is *lossy by construction*: once the task ends and the process discards that context (or the window fills up and older turns get evicted), that information is gone unless something explicitly promoted it into long-term storage first. Deciding what to promote is exactly the boundary between short-term and long-term memory.

## Long-Term Memory

Long-term memory is information that needs to outlive the current context window — sometimes the current conversation, sometimes the current session, sometimes the current *process*. It has to live in external, durable storage: a relational database, a key-value store, a vector index, a graph database, or even flat files, because the LLM's context is wiped clean the moment you start a new conversation or restart the agent process.

The critical architectural difference from short-term memory is that long-term memory is **write-selective and read-selective**. You do not append every message ever exchanged into long-term storage verbatim (that's just an unbounded log, not memory), and you do not load all of long-term memory into every prompt (you'd blow the context budget instantly and drown the model in irrelevant text). Instead, a real long-term memory system has two decision points: an *extraction* step that decides what from the current interaction is worth persisting, and a *retrieval* step that decides what from the store is relevant enough to bring back for the current turn.

```python
class Long_Term_Memory_Interface:
    """A minimal separation of write-time and read-time concerns."""

    def __init__(self, store, extractor_llm):
        self.store = store                 # any durable backend
        self.extractor_llm = extractor_llm  # decides what's worth keeping

    def maybe_persist(self, conversation_turn: str):
        # Write-time: not everything said is worth remembering forever
        verdict = self.extractor_llm.generate(f"""
        Turn: "{conversation_turn}"
        Does this contain a durable fact, preference, or decision worth
        remembering beyond this conversation? If yes, extract it as a
        short standalone statement. If no, reply exactly: SKIP.
        """)
        if verdict.strip() != "SKIP":
            self.store.write(verdict.strip())

    def retrieve_relevant(self, current_query: str, top_k: int = 5):
        # Read-time: only pull what's relevant to *this* turn
        return self.store.search(current_query, top_k=top_k)
```

Long-term memory is where the episodic/semantic/procedural taxonomy earns its keep, because "durable information worth keeping" is not one uniform thing — a past event, a general fact, and a reusable skill need to be captured differently, stored differently, and queried differently.

## The Cognitive-Science Taxonomy: Episodic, Semantic, Procedural

This three-way split comes from memory research in cognitive psychology, most famously Endel Tulving's distinction between episodic and semantic memory, with procedural memory as a third, separate system for skills and habits. It maps onto agent systems more literally than most borrowed metaphors do, because the distinction tracks a real difference in how the information should be *used* at inference time.

### Episodic Memory: "What Happened"

Episodic memory stores specific, time-stamped, contextually grounded experiences — individual events with a "when," a "what," and usually an outcome. For an agent, this means concrete records like "on this task, I called the `deploy` tool with these arguments, it failed with a timeout, and I recovered by retrying with a longer timeout." It is inherently about a *particular instance*, not a generalization.

```python
class Episode:
    def __init__(self, task, actions, outcome, timestamp=None):
        self.task = task
        self.actions = actions
        self.outcome = outcome            # {"success": bool, "detail": str}
        self.timestamp = timestamp or datetime.now().isoformat()

class Episodic_Memory:
    def __init__(self):
        self.episodes: list[Episode] = []

    def record(self, task, actions, outcome):
        self.episodes.append(Episode(task, actions, outcome))

    def recall_similar(self, current_task, embed_fn, top_k=3):
        # Retrieval is similarity-based: "has something like this happened before?"
        query_vec = embed_fn(current_task)
        scored = [
            (cosine_similarity(query_vec, embed_fn(ep.task)), ep)
            for ep in self.episodes
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]
```

The value of episodic memory is precedent: it lets an agent say, in effect, "I've been in a situation like this before, and here's specifically what happened," which is a much stronger grounding signal than a generic instruction. A coding agent that remembers "last time I ran the test suite in this repo without activating the virtualenv, every import failed" is using episodic recall, not a general rule — it's tied to a specific prior event.

### Semantic Memory: "What Is True"

Semantic memory stores decontextualized facts — knowledge that is true independent of when or how it was learned. "The user's name is Alice." "The production database is PostgreSQL 14." "Refunds above $500 require manager approval." These are facts and entity relationships, not events; you generally don't care *when* the agent learned that the user's name is Alice, you just want the fact available whenever it's relevant.

```python
class Semantic_Memory:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def store_fact(self, fact: str, entity: str = None):
        self.vector_store.add(
            embedding=self.embedder.embed(fact),
            document=fact,
            metadata={"entity": entity, "type": "fact"},
        )

    def recall(self, query: str, top_k: int = 5):
        return self.vector_store.search(self.embedder.embed(query), top_k=top_k)
```

Semantic memory is typically what people mean when they talk about "giving an agent long-term knowledge" — a vector store of facts, or a knowledge graph of entities and relationships, that gets queried at the start of a turn to inject relevant background. Because it's decontextualized, semantic memory is also the easiest to get *wrong* through staleness: a fact stored six months ago ("the API key rotates every 90 days") can become false without any signal that it's expired, which is why production semantic memory systems need some notion of confidence, recency, or explicit invalidation.

### Procedural Memory: "How to Do It"

Procedural memory stores reusable methods — not a fact, not an event, but a *process* that worked before and can be applied again to similar problems. In humans, procedural memory is famously implicit (you can ride a bike without being able to explain the physics), but in agent systems it's usually made explicit and inspectable: a stored sequence of tool calls, a prompt template that reliably produces good output for a task type, or a multi-step playbook.

```python
class Procedural_Memory:
    def __init__(self):
        self.procedures = {}       # task_type -> list of {steps, success}

    def learn(self, task_type: str, steps: list, success: bool):
        self.procedures.setdefault(task_type, []).append(
            {"steps": steps, "success": success}
        )

    def best_for(self, task_type: str):
        candidates = [p for p in self.procedures.get(task_type, []) if p["success"]]
        return candidates[-1]["steps"] if candidates else None
```

Procedural memory is what allows an agent to *improve at a class of problems* rather than just recall isolated instances. This is the mechanism behind "skill library" designs seen in autonomous agent research (the Voyager Minecraft agent is a well-known example: it writes and stores JavaScript functions for skills like "chop a tree" or "craft a pickaxe," and composes previously-learned skills to tackle harder tasks instead of re-deriving them from scratch every time).

### One Example Across All Three

A customer-support agent illustrates how the three types coexist and complement each other on the very same underlying stream of interactions:

- **Episodic**: "Ticket #4521, last Tuesday — the customer was frustrated about a delayed refund; we escalated to a human agent and it was resolved within an hour."
- **Semantic**: "Refunds above $500 require manager approval." (A durable policy fact, true regardless of any specific ticket.)
- **Procedural**: "Standard refund-handling flow: (1) verify the order exists, (2) check it falls within the refund policy window, (3) compute the refund amount, (4) call the payments API, (5) send confirmation email." (A reusable workflow.)

Notice the direction of information flow: episodic memories are the raw material, and semantic and procedural memories are often *distilled* from patterns across many episodes. If the agent notices, across dozens of episodic records, that escalating angry refund customers to a human within the first two messages correlates with faster resolution, that pattern can be promoted into a procedural rule. This distillation process — usually called reflection or consolidation — is covered in more depth in the memory architectures chapter, but it's worth flagging here as the reason episodic memory isn't just a log file: it's the input to a learning loop.

## Mapping the Taxonomy onto Real Systems

In production agent stacks, these categories usually correspond to genuinely different storage technologies and retrieval triggers, not just conceptual labels:

- **Working memory** lives in the orchestration framework's runtime state — a Python dict, a LangGraph `StateGraph`'s state object, or simply the growing messages list passed to the LLM API on each call. It requires no query; it's just always present.
- **Episodic memory** is usually a datastore keyed by time and task, queried by similarity ("find past episodes like this one") — frameworks like Mem0 or Zep, or a homegrown table plus a vector index, are typical implementations. Retrieval is triggered when the agent starts a new task and wants precedent.
- **Semantic memory** is almost always vector-store-backed (for fuzzy factual recall) or graph-backed (for explicit entity relationships), queried whenever the current turn's topic touches an entity or fact the system might already know about.
- **Procedural memory** is often just structured storage (a database table or even a directory of files) keyed by task type or skill name, queried when the agent recognizes it's facing a known category of problem rather than something novel.

### A Quick Reference Comparison

It helps to have the four categories side by side, since interview questions often probe whether you can place a given piece of information into the right bucket on the spot:

| Memory Type | Answers | Duration | Typical Storage | Retrieval Trigger |
|---|---|---|---|---|
| Working (short-term) | "What's happening right now?" | Current turn/task | Context window / runtime state | Always present, no query needed |
| Episodic | "What happened before?" | Permanent, timestamped | Event log, table + vector index | Similarity to current task |
| Semantic | "What is true?" | Permanent, decontextualized | Vector store, knowledge graph | Topical/entity overlap with current turn |
| Procedural | "How do I do this?" | Permanent, reusable | Keyed store by task/skill type | Recognized task category |

A detail worth internalizing for anyone building or discussing these systems: the boundaries are not always clean in implementation, only in intent. A single vector store can physically hold episodic records and semantic facts side by side, distinguished only by a `type` field in the metadata. What makes the taxonomy useful isn't that it forces separate infrastructure — it's that it forces you to be explicit, for every write and every read, about *which kind* of information you're handling, because that decision changes how you'd want to score, filter, and expire it. A semantic fact about a compliance policy might never expire; an episodic record of a single failed API call from eight months ago is almost always safe to prune or fold into a distilled procedural lesson instead of keeping verbatim.

### Why Frameworks Blur This on Purpose

It's worth noting that most agent frameworks don't expose "episodic memory" or "semantic memory" as first-class, differently-named APIs — they expose something more generic, like a key-value store, a "memory" abstraction with a `namespace` parameter, or a document store with metadata filters, and leave the taxonomy as a modeling choice made on top. LangGraph's persistent store abstraction, for instance, is namespace- and key-based and completely agnostic to whether you use it to hold episodic events or semantic facts — the distinction lives entirely in how your application organizes namespaces and what it puts in the metadata. The same is true of most vector database SDKs: they don't know or care whether a stored embedding represents a fact or an event, they just index vectors and return nearest neighbors. This is a deliberate design choice on the framework's part (staying unopinionated keeps the primitive general-purpose), but it means the taxonomy discussed in this chapter has to be imposed by *you*, in your application layer, through consistent metadata tagging and separate query logic — it will not appear automatically just because you picked a sufficiently sophisticated memory framework.

## Practical Guidance: What Goes Where

When you're deciding how to route a piece of information, three questions in sequence usually settle it. Is this needed only to finish the current turn or task? If yes, it belongs in working memory and can be discarded afterward. Does it need to survive beyond this session, and is it tied to a specific event with a "when" and an outcome? If yes, it's episodic. Is it a fact that would be true regardless of when it was learned, independent of any specific event? If yes, it's semantic. Is it a *method* — a way of doing something — that would help with future, similar tasks? If yes, it's procedural.

A common failure mode in early memory implementations is dumping everything — chat logs, extracted facts, and successful action sequences alike — into a single undifferentiated vector store and calling it "memory." This works for demos but degrades badly in production, because the categories have different retrieval semantics: a query like "how did I solve this before" wants episodic precedent, "what does this user prefer" wants semantic facts, and "what's the standard way to do this" wants a procedure. Mixing them in one index means every query returns a noisy blend of event logs, facts, and workflows, and the agent has to do extra work at read-time to figure out which retrieved snippet is actually useful — work that could have been avoided entirely by keeping the three types in separate, purpose-built stores with separate retrieval logic. The next three chapters build on exactly this separation: how to manage what's kept in the scarce short-term context (Chapter 2), and how to actually implement long-term stores for these different memory types in production (Chapter 3).
