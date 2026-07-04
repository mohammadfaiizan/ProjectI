# Memory Architectures and Implementations

The previous two chapters covered the conceptual vocabulary (what kinds of memory an agent needs) and the resource constraints (why the context window forces hard choices about what to keep). This chapter is about the concrete engineering patterns you actually implement: the handful of memory architectures that show up, in some variation, in essentially every production agent system. Each pattern trades off simplicity, cost, fidelity, and query power differently, and picking the right one (or the right combination) is one of the highest-leverage design decisions in building an agent.

## Pattern 1: Simple Buffer Memory

The simplest possible memory implementation is a buffer: an ordered list of messages that grows as the conversation proceeds, sent to the model in full (or up to some cutoff) on every call. There is no summarization, no retrieval, no scoring — just a list.

```python
class Buffer_Memory:
    """The baseline every other pattern is compared against."""

    def __init__(self, system_prompt: str, max_messages: int = None):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_name: str, result: str):
        self.messages.append({
            "role": "tool",
            "name": tool_name,
            "content": result,
        })

    def get_messages(self):
        history = self.messages
        if self.max_messages:
            history = history[-self.max_messages:]
        return [{"role": "system", "content": self.system_prompt}] + history
```

Buffer memory's appeal is that it's trivially correct: nothing is inferred, nothing is compressed, there's no risk of a summarization step silently dropping something important. Its weakness is that it doesn't scale — cost and latency grow linearly with conversation length, and eventually it either blows the context window or has to bolt on a truncation rule, at which point it stops being "simple" and starts being a sliding window (Chapter 2). Buffer memory is the right choice exactly when conversations are short-lived by design: single-shot tasks, short support interactions, or any agent that's explicitly stateless across calls apart from the current exchange. It's also the correct *building block* to reach for first — every more sophisticated pattern below is best understood as "buffer memory, plus one additional mechanism to keep it bounded."

## Pattern 2: Running Summary Memory

Running summary memory replaces the raw buffer with an LLM-maintained summary that's updated incrementally as the conversation grows, so the token cost of "remembering the whole conversation" stays roughly flat instead of growing without bound.

```python
class Running_Summary_Memory:
    def __init__(self, llm, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.summary = ""
        self.pending = []          # not yet folded into the summary

    def add_message(self, role: str, content: str):
        self.pending.append({"role": role, "content": content})

    def consolidate(self):
        """Call this periodically (every N turns, or on a token threshold)."""
        if not self.pending:
            return
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.pending)
        self.summary = self.llm.generate(f"""
        Current summary: {self.summary or "(empty)"}
        New turns: {transcript}

        Update the summary to include the new turns. Preserve facts,
        decisions, names, and numbers. Remove resolved small talk.
        """)
        self.pending = []

    def get_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"Conversation summary:\n{self.summary}"})
        messages.extend(self.pending)   # anything not yet folded in stays verbatim
        return messages
```

The key design decision in a running-summary implementation is *when* consolidation runs. Running it on every single message maximizes cost (one extra LLM call per turn) but minimizes the size of `pending`; running it every N turns is cheaper but means a burst of `pending` messages sits un-compressed until the threshold hits. Most production systems trigger consolidation on a token threshold rather than a message-count threshold, since a handful of verbose tool outputs can blow the budget just as easily as many short chat turns.

The structural risk with any pure summary approach is precision loss under repeated compounding: a fact stated exactly once, several consolidation cycles ago, might survive three or four summarize-of-a-summary passes intact, or might get paraphrased into something subtly wrong, or might get dropped as "not important" by a summarization prompt that didn't realize it would matter later. This is rarely visible in a demo and reliably shows up in production once conversations run long — which is exactly why summary memory is almost always paired with a small verbatim window (as in the hybrid pattern from Chapter 2) rather than used in isolation.

## Pattern 3: Vector-Store-Backed Retrieval Memory

Vector-backed memory takes a fundamentally different approach: instead of trying to keep *all* history accessible by compressing it, it stores every memorable unit (a fact, a past exchange, a document chunk) as an embedding in a vector index, and retrieves only the handful of entries relevant to the current query at read-time. This is the same mechanism that powers retrieval-augmented generation, applied to the agent's own memory rather than to a static document corpus, and it's the backbone of most "the agent remembers things about me across sessions" features in real products.

```python
class Vector_Memory:
    def __init__(self, embed_fn, vector_store):
        """
        embed_fn: text -> vector
        vector_store: any index exposing .add(id, vector, document, metadata)
                       and .search(vector, top_k) -> [(id, document, metadata, score), ...]
        """
        self.embed_fn = embed_fn
        self.vector_store = vector_store
        self._next_id = 0

    def remember(self, text: str, metadata: dict = None):
        vec = self.embed_fn(text)
        entry_id = f"mem_{self._next_id}"
        self._next_id += 1
        self.vector_store.add(
            id=entry_id,
            vector=vec,
            document=text,
            metadata={**(metadata or {}), "stored_at": datetime.now().isoformat()},
        )
        return entry_id

    def recall(self, query: str, top_k: int = 5, min_score: float = 0.7):
        vec = self.embed_fn(query)
        results = self.vector_store.search(vec, top_k=top_k)
        return [r for r in results if r["score"] >= min_score]

    def forget(self, entry_id: str):
        self.vector_store.delete(entry_id)
```

A production version of this pattern needs a few things the toy version glosses over. First, deduplication at write time — without it, restating the same fact across many turns produces many near-duplicate vectors that all compete for the same retrieval slots and dilute the quality of the top-k results. Second, a decision about *what unit* gets embedded: whole messages, extracted single-sentence facts, or fixed-size chunks of longer content all behave differently under similarity search, and extracted facts (produced by an LLM call at write time, the same way the long-term-memory extractor in Chapter 1 works) generally retrieve far more precisely than raw conversational turns, because a raw turn often mixes several unrelated pieces of information that only partially match any given query.

```python
class Fact_Extracting_Vector_Memory(Vector_Memory):
    """Extracts atomic facts before embedding, rather than embedding raw turns."""

    def __init__(self, embed_fn, vector_store, extractor_llm):
        super().__init__(embed_fn, vector_store)
        self.extractor_llm = extractor_llm

    def remember_from_turn(self, turn_text: str):
        facts = self.extractor_llm.generate(f"""
        Extract a list of standalone, atomic facts worth remembering
        long-term from this message. One fact per line. If none, reply
        with an empty response.

        Message: "{turn_text}"
        """)
        stored_ids = []
        for line in filter(None, facts.splitlines()):
            stored_ids.append(self.remember(line.strip()))
        return stored_ids
```

Third, recency and importance should usually weigh in alongside pure semantic similarity — a highly similar but three-month-stale fact might be worth less than a moderately similar but very recent one, especially for fast-changing information. A common refinement is to score candidates with a blended function rather than raw cosine similarity alone:

```python
def score_candidate(similarity, stored_at, importance=0.5, half_life_days=30):
    age_days = (datetime.now() - stored_at).days
    recency_weight = 0.5 ** (age_days / half_life_days)   # exponential decay
    return 0.6 * similarity + 0.25 * recency_weight + 0.15 * importance
```

Vector-backed memory's core strength is that it scales to enormous amounts of stored information without growing the prompt — retrieval cost grows with index size, not with conversation length, and modern approximate-nearest-neighbor indexes keep that cost low even at millions of entries. Its core weakness is that it's fundamentally a *fuzzy* retrieval mechanism: it's good at "find things related to this topic" and bad at "find the exact fact that answers this precise question" or "enumerate everything true about entity X," both of which are better served by structured storage.

## Pattern 4: Knowledge-Graph-Based Memory

Where vector memory treats memory as a bag of loosely related text chunks, knowledge-graph memory treats it as an explicit structure of entities and the relationships between them — nodes for people, objects, and concepts, edges for the relationships that connect them ("Alice works_at Acme," "Acme uses PostgreSQL," "PostgreSQL requires_approval_from DBA-team-for schema changes"). This trades the fuzzy recall of embeddings for precise, queryable structure.

```python
class Knowledge_Graph_Memory:
    def __init__(self, graph_store):
        """
        graph_store: any backend exposing basic node/edge operations
        (e.g., a thin wrapper over Neo4j, or an in-memory adjacency structure).
        """
        self.graph = graph_store

    def add_entity(self, name: str, entity_type: str, attributes: dict = None):
        self.graph.upsert_node(name, labels=[entity_type], properties=attributes or {})

    def add_relationship(self, source: str, relation: str, target: str, attributes: dict = None):
        self.graph.upsert_edge(source, relation, target, properties=attributes or {})

    def query_relationships(self, entity: str, relation: str = None):
        # e.g., "what does Acme use?" -> traverse outgoing 'uses' edges from Acme
        return self.graph.get_edges(source=entity, relation=relation)

    def find_path(self, entity_a: str, entity_b: str, max_hops: int = 3):
        # e.g., "how is Alice connected to the DBA team?" -> multi-hop traversal
        return self.graph.shortest_path(entity_a, entity_b, max_hops=max_hops)
```

Knowledge graphs shine in exactly the cases where vector search struggles: multi-hop reasoning ("who approves changes to the database that Acme's checkout service depends on?"), exhaustive enumeration ("list every project Alice has worked on," which a similarity search can silently under-return if some projects are phrased very differently from the query), and consistency ("if the DB was renamed, update every fact that referenced the old name," which is a single node rename in a graph and an untraceable set of stale embeddings otherwise). The cost is that knowledge graphs require either a reliable entity/relationship extraction pipeline (usually an LLM call that turns free text into structured triples) or manual curation, and both are more engineering-intensive to stand up and maintain than "embed the text and throw it in a vector index."

```python
class Triple_Extractor:
    def __init__(self, llm, kg_memory: Knowledge_Graph_Memory):
        self.llm = llm
        self.kg = kg_memory

    def extract_and_store(self, text: str):
        # Ask the model to emit (subject, relation, object) triples
        raw = self.llm.generate(f"""
        Extract factual relationships from this text as triples in the
        exact format "subject | relation | object", one per line.
        Only extract clear, unambiguous relationships.

        Text: "{text}"
        """)
        for line in filter(None, raw.splitlines()):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                subject, relation, obj = parts
                self.kg.add_entity(subject, "Entity")
                self.kg.add_entity(obj, "Entity")
                self.kg.add_relationship(subject, relation, obj)
```

In practice, knowledge-graph memory is rarely the *only* long-term store in a system; it's typically layered alongside vector memory, handling the subset of information that's genuinely relational and benefits from exact traversal, while vector memory handles the larger, fuzzier body of unstructured facts and past exchanges.

## Choosing and Combining Patterns

None of these four patterns is a strict upgrade over the others — they answer different questions well. Buffer memory answers "what was literally said" with perfect fidelity but no scale. Running summary answers "what's the gist of everything so far" with bounded cost but degrading precision over time. Vector memory answers "what do I know that's related to this" at scale, with fuzzy recall. Knowledge-graph memory answers "how are these things precisely connected" with exact structure, at the cost of an extraction and maintenance pipeline.

A realistic production agent composes several of these rather than picking one:

```python
class Composite_Agent_Memory:
    def __init__(self, llm, embed_fn, vector_store, graph_store, system_prompt):
        self.working = Buffer_Memory(system_prompt, max_messages=12)
        self.summary = Running_Summary_Memory(llm, system_prompt)
        self.semantic = Vector_Memory(embed_fn, vector_store)
        self.graph = Knowledge_Graph_Memory(graph_store)
        self.extractor = Triple_Extractor(llm, self.graph)

    def observe_turn(self, role: str, content: str):
        self.working.add_user_message(content) if role == "user" else self.working.add_assistant_message(content)
        self.summary.add_message(role, content)
        if role == "user":
            self.semantic.remember(content)
            self.extractor.extract_and_store(content)

    def build_context(self, current_query: str):
        recent = self.working.get_messages()
        facts = self.semantic.recall(current_query, top_k=5)
        related_entities = self.graph.query_relationships(current_query)
        return {
            "recent_messages": recent,
            "relevant_facts": [f["document"] for f in facts],
            "graph_relationships": related_entities,
        }
```

The right combination is a function of what the agent actually needs to be good at: a coding assistant leans heavily on buffer memory for the current file/task and vector memory for prior similar bugs or design decisions; a personal assistant leans on vector memory for preferences and a lightweight knowledge graph for relationships between people, events, and commitments; a customer-support agent typically needs all four, since it has short-lived conversational context, long conversations that need summarizing, a large base of past ticket precedent best served by vector search, and account/entitlement relationships that are exactly the kind of precise, multi-hop structure a knowledge graph is built for. Choosing well starts with the question this chapter opened on: what does this specific piece of information need to be — a fact, a precedent, a relationship, or just the last thing that was said — and let that answer drive the storage pattern, not the other way around.
