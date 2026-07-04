# Context Window Management

## Why the Context Window Is Still a Scarce Resource

It's tempting to think that context window management is a problem that's solving itself — every few months a new model ships with a bigger window, and at some point (1M tokens, 2M tokens) the whole issue should just disappear. In practice it doesn't, for three reasons that are worth internalizing before looking at any specific technique.

The first reason is cost. Most commercial LLM APIs bill per input token, and that bill is not a rounding error at scale: an agent that habitually stuffs 100K tokens of history into every single call, when only a few hundred tokens of that history are actually relevant to the current step, is paying for the same wasted context over and over, on every turn of every conversation, multiplied across every user. A larger window doesn't make this cheaper — it makes it *easier to accidentally do*, because there's no forcing function stopping you from filling it.

The second reason is latency. Time-to-first-token and total generation time both scale with the size of the input the model has to process (the prefill phase has to attend over every input token before generating anything). An agent with a bloated context is a slower agent, and in an interactive product, latency is a feature, not an afterthought. This cost doesn't vanish just because the window *could* hold more — it scales with what you actually put in it.

The third and most important reason is quality, and it's the one people most often get wrong: **more context is not free even when it's cheap and fast**, because attention is a finite, shared resource inside the model. Every additional token you add competes for the same fixed attention budget across every layer and head. A model asked to reason over 200,000 tokens of mostly-irrelevant history is not simply "using more of its capacity" — it is spreading its ability to attend to any single piece of that history more thinly, and empirically this shows up as degraded recall and degraded reasoning, not just degraded speed. This is precisely why the "lost in the middle" phenomenon (covered in detail below) exists at all: the problem isn't that the model runs out of room, it's that the model's *effective* attention to any given piece of information is not uniform across a long context, no matter how large the nominal window is.

Put together, this means context window management isn't a stopgap technique that bigger models will make obsolete — it's a permanent discipline of curating what a model actually needs to see, the same way a good technical writer curates a document instead of pasting in every source they consulted.

## The Four Levers: Compress, Truncate, Select, and Budget

There are really only four moves available when the information you'd like the model to have exceeds what you're willing (or able) to put in the prompt: summarize it down to something smaller, cut off the parts you're willing to lose, select only the parts that are relevant right now, or explicitly budget and prioritize across competing sources of context. Real systems combine all four; it's useful to look at each on its own first.

### Summarization-Based Compression

The idea behind summarization is to periodically replace a growing block of raw history with a much shorter LLM-generated summary that preserves the information that matters while discarding the information that doesn't (small talk, redundant confirmations, verbose tool output that's already been acted on). This trades a small amount of *fidelity* for a large amount of *space*.

```python
class Summarizing_Memory:
    def __init__(self, llm, system_prompt, flush_after=12):
        self.llm = llm
        self.system_prompt = system_prompt
        self.summary = ""
        self.recent = []
        self.flush_after = flush_after

    def add(self, role, content):
        self.recent.append({"role": role, "content": content})
        if len(self.recent) >= self.flush_after:
            self._compress()

    def _compress(self):
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.recent)
        self.summary = self.llm.generate(f"""
        Existing summary of the conversation so far:
        {self.summary or "(none yet)"}

        New messages to fold in:
        {transcript}

        Produce an updated summary. Preserve names, numbers, decisions,
        open questions, and anything the user explicitly asked to be
        remembered. Drop small talk and resolved back-and-forth. Keep it
        under 200 words.
        """)
        self.recent = []   # the raw messages are now represented by the summary

    def to_prompt_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"Summary so far:\n{self.summary}"})
        messages.extend(self.recent)
        return messages
```

The obvious risk with summarization is lossy compounding: summarizing a summary of a summary, repeatedly, gradually erodes detail, and there's no way to get back information that a prior summarization pass dropped. This is why most production systems don't summarize *everything*, and instead only summarize the portion of history that's aging out of the active window (see the hybrid pattern below), while keeping verbatim access to anything recent or explicitly flagged as important.

### Sliding Window Truncation

The simplest possible strategy is to just keep the last N messages (or the last N tokens) and drop everything older, with no attempt to preserve what was dropped. This is cheap to implement, has predictable and bounded cost, and is often good enough for short-lived, low-stakes interactions like a simple FAQ chatbot.

```python
class Sliding_Window:
    def __init__(self, system_prompt, max_messages=20):
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages:]   # hard cutoff

    def to_prompt_messages(self):
        return [{"role": "system", "content": self.system_prompt}] + self.messages
```

The failure mode is equally simple: information loss is abrupt and total. If the user mentioned a critical constraint 25 messages ago and the window only holds the last 20, that constraint is just gone — not summarized, not degraded, completely absent — and the agent has no way of knowing it ever existed. Sliding windows are a reasonable default only when you're confident that nothing said more than N turns ago could plausibly still matter.

### Relevance-Based Selection

Rather than keeping things because they're *recent* (sliding window) or compressing everything indiscriminately (summarization), relevance-based selection asks a sharper question for every candidate piece of context: is this specifically useful for answering *the current query*? This is the same idea that powers retrieval-augmented generation, applied to the agent's own history and working state rather than to an external document corpus.

```python
def select_relevant_context(current_query, candidate_chunks, embed_fn, token_budget, count_tokens_fn):
    """
    candidate_chunks: prior messages, tool outputs, retrieved facts — anything
    that competes for a slot in the prompt.
    """
    query_vec = embed_fn(current_query)
    scored = sorted(
        candidate_chunks,
        key=lambda c: cosine_similarity(query_vec, embed_fn(c["text"])),
        reverse=True,
    )

    selected, used = [], 0
    for chunk in scored:
        tokens = count_tokens_fn(chunk["text"])
        if used + tokens > token_budget:
            continue   # skip low-relevance chunks that don't fit, don't just cut off at the end
        selected.append(chunk)
        used += tokens
    return selected
```

Relevance-based selection is strictly more work than truncation — it requires embeddings or some other scoring mechanism, and it needs to run on every turn — but it's the only one of the three approaches that scales to genuinely large histories without either destroying old information (sliding window) or slowly blurring it (summarization). Its own failure mode is over-narrowing: an aggressive relevance filter can miss context that doesn't look lexically or semantically similar to the current query but is nonetheless necessary (a constraint mentioned once, early on, phrased very differently from how it would come up again). This is usually mitigated by always including a few "anchor" items regardless of similarity score — the system prompt, the most recent turn, and anything explicitly pinned as important.

### Combining the Levers: A Hybrid Pattern

Production agents rarely pick just one strategy. A typical, pragmatic combination keeps a small sliding window of verbatim recent turns for local coherence, maintains a running summary for everything older, and layers relevance-based retrieval on top for long-term facts that live outside the conversation entirely (covered in Chapter 3):

```python
class Hybrid_Context_Manager:
    def __init__(self, llm, system_prompt, window_size=8):
        self.llm = llm
        self.system_prompt = system_prompt
        self.window = []
        self.summary = ""
        self.window_size = window_size

    def add(self, role, content):
        self.window.append({"role": role, "content": content})
        if len(self.window) > self.window_size:
            overflow = self.window[: len(self.window) - self.window_size]
            self.window = self.window[-self.window_size:]
            self._fold_into_summary(overflow)

    def _fold_into_summary(self, overflow_messages):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in overflow_messages)
        self.summary = self.llm.generate(
            f"Prior summary:\n{self.summary}\n\nFold in these older messages, "
            f"preserving key facts and decisions:\n{text}"
        )

    def to_prompt_messages(self, retrieved_facts=None):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"Earlier context:\n{self.summary}"})
        if retrieved_facts:
            messages.append({"role": "system", "content": "Relevant facts:\n" + "\n".join(retrieved_facts)})
        messages.extend(self.window)
        return messages
```

## Explicit Token Budgeting

Once an agent is pulling from more than one source of context — conversation history, retrieved documents, tool results, a memory summary — it stops being enough to just "add things until it doesn't fit." You need an explicit budget that allocates the window across sources by priority, so that a burst of tool output doesn't silently crowd out the conversation history the user actually cares about.

```python
class Context_Budget:
    def __init__(self, total_tokens=8000, reserve_for_output=1000):
        self.total = total_tokens
        self.reserve_for_output = reserve_for_output
        self.allocations = {
            "system_prompt": 500,
            "memory_summary": 1000,
            "retrieved_context": 2000,
            "conversation": 2500,
            "tool_results": 1000,
        }

    def available_for(self, section):
        return self.allocations.get(section, 0)

    def total_allocated(self):
        return sum(self.allocations.values()) + self.reserve_for_output
```

The output reserve deserves special mention: it's a common bug to compute the input budget as if the entire window were available for input, and then have the request fail or get silently truncated because the model's response also has to fit inside the same total window. Always reserve headroom for generation before doing anything else.

## Prompt Caching as a Complementary Mitigation

Compression, truncation, and selection all reduce *how much* context you send. A separate, complementary lever reduces the *cost* of sending context you've already decided you need: prompt caching (offered under various names — prompt caching, context caching, cached input tokens — by most major model providers). The mechanism exploits the fact that transformer inference over a prefix can have its intermediate key/value activations computed once and reused, as long as the prefix is byte-for-byte identical across calls. If your system prompt, your tool definitions, and a stable block of background context are the same on every call within a session, the provider can skip recomputing attention over that prefix on every request and charge a steep discount (commonly on the order of 90% off) for the cached portion.

This changes the cost calculus for context management in a way that's worth knowing explicitly: it rewards *stable, front-loaded* context and penalizes context that changes on every turn. A system prompt and a set of tool schemas that never change within a session are ideal caching candidates; a running summary that gets rewritten every few turns, or a set of freshly retrieved documents that differ on every query, defeat the cache because any change to a prefix invalidates the cached activations for everything after it. This has a direct architectural implication for how you order the prompt: stable material (system prompt, tool definitions, pinned facts) should be placed first and kept byte-identical across calls whenever possible, and volatile material (the latest retrieved chunks, the newest conversation turns) should be placed after it, so that the volatile suffix doesn't invalidate caching for the stable prefix that precedes it.

```python
def build_cache_friendly_prompt(stable_system_prompt, stable_tool_defs, volatile_history, volatile_retrieved):
    """
    Stable, unchanging content goes first so providers can cache its
    key/value activations across repeated calls within a session.
    Volatile content goes last so its constant changes don't invalidate
    the cached prefix.
    """
    return [
        {"role": "system", "content": stable_system_prompt},   # identical every call -> cached
        {"role": "system", "content": stable_tool_defs},        # identical every call -> cached
        *volatile_history,                                      # changes every turn
        {"role": "system", "content": volatile_retrieved},      # changes every query
    ]
```

Prompt caching doesn't reduce the quality risk from a large context (the "lost in the middle" effect below still applies to a cached prefix just as much as an uncached one), but it substantially changes the cost/latency argument for keeping a moderately large, stable block of context (a big tool catalog, a detailed style guide, reference documentation) resident across a session rather than trying to aggressively trim it — as long as it stays identical across calls, its marginal cost drops sharply after the first request.

## The "Lost in the Middle" Phenomenon

Even when everything fits comfortably inside the context window, *where* information sits within that window measurably affects how well the model uses it. Multiple studies of long-context retrieval (most famously the "Lost in the Middle" line of research) show a consistent U-shaped performance curve: models are noticeably better at using information placed near the very beginning of the context (primacy) or the very end, closest to the query (recency), and measurably worse at using information buried in the middle — even though nothing about that information is objectively different, and even though the model's advertised context window comfortably covers the whole thing.

This isn't a bug that a specific model will patch out; it's a structural consequence of how self-attention and positional encoding interact over long sequences, and it shows up across model families and context lengths. The practical implication is that **where** you place something in the prompt is a real design decision, not an afterthought:

```python
def assemble_prompt_with_position_awareness(system_prompt, critical_facts, background_docs, user_query):
    """
    Put the highest-value information at the edges (start/end) of the context,
    and lower-priority or bulkier material in the middle, since the middle
    is where recall degrades most.
    """
    parts = []
    parts.append(system_prompt)                      # start: always attended to well
    parts.append("Background (lower priority):\n" + "\n".join(background_docs))  # middle
    parts.append("Critical facts for this query:\n" + "\n".join(critical_facts)) # near the end
    parts.append(f"User query: {user_query}")         # end: highest recency weight
    return "\n\n".join(parts)
```

Two concrete mitigations follow directly from this. First, when assembling a prompt from multiple retrieved chunks, don't just concatenate them in retrieval-score order — deliberately place the most important 1-2 chunks last, immediately before the query, since that position gets the strongest recency-driven attention. Second, treat "it fits in the context window" and "the model will reliably use it" as two different claims; a fact you truly cannot afford to have missed (a safety constraint, a hard requirement) is often better re-stated explicitly near the query rather than trusted to be recalled correctly from somewhere in the middle of a long retrieved-document dump. This is also a strong argument, independent of cost, for keeping contexts as lean as possible in the first place: the compression and selection techniques earlier in this chapter aren't just about fitting inside a budget, they're about keeping the signal-to-noise ratio high enough that the model can actually find and use what matters, regardless of how large the window nominally is.

## Putting It Together

None of these techniques is a universal answer; they're a toolkit, and the right combination depends on the shape of the workload. A short-lived customer support chat can often get away with a sliding window alone. A long-running coding agent that accumulates large tool outputs benefits enormously from aggressive summarization of anything more than a few turns old, since raw tool output rarely needs to be re-read verbatim once its result has been acted on. An agent doing research over a large corpus needs relevance-based selection as its primary mechanism, because most of the corpus is irrelevant to any single query and no amount of summarization would make dumping the whole thing in-context a good idea. And every one of these, regardless of strategy, benefits from explicit token budgeting and position-aware assembly, because those two disciplines are what keep "fits in the window" and "the model will actually use it well" from silently drifting apart as the system scales. The next chapter turns these levers into concrete, reusable memory architectures you can drop into an agent.
