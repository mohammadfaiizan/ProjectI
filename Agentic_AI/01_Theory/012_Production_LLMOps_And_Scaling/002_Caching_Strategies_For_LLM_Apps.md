# Caching Strategies for LLM Applications

## Why Caching Matters More Here Than in Typical Web Systems

Caching is an old idea, but it earns a disproportionately large role in LLM applications for a simple reason: the thing you're caching is expensive on two axes at once. A cache hit in a normal web app saves you a database round trip, usually a few milliseconds and negligible marginal cost. A cache hit in an LLM application can save you 500ms-5s of latency and a real, metered dollar cost per request, because every token you don't have to generate is a token the provider doesn't bill you for and your user doesn't have to wait for. That combination — caching as both a cost lever and a latency lever, at LLM-scale unit economics — is why production LLM systems tend to have not one caching layer but several, stacked at different points in the request path, each catching a different kind of redundancy.

It helps to think of these layers as a funnel, from cheapest-and-narrowest to most-general-and-most-expensive-to-implement-well: exact-match caching catches only identical requests; semantic caching catches requests that mean the same thing but aren't worded the same; and provider-level prompt caching catches partial redundancy *within* a request — a shared prefix even when the full request is unique. A mature system runs a request through this funnel in order, generating fresh output only when nothing upstream could serve it.

## Layer One: Exact-Match Response Caching

The simplest and cheapest layer is the one you'd build for any expensive, idempotent function call: hash the input, look up the hash in a fast key-value store (Redis is the default choice), and return the cached response on a hit. This is effective wherever genuine exact repetition exists — FAQ-style chatbots where many users ask literally the same question, batch pipelines that occasionally reprocess the same document, or any workload with a long tail of popular, identical queries.

```python
import hashlib
import json
import time

class ExactMatchCache:
    def __init__(self, redis_client, ttl_seconds=3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _key(self, model, messages, params):
        # Include everything that affects output: model, full message history,
        # and generation params. Changing any of these must be a cache miss.
        payload = json.dumps(
            {"model": model, "messages": messages, "params": params},
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return f"llm_cache:{digest}"

    def get(self, model, messages, params):
        key = self._key(model, messages, params)
        cached = self.redis.get(key)
        if cached:
            self.hits += 1
            return json.loads(cached)
        self.misses += 1
        return None

    def set(self, model, messages, params, response):
        key = self._key(model, messages, params)
        self.redis.setex(key, self.ttl, json.dumps(response))

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```

Two design details matter more than they look. First, the cache key must include the generation parameters (temperature, top_p, model version, system prompt version), not just the user message — otherwise you'll silently serve a response generated under different settings than what was requested, which is a subtle correctness bug that's easy to introduce and hard to notice in testing. Second, TTL should reflect how quickly the *correct answer* to a query can change, not an arbitrary infrastructure default: caching "what's your refund policy" for 24 hours is safe, caching "what's the weather right now" for even five minutes is a bug. Exact-match caching is precise (a hit is always safe to serve, assuming the key correctly captures everything that matters) but narrow — in a real chat product, the fraction of user turns that are byte-identical to a previous turn is often surprisingly small, because natural language has enormous surface variety even when the underlying intent repeats constantly. That gap is exactly what semantic caching exists to close.

## Layer Two: Semantic Caching

Semantic caching relaxes the matching criterion from "identical string" to "similar meaning," using the same embedding-and-vector-search machinery that powers RAG retrieval. Instead of hashing the raw query, you embed it, search a vector index of previously-seen queries for near neighbors, and if the closest previous query is above a similarity threshold, you serve its cached response instead of calling the LLM again. This catches the enormous space of paraphrases that exact-match caching misses entirely: "how do I reset my password," "I forgot my password, help," and "password reset steps please" are three different strings but one semantic query, and a well-tuned semantic cache serves all three from a single generation.

```python
import numpy as np

class SemanticCache:
    def __init__(self, embed_fn, vector_store, similarity_threshold=0.92):
        self.embed_fn = embed_fn
        self.vector_store = vector_store   # supports add(vec, payload) and search(vec, k)
        self.threshold = similarity_threshold

    def lookup(self, query: str):
        query_vec = self.embed_fn(query)
        results = self.vector_store.search(query_vec, k=1)
        if not results:
            return None
        best_match, score = results[0]
        if score >= self.threshold:
            return best_match["response"]
        return None

    def store(self, query: str, response: str):
        query_vec = self.embed_fn(query)
        self.vector_store.add(query_vec, {"query": query, "response": response})


def cached_generate(query, semantic_cache, llm_generate_fn):
    cached_response = semantic_cache.lookup(query)
    if cached_response is not None:
        return cached_response, {"cache": "semantic_hit"}

    response = llm_generate_fn(query)
    semantic_cache.store(query, response)
    return response, {"cache": "miss"}
```

The threshold is the whole game here, and it's a precision/recall tradeoff you have to tune empirically, not guess. Set it too low (too permissive) and you'll serve a cached answer for a query that's superficially similar but actually different in a way that matters — "how do I cancel my subscription" and "how do I cancel my order" might embed closely because they share so much surface vocabulary, but they need different answers, and serving the wrong one silently is far worse than the latency cost of a fresh generation. Set it too high (too strict) and the cache rarely fires, giving you all the infrastructure complexity of semantic caching with almost none of the savings. In practice, teams tune this threshold against a labeled set of query pairs (same-intent vs. different-intent) the same way you'd calibrate a retrieval relevance cutoff, and they re-validate it whenever the embedding model changes, since thresholds are not portable across embedding models — a 0.92 cosine cutoff calibrated for one embedding model means nothing for another.

There's a second, sharper risk worth naming explicitly because it's the classic semantic-caching failure mode in interviews: staleness and context-blindness. A cached response was correct for the context it was generated in — but semantic similarity of the *query* says nothing about whether the surrounding conversation, user identity, or point-in-time facts are still the same. If user A's query "what's my account balance" gets cached and served to user B whose query embeds similarly, you've leaked one user's data to another — a correctness and security bug, not just a quality one. The practical mitigation is to scope semantic caches per-user (or per-tenant) rather than globally whenever the response could depend on anything besides the query text itself, and to exclude time-sensitive or personalized queries from the semantic cache layer entirely, routing them to fresh generation (or the exact-match layer only, with a short TTL) by classifying query type before the cache lookup.

```python
PERSONALIZED_OR_TIME_SENSITIVE = {"account_balance", "order_status", "current_weather", "live_price"}

def route_to_cache_layer(query, intent_classifier):
    intent = intent_classifier(query)
    if intent in PERSONALIZED_OR_TIME_SENSITIVE:
        return "no_cache"       # always regenerate, correctness matters more than cost here
    return "semantic_cache"
```

## Layer Three: Provider-Level Prompt Caching

The first two layers cache at the level of a whole request/response pair. Provider-level prompt caching operates one level deeper: it caches the *KV cache* — the internal attention keys and values — for a shared prefix of tokens across multiple requests, so the provider doesn't have to recompute the prefill pass over that shared prefix every single time. This matters enormously for any application that sends a large, mostly-static context on every call: a long system prompt, a set of tool definitions, a retrieved document, or a multi-shot example set, followed by a small amount of genuinely new content (the user's latest turn) at the end.

Recall from how transformer inference works: generating the first output token requires a "prefill" pass over the entire input prompt to build up the KV cache for every input token, and this prefill cost scales with prompt length. If your system prompt plus tool definitions plus retrieved context is 3,000 tokens and the actual new user message is 20 tokens, you're paying full prefill compute for those 3,000 tokens on *every single request*, even though they're byte-identical across thousands of calls. Prompt caching lets the provider compute that prefill once, store the resulting KV cache, and reuse it for every subsequent request that shares the same prefix — turning an O(prompt length) prefill cost into an O(1) cache lookup for the cached portion, with fresh compute needed only for the new suffix tokens.

Different providers expose this with different mechanics, and knowing the shape of the differences matters for how you design prompts. Anthropic's prompt caching is explicit: you mark specific points in the prompt with a `cache_control` breakpoint, and everything up to that breakpoint becomes a cacheable unit; cached content has a short default TTL (on the order of minutes) refreshed on each cache hit, with an extended-TTL option for content you know will be reused over a longer window; there's a minimum prefix length (roughly 1,024 tokens for larger models, less for smaller ones) below which caching isn't worth the overhead and the API won't apply it. OpenAI's automatic prompt caching works with less explicit control — prompts over roughly 1,024 tokens are automatically eligible for caching against recently-used prefixes with no special markup required, trading control for simplicity. Gemini's context caching is the most explicit of the three: you create a named cache object ahead of time with its own TTL and pay a small storage cost for the cache's lifetime, then reference it by ID in subsequent calls — a model better suited to workloads where you know in advance you'll be reusing a specific large context many times (a large document you'll be asked many questions about, for instance) rather than an emergent, request-driven caching pattern.

```python
def build_cacheable_messages(system_prompt, tool_definitions, retrieved_context, user_message):
    """Structure a request so the static, reusable portion comes first and is
    marked as a cache boundary, and only the genuinely novel content comes last.
    This ordering is what makes provider-side prompt caching effective at all --
    if user-specific content is interleaved into the "shared" prefix, the whole
    prefix becomes unique per request and nothing is cacheable."""
    return [
        {
            "role": "system",
            "content": system_prompt + "\n\n" + tool_definitions,
            "cache_control": {"type": "ephemeral"},   # Anthropic-style breakpoint
        },
        {
            "role": "user",
            "content": retrieved_context,
            "cache_control": {"type": "ephemeral"},   # a second breakpoint further in
        },
        {
            "role": "user",
            "content": user_message,   # left uncached -- unique per request
        },
    ]
```

The cost model is worth internalizing because it's not simply "free": providers typically charge a small premium to *write* a cache entry (since it still requires the initial prefill), a steep discount (commonly 90% or more off the base input-token price) to *read* from an existing cache entry, and in Gemini's explicit-cache model, an ongoing storage cost for keeping the cache alive between uses. This means prompt caching pays off specifically when the same prefix is reused many times within the cache's TTL window — a single reuse might not recoup the write premium, but a system prompt hit thousands of times per hour recoups it almost immediately and then keeps paying dividends. The practical design implication is to structure your prompts deliberately: put everything static and shareable (system instructions, tool schemas, few-shot examples, a retrieved document that multiple turns of a conversation will reference) as early as possible and behind a cache boundary, and push anything request-unique (the current user turn, a timestamp, a session ID) to the very end, since any earlier position (or any tiny variation, even whitespace) after a cache breakpoint changes the token sequence just enough to invalidate the cache for everything downstream of that change.

## Composing the Layers in a Real Request Path

A production system runs these three layers as a funnel, cheapest and most precise checks first:

```python
async def handle_generation_request(query, session, exact_cache, semantic_cache,
                                      intent_classifier, llm_client):
    # Layer 1: exact match, always safe if it hits
    exact_hit = exact_cache.get(session.model, session.messages_with(query), session.params)
    if exact_hit:
        return exact_hit, "exact_cache_hit"

    # Layer 2: semantic match, but only for query types where staleness/personalization risk is low
    if route_to_cache_layer(query, intent_classifier) == "semantic_cache":
        semantic_hit = semantic_cache.lookup(query)
        if semantic_hit:
            return semantic_hit, "semantic_cache_hit"

    # Layer 3: no full-response cache hit, but the request is still built to maximize
    # provider-side prompt cache reuse on the static prefix (system prompt, tools, context)
    messages = build_cacheable_messages(
        session.system_prompt, session.tool_defs, session.retrieved_context, query
    )
    response = await llm_client.generate(model=session.model, messages=messages, params=session.params)

    exact_cache.set(session.model, session.messages_with(query), session.params, response)
    if route_to_cache_layer(query, intent_classifier) == "semantic_cache":
        semantic_cache.store(query, response)

    return response, "generated"
```

Note the ordering isn't arbitrary: exact-match is checked first because it's the cheapest operation (a hash lookup) and the only layer with zero risk of serving a wrong-but-similar answer. Semantic cache is checked second, gated by the intent classifier, because it's more expensive (an embedding call plus a vector search) and carries real correctness risk if misapplied. Provider-level prompt caching isn't a "check" at all in this flow — it's a structural property of how you build the request, engaged automatically by the provider whenever both layers above miss, so it costs nothing extra to attempt and simply reduces the cost/latency of the generation call that happens anyway.

## Invalidation, Staleness, and Measuring What You Built

Cache invalidation earns its reputation as one of the two hard problems in computer science partly because LLM caching adds a dimension that plain data caching doesn't have: even when the underlying facts haven't changed, the *desired behavior* can change — you ship a new system prompt, and every cached response generated under the old prompt is now stale even though nothing about the user's query changed. The clean fix is to always include a prompt/config version identifier as part of every cache key (exact-match) or as metadata checked before serving (semantic), so that a prompt deployment automatically and correctly invalidates old cache entries without needing an explicit flush step that someone might forget to run.

```python
def cache_key_with_version(model, messages, params, prompt_version):
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params, "prompt_version": prompt_version},
        sort_keys=True,
    )
    return f"llm_cache:{hashlib.sha256(payload.encode()).hexdigest()}"
```

Finally, treat cache hit rate and the resulting cost/latency savings as first-class metrics, not an afterthought — they're the entire justification for the engineering complexity of a caching layer, and they degrade silently if left unmonitored (an embedding model upgrade that isn't matched by re-tuning the semantic threshold, for instance, can quietly collapse a semantic cache's hit rate to near zero without any errors being thrown). Track hit rate, cost saved (estimated as hits times the average cost of the request that would have been generated), and — specifically for the semantic layer — a sampled false-positive rate from periodic human or LLM-judge review of served cache hits, since that's the one failure mode that won't show up as an error or a latency spike, only as a slow, easy-to-miss degradation in answer quality.

## Cache Stampedes and the Thundering Herd Problem

A subtle failure mode shows up precisely when caching is working well: a popular query's cache entry expires (TTL elapses, or a deploy invalidates it), and if that query is popular enough, dozens or hundreds of concurrent requests can all miss the cache in the same instant and all independently trigger a fresh, expensive LLM generation for what is effectively the same request — the exact multiplicative cost blowup caching was supposed to prevent, concentrated into a single unlucky moment. This is the classic "cache stampede" or "dog-piling" problem from traditional caching systems, and it applies to LLM response caches with even higher stakes than a database cache, because the cost of each redundant regeneration is so much higher than a redundant database query would be.

The standard fix is request coalescing: the first request that misses the cache for a given key acquires a short-lived lock (or registers itself as "in flight" in the cache store) and proceeds to generate, while every subsequent request for the same key, arriving while generation is still in flight, waits on that same in-flight computation instead of independently starting a new one.

```python
import asyncio

class CoalescingCache:
    def __init__(self, cache):
        self.cache = cache
        self.in_flight = {}   # key -> asyncio.Future, shared by concurrent waiters

    async def get_or_generate(self, key, generate_fn):
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if key in self.in_flight:
            # Someone else is already generating this -- await their result instead
            # of starting a redundant, costly duplicate generation.
            return await self.in_flight[key]

        future = asyncio.get_event_loop().create_future()
        self.in_flight[key] = future
        try:
            result = await generate_fn()
            self.cache.set(key, result)
            future.set_result(result)
            return result
        finally:
            del self.in_flight[key]
```

A second, complementary mitigation is staggering expiration rather than letting many entries expire at the same fixed instant: adding a small random jitter to each entry's TTL (`ttl = base_ttl + random.uniform(0, base_ttl * 0.1)`) spreads out expirations that would otherwise cluster — for instance, every cache entry written during a single traffic burst expiring in the same follow-up burst an hour later — smoothing out the load spike that would otherwise hit the LLM provider all at once.

## Negative Caching

It's easy to assume caching only applies to successful responses, but caching the fact that a request *failed* (or was rejected by a guardrail, or resulted in a refusal) is often just as valuable. If a particular malformed input, a known jailbreak attempt, or a request type your guardrails reliably reject arrives repeatedly, there's no reason to spend a full LLM call re-discovering that rejection every time — caching the rejection (with a shorter TTL than a normal success, since you want to periodically re-check in case the policy or model changes) saves the same cost and latency a positive cache hit would, for a class of traffic that positive-response caching structurally can't help with.

```python
def cached_generate_with_negative_cache(query, cache, guardrail_fn, generate_fn):
    cached = cache.get(query)
    if cached is not None:
        if cached.get("rejected"):
            return None, "cached_rejection"
        return cached["response"], "cache_hit"

    guardrail_result = guardrail_fn(query)
    if not guardrail_result.allowed:
        cache.set(query, {"rejected": True, "reason": guardrail_result.reason}, ttl=300)
        return None, "rejected"

    response = generate_fn(query)
    cache.set(query, {"rejected": False, "response": response}, ttl=3600)
    return response, "generated"
```

## Hierarchical Caching: Local, Distributed, and Provider Layers

Real production systems typically stack an additional, even cheaper tier in front of everything discussed so far: a small in-process (in-memory) LRU cache on each application server, holding only the hottest handful of recent entries, checked before the network round trip to Redis. This local tier can't hold much and doesn't share state across server instances, but for the small set of genuinely hot keys (a trending FAQ, a viral shared prompt) it eliminates even the Redis network hop, which matters when a caching layer's own latency starts to become a meaningful fraction of total response time for a cache hit.

```python
from functools import lru_cache
import time

class TieredCache:
    def __init__(self, redis_cache, local_maxsize=256, local_ttl=30):
        self.redis_cache = redis_cache
        self.local_ttl = local_ttl
        self.local_store = {}   # key -> (value, inserted_at)
        self.local_maxsize = local_maxsize

    def get(self, key):
        local_hit = self.local_store.get(key)
        if local_hit and (time.time() - local_hit[1]) < self.local_ttl:
            return local_hit[0]

        redis_value = self.redis_cache.get(key)
        if redis_value is not None:
            self._store_local(key, redis_value)
        return redis_value

    def _store_local(self, key, value):
        if len(self.local_store) >= self.local_maxsize:
            oldest_key = min(self.local_store, key=lambda k: self.local_store[k][1])
            del self.local_store[oldest_key]
        self.local_store[key] = (value, time.time())
```

Putting the full picture together, a request in a mature system checks, in increasing order of cost: an in-process local cache (sub-millisecond, server-local), a distributed exact-match cache (single-digit milliseconds, shared across the fleet), a semantic cache gated by intent classification (tens of milliseconds, an embedding call plus a vector search), and finally a generation call structured to maximize provider-side prompt cache reuse on its static prefix. Each tier exists because it catches a kind of redundancy the tier before it structurally cannot, and the aggregate effect on both cost and latency compounds — teams that implement all four layers well commonly report 40-70% of traffic served without a full fresh generation, which is a direct, multiplicative reduction in both LLM spend and median user-facing latency.
