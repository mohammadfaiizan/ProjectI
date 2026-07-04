# Production RAG System Design

## From Prototype to Production: What Actually Changes

Every earlier chapter in this section has, implicitly, been about a system that looks like this: a fixed set of documents gets chunked and embedded once, loaded into a vector index, and then queried by a single user (you, in a notebook, or a small internal demo audience) who is fairly forgiving about a slow or occasionally wrong answer. That system is enormously useful for developing intuition about chunking, embeddings, hybrid retrieval, reranking, and evaluation — but it is not the system a senior or staff engineer is being asked to design when an interviewer says "design a production RAG system for our support ticket corpus" or "design RAG search over our internal wiki for 5,000 enterprise customers." The gap between the prototype and the production system is not a matter of scaling up the same components; it's a matter of the underlying assumptions changing in ways that invalidate the prototype's design entirely. This chapter is about those changed assumptions and the architecture that answers them.

Five assumptions break when a RAG system goes from prototype to production, and each one motivates a section of this chapter:

**The document set stops being static.** A prototype indexes once and never touches the index again. A production system ingests a continuously changing corpus — new support tickets every minute, wiki pages edited throughout the day, product documentation that ships alongside every release, contracts uploaded by users at arbitrary times. If your only tool for handling a document update is "re-run the full indexing pipeline," you are stuck choosing between staleness (batch-reindex nightly, and users see wrong answers about anything that changed today) or runaway cost (reindex constantly, and your embedding bill and compute grow with the full corpus size on every single change, not with the size of the change). Section 2 is about escaping that trade-off with incremental, event-driven indexing.

**Freshness becomes an explicit, cost-bearing requirement, not a side effect of run-once indexing.** Once documents change continuously, you have to decide, deliberately, how fresh "fresh enough" is for each part of the system, and you have to build the mechanics — tombstoning, versioning, cache invalidation — that make a chosen freshness guarantee actually hold. Section 3 covers this.

**A single logical corpus becomes many tenants' worth of data that must never leak into each other's answers.** A prototype has one implicit tenant: whoever is running the notebook. A production system serving multiple customers, teams, or user accounts against the same infrastructure has to guarantee that tenant A's private documents can never appear in tenant B's retrieved context — a correctness requirement that is also a security requirement, and one of the most common places production RAG systems actually fail in the wild. Section 4 covers the architectural options.

**Latency and cost stop being informal and become budgets you are held to.** A demo can take eight seconds and nobody minds. A production system usually has an SLA — some target end-to-end response time, often in the low single-digit seconds for a synchronous request-response product — and a cost-per-query figure that has to make the product's unit economics work at the traffic volume you actually expect. Section 5 walks through how to build and defend that budget stage by stage.

**Failures have to be caught by the system, not by a human noticing a bad answer in a demo.** In a prototype, "the retrieval seems off today" is diagnosed by you, personally, reading the output. In production, at scale, across many tenants and thousands of queries a day, nobody is reading every answer — which means the system needs monitoring and alerting that catches regressions in retrieval quality, generation faithfulness, and latency automatically, tied to the same evaluation methodology from Chapter 8 but running continuously against live traffic rather than as an occasional offline exercise. Sections 6 and 7 cover the concrete failure modes this monitoring has to catch and what the monitoring stack itself looks like.

None of this replaces anything from the earlier chapters — the chunking strategy, embedding model, hybrid retrieval, reranker, and advanced retrieval architecture you chose are all still exactly as important. This chapter is about the operational shell that has to exist around those choices before the system can be trusted with real traffic and real tenants.

## Incremental and Streaming Indexing

### Why Full Re-Indexing Doesn't Scale

The naive indexing pipeline treats every change — one edited paragraph in one document — as a reason to re-chunk and re-embed the entire corpus. This fails along two independent axes as the corpus grows, and it's worth being explicit about both because interviewers often want to hear that you understand they are separate problems with separate causes.

The first axis is cost. Re-embedding the whole corpus costs roughly `O(corpus size)` in embedding API calls or GPU-time, every time it runs, regardless of how small the actual change was. If you have ten million chunks and one document's one paragraph changed, a full reindex still touches all ten million chunks. Run that nightly against a corpus that's growing, and your indexing cost grows without bound even if your rate of actual content change stays flat.

The second axis is staleness, and it's a latency problem in its own right: if reindexing is expensive enough that you can only afford to run it, say, once a day, then any document edited five minutes after the last run is invisible — or worse, answered from a stale, superseded version — for up to 24 hours. For content that changes slowly (a company's founding history) this doesn't matter. For content that changes fast (a support ticket's current status, a product's current pricing, an internal wiki page mid-edit) a full day of staleness produces confidently wrong answers, which is a worse failure mode than a slow-but-correct answer because users trust the system less afterward.

The fix is to decouple "a change happened somewhere in the corpus" from "we do work proportional to the whole corpus." That decoupling has three components in most production systems: content-hash-based change detection at the chunk level, an event-driven trigger that fires only for the document that actually changed, and a queue that absorbs bursty ingestion so the embedding/upsert stage runs at a steady, controllable rate rather than spiking with the source.

### Content-Hash Change Detection at the Chunk Level

The key realization is that the unit of "did this change" should be the chunk, not the parent document. A single-page wiki article might produce forty chunks after the chunking pipeline from Chapter 2 runs over it; if an editor fixes one typo in one paragraph, only the one or two chunks derived from that paragraph actually changed. Re-embedding the other thirty-eight is pure waste — the surrounding text, and therefore the embedding, of the untouched chunks hasn't moved at all.

The mechanism is simple and doesn't require anything exotic: store a content hash (SHA-256 over the normalized chunk text is standard) alongside every chunk's embedding and metadata in your indexing state store. When a document is re-chunked after an edit, compute the hash of every resulting chunk, compare it against the previously stored hash for the chunk occupying that position (or matched by a stable chunk ID if your chunker assigns one), and only send chunks with a changed hash to the embedding model. Chunks whose hash matches what's already stored are left untouched in the vector index — no embedding call, no upsert, no cost.

```python
import hashlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkRecord:
    chunk_id: str          # stable id, e.g. f"{doc_id}::{chunk_index}"
    doc_id: str
    text: str
    content_hash: str
    embedding: Optional[list] = None


def content_hash(text: str) -> str:
    # Normalize before hashing so irrelevant whitespace/casing
    # differences don't trigger spurious re-embeddings.
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class IncrementalIndexer:
    """Re-chunks a changed document and only re-embeds chunks whose
    content actually changed, leaving unchanged chunks untouched
    in the vector store and the hash store."""

    def __init__(self, embed_fn, upsert_fn, delete_fn, hash_store: dict):
        self.embed_fn = embed_fn        # text -> embedding vector
        self.upsert_fn = upsert_fn      # (chunk_id, embedding, metadata) -> None
        self.delete_fn = delete_fn      # chunk_id -> None
        self.hash_store = hash_store    # chunk_id -> last-known content_hash

    def index_document(self, doc_id: str, chunks: list[str]) -> dict:
        new_records = [
            ChunkRecord(
                chunk_id=f"{doc_id}::{i}",
                doc_id=doc_id,
                text=text,
                content_hash=content_hash(text),
            )
            for i, text in enumerate(chunks)
        ]

        new_ids = {c.chunk_id for c in new_records}
        old_ids = {cid for cid in self.hash_store if cid.startswith(f"{doc_id}::")}

        stats = {"embedded": 0, "skipped": 0, "deleted": 0}

        # Chunks that no longer exist after re-chunking (doc shrank,
        # or chunk boundaries shifted) must be removed, not left as
        # orphaned stale vectors that can still be retrieved.
        for stale_id in old_ids - new_ids:
            self.delete_fn(stale_id)
            del self.hash_store[stale_id]
            stats["deleted"] += 1

        for record in new_records:
            previous_hash = self.hash_store.get(record.chunk_id)
            if previous_hash == record.content_hash:
                stats["skipped"] += 1
                continue  # text unchanged -> embedding is still valid, skip it

            record.embedding = self.embed_fn(record.text)
            self.upsert_fn(
                record.chunk_id,
                record.embedding,
                {"doc_id": doc_id, "content_hash": record.content_hash},
            )
            self.hash_store[record.chunk_id] = record.content_hash
            stats["embedded"] += 1

        return stats
```

This is the whole mechanism — it's deliberately unglamorous. What makes it production-grade is where it's invoked from and how ingestion spikes are handled around it, which is the next two pieces.

### Event-Driven and CDC-Style Triggers

The indexer above needs to be called exactly when a document actually changes, and only with that document — not on a schedule that scans the whole corpus looking for changes. Two patterns dominate here. If the source of truth is a document store, CMS, or wiki with webhook support, register a webhook that fires on create/update/delete and carries the document ID (and ideally the new content) in the payload; the webhook handler enqueues a re-chunk-and-embed job for that one document. If the source of truth is a database (tickets, product catalog rows, structured records that get turned into text for embedding), a change-data-capture stream off the database's write-ahead log (Debezium against Postgres/MySQL binlogs is the common open-source choice) gives the same effect without requiring the source system to have first-class webhook support — every row insert/update/delete becomes an event, and a consumer turns it into an indexing job scoped to that row's derived document.

Either pattern converts "the corpus changed" from a fact you'd otherwise have to discover by periodically diffing the whole corpus into a fact you're told about, in real time, at the granularity of the single document that changed. That granularity is what makes the content-hash check in the indexer effective — you're only ever re-chunking one document per event, and within that document only the changed chunks pay the embedding cost.

### Queue-Based Decoupling of Ingestion and Embedding

Even with per-document triggers, real ingestion is bursty: a CMS migration pushes ten thousand document updates in a minute, or a nightly batch job from an upstream system dumps a day's worth of changed rows all at once. If the webhook or CDC handler calls the embedding API synchronously and inline, a burst like that either overwhelms the embedding provider's rate limits or creates a large backlog of blocked webhook handlers, and either way indexing latency for everyone — including documents that changed in a quiet period — becomes unpredictable.

The standard fix is a queue between event ingestion and the embed/upsert work: the webhook or CDC consumer's only job is to validate the event and push a small message (document ID, or the row's changed data) onto a queue (SQS, Kafka, Pub/Sub, or a Celery/RQ-backed job queue for smaller deployments), and a separate, horizontally scalable pool of workers pulls from that queue at a controlled concurrency and does the actual re-chunk/hash-check/embed/upsert work from the indexer above. This decoupling means ingestion spikes are absorbed as queue depth rather than as failed API calls or backpressure on the source system, embedding throughput can be tuned independently (worker count, batch size, rate limiting against the embedding provider's quota) from ingestion volume, and a slow or temporarily-down embedding provider degrades to "the queue grows and indexing lags a bit" rather than "webhook deliveries start failing and events are lost." The trade-off is exactly the freshness-vs-throughput tension from the framing section: a deep queue under load means some documents take longer to become searchable, which is a deliberate, tunable choice rather than an accident, and it's usually the right one compared to the alternatives.

### Backfill and Reconciliation for Missed Events

Event-driven pipelines have a failure mode of their own that a batch-reindex pipeline never has to worry about: an event can simply be lost — a webhook delivery fails and isn't retried, a CDC connector falls behind and drops offsets after a broker retention window expires, or a worker crashes mid-job after dequeuing a message but before acknowledging it, depending on the queue's delivery guarantees. Any of these leaves a document permanently stale in the index with nothing in the event stream to ever trigger reprocessing, and because the failure is silent, it looks identical from the outside to a document that simply hasn't changed. Relying purely on events for correctness is therefore not enough; production pipelines pair the event-driven path with a periodic, low-priority reconciliation job that walks the source corpus (or a change-tracking table with `updated_at` timestamps) and compares each document's current content hash against what the index has on record, enqueuing a re-index job for anything that's drifted. This reconciliation pass is deliberately cheap to run relative to a full reindex — it's a hash comparison and a diff, not a re-embed of everything — and it exists purely as a correctness backstop underneath the event-driven fast path, catching the small number of documents the event stream missed rather than being the primary mechanism for freshness.

## Freshness, Versioning, and Cache Invalidation

### The Freshness/Cost Tension

Incremental indexing solves the cost side of staleness — you're no longer paying to re-embed the whole corpus — but it doesn't remove the fact that indexing takes some nonzero time after a source document changes, and during that window the index is answering from stale content. For most content this window (seconds to low minutes, with the queue-based pipeline above) is fine. For content where staleness is actively harmful — a ticket's live status, a price, a policy that changed this morning — the system needs an explicit answer to "how do we make sure a query never gets a chunk we already know is wrong," and the answer generally isn't "make indexing infinitely fast," because that reintroduces the cost problem. It's tombstoning and versioning.

### Tombstoning Deleted or Superseded Content

When a document is deleted or superseded, the naive approach — leave its chunks in the vector index until the next full reindex cleans them up — means a deleted document can still be retrieved and cited in an answer for however long that cleanup interval is, which for a low-frequency batch cleanup job can be days. Tombstoning fixes this by treating deletion as an immediate, cheap metadata flip rather than a deferred, expensive removal: when a document is deleted, a `deleted: true` (or `superseded_by: <new_doc_id>`) flag is set on its chunks' metadata immediately, and every retrieval query filters out tombstoned chunks as part of its metadata filter — the same filtering mechanism used for tenant isolation in Section 4. The actual removal of the underlying vectors from the index (freeing storage, shrinking the ANN graph) happens later, in a cheap batch compaction pass, but it's decoupled from correctness: correctness is guaranteed the instant the tombstone flag is set, because the retrieval query simply never sees tombstoned chunks, regardless of when physical deletion actually runs.

### Versioning: Effective Timestamps and Pinning to a Version

Some products need more than "always serve the latest version and hide anything superseded" — they need to answer questions against a specific historical state, either because a user is asking "what did the contract say in March" or because an audit trail requires reproducing exactly what the system would have answered at a past point in time. The mechanism is to attach a version identifier and an effective-timestamp range to every chunk (`effective_from`, `effective_to`, with `effective_to` null or infinite for the current version) rather than overwriting a chunk in place when its source document changes. A query for "latest" filters to chunks where `effective_to` is null; a query pinned to a historical point in time filters to chunks where the requested timestamp falls inside `[effective_from, effective_to)`. This is more storage (old chunk versions are retained rather than overwritten) in exchange for the ability to reconstruct any past state exactly — a trade-off that's usually worth it for regulated domains (legal, financial, healthcare) and unnecessary overhead for domains where only the current answer ever matters.

### Caching Layers and the Invalidation Problem They Introduce

Three distinct caching layers show up in a mature RAG pipeline, each targeting a different repeated-cost pattern. An **embedding cache** keyed on the normalized query text avoids re-embedding identical or near-identical queries, which matters most for high-frequency queries (a FAQ-style question asked by many different users) since embedding cost is small per call but not zero at scale. A **retrieval result cache** keyed on (query, tenant, filters) stores the full ranked chunk list for hot queries, skipping both the embedding call and the vector search entirely on a cache hit — this is where the latency win is largest, since first-stage retrieval and reranking are the more expensive steps. A **generation cache** keyed on (query, retrieved-context-hash) stores the full LLM output, skipping the most expensive stage entirely when the exact same question is asked against the exact same underlying context; because it's keyed on a hash of the actual context (not just the query), it correctly misses if retrieval returns different chunks even for a repeated query.

The invalidation problem is the same one that makes cache invalidation famously hard everywhere, but RAG gives it a specific, concrete shape: a cached answer is only valid as long as the source documents it was built from haven't changed underneath it. If document D changes and a cached generation was built from a chunk of D, that cache entry is now potentially answering from stale information and has to be invalidated — but the cache doesn't know, by default, which entries depended on which documents, because the cache key is the query and context hash, not a list of source document IDs. The fix is to maintain a reverse index from document ID to the set of cache keys that were built using a chunk from that document (a simple `doc_id -> {cache_key, ...}` mapping, updated at cache-write time whenever a generation or retrieval result is cached), so that when the incremental indexer processes an update to a document, it can look up and evict every cache entry that depended on it in the same operation, rather than either leaving stale answers cached indefinitely or invalidating the entire cache on every document change (which throws away the hit rate on everything unrelated to that document). This reverse-index-driven invalidation is the piece most prototype-grade caching implementations skip, and it's exactly the piece that makes caching safe to turn on for a corpus that changes continuously rather than only for a static one.

```python
class DependencyTrackedCache:
    """Generation/retrieval cache that records which source documents
    each cache entry was built from, so a document update can evict
    exactly the entries it invalidates -- no more, no less."""

    def __init__(self):
        self.cache: dict[str, dict] = {}          # cache_key -> value
        self.doc_to_keys: dict[str, set] = {}      # doc_id -> {cache_key, ...}

    def put(self, cache_key: str, value, source_doc_ids: set[str]):
        self.cache[cache_key] = value
        for doc_id in source_doc_ids:
            self.doc_to_keys.setdefault(doc_id, set()).add(cache_key)

    def get(self, cache_key: str):
        return self.cache.get(cache_key)

    def invalidate_document(self, doc_id: str) -> int:
        # Called by the incremental indexer immediately after a document
        # is re-embedded, so stale answers never outlive the source edit.
        stale_keys = self.doc_to_keys.pop(doc_id, set())
        for key in stale_keys:
            self.cache.pop(key, None)
        return len(stale_keys)
```

## Multi-Tenant Document Isolation

### Three Architectural Approaches

Serving multiple tenants — separate customers, separate teams, or separate user accounts, each of whom must never see another's private documents in their retrieved context — is one of the defining differences between a prototype and a production RAG system, and it has to be solved at the retrieval layer, not just at the application layer (more on why below). Three architectural patterns cover most real deployments.

**A fully separate vector index or collection per tenant** gives the strongest isolation: tenant A's documents physically live in a different index than tenant B's, so there is no query path by which a bug in a metadata filter could leak data across tenants — the isolation is structural, not a runtime check that has to be gotten right every time. Security reasoning about this design is simple enough to explain in one sentence to an auditor, which matters a great deal in regulated industries. The cost is resource utilization and operational overhead: most vector databases have per-index overhead (memory for the ANN graph, connection/index-management overhead), so thousands of small tenants each getting a dedicated index can mean paying that fixed overhead thousands of times over for tenants who individually have a tiny corpus, and operationally, provisioning, monitoring, and upgrading thousands of indexes is a materially harder problem than managing one.

**A shared index with a mandatory tenant-id metadata filter on every query** flips the trade-off: one index holds every tenant's chunks, tagged with a `tenant_id` field, and every retrieval query includes a filter clause restricting results to the requesting tenant's `tenant_id`. Resource utilization is far better — one index amortizes its fixed overhead across all tenants, and a small tenant's corpus doesn't need its own dedicated infrastructure. The catch, and it's a real one, is that this design's correctness and performance both depend entirely on how well the vector store implements *filtered* ANN search. A naive implementation runs the ANN search first and applies the tenant filter afterward as a post-filter step; if the tenant's chunks happen to be sparse within the region of the vector space the ANN index searches, a post-filter can silently return fewer than the requested `k` results (or even zero) even though matching chunks exist elsewhere in the index, because the ANN search never surfaced them as candidates before filtering. A well-implemented vector store instead pushes the filter into the search itself (pre-filtering or filter-aware graph traversal, depending on the index type), so the ANN search only ever considers candidates within the tenant's partition — this is a vector-store-specific capability, and evaluating it (does filtered search on this store degrade recall or add meaningful latency as filter selectivity increases) is exactly the kind of vector-store selection question covered in the vector databases chapter; the point to carry into this chapter is that "shared index with metadata filtering" is only actually safe and correct if that filtering is efficient and complete, and that has to be verified empirically against the specific vector store, not assumed.

**A hybrid, sharded approach** splits the difference by tiering tenants: a small number of very large or contractually-demanding enterprise tenants each get a dedicated index (paying the per-index overhead, but justified by their scale and isolation requirements), while a long tail of small tenants share one or more pooled indexes with tenant-id filtering. This is the pattern most B2B SaaS RAG products converge on in practice, because it matches the actual shape of their tenant distribution — a handful of large accounts and many small ones — rather than treating every tenant identically. Making this work operationally requires a routing layer that every query and every indexing job passes through before it can decide which physical index to talk to:

```python
def resolve_index_for_tenant(tenant_registry, tenant_id: str) -> str:
    """Central routing decision: dedicated index for large/contractual
    tenants, a pooled shard for everyone else. Both indexing jobs and
    retrieval queries must call this -- never hardcode or infer the
    target index any other way, or the two paths can silently drift
    and start writing/reading from different places for the same tenant.
    """
    tenant = tenant_registry.get(tenant_id)
    if tenant.tier == "enterprise" or tenant.chunk_count > DEDICATED_INDEX_THRESHOLD:
        return f"dedicated::{tenant_id}"
    # Deterministic hash-based bucketing keeps a given small tenant on
    # the same pooled shard across restarts, which matters for caching
    # locality and for capping how many tenants share one shard.
    shard_id = hash(tenant_id) % NUM_POOLED_SHARDS
    return f"pooled::shard-{shard_id}"


def search_for_tenant(vector_store, query_embedding, tenant_id: str, top_k: int = 10):
    """Every retrieval call is scoped to exactly one tenant. The filter
    is applied by the vector store's query API itself (server-side),
    never by fetching unfiltered results and filtering in application
    code -- filtering after retrieval both leaks capacity (you fetched
    another tenant's data into your process) and can under-return if
    the true top-k for this tenant isn't within the unfiltered top-k.
    """
    return vector_store.query(
        vector=query_embedding,
        top_k=top_k,
        filter={"tenant_id": {"$eq": tenant_id}},   # enforced server-side
        include_metadata=True,
    )
```

### The Noisy Neighbor Problem

Shared indexes introduce a resource-contention failure mode that dedicated-per-tenant indexes don't have: one tenant with an unusually large corpus, or one whose documents are updated unusually frequently (constant re-embedding and upserts), can degrade search latency or index quality for every other tenant sharing that index, purely as a side effect of sharing infrastructure — this is the "noisy neighbor" problem, borrowed from the same term in shared-database and shared-compute contexts. A large tenant's chunks can dominate an ANN index's graph structure in ways that make traversal slower for everyone querying that index, or a tenant running frequent bulk re-indexing can saturate the shared upsert throughput and cause other tenants' incremental updates to queue up behind it. This is precisely the argument for the hybrid/sharded approach above: identifying tenants whose size or update frequency is far outside the norm and migrating them to a dedicated index isn't just a security or compliance nicety, it's a direct fix for a performance problem that a purely shared architecture cannot otherwise contain.

### Tenant Filtering Must Be Enforced at the Retrieval-Query Level

This point is worth stating as its own explicit rule because it is a genuinely common real-world bug class, not a theoretical concern: tenant isolation must be enforced inside the retrieval query itself — as a server-side filter clause evaluated by the vector store, as shown above — and never only at the application or UI layer. A design where the application fetches results without a tenant filter and then filters the returned list in Python before displaying it to the user is not tenant isolation at all; it's tenant isolation of the UI, while the underlying retrieval call, the LLM prompt built from unfiltered results, and any logging or caching layer downstream of that call have already been exposed to cross-tenant data. Concretely, if the retrieval step doesn't filter by tenant, then the "irrelevant" chunks from other tenants are still fetched into the process, still eligible to be selected by a reranker, and — if a filtering bug in the UI layer is ever introduced or bypassed (a common occurrence when a new code path, like a debug endpoint or an internal admin tool, is added without threading the same filter through) — still capable of appearing directly in a response shown to the wrong tenant. The fix is architectural discipline: the tenant ID should be a mandatory, non-optional parameter of the retrieval function itself (not a filter applied to its output), ideally enforced at the lowest layer that talks to the vector store, so that it is structurally impossible to call retrieval without specifying whose data to search.

## Cost and Latency Budget Across the Pipeline

### Building an End-to-End Budget

A production RAG system usually has a target end-to-end latency — commonly somewhere in the 2-3 second range for an interactive product, tighter for anything advertised as "instant," looser for anything explicitly framed as doing deep research — and the way to hit that target reliably is to allocate it explicitly across pipeline stages rather than hoping the sum comes in under budget. A representative allocation for a 2.5 second budget looks like this:

| Stage | Typical latency | Notes |
|---|---|---|
| Query embedding | 20-50 ms | One embedding call; cheap and fast unless queuing behind a shared rate limit |
| First-stage retrieval (ANN + optional sparse) | 50-150 ms | Scales with index size and filter selectivity, not corpus size directly |
| Reranking | 100-300 ms | Scales with candidate pool size (Chapter 6) and cross-encoder size |
| Prompt construction | 5-20 ms | Formatting, deduplication, truncation; CPU-bound, essentially free |
| LLM generation | 1500-2200 ms | Dominant cost; scales with prompt + completion tokens |

The generation stage dominates both the latency and the cost budget in almost every real deployment, which has two important implications. First, optimizing the other four stages below a certain point stops mattering for perceived latency — shaving 50ms off retrieval when generation takes 2 seconds is not where the budget is won or lost. Second, and more actionable: streaming the LLM's output token-by-token to the user, rather than waiting for the full completion before displaying anything, doesn't reduce total generation time at all, but it changes perceived latency dramatically, because the user sees the first tokens arrive in a few hundred milliseconds and reads the rest as it streams in, rather than staring at a blank loading state for the full 1.5-2.2 seconds. Nearly every production RAG product streams generation for exactly this reason — it's one of the highest-leverage, lowest-risk latency interventions available, and it costs nothing in actual compute.

### Cost Follows the Same Shape as Latency

The cost budget mirrors the latency budget: embedding cost per query is close to negligible (one short query embedded per request, against models priced in fractions of a cent per thousand tokens), vector search cost scales with queries-per-second and index size (more relevant to infrastructure capacity planning than per-query cost accounting), reranking cost scales with how many candidates are pushed through the cross-encoder, and LLM generation cost — priced per input and output token — is typically the dominant line item, often by an order of magnitude over every other stage combined, especially once a large system prompt, retrieved context, and conversation history are all counted as input tokens on every single call.

```python
def per_query_cost(
    embedding_cost_per_1k_tokens: float,
    query_tokens: int,
    rerank_cost_per_1k_candidates: float,
    num_candidates_reranked: int,
    llm_input_cost_per_1k_tokens: float,
    llm_output_cost_per_1k_tokens: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    embedding_cost = (query_tokens / 1000) * embedding_cost_per_1k_tokens
    rerank_cost = (num_candidates_reranked / 1000) * rerank_cost_per_1k_candidates
    llm_cost = (
        (prompt_tokens / 1000) * llm_input_cost_per_1k_tokens
        + (completion_tokens / 1000) * llm_output_cost_per_1k_tokens
    )
    total = embedding_cost + rerank_cost + llm_cost
    return {
        "embedding_cost": embedding_cost,
        "rerank_cost": rerank_cost,
        "llm_cost": llm_cost,
        "total_cost": total,
        "llm_share_of_total": llm_cost / total if total else 0.0,
    }


# Illustrative numbers, not vendor-specific pricing:
breakdown = per_query_cost(
    embedding_cost_per_1k_tokens=0.00002,
    query_tokens=20,
    rerank_cost_per_1k_candidates=0.001,
    num_candidates_reranked=50,
    llm_input_cost_per_1k_tokens=0.003,
    llm_output_cost_per_1k_tokens=0.015,
    prompt_tokens=2500,   # system prompt + retrieved context + history
    completion_tokens=300,
)
print(breakdown)
# LLM generation typically comes out to 90%+ of total per-query cost
# once realistic prompt sizes (context-stuffed) are used.
```

### Defending the Budget Under Load

A budget built around typical-case (p50) latency per stage is necessary but not sufficient, because the stages don't degrade independently under load — a traffic spike, an upstream LLM provider slowdown, or a vector store under memory pressure can push any single stage well past its allotted slice, and the budget needs an explicit answer for what happens then, rather than letting p99 latency simply balloon unbounded. The standard approach is a set of graceful-degradation rules triggered by per-stage latency or error thresholds: if reranking is running hot (queue backing up or per-call latency crossing a threshold), fall back to serving the first-stage retrieval ranking unreranked rather than blocking the whole request on a saturated reranker; if the primary generation model's latency or error rate spikes, fail over to a smaller or faster backup model rather than letting requests time out entirely; if retrieval itself is slow (an overloaded shard, a noisy-neighbor tenant per Section 4), cap `top_k` more aggressively for the duration of the incident rather than serving a request that blows the entire latency budget on one stage. None of these are quality-neutral — a degraded response with fewer or unreranked chunks is worse than the normal path — but a slightly-worse-but-fast response under load is almost always the right trade against a technically-complete-but-slow-or-failed one, and building these fallbacks in deliberately, ahead of time, is what separates a budget that's actually defended from one that's just an aspirational number in a design doc.

### Levers for Each Stage

Because generation dominates both budgets, the highest-leverage cost and latency levers target it directly: trimming and compressing retrieved context before it goes into the prompt (deduplicating near-identical chunks, summarizing long chunks, dropping chunks the reranker scored low rather than including everything retrieval returned) reduces prompt tokens directly, and choosing an appropriately-sized generation model for the task — rather than defaulting to the largest available model for every query — is usually the single biggest cost lever available, since output tokens are typically priced several times higher than input tokens and larger models are priced higher still per token. That said, the other stages aren't free wins to ignore: embedding cost can be trimmed with a smaller embedding model or Matryoshka-style truncated embeddings (Chapter 3) when the corpus and query volume are large enough for the per-call savings to add up; retrieval cost and latency benefit from tighter `top_k` values and metadata pre-filtering that narrows the ANN search space before it runs (rather than retrieving broadly and filtering after, which both wastes compute and risks the under-filtering problem from Section 4); and reranking cost and latency scale directly with candidate pool size, so right-sizing how many candidates get pushed into the cross-encoder — per the trade-offs in Chapter 6 — is a direct lever on that stage specifically without touching retrieval or generation at all.

## Common Production Failure Modes

Production RAG systems fail in a recurring, learnable set of ways, most of which are invisible in a small-scale demo and only surface under real traffic, a growing corpus, or time (as models and indexes are upgraded). Knowing this list, and the mitigation for each, is one of the highest-value things to be able to produce fluently in a systems-design interview, because it signals you've actually operated one of these systems rather than only built the happy path.

**Retrieval-generation mismatch.** Retrieval does its job and returns the correct chunks, but the LLM either ignores them and answers from its parametric memory anyway (particularly likely when the retrieved context contradicts what the model "believes" from pretraining, or when the question is common enough that the model is confident it already knows the answer), or it misreads the context because multiple retrieved chunks contain conflicting information (an old and a new version of a policy both surfaced, for instance) and the model picks the wrong one or blends them incoherently. Mitigation: prompt the model explicitly to answer only from the provided context and to say so when the context is insufficient or conflicting, surface source attribution so conflicts are visible rather than silently resolved, and — since this is fundamentally a generation-faithfulness problem — track it with the faithfulness/groundedness metrics from Chapter 8 sampled continuously against production traffic, not just offline.

**Context window overflow and silent truncation.** When too many retrieved chunks (plus system prompt, conversation history, and instructions) are stuffed into the prompt, either the API truncates silently from one end or the application code's own token-budget logic drops chunks without surfacing that anything was dropped — the response comes back looking normal, but it was generated from an incomplete context, and nothing in the output signals that. Mitigation: compute the token budget explicitly before calling the LLM (reserve fixed space for system prompt, instructions, and expected completion length, then fit as many top-ranked chunks as the remainder allows), and log/alert whenever chunks had to be dropped to fit, so this is a monitored, visible event rather than a silent one.

**"Lost in the middle" degradation.** Even when every retrieved chunk fits comfortably inside the context window, long-context LLMs are empirically worse at using information placed in the middle of a long prompt compared to information near the start or the end — a well-documented effect independent of context overflow. A chunk that's genuinely the best match, if it lands in the middle of a long list of retrieved context, can be effectively ignored by the model even though it's technically present in the prompt. Mitigation: order retrieved chunks by relevance with the most important chunks placed near the beginning and/or end of the context block rather than in true rank order buried in the middle, and keep the total number of chunks passed to generation as small as reranking (Chapter 6) can responsibly make it, since fewer, better chunks reduce how much "middle" exists for anything to get lost in.

**Stale or mismatched embeddings after a silent model upgrade.** This is one of the most dangerous failure modes precisely because it's silent and produces plausible-looking (but meaningless) results rather than an obvious error: if the embedding model is upgraded — a new version, a different provider, even a minor point release with retrained weights — and only new or changed documents get embedded with the new model while the bulk of the existing index still holds vectors from the old model, cosine similarity between a query embedded with the new model and a document embedded with the old model is not a meaningful relevance signal, because the two vectors live in different, generally incompatible embedding spaces. The system doesn't error out; it just returns semantically wrong nearest neighbors that happen to have a numeric similarity score, so retrieval quality degrades without any exception being thrown anywhere. Mitigation: treat an embedding model change as a full-corpus re-embedding event, not an incremental one — every chunk needs a new vector from the new model before the new model is used for any query — and tag every stored vector with the embedding model version it was produced by so a mixed-version index can be detected programmatically (and blocked from serving) rather than discovered by users getting bad answers.

**Chunk-boundary information loss.** A fact needed to answer a question is split across a chunk boundary — half a sentence, or a table row and its header, land in different chunks — and neither chunk alone contains the complete fact, even though the source document clearly does. This traces directly back to the chunking strategy decisions in Chapter 2: fixed-size chunking with no overlap is the most exposed to this failure, while semantic or structure-aware chunking with sensible overlap reduces but does not eliminate it. Mitigation: chunk overlap sized to typical fact length for the domain, structure-aware chunking that respects natural boundaries (don't split a table from its header, don't split a numbered list item), and, where the corpus and query patterns justify it, techniques like small-to-big retrieval or including a chunk's immediate neighbors alongside it in the generation context so a boundary-split fact still ends up complete in the prompt even if only one of the two chunks scored high enough to be retrieved directly.

**Duplicate and near-duplicate documents inflating retrieval.** The same content ingested from multiple sources (a policy document that exists both in the wiki and as an attached PDF, or a support answer copy-pasted across several tickets) means several near-identical chunks compete for the same top-k slots, crowding out genuinely diverse relevant content and wasting reranking and generation budget on redundant copies of the same information. Mitigation: deduplicate at ingestion time using a similarity threshold on chunk embeddings or a min-hash/simhash style near-duplicate detector before chunks ever reach the index, and, as a runtime backstop, apply diversity-aware selection (such as maximal marginal relevance) at the retrieval or reranking stage so that even undetected near-duplicates don't all get selected together.

**Silent index corruption or drift going undetected.** A broken indexing job that partially fails and leaves a document half-indexed, a reranker or embedding model swapped in without validation, or gradual drift between the live corpus and what's actually reflected in the index — all of these degrade retrieval quality without producing an error anyone sees, because there's no automated evaluation running against production behavior to notice the regression, only users eventually noticing bad answers and complaining, by which point the problem has likely been live for days. Mitigation is the subject of the next section: continuous, automated retrieval-quality monitoring against production traffic, not just offline evaluation run once before a launch.

## Monitoring and Observability

Chapter 8 drew the distinction between offline evaluation (a curated test set of queries with known-relevant documents and/or reference answers, run against the pipeline before shipping a change) and online evaluation (signals gathered from real production traffic, where ground truth is rarely available directly). A production RAG system needs both running continuously, wired to alerting, because the offline suite alone only catches regressions you thought to test for at the point you last ran it — it says nothing about what's happening to real traffic between evaluation runs, and several of the failure modes above (embedding version drift, index corruption, a slow creep in duplicate content) are exactly the kind of thing that shows up in production traffic first.

A useful way to keep these signals honest is to write the alerting rules as direct, literal statements of the failure modes from the previous section, rather than as generic anomaly detection over an undifferentiated metric soup:

```python
def check_alerts(metrics: dict) -> list[str]:
    """Each rule is a direct, named statement of one of the failure
    modes from the previous section -- an alert should always be
    traceable back to a specific, known way the system can break."""
    alerts = []

    if metrics["embedding_versions_in_index"] > 1:
        alerts.append("MIXED EMBEDDING VERSIONS: index contains vectors "
                       "from more than one model version -- similarity "
                       "scores across them are meaningless.")

    if metrics["context_truncation_rate"] > 0.02:
        alerts.append("CONTEXT OVERFLOW: >2% of requests are dropping "
                       "retrieved chunks to fit the prompt budget.")

    if metrics["indexing_queue_age_seconds"] > 900:
        alerts.append("INDEXING BACKLOG: oldest unprocessed change is "
                       "over 15 minutes old -- freshness SLA at risk.")

    if metrics["retrieval_score_p50_delta_vs_baseline"] < -0.15:
        alerts.append("RETRIEVAL QUALITY REGRESSION: median top-1 "
                       "similarity score dropped >15% vs. baseline -- "
                       "possible index corruption or noisy neighbor.")

    if metrics["duplicate_chunk_rate_in_topk"] > 0.25:
        alerts.append("DUPLICATE INFLATION: >25% of top-k results are "
                       "near-duplicates of another result in the same set.")

    return alerts
```

Four categories of signal make up a reasonably complete monitoring stack. **Retrieval quality metrics sampled continuously in production** — since true relevance labels aren't available for live queries, this typically means a combination of proxy signals (click-through or explicit thumbs-up/down on cited sources, retrieval score distributions that would flag an unusual drop, agreement between the primary retriever and a periodically-run stronger reference retriever on a sampled subset of live queries) and a rotating sample of live queries sent to human or LLM-judge annotation on a regular cadence, which is the online analogue of the offline retrieval metrics (recall@k, MRR, NDCG) from Chapter 8. **Generation faithfulness spot-checks** — an LLM-as-judge pass (Chapter 8's methodology, applied continuously rather than once) run against a sample of production query/context/answer triples, scoring groundedness and flagging answers that appear to contradict or go beyond the retrieved context, which is the direct, ongoing defense against the retrieval-generation mismatch failure mode. **Latency percentiles broken out per pipeline stage** — p50/p95/p99 for embedding, retrieval, reranking, and generation individually, not just end-to-end, because an end-to-end p99 regression is far faster to diagnose when you can immediately see which stage's percentile moved rather than having to re-instrument after the fact; this is also where the budget from Section 5 earns its keep, since it gives you a concrete expected value per stage to alert against. **Alerting tied explicitly to the failure modes above** — an alert on a sudden shift in the distribution of number-of-chunks-retrieved-per-query or the rate of context truncation events (context overflow), an alert if any vector in the index is tagged with an embedding model version different from the currently-configured model (embedding version drift), an alert on retrieval score distributions dropping sharply for a tenant or shard (possible index corruption or a noisy-neighbor effect), and an alert on indexing queue depth or age of oldest unprocessed message (a stuck or backlogged incremental indexing pipeline, which left unnoticed silently turns into the staleness problem Section 2 was built to solve). None of these are exotic to build — they're mostly counters, histograms, and scheduled sampling jobs — but a production RAG system without them is a system where the failure modes in this chapter are guaranteed to eventually happen, and guaranteed to be discovered by a user rather than by the team that owns the system, which is precisely the gap the framing section opened with.
