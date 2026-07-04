# Indexing and Metadata Filtering

## Why Pure Vector Search Is Rarely Enough

Almost no real production retrieval system does pure, unconstrained nearest-neighbor search. A customer support search needs to search only within one company's documents. A multi-tenant SaaS product needs strict isolation between tenants' data. A recommendation system needs to exclude items already purchased, or restrict by category, price range, or availability. A RAG system over a document corpus needs to filter by access permissions, document freshness, or source type. The moment any of these requirements shows up — and they show up in essentially every real deployment — vector search stops being just "find the nearest points" and becomes "find the nearest points *among those satisfying a set of constraints*," and the way that constraint gets applied has significant, sometimes surprising, implications for both correctness and performance.

This chapter covers three interlocking problems: how to combine metadata filtering with ANN search without breaking either the accuracy or the speed guarantees that made you choose ANN in the first place, how to structure indexes for multi-tenant workloads, and why updating or deleting vectors from an ANN index is a genuinely harder problem than updating a row in a relational table.

## Pre-Filtering vs Post-Filtering

The naive approach to combining a metadata filter with vector search is **post-filtering**: run the ANN search first to get the top-k nearest neighbors by vector similarity alone, then discard any results that don't match the metadata filter. This is the simplest possible implementation — it requires no changes to the ANN index itself — and it is also frequently wrong in a way that's easy to miss during development and painful to discover in production.

The failure mode is straightforward once you see it: if you ask for the top 10 nearest neighbors and then filter by `tenant_id = 'acme_corp'`, and it happens that only 2 of those top 10 vectors belong to `acme_corp`, you get 2 results back instead of 10 — even if `acme_corp` actually has hundreds of vectors that are reasonably close to the query, just not close enough to have made the *unfiltered* top 10. The problem compounds as filter selectivity increases: a filter that only 1% of the corpus matches, applied after retrieving the top 10 unfiltered candidates, will very often return zero or one results even when perfectly good matches exist deeper in the ranking. This isn't a rare edge case in multi-tenant systems — it's the default behavior whenever tenants have wildly different corpus sizes, which is almost always.

```python
def naive_post_filter_search(query_vec, index, metadata_store, tenant_id, top_k=10):
    """The trap: filtering after the fact silently starves selective filters."""
    # Ask the ANN index for top_k globally, with no awareness of the filter
    candidates = index.search(query_vec, top_k=top_k)
    filtered = [c for c in candidates if metadata_store[c.id]["tenant_id"] == tenant_id]
    return filtered  # could be empty even if acme_corp has great matches available
```

A common band-aid is to over-fetch: ask the ANN index for, say, `top_k * 20` candidates, then filter down to `top_k`. This helps in mild cases but is fragile — it just moves the threshold at which selectivity breaks it, and it wastes compute on every query by retrieving and scoring far more candidates than needed, most of which get thrown away. It's a reasonable stopgap for filters you know are only mildly selective, not a real solution for the general problem.

**Pre-filtering** solves this properly by making the metadata constraint part of the search itself, so the ANN algorithm only ever considers candidates that already satisfy the filter. The naive way to implement this — literally restrict the candidate vector set to matching IDs before running any distance computation — degenerates back to a brute-force scan over the filtered subset, throwing away the whole benefit of the ANN index if the filtered subset is still large. The better approaches integrate filtering directly into the ANN traversal.

For graph-based indexes like HNSW, this means making the greedy graph walk **filter-aware**: during traversal, the algorithm still walks the full graph structure (since the graph's connectivity is what gives you the logarithmic search speed), but at each step it evaluates whether a candidate node satisfies the filter before adding it to the result set, while continuing to traverse through non-matching nodes so the search doesn't get stuck in a region with no matches. Qdrant is frequently cited for making this pattern efficient and correct by design. For partition-based indexes like IVF, one practical pattern is maintaining metadata-aware routing — for instance, in a heavily multi-tenant deployment, giving each tenant (or metadata partition) entirely separate clusters or even separate indexes, so the "filter" is really "which index do I search" rather than a runtime predicate at all — an approach that trades some flexibility for filtering that's essentially free.

```python
def filter_aware_greedy_search(query_vec, entry_node, layer, get_vector_fn,
                                  passes_filter_fn, max_steps=200):
    """Sketch of filter-aware HNSW traversal: keep walking through non-matching
    nodes (they're still useful for graph connectivity), but only ever
    *return* nodes that satisfy the filter."""
    visited = set()
    frontier = [entry_node]
    results = []

    for _ in range(max_steps):
        if not frontier:
            break
        current = frontier.pop(0)
        if current.id in visited:
            continue
        visited.add(current.id)

        if passes_filter_fn(current):
            dist = np.linalg.norm(query_vec - get_vector_fn(current))
            results.append((dist, current))

        # Continue exploring neighbors regardless of whether `current` matched --
        # this is the key difference from naive pre-filter-then-search.
        for neighbor in current.neighbors[layer]:
            if neighbor.id not in visited:
                frontier.append(neighbor)

    results.sort(key=lambda x: x[0])
    return results
```

The practical takeaway for a senior engineer evaluating or operating a vector database: always check, explicitly, whether the system's filtering is implemented as true pre-filtering integrated into the index traversal, or as post-filtering with some amount of over-fetching hidden behind the scenes. This is exactly the kind of detail vendor documentation sometimes glosses over, and it's worth testing directly — construct a query with a deliberately selective filter (matching well under 1% of your corpus) and verify you still get a full, high-quality result set back, rather than trusting marketing claims about "native metadata filtering."

## Structuring Metadata for Filter Performance

Not all metadata filters are equal in cost even within a system that does correct pre-filtering. Equality filters on low-cardinality fields (status, category, tenant_id) are cheap to index and evaluate — they behave much like a standard database index on a low-cardinality column, and most vector databases let you explicitly declare which metadata fields should be indexed for filtering (Qdrant's payload indexes, Weaviate's inverted index configuration on schema properties, Pinecone's automatic metadata indexing). Range filters (date ranges, numeric thresholds) and filters on high-cardinality fields (arbitrary user IDs, free-text tags) are more expensive and, in some systems, require different index structures under the hood (e.g., a B-tree-like structure for ranges versus a hash-like structure for equality).

A pattern worth knowing explicitly: if you know your dominant filter pattern in advance (e.g., "every query filters by `tenant_id` first, and everything else is secondary"), it is often far cheaper and simpler to physically partition data by that field — separate collections, separate index shards, or separate namespaces per tenant — than to rely on a general-purpose filtering engine to make that filter fast at scale. This is the same intuition behind physical table partitioning in relational databases, applied to vector indexes, and it's the basis for the multi-tenancy patterns discussed next.

## Multi-Tenancy and Namespace Isolation

Multi-tenant vector search has to satisfy two goals that pull in opposite directions: strict data isolation between tenants (a bug that lets tenant A's query return tenant B's vectors is a serious security incident, not just a quality bug), and reasonable resource efficiency (you generally can't afford one fully separate, fully provisioned index per tenant if you have thousands of tenants, many of them small).

There are three broad patterns in production use, and the right one depends heavily on tenant count and tenant size distribution. **Shared collection with a tenant_id metadata field** is the simplest: all tenants' vectors live in one collection, and every query includes a mandatory `tenant_id` filter, relying on the pre-filtering guarantees discussed above for both correctness and isolation. This scales well operationally (one index to manage, one capacity plan) but means every tenant's data shares the same underlying ANN graph or partition structure, which can create noisy-neighbor effects — a few extremely large tenants can dominate cluster assignments or graph density in ways that subtly affect smaller tenants' recall and latency.

**Native namespace/partition mechanisms** (Pinecone namespaces, Qdrant's payload-based sharding with tenant-aware placement, Weaviate's multi-tenancy mode) sit in the middle: logically and often physically the data is still colocated within the same broader deployment, but the database explicitly understands "this is a tenant boundary" and provides guarantees (or performance benefits) around it — for example, Weaviate's multi-tenancy mode gives each tenant its own dedicated shard, so a tenant's HNSW graph is entirely separate from every other tenant's, eliminating both the isolation risk and the noisy-neighbor problem, while still letting the operator manage all tenants through one deployment rather than provisioning infrastructure per tenant by hand.

**Fully separate collections or indexes per tenant** gives the strongest isolation and performance predictability (one tenant's data volume or query pattern genuinely cannot affect another's), and it's the natural choice for a smaller number of large, high-value tenants (think dozens to low hundreds, each with meaningfully large corpora) or for compliance regimes that require physically separated storage per customer. It becomes operationally unwieldy at thousands of tenants, particularly many small ones, because per-collection overhead (memory for index structures, connection/resource pooling, provisioning automation) doesn't amortize well when most tenants only have a few thousand vectors each.

```python
class TenantAwareVectorStore:
    """Illustrates the decision point: route to per-tenant collections for
    large tenants, and a shared filtered collection for the long tail of
    small tenants -- a common hybrid pattern in practice."""

    LARGE_TENANT_THRESHOLD = 1_000_000  # vectors

    def __init__(self, client):
        self.client = client
        self.large_tenant_ids = set()

    def get_collection_for(self, tenant_id: str) -> str:
        if tenant_id in self.large_tenant_ids:
            return f"tenant_{tenant_id}"
        return "shared_small_tenants"

    def search(self, tenant_id: str, query_vec, top_k=10):
        collection = self.get_collection_for(tenant_id)
        if collection == "shared_small_tenants":
            return self.client.search(
                collection_name=collection,
                query_vector=query_vec,
                query_filter={"must": [{"key": "tenant_id", "match": {"value": tenant_id}}]},
                limit=top_k,
            )
        return self.client.search(collection_name=collection, query_vector=query_vec, limit=top_k)
```

This hybrid pattern — dedicated resources for the handful of tenants large enough to need them, shared filtered infrastructure for everyone else, with a defined migration path as a tenant crosses the size threshold — is common in real multi-tenant SaaS vector search deployments precisely because it avoids the worst of both extremes.

## Building and Updating Indexes at Scale

Initial index construction for a large corpus is usually a batch process: embed everything, then build the ANN structure (train IVF centroids, or insert every vector into the HNSW graph) in one pass, often on a schedule (nightly, or triggered by a data pipeline) rather than live. This batch-build approach is straightforward to reason about and lets you use the full dataset to make good clustering or graph-quality decisions, but it means there's a window where the "live" index is stale relative to the newest data, and swapping in a freshly built index (blue/green style) needs careful coordination so queries don't hit a half-built index or an inconsistent view mid-swap.

Incremental updates — adding a small number of new vectors to an already-built, already-serving index — behave very differently depending on index type, and this difference is one of the most consequential and least obvious factors in choosing between HNSW-family and IVF-family systems for a workload with continuous writes. HNSW is naturally incremental: inserting a node means running the same construction-time greedy neighbor-search-and-connect process for just that one new vector, without touching the rest of the graph's structure. This is why HNSW-based systems (Qdrant, Weaviate, Pinecone's underlying implementation) handle "insert a document the moment it's created" workloads gracefully. IVF is much less naturally incremental: new vectors can be assigned to their nearest *existing* centroid cheaply enough, but the centroids themselves were computed from whatever data existed at training time — if the data distribution drifts meaningfully (new topics, new document types, seasonal shifts in a product catalog), the original partitioning becomes a progressively worse fit, and periodically re-running the clustering step (re-training) becomes necessary, which is a heavier, batch-style operation.

## Deletes and Updates: Trickier Than It Sounds

Deleting a row from a relational table is conceptually simple: the row is gone, the B-tree index is rebalanced, done. Deleting a vector from an ANN index is genuinely harder, and this is one of the more interview-worthy nuances of vector database internals, because the naive intuition ("just remove it") doesn't map cleanly onto how these structures work.

In an HNSW graph, a node isn't just a data point — it's also part of the *navigation structure* other nodes rely on to reach their own neighbors. If you naively delete a node and all its edges, you can partition the graph or strand other nodes whose only path to certain regions went through the deleted node, silently degrading recall for queries that have nothing to do with the deleted content. Because of this, most production HNSW implementations don't perform true, immediate physical deletion. Instead they do **soft/tombstone deletion**: the vector is marked as deleted in a side table or bitmap, excluded from search results, but its graph edges and node structure are left in place temporarily so the graph's connectivity remains intact for other nodes' searches. Periodically, a background compaction process rebuilds the affected portion of the graph (or the whole graph, for smaller indexes) to actually reclaim the space and repair connectivity properly. This is directly analogous to how LSM-tree-based storage engines (RocksDB, Cassandra) handle deletes with tombstones and compaction, and it's not a coincidence — both are dealing with the same underlying tension between fast, non-disruptive point operations and eventually needing to reclaim space and repair structure.

```python
class TombstoneAwareIndex:
    """Illustrates soft-delete semantics for an HNSW-style index."""

    def __init__(self, hnsw_index):
        self.index = hnsw_index
        self.tombstones = set()

    def delete(self, vector_id):
        # Mark deleted; do NOT touch graph edges yet -- other nodes' paths
        # may still route through this node's position in the graph.
        self.tombstones.add(vector_id)

    def search(self, query_vec, top_k=10, overfetch_factor=3):
        # Overfetch because some raw graph results will be tombstoned
        raw_results = self.index.search(query_vec, top_k=top_k * overfetch_factor)
        live_results = [r for r in raw_results if r.id not in self.tombstones]
        return live_results[:top_k]

    def compact(self):
        """Periodic maintenance: physically rebuild, dropping tombstoned nodes
        and repairing any connectivity that relied on them."""
        surviving_ids = [vid for vid in self.index.all_ids() if vid not in self.tombstones]
        self.index = self.index.rebuild_from(surviving_ids)
        self.tombstones.clear()
```

Updates (re-embedding a document because its content changed, or re-embedding an entire corpus because you switched embedding models) are, under the hood, almost always implemented as a delete-plus-insert rather than an in-place mutation, precisely because there's no meaningful way to "adjust" a vector's position within an HNSW graph or an IVF cluster assignment in place — its neighbors and edges were chosen based on its original position, and a changed vector may belong in an entirely different part of the graph or a different cluster. This has a very concrete operational consequence worth internalizing: a bulk re-embedding operation (say, after adopting a better embedding model) is not a cheap metadata update — it is functionally a full index rebuild, with the same batch-processing considerations, cost, and downtime/blue-green concerns as the original bulk load. Teams sometimes underestimate this and are surprised when "just switching embedding models" turns into a multi-hour or multi-day reprocessing job with a live cutover plan, rather than a quick config change.

A related, easy-to-miss operational issue is **tombstone accumulation** under high delete/update churn: if compaction doesn't run frequently enough relative to your delete rate, you can end up in a state where a large fraction of the "live" graph is actually tombstoned, forcing every query to overfetch and filter heavily, quietly degrading both latency and effective recall well before anyone notices via top-line metrics like index size or vector count. Monitoring the tombstone ratio (deleted-but-not-yet-compacted vectors as a fraction of total) is a metric worth explicitly tracking and alerting on in any production deployment with meaningful update or delete volume, alongside the more obvious metrics like query latency and recall on a held-out validation set.
