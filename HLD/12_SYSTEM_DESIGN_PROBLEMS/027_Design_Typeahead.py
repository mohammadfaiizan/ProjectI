"""
TYPEAHEAD / AUTOCOMPLETE SERVICE
==================================

FUNCTIONAL REQUIREMENTS:
- Return top-k suggestions as user types each character
- Suggestions ranked by frequency (search popularity)
- New searches update the suggestion corpus
- Filter suggestions by locale/language
- Personalised suggestions based on user history

NON-FUNCTIONAL REQUIREMENTS:
- 100 M DAU, 10 search requests/user/day = 1 B requests/day
- Response time: < 10 ms p99 (suggestions feel instant at ≤ 50 ms)
- 5 M unique query terms, avg 30 characters
- Write (new search) frequency: 1% of reads = 10 M writes/day
- Suggestions stale by up to 1 hour is acceptable

ARCHITECTURE:
  Client ──HTTPS──▶ API GW ──▶ Typeahead Service ──▶ Cache (Redis)
                                     │                    │ miss
                                     │              Trie Service
                                     │                    │
                              Aggregation Job ──▶ Trie Store (DB)
                              (Hadoop/Spark)

KEY DESIGN DECISIONS:
1. DATA STRUCTURE — Trie (prefix tree) is the classic approach.
   Each node stores: character, children, top-k suggestions (cache at node).
   Alternative: Redis Sorted Sets (ZADD prefix:char score word).

2. TRIE VS REDIS SORTED SETS:
   Trie:
   - Pros: efficient prefix traversal, natural structure
   - Cons: requires distributed/sharded trie; hard to update in place
   Redis Sorted Sets:
   - Pros: simple distribution, atomic INCR, O(log n) range query
   - Cons: more memory (all prefixes stored), less natural
   Production: Yelp, Google use a distributed trie.
   Twitter uses Redis ZADD.

3. TRIE UPDATE STRATEGY:
   Don't update trie on every search (would invalidate caches on hot path).
   Instead: batch updates via Hadoop Map-Reduce job (hourly/daily).
   Job counts query frequency from logs → rebuilds trie with new frequencies.

4. TRIE SHARDING:
   Partition by first character: 26 trie servers (a-z).
   Or by first 2 chars: 676 partitions.
   Each partition fits in memory (5M queries × 100 bytes = 500 MB total).

5. CACHING:
   Store top-k at each trie node → O(1) retrieval for exact prefix hit.
   LRU cache of prefix → [suggestions] in front of trie.

6. PERSONALISATION:
   Blend global frequency (70%) with user history (30%).
   User history stored in Redis with TTL (30 days).

7. TRENDING QUERIES:
   Real-time window (last 1h): Redis sorted set with sliding window.
   Inject trending terms into suggestions if they match prefix.
"""

from __future__ import annotations
import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import math
import threading


# ---------------------------------------------------------------------------
# Trie Node
# ---------------------------------------------------------------------------

@dataclass
class TrieNode:
    char: str = ""
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False
    frequency: int = 0
    # Cache top-k suggestions at this node (updated on each batch rebuild)
    cached_suggestions: List[Tuple[int, str]] = field(default_factory=list)

    def update_cache(self, k: int = 10) -> None:
        """Collect top-k queries rooted at this node (DFS)."""
        candidates = []
        self._collect(candidates)
        self.cached_suggestions = heapq.nlargest(k, candidates, key=lambda x: x[0])

    def _collect(self, result: List[Tuple[int, str]], prefix: str = "") -> None:
        if self.is_end:
            result.append((self.frequency, prefix))
        for char, child in self.children.items():
            child._collect(result, prefix + char)


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

class Trie:
    """Prefix tree with top-k caching at each node."""

    def __init__(self, k: int = 10):
        self._root = TrieNode()
        self.k = k

    def insert(self, query: str, frequency: int) -> None:
        node = self._root
        for char in query.lower():
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end = True
        node.frequency = frequency

    def update_all_caches(self) -> None:
        """Rebuild top-k cache at every node (after batch insert)."""
        self._update_subtree(self._root, "")

    def _update_subtree(self, node: TrieNode, prefix: str) -> List[Tuple[int, str]]:
        """Return list of (frequency, query) for this subtree."""
        candidates = []
        if node.is_end:
            candidates.append((node.frequency, prefix))
        for char, child in node.children.items():
            candidates.extend(self._update_subtree(child, prefix + char))
        node.cached_suggestions = heapq.nlargest(self.k, candidates, key=lambda x: x[0])
        return candidates

    def search(self, prefix: str) -> List[Tuple[int, str]]:
        """Return top-k suggestions for prefix. O(len(prefix)) with caching."""
        node = self._root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        return node.cached_suggestions

    def increment(self, query: str, delta: int = 1) -> None:
        """Increment frequency (for real-time updates). Does NOT update caches."""
        node = self._root
        for char in query.lower():
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end = True
        node.frequency += delta

    def size(self) -> int:
        """Count number of complete queries stored."""
        return self._count_nodes(self._root)

    def _count_nodes(self, node: TrieNode) -> int:
        count = 1 if node.is_end else 0
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


# ---------------------------------------------------------------------------
# Redis Sorted Set approach (alternative to Trie)
# ---------------------------------------------------------------------------

class RedisSortedSetTypeahead:
    """
    Alternative to Trie using sorted sets per prefix.
    Key: prefix → {query: score}
    More memory but simpler to scale horizontally.
    """

    def __init__(self):
        # Simulated sorted sets: prefix_key → {query: score}
        self._sets: Dict[str, Dict[str, float]] = defaultdict(dict)

    def index_query(self, query: str, score: float) -> None:
        """Index all prefixes of query."""
        q = query.lower()
        for length in range(1, len(q) + 1):
            prefix = q[:length]
            self._sets[prefix][q] = score

    def suggest(self, prefix: str, k: int = 10) -> List[Tuple[float, str]]:
        """Return top-k suggestions sorted by score DESC."""
        candidates = self._sets.get(prefix.lower(), {})
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k]

    def increment(self, query: str, delta: float = 1.0) -> None:
        """Increment score for a query across all its prefixes."""
        q = query.lower()
        for length in range(1, len(q) + 1):
            prefix = q[:length]
            self._sets[prefix][q] = self._sets[prefix].get(q, 0) + delta

    def memory_estimate_bytes(self) -> int:
        total = 0
        for prefix, scores in self._sets.items():
            for query, score in scores.items():
                total += len(prefix) + len(query) + 16  # rough estimate
        return total


# ---------------------------------------------------------------------------
# Trending Queries (real-time window)
# ---------------------------------------------------------------------------

class TrendingService:
    """
    Counts queries in a sliding time window.
    Uses Redis sorted sets with time-bucketed scoring.
    """

    def __init__(self, window_seconds: int = 3600):
        self._window = window_seconds
        self._events: List[Tuple[float, str]] = []   # (timestamp, query)
        self._lock = threading.Lock()

    def record(self, query: str) -> None:
        with self._lock:
            self._events.append((time.time(), query.lower()))
            # Trim old events
            cutoff = time.time() - self._window
            self._events = [(ts, q) for ts, q in self._events if ts > cutoff]

    def trending(self, k: int = 10) -> List[Tuple[int, str]]:
        """Return top-k trending queries in current window."""
        with self._lock:
            cutoff = time.time() - self._window
            counts: Dict[str, int] = defaultdict(int)
            for ts, q in self._events:
                if ts > cutoff:
                    counts[q] += 1
            return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]

    def trending_for_prefix(self, prefix: str, k: int = 5) -> List[str]:
        """Filter trending queries that start with prefix."""
        return [q for _, q in self.trending(50) if q.startswith(prefix.lower())][:k]


# ---------------------------------------------------------------------------
# Personalisation
# ---------------------------------------------------------------------------

class PersonalisationService:
    """Tracks user query history to boost personal suggestions."""

    def __init__(self, history_size: int = 50, ttl_days: int = 30):
        # user_id → [(timestamp, query)]
        self._history: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self._history_size = history_size
        self._ttl = ttl_days * 86400

    def record(self, user_id: str, query: str) -> None:
        history = self._history[user_id]
        history.append((time.time(), query.lower()))
        # Trim TTL
        cutoff = time.time() - self._ttl
        history[:] = [(ts, q) for ts, q in history if ts > cutoff]
        # Keep only recent N
        if len(history) > self._history_size:
            history[:] = history[-self._history_size:]

    def personal_scores(self, user_id: str) -> Dict[str, float]:
        """Score queries by recency and frequency in user history."""
        scores: Dict[str, float] = defaultdict(float)
        now = time.time()
        for ts, query in self._history.get(user_id, []):
            age_days = (now - ts) / 86400
            recency_weight = math.exp(-age_days / 7)  # 7-day half-life
            scores[query] += recency_weight
        return dict(scores)

    def blend(self, global_suggestions: List[Tuple[int, str]],
              user_scores: Dict[str, float],
              global_weight: float = 0.7) -> List[str]:
        """Blend global frequency with personal scores."""
        if not global_suggestions:
            return []

        max_global = global_suggestions[0][0] if global_suggestions else 1
        max_personal = max(user_scores.values(), default=1)

        combined: Dict[str, float] = {}
        for freq, query in global_suggestions:
            norm_global = freq / max_global
            norm_personal = user_scores.get(query, 0) / max_personal
            combined[query] = (global_weight * norm_global +
                                (1 - global_weight) * norm_personal)

        # Add personal queries not in global list
        for query, score in user_scores.items():
            if query not in combined:
                combined[query] = (1 - global_weight) * score / max_personal

        return [q for q, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)]


# ---------------------------------------------------------------------------
# Typeahead Service (orchestrator)
# ---------------------------------------------------------------------------

class TypeaheadService:
    def __init__(self, k: int = 10):
        self._trie = Trie(k=k)
        self._trending = TrendingService(window_seconds=3600)
        self._personalise = PersonalisationService()
        self._k = k

    def build_from_corpus(self, query_freq: Dict[str, int]) -> None:
        """Batch-build trie from aggregated query frequencies."""
        for query, freq in query_freq.items():
            self._trie.insert(query, freq)
        self._trie.update_all_caches()

    def on_search(self, query: str, user_id: Optional[str] = None) -> None:
        """Record a completed search."""
        self._trie.increment(query)
        self._trending.record(query)
        if user_id:
            self._personalise.record(user_id, query)

    def suggest(self, prefix: str, user_id: Optional[str] = None,
                include_trending: bool = True) -> List[str]:
        """Return up to k suggestions for prefix."""
        # 1. Global trie suggestions
        global_sug = self._trie.search(prefix)

        # 2. Trending overlay
        trending_sug = []
        if include_trending:
            trending_sug = self._trending.trending_for_prefix(prefix, k=3)

        # 3. Personalisation
        if user_id:
            user_scores = self._personalise.personal_scores(user_id)
            # Filter personal scores to matching prefix
            user_scores = {q: s for q, s in user_scores.items()
                           if q.startswith(prefix.lower())}
            blended = self._personalise.blend(global_sug, user_scores)
        else:
            blended = [q for _, q in global_sug]

        # 4. Merge: trending first (inject up to 2), then blended
        seen = set()
        results = []
        for q in trending_sug:
            if q not in seen:
                seen.add(q)
                results.append(q)
        for q in blended:
            if q not in seen:
                seen.add(q)
                results.append(q)
                if len(results) >= self._k:
                    break

        return results[:self._k]


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_trie_basic():
    print("\n=== 1. Basic Trie Operations ===")
    trie = Trie(k=5)

    # Build from corpus
    corpus = {
        "apple": 1000,
        "apple pie": 800,
        "apple watch": 600,
        "application": 400,
        "apply": 300,
        "app store": 200,
        "amazon": 900,
        "android": 500,
        "animation": 150,
    }
    for query, freq in corpus.items():
        trie.insert(query, freq)
    trie.update_all_caches()

    print(f"Trie size: {trie.size()} queries")

    for prefix in ["a", "ap", "app", "appl", "apple"]:
        suggestions = trie.search(prefix)
        sug_text = [f"'{q}' ({f})" for f, q in suggestions[:5]]
        print(f"  Prefix '{prefix}': {sug_text}")


def demonstrate_2_redis_sorted_set():
    print("\n=== 2. Redis Sorted Set Alternative ===")
    rss = RedisSortedSetTypeahead()

    queries = [
        ("weather today", 5000),
        ("weather forecast", 3000),
        ("weather nyc", 2500),
        ("web development", 4000),
        ("web design", 3500),
        ("web scraping", 1000),
    ]
    for q, score in queries:
        rss.index_query(q, float(score))

    for prefix in ["we", "wea", "weather"]:
        results = rss.suggest(prefix, k=5)
        print(f"  Suggest '{prefix}': {[q for _, q in results]}")

    mem_kb = rss.memory_estimate_bytes() / 1024
    print(f"\nEstimated memory: {mem_kb:.1f} KB for {len(queries)} queries")


def demonstrate_3_real_time_trending():
    print("\n=== 3. Real-time Trending Queries ===")
    trending_svc = TrendingService(window_seconds=3600)

    # Simulate search events
    events = [
        ("world cup", 150),
        ("iphone 16", 80),
        ("python tutorial", 60),
        ("world cup score", 120),
        ("iphone 16 price", 70),
        ("hurricane", 200),
        ("hurricane path", 180),
    ]
    for query, count in events:
        for _ in range(count):
            trending_svc.record(query)

    top_trending = trending_svc.trending(k=5)
    print(f"Top trending queries:")
    for query, count in top_trending:
        print(f"  '{query}': {count} searches")

    prefix_trending = trending_svc.trending_for_prefix("hurr")
    print(f"\nTrending for prefix 'hurr': {prefix_trending}")


def demonstrate_4_personalisation():
    print("\n=== 4. Personalised Suggestions ===")
    trie = Trie(k=10)
    corpus = {
        "python": 10000, "pytorch": 5000, "python tutorial": 8000,
        "pandas": 7000, "programming": 6000, "product management": 3000,
        "postgresql": 4000,
    }
    for q, f in corpus.items():
        trie.insert(q, f)
    trie.update_all_caches()

    personal = PersonalisationService()
    # Alice is a data scientist — searches ML stuff often
    alice_searches = ["pytorch", "pytorch documentation", "pandas", "pandas dataframe",
                      "pytorch tutorial", "pandas groupby"]
    for q in alice_searches:
        personal.record("alice", q)

    # Bob is a software engineer
    bob_searches = ["postgresql", "postgresql index", "python tutorial"]
    for q in bob_searches:
        personal.record("bob", q)

    prefix = "py"
    global_sug = trie.search(prefix)
    print(f"Global suggestions for 'py': {[q for _, q in global_sug[:5]]}")

    alice_scores = {q: s for q, s in personal.personal_scores("alice").items()
                    if q.startswith(prefix)}
    alice_blended = personal.blend(global_sug, alice_scores, global_weight=0.6)
    print(f"\nAlice's personalised 'py' suggestions: {alice_blended[:5]}")

    bob_scores = {q: s for q, s in personal.personal_scores("bob").items()
                  if q.startswith(prefix)}
    bob_blended = personal.blend(global_sug, bob_scores, global_weight=0.6)
    print(f"Bob's personalised 'py' suggestions: {bob_blended[:5]}")


def demonstrate_5_full_service():
    print("\n=== 5. Full Typeahead Service ===")
    svc = TypeaheadService(k=5)

    # Build corpus
    svc.build_from_corpus({
        "facebook": 9000, "facebook login": 8000, "facebook marketplace": 6000,
        "fast food near me": 7000, "flights": 8500, "flights to london": 5000,
        "funny videos": 4000, "free movies": 4500,
    })

    # Simulate user searching
    for q in ["facebook", "fast food near me", "flights"]:
        svc.on_search(q, user_id="user_bob")

    # Bob types "fa" — should see personalised boost for facebook & fast food
    print(f"Bob types 'fa':")
    suggestions = svc.suggest("fa", user_id="user_bob")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s}")

    # Anonymous user
    print(f"\nAnonymous user types 'fa':")
    anon_suggestions = svc.suggest("fa")
    for i, s in enumerate(anon_suggestions, 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    demonstrate_1_trie_basic()
    demonstrate_2_redis_sorted_set()
    demonstrate_3_real_time_trending()
    demonstrate_4_personalisation()
    demonstrate_5_full_service()
