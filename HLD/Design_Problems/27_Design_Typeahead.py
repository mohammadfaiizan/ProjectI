"""
Problem 27: Design a Typeahead / Autocomplete System
======================================================
Working simulation of a typeahead system with:
- TrieNode with min-heap top-K tracking
- Trie class with insert, search, and frequency update
- TopKTracker using min-heap of size K
- FrequencyUpdater to propagate frequency changes up the trie
- TypeaheadService with get_suggestions, add_query_log, build_from_query_log
- Demo showing trie construction from query logs and prefix search
"""

import heapq
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional
import time
import re


# ─── Top-K Tracker ────────────────────────────────────────────────────────────

class TopKTracker:
    """
    Maintain top-K elements by score using a min-heap of size K.
    Min-heap ensures O(log K) insertion; evicts minimum when full.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self._heap: list[tuple[int, str]] = []   # (frequency, query)
        self._set: set[str] = set()              # fast membership check

    def add(self, query: str, frequency: int) -> bool:
        """Add/update a query. Returns True if top-K changed."""
        if query in self._set:
            # Remove existing entry (lazy deletion via rebuild)
            self._heap = [(f, q) for f, q in self._heap if q != query]
            heapq.heapify(self._heap)
            self._set.discard(query)

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (frequency, query))
            self._set.add(query)
            return True
        elif frequency > self._heap[0][0]:  # Better than current minimum
            evicted = heapq.heapreplace(self._heap, (frequency, query))
            self._set.discard(evicted[1])
            self._set.add(query)
            return True
        return False

    def get_top_k(self, descending: bool = True) -> list[tuple[str, int]]:
        """Return list of (query, frequency) sorted by frequency."""
        result = [(q, f) for f, q in self._heap]
        return sorted(result, key=lambda x: x[1], reverse=descending)

    def min_frequency(self) -> int:
        return self._heap[0][0] if self._heap else 0

    def __len__(self) -> int:
        return len(self._heap)


# ─── Trie Node ────────────────────────────────────────────────────────────────

class TrieNode:
    """
    Single node in the trie.
    Each node maintains a top-K tracker for all completions passing through it.
    """

    def __init__(self, k: int = 10):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.frequency: int = 0
        self.top_k: TopKTracker = TopKTracker(k)

    def __repr__(self) -> str:
        return f"TrieNode(is_end={self.is_end}, freq={self.frequency}, top_k_size={len(self.top_k)})"


# ─── Trie ─────────────────────────────────────────────────────────────────────

class Trie:
    """
    Trie with per-node top-K tracking.
    Each node stores the K most frequent complete queries that pass through it.
    This enables O(L) prefix lookup returning top-K suggestions without full subtree scan.
    """

    def __init__(self, k: int = 10):
        self.root = TrieNode(k)
        self.k = k
        self._word_freq: dict[str, int] = {}   # store all inserted words' frequencies

    def insert(self, word: str, frequency: int) -> None:
        """Insert a word with its frequency; propagate to all ancestor nodes."""
        word = word.lower().strip()
        if not word:
            return

        self._word_freq[word] = frequency
        node = self.root
        # Update root top-K
        node.top_k.add(word, frequency)

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(self.k)
            node = node.children[char]
            node.top_k.add(word, frequency)

        node.is_end = True
        node.frequency = frequency

    def update_frequency(self, word: str, new_frequency: int) -> None:
        """Update frequency of an existing word; propagate through ancestors."""
        word = word.lower().strip()
        if word not in self._word_freq:
            self.insert(word, new_frequency)
            return

        self._word_freq[word] = new_frequency
        # Re-propagate: update all nodes on the path
        node = self.root
        node.top_k.add(word, new_frequency)
        for char in word:
            if char not in node.children:
                break
            node = node.children[char]
            node.top_k.add(word, new_frequency)
        node.frequency = new_frequency

    def search(self, prefix: str, k: Optional[int] = None) -> list[tuple[str, int]]:
        """
        Return top-K suggestions for a given prefix.
        O(L) traversal to prefix node, then O(K log K) to sort.
        """
        k = k or self.k
        prefix = prefix.lower().strip()
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        return node.top_k.get_top_k(descending=True)[:k]

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word starts with the given prefix."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def get_all_words(self, prefix: str = "") -> list[tuple[str, int]]:
        """Retrieve all (word, frequency) pairs with the given prefix (for debugging)."""
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]
        results = []
        self._dfs(node, prefix, results)
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _dfs(self, node: TrieNode, current: str, results: list) -> None:
        if node.is_end:
            results.append((current, node.frequency))
        for char, child in node.children.items():
            self._dfs(child, current + char, results)


# ─── Frequency Updater ────────────────────────────────────────────────────────

class FrequencyUpdater:
    """
    Manages incremental frequency updates from query logs.
    Batches updates and applies them to the trie periodically.
    """

    def __init__(self, trie: Trie):
        self.trie = trie
        self._pending: dict[str, int] = defaultdict(int)  # query → delta count
        self._flush_count = 0

    def record_query(self, query: str) -> None:
        """Record a single query occurrence (batched, not immediately applied)."""
        query = query.lower().strip()
        if query:
            self._pending[query] += 1

    def flush(self) -> int:
        """Apply all pending frequency updates to the trie. Returns number of updates."""
        if not self._pending:
            return 0

        count = 0
        for query, delta in self._pending.items():
            existing = self.trie._word_freq.get(query, 0)
            self.trie.update_frequency(query, existing + delta)
            count += 1

        self._pending.clear()
        self._flush_count += 1
        return count

    def get_pending_count(self) -> int:
        return len(self._pending)


# ─── Typeahead Service ────────────────────────────────────────────────────────

class TypeaheadService:
    """
    High-level typeahead service combining:
    - Trie for prefix lookup
    - Frequency updater for real-time trending
    - Simple result cache for common prefixes
    - Basic spell correction (edit distance 1)
    """

    def __init__(self, k: int = 10, cache_size: int = 1000):
        self.k = k
        self.trie = Trie(k)
        self.frequency_updater = FrequencyUpdater(self.trie)
        self._cache: dict[str, list] = {}       # prefix → suggestions (LRU approximation)
        self._cache_keys: list[str] = []        # insertion order for eviction
        self.cache_size = cache_size
        self._query_log: list[tuple[float, str]] = []  # (timestamp, query)

    def build_from_query_log(self, query_counts: dict[str, int]) -> None:
        """Build trie from precomputed query → count mapping."""
        print(f"Building trie from {len(query_counts)} unique queries...")
        start = time.time()
        for query, count in sorted(query_counts.items(), key=lambda x: x[1], reverse=True):
            self.trie.insert(query, count)
        elapsed = time.time() - start
        print(f"Trie built in {elapsed:.4f}s | "
              f"Total unique queries: {len(self.trie._word_freq)}")

    def get_suggestions(self, prefix: str, user_id: Optional[str] = None) -> list[dict]:
        """
        Get top-K suggestions for a prefix.
        Checks cache first; falls back to trie.
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []

        # Cache hit
        cache_key = f"{prefix}:{user_id or 'global'}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Trie lookup
        results = self.trie.search(prefix, self.k)

        # If no results, try spell correction
        if not results and len(prefix) > 2:
            corrected = self._spell_correct(prefix)
            if corrected and corrected != prefix:
                results = self.trie.search(corrected, self.k)
                if results:
                    print(f"  [Spell corrected] '{prefix}' → '{corrected}'")

        suggestions = [
            {"query": query, "frequency": freq, "score": round(freq / max(1, self._max_freq()), 4)}
            for query, freq in results
        ]

        # Cache result
        self._cache_put(cache_key, suggestions)
        return suggestions

    def add_query_log(self, query: str, auto_flush_threshold: int = 100) -> None:
        """Log a user query; auto-flush to trie when threshold reached."""
        self._query_log.append((time.time(), query))
        self.frequency_updater.record_query(query)
        # Invalidate cache for affected prefixes
        for i in range(1, len(query) + 1):
            prefix = query[:i].lower()
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]

        if self.frequency_updater.get_pending_count() >= auto_flush_threshold:
            n = self.frequency_updater.flush()
            print(f"  [FrequencyUpdater] Auto-flushed {n} query frequency updates to trie")

    def force_flush(self) -> int:
        """Manually flush pending frequency updates."""
        return self.frequency_updater.flush()

    def get_trending(self, window_seconds: float = 3600.0, top_n: int = 10) -> list[tuple[str, int]]:
        """Return top-N queries from the last window_seconds."""
        cutoff = time.time() - window_seconds
        recent = [q for ts, q in self._query_log if ts >= cutoff]
        return Counter(recent).most_common(top_n)

    def _spell_correct(self, word: str) -> Optional[str]:
        """Simple edit-distance-1 corrections: delete, transpose, replace, insert."""
        alphabet = 'abcdefghijklmnopqrstuvwxyz '
        candidates = set()

        # Deletes
        for i in range(len(word)):
            candidates.add(word[:i] + word[i+1:])
        # Transposes
        for i in range(len(word) - 1):
            candidates.add(word[:i] + word[i+1] + word[i] + word[i+2:])
        # Replaces
        for i in range(len(word)):
            for c in alphabet:
                candidates.add(word[:i] + c + word[i+1:])
        # Inserts
        for i in range(len(word) + 1):
            for c in alphabet:
                candidates.add(word[:i] + c + word[i:])

        # Return first candidate that exists in trie
        for candidate in sorted(candidates):
            if self.trie.starts_with(candidate) and candidate != word:
                return candidate
        return None

    def _max_freq(self) -> int:
        if not self.trie._word_freq:
            return 1
        return max(self.trie._word_freq.values())

    def _cache_put(self, key: str, value: list) -> None:
        if key not in self._cache:
            if len(self._cache_keys) >= self.cache_size:
                oldest = self._cache_keys.pop(0)
                self._cache.pop(oldest, None)
            self._cache_keys.append(key)
        self._cache[key] = value

    def print_stats(self) -> None:
        print(f"\nTypeahead Service Stats:")
        print(f"  Unique queries in trie : {len(self.trie._word_freq)}")
        print(f"  Cache entries          : {len(self._cache)}")
        print(f"  Query log entries      : {len(self._query_log)}")
        print(f"  Pending freq updates   : {self.frequency_updater.get_pending_count()}")


# ─── Demo / Simulation ────────────────────────────────────────────────────────

SAMPLE_QUERY_LOG = [
    # Format: (query, count) — simulating aggregated search data
    ("how to cook pasta", 15_000_000),
    ("how to lose weight fast", 12_000_000),
    ("how to tie a tie", 9_500_000),
    ("how to make money online", 8_000_000),
    ("how to get a passport", 6_500_000),
    ("how to write a resume", 5_000_000),
    ("how to invest in stocks", 4_200_000),
    ("how to learn python", 3_800_000),
    ("how to build a website", 3_500_000),
    ("how to start a business", 3_200_000),
    ("python tutorial for beginners", 11_000_000),
    ("python list comprehension", 7_500_000),
    ("python string formatting", 6_000_000),
    ("python dictionary methods", 4_500_000),
    ("python virtual environment", 3_000_000),
    ("weather today", 50_000_000),
    ("weather forecast", 45_000_000),
    ("weather in new york", 20_000_000),
    ("weather app", 18_000_000),
    ("weather tomorrow", 15_000_000),
    ("amazon prime", 30_000_000),
    ("amazon delivery", 25_000_000),
    ("amazon shopping", 22_000_000),
    ("best restaurants near me", 28_000_000),
    ("best movies 2024", 18_000_000),
    ("best laptops", 12_000_000),
    ("best pizza near me", 10_000_000),
    ("youtube music", 40_000_000),
    ("youtube videos", 35_000_000),
    ("youtube kids", 20_000_000),
]

TRENDING_QUERIES = [
    "python ai libraries",
    "python machine learning",
    "python ai libraries",
    "how to use chatgpt",
    "how to use chatgpt",
    "how to use chatgpt",
    "python ai libraries",
    "how to use chatgpt",
    "ai tools for coding",
    "ai tools for coding",
    "python ai libraries",
]


def run_simulation():
    print("=" * 60)
    print("TYPEAHEAD / AUTOCOMPLETE SYSTEM SIMULATION")
    print("=" * 60)

    # ── Build service ──────────────────────────────────────────
    service = TypeaheadService(k=5)
    query_counts = {q: c for q, c in SAMPLE_QUERY_LOG}
    service.build_from_query_log(query_counts)

    # ── Prefix searches ───────────────────────────────────────
    test_prefixes = ["how", "how to", "pyt", "python", "weath", "yo", "best", "am"]

    print("\n--- Prefix Search Results ---")
    for prefix in test_prefixes:
        suggestions = service.get_suggestions(prefix)
        print(f"\n  Prefix: '{prefix}'")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s['query']:<40} freq={s['frequency']:>12,}  score={s['score']:.4f}")

    # ── Cache verification ─────────────────────────────────────
    print("\n--- Cache Behavior (second call should be instant) ---")
    t0 = time.perf_counter()
    service.get_suggestions("how to")
    t1 = time.perf_counter()
    service.get_suggestions("how to")  # cached
    t2 = time.perf_counter()
    print(f"  First call:  {(t1-t0)*1000:.3f}ms")
    print(f"  Second call: {(t2-t1)*1000:.3f}ms (from cache)")

    # ── Trending queries ──────────────────────────────────────
    print("\n--- Simulating Trending Query Log ---")
    for q in TRENDING_QUERIES:
        service.add_query_log(q)
    service.force_flush()

    print("\n  Top trending queries (last hour):")
    for rank, (query, count) in enumerate(service.get_trending(top_n=5), 1):
        print(f"    {rank}. '{query}' — {count} recent searches")

    print("\n  Updated suggestions for 'python ai':")
    for s in service.get_suggestions("python ai"):
        print(f"    {s['query']:<40} freq={s['frequency']:>10,}")

    print("\n  Updated suggestions for 'how to use':")
    for s in service.get_suggestions("how to use"):
        print(f"    {s['query']:<40} freq={s['frequency']:>10,}")

    # ── Spell correction ──────────────────────────────────────
    print("\n--- Spell Correction Demo ---")
    typos = ["hwo to", "pytohn", "weathr"]
    for typo in typos:
        suggestions = service.get_suggestions(typo)
        if suggestions:
            print(f"  '{typo}' → top suggestion: '{suggestions[0]['query']}'")
        else:
            print(f"  '{typo}' → no suggestions found")

    # ── TopKTracker standalone demo ───────────────────────────
    print("\n--- TopKTracker Standalone Demo (K=3) ---")
    tracker = TopKTracker(k=3)
    items = [("apple", 100), ("banana", 200), ("cherry", 150), ("date", 250), ("elderberry", 180)]
    for item, freq in items:
        changed = tracker.add(item, freq)
        print(f"  Added '{item}' (freq={freq}) — top-K changed: {changed} | current min: {tracker.min_frequency()}")
    print(f"  Final top-3: {tracker.get_top_k()}")

    # ── Trie word enumeration demo ────────────────────────────
    print("\n--- All Words with Prefix 'youtube' ---")
    for word, freq in service.trie.get_all_words("youtube"):
        print(f"  {word:<40} {freq:>12,}")

    # ── Stats ─────────────────────────────────────────────────
    service.print_stats()

    print("\n" + "=" * 60)
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
