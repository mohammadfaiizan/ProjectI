"""
DISTRIBUTED WEB CRAWLER
========================

FUNCTIONAL REQUIREMENTS:
- Discover and download web pages starting from seed URLs
- Extract links and add newly discovered URLs to the frontier
- Respect robots.txt and crawl-delay directives
- Avoid revisiting the same URL within a crawl cycle
- Store downloaded page content for indexing

NON-FUNCTIONAL REQUIREMENTS:
- Crawl 1 B pages in 30 days = 385 pages/second
- URL frontier: 10 B URLs (prioritised queue)
- Storage: 500 KB average page → 500 TB total raw HTML
- Politeness: max 1 req/domain/second (default robots.txt)
- Distributed: 100 crawler workers

ARCHITECTURE:
  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐
  │ Seed URLs   │──▶ │ URL Frontier  │──▶ │ Crawler Workers  │
  └─────────────┘    │ (priority Q)  │    └────────┬─────────┘
                     └───────────────┘             │
                            ▲                      │ fetched HTML
                   URL Extractor                   ▼
                   + Dedup Filter          ┌──────────────────┐
                                           │   Content Store  │
                                           │   (S3 + Kafka)   │
                                           └──────────────────┘

KEY DESIGN DECISIONS:
1. URL FRONTIER — two-level queue for politeness:
   - Back queues: one per domain, each ordered by crawl time
   - Front queues: priority queues selecting from back queues
   - Selector: picks back queue based on politeness (earliest allowed fetch time)
   This ensures per-domain crawl rate limits without global lock.

2. URL DEDUPLICATION:
   - Bloom filter (approx): 10 B URLs × 10 bits = 12.5 GB → ~1% false positive
   - Exact dedup for production: URL fingerprint (MD5 → 16 bytes) in Cassandra
   - Canonicalize URLs before dedup (lowercase host, remove fragment, sort query params)

3. POLITENESS:
   - robots.txt cached per domain (TTL: 24h)
   - Crawl-Delay directive respected
   - Default: 1 req/10s/domain (conservative)
   - User-Agent: identify as googlebot-style crawler

4. LINK EXTRACTION:
   - Parse HTML (BeautifulSoup/lxml) for <a href=...> tags
   - Resolve relative URLs against base URL
   - Filter: only http/https, max depth, domain whitelist/blacklist

5. DISTRIBUTED COORDINATION:
   - URL partitioned by domain hash → consistent crawler ownership
   - Heartbeat + task queue (Kafka) for work distribution
   - Checkpointing frontier state to survive worker failure

6. CONTENT DEDUP:
   - SimHash of page content → detect near-duplicate pages
   - Same SimHash distance < threshold → skip indexing (not storing raw HTML)

7. PRIORITY SCORING:
   - Page importance (inbound links count, PageRank estimate)
   - Freshness (time since last crawl)
   - Domain authority (news sites, .gov, .edu = high priority)
"""

from __future__ import annotations
import time
import uuid
import hashlib
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from enum import Enum
import threading
import re
from urllib.parse import urlparse, urljoin, urlunparse, urlencode, parse_qs


# ---------------------------------------------------------------------------
# URL Canonicaliser
# ---------------------------------------------------------------------------

class URLCanonicaliser:
    """Normalises URLs to ensure deduplication works correctly."""

    # Query params to strip (session IDs, tracking params)
    STRIP_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "sid", "sessionid",
                     "fbclid", "gclid", "ref", "affiliate"}

    @staticmethod
    def canonicalise(url: str, base_url: str = "") -> Optional[str]:
        try:
            # Resolve relative URLs
            if base_url:
                url = urljoin(base_url, url)

            parsed = urlparse(url)

            # Only http/https
            if parsed.scheme not in ("http", "https"):
                return None

            # Lowercase host
            host = parsed.netloc.lower()
            if not host:
                return None

            # Strip fragment
            path = parsed.path or "/"

            # Normalise path (remove ./ ../)
            parts = []
            for part in path.split("/"):
                if part == "..":
                    if parts:
                        parts.pop()
                elif part != ".":
                    parts.append(part)
            path = "/".join(parts) or "/"

            # Sort and filter query params
            query_params = parse_qs(parsed.query, keep_blank_values=False)
            filtered = {k: v for k, v in query_params.items()
                        if k.lower() not in URLCanonicaliser.STRIP_PARAMS}
            sorted_query = "&".join(f"{k}={v[0]}" for k, v in sorted(filtered.items()))

            return urlunparse((parsed.scheme, host, path, "", sorted_query, ""))
        except Exception:
            return None

    @staticmethod
    def domain(url: str) -> Optional[str]:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return None

    @staticmethod
    def fingerprint(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Bloom Filter for URL Deduplication
# ---------------------------------------------------------------------------

class BloomFilter:
    """
    Space-efficient probabilistic membership test.
    False positive rate ~1% with m=10*n bits and k=7 hash functions.
    """

    def __init__(self, capacity: int = 1_000_000, error_rate: float = 0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        # m = -n * ln(p) / (ln 2)^2
        self.m = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        # k = m/n * ln2
        self.k = int(self.m / capacity * math.log(2))
        self._bits = bytearray(self.m // 8 + 1)
        self._count = 0

    def _hashes(self, item: str) -> List[int]:
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str) -> None:
        for bit_idx in self._hashes(item):
            self._bits[bit_idx // 8] |= (1 << (bit_idx % 8))
        self._count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            self._bits[bit_idx // 8] & (1 << (bit_idx % 8))
            for bit_idx in self._hashes(item)
        )

    @property
    def fill_ratio(self) -> float:
        bits_set = sum(bin(b).count("1") for b in self._bits)
        return bits_set / self.m


# ---------------------------------------------------------------------------
# Robots.txt Cache
# ---------------------------------------------------------------------------

@dataclass
class RobotsRules:
    domain: str
    disallowed: List[str]
    crawl_delay_seconds: float = 1.0
    fetched_at: float = field(default_factory=time.time)
    allow_all: bool = True

    def is_allowed(self, path: str, user_agent: str = "*") -> bool:
        if not self.allow_all:
            for rule in self.disallowed:
                if path.startswith(rule):
                    return False
        return True

    @property
    def is_stale(self) -> bool:
        return time.time() - self.fetched_at > 86400  # 24h TTL


class RobotsTxtCache:
    """Caches robots.txt rules per domain."""

    DEFAULT_RULES = RobotsRules("", [], crawl_delay_seconds=1.0)

    def __init__(self):
        self._cache: Dict[str, RobotsRules] = {}
        self._lock = threading.Lock()

    def get_rules(self, domain: str) -> RobotsRules:
        with self._lock:
            rules = self._cache.get(domain)
            if rules and not rules.is_stale:
                return rules
            # Simulate fetch (in real system: HTTP GET /robots.txt)
            rules = self._fetch_robots(domain)
            self._cache[domain] = rules
            return rules

    def _fetch_robots(self, domain: str) -> RobotsRules:
        """Simulate fetching robots.txt. Real: HTTP client."""
        # Simulated rules for demo
        simulated = {
            "example.com": RobotsRules("example.com", ["/private/", "/admin/"],
                                        crawl_delay_seconds=2.0),
            "news.com": RobotsRules("news.com", [], crawl_delay_seconds=0.5),
            "blocked.com": RobotsRules("blocked.com", ["/"], allow_all=False),
        }
        return simulated.get(domain, RobotsRules(domain, [], crawl_delay_seconds=1.0))


# ---------------------------------------------------------------------------
# URL Frontier — two-level polite queue
# ---------------------------------------------------------------------------

@dataclass
class FrontierURL:
    url: str
    domain: str
    priority: float     # Higher = crawl sooner
    earliest_fetch: float  # Politeness: don't fetch before this time
    depth: int = 0
    added_at: float = field(default_factory=time.time)

    def __lt__(self, other: "FrontierURL") -> bool:
        return self.priority > other.priority  # Max-heap by priority


class URLFrontier:
    """
    Two-level queue: front queue (priority) + per-domain back queues.
    Politeness enforced via earliest_fetch per domain.
    """

    def __init__(self, robots_cache: RobotsTxtCache, max_size: int = 100_000):
        self._robots = robots_cache
        self._max_size = max_size
        # domain → deque of FrontierURL (sorted by priority)
        self._back_queues: Dict[str, deque] = defaultdict(deque)
        # domain → next_allowed_fetch_time
        self._domain_next_fetch: Dict[str, float] = {}
        self._total = 0
        self._lock = threading.Lock()

    def add(self, url: str, priority: float = 0.5, depth: int = 0) -> bool:
        with self._lock:
            if self._total >= self._max_size:
                return False
            domain = URLCanonicaliser.domain(url) or "unknown"
            rules = self._robots.get_rules(domain)
            path = urlparse(url).path
            if not rules.is_allowed(path):
                return False  # Disallowed by robots.txt

            delay = rules.crawl_delay_seconds
            earliest = self._domain_next_fetch.get(domain, time.time())
            entry = FrontierURL(url, domain, priority, earliest_fetch=earliest, depth=depth)
            self._back_queues[domain].append(entry)
            self._total += 1
            return True

    def get_next(self) -> Optional[FrontierURL]:
        """Get next URL that can be fetched right now (politeness)."""
        with self._lock:
            now = time.time()
            # Find domain with earliest allowed fetch time that has items
            best_domain = None
            best_time = float("inf")
            for domain, q in self._back_queues.items():
                if not q:
                    continue
                next_time = self._domain_next_fetch.get(domain, 0)
                if next_time <= now and next_time < best_time:
                    best_time = next_time
                    best_domain = domain

            if not best_domain:
                return None

            entry = self._back_queues[best_domain].popleft()
            # Update domain's next fetch time based on crawl-delay
            rules = self._robots.get_rules(best_domain)
            self._domain_next_fetch[best_domain] = time.time() + rules.crawl_delay_seconds
            self._total -= 1
            if not self._back_queues[best_domain]:
                del self._back_queues[best_domain]
            return entry

    def size(self) -> int:
        return self._total

    def active_domains(self) -> int:
        return len(self._back_queues)


# ---------------------------------------------------------------------------
# Content Store
# ---------------------------------------------------------------------------

@dataclass
class CrawledPage:
    page_id: str
    url: str
    domain: str
    status_code: int
    content_type: str
    html: str
    content_hash: str      # SHA-256 for exact dedup
    content_length: int
    links_found: int
    crawled_at: float = field(default_factory=time.time)
    depth: int = 0


class ContentStore:
    def __init__(self):
        self._pages: Dict[str, CrawledPage] = {}   # page_id → page
        self._by_url: Dict[str, str] = {}           # url → page_id
        self._content_hashes: Set[str] = set()      # exact content dedup

    def save(self, page: CrawledPage) -> bool:
        """Returns True if content is new (not duplicate)."""
        if page.content_hash in self._content_hashes:
            return False  # Content duplicate
        self._pages[page.page_id] = page
        self._by_url[page.url] = page.page_id
        self._content_hashes.add(page.content_hash)
        return True

    def get(self, url: str) -> Optional[CrawledPage]:
        pid = self._by_url.get(url)
        return self._pages.get(pid) if pid else None

    @property
    def total_pages(self) -> int:
        return len(self._pages)

    @property
    def total_bytes(self) -> int:
        return sum(p.content_length for p in self._pages.values())


# ---------------------------------------------------------------------------
# Crawler Worker
# ---------------------------------------------------------------------------

class HTMLParser:
    """Simulates HTML parsing for link extraction."""

    LINK_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

    @staticmethod
    def extract_links(html: str, base_url: str) -> List[str]:
        raw_links = HTMLParser.LINK_PATTERN.findall(html)
        links = []
        for link in raw_links:
            canonical = URLCanonicaliser.canonicalise(link, base_url)
            if canonical:
                links.append(canonical)
        return list(set(links))

    @staticmethod
    def extract_text(html: str) -> str:
        """Strip HTML tags (simplified)."""
        clean = re.sub(r"<[^>]+>", " ", html)
        return " ".join(clean.split())


class CrawlerWorker:
    """Simulates a crawler worker that fetches URLs."""

    def __init__(self, worker_id: str, frontier: URLFrontier,
                 store: ContentStore, bloom: BloomFilter):
        self.worker_id = worker_id
        self._frontier = frontier
        self._store = store
        self._bloom = bloom
        self.pages_crawled = 0
        self.pages_skipped = 0
        self.bytes_downloaded = 0

    def crawl_one(self) -> Optional[CrawledPage]:
        """Fetch one URL from frontier."""
        entry = self._frontier.get_next()
        if not entry:
            return None

        # Check dedup
        fp = URLCanonicaliser.fingerprint(entry.url)
        if fp in self._bloom:
            self.pages_skipped += 1
            return None
        self._bloom.add(fp)

        # Simulate HTTP fetch
        html, status = self._fetch(entry.url)
        if status != 200 or not html:
            return None

        # Parse links
        links = HTMLParser.extract_links(html, entry.url)

        # Priority function for extracted links (simulated)
        for link in links[:50]:  # Limit link extraction to avoid explosion
            priority = 0.5 - entry.depth * 0.1  # Depth penalty
            self._frontier.add(link, priority=max(0.1, priority),
                                depth=entry.depth + 1)

        # Save page
        content_hash = hashlib.sha256(html.encode()).hexdigest()
        page = CrawledPage(
            page_id=str(uuid.uuid4()),
            url=entry.url,
            domain=entry.domain,
            status_code=status,
            content_type="text/html",
            html=html[:500],   # truncate for demo
            content_hash=content_hash,
            content_length=len(html),
            links_found=len(links),
            depth=entry.depth,
        )
        is_new = self._store.save(page)
        if is_new:
            self.pages_crawled += 1
            self.bytes_downloaded += len(html)
        return page if is_new else None

    def _fetch(self, url: str) -> Tuple[str, int]:
        """Simulate HTTP fetch. Returns (html, status_code)."""
        # Simulate occasional failures
        if random.random() < 0.05:
            return "", 404
        if random.random() < 0.02:
            return "", 500

        domain = URLCanonicaliser.domain(url) or "unknown"
        # Generate realistic-looking simulated HTML
        path = urlparse(url).path
        html = (
            f'<html><head><title>Page at {path}</title></head>'
            f'<body><p>Content of {domain}{path}</p>'
            f'<a href="/page1">Link 1</a>'
            f'<a href="/page2">Link 2</a>'
            f'<a href="https://other.com/article">External Link</a>'
            f'</body></html>'
        )
        return html, 200


# ---------------------------------------------------------------------------
# Crawl Manager
# ---------------------------------------------------------------------------

class CrawlManager:
    """Orchestrates the crawl."""

    def __init__(self, seed_urls: List[str]):
        self._robots = RobotsTxtCache()
        self._frontier = URLFrontier(self._robots, max_size=10_000)
        self._store = ContentStore()
        self._bloom = BloomFilter(capacity=100_000, error_rate=0.01)
        self._workers: List[CrawlerWorker] = []

        # Seed the frontier
        for url in seed_urls:
            canonical = URLCanonicaliser.canonicalise(url)
            if canonical:
                self._frontier.add(canonical, priority=1.0, depth=0)

    def add_worker(self, n: int = 1) -> None:
        for _ in range(n):
            w = CrawlerWorker(
                f"worker_{len(self._workers)+1}",
                self._frontier, self._store, self._bloom
            )
            self._workers.append(w)

    def crawl(self, max_pages: int = 50) -> Dict:
        """Run crawl until max_pages or frontier empty."""
        pages = 0
        while pages < max_pages:
            crawled = False
            for worker in self._workers:
                if self._frontier.size() == 0:
                    break
                result = worker.crawl_one()
                if result:
                    pages += 1
                    crawled = True
                if pages >= max_pages:
                    break
            if not crawled and self._frontier.size() == 0:
                break

        total_crawled = sum(w.pages_crawled for w in self._workers)
        total_skipped = sum(w.pages_skipped for w in self._workers)
        total_bytes = sum(w.bytes_downloaded for w in self._workers)

        return {
            "pages_crawled": total_crawled,
            "pages_skipped_dedup": total_skipped,
            "bytes_downloaded": total_bytes,
            "unique_content": self._store.total_pages,
            "frontier_remaining": self._frontier.size(),
            "bloom_fill": self._bloom.fill_ratio,
        }

    @property
    def store(self) -> ContentStore:
        return self._store

    @property
    def frontier(self) -> URLFrontier:
        return self._frontier


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def demonstrate_1_url_canonicalisation():
    print("\n=== 1. URL Canonicalisation ===")
    test_cases = [
        ("http://Example.COM/foo/../bar?b=1&a=2", ""),
        ("https://example.com/page#section", ""),
        ("/relative/path", "https://example.com/base/"),
        ("https://site.com/page?utm_source=google&id=1", ""),
        ("ftp://example.com/file", ""),
        ("  https://GOOGLE.com/search?q=python  ", ""),
    ]
    for url, base in test_cases:
        canonical = URLCanonicaliser.canonicalise(url.strip(), base)
        print(f"  {url.strip()[:50]:<55} → {canonical}")


def demonstrate_2_bloom_filter():
    print("\n=== 2. Bloom Filter for URL Dedup ===")
    bloom = BloomFilter(capacity=10000, error_rate=0.01)

    # Add 5000 URLs
    added = set()
    for i in range(5000):
        url = f"https://example.com/page/{i}"
        bloom.add(URLCanonicaliser.fingerprint(url))
        added.add(url)

    # Check known URLs (should all return True)
    hits = sum(1 for url in added if URLCanonicaliser.fingerprint(url) in bloom)
    print(f"True positives: {hits}/{len(added)} ({hits/len(added):.1%})")

    # Check unknown URLs (should mostly return False)
    false_positives = 0
    for i in range(5000, 10000):
        url = f"https://example.com/page/{i}"
        if URLCanonicaliser.fingerprint(url) in bloom:
            false_positives += 1
    print(f"False positives: {false_positives}/5000 "
          f"({false_positives/5000:.2%}) — target: ~1%")
    print(f"Bloom filter fill ratio: {bloom.fill_ratio:.2%}")
    print(f"Bit array size: {bloom.m // 8} bytes ({bloom.m} bits)")


def demonstrate_3_robots_and_politeness():
    print("\n=== 3. Robots.txt & Politeness ===")
    robots = RobotsTxtCache()

    test_cases = [
        ("example.com", "/public/page"),
        ("example.com", "/private/data"),
        ("blocked.com", "/"),
        ("news.com", "/article/123"),
    ]
    for domain, path in test_cases:
        rules = robots.get_rules(domain)
        allowed = rules.is_allowed(path)
        print(f"  {domain}{path}: "
              f"{'ALLOWED' if allowed else 'BLOCKED'} "
              f"(delay={rules.crawl_delay_seconds}s)")


def demonstrate_4_frontier():
    print("\n=== 4. URL Frontier — Polite Crawling ===")
    robots = RobotsTxtCache()
    frontier = URLFrontier(robots, max_size=1000)

    # Add URLs from different domains
    seeds = [
        ("https://news.com/article/1", 0.9),
        ("https://news.com/article/2", 0.8),
        ("https://example.com/page/1", 0.7),
        ("https://blog.com/post/1", 0.6),
        ("https://blocked.com/page", 0.9),   # Should be rejected
    ]
    for url, priority in seeds:
        added = frontier.add(url, priority=priority)
        print(f"  Add {url[:50]}: {'OK' if added else 'REJECTED (robots.txt)'}")

    print(f"\nFrontier size: {frontier.size()}, active domains: {frontier.active_domains()}")

    # Drain frontier
    fetched = []
    for _ in range(5):
        entry = frontier.get_next()
        if entry:
            fetched.append(entry)
    print(f"Fetched {len(fetched)} URLs (politeness ordering):")
    for e in fetched:
        print(f"  [{e.domain}] {e.url}")


def demonstrate_5_full_crawl():
    print("\n=== 5. Full Crawl Simulation ===")
    seeds = [
        "https://news.com/",
        "https://example.com/",
        "https://blog.com/",
        "https://tech.io/",
    ]
    manager = CrawlManager(seeds)
    manager.add_worker(n=3)

    stats = manager.crawl(max_pages=30)

    print(f"Crawl complete:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        elif isinstance(v, int) and v > 1000:
            print(f"  {k}: {v:,}")
        else:
            print(f"  {k}: {v}")

    # Show sample pages
    pages = list(manager.store._pages.values())[:3]
    print(f"\nSample crawled pages:")
    for p in pages:
        print(f"  [{p.status_code}] {p.url} ({p.content_length}B, {p.links_found} links)")


if __name__ == "__main__":
    demonstrate_1_url_canonicalisation()
    demonstrate_2_bloom_filter()
    demonstrate_3_robots_and_politeness()
    demonstrate_4_frontier()
    demonstrate_5_full_crawl()
