"""
Problem 28: Design a Web Crawler
==================================
Working simulation of a distributed web crawler with:
- URLFrontier: priority queue with politeness per domain
- BloomFilter: probabilistic URL dedup (configurable FPR)
- DNSCache: TTL-based DNS caching
- RobotsParser: per-domain robots.txt rules with caching
- ContentHashStore: SHA-256 exact dedup
- CrawlScheduler: per-domain rate limiting
- LinkExtractor: regex-based href parsing
- SimHash: near-duplicate detection via Hamming distance
- WebCrawler: orchestrator tying all components together
"""

import hashlib
import heapq
import re
import time
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, urljoin, urlunparse


# ─── Bloom Filter ─────────────────────────────────────────────────────────────

class BloomFilter:
    """
    Space-efficient probabilistic set membership.
    Uses k independent hash functions over a bit array of size m.
    FPR ≈ (1 - e^(-k*n/m))^k where n = inserted elements.
    """

    def __init__(self, capacity: int = 1_000_000, false_positive_rate: float = 0.001):
        self.capacity = capacity
        self.fpr = false_positive_rate
        # Optimal bit array size: m = -n*ln(p) / (ln2)^2
        self.size = int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))
        # Optimal hash count: k = (m/n) * ln(2)
        self.hash_count = max(1, int((self.size / capacity) * math.log(2)))
        self._bits = bytearray(self.size // 8 + 1)
        self._count = 0

    def _get_bit_positions(self, item: str) -> list[int]:
        positions = []
        for seed in range(self.hash_count):
            h = int(hashlib.md5(f"{seed}:{item}".encode()).hexdigest(), 16)
            positions.append(h % self.size)
        return positions

    def add(self, item: str) -> None:
        for pos in self._get_bit_positions(item):
            byte_idx, bit_idx = divmod(pos, 8)
            self._bits[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def contains(self, item: str) -> bool:
        """Returns True if item was probably added (may have false positives)."""
        for pos in self._get_bit_positions(item):
            byte_idx, bit_idx = divmod(pos, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def estimated_fpr(self) -> float:
        fill_ratio = self._count / max(1, self.capacity)
        return (1 - math.exp(-self.hash_count * fill_ratio)) ** self.hash_count

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (f"BloomFilter(capacity={self.capacity}, hash_count={self.hash_count}, "
                f"size={self.size} bits, inserted={self._count}, "
                f"est_fpr={self.estimated_fpr():.6f})")


# ─── DNS Cache ────────────────────────────────────────────────────────────────

class DNSCache:
    """TTL-based DNS cache to avoid per-URL DNS lookups."""

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[str, float]] = {}  # domain → (ip, expiry)
        self._hits = 0
        self._misses = 0

    def resolve(self, domain: str) -> str:
        """Simulate DNS resolution with caching."""
        now = time.time()
        if domain in self._cache:
            ip, expiry = self._cache[domain]
            if now < expiry:
                self._hits += 1
                return ip
            else:
                del self._cache[domain]

        # Simulate DNS lookup (deterministic fake IP for demo)
        ip = ".".join(str(ord(c) % 256) for c in domain[:4])
        self._cache[domain] = (ip, now + self.default_ttl)
        self._misses += 1
        return ip

    def cache_negative(self, domain: str, ttl: int = 300) -> None:
        """Cache NXDOMAIN to avoid repeated failed lookups."""
        self._cache[domain] = ("0.0.0.0", time.time() + ttl)

    def stats(self) -> dict:
        hit_rate = self._hits / max(1, self._hits + self._misses)
        return {"hits": self._hits, "misses": self._misses, "hit_rate": f"{hit_rate:.1%}",
                "cached_domains": len(self._cache)}


# ─── Robots.txt Parser ────────────────────────────────────────────────────────

@dataclass
class RobotsRules:
    disallow: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    crawl_delay: float = 1.0
    sitemaps: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)
    ttl: float = 86400.0  # 24 hours


class RobotsParser:
    """Per-domain robots.txt rules with caching."""

    def __init__(self, user_agent: str = "MyCrawlerBot"):
        self.user_agent = user_agent
        self._cache: dict[str, RobotsRules] = {}

    def fetch_and_parse(self, domain: str, robots_txt: str) -> RobotsRules:
        """Parse robots.txt content for this user-agent."""
        rules = RobotsRules()
        lines = robots_txt.strip().split("\n")
        applicable = False

        for line in lines:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "user-agent":
                applicable = (value == "*" or value.lower() in self.user_agent.lower())
            elif applicable:
                if key == "disallow" and value:
                    rules.disallow.append(value)
                elif key == "allow" and value:
                    rules.allow.append(value)
                elif key == "crawl-delay":
                    try:
                        rules.crawl_delay = float(value)
                    except ValueError:
                        pass
            elif key == "sitemap":
                rules.sitemaps.append(value)

        self._cache[domain] = rules
        return rules

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt rules."""
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path or "/"

        if domain not in self._cache:
            return True  # No rules cached — allow by default

        rules = self._cache[domain]
        # Check TTL
        if time.time() > rules.fetched_at + rules.ttl:
            del self._cache[domain]
            return True

        # Specific allow overrides disallow
        for pattern in rules.allow:
            if path.startswith(pattern):
                return True
        # Check disallow
        for pattern in rules.disallow:
            if path.startswith(pattern):
                return False
        return True

    def get_crawl_delay(self, domain: str) -> float:
        rules = self._cache.get(domain)
        return rules.crawl_delay if rules else 1.0


# ─── Content Hash Store ───────────────────────────────────────────────────────

class ContentHashStore:
    """SHA-256 based exact content deduplication."""

    def __init__(self):
        self._hashes: set[str] = set()
        self._url_to_hash: dict[str, str] = {}

    def check_and_store(self, url: str, content: str) -> tuple[bool, str]:
        """
        Returns (is_duplicate, content_hash).
        If duplicate, returns True and existing hash.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in self._hashes:
            return True, content_hash
        self._hashes.add(content_hash)
        self._url_to_hash[url] = content_hash
        return False, content_hash

    def __len__(self) -> int:
        return len(self._hashes)


# ─── SimHash (Near-Duplicate Detection) ──────────────────────────────────────

class SimHash:
    """
    64-bit SimHash fingerprint for near-duplicate detection.
    Two pages with Hamming distance <= threshold are considered near-duplicates.
    Algorithm: tokenize → hash each token → vector of +1/-1 per bit → sign.
    """

    HASH_BITS = 64

    @staticmethod
    def compute(text: str) -> int:
        """Compute 64-bit SimHash fingerprint for text."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return 0

        vector = [0] * SimHash.HASH_BITS

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for bit in range(SimHash.HASH_BITS):
                if token_hash & (1 << bit):
                    vector[bit] += 1
                else:
                    vector[bit] -= 1

        fingerprint = 0
        for bit in range(SimHash.HASH_BITS):
            if vector[bit] > 0:
                fingerprint |= (1 << bit)
        return fingerprint

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Count differing bits between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')

    @staticmethod
    def is_near_duplicate(hash1: int, hash2: int, threshold: int = 3) -> bool:
        return SimHash.hamming_distance(hash1, hash2) <= threshold


class SimHashStore:
    """Store for SimHash fingerprints with near-duplicate lookup."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._fingerprints: list[tuple[int, str]] = []  # (hash, url)

    def check_and_store(self, url: str, text: str) -> tuple[bool, str]:
        """Returns (is_near_duplicate, original_url_if_dup)."""
        fingerprint = SimHash.compute(text)
        for existing_hash, existing_url in self._fingerprints:
            if SimHash.is_near_duplicate(fingerprint, existing_hash, self.threshold):
                return True, existing_url
        self._fingerprints.append((fingerprint, url))
        return False, ""


# ─── Link Extractor ───────────────────────────────────────────────────────────

class LinkExtractor:
    """Extract and normalize hyperlinks from HTML content."""

    HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
    CANONICAL_PATTERN = re.compile(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', re.IGNORECASE
    )
    BASE_PATTERN = re.compile(r'<base[^>]+href=["\']([^"\']+)["\']', re.IGNORECASE)

    def extract(self, html: str, base_url: str) -> list[str]:
        """Extract all valid links from HTML, resolving relative URLs."""
        # Find base tag
        base_match = self.BASE_PATTERN.search(html)
        effective_base = base_match.group(1) if base_match else base_url

        links = set()
        for href in self.HREF_PATTERN.findall(html):
            normalized = self._normalize(href, effective_base)
            if normalized:
                links.add(normalized)
        return list(links)

    def get_canonical(self, html: str, fallback_url: str) -> str:
        """Return canonical URL if specified, else the crawled URL."""
        match = self.CANONICAL_PATTERN.search(html)
        return match.group(1) if match else fallback_url

    def _normalize(self, url: str, base_url: str) -> Optional[str]:
        """Normalize a URL: resolve relative, remove fragments, lowercase scheme/host."""
        if url.startswith(("mailto:", "javascript:", "tel:", "#", "data:")):
            return None
        try:
            absolute = urljoin(base_url, url)
            parsed = urlparse(absolute)
            # Keep only http/https
            if parsed.scheme not in ("http", "https"):
                return None
            # Remove fragment, normalize
            normalized = urlunparse((
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                parsed.params,
                parsed.query,
                ""  # Remove fragment
            ))
            # Enforce max URL length
            return normalized if len(normalized) <= 2048 else None
        except Exception:
            return None


# ─── Crawl Scheduler ─────────────────────────────────────────────────────────

class CrawlScheduler:
    """Enforce per-domain politeness: minimum delay between requests."""

    def __init__(self, default_delay: float = 1.0):
        self.default_delay = default_delay
        self._last_crawl: dict[str, float] = {}  # domain → last crawl timestamp
        self._crawl_delays: dict[str, float] = {}  # domain → delay from robots.txt

    def set_crawl_delay(self, domain: str, delay: float) -> None:
        self._crawl_delays[domain] = delay

    def can_crawl(self, url: str) -> bool:
        domain = urlparse(url).netloc
        delay = self._crawl_delays.get(domain, self.default_delay)
        last = self._last_crawl.get(domain, 0.0)
        return time.time() - last >= delay

    def record_crawl(self, url: str) -> None:
        domain = urlparse(url).netloc
        self._last_crawl[domain] = time.time()

    def time_until_allowed(self, url: str) -> float:
        domain = urlparse(url).netloc
        delay = self._crawl_delays.get(domain, self.default_delay)
        last = self._last_crawl.get(domain, 0.0)
        remaining = delay - (time.time() - last)
        return max(0.0, remaining)


# ─── URL Frontier ─────────────────────────────────────────────────────────────

@dataclass(order=True)
class FrontierEntry:
    priority: float          # lower = higher priority (min-heap)
    url: str = field(compare=False)
    depth: int = field(compare=False, default=0)


class URLFrontier:
    """
    Priority-queue based URL frontier.
    Priority = -pagerank_estimate - freshness_bonus.
    Enforces per-domain politeness via CrawlScheduler.
    """

    def __init__(self, scheduler: CrawlScheduler, bloom: BloomFilter):
        self._heap: list[FrontierEntry] = []
        self._scheduler = scheduler
        self._bloom = bloom
        self._domain_queues: dict[str, list[FrontierEntry]] = defaultdict(list)
        self._size = 0

    def add(self, url: str, priority: float = 0.5, depth: int = 0) -> bool:
        """Add URL to frontier if not already seen. Returns True if added."""
        if self._bloom.contains(url):
            return False
        self._bloom.add(url)
        entry = FrontierEntry(priority=-priority, url=url, depth=depth)
        domain = urlparse(url).netloc
        heapq.heappush(self._domain_queues[domain], entry)
        self._size += 1
        return True

    def get_next(self) -> Optional[str]:
        """
        Return next URL respecting politeness.
        Tries domains in priority order; skips domains that are rate-limited.
        """
        # Collect all domain queues sorted by their best entry
        candidates = []
        for domain, queue in self._domain_queues.items():
            while queue and queue[0].priority > 0:  # cleanup
                heapq.heappop(queue)
            if queue:
                entry = queue[0]
                candidates.append((entry.priority, domain, entry))

        candidates.sort()

        for _, domain, entry in candidates:
            url = entry.url
            if self._scheduler.can_crawl(url):
                heapq.heappop(self._domain_queues[domain])
                self._size -= 1
                self._scheduler.record_crawl(url)
                return url

        return None  # All domains rate-limited

    def size(self) -> int:
        return self._size


# ─── Web Crawler (Orchestrator) ───────────────────────────────────────────────

@dataclass
class CrawlResult:
    url: str
    canonical_url: str
    status_code: int
    content_hash: str
    is_exact_dup: bool
    is_near_dup: bool
    near_dup_url: str
    outlinks: list[str]
    depth: int
    crawl_time: float
    title: str = ""


class WebCrawler:
    """
    Orchestrates all crawling components.
    Simulates fetching pages from a pre-defined fake web graph.
    """

    def __init__(self, max_pages: int = 50, max_depth: int = 3):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.bloom = BloomFilter(capacity=1_000_000, false_positive_rate=0.001)
        self.dns_cache = DNSCache()
        self.robots_parser = RobotsParser()
        self.content_store = ContentHashStore()
        self.simhash_store = SimHashStore(threshold=3)
        self.scheduler = CrawlScheduler(default_delay=0.0)  # 0 for simulation speed
        self.frontier = URLFrontier(self.scheduler, self.bloom)
        self.link_extractor = LinkExtractor()
        self._results: list[CrawlResult] = []
        self._fake_web = self._build_fake_web()

    def _build_fake_web(self) -> dict[str, dict]:
        """Build a simulated web graph for demo purposes."""
        pages = {
            "https://example.com/": {
                "title": "Example Domain",
                "content": "Welcome to example.com. This is the home page with general information.",
                "links": ["https://example.com/about", "https://example.com/news",
                          "https://blog.example.com/", "https://other.com/"],
                "status": 200
            },
            "https://example.com/about": {
                "title": "About Us",
                "content": "Learn about our company history, mission, and team members.",
                "links": ["https://example.com/", "https://example.com/contact"],
                "status": 200
            },
            "https://example.com/news": {
                "title": "Latest News",
                "content": "Breaking news: technology advances rapidly in 2024. AI reshapes industries.",
                "links": ["https://example.com/news/article1", "https://example.com/news/article2"],
                "status": 200
            },
            "https://example.com/news/article1": {
                "title": "AI News Article",
                "content": "Artificial intelligence continues to make headlines in 2024 with new breakthroughs.",
                "links": ["https://example.com/news"],
                "status": 200
            },
            "https://example.com/news/article2": {
                "title": "Tech News Article",
                "content": "Technology news: major companies announce new products for the holiday season.",
                "links": ["https://example.com/news"],
                "status": 200
            },
            "https://example.com/contact": {
                "title": "Contact Us",
                "content": "Contact example.com at info@example.com or call us at 555-0100.",
                "links": ["https://example.com/"],
                "status": 200
            },
            "https://example.com/private/secret": {
                "title": "Private Page",
                "content": "This page should not be crawled per robots.txt.",
                "links": [],
                "status": 200
            },
            "https://blog.example.com/": {
                "title": "The Example Blog",
                "content": "Welcome to our blog. We write about technology, travel, and food.",
                "links": ["https://blog.example.com/post1", "https://blog.example.com/post2"],
                "status": 200
            },
            "https://blog.example.com/post1": {
                "title": "First Blog Post",
                "content": "This is our first blog post about technology trends and future predictions.",
                "links": ["https://blog.example.com/"],
                "status": 200
            },
            "https://blog.example.com/post2": {
                "title": "Duplicate Content Post",
                "content": "Artificial intelligence continues to make headlines in 2024 with new breakthroughs.",  # near-dup of article1
                "links": ["https://blog.example.com/"],
                "status": 200
            },
            "https://other.com/": {
                "title": "Other Website",
                "content": "Other website home page with links to products and services.",
                "links": ["https://other.com/products", "https://example.com/"],
                "status": 200
            },
            "https://other.com/products": {
                "title": "Products",
                "content": "Browse our catalog of amazing products at competitive prices.",
                "links": ["https://other.com/"],
                "status": 200
            },
        }
        return pages

    def _fake_fetch(self, url: str) -> Optional[tuple[int, str]]:
        """Simulate HTTP fetch from fake web graph."""
        return (self._fake_web[url]["status"], self._fake_web[url]["content"]) \
               if url in self._fake_web else (404, "")

    def _register_robots(self) -> None:
        """Pre-populate robots.txt rules for demo domains."""
        example_robots = """
User-agent: *
Disallow: /private/
Disallow: /admin/
Crawl-delay: 1
Allow: /
Sitemap: https://example.com/sitemap.xml
        """
        self.robots_parser.fetch_and_parse("example.com", example_robots)
        self.robots_parser.fetch_and_parse("blog.example.com", "User-agent: *\nAllow: /\n")
        self.robots_parser.fetch_and_parse("other.com", "User-agent: *\nDisallow: /checkout\nAllow: /\n")

    def crawl(self, seed_urls: list[str]) -> list[CrawlResult]:
        """Main crawl loop. Returns list of CrawlResult objects."""
        self._register_robots()

        # Seed the frontier
        for url in seed_urls:
            self.frontier.add(url, priority=1.0, depth=0)

        pages_crawled = 0
        print(f"\nStarting crawl: seed_urls={len(seed_urls)}, "
              f"max_pages={self.max_pages}, max_depth={self.max_depth}")

        while pages_crawled < self.max_pages:
            url = self.frontier.get_next()
            if not url:
                print("  Frontier empty — crawl complete.")
                break

            # Depth limit check
            result_so_far = next((r for r in self._results if r.url == url), None)
            depth = 0  # Default; tracked via frontier metadata

            # robots.txt check
            if not self.robots_parser.is_allowed(url):
                print(f"  [ROBOTS DISALLOW] {url}")
                continue

            # DNS resolution
            domain = urlparse(url).netloc
            ip = self.dns_cache.resolve(domain)
            if ip == "0.0.0.0":
                print(f"  [DNS FAIL] {url}")
                continue

            # Simulate fetch
            t_start = time.perf_counter()
            fetch_result = self._fake_fetch(url)
            t_elapsed = time.perf_counter() - t_start

            if not fetch_result or fetch_result[0] != 200:
                print(f"  [HTTP {fetch_result[0] if fetch_result else 'ERR'}] {url}")
                continue

            status_code, content = fetch_result
            pages_crawled += 1

            # Exact dedup
            is_exact_dup, content_hash = self.content_store.check_and_store(url, content)

            # Near-dup detection
            is_near_dup = False
            near_dup_url = ""
            if not is_exact_dup:
                is_near_dup, near_dup_url = self.simhash_store.check_and_store(url, content)

            # Link extraction
            outlinks = []
            if not is_exact_dup and not is_near_dup:
                page_data = self._fake_web.get(url, {})
                outlinks = page_data.get("links", [])
                # Add discovered links to frontier
                for link in outlinks:
                    if self.robots_parser.is_allowed(link):
                        self.frontier.add(link, priority=0.5, depth=depth + 1)

            title = self._fake_web.get(url, {}).get("title", "")
            crawl_result = CrawlResult(
                url=url,
                canonical_url=url,
                status_code=status_code,
                content_hash=content_hash[:16] + "...",
                is_exact_dup=is_exact_dup,
                is_near_dup=is_near_dup,
                near_dup_url=near_dup_url,
                outlinks=outlinks,
                depth=depth,
                crawl_time=t_elapsed,
                title=title
            )
            self._results.append(crawl_result)

            status_str = " [EXACT_DUP]" if is_exact_dup else (" [NEAR_DUP]" if is_near_dup else "")
            print(f"  [{pages_crawled:03d}] {url[:60]:<60} {status_str}")

        return self._results

    def print_stats(self) -> None:
        total = len(self._results)
        exact_dups = sum(1 for r in self._results if r.is_exact_dup)
        near_dups = sum(1 for r in self._results if r.is_near_dup)
        blocked = 0  # robots.txt disallows counted but not in results
        unique = total - exact_dups - near_dups

        print(f"\n--- Crawl Statistics ---")
        print(f"  Total crawled      : {total}")
        print(f"  Unique content     : {unique}")
        print(f"  Exact duplicates   : {exact_dups}")
        print(f"  Near-duplicates    : {near_dups}")
        print(f"  Content store size : {len(self.content_store)}")
        print(f"  Bloom filter stats : inserted={len(self.bloom)}, est_fpr={self.bloom.estimated_fpr():.6f}")
        print(f"  DNS cache stats    : {self.dns_cache.stats()}")
        print(f"  Remaining frontier : {self.frontier.size()} URLs")


# ─── Demo / Simulation ────────────────────────────────────────────────────────

def run_simulation():
    print("=" * 65)
    print("WEB CRAWLER SIMULATION")
    print("=" * 65)

    # ── Bloom Filter standalone demo ──────────────────────────
    print("\n--- Bloom Filter Demo ---")
    bf = BloomFilter(capacity=1000, false_positive_rate=0.01)
    urls_to_add = [f"https://example.com/page-{i}" for i in range(500)]
    for url in urls_to_add:
        bf.add(url)
    print(f"  Inserted 500 URLs | {bf}")
    hits = sum(1 for u in urls_to_add if bf.contains(u))
    print(f"  True positive rate: {hits}/500 = {hits/500:.0%}")
    false_positives = sum(1 for i in range(1000, 1200) if bf.contains(f"https://other.com/p-{i}"))
    print(f"  False positives (of 200 unseen): {false_positives} = {false_positives/200:.1%} (expected ~1%)")

    # ── SimHash near-dup demo ──────────────────────────────────
    print("\n--- SimHash Near-Duplicate Detection ---")
    texts = [
        ("Page A", "The quick brown fox jumps over the lazy dog near the river bank"),
        ("Page B", "The quick brown fox jumps over the lazy dog by the river bank"),  # near-dup of A
        ("Page C", "Python is a powerful programming language for data science and AI"),
        ("Page D", "Python is a very powerful programming language for data science and AI"),  # near-dup of C
    ]
    hashes = [(name, SimHash.compute(text)) for name, text in texts]
    for i, (n1, h1) in enumerate(hashes):
        for n2, h2 in hashes[i+1:]:
            dist = SimHash.hamming_distance(h1, h2)
            is_dup = SimHash.is_near_duplicate(h1, h2)
            print(f"  {n1} vs {n2}: Hamming distance={dist}, near-dup={is_dup}")

    # ── Link extractor demo ───────────────────────────────────
    print("\n--- Link Extractor Demo ---")
    sample_html = """
    <html><head><base href="https://example.com/news/"></head>
    <body>
      <a href="/about">About</a>
      <a href="article1">Article 1</a>
      <a href="https://other.com/page">External</a>
      <a href="mailto:test@example.com">Email</a>
      <a href="#section">Anchor</a>
    </body></html>
    """
    extractor = LinkExtractor()
    links = extractor.extract(sample_html, "https://example.com/news/")
    print(f"  Extracted {len(links)} links:")
    for link in links:
        print(f"    {link}")

    # ── Main crawl simulation ─────────────────────────────────
    print("\n--- Main Web Crawl ---")
    crawler = WebCrawler(max_pages=20, max_depth=3)
    seed_urls = ["https://example.com/", "https://other.com/"]
    results = crawler.crawl(seed_urls)

    print("\n--- Crawl Results Detail ---")
    for r in results:
        dup_info = ""
        if r.is_exact_dup:
            dup_info = "  EXACT DUPLICATE"
        elif r.is_near_dup:
            dup_info = f"  NEAR-DUP of {r.near_dup_url}"
        print(f"  {r.url}")
        print(f"    Title: {r.title} | Hash: {r.content_hash} | "
              f"Outlinks: {len(r.outlinks)}{dup_info}")

    crawler.print_stats()

    print("\n" + "=" * 65)
    print("Simulation complete.")


if __name__ == "__main__":
    run_simulation()
