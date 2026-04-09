"""
DNS AND HOW IT WORKS
=====================

Problem Statement:
DNS (Domain Name System) translates human-readable domain names to IP addresses.
Understanding DNS is critical for system design: it affects latency, availability,
and enables powerful patterns like GeoDNS and load balancing at the DNS level.

DNS Resolution Steps (Recursive):
  1. Client checks local OS cache
  2. Recursive resolver (ISP / 8.8.8.8) checks its cache
  3. Root nameserver → ".com" TLD nameserver → Authoritative nameserver
  4. Authoritative returns the A record (IP address)

DNS Record Types:
  A      : domain → IPv4 address
  AAAA   : domain → IPv6 address
  CNAME  : domain → another domain (alias)
  MX     : domain → mail server
  TXT    : arbitrary text (SPF, DKIM, verification)
  NS     : domain → nameserver
  SOA    : start of authority

TTL (Time To Live):
  Controls how long records are cached. Lower TTL = faster propagation but more DNS traffic.
  For failover: TTL=60s. For stable records: TTL=3600s.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import hashlib


class DNSRecordType(Enum):
    A     = "A"      # IPv4
    AAAA  = "AAAA"   # IPv6
    CNAME = "CNAME"  # Alias
    MX    = "MX"     # Mail
    TXT   = "TXT"    # Text
    NS    = "NS"     # Nameserver
    SOA   = "SOA"    # Start of authority
    PTR   = "PTR"    # Reverse lookup


@dataclass
class DNSRecord:
    name   : str
    rtype  : DNSRecordType
    value  : str
    ttl    : int = 300   # seconds
    priority: int = 0    # for MX records


@dataclass
class DNSCacheEntry:
    record     : DNSRecord
    cached_at  : float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.record.ttl


# ─────────────────────────────────────────────
# DNS CACHE
# ─────────────────────────────────────────────

class DNSCache:
    """LRU-like DNS cache with TTL expiry."""

    def __init__(self, name: str = "cache"):
        self.name    = name
        self._store  : Dict[str, DNSCacheEntry] = {}
        self.hits    = 0
        self.misses  = 0

    def _key(self, name: str, rtype: DNSRecordType) -> str:
        return f"{name}:{rtype.value}"

    def get(self, name: str, rtype: DNSRecordType) -> Optional[DNSRecord]:
        k     = self._key(name, rtype)
        entry = self._store.get(k)
        if entry is None or entry.is_expired:
            if entry and entry.is_expired:
                del self._store[k]
            self.misses += 1
            return None
        self.hits += 1
        return entry.record

    def set(self, record: DNSRecord):
        k = self._key(record.name, record.rtype)
        self._store[k] = DNSCacheEntry(record)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total else 0
        print(f"  Cache [{self.name}]: hits={self.hits}  misses={self.misses}  hit_rate={hit_rate:.0f}%")


# ─────────────────────────────────────────────
# DNS SERVERS
# ─────────────────────────────────────────────

class AuthoritativeDNS:
    """Holds the actual records for a zone."""

    def __init__(self, zone: str):
        self.zone    = zone
        self._records: Dict[str, DNSRecord] = {}

    def add_record(self, record: DNSRecord):
        key = f"{record.name}:{record.rtype.value}"
        self._records[key] = record

    def query(self, name: str, rtype: DNSRecordType) -> Optional[DNSRecord]:
        key = f"{name}:{rtype.value}"
        return self._records.get(key)

    def list_records(self):
        print(f"\n  Zone: {self.zone}")
        print(f"  {'Name':<35} {'Type':<8} {'TTL':<8} {'Value'}")
        print(f"  {'─'*70}")
        for r in self._records.values():
            print(f"  {r.name:<35} {r.rtype.value:<8} {r.ttl:<8} {r.value}")


class RecursiveResolver:
    """Resolves names by walking the DNS hierarchy: root → TLD → authoritative."""

    def __init__(self):
        self.cache         = DNSCache("resolver_cache")
        self._auth_servers : Dict[str, AuthoritativeDNS] = {}
        self.resolution_log: List[str] = []

    def register_authoritative(self, zone: str, server: AuthoritativeDNS):
        self._auth_servers[zone] = server

    def _log(self, msg: str):
        self.resolution_log.append(msg)
        print(f"  DNS: {msg}")

    def resolve(self, name: str, rtype: DNSRecordType = DNSRecordType.A) -> Optional[str]:
        # 1. Check cache
        cached = self.cache.get(name, rtype)
        if cached:
            self._log(f"[cache HIT] {name} → {cached.value}  (TTL remaining)")
            return cached.value

        # 2. Walk the hierarchy
        self._log(f"[cache MISS] {name} — querying hierarchy")
        self._log(f"  → root nameserver (finds .com TLD server)")
        self._log(f"  → .com TLD server (finds example.com authoritative)")

        # Find matching authoritative server
        parts   = name.split(".")
        zone    = ".".join(parts[-2:])   # e.g., "example.com"
        auth    = self._auth_servers.get(zone)
        if not auth:
            self._log(f"  ✗ No authoritative server for zone: {zone}")
            return None

        record = auth.query(name, rtype)
        if record:
            self._log(f"  → authoritative [{zone}] → {record.value}  (TTL={record.ttl}s)")
            self.cache.set(record)
            return record.value

        # CNAME follow
        cname = auth.query(name, DNSRecordType.CNAME)
        if cname:
            self._log(f"  → CNAME found: {name} → {cname.value}  (following…)")
            return self.resolve(cname.value, rtype)

        return None


# ─────────────────────────────────────────────
# GEO DNS
# ─────────────────────────────────────────────

@dataclass
class GeoDNSEntry:
    region    : str
    ip_address: str
    latency_to_regions: Dict[str, float] = field(default_factory=dict)  # region → ms


class GeoDNS:
    """Returns different IPs based on client geographic location."""

    def __init__(self, domain: str):
        self.domain  = domain
        self._entries: List[GeoDNSEntry] = []

    def add_region(self, entry: GeoDNSEntry):
        self._entries.append(entry)

    def resolve(self, client_region: str) -> Optional[str]:
        """Returns IP of the region with lowest latency for this client."""
        best      = None
        best_lat  = float("inf")
        for e in self._entries:
            lat = e.latency_to_regions.get(client_region, float("inf"))
            if lat < best_lat:
                best_lat = lat
                best     = e
        if best:
            print(f"  GeoDNS: client in [{client_region}] → {best.ip_address} "
                  f"(region={best.region}, latency={best_lat}ms)")
            return best.ip_address
        return None


class TTLManager:
    """Helps choose appropriate TTL for different use cases."""

    RECOMMENDATIONS = {
        "failover_critical": (30,   "Low TTL for fast failover (30s), but high DNS load"),
        "standard_api"     : (300,  "5 min TTL — balance between cache and propagation"),
        "stable_static"    : (3600, "1 hour — rarely changing content"),
        "cdn_edge"         : (60,   "1 min — CDN manages sub-second switching internally"),
        "mx_records"       : (3600, "Mail rarely changes; high TTL OK"),
        "before_migration" : (60,   "Lower TTL before planned IP change for fast propagation"),
    }

    @classmethod
    def recommend(cls, use_case: str) -> tuple:
        return cls.RECOMMENDATIONS.get(use_case, (300, "Default — 5 min"))

    @classmethod
    def print_guide(cls):
        print("\n  TTL RECOMMENDATION GUIDE:")
        print(f"  {'Use Case':<22} {'TTL':<8} {'Reason'}")
        print(f"  {'─'*70}")
        for uc, (ttl, reason) in cls.RECOMMENDATIONS.items():
            print(f"  {uc:<22} {ttl:<8}s  {reason}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_dns_and_how_it_works():
    print("=" * 65)
    print("DNS AND HOW IT WORKS")
    print("=" * 65)

    # ── Build Zone ────────────────────────────
    print("\n[1] AUTHORITATIVE DNS ZONE: example.com")
    print("─" * 50)
    auth = AuthoritativeDNS("example.com")
    auth.add_record(DNSRecord("example.com",        DNSRecordType.A,     "93.184.216.34",    ttl=3600))
    auth.add_record(DNSRecord("www.example.com",    DNSRecordType.CNAME, "example.com",      ttl=3600))
    auth.add_record(DNSRecord("api.example.com",    DNSRecordType.A,     "93.184.216.35",    ttl=300))
    auth.add_record(DNSRecord("cdn.example.com",    DNSRecordType.CNAME, "cdn.cloudfront.net",ttl=60))
    auth.add_record(DNSRecord("example.com",        DNSRecordType.MX,    "mail.example.com", ttl=3600, priority=10))
    auth.add_record(DNSRecord("example.com",        DNSRecordType.TXT,   "v=spf1 include:_spf.google.com ~all", ttl=3600))
    auth.add_record(DNSRecord("example.com",        DNSRecordType.NS,    "ns1.domaincontrol.com", ttl=86400))
    auth.list_records()

    # ── Recursive Resolution ──────────────────
    print("\n\n[2] DNS RESOLUTION WALK-THROUGH")
    print("─" * 50)
    resolver = RecursiveResolver()
    resolver.register_authoritative("example.com", auth)

    print("\n  Resolving api.example.com (first lookup — cache MISS):")
    ip = resolver.resolve("api.example.com", DNSRecordType.A)
    print(f"  Result: {ip}")

    print("\n  Resolving api.example.com (second lookup — cache HIT):")
    ip2 = resolver.resolve("api.example.com", DNSRecordType.A)
    print(f"  Result: {ip2}")

    print("\n  Resolving www.example.com (follows CNAME):")
    ip3 = resolver.resolve("www.example.com", DNSRecordType.A)
    print(f"  Result: {ip3}")

    resolver.cache.stats()

    # ── GeoDNS ────────────────────────────────
    print("\n\n[3] GEO DNS — ROUTING BY CLIENT REGION")
    print("─" * 50)
    geo = GeoDNS("api.example.com")
    geo.add_region(GeoDNSEntry("us-east-1",  "52.1.1.1",
                               latency_to_regions={"us": 5, "eu": 120, "asia": 200}))
    geo.add_region(GeoDNSEntry("eu-west-1",  "34.2.2.2",
                               latency_to_regions={"us": 110, "eu": 8, "asia": 180}))
    geo.add_region(GeoDNSEntry("ap-south-1", "13.3.3.3",
                               latency_to_regions={"us": 220, "eu": 190, "asia": 12}))

    for region in ["us", "eu", "asia"]:
        geo.resolve(region)

    # ── TTL Guide ─────────────────────────────
    print("\n\n[4] TTL GUIDE")
    print("─" * 50)
    TTLManager.print_guide()

    # ── DNS-based Load Balancing ──────────────
    print("\n\n[5] DNS LOAD BALANCING PATTERNS")
    print("─" * 50)
    patterns = [
        ("Round-Robin DNS",  "Multiple A records for same domain; resolver cycles through",
                             "Simple; no health checking; sticky per resolver TTL"),
        ("GeoDNS",           "Return different IP based on client geography",
                             "Low latency globally; requires geo-IP database"),
        ("Weighted DNS",     "Return IP A 80% of time, IP B 20% (canary)",
                             "Traffic splitting; requires DNS provider support"),
        ("Failover DNS",     "Primary IP normally; secondary IP on health-check failure",
                             "DR failover; requires low TTL + health checks"),
        ("Anycast DNS",      "Same IP announced from multiple PoPs via BGP; client routes to nearest",
                             "Cloudflare 1.1.1.1 / Google 8.8.8.8 use this"),
    ]
    for name, how, notes in patterns:
        print(f"\n  [{name}]")
        print(f"    How  : {how}")
        print(f"    Notes: {notes}")


if __name__ == "__main__":
    demonstrate_dns_and_how_it_works()
