"""
SSL TERMINATION AND TLS OFFLOADING
=====================================

Problem Statement:
TLS handshakes are CPU-intensive (asymmetric cryptography). If every backend
server handles TLS, you waste CPU on encryption instead of business logic.
SSL termination at the load balancer offloads this work centrally and
simplifies certificate management.

SSL Termination Approaches:
  1. Terminate at LB / Reverse Proxy (most common)
     Client → [TLS] → LB → [HTTP] → Backend
     + Centralized cert management
     + Backend gets plain HTTP (faster)
     - Internal traffic unencrypted (ok in trusted private network)

  2. TLS Pass-through (end-to-end encryption)
     Client → [TLS] → LB (forwards TCP) → [TLS] → Backend
     + End-to-end encryption
     - LB can't inspect/route on HTTP headers
     - Each backend manages its own certs

  3. TLS Re-encryption (mTLS)
     Client → [TLS] → LB → [new TLS] → Backend
     + LB terminates & inspects; re-encrypts to backend (mTLS)
     + Compliance requirement (PCI DSS, HIPAA)
     - More CPU overhead

TLS Handshake Steps (TLS 1.3):
  1. ClientHello   (ciphers, random)
  2. ServerHello   (chosen cipher, cert, random)
  3. Certificate Verify
  4. Finished      (0-RTT or 1-RTT)
  → Symmetric key established; AES-GCM encryption begins
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import hashlib
import random


class TLSMode(Enum):
    TERMINATE       = "terminate"        # LB terminates TLS
    PASSTHROUGH     = "passthrough"      # LB forwards raw TCP
    RE_ENCRYPT      = "re_encrypt"       # LB terminates + new TLS to backend


class TLSVersion(Enum):
    TLS_1_0 = "TLSv1.0"
    TLS_1_1 = "TLSv1.1"
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class CipherSuite(Enum):
    TLS_AES_256_GCM_SHA384       = "TLS_AES_256_GCM_SHA384"        # TLS 1.3 only
    TLS_CHACHA20_POLY1305_SHA256 = "TLS_CHACHA20_POLY1305_SHA256"  # TLS 1.3 only
    ECDHE_RSA_AES256_GCM_SHA384  = "ECDHE-RSA-AES256-GCM-SHA384"  # TLS 1.2
    RC4_MD5                      = "RC4-MD5"                        # WEAK


@dataclass
class Certificate:
    domain      : str
    issuer      : str
    valid_days  : int
    key_size    : int    # RSA bits (2048/4096) or ECDSA curve
    san_domains : List[str] = field(default_factory=list)
    issued_at   : float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.issued_at > self.valid_days * 86400

    @property
    def days_remaining(self) -> int:
        elapsed = (time.time() - self.issued_at) / 86400
        return max(0, self.valid_days - int(elapsed))

    @property
    def auto_renew_needed(self) -> bool:
        return self.days_remaining < 30


@dataclass
class TLSHandshake:
    client_ip      : str
    tls_version    : TLSVersion
    cipher_suite   : CipherSuite
    session_reused : bool = False
    cpu_ms         : float = 0.0    # CPU time for handshake
    rtt_ms         : float = 0.0

    @property
    def total_ms(self) -> float:
        return self.cpu_ms + self.rtt_ms


# ─────────────────────────────────────────────
# TLS SESSION CACHE
# ─────────────────────────────────────────────

class TLSSessionCache:
    """
    Cache TLS session tickets/IDs to avoid full handshake on reconnect.
    Session resumption reduces CPU by ~10x.
    """

    def __init__(self, ttl_s: int = 3600):
        self.ttl_s    = ttl_s
        self._cache   : Dict[str, Dict] = {}
        self.hits     = 0
        self.misses   = 0

    def _key(self, client_ip: str, server_name: str) -> str:
        return hashlib.md5(f"{client_ip}:{server_name}".encode()).hexdigest()

    def get(self, client_ip: str, sni: str) -> bool:
        key   = self._key(client_ip, sni)
        entry = self._cache.get(key)
        if entry and time.time() - entry["created_at"] < self.ttl_s:
            self.hits += 1
            return True
        self.misses += 1
        return False

    def set(self, client_ip: str, sni: str):
        key = self._key(client_ip, sni)
        self._cache[key] = {"created_at": time.time()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ─────────────────────────────────────────────
# SSL TERMINATOR
# ─────────────────────────────────────────────

class SSLTerminator:
    """
    Handles TLS at the edge. Presents certificate to clients,
    forwards plain HTTP to backend pool.
    """

    # Simulated CPU cost per handshake in ms
    CPU_COST = {
        (TLSVersion.TLS_1_3, True):  1.0,   # session resumed
        (TLSVersion.TLS_1_3, False): 8.0,   # full RSA-2048 handshake
        (TLSVersion.TLS_1_2, False): 12.0,
        (TLSVersion.TLS_1_2, True):  1.5,
    }

    def __init__(self, cert: Certificate, mode: TLSMode = TLSMode.TERMINATE):
        self.cert         = cert
        self.mode         = mode
        self.session_cache = TLSSessionCache()
        self.handshakes   : List[TLSHandshake] = []
        self.version_stats: Dict[str, int] = {}

    def handle_client(self, client_ip: str, sni: str,
                       preferred_version: TLSVersion = TLSVersion.TLS_1_3) -> TLSHandshake:
        # Check session cache
        resumed = self.session_cache.get(client_ip, sni)
        if not resumed:
            self.session_cache.set(client_ip, sni)

        # Negotiate version (min TLS 1.2)
        version = preferred_version
        if version in (TLSVersion.TLS_1_0, TLSVersion.TLS_1_1):
            version = TLSVersion.TLS_1_2   # downgrade rejected

        # Select cipher
        if version == TLSVersion.TLS_1_3:
            cipher = CipherSuite.TLS_AES_256_GCM_SHA384
        else:
            cipher = CipherSuite.ECDHE_RSA_AES256_GCM_SHA384

        cpu_ms = self.CPU_COST.get((version, resumed), 10.0) + random.uniform(0, 2)
        rtt_ms = 30.0 + random.uniform(-5, 15)   # simulated round-trip

        hs = TLSHandshake(client_ip, version, cipher, resumed, round(cpu_ms, 2), round(rtt_ms, 2))
        self.handshakes.append(hs)
        self.version_stats[version.value] = self.version_stats.get(version.value, 0) + 1
        return hs

    def report(self):
        total = len(self.handshakes)
        avg_cpu = sum(h.cpu_ms for h in self.handshakes) / max(1, total)
        resumed = sum(1 for h in self.handshakes if h.session_reused)
        print(f"\n  SSLTerminator Report:")
        print(f"    Total handshakes  : {total}")
        print(f"    Session resumed   : {resumed} ({resumed/max(1,total):.0%})")
        print(f"    Avg CPU per hs    : {avg_cpu:.1f}ms")
        print(f"    Session cache     : hits={self.session_cache.hits}  "
              f"misses={self.session_cache.misses}  "
              f"hit_rate={self.session_cache.hit_rate:.0%}")
        print(f"    Version breakdown : {self.version_stats}")
        if self.cert.auto_renew_needed:
            print(f"    ⚠  Certificate expires in {self.cert.days_remaining} days — renew!")
        else:
            print(f"    ✅ Certificate valid for {self.cert.days_remaining} days")


# ─────────────────────────────────────────────
# CERTIFICATE MANAGER
# ─────────────────────────────────────────────

class CertificateManager:
    """
    Manages certificates across multiple domains.
    Alerts on expiry and auto-renews via ACME (Let's Encrypt).
    """

    def __init__(self):
        self._certs: Dict[str, Certificate] = {}

    def add(self, cert: Certificate):
        self._certs[cert.domain] = cert
        for san in cert.san_domains:
            self._certs[san] = cert

    def get(self, domain: str) -> Optional[Certificate]:
        return self._certs.get(domain)

    def audit(self):
        print(f"\n  Certificate Audit:")
        print(f"  {'Domain':<35} {'Issuer':<20} {'Days Left':<12} {'Status'}")
        print(f"  {'─'*80}")
        seen = set()
        for domain, cert in self._certs.items():
            if cert.domain in seen:
                continue
            seen.add(cert.domain)
            status = ("🔴 EXPIRED" if cert.is_expired else
                      "⚠  RENEW SOON" if cert.auto_renew_needed else
                      "✅ OK")
            print(f"  {cert.domain:<35} {cert.issuer:<20} {cert.days_remaining:<12} {status}")


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_ssl_termination():
    print("=" * 65)
    print("SSL TERMINATION AND TLS OFFLOADING")
    print("=" * 65)

    # ── TLS Handshake Simulation ──────────────
    print("\n[1] TLS HANDSHAKE WALKTHROUGH (TLS 1.3)")
    print("─" * 55)
    steps = [
        ("1→", "ClientHello",    "TLS 1.3, cipher list, random nonce, SNI=api.example.com"),
        ("2←", "ServerHello",    "Chosen: TLS_AES_256_GCM_SHA384, server random"),
        ("3←", "Certificate",    "x.509 cert, public key (RSA 2048 / ECDSA P-256)"),
        ("4←", "CertVerify",     "Signature proves server owns private key"),
        ("5←", "Finished",       "Server auth complete — 1 RTT total"),
        ("6→", "Finished",       "Client confirms — symmetric key derived"),
        ("7→", "AppData",        "HTTP/1.1 or HTTP/2 over encrypted channel"),
    ]
    print(f"  {'Dir':<4} {'Message':<16} Details")
    print(f"  {'─'*65}")
    for direction, msg, detail in steps:
        print(f"  {direction:<4} {msg:<16} {detail}")
    print("\n  TLS 1.3: 1-RTT (vs TLS 1.2: 2-RTT) — 50% faster handshake")

    # ── SSL Terminator ────────────────────────
    print("\n\n[2] SSL TERMINATION AT LOAD BALANCER")
    print("─" * 55)
    cert = Certificate(
        domain="api.example.com",
        issuer="Let's Encrypt",
        valid_days=90,
        key_size=256,   # ECDSA P-256
        san_domains=["www.api.example.com", "*.api.example.com"]
    )
    terminator = SSLTerminator(cert, TLSMode.TERMINATE)

    # Simulate client connections
    clients = [
        ("1.2.3.1", "api.example.com",  TLSVersion.TLS_1_3),
        ("1.2.3.2", "api.example.com",  TLSVersion.TLS_1_3),
        ("1.2.3.1", "api.example.com",  TLSVersion.TLS_1_3),  # resumed
        ("1.2.3.3", "api.example.com",  TLSVersion.TLS_1_1),  # downgraded to 1.2
        ("1.2.3.4", "api.example.com",  TLSVersion.TLS_1_2),
        ("1.2.3.1", "api.example.com",  TLSVersion.TLS_1_3),  # resumed again
    ]
    print(f"  {'Client':<12} {'SNI':<25} {'Version':<10} {'Resumed':<10} {'CPU ms'}")
    print(f"  {'─'*65}")
    for ip, sni, ver in clients:
        hs = terminator.handle_client(ip, sni, ver)
        print(f"  {ip:<12} {sni:<25} {hs.tls_version.value:<10} "
              f"{'Yes' if hs.session_reused else 'No':<10} {hs.cpu_ms:.1f}ms")

    terminator.report()

    # ── Certificate Manager ───────────────────
    print("\n\n[3] CERTIFICATE MANAGEMENT")
    print("─" * 55)
    cm = CertificateManager()
    cm.add(Certificate("api.example.com",    "Let's Encrypt", 90,  256, ["*.api.example.com"]))
    cm.add(Certificate("app.example.com",    "DigiCert",      365, 256))
    cm.add(Certificate("legacy.example.com", "GlobalSign",    365, 2048))
    # Simulate about-to-expire cert
    old_cert = Certificate("old.example.com", "Comodo", 90, 2048)
    old_cert.issued_at = time.time() - (70 * 86400)   # 70 days old (20 remaining)
    cm.add(old_cert)
    cm.audit()

    # ── Termination Modes ─────────────────────
    print("\n\n[4] TLS TERMINATION MODES")
    print("─" * 55)
    rows = [
        ("Terminate at LB",   "Client-TLS→LB→HTTP→Backend",  "Simple cert mgmt, fast backend", "Internal traffic unencrypted"),
        ("TLS Pass-through",  "Client-TLS→LB→TLS→Backend",   "End-to-end encryption",          "LB can't inspect HTTP headers"),
        ("Re-encrypt (mTLS)", "Client-TLS→LB→new-TLS→Backend","Full inspection + encrypt",      "More CPU; complex cert setup"),
    ]
    for mode, flow, pros, cons in rows:
        print(f"\n  {mode}:")
        print(f"    Flow  : {flow}")
        print(f"    Pro   : {pros}")
        print(f"    Con   : {cons}")

    # ── Performance Numbers ───────────────────
    print("\n\n[5] TLS PERFORMANCE COMPARISON")
    print("─" * 55)
    perf = [
        ("TLS 1.3 new",       "8-15ms", "1 RTT",  "Fast — uses early data"),
        ("TLS 1.3 resumed",   "1-3ms",  "0 RTT",  "Session ticket reuse"),
        ("TLS 1.2 new",       "15-30ms","2 RTT",  "Older protocol"),
        ("TLS 1.2 resumed",   "2-5ms",  "1 RTT",  "Session ID / ticket"),
        ("SSL offloading",    "—",      "—",       "Dedicated SSL accelerator HW"),
    ]
    print(f"  {'Scenario':<25} {'CPU/hs':<10} {'RTT':<8} Notes")
    print(f"  {'─'*60}")
    for scenario, cpu, rtt, notes in perf:
        print(f"  {scenario:<25} {cpu:<10} {rtt:<8} {notes}")

    print("\n  OCSP Stapling: LB caches cert revocation response → saves RTT")
    print("  HSTS: HTTP Strict Transport Security → browser enforces HTTPS")
    print("  Certificate Transparency: public ledger of all issued certs")


if __name__ == "__main__":
    demonstrate_ssl_termination()
