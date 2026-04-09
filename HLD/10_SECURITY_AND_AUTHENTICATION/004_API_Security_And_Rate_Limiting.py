"""
API SECURITY AND RATE LIMITING
================================

Problem Statement:
Public APIs face abuse: brute-force attacks, credential stuffing,
scraping, DoS attacks. Rate limiting and security controls prevent abuse
without blocking legitimate users.

Rate Limiting Algorithms:
  Fixed Window:    N requests per window (e.g., 100 req/min).
                   Problem: burst at window boundary (2x rate briefly).
  Sliding Window:  Smooth window using timestamp log.
                   Memory: O(N) per user per window.
  Token Bucket:    Bucket fills at rate R. Each request consumes token.
                   Allows burst up to bucket capacity. Smooth long-term.
  Leaky Bucket:    Queue outgoing requests. Process at fixed rate.
                   Absorbs burst. Output is smooth (good for downstream).

Rate Limit Headers (RFC 6585 / IETF draft):
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 73
  X-RateLimit-Reset: 1704000000  (Unix timestamp when window resets)
  Retry-After: 30   (on 429 response)

API Key Security:
  Generate: CSPRNG 32+ bytes. Store hashed (SHA-256).
  Prefix: "sk_live_" for production, "sk_test_" for sandbox.
  Scopes: limit what each key can do (read-only, write, admin).
  Per-key rate limits: different limits for free vs paid tiers.

OWASP API Top 10 (2023):
  1. Broken Object Level Authorization (BOLA/IDOR)
  2. Broken Authentication
  3. Broken Object Property Level Authorization
  4. Unrestricted Resource Consumption (rate limiting)
  5. Broken Function Level Authorization
  6. Server-Side Request Forgery (SSRF)
  7. Security Misconfiguration
  8. Lack of Protection from Automated Threats
  9. Improper Inventory Management
  10. Unsafe Consumption of APIs

HMAC Request Signing:
  Prevents replay attacks and request tampering.
  Client: sign(method + path + body + timestamp + nonce, secret).
  Server: recompute signature. Reject if timestamp too old (>5min).
  Used by: AWS SigV4, Stripe webhook verification.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import hashlib
import hmac
import time
import uuid
import secrets
import threading


# ─────────────────────────────────────────────
# RATE LIMITERS
# ─────────────────────────────────────────────

class FixedWindowRateLimiter:
    """100 requests per minute. Simple but allows 2x burst at boundary."""

    def __init__(self, limit: int, window_s: float = 60):
        self.limit    = limit
        self.window_s = window_s
        self._windows : Dict[str, Tuple[int, float]] = {}   # key → (count, window_start)
        self._lock    = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        now = time.time()
        with self._lock:
            count, win_start = self._windows.get(key, (0, now))
            if now - win_start >= self.window_s:
                count, win_start = 0, now
            if count >= self.limit:
                reset_at = win_start + self.window_s
                return False, {"limit": self.limit, "remaining": 0,
                               "reset": reset_at, "retry_after": int(reset_at - now)}
            count += 1
            self._windows[key] = (count, win_start)
            return True, {"limit": self.limit, "remaining": self.limit - count,
                          "reset": win_start + self.window_s}


class SlidingWindowRateLimiter:
    """Sliding window log: precise, memory O(N requests)."""

    def __init__(self, limit: int, window_s: float = 60):
        self.limit    = limit
        self.window_s = window_s
        self._logs    : Dict[str, deque] = {}
        self._lock    = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, Dict]:
        now = time.time()
        cutoff = now - self.window_s
        with self._lock:
            if key not in self._logs:
                self._logs[key] = deque()
            log = self._logs[key]
            # Remove old entries
            while log and log[0] < cutoff:
                log.popleft()
            if len(log) >= self.limit:
                oldest    = log[0]
                retry     = oldest + self.window_s - now
                return False, {"limit": self.limit, "remaining": 0,
                               "retry_after": int(retry + 1)}
            log.append(now)
            return True, {"limit": self.limit, "remaining": self.limit - len(log)}


class TokenBucketRateLimiter:
    """Token bucket: smooth rate with burst capability."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity:    max burst size (tokens).
        refill_rate: tokens added per second.
        """
        self.capacity    = capacity
        self.refill_rate = refill_rate
        self._buckets    : Dict[str, Tuple[float, float]] = {}   # key → (tokens, last_refill)
        self._lock       = threading.Lock()

    def is_allowed(self, key: str, cost: int = 1) -> Tuple[bool, Dict]:
        now = time.time()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.capacity), now))
            # Refill
            elapsed = now - last
            tokens  = min(self.capacity, tokens + elapsed * self.refill_rate)
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return True, {"tokens_remaining": int(tokens), "capacity": self.capacity}
            wait_s = (cost - tokens) / self.refill_rate
            self._buckets[key] = (tokens, now)
            return False, {"tokens_remaining": int(tokens), "retry_after": wait_s}


# ─────────────────────────────────────────────
# API KEY MANAGEMENT
# ─────────────────────────────────────────────

@dataclass
class APIKey:
    key_id     : str
    prefix     : str
    key_hash   : str     # SHA-256 of the raw key
    owner_id   : str
    scopes     : List[str]
    tier       : str     # "free", "basic", "pro", "enterprise"
    created_at : float = field(default_factory=time.time)
    last_used  : Optional[float] = None
    active     : bool = True


class APIKeyManager:
    """
    Generates, stores (hashed), and validates API keys.
    Raw key shown only once at creation.
    """

    RATE_LIMITS = {
        "free"       : (60,   60),    # 60 req/min
        "basic"      : (300,  60),    # 300 req/min
        "pro"        : (1000, 60),    # 1000 req/min
        "enterprise" : (10000,60),    # 10000 req/min
    }

    def __init__(self):
        self._keys : Dict[str, APIKey] = {}   # key_id → APIKey
        self._rate_limiters: Dict[str, TokenBucketRateLimiter] = {}

    def create_key(self, owner_id: str, scopes: List[str],
                   tier: str = "free", env: str = "live") -> Tuple[str, APIKey]:
        """Returns (raw_key, APIKey). Raw key not stored — show once."""
        raw_key  = secrets.token_urlsafe(32)
        key_id   = f"{env}_{uuid.uuid4().hex[:8]}"
        prefix   = f"sk_{env}_{key_id[:8]}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key  = APIKey(key_id=key_id, prefix=prefix, key_hash=key_hash,
                           owner_id=owner_id, scopes=scopes, tier=tier)
        self._keys[key_id] = api_key

        limit, window = self.RATE_LIMITS[tier]
        self._rate_limiters[key_id] = TokenBucketRateLimiter(
            capacity=limit, refill_rate=limit/window
        )
        return f"{prefix}_{raw_key}", api_key

    def validate(self, raw_key: str) -> Tuple[Optional[APIKey], Optional[str]]:
        """Returns (APIKey, error) — timing-safe comparison."""
        try:
            parts    = raw_key.split("_", 3)
            key_id_part = f"{parts[0]}_{parts[1]}_{parts[2]}"  # env_live_xxxx
            actual_key = parts[3] if len(parts) > 3 else raw_key
        except IndexError:
            return None, "malformed_key"

        # Find matching key by hash (O(N) but necessary for security)
        raw_hash = hashlib.sha256(actual_key.encode()).hexdigest()
        for api_key in self._keys.values():
            if hmac.compare_digest(api_key.key_hash, raw_hash) and api_key.active:
                api_key.last_used = time.time()
                return api_key, None
        return None, "invalid_key"

    def check_rate_limit(self, key_id: str) -> Tuple[bool, Dict]:
        rl = self._rate_limiters.get(key_id)
        if not rl:
            return False, {"error": "no_rate_limiter"}
        return rl.is_allowed(key_id)

    def revoke(self, key_id: str):
        key = self._keys.get(key_id)
        if key:
            key.active = False


# ─────────────────────────────────────────────
# HMAC REQUEST SIGNING
# ─────────────────────────────────────────────

class HMACRequestSigner:
    """
    AWS SigV4 / Stripe-style HMAC request signing.
    Prevents replay attacks via timestamp + nonce.
    """

    def __init__(self, secret: str, max_age_s: float = 300):
        self._secret = secret.encode()
        self.max_age = max_age_s
        self._seen_nonces: Dict[str, float] = {}   # nonce → timestamp

    def sign(self, method: str, path: str, body: str) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        nonce     = secrets.token_hex(16)
        canonical = f"{method.upper()}\n{path}\n{body}\n{timestamp}\n{nonce}"
        signature = hmac.new(self._secret, canonical.encode(),
                              hashlib.sha256).hexdigest()
        return {
            "X-Timestamp": timestamp,
            "X-Nonce"    : nonce,
            "X-Signature": signature,
        }

    def verify(self, method: str, path: str, body: str,
               headers: Dict[str, str]) -> Tuple[bool, str]:
        timestamp = headers.get("X-Timestamp")
        nonce     = headers.get("X-Nonce")
        signature = headers.get("X-Signature")

        if not all([timestamp, nonce, signature]):
            return False, "missing_headers"

        # Replay prevention
        try:
            ts = int(timestamp)
        except ValueError:
            return False, "invalid_timestamp"
        if abs(time.time() - ts) > self.max_age:
            return False, "request_too_old"
        if nonce in self._seen_nonces:
            return False, "nonce_replay"

        canonical    = f"{method.upper()}\n{path}\n{body}\n{timestamp}\n{nonce}"
        expected_sig = hmac.new(self._secret, canonical.encode(),
                                 hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return False, "invalid_signature"

        self._seen_nonces[nonce] = time.time()
        self._cleanup_nonces()
        return True, "ok"

    def _cleanup_nonces(self):
        cutoff = time.time() - self.max_age * 2
        self._seen_nonces = {n: t for n, t in self._seen_nonces.items()
                              if t > cutoff}


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_api_security():
    print("=" * 65)
    print("API SECURITY AND RATE LIMITING")
    print("=" * 65)

    # ── Rate Limiter Comparison ───────────────────
    print("\n[1] RATE LIMITER ALGORITHMS")
    print("─" * 55)

    fixed   = FixedWindowRateLimiter(limit=5, window_s=60)
    sliding = SlidingWindowRateLimiter(limit=5, window_s=60)
    token   = TokenBucketRateLimiter(capacity=5, refill_rate=5/60)

    print("  Sending 7 requests (limit=5):")
    for i in range(7):
        fa, fh  = fixed.is_allowed("user1")
        sa, sh  = sliding.is_allowed("user1")
        ta, th  = token.is_allowed("user1")
        print(f"    Req {i+1}: Fixed={'OK' if fa else 'LIMIT'} "
              f"Sliding={'OK' if sa else 'LIMIT'} "
              f"Token={'OK' if ta else 'LIMIT'} "
              f"(remaining: {th.get('tokens_remaining','N/A')})")

    # ── Token Bucket Burst ────────────────────────
    print("\n\n[2] TOKEN BUCKET — BURST HANDLING")
    print("─" * 55)

    bucket = TokenBucketRateLimiter(capacity=10, refill_rate=2.0)  # 2 req/s
    print("  Burst of 10 requests (bucket capacity=10):")
    for i in range(12):
        ok, info = bucket.is_allowed("burster")
        status = "OK" if ok else f"BLOCKED (retry in {info.get('retry_after',0):.1f}s)"
        print(f"    Req {i+1:2d}: {status}  tokens={info.get('tokens_remaining','N/A')}")

    # ── API Key Management ────────────────────────
    print("\n\n[3] API KEY MANAGEMENT")
    print("─" * 55)

    manager = APIKeyManager()
    raw_key, key_obj = manager.create_key(
        "owner-123", ["read:data", "write:data"], tier="pro"
    )
    print(f"  Generated key: {raw_key[:25]}...  tier={key_obj.tier}")
    print(f"  Key ID: {key_obj.key_id}  scopes={key_obj.scopes}")

    # Extract the actual raw part after prefix
    parts   = raw_key.split("_", 3)
    raw_part = parts[3] if len(parts) > 3 else raw_key

    validated, err = manager.validate(raw_key)
    print(f"  Validate correct key: {'OK' if validated else err}")

    invalid, err2 = manager.validate("sk_live_badkey_wrongsecret")
    print(f"  Validate wrong key:   {err2}")

    ok, rl_info = manager.check_rate_limit(key_obj.key_id)
    print(f"  Rate limit check: allowed={ok} info={rl_info}")

    # ── HMAC Request Signing ──────────────────────
    print("\n\n[4] HMAC REQUEST SIGNING (replay protection)")
    print("─" * 55)

    signer = HMACRequestSigner(secret="webhook-secret-xyz", max_age_s=300)

    method  = "POST"
    path    = "/api/v1/payments"
    body    = '{"amount": 9999, "currency": "USD"}'

    headers = signer.sign(method, path, body)
    print(f"  Signed request headers:")
    for k, v in headers.items():
        print(f"    {k}: {v[:30]}...")

    valid, reason = signer.verify(method, path, body, headers)
    print(f"  Verify valid request: {valid} ({reason})")

    # Replay attack
    valid2, reason2 = signer.verify(method, path, body, headers)
    print(f"  Replay attack blocked: {not valid2} ({reason2})")

    # Tampered body
    tampered_headers = {**headers}
    valid3, reason3  = signer.verify(method, path, '{"amount": 1}', tampered_headers)
    print(f"  Tampered body blocked: {not valid3} ({reason3})")

    # ── OWASP API Top 10 ──────────────────────────
    print("\n\n[5] OWASP API TOP 10 — PREVENTION CHECKLIST")
    print("─" * 55)

    owasp = [
        ("BOLA/IDOR",         "Validate user owns resource: WHERE id=? AND user_id=?"),
        ("Broken Auth",       "JWT kid validation; no alg=none; short TTL"),
        ("Excess data",       "Return only needed fields; never expose full objects"),
        ("Rate limiting",     "Token bucket per API key; 429 with Retry-After"),
        ("Function-level",    "Check role on every endpoint, not just object"),
        ("SSRF",              "Allowlist outbound IPs; block 169.254.169.254 (metadata)"),
        ("Security misconfig","Disable debug endpoints; remove default creds"),
        ("Automated threats", "CAPTCHA; device fingerprint; bot detection"),
        ("API inventory",     "Document all endpoints; deprecate old versions"),
        ("Unsafe consumption","Validate all third-party API responses"),
    ]
    for vuln, prevention in owasp:
        print(f"  {vuln:<22} {prevention}")


if __name__ == "__main__":
    demonstrate_api_security()
