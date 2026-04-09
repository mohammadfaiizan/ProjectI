"""
MICROSERVICES SECURITY PATTERNS
==================================

Problem Statement:
In a microservices system, the attack surface is multiplied — every
service-to-service call is a potential entry point. Traditional
perimeter security ("trust everything inside the network") breaks down
when dozens of services communicate over internal networks.
Zero Trust: verify every call, regardless of where it originates.

Service-to-Service Authentication Patterns:

  1. Mutual TLS (mTLS):
     Both client AND server present certificates to each other.
     Cryptographically verifies identity of both parties.
     Works at: TCP level (transparent to application code with service mesh).
     Used by: Istio, Consul Connect, Linkerd.

  2. JWT Bearer Token (service identity):
     Service A gets a JWT from an auth server (client_credentials grant).
     Service A sends JWT in Authorization header to Service B.
     Service B validates JWT signature and checks issuer/audience.
     Stateless; scales well; works across network boundaries.

  3. API Key:
     Simpler; suitable for gateway-to-backend or 3rd party.
     No expiry built-in; revocation requires key rotation.
     Not suitable for internal service-to-service.

  4. SPIFFE/SVID (Secure Production Identity Framework):
     Cryptographic identity document issued to each service workload.
     SPIFFE ID: spiffe://trust-domain/service-name
     SVID: X.509 certificate or JWT signed by SPIFFE authority.
     Rotated automatically; no hardcoded secrets.

Zero Trust Principles:
  - Never trust; always verify — even inside the cluster.
  - Least privilege — each service has minimum required permissions.
  - Every call authenticated AND authorized.
  - Short-lived credentials — rotate/expire tokens frequently.
  - Audit all access.

User JWT Token Forwarding:
  User authenticates at edge (API gateway).
  Gateway validates token; extracts user_id, scopes.
  Passes user JWT (or a new service-signed token) to downstream services.
  Downstream services can authorize based on user identity.
  Pattern: gateway issues a new "internal" JWT that includes user claims
  but is signed by the internal auth service (token exchange).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time
import uuid
import hashlib
import hmac
import base64
import json


# ─────────────────────────────────────────────
# JWT UTILITIES (simplified — no crypto library)
# ─────────────────────────────────────────────

SECRET_KEY = "super-secret-internal-signing-key-do-not-expose"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_jwt(claims: Dict, secret: str = SECRET_KEY,
               ttl_seconds: int = 3600) -> str:
    """Create a simplified JWT (header.payload.signature)."""
    header  = {"alg": "HS256", "typ": "JWT"}
    payload = {**claims, "iat": int(time.time()),
               "exp": int(time.time()) + ttl_seconds,
               "jti": str(uuid.uuid4())[:8]}
    h_enc = _b64url_encode(json.dumps(header).encode())
    p_enc = _b64url_encode(json.dumps(payload).encode())
    signing_input = f"{h_enc}.{p_enc}"
    sig = hmac.new(secret.encode(), signing_input.encode(),
                   hashlib.sha256).digest()
    s_enc = _b64url_encode(sig)
    return f"{h_enc}.{p_enc}.{s_enc}"


def verify_jwt(token: str, secret: str = SECRET_KEY,
               required_audience: Optional[str] = None) -> Dict:
    """Verify JWT; return claims dict or raise on invalid."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid token format")
    h_enc, p_enc, s_enc = parts
    signing_input = f"{h_enc}.{p_enc}"
    expected_sig  = hmac.new(secret.encode(), signing_input.encode(),
                             hashlib.sha256).digest()
    actual_sig    = _b64url_decode(s_enc)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(p_enc))
    now     = int(time.time())
    if payload.get("exp", 0) < now:
        raise ValueError("Token expired")
    if required_audience and payload.get("aud") != required_audience:
        raise ValueError(f"Wrong audience: expected {required_audience}")
    return payload


# ─────────────────────────────────────────────
# mTLS STUB (certificate-based identity)
# ─────────────────────────────────────────────

@dataclass
class Certificate:
    """Simulated X.509 certificate for mTLS."""
    common_name  : str           # service identity: spiffe://cluster/service-name
    issuer       : str           # CA that signed this cert
    valid_until  : float         # unix timestamp
    fingerprint  : str = field(default="")

    def __post_init__(self):
        if not self.fingerprint:
            raw = f"{self.common_name}:{self.issuer}:{self.valid_until}"
            self.fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def is_valid(self) -> bool:
        return time.time() < self.valid_until

    @property
    def spiffe_id(self) -> str:
        return self.common_name


class CertificateAuthority:
    """Internal CA that issues and verifies service certificates."""

    def __init__(self, name: str):
        self.name     = name
        self._issued  : Dict[str, Certificate] = {}

    def issue(self, service_name: str, ttl_seconds: int = 86400) -> Certificate:
        spiffe_id = f"spiffe://cluster.local/{service_name}"
        cert = Certificate(
            common_name = spiffe_id,
            issuer      = self.name,
            valid_until = time.time() + ttl_seconds,
        )
        self._issued[cert.fingerprint] = cert
        return cert

    def verify(self, cert: Certificate) -> bool:
        """Check cert was issued by this CA and is not expired."""
        return (cert.issuer == self.name and
                cert.is_valid and
                cert.fingerprint in self._issued)


class MtlsChannel:
    """Simulates a mutual TLS handshake between two services."""

    def __init__(self, ca: CertificateAuthority):
        self.ca = ca

    def connect(self, client_cert: Certificate,
                server_cert: Certificate) -> Dict:
        """Returns connection info or raises on failure."""
        client_ok = self.ca.verify(client_cert)
        server_ok = self.ca.verify(server_cert)

        if not client_ok:
            raise PermissionError("mTLS: client certificate rejected by CA")
        if not server_ok:
            raise PermissionError("mTLS: server certificate rejected by CA")

        return {
            "status"     : "established",
            "client"     : client_cert.spiffe_id,
            "server"     : server_cert.spiffe_id,
            "protocol"   : "TLSv1.3",
            "cipher"     : "TLS_AES_256_GCM_SHA384",
        }


# ─────────────────────────────────────────────
# AUTH SERVER (issues service JWTs)
# ─────────────────────────────────────────────

@dataclass
class ServiceCredential:
    client_id     : str
    client_secret : str
    allowed_scopes: Set[str]


class AuthServer:
    """
    Issues JWTs via client_credentials grant (service-to-service auth).
    Also handles user JWT forwarding (token exchange).
    """

    def __init__(self, issuer: str, secret: str = SECRET_KEY):
        self.issuer   = issuer
        self._secret  = secret
        self._clients : Dict[str, ServiceCredential] = {}

    def register_client(self, cred: ServiceCredential):
        self._clients[cred.client_id] = cred

    def client_credentials(self, client_id: str, client_secret: str,
                           scopes: List[str]) -> str:
        """OAuth2 client_credentials grant — service-to-service."""
        cred = self._clients.get(client_id)
        if not cred or cred.client_secret != client_secret:
            raise PermissionError("Invalid client credentials")
        granted = set(scopes) & cred.allowed_scopes
        if not granted:
            raise PermissionError(f"No scopes granted. Requested: {scopes}")
        return create_jwt({
            "sub" : client_id,
            "iss" : self.issuer,
            "aud" : "internal-api",
            "scp" : list(granted),
            "type": "service",
        }, self._secret, ttl_seconds=600)

    def token_exchange(self, user_token: str, target_service: str) -> str:
        """
        Exchange a user JWT for an internal service-signed token
        that includes user claims. Used for user context propagation.
        """
        claims = verify_jwt(user_token, self._secret)
        return create_jwt({
            "sub"    : claims.get("sub"),
            "iss"    : self.issuer,
            "aud"    : target_service,
            "user_id": claims.get("sub"),
            "scopes" : claims.get("scp", []),
            "type"   : "forwarded",
            "orig_jti": claims.get("jti"),
        }, self._secret, ttl_seconds=300)


# ─────────────────────────────────────────────
# SERVICE AUTH MIDDLEWARE
# ─────────────────────────────────────────────

class ServiceAuthMiddleware:
    """Validates incoming JWT on a service; enforces scopes."""

    def __init__(self, service_name: str, auth_server_secret: str = SECRET_KEY):
        self._service = service_name
        self._secret  = auth_server_secret
        self._audit   : List[Dict] = []

    def authenticate(self, token: str,
                     required_scope: Optional[str] = None) -> Dict:
        try:
            claims = verify_jwt(token, self._secret)
        except ValueError as e:
            self._audit.append({"service": self._service, "result": "DENIED",
                                 "reason": str(e)})
            raise PermissionError(f"Auth failed: {e}")

        if required_scope:
            scopes = claims.get("scp", [])
            if required_scope not in scopes:
                self._audit.append({"service": self._service, "result": "DENIED",
                                     "reason": f"missing scope: {required_scope}"})
                raise PermissionError(f"Insufficient scope: need '{required_scope}'")

        self._audit.append({"service": self._service, "result": "ALLOWED",
                             "caller": claims.get("sub")})
        return claims


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_microservices_security():
    print("=" * 65)
    print("MICROSERVICES SECURITY PATTERNS")
    print("=" * 65)

    auth_server = AuthServer("https://auth.internal.cluster")
    auth_server.register_client(ServiceCredential(
        "order-service", "secret-ord-123", {"orders:read", "orders:write", "inventory:read"}))
    auth_server.register_client(ServiceCredential(
        "payment-service", "secret-pay-456", {"payments:write", "orders:read"}))
    auth_server.register_client(ServiceCredential(
        "reporting-service", "secret-rep-789", {"orders:read"}))

    # ── 1. Service-to-service JWT ─────────────────
    print("\n[1] SERVICE JWT — CLIENT_CREDENTIALS GRANT")
    print("─" * 55)

    token = auth_server.client_credentials(
        "order-service", "secret-ord-123",
        ["orders:write", "inventory:read"])
    claims = verify_jwt(token)
    print(f"  Issued JWT to order-service:")
    print(f"    sub={claims['sub']}  iss={claims['iss']}")
    print(f"    scopes={claims['scp']}  type={claims['type']}")
    print(f"    expires_in={claims['exp'] - int(time.time())}s")

    # ── 2. JWT validation at receiving service ────
    print("\n\n[2] RECEIVING SERVICE VALIDATES JWT")
    print("─" * 55)
    inventory_auth = ServiceAuthMiddleware("inventory-service")

    print("  order-service calls inventory with valid token + correct scope:")
    try:
        c = inventory_auth.authenticate(token, required_scope="inventory:read")
        print(f"    ALLOWED — caller={c['sub']} scope=inventory:read")
    except PermissionError as e:
        print(f"    DENIED — {e}")

    print("  order-service calls payments (no payment scope in its token):")
    payment_auth = ServiceAuthMiddleware("payment-service")
    try:
        c = payment_auth.authenticate(token, required_scope="payments:write")
        print(f"    ALLOWED — {c['sub']}")
    except PermissionError as e:
        print(f"    DENIED — {e}")

    print("  Sending expired/invalid token:")
    try:
        bad_token = create_jwt({"sub": "evil-service"}, ttl_seconds=-10)
        inventory_auth.authenticate(bad_token)
    except PermissionError as e:
        print(f"    DENIED — {e}")

    # ── 3. mTLS simulation ────────────────────────
    print("\n\n[3] MUTUAL TLS — SERVICE IDENTITY VIA CERTIFICATES")
    print("─" * 55)

    ca      = CertificateAuthority("internal-ca.cluster")
    channel = MtlsChannel(ca)

    order_cert   = ca.issue("order-service",   ttl_seconds=86400)
    payment_cert = ca.issue("payment-service", ttl_seconds=86400)

    print("  Establishing mTLS channel (both sides present cert):")
    try:
        conn = channel.connect(order_cert, payment_cert)
        print(f"    Status:   {conn['status']}")
        print(f"    Client:   {conn['client']}")
        print(f"    Server:   {conn['server']}")
        print(f"    Protocol: {conn['protocol']}")
    except PermissionError as e:
        print(f"    FAILED: {e}")

    print("\n  Unknown service (no CA-issued cert) attempts to connect:")
    fake_cert = Certificate("spiffe://cluster.local/evil-service",
                            "rogue-ca", time.time() + 3600)
    try:
        channel.connect(fake_cert, payment_cert)
    except PermissionError as e:
        print(f"    REJECTED: {e}")

    # ── 4. User JWT forwarding ────────────────────
    print("\n\n[4] USER JWT TOKEN FORWARDING (TOKEN EXCHANGE)")
    print("─" * 55)

    user_token = create_jwt({
        "sub"  : "user-alice",
        "iss"  : "https://auth.example.com",
        "aud"  : "api-gateway",
        "scp"  : ["orders:read", "profile:read"],
        "email": "alice@example.com",
    })
    print(f"  User token received at API gateway for user-alice")

    forwarded_token = auth_server.token_exchange(user_token, "order-service")
    fwd_claims      = verify_jwt(forwarded_token)
    print(f"  Exchanged internal token for order-service:")
    print(f"    sub={fwd_claims['sub']}  aud={fwd_claims['aud']}")
    print(f"    type={fwd_claims['type']}  user_id={fwd_claims['user_id']}")
    print(f"  → Downstream services know the calling user without re-authenticating.")

    # ── 5. SPIFFE/SVID concept ────────────────────
    print("\n\n[5] SPIFFE/SVID — CRYPTOGRAPHIC SERVICE IDENTITY")
    print("─" * 55)
    services = ["order-service", "payment-service", "inventory-service"]
    print(f"  SPIFFE IDs issued by internal CA:")
    for svc in services:
        cert = ca.issue(svc)
        print(f"    {cert.spiffe_id:<50} fp={cert.fingerprint}")
    print(f"\n  → Certs auto-rotate; no hardcoded secrets; verified by CA.")

    # ── 6. Zero trust summary ─────────────────────
    print("\n\n[6] ZERO TRUST CHECKLIST")
    print("─" * 55)
    checks = [
        ("Verify every call",         "JWT/mTLS on every service-to-service request"),
        ("Least privilege",           "Token scopes match exactly what service needs"),
        ("Short-lived credentials",   "JWTs expire in minutes; certs rotated daily"),
        ("No network perimeter trust","Internal network calls are not trusted by default"),
        ("Audit everything",          "Every auth decision logged with caller + result"),
        ("Mutual authentication",     "Both sides prove identity (mTLS or JWT exchange)"),
        ("Context propagation",       "User identity flows through service chain"),
    ]
    for check, impl in checks:
        print(f"  {check:<28} {impl}")

    # ── 7. Auth log ───────────────────────────────
    print("\n\n[7] AUTH AUDIT LOG (inventory-service)")
    print("─" * 55)
    for entry in inventory_auth._audit:
        print(f"  result={entry['result']:<8} "
              f"caller={entry.get('caller','—'):<20} "
              f"reason={entry.get('reason','ok')}")


if __name__ == "__main__":
    demonstrate_microservices_security()
