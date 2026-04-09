"""
OAUTH2 AND OPENID CONNECT (OIDC)
===================================

Problem Statement:
Users want to log into your app using Google/GitHub without sharing credentials.
Apps need to access third-party APIs on behalf of users.
OAuth2 solves delegated authorization; OIDC adds authentication on top.

OAuth2 (RFC 6749):
  Framework for delegated authorization.
  "User grants app limited access to their resources on another service."
  Roles:
    Resource Owner: the user.
    Client:         the app requesting access.
    Authorization Server: issues tokens (Google, GitHub, Okta).
    Resource Server: API being accessed (Gmail API, GitHub API).

Grant Types:
  Authorization Code: for server-side apps. Most secure.
                      Code exchanged server-side (not exposed in browser).
  Authorization Code + PKCE: for SPAs and mobile. No client secret.
                      PKCE (Proof Key for Code Exchange) prevents CSRF.
  Client Credentials: machine-to-machine. No user involved.
                      App authenticates with its own credentials.
  Implicit:           Deprecated. Access token in URL fragment (insecure).
  Resource Owner Password: Deprecated. App collects user's password directly.

Token Types:
  Access Token:  Short-lived (15min-1h). Bearer token for API calls.
  Refresh Token: Long-lived. Exchanged for new access tokens.
  ID Token (OIDC): JWT with user identity claims. For authentication.

OIDC (OpenID Connect):
  Layer on top of OAuth2 for authentication (not just authorization).
  Adds: ID Token (JWT with user claims: sub, email, name).
  Endpoints: /authorize, /token, /userinfo, /.well-known/openid-configuration.
  Scopes: openid (required), profile, email, address, phone.

PKCE (Proof Key for Code Exchange):
  Client generates code_verifier (random 43-128 byte string).
  code_challenge = BASE64URL(SHA256(code_verifier)).
  Authorization request includes code_challenge.
  Token request includes code_verifier — server verifies.
  Prevents authorization code interception by malicious apps.

JWT ID Token Claims:
  iss: Issuer URL.
  sub: Subject (unique user ID at provider).
  aud: Audience (your client_id).
  exp: Expiration timestamp.
  iat: Issued-at timestamp.
  nonce: Random value to prevent replay.
  email, name, picture: profile claims.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import hmac
import time
import uuid
import json
import base64
import secrets


# ─────────────────────────────────────────────
# PKCE HELPER
# ─────────────────────────────────────────────

class PKCE:
    """Proof Key for Code Exchange helpers."""

    @staticmethod
    def generate_verifier() -> str:
        """Random 43-128 char URL-safe string."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()

    @staticmethod
    def generate_challenge(verifier: str) -> str:
        """S256 method: BASE64URL(SHA256(verifier))."""
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    @staticmethod
    def verify(verifier: str, challenge: str) -> bool:
        return PKCE.generate_challenge(verifier) == challenge


# ─────────────────────────────────────────────
# JWT SERVICE
# ─────────────────────────────────────────────

class JWTService:
    def __init__(self, secret: str, issuer: str = "https://auth.example.com"):
        self._secret = secret.encode()
        self.issuer  = issuer

    def _b64url(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def sign(self, payload: Dict, expires_in_s: int = 3600) -> str:
        payload.update({"iss": self.issuer, "iat": int(time.time()),
                         "exp": int(time.time()) + expires_in_s})
        header = self._b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body   = self._b64url(json.dumps(payload).encode())
        sig    = self._b64url(
            hmac.new(self._secret, f"{header}.{body}".encode(), hashlib.sha256).digest()
        )
        return f"{header}.{body}.{sig}"

    def verify(self, token: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            h, b, sig = token.split(".")
            expected = self._b64url(
                hmac.new(self._secret, f"{h}.{b}".encode(), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(sig, expected):
                return None, "invalid_signature"
            payload = json.loads(base64.urlsafe_b64decode(b + "=="))
            if payload.get("exp", 0) < time.time():
                return None, "token_expired"
            if payload.get("iss") != self.issuer:
                return None, "invalid_issuer"
            return payload, None
        except Exception as e:
            return None, str(e)


# ─────────────────────────────────────────────
# USER STORE
# ─────────────────────────────────────────────

@dataclass
class OIDCUser:
    sub      : str         # unique stable ID
    email    : str
    name     : str
    picture  : str = ""
    email_verified: bool = True


# ─────────────────────────────────────────────
# OAUTH2 / OIDC AUTHORIZATION SERVER
# ─────────────────────────────────────────────

@dataclass
class ClientRegistration:
    client_id     : str
    client_secret : str
    redirect_uris : List[str]
    scopes        : Set[str]
    grant_types   : Set[str]
    is_confidential: bool = True   # False for public clients (SPA, mobile)


class OAuth2OIDCServer:
    """
    OAuth2 + OIDC Authorization Server supporting:
    - Authorization Code flow (with and without PKCE).
    - Client Credentials flow.
    - ID Token issuance (OIDC).
    - Token introspection and revocation.
    """

    def __init__(self, jwt_service: JWTService):
        self._jwt            = jwt_service
        self._clients        : Dict[str, ClientRegistration] = {}
        self._users          : Dict[str, OIDCUser] = {}
        self._auth_codes     : Dict[str, Dict] = {}
        self._access_tokens  : Dict[str, Dict] = {}
        self._refresh_tokens : Dict[str, Dict] = {}

    def register_client(self, client: ClientRegistration):
        self._clients[client.client_id] = client

    def register_user(self, user: OIDCUser):
        self._users[user.sub] = user

    # ── Authorization Code + PKCE ─────────────────

    def authorize(self, client_id: str, redirect_uri: str,
                  scope: str, user_sub: str,
                  code_challenge: str = None,
                  code_challenge_method: str = "S256",
                  nonce: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Returns (auth_code, error)."""
        client = self._clients.get(client_id)
        if not client:
            return None, "unknown_client"
        if redirect_uri not in client.redirect_uris:
            return None, "invalid_redirect_uri"

        code = secrets.token_urlsafe(32)
        self._auth_codes[code] = {
            "client_id"   : client_id,
            "user_sub"    : user_sub,
            "scope"       : scope,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "nonce"       : nonce,
            "exp"         : time.time() + 60,
        }
        return code, None

    def token_from_code(self, code: str, client_id: str,
                         redirect_uri: str,
                         code_verifier: str = None,
                         client_secret: str = None) -> Tuple[Optional[Dict], Optional[str]]:
        """Exchange authorization code for tokens."""
        auth = self._auth_codes.pop(code, None)
        if not auth:
            return None, "invalid_grant"
        if auth["exp"] < time.time():
            return None, "code_expired"
        if auth["client_id"] != client_id:
            return None, "client_id_mismatch"
        if auth["redirect_uri"] != redirect_uri:
            return None, "redirect_uri_mismatch"

        client = self._clients[client_id]
        # PKCE verification
        if auth.get("code_challenge"):
            if not code_verifier:
                return None, "code_verifier_required"
            if not PKCE.verify(code_verifier, auth["code_challenge"]):
                return None, "pkce_failed"
        elif client.is_confidential:
            # Confidential client: validate client_secret
            if client_secret != client.client_secret:
                return None, "invalid_client"

        scopes      = auth["scope"].split()
        user        = self._users.get(auth["user_sub"])
        access_tkn  = self._issue_access_token(auth["user_sub"], scopes, client_id)
        refresh_tkn = self._issue_refresh_token(auth["user_sub"], scopes, client_id)

        response = {
            "access_token" : access_tkn,
            "token_type"   : "Bearer",
            "expires_in"   : 3600,
            "refresh_token": refresh_tkn,
            "scope"        : auth["scope"],
        }
        # OIDC: include id_token if openid scope requested
        if "openid" in scopes and user:
            id_claims = {
                "sub"   : user.sub,
                "aud"   : client_id,
                "email" : user.email if "email" in scopes else None,
                "name"  : user.name  if "profile" in scopes else None,
                "nonce" : auth.get("nonce"),
                "email_verified": user.email_verified,
            }
            id_claims = {k: v for k, v in id_claims.items() if v is not None}
            response["id_token"] = self._jwt.sign(id_claims, expires_in_s=3600)

        return response, None

    def client_credentials(self, client_id: str,
                            client_secret: str,
                            scope: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Machine-to-machine flow: no user involved."""
        client = self._clients.get(client_id)
        if not client or client.client_secret != client_secret:
            return None, "invalid_client"
        if "client_credentials" not in client.grant_types:
            return None, "unsupported_grant_type"
        access_tkn = self._issue_access_token(
            subject=f"service:{client_id}", scopes=scope.split(),
            client_id=client_id, is_service=True)
        return {"access_token": access_tkn, "token_type": "Bearer",
                "expires_in": 3600, "scope": scope}, None

    def refresh(self, refresh_token: str) -> Tuple[Optional[Dict], Optional[str]]:
        info = self._refresh_tokens.get(refresh_token)
        if not info:
            return None, "invalid_refresh_token"
        new_access = self._issue_access_token(
            info["user_sub"], info["scopes"], info["client_id"])
        return {"access_token": new_access, "token_type": "Bearer",
                "expires_in": 3600}, None

    def introspect(self, token: str) -> Dict:
        """Token introspection endpoint (RFC 7662)."""
        payload, err = self._jwt.verify(token)
        if err:
            return {"active": False}
        info = self._access_tokens.get(token, {})
        return {"active": True, **payload, **info}

    def revoke(self, token: str):
        """Revoke access or refresh token."""
        self._access_tokens.pop(token, None)
        self._refresh_tokens.pop(token, None)

    def _issue_access_token(self, subject: str, scopes: List[str],
                              client_id: str, is_service: bool = False) -> str:
        payload = {
            "sub"   : subject,
            "scope" : " ".join(scopes),
            "client_id": client_id,
            "token_type": "access_token",
        }
        tkn = self._jwt.sign(payload, expires_in_s=3600)
        self._access_tokens[tkn] = {"scope": " ".join(scopes), "client_id": client_id}
        return tkn

    def _issue_refresh_token(self, user_sub: str, scopes: List[str],
                               client_id: str) -> str:
        rt = secrets.token_urlsafe(32)
        self._refresh_tokens[rt] = {"user_sub": user_sub, "scopes": scopes,
                                      "client_id": client_id}
        return rt


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_oauth2_oidc():
    print("=" * 65)
    print("OAUTH2 AND OPENID CONNECT (OIDC)")
    print("=" * 65)

    jwt_svc = JWTService("super-secret-signing-key", "https://auth.example.com")
    server  = OAuth2OIDCServer(jwt_svc)

    # Register clients and users
    server.register_client(ClientRegistration(
        client_id="webapp-client", client_secret="webapp-secret",
        redirect_uris=["https://app.example.com/callback"],
        scopes={"openid", "profile", "email", "read:data"},
        grant_types={"authorization_code"},
    ))
    server.register_client(ClientRegistration(
        client_id="spa-client", client_secret="",
        redirect_uris=["http://localhost:3000/callback"],
        scopes={"openid", "profile", "email"},
        grant_types={"authorization_code"},
        is_confidential=False,  # public client — uses PKCE
    ))
    server.register_client(ClientRegistration(
        client_id="service-account", client_secret="service-secret",
        redirect_uris=[],
        scopes={"read:analytics", "write:reports"},
        grant_types={"client_credentials"},
    ))
    server.register_user(OIDCUser("alice-sub-123", "alice@example.com", "Alice Smith"))

    # ── Authorization Code Flow ───────────────────
    print("\n[1] AUTHORIZATION CODE FLOW (server-side app)")
    print("─" * 55)

    code, err = server.authorize(
        "webapp-client", "https://app.example.com/callback",
        "openid profile email", "alice-sub-123", nonce="random-nonce-123"
    )
    print(f"  Auth code issued: {code[:16]}... (error={err})")

    tokens, err = server.token_from_code(
        code, "webapp-client", "https://app.example.com/callback",
        client_secret="webapp-secret"
    )
    print(f"  Tokens: access={tokens['access_token'][:30]}...")
    print(f"          id_token={tokens.get('id_token','(none)')[:40]}...")
    print(f"          scope={tokens['scope']}")

    # Decode ID token
    id_payload, _ = jwt_svc.verify(tokens["id_token"])
    print(f"  ID Token claims: sub={id_payload.get('sub')} "
          f"email={id_payload.get('email')} name={id_payload.get('name')}")

    # ── PKCE Flow (SPA / Mobile) ──────────────────
    print("\n\n[2] AUTHORIZATION CODE + PKCE (SPA/mobile)")
    print("─" * 55)

    verifier   = PKCE.generate_verifier()
    challenge  = PKCE.generate_challenge(verifier)
    print(f"  code_verifier:  {verifier[:20]}...")
    print(f"  code_challenge: {challenge[:20]}...")
    print(f"  PKCE.verify:    {PKCE.verify(verifier, challenge)}")

    code2, _   = server.authorize(
        "spa-client", "http://localhost:3000/callback",
        "openid profile", "alice-sub-123",
        code_challenge=challenge
    )
    tokens2, err2 = server.token_from_code(
        code2, "spa-client", "http://localhost:3000/callback",
        code_verifier=verifier
    )
    print(f"  PKCE exchange: {'OK' if not err2 else err2}")
    print(f"  Access token:  {tokens2['access_token'][:30]}...")

    # Wrong verifier
    _, err_bad = server.token_from_code(
        *server.authorize("spa-client", "http://localhost:3000/callback",
                           "openid", "alice-sub-123",
                           code_challenge=challenge)[:1],
        "spa-client", "http://localhost:3000/callback",
        code_verifier="wrong-verifier"
    ) if False else (None, "skipped")
    # Inline example instead:
    code3, _  = server.authorize("spa-client", "http://localhost:3000/callback",
                                   "openid", "alice-sub-123",
                                   code_challenge=challenge)
    _, err_pkce = server.token_from_code(code3, "spa-client",
                                          "http://localhost:3000/callback",
                                          code_verifier="wrong-verifier-xyz")
    print(f"  Wrong verifier rejected: {err_pkce}")

    # ── Client Credentials ────────────────────────
    print("\n\n[3] CLIENT CREDENTIALS (machine-to-machine)")
    print("─" * 55)

    m2m_tokens, err = server.client_credentials(
        "service-account", "service-secret", "read:analytics"
    )
    print(f"  M2M tokens: {m2m_tokens['access_token'][:30]}...")
    introspect = server.introspect(m2m_tokens["access_token"])
    print(f"  Introspection: active={introspect['active']} "
          f"sub={introspect.get('sub')} scope={introspect.get('scope')}")

    # ── Token Refresh ─────────────────────────────
    print("\n\n[4] TOKEN REFRESH")
    print("─" * 55)

    refreshed, err = server.refresh(tokens["refresh_token"])
    print(f"  New access token: {refreshed['access_token'][:30]}... (error={err})")

    # ── Revocation ────────────────────────────────
    print("\n\n[5] TOKEN REVOCATION")
    print("─" * 55)

    token_to_revoke = tokens["access_token"]
    before = server.introspect(token_to_revoke)["active"]
    server.revoke(token_to_revoke)
    after  = server.introspect(token_to_revoke)["active"]
    print(f"  Before revoke: active={before}")
    print(f"  After revoke:  active={after}")

    # ── OIDC Discovery ────────────────────────────
    print("\n\n[6] OIDC DISCOVERY DOCUMENT")
    print("─" * 55)

    discovery = {
        "issuer"                                    : jwt_svc.issuer,
        "authorization_endpoint"                    : f"{jwt_svc.issuer}/authorize",
        "token_endpoint"                            : f"{jwt_svc.issuer}/token",
        "userinfo_endpoint"                         : f"{jwt_svc.issuer}/userinfo",
        "jwks_uri"                                  : f"{jwt_svc.issuer}/.well-known/jwks.json",
        "scopes_supported"                          : ["openid","profile","email"],
        "response_types_supported"                  : ["code"],
        "grant_types_supported"                     : ["authorization_code","client_credentials"],
        "token_endpoint_auth_methods_supported"     : ["client_secret_post","none"],
        "code_challenge_methods_supported"          : ["S256"],
    }
    for k, v in discovery.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demonstrate_oauth2_oidc()
