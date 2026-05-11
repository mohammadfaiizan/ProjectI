# Networking and Protocols for System Design

> Network fundamentals that directly impact architectural decisions. Understanding these protocols helps you choose the right communication mechanism and explain trade-offs in interviews.

---

## HTTP/1.1 vs HTTP/2 vs HTTP/3

### HTTP/1.1 — The Foundation

Released in 1997, HTTP/1.1 is the baseline that everything else improves upon.

**Key characteristics:**
- Text-based protocol (headers and status lines are ASCII)
- One request per TCP connection at a time (with keep-alive: multiple requests, but sequential)
- Connection reuse via `Connection: keep-alive`
- Chunked transfer encoding (streaming responses)

**Head-of-Line (HOL) Blocking in HTTP/1.1:**
```
TCP Connection (keep-alive):
Request 1 ──────────────────────→ Response 1
                                             Request 2 ──→ Response 2

Request 2 must wait for Request 1 to complete!
→ This is HTTP/1.1 HOL blocking

Workaround: Browsers open 6-8 parallel TCP connections per domain
→ But each TCP connection has overhead (handshake, slow start)
→ Domain sharding (multiple domains): anti-pattern for HTTP/2
```

### HTTP/2 — Multiplexing

Released in 2015, HTTP/2 solves HTTP/1.1's performance limitations.

**Key features:**

**1. Multiplexing (single TCP connection, multiple concurrent requests):**
```
Single TCP connection:
Stream 1: ─────────────────────────── (request 1)
Stream 2:   ─────────── (request 2, concurrent!)
Stream 3:     ─────────────────────── (request 3, concurrent!)

All on ONE TCP connection → no HOL blocking at HTTP layer
→ No need for 6+ connections per domain
```

**2. Binary Protocol:**
```
HTTP/1.1: GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n (text)
HTTP/2:   [frame type][flags][stream id][payload] (binary)

Binary is more efficient to parse, less error-prone
```

**3. Header Compression (HPACK):**
```
HTTP/1.1: Same headers sent on every request (~400-1000 bytes)
HTTP/2:   HPACK compression + header table (reference common headers)
           First request: full headers
           Subsequent: "same headers as before" (or just the diff)
→ 85-88% reduction in header size for typical web traffic
```

**4. Server Push:**
```
Client requests /index.html
Server pushes /style.css and /script.js proactively
→ Client doesn't have to request them separately

Reality check: Server push is being deprecated (Chrome removed support 2022)
→ HTTP/103 Early Hints is the modern replacement
```

**5. Stream Prioritization:**
```
Mark HTML as highest priority, images as lower priority
→ Critical resources load first
```

**HTTP/2 Limitation — TCP-level HOL Blocking:**
```
HTTP/2 over TCP: single TCP connection
If one TCP packet is lost: ALL streams stall while TCP retransmits!
→ HTTP/2 solves HTTP-level HOL blocking but not TCP-level HOL blocking
→ This is why HTTP/3 uses QUIC (UDP-based)
```

### HTTP/3 — QUIC Protocol

HTTP/3 replaces TCP with QUIC, a new transport protocol built on UDP.

**QUIC Key Innovations:**

**1. UDP-based (no TCP HOL blocking):**
```
Each stream has independent packet loss recovery
Packet loss on Stream 1 doesn't block Stream 2 or 3
→ Eliminates TCP-level HOL blocking
→ 0-RTT or 1-RTT connection establishment (vs TCP+TLS: 2-3 RTTs)
```

**2. Connection Migration:**
```
TCP: connection = (src_ip, src_port, dst_ip, dst_port)
→ Change IP (WiFi → 4G) = new connection = re-establish TLS!

QUIC: connection = connection_id (persistent)
→ Change IP = same connection continues!
→ Perfect for mobile (switching between WiFi and cellular)
```

**3. Built-in TLS 1.3:**
```
TCP + TLS: TCP handshake (1 RTT) + TLS handshake (1-2 RTT) = 2-3 RTTs
QUIC: combined handshake = 1 RTT (or 0 RTT for resuming connections)
→ Significantly faster connection establishment
```

### HTTP Version Comparison Table

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | QUIC (UDP) |
| Multiplexing | No (HOL blocking) | Yes | Yes |
| Header compression | No | HPACK | QPACK |
| Binary framing | No (text) | Yes | Yes |
| Server push | No | Yes (deprecated) | Yes |
| TCP HOL blocking | Yes | Yes | No |
| Connection migration | No | No | Yes |
| TLS | Separate | Separate | Built-in (required) |
| RTT for connection | 2-3 RTTs | 2-3 RTTs | 0-1 RTT |
| Adoption (2024) | ~25% | ~40% | ~30%+ |

---

## HTTPS and TLS Handshake

### TLS 1.2 Handshake (Legacy — 2 RTTs)

```
Client                                    Server
  │                                          │
  │──── ClientHello ──────────────────────→ │
  │     (TLS version, cipher suites,         │
  │      random bytes, session ID)           │
  │                                          │
  │ ←── ServerHello + Certificate ──────── │
  │     (chosen cipher, server cert,         │
  │      server random bytes)               │
  │                                          │
  │ [Client validates server certificate]    │
  │ [Client generates pre-master secret]     │
  │                                          │
  │──── ClientKeyExchange + ChangeCipherSpec→│
  │     (pre-master secret encrypted with    │
  │      server's public key)               │
  │                                          │
  │ ←── ChangeCipherSpec + Finished ─────  │
  │                                          │
  │──── HTTP Request ────────────────────→  │
  │                                          │
Total: 2 RTTs before first byte of data
```

### TLS 1.3 Handshake (Modern — 1 RTT)

```
Client                                    Server
  │                                          │
  │──── ClientHello + Key Share ──────────→ │
  │     (TLS version, cipher suites,         │
  │      Diffie-Hellman key share)           │
  │                                          │
  │ ←── ServerHello + Key Share ──────────  │
  │     + Certificate + Finished ──────────  │
  │     (DH key share, certificate,          │
  │      encrypted with derived key)        │
  │                                          │
  │──── Finished + HTTP Request ───────── → │
  │                                          │
Total: 1 RTT (session data sent immediately after server hello)
```

### TLS 0-RTT Resumption

```
For known clients (session resumption):
Client sends application data in the FIRST message!
→ 0 RTT before data reaches server

Risk: 0-RTT data is replay-attackable
→ Only safe for idempotent requests (GET, not POST /payment)
```

### Certificate Chain of Trust

```
Root CA (trusted by OS/browser)
    └── Intermediate CA
            └── Server Certificate (example.com)

Browser checks:
1. Server cert signed by Intermediate CA ✓
2. Intermediate CA signed by Root CA ✓
3. Root CA in OS trust store ✓
4. Certificate not expired ✓
5. Certificate not revoked (OCSP/CRL) ✓
6. CN/SAN matches hostname ✓
→ Connection trusted!
```

### Important TLS Concepts for System Design

```
Certificate Pinning:
- App hardcodes expected server certificate/public key
- Prevents MITM even if CA is compromised
- Challenge: certificate rotation requires app update

mTLS (Mutual TLS):
- Both client AND server present certificates
- Server verifies client identity (not just vice versa)
- Used in: microservices, service mesh, internal APIs
- Unlike regular TLS: client also needs a certificate

HSTS (HTTP Strict Transport Security):
- Response header: Strict-Transport-Security: max-age=31536000
- Browser remembers: always use HTTPS for this domain
- Prevents SSL stripping attacks
```

---

## REST vs GraphQL vs gRPC — Decision Matrix

### REST (Representational State Transfer)

**Principles:**
- Resources identified by URLs
- Standard HTTP verbs (GET, POST, PUT, DELETE, PATCH)
- Stateless
- Uniform interface

```
REST API example:
GET    /users/{id}              → Get user
POST   /users                  → Create user
PUT    /users/{id}              → Replace user  
PATCH  /users/{id}              → Update user fields
DELETE /users/{id}              → Delete user
GET    /users/{id}/posts        → Get user's posts
POST   /users/{id}/posts        → Create post for user
```

**REST Pros:**
- Simple, universally understood
- Cacheable (GET responses can be cached by CDN, browsers)
- Works with any HTTP client
- Great tooling (Postman, Swagger, OpenAPI)
- Easy to debug (human-readable)

**REST Cons:**
- Over-fetching: GET /user returns all fields, you only need name+email
- Under-fetching: Need /user AND /posts AND /comments = 3 round trips
- Versioning is messy (v1, v2 in URL or header)
- No strong type system (JSON is loosely typed)

### GraphQL

**Concept:** Client specifies exactly what data it needs in a query.

```graphql
# Client asks for EXACTLY what it needs — no more, no less
query {
  user(id: "123") {
    name          # only name
    email         # only email
    posts {       # nested in same request
      title
      createdAt
    }
    followers {
      count       # only the count, not the full follower objects
    }
  }
}
```

**GraphQL Pros:**
- Solves over-fetching and under-fetching
- Single endpoint (`/graphql`)
- Strongly typed schema (introspection)
- Great for complex, nested data (social graphs, product catalogs)
- Frontend teams can iterate without backend changes

**GraphQL Cons:**
- Caching is hard (all POST requests to `/graphql` — can't use HTTP cache)
- N+1 query problem (requires DataLoader batching)
- Complexity for simple use cases
- Learning curve
- Security: unbounded queries can overload servers (query depth limits needed)

**N+1 Problem in GraphQL:**
```
Query: Get 100 users and their posts

Without DataLoader:
→ SELECT * FROM users LIMIT 100    (1 query)
→ SELECT * FROM posts WHERE user_id = 1  (for user 1)
→ SELECT * FROM posts WHERE user_id = 2  (for user 2)
→ ... × 100 users
= 101 queries! (N+1 problem)

With DataLoader:
→ SELECT * FROM users LIMIT 100
→ SELECT * FROM posts WHERE user_id IN (1,2,...,100)  (batched!)
= 2 queries total
```

### gRPC (Google Remote Procedure Call)

**Concept:** Define service and message types in Protobuf, generate client/server code.

```protobuf
// Service definition
service UserService {
  rpc GetUser(GetUserRequest) returns (User);
  rpc ListUsers(ListUsersRequest) returns (stream User);  // server streaming
  rpc CreateUser(CreateUserRequest) returns (User);
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
}
```

**gRPC Pros:**
- High performance: binary (Protobuf) + HTTP/2 multiplexing
- Strongly typed with code generation
- Bi-directional streaming (client streaming, server streaming, bidirectional)
- Built-in load balancing, retries, deadlines
- Ideal for microservices-to-microservices communication

**gRPC Cons:**
- Not human-readable (binary)
- Requires HTTP/2 (limited browser support — need gRPC-web proxy)
- Steeper learning curve
- Cannot be called from curl/browser directly
- Protobuf schema evolution requires care

### Decision Matrix

| Criteria | REST | GraphQL | gRPC |
|---|---|---|---|
| Client type | Any (browser, mobile, server) | Browser, mobile | Server-to-server |
| Performance | Medium | Medium | High |
| Caching | Excellent (HTTP caching) | Complex | Manual |
| Streaming | No (SSE/WebSocket needed) | Subscriptions | Native |
| Type safety | Weak (OpenAPI helps) | Strong | Strong (Protobuf) |
| Learning curve | Low | Medium | High |
| Tooling maturity | Excellent | Good | Good |
| Browser support | Full | Full | Limited (need proxy) |
| Schema evolution | Versioning in URL | Additive only (deprecation) | Protobuf backward compat |

**When to use each:**

```
REST: 
→ Public APIs consumed by external developers
→ Simple CRUD operations
→ When HTTP caching is valuable
→ When simplicity > performance

GraphQL:
→ Complex UI with many different data requirements
→ Rapid frontend iteration without backend changes
→ Mobile apps (bandwidth-sensitive, need minimal data)
→ Aggregating multiple backend services into one API

gRPC:
→ Internal microservice communication
→ High-throughput, low-latency requirements
→ Bidirectional streaming (chat, live updates)
→ Polyglot environments (generate clients in any language)
```

---

## Real-Time Communication Patterns

### HTTP Long Polling

```
Client:                     Server:
Request: GET /events  ──→  Hold request open...
                           (no events available yet)
                           (30 seconds later: event arrives)
                    ←──   Response: {event: "new_message"}
                    
Client immediately re-polls:
Request: GET /events  ──→  Hold request open...
```

**Pros:** Works everywhere, simple, no persistent connection required
**Cons:** Server holds connection open (resource waste), high latency (poll interval), many connections under load

### Server-Sent Events (SSE)

```
Client: GET /events                HTTP/1.1
        Accept: text/event-stream  Connection: keep-alive

Server sends stream:
data: {"type":"message","content":"Hello"}\n\n
data: {"type":"notification","count":3}\n\n
event: heartbeat\n
data: ping\n\n
```

**SSE Protocol:**
- Unidirectional: server → client only
- Built on HTTP (works through proxies/firewalls)
- Automatic reconnection on disconnect (built into browser EventSource API)
- Text-based (UTF-8)
- Each event separated by double newline

**Pros:** Simple, built-in browser support (EventSource), automatic reconnect, HTTP/2 compatible
**Cons:** Unidirectional only, text-only, older browsers limit connections to 6/domain (HTTP/1.1)

### WebSocket

```
// HTTP Upgrade handshake
Client → Server:
GET /chat HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

Server → Client:
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

// After upgrade: full-duplex binary or text frames
Client → Server: {"action":"sendMessage","content":"Hello"}
Server → Client: {"type":"message","from":"user2","content":"Hi!"}
```

**Pros:** Full-duplex (both sides send simultaneously), low overhead after handshake, binary support
**Cons:** Not HTTP (firewalls/proxies may block), no built-in reconnection (must implement), stateful connection (scaling is harder)

### WebSocket Scaling Challenges

```
Problem: WebSocket connections are persistent and stateful
Each message must go to the specific server holding the connection

Solution 1: Sticky sessions (user → same server)
→ Problem: Uneven load, server failure = disconnect

Solution 2: Pub/Sub message routing
User A connected to Server 1
User B connected to Server 2
User A sends message to User B:
→ Server 1 publishes to Redis Pub/Sub channel "user_B_inbox"
→ Server 2 subscribed to "user_B_inbox", receives message
→ Server 2 delivers to User B's WebSocket connection

This is how WhatsApp, Slack, Discord scale WebSockets
```

### Real-Time Comparison Table

| Feature | Long Polling | SSE | WebSocket |
|---|---|---|---|
| Direction | Client→Server, Server→Client (alternate) | Server→Client only | Full-duplex |
| Protocol | HTTP | HTTP | WS (upgraded HTTP) |
| Browser support | Full | Full (not IE11) | Full |
| Proxy/firewall | Works (HTTP) | Works (HTTP) | May be blocked |
| Reconnection | Manual | Automatic | Manual |
| Latency | High (poll interval) | Low | Very low |
| Overhead per message | High (HTTP headers) | Low | Very low (2-10 bytes) |
| Scaling | Easier (stateless) | Medium | Hard (stateful) |
| Use case | Infrequent updates | News feeds, stock prices | Chat, gaming, live collab |

---

## DNS Resolution Process

### DNS Hierarchy

```
Root DNS Servers (13 clusters, anycast)
    └── TLD Name Servers (.com, .org, .uk)
            └── Authoritative Name Servers (for example.com)
                    └── DNS Record (A, AAAA, CNAME, MX, etc.)
```

### Recursive DNS Resolution (Full Process)

```
Browser: "I need the IP for api.example.com"

Step 1: Check local cache (browser cache, OS cache)
        → Hit: done! return cached IP (respects TTL)
        → Miss: continue...

Step 2: Query Recursive Resolver (your ISP or 8.8.8.8)
        → Recursive resolver checks its cache
        → Miss: continue...

Step 3: Recursive resolver queries Root Name Server
        → "Who handles .com?"
        → Root: "TLD .com is handled by a.gtld-servers.net"

Step 4: Recursive resolver queries .com TLD server
        → "Who handles example.com?"
        → TLD: "example.com is handled by ns1.example.com"

Step 5: Recursive resolver queries example.com authoritative server
        → "What's the IP for api.example.com?"
        → Authoritative: "api.example.com → 93.184.216.34"

Step 6: Recursive resolver caches result (per TTL)
        Returns 93.184.216.34 to client

Step 7: Client caches result, connects to 93.184.216.34
```

### DNS Record Types

| Record | Purpose | Example |
|---|---|---|
| A | IPv4 address | api.example.com → 1.2.3.4 |
| AAAA | IPv6 address | api.example.com → 2001:db8::1 |
| CNAME | Alias to another hostname | www.example.com → example.com |
| MX | Mail server | example.com → mail.example.com (priority 10) |
| TXT | Arbitrary text (SPF, DKIM, verification) | v=spf1 include:... |
| NS | Authoritative name server | example.com → ns1.example.com |
| SOA | Start of Authority (zone metadata) | Serial, refresh, retry values |
| SRV | Service location (port, priority, weight) | _http._tcp.example.com |
| PTR | Reverse lookup (IP → hostname) | 4.3.2.1.in-addr.arpa → host |

### TTL (Time to Live) Strategy

```
TTL = how long DNS resolvers can cache this record

Low TTL (60-300s):
✓ Use for: services that may need failover, canary deployments
✗ Cost: more DNS queries, higher latency (cache miss more often)
✗ Warning: change TTL BEFORE the DNS change — otherwise cache is stale!

High TTL (3600-86400s):
✓ Use for: stable services, reduces DNS query load
✓ Performance: cache hit means 0ms DNS lookup
✗ Failover takes up to TTL duration to propagate

Best practice for failover:
1. Lower TTL to 60s (24-48 hours before planned change)
2. Wait for caches to expire (max old TTL time)
3. Make DNS change
4. Wait for caches to expire (new 60s TTL)
5. Verify change propagated
6. Raise TTL back to 3600s
```

### Anycast DNS

```
Anycast: Many servers share the same IP address
→ Routers direct queries to the nearest server (BGP routing)

Root DNS servers use anycast:
IP 198.41.0.4 (a.root-servers.net) → answered by nearest of ~100 nodes globally

Benefits:
→ Low latency (nearest node responds)
→ DDoS resistance (attack spread across many nodes)
→ Automatic failover (BGP reroutes if node down)

CDNs also use anycast for edge node resolution
```

---

## TCP vs UDP — System Design Perspective

### TCP (Transmission Control Protocol)

**Guarantees:**
1. **Ordered delivery:** Packets arrive in sequence (reordered if needed)
2. **Reliable delivery:** Lost packets are retransmitted
3. **Error checking:** Checksum on every segment
4. **Flow control:** Receiver controls send rate (window)
5. **Congestion control:** Reduces rate when network congested (slow start, AIMD)

**TCP Connection (3-way handshake):**
```
Client → Server: SYN (seq=x)
Server → Client: SYN-ACK (seq=y, ack=x+1)
Client → Server: ACK (ack=y+1)
→ 1 RTT before data can be sent (not counting TLS)
```

**TCP Teardown (4-way):**
```
Client → Server: FIN
Server → Client: ACK
Server → Client: FIN
Client → Server: ACK
→ TIME_WAIT state: client waits 2×MSL (~4 minutes)
```

### UDP (User Datagram Protocol)

**Characteristics:**
- No connection establishment
- No ordering guarantee
- No retransmission on loss
- No flow/congestion control
- Very low overhead (8-byte header vs TCP's 20-byte minimum)

**UDP Header:**
```
Source Port (2 bytes) | Destination Port (2 bytes)
Length (2 bytes)      | Checksum (2 bytes)
Data...
```

### TCP vs UDP Decision Guide

| Scenario | Protocol | Reason |
|---|---|---|
| Web browsing, APIs | TCP | Reliability required |
| File transfer | TCP | No data loss acceptable |
| Email | TCP | Reliability required |
| Database connections | TCP | Consistency required |
| Video streaming (live) | UDP (or QUIC) | Tolerate loss, latency critical |
| Online gaming | UDP | Low latency > reliability |
| VoIP, video calls | UDP (RTP over UDP) | Latency > reliability |
| DNS queries | UDP (small queries) | Fast, single packet |
| DNS zone transfer | TCP | Large, needs reliability |
| IoT sensor data | UDP | Lightweight, occasional loss OK |
| DHCP | UDP | Broadcast-based |

**Rule of thumb:**
```
TCP when: You CANNOT lose any data, ordering matters, connection-oriented
UDP when: Latency matters more than perfect delivery, stateless is OK,
          application handles retransmission/ordering itself
          OR when you build your own reliability layer (QUIC)
```

---

## CDN Architecture

### CDN Components

```
User                 CDN Edge                   Origin Server
  │                    (PoP)                         │
  │──── Request ──→  ┌─────────┐                    │
  │                  │ Edge    │── Cache Miss? ──→   │
  │                  │ Cache   │                    (fetch from origin)
  │←── Response ───  │         │←── Response ─────── │
  │   (from edge)    └─────────┘    (cached for TTL)
```

**PoP (Point of Presence):** Edge location with servers close to users (100+ globally for major CDNs)

### CDN Caching Layers

```
User → Edge PoP → Regional Cache → Origin

1. Edge cache: Closest to user, smallest (SSD)
2. Regional cache: Covers many edge nodes, larger
3. Origin: Source of truth, handles cache misses

Cache hierarchy reduces origin load:
→ Edge hit: ~5-20ms latency
→ Regional hit: ~20-50ms latency
→ Origin miss: ~50-200ms latency
```

### Cache-Control Headers

```http
# Public content (CDN can cache)
Cache-Control: public, max-age=86400

# Private content (CDN cannot cache, user's browser can)
Cache-Control: private, max-age=3600

# No caching anywhere
Cache-Control: no-store

# Cache but revalidate every request
Cache-Control: no-cache  
(confusingly named: it DOES cache, but must revalidate with origin first)

# CDN caches differently than browser
Cache-Control: public, max-age=60, s-maxage=3600
(browser: 60s, CDN: 3600s)

# Immutable (hash in URL, cache forever)
Cache-Control: public, max-age=31536000, immutable
(use for /static/app.a1b2c3d4.js — hash changes when content changes)
```

### CDN Cache Invalidation

```
Problem: CDN cached old version, you deployed new code

Options:
1. Versioned URLs (best):
   /static/app.abc123.js → change URL → CDN fetches new version
   Old URL still works for users loading old cached page

2. Purge/Invalidation API:
   POST /cdn/purge { "urls": ["/index.html"] }
   → Tells CDN to evict these URLs from all edge caches
   → Propagation time: seconds to minutes
   
3. Wait for TTL:
   Set short TTL for frequently changing content
   → Not suitable for immediate invalidation needs

4. Surrogate Keys (Varnish, Fastly, Cloudflare):
   Tag cached objects: Cache-Tag: product_123, category_electronics
   → Purge by tag: "invalidate all objects tagged product_123"
```

### CDN for Dynamic Content

```
Static CDN: Cache files that rarely change (images, JS, CSS, videos)

Dynamic CDN acceleration (doesn't cache, but optimizes):
→ TCP connection reuse (CDN → origin always has warm connection)
→ Optimized routing (CDN finds fastest path to origin)
→ Protocol optimization (HTTP/2, HTTP/3 to edge even if origin uses HTTP/1.1)
→ TLS termination at edge (closer to user)

Examples: Cloudflare Workers, Lambda@Edge — run code at edge
→ Can cache personalized content (vary by cookie/header)
→ A/B testing at edge
→ Authentication at edge
```

---

## Reverse Proxy vs Forward Proxy vs API Gateway

### Forward Proxy

```
Client → [Forward Proxy] → Internet

Use cases:
→ Corporate firewall (inspects employee traffic)
→ Anonymization (hide client IP)
→ Content filtering (block certain sites)
→ Caching (corporate cache)

The proxy acts on behalf of the CLIENT
Client knows about the proxy; server doesn't know real client IP
```

### Reverse Proxy

```
Internet → [Reverse Proxy] → Backend Servers

Use cases:
→ Load balancing (distribute across backend servers)
→ SSL termination (handle TLS, backend uses plain HTTP)
→ Caching (cache responses)
→ Compression (gzip responses)
→ Rate limiting
→ Authentication (forward auth)
→ Hiding backend topology (clients don't know about backend servers)

The proxy acts on behalf of the SERVER
Server knows about the proxy; client doesn't know which backend served them

Examples: Nginx, HAProxy, Caddy, AWS ALB
```

### API Gateway

An API Gateway is a specialized reverse proxy for API traffic with additional API management features.

```
Client → [API Gateway] → [Service A]
                       → [Service B]
                       → [Service C]

API Gateway responsibilities:
→ Authentication/Authorization (verify JWT, API keys)
→ Rate limiting (per client, per endpoint)
→ Request routing (path-based, version-based)
→ Request/response transformation
→ Protocol translation (REST → gRPC)
→ Analytics and logging
→ Circuit breaking
→ API versioning (v1, v2 routing)
→ Developer portal / documentation

Examples: Kong, AWS API Gateway, Apigee, Nginx API Gateway
```

### Comparison

| Feature | Forward Proxy | Reverse Proxy | API Gateway |
|---|---|---|---|
| Proxy for | Client | Server | Server (API-specific) |
| Client awareness | Client knows | Client unaware | Client unaware |
| Load balancing | No | Yes | Yes |
| SSL termination | No | Yes | Yes |
| Authentication | No | Optional | Yes (primary feature) |
| Rate limiting | No | Basic | Advanced (per-key, per-plan) |
| Analytics | No | Basic | Full API analytics |
| Protocol translation | No | No | Yes |

---

## OSI Model — Layers Relevant to System Design

### OSI Model Overview

```
Layer 7 — Application   (HTTP, HTTPS, FTP, SMTP, DNS, WebSocket)
Layer 6 — Presentation  (TLS/SSL, compression, encoding)
Layer 5 — Session       (session establishment, RPC)
Layer 4 — Transport     (TCP, UDP, QUIC — ports, reliability)
Layer 3 — Network       (IP, ICMP, BGP — routing between networks)
Layer 2 — Data Link     (Ethernet, WiFi — within network segment)
Layer 1 — Physical      (cables, fiber, radio waves)
```

### L4 vs L7 Load Balancing

**L4 Load Balancing (Transport Layer):**
```
Balances based on: IP address + TCP port
Can't see: HTTP headers, path, cookies

Works by:
→ Source NAT (rewrite source IP) or
→ DSR (Direct Server Return — server responds directly to client)

Pros: Fast (no parsing), works for any TCP/UDP protocol
Cons: Can't do path-based routing, can't inspect content

Example: AWS NLB (Network Load Balancer)
Use when: Non-HTTP protocols, very high throughput, consistent hashing needed
```

**L7 Load Balancing (Application Layer):**
```
Balances based on: HTTP headers, URL path, cookies, content type

Can do:
→ Path-based routing: /api/* → service A, /static/* → CDN
→ Host-based routing: api.example.com → service A, www → service B
→ Content-based routing: route by Accept-Language header
→ Sticky sessions: read session cookie, route to same server
→ SSL termination: decrypt here, forward HTTP to backend
→ Health checking: actually test HTTP endpoint (not just TCP)

Pros: Intelligent routing, SSL offload, content-aware
Cons: More overhead (parse HTTP), can't do non-HTTP protocols

Example: AWS ALB (Application Load Balancer), Nginx, HAProxy (mode http)
Use when: HTTP/HTTPS traffic, microservices routing, SSL termination
```

### Key Network Concepts for System Design

**NAT (Network Address Translation):**
```
Private IPs (10.x, 172.16.x, 192.168.x) can't route on public internet
NAT gateway translates: private_IP:port ↔ public_IP:port
→ Many servers share one public IP
→ Stateful: maintains translation table
```

**Anycast vs Unicast:**
```
Unicast: one sender → one specific receiver (normal TCP connection)
Anycast: one IP address → multiple servers, nearest responds
→ Used by: CDNs, DNS, DDoS protection (traffic absorbed at nearest node)
```

---

## WebRTC (Peer-to-Peer Communication)

### WebRTC Architecture

```
Peer A                                              Peer B
  │                                                    │
  │──── Offer (SDP) ──→  [Signaling Server]  ←────   │
  │                          │                         │
  │ ←── Answer (SDP) ────── │                         │
  │                          │                         │
  │── ICE Candidates ──────→ │ ←──── ICE Candidates ──│
  │                                                    │
  │─────── Direct P2P Connection (once established) ──→│
  │ (UDP for media, DataChannel for data)              │
```

**Signaling:** Exchange of session descriptions (SDP = Session Description Protocol) through a server. WebRTC doesn't specify signaling protocol — use WebSocket, REST, XMPP, etc.

**ICE (Interactive Connectivity Establishment):** Finding the best path between peers:
1. Try direct connection (same network)
2. Try STUN (learn your public IP, try direct with NAT traversal)
3. Try TURN (relay server if P2P fails — ~15-20% of connections need relay)

**STUN vs TURN:**
```
STUN (Session Traversal Utilities for NAT):
→ Tells peer its public IP:port as seen from outside NAT
→ Enables hole-punching through symmetric NAT
→ Free to run, minimal bandwidth (just coordination)

TURN (Traversal Using Relays around NAT):
→ Relay server that forwards all media between peers
→ Used when STUN fails (strict firewalls, symmetric NAT)
→ Expensive: relays all video/audio bandwidth
→ Need to budget: ~15-20% of users need TURN
→ TURN bandwidth: 2× bitrate × number of relayed connections
```

**WebRTC Use Cases:**
- Video calls (Google Meet, Zoom uses WebRTC)
- Peer-to-peer file sharing
- Screen sharing
- Real-time gaming
- Live streaming (one-to-many with SFU — Selective Forwarding Unit)

---

## Networking in Cloud (VPC, Subnets, Security Groups)

### VPC (Virtual Private Cloud)

```
AWS Region
└── VPC (10.0.0.0/16 — 65,536 IPs)
    ├── Public Subnet (10.0.1.0/24 — 256 IPs)
    │   [Internet Gateway attached]
    │   → Can receive traffic from internet
    │   → Load balancers, NAT gateways, bastion hosts
    │
    ├── Private Subnet - App (10.0.2.0/24)
    │   [No internet gateway]
    │   → Can reach internet via NAT Gateway
    │   → Application servers, microservices
    │
    └── Private Subnet - DB (10.0.3.0/24)
        [No internet access at all]
        → Databases, cache servers
        → Only accessible from app subnet
```

### Security Groups (Stateful Firewall at Instance Level)

```
Web Server Security Group:
  Inbound:
    - Port 443 (HTTPS) from 0.0.0.0/0 (anyone)
    - Port 22 (SSH) from 10.0.0.0/8 (internal only)
  Outbound:
    - Port 3306 (MySQL) to DB Security Group
    - Port 6379 (Redis) to Cache Security Group

DB Security Group:
  Inbound:
    - Port 3306 (MySQL) from Web Server Security Group ONLY
    - No public internet access!
  Outbound:
    - All (for OS updates via NAT)

Stateful: If you allow inbound on port 80, 
          the return traffic is automatically allowed
          (no need to add outbound rule for established connections)
```

### NAT Gateway

```
Private Subnet Instance needs to reach internet (software updates, API calls):

Private instance (10.0.2.5) 
    → NAT Gateway (in public subnet, has elastic IP 54.1.2.3)
    → Internet
    ← Response comes back to 54.1.2.3 (NAT)
    → NAT translates back to 10.0.2.5

Key: Internet cannot initiate connection to private instances
     Private instances CAN reach internet
```

### VPC Peering and Transit Gateway

```
VPC Peering: Connect two VPCs (same or different accounts/regions)
→ Traffic stays in AWS network (not public internet)
→ Limitation: non-transitive (A↔B, B↔C doesn't mean A↔C)

Transit Gateway: Hub-and-spoke for many VPCs
→ Central router, connect many VPCs and on-premises
→ Transitive routing supported
→ Use when: > 5 VPCs to connect
```

---

## Service Mesh (Istio/Envoy)

### What Problem Does a Service Mesh Solve?

Without service mesh, every microservice must implement:
- mTLS (mutual authentication between services)
- Load balancing
- Circuit breaking
- Distributed tracing
- Retries
- Rate limiting

**Service mesh moves this to the infrastructure layer.**

### Sidecar Pattern

```
┌─────────────────────────────────────────────────────────┐
│  Pod / VM                                                │
│   ┌───────────────┐      ┌─────────────────────────┐   │
│   │  Application  │◄────►│  Sidecar Proxy (Envoy)  │   │
│   │   Container   │      │  - mTLS                 │   │
│   │               │      │  - Load balancing        │   │
│   │               │      │  - Circuit breaking      │   │
│   │               │      │  - Observability         │   │
│   └───────────────┘      └─────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         ↕ All traffic flows through sidecar
```

The application code is completely unaware of:
- mTLS encryption between services
- Retries and circuit breaking
- Distributed trace header propagation
- Traffic policies

### Istio Architecture

```
Control Plane (Istiod):
  → Pushes configuration to all Envoy sidecars
  → Certificate Authority (issues service certificates for mTLS)
  → Collects telemetry

Data Plane (Envoy sidecars):
  → Intercepts all network traffic (via iptables rules)
  → Enforces policies
  → Reports metrics/traces

Service A → [Envoy A] ─(mTLS encrypted)─ [Envoy B] → Service B
           ↑                              ↑
   sidecar, same pod               sidecar, same pod
```

### Service Mesh Trade-offs

```
Benefits:
✓ Zero-code mTLS (service identity, encrypted service-to-service)
✓ Traffic management (canary, A/B, weighted routing)
✓ Observability (automatic distributed tracing, metrics)
✓ Consistent retry/circuit breaking policy

Costs:
✗ Latency overhead (~1-5ms per hop for sidecar processing)
✗ Operational complexity (another system to manage)
✗ Memory overhead (~50-100MB per sidecar)
✗ Steep learning curve

When to adopt:
→ 10+ microservices
→ Strong security requirements (zero-trust)
→ Platform team to own it
→ NOT for: small teams, monolith, early startup
```

---

## Quick Reference: Protocol Comparison Table

### When to Use Each Protocol

| Protocol | Transport | Latency | Direction | Use Case |
|---|---|---|---|---|
| HTTP/1.1 | TCP | Medium | Request/Response | Legacy APIs, simple services |
| HTTP/2 | TCP | Low | Request/Response + Server Push | Modern web APIs |
| HTTP/3/QUIC | UDP | Very Low | Request/Response | Mobile apps, high-packet-loss networks |
| WebSocket | TCP (upgraded) | Very Low | Full-duplex | Chat, live collaboration, gaming |
| SSE | TCP (HTTP) | Low | Server→Client | Dashboards, news feeds, notifications |
| gRPC | HTTP/2 | Very Low | Full-duplex streaming | Microservices, high-throughput internal APIs |
| GraphQL | HTTP | Low | Request/Response | Complex UI data requirements |
| REST | HTTP | Medium | Request/Response | Public APIs, CRUD services |
| WebRTC | UDP | Ultra Low | Full-duplex P2P | Video calls, voice calls, P2P data |
| MQTT | TCP | Low | Pub/Sub | IoT, mobile messaging |
| AMQP | TCP | Low | Message Queue | RabbitMQ, enterprise messaging |

### Load Balancer Type Selection

| Criteria | L4 (NLB) | L7 (ALB/nginx) |
|---|---|---|
| Protocol | Any TCP/UDP | HTTP/HTTPS only |
| Routing granularity | IP:port | URL, headers, content |
| TLS termination | No (passthrough) | Yes |
| Performance | Faster | Slightly slower (parsing) |
| Sticky sessions | Basic (IP hash) | Cookie-based |
| Health check | TCP connection | HTTP endpoint |

### DNS Record Selection

| Scenario | Record Type |
|---|---|
| Point domain to IP | A (IPv4) or AAAA (IPv6) |
| Alias domain to another domain | CNAME |
| CloudFront/S3 root domain | Alias (AWS) or ANAME |
| Route to nearest PoP | A with anycast |
| Blue-green deployment | Change A record |
| Canary (10% new version) | Weighted routing (Route53) |

### Security Protocol Selection

| Need | Solution |
|---|---|
| Encrypt in transit | TLS 1.3 |
| Encrypt at rest | AES-256 |
| Service-to-service auth | mTLS |
| User authentication | JWT + HTTPS |
| API key management | API Gateway |
| DDoS protection | Anycast CDN (Cloudflare) |
| WAF | AWS WAF, Cloudflare WAF |

---

## Networking Interview Cheat Sheet

```
Q: "How does a request from browser reach your server?"
A: Browser → DNS lookup → TCP handshake → TLS handshake → 
   HTTP request → CDN (cache hit?) → Load Balancer → 
   App Server → Cache lookup → Database → Response back

Q: "Why use HTTP/2 over HTTP/1.1?"
A: Multiplexing (one TCP connection, multiple concurrent requests),
   header compression (HPACK saves 85%+ bandwidth),
   binary framing (efficient parsing)

Q: "When would you use WebSocket vs SSE?"
A: SSE: server → client only (notifications, live dashboard, news feed)
   WebSocket: bidirectional (chat, collaborative editing, gaming)
   SSE is simpler and works better through HTTP/2

Q: "How does a CDN improve performance?"
A: Caches content at edge nodes close to users (reduces latency 10x+),
   reduces origin load, built-in DDoS protection, TLS termination at edge

Q: "What is mTLS and when do you need it?"
A: Both parties present TLS certificates (not just server).
   Used in: microservices, zero-trust networks, service mesh.
   Ensures: "I know this is ServiceA, not just an authenticated user"

Q: "REST vs gRPC — when to choose?"
A: REST for: public APIs, browser clients, when caching matters
   gRPC for: internal microservices, bidirectional streaming, 
            high-throughput, polyglot environments
```

---

*Reference: "High Performance Browser Networking" by Ilya Grigorik, MDN Web Docs, Cloudflare Learning Center, IETF RFCs for HTTP/2 (7540), HTTP/3 (9114), QUIC (9000)*
