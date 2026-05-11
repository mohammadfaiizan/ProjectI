# Networking and Protocols — Interview Q&A

> 20 questions | Easy: Q1–Q7 | Medium: Q8–Q15 | Hard: Q16–Q20

---

## EASY (Q1–Q7)

---

### Q1. What happens when you type a URL in a browser? (Full flow)

**Answer:**

This is one of the most comprehensive networking questions in interviews. The full flow involves DNS, TCP, TLS, HTTP, and rendering — all in under a second.

**Complete step-by-step:**

```
1. URL PARSING
   Browser parses: https://www.example.com/path?q=hello
   Protocol: HTTPS | Host: www.example.com | Path: /path | Query: q=hello

2. DNS RESOLUTION (detailed in Q4)
   Browser cache → OS cache → Recursive resolver → Root → TLD → Authoritative
   Result: 93.184.216.34 (IP address)

3. TCP CONNECTION
   Browser initiates 3-way handshake with server IP on port 443:
   Client → [SYN]           → Server
   Client ← [SYN-ACK]       ← Server
   Client → [ACK]           → Server
   TCP connection established

4. TLS HANDSHAKE (since HTTPS)
   Client → [ClientHello: TLS version, cipher suites, random]
   Client ← [ServerHello: chosen cipher, server cert, random]
   Client validates cert against CA chain
   Client → [Key exchange: encrypted pre-master secret]
   Both derive session keys
   Client → [Finished (encrypted)]
   Client ← [Finished (encrypted)]
   Encrypted channel established

5. HTTP REQUEST
   GET /path?q=hello HTTP/2
   Host: www.example.com
   Accept: text/html
   Authorization: Bearer <token>

6. SERVER PROCESSING
   Nginx/LB receives request → routes to app server
   App server queries DB/cache → builds HTML response

7. HTTP RESPONSE
   HTTP/2 200 OK
   Content-Type: text/html
   Content-Encoding: gzip
   <html>...</html>

8. BROWSER RENDERING
   Parse HTML → build DOM
   Parse CSS → build CSSOM
   Execute JS (may trigger more requests)
   Render page (layout → paint → composite)
```

**Latency breakdown (typical):**
| Step | Typical Duration |
|---|---|
| DNS lookup | 20–120ms (first visit) / 0ms (cached) |
| TCP handshake | 1 RTT (~30ms US) |
| TLS handshake | 1–2 RTTs (~30–60ms) |
| Server processing | 50–200ms |
| Data transfer | Varies with page size |

---

### Q2. What is the difference between HTTP/1.1, HTTP/2, and HTTP/3?

**Answer:**

Each version addresses fundamental bottlenecks in the previous.

**HTTP/1.1 (1997):**
- One request per TCP connection at a time (head-of-line blocking)
- Workaround: browsers open 6–8 parallel TCP connections per domain
- Supports persistent connections (keep-alive) but still sequential
- Text-based protocol (verbose headers, no compression)

**HTTP/2 (2015):**
- **Multiplexing:** Multiple requests/responses on a single TCP connection simultaneously using streams
- **Header compression:** HPACK algorithm reduces redundant header data
- **Server push:** Server can proactively send assets before client requests them
- **Binary framing:** More efficient than text-based HTTP/1.1
- Still uses TCP — TCP-level head-of-line blocking remains (packet loss stalls all streams)

**HTTP/3 (2022):**
- Built on **QUIC** (UDP-based) instead of TCP
- Eliminates TCP head-of-line blocking — lost packets only stall one stream, not all
- **0-RTT connection resumption:** Reconnect with zero round trips if session exists
- Built-in TLS 1.3 (handshake integrated into QUIC)
- Better performance on lossy networks (mobile, satellite)

**Comparison table:**

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | QUIC (UDP) |
| Multiplexing | No | Yes (streams) | Yes (streams) |
| Header compression | No | HPACK | QPACK |
| HOL blocking | App + TCP level | TCP level only | Eliminated |
| Encryption | Optional | Optional (most use HTTPS) | Mandatory |
| Connection setup | 1-RTT TCP + 2-RTT TLS | Same | 0-RTT (resumption) |
| Server push | No | Yes | Yes (limited) |

**When to use what:**
- HTTP/2 is the standard for most modern APIs and web applications.
- HTTP/3 is valuable for real-time apps, high-latency networks, mobile, gaming.

---

### Q3. What is the difference between TCP and UDP, and when do you use each?

**Answer:**

**TCP (Transmission Control Protocol):** Connection-oriented, reliable, ordered delivery with error checking. Guarantees every byte arrives in order.

**UDP (User Datagram Protocol):** Connectionless, unreliable, no ordering guarantees. Fire-and-forget packets.

**TCP mechanisms:**
```
1. Connection establishment: 3-way handshake (SYN, SYN-ACK, ACK)
2. Reliability: Sequence numbers + acknowledgments
3. Retransmission: Lost packets are retransmitted automatically
4. Flow control: Receiver advertises window size
5. Congestion control: Slow start, congestion avoidance, fast retransmit
6. Ordered delivery: Out-of-order packets are buffered until gap is filled
7. Connection teardown: 4-way FIN handshake
```

**UDP characteristics:**
```
- No handshake → lower connection overhead
- No retransmission → lower latency, but packets can be lost
- No ordering → application must handle out-of-order packets
- No flow control → sender can overwhelm receiver
- Much smaller header: 8 bytes (vs TCP 20 bytes)
```

**Use case decision table:**

| Use Case | Protocol | Reason |
|---|---|---|
| Web browsing (HTTP/HTTPS) | TCP | Need all data reliably |
| File transfer (FTP, SFTP) | TCP | Missing bytes corrupt files |
| Email (SMTP, IMAP) | TCP | Need reliable delivery |
| Video streaming (Netflix) | TCP (QUIC/HTTP3) | Adaptive bitrate; reliability matters |
| Online gaming | UDP | Stale data worse than missing data; low latency critical |
| VoIP / Real-time audio | UDP | A missed audio packet is better skipped than delayed |
| Live video (WebRTC) | UDP (SRTP) | Latency > reliability for real-time |
| DNS queries | UDP (small) | Single request/response; retry if no response |
| DNS zone transfers | TCP | Large data, need reliability |
| DHCP | UDP | Broadcast before IP assigned |

---

### Q4. Explain DNS resolution steps in detail.

**Answer:**

DNS (Domain Name System) translates human-readable domain names into IP addresses. It is a hierarchical, distributed database.

**DNS resolution flow:**

```
User types: www.example.com

Step 1: BROWSER CACHE
  Check browser DNS cache (TTL-based)
  If found → return IP immediately

Step 2: OS CACHE
  Check /etc/hosts (local override file)
  Check OS DNS cache
  If found → return IP

Step 3: RECURSIVE RESOLVER (ISP or custom like 8.8.8.8)
  OS contacts configured DNS resolver
  Resolver checks its own cache
  If not cached → begins iterative resolution

Step 4: ROOT NAME SERVER
  Resolver queries root server (.)
  Root returns: "For .com, go ask: a.gtld-servers.net"
  (13 root server clusters worldwide, operated by 12 organizations)

Step 5: TLD NAME SERVER
  Resolver queries .com TLD server
  TLD returns: "For example.com, go ask: ns1.example.com"
  (TLD servers managed by registries like Verisign for .com)

Step 6: AUTHORITATIVE NAME SERVER
  Resolver queries ns1.example.com
  Authoritative server returns: 93.184.216.34 (A record)
  TTL = 3600 seconds

Step 7: RESPONSE CHAIN
  Resolver caches result for TTL duration
  Returns IP to OS → OS returns to browser
  Browser connects to 93.184.216.34
```

**DNS record types:**

| Record | Purpose | Example |
|---|---|---|
| A | IPv4 address | example.com → 93.184.216.34 |
| AAAA | IPv6 address | example.com → 2001:db8::1 |
| CNAME | Alias to another name | www → example.com |
| MX | Mail server | example.com → mail.example.com |
| TXT | Text data (SPF, DKIM) | v=spf1 include:... |
| NS | Name server | example.com → ns1.example.com |
| PTR | Reverse lookup | 34.216.184.93.in-addr.arpa → example.com |
| SRV | Service discovery | _http._tcp.example.com |

**TTL trade-off:**
- Low TTL (60s): Fast failover, high DNS query volume.
- High TTL (86400s): Fewer queries, slow failover on IP changes.

---

### Q5. What is the difference between a reverse proxy and a forward proxy?

**Answer:**

**Forward proxy:** Sits in front of *clients*. Clients send requests through the proxy, which forwards to servers on their behalf. The server sees the proxy's IP, not the client's.

**Reverse proxy:** Sits in front of *servers*. Clients send requests to the proxy, which forwards to backend servers. The client sees only the proxy, not the backend servers.

```
FORWARD PROXY
  [Client A] ──┐
  [Client B] ──┼──> [Forward Proxy] ──> [Internet/Server]
  [Client C] ──┘
  
  Use: Corporate network filter, bypass geo-restrictions, anonymization

REVERSE PROXY
  [Client A] ──┐
  [Client B] ──┼──> [Reverse Proxy] ──> [Server 1]
  [Client C] ──┘              └──────> [Server 2]
                                └────> [Server 3]
  
  Use: Load balancing, SSL termination, caching, security, compression
```

**Forward proxy use cases:**
- Corporate internet filtering (block social media)
- Anonymizing client identity (VPN behavior)
- Caching for outbound requests (reduce bandwidth)
- Bypassing geo-restrictions

**Reverse proxy use cases:**
- Load balancing across backend servers
- SSL/TLS termination (offload crypto from app servers)
- Request routing (path-based, header-based)
- Response caching (Varnish, Nginx)
- DDoS protection (hide real server IPs)
- Compression and request/response transformation
- Web Application Firewall (WAF)

**Common implementations:**
| Tool | Forward Proxy | Reverse Proxy |
|---|---|---|
| Nginx | Yes | Yes |
| HAProxy | Limited | Yes |
| Squid | Yes | Limited |
| Traefik | No | Yes |
| AWS CloudFront | No | Yes (CDN) |
| Envoy | No | Yes (service mesh) |

---

### Q6. What is a CDN (Content Delivery Network) and how does it work?

**Answer:**

A CDN is a geographically distributed network of servers (Points of Presence, PoPs) that cache and serve content from locations close to users, reducing latency and origin server load.

**CDN request flow:**

```
WITHOUT CDN:
  User in Tokyo ──── 150ms ────> Origin Server (New York)
  
WITH CDN:
  User in Tokyo ──── 5ms ────> CDN PoP (Tokyo) ──── cache hit ────> Response
  
  On cache miss:
  User in Tokyo → CDN PoP (Tokyo) → Origin (New York) → CDN caches → User
  Subsequent requests: served from Tokyo PoP (5ms)
```

**How CDNs work:**

```
1. DNS MAGIC: CDN uses anycast or geographic DNS
   When user resolves cdn.example.com, DNS returns IP of
   nearest CDN edge node, not origin

2. EDGE CACHING:
   Static assets (JS, CSS, images, videos) cached at edge
   Cache key = URL + Vary headers
   TTL controlled by Cache-Control headers from origin

3. ORIGIN PULL (lazy caching):
   CDN fetches from origin only on cache miss
   Subsequent requests for same content: served from edge

4. ORIGIN PUSH (active caching):
   Content pre-loaded to edge nodes before users request it
   Used for large file deployments (software downloads)

5. CACHE INVALIDATION:
   CDN purge API clears specific URLs
   TTL-based expiry for routine updates
```

**CDN benefits:**

| Benefit | How |
|---|---|
| Reduced latency | Serve from geographically close edge node |
| Reduced origin load | 80-95% of requests served from CDN cache |
| DDoS protection | Distribute attack traffic across hundreds of PoPs |
| Scalability | CDN absorbs traffic spikes |
| Availability | Origin can be down; CDN serves cached content |

**What CDNs serve:**
- Static: HTML, CSS, JS, images, fonts, videos.
- Dynamic (edge compute): Personalized content using edge workers (Cloudflare Workers, Lambda@Edge).

**Major CDN providers:** Cloudflare, Akamai, AWS CloudFront, Fastly, Google Cloud CDN.

---

### Q7. What are the OSI model layers, and which are most relevant to system design?

**Answer:**

The OSI (Open Systems Interconnection) model has 7 layers, each with specific responsibilities. For system design interviews, layers 3, 4, and 7 are most relevant.

**All 7 layers:**

```
Layer 7: APPLICATION  — HTTP, HTTPS, DNS, SMTP, gRPC, WebSocket
Layer 6: PRESENTATION — Encryption (TLS), compression, encoding (JSON/XML)
Layer 5: SESSION      — Session management, authentication
Layer 4: TRANSPORT    — TCP, UDP — port numbers, reliable delivery
Layer 3: NETWORK      — IP routing, ICMP — IP addresses, routing
Layer 2: DATA LINK    — Ethernet, WiFi — MAC addresses, switching
Layer 1: PHYSICAL     — Cables, radio waves, electrical signals
```

**System design relevance:**

| Layer | Relevance |
|---|---|
| L7 — Application | API design (REST/gRPC), HTTP caching, WebSockets |
| L4 — Transport | TCP vs UDP choice, L4 vs L7 load balancing, firewalls |
| L3 — Network | IP routing, anycast, BGP, network segmentation (VPC) |
| L2 — Data Link | Rarely direct; matters for on-prem network design |

**L4 vs L7 load balancing:**
```
L4 (Transport) Load Balancer:
  Routes based on IP + TCP port only
  Does NOT inspect HTTP content
  Faster (less processing), but dumber routing
  Example: AWS Network Load Balancer (NLB)

L7 (Application) Load Balancer:
  Inspects HTTP headers, URL paths, cookies
  Can route /api → API servers, /images → media servers
  Can terminate SSL, inject headers, perform A/B routing
  Example: AWS Application Load Balancer (ALB), Nginx
```

---

## MEDIUM (Q8–Q15)

---

### Q8. How does the TLS handshake work?

**Answer:**

TLS (Transport Layer Security) establishes an encrypted, authenticated channel between client and server. TLS 1.3 (current standard) simplified the handshake to 1-RTT (vs 2-RTT in TLS 1.2).

**TLS 1.3 handshake (1-RTT):**

```
CLIENT                                    SERVER
  │                                          │
  │── ClientHello ──────────────────────────>│
  │   (TLS version, cipher suites,           │
  │    client random, key share)             │
  │                                          │
  │<─────────────────────── ServerHello ─────│
  │<────────────────── Certificate ──────────│
  │<────────────────── CertificateVerify ────│
  │<────────────────── Finished ─────────────│
  │   (server random, chosen cipher,         │
  │    server key share, encrypted cert)     │
  │                                          │
  │ [Client verifies cert against CA chain]  │
  │ [Both derive session keys using ECDHE]   │
  │                                          │
  │── Finished ─────────────────────────────>│
  │── (First encrypted request) ────────────>│
  │                                          │
         ENCRYPTED CHANNEL ESTABLISHED
```

**Key concepts:**

**Certificate verification:**
```
Server cert signed by → Intermediate CA → Root CA
Browser has Root CA certs built-in
Chain: example.com cert → DigiCert CA → Root CA (trusted)
Browser validates: not expired, domain matches, chain valid
```

**ECDHE key exchange (why it's secure):**
```
ECDHE = Elliptic Curve Diffie-Hellman Ephemeral
- Both sides generate temporary key pairs
- Exchange public keys
- Each derives the same shared secret WITHOUT transmitting it
- "Ephemeral" = new key pair per session → perfect forward secrecy
  (Compromise of server's long-term key does NOT expose past sessions)
```

**TLS 1.3 improvements over TLS 1.2:**
| Property | TLS 1.2 | TLS 1.3 |
|---|---|---|
| Handshake RTTs | 2 RTT | 1 RTT |
| 0-RTT resumption | No | Yes (session tickets) |
| Cipher suites | Many (some weak) | 5 strong only |
| Forward secrecy | Optional | Mandatory |
| RSA key exchange | Allowed | Removed |

---

### Q9. Compare REST, GraphQL, and gRPC. When do you use each?

**Answer:**

These are three fundamentally different API paradigms, each with specific strengths.

**REST (Representational State Transfer):**
```
Design: Resource-based URLs, standard HTTP verbs
GET    /users/123
POST   /users
PUT    /users/123
DELETE /users/123

Response: JSON (typically)
Protocol: HTTP/1.1 or HTTP/2
```

**GraphQL:**
```
Design: Single endpoint, client specifies exact data shape
POST /graphql

Query:
{
  user(id: "123") {
    name
    email
    posts(last: 5) {
      title
      createdAt
    }
  }
}

Response: Only the requested fields — no over-fetching
```

**gRPC:**
```
Design: Strongly typed contracts via Protocol Buffers (.proto files)
// user.proto
service UserService {
  rpc GetUser (UserRequest) returns (UserResponse);
  rpc ListUsers (ListRequest) returns (stream UserResponse);
}

Transport: HTTP/2 (binary, multiplexed)
Response: Binary Protocol Buffers (compact, fast to serialize/deserialize)
```

**Comparison matrix:**

| Dimension | REST | GraphQL | gRPC |
|---|---|---|---|
| Protocol | HTTP | HTTP | HTTP/2 |
| Format | JSON | JSON | Binary (Protobuf) |
| Schema contract | OpenAPI (optional) | Strongly typed schema | .proto files (required) |
| Over-fetching | Common | Eliminated | Eliminated |
| Under-fetching | Requires multiple calls | Eliminated (one query) | Eliminated (one call) |
| Browser support | Native | Native | Limited (needs grpc-web) |
| Streaming | Limited (SSE, chunked) | Subscriptions | Bidirectional streaming |
| Performance | Moderate | Moderate | High (5–10× faster than REST) |
| Learning curve | Low | Medium | High |
| Best for | Public APIs, CRUD | Complex client queries, mobile | Internal microservices, high-perf |

**Decision guide:**
- **REST:** Public-facing APIs, third-party integrations, when simplicity and familiarity matter.
- **GraphQL:** Mobile apps (minimize data transfer), complex frontend data requirements, BFF (Backend for Frontend) pattern.
- **gRPC:** Internal microservice communication, real-time streaming, polyglot microservices where performance matters.

---

### Q10. Compare WebSocket, Server-Sent Events (SSE), and long polling.

**Answer:**

All three solve the "server push" problem — how to get data from server to client without the client constantly polling.

**Long Polling:**
```
Client ──── GET /events ────────────────────> Server
                                              (Server holds request open)
                                              (Event occurs after 10s)
Client <─── HTTP 200 {event data} ─────────── Server
Client ──── GET /events (immediately) ──────> Server
                                              (waits again...)

Characteristics:
  - Standard HTTP; works everywhere
  - High server connection overhead (connections held open)
  - Half-duplex: client always initiates
  - Latency: dependent on poll interval
  - Firewall friendly (standard HTTP)
```

**Server-Sent Events (SSE):**
```
Client ──── GET /stream ────────────────────> Server
Client <─── HTTP 200 (keep-alive stream) ─── Server
            data: {"event": "message1"}       (immediate)
            data: {"event": "message2"}       (5s later)
            data: {"event": "message3"}       (2s later)
            ...connection stays open...

Characteristics:
  - One-directional: server → client only
  - Built on HTTP (EventSource browser API)
  - Automatic reconnection built into browser
  - Text-only (not binary)
  - Works through HTTP/2 multiplexing efficiently
```

**WebSocket:**
```
Client ──── HTTP Upgrade Request ───────────> Server
Client <─── 101 Switching Protocols ──────── Server

Now fully bidirectional:
Client ──── "send message" ─────────────────> Server
Client <─── "new message from user B" ─────── Server
Client <─── "user C joined" ───────────────── Server
Client ──── "typing indicator" ─────────────> Server

Characteristics:
  - Full-duplex: both sides can send anytime
  - Low overhead after upgrade (no HTTP headers per message)
  - Supports binary and text
  - Must manage connection lifecycle manually
  - Some firewalls/proxies block WebSocket upgrades
```

**Decision table:**

| Scenario | Best Choice | Reason |
|---|---|---|
| Live chat (WhatsApp-like) | WebSocket | Full-duplex, real-time bidirectional |
| Stock price feed | SSE | Server-only push; simpler implementation |
| Notifications (Gmail badge) | SSE or Long Polling | Simple, HTTP-compatible |
| Multiplayer game | WebSocket | Low latency, bidirectional |
| Live sports scores | SSE | Unidirectional, reconnect auto-handles |
| Collaborative editing | WebSocket | Must sync changes both ways |
| Simple status updates | Long Polling | Works everywhere, simple |

---

### Q11. How does HTTP/2 multiplexing work?

**Answer:**

HTTP/2 multiplexing allows multiple concurrent request-response exchanges over a **single TCP connection**, solving HTTP/1.1's requirement for parallel connections.

**The problem with HTTP/1.1:**
```
HTTP/1.1 — Sequential requests on each connection:
  Conn 1: [Request A] ─────> [Response A] [Request B] ─> [Response B]
  Conn 2: [Request C] ─────> [Response C]
  Conn 3: [Request D] ─────> [Response D]
  Browsers open 6-8 connections — overhead-heavy
```

**HTTP/2 solution — Streams:**
```
HTTP/2 — Concurrent streams on ONE connection:
  
  TCP Connection
  ├── Stream 1: GET /html    ─────────────────> [Response HTML]
  ├── Stream 3: GET /style   ──────────────> [Response CSS]
  ├── Stream 5: GET /script  ────────────────> [Response JS]
  └── Stream 7: GET /logo    ──────────> [Response image]
  
  All interleaved as binary frames on the same TCP connection
```

**HTTP/2 framing layer:**
```
Frame structure:
┌─────────────────────────────────────────────┐
│ Length (24 bits) │ Type (8 bits) │ Flags (8) │
│          Stream ID (31 bits)                │
│                  Payload                    │
└─────────────────────────────────────────────┘

Frame types:
  HEADERS — HTTP request/response headers
  DATA    — Request/response body
  SETTINGS — Connection configuration
  WINDOW_UPDATE — Flow control
  PRIORITY — Stream priority
  PUSH_PROMISE — Server push announcement
  RST_STREAM — Cancel a stream
```

**Key HTTP/2 features:**

| Feature | Description |
|---|---|
| Multiplexing | Multiple streams on one TCP connection |
| Stream prioritization | Streams have weights and dependencies |
| Header compression (HPACK) | Uses static + dynamic header tables; reduces redundant headers |
| Server push | Server sends assets before client requests them |
| Binary framing | Compact, less error-prone than text |
| Flow control | Per-stream and per-connection flow windows |

**Remaining limitation:** TCP-level head-of-line blocking. If a TCP packet is lost, all HTTP/2 streams wait for that packet to be retransmitted (even streams not using that data). HTTP/3/QUIC fixes this.

---

### Q12. What is the QUIC protocol?

**Answer:**

QUIC (Quick UDP Internet Connections) is a transport protocol developed by Google and standardized by IETF as RFC 9000. It is the transport layer for HTTP/3, replacing TCP+TLS.

**Why QUIC was created:**
```
TCP limitations:
1. Head-of-line blocking: One lost packet stalls entire connection
2. Slow handshake: TCP 3-way + TLS 2-RTT = 3 round trips total
3. No stream multiplexing at transport layer
4. Hard to update (TCP in OS kernel; slow to change)
5. Connection tied to IP:port (mobile handoff breaks connections)
```

**QUIC design:**
```
┌────────────────────────────────────────────────┐
│                  HTTP/3 (Application)          │
├────────────────────────────────────────────────┤
│                  QUIC (Transport)              │
│  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Connection  │  │  Stream Multiplexing  │  │
│  │  Management  │  │  (independent streams)│  │
│  └──────────────┘  └───────────────────────┘  │
│  ┌───────────────────────────────────────────┐ │
│  │         TLS 1.3 (integrated)              │ │
│  └───────────────────────────────────────────┘ │
├────────────────────────────────────────────────┤
│                  UDP (Network)                 │
└────────────────────────────────────────────────┘
```

**Key QUIC features:**

| Feature | Benefit |
|---|---|
| Built on UDP | Deployable without OS kernel changes |
| 0-RTT connection resumption | Reconnect with zero round trips using session tickets |
| 1-RTT new connections | TLS 1.3 integrated into QUIC handshake |
| Independent streams | Packet loss affects only the stream using that packet |
| Connection migration | Connection ID (not IP:port) → works across network changes (WiFi → LTE) |
| Mandatory encryption | All QUIC traffic is encrypted (no plaintext mode) |
| Improved ACK | More granular ACK ranges → faster loss detection |

**0-RTT in practice:**
```
First visit:  Client ─1-RTT─> Server (standard QUIC handshake)
              Server gives client a session ticket

Return visit: Client ──0-RTT──> Server
              Client sends data in first packet along with session ticket
              Server responds immediately
              
              0-RTT tradeoff: Replay attack vulnerability for non-idempotent requests
```

**Adoption:** HTTP/3 accounts for ~30% of web traffic as of 2024. Supported by all major browsers and Cloudflare, Google, and Facebook's infrastructure.

---

### Q13. What is an API gateway and how does it differ from a reverse proxy?

**Answer:**

Both sit between clients and backends, but they operate at different abstraction levels with very different responsibilities.

**Reverse Proxy:** A network-level component that forwards requests to backend servers. It is primarily concerned with routing and traffic management.

**API Gateway:** An application-level component that acts as the single entry point for all client requests. It understands APIs — authentication, rate limiting, request transformation, protocol translation.

```
REVERSE PROXY                    API GATEWAY
────────────────────             ────────────────────────────────────
Client ──> Proxy ──> Backend     Client ──> API GW ──> Service A
                                               │──> Service B
Focus:                                         │──> Service C
  Routing                        
  Load balancing                 Focus:
  SSL termination                  Authentication (JWT, OAuth, API key)
  Caching                          Rate limiting per client
                                   Request/Response transformation
                                   Protocol translation (REST → gRPC)
                                   API composition (aggregate responses)
                                   Analytics and billing metering
                                   Circuit breaking
                                   Service discovery integration
```

**Feature comparison:**

| Feature | Reverse Proxy | API Gateway |
|---|---|---|
| Load balancing | Yes | Yes |
| SSL termination | Yes | Yes |
| Authentication | No | Yes |
| Rate limiting | Basic (IP-based) | Advanced (per user/key/plan) |
| Request routing | URL/header based | Full API routing |
| Request transformation | Limited | Yes (header, body manipulation) |
| Protocol translation | No | Yes (REST↔gRPC, HTTP↔WebSocket) |
| API versioning | No | Yes |
| Developer portal | No | Often included |

**Examples:**
- **Reverse Proxy:** Nginx, HAProxy, Traefik
- **API Gateway:** AWS API Gateway, Kong, Apigee, Tyk, Azure API Management

**When to use API Gateway:** Any public API that needs authentication, rate limiting, and analytics. Microservices architectures with many backend services.

**When reverse proxy is enough:** Internal service-to-service routing, simple load balancing, SSL termination only.

---

### Q14. How does DNS load balancing work?

**Answer:**

DNS load balancing distributes traffic across multiple servers by returning different IP addresses in response to the same DNS query, without requiring a traditional load balancer hardware/software.

**Basic DNS round-robin:**
```
DNS query: api.example.com

Response 1: 10.0.0.1  (Server A)
Response 2: 10.0.0.2  (Server B)
Response 3: 10.0.0.3  (Server C)
Response 4: 10.0.0.1  (Server A again — cycles)

DNS server rotates which IP is returned first in the list
```

**DNS load balancing types:**

**Round Robin DNS:**
```
Client 1 asks → DNS returns [10.0.0.1, 10.0.0.2, 10.0.0.3]
Client 2 asks → DNS returns [10.0.0.2, 10.0.0.3, 10.0.0.1]
Client 3 asks → DNS returns [10.0.0.3, 10.0.0.1, 10.0.0.2]
Client connects to FIRST IP in list
```

**Geographic DNS (GeoDNS):**
```
User from Tokyo  → DNS returns 103.31.4.1  (Tokyo server)
User from London → DNS returns 146.75.28.1 (London server)
User from NYC    → DNS returns 151.101.1.1 (New York server)
```

**Weighted DNS:**
```
Server A (weight 70%): 10.0.0.1
Server B (weight 20%): 10.0.0.2
Server C (weight 10%): 10.0.0.3
DNS returns A's IP 70% of the time
```

**Limitations of DNS load balancing:**
| Limitation | Problem | Solution |
|---|---|---|
| TTL caching | Clients cache DNS response; traffic keeps going to dead server | Low TTL (60s), but increases DNS load |
| No health checking | DNS does not know if server is down | Pair with health-check-aware DNS (Route 53) |
| Not truly balanced | Client OS may ignore TTL; caches can be sticky | Use with application-level LB too |
| Sticky clients | Mobile apps may cache DNS indefinitely | Nothing can fully prevent this |

**AWS Route 53 health-check routing:** Solves the health checking limitation — Route 53 actively monitors endpoints and removes failed IPs from DNS responses automatically, providing DNS load balancing with health-check awareness.

---

### Q15. What is mTLS and when is it used for service-to-service authentication?

**Answer:**

**mTLS (mutual TLS)** extends standard TLS to authenticate both sides of the connection. In standard TLS, only the server presents a certificate. In mTLS, both the server AND the client present certificates.

**Standard TLS vs mTLS:**
```
STANDARD TLS:
  Client ──────────────────────────────> Server
  Client: "Who are you?"
  Server: presents certificate (verified by CA)
  Client: "OK, I trust you. Let's talk."
  Server: never verifies client identity

MUTUAL TLS (mTLS):
  Client ──────────────────────────────> Server
  Client: "Who are you?"
  Server: presents certificate (verified by CA)
  Client: presents its own certificate
  Server: "I verify your cert. I trust you."
  Both verified → encrypted channel established
```

**mTLS use cases in system design:**

```
1. MICROSERVICE AUTHENTICATION (Zero-Trust Architecture)
   [Order Service]──mTLS──[Payment Service]
   Payment Service only accepts connections from certificates
   signed by the internal CA (not just any caller)

2. API CLIENT AUTHENTICATION
   High-value B2B APIs where API keys are not secure enough
   Bank → third-party fintech: both present certs

3. SERVICE MESH (Istio, Linkerd)
   mTLS automatically between ALL service pods
   Service identity = certificate CN (common name)
   [Service A] ──mTLS (cert: service-a.default.svc)──> [Service B]

4. IoT DEVICE AUTHENTICATION
   Device sends cert installed at manufacturing time
   Server verifies cert chain → device is genuine
```

**Certificate management in microservices:**
```
Internal CA (e.g., HashiCorp Vault, AWS PCA):
  ├── Issues short-lived certs to each service (24h TTL)
  ├── Service auto-rotates cert before expiry
  └── Revocation via OCSP or short TTL

Service mesh sidecar proxy (Envoy):
  ├── Handles mTLS transparently
  ├── Application code does NOT change
  └── Certificates managed by control plane (Istio Citadel)
```

---

## HARD (Q16–Q20)

---

### Q16. Explain WebRTC STUN and TURN servers. When is each used?

**Answer:**

WebRTC enables peer-to-peer (P2P) real-time audio/video communication directly between browsers. The challenge is that most devices are behind NAT (Network Address Translation) and firewalls, making direct P2P connections difficult.

**The NAT problem:**
```
Peer A:
  Private IP: 192.168.1.5
  Public IP:  203.0.113.1  (NAT device's IP)
  
Peer B:
  Private IP: 10.0.0.7
  Public IP:  198.51.100.1

Problem: Peer A doesn't know its own public IP.
         If A sends packets to B directly, NAT drops them
         (no inbound mapping exists yet)
```

**STUN (Session Traversal Utilities for NAT):**

```
STUN server tells a client its own public IP:port as seen from the internet.

Peer A ──"What is my public IP?"──> STUN Server
Peer A <──"Your public endpoint is 203.0.113.1:54321"── STUN Server

Now Peer A knows: public IP + port

Both peers exchange their public endpoints via signaling server.
Peers attempt direct connection using ICE (Interactive Connectivity Establishment).

Works for: Full cone NAT, restricted cone NAT (~60-70% of cases)
Fails for: Symmetric NAT (corporate firewalls, strict NATs)
Cost: Nearly free (tiny bandwidth, open source servers)
```

**TURN (Traversal Using Relays around NAT):**

```
When direct P2P fails (symmetric NAT), TURN acts as a relay:

Peer A ──media──> TURN Server ──media──> Peer B

Both peers connect to TURN server, which relays all media.

Works for: ALL NAT types including symmetric NAT
Cost: High — TURN server must relay all media (bandwidth ≈ call bitrate × 2)
Latency: Higher than direct P2P (adds one network hop)
Required for: ~30-40% of WebRTC calls
```

**ICE candidate process:**
```
1. Gather ICE candidates for both peers:
   - Host candidate:   local IP (192.168.1.5:5000)
   - Server-reflexive: STUN-discovered public IP (203.0.113.1:54321)
   - Relay:            TURN server address (if P2P fails)

2. Exchange candidates via signaling server (WebSocket/HTTP)

3. Connectivity checks (STUN binding requests to each candidate pair)

4. Select best working candidate pair:
   Direct P2P > server-reflexive > TURN relay
```

**Architecture for WebRTC app:**
```
                  ┌─────────────┐
                  │  Signaling  │ (WebSocket server for SDP exchange)
                  │   Server    │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    [Peer A] ──STUN──> [STUN]    [Peer A] ──TURN──> [TURN] ──> [Peer B]
                                  (fallback when direct fails)
```

---

### Q17. What is anycast routing and how is it used in system design?

**Answer:**

Anycast is a routing strategy where the same IP address is advertised from multiple physical locations simultaneously. The network (BGP routing) automatically directs traffic to the topologically nearest instance of that IP.

**How anycast works:**
```
Normal (unicast): 1 IP address → 1 destination
Anycast:          1 IP address → MANY destinations (nearest wins)

Example: IP 1.1.1.1 (Cloudflare DNS) is announced from 250+ PoPs

User in Tokyo  ─── routing decision ──> Tokyo PoP  (1.1.1.1)
User in London ─── routing decision ──> London PoP (1.1.1.1)
User in NYC    ─── routing decision ──> NYC PoP    (1.1.1.1)

All reach the "same" IP, but different physical servers
```

**BGP anycast mechanism:**
```
Multiple data centers each announce:
  "I can route to 198.51.100.1"

BGP routing protocol selects the path with fewest AS hops
(autonomous system hops, approximately correlates to physical distance)

User's ISP routes to nearest announcing datacenter
```

**Use cases in system design:**

| Use Case | Details |
|---|---|
| DNS resolution | Root DNS servers (13 addresses → 600+ physical servers worldwide) |
| DDoS mitigation | Attack traffic absorbed by nearest PoP instead of hitting origin |
| CDN routing | User request sent to nearest edge node automatically |
| Global load balancing | No DNS tricks needed; routing layer handles it |
| Network time (NTP) | pool.ntp.org uses anycast |

**Anycast vs GeoDNS:**
```
GeoDNS:
  - DNS layer routing
  - TTL caching means stale routing for minutes
  - Flexible policy (latency-based, weighted)
  - Works at DNS level; doesn't help after DNS is resolved

Anycast:
  - Network layer (BGP) routing — no caching
  - Immediate rerouting if a PoP goes down
  - Less flexible policy
  - Works at packet level for every request
```

**Anycast for DDoS:**
```
Attack: 1 Tbps DDoS against 1.1.1.1

Without anycast: Single datacenter overwhelmed
With anycast:    1 Tbps spread across 250 PoPs
                 Each PoP absorbs ~4 Gbps → manageable
```

**Limitation:** Anycast provides no session affinity. Successive packets from the same client *could* route to different servers if BGP tables change (rare in practice). TCP sessions generally stay on one server, but if routing changes mid-session, TCP connection breaks. Solution: Use anycast for UDP services (DNS) or pair with short-lived TCP connections.

---

### Q18. How does TCP congestion control work, and why does it matter for system design?

**Answer:**

TCP congestion control prevents senders from overwhelming the network. It dynamically adjusts the sender's transmission rate based on observed network conditions. Understanding this helps explain why some protocols (gRPC over HTTP/2, QUIC) perform better under specific conditions.

**Core concepts:**

```
Congestion Window (cwnd): Sender's self-imposed limit on unacknowledged data
Receive Window (rwnd):    Receiver's buffer capacity (flow control)
Effective window = min(cwnd, rwnd)
```

**TCP Reno congestion control phases:**

```
PHASE 1: SLOW START
  Start: cwnd = 1 MSS (Maximum Segment Size, ~1460 bytes)
  On each ACK: cwnd += 1 MSS
  cwnd doubles each RTT (exponential growth)
  Until cwnd hits ssthresh (slow start threshold)

  cwnd: 1 → 2 → 4 → 8 → 16 → ... → ssthresh

PHASE 2: CONGESTION AVOIDANCE
  After ssthresh: cwnd += 1/cwnd per ACK (linear growth)
  cwnd grows by ~1 MSS per RTT
  
  cwnd: ssthresh → ssthresh+1 → ssthresh+2 → ...

PHASE 3: PACKET LOSS DETECTION
  Timeout:        ssthresh = cwnd/2; cwnd = 1 (restart slow start)
  3 Duplicate ACKs (fast retransmit):
                  ssthresh = cwnd/2; cwnd = ssthresh (skip slow start)
```

**Visual:**
```
cwnd
 │                              *
 │                           *
 │                         *
 │               *
 │            *
 │         *
 │      *
 │   *              * ← congestion detected, drop cwnd
 │ *           *
 │*       *
 └──────────────────────────────> time
  [SS]  [CA]     [CA]    [SS/CA]
```

**TCP BBR (Bottleneck Bandwidth and RTT) — modern alternative:**
```
BBR (Google, 2016): Model-based congestion control
- Does NOT rely on packet loss as a congestion signal
- Measures actual bandwidth and RTT to estimate bottleneck
- Sends at estimated capacity, not reaction to loss
- 2-25× throughput improvement on high-bandwidth, lossy links
- Used by Google, Netflix, YouTube
```

**System design implications:**

| Scenario | Impact of TCP Congestion Control |
|---|---|
| High latency links (satellite, 600ms RTT) | Slow start takes many RTTs to ramp up; use BBR or QUIC |
| Lossy WiFi | 3 dup ACKs triggers cwnd halving for non-packet-loss events |
| HTTP/2 over one TCP connection | Packet loss stalls ALL streams (see HTTP/3 motivation) |
| Microservice calls (short-lived TCP) | Often caught in slow-start phase; connection pooling/keep-alive is critical |
| Large file transfers | Bottleneck is often cwnd ramp-up time on first connection |

---

### Q19. What is a service mesh and how does it relate to networking in microservices?

**Answer:**

A service mesh is a dedicated infrastructure layer that handles service-to-service communication concerns — observability, security, and traffic management — transparently, without changes to application code.

**The problem a service mesh solves:**
```
Without service mesh — each service implements:
  [Service A]                [Service B]
  ├── Retry logic            ├── Retry logic
  ├── Circuit breaker        ├── Circuit breaker
  ├── Load balancing         ├── Load balancing
  ├── mTLS                   ├── mTLS
  ├── Tracing                ├── Tracing
  └── Metrics                └── Metrics
  
  → Duplicated across all services, in multiple languages
```

**Service mesh architecture (Sidecar pattern):**
```
┌─────────────────────────────────────┐
│   Pod / VM                          │
│  ┌─────────────┐  ┌──────────────┐  │
│  │  Application│  │ Sidecar Proxy│  │
│  │  (no mesh   │  │  (Envoy)     │  │
│  │   code)     │◄─►  ├─ mTLS     │  │
│  └─────────────┘  │  ├─ Retries  │  │
│                   │  ├─ Tracing  │  │
│                   │  └─ Metrics  │  │
│                   └──────────────┘  │
└─────────────────────────────────────┘
         │
         │ All traffic flows through sidecar
         │
┌─────────────────────────────────────┐
│   Control Plane (Istio / Linkerd)   │
│  ├── Certificate management (mTLS) │
│  ├── Traffic policies              │
│  ├── Service discovery             │
│  └── Telemetry collection          │
└─────────────────────────────────────┘
```

**Service mesh capabilities:**

| Capability | Description |
|---|---|
| mTLS everywhere | All service traffic encrypted and authenticated automatically |
| Distributed tracing | Trace IDs propagated across all service calls (Jaeger, Zipkin) |
| Circuit breaking | Automatic circuit breaker without code changes |
| Retries and timeouts | Configurable per route in control plane |
| Traffic splitting | 90% → v1, 10% → v2 (canary deployments) |
| Load balancing | L7 load balancing with health awareness |
| Rate limiting | Per-service rate limiting without code |
| Observability | Automatic metrics (latency, error rate, throughput) per service pair |

**Popular service meshes:**
| Product | Proxy | Strengths |
|---|---|---|
| Istio | Envoy | Feature-rich, Kubernetes-native, complex |
| Linkerd | Linkerd-proxy (Rust) | Lightweight, simple, less features |
| Consul Connect | Envoy | Multi-platform (not just K8s) |
| AWS App Mesh | Envoy | Deep AWS integration |

**Trade-offs:**
- **Benefits:** Uniform observability, security without code changes, operational consistency.
- **Costs:** CPU/memory overhead per sidecar (5-10% overhead), increased operational complexity, latency per sidecar hop (0.5-1ms).

**When to use:** Kubernetes-based microservices with > 10 services, where consistent observability and security are requirements.

---

### Q20. Compare L4 vs L7 load balancing in depth — include algorithms, use cases, and AWS equivalents.

**Answer:**

Layer 4 and Layer 7 load balancing represent fundamentally different approaches to distributing traffic. The choice affects performance, routing flexibility, and feature richness.

**L4 Load Balancing (Transport Layer):**
```
Operates on: IP addresses + TCP/UDP port numbers
Visibility:  Cannot read HTTP headers, URL paths, or content
Routing:     Based on network tuple: src IP, dest IP, src port, dest port

How it works:
  Client ──[SYN to VIP 10.0.0.1:443]──> L4 LB
  L4 LB selects backend 10.0.0.5:443 based on algorithm
  L4 LB rewrites destination IP (NAT) or uses DSR
  All packets for this TCP connection go to same backend
  L4 LB maintains connection table (src:port → backend mapping)
```

**L7 Load Balancing (Application Layer):**
```
Operates on: Full HTTP request content
Visibility:  Headers, URL path, cookies, request body, hostname
Routing:     Content-aware decisions

How it works:
  Client ──[HTTP GET /api/users]──> L7 LB
  L7 LB terminates TCP+TLS from client
  L7 LB READS the HTTP request
  Routes based on: path, header, cookie, hostname
  L7 LB opens NEW TCP connection to selected backend
  
  Example routing rules:
    /api/*         → API server pool
    /static/*      → Static server pool
    Host: admin.*  → Admin server pool
    Cookie: A/B=B  → B variant pool
```

**Algorithm comparison:**

| Algorithm | L4 | L7 | Description |
|---|---|---|---|
| Round Robin | Yes | Yes | Distribute sequentially |
| Weighted RR | Yes | Yes | Prefer higher-capacity servers |
| Least Connections | Yes | Yes | Send to server with fewest active connections |
| IP Hash | Yes | No | Hash client IP → sticky to same server |
| URL Hash | No | Yes | Same URL always goes to same backend (cache efficiency) |
| Least Response Time | No | Yes | Route to fastest-responding backend |
| Random | Yes | Yes | Random selection |

**Feature comparison:**

| Feature | L4 LB | L7 LB |
|---|---|---|
| SSL termination | No (pass-through) | Yes |
| Content-based routing | No | Yes |
| HTTP header manipulation | No | Yes |
| Cookie-based stickiness | No | Yes |
| Rate limiting by user | No | Yes |
| A/B testing routing | No | Yes |
| WebSocket support | Yes (connection passthrough) | Yes (upgrade-aware) |
| Performance | Very high (wire speed) | High (with processing overhead) |
| Latency added | ~0.1ms | ~1-5ms |

**AWS equivalents:**

| AWS Service | Layer | Key Features |
|---|---|---|
| Network Load Balancer (NLB) | L4 | Ultra-low latency, TCP/UDP, static IP, TLS passthrough |
| Application Load Balancer (ALB) | L7 | URL routing, host-based, sticky sessions, WAF integration, gRPC |
| Classic Load Balancer (CLB) | L4+L7 | Legacy, not recommended for new systems |
| Global Accelerator | L4 | Anycast, global traffic optimization, DDoS |

**Decision guide:**
```
Use L4 (NLB) when:
  - Ultra-low latency is critical (gaming, trading)
  - Non-HTTP protocols (TCP custom, UDP, gRPC direct)
  - Need static IP for whitelisting
  - TLS passthrough to backend (end-to-end encryption)

Use L7 (ALB) when:
  - Need path/host/header-based routing
  - Multiple services behind one LB (microservices)
  - A/B testing or canary deployments
  - Need WAF (Web Application Firewall) integration
  - HTTP/2 or WebSocket support with visibility
```

**Combining L4 and L7:**
```
Internet → [L4 LB (DDoS, static IP)] → [L7 LB (routing, SSL)] → Backends

Common in large deployments: NLB in front for static IP + DDoS absorption,
ALB behind for intelligent routing to microservices.
```

---

## Quick Reference

### Protocol Comparison
| Protocol | Transport | Format | Best For |
|---|---|---|---|
| REST | HTTP | JSON | Public APIs |
| GraphQL | HTTP | JSON | Complex client queries |
| gRPC | HTTP/2 | Protobuf | Internal microservices |
| WebSocket | TCP | Binary/Text | Real-time bidirectional |
| SSE | HTTP | Text | Server push (unidirectional) |

### HTTP Version Summary
| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 |
|---|---|---|---|
| Transport | TCP | TCP | QUIC/UDP |
| Multiplexing | No | Yes | Yes |
| HOL blocking | App+TCP | TCP only | None |
| Header compression | No | HPACK | QPACK |
| 0-RTT resumption | No | No | Yes |

### Load Balancer Decision
| Need | Use |
|---|---|
| Path-based routing | L7 (ALB) |
| Ultra-low latency | L4 (NLB) |
| Static IP | L4 (NLB) |
| SSL termination | L7 (ALB) |
| DDoS + global | Anycast + L4 |
| Service-to-service | Service Mesh |

### DNS Record Types
| Record | Use |
|---|---|
| A | IPv4 address mapping |
| CNAME | Alias (e.g., www → root domain) |
| MX | Mail server |
| TXT | SPF, DKIM verification |
| SRV | Service discovery |

### Networking Latency Reference
| Operation | Latency |
|---|---|
| DNS lookup (cached) | 0 ms |
| DNS lookup (uncached) | 20–120 ms |
| TCP handshake (same DC) | ~0.5 ms |
| TCP + TLS (US–EU) | ~200 ms |
| HTTP/2 with multiplexing | ~30 ms (RTT) |
| WebSocket message | ~1 ms (same DC) |
