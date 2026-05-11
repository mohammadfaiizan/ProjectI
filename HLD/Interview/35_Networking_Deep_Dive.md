# 35 — Networking Deep Dive

---

## Easy (Q1–Q7)

---

### Q1. Describe the TCP three-way handshake and why connection establishment adds latency.

**Answer:**

The TCP three-way handshake is the process by which a client and server establish a reliable connection before any data is exchanged. It takes one full round-trip time (RTT) before the first byte of application data can be sent.

**Handshake sequence:**
```
Client                      Server
   │                           │
   │──── SYN (seq=x) ─────────►│  Client: "I want to connect, my seq starts at x"
   │                           │
   │◄─── SYN-ACK (seq=y, ack=x+1)│  Server: "OK, my seq starts at y, I got your x"
   │                           │
   │──── ACK (ack=y+1) ────────►│  Client: "I got your y, connection established"
   │                           │
   │──── [HTTP GET /] ─────────►│  ← First data after 1 RTT of handshake overhead
```

**Latency cost:**
```
Geographic example:
  Client (New York) → Server (London) RTT: 80ms
  
  Handshake cost: 1 RTT = 80ms
  TLS 1.2 + TCP: 1 RTT (TCP) + 2 RTT (TLS) = 3 RTT = 240ms before first byte
  TLS 1.3 + TCP: 1 RTT (TCP) + 1 RTT (TLS) = 2 RTT = 160ms
  HTTP/3 (QUIC): 1 RTT total (TCP + TLS combined)
  HTTP/3 0-RTT: 0ms if session already established
```

**SYN fields established during handshake:**
- Initial sequence numbers (ISN) for both sides — random for security
- Maximum Segment Size (MSS) — max data per TCP segment
- Window scale factor — for large throughput
- Selective ACK (SACK) support

**Reducing handshake overhead:**
```
1. Connection pooling: reuse established connections
   HTTP keep-alive: single connection for multiple requests
   Database pools: pre-established DB connections

2. TCP Fast Open (TFO):
   First connect: normal handshake + TFO cookie stored
   Subsequent connects: send data IN the SYN packet (0 RTT)
   Supported by Linux, nginx, Chrome
   
3. Persistent connections:
   REST: Connection: keep-alive header
   gRPC: HTTP/2 multiplexes on one connection (many RPCs, one handshake)
```

The handshake is non-negotiable for new TCP connections. The best optimization is to not do it — reuse connections wherever possible.

---

### Q2. Explain TCP slow start and congestion control. What are CWND and SSTHRESH?

**Answer:**

TCP congestion control prevents the network from being overwhelmed by gradually ramping up the data rate and backing off when congestion is detected.

**CWND (Congestion Window):** How many bytes the sender can have "in flight" (sent but not yet acknowledged) at any time. This is the sender-side limit.

**SSTHRESH (Slow Start Threshold):** The point at which TCP switches from aggressive growth (slow start) to conservative growth (congestion avoidance).

**Slow start phase:**
```
Initially: CWND = 1 MSS (Maximum Segment Size, typically 1460 bytes)
Each ACK received: CWND doubles

RTT 1: CWND = 1 MSS → send 1 segment
RTT 2: CWND = 2 MSS → send 2 segments (1 ACK received → +1)
RTT 3: CWND = 4 MSS → send 4 segments
RTT 4: CWND = 8 MSS → send 8 segments
...
RTT n: CWND = 2^n MSS

Doubles every RTT until CWND reaches SSTHRESH (or loss detected)
"Slow start" is misleading — it's exponential growth, but starts slow
```

**Congestion avoidance phase (after reaching SSTHRESH):**
```
CWND increases by 1 MSS per RTT (linear growth)
  RTT 5: CWND = 17, 18, 19, 20... (one per round trip)
```

**Congestion event (packet loss detected):**
```
On timeout (severe loss):
  SSTHRESH = CWND / 2
  CWND = 1 MSS (restart slow start!)
  Result: Very aggressive recovery, potential stall

On triple duplicate ACK (mild loss, fast retransmit):
  SSTHRESH = CWND / 2
  CWND = SSTHRESH (skip slow start, enter congestion avoidance)
```

**Visual timeline:**
```
CWND
16 │           *
   │          * *
8  │         *   *
   │        *     *
4  │       *       *───────────
2  │      *         (CA phase)
1  │─────*
   └──────────────────────────── Time
        SS    SSTHRESH
```

**Modern algorithms:**

| Algorithm | Behavior | Best For |
|-----------|----------|----------|
| TCP Cubic (Linux default) | Aggressive recovery based on elapsed time | High-bandwidth, high-latency |
| BBR (Google) | Model-based, maintains target BDP | Long-distance, variable loss |
| RENO | Classic, loss-based | Short distances, low loss |

BBR (Bottleneck Bandwidth and RTT) is now used by Google, YouTube, and Cloudflare. It estimates available bandwidth and keeps sending at that rate rather than reacting to loss events, significantly improving throughput on modern networks.

---

### Q3. What is HTTP keep-alive and connection reuse? Why does it matter for latency?

**Answer:**

HTTP keep-alive allows multiple request-response pairs to be sent over a single TCP connection, avoiding the cost of establishing a new TCP connection for every request.

**Without keep-alive (HTTP/1.0):**
```
Request 1:
  TCP SYN ──────────────────────────► (1 RTT handshake)
  HTTP GET /index.html ─────────────►
              ◄────────────────────── 200 OK + HTML
  TCP FIN ──────────────────────────► (connection closed)

Request 2 (for /style.css):
  TCP SYN ──────────────────────────► (another 1 RTT handshake!)
  HTTP GET /style.css ──────────────►
              ◄────────────────────── 200 OK + CSS
  TCP FIN ──────────────────────────►

Cost: 2 × (1 RTT TCP handshake) = extra 2 RTTs wasted on overhead
For 50 resources on a page: 50 × RTT wasted
```

**With keep-alive (HTTP/1.1 default):**
```
TCP SYN ──────────────────────────► (1 RTT handshake — once)
HTTP GET /index.html ─────────────►
        ◄────────────────────────── 200 OK + HTML
HTTP GET /style.css ──────────────►  ← same connection reused
        ◄────────────────────────── 200 OK + CSS
HTTP GET /app.js ─────────────────►  ← same connection again
        ◄────────────────────────── 200 OK + JS
...50 requests, 1 TCP handshake
TCP FIN (when idle timeout reached)
```

**Problem with keep-alive: Head-of-line blocking**
```
HTTP/1.1: requests are sequential on each connection
  Request 1: large image (200KB, 500ms) ← blocks
  Request 2: tiny CSS (5KB, 10ms) ← waiting for image to finish

Workaround: open 6 parallel connections per domain (browser default)
  But: 6 × TCP handshake cost, 6× memory overhead
```

**HTTP/2 multiplexing (better solution):**
```
HTTP/2: multiple streams on ONE connection, no blocking
  Stream 1 → GET /image.jpg (in progress)
  Stream 2 → GET /style.css (returns immediately, not blocked)
  Stream 3 → GET /app.js    (returns immediately)
  All on same TCP connection
```

**Tuning keep-alive parameters:**
```nginx
# Nginx: keep-alive to upstream servers
upstream backend {
    server 10.0.1.1:8080;
    keepalive 32;  # Max 32 idle connections in pool per worker
}

http {
    keepalive_timeout 65s;     # Close idle connection after 65s
    keepalive_requests 1000;   # Close after 1000 requests (prevents memory leaks)
}
```

For database connections: connection pooling (PgBouncer, HikariCP) is the database equivalent — always enabled in production.

---

### Q4. What is HTTP/2 multiplexing and how does it solve head-of-line blocking at L7 (but not at L4)?

**Answer:**

HTTP/2 introduces multiplexing: sending multiple requests and responses concurrently over a single TCP connection, eliminating HTTP/1.1's sequential (head-of-line) blocking.

**HTTP/1.1 head-of-line blocking:**
```
Connection 1: [req1: 500ms large response ] [req2: 10ms]   req2 waits!
Connection 2: [req3: 10ms] 
Connection 3: [req4: 20ms]

Max 6 connections, so 7th request waits for a connection to free up
```

**HTTP/2 multiplexing (L7 solution):**
```
Single TCP connection with multiple streams:
  
  Stream 1 frames: [HDR][DATA][DATA][DATA...] ← large download
  Stream 2 frames: [HDR][DATA]               ← fast CSS, returned immediately
  Stream 3 frames: [HDR][DATA]               ← fast JS, returned immediately
  
  All interleaved on same connection:
  [S2:HDR][S1:HDR][S3:HDR][S2:DATA][S3:DATA][S1:DATA][S1:DATA]...
  
  S2 and S3 complete in 10ms even while S1 is still streaming
```

**HTTP/2 stream prioritization:**
```
Browser can signal: "render-critical CSS (S2) > background image (S1)"
  S2 weight: 256 (high)
  S1 weight: 32  (low)
Server allocates bandwidth proportionally
```

**Why TCP head-of-line blocking (L4) still exists:**
```
HTTP/2 runs over TCP. TCP guarantees in-order delivery.

If a TCP packet is lost:
  Packet: [S2:DATA][S1:DATA] [LOST: S3:DATA][S2:DATA][S1:DATA]
  
TCP requires: retransmit lost packet, then deliver subsequent packets in order
  → ALL streams stall waiting for the retransmission
  → HTTP/2's stream multiplexing cannot help — it's below TCP's layer
  
HTTP/2 over TCP: Solves L7 HoL blocking (sequential HTTP requests)
                 Does NOT solve L4 HoL blocking (TCP retransmission)
```

**HTTP/3 solution (QUIC, no L4 HoL blocking):**
```
QUIC runs over UDP, implements its own transport layer
  Each stream is independently acknowledged
  Lost UDP packet: only the affected stream stalls
  Other streams continue unaffected
  
HTTP/3 = HTTP/2 semantics + QUIC transport (no TCP HoL blocking)
```

**When HTTP/2 multiplexing matters most:**
- Pages with many small resources (CSS, JS, images)
- High-latency connections (mobile networks)
- APIs making many small parallel requests

---

### Q5. How does DNS caching work at multiple levels and what are the TTL implications?

**Answer:**

DNS resolution involves a chain of caches, each with its own TTL (Time To Live). Understanding this chain is critical for cache invalidation, CDN behavior, and incident response.

**The DNS resolution chain:**
```
Browser                 OS              Resolver              Authoritative
  Cache                Cache             Cache                  DNS Server
(minutes)           (seconds)          (hours)               (source of truth)
    │                   │                  │                       │
    │ 1. Check browser cache               │                       │
    │   (cached? return. TTL expired?)     │                       │
    │──────────────────►│                  │                       │
    │ 2. Check OS cache (nscd, systemd-resolved)                   │
    │   (/etc/hosts checked first)         │                       │
    │──────────────────────────────────────►│                      │
    │ 3. Check resolver cache              │                       │
    │   (ISP resolver, 8.8.8.8, 1.1.1.1)  │                       │
    │──────────────────────────────────────────────────────────────►│
    │ 4. Authoritative server responds     │  (if resolver cache miss)
    │ TTL from here propagates back        │                       │
    │◄─────────────────────────────────────────────────────────────│
```

**TTL implications:**

| Layer | Typical TTL | Implication |
|-------|------------|-------------|
| Browser cache | 1-60 minutes (min of TTL) | Long TTL = stale after failover |
| OS cache | Matches DNS TTL | Can be flushed with `systemd-resolve --flush-caches` |
| Resolver cache | Matches DNS TTL (may cap at max) | ISP may ignore TTL minimums |
| Authoritative | Set by you (TTL field) | Source of truth |

**TTL trade-offs:**
```
High TTL (3600s = 1 hour):
  Pro: Fewer DNS queries → faster resolution (cache hits)
  Pro: Less load on authoritative servers
  Con: Change takes 1 hour to propagate (e.g., IP change during incident)

Low TTL (30-60s):
  Pro: Changes propagate quickly (important for failover, CDN switching)
  Con: More DNS queries → more latency on cache miss
  Con: More load on resolvers

Disaster recovery practice:
  Normal: TTL = 3600s
  Before planned migration: lower to TTL = 60s
  After migration: raise back to 3600s
```

**DNS record types relevant to system design:**

| Record | Purpose | Example |
|--------|---------|---------|
| A | IPv4 address | api.example.com → 1.2.3.4 |
| AAAA | IPv6 address | api.example.com → 2001:db8::1 |
| CNAME | Alias to another name | www → api.example.com |
| MX | Mail server | example.com → mail.example.com |
| TXT | Verification, SPF/DKIM | "v=spf1 include:sendgrid..." |
| SRV | Service discovery | _grpc._tcp.service → host:port |
| NS | Nameserver delegation | example.com → ns1.route53.com |

**DNS negative caching (NXDOMAIN):**
```
If DNS query returns NXDOMAIN (not found):
  Negative TTL cached (from SOA record's minimum field)
  Typically 300-3600 seconds
  Impact: If you deploy new service, DNS propagation still takes time
          Cannot instantly create new DNS record and have it work
```

---

### Q6. What is Anycast routing and how does it route one IP to different servers in different regions?

**Answer:**

Anycast is a routing method where the same IP address is announced from multiple geographic locations. The network (BGP routing) automatically routes packets to the topologically nearest node advertising that IP.

**How Anycast works:**
```
Cloudflare announces 1.1.1.1 from:
  Data center in New York     (AS13335: prefix 1.1.1.0/24)
  Data center in London       (AS13335: prefix 1.1.1.0/24)
  Data center in Tokyo        (AS13335: prefix 1.1.1.0/24)
  Data center in Singapore    (AS13335: prefix 1.1.1.0/24)

User in London queries 1.1.1.1:
  BGP routing table: 1.1.1.0/24 is reachable via London DC (2 hops)
                     or via New York DC (15 hops)
  → BGP selects shorter path → London DC receives the packet
  
User in Tokyo queries 1.1.1.1:
  BGP routing table: → Tokyo DC (nearest)
  
Same IP: 1.1.1.1 — but geographically different servers handle each user
```

**CDN use of Anycast:**
```
Cloudflare / Akamai / Fastly:
  Edge nodes worldwide all announce the same IP ranges
  Client connects to "closest" edge node (by BGP distance)
  Edge serves cached content or proxies to origin

Benefits:
  1. Low latency: client connects to nearest server (not origin in US)
  2. DDoS resilience: attack traffic absorbed at many edges simultaneously
     (A 1 Tbps DDoS is diluted across 200 PoPs = 5 Gbps per PoP)
  3. Automatic failover: if London DC goes down, BGP withdraws its route
     → traffic automatically reroutes to Frankfurt DC
     No DNS TTL delay, no manual intervention
```

**Anycast vs Unicast vs Multicast:**

| Method | Routing | Receiver Count | Use Case |
|--------|---------|----------------|----------|
| Unicast | One specific address | One | Normal web traffic |
| Anycast | Many nodes, same IP | Nearest one | DNS, CDN, DDoS mitigation |
| Multicast | Group subscription | All subscribed | Video streaming (IPTV) |
| Broadcast | All on subnet | All | ARP, DHCP discovery |

**Why DNS servers use Anycast:**
```
8.8.8.8 (Google's DNS) is Anycast:
  query from Sydney → Sydney Google PoP
  query from Paris → Paris Google PoP
  
If one PoP fails, no DNS configuration change needed
BGP automatically routes around the failure
```

---

### Q7. How does NAT (Network Address Translation) work and why does it complicate P2P connections?

**Answer:**

NAT allows multiple devices with private IP addresses to share a single public IP address by translating addresses and ports at the gateway.

**How NAT works:**
```
Private Network (behind router):
  Device A: 192.168.1.10:50000
  Device B: 192.168.1.11:50001
  
NAT Router:
  Public IP: 203.0.113.1 (one public IP for all devices)
  NAT Table:
    203.0.113.1:8001 ↔ 192.168.1.10:50000
    203.0.113.1:8002 ↔ 192.168.1.11:50001

Device A sends: src=192.168.1.10:50000, dst=8.8.8.8:80
NAT rewrites:   src=203.0.113.1:8001,  dst=8.8.8.8:80

Google responds: src=8.8.8.8:80, dst=203.0.113.1:8001
NAT rewrites:    src=8.8.8.8:80, dst=192.168.1.10:50000 (lookup NAT table)
```

**Why NAT breaks P2P:**
```
Device A (behind NAT A): real IP=192.168.1.10, public IP=203.0.113.1
Device B (behind NAT B): real IP=192.168.1.20, public IP=198.51.100.1

A wants to connect to B directly:
  A sends to 198.51.100.1 → NAT B receives it
  NAT B: "is 203.0.113.1 in my NAT table?" → NO
  NAT B drops the packet (unsolicited inbound → blocked)
  
  B tries simultaneously to reach A → same problem
  
  Neither can initiate because neither NAT has a hole-punch entry
```

**WebRTC NAT traversal (ICE/STUN/TURN):**
```
STUN (Session Traversal Utilities for NAT):
  A asks STUN server: "What is my public IP:port?"
  STUN: "You appear as 203.0.113.1:8001"
  A shares this with B via signaling server (HTTP/WebSocket)
  
  B also discovers its public endpoint: 198.51.100.1:9001
  
  Hole punching:
  A → sends to B's public endpoint (simultaneously with...)
  B → sends to A's public endpoint
  
  Both NATs see "outbound" traffic → punch holes → P2P connection!

TURN (Traversal Using Relays around NAT) — fallback:
  If hole punching fails (symmetric NAT):
  A → TURN server → B (relay, not P2P)
  Cost: TURN server handles all traffic
  Use: < 15% of WebRTC calls (most use STUN/direct P2P)
```

**IPv6 eliminates NAT:**
IPv6 provides enough addresses (3.4 × 10^38) for every device to have a public IP. No NAT needed, so P2P connections are trivial. This is one of the major IPv6 benefits for application developers.

---

## Medium (Q8–Q15)

---

### Q8. Explain QUIC and HTTP/3. Why does HTTP/3 use UDP and how does it reimplement reliability?

**Answer:**

QUIC (Quick UDP Internet Connections) is a transport protocol developed by Google, now standardized by IETF. HTTP/3 runs over QUIC instead of TCP, solving the TCP head-of-line blocking problem at the transport layer.

**Why UDP as the base:**
```
TCP is in the kernel — cannot be easily modified:
  - TCP's HoL blocking is structural (guaranteed in-order delivery)
  - Changing TCP requires OS kernel update on every device in the world
  - Takes decades for universal deployment

UDP: Just sends datagrams, no ordering, no reliability
  - Application layer (QUIC) can implement exactly the reliability it needs
  - QUIC is in user space → deployable with software updates (no kernel change needed)
  - QUIC can implement stream-level ordering (not connection-level)
```

**How QUIC reimplements reliability:**
```
QUIC connections: identified by 64-bit Connection ID (not IP:port)
  → Mobile users can switch WiFi → cellular without reconnecting
  → IP changes don't break QUIC connections (TCP breaks)

QUIC reliability (per stream):
  Sender: assigns packet number to each UDP datagram (never reused)
  Receiver: sends ACK ranges (like TCP SACK, always enabled)
  Sender: retransmits if ACK not received within timeout
  
  Critical: loss of packet on Stream 1 → only Stream 1 retransmits
            Stream 2 and Stream 3 continue unaffected
            (Unlike TCP: loss blocks ALL streams)
```

**QUIC vs TCP comparison:**

| Property | TCP | QUIC |
|----------|-----|------|
| Transport | L4 kernel protocol | L4 user-space over UDP |
| HoL blocking | Connection-level (all streams blocked) | Stream-level (only affected stream) |
| Connection establishment | 1 RTT (3-way handshake) | 1 RTT (combined with TLS 1.3) |
| Resumption | TLS session resumption | 0-RTT for known servers |
| Migration | Breaks on IP change | Survives IP change (Connection ID) |
| Header | Fixed (40 bytes) | Variable, encrypted |
| Middlebox compatibility | Universal | Firewalls may block UDP 443 |

**HTTP/3 handshake (0 RTT possible):**
```
First connection:
  0ms: Client → QUIC Initial (crypto hello)
  50ms: Server → QUIC Handshake (cert, QUIC params)
  100ms: Client → Handshake complete + first HTTP/3 request
  150ms: Server → HTTP/3 response received
  Total: 1.5 RTT (better than TCP+TLS 1.3 = 2 RTT)

Subsequent connection (0-RTT):
  0ms: Client → QUIC Initial with 0-RTT data (HTTP request included!)
  50ms: Server → Response
  Total: 0.5 RTT (TCP: 2 RTT minimum)
```

**Real-world QUIC performance:**
- Google measured 3% improvement in YouTube rebuffer rate
- Google Search: 8% reduction in mean page load time on mobile
- Most improvement seen on lossy networks (mobile, congested WiFi)

---

### Q9. Describe TLS 1.3 improvements: 1-RTT handshake, 0-RTT resumption, and forward secrecy.

**Answer:**

TLS 1.3 (RFC 8446, 2018) dramatically simplifies and improves TLS/SSL, reducing handshake latency and removing insecure cipher suites.

**TLS 1.2 handshake (2 RTTs):**
```
Client                              Server
  │──── ClientHello ───────────────►│  RTT 1: hello + cipher negotiation
  │◄─── ServerHello + Certificate──│
  │◄─── ServerHelloDone ───────────│
  │                                 │
  │──── ClientKeyExchange ─────────►│  RTT 2: key exchange
  │──── ChangeCipherSpec ──────────►│
  │──── Finished ──────────────────►│
  │◄─── ChangeCipherSpec ──────────│
  │◄─── Finished ──────────────────│
  │                                 │
  │──── [HTTP GET /] ──────────────►│  ← Data starts after 2 RTT
```

**TLS 1.3 handshake (1 RTT):**
```
Client                              Server
  │──── ClientHello ───────────────►│  Includes key_share in hello (Diffie-Hellman params)
  │    (+ supported cipher suites)  │  Server can compute shared secret immediately
  │◄─── ServerHello ───────────────│
  │◄─── Certificate + CertVerify ──│  Server sends all this in ONE round trip
  │◄─── Finished ──────────────────│
  │                                 │
  │──── Finished ──────────────────►│
  │──── [HTTP GET /] ──────────────►│  ← Data starts after 1 RTT (vs 2 in TLS 1.2!)
```

**0-RTT resumption (for returning clients):**
```
Client (has session ticket from previous visit)
  │──── ClientHello + 0-RTT data ──►│  Send HTTP request WITH the hello!
  │    (uses Pre-Shared Key)         │
  │◄─── ServerHello + Response ─────│
  
0 RTTs before data! But...
```

**0-RTT security warning (replay attacks):**
```
0-RTT data is not forward secret and is replayable:
  Attacker captures: ClientHello + 0-RTT {POST /payment, amount=100}
  Attacker replays: resends same bytes
  Server cannot distinguish replay from legitimate request
  
Mitigation:
  Use 0-RTT ONLY for idempotent, non-sensitive requests (GET /homepage)
  Never use 0-RTT for POST/PUT with side effects
  Server can optionally reject 0-RTT (Early-Data: 1 header)
```

**Forward secrecy:**
```
TLS 1.2 without PFS (RSA key exchange):
  Server has one long-term private key
  Client sends: encrypted_premaster_secret (encrypted with server's public key)
  
  If attacker:
    Records all past traffic
    Later steals server's private key (breach, subpoena)
    → Decrypts ALL past recorded traffic!
  
TLS 1.3 (ephemeral Diffie-Hellman — always):
  New key pair generated PER SESSION (ephemeral key)
  Session keys deleted after session ends
  
  Attacker steals server's long-term key later:
  → Cannot decrypt past traffic (session keys were deleted)
  → Each session's security is independent = Forward Secrecy
```

**Removed from TLS 1.3:**
- RSA key exchange (no forward secrecy)
- DHE with static params
- RC4, DES, 3DES, MD5, SHA-1
- Compression (CRIME attack)
- Renegotiation

---

### Q10. How does BGP work? What is BGP hijacking?

**Answer:**

BGP (Border Gateway Protocol) is the protocol that routes traffic between autonomous systems (AS) on the internet — effectively the routing protocol of the internet itself.

**Internet structure:**
```
Internet = thousands of Autonomous Systems (AS)
  AS = network operated by one organization under one routing policy
  
Examples:
  AS15169: Google
  AS16509: Amazon AWS
  AS20940: Akamai
  AS701:   Verizon
  AS3356:  Lumen (CenturyLink)
```

**How BGP works:**
```
Each AS announces its IP prefixes to neighboring ASes:
  Google (AS15169) announces: 8.8.0.0/24, 172.217.0.0/16, 64.233.0.0/16
  
  Google tells its neighbors (Verizon, Comcast, etc.):
  "To reach 8.8.0.0/24, send traffic to me (AS15169)"
  
  Verizon tells its customers:
  "To reach 8.8.0.0/24, send to Verizon → Google"
  
  Path: AS701 → AS15169 (for 8.8.0.0/24)

BGP selects "best path" using:
  1. Shortest AS path (fewest ASes to cross)
  2. Local preference (business relationships)
  3. MED (Multi-Exit Discriminator) when multiple paths exist
```

**BGP hijacking:**
```
Legitimate: AS15169 announces 8.8.0.0/24 (Google DNS)
  Internet routes queries for 8.8.8.8 to Google

Attack: Malicious AS (e.g., AS666) announces 8.8.0.0/24 with forged route
  OR announces a more specific prefix: 8.8.8.0/25 (longer prefix wins in BGP)
  
  BGP rule: More specific prefix wins
  8.8.8.0/25 beats 8.8.0.0/24 (25-bit vs 24-bit mask)
  
  Result: Some routers now send traffic for 8.8.8.8 → Malicious AS
  → Traffic intercepted (MitM) or blackholed

Famous incidents:
  2010: Pakistan Telecom hijacked YouTube globally for 2 hours
  2018: Amazon Route53 DNS hijacked to steal $150k in cryptocurrency
  2019: European traffic routed through China Telecom for 2 hours
```

**BGP security mitigations:**
```
1. RPKI (Resource Public Key Infrastructure):
   ASes cryptographically sign their IP prefix announcements
   ROA (Route Origin Authorization): "only AS15169 can announce 8.8.0.0/24"
   Validators reject cryptographically invalid routes
   Deployed by: Cloudflare, Amazon, Google, RIPE (partially)

2. BGPsec:
   Signs each AS-PATH hop
   Prevents path manipulation (harder, less deployed than RPKI)

3. Filtering at IXPs:
   Internet Exchange Points filter obviously bogus routes
   IRR (Internet Routing Registry) provides expected routes for filtering
```

---

### Q11. What is the TIME_WAIT TCP state and why does it matter for high-connection-rate services?

**Answer:**

TIME_WAIT is a TCP state that a connection enters after it initiates the active close (sends the first FIN). It lasts 2 × MSL (Maximum Segment Lifetime), typically 60 seconds on Linux.

**Purpose of TIME_WAIT:**
```
TCP 4-way close:
  Closer (C) ──── FIN ─────────────────────►  (Active close)
  Remote (R) ◄─── ACK ────────────────────────
  Remote (R) ──── FIN ────────────────────────►
  Closer (C) ◄─── ACK ─────────────────────── 
  [C enters TIME_WAIT for 2×MSL = 60 seconds]

Why TIME_WAIT is needed:
  1. Ensure the final ACK was received:
     If the last ACK is lost, Remote retransmits FIN
     TIME_WAIT allows Closer to retransmit ACK
     
  2. Prevent old duplicate packets from corrupting new connections:
     Delayed packets from old connection must expire before same 
     (src_ip, src_port, dst_ip, dst_port) tuple can be reused
     2×MSL ensures all packets from old connection are gone
```

**The high-connection-rate problem:**
```
Each short-lived connection occupies TIME_WAIT for 60 seconds

Load balancer handling 10,000 connections/second:
  After 60 seconds: 10,000 × 60 = 600,000 sockets in TIME_WAIT
  Each socket: ~3KB memory
  Total: 600,000 × 3KB = ~1.8GB RAM just for TIME_WAIT sockets!
  
  Worse: Port exhaustion
  Ephemeral port range: 32,768 – 60,999 = ~28,000 ports
  If all used by TIME_WAIT sockets: "Address already in use" errors
  New connections fail
```

**Mitigations:**

**1. SO_REUSEADDR and tcp_tw_reuse:**
```bash
# Allow reuse of TIME_WAIT sockets for new connections
# (Safe: uses timestamps to prevent old packet confusion)
sysctl -w net.ipv4.tcp_tw_reuse=1
```

**2. Reduce TIME_WAIT duration:**
```bash
# Not recommended for internet-facing; packet lifetime must be < MSL
# Internal services only
sysctl -w net.ipv4.tcp_fin_timeout=10  # 10s instead of 60s
```

**3. Connection pooling (best solution):**
```
Reuse existing connections instead of creating new ones
TIME_WAIT only created on close — if connections are never closed, no TIME_WAIT

nginx: keepalive 32; (pool of 32 connections per upstream)
databases: connection pool of 20 persistent connections
gRPC: single long-lived HTTP/2 connection, no TIME_WAIT issue
```

**4. Increase ephemeral port range:**
```bash
sysctl -w net.ipv4.ip_local_port_range="1024 65535"
# 64,511 ports instead of ~28,000
```

**5. SO_REUSEPORT (for servers):**
```bash
# Multiple processes bind to same port (Linux 3.9+)
# Each process accepts connections independently
# Reduces contention on accept() syscall
sysctl -w net.core.somaxconn=65535
# Application uses SO_REUSEPORT socket option
```

---

### Q12. Explain how SSL termination at a load balancer differs from end-to-end TLS.

**Answer:**

SSL/TLS can be terminated at different layers, each with different security and performance trade-offs.

**SSL termination at load balancer:**
```
Internet
    │ (HTTPS — encrypted)
    ▼
Load Balancer (AWS ALB / Nginx / HAProxy)
    │ [Decrypts here: has SSL certificate and private key]
    │ (HTTP — plain text, or re-encrypted HTTPS on internal network)
    ▼
Backend Servers (no SSL cert needed)
```

**End-to-end TLS (SSL passthrough):**
```
Internet
    │ (HTTPS — encrypted)
    ▼
Load Balancer (layer 4 only — forwards encrypted bytes blindly)
    │ (HTTPS — still encrypted, LB cannot inspect)
    ▼
Backend Servers (each has SSL certificate, terminates TLS)
```

**Detailed comparison:**

| Property | Termination at LB | End-to-End TLS |
|----------|------------------|----------------|
| Certificate location | Only on LB | On each backend server |
| LB can inspect HTTP | Yes (headers, cookies, path) | No (encrypted passthrough) |
| HTTP-based routing (host header, path) | Yes | No (L4 only) |
| WAF / DDoS protection | Yes (LB sees plaintext) | Limited |
| Internal traffic | Plaintext (trust internal) | Encrypted |
| Compliance (HIPAA, PCI) | May not satisfy | Satisfies "in transit" requirement |
| CPU cost | Centralized on LB | Distributed to backends |
| Certificate management | One place | All backend servers |
| Perfect forward secrecy | Yes (LB) | Yes (backends) |

**Security consideration — network trust zone:**
```
Termination at LB (secure if):
  - Internal network is trusted (private VPC, service mesh mTLS)
  - Backend-to-LB communication uses mTLS (Istio)
  - Backend servers are not accessible from outside

End-to-end TLS required when:
  - Internal network cannot be trusted (zero-trust architecture)
  - Regulatory requirement specifies data must be encrypted in transit
    at ALL hops (PCI DSS 4.0, some HIPAA interpretations)
  - Multi-tenant environment where LB operator is different from backend operator
```

**Hybrid (re-encryption):**
```
Client → [HTTPS] → LB → [HTTPS with internal cert] → Backend

LB terminates external TLS, re-encrypts with internal cert
Benefits: LB can inspect (for routing, WAF), but traffic never travels plaintext
Used by: Google internal services, most cloud providers internally
```

**mTLS with service mesh (best practice):**
```
Istio Envoy sidecar handles all TLS:
  Client app → plain HTTP → Envoy sidecar → mTLS → Envoy sidecar → Server app
  
  Application code has no TLS logic
  Automatic certificate rotation via Istiod (Citadel)
  Both sides authenticated (mutual TLS)
```

---

### Q13. Compare TCP and UDP for real-time applications. Which protocols choose each and why?

**Answer:**

Real-time applications face a fundamental tension: reliability (TCP) vs low latency (UDP). The choice shapes the entire application architecture.

**TCP characteristics:**
```
Reliable: All bytes delivered, in order
Ordered:  Byte 1 before Byte 2 before Byte 3
Congestion control: Backs off under network stress
Connection-oriented: Handshake required
Use: When correctness > latency
```

**UDP characteristics:**
```
Unreliable: Packets may be lost, no retransmission
Unordered:  Packets may arrive out of order
No flow control: Send at any rate
Connectionless: No handshake
Use: When latency > correctness
```

**Real-time application trade-off matrix:**

| Application | Tolerance | Lost Data Effect | Choose |
|-------------|-----------|-----------------|--------|
| Web browsing | High latency ok | Missing byte = broken page | TCP |
| Video streaming (YouTube) | Buffered, can wait | Brief glitch | TCP (QUIC) |
| Online gaming | Can tolerate loss | Player teleports | UDP |
| VoIP (Zoom, Teams) | 150ms max | Crackle | UDP (RTP) |
| Live video (Twitch) | 5-10s buffer | Pause | TCP (RTMP/HLS) |
| Video conferencing | 400ms max | Artifact | UDP (WebRTC) |
| DNS queries | One-shot | Retry at app layer | UDP |
| File transfer | Correctness | Corruption | TCP |

**Why gaming and VoIP use UDP:**
```
Online game: player position updates at 60 Hz (every 16ms)
  If packet arrives late: the position it described is already outdated
  Better to skip it (UDP) than to wait for TCP retransmission (blocks newer updates)
  
VoIP at 20ms packet intervals:
  Late packet (>80ms): audio jitter buffer discards it anyway
  Retransmitting a 50ms old audio packet = worse than silence (jitter buffer fills)
  UDP: just send new packets, ignore lost ones
```

**Application-level reliability on UDP:**
```python
# RTP (Real-Time Protocol) sequence numbers: 
# detect loss without TCP retransmission
class RTPPacket:
    seq_num: int        # Sequence number
    timestamp: int      # For jitter compensation
    ssrc: int           # Source identifier
    payload: bytes      # Audio/video data

# Receiver: detect gaps in sequence numbers
def process_rtp(packet: RTPPacket):
    if packet.seq_num == expected_seq:
        decode(packet.payload)
        expected_seq += 1
    elif packet.seq_num > expected_seq:
        # Gap detected: packet lost
        use_concealment()  # Noise fill, extrapolation
        expected_seq = packet.seq_num + 1
    else:
        # Duplicate or out-of-order: discard
        pass
```

**QUIC: the best of both worlds**
```
QUIC = UDP base + reliability per stream
  Stream 1 (video): reliable, ordered within stream
  Stream 2 (audio): reliable, ordered within stream
  
  Loss on video stream → only video pauses (audio continues)
  TCP: loss anywhere → everything pauses
```

---

### Q14. How does a load balancer pass the client IP to the backend? Explain X-Forwarded-For and PROXY protocol.

**Answer:**

When a load balancer terminates a connection and opens a new one to the backend, the backend sees the load balancer's IP as the "client." Passing the real client IP is critical for rate limiting, geolocation, logging, and security.

**The problem:**
```
Client (1.2.3.4) → Load Balancer (10.0.0.1) → Backend Server

Backend sees: connection from 10.0.0.1 (the LB's internal IP)
Backend logs: "request from 10.0.0.1" — useless for security/analytics
```

**Solution 1: X-Forwarded-For HTTP header**
```
Load Balancer adds header before forwarding:
  X-Forwarded-For: 1.2.3.4

If request passes through multiple proxies:
  Original client: 1.2.3.4
  First proxy (CDN): adds X-Forwarded-For: 1.2.3.4
  Second proxy (LB): adds to list: X-Forwarded-For: 1.2.3.4, 203.0.113.1
  
Backend reads: X-Forwarded-For: 1.2.3.4, 203.0.113.1
  First value (leftmost) = original client
  Each subsequent value = proxy that added itself
```

**Reading XFF safely:**
```python
def get_real_client_ip(request):
    xff = request.headers.get("X-Forwarded-For", "")
    
    if xff:
        # Danger: anyone can forge this header
        # Only trust IPs added by YOUR infrastructure
        # Count backwards from right (rightmost = added by your LB = trusted)
        ips = [ip.strip() for ip in xff.split(",")]
        
        # If behind 2 trusted proxies: take 3rd from right
        num_trusted_proxies = 2
        trusted_index = -(num_trusted_proxies + 1)
        
        return ips[trusted_index] if len(ips) > num_trusted_proxies else ips[0]
    
    return request.remote_addr

# Alternative: use X-Real-IP (nginx sets this to original client IP)
# More reliable than parsing XFF chain
```

**Solution 2: PROXY Protocol (TCP-level, more robust)**
```
Works at L4 — before any HTTP parsing
Load Balancer prepends PROXY protocol header to TCP stream:
  PROXY TCP4 1.2.3.4 10.0.0.1 56789 80\r\n
  [actual HTTP request follows]

Backend reads PROXY protocol header first, extracts real client IP
Does not require HTTP — works for any TCP protocol
```

```nginx
# nginx: accept PROXY protocol from load balancer
server {
    listen 80 proxy_protocol;
    real_ip_header proxy_protocol;
    set_real_ip_from 10.0.0.0/8;  # Trust PROXY protocol from internal LBs only
    
    access_log /var/log/nginx/access.log with $proxy_protocol_addr as client IP;
}
```

**Comparison:**

| Property | X-Forwarded-For | PROXY Protocol |
|----------|----------------|----------------|
| Layer | L7 (HTTP header) | L4 (TCP stream prefix) |
| Protocol support | HTTP/HTTPS only | Any TCP |
| Spoofing risk | Higher (HTTP header forgeable) | Lower (TCP-level) |
| Zero-overhead | Yes | Yes (tiny header) |
| gRPC support | Yes (HTTP/2 header) | Yes (TCP) |
| AWS ALB | Sets X-Forwarded-For | Supports PROXY protocol v2 |

---

### Q15. What is network bandwidth vs latency? Which bottleneck matters more in different scenarios?

**Answer:**

**Bandwidth:** The maximum rate of data transfer (bits per second). How wide the pipe is.

**Latency:** The time for one bit to travel from source to destination (milliseconds). How long the pipe is.

```
Analogy: Garden hose to destination city
  Bandwidth = diameter of hose (how much water flows per second)
  Latency   = length of hose (how long water takes to arrive)
  
  Big pipe to far city: high bandwidth, high latency
  Thin pipe nearby: low bandwidth, low latency
```

**Bandwidth × Latency = Bandwidth-Delay Product (BDP):**
```
BDP = how many bytes are "in flight" at any time
BDP = 100 Mbps × 50ms = 100,000,000 × 0.05 = 5,000,000 bytes = ~4.7 MB

This is how much data must be "in flight" to fully utilize the pipe
TCP window size must be >= BDP for full throughput
```

**When latency dominates:**
```
Small request/response (most web API calls):
  Data size: 5KB response
  Bandwidth: 1 Gbps (plenty)
  Latency: 50ms RTT
  
  Transfer time = latency + (data_size / bandwidth)
               = 50ms + (5,000 bytes / 125,000,000 bytes/sec)
               = 50ms + 0.04ms
               = ~50ms (latency completely dominates)
  
  Halving bandwidth → 50.08ms (1% change)
  Halving latency   → 25.04ms (50% change)
  
  → Reduce latency (CDN edge nodes, smaller payload, pipelining)
```

**When bandwidth dominates:**
```
Large file transfer:
  Data size: 1 GB
  Bandwidth: 10 Mbps (constrained)
  Latency: 5ms
  
  Transfer time = latency + (data_size / bandwidth)
               = 5ms + (1,000,000,000 / 1,250,000 bytes/sec)
               = 5ms + 800s
               = ~800 seconds (bandwidth completely dominates)
  
  Halving latency → 800 seconds (0% change)
  Doubling bandwidth → 400 seconds (50% improvement)
  
  → Increase bandwidth, compression, parallel transfers
```

**Practical scenarios:**

| Scenario | Bottleneck | Solution |
|----------|-----------|----------|
| API request (< 100KB) | Latency | CDN, caching, fewer round trips |
| Video streaming (2Mbps constant) | Bandwidth | CDN bandwidth, adaptive bitrate |
| Database query (1KB response) | Latency (network + query) | Read replicas, connection pooling |
| File upload (100MB) | Bandwidth | Parallel chunks, compression |
| Microservice chain (10 hops) | Latency (accumulative) | Reduce hops, gRPC multiplexing |
| Video call (1Mbps bidirectional) | Latency (< 150ms required) | Edge nodes, QUIC |

**Estimating transfer time rule of thumb:**
```
Latency limited: response_time ≈ N_roundtrips × RTT
  Worst case (HTTP/1.1): DNS(1) + TCP(1) + TLS(2) + HTTP(1) = 5 RTTs
  Best case (HTTP/3 cached): 0.5 RTT (0-RTT QUIC)

Bandwidth limited: transfer_time = file_size / bandwidth
  1GB at 100Mbps: 80 seconds
  1GB at 1Gbps: 8 seconds
```

---

## Hard (Q16–Q20)

---

### Q16. Deep dive into TCP congestion control: how does BBR differ from CUBIC and when should you use each?

**Answer:**

TCP congestion control must infer available bandwidth and optimal sending rate without explicit feedback from the network. CUBIC and BBR take fundamentally different approaches.

**CUBIC (Linux default since 2.6.19):**
```
Loss-based: assumes packet loss = congestion signal
Algorithm: cubic function determines window growth

CWND_cubic(t) = C × (t - K)³ + W_max
  where:
    t     = time since last congestion event
    K     = time when CWND would reach W_max without loss
    W_max = window size at time of last congestion
    C     = scaling factor (0.4)

Growth behavior:
  - Slow near W_max (conservative near previous loss point)
  - Fast away from W_max (aggressive recovery)
  - Cubic shape: fast → slow → fast
```

**CUBIC problem — shallow probing:**
```
On 100ms RTT link with 1% random loss:
  CUBIC backs off every ~100 packets (1% loss rate)
  Window never fully opens
  
  Actual bandwidth: 100 Mbps
  CUBIC utilization: ~40 Mbps (misidentifies random loss as congestion)
  
  Wireless networks, long-distance links have high random loss
  CUBIC drastically underutilizes available bandwidth on these paths
```

**BBR (Bottleneck Bandwidth and RTT, Google 2016):**
```
Model-based: maintains model of the bottleneck bandwidth (BW) and RTT
  Does NOT use packet loss as primary congestion signal
  
BBR measures:
  - Maximum delivery rate (max bandwidth observed)
  - Minimum RTT (round-trip time without queuing)
  
BBR target: send at max_bandwidth, keep inflight ≈ BDP
  BDP = max_bandwidth × min_RTT (optimal "pipe full" point)
  
BBR sending rate algorithm:
  Normal mode: send at estimated bottleneck BW × gain_factor
  Probe bandwidth (12.5% of time): increase rate to detect new bandwidth
  Drain (12.5% of time): decrease rate to drain any queues created
```

**BBR vs CUBIC comparison:**

| Dimension | CUBIC | BBR |
|-----------|-------|-----|
| Signal for congestion | Packet loss | Delivery rate + RTT model |
| High random loss networks | Underperforms | Performs well (ignores random loss) |
| Shallow buffers | Good | Good |
| Bufferbloat (large buffers) | Fills buffers (adds latency) | Avoids buffer buildup |
| Short flows (< 10 packets) | Slow start works | BBR startup phase |
| Fairness with CUBIC peers | N/A | May be unfair to CUBIC connections |
| Implementation | Kernel | Kernel (Linux 4.9+) |
| RTT inflation | Can increase | Minimizes |

**Enabling BBR:**
```bash
# Check current default
sysctl net.ipv4.tcp_congestion_control

# Enable BBR globally
sysctl -w net.ipv4.tcp_congestion_control=bbr
sysctl -w net.core.default_qdisc=fq  # Fair Queuing (required with BBR)

# Persist across reboots
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
```

**When to use each:**
```
Use BBR:
  - Long-distance connections (trans-Pacific, cloud to mobile users)
  - High-bandwidth delay product links (100 Gbps data center transfers)
  - Networks with random packet loss (wireless, congested ISP links)
  - Streaming video (consistent throughput matters)
  - Google uses BBR for all their Internet traffic

Use CUBIC:
  - Data center east-west traffic (low latency, low loss)
  - When inter-operability with CUBIC peers is critical
  - When BBR's bandwidth probing causes unacceptable burst behavior
  - Mixed environments where BBR/CUBIC fairness is a concern
```

**Real-world impact (Google's BBR paper):**
- 2,700 km path: BBR 2,700× more throughput than CUBIC (random loss environment)
- YouTube CDN with BBR: median throughput up 4%
- BBR2 (2019): improved fairness with CUBIC while maintaining BBR benefits

---

### Q17. Design a high-performance networking stack for an API gateway handling 1 million RPS with minimal latency.

**Answer:**

An API gateway at 1M RPS is a complex systems problem requiring careful optimization at every layer of the networking stack.

**Architecture overview:**
```
Internet → Anycast IP (Cloudflare/AWS Shield) → DDoS scrubbing
    ↓
L4 Load Balancer (ECMP, multiple VIPs) — hardware or SR-IOV
    ↓
API Gateway Cluster (100 nodes × 10k RPS/node)
    ↓
Backend services (service mesh)
```

**Linux kernel tuning for 1M RPS:**
```bash
# TCP settings for high connection rate
sysctl -w net.ipv4.tcp_max_syn_backlog=65536
sysctl -w net.core.somaxconn=65535        # Accept queue depth
sysctl -w net.ipv4.tcp_syncookies=1       # SYN flood protection
sysctl -w net.ipv4.tcp_tw_reuse=1         # Reuse TIME_WAIT sockets

# Buffer sizes: 1M RPS × 1KB avg request = 1 GB/sec throughput
sysctl -w net.core.rmem_max=134217728     # 128MB receive buffer
sysctl -w net.core.wmem_max=134217728     # 128MB send buffer
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"

# Increase ephemeral port range
sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# Reduce TIME_WAIT (internal traffic only)
sysctl -w net.ipv4.tcp_fin_timeout=10

# Enable BBR
sysctl -w net.ipv4.tcp_congestion_control=bbr
sysctl -w net.core.default_qdisc=fq

# CPU affinity: bind network interrupts to specific CPUs
ethtool -X eth0 hfunc toeplitz                  # RSS hash function
echo 1 > /proc/irq/$(cat /proc/interrupts | grep eth0-rx-0 | awk '{print $1}')/smp_affinity_list
```

**SO_REUSEPORT for multi-process binding:**
```python
# Python with SO_REUSEPORT: multiple workers bind same port
# OS distributes connections across workers (no shared accept() contention)
import socket
import os

def create_reuseport_socket(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle
    sock.bind((host, port))
    return sock

# Gunicorn config: 32 worker processes, each with SO_REUSEPORT
# binding to same port 8080
workers = 32
reuse_port = True
worker_connections = 1000
```

**Connection pooling to backends:**
```python
# Each gateway node: pool of persistent connections per backend
class BackendConnectionPool:
    def __init__(self, backend_addr, pool_size=100):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.backend_addr = backend_addr
        
        # Pre-warm connections
        for _ in range(pool_size):
            asyncio.ensure_future(self._add_connection())
    
    async def _add_connection(self):
        # HTTP/2 connection (multiplexes many requests)
        conn = await create_http2_connection(self.backend_addr)
        await self.pool.put(conn)
    
    async def proxy_request(self, request):
        conn = await asyncio.wait_for(self.pool.get(), timeout=0.1)
        try:
            response = await conn.request(request)
            return response
        finally:
            await self.pool.put(conn)
```

**Zero-copy networking with sendfile:**
```python
# For large responses: bypass user-space copy
import os

async def serve_static_file(path: str, response):
    fd = os.open(path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        response.set_header("Content-Length", str(file_size))
        await response.sendfile(fd, offset=0, count=file_size)
        # sendfile() copies directly kernel buffer → socket buffer
        # No user-space copy: ~40% CPU reduction for file serving
    finally:
        os.close(fd)
```

**Profiling 1M RPS bottlenecks:**
```
Benchmark tool: wrk, hey, k6
  wrk -t12 -c400 -d30s http://api-gateway/endpoint
  
Expected bottlenecks at 1M RPS:
  1. CPU (at ~80% utilization): scale horizontally or optimize hot path
  2. NIC (at line rate 10Gbps = 1.25 GB/s):
     - Upgrade to 25GbE or 100GbE NIC
     - SR-IOV: direct NIC access bypassing kernel
     - DPDK: bypass kernel networking entirely (data plane dev kit)
  3. Accept queue (somaxconn):
     - Monitor: ss -s | grep "SYN"
     - Increase somaxconn and application backlog
  4. File descriptors:
     - ulimit -n 1000000
     - /etc/limits.conf: * soft nofile 1000000
```

---

### Q18. Explain how a CDN works at a networking level — from DNS to TCP to cache to origin.

**Answer:**

A CDN is a global network of edge servers that cache content near users, reducing origin load and improving latency.

**Full request flow for https://cdn.example.com/image.jpg:**

**Step 1: DNS resolution**
```
Browser queries: cdn.example.com
  → CNAME: example.cdnprovider.net
  → cdnprovider.net has Anycast: returns multiple IPs, all serving nearest PoP
  → Browser gets: 203.0.113.45 (nearest CDN edge IP)
  
DNS TTL: 30s (low TTL allows failover and traffic redistribution)
```

**Step 2: TCP + TLS to edge**
```
Browser ──── TCP SYN ────────────────► CDN Edge (nearest PoP, ~10ms RTT)
         ◄── SYN-ACK ────────────────
         ──── ACK + TLS ClientHello ──►
         ◄──── TLS ServerHello ──────   (CDN has SSL cert for example.com)
         
TLS 1.3: 1 RTT = 10ms (nearby edge) vs 200ms (US origin from Europe)
```

**Step 3: Cache lookup at edge**
```
Edge server: check local cache
  Cache key: {method: GET, host: cdn.example.com, path: /image.jpg, 
               Vary: Accept-Encoding, Accept}
  
  Cache HIT (image in memory/SSD):
    → Return 200 with X-Cache: HIT header
    → Total latency: ~15ms (edge RTT + cache read)
  
  Cache MISS:
    → Proceed to origin
    → Simultaneously: lock this cache key (prevent stampede)
                     return 503 or serve stale if origin is down
```

**Step 4: Origin fetch (on cache miss)**
```
CDN Edge ──── TCP to origin (via CDN's private backbone) ────► Origin
             (CDN backbone: private fiber, 50ms globally vs 150ms public internet)
             ──── HTTP/2 GET /image.jpg ──────────────────────►
             ◄──── 200 OK + Cache-Control: max-age=86400 ──────
             [Store in edge cache]
             ──── Return to browser ──────────────────────────►
```

**Cache TTL and revalidation:**
```
Origin headers that control CDN behavior:
  Cache-Control: max-age=86400, s-maxage=31536000
    - max-age: browser cache (1 day)
    - s-maxage: CDN cache (1 year) — overrides max-age for shared caches
    
  Cache-Control: no-store → CDN never caches (auth pages, dynamic APIs)
  Cache-Control: no-cache → Revalidate with origin on every request (ETag)
  Vary: Accept-Language → Cache separate copies per language
  
  Surrogate-Key / Cache-Tag: cdn.example.com/image-001
    → Allows targeted cache invalidation: "purge all images tagged 'product-42'"
    → CDN API call: PURGE /image.jpg or purge by tag
```

**Cache hit ratio optimization:**
```python
# Cache key design: normalize unnecessary variation
nginx cache key:
  $scheme$request_method$host$request_uri  # Default
  → Problem: ?sessionid=abc123 → unique key per user, 0% hit rate
  
Better:
  $scheme$request_method$host$uri  # Exclude query string for cacheable resources
  → Static assets: 99% cache hit rate
  
For APIs with query params:
  Sort query params (cache_key normalizer)
  ?a=1&b=2 and ?b=2&a=1 → same cache key
```

**CDN origin shield (reduce origin load):**
```
Without shield:
  1000 edge PoPs, each cache miss → 1000 concurrent origin requests

With origin shield:
  1000 edges → shield PoP (cache) → origin
  Cache miss in edge → goes to shield first
  If shield has it → serve from shield (no origin hit)
  If shield misses → one origin request for all 1000 edges
  
  Origin load reduction: 99.9% on popular content
```

---

### Q19. Design a global low-latency system for a multiplayer game with sub-100ms latency requirements.

**Answer:**

Multiplayer games have some of the most demanding networking requirements: sub-100ms round-trip latency, minimal jitter, and tolerance for packet loss.

**Latency budget analysis:**
```
Total budget: 100ms for player input → server → all clients see update

  Network RTT budget: 50ms (player → game server ← opponent)
  Server processing: 5ms (game logic, collision detection)
  Client rendering: 16ms (60 FPS frame budget)
  Jitter buffer: 15ms
  Safety margin: 14ms
  
For 50ms network RTT: game server must be within ~2,500km of players
→ Need regional servers on every continent
```

**Architecture:**
```
Players                  Regional Game Servers              Global State
┌─────────┐             ┌─────────────────────┐             ┌─────────────┐
│ Player A│─── UDP ────►│ US-East Game Server │──── gRPC ──►│ Match Maker │
│(New York)│            │ (100ms game loops)  │             │ (any region)│
└─────────┘             └────────────┬────────┘             └─────────────┘
                                     │ (authoritative state)
┌─────────┐             ┌────────────▼────────┐
│ Player B│─── UDP ────►│   (same server)     │
│(Boston) │            └─────────────────────┘
└─────────┘
```

**UDP-based game protocol:**
```python
import socket
import struct
import time

class GameProtocol:
    # Fixed-size header: 16 bytes
    # Format: seq(4B) + ack(4B) + ack_bits(4B) + timestamp(4B)
    HEADER_FORMAT = "!IIII"
    HEADER_SIZE = 16
    
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.remote_ack = 0
        self.ack_bits = 0  # Bitmask of last 32 received packets
    
    def send_input(self, player_input: dict):
        """Send player input update (16Hz = 62ms interval)."""
        header = struct.pack(
            self.HEADER_FORMAT,
            self.seq,
            self.remote_ack,
            self.ack_bits,
            int(time.time() * 1000) & 0xFFFFFFFF
        )
        
        payload = encode_input(player_input)  # Compact binary encoding
        self.sock.sendto(header + payload, game_server_addr)
        self.seq += 1
    
    def calculate_rtt(self, ack_seq: int, ack_timestamp: int) -> float:
        """Calculate RTT from ACK timestamp."""
        now = int(time.time() * 1000)
        return now - ack_timestamp
```

**Lag compensation (client-side prediction + server reconciliation):**
```
Client-side prediction:
  Input → Client predicts own position immediately (no wait for server)
  Feels instant to local player (0ms perceived latency for own movement)

Server authoritative update:
  Server confirms position 50ms later
  If different from predicted: "rubber banding" (correct silently if small diff)
  
Rollback netcode (used by fighting games, Rocket League):
  When peer input arrives late:
    Roll game state back to when packet should have arrived
    Apply actual input
    Re-simulate forward to current frame
    Extremely smooth for small latency (<100ms)
```

**Jitter buffer for remote player positions:**
```python
class JitterBuffer:
    def __init__(self, buffer_ms=50):
        self.buffer_ms = buffer_ms
        self.packets = []
    
    def receive(self, packet):
        """Buffer incoming packets, adding delay for smoothness."""
        self.packets.append((time.time(), packet))
    
    def get_packets_for_now(self):
        """Return packets that should be displayed now."""
        cutoff = time.time() - (self.buffer_ms / 1000)
        ready = [p for ts, p in self.packets if ts <= cutoff]
        self.packets = [(ts, p) for ts, p in self.packets if ts > cutoff]
        return ready
```

**Regional server selection:**
```python
# Client pings all regional servers during matchmaking
async def select_best_server(player_id: str):
    regions = ['us-east', 'eu-west', 'ap-southeast', 'sa-east']
    
    pings = await asyncio.gather(*[
        ping_server(f"game-{r}.example.com") for r in regions
    ])
    
    # Sort by RTT
    ranked = sorted(zip(regions, pings), key=lambda x: x[1])
    
    # Select lowest latency region with available capacity
    for region, rtt in ranked:
        if rtt < 100 and await get_server_capacity(region) > 0:
            return region, rtt
    
    return ranked[0][0], ranked[0][1]  # Best available
```

---

### Q20. How do you implement a zero-trust network architecture? Compare VPN, service mesh, and ZTNA.

**Answer:**

Zero-trust network access (ZTNA) abandons the "trusted internal network" assumption. Every connection must be authenticated, authorized, and encrypted — regardless of source location.

**Traditional VPN model (perimeter-based):**
```
Internet → Firewall → VPN Tunnel → Corporate Network (trusted zone)
                                   ├── Finance servers (trusted)
                                   ├── HR systems (trusted)
                                   └── Engineering systems (trusted)

Problem: VPN grants broad network access
  - Compromised laptop = attacker on trusted network
  - Lateral movement: compromise one system → access all
  - No per-application authorization
  - Performance: all traffic routed through VPN concentrator
```

**Zero-trust principles:**
```
1. Verify explicitly: Always authenticate (MFA, device health)
2. Least privilege: Minimum access needed (not full network access)
3. Assume breach: Encrypt everywhere, segment, minimize blast radius
4. Continuous verification: Not just at login — re-verify context continuously
```

**Implementation options:**

**Option 1: Service Mesh (east-west, internal services)**
```
Istio service mesh:
  Every service gets:
    - SPIFFE/SVID identity (X.509 cert rotated every 24 hours)
    - mTLS: both sides authenticate (service A authenticates to service B)
    - Authorization policies: service A can only call specific endpoints on service B
  
  Zero-trust rules:
    payment-service can call: order-service:/orders/{id} (GET only)
    payment-service cannot call: user-service:/users/{id}/delete

# Istio AuthorizationPolicy: zero-trust within cluster
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-nothing  # Default deny all
spec: {}              # Empty spec = deny all

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-payment-to-order
  namespace: orders
spec:
  selector:
    matchLabels:
      app: order-service
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/payments/sa/payment-service"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/orders/*"]
```

**Option 2: ZTNA (north-south, user → application)**
```
Cloudflare Access / Zscaler ZPA / BeyondCorp:

User (home/cafe) → ZTNA Client (device agent) → ZTNA Gateway
                   [Device health check:         [Identity check:]
                    OS patches up to date?         SSO: valid MFA?
                    Disk encrypted?                Group membership?
                    No malware detected?           Location risk?
                    Corporate device?]
                                        ↓ (all checks pass)
                                        ↓
                                   Application
                                   (never exposed to internet directly)

vs VPN:
  VPN: user → VPN → FULL network access (after one login)
  ZTNA: user → check → SPECIFIC application access (re-checked each session)
```

**Option 3: Comparison**

| Dimension | VPN | Service Mesh | ZTNA |
|-----------|-----|-------------|------|
| Traffic type | North-south (user→corp) | East-west (service→service) | North-south (user→app) |
| Granularity | Network-level | Service/endpoint-level | Application-level |
| User experience | Slow, all traffic routed | Transparent | Fast (direct to app) |
| Lateral movement | High risk | Prevented by policy | N/A |
| Device trust | No (just login) | N/A | Yes (device health) |
| Performance | Bottleneck at VPN | No bottleneck | Direct connection |
| Use case | Legacy apps, network level | Microservices | Modern SaaS, remote access |

**Full zero-trust architecture:**
```
External users → Cloudflare Access (ZTNA) → Internal apps
    ├── Device health checked (Crowdstrike, Intune)
    ├── SSO via Okta (MFA required)
    └── Per-app policy (marketing team: no access to finance apps)

Internal services → Istio service mesh (mTLS)
    ├── Every service has SPIFFE identity
    ├── AuthorizationPolicy: default deny all
    └── Explicit allow per service-to-service path

Databases → AWS IAM authentication (no static passwords)
    ├── Each service uses IAM role (no shared credentials)
    └── Temporary credentials (STS) rotated every hour

Secrets → HashiCorp Vault (dynamic secrets)
    ├── Kubernetes auth: pod identity → Vault policy → secret
    └── TTL: 1 hour; auto-renewed while pod runs
```

**Implementation sequence for migrating from VPN to ZTNA:**
```
Month 1-3: Deploy ZTNA alongside VPN (shadow mode)
Month 4-6: Migrate non-critical apps to ZTNA, monitor
Month 7-9: Migrate critical apps, test thoroughly
Month 10:  Restrict VPN to legacy apps only
Month 12+: Retire VPN entirely (legacy apps migrated or wrapped)
```

---

## Quick Reference

### TCP Handshake Costs

| Connection Type | Latency Cost (50ms RTT example) |
|----------------|--------------------------------|
| New TCP | 50ms (1 RTT) |
| New TCP + TLS 1.2 | 150ms (3 RTT) |
| New TCP + TLS 1.3 | 100ms (2 RTT) |
| QUIC (HTTP/3) first | 50ms (1 RTT — TLS included) |
| QUIC 0-RTT | ~0ms (0 RTT data) |
| Reused TCP + TLS | 0ms (no handshake) |

### Latency vs Bandwidth: When Each Dominates

| Payload Size | Bandwidth (1Gbps) | Latency (50ms RTT) | Bottleneck |
|-------------|------------------|-------------------|------------|
| 1 KB | 0.008ms | 50ms | Latency (6250x) |
| 100 KB | 0.8ms | 50ms | Latency (62x) |
| 10 MB | 80ms | 50ms | Mixed |
| 1 GB | 8000ms | 50ms | Bandwidth (160x) |

### DNS Record Types

| Record | Maps | Example |
|--------|------|---------|
| A | name → IPv4 | api.example.com → 1.2.3.4 |
| AAAA | name → IPv6 | api.example.com → 2001:db8::1 |
| CNAME | name → name | www → api.example.com |
| MX | domain → mail host | example.com → mx.google.com |
| TXT | domain → text | SPF, DKIM verification |
| NS | zone → nameserver | example.com → ns1.route53.aws |

### HTTP Version Comparison

| Feature | HTTP/1.1 | HTTP/2 | HTTP/3 (QUIC) |
|---------|----------|--------|---------------|
| Multiplexing | No (6 parallel connections) | Yes (streams) | Yes (streams) |
| L7 HoL blocking | Yes | No | No |
| L4 HoL blocking | Yes | Yes | No |
| Header compression | No | HPACK | QPACK |
| Server push | No | Yes | Yes |
| Transport | TCP | TCP | UDP (QUIC) |
| Connection migration | No | No | Yes |

### TCP Tuning Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| net.core.somaxconn | 65535 | Accept queue depth |
| net.ipv4.tcp_max_syn_backlog | 65536 | SYN queue depth |
| net.ipv4.tcp_tw_reuse | 1 | Reuse TIME_WAIT sockets |
| net.ipv4.tcp_fin_timeout | 10 | TIME_WAIT duration (internal only) |
| net.ipv4.ip_local_port_range | 1024 65535 | Ephemeral port range |
| net.ipv4.tcp_congestion_control | bbr | Congestion algorithm |
| net.core.default_qdisc | fq | Queue discipline (needed for BBR) |

### VPN vs ZTNA vs Service Mesh

| | VPN | ZTNA | Service Mesh |
|-|-----|------|-------------|
| Use case | User → corporate network | User → specific app | Service → service |
| Granularity | Network | Application | API endpoint |
| Device trust | Login only | Device + identity | N/A |
| Lateral movement | Possible | Prevented | Prevented |
| Performance | LB at VPN | Direct | In-process |
