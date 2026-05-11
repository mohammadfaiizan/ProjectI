# Load Balancing and Rate Limiting

## Table of Contents
1. [Load Balancing Fundamentals](#load-balancing-fundamentals)
2. [Load Balancing Algorithms](#load-balancing-algorithms)
3. [L4 vs L7 Load Balancing](#l4-vs-l7-load-balancing)
4. [Session Persistence / Sticky Sessions](#session-persistence--sticky-sessions)
5. [Health Checks and Circuit Breaker Integration](#health-checks-and-circuit-breaker-integration)
6. [Global Server Load Balancing (GSLB)](#global-server-load-balancing-gslb)
7. [Auto-Scaling with Load Balancers](#auto-scaling-with-load-balancers)
8. [SSL/TLS Termination](#ssltls-termination)
9. [Nginx vs HAProxy vs AWS ALB/NLB](#nginx-vs-haproxy-vs-aws-albnlb)
10. [Rate Limiting Algorithms](#rate-limiting-algorithms)
11. [Rate Limiting: Where to Implement](#rate-limiting-where-to-implement)
12. [Throttling Strategies](#throttling-strategies)
13. [DDoS Protection at Load Balancer Layer](#ddos-protection-at-load-balancer-layer)
14. [Quick Reference Tables](#quick-reference-tables)

---

## Load Balancing Fundamentals

### Why Load Balancing?

A single server has finite capacity. As traffic grows, a single machine becomes a bottleneck and a single point of failure. Load balancing solves three problems:

1. **Scalability** — Distribute traffic across multiple servers so the system handles more requests.
2. **Availability** — Route around failed or unhealthy servers automatically.
3. **Performance** — Serve requests from the geographically or computationally closest resource.

Without a load balancer, horizontal scaling is impossible because clients must know which individual server to contact. The load balancer provides a single virtual IP (VIP) or DNS name while hiding the pool of backend servers.

### Where Load Balancers Sit in the Stack

```
Client
  |
  v
[DNS Load Balancing]          <-- GeoDNS, Round-Robin DNS (Layer before TCP)
  |
  v
[L4 Load Balancer]            <-- TCP/UDP level (e.g., AWS NLB, HAProxy TCP mode)
  |
  v
[L7 Load Balancer]            <-- HTTP/HTTPS level (e.g., AWS ALB, Nginx, HAProxy HTTP)
  |
  v
[Application Servers]
  |
  v
[Database Load Balancer]      <-- Read replicas, ProxySQL, pgBouncer
```

**DNS Load Balancing**
- Multiple A records for the same domain, different IPs.
- The DNS resolver returns different IPs to different clients (round-robin or geo-based).
- Cheap but lacks health awareness; TTL makes failover slow (60–300s).
- Used for global routing, not fine-grained balancing.

**L4 Load Balancing**
- Operates at the TCP/UDP transport layer.
- Inspects source IP, destination IP, source port, destination port.
- Does not read HTTP headers or body.
- Extremely fast — packets are forwarded with minimal processing.

**L7 Load Balancing**
- Operates at the HTTP/HTTPS application layer.
- Can inspect URL path, Host header, cookie, query parameters, request body.
- Enables content-based routing (e.g., `/api/*` → API servers, `/static/*` → CDN origin).
- Supports SSL termination, header injection, and request rewriting.

### Load Balancer Deployment Models

| Model | Description | Example |
|---|---|---|
| Hardware appliance | Dedicated physical device | F5 BIG-IP |
| Software LB on VM | Software running on commodity hardware | Nginx, HAProxy |
| Cloud-managed LB | Fully managed cloud service | AWS ALB, GCP HTTPS LB |
| Sidecar proxy | Per-pod proxy in service mesh | Envoy (Istio) |

---

## Load Balancing Algorithms

### 1. Round Robin

Requests are distributed sequentially to each server in the pool, cycling back to the first after reaching the last.

```
Server pool: [A, B, C]
Request 1 -> A
Request 2 -> B
Request 3 -> C
Request 4 -> A  (wraps)
```

**Pros:** Simple, equal distribution, no state needed.
**Cons:** Ignores server capacity differences; a slow request on server A still means the next request goes to B while A is loaded.

**Best for:** Homogeneous servers, short-lived stateless requests.

### 2. Weighted Round Robin

Each server is assigned a weight proportional to its capacity. Servers with higher weight receive proportionally more requests.

```
Server A: weight=3
Server B: weight=1
Server C: weight=2

Sequence: A, A, A, B, C, C, A, A, A, B, C, C ...
```

**Pros:** Handles heterogeneous server pools (e.g., different instance types).
**Cons:** Requires manual weight configuration; does not adapt to real-time load.

**Best for:** Mixed instance type clusters, gradual traffic shifting during deployments (canary releases).

### 3. Least Connections

Each new request is routed to the server with the fewest active connections at that moment.

```
Server A: 10 active connections
Server B: 2 active connections  <- next request goes here
Server C: 7 active connections
```

**Pros:** Adapts to real-time load; prevents hot servers.
**Cons:** Requires the load balancer to track connection counts (more state).

**Best for:** Long-lived connections (WebSockets, database connections, file uploads) where request duration varies.

### 4. Weighted Least Connections

Combines weights with active connection counts. The score is: `active_connections / weight`. The server with the lowest score wins.

```
Server A: weight=4, connections=8  -> score = 2.0
Server B: weight=2, connections=2  -> score = 1.0  <- selected
Server C: weight=1, connections=3  -> score = 3.0
```

**Best for:** Heterogeneous pools with variable request durations.

### 5. Least Response Time

Routes to the server with the lowest combination of active connections and average response time.

```
score = active_connections * avg_response_time_ms
```

**Pros:** Most reflective of real-world server performance.
**Cons:** Requires response time tracking; adds latency overhead on the LB.

**Best for:** Microservices where some servers may be degraded due to GC, CPU pressure, etc.

### 6. IP Hash (Source IP Affinity)

A hash of the client's IP address determines which server handles the request. The same client always goes to the same server (as long as the pool size is unchanged).

```python
server_index = hash(client_ip) % len(server_pool)
```

**Pros:** Simple stickiness without cookies; useful when session state is on the server.
**Cons:** Adding/removing servers changes all mappings (use consistent hashing to fix this). Clients behind NAT all go to the same server.

### 7. Consistent Hashing

Servers and requests are both mapped to positions on a virtual ring (hash ring). A request is routed to the first server at or after its position on the ring (clockwise).

```
Ring (0 to 2^32 - 1):

    0
   / \
  S1  S2
   \  /
    S3

Request hash -> find nearest server clockwise
```

**Key property:** When a server is added or removed, only `K/N` keys need to be remapped (where K = keys, N = nodes), compared to `K` keys for naive modulo hashing.

**Virtual nodes:** Each physical server maps to multiple positions on the ring (e.g., 150 virtual nodes per server). This prevents hotspots when the pool is small.

**Pros:** Minimal disruption on scaling events; natural session affinity.
**Cons:** More complex implementation; requires a consistent hashing library.

**Best for:** Distributed caches (Redis Cluster, Memcached), database sharding, CDN routing.

---

## L4 vs L7 Load Balancing

### L4 (Transport Layer) Load Balancing

L4 load balancers operate at the TCP/UDP level. They see:
- Source IP and port
- Destination IP and port
- Connection state (SYN, FIN, RST)

They do NOT see HTTP headers, URLs, or cookies.

**How it works — Direct Server Return (DSR) mode:**
```
Client -> [LB receives SYN, modifies DST_IP to backend, forwards]
Client <-> Backend (data flows directly after TCP handshake)
Backend -> Client (response bypasses LB in DSR)
```

**Characteristics:**
- Latency: sub-millisecond overhead
- Throughput: millions of packets/second
- SSL: passthrough only (LB does not decrypt)
- Routing: only by IP/port, no content awareness
- Connection: maintains per-connection state (NAT table)

### L7 (Application Layer) Load Balancing

L7 load balancers terminate the client connection, inspect the HTTP request, and open a new connection to the backend.

```
Client TLS -> [LB: terminates TLS, reads HTTP headers, routes based on path/host]
                       |
                       v
              [Backend: receives plain HTTP]
```

**Routing capabilities:**
- Host-based: `api.example.com` -> API cluster, `www.example.com` -> Web cluster
- Path-based: `/api/v2/*` -> v2 servers, `/api/v1/*` -> v1 legacy servers
- Header-based: `X-Canary: true` -> canary servers
- Cookie-based: sticky session routing
- Query-param-based: routing A/B test cohorts

**Additional features:**
- Request/response header manipulation
- SSL/TLS termination and re-encryption
- HTTP/2 and gRPC multiplexing
- WebSocket support
- Rate limiting per URL pattern
- WAF integration
- Access logging with full request details

### L4 vs L7 Comparison Table

| Dimension | L4 | L7 |
|---|---|---|
| OSI Layer | Transport (4) | Application (7) |
| Protocol awareness | TCP/UDP only | HTTP, HTTPS, gRPC, WebSocket |
| Routing basis | IP + Port | URL, Host, Headers, Cookies |
| SSL termination | No (passthrough) | Yes |
| Performance | Higher throughput, lower latency | Slightly more overhead |
| Content inspection | No | Yes |
| Sticky sessions | IP-based only | Cookie-based or IP-based |
| Use case | TCP services, databases, raw speed | Web apps, API gateways, microservices |
| AWS equivalent | NLB | ALB |

### When to Use L4

- Non-HTTP protocols (SMTP, MQTT, custom TCP protocols)
- Need to preserve client IP without X-Forwarded-For
- Extreme throughput requirements (millions of connections)
- Database load balancing (MySQL, PostgreSQL)
- Gaming servers, VoIP

### When to Use L7

- HTTP/HTTPS web applications
- Microservices with URL-based routing
- gRPC services
- A/B testing, canary deployments
- Need for WAF, rate limiting, authentication offload

---

## Session Persistence / Sticky Sessions

### The Problem

HTTP is stateless, but many applications store session state in server memory (e.g., shopping carts, login sessions). Without stickiness, a user might be routed to a different server on each request, losing their session.

### Implementation Methods

**1. Cookie-Based Stickiness (Recommended)**

The load balancer inserts a cookie into the response that identifies the backend server. On subsequent requests, the LB reads the cookie and routes to the same server.

```
Set-Cookie: AWSALB=server_b_token; Path=/; Max-Age=86400; HttpOnly
```

AWS ALB calls this "sticky sessions" with duration-based or application-based cookies.

**2. IP Hash Stickiness**

Hash the client IP to a server. Simple but problematic when:
- Many clients share one IP (corporate NAT, mobile carrier NAT)
- Client IP changes (mobile network handoff)

**3. Application-Level Session Sharing (Better Alternative)**

Store sessions in a shared store (Redis, Memcached, DynamoDB) rather than on the server. Any server can serve any request.

```
Client -> Any Server -> Redis (session store)
```

This is the preferred architecture because it:
- Enables true horizontal scaling
- Eliminates sticky session failure scenarios
- Simplifies zero-downtime deployments

### Sticky Session Trade-offs

| Aspect | Sticky Sessions | Shared Session Store |
|---|---|---|
| Architecture complexity | Low | Medium |
| Horizontal scalability | Limited (uneven distribution) | Full |
| Failure impact | Session lost if server dies | Session preserved |
| Deployment complexity | Drain connections before restart | None |
| Performance | Slightly better (local memory) | Small Redis latency (~0.5ms) |

### Connection Draining

When removing a server from the pool (deployment, failure), connection draining ensures:
1. No new requests are routed to the server being removed.
2. Existing connections are allowed to complete (typically up to 300 seconds).
3. After draining, the server is deregistered.

AWS ALB calls this "deregistration delay." It is essential for zero-downtime deployments.

---

## Health Checks and Circuit Breaker Integration

### Passive Health Checks

The load balancer monitors responses from the backend passively. If a server returns errors (5xx) or times out repeatedly, it is marked unhealthy.

```
If error_count > threshold in time_window:
    mark server as unhealthy
    stop routing to it
    after recovery_timeout: probe with single request
    if success: mark healthy again
```

**Pros:** No extra traffic generated.
**Cons:** Real user requests are used to detect failures. Users see errors during the detection window.

### Active Health Checks

The load balancer sends periodic synthetic requests to a dedicated health endpoint.

```yaml
# Nginx upstream health check
upstream backend {
    server 10.0.1.1:8080;
    server 10.0.1.2:8080;
    check interval=3000 rise=2 fall=3 timeout=1000 type=http;
    check_http_send "GET /health HTTP/1.0\r\n\r\n";
    check_http_expect_alive http_2xx;
}
```

**Health endpoint design:**
```python
@app.get("/health")
def health_check():
    # Check critical dependencies
    db_ok = check_database_connection()
    cache_ok = check_redis_connection()
    
    if db_ok and cache_ok:
        return {"status": "healthy"}, 200
    else:
        return {"status": "unhealthy", "db": db_ok, "cache": cache_ok}, 503
```

**Health check types:**
- **Shallow/liveness check** — Is the process alive? Just returns 200.
- **Deep/readiness check** — Are all dependencies accessible? Returns 503 if not ready.

Load balancers should use the readiness check to avoid routing to a server that is alive but can't serve traffic.

### Circuit Breaker Pattern

The circuit breaker is a proxy that tracks failure rates and "opens" (stops sending requests) when failures exceed a threshold.

```
States:
CLOSED -> (errors > threshold) -> OPEN -> (after timeout) -> HALF-OPEN
                                                                  |
                                              success: CLOSED <---+---> failure: OPEN
```

**Integration with load balancer:**
```
Client -> Load Balancer -> [Circuit Breaker per backend] -> Backend
```

Tools: Resilience4j (Java), Hystrix (deprecated), Envoy (service mesh), AWS App Mesh.

```java
// Resilience4j circuit breaker
CircuitBreakerConfig config = CircuitBreakerConfig.custom()
    .failureRateThreshold(50)           // Open when 50% of requests fail
    .waitDurationInOpenState(Duration.ofSeconds(30))
    .slidingWindowSize(10)
    .build();
```

---

## Global Server Load Balancing (GSLB)

### Purpose

GSLB routes users to the geographically nearest (or best-performing) data center. This reduces latency and provides disaster recovery across regions.

### GeoDNS

DNS servers return different IP addresses based on the geographic location of the DNS resolver.

```
Client in US-East:  example.com -> 54.10.0.1  (US-East data center)
Client in EU-West:  example.com -> 52.30.0.1  (EU-West data center)
Client in AP-South: example.com -> 13.70.0.1  (AP-South data center)
```

**Routing policies (AWS Route 53):**
- **Geolocation routing** — Route based on continent/country/state.
- **Latency-based routing** — Route to the region with lowest measured latency.
- **Failover routing** — Primary/secondary with health checks.
- **Weighted routing** — Percentage-based traffic splitting across regions.

**Limitations of GeoDNS:**
- DNS TTL means failover takes minutes, not seconds.
- Clients using non-local DNS resolvers (8.8.8.8) may be misrouted.
- Does not handle server-level failures, only region-level.

### Anycast

A single IP address is announced from multiple data centers via BGP. The internet's routing tables direct traffic to the nearest announcement point.

```
IP: 1.1.1.1 is announced from:
  - Data center in New York
  - Data center in London
  - Data center in Tokyo

A client in Paris connects to 1.1.1.1 -> BGP routes to London
A client in Sydney connects to 1.1.1.1 -> BGP routes to Tokyo
```

**Pros:** Automatic failover at BGP level (seconds, not minutes). No DNS TTL issues.
**Cons:** Requires BGP routing expertise. Troubleshooting is harder. Session affinity is difficult.

**Used by:** Cloudflare, Google (8.8.8.8), AWS Shield.

### GSLB with Health Checks

GSLB health checks operate at the DNS level:

```
GSLB monitors each data center endpoint.
If data center A fails health check:
    Remove data center A's IP from DNS responses.
    All traffic shifts to remaining healthy data centers.
```

---

## Auto-Scaling with Load Balancers

### How Auto-Scaling Works with a Load Balancer

```
[CloudWatch / Metrics] -> [Auto Scaling Group] -> [EC2 Instance] -> [Register with LB Target Group]
```

1. Metric breaches threshold (CPU > 70% for 2 minutes).
2. Auto Scaling Group launches new EC2 instance.
3. Instance runs startup script, passes health check.
4. Instance is registered with the load balancer target group.
5. Load balancer starts routing traffic to the new instance.

For scale-in (removing instances):
1. ASG marks instance for termination.
2. Load balancer begins connection draining (deregistration delay).
3. After drain completes, instance is terminated.

### Scale-Out Triggers

| Metric | Threshold Example | Notes |
|---|---|---|
| CPU Utilization | > 70% for 3 minutes | Classic trigger |
| Request count per target | > 1000 req/min/instance | Better for LB-fronted apps |
| Average response time | > 500ms | Application-level health |
| SQS queue depth | > 100 messages | Worker/consumer scale-out |
| Custom metric | Any CloudWatch metric | Most flexible |

### Cooldown Periods

The cooldown period prevents the auto-scaler from launching (or terminating) more instances before the previous scaling action takes effect.

- **Scale-out cooldown:** Wait after adding instances before adding more. Default: 300s. Set lower (60–120s) if instances bootstrap quickly.
- **Scale-in cooldown:** Wait after removing instances before removing more. Default: 300s. Set higher to prevent flapping.

### Predictive Scaling

Instead of reactive scaling, predictive scaling uses ML to forecast traffic patterns and pre-provisions capacity.

```
Training: 14 days of historical traffic data
Forecast: Next 48 hours
Action: Launch instances 10 minutes before predicted spike
```

AWS Predictive Scaling, GCP Predictive Autoscaling.

---

## SSL/TLS Termination

### What Is SSL Termination?

The load balancer decrypts incoming HTTPS traffic, inspects the plain HTTP, and forwards it to backends over HTTP (or re-encrypts).

```
Client --[HTTPS]--> [Load Balancer: decrypt] --[HTTP]--> Backend
```

### Why Terminate at the Load Balancer?

1. **Performance** — SSL handshakes are CPU-intensive. Offloading to a dedicated LB (with hardware acceleration) reduces CPU load on application servers.
2. **Certificate management** — Manage one certificate at the LB rather than on every server.
3. **L7 inspection** — The LB must decrypt to read HTTP headers for routing decisions.
4. **Simpler backends** — Application servers only handle HTTP.

### SSL Passthrough

For end-to-end encryption where backends must handle SSL (compliance requirements, mutual TLS):

```
Client --[HTTPS]--> [L4 LB: forward TCP] --[HTTPS]--> Backend (handles SSL)
```

- LB cannot inspect HTTP content (URL routing is impossible).
- Stick to L4 load balancer (NLB).

### Re-encryption (SSL Bridging)

LB decrypts, inspects, then re-encrypts to the backend:

```
Client --[HTTPS]--> [LB: decrypt, inspect, re-encrypt] --[HTTPS]--> Backend
```

- Satisfies compliance requirements for encryption in transit.
- More CPU overhead on the LB.

### Certificate Management

Modern load balancers integrate with certificate management services:
- **AWS ACM (Certificate Manager)** — Free certs, auto-renewal, native ALB/NLB integration.
- **Let's Encrypt with Certbot** — Free, 90-day certs, auto-renewal via cron.
- **HashiCorp Vault PKI** — Internal CA for service-to-service TLS.

---

## Nginx vs HAProxy vs AWS ALB/NLB

### Nginx

Originally a web server, Nginx is also widely used as a reverse proxy and load balancer.

```nginx
upstream app_servers {
    least_conn;
    server 10.0.1.1:8080 weight=3;
    server 10.0.1.2:8080 weight=2;
    server 10.0.1.3:8080 backup;  # only used if others fail
    keepalive 32;  # persistent connections to backends
}

server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;

    location /api/ {
        proxy_pass http://app_servers;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
    }
}
```

**Strengths:** Excellent static file serving, HTTP/2, caching, flexible config.
**Weaknesses:** Health checks require paid Nginx Plus; limited real-time stats in OSS.

### HAProxy

Purpose-built load balancer, extremely high performance.

```haproxy
frontend http_front
    bind *:80
    default_backend app_servers
    acl url_api path_beg /api
    use_backend api_servers if url_api

backend app_servers
    balance leastconn
    option httpchk GET /health
    server srv1 10.0.1.1:8080 check inter 3s rise 2 fall 3
    server srv2 10.0.1.2:8080 check inter 3s rise 2 fall 3
    server srv3 10.0.1.3:8080 check inter 3s rise 2 fall 3 backup

backend api_servers
    balance roundrobin
    server api1 10.0.2.1:8080 check
    server api2 10.0.2.2:8080 check
```

**Strengths:** Industry-leading performance, rich health check options, real-time stats dashboard, TCP and HTTP modes.
**Weaknesses:** Not a web server (no static file serving), configuration syntax less familiar.

### AWS ALB (Application Load Balancer)

Fully managed L7 load balancer integrated with the AWS ecosystem.

**Key features:**
- Host-based and path-based routing
- Native integration with ECS, EKS, Lambda, Cognito
- WAF integration
- gRPC support
- Automatic scaling of the LB itself
- Access logs to S3, metrics to CloudWatch

**Pricing:** Per LCU (Load Balancer Capacity Unit) based on connections, bandwidth, rule evaluations.

### AWS NLB (Network Load Balancer)

Fully managed L4 load balancer.

**Key features:**
- Ultra-low latency (sub-millisecond)
- Static Elastic IP per AZ (useful for whitelisting)
- Preserves client IP natively
- Handles millions of requests/second
- TLS termination (as of 2019)
- Zonal isolation for latency-sensitive workloads

### Comparison Table

| Feature | Nginx | HAProxy | AWS ALB | AWS NLB |
|---|---|---|---|---|
| Layer | L7 (+ L4) | L4 + L7 | L7 | L4 |
| SSL Termination | Yes | Yes | Yes | Yes |
| Health Checks | Basic (Plus: advanced) | Excellent | Good | Good |
| Routing | URL, host, regex | ACL-based | URL, host, header | Port/IP |
| Auto-scaling | Manual | Manual | Automatic | Automatic |
| Static IP | No | No | No (use NLB) | Yes |
| Managed | No | No | Yes | Yes |
| Best for | Web server + LB | High-perf TCP/HTTP | AWS ecosystem | TCP, low latency |

---

## Rate Limiting Algorithms

### 1. Token Bucket

A bucket holds up to `capacity` tokens. Tokens are added at a fixed rate (`refill_rate` per second). Each request consumes one token. If the bucket is empty, the request is rejected or queued.

```python
class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True  # allowed
        return False  # rejected
```

**Allows bursting:** A bucket at full capacity allows a burst of `capacity` requests immediately, then sustains `refill_rate` requests/second.

**Pros:** Smooth average rate, allows controlled bursts.
**Cons:** Requires per-user state; distributed implementation needs atomic operations.

**Used by:** AWS API Gateway, Stripe API.

### 2. Leaky Bucket

Requests enter a queue (the bucket) and are processed at a fixed output rate. If the queue is full, excess requests are dropped.

```
Requests -> [Queue: max 100] -> Process at 50 req/sec -> Backend
                |
              DROP if full
```

**Pros:** Strictly enforces a constant output rate; smooths out traffic spikes.
**Cons:** Does not allow any bursting; queue adds latency; memory for the queue.

**Used by:** Traffic shaping in networking equipment.

**Token Bucket vs Leaky Bucket:**
- Token bucket: allows bursts, enforces average rate.
- Leaky bucket: no bursts, enforces constant rate.

### 3. Fixed Window Counter

Divide time into fixed windows (e.g., each minute). Count requests in the current window. Reject if count exceeds limit.

```python
def is_allowed(user_id, limit=100, window_seconds=60):
    window_key = f"{user_id}:{int(time.time() / window_seconds)}"
    count = redis.incr(window_key)
    if count == 1:
        redis.expire(window_key, window_seconds)
    return count <= limit
```

**Pros:** Very simple and memory-efficient.
**Cons:** Boundary problem — a user can make 100 requests at 00:59 and 100 more at 01:01, sending 200 requests in 2 seconds.

```
Window 1: [00:00 - 01:00]  100 requests at 00:59
Window 2: [01:00 - 02:00]  100 requests at 01:01
-> 200 requests in 2 seconds, despite 100/minute limit
```

### 4. Sliding Window Log

Store a timestamp for each request in a sorted set. Count requests in the sliding window `[now - window, now]`.

```python
def is_allowed(user_id, limit=100, window_seconds=60):
    now = time.time()
    window_start = now - window_seconds
    key = f"rate:{user_id}"

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)       # remove old entries
    pipe.zadd(key, {str(now): now})                    # add current request
    pipe.zcard(key)                                    # count in window
    pipe.expire(key, window_seconds)
    results = pipe.execute()

    return results[2] <= limit
```

**Pros:** Precise, no boundary problem.
**Cons:** Memory-intensive — stores a timestamp per request. For 100 req/min limit, stores up to 100 entries per user.

### 5. Sliding Window Counter

Approximation that combines fixed windows with interpolation. Memory-efficient.

```
current_window_count + previous_window_count * (overlap_ratio)

Example:
- Window: 60 seconds
- Previous window: 80 requests
- Current window (30s in): 40 requests
- Overlap with previous: 50%

Estimated count = 40 + 80 * 0.5 = 80 requests
```

```python
def is_allowed(user_id, limit=100, window_seconds=60):
    now = time.time()
    current_window = int(now / window_seconds)
    prev_window = current_window - 1
    elapsed_in_current = (now % window_seconds) / window_seconds

    current_count = int(redis.get(f"{user_id}:{current_window}") or 0)
    prev_count = int(redis.get(f"{user_id}:{prev_window}") or 0)

    estimated = current_count + prev_count * (1 - elapsed_in_current)
    
    if estimated < limit:
        redis.incr(f"{user_id}:{current_window}")
        return True
    return False
```

**Pros:** Memory-efficient (2 keys per user), accurate within ~1% of sliding window log.
**Cons:** Approximation, not exact.

---

## Rate Limiting: Where to Implement

### Client-Side Rate Limiting

Implemented in the client SDK. Prevents excessive requests from being sent.

```python
class RateLimitedClient:
    def __init__(self, rate_limit=10):
        self.bucket = TokenBucket(capacity=10, refill_rate=rate_limit)
    
    def request(self, endpoint):
        if not self.bucket.consume():
            raise RateLimitException("Client-side rate limit exceeded")
        return http.get(endpoint)
```

**Use case:** SDKs, mobile apps, batch jobs. Reduces unnecessary network traffic.
**Limitation:** Easily bypassed; cannot protect the server from malicious clients.

### API Gateway Rate Limiting

The API gateway is the best place for rate limiting in a microservices architecture.

```
Client -> [API Gateway: rate limit per user/IP/API key] -> Microservices
```

**Benefits:**
- Centralized enforcement — all services protected automatically.
- No code changes in individual services.
- Can enforce different limits per tier (free: 100/min, pro: 10000/min).

**Tools:** Kong, AWS API Gateway, Apigee, Nginx with lua-resty-limit-traffic.

### Service-Level Rate Limiting

Individual microservices enforce their own rate limits as a defense-in-depth measure.

**Use case:** Protecting a specific expensive resource (e.g., the ML inference endpoint only allows 10 requests/second per user, regardless of what the gateway allows).

### Distributed Rate Limiting with Redis

In a multi-instance deployment, each instance must share rate limit state.

```
Instance 1  \
Instance 2  -> [Redis: atomic counter per user] 
Instance 3  /
```

**Atomic Redis operations:**

```python
# Using Redis Lua script for atomic sliding window
lua_script = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1  -- allowed
else
    return 0  -- rejected
end
"""
```

**Rate limit response headers (standard practice):**

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1699999999
Retry-After: 30  (only on 429 responses)
```

---

## Throttling Strategies

### Hard Limit (Reject)

Requests exceeding the limit are immediately rejected with HTTP 429 Too Many Requests.

```
Request 101 in window of 100 -> 429 Too Many Requests
```

**Pros:** Simple, predictable, protects the server.
**Cons:** Poor user experience for legitimate users who occasionally burst.

### Soft Limit (Degraded Service)

Requests exceeding the limit receive a degraded response (e.g., lower priority, cached response, reduced feature set).

```
Free tier at limit -> return cached data (1 minute old)
Premium tier at limit -> return real-time data with a delay
```

### Queue-Based Throttling

Excess requests are queued rather than rejected. They are processed when capacity is available.

```
Normal traffic:   Request -> [LB] -> Server (immediate)
Throttled traffic: Request -> [Queue] -> Server (after delay)
```

**Pros:** No requests lost; better for batch or non-latency-sensitive workflows.
**Cons:** Queue can grow unbounded; requests may time out while queued; memory pressure.

**Implementation:** SQS, RabbitMQ, Celery queues with rate-limited consumers.

### Priority Throttling

Different request classes have different limits. Premium users or critical operations bypass throttling.

```python
def get_rate_limit(user):
    if user.tier == "enterprise":
        return 10000  # per minute
    elif user.tier == "pro":
        return 1000
    else:
        return 100
```

---

## DDoS Protection at Load Balancer Layer

### Types of DDoS Attacks

| Layer | Attack Type | Example |
|---|---|---|
| L3/L4 | Volumetric | UDP flood, ICMP flood, SYN flood |
| L4 | Protocol | TCP state exhaustion |
| L7 | Application | HTTP flood, Slowloris, DNS amplification |

### L3/L4 Protection

**SYN cookies:** When SYN flood detected, LB uses SYN cookies to avoid allocating state until handshake completes.

**Traffic scrubbing:** Upstream provider (AWS Shield, Cloudflare) absorbs volumetric attacks before they reach the LB.

**Rate limiting by IP:** Limit new connections per IP per second.

```nginx
# Nginx: limit connection rate
limit_conn_zone $binary_remote_addr zone=addr:10m;
limit_conn addr 100;  # max 100 concurrent connections per IP

limit_req_zone $binary_remote_addr zone=req:10m rate=10r/s;
limit_req zone=req burst=20 nodelay;
```

### L7 Protection

**WAF (Web Application Firewall):** Inspect HTTP requests for malicious patterns (SQLi, XSS, bot signatures).

```
Cloudflare WAF -> [rules: block bad bots, SQL injection patterns]
AWS WAF        -> [managed rule groups + custom rules]
```

**Slowloris mitigation:** Attacker opens many HTTP connections and sends headers very slowly, exhausting the LB's connection pool.

```nginx
client_header_timeout 10s;  # Max time to receive full request headers
client_body_timeout 10s;    # Max time between body chunk reads
keepalive_timeout 65s;      # Close idle keepalive connections
```

**Challenge pages (Cloudflare):** Suspicious traffic receives a JavaScript challenge before being allowed through.

**IP reputation blocking:** Block known malicious IPs using threat intelligence feeds.

### AWS DDoS Protection

- **AWS Shield Standard** — Automatic protection against L3/L4 attacks, free for all AWS customers.
- **AWS Shield Advanced** — $3000/month + data transfer costs. Includes:
  - 24/7 DDoS response team (DRT)
  - Advanced attack detection
  - Financial protection (cost protection during attacks)
  - Integration with WAF

---

## Quick Reference Tables

### Load Balancing Algorithm Comparison

| Algorithm | State Required | Handles Heterogeneous Servers | Handles Variable Request Duration | Session Affinity | Best For |
|---|---|---|---|---|---|
| Round Robin | None | No | No | No | Homogeneous, short requests |
| Weighted Round Robin | Weights | Yes | No | No | Mixed instance types |
| Least Connections | Per-server count | Implicit | Yes | No | Long-lived connections |
| Weighted Least Conn | Per-server count + weights | Yes | Yes | No | Mixed instances, variable duration |
| Least Response Time | Response time + count | Yes | Yes | No | Latency-sensitive, degraded servers |
| IP Hash | Hash function | No | No | Yes (by IP) | Stateful apps (workaround) |
| Consistent Hashing | Hash ring | No | No | Yes (by key) | Distributed caches, sharding |

### Rate Limiting Algorithm Trade-offs

| Algorithm | Memory | Precision | Burst Handling | Complexity | Best For |
|---|---|---|---|---|---|
| Token Bucket | O(1) per user | High | Allows controlled burst | Medium | API rate limits (most common) |
| Leaky Bucket | O(queue size) | High | No burst, constant rate | Medium | Traffic shaping, queue-based |
| Fixed Window | O(1) per user | Low (boundary issue) | Potential double-rate at boundary | Low | Simple, non-critical limits |
| Sliding Window Log | O(limit) per user | Exact | No (counts all) | Medium | Strict limits, low-volume users |
| Sliding Window Counter | O(1) per user | ~99% accurate | No | Low | Production rate limiting at scale |

### HTTP Rate Limit Response Headers

| Header | Meaning |
|---|---|
| X-RateLimit-Limit | Total requests allowed in window |
| X-RateLimit-Remaining | Requests remaining in current window |
| X-RateLimit-Reset | Unix timestamp when window resets |
| Retry-After | Seconds until client can retry (on 429) |

### Load Balancer Decision Guide

```
Need to route based on URL/headers?
  YES -> L7 (ALB, Nginx, HAProxy HTTP)
  NO  ->
    Need static IP or ultra-low latency?
      YES -> L4 (NLB, HAProxy TCP)
      NO  -> L7 is fine

On AWS?
  Web/API (HTTP/HTTPS, gRPC)  -> ALB
  TCP/UDP, static IP needed   -> NLB
  Self-managed on EC2         -> Nginx or HAProxy

Traffic is global (multi-region)?
  DNS-based, slow failover OK -> Route 53 GeoDNS
  Fast failover needed        -> Anycast (Cloudflare) + regional LBs
```
