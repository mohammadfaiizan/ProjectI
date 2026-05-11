# HLD Interview Q&A — File 18: CDN and Content Delivery

> 20 questions across Easy (Q1–7), Medium (Q8–15), and Hard (Q16–20).
> Each answer is 150–300+ words with diagrams, tables, or code where helpful.

---

## EASY (Q1–Q7)

---

### Q1. How does a CDN work? What are edge nodes, PoPs, and origin servers?

**Answer:**

A **CDN (Content Delivery Network)** is a globally distributed network of servers that caches and delivers content from locations physically close to end users, reducing latency and origin server load.

**Core components:**

**Origin Server:** Your main server (or cluster) — the authoritative source of content. Could be AWS EC2, your own data center, or an S3 bucket.

**PoP (Point of Presence):** A data center location where CDN edge servers are deployed. Major CDNs have 100–300+ PoPs globally (Cloudflare has 300+, Akamai has 3000+).

**Edge Node:** An individual server within a PoP that caches content and serves requests. Sits physically close to end users.

**Request flow:**
```
Without CDN:
  User in Tokyo → Origin Server in Virginia → 150ms RTT

With CDN:
  User in Tokyo → CDN Edge Node in Tokyo → 5ms RTT
                  └── (if cache miss) → Origin in Virginia → 150ms (once)
                      └── Cache stored at Tokyo edge → all future requests: 5ms
```

**DNS-based routing:**
```
1. User browser: DNS lookup for cdn.example.com
2. CDN's authoritative DNS returns IP of nearest edge node
   (based on user's resolver IP / anycast)
3. User connects to that edge node directly
```

**Cache hierarchy:**
```
Edge Node (L1 cache) → Regional Cache (L2) → Origin Shield → Origin
   ↑                        ↑                    ↑
  Fastest               Second tier          Last resort before origin
```

**Benefits:**
- Latency reduction (geographical proximity)
- Origin offload (CDN absorbs 90%+ of traffic)
- DDoS mitigation (distributed absorption)
- Bandwidth cost reduction (CDN buys bandwidth in bulk)
- Availability (CDN serves stale content if origin is down)

---

### Q2. What is the difference between Push CDN and Pull CDN?

**Answer:**

**Pull CDN (most common):**

Content is pulled from origin on the first cache miss. The CDN acts lazily — it only fetches content when a user first requests it.

```
1. User requests image.png from CDN edge node
2. Edge node: cache miss → forward request to origin
3. Origin returns image.png to edge node
4. Edge node caches image.png (per Cache-Control TTL)
5. Edge node returns image.png to user
6. All subsequent users get cache hit from edge node
```

**Pros:** Simple setup, no pre-population needed, only stores content that's actually requested
**Cons:** First user after cache expiry sees cache miss latency ("cold start")

**Push CDN:**

You proactively push content to CDN edge nodes before any user requests it.

```
1. You publish new video to origin
2. Your system calls CDN API: PUT /content/video123.mp4
3. CDN distributes video to all/selected edge nodes immediately
4. All users get cache hits from the first request
```

**Pros:** No cache misses, predictable performance, works for content that must be fast on first access
**Cons:** Requires management of what to push/delete, storage costs for all content everywhere, complex invalidation

**Decision Guide:**

| Factor | Pull CDN | Push CDN |
|---|---|---|
| Content type | Dynamic/frequently changing | Static/large files |
| Traffic pattern | Unpredictable | Predictable, uniform access |
| Content volume | Large catalog, sparse access | Smaller catalog, high access |
| First-request latency | Acceptable | Must be fast |
| Management complexity | Low | High |
| Best for | Websites, APIs, images | Software downloads, video files, firmware updates |

**Hybrid:** Most platforms use pull CDN for web content + push CDN for large media files (video game downloads, OS updates).

---

### Q3. What do Cache-Control headers mean? Explain max-age, s-maxage, no-cache, no-store, and must-revalidate.

**Answer:**

`Cache-Control` is an HTTP header that governs caching behavior for both browsers (private caches) and CDNs/proxies (shared caches).

**Directives explained:**

```http
Cache-Control: max-age=3600, s-maxage=86400, must-revalidate
```

**max-age=N:**
- Content is fresh for N seconds from the response time
- Applies to ALL caches (browser + CDN)
- `max-age=3600` → cache for 1 hour

**s-maxage=N:**
- Overrides `max-age` for **shared caches** (CDNs, proxies) only
- Browser ignores `s-maxage` and uses `max-age`
- Allows CDN to cache longer than browsers: `max-age=60, s-maxage=86400`
  - Browser: re-fetch every minute
  - CDN: cache for 24 hours

**no-cache:**
- Misleading name — does NOT mean "don't cache"
- Means: cache the content but **revalidate** with origin on every request
- CDN stores it but sends a conditional request (`If-None-Match`) to origin
- If origin says "304 Not Modified", CDN serves cached version
- Ensures freshness while avoiding full re-download

**no-store:**
- True "don't cache" — neither browser nor CDN should store the response
- Used for sensitive content: bank statements, medical records, auth tokens
- Every request hits the origin

**must-revalidate:**
- Once the content expires (past max-age), it MUST be revalidated — do not serve stale
- Without this, caches may serve stale content when the origin is unreachable
- Important for: inventory pages, pricing, legal content

**Real-world examples:**
```http
# Static asset with content hash in URL (e.g., app.a3b4c5.js)
Cache-Control: public, max-age=31536000, immutable

# HTML page
Cache-Control: no-cache  # Always revalidate, but cache for conditional requests

# User dashboard (personalized)
Cache-Control: private, max-age=0, must-revalidate

# API response (can CDN cache but not browser)
Cache-Control: public, s-maxage=60, max-age=0
```

---

### Q4. What are the strategies for CDN cache invalidation?

**Answer:**

CDN cache invalidation is notoriously hard (it's one of the "two hard problems in computer science"). Here are the main strategies:

**1. TTL Expiry (Time-to-Live):**

The simplest approach: set a TTL, wait for content to expire naturally.
```http
Cache-Control: max-age=3600  # Expire after 1 hour
```
- No action needed from your side
- Stale content served until TTL expires
- TTL is a tradeoff: long TTL = stale content; short TTL = more origin hits

**2. Cache Purge / Invalidation API:**
```bash
# Cloudflare purge specific URL
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "Authorization: Bearer {token}" \
  -d '{"files": ["https://example.com/image.png"]}'

# Or purge by cache tag (more scalable)
curl -d '{"tags": ["product-category-electronics"]}'
```

**Pros:** Immediate effect, surgical control
**Cons:** API calls cost money (Cloudflare charges per purge), can overload origin if mass invalidation triggers cache rebuilding

**3. Versioned URLs (Cache Busting):**

Embed content hash or version in the URL. Old URL keeps cached, new URL has no cache.
```html
<!-- Old -->
<img src="/images/product.jpg">

<!-- New — hash changes when content changes -->
<img src="/images/product.a3f4b2.jpg">
```

**Pros:** No TTL management, zero propagation delay, simple
**Cons:** Requires build pipeline integration (webpack/Vite handle this), URL references must be updated

**4. Surrogate Keys / Cache Tags:**
Tag cached objects with logical identifiers:
```http
# Origin response headers
Surrogate-Key: product-123 category-electronics user-456
Cache-Tag: product-123 category-electronics
```

```bash
# Purge all cached objects tagged with product-123
curl -d '{"tags": ["product-123"]}'
# Instantly invalidates product page, product images, API responses — all at once
```

**Best practice:** Use versioned URLs for static assets (JS/CSS/images), TTL for dynamic content, purge API for breaking updates.

---

### Q5. How does adaptive bitrate streaming (HLS/DASH) work?

**Answer:**

**Adaptive Bitrate (ABR) Streaming** adjusts video quality in real-time based on the viewer's current network conditions, eliminating buffering.

**How it works:**

```
Original Video → Encoder → Multiple Quality Variants
                            ├── 1080p @ 8 Mbps
                            ├── 720p  @ 4 Mbps
                            ├── 480p  @ 1.5 Mbps
                            └── 360p  @ 0.7 Mbps

Each variant is split into segments (2-6 seconds each):
  1080p/seg001.ts, 1080p/seg002.ts, ...
  720p/seg001.ts,  720p/seg002.ts,  ...
```

**Manifest file (HLS .m3u8):**
```
#EXTM3U
#EXT-X-VERSION:3

#EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080
1080p/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=4000000,RESOLUTION=1280x720
720p/playlist.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=854x480
480p/playlist.m3u8
```

**Segment playlist (720p/playlist.m3u8):**
```
#EXTM3U
#EXT-X-TARGETDURATION:6
#EXTINF:6.0,
seg001.ts
#EXTINF:6.0,
seg002.ts
#EXTINF:6.0,
seg003.ts
```

**Player adaptive algorithm:**
```javascript
// Simplified ABR logic
function selectNextQuality(bufferLevel, downloadedSegmentBitrate) {
    if (bufferLevel < 5) {
        // Low buffer — select lowest quality to avoid rebuffer
        return LOWEST_QUALITY;
    } else if (bufferLevel > 20 && downloadedSegmentBitrate > currentBitrate * 1.5) {
        // High buffer + fast network → step up quality
        return stepUp(currentQuality);
    } else if (downloadedSegmentBitrate < currentBitrate * 0.8) {
        // Slow network → step down quality
        return stepDown(currentQuality);
    }
    return currentQuality;
}
```

**HLS vs DASH:**

| Aspect | HLS (HTTP Live Streaming) | DASH (Dynamic Adaptive Streaming) |
|---|---|---|
| Origin | Apple | MPEG standard |
| Format | .ts segments, .m3u8 manifest | .mp4 segments, .mpd manifest |
| DRM | FairPlay (Apple) | Widevine, PlayReady |
| Latency | 6-30s (Low Latency HLS: <2s) | 2-30s |
| Support | All Apple devices + modern browsers | All modern browsers except Safari |

**CDN optimization for HLS/DASH:** Segment files are perfect for CDN caching — each segment is immutable with a content-hash URL. Manifests (.m3u8/.mpd) have short TTLs (5-10s for live, immutable for VOD).

---

### Q6. What content is NOT suitable for CDN caching?

**Answer:**

Not all content benefits from CDN caching. Some content should bypass the CDN or be cached with extreme care.

**Content that should NOT be CDN-cached:**

**1. Personalized/User-specific responses:**
```http
# This varies per user — cannot be cached shared
GET /api/dashboard
Authorization: Bearer user_specific_token
→ Response: {"user": "alice", "balance": "$1,234.56"}
```
If cached, Alice's data could be served to Bob.

**2. Session-dependent pages:**
```http
GET /checkout/cart
Cookie: session_id=abc123
→ Varies per user; caching without vary on Cookie is dangerous
```

**3. Real-time data:**
```
Stock prices changing every millisecond
Live auction bids
Real-time inventory counts
```
Even a 5-second CDN cache makes these inaccurate.

**4. Write operations (POST/PUT/DELETE/PATCH):**
CDNs generally don't cache non-GET/HEAD requests. Writes must always reach origin.

**5. Payment and financial transactions:**
```
POST /api/payments/charge
→ Never cache; always hits origin
```

**6. Authentication endpoints:**
```
POST /api/auth/login
→ Must hit origin; caching credentials is a security nightmare
```

**7. Highly dynamic search results:**
```
GET /search?q=iphone&sort=price&filter=in_stock&page=1
→ Too many permutations; rarely repeated exactly
```

**Correct use of `Cache-Control` to prevent CDN caching:**
```http
# User-specific page
Cache-Control: private, no-store

# Real-time data
Cache-Control: no-cache, no-store, must-revalidate

# Or just let the CDN respect Vary header
Vary: Cookie, Authorization
```

**Edge cases:** Some CDNs can cache authenticated content using signed tokens or edge-side personalization (ESI) — but this requires careful security review.

---

### Q7. How do CDNs protect against DDoS attacks?

**Answer:**

CDNs are among the most effective DDoS mitigation tools because they distribute traffic across a massive network and have purpose-built protection layers.

**Layers of CDN DDoS protection:**

**1. Anycast network diffusion:**
```
Attack: 500 Gbps DDoS targeting example.com
CDN has 300 PoPs globally, each absorbing traffic

Without CDN: 500 Gbps hits single origin → dead
With CDN: 500 Gbps spread across 300 PoPs → ~1.7 Gbps per PoP → absorbed
```

**2. Rate limiting at edge:**
```
Rules:
  - Max 100 requests/IP/second
  - Max 1000 requests/IP/minute
  - Block IPs with > 10K requests/5 minutes
```

**3. WAF (Web Application Firewall):**
```
Block patterns:
  - SQL injection: '  OR  1=1
  - XSS: <script>alert(1)</script>
  - Log4Shell: ${jndi:ldap://...}
  - OWASP Top 10 signatures
```

**4. Bot detection and challenge:**
```
Suspicious traffic → CAPTCHA challenge (Cloudflare Turnstile)
                   → JavaScript challenge (browser fingerprinting)
                   → IP reputation check (known bad actors)
```

**5. Signed URLs / Hotlink prevention:**
```python
# Generate signed URL that expires in 1 hour
import hmac, hashlib, time

def generate_signed_url(resource, secret_key, ttl=3600):
    expiry = int(time.time()) + ttl
    signature = hmac.new(
        secret_key.encode(),
        f"{resource}:{expiry}".encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{resource}?expires={expiry}&sig={signature}"
```

**6. IP reputation and geoblocking:**
- Block known botnet IPs
- Geoblocking: if your service doesn't operate in Russia, block Russian IPs at CDN edge

---

## MEDIUM (Q8–Q15)

---

### Q8. What is an origin shield, and how does it protect origin servers?

**Answer:**

An **origin shield** (also called a "shield PoP" or "mid-tier cache") is an additional caching layer inserted between edge nodes and the origin server. It acts as a single point of contact for the origin, consolidating cache fill requests.

**Without origin shield:**
```
Edge Node - Tokyo        ─┐
Edge Node - Singapore    ─┤── All cache misses → Origin Server
Edge Node - Sydney       ─┘   (many requests, many origin hits)
```

**With origin shield:**
```
Edge Node - Tokyo      ─┐
Edge Node - Singapore  ─┼── All cache misses → Shield PoP (Singapore)
Edge Node - Sydney     ─┘                           │
                                                     └── ONE request to Origin
                                                         (if shield also misses)
```

**Benefits:**

1. **Origin offload:** A 100K request/second traffic spike from Asia Pacific hits the shield PoP once per unique URL, not 100K times at origin.

2. **Thundering herd protection:** When cache expires (TTL), multiple edge nodes race to fill — without shield, all hit origin simultaneously. With shield, only the shield node hits origin.

3. **Connection optimization:** Shield maintains a small pool of persistent connections to origin. Edge nodes can handle millions of user connections while only a few reach the origin.

4. **Bandwidth savings:** Traffic between edge nodes and shield is often cheaper (CDN backbone vs public internet).

**Configuration (Fastly example):**
```
CDN Configuration:
  Origin: api.mycompany.com
  Shield PoP: New York (closest to origin in Virginia)
  
  Cache hierarchy:
    Edge miss → New York shield
    Shield miss → origin (api.mycompany.com)
    Shield hit → return to edge immediately
```

**When to use origin shield:**
- Origin is in one region (shield in same region reduces origin load)
- Content has reasonable TTL (shield can actually cache it)
- High traffic with bursty patterns (flash sales, viral content)

**When NOT to use:**
- Purely private/authenticated content (bypasses cache anyway)
- Low-latency requirements where the extra hop hurts

---

### Q9. How do CDN and edge computing (Lambda@Edge, Cloudflare Workers) differ, and what are the use cases?

**Answer:**

Traditional CDN caches static responses. **Edge computing** runs code at CDN edge nodes, enabling dynamic logic without round-tripping to the origin.

**Traditional CDN:**
```
Request → Edge Node → (cache hit?) → Return cached response
                    → (cache miss?) → Forward to origin
```

**Edge Computing:**
```
Request → Edge Node → Execute JavaScript/WASM code
                       ├── Modify request (add headers, auth)
                       ├── Generate response (no origin needed)
                       ├── A/B test redirect
                       ├── Personalize content
                       └── Authenticate/authorize at edge
```

**Lambda@Edge (AWS CloudFront):**
Runs Node.js/Python at CloudFront edge in 4 event hooks:
```
Viewer Request → Origin Request → Origin Response → Viewer Response
    ↑                  ↑                 ↑                ↑
[Modify req]    [Cache key]       [Transform]      [Modify response]
```

**Cloudflare Workers:**
Runs JavaScript at 300+ PoPs, <1ms cold start:
```javascript
addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
    // A/B testing at edge
    const testGroup = Math.random() < 0.5 ? 'A' : 'B';
    
    if (testGroup === 'B') {
        const url = new URL(request.url);
        url.pathname = '/new-homepage';
        request = new Request(url, request);
    }
    
    // Authentication at edge — no origin round-trip
    const token = request.headers.get('Authorization');
    if (!validateJWT(token)) {
        return new Response('Unauthorized', { status: 401 });
    }
    
    return fetch(request);
}
```

**Use cases for edge computing:**

| Use Case | Traditional CDN? | Edge Computing? |
|---|---|---|
| Static file serving | Yes | No |
| Image resizing/optimization | Partial | Yes |
| A/B testing | No | Yes |
| Authentication/JWT validation | No | Yes |
| Geolocation-based redirects | Partial | Yes |
| Bot detection | Partial | Yes |
| SSR (Server-Side Rendering) | No | Yes (Cloudflare Workers) |
| Personalization | No | Yes |

**Limitations:** Edge computing has strict limits (CPU time, memory, no persistent state beyond KV store). Complex business logic still belongs at origin.

---

### Q10. How does a multi-CDN strategy work, and when is it worth the complexity?

**Answer:**

A **multi-CDN** strategy uses two or more CDN providers simultaneously to improve performance, resilience, and cost.

**Architecture:**
```
DNS (Traffic Manager / Global Load Balancer)
  ├── Cloudflare CDN (40% traffic)
  ├── Fastly CDN (35% traffic)
  └── Akamai CDN (25% traffic)
```

**Traffic distribution strategies:**

**1. DNS-based weighted routing:**
```
Route 53 weighted records:
  cdn1.example.com → 40 weight (Cloudflare)
  cdn2.example.com → 35 weight (Fastly)
  cdn3.example.com → 25 weight (Akamai)
```

**2. Performance-based routing:**
```
Measure latency to each CDN from user location
Route to fastest CDN for that geography

e.g., Southeast Asia → Cloudflare (better coverage)
      Europe → Akamai (more PoPs)
      Latin America → Fastly
```

**3. Failover routing:**
```
Primary: Fastly
If Fastly availability < 99.5% for 5 minutes:
  → Failover to Cloudflare
```

**Benefits:**
- **Resilience:** CDN outages are rare but catastrophic without backup (Fastly outage in 2021 took down Reddit, GitHub, Twitch, NY Times for ~1 hour)
- **Performance:** Different CDNs have different strengths per geography
- **Cost optimization:** Leverage competitive pricing; shift traffic to cheaper provider
- **Vendor negotiation leverage:** Not locked into one provider

**Challenges:**
```
1. Cache inconsistency:
   User gets version A from Fastly, refreshes, gets version B from Cloudflare
   Solution: ensure cache invalidation on ALL CDNs, or use versioned URLs

2. Observability complexity:
   Logs split across multiple vendors
   Need unified monitoring layer (e.g., Datadog synthetic monitoring to all CDNs)

3. Cost:**
   Multi-CDN management tools (NS1, Cedexis, Imperva) add cost
   Engineering overhead to maintain configurations in sync
```

**When to use multi-CDN:**
- Revenue per minute of downtime is very high (e.g., e-commerce checkout flow)
- Serving users in regions where one CDN has weak coverage
- CDN spend > $100K/month (ROI for optimization)

---

### Q11. How do video streaming platforms like Netflix and YouTube use CDN?

**Answer:**

Netflix and YouTube have very different CDN strategies — one built a proprietary CDN, the other uses a hybrid.

**Netflix — Open Connect (proprietary CDN):**

Netflix runs its own CDN hardware called **Open Connect Appliances (OCAs)** that it installs directly in ISP data centers worldwide.

```
Netflix Architecture:
  AWS (Content ingestion + transcoding + control plane)
         ↓
  Open Connect Appliances (inside ISPs)
         ↓
  ISP backbone → Last mile → Customer

OCA capabilities:
  - 100-200TB storage per appliance
  - 100Gbps+ network interfaces
  - Runs Netflix's custom cache/delivery software
```

**Why Netflix built its own CDN:**
1. **Cost:** At Netflix's scale, CDN bandwidth costs are enormous. ISPs host OCAs for free (saves their egress costs too)
2. **Quality control:** Direct relationship with ISPs; can negotiate and optimize peering
3. **Cache hit rate:** Netflix content is predictable (a new movie release → millions watch same file) → very high cache hit rates with proactive content pushing
4. **Personalization:** Can route user to nearest OCA based on account/device data

**Netflix proactive caching:**
```
Title release day - 1:
  Netflix: "Breaking Bad S1 will be popular tomorrow"
  Push all episodes to OCAs worldwide during off-peak hours
  
Next day: 90%+ of Breaking Bad traffic served from OCAs within ISP
  Origin (AWS S3): minimal traffic
```

**YouTube — Google's global network:**
YouTube uses Google's own fiber backbone network (not a CDN product) plus Google's edge caches embedded at ISPs, similar to Netflix's model. Google has peering relationships with virtually every major ISP.

**Key insight:** At YouTube/Netflix scale, the economics favor building your own infrastructure. CDN providers are essentially doing the same thing but as a service — you're paying their margin.

---

### Q12. How do you handle CDN caching for authenticated content?

**Answer:**

Authenticated content is the hardest CDN use case because it's personalized and the CDN must not serve user A's data to user B.

**Option 1: Bypass CDN entirely (simple but wasteful)**
```http
Cache-Control: private, no-store
```
Every request hits origin. Safe but loses CDN benefits for latency reduction.

**Option 2: Segment public vs private content**
```
Public (CDN cached): product images, page templates, static assets
Private (bypass CDN): user data, shopping cart, payment info

Page rendering:
  Serve cached page shell → Load personalized content via API call
  (ESI pattern or client-side data fetching)
```

**Option 3: Signed URLs for private media**
```python
# Generate time-limited signed URL for private content
import hmac, hashlib, base64, time

def sign_url(resource_url, user_id, ttl_seconds=3600):
    expiry = int(time.time()) + ttl_seconds
    message = f"{resource_url}:{user_id}:{expiry}"
    signature = hmac.new(
        CDN_SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{resource_url}?uid={user_id}&exp={expiry}&sig={signature}"

# CDN edge validates signature (via edge function or CDN-level signing)
# If valid → serve content
# If expired or invalid → 403 Forbidden
```

**Option 4: Edge-side authentication (Cloudflare Workers)**
```javascript
// Validate JWT at edge, then set internal header for origin
async function handleRequest(request) {
    const jwt = request.headers.get('Authorization')?.replace('Bearer ', '');
    const user = await validateJWT(jwt);  // Validate against shared secret
    
    if (!user) return new Response('Unauthorized', { status: 401 });
    
    // Add user context header for origin, remove auth header for caching
    const modifiedRequest = new Request(request.url, {
        ...request,
        headers: {
            'X-User-Id': user.id,
            'X-User-Role': user.role,
            // Don't forward Authorization header — prevents cache poisoning
        }
    });
    
    return fetch(modifiedRequest);
}
```

**Option 5: Vary header (per-user cache segments)**
```http
Vary: Authorization
```
Dangerous: CDN creates separate cache entry per Authorization value → cache is useless, and leaks cache key information.

**Best practice:** Serve authenticated pages as private, but maximize CDN use for shared assets. Use signed URLs for private media (profile photos, private documents).

---

### Q13. How do you measure CDN effectiveness? What are cache hit ratio and origin offload?

**Answer:**

**Key CDN metrics:**

**1. Cache Hit Ratio (CHR):**
```
CHR = Cache Hits / (Cache Hits + Cache Misses) × 100%

Example:
  Total requests:  10,000,000
  Cache hits:       9,200,000
  Cache misses:       800,000
  CHR = 92%
```

A good CDN should achieve 85-99% CHR for static assets, depending on content type and TTL settings.

**2. Origin Offload:**
```
Origin Offload = (1 - Cache Misses / Total Requests) × 100%
= Same as CHR but framed from origin's perspective

If CHR = 92% → Origin offload = 92% → Origin handles only 8% of traffic
```

**3. Byte Hit Ratio:**
```
Byte Hit Ratio = Bytes served from cache / Total bytes served

Important because large files (video) skew hit ratio:
  1,000 small CSS file hits + 1 large 100MB video miss
  Request CHR = 99.9%, but Byte CHR could be much lower
```

**4. TTFB (Time to First Byte) — edge vs origin:**
```
Good CDN performance:
  Edge TTFB: 10-50ms
  Origin TTFB: 100-500ms

If edge TTFB ≈ origin TTFB → CDN isn't routing to nearest edge
```

**5. Error rates by CDN PoP:**
```
Monitor 5xx error rates per edge location
Sudden spike in errors at specific PoP → potential issue
```

**Dashboard query (pseudo-SQL):**
```sql
SELECT
    cdn_pop,
    COUNT(*) as total_requests,
    SUM(CASE WHEN cache_status = 'HIT' THEN 1 ELSE 0 END) as hits,
    AVG(ttfb_ms) as avg_ttfb,
    ROUND(SUM(CASE WHEN cache_status = 'HIT' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as hit_ratio
FROM cdn_access_logs
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY cdn_pop
ORDER BY hit_ratio ASC;  -- Find worst-performing PoPs
```

**Common causes of low CHR:**
- TTL too short
- Vary header too broad (Vary: Cookie creates too many cache keys)
- Too many unique query parameters in URLs
- High proportion of authenticated/private content

---

### Q14. How does anycast routing work in CDNs?

**Answer:**

**Anycast** is a network routing method where the same IP address is announced from multiple locations simultaneously. The internet's routing infrastructure (BGP) automatically routes packets to the topologically nearest announcer.

**Unicast vs Anycast:**
```
Unicast:
  User in Tokyo queries 203.0.113.1
  That IP belongs to ONE server in Virginia → 150ms RTT

Anycast:
  User in Tokyo queries 198.51.100.1
  That same IP is announced from:
    - Tokyo PoP
    - Singapore PoP
    - Sydney PoP
    - Virginia PoP
  Internet routes Tokyo user to Tokyo PoP → 5ms RTT
  Internet routes London user to Frankfurt PoP → 8ms RTT
```

**How it works (BGP):**
```
Cloudflare Tokyo PoP:
  Announces via BGP: "I can route to 104.16.0.0/12"
  
Cloudflare Frankfurt PoP:
  Announces via BGP: "I can route to 104.16.0.0/12"

Internet (BGP):
  Tokyo ISP: shortest BGP path to 104.16.0.0/12 → Tokyo PoP
  London ISP: shortest BGP path to 104.16.0.0/12 → Frankfurt PoP
```

**Failover:** If Tokyo PoP goes down, it stops announcing the prefix. BGP converges (typically 30-90 seconds) and Tokyo users are rerouted to the next-nearest PoP.

**Anycast for DNS:**
```
8.8.8.8 (Google DNS) → Anycast
1.1.1.1 (Cloudflare DNS) → Anycast

Why DNS needs anycast:
  DNS resolution happens on every single domain lookup
  Anycast ensures DNS is always fast regardless of user location
```

**Limitation — stateful connections:**
UDP (DNS) works perfectly with anycast. TCP is trickier: BGP route changes during a connection could send packets to a different server. Modern CDNs handle this with **connection migration** or by using anycast only for the initial handshake, then switching to unicast for the session.

---

### Q15. What are CDN cost optimization strategies?

**Answer:**

CDN costs are typically charged per GB of data transferred + per request. At scale (Netflix: petabytes/day, YouTube: exabytes), optimization is critical.

**1. Image optimization:**
```
PNG (1.2MB) → WebP (320KB) → AVIF (180KB)
Savings: 75-85% bandwidth reduction

Serve modern formats conditionally:
Accept: image/avif,image/webp,image/png  ← browser sends this
CDN/edge: serve best format browser supports
```

**2. Compression:**
```http
# Ensure CDN compresses text assets
Content-Encoding: gzip       # ~70% reduction for HTML/CSS/JS
Content-Encoding: br         # Brotli: ~85% reduction (better than gzip)
```

**3. Long TTLs for versioned assets:**
```http
# Content-hashed files: cache forever
Cache-Control: public, max-age=31536000, immutable
# No cache expiry = no origin hits = no bandwidth costs for stale content
```

**4. Tiered CDN pricing:**
```
Bandwidth pricing example (Cloudflare 2024):
  0-10 TB:   $0.085/GB
  10-50 TB:  $0.065/GB
  50+ TB:    $0.040/GB

Consolidating traffic to one CDN vs splitting across three
may actually be more cost-effective due to volume discounts
```

**5. Lazy loading:**
```javascript
// Don't load images until user scrolls to them
<img loading="lazy" src="/product.jpg">
// Reduces CDN requests for below-fold content that user never sees
```

**6. Cache warmth optimization:**
```python
# Find content with low cache hit ratio
low_chr_paths = analyze_logs(threshold=0.5)

# For high-value, low-CHR content:
# Option A: Increase TTL (if content doesn't change often)
# Option B: Use cache key normalization (strip tracking params)
# Option C: Prefetch/warm cache after publish
```

**7. CDN-level request collapsing:**
When multiple requests for the same uncached resource arrive simultaneously, CDN makes only ONE request to origin (request coalescing). Ensure your CDN supports this.

---

## HARD (Q16–Q20)

---

### Q16. How would you design a CDN-backed system to serve 1 billion daily media requests?

**Answer:**

1 billion requests/day = ~11,574 requests/second average, with peaks potentially 10-50x that.

**Scale assumptions:**
```
1B requests/day
Average object size: 200KB (mix of images, small videos)
Total data transferred: 1B × 200KB = 200TB/day
Peak throughput: 11,574 RPS × 10x = 115,740 RPS
Assume 95% CDN cache hit rate → 5.8% = ~6,700 origin hits/second
```

**Architecture:**
```
Global DNS (Anycast — Route 53/Cloudflare)
         ↓
Multi-CDN Layer (Cloudflare + Fastly + Akamai)
         ↓
Origin Shield (per-region, 4 regions)
         ↓
Object Storage (S3 / GCS — multi-region)
         ↓
Media Processing Pipeline (async)
```

**Detailed components:**

**Ingest pipeline:**
```python
# When user uploads media:
def ingest_media(raw_file, media_id):
    # 1. Store original in S3 (versioned)
    s3.put(f"originals/{media_id}", raw_file)
    
    # 2. Queue async processing
    sqs.send({
        "media_id": media_id,
        "operations": [
            {"type": "resize", "widths": [320, 640, 1280]},
            {"type": "convert", "formats": ["webp", "avif"]},
            {"type": "compress", "quality": 85}
        ]
    })

# Processing worker generates:
# /media/{id}/320.webp, /media/{id}/320.avif, /media/{id}/320.jpg
# /media/{id}/640.webp, etc.
```

**CDN configuration:**
```
URL structure (cache-key friendly):
  /media/{sha256_hash}/{width}.{format}
  e.g., /media/a3b4c5d6.../640.webp

Cache-Control: public, max-age=31536000, immutable
  (content-addressed URLs never change content)

CDN CNAME: media.example.com → cdn.fastly.com
           Fallback: media-backup.example.com → cdn.cloudflare.com
```

**Origin shield per region:**
```
US-East:  shield PoP in Northern Virginia (close to AWS us-east-1)
EU-West:  shield PoP in Frankfurt
AP-South: shield PoP in Singapore  
AP-North: shield PoP in Tokyo

Each shield maintains LRU cache of recently requested media
Shield miss → S3 (read-only; no compute)
```

**Handling traffic spikes (viral content):**
```python
# Detect viral content early (within minutes of upload)
class ViralContentDetector:
    def check(self, media_id):
        requests_last_5min = redis.get(f"req_count:{media_id}")
        if requests_last_5min > 10000:
            # Pre-warm all CDN edge nodes
            self.prefetch_to_all_pops(media_id)
            # Notify origin to increase readiness
```

**Cost at 200TB/day:**
```
CDN bandwidth: 200TB/day × $0.04/GB = $8,000/day = $2.9M/year
With 95% hit rate: origin handles 10TB/day
S3 egress to origin shield: 10TB × $0.09/GB = $900/day
Total CDN + storage: ~$3.2M/year (estimated)
```

---

### Q17. How does CDN handle WebSocket connections?

**Answer:**

WebSocket connections are fundamentally different from HTTP requests — they are long-lived, stateful, and require full-duplex communication. Traditional CDN caching doesn't apply.

**The challenge:**
```
Traditional CDN: Request → Edge → Cache → Response (done)
WebSocket CDN:   Handshake → Persistent connection → Data flows both ways
                 (minutes to hours long connection)
```

**CDN WebSocket proxy architecture:**
```
Client ←→ CDN Edge Node ←→ Origin WebSocket Server
                ↑
          TCP proxy (not HTTP cache)
          CDN forwards all frames transparently
```

**CDN capabilities for WebSocket:**

**1. TLS termination at edge:**
```
Client → [TLS encrypted] → CDN Edge (TLS terminated) → [CDN internal network] → Origin
Benefits: TLS handshake is geographically close to user (lower latency)
         Origin doesn't need TLS hardware (offloaded)
```

**2. Connection persistence and reuse:**
```
100K clients connect to CDN edge
CDN multiplexes over fewer TCP connections to origin
Edge ←→ Origin: 100 persistent connections (connection pooling)
vs.
Without CDN: 100K raw TCP connections to origin
```

**3. DDoS protection:**
```
CDN can detect WebSocket connection flood:
  - Rate limit new WebSocket handshakes per IP
  - Challenge suspicious IPs before allowing upgrade
  - Drop connections from known bad actors
```

**CDN routing for WebSocket (sticky sessions):**
```
Problem: If CDN routes different frames to different backends, state is lost
Solution: Consistent hashing by connection ID → always same backend

nginx sticky session:
upstream websocket_backends {
    ip_hash;  # OR hash $cookie_session_id;
    server backend1:8080;
    server backend2:8080;
}
```

**Limitations:**
- CDN cannot cache WebSocket frames (stateful, real-time)
- CDN adds marginal latency (2-5ms proxy overhead per round-trip)
- Long-lived connections consume CDN resources (concurrent connection limits)
- Geographic benefits are real but smaller than for HTTP (connection setup vs ongoing data)

**Best practice:** Use CDN for WebSocket TLS termination, DDoS protection, and routing. For very latency-sensitive WebSocket applications (trading, gaming), consider direct CDN bypass for the data path.

---

### Q18. How does CDN ensure correct crawling for SEO?

**Answer:**

Misconfigured CDN can severely harm SEO by serving stale content to crawlers, blocking bots, or creating duplicate content issues.

**Key CDN and SEO considerations:**

**1. Robots.txt accessibility:**
```
robots.txt must ALWAYS be fresh and accessible
Cache-Control: max-age=3600  (not too long — you may need to update urgently)

If CDN serves stale robots.txt that disallows Googlebot:
  → Google stops crawling your entire site until cache expires!
```

**2. Sitemap freshness:**
```http
# sitemap.xml — should be fresh but CDN-cacheable
Cache-Control: max-age=3600, must-revalidate
# 1 hour cache, but must validate with origin before serving stale
```

**3. Canonical URLs — CDN must not serve content on wrong domains:**
```
Problem: CDN might serve your content on multiple domains:
  https://example.com/page
  https://cdn.example.com/page
  https://www.example.com/page

All pointing to same content = duplicate content penalty

Solution: Canonical header from origin
  Link: <https://example.com/page>; rel="canonical"
  CDN must pass this through unchanged
```

**4. Status codes must be accurate:**
```
If origin returns 301/302 redirect:
  CDN must NOT cache this beyond the redirect TTL (or cache it very short)
  A cached 301 pointing to wrong URL is very hard to fix

If origin returns 404:
  Cache 404s briefly (5 minutes) to avoid hammering origin on bad URLs
  Cache-Control: max-age=300 on 404 responses
```

**5. Googlebot IP allowlisting:**
```
CDN WAF rules: NEVER block Googlebot
  Verify Googlebot: reverse DNS lookup confirms google.com
  Add exception: if User-Agent matches Googlebot AND reverse-DNS = googlebot.com
    → Allow through WAF, bypass rate limiting
```

**6. PageSpeed / Core Web Vitals (Google's ranking signal):**
```
CDN directly improves:
  ├── LCP (Largest Contentful Paint): faster image delivery from edge
  ├── TTFB (Time to First Byte): edge serves faster than origin
  └── CLS (Cumulative Layout Shift): image dimensions set correctly

CDN configuration for Core Web Vitals:
  - Enable HTTP/2 (multiplexing) or HTTP/3 (QUIC)
  - Enable Brotli compression
  - Enable HTTP Early Hints (103) to preload critical resources
```

---

### Q19. What is latency-based vs geolocation-based routing, and when does each fail?

**Answer:**

When a user makes a request, CDNs and DNS systems must decide which edge server to send them to. The two main approaches are geolocation-based and latency-based.

**Geolocation-based routing:**
```
User's IP → MaxMind/IP2Location database → Geographic location
User in Tokyo → Route to Tokyo PoP
User in London → Route to London PoP
```

**Implementation:**
```
Route 53 Geolocation routing policy:
  Australia → ap-southeast-2 (Sydney)
  Europe → eu-west-1 (Ireland)
  Default → us-east-1
```

**When geolocation fails:**
1. **VPN users:** VPN IP is in a different country than the physical user
2. **IP geolocation database inaccuracies:** ~1-2% error rate, can misroute by thousands of miles
3. **Satellite internet:** Starlink IPs may resolve to unexpected locations
4. **Offshore routing:** Some ISPs route traffic internationally before it exits nationally

**Latency-based routing:**
```
Measure actual RTT from candidate PoPs to user's recursive DNS resolver
Route to lowest-latency PoP

Route 53 Latency routing:
  Maintains real-time latency measurements per AWS region
  Routes to lowest-latency region dynamically
```

**When latency-based fails:**
1. **Resolver != user location:** Corporate networks use a central DNS resolver that may be geographically distant from the actual user
2. **Latency ≠ throughput:** Low latency PoP may be congested; higher latency PoP may provide better throughput for large files
3. **Cold start problem:** New PoPs have no latency data — initial routing is suboptimal

**Hybrid approach (Real User Monitoring):**
```javascript
// Measure actual performance to different CDN PoPs from the browser
// Report back to decision system
const probes = ['cdn-a.example.com', 'cdn-b.example.com', 'cdn-c.example.com'];
for (const cdn of probes) {
    const start = performance.now();
    await fetch(`https://${cdn}/probe.png`);
    const latency = performance.now() - start;
    // Report latency back → inform future routing decisions
}
```

**Conclusion:** Geolocation is fast but imprecise; latency-based is more accurate but requires measurement infrastructure. Best systems combine both with fallback logic.

---

### Q20. How do you design a CDN cost optimization strategy that reduces spend by 50% without degrading performance?

**Answer:**

CDN is often one of the top 3 infrastructure costs for media-heavy applications. A systematic approach can achieve 40-60% cost reduction.

**Baseline analysis (find where money goes):**
```sql
-- Analyze CDN access logs
SELECT
    content_type,
    url_pattern,
    COUNT(*) as requests,
    SUM(bytes_sent) as total_bytes,
    AVG(cache_hit) as hit_ratio,
    SUM(bytes_sent) * 0.08 / 1e9 as estimated_cost_usd  -- $0.08/GB
FROM cdn_logs
GROUP BY content_type, url_pattern
ORDER BY total_bytes DESC
LIMIT 50;
```

**Optimization levers:**

**1. Image format modernization (25-40% savings):**
```python
# Serve WebP/AVIF instead of JPEG/PNG
# AVIF: 50% smaller than JPEG at same quality
# WebP: 30% smaller than JPEG

# Cloudflare Polish: automatic image optimization
# Fastly Image Optimizer: on-the-fly conversion

# Implementation:
def get_optimal_format(accept_header):
    if 'image/avif' in accept_header: return 'avif'
    if 'image/webp' in accept_header: return 'webp'
    return 'jpeg'
```

**2. Aggressive TTL tuning for cacheable content (10-20% savings):**
```python
# Audit content that changes infrequently but has short TTLs
content_analysis = {
    "product_images": {"current_ttl": 3600, "change_frequency": "monthly",
                       "recommendation": 2592000},  # 30 days
    "css_bundles": {"current_ttl": 3600, "change_frequency": "per_deploy",
                    "recommendation": "immutable + versioned URL"},
    "api_responses": {"current_ttl": 0, "change_frequency": "per_request",
                      "recommendation": "add s-maxage=60 where safe"}
}
```

**3. Cache key normalization (5-15% savings):**
```
Problem: Same image served as:
  /image.jpg?utm_source=email
  /image.jpg?utm_source=social
  /image.jpg?fbclid=12345
  
CDN treats these as 3 different cache entries → 0% hit rate!

Fix: Strip tracking parameters from cache key
Cloudflare Page Rule: Strip query strings for /images/* paths
```

**4. Lazy loading + right-sizing images (10-20% savings):**
```html
<!-- Only load what's visible -->
<img loading="lazy" src="/product-640.webp"
     srcset="/product-320.webp 320w, /product-640.webp 640w, /product-1280.webp 1280w"
     sizes="(max-width: 640px) 320px, (max-width: 1280px) 640px, 1280px">
```

**5. Multi-CDN arbitrage (5-10% savings):**
```
Shift high-volume traffic to cheapest CDN for that traffic type
Video streaming: negotiate volume discount with Cloudflare
API traffic: Fastly (better caching granularity)
Images: Bunny CDN (significantly cheaper than Cloudflare/Fastly)
```

**Implementation roadmap:**
```
Week 1-2: Image format conversion → immediate 25-30% savings
Week 3-4: Cache key normalization → immediate 5-15% savings  
Month 2: TTL audit and extension → 10-20% savings
Month 3: Multi-CDN negotiation → 5-10% savings
Total: 45-75% potential savings
```

---

## Quick Reference

### Cache-Control Cheat Sheet
| Header | Browser | CDN | Meaning |
|---|---|---|---|
| `max-age=N` | Cache N sec | Cache N sec | Both cache |
| `s-maxage=N` | Uses max-age | Cache N sec | CDN override |
| `no-cache` | Must revalidate | Must revalidate | Cache but check freshness |
| `no-store` | Don't cache | Don't cache | Never cache |
| `private` | Cache | Don't cache | Browser only |
| `immutable` | Cache forever | Cache forever | Content won't change |

### CDN Decision Tree
```
Is content personalized? → Yes → Cache-Control: private
Is content real-time? → Yes → Cache-Control: no-store
Is content static with hash in URL? → Yes → max-age=31536000, immutable
Is content semi-dynamic? → Yes → s-maxage=3600, max-age=60
```

### Cache Invalidation Strategies
1. TTL — simple, eventual consistency
2. Purge API — immediate, surgical
3. Versioned URLs — zero invalidation needed (best for static assets)
4. Cache tags — purge groups of related content

### HLS/DASH Summary
```
Video → Encode → N bitrate variants → Split into 2-6s segments
Player downloads manifest → picks quality based on bandwidth
Adapts quality every segment boundary
```

### CDN Metrics
```
Cache Hit Ratio (CHR) = Hits / (Hits + Misses) × 100%
Good: > 90% for static, > 60% for semi-dynamic
Origin Offload = same as CHR
```

### Edge Computing vs CDN
- CDN = cache files
- Edge Computing = run code at CDN node (Lambda@Edge, CF Workers)
- Use edge for: A/B tests, auth, image resizing, bot detection

### Multi-CDN When To Use
- CDN outage risk is unacceptable
- CDN spend > $100K/month
- Global reach with uneven coverage across providers

### Cost Optimization Priority
1. Image format (WebP/AVIF) — biggest impact
2. Cache key normalization — eliminate duplicate cache entries
3. TTL extension for slow-changing content
4. Lazy loading — don't load what isn't seen
