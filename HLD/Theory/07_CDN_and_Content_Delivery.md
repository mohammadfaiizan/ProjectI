# CDN and Content Delivery

## Table of Contents
1. [CDN Fundamentals](#cdn-fundamentals)
2. [Push CDN vs Pull CDN](#push-cdn-vs-pull-cdn)
3. [Cache-Control Headers](#cache-control-headers)
4. [CDN Cache Invalidation](#cdn-cache-invalidation)
5. [Static vs Dynamic Content Caching](#static-vs-dynamic-content-caching)
6. [CDN for Video Streaming](#cdn-for-video-streaming)
7. [CDN for API Acceleration](#cdn-for-api-acceleration)
8. [Edge Computing](#edge-computing)
9. [CDN Security](#cdn-security)
10. [Origin Shield / Mid-Tier Caching](#origin-shield--mid-tier-caching)
11. [Multi-CDN Strategy](#multi-cdn-strategy)
12. [CDN Providers Comparison](#cdn-providers-comparison)
13. [Measuring CDN Performance](#measuring-cdn-performance)
14. [When NOT to Use CDN](#when-not-to-use-cdn)
15. [Geographic Routing](#geographic-routing)
16. [Quick Reference](#quick-reference)

---

## CDN Fundamentals

### What Is a CDN?

A Content Delivery Network (CDN) is a geographically distributed network of servers (called Points of Presence, or PoPs) that cache and serve content to users from the location closest to them.

Without a CDN:
```
User in Tokyo -> Request -> Origin server in New York (~150ms RTT)
```

With a CDN:
```
User in Tokyo -> Request -> CDN PoP in Tokyo (~5ms RTT)
                            CDN serves cached content
                            CDN fetches from origin only on cache miss
```

### How It Works — Request Flow

```
1. User types example.com in browser
2. DNS resolves example.com to CDN edge IP (e.g., via CNAME to cdn-provider.net)
3. CDN routes DNS response to nearest PoP
4. Browser connects to CDN PoP
5. CDN checks its cache for the requested URL:
   HIT:  CDN returns cached response directly
   MISS: CDN fetches from origin, caches response, returns to user
```

### Points of Presence (PoPs)

A PoP is a CDN data center at a specific geographic location. Major CDNs have 100–400+ PoPs worldwide.

```
CDN PoP structure (Cloudflare example):
  - Metal servers with large SSD caches
  - Anycast routing: multiple PoPs share the same IP block
  - Peer with local ISPs for low latency
  - Redundant network connectivity
```

**Tier 1 PoPs (major cities):** Full feature set, large cache, direct ISP peering.
**Tier 2 PoPs (smaller cities):** Smaller cache, peer with Tier 1 PoPs on cache miss.

### CDN DNS Integration

Two mechanisms for steering users to the nearest CDN PoP:

**CNAME-based:**
```
example.com CNAME -> example.cdn-provider.net (CDN handles geo-routing)
```

**Anycast:**
```
CDN advertises the same IP block from all PoPs.
BGP routing directs users to the nearest PoP automatically.
Cloudflare, AWS CloudFront use Anycast.
```

### Core Benefits

| Benefit | Mechanism | Impact |
|---|---|---|
| Reduced latency | Serve from PoP near user | 50–200ms reduction per request |
| Reduced origin load | Cache hits never reach origin | 80–95% origin traffic reduction |
| DDoS absorption | CDN's capacity far exceeds any origin | Attacks absorbed at edge |
| Improved availability | Origin can go down; cached content still served | Near-100% availability for static assets |
| Cost reduction | Less bandwidth from origin (expensive) | Significant cost savings at scale |

---

## Push CDN vs Pull CDN

### Pull CDN (Most Common)

The CDN pulls content from the origin on the first request for each piece of content. Subsequent requests are served from cache.

```
First request (cache miss):
User -> CDN -> [MISS] -> Origin -> CDN caches -> User

Subsequent requests (cache hit):
User -> CDN -> [HIT] -> User (origin not contacted)
```

**Configuration:** Point the CDN to your origin URL. No upload step required.

```
CDN origin configuration:
  Origin: https://api.example.com
  Cache behavior: cache based on Cache-Control headers
```

**Pros:**
- No upfront upload step — content is cached on demand.
- Only popular content gets cached (no wasted storage for rarely-accessed items).
- Simple to configure.

**Cons:**
- First user after cache miss gets high latency (origin fetch).
- Cache must warm up before full performance is achieved.
- If origin goes down, uncached content is unavailable.

**Best for:** Most websites, APIs, media platforms where content is read-heavy and access patterns are unpredictable.

### Push CDN

You proactively upload (push) content to CDN edge servers before users request it.

```
Deployment pipeline:
Build -> Upload assets to CDN storage -> CDN distributes to all PoPs

User request:
User -> CDN -> [Always HIT, content pre-loaded] -> User
```

**Configuration:** CDN provides an API or CLI to upload files. You control which files are pushed and when.

```bash
# Example: push to Cloudflare R2 + CDN
aws s3 sync ./dist/ s3://my-cdn-bucket/ --cache-control "max-age=31536000"
# Or via CDN API:
curl -X POST https://api.cdn.example.com/upload \
  -F "file=@./dist/app.v2.js" \
  -H "Authorization: Bearer $CDN_TOKEN"
```

**Pros:**
- Zero cache misses — content is always ready on all PoPs.
- Origin can be turned off after upload (for static sites).
- Predictable performance from the first request.

**Cons:**
- Must re-upload on every content change.
- Storage costs on CDN for all pushed content (not just popular items).
- More complex deployment pipeline.

**Best for:** Static site generators, software distribution, known-popular content (major release assets, scheduled campaign assets).

### Push vs Pull Decision Matrix

| Factor | Push CDN | Pull CDN |
|---|---|---|
| Content type | Static, pre-built | Dynamic or unpredictable |
| Content volume | Small to medium | Any size |
| Update frequency | Infrequent | Frequent |
| First-request latency | Zero (always cached) | High (origin fetch) |
| Deployment complexity | Higher | Lower |
| Origin dependency | None after push | Required on miss |
| Storage cost | Pay for all content | Pay for cached hot content |
| Typical use | Static sites, software releases | Web apps, media, APIs |

---

## Cache-Control Headers

Cache-Control is an HTTP header that instructs browsers and CDN edge servers on how to cache a response.

### Key Directives

**`max-age=N`** — Response can be cached for N seconds (browser and CDN).

```
Cache-Control: max-age=86400  (cache for 1 day)
```

**`s-maxage=N`** — Like max-age, but applies to shared caches (CDNs) only. Overrides max-age for CDNs.

```
Cache-Control: max-age=60, s-maxage=86400
# Browser: cache for 60 seconds (user sees fresh data relatively often)
# CDN: cache for 1 day (CDN stays warm, protects origin)
```

**`no-cache`** — Do NOT serve from cache without revalidating with origin first. The response IS stored in cache, but must be validated before use.

```
Cache-Control: no-cache
# CDN will check with origin: "Is this still valid?" (ETag or Last-Modified)
# If valid: serve cached version (fast, no body transfer)
# If stale: fetch new version
```

**`no-store`** — Do not cache at all. Response must never be stored.

```
Cache-Control: no-store
# Use for: sensitive data (banking, medical), personalized responses
# Every request goes to origin
```

**`must-revalidate`** — Once the cached item expires, it MUST be revalidated. Do not serve stale content even if origin is unavailable.

```
Cache-Control: max-age=3600, must-revalidate
```

**`stale-while-revalidate=N`** — Serve stale content for up to N seconds while asynchronously fetching fresh content.

```
Cache-Control: max-age=60, stale-while-revalidate=600
# After 60s: serve stale AND refresh in background (user sees no latency)
# After 660s: must revalidate before serving
```

**`stale-if-error=N`** — Serve stale content for up to N seconds if origin returns an error.

```
Cache-Control: max-age=3600, stale-if-error=86400
# If origin is down: serve up to 24-hour-old cached content rather than error
# Great for resilience
```

**`private`** — Only the user's browser can cache this response. CDN must NOT cache it.

```
Cache-Control: private, max-age=3600
# Use for: authenticated pages, user-specific content
```

**`public`** — Explicitly marks response as cacheable by shared caches (CDNs), even if it would normally not be (e.g., responses to authenticated requests).

```
Cache-Control: public, max-age=3600
```

### ETag and Conditional Requests

ETags enable efficient revalidation.

```
Origin response:
HTTP/1.1 200 OK
ETag: "v2-abc123def456"
Cache-Control: max-age=60

After 60 seconds, CDN revalidates:
GET /resource
If-None-Match: "v2-abc123def456"

Origin (unchanged):
HTTP/1.1 304 Not Modified
(no body -- CDN serves cached body, saves bandwidth)

Origin (changed):
HTTP/1.1 200 OK
ETag: "v3-xyz789"
(new content)
```

### Complete Cache-Control Strategy by Content Type

```
# Static assets with content hash in filename (e.g., app.a3f9b2c.js)
Cache-Control: public, max-age=31536000, immutable
# 1 year, immutable = no revalidation needed (hash guarantees freshness)

# HTML pages (frequently updated, but serve stale during revalidation)
Cache-Control: public, max-age=60, stale-while-revalidate=3600

# API responses (vary by content, short-lived)
Cache-Control: public, s-maxage=300, max-age=60, stale-while-revalidate=60

# User-specific API responses
Cache-Control: private, max-age=60

# Highly sensitive data (passwords, payment info)
Cache-Control: no-store, no-cache

# Media files (images, videos — rarely change)
Cache-Control: public, max-age=2592000  (30 days)
```

---

## CDN Cache Invalidation

### TTL Expiry (Default)

Content automatically expires after `max-age` or `s-maxage`. The simplest approach but means stale content is served until expiry.

**Best practice:** Use very long TTLs (1 year) for versioned static assets and short TTLs (minutes) for content that may change.

### Versioned URLs (Recommended for Static Assets)

Embed a content hash or version number in the URL. When content changes, the URL changes, so the old cached version is naturally abandoned.

```html
<!-- Old version -->
<script src="/js/app.a1b2c3d4.js"></script>
<!-- After build: new content hash -->
<script src="/js/app.e5f6g7h8.js"></script>
```

**Pros:** Infinite cache TTL on static assets. No invalidation API calls needed. Works perfectly across all CDN providers.
**Cons:** Requires build tool integration (Webpack, Vite, etc.).

### Explicit Cache Purge (Invalidation API)

CDN providers expose an API to immediately purge specific URLs or patterns.

```bash
# Cloudflare purge specific URL
curl -X POST https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache \
  -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"files": ["https://example.com/api/products/123"]}'

# Cloudflare purge by tag (requires Enterprise)
curl -X POST .../purge_cache \
  -d '{"tags": ["product-123", "category-electronics"]}'

# AWS CloudFront invalidation
aws cloudfront create-invalidation \
  --distribution-id E1234567890 \
  --paths "/api/products/*" "/images/product-123.jpg"
```

**Propagation time:** Purge typically propagates to all PoPs within 1–5 seconds (Cloudflare), or up to 15 minutes (older CDNs like CloudFront in some cases).

**Cost:** CloudFront charges for invalidations beyond 1000/month. Cloudflare includes purges in all plans.

### Cache Tags / Surrogate Keys

Tag responses at the origin, then purge all content with a specific tag.

```
# Origin response header
Surrogate-Key: product-123 category-electronics homepage-featured
Cache-Tag: product-123  (CloudFront terminology)

# Purge all content tagged with "product-123"
# Instantly invalidates all CDN edges: product page, category listing, homepage, search results
```

This is the most powerful invalidation pattern. When a product is updated, you can purge all pages that contain it with a single API call.

**CDN support:** Fastly (Surrogate-Key), Cloudflare Enterprise (Cache-Tag), Varnish (xkey).

---

## Static vs Dynamic Content Caching

### Static Content

Files that are the same for all users and do not change frequently:

- HTML, CSS, JavaScript bundles
- Images, videos, fonts
- PDF files, downloadable assets

**Caching approach:**
```
Cache-Control: public, max-age=31536000, immutable
Vary:  (none — same for all users)
```

CDN caches indefinitely until TTL or explicit purge. Very high hit ratios (95%+).

### Dynamic Content

Content that varies by user, time, query parameters, or request context:

- API responses with user-specific data
- Personalized recommendation results
- Real-time inventory/pricing
- Session-dependent HTML

**Caching approach (where applicable):**
```
Cache-Control: private, no-store  (for user-specific)

OR, for semi-dynamic public content:
Cache-Control: public, s-maxage=60, stale-while-revalidate=300
Vary: Accept-Language, Accept-Encoding  (cache separate versions per lang/encoding)
```

### The `Vary` Header

Instructs CDN to cache separate versions of a response based on request headers.

```
Response: Vary: Accept-Language

CDN stores:
  GET /homepage + Accept-Language: en -> cache key A
  GET /homepage + Accept-Language: fr -> cache key B
  GET /homepage + Accept-Language: de -> cache key C
```

**Warning:** Overuse of Vary fragments the cache, reducing hit ratios. Only vary on headers that genuinely produce different content.

Common Vary headers:
- `Vary: Accept-Encoding` — gzip vs brotli (almost always appropriate)
- `Vary: Accept-Language` — multi-language content
- `Vary: Authorization` — DO NOT vary on Authorization (fragments cache per user, defeats purpose)

---

## CDN for Video Streaming

### The Challenge

Video files are large (100MB to 10GB+). Streaming them efficiently requires:
1. Serving bytes starting from any position (range requests).
2. Adapting to the viewer's bandwidth (adaptive bitrate).
3. Distributing globally without overloading origin.

### Segmented Delivery

Modern video streaming protocols break video into small segments (2–10 seconds each).

**HLS (HTTP Live Streaming — Apple):**
```
master.m3u8 (playlist file)
  -> 360p/index.m3u8 -> 360p/seg001.ts, 360p/seg002.ts ...
  -> 720p/index.m3u8 -> 720p/seg001.ts, 720p/seg002.ts ...
  -> 1080p/index.m3u8 -> 1080p/seg001.ts, 1080p/seg002.ts ...
```

**DASH (Dynamic Adaptive Streaming over HTTP — standard):**
```
manifest.mpd (Media Presentation Description)
  -> video/1080p/segment_1.m4s, segment_2.m4s ...
  -> video/720p/segment_1.m4s ...
  -> audio/en/segment_1.m4s ...
```

**How CDN helps:**
- Manifest files (.m3u8, .mpd) are small and cached at the edge.
- Video segments are independent files; each segment is independently cacheable.
- CDN serves segments from the nearest PoP; player requests next segment while playing current.

### Adaptive Bitrate (ABR)

The player monitors download speed and buffer level, then selects the appropriate quality tier.

```
Buffer healthy + fast network -> request 1080p segment
Buffer low + slow network     -> switch to 360p segment
```

CDN ensures all quality tiers are cached at the edge. Quality switches happen within the same CDN PoP.

### HTTP Range Requests

For large video files without segmentation, HTTP range requests allow clients to request specific byte ranges:

```
GET /video.mp4 HTTP/1.1
Range: bytes=10000000-20000000

HTTP/1.1 206 Partial Content
Content-Range: bytes 10000000-20000000/500000000
```

CDN must support range requests and can cache segments independently.

---

## CDN for API Acceleration

### Edge Caching of API Responses

Public API responses (no user-specific data) can be cached at the CDN edge.

```python
@app.get("/api/v1/products")
def list_products():
    products = fetch_from_db()
    response = make_response(json.dumps(products))
    response.headers["Cache-Control"] = "public, s-maxage=300, stale-while-revalidate=60"
    return response
```

**Cache key design for APIs:**
```
Default cache key: URL + selected request headers
Customize: add query params to cache key

CDN config (Cloudflare):
  Cache key: /api/products?sort=price&category=electronics
  (sort and category are cache-key components)
```

### API Gateway + CDN

```
Mobile App
    |
    v
[CDN Edge] -> cache public API responses
    |
    v
[API Gateway] -> auth, rate limit, routing
    |
    v
[Microservices]
```

CDN reduces load on API gateway for cacheable endpoints (product listings, catalog, static config).

---

## Edge Computing

### What Is Edge Computing?

Running code at CDN PoPs (the "edge") rather than at a central origin. Reduces latency by executing logic close to the user.

### Lambda@Edge (AWS CloudFront)

Executes Node.js or Python functions at CloudFront edge locations.

**Trigger points:**

```
Viewer Request  -> (function) -> CloudFront Cache Check
                                      |
                                 MISS: Origin Request -> (function) -> Origin
                                  HIT: <--
                                      |
                                 Origin Response -> (function) -> Viewer Response -> (function) -> User
```

**Use cases:**

```javascript
// A/B testing at edge (viewer request)
exports.handler = async (event) => {
    const request = event.Records[0].cf.request;
    const bucket = Math.random() < 0.5 ? 'control' : 'treatment';
    request.uri = `/${bucket}${request.uri}`;
    return request;
};

// Auth token validation at edge (no round trip to origin)
exports.handler = async (event) => {
    const request = event.Records[0].cf.request;
    const token = request.headers['authorization']?.[0]?.value;
    
    if (!isValidToken(token)) {
        return { status: '401', body: 'Unauthorized' };
    }
    return request;
};

// Geographic redirect
exports.handler = async (event) => {
    const request = event.Records[0].cf.request;
    const country = request.headers['cloudfront-viewer-country']?.[0]?.value;
    
    if (country === 'DE') {
        return {
            status: '302',
            headers: { location: [{ value: 'https://de.example.com' + request.uri }] }
        };
    }
    return request;
};
```

**Limitations:** Max 5 seconds execution, 1MB code size, no disk access, no outbound network (except origin).

### Cloudflare Workers

More capable than Lambda@Edge. Full V8 JavaScript runtime at the edge.

```javascript
addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
    // Access KV storage (edge key-value store)
    const cached = await CACHE_KV.get(request.url);
    if (cached) return new Response(cached);
    
    // Modify request before forwarding
    const modifiedRequest = new Request(request, {
        headers: { ...Object.fromEntries(request.headers), 'X-Custom': 'value' }
    });
    
    const response = await fetch(modifiedRequest);
    
    // Cache in KV
    await CACHE_KV.put(request.url, await response.text(), { expirationTtl: 300 });
    
    return response;
}
```

**Cloudflare Workers KV:** Globally replicated key-value store accessible from edge workers. Eventually consistent.

**Cloudflare Durable Objects:** Strongly consistent objects, useful for real-time collaboration, game state, chat rooms.

### Edge Computing Use Cases

| Use Case | Benefit |
|---|---|
| Authentication / JWT validation | No round trip to auth service |
| A/B testing | Instant variant selection at edge |
| Geolocation-based redirects | No origin contact needed |
| Request/response transformation | Modify headers, rewrite URLs |
| Personalization hints | Add context headers for origin |
| Bot detection | Block before reaching origin |
| Edge rendering (ESI) | Compose page fragments at edge |

---

## CDN Security

### DDoS Protection

CDN providers have massive network capacity (Cloudflare: 321 Tbps+, Akamai: 400+ Tbps) that can absorb volumetric attacks.

**How it works:**
1. Attack traffic enters CDN's Anycast network.
2. Attack is distributed across CDN's global capacity.
3. CDN's scrubbing filters identify and drop attack traffic.
4. Legitimate traffic continues to origin (or is served from cache).

### WAF at Edge (Web Application Firewall)

WAF rules inspect HTTP requests and block malicious patterns before they reach the origin.

```
Cloudflare WAF rules:
  - OWASP Core Rule Set (SQLi, XSS, RCE detection)
  - Custom rules (rate limiting, IP blocking, user-agent filtering)
  - Bot management (challenge automated traffic)
  - API protection (schema validation)

AWS WAF rule groups:
  - AWS Managed Rules (free with WAF)
  - AWS Marketplace managed rules (third-party)
  - Custom rules
```

**Example Cloudflare WAF rule:**
```
(http.request.uri.query contains "UNION SELECT") or
(http.request.uri.query contains "' OR '1'='1") or
(http.request.body contains "<script>")
-> Block
```

### SSL Offload

CDN terminates SSL/TLS at the edge, reducing SSL processing load on origin servers.

```
User --[HTTPS/TLS 1.3]--> CDN Edge
CDN Edge --[HTTP or HTTPS]--> Origin

Benefits:
- TLS 1.3 negotiated at edge (closer to user, faster handshake)
- Origin only needs to handle HTTP (or keep HTTPS for compliance)
- Certificate management centralized at CDN
```

**Full SSL (end-to-end):** CDN re-encrypts to origin. Satisfies compliance requirements.
**Flexible SSL:** CDN to origin over HTTP. Not recommended for sensitive data.
**Strict SSL:** CDN verifies origin certificate. Prevents man-in-the-middle between CDN and origin.

### Hot-Linking Prevention

Prevent other websites from embedding your media (bandwidth theft).

```
# Nginx: allow only your domain
valid_referers none blocked example.com *.example.com;
if ($invalid_referer) {
    return 403;
}
```

CDN-based hot-linking protection via signed URLs or referer rules:

```python
# Generate signed CloudFront URL
import boto3
from botocore.signers import CloudFrontSigner
import rsa
from datetime import datetime, timedelta

def generate_signed_url(url, expiry_minutes=60):
    key_id = 'K1234567890ABC'
    private_key = open('private_key.pem', 'rb').read()
    
    signer = CloudFrontSigner(key_id, lambda msg: rsa.sign(msg, rsa.PrivateKey.load_pkcs1(private_key), 'SHA-1'))
    
    expiry = datetime.utcnow() + timedelta(minutes=expiry_minutes)
    return signer.generate_presigned_url(url, date_less_than=expiry)
```

---

## Origin Shield / Mid-Tier Caching

### The Problem

With hundreds of CDN PoPs worldwide, a cache miss on every PoP independently hits the origin, potentially generating hundreds of simultaneous origin requests for the same content.

```
Without origin shield:
PoP in Tokyo (miss) -> Origin
PoP in Sydney (miss) -> Origin
PoP in Singapore (miss) -> Origin
= 3 concurrent origin requests for the same content
```

### Origin Shield Solution

A single designated PoP (the "shield" or "parent" node) sits between all edge PoPs and the origin. Cache misses from edge PoPs go to the shield, not directly to origin.

```
With origin shield:
PoP in Tokyo (miss) -> Shield PoP in Japan
PoP in Sydney (miss) -> Shield PoP in Japan  (already fetching or cached)
PoP in Singapore (miss) -> Shield PoP in Japan  (cache hit now)
= 1 origin request total
```

**Benefits:**
- Dramatically reduces origin load (10–50x reduction).
- Protects origin from thundering herd after purge (many PoPs simultaneously miss).
- Origin shield is geographically close to origin (low latency for that hop).

**CDN implementation:**
- AWS CloudFront: Origin Shield (select a region closest to origin).
- Cloudflare: Tiered Cache (automatically creates shield topology).
- Fastly: Shielding (designate a POP as the shield).

---

## Multi-CDN Strategy

### Why Multi-CDN?

Relying on a single CDN creates a single point of failure. CDN outages are rare but impactful.

```
Single CDN failure scenario:
  CDN provider has global outage
  -> All your traffic origin-falls-back
  -> Origin overwhelmed
  -> Site goes down
```

### Multi-CDN Implementation

**Approach 1: DNS-based routing**

```
GeoDNS or traffic manager routes different geographic regions to different CDNs:
  US users     -> Cloudflare
  EU users     -> Akamai
  APAC users   -> AWS CloudFront
```

**Approach 2: Active-passive failover**

```
Primary: Cloudflare
If Cloudflare health check fails -> DNS switches to Fastly (secondary)
Switchover time: 30–60 seconds (limited by DNS TTL)
```

**Approach 3: Weighted multi-CDN**

```
Traffic split: 70% Cloudflare, 30% Fastly
Benefits: continuous comparison of CDN performance, gradual migration
```

**Approach 4: CDN with edge routing layer**

```
User -> [NS1 / AWS Route 53 / Cedexis] -> [Best CDN based on real-time health + performance]
```

Tools: NS1 Pulsar, Cedexis (now Citrix), Catchpoint.

### Multi-CDN Considerations

- **Cache consistency:** Each CDN has its own cache. Same URL may serve different versions from different CDNs.
- **Purge complexity:** Must call purge APIs for all CDNs simultaneously.
- **Cost:** Usually paying for multiple CDN vendors.
- **Origin shield:** Each CDN needs its own origin shield configuration.

---

## CDN Providers Comparison

### Cloudflare

- **Network:** 321+ Tbps capacity, 330+ PoPs, Anycast everywhere.
- **Strengths:** Best-in-class DDoS protection, free WAF on all plans, Workers for edge compute, excellent free tier.
- **Pricing:** Free tier generous, Pro $20/month, Business $200/month, Enterprise custom.
- **Unique features:** Argo Smart Routing (optimized paths), Magic Transit (network-level protection), Zaraz (third-party script optimization).
- **Best for:** Most web applications, DDoS protection, edge computing.

### Akamai

- **Network:** 4000+ PoPs, oldest and largest CDN network.
- **Strengths:** Deep enterprise feature set, best coverage in emerging markets, strong media delivery.
- **Pricing:** Enterprise contracts only, expensive.
- **Unique features:** Edge Side Includes (ESI), Adaptive Media Delivery, Ion (web performance).
- **Best for:** Large enterprises, media companies, financial services requiring SLA guarantees.

### AWS CloudFront

- **Network:** 600+ PoPs across 90+ cities.
- **Strengths:** Native AWS integration (S3, ALB, Lambda@Edge, API Gateway), pay-as-you-go.
- **Pricing:** $0.0085–$0.17/GB depending on region, $0.01/10,000 HTTPS requests.
- **Unique features:** Lambda@Edge, Origin Shield, CloudFront Functions, Origin Access Control for S3.
- **Best for:** AWS-native applications, S3-based static sites, tight AWS ecosystem integration.

### Fastly

- **Network:** 60+ PoPs, but strategically placed at major internet exchanges.
- **Strengths:** Varnish-based VCL for powerful cache customization, instant purge (<150ms globally), real-time logs.
- **Pricing:** Pay-as-you-go, $0.12/GB in US, higher elsewhere.
- **Unique features:** Compute@Edge (Rust/WASM), instant global purge, real-time log streaming.
- **Best for:** Developers who need fine-grained cache control, media companies, instant invalidation requirements.

### CDN Comparison Table

| Feature | Cloudflare | Akamai | AWS CloudFront | Fastly |
|---|---|---|---|---|
| PoP count | 330+ | 4000+ | 600+ | 60+ |
| Edge compute | Workers (JS) | EdgeWorkers (JS) | Lambda@Edge (Node/Python) | Compute@Edge (Rust/WASM) |
| Free tier | Yes (generous) | No | Limited | No |
| Purge speed | ~1s | ~5s | ~15min (invalidation) | <150ms |
| DDoS protection | Industry-leading | Excellent | Good (Shield Advanced) | Good |
| Real-time logs | Yes | Yes | S3/Kinesis (delayed) | Yes (streaming) |
| Pricing model | Plan-based | Enterprise contract | Pay-as-you-go | Pay-as-you-go |
| Best pricing at scale | Business/Enterprise | Negotiated | Mid-range | High |

---

## Measuring CDN Performance

### Cache Hit Ratio

```
Cache Hit Ratio = CDN cache hits / (CDN cache hits + CDN cache misses)

Target: > 90% for static assets, > 70% for semi-dynamic content
```

**How to improve cache hit ratio:**
- Increase TTL for content that rarely changes.
- Normalize URLs (remove unnecessary query parameters from cache key).
- Use cache tags to avoid over-invalidating.
- Enable origin shield to avoid per-PoP cold starts.

### Origin Offload Ratio

```
Origin Offload = 1 - (origin requests / total CDN requests)

Example: 1M requests to CDN, 50K reach origin
Offload = 1 - (50K / 1M) = 95%
```

Higher origin offload means lower infrastructure costs and better origin resilience.

### P95/P99 Latency

Measure the 95th and 99th percentile of response times for CDN-served content.

```
Target (static assets from CDN):
  P50: < 10ms
  P95: < 50ms
  P99: < 100ms

If P95 is high: investigate PoP coverage, check for cache misses at those locations
```

### TTFB (Time to First Byte)

The time from the client sending a request to receiving the first byte of the response.

```
TTFB = Network latency + CDN processing time + (if miss: origin latency)

Cache HIT TTFB: ~5–30ms (pure network + CDN overhead)
Cache MISS TTFB: ~100–500ms (includes origin processing)
```

### Real User Monitoring (RUM)

Embed a small JavaScript snippet that measures actual user experience metrics:

```javascript
// Web Vitals measurement
import { getCLS, getFID, getLCP } from 'web-vitals';

getCLS(metric => sendToAnalytics('CLS', metric.value));
getFID(metric => sendToAnalytics('FID', metric.value));
getLCP(metric => sendToAnalytics('LCP', metric.value));
```

CDN providers with RUM: Cloudflare Browser Insights, Akamai mPulse, Fastly Insights.

---

## When NOT to Use CDN

### Highly Personalized Content

If every response is unique per user, CDN provides no benefit (0% hit ratio) and adds latency.

```
Personalized recommendation feed: different for every user
Financial portfolio data: unique per account
Private messages: never cached

Solution: Serve directly from origin. Optimize origin with database caching, read replicas.
```

### Real-Time Data

Data that must be fresh to the second cannot tolerate CDN TTL.

```
Live stock prices, sports scores, auction bids
-> CDN cache would serve stale prices -> user makes wrong decisions

Solution: WebSocket or SSE directly to origin, or via a real-time CDN feature 
(Cloudflare Durable Objects, Fastly real-time messaging)
```

### Authenticated Private Content

Content that differs based on authentication should not be cached by CDN without careful design.

```
Wrong: Cache API response that includes user's private data
-> User A gets User B's data (cache poisoning)

Exception: If you include authentication in cache key (defeats most CDN benefits)
Better: Split API: public endpoint (CDN-cacheable) + private endpoint (no CDN)
```

### Write-Heavy API Endpoints

CDN only helps reads. POST/PUT/DELETE endpoints with no response body don't benefit.

```
POST /api/orders -> CDN passes through, no caching value
-> Don't route write endpoints through CDN (or configure CDN to pass through)
```

### Content Behind VPN / Internal Networks

Corporate intranets and internal tools behind VPNs typically have users in specific locations. A CDN PoP adds a hop without geographic benefit.

---

## Geographic Routing

### Latency-Based Routing

Route users to the region with the lowest measured round-trip latency.

```
AWS Route 53 Latency-Based Routing:
  - Route 53 measures latency from known AWS endpoints
  - Routes user to the region with lowest current latency
  - Not purely geographic — accounts for network conditions
```

**Best for:** Applications where users care about response time and you have multi-region deployments.

### Geolocation-Based Routing

Route users based on their detected geographic location (continent, country, state/region).

```
Route 53 Geolocation:
  Users in Germany    -> eu-west-1
  Users in Japan      -> ap-northeast-1
  Users in US         -> us-east-1
  All others (default) -> us-east-1

Use cases:
  - Legal compliance (GDPR: EU users must be served from EU)
  - Language/locale routing
  - Content licensing (stream only in licensed regions)
```

**Geolocation vs Latency-Based:**
- Geolocation is deterministic but may not give the lowest latency.
- Latency-based adapts to network conditions but is less predictable.

### CDN Geographic Routing

CDN handles geographic routing automatically — Anycast routing or GeoDNS directs users to the nearest PoP. You configure:

```
CloudFront: select which regions to deploy to (or all regions)
Cloudflare: serves from nearest PoP automatically (Anycast)
Akamai: configure regional routing rules in portal
```

**Geo-blocking:** Block access from specific countries/regions:

```
# Cloudflare WAF rule
(ip.geoip.country in {"RU" "CN" "KP"}) -> Block

# AWS WAF
aws wafv2 put-geo-match-statement --country-codes RU CN KP --action BLOCK
```

---

## Quick Reference

### CDN Header Cheat Sheet

| Header | Example | Effect |
|---|---|---|
| `Cache-Control: max-age=N` | `max-age=86400` | Cache for N seconds (browser + CDN) |
| `Cache-Control: s-maxage=N` | `s-maxage=86400` | CDN caches for N seconds (overrides max-age for CDN) |
| `Cache-Control: no-cache` | - | Must revalidate before serving |
| `Cache-Control: no-store` | - | Never cache |
| `Cache-Control: private` | - | Browser only, CDN must not cache |
| `Cache-Control: public` | - | CDN may cache |
| `Cache-Control: immutable` | - | Never revalidate (use with versioned URLs) |
| `Cache-Control: stale-while-revalidate=N` | `stale-while-revalidate=600` | Serve stale for N seconds while refreshing |
| `Cache-Control: stale-if-error=N` | `stale-if-error=86400` | Serve stale for N seconds on origin error |
| `ETag: "hash"` | `ETag: "abc123"` | Enables conditional GET |
| `Vary: Header-Name` | `Vary: Accept-Language` | Cache separate versions per header value |
| `Surrogate-Key: tag1 tag2` | `Surrogate-Key: product-123` | Tag for targeted purge |
| `CDN-Cache-Control` | `CDN-Cache-Control: max-age=600` | Cloudflare-specific CDN cache control |

### Push vs Pull CDN Decision Matrix

| Scenario | Recommended |
|---|---|
| Static HTML/CSS/JS build artifacts | Push CDN or Pull + versioned URLs |
| User-uploaded images | Pull CDN (from S3 origin) |
| Software downloads (.exe, .dmg) | Push CDN |
| API responses (public, cacheable) | Pull CDN |
| Video files | Pull CDN (range request support) |
| Frequently updated content | Pull CDN (simpler cache management) |
| Pre-generated dataset exports | Push CDN |
| Origin can go offline after deployment | Push CDN |

### CDN Configuration Checklist

```
[ ] Set appropriate Cache-Control headers on all response types
[ ] Use versioned URLs (content hash) for static assets
[ ] Configure origin shield to protect origin
[ ] Enable Gzip/Brotli compression at CDN level
[ ] Set up purge webhook in deployment pipeline
[ ] Enable WAF rules for DDoS and attack protection
[ ] Configure SSL/TLS (prefer full/strict mode)
[ ] Set up RUM or CDN analytics for monitoring
[ ] Configure fallback behavior on CDN outage
[ ] Test from multiple geographic locations
[ ] Verify Vary header usage (don't over-fragment cache)
[ ] Set up cache hit ratio alerting (alert if < 80%)
```
