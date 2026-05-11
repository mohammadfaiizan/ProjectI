# 27 — Cost Optimization in System Design

---

## Easy (Q1–Q7)

---

### Q1. Why is cost a non-functional requirement in system design interviews?

In the context of system design interviews, **non-functional requirements (NFRs)** define the quality attributes of a system: scalability, availability, performance, security — and cost. Cost is frequently omitted by candidates who focus exclusively on making the system work at scale, but interviewers at senior and staff levels expect cost awareness as evidence of engineering maturity.

**Why cost matters:**
- A system that handles 1 million requests/second using 10,000 bare-metal servers is not impressive — it is wasteful. The interesting challenge is doing it efficiently.
- At production scale, over-provisioned infrastructure routinely costs companies millions per year in wasted spend. Netflix, Airbnb, and Lyft have each published case studies of 40–70% cost reductions from optimisation efforts.
- Engineering decisions made during design are the largest cost levers — retrofitting cost efficiency after launch is far harder than designing for it.

**How to address cost in interviews:**

1. **Back-of-envelope estimation** — estimate request volume, storage growth, and compute needs. Translate to approximate dollar cost.
2. **Justify each component** — for every queue, cache, or database replica you add, briefly explain what cost–benefit trade-off it represents.
3. **Identify the dominant cost driver** — is it compute, storage, or data transfer? Focus optimisation effort there.
4. **Propose cost–performance trade-offs explicitly** — "We could run this on dedicated instances at $X/month, or on spot instances at $X/3 with a small availability trade-off."

**Common cost NFRs to mention:**
- "System must run within $Y/month budget."
- "Cost per transaction must not exceed $Z."
- "Infrastructure cost should scale linearly (not quadratically) with user growth."

Treating cost as a design constraint — not an afterthought — is what separates principal-level thinking from junior-level thinking.

---

### Q2. What are the three biggest cost drivers in cloud infrastructure?

Understanding where cloud spend actually goes is the foundation of any cost optimisation strategy. The three dominant cost categories are **compute**, **storage**, and **data transfer**.

**1. Compute (typically 40–60% of cloud bill):**
- EC2 / GCE / Azure VM instances, Lambda invocations, ECS/Fargate tasks.
- Cost drivers: instance type choice, idle capacity (over-provisioning), lack of auto-scaling.
- Optimisation levers: right-sizing, reserved/spot instances, auto-scaling, serverless for variable workloads.

**2. Storage (typically 20–30%):**
- S3, EBS volumes, RDS storage, DynamoDB capacity units, Glacier.
- Cost drivers: storing data indefinitely, using high-performance storage tiers for cold data, no compression/deduplication.
- Optimisation levers: S3 lifecycle policies, Intelligent Tiering, data compression, deduplication, tiered storage.

**3. Data Transfer / Egress (typically 10–20%, but highly variable):**
- Cloud providers charge for data **leaving** their network (egress) but not for data entering (ingress).
- AWS charges: $0.09/GB for first 10 TB/month egress to the internet. At 1 PB/month: $90,000/month in egress alone.
- Cross-AZ traffic: $0.01/GB each way (often overlooked but adds up at scale).
- Optimisation levers: CDN (cache at edge, reduce origin egress), same-region/AZ data placement, compression.

**Illustrative cost breakdown for a hypothetical 10M DAU service:**

| Category | Monthly Cost | % of Total |
|---|---|---|
| EC2 compute (app + DB) | $45,000 | 52% |
| RDS storage + IOPS | $18,000 | 21% |
| S3 storage (media) | $8,000 | 9% |
| CloudFront + egress | $12,000 | 14% |
| Other (Route53, logs) | $3,500 | 4% |
| **Total** | **$86,500** | 100% |

The most common mistake: spending engineering effort optimising a $500/month component when a $45,000/month component is untouched.

---

### Q3. What is right-sizing and how do you use CPU/memory metrics to choose instance type?

**Right-sizing** is the process of matching the provisioned compute resources (CPU, memory, disk I/O) to the actual workload requirements — neither over-provisioning (wasting money) nor under-provisioning (causing performance degradation).

**The right-sizing process:**

1. **Collect metrics** over a representative period (30–90 days minimum):
   - P95 and P99 CPU utilisation (not average — average masks spikes)
   - P95 memory utilisation
   - Network throughput
   - Disk IOPS (for I/O-heavy workloads)

2. **Apply the 60–70% utilisation target:**
   - If P95 CPU is < 20%, the instance is over-provisioned.
   - Target: P95 CPU at 60–70% of available capacity (headroom for spikes).

3. **Match to instance family:**

| Workload | Instance family | Characteristics |
|---|---|---|
| Web servers (balanced) | t3/t3a, m5 | General purpose |
| CPU-intensive (ML, encoding) | c5, c6g | Compute optimised |
| Memory-intensive (Redis, analytics) | r5, r6g | Memory optimised |
| Storage-intensive (Kafka, Cassandra) | i3, i4i | Storage optimised (NVMe) |
| GPU workloads | p3, g4dn | GPU instances |

**Example:**
```
Current: m5.4xlarge (16 vCPU, 64 GB RAM) at $0.768/hr
P95 CPU: 18% → using ~3 vCPUs
P95 Memory: 35% → using ~22 GB

Right-sized: m5.xlarge (4 vCPU, 16 GB RAM) at $0.192/hr
P95 CPU on new instance: ~72% ✓ (within 60-70% target with headroom)
P95 Memory: still tight → try m5.2xlarge (8 vCPU, 32 GB RAM) at $0.384/hr

Savings: $0.768 → $0.384 = 50% cost reduction per instance
At 100 instances: $38,400/month savings
```

**Tools:** AWS Compute Optimizer (uses ML to recommend right-sized instances), CloudWatch metrics, Datadog cloud cost integration, Spot.io.

---

### Q4. Compare reserved, on-demand, and spot instances. When should you use each?

Cloud providers offer multiple **purchasing models** for compute, each with a different cost–commitment–flexibility trade-off.

**On-Demand:**
- Pay by the hour/second, no commitment.
- Most expensive (baseline price).
- Use for: unpredictable workloads, dev/test environments, short-lived jobs.

**Reserved Instances (RIs) / Savings Plans:**
- Commit to a 1-year or 3-year term.
- Discount: 30–60% off on-demand pricing.
- AWS Savings Plans are more flexible than classic RIs — apply to any EC2 instance family.
- Use for: stable, predictable baseline compute (web servers, databases, background workers).

**Spot Instances:**
- Use spare EC2 capacity at 60–90% discount.
- Can be interrupted with 2-minute warning when AWS needs the capacity back.
- Use for: fault-tolerant, stateless, interruptible workloads (batch processing, ML training, CI/CD workers).

**The layered cost strategy:**

```
Load
  │
  │  100% Baseline load ──────▶ Reserved Instances (cheapest per-unit)
  │
  │  Predictable growth ───────▶ On-Demand (no lock-in)
  │
  │  Burst / batch / fault-tolerant ▶ Spot Instances (cheapest, interruptible)
  │
Time
```

**Cost comparison (m5.xlarge, us-east-1, Linux):**

| Purchasing Model | Hourly Rate | Monthly (730 hrs) | vs On-Demand |
|---|---|---|---|
| On-Demand | $0.192 | $140.16 | Baseline |
| 1-yr Reserved (no upfront) | $0.119 | $86.87 | -38% |
| 3-yr Reserved (all upfront) | $0.076 | $55.48 | -60% |
| Spot (typical) | $0.060 | $43.80 | -70% |

**Anti-pattern to avoid:** Reserving instances for workloads that scale up and down. Reserved instances are a fixed commitment — if you reserve 100 instances and scale down to 20 during low-traffic periods, you are paying for 80 idle reserved instances. Use Savings Plans (more flexible than RIs) or auto-scaling with on-demand for variable workloads.

---

### Q5. How does auto-scaling prevent paying for idle capacity?

**Auto-scaling** dynamically adjusts the number of running instances based on real-time demand, eliminating the gap between provisioned capacity and actual utilisation.

**The idle capacity problem:**
```
Without auto-scaling:
  Peak load:    1000 req/s → provision 50 instances
  Nighttime:    50 req/s  → still running 50 instances
  Idle waste:   45 instances running at < 5% utilisation
  Monthly cost: 50 × $140 = $7,000
  Actual need:  3 × $140 = $420 at night (90% waste)
```

**With auto-scaling:**
```
  Peak:    50 instances running (scale-out triggered)
  Night:   3 instances running (scale-in triggered)
  Average: ~12 instances over 24 hours
  Monthly: 12 × $140 = $1,680 (76% reduction)
```

**Types of auto-scaling:**

| Type | Trigger | Response | Best for |
|---|---|---|---|
| Target tracking | Maintain CPU at 60% | Automatic | Most web workloads |
| Step scaling | CloudWatch alarm thresholds | Step-function | Predictable load patterns |
| Scheduled scaling | Time-based | Pre-emptive | Known traffic patterns (office hours) |
| Predictive scaling | ML-based forecast | Proactive | Spiky but predictable workloads |

**Key configuration parameters:**
```yaml
auto_scaling:
  min_capacity: 2          # Never drop below 2 (availability floor)
  max_capacity: 100        # Cost ceiling
  target_cpu_utilization: 60%
  scale_out_cooldown: 60s  # Prevent flapping
  scale_in_cooldown: 300s  # Be conservative on scale-in
  warmup_period: 120s      # Time for new instances to be ready
```

**Critical detail:** Configure scale-in conservatively. Premature scale-in followed by an immediate scale-out is operationally noisy and can cause brief capacity gaps. Always set `scale_in_cooldown` longer than `scale_out_cooldown`.

---

### Q6. Why is data egress expensive and how do you reduce it?

**Egress** refers to data leaving a cloud provider's network. Cloud providers charge for it because it consumes expensive backbone internet capacity that they must procure from upstream providers.

**AWS egress pricing (illustrative):**
- First 1 GB/month: free
- Next 9.999 TB: $0.09/GB
- Next 40 TB: $0.085/GB
- Cross-AZ traffic: $0.01/GB each direction

**Scale example:**
```
Video streaming service: 10 PB/month egress to users
Cost: 10 × 1024 TB × $0.085/GB = $870,400/month in egress alone
```

**Reduction strategies:**

**1. CDN (largest impact):**
- CloudFront/Fastly charges $0.0075–$0.01/GB (compared to $0.085/GB for direct S3 egress).
- Cache hit rate of 90% → 90% of traffic never hits origin → 90% reduction in origin egress.
- Net: CDN costs + reduced origin egress usually 60–75% cheaper than serving direct.

**2. Same-region data placement:**
- Place application servers in the same region as their databases and object storage.
- Cross-region data transfer: $0.02/GB — avoidable with correct architecture.

**3. Same-AZ replicas for high-bandwidth services:**
- A Kafka consumer reading 1 TB/day from a broker in a different AZ costs $0.01/GB × 1024 = $10.24/day = $307/month.
- Deploy consumers in the same AZ as the Kafka broker.

**4. Compression:**
- Compress API responses (gzip, Brotli): 60–80% size reduction for JSON/text.
- Compress before storing in S3: reduces both storage and egress cost.

**5. Avoid cross-region replication for non-critical data:**
- Every byte replicated to another region costs egress. Only replicate what is necessary.

---

### Q7. How do S3 storage tiers and lifecycle policies reduce storage cost?

S3 offers multiple **storage classes** with different cost, availability, and retrieval latency profiles. A lifecycle policy automatically moves objects between tiers based on age or access patterns, eliminating the need for manual management.

**S3 storage classes and approximate costs (us-east-1):**

| Storage Class | Cost/GB/month | Retrieval Cost | Min. Duration | Use case |
|---|---|---|---|---|
| S3 Standard | $0.023 | Free | None | Frequently accessed |
| S3 Standard-IA | $0.0125 | $0.01/GB | 30 days | Infrequent access |
| S3 One Zone-IA | $0.01 | $0.01/GB | 30 days | Infrequent, reproducible |
| S3 Glacier Instant | $0.004 | $0.03/GB | 90 days | Archives, ms retrieval |
| S3 Glacier Flexible | $0.0036 | $0.01/GB + time | 90 days | Archives, min-hr retrieval |
| S3 Glacier Deep Archive | $0.00099 | $0.02/GB | 180 days | Compliance archives |
| S3 Intelligent Tiering | $0.023 + $0.0025 monitoring fee | Free | None | Unknown access patterns |

**Lifecycle policy example:**
```json
{
  "Rules": [{
    "Status": "Enabled",
    "Transitions": [
      {"Days": 30,  "StorageClass": "STANDARD_IA"},
      {"Days": 90,  "StorageClass": "GLACIER"},
      {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
    ],
    "Expiration": {"Days": 2555}
  }]
}
```
This moves objects from Standard → IA after 30 days, to Glacier after 90, Deep Archive after 1 year, and deletes after 7 years.

**Cost impact example:**
```
1 TB of log files created monthly, retained 1 year:
  All Standard:     1000 GB × $0.023 = $23/month cumulative → $276/year
  With lifecycle:
    Month 1 (Standard):        $23.00
    Months 2-3 (Standard-IA):  $12.50 × 2 = $25.00
    Months 4-12 (Glacier):     $3.60 × 9 = $32.40
    Total for 1 TB cohort:     $80.40 vs $276 → 71% reduction
```

**Intelligent Tiering** is the no-thought option: S3 automatically moves objects based on actual access patterns. The $0.0025/GB/month monitoring fee is offset by savings on objects not accessed for 30+ days. Ideal when access patterns are unknown or unpredictable.

---

## Medium (Q8–Q15)

---

### Q8. How do caching and read replicas reduce database cost?

Database costs are driven by **compute** (instance type), **storage** (provisioned IOPS), and **request volume** (DynamoDB RCUs, Aurora I/O). Caching and read replicas address different dimensions of this cost.

**Caching (reduces DB query volume):**
```
Without cache:
  100,000 product page views/hour
  Each view: 5 DB queries
  Total: 500,000 DB queries/hour

With Redis cache (95% hit rate):
  100,000 views, 5 queries each = 500,000 potential queries
  Cache serves 95% = 475,000 queries from Redis ($0.01/GB/hr vs DB cost)
  DB serves only 25,000 queries/hour (5% of original load)
  DB instance can be downsized by 1-2 tiers
```

**DynamoDB cost impact:**
- Each RCU (Read Capacity Unit) handles 4 KB of strongly consistent reads.
- At 500,000 reads/hour for a 1 KB item: 500,000 RCUs/hour = 8,333 RCUs provisioned.
- Provisioned at $0.00013/RCU-hour: 8,333 × $0.00013 = $1.08/hour = $777/month.
- With 95% cache hit: 417 RCUs provisioned = $0.054/hour = $39/month. **95% cost reduction on DB reads.**

**Read Replicas (distribute query load):**
```
Primary DB instance: handles writes + remaining reads
Read Replica 1, 2, 3: each handles a slice of read traffic

Cost: 3 replicas × $200/month = $600/month
But primary can be downsized from db.r5.4xlarge ($2,000/month)
  to db.r5.xlarge ($500/month)
Net: $600 + $500 = $1,100 vs $2,000 → 45% reduction
     Plus read replicas serve different AZs (HA benefit)
```

**Aurora Serverless for variable workloads:**
- Scales compute in 0.5 ACU increments. Costs nothing when paused (dev/test).
- For workloads with high variance (10x difference between peak and trough), Aurora Serverless can be 60–80% cheaper than provisioned.
- At $0.06/ACU-hour vs db.r5.large at $0.24/hour: only pay for what you use.

---

### Q9. Compare serverless cost model (pay-per-request) vs always-on servers. When is each better?

**Serverless (Lambda / Cloud Functions / Azure Functions):**
- Pricing: $0.0000002/request + $0.0000166667/GB-second compute.
- You pay only when code is executed — zero cost when idle.
- Cold starts (50–500 ms) can add latency for infrequently invoked functions.

**Always-on server (EC2 / GCE / ECS container):**
- Pricing: hourly regardless of whether requests are being handled.
- No cold start — instances are always warm.
- Minimum cost even at zero traffic.

**Break-even analysis:**

```
Lambda cost per invocation (128 MB, 100 ms execution):
  Compute: 0.128 GB × 0.1 s × $0.0000166667 = $0.000000213
  Request: $0.0000002
  Total:   ~$0.0000004/invocation = $0.40/1M invocations

t3.micro on-demand: $0.0104/hr = $7.59/month
  Handles ~200 req/s = 720,000 req/hr = 17.3M req/month

Break-even: 17.3M requests/month
  Below this: Lambda is cheaper
  Above this: EC2 t3.micro is cheaper
```

**Decision framework:**

| Scenario | Recommendation |
|---|---|
| < 1M requests/month | Serverless (always cheaper) |
| Spiky traffic (0 to 10k/s bursts) | Serverless (no idle cost between spikes) |
| Predictable sustained load | Reserved EC2 / containers |
| > 10M requests/month | EC2/containers (serverless gets expensive) |
| Sub-100ms latency P99 required | EC2/containers (cold starts are unpredictable) |
| Long-running jobs (> 15 min) | EC2/containers (Lambda max 15-min timeout) |

**Serverless at scale can become expensive:**
```
100M requests/day × 30 days = 3B requests/month
Lambda cost: 3B × $0.0000004 = $1,200/month
20 × t3.medium (handles load): 20 × $0.0416/hr × 730 = $607/month

→ At this scale, containers are ~50% cheaper
```

The decision is not binary — a common pattern is running a base fleet of containers for sustained traffic and using Lambda for burst overflow.

---

### Q10. How do you estimate cost for a new system? Demonstrate back-of-envelope cost calculation.

Back-of-envelope cost estimation is a critical skill for system design interviews. The goal is a rough magnitude estimate (order of magnitude) in 3–5 minutes, not a precise number.

**Step 1: Estimate traffic and data volumes**
```
Example: Design Twitter-scale read-heavy social feed
  DAU: 200M users
  Posts per user per day: 3 (average)
  Reads per user per day: 100 timeline reads × 20 posts each = 2000 post reads
  Write QPS: 200M × 3 / 86,400 = ~7,000 writes/second
  Read QPS:  200M × 2000 / 86,400 = ~4.6M reads/second
```

**Step 2: Compute requirements**
```
  Assume each server handles 10,000 read req/s (typical for a cached read tier)
  Read servers needed: 4,600,000 / 10,000 = 460 servers
  Write servers: 7,000 / 2,000 = 4 servers (writes are much lower)

  Instance type: m5.xlarge (4 vCPU, 16 GB) at $0.192/hr
  460 × $0.192 × 730 hr/month = $64,550/month
  Reserved (40% discount): ~$38,700/month
```

**Step 3: Storage requirements**
```
  Each tweet: ~1 KB
  New tweets/day: 200M × 3 = 600M tweets/day
  Per year: 600M × 365 = 219B tweets/year
  Storage: 219B × 1 KB = 219 TB/year
  S3 cost: 219 TB × $0.023/GB = ~$5,000/month
```

**Step 4: Caching**
```
  Active tweets (last 7 days): 600M × 7 = 4.2B tweets × 1 KB = 4.2 TB
  Cache 1% of hot data: 42 GB
  Elasticache r6g.2xlarge (52 GB): $0.432/hr × 730 = $315/month
```

**Step 5: Data transfer**
```
  Each read returns 20 tweets × 1 KB = 20 KB
  4.6M reads/s × 20 KB = 92 GB/s egress → 7,900 TB/month
  Via CDN at $0.01/GB: 7,900,000 GB × $0.01 = $79,000/month
  (This is the dominant cost — CDN for dynamic content is expensive at Twitter scale)
```

**Summary table:**
| Component | Monthly Cost |
|---|---|
| Compute (EC2 reserved) | $38,700 |
| Database (Cassandra on i3) | $15,000 |
| Cache (Elasticache) | $315 |
| Storage (S3) | $5,000 |
| CDN/egress | $79,000 |
| **Total (rough)** | **~$138,000/month** |

In an interview, present the dominant cost (CDN in this case) and suggest the key optimisation (aggressive caching to reduce CDN costs).

---

### Q11. What is the true cost of unnecessary microservices?

Over-engineering with microservices imposes costs that are not visible in initial architecture diagrams but emerge at runtime and operations time.

**Direct infrastructure costs of unnecessary service decomposition:**

```
Monolith (1 service):
  2 × m5.2xlarge = $0.384/hr × 2 × 730 = $560/month

Unnecessary microservices (10 services, each needing HA):
  Each service: 2 × t3.medium = $0.0416 × 2 × 730 = $60.73/month
  10 services: 10 × $60.73 = $607/month
  Plus: 10 × ALB = 10 × $16.43/month = $164/month
  Plus: 10 × API Gateway (if used) = 10 × $30/month = $300/month
  Total: $1,071/month vs $560 → 91% more expensive
```

**Indirect operational costs:**

| Cost Type | Description | Magnitude |
|---|---|---|
| Service mesh overhead | Envoy/Istio sidecar proxies per pod: 50–100 MB RAM, 5–10% CPU overhead | 10–15% of compute cost |
| Observability | 10x more logs, traces, metrics: proportionally higher Datadog/Splunk bill | Can double observability cost |
| Network cost | Every inter-service call = cross-AZ traffic at $0.01/GB | Adds up for chatty services |
| Engineering time | Each new service = CI/CD pipeline, runbook, on-call rotation setup | Significant hidden cost |
| Latency tax | Each network hop adds 1–5 ms; 5 services in a chain = 5–25 ms extra latency | May require more compute to compensate |

**The decision framework:**
```
Is this service decomposition driven by:
  ✓ Different scaling requirements → justified
  ✓ Different deployment cadences → justified
  ✓ Different team ownership boundaries → justified
  ✗ "Microservices are best practice" → not justified
  ✗ Service does < 1000 req/day → almost certainly not justified
  ✗ Two engineers maintain both services → not justified
```

The principle: **start with a well-structured monolith, extract services only when a concrete scaling or team boundary justification exists**. The most expensive microservice is the one that solves no problem.

---

### Q12. How do you analyse build vs buy cost? Example: self-managed Kafka vs Confluent Cloud.

The **build vs buy decision** is rarely purely technical — it is fundamentally a cost and resource allocation decision. The true cost of "building" (self-managing) includes not just infrastructure but engineering time, operational overhead, and opportunity cost.

**Self-managed Kafka on EC2:**

```
Cluster for 500 MB/s throughput, 3x replication, 7-day retention:
  Brokers: 6 × i3.2xlarge ($0.624/hr) × 730 = $2,733/month
  ZooKeeper: 3 × t3.large ($0.0832/hr) × 730 = $182/month
  Storage: 500 MB/s × 86,400 s/day × 7 days × 3 replicas
           = ~900 TB needed → EBS gp3 at $0.08/GB = $72,000/month

  Wait — use i3.2xlarge local NVMe (1.9 TB each × 6 = 11.4 TB)
  6 brokers is not enough for 900 TB → need ~36 brokers (more storage than compute)
  36 × $0.624 × 730 = $16,400/month (compute only)
  
  Engineering: 1 dedicated SRE to maintain Kafka
  SRE cost: $200K salary / 12 = $16,700/month
  
  Total: ~$33,100/month
```

**Confluent Cloud (managed Kafka):**
```
  500 MB/s throughput: ~100 CKUs (Confluent Units)
  CKU cost: $0.44/CKU-hr × 100 × 730 = $32,120/month
  Storage: included (or $0.0000008/byte/hr)
  
  Total: ~$32,000–$38,000/month
  Engineering: ~0.1 SRE equivalent (monitoring only)
```

**Build vs buy comparison:**

| Factor | Self-Managed | Confluent Cloud |
|---|---|---|
| Infrastructure cost | $16,400/month | $32,120/month |
| Engineering (SRE) | $16,700/month | ~$1,700/month |
| Total | ~$33,100/month | ~$33,820/month |
| Flexibility | Full control | Limited by Confluent API |
| Incident response | Internal SRE on-call | Confluent SLA |
| Upgrade burden | Manual | Managed |
| Compliance | Full control | Depends on Confluent certifications |

In this example, cost is comparable — but Confluent Cloud frees one SRE to work on product features rather than Kafka operations. The true ROI of "buy" is often not cost reduction but **engineering velocity recovered**.

---

### Q13. How do you manage log retention costs at scale?

Logging is one of the most commonly under-optimised cost areas. At scale, raw logging of everything at DEBUG level is prohibitively expensive.

**The log volume problem:**
```
1,000 services × 100 req/s each = 100,000 req/s total
Average log lines per request: 20 lines × 500 bytes = 10 KB/request
Log throughput: 100,000 × 10 KB = 1 GB/s = 86 TB/day

Splunk cost: ~$150/GB/day ingested
Daily cost: 86 TB × $150 = $12,900,000/day ← clearly untenable
```

**Cost reduction strategies:**

**1. Log level control (most impactful):**
```python
# Production: ERROR and WARN only (reduce volume 90–99%)
# Staging: INFO
# Debug: on-demand only, with time-limited activation

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARN')
```

**2. Structured logging with sampling:**
```python
import structlog
logger = structlog.get_logger()

# Only log 1% of successful requests; log 100% of errors
if random.random() < 0.01 or status_code >= 500:
    logger.info("request", path=path, status=status_code, latency_ms=latency)
```

**3. Tiered log storage:**
```
Hot tier (Elasticsearch/Splunk): last 7 days for search and alerting
  Cost: $3/GB/day × 7 days retention = high but short window

Warm tier (S3 + Athena): 7 days to 90 days
  S3: $0.023/GB/month
  Athena queries: $5/TB scanned

Cold tier (S3 Glacier): 90 days to 7 years (compliance)
  Glacier: $0.004/GB/month

For 1 TB/day:
  Hot (7 days): 7 TB × $3/GB/day = $21,000/month → only 7 days of searchable logs
  Warm (83 days): 83 TB × $0.023/GB = $1,909/month
  Cold (remaining): archive cost
```

**4. Metrics instead of logs:**
- Replace logging request counts with a Prometheus counter. One counter = 24 bytes; one log line = 500 bytes.
- For high-cardinality data (per-request timing), use distributed tracing with a **1% sampling rate** instead of logging every request.

**5. Log aggregation compression:**
- Elasticsearch with Deflate compression: 4–10x size reduction.
- Kinesis Firehose with gzip before writing to S3: additional 5x reduction on text logs.

---

### Q14. How does observability cost grow with cardinality, and how do you control it?

**Cardinality** in metrics refers to the number of unique time series created by unique combinations of label values. High cardinality is the primary driver of spiralling observability costs.

**The cardinality explosion:**
```python
# Low cardinality (safe):
http_requests_total{method="GET", status="200"}           # 10 combinations
http_requests_total{method="POST", status="500"}

# High cardinality (dangerous):
http_requests_total{user_id="user-550e8400"}              # 10M combinations
http_requests_total{url="/api/users/550e8400/profile"}    # unbounded
```

With 10M users, using `user_id` as a label creates 10M time series. At $0.30/million time series/month (Datadog), this costs $3,000/month for one metric alone.

**Prometheus/Datadog cardinality cost model:**
```
Cost = (number of unique label combinations) × (metrics count) × (rate per series)
     = 10M user_ids × 5 HTTP metrics × $0.30/M/month
     = $15,000/month for user-level metrics alone
```

**Control strategies:**

| Strategy | Description | Impact |
|---|---|---|
| Remove high-cardinality labels | Never use user_id, request_id as labels | Largest impact |
| Use histograms not per-request metrics | Record latency as histogram buckets, not per-request | 99%+ reduction |
| Trace sampling | Sample 1% of traces; keep 100% of error traces | 100x volume reduction |
| Metric aggregation | Aggregate at collection point, not at ingest | Reduce series before they hit the paid tier |
| Allowlist metrics | Only collect metrics you have dashboards/alerts for | Eliminate unused series |

**Trace sampling strategy:**
```python
# Head-based sampling: decide at start of request
sampler = TraceIdRatioBased(0.01)   # 1% of all requests

# Tail-based sampling: decide after completion (more intelligent)
if response.status >= 500 or response.latency_ms > 5000:
    keep = True     # Always keep errors and slow requests
else:
    keep = random() < 0.01    # 1% of normal requests
```

At $0.10/100K traces, 1% sampling on 100M requests/day = 1M traces/day = 1B/month = $1,000/month vs $100,000/month for 100% sampling.

---

### Q15. What are the diminishing returns of moving from 99.9% to 99.99% to 99.999% availability?

Availability targets have a **non-linear relationship with cost**. Each additional nine of availability roughly doubles to triples the engineering and infrastructure investment required.

**Downtime per year by availability level:**

| Availability | Downtime/year | Downtime/month | Downtime/week |
|---|---|---|---|
| 99% | 87.6 hours | 7.3 hours | 1.7 hours |
| 99.9% | 8.76 hours | 43.8 minutes | 10.1 minutes |
| 99.99% | 52.6 minutes | 4.4 minutes | 1.0 minutes |
| 99.999% | 5.26 minutes | 26.3 seconds | 6 seconds |
| 99.9999% | 31.5 seconds | 2.6 seconds | 0.6 seconds |

**Cost of each additional nine:**

```
99.9% → Basic HA:
  2 AZs, active-passive DB failover, basic load balancer
  Infrastructure premium over single AZ: ~20%
  Engineering investment: Low
  Estimated cost: $50K/month for a mid-size service

99.99% → Multi-AZ active-active:
  3 AZs, synchronous DB replication, hot standby, auto-failover
  Requires extensive testing, chaos engineering
  Infrastructure premium: ~50-80% over 99.9%
  Estimated cost: $85K/month (+70%)

99.999% → Multi-region active-active:
  2+ regions, global load balancing, zero-downtime deploys
  Requires distributed systems expertise, complex runbooks
  Infrastructure premium: 2-3x over 99.99%
  Estimated cost: $200K+/month (+135%)
```

**Business justification calculation:**
```
Revenue per hour: $100,000
99.9% → 8.76 hrs downtime/year × $100K = $876K potential revenue loss
99.99% → 0.88 hrs × $100K = $88K potential revenue loss
Value of going 99.9% → 99.99%: $788K/year

Extra infrastructure cost to go 99.9% → 99.99%: $35K/month × 12 = $420K/year

ROI positive: yes (save $788K, spend $420K)

Value of 99.99% → 99.999%: $85K/year
Extra cost: $115K/month × 12 = $1.38M/year

ROI negative: spend $1.38M to prevent $85K in losses
```

The engineering principle: **design for the availability that is financially justified, not for the highest technically achievable**. Always do the business case math before committing to an extra nine.

---

## Hard (Q16–Q20)

---

### Q16. How do you optimise data egress across AZs and regions in a microservices architecture?

In a microservices architecture with dozens of services calling each other, **inter-service data transfer costs** can exceed the cost of the compute running those services — particularly for services that exchange large payloads.

**Understanding AZ transfer costs:**
```
us-east-1a → us-east-1b: $0.01/GB each direction = $0.02/GB round trip
At 10 GB/s inter-service traffic: $0.02 × 10 GB × 86,400 s/day = $17,280/day
Monthly: $518,400 — from AZ transfer alone
```

**Optimisation strategies:**

**1. Topology-aware placement:**
```yaml
# Kubernetes: prefer pods in same AZ as their dependencies
affinity:
  podAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchLabels:
            app: user-database
        topologyKey: topology.kubernetes.io/zone
```

**2. Payload size reduction:**
- Use Protobuf/Avro instead of JSON: 5–10x size reduction for structured data.
- Enable gRPC with gzip compression: additional 40–60% reduction.
- Field masking: return only requested fields, not full objects.

```protobuf
// Instead of returning entire User object (2 KB)
// Return only the fields the caller needs (200 bytes)
message GetUserResponse {
  string user_id = 1;
  string display_name = 2;
  // Omit: billing info, preferences, history, etc.
}
```

**3. Caching at service boundaries:**
- Service A calls Service B 100 times/second for user profile data.
- Profile changes < 1 time/hour.
- Cache profile at Service A layer with 5-minute TTL → 99.97% of calls served locally.
- Inter-AZ calls: 100/s → 0.03/s (99.97% reduction).

**4. Read from local replica:**
- Deploy read replicas of databases in each AZ.
- Services read from their local AZ replica (no AZ transfer cost).
- Writes go to primary (one AZ transfer), reads are free.

**5. S3 Transfer Acceleration vs same-region transfer:**
- For services that must read from S3, deploy in the same region.
- S3 Transfer Acceleration adds cost — only use for cross-region acceleration, not same-region.

**Cost monitoring:**
```
Tools:
  AWS Cost Explorer: group by Usage Type → filter "DataTransfer-Regional"
  AWS VPC Flow Logs + Athena: identify top talkers by AZ pair
  Grafana dashboards: inter-service call volume × payload size
```

---

### Q17. Explain CDN cost vs origin server cost trade-off. When can CDN be more expensive?

CDN is almost always cost-effective for static content but can become more expensive than origin serving in specific scenarios. Understanding the math prevents over-relying on CDN as a universal solution.

**Standard CDN economics (static content):**
```
Origin serving 100 TB/month of images:
  EC2 bandwidth cost: 100 TB × 1024 GB × $0.085/GB = $8,704/month
  EC2 compute (serving 1 Gbps): 2 × c5.xlarge = $0.192 × 2 × 730 = $280/month
  Total without CDN: $8,984/month

With CloudFront (90% cache hit rate):
  CDN serves: 90 TB × 1024 GB × $0.0075/GB = $690/month (CloudFront pricing)
  Origin: 10 TB × $0.085 = $870/month (reduced origin egress)
  EC2 compute (only 10% traffic): 1 × t3.medium = $30/month
  Total with CDN: $1,590/month

Savings: $7,394/month (82% reduction)
```

**When CDN becomes more expensive:**

**1. Very low cache hit rate (dynamic, personalised content):**
```
Dynamic API responses (0% cacheable):
  CDN: all 100 TB × $0.0075/GB = $768/month
  Plus: origin still serves 100% of requests = origin costs unchanged
  CDN cost is additive, not replacing origin cost
  
Break-even cache hit rate: ~60–70% (below this, CDN adds cost without proportional saving)
```

**2. Short-lived content (frequent invalidation):**
```
News website with articles that update every 5 minutes:
  Cache TTL: 5 minutes = objects expire 288 times per day
  Each expiry triggers origin fetch = near 100% cache miss for popular stories
  CDN costs the same as origin + CDN service fee
```

**3. CDN with premium features (WAF, bot protection):**
```
Cloudflare Business: $200/month per domain
Cloudflare WAF: $5/month per rule set
DDoS protection: ~$3,000/month for enterprise DDoS SLA
These costs are fixed regardless of traffic volume
For low-traffic sites: CDN premium features may cost more than origin
```

**Decision matrix:**

| Content Type | Cache Hit Rate | CDN Cost vs Origin |
|---|---|---|
| Static assets (JS/CSS/images) | 95%+ | 80–90% cheaper |
| Product catalogue (update/day) | 80% | 60–70% cheaper |
| News articles (update/hour) | 50% | 10–20% cheaper |
| Personalised API responses | 0% | More expensive (adds cost) |
| Real-time data (WebSocket) | N/A | Not applicable |

---

### Q18. How should you right-size a database instance? Cover RDS instance selection and storage autoscaling.

Database right-sizing is more nuanced than compute right-sizing because database performance is multi-dimensional: CPU, memory (buffer pool size), IOPS, and network throughput all interact.

**Step 1: Understand the bottleneck**
```
Common database bottlenecks:

CPU-bound: Complex queries, many connections, OLAP workloads
  → Scale up CPU (db.r5 → db.r5.2xlarge), add read replicas

Memory-bound: Buffer pool thrashing (buffer pool hit ratio < 99%)
  → Scale up memory (db.r5 → db.r6g.2xlarge)
  → Rule: buffer pool should be 80–100% of working dataset

IOPS-bound: High write volume, insufficient disk throughput
  → Increase provisioned IOPS (gp2 → gp3, or io1)
  → gp3: 3,000 IOPS baseline, up to 16,000 at $0.02/provisioned IOPS

Network-bound: Read replicas saturating network
  → Upgrade to instance with higher network bandwidth
```

**RDS instance family guide:**

| Family | vCPU:RAM ratio | Use case | Example |
|---|---|---|---|
| db.t3/t4g | 1:2 | Dev/test, low traffic | db.t3.medium (2 vCPU, 4 GB) |
| db.m5/m6g | 1:4 | General purpose | db.m5.xlarge (4 vCPU, 16 GB) |
| db.r5/r6g | 1:8 | Memory-intensive (large buffer pool) | db.r5.2xlarge (8 vCPU, 64 GB) |
| db.x2g | 1:32 | In-memory databases, SAP HANA | db.x2g.xlarge (4 vCPU, 128 GB) |

**Buffer pool sizing:**
```sql
-- Check buffer pool hit ratio (MySQL)
SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool%';
-- Innodb_buffer_pool_read_requests / (read_requests + reads_from_disk)
-- Target: > 99%

-- If ratio is 95%, buffer pool is too small
-- Increase instance size until ratio reaches 99%+
```

**Storage autoscaling:**
```terraform
resource "aws_db_instance" "main" {
  instance_class        = "db.r5.xlarge"
  allocated_storage     = 100    # GB initial
  max_allocated_storage = 1000   # GB maximum (autoscaling ceiling)
  storage_type          = "gp3"
  iops                  = 3000   # Baseline, increase if IOPS-bound
}
# RDS automatically expands storage when 10% free space remains
# No downtime for storage expansion
# Prevents disk-full incidents without manual intervention
```

**Cost-performance optimisation example:**
```
Current: db.r5.4xlarge (16 vCPU, 128 GB RAM) — $1,557/month
Buffer pool hit ratio: 99.8% (memory not the bottleneck)
CPU P95: 15% (not CPU-bound)
IOPS consumed: 2,000/s (low)

Diagnosis: Over-provisioned on both CPU and memory
Right-size: db.r5.xlarge (4 vCPU, 32 GB RAM) — $389/month
Buffer pool hit ratio (projected): still > 99% (working set fits in 32 GB)
Savings: $1,168/month (75% reduction)
```

---

### Q19. How do you design a cost-aware data pipeline for a petabyte-scale analytics system?

Petabyte-scale analytics is one of the most expensive workloads in cloud — poor design can easily generate $500K+/month in avoidable costs. Cost must be a first-class design constraint, not an afterthought.

**Architecture for cost-optimised analytics:**

```
Data Sources
     │
     ▼
Kinesis Data Streams / Kafka
(raw event ingestion — $0.015/shard-hr)
     │
     ▼
S3 Raw Data Lake (compressed Parquet)
(storage: $0.023/GB, but Parquet = 5-10x compression vs JSON)
     │
     ├──▶ Batch ETL (AWS Glue / Spark on EMR Spot)
     │    (process on-demand, use spot instances 60-80% cheaper)
     │
     └──▶ S3 Curated Tables (Iceberg/Delta format)
              │
              ├──▶ Athena (serverless SQL, $5/TB scanned)
              └──▶ Redshift Spectrum (for complex joins)
```

**Key cost levers:**

**1. File format matters enormously:**
```
1 TB JSON logs:
  S3 storage: 1 TB × $0.023/GB = $23,000/month
  Athena scan: 1 TB × $5/TB = $5/query

Convert to Parquet + Snappy compression:
  Compression ratio: ~10x → 100 GB
  S3 storage: 100 GB × $0.023 = $2,300/month (90% reduction)
  Athena scan: 100 GB × $5/TB = $0.50/query (90% reduction)
```

**2. Partition design for query pruning:**
```
Bad partitioning:  s3://data-lake/events/event_id=abc123/
  → Every query scans all data

Good partitioning: s3://data-lake/events/year=2024/month=01/day=15/hour=09/
  → Query "WHERE date = '2024-01-15'" scans only 1/365th of data
  → 99.7% cost reduction for date-filtered queries
```

**3. Spot instances for batch processing:**
```python
# EMR cluster for daily batch ETL
emr_config = {
    "master": "m5.xlarge (on-demand)",  # Small, on-demand for reliability
    "core": "m5.2xlarge × 10 (spot)",  # Large workers on spot
    "spot_bid": "max_on_demand_price",  # Auto-bid at on-demand price ceiling
    "spot_timeout": "terminate"         # If spot unavailable, use on-demand
}
# Spot saves 60-80% on core node compute
```

**4. Query cost governance:**
```sql
-- Athena: Always specify partition filter to limit scan
SELECT COUNT(*) FROM events
WHERE year = '2024' AND month = '01'  -- Partition pruning
  AND event_type = 'purchase'          -- Column pruning via Parquet
  
-- Without partition filter: scans 10 TB = $50
-- With partition filter: scans 100 GB = $0.50
```

**5. Tiered query service:**
```
Frequent operational queries (last 7 days):
  → Redshift (reserved nodes): fast, fixed cost, $250/node/month

Ad-hoc historical queries (months to years):
  → Athena: pay-per-query, $5/TB, no cluster cost

Data science workloads (ML feature engineering):
  → EMR Spot + autoscaling: pay per job, not per cluster-hour
```

---

### Q20. Design a multi-region cost-optimised architecture for a SaaS application serving 10M users globally.

This question integrates all cost optimisation principles into a realistic end-to-end architecture decision.

**Requirements assumed:**
- 10M DAU globally distributed: 40% US, 35% EU, 25% APAC
- Read-heavy (95% reads, 5% writes)
- 99.99% availability target
- Monthly budget target: < $200K

**Architecture design:**

```
                    Route 53 GeoDNS
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      us-east-1       eu-west-1     ap-southeast-1
      (Primary)       (Active)       (Active)

Each region:
  CloudFront CDN  ←── Static assets cached at edge (95%+ hit rate)
       │
  ALB + WAF
       │
  ECS Fargate (auto-scaling, Spot for non-critical tasks)
       │
  ElastiCache Redis (read cache, 90% cache hit rate)
       │
  Aurora PostgreSQL
  - Primary: us-east-1 (all writes)
  - Read replicas: eu-west-1, ap-southeast-1
```

**Cost breakdown by region:**

```
Per region (us-east-1 primary shown, replicas slightly cheaper):

Compute:
  ECS Fargate: auto-scaling, avg 20 tasks × 0.5 vCPU × 1 GB
  Fargate cost: 20 × $0.04048/vCPU-hr × 0.5 × 730 = $295/month
  Spot for batch: $0.01/task × 10K tasks/day = $100/month

Database:
  Aurora db.r6g.2xlarge (primary): $0.58/hr × 730 = $423/month
  Aurora read replica × 2: $0.29 × 2 × 730 = $423/month
  Storage: 500 GB × $0.10/GB = $50/month
  
Cache (ElastiCache):
  r6g.large × 2: $0.166/hr × 2 × 730 = $242/month
  
CDN (CloudFront):
  1 TB/month static assets × $0.0075/GB = $7.50/month
  Cache hit rate 95%, origin egress reduced by 95%
  
Data transfer:
  1 TB/month user-facing egress: $75/month (via CDN)
  
Total per region: ~$1,615/month
Three regions: ~$4,845/month
```

**Reserved instance savings:**
```
1-year reserved Aurora + ElastiCache: 40% discount
Fargate Savings Plan: 20% discount

Before reserved: $4,845/month
After reserved:  $3,300/month
Annual commitment: $3,300 × 12 = $39,600/year
```

**Other infrastructure costs:**
```
Route 53: $50/month (health checks + hosted zone)
S3 (assets, logs): $200/month
CloudWatch + alerting: $150/month
ACM, IAM, misc: $50/month
Total other: $450/month
```

**Grand total: ~$3,750/month** — well within the $200K/month budget for 10M DAU.

The key cost levers: Fargate auto-scaling (no idle compute), aggressive caching (90% DB load deflected), CDN for static assets (origin egress reduced 95%), and reserved instances for predictable baseline load. The architecture achieves 99.99% through multi-region active-active with Aurora Global Database — failover within 1 minute for global database reads, primary failover in 60 seconds.

---

## Quick Reference

| Topic | Key Point |
|---|---|
| Cost as NFR | Treat cost like latency — define a budget target and optimise toward it |
| Three cost drivers | Compute (40–60%), storage (20–30%), egress (10–20%) |
| Right-sizing | Target P95 CPU at 60–70%; use Compute Optimizer for recommendations |
| Reserved instances | 30–60% savings; commit only to predictable baseline load |
| Spot instances | 60–90% savings; use for fault-tolerant, interruptible workloads |
| Auto-scaling | Eliminate idle capacity; scale-in cooldown longer than scale-out |
| Egress reduction | CDN cuts egress 80–90%; place services in same AZ as data |
| S3 lifecycle policies | Move to IA/Glacier by age; 70%+ storage cost reduction |
| Caching | 95% cache hit rate → 95% DB cost reduction; cache is almost always ROI positive |
| Serverless break-even | < 1M req/month: serverless wins; > 10M req/month: containers usually cheaper |
| Back-of-envelope | Estimate traffic → compute → storage → egress; identify dominant cost |
| Microservices cost | Each unnecessary service: +ALB, +observability, +SRE overhead |
| Build vs buy | Include engineering salary in self-managed cost; buy often breaks even |
| Log cost control | Log WARN/ERROR in prod; sample 1% of success; tiered storage (hot/warm/cold) |
| Cardinality | Never use user_id/request_id as metric labels; use histograms |
| Availability cost | 99.9% → 99.99% → 2x cost; 99.99% → 99.999% → 3x cost; do the business case |
| AZ transfer | $0.01/GB each direction; place consumers in same AZ as producers |
| CDN economics | Only cost-effective with > 60% cache hit rate; dynamic content CDN adds cost |
| DB right-sizing | Buffer pool hit ratio > 99% = memory adequate; CPU P95 < 60% = over-provisioned |
| Data pipeline | Parquet + partitioning = 90% query cost reduction vs raw JSON |
