# HLD Interview Q&A — File 11: Monitoring and Observability

---

## Easy Questions (Q1–Q7)

---

### Q1. What are the three pillars of observability, and what question does each pillar answer?

**Answer:**

Observability is the ability to understand the internal state of a system from its external outputs. The three pillars are metrics, logs, and traces — each answering a distinct question.

**Metrics** answer: "Is something wrong right now?"
Metrics are numerical time-series measurements aggregated over time (e.g., CPU %, request rate, error count). They are cheap to store and fast to query, making them ideal for dashboards and alerting. Tools: Prometheus, Datadog, CloudWatch.

**Logs** answer: "Why did it go wrong?"
Logs are discrete, timestamped text (or structured) records of events. They provide the narrative of what happened. Tools: ELK Stack, Loki, Splunk.

**Traces** answer: "Where did it go wrong across services?"
A distributed trace follows a single request through multiple services, measuring latency at each hop. Tools: Jaeger, Zipkin, AWS X-Ray.

```
User Request
    │
    ├─ [Metric]  → HTTP 500 rate spiked at 14:32
    │
    ├─ [Log]     → ERROR: DB connection timeout in OrderService
    │
    └─ [Trace]   → API Gateway (5ms) → OrderService (2s) → DB (TIMEOUT)
```

| Pillar  | Question       | Cardinality | Storage Cost | Latency |
|---------|---------------|-------------|--------------|---------|
| Metrics | Is it broken? | Low         | Low          | Seconds |
| Logs    | Why broken?   | High        | High         | Minutes |
| Traces  | Where broken? | High        | Medium       | Minutes |

A system is truly observable only when all three pillars are correlated — a trace ID that appears in both a log line and a metric label is the gold standard.

---

### Q2. What is the difference between monitoring and observability?

**Answer:**

These terms are often used interchangeably but represent different philosophies.

**Monitoring** is about tracking known failure modes. You define upfront what you want to watch — CPU, memory, error rate — and set alerts for when those cross thresholds. It answers "Did the thing I expected to break actually break?" It is reactive and requires you to know what questions to ask.

**Observability** is about understanding unknown failure modes. In a complex distributed system, failures emerge from combinations of states you never anticipated. Observability means your system emits enough rich data (metrics, logs, traces) that you can ask *any* question after the fact — including questions you didn't think to ask before.

```
Monitoring:        Known unknowns
                   "Alert me if DB CPU > 80%"

Observability:     Unknown unknowns
                   "Why did p99 spike for users in eu-west-1 on Tuesday?"
```

**Analogy:** Monitoring is like a car dashboard (tells you oil is low). Observability is like having full telemetry that lets engineers debug *why* the engine behaved strangely on a specific Tuesday afternoon.

**In practice:** Monitoring is a subset of observability. You can have monitoring without observability (checking if a server is up), but you cannot have true observability without monitoring (the metrics/alerts are part of the observability stack).

Modern teams aim for observability because microservices create too many failure combinations to enumerate upfront.

---

### Q3. What are SLI, SLO, and SLA? How do you define them?

**Answer:**

These three acronyms form the reliability contract hierarchy for a service.

**SLI (Service Level Indicator):** A specific measurable metric that represents a dimension of reliability. It is the raw signal.
- Examples: request success rate, p99 latency, availability percentage, error rate.

**SLO (Service Level Objective):** An internal target for an SLI over a time window. It is a promise your engineering team makes to itself.
- Example: "99.9% of requests will succeed over any rolling 28-day window."

**SLA (Service Level Agreement):** An external contract with customers that includes financial consequences (credits/penalties) for breach. It is typically set lower than the SLO to give a buffer.
- Example: "We guarantee 99.5% uptime. If we miss it, you get a 10% credit."

```
SLI  →  What you measure      (success_rate = successful_requests / total_requests)
SLO  →  What you target       (success_rate ≥ 99.9% over 28 days)
SLA  →  What you promise      (success_rate ≥ 99.5% or credits apply)

Buffer: SLA is always weaker than SLO to avoid SLA breach when SLO is missed.
```

**Defining good SLIs:**
1. Choose metrics that reflect user experience, not internal implementation (e.g., "checkout succeeded" not "DB query time").
2. Keep the number small — 3-5 SLIs per service is ideal.
3. Make them measurable with existing instrumentation.

**Defining good SLOs:**
1. Start with historical data — what have you actually achieved?
2. Align with user expectations, not engineering pride.
3. Revisit quarterly.

---

### Q4. What is an error budget, and how do teams use it?

**Answer:**

An error budget is the maximum allowable unreliability permitted within an SLO window. It transforms reliability from an abstract goal into a quantitative resource that can be spent and managed.

**Calculation:**
```
SLO = 99.9% availability over 28 days
Error budget = 100% - 99.9% = 0.1%
Total minutes in 28 days = 40,320
Allowed downtime = 40,320 × 0.001 = ~40 minutes
```

**How teams use it:**

1. **Gate deployments:** If error budget is > 50% remaining, teams can ship new features aggressively. If < 10% remains, all hands go to reliability work and deployments are frozen.

2. **Resolve developer vs. ops tension:** Instead of arguments about "is it safe to deploy?", the error budget provides an objective answer.

3. **Inform postmortems:** Incidents are analyzed in terms of how much budget they consumed.

4. **Burn rate alerts:** Instead of alerting when SLO is breached (too late), alert when the budget is burning too fast.

```
Error Budget Policy:
┌────────────────────────────────────────────────────────┐
│ Remaining Budget │ Engineering Focus                   │
├──────────────────┼─────────────────────────────────────┤
│ > 50%            │ Ship features, acceptable risk       │
│ 25–50%           │ Balance features and reliability     │
│ 10–25%           │ Reliability work prioritized         │
│ < 10%            │ Feature freeze, incident response    │
└────────────────────────────────────────────────────────┘
```

The error budget concept comes from Google's SRE book and is one of the most powerful ideas in modern reliability engineering.

---

### Q5. What is structured logging, and why does it win over unstructured logging?

**Answer:**

**Unstructured logging** produces human-readable text lines with no consistent format:
```
2024-01-15 14:23:01 ERROR Failed to process order 12345 for user john@example.com: timeout after 5000ms
```

**Structured logging** produces machine-parseable key-value records, typically in JSON:
```json
{
  "timestamp": "2024-01-15T14:23:01Z",
  "level": "ERROR",
  "message": "Failed to process order",
  "order_id": "12345",
  "user_email": "john@example.com",
  "duration_ms": 5000,
  "error_type": "timeout",
  "service": "order-service",
  "trace_id": "abc123"
}
```

**Why structured logging wins:**

| Criterion         | Unstructured         | Structured              |
|-------------------|----------------------|-------------------------|
| Querying          | Regex (slow, fragile)| Field-based (fast)      |
| Aggregation       | Manual parsing       | Native (COUNT by field) |
| Alerting          | Hard                 | Trivial                 |
| Cross-service     | Manual correlation   | trace_id join           |
| Machine readability | No               | Yes                     |

**Practical example — querying in ELK:**
```
# Unstructured: regex to find timeout errors for a specific order
message: /order 12345.*timeout/

# Structured: field-based query
order_id: "12345" AND error_type: "timeout"
```

The structured query is 10-100x faster and more reliable. Fields can be indexed, while regex scans raw text.

**Best practice:** Always include `trace_id`, `service`, `level`, `timestamp`, and relevant business identifiers (user_id, order_id) in every log line.

---

### Q6. What is the ELK stack architecture?

**Answer:**

ELK stands for Elasticsearch, Logstash, and Kibana. Together they form a widely-used log aggregation, search, and visualization platform.

```
Application Servers
       │
       ▼
  [Filebeat / Fluentd]    ← Lightweight log shippers on each host
       │
       ▼
  [Logstash]              ← Parse, filter, transform, enrich logs
       │
       ▼
  [Elasticsearch]         ← Distributed search and analytics engine
       │
       ▼
  [Kibana]                ← Dashboard and query UI
```

**Component responsibilities:**

**Filebeat:** Lightweight agent running on each server. Tails log files and ships them to Logstash or directly to Elasticsearch. Handles backpressure.

**Logstash:** Processes raw log data — parses formats (grok patterns), adds fields (geo-IP from IP address), filters noise, routes to destinations. CPU-intensive.

**Elasticsearch:** Stores logs as JSON documents in inverted indexes. Provides full-text search and aggregations. Horizontally scalable via sharding.

**Kibana:** Web UI for querying (KQL), visualizing (dashboards, lens), and exploring logs. Supports alerting.

**Modern evolution — EFK stack:** Logstash is often replaced by Fluentd (more resource-efficient) giving EFK. Elastic also offers a Beats family (Metricbeat, Heartbeat) for metrics and uptime.

**Scaling concern:** Elasticsearch is expensive at scale. Alternatives like Grafana Loki (index-free, label-based) are popular for pure log storage because they are much cheaper.

---

### Q7. What are liveness, readiness, and startup probes in Kubernetes?

**Answer:**

Kubernetes uses these three probe types to manage container health and traffic routing.

**Liveness Probe:** Answers "Is the container still alive and should it keep running?"
- If it fails, Kubernetes **restarts** the container.
- Use case: detect deadlocks or infinite loops where the app is running but stuck.

**Readiness Probe:** Answers "Is the container ready to receive traffic?"
- If it fails, Kubernetes **removes the pod from the Service endpoints** (no traffic is sent), but does NOT restart it.
- Use case: during startup, during cache warming, or temporary dependency outages.

**Startup Probe:** Answers "Has the container finished its slow startup?"
- Disables liveness and readiness probes until it succeeds.
- If it fails after `failureThreshold` attempts, Kubernetes restarts the container.
- Use case: legacy apps with slow initialization (JVM warmup, DB migration).

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  periodSeconds: 5

startupProbe:
  httpGet:
    path: /health/startup
    port: 8080
  failureThreshold: 30    # 30 * 10s = 5 minutes to start
  periodSeconds: 10
```

| Probe     | Failure Action     | Use Case                    |
|-----------|--------------------|-----------------------------|
| Liveness  | Restart container  | Deadlock detection          |
| Readiness | Remove from LB     | Warm-up, dependency outage  |
| Startup   | Restart if timeout | Slow-starting applications  |

---

## Medium Questions (Q8–Q15)

---

### Q8. How does Prometheus work? Explain the scrape model, PromQL, and Alertmanager.

**Answer:**

Prometheus is a pull-based monitoring system. Unlike push-based systems, Prometheus **scrapes** metrics from targets at a configured interval.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                        Prometheus Server                     │
│  ┌──────────┐   ┌────────────┐   ┌───────────────────────┐  │
│  │ Retrieval│   │  TSDB      │   │  HTTP API / PromQL    │  │
│  │ (Scraper)│   │ (Storage)  │   │  Engine               │  │
│  └────┬─────┘   └─────┬──────┘   └───────────┬───────────┘  │
└───────┼───────────────┼───────────────────────┼─────────────┘
        │               │                       │
        ▼               ▼                       ▼
   Target /metrics   Disk             Grafana / Alertmanager
   endpoints
```

**Scrape model:**
- Each service exposes a `/metrics` HTTP endpoint in Prometheus exposition format.
- Prometheus polls each target every `scrape_interval` (default 15s).
- Pull model advantage: Prometheus controls the rate, dead services are obvious (scrape fails).
- **Pushgateway** exists for short-lived jobs that can't be scraped.

**PromQL examples:**
```promql
# Request rate per second (5-minute window)
rate(http_requests_total[5m])

# Error ratio
sum(rate(http_requests_total{status=~"5.."}[5m])) 
  / sum(rate(http_requests_total[5m]))

# 99th percentile latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# CPU usage by pod
100 - (avg by (pod) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

**Alertmanager:**
- Receives alerts fired by Prometheus rule evaluation.
- Handles: deduplication (same alert from multiple instances), grouping, silencing, inhibition, and routing to receivers (PagerDuty, Slack, email).

```yaml
# Prometheus alert rule
- alert: HighErrorRate
  expr: rate(http_errors_total[5m]) / rate(http_requests_total[5m]) > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Error rate above 5%"
```

---

### Q9. How does distributed tracing work? Explain trace IDs, spans, and context propagation.

**Answer:**

Distributed tracing reconstructs the full journey of a request across multiple services by attaching a unique identifier to every request and recording timing data at each hop.

**Core concepts:**

**Trace:** The entire end-to-end journey of one request. Identified by a globally unique `trace_id`.

**Span:** A single unit of work within a trace (e.g., one service call, one DB query). Each span has:
- `span_id` (unique within trace)
- `parent_span_id` (links to caller)
- `start_time`, `end_time`
- `service_name`, `operation_name`
- Tags and logs

**Context propagation:** The `trace_id` and `span_id` must be passed between services, typically via HTTP headers (W3C TraceContext standard):
```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
              version  traceID                     spanID              flags
```

**Visualization:**
```
Trace ID: 4bf92f35...
│
├── [span] API Gateway        0ms ─────────────────── 250ms
│       │
│       ├── [span] AuthService    5ms ── 30ms
│       │
│       └── [span] OrderService  35ms ─────────────── 240ms
│               │
│               ├── [span] DB Query (select)  40ms ── 80ms
│               │
│               └── [span] PaymentService     90ms ── 235ms
│                       │
│                       └── [span] Stripe API  95ms ── 230ms  ← SLOW
```

**Context propagation in code:**
```python
# OpenTelemetry Python example
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer(__name__)

# In service A (outgoing call)
with tracer.start_as_current_span("call-order-service") as span:
    headers = {}
    inject(headers)  # Injects traceparent header
    response = requests.post("http://order-service/orders", headers=headers)

# In service B (incoming request)
context = extract(request.headers)  # Extracts trace context
with tracer.start_as_current_span("process-order", context=context):
    ...
```

---

### Q10. What is OpenTelemetry and how does it compare to vendor SDKs?

**Answer:**

OpenTelemetry (OTel) is a CNCF project providing a vendor-neutral, open-source observability framework for generating, collecting, and exporting telemetry data (metrics, logs, traces).

**The problem it solves:**
Before OTel, every observability vendor (Datadog, New Relic, Dynatrace, Jaeger) had its own SDK. Switching vendors meant rewriting instrumentation across every service.

**OpenTelemetry architecture:**
```
Your Application
      │
      ▼
[OTel SDK / Auto-Instrumentation]
      │  Generates spans, metrics, logs
      ▼
[OTel Collector]    ← Optional but recommended
      │  Receives, processes, exports
      ├──────────────────────────────────┐
      ▼                                  ▼
[Jaeger / Zipkin]            [Datadog / New Relic]
(open-source backend)        (commercial backend)
```

**Comparison:**

| Dimension          | Vendor SDK (e.g., Datadog)    | OpenTelemetry                      |
|--------------------|-------------------------------|------------------------------------|
| Vendor lock-in     | High                          | None (swap backends freely)        |
| Features           | Rich (APM, profiling, RUM)    | Core telemetry only                |
| Auto-instrumentation| Agent-based, turnkey         | Good but some manual config        |
| Community          | Vendor-supported              | CNCF, huge community               |
| Maturity           | Production-proven             | Traces/Metrics stable, Logs newer  |
| Cost               | Per-host pricing              | Free (pay for backend)             |

**Recommendation:** Use OTel for instrumentation (it is the standard), then route to whichever backend fits your budget and feature needs. Instrumentation code stays identical; only the exporter config changes.

```python
# OTel — same code, different exporter config
from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # or
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # Datadog/New Relic
```

---

### Q11. How do you alert without causing alert fatigue?

**Answer:**

Alert fatigue occurs when too many low-quality alerts cause engineers to ignore or disable them, defeating the purpose of monitoring. The solution is to alert on symptoms, not causes.

**Symptom-based vs cause-based alerting:**
```
CAUSE-BASED (bad):
  - "CPU > 80%"
  - "Memory > 90%"
  - "DB connections > 100"
  → Engineers get paged at 3am for things that may not affect users.

SYMPTOM-BASED (good):
  - "Error rate > 1%"
  - "p99 latency > 2 seconds"
  - "SLO burn rate is too fast"
  → Engineers only get paged when users are actually affected.
```

**Rules for high-quality alerts:**

1. **Every alert must be actionable.** If there's nothing to do, don't alert.
2. **Every alert must be urgent.** Non-urgent issues belong in dashboards or tickets.
3. **Alerts must have runbooks.** The alert annotation should link to what to do.
4. **Tune aggressively.** If an alert fires and always resolves itself, delete it.
5. **Use multi-window alerts.** Don't alert on 1-minute spikes; require sustained degradation.

```yaml
# Bad: Cause-based
- alert: HighCPU
  expr: node_cpu_usage > 0.8
  
# Good: Symptom-based with duration
- alert: HighErrorRate
  expr: |
    sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m])) > 0.01
  for: 5m   # Must be sustained, not a spike
  annotations:
    runbook: "https://wiki/runbooks/high-error-rate"
```

**Tiered alert routing:**
```
P1 (page immediately) → User-facing SLO breach
P2 (page in 30min)    → SLO burn rate fast-burn
P3 (Slack message)    → Anomaly, investigate next business day
P4 (ticket)           → Trend worth monitoring
```

---

### Q12. What are burn rate alerts? Explain fast burn vs slow burn.

**Answer:**

Burn rate alerts are the most sophisticated alerting strategy for SLO-based systems. Instead of alerting when an SLO is breached (too late), they alert when the error budget is being consumed at a rate that would exhaust it within a defined window.

**Burn rate formula:**
```
Burn rate = (actual error rate) / (acceptable error rate)

If SLO = 99.9%, acceptable error rate = 0.1%
If actual error rate = 1.0%, burn rate = 1.0% / 0.1% = 10x

At 10x burn rate, 28-day budget exhausted in:
28 days / 10 = 2.8 days
```

**Multi-window, multi-burn-rate approach (Google SRE recommendation):**

```
┌──────────────────┬───────────┬───────────┬─────────────────────────────┐
│ Severity         │ Burn Rate │ Windows   │ Budget consumed if not fixed │
├──────────────────┼───────────┼───────────┼─────────────────────────────┤
│ Critical (page)  │ 14.4x     │ 1h + 5m   │ 2% in 1 hour                │
│ High (page)      │ 6x        │ 6h + 30m  │ 5% in 6 hours               │
│ Medium (ticket)  │ 3x        │ 3d + 6h   │ 10% in 3 days               │
│ Low (ticket)     │ 1x        │ 28d       │ SLO just met                 │
└──────────────────┴───────────┴───────────┴─────────────────────────────┘
```

**Two-window requirement:** Use two windows (short + long) per alert to avoid false positives from short spikes while still catching real problems quickly.

```promql
# Fast burn: 14.4x in 1 hour (critical page)
(
  sum(rate(http_requests_total{status=~"5.."}[1h]))
  / sum(rate(http_requests_total[1h]))
) > (14.4 * 0.001)
AND
(
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))
) > (14.4 * 0.001)
```

**Why two windows?** The 1h window confirms the trend is real; the 5m window ensures it is still happening now (not a resolved blip).

---

### Q13. What is synthetic monitoring versus Real User Monitoring (RUM)?

**Answer:**

Both approaches measure user experience, but from different perspectives.

**Synthetic Monitoring (Active Monitoring):**
Simulates user interactions using scripted bots that run on a schedule from multiple geographic locations. It tests availability and performance even when no real users are present.

```
Synthetic probe (every 1 min from 5 regions)
    ↓
Simulates: Load homepage → Login → Search → Checkout
    ↓
Records: availability, response time, page load time
    ↓
Alert if any step fails or exceeds threshold
```

Tools: Pingdom, DataDog Synthetics, Grafana k6, AWS CloudWatch Canaries.

**Real User Monitoring (RUM):**
Collects performance data from actual users' browsers or mobile apps using injected JavaScript or SDK. Captures real-world conditions (slow networks, old devices, diverse geographies).

```javascript
// RUM snippet injected into HTML
// Captures: page load, Core Web Vitals, JS errors, API calls
<script src="rum-sdk.js" data-app-id="my-app"></script>
```

Tools: Google Analytics, Datadog RUM, New Relic Browser, Sentry.

**Comparison:**

| Dimension          | Synthetic                      | RUM                           |
|--------------------|-------------------------------|-------------------------------|
| Data availability  | Always (no users needed)       | Only when users visit         |
| Coverage           | Key user journeys only         | All real user interactions    |
| Consistency        | Controlled (same every time)   | Variable (real conditions)    |
| Root cause         | Easy to reproduce              | Hard to reproduce             |
| Geographic spread  | Controlled locations           | Real user locations           |
| New feature testing| Yes (before launch)            | No (must be live)             |

**Best practice:** Use both. Synthetic for proactive alerting and SLO measurement; RUM for understanding real user experience and identifying long-tail issues.

---

### Q14. How do you measure and improve MTTR (Mean Time To Recovery)?

**Answer:**

MTTR is the average time from when an incident starts to when normal service is restored. It is one of the four DORA metrics (alongside Deployment Frequency, Lead Time, Change Failure Rate).

**MTTR Components:**
```
Incident Timeline:
│
├── [Detection lag]    Alert fires → Engineer acknowledges
│
├── [Diagnosis lag]    Engineer acknowledged → Root cause found
│
├── [Mitigation lag]   Root cause found → Service restored
│
└── [Verification lag] Service restored → Confirmed healthy

MTTR = Detection + Diagnosis + Mitigation + Verification
```

**How to reduce each component:**

| Component    | Improvement                                                      |
|--------------|------------------------------------------------------------------|
| Detection    | Burn rate alerts, synthetic monitoring, lower alert thresholds   |
| Diagnosis    | Distributed tracing, correlated metrics+logs, runbooks          |
| Mitigation   | Feature flags (instant rollback), canary deployments, playbooks |
| Verification | SLO dashboards, automated smoke tests post-deploy               |

**Practical improvements:**
1. **Runbooks for every alert:** Pre-written diagnosis steps reduce diagnosis time from 30min to 5min.
2. **Feature flags:** Roll back a feature without a deployment (seconds vs minutes).
3. **Postmortems:** Each incident improves the system. Track action items.
4. **Chaos engineering:** Practice recovery before real incidents.

```
Measuring MTTR:
MTTR = Sum(time_to_recovery for all incidents) / count(incidents)

Target ranges:
  Elite  : < 1 hour
  High   : < 1 day  
  Medium : < 1 week
  Low    : > 1 week
```

**Avoid gaming MTTR:** Closing incidents before full verification to improve the metric is counterproductive. Track true recovery (users no longer affected).

---

### Q15. What is capacity planning methodology?

**Answer:**

Capacity planning ensures a system has sufficient resources to handle current and future load without over-provisioning (wasting money) or under-provisioning (causing outages).

**Step-by-step methodology:**

**Step 1: Baseline current utilization**
```
Collect metrics: CPU, memory, disk I/O, network bandwidth, DB connections
For each service, find p50/p95/p99 utilization
```

**Step 2: Model growth**
```
Options:
  - Linear extrapolation: traffic grows 10%/month → project 6 months
  - Business-driven: "Q4 campaigns will 3x traffic"
  - Seasonal: e-commerce peaks at holidays
```

**Step 3: Load test to find limits**
```
Run load tests (k6, Locust, JMeter) to find:
  - Saturation point: where latency degrades
  - Breaking point: where errors start
  - Recovery behavior: does it come back?
```

**Step 4: Calculate headroom**
```
Rule of thumb: never exceed 70% of capacity in steady state
                never exceed 50% of capacity for write-heavy systems

If breaking point = 10,000 RPS:
  Target operating ceiling = 7,000 RPS (70%)
  Alert threshold = 6,000 RPS (60%)
```

**Step 5: Plan for N+2 redundancy**
```
If 3 instances handle peak load:
  Run 5 instances (2 for fault tolerance + growth buffer)
```

**Step 6: Review regularly**
Capacity plans become stale. Review monthly, and always after major feature launches or architecture changes.

**Tools:** AWS Compute Optimizer, GCP Recommender, custom Grafana dashboards with trend lines, spreadsheet models for cost projection.

---

## Hard Questions (Q16–Q20)

---

### Q16. What are the USE and RED methods, and what metrics matter most for a web service?

**Answer:**

USE and RED are complementary mental models for selecting the most important metrics for any system. Together they cover both infrastructure and application layers.

**USE Method (Brendan Gregg) — for resources:**
- **Utilization:** What percentage of the resource's capacity is being used?
- **Saturation:** Is work being queued because the resource is full?
- **Errors:** Is the resource returning errors?

Apply USE to every physical and virtual resource: CPU, memory, disk, network, database connection pool, thread pool.

```
Resource          │ Utilization         │ Saturation           │ Errors
──────────────────┼─────────────────────┼──────────────────────┼─────────────────
CPU               │ cpu_usage %         │ cpu_run_queue_length │ cpu_throttle_count
Memory            │ mem_used/mem_total  │ swap_usage           │ OOM kills
Disk              │ disk_io_util %      │ disk_io_wait         │ disk_read_errors
DB Connection Pool│ active/pool_size    │ wait_queue_length    │ connection_refused
```

**RED Method (Tom Wilkie) — for services/microservices:**
- **Rate:** How many requests per second is the service handling?
- **Errors:** What fraction of requests are failing?
- **Duration:** How long are requests taking? (distribution, not average)

```promql
# Rate
rate(http_requests_total[5m])

# Error ratio
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Duration (p99)
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

**Golden Signals (Google SRE) — for user-facing services:**
1. Latency (distinguish successful vs error latency)
2. Traffic (rate of demand)
3. Errors (rate of failed requests)
4. Saturation (how "full" the service is)

**Recommended minimal dashboard for any web service:**
```
┌─────────────────────────────────────────────────────────┐
│  Request Rate (RPS)     │  Error Rate (%)                │
├─────────────────────────┼────────────────────────────────┤
│  p50 / p95 / p99 Latency│  SLO Error Budget Remaining    │
├─────────────────────────┼────────────────────────────────┤
│  CPU Utilization        │  Memory Utilization            │
├─────────────────────────┼────────────────────────────────┤
│  DB Connection Pool     │  Dependent Service Error Rates │
└─────────────────────────────────────────────────────────┘
```

These ~10 metrics cover 90% of incident diagnosis for a typical web service.

---

### Q17. What are incident severity levels, escalation paths, and how should they be structured?

**Answer:**

Incident severity levels provide a shared language for the urgency and impact of a system failure. They determine response time, escalation path, and communication requirements.

**Standard severity model (5-level):**

```
┌──────┬─────────────────────────────────────────────────────────────────────┐
│  SEV │ Definition                                                           │
├──────┼─────────────────────────────────────────────────────────────────────┤
│  SEV1│ Complete outage. All users affected. Revenue loss per minute.        │
│      │ Example: Checkout is down, authentication is broken.                 │
├──────┼─────────────────────────────────────────────────────────────────────┤
│  SEV2│ Major feature degraded. >20% of users affected. No workaround.      │
│      │ Example: Search returns wrong results, payment latency > 30s.       │
├──────┼─────────────────────────────────────────────────────────────────────┤
│  SEV3│ Partial degradation. Small % of users affected or workaround exists.│
│      │ Example: Slow image loading, single region affected.                │
├──────┼─────────────────────────────────────────────────────────────────────┤
│  SEV4│ Minor issue. Single user or edge case. No broad impact.             │
│      │ Example: Specific browser rendering bug.                            │
├──────┼─────────────────────────────────────────────────────────────────────┤
│  SEV5│ Cosmetic issue. No user impact. Fix in next sprint.                  │
└──────┴─────────────────────────────────────────────────────────────────────┘
```

**Escalation path and SLAs:**
```
SEV1: Page on-call immediately → Incident Commander engaged in 5min
      → VP Engineering notified in 15min → Status page updated in 15min
      → All-hands war room → Customer communication in 30min

SEV2: Page on-call → Acknowledge in 15min → IC engaged in 30min
      → Status page updated in 30min

SEV3: Slack alert → Acknowledge next business hour → Ticket created

SEV4/5: Ticket in backlog
```

**Roles during a SEV1/SEV2:**
- **Incident Commander (IC):** Coordinates response, controls communication, not debugging.
- **Technical Lead:** Diagnoses and drives mitigation.
- **Communications Lead:** Updates status page, customer messaging, executive summaries.
- **Scribe:** Documents timeline, decisions, and action items in real-time.

**Dos and Don'ts:**
- DO: Declare early, downgrade if needed. Early declaration enables resources.
- DON'T: Delay declaring because you think it will resolve itself.
- DO: Over-communicate to stakeholders; silence is worse than bad news.

---

### Q18. What is a blameless postmortem, and what is its structure?

**Answer:**

A blameless postmortem (also called a retrospective or incident review) is a structured analysis of an incident focused on systemic improvements rather than individual fault. The philosophy, pioneered by Google SRE, recognizes that humans made reasonable decisions given the information available at the time — failures are system problems, not people problems.

**Why blameless?**
Blame culture creates incentives to hide problems and avoid risk. Blameless culture creates incentives to surface problems and experiment. The goal is to prevent recurrence, not to assign punishment.

**Postmortem structure:**

```
─────────────────────────────────────────────────────────
INCIDENT POSTMORTEM: Checkout Service Outage
Date: 2024-01-15  |  Severity: SEV1  |  Duration: 47 min
─────────────────────────────────────────────────────────

1. SUMMARY (3–5 sentences)
   What happened, what was the impact, what resolved it.

2. IMPACT
   - Users affected: ~120,000
   - Revenue impact: ~$85,000
   - Duration: 14:23 – 15:10 UTC

3. TIMELINE (chronological, factual, no blame)
   14:23  Deploy of v2.4.1 to checkout-service completed
   14:25  Error rate begins climbing (not yet alerted)
   14:31  Burn rate alert fires, on-call paged
   14:38  On-call identifies deploy correlation
   14:45  Rollback initiated
   14:58  Error rate returns to baseline
   15:10  Incident closed after monitoring 12 min stable

4. ROOT CAUSE(S)
   Immediate: New DB query missing an index, causing full table scans
   Contributing: No load testing of new query against production data volume
   Contributing: Alerting threshold was too conservative (5 min delay)

5. WHAT WENT WELL
   - Burn rate alert fired within 8 minutes of degradation
   - Rollback process was fast (7 minutes)
   - Communication to stakeholders was timely

6. WHAT WENT POORLY
   - No performance testing of DB queries in CI/CD pipeline
   - No query execution plan review in code review checklist

7. ACTION ITEMS (each with owner and due date)
   ┌─────────────────────────────────────────────┬─────────┬──────────┐
   │ Action                                      │ Owner   │ Due Date │
   ├─────────────────────────────────────────────┼─────────┼──────────┤
   │ Add query EXPLAIN analysis to CI            │ @alice  │ Jan 22   │
   │ Add DB index for new query                  │ @bob    │ Jan 16   │
   │ Reduce burn rate alert detection window     │ @carol  │ Jan 19   │
   └─────────────────────────────────────────────┴─────────┴──────────┘
```

**5 Whys technique for root cause:**
```
Why did checkout fail?        → DB queries were timing out
Why were queries timing out?  → Missing index causing full table scan
Why was index missing?        → Not added in migration script
Why not in migration?         → No review checklist item for indexes
Why no checklist item?        → Process never accounted for query performance
Root cause: Lack of query performance gate in review/CI process
```

Track action item completion rate as a metric — postmortems without follow-through are theater.

---

### Q19. How do you implement distributed tracing in microservices end-to-end?

**Answer:**

Implementing distributed tracing requires instrumentation at every service boundary, a collection infrastructure, and a backend for storage and visualization.

**Architecture:**
```
Client → [API Gateway] → [Service A] → [Service B] → [Database]
                ↓              ↓              ↓
         OTel SDK        OTel SDK       OTel SDK
                ↓              ↓              ↓
              [OpenTelemetry Collector (sidecar or centralized)]
                              ↓
                       [Jaeger / Tempo / Datadog]
```

**Step-by-step implementation:**

**Step 1: Instrument each service with OTel SDK**
```python
# Python (FastAPI) — auto-instrumentation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Setup
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317"))
)
trace.set_tracer_provider(provider)

# Auto-instrument HTTP framework and outgoing calls
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()
```

**Step 2: Add custom spans for business logic**
```python
tracer = trace.get_tracer("order-service")

async def process_order(order_id: str):
    with tracer.start_as_current_span("process_order") as span:
        span.set_attribute("order.id", order_id)
        span.set_attribute("order.region", "us-east-1")
        
        result = await validate_inventory(order_id)
        if not result:
            span.set_status(trace.StatusCode.ERROR, "Inventory unavailable")
            span.record_exception(InventoryException(order_id))
```

**Step 3: Deploy OTel Collector**
```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
exporters:
  jaeger:
    endpoint: jaeger:14250
  prometheus:
    endpoint: 0.0.0.0:8889
service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
```

**Step 4: Correlate traces with logs**
```python
# Inject trace ID into log context
import logging
from opentelemetry import trace

class TraceIdFilter(logging.Filter):
    def filter(self, record):
        span = trace.get_current_span()
        ctx = span.get_span_context()
        record.trace_id = format(ctx.trace_id, '032x') if ctx.is_valid else 'no-trace'
        return True
```

**Step 5: Set sampling strategy**
```yaml
# Don't trace 100% of requests in production
sampler:
  type: probabilistic  # or tail_sampling for sampling on error
  param: 0.1           # 10% of traces
```

**Tail-based sampling** (sample 100% of error traces, 1% of success traces) is ideal for production — implemented in the OTel Collector.

---

### Q20. What are anomaly detection approaches for monitoring: static thresholds vs ML-based?

**Answer:**

Anomaly detection is the practice of automatically identifying deviations from normal system behavior. The right approach depends on the nature of the metric and the resources available.

**Static Threshold Alerting:**
The simplest approach — set a fixed threshold, alert when exceeded.

```yaml
# Prometheus static threshold
- alert: HighErrorRate
  expr: error_rate > 0.05  # Alert if error rate > 5%
```

Pros: Simple, predictable, easy to reason about.
Cons: Does not account for seasonal patterns. A 5% error rate may be normal during peak hours but catastrophic at 3am. Generates false positives (spikes) and false negatives (gradual drift).

**Dynamic/Seasonal Threshold:**
Compares current value against the same time in a previous period.

```promql
# Alert if current traffic is 50% below last week's same hour
(
  sum(rate(http_requests_total[10m]))
  /
  sum(rate(http_requests_total[10m] offset 1w))
) < 0.5
```

**ML-based Anomaly Detection approaches:**

| Technique              | How it Works                                         | Use Case                          |
|------------------------|------------------------------------------------------|-----------------------------------|
| Z-score / 3σ rule      | Flag values > 3 std deviations from rolling mean     | Simple, fast, works on normal distributions |
| EWMA (exponentially weighted moving average) | Weight recent data more heavily | Smooths noise, detects trend shifts |
| Facebook Prophet       | Decompose time-series into trend + seasonality + holidays | Weekly/daily patterns           |
| Isolation Forest       | Random trees isolate outliers faster than normal points | Multivariate, no labeled data   |
| LSTM / Transformer     | Deep learning predicts expected value; flag if actual differs | Complex seasonal patterns      |

**Practical architecture:**
```
Prometheus metrics
        ↓
  Feature engineering
  (rolling mean, stddev, day-of-week, hour-of-day)
        ↓
  Anomaly model (Prophet / Isolation Forest)
        ↓
  Anomaly score per metric per time window
        ↓
  Threshold on anomaly score → Alert
```

**Hybrid recommendation:**
1. Use static thresholds for known SLIs (error rate, latency) — predictable, fast.
2. Use burn rate alerts for SLO budget tracking.
3. Use ML-based anomaly detection for business metrics (orders/minute, signups/hour) where you don't know the threshold but you know what "normal" looks like.

**Tooling:** Grafana Machine Learning (based on Prophet), AWS CloudWatch Anomaly Detection, Datadog Watchdog, Elastic ML.

---

## Quick Reference

```
THREE PILLARS OF OBSERVABILITY
  Metrics  → Is something wrong? (Prometheus, Datadog)
  Logs     → Why did it go wrong? (ELK, Loki)
  Traces   → Where across services? (Jaeger, Zipkin, Tempo)

SLI / SLO / SLA
  SLI  = measurement (success rate = successful / total)
  SLO  = internal target (≥99.9% over 28 days)
  SLA  = external contract (≥99.5% or credits)
  Error Budget = 1 - SLO = 0.1% = ~40 minutes/28 days

BURN RATE ALERTS
  Critical (page)  : 14.4x burn, 1h + 5m windows
  High (page)      : 6x burn, 6h + 30m windows
  Medium (ticket)  : 3x burn, 3d + 6h windows

KUBERNETES PROBES
  Liveness  → fails = restart container
  Readiness → fails = remove from load balancer
  Startup   → disables others until app is ready

USE METHOD (Resources)
  Utilization / Saturation / Errors

RED METHOD (Services)
  Rate / Errors / Duration

MTTR COMPONENTS
  Detection → Diagnosis → Mitigation → Verification

PROMETHEUS SCRAPE
  Pull model: Prometheus polls /metrics endpoints
  Pushgateway: for short-lived jobs

DISTRIBUTED TRACING
  trace_id  → unique per request
  span_id   → unique per hop
  W3C header: traceparent

STRUCTURED LOGGING
  Always include: trace_id, service, level, timestamp, business_id

ALERT QUALITY RULES
  1. Actionable (something to do)
  2. Urgent (user-impacting)
  3. Has runbook
  4. Symptom-based, not cause-based
  5. Multi-window (not single spike)

POSTMORTEM SECTIONS
  Summary → Impact → Timeline → Root Cause →
  What Went Well → What Went Poorly → Action Items
```

---

*File 11 of 15 — Monitoring and Observability*
