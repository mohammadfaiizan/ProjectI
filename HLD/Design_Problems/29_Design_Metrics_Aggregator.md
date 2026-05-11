# Problem 29: Design a Metrics Aggregation System

---

## 1. Problem Statement & Clarifying Questions

### Problem Statement
Design a distributed metrics aggregation system capable of collecting, storing, aggregating, and querying 10 million metrics per second from thousands of microservices, with support for alerting, dashboards, and long-term retention with downsampling.

### Clarifying Questions
1. **Scale**: How many metrics/second? How many unique metric series? (10M metrics/sec, 10B unique series)
2. **Metric types**: Counters only, or also gauges, histograms, summaries? (All four types)
3. **Retention**: How long to keep data? At what resolution? (Raw 15s for 15 days, 1m for 60 days, 1h for 2 years)
4. **Query patterns**: Real-time dashboards vs batch analytics? (Both — low latency for dashboards)
5. **Alerting**: Threshold-based or anomaly-based? Both?
6. **Push or pull model**: Prometheus-style scraping or StatsD-style push? (Both)
7. **Cardinality**: How many label combinations per metric? (Can cause "cardinality explosion")
8. **Availability**: Is it acceptable to lose some metrics during outages? (Yes, but alert on gaps)
9. **Query language**: PromQL-like or custom? (PromQL subset)
10. **Cost constraints**: On-prem vs cloud? (Cloud-native, optimize for cost via compression)

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements
- Ingest metrics via push (StatsD/HTTP) and pull (Prometheus scrape) models
- Support metric types: Counter, Gauge, Histogram, Summary
- Store time-series data with labels/tags for multi-dimensional querying
- Aggregate metrics over time windows (1m, 5m, 1h, 1d)
- Downsample older data (1s raw → 1m rollup → 1h rollup)
- Query metrics with PromQL-like expressions (rate, sum, avg, histogram_quantile)
- Evaluate alert rules against metrics in real time; fire alert notifications
- Support Grafana-compatible query API for dashboards
- Enforce data retention policies with automatic cleanup

### Non-Functional Requirements
- **Ingestion throughput**: 10M metrics/sec = 10 billion data points/day
- **Query latency**: Dashboard queries < 1 second; alert evaluation < 10 seconds
- **Availability**: 99.9% for ingestion; 99.99% for alerting
- **Durability**: At-least-once delivery; tolerate < 0.01% data loss
- **Compression**: 90%+ compression ratio via Gorilla encoding (typical: 1.37 bytes/sample)
- **Scalability**: Horizontally scale ingestion and storage independently
- **Cardinality limit**: Warn/reject metric series > 10M unique label combinations

---

## 3. Capacity Estimation

### Ingestion
- 10M metrics/sec = 600M metrics/min = 864B metrics/day
- Each metric sample: (metric_name=50B, labels=100B, timestamp=8B, value=8B) = ~166 bytes raw
- Compressed (Gorilla): ~1.37 bytes/sample
- Raw storage/day: 864B × 1.37 bytes = ~1.2 TB/day
- After downsampling (keep 15 days raw + 60 days 1m + 2 years 1h):
  - Raw 15s: 15 days × 1.2 TB = 18 TB
  - 1m rollups: 60 days × 1.2 TB / 4 = 18 TB
  - 1h rollups: 730 days × 1.2 TB / 240 = 3.65 TB
  - Total: ~40 TB active storage

### Query Load
- 10K active dashboards × 20 queries/refresh × 1 refresh/30s = ~6,700 QPS
- Alert evaluation: 10K alert rules × every 30s = 330 evaluations/sec

### Nodes
- Ingestion: 10M samples/sec ÷ 500K samples/sec per node = 20 ingestion nodes
- Storage: 40 TB ÷ 2 TB per node = 20 storage nodes (with 3× replication = 60 nodes)

---

## 4. High-Level Architecture (ASCII Diagram)

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    METRICS SOURCES                                      │
 │   Microservices │ Kubernetes Pods │ Databases │ Load Balancers          │
 │   (Push: StatsD/HTTP)        (Pull: Prometheus scrape endpoint /metrics) │
 └───────┬───────────────────────────────────────────────┬────────────────┘
         │ Push (UDP/HTTP)                               │ Scrape (HTTP GET)
 ┌───────▼──────────────────┐             ┌─────────────▼────────────────────┐
 │   PUSH INGESTION GATEWAY  │             │   PROMETHEUS SCRAPER SERVICE      │
 │   StatsD UDP listener     │             │   Target discovery (K8s SD)       │
 │   HTTP metrics endpoint   │             │   robots.txt-like config          │
 │   Schema validation       │             │   Concurrent scrape workers       │
 │   Rate limiting           │             │   Scrape interval: 15s default    │
 └───────────┬──────────────┘             └────────────────┬─────────────────┘
             │                                             │
 ┌───────────▼─────────────────────────────────────────────▼───────────────────┐
 │                           KAFKA (INGESTION BUFFER)                           │
 │   Topic: raw-metrics    │    Partitioned by metric_name hash                 │
 │   Retention: 4 hours    │    1M messages/sec per partition (50 partitions)   │
 └────────────────────────────────────────────┬────────────────────────────────┘
                                              │
       ┌──────────────────────────────────────▼──────────────────────────────┐
       │                     FLINK STREAM PROCESSOR                           │
       │  Tumbling windows: 1m, 5m                                           │
       │  Sliding windows for rate calculation                               │
       │  Aggregations: sum, avg, min, max, count, percentiles               │
       │  Cardinality enforcement (reject > 10M unique series)               │
       └──────────┬────────────────────────────────┬───────────────────────┘
                  │ Raw samples                    │ Aggregated rollups
  ┌───────────────▼──────────────┐    ┌────────────▼──────────────────────┐
  │  TIME-SERIES DB (Raw)         │    │  TIME-SERIES DB (Rollups)          │
  │  InfluxDB / Prometheus TSDB   │    │  InfluxDB (downsampled)           │
  │  Retention: 15 days at 15s    │    │  5m: 60 days; 1h: 2 years        │
  │  Gorilla compression          │    │  Gorilla compression              │
  └──────────────────────────────┘    └───────────────────────────────────┘
                  │                                │
  ┌───────────────▼────────────────────────────────▼──────────────────────┐
  │                       QUERY ENGINE                                      │
  │   PromQL parser → query planner → parallel shard execution            │
  │   Results merge → downsampling-aware routing                          │
  └──────┬───────────────────────────────────────────────────────────────┘
         │                    │                           │
  ┌──────▼──────┐    ┌────────▼──────────┐    ┌──────────▼──────────────┐
  │  GRAFANA    │    │  ALERT ENGINE      │    │  ROLLUP ENGINE          │
  │  Dashboards │    │  Rule evaluation   │    │  1m → 5m → 1h → 1d     │
  │  /api/query │    │  PagerDuty/Slack   │    │  Scheduled downsampling │
  └─────────────┘    └───────────────────┘    └─────────────────────────┘
```

---

## 5. Component Deep-Dive

### 5.1 Metric Types
**Counter**: Monotonically increasing value (e.g., HTTP requests total)
- `http_requests_total{method="GET", status="200"} = 1523456`
- Query: `rate(http_requests_total[5m])` → requests/sec over last 5 min

**Gauge**: Arbitrary value that can go up or down (e.g., memory usage, temperature)
- `memory_usage_bytes{host="server-1"} = 8589934592`
- Query: `avg(memory_usage_bytes) by (host)`

**Histogram**: Sample observations in configurable buckets (e.g., request duration)
- `http_request_duration_seconds_bucket{le="0.1"} = 2345`
- `http_request_duration_seconds_bucket{le="0.5"} = 5678`
- Query: `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`

**Summary**: Similar to histogram but calculates quantiles client-side
- `http_request_duration_seconds{quantile="0.9"} = 0.42`

### 5.2 Time-Series Data Model
A time-series is uniquely identified by its metric name + label set:
```
{__name__="http_requests_total", job="api-server", instance="10.0.0.1:8080", method="GET", status="200"}
```
- **Series ID**: Hash of sorted label key-value pairs → 64-bit integer
- **Data point**: (series_id, timestamp_ms, float64_value) = 20 bytes raw
- **Chunk**: 120 data points per chunk (2 hours at 1-min scrape) = Gorilla-compressed block

### 5.3 Gorilla Encoding (Facebook's Time-Series Compression)
Delta-of-delta for timestamps:
```
t0 = 1700000000
t1 = 1700000060  → delta = 60
t2 = 1700000120  → delta = 60, delta-of-delta = 0  (store 0: 1 bit)
t3 = 1700000180  → delta = 60, delta-of-delta = 0  (store 0: 1 bit)
```
Most timestamps have delta-of-delta = 0 → encode with just 1 bit.

XOR encoding for values:
```
v0 = 12.0   (64-bit IEEE 754)
v1 = 12.5   → XOR(v0, v1) = leading zeros + meaningful bits + trailing zeros
v2 = 12.5   → XOR(v1, v2) = 0  (store 0: 1 bit — same value!)
```
Result: ~1.37 bytes/sample vs 16 bytes raw → **11.7× compression**.

### 5.4 Aggregation Windows
**Tumbling Window (non-overlapping):**
```
[00:00 - 01:00] → 1 aggregated value
[01:00 - 02:00] → 1 aggregated value
```
Used for: 1m, 5m, 1h rollups stored in time-series DB.

**Sliding Window (PromQL range vector):**
```
rate(metric[5m]) at t=T → use data points from [T-5m, T]
```
Used for: real-time rate calculations in queries and alerts.

### 5.5 Cardinality Explosion Problem
**Problem**: `http_requests_total{user_id="..."}` with 10M users → 10M unique series
- Each series = separate chunk in storage → 10M × 4 KB chunks = 40 GB per metric
- At 10M metrics × 10M series = 10^14 series → petabytes per metric

**Prevention Strategies:**
1. **High-cardinality label detection**: Alert when unique values for a label > 1000
2. **Label dropping**: Drop user_id, request_id labels before storage
3. **Cardinality limit**: Reject metric series when total exceeds 10M per metric name
4. **Aggregation at source**: Aggregate by user_id bucket (0-999) at instrumentation layer

### 5.6 Alert Engine
```python
# Alert rule example (PromQL)
alert: HighErrorRate
expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
for: 2m  # Alert must fire continuously for 2 minutes before notification
labels:
  severity: critical
annotations:
  summary: "Error rate > 5% for {{ $labels.job }}"
```

**Evaluation loop:**
1. Every 30 seconds, evaluate each alert rule's PromQL expression
2. If expression returns non-empty result → alert is "pending"
3. If pending for `for` duration → alert fires → notify (PagerDuty, Slack, email)
4. If expression returns empty → alert resolves → send resolution notification

### 5.7 Rollup (Downsampling) Engine
- Scheduled Flink job runs every 5 minutes
- Takes 1-minute aggregates → computes 5-minute aggregates
- Takes 5-minute aggregates → computes 1-hour aggregates (runs hourly)
- Aggregation functions stored per rollup: min, max, sum, count (to support any query)
- After rollup written: raw data beyond retention window deleted from storage

---

## 6. Database Design

### Time-Series Storage Schema (InfluxDB)
```
Measurement: http_requests_total
Tags (indexed): job, instance, method, status_code
Fields: value (float64)
Timestamp: epoch nanoseconds

Write: http_requests_total,job=api,instance=10.0.0.1,method=GET,status=200 value=1523456 1700000000000000000
```

### Alert Rules (PostgreSQL)
```sql
CREATE TABLE alert_rules (
    rule_id      UUID PRIMARY KEY,
    name         VARCHAR(256) NOT NULL,
    expr         TEXT NOT NULL,          -- PromQL expression
    for_duration INTERVAL DEFAULT '0s',
    labels       JSONB,
    annotations  JSONB,
    is_active    BOOLEAN DEFAULT TRUE,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE alert_history (
    id           BIGSERIAL PRIMARY KEY,
    rule_id      UUID REFERENCES alert_rules,
    state        VARCHAR(16) NOT NULL,   -- FIRING, RESOLVED, PENDING
    fired_at     TIMESTAMPTZ,
    resolved_at  TIMESTAMPTZ,
    labels       JSONB,
    value        FLOAT8
);
```

### Metric Metadata (Redis + PostgreSQL)
```
# Redis: fast lookup during ingestion
metric_meta:{metric_name} → HASH {
    type: "counter" | "gauge" | "histogram" | "summary",
    help: "description string",
    unit: "seconds" | "bytes" | "",
    cardinality: 15234,       -- current unique series count
    last_seen: timestamp
}

# PostgreSQL: durable storage + cardinality audit
CREATE TABLE metric_metadata (
    metric_name    VARCHAR(256) PRIMARY KEY,
    metric_type    VARCHAR(16),
    help_text      TEXT,
    unit           VARCHAR(64),
    max_cardinality BIGINT DEFAULT 10000000,
    created_at     TIMESTAMPTZ
);
```

---

## 7. API Design

### Write API (HTTP/gRPC)
```
POST /v1/write (Prometheus remote write format)
Content-Type: application/x-protobuf
Body: WriteRequest protobuf (timeseries[] with labels[] and samples[])

POST /v1/metrics (StatsD over HTTP)
Body: { "metrics": [
  { "name": "http_requests_total", "type": "counter",
    "value": 1, "tags": {"method": "GET", "status": "200"},
    "timestamp": 1700000000 }
]}
```

### Query API (PromQL compatible)
```
GET /api/v1/query?query=rate(http_requests_total[5m])&time=1700000000
Response: { "status": "success", "data": { "resultType": "vector", "result": [...] } }

GET /api/v1/query_range?query=rate(http_requests_total[5m])&start=1699900000&end=1700000000&step=60
Response: { "status": "success", "data": { "resultType": "matrix", "result": [...] } }
```

### Management API
```
GET  /v1/metrics                          # List all metric names
GET  /v1/metrics/{name}/metadata          # Metric type, cardinality, labels
POST /v1/alerts                           # Create alert rule
GET  /v1/alerts/active                    # Currently firing alerts
POST /v1/retention                        # Update retention policy
```

---

## 8. Scalability & Bottlenecks

### Bottleneck 1: Ingestion Bandwidth (10M samples/sec)
- Single Kafka broker handles ~1M messages/sec
- **Solution**: 50-partition Kafka topic; 10 ingestion gateway nodes; auto-scaling

### Bottleneck 2: Storage Write Amplification
- Gorilla chunks must be periodically flushed to persistent storage
- **Solution**: Write-ahead log in RAM, flush every 2 hours (Prometheus's approach); 3× replication

### Bottleneck 3: Query Fan-Out for Aggregation
- `sum(rate(metric[5m]))` across 10M series = querying all shards simultaneously
- **Solution**: Query planner routes to specific shards based on labels; partial aggregation at each shard

### Bottleneck 4: Alert Storm During Outage
- Major outage → thousands of alerts fire simultaneously → notification service overwhelmed
- **Solution**: Alert grouping (group by job/service); inhibition rules (suppress child alerts when parent fires); rate limiting on notification sending

### Bottleneck 5: Long-Term Storage Cost
- 2 years of 1h rollups: 3.65 TB — manageable
- 2 years of raw data would be: 1.2 TB/day × 730 = 876 TB — unacceptable
- **Solution**: Aggressive downsampling + TTL-based deletion; cold storage in S3 for compliance

---

## 9. Trade-offs & Design Decisions

### Decision 1: Push vs Pull Model
- **Push (StatsD)**: Zero config on server side; works behind NAT; high ingest load on server
- **Pull (Prometheus)**: Server controls scrape rate; natural backpressure; requires reachable endpoints
- **Choice**: Both — pull for Kubernetes services (discoverable), push for batch jobs and legacy systems

### Decision 2: InfluxDB vs Prometheus TSDB vs TimescaleDB
- **Prometheus TSDB**: Best compression, tight PromQL integration, limited scalability (single node)
- **InfluxDB**: Horizontal scaling, SQL-like query, higher memory usage
- **TimescaleDB**: PostgreSQL extension, SQL queries, mature ecosystem, good for complex queries
- **Choice**: Prometheus TSDB for hot data (15 days); InfluxDB for warm/cold rollups; Thanos/Cortex for horizontal scaling

### Decision 3: Kafka Buffer vs Direct Write to TSDB
- **Direct write**: Lower latency (< 1s); but TSDB can't sustain 10M writes/sec
- **Kafka buffer**: 2-5 seconds additional latency; but provides replay on TSDB failure, backpressure
- **Choice**: Kafka buffer — essential for reliability at 10M samples/sec scale

### Decision 4: Server-Side vs Client-Side Quantile Aggregation
- **Histogram (server-side)**: Raw bucket counts shipped → accurate aggregation across replicas
- **Summary (client-side)**: Quantile already computed — cannot re-aggregate across instances
- **Choice**: Histograms for distributed systems; summaries only for single-instance metrics

### Decision 5: Retention vs Cost
- **Keep raw data forever**: Perfect query fidelity; prohibitive storage cost
- **Aggressive downsampling**: Much lower cost; loses high-frequency detail for old data
- **Choice**: Tiered retention — raw 15 days (operational), 1m 60 days (recent history), 1h 2 years (trend analysis)

---

## 10. Key Interview Talking Points

### 1. Why Time-Series DBs Are Different From Relational DBs
Explain: sequential write-heavy workload (always append latest timestamp), no UPDATE/DELETE (immutable time-series), query patterns always involve time ranges, compression specifically designed for temporal data (delta encoding), automatic expiry/TTL essential.

### 2. Gorilla Encoding Deep Dive
Walk through the two components: delta-of-delta for timestamps (most scrapes are exactly 15 seconds apart → delta=15, delta-of-delta=0 → 1 bit encoding), XOR encoding for values (consecutive values often identical or differ only in low-order bits → many leading zeros in XOR → short code).

### 3. Cardinality Explosion War Story
Classic mistake: adding `user_id` label to HTTP metrics. 10M users × 1 metric = 10M series. Each series has its own chunk. Query `sum(http_requests_total)` must merge 10M series → OOM. Solution: aggregate by user tier, region, or quantile at instrumentation layer.

### 4. The Pull Model Advantage for Reliability
With push, if a service crashes, you lose metrics. With pull, if the service fails to respond to scrape, you know immediately (scrape failure = alert). Additionally, pull means the metrics server controls its own load; a misbehaving service can't flood it.

### 5. Horizontal Scaling with Thanos
Prometheus is inherently single-node for queries. Thanos adds: Store Gateway (queries cold data in S3), Query Frontend (fan-out across multiple Prometheus instances), Compactor (offline downsampling), Ruler (centralized alerting). This is how you scale to 10B series globally.

### 6. Alert Evaluation at Scale
10K alert rules each requiring a PromQL query every 30 seconds = 333 queries/sec just for alerts. At complex queries, this strains the query engine. Solution: dedicated alert evaluation engine (Prometheus Ruler) separate from dashboard query path; pre-compute common sub-expressions.

### 7. Metrics vs Logs vs Traces (Observability Pillars)
- **Metrics**: Aggregated numerical data; efficient at scale; loses individual request context
- **Logs**: Discrete events; expensive at scale; full context per event
- **Traces**: Request flow across services; distributed tracing (OpenTelemetry)
- **Correlation**: Metrics detect the problem, logs diagnose it, traces identify the root cause
