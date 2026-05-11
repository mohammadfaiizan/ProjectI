# 14. Monitoring and Observability

## Table of Contents
1. [Three Pillars of Observability](#1-three-pillars-of-observability)
2. [Metrics: Types and Data Models](#2-metrics-types-and-data-models)
3. [Prometheus Architecture](#3-prometheus-architecture)
4. [Grafana Dashboards](#4-grafana-dashboards)
5. [Log Aggregation](#5-log-aggregation)
6. [Log Shipping](#6-log-shipping)
7. [Distributed Tracing](#7-distributed-tracing)
8. [Tracing Systems: Jaeger vs Zipkin vs X-Ray](#8-tracing-systems-jaeger-vs-zipkin-vs-x-ray)
9. [OpenTelemetry](#9-opentelemetry)
10. [SLI, SLO, and SLA](#10-sli-slo-and-sla)
11. [Error Budgets and Burn Rate](#11-error-budgets-and-burn-rate)
12. [Alerting Principles](#12-alerting-principles)
13. [On-Call Design](#13-on-call-design)
14. [Health Checks](#14-health-checks)
15. [Synthetic Monitoring](#15-synthetic-monitoring)
16. [Anomaly Detection](#16-anomaly-detection)
17. [Capacity Planning](#17-capacity-planning)
18. [Incident Management](#18-incident-management)
19. [Performance Profiling](#19-performance-profiling)
20. [RUM vs Synthetic Monitoring](#20-rum-vs-synthetic-monitoring)
21. [Quick Reference](#21-quick-reference)

---

## 1. Three Pillars of Observability

Observability is the ability to understand the internal state of a system from its external outputs.

### The Three Pillars

```
┌──────────────┬──────────────────────────────────────────────────────────────┐
│  Pillar      │  What question does it answer?                               │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  Metrics     │  "What is happening?" — aggregated numerical measurements    │
│              │  over time. Is the system healthy at this moment?            │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  Logs        │  "Why did it happen?" — discrete events with context.        │
│              │  What exactly occurred, with what parameters?                │
├──────────────┼──────────────────────────────────────────────────────────────┤
│  Traces      │  "Where did it happen?" — request journey across services.   │
│              │  Which service/operation caused the latency or error?        │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

### Investigative Workflow

```
Alert fires (Metric threshold exceeded)
     │
     ▼
Check dashboard → latency p99 spiked on /checkout endpoint
     │
     ▼
Query logs → ERROR: "database connection timeout" at 14:32:05
     │
     ▼
Check traces → checkout-service → payment-service → db-proxy: 4.2s at db-proxy
     │
     ▼
Root cause: DB connection pool exhausted
```

### Monitoring vs Observability

| Monitoring | Observability |
|---|---|
| Checking known failure modes | Understanding unknown failure modes |
| Reactive — waiting for alerts | Proactive — exploring system behavior |
| "Is the system up?" | "Why is this request slow for this user segment?" |
| Predetermined dashboards | Ad-hoc exploration |
| Older concept | Broader engineering practice |

### The Fourth Pillar: Events / Profiling (Emerging)

Some argue for additional pillars:
- **Events**: Discrete business events (user signed up, payment processed)
- **Profiles**: Continuous CPU/memory sampling for performance analysis
- **Exceptions**: Error tracking (Sentry, Rollbar) for crash analysis

---

## 2. Metrics: Types and Data Models

### Four Core Metric Types

#### Counter
Monotonically increasing value that only goes up (or resets to zero on restart).

```
Use for: HTTP request count, errors, bytes sent, completed jobs

http_requests_total{method="GET", status="200", path="/api/orders"} 4827
http_requests_total{method="POST", status="500", path="/api/orders"} 12

# Useful operation: rate() — compute per-second rate over window
rate(http_requests_total[5m])  = requests per second over 5 minutes
```

#### Gauge
Can go up or down. Represents a current state/value.

```
Use for: CPU usage, memory, queue depth, active connections, temperature

process_resident_memory_bytes 45678912
http_connections_active 127
queue_depth{queue="email"} 1847
```

#### Histogram
Samples observations and counts them in configurable buckets. Also tracks sum and count.

```
Use for: Request latency, response size, payment amount

http_request_duration_seconds_bucket{le="0.005"} 1200  # requests under 5ms
http_request_duration_seconds_bucket{le="0.01"}  2100
http_request_duration_seconds_bucket{le="0.025"} 3800
http_request_duration_seconds_bucket{le="0.05"}  4200
http_request_duration_seconds_bucket{le="+Inf"}  4500
http_request_duration_seconds_sum 450.23          # Total seconds
http_request_duration_seconds_count 4500           # Total requests

# Quantiles approximated: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

#### Summary
Like histogram but calculates quantiles on the client side. Not aggregatable across instances.

```
Use for: Pre-calculated quantiles when you can't reaggregate (avoid in distributed systems)

rpc_duration_seconds{quantile="0.5"}  0.01
rpc_duration_seconds{quantile="0.9"}  0.05
rpc_duration_seconds{quantile="0.99"} 0.1
rpc_duration_seconds_sum 150.23
rpc_duration_seconds_count 3000
```

### Histogram vs Summary

| Aspect | Histogram | Summary |
|---|---|---|
| Quantile calc | Server-side (PromQL) | Client-side |
| Aggregation across replicas | Yes | No |
| Accuracy | Approximate (bucket-based) | Exact (for configured quantiles) |
| Configuration | Bucket boundaries upfront | Quantile objectives upfront |
| Preferred for Prometheus | Yes | Limited use cases |

### Prometheus Data Model

Every metric is identified by:
```
<metric_name>{label1="value1", label2="value2"} <value> [timestamp]

http_requests_total{job="api-server", instance="10.0.0.1:8080", method="GET", status="200"} 4827 1700000000
```

**Label cardinality warning**: Each unique label combination creates a new time series. Never use high-cardinality values as labels (user IDs, email addresses, request IDs — this causes "cardinality explosion").

```
# BAD — creates millions of time series
http_requests_total{user_id="12345678"} 1

# GOOD — aggregate
http_requests_total{user_tier="premium"} 45000
```

### USE Method (for resources)

For every resource (CPU, disk, network), track:
- **U**tilization: Percentage of time resource is busy
- **S**aturation: Queue depth / wait time / work backed up
- **E**rrors: Error events per second

### RED Method (for services)

For every service endpoint, track:
- **R**ate: Requests per second
- **E**rrors: Error rate (5xx responses)
- **D**uration: Response time distribution (p50, p95, p99)

---

## 3. Prometheus Architecture

### Components

```
┌────────────────────────────────────────────────────────────────┐
│                        Prometheus Ecosystem                     │
│                                                                  │
│  ┌──────────────┐    ┌───────────────────────────────────────┐  │
│  │  Exporters   │    │           Prometheus Server            │  │
│  │              │    │  ┌────────────┐  ┌─────────────────┐  │  │
│  │  node_exp.   │◄───│  │  Retrieval │  │  TSDB           │  │  │
│  │  mysqld_exp. │    │  │  (scraper) │  │  (Time Series   │  │  │
│  │  redis_exp.  │    │  └────────────┘  │   Database)     │  │  │
│  │  kube_sm     │    │  ┌────────────┐  └─────────────────┘  │  │
│  └──────────────┘    │  │ Rules Eval │                        │  │
│                      │  └────────────┘                        │  │
│  ┌──────────────┐    └──────────────┬───────────────────────┘  │
│  │  Push        │                   │ query                     │
│  │  Gateway     │◄── short-lived    │                           │
│  │  (batch jobs)│    jobs push here ▼                           │
│  └──────────────┘    ┌─────────────────┐  ┌────────────────┐   │
│                       │   Alertmanager   │  │    Grafana     │   │
│                       │  - deduplicate   │  │  - dashboards  │   │
│                       │  - route         │  │  - alerting    │   │
│                       │  - silence       │  │                │   │
│                       └────────┬────────┘  └────────────────┘   │
│                                │                                  │
│                       ┌────────▼────────┐                        │
│                       │  PagerDuty /    │                        │
│                       │  Slack / Email  │                        │
│                       └─────────────────┘                        │
└────────────────────────────────────────────────────────────────┘
```

### Scraping Model

```yaml
# prometheus.yml
global:
  scrape_interval: 15s      # How often to scrape targets
  evaluation_interval: 15s  # How often to evaluate rules

scrape_configs:
  - job_name: 'api-servers'
    static_configs:
      - targets: ['api-1:8080', 'api-2:8080']
    metrics_path: /metrics
    scheme: https
    tls_config:
      ca_file: /etc/ssl/ca.crt

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### PromQL Basics

```promql
# Simple query — current value
http_requests_total

# Filter by label
http_requests_total{status="500", job="api"}

# Rate of increase (per second) over 5-minute window
rate(http_requests_total[5m])

# Error rate percentage
100 * sum(rate(http_requests_total{status=~"5.."}[5m])) 
    / sum(rate(http_requests_total[5m]))

# p99 latency
histogram_quantile(0.99, 
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))

# CPU usage by pod
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Alert: high error rate
(sum(rate(http_requests_total{status=~"5.."}[5m])) / 
 sum(rate(http_requests_total[5m]))) > 0.05

# Alert: latency p99 > 1 second
histogram_quantile(0.99, 
  rate(http_request_duration_seconds_bucket[5m])) > 1
```

### Alert Rules

```yaml
# alerting_rules.yml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          (sum(rate(http_requests_total{status=~"5.."}[5m])) /
           sum(rate(http_requests_total[5m]))) > 0.05
        for: 5m           # Must be true for 5 minutes before firing
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: "Error rate is {{ $value | humanizePercentage }}"
          runbook: "https://wiki.example.com/runbooks/high-error-rate"
```

### Retention and Storage

- Default retention: 15 days (configure with `--storage.tsdb.retention.time`)
- Remote write to long-term storage: Thanos, Cortex, Mimir, Victoria Metrics
- Thanos: sidecar pattern to upload to object storage (S3), global query view

---

## 4. Grafana Dashboards

### Panel Types

| Panel Type | Use Case |
|---|---|
| Time series | Metric trends over time, CPU/latency graphs |
| Stat | Single value with color threshold (current error rate) |
| Gauge | Dial showing value relative to min/max (disk usage %) |
| Bar chart | Comparison across categories |
| Table | Multi-column metric data |
| Heatmap | Histogram/latency distribution over time |
| Logs | Display logs inline with metrics |
| Traces | Display traces inline (Tempo integration) |
| Geomap | Geographic visualization |

### Dashboard Variables

```
# Template variable for environment selection
Variable: environment
Type: Query
Query: label_values(http_requests_total, environment)
Refresh: On Dashboard Load

# Use in panel query
http_requests_total{environment="$environment", service="$service"}
```

### Golden Signals Dashboard Template

```
Row 1: Traffic
  - Requests per second (rate)
  - Request breakdown by status code (stacked bar)

Row 2: Errors
  - Error rate percentage (stat with thresholds: green<1%, yellow<5%, red>5%)
  - Error count by type (pie chart)

Row 3: Latency
  - p50, p95, p99 latency time series
  - Latency heatmap

Row 4: Saturation
  - CPU utilization by pod
  - Memory usage
  - Active connections / queue depth
```

### Alerting in Grafana

```yaml
# Grafana alert rule
apiVersion: 1
groups:
  - name: Application Alerts
    rules:
      - uid: alert_high_latency
        title: High p99 Latency
        condition: B
        data:
          - refId: A
            queryType: range
            expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
          - refId: B
            type: threshold
            expression: A
            conditions:
              - evaluator:
                  params: [1.0]
                  type: gt
```

---

## 5. Log Aggregation

### Structured Logging (JSON)

```python
import structlog, uuid

logger = structlog.get_logger()

# Instead of:
logging.info(f"User {user_id} processed order {order_id} in {duration}ms")

# Use structured logging:
logger.info(
    "order.processed",
    user_id=user_id,
    order_id=order_id,
    duration_ms=duration,
    items_count=len(items),
    total_amount=total,
    trace_id=request.headers.get("X-Trace-Id"),
    span_id=str(uuid.uuid4())
)
```

Output:
```json
{
  "event": "order.processed",
  "level": "info",
  "timestamp": "2024-01-15T10:30:00.123Z",
  "service": "order-service",
  "user_id": 42,
  "order_id": "ord_abc123",
  "duration_ms": 245,
  "items_count": 3,
  "total_amount": 149.99,
  "trace_id": "abc123def456",
  "span_id": "789xyz"
}
```

### Log Levels

| Level | When to Use | Example |
|---|---|---|
| TRACE | Extremely detailed, usually disabled | Function entry/exit |
| DEBUG | Development debugging | SQL queries, serialized objects |
| INFO | Normal operational events | "Request processed", "User logged in" |
| WARN | Unexpected but handled situations | "Retry attempt 2/3", "Cache miss" |
| ERROR | Errors that need investigation | "Database connection failed" |
| FATAL/CRITICAL | System cannot continue | "Unable to bind to port", app shutdown |

**Production rule**: Set level to INFO or WARN. DEBUG in production causes log storms and cost spikes.

### ELK Stack

```
┌──────────────┐     ┌───────────┐     ┌─────────────────┐     ┌──────────┐
│  Application │────►│  Logstash │────►│  Elasticsearch  │────►│  Kibana  │
│  Servers     │     │ (parse,   │     │  (store, index, │     │ (search, │
│              │     │  filter,  │     │   search)       │     │  viz,    │
│              │     │  enrich)  │     │                 │     │  alert)  │
└──────────────┘     └───────────┘     └─────────────────┘     └──────────┘
```

**Logstash pipeline**:
```ruby
input {
  beats { port => 5044 }
}

filter {
  json { source => "message" }
  
  date {
    match => ["timestamp", "ISO8601"]
    target => "@timestamp"
  }
  
  mutate {
    remove_field => ["message", "host"]
    add_field => { "environment" => "${ENVIRONMENT}" }
  }
  
  # Geolocate IPs
  geoip { source => "ip_address" }
}

output {
  elasticsearch {
    hosts => ["es-1:9200", "es-2:9200"]
    index => "logs-%{[service]}-%{+YYYY.MM.dd}"
  }
}
```

**Elasticsearch index strategy**:
```
Index per day:  logs-api-2024.01.15  (easy time-based deletion with ILM)
Index per service per day: logs-order-service-2024.01.15
```

Index Lifecycle Management (ILM):
- **Hot**: Active writes + reads (SSD nodes)
- **Warm**: Read-only, less frequent access (HDD nodes)
- **Cold**: Rare access, no replicas (object storage)
- **Delete**: After retention period

---

## 6. Log Shipping

### Fluentd

Full-featured log collector with rich plugin ecosystem.

```xml
<!-- fluent.conf -->
<source>
  @type tail
  path /var/log/app/*.log
  pos_file /var/log/fluentd/app.log.pos
  tag app.logs
  <parse>
    @type json
  </parse>
</source>

<filter app.logs>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    environment "#{ENV['ENVIRONMENT']}"
  </record>
</filter>

<match app.logs>
  @type elasticsearch
  host elasticsearch.logging.svc
  port 9200
  index_name logs-${tag}-%Y.%m.%d
  <buffer tag,time>
    @type file
    path /var/log/fluentd/buffer
    flush_interval 5s
    retry_max_times 10
  </buffer>
</match>
```

### Fluent Bit

Lightweight version of Fluentd. Written in C. Ideal for Kubernetes sidecars and IoT/edge.

```ini
[INPUT]
    Name              tail
    Path              /var/log/containers/*.log
    Parser            docker
    Tag               kube.*
    Refresh_Interval  5

[FILTER]
    Name         kubernetes
    Match        kube.*
    Kube_URL     https://kubernetes.default.svc
    Merge_Log    On

[OUTPUT]
    Name         es
    Match        *
    Host         elasticsearch
    Port         9200
    Index        logs
    Type         _doc
```

### Vector

Modern, high-performance log shipper (Rust). Can replace both Fluentd and Logstash.

```toml
# vector.toml
[sources.app_logs]
type = "file"
includes = ["/var/log/app/*.log"]

[transforms.parse_json]
type = "remap"
inputs = ["app_logs"]
source = '''
. = parse_json!(string!(.message))
.environment = get_env_var!("ENVIRONMENT")
'''

[sinks.elasticsearch]
type = "elasticsearch"
inputs = ["parse_json"]
endpoint = "http://elasticsearch:9200"
index = "logs-%Y.%m.%d"
```

### Log Shipper Comparison

| Aspect | Logstash | Fluentd | Fluent Bit | Vector |
|---|---|---|---|---|
| Language | JRuby/Java | Ruby/C | C | Rust |
| Memory footprint | High (~500MB) | Medium (~60MB) | Very low (~1MB) | Low (~10MB) |
| Throughput | High | High | Very high | Highest |
| Plugin ecosystem | Rich | Rich | Moderate | Growing |
| Best for | Heavy transformation | General purpose | Edge/K8s sidecar | High-perf, multi-use |

---

## 7. Distributed Tracing

### Core Concepts

```
Trace: Complete journey of a single request through the system
  Trace ID: b7ad6b7169203331  (globally unique, propagated across services)

Span: A single unit of work within a trace
  Span ID: 6f4b6612415de578
  Parent Span ID: null (for root span) or parent's span ID

Baggage: Key-value pairs propagated in-band with the request
  (e.g., user tier, feature flags, A/B test cohort)
```

### Trace Visualization

```
Trace: user-checkout-request (total: 850ms)
│
├── [0ms - 850ms] checkout-service.HandleCheckout (850ms)
│   │
│   ├── [10ms - 250ms] payment-service.ProcessPayment (240ms)
│   │   ├── [15ms - 100ms] fraud-service.CheckFraud (85ms)
│   │   └── [110ms - 240ms] stripe-api.ChargeCard (130ms) ← SLOW
│   │
│   ├── [260ms - 450ms] inventory-service.ReserveItems (190ms)
│   │   └── [265ms - 440ms] db.SELECT items (175ms) ← SLOW
│   │
│   └── [460ms - 850ms] notification-service.SendEmail (390ms) ← ASYNC
```

### W3C TraceContext Standard

HTTP headers for trace propagation:
```http
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             ^  ^                               ^                ^ ^
             |  trace-id (16 bytes hex)         span-id (8 bytes) | sampled flag
             version                                              flags

tracestate: vendor1=opaqueValue1,vendor2=opaqueValue2
```

### Sampling Strategies

```
Head-based sampling (decision at start):
  - Sample 1% of all requests
  - Simple but may miss rare errors

Tail-based sampling (decision at end, Jaeger collector):
  - Always sample errors
  - Always sample slow requests (p99+)
  - Sample 1% of healthy fast requests
  - Requires buffering entire trace before decision

Priority sampling:
  - Force sample specific requests (e.g., user IDs for investigation)
  - X-B3-Flags: 1  (force trace)
```

---

## 8. Tracing Systems: Jaeger vs Zipkin vs X-Ray

### Jaeger (CNCF)

```
Architecture:
  App ──(UDP/HTTP)──► Jaeger Agent ──► Jaeger Collector ──► Storage
                      (per-host)       (validates, indexes)   (Cassandra/ES)
                                                              ┌──────────┐
  Query API ◄─────────────────────────────────────────────── │   UI     │
```

- Open source, CNCF graduated
- Supports Jaeger native protocol + OpenTelemetry
- Adaptive sampling built-in
- Storage: Cassandra (high scale), Elasticsearch, Badger (single node)
- Good Kubernetes integration

### Zipkin

```
Architecture:
  App ──(HTTP/Kafka)──► Zipkin Server ──► Storage (MySQL/ES/Cassandra)
                        (all-in-one or    │
                         collector +      └► Zipkin UI
                         storage)
```

- Older, simpler architecture
- B3 propagation format (also supported by Jaeger)
- Less active development than Jaeger
- Good for simpler setups

### AWS X-Ray

```
Architecture:
  App ──► X-Ray SDK ──► X-Ray Daemon ──► X-Ray API ──► Console
          (instruments   (local UDP        (AWS service)  / CloudWatch
           requests)      aggregator)
```

- Native AWS integration (Lambda, ECS, EC2 auto-instrumented)
- Service map automatically generated
- Sampling rules configurable via console
- Limited to AWS ecosystem
- Good default for AWS-native applications

### Comparison

| Aspect | Jaeger | Zipkin | AWS X-Ray |
|---|---|---|---|
| Open source | Yes | Yes | No |
| OTel support | Yes (native) | Yes | Partial |
| Deployment | K8s-friendly | Simple | Managed |
| Storage options | Multiple | Multiple | AWS-managed |
| Sampling | Adaptive | Configurable | Rule-based |
| AWS native | Manual | Manual | Yes |
| Cost | Infra cost | Infra cost | Pay per trace |

---

## 9. OpenTelemetry

### Why OpenTelemetry?

Before OTel, each vendor had its own SDK:
```
Datadog SDK    → only sends to Datadog
Jaeger SDK     → only sends to Jaeger
Zipkin SDK     → only sends to Zipkin
NewRelic SDK   → only sends to NewRelic
```

OpenTelemetry = vendor-neutral instrumentation. Write once, send anywhere.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Your Application                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              OpenTelemetry SDK                        │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │   │
│  │  │  Tracer  │  │  Meter   │  │  Logger (OTel Logs) │ │   │
│  │  │  Provider│  │  Provider│  │  Provider          │ │   │
│  │  └──────────┘  └──────────┘  └────────────────────┘ │   │
│  │              Exporter (OTLP)                          │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────┘
                          │ OTLP (gRPC or HTTP)
                          ▼
               ┌──────────────────────┐
               │  OTel Collector      │
               │  - receive           │
               │  - process (sample,  │
               │    filter, enrich)   │
               │  - export            │
               └──────────┬───────────┘
                          │
              ┌───────────┼──────────┐
              ▼           ▼          ▼
           Jaeger      Datadog    Prometheus
           (traces)    (all)      (metrics)
```

### Instrumentation Example (Python)

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup
provider = TracerProvider()
exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317")
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

# Instrumentation
def process_order(order_id: str):
    with tracer.start_as_current_span("process_order") as span:
        span.set_attribute("order.id", order_id)
        span.set_attribute("service.name", "order-service")
        
        try:
            result = charge_payment(order_id)
            span.set_attribute("payment.status", "success")
            return result
        except PaymentError as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            raise
```

### Auto-Instrumentation

```bash
# Zero-code instrumentation for Python
pip install opentelemetry-distro opentelemetry-exporter-otlp
opentelemetry-bootstrap -a install

OTEL_SERVICE_NAME=order-service \
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
opentelemetry-instrument python app.py
```

Auto-instrumented frameworks: Flask, Django, FastAPI, SQLAlchemy, requests, Redis, gRPC, Celery.

---

## 10. SLI, SLO, and SLA

### Definitions

**SLI (Service Level Indicator)**
A carefully defined quantitative measure of some aspect of the level of service being provided.

```
Common SLIs:
  Availability:  (successful requests) / (total requests)
  Latency:       proportion of requests served within threshold
  Throughput:    requests per second
  Error rate:    errors / total requests
  Freshness:     how recently data was updated (for data pipelines)
  Durability:    probability of data not being lost (storage systems)
```

**SLO (Service Level Objective)**
A target value or range of values for an SLI.

```
Examples:
  99.9% of requests return HTTP 200 in 30 days
  p99 latency < 200ms for 95% of 5-minute windows
  Availability >= 99.95% per quarter
  Error rate < 0.1%
```

**SLA (Service Level Agreement)**
A legal contract between provider and customer, specifying SLOs and consequences.

```
SLA: "99.9% monthly availability. If we drop below:
  99.5% → 10% service credit
  99.0% → 25% service credit
  95.0% → 100% service credit"
```

### Relationship

```
SLA (external, legal)
 └── SLO (internal target, stricter than SLA)
       └── SLI (measurement)
```

**Best practice**: Set internal SLO stricter than external SLA.

If SLA = 99.9%, set internal SLO = 99.95% so you have buffer before breaching legal commitments.

### Availability Nines

| Availability | Downtime per Year | Downtime per Month | Downtime per Week |
|---|---|---|---|
| 99% (2 nines) | 3.65 days | 7.3 hours | 1.7 hours |
| 99.9% (3 nines) | 8.77 hours | 43.8 minutes | 10.1 minutes |
| 99.95% | 4.38 hours | 21.9 minutes | 5 minutes |
| 99.99% (4 nines) | 52.6 minutes | 4.4 minutes | 1 minute |
| 99.999% (5 nines) | 5.26 minutes | 26 seconds | 6 seconds |

### Good SLI Selection

```
BAD SLIs:
  CPU utilization > 80%  → This is a cause, not a symptom
  Memory usage > 70%     → Doesn't directly measure user experience

GOOD SLIs:
  Request success rate   → Directly measures user experience
  Request latency        → Directly measures user experience
  Data freshness         → Measures if users get recent data
```

---

## 11. Error Budgets and Burn Rate

### Error Budget

```
If SLO = 99.9% availability over 30 days:
  Error budget = 1 - 0.999 = 0.001 = 0.1%
  
  30 days = 43,200 minutes
  Error budget = 43,200 × 0.001 = 43.2 minutes of downtime allowed

If you deploy 10x per month and each deployment causes 1 min downtime:
  Deployments consume: 10 × 1 = 10 minutes
  Remaining budget: 43.2 - 10 = 33.2 minutes

When budget is exhausted:
  → Stop feature deployments until next period
  → Focus engineering on reliability
```

### Burn Rate

Burn rate = how fast you're consuming the error budget relative to normal.

```
Burn rate of 1 = consuming exactly at SLO rate
  (will exactly exhaust budget by end of window)

Burn rate of 2 = consuming 2x as fast
  (will exhaust budget halfway through window)

Burn rate of 14.4 = consuming 14.4x as fast
  (will exhaust daily budget in 1 hour)
```

### Multi-Window Multi-Burn-Rate Alerts (Google SRE Book)

```
Fast burn (page immediately):
  Short window: 5% budget consumed in 1 hour
  → Burn rate > 14.4x
  → Alert within 2 minutes

Slow burn (ticket + potential page):
  Short window: 10% budget consumed in 6 hours
  → Burn rate > 6x for 30 minutes

Slow drip (ticket):
  10% budget consumed in 3 days
  → Burn rate > 1x sustained
```

```promql
# Burn rate calculation (30-day window)
# SLO: 99.9% availability

# 1-hour burn rate
(
  1 - 
  sum(rate(http_requests_total{status!~"5.."}[1h])) /
  sum(rate(http_requests_total[1h]))
) / (1 - 0.999)

# Alert: fast burn (1h window, >14.4x burn rate)
(
  (1 - sum(rate(http_requests_total{status!~"5.."}[1h])) /
       sum(rate(http_requests_total[1h])))
  /
  (1 - 0.999)
) > 14.4

# Also check 5-minute window to reduce false positives
AND
(
  (1 - sum(rate(http_requests_total{status!~"5.."}[5m])) /
       sum(rate(http_requests_total[5m])))
  /
  (1 - 0.999)
) > 14.4
```

### Burn Rate Alert Table

| Alert | Window | Burn Rate | Budget Consumed | Response |
|---|---|---|---|---|
| Page immediately | 1h + 5m | 14.4x | 2% in 1h | Incident response |
| Page immediately | 6h + 30m | 6x | 5% in 6h | Incident response |
| Ticket | 1d + 2h | 3x | 10% in 3d | Business hours |
| Ticket | 3d + 6h | 1x | 10% in 30d | Long-term fix |

---

## 12. Alerting Principles

### Alert on Symptoms, Not Causes

```
BAD alert (cause): CPU > 80%
  → High CPU might not affect users
  → Some workloads are expected to have high CPU

GOOD alert (symptom): Error rate > 1%
  → Users are definitely experiencing errors
  → Always actionable

BAD: Redis memory > 70%
GOOD: Cache miss rate spike (symptom of full cache)

BAD: Database connection pool at 90%
GOOD: p99 latency > 2s (users experiencing slowness)
```

### Alert Fatigue

Causes:
- Too many alerts with no actionable response
- Alerts that fire too frequently (noisy)
- Alerts that auto-resolve without investigation

Solutions:
- Alerts must be actionable: either page someone or file a ticket
- If you see an alert > 3x in a week without action: fix or delete it
- `for` clause in Prometheus: require condition holds for N minutes
- Alert deduplication in Alertmanager
- Grouping related alerts into single notification

### Alertmanager Configuration

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/...'

route:
  group_by: ['alertname', 'service']
  group_wait: 30s          # Wait before sending first notification
  group_interval: 5m       # Wait before sending updates
  repeat_interval: 4h      # Resend if still firing after this long
  receiver: 'team-slack'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true         # Also send to default receiver
    - match:
        team: backend
      receiver: 'backend-slack'

receivers:
  - name: 'pagerduty'
    pagerduty_configs:
      - routing_key: '<PD routing key>'
  - name: 'team-slack'
    slack_configs:
      - channel: '#alerts'
        text: '{{ .CommonAnnotations.description }}'

inhibit_rules:
  # Suppress warnings if critical alert fires for same service
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['service', 'alertname']
```

### Runbooks

Every alert must have a runbook link:
```markdown
## Alert: HighErrorRate

### Trigger Condition
Error rate > 5% for 5 minutes

### Investigation Steps
1. Check Grafana dashboard: [link]
2. Query recent errors: `kubectl logs -l app=api --since=10m | grep ERROR`
3. Check recent deployments: [Argo CD link]
4. Check downstream service health: payment-service, inventory-service

### Common Causes and Mitigations
- New deployment with bug → rollback: `kubectl rollout undo deployment/api`
- Database connection exhaustion → check connection pool metrics
- Downstream service outage → check dependency health dashboard

### Escalation
If unresolved in 30 minutes → escalate to on-call lead
```

---

## 13. On-Call Design

### Escalation Policies

```
Level 1 (0-15 min): Primary on-call engineer
Level 2 (15-30 min): Secondary on-call engineer  
Level 3 (30-60 min): On-call manager / team lead
Level 4 (60+ min): Director / VP of Engineering
```

### Rotation Schedules

```
Weekly rotation:
  Mon → Sun: Engineer A
  Mon → Sun: Engineer B (next week)
  
Follow-the-sun:
  APAC business hours: APAC team
  EU business hours: EU team
  US business hours: US team
  
Advantage: No one paged at 3am
Requirement: Sufficient team size in each timezone
```

### On-Call Hygiene

- **Working hours ratio**: On-call burden should not exceed 25% of engineering time
- **Compensation**: On-call pay or comp time for nights/weekends
- **Handoff notes**: Write brief summary before rotation ends
- **Alert review**: Weekly meeting to review alert quality
- **Toil reduction**: If you're solving the same incident repeatedly → automate the fix

### PagerDuty vs OpsGenie

| Aspect | PagerDuty | OpsGenie |
|---|---|---|
| Scheduling | Advanced | Advanced |
| Escalation policies | Yes | Yes |
| Integrations | 700+ | 200+ |
| Status pages | Yes (add-on) | Yes |
| Price | Higher | Lower |
| Mobile app | Yes | Yes |
| On-call analytics | Advanced | Basic |

---

## 14. Health Checks

### Kubernetes Probe Types

#### Liveness Probe
"Is the process alive and not stuck?"
- If fails: Kubernetes kills and restarts the container

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 30    # Wait 30s after container start
  periodSeconds: 10          # Check every 10s
  timeoutSeconds: 5          # Fail if no response in 5s
  failureThreshold: 3        # Restart after 3 consecutive failures
```

Liveness endpoint should be simple:
```python
@app.get("/health/live")
def liveness():
    # Just verify process is running and not deadlocked
    return {"status": "alive"}
```

#### Readiness Probe
"Is the process ready to serve traffic?"
- If fails: Kubernetes removes pod from load balancer (no traffic sent)
- Does NOT restart the container

```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  periodSeconds: 5
  failureThreshold: 3
```

Readiness endpoint checks dependencies:
```python
@app.get("/health/ready")
async def readiness():
    checks = {}
    
    try:
        await db.execute("SELECT 1")
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
    
    try:
        await redis.ping()
        checks["cache"] = "ok"
    except Exception:
        checks["cache"] = "error"
    
    all_healthy = all(v == "ok" for v in checks.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse({"status": checks}, status_code=status_code)
```

#### Startup Probe
"Has the container finished starting up?"
- Runs before liveness/readiness
- Useful for slow-starting apps

```yaml
startupProbe:
  httpGet:
    path: /health/live
    port: 8080
  failureThreshold: 30     # 30 × 10s = 5 minutes to start
  periodSeconds: 10
```

### Health Check Patterns

```json
// Detailed health response
{
  "status": "degraded",
  "version": "1.2.3",
  "checks": {
    "database": {
      "status": "ok",
      "latency_ms": 5,
      "connections_active": 45,
      "connections_max": 100
    },
    "cache": {
      "status": "ok",
      "latency_ms": 1
    },
    "payment_gateway": {
      "status": "degraded",
      "latency_ms": 2300,
      "message": "High latency detected"
    }
  }
}
```

---

## 15. Synthetic Monitoring

### Blackbox Exporter (Prometheus)

Tests external endpoints from the outside.

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: [200]
      method: GET
      tls_config:
        insecure_skip_verify: false

  http_post:
    prober: http
    http:
      method: POST
      body: '{"test": true}'
      headers:
        Content-Type: application/json

  tcp_connect:
    prober: tcp
    timeout: 5s

  icmp:
    prober: icmp
    timeout: 5s
```

```yaml
# Prometheus scrape config for blackbox
- job_name: 'blackbox'
  metrics_path: /probe
  params:
    module: [http_2xx]
  static_configs:
    - targets:
      - https://api.example.com/health
      - https://app.example.com/login
  relabel_configs:
    - source_labels: [__address__]
      target_label: __param_target
    - target_label: __address__
      replacement: blackbox-exporter:9115
```

### User Journey Monitoring

```javascript
// Playwright synthetic test (run every 5 minutes from multiple regions)
const { chromium } = require('playwright');

async function checkCheckoutFlow() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  const start = Date.now();
  
  await page.goto('https://shop.example.com');
  await page.click('[data-test="product-1"]');
  await page.click('[data-test="add-to-cart"]');
  await page.click('[data-test="checkout"]');
  
  const loginForm = await page.waitForSelector('[data-test="email-input"]');
  
  const duration = Date.now() - start;
  
  // Report to monitoring
  metrics.gauge('synthetic.checkout_flow_ms', duration, {region: REGION});
  
  await browser.close();
}
```

### Synthetic vs Real User Monitoring

```
Synthetic: Controlled, scheduled, consistent
  + Always runs (even with 0 real users)
  + Tests specific user journeys
  + Multi-region visibility
  - Doesn't represent real user diversity
  - Can miss issues affecting specific user segments

RUM: Real users, real browsers, real conditions
  + Actual user experience data
  + Catches browser-specific issues
  + Reflects geographic performance
  - No data until users visit
  - Cannot test pre-launch
```

---

## 16. Anomaly Detection

### Statistical Thresholds

```python
# Dynamic threshold using rolling mean + standard deviations
def is_anomalous(current_value, historical_values, threshold_stddev=3):
    mean = np.mean(historical_values)
    std = np.std(historical_values)
    
    z_score = (current_value - mean) / std
    return abs(z_score) > threshold_stddev
```

**Prometheus: Predicting vs threshold**:
```promql
# Alert when metric deviates significantly from weekly pattern
abs(
  http_request_rate - 
  avg_over_time(http_request_rate[7d] offset 1w)
) > 0.2 * avg_over_time(http_request_rate[7d] offset 1w)
```

### ML-Based Anomaly Detection

**Datadog Anomaly Monitor**:
- Uses DBSCAN, statistical seasonality detection
- Automatically accounts for daily/weekly patterns
- Reduces alert fatigue for cyclically varying metrics

**Dynatrace Davis AI**:
- Causal AI — determines root cause, not just anomaly
- Automatically correlates metrics, logs, traces
- Reduces MTTR by pinpointing problem service

### Techniques

| Technique | Use Case | Complexity |
|---|---|---|
| Static threshold | Known baseline | Low |
| Percentile bands | Slowly changing metrics | Low |
| Z-score / MAD | General anomaly | Medium |
| Seasonal decomposition | Cyclical patterns | Medium |
| Isolation Forest | Multivariate anomalies | High |
| LSTM / Autoencoders | Complex time series | High |

---

## 17. Capacity Planning

### Trend Analysis

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Historical monthly data points (requests per second)
months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
rps = np.array([1000, 1100, 1250, 1300, 1500, 1700])

model = LinearRegression()
model.fit(months, rps)

# Predict 3, 6, 12 months ahead
future_months = np.array([9, 12, 18]).reshape(-1, 1)
predictions = model.predict(future_months)
print(f"9 months: {predictions[0]:.0f} RPS")
print(f"12 months: {predictions[1]:.0f} RPS")
print(f"18 months: {predictions[2]:.0f} RPS")
```

### Headroom Calculation

```
Current usage: 65% CPU
Current capacity: 20 pods × 4 vCPU = 80 vCPU
Current load: 52 vCPU

Growth rate: 15% month-over-month
Target headroom: Never exceed 70% utilization

Months until 70% breach:
  52 × (1.15)^n = 80 × 0.70 = 56
  (1.15)^n = 56/52 = 1.077
  n = log(1.077) / log(1.15) ≈ 0.5 months

→ Need to scale within 2 weeks!
```

### Capacity Planning Framework

```
1. Measure: Current resource utilization by service
2. Trend: Growth rate from historical data (linear, exponential)
3. Project: Future needs at P50 (expected), P90 (likely), P99 (worst case)
4. Plan: Lead time for provisioning (cloud: minutes; hardware: months)
5. Thresholds: Trigger scaling at 60-70% (not 90%)
6. Review: Quarterly capacity review meetings
```

---

## 18. Incident Management

### Severity Levels

| Severity | Definition | Response Time | Examples |
|---|---|---|---|
| P0/SEV0 | Complete outage, all users affected | Immediate (< 5 min) | Site down, data loss |
| P1/SEV1 | Major degradation, most users affected | < 15 minutes | Checkout broken, auth failing |
| P2/SEV2 | Partial degradation, subset affected | < 1 hour | 10% error rate on one region |
| P3/SEV3 | Minor issue, workaround available | Business hours | Slow dashboard, cosmetic bug |
| P4/SEV4 | No immediate user impact | Next sprint | Performance degradation trend |

### Incident Lifecycle

```
Detection → Triage → Declare Incident → Respond → Resolve → Postmortem

1. DETECTION (automated alert or user report)
   - Alert fires in PagerDuty
   - Ack within 5 minutes

2. TRIAGE (< 15 minutes)
   - Is this an incident? (yes/no)
   - What is the severity?
   - Who is the Incident Commander?

3. RESPONSE
   - Open incident bridge (Zoom/Slack channel)
   - Incident Commander: coordinate, delegate, communicate
   - Responders: investigate and fix
   - Comms role: update status page, notify stakeholders

4. RESOLUTION
   - Service restored to normal
   - Monitor for recurrence
   - Declare incident resolved

5. POSTMORTEM (within 48 hours for P0/P1)
```

### Blameless Postmortem

```markdown
## Incident Postmortem: API Outage - 2024-01-15

### Incident Summary
Duration: 47 minutes
Impact: 100% of API requests failed with 503
Severity: P0

### Timeline (UTC)
14:32 - Deploy of version 2.4.1 completed
14:35 - Error rate spike to 100% detected (by alert)
14:37 - On-call acked alert, started investigation
14:42 - Root cause identified: DB connection string misconfigured in deploy
14:45 - Rollback initiated
14:52 - Service restored
15:19 - Monitoring confirmed stability

### Root Cause
New environment variable DB_HOST was not set in production deployment.
Application defaulted to localhost, failing all DB queries.

### Contributing Factors
- No pre-deployment validation of required env vars
- Staging environment had correct value (human error not caught)
- Deployment checklist not followed

### Action Items
| Action | Owner | Due Date |
|--------|-------|----------|
| Add env var validation to deployment pipeline | @alice | 2024-01-22 |
| Add smoke test post-deployment | @bob | 2024-01-25 |
| Update deployment checklist | @alice | 2024-01-18 |

### What Went Well
- Alert fired within 3 minutes
- Rollback procedure was fast and well-practiced
- Incident comms kept stakeholders informed

Note: This was a process failure, not individual failure. No blame.
```

---

## 19. Performance Profiling

### CPU Flame Graphs

```
Generated by: perf (Linux), py-spy (Python), async-profiler (JVM), pprof (Go)

Visualization:
  - X axis: alphabetical by function name (NOT time)
  - Width: time spent in function and its callees
  - Y axis: call stack depth
  - Wide bars at bottom: hot functions consuming most CPU

Reading: Look for wide plateaus — those are the bottlenecks
```

```bash
# Python CPU profiling
pip install py-spy
py-spy record -o profile.svg --pid 12345

# Go pprof
import _ "net/http/pprof"
# GET http://localhost:6060/debug/pprof/profile?seconds=30
go tool pprof http://localhost:6060/debug/pprof/profile

# JVM async-profiler
./profiler.sh -d 30 -f profile.html <PID>
```

### Memory Profiling

```python
# Python memory profiling
from memory_profiler import profile

@profile
def my_function():
    data = [i for i in range(1000000)]
    return sum(data)

# Output:
# Line #    Mem usage    Increment   Line Contents
# 5       50.1 MiB      0.0 MiB    @profile
# 6       50.1 MiB      0.0 MiB    def my_function():
# 7      127.8 MiB     77.7 MiB        data = [i for i in range(1000000)]
# 8       50.5 MiB    -77.3 MiB        return sum(data)
```

### Database Query Analysis

```sql
-- PostgreSQL EXPLAIN ANALYZE
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.created_at > NOW() - INTERVAL '7 days'
AND u.country = 'US';

-- Look for:
-- Seq Scan on large table → missing index
-- Nested Loop on large result sets → might need Hash Join
-- High "actual rows" vs "estimated rows" → stale statistics
-- Buffers: hit (cached) vs read (disk)

-- Find slow queries
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 20;
```

---

## 20. RUM vs Synthetic Monitoring

### Real User Monitoring (RUM)

Measures performance from actual user browsers/devices.

```javascript
// Web Vitals (Core Web Vitals — Google ranking factor)
import { getLCP, getFID, getCLS, getFCP, TTFB } from 'web-vitals';

getLCP(metric => sendToAnalytics('LCP', metric.value));  // Largest Contentful Paint
getFID(metric => sendToAnalytics('FID', metric.value));  // First Input Delay
getCLS(metric => sendToAnalytics('CLS', metric.value));  // Cumulative Layout Shift
getFCP(metric => sendToAnalytics('FCP', metric.value));  // First Contentful Paint

// Datadog RUM
import { datadogRum } from '@datadog/browser-rum';

datadogRum.init({
  applicationId: 'app-id',
  clientToken: 'pub...',
  site: 'datadoghq.com',
  sampleRate: 100,
  trackInteractions: true,
  defaultPrivacyLevel: 'mask-user-input'
});
```

### Core Web Vitals Thresholds

| Metric | Good | Needs Improvement | Poor |
|---|---|---|---|
| LCP (loading) | < 2.5s | 2.5s - 4.0s | > 4.0s |
| FID (interactivity) | < 100ms | 100 - 300ms | > 300ms |
| CLS (visual stability) | < 0.1 | 0.1 - 0.25 | > 0.25 |
| TTFB (server response) | < 800ms | 800ms - 1.8s | > 1.8s |

### RUM vs Synthetic Comparison

| Aspect | RUM | Synthetic |
|---|---|---|
| Data source | Real users | Simulated users |
| Coverage | All user journeys (actual) | Specific journeys (configured) |
| Pre-launch testing | No | Yes |
| Geographic coverage | Wherever users are | Where you configure probes |
| Volume | High (all users) | Low (scheduled runs) |
| Variability | High (real conditions) | Low (controlled) |
| Privacy | PII concerns | No user data |
| Best for | Understanding real UX | Detecting regressions, alerting |

**Best practice**: Use both. Synthetic for alerting (consistent signal), RUM for UX insights.

---

## 21. Quick Reference

### Observability Tooling Landscape

| Category | Open Source | Commercial |
|---|---|---|
| Metrics | Prometheus, VictoriaMetrics | Datadog, New Relic, Dynatrace |
| Visualization | Grafana | Datadog, Splunk |
| Logging | ELK, Loki, OpenSearch | Datadog Logs, Splunk, Sumo Logic |
| Tracing | Jaeger, Zipkin, Tempo | Datadog APM, Dynatrace |
| Unified (OTel) | OpenTelemetry Collector | Honeycomb, Lightstep |
| Incident Mgmt | n/a | PagerDuty, OpsGenie, FireHydrant |
| Synthetic | Blackbox Exporter | Datadog Synthetics, Pingdom |
| Error Tracking | Sentry (OSS) | Sentry Cloud, Rollbar, Bugsnag |

### SLO Burn Rate Calculation Table

| Alert Window | Burn Rate | Budget Consumed | Response |
|---|---|---|---|
| 1h | 14.4x | 2% in 1h | Page now (critical) |
| 6h | 6x | 5% in 6h | Page now (high) |
| 1d | 3x | 10% in 3d | Ticket + possible page |
| 3d | 1x | 10% in 30d | Ticket (business hours) |

### Four Golden Signals (Google SRE)

```
1. Latency:   Time to service a request (p50, p95, p99)
2. Traffic:   Demand on your system (RPS, QPS, events/sec)
3. Errors:    Rate of failed requests (HTTP 5xx, exceptions)
4. Saturation: How full your service is (CPU %, queue depth, memory %)
```

### Incident Severity Quick Reference

```
P0: Site down → Wake everyone up NOW
P1: Major feature broken → Page on-call immediately  
P2: Partial degradation → Page on-call, can wait for acknowledgment
P3: Minor issue → Ticket, fix in sprint
P4: Technical debt / toil → Backlog
```

### Prometheus Query Cheat Sheet

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate %
100 * sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# p99 latency
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# Apdex score (satisfy < 0.3s, tolerate < 1.2s)
(sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m])) +
 sum(rate(http_request_duration_seconds_bucket{le="1.2"}[5m]))) /
(2 * sum(rate(http_requests_total[5m])))

# Pod memory usage
container_memory_working_set_bytes{container!="POD", namespace="production"}

# Disk usage %
100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)
```
