# HLD Interview Q&A — File 14: Data Pipelines and Analytics

---

## Easy Questions (Q1–Q7)

---

### Q1. What is the difference between batch processing and stream processing, and when do you use each?

**Answer:**

**Batch processing** collects data over a period, then processes the entire dataset at once in a scheduled job. There is an inherent delay between data arrival and result availability.

```
Data accumulates → Job runs (e.g., midnight) → Results available hours later

Example: Daily sales report generated at 2am for the previous day.
         ETL job runs nightly to load data warehouse.
         Monthly billing computation.
```

**Stream processing** processes each event continuously as it arrives, producing results with low latency (milliseconds to seconds).

```
Data arrives → Processed immediately → Result available in near-real-time

Example: Fraud detection on each credit card transaction.
         Live dashboard of website traffic.
         Real-time alerting on application errors.
```

**Comparison:**

| Dimension          | Batch                           | Stream                           |
|--------------------|---------------------------------|----------------------------------|
| Latency            | Hours to days                   | Milliseconds to seconds          |
| Throughput         | Very high (optimized for bulk)  | High (but lower than batch peak) |
| Complexity         | Lower                           | Higher (state, windows, ordering)|
| Cost               | Lower (off-peak compute)        | Higher (always-on compute)       |
| Data completeness  | Complete data available         | May process late-arriving data   |
| Failure recovery   | Re-run the job                  | Complex (checkpointing, replays) |
| Use cases          | Reports, ML training, ETL       | Fraud, monitoring, recommendations|

**Decision guide:**
```
Need results in < 1 minute?    → Stream processing
Processing historical data?    → Batch
Can afford overnight delay?    → Batch (cheaper, simpler)
Need real-time user impact?    → Stream
Building ML training datasets? → Batch
```

**Hybrid (Lambda/Kappa):** Many systems combine both — batch for accurate historical analysis, stream for real-time estimates.

---

### Q2. What is ETL vs ELT, and why has ELT won in the cloud?

**Answer:**

**ETL (Extract, Transform, Load):**
Data is extracted from sources, transformed (cleaned, joined, aggregated) in a separate processing layer, then loaded into the destination (data warehouse).

```
Source DB → [Extract] → [Transform in ETL server] → [Load to DW]
```

**ELT (Extract, Load, Transform):**
Data is extracted and loaded raw into the destination first, then transformed inside the destination using SQL.

```
Source DB → [Extract] → [Load raw to cloud DW] → [Transform with SQL inside DW]
```

**Why ELT won in the cloud:**

1. **Cloud data warehouses are powerful enough:** BigQuery, Snowflake, Redshift can run massively parallel SQL transformations on petabytes. The DW IS the transformation engine.

2. **Raw data preservation:** Loading raw data first means you can re-transform with different logic without re-extracting from the source (source systems change, become unavailable).

3. **Faster to load:** No transformation bottleneck — load raw data quickly, transform asynchronously later.

4. **SQL is the universal skill:** Data analysts can write SQL transformations. ETL required engineers.

5. **dbt (data build tool):** Made ELT transformations modular, testable, and version-controlled.

```sql
-- ELT transformation in BigQuery (dbt model)
-- models/orders_daily.sql
SELECT
    DATE(created_at) AS order_date,
    COUNT(*) AS order_count,
    SUM(total) AS revenue
FROM {{ source('raw', 'orders') }}
WHERE status = 'completed'
GROUP BY 1
```

**Tool landscape:**

| Phase     | ETL                       | ELT                          |
|-----------|---------------------------|------------------------------|
| Extract   | Informatica, Talend       | Fivetran, Airbyte, Stitch    |
| Transform | Same ETL tool             | dbt, Spark SQL, stored procs |
| Load      | Same ETL tool             | Fivetran (automated)         |
| Store     | On-prem DW                | BigQuery, Snowflake, Redshift|

---

### Q3. What is Apache Kafka and why is it a strong data pipeline backbone?

**Answer:**

Apache Kafka is a distributed, fault-tolerant, append-only log. It acts as a persistent, high-throughput message queue for decoupling data producers from consumers.

**Core concepts:**
```
Topic: A named stream of records (like a table in a DB)
Partition: A topic is divided into ordered, immutable partitions (parallelism unit)
Producer: Writes records to topics
Consumer Group: A group of consumers that together consume a topic
  → Each partition consumed by exactly one consumer in the group
Offset: Position within a partition (enables replay and checkpointing)
Broker: A Kafka server node
ZooKeeper/KRaft: Coordinates cluster metadata
```

**Why Kafka works as a pipeline backbone:**

1. **Decoupling:** Producers and consumers are independent. Add new consumers without changing producers.
2. **Persistence:** Messages retained for days/weeks (default 7 days). Replay data for new consumers or recovery.
3. **High throughput:** Sequential disk writes (append-only). 1 million messages/second per broker is achievable.
4. **Scalability:** Add partitions for more parallelism. Add consumer instances to scale processing.
5. **Multiple consumer groups:** Same data consumed by fraud detection, analytics, notification service simultaneously — each independently.

```
[Order Service] → Kafka topic: "orders"
                         ↓
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
[Analytics DB]   [Fraud Service]   [Inventory Service]
(Consumer Grp A) (Consumer Grp B)  (Consumer Grp C)

Each consumer group maintains its own offset → independent progress
```

**Throughput tuning:**
```python
# Producer: batch messages for higher throughput
producer = KafkaProducer(
    batch_size=65536,        # 64KB batch
    linger_ms=10,            # Wait 10ms to accumulate batch
    compression_type='lz4',  # Compress batches
    acks='all'               # Wait for all ISR replicas
)
```

---

### Q4. What is Change Data Capture (CDC), and how does Debezium work with PostgreSQL?

**Answer:**

**Change Data Capture (CDC)** is a technique for capturing every insert, update, and delete in a database and streaming those changes to downstream systems in real-time. It is the foundation for keeping multiple data stores in sync.

**Why CDC instead of polling:**
```
Polling approach:
  SELECT * FROM orders WHERE updated_at > ?
  Problems: Misses deletes, requires updated_at on every table, high DB load, not real-time

CDC approach:
  Reads database transaction log (WAL in PostgreSQL)
  Captures every change at the SQL level
  No impact on the source database queries
  Captures deletes (not possible with polling)
```

**How Debezium works with PostgreSQL:**

PostgreSQL has a Write-Ahead Log (WAL) — every INSERT/UPDATE/DELETE is written to the WAL before it is applied. Debezium uses PostgreSQL's **logical replication** feature to read the WAL as a stream of change events.

```
Step 1: Enable logical replication in PostgreSQL
  postgresql.conf:
    wal_level = logical
    max_replication_slots = 4
    max_wal_senders = 4

Step 2: Create a replication slot
  SELECT pg_create_logical_replication_slot('debezium', 'pgoutput');

Step 3: Debezium connector reads from replication slot
  → Transforms WAL events into Kafka messages
  → Each change event includes: before state + after state + transaction metadata
```

**Kafka message format:**
```json
{
  "before": {"id": 123, "status": "pending"},
  "after":  {"id": 123, "status": "shipped"},
  "source": {
    "table": "orders",
    "db": "shop",
    "ts_ms": 1705324800000,
    "lsn": 12345678
  },
  "op": "u"  // c=create, u=update, d=delete, r=read(snapshot)
}
```

**Use cases:**
- Keep Elasticsearch in sync with PostgreSQL (search index updated on every change).
- Audit log: every change captured durably.
- Cache invalidation: invalidate Redis cache on DB change.
- Event-driven microservices: services react to DB changes without polling.

---

### Q5. What is the Lambda architecture? Explain batch layer, speed layer, and serving layer.

**Answer:**

Lambda architecture (coined by Nathan Marz, 2011) is a data processing architecture that handles both historical (batch) and real-time (stream) data processing in parallel, combining results at query time.

**Motivation:** Batch processing is accurate but slow. Stream processing is fast but approximate/incomplete. Lambda runs both and merges results.

```
                     ┌──────────────────────────────┐
Raw Data ────────────┤                              │
                     │         Batch Layer          │
                     │  (Hadoop/Spark, reprocesses  │
                     │   full historical dataset)   │
                     │                              │
                     └───────────────┬──────────────┘
                                     │  Batch views
                                     ▼
Raw Data ────────────┐       ┌───────────────┐
                     │       │  Serving Layer│ ← Query merges both views
                     │       │  (Druid/Hive) │
                     │       └───────────────┘
                     │               ▲
                     │               │  Realtime views
                     └──────────────┐│
                                    ││
                     ┌──────────────┘│
                     │  Speed Layer  │
                     │  (Kafka +     │
                     │   Flink)      │
                     └───────────────┘
```

**Three layers:**

**Batch Layer:**
- Stores the master dataset (immutable, append-only raw data).
- Periodically (hourly, daily) reprocesses the entire dataset to produce accurate batch views.
- Corrects errors from the speed layer (has the full picture).
- Technology: HDFS/S3 + Spark/Hadoop.

**Speed Layer:**
- Processes incoming data in real-time.
- Produces real-time views covering only the gap between the latest batch view and now.
- Approximate (may miss late data), but low latency.
- Technology: Kafka + Flink/Spark Streaming.

**Serving Layer:**
- Stores batch views and real-time views.
- At query time, merges both to produce the final result.
- `result = batch_view + speed_view (for recent period)`

**Example: Counting page views by URL**
```
Batch view:   views_count["url_A"] = 1,000,000  (as of midnight)
Speed view:   views_count["url_A"] = 5,432       (since midnight)
Query result: 1,005,432
```

**Problem with Lambda:** Maintain two codebases (batch and stream) for the same computation. The Kappa architecture simplifies this.

---

### Q6. What is the Kappa architecture? When do you use it instead of Lambda?

**Answer:**

**Kappa architecture** (proposed by Jay Kreps, LinkedIn, 2014) simplifies Lambda by eliminating the separate batch layer. Everything is stream processing — historical reprocessing is done by replaying the stream from the beginning.

```
Lambda:
  Batch Layer (Spark) + Speed Layer (Flink) → Merge → Serving Layer
  Two codebases, two systems, double maintenance

Kappa:
  Stream Layer only (Kafka + Flink) → Serving Layer
  One codebase, one system
```

**How Kappa handles historical reprocessing:**
```
Kafka retains data indefinitely (or for a long period).
When logic changes, start a new consumer with offset=0:
  → New consumer reads from the beginning
  → Builds a new materialized view
  → Once caught up, switch traffic to new view
  → Delete old view
```

```
Version 1 consumer: offset=current → produces "view_v1"
Version 2 consumer: starts at offset=0, reads all history → produces "view_v2"
When view_v2 is current: swap serving layer to use "view_v2", retire "view_v1"
```

**Lambda vs Kappa decision:**

| Scenario                                            | Choose   | Reason                               |
|-----------------------------------------------------|----------|--------------------------------------|
| Business logic changes rarely                       | Lambda   | Batch layer more accurate            |
| Need full historical reprocessing frequently         | Kappa    | Simpler to replay stream             |
| Team cannot maintain two codebases                  | Kappa    | Less operational overhead            |
| Streaming framework cannot match batch accuracy     | Lambda   | Batch layer fills the gap            |
| Data window is short (< 7 days)                     | Kappa    | Retention covers all needed history  |
| Long history needed (years of data)                 | Lambda   | Object storage cheaper than Kafka    |
| All computation can be expressed as stream          | Kappa    | Unified codebase                     |

**Modern trend:** Kappa with a data lakehouse as the persistent store (Kafka for streaming, Iceberg/Delta for long-term storage), using Flink for both real-time and historical processing.

---

### Q7. What are windowing operations in stream processing? Explain tumbling, sliding, and session windows.

**Answer:**

Windowing groups stream events into finite buckets for aggregation. Without windows, you cannot compute "count per minute" or "sum per hour" on an infinite stream.

**Tumbling Window:**
Fixed-size, non-overlapping windows. Each event belongs to exactly one window.
```
Window size: 1 minute

|──── window 1 ────|──── window 2 ────|──── window 3 ────|
00:00           01:00            02:00             03:00

Events in window 1: [00:05, 00:23, 00:47] → count = 3
Events in window 2: [01:12, 01:55]        → count = 2
```
Use case: "Count requests per minute", "Revenue per hour", "Errors per 5-minute period."

**Sliding Window:**
Fixed-size windows that advance by a smaller step (slide interval). Events can belong to multiple overlapping windows.
```
Window size: 5 min, Slide: 1 min

|─────── window 1 ───────|
   |─────── window 2 ───────|
      |─────── window 3 ───────|

An event at 14:03 appears in windows covering [14:00–14:05], [14:01–14:06], [14:02–14:07], etc.
```
Use case: "p99 latency over the last 5 minutes, updated every minute", "Moving average."

**Session Window:**
Dynamic-size windows grouped by activity with gaps. A session window closes when there is no new event for a specified gap period.
```
Gap timeout: 30 minutes

Session 1: [10:00, 10:05, 10:12] → gap of > 30 min → session closes
Session 2: [11:00, 11:25, 11:28] → gap of > 30 min → session closes
```
Use case: "User session analytics", "Group related events by activity burst."

```python
# Apache Flink windowing example
stream.keyBy("user_id")
      .window(TumblingEventTimeWindows.of(Time.minutes(1)))  # Tumbling
      .aggregate(new CountAggregator())

stream.keyBy("user_id")
      .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))  # Sliding

stream.keyBy("user_id")
      .window(EventTimeSessionWindows.withGap(Time.minutes(30)))  # Session
```

**Event time vs processing time:** Window assignment should use event time (when event occurred) not processing time (when it arrived at Flink), to handle out-of-order events correctly.

---

## Medium Questions (Q8–Q15)

---

### Q8. How does Apache Flink compare to Spark Streaming?

**Answer:**

Both Flink and Spark are distributed stream processing frameworks, but they have different processing models with real trade-offs.

**Spark Streaming (DStream / Structured Streaming):**
- Original DStream model: **micro-batch** — collects events for a small time window (e.g., 500ms), then processes as a batch.
- Structured Streaming: Improved model with lower latency but still fundamentally micro-batch (with a "continuous mode" for sub-second latency).
- Deep integration with Spark ecosystem (Spark SQL, MLlib, Delta Lake).

**Apache Flink:**
- True **event-by-event** stream processing. Each event is processed as it arrives.
- Designed from the ground up for streaming; batch is a special case of streaming.
- Native support for event time, watermarks, complex state, and exactly-once semantics.

**Comparison:**

| Dimension                  | Spark Structured Streaming  | Apache Flink                   |
|----------------------------|-----------------------------|--------------------------------|
| Processing model           | Micro-batch (10ms–seconds)  | True event-by-event            |
| Latency                    | ~100ms–1s typical           | ~10ms typical                  |
| Batch + stream unification | Excellent (same API)        | Good (DataStream + Table API)  |
| Exactly-once               | Yes                         | Yes (with checkpointing)       |
| State management           | Good (structured streaming) | Excellent (RocksDB backend)    |
| Event time / watermarks    | Good                        | Excellent                      |
| Complex event processing   | Limited                     | Strong (pattern matching)      |
| Ecosystem integration      | Excellent (Spark ecosystem) | Good (Kafka-native)            |
| Operational maturity       | Very mature                 | Mature                         |
| Learning curve             | Lower (SQL-first)           | Higher                         |

**When to choose Flink:**
- Latency requirements < 100ms.
- Complex stateful processing (joins across streams with large windows).
- Per-event processing semantics critical (e.g., financial transactions).
- Large state that must survive failures (Flink's RocksDB state backend is superior).

**When to choose Spark Streaming:**
- Already using Spark (batch + stream from one framework).
- Team expertise is in Spark/SQL.
- Latency of 1–10 seconds is acceptable.
- Deep integration with Delta Lake or Spark MLlib needed.

---

### Q9. What is event time vs processing time, and how do watermarks handle late data?

**Answer:**

**Event time:** The time at which the event actually occurred (embedded in the event data). This is the correct basis for most business logic ("how many orders were placed in the 2pm–3pm hour?").

**Processing time:** The time at which the event is processed by the stream engine. This varies based on network delays, consumer lag, and system load.

```
Event:       Order placed at 14:23:05 (event time)
Kafka lag:   15 seconds
Flink:       Processes event at 14:23:20 (processing time)
Difference:  15 seconds of skew
```

**Why event time matters:**
```
Tumbling window: 14:00–15:00

Event at event_time 14:58 arrives with 5 minutes delay at processing_time 15:03.

Using processing time: Event assigned to window 15:00–16:00 ← WRONG
Using event time:      Event assigned to window 14:00–15:00 ← CORRECT
```

**Watermarks — handling late data:**
A watermark is a timestamp assertion: "I believe all events with timestamp ≤ W have been received." It tells the engine when it is safe to close a window and emit results.

```
Stream: events with timestamps 14:30, 14:32, 14:28, 14:35, 14:31...
Order may differ due to network delays.

Watermark = max(observed_event_time) - allowed_lateness (e.g., 2 minutes)

At event_time=14:35: watermark = 14:33
  → Window [14:00–14:30] can be closed now (watermark > window end)
  → Windows [14:30–15:00] stays open (watermark < window end)
```

```python
# Flink watermark strategy
WatermarkStrategy
    .forBoundedOutOfOrderness(Duration.ofMinutes(2))  # Allow 2 min lateness
    .withTimestampAssigner((event, ts) -> event.getEventTimestamp())
```

**Late data handling options:**
1. **Drop late events:** Simple, some data loss.
2. **Side output:** Route late events to a separate stream for later processing.
3. **Update results:** Allow windows to update when late data arrives (triggers).

```python
# Flink: emit late events to side output
DataStream<Order> lateOrders = mainStream.getSideOutput(lateOutputTag);
lateOrders.addSink(deadLetterQueueSink);
```

---

### Q10. How does Apache Flink achieve exactly-once processing?

**Answer:**

Exactly-once processing guarantees that each event affects the output exactly once — not zero times (at-most-once) and not more than once (at-least-once). This is critical for financial systems, billing, and deduplication-sensitive applications.

**The challenge:**
```
At-most-once:  Drop messages on failure. Simple, but data loss.
At-least-once: Retry on failure. No data loss, but duplicates.
Exactly-once:  No loss, no duplicates. Complex to achieve.
```

**Flink's mechanism: Chandy-Lamport Distributed Snapshots + Kafka Transactions**

**Step 1: Checkpointing**
Flink periodically injects "barrier" markers into the data stream. When a barrier passes through all operators, the current state is snapshotted to durable storage (HDFS, S3).

```
Stream: [event1, event2, ────barrier_42────, event3, event4, ────barrier_43────]

When all operators receive barrier_42:
  → Snapshot all operator states (stateful aggregations, joins) to S3
  → Record Kafka consumer offsets in snapshot
  → Snapshot is checkpoint 42

On failure:
  → Restart from checkpoint 42
  → Replay Kafka from the stored offsets
  → State restored to exactly what it was at barrier_42
```

**Step 2: Exactly-once to Kafka sinks (two-phase commit)**
Even if Flink processes exactly once internally, it must ensure sink writes are also exactly-once.

```
Phase 1 (Pre-commit): When checkpoint begins, Flink pre-commits output 
                       to Kafka (writes to Kafka transaction, not committed)
Phase 2 (Commit):     When checkpoint completes successfully, Flink 
                       commits the Kafka transaction

On failure before commit:
  → Flink restores from checkpoint
  → Kafka transaction is aborted (consumers don't see it)
  → Events replayed and re-produced
```

**Conditions for exactly-once end-to-end:**
```
1. Source must be replayable (Kafka with committed offsets ✓)
2. Operators must have recoverable state (Flink checkpoints ✓)
3. Sinks must support idempotent writes OR transactional writes
   - Idempotent: writing same data twice = same result (Elasticsearch with _id, DynamoDB with PK)
   - Transactional: Kafka (2PC), JDBC (database transactions)
```

**Performance cost:**
Exactly-once has overhead (transaction coordination, checkpoint overhead). For lower latency, at-least-once + idempotent sinks is a common production choice.

---

### Q11. What are data serialization formats? When to use Avro, Parquet, ORC, and Protobuf?

**Answer:**

The serialization format affects storage size, read/write speed, schema evolution, and ecosystem compatibility.

**Avro:**
- Row-oriented format.
- Schema stored separately in a Schema Registry; data is compact binary.
- Excellent schema evolution support.
- Best for: Message queues (Kafka), streaming systems, event serialization.

```json
// Avro schema
{
  "type": "record", "name": "Order",
  "fields": [
    {"name": "id", "type": "long"},
    {"name": "total", "type": "double"},
    {"name": "status", "type": {"type": "enum", "symbols": ["pending","shipped"]}}
  ]
}
```

**Parquet:**
- Columnar format (stores data column by column, not row by row).
- Excellent compression (similar values in a column compress well).
- Predicate pushdown: skip reading columns/rows not needed by query.
- Best for: Data lake analytics, OLAP queries, storage in S3/GCS.

```
Row store:   [id=1, name=Alice, age=30] [id=2, name=Bob, age=25]
Columnar:    [id: 1,2] [name: Alice,Bob] [age: 30,25]

Query: SELECT AVG(age) FROM users
Parquet: Only reads the "age" column → 10x+ less I/O
```

**ORC (Optimized Row Columnar):**
- Similar to Parquet but optimized for Hive.
- Built-in indexes (bloom filters, zone maps) for fast query pruning.
- Best for: Hive workloads, HDInsight, environments where Parquet is not the standard.

**Protobuf (Protocol Buffers):**
- Compact binary format, strongly typed, generated code for 10+ languages.
- Schema defined in `.proto` files; code generation for serialization/deserialization.
- Best for: gRPC APIs, service-to-service messaging, config storage (not analytics).

**Decision matrix:**

| Scenario                          | Format   | Reason                               |
|-----------------------------------|----------|--------------------------------------|
| Kafka message serialization       | Avro     | Schema registry, compact, evolvable  |
| Data lake analytics on S3         | Parquet  | Columnar, compression, predicate push|
| Hive-based analytics              | ORC      | Hive-native optimizations            |
| gRPC API payloads                 | Protobuf | Generated code, fast, compact        |
| REST API payloads                 | JSON     | Human-readable, universal            |
| ML feature store                  | Parquet  | Columnar, fast for ML training reads |

---

### Q12. What is a data lakehouse, and how do Delta Lake, Iceberg, and Hudi differ?

**Answer:**

**Data Lake problem:** Raw data in S3/GCS/ADLS is cheap to store but lacks ACID transactions, schema enforcement, and efficient updates. Small files accumulate. No versioning.

**Data Warehouse problem:** Fast SQL, ACID, but expensive, closed format, can't store unstructured data, separate from the lake.

**Data Lakehouse** = Data Lake storage + Data Warehouse features (ACID, schema enforcement, time travel, efficient updates) on open-source formats.

```
Traditional stack:
  Raw data → [Data Lake (S3)] → [Data Warehouse (Redshift/BigQuery)]
  Two copies of data, ETL between them, expensive

Lakehouse stack:
  Raw data → [Data Lakehouse (S3 + Delta/Iceberg/Hudi)]
  One copy, open format, warehouse-like capabilities
```

**Delta Lake (by Databricks):**
- Transaction log (delta log) stored alongside Parquet files tracks all changes.
- Supports ACID, time travel (`VERSION AS OF N`, `TIMESTAMP AS OF`), schema enforcement, merge (upserts).
- Deep Spark integration (Databricks native).
- Open-sourced as Linux Foundation Delta format.

**Apache Iceberg (by Netflix/Apple):**
- Table format specification (engine-agnostic: Spark, Flink, Trino, Dremio, BigQuery).
- Better partition evolution (change partitioning without rewriting data).
- Hidden partitioning (users don't need to specify partition predicates in queries).
- Strong multi-engine support.

**Apache Hudi (by Uber):**
- Optimized for CDC-style update-heavy workloads (incremental ingestion).
- Two table types: Copy-on-Write (COW, read-optimized) and Merge-on-Read (MOR, write-optimized).
- Native incremental querying (efficiently get records changed since timestamp X).
- Strong integration with Spark and Flink.

**Comparison:**

| Dimension            | Delta Lake     | Apache Iceberg     | Apache Hudi        |
|----------------------|----------------|--------------------|--------------------|
| ACID transactions    | Yes            | Yes                | Yes                |
| Time travel          | Yes            | Yes                | Yes                |
| Multi-engine         | Improving      | Excellent          | Good               |
| Incremental query    | Limited        | Limited            | Excellent          |
| Update performance   | Good           | Good               | Excellent (MOR)    |
| Partition evolution  | Limited        | Excellent          | Good               |
| Primary ecosystem    | Databricks     | Engine-agnostic    | Spark/Uber         |

---

### Q13. What is Apache Airflow, and how does it orchestrate data pipelines?

**Answer:**

**Apache Airflow** is a platform for programmatically authoring, scheduling, and monitoring workflows. Workflows are defined as DAGs (Directed Acyclic Graphs) in Python.

**Core concepts:**
```
DAG (Directed Acyclic Graph):
  A workflow definition — which tasks run, in what order, with what dependencies.
  Cannot have cycles (task cannot depend on itself, directly or indirectly).

Task / Operator:
  A unit of work (Python function, Bash command, SQL query, Spark job, HTTP call).

Task Instance:
  A specific execution of a task at a given point in time.

Scheduler:
  Reads DAG files, determines which task instances should run, puts them on the queue.

Executor:
  Runs task instances (LocalExecutor, CeleryExecutor, KubernetesExecutor).
```

**Example DAG:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="daily_sales_pipeline",
    schedule_interval="0 2 * * *",  # Daily at 2am
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 3, "retry_delay": timedelta(minutes=5)},
) as dag:
    
    extract = PythonOperator(
        task_id="extract_from_postgres",
        python_callable=extract_fn
    )
    
    transform = PythonOperator(
        task_id="transform_data",
        python_callable=transform_fn
    )
    
    load = PostgresOperator(
        task_id="load_to_warehouse",
        sql="INSERT INTO sales_summary SELECT ...",
        postgres_conn_id="warehouse"
    )
    
    extract >> transform >> load  # Define dependencies
```

**Features that make Airflow valuable:**
- **Backfill:** Re-run historical dates with `airflow dags backfill -s 2024-01-01 -e 2024-01-31`.
- **Sensors:** Wait for external conditions (file in S3, row in DB, API availability).
- **Dynamic DAGs:** Python code generates task graphs programmatically (parametric pipelines).
- **XCom:** Share small data between tasks (large data should go through S3/DB).
- **Retry + alerting:** Automatic retries with configurable delay; email/Slack on failure.

**Limitations:** Airflow is a scheduler, not a processing engine. It orchestrates but doesn't process data. For data quality, integrate Great Expectations. For processing, use Spark/dbt operators.

---

### Q14. What is data quality validation, and how does Great Expectations work?

**Answer:**

**Data quality validation** ensures that data in a pipeline meets expectations for completeness, accuracy, consistency, and freshness before being used for analytics or served to users. Poor data quality produces wrong business decisions.

**Common data quality dimensions:**
```
Completeness:  No null values in required columns
Uniqueness:    No duplicate records (primary key constraint)
Validity:      Values in expected range or domain
Consistency:   Cross-table constraints hold (foreign key integrity)
Freshness:     Data is recent (no stale table)
Accuracy:      Values match source of truth
```

**Great Expectations (GX):**
Great Expectations is a Python framework for defining, validating, and documenting data quality expectations as code.

```python
import great_expectations as gx

context = gx.get_context()
datasource = context.sources.add_pandas_filesystem(
    name="orders_source", base_directory="./data"
)

# Define expectations on a batch of data
validator = context.get_validator(...)
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_unique("order_id")
validator.expect_column_values_to_be_between("total", min_value=0, max_value=100000)
validator.expect_column_values_to_be_in_set("status", ["pending","shipped","cancelled"])
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10000000)

# Run validation
results = validator.validate()
if not results.success:
    raise DataQualityError(f"Validation failed: {results}")
```

**Pipeline integration:**
```
Extract → [GX Validation Gate] → Transform → [GX Validation Gate] → Load

If validation fails:
  → Pipeline halts (data not loaded into warehouse)
  → Alert sent (Slack, PagerDuty)
  → Raw data sent to quarantine storage for investigation
```

**Data Docs:**
Great Expectations generates HTML documentation showing your expectations and recent validation results — a self-documenting data contract.

**Alternative tools:** dbt tests (for SQL transformations in the warehouse), Soda Core, AWS Deequ (Spark-based).

---

### Q15. What is pipeline observability? Cover data freshness, lag monitoring, and DLQ.

**Answer:**

**Pipeline observability** extends the principles of application observability (metrics, logs, traces) to data pipelines, providing visibility into data health, timeliness, and completeness.

**Key observability dimensions:**

**Data Freshness:**
"How old is the most recently processed data in each table/topic?"
```sql
-- Monitor max processed timestamp
SELECT MAX(processed_at) as last_record, 
       CURRENT_TIMESTAMP - MAX(processed_at) AS lag
FROM orders_processed
WHERE date = CURRENT_DATE;

-- Alert if data is more than 15 minutes stale
```

**Consumer Lag (Kafka):**
The gap between the latest message produced and the latest message consumed. High lag = consumers falling behind.
```bash
kafka-consumer-groups.sh --describe --group my-consumer
# Shows: TOPIC | PARTITION | CURRENT-OFFSET | LOG-END-OFFSET | LAG

# Alert if lag > 100,000 messages or > 5 minutes behind
```

**Throughput metrics:**
```
Records processed per second (rate)
Records written to sink per second
Batch processing time per run
Records dropped (DLQ count)
```

**Dead Letter Queue (DLQ):**
A DLQ is a holding area for messages that could not be processed successfully after all retries. Instead of blocking the pipeline, bad records are moved to the DLQ for investigation.

```python
# Kafka consumer with DLQ
def process_message(message):
    try:
        transform_and_write(message)
    except TransformationError as e:
        # Move to DLQ with error metadata
        dlq_producer.produce(
            "pipeline.orders.dlq",
            value=json.dumps({
                "original_message": message.value(),
                "error": str(e),
                "error_type": type(e).__name__,
                "failed_at": datetime.utcnow().isoformat(),
                "retry_count": message.headers().get("retry_count", 0)
            })
        )
        metrics.increment("pipeline.dlq.records")
```

**DLQ monitoring:**
```
Alert if DLQ message count increases by > 1% of processed volume.
Build a DLQ dashboard showing error distribution, error types, and affected data.
Implement DLQ reprocessing: after fix, replay DLQ messages through corrected pipeline.
```

**Pipeline SLO examples:**
```
Data freshness SLO: p95 data age < 5 minutes
Processing SLO:     Daily job completes before 6am
Completeness SLO:   > 99.9% of source records reach the sink
Error rate SLO:     < 0.1% of records go to DLQ
```

---

## Hard Questions (Q16–Q20)

---

### Q16. How do you handle schema evolution in streaming pipelines? Cover Avro compatibility rules.

**Answer:**

Schema evolution is one of the most critical operational challenges in data pipelines. As business requirements change, message schemas evolve — fields are added, removed, or renamed. Without careful management, this breaks producers and consumers.

**Schema Registry:**
A central service (Confluent Schema Registry, AWS Glue Schema Registry) that stores and enforces schema versions. Producers register schemas; consumers fetch them by ID.

```
Producer:
  1. Register/check schema in Schema Registry → gets schema_id
  2. Serialize message with schema_id prefix: [magic_byte][schema_id][avro_bytes]

Consumer:
  1. Read schema_id from message prefix
  2. Fetch schema from Schema Registry
  3. Deserialize message
```

**Avro compatibility modes:**

**Backward compatibility (default):**
New schema can read data written with old schema. Old fields removed, new optional fields added.
```json
// Old schema
{"name": "age", "type": "int"}

// New schema (backward compatible: removed 'age', added optional 'birth_year')
{"name": "birth_year", "type": ["null","int"], "default": null}

Consumer using new schema reads old messages:
  → 'age' field absent → ignored (old fields can be removed)
  → 'birth_year' field absent → uses default null ✓
```

**Forward compatibility:**
Old schema can read data written with new schema. New fields added with defaults.
```json
// New schema adds field
{"name": "loyalty_tier", "type": ["null","string"], "default": null}

Old consumer reading new message:
  → sees 'loyalty_tier' field → ignores unknown fields ✓
```

**Full compatibility:**
Both backward AND forward compatible simultaneously. Safe for rolling upgrades where consumers and producers don't all upgrade at once.

**Breaking changes to always avoid:**
```
✗ Removing a required (non-nullable) field without default
✗ Changing field type (int → string)
✗ Renaming a field (must use aliases)
✗ Adding a required field without default

Safe changes:
✓ Add optional field with default
✓ Remove optional field (if backward compat mode)
✓ Widen numeric type (int → long)
```

**Field renaming (using Avro aliases):**
```json
{
  "name": "order_total",       // New name
  "type": "double",
  "aliases": ["total"]         // Old name — Avro maps old field to new name during deserialization
}
```

---

### Q17. What is columnar storage and why is it 10–100x faster for analytics?

**Answer:**

Understanding columnar storage requires understanding the difference between OLTP (transactional) and OLAP (analytical) access patterns.

**Row-oriented storage (PostgreSQL, MySQL — OLTP):**
Stores entire rows contiguously on disk. Ideal for fetching complete records by primary key.
```
Disk layout:
[id=1, name=Alice, age=30, salary=90000]
[id=2, name=Bob,   age=25, salary=70000]
[id=3, name=Carol, age=35, salary=80000]

Query: SELECT * FROM users WHERE id = 2
→ Seek to row 2 on disk → Read entire row → Perfect for OLTP
```

**Columnar storage (Parquet, ORC, BigQuery — OLAP):**
Stores each column's values contiguously.
```
Disk layout:
[id:  1, 2, 3]
[name: Alice, Bob, Carol]
[age: 30, 25, 35]
[salary: 90000, 70000, 80000]

Query: SELECT AVG(salary) FROM users
→ Only read [salary] column → All other columns skipped → Massive I/O reduction
```

**Why 10–100x faster for analytics:**

**1. Column pruning (less I/O):**
```
Table: 100 columns, 1 billion rows
Query selects 3 columns
Row store:    Read all 100 columns × 1B rows = 100 TB read
Column store: Read 3 columns × 1B rows       = 3 TB read  → 33x less I/O
```

**2. Better compression:**
Columns contain homogeneous data. Compression algorithms (Run-Length Encoding, Dictionary Encoding, Delta Encoding) achieve far higher ratios on column data.
```
Column [status]: pending, pending, pending, shipped, shipped, ...
Run-length encoding: (pending, 3), (shipped, 2), ...
Compression ratio: 10-50x

Mixed row: [pending, 5.99, 2024-01-15, user_42, pending, 6.99, ...]
Compression ratio: 2-3x (diverse types, low redundancy)
```

**3. SIMD vectorized processing:**
Modern CPUs have SIMD (Single Instruction Multiple Data) instructions that process multiple values in a single clock cycle. Columnar data enables vectorized operations:
```
Columnar: [salary: 90000, 70000, 80000, 75000, 95000]
SIMD: Process 8 integers per clock cycle for AVG computation
```

**4. Predicate pushdown and late materialization:**
```
Query: SELECT name FROM users WHERE age > 30 AND salary > 80000

Step 1: Read [age] column, apply filter → matching row IDs = {1, 3}
Step 2: Read [salary] column for rows {1,3} → filter → {1}
Step 3: Read [name] column for row {1} only

Only accessed rows that pass all filters. Never read irrelevant rows.
```

**Parquet file structure:**
```
File → Row Groups (128MB chunks)
  → Column Chunks
    → Pages (data pages with min/max statistics)
      → Encoded data (dictionary, RLE, bit-packing)

Min/max statistics: Query engine can skip entire row groups without reading them.
```

---

### Q18. How do you backfill a stream processing pipeline?

**Answer:**

**Backfilling** is re-processing historical data through a stream pipeline — either because the pipeline was newly created and needs historical data, or because a bug was found and historical output needs to be corrected.

**The challenge:**
```
Stream pipeline: Always running, processing real-time events
Backfill: Needs to process millions/billions of historical events
           While not disrupting real-time processing
           And producing correct output without duplicates
```

**Strategy 1: Kafka replay (Kappa architecture)**
If Kafka retains data long enough and the backfill period is within retention:
```python
# Start a new consumer group with offset=beginning
kafka-consumer-groups.sh --reset-offsets \
  --group backfill-consumer-v2 \
  --topic orders \
  --to-earliest \
  --execute

# New consumer group processes all historical data
# Writes to separate output topic/table: orders_aggregate_v2
# Once caught up, swap production to point to v2
```

**Strategy 2: Historical replay from data lake**
For backfills beyond Kafka retention, or for batch data:
```python
# Flink can read from both Kafka (real-time) and S3/HDFS (historical)
# Use a bounded source for historical data

historical_source = FileSource.for_record_stream_format(
    AvroReaderFormat.forSchema(schema), Path("s3://datalake/orders/2024/")
)

# Run as a Flink batch job reading historical Parquet files
# Write to separate table, then merge or swap aliases
```

**Strategy 3: Shadow pipeline**
Run the new pipeline code in parallel with the old one during a transition period:
```
Real-time events → [Old Pipeline v1] → orders_v1 (production)
Real-time events → [New Pipeline v2] → orders_v2 (shadow)

Validate v2 output matches v1 for current data.
Run v2 backfill to populate historical data.
Cut over production to v2.
```

**Managing output during backfill:**

Problem: Backfill can flood the output sink (Kafka, DB) and affect production.

Solutions:
```
1. Write to a separate output topic/table during backfill (then alias swap)
2. Rate-limit the backfill consumer (throttle reads)
3. Run backfill in a separate cluster
4. Use time-based write routing: backfill writes to archive, real-time to live
```

**Exactly-once during backfill:**
Use idempotent writes (upsert with primary key) so re-running the backfill does not create duplicates:
```sql
INSERT INTO orders_summary (order_date, revenue, order_count)
VALUES ('2024-01-15', 50000, 1234)
ON CONFLICT (order_date) DO UPDATE
  SET revenue = EXCLUDED.revenue,
      order_count = EXCLUDED.order_count;
```

---

### Q19. How do you design a real-time analytics pipeline: IoT → Kafka → Flink → ClickHouse?

**Answer:**

This stack is a production-proven architecture for ingesting high-velocity IoT data and making it queryable in real-time with sub-second query latency.

**Full architecture:**
```
IoT Devices (millions of sensors)
      │  HTTP/MQTT
      ▼
[IoT Gateway / MQTT Broker]
      │  Event batches
      ▼
[Kafka Cluster]
  Topics: sensor.temperature, sensor.pressure, sensor.vibration
  Partitioning: by device_id (ensures order per device)
  Retention: 7 days (replay + backfill buffer)
      │  Consumer group
      ▼
[Flink Cluster]
  - Parse and validate events
  - Enrich (device metadata lookup from Redis)
  - Windowed aggregations (1-min tumbling window per device per metric)
  - Anomaly detection (deviation from rolling mean)
  - Output 1: Raw events → Kafka (for cold storage to S3)
  - Output 2: Aggregations → ClickHouse
      │               │
      ▼               ▼
[S3 / Data Lake]  [ClickHouse]
(raw storage,      (real-time OLAP,
 long-term)         dashboards)
```

**Each component's role:**

**IoT Gateway:**
```python
# Receive MQTT messages, batch into Kafka
# Validate device authentication
# De-duplicate within 10-second window (devices retry on network errors)
# Add reception_timestamp, gateway_id to event
```

**Kafka configuration:**
```properties
num.partitions=48                    # High parallelism for IoT volume
replication.factor=3                 # Fault tolerance
retention.ms=604800000               # 7 days
compression.type=lz4                 # Compress batches (sensors repeat similar values)
```

**Flink processing job:**
```java
// 1-minute tumbling aggregation per device per metric type
DataStream<SensorEvent> events = env
    .addSource(new FlinkKafkaConsumer<>("sensor.temperature", schema, kafkaProps));

events
    .keyBy(SensorEvent::getDeviceId)
    .window(TumblingEventTimeWindows.of(Time.minutes(1)))
    .aggregate(new SensorAggregator())   // min, max, avg, count
    .addSink(new ClickHouseSink("sensor_aggregates_1min"));
```

**ClickHouse table design:**
```sql
CREATE TABLE sensor_aggregates_1min (
    device_id     String,
    metric_type   LowCardinality(String),
    window_start  DateTime,
    min_value     Float64,
    max_value     Float64,
    avg_value     Float64,
    event_count   UInt32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (device_id, metric_type, window_start);

-- Query: dashboard showing temperature for device 42 in last hour
SELECT window_start, avg_value
FROM sensor_aggregates_1min
WHERE device_id = 'device_42'
  AND metric_type = 'temperature'
  AND window_start >= now() - INTERVAL 1 HOUR
ORDER BY window_start;
-- Runs in < 50ms even with billions of rows
```

**ClickHouse key properties:**
- Columnar storage → 10–100x faster analytical queries than PostgreSQL.
- MergeTree engine → high write throughput (millions of rows/second per server).
- Real-time inserts → data queryable immediately after insert.
- ReplicatedMergeTree → fault-tolerant via ZooKeeper/ClickHouse Keeper.

**End-to-end latency:** IoT event → dashboard visible: typically 2–5 seconds.

---

### Q20. What are exactly-once, at-least-once, and at-most-once delivery semantics in data pipelines?

**Answer:**

Delivery semantics define what happens to a message when a failure occurs during pipeline processing. The choice affects both data correctness and system complexity.

**At-most-once delivery:**
Send the message, don't retry. If the processing node crashes, the message is lost.
```
Producer → sends message → doesn't wait for ACK
           if crash before processing: message is lost

Best for: Metrics, logs, monitoring data where occasional loss is acceptable
          Lowest latency and complexity
```

**At-least-once delivery:**
Retry until acknowledged. Messages are never lost, but may be processed multiple times.
```
Producer → sends message → waits for ACK → if timeout: RETRIES
           if crash after processing but before ACK: message processed TWICE

Best for: Most business events. Handle duplicates with idempotency.
```

**Exactly-once delivery:**
Each message affects the output exactly once, regardless of failures.
```
Producer sends → state checkpointed → committed to sink atomically
                if crash: state restored, message reprocessed
                output shows message exactly once

Best for: Financial transactions, billing, inventory updates where duplicates are catastrophic
```

**End-to-end exactly-once requirements:**
```
1. Source is replayable       : Kafka (committed offsets) ✓
2. Processing is idempotent   : Deterministic transformation ✓
3. State is checkpointed      : Flink checkpoints to S3 ✓
4. Sink supports exactly-once :
   Option A: Idempotent writes (upsert by primary key)
   Option B: Transactional writes (2-phase commit)
```

**Implementing in practice:**

```python
# Kafka producer: acks='all' for at-least-once
# (idempotent producer mode for exactly-once within Kafka)
producer = KafkaProducer(
    acks='all',              # Wait for all ISR replicas
    enable_idempotence=True, # Kafka deduplicates within producer session
    retries=5
)

# Database sink: idempotent upsert for exactly-once semantics
def write_to_db(aggregated_result):
    db.execute("""
        INSERT INTO order_counts (order_date, count, revenue)
        VALUES (%s, %s, %s)
        ON CONFLICT (order_date) DO UPDATE
          SET count = EXCLUDED.count, revenue = EXCLUDED.revenue
    """, (result.date, result.count, result.revenue))
    # Safe to retry: same result written multiple times = idempotent
```

**Comparison:**

| Semantic         | Data Loss | Duplicates | Latency | Complexity | Best For                    |
|------------------|-----------|------------|---------|------------|-----------------------------|
| At-most-once     | Possible  | No         | Lowest  | Lowest     | Metrics, logs               |
| At-least-once    | No        | Possible   | Medium  | Medium     | Most events (+ idempotency) |
| Exactly-once     | No        | No         | Highest | Highest    | Financial, billing, inventory|

**Rule of thumb:** Design for at-least-once + idempotent sinks. This gives you exactly-once semantics with less complexity than true exactly-once pipelines.

---

## Quick Reference

```
BATCH vs STREAM
  Batch:  Hours/days latency, high throughput, simple, cheap
  Stream: Milliseconds latency, complex state, expensive

ETL vs ELT
  ETL: Transform before load (old approach, separate engine)
  ELT: Load raw, transform in warehouse (BigQuery/Snowflake + dbt)

KAFKA CORE CONCEPTS
  Topic → Partition → Offset
  Consumer Group: each partition consumed by one instance
  Retention: default 7 days (enables replay)

CDC (Debezium)
  Reads PostgreSQL WAL via logical replication
  before + after state + operation (c/u/d/r)
  Writes change events to Kafka

LAMBDA vs KAPPA
  Lambda: Batch layer + Speed layer + Serving (complex, accurate)
  Kappa:  Stream only + replay for reprocessing (simple)

WINDOWS IN STREAM PROCESSING
  Tumbling: Fixed, non-overlapping (count per minute)
  Sliding:  Overlapping (rolling average over last 5 min)
  Session:  Gap-based grouping (user sessions)

WATERMARKS
  = max(event_time) - allowed_lateness
  Signal to close windows when late events are unlikely

FLINK EXACTLY-ONCE
  Chandy-Lamport checkpoints + Kafka 2-phase commit
  Snapshot state + Kafka offsets → restore on failure

AVRO COMPATIBILITY
  Backward: New reads old (new optional fields, remove old fields)
  Forward:  Old reads new (new fields must have defaults)
  Full:     Both directions safe

COLUMNAR STORAGE BENEFITS
  Column pruning: only read needed columns
  Better compression: homogeneous data per column
  SIMD vectorization: process multiple values per cycle
  Predicate pushdown: skip row groups by min/max stats

DATA FORMATS
  Avro    → Kafka messaging (row, schema registry, evolvable)
  Parquet → Data lake analytics (columnar, compressed)
  ORC     → Hive workloads (columnar, bloom filters)
  Protobuf→ gRPC APIs (compact binary, code gen)

DEAD LETTER QUEUE
  Unprocessable messages → DLQ (don't block pipeline)
  Monitor DLQ growth rate as pipeline health signal
  Replay from DLQ after fix

DELIVERY SEMANTICS
  At-most-once:  No retry, possible data loss
  At-least-once: Retry, possible duplicates
  Exactly-once:  Checkpoint + idempotent/transactional sink

CLICKHOUSE KEY PROPERTIES
  Columnar MergeTree engine
  Millions of rows/second insert throughput
  Sub-second analytical queries on billions of rows
```

---

*File 14 of 15 — Data Pipelines and Analytics*
