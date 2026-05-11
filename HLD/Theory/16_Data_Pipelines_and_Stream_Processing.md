# 16. Data Pipelines and Stream Processing

## Table of Contents
1. [ETL vs ELT](#1-etl-vs-elt)
2. [Batch Processing: Hadoop and Spark](#2-batch-processing-hadoop-and-spark)
3. [Stream Processing: Flink, Spark Streaming, Kafka Streams](#3-stream-processing-flink-spark-streaming-kafka-streams)
4. [Lambda Architecture](#4-lambda-architecture)
5. [Kappa Architecture](#5-kappa-architecture)
6. [Change Data Capture (CDC)](#6-change-data-capture-cdc)
7. [Data Pipeline Components](#7-data-pipeline-components)
8. [Apache Airflow](#8-apache-airflow)
9. [Data Quality](#9-data-quality)
10. [Event Time vs Processing Time vs Ingestion Time](#10-event-time-vs-processing-time-vs-ingestion-time)
11. [Windowing Operations](#11-windowing-operations)
12. [Watermarks and Late Data](#12-watermarks-and-late-data)
13. [Exactly-Once Processing Semantics](#13-exactly-once-processing-semantics)
14. [Data Serialization Formats](#14-data-serialization-formats)
15. [Schema Evolution](#15-schema-evolution)
16. [Data Lineage and Catalog](#16-data-lineage-and-catalog)
17. [Real-Time Analytics Pipeline](#17-real-time-analytics-pipeline)
18. [Data Lakehouse](#18-data-lakehouse)
19. [Backfilling and Reprocessing](#19-backfilling-and-reprocessing)
20. [Pipeline Observability](#20-pipeline-observability)
21. [Quick Reference](#21-quick-reference)

---

## 1. ETL vs ELT

### ETL (Extract, Transform, Load)

```
Source Systems → [Extract] → [Transform] → [Load] → Data Warehouse
                  (raw data)  (clean, shape)  (store)
                             ↑
                    Transformation happens BEFORE loading
                    Typically done by dedicated ETL server
```

**When ETL was dominant**: When data warehouses (Teradata, Oracle) were expensive per GB and compute was separate. Transform before loading because storage was costly.

```python
# Traditional ETL pipeline
def etl_pipeline():
    # Extract
    raw_orders = extract_from_postgres("SELECT * FROM orders")
    
    # Transform (outside warehouse)
    transformed = []
    for order in raw_orders:
        transformed.append({
            'order_id': order['id'],
            'revenue': order['total'] * (1 - order['discount_pct'] / 100),
            'order_date': parse_date(order['created_at']).strftime('%Y-%m-%d'),
            'customer_segment': classify_customer(order['customer_id'])
        })
    
    # Load
    load_to_snowflake(transformed, table='fact_orders')
```

### ELT (Extract, Load, Transform)

```
Source Systems → [Extract] → [Load] → [Transform] → Analytics
                  (raw data)  (store)   (in-warehouse)
                                       ↑
                    Transformation happens AFTER loading
                    Using warehouse's own compute (dbt, SQL)
```

**Why ELT is now dominant**: Cloud data warehouses (BigQuery, Snowflake, Redshift) are cheap storage + powerful compute. Load raw data first, transform later with SQL.

```sql
-- ELT: Transform inside warehouse using dbt (Data Build Tool)

-- models/staging/stg_orders.sql
SELECT
    id AS order_id,
    customer_id,
    total * (1 - discount_pct / 100.0) AS revenue,
    DATE(created_at) AS order_date,
    status,
    CASE
        WHEN total > 500 THEN 'high_value'
        WHEN total > 100 THEN 'medium_value'
        ELSE 'low_value'
    END AS order_tier
FROM raw.orders
WHERE status != 'test'

-- models/marts/fact_orders.sql
SELECT
    o.*,
    c.country,
    c.acquisition_source
FROM {{ ref('stg_orders') }} o
JOIN {{ ref('stg_customers') }} c ON o.customer_id = c.customer_id
```

### ETL vs ELT Comparison

| Aspect | ETL | ELT |
|---|---|---|
| Transformation location | External ETL server | Inside data warehouse |
| Raw data stored? | No (only transformed) | Yes (full history) |
| Re-transformation | Hard (re-run ETL) | Easy (rerun SQL) |
| Scalability | Limited by ETL server | Scales with warehouse |
| Cost model | ETL compute + storage | Storage cheap, compute on demand |
| Tooling | Informatica, SSIS, Talend | dbt, Dataform, Spark SQL |
| Latency | Higher | Lower for simple transforms |
| Best for | Privacy requirements (PII reduction), complex logic | Analytics, ML, modern data stack |

---

## 2. Batch Processing: Hadoop and Spark

### Hadoop MapReduce

```
Input Data (HDFS) → Map phase → Shuffle/Sort → Reduce phase → Output (HDFS)

Word count example:
  Input: "hello world hello"
  Map: [(hello,1), (world,1), (hello,1)]
  Shuffle: {hello: [1,1], world: [1]}
  Reduce: {hello: 2, world: 1}
```

**Architecture**:
```
HDFS (Hadoop Distributed File System):
  NameNode: Metadata server — knows location of data blocks
  DataNodes: Actual data storage (replicated 3x by default)

YARN (Resource Manager):
  ResourceManager: Cluster resource scheduling
  NodeManager: Per-node resource tracking
  ApplicationMaster: Per-job coordination
```

**Hadoop limitations**:
- Writes all intermediate data to disk (slow for multi-step jobs)
- Fixed two-phase Map+Reduce model
- Complex Java API
- Largely superseded by Spark for batch

### Apache Spark

In-memory distributed computing engine. 10-100x faster than Hadoop MapReduce.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("OrderAnalytics") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Read from various sources
orders_df = spark.read.parquet("s3://data-lake/orders/")
customers_df = spark.read.format("jdbc").option("url", jdbc_url).load()

# Transformations (lazy — not executed until action)
result = orders_df \
    .filter(F.col("status") == "delivered") \
    .filter(F.col("created_at") >= "2024-01-01") \
    .join(customers_df, on="customer_id") \
    .groupBy("country", "product_category") \
    .agg(
        F.sum("revenue").alias("total_revenue"),
        F.count("order_id").alias("order_count"),
        F.avg("revenue").alias("avg_order_value"),
        F.percentile_approx("revenue", 0.5).alias("median_revenue")
    ) \
    .orderBy(F.desc("total_revenue"))

# Action — triggers execution
result.write.mode("overwrite").parquet("s3://data-lake/reports/revenue_by_country/")
```

### Spark Architecture

```
Driver Program (your code)
     │
     │  SparkContext / SparkSession
     │
     ▼
Cluster Manager (YARN / Kubernetes / Standalone)
     │
     ├── Executor (Worker 1) — 4 cores, 4GB RAM
     │     ├── Task 1 (partition 1)
     │     ├── Task 2 (partition 2)
     │     └── Cache (RDD/DataFrame partitions)
     │
     ├── Executor (Worker 2) — 4 cores, 4GB RAM
     │     ├── Task 3 (partition 3)
     │     └── Task 4 (partition 4)
     │
     └── Executor (Worker N) ...
```

### Spark Core Concepts

```python
# RDD (Resilient Distributed Dataset) — low-level
rdd = spark.sparkContext.textFile("hdfs://data/logs/*.txt")
words = rdd.flatMap(lambda line: line.split())
counts = words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)

# DataFrame (high-level, Catalyst optimizer)
df = spark.read.json("s3://data/events/")

# Catalyst Optimizer: automatically optimizes:
# - Predicate pushdown (filter early)
# - Constant folding
# - Column pruning
# - Join reordering

# Caching (keep in memory for repeated access)
orders_df.cache()
orders_df.persist(StorageLevel.MEMORY_AND_DISK)  # Spill to disk if needed

# Partitioning
df.repartition(200)  # Shuffle to N partitions (expensive)
df.coalesce(10)      # Reduce partitions (no shuffle)
df.repartitionByRange("country")  # Range partition for sorted output
```

### Spark Optimization Tips

```python
# 1. Avoid data skew — add salt to skewed join keys
# 2. Broadcast small tables
from pyspark.sql.functions import broadcast
result = large_df.join(broadcast(small_lookup_df), "key")

# 3. Adaptive Query Execution (Spark 3.0+)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# 4. Avoid UDFs (Python UDFs are slow — serialization overhead)
# Instead of:
@udf(returnType=StringType())
def slow_udf(value):
    return value.upper()

# Use built-in functions:
df.withColumn("upper", F.upper("value"))
```

---

## 3. Stream Processing: Flink, Spark Streaming, Kafka Streams

### Apache Flink

True stream processing engine with strong state management and exactly-once semantics.

```java
// Flink DataStream API
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

DataStream<Order> orders = env
    .addSource(new FlinkKafkaConsumer<>("orders", new OrderSchema(), kafkaProps));

DataStream<RevenueByCategory> revenue = orders
    .filter(order -> order.getStatus().equals("delivered"))
    .keyBy(Order::getCategory)
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .aggregate(new RevenueAggregator(), new RevenueWindowFunction());

revenue.addSink(new FlinkKafkaProducer<>("revenue-by-hour", new RevenueSchema(), kafkaProps));

env.execute("Revenue Aggregation Job");
```

```python
# Flink PyFlink Table API
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

t_env.execute_sql("""
    CREATE TABLE orders (
        order_id STRING,
        product_category STRING,
        revenue DOUBLE,
        order_time TIMESTAMP(3),
        WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'orders',
        'properties.bootstrap.servers' = 'kafka:9092',
        'format' = 'json'
    )
""")

t_env.execute_sql("""
    SELECT
        product_category,
        TUMBLE_START(order_time, INTERVAL '1' HOUR) AS window_start,
        SUM(revenue) AS total_revenue,
        COUNT(*) AS order_count
    FROM orders
    GROUP BY
        product_category,
        TUMBLE(order_time, INTERVAL '1' HOUR)
""")
```

### Spark Streaming (Structured Streaming)

Micro-batch processing with stream semantics. Batch Spark API extended to streams.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType

spark = SparkSession.builder.appName("StreamingRevenue").getOrCreate()

# Read from Kafka
orders_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "orders") \
    .load()

# Parse JSON
schema = StructType() \
    .add("order_id", StringType()) \
    .add("category", StringType()) \
    .add("revenue", DoubleType()) \
    .add("order_time", TimestampType())

orders = orders_stream \
    .select(F.from_json(F.col("value").cast("string"), schema).alias("data")) \
    .select("data.*")

# Windowed aggregation
revenue = orders \
    .withWatermark("order_time", "10 minutes") \
    .groupBy(
        F.col("category"),
        F.window("order_time", "1 hour")
    ) \
    .agg(F.sum("revenue").alias("total_revenue"))

# Write to sink
query = revenue \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "revenue-by-hour") \
    .option("checkpointLocation", "s3://checkpoints/revenue/") \
    .outputMode("update") \
    .start()

query.awaitTermination()
```

### Kafka Streams

Library (not cluster), embedded in application. No separate cluster to manage.

```java
StreamsBuilder builder = new StreamsBuilder();

KStream<String, Order> ordersStream = builder.stream("orders",
    Consumed.with(Serdes.String(), orderSerde));

// Stateful processing with state store
KTable<Windowed<String>, Long> orderCounts = ordersStream
    .filter((key, order) -> order.getStatus().equals("delivered"))
    .groupBy((key, order) -> order.getCategory())
    .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofHours(1)))
    .count();

orderCounts.toStream()
    .map((windowedKey, count) -> KeyValue.pair(
        windowedKey.key(),
        new CategoryCount(windowedKey.key(), windowedKey.window().start(), count)
    ))
    .to("order-counts-by-category");

KafkaStreams streams = new KafkaStreams(builder.build(), props);
streams.start();
```

### Stream Processing Comparison

| Aspect | Apache Flink | Spark Streaming | Kafka Streams |
|---|---|---|---|
| Processing model | True streaming | Micro-batch (100ms-1s) | True streaming |
| Latency | Milliseconds | Seconds | Milliseconds |
| Exactly-once | Yes (Flink native) | Yes (Spark 2.0+) | Yes |
| State management | Built-in RocksDB | Structured Streaming | Built-in RocksDB |
| Deployment | Separate cluster | Spark cluster | Embedded in app |
| Complexity | High | Medium | Low |
| Throughput | Very high | Very high | High |
| SQL support | Flink SQL | Spark SQL | ksqlDB (separate) |
| Best for | Complex event processing, CEP | Teams already using Spark | Simple Kafka transformations |

---

## 4. Lambda Architecture

### Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Data Ingestion Layer          │
                    │     (Kafka / Kinesis)                │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┼──────────────────────┐
                    │              │                        │
           ┌────────▼──────┐      │                        │
           │  Batch Layer  │      │                        │
           │  (Spark/Hive) │      │                        │
           │  Reprocesses  │      │                        │
           │  ALL data     │      │                        │
           │  periodically │      │                        │
           └────────┬──────┘      │                        │
                    │             │                         │
                    ▼             ▼                         │
           ┌─────────────┐  ┌────────────────────────┐    │
           │ Batch Views │  │     Speed Layer         │    │
           │ (pre-computed│  │   (Flink/Spark Stream)  │    │
           │  aggregations│  │   Processes only recent  │    │
           │  in Hive/ES) │  │   data (hours/days)     │    │
           └──────┬──────┘  └────────────┬────────────┘   │
                  │                       │                  │
                  └──────────┬────────────┘                 │
                             │                               │
                    ┌────────▼───────────────────┐          │
                    │       Serving Layer          │         │
                    │  (Merges batch + speed views)│         │
                    │  (Cassandra / HBase / Redis) │         │
                    └────────────────────────────┘          │
```

### How It Works

```
Query result = batch_view(all historical data) + speed_view(recent data not yet in batch)

Example: "Total orders in last 30 days"
  Batch view: Pre-computed orders up to 6 hours ago (recomputed every 6h)
  Speed view: Orders in last 6 hours (real-time)
  Answer: batch_view + speed_view
```

### Lambda Pros and Cons

**Pros**:
- Fault-tolerant: batch layer can recompute if speed layer corrupts
- Historical reprocessing possible without touching speed layer
- Each layer can use optimal technology

**Cons**:
- Two codebases (batch + streaming) for same logic — divergence bugs
- Operational complexity of maintaining two separate systems
- Latency for batch layer (hours)
- Harder to debug inconsistencies between layers
- High infrastructure cost

---

## 5. Kappa Architecture

### Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │     Event Store (Kafka)               │
                    │     (long retention: months/years)   │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┼──────────────────────┐
                    │              │                        │
                    │  Current version stream processor     │
                    │  (Flink job consuming from "now")    │
                    │                                       │
                    │  Reprocessing stream processor        │
                    │  (Flink job consuming from beginning) │
                    │  → writes to new output tables        │
                    │  → cutover when caught up             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │           Serving Layer               │
                    │       (ClickHouse / Cassandra)       │
                    └──────────────────────────────────────┘
```

### Key Insight

There is no separate batch layer. The stream processor handles everything:
- Real-time processing of current data
- Historical reprocessing by replaying Kafka from offset 0

### When Kappa Replaces Lambda

Use Kappa when:
- Stream processing can handle the required latency
- Kafka retention is sufficient for reprocessing (weeks/months)
- Single codebase is operationally preferred
- Team has strong stream processing expertise

Keep Lambda when:
- Batch jobs need SQL/Hive for complex transformations
- Historical data is much larger than Kafka can retain
- Exact reproducibility of historical computation is required
- Machine learning training requires full dataset batches

### Lambda vs Kappa Comparison

| Aspect | Lambda | Kappa |
|---|---|---|
| Layers | Batch + Speed + Serving | Streaming + Serving |
| Codebases | 2 (divergence risk) | 1 |
| Reprocessing | Batch re-runs | Replay Kafka |
| Latency | High for batch layer | Low (streaming only) |
| Complexity | High (two stacks) | Lower |
| Data retention | Can archive to S3 | Kafka retention (limited) |
| Historical queries | Efficient (batch views) | Reprocess from Kafka |
| Best for | Complex analytics + real-time | Real-time + simple historical |

---

## 6. Change Data Capture (CDC)

### What is CDC?

CDC captures every INSERT, UPDATE, DELETE from a database and streams them as events.

```
PostgreSQL WAL (Write-Ahead Log) → Debezium → Kafka
                                              │
                              ┌───────────────┼──────────────────┐
                              │               │                    │
                         Elasticsearch    Data Warehouse       Other Services
                         (search index)  (Snowflake/BigQuery) (cache, etc.)
```

### Debezium

Most popular CDC tool. Reads database transaction logs (WAL, binlog).

```yaml
# Debezium PostgreSQL connector configuration
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "secret",
    "database.dbname": "mydb",
    "database.server.name": "mydb",
    "table.include.list": "public.orders,public.users",
    "plugin.name": "pgoutput",
    "publication.autocreate.mode": "filtered",
    "decimal.handling.mode": "double",
    "snapshot.mode": "initial"  // Take initial snapshot, then stream changes
  }
}
```

### CDC Event Format

```json
// Debezium change event on Kafka topic: mydb.public.orders
{
  "before": {
    "id": 123,
    "status": "pending",
    "total": 149.99
  },
  "after": {
    "id": 123,
    "status": "shipped",
    "total": 149.99
  },
  "source": {
    "version": "2.3.0",
    "connector": "postgresql",
    "db": "mydb",
    "schema": "public",
    "table": "orders",
    "lsn": 34567890,
    "ts_ms": 1700000000000
  },
  "op": "u",    // c=create, u=update, d=delete, r=read(snapshot)
  "ts_ms": 1700000000100
}
```

### CDC Use Cases

```
1. Database → Search index synchronization
   PostgreSQL orders → Debezium → Kafka → Elasticsearch
   Every order change immediately searchable

2. Database → Cache invalidation
   User data changes → Debezium → Kafka → Cache invalidation service → Redis

3. Microservice event sourcing
   "Order updated" events from DB changes, not application code

4. Cross-datacenter replication
   Primary DB → Debezium → Kafka → Secondary DB

5. Audit trail
   All database changes captured with before/after states
```

### Logical Replication (PostgreSQL)

```sql
-- Enable logical replication in postgresql.conf
-- wal_level = logical

-- Create replication slot (Debezium creates this automatically)
SELECT pg_create_logical_replication_slot('debezium', 'pgoutput');

-- Create publication for specific tables
CREATE PUBLICATION debezium_pub FOR TABLE orders, users, products;

-- Grant replication permissions
CREATE USER debezium WITH REPLICATION LOGIN PASSWORD 'secret';
GRANT SELECT ON orders, users, products TO debezium;
```

---

## 7. Data Pipeline Components

### Pipeline Architecture

```
Ingest → Transform → Load → Orchestrate

Ingest:
  - Batch: SFTP, S3 drops, database exports, API pulls
  - Stream: Kafka, Kinesis, Pub/Sub, webhooks
  - CDC: Debezium, AWS DMS

Transform:
  - Batch: Spark, dbt, Hive
  - Stream: Flink, Kafka Streams, Spark Streaming

Load (Serving):
  - OLAP warehouse: Snowflake, BigQuery, Redshift, ClickHouse
  - NoSQL: Cassandra (time-series), MongoDB
  - Search: Elasticsearch
  - Cache: Redis (materialized aggregates)

Orchestrate:
  - Airflow, Prefect, Dagster, Temporal
  - Dependency management, scheduling, retry, alerting
```

### Data Pipeline Patterns

```
Fan-out:
  Raw event → [Kafka topic] → Consumer A (analytics)
                            → Consumer B (notifications)
                            → Consumer C (fraud detection)

Enrichment:
  Raw event + Reference data → Enriched event
  (e.g., add customer country from customer table)

Aggregation:
  1000 raw events → 1 summary record (per hour, per user, etc.)

Normalization / Denormalization:
  Normalize: Split data for storage efficiency
  Denormalize: Join for query performance
```

---

## 8. Apache Airflow

### Core Concepts

```
DAG (Directed Acyclic Graph):
  - A workflow with defined tasks and dependencies
  - Runs on a schedule or triggered

Task:
  - Unit of work (operator)
  - Has upstream and downstream dependencies

Operator:
  - Template for a task type
  - PythonOperator, BashOperator, SparkSubmitOperator, S3ToRedshiftOperator
```

### DAG Example

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email': ['data-alerts@example.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='daily_revenue_pipeline',
    default_args=default_args,
    schedule_interval='0 6 * * *',   # Every day at 6am UTC
    catchup=False,                    # Don't backfill missed runs
    tags=['revenue', 'daily']
) as dag:
    
    # Wait for source data to arrive
    wait_for_orders = S3KeySensor(
        task_id='wait_for_orders_file',
        bucket_name='data-lake',
        bucket_key='raw/orders/{{ ds }}/*.parquet',  # Template with date
        poke_interval=300,
        timeout=3600
    )
    
    # Transform with Spark
    transform_orders = SparkSubmitOperator(
        task_id='transform_orders',
        application='s3://code/spark/transform_orders.py',
        application_args=['--date', '{{ ds }}'],
        conf={'spark.executor.memory': '4g'},
        conn_id='spark_default'
    )
    
    # Load to data warehouse
    load_to_warehouse = PythonOperator(
        task_id='load_to_snowflake',
        python_callable=load_revenue_to_snowflake,
        op_kwargs={'date': '{{ ds }}'}
    )
    
    # Data quality check
    def check_revenue_not_null(**context):
        result = run_query("SELECT COUNT(*) FROM fact_orders WHERE revenue IS NULL")
        if result > 0:
            raise ValueError(f"Found {result} orders with null revenue!")
    
    quality_check = PythonOperator(
        task_id='quality_check',
        python_callable=check_revenue_not_null
    )
    
    # Notify on success
    notify_success = EmptyOperator(task_id='notify_success')
    
    # Define dependencies
    wait_for_orders >> transform_orders >> load_to_warehouse >> quality_check >> notify_success
```

### Airflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Airflow Components                        │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Scheduler   │  │  Webserver   │  │  Triggerer         │ │
│  │  (schedules  │  │  (UI + API)  │  │  (deferrable tasks)│ │
│  │   DAG runs)  │  │              │  │                    │ │
│  └──────┬───────┘  └──────────────┘  └────────────────────┘ │
│         │                                                     │
│         │ submits task instances                              │
│         ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Executor                           │   │
│  │  LocalExecutor: same process (dev only)               │   │
│  │  CeleryExecutor: Redis/RabbitMQ + worker pool         │   │
│  │  KubernetesExecutor: K8s pod per task (scalable)      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Metadata Database (PostgreSQL / MySQL)               │   │
│  │  Stores: DAG definitions, task states, XCom values    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Airflow vs Prefect vs Dagster

| Aspect | Airflow | Prefect | Dagster |
|---|---|---|---|
| Paradigm | DAG-first | Code-first, dynamic | Asset-first |
| Testing | Hard | Easy | Easy |
| Dynamic DAGs | Complex | Native | Native |
| Data awareness | No | No | Yes (assets) |
| UI | Good | Good | Excellent |
| Learning curve | High | Medium | Medium |
| Kubernetes native | KubernetesOperator | Yes | Yes |
| Best for | Complex schedules, large orgs | Modern pipelines | Data-centric orgs |

---

## 9. Data Quality

### Dimensions of Data Quality

```
Completeness:  Are all expected records present? (no missing rows)
Freshness:     Is the data up to date? (no stale data)
Accuracy:      Are values correct? (no wrong values)
Consistency:   Same data matches across systems?
Validity:      Values conform to expected format/range?
Uniqueness:    No unexpected duplicates?
```

### Great Expectations

Python library for data quality checks as code.

```python
import great_expectations as gx

context = gx.get_context()
validator = context.sources.pandas_default.read_parquet("s3://data/orders/2024-01-15.parquet")

# Define expectations
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_unique("order_id")
validator.expect_column_values_to_be_between("revenue", min_value=0, max_value=100000)
validator.expect_column_values_to_be_in_set("status", 
    ["pending", "processing", "shipped", "delivered", "cancelled"])
validator.expect_column_values_to_match_regex("order_id", r"^ord_[a-z0-9]{8}$")
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=1000000)

# Check if orders count isn't more than 20% lower than yesterday
validator.expect_table_row_count_to_be_between(
    min_value=int(yesterday_count * 0.8),
    max_value=int(yesterday_count * 1.5)
)

results = validator.validate()
if not results.success:
    raise DataQualityError(f"Data quality check failed: {results}")
```

### Schema Validation with Pydantic

```python
from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class OrderRecord(BaseModel):
    order_id: str
    customer_id: int
    status: OrderStatus
    revenue: float
    created_at: datetime
    
    @validator('revenue')
    def revenue_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Revenue cannot be negative')
        return round(v, 2)
    
    @validator('order_id')
    def order_id_format(cls, v):
        import re
        if not re.match(r'^ord_[a-z0-9]{8}$', v):
            raise ValueError('Invalid order ID format')
        return v
```

---

## 10. Event Time vs Processing Time vs Ingestion Time

### Three Notions of Time

```
Real world                    Pipeline
    │
Event occurs at t=00:00       ← Event time (when it happened)
    │
    │ (network delay, batching)
    │
Event arrives at Kafka t=00:05 ← Ingestion time (when it entered the pipeline)
    │
    │ (processing lag, queue)
    │
Event processed by Flink t=00:08 ← Processing time (when Flink sees it)
```

### Why Event Time Matters

```
Scenario: Count orders per hour

Processing time approach:
  - Simple, no state needed
  - Problem: late-arriving events (order placed at 1:55pm, arrives at 2:03pm)
  - Gets counted in 2pm bucket, not 1pm bucket — WRONG

Event time approach:
  - Use the timestamp from the event itself
  - Order placed at 1:55pm → counted in 1pm bucket — CORRECT
  - Challenge: must wait for late data before finalizing window
```

### Event Time in Flink

```java
// Tell Flink to use event time
env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

// Extract event time from record
DataStream<Order> orders = rawStream
    .assignTimestampsAndWatermarks(
        WatermarkStrategy
            .<Order>forBoundedOutOfOrderness(Duration.ofMinutes(5))  // Allow 5min late
            .withTimestampAssigner((order, timestamp) -> order.getOrderTime().toEpochMilli())
    );
```

### Time Type Comparison

| Aspect | Event Time | Processing Time | Ingestion Time |
|---|---|---|---|
| Timestamp source | Event payload | System clock | Kafka/broker timestamp |
| Accuracy | Most accurate | May miss late events | Intermediate |
| Complexity | High (watermarks) | Low | Medium |
| Deterministic | Yes (replay → same result) | No | Partially |
| Late data handling | Via watermarks | Ignored | Limited |
| Best for | Analytics requiring accuracy | Low-latency monitoring | Simple use cases |

---

## 11. Windowing Operations

### Tumbling Window

```
Time: 0───────────────────────────────────────────────────►
      │   Window 1    │   Window 2    │   Window 3    │
      │  [10:00-11:00]│  [11:00-12:00]│  [12:00-13:00]│

Events: ●    ●  ●        ●   ●    ●       ●  ●
```

Fixed-size, non-overlapping windows. Each event belongs to exactly one window.

```python
# Flink tumbling window
orders
    .keyBy("category")
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .aggregate(RevenueAggregator())

# Spark Structured Streaming
orders.groupBy(
    F.window("order_time", "1 hour"),  # Tumbling 1-hour windows
    "category"
).sum("revenue")
```

### Sliding Window

```
Time: 0──────────────────────────────────────────────────►
      │         Window 1          │
               │         Window 2          │
                        │         Window 3          │

Window size: 1 hour
Slide interval: 30 minutes
→ Each window overlaps by 30 minutes
→ Event may appear in multiple windows
```

```python
# Flink sliding window: 1-hour window, slides every 30 minutes
orders
    .keyBy("category")
    .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(30)))
    .aggregate(RevenueAggregator())

# Spark
orders.groupBy(
    F.window("order_time", "1 hour", "30 minutes"),  # size, slide
    "category"
).sum("revenue")
```

### Session Window

```
User activity:
  User A: ●──●──●         (20s gap)      ●──●
           ←session 1→                   ←session 2→
                      ↑gap > threshold → session split
```

Groups events within a gap timeout of each other. No fixed size — ends when no events for N minutes.

```python
# Flink session window (gap = 30 minutes)
user_events
    .keyBy("user_id")
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .aggregate(SessionAggregator())
```

### Global Window

All events belong to a single window. Typically used with custom triggers.

```python
# Trigger-based processing
orders
    .keyBy("order_id")
    .window(GlobalWindows.create())
    .trigger(CountTrigger.of(100))  # Fire every 100 events
    .aggregate(BatchAggregator())
```

### Windowing Comparison

| Window Type | Size | Overlap | Use Case |
|---|---|---|---|
| Tumbling | Fixed | None | Hourly reports, daily summaries |
| Sliding | Fixed | Yes (configurable) | Moving averages, rolling metrics |
| Session | Variable | None | User session analysis, clickstreams |
| Global | Unbounded | N/A | Count-based or custom trigger |

---

## 12. Watermarks and Late Data

### The Problem

```
Event time:     1:00  1:01  1:02  1:03  1:04  1:05
Events arrive:  1:03  1:02  1:04  1:05  1:03* 1:06
                                        ↑
                             Late event! Order at 1:03 arrived at 1:07

When should we close and emit the 1:00-1:02 window?
  Close too early → miss late events
  Close too late  → high latency
```

### Watermarks

A watermark is a statement: "All events with timestamp < watermark have been received."

```
Watermark = max_seen_event_time - allowed_lateness

Example with 5-minute allowed lateness:
  Events seen up to event_time=1:00:00
  Watermark = 1:00:00 - 5:00 = 12:55:00

  When watermark crosses 1:00:00 → close all windows ending before 1:00:00
```

```java
// Flink watermark with bounded out-of-orderness
WatermarkStrategy
    .<Order>forBoundedOutOfOrderness(Duration.ofMinutes(5))
    .withTimestampAssigner((event, ts) -> event.getTimestamp())

// When Flink receives events:
// event_time=14:30, watermark=14:25 (30-5)
// event_time=14:35, watermark=14:30
// → Window [14:00-14:30) can now be emitted (watermark > 14:30)
```

### Handling Late Data

```java
OutputTag<Order> lateOutputTag = new OutputTag<>("late-orders", Types.POJO(Order.class));

SingleOutputStreamOperator<Revenue> mainStream = orders
    .keyBy("category")
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .allowedLateness(Time.minutes(30))     // Wait extra 30min after watermark
    .sideOutputLateData(lateOutputTag)     // Capture data too late even for this
    .aggregate(new RevenueAggregator());

// Process late data separately
DataStream<Order> lateOrders = mainStream.getSideOutput(lateOutputTag);
lateOrders.addSink(new LateOrderSink());  // Alert/manual reconciliation
```

---

## 13. Exactly-Once Processing Semantics

### Delivery Semantics

```
At-most-once:   Message may be lost, never processed twice
  - Simplest implementation
  - Fire-and-forget producers
  - No retries

At-least-once:  Message will be processed at least once, may be duplicated
  - Retry on failure (most systems default)
  - Consumer must be idempotent

Exactly-once:   Message processed exactly once, end-to-end
  - Most complex, some performance cost
  - Required for financial transactions, order processing
```

### Flink Exactly-Once via Checkpointing

```
Flink uses distributed snapshots (Chandy-Lamport algorithm):

1. Checkpoint barrier injected into source streams
2. Operators process records until they see the barrier
3. Each operator saves its state snapshot to distributed storage (S3/HDFS)
4. Sink commits transaction once checkpoint completes
5. On failure: restore from last complete checkpoint, replay from barrier position
```

```java
// Enable exactly-once checkpointing
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.enableCheckpointing(60000);  // Checkpoint every 60 seconds
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30000);
env.getCheckpointConfig().setCheckpointTimeout(120000);

// Use transactional Kafka sink
FlinkKafkaProducer<String> kafkaSink = new FlinkKafkaProducer<>(
    "output-topic",
    new SimpleStringSchema(),
    kafkaProps,
    FlinkKafkaProducer.Semantic.EXACTLY_ONCE  // Uses Kafka transactions
);
```

### Kafka Exactly-Once Transactions

```java
// Kafka producer with idempotence + transactions
Properties props = new Properties();
props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, "true");
props.put(ProducerConfig.TRANSACTIONAL_ID_CONFIG, "producer-1");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

try {
    producer.beginTransaction();
    
    producer.send(new ProducerRecord<>("output", key1, value1));
    producer.send(new ProducerRecord<>("output", key2, value2));
    
    // Commit offset to input topic atomically with output
    producer.sendOffsetsToTransaction(offsets, consumerGroupId);
    
    producer.commitTransaction();
} catch (Exception e) {
    producer.abortTransaction();
    throw e;
}
```

### Idempotency as Alternative

When exactly-once is complex, design consumers to be idempotent:

```python
def process_order_payment(order_id: str, amount: float):
    # Check if already processed (idempotency key)
    if redis.exists(f"payment:processed:{order_id}"):
        return  # Already processed, skip
    
    # Process payment
    payment_result = stripe.charge(amount)
    
    # Mark as processed with TTL
    redis.setex(f"payment:processed:{order_id}", 86400, payment_result.id)
    
    return payment_result
```

---

## 14. Data Serialization Formats

### Apache Avro

```json
// Schema
{
  "type": "record",
  "name": "Order",
  "namespace": "com.example",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "customer_id", "type": "long"},
    {"name": "total", "type": "double"},
    {"name": "status", "type": {"type": "enum", "name": "Status",
      "symbols": ["PENDING", "SHIPPED", "DELIVERED"]}},
    {"name": "tags", "type": {"type": "array", "items": "string"}, "default": []}
  ]
}
```

```python
import avro.schema
import avro.io
import io

schema = avro.schema.parse(open("order.avsc").read())
writer = avro.io.DatumWriter(schema)
bytes_writer = io.BytesIO()
encoder = avro.io.BinaryEncoder(bytes_writer)
writer.write({"order_id": "ord_123", "customer_id": 42, "total": 99.99}, encoder)
```

### Apache Parquet

Column-oriented format for analytics.

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Write
table = pa.Table.from_pandas(orders_df, schema=pa.schema([
    ('order_id', pa.string()),
    ('customer_id', pa.int64()),
    ('total', pa.float64()),
    ('created_at', pa.timestamp('us'))
]))

pq.write_table(table, 's3://data-lake/orders/2024/01/15/part-0.parquet',
               compression='snappy',
               row_group_size=1000000)

# Read only needed columns (column pruning)
table = pq.read_table('orders.parquet', columns=['order_id', 'total'])

# Predicate pushdown (skip row groups based on min/max statistics)
filters = [('created_at', '>=', pd.Timestamp('2024-01-01'))]
table = pq.read_table('orders.parquet', filters=filters)
```

### ORC (Optimized Row Columnar)

Similar to Parquet. Better compression in some cases. More common in Hive ecosystem.

### Protobuf

Best for inter-service communication (gRPC) and streaming with strict schema.

### Format Comparison

| Format | Type | Compression | Schema | Use Case |
|---|---|---|---|---|
| JSON | Row | None (or gzip) | Optional | REST APIs, flexibility |
| Avro | Row | Snappy/Deflate | Required (embedded) | Kafka streaming, schema registry |
| Parquet | Column | Snappy/ZSTD | Embedded | Analytics, data lake |
| ORC | Column | ZLIB/Snappy | Embedded | Hive/HDFS analytics |
| Protobuf | Row | N/A | Required (external) | gRPC, inter-service |
| CSV | Row | None | None | Legacy, simple exchange |

### Schema Registry

```
Producer writes Avro message:
  1. Register schema with Schema Registry → get schema_id (e.g., 42)
  2. Write message: [magic byte][schema_id=42][avro_payload]
  
Consumer reads Avro message:
  1. Parse magic byte + schema_id
  2. Fetch schema from Registry by ID 42
  3. Deserialize payload using schema
```

```python
from confluent_kafka.avro import AvroProducer, AvroConsumer
from confluent_kafka import avro

schema = avro.load('order.avsc')

producer = AvroProducer({
    'bootstrap.servers': 'kafka:9092',
    'schema.registry.url': 'http://schema-registry:8081'
}, default_value_schema=schema)

producer.produce(topic='orders', value=order_dict)
```

---

## 15. Schema Evolution

### Compatibility Types

```
Forward compatible:
  New schema CAN read data written with OLD schema
  → Add new fields with defaults
  → Consumers can be upgraded first

Backward compatible:
  Old schema CAN read data written with NEW schema
  → Add new fields with defaults
  → Only add, never remove or rename
  → Producers can be upgraded first (write new data, old consumers still read it)

Full compatible:
  Both forward and backward compatible
  → Most restrictive: only add optional fields with defaults

None:
  No compatibility guarantee
  → Breaking changes (delete/rename required fields)
  → Requires coordinated migration
```

### Safe vs Breaking Changes

```python
# Safe changes (backward compatible):
# ✅ Add optional field with default
{"name": "email", "type": ["null", "string"], "default": null}

# ✅ Change field to wider type (int → long)
{"name": "count", "type": "long"}  # was int

# ✅ Add new enum value (with Avro FORWARD compatibility)

# Breaking changes (require migration plan):
# ❌ Remove required field
# ❌ Rename field
# ❌ Change type to incompatible type (string → int)
# ❌ Change required field to have different semantics
```

### Migration Strategy

```
1. Additive change: Just add field with default → no migration needed

2. Rename: 
   V1: {"name": "cust_id", "type": "long"}
   V2: {"name": "customer_id", "type": "long", "aliases": ["cust_id"]}
   → Both names work during transition

3. Remove field:
   V1 → V2 (mark deprecated) → V3 (remove)
   Keep field for 2 versions with warning

4. Type change (incompatible):
   Add new field alongside old:
   V1: {"name": "price", "type": "string"}    # mistake: "9.99"
   V2: {"name": "price_str", "type": "string"},  # keep old
       {"name": "price", "type": "double"}       # add new
```

---

## 16. Data Lineage and Catalog

### Data Lineage

Tracks where data comes from, how it was transformed, where it goes.

```
Source: PostgreSQL orders table
    │
    ├── Debezium CDC → Kafka orders topic
    │       │
    │       └── Flink job (transform) → Kafka enriched_orders topic
    │               │
    │               └── Clickhouse table → Grafana dashboard
    │
    └── dbt model → Snowflake fact_orders table
            │
            └── Tableau dashboard → Finance team report
```

### Apache Atlas

Open-source metadata and governance platform.

```python
from apache_atlas.client import AtlasClient

client = AtlasClient("http://atlas:21000", ("admin", "admin"))

# Create entity for a dataset
entity = {
    "typeName": "hive_table",
    "attributes": {
        "qualifiedName": "mydb.orders@cluster1",
        "name": "orders",
        "description": "Raw orders from production database",
        "owner": "data-team",
        "tableType": "MANAGED_TABLE"
    }
}

# Create lineage between process and datasets
lineage = {
    "typeName": "spark_process",
    "attributes": {
        "qualifiedName": "transform_orders_job",
        "inputs": [{"guid": orders_guid}],
        "outputs": [{"guid": enriched_orders_guid}]
    }
}
```

### DataHub (LinkedIn)

Modern data catalog with rich metadata, business glossary, data contracts.

```python
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import *

emitter = DatahubRestEmitter("http://datahub-gms:8080")

# Emit dataset metadata
dataset_urn = make_dataset_urn("kafka", "orders")
emitter.emit_mce(MetadataChangeEventClass(
    proposedSnapshot=DatasetSnapshotClass(
        urn=dataset_urn,
        aspects=[
            DatasetPropertiesClass(
                description="Order events from checkout service",
                customProperties={"team": "platform", "sla": "5min"}
            ),
            SchemaMetadataClass(...)
        ]
    )
))
```

### OpenLineage

Standard specification for data lineage events.

```json
// OpenLineage event
{
  "eventType": "START",
  "eventTime": "2024-01-15T10:30:00Z",
  "run": {"runId": "uuid"},
  "job": {"namespace": "spark", "name": "transform_orders"},
  "inputs": [
    {"namespace": "kafka", "name": "orders"}
  ],
  "outputs": [
    {"namespace": "snowflake", "name": "analytics.fact_orders"}
  ]
}
```

---

## 17. Real-Time Analytics Pipeline

### IoT → Kafka → Flink → ClickHouse → Grafana

```
Scenario: IoT sensors sending temperature readings from 10,000 devices

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  IoT Devices (10,000)                                            │
  │  → publish to MQTT broker every 30 seconds                       │
  └───────────────────────────────┬──────────────────────────────── ┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  MQTT → Kafka Bridge (EMQ X)      │
                    │  Topic: iot.sensor.temperature    │
                    └──────────────┬──────────────────-┘
                                   │ (~333 events/sec)
                    ┌──────────────▼──────────────────┐
                    │  Apache Kafka                     │
                    │  10 partitions, 7-day retention  │
                    └──────────────┬──────────────────-┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Apache Flink                     │
                    │  - Parse + validate events        │
                    │  - 1-minute tumbling windows      │
                    │  - Compute: min, max, avg, p95    │
                    │  - Alert if temp > threshold      │
                    └──────────────┬──────────────────-┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  ClickHouse                       │
                    │  - MergeTree table (columnar)     │
                    │  - 90-day retention               │
                    │  - Sub-second query latency       │
                    └──────────────┬──────────────────-┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Grafana Dashboard                │
                    │  - Real-time temperature maps     │
                    │  - Alert when sensor anomalous   │
                    └──────────────────────────────────┘
```

### ClickHouse Table Design

```sql
CREATE TABLE iot_readings (
    sensor_id        String,
    building_id      String,
    floor            Int32,
    temperature      Float32,
    humidity         Float32,
    event_time       DateTime64(3),
    processing_time  DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMMDD(event_time)    -- Partition by day
ORDER BY (building_id, sensor_id, event_time)  -- Sort key for efficient queries
TTL event_time + INTERVAL 90 DAY;       -- Auto-delete after 90 days

-- Materialized view for pre-aggregated 1-minute stats
CREATE MATERIALIZED VIEW iot_minute_stats
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMMDD(window_start)
ORDER BY (building_id, sensor_id, window_start)
AS SELECT
    sensor_id,
    building_id,
    toStartOfMinute(event_time) AS window_start,
    min(temperature) AS min_temp,
    max(temperature) AS max_temp,
    avg(temperature) AS avg_temp,
    count() AS reading_count
FROM iot_readings
GROUP BY building_id, sensor_id, window_start;
```

---

## 18. Data Lakehouse

### Problem with Traditional Architecture

```
Data Lake (S3/HDFS):
  + Cheap storage
  + Schema-on-read flexibility
  - No ACID transactions
  - Slow updates/deletes
  - No versioning
  - Data quality issues

Data Warehouse (Snowflake/BigQuery):
  + ACID transactions
  + Good performance
  + Data quality
  - Expensive
  - Data duplication from lake
  - Not good for ML (raw data needed)
```

### Data Lakehouse = ACID on Object Storage

Combines: cheap object storage (S3) + ACID transactions + warehouse-quality performance

### Delta Lake (Databricks)

```python
from delta import DeltaTable
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# Create Delta table
df.write.format("delta").mode("overwrite").save("s3://data-lake/delta/orders/")

# ACID UPDATE (not possible on raw Parquet)
delta_table = DeltaTable.forPath(spark, "s3://data-lake/delta/orders/")
delta_table.update(
    condition="status = 'cancelled'",
    set={"revenue": "0.0"}
)

# ACID DELETE (GDPR right to erasure)
delta_table.delete("customer_id = '12345'")

# MERGE (upsert)
delta_table.alias("target").merge(
    updates_df.alias("source"),
    "target.order_id = source.order_id"
).whenMatchedUpdate(set={"status": "source.status", "updated_at": "source.updated_at"}) \
 .whenNotMatchedInsert(values={"order_id": "source.order_id", "status": "source.status"}) \
 .execute()

# Time travel
orders_yesterday = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-14") \
    .load("s3://data-lake/delta/orders/")

orders_version_5 = spark.read.format("delta") \
    .option("versionAsOf", 5) \
    .load("s3://data-lake/delta/orders/")
```

### Apache Iceberg

```python
spark.sql("""
    CREATE TABLE iceberg.orders (
        order_id STRING,
        customer_id BIGINT,
        total DOUBLE,
        status STRING,
        created_at TIMESTAMP
    ) USING iceberg
    LOCATION 's3://data-lake/iceberg/orders'
    PARTITIONED BY (days(created_at))
""")

# Schema evolution — add column without rewrite
spark.sql("ALTER TABLE iceberg.orders ADD COLUMN discount DOUBLE AFTER total")

# Hidden partitioning — no more partition columns in queries
spark.sql("SELECT * FROM iceberg.orders WHERE created_at = '2024-01-15'")
# Automatically uses date partition without requiring partition column in WHERE

# Partition evolution — change partitioning strategy without rewrite
spark.sql("ALTER TABLE iceberg.orders REPLACE PARTITION FIELD days(created_at) WITH months(created_at)")
```

### Lakehouse Comparison

| Feature | Delta Lake | Apache Iceberg | Apache Hudi |
|---|---|---|---|
| ACID transactions | Yes | Yes | Yes |
| Time travel | Yes (version + timestamp) | Yes | Yes |
| Schema evolution | Yes | Yes (best) | Yes |
| Partition evolution | Limited | Yes | Yes |
| Streaming reads | Yes | Yes | Yes |
| Creator | Databricks | Netflix/Apple | Uber |
| Catalog support | Unity, Hive, Glue | Hive, Glue, REST | Hive, Glue |
| Best for | Databricks ecosystem | Multi-engine, schema evolution | Upsert-heavy workloads |

---

## 19. Backfilling and Reprocessing

### When to Backfill

- Bug fix in transformation logic
- New feature requires historical data
- Schema change that requires recomputation
- New data source retroactively added

### Backfill Strategies

#### Full Reprocessing

```python
# Airflow backfill from command line
airflow dags backfill \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --reset-dagruns \
    daily_revenue_pipeline

# Script-based backfill with parallelism
import concurrent.futures
from datetime import date, timedelta

def backfill_date(target_date):
    orders = extract_orders_for_date(target_date)
    transformed = transform(orders)
    load_to_warehouse(transformed, partition=target_date)

dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(365)]

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(backfill_date, d) for d in dates]
    for f in concurrent.futures.as_completed(futures):
        f.result()  # raises if failed
```

#### Kafka Reprocessing (Kappa)

```bash
# Set consumer group to beginning of topic
kafka-consumer-groups.sh \
    --bootstrap-server kafka:9092 \
    --group reprocessing-job \
    --topic orders \
    --reset-offsets \
    --to-earliest \
    --execute

# Start Flink job reading from beginning
# Write to new output table
# Cutover at specific timestamp
# Swap table alias
```

#### Incremental Backfill

```python
# Only reprocess changed records (using CDC or update timestamp)
def incremental_backfill(start_date, end_date):
    # Find all records modified since last correct processing
    stale_records = db.query("""
        SELECT order_id FROM orders
        WHERE updated_at BETWEEN %s AND %s
        AND processed_version < 2  -- current correct version
    """, (start_date, end_date))
    
    for batch in chunked(stale_records, 1000):
        reprocess_orders(batch)
        mark_as_processed(batch, version=2)
```

---

## 20. Pipeline Observability

### Data Freshness

```sql
-- Monitor data freshness: how old is the latest record?
SELECT
    NOW() - MAX(event_time) AS data_age,
    MAX(event_time) AS latest_event
FROM fact_orders;
-- Alert if data_age > 15 minutes (expected near real-time)
```

```promql
# Prometheus metric: data freshness gauge
data_freshness_seconds{pipeline="orders", sink="clickhouse"} 45

# Alert if stale
data_freshness_seconds > 900  # 15 minutes
```

### Lag Monitoring (Kafka)

```bash
# Consumer group lag per partition
kafka-consumer-groups.sh \
    --bootstrap-server kafka:9092 \
    --describe \
    --group flink-revenue-job

# Output:
# TOPIC    PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# orders   0          145230          145230          0
# orders   1          144987          145015          28   ← 28 messages behind!
```

```python
# Expose lag as Prometheus metric
from prometheus_client import Gauge

kafka_lag = Gauge('kafka_consumer_lag',
                  'Kafka consumer group lag',
                  ['topic', 'partition', 'consumer_group'])

for partition in lag_info:
    kafka_lag.labels(
        topic=partition.topic,
        partition=partition.partition,
        consumer_group=group_id
    ).set(partition.lag)
```

### Dead Letter Queues

```
Normal flow:
  Source Topic → Consumer → Process → Sink
  
Failed messages flow:
  Source Topic → Consumer → Process [FAIL] → DLQ Topic

DLQ handling:
  1. Alert when DLQ has messages
  2. Manual investigation of failed messages
  3. Fix bug, then replay from DLQ back to main topic
  4. Or: separate pipeline for DLQ with looser processing
```

```python
def process_with_dlq(message):
    try:
        result = process_message(message)
        sink.write(result)
    except ValidationError as e:
        # Known bad data → DLQ
        dlq_producer.send('orders-dlq', {
            'original_message': message,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'pipeline': 'revenue-aggregation'
        })
    except Exception as e:
        # Unknown error → retry first, then DLQ
        raise  # Will be retried by consumer
```

### Pipeline Health Dashboard

```
┌────────────────────────────────────────────────────────┐
│ Pipeline: Revenue Aggregation                           │
├──────────────────┬─────────────────────────────────────┤
│ Input Lag        │ 42 messages (0.5s at current rate)  │
│ Throughput       │ 8,400 events/sec                     │
│ Error Rate       │ 0.01%                               │
│ DLQ Size         │ 0 messages                          │
│ Data Freshness   │ 8 seconds                           │
│ Checkpoints      │ Last: 45s ago, Success: 100%         │
│ Output Records   │ 2.4M in last hour                   │
└──────────────────┴─────────────────────────────────────┘
```

---

## 21. Quick Reference

### Batch vs Stream Decision Matrix

```
What is the latency requirement?
  └── Seconds to minutes → Stream Processing (Flink, Kafka Streams)
  └── Minutes to hours  → Micro-batch (Spark Streaming)
  └── Hours to days     → Batch (Spark, dbt)

How much historical data is processed each run?
  └── Incremental (new data only) → Stream / micro-batch
  └── Full reprocessing           → Batch

Is the query complex ML or deep aggregation?
  └── Yes → Batch (Spark ML, large Spark jobs)
  └── No  → Stream

Is the data source a queue (Kafka)?
  └── Yes → Stream processing is natural
  └── No (database, files) → Batch

Budget/complexity constraints?
  └── Low complexity → Lambda if already have both systems
  └── Prefer simplicity → Kappa (stream-only)
```

### Windowing Types Comparison

| Window | Size | Overlap | Trigger | Best For |
|---|---|---|---|---|
| Tumbling | Fixed | None | End of window | Hourly/daily reports |
| Sliding | Fixed | Yes | Configurable slide | Moving averages |
| Session | Variable | None | Inactivity gap | User sessions |
| Global | Unlimited | N/A | Custom (count/time) | Irregular batches |

### Stream Processing Cheat Sheet

```
Event time > Processing time (for accuracy)
Watermarks = max_seen_event_time - allowed_lateness
Late data → side output or allowedLateness
Exactly-once = checkpointing + transactional sink
State store = RocksDB (local, fast)
Backpressure = slow downstream → slow upstream naturally

Key metrics to monitor:
  - Consumer group lag (Kafka)
  - Checkpoint duration + success rate (Flink)
  - Data freshness (time since last output)
  - DLQ depth
  - Watermark lag
```

### Format Selection Guide

```
Kafka streaming between services → Avro + Schema Registry
Analytics queries on data lake → Parquet (columnar, compressed)
Hive/Hadoop ecosystem → ORC
gRPC inter-service → Protobuf
Simple API exchange → JSON
High-frequency time-series → Binary (Protobuf or Avro)
ML feature store → Parquet
```

### Architecture Decision Guide

```
New analytics pipeline:
  Data < 1TB, batch only → dbt + BigQuery/Snowflake (ELT)
  Data > 1TB, mixed → Spark + data lake + dbt
  Real-time dashboard → Kafka + Flink + ClickHouse
  Both batch + real-time → Lambda (mature stack) or Kappa (if Kafka retention OK)

CDC needed?
  PostgreSQL → Debezium (pgoutput plugin)
  MySQL → Debezium (binlog)
  MongoDB → Debezium (oplog)
  Oracle → Debezium (LogMiner) or GoldenGate

Data quality checks:
  Simple → Great Expectations in Airflow DAG
  Complex → dbt tests + Great Expectations
  Real-time → Flink assertions + DLQ
```
