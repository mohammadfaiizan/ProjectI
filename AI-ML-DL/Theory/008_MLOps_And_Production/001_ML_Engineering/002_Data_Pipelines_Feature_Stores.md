# Data Pipelines and Feature Stores

## Table of Contents

1. [Introduction to Data Pipelines](#introduction-to-data-pipelines)
2. [ETL vs ELT Architectures](#etl-vs-elt-architectures)
3. [Feature Engineering at Scale](#feature-engineering-at-scale)
4. [Feature Store Architecture](#feature-store-architecture)
5. [Data Versioning and Lineage](#data-versioning-and-lineage)
6. [Data Quality and Validation](#data-quality-and-validation)
7. [Batch vs Streaming Pipelines](#batch-vs-streaming-pipelines)
8. [Distributed Processing Frameworks](#distributed-processing-frameworks)
9. [Feature Serving Patterns](#feature-serving-patterns)
10. [Key Takeaways](#key-takeaways)

## Introduction to Data Pipelines

Data pipelines form the backbone of machine learning systems, responsible for ingesting, transforming, and serving data to both training and inference workloads. Unlike traditional data pipelines, ML data pipelines must handle the unique requirements of feature engineering, versioning, and serving at scale.

### Core Components

A typical ML data pipeline consists of:

- **Data Ingestion**: Collecting data from various sources (databases, APIs, files, streams)
- **Data Transformation**: Cleaning, normalizing, and engineering features
- **Data Storage**: Storing processed data in formats optimized for ML workloads
- **Data Validation**: Ensuring data quality and consistency
- **Data Serving**: Providing fast access to features for training and inference

### Pipeline Characteristics

ML data pipelines must exhibit:

- **Reproducibility**: Same inputs produce same outputs
- **Scalability**: Handle increasing data volumes efficiently
- **Reliability**: Gracefully handle failures and recover automatically
- **Observability**: Provide visibility into pipeline health and data quality
- **Versioning**: Track changes to data and transformations over time

### Pipeline Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw Data   │────▶│  Transform   │────▶│  Processed   │
│   Sources    │     │   Pipeline   │     │     Data     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ├──────────────┐
                                                  │              │
                                                  ▼              ▼
                                          ┌──────────────┐ ┌──────────────┐
                                          │   Feature   │ │   Training   │
                                          │    Store    │ │    Dataset   │
                                          └──────────────┘ └──────────────┘
```

## ETL vs ELT Architectures

### ETL (Extract, Transform, Load)

ETL pipelines transform data before loading into the target system:

**Process Flow**:
1. Extract data from source systems
2. Transform data in a staging area
3. Load transformed data into target system

**Advantages**:
- Data is cleaned and validated before storage
- Reduces storage costs by storing only processed data
- Better for structured transformations with clear schemas

**Disadvantages**:
- Requires upfront schema design
- Less flexible for exploratory analysis
- Transformation logic must be defined before loading

### ELT (Extract, Load, Transform)

ELT pipelines load raw data first, then transform on-demand:

**Process Flow**:
1. Extract data from source systems
2. Load raw data into target system
3. Transform data when needed

**Advantages**:
- More flexible for schema evolution
- Preserves raw data for reprocessing
- Better for cloud data warehouses with compute separation
- Enables exploratory analysis on raw data

**Disadvantages**:
- Higher storage costs
- Requires more compute resources for transformations
- Data quality issues may propagate

### Hybrid Approaches

Many modern systems use hybrid approaches:

```python
class HybridPipeline:
    def __init__(self):
        self.raw_data_lake = DataLake()
        self.processed_data_warehouse = DataWarehouse()
        self.feature_store = FeatureStore()
    
    def ingest(self, source_data):
        """Load raw data into data lake"""
        raw_path = self.raw_data_lake.store(source_data)
        return raw_path
    
    def transform(self, raw_path, transformation_config):
        """Transform data and store in warehouse"""
        raw_data = self.raw_data_lake.load(raw_path)
        transformed_data = self.apply_transformations(
            raw_data, transformation_config
        )
        processed_path = self.processed_data_warehouse.store(transformed_data)
        return processed_path
    
    def feature_engineering(self, processed_path, feature_config):
        """Engineer features and store in feature store"""
        processed_data = self.processed_data_warehouse.load(processed_path)
        features = self.compute_features(processed_data, feature_config)
        self.feature_store.ingest(features)
        return features
```

### When to Use ETL vs ELT

| Scenario | Recommended Approach | Reason |
|----------|---------------------|--------|
| Structured data with fixed schema | ETL | Clear transformation requirements |
| Exploratory data science | ELT | Need raw data for analysis |
| Real-time requirements | ETL | Pre-computed transformations reduce latency |
| Cloud data warehouse | ELT | Leverage warehouse compute |
| Compliance/audit needs | ELT | Preserve raw data for compliance |

## Feature Engineering at Scale

Feature engineering transforms raw data into features that machine learning models can effectively use. At scale, this requires careful consideration of computational efficiency, consistency, and maintainability.

### Feature Types

**Numerical Features**:
- Continuous values (age, price, temperature)
- Require normalization or standardization
- May need outlier handling

**Categorical Features**:
- Discrete values (country, product category)
- Require encoding (one-hot, label, target encoding)
- May have high cardinality issues

**Temporal Features**:
- Time-based features (day of week, hour, time since event)
- Require proper timezone handling
- May need cyclical encoding

**Text Features**:
- Natural language data
- Require tokenization, embedding, or TF-IDF
- May need preprocessing (lowercasing, stemming)

**Aggregated Features**:
- Statistical aggregations (mean, sum, count)
- Window-based features (rolling averages)
- Group-by aggregations

### Feature Engineering Patterns

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, avg, sum as spark_sum
from pyspark.sql.window import Window

class FeatureEngineeringPipeline:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def create_temporal_features(self, df):
        """Extract temporal features from timestamp column"""
        return df.withColumn("hour", hour(col("timestamp"))) \
                 .withColumn("day_of_week", dayofweek(col("timestamp"))) \
                 .withColumn("month", month(col("timestamp"))) \
                 .withColumn("is_weekend", when(
                     dayofweek(col("timestamp")).isin([1, 7]), 1
                 ).otherwise(0))
    
    def create_aggregated_features(self, df, group_cols, window_spec):
        """Create window-based aggregated features"""
        window = Window.partitionBy(group_cols).rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        
        return df.withColumn("cumulative_count", count("*").over(window)) \
                 .withColumn("rolling_avg", avg("value").over(window)) \
                 .withColumn("rolling_sum", spark_sum("value").over(window))
    
    def create_categorical_features(self, df, categorical_cols):
        """Encode categorical features"""
        from pyspark.ml.feature import StringIndexer, OneHotEncoder
        
        indexed_df = df
        for col_name in categorical_cols:
            indexer = StringIndexer(
                inputCol=col_name,
                outputCol=f"{col_name}_indexed"
            )
            indexed_df = indexer.fit(indexed_df).transform(indexed_df)
            
            encoder = OneHotEncoder(
                inputCol=f"{col_name}_indexed",
                outputCol=f"{col_name}_encoded"
            )
            indexed_df = encoder.fit(indexed_df).transform(indexed_df)
        
        return indexed_df
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features between pairs"""
        result_df = df
        for col1, col2 in feature_pairs:
            result_df = result_df.withColumn(
                f"{col1}_x_{col2}",
                col(col1) * col(col2)
            )
        return result_df
```

### Feature Engineering Best Practices

**Consistency**:
- Same transformations for training and inference
- Version control for feature definitions
- Automated testing of feature computation

**Performance**:
- Incremental computation for large datasets
- Caching frequently used features
- Parallel processing where possible

**Maintainability**:
- Modular feature definitions
- Clear documentation
- Reusable feature functions

**Quality**:
- Validate feature distributions
- Monitor for feature drift
- Handle missing values consistently

## Feature Store Architecture

Feature stores provide centralized storage and serving of features for both training and inference, ensuring consistency and reducing duplication of feature engineering logic.

### Core Components

**Offline Feature Store**:
- Stores historical features for training
- Optimized for batch reads
- Supports time-travel queries
- Typically uses data warehouses or data lakes

**Online Feature Store**:
- Serves features for real-time inference
- Optimized for low-latency lookups
- Supports point-in-time queries
- Typically uses key-value stores or specialized databases

**Feature Registry**:
- Metadata about features
- Feature definitions and schemas
- Lineage and dependencies
- Access control and governance

### Feature Store Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Feature Registry                        │
│    (Metadata, Schemas, Lineage, Access Control)         │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐         ┌──────────────┐
│   Offline    │         │    Online    │
│ Feature Store│         │ Feature Store│
│  (Training)  │         │  (Inference) │
└──────┬───────┘         └──────┬───────┘
       │                        │
       │                        │
       ▼                        ▼
┌──────────────┐         ┌──────────────┐
│   Training   │         │   Serving    │
│   Pipeline   │         │   Pipeline   │
└──────────────┘         └──────────────┘
```

### Feast Feature Store

Feast is an open-source feature store that provides both offline and online storage:

```python
from feast import FeatureStore, Entity, Feature, ValueType
from feast.data_source import FileSource
from datetime import timedelta

# Define entities
driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Driver identifier"
)

# Define data source
driver_stats_source = FileSource(
    name="driver_stats",
    path="s3://feast-bucket/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Define features
driver_features = [
    Feature(name="avg_daily_trips", dtype=ValueType.FLOAT),
    Feature(name="total_trips", dtype=ValueType.INT64),
    Feature(name="rating", dtype=ValueType.FLOAT)
]

# Create feature view
driver_stats_fv = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=365),
    features=driver_features,
    source=driver_stats_source
)

# Initialize feature store
store = FeatureStore(repo_path="./feature_repo")

# Register features
store.apply([driver, driver_stats_fv])

# Retrieve features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_stats:avg_daily_trips", "driver_stats:rating"]
).to_df()

# Retrieve features for inference
features = store.get_online_features(
    features=["driver_stats:avg_daily_trips", "driver_stats:rating"],
    entity_rows=[{"driver_id": 1234}]
).to_dict()
```

### Tecton Feature Store

Tecton provides a managed feature store with automatic backfilling and monitoring:

```python
from tecton import batch_feature_view, FeatureAggregation
from tecton.aggregation_functions import last, mean, count
from datetime import datetime, timedelta

@batch_feature_view(
    sources=[transactions],
    entities=[user],
    mode="spark_sql",
    aggregation_interval=timedelta(days=1),
    aggregations=[
        FeatureAggregation(column="amount", function=mean, time_window=timedelta(days=30)),
        FeatureAggregation(column="amount", function=last, time_window=timedelta(days=7)),
        FeatureAggregation(column="transaction_id", function=count, time_window=timedelta(days=30))
    ],
    batch_schedule=timedelta(days=1)
)
def user_transaction_features(transactions):
    return f"""
        SELECT
            user_id,
            timestamp,
            amount,
            transaction_id
        FROM {transactions}
    """

# Materialize features
user_transaction_features.materialize(
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 1, 31)
)

# Retrieve features
from tecton import FeatureService

transaction_feature_service = FeatureService(
    name="transaction_features",
    features=[user_transaction_features]
)

# Online serving
features = transaction_feature_service.get_online_features(
    join_keys={"user_id": "12345"}
).to_dict()
```

### Feature Store Comparison

| Feature | Feast | Tecton | AWS SageMaker Feature Store |
|---------|-------|--------|----------------------------|
| Deployment | Self-hosted | Managed | Managed |
| Offline Store | File-based, BigQuery, Redshift | Snowflake, Databricks | S3 |
| Online Store | Redis, DynamoDB | DynamoDB, Redis | DynamoDB |
| Backfilling | Manual | Automatic | Manual |
| Monitoring | Basic | Advanced | Integrated with SageMaker |
| Cost | Infrastructure costs | Per-feature pricing | Pay-per-use |

## Data Versioning and Lineage

Data versioning tracks changes to datasets over time, enabling reproducibility and rollback capabilities. Data lineage tracks the flow of data through transformations, providing visibility into data dependencies.

### Data Versioning with DVC

DVC (Data Version Control) provides Git-like versioning for data:

```python
# Initialize DVC repository
# dvc init

# Add data file to version control
# dvc add data/raw/dataset.csv

# Commit to Git
# git add data/raw/dataset.csv.dvc .gitignore
# git commit -m "Add dataset version 1.0"

# Create new version
# dvc add data/raw/dataset.csv
# git add data/raw/dataset.csv.dvc
# git commit -m "Update dataset to version 2.0"

# Checkout specific version
# git checkout <commit-hash>
# dvc checkout
```

### DVC Pipelines

DVC supports pipeline definitions for reproducible data processing:

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python scripts/prepare.py data/raw data/prepared
    deps:
      - scripts/prepare.py
      - data/raw
    outs:
      - data/prepared
    metrics:
      - metrics/prepare.json
  
  train:
    cmd: python scripts/train.py data/prepared models/model.pkl
    deps:
      - scripts/train.py
      - data/prepared
    outs:
      - models/model.pkl
    metrics:
      - metrics/train.json
    params:
      - train.learning_rate
      - train.epochs
```

### Data Lineage Tracking

Data lineage tracks data flow through transformations:

```python
from datetime import datetime
from typing import List, Dict, Optional

class DataLineageTracker:
    def __init__(self):
        self.lineage_graph = {}
    
    def register_transformation(
        self,
        output_dataset: str,
        input_datasets: List[str],
        transformation_code: str,
        metadata: Optional[Dict] = None
    ):
        """Register a transformation in the lineage graph"""
        self.lineage_graph[output_dataset] = {
            "inputs": input_datasets,
            "transformation": transformation_code,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
    
    def get_lineage(self, dataset: str) -> Dict:
        """Get full lineage for a dataset"""
        if dataset not in self.lineage_graph:
            return {"dataset": dataset, "lineage": []}
        
        node = self.lineage_graph[dataset]
        lineage = [node]
        
        for input_dataset in node["inputs"]:
            input_lineage = self.get_lineage(input_dataset)
            lineage.extend(input_lineage["lineage"])
        
        return {"dataset": dataset, "lineage": lineage}
    
    def get_downstream(self, dataset: str) -> List[str]:
        """Get all datasets that depend on this dataset"""
        downstream = []
        for output_dataset, node in self.lineage_graph.items():
            if dataset in node["inputs"]:
                downstream.append(output_dataset)
                downstream.extend(self.get_downstream(output_dataset))
        return list(set(downstream))

# Usage
tracker = DataLineageTracker()
tracker.register_transformation(
    output_dataset="features/user_features",
    input_datasets=["raw/user_data", "raw/transaction_data"],
    transformation_code="scripts/feature_engineering.py",
    metadata={"version": "1.0", "author": "data_team"}
)

lineage = tracker.get_lineage("features/user_features")
```

### Versioning Strategies

**Snapshot Versioning**:
- Store complete copies of datasets at each version
- Simple but storage-intensive
- Good for small datasets

**Delta Versioning**:
- Store only changes between versions
- Storage-efficient for large datasets
- Requires reconstruction logic

**Timestamp-based Versioning**:
- Version by timestamp or date
- Natural for time-series data
- Easy to query historical versions

## Data Quality and Validation

Data quality validation ensures that data meets expected standards before being used in ML pipelines. This includes schema validation, statistical checks, and business rule validation.

### Schema Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class UserSchema(BaseModel):
    user_id: int = Field(..., gt=0, description="Positive user ID")
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)
    registration_date: datetime
    is_active: bool
    
    @validator('age')
    def validate_age(cls, v):
        if v < 18:
            raise ValueError('User must be at least 18 years old')
        return v

def validate_schema(data: List[dict], schema_class):
    """Validate data against schema"""
    errors = []
    for i, record in enumerate(data):
        try:
            schema_class(**record)
        except Exception as e:
            errors.append({
                "record_index": i,
                "record": record,
                "error": str(e)
            })
    return errors
```

### Statistical Validation

```python
import numpy as np
import pandas as pd
from scipy import stats

class StatisticalValidator:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_stats = self.compute_statistics(reference_data)
    
    def compute_statistics(self, df: pd.DataFrame) -> dict:
        """Compute reference statistics"""
        stats_dict = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            stats_dict[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
                "q25": df[col].quantile(0.25),
                "q75": df[col].quantile(0.75)
            }
        return stats_dict
    
    def validate_distribution(self, current_data: pd.DataFrame, 
                             threshold: float = 0.05) -> dict:
        """Validate distribution using Kolmogorov-Smirnov test"""
        results = {}
        for col, ref_stats in self.reference_stats.items():
            if col in current_data.columns:
                ks_statistic, p_value = stats.ks_2samp(
                    self.reference_data[col],
                    current_data[col]
                )
                results[col] = {
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                    "drift_detected": p_value < threshold
                }
        return results
    
    def validate_bounds(self, current_data: pd.DataFrame) -> dict:
        """Validate that values are within expected bounds"""
        results = {}
        for col, ref_stats in self.reference_stats.items():
            if col in current_data.columns:
                out_of_bounds = (
                    (current_data[col] < ref_stats["min"]) |
                    (current_data[col] > ref_stats["max"])
                )
                results[col] = {
                    "out_of_bounds_count": out_of_bounds.sum(),
                    "out_of_bounds_pct": out_of_bounds.mean() * 100
                }
        return results
```

### Data Quality Checks

```python
class DataQualityChecker:
    def __init__(self):
        self.checks = []
    
    def add_check(self, name: str, check_function, threshold: float = None):
        """Add a data quality check"""
        self.checks.append({
            "name": name,
            "function": check_function,
            "threshold": threshold
        })
    
    def run_checks(self, data: pd.DataFrame) -> dict:
        """Run all quality checks"""
        results = {}
        for check in self.checks:
            result = check["function"](data)
            passed = True
            if check["threshold"] is not None:
                passed = result <= check["threshold"]
            
            results[check["name"]] = {
                "result": result,
                "passed": passed,
                "threshold": check["threshold"]
            }
        return results

# Example usage
checker = DataQualityChecker()

# Check for missing values
checker.add_check(
    "missing_values",
    lambda df: df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
    threshold=0.05
)

# Check for duplicates
checker.add_check(
    "duplicates",
    lambda df: df.duplicated().sum() / len(df),
    threshold=0.01
)

# Check for negative values in positive columns
checker.add_check(
    "negative_values",
    lambda df: (df.select_dtypes(include=[np.number]) < 0).sum().sum(),
    threshold=0
)

results = checker.run_checks(data)
```

## Batch vs Streaming Pipelines

ML pipelines can process data in batch mode (periodic processing of accumulated data) or streaming mode (continuous processing of data as it arrives).

### Batch Processing

Batch processing accumulates data over a time period and processes it together:

**Characteristics**:
- Periodic execution (hourly, daily, weekly)
- Processes large volumes efficiently
- Simpler error handling and recovery
- Lower infrastructure complexity

**Use Cases**:
- Historical feature computation
- Model retraining
- Reporting and analytics
- Data warehouse ETL

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def process_batch_data(**context):
    """Process batch data"""
    execution_date = context['execution_date']
    
    # Load data for the batch period
    start_date = execution_date - timedelta(days=1)
    end_date = execution_date
    
    data = load_data_from_source(start_date, end_date)
    
    # Transform data
    transformed_data = transform_data(data)
    
    # Store results
    store_results(transformed_data, execution_date)

dag = DAG(
    'batch_feature_pipeline',
    default_args={
        'owner': 'data_team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'retries': 3,
        'retry_delay': timedelta(minutes=5)
    },
    schedule_interval='@daily',
    catchup=False
)

process_task = PythonOperator(
    task_id='process_batch_data',
    python_callable=process_batch_data,
    dag=dag
)
```

### Streaming Processing

Streaming processing handles data continuously as it arrives:

**Characteristics**:
- Real-time or near-real-time processing
- Low latency requirements
- More complex error handling
- Higher infrastructure complexity

**Use Cases**:
- Real-time feature computation
- Online inference
- Fraud detection
- Anomaly detection

```python
from kafka import KafkaConsumer
from kafka import KafkaProducer
import json

class StreamingFeaturePipeline:
    def __init__(self, input_topic, output_topic):
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def process_stream(self):
        """Process streaming data"""
        for message in self.consumer:
            try:
                raw_data = message.value
                
                # Transform data
                features = self.compute_features(raw_data)
                
                # Validate features
                if self.validate_features(features):
                    # Send to output topic
                    self.producer.send(
                        self.output_topic,
                        value=features,
                        key=raw_data['user_id'].encode('utf-8')
                    )
                else:
                    # Send to dead letter queue
                    self.send_to_dlq(raw_data, "validation_failed")
            
            except Exception as e:
                # Handle errors
                self.handle_error(message, e)
    
    def compute_features(self, data):
        """Compute features from raw data"""
        # Feature computation logic
        return {
            "user_id": data["user_id"],
            "feature_1": self.calculate_feature_1(data),
            "feature_2": self.calculate_feature_2(data),
            "timestamp": datetime.now().isoformat()
        }
```

### Lambda Architecture

Lambda architecture combines batch and streaming processing:

```
┌──────────────┐
│   Data       │
│   Source     │
└──────┬───────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│   Batch      │ │  Streaming   │
│   Layer      │ │    Layer     │
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                │
                ▼
         ┌──────────────┐
         │   Serving    │
         │    Layer     │
         └──────────────┘
```

## Distributed Processing Frameworks

Large-scale ML pipelines require distributed processing frameworks to handle data volumes and computational requirements.

### Apache Spark

Apache Spark provides distributed data processing with in-memory computing:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, sum as spark_sum
from pyspark.sql.window import Window

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Read data
df = spark.read.parquet("s3://bucket/raw_data/")

# Transform data
window_spec = Window.partitionBy("user_id").orderBy("timestamp") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

features_df = df \
    .withColumn("cumulative_sum", spark_sum("amount").over(window_spec)) \
    .withColumn("rolling_avg", avg("amount").over(
        window_spec.rowsBetween(-7, 0)
    )) \
    .withColumn("days_since_last_transaction", 
        datediff("timestamp", lag("timestamp", 1).over(window_spec))
    )

# Write results
features_df.write \
    .mode("overwrite") \
    .parquet("s3://bucket/features/")
```

### Apache Beam

Apache Beam provides a unified programming model for batch and streaming:

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class ComputeFeatures(beam.DoFn):
    def process(self, element):
        """Compute features for each element"""
        user_id = element['user_id']
        transactions = element['transactions']
        
        features = {
            'user_id': user_id,
            'total_amount': sum(t['amount'] for t in transactions),
            'transaction_count': len(transactions),
            'avg_amount': sum(t['amount'] for t in transactions) / len(transactions),
            'last_transaction_date': max(t['date'] for t in transactions)
        }
        
        yield features

# Define pipeline
options = PipelineOptions([
    '--runner=DataflowRunner',
    '--project=my-project',
    '--region=us-central1'
])

with beam.Pipeline(options=options) as pipeline:
    (pipeline
     | 'ReadTransactions' >> beam.io.ReadFromBigQuery(
         query='SELECT * FROM transactions',
         use_standard_sql=True
     )
     | 'GroupByUser' >> beam.GroupByKey()
     | 'ComputeFeatures' >> beam.ParDo(ComputeFeatures())
     | 'WriteFeatures' >> beam.io.WriteToBigQuery(
         table='features.user_features',
         write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
     ))
```

### Framework Comparison

| Framework | Batch | Streaming | Language Support | Use Case |
|-----------|-------|-----------|------------------|----------|
| Apache Spark | Yes | Yes | Python, Scala, Java, R | General-purpose data processing |
| Apache Beam | Yes | Yes | Python, Java, Go | Unified batch/streaming |
| Flink | Yes | Yes | Java, Scala, Python | Low-latency streaming |
| Dask | Yes | Limited | Python | Python-native distributed computing |

## Feature Serving Patterns

Feature serving provides low-latency access to features for real-time inference. Different patterns optimize for different requirements.

### Precomputed Feature Serving

Features are computed offline and served from a fast lookup store:

```python
import redis
import json
from datetime import datetime

class PrecomputedFeatureServer:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get_features(self, entity_id: str, feature_names: List[str]) -> dict:
        """Retrieve precomputed features"""
        features = {}
        for feature_name in feature_names:
            key = f"features:{entity_id}:{feature_name}"
            value = self.redis.get(key)
            if value:
                features[feature_name] = json.loads(value)
        return features
    
    def update_features(self, entity_id: str, feature_updates: dict):
        """Update features in cache"""
        pipe = self.redis.pipeline()
        for feature_name, feature_value in feature_updates.items():
            key = f"features:{entity_id}:{feature_name}"
            pipe.setex(
                key,
                3600,  # TTL in seconds
                json.dumps(feature_value)
            )
        pipe.execute()
```

### On-Demand Feature Computation

Features are computed on-demand during inference:

```python
class OnDemandFeatureServer:
    def __init__(self, feature_computers: dict):
        self.feature_computers = feature_computers
        self.cache = {}
    
    def get_features(self, entity_id: str, feature_names: List[str], 
                    context: dict) -> dict:
        """Compute features on-demand"""
        features = {}
        for feature_name in feature_names:
            # Check cache first
            cache_key = f"{entity_id}:{feature_name}"
            if cache_key in self.cache:
                features[feature_name] = self.cache[cache_key]
            else:
                # Compute feature
                computer = self.feature_computers[feature_name]
                feature_value = computer.compute(entity_id, context)
                features[feature_name] = feature_value
                self.cache[cache_key] = feature_value
        return features
```

### Hybrid Feature Serving

Combines precomputed and on-demand features:

```python
class HybridFeatureServer:
    def __init__(self, precomputed_store, on_demand_computers):
        self.precomputed_store = precomputed_store
        self.on_demand_computers = on_demand_computers
    
    def get_features(self, entity_id: str, feature_config: dict) -> dict:
        """Get features using hybrid approach"""
        features = {}
        
        # Get precomputed features
        precomputed_features = self.precomputed_store.get_features(
            entity_id,
            feature_config.get("precomputed", [])
        )
        features.update(precomputed_features)
        
        # Compute on-demand features
        for feature_name in feature_config.get("on_demand", []):
            computer = self.on_demand_computers[feature_name]
            features[feature_name] = computer.compute(entity_id)
        
        return features
```

### Feature Serving Best Practices

**Latency Optimization**:
- Cache frequently accessed features
- Use fast lookup stores (Redis, DynamoDB)
- Minimize network round trips
- Batch feature requests when possible

**Consistency**:
- Ensure feature computation matches training
- Version feature definitions
- Handle missing features gracefully

**Scalability**:
- Distribute feature serving across multiple instances
- Use connection pooling
- Implement rate limiting

## Key Takeaways

- Data pipelines are critical infrastructure for ML systems, requiring careful design for reproducibility, scalability, and reliability
- ETL vs ELT choice depends on data characteristics, use case requirements, and infrastructure capabilities
- Feature engineering at scale requires distributed processing, consistent transformations, and proper versioning
- Feature stores provide centralized feature management, ensuring consistency between training and inference
- Data versioning with tools like DVC enables reproducibility and rollback capabilities
- Data lineage tracking provides visibility into data dependencies and transformation flows
- Data quality validation is essential for maintaining model performance and preventing production issues
- Batch processing suits periodic, high-volume workloads while streaming enables real-time feature computation
- Distributed frameworks like Spark and Beam enable scalable feature engineering and data processing
- Feature serving patterns must balance latency, consistency, and computational costs
- Hybrid approaches combining batch and streaming processing provide flexibility for diverse ML workloads
