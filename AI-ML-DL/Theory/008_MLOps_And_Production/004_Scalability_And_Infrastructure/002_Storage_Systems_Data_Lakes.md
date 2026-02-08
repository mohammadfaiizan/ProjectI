# Storage Systems and Data Lakes

## Table of Contents

1. [Introduction to Data Storage](#introduction-to-data-storage)
2. [Data Lakes Architecture](#data-lakes-architecture)
3. [Data Warehouses](#data-warehouses)
4. [Lakehouse Architecture](#lakehouse-architecture)
5. [Parquet and Arrow Formats](#parquet-and-arrow-formats)
6. [Metadata Management](#metadata-management)
7. [Data Catalog](#data-catalog)
8. [Storage Optimization](#storage-optimization)
9. [Data Lifecycle Management](#data-lifecycle-management)
10. [Key Takeaways](#key-takeaways)

## Introduction to Data Storage

ML systems require efficient storage solutions for:

- **Training Data**: Large datasets for model training
- **Features**: Computed features for training and serving
- **Models**: Model artifacts and checkpoints
- **Logs**: Prediction logs and monitoring data
- **Metadata**: Data lineage, schemas, and governance

### Storage Requirements

- **Scalability**: Handle petabytes of data
- **Performance**: Fast reads for training, low latency for serving
- **Cost Efficiency**: Optimize storage costs
- **Durability**: Ensure data reliability
- **Access Patterns**: Support batch and streaming access

## Data Lakes Architecture

### Data Lake Layers

```
┌─────────────────────────────────────────┐
│         Bronze Layer (Raw)              │
│  - Raw ingested data                   │
│  - No transformation                   │
│  - Preserve original format            │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Silver Layer (Cleaned)         │
│  - Cleaned and validated data          │
│  - Standardized schemas                │
│  - Data quality checks                 │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         Gold Layer (Curated)            │
│  - Business-ready data                  │
│  - Aggregated and enriched              │
│  - Optimized for consumption            │
└─────────────────────────────────────────┘
```

### AWS S3 Data Lake

```python
import boto3
import pandas as pd

class S3DataLake:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
    
    def ingest_raw_data(self, local_path, s3_key):
        """Ingest raw data to bronze layer"""
        bronze_key = f"bronze/{s3_key}"
        self.s3.upload_file(local_path, self.bucket_name, bronze_key)
        return bronze_key
    
    def process_to_silver(self, bronze_key, silver_key):
        """Process data to silver layer"""
        # Download from bronze
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=bronze_key)
        df = pd.read_parquet(obj['Body'])
        
        # Clean and validate
        df_cleaned = self.clean_data(df)
        
        # Upload to silver
        silver_path = f"silver/{silver_key}"
        buffer = io.BytesIO()
        df_cleaned.to_parquet(buffer, index=False)
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=silver_path,
            Body=buffer.getvalue()
        )
        return silver_path
    
    def aggregate_to_gold(self, silver_keys, gold_key):
        """Aggregate to gold layer"""
        dfs = []
        for key in silver_keys:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            df = pd.read_parquet(obj['Body'])
            dfs.append(df)
        
        # Aggregate
        aggregated = pd.concat(dfs).groupby('key').agg({
            'value': 'sum',
            'count': 'sum'
        }).reset_index()
        
        # Upload to gold
        gold_path = f"gold/{gold_key}"
        buffer = io.BytesIO()
        aggregated.to_parquet(buffer, index=False)
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=gold_path,
            Body=buffer.getvalue()
        )
        return gold_path
```

### Azure Data Lake Storage

```python
from azure.storage.filedatalake import DataLakeServiceClient

class AzureDataLake:
    def __init__(self, account_name, account_key):
        self.service_client = DataLakeServiceClient(
            account_url=f"https://{account_name}.dfs.core.windows.net",
            credential=account_key
        )
        self.file_system_client = self.service_client.get_file_system_client(
            file_system="datalake"
        )
    
    def upload_file(self, local_path, lake_path):
        """Upload file to data lake"""
        file_client = self.file_system_client.get_file_client(lake_path)
        
        with open(local_path, 'rb') as f:
            file_client.upload_data(f, overwrite=True)
    
    def read_parquet(self, lake_path):
        """Read Parquet file from data lake"""
        file_client = self.file_system_client.get_file_client(lake_path)
        data = file_client.download_file().readall()
        
        return pd.read_parquet(io.BytesIO(data))
```

## Data Warehouses

### Snowflake

```python
import snowflake.connector

class SnowflakeWarehouse:
    def __init__(self, account, user, password, warehouse, database):
        self.conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database
        )
    
    def query(self, sql):
        """Execute SQL query"""
        cursor = self.conn.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def load_from_s3(self, s3_path, table_name):
        """Load data from S3"""
        sql = f"""
        COPY INTO {table_name}
        FROM '{s3_path}'
        FILE_FORMAT = (TYPE = 'PARQUET')
        """
        self.query(sql)
```

### BigQuery

```python
from google.cloud import bigquery

class BigQueryWarehouse:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)
    
    def query(self, sql):
        """Execute SQL query"""
        query_job = self.client.query(sql)
        return query_job.to_dataframe()
    
    def load_from_gcs(self, gcs_path, table_id):
        """Load data from GCS"""
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET
        )
        
        uri = f"gs://{gcs_path}"
        load_job = self.client.load_table_from_uri(
            uri, table_id, job_config=job_config
        )
        load_job.result()
```

## Lakehouse Architecture

Lakehouse combines data lake and data warehouse benefits.

### Delta Lake

```python
from delta import DeltaTable

class DeltaLakehouse:
    def __init__(self, spark):
        self.spark = spark
    
    def create_delta_table(self, df, table_path):
        """Create Delta table"""
        df.write.format("delta").mode("overwrite").save(table_path)
    
    def read_delta_table(self, table_path):
        """Read Delta table"""
        return self.spark.read.format("delta").load(table_path)
    
    def update_delta_table(self, table_path, updates_df):
        """Update Delta table"""
        delta_table = DeltaTable.forPath(self.spark, table_path)
        
        delta_table.alias("target").merge(
            updates_df.alias("updates"),
            "target.id = updates.id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    
    def time_travel(self, table_path, version):
        """Time travel to previous version"""
        return self.spark.read.format("delta").option(
            "versionAsOf", version
        ).load(table_path)
```

### Apache Iceberg

```python
class IcebergLakehouse:
    def __init__(self, spark):
        self.spark = spark
        self.spark.conf.set("spark.sql.catalog.spark_catalog", 
                           "org.apache.iceberg.spark.SparkCatalog")
    
    def create_iceberg_table(self, df, table_name):
        """Create Iceberg table"""
        df.writeTo(table_name).create()
    
    def read_iceberg_table(self, table_name):
        """Read Iceberg table"""
        return self.spark.table(table_name)
    
    def snapshot_query(self, table_name, snapshot_id):
        """Query specific snapshot"""
        return self.spark.read \
            .option("snapshot-id", snapshot_id) \
            .table(table_name)
```

## Parquet and Arrow Formats

### Parquet Format

```python
import pyarrow.parquet as pq
import pyarrow as pa

class ParquetStorage:
    def __init__(self):
        pass
    
    def write_parquet(self, df, path, compression='snappy'):
        """Write DataFrame to Parquet"""
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            path,
            compression=compression,
            use_dictionary=True,
            write_statistics=True
        )
    
    def read_parquet(self, path, columns=None):
        """Read Parquet file"""
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()
    
    def read_partitioned_parquet(self, base_path):
        """Read partitioned Parquet dataset"""
        dataset = pq.ParquetDataset(base_path)
        return dataset.read().to_pandas()
    
    def optimize_parquet(self, input_path, output_path):
        """Optimize Parquet file"""
        table = pq.read_table(input_path)
        
        # Repartition for optimal file size
        table = table.repartition(10)
        
        # Write with compression
        pq.write_table(
            table,
            output_path,
            compression='zstd',
            row_group_size=1000000
        )
```

### Apache Arrow

```python
import pyarrow as pa
import pyarrow.compute as pc

class ArrowStorage:
    def __init__(self):
        pass
    
    def create_arrow_table(self, data):
        """Create Arrow table"""
        return pa.table(data)
    
    def filter_arrow_table(self, table, condition):
        """Filter Arrow table"""
        return pc.filter(table, condition)
    
    def aggregate_arrow_table(self, table, group_by, aggregations):
        """Aggregate Arrow table"""
        return table.group_by(group_by).aggregate(aggregations)
    
    def convert_to_pandas(self, table):
        """Convert Arrow table to Pandas"""
        return table.to_pandas()
```

## Metadata Management

### Schema Registry

```python
class SchemaRegistry:
    def __init__(self):
        self.schemas = {}
    
    def register_schema(self, schema_name, schema_definition):
        """Register schema"""
        self.schemas[schema_name] = {
            'definition': schema_definition,
            'version': self.get_next_version(schema_name),
            'created_at': time.time()
        }
    
    def get_schema(self, schema_name, version=None):
        """Get schema"""
        if version:
            return self.schemas.get(f"{schema_name}_v{version}")
        return self.schemas.get(schema_name)
    
    def validate_data(self, data, schema_name):
        """Validate data against schema"""
        schema = self.get_schema(schema_name)
        # Validation logic
        return True
```

### Data Lineage

```python
class DataLineage:
    def __init__(self):
        self.lineage_graph = {}
    
    def add_lineage(self, source, target, transformation):
        """Add lineage relationship"""
        if target not in self.lineage_graph:
            self.lineage_graph[target] = []
        
        self.lineage_graph[target].append({
            'source': source,
            'transformation': transformation,
            'timestamp': time.time()
        })
    
    def get_lineage(self, dataset):
        """Get lineage for dataset"""
        return self.lineage_graph.get(dataset, [])
    
    def trace_upstream(self, dataset):
        """Trace all upstream dependencies"""
        upstream = set()
        to_process = [dataset]
        
        while to_process:
            current = to_process.pop()
            lineage = self.get_lineage(current)
            for item in lineage:
                upstream.add(item['source'])
                to_process.append(item['source'])
        
        return upstream
```

## Data Catalog

### Catalog Implementation

```python
class DataCatalog:
    def __init__(self):
        self.catalog = {}
    
    def register_dataset(self, name, metadata):
        """Register dataset in catalog"""
        self.catalog[name] = {
            'name': name,
            'metadata': metadata,
            'registered_at': time.time(),
            'tags': [],
            'owners': []
        }
    
    def search_datasets(self, query, filters=None):
        """Search datasets"""
        results = []
        for name, entry in self.catalog.items():
            if query.lower() in name.lower() or \
               query.lower() in str(entry['metadata']).lower():
                if filters:
                    if self.matches_filters(entry, filters):
                        results.append(entry)
                else:
                    results.append(entry)
        return results
    
    def add_tags(self, dataset_name, tags):
        """Add tags to dataset"""
        if dataset_name in self.catalog:
            self.catalog[dataset_name]['tags'].extend(tags)
    
    def get_dataset_info(self, dataset_name):
        """Get dataset information"""
        return self.catalog.get(dataset_name)
```

### Apache Atlas Integration

```python
from atlasclient.client import Atlas

class AtlasCatalog:
    def __init__(self, atlas_url, username, password):
        self.client = Atlas(atlas_url, username=username, password=password)
    
    def create_entity(self, entity_type, entity_data):
        """Create entity in Atlas"""
        entity = {
            'typeName': entity_type,
            'attributes': entity_data
        }
        return self.client.entity_post.create(data={'entity': entity})
    
    def search_entities(self, query):
        """Search entities"""
        return self.client.search_basic(query)
    
    def get_lineage(self, entity_guid):
        """Get lineage for entity"""
        return self.client.entity_guid.get_lineage(entity_guid)
```

## Storage Optimization

### Partitioning

```python
class PartitionedStorage:
    def __init__(self):
        pass
    
    def write_partitioned(self, df, base_path, partition_cols):
        """Write partitioned data"""
        df.write.partitionBy(partition_cols).parquet(base_path)
    
    def read_partitioned(self, base_path, partition_filters=None):
        """Read partitioned data with filters"""
        if partition_filters:
            return spark.read.parquet(base_path).filter(partition_filters)
        return spark.read.parquet(base_path)
```

### Compression

```python
class CompressionOptimizer:
    def __init__(self):
        self.compression_formats = {
            'snappy': {'ratio': 1.5, 'speed': 'fast'},
            'gzip': {'ratio': 2.5, 'speed': 'medium'},
            'zstd': {'ratio': 3.0, 'speed': 'fast'},
            'lz4': {'ratio': 2.0, 'speed': 'very_fast'}
        }
    
    def choose_compression(self, use_case='balanced'):
        """Choose compression based on use case"""
        if use_case == 'speed':
            return 'lz4'
        elif use_case == 'size':
            return 'zstd'
        else:
            return 'snappy'
```

### Caching

```python
class StorageCache:
    def __init__(self, cache_size_gb=100):
        self.cache = {}
        self.cache_size = cache_size_gb * 1024 * 1024 * 1024
        self.current_size = 0
    
    def get(self, key):
        """Get from cache"""
        if key in self.cache:
            return self.cache[key]['data']
        return None
    
    def put(self, key, data):
        """Put in cache"""
        data_size = len(str(data))
        
        # Evict if needed
        while self.current_size + data_size > self.cache_size:
            self.evict_oldest()
        
        self.cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'size': data_size
        }
        self.current_size += data_size
    
    def evict_oldest(self):
        """Evict oldest entry"""
        if self.cache:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            self.current_size -= self.cache[oldest_key]['size']
            del self.cache[oldest_key]
```

## Data Lifecycle Management

### Lifecycle Policies

```python
class DataLifecycleManager:
    def __init__(self):
        self.policies = {}
    
    def create_policy(self, policy_name, rules):
        """Create lifecycle policy"""
        self.policies[policy_name] = {
            'rules': rules,
            'created_at': time.time()
        }
    
    def apply_policy(self, policy_name, dataset_path):
        """Apply lifecycle policy"""
        policy = self.policies[policy_name]
        
        for rule in policy['rules']:
            if self.should_apply_rule(dataset_path, rule):
                self.apply_rule(dataset_path, rule)
    
    def should_apply_rule(self, dataset_path, rule):
        """Check if rule should be applied"""
        dataset_age = self.get_dataset_age(dataset_path)
        return dataset_age > rule['age_days']
    
    def apply_rule(self, dataset_path, rule):
        """Apply rule action"""
        action = rule['action']
        
        if action == 'archive':
            self.archive_dataset(dataset_path)
        elif action == 'delete':
            self.delete_dataset(dataset_path)
        elif action == 'move_to_cold_storage':
            self.move_to_cold_storage(dataset_path)
```

### Tiered Storage

```python
class TieredStorage:
    def __init__(self):
        self.tiers = {
            'hot': {'cost': 0.023, 'access_time': 'ms'},
            'warm': {'cost': 0.012, 'access_time': 'seconds'},
            'cold': {'cost': 0.004, 'access_time': 'minutes'},
            'archive': {'cost': 0.00099, 'access_time': 'hours'}
        }
    
    def move_to_tier(self, dataset_path, tier):
        """Move dataset to storage tier"""
        # Implementation
        pass
    
    def get_from_tier(self, dataset_path, tier):
        """Retrieve dataset from tier"""
        # Implementation
        pass
```

## Key Takeaways

- Data lakes provide scalable storage for raw, cleaned, and curated data
- Data warehouses offer optimized storage for analytical queries
- Lakehouse architecture combines benefits of both lakes and warehouses
- Parquet and Arrow formats provide efficient columnar storage
- Metadata management tracks schemas, lineage, and governance
- Data catalogs enable discovery and understanding of datasets
- Storage optimization through partitioning, compression, and caching improves performance
- Data lifecycle management automates data retention and archival
- Tiered storage optimizes costs based on access patterns
- Effective storage design balances performance, cost, and accessibility
