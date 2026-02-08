# Batch vs Real-Time Inference

## Table of Contents

1. [Introduction to Inference Types](#introduction-to-inference-types)
2. [Batch Processing Architecture](#batch-processing-architecture)
3. [Streaming Inference Architecture](#streaming-inference-architecture)
4. [Latency Requirements](#latency-requirements)
5. [Throughput Optimization](#throughput-optimization)
6. [Async Processing](#async-processing)
7. [Message Queues](#message-queues)
8. [Hybrid Approaches](#hybrid-approaches)
9. [Cost Considerations](#cost-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Inference Types

ML inference can be categorized into batch and real-time processing, each serving different use cases with distinct requirements.

### Batch Inference

**Characteristics**:
- Processes large volumes of data offline
- Latency measured in minutes to hours
- High throughput for bulk operations
- Cost-effective for large-scale processing
- Suitable for non-interactive use cases

**Use Cases**:
- Generating daily reports
- Batch scoring of historical data
- ETL pipelines
- Model evaluation on test sets
- Offline feature computation

### Real-Time Inference

**Characteristics**:
- Processes individual requests immediately
- Latency measured in milliseconds to seconds
- Variable throughput based on traffic
- Higher cost per prediction
- Suitable for interactive applications

**Use Cases**:
- Recommendation systems
- Fraud detection
- Real-time personalization
- Chatbots and virtual assistants
- Anomaly detection

### Comparison

| Aspect | Batch Inference | Real-Time Inference |
|--------|-----------------|-------------------|
| Latency | Minutes to hours | Milliseconds to seconds |
| Throughput | Very high | Variable |
| Cost per Prediction | Low | Higher |
| Infrastructure | Distributed systems | Stateless services |
| Use Case | Offline processing | Interactive applications |

## Batch Processing Architecture

### Traditional Batch Processing

```
┌─────────────┐
│   Data      │
│   Source    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Extract   │
│   Data      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Batch     │
│  Inference  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Store     │
│  Results    │
└─────────────┘
```

### Spark-Based Batch Processing

```python
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize Spark
spark = SparkSession.builder \
    .appName("BatchInference") \
    .getOrCreate()

# Load data
df = spark.read.parquet("s3://data-lake/input/")

# Load model
model = PipelineModel.load("s3://models/classification_model")

# Batch prediction
predictions = model.transform(df)

# Save results
predictions.select("id", "prediction", "probability") \
    .write \
    .mode("overwrite") \
    .parquet("s3://data-lake/predictions/")
```

### Distributed Batch Processing

```python
import multiprocessing as mp
import pandas as pd
import pickle

def process_chunk(chunk_data):
    """Process a chunk of data"""
    model = pickle.load(open('model.pkl', 'rb'))
    predictions = model.predict(chunk_data)
    return predictions

def batch_predict_parallel(input_file, output_file, num_workers=4):
    # Read data
    df = pd.read_csv(input_file)
    
    # Split into chunks
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process in parallel
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Combine results
    all_predictions = []
    for result in results:
        all_predictions.extend(result)
    
    # Save
    df['prediction'] = all_predictions
    df.to_csv(output_file, index=False)
```

### Scheduled Batch Jobs

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_batch_inference():
    # Load data
    df = pd.read_parquet("s3://data/input/")
    
    # Load model
    model = load_model("s3://models/latest/")
    
    # Predict
    df['prediction'] = model.predict(df[features])
    
    # Save results
    df.to_parquet("s3://data/predictions/", partition_cols=['date'])

dag = DAG(
    'batch_inference',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1)
)

batch_task = PythonOperator(
    task_id='batch_inference',
    python_callable=run_batch_inference,
    dag=dag
)
```

## Streaming Inference Architecture

### Real-Time Processing Pipeline

```
┌─────────────┐
│   Event     │
│   Stream    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Message   │
│   Queue     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Inference  │     │  Inference  │     │  Inference  │
│   Worker 1  │     │   Worker 2  │     │   Worker N  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                  ┌────────▼────────┐
                  │   Results       │
                  │   Store/Stream  │
                  └─────────────────┘
```

### Kafka-Based Streaming

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Consumer
consumer = KafkaConsumer(
    'input-topic',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Process stream
for message in consumer:
    data = message.value
    features = data['features']
    
    # Predict
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0].tolist()
    
    # Send result
    result = {
        'id': data['id'],
        'prediction': float(prediction),
        'probabilities': probability,
        'timestamp': data['timestamp']
    }
    producer.send('output-topic', value=result)
```

### Apache Flink Streaming

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

env = StreamExecutionEnvironment.get_execution_environment()
table_env = StreamTableEnvironment.create(env)

# Define source
table_env.execute_sql("""
    CREATE TABLE input_stream (
        id STRING,
        features ARRAY<DOUBLE>,
        ts TIMESTAMP(3)
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'input-topic',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# Define sink
table_env.execute_sql("""
    CREATE TABLE output_stream (
        id STRING,
        prediction DOUBLE,
        ts TIMESTAMP(3)
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'output-topic',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# Process with UDF
@udf(input_types=[DataTypes.ARRAY(DataTypes.DOUBLE())],
     result_type=DataTypes.DOUBLE())
def predict(features):
    return float(model.predict([features])[0])

table_env.register_function("predict", predict)

# Execute query
table_env.execute_sql("""
    INSERT INTO output_stream
    SELECT id, predict(features) as prediction, ts
    FROM input_stream
""")
```

## Latency Requirements

### Latency Categories

**Ultra-Low Latency (<10ms)**:
- High-frequency trading
- Real-time gaming
- Autonomous vehicles

**Low Latency (10-100ms)**:
- Web search
- Recommendation systems
- Fraud detection

**Medium Latency (100ms-1s)**:
- Content personalization
- Email classification
- Image recognition

**High Latency (>1s)**:
- Batch processing
- Report generation
- Data analysis

### Latency Optimization Techniques

**Model Optimization**:
```python
# Quantization
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

**Caching**:
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=10000)
def cached_predict(features_hash):
    return model.predict(features)

def predict_with_cache(features):
    features_hash = hashlib.md5(str(features).encode()).hexdigest()
    return cached_predict(features_hash)
```

**Precomputation**:
```python
# Precompute common predictions
common_queries = load_common_queries()
precomputed = {query: model.predict(query) for query in common_queries}

def predict(query):
    if query in precomputed:
        return precomputed[query]
    return model.predict(query)
```

## Throughput Optimization

### Batching

```python
import asyncio
from collections import deque

class BatchProcessor:
    def __init__(self, batch_size=32, timeout=0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = deque()
        self.lock = asyncio.Lock()
    
    async def add_request(self, features, future):
        async with self.lock:
            self.queue.append((features, future))
            if len(self.queue) >= self.batch_size:
                await self.process_batch()
    
    async def process_batch(self):
        batch = []
        futures = []
        for _ in range(min(self.batch_size, len(self.queue))):
            features, future = self.queue.popleft()
            batch.append(features)
            futures.append(future)
        
        if batch:
            predictions = model.predict_batch(batch)
            for future, pred in zip(futures, predictions):
                future.set_result(pred)
    
    async def start_timer(self):
        while True:
            await asyncio.sleep(self.timeout)
            async with self.lock:
                if self.queue:
                    await self.process_batch()
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import queue

class ParallelPredictor:
    def __init__(self, num_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.queue = queue.Queue()
    
    def predict_async(self, features):
        future = self.executor.submit(model.predict, [features])
        return future
    
    def predict_batch_parallel(self, features_list):
        futures = [self.predict_async(f) for f in features_list]
        return [f.result() for f in futures]
```

### GPU Batching

```python
import torch

class GPUBatchPredictor:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
    
    def predict_batch(self, features_list):
        # Convert to tensor
        features_tensor = torch.tensor(features_list).to(self.device)
        
        # Process in batches
        predictions = []
        for i in range(0, len(features_tensor), self.batch_size):
            batch = features_tensor[i:i+self.batch_size]
            with torch.no_grad():
                batch_preds = self.model(batch)
            predictions.extend(batch_preds.cpu().numpy())
        
        return predictions
```

## Async Processing

### AsyncIO Implementation

```python
import asyncio
from aiohttp import web
import aiohttp

app = web.Application()

async def predict_handler(request):
    data = await request.json()
    features = data['features']
    
    # Async prediction
    loop = asyncio.get_event_loop()
    prediction = await loop.run_in_executor(
        None, model.predict, [features]
    )
    
    return web.json_response({'prediction': prediction.tolist()})

app.router.add_post('/predict', predict_handler)

if __name__ == '__main__':
    web.run_app(app, port=8080)
```

### Celery for Async Tasks

```python
from celery import Celery

app = Celery('inference', broker='redis://localhost:6379')

@app.task
def predict_task(features):
    return model.predict([features]).tolist()

# Client
result = predict_task.delay(features)
prediction = result.get(timeout=10)
```

## Message Queues

### RabbitMQ

```python
import pika
import json
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Setup connection
connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost')
)
channel = connection.channel()

# Declare queue
channel.queue_declare(queue='inference_queue', durable=True)

def process_message(ch, method, properties, body):
    data = json.loads(body)
    features = data['features']
    
    # Predict
    prediction = model.predict([features])[0]
    
    # Send response
    response = {'prediction': float(prediction)}
    ch.basic_publish(
        exchange='',
        routing_key=properties.reply_to,
        properties=pika.BasicProperties(
            correlation_id=properties.correlation_id
        ),
        body=json.dumps(response)
    )
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Consume messages
channel.basic_qos(prefetch_count=1)
channel.basic_consume(
    queue='inference_queue',
    on_message_callback=process_message
)
channel.start_consuming()
```

### Redis Queue

```python
import redis
from rq import Queue
import pickle

# Setup
redis_conn = redis.Redis()
q = Queue('inference', connection=redis_conn)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

def predict(features):
    return model.predict([features]).tolist()

# Enqueue job
job = q.enqueue(predict, features)
result = job.result
```

### AWS SQS

```python
import boto3
import json
import pickle

sqs = boto3.client('sqs')
queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789/inference-queue'

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Process messages
while True:
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20
    )
    
    if 'Messages' in response:
        for message in response['Messages']:
            data = json.loads(message['Body'])
            features = data['features']
            
            # Predict
            prediction = model.predict([features])[0]
            
            # Send to output queue
            output_queue = sqs.get_queue_url(
                QueueName='output-queue'
            )['QueueUrl']
            
            sqs.send_message(
                QueueUrl=output_queue,
                MessageBody=json.dumps({
                    'id': data['id'],
                    'prediction': float(prediction)
                })
            )
            
            # Delete message
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
```

## Hybrid Approaches

### Lambda Architecture

Combines batch and streaming processing:

```
┌─────────────┐
│   Stream    │──┐
│  (Real-time)│  │
└─────────────┘  │
                 │
┌─────────────┐  │     ┌─────────────┐
│   Batch     │──┼────▶│   Serving   │
│  (Historical)│  │     │    Layer    │
└─────────────┘  │     └─────────────┘
                 │
                 ▼
         ┌───────────────┐
         │   Merge       │
         │   Results     │
         └───────────────┘
```

### Request Routing

```python
class InferenceRouter:
    def __init__(self):
        self.batch_threshold = 100
        self.batch_processor = BatchProcessor()
        self.real_time_processor = RealTimeProcessor()
    
    def predict(self, requests):
        if len(requests) == 1:
            # Single request - real-time
            return self.real_time_processor.predict(requests[0])
        elif len(requests) < self.batch_threshold:
            # Small batch - real-time with batching
            return self.real_time_processor.predict_batch(requests)
        else:
            # Large batch - offline processing
            return self.batch_processor.process(requests)
```

## Cost Considerations

### Cost Analysis

**Batch Processing**:
- Lower cost per prediction
- Efficient resource utilization
- Can use spot instances
- Cost: ~$0.001-0.01 per 1000 predictions

**Real-Time Processing**:
- Higher cost per prediction
- Requires always-on infrastructure
- Premium instance types
- Cost: ~$0.01-0.10 per 1000 predictions

### Cost Optimization

**Batch Processing**:
```python
# Use spot instances
instance_config = {
    'instance_type': 'ml.m5.xlarge',
    'use_spot_instances': True,
    'max_wait_time': 3600  # 1 hour
}
```

**Real-Time Processing**:
```python
# Auto-scaling
autoscaling_config = {
    'min_instances': 1,
    'max_instances': 10,
    'target_cpu_utilization': 70
}
```

## Key Takeaways

- Batch inference processes large volumes offline with high throughput and lower cost
- Real-time inference serves individual requests with low latency for interactive applications
- Latency requirements determine the appropriate inference approach
- Throughput optimization through batching, parallel processing, and GPU utilization
- Async processing enables non-blocking inference for better resource utilization
- Message queues decouple producers and consumers for scalable architectures
- Hybrid approaches combine batch and real-time processing for optimal performance
- Cost considerations favor batch processing for large volumes and real-time for low-latency needs
- The choice between batch and real-time depends on latency requirements, throughput needs, and cost constraints
- Modern systems often combine both approaches to serve different use cases efficiently
