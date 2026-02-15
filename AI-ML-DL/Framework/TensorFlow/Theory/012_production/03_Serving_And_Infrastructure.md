# Serving and Infrastructure

## Table of Contents

1. [TensorFlow Serving Overview](#1-tensorflow-serving-overview)
2. [Model Directory Structure](#2-model-directory-structure)
3. [Docker Deployment](#3-docker-deployment)
4. [REST and gRPC APIs](#4-rest-and-grpc-apis)
5. [TFX Pipeline](#5-tfx-pipeline)
6. [Cloud Deployment with Vertex AI](#6-cloud-deployment-with-vertex-ai)

---

## 1. TensorFlow Serving Overview

**TensorFlow Serving** is a flexible, high-performance serving system for machine learning models. It is designed for production environments and supports versioning, batching, and multiple model types.

### Key Features

- **Version management**: Load multiple model versions; serve latest or specific version
- **Hot reload**: Add new versions without downtime
- **Batching**: Dynamic batching for throughput optimization
- **REST and gRPC**: Standard APIs for prediction requests

### Architecture

TF Serving runs as a separate process. It monitors a model directory, loads models on startup or when new versions appear, and serves predictions via HTTP (REST) or gRPC.

---

## 2. Model Directory Structure

TF Serving expects a specific directory layout. Each model has a base path; each version is a subdirectory containing a SavedModel.

### Layout

```
/path/to/models/
  model_name/
    1/
      saved_model.pb
      variables/
        variables.data-00000-of-00001
        variables.index
    2/
      saved_model.pb
      variables/
        ...
```

- **model_name**: Logical name (e.g., `my_classifier`)
- **1, 2, ...**: Version numbers (integers)
- Each version directory is a complete **SavedModel**

### Version Selection

- **Latest**: Serves the highest version number
- **Specific**: Request version via API (e.g., `models/my_model/versions/2`)
- **Labels**: Optional version labels (e.g., `stable`, `canary`)

---

## 3. Docker Deployment

### Pull and Run

```bash
docker pull tensorflow/serving
```

```bash
docker run -p 8501:8501 -p 8500:8500 \
  -v "/path/to/models:/models/my_model" \
  -e MODEL_NAME=my_model \
  tensorflow/serving
```

- **8501**: REST API
- **8500**: gRPC API
- **-v**: Mount model directory; path format is `/models/<MODEL_NAME>`
- **MODEL_NAME**: Must match the directory name under `/models`

### Multiple Models

```bash
docker run -p 8501:8501 -p 8500:8500 \
  -v "/path/to/models:/models" \
  -e MODEL_NAME=model1 \
  tensorflow/serving
```

For multiple models, use a config file or run separate containers.

### Model Config File

Create `models.config`:

```protobuf
model_config_list {
  config {
    name: 'model1'
    base_path: '/models/model1'
    model_platform: 'tensorflow'
  }
  config {
    name: 'model2'
    base_path: '/models/model2'
    model_platform: 'tensorflow'
  }
}
```

```bash
docker run -p 8501:8501 -p 8500:8500 \
  -v "/path/to/models:/models" \
  -v "/path/to/models.config:/models/models.config" \
  -e MODEL_CONFIG_FILE=/models/models.config \
  tensorflow/serving
```

---

## 4. REST and gRPC APIs

### REST API

**Endpoint**: `POST http://localhost:8501/v1/models/<model_name>:predict`

**Request body** (instances format):

```json
{
  "instances": [
    [1.0, 2.0, 3.0, ...],
    [4.0, 5.0, 6.0, ...]
  ]
}
```

**Request body** (inputs format, for named inputs):

```json
{
  "signature_name": "serving_default",
  "inputs": {
    "input_1": [[1.0, 2.0], [3.0, 4.0]]
  }
}
```

**Response**:

```json
{
  "predictions": [
    [0.1, 0.2, 0.7],
    [0.8, 0.1, 0.1]
  ]
}
```

### Versioned Request

```
POST http://localhost:8501/v1/models/my_model/versions/2:predict
```

### gRPC API

**Endpoint**: `localhost:8500`

**Service**: `tensorflow.serving.PredictionService`

**Method**: `Predict`

gRPC offers lower latency and binary serialization. Use the `tensorflow_serving` package for client code.

### Health and Metadata

- **REST**: `GET http://localhost:8501/v1/models/my_model` (model status)
- **gRPC**: `GetModelMetadata` for signature and tensor info

---

## 5. TFX Pipeline

**TFX (TensorFlow Extended)** is a platform for building production ML pipelines. It orchestrates data ingestion, preprocessing, training, evaluation, and deployment.

### Core Components

| Component | Purpose |
|-----------|---------|
| **ExampleGen** | Ingests data from CSV, TFRecord, BigQuery, etc. |
| **Transform** | Preprocessing, feature engineering, schema validation |
| **Trainer** | Trains model with TensorFlow/Keras |
| **Evaluator** | Validates metrics against baseline; gates deployment |
| **Pusher** | Deploys model to TF Serving, Vertex AI, or other targets |

### Pipeline DAG

```
ExampleGen -> Transform -> Trainer -> Evaluator -> Pusher
                |              |
                +--------------+
                (Transform output feeds Trainer)
```

**Evaluator** compares new model metrics to a baseline. If the new model passes (e.g., accuracy above threshold), **Pusher** deploys it.

### ExampleGen

```python
from tfx.components import CsvExampleGen
example_gen = CsvExampleGen(input_base="/path/to/data")
```

### Transform

```python
from tfx.components import Transform
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file="/path/to/preprocessing.py"
)
```

### Trainer

```python
from tfx.components import Trainer
trainer = Trainer(
    module_file="/path/to/trainer.py",
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema']
)
```

### Pusher

```python
from tfx.components import Pusher
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=...
)
```

### Orchestration

TFX pipelines run on **Apache Beam** (local or Dataflow), **Kubeflow Pipelines**, or **Vertex AI Pipelines**.

---

## 6. Cloud Deployment with Vertex AI

**Vertex AI** (Google Cloud) provides managed ML infrastructure including model registry, endpoints, and deployment.

### Key Concepts

- **Model Registry**: Versioned storage for models (SavedModel, custom containers)
- **Endpoint**: Serving URL; can host multiple deployments
- **Deployment**: Model instance on an endpoint (replica count, machine type)

### Deployment Flow

1. **Upload model** to Model Registry
2. **Create endpoint** (or use existing)
3. **Deploy model** to endpoint with desired machine type and replica count

### Python SDK Example

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="my_model",
    artifact_uri="gs://bucket/model"
)

endpoint = aiplatform.Endpoint.create(display_name="my_endpoint")

endpoint.deploy(
    model,
    deployed_model_display_name="v1",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3
)
```

### Prediction

```python
response = endpoint.predict(instances=[[...]])
```

### Auto-scaling

Vertex AI supports auto-scaling based on traffic. Configure `min_replica_count` and `max_replica_count` for the deployment.

### Comparison: TF Serving vs Vertex AI

| Aspect | TF Serving | Vertex AI |
|--------|-------------|-----------|
| Setup | Self-managed, Docker | Managed service |
| Scaling | Manual or K8s | Auto-scaling |
| Monitoring | Custom | Built-in |
| Cost | Infrastructure only | Per prediction + infra |

---

## Summary

- **TF Serving** provides high-performance model serving with versioning and batching
- Model directory: `model_name/version/saved_model.pb` and `variables/`
- **Docker** is the standard deployment method; use `-v` to mount models
- **REST** (8501) and **gRPC** (8500) APIs for predictions
- **TFX** pipelines automate data, training, evaluation, and deployment
- **Vertex AI** offers managed deployment with auto-scaling and monitoring
