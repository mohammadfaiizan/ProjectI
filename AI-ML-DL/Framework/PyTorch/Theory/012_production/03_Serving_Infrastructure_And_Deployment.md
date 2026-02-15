# Serving Infrastructure and Deployment

## Table of Contents

1. [TorchServe](#1-torchserve)
2. [C++ Deployment with LibTorch](#2-c-deployment-with-libtorch)
3. [Containerized Deployment](#3-containerized-deployment)
4. [Cloud Deployment](#4-cloud-deployment)

---

## 1. TorchServe

TorchServe is a model serving framework for PyTorch that provides scalable inference with REST and gRPC APIs.

### 1.1 Model Archiver (MAR)

The **torch-model-archiver** packages a model and handler into a Model Archive (`.mar`) file:

```bash
torch-model-archiver \
    --model-name demo_cnn \
    --version 1.0 \
    --serialized-file model.pth \
    --handler image_classifier.py \
    --export-path model-store \
    --extra-files class_names.json
```

| Parameter | Description |
|-----------|-------------|
| --model-name | Name of the model |
| --serialized-file | Model weights file |
| --handler | Python handler for pre/post-processing |
| --extra-files | Additional files (config, class names) |

### 1.2 Handler Classes

Handlers implement `initialize`, `preprocess`, `inference`, and `postprocess`:

```python
from ts.torch_handler.base_handler import BaseHandler

class ImageClassificationHandler(BaseHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        self.model = load_model(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        images = [decode_image(row) for row in data]
        return torch.stack(images)

    def inference(self, data):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data):
        probs = F.softmax(data, dim=1)
        return [{"predictions": get_top_k(probs[i])} for i in range(len(probs))]
```

### 1.3 Custom Handlers

Custom handlers extend BaseHandler to support domain-specific preprocessing (e.g., image resizing, normalization) and postprocessing (e.g., formatting predictions, applying thresholds).

### 1.4 REST and gRPC API

**REST API (default port 8080):**

```bash
curl http://localhost:8080/predictions/demo_cnn -T image.jpg
curl http://localhost:8080/ping
```

**Management API (port 8081):**

```bash
curl http://localhost:8081/models
curl -X POST "http://localhost:8081/models?url=demo_cnn.mar&initial_workers=1"
curl -X PUT "http://localhost:8081/models/demo_cnn?min_worker=1&max_worker=3"
curl -X DELETE http://localhost:8081/models/demo_cnn
```

**gRPC:** Ports 7070 (inference) and 7071 (management).

### 1.5 Batch Inference

Configure batch size and max batch delay in `config.properties`:

```properties
batch_size=4
max_batch_delay=5000
```

TorchServe batches requests and runs inference when the batch is full or the delay is exceeded.

### 1.6 Model Management API

| Endpoint | Method | Description |
|----------|--------|--------------|
| /models | GET | List models |
| /models | POST | Register model |
| /models/{name} | GET | Model info |
| /models/{name} | PUT | Scale workers |
| /models/{name} | DELETE | Unregister |

---

## 2. C++ Deployment with LibTorch

LibTorch is the C++ distribution of PyTorch for deploying models without Python.

### 2.1 Loading ScriptModule in C++

```cpp
#include <torch/script.h>

torch::jit::script::Module model;
model = torch::jit::load("model.pt");
model.to(device);
model.eval();
```

### 2.2 CMake Setup

```cmake
cmake_minimum_required(VERSION 3.12)
project(pytorch_inference)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(inference main.cpp)
target_link_libraries(inference "${TORCH_LIBRARIES}")
target_link_libraries(inference "${OpenCV_LIBS}")
```

Build:

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

### 2.3 Tensor Operations in C++

```cpp
std::vector<torch::jit::IValue> inputs;
inputs.push_back(tensor);

torch::NoGradGuard no_grad;
auto output = model.forward(inputs).toTensor();

auto output_accessor = output.accessor<float, 2>();
for (int i = 0; i < output.size(1); ++i) {
    float prob = output_accessor[0][i];
}
```

### 2.4 Preprocessing in C++

```cpp
cv::Mat image = cv::imread("image.jpg");
cv::resize(image, image, cv::Size(224, 224));
cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

std::vector<float> input_data;
for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 224; ++h) {
        for (int w = 0; w < 224; ++w) {
            float pixel = image.at<cv::Vec3b>(h, w)[c] / 255.0f;
            pixel = (pixel - mean[c]) / std_dev[c];
            input_data.push_back(pixel);
        }
    }
}

auto tensor = torch::from_blob(input_data.data(), {1, 3, 224, 224}, torch::kFloat32);
```

### 2.5 Export from Python

Models must be traced or scripted before C++ loading:

```python
model.eval()
traced = torch.jit.trace(model, example_input)
traced.save("model.pt")
```

---

## 3. Containerized Deployment

### 3.1 Docker for PyTorch

**Multi-stage build** reduces image size by separating build and runtime:

```dockerfile
FROM python:3.9-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.9-slim as production
WORKDIR /app
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY app.py model.pth .
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "app.py"]
```

### 3.2 GPU Containers

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch torchvision
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with NVIDIA runtime:

```bash
docker run --gpus all -p 8000:8000 pytorch-inference:gpu
```

### 3.3 docker-compose

```yaml
version: '3.8'

services:
  pytorch-model:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          memory: 2G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3.4 FastAPI Application

```python
from fastapi import FastAPI, UploadFile, File
import torch

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    model = load_trained_model()
    model.eval()

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = decode_image(await file.read())
    input_tensor = preprocess(image)
    with torch.no_grad():
        output = model(input_tensor)
    return format_predictions(output)
```

---

## 4. Cloud Deployment

### 4.1 AWS SageMaker Concepts

- **Model**: Container image + model artifacts (S3)
- **Endpoint configuration**: Instance type, instance count
- **Endpoint**: Deployed model serving inference

### 4.2 Inference Endpoints

```python
import boto3

sagemaker = boto3.client('sagemaker')
sagemaker.create_model(
    ModelName='pytorch-model',
    PrimaryContainer={
        'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.0-gpu-py38',
        'ModelDataUrl': 's3://bucket/model.tar.gz'
    },
    ExecutionRoleArn='arn:aws:iam::account:role/SageMakerRole'
)

sagemaker.create_endpoint_config(
    EndpointConfigName='config',
    ProductionVariants=[{
        'VariantName': 'primary',
        'ModelName': 'pytorch-model',
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 1
    }]
)

sagemaker.create_endpoint(
    EndpointName='pytorch-endpoint',
    EndpointConfigName='config'
)
```

### 4.3 Auto-Scaling

SageMaker supports auto-scaling based on InvocationsPerInstance or Custom metrics. Configure via `sagemaker.create_endpoint_config` with `AutoScalingConfig`.

### 4.4 Serverless Inference

**AWS Lambda** for sporadic, low-latency workloads:

- Package model and inference code in a deployment zip
- Use Lambda layers for large dependencies
- Cold starts can add 1–5 seconds; consider Provisioned Concurrency
- 15-minute timeout, ~10GB memory limit

### 4.5 SageMaker vs Lambda vs ECS

| Service | Use Case | Pros | Cons |
|---------|----------|------|------|
| SageMaker | Production ML | Managed, auto-scaling, A/B testing | Higher cost, AWS-specific |
| Lambda | Lightweight, sporadic | Pay-per-request, no servers | Cold starts, size limits |
| ECS/Fargate | Custom containers | Flexible, cost control | More ops work |

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| TorchServe | MAR archives, custom handlers, REST/gRPC, batch inference |
| LibTorch | Trace/script in Python, load in C++, CMake + OpenCV |
| Containers | Multi-stage builds, non-root user, health checks, GPU runtime |
| Cloud | SageMaker for managed ML; Lambda for serverless; ECS for custom containers |
