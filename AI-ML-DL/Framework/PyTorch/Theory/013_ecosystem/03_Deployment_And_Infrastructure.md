# Deployment and Infrastructure

## Table of Contents

1. [Docker for PyTorch](#1-docker-for-pytorch)
2. [Kubernetes](#2-kubernetes)

---

## 1. Docker for PyTorch

Docker provides reproducible, portable environments for PyTorch training and serving.

### 1.1 Dockerfile Best Practices

| Practice | Description |
|----------|-------------|
| Use specific version tags | `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` |
| Layer ordering | Dependencies before application code |
| Multi-stage builds | Separate build and runtime stages |
| Non-root user | Run containers as unprivileged user |
| Health checks | Verify container readiness |

### 1.2 NVIDIA Base Images

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]
```

### 1.3 Multi-Stage Builds

```dockerfile
FROM python:3.9-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "inference.py"]
```

### 1.4 GPU Support

```dockerfile
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-7 \
    && rm -rf /var/lib/apt/lists/*
```

For runtime, use `nvidia-docker` or Docker's `--gpus all` flag.

### 1.5 docker-compose for Training and Serving

**Training stack:**

```yaml
version: '3.8'

services:
  pytorch-training:
    build:
      context: .
      dockerfile: Dockerfile.training
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    environment:
      - TORCH_HOME=/app/.torch
      - WANDB_API_KEY=${WANDB_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  tensorboard:
    image: tensorflow/tensorflow:latest
    command: tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    volumes:
      - ./logs:/logs
    ports:
      - "6006:6006"
```

**Serving stack:**

```yaml
services:
  pytorch-api:
    build:
      context: .
      dockerfile: Dockerfile.inference
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/model.pth
    volumes:
      - ./models:/app/models:ro
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
```

---

## 2. Kubernetes

Kubernetes orchestrates containerized PyTorch workloads at scale.

### 2.1 Pod Specs for GPU Workloads

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training
spec:
  containers:
  - name: pytorch
    image: pytorch-app:training
    resources:
      limits:
        cpu: "4"
        memory: "8Gi"
        nvidia.com/gpu: "1"
      requests:
        cpu: "1"
        memory: "2Gi"
        nvidia.com/gpu: "1"
    env:
    - name: PYTHONUNBUFFERED
      value: "1"
    volumeMounts:
    - name: data-volume
      mountPath: /workspace/data
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: pytorch-data-pvc
```

### 2.2 Resource Requests and Limits

| Resource | Training | Inference |
|----------|----------|-----------|
| CPU | 4 cores | 2 cores |
| Memory | 8Gi | 4Gi |
| GPU | 1 | 0-1 |

### 2.3 PyTorch on K8s

Deploy training jobs as **Jobs** and inference as **Deployments**:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pytorch-training
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: pytorch-trainer
        image: pytorch-app:training
        command: ["python", "train.py"]
        args: ["--epochs", "50", "--batch-size", "64"]
        resources:
          limits:
            nvidia.com/gpu: "1"
  backoffLimit: 3
```

### 2.4 Kubeflow Concepts

Kubeflow provides ML-specific Kubernetes resources:

- **PyTorchJob**: Native PyTorch distributed training
- **MPIJob**: Horovod-style distributed training
- **Katib**: Hyperparameter tuning
- **Pipeline**: ML workflow orchestration

### 2.5 Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pytorch-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pytorch-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2.6 Distributed Training on K8s

**PyTorchJob (Kubeflow):**

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch-app:distributed
            command: ["python", "train_distributed.py"]
            resources:
              limits:
                nvidia.com/gpu: "1"
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch-app:distributed
            command: ["python", "train_distributed.py"]
            resources:
              limits:
                nvidia.com/gpu: "1"
```

**Volcano** and **PyTorch Operator** provide alternative schedulers for gang scheduling and GPU sharing.

### 2.7 Inference Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pytorch-inference
  template:
    metadata:
      labels:
        app: pytorch-inference
    spec:
      containers:
      - name: pytorch-inference
        image: pytorch-app:inference
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-volume
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

### 2.8 ConfigMaps and Secrets

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pytorch-config
data:
  model_config.yaml: |
    model:
      type: classification
      num_classes: 10
---
apiVersion: v1
kind: Secret
metadata:
  name: pytorch-secrets
type: Opaque
data:
  wandb-api-key: <base64-encoded>
```
