# Containerization and Orchestration

## Table of Contents

1. [Introduction to Containerization](#introduction-to-containerization)
2. [Docker for ML](#docker-for-ml)
3. [Containerizing ML Workloads](#containerizing-ml-workloads)
4. [Kubernetes Overview](#kubernetes-overview)
5. [Kubernetes Orchestration for ML](#kubernetes-orchestration-for-ml)
6. [Microservices Architecture](#microservices-architecture)
7. [Helm Charts](#helm-charts)
8. [Resource Management](#resource-management)
9. [GPU Scheduling](#gpu-scheduling)
10. [Key Takeaways](#key-takeaways)

## Introduction to Containerization

Containerization packages applications and their dependencies into isolated, portable containers. For ML systems, containers provide:

- **Reproducibility**: Consistent environments across development, staging, and production
- **Isolation**: Separate dependencies for different models and services
- **Portability**: Run anywhere Docker/Kubernetes is supported
- **Scalability**: Easy horizontal scaling of containerized services
- **Resource Efficiency**: Better resource utilization than VMs

### Containers vs Virtual Machines

| Aspect | Containers | Virtual Machines |
|--------|-----------|------------------|
| Isolation | Process-level | OS-level |
| Startup Time | Seconds | Minutes |
| Resource Overhead | Low | High |
| Portability | High | Medium |
| Security | Process isolation | Full OS isolation |

### Container Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer                │
│  (ML Model, API Server, Training Code)   │
├─────────────────────────────────────────┤
│         Container Runtime                │
│  (Docker, containerd, CRI-O)            │
├─────────────────────────────────────────┤
│         Operating System                 │
│  (Linux Kernel)                          │
├─────────────────────────────────────────┤
│         Hardware                         │
└─────────────────────────────────────────┘
```

## Docker for ML

### Basic Dockerfile

```dockerfile
# Base image with Python and ML libraries
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl

# Run application
CMD ["python", "src/app.py"]
```

### Multi-Stage Builds

```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN pip install --user --no-cache-dir \
    numpy \
    pandas \
    scikit-learn

# Training stage
FROM builder as trainer

WORKDIR /train

COPY training/ ./training/
COPY data/ ./data/

RUN python training/train.py

# Production stage
FROM python:3.9-slim as production

WORKDIR /app

# Copy only runtime dependencies
COPY --from=builder /root/.local /root/.local
COPY --from=trainer /train/models/model.pkl ./models/

COPY src/ ./src/

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8080

CMD ["python", "src/app.py"]
```

### GPU Support

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

CMD ["python3", "src/app.py"]
```

## Containerizing ML Workloads

### Training Container

```dockerfile
FROM python:3.9-slim

WORKDIR /train

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY training/ ./training/
COPY data/ ./data/

# Set entrypoint
ENTRYPOINT ["python", "training/train.py"]
CMD ["--config", "config.yaml"]
```

### Serving Container

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and serving code
COPY models/ ./models/
COPY serving/ ./serving/

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "serving/app.py"]
```

### Feature Engineering Container

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies including Spark
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy feature engineering code
COPY feature_engineering/ ./feature_engineering/

# Set Spark environment
ENV SPARK_HOME=/opt/spark
ENV PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH

CMD ["python", "feature_engineering/pipeline.py"]
```

### Docker Compose for ML Stack

```yaml
version: '3.8'

services:
  training:
    build: ./training
    volumes:
      - ./data:/data
      - ./models:/models
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
  
  serving:
    build: ./serving
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - MODEL_PATH=/models/model.pkl
    depends_on:
      - training
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.0.0
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    environment:
      - BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
      - DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  mlflow-data:
  redis-data:
```

## Kubernetes Overview

Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications.

### Core Concepts

**Pod**: Smallest deployable unit, contains one or more containers
**Deployment**: Manages replica sets of pods
**Service**: Provides stable network access to pods
**ConfigMap**: Stores configuration data
**Secret**: Stores sensitive data
**Namespace**: Logical separation of resources

### Kubernetes Architecture

```
┌─────────────────────────────────────────────────┐
│              Control Plane                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   API    │  │ Scheduler│  │ etcd     │      │
│  │  Server  │  │          │  │          │      │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────┐      ┌────────▼────────┐
│   Worker     │      │    Worker       │
│   Node 1     │      │    Node 2       │
│  ┌────────┐  │      │   ┌────────┐    │
│  │  Kubelet│  │      │   │ Kubelet│   │
│  └────────┘  │      │   └────────┘    │
│  ┌────────┐  │      │   ┌────────┐    │
│  │  Proxy │  │      │   │ Proxy │    │
│  └────────┘  │      │   └────────┘    │
│  ┌────────┐  │      │   ┌────────┐    │
│  │  Pods  │  │      │   │ Pods  │    │
│  └────────┘  │      │   └────────┘    │
└──────────────┘      └────────────────┘
```

## Kubernetes Orchestration for ML

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-serving
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: ml-model:latest
        ports:
        - containerPort: 8080
        env:
        - name: MODEL_PATH
          value: "/models/model.pkl"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service Manifest

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: LoadBalancer
```

### ConfigMap for Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-model-config
data:
  model_path: "/models/model.pkl"
  batch_size: "32"
  max_concurrent_requests: "100"
  log_level: "INFO"
```

### Secret for Sensitive Data

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-model-secrets
type: Opaque
data:
  api_key: <base64-encoded-api-key>
  database_password: <base64-encoded-password>
```

### Job for Training

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training-job
spec:
  completions: 1
  parallelism: 1
  template:
    spec:
      containers:
      - name: trainer
        image: ml-training:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "8000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: data
          mountPath: /data
        - name: models
          mountPath: /models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: model-artifacts-pvc
      restartPolicy: Never
```

## Microservices Architecture

### Service Decomposition

Break ML systems into microservices:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Feature    │     │    Model     │     │   Prediction │
│   Service    │────▶│   Service    │────▶│    Service   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   API Gateway   │
                   └─────────────────┘
```

### Feature Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: feature-service
        image: feature-service:latest
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: FEATURE_STORE_URL
          value: "http://feature-store:8080"
```

### Model Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: model-service
        image: model-service:latest
        env:
        - name: MODEL_REGISTRY_URL
          value: "http://mlflow:5000"
        - name: MODEL_NAME
          value: "classification_model"
        - name: MODEL_STAGE
          value: "Production"
```

## Helm Charts

Helm is a package manager for Kubernetes that simplifies deployment of complex applications.

### Chart Structure

```
ml-model-chart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
└── charts/
```

### Chart.yaml

```yaml
apiVersion: v2
name: ml-model
description: ML Model Serving Helm Chart
type: application
version: 1.0.0
appVersion: "1.0"
```

### values.yaml

```yaml
replicaCount: 3

image:
  repository: ml-model
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

model:
  name: classification_model
  version: "1.0.0"
  path: "/models/model.pkl"
```

### Deployment Template

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-model.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
        env:
        - name: MODEL_PATH
          value: {{ .Values.model.path | quote }}
```

### Installing Chart

```bash
# Install chart
helm install ml-model ./ml-model-chart

# Upgrade chart
helm upgrade ml-model ./ml-model-chart

# List releases
helm list

# Uninstall
helm uninstall ml-model
```

## Resource Management

### Resource Requests and Limits

```yaml
resources:
  requests:
    memory: "2Gi"      # Guaranteed memory
    cpu: "1000m"       # Guaranteed CPU (1 core)
  limits:
    memory: "4Gi"      # Maximum memory
    cpu: "2000m"       # Maximum CPU (2 cores)
```

### Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
```

### Limit Ranges

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-limit-range
spec:
  limits:
  - default:
      memory: "2Gi"
      cpu: "1000m"
    defaultRequest:
      memory: "1Gi"
      cpu: "500m"
    type: Container
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-serving
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

## GPU Scheduling

### Node Feature Discovery

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nfd-worker
spec:
  template:
    spec:
      containers:
      - name: node-feature-discovery
        image: k8s.gcr.io/nfd/node-feature-discovery:v0.14.0
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
```

### GPU Node Labels

```bash
# Label GPU nodes
kubectl label nodes gpu-node-1 accelerator=nvidia-tesla-v100
kubectl label nodes gpu-node-2 accelerator=nvidia-tesla-v100
```

### GPU-Aware Scheduling

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-training
spec:
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: trainer
        image: gpu-training:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Device Plugin

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
spec:
  template:
    spec:
      containers:
      - name: nvidia-device-plugin-ctr
        image: nvcr.io/nvidia/k8s-device-plugin:v0.14.0
        env:
        - name: FAIL_ON_INIT_ERROR
          value: "false"
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
```

### Multi-GPU Training

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: multi-gpu-training
spec:
  completions: 1
  parallelism: 1
  template:
    spec:
      containers:
      - name: trainer
        image: multi-gpu-training:latest
        command: ["python", "train.py", "--distributed"]
        resources:
          limits:
            nvidia.com/gpu: 4
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
```

## Key Takeaways

- Containerization provides reproducibility, isolation, and portability for ML workloads
- Docker enables consistent environments across development, staging, and production
- Multi-stage builds optimize container size and build times
- Kubernetes orchestrates containerized ML applications at scale
- Microservices architecture decomposes ML systems into manageable services
- Helm charts simplify deployment and management of complex Kubernetes applications
- Resource management ensures fair allocation and prevents resource exhaustion
- GPU scheduling enables efficient utilization of GPU resources for training
- Health checks and probes ensure service reliability and availability
- Persistent volumes provide storage for models, data, and artifacts
