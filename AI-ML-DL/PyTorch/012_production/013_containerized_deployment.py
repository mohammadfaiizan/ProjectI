import torch
import torch.nn as nn
import os
import json
from typing import Dict, List, Tuple, Optional, Any

# Sample Model for Containerized Deployment
class ContainerizedModel(nn.Module):
    """Model designed for containerized deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Docker Configuration Generator
class DockerfileGenerator:
    """Generate Dockerfiles for PyTorch model deployment"""
    
    @staticmethod
    def create_inference_dockerfile(base_image: str = "python:3.9-slim",
                                   requirements_file: str = "requirements.txt",
                                   model_file: str = "model.pth",
                                   app_file: str = "app.py") -> str:
        """Create Dockerfile for inference service"""
        
        dockerfile_content = f'''
# Multi-stage build for optimized production image
FROM {base_image} as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Production stage
FROM {base_image} as production

WORKDIR /app

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application files
COPY {app_file} .
COPY {model_file} .
COPY model_utils.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "{app_file}"]
'''
        return dockerfile_content
    
    @staticmethod
    def create_training_dockerfile(base_image: str = "pytorch/pytorch:latest") -> str:
        """Create Dockerfile for training workloads"""
        
        dockerfile_content = f'''
FROM {base_image}

# Set working directory
WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \\
    tensorboard \\
    wandb \\
    mlflow \\
    scikit-learn \\
    matplotlib \\
    seaborn

# Copy training code
COPY train.py .
COPY data_loader.py .
COPY model.py .
COPY config.yaml .

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/outputs /workspace/logs

# Set environment variables
ENV PYTHONPATH=/workspace
ENV TORCH_HOME=/workspace/.torch

# Default command
CMD ["python", "train.py"]
'''
        return dockerfile_content
    
    @staticmethod
    def create_gpu_dockerfile(cuda_version: str = "11.8") -> str:
        """Create GPU-enabled Dockerfile"""
        
        dockerfile_content = f'''
FROM nvidia/cuda:{cuda_version}-runtime-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional packages
RUN pip install \\
    fastapi \\
    uvicorn \\
    pillow \\
    numpy \\
    requests

WORKDIR /app

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'CUDA devices: {{torch.cuda.device_count()}}')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        return dockerfile_content

# Docker Compose Generator
class DockerComposeGenerator:
    """Generate Docker Compose configurations"""
    
    @staticmethod
    def create_inference_compose(service_name: str = "pytorch-model",
                                image_name: str = "pytorch-inference:latest",
                                port: int = 8000) -> str:
        """Create Docker Compose for inference service"""
        
        compose_content = f'''
version: '3.8'

services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    image: {image_name}
    container_name: {service_name}
    ports:
      - "{port}:{port}"
    environment:
      - PYTHONPATH=/app
      - TORCH_SERVE_LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    depends_on:
      - {service_name}

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:

networks:
  default:
    name: pytorch-network
'''
        return compose_content
    
    @staticmethod
    def create_gpu_compose(service_name: str = "pytorch-gpu") -> str:
        """Create Docker Compose with GPU support"""
        
        compose_content = f'''
version: '3.8'

services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    container_name: {service_name}
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

# For Docker Compose v2.3+ with GPU support
# Use: docker-compose --compatibility up
'''
        return compose_content

# Flask/FastAPI Application Generator
class ApplicationGenerator:
    """Generate application code for containerized deployment"""
    
    @staticmethod
    def create_fastapi_app(model_class_name: str = "ContainerizedModel") -> str:
        """Create FastAPI application for model serving"""
        
        app_content = f'''
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import logging
import time
from typing import Dict, List
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model class
# Assuming model class is defined in model_utils.py
from model_utils import {model_class_name}

app = FastAPI(
    title="PyTorch Model API",
    description="REST API for PyTorch model inference",
    version="1.0.0"
)

# Global variables
model = None
device = None
transform = None

def load_model():
    """Load model on startup"""
    global model, device, transform
    
    try:
        # Detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {{device}}")
        
        # Load model
        model = {model_class_name}(num_classes=10)
        model.load_state_dict(torch.load('model.pth', map_location=device))
        model.to(device)
        model.eval()
        
        # Define preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {{"message": "PyTorch Model API", "status": "running"}}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Quick model test
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(test_input)
        
        return {{
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(device),
            "timestamp": time.time()
        }}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {{e}}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make prediction on uploaded image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5)
        
        predictions = []
        for i in range(5):
            predictions.append({{
                "class_id": top_indices[0][i].item(),
                "class_name": f"class_{{top_indices[0][i].item()}}",
                "probability": top_probs[0][i].item()
            }})
        
        return {{
            "predictions": predictions,
            "inference_time_ms": inference_time,
            "model_version": "1.0.0"
        }}
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{e}}")

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Make predictions on multiple images"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    try:
        batch_inputs = []
        
        # Process all images
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = transform(image)
            batch_inputs.append(input_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(batch_inputs).to(device)
        
        # Run batch inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Process results
        results = []
        for i in range(len(files)):
            top_probs, top_indices = torch.topk(probabilities[i:i+1], k=3)
            
            predictions = []
            for j in range(3):
                predictions.append({{
                    "class_id": top_indices[0][j].item(),
                    "probability": top_probs[0][j].item()
                }})
            
            results.append({{
                "filename": files[i].filename,
                "predictions": predictions
            }})
        
        return {{
            "results": results,
            "batch_size": len(files),
            "total_inference_time_ms": inference_time,
            "avg_per_image_ms": inference_time / len(files)
        }}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {{e}}")

@app.get("/model_info")
async def model_info():
    """Get model information"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {{
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "input_shape": [3, 224, 224],
        "num_classes": 10
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        return app_content
    
    @staticmethod
    def create_model_utils() -> str:
        """Create model utilities file"""
        
        utils_content = '''
import torch
import torch.nn as nn

class ContainerizedModel(nn.Module):
    """Model designed for containerized deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
'''
        return utils_content
    
    @staticmethod
    def create_requirements() -> str:
        """Create requirements.txt file"""
        
        requirements = '''
torch>=1.12.0
torchvision>=0.13.0
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
python-multipart>=0.0.5
pillow>=8.3.0
numpy>=1.21.0
requests>=2.25.0
'''
        return requirements

# Container Deployment Manager
class ContainerDeploymentManager:
    """Manage containerized deployments"""
    
    def __init__(self, deployment_dir: str = "container_deployment"):
        self.deployment_dir = deployment_dir
        os.makedirs(deployment_dir, exist_ok=True)
    
    def create_deployment_package(self, model: nn.Module,
                                 model_name: str = "pytorch_model",
                                 deployment_type: str = "inference") -> str:
        """Create complete deployment package"""
        
        # Create model directory
        model_dir = os.path.join(self.deployment_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Generate application files
        app_generator = ApplicationGenerator()
        
        # FastAPI app
        app_content = app_generator.create_fastapi_app()
        with open(os.path.join(model_dir, "app.py"), 'w') as f:
            f.write(app_content)
        
        # Model utilities
        utils_content = app_generator.create_model_utils()
        with open(os.path.join(model_dir, "model_utils.py"), 'w') as f:
            f.write(utils_content)
        
        # Requirements
        requirements = app_generator.create_requirements()
        with open(os.path.join(model_dir, "requirements.txt"), 'w') as f:
            f.write(requirements)
        
        # Generate Docker files
        dockerfile_gen = DockerfileGenerator()
        compose_gen = DockerComposeGenerator()
        
        # Dockerfile
        dockerfile = dockerfile_gen.create_inference_dockerfile()
        with open(os.path.join(model_dir, "Dockerfile"), 'w') as f:
            f.write(dockerfile)
        
        # Docker Compose
        compose = compose_gen.create_inference_compose(service_name=model_name)
        with open(os.path.join(model_dir, "docker-compose.yml"), 'w') as f:
            f.write(compose)
        
        # GPU Dockerfile
        gpu_dockerfile = dockerfile_gen.create_gpu_dockerfile()
        with open(os.path.join(model_dir, "Dockerfile.gpu"), 'w') as f:
            f.write(gpu_dockerfile)
        
        # Build and run scripts
        self._create_build_scripts(model_dir, model_name)
        
        # README
        self._create_readme(model_dir, model_name)
        
        print(f"✓ Deployment package created: {model_dir}")
        return model_dir
    
    def _create_build_scripts(self, model_dir: str, model_name: str):
        """Create build and run scripts"""
        
        # Build script
        build_script = f'''#!/bin/bash

echo "Building {model_name} Docker image..."

# Build the image
docker build -t {model_name}:latest .

echo "Build completed: {model_name}:latest"

# Optional: Build GPU version
# docker build -f Dockerfile.gpu -t {model_name}:gpu .
'''
        
        build_path = os.path.join(model_dir, "build.sh")
        with open(build_path, 'w') as f:
            f.write(build_script)
        os.chmod(build_path, 0o755)
        
        # Run script
        run_script = f'''#!/bin/bash

echo "Running {model_name} container..."

# Run with Docker Compose
docker-compose up -d

echo "Container started. API available at http://localhost:8000"
echo "Health check: curl http://localhost:8000/health"
echo "API docs: http://localhost:8000/docs"

# To stop: docker-compose down
'''
        
        run_path = os.path.join(model_dir, "run.sh")
        with open(run_path, 'w') as f:
            f.write(run_script)
        os.chmod(run_path, 0o755)
    
    def _create_readme(self, model_dir: str, model_name: str):
        """Create README file"""
        
        readme_content = f'''# {model_name} Container Deployment

## Overview
This package contains everything needed to deploy the {model_name} PyTorch model in a containerized environment.

## Files
- `app.py`: FastAPI application for model serving
- `model_utils.py`: Model class definition
- `model.pth`: Trained model weights
- `Dockerfile`: Container definition for CPU inference
- `Dockerfile.gpu`: Container definition for GPU inference
- `docker-compose.yml`: Service orchestration
- `requirements.txt`: Python dependencies
- `build.sh`: Build script
- `run.sh`: Run script

## Quick Start

### 1. Build the container
```bash
./build.sh
```

### 2. Run the service
```bash
./run.sh
```

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Upload image for prediction
curl -X POST "http://localhost:8000/predict" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@test_image.jpg"
```

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `GET /model_info`: Model information
- `POST /predict`: Single image prediction
- `POST /predict_batch`: Batch image prediction
- `GET /docs`: Interactive API documentation

## GPU Support

To use GPU acceleration:

1. Build GPU image:
```bash
docker build -f Dockerfile.gpu -t {model_name}:gpu .
```

2. Run with GPU support:
```bash
docker run --gpus all -p 8000:8000 {model_name}:gpu
```

## Production Deployment

### Kubernetes
1. Create Kubernetes manifests
2. Deploy using `kubectl apply`
3. Configure ingress and load balancing

### Docker Swarm
```bash
docker stack deploy -c docker-compose.yml {model_name}
```

### Cloud Platforms
- AWS ECS/EKS
- Google Cloud Run/GKE
- Azure Container Instances/AKS

## Monitoring

The service includes:
- Health check endpoint
- Prometheus metrics (optional)
- Grafana dashboards (optional)
- Structured logging

## Scaling

Horizontal scaling options:
- Multiple container replicas
- Load balancer configuration
- Auto-scaling based on CPU/memory

## Security

Security considerations:
- Non-root user in container
- Minimal base image
- Input validation
- Rate limiting (implement as needed)

## Troubleshooting

Common issues:
- Port conflicts: Change port in docker-compose.yml
- Memory issues: Increase container memory limits
- GPU not detected: Verify NVIDIA Docker runtime
- Model loading fails: Check model file path and permissions
'''
        
        with open(os.path.join(model_dir, "README.md"), 'w') as f:
            f.write(readme_content)

if __name__ == "__main__":
    print("Containerized Model Deployment")
    print("=" * 35)
    
    # Create sample model
    model = ContainerizedModel(num_classes=10)
    
    print("\n1. Deployment Package Creation")
    print("-" * 36)
    
    # Create deployment manager
    deployment_manager = ContainerDeploymentManager("demo_container_deployment")
    
    # Create deployment package
    deployment_dir = deployment_manager.create_deployment_package(
        model=model,
        model_name="image_classifier",
        deployment_type="inference"
    )
    
    print("\n2. Docker Configuration")
    print("-" * 26)
    
    # Generate additional Docker configurations
    dockerfile_gen = DockerfileGenerator()
    compose_gen = DockerComposeGenerator()
    
    # Training Dockerfile
    training_dockerfile = dockerfile_gen.create_training_dockerfile()
    with open(os.path.join(deployment_dir, "Dockerfile.train"), 'w') as f:
        f.write(training_dockerfile)
    
    # GPU Compose
    gpu_compose = compose_gen.create_gpu_compose("image_classifier_gpu")
    with open(os.path.join(deployment_dir, "docker-compose.gpu.yml"), 'w') as f:
        f.write(gpu_compose)
    
    print("✓ Additional Docker configurations created")
    
    print("\n3. Container Best Practices")
    print("-" * 32)
    
    best_practices = [
        "Use multi-stage builds to reduce image size",
        "Run containers as non-root user for security",
        "Implement proper health checks",
        "Use .dockerignore to exclude unnecessary files",
        "Pin dependency versions for reproducibility",
        "Implement graceful shutdown handling",
        "Use environment variables for configuration",
        "Monitor container resource usage",
        "Implement proper logging and monitoring",
        "Use secrets management for sensitive data",
        "Regular security scanning of images",
        "Optimize layer caching for faster builds"
    ]
    
    print("Container Deployment Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n4. Production Deployment Options")
    print("-" * 38)
    
    deployment_options = {
        "Docker Compose": "Simple multi-container orchestration",
        "Kubernetes": "Production-grade container orchestration",
        "Docker Swarm": "Native Docker clustering",
        "AWS ECS": "Amazon Elastic Container Service",
        "Google Cloud Run": "Serverless container platform",
        "Azure Container Instances": "Serverless containers on Azure",
        "Nomad": "HashiCorp container orchestration"
    }
    
    print("Production Deployment Platforms:")
    for platform, description in deployment_options.items():
        print(f"  {platform}: {description}")
    
    print("\n5. Monitoring and Observability")
    print("-" * 36)
    
    monitoring_tools = [
        "Prometheus: Metrics collection and alerting",
        "Grafana: Metrics visualization and dashboards",
        "Jaeger: Distributed tracing",
        "ELK Stack: Centralized logging",
        "DataDog: All-in-one monitoring platform",
        "New Relic: Application performance monitoring",
        "Sentry: Error tracking and performance monitoring"
    ]
    
    print("Monitoring Tools:")
    for tool in monitoring_tools:
        print(f"  - {tool}")
    
    print("\n6. Security Considerations")
    print("-" * 30)
    
    security_practices = [
        "Use minimal base images (alpine, distroless)",
        "Scan images for vulnerabilities",
        "Run containers as non-root user",
        "Use secrets management for sensitive data",
        "Implement network policies and firewalls",
        "Regular updates of base images and dependencies",
        "Input validation and sanitization",
        "Rate limiting and DDoS protection",
        "SSL/TLS encryption for API endpoints",
        "Authentication and authorization"
    ]
    
    print("Security Best Practices:")
    for i, practice in enumerate(security_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Scaling Strategies")
    print("-" * 24)
    
    scaling_strategies = {
        "Horizontal Scaling": "Add more container replicas",
        "Vertical Scaling": "Increase container resources",
        "Auto-scaling": "Dynamic scaling based on metrics",
        "Load Balancing": "Distribute traffic across replicas",
        "Caching": "Cache frequently accessed predictions",
        "Async Processing": "Use message queues for batch processing"
    }
    
    print("Scaling Strategies:")
    for strategy, description in scaling_strategies.items():
        print(f"  {strategy}: {description}")
    
    print("\n8. Deployment Commands")
    print("-" * 25)
    
    commands = f'''
# Build image
docker build -t image_classifier:latest {deployment_dir}

# Run container
docker run -p 8000:8000 image_classifier:latest

# Run with Docker Compose
cd {deployment_dir}
docker-compose up -d

# Check container status
docker ps

# View logs
docker-compose logs -f

# Scale service
docker-compose up -d --scale image_classifier=3

# Stop services
docker-compose down
'''
    
    print("Common Deployment Commands:")
    print(commands)
    
    print("\n9. API Testing")
    print("-" * 17)
    
    api_tests = '''
# Test health endpoint
curl http://localhost:8000/health

# Test model info
curl http://localhost:8000/model_info

# Test prediction (with image file)
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@test_image.jpg"

# View API documentation
# Open http://localhost:8000/docs in browser
'''
    
    print("API Testing Commands:")
    print(api_tests)
    
    print("\nContainerized deployment demonstration completed!")
    print(f"Generated deployment package: {deployment_dir}")
    print("\nFiles created:")
    print("  - Dockerfile (CPU inference)")
    print("  - Dockerfile.gpu (GPU inference)")
    print("  - Dockerfile.train (Training workloads)")
    print("  - docker-compose.yml (Service orchestration)")
    print("  - app.py (FastAPI application)")
    print("  - model_utils.py (Model definition)")
    print("  - requirements.txt (Dependencies)")
    print("  - build.sh & run.sh (Convenience scripts)")
    print("  - README.md (Documentation)")
    
    print("\nTo deploy:")
    print(f"1. cd {deployment_dir}")
    print("2. ./build.sh")
    print("3. ./run.sh")
    print("4. Test: curl http://localhost:8000/health")