import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime
import subprocess
import shutil

# Docker Integration and Containerized PyTorch Applications
# Note: This demonstrates Docker concepts and provides deployment utilities

class DockerConfig:
    """Configuration for Docker deployment"""
    
    def __init__(self):
        self.base_images = {
            "pytorch_cpu": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
            "pytorch_gpu": "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
            "python_slim": "python:3.9-slim",
            "ubuntu": "ubuntu:20.04"
        }
        
        self.common_packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "jupyter",
            "tqdm",
            "requests"
        ]
        
        self.ml_packages = [
            "torchvision",
            "torchaudio",
            "transformers",
            "datasets",
            "wandb",
            "tensorboard",
            "pytorch-lightning"
        ]

class DockerfileGenerator:
    """Generate Dockerfiles for PyTorch applications"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
    
    def generate_training_dockerfile(self, 
                                   base_image: str = "pytorch_cpu",
                                   requirements_file: str = "requirements.txt",
                                   app_name: str = "pytorch_app") -> str:
        """Generate Dockerfile for training applications"""
        
        base_img = self.config.base_images.get(base_image, base_image)
        
        dockerfile = f"""# PyTorch Training Application Dockerfile
FROM {base_img}

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.torch

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    wget \\
    git \\
    vim \\
    htop \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Copy application code
COPY . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs /app/checkpoints

# Set permissions
RUN chmod +x /app/scripts/* || true

# Expose ports
EXPOSE 8888 6006 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD python -c "import torch; print('PyTorch version:', torch.__version__)" || exit 1

# Default command
CMD ["python", "train.py"]
"""
        
        return dockerfile
    
    def generate_inference_dockerfile(self, 
                                    base_image: str = "pytorch_cpu",
                                    model_path: str = "model.pth",
                                    app_name: str = "pytorch_inference") -> str:
        """Generate Dockerfile for inference/serving applications"""
        
        base_img = self.config.base_images.get(base_image, base_image)
        
        dockerfile = f"""# PyTorch Inference Application Dockerfile
FROM {base_img}

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/{model_path}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional serving dependencies
RUN pip install fastapi uvicorn gunicorn

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY inference.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        return dockerfile
    
    def generate_jupyter_dockerfile(self, base_image: str = "pytorch_cpu") -> str:
        """Generate Dockerfile for Jupyter development environment"""
        
        base_img = self.config.base_images.get(base_image, base_image)
        
        dockerfile = f"""# PyTorch Jupyter Development Environment
FROM {base_img}

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    wget \\
    git \\
    vim \\
    htop \\
    graphviz \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \\
    jupyter \\
    jupyterlab \\
    notebook \\
    ipywidgets \\
    matplotlib \\
    seaborn \\
    plotly \\
    pandas \\
    numpy \\
    scikit-learn \\
    transformers \\
    datasets \\
    wandb \\
    tensorboard \\
    pytorch-lightning \\
    torchvision \\
    torchaudio

# Configure Jupyter
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Create workspace directories
RUN mkdir -p /workspace/notebooks /workspace/data /workspace/models

# Expose ports
EXPOSE 8888 6006

# Default command
CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0"]
"""
        
        return dockerfile
    
    def generate_multi_stage_dockerfile(self, app_name: str = "pytorch_app") -> str:
        """Generate multi-stage Dockerfile for optimized production builds"""
        
        dockerfile = f"""# Multi-stage PyTorch Application Dockerfile

# Stage 1: Build stage
FROM python:3.9-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Training stage (optional)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as training

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy training code
COPY train.py data_utils.py model.py ./
COPY data/ ./data/

# Run training (if building trained model)
# RUN python train.py --save-model

# Stage 3: Production stage
FROM python:3.9-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY inference.py model.py ./
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "inference.py"]
"""
        
        return dockerfile

class DockerComposeGenerator:
    """Generate Docker Compose files for PyTorch applications"""
    
    def __init__(self):
        pass
    
    def generate_training_compose(self, gpu_support: bool = False) -> str:
        """Generate Docker Compose for training setup"""
        
        gpu_config = ""
        if gpu_support:
            gpu_config = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]"""
        
        compose = f"""version: '3.8'

services:
  pytorch-training:
    build:
      context: .
      dockerfile: Dockerfile.training
    container_name: pytorch-training
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    environment:
      - PYTHONUNBUFFERED=1
      - TORCH_HOME=/app/.torch
      - WANDB_API_KEY=${{WANDB_API_KEY}}
    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard{gpu_config}
    restart: unless-stopped
    
  tensorboard:
    image: tensorflow/tensorflow:latest
    container_name: tensorboard
    command: tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    volumes:
      - ./logs:/logs
    ports:
      - "6007:6006"
    depends_on:
      - pytorch-training

  redis:
    image: redis:alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  pytorch_data:
  pytorch_models:
"""
        
        return compose
    
    def generate_inference_compose(self, replicas: int = 2) -> str:
        """Generate Docker Compose for inference/serving setup"""
        
        compose = f"""version: '3.8'

services:
  pytorch-api:
    build:
      context: .
      dockerfile: Dockerfile.inference
    ports:
      - "8000-8010:8000"
    environment:
      - MODEL_PATH=/app/models/model.pth
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    deploy:
      replicas: {replicas}
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - pytorch-api
    restart: unless-stopped

  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
    restart: unless-stopped

volumes:
  redis_data:
"""
        
        return compose
    
    def generate_development_compose(self) -> str:
        """Generate Docker Compose for development environment"""
        
        compose = """version: '3.8'

services:
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    container_name: pytorch-jupyter
    ports:
      - "8888:8888"
      - "6006:6006"
    volumes:
      - ./notebooks:/workspace/notebooks
      - ./data:/workspace/data
      - ./models:/workspace/models
      - ./scripts:/workspace/scripts
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=pytorch-development
    restart: unless-stopped

  postgres:
    image: postgres:13
    container_name: postgres-db
    environment:
      - POSTGRES_DB=experiments
      - POSTGRES_USER=pytorch
      - POSTGRES_PASSWORD=pytorch123
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  mlflow:
    image: python:3.9-slim
    container_name: mlflow-server
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --backend-store-uri postgresql://pytorch:pytorch123@postgres:5432/experiments
               --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres_data:
  mlflow_artifacts:
"""
        
        return compose

class ContainerizedModel:
    """Wrapper for containerized PyTorch models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.loaded = False
    
    def load_model(self):
        """Load model from file"""
        try:
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.loaded = True
            print(f"✓ Model loaded from {self.model_path}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Make prediction"""
        if not self.loaded:
            self.load_model()
        
        with torch.no_grad():
            input_data = input_data.to(self.device)
            output = self.model(input_data)
            return output
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for containerized service"""
        return {
            "status": "healthy" if self.loaded else "unhealthy",
            "model_path": self.model_path,
            "device": self.device,
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }

class DockerDeploymentManager:
    """Manage Docker deployments for PyTorch applications"""
    
    def __init__(self, project_name: str = "pytorch-app"):
        self.project_name = project_name
        self.dockerfile_generator = DockerfileGenerator(DockerConfig())
        self.compose_generator = DockerComposeGenerator()
    
    def create_project_structure(self, project_dir: str):
        """Create project directory structure"""
        
        project_path = Path(project_dir)
        project_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "app", "models", "data", "outputs", "logs", 
            "checkpoints", "scripts", "notebooks", "configs"
        ]
        
        for subdir in subdirs:
            (project_path / subdir).mkdir(exist_ok=True)
        
        print(f"✓ Created project structure in {project_dir}")
        
        return project_path
    
    def generate_requirements_txt(self, project_path: Path, 
                                include_dev: bool = False) -> str:
        """Generate requirements.txt file"""
        
        base_requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "tqdm>=4.62.0",
            "pyyaml>=6.0",
            "requests>=2.28.0"
        ]
        
        ml_requirements = [
            "pytorch-lightning>=2.0.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
            "optuna>=3.0.0",
            "hydra-core>=1.2.0"
        ]
        
        dev_requirements = [
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
            "ipywidgets>=8.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=2.20.0"
        ]
        
        serving_requirements = [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "gunicorn>=20.1.0",
            "pydantic>=1.10.0",
            "python-multipart>=0.0.6"
        ]
        
        all_requirements = base_requirements + ml_requirements + serving_requirements
        
        if include_dev:
            all_requirements.extend(dev_requirements)
        
        requirements_content = "\n".join(sorted(all_requirements))
        
        requirements_path = project_path / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements_content)
        
        print(f"✓ Generated requirements.txt with {len(all_requirements)} packages")
        
        return requirements_content
    
    def generate_inference_app(self, project_path: Path) -> str:
        """Generate FastAPI inference application"""
        
        app_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import json
import logging
from typing import List, Dict, Any
import uvicorn
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PyTorch Model API", version="1.0.0")

# Global model instance
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionRequest(BaseModel):
    data: List[List[float]]
    
class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence: List[float]

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    
    model_path = os.getenv("MODEL_PATH", "models/model.pth")
    
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "torch_version": torch.__version__
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to tensor
        input_tensor = torch.tensor(request.data, dtype=torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predictions = F.softmax(output, dim=1)
            
            # Get predictions and confidence
            pred_values = predictions.max(dim=1)[0].cpu().numpy().tolist()
            pred_classes = predictions.argmax(dim=1).cpu().numpy().tolist()
        
        return PredictionResponse(
            predictions=pred_classes,
            confidence=pred_values
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "device": str(device)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        app_path = project_path / "inference.py"
        with open(app_path, 'w') as f:
            f.write(app_code)
        
        print("✓ Generated FastAPI inference application")
        return app_code
    
    def generate_training_script(self, project_path: Path) -> str:
        """Generate training script for containerized training"""
        
        train_code = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import logging
import os
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class SyntheticDataset(Dataset):
    def __init__(self, size, input_dim, num_classes):
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_model(args):
    """Training function"""
    logger.info("Starting training...")
    
    # Create model
    model = SimpleModel(args.input_size, args.hidden_size, args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset
    dataset = SyntheticDataset(args.dataset_size, args.input_size, args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save training config
    config = vars(args)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training in Docker")
    parser.add_argument("--input-size", type=int, default=784, help="Input size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--dataset-size", type=int, default=10000, help="Dataset size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="/app/outputs", help="Output directory")
    
    args = parser.parse_args()
    train_model(args)
'''
        
        train_path = project_path / "train.py"
        with open(train_path, 'w') as f:
            f.write(train_code)
        
        print("✓ Generated containerized training script")
        return train_code
    
    def generate_docker_files(self, project_path: Path, 
                            deployment_type: str = "all"):
        """Generate Docker files for the project"""
        
        files_created = []
        
        if deployment_type in ["all", "training"]:
            # Training Dockerfile
            training_dockerfile = self.dockerfile_generator.generate_training_dockerfile()
            with open(project_path / "Dockerfile.training", 'w') as f:
                f.write(training_dockerfile)
            files_created.append("Dockerfile.training")
        
        if deployment_type in ["all", "inference"]:
            # Inference Dockerfile
            inference_dockerfile = self.dockerfile_generator.generate_inference_dockerfile()
            with open(project_path / "Dockerfile.inference", 'w') as f:
                f.write(inference_dockerfile)
            files_created.append("Dockerfile.inference")
        
        if deployment_type in ["all", "jupyter"]:
            # Jupyter Dockerfile
            jupyter_dockerfile = self.dockerfile_generator.generate_jupyter_dockerfile()
            with open(project_path / "Dockerfile.jupyter", 'w') as f:
                f.write(jupyter_dockerfile)
            files_created.append("Dockerfile.jupyter")
        
        if deployment_type in ["all", "production"]:
            # Multi-stage production Dockerfile
            production_dockerfile = self.dockerfile_generator.generate_multi_stage_dockerfile()
            with open(project_path / "Dockerfile", 'w') as f:
                f.write(production_dockerfile)
            files_created.append("Dockerfile")
        
        print(f"✓ Generated Docker files: {files_created}")
        return files_created
    
    def generate_compose_files(self, project_path: Path):
        """Generate Docker Compose files"""
        
        files_created = []
        
        # Training compose
        training_compose = self.compose_generator.generate_training_compose()
        with open(project_path / "docker-compose.training.yml", 'w') as f:
            f.write(training_compose)
        files_created.append("docker-compose.training.yml")
        
        # Inference compose
        inference_compose = self.compose_generator.generate_inference_compose()
        with open(project_path / "docker-compose.inference.yml", 'w') as f:
            f.write(inference_compose)
        files_created.append("docker-compose.inference.yml")
        
        # Development compose
        dev_compose = self.compose_generator.generate_development_compose()
        with open(project_path / "docker-compose.dev.yml", 'w') as f:
            f.write(dev_compose)
        files_created.append("docker-compose.dev.yml")
        
        print(f"✓ Generated Docker Compose files: {files_created}")
        return files_created
    
    def generate_deployment_scripts(self, project_path: Path):
        """Generate deployment and management scripts"""
        
        scripts_dir = project_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Build script
        build_script = '''#!/bin/bash
set -e

echo "Building PyTorch Docker images..."

# Build training image
echo "Building training image..."
docker build -f Dockerfile.training -t pytorch-app:training .

# Build inference image
echo "Building inference image..."
docker build -f Dockerfile.inference -t pytorch-app:inference .

# Build Jupyter image
echo "Building Jupyter image..."
docker build -f Dockerfile.jupyter -t pytorch-app:jupyter .

echo "All images built successfully!"
'''
        
        with open(scripts_dir / "build.sh", 'w') as f:
            f.write(build_script)
        
        # Deploy script
        deploy_script = '''#!/bin/bash
set -e

ENVIRONMENT=${1:-"development"}

echo "Deploying PyTorch application in $ENVIRONMENT mode..."

case $ENVIRONMENT in
    "development")
        docker-compose -f docker-compose.dev.yml up -d
        ;;
    "training")
        docker-compose -f docker-compose.training.yml up -d
        ;;
    "inference")
        docker-compose -f docker-compose.inference.yml up -d
        ;;
    *)
        echo "Unknown environment: $ENVIRONMENT"
        echo "Available environments: development, training, inference"
        exit 1
        ;;
esac

echo "Deployment completed!"
echo "Check status with: docker-compose ps"
'''
        
        with open(scripts_dir / "deploy.sh", 'w') as f:
            f.write(deploy_script)
        
        # Cleanup script
        cleanup_script = '''#!/bin/bash
set -e

echo "Cleaning up Docker resources..."

# Stop all containers
docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
docker-compose -f docker-compose.training.yml down 2>/dev/null || true
docker-compose -f docker-compose.inference.yml down 2>/dev/null || true

# Remove unused images
docker image prune -f

# Remove unused volumes
docker volume prune -f

echo "Cleanup completed!"
'''
        
        with open(scripts_dir / "cleanup.sh", 'w') as f:
            f.write(cleanup_script)
        
        # Make scripts executable
        for script in ["build.sh", "deploy.sh", "cleanup.sh"]:
            os.chmod(scripts_dir / script, 0o755)
        
        print("✓ Generated deployment scripts")
        return ["build.sh", "deploy.sh", "cleanup.sh"]

class DockerUtilities:
    """Utility functions for Docker operations"""
    
    @staticmethod
    def check_docker_availability() -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def get_docker_info() -> Dict[str, Any]:
        """Get Docker system information"""
        if not DockerUtilities.check_docker_availability():
            return {"available": False}
        
        try:
            result = subprocess.run(["docker", "system", "info", "--format", "json"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return {
                    "available": True,
                    "containers_running": info.get("ContainersRunning", 0),
                    "images": info.get("Images", 0),
                    "server_version": info.get("ServerVersion", "unknown"),
                    "architecture": info.get("Architecture", "unknown")
                }
        except Exception:
            pass
        
        return {"available": True, "info": "limited"}
    
    @staticmethod
    def optimize_dockerfile_for_caching() -> List[str]:
        """Best practices for Docker layer caching"""
        
        tips = [
            "Place frequently changing layers (like COPY . .) at the end",
            "Install dependencies before copying application code",
            "Use .dockerignore to exclude unnecessary files",
            "Combine RUN commands to reduce layers",
            "Use specific version tags for base images",
            "Cache package manager downloads",
            "Use multi-stage builds for production",
            "Order layers from least to most frequently changing"
        ]
        
        return tips

if __name__ == "__main__":
    print("Docker PyTorch Deployment")
    print("=" * 29)
    
    print("\n1. Docker Environment Check")
    print("-" * 28)
    
    docker_available = DockerUtilities.check_docker_availability()
    
    if docker_available:
        print("✓ Docker is available")
        docker_info = DockerUtilities.get_docker_info()
        
        if docker_info.get("info") != "limited":
            print(f"  Server Version: {docker_info.get('server_version', 'unknown')}")
            print(f"  Running Containers: {docker_info.get('containers_running', 0)}")
            print(f"  Images: {docker_info.get('images', 0)}")
    else:
        print("✗ Docker not available")
        print("  Install Docker to use containerized deployments")
    
    print("\n2. Project Setup")
    print("-" * 17)
    
    # Create deployment manager
    deployment_manager = DockerDeploymentManager("pytorch-demo")
    
    # Create project structure
    project_path = deployment_manager.create_project_structure("./pytorch-docker-demo")
    
    # Generate requirements
    requirements = deployment_manager.generate_requirements_txt(project_path, include_dev=True)
    print(f"  Generated requirements.txt with {len(requirements.split())} packages")
    
    print("\n3. Application Code Generation")
    print("-" * 34)
    
    # Generate training script
    deployment_manager.generate_training_script(project_path)
    
    # Generate inference app
    deployment_manager.generate_inference_app(project_path)
    
    print("\n4. Docker Files Generation")
    print("-" * 29)
    
    # Generate Docker files
    docker_files = deployment_manager.generate_docker_files(project_path, "all")
    
    # Generate Docker Compose files
    compose_files = deployment_manager.generate_compose_files(project_path)
    
    print("\n5. Deployment Scripts")
    print("-" * 22)
    
    # Generate deployment scripts
    scripts = deployment_manager.generate_deployment_scripts(project_path)
    
    print(f"  Generated scripts: {scripts}")
    
    print("\n6. Docker Best Practices")
    print("-" * 26)
    
    best_practices = [
        "Use specific version tags for reproducible builds",
        "Minimize image size with multi-stage builds",
        "Use .dockerignore to exclude unnecessary files",
        "Run containers as non-root users for security",
        "Use environment variables for configuration",
        "Implement proper health checks",
        "Log to stdout/stderr for container orchestration",
        "Use secrets management for sensitive data",
        "Optimize layer caching for faster builds",
        "Implement graceful shutdown handling",
        "Monitor resource usage and set limits",
        "Use container orchestration for production"
    ]
    
    print("Docker Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Optimization Tips")
    print("-" * 20)
    
    optimization_tips = DockerUtilities.optimize_dockerfile_for_caching()
    
    print("Dockerfile Optimization Tips:")
    for i, tip in enumerate(optimization_tips, 1):
        print(f"{i}. {tip}")
    
    print("\n8. Deployment Commands")
    print("-" * 23)
    
    commands = {
        "Build images": "cd pytorch-docker-demo && ./scripts/build.sh",
        "Deploy development": "./scripts/deploy.sh development",
        "Deploy training": "./scripts/deploy.sh training",
        "Deploy inference": "./scripts/deploy.sh inference",
        "View logs": "docker-compose logs -f",
        "Check status": "docker-compose ps",
        "Cleanup": "./scripts/cleanup.sh"
    }
    
    print("Common Deployment Commands:")
    for command, example in commands.items():
        print(f"  {command}: {example}")
    
    print("\n9. Production Considerations")
    print("-" * 31)
    
    production_considerations = [
        "Resource limits: Set memory and CPU limits",
        "Health checks: Implement comprehensive health endpoints",
        "Monitoring: Use Prometheus, Grafana for metrics",
        "Logging: Centralized logging with ELK stack",
        "Security: Regular security scans and updates",
        "Scaling: Use horizontal pod autoscaling",
        "Storage: Persistent volumes for model artifacts",
        "Networking: Service mesh for communication",
        "CI/CD: Automated build and deployment pipelines",
        "Backup: Regular backup of models and data"
    ]
    
    print("Production Considerations:")
    for consideration in production_considerations:
        print(f"  - {consideration}")
    
    print("\n10. Common Use Cases")
    print("-" * 21)
    
    use_cases = {
        "Model Training": "Reproducible training environments",
        "Model Serving": "Scalable inference APIs",
        "Development": "Consistent development environments",
        "CI/CD": "Automated testing and deployment",
        "Research": "Shareable research environments",
        "Edge Deployment": "Lightweight inference containers",
        "Batch Processing": "Scalable data processing",
        "Experimentation": "Isolated experiment environments"
    }
    
    print("Common Docker Use Cases:")
    for use_case, description in use_cases.items():
        print(f"  {use_case}: {description}")
    
    print("\nDocker PyTorch deployment demonstration completed!")
    print("Key components covered:")
    print("  - Dockerfile generation for different use cases")
    print("  - Docker Compose for multi-service deployments")
    print("  - Containerized training and inference applications")
    print("  - Deployment scripts and automation")
    print("  - Best practices and optimization techniques")
    print("  - Production deployment considerations")
    
    print("\nDocker enables:")
    print("  - Consistent environments across development and production")
    print("  - Easy scaling and orchestration of ML workloads")
    print("  - Reproducible model training and deployment")
    print("  - Simplified dependency management")
    print("  - Portable applications across different platforms")