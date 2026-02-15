import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import shutil
import tempfile
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import requests

# Sample Models for TorchServe
class ServeCompatibleCNN(nn.Module):
    """CNN designed for TorchServe deployment"""
    
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
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImageClassificationModel(nn.Module):
    """Image classification model for TorchServe"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Simplified ResNet-like architecture
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual blocks (simplified)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride=2))
        
        for _ in range(1, num_blocks):
            layers.append(self._make_block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# TorchServe Model Archive Creator
class TorchServeArchiver:
    """Create model archives for TorchServe deployment"""
    
    def __init__(self, work_dir: str = "torchserve_workspace"):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
        # Create subdirectories
        self.models_dir = os.path.join(work_dir, "models")
        self.handlers_dir = os.path.join(work_dir, "handlers")
        self.archives_dir = os.path.join(work_dir, "model-store")
        
        for dir_path in [self.models_dir, self.handlers_dir, self.archives_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_model(self, model: nn.Module, model_name: str,
                  example_input: torch.Tensor) -> str:
        """Save model for TorchServe archiving"""
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        
        # Save model state dict
        model.eval()
        torch.save(model.state_dict(), model_path)
        
        print(f"✓ Model saved: {model_path}")
        return model_path
    
    def create_custom_handler(self, handler_name: str, 
                            model_class_name: str = "ServeCompatibleCNN",
                            num_classes: int = 10) -> str:
        """Create custom handler for TorchServe"""
        
        handler_code = f'''
import torch
import torch.nn.functional as F
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
import json
import logging
import io
from PIL import Image

logger = logging.getLogger(__name__)

class {handler_name}(BaseHandler):
    """
    Custom handler for image classification using {model_class_name}
    """
    
    def __init__(self):
        super().__init__()
        self.transform = None
        self.class_names = None
    
    def initialize(self, context):
        """Initialize model and preprocessing"""
        
        # Load model
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Initialize model architecture
        from {model_class_name} import {model_class_name}
        self.model = {model_class_name}(num_classes={num_classes})
        
        # Load model weights
        model_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names if available
        class_names_path = os.path.join(model_dir, "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            self.class_names = [f"class_{{i}}" for i in range({num_classes})]
        
        logger.info("Model initialized successfully")
    
    def preprocess(self, data):
        """Preprocess input data"""
        
        images = []
        
        for row in data:
            # Handle different input formats
            if isinstance(row, dict):
                # JSON input with base64 encoded image
                if "data" in row:
                    image_data = row["data"]
                    if isinstance(image_data, str):
                        # Base64 encoded image
                        import base64
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    else:
                        # Direct image bytes
                        image = Image.open(io.BytesIO(image_data)).convert('RGB')
                else:
                    raise ValueError("Invalid input format")
            else:
                # Direct image bytes
                image = Image.open(io.BytesIO(row)).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Fallback basic preprocessing
                image_tensor = transforms.ToTensor()(image)
            
            images.append(image_tensor)
        
        return torch.stack(images)
    
    def inference(self, data):
        """Run model inference"""
        
        with torch.no_grad():
            outputs = self.model(data)
            # Apply softmax for probabilities
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities
    
    def postprocess(self, data):
        """Postprocess model outputs"""
        
        results = []
        
        for output in data:
            # Get top predictions
            top_probs, top_indices = torch.topk(output, k=min(5, len(output)))
            
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                predictions.append({{
                    "class": self.class_names[idx.item()],
                    "probability": prob.item()
                }})
            
            results.append({{
                "predictions": predictions,
                "predicted_class": self.class_names[top_indices[0].item()],
                "confidence": top_probs[0].item()
            }})
        
        return results
'''
        
        handler_path = os.path.join(self.handlers_dir, f"{handler_name.lower()}.py")
        
        with open(handler_path, 'w') as f:
            f.write(handler_code)
        
        print(f"✓ Custom handler created: {handler_path}")
        return handler_path
    
    def create_model_archive(self, model_name: str, 
                           model_path: str,
                           handler_path: Optional[str] = None,
                           extra_files: Optional[List[str]] = None,
                           version: str = "1.0") -> str:
        """Create TorchServe model archive"""
        
        archive_name = f"{model_name}.mar"
        archive_path = os.path.join(self.archives_dir, archive_name)
        
        # Build torch-model-archiver command
        cmd = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", version,
            "--serialized-file", model_path,
            "--export-path", self.archives_dir,
            "--force"  # Overwrite if exists
        ]
        
        # Add handler if provided
        if handler_path:
            cmd.extend(["--handler", handler_path])
        
        # Add extra files if provided
        if extra_files:
            cmd.extend(["--extra-files", ",".join(extra_files)])
        
        try:
            # Run torch-model-archiver
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Model archive created: {archive_path}")
            return archive_path
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Archive creation failed: {e.stderr}")
            return None
        except FileNotFoundError:
            print("✗ torch-model-archiver not found. Install TorchServe first:")
            print("pip install torchserve torch-model-archiver torch-workflow-archiver")
            return None

# TorchServe Configuration Manager
class TorchServeConfig:
    """Manage TorchServe configuration"""
    
    def __init__(self, config_dir: str = "torchserve_config"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def create_config_properties(self, 
                                inference_address: str = "http://0.0.0.0:8080",
                                management_address: str = "http://0.0.0.0:8081",
                                metrics_address: str = "http://0.0.0.0:8082",
                                **kwargs) -> str:
        """Create TorchServe configuration file"""
        
        config = {
            "inference_address": inference_address,
            "management_address": management_address,
            "metrics_address": metrics_address,
            "grpc_inference_port": "7070",
            "grpc_management_port": "7071",
            "enable_envvars_config": "true",
            "install_py_dep_per_model": "true",
            "enable_metrics_api": "true",
            "metrics_format": "prometheus",
            "number_of_netty_threads": "4",
            "job_queue_size": "10",
            "number_of_gpu": "0",
            "batch_size": "1",
            "max_batch_delay": "5000",
            "response_timeout": "120",
            "unregister_model_timeout": "120",
            "decode_input_request": "true",
            "workflow_store": "./workflow-store",
            **kwargs
        }
        
        config_path = os.path.join(self.config_dir, "config.properties")
        
        with open(config_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}={value}\\n")
        
        print(f"✓ TorchServe config created: {config_path}")
        return config_path
    
    def create_model_config(self, model_name: str,
                          batch_size: int = 1,
                          max_batch_delay: int = 5000,
                          response_timeout: int = 120,
                          **kwargs) -> str:
        """Create model-specific configuration"""
        
        model_config = {
            "minWorkers": 1,
            "maxWorkers": 3,
            "batchSize": batch_size,
            "maxBatchDelay": max_batch_delay,
            "responseTimeout": response_timeout,
            **kwargs
        }
        
        config_path = os.path.join(self.config_dir, f"{model_name}_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✓ Model config created: {config_path}")
        return config_path

# TorchServe Client
class TorchServeClient:
    """Client for interacting with TorchServe"""
    
    def __init__(self, 
                 inference_url: str = "http://localhost:8080",
                 management_url: str = "http://localhost:8081"):
        self.inference_url = inference_url
        self.management_url = management_url
    
    def health_check(self) -> bool:
        """Check if TorchServe is running"""
        
        try:
            response = requests.get(f"{self.inference_url}/ping", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List registered models"""
        
        try:
            response = requests.get(f"{self.management_url}/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to list models: {e}")
            return []
    
    def register_model(self, model_name: str, mar_file_path: str,
                      initial_workers: int = 1) -> bool:
        """Register a model with TorchServe"""
        
        params = {
            "model_name": model_name,
            "url": mar_file_path,
            "initial_workers": initial_workers,
            "synchronous": "true"
        }
        
        try:
            response = requests.post(f"{self.management_url}/models", params=params)
            response.raise_for_status()
            print(f"✓ Model registered: {model_name}")
            return True
        except requests.RequestException as e:
            print(f"✗ Model registration failed: {e}")
            return False
    
    def unregister_model(self, model_name: str) -> bool:
        """Unregister a model from TorchServe"""
        
        try:
            response = requests.delete(f"{self.management_url}/models/{model_name}")
            response.raise_for_status()
            print(f"✓ Model unregistered: {model_name}")
            return True
        except requests.RequestException as e:
            print(f"✗ Model unregistration failed: {e}")
            return False
    
    def predict(self, model_name: str, data: bytes) -> Dict[str, Any]:
        """Make prediction request"""
        
        try:
            response = requests.post(
                f"{self.inference_url}/predictions/{model_name}",
                data=data,
                headers={"Content-Type": "application/octet-stream"}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def get_model_description(self, model_name: str) -> Dict[str, Any]:
        """Get model description and metadata"""
        
        try:
            response = requests.get(f"{self.management_url}/models/{model_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to get model description: {e}")
            return {}
    
    def scale_workers(self, model_name: str, min_workers: int, max_workers: int) -> bool:
        """Scale model workers"""
        
        params = {
            "min_worker": min_workers,
            "max_worker": max_workers,
            "synchronous": "true"
        }
        
        try:
            response = requests.put(f"{self.management_url}/models/{model_name}", params=params)
            response.raise_for_status()
            print(f"✓ Workers scaled for {model_name}: {min_workers}-{max_workers}")
            return True
        except requests.RequestException as e:
            print(f"✗ Worker scaling failed: {e}")
            return False

# Docker Deployment Helper
class TorchServeDocker:
    """Helper for Docker-based TorchServe deployment"""
    
    @staticmethod
    def create_dockerfile(base_image: str = "pytorch/torchserve:latest") -> str:
        """Create Dockerfile for TorchServe deployment"""
        
        dockerfile_content = f'''
FROM {base_image}

# Install additional dependencies if needed
RUN pip install pillow requests

# Copy model archives
COPY model-store /home/model-server/model-store

# Copy configuration
COPY config.properties /home/model-server/config.properties

# Expose ports
EXPOSE 8080 8081 8082 7070 7071

# Start TorchServe
CMD ["torchserve", \\
     "--start", \\
     "--model-store", "/home/model-server/model-store", \\
     "--ts-config", "/home/model-server/config.properties"]
'''
        
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_content)
        
        print("✓ Dockerfile created")
        return dockerfile_content
    
    @staticmethod
    def create_docker_compose(service_name: str = "torchserve") -> str:
        """Create Docker Compose file"""
        
        compose_content = f'''
version: '3.8'

services:
  {service_name}:
    build: .
    ports:
      - "8080:8080"  # Inference API
      - "8081:8081"  # Management API
      - "8082:8082"  # Metrics API
    volumes:
      - ./model-store:/home/model-server/model-store
      - ./logs:/home/model-server/logs
    environment:
      - JAVA_OPTS=-Xmx2g
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
'''
        
        with open("docker-compose.yml", 'w') as f:
            f.write(compose_content)
        
        print("✓ Docker Compose file created")
        return compose_content

if __name__ == "__main__":
    print("TorchServe Model Serving")
    print("=" * 28)
    
    # Create sample model
    model = ServeCompatibleCNN(num_classes=10)
    
    # Create example input
    example_input = torch.randn(1, 3, 224, 224)
    
    print("\n1. Model Archive Creation")
    print("-" * 29)
    
    archiver = TorchServeArchiver("demo_torchserve")
    
    # Save model
    model_path = archiver.save_model(model, "demo_cnn", example_input)
    
    # Create custom handler
    handler_path = archiver.create_custom_handler(
        "ImageClassificationHandler", 
        "ServeCompatibleCNN", 
        num_classes=10
    )
    
    # Create class names file
    class_names = [f"class_{i}" for i in range(10)]
    class_names_path = os.path.join(archiver.models_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    
    # Create model archive
    archive_path = archiver.create_model_archive(
        "demo_cnn",
        model_path,
        handler_path,
        extra_files=[class_names_path]
    )
    
    print("\n2. TorchServe Configuration")
    print("-" * 31)
    
    config_manager = TorchServeConfig("demo_torchserve/config")
    
    # Create main configuration
    config_path = config_manager.create_config_properties(
        number_of_gpu="0",  # CPU only for demo
        batch_size="4",
        max_batch_delay="100"
    )
    
    # Create model-specific configuration
    model_config_path = config_manager.create_model_config(
        "demo_cnn",
        batch_size=4,
        max_batch_delay=100
    )
    
    print("\n3. Docker Deployment Setup")
    print("-" * 31)
    
    docker_helper = TorchServeDocker()
    
    # Create Dockerfile
    dockerfile = docker_helper.create_dockerfile()
    
    # Create Docker Compose
    compose_file = docker_helper.create_docker_compose()
    
    print("\n4. TorchServe Client Operations")
    print("-" * 35)
    
    client = TorchServeClient()
    
    # Check if TorchServe is running
    if client.health_check():
        print("✓ TorchServe is running")
        
        # List current models
        models = client.list_models()
        print(f"Current models: {len(models)}")
        
        # Try to register our model (will fail if TorchServe not running)
        if archive_path:
            success = client.register_model("demo_cnn", archive_path)
            if success:
                # Get model description
                description = client.get_model_description("demo_cnn")
                print(f"Model description: {description}")
                
                # Scale workers
                client.scale_workers("demo_cnn", 1, 3)
    else:
        print("✗ TorchServe is not running")
        print("Start TorchServe with:")
        print(f"torchserve --start --model-store {archiver.archives_dir} --ts-config {config_path}")
    
    print("\n5. Production Deployment Guide")
    print("-" * 36)
    
    deployment_steps = [
        "1. Prepare model archives with proper handlers",
        "2. Configure TorchServe for your environment",
        "3. Set up Docker container with health checks",
        "4. Deploy with orchestration (Kubernetes, Docker Swarm)",
        "5. Configure load balancer and auto-scaling",
        "6. Set up monitoring and logging",
        "7. Implement model versioning strategy",
        "8. Test with production-like traffic"
    ]
    
    print("Production Deployment Steps:")
    for step in deployment_steps:
        print(f"  {step}")
    
    print("\n6. TorchServe Best Practices")
    print("-" * 33)
    
    best_practices = [
        "Use custom handlers for preprocessing/postprocessing",
        "Configure appropriate batch sizes for your use case",
        "Set up health checks and monitoring",
        "Use model versioning for A/B testing",
        "Configure auto-scaling based on metrics",
        "Implement proper error handling in handlers",
        "Use GPU workers for compute-intensive models",
        "Monitor memory usage and optimize accordingly",
        "Set appropriate timeouts for requests",
        "Use metrics API for performance monitoring",
        "Implement graceful model updates",
        "Test thoroughly before production deployment"
    ]
    
    print("TorchServe Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. API Usage Examples")
    print("-" * 25)
    
    api_examples = '''
# Management API Examples:

# List models
curl http://localhost:8081/models

# Register model
curl -X POST "http://localhost:8081/models?url=demo_cnn.mar&initial_workers=1"

# Scale workers
curl -X PUT "http://localhost:8081/models/demo_cnn?min_worker=1&max_worker=3"

# Get model info
curl http://localhost:8081/models/demo_cnn

# Unregister model
curl -X DELETE http://localhost:8081/models/demo_cnn

# Inference API Examples:

# Make prediction
curl -X POST http://localhost:8080/predictions/demo_cnn -T image.jpg

# Health check
curl http://localhost:8080/ping

# Metrics
curl http://localhost:8082/metrics
'''
    
    print(api_examples)
    
    print("\n8. Monitoring and Metrics")
    print("-" * 29)
    
    monitoring_metrics = [
        "Request latency and throughput",
        "Model accuracy and drift detection",
        "Resource utilization (CPU, memory, GPU)",
        "Error rates and failure patterns",
        "Queue depths and worker utilization",
        "Model loading and unloading times",
        "Batch processing efficiency",
        "Network I/O and request sizes"
    ]
    
    print("Key Metrics to Monitor:")
    for i, metric in enumerate(monitoring_metrics, 1):
        print(f"{i}. {metric}")
    
    print("\nTorchServe deployment demonstration completed!")
    print("Generated files:")
    print("  - demo_torchserve/ (workspace with models, handlers, archives)")
    print("  - Dockerfile (container definition)")
    print("  - docker-compose.yml (service definition)")
    print("  - config.properties (TorchServe configuration)")
    
    print("\nTo deploy:")
    print("1. Start TorchServe: torchserve --start --model-store demo_torchserve/model-store")
    print("2. Or use Docker: docker-compose up")
    print("3. Test with: curl http://localhost:8080/ping")