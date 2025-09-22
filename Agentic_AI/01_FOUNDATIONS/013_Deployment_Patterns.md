# Deployment Patterns: Agent Production Strategies

## Deployment Architecture Overview

| Pattern | Scalability | Complexity | Reliability | Cost | Best For |
|---------|-------------|------------|-------------|------|----------|
| **Single Instance** | Low | Low | Low | Low | Prototypes, small apps |
| **Load Balanced** | Medium | Medium | Medium | Medium | Web services |
| **Microservices** | High | High | High | High | Enterprise systems |
| **Serverless** | Very High | Medium | High | Variable | Event-driven apps |
| **Edge Deployment** | Medium | High | Medium | Medium | Low-latency needs |
| **Hybrid Cloud** | Very High | Very High | Very High | High | Enterprise, compliance |

---

## Single Instance Deployment

### **Simple Docker Deployment**
```python
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  agent-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

### **Basic Health Monitoring**
```python
from fastapi import FastAPI, HTTPException
import psutil
import time

class SimpleAgentDeployment:
    def __init__(self):
        self.app = FastAPI()
        self.agent = YourAgent()
        self.start_time = time.time()
        self.request_count = 0
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "uptime": time.time() - self.start_time,
                "requests_processed": self.request_count,
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent()
            }
        
        @self.app.post("/process")
        async def process_request(request: dict):
            try:
                self.request_count += 1
                result = await self.agent.process(request)
                return {"result": result, "status": "success"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
```

---

## Load Balanced Deployment

### **Kubernetes Deployment**
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: your-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### **Load Balancer Configuration**
```python
import asyncio
import random
from typing import List

class LoadBalancedAgentService:
    def __init__(self, agent_instances: List[str]):
        self.agent_instances = agent_instances
        self.health_status = {instance: True for instance in agent_instances}
        self.request_counts = {instance: 0 for instance in agent_instances}
        
        # Start health monitoring
        asyncio.create_task(self.monitor_health())
    
    async def route_request(self, request):
        """Route request to healthy instance with least load"""
        healthy_instances = [
            instance for instance in self.agent_instances 
            if self.health_status[instance]
        ]
        
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Select instance with least load
        selected_instance = min(
            healthy_instances,
            key=lambda x: self.request_counts[x]
        )
        
        try:
            self.request_counts[selected_instance] += 1
            result = await self.send_request(selected_instance, request)
            return result
        finally:
            self.request_counts[selected_instance] -= 1
    
    async def monitor_health(self):
        """Continuously monitor instance health"""
        while True:
            for instance in self.agent_instances:
                try:
                    health_response = await self.check_health(instance)
                    self.health_status[instance] = health_response['healthy']
                except:
                    self.health_status[instance] = False
            
            await asyncio.sleep(10)  # Check every 10 seconds
```

---

## Microservices Architecture

### **Service Decomposition**
```python
# Agent Gateway Service
class AgentGateway:
    def __init__(self):
        self.nlp_service = NLPServiceClient()
        self.reasoning_service = ReasoningServiceClient()
        self.action_service = ActionServiceClient()
        self.memory_service = MemoryServiceClient()
    
    async def process_request(self, request):
        """Orchestrate request across microservices"""
        
        # 1. Natural Language Processing
        nlp_result = await self.nlp_service.process(request['text'])
        
        # 2. Retrieve relevant memories
        memories = await self.memory_service.retrieve_memories(
            nlp_result['intent'], nlp_result['entities']
        )
        
        # 3. Reasoning and planning
        plan = await self.reasoning_service.create_plan(
            nlp_result, memories
        )
        
        # 4. Execute actions
        results = []
        for action in plan['actions']:
            result = await self.action_service.execute(action)
            results.append(result)
        
        # 5. Store new memories
        await self.memory_service.store_interaction(
            request, nlp_result, plan, results
        )
        
        return {
            'response': plan['response'],
            'actions_taken': results,
            'confidence': plan['confidence']
        }

# Individual Service Example
class ReasoningService:
    def __init__(self):
        self.app = FastAPI()
        self.reasoning_engine = ReasoningEngine()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/create-plan")
        async def create_plan(request: PlanRequest):
            plan = await self.reasoning_engine.create_plan(
                request.intent,
                request.entities,
                request.context
            )
            return plan
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "service": "reasoning"}
```

### **Service Discovery and Communication**
```python
import consul
import aiohttp

class ServiceRegistry:
    def __init__(self, consul_host='localhost', consul_port=8500):
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self.services = {}
    
    def register_service(self, service_name, host, port, health_check_url):
        """Register service with discovery system"""
        self.consul.agent.service.register(
            name=service_name,
            service_id=f"{service_name}-{host}-{port}",
            address=host,
            port=port,
            check=consul.Check.http(health_check_url, interval="10s")
        )
    
    async def discover_service(self, service_name):
        """Discover healthy instances of a service"""
        _, services = self.consul.health.service(service_name, passing=True)
        return [
            f"http://{service['Service']['Address']}:{service['Service']['Port']}"
            for service in services
        ]

class ServiceClient:
    def __init__(self, service_name, service_registry):
        self.service_name = service_name
        self.service_registry = service_registry
        self.session = aiohttp.ClientSession()
    
    async def call_service(self, endpoint, data=None):
        """Call service with automatic discovery and failover"""
        instances = await self.service_registry.discover_service(self.service_name)
        
        for instance in instances:
            try:
                url = f"{instance}{endpoint}"
                if data:
                    async with self.session.post(url, json=data) as response:
                        return await response.json()
                else:
                    async with self.session.get(url) as response:
                        return await response.json()
            except Exception as e:
                continue  # Try next instance
        
        raise Exception(f"All instances of {self.service_name} failed")
```

---

## Serverless Deployment

### **AWS Lambda Deployment**
```python
import json
import boto3
from typing import Dict, Any

class ServerlessAgent:
    def __init__(self):
        self.agent = self.initialize_agent()
        
        # Initialize AWS clients
        self.dynamodb = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        self.ssm = boto3.client('ssm')
    
    def lambda_handler(self, event: Dict[str, Any], context):
        """AWS Lambda handler function"""
        try:
            # Extract request from API Gateway event
            if 'body' in event:
                request_body = json.loads(event['body'])
            else:
                request_body = event
            
            # Process request
            result = self.agent.process(request_body)
            
            # Return response
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'result': result,
                    'requestId': context.aws_request_id
                })
            }
            
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'error': str(e),
                    'requestId': context.aws_request_id
                })
            }
    
    def initialize_agent(self):
        """Initialize agent with optimized cold start"""
        # Load model/config from S3 or environment
        config = self.load_config()
        
        # Initialize with minimal dependencies
        agent = OptimizedAgent(config)
        
        return agent

# serverless.yml for Serverless Framework
"""
service: agent-service

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  environment:
    DYNAMODB_TABLE: ${self:service}-${opt:stage, self:provider.stage}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - dynamodb:Query
        - dynamodb:Scan
        - dynamodb:GetItem
        - dynamodb:PutItem
        - dynamodb:UpdateItem
        - dynamodb:DeleteItem
      Resource: 
        - "arn:aws:dynamodb:${opt:region, self:provider.region}:*:table/${self:provider.environment.DYNAMODB_TABLE}"

functions:
  agent:
    handler: handler.lambda_handler
    timeout: 30
    memory: 512
    events:
      - http:
          path: /process
          method: post
          cors: true

resources:
  Resources:
    AgentDynamoDbTable:
      Type: 'AWS::DynamoDB::Table'
      Properties:
        TableName: ${self:provider.environment.DYNAMODB_TABLE}
        AttributeDefinitions:
          - AttributeName: id
            AttributeType: S
        KeySchema:
          - AttributeName: id
            KeyType: HASH
        BillingMode: PAY_PER_REQUEST
"""
```

### **Cold Start Optimization**
```python
import time
import pickle
import os

class OptimizedServerlessAgent:
    _cached_model = None
    _cache_timestamp = None
    _cache_ttl = 300  # 5 minutes
    
    def __init__(self):
        self.model = self.get_cached_model()
    
    @classmethod
    def get_cached_model(cls):
        """Get model with caching to reduce cold starts"""
        current_time = time.time()
        
        # Check if cache is valid
        if (cls._cached_model is not None and 
            cls._cache_timestamp is not None and
            current_time - cls._cache_timestamp < cls._cache_ttl):
            return cls._cached_model
        
        # Load model
        model_path = os.getenv('MODEL_PATH', '/tmp/model.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Download and cache model
            model = cls.download_and_cache_model(model_path)
        
        # Update cache
        cls._cached_model = model
        cls._cache_timestamp = current_time
        
        return model
    
    @classmethod
    def download_and_cache_model(cls, cache_path):
        """Download model and cache locally"""
        # Download from S3 or model registry
        s3_client = boto3.client('s3')
        s3_client.download_file(
            os.getenv('MODEL_BUCKET'),
            os.getenv('MODEL_KEY'),
            cache_path
        )
        
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
```

---

## Edge Deployment

### **Edge Computing Setup**
```python
import edge_tpu.detection.engine as detection_engine
import numpy as np

class EdgeAgent:
    def __init__(self, model_path, edge_device='cpu'):
        self.edge_device = edge_device
        self.model = self.load_optimized_model(model_path)
        
        # Initialize edge-specific optimizations
        if edge_device == 'tpu':
            self.inference_engine = detection_engine.DetectionEngine(model_path)
        elif edge_device == 'gpu':
            self.setup_gpu_optimization()
        else:
            self.setup_cpu_optimization()
    
    def load_optimized_model(self, model_path):
        """Load model optimized for edge deployment"""
        if self.edge_device == 'tpu':
            # Load TensorFlow Lite model for Edge TPU
            return self.load_tflite_model(model_path)
        elif self.edge_device == 'gpu':
            # Load TensorRT optimized model
            return self.load_tensorrt_model(model_path)
        else:
            # Load quantized CPU model
            return self.load_quantized_model(model_path)
    
    def process_with_edge_optimization(self, input_data):
        """Process input with edge-specific optimizations"""
        
        # Preprocess for edge device
        processed_input = self.preprocess_for_edge(input_data)
        
        # Run inference
        if self.edge_device == 'tpu':
            result = self.tpu_inference(processed_input)
        elif self.edge_device == 'gpu':
            result = self.gpu_inference(processed_input)
        else:
            result = self.cpu_inference(processed_input)
        
        # Postprocess results
        return self.postprocess_edge_result(result)
    
    def preprocess_for_edge(self, input_data):
        """Optimize preprocessing for edge constraints"""
        # Reduce precision, resize inputs, etc.
        if isinstance(input_data, str):
            # Limit text length for processing
            input_data = input_data[:512]
        
        return input_data

# Edge deployment with Docker
"""
# Dockerfile.edge
FROM arm64v8/python:3.9-slim

# Install edge-specific dependencies
RUN apt-get update && apt-get install -y \
    libedgetpu1-std \
    python3-edgetpu

WORKDIR /app
COPY requirements.edge.txt .
RUN pip install -r requirements.edge.txt

COPY . .

# Optimize for ARM architecture
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

CMD ["python", "edge_agent.py"]
"""
```

---

## High Availability Patterns

### **Multi-Region Deployment**
```python
class MultiRegionDeployment:
    def __init__(self):
        self.regions = {
            'us-east-1': {'primary': True, 'endpoint': 'https://agent-us-east.example.com'},
            'eu-west-1': {'primary': False, 'endpoint': 'https://agent-eu-west.example.com'},
            'ap-southeast-1': {'primary': False, 'endpoint': 'https://agent-ap-se.example.com'}
        }
        self.health_checker = RegionHealthChecker(self.regions)
    
    async def route_to_nearest_healthy_region(self, user_location, request):
        """Route request to nearest healthy region"""
        
        # Get healthy regions
        healthy_regions = await self.health_checker.get_healthy_regions()
        
        if not healthy_regions:
            raise Exception("No healthy regions available")
        
        # Calculate distances and select nearest
        nearest_region = self.calculate_nearest_region(
            user_location, healthy_regions
        )
        
        # Route request
        return await self.send_to_region(nearest_region, request)
    
    async def handle_region_failover(self, failed_region):
        """Handle failover when a region goes down"""
        
        # Mark region as unhealthy
        self.regions[failed_region]['healthy'] = False
        
        # Redirect traffic to backup regions
        backup_regions = self.get_backup_regions(failed_region)
        
        # Scale up backup regions if needed
        for region in backup_regions:
            await self.scale_up_region(region)
        
        # Update load balancer configuration
        await self.update_load_balancer_config()

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if self.should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
            
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

---

## Monitoring and Observability

### **Comprehensive Monitoring Setup**
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import structlog
import opentelemetry.trace as trace

class ProductionMonitoring:
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'agent_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'agent_active_requests',
            'Number of active requests'
        )
        
        # Structured logging
        self.logger = structlog.get_logger()
        
        # Distributed tracing
        self.tracer = trace.get_tracer(__name__)
    
    def track_request(self, method, endpoint):
        """Decorator to track request metrics"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(f"{method} {endpoint}") as span:
                    start_time = time.time()
                    self.active_requests.inc()
                    
                    try:
                        result = await func(*args, **kwargs)
                        status = 'success'
                        
                        # Add span attributes
                        span.set_attribute("http.method", method)
                        span.set_attribute("http.route", endpoint)
                        span.set_attribute("http.status_code", 200)
                        
                        return result
                        
                    except Exception as e:
                        status = 'error'
                        span.set_attribute("error", True)
                        span.set_attribute("error.message", str(e))
                        
                        self.logger.error(
                            "Request failed",
                            method=method,
                            endpoint=endpoint,
                            error=str(e),
                            trace_id=span.get_span_context().trace_id
                        )
                        raise
                        
                    finally:
                        duration = time.time() - start_time
                        
                        # Record metrics
                        self.request_count.labels(
                            method=method,
                            endpoint=endpoint,
                            status=status
                        ).inc()
                        
                        self.request_duration.labels(
                            method=method,
                            endpoint=endpoint
                        ).observe(duration)
                        
                        self.active_requests.dec()
                        
                        # Log request
                        self.logger.info(
                            "Request completed",
                            method=method,
                            endpoint=endpoint,
                            duration=duration,
                            status=status
                        )
            
            return wrapper
        return decorator
```

---

## Deployment Best Practices

### **Security Considerations**
```python
import secrets
import jwt
from cryptography.fernet import Fernet

class SecurityLayer:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.jwt_secret = secrets.token_urlsafe(32)
    
    def encrypt_sensitive_data(self, data):
        """Encrypt sensitive data before storage"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data):
        """Decrypt sensitive data after retrieval"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_api_token(self, user_id, permissions):
        """Generate JWT token for API access"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': time.time() + 3600  # 1 hour expiry
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
    
    def validate_api_token(self, token):
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
```

### **Configuration Management**
```python
import os
from pydantic import BaseSettings

class DeploymentConfig(BaseSettings):
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API Configuration
    api_key: str
    max_requests_per_minute: int = 100
    request_timeout: int = 30
    
    # Database
    database_url: str
    redis_url: str
    
    # Security
    secret_key: str
    allowed_origins: list = ["*"]
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Model Configuration
    model_path: str
    model_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"

# Environment-specific configurations
class ProductionConfig(DeploymentConfig):
    debug: bool = False
    log_level: str = "WARNING"
    allowed_origins: list = ["https://yourdomain.com"]

class DevelopmentConfig(DeploymentConfig):
    debug: bool = True
    log_level: str = "DEBUG"
```

This comprehensive deployment guide provides patterns for deploying AI agents across various environments and scales, from simple single-instance deployments to complex multi-region, high-availability systems.
