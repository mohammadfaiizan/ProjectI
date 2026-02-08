# Serving Architectures and APIs

## Table of Contents

1. [Introduction to Model Serving](#introduction-to-model-serving)
2. [REST APIs for ML](#rest-apis-for-ml)
3. [gRPC for High-Performance Serving](#grpc-for-high-performance-serving)
4. [Model Serving Frameworks](#model-serving-frameworks)
5. [Batch vs Online Inference](#batch-vs-online-inference)
6. [Model Formats](#model-formats)
7. [API Design Patterns](#api-design-patterns)
8. [Performance Optimization](#performance-optimization)
9. [Security Considerations](#security-considerations)
10. [Key Takeaways](#key-takeaways)

## Introduction to Model Serving

Model serving is the process of making trained ML models available for inference in production environments. Effective model serving requires:

- **Low Latency**: Fast response times for real-time applications
- **High Throughput**: Handle many concurrent requests
- **Scalability**: Scale up/down based on demand
- **Reliability**: High availability and fault tolerance
- **Versioning**: Support multiple model versions
- **Monitoring**: Track performance and errors

### Serving Architecture

```
┌─────────────┐
│   Client    │
│ Application │
└──────┬──────┘
       │ HTTP/gRPC
       ▼
┌─────────────┐
│   API       │
│  Gateway    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Model     │     │   Model     │     │   Model     │
│  Server 1   │     │  Server 2   │     │  Server N   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                  ┌────────▼────────┐
                  │   Load          │
                  │   Balancer      │
                  └─────────────────┘
```

## REST APIs for ML

REST (Representational State Transfer) APIs are the most common approach for model serving.

### Basic Flask API

```python
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        return jsonify({
            'prediction': float(prediction),
            'probabilities': probability
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### FastAPI Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="ML Model API")

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    probabilities: list[float]

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        
        return PredictionResponse(
            prediction=float(prediction),
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Request/Response Formats

**Single Prediction**:
```json
POST /predict
{
  "features": [1.0, 2.0, 3.0, 4.0]
}

Response:
{
  "prediction": 1,
  "probabilities": [0.1, 0.9]
}
```

**Batch Prediction**:
```json
POST /predict/batch
{
  "instances": [
    {"features": [1.0, 2.0, 3.0]},
    {"features": [4.0, 5.0, 6.0]}
  ]
}

Response:
{
  "predictions": [
    {"prediction": 0, "probabilities": [0.8, 0.2]},
    {"prediction": 1, "probabilities": [0.3, 0.7]}
  ]
}
```

## gRPC for High-Performance Serving

gRPC provides high-performance, low-latency communication using Protocol Buffers.

### Protocol Buffer Definition

```protobuf
// model_service.proto
syntax = "proto3";

service ModelService {
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc PredictBatch(BatchPredictRequest) returns (BatchPredictResponse);
}

message PredictRequest {
  repeated float features = 1;
}

message PredictResponse {
  float prediction = 1;
  repeated float probabilities = 2;
}

message BatchPredictRequest {
  repeated PredictRequest instances = 1;
}

message BatchPredictResponse {
  repeated PredictResponse predictions = 2;
}
```

### gRPC Server

```python
import grpc
from concurrent import futures
import model_service_pb2
import model_service_pb2_grpc
import pickle
import numpy as np

class ModelServicer(model_service_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    
    def Predict(self, request, context):
        features = np.array(request.features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return model_service_pb2.PredictResponse(
            prediction=float(prediction),
            probabilities=probabilities.tolist()
        )
    
    def PredictBatch(self, request, context):
        predictions = []
        for instance in request.instances:
            features = np.array(instance.features).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            predictions.append(
                model_service_pb2.PredictResponse(
                    prediction=float(prediction),
                    probabilities=probabilities.tolist()
                )
            )
        
        return model_service_pb2.BatchPredictResponse(predictions=predictions)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_service_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

### gRPC Client

```python
import grpc
import model_service_pb2
import model_service_pb2_grpc

def predict(features):
    channel = grpc.insecure_channel('localhost:50051')
    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    
    request = model_service_pb2.PredictRequest(features=features)
    response = stub.Predict(request)
    
    return response.prediction, response.probabilities
```

## Model Serving Frameworks

### TensorFlow Serving

TensorFlow Serving is optimized for TensorFlow models.

**Docker Deployment**:
```bash
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

**REST API**:
```python
import requests
import json

data = {
    "instances": [[1.0, 2.0, 3.0, 4.0]]
}

response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=json.dumps(data)
)
predictions = response.json()['predictions']
```

**gRPC API**:
```python
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'my_model'
request.model_spec.signature_name = 'serving_default'
request.inputs['features'].CopyFrom(tf.make_tensor_proto(features))

result = stub.Predict(request, 10.0)
```

### NVIDIA Triton Inference Server

Triton supports multiple frameworks and provides optimized GPU inference.

**Model Repository Structure**:
```
model_repository/
├── my_model/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
```

**Config File**:
```protobuf
name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [4]
  }
]
output [
  {
    name: "prediction"
    data_type: TYPE_FP32
    dims: [1]
  }
]
```

**Client**:
```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url='localhost:8000')

inputs = httpclient.InferInput('features', [1, 4], 'FP32')
inputs.set_data_from_numpy(features)

outputs = httpclient.InferRequestedOutput('prediction')

response = client.infer('my_model', inputs=[inputs], outputs=[outputs])
prediction = response.as_numpy('prediction')
```

### TorchServe

TorchServe is PyTorch's model serving framework.

**Model Archive**:
```bash
torch-model-archiver --model-name my_model \
  --version 1.0 \
  --serialized-file model.pth \
  --handler handler.py \
  --export-path model_store
```

**Handler**:
```python
import torch
import json

class ModelHandler:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def initialize(self, context):
        model_path = context.system_properties['model_dir']
        self.model = torch.load(f'{model_path}/model.pth')
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, requests):
        inputs = []
        for request in requests:
            data = json.loads(request['body'])
            inputs.append(torch.tensor(data['features']))
        return torch.stack(inputs)
    
    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(inputs.to(self.device))
        return outputs
    
    def postprocess(self, outputs):
        predictions = outputs.cpu().numpy().tolist()
        return [{'prediction': pred} for pred in predictions]
```

**Start Server**:
```bash
torchserve --start --model-store model_store --models my_model=my_model.mar
```

## Batch vs Online Inference

### Online Inference

Real-time predictions with low latency requirements.

**Characteristics**:
- Latency: <100ms typically
- Throughput: Variable based on traffic
- Use Case: User-facing applications
- Architecture: Stateless, horizontally scalable

**Implementation**:
```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Synchronous prediction
    prediction = model.predict(request.features)
    return {"prediction": prediction}
```

### Batch Inference

Process large volumes of data offline.

**Characteristics**:
- Latency: Minutes to hours acceptable
- Throughput: High volume processing
- Use Case: ETL pipelines, reporting
- Architecture: Distributed processing

**Implementation**:
```python
def batch_predict(input_file, output_file, batch_size=1000):
    df = pd.read_csv(input_file)
    predictions = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_features = batch[feature_columns].values
        batch_predictions = model.predict(batch_features)
        predictions.extend(batch_predictions)
    
    df['prediction'] = predictions
    df.to_csv(output_file, index=False)
```

### Hybrid Approach

```python
class InferenceService:
    def __init__(self):
        self.model = load_model()
        self.batch_queue = Queue()
        self.batch_processor = Thread(target=self.process_batch)
        self.batch_processor.start()
    
    def predict_online(self, features):
        # Fast path for real-time
        if self.is_simple_request(features):
            return self.model.predict(features)
        else:
            # Queue for batch processing
            future = Future()
            self.batch_queue.put((features, future))
            return future.result()
    
    def process_batch(self):
        while True:
            batch = []
            futures = []
            
            # Collect batch
            while len(batch) < 100:
                try:
                    features, future = self.batch_queue.get(timeout=1)
                    batch.append(features)
                    futures.append(future)
                except:
                    break
            
            if batch:
                predictions = self.model.predict_batch(batch)
                for future, pred in zip(futures, predictions):
                    future.set_result(pred)
```

## Model Formats

### ONNX (Open Neural Network Exchange)

ONNX provides interoperability across frameworks.

**Export to ONNX**:
```python
import torch
import onnx

# PyTorch to ONNX
dummy_input = torch.randn(1, 4)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['features'],
    output_names=['prediction']
)

# scikit-learn to ONNX
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('features', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Inference with ONNX Runtime**:
```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
inputs = {session.get_inputs()[0].name: features}
outputs = session.run(None, inputs)
prediction = outputs[0]
```

### TensorRT

NVIDIA TensorRT optimizes models for GPU inference.

**Conversion**:
```python
import tensorrt as trt

# Build TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

### SavedModel Format

TensorFlow's standard format.

```python
# Save model
model.save('saved_model')

# Load and serve
import tensorflow as tf
loaded_model = tf.saved_model.load('saved_model')
predict_fn = loaded_model.signatures['serving_default']
prediction = predict_fn(tf.constant(features))
```

## API Design Patterns

### Versioning

```python
# URL versioning
@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    # Version 1 implementation
    pass

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    # Version 2 implementation
    pass

# Header versioning
@app.route('/predict', methods=['POST'])
def predict():
    version = request.headers.get('API-Version', 'v1')
    if version == 'v1':
        return predict_v1()
    elif version == 'v2':
        return predict_v2()
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Prediction logic
    pass
```

### Request Validation

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    features: list[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Features must have 4 elements')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        return v
```

## Performance Optimization

### Model Optimization

**Quantization**:
```python
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**Pruning**:
```python
import tensorflow_model_optimization as tfmot

# Prune model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=1000
    )
}
model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

### Caching

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=10000)
def cached_predict(features_hash):
    # Prediction logic
    pass

def predict_with_cache(features):
    features_str = json.dumps(features, sort_keys=True)
    features_hash = hashlib.md5(features_str.encode()).hexdigest()
    return cached_predict(features_hash)
```

### Batching

```python
class BatchPredictor:
    def __init__(self, batch_size=32, timeout=0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = []
        self.lock = threading.Lock()
    
    def predict(self, features):
        with self.lock:
            self.queue.append(features)
            if len(self.queue) >= self.batch_size:
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]
                return self.process_batch(batch)
        
        # Wait for timeout
        time.sleep(self.timeout)
        with self.lock:
            if self.queue:
                batch = self.queue
                self.queue = []
                return self.process_batch(batch)
```

## Security Considerations

### Authentication

```python
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Missing token'}), 401
        
        try:
            token = token.replace('Bearer ', '')
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_auth
def predict():
    # Prediction logic
    pass
```

### Input Validation

```python
def validate_input(features):
    # Check for malicious input
    if len(features) > 1000:
        raise ValueError('Too many features')
    
    # Check for NaN or Inf
    if any(math.isnan(f) or math.isinf(f) for f in features):
        raise ValueError('Invalid feature values')
    
    return features
```

### Rate Limiting by User

```python
@limiter.limit("100 per hour", key_func=lambda: request.headers.get('User-ID'))
def predict():
    # Prediction logic
    pass
```

## Key Takeaways

- REST APIs provide simplicity and wide compatibility for model serving
- gRPC offers superior performance for high-throughput, low-latency applications
- Model serving frameworks (TFServing, Triton, TorchServe) provide optimized inference
- Batch inference handles large volumes efficiently, while online inference serves real-time requests
- Model formats (ONNX, TensorRT) enable framework interoperability and optimization
- API design patterns (versioning, rate limiting, validation) ensure robust serving
- Performance optimization through quantization, pruning, caching, and batching improves efficiency
- Security measures (authentication, input validation) protect models and data
- The choice of serving architecture depends on latency, throughput, and scalability requirements
- Hybrid approaches combine the benefits of batch and online inference
