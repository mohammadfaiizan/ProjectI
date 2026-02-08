# Continuous Integration and Deployment

## Table of Contents

1. [Introduction to CI/CD for ML](#introduction-to-cicd-for-ml)
2. [CI/CD Pipeline Architecture](#cicd-pipeline-architecture)
3. [Automated Testing for ML](#automated-testing-for-ml)
4. [Data Testing](#data-testing)
5. [Model Testing](#model-testing)
6. [Code Testing](#code-testing)
7. [Deployment Pipelines](#deployment-pipelines)
8. [GitOps for ML](#gitops-for-ml)
9. [Infrastructure as Code](#infrastructure-as-code)
10. [Key Takeaways](#key-takeaways)

## Introduction to CI/CD for ML

Continuous Integration and Continuous Deployment (CI/CD) for ML extends traditional software CI/CD practices to handle the unique challenges of ML systems:

- **Data Dependencies**: Models depend on data, not just code
- **Model Artifacts**: Large binary files that need versioning
- **Reproducibility**: Training must be reproducible across environments
- **Testing Complexity**: Testing models requires different approaches than code
- **Long Training Times**: Training can take hours or days

### ML CI/CD Challenges

**Traditional CI/CD**:
- Code changes trigger builds
- Fast feedback loops (minutes)
- Deterministic tests
- Single artifact (application)

**ML CI/CD**:
- Data, code, and model changes trigger builds
- Slower feedback loops (hours/days for training)
- Non-deterministic tests (model performance)
- Multiple artifacts (code, data, models)

### CI/CD Stages for ML

1. **Continuous Integration**: Build, test code and data
2. **Continuous Training**: Train models on new data
3. **Continuous Validation**: Validate model performance
4. **Continuous Deployment**: Deploy validated models

## CI/CD Pipeline Architecture

### Pipeline Stages

```
┌─────────────┐
│   Source    │──┐
│   Control   │  │
│  (Git)      │  │
└─────────────┘  │
                 │
                 ▼
         ┌───────────────┐
         │   Trigger     │
         │  (Webhook)    │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Build &     │
         │   Test Code   │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │  Test Data    │
         │  & Schema     │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Train       │
         │   Model       │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │  Test Model   │
         │  Performance  │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Deploy      │
         │   Model       │
         └───────────────┘
```

### GitHub Actions Example

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily training

jobs:
  test-code:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src/

  test-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Test data quality
        run: |
          python scripts/validate_data.py
          python scripts/test_schema.py

  train-model:
    needs: [test-code, test-data]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Train model
        run: python scripts/train.py
      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: model
          path: models/

  test-model:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v2
        with:
          name: model
      - name: Test model performance
        run: python scripts/test_model.py

  deploy:
    needs: test-model
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: python scripts/deploy.py --env staging
```

## Automated Testing for ML

### Testing Pyramid for ML

```
        ┌─────────────┐
        │   E2E Tests │  (Few, slow, expensive)
        │  (Full Flow)│
        └─────────────┘
              │
        ┌─────┴─────┐
        │ Integration│  (Some, moderate speed)
        │   Tests    │
        └─────┬─────┘
              │
        ┌─────┴─────┐
        │   Unit    │  (Many, fast, cheap)
        │   Tests   │
        └───────────┘
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Data Tests**: Test data quality and schema
4. **Model Tests**: Test model behavior and performance
5. **End-to-End Tests**: Test complete workflows

## Data Testing

### Schema Validation

```python
import pandas as pd
from pandera import DataFrameSchema, Column, Check

# Define schema
schema = DataFrameSchema({
    "user_id": Column(int, checks=Check.greater_than(0)),
    "age": Column(int, checks=Check.in_range(0, 120)),
    "email": Column(str, checks=Check.str_matches(r'^[\w\.-]+@[\w\.-]+\.\w+$')),
    "purchase_amount": Column(float, checks=Check.greater_than(0)),
    "timestamp": Column(pd.Timestamp)
})

# Validate data
def test_data_schema(df):
    try:
        schema.validate(df)
        return True
    except Exception as e:
        print(f"Schema validation failed: {e}")
        return False
```

### Data Quality Tests

```python
import pytest
import pandas as pd

def test_data_completeness(df):
    """Test that required columns are not null"""
    required_columns = ["user_id", "email", "purchase_amount"]
    for col in required_columns:
        assert df[col].notna().all(), f"Column {col} has null values"

def test_data_uniqueness(df):
    """Test that user_id is unique"""
    assert df["user_id"].is_unique, "user_id is not unique"

def test_data_range(df):
    """Test that values are within expected ranges"""
    assert df["age"].between(0, 120).all(), "Age values out of range"
    assert df["purchase_amount"].ge(0).all(), "Negative purchase amounts"

def test_data_consistency(df):
    """Test data consistency rules"""
    # Example: if user has subscription, subscription_date should not be null
    subscribed_users = df[df["has_subscription"] == True]
    assert subscribed_users["subscription_date"].notna().all(), \
        "Subscribed users missing subscription_date"
```

### Data Distribution Tests

```python
from scipy import stats

def test_data_distribution(df, reference_df):
    """Test that data distribution hasn't changed significantly"""
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        ks_stat, p_value = stats.ks_2samp(
            reference_df[col].dropna(),
            df[col].dropna()
        )
        assert p_value > 0.05, \
            f"Distribution changed for {col} (p={p_value})"
```

### Data Freshness Tests

```python
from datetime import datetime, timedelta

def test_data_freshness(df):
    """Test that data is recent"""
    max_age_days = 7
    latest_timestamp = df["timestamp"].max()
    age = datetime.now() - latest_timestamp
    assert age.days <= max_age_days, \
        f"Data is {age.days} days old, max allowed is {max_age_days}"
```

## Model Testing

### Model Performance Tests

```python
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score

def test_model_accuracy(model, X_test, y_test, threshold=0.85):
    """Test that model accuracy meets threshold"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= threshold, \
        f"Model accuracy {accuracy} below threshold {threshold}"

def test_model_precision(model, X_test, y_test, threshold=0.80):
    """Test that model precision meets threshold"""
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    assert precision >= threshold, \
        f"Model precision {precision} below threshold {threshold}"

def test_model_recall(model, X_test, y_test, threshold=0.75):
    """Test that model recall meets threshold"""
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, average='weighted')
    assert recall >= threshold, \
        f"Model recall {recall} below threshold {threshold}"
```

### Model Behavior Tests

```python
def test_model_prediction_shape(model, X_test):
    """Test that predictions have correct shape"""
    predictions = model.predict(X_test)
    assert predictions.shape[0] == X_test.shape[0], \
        "Prediction shape mismatch"

def test_model_prediction_range(model, X_test):
    """Test that predictions are in valid range"""
    predictions = model.predict(X_test)
    assert predictions.min() >= 0 and predictions.max() <= 1, \
        "Predictions out of valid range [0, 1]"

def test_model_consistency(model, X_test):
    """Test that model produces consistent predictions"""
    predictions1 = model.predict(X_test)
    predictions2 = model.predict(X_test)
    assert (predictions1 == predictions2).all(), \
        "Model predictions are not consistent"
```

### Model Fairness Tests

```python
def test_model_fairness(model, X_test, y_test, sensitive_attribute):
    """Test that model doesn't discriminate on sensitive attribute"""
    predictions = model.predict(X_test)
    
    # Group by sensitive attribute
    groups = X_test[sensitive_attribute].unique()
    accuracies = {}
    
    for group in groups:
        mask = X_test[sensitive_attribute] == group
        group_acc = accuracy_score(
            y_test[mask],
            predictions[mask]
        )
        accuracies[group] = group_acc
    
    # Check that accuracy difference is small
    max_acc = max(accuracies.values())
    min_acc = min(accuracies.values())
    assert max_acc - min_acc < 0.05, \
        f"Model accuracy varies significantly across groups: {accuracies}"
```

### Model Robustness Tests

```python
import numpy as np

def test_model_robustness_to_noise(model, X_test, noise_level=0.01):
    """Test that model is robust to small input perturbations"""
    predictions_original = model.predict(X_test)
    
    # Add noise
    X_noisy = X_test + np.random.normal(0, noise_level, X_test.shape)
    predictions_noisy = model.predict(X_noisy)
    
    # Check that predictions don't change drastically
    change_rate = (predictions_original != predictions_noisy).mean()
    assert change_rate < 0.1, \
        f"Model too sensitive to noise: {change_rate*100}% predictions changed"
```

## Code Testing

### Unit Tests

```python
import pytest
from src.feature_engineering import FeatureEngineer

def test_feature_engineer_init():
    """Test FeatureEngineer initialization"""
    fe = FeatureEngineer()
    assert fe is not None

def test_feature_engineer_fit_transform():
    """Test feature engineering fit_transform"""
    fe = FeatureEngineer()
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    X_transformed = fe.fit_transform(X_train)
    assert X_transformed.shape[0] == X_train.shape[0]

def test_feature_engineer_consistency():
    """Test that transform is consistent after fit"""
    fe = FeatureEngineer()
    X_train = pd.DataFrame({"feature1": [1, 2, 3]})
    fe.fit(X_train)
    X_transformed1 = fe.transform(X_train)
    X_transformed2 = fe.transform(X_train)
    assert (X_transformed1 == X_transformed2).all().all()
```

### Integration Tests

```python
def test_training_pipeline():
    """Test complete training pipeline"""
    from src.pipeline import TrainingPipeline
    
    pipeline = TrainingPipeline()
    model = pipeline.train("data/train.csv")
    
    assert model is not None
    assert hasattr(model, 'predict')

def test_prediction_pipeline():
    """Test complete prediction pipeline"""
    from src.pipeline import PredictionPipeline
    
    pipeline = PredictionPipeline()
    predictions = pipeline.predict("data/test.csv")
    
    assert predictions is not None
    assert len(predictions) > 0
```

## Deployment Pipelines

### Deployment Stages

```
┌─────────────┐
│   Build     │──┐
│   Artifacts │  │
└─────────────┘  │
                 │
                 ▼
         ┌───────────────┐
         │   Validate    │
         │   Artifacts   │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Deploy to   │
         │   Staging     │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Run E2E     │
         │   Tests       │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │   Deploy to   │
         │   Production  │
         └───────────────┘
```

### Deployment Script

```python
import mlflow
from mlflow.tracking import MlflowClient
import kubernetes
from kubernetes import client, config

def deploy_model(model_name, version, environment):
    """Deploy model to specified environment"""
    
    # Load model from registry
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Build Docker image
    build_docker_image(model_name, version)
    
    # Deploy to Kubernetes
    if environment == "staging":
        deploy_to_staging(model_name, version)
    elif environment == "production":
        deploy_to_production(model_name, version)
    else:
        raise ValueError(f"Unknown environment: {environment}")

def build_docker_image(model_name, version):
    """Build Docker image with model"""
    import subprocess
    subprocess.run([
        "docker", "build",
        "-t", f"{model_name}:{version}",
        "-f", "Dockerfile",
        "."
    ])

def deploy_to_kubernetes(model_name, version, namespace):
    """Deploy model to Kubernetes"""
    config.load_incluster_config()
    v1 = client.AppsV1Api()
    
    deployment = client.V1Deployment(
        metadata=client.V1ObjectMeta(name=f"{model_name}-deployment"),
        spec=client.V1DeploymentSpec(
            replicas=3,
            selector=client.V1LabelSelector(
                match_labels={"app": model_name}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={"app": model_name}
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name=model_name,
                            image=f"{model_name}:{version}",
                            ports=[client.V1ContainerPort(container_port=8080)]
                        )
                    ]
                )
            )
        )
    )
    
    v1.create_namespaced_deployment(namespace=namespace, body=deployment)
```

## GitOps for ML

GitOps applies DevOps best practices to ML operations using Git as the source of truth.

### GitOps Principles

1. **Declarative**: System state defined in Git
2. **Versioned**: All changes tracked in Git
3. **Automated**: Changes automatically applied
4. **Observable**: System state visible and auditable

### GitOps Workflow

```
┌─────────────┐
│   Developer │
│   Pushes    │
│   Changes   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     Git     │
│  Repository │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   CI/CD     │
│   Pipeline  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Kubernetes │
│  Cluster    │
└─────────────┘
```

### ArgoCD for ML

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ml-model-app
spec:
  project: default
  source:
    repoURL: https://github.com/org/ml-configs
    path: models/classification
    targetRevision: main
  destination:
    server: https://kubernetes.default.svc
    namespace: ml-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Model Configuration in Git

```yaml
# models/classification/config.yaml
apiVersion: ml.argoproj.io/v1alpha1
kind: ModelDeployment
metadata:
  name: classification-model
spec:
  model:
    name: classification_model
    version: "1.2.0"
    registry: mlflow://models/classification_model/Production
  serving:
    replicas: 3
    resources:
      requests:
        memory: "2Gi"
        cpu: "1000m"
      limits:
        memory: "4Gi"
        cpu: "2000m"
  autoscaling:
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

## Infrastructure as Code

Infrastructure as Code (IaC) manages infrastructure through code rather than manual configuration.

### Terraform for ML Infrastructure

```hcl
# infrastructure/main.tf
provider "aws" {
  region = "us-east-1"
}

# S3 bucket for model artifacts
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "ml-model-artifacts"
  
  versioning {
    enabled = true
  }
  
  lifecycle {
    prevent_destroy = true
  }
}

# SageMaker endpoint
resource "aws_sagemaker_endpoint" "model_endpoint" {
  name = "classification-endpoint"
  
  endpoint_config_name = aws_sagemaker_endpoint_configuration.model_config.name
}

resource "aws_sagemaker_endpoint_configuration" "model_config" {
  name = "classification-endpoint-config"
  
  production_variants {
    variant_name           = "primary"
    model_name             = aws_sagemaker_model.model.name
    initial_instance_count = 2
    instance_type          = "ml.m5.xlarge"
  }
}

resource "aws_sagemaker_model" "model" {
  name               = "classification-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn
  
  primary_container {
    image = "${aws_ecr_repository.model_repo.repository_url}:latest"
    model_data_url = "s3://${aws_s3_bucket.model_artifacts.bucket}/models/model.tar.gz"
  }
}
```

### CloudFormation for ML

```yaml
# infrastructure/cloudformation.yaml
Resources:
  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: ml-model-artifacts
      VersioningConfiguration:
        Status: Enabled
  
  SageMakerEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointName: classification-endpoint
      EndpointConfigName: !Ref EndpointConfig
  
  EndpointConfig:
    Type: AWS::SageMaker::EndpointConfig
    Properties:
      EndpointConfigName: classification-endpoint-config
      ProductionVariants:
        - VariantName: primary
          ModelName: !Ref Model
          InitialInstanceCount: 2
          InstanceType: ml.m5.xlarge
```

### Pulumi for ML

```python
# infrastructure/main.py
import pulumi
import pulumi_aws as aws

# S3 bucket
model_bucket = aws.s3.Bucket(
    "model-artifacts",
    versioning=aws.s3.BucketVersioningArgs(enabled=True)
)

# SageMaker endpoint
endpoint_config = aws.sagemaker.EndpointConfiguration(
    "endpoint-config",
    production_variants=[aws.sagemaker.EndpointConfigurationProductionVariantArgs(
        variant_name="primary",
        model_name=model.name,
        initial_instance_count=2,
        instance_type="ml.m5.xlarge"
    )]
)

endpoint = aws.sagemaker.Endpoint(
    "endpoint",
    endpoint_config_name=endpoint_config.name
)
```

## Key Takeaways

- CI/CD for ML extends traditional practices to handle data, models, and longer feedback loops
- Automated testing must cover code, data, and models with appropriate test types
- Data testing ensures data quality, schema compliance, and distribution stability
- Model testing validates performance, behavior, fairness, and robustness
- Deployment pipelines automate model deployment through staging to production
- GitOps provides declarative, versioned, and automated ML operations
- Infrastructure as Code enables reproducible and versioned infrastructure
- Testing pyramid balances test coverage with execution speed and cost
- Continuous training enables models to adapt to new data automatically
- Comprehensive CI/CD pipelines reduce manual errors and improve deployment velocity
