# Model Versioning and Experiment Tracking

## Table of Contents

1. [Introduction to Experiment Tracking](#introduction-to-experiment-tracking)
2. [MLflow Overview](#mlflow-overview)
3. [Weights & Biases](#weights--biases)
4. [Experiment Management](#experiment-management)
5. [Model Registry](#model-registry)
6. [Artifact Storage](#artifact-storage)
7. [Reproducibility](#reproducibility)
8. [Hyperparameter Logging](#hyperparameter-logging)
9. [Comparison of Tools](#comparison-of-tools)
10. [Key Takeaways](#key-takeaways)

## Introduction to Experiment Tracking

Experiment tracking is the practice of logging and organizing ML experiments to enable reproducibility, comparison, and collaboration. Effective experiment tracking captures:

- **Parameters**: Hyperparameters, model architecture, data configurations
- **Metrics**: Training and validation metrics, custom evaluation metrics
- **Artifacts**: Model files, visualizations, datasets, logs
- **Code**: Git commit hashes, code versions, environment specifications
- **Metadata**: Experiment descriptions, tags, team members, status

### Benefits

- **Reproducibility**: Recreate experiments with exact configurations
- **Comparison**: Compare different experiments and identify best models
- **Collaboration**: Share experiments and insights with team members
- **Debugging**: Track down issues by examining experiment history
- **Compliance**: Maintain audit trails for regulatory requirements

### Core Concepts

**Run**: A single execution of training code
**Experiment**: A collection of related runs
**Artifact**: Files produced during runs (models, plots, data)
**Metric**: Numeric values tracked over time (loss, accuracy)
**Parameter**: Input configuration values (learning rate, batch size)

## MLflow Overview

MLflow is an open-source platform for managing the ML lifecycle, including experiment tracking, model registry, and deployment.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MLflow Components                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Tracking   │  │    Model     │  │   Projects   │ │
│  │    Server    │  │   Registry   │  │   (Code)     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                  │                  │         │
│         └──────────────────┴──────────────────┘         │
│                          │                              │
│                  ┌───────▼────────┐                     │
│                  │  Artifact Store │                     │
│                  │  (S3, GCS, etc)│                     │
│                  └────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Basic Usage

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create or set experiment
mlflow.set_experiment("classification_experiment")

# Start run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("train_size", len(X_train))
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.png")
```

### Advanced Logging

```python
# Log multiple metrics at once
mlflow.log_metrics({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "train_acc": train_acc,
    "val_acc": val_acc
})

# Log metrics over time (for each epoch)
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    mlflow.log_metric("train_loss", train_loss, step=epoch)

# Log nested parameters
mlflow.log_params({
    "model": {
        "type": "neural_network",
        "layers": 3,
        "hidden_units": 128
    },
    "optimizer": {
        "type": "adam",
        "lr": 0.001,
        "beta1": 0.9
    }
})

# Log dataframes
mlflow.log_table("predictions", predictions_df)

# Log images
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(history['loss'])
mlflow.log_figure(fig, "training_curve.png")

# Log text
mlflow.log_text("Model trained on 100k samples", "notes.txt")
```

### MLflow Projects

MLflow Projects package code for reproducible runs:

```yaml
# MLproject file
name: classification_project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "data/train.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
    command: "python train.py {data_path} {n_estimators} {max_depth}"
```

```python
# Run MLflow project
import mlflow.projects

mlflow.projects.run(
    uri=".",
    entry_point="main",
    parameters={
        "data_path": "data/train.csv",
        "n_estimators": 200,
        "max_depth": 15
    },
    experiment_id=experiment_id
)
```

## Weights & Biases

Weights & Biases (W&B) is a popular experiment tracking and visualization platform with strong support for deep learning.

### Setup

```python
import wandb

# Initialize
wandb.init(
    project="classification_project",
    name="experiment_001",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 100
    }
)
```

### Logging

```python
# Log metrics
wandb.log({"loss": loss, "accuracy": accuracy})

# Log metrics with step
for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    wandb.log({"train_loss": train_loss}, step=epoch)

# Log images
wandb.log({"predictions": wandb.Image(prediction_image)})
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=y_true,
    preds=y_pred,
    class_names=class_names
)})

# Log tables
wandb.log({"predictions_table": wandb.Table(
    columns=["id", "prediction", "true_label"],
    data=zip(ids, predictions, true_labels)
)})

# Log histograms
wandb.log({"weights": wandb.Histogram(model_weights)})

# Log audio
wandb.log({"audio": wandb.Audio(audio_data, sample_rate=16000)})

# Log video
wandb.log({"video": wandb.Video(video_array, fps=30)})

# Log model
wandb.log({"model": wandb.Model(model)})
```

### Hyperparameter Sweeps

```yaml
# sweep_config.yaml
program: train.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
    distribution: log_uniform
  batch_size:
    values: [16, 32, 64, 128]
  optimizer:
    values: ["adam", "sgd", "rmsprop"]
```

```python
# Run sweep
sweep_id = wandb.sweep(sweep_config, project="classification_project")
wandb.agent(sweep_id, function=train, count=50)
```

### Artifacts

```python
# Create artifact
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model.pkl")
artifact.add_dir("checkpoints/")
wandb.log_artifact(artifact)

# Use artifact
run = wandb.init()
artifact = run.use_artifact("model:latest")
artifact_dir = artifact.download()
model = load_model(os.path.join(artifact_dir, "model.pkl"))
```

## Experiment Management

### Organizing Experiments

**By Project**: Group related experiments
**By Tags**: Label experiments with tags (e.g., "baseline", "experimental")
**By Team**: Assign experiments to team members
**By Status**: Mark experiments as "running", "completed", "failed"

### Experiment Comparison

```python
import mlflow

# Query runs
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

# Compare runs
comparison_df = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# Display comparison
print(comparison_df[["run_id", "metrics.accuracy", "params.learning_rate"]])
```

### Experiment Templates

Create templates for common experiment types:

```python
class ExperimentTemplate:
    def __init__(self, name, base_config):
        self.name = name
        self.base_config = base_config
    
    def create_run(self, overrides=None):
        config = self.base_config.copy()
        if overrides:
            config.update(overrides)
        
        with mlflow.start_run(run_name=self.name):
            mlflow.log_params(config)
            # Run training with config
            return train_model(config)

# Usage
baseline_template = ExperimentTemplate(
    name="baseline",
    base_config={
        "model": "random_forest",
        "n_estimators": 100,
        "max_depth": 10
    }
)

run = baseline_template.create_run(overrides={"n_estimators": 200})
```

## Model Registry

A model registry provides centralized model storage, versioning, and deployment management.

### MLflow Model Registry

```python
# Register model
model_name = "classification_model"
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name=model_name
)

# Get model versions
from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.get_latest_versions(model_name)

# Transition model stage
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Production"
)

# Load model from registry
model = mlflow.sklearn.load_model(
    f"models:/{model_name}/Production"
)
```

### Model Stages

- **None**: Initial stage after registration
- **Staging**: Testing in staging environment
- **Production**: Deployed to production
- **Archived**: Deprecated models

### Model Versioning

```python
# Create new version
run_id = mlflow.active_run().info.run_id
model_version = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

# Add model version description
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description="Improved accuracy with feature engineering"
)

# Add tags
client.set_model_version_tag(
    name=model_name,
    version=model_version.version,
    key="team",
    value="ml_team"
)
```

### Model Lineage

Track model lineage to understand dependencies:

```python
# Log model dependencies
mlflow.log_param("training_data_version", "v1.2")
mlflow.log_param("feature_store_version", "v2.0")
mlflow.log_param("code_commit", git_commit_hash)

# Query by lineage
runs = mlflow.search_runs(
    filter_string="params.training_data_version = 'v1.2'"
)
```

## Artifact Storage

Artifacts are files produced during experiments that need to be stored and versioned.

### Storage Backends

**Local Filesystem**:
```python
mlflow.set_tracking_uri("file:///path/to/mlruns")
```

**S3**:
```python
mlflow.set_tracking_uri("s3://my-bucket/mlflow")
```

**Azure Blob Storage**:
```python
mlflow.set_tracking_uri("wasbs://container@account.blob.core.windows.net/mlflow")
```

**Google Cloud Storage**:
```python
mlflow.set_tracking_uri("gs://my-bucket/mlflow")
```

### Artifact Management

```python
# Log single artifact
mlflow.log_artifact("model.pkl")

# Log directory
mlflow.log_artifacts("outputs/", artifact_path="results")

# Log with custom path
mlflow.log_artifact("plot.png", artifact_path="visualizations")

# Download artifacts
client = MlflowClient()
client.download_artifacts(run_id, "model", "local_path/")
```

### Large Artifact Handling

For large artifacts, use external storage:

```python
# Upload to S3 first
import boto3
s3 = boto3.client('s3')
s3.upload_file("large_model.pkl", "my-bucket", "models/model.pkl")

# Log reference
mlflow.log_param("model_s3_path", "s3://my-bucket/models/model.pkl")
```

## Reproducibility

Reproducibility ensures experiments can be recreated with identical results.

### Environment Capture

```python
# Log Python version
import sys
mlflow.log_param("python_version", sys.version)

# Log package versions
import pkg_resources
installed_packages = {d.project_name: d.version 
                     for d in pkg_resources.working_set}
mlflow.log_params(installed_packages)

# Log conda environment
mlflow.log_artifact("conda.yaml")

# Log Docker image
mlflow.log_param("docker_image", "ml-training:v1.0")
```

### Code Versioning

```python
# Log git commit
import subprocess
git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode("utf-8").strip()
mlflow.log_param("git_commit", git_commit)

# Log git branch
git_branch = subprocess.check_output(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"]
).decode("utf-8").strip()
mlflow.log_param("git_branch", git_branch)

# Log code as artifact
mlflow.log_artifact("train.py")
mlflow.log_artifacts("src/", artifact_path="code")
```

### Random Seed Management

```python
import random
import numpy as np
import torch

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Log seed
mlflow.log_param("random_seed", seed)
```

### Reproducible Data Loading

```python
# Log data version
mlflow.log_param("data_version", "v1.2")
mlflow.log_param("data_hash", compute_data_hash("data/train.csv"))

# Log data splits
mlflow.log_param("train_size", len(X_train))
mlflow.log_param("val_size", len(X_val))
mlflow.log_param("test_size", len(X_test))
mlflow.log_param("train_test_split_seed", 42)
```

## Hyperparameter Logging

Comprehensive hyperparameter logging enables analysis and optimization.

### Structured Logging

```python
# Log all hyperparameters
hyperparameters = {
    "model": {
        "type": "neural_network",
        "layers": [128, 64, 32],
        "activation": "relu",
        "dropout": 0.2
    },
    "training": {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy"
    },
    "data": {
        "augmentation": True,
        "normalization": "standard",
        "train_val_split": 0.2
    }
}

# Log nested parameters
for category, params in hyperparameters.items():
    for key, value in params.items():
        mlflow.log_param(f"{category}.{key}", value)
```

### Hyperparameter Sweeps

```python
# Grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

for params in ParameterGrid(param_grid):
    with mlflow.start_run():
        mlflow.log_params(params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", score)
```

### Hyperparameter Analysis

```python
# Query runs with different hyperparameters
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"]
)

# Analyze hyperparameter impact
import pandas as pd
df = runs[["params.learning_rate", "params.batch_size", "metrics.accuracy"]]
correlation = df.corr()
print(correlation)
```

## Comparison of Tools

### MLflow vs W&B

| Feature | MLflow | Weights & Biases |
|---------|--------|------------------|
| Open Source | Yes | Partially (core is open) |
| Self-Hosted | Yes | Limited |
| Model Registry | Built-in | Limited |
| Visualization | Basic | Advanced |
| Hyperparameter Sweeps | Limited | Advanced |
| Collaboration | Basic | Advanced |
| Pricing | Free | Freemium |
| Deep Learning Support | Good | Excellent |

### Other Tools

**Neptune**: Advanced experiment tracking with strong visualization
**Comet**: Experiment tracking with model monitoring
**TensorBoard**: Visualization tool for TensorFlow/PyTorch
**Sacred**: Lightweight experiment tracking framework

### Choosing a Tool

Consider:
- **Team Size**: Larger teams benefit from collaboration features
- **Budget**: Open-source vs paid solutions
- **Infrastructure**: Self-hosted vs managed
- **Framework Support**: Deep learning vs traditional ML
- **Integration**: Existing tooling and workflows

## Key Takeaways

- Experiment tracking is essential for reproducibility, comparison, and collaboration in ML
- MLflow provides comprehensive ML lifecycle management with model registry
- Weights & Biases offers advanced visualization and hyperparameter optimization
- Model registries enable centralized model versioning and deployment management
- Artifact storage must scale to handle large model files and datasets
- Reproducibility requires capturing environment, code, data, and random seeds
- Hyperparameter logging enables analysis and optimization of model configurations
- Experiment organization through projects, tags, and metadata improves discoverability
- Comparison tools help identify best-performing models and configurations
- The choice of experiment tracking tool depends on team needs, budget, and infrastructure
