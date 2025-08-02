import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
from datetime import datetime
import tempfile
import shutil

# Note: MLflow operations require the mlflow package
# Install with: pip install mlflow

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.models import ModelSignature, infer_signature
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.entities import ViewType
    import mlflow.sklearn  # For sklearn model comparisons
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

# Model Classes
class MLflowModel(nn.Module):
    """PyTorch model with MLflow integration"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        
        # Model metadata
        self.metadata = {
            "model_type": "MLflowModel",
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "num_classes": num_classes,
            "dropout": dropout,
            "total_parameters": self.count_parameters()
        }
    
    def forward(self, x):
        return self.model(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLflowDataset(Dataset):
    """Dataset with MLflow metadata tracking"""
    
    def __init__(self, size: int, input_dim: int, num_classes: int, 
                 noise_level: float = 0.1, random_seed: int = 42):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.noise_level = noise_level
        self.random_seed = random_seed
        
        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        
        # Create labels with pattern
        weights = torch.randn(input_dim, num_classes)
        logits = self.data @ weights + noise_level * torch.randn(size, num_classes)
        self.labels = torch.argmax(logits, dim=1)
        
        # Dataset metadata
        self.metadata = {
            "dataset_size": size,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "noise_level": noise_level,
            "random_seed": random_seed,
            "class_distribution": self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution"""
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MLflow Integration Manager
class MLflowManager:
    """Comprehensive MLflow integration for PyTorch workflows"""
    
    def __init__(self, experiment_name: str = "pytorch_experiments",
                 tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = None
        self.experiment_id = None
        self.run_id = None
        self.run = None
        
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()
        
        print(f"✓ MLflow setup completed for experiment: {self.experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start MLflow run"""
        if not MLFLOW_AVAILABLE:
            print("MLflow not available - tracking simulated")
            return "simulated_run"
        
        self.run = mlflow.start_run(run_name=run_name, tags=tags)
        self.run_id = self.run.info.run_id
        
        print(f"✓ Started MLflow run: {self.run_id}")
        return self.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if not MLFLOW_AVAILABLE:
            print(f"Parameters logged (simulated): {params}")
            return
        
        # Convert complex objects to strings
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                clean_params[key] = str(value)
            else:
                clean_params[key] = value
        
        mlflow.log_params(clean_params)
        print(f"✓ Logged {len(clean_params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if not MLFLOW_AVAILABLE:
            print(f"Metrics logged (simulated): {metrics}")
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        print(f"✓ Logged {len(metrics)} metrics" + (f" at step {step}" if step else ""))
    
    def log_artifacts(self, artifacts: Dict[str, Any], artifact_path: str = "artifacts"):
        """Log various artifacts"""
        if not MLFLOW_AVAILABLE:
            print(f"Artifacts logged (simulated): {list(artifacts.keys())}")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for name, artifact in artifacts.items():
                artifact_file = temp_path / name
                
                if isinstance(artifact, dict):
                    # JSON artifact
                    with open(artifact_file, 'w') as f:
                        json.dump(artifact, f, indent=2)
                elif isinstance(artifact, str):
                    # Text artifact
                    with open(artifact_file, 'w') as f:
                        f.write(artifact)
                elif hasattr(artifact, 'save'):
                    # Object with save method (like plots)
                    artifact.save(artifact_file)
                else:
                    # Pickle other objects
                    with open(artifact_file, 'wb') as f:
                        pickle.dump(artifact, f)
            
            mlflow.log_artifacts(temp_dir, artifact_path)
        
        print(f"✓ Logged {len(artifacts)} artifacts")
    
    def log_model(self, model: nn.Module, artifact_path: str = "model",
                  input_example: Optional[torch.Tensor] = None,
                  signature: Optional[ModelSignature] = None) -> str:
        """Log PyTorch model"""
        if not MLFLOW_AVAILABLE:
            print("Model logged (simulated)")
            return "simulated_model_uri"
        
        # Infer signature if not provided
        if signature is None and input_example is not None:
            with torch.no_grad():
                model.eval()
                output_example = model(input_example)
                signature = infer_signature(
                    input_example.numpy(), 
                    output_example.numpy()
                )
        
        # Log model
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example.numpy() if input_example is not None else None
        )
        
        print(f"✓ Model logged to: {model_info.model_uri}")
        return model_info.model_uri
    
    def register_model(self, model_uri: str, model_name: str, 
                      version_description: Optional[str] = None) -> str:
        """Register model in MLflow Model Registry"""
        if not MLFLOW_AVAILABLE:
            print(f"Model registered (simulated): {model_name}")
            return "simulated_version"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        # Add version description
        if version_description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=version_description
            )
        
        print(f"✓ Model registered: {model_name} v{model_version.version}")
        return model_version.version
    
    def transition_model_stage(self, model_name: str, version: str, 
                             stage: str, archive_existing: bool = True):
        """Transition model to different stage"""
        if not MLFLOW_AVAILABLE:
            print(f"Model stage transition (simulated): {model_name} v{version} -> {stage}")
            return
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        
        print(f"✓ Model {model_name} v{version} transitioned to {stage}")
    
    def end_run(self):
        """End current MLflow run"""
        if not MLFLOW_AVAILABLE:
            print("Run ended (simulated)")
            return
        
        mlflow.end_run()
        print(f"✓ Ended run: {self.run_id}")

# MLflow Training Manager
class MLflowTrainer:
    """Training with comprehensive MLflow tracking"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, mlflow_manager: MLflowManager):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mlflow_manager = mlflow_manager
        
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.training_history = []
    
    def setup_training(self, learning_rate: float = 0.001, 
                      weight_decay: float = 1e-4, optimizer_name: str = "Adam"):
        """Setup training components"""
        
        if optimizer_name.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Log training setup
        training_params = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": optimizer_name,
            "criterion": "CrossEntropyLoss",
            "scheduler": "ReduceLROnPlateau"
        }
        
        self.mlflow_manager.log_params(training_params)
        
        print("✓ Training setup completed")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Log batch metrics periodically
            if batch_idx % 50 == 0:
                step = self.epoch * len(self.train_loader) + batch_idx
                self.mlflow_manager.log_metrics({
                    "batch_loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }, step=step)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = correct / total
        
        return avg_val_loss, val_accuracy
    
    def train(self, num_epochs: int = 50, log_model_every: int = 10,
              save_best_model: bool = True):
        """Complete training loop with MLflow tracking"""
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss, train_accuracy = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Log epoch metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
            
            self.mlflow_manager.log_metrics(epoch_metrics, step=epoch)
            
            # Track best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_val_loss = val_loss
                
                if save_best_model:
                    self._log_best_model(epoch)
            
            # Log model periodically
            if epoch % log_model_every == 0:
                self._log_checkpoint(epoch)
            
            # Store training history
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Train Acc = {train_accuracy:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val Acc = {val_accuracy:.4f}")
        
        # Log final artifacts
        self._log_training_artifacts()
        
        print(f"✅ Training completed! Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history
        }
    
    def _log_best_model(self, epoch: int):
        """Log best model to MLflow"""
        # Create input example for signature
        sample_input = next(iter(self.val_loader))[0][:1]  # Single sample
        
        # Log model with metadata
        model_uri = self.mlflow_manager.log_model(
            model=self.model,
            artifact_path=f"best_model_epoch_{epoch}",
            input_example=sample_input
        )
        
        # Add model tags
        if MLFLOW_AVAILABLE:
            mlflow.set_tag("best_model_epoch", epoch)
            mlflow.set_tag("best_val_accuracy", self.best_val_accuracy)
    
    def _log_checkpoint(self, epoch: int):
        """Log model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "training_history": self.training_history
        }
        
        artifacts = {f"checkpoint_epoch_{epoch}.json": checkpoint}
        self.mlflow_manager.log_artifacts(artifacts, "checkpoints")
    
    def _log_training_artifacts(self):
        """Log training artifacts"""
        
        # Training history
        history_artifact = {
            "training_history": self.training_history,
            "best_metrics": {
                "best_val_accuracy": self.best_val_accuracy,
                "best_val_loss": self.best_val_loss
            }
        }
        
        # Model summary
        model_summary = {
            "architecture": str(self.model),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        artifacts = {
            "training_history.json": history_artifact,
            "model_summary.json": model_summary
        }
        
        self.mlflow_manager.log_artifacts(artifacts, "training_artifacts")

# Model Registry Manager
class MLflowModelRegistry:
    """Manage models in MLflow Model Registry"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        self.client = None
        
        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
    
    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        if not MLFLOW_AVAILABLE:
            return [{"name": "simulated_model", "version": "1", "stage": "None"}]
        
        registered_models = self.client.list_registered_models()
        
        models_info = []
        for model in registered_models:
            latest_versions = self.client.get_latest_versions(model.name)
            for version in latest_versions:
                models_info.append({
                    "name": model.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "description": version.description
                })
        
        return models_info
    
    def get_model_version_details(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get detailed information about a model version"""
        if not MLFLOW_AVAILABLE:
            return {"name": model_name, "version": version, "status": "simulated"}
        
        model_version = self.client.get_model_version(model_name, version)
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "description": model_version.description,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "source": model_version.source,
            "run_id": model_version.run_id,
            "status": model_version.status,
            "tags": model_version.tags
        }
    
    def compare_model_versions(self, model_name: str, 
                             versions: List[str]) -> Dict[str, Any]:
        """Compare multiple versions of a model"""
        if not MLFLOW_AVAILABLE:
            return {"comparison": "simulated"}
        
        comparison = {}
        
        for version in versions:
            model_version = self.client.get_model_version(model_name, version)
            
            # Get run metrics if available
            run_id = model_version.run_id
            if run_id:
                run = self.client.get_run(run_id)
                metrics = run.data.metrics
                params = run.data.params
            else:
                metrics = {}
                params = {}
            
            comparison[f"v{version}"] = {
                "stage": model_version.current_stage,
                "metrics": metrics,
                "params": params,
                "creation_time": model_version.creation_timestamp
            }
        
        return comparison
    
    def promote_model(self, model_name: str, version: str, 
                     target_stage: str = "Production") -> bool:
        """Promote model to target stage"""
        if not MLFLOW_AVAILABLE:
            print(f"Model promotion (simulated): {model_name} v{version} -> {target_stage}")
            return True
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage,
                archive_existing_versions=True
            )
            
            print(f"✓ Model {model_name} v{version} promoted to {target_stage}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to promote model: {e}")
            return False
    
    def archive_old_versions(self, model_name: str, keep_latest: int = 3):
        """Archive old model versions"""
        if not MLFLOW_AVAILABLE:
            print(f"Archiving old versions (simulated): {model_name}")
            return
        
        versions = self.client.get_latest_versions(model_name)
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.creation_timestamp, reverse=True)
        
        # Archive versions beyond keep_latest
        for version in versions[keep_latest:]:
            if version.current_stage not in ["Production", "Staging"]:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                
                print(f"✓ Archived {model_name} v{version.version}")

# Model Deployment Manager
class MLflowModelDeployment:
    """Deploy models from MLflow Model Registry"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        if MLFLOW_AVAILABLE and tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
    
    def load_model_from_registry(self, model_name: str, 
                               stage: str = "Production") -> Optional[nn.Module]:
        """Load model from registry"""
        if not MLFLOW_AVAILABLE:
            print(f"Model loading (simulated): {model_name} from {stage}")
            return None
        
        model_uri = f"models:/{model_name}/{stage}"
        
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print(f"✓ Loaded model {model_name} from {stage} stage")
            return model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return None
    
    def create_inference_service(self, model_name: str, 
                               stage: str = "Production") -> callable:
        """Create inference service for deployed model"""
        
        model = self.load_model_from_registry(model_name, stage)
        
        if model is None:
            # Fallback model for demo
            model = MLflowModel(784, [128, 64], 10)
            print("Using fallback model for inference service")
        
        model.eval()
        
        def inference_fn(input_data: torch.Tensor) -> Dict[str, Any]:
            """Inference function"""
            with torch.no_grad():
                if len(input_data.shape) == 1:
                    input_data = input_data.unsqueeze(0)
                
                outputs = model(input_data)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
                return {
                    "predictions": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                    "confidence": confidence.tolist()
                }
        
        return inference_fn
    
    def benchmark_model_performance(self, model_name: str, 
                                  test_data: torch.Tensor, 
                                  stage: str = "Production") -> Dict[str, float]:
        """Benchmark model performance"""
        
        inference_fn = self.create_inference_service(model_name, stage)
        
        # Measure inference time
        start_time = time.time()
        results = inference_fn(test_data)
        end_time = time.time()
        
        inference_time = end_time - start_time
        throughput = len(test_data) / inference_time
        
        performance_metrics = {
            "inference_time_seconds": inference_time,
            "throughput_samples_per_second": throughput,
            "batch_size": len(test_data),
            "average_time_per_sample": inference_time / len(test_data)
        }
        
        return performance_metrics

# Experiment Analysis
class MLflowExperimentAnalyzer:
    """Analyze and compare MLflow experiments"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri
        self.client = None
        
        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
    
    def get_experiment_runs(self, experiment_name: str, 
                          max_results: int = 100) -> List[Dict[str, Any]]:
        """Get all runs from an experiment"""
        if not MLFLOW_AVAILABLE:
            return [{"run_id": "simulated", "status": "FINISHED"}]
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return []
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=max_results
        )
        
        runs_info = []
        for run in runs:
            runs_info.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        
        return runs_info
    
    def find_best_runs(self, experiment_name: str, 
                      metric_name: str, ascending: bool = False) -> List[Dict[str, Any]]:
        """Find best runs based on metric"""
        
        runs = self.get_experiment_runs(experiment_name)
        
        # Filter runs with the specified metric
        valid_runs = [run for run in runs if metric_name in run.get("metrics", {})]
        
        # Sort by metric
        sorted_runs = sorted(
            valid_runs,
            key=lambda x: x["metrics"][metric_name],
            reverse=not ascending
        )
        
        return sorted_runs
    
    def compare_hyperparameters(self, experiment_name: str, 
                              top_k: int = 5) -> Dict[str, Any]:
        """Compare hyperparameters of top performing runs"""
        
        best_runs = self.find_best_runs(experiment_name, "val_accuracy", ascending=False)
        top_runs = best_runs[:top_k]
        
        comparison = {
            "runs": [],
            "parameter_analysis": {}
        }
        
        all_params = set()
        for run in top_runs:
            run_params = run.get("params", {})
            all_params.update(run_params.keys())
            
            comparison["runs"].append({
                "run_id": run["run_id"],
                "val_accuracy": run.get("metrics", {}).get("val_accuracy", 0),
                "params": run_params
            })
        
        # Analyze parameter distributions
        for param in all_params:
            values = []
            for run in top_runs:
                if param in run.get("params", {}):
                    values.append(run["params"][param])
            
            comparison["parameter_analysis"][param] = {
                "values": values,
                "unique_values": list(set(values)),
                "most_common": max(set(values), key=values.count) if values else None
            }
        
        return comparison

if __name__ == "__main__":
    print("MLflow Model Lifecycle Management")
    print("=" * 37)
    
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Install with: pip install mlflow")
        print("Showing simulated MLflow integration...")
    
    print("\n1. MLflow Setup")
    print("-" * 17)
    
    # Initialize MLflow manager
    mlflow_manager = MLflowManager(
        experiment_name="pytorch_ecosystem_demo",
        tracking_uri=None  # Use local tracking
    )
    
    print("\n2. Model and Data Preparation")
    print("-" * 33)
    
    # Create model and dataset
    model = MLflowModel(
        input_size=784,
        hidden_sizes=[256, 128, 64],
        num_classes=10,
        dropout=0.3
    )
    
    # Create datasets
    train_dataset = MLflowDataset(size=5000, input_dim=784, num_classes=10)
    val_dataset = MLflowDataset(size=1000, input_dim=784, num_classes=10)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"✓ Model created with {model.count_parameters():,} parameters")
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    
    print("\n3. Training with MLflow Tracking")
    print("-" * 35)
    
    # Start MLflow run
    run_id = mlflow_manager.start_run(
        run_name="pytorch_demo_training",
        tags={
            "model_type": "MLP",
            "framework": "PyTorch",
            "purpose": "demonstration"
        }
    )
    
    # Log model and dataset metadata
    model_params = model.metadata
    dataset_params = train_dataset.metadata
    
    all_params = {**model_params, **dataset_params}
    mlflow_manager.log_params(all_params)
    
    # Create trainer
    trainer = MLflowTrainer(model, train_loader, val_loader, mlflow_manager)
    
    # Setup training
    trainer.setup_training(
        learning_rate=0.001,
        weight_decay=1e-4,
        optimizer_name="Adam"
    )
    
    # Train model
    training_results = trainer.train(
        num_epochs=15,  # Reduced for demo
        log_model_every=5,
        save_best_model=True
    )
    
    print(f"Training completed with best accuracy: {training_results['best_val_accuracy']:.4f}")
    
    print("\n4. Model Registration")
    print("-" * 22)
    
    # Log final model
    sample_input = next(iter(val_loader))[0][:1]
    model_uri = mlflow_manager.log_model(
        model=model,
        artifact_path="final_model",
        input_example=sample_input
    )
    
    # Register model
    model_name = "pytorch_demo_classifier"
    version = mlflow_manager.register_model(
        model_uri=model_uri,
        model_name=model_name,
        version_description="Demo PyTorch classifier with MLflow integration"
    )
    
    print(f"✓ Model registered: {model_name} v{version}")
    
    # End MLflow run
    mlflow_manager.end_run()
    
    print("\n5. Model Registry Management")
    print("-" * 31)
    
    # Initialize model registry
    registry = MLflowModelRegistry()
    
    # List registered models
    registered_models = registry.list_registered_models()
    print(f"✓ Found {len(registered_models)} registered model versions")
    
    for model_info in registered_models[-3:]:  # Show last 3
        print(f"  {model_info['name']} v{model_info['version']} - {model_info['stage']}")
    
    # Get model version details
    if MLFLOW_AVAILABLE and version != "simulated_version":
        model_details = registry.get_model_version_details(model_name, version)
        print(f"✓ Model details retrieved for {model_name} v{version}")
        print(f"  Stage: {model_details['stage']}")
        print(f"  Status: {model_details['status']}")
    
    print("\n6. Model Deployment")
    print("-" * 20)
    
    # Initialize deployment manager
    deployment = MLflowModelDeployment()
    
    # Create inference service
    inference_service = deployment.create_inference_service(model_name, "None")
    
    # Test inference
    test_input = torch.randn(5, 784)  # 5 test samples
    predictions = inference_service(test_input)
    
    print(f"✓ Inference service created")
    print(f"  Test predictions: {predictions['predictions']}")
    print(f"  Confidence scores: {[f'{c:.3f}' for c in predictions['confidence']]}")
    
    # Benchmark performance
    performance = deployment.benchmark_model_performance(
        model_name, test_input, "None"
    )
    
    print(f"✓ Performance benchmark:")
    print(f"  Throughput: {performance['throughput_samples_per_second']:.1f} samples/sec")
    print(f"  Avg time per sample: {performance['average_time_per_sample']*1000:.2f} ms")
    
    print("\n7. Experiment Analysis")
    print("-" * 23)
    
    # Initialize experiment analyzer
    analyzer = MLflowExperimentAnalyzer()
    
    # Get experiment runs
    runs = analyzer.get_experiment_runs("pytorch_ecosystem_demo", max_results=10)
    print(f"✓ Found {len(runs)} runs in experiment")
    
    # Find best runs
    best_runs = analyzer.find_best_runs(
        "pytorch_ecosystem_demo", 
        "val_accuracy", 
        ascending=False
    )
    
    if best_runs:
        print(f"✓ Best run accuracy: {best_runs[0].get('metrics', {}).get('val_accuracy', 'N/A')}")
    
    print("\n8. MLflow Best Practices")
    print("-" * 27)
    
    best_practices = [
        "Use descriptive experiment and run names",
        "Log all relevant hyperparameters and metrics",
        "Version your datasets alongside models",
        "Use model signatures for input/output validation",
        "Implement model versioning and staging workflows",
        "Track model lineage and dependencies",
        "Use tags for organizing and filtering runs",
        "Log artifacts for reproducibility",
        "Implement model performance monitoring",
        "Use model registry for production deployments",
        "Automate model promotion workflows",
        "Track model drift and performance degradation"
    ]
    
    print("MLflow Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. MLflow Integration Patterns")
    print("-" * 32)
    
    integration_patterns = {
        "CI/CD Integration": "Automate model training and deployment pipelines",
        "A/B Testing": "Compare model versions in production",
        "Model Monitoring": "Track model performance over time",
        "Experiment Tracking": "Comprehensive experiment management",
        "Model Governance": "Approval workflows for model promotion",
        "Multi-Environment": "Separate dev/staging/prod model registries",
        "Team Collaboration": "Shared model registry and experiments",
        "Auto-scaling": "Dynamic model serving based on demand"
    }
    
    print("MLflow Integration Patterns:")
    for pattern, description in integration_patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n10. Advanced MLflow Features")
    print("-" * 31)
    
    advanced_features = [
        "Model serving with MLflow Models",
        "Multi-model serving and routing",
        "Custom model flavors for different frameworks",
        "Model evaluation and validation pipelines",
        "Automated hyperparameter tuning integration",
        "Model lineage tracking across pipelines",
        "Custom metrics and logging plugins",
        "Integration with cloud providers (AWS, Azure, GCP)",
        "Kubernetes deployment with MLflow",
        "Model registry webhooks and notifications"
    ]
    
    print("Advanced MLflow Features:")
    for feature in advanced_features:
        print(f"  - {feature}")
    
    print("\nMLflow model lifecycle demonstration completed!")
    print("Key components covered:")
    print("  - Complete experiment tracking and logging")
    print("  - Model registry and versioning")
    print("  - Model deployment and serving")
    print("  - Performance benchmarking")
    print("  - Experiment analysis and comparison")
    print("  - Production workflow integration")
    
    print("\nMLflow enables:")
    print("  - End-to-end ML lifecycle management")
    print("  - Reproducible experiments and deployments")
    print("  - Collaborative model development")
    print("  - Production model governance")
    print("  - Automated ML workflows and pipelines")