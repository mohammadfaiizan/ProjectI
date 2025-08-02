import torch
import torch.nn as nn
import json
import tempfile
import os
from typing import Dict, List, Tuple, Optional, Any

# Note: MLflow operations require the mlflow package
# Install with: pip install mlflow

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Install with: pip install mlflow")

# Sample Models for MLflow Registry
class MLflowCompatibleCNN(nn.Module):
    """CNN model compatible with MLflow tracking"""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
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
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# MLflow Model Registry Manager
class MLflowModelRegistry:
    """Manage models with MLflow Model Registry"""
    
    def __init__(self, tracking_uri: str = None, registry_uri: str = None):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set registry URI if different from tracking
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.client = MlflowClient()
        print(f"✓ MLflow client initialized")
        print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"  Registry URI: {mlflow.get_registry_uri()}")
    
    def register_model(self, model: nn.Module,
                      model_name: str,
                      experiment_name: str = "model_registry_demo",
                      description: str = "",
                      tags: Dict[str, str] = None,
                      hyperparameters: Dict[str, Any] = None,
                      metrics: Dict[str, float] = None,
                      artifacts: Dict[str, str] = None) -> str:
        """Register model in MLflow registry"""
        
        if tags is None:
            tags = {}
        if hyperparameters is None:
            hyperparameters = {}
        if metrics is None:
            metrics = {}
        if artifacts is None:
            artifacts = {}
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            # Log hyperparameters
            if hasattr(model, 'num_classes'):
                hyperparameters['num_classes'] = model.num_classes
            if hasattr(model, 'dropout_rate'):
                hyperparameters['dropout_rate'] = model.dropout_rate
            
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model info
            model_info = {
                'model_class': model.__class__.__name__,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            mlflow.log_params(model_info)
            
            # Log artifacts
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path, artifact_name)
            
            # Log model
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name
            ).model_uri
            
            # Add tags to run
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            
            run_id = run.info.run_id
        
        print(f"✓ Model registered: {model_name}")
        print(f"  Run ID: {run_id}")
        print(f"  Model URI: {model_uri}")
        
        return run_id
    
    def load_model(self, model_name: str, 
                  version: str = "latest",
                  stage: str = None) -> nn.Module:
        """Load model from MLflow registry"""
        
        if stage:
            # Load by stage
            model_uri = f"models:/{model_name}/{stage}"
        elif version == "latest":
            # Load latest version
            model_uri = f"models:/{model_name}/latest"
        else:
            # Load specific version
            model_uri = f"models:/{model_name}/{version}"
        
        try:
            model = mlflow.pytorch.load_model(model_uri)
            print(f"✓ Model loaded: {model_name} ({version if not stage else stage})")
            return model
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        
        try:
            models = self.client.search_registered_models()
            
            model_list = []
            for model in models:
                model_info = {
                    'name': model.name,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'description': model.description,
                    'latest_versions': []
                }
                
                # Get latest versions
                for version in model.latest_versions:
                    version_info = {
                        'version': version.version,
                        'stage': version.current_stage,
                        'run_id': version.run_id,
                        'status': version.status
                    }
                    model_info['latest_versions'].append(version_info)
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            print(f"✗ Failed to list models: {e}")
            return []
    
    def get_model_version_details(self, model_name: str, 
                                 version: str) -> Dict[str, Any]:
        """Get detailed information about a model version"""
        
        try:
            model_version = self.client.get_model_version(model_name, version)
            
            # Get run details
            run = self.client.get_run(model_version.run_id)
            
            details = {
                'model_name': model_name,
                'version': version,
                'stage': model_version.current_stage,
                'status': model_version.status,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'description': model_version.description,
                'run_id': model_version.run_id,
                'source': model_version.source,
                'run_details': {
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'params': run.data.params,
                    'metrics': run.data.metrics,
                    'tags': run.data.tags
                }
            }
            
            return details
            
        except Exception as e:
            print(f"✗ Failed to get model details: {e}")
            return {}
    
    def transition_model_stage(self, model_name: str, 
                              version: str,
                              stage: str,
                              archive_existing: bool = True) -> bool:
        """Transition model version to a new stage"""
        
        valid_stages = ["None", "Staging", "Production", "Archived"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage. Must be one of: {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            
            print(f"✓ Model {model_name} v{version} transitioned to {stage}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to transition model stage: {e}")
            return False
    
    def delete_model_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        
        try:
            self.client.delete_model_version(model_name, version)
            print(f"✓ Model version deleted: {model_name} v{version}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to delete model version: {e}")
            return False
    
    def update_model_description(self, model_name: str, 
                               description: str,
                               version: str = None) -> bool:
        """Update model or model version description"""
        
        try:
            if version:
                # Update model version description
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )
                print(f"✓ Updated description for {model_name} v{version}")
            else:
                # Update model description
                self.client.update_registered_model(
                    name=model_name,
                    description=description
                )
                print(f"✓ Updated description for {model_name}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to update description: {e}")
            return False
    
    def search_models(self, filter_string: str = None,
                     max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for models with optional filters"""
        
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results
            )
            
            search_results = []
            for model in models:
                search_results.append({
                    'name': model.name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'tags': model.tags
                })
            
            return search_results
            
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return []

# Model Deployment Manager
class MLflowDeploymentManager:
    """Manage model deployments using MLflow"""
    
    def __init__(self, registry: MLflowModelRegistry):
        self.registry = registry
        self.deployments = {}
    
    def deploy_model(self, model_name: str,
                    deployment_name: str,
                    stage: str = "Production",
                    deployment_config: Dict[str, Any] = None) -> str:
        """Deploy model to a target environment"""
        
        if deployment_config is None:
            deployment_config = {}
        
        try:
            # Load model from registry
            model = self.registry.load_model(model_name, stage=stage)
            
            # Create deployment info
            deployment_info = {
                'model_name': model_name,
                'deployment_name': deployment_name,
                'stage': stage,
                'model': model,
                'config': deployment_config,
                'deployed_at': mlflow.utils.time.get_current_time_millis()
            }
            
            self.deployments[deployment_name] = deployment_info
            
            print(f"✓ Model deployed: {deployment_name}")
            print(f"  Model: {model_name} ({stage})")
            
            return deployment_name
            
        except Exception as e:
            print(f"✗ Deployment failed: {e}")
            raise
    
    def update_deployment(self, deployment_name: str,
                         new_model_version: str = None,
                         new_stage: str = None) -> bool:
        """Update existing deployment"""
        
        if deployment_name not in self.deployments:
            print(f"✗ Deployment not found: {deployment_name}")
            return False
        
        try:
            deployment = self.deployments[deployment_name]
            model_name = deployment['model_name']
            
            # Determine what to load
            if new_stage:
                updated_model = self.registry.load_model(model_name, stage=new_stage)
                deployment['stage'] = new_stage
            elif new_model_version:
                updated_model = self.registry.load_model(model_name, version=new_model_version)
                deployment['version'] = new_model_version
            else:
                print("✗ No update specified")
                return False
            
            # Update deployment
            deployment['model'] = updated_model
            deployment['updated_at'] = mlflow.utils.time.get_current_time_millis()
            
            print(f"✓ Deployment updated: {deployment_name}")
            return True
            
        except Exception as e:
            print(f"✗ Update failed: {e}")
            return False
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments"""
        
        deployment_list = []
        for name, info in self.deployments.items():
            deployment_info = {
                'deployment_name': name,
                'model_name': info['model_name'],
                'stage': info.get('stage', 'unknown'),
                'deployed_at': info.get('deployed_at', 0),
                'status': 'active'
            }
            deployment_list.append(deployment_info)
        
        return deployment_list

# Model Lifecycle Manager
class ModelLifecycleManager:
    """Manage complete model lifecycle with MLflow"""
    
    def __init__(self, registry: MLflowModelRegistry):
        self.registry = registry
        self.lifecycle_rules = {}
    
    def add_lifecycle_rule(self, rule_name: str, 
                          conditions: Dict[str, Any],
                          actions: List[str]):
        """Add lifecycle management rule"""
        
        self.lifecycle_rules[rule_name] = {
            'conditions': conditions,
            'actions': actions,
            'created_at': mlflow.utils.time.get_current_time_millis()
        }
        
        print(f"✓ Lifecycle rule added: {rule_name}")
    
    def check_lifecycle_rules(self, model_name: str, version: str) -> List[str]:
        """Check if model version meets any lifecycle rules"""
        
        triggered_actions = []
        
        try:
            # Get model details
            model_details = self.registry.get_model_version_details(model_name, version)
            
            for rule_name, rule in self.lifecycle_rules.items():
                conditions = rule['conditions']
                actions = rule['actions']
                
                # Check conditions
                rule_triggered = True
                
                # Check age condition
                if 'max_age_days' in conditions:
                    creation_time = model_details.get('creation_timestamp', 0)
                    current_time = mlflow.utils.time.get_current_time_millis()
                    age_days = (current_time - creation_time) / (1000 * 60 * 60 * 24)
                    
                    if age_days < conditions['max_age_days']:
                        rule_triggered = False
                
                # Check stage condition
                if 'stage' in conditions:
                    if model_details.get('stage') != conditions['stage']:
                        rule_triggered = False
                
                # Check metric conditions
                if 'min_metric_value' in conditions:
                    metric_name = conditions['min_metric_value']['metric']
                    min_value = conditions['min_metric_value']['value']
                    
                    model_metrics = model_details.get('run_details', {}).get('metrics', {})
                    if metric_name not in model_metrics or float(model_metrics[metric_name]) < min_value:
                        rule_triggered = False
                
                if rule_triggered:
                    print(f"🔄 Lifecycle rule triggered: {rule_name}")
                    triggered_actions.extend(actions)
            
            return triggered_actions
            
        except Exception as e:
            print(f"✗ Error checking lifecycle rules: {e}")
            return []
    
    def execute_lifecycle_actions(self, model_name: str, 
                                 version: str,
                                 actions: List[str]) -> Dict[str, bool]:
        """Execute lifecycle actions on model version"""
        
        results = {}
        
        for action in actions:
            try:
                if action == "archive":
                    success = self.registry.transition_model_stage(
                        model_name, version, "Archived"
                    )
                    results[action] = success
                
                elif action == "delete":
                    success = self.registry.delete_model_version(model_name, version)
                    results[action] = success
                
                elif action.startswith("transition_to_"):
                    stage = action.replace("transition_to_", "")
                    success = self.registry.transition_model_stage(
                        model_name, version, stage
                    )
                    results[action] = success
                
                else:
                    print(f"⚠️  Unknown action: {action}")
                    results[action] = False
                    
            except Exception as e:
                print(f"✗ Action failed: {action} - {e}")
                results[action] = False
        
        return results

if __name__ == "__main__":
    print("MLflow Model Registry")
    print("=" * 25)
    
    if not MLFLOW_AVAILABLE:
        print("MLflow not available. Please install with: pip install mlflow")
        exit(1)
    
    # Initialize MLflow registry
    registry = MLflowModelRegistry()
    
    print("\n1. Model Registration")
    print("-" * 24)
    
    # Create sample models
    model_v1 = MLflowCompatibleCNN(num_classes=10, dropout_rate=0.3)
    model_v2 = MLflowCompatibleCNN(num_classes=10, dropout_rate=0.2)
    
    # Register models
    run_id_v1 = registry.register_model(
        model=model_v1,
        model_name="image_classifier",
        description="Baseline CNN for image classification",
        tags={"version": "1.0", "architecture": "cnn", "framework": "pytorch"},
        hyperparameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 100},
        metrics={"accuracy": 0.85, "f1_score": 0.83, "loss": 0.42}
    )
    
    run_id_v2 = registry.register_model(
        model=model_v2,
        model_name="image_classifier",
        description="Improved CNN with reduced dropout",
        tags={"version": "2.0", "architecture": "cnn", "framework": "pytorch"},
        hyperparameters={"learning_rate": 0.001, "batch_size": 32, "epochs": 120},
        metrics={"accuracy": 0.88, "f1_score": 0.86, "loss": 0.38}
    )
    
    print("\n2. Model Listing and Search")
    print("-" * 30)
    
    # List all models
    models = registry.list_models()
    
    print("Registered Models:")
    for model in models:
        print(f"  {model['name']}:")
        print(f"    Description: {model['description']}")
        print(f"    Versions:")
        for version in model['latest_versions']:
            print(f"      v{version['version']} ({version['stage']}) - {version['status']}")
    
    # Search models
    search_results = registry.search_models(filter_string="name = 'image_classifier'")
    print(f"\nSearch results: {len(search_results)} models found")
    
    print("\n3. Model Version Details")
    print("-" * 29)
    
    # Get details for version 2
    details = registry.get_model_version_details("image_classifier", "2")
    
    if details:
        print("Model Version 2 Details:")
        print(f"  Stage: {details['stage']}")
        print(f"  Status: {details['status']}")
        print(f"  Run ID: {details['run_id']}")
        print(f"  Metrics: {details['run_details']['metrics']}")
        print(f"  Parameters: {details['run_details']['params']}")
    
    print("\n4. Stage Transitions")
    print("-" * 23)
    
    # Transition model to staging
    success = registry.transition_model_stage("image_classifier", "2", "Staging")
    
    if success:
        # Transition to production
        success = registry.transition_model_stage("image_classifier", "2", "Production")
    
    print("\n5. Model Loading")
    print("-" * 18)
    
    # Load models from registry
    try:
        production_model = registry.load_model("image_classifier", stage="Production")
        print("✓ Production model loaded successfully")
        
        # Test inference
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = production_model(test_input)
            print(f"  Output shape: {output.shape}")
    
    except Exception as e:
        print(f"✗ Failed to load production model: {e}")
    
    print("\n6. Deployment Management")
    print("-" * 26)
    
    # Setup deployment manager
    deployment_manager = MLflowDeploymentManager(registry)
    
    # Deploy model
    deployment_name = deployment_manager.deploy_model(
        model_name="image_classifier",
        deployment_name="prod_classifier",
        stage="Production",
        deployment_config={"batch_size": 32, "workers": 4}
    )
    
    # List deployments
    deployments = deployment_manager.list_deployments()
    print("\nActive Deployments:")
    for deployment in deployments:
        print(f"  {deployment['deployment_name']}: {deployment['model_name']} ({deployment['stage']})")
    
    print("\n7. Lifecycle Management")
    print("-" * 25)
    
    # Setup lifecycle manager
    lifecycle_manager = ModelLifecycleManager(registry)
    
    # Add lifecycle rules
    lifecycle_manager.add_lifecycle_rule(
        rule_name="archive_old_staging",
        conditions={
            "max_age_days": 30,
            "stage": "Staging"
        },
        actions=["archive"]
    )
    
    lifecycle_manager.add_lifecycle_rule(
        rule_name="promote_high_accuracy",
        conditions={
            "min_metric_value": {"metric": "accuracy", "value": 0.90},
            "stage": "Staging"
        },
        actions=["transition_to_Production"]
    )
    
    # Check lifecycle rules
    triggered_actions = lifecycle_manager.check_lifecycle_rules("image_classifier", "2")
    
    if triggered_actions:
        print(f"Triggered actions: {triggered_actions}")
        
        # Execute actions
        results = lifecycle_manager.execute_lifecycle_actions(
            "image_classifier", "2", triggered_actions
        )
        
        print("Action results:", results)
    else:
        print("No lifecycle rules triggered")
    
    print("\n8. MLflow Best Practices")
    print("-" * 28)
    
    best_practices = [
        "Use meaningful experiment and model names",
        "Log comprehensive hyperparameters and metrics",
        "Include model descriptions and tags for searchability",
        "Implement proper stage transition workflows",
        "Use lifecycle management for model governance",
        "Track model lineage and dependencies",
        "Monitor model performance in production",
        "Implement automated testing for model versions",
        "Use model signatures for input/output validation",
        "Regular cleanup of old/unused model versions",
        "Backup MLflow tracking and registry data",
        "Implement access controls for production models"
    ]
    
    print("MLflow Model Registry Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. Integration Patterns")
    print("-" * 25)
    
    integration_patterns = {
        "CI/CD Pipeline": "Automate model registration and testing",
        "A/B Testing": "Deploy multiple model versions for comparison",
        "Blue-Green Deployment": "Use staging transitions for zero-downtime deployment",
        "Canary Releases": "Gradual rollout of new model versions",
        "Model Monitoring": "Track model performance and trigger retraining",
        "Feature Store": "Integrate with feature stores for complete ML pipeline"
    }
    
    print("Common Integration Patterns:")
    for pattern, description in integration_patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n10. Troubleshooting Guide")
    print("-" * 28)
    
    troubleshooting = [
        "Model loading fails: Check MLflow server connectivity and model URI",
        "Stage transition denied: Verify permissions and model status",
        "Search returns no results: Check filter syntax and model names",
        "Deployment issues: Validate model compatibility and dependencies",
        "Performance degradation: Monitor tracking server resources",
        "Version conflicts: Implement proper versioning strategy"
    ]
    
    print("Common Issues and Solutions:")
    for i, issue in enumerate(troubleshooting, 1):
        print(f"{i}. {issue}")
    
    print("\nMLflow Model Registry demonstration completed!")
    print("Key features demonstrated:")
    print("  - Model registration with metadata")
    print("  - Stage-based model lifecycle")
    print("  - Model search and discovery")
    print("  - Deployment management")
    print("  - Automated lifecycle rules")
    
    print("\nTo set up MLflow server:")
    print("  1. Install MLflow: pip install mlflow")
    print("  2. Start tracking server: mlflow server --host 0.0.0.0 --port 5000")
    print("  3. Access UI: http://localhost:5000")
    print("  4. Configure client: mlflow.set_tracking_uri('http://localhost:5000')")