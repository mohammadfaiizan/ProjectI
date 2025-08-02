import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import pickle
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import datetime

# Sample Models for Save/Load Demo
class ProductionCNN(nn.Module):
    """Production-ready CNN with metadata support"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
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
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetBlock(nn.Module):
    """ResNet building block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ProductionResNet(nn.Module):
    """Production ResNet with configurable depth"""
    
    def __init__(self, num_classes=10, layers=[2, 2, 2, 2]):
        super().__init__()
        
        self.num_classes = num_classes
        self.layers = layers
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.in_channels = 64
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Model Saver/Loader Classes
class ModelSaver:
    """Comprehensive model saving utilities"""
    
    def __init__(self, save_dir: str = "saved_models"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_state_dict(self, model: nn.Module, filepath: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Save model state dictionary with metadata"""
        
        save_path = self.save_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        # Add model configuration if available
        if hasattr(model, 'num_classes'):
            save_data['num_classes'] = model.num_classes
        if hasattr(model, 'input_channels'):
            save_data['input_channels'] = model.input_channels
        if hasattr(model, 'layers'):
            save_data['layers'] = model.layers
        
        # Add custom metadata
        if metadata:
            save_data['metadata'] = metadata
        
        # Calculate checksum
        state_dict_str = str(model.state_dict())
        save_data['checksum'] = hashlib.md5(state_dict_str.encode()).hexdigest()
        
        torch.save(save_data, save_path)
        print(f"Model state dict saved to: {save_path}")
    
    def save_complete_model(self, model: nn.Module, filepath: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """Save complete model (architecture + weights)"""
        
        save_path = self.save_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata to model if provided
        if metadata:
            model.metadata = metadata
        
        # Add timestamp
        model.save_timestamp = datetime.datetime.now().isoformat()
        
        torch.save(model, save_path)
        print(f"Complete model saved to: {save_path}")
    
    def save_for_inference(self, model: nn.Module, filepath: str,
                          example_input: torch.Tensor,
                          optimize: bool = True):
        """Save model optimized for inference"""
        
        save_path = self.save_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set model to eval mode
        model.eval()
        
        # Apply optimizations if requested
        if optimize:
            # Fuse conv-bn layers if possible
            model = self._fuse_conv_bn(model)
        
        # Trace the model for faster inference
        try:
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save(str(save_path))
            print(f"Traced model saved to: {save_path}")
        except Exception as e:
            print(f"Tracing failed: {e}")
            # Fall back to regular save
            torch.save(model, save_path)
            print(f"Model saved (without tracing) to: {save_path}")
    
    def save_with_optimizer(self, model: nn.Module, optimizer,
                           epoch: int, loss: float, filepath: str,
                           scheduler=None):
        """Save model with training state"""
        
        save_path = self.save_dir / filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, save_path)
        print(f"Training checkpoint saved to: {save_path}")
    
    def _fuse_conv_bn(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn layers for inference optimization"""
        # This is a simplified version - real implementation would be more complex
        return model

class ModelLoader:
    """Comprehensive model loading utilities"""
    
    def __init__(self, save_dir: str = "saved_models"):
        self.save_dir = Path(save_dir)
    
    def load_state_dict(self, filepath: str, model_class,
                       model_kwargs: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Load model from state dictionary"""
        
        load_path = self.save_dir / filepath
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load saved data
        saved_data = torch.load(load_path, map_location='cpu')
        
        # Extract model configuration
        if model_kwargs is None:
            model_kwargs = {}
        
        # Use saved configuration if available
        for key in ['num_classes', 'input_channels', 'layers']:
            if key in saved_data and key not in model_kwargs:
                model_kwargs[key] = saved_data[key]
        
        # Create model instance
        model = model_class(**model_kwargs)
        
        # Load state dict
        model.load_state_dict(saved_data['state_dict'])
        
        # Verify checksum if available
        if 'checksum' in saved_data:
            current_checksum = hashlib.md5(str(model.state_dict()).encode()).hexdigest()
            if current_checksum != saved_data['checksum']:
                print("Warning: Model checksum mismatch!")
        
        print(f"Model loaded from: {load_path}")
        return model
    
    def load_complete_model(self, filepath: str) -> nn.Module:
        """Load complete model"""
        
        load_path = self.save_dir / filepath
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model = torch.load(load_path, map_location='cpu')
        print(f"Complete model loaded from: {load_path}")
        return model
    
    def load_for_inference(self, filepath: str) -> nn.Module:
        """Load model optimized for inference"""
        
        load_path = self.save_dir / filepath
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            # Try loading as traced model first
            model = torch.jit.load(str(load_path), map_location='cpu')
            print(f"Traced model loaded from: {load_path}")
        except Exception:
            # Fall back to regular loading
            model = torch.load(load_path, map_location='cpu')
            print(f"Regular model loaded from: {load_path}")
        
        model.eval()
        return model
    
    def load_checkpoint(self, filepath: str, model_class, 
                       model_kwargs: Optional[Dict[str, Any]] = None):
        """Load training checkpoint"""
        
        load_path = self.save_dir / filepath
        
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # Create model
        if model_kwargs is None:
            model_kwargs = {}
        
        model = model_class(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Checkpoint loaded from: {load_path}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        
        return model, checkpoint

class ModelRegistry:
    """Model registry for versioning and management"""
    
    def __init__(self, registry_dir: str = "model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"
        self.models_dir = self.registry_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "next_version": 1}
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model: nn.Module, model_name: str,
                      description: str = "", tags: List[str] = None,
                      metrics: Dict[str, float] = None) -> str:
        """Register a new model version"""
        
        if tags is None:
            tags = []
        if metrics is None:
            metrics = {}
        
        # Generate version
        version = str(self.registry["next_version"])
        self.registry["next_version"] += 1
        
        # Create model entry
        model_id = f"{model_name}_v{version}"
        
        # Save model
        model_path = self.models_dir / f"{model_id}.pth"
        
        model_info = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "description": description,
            "tags": tags,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "file_path": str(model_path),
            "model_class": model.__class__.__name__,
        }
        
        # Add model configuration
        if hasattr(model, 'num_classes'):
            model_info['num_classes'] = model.num_classes
        if hasattr(model, 'input_channels'):
            model_info['input_channels'] = model.input_channels
        
        # Save model file
        torch.save({
            'model': model,
            'model_info': model_info
        }, model_path)
        
        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {}
        
        self.registry["models"][model_name][version] = model_info
        self._save_registry()
        
        print(f"Model registered: {model_id}")
        return model_id
    
    def get_model(self, model_name: str, version: str = "latest") -> nn.Module:
        """Get model by name and version"""
        
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_versions = self.registry["models"][model_name]
        
        if version == "latest":
            # Get latest version
            latest_version = max(model_versions.keys(), key=int)
            model_info = model_versions[latest_version]
        else:
            if version not in model_versions:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            model_info = model_versions[version]
        
        # Load model
        model_path = Path(model_info["file_path"])
        saved_data = torch.load(model_path, map_location='cpu')
        
        return saved_data['model']
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all registered models and their versions"""
        
        result = {}
        for model_name, versions in self.registry["models"].items():
            result[model_name] = list(versions.keys())
        
        return result
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get model information"""
        
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_versions = self.registry["models"][model_name]
        
        if version == "latest":
            latest_version = max(model_versions.keys(), key=int)
            return model_versions[latest_version]
        else:
            if version not in model_versions:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            return model_versions[version]

# Backup and Migration Utilities
class ModelBackup:
    """Model backup and migration utilities"""
    
    def __init__(self, backup_dir: str = "model_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, source_dir: str, backup_name: str = None):
        """Create compressed backup of models"""
        
        import shutil
        
        if backup_name is None:
            backup_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        source_path = Path(source_dir)
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        # Create compressed archive
        shutil.make_archive(
            str(backup_path.with_suffix('')),
            'gztar',
            str(source_path.parent),
            str(source_path.name)
        )
        
        print(f"Backup created: {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, backup_path: str, restore_dir: str):
        """Restore models from backup"""
        
        import shutil
        import tarfile
        
        backup_file = Path(backup_path)
        restore_path = Path(restore_dir)
        restore_path.mkdir(parents=True, exist_ok=True)
        
        # Extract archive
        with tarfile.open(backup_file, 'r:gz') as tar:
            tar.extractall(str(restore_path.parent))
        
        print(f"Backup restored to: {restore_path}")

class ModelMigration:
    """Model migration utilities for version compatibility"""
    
    @staticmethod
    def migrate_state_dict(old_state_dict: Dict[str, torch.Tensor],
                          migration_map: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Migrate state dict with parameter name changes"""
        
        new_state_dict = {}
        
        for old_key, tensor in old_state_dict.items():
            new_key = migration_map.get(old_key, old_key)
            new_state_dict[new_key] = tensor
        
        return new_state_dict
    
    @staticmethod
    def convert_bn_to_gn(model: nn.Module, num_groups: int = 32) -> nn.Module:
        """Convert BatchNorm layers to GroupNorm"""
        
        def replace_bn_with_gn(module):
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features
                    gn = nn.GroupNorm(num_groups, num_channels)
                    setattr(module, name, gn)
                else:
                    replace_bn_with_gn(child)
        
        replace_bn_with_gn(model)
        return model

if __name__ == "__main__":
    print("Model Saving and Loading")
    print("=" * 30)
    
    # Create sample models
    cnn_model = ProductionCNN(num_classes=10)
    resnet_model = ProductionResNet(num_classes=10)
    
    # Initialize saver and loader
    saver = ModelSaver("demo_models")
    loader = ModelLoader("demo_models")
    
    print("\n1. Basic State Dict Save/Load")
    print("-" * 35)
    
    # Save state dict
    metadata = {
        "accuracy": 0.95,
        "dataset": "CIFAR-10",
        "training_epochs": 100
    }
    
    saver.save_state_dict(cnn_model, "cnn_state_dict.pth", metadata)
    
    # Load state dict
    loaded_cnn = loader.load_state_dict("cnn_state_dict.pth", ProductionCNN,
                                       {"num_classes": 10})
    
    print("✓ State dict save/load successful")
    
    print("\n2. Complete Model Save/Load")
    print("-" * 32)
    
    # Save complete model
    saver.save_complete_model(resnet_model, "resnet_complete.pth", metadata)
    
    # Load complete model
    loaded_resnet = loader.load_complete_model("resnet_complete.pth")
    
    print("✓ Complete model save/load successful")
    
    print("\n3. Inference-Optimized Save/Load")
    print("-" * 40)
    
    # Save for inference
    example_input = torch.randn(1, 3, 32, 32)
    saver.save_for_inference(cnn_model, "cnn_inference.pth", example_input)
    
    # Load for inference
    inference_model = loader.load_for_inference("cnn_inference.pth")
    
    # Test inference
    with torch.no_grad():
        output = inference_model(example_input)
        print(f"✓ Inference model output shape: {output.shape}")
    
    print("\n4. Training Checkpoint Save/Load")
    print("-" * 40)
    
    # Create optimizer
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Save checkpoint
    saver.save_with_optimizer(cnn_model, optimizer, epoch=50, loss=0.234,
                             filepath="checkpoint_epoch_50.pth", scheduler=scheduler)
    
    # Load checkpoint
    loaded_model, checkpoint_data = loader.load_checkpoint(
        "checkpoint_epoch_50.pth", ProductionCNN, {"num_classes": 10}
    )
    
    print("✓ Training checkpoint save/load successful")
    
    print("\n5. Model Registry")
    print("-" * 20)
    
    # Initialize registry
    registry = ModelRegistry("demo_registry")
    
    # Register models
    cnn_id = registry.register_model(
        cnn_model, "ProductionCNN",
        description="CNN for image classification",
        tags=["vision", "cnn", "production"],
        metrics={"accuracy": 0.95, "f1_score": 0.94}
    )
    
    resnet_id = registry.register_model(
        resnet_model, "ProductionResNet",
        description="ResNet for image classification",
        tags=["vision", "resnet", "production"],
        metrics={"accuracy": 0.97, "f1_score": 0.96}
    )
    
    # List models
    print("\nRegistered models:")
    models_list = registry.list_models()
    for name, versions in models_list.items():
        print(f"  {name}: versions {versions}")
    
    # Get model info
    cnn_info = registry.get_model_info("ProductionCNN", "latest")
    print(f"\nCNN Info: {cnn_info['description']}")
    print(f"Metrics: {cnn_info['metrics']}")
    
    # Load model from registry
    registry_cnn = registry.get_model("ProductionCNN", "latest")
    print("✓ Model loaded from registry")
    
    print("\n6. Model Backup and Migration")
    print("-" * 35)
    
    # Create backup
    backup_tool = ModelBackup("demo_backups")
    backup_path = backup_tool.create_backup("demo_models")
    
    # Migration example
    migration_tool = ModelMigration()
    
    # Example migration map for parameter name changes
    migration_map = {
        "old_layer.weight": "new_layer.weight",
        "old_layer.bias": "new_layer.bias"
    }
    
    # Convert BatchNorm to GroupNorm
    gn_model = migration_tool.convert_bn_to_gn(ProductionCNN(num_classes=10))
    print("✓ Model converted from BatchNorm to GroupNorm")
    
    print("\n7. Model Validation")
    print("-" * 23)
    
    # Validate models are equivalent
    def validate_models_equivalent(model1, model2, input_tensor):
        """Check if two models produce the same output"""
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
            
            # Check if outputs are close
            max_diff = torch.max(torch.abs(output1 - output2)).item()
            return max_diff < 1e-6, max_diff
    
    # Test original vs loaded model
    is_equivalent, max_diff = validate_models_equivalent(
        cnn_model, loaded_cnn, example_input
    )
    
    print(f"Models equivalent: {is_equivalent}")
    print(f"Max difference: {max_diff:.2e}")
    
    print("\n8. Best Practices Summary")
    print("-" * 32)
    
    best_practices = [
        "Always save model metadata (architecture, hyperparams, metrics)",
        "Use state_dict for flexibility, complete model for simplicity",
        "Include checksums for integrity verification",
        "Save optimizer state for resuming training",
        "Use model registry for version management",
        "Create regular backups of important models",
        "Test loaded models before deployment",
        "Use inference-optimized formats for production",
        "Document model versions and changes",
        "Implement model migration for compatibility"
    ]
    
    print("Model Save/Load Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\nModel saving and loading demonstration completed!")
    print("Generated directories:")
    print("  - demo_models/ (saved models)")
    print("  - demo_registry/ (model registry)")
    print("  - demo_backups/ (model backups)")