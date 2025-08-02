import torch
import torch.nn as nn
import os
import json
import datetime
import hashlib
import shutil
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Sample Model for Versioning Demo
class VersionedModel(nn.Module):
    """Sample model with versioning support"""
    
    def __init__(self, num_classes: int = 10, version: str = "1.0.0"):
        super().__init__()
        
        self.version = version
        self.num_classes = num_classes
        
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

# Version Management System
class ModelVersionManager:
    """Comprehensive model version management system"""
    
    def __init__(self, base_dir: str = "model_versions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.registry_dir = self.base_dir / "registry"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.registry_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load or create registry
        self.registry_file = self.registry_dir / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": {},
                "created_at": datetime.datetime.now().isoformat(),
                "version": "1.0.0"
            }
    
    def _save_registry(self):
        """Save registry to file"""
        
        self.registry["updated_at"] = datetime.datetime.now().isoformat()
        
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model file"""
        
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def register_model(self, model: nn.Module, 
                      model_name: str,
                      version: str,
                      description: str = "",
                      tags: List[str] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """Register a new model version"""
        
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        # Create model directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_file = model_dir / f"{model_name}_v{version}.pth"
        torch.save(model.state_dict(), model_file)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_file)
        
        # Create version metadata
        version_metadata = {
            "model_name": model_name,
            "version": version,
            "description": description,
            "tags": tags,
            "model_hash": model_hash,
            "model_file": str(model_file.relative_to(self.base_dir)),
            "file_size_bytes": model_file.stat().st_size,
            "created_at": datetime.datetime.now().isoformat(),
            "model_class": model.__class__.__name__,
            "custom_metadata": metadata
        }
        
        # Add model configuration if available
        if hasattr(model, 'num_classes'):
            version_metadata['num_classes'] = model.num_classes
        if hasattr(model, 'version'):
            version_metadata['model_version'] = model.version
        
        # Save version metadata
        metadata_file = self.metadata_dir / f"{model_name}_v{version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(version_metadata, f, indent=2)
        
        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "versions": {},
                "latest_version": version
            }
        
        self.registry["models"][model_name]["versions"][version] = {
            "metadata_file": str(metadata_file.relative_to(self.base_dir)),
            "registered_at": datetime.datetime.now().isoformat()
        }
        
        # Update latest version
        versions = list(self.registry["models"][model_name]["versions"].keys())
        latest = self._get_latest_version(versions)
        self.registry["models"][model_name]["latest_version"] = latest
        
        self._save_registry()
        
        print(f"✓ Model registered: {model_name} v{version}")
        return f"{model_name}_v{version}"
    
    def load_model(self, model_name: str, 
                  version: str = "latest",
                  model_class: type = None) -> nn.Module:
        """Load a specific model version"""
        
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Get version
        if version == "latest":
            version = self.registry["models"][model_name]["latest_version"]
        
        if version not in self.registry["models"][model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        # Load metadata
        metadata_file = self.base_dir / self.registry["models"][model_name]["versions"][version]["metadata_file"]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_file = self.base_dir / metadata["model_file"]
        
        if model_class is None:
            # Try to determine model class from metadata
            model_class_name = metadata.get("model_class", "VersionedModel")
            if model_class_name == "VersionedModel":
                model_class = VersionedModel
            else:
                raise ValueError(f"Model class '{model_class_name}' not available. Please provide model_class parameter.")
        
        # Create model instance
        model_kwargs = {}
        if "num_classes" in metadata:
            model_kwargs["num_classes"] = metadata["num_classes"]
        if "model_version" in metadata:
            model_kwargs["version"] = metadata["model_version"]
        
        model = model_class(**model_kwargs)
        
        # Load state dict
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"✓ Model loaded: {model_name} v{version}")
        return model
    
    def list_models(self) -> Dict[str, List[str]]:
        """List all models and their versions"""
        
        result = {}
        for model_name, model_info in self.registry["models"].items():
            result[model_name] = list(model_info["versions"].keys())
        
        return result
    
    def get_model_info(self, model_name: str, version: str = "latest") -> Dict[str, Any]:
        """Get detailed information about a model version"""
        
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")
        
        if version == "latest":
            version = self.registry["models"][model_name]["latest_version"]
        
        if version not in self.registry["models"][model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found")
        
        # Load metadata
        metadata_file = self.base_dir / self.registry["models"][model_name]["versions"][version]["metadata_file"]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific model version"""
        
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")
        
        if version not in self.registry["models"][model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found")
        
        # Get metadata
        metadata_file = self.base_dir / self.registry["models"][model_name]["versions"][version]["metadata_file"]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Delete files
        model_file = self.base_dir / metadata["model_file"]
        
        if model_file.exists():
            os.remove(model_file)
        if metadata_file.exists():
            os.remove(metadata_file)
        
        # Update registry
        del self.registry["models"][model_name]["versions"][version]
        
        # Update latest version if necessary
        remaining_versions = list(self.registry["models"][model_name]["versions"].keys())
        if remaining_versions:
            self.registry["models"][model_name]["latest_version"] = self._get_latest_version(remaining_versions)
        else:
            # No versions left, remove model entirely
            del self.registry["models"][model_name]
        
        self._save_registry()
        
        print(f"✓ Deleted: {model_name} v{version}")
        return True
    
    def _get_latest_version(self, versions: List[str]) -> str:
        """Get the latest version from a list of version strings"""
        
        # Simple semantic version sorting
        def version_key(v):
            try:
                parts = v.split('.')
                return tuple(int(part) for part in parts)
            except ValueError:
                return (0, 0, 0)
        
        return max(versions, key=version_key)
    
    def create_backup(self, backup_path: str = None) -> str:
        """Create backup of entire model repository"""
        
        if backup_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"model_backup_{timestamp}.tar.gz"
        
        shutil.make_archive(
            backup_path.replace('.tar.gz', ''),
            'gztar',
            str(self.base_dir.parent),
            str(self.base_dir.name)
        )
        
        print(f"✓ Backup created: {backup_path}")
        return backup_path

# Model Comparison and Analysis
class ModelVersionComparator:
    """Compare different model versions"""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
    
    def compare_versions(self, model_name: str, 
                        versions: List[str] = None) -> Dict[str, Any]:
        """Compare multiple versions of a model"""
        
        if versions is None:
            # Compare all versions
            all_models = self.version_manager.list_models()
            if model_name not in all_models:
                raise ValueError(f"Model '{model_name}' not found")
            versions = all_models[model_name]
        
        comparison = {}
        
        for version in versions:
            try:
                metadata = self.version_manager.get_model_info(model_name, version)
                
                comparison[version] = {
                    'file_size_mb': metadata['file_size_bytes'] / (1024**2),
                    'created_at': metadata['created_at'],
                    'description': metadata.get('description', ''),
                    'tags': metadata.get('tags', []),
                    'model_hash': metadata['model_hash']
                }
                
            except Exception as e:
                comparison[version] = {'error': str(e)}
        
        return comparison
    
    def find_differences(self, model_name: str, 
                        version1: str, version2: str) -> Dict[str, Any]:
        """Find differences between two model versions"""
        
        metadata1 = self.version_manager.get_model_info(model_name, version1)
        metadata2 = self.version_manager.get_model_info(model_name, version2)
        
        differences = {
            'file_size_diff_mb': (metadata2['file_size_bytes'] - metadata1['file_size_bytes']) / (1024**2),
            'hash_changed': metadata1['model_hash'] != metadata2['model_hash'],
            'description_changed': metadata1.get('description', '') != metadata2.get('description', ''),
            'tags_changed': set(metadata1.get('tags', [])) != set(metadata2.get('tags', [])),
            'time_diff_days': self._calculate_time_diff(metadata1['created_at'], metadata2['created_at'])
        }
        
        return differences
    
    def _calculate_time_diff(self, time1_str: str, time2_str: str) -> float:
        """Calculate time difference in days"""
        
        try:
            time1 = datetime.datetime.fromisoformat(time1_str.replace('Z', '+00:00'))
            time2 = datetime.datetime.fromisoformat(time2_str.replace('Z', '+00:00'))
            return (time2 - time1).total_seconds() / (24 * 3600)
        except:
            return 0.0

# Semantic Versioning
class SemanticVersioning:
    """Semantic versioning utilities for models"""
    
    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int]:
        """Parse semantic version string"""
        
        try:
            parts = version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return major, minor, patch
        except ValueError:
            return 0, 0, 0
    
    @staticmethod
    def increment_version(current_version: str, 
                         level: str = "patch") -> str:
        """Increment version according to semantic versioning"""
        
        major, minor, patch = SemanticVersioning.parse_version(current_version)
        
        if level == "major":
            major += 1
            minor = 0
            patch = 0
        elif level == "minor":
            minor += 1
            patch = 0
        elif level == "patch":
            patch += 1
        else:
            raise ValueError("Level must be 'major', 'minor', or 'patch'")
        
        return f"{major}.{minor}.{patch}"
    
    @staticmethod
    def suggest_version_increment(changes: Dict[str, Any]) -> str:
        """Suggest version increment based on changes"""
        
        # Simple heuristics for version increment
        if changes.get('architecture_changed', False):
            return "major"
        elif changes.get('new_features', False) or changes.get('significant_accuracy_change', False):
            return "minor"
        else:
            return "patch"

if __name__ == "__main__":
    print("Model Versioning and Management")
    print("=" * 35)
    
    # Initialize version manager
    version_manager = ModelVersionManager("demo_model_versions")
    
    print("\n1. Model Registration")
    print("-" * 24)
    
    # Create sample models with different versions
    model_v1 = VersionedModel(num_classes=10, version="1.0.0")
    model_v1_1 = VersionedModel(num_classes=10, version="1.1.0")
    model_v2 = VersionedModel(num_classes=10, version="2.0.0")
    
    # Register models
    version_manager.register_model(
        model_v1, "image_classifier", "1.0.0",
        description="Initial model version",
        tags=["cnn", "classification", "baseline"],
        metadata={"accuracy": 0.85, "dataset": "CIFAR-10"}
    )
    
    version_manager.register_model(
        model_v1_1, "image_classifier", "1.1.0",
        description="Improved model with better regularization",
        tags=["cnn", "classification", "improved"],
        metadata={"accuracy": 0.87, "dataset": "CIFAR-10", "improvements": "added dropout"}
    )
    
    version_manager.register_model(
        model_v2, "image_classifier", "2.0.0",
        description="Major architecture update",
        tags=["cnn", "classification", "major_update"],
        metadata={"accuracy": 0.91, "dataset": "CIFAR-10", "changes": "new architecture"}
    )
    
    print("\n2. Model Listing and Information")
    print("-" * 37)
    
    # List all models
    all_models = version_manager.list_models()
    print("Available models:")
    for model_name, versions in all_models.items():
        print(f"  {model_name}: versions {versions}")
    
    # Get model information
    latest_info = version_manager.get_model_info("image_classifier", "latest")
    print(f"\nLatest version info:")
    print(f"  Version: {latest_info['version']}")
    print(f"  Description: {latest_info['description']}")
    print(f"  Created: {latest_info['created_at']}")
    print(f"  File size: {latest_info['file_size_bytes'] / (1024**2):.2f} MB")
    print(f"  Accuracy: {latest_info['custom_metadata'].get('accuracy', 'N/A')}")
    
    print("\n3. Model Loading")
    print("-" * 18)
    
    # Load different versions
    loaded_v1 = version_manager.load_model("image_classifier", "1.0.0", VersionedModel)
    loaded_latest = version_manager.load_model("image_classifier", "latest", VersionedModel)
    
    print(f"Loaded v1.0.0: {loaded_v1.version}")
    print(f"Loaded latest: {loaded_latest.version}")
    
    print("\n4. Version Comparison")
    print("-" * 24)
    
    comparator = ModelVersionComparator(version_manager)
    
    # Compare all versions
    comparison = comparator.compare_versions("image_classifier")
    
    print("Version Comparison:")
    print("-" * 20)
    print(f"{'Version':<10} {'Size (MB)':<12} {'Created':<20} {'Description':<30}")
    print("-" * 75)
    
    for version, info in comparison.items():
        if 'error' not in info:
            size_mb = info['file_size_mb']
            created = info['created_at'][:10]  # Just the date
            description = info['description'][:28] + ".." if len(info['description']) > 30 else info['description']
            
            print(f"{version:<10} {size_mb:<12.2f} {created:<20} {description:<30}")
    
    # Find differences between versions
    diff_1_to_2 = comparator.find_differences("image_classifier", "1.0.0", "2.0.0")
    
    print(f"\nDifferences between v1.0.0 and v2.0.0:")
    print(f"  File size change: {diff_1_to_2['file_size_diff_mb']:+.2f} MB")
    print(f"  Hash changed: {diff_1_to_2['hash_changed']}")
    print(f"  Time difference: {diff_1_to_2['time_diff_days']:.1f} days")
    
    print("\n5. Semantic Versioning")
    print("-" * 25)
    
    # Demonstrate semantic versioning
    current_version = "1.2.3"
    
    patch_version = SemanticVersioning.increment_version(current_version, "patch")
    minor_version = SemanticVersioning.increment_version(current_version, "minor")
    major_version = SemanticVersioning.increment_version(current_version, "major")
    
    print(f"Current version: {current_version}")
    print(f"Patch increment: {patch_version}")
    print(f"Minor increment: {minor_version}")
    print(f"Major increment: {major_version}")
    
    # Suggest version increment
    changes = {
        "architecture_changed": False,
        "new_features": True,
        "significant_accuracy_change": False
    }
    
    suggested_level = SemanticVersioning.suggest_version_increment(changes)
    suggested_version = SemanticVersioning.increment_version(current_version, suggested_level)
    
    print(f"\nSuggested increment level: {suggested_level}")
    print(f"Suggested next version: {suggested_version}")
    
    print("\n6. Model Repository Backup")
    print("-" * 31)
    
    # Create backup
    backup_file = version_manager.create_backup()
    
    print("\n7. Best Practices Summary")
    print("-" * 29)
    
    best_practices = [
        "Use semantic versioning (major.minor.patch)",
        "Include meaningful descriptions for each version",
        "Tag versions with relevant metadata",
        "Store model metrics and performance data",
        "Create backups of model repositories regularly",
        "Use consistent naming conventions",
        "Document breaking changes in major versions",
        "Keep track of training datasets and parameters",
        "Implement automated version testing",
        "Monitor model performance in production",
        "Plan for model deprecation and migration",
        "Maintain compatibility matrices"
    ]
    
    print("Model Versioning Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Version Management Workflow")
    print("-" * 35)
    
    workflow = [
        "1. Develop and train new model",
        "2. Evaluate against previous versions",
        "3. Determine appropriate version increment",
        "4. Register model with metadata",
        "5. Test deployment of new version",
        "6. Deploy to staging environment",
        "7. A/B test against current version",
        "8. Deploy to production",
        "9. Monitor performance metrics",
        "10. Document lessons learned"
    ]
    
    print("Version Management Workflow:")
    for step in workflow:
        print(f"  {step}")
    
    print("\nModel versioning demonstration completed!")
    print("Generated structure:")
    print("  - demo_model_versions/")
    print("    - models/ (model files)")
    print("    - metadata/ (version metadata)")
    print("    - registry/ (model registry)")
    print(f"  - {backup_file} (backup archive)")
    
    print("\nKey features demonstrated:")
    print("  - Semantic versioning")
    print("  - Model registration and loading")
    print("  - Version comparison and analysis")
    print("  - Automated backup creation")
    print("  - Metadata management")