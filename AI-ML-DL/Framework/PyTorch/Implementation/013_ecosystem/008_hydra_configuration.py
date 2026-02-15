import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import yaml
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

# Note: Hydra operations require the hydra-core package
# Install with: pip install hydra-core

try:
    import hydra
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import DictConfig, OmegaConf, MISSING
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("Warning: Hydra not available. Install with: pip install hydra-core")

# Configuration Data Classes
@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "simple_mlp"
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    num_classes: int = 10
    dropout: float = 0.2
    activation: str = "relu"
    batch_norm: bool = True

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "step"
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5  # For ReduceLROnPlateau
    factor: float = 0.5  # For ReduceLROnPlateau
    milestones: List[int] = field(default_factory=lambda: [30, 60, 80])

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "synthetic"
    data_dir: str = "./data"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_size: int = 10000
    val_size: int = 2000
    test_size: int = 1000
    input_dim: int = 784
    num_classes: int = 10
    noise_level: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    save_every: int = 10
    log_every: int = 50
    validate_every: int = 1
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    save_best_only: bool = True
    resume_from_checkpoint: Optional[str] = None

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "default_experiment"
    description: str = "Default experiment description"
    tags: List[str] = field(default_factory=list)
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "./outputs"
    log_level: str = "INFO"

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

# Model Factory
class ModelFactory:
    """Factory for creating models based on configuration"""
    
    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        """Create model from configuration"""
        
        if config.name == "simple_mlp":
            return ModelFactory._create_mlp(config)
        elif config.name == "cnn":
            return ModelFactory._create_cnn(config)
        elif config.name == "resnet":
            return ModelFactory._create_resnet(config)
        else:
            raise ValueError(f"Unknown model type: {config.name}")
    
    @staticmethod
    def _create_mlp(config: ModelConfig) -> nn.Module:
        """Create MLP model"""
        
        class MLP(nn.Module):
            def __init__(self, config: ModelConfig):
                super().__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                prev_size = config.input_size
                
                # Hidden layers
                for hidden_size in config.hidden_sizes:
                    layers = [nn.Linear(prev_size, hidden_size)]
                    
                    if config.batch_norm:
                        layers.append(nn.BatchNorm1d(hidden_size))
                    
                    if config.activation == "relu":
                        layers.append(nn.ReLU())
                    elif config.activation == "gelu":
                        layers.append(nn.GELU())
                    elif config.activation == "tanh":
                        layers.append(nn.Tanh())
                    
                    layers.append(nn.Dropout(config.dropout))
                    self.layers.extend(layers)
                    prev_size = hidden_size
                
                # Output layer
                self.output = nn.Linear(prev_size, config.num_classes)
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                x = self.output(x)
                return x
        
        return MLP(config)
    
    @staticmethod
    def _create_cnn(config: ModelConfig) -> nn.Module:
        """Create CNN model"""
        
        class CNN(nn.Module):
            def __init__(self, config: ModelConfig):
                super().__init__()
                
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.BatchNorm2d(32) if config.batch_norm else nn.Identity(),
                    nn.ReLU() if config.activation == "relu" else nn.GELU(),
                    nn.MaxPool2d(2),
                    
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64) if config.batch_norm else nn.Identity(),
                    nn.ReLU() if config.activation == "relu" else nn.GELU(),
                    nn.MaxPool2d(2),
                    
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 16, config.hidden_sizes[0]),
                    nn.ReLU() if config.activation == "relu" else nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_sizes[0], config.num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return CNN(config)
    
    @staticmethod
    def _create_resnet(config: ModelConfig) -> nn.Module:
        """Create simple ResNet-style model"""
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        class SimpleResNet(nn.Module):
            def __init__(self, config: ModelConfig):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
                self.bn1 = nn.BatchNorm2d(32)
                
                self.layer1 = ResidualBlock(32, 32)
                self.layer2 = ResidualBlock(32, 64, 2)
                self.layer3 = ResidualBlock(64, 128, 2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(128, config.num_classes)
                self.dropout = nn.Dropout(config.dropout)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        return SimpleResNet(config)

# Optimizer Factory
class OptimizerFactory:
    """Factory for creating optimizers"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
        """Create optimizer from configuration"""
        
        if config.name.lower() == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.name.lower() == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.name.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.name.lower() == "rmsprop":
            return torch.optim.RMSprop(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.name}")

# Scheduler Factory
class SchedulerFactory:
    """Factory for creating learning rate schedulers"""
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, 
                        config: SchedulerConfig) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create scheduler from configuration"""
        
        if config.name.lower() == "none":
            return None
        elif config.name.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.name.lower() == "multistep":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.milestones,
                gamma=config.gamma
            )
        elif config.name.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config.patience,
                factor=config.factor,
                verbose=True
            )
        elif config.name.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=100,  # This should ideally come from config
                eta_min=0
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.name}")

# Dataset Classes
class ConfigurableDataset(Dataset):
    """Dataset that can be configured via Hydra"""
    
    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Determine size based on split
        if split == "train":
            self.size = config.train_size
        elif split == "val":
            self.size = config.val_size
        elif split == "test":
            self.size = config.test_size
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Generate data
        self.data = torch.randn(self.size, config.input_dim)
        
        # Generate labels with some pattern
        if config.dataset_name == "synthetic":
            # Create labels based on data patterns
            weights = torch.randn(config.input_dim, config.num_classes)
            logits = self.data @ weights + config.noise_level * torch.randn(self.size, config.num_classes)
            self.labels = torch.argmax(logits, dim=1)
        else:
            # Random labels
            self.labels = torch.randint(0, config.num_classes, (self.size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Data Module
class DataModule:
    """Data module with Hydra configuration"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self):
        """Setup data loaders"""
        
        # Create datasets
        train_dataset = ConfigurableDataset(self.config, "train")
        val_dataset = ConfigurableDataset(self.config, "val")
        test_dataset = ConfigurableDataset(self.config, "test")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print(f"✓ Data loaders created: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

# Trainer with Hydra Configuration
class HydraTrainer:
    """Trainer configured with Hydra"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data_module = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup output directory
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self._set_seed(config.experiment.seed)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config.experiment.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup(self):
        """Setup model, optimizer, scheduler, and data"""
        
        # Create model
        self.model = ModelFactory.create_model(self.config.model)
        print(f"✓ Created model: {self.config.model.name}")
        
        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(self.model, self.config.optimizer)
        print(f"✓ Created optimizer: {self.config.optimizer.name}")
        
        # Create scheduler
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, self.config.scheduler)
        if self.scheduler:
            print(f"✓ Created scheduler: {self.config.scheduler.name}")
        
        # Setup data
        self.data_module = DataModule(self.config.data)
        self.data_module.setup()
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save configuration to output directory"""
        if HYDRA_AVAILABLE:
            config_path = self.output_dir / "config.yaml"
            with open(config_path, 'w') as f:
                OmegaConf.save(self.config, f)
            print(f"✓ Configuration saved to {config_path}")
        else:
            # Fallback: save as JSON
            config_dict = self._config_to_dict(self.config)
            config_path = self.output_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✓ Configuration saved to {config_path}")
    
    def _config_to_dict(self, config):
        """Convert config to dictionary (fallback for when Hydra is not available)"""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        return config
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.data_module.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % self.config.training.log_every == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        epoch_loss = total_loss / len(self.data_module.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.data_module.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss = total_loss / len(self.data_module.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """Complete training loop"""
        print(f"🚀 Starting training for {self.config.training.num_epochs} epochs...")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.training.validate_every == 0:
                val_loss, val_acc = self.validate()
                
                # Scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if self.config.training.early_stopping:
                    if val_acc > best_val_acc + self.config.training.min_delta:
                        best_val_acc = val_acc
                        patience_counter = 0
                        
                        # Save best model
                        if self.config.training.save_best_only:
                            self._save_model(epoch, "best")
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= self.config.training.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, "
                      f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
            
            # Regular checkpointing
            if epoch % self.config.training.save_every == 0:
                self._save_model(epoch, "checkpoint")
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def _save_model(self, epoch: int, model_type: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.output_dir / f"{model_type}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved {model_type} model to {checkpoint_path}")

# Configuration Management Utilities
class ConfigManager:
    """Utilities for managing Hydra configurations"""
    
    @staticmethod
    def register_configs():
        """Register configuration schemas with Hydra"""
        if not HYDRA_AVAILABLE:
            return
        
        cs = ConfigStore.instance()
        cs.store(name="config", node=Config)
        cs.store(group="model", name="simple_mlp", node=ModelConfig(name="simple_mlp"))
        cs.store(group="model", name="cnn", node=ModelConfig(name="cnn"))
        cs.store(group="model", name="resnet", node=ModelConfig(name="resnet"))
        cs.store(group="optimizer", name="adam", node=OptimizerConfig(name="adam"))
        cs.store(group="optimizer", name="sgd", node=OptimizerConfig(name="sgd"))
        cs.store(group="scheduler", name="step", node=SchedulerConfig(name="step"))
        cs.store(group="scheduler", name="plateau", node=SchedulerConfig(name="plateau"))
        
        print("✓ Configuration schemas registered")
    
    @staticmethod
    def create_config_files():
        """Create example configuration files"""
        config_dir = Path("conf")
        config_dir.mkdir(exist_ok=True)
        
        # Main config
        main_config = """
defaults:
  - model: simple_mlp
  - optimizer: adam
  - scheduler: step
  - _self_

data:
  dataset_name: synthetic
  batch_size: 64
  train_size: 5000
  val_size: 1000
  test_size: 500

training:
  num_epochs: 50
  early_stopping: true
  patience: 10

experiment:
  name: hydra_demo
  seed: 42
  output_dir: ./outputs
"""
        
        with open(config_dir / "config.yaml", 'w') as f:
            f.write(main_config)
        
        # Model configs
        model_dir = config_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        mlp_config = """
name: simple_mlp
input_size: 784
hidden_sizes: [256, 128, 64]
num_classes: 10
dropout: 0.3
activation: relu
batch_norm: true
"""
        
        with open(model_dir / "simple_mlp.yaml", 'w') as f:
            f.write(mlp_config)
        
        cnn_config = """
name: cnn
input_size: 784  # Not used for CNN
hidden_sizes: [128]
num_classes: 10
dropout: 0.5
activation: relu
batch_norm: true
"""
        
        with open(model_dir / "cnn.yaml", 'w') as f:
            f.write(cnn_config)
        
        print(f"✓ Configuration files created in {config_dir}")

# Hydra Application
if HYDRA_AVAILABLE:
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def train_with_hydra(cfg: DictConfig) -> None:
        """Main training function with Hydra"""
        
        print("🔧 Hydra Configuration:")
        print(OmegaConf.to_yaml(cfg))
        
        # Convert DictConfig to structured config
        config = OmegaConf.structured(Config)
        config = OmegaConf.merge(config, cfg)
        
        # Create trainer
        trainer = HydraTrainer(config)
        trainer.setup()
        
        # Train model
        best_acc = trainer.train()
        
        # Return metric for Hydra optimization
        return best_acc

# Fallback training function
def train_without_hydra():
    """Training function without Hydra (fallback)"""
    
    print("Training without Hydra (using default config)")
    
    # Create default config
    config = Config()
    
    # Create trainer
    trainer = HydraTrainer(config)
    trainer.setup()
    
    # Train model
    best_acc = trainer.train()
    
    return best_acc

if __name__ == "__main__":
    print("Hydra Configuration Management")
    print("=" * 33)
    
    if not HYDRA_AVAILABLE:
        print("Hydra not available. Install with: pip install hydra-core")
        print("Showing fallback configuration management...")
    
    print("\n1. Configuration Management Setup")
    print("-" * 36)
    
    config_manager = ConfigManager()
    
    if HYDRA_AVAILABLE:
        # Register configurations
        config_manager.register_configs()
        
        # Create configuration files
        config_manager.create_config_files()
    
    print("\n2. Default Configuration")
    print("-" * 27)
    
    # Show default configuration
    default_config = Config()
    
    print("Default Configuration Structure:")
    print(f"  Model: {default_config.model.name}")
    print(f"  Input Size: {default_config.model.input_size}")
    print(f"  Hidden Sizes: {default_config.model.hidden_sizes}")
    print(f"  Optimizer: {default_config.optimizer.name}")
    print(f"  Learning Rate: {default_config.optimizer.lr}")
    print(f"  Batch Size: {default_config.data.batch_size}")
    print(f"  Epochs: {default_config.training.num_epochs}")
    
    print("\n3. Configuration Factories")
    print("-" * 29)
    
    # Test model factory
    test_model = ModelFactory.create_model(default_config.model)
    print(f"✓ Created model: {test_model.__class__.__name__}")
    
    # Test optimizer factory
    test_optimizer = OptimizerFactory.create_optimizer(test_model, default_config.optimizer)
    print(f"✓ Created optimizer: {test_optimizer.__class__.__name__}")
    
    # Test scheduler factory
    test_scheduler = SchedulerFactory.create_scheduler(test_optimizer, default_config.scheduler)
    if test_scheduler:
        print(f"✓ Created scheduler: {test_scheduler.__class__.__name__}")
    
    print("\n4. Training with Configuration")
    print("-" * 32)
    
    if HYDRA_AVAILABLE:
        print("To run with Hydra configuration:")
        print("  python 008_hydra_configuration.py")
        print("  python 008_hydra_configuration.py model=cnn optimizer=sgd")
        print("  python 008_hydra_configuration.py training.num_epochs=20 data.batch_size=128")
        
        print("\nRunning demo training with default config...")
    
    # Run training (reduced epochs for demo)
    demo_config = Config()
    demo_config.training.num_epochs = 5  # Reduced for demo
    demo_config.data.train_size = 500
    demo_config.data.val_size = 100
    
    trainer = HydraTrainer(demo_config)
    trainer.setup()
    best_acc = trainer.train()
    
    print(f"Demo training completed with best accuracy: {best_acc:.2f}%")
    
    print("\n5. Hydra Benefits")
    print("-" * 18)
    
    benefits = [
        "Hierarchical configuration with composition",
        "Command-line override of any configuration parameter",
        "Type safety with structured configurations",
        "Configuration validation and error checking",
        "Multiple configuration sources (YAML, Python, CLI)",
        "Configuration templates and inheritance",
        "Automatic working directory management",
        "Integration with hyperparameter optimization",
        "Configuration logging and reproducibility",
        "Plugin system for extensibility",
        "Multi-run support for experiments",
        "Configuration schema documentation"
    ]
    
    print("Hydra Configuration Benefits:")
    for i, benefit in enumerate(benefits, 1):
        print(f"{i:2d}. {benefit}")
    
    print("\n6. Best Practices")
    print("-" * 19)
    
    best_practices = [
        "Use structured configs with dataclasses for type safety",
        "Organize configurations hierarchically with groups",
        "Provide sensible defaults for all parameters",
        "Use config composition for modularity",
        "Document configuration parameters clearly",
        "Validate configurations at startup",
        "Use descriptive names for configuration files",
        "Version control your configuration files",
        "Use interpolation for computed values",
        "Implement configuration migration for updates",
        "Use defaults for common experiment setups",
        "Log final resolved configuration for reproducibility"
    ]
    
    print("Hydra Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Advanced Features")
    print("-" * 21)
    
    advanced_features = [
        "Multi-run experiments with parameter sweeps",
        "Remote job execution with different launchers",
        "Configuration composition and inheritance",
        "Custom resolvers for dynamic configuration",
        "Plugin architecture for extensibility",
        "Integration with optimization frameworks",
        "Configuration debugging and inspection",
        "Automatic working directory management",
        "Configuration packaging and distribution",
        "Runtime configuration modification"
    ]
    
    print("Hydra Advanced Features:")
    for feature in advanced_features:
        print(f"  - {feature}")
    
    print("\n8. Common Use Cases")
    print("-" * 20)
    
    use_cases = {
        "Hyperparameter Tuning": "Systematic exploration of parameter spaces",
        "Multi-Environment Setup": "Different configs for dev/staging/prod",
        "Ablation Studies": "Compare different model components",
        "Reproducible Research": "Version-controlled experiment configuration",
        "Team Collaboration": "Shared configuration standards",
        "A/B Testing": "Compare different model configurations",
        "Configuration Management": "Centralized configuration for complex systems",
        "Experiment Tracking": "Systematic organization of ML experiments"
    }
    
    print("Common Hydra Use Cases:")
    for use_case, description in use_cases.items():
        print(f"  {use_case}: {description}")
    
    print("\n9. Integration Patterns")
    print("-" * 24)
    
    integration_patterns = [
        "PyTorch Lightning: Use Hydra for trainer configuration",
        "Weights & Biases: Log Hydra configs to W&B experiments",
        "Ray Tune: Use Hydra configs in Ray training functions",
        "MLflow: Track configuration parameters as MLflow params",
        "Docker: Mount configuration directories as volumes",
        "Kubernetes: Use ConfigMaps for Hydra configurations",
        "CI/CD: Validate configurations in automated pipelines",
        "Jupyter: Load Hydra configs in notebook experiments"
    ]
    
    print("Hydra Integration Patterns:")
    for pattern in integration_patterns:
        print(f"  - {pattern}")
    
    print("\nHydra configuration management demonstration completed!")
    print("Key components covered:")
    print("  - Structured configuration with dataclasses")
    print("  - Configuration factories for models, optimizers, schedulers")
    print("  - Hierarchical configuration composition")
    print("  - Command-line configuration overrides")
    print("  - Configuration validation and type safety")
    print("  - Integration with PyTorch training workflows")
    
    print("\nHydra enables:")
    print("  - Clean separation of code and configuration")
    print("  - Reproducible and configurable experiments")
    print("  - Easy hyperparameter sweeps and ablation studies")
    print("  - Type-safe configuration management")
    print("  - Flexible experiment organization")