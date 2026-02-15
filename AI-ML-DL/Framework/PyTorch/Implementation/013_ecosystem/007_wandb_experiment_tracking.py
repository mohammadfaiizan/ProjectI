import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from datetime import datetime

# Note: Weights & Biases operations require the wandb package
# Install with: pip install wandb

try:
    import wandb
    from wandb.keras import WandbCallback  # For integration examples
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available. Install with: pip install wandb")

# Model Definitions
class ExperimentModel(nn.Module):
    """Neural network for experiment tracking demonstration"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Store architecture info
        self.architecture = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'num_classes': num_classes,
            'dropout': dropout,
            'total_params': self.count_parameters()
        }
    
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNNExperimentModel(nn.Module):
    """CNN model for computer vision experiments"""
    
    def __init__(self, num_classes: int = 10, num_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.num_channels = num_channels
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dataset Classes
class SyntheticDataset(Dataset):
    """Synthetic dataset for demonstration"""
    
    def __init__(self, size: int, input_dim: int, num_classes: int, noise_level: float = 0.1):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate data with some structure
        self.data = torch.randn(size, input_dim)
        
        # Create labels with some pattern
        weights = torch.randn(input_dim, num_classes)
        logits = self.data @ weights + noise_level * torch.randn(size, num_classes)
        self.labels = torch.argmax(logits, dim=1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SyntheticImageDataset(Dataset):
    """Synthetic image dataset"""
    
    def __init__(self, size: int, channels: int = 3, height: int = 32, width: int = 32, num_classes: int = 10):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width
        self.num_classes = num_classes
        
        # Generate synthetic images
        self.images = torch.randn(size, channels, height, width)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Weights & Biases Integration
class WandBTracker:
    """Comprehensive W&B experiment tracking"""
    
    def __init__(self, project_name: str = "pytorch-experiments", 
                 entity: Optional[str] = None, offline: bool = False):
        self.project_name = project_name
        self.entity = entity
        self.offline = offline
        self.run = None
        self.step = 0
    
    def start_experiment(self, config: Dict[str, Any], 
                        experiment_name: Optional[str] = None,
                        tags: List[str] = None, 
                        notes: str = None):
        """Start a new W&B experiment"""
        if not WANDB_AVAILABLE:
            print("W&B not available - tracking will be simulated")
            return
        
        # Initialize wandb run
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=config,
            name=experiment_name,
            tags=tags or [],
            notes=notes,
            mode="offline" if self.offline else "online"
        )
        
        print(f"✓ Started W&B experiment: {self.run.name if self.run else 'simulated'}")
        
        # Log system info
        if self.run:
            self.log_system_info()
    
    def log_system_info(self):
        """Log system and environment information"""
        if not self.run:
            return
        
        system_info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
        
        wandb.config.update({"system_info": system_info})
    
    def log_model_architecture(self, model: nn.Module):
        """Log model architecture information"""
        if not self.run:
            print("Model architecture logged (simulated)")
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model info
        model_info = {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        # Log architecture details if available
        if hasattr(model, 'architecture'):
            model_info.update(model.architecture)
        
        wandb.config.update({"model_info": model_info})
        
        # Log model summary as text
        model_summary = str(model)
        wandb.log({"model_summary": wandb.Html(f"<pre>{model_summary}</pre>")})
        
        print(f"✓ Logged model with {total_params:,} parameters")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log training metrics"""
        if step is None:
            step = self.step
            self.step += 1
        
        if not self.run:
            print(f"Step {step}: {metrics}")
            return
        
        wandb.log(metrics, step=step)
    
    def log_images(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None,
                   predictions: Optional[torch.Tensor] = None, 
                   caption: str = "Images", max_images: int = 25):
        """Log images to W&B"""
        if not self.run:
            print(f"Logged {len(images)} images (simulated)")
            return
        
        # Convert to numpy and ensure proper format
        if images.dim() == 4:  # Batch of images
            images_np = images[:max_images].cpu().numpy()
        else:
            images_np = images.unsqueeze(0).cpu().numpy()
        
        # Prepare captions
        captions = []
        for i in range(min(len(images_np), max_images)):
            caption_text = f"Image {i}"
            if labels is not None and i < len(labels):
                caption_text += f" | True: {labels[i].item()}"
            if predictions is not None and i < len(predictions):
                caption_text += f" | Pred: {predictions[i].item()}"
            captions.append(caption_text)
        
        # Log to W&B
        wandb_images = [
            wandb.Image(img, caption=cap) 
            for img, cap in zip(images_np, captions)
        ]
        
        wandb.log({caption: wandb_images})
        print(f"✓ Logged {len(wandb_images)} images")
    
    def log_histogram(self, tensor: torch.Tensor, name: str):
        """Log tensor histogram"""
        if not self.run:
            print(f"Histogram logged for {name} (simulated)")
            return
        
        wandb.log({f"{name}_histogram": wandb.Histogram(tensor.detach().cpu().numpy())})
    
    def log_gradients(self, model: nn.Module):
        """Log gradient histograms"""
        if not self.run:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                wandb.log({
                    f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()),
                    f"parameters/{name}": wandb.Histogram(param.detach().cpu().numpy())
                })
    
    def log_learning_rate(self, optimizer):
        """Log current learning rate"""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_metrics({f"learning_rate_group_{i}": lr})
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: Optional[List[str]] = None):
        """Log confusion matrix"""
        if not self.run:
            print("Confusion matrix logged (simulated)")
            return
        
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def log_custom_chart(self, data: Dict[str, List], title: str, x_label: str, y_label: str):
        """Log custom chart"""
        if not self.run:
            print(f"Custom chart '{title}' logged (simulated)")
            return
        
        # Create table for custom plotting
        table = wandb.Table(data=data, columns=list(data.keys()))
        
        wandb.log({
            title: wandb.plot.line(
                table, 
                x=x_label, 
                y=y_label, 
                title=title
            )
        })
    
    def log_model_artifact(self, model: nn.Module, name: str = "model"):
        """Log model as artifact"""
        if not self.run:
            print(f"Model artifact '{name}' logged (simulated)")
            return
        
        # Save model
        model_path = f"{name}.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create artifact
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        
        # Log artifact
        wandb.log_artifact(artifact)
        
        # Clean up
        os.remove(model_path)
        
        print(f"✓ Model saved as artifact: {name}")
    
    def finish_experiment(self):
        """Finish the W&B experiment"""
        if not self.run:
            print("Experiment finished (simulated)")
            return
        
        wandb.finish()
        print("✓ W&B experiment finished")

# Experiment Management
class ExperimentManager:
    """Manage multiple experiments with W&B"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiments = []
    
    def run_hyperparameter_sweep(self, sweep_config: Dict[str, Any], 
                                 train_function: callable, count: int = 5):
        """Run hyperparameter sweep"""
        if not WANDB_AVAILABLE:
            print("W&B sweep simulated")
            return
        
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        
        print(f"✓ Created sweep: {sweep_id}")
        
        # Run sweep
        wandb.agent(sweep_id, train_function, count=count)
        
        return sweep_id
    
    def compare_experiments(self, run_ids: List[str]):
        """Compare multiple experiments"""
        if not WANDB_AVAILABLE:
            print("Experiment comparison simulated")
            return
        
        # Get runs
        api = wandb.Api()
        runs = [api.run(f"{self.project_name}/{run_id}") for run_id in run_ids]
        
        # Compare metrics
        comparison_data = {}
        for run in runs:
            comparison_data[run.name] = run.summary
        
        print("Experiment comparison:")
        for name, metrics in comparison_data.items():
            print(f"  {name}: {metrics}")
        
        return comparison_data
    
    def create_report(self, run_ids: List[str], title: str = "Experiment Report"):
        """Create W&B report"""
        if not WANDB_AVAILABLE:
            print(f"Report '{title}' created (simulated)")
            return
        
        # This would create an actual W&B report
        print(f"✓ Report '{title}' created with {len(run_ids)} experiments")

# Training with W&B Integration
class WandBTrainer:
    """Trainer with comprehensive W&B integration"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, tracker: WandBTracker):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tracker = tracker
        
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
    
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Setup optimizer, criterion, and scheduler"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        print("✓ Training setup completed")
    
    def train_epoch(self, epoch: int):
        """Train for one epoch with W&B logging"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Log batch metrics
            if batch_idx % 50 == 0:
                batch_accuracy = 100. * correct / total
                self.tracker.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_accuracy': batch_accuracy,
                    'epoch': epoch
                })
                
                # Log learning rate
                self.tracker.log_learning_rate(self.optimizer)
        
        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total
        
        # Log confusion matrix every few epochs
        if epoch % 5 == 0:
            self.tracker.log_confusion_matrix(
                np.array(all_targets), 
                np.array(all_predictions)
            )
        
        return avg_val_loss, val_accuracy, all_predictions, all_targets
    
    def train(self, num_epochs: int = 50, log_gradients: bool = False,
              log_images_epoch: int = 10):
        """Complete training loop with W&B logging"""
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        best_val_accuracy = 0.0
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_accuracy = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            val_loss, val_accuracy, predictions, targets = self.validate_epoch(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Log epoch metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            
            self.tracker.log_metrics(metrics)
            
            # Log gradients
            if log_gradients and epoch % 10 == 0:
                self.tracker.log_gradients(self.model)
            
            # Log sample images
            if epoch % log_images_epoch == 0:
                sample_data, sample_targets = next(iter(self.val_loader))
                with torch.no_grad():
                    sample_outputs = self.model(sample_data[:8])
                    sample_predictions = torch.argmax(sample_outputs, dim=1)
                
                self.tracker.log_images(
                    sample_data[:8], 
                    sample_targets[:8], 
                    sample_predictions,
                    f"Validation_Samples_Epoch_{epoch}"
                )
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.tracker.log_model_artifact(self.model, f"best_model_epoch_{epoch}")
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                      f"Train Acc = {train_accuracy:.2f}%, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val Acc = {val_accuracy:.2f}%")
        
        # Log training curves
        self.tracker.log_custom_chart(
            {
                "epoch": list(range(num_epochs)),
                "train_loss": train_losses,
                "val_loss": val_losses
            },
            "Training Curves - Loss",
            "epoch",
            "loss"
        )
        
        self.tracker.log_custom_chart(
            {
                "epoch": list(range(num_epochs)),
                "train_accuracy": train_accuracies,
                "val_accuracy": val_accuracies
            },
            "Training Curves - Accuracy",
            "epoch",
            "accuracy"
        )
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1]
        }

# Hyperparameter Sweep Example
def create_sweep_configuration():
    """Create W&B sweep configuration"""
    
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'hidden_size': {
                'values': [64, 128, 256, 512]
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.5
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            'eta': 2
        }
    }
    
    return sweep_config

def sweep_train_function():
    """Training function for W&B sweep"""
    if not WANDB_AVAILABLE:
        print("Sweep training function (simulated)")
        return
    
    # Initialize W&B run
    run = wandb.init()
    config = wandb.config
    
    # Create model with sweep parameters
    model = ExperimentModel(
        input_size=50,
        hidden_sizes=[config.hidden_size, config.hidden_size // 2],
        num_classes=5,
        dropout=config.dropout
    )
    
    # Create datasets
    train_dataset = SyntheticDataset(1000, 50, 5)
    val_dataset = SyntheticDataset(200, 50, 5)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop (simplified)
    for epoch in range(20):  # Reduced for sweep
        model.train()
        train_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100. * correct / total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy
        })

if __name__ == "__main__":
    print("Weights & Biases Experiment Tracking")
    print("=" * 40)
    
    if not WANDB_AVAILABLE:
        print("W&B not available. Install with: pip install wandb")
        print("Showing simulated tracking examples...")
    
    print("\n1. Basic Experiment Setup")
    print("-" * 28)
    
    # Configuration
    config = {
        "model_type": "MLP",
        "input_size": 50,
        "hidden_sizes": [128, 64],
        "num_classes": 5,
        "learning_rate": 0.001,
        "batch_size": 32,
        "num_epochs": 30,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau"
    }
    
    # Start experiment tracking
    tracker = WandBTracker(
        project_name="pytorch-ecosystem-demo",
        offline=True  # Set to False for online tracking
    )
    
    tracker.start_experiment(
        config=config,
        experiment_name="basic_experiment_demo",
        tags=["demo", "mlp", "synthetic"],
        notes="Demonstration of W&B integration with PyTorch"
    )
    
    print("\n2. Model Creation and Logging")
    print("-" * 33)
    
    # Create model
    model = ExperimentModel(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    )
    
    # Log model architecture
    tracker.log_model_architecture(model)
    
    print("\n3. Dataset Creation")
    print("-" * 21)
    
    # Create datasets
    train_dataset = SyntheticDataset(size=2000, input_dim=50, num_classes=5)
    val_dataset = SyntheticDataset(size=500, input_dim=50, num_classes=5)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print(f"✓ Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    
    print("\n4. Training with W&B Tracking")
    print("-" * 32)
    
    # Create trainer
    trainer = WandBTrainer(model, train_loader, val_loader, tracker)
    
    # Setup training
    trainer.setup_training(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Run training
    results = trainer.train(
        num_epochs=15,  # Reduced for demo
        log_gradients=True,
        log_images_epoch=5
    )
    
    print(f"Training results: {results}")
    
    print("\n5. Advanced Logging Features")
    print("-" * 31)
    
    # Log model weights histogram
    for name, param in model.named_parameters():
        if 'weight' in name:
            tracker.log_histogram(param, f"weights/{name}")
    
    # Log custom metrics
    tracker.log_metrics({
        "model_complexity": model.count_parameters(),
        "final_performance": results['best_val_accuracy'],
        "training_efficiency": results['best_val_accuracy'] / 15  # accuracy per epoch
    })
    
    print("✓ Advanced metrics logged")
    
    print("\n6. Hyperparameter Sweep Configuration")
    print("-" * 40)
    
    sweep_config = create_sweep_configuration()
    
    print("Sweep configuration:")
    print(json.dumps(sweep_config, indent=2))
    
    # Demonstrate sweep (commented out for demo)
    if WANDB_AVAILABLE:
        print("\n# To run sweep, uncomment and execute:")
        print("# experiment_manager = ExperimentManager('pytorch-ecosystem-demo')")
        print("# sweep_id = experiment_manager.run_hyperparameter_sweep(sweep_config, sweep_train_function, count=5)")
    
    print("\n7. W&B Best Practices")
    print("-" * 24)
    
    best_practices = [
        "Log hyperparameters at the start of training",
        "Track both training and validation metrics",
        "Use meaningful experiment names and tags",
        "Log model architecture and complexity",
        "Save model artifacts for best performing runs",
        "Log sample predictions and images",
        "Track gradient and weight histograms",
        "Use early stopping and learning rate scheduling",
        "Create comparison reports for multiple experiments",
        "Document experiment notes and observations",
        "Use sweeps for systematic hyperparameter optimization",
        "Monitor system metrics (GPU usage, memory, etc.)",
        "Version control your code and log git commits",
        "Use W&B Tables for structured data logging"
    ]
    
    print("W&B Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Integration Patterns")
    print("-" * 24)
    
    integration_patterns = {
        "PyTorch Lightning": "Use WandbLogger for automatic logging",
        "Hugging Face": "Built-in W&B integration for transformers",
        "FastAI": "WandbCallback for automatic tracking",
        "Scikit-learn": "Manual logging of model metrics and parameters",
        "Ray Tune": "W&B integration for hyperparameter optimization",
        "Optuna": "Log optimization trials and best parameters",
        "Docker": "Log container information and environment",
        "Kubernetes": "Track distributed training across pods"
    }
    
    print("W&B Integration Patterns:")
    for framework, integration in integration_patterns.items():
        print(f"  {framework}: {integration}")
    
    print("\n9. Advanced Features")
    print("-" * 21)
    
    advanced_features = [
        "Artifacts: Version datasets, models, and other files",
        "Sweeps: Automated hyperparameter optimization",
        "Reports: Create publication-ready experiment reports",
        "Tables: Log structured data and create custom charts",
        "Alerts: Get notified when experiments finish or fail",
        "Model Registry: Track model versions and deployment",
        "Launch: Execute experiments in cloud environments",
        "Collaborative workspaces for team projects",
        "Custom metrics and visualizations",
        "Integration with MLOps pipelines",
        "A/B testing and model comparison",
        "Automated model evaluation and testing"
    ]
    
    print("W&B Advanced Features:")
    for feature in advanced_features:
        print(f"  - {feature}")
    
    print("\n10. Common Use Cases")
    print("-" * 21)
    
    use_cases = [
        "Research experiment tracking and comparison",
        "Hyperparameter optimization for model tuning",
        "Model versioning and artifact management",
        "Team collaboration on ML projects",
        "Reproducible experiment workflows",
        "Model performance monitoring",
        "A/B testing different model architectures",
        "Dataset versioning and lineage tracking",
        "Automated model evaluation pipelines",
        "Production model monitoring and alerts"
    ]
    
    print("Common W&B Use Cases:")
    for i, use_case in enumerate(use_cases, 1):
        print(f"{i:2d}. {use_case}")
    
    # Finish experiment
    tracker.finish_experiment()
    
    print("\nWeights & Biases experiment tracking demonstration completed!")
    print("Key components covered:")
    print("  - Comprehensive experiment tracking and logging")
    print("  - Model architecture and performance monitoring")
    print("  - Advanced visualization and custom charts")
    print("  - Hyperparameter sweep configuration")
    print("  - Model artifact management")
    print("  - Integration patterns and best practices")
    
    print("\nW&B enables:")
    print("  - Effortless experiment tracking and comparison")
    print("  - Collaborative ML research and development")
    print("  - Reproducible and well-documented experiments")
    print("  - Automated hyperparameter optimization")
    print("  - Model versioning and deployment workflows")