import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
import time

# Note: Ray operations require the ray package
# Install with: pip install ray[tune] ray[train]

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.train import Trainer as RayTrainer
    from ray.train.torch import TorchTrainer
    from ray.train.torch import TorchConfig
    from ray.air import session
    from ray.air.checkpoint import Checkpoint
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Install with: pip install ray[tune] ray[train]")

# Model Definitions
class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class CNNModel(nn.Module):
    """CNN model for image classification"""
    
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Dataset Classes
class SyntheticDataset(Dataset):
    """Synthetic dataset for testing"""
    
    def __init__(self, size: int, input_dim: int, num_classes: int, task_type: str = 'classification'):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_type = task_type
        
        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        
        if task_type == 'classification':
            # Create labels with some pattern
            self.labels = torch.randint(0, num_classes, (size,))
        else:  # regression
            self.labels = torch.randn(size, 1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SyntheticImageDataset(Dataset):
    """Synthetic image dataset"""
    
    def __init__(self, size: int, image_size: Tuple[int, int] = (28, 28), 
                 num_channels: int = 1, num_classes: int = 10):
        self.size = size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        # Generate synthetic images
        self.images = torch.randn(size, num_channels, *image_size)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Training Functions for Ray
def train_model_simple(config: Dict[str, Any]) -> None:
    """Simple training function for Ray Tune"""
    
    # Extract hyperparameters
    lr = config["lr"]
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    
    # Create model
    model = SimpleModel(input_size=20, hidden_size=hidden_size, num_classes=3)
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(size=1000, input_dim=20, num_classes=3)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):  # Short training for demo
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        train_acc /= len(train_dataset)
        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)
        
        # Report to Ray Tune
        if RAY_AVAILABLE:
            session.report({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch": epoch
            })

def train_model_with_checkpoints(config: Dict[str, Any]) -> None:
    """Training function with checkpointing support"""
    
    # Model and data setup
    lr = config["lr"]
    batch_size = config["batch_size"]
    hidden_size = config["hidden_size"]
    
    model = SimpleModel(input_size=20, hidden_size=hidden_size, num_classes=3)
    
    # Load checkpoint if available
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_dict = checkpoint.to_dict()
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        start_epoch = checkpoint_dict["epoch"] + 1
    else:
        start_epoch = 0
    
    # Setup data and training
    dataset = SyntheticDataset(size=1000, input_dim=20, num_classes=3)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(start_epoch, 20):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_acc += pred.eq(target.view_as(pred)).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_dataset)
        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)
        
        # Create checkpoint
        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }
        
        # Report to Ray Tune with checkpoint
        if RAY_AVAILABLE:
            session.report(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch
                },
                checkpoint=Checkpoint.from_dict(checkpoint_dict)
            )

# Distributed Training with Ray Train
if RAY_AVAILABLE:
    def train_func_distributed(config: Dict[str, Any]) -> None:
        """Distributed training function for Ray Train"""
        
        # Get distributed training context
        rank = session.get_world_rank()
        world_size = session.get_world_size()
        
        print(f"Worker {rank}/{world_size} starting training")
        
        # Model setup
        model = SimpleModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_classes=config["num_classes"]
        )
        
        # Wrap model for distributed training
        model = session.prepare_model(model)
        
        # Dataset setup
        dataset = SyntheticDataset(
            size=config["dataset_size"],
            input_dim=config["input_size"],
            num_classes=config["num_classes"]
        )
        
        # Split dataset for each worker
        indices = list(range(len(dataset)))
        worker_indices = indices[rank::world_size]
        worker_dataset = torch.utils.data.Subset(dataset, worker_indices)
        
        # DataLoader
        train_loader = DataLoader(
            worker_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )
        train_loader = session.prepare_data_loader(train_loader)
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(config["num_epochs"]):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                epoch_acc += pred.eq(target.view_as(pred)).sum().item()
                num_batches += 1
            
            # Calculate averages
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            avg_acc = epoch_acc / len(worker_dataset) if len(worker_dataset) > 0 else 0
            
            # Report metrics (only from rank 0)
            if rank == 0:
                session.report({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc
                })
            
            print(f"Worker {rank} - Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.4f}")

# Ray Tune Hyperparameter Optimization
class RayTuneOptimizer:
    """Ray Tune hyperparameter optimization wrapper"""
    
    def __init__(self):
        self.results = None
        self.best_config = None
    
    def run_basic_search(self, num_samples: int = 10):
        """Run basic hyperparameter search"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Define search space
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "hidden_size": tune.choice([32, 64, 128, 256])
        }
        
        # Define scheduler (early stopping)
        scheduler = ASHAScheduler(
            metric="val_accuracy",
            mode="max",
            max_t=10,
            grace_period=3,
            reduction_factor=2
        )
        
        # Reporter for progress
        reporter = CLIReporter(
            metric_columns=["train_loss", "val_loss", "train_accuracy", "val_accuracy", "training_iteration"]
        )
        
        # Run tuning
        result = tune.run(
            train_model_simple,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir="./ray_results"
        )
        
        self.results = result
        self.best_config = result.best_config
        
        print(f"✓ Best config: {self.best_config}")
        print(f"✓ Best validation accuracy: {result.best_result['val_accuracy']:.4f}")
        
        return result
    
    def run_advanced_search(self, num_samples: int = 20):
        """Run advanced hyperparameter search with Hyperopt"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        try:
            from hyperopt import hp
            
            # Advanced search space
            config = {
                "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-1)),
                "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
                "hidden_size": hp.choice("hidden_size", [32, 64, 128, 256, 512]),
                "dropout": hp.uniform("dropout", 0.1, 0.5)
            }
            
            # Search algorithm
            search_alg = HyperOptSearch(
                metric="val_accuracy",
                mode="max",
                n_initial_points=5
            )
            
            # Scheduler
            scheduler = ASHAScheduler(
                metric="val_accuracy",
                mode="max",
                max_t=15,
                grace_period=5
            )
            
            # Run search
            result = tune.run(
                train_model_simple,
                config=config,
                num_samples=num_samples,
                search_alg=search_alg,
                scheduler=scheduler,
                local_dir="./ray_results_advanced"
            )
            
            self.results = result
            self.best_config = result.best_config
            
            print(f"✓ Advanced search completed")
            print(f"✓ Best config: {self.best_config}")
            
            return result
            
        except ImportError:
            print("Hyperopt not available, falling back to basic search")
            return self.run_basic_search(num_samples)
    
    def run_population_based_training(self, num_samples: int = 8):
        """Run Population Based Training"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        # PBT scheduler
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="val_accuracy",
            mode="max",
            perturbation_interval=5,
            hyperparam_mutations={
                "lr": [1e-4, 1e-3, 1e-2],
                "batch_size": [16, 32, 64]
            }
        )
        
        # Initial config
        config = {
            "lr": tune.choice([1e-4, 1e-3, 1e-2]),
            "batch_size": tune.choice([16, 32, 64]),
            "hidden_size": 128
        }
        
        # Run PBT
        result = tune.run(
            train_model_with_checkpoints,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            stop={"training_iteration": 20},
            local_dir="./ray_results_pbt"
        )
        
        print(f"✓ Population Based Training completed")
        print(f"✓ Best config: {result.best_config}")
        
        return result

# Distributed Training Manager
class RayDistributedTrainer:
    """Manager for distributed training with Ray Train"""
    
    def __init__(self):
        self.trainer = None
        self.results = None
    
    def setup_distributed_training(self, num_workers: int = 2):
        """Setup distributed training"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Training configuration
        train_config = {
            "input_size": 20,
            "hidden_size": 128,
            "num_classes": 3,
            "dataset_size": 2000,
            "batch_size": 32,
            "lr": 0.001,
            "num_epochs": 10
        }
        
        # Create Ray TorchTrainer
        self.trainer = TorchTrainer(
            train_loop_per_worker=train_func_distributed,
            train_loop_config=train_config,
            scaling_config={
                "num_workers": num_workers,
                "use_gpu": False  # Set to True if GPUs available
            }
        )
        
        print(f"✓ Setup distributed training with {num_workers} workers")
    
    def run_distributed_training(self):
        """Run distributed training"""
        if not RAY_AVAILABLE or self.trainer is None:
            print("Ray or trainer not available")
            return
        
        print("🚀 Starting distributed training...")
        
        # Run training
        self.results = self.trainer.fit()
        
        print("✅ Distributed training completed!")
        
        # Print results
        if self.results:
            print(f"Final metrics: {self.results.metrics}")
        
        return self.results

# Ray Cluster Management
class RayClusterManager:
    """Utilities for Ray cluster management"""
    
    @staticmethod
    def initialize_ray(address: Optional[str] = None, **kwargs):
        """Initialize Ray cluster"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        if ray.is_initialized():
            ray.shutdown()
        
        if address:
            # Connect to existing cluster
            ray.init(address=address, **kwargs)
            print(f"✓ Connected to Ray cluster at {address}")
        else:
            # Start local cluster
            ray.init(**kwargs)
            print("✓ Started local Ray cluster")
        
        # Print cluster info
        print(f"Ray cluster nodes: {len(ray.nodes())}")
        print(f"Available resources: {ray.cluster_resources()}")
    
    @staticmethod
    def get_cluster_info():
        """Get Ray cluster information"""
        if not RAY_AVAILABLE or not ray.is_initialized():
            return {}
        
        info = {
            "nodes": len(ray.nodes()),
            "resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
            "cluster_info": ray.cluster_resources()
        }
        
        return info
    
    @staticmethod
    def shutdown_ray():
        """Shutdown Ray cluster"""
        if RAY_AVAILABLE and ray.is_initialized():
            ray.shutdown()
            print("✓ Ray cluster shutdown")

# Example Applications
class RayExamples:
    """Example applications using Ray for ML"""
    
    @staticmethod
    def neural_architecture_search():
        """Example of Neural Architecture Search with Ray Tune"""
        if not RAY_AVAILABLE:
            print("Ray not available")
            return
        
        def train_architecture(config):
            # Variable architecture based on config
            layers = []
            input_size = 20
            
            for i, layer_size in enumerate(config["layers"]):
                layers.extend([
                    nn.Linear(input_size if i == 0 else config["layers"][i-1], layer_size),
                    nn.ReLU(),
                    nn.Dropout(config["dropout"])
                ])
                
            layers.append(nn.Linear(config["layers"][-1], 3))
            
            model = nn.Sequential(*layers)
            
            # Training code (simplified)
            dataset = SyntheticDataset(1000, 20, 3)
            loader = DataLoader(dataset, batch_size=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(5):
                total_loss = 0
                for data, target in loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                session.report({"loss": total_loss / len(loader), "epoch": epoch})
        
        # Architecture search space
        config = {
            "layers": tune.choice([
                [64, 32],
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128, 64]
            ]),
            "dropout": tune.uniform(0.1, 0.5),
            "lr": tune.loguniform(1e-4, 1e-2)
        }
        
        # Run search
        result = tune.run(
            train_architecture,
            config=config,
            num_samples=10,
            metric="loss",
            mode="min"
        )
        
        print(f"✓ Best architecture: {result.best_config}")
        
        return result
    
    @staticmethod
    def multi_objective_optimization():
        """Example of multi-objective optimization"""
        if not RAY_AVAILABLE:
            return
        
        def train_with_multiple_objectives(config):
            # Train model and return multiple metrics
            model = SimpleModel(20, config["hidden_size"], 3)
            
            # Simplified training
            dataset = SyntheticDataset(500, 20, 3)
            loader = DataLoader(dataset, batch_size=config["batch_size"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            
            total_params = sum(p.numel() for p in model.parameters())
            
            for epoch in range(3):
                total_loss = 0
                for data, target in loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                # Report multiple objectives
                session.report({
                    "accuracy": 0.8 - total_loss / len(loader),  # Mock accuracy
                    "model_size": total_params,
                    "epoch": epoch
                })
        
        config = {
            "hidden_size": tune.choice([32, 64, 128, 256]),
            "batch_size": tune.choice([16, 32, 64]),
            "lr": tune.loguniform(1e-4, 1e-2)
        }
        
        # Search for Pareto optimal solutions
        result = tune.run(
            train_with_multiple_objectives,
            config=config,
            num_samples=15,
            metric="accuracy",
            mode="max"
        )
        
        print("✓ Multi-objective optimization completed")
        return result

if __name__ == "__main__":
    print("Ray Distributed Training")
    print("=" * 27)
    
    if not RAY_AVAILABLE:
        print("Ray not available. Install with: pip install ray[tune] ray[train]")
        print("Showing conceptual examples...")
    
    print("\n1. Ray Cluster Setup")
    print("-" * 21)
    
    cluster_manager = RayClusterManager()
    
    # Initialize Ray
    cluster_manager.initialize_ray(num_cpus=4, num_gpus=0)
    
    # Get cluster info
    if RAY_AVAILABLE:
        cluster_info = cluster_manager.get_cluster_info()
        print(f"Cluster nodes: {cluster_info.get('nodes', 'N/A')}")
        print(f"Available CPUs: {cluster_info.get('resources', {}).get('CPU', 'N/A')}")
    
    print("\n2. Hyperparameter Optimization")
    print("-" * 33)
    
    optimizer = RayTuneOptimizer()
    
    # Run basic search
    if RAY_AVAILABLE:
        print("Running basic hyperparameter search...")
        basic_result = optimizer.run_basic_search(num_samples=5)
        
        if basic_result:
            print(f"✓ Best configuration found:")
            for key, value in optimizer.best_config.items():
                print(f"  {key}: {value}")
    
    print("\n3. Advanced Optimization")
    print("-" * 25)
    
    if RAY_AVAILABLE:
        print("Running advanced search with early stopping...")
        # Note: In practice, you'd run more samples
        advanced_result = optimizer.run_advanced_search(num_samples=3)
    
    print("\n4. Population Based Training")
    print("-" * 31)
    
    if RAY_AVAILABLE:
        print("Running Population Based Training...")
        pbt_result = optimizer.run_population_based_training(num_samples=4)
    
    print("\n5. Distributed Training")
    print("-" * 24)
    
    distributed_trainer = RayDistributedTrainer()
    
    # Setup and run distributed training
    distributed_trainer.setup_distributed_training(num_workers=2)
    
    if RAY_AVAILABLE:
        print("Running distributed training...")
        distributed_results = distributed_trainer.run_distributed_training()
    
    print("\n6. Advanced Examples")
    print("-" * 20)
    
    examples = RayExamples()
    
    if RAY_AVAILABLE:
        print("Neural Architecture Search example...")
        nas_result = examples.neural_architecture_search()
        
        print("Multi-objective optimization example...")
        mo_result = examples.multi_objective_optimization()
    
    print("\n7. Ray Benefits for ML")
    print("-" * 22)
    
    benefits = [
        "Scalable hyperparameter tuning with intelligent search algorithms",
        "Distributed training across multiple machines and GPUs",
        "Fault-tolerant execution with automatic recovery",
        "Efficient resource utilization and scheduling",
        "Integration with popular ML frameworks",
        "Advanced scheduling algorithms (ASHA, PBT, etc.)",
        "Built-in experiment tracking and visualization",
        "Easy scaling from laptop to cluster",
        "Support for various search algorithms (Hyperopt, Optuna, etc.)",
        "Automatic checkpointing and resume capabilities",
        "Multi-objective optimization support",
        "Real-time monitoring and progress reporting"
    ]
    
    print("Ray Benefits for Machine Learning:")
    for i, benefit in enumerate(benefits, 1):
        print(f"{i:2d}. {benefit}")
    
    print("\n8. Best Practices")
    print("-" * 19)
    
    best_practices = [
        "Use Ray Tune for hyperparameter optimization",
        "Implement proper checkpointing in training functions",
        "Use schedulers for early stopping of poor trials",
        "Monitor resource usage and adjust accordingly",
        "Use Ray Train for multi-node distributed training",
        "Implement proper error handling in training functions",
        "Use appropriate search algorithms for your problem",
        "Profile your training code before scaling",
        "Use Ray's built-in logging and monitoring",
        "Consider fault tolerance in long-running experiments",
        "Use proper data loading strategies for distributed training",
        "Optimize communication overhead in distributed setups"
    ]
    
    print("Ray Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. Common Use Cases")
    print("-" * 20)
    
    use_cases = {
        "Hyperparameter Tuning": "Optimize model hyperparameters efficiently",
        "Neural Architecture Search": "Automatically discover optimal architectures",
        "Distributed Training": "Scale training across multiple GPUs/nodes",
        "Model Selection": "Compare different model architectures",
        "Data Parallel Training": "Train with large datasets across workers",
        "Reinforcement Learning": "Parallel environment simulation",
        "AutoML Pipelines": "End-to-end automated machine learning",
        "A/B Testing": "Compare different training strategies",
        "Resource Optimization": "Optimize compute resource allocation",
        "Multi-Task Learning": "Train multiple related tasks simultaneously"
    }
    
    print("Common Ray Use Cases:")
    for use_case, description in use_cases.items():
        print(f"  {use_case}: {description}")
    
    print("\n10. Integration Ecosystem")
    print("-" * 26)
    
    integrations = [
        "PyTorch: Native integration with Ray Train",
        "TensorFlow: Distributed training support",
        "Scikit-learn: Parallel model selection",
        "XGBoost: Distributed gradient boosting",
        "Horovod: High-performance distributed training",
        "MLflow: Experiment tracking integration",
        "Weights & Biases: Advanced logging and visualization",
        "Kubernetes: Cloud-native deployment",
        "AWS/GCP/Azure: Cloud platform integration",
        "Optuna: Advanced hyperparameter optimization"
    ]
    
    print("Ray Integration Ecosystem:")
    for integration in integrations:
        print(f"  - {integration}")
    
    # Cleanup
    print("\n11. Cleanup")
    print("-" * 13)
    
    cluster_manager.shutdown_ray()
    
    print("\nRay distributed training demonstration completed!")
    print("Key components covered:")
    print("  - Ray cluster setup and management")
    print("  - Hyperparameter optimization with Ray Tune")
    print("  - Advanced scheduling algorithms (ASHA, PBT)")
    print("  - Distributed training with Ray Train")
    print("  - Multi-objective optimization")
    print("  - Neural architecture search")
    print("  - Best practices and integration patterns")
    
    print("\nRay enables:")
    print("  - Effortless scaling from single machine to cluster")
    print("  - Intelligent hyperparameter search algorithms")
    print("  - Fault-tolerant distributed training")
    print("  - Efficient resource utilization")
    print("  - Advanced ML experimentation workflows")