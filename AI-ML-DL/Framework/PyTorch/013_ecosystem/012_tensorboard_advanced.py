import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import time
from datetime import datetime
from pathlib import Path

# Note: TensorBoard operations require the tensorboard package
# Install with: pip install tensorboard

try:
    from torch.utils.tensorboard import SummaryWriter
    import tensorboard
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# Advanced TensorBoard Integration for PyTorch
class AdvancedTensorBoard:
    """Advanced TensorBoard logging and visualization"""
    
    def __init__(self, log_dir: str = "./tensorboard_logs", 
                 experiment_name: str = "pytorch_experiment"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.writer = None
        self.global_step = 0
        
        if TENSORBOARD_AVAILABLE:
            self._setup_writer()
    
    def _setup_writer(self):
        """Setup TensorBoard writer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.log_dir / f"{self.experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(run_dir))
        print(f"✓ TensorBoard writer created: {run_dir}")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar values"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Scalar logged (simulated): {tag} = {value}")
            return
        
        step = step or self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], 
                   step: Optional[int] = None):
        """Log multiple scalars in one plot"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Scalars logged (simulated): {tag} = {scalar_dict}")
            return
        
        step = step or self.global_step
        self.writer.add_scalars(tag, scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, 
                     step: Optional[int] = None):
        """Log histogram of tensor values"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Histogram logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, 
                 step: Optional[int] = None, dataformats: str = 'CHW'):
        """Log single image"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Image logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_image(tag, image, step, dataformats=dataformats)
    
    def log_images(self, tag: str, images: torch.Tensor, 
                  step: Optional[int] = None, dataformats: str = 'NCHW'):
        """Log grid of images"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Images logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_images(tag, images, step, dataformats=dataformats)
    
    def log_figure(self, tag: str, figure: plt.Figure, 
                  step: Optional[int] = None):
        """Log matplotlib figure"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Figure logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_figure(tag, figure, step)
    
    def log_graph(self, model: nn.Module, input_sample: torch.Tensor):
        """Log model graph"""
        if not TENSORBOARD_AVAILABLE:
            print("Model graph logged (simulated)")
            return
        
        self.writer.add_graph(model, input_sample)
        print("✓ Model graph logged to TensorBoard")
    
    def log_embedding(self, features: torch.Tensor, 
                     metadata: Optional[List[str]] = None,
                     label_img: Optional[torch.Tensor] = None,
                     tag: str = "embeddings"):
        """Log high-dimensional embeddings"""
        if not TENSORBOARD_AVAILABLE:
            print("Embeddings logged (simulated)")
            return
        
        self.writer.add_embedding(
            features, 
            metadata=metadata, 
            label_img=label_img, 
            tag=tag
        )
        print(f"✓ Embeddings logged: {features.shape}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], 
                           metrics: Dict[str, float]):
        """Log hyperparameters and metrics"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Hyperparameters logged (simulated): {hparams}")
            return
        
        self.writer.add_hparams(hparams, metrics)
        print("✓ Hyperparameters logged")
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text data"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Text logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_text(tag, text, step)
    
    def log_pr_curve(self, tag: str, labels: torch.Tensor, 
                    predictions: torch.Tensor, step: Optional[int] = None):
        """Log precision-recall curve"""
        if not TENSORBOARD_AVAILABLE:
            print(f"PR curve logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_pr_curve(tag, labels, predictions, step)
    
    def log_audio(self, tag: str, audio: torch.Tensor, 
                 sample_rate: int = 22050, step: Optional[int] = None):
        """Log audio data"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Audio logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_audio(tag, audio, step, sample_rate)
    
    def log_video(self, tag: str, video: torch.Tensor, 
                 fps: int = 4, step: Optional[int] = None):
        """Log video data"""
        if not TENSORBOARD_AVAILABLE:
            print(f"Video logged (simulated): {tag}")
            return
        
        step = step or self.global_step
        self.writer.add_video(tag, video, step, fps)
    
    def flush(self):
        """Flush pending events"""
        if TENSORBOARD_AVAILABLE and self.writer:
            self.writer.flush()
    
    def close(self):
        """Close the writer"""
        if TENSORBOARD_AVAILABLE and self.writer:
            self.writer.close()
            print("✓ TensorBoard writer closed")

# Model with TensorBoard Integration
class TensorBoardModel(nn.Module):
    """Model with built-in TensorBoard logging"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], 
                 num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # Build layers
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Store layer names for logging
        self.layer_names = [f"hidden_{i}" for i in range(len(hidden_sizes))] + ["output"]
    
    def forward(self, x, log_activations: bool = False, 
               tensorboard: Optional[AdvancedTensorBoard] = None):
        """Forward pass with optional activation logging"""
        activations = {}
        
        current_input = x
        
        # Hidden layers
        for i, layer in enumerate(self.layers):
            current_input = F.relu(layer(current_input))
            current_input = self.dropout(current_input)
            
            if log_activations and tensorboard:
                layer_name = self.layer_names[i]
                activations[layer_name] = current_input.clone()
                tensorboard.log_histogram(f"activations/{layer_name}", current_input)
        
        # Output layer
        output = self.output_layer(current_input)
        
        if log_activations and tensorboard:
            activations["output"] = output.clone()
            tensorboard.log_histogram("activations/output", output)
        
        return output, activations if log_activations else output

# Custom Dataset for TensorBoard Demo
class TensorBoardDataset(Dataset):
    """Dataset with TensorBoard visualization support"""
    
    def __init__(self, size: int, input_dim: int, num_classes: int):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Create class names for visualization
        self.class_names = [f"Class_{i}" for i in range(num_classes)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def get_sample_batch(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample batch for visualization"""
        indices = torch.randperm(self.size)[:batch_size]
        return self.data[indices], self.labels[indices]

# TensorBoard Training Manager
class TensorBoardTrainer:
    """Training with comprehensive TensorBoard logging"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, tensorboard: AdvancedTensorBoard):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tensorboard = tensorboard
        
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
    
    def setup_training(self, learning_rate: float = 0.001, 
                      weight_decay: float = 1e-4):
        """Setup training components and log to TensorBoard"""
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Log model graph
        sample_input = next(iter(self.train_loader))[0][:1]
        self.tensorboard.log_graph(self.model, sample_input)
        
        # Log hyperparameters
        hparams = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau"
        }
        
        # We'll update metrics after training
        self._initial_hparams = hparams
        
        print("✓ Training setup completed with TensorBoard logging")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train one epoch with detailed TensorBoard logging"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass with activation logging (every 100 batches)
            log_activations = (batch_idx % 100 == 0)
            
            if isinstance(self.model, TensorBoardModel):
                output, activations = self.model(
                    data, 
                    log_activations=log_activations,
                    tensorboard=self.tensorboard if log_activations else None
                )
            else:
                output = self.model(data)
            
            loss = self.criterion(output, target)
            loss.backward()
            
            # Log gradient norms
            if batch_idx % 50 == 0:
                self._log_gradient_norms()
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Log batch metrics
            if batch_idx % 20 == 0:
                self.tensorboard.log_scalar(
                    "batch/train_loss", 
                    loss.item(), 
                    self.global_step
                )
                
                self.tensorboard.log_scalar(
                    "batch/learning_rate",
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )
            
            self.global_step += 1
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """Validate with TensorBoard logging"""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                if isinstance(output, tuple):
                    output = output[0]  # Handle TensorBoardModel output
                
                val_loss += self.criterion(output, target).item()
                
                # Get predictions and probabilities
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu())
                all_targets.extend(target.cpu())
                all_probabilities.extend(probabilities.cpu())
        
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = correct / total
        
        # Convert to tensors for TensorBoard
        predictions_tensor = torch.stack(all_predictions)
        targets_tensor = torch.stack(all_targets)
        probabilities_tensor = torch.stack(all_probabilities)
        
        return avg_val_loss, val_accuracy, predictions_tensor, probabilities_tensor
    
    def _log_gradient_norms(self):
        """Log gradient norms for each layer"""
        total_norm = 0.0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Log individual layer gradients
                self.tensorboard.log_scalar(
                    f"gradients/{name}", 
                    param_norm.item(), 
                    self.global_step
                )
        
        total_norm = total_norm ** (1. / 2)
        self.tensorboard.log_scalar(
            "gradients/total_norm", 
            total_norm, 
            self.global_step
        )
    
    def _log_weight_histograms(self):
        """Log weight histograms for each layer"""
        for name, param in self.model.named_parameters():
            self.tensorboard.log_histogram(
                f"weights/{name}", 
                param, 
                self.epoch
            )
    
    def _log_confusion_matrix_figure(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor):
        """Create and log confusion matrix figure"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Convert to numpy
        pred_np = predictions.numpy()
        target_np = targets.numpy()
        
        # Compute confusion matrix
        cm = confusion_matrix(target_np, pred_np)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - Epoch {self.epoch}')
        
        # Log figure
        self.tensorboard.log_figure("validation/confusion_matrix", fig, self.epoch)
        plt.close(fig)
    
    def _log_class_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Log per-class accuracy"""
        num_classes = len(torch.unique(targets))
        
        for class_idx in range(num_classes):
            class_mask = (targets == class_idx)
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_accuracy = (class_predictions == class_idx).float().mean()
                
                self.tensorboard.log_scalar(
                    f"validation/class_{class_idx}_accuracy",
                    class_accuracy.item(),
                    self.epoch
                )
    
    def _log_sample_predictions(self, data: torch.Tensor, targets: torch.Tensor,
                               predictions: torch.Tensor, num_samples: int = 8):
        """Log sample predictions for image data"""
        
        # Only log if data looks like images (3D or 4D)
        if len(data.shape) >= 3:
            # Select random samples
            indices = torch.randperm(len(data))[:num_samples]
            sample_data = data[indices]
            sample_targets = targets[indices]
            sample_predictions = predictions[indices]
            
            # Create figure with predictions
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            axes = axes.ravel()
            
            for i in range(min(num_samples, 8)):
                if len(sample_data[i].shape) == 3 and sample_data[i].shape[0] in [1, 3]:
                    # Image data
                    img = sample_data[i].permute(1, 2, 0) if sample_data[i].shape[0] == 3 else sample_data[i][0]
                    axes[i].imshow(img, cmap='gray' if sample_data[i].shape[0] == 1 else None)
                else:
                    # Non-image data - show as heatmap
                    axes[i].imshow(sample_data[i].reshape(28, 28), cmap='gray')  # Assume 28x28 for demo
                
                axes[i].set_title(f'True: {sample_targets[i].item()}, Pred: {sample_predictions[i].item()}')
                axes[i].axis('off')
            
            # Log figure
            self.tensorboard.log_figure("validation/sample_predictions", fig, self.epoch)
            plt.close(fig)
    
    def train(self, num_epochs: int = 50, log_embeddings_every: int = 10):
        """Complete training loop with comprehensive TensorBoard logging"""
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss, train_accuracy = self.train_epoch()
            
            # Validation
            val_loss, val_accuracy, val_predictions, val_probabilities = self.validate_epoch()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Log epoch metrics
            self.tensorboard.log_scalars(
                "epoch/loss",
                {"train": train_loss, "validation": val_loss},
                epoch
            )
            
            self.tensorboard.log_scalars(
                "epoch/accuracy", 
                {"train": train_accuracy, "validation": val_accuracy},
                epoch
            )
            
            # Log weight histograms every 5 epochs
            if epoch % 5 == 0:
                self._log_weight_histograms()
            
            # Log confusion matrix every 10 epochs
            if epoch % 10 == 0:
                self._log_confusion_matrix_figure(val_predictions, val_targets)
                self._log_class_accuracy(val_predictions, val_targets)
            
            # Log PR curves
            if epoch % 10 == 0:
                for class_idx in range(val_probabilities.shape[1]):
                    class_labels = (val_targets == class_idx).float()
                    class_probs = val_probabilities[:, class_idx]
                    
                    self.tensorboard.log_pr_curve(
                        f"pr_curve/class_{class_idx}",
                        class_labels,
                        class_probs,
                        epoch
                    )
            
            # Log sample predictions
            if epoch % 10 == 0:
                sample_data, sample_targets = next(iter(self.val_loader))
                with torch.no_grad():
                    sample_outputs = self.model(sample_data[:8])
                    if isinstance(sample_outputs, tuple):
                        sample_outputs = sample_outputs[0]
                    sample_predictions = torch.argmax(sample_outputs, dim=1)
                
                self._log_sample_predictions(
                    sample_data, sample_targets, sample_predictions
                )
            
            # Log embeddings
            if epoch % log_embeddings_every == 0 and epoch > 0:
                self._log_embeddings()
            
            # Track best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                
                # Log text about best model
                self.tensorboard.log_text(
                    "training/best_model_update",
                    f"New best model at epoch {epoch}: {val_accuracy:.4f} accuracy",
                    epoch
                )
            
            # Print progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Train Acc = {train_accuracy:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"Val Acc = {val_accuracy:.4f}")
        
        # Log final hyperparameters with metrics
        final_metrics = {
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss
        }
        
        self.tensorboard.log_hyperparameters(self._initial_hparams, final_metrics)
        
        print(f"✅ Training completed! Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss
        }
    
    def _log_embeddings(self):
        """Log embeddings for visualization"""
        self.model.eval()
        
        # Collect features from validation set
        features = []
        labels = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(self.val_loader):
                if i >= 10:  # Limit for performance
                    break
                
                # Get features from second-to-last layer
                output = self.model(data)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Use output as features (in real scenario, extract from hidden layer)
                features.append(output)
                labels.extend([f"Class_{t.item()}" for t in target])
        
        if features:
            features_tensor = torch.cat(features, dim=0)
            
            # Use subset for embedding visualization
            subset_size = min(500, len(features_tensor))
            indices = torch.randperm(len(features_tensor))[:subset_size]
            
            self.tensorboard.log_embedding(
                features_tensor[indices],
                metadata=[labels[i] for i in indices],
                tag=f"embeddings_epoch_{self.epoch}"
            )

# TensorBoard Utilities
class TensorBoardUtilities:
    """Utility functions for TensorBoard"""
    
    @staticmethod
    def create_learning_curves_comparison(tensorboard: AdvancedTensorBoard,
                                        experiments: Dict[str, Dict[str, List[float]]]):
        """Create learning curves comparison"""
        
        for metric_name in ["loss", "accuracy"]:
            metric_dict = {}
            
            for exp_name, exp_data in experiments.items():
                if metric_name in exp_data:
                    # Log each experiment's curve
                    for epoch, value in enumerate(exp_data[metric_name]):
                        tensorboard.log_scalar(
                            f"comparison/{metric_name}/{exp_name}",
                            value,
                            epoch
                        )
    
    @staticmethod
    def log_model_comparison(tensorboard: AdvancedTensorBoard,
                           models: Dict[str, nn.Module],
                           sample_input: torch.Tensor):
        """Log multiple models for comparison"""
        
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
                
                # Log model parameters count
                param_count = sum(p.numel() for p in model.parameters())
                tensorboard.log_scalar(f"model_comparison/parameters/{model_name}", param_count)
                
                # Log model complexity (FLOPs would be better)
                tensorboard.log_text(
                    f"model_comparison/architecture/{model_name}",
                    str(model)
                )
    
    @staticmethod
    def create_custom_dashboard(log_dir: str):
        """Create custom TensorBoard dashboard layout"""
        
        # This would typically involve creating custom TensorBoard plugins
        # For demo, we'll create a simple layout file
        
        layout_config = {
            "layout": {
                "Training Metrics": {
                    "Loss Comparison": ["Multiline", ["epoch/loss"]],
                    "Accuracy Trends": ["Multiline", ["epoch/accuracy"]]
                },
                "Model Analysis": {
                    "Weight Distributions": ["Histogram", ["weights/.*"]],
                    "Gradient Norms": ["Multiline", ["gradients/.*"]]
                },
                "Validation": {
                    "PR Curves": ["PR Curve", ["pr_curve/.*"]],
                    "Confusion Matrix": ["Image", ["validation/confusion_matrix"]]
                }
            }
        }
        
        print("Custom dashboard configuration created")
        return layout_config

if __name__ == "__main__":
    print("Advanced TensorBoard Integration")
    print("=" * 36)
    
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Install with: pip install tensorboard")
        print("Showing simulated TensorBoard integration...")
    
    print("\n1. TensorBoard Setup")
    print("-" * 22)
    
    # Initialize TensorBoard
    tensorboard = AdvancedTensorBoard(
        log_dir="./tensorboard_demo_logs",
        experiment_name="pytorch_tensorboard_demo"
    )
    
    print("\n2. Model and Data Preparation")
    print("-" * 33)
    
    # Create model with TensorBoard integration
    model = TensorBoardModel(
        input_size=784,
        hidden_sizes=[256, 128, 64],
        num_classes=10,
        dropout=0.3
    )
    
    # Create datasets
    train_dataset = TensorBoardDataset(size=5000, input_dim=784, num_classes=10)
    val_dataset = TensorBoardDataset(size=1000, input_dim=784, num_classes=10)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"✓ Model created with TensorBoard integration")
    print(f"✓ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
    
    print("\n3. Advanced TensorBoard Logging")
    print("-" * 34)
    
    # Log model architecture
    sample_input = torch.randn(1, 784)
    tensorboard.log_graph(model, sample_input)
    
    # Log sample data visualizations
    sample_batch, sample_labels = train_dataset.get_sample_batch(16)
    
    # Create sample images (reshape for demo)
    sample_images = sample_batch.view(-1, 1, 28, 28)  # Assume 28x28 images
    tensorboard.log_images("data/sample_batch", sample_images[:8])
    
    # Log data distribution
    tensorboard.log_histogram("data/input_distribution", sample_batch)
    tensorboard.log_histogram("data/label_distribution", sample_labels.float())
    
    print("✓ Model graph and sample data logged")
    
    print("\n4. Training with Comprehensive Logging")
    print("-" * 41)
    
    # Create trainer with TensorBoard integration
    trainer = TensorBoardTrainer(model, train_loader, val_loader, tensorboard)
    
    # Setup training
    trainer.setup_training(learning_rate=0.001, weight_decay=1e-4)
    
    # Train with comprehensive logging
    results = trainer.train(
        num_epochs=20,  # Reduced for demo
        log_embeddings_every=10
    )
    
    print(f"Training completed with best accuracy: {results['best_val_accuracy']:.4f}")
    
    print("\n5. Custom Visualizations")
    print("-" * 27)
    
    # Create custom matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Random data visualization
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax1.plot(x, y1, label='sin(x)')
    ax1.plot(x, y2, label='cos(x)')
    ax1.legend()
    ax1.set_title('Trigonometric Functions')
    
    # Plot 2: Model complexity analysis
    layer_names = ['Input', 'Hidden1', 'Hidden2', 'Hidden3', 'Output']
    layer_sizes = [784, 256, 128, 64, 10]
    
    ax2.bar(layer_names, layer_sizes)
    ax2.set_title('Model Layer Sizes')
    ax2.set_ylabel('Number of Units')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Log custom figure
    tensorboard.log_figure("analysis/custom_visualization", fig)
    plt.close(fig)
    
    print("✓ Custom visualizations logged")
    
    print("\n6. Advanced Features Demo")
    print("-" * 28)
    
    # Log embeddings manually
    features = torch.randn(100, 50)  # 100 samples, 50-dimensional features
    metadata = [f"Sample_{i}" for i in range(100)]
    tensorboard.log_embedding(features, metadata, tag="manual_embeddings")
    
    # Log text summaries
    training_summary = f"""
    Training Summary:
    - Final training loss: {results['final_train_loss']:.4f}
    - Final validation loss: {results['final_val_loss']:.4f}
    - Best validation accuracy: {results['best_val_accuracy']:.4f}
    - Model parameters: {sum(p.numel() for p in model.parameters()):,}
    """
    
    tensorboard.log_text("training/summary", training_summary)
    
    # Log hyperparameter comparison
    hparams_experiments = {
        "lr_001": {"learning_rate": 0.001, "best_accuracy": 0.85},
        "lr_0001": {"learning_rate": 0.0001, "best_accuracy": 0.82},
        "lr_01": {"learning_rate": 0.01, "best_accuracy": 0.78}
    }
    
    for exp_name, hparams in hparams_experiments.items():
        metrics = {"accuracy": hparams.pop("best_accuracy")}
        tensorboard.log_hyperparameters(hparams, metrics)
    
    print("✓ Advanced features demonstrated")
    
    print("\n7. TensorBoard Best Practices")
    print("-" * 32)
    
    best_practices = [
        "Use descriptive and hierarchical tag names",
        "Log both training and validation metrics",
        "Include model graph visualization",
        "Log weight and gradient histograms",
        "Use scalar summaries for key metrics",
        "Log sample predictions for visual verification",
        "Include hyperparameter logging",
        "Use embeddings for high-dimensional data visualization",
        "Log confusion matrices for classification tasks",
        "Include PR curves for imbalanced datasets",
        "Use custom figures for domain-specific visualizations",
        "Organize experiments with meaningful names"
    ]
    
    print("TensorBoard Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Utility Functions")
    print("-" * 21)
    
    utilities = TensorBoardUtilities()
    
    # Create learning curves comparison
    experiments_data = {
        "experiment_1": {
            "loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
        },
        "experiment_2": {
            "loss": [0.9, 0.7, 0.5, 0.4, 0.3],
            "accuracy": [0.5, 0.65, 0.75, 0.8, 0.85]
        }
    }
    
    utilities.create_learning_curves_comparison(tensorboard, experiments_data)
    
    # Create custom dashboard layout
    dashboard_config = utilities.create_custom_dashboard("./tensorboard_demo_logs")
    
    print("✓ Utility functions demonstrated")
    
    print("\n9. TensorBoard Commands")
    print("-" * 24)
    
    commands = {
        "Start TensorBoard": "tensorboard --logdir=./tensorboard_demo_logs",
        "Specific port": "tensorboard --logdir=./tensorboard_demo_logs --port=6007",
        "Remote access": "tensorboard --logdir=./tensorboard_demo_logs --host=0.0.0.0",
        "Compare experiments": "tensorboard --logdir_spec=exp1:./logs/exp1,exp2:./logs/exp2",
        "Reload interval": "tensorboard --logdir=./logs --reload_interval=1",
        "Profile memory": "tensorboard --logdir=./logs --profile_memory=true"
    }
    
    print("Common TensorBoard Commands:")
    for command, example in commands.items():
        print(f"  {command}: {example}")
    
    print("\n10. Integration Patterns")
    print("-" * 25)
    
    integration_patterns = [
        "PyTorch Lightning: Automatic TensorBoard logging",
        "Weights & Biases: Export TensorBoard logs to W&B",
        "MLflow: Integration with MLflow tracking",
        "Jupyter: View TensorBoard in notebooks with %tensorboard magic",
        "Docker: Mount TensorBoard logs volume",
        "Cloud: Upload logs to cloud storage for remote viewing",
        "CI/CD: Automated logging in training pipelines",
        "Production: Monitor model performance in deployment"
    ]
    
    print("TensorBoard Integration Patterns:")
    for pattern in integration_patterns:
        print(f"  - {pattern}")
    
    # Flush and close TensorBoard
    tensorboard.flush()
    tensorboard.close()
    
    print("\nAdvanced TensorBoard demonstration completed!")
    print("Key components covered:")
    print("  - Comprehensive logging (scalars, histograms, images, text)")
    print("  - Model graph and embedding visualization")
    print("  - Advanced training metrics and analysis")
    print("  - Custom figures and visualizations")
    print("  - Hyperparameter comparison and tracking")
    print("  - Best practices and integration patterns")
    
    print("\nTensorBoard enables:")
    print("  - Rich visualization of training progress")
    print("  - Model architecture understanding")
    print("  - Experiment comparison and analysis")
    print("  - Debugging and optimization insights")
    print("  - Collaborative model development")
    
    if TENSORBOARD_AVAILABLE:
        print(f"\n🚀 Start TensorBoard with:")
        print(f"   tensorboard --logdir=./tensorboard_demo_logs")
        print(f"   Then open http://localhost:6006")