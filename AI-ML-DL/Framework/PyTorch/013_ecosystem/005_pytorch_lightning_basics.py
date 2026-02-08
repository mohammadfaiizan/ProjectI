import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import matplotlib.pyplot as plt

# Note: PyTorch Lightning operations require the pytorch-lightning package
# Install with: pip install pytorch-lightning

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.strategies import DDPStrategy
    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    PYTORCH_LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available. Install with: pip install pytorch-lightning")

# Basic Lightning Module
if PYTORCH_LIGHTNING_AVAILABLE:
    class SimpleLightningModel(pl.LightningModule):
        """Basic PyTorch Lightning model for classification"""
        
        def __init__(self, input_size: int, hidden_size: int, num_classes: int, 
                     learning_rate: float = 0.001):
            super().__init__()
            
            # Save hyperparameters
            self.save_hyperparameters()
            
            # Model architecture
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes)
            )
            
            # Metrics storage
            self.training_step_outputs = []
            self.validation_step_outputs = []
            self.test_step_outputs = []
        
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            
            # Calculate accuracy
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            # Log metrics
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
            
            # Store outputs for epoch end
            self.training_step_outputs.append({
                'loss': loss,
                'acc': acc,
                'batch_size': x.size(0)
            })
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            # Log metrics
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            
            self.validation_step_outputs.append({
                'loss': loss,
                'acc': acc,
                'batch_size': x.size(0)
            })
            
            return loss
        
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)
            
            self.test_step_outputs.append({
                'loss': loss,
                'acc': acc,
                'predictions': y_hat.argmax(dim=1),
                'targets': y
            })
            
            return loss
        
        def on_train_epoch_end(self):
            # Compute epoch-level metrics
            if self.training_step_outputs:
                avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
                avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()
                
                self.log('train_epoch_loss', avg_loss)
                self.log('train_epoch_acc', avg_acc)
                
                # Clear outputs
                self.training_step_outputs.clear()
        
        def on_validation_epoch_end(self):
            if self.validation_step_outputs:
                avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
                avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
                
                self.log('val_epoch_loss', avg_loss)
                self.log('val_epoch_acc', avg_acc)
                
                self.validation_step_outputs.clear()
        
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler"""
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'frequency': 1
                }
            }

    # CNN Lightning Module
    class CNNLightningModel(pl.LightningModule):
        """CNN model using PyTorch Lightning"""
        
        def __init__(self, num_classes: int = 10, learning_rate: float = 0.001):
            super().__init__()
            
            self.save_hyperparameters()
            
            # CNN architecture
            self.conv_layers = nn.Sequential(
                # First block
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Second block
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                # Third block
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
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
            
            # For storing outputs
            self.training_step_outputs = []
            self.validation_step_outputs = []
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.classifier(x)
            return x
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            self.log('train_loss', loss, on_step=True, on_epoch=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True)
            
            self.training_step_outputs.append(loss)
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            self.log('val_loss', loss, on_epoch=True)
            self.log('val_acc', acc, on_epoch=True)
            
            self.validation_step_outputs.append(loss)
            return loss
        
        def on_train_epoch_end(self):
            if self.training_step_outputs:
                avg_loss = torch.stack(self.training_step_outputs).mean()
                self.log('train_epoch_loss', avg_loss)
                self.training_step_outputs.clear()
        
        def on_validation_epoch_end(self):
            if self.validation_step_outputs:
                avg_loss = torch.stack(self.validation_step_outputs).mean()
                self.log('val_epoch_loss', avg_loss)
                self.validation_step_outputs.clear()
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.1
            )
            
            return [optimizer], [scheduler]

    # Advanced Lightning Module with Custom Callbacks
    class AdvancedLightningModel(pl.LightningModule):
        """Advanced Lightning model with custom features"""
        
        def __init__(self, input_size: int, hidden_sizes: List[int], 
                     num_classes: int, learning_rate: float = 0.001,
                     weight_decay: float = 1e-4, dropout: float = 0.3):
            super().__init__()
            
            self.save_hyperparameters()
            
            # Build dynamic architecture
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, num_classes))
            
            self.model = nn.Sequential(*layers)
            
            # Metrics
            self.training_step_outputs = []
            self.validation_step_outputs = []
            
            # For gradient tracking
            self.gradient_norms = []
        
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            
            # Calculate additional metrics
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            # Log to different loggers
            self.log_dict({
                'train_loss': loss,
                'train_acc': acc,
                'learning_rate': self.optimizers().param_groups[0]['lr']
            }, on_step=True, on_epoch=True)
            
            self.training_step_outputs.append({
                'loss': loss,
                'acc': acc
            })
            
            return loss
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            
            # Calculate confidence scores
            probs = F.softmax(y_hat, dim=1)
            confidence = probs.max(dim=1)[0].mean()
            
            self.log_dict({
                'val_loss': loss,
                'val_acc': acc,
                'val_confidence': confidence
            }, on_epoch=True)
            
            self.validation_step_outputs.append({
                'loss': loss,
                'acc': acc,
                'confidence': confidence
            })
            
            return loss
        
        def on_train_epoch_end(self):
            if self.training_step_outputs:
                # Compute gradient norm
                total_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                self.gradient_norms.append(total_norm)
                self.log('gradient_norm', total_norm)
                
                self.training_step_outputs.clear()
        
        def on_validation_epoch_end(self):
            if self.validation_step_outputs:
                self.validation_step_outputs.clear()
        
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            
            # Cosine annealing scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

    # Custom Callbacks
    class CustomMetricsCallback(pl.Callback):
        """Custom callback for additional metrics and logging"""
        
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
        
        def on_train_epoch_end(self, trainer, pl_module):
            # Log custom metrics
            if hasattr(pl_module, 'gradient_norms') and pl_module.gradient_norms:
                avg_grad_norm = np.mean(pl_module.gradient_norms)
                pl_module.log('avg_gradient_norm', avg_grad_norm)
        
        def on_validation_epoch_end(self, trainer, pl_module):
            # Custom validation logic
            current_val_loss = trainer.callback_metrics.get('val_loss')
            if current_val_loss is not None:
                self.val_losses.append(current_val_loss.item())
                
                # Log moving average
                if len(self.val_losses) >= 5:
                    moving_avg = np.mean(self.val_losses[-5:])
                    pl_module.log('val_loss_ma5', moving_avg)

    class PrintingCallback(pl.Callback):
        """Callback for custom printing and monitoring"""
        
        def on_train_start(self, trainer, pl_module):
            print("🚀 Training started!")
        
        def on_train_end(self, trainer, pl_module):
            print("✅ Training completed!")
        
        def on_epoch_end(self, trainer, pl_module):
            if trainer.current_epoch % 10 == 0:
                print(f"📊 Epoch {trainer.current_epoch} completed")

# Data Module
if PYTORCH_LIGHTNING_AVAILABLE:
    class CustomDataModule(pl.LightningDataModule):
        """Custom data module for PyTorch Lightning"""
        
        def __init__(self, data_size: int = 1000, input_size: int = 20, 
                     num_classes: int = 3, batch_size: int = 32):
            super().__init__()
            self.data_size = data_size
            self.input_size = input_size
            self.num_classes = num_classes
            self.batch_size = batch_size
            
            # Will be set in setup()
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
        
        def setup(self, stage: Optional[str] = None):
            """Create datasets"""
            
            # Generate synthetic data
            X = torch.randn(self.data_size, self.input_size)
            
            # Create labels with some pattern
            y = ((X[:, 0] + X[:, 1]) > 0).long()
            y = y + ((X[:, 2] + X[:, 3]) > 1).long()
            y = torch.clamp(y, 0, self.num_classes - 1)
            
            # Create dataset
            dataset = torch.utils.data.TensorDataset(X, y)
            
            # Split dataset
            train_size = int(0.7 * len(dataset))
            val_size = int(0.2 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
        
        def train_dataloader(self):
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0  # Set to 0 for compatibility
            )
        
        def val_dataloader(self):
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0
            )
        
        def test_dataloader(self):
            return DataLoader(
                self.test_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=0
            )

# Training Pipeline
class LightningTrainer:
    """Wrapper for PyTorch Lightning training pipeline"""
    
    def __init__(self, model_type: str = 'simple'):
        self.model_type = model_type
        self.model = None
        self.trainer = None
        self.data_module = None
    
    def create_model(self, **kwargs):
        """Create Lightning model"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            print("PyTorch Lightning not available")
            return
        
        if self.model_type == 'simple':
            self.model = SimpleLightningModel(**kwargs)
        elif self.model_type == 'cnn':
            self.model = CNNLightningModel(**kwargs)
        elif self.model_type == 'advanced':
            self.model = AdvancedLightningModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✓ Created {self.model_type} Lightning model")
    
    def create_data_module(self, **kwargs):
        """Create data module"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        self.data_module = CustomDataModule(**kwargs)
        print("✓ Created data module")
    
    def setup_trainer(self, max_epochs: int = 50, gpus: int = 0, 
                     enable_checkpointing: bool = True,
                     enable_logging: bool = True):
        """Setup PyTorch Lightning trainer"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        callbacks = []
        
        # Add callbacks
        if enable_checkpointing:
            checkpoint_callback = ModelCheckpoint(
                dirpath='./checkpoints',
                filename='{epoch}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3,
                save_last=True
            )
            callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode='min'
        )
        callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # Custom callbacks
        callbacks.extend([
            CustomMetricsCallback(),
            PrintingCallback()
        ])
        
        # Logger
        logger = None
        if enable_logging:
            logger = TensorBoardLogger('tb_logs', name='lightning_model')
        
        # Create trainer
        self.trainer = Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator='cpu',  # Use CPU for compatibility
            devices=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=True
        )
        
        print(f"✓ Setup trainer with {max_epochs} epochs")
    
    def train(self):
        """Train the model"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            print("PyTorch Lightning not available")
            return
        
        if self.model is None or self.trainer is None or self.data_module is None:
            print("Model, trainer, or data module not setup")
            return
        
        print("🏃 Starting training...")
        self.trainer.fit(self.model, self.data_module)
        print("✅ Training completed!")
    
    def test(self):
        """Test the model"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        if self.trainer is None or self.data_module is None:
            print("Trainer or data module not setup")
            return
        
        print("🧪 Testing model...")
        self.trainer.test(self.model, self.data_module)
        print("✅ Testing completed!")

# Advanced Features Demo
class LightningAdvancedFeatures:
    """Demonstrate advanced PyTorch Lightning features"""
    
    @staticmethod
    def demonstrate_auto_lr_find():
        """Demonstrate automatic learning rate finding"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        print("🔍 Demonstrating Auto LR Find...")
        
        # Create model and data
        model = SimpleLightningModel(input_size=20, hidden_size=64, num_classes=3)
        data_module = CustomDataModule()
        
        # Create trainer
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False
        )
        
        # Auto find learning rate
        lr_finder = trainer.tuner.lr_find(model, data_module)
        
        # Get suggestion
        suggested_lr = lr_finder.suggestion()
        print(f"📊 Suggested learning rate: {suggested_lr}")
        
        # Update model
        model.hparams.learning_rate = suggested_lr
        print("✓ Learning rate updated")
    
    @staticmethod
    def demonstrate_auto_batch_size():
        """Demonstrate automatic batch size scaling"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        print("📏 Demonstrating Auto Batch Size Scaling...")
        
        # Create model and data
        model = SimpleLightningModel(input_size=20, hidden_size=64, num_classes=3)
        data_module = CustomDataModule(batch_size=32)
        
        # Create trainer
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            enable_progress_bar=False
        )
        
        # Auto scale batch size
        trainer.tune(model, data_module)
        
        print(f"📊 Optimized batch size: {data_module.batch_size}")
    
    @staticmethod
    def demonstrate_model_summary():
        """Demonstrate model summary"""
        if not PYTORCH_LIGHTNING_AVAILABLE:
            return
        
        print("📋 Model Summary:")
        
        model = AdvancedLightningModel(
            input_size=784, 
            hidden_sizes=[512, 256, 128], 
            num_classes=10
        )
        
        # Print model summary
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

def demo_lightning_features():
    """Demonstrate various Lightning features"""
    
    print("⚡ PyTorch Lightning Features Demo")
    print("=" * 40)
    
    if not PYTORCH_LIGHTNING_AVAILABLE:
        print("PyTorch Lightning not available - showing conceptual examples")
        return
    
    # 1. Auto LR Find
    LightningAdvancedFeatures.demonstrate_auto_lr_find()
    
    print("\n" + "-" * 40)
    
    # 2. Auto Batch Size
    LightningAdvancedFeatures.demonstrate_auto_batch_size()
    
    print("\n" + "-" * 40)
    
    # 3. Model Summary
    LightningAdvancedFeatures.demonstrate_model_summary()

if __name__ == "__main__":
    print("PyTorch Lightning Fundamentals")
    print("=" * 35)
    
    if not PYTORCH_LIGHTNING_AVAILABLE:
        print("PyTorch Lightning not available. Install with: pip install pytorch-lightning")
        print("Showing conceptual examples...")
    
    print("\n1. Basic Lightning Model")
    print("-" * 26)
    
    # Create trainer pipeline
    trainer = LightningTrainer(model_type='simple')
    
    # Setup components
    trainer.create_model(
        input_size=20,
        hidden_size=64,
        num_classes=3,
        learning_rate=0.001
    )
    
    trainer.create_data_module(
        data_size=1000,
        input_size=20,
        num_classes=3,
        batch_size=32
    )
    
    trainer.setup_trainer(
        max_epochs=20,
        enable_checkpointing=True,
        enable_logging=True
    )
    
    # Train model
    if PYTORCH_LIGHTNING_AVAILABLE:
        trainer.train()
        trainer.test()
    
    print("\n2. Advanced Features Demo")
    print("-" * 28)
    
    demo_lightning_features()
    
    print("\n3. Lightning Benefits")
    print("-" * 22)
    
    benefits = [
        "Automatic optimization and training loops",
        "Built-in support for distributed training",
        "Comprehensive logging and checkpointing",
        "Easy model testing and validation",
        "Automatic learning rate and batch size finding",
        "Integration with popular loggers (TensorBoard, W&B)",
        "Clean separation of research and engineering code",
        "Extensive callback system for customization",
        "Support for multiple accelerators (GPU, TPU)",
        "Reproducible experiments with seed setting",
        "Easy model deployment and serving",
        "Rich ecosystem of extensions and plugins"
    ]
    
    print("PyTorch Lightning Benefits:")
    for i, benefit in enumerate(benefits, 1):
        print(f"{i:2d}. {benefit}")
    
    print("\n4. Best Practices")
    print("-" * 19)
    
    best_practices = [
        "Use LightningDataModule for data handling",
        "Implement proper logging in training steps",
        "Use callbacks for custom training logic",
        "Save hyperparameters with save_hyperparameters()",
        "Implement proper validation and test steps",
        "Use learning rate schedulers appropriately",
        "Monitor gradient norms to detect training issues",
        "Use early stopping to prevent overfitting",
        "Implement proper checkpointing strategies",
        "Use distributed training for large models",
        "Profile your code to identify bottlenecks",
        "Use automatic mixed precision for efficiency"
    ]
    
    print("Lightning Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n5. Common Patterns")
    print("-" * 19)
    
    patterns = {
        "Classification": "Standard supervised learning with cross-entropy loss",
        "Regression": "Continuous target prediction with MSE loss",
        "GANs": "Generator-discriminator training with custom training steps",
        "VAEs": "Variational autoencoders with reconstruction + KL loss",
        "Self-Supervised": "Contrastive or reconstruction-based training",
        "Multi-Task": "Multiple outputs with combined loss functions",
        "Transfer Learning": "Fine-tuning pretrained models",
        "Reinforcement Learning": "Policy gradient or Q-learning implementations"
    }
    
    print("Common Lightning Patterns:")
    for pattern, description in patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n6. Integration Ecosystem")
    print("-" * 26)
    
    integrations = [
        "Weights & Biases: Experiment tracking and hyperparameter tuning",
        "TensorBoard: Visualization and monitoring",
        "Hydra: Configuration management",
        "MLflow: Model lifecycle management",
        "Ray: Hyperparameter optimization and distributed training",
        "Optuna: Hyperparameter optimization",
        "DeepSpeed: Memory-efficient training",
        "FairScale: Model parallelism and optimization",
        "Torchmetrics: Comprehensive metrics library",
        "Lightning Bolts: Collection of models and utilities"
    ]
    
    print("Lightning Ecosystem Integrations:")
    for integration in integrations:
        print(f"  - {integration}")
    
    print("\n7. Common Use Cases")
    print("-" * 21)
    
    use_cases = [
        "Research experimentation with clean code structure",
        "Production model training with robust logging",
        "Distributed training across multiple GPUs/nodes",
        "Hyperparameter optimization and model selection",
        "Model versioning and experiment tracking",
        "Easy deployment and serving workflows",
        "Educational purposes with clear abstractions",
        "Rapid prototyping of new architectures"
    ]
    
    print("Common Lightning Use Cases:")
    for i, use_case in enumerate(use_cases, 1):
        print(f"{i}. {use_case}")
    
    print("\nPyTorch Lightning fundamentals demonstration completed!")
    print("Key components covered:")
    print("  - Lightning Module creation and customization")
    print("  - Data Module for clean data handling")
    print("  - Trainer configuration and callbacks")
    print("  - Advanced features (auto LR find, auto batch size)")
    print("  - Custom callbacks and logging")
    print("  - Best practices and common patterns")
    
    print("\nPyTorch Lightning enables:")
    print("  - Cleaner, more maintainable ML code")
    print("  - Automatic handling of training boilerplate")
    print("  - Easy scaling to multiple devices")
    print("  - Comprehensive experiment tracking")
    print("  - Reproducible research workflows")