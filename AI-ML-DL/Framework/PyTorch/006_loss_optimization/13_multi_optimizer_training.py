#!/usr/bin/env python3
"""PyTorch Multi-Optimizer Training - Multiple optimizers in one model"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import defaultdict
import numpy as np

print("=== Multi-Optimizer Training Overview ===")

print("Multi-optimizer training scenarios:")
print("1. Different learning rates for different layers")
print("2. Feature extraction vs fine-tuning")
print("3. Adversarial training (generator vs discriminator)")
print("4. Multi-task learning")
print("5. Progressive layer unfreezing")
print("6. Domain adaptation")
print("7. Meta-learning")
print("8. Ensemble training")

print("\n=== Basic Multi-Optimizer Setup ===")

class MultiLayerModel(nn.Module):
    def __init__(self, input_size=128, hidden_sizes=[256, 512, 256], num_classes=10):
        super().__init__()
        
        # Backbone layers
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Middle layers
        self.middle = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Head layers
        self.head = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.middle(x)
        x = self.head(x)
        return x
    
    def get_parameter_groups(self):
        """Get different parameter groups for different optimizers"""
        return {
            'backbone': list(self.backbone.parameters()),
            'middle': list(self.middle.parameters()),
            'head': list(self.head.parameters())
        }

# Create model and sample data
model = MultiLayerModel()
sample_input = torch.randn(64, 128)
sample_target = torch.randint(0, 10, (64,))
loss_fn = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Get parameter groups
param_groups = model.get_parameter_groups()
for group_name, params in param_groups.items():
    param_count = sum(p.numel() for p in params)
    print(f"  {group_name}: {param_count:,} parameters")

print("\n=== Basic Multi-Optimizer Training ===")

class MultiOptimizerTrainer:
    """Basic multi-optimizer trainer"""
    
    def __init__(self, model, optimizers_config, schedulers_config=None):
        self.model = model
        self.optimizers = {}
        self.schedulers = {}
        
        # Setup optimizers
        param_groups = model.get_parameter_groups()
        for name, config in optimizers_config.items():
            if name in param_groups:
                optimizer_class = config['optimizer']
                optimizer_params = config.get('params', {})
                self.optimizers[name] = optimizer_class(param_groups[name], **optimizer_params)
        
        # Setup schedulers
        if schedulers_config:
            for name, config in schedulers_config.items():
                if name in self.optimizers:
                    scheduler_class = config['scheduler']
                    scheduler_params = config.get('params', {})
                    self.schedulers[name] = scheduler_class(self.optimizers[name], **scheduler_params)
    
    def zero_grad(self):
        """Zero gradients for all optimizers"""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def step(self):
        """Step all optimizers"""
        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def scheduler_step(self, epoch=None, metrics=None):
        """Step all schedulers"""
        for name, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'step'):
                # Handle different scheduler types
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    if metrics is not None:
                        scheduler.step(metrics)
                else:
                    scheduler.step()
    
    def get_learning_rates(self):
        """Get current learning rates"""
        lrs = {}
        for name, optimizer in self.optimizers.items():
            lrs[name] = [group['lr'] for group in optimizer.param_groups]
        return lrs
    
    def train_step(self, input_data, targets, loss_fn):
        """Perform a single training step"""
        self.zero_grad()
        
        # Forward pass
        outputs = self.model(input_data)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Step optimizers
        self.step()
        
        return loss.item()

# Configure different optimizers for different parts
optimizers_config = {
    'backbone': {
        'optimizer': optim.Adam,
        'params': {'lr': 0.0001, 'weight_decay': 1e-4}
    },
    'middle': {
        'optimizer': optim.SGD,
        'params': {'lr': 0.001, 'momentum': 0.9}
    },
    'head': {
        'optimizer': optim.AdamW,
        'params': {'lr': 0.01, 'weight_decay': 0.01}
    }
}

schedulers_config = {
    'backbone': {
        'scheduler': lr_scheduler.ExponentialLR,
        'params': {'gamma': 0.95}
    },
    'middle': {
        'scheduler': lr_scheduler.StepLR,
        'params': {'step_size': 5, 'gamma': 0.5}
    },
    'head': {
        'scheduler': lr_scheduler.CosineAnnealingLR,
        'params': {'T_max': 10}
    }
}

# Create trainer
trainer = MultiOptimizerTrainer(model, optimizers_config, schedulers_config)

print("Training with multiple optimizers:")
for epoch in range(12):
    loss = trainer.train_step(sample_input, sample_target, loss_fn)
    trainer.scheduler_step()
    
    if epoch % 3 == 0:
        lrs = trainer.get_learning_rates()
        print(f"  Epoch {epoch:2d}: Loss = {loss:.6f}")
        for name, lr_list in lrs.items():
            print(f"    {name:8}: LR = {lr_list[0]:.6f}")

print("\n=== Transfer Learning with Multi-Optimizers ===")

class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_features, num_classes=10):
        super().__init__()
        # Pretrained feature extractor (frozen initially)
        self.features = pretrained_features
        
        # New classifier head
        feature_size = 512  # Assuming features output 512-dim vectors
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initially freeze feature extractor
        self.freeze_features()
    
    def freeze_features(self):
        """Freeze feature extractor parameters"""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_features(self):
        """Unfreeze feature extractor parameters"""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Simple feature extraction simulation
        features = self.features(x)
        # Global average pooling simulation
        features = features.mean(dim=-1)
        return self.classifier(features)
    
    def get_parameter_groups(self):
        """Get parameter groups for transfer learning"""
        groups = {}
        
        # Feature extractor parameters (if unfrozen)
        feature_params = [p for p in self.features.parameters() if p.requires_grad]
        if feature_params:
            groups['features'] = feature_params
        
        # Classifier parameters
        groups['classifier'] = list(self.classifier.parameters())
        
        return groups

# Create a simple "pretrained" feature extractor
pretrained_features = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU()
)

transfer_model = TransferLearningModel(pretrained_features)

print("Transfer learning with multi-optimizer training:")

class TransferTrainer:
    """Transfer learning trainer with progressive unfreezing"""
    
    def __init__(self, model):
        self.model = model
        self.optimizers = {}
        self.phase = "classifier_only"
        self.setup_phase1()
    
    def setup_phase1(self):
        """Phase 1: Train only classifier"""
        self.model.freeze_features()
        param_groups = self.model.get_parameter_groups()
        
        self.optimizers = {
            'classifier': optim.Adam(param_groups['classifier'], lr=0.001)
        }
        self.phase = "classifier_only"
        print("  Phase 1: Training classifier only")
    
    def setup_phase2(self):
        """Phase 2: Fine-tune entire model"""
        self.model.unfreeze_features()
        param_groups = self.model.get_parameter_groups()
        
        self.optimizers = {
            'features': optim.Adam(param_groups['features'], lr=0.0001),  # Lower LR for pretrained
            'classifier': optim.Adam(param_groups['classifier'], lr=0.001)  # Higher LR for new layers
        }
        self.phase = "full_model"
        print("  Phase 2: Fine-tuning entire model")
    
    def train_step(self, input_data, targets, loss_fn):
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_data)
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Step optimizers
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        return loss.item()
    
    def get_learning_rates(self):
        lrs = {}
        for name, optimizer in self.optimizers.items():
            lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

# Test transfer learning trainer
transfer_trainer = TransferTrainer(transfer_model)

# Phase 1: Train classifier only
for epoch in range(8):
    loss = transfer_trainer.train_step(sample_input, sample_target, loss_fn)
    
    if epoch % 2 == 0:
        lrs = transfer_trainer.get_learning_rates()
        print(f"    Epoch {epoch:2d}: Loss = {loss:.6f}, LRs = {lrs}")
    
    # Switch to phase 2 after 4 epochs
    if epoch == 4:
        transfer_trainer.setup_phase2()

print("\n=== Adversarial Training with Multi-Optimizers ===")

class SimpleGAN(nn.Module):
    def __init__(self, latent_dim=64, data_dim=128):
        super().__init__()
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def generate(self, batch_size, device='cpu'):
        """Generate fake data"""
        noise = torch.randn(batch_size, 64).to(device)
        return self.generator(noise)
    
    def discriminate(self, data):
        """Discriminate real vs fake"""
        return self.discriminator(data)
    
    def get_parameter_groups(self):
        return {
            'generator': list(self.generator.parameters()),
            'discriminator': list(self.discriminator.parameters())
        }

class GANTrainer:
    """GAN trainer with separate optimizers"""
    
    def __init__(self, gan_model):
        self.model = gan_model
        param_groups = gan_model.get_parameter_groups()
        
        # Separate optimizers for generator and discriminator
        self.optimizers = {
            'generator': optim.Adam(param_groups['generator'], lr=0.0002, betas=(0.5, 0.999)),
            'discriminator': optim.Adam(param_groups['discriminator'], lr=0.0002, betas=(0.5, 0.999))
        }
        
        self.criterion = nn.BCELoss()
    
    def train_discriminator(self, real_data):
        """Train discriminator on real and fake data"""
        self.optimizers['discriminator'].zero_grad()
        
        batch_size = real_data.size(0)
        
        # Train on real data
        real_labels = torch.ones(batch_size, 1)
        real_output = self.model.discriminate(real_data)
        real_loss = self.criterion(real_output, real_labels)
        
        # Train on fake data
        fake_data = self.model.generate(batch_size)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = self.model.discriminate(fake_data.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizers['discriminator'].step()
        
        return d_loss.item()
    
    def train_generator(self, batch_size):
        """Train generator to fool discriminator"""
        self.optimizers['generator'].zero_grad()
        
        # Generate fake data
        fake_data = self.model.generate(batch_size)
        fake_labels = torch.ones(batch_size, 1)  # Generator wants discriminator to think it's real
        
        # Get discriminator's opinion on fake data
        fake_output = self.model.discriminate(fake_data)
        g_loss = self.criterion(fake_output, fake_labels)
        
        g_loss.backward()
        self.optimizers['generator'].step()
        
        return g_loss.item()

# Test GAN training
print("Adversarial training with multi-optimizers:")
gan = SimpleGAN()
gan_trainer = GANTrainer(gan)

# Simulate real data
real_data = torch.randn(32, 128)

for epoch in range(10):
    # Train discriminator
    d_loss = gan_trainer.train_discriminator(real_data)
    
    # Train generator
    g_loss = gan_trainer.train_generator(32)
    
    if epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}: D_loss = {d_loss:.4f}, G_loss = {g_loss:.4f}")

print("\n=== Multi-Task Learning ===")

class MultiTaskModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 classes
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single regression output
        )
    
    def forward(self, x, task='both'):
        shared_features = self.shared_backbone(x)
        
        outputs = {}
        if task in ['both', 'classification']:
            outputs['classification'] = self.classification_head(shared_features)
        if task in ['both', 'regression']:
            outputs['regression'] = self.regression_head(shared_features)
        
        return outputs
    
    def get_parameter_groups(self):
        return {
            'shared': list(self.shared_backbone.parameters()),
            'classification': list(self.classification_head.parameters()),
            'regression': list(self.regression_head.parameters())
        }

class MultiTaskTrainer:
    """Multi-task trainer with task-specific optimizers"""
    
    def __init__(self, model):
        self.model = model
        param_groups = model.get_parameter_groups()
        
        # Different optimizers for different components
        self.optimizers = {
            'shared': optim.Adam(param_groups['shared'], lr=0.001),
            'classification': optim.Adam(param_groups['classification'], lr=0.001),
            'regression': optim.Adam(param_groups['regression'], lr=0.0001)  # Lower LR for regression
        }
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
    
    def train_step(self, input_data, class_targets, reg_targets, task_weights=None):
        """Train on multiple tasks"""
        if task_weights is None:
            task_weights = {'classification': 1.0, 'regression': 1.0}
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(input_data, task='both')
        
        # Compute losses
        class_loss = self.classification_loss(outputs['classification'], class_targets)
        reg_loss = self.regression_loss(outputs['regression'], reg_targets)
        
        # Weighted total loss
        total_loss = (task_weights['classification'] * class_loss + 
                     task_weights['regression'] * reg_loss)
        
        # Backward pass
        total_loss.backward()
        
        # Step optimizers
        for optimizer in self.optimizers.values():
            optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'classification_loss': class_loss.item(),
            'regression_loss': reg_loss.item()
        }

# Test multi-task training
print("Multi-task learning with multi-optimizers:")
multitask_model = MultiTaskModel()
multitask_trainer = MultiTaskTrainer(multitask_model)

# Sample data for both tasks
input_data = torch.randn(32, 128)
class_targets = torch.randint(0, 10, (32,))
reg_targets = torch.randn(32, 1)

for epoch in range(8):
    # Vary task weights over time
    epoch_weights = {
        'classification': 1.0,
        'regression': 0.5 + 0.5 * (epoch / 10)  # Gradually increase regression importance
    }
    
    losses = multitask_trainer.train_step(input_data, class_targets, reg_targets, epoch_weights)
    
    if epoch % 2 == 0:
        print(f"  Epoch {epoch:2d}: Total = {losses['total_loss']:.4f}, "
              f"Class = {losses['classification_loss']:.4f}, "
              f"Reg = {losses['regression_loss']:.4f}")

print("\n=== Progressive Layer Unfreezing ===")

class ProgressiveUnfreezingTrainer:
    """Trainer that progressively unfreezes layers"""
    
    def __init__(self, model, unfreeze_schedule):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule  # {epoch: [layer_names]}
        self.current_epoch = 0
        
        # Initially freeze all layers
        self.freeze_all_layers()
        self.setup_optimizers()
    
    def freeze_all_layers(self):
        """Freeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self, layer_names):
        """Unfreeze specific layers"""
        for name, module in self.model.named_modules():
            for layer_name in layer_names:
                if layer_name in name:
                    for param in module.parameters():
                        param.requires_grad = True
                    print(f"    Unfroze layer: {name}")
    
    def setup_optimizers(self):
        """Setup optimizers for currently unfrozen parameters"""
        # Get all parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if trainable_params:
            self.optimizer = optim.Adam(trainable_params, lr=0.001)
        else:
            self.optimizer = None
    
    def step_epoch(self):
        """Step to next epoch and check for layer unfreezing"""
        self.current_epoch += 1
        
        # Check if we need to unfreeze layers
        if self.current_epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[self.current_epoch]
            print(f"  Epoch {self.current_epoch}: Unfreezing layers: {layers_to_unfreeze}")
            self.unfreeze_layers(layers_to_unfreeze)
            self.setup_optimizers()
    
    def train_step(self, input_data, targets, loss_fn):
        """Training step with current optimizer"""
        if self.optimizer is None:
            return 0.0
        
        self.optimizer.zero_grad()
        
        outputs = self.model(input_data)
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Test progressive unfreezing
print("Progressive layer unfreezing:")
progressive_model = MultiLayerModel()

# Define unfreezing schedule
unfreeze_schedule = {
    1: ['head'],      # Unfreeze head first
    4: ['middle'],    # Then middle layers
    7: ['backbone']   # Finally backbone
}

progressive_trainer = ProgressiveUnfreezingTrainer(progressive_model, unfreeze_schedule)

for epoch in range(10):
    progressive_trainer.step_epoch()
    
    loss = progressive_trainer.train_step(sample_input, sample_target, loss_fn)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in progressive_model.parameters() if p.requires_grad)
    
    print(f"    Epoch {epoch+1:2d}: Loss = {loss:.6f}, Trainable params = {trainable_params:,}")

print("\n=== Multi-Optimizer Best Practices ===")

print("When to Use Multiple Optimizers:")
print("1. Transfer learning: Different LRs for pretrained vs new layers")
print("2. Adversarial training: Separate optimizers for generator/discriminator")
print("3. Multi-task learning: Task-specific optimization strategies")
print("4. Progressive training: Different phases need different optimizers")
print("5. Large models: Different learning rates for different depths")

print("\nOptimizer Selection Guidelines:")
print("1. Pretrained layers: Lower learning rates (Adam/AdamW)")
print("2. New layers: Higher learning rates (Adam/SGD)")
print("3. Batch norm layers: Often benefit from different LR")
print("4. Embedding layers: Usually lower learning rates")
print("5. Output layers: Can handle higher learning rates")

print("\nScheduler Coordination:")
print("1. Use compatible schedulers across optimizers")
print("2. Consider relative learning rate relationships")
print("3. Some schedulers (plateau) need validation metrics")
print("4. Warm-up can be applied selectively")
print("5. Monitor convergence of all components")

print("\nImplementation Tips:")
print("1. Group parameters logically (backbone, head, etc.)")
print("2. Use descriptive names for optimizer groups")
print("3. Implement proper state saving/loading")
print("4. Monitor learning rates and gradients separately")
print("5. Consider gradient accumulation per optimizer")

print("\nCommon Pitfalls:")
print("1. Forgetting to zero_grad() all optimizers")
print("2. Inconsistent scheduler stepping")
print("3. Optimizer state device mismatches")
print("4. Learning rate scheduling conflicts")
print("5. Memory leaks from maintaining multiple optimizers")

print("\nDebugging Multi-Optimizer Training:")
print("1. Log learning rates for all optimizers")
print("2. Monitor gradient norms per parameter group")
print("3. Track loss contributions from different components")
print("4. Validate parameter updates are happening")
print("5. Check for gradient flow issues")

print("\n=== Multi-Optimizer Training Complete ===")

# Memory cleanup
del model, sample_input, sample_target