import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import io
from PIL import Image

# Sample Models for TensorBoard Logging
class TensorBoardTestCNN(nn.Module):
    """CNN model for TensorBoard demonstration"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AttentionVisualizationModel(nn.Module):
    """Model with attention for visualization"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Store attention for visualization
        self.last_attention = None
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Compute attention
        attention_weights = self.attention(features)
        self.last_attention = attention_weights.detach()
        
        # Apply attention
        attended_features = features * attention_weights
        
        # Classification
        pooled = self.global_pool(attended_features)
        output = self.classifier(pooled)
        
        return output

# TensorBoard Logging Utilities
class TensorBoardLogger:
    """Comprehensive TensorBoard logging utilities"""
    
    def __init__(self, log_dir: str = 'runs/experiment'):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_scalars(self, scalars: Dict[str, float], step: Optional[int] = None):
        """Log scalar values"""
        if step is None:
            step = self.step
        
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
    
    def log_images(self, images: Dict[str, torch.Tensor], step: Optional[int] = None):
        """Log images to TensorBoard"""
        if step is None:
            step = self.step
        
        for name, image_tensor in images.items():
            if image_tensor.dim() == 4:
                # Batch of images
                grid = torchvision.utils.make_grid(image_tensor, normalize=True)
                self.writer.add_image(name, grid, step)
            elif image_tensor.dim() == 3:
                # Single image
                self.writer.add_image(name, image_tensor, step)
    
    def log_histograms(self, tensors: Dict[str, torch.Tensor], step: Optional[int] = None):
        """Log tensor histograms"""
        if step is None:
            step = self.step
        
        for name, tensor in tensors.items():
            self.writer.add_histogram(name, tensor, step)
    
    def log_model_graph(self, model: nn.Module, input_tensor: torch.Tensor):
        """Log model computational graph"""
        self.writer.add_graph(model, input_tensor)
    
    def log_embeddings(self, embeddings: torch.Tensor, metadata: Optional[List] = None,
                      label_img: Optional[torch.Tensor] = None, tag: str = 'embeddings'):
        """Log high-dimensional embeddings"""
        self.writer.add_embedding(embeddings, metadata=metadata, 
                                 label_img=label_img, tag=tag)
    
    def log_pr_curve(self, predictions: torch.Tensor, labels: torch.Tensor,
                    class_names: Optional[List[str]] = None, step: Optional[int] = None):
        """Log precision-recall curves"""
        if step is None:
            step = self.step
        
        if class_names is None:
            num_classes = predictions.shape[1]
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        for i, class_name in enumerate(class_names):
            class_predictions = predictions[:, i]
            class_labels = (labels == i).float()
            self.writer.add_pr_curve(f'pr_curve/{class_name}', 
                                   class_labels, class_predictions, step)
    
    def log_confusion_matrix(self, predictions: torch.Tensor, labels: torch.Tensor,
                           class_names: Optional[List[str]] = None, step: Optional[int] = None):
        """Log confusion matrix as image"""
        if step is None:
            step = self.step
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Convert to numpy
        pred_labels = predictions.argmax(dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Convert to image tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = transforms.ToTensor()(image)
        
        self.writer.add_image('confusion_matrix', image_tensor, step)
        plt.close()
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """Log current learning rate"""
        if step is None:
            step = self.step
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)
    
    def log_gradients(self, model: nn.Module, step: Optional[int] = None):
        """Log gradient statistics"""
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
                self.writer.add_scalar(f'gradient_norms/{name}', 
                                     param.grad.norm().item(), step)
    
    def log_weights(self, model: nn.Module, step: Optional[int] = None):
        """Log model weights"""
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param, step)
            self.writer.add_scalar(f'weight_norms/{name}', param.norm().item(), step)
    
    def increment_step(self):
        """Increment global step counter"""
        self.step += 1
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

class ModelTrainingLogger:
    """Log model training process to TensorBoard"""
    
    def __init__(self, model: nn.Module, log_dir: str = 'runs/training'):
        self.model = model
        self.logger = TensorBoardLogger(log_dir)
        self.epoch = 0
        self.step = 0
    
    def log_epoch_start(self, epoch: int):
        """Log start of epoch"""
        self.epoch = epoch
        
        # Log model weights and architecture (first epoch only)
        if epoch == 0:
            # Create dummy input for graph logging
            dummy_input = torch.randn(1, 3, 32, 32)
            if next(self.model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()
            
            self.logger.log_model_graph(self.model, dummy_input)
            self.logger.log_weights(self.model, self.step)
    
    def log_batch(self, loss: float, batch_idx: int, outputs: torch.Tensor, 
                 targets: torch.Tensor, data: torch.Tensor, 
                 optimizer: torch.optim.Optimizer):
        """Log training batch information"""
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean().item()
        
        # Log scalars
        scalars = {
            'train/loss': loss,
            'train/accuracy': accuracy,
            'train/epoch': self.epoch
        }
        self.logger.log_scalars(scalars, self.step)
        
        # Log learning rate
        self.logger.log_learning_rate(optimizer, self.step)
        
        # Log gradients (every 10 batches)
        if batch_idx % 10 == 0:
            self.logger.log_gradients(self.model, self.step)
        
        # Log sample images (first batch of epoch)
        if batch_idx == 0:
            sample_images = data[:8]  # First 8 images
            self.logger.log_images({'train/samples': sample_images}, self.step)
        
        self.step += 1
    
    def log_validation(self, val_loss: float, val_accuracy: float, 
                      predictions: torch.Tensor, labels: torch.Tensor,
                      sample_images: Optional[torch.Tensor] = None):
        """Log validation results"""
        
        # Log validation metrics
        scalars = {
            'val/loss': val_loss,
            'val/accuracy': val_accuracy
        }
        self.logger.log_scalars(scalars, self.step)
        
        # Log confusion matrix
        self.logger.log_confusion_matrix(predictions, labels, step=self.step)
        
        # Log PR curves
        if predictions.shape[1] <= 10:  # Only for small number of classes
            self.logger.log_pr_curve(F.softmax(predictions, dim=1), labels, step=self.step)
        
        # Log sample validation images
        if sample_images is not None:
            self.logger.log_images({'val/samples': sample_images}, self.step)
    
    def log_epoch_end(self, train_loss: float, train_acc: float, 
                     val_loss: float, val_acc: float):
        """Log end of epoch summary"""
        
        # Log epoch summaries
        scalars = {
            'epoch/train_loss': train_loss,
            'epoch/train_accuracy': train_acc,
            'epoch/val_loss': val_loss,
            'epoch/val_accuracy': val_acc
        }
        self.logger.log_scalars(scalars, self.epoch)
        
        # Log weights every few epochs
        if self.epoch % 5 == 0:
            self.logger.log_weights(self.model, self.step)
    
    def close(self):
        """Close logger"""
        self.logger.close()

class AdvancedTensorBoardLogger:
    """Advanced TensorBoard logging with custom visualizations"""
    
    def __init__(self, log_dir: str = 'runs/advanced'):
        self.logger = TensorBoardLogger(log_dir)
    
    def log_attention_maps(self, model: AttentionVisualizationModel, 
                          images: torch.Tensor, step: int):
        """Log attention visualizations"""
        
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            attention_maps = model.last_attention
        
        # Visualize attention for first few images
        num_images = min(4, images.size(0))
        
        for i in range(num_images):
            # Original image
            orig_img = images[i]
            
            # Attention map
            attention = attention_maps[i, 0]  # First (and only) channel
            
            # Resize attention to match image size
            attention_resized = F.interpolate(
                attention.unsqueeze(0).unsqueeze(0),
                size=orig_img.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Create attention overlay
            # Normalize attention to [0, 1]
            attention_norm = (attention_resized - attention_resized.min()) / \
                           (attention_resized.max() - attention_resized.min())
            
            # Create RGB attention map
            attention_rgb = torch.stack([
                attention_norm,
                torch.zeros_like(attention_norm),
                1 - attention_norm
            ])
            
            # Log images
            self.logger.log_images({
                f'attention/image_{i}': orig_img,
                f'attention/map_{i}': attention_rgb,
            }, step)
    
    def log_feature_maps(self, model: nn.Module, images: torch.Tensor, 
                        layer_name: str, step: int):
        """Log feature maps from specific layer"""
        
        # Hook to capture feature maps
        feature_maps = {}
        
        def hook(module, input, output):
            feature_maps['features'] = output.detach()
        
        # Register hook
        target_layer = None
        for name, layer in model.named_modules():
            if name == layer_name:
                target_layer = layer
                break
        
        if target_layer is None:
            print(f"Layer {layer_name} not found")
            return
        
        handle = target_layer.register_forward_hook(hook)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(images[:1])  # Process only first image
        
        # Remove hook
        handle.remove()
        
        # Visualize feature maps
        if 'features' in feature_maps:
            features = feature_maps['features'][0]  # First image
            
            # Select subset of feature maps to visualize
            num_features = min(16, features.size(0))
            selected_features = features[:num_features]
            
            # Normalize features for visualization
            for i in range(num_features):
                feature_map = selected_features[i]
                feature_map = (feature_map - feature_map.min()) / \
                             (feature_map.max() - feature_map.min())
                selected_features[i] = feature_map
            
            # Create grid
            grid = torchvision.utils.make_grid(
                selected_features.unsqueeze(1), 
                nrow=4, normalize=False
            )
            
            self.logger.log_images({f'features/{layer_name}': grid}, step)
    
    def log_weight_distributions(self, model: nn.Module, step: int):
        """Log weight distribution analysis"""
        
        weight_stats = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weights = param.data.flatten()
                
                # Calculate statistics
                weight_stats[f'{name}/mean'] = weights.mean().item()
                weight_stats[f'{name}/std'] = weights.std().item()
                weight_stats[f'{name}/min'] = weights.min().item()
                weight_stats[f'{name}/max'] = weights.max().item()
                
                # Log histogram
                self.logger.log_histograms({f'weight_dist/{name}': weights}, step)
        
        # Log statistics
        self.logger.log_scalars(weight_stats, step)
    
    def log_gradient_flow(self, model: nn.Module, step: int):
        """Log gradient flow analysis"""
        
        gradient_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[f'grad_flow/{name}'] = grad_norm
        
        self.logger.log_scalars(gradient_norms, step)
        
        # Create gradient flow plot
        layer_names = []
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_names.append(name.split('.')[-1][:10])  # Truncate names
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(grad_norms, 'b-o')
            plt.xlabel('Layer')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Flow')
            plt.xticks(range(len(layer_names)), layer_names, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_tensor = transforms.ToTensor()(image)
            
            self.logger.log_images({'gradient_flow': image_tensor}, step)
            plt.close()
    
    def log_training_dynamics(self, losses: List[float], accuracies: List[float],
                            learning_rates: List[float], step: int):
        """Log training dynamics overview"""
        
        # Create training dynamics plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        steps = range(len(losses))
        
        # Loss plot
        axes[0, 0].plot(steps, losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(steps, accuracies)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].plot(steps, learning_rates)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss vs Accuracy scatter
        axes[1, 1].scatter(losses, accuracies, alpha=0.6)
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = transforms.ToTensor()(image)
        
        self.logger.log_images({'training_dynamics': image_tensor}, step)
        plt.close()
    
    def close(self):
        """Close logger"""
        self.logger.close()

# Training Loop with TensorBoard Integration
def train_with_tensorboard(model: nn.Module, train_loader, val_loader, 
                          num_epochs: int = 5, log_dir: str = 'runs/training'):
    """Train model with comprehensive TensorBoard logging"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup logging
    training_logger = ModelTrainingLogger(model, log_dir)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Log epoch start
        training_logger.log_epoch_start(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Log batch
            training_logger.log_batch(
                loss.item(), batch_idx, outputs, targets, data, optimizer
            )
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []
        sample_images = None
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data, targets = data.to(device), targets.to(device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # Collect predictions for analysis
                all_predictions.append(outputs.cpu())
                all_labels.append(targets.cpu())
                
                # Save sample images from first batch
                if batch_idx == 0:
                    sample_images = data[:8].cpu()
        
        # Calculate epoch metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        # Combine predictions and labels
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        # Log validation results
        training_logger.log_validation(
            val_loss_avg, val_acc, all_predictions, all_labels, sample_images
        )
        
        # Log epoch end
        training_logger.log_epoch_end(train_loss_avg, train_acc, val_loss_avg, val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
    
    training_logger.close()
    print(f"Training completed! TensorBoard logs saved to: {log_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    print("TensorBoard Integration")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    train_data = torch.randn(200, 3, 32, 32)
    train_labels = torch.randint(0, 10, (200,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_data = torch.randn(80, 3, 32, 32)
    val_labels = torch.randint(0, 10, (80,))
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print("\n1. Basic TensorBoard Logging")
    print("-" * 35)
    
    # Basic logging demonstration
    logger = TensorBoardLogger('runs/basic_demo')
    
    # Log some sample data
    sample_scalars = {
        'accuracy': 0.85,
        'loss': 0.45,
        'learning_rate': 0.001
    }
    logger.log_scalars(sample_scalars, step=0)
    
    # Log sample images
    sample_images = train_data[:8]
    logger.log_images({'sample_batch': sample_images}, step=0)
    
    # Log histograms
    sample_weights = torch.randn(1000)
    logger.log_histograms({'sample_weights': sample_weights}, step=0)
    
    print("Basic logging completed")
    
    print("\n2. Model Graph Logging")
    print("-" * 28)
    
    # Create model and log computational graph
    model = TensorBoardTestCNN(num_classes=10)
    sample_input = torch.randn(1, 3, 32, 32)
    
    logger.log_model_graph(model, sample_input)
    print("Model graph logged")
    
    logger.close()
    
    print("\n3. Training with TensorBoard")
    print("-" * 33)
    
    # Train model with comprehensive logging
    model = TensorBoardTestCNN(num_classes=10)
    
    print("Starting training with TensorBoard logging...")
    train_with_tensorboard(
        model, train_loader, val_loader, 
        num_epochs=3, log_dir='runs/training_demo'
    )
    
    print("\n4. Advanced Logging Features")
    print("-" * 35)
    
    # Advanced logging demonstration
    advanced_logger = AdvancedTensorBoardLogger('runs/advanced_demo')
    
    # Create attention model for advanced visualizations
    attention_model = AttentionVisualizationModel(num_classes=10).to(device)
    
    sample_batch = train_data[:4].to(device)
    
    print("Logging attention maps...")
    advanced_logger.log_attention_maps(attention_model, sample_batch, step=0)
    
    print("Logging feature maps...")
    advanced_logger.log_feature_maps(
        attention_model, sample_batch, 'backbone.0', step=0
    )
    
    print("Logging weight distributions...")
    advanced_logger.log_weight_distributions(attention_model, step=0)
    
    # Simulate training for gradient flow logging
    print("Logging gradient flow...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(attention_model.parameters(), lr=0.001)
    
    # Training step to generate gradients
    attention_model.train()
    outputs = attention_model(sample_batch)
    loss = criterion(outputs, torch.randint(0, 10, (4,)).to(device))
    loss.backward()
    
    advanced_logger.log_gradient_flow(attention_model, step=0)
    
    # Log training dynamics
    print("Logging training dynamics...")
    sample_losses = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
    sample_accuracies = [0.3, 0.5, 0.6, 0.7, 0.75, 0.8]
    sample_lrs = [0.001, 0.001, 0.0005, 0.0005, 0.0001, 0.0001]
    
    advanced_logger.log_training_dynamics(
        sample_losses, sample_accuracies, sample_lrs, step=0
    )
    
    advanced_logger.close()
    
    print("\n5. Embedding Visualization")
    print("-" * 32)
    
    # Create embeddings for visualization
    embedding_logger = TensorBoardLogger('runs/embeddings_demo')
    
    # Generate sample embeddings
    num_samples = 100
    embedding_dim = 50
    
    embeddings = torch.randn(num_samples, embedding_dim)
    
    # Create metadata (labels)
    labels = torch.randint(0, 10, (num_samples,)).tolist()
    metadata = [f'Class_{label}' for label in labels]
    
    # Create label images (small thumbnails)
    label_images = torch.randn(num_samples, 3, 32, 32)
    
    print("Logging embeddings...")
    embedding_logger.log_embeddings(
        embeddings, metadata=metadata, 
        label_img=label_images, tag='feature_embeddings'
    )
    
    embedding_logger.close()
    
    print("\n6. Hyperparameter Logging")
    print("-" * 32)
    
    # Demonstrate hyperparameter comparison
    hparam_logger = TensorBoardLogger('runs/hparams_demo')
    
    # Define hyperparameters to compare
    hparam_configs = [
        {'lr': 0.001, 'batch_size': 16, 'dropout': 0.3},
        {'lr': 0.01, 'batch_size': 32, 'dropout': 0.5},
        {'lr': 0.0001, 'batch_size': 8, 'dropout': 0.2}
    ]
    
    # Simulate results for each configuration
    for i, hparams in enumerate(hparam_configs):
        # Simulate training results
        final_accuracy = 0.7 + 0.1 * np.random.randn()
        final_loss = 0.5 + 0.2 * np.random.randn()
        
        # Log hyperparameters and metrics
        hparam_logger.writer.add_hparams(
            hparams,
            {'accuracy': final_accuracy, 'loss': final_loss},
            run_name=f'run_{i}'
        )
    
    hparam_logger.close()
    
    print("\n7. Custom Metrics and Plots")
    print("-" * 34)
    
    # Custom visualization examples
    custom_logger = TensorBoardLogger('runs/custom_demo')
    
    # Log custom metrics over time
    steps = range(100)
    for step in steps:
        # Simulate training metrics with noise
        accuracy = 0.5 + 0.3 * (1 - np.exp(-step/20)) + 0.05 * np.random.randn()
        loss = 2.0 * np.exp(-step/15) + 0.1 * np.random.randn()
        
        custom_logger.log_scalars({
            'metrics/accuracy': accuracy,
            'metrics/loss': loss,
            'metrics/step': step
        }, step)
    
    # Create and log custom figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/10)
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title('Custom Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    
    # Convert plot to image and log
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = transforms.ToTensor()(image)
    custom_logger.log_images({'custom_plot': image_tensor}, step=0)
    plt.close()
    
    custom_logger.close()
    
    print("\nTensorBoard integration completed!")
    print("\nGenerated TensorBoard logs:")
    print("  - runs/basic_demo/")
    print("  - runs/training_demo/")
    print("  - runs/advanced_demo/")
    print("  - runs/embeddings_demo/")
    print("  - runs/hparams_demo/")
    print("  - runs/custom_demo/")
    print("\nTo view logs, run:")
    print("  tensorboard --logdir runs/")
    print("  Then open http://localhost:6006 in your browser")
    
    print("\nTensorBoard Features Demonstrated:")
    features = [
        "Model computational graph visualization",
        "Training metrics (loss, accuracy, learning rate)",
        "Weight and gradient histograms",
        "Image and attention map visualization", 
        "Feature map visualization",
        "Confusion matrices and PR curves",
        "High-dimensional embedding visualization",
        "Hyperparameter comparison",
        "Custom plots and metrics",
        "Training dynamics analysis"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i:2d}. {feature}")