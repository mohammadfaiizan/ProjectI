"""
LPIPS Training Framework
========================

Comprehensive training framework for LPIPS (Learned Perceptual Image Patch Similarity).
Implements 2AFC training, validation, and evaluation with comprehensive metrics.

This module includes:
- LPIPS trainer with 2AFC loss
- Training loop with validation
- Evaluation metrics and correlation analysis
- Model checkpointing and resuming
- Tensorboard logging
- Comparison with traditional metrics

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import warnings

from lpips_model import LPIPS, LPIPSLoss, create_lpips_model
from data_loader import LPIPSDataModule, JNDDataset


class LPIPSTrainer:
    """
    Comprehensive trainer for LPIPS model
    
    Handles training, validation, evaluation, and analysis of LPIPS models
    using 2AFC loss on JND datasets.
    """
    
    def __init__(self,
                 model: LPIPS,
                 data_module: LPIPSDataModule,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-6,
                 optimizer_type: str = 'adam',
                 scheduler_type: str = 'cosine',
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 experiment_name: str = 'lpips_training'):
        """
        Initialize LPIPS trainer
        
        Args:
            model: LPIPS model to train
            data_module: Data module for training/validation data
            device: Device for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
            scheduler_type: Type of lr scheduler ('cosine', 'step', 'plateau')
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for tensorboard logs
            experiment_name: Name of the experiment
        """
        self.model = model.to(device)
        self.data_module = data_module
        self.device = device
        self.experiment_name = experiment_name
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer_type, learning_rate, weight_decay)
        
        # Initialize loss function
        self.loss_fn = LPIPSLoss(use_gpu=(device == 'cuda'))
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_correlation = 0.0
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_correlation': [],
            'learning_rates': []
        }
        
        print(f"LPIPS Trainer initialized:")
        print(f"  Model: {model.backbone_name}")
        print(f"  Device: {device}")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Experiment: {experiment_name}")
    
    def _create_optimizer(self, optimizer_type: str, lr: float, weight_decay: float) -> optim.Optimizer:
        """Create optimizer for training"""
        
        # Only train the linear layers, freeze the backbone
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'linear_layers' in name:
                trainable_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        print(f"Training {len(trainable_params)} linear layer parameters")
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        
        if scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        elif scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=10)
        elif scheduler_type.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Enable gradient computation for linear layers only
        for name, param in self.model.named_parameters():
            if 'linear_layers' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        
        pbar = tqdm(train_loader, desc=f'Train Epoch {self.current_epoch}')
        
        for batch_idx, (ref_imgs, img1s, img2s, judgments) in enumerate(pbar):
            # Move to device
            ref_imgs = ref_imgs.to(self.device)
            img1s = img1s.to(self.device)
            img2s = img2s.to(self.device)
            judgments = judgments.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss
            loss, metrics = self.loss_fn(self.model, ref_imgs, img1s, img2s, judgments)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log batch metrics to tensorboard
            global_step = self.current_epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('train/batch_loss', metrics['loss'], global_step)
            self.writer.add_scalar('train/batch_accuracy', metrics['accuracy'], global_step)
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Store predictions for correlation analysis
        all_lpips_distances = []
        all_human_preferences = []
        all_distance_diffs = []
        
        val_loader = self.data_module.val_dataloader()
        
        with torch.no_grad():
            for ref_imgs, img1s, img2s, judgments in tqdm(val_loader, desc='Validation'):
                # Move to device
                ref_imgs = ref_imgs.to(self.device)
                img1s = img1s.to(self.device)
                img2s = img2s.to(self.device)
                judgments = judgments.to(self.device)
                
                # Forward pass and compute loss
                loss, metrics = self.loss_fn(self.model, ref_imgs, img1s, img2s, judgments)
                
                # Update metrics
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                num_batches += 1
                
                # Store for correlation analysis
                dist1 = self.model(ref_imgs, img1s)
                dist2 = self.model(ref_imgs, img2s)
                
                all_lpips_distances.extend(dist1.cpu().numpy().flatten())
                all_lpips_distances.extend(dist2.cpu().numpy().flatten())
                
                # Human preference: 0 if img1 preferred, 1 if img2 preferred
                # Convert to distance preference: higher distance = less preferred
                human_prefs = []
                distance_diffs = []
                
                for i in range(len(judgments)):
                    if judgments[i] == 0:  # img1 preferred
                        human_prefs.extend([1, 0])  # dist1 should be smaller
                    else:  # img2 preferred
                        human_prefs.extend([0, 1])  # dist2 should be smaller
                    
                    # Distance difference for this comparison
                    diff = (dist1[i] - dist2[i]).item()
                    distance_diffs.append(diff)
                
                all_human_preferences.extend(human_prefs)
                all_distance_diffs.extend(distance_diffs)
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Calculate correlation with human preferences
        if len(all_lpips_distances) > 0 and len(all_human_preferences) > 0:
            # Ensure same length
            min_len = min(len(all_lpips_distances), len(all_human_preferences))
            lpips_subset = all_lpips_distances[:min_len]
            human_subset = all_human_preferences[:min_len]
            
            if len(set(human_subset)) > 1:  # Ensure variance in human preferences
                correlation, _ = pearsonr(lpips_subset, human_subset)
                correlation = abs(correlation)  # Take absolute value
            else:
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'correlation': correlation
        }
    
    def train(self, num_epochs: int = 100, 
             validate_every: int = 1,
             save_every: int = 10,
             early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Complete training loop
        
        Args:
            num_epochs: Number of training epochs
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"Starting LPIPS training for {num_epochs} epochs...")
        
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % validate_every == 0:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'correlation': 0.0}
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['accuracy'].append(train_metrics['accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            self.train_history['val_correlation'].append(val_metrics['correlation'])
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to tensorboard
            self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/epoch_accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('val/epoch_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/epoch_accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('val/correlation', val_metrics['correlation'], epoch)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model
            is_best = False
            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_correlation = val_metrics['correlation']
                best_epoch = epoch
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.3f}, "
                  f"Val Acc: {val_metrics['accuracy']:.3f}, "
                  f"Val Corr: {val_metrics['correlation']:.3f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break
        
        # Final logging
        print(f"\nTraining completed!")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Best validation correlation: {self.best_val_correlation:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.train_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_correlation': self.best_val_correlation,
            'model_config': self.model.get_model_info()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with accuracy {self.best_val_accuracy:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_history = checkpoint['train_history']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_val_correlation = checkpoint['best_val_correlation']
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        return epoch
    
    def evaluate(self, test_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of trained LPIPS model
        
        Args:
            test_loader: Optional test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_loader is None:
            test_loader = self.data_module.test_dataloader()
        
        self.model.eval()
        
        # Evaluation metrics
        all_lpips_distances = []
        all_human_judgments = []
        all_traditional_metrics = {
            'l1': [],
            'l2': [],
            'ssim': []
        }
        
        total_accuracy = 0.0
        num_batches = 0
        
        print("Running comprehensive evaluation...")
        
        with torch.no_grad():
            for ref_imgs, img1s, img2s, judgments in tqdm(test_loader, desc='Evaluation'):
                # Move to device
                ref_imgs = ref_imgs.to(self.device)
                img1s = img1s.to(self.device)
                img2s = img2s.to(self.device)
                judgments = judgments.to(self.device)
                
                # LPIPS distances
                dist1 = self.model(ref_imgs, img1s)
                dist2 = self.model(ref_imgs, img2s)
                
                # 2AFC accuracy
                logits = dist1 - dist2
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                accuracy = (predictions == judgments.float()).float().mean()
                total_accuracy += accuracy.item()
                num_batches += 1
                
                # Store distances and judgments for correlation analysis
                all_lpips_distances.extend(dist1.cpu().numpy().flatten())
                all_lpips_distances.extend(dist2.cpu().numpy().flatten())
                
                # Convert judgments to distance preferences
                for i in range(len(judgments)):
                    if judgments[i] == 0:  # img1 preferred
                        all_human_judgments.extend([0, 1])  # dist1 smaller, dist2 larger
                    else:  # img2 preferred
                        all_human_judgments.extend([1, 0])  # dist1 larger, dist2 smaller
                
                # Traditional metrics
                for i in range(ref_imgs.shape[0]):
                    # L1 distance
                    l1_1 = torch.nn.functional.l1_loss(ref_imgs[i], img1s[i]).item()
                    l1_2 = torch.nn.functional.l1_loss(ref_imgs[i], img2s[i]).item()
                    all_traditional_metrics['l1'].extend([l1_1, l1_2])
                    
                    # L2 distance
                    l2_1 = torch.nn.functional.mse_loss(ref_imgs[i], img1s[i]).item()
                    l2_2 = torch.nn.functional.mse_loss(ref_imgs[i], img2s[i]).item()
                    all_traditional_metrics['l2'].extend([l2_1, l2_2])
                    
                    # Simplified SSIM
                    ssim_1 = self._compute_ssim(ref_imgs[i], img1s[i])
                    ssim_2 = self._compute_ssim(ref_imgs[i], img2s[i])
                    all_traditional_metrics['ssim'].extend([ssim_1, ssim_2])
        
        # Calculate final metrics
        avg_accuracy = total_accuracy / num_batches
        
        # Correlation analysis
        correlations = {}
        
        # LPIPS correlation
        if len(all_lpips_distances) > 0 and len(all_human_judgments) > 0:
            min_len = min(len(all_lpips_distances), len(all_human_judgments))
            lpips_subset = all_lpips_distances[:min_len]
            human_subset = all_human_judgments[:min_len]
            
            if len(set(human_subset)) > 1:
                pearson_corr, _ = pearsonr(lpips_subset, human_subset)
                spearman_corr, _ = spearmanr(lpips_subset, human_subset)
                correlations['lpips_pearson'] = abs(pearson_corr)
                correlations['lpips_spearman'] = abs(spearman_corr)
            else:
                correlations['lpips_pearson'] = 0.0
                correlations['lpips_spearman'] = 0.0
        
        # Traditional metrics correlations
        for metric_name, metric_values in all_traditional_metrics.items():
            if len(metric_values) > 0:
                min_len = min(len(metric_values), len(all_human_judgments))
                metric_subset = metric_values[:min_len]
                human_subset = all_human_judgments[:min_len]
                
                if len(set(human_subset)) > 1:
                    # For SSIM, higher is better (inverse correlation)
                    if metric_name == 'ssim':
                        metric_subset = [-x for x in metric_subset]
                    
                    pearson_corr, _ = pearsonr(metric_subset, human_subset)
                    correlations[f'{metric_name}_pearson'] = abs(pearson_corr)
        
        evaluation_results = {
            'accuracy': avg_accuracy,
            **correlations
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"2AFC Accuracy: {avg_accuracy:.4f}")
        print(f"LPIPS Pearson Correlation: {correlations.get('lpips_pearson', 0):.4f}")
        print(f"LPIPS Spearman Correlation: {correlations.get('lpips_spearman', 0):.4f}")
        
        print("\nComparison with Traditional Metrics:")
        for metric in ['l1', 'l2', 'ssim']:
            corr = correlations.get(f'{metric}_pearson', 0)
            print(f"  {metric.upper()} Pearson Correlation: {corr:.4f}")
        
        return evaluation_results
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute simplified SSIM between two images"""
        # Convert to grayscale
        if img1.shape[0] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114]).to(img1.device).view(3, 1, 1)
            img1_gray = (img1 * weights).sum(dim=0)
            img2_gray = (img2 * weights).sum(dim=0)
        else:
            img1_gray = img1.squeeze(0)
            img2_gray = img2.squeeze(0)
        
        # Calculate SSIM
        mu1 = img1_gray.mean()
        mu2 = img2_gray.mean()
        
        sigma1_sq = ((img1_gray - mu1) ** 2).mean()
        sigma2_sq = ((img2_gray - mu2) ** 2).mean()
        sigma12 = ((img1_gray - mu1) * (img2_gray - mu2)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.item()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.train_history['loss']))
        
        # Loss plot
        ax1.plot(epochs, self.train_history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.train_history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_history['accuracy'], label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.train_history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Correlation plot
        ax3.plot(epochs, self.train_history['val_correlation'], label='Val Correlation', linewidth=2, color='green')
        ax3.set_title('Validation Correlation with Human Judgment', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax4.plot(epochs, self.train_history['learning_rates'], linewidth=2, color='orange')
        ax4.set_title('Learning Rate Schedule', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


def create_lpips_trainer(backbone: str = 'vgg',
                        data_dir: str = './LPIPS_Data',
                        batch_size: int = 32,
                        learning_rate: float = 1e-4,
                        experiment_name: str = None) -> LPIPSTrainer:
    """
    Factory function to create LPIPS trainer
    
    Args:
        backbone: Backbone network ('alexnet', 'vgg', 'squeezenet')
        data_dir: Path to JND dataset
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        experiment_name: Name of experiment
        
    Returns:
        Configured LPIPS trainer
    """
    if experiment_name is None:
        experiment_name = f'lpips_{backbone}_{int(time.time())}'
    
    # Create model
    model = create_lpips_model(backbone=backbone, pretrained=True)
    
    # Create data module
    data_module = LPIPSDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Setup data
    data_module.setup()
    
    # Create trainer
    trainer = LPIPSTrainer(
        model=model,
        data_module=data_module,
        learning_rate=learning_rate,
        experiment_name=experiment_name
    )
    
    return trainer


def main():
    """Demonstration of LPIPS training"""
    print("=" * 60)
    print("LPIPS Training Demonstration")
    print("=" * 60)
    
    # Create synthetic dataset for demonstration
    from data_loader import create_synthetic_jnd_dataset
    
    synthetic_dir = "./demo_jnd_data"
    print("Creating synthetic dataset for demonstration...")
    create_synthetic_jnd_dataset(synthetic_dir, num_samples=200)
    
    # Create trainer
    trainer = create_lpips_trainer(
        backbone='vgg',
        data_dir=synthetic_dir,
        batch_size=8,
        learning_rate=1e-3,
        experiment_name='lpips_demo'
    )
    
    # Train for a few epochs
    print("\nStarting training...")
    history = trainer.train(num_epochs=5, validate_every=1)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()
    
    print("\nTraining demonstration complete!")
    
    return trainer, eval_results


if __name__ == "__main__":
    trainer, results = main()