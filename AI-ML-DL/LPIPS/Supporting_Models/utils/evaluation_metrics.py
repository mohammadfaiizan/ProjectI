"""
Shared Evaluation Metrics for LPIPS Supporting Models
=====================================================

Common evaluation metrics and utilities for AlexNet, VGG, and SqueezeNet
implementations, including classification metrics, efficiency metrics,
and LPIPS-specific evaluations.

This module provides:
- Standard classification metrics
- Efficiency and performance metrics
- LPIPS-specific evaluation utilities
- Model comparison frameworks
- Benchmarking tools

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
import pandas as pd


class ClassificationMetrics:
    """
    Comprehensive classification evaluation metrics
    """
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.top5_predictions = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions
        
        Args:
            outputs: Model outputs (logits) [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Convert to probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Top-1 predictions
        _, top1_pred = outputs.max(1)
        self.predictions.extend(top1_pred.cpu().numpy())
        
        # Top-5 predictions
        _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
        self.top5_predictions.extend(top5_pred.cpu().numpy())
        
        # Targets
        self.targets.extend(targets.cpu().numpy())
        
        # Confidence scores
        confidences = probabilities.max(1)[0]
        self.confidences.extend(confidences.cpu().numpy())
    
    def compute_accuracy(self) -> Dict[str, float]:
        """Compute top-1 and top-5 accuracy"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        top5_predictions = np.array(self.top5_predictions)
        
        # Top-1 accuracy
        top1_correct = (predictions == targets).sum()
        top1_accuracy = top1_correct / len(targets) * 100
        
        # Top-5 accuracy
        top5_correct = 0
        for i, target in enumerate(targets):
            if target in top5_predictions[i]:
                top5_correct += 1
        top5_accuracy = top5_correct / len(targets) * 100
        
        return {
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'total_samples': len(targets)
        }
    
    def compute_per_class_metrics(self) -> Dict:
        """Compute per-class precision, recall, and F1-score"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Get unique classes present in targets
        unique_classes = np.unique(targets)
        
        per_class_metrics = {}
        
        for class_id in unique_classes:
            # True positives, false positives, false negatives
            tp = ((predictions == class_id) & (targets == class_id)).sum()
            fp = ((predictions == class_id) & (targets != class_id)).sum()
            fn = ((predictions != class_id) & (targets == class_id)).sum()
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[int(class_id)] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': (targets == class_id).sum()
            }
        
        return per_class_metrics
    
    def compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute confidence-related metrics"""
        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        correct_mask = (predictions == targets)
        
        return {
            'mean_confidence': confidences.mean(),
            'std_confidence': confidences.std(),
            'mean_correct_confidence': confidences[correct_mask].mean(),
            'mean_incorrect_confidence': confidences[~correct_mask].mean(),
            'confidence_accuracy_correlation': np.corrcoef(confidences, correct_mask.astype(float))[0, 1]
        }
    
    def plot_confusion_matrix(self, normalize: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Limit to first 10 classes for readability
        if self.num_classes > 10:
            mask = targets < 10
            predictions = predictions[mask]
            targets = targets[mask]
        
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class EfficiencyMetrics:
    """
    Model efficiency and performance metrics
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.inference_times = []
        self.memory_usage = []
        self.throughput_measurements = []
    
    def measure_inference_time(self, 
                              model: nn.Module, 
                              input_tensor: torch.Tensor, 
                              num_runs: int = 100,
                              warmup_runs: int = 10) -> Dict[str, float]:
        """
        Measure model inference time
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor for testing
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize if using CUDA
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(input_tensor)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        inference_times = np.array(inference_times)
        
        results = {
            'mean_time_ms': inference_times.mean() * 1000,
            'std_time_ms': inference_times.std() * 1000,
            'min_time_ms': inference_times.min() * 1000,
            'max_time_ms': inference_times.max() * 1000,
            'median_time_ms': np.median(inference_times) * 1000,
            'fps': 1.0 / inference_times.mean(),
            'batch_size': input_tensor.shape[0],
            'throughput_images_per_second': input_tensor.shape[0] / inference_times.mean()
        }
        
        self.inference_times.extend(inference_times)
        return results
    
    def measure_memory_usage(self, 
                           model: nn.Module, 
                           input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Measure GPU memory usage during inference
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor for testing
            
        Returns:
            Dictionary with memory usage statistics
        """
        if self.device != 'cuda':
            return {'note': 'Memory measurement requires CUDA'}
        
        model.eval()
        
        # Clear cache and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure baseline memory
        baseline_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        
        memory_used = peak_memory - baseline_memory
        
        results = {
            'baseline_memory_mb': baseline_memory / 1024**2,
            'peak_memory_mb': peak_memory / 1024**2,
            'current_memory_mb': current_memory / 1024**2,
            'memory_used_mb': memory_used / 1024**2,
            'memory_per_image_mb': memory_used / input_tensor.shape[0] / 1024**2,
            'batch_size': input_tensor.shape[0]
        }
        
        self.memory_usage.append(results)
        return results
    
    def measure_throughput_scaling(self, 
                                 model: nn.Module, 
                                 batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
                                 input_shape: Tuple[int, int, int] = (3, 224, 224)) -> Dict:
        """
        Measure throughput scaling with batch size
        
        Args:
            model: PyTorch model
            batch_sizes: List of batch sizes to test
            input_shape: Input tensor shape (C, H, W)
            
        Returns:
            Dictionary with scaling results
        """
        scaling_results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
            
            # Measure timing
            timing_results = self.measure_inference_time(model, input_tensor, num_runs=50)
            
            # Measure memory (if CUDA)
            memory_results = self.measure_memory_usage(model, input_tensor)
            
            scaling_results[batch_size] = {
                'timing': timing_results,
                'memory': memory_results
            }
        
        return scaling_results
    
    def analyze_efficiency_trends(self, scaling_results: Dict) -> Dict:
        """Analyze efficiency trends from scaling results"""
        batch_sizes = list(scaling_results.keys())
        throughputs = [scaling_results[bs]['timing']['throughput_images_per_second'] 
                      for bs in batch_sizes]
        
        # Memory efficiency (if available)
        memory_per_image = []
        for bs in batch_sizes:
            if 'memory_per_image_mb' in scaling_results[bs]['memory']:
                memory_per_image.append(scaling_results[bs]['memory']['memory_per_image_mb'])
        
        analysis = {
            'optimal_batch_size': batch_sizes[np.argmax(throughputs)],
            'max_throughput': max(throughputs),
            'throughput_scaling_efficiency': throughputs[-1] / throughputs[0] / (batch_sizes[-1] / batch_sizes[0])
        }
        
        if memory_per_image:
            analysis['memory_efficiency'] = {
                'min_memory_per_image': min(memory_per_image),
                'memory_scaling_factor': memory_per_image[-1] / memory_per_image[0]
            }
        
        return analysis


class LPIPSEvaluationMetrics:
    """
    LPIPS-specific evaluation metrics and utilities
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.lpips_distances = []
        self.traditional_metrics = []
        self.human_preferences = []
    
    def compute_traditional_metrics(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
        """
        Compute traditional image similarity metrics
        
        Args:
            img1, img2: Input images [B, C, H, W]
            
        Returns:
            Dictionary with traditional metrics
        """
        # Ensure images are in [0, 1] range
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        
        # L1 distance
        l1_distance = F.l1_loss(img1, img2).item()
        
        # L2 distance (MSE)
        l2_distance = F.mse_loss(img1, img2).item()
        
        # PSNR
        mse = F.mse_loss(img1, img2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        
        # SSIM (simplified version)
        ssim = self._compute_ssim(img1, img2)
        
        return {
            'l1_distance': l1_distance,
            'l2_distance': l2_distance,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """Simplified SSIM computation"""
        # Convert to grayscale for simplicity
        if img1.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114]).to(img1.device).view(1, 3, 1, 1)
            img1_gray = (img1 * weights).sum(dim=1, keepdim=True)
            img2_gray = (img2 * weights).sum(dim=1, keepdim=True)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Simple correlation-based SSIM approximation
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
    
    def evaluate_correlation_with_human_perception(self, 
                                                  lpips_distances: List[float],
                                                  traditional_metrics: List[Dict[str, float]],
                                                  human_preferences: List[int]) -> Dict:
        """
        Evaluate correlation between different metrics and human perception
        
        Args:
            lpips_distances: List of LPIPS distances
            traditional_metrics: List of traditional metric dictionaries
            human_preferences: List of human preference labels (0 or 1)
            
        Returns:
            Correlation analysis results
        """
        lpips_array = np.array(lpips_distances)
        preferences_array = np.array(human_preferences)
        
        # Extract traditional metrics
        l1_distances = [m['l1_distance'] for m in traditional_metrics]
        l2_distances = [m['l2_distance'] for m in traditional_metrics]
        psnr_values = [m['psnr'] for m in traditional_metrics]
        ssim_values = [m['ssim'] for m in traditional_metrics]
        
        # Compute correlations with human preferences
        correlations = {
            'lpips_correlation': np.corrcoef(lpips_array, preferences_array)[0, 1],
            'l1_correlation': np.corrcoef(l1_distances, preferences_array)[0, 1],
            'l2_correlation': np.corrcoef(l2_distances, preferences_array)[0, 1],
            'psnr_correlation': np.corrcoef(psnr_values, preferences_array)[0, 1],
            'ssim_correlation': np.corrcoef(ssim_values, preferences_array)[0, 1]
        }
        
        # Rank correlations
        rank_correlations = {}
        for metric_name, correlation in correlations.items():
            rank_correlations[f'{metric_name}_rank'] = correlation
        
        return {
            'pearson_correlations': correlations,
            'rank_correlations': rank_correlations,
            'best_traditional_metric': max(correlations.items(), key=lambda x: abs(x[1]) if x[0] != 'lpips_correlation' else 0),
            'lpips_improvement': correlations['lpips_correlation'] - max([v for k, v in correlations.items() if k != 'lpips_correlation'])
        }
    
    def create_perceptual_distance_analysis(self, 
                                          distances: Dict[str, List[float]]) -> Dict:
        """
        Analyze perceptual distance distributions
        
        Args:
            distances: Dictionary with metric names and distance lists
            
        Returns:
            Statistical analysis of distances
        """
        analysis = {}
        
        for metric_name, distance_list in distances.items():
            distances_array = np.array(distance_list)
            
            analysis[metric_name] = {
                'mean': distances_array.mean(),
                'std': distances_array.std(),
                'min': distances_array.min(),
                'max': distances_array.max(),
                'median': np.median(distances_array),
                'percentile_25': np.percentile(distances_array, 25),
                'percentile_75': np.percentile(distances_array, 75),
                'histogram': np.histogram(distances_array, bins=20)[0].tolist()
            }
        
        return analysis


class ModelComparisonFramework:
    """
    Framework for comparing different models across multiple metrics
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, model: nn.Module):
        """Add a model to the comparison"""
        self.models[name] = model.to(self.device)
        self.models[name].eval()
    
    def run_comprehensive_comparison(self, 
                                   test_loader: torch.utils.data.DataLoader,
                                   include_efficiency: bool = True,
                                   include_accuracy: bool = True) -> Dict:
        """
        Run comprehensive comparison across all models
        
        Args:
            test_loader: Data loader for evaluation
            include_efficiency: Whether to include efficiency metrics
            include_accuracy: Whether to include accuracy metrics
            
        Returns:
            Comprehensive comparison results
        """
        comparison_results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            model_results = {}
            
            # Accuracy evaluation
            if include_accuracy:
                print(f"  Computing accuracy metrics for {model_name}...")
                accuracy_results = self._evaluate_model_accuracy(model, test_loader)
                model_results['accuracy'] = accuracy_results
            
            # Efficiency evaluation
            if include_efficiency:
                print(f"  Computing efficiency metrics for {model_name}...")
                efficiency_results = self._evaluate_model_efficiency(model)
                model_results['efficiency'] = efficiency_results
            
            # Model complexity
            complexity_results = self._analyze_model_complexity(model)
            model_results['complexity'] = complexity_results
            
            comparison_results[model_name] = model_results
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(comparison_results)
        comparison_results['summary'] = comparison_summary
        
        return comparison_results
    
    def _evaluate_model_accuracy(self, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model accuracy"""
        metrics = ClassificationMetrics()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                metrics.update(output, target)
        
        accuracy = metrics.compute_accuracy()
        confidence = metrics.compute_confidence_metrics()
        
        return {
            'accuracy_metrics': accuracy,
            'confidence_metrics': confidence
        }
    
    def _evaluate_model_efficiency(self, model: nn.Module) -> Dict:
        """Evaluate model efficiency"""
        efficiency_metrics = EfficiencyMetrics(self.device)
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 8, 16]
        input_shape = (3, 224, 224)
        
        scaling_results = efficiency_metrics.measure_throughput_scaling(
            model, batch_sizes, input_shape
        )
        
        efficiency_analysis = efficiency_metrics.analyze_efficiency_trends(scaling_results)
        
        return {
            'scaling_results': scaling_results,
            'efficiency_analysis': efficiency_analysis
        }
    
    def _analyze_model_complexity(self, model: nn.Module) -> Dict:
        """Analyze model complexity"""
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_efficiency': total_params / 1e6  # Parameters in millions
        }
    
    def _generate_comparison_summary(self, results: Dict) -> Dict:
        """Generate comparison summary"""
        summary = {
            'model_rankings': {},
            'best_models': {},
            'trade_offs': {}
        }
        
        model_names = [name for name in results.keys() if name != 'summary']
        
        # Rank by accuracy
        if 'accuracy' in results[model_names[0]]:
            accuracy_scores = {name: results[name]['accuracy']['accuracy_metrics']['top1_accuracy'] 
                             for name in model_names}
            summary['model_rankings']['accuracy'] = sorted(accuracy_scores.items(), 
                                                         key=lambda x: x[1], reverse=True)
            summary['best_models']['accuracy'] = summary['model_rankings']['accuracy'][0][0]
        
        # Rank by efficiency (throughput)
        if 'efficiency' in results[model_names[0]]:
            throughput_scores = {}
            for name in model_names:
                scaling_results = results[name]['efficiency']['scaling_results']
                # Use batch size 8 throughput as reference
                if 8 in scaling_results:
                    throughput_scores[name] = scaling_results[8]['timing']['throughput_images_per_second']
            
            if throughput_scores:
                summary['model_rankings']['throughput'] = sorted(throughput_scores.items(), 
                                                               key=lambda x: x[1], reverse=True)
                summary['best_models']['throughput'] = summary['model_rankings']['throughput'][0][0]
        
        # Rank by parameter efficiency
        param_counts = {name: results[name]['complexity']['total_parameters'] for name in model_names}
        summary['model_rankings']['parameter_efficiency'] = sorted(param_counts.items(), 
                                                                  key=lambda x: x[1])
        summary['best_models']['parameter_efficiency'] = summary['model_rankings']['parameter_efficiency'][0][0]
        
        return summary
    
    def visualize_comparison(self, results: Dict, save_path: Optional[str] = None):
        """Create visualization of model comparison"""
        model_names = [name for name in results.keys() if name != 'summary']
        
        # Extract metrics for plotting
        accuracies = []
        throughputs = []
        param_counts = []
        model_sizes = []
        
        for name in model_names:
            if 'accuracy' in results[name]:
                accuracies.append(results[name]['accuracy']['accuracy_metrics']['top1_accuracy'])
            else:
                accuracies.append(0)
            
            if 'efficiency' in results[name] and 8 in results[name]['efficiency']['scaling_results']:
                throughputs.append(results[name]['efficiency']['scaling_results'][8]['timing']['throughput_images_per_second'])
            else:
                throughputs.append(0)
            
            param_counts.append(results[name]['complexity']['total_parameters'] / 1e6)  # In millions
            model_sizes.append(results[name]['complexity']['model_size_mb'])
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        ax1.bar(model_names, accuracies)
        ax1.set_title('Top-1 Accuracy Comparison')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax2.bar(model_names, throughputs)
        ax2.set_title('Throughput Comparison')
        ax2.set_ylabel('Images/Second')
        ax2.tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        ax3.bar(model_names, param_counts)
        ax3.set_title('Parameter Count Comparison')
        ax3.set_ylabel('Parameters (Millions)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Efficiency vs Accuracy scatter plot
        ax4.scatter(param_counts, accuracies, s=100)
        for i, name in enumerate(model_names):
            ax4.annotate(name, (param_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Parameters (Millions)')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy vs Model Size')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig


def main():
    """Demonstration of evaluation metrics"""
    print("=== Evaluation Metrics Demonstration ===\n")
    
    # Create dummy data for demonstration
    batch_size = 32
    num_classes = 10
    
    # Simulate model outputs and targets
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Classification metrics
    print("1. Classification Metrics")
    metrics = ClassificationMetrics(num_classes)
    metrics.update(outputs, targets)
    
    accuracy = metrics.compute_accuracy()
    print(f"Top-1 Accuracy: {accuracy['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {accuracy['top5_accuracy']:.2f}%")
    
    confidence = metrics.compute_confidence_metrics()
    print(f"Mean Confidence: {confidence['mean_confidence']:.3f}")
    
    # Traditional metrics comparison
    print("\n2. Traditional Metrics")
    lpips_metrics = LPIPSEvaluationMetrics()
    
    # Simulate two images
    img1 = torch.rand(1, 3, 224, 224)
    img2 = torch.rand(1, 3, 224, 224)
    
    traditional = lpips_metrics.compute_traditional_metrics(img1, img2)
    print(f"L1 Distance: {traditional['l1_distance']:.4f}")
    print(f"L2 Distance: {traditional['l2_distance']:.4f}")
    print(f"PSNR: {traditional['psnr']:.2f} dB")
    print(f"SSIM: {traditional['ssim']:.4f}")
    
    print("\n=== Evaluation metrics demonstration completed ===")


if __name__ == "__main__":
    main()