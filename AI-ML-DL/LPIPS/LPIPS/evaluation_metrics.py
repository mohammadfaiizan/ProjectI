"""
LPIPS Evaluation Metrics and Analysis
=====================================

Comprehensive evaluation framework for LPIPS with traditional metrics comparison,
correlation analysis, and visualization tools.

This module provides:
- Traditional image quality metrics (PSNR, SSIM, L1, L2)
- LPIPS evaluation and comparison
- Human perception correlation analysis
- Statistical significance testing
- Comprehensive visualization tools
- Perceptual quality benchmarking

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import stats
import sklearn.metrics as sk_metrics
from pathlib import Path
import json
import warnings
from tqdm import tqdm

from lpips_model import LPIPS


class TraditionalMetrics:
    """
    Traditional image quality metrics for comparison with LPIPS
    """
    
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
        """
        Calculate PSNR between two images
        
        Args:
            img1, img2: Input images [C, H, W]
            max_val: Maximum possible pixel value
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        
        psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr_val.item()
    
    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, 
             data_range: float = 1.0) -> float:
        """
        Calculate SSIM between two images
        
        Args:
            img1, img2: Input images [C, H, W]
            window_size: Size of sliding window
            data_range: Range of the data
            
        Returns:
            SSIM value
        """
        # Convert to grayscale if RGB
        if img1.shape[0] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114]).to(img1.device).view(3, 1, 1)
            img1_gray = (img1 * weights).sum(dim=0, keepdim=True)
            img2_gray = (img2 * weights).sum(dim=0, keepdim=True)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate means
        mu1 = img1_gray.mean()
        mu2 = img2_gray.mean()
        
        # Calculate variances and covariance
        sigma1_sq = ((img1_gray - mu1) ** 2).mean()
        sigma2_sq = ((img2_gray - mu2) ** 2).mean()
        sigma12 = ((img1_gray - mu1) * (img2_gray - mu2)).mean()
        
        # SSIM constants
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_val = numerator / denominator
        return ssim_val.item()
    
    @staticmethod
    def lpnorm(img1: torch.Tensor, img2: torch.Tensor, p: int = 2) -> float:
        """
        Calculate Lp norm between two images
        
        Args:
            img1, img2: Input images
            p: Norm order (1 for L1, 2 for L2)
            
        Returns:
            Lp norm value
        """
        if p == 1:
            return F.l1_loss(img1, img2).item()
        elif p == 2:
            return F.mse_loss(img1, img2).item()
        else:
            diff = torch.abs(img1 - img2)
            return torch.mean(diff ** p).item() ** (1.0 / p)
    
    @staticmethod
    def compute_all_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, float]:
        """
        Compute all traditional metrics
        
        Args:
            img1, img2: Input images
            
        Returns:
            Dictionary of metric values
        """
        return {
            'psnr': TraditionalMetrics.psnr(img1, img2),
            'ssim': TraditionalMetrics.ssim(img1, img2),
            'l1': TraditionalMetrics.lpnorm(img1, img2, p=1),
            'l2': TraditionalMetrics.lpnorm(img1, img2, p=2),
            'linf': torch.max(torch.abs(img1 - img2)).item()
        }


class LPIPSEvaluator:
    """
    Comprehensive evaluator for LPIPS models
    """
    
    def __init__(self, lpips_model: LPIPS, device: str = 'cuda'):
        """
        Initialize LPIPS evaluator
        
        Args:
            lpips_model: Trained LPIPS model
            device: Device for computation
        """
        self.lpips_model = lpips_model.to(device)
        self.device = device
        self.lpips_model.eval()
        
        self.traditional_metrics = TraditionalMetrics()
        
    def evaluate_image_pairs(self, 
                           image_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                           human_judgments: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Evaluate LPIPS on image pairs
        
        Args:
            image_pairs: List of (img1, img2) tuples
            human_judgments: Optional human perceptual similarity judgments
            
        Returns:
            Dictionary of evaluation results
        """
        lpips_distances = []
        traditional_results = {
            'psnr': [],
            'ssim': [],
            'l1': [],
            'l2': [],
            'linf': []
        }
        
        print("Evaluating image pairs...")
        
        with torch.no_grad():
            for img1, img2 in tqdm(image_pairs):
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # Ensure batch dimension
                if img1.dim() == 3:
                    img1 = img1.unsqueeze(0)
                if img2.dim() == 3:
                    img2 = img2.unsqueeze(0)
                
                # LPIPS distance
                lpips_dist = self.lpips_model(img1, img2).item()
                lpips_distances.append(lpips_dist)
                
                # Traditional metrics
                traditional = self.traditional_metrics.compute_all_metrics(
                    img1.squeeze(0), img2.squeeze(0)
                )
                
                for metric, value in traditional.items():
                    traditional_results[metric].append(value)
        
        results = {
            'lpips_distances': lpips_distances,
            'traditional_metrics': traditional_results,
            'num_pairs': len(image_pairs)
        }
        
        # Correlation analysis if human judgments provided
        if human_judgments is not None:
            correlations = self._compute_correlations(
                lpips_distances, traditional_results, human_judgments
            )
            results['correlations'] = correlations
        
        # Statistical analysis
        stats_analysis = self._compute_statistics(lpips_distances, traditional_results)
        results['statistics'] = stats_analysis
        
        return results
    
    def evaluate_2afc_dataset(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Evaluate LPIPS on 2AFC dataset
        
        Args:
            dataloader: DataLoader with (ref, img1, img2, judgment) samples
            
        Returns:
            Evaluation results including accuracy and correlation
        """
        all_predictions = []
        all_judgments = []
        all_lpips_distances = []
        all_traditional_metrics = {metric: [] for metric in ['psnr', 'ssim', 'l1', 'l2']}
        
        total_correct = 0
        total_samples = 0
        
        print("Evaluating 2AFC dataset...")
        
        with torch.no_grad():
            for ref_imgs, img1s, img2s, judgments in tqdm(dataloader):
                # Move to device
                ref_imgs = ref_imgs.to(self.device)
                img1s = img1s.to(self.device)
                img2s = img2s.to(self.device)
                judgments = judgments.to(self.device)
                
                batch_size = ref_imgs.shape[0]
                
                # Compute LPIPS distances
                dist1 = self.lpips_model(ref_imgs, img1s)
                dist2 = self.lpips_model(ref_imgs, img2s)
                
                # Make predictions (0 if img1 closer, 1 if img2 closer)
                predictions = (dist1 > dist2).long()
                
                # Count correct predictions
                correct = (predictions.squeeze() == judgments).sum().item()
                total_correct += correct
                total_samples += batch_size
                
                # Store for correlation analysis
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_judgments.extend(judgments.cpu().numpy().flatten())
                all_lpips_distances.extend(dist1.cpu().numpy().flatten())
                all_lpips_distances.extend(dist2.cpu().numpy().flatten())
                
                # Traditional metrics for comparison
                for i in range(batch_size):
                    # Metrics for ref vs img1
                    trad1 = self.traditional_metrics.compute_all_metrics(
                        ref_imgs[i], img1s[i]
                    )
                    # Metrics for ref vs img2
                    trad2 = self.traditional_metrics.compute_all_metrics(
                        ref_imgs[i], img2s[i]
                    )
                    
                    for metric in ['psnr', 'ssim', 'l1', 'l2']:
                        all_traditional_metrics[metric].extend([trad1[metric], trad2[metric]])
        
        # Calculate accuracy
        accuracy = total_correct / total_samples
        
        # Create human preference array for correlation
        human_preferences = []
        for i in range(0, len(all_lpips_distances), 2):
            judgment = all_judgments[i // 2] if i // 2 < len(all_judgments) else 0
            if judgment == 0:  # img1 preferred
                human_preferences.extend([1, 0])  # dist1 should be smaller
            else:  # img2 preferred
                human_preferences.extend([0, 1])  # dist2 should be smaller
        
        # Compute correlations
        correlations = {}
        if len(all_lpips_distances) > 0 and len(human_preferences) > 0:
            min_len = min(len(all_lpips_distances), len(human_preferences))
            
            # LPIPS correlation
            lpips_subset = all_lpips_distances[:min_len]
            human_subset = human_preferences[:min_len]
            
            if len(set(human_subset)) > 1:
                pearson_r, pearson_p = pearsonr(lpips_subset, human_subset)
                spearman_r, spearman_p = spearmanr(lpips_subset, human_subset)
                
                correlations['lpips'] = {
                    'pearson_r': abs(pearson_r),
                    'pearson_p': pearson_p,
                    'spearman_r': abs(spearman_r),
                    'spearman_p': spearman_p
                }
            
            # Traditional metrics correlations
            for metric_name, metric_values in all_traditional_metrics.items():
                if len(metric_values) >= min_len:
                    metric_subset = metric_values[:min_len]
                    
                    # For SSIM and PSNR, invert because higher is better
                    if metric_name in ['ssim', 'psnr']:
                        metric_subset = [-x for x in metric_subset]
                    
                    if len(set(human_subset)) > 1:
                        pearson_r, pearson_p = pearsonr(metric_subset, human_subset)
                        
                        correlations[metric_name] = {
                            'pearson_r': abs(pearson_r),
                            'pearson_p': pearson_p
                        }
        
        return {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'correlations': correlations,
            'predictions': all_predictions,
            'judgments': all_judgments,
            'lpips_distances': all_lpips_distances
        }
    
    def _compute_correlations(self, 
                            lpips_distances: List[float],
                            traditional_results: Dict[str, List[float]],
                            human_judgments: List[float]) -> Dict[str, Dict[str, float]]:
        """Compute correlations between metrics and human judgments"""
        correlations = {}
        
        # LPIPS correlation
        if len(lpips_distances) > 0 and len(human_judgments) > 0:
            pearson_r, pearson_p = pearsonr(lpips_distances, human_judgments)
            spearman_r, spearman_p = spearmanr(lpips_distances, human_judgments)
            
            correlations['lpips'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
        
        # Traditional metrics correlations
        for metric_name, metric_values in traditional_results.items():
            if len(metric_values) > 0 and len(human_judgments) > 0:
                # Adjust for metrics where higher is better
                if metric_name in ['ssim', 'psnr']:
                    adjusted_values = [-x for x in metric_values]
                else:
                    adjusted_values = metric_values
                
                pearson_r, pearson_p = pearsonr(adjusted_values, human_judgments)
                spearman_r, spearman_p = spearmanr(adjusted_values, human_judgments)
                
                correlations[metric_name] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
        
        return correlations
    
    def _compute_statistics(self, 
                          lpips_distances: List[float],
                          traditional_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistical analysis of metric distributions"""
        statistics = {}
        
        # LPIPS statistics
        lpips_array = np.array(lpips_distances)
        statistics['lpips'] = {
            'mean': lpips_array.mean(),
            'std': lpips_array.std(),
            'min': lpips_array.min(),
            'max': lpips_array.max(),
            'median': np.median(lpips_array),
            'q25': np.percentile(lpips_array, 25),
            'q75': np.percentile(lpips_array, 75)
        }
        
        # Traditional metrics statistics
        for metric_name, metric_values in traditional_results.items():
            metric_array = np.array(metric_values)
            statistics[metric_name] = {
                'mean': metric_array.mean(),
                'std': metric_array.std(),
                'min': metric_array.min(),
                'max': metric_array.max(),
                'median': np.median(metric_array),
                'q25': np.percentile(metric_array, 25),
                'q75': np.percentile(metric_array, 75)
            }
        
        return statistics
    
    def compare_architectures(self, 
                            architectures: Dict[str, LPIPS],
                            dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Compare different LPIPS architectures
        
        Args:
            architectures: Dictionary of {name: LPIPS_model}
            dataloader: Evaluation dataloader
            
        Returns:
            Comparison results
        """
        comparison_results = {}
        
        for arch_name, model in architectures.items():
            print(f"Evaluating {arch_name}...")
            
            # Temporarily switch model
            original_model = self.lpips_model
            self.lpips_model = model.to(self.device)
            self.lpips_model.eval()
            
            # Evaluate
            results = self.evaluate_2afc_dataset(dataloader)
            comparison_results[arch_name] = results
            
            # Restore original model
            self.lpips_model = original_model
        
        # Create summary comparison
        summary = {
            'accuracy_ranking': [],
            'correlation_ranking': [],
            'best_overall': None
        }
        
        # Rank by accuracy
        accuracy_scores = {name: results['accuracy'] for name, results in comparison_results.items()}
        summary['accuracy_ranking'] = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Rank by correlation (use LPIPS Pearson correlation)
        correlation_scores = {}
        for name, results in comparison_results.items():
            if 'correlations' in results and 'lpips' in results['correlations']:
                correlation_scores[name] = results['correlations']['lpips']['pearson_r']
            else:
                correlation_scores[name] = 0.0
        
        summary['correlation_ranking'] = sorted(correlation_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Determine best overall (weighted combination)
        overall_scores = {}
        for name in comparison_results.keys():
            acc_score = accuracy_scores[name]
            corr_score = correlation_scores[name]
            overall_scores[name] = 0.5 * acc_score + 0.5 * corr_score
        
        summary['best_overall'] = max(overall_scores.items(), key=lambda x: x[1])[0]
        
        comparison_results['summary'] = summary
        
        return comparison_results


class LPIPSVisualizer:
    """
    Visualization tools for LPIPS evaluation results
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_correlation_comparison(self, 
                                  correlations: Dict[str, Dict[str, float]], 
                                  save_path: Optional[str] = None):
        """
        Plot correlation comparison between LPIPS and traditional metrics
        
        Args:
            correlations: Dictionary of correlation results
            save_path: Optional path to save plot
        """
        metrics = list(correlations.keys())
        pearson_correlations = [correlations[metric].get('pearson_r', 0) for metric in metrics]
        spearman_correlations = [correlations[metric].get('spearman_r', 0) for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, pearson_correlations, width, label='Pearson', alpha=0.8)
        bars2 = ax.bar(x + width/2, spearman_correlations, width, label='Spearman', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Correlation with Human Judgment')
        ax.set_title('Correlation Comparison: LPIPS vs Traditional Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distance_distributions(self, 
                                  lpips_distances: List[float],
                                  traditional_metrics: Dict[str, List[float]],
                                  save_path: Optional[str] = None):
        """
        Plot distribution of distance values for different metrics
        
        Args:
            lpips_distances: LPIPS distance values
            traditional_metrics: Traditional metric values
            save_path: Optional path to save plot
        """
        num_metrics = len(traditional_metrics) + 1
        fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        # Plot LPIPS distribution
        axes[0].hist(lpips_distances, bins=50, alpha=0.7, density=True, color='red')
        axes[0].set_title('LPIPS Distance Distribution')
        axes[0].set_xlabel('Distance')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)
        
        # Plot traditional metrics distributions
        for i, (metric_name, metric_values) in enumerate(traditional_metrics.items(), 1):
            if i < len(axes):
                axes[i].hist(metric_values, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f'{metric_name.upper()} Distribution')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_scatter_correlations(self, 
                                lpips_distances: List[float],
                                human_judgments: List[float],
                                traditional_metrics: Dict[str, List[float]],
                                save_path: Optional[str] = None):
        """
        Plot scatter plots showing correlations with human judgments
        
        Args:
            lpips_distances: LPIPS distance values
            human_judgments: Human judgment values
            traditional_metrics: Traditional metric values
            save_path: Optional path to save plot
        """
        num_metrics = len(traditional_metrics) + 1
        fig, axes = plt.subplots(2, (num_metrics + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        # LPIPS scatter plot
        axes[0].scatter(lpips_distances, human_judgments, alpha=0.6, color='red')
        correlation, _ = pearsonr(lpips_distances, human_judgments)
        axes[0].set_title(f'LPIPS vs Human Judgment\n(r = {correlation:.3f})')
        axes[0].set_xlabel('LPIPS Distance')
        axes[0].set_ylabel('Human Judgment')
        axes[0].grid(True, alpha=0.3)
        
        # Traditional metrics scatter plots
        for i, (metric_name, metric_values) in enumerate(traditional_metrics.items(), 1):
            if i < len(axes):
                # Adjust for metrics where higher is better
                if metric_name in ['ssim', 'psnr']:
                    adjusted_values = [-x for x in metric_values]
                else:
                    adjusted_values = metric_values
                
                axes[i].scatter(adjusted_values, human_judgments, alpha=0.6)
                correlation, _ = pearsonr(adjusted_values, human_judgments)
                axes[i].set_title(f'{metric_name.upper()} vs Human Judgment\n(r = {correlation:.3f})')
                axes[i].set_xlabel(f'{metric_name.upper()} {"(inverted)" if metric_name in ["ssim", "psnr"] else ""}')
                axes[i].set_ylabel('Human Judgment')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_architecture_comparison(self, 
                                   comparison_results: Dict[str, Any],
                                   save_path: Optional[str] = None):
        """
        Plot comparison between different LPIPS architectures
        
        Args:
            comparison_results: Results from compare_architectures
            save_path: Optional path to save plot
        """
        architectures = [name for name in comparison_results.keys() if name != 'summary']
        
        # Extract metrics
        accuracies = []
        correlations = []
        
        for arch in architectures:
            results = comparison_results[arch]
            accuracies.append(results['accuracy'])
            
            if 'correlations' in results and 'lpips' in results['correlations']:
                correlations.append(results['correlations']['lpips']['pearson_r'])
            else:
                correlations.append(0.0)
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        bars1 = ax1.bar(architectures, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('2AFC Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Correlation comparison
        bars2 = ax2.bar(architectures, correlations, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Human Correlation Comparison')
        ax2.set_ylabel('Pearson Correlation')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, corr in zip(bars2, correlations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Combined scatter plot
        ax3.scatter(accuracies, correlations, s=200, c=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        
        for i, arch in enumerate(architectures):
            ax3.annotate(arch, (accuracies[i], correlations[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax3.set_xlabel('2AFC Accuracy')
        ax3.set_ylabel('Human Correlation')
        ax3.set_title('Accuracy vs Correlation Trade-off')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_evaluation_report(evaluation_results: Dict[str, Any], 
                           output_path: str = 'lpips_evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report
    
    Args:
        evaluation_results: Results from evaluation
        output_path: Path to save report
    """
    # Prepare report data
    report = {
        'evaluation_summary': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': evaluation_results.get('total_samples', 0),
            'accuracy': evaluation_results.get('accuracy', 0.0)
        },
        'correlation_analysis': evaluation_results.get('correlations', {}),
        'statistical_analysis': evaluation_results.get('statistics', {}),
        'key_findings': []
    }
    
    # Generate key findings
    if 'correlations' in evaluation_results:
        correlations = evaluation_results['correlations']
        
        # Find best performing metric
        best_metric = None
        best_correlation = 0.0
        
        for metric, corr_data in correlations.items():
            pearson_r = corr_data.get('pearson_r', 0)
            if abs(pearson_r) > best_correlation:
                best_correlation = abs(pearson_r)
                best_metric = metric
        
        if best_metric:
            report['key_findings'].append(
                f"Best performing metric: {best_metric.upper()} with correlation {best_correlation:.3f}"
            )
        
        # LPIPS performance
        if 'lpips' in correlations:
            lpips_corr = correlations['lpips'].get('pearson_r', 0)
            report['key_findings'].append(
                f"LPIPS correlation with human judgment: {abs(lpips_corr):.3f}"
            )
        
        # Compare with traditional metrics
        traditional_correlations = {k: v for k, v in correlations.items() if k != 'lpips'}
        if traditional_correlations:
            best_traditional = max(traditional_correlations.items(), 
                                 key=lambda x: abs(x[1].get('pearson_r', 0)))
            
            best_trad_corr = abs(best_traditional[1].get('pearson_r', 0))
            report['key_findings'].append(
                f"Best traditional metric: {best_traditional[0].upper()} with correlation {best_trad_corr:.3f}"
            )
            
            if 'lpips' in correlations:
                improvement = abs(lpips_corr) - best_trad_corr
                report['key_findings'].append(
                    f"LPIPS improvement over best traditional metric: {improvement:.3f}"
                )
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Evaluation report saved to {output_path}")


def main():
    """Demonstration of LPIPS evaluation metrics"""
    print("=" * 60)
    print("LPIPS Evaluation Metrics Demonstration")
    print("=" * 60)
    
    # Create dummy data for demonstration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple LPIPS model for testing
    from lpips_model import create_lpips_model
    lpips_model = create_lpips_model('vgg', pretrained=True)
    
    # Create evaluator
    evaluator = LPIPSEvaluator(lpips_model, device)
    
    # Generate test image pairs
    num_pairs = 50
    image_pairs = []
    human_judgments = []
    
    for i in range(num_pairs):
        img1 = torch.rand(3, 224, 224)
        img2 = torch.rand(3, 224, 224)
        
        # Simulate human judgment (closer images should have higher similarity)
        similarity = np.random.random()
        
        image_pairs.append((img1, img2))
        human_judgments.append(similarity)
    
    # Evaluate
    print("Running evaluation...")
    results = evaluator.evaluate_image_pairs(image_pairs, human_judgments)
    
    print(f"Evaluated {results['num_pairs']} image pairs")
    
    if 'correlations' in results:
        print("\nCorrelation Results:")
        for metric, corr_data in results['correlations'].items():
            print(f"  {metric.upper()}: {corr_data['pearson_r']:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = LPIPSVisualizer()
    
    if 'correlations' in results:
        visualizer.plot_correlation_comparison(results['correlations'])
    
    visualizer.plot_distance_distributions(
        results['lpips_distances'], 
        results['traditional_metrics']
    )
    
    # Create evaluation report
    create_evaluation_report(results)
    
    print("\nEvaluation demonstration complete!")


if __name__ == "__main__":
    main()