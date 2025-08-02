import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Sample Model for Error Analysis
class ErrorAnalysisModel(nn.Module):
    """Model for error analysis demonstration"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Error Analysis Tools
class ErrorAnalyzer:
    """Comprehensive error analysis for classification models"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def analyze_predictions(self, data_loader, class_names: List[str] = None) -> Dict[str, Any]:
        """Analyze model predictions and errors"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_confidences = []
        misclassified_indices = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                # Find misclassified samples
                misclassified = (predictions != targets).cpu().numpy()
                batch_start = batch_idx * data_loader.batch_size
                misclassified_indices.extend([
                    batch_start + i for i, is_wrong in enumerate(misclassified) if is_wrong
                ])
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        accuracy = (all_predictions == all_targets).mean()
        
        analysis_results = {
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'confidences': all_confidences,
            'misclassified_indices': misclassified_indices,
            'accuracy': accuracy,
            'num_samples': len(all_targets),
            'num_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(all_targets)
        }
        
        return analysis_results
    
    def create_confusion_matrix(self, analysis_results: Dict[str, Any], 
                              class_names: List[str] = None,
                              normalize: bool = False) -> np.ndarray:
        """Create and visualize confusion matrix"""
        
        predictions = analysis_results['predictions']
        targets = analysis_results['targets']
        
        # Create confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues')
        
        plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def analyze_confidence_errors(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationship between confidence and errors"""
        
        predictions = analysis_results['predictions']
        targets = analysis_results['targets']
        confidences = analysis_results['confidences']
        
        # Split into correct and incorrect predictions
        correct_mask = predictions == targets
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        # Calculate statistics
        confidence_analysis = {
            'correct_confidence_mean': np.mean(correct_confidences),
            'correct_confidence_std': np.std(correct_confidences),
            'incorrect_confidence_mean': np.mean(incorrect_confidences),
            'incorrect_confidence_std': np.std(incorrect_confidences),
            'confidence_gap': np.mean(correct_confidences) - np.mean(incorrect_confidences)
        }
        
        # Plot confidence distributions
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', density=True)
        plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Confidence vs accuracy plot
        plt.subplot(1, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
            if bin_mask.sum() > 0:
                bin_accuracy = (predictions[bin_mask] == targets[bin_mask]).mean()
                bin_accuracies.append(bin_accuracy)
                bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=6)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return confidence_analysis
    
    def per_class_analysis(self, analysis_results: Dict[str, Any],
                          class_names: List[str] = None) -> Dict[str, Any]:
        """Analyze errors per class"""
        
        predictions = analysis_results['predictions']
        targets = analysis_results['targets']
        
        num_classes = len(np.unique(targets))
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
        
        per_class_stats = {}
        
        for class_idx in range(num_classes):
            class_mask = targets == class_idx
            
            if class_mask.sum() > 0:
                class_predictions = predictions[class_mask]
                class_targets = targets[class_mask]
                
                # Calculate metrics
                precision = (class_predictions == class_idx).sum() / (predictions == class_idx).sum() if (predictions == class_idx).sum() > 0 else 0
                recall = (class_predictions == class_idx).sum() / class_mask.sum()
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                per_class_stats[class_names[class_idx]] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': class_mask.sum(),
                    'accuracy': (class_predictions == class_targets).mean()
                }
        
        # Plot per-class metrics
        class_names_list = list(per_class_stats.keys())
        precisions = [per_class_stats[name]['precision'] for name in class_names_list]
        recalls = [per_class_stats[name]['recall'] for name in class_names_list]
        f1_scores = [per_class_stats[name]['f1_score'] for name in class_names_list]
        
        x = np.arange(len(class_names_list))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.7)
        plt.bar(x, recalls, width, label='Recall', alpha=0.7)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, class_names_list, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('per_class_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return per_class_stats

class BiasAnalyzer:
    """Analyze model bias and fairness"""
    
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def analyze_subgroup_performance(self, data_loader, subgroup_labels: List[int],
                                   subgroup_names: List[str] = None) -> Dict[str, Any]:
        """Analyze performance across different subgroups"""
        
        if subgroup_names is None:
            subgroup_names = [f'Subgroup {i}' for i in range(len(np.unique(subgroup_labels)))]
        
        # Get predictions
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        subgroup_labels = np.array(subgroup_labels)
        
        # Analyze by subgroup
        subgroup_stats = {}
        
        for subgroup_idx, subgroup_name in enumerate(subgroup_names):
            subgroup_mask = subgroup_labels == subgroup_idx
            
            if subgroup_mask.sum() > 0:
                subgroup_predictions = all_predictions[subgroup_mask]
                subgroup_targets = all_targets[subgroup_mask]
                
                accuracy = (subgroup_predictions == subgroup_targets).mean()
                
                subgroup_stats[subgroup_name] = {
                    'accuracy': accuracy,
                    'size': subgroup_mask.sum(),
                    'proportion': subgroup_mask.mean()
                }
        
        # Plot subgroup comparison
        subgroup_names_list = list(subgroup_stats.keys())
        accuracies = [subgroup_stats[name]['accuracy'] for name in subgroup_names_list]
        sizes = [subgroup_stats[name]['size'] for name in subgroup_names_list]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars = ax1.bar(subgroup_names_list, accuracies, alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Subgroup')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Subgroup size
        ax2.pie(sizes, labels=subgroup_names_list, autopct='%1.1f%%')
        ax2.set_title('Subgroup Size Distribution')
        
        plt.tight_layout()
        plt.savefig('subgroup_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return subgroup_stats

if __name__ == "__main__":
    print("Error Analysis Tools")
    print("=" * 20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and data
    model = ErrorAnalysisModel(num_classes=10).to(device)
    
    # Create sample dataset
    sample_data = torch.randn(500, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (500,))
    sample_dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=32, shuffle=False)
    
    print("\n1. Basic Error Analysis")
    print("-" * 25)
    
    # Analyze predictions
    analyzer = ErrorAnalyzer(model, device)
    analysis_results = analyzer.analyze_predictions(sample_loader)
    
    print(f"Overall Accuracy: {analysis_results['accuracy']:.4f}")
    print(f"Error Rate: {analysis_results['error_rate']:.4f}")
    print(f"Number of Errors: {analysis_results['num_errors']}")
    
    # Create confusion matrix
    class_names = [f'Class {i}' for i in range(10)]
    cm = analyzer.create_confusion_matrix(analysis_results, class_names, normalize=True)
    
    print("\n2. Confidence Analysis")
    print("-" * 22)
    
    # Analyze confidence
    confidence_analysis = analyzer.analyze_confidence_errors(analysis_results)
    
    print(f"Correct Confidence Mean: {confidence_analysis['correct_confidence_mean']:.4f}")
    print(f"Incorrect Confidence Mean: {confidence_analysis['incorrect_confidence_mean']:.4f}")
    print(f"Confidence Gap: {confidence_analysis['confidence_gap']:.4f}")
    
    print("\n3. Per-Class Analysis")
    print("-" * 23)
    
    # Per-class analysis
    per_class_stats = analyzer.per_class_analysis(analysis_results, class_names)
    
    print("Per-Class Performance:")
    for class_name, stats in per_class_stats.items():
        print(f"  {class_name}: Precision={stats['precision']:.3f}, "
              f"Recall={stats['recall']:.3f}, F1={stats['f1_score']:.3f}")
    
    print("\n4. Bias Analysis")
    print("-" * 18)
    
    # Create synthetic subgroup labels
    subgroup_labels = np.random.randint(0, 3, len(sample_labels))
    subgroup_names = ['Group A', 'Group B', 'Group C']
    
    bias_analyzer = BiasAnalyzer(model, device)
    subgroup_stats = bias_analyzer.analyze_subgroup_performance(
        sample_loader, subgroup_labels, subgroup_names
    )
    
    print("Subgroup Performance:")
    for group_name, stats in subgroup_stats.items():
        print(f"  {group_name}: Accuracy={stats['accuracy']:.3f}, Size={stats['size']}")
    
    print("\nError analysis completed!")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - confidence_analysis.png")
    print("  - per_class_analysis.png")
    print("  - subgroup_analysis.png")