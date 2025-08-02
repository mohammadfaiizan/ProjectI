import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

# Sample Models for Validation
class ValidationTestCNN(nn.Module):
    """CNN for validation testing"""
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Cross-Validation
class CrossValidator:
    """Cross-validation utilities for PyTorch models"""
    
    def __init__(self, model_class, model_params: Dict[str, Any], device='cuda'):
        self.model_class = model_class
        self.model_params = model_params
        self.device = device
        self.cv_results = []
    
    def k_fold_validation(self, dataset, k_folds: int = 5, 
                         train_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        
        if train_params is None:
            train_params = {'epochs': 10, 'lr': 0.001, 'batch_size': 32}
        
        # Convert dataset to lists for indexing
        data_list = []
        label_list = []
        
        for i in range(len(dataset)):
            data, label = dataset[i]
            data_list.append(data)
            label_list.append(label)
        
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)
        
        # Create k-fold splits
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(data_tensor)):
            print(f"Training fold {fold + 1}/{k_folds}...")
            
            # Split data
            train_data = data_tensor[train_idx]
            train_labels = label_tensor[train_idx]
            val_data = data_tensor[val_idx]
            val_labels = label_tensor[val_idx]
            
            # Create datasets
            train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_params['batch_size'], shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=train_params['batch_size'], shuffle=False
            )
            
            # Create fresh model for this fold
            model = self.model_class(**self.model_params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            fold_result = self._train_and_validate(
                model, train_loader, val_loader, optimizer, criterion, 
                train_params['epochs'], fold
            )
            
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(fold_results)
        self.cv_results = fold_results
        
        return aggregated_results
    
    def stratified_k_fold_validation(self, dataset, k_folds: int = 5,
                                   train_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform stratified k-fold cross-validation"""
        
        if train_params is None:
            train_params = {'epochs': 10, 'lr': 0.001, 'batch_size': 32}
        
        # Extract data and labels
        data_list = []
        label_list = []
        
        for i in range(len(dataset)):
            data, label = dataset[i]
            data_list.append(data)
            label_list.append(label)
        
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)
        
        # Create stratified k-fold splits
        skfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skfold.split(data_tensor, label_tensor)):
            print(f"Training stratified fold {fold + 1}/{k_folds}...")
            
            # Split data
            train_data = data_tensor[train_idx]
            train_labels = label_tensor[train_idx]
            val_data = data_tensor[val_idx]
            val_labels = label_tensor[val_idx]
            
            # Create datasets
            train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_params['batch_size'], shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=train_params['batch_size'], shuffle=False
            )
            
            # Create fresh model for this fold
            model = self.model_class(**self.model_params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            fold_result = self._train_and_validate(
                model, train_loader, val_loader, optimizer, criterion,
                train_params['epochs'], fold
            )
            
            fold_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(fold_results)
        self.cv_results = fold_results
        
        return aggregated_results
    
    def _train_and_validate(self, model, train_loader, val_loader, 
                           optimizer, criterion, epochs: int, fold: int) -> Dict[str, Any]:
        """Train and validate model for one fold"""
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
        
        return {
            'fold': fold,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_accuracy': val_accuracies[-1],
            'best_val_accuracy': max(val_accuracies),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results"""
        
        final_accuracies = [result['final_val_accuracy'] for result in fold_results]
        best_accuracies = [result['best_val_accuracy'] for result in fold_results]
        final_losses = [result['final_val_loss'] for result in fold_results]
        
        return {
            'mean_final_accuracy': np.mean(final_accuracies),
            'std_final_accuracy': np.std(final_accuracies),
            'mean_best_accuracy': np.mean(best_accuracies),
            'std_best_accuracy': np.std(best_accuracies),
            'mean_final_loss': np.mean(final_losses),
            'std_final_loss': np.std(final_losses),
            'individual_results': fold_results
        }
    
    def plot_cv_results(self):
        """Plot cross-validation results"""
        
        if not self.cv_results:
            print("No CV results to plot")
            return
        
        n_folds = len(self.cv_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss curves
        for i, result in enumerate(self.cv_results):
            axes[0, 0].plot(result['train_losses'], label=f'Fold {i+1} Train', alpha=0.7)
            axes[0, 1].plot(result['val_losses'], label=f'Fold {i+1} Val', alpha=0.7)
        
        axes[0, 0].set_title('Training Loss by Fold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss by Fold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Validation accuracy curves
        for i, result in enumerate(self.cv_results):
            axes[1, 0].plot(result['val_accuracies'], label=f'Fold {i+1}', alpha=0.7)
        
        axes[1, 0].set_title('Validation Accuracy by Fold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final accuracy distribution
        final_accuracies = [result['final_val_accuracy'] for result in self.cv_results]
        axes[1, 1].bar(range(1, n_folds + 1), final_accuracies, alpha=0.7)
        axes[1, 1].set_title('Final Validation Accuracy by Fold')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add mean line
        mean_acc = np.mean(final_accuracies)
        axes[1, 1].axhline(y=mean_acc, color='red', linestyle='--', 
                          label=f'Mean: {mean_acc:.3f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# Hold-out Validation
class HoldoutValidator:
    """Hold-out validation utilities"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def train_val_test_split(self, dataset, train_ratio: float = 0.7, 
                           val_ratio: float = 0.15, test_ratio: float = 0.15,
                           random_state: int = 42) -> Tuple[Any, Any, Any]:
        """Split dataset into train, validation, and test sets"""
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Extract data and labels
        data_list = []
        label_list = []
        
        for i in range(len(dataset)):
            data, label = dataset[i]
            data_list.append(data)
            label_list.append(label)
        
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)
        
        # First split: separate test set
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            data_tensor, label_tensor, 
            test_size=test_ratio, 
            random_state=random_state,
            stratify=label_tensor
        )
        
        # Second split: separate train and validation
        val_size = val_ratio / (train_ratio + val_ratio)
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels,
            test_size=val_size,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        return train_dataset, val_dataset, test_dataset
    
    def validate_model(self, model, train_dataset, val_dataset, test_dataset,
                      train_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train and validate model with hold-out method"""
        
        if train_params is None:
            train_params = {'epochs': 20, 'lr': 0.001, 'batch_size': 32}
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_params['batch_size'], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=train_params['batch_size'], shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=train_params['batch_size'], shuffle=False
        )
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        # Training loop
        for epoch in range(train_params['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss, val_accuracy = self._evaluate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        test_loss, test_accuracy = self._evaluate_model(model, test_loader, criterion)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_accuracy': val_accuracies[-1]
        }
    
    def _evaluate_model(self, model, data_loader, criterion):
        """Evaluate model on given dataset"""
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy

# Learning Curves and Validation Curves
class ValidationCurveAnalyzer:
    """Analyze learning and validation curves"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def generate_learning_curves(self, model_class, model_params: Dict[str, Any],
                                dataset, train_sizes: List[float] = None,
                                cv_folds: int = 3) -> Dict[str, Any]:
        """Generate learning curves showing performance vs training set size"""
        
        if train_sizes is None:
            train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        # Convert dataset to tensors
        data_list = []
        label_list = []
        
        for i in range(len(dataset)):
            data, label = dataset[i]
            data_list.append(data)
            label_list.append(label)
        
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list)
        
        results = []
        
        for train_size in train_sizes:
            print(f"Training with {train_size*100}% of data...")
            
            size_results = []
            
            # Perform multiple runs for this training size
            for run in range(cv_folds):
                # Sample training data
                n_samples = int(len(dataset) * train_size)
                indices = torch.randperm(len(dataset))[:n_samples]
                
                train_data = data_tensor[indices]
                train_labels = label_tensor[indices]
                
                # Use remaining data for validation
                val_indices = torch.randperm(len(dataset))[n_samples:n_samples+500]  # Fixed val size
                val_data = data_tensor[val_indices]
                val_labels = label_tensor[val_indices]
                
                # Create datasets
                train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
                val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
                
                # Train model
                model = model_class(**model_params).to(self.device)
                result = self._train_for_learning_curve(model, train_dataset, val_dataset)
                size_results.append(result)
            
            # Average results for this training size
            avg_result = {
                'train_size': train_size,
                'train_accuracy': np.mean([r['train_accuracy'] for r in size_results]),
                'val_accuracy': np.mean([r['val_accuracy'] for r in size_results]),
                'train_accuracy_std': np.std([r['train_accuracy'] for r in size_results]),
                'val_accuracy_std': np.std([r['val_accuracy'] for r in size_results])
            }
            
            results.append(avg_result)
        
        return results
    
    def plot_learning_curves(self, learning_curve_results: List[Dict[str, Any]]):
        """Plot learning curves"""
        
        train_sizes = [r['train_size'] for r in learning_curve_results]
        train_accuracies = [r['train_accuracy'] for r in learning_curve_results]
        val_accuracies = [r['val_accuracy'] for r in learning_curve_results]
        train_stds = [r['train_accuracy_std'] for r in learning_curve_results]
        val_stds = [r['val_accuracy_std'] for r in learning_curve_results]
        
        plt.figure(figsize=(10, 6))
        
        # Plot with error bars
        plt.errorbar(train_sizes, train_accuracies, yerr=train_stds, 
                    label='Training Accuracy', marker='o', capsize=5)
        plt.errorbar(train_sizes, val_accuracies, yerr=val_stds,
                    label='Validation Accuracy', marker='s', capsize=5)
        
        plt.xlabel('Training Set Size (fraction)')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_validation_curves(self, model_class, model_params: Dict[str, Any],
                                 dataset, param_name: str, param_values: List[Any],
                                 cv_folds: int = 3) -> Dict[str, Any]:
        """Generate validation curves for hyperparameter tuning"""
        
        results = []
        
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}...")
            
            # Update model parameters
            current_model_params = model_params.copy()
            current_model_params[param_name] = param_value
            
            # Perform cross-validation for this parameter value
            validator = CrossValidator(model_class, current_model_params, self.device)
            cv_result = validator.k_fold_validation(dataset, k_folds=cv_folds,
                                                  train_params={'epochs': 10, 'lr': 0.001, 'batch_size': 32})
            
            results.append({
                'param_value': param_value,
                'mean_val_accuracy': cv_result['mean_final_accuracy'],
                'std_val_accuracy': cv_result['std_final_accuracy']
            })
        
        return results
    
    def plot_validation_curves(self, validation_curve_results: List[Dict[str, Any]], 
                             param_name: str):
        """Plot validation curves"""
        
        param_values = [r['param_value'] for r in validation_curve_results]
        mean_accuracies = [r['mean_val_accuracy'] for r in validation_curve_results]
        std_accuracies = [r['std_val_accuracy'] for r in validation_curve_results]
        
        plt.figure(figsize=(10, 6))
        
        plt.errorbar(param_values, mean_accuracies, yerr=std_accuracies,
                    marker='o', capsize=5, linewidth=2, markersize=6)
        
        plt.xlabel(f'{param_name}')
        plt.ylabel('Validation Accuracy')
        plt.title(f'Validation Curve for {param_name}')
        plt.grid(True, alpha=0.3)
        
        # Mark best parameter value
        best_idx = np.argmax(mean_accuracies)
        best_param = param_values[best_idx]
        best_accuracy = mean_accuracies[best_idx]
        
        plt.axvline(x=best_param, color='red', linestyle='--', alpha=0.7,
                   label=f'Best: {param_name}={best_param} (acc={best_accuracy:.3f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'validation_curve_{param_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return best_param, best_accuracy
    
    def _train_for_learning_curve(self, model, train_dataset, val_dataset):
        """Train model for learning curve analysis"""
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Train for fixed number of epochs
        for epoch in range(10):
            model.train()
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        train_accuracy = self._calculate_accuracy(model, train_loader)
        val_accuracy = self._calculate_accuracy(model, val_loader)
        
        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }
    
    def _calculate_accuracy(self, model, data_loader):
        """Calculate accuracy on dataset"""
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total

# Model Comparison and Statistical Testing
class ModelComparator:
    """Compare multiple models using statistical tests"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compare_models(self, model_configs: List[Dict[str, Any]], dataset,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Compare multiple models using cross-validation"""
        
        results = {}
        
        for config in model_configs:
            model_name = config['name']
            model_class = config['model_class']
            model_params = config['model_params']
            
            print(f"Evaluating {model_name}...")
            
            # Perform cross-validation
            validator = CrossValidator(model_class, model_params, self.device)
            cv_result = validator.k_fold_validation(dataset, k_folds=cv_folds)
            
            results[model_name] = cv_result
        
        return results
    
    def statistical_significance_test(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests between models"""
        
        from scipy import stats
        
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        # Extract individual fold results
        model_accuracies = {}
        for name, result in model_results.items():
            fold_results = result['individual_results']
            accuracies = [fold['final_val_accuracy'] for fold in fold_results]
            model_accuracies[name] = accuracies
        
        # Pairwise t-tests
        pairwise_results = {}
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1 = model_names[i]
                model2 = model_names[j]
                
                acc1 = model_accuracies[model1]
                acc2 = model_accuracies[model2]
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(acc1, acc2)
                
                pairwise_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'model1_mean': np.mean(acc1),
                    'model2_mean': np.mean(acc2)
                }
        
        return pairwise_results
    
    def plot_model_comparison(self, model_results: Dict[str, Any]):
        """Plot model comparison results"""
        
        model_names = list(model_results.keys())
        
        # Extract data for plotting
        mean_accuracies = [model_results[name]['mean_final_accuracy'] for name in model_names]
        std_accuracies = [model_results[name]['std_final_accuracy'] for name in model_names]
        
        # Individual fold accuracies for box plot
        fold_accuracies = []
        for name in model_names:
            fold_results = model_results[name]['individual_results']
            accuracies = [fold['final_val_accuracy'] for fold in fold_results]
            fold_accuracies.append(accuracies)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot with error bars
        bars = ax1.bar(model_names, mean_accuracies, yerr=std_accuracies, 
                      capsize=5, alpha=0.7)
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Model Comparison (Mean ± Std)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_acc, std_acc in zip(bars, mean_accuracies, std_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_acc,
                    f'{mean_acc:.3f}±{std_acc:.3f}', ha='center', va='bottom')
        
        # Box plot
        ax2.boxplot(fold_accuracies, labels=model_names)
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Model Comparison (Distribution)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("Model Validation Techniques")
    print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample dataset
    sample_data = torch.randn(1000, 3, 32, 32)
    sample_labels = torch.randint(0, 10, (1000,))
    sample_dataset = torch.utils.data.TensorDataset(sample_data, sample_labels)
    
    print("\n1. Cross-Validation")
    print("-" * 22)
    
    # Cross-validation
    validator = CrossValidator(ValidationTestCNN, {'num_classes': 10, 'dropout_rate': 0.3}, device)
    
    print("Performing 5-fold cross-validation...")
    cv_results = validator.k_fold_validation(sample_dataset, k_folds=5,
                                           train_params={'epochs': 5, 'lr': 0.001, 'batch_size': 32})
    
    print("Cross-Validation Results:")
    print(f"Mean Final Accuracy: {cv_results['mean_final_accuracy']:.4f} ± {cv_results['std_final_accuracy']:.4f}")
    print(f"Mean Best Accuracy: {cv_results['mean_best_accuracy']:.4f} ± {cv_results['std_best_accuracy']:.4f}")
    print(f"Mean Final Loss: {cv_results['mean_final_loss']:.4f} ± {cv_results['std_final_loss']:.4f}")
    
    # Plot CV results
    validator.plot_cv_results()
    
    print("\n2. Stratified Cross-Validation")
    print("-" * 35)
    
    # Stratified cross-validation
    print("Performing stratified 5-fold cross-validation...")
    stratified_cv_results = validator.stratified_k_fold_validation(sample_dataset, k_folds=5,
                                                                 train_params={'epochs': 5, 'lr': 0.001, 'batch_size': 32})
    
    print("Stratified Cross-Validation Results:")
    print(f"Mean Final Accuracy: {stratified_cv_results['mean_final_accuracy']:.4f} ± {stratified_cv_results['std_final_accuracy']:.4f}")
    print(f"Mean Best Accuracy: {stratified_cv_results['mean_best_accuracy']:.4f} ± {stratified_cv_results['std_best_accuracy']:.4f}")
    
    print("\n3. Hold-out Validation")
    print("-" * 27)
    
    # Hold-out validation
    holdout_validator = HoldoutValidator(device)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = holdout_validator.train_val_test_split(
        sample_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Train and validate
    model = ValidationTestCNN(num_classes=10, dropout_rate=0.3).to(device)
    holdout_results = holdout_validator.validate_model(
        model, train_dataset, val_dataset, test_dataset,
        train_params={'epochs': 10, 'lr': 0.001, 'batch_size': 32}
    )
    
    print("Hold-out Validation Results:")
    print(f"Best Validation Accuracy: {holdout_results['best_val_accuracy']:.4f}")
    print(f"Final Validation Accuracy: {holdout_results['final_val_accuracy']:.4f}")
    print(f"Test Accuracy: {holdout_results['test_accuracy']:.4f}")
    
    # Plot training curves
    epochs = range(len(holdout_results['train_losses']))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, holdout_results['train_losses'], label='Training Loss')
    plt.plot(epochs, holdout_results['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, holdout_results['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('holdout_validation_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n4. Learning Curves")
    print("-" * 20)
    
    # Learning curves
    curve_analyzer = ValidationCurveAnalyzer(device)
    
    print("Generating learning curves...")
    learning_curves = curve_analyzer.generate_learning_curves(
        ValidationTestCNN, {'num_classes': 10, 'dropout_rate': 0.3},
        sample_dataset, train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv_folds=3
    )
    
    curve_analyzer.plot_learning_curves(learning_curves)
    
    print("Learning Curve Analysis:")
    for result in learning_curves:
        train_size = result['train_size']
        train_acc = result['train_accuracy']
        val_acc = result['val_accuracy']
        print(f"  {train_size*100:3.0f}% data: Train={train_acc:.3f}, Val={val_acc:.3f}")
    
    print("\n5. Validation Curves")
    print("-" * 23)
    
    # Validation curves for dropout rate
    print("Generating validation curves for dropout rate...")
    dropout_validation_curves = curve_analyzer.generate_validation_curves(
        ValidationTestCNN, {'num_classes': 10},
        sample_dataset, 'dropout_rate', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], cv_folds=3
    )
    
    best_dropout, best_dropout_acc = curve_analyzer.plot_validation_curves(
        dropout_validation_curves, 'dropout_rate'
    )
    
    print(f"Best dropout rate: {best_dropout} (accuracy: {best_dropout_acc:.4f})")
    
    print("\n6. Model Comparison")
    print("-" * 22)
    
    # Compare different models
    model_configs = [
        {
            'name': 'Low Dropout CNN',
            'model_class': ValidationTestCNN,
            'model_params': {'num_classes': 10, 'dropout_rate': 0.1}
        },
        {
            'name': 'Medium Dropout CNN',
            'model_class': ValidationTestCNN,
            'model_params': {'num_classes': 10, 'dropout_rate': 0.3}
        },
        {
            'name': 'High Dropout CNN',
            'model_class': ValidationTestCNN,
            'model_params': {'num_classes': 10, 'dropout_rate': 0.5}
        }
    ]
    
    comparator = ModelComparator(device)
    
    print("Comparing different models...")
    comparison_results = comparator.compare_models(model_configs, sample_dataset, cv_folds=3)
    
    print("\nModel Comparison Results:")
    print("-" * 30)
    for name, result in comparison_results.items():
        mean_acc = result['mean_final_accuracy']
        std_acc = result['std_final_accuracy']
        print(f"{name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Statistical significance testing
    significance_results = comparator.statistical_significance_test(comparison_results)
    
    print("\nStatistical Significance Tests:")
    print("-" * 35)
    for comparison, result in significance_results.items():
        p_value = result['p_value']
        significant = result['significant']
        model1_mean = result['model1_mean']
        model2_mean = result['model2_mean']
        
        print(f"{comparison}:")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {significant}")
        print(f"  Mean difference: {model1_mean - model2_mean:.4f}")
    
    # Plot model comparison
    comparator.plot_model_comparison(comparison_results)
    
    print("\n7. Validation Best Practices")
    print("-" * 35)
    
    best_practices = [
        "Use stratified splits to maintain class distribution",
        "Ensure validation set is representative of test conditions",
        "Use cross-validation for robust performance estimates",
        "Keep test set completely separate until final evaluation",
        "Monitor for overfitting using validation curves",
        "Use learning curves to diagnose bias/variance issues",
        "Apply statistical tests when comparing models",
        "Consider multiple evaluation metrics beyond accuracy",
        "Validate hyperparameters using separate validation set",
        "Use early stopping based on validation performance",
        "Ensure sufficient data for reliable validation estimates",
        "Account for computational cost vs validation reliability trade-offs"
    ]
    
    print("Validation Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Validation Summary")
    print("-" * 24)
    
    print("Validation Method Comparison:")
    print("-" * 30)
    print("Method                | Reliability | Computational Cost | Use Case")
    print("-" * 70)
    print("Hold-out             | Medium      | Low               | Quick validation")
    print("k-fold CV            | High        | Medium            | Standard practice")
    print("Stratified k-fold CV | High        | Medium            | Imbalanced datasets")
    print("Leave-one-out CV     | Very High   | Very High         | Small datasets")
    print("Time-series split    | High        | Medium            | Temporal data")
    
    print("\nValidation technique demonstration completed!")
    print("Generated files:")
    print("  - cross_validation_results.png")
    print("  - learning_curves.png")
    print("  - validation_curve_dropout_rate.png")
    print("  - holdout_validation_curves.png")
    print("  - model_comparison.png")