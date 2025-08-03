LPIPS Interview Questions & Answers - Training Methodology
============================================================

This file contains questions about LPIPS training methodology, dataset preparation,
optimization strategies, and training best practices.

============================================================

Q1: What are the key considerations when preparing a JND dataset for LPIPS training?

A1: JND dataset preparation involves several critical considerations:

**Data Quality Requirements**:
- Minimum 10k-100k triplets for robust training
- Balanced distribution across different image categories
- High-quality human annotations with consistent instructions
- Multiple annotators per triplet to reduce noise (ideally 3-5 annotators)

**Image Characteristics**:
- Diverse content: natural images, faces, objects, textures
- Various distortion types: compression, blur, noise, geometric transforms
- Multiple quality levels to span perceptual difference ranges
- Consistent image resolution (typically 256x256 or higher)

**Annotation Protocol**:
- Clear 2AFC instructions: "Which image is more similar to the reference?"
- Balanced judgment distribution (roughly 50/50 split)
- Quality control: remove inconsistent annotators or ambiguous triplets
- Validation set with expert annotations for reliable evaluation

**Data Format Standardization**:
- Triplet structure: (reference, comparison1, comparison2, judgment)
- Consistent file formats and naming conventions
- Metadata inclusion: distortion type, quality level, annotator info
- Train/validation/test splits with no image overlap

============================================================

Q2: How do you handle imbalanced datasets in LPIPS training?

A2: Imbalanced datasets can significantly impact LPIPS performance:

**Detection of Imbalance**:
```python
def analyze_dataset_balance(dataset):
    judgments = [sample['judgment'] for sample in dataset]
    class_0_count = sum(1 for j in judgments if j == 0)
    class_1_count = sum(1 for j in judgments if j == 1)
    
    imbalance_ratio = max(class_0_count, class_1_count) / min(class_0_count, class_1_count)
    
    return {
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': imbalance_ratio > 1.5
    }
```

**Mitigation Strategies**:

1. **Weighted Loss Function**:
```python
def create_weighted_loss(dataset):
    judgments = [sample['judgment'] for sample in dataset]
    pos_weight = torch.tensor([judgments.count(0) / judgments.count(1)])
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

2. **Data Resampling**:
```python
class BalancedJNDDataset(Dataset):
    def __init__(self, original_dataset, balance_method='oversample'):
        self.original_dataset = original_dataset
        self.balanced_indices = self._create_balanced_indices(balance_method)
    
    def _create_balanced_indices(self, method):
        class_0_indices = [i for i, sample in enumerate(self.original_dataset) 
                          if sample['judgment'] == 0]
        class_1_indices = [i for i, sample in enumerate(self.original_dataset) 
                          if sample['judgment'] == 1]
        
        if method == 'oversample':
            # Oversample minority class
            min_class = min(len(class_0_indices), len(class_1_indices))
            max_class = max(len(class_0_indices), len(class_1_indices))
            
            if len(class_0_indices) < len(class_1_indices):
                class_0_indices = np.random.choice(class_0_indices, max_class, replace=True)
            else:
                class_1_indices = np.random.choice(class_1_indices, max_class, replace=True)
        
        return np.concatenate([class_0_indices, class_1_indices])
```

3. **Stratified Sampling**:
```python
from sklearn.model_selection import train_test_split

def create_stratified_splits(dataset, test_size=0.2, val_size=0.1):
    judgments = [sample['judgment'] for sample in dataset]
    indices = list(range(len(dataset)))
    
    # First split: train+val vs test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=judgments, random_state=42
    )
    
    # Second split: train vs val
    train_val_judgments = [judgments[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size/(1-test_size), 
        stratify=train_val_judgments, random_state=42
    )
    
    return train_indices, val_indices, test_indices
```

============================================================

Q3: What are effective learning rate scheduling strategies for LPIPS training?

A3: Learning rate scheduling is crucial for LPIPS convergence:

**Recommended Schedules**:

1. **Cosine Annealing** (Most Effective):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-7
)

# Warm restart variant
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-7
)
```

2. **Plateau-based Scheduling**:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, 
    threshold=0.001, min_lr=1e-7
)

# Use validation accuracy as metric
for epoch in range(num_epochs):
    val_accuracy = validate_model()
    scheduler.step(val_accuracy)
```

3. **Multi-step Scheduling**:
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1
)
```

4. **Custom Adaptive Scheduling**:
```python
class AdaptiveLPIPSScheduler:
    def __init__(self, optimizer, initial_lr=1e-4):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.best_correlation = 0.0
        self.patience_counter = 0
        self.patience = 15
        
    def step(self, val_correlation):
        if val_correlation > self.best_correlation:
            self.best_correlation = val_correlation
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Reduced LR to {param_group['lr']}")
                
                self.patience_counter = 0
                self.patience = min(self.patience + 5, 25)  # Increase patience
```

**Learning Rate Range Selection**:
- Initial LR: 1e-4 to 1e-3 (linear layers only)
- Final LR: 1e-7 to 1e-6
- Use learning rate finder for optimal range discovery

============================================================

Q4: How do you implement effective data augmentation strategies for LPIPS training?

A4: Data augmentation for LPIPS requires careful consideration of perceptual relationships:

**Safe Augmentations** (Preserve perceptual relationships):
```python
class PerceptuallyConsistentAugmentation:
    def __init__(self):
        # Augmentations that maintain relative perceptual similarities
        self.safe_transforms = {
            'geometric': transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
            ]),
            'photometric': transforms.Compose([
                transforms.ColorJitter(brightness=0.05, contrast=0.05, 
                                     saturation=0.05, hue=0.02),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))
                ], p=0.3)
            ]),
            'crop': transforms.RandomResizedCrop(224, scale=(0.95, 1.0), ratio=(0.98, 1.02))
        }
    
    def augment_triplet_consistently(self, ref_img, img1, img2):
        """Apply same augmentation to all images in triplet"""
        # Choose random augmentation
        aug_type = random.choice(['geometric', 'photometric', 'crop'])
        transform = self.safe_transforms[aug_type]
        
        # Set same random seed for consistent transformation
        seed = random.randint(0, 2**32-1)
        
        torch.manual_seed(seed)
        aug_ref = transform(ref_img)
        torch.manual_seed(seed)
        aug_img1 = transform(img1)
        torch.manual_seed(seed)
        aug_img2 = transform(img2)
        
        return aug_ref, aug_img1, aug_img2
```

**Advanced Augmentation Strategies**:
```python
class AdaptiveAugmentation:
    def __init__(self, difficulty_schedule=True):
        self.difficulty_schedule = difficulty_schedule
        self.current_epoch = 0
        
    def get_augmentation_strength(self):
        """Gradually increase augmentation strength during training"""
        if not self.difficulty_schedule:
            return 1.0
        
        # Start with mild augmentations, increase over time
        max_epochs = 100
        strength = min(self.current_epoch / max_epochs, 1.0)
        return 0.3 + 0.7 * strength  # Range: 0.3 to 1.0
    
    def create_epoch_augmentation(self, epoch):
        """Create augmentation pipeline for specific epoch"""
        self.current_epoch = epoch
        strength = self.get_augmentation_strength()
        
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5 * strength),
            transforms.ColorJitter(
                brightness=0.1 * strength,
                contrast=0.1 * strength,
                saturation=0.1 * strength,
                hue=0.05 * strength
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5 * strength))
            ], p=0.2 * strength)
        ])
```

**Augmentation Validation**:
```python
def validate_augmentation_quality(original_triplet, augmented_triplet, lpips_model):
    """Ensure augmentation doesn't change perceptual relationships significantly"""
    ref_orig, img1_orig, img2_orig, judgment = original_triplet
    ref_aug, img1_aug, img2_aug, _ = augmented_triplet
    
    # Compute original distances
    dist1_orig = lpips_model(ref_orig.unsqueeze(0), img1_orig.unsqueeze(0))
    dist2_orig = lpips_model(ref_orig.unsqueeze(0), img2_orig.unsqueeze(0))
    
    # Compute augmented distances
    dist1_aug = lpips_model(ref_aug.unsqueeze(0), img1_aug.unsqueeze(0))
    dist2_aug = lpips_model(ref_aug.unsqueeze(0), img2_aug.unsqueeze(0))
    
    # Check if relative ordering is preserved
    orig_preference = 0 if dist1_orig < dist2_orig else 1
    aug_preference = 0 if dist1_aug < dist2_aug else 1
    
    return orig_preference == aug_preference  # True if ordering preserved
```

============================================================

Q5: What are the best practices for monitoring and debugging LPIPS training?

A5: Comprehensive monitoring is essential for successful LPIPS training:

**Key Metrics to Track**:
```python
class LPIPSTrainingMonitor:
    def __init__(self, log_dir='./logs'):
        self.writer = SummaryWriter(log_dir)
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_correlation': [],
            'learning_rates': [],
            'gradient_norms': []
        }
    
    def log_training_step(self, epoch, batch_idx, metrics, model):
        """Log detailed training metrics"""
        global_step = epoch * 1000 + batch_idx  # Approximate
        
        # Basic metrics
        self.writer.add_scalar('train/loss', metrics['loss'], global_step)
        self.writer.add_scalar('train/accuracy', metrics['accuracy'], global_step)
        
        # Gradient monitoring
        grad_norm = self._compute_gradient_norm(model)
        self.writer.add_scalar('train/gradient_norm', grad_norm, global_step)
        
        # Learning rate
        lr = model.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', lr, global_step)
        
        # Distance statistics
        if 'mean_dist1' in metrics:
            self.writer.add_scalar('train/mean_dist1', metrics['mean_dist1'], global_step)
            self.writer.add_scalar('train/mean_dist2', metrics['mean_dist2'], global_step)
    
    def _compute_gradient_norm(self, model):
        """Compute gradient norm for monitoring"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def log_validation_step(self, epoch, val_metrics):
        """Log validation metrics"""
        self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        self.writer.add_scalar('val/correlation', val_metrics['correlation'], epoch)
        
        # Update history
        self.metrics_history['val_loss'].append(val_metrics['loss'])
        self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])
        self.metrics_history['val_correlation'].append(val_metrics['correlation'])
```

**Training Diagnostics**:
```python
class LPIPSTrainingDiagnostics:
    def __init__(self):
        self.diagnosis_results = {}
    
    def diagnose_training_issues(self, trainer):
        """Comprehensive training diagnosis"""
        issues = []
        
        # Check learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        if current_lr > 1e-3:
            issues.append("Learning rate too high - may cause instability")
        elif current_lr < 1e-7:
            issues.append("Learning rate too low - training may stagnate")
        
        # Check gradient flow
        grad_norm = self._check_gradient_flow(trainer.model)
        if grad_norm < 1e-6:
            issues.append("Vanishing gradients detected")
        elif grad_norm > 10:
            issues.append("Exploding gradients detected - consider gradient clipping")
        
        # Check loss progression
        if len(trainer.train_history['val_loss']) > 10:
            recent_loss = trainer.train_history['val_loss'][-5:]
            if all(recent_loss[i] >= recent_loss[i-1] for i in range(1, len(recent_loss))):
                issues.append("Validation loss increasing - possible overfitting")
        
        # Check accuracy progression
        if len(trainer.train_history['val_accuracy']) > 5:
            recent_acc = trainer.train_history['val_accuracy'][-5:]
            if max(recent_acc) - min(recent_acc) < 0.01:
                issues.append("Accuracy plateaued - consider learning rate adjustment")
        
        return issues
    
    def _check_gradient_flow(self, model):
        """Check for gradient flow issues"""
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None and 'linear_layers' in name:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def suggest_hyperparameter_adjustments(self, current_metrics):
        """Suggest hyperparameter adjustments based on performance"""
        suggestions = []
        
        if current_metrics['val_accuracy'] < 0.6:
            suggestions.append("Low accuracy: Try reducing learning rate or increasing model complexity")
        
        if current_metrics['val_correlation'] < 0.5:
            suggestions.append("Low correlation: Check dataset quality or try different backbone")
        
        if current_metrics['val_loss'] > current_metrics['train_loss'] + 0.1:
            suggestions.append("Overfitting detected: Add regularization or reduce model complexity")
        
        return suggestions
```

**Debugging Common Issues**:
```python
def debug_training_convergence(trainer, debug_batch_size=8):
    """Debug training convergence issues"""
    
    # Test with single batch
    trainer.model.train()
    
    # Get single batch
    dataloader = trainer.data_module.train_dataloader()
    batch = next(iter(dataloader))
    ref_imgs, img1s, img2s, judgments = batch
    
    # Reduce to smaller batch for debugging
    ref_imgs = ref_imgs[:debug_batch_size]
    img1s = img1s[:debug_batch_size]
    img2s = img2s[:debug_batch_size]
    judgments = judgments[:debug_batch_size]
    
    # Test forward pass
    try:
        loss, metrics = trainer.loss_fn(trainer.model, ref_imgs, img1s, img2s, judgments)
        print(f"Forward pass successful: Loss={loss.item():.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return
    
    # Test backward pass
    try:
        loss.backward()
        print("Backward pass successful")
        
        # Check gradients
        grad_info = {}
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_info[name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                }
        
        print("Gradient statistics:")
        for name, stats in grad_info.items():
            print(f"  {name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            
    except Exception as e:
        print(f"Backward pass failed: {e}")
    
    # Test optimizer step
    try:
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        print("Optimizer step successful")
    except Exception as e:
        print(f"Optimizer step failed: {e}")
```

============================================================

Q6: How do you handle overfitting in LPIPS training?

A6: Overfitting is a common issue in LPIPS training due to limited data:

**Detection Strategies**:
```python
class OverfittingDetector:
    def __init__(self, patience=10, threshold=0.02):
        self.patience = patience
        self.threshold = threshold
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def check_overfitting(self, train_loss, val_loss, val_accuracy_history):
        """Detect overfitting patterns"""
        signs_of_overfitting = []
        
        # Large gap between train and validation loss
        if val_loss - train_loss > self.threshold:
            signs_of_overfitting.append("Large train-val loss gap")
        
        # Validation loss increasing while train loss decreasing
        if len(val_accuracy_history) > 5:
            recent_trend = np.polyfit(range(5), val_accuracy_history[-5:], 1)[0]
            if recent_trend < -0.001:  # Decreasing trend
                signs_of_overfitting.append("Validation accuracy declining")
        
        # Early stopping check
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            
        if self.epochs_without_improvement >= self.patience:
            signs_of_overfitting.append("Training stagnated - early stopping triggered")
        
        return signs_of_overfitting
```

**Prevention Techniques**:

1. **Regularization**:
```python
class RegularizedLPIPS(nn.Module):
    def __init__(self, backbone='vgg', dropout_rate=0.5, weight_decay=1e-4):
        super().__init__()
        # ... initialize backbone and features ...
        
        # Add dropout to linear layers
        self.linear_layers = nn.ModuleList()
        for channels in self.feature_channels:
            layers = [
                nn.Dropout(dropout_rate),
                nn.Conv2d(channels, 1, kernel_size=1, bias=False)
            ]
            self.linear_layers.append(nn.Sequential(*layers))
    
    def apply_weight_decay(self, optimizer):
        """Apply weight decay only to linear layers"""
        for name, param in self.named_parameters():
            if 'linear_layers' in name:
                param.data -= optimizer.param_groups[0]['weight_decay'] * param.data
```

2. **Data Augmentation**:
```python
class AntiOverfittingAugmentation:
    def __init__(self, augmentation_strength=0.5):
        self.strength = augmentation_strength
        
    def create_strong_augmentation(self):
        """Strong augmentation to prevent overfitting"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10 * self.strength),
            transforms.ColorJitter(
                brightness=0.2 * self.strength,
                contrast=0.2 * self.strength,
                saturation=0.2 * self.strength,
                hue=0.1 * self.strength
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=50.0, sigma=5.0)
            ], p=0.2)
        ])
```

3. **Cross-Validation**:
```python
class LPIPSCrossValidator:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        
    def cross_validate(self, dataset, model_factory, train_func):
        """Perform k-fold cross-validation"""
        fold_results = []
        
        # Create folds
        folds = self._create_stratified_folds(dataset)
        
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            print(f"Training fold {fold_idx + 1}/{self.n_folds}")
            
            # Create fold datasets
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            # Create fresh model
            model = model_factory()
            
            # Train on fold
            fold_result = train_func(model, train_subset, val_subset)
            fold_results.append(fold_result)
        
        return self._aggregate_results(fold_results)
    
    def _create_stratified_folds(self, dataset):
        """Create stratified folds maintaining class balance"""
        # Implementation of stratified k-fold splitting
        # ... (detailed implementation)
```

============================================================

Q7: What are effective strategies for handling noisy labels in JND datasets?

A7: Human annotation noise is inevitable in perceptual datasets:

**Noise Detection**:
```python
class JNDNoiseDetector:
    def __init__(self, noise_threshold=0.3):
        self.noise_threshold = noise_threshold
        
    def detect_noisy_samples(self, dataset, pretrained_lpips_model):
        """Detect potentially mislabeled samples"""
        noisy_samples = []
        
        for idx, (ref_img, img1, img2, human_judgment) in enumerate(dataset):
            # Get LPIPS predictions
            with torch.no_grad():
                dist1 = pretrained_lpips_model(ref_img.unsqueeze(0), img1.unsqueeze(0))
                dist2 = pretrained_lpips_model(ref_img.unsqueeze(0), img2.unsqueeze(0))
                
                lpips_prediction = 0 if dist1 < dist2 else 1
                
                # Check for disagreement
                if lpips_prediction != human_judgment:
                    confidence = abs(dist1 - dist2).item()
                    
                    # High confidence disagreement suggests noise
                    if confidence > self.noise_threshold:
                        noisy_samples.append({
                            'index': idx,
                            'human_judgment': human_judgment,
                            'lpips_prediction': lpips_prediction,
                            'confidence': confidence,
                            'dist1': dist1.item(),
                            'dist2': dist2.item()
                        })
        
        return sorted(noisy_samples, key=lambda x: x['confidence'], reverse=True)
```

**Noise Handling Strategies**:

1. **Label Smoothing**:
```python
class LabelSmoothingLPIPSLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, lpips_model, ref_img, img1, img2, human_judgment):
        dist1 = lpips_model(ref_img, img1)
        dist2 = lpips_model(ref_img, img2)
        
        logits = dist1 - dist2
        
        # Apply label smoothing
        targets = human_judgment.float()
        smooth_targets = targets * self.confidence + (1 - targets) * (1 - self.confidence)
        
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), smooth_targets)
        
        # Regular accuracy computation
        predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
        accuracy = (predictions == targets).float().mean()
        
        return loss, {'loss': loss.item(), 'accuracy': accuracy.item()}
```

2. **Sample Reweighting**:
```python
class NoiseRobustTrainer:
    def __init__(self, model, confidence_threshold=0.8):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.sample_weights = None
        
    def compute_sample_weights(self, dataset):
        """Compute weights based on sample reliability"""
        weights = []
        
        with torch.no_grad():
            for ref_img, img1, img2, human_judgment in dataset:
                dist1 = self.model(ref_img.unsqueeze(0), img1.unsqueeze(0))
                dist2 = self.model(ref_img.unsqueeze(0), img2.unsqueeze(0))
                
                # Compute confidence based on distance difference
                confidence = torch.sigmoid(torch.abs(dist1 - dist2) * 5)  # Scale factor
                
                # Reduce weight for low-confidence samples
                if confidence < self.confidence_threshold:
                    weight = confidence.item()
                else:
                    weight = 1.0
                
                weights.append(weight)
        
        return torch.tensor(weights)
    
    def weighted_loss(self, loss, batch_indices):
        """Apply sample-specific weights to loss"""
        if self.sample_weights is not None:
            batch_weights = self.sample_weights[batch_indices]
            weighted_loss = loss * batch_weights
            return weighted_loss.mean()
        return loss
```

3. **Co-training with Multiple Models**:
```python
class CoTrainingLPIPS:
    def __init__(self, model1, model2, disagreement_threshold=0.5):
        self.model1 = model1
        self.model2 = model2
        self.disagreement_threshold = disagreement_threshold
        
    def co_train_step(self, ref_imgs, img1s, img2s, judgments):
        """Co-training step with disagreement filtering"""
        
        # Get predictions from both models
        with torch.no_grad():
            dist1_m1 = self.model1(ref_imgs, img1s)
            dist2_m1 = self.model1(ref_imgs, img2s)
            pred1 = (dist1_m1 < dist2_m1).float()
            
            dist1_m2 = self.model2(ref_imgs, img1s)
            dist2_m2 = self.model2(ref_imgs, img2s)
            pred2 = (dist1_m2 < dist2_m2).float()
        
        # Filter samples where models agree
        agreement_mask = (pred1 == pred2).squeeze()
        
        if agreement_mask.sum() > 0:
            # Train on agreed samples
            filtered_refs = ref_imgs[agreement_mask]
            filtered_img1s = img1s[agreement_mask]
            filtered_img2s = img2s[agreement_mask]
            filtered_judgments = judgments[agreement_mask]
            
            # Train both models on filtered data
            loss1, _ = self.train_single_model(self.model1, filtered_refs, 
                                             filtered_img1s, filtered_img2s, 
                                             filtered_judgments)
            loss2, _ = self.train_single_model(self.model2, filtered_refs, 
                                             filtered_img1s, filtered_img2s, 
                                             filtered_judgments)
            
            return (loss1 + loss2) / 2
        else:
            # No agreement, skip this batch
            return torch.tensor(0.0, requires_grad=True)
```

============================================================

Q8: How do you implement curriculum learning for LPIPS training?

A8: Curriculum learning can significantly improve LPIPS training efficiency:

**Difficulty Assessment**:
```python
class LPIPSDifficultyEstimator:
    def __init__(self):
        self.difficulty_metrics = {
            'distance_difference': self._compute_distance_difference,
            'content_complexity': self._compute_content_complexity,
            'annotation_confidence': self._compute_annotation_confidence
        }
    
    def _compute_distance_difference(self, triplet):
        """Smaller distance differences are harder to judge"""
        ref_img, img1, img2, _ = triplet
        
        # Use simple L2 distance as proxy
        dist1 = F.mse_loss(ref_img, img1, reduction='mean')
        dist2 = F.mse_loss(ref_img, img2, reduction='mean')
        
        # Smaller difference = higher difficulty
        diff = abs(dist1 - dist2)
        difficulty = 1.0 / (1.0 + diff * 10)  # Scale factor
        
        return difficulty
    
    def _compute_content_complexity(self, triplet):
        """More complex content is harder to judge"""
        ref_img = triplet[0]
        
        # Use gradient magnitude as complexity measure
        gray_img = rgb_to_grayscale(ref_img)
        grad_x = torch.diff(gray_img, dim=-1)
        grad_y = torch.diff(gray_img, dim=-2)
        
        complexity = torch.mean(torch.sqrt(grad_x**2 + grad_y**2))
        
        # Normalize to [0, 1]
        return torch.clamp(complexity / 0.5, 0, 1)
    
    def _compute_annotation_confidence(self, triplet, annotator_info=None):
        """Lower annotator confidence indicates higher difficulty"""
        if annotator_info is None:
            return 0.5  # Default medium difficulty
        
        # If multiple annotators, use agreement as confidence
        agreements = annotator_info.get('agreements', [1.0])
        confidence = np.mean(agreements)
        
        return 1.0 - confidence  # Higher disagreement = higher difficulty
    
    def estimate_difficulty(self, triplet, annotator_info=None):
        """Combine multiple difficulty metrics"""
        difficulties = []
        
        for metric_name, metric_func in self.difficulty_metrics.items():
            if metric_name == 'annotation_confidence':
                diff = metric_func(triplet, annotator_info)
            else:
                diff = metric_func(triplet)
            difficulties.append(diff)
        
        # Weighted combination
        weights = [0.4, 0.3, 0.3]  # Adjust based on importance
        combined_difficulty = sum(w * d for w, d in zip(weights, difficulties))
        
        return combined_difficulty
```

**Curriculum Scheduling**:
```python
class CurriculumScheduler:
    def __init__(self, dataset, difficulty_estimator, curriculum_type='linear'):
        self.dataset = dataset
        self.difficulty_estimator = difficulty_estimator
        self.curriculum_type = curriculum_type
        
        # Compute difficulties for all samples
        self.difficulties = self._compute_all_difficulties()
        self.sorted_indices = self._sort_by_difficulty()
        
    def _compute_all_difficulties(self):
        """Precompute difficulties for entire dataset"""
        difficulties = []
        
        for idx in range(len(self.dataset)):
            triplet = self.dataset[idx]
            difficulty = self.difficulty_estimator.estimate_difficulty(triplet)
            difficulties.append(difficulty)
        
        return np.array(difficulties)
    
    def _sort_by_difficulty(self):
        """Sort dataset indices by difficulty"""
        return np.argsort(self.difficulties)
    
    def get_epoch_subset(self, epoch, total_epochs):
        """Get subset of data for specific epoch"""
        if self.curriculum_type == 'linear':
            # Gradually include more difficult samples
            progress = epoch / total_epochs
            subset_size = int(len(self.dataset) * (0.3 + 0.7 * progress))
            
        elif self.curriculum_type == 'exponential':
            # Exponential growth in dataset size
            progress = epoch / total_epochs
            subset_size = int(len(self.dataset) * (1 - np.exp(-3 * progress)))
            
        elif self.curriculum_type == 'step':
            # Step-wise curriculum
            if epoch < total_epochs // 3:
                subset_size = len(self.dataset) // 3
            elif epoch < 2 * total_epochs // 3:
                subset_size = 2 * len(self.dataset) // 3
            else:
                subset_size = len(self.dataset)
        
        # Return easiest samples up to subset_size
        selected_indices = self.sorted_indices[:subset_size]
        
        return selected_indices

class CurriculumDataLoader:
    def __init__(self, dataset, curriculum_scheduler, batch_size=32):
        self.dataset = dataset
        self.curriculum_scheduler = curriculum_scheduler
        self.batch_size = batch_size
        
    def get_epoch_dataloader(self, epoch, total_epochs):
        """Get data loader for specific epoch based on curriculum"""
        epoch_indices = self.curriculum_scheduler.get_epoch_subset(epoch, total_epochs)
        
        # Create subset dataset
        epoch_dataset = torch.utils.data.Subset(self.dataset, epoch_indices)
        
        # Create data loader
        return DataLoader(
            epoch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
```

**Curriculum Training Integration**:
```python
class CurriculumLPIPSTrainer(LPIPSTrainer):
    def __init__(self, model, dataset, curriculum_type='linear'):
        super().__init__(model, dataset)
        
        # Initialize curriculum components
        self.difficulty_estimator = LPIPSDifficultyEstimator()
        self.curriculum_scheduler = CurriculumScheduler(
            dataset, self.difficulty_estimator, curriculum_type
        )
        self.curriculum_dataloader = CurriculumDataLoader(
            dataset, self.curriculum_scheduler
        )
        
    def train_with_curriculum(self, num_epochs=100):
        """Train with curriculum learning"""
        
        for epoch in range(num_epochs):
            # Get curriculum-based data loader for this epoch
            epoch_dataloader = self.curriculum_dataloader.get_epoch_dataloader(
                epoch, num_epochs
            )
            
            print(f"Epoch {epoch}: Training on {len(epoch_dataloader.dataset)} samples")
            
            # Regular training step
            epoch_metrics = self.train_epoch_with_dataloader(epoch_dataloader)
            
            # Validation (on full validation set)
            val_metrics = self.validate_epoch()
            
            # Log curriculum progress
            subset_size = len(epoch_dataloader.dataset)
            total_size = len(self.dataset)
            curriculum_progress = subset_size / total_size
            
            self.writer.add_scalar('curriculum/progress', curriculum_progress, epoch)
            self.writer.add_scalar('curriculum/subset_size', subset_size, epoch)
            
            print(f"Curriculum progress: {curriculum_progress:.2%} "
                  f"({subset_size}/{total_size} samples)")
    
    def train_epoch_with_dataloader(self, dataloader):
        """Train single epoch with given dataloader"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for ref_imgs, img1s, img2s, judgments in dataloader:
            # Move to device
            ref_imgs = ref_imgs.to(self.device)
            img1s = img1s.to(self.device)
            img2s = img2s.to(self.device)
            judgments = judgments.to(self.device)
            
            # Training step
            self.optimizer.zero_grad()
            loss, metrics = self.loss_fn(self.model, ref_imgs, img1s, img2s, judgments)
            loss.backward()
            self.optimizer.step()
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
```

This comprehensive training methodology provides robust strategies for effective LPIPS training.