# Validation, Error Analysis, and Inference

## Table of Contents

- [TensorBoard Integration](#tensorboard-integration)
- [Inference Optimization](#inference-optimization)
- [Validation Techniques](#validation-techniques)
- [Error Analysis](#error-analysis)

---

## TensorBoard Integration

**TensorBoard** provides visualization of training metrics, model graphs, histograms, and images. Use **SummaryWriter** for **scalar**, **histogram**, **image**, and **graph logging**, plus **hyperparameter tracking**.

### SummaryWriter Basics

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment')
```

### Scalar Logging

```python
writer.add_scalar('train/loss', loss.item(), step)
writer.add_scalar('train/accuracy', accuracy, step)
writer.add_scalar('val/loss', val_loss, step)
writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)
```

### Histogram Logging

```python
for name, param in model.named_parameters():
    writer.add_histogram(f'weights/{name}', param, step)
    if param.grad is not None:
        writer.add_histogram(f'gradients/{name}', param.grad, step)
```

### Image Logging

```python
import torchvision

grid = torchvision.utils.make_grid(images[:8], normalize=True)
writer.add_image('train/samples', grid, step)
```

### Graph Logging

```python
dummy_input = torch.randn(1, 3, 32, 32)
writer.add_graph(model, dummy_input)
```

### Hyperparameter Tracking

```python
writer.add_hparams(
    {'lr': 0.001, 'batch_size': 32, 'dropout': 0.3},
    {'accuracy': 0.85, 'loss': 0.45},
    run_name='run_1'
)
```

### PR Curves and Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image

predictions = model(data).argmax(dim=1).cpu().numpy()
labels = targets.cpu().numpy()
cm = confusion_matrix(labels, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
writer.add_image('confusion_matrix', img_tensor, step)
plt.close()

for i in range(num_classes):
    class_preds = probs[:, i]
    class_labels = (labels == i).float()
    writer.add_pr_curve(f'pr_curve/class_{i}', class_labels, class_preds, step)
```

---

## Inference Optimization

**Inference optimization** reduces latency and memory. Key techniques: **torch.no_grad**, **torch.inference_mode**, **batched inference**, **model.eval**, **TorchScript**, and **operator fusion**.

### torch.no_grad and torch.inference_mode

**torch.no_grad** disables gradient computation. **torch.inference_mode** is stricter and can enable additional optimizations:

```python
model.eval()

with torch.no_grad():
    outputs = model(input_tensor)

with torch.inference_mode():
    outputs = model(input_tensor)
```

### Batched Inference

Process multiple samples at once for better GPU utilization:

```python
def batched_inference(model, data_loader, batch_size=32):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds)
```

### model.eval

Switch BatchNorm and Dropout to evaluation behavior:

```python
model.eval()
```

### TorchScript for Inference

Convert model to TorchScript for deployment without Python:

```python
model.eval()
example_input = torch.randn(1, 3, 224, 224)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

loaded = torch.jit.load("model_traced.pt")
output = loaded(example_input)
```

### Operator Fusion

Fuse Conv-BatchNorm for fewer kernel launches. Module names depend on model structure:

```python
model.eval()
fused_model = torch.quantization.fuse_modules(
    model,
    [['features.0', 'features.1'], ['features.3', 'features.4']]
)
```

### Additional Optimizations

```python
model = model.to(memory_format=torch.channels_last)
```

Use `torch.backends.cudnn.benchmark = True` for fixed input sizes.

---

## Validation Techniques

**Validation techniques** ensure reliable performance estimates. Use **k-fold cross-validation**, **stratified validation**, **metrics computation**, **confusion matrix**, and **ROC/AUC**.

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(data_tensor)):
    train_data = data_tensor[train_idx]
    train_labels = labels_tensor[train_idx]
    val_data = data_tensor[val_idx]
    val_labels = labels_tensor[val_idx]

    model = create_fresh_model()
    train_model(model, train_data, train_labels)
    val_acc = evaluate(model, val_data, val_labels)
    fold_results.append(val_acc)

mean_acc = np.mean(fold_results)
std_acc = np.std(fold_results)
print(f"CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
```

### Stratified Validation

Maintain class distribution in each fold:

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skfold.split(data_tensor, labels_tensor):
    train_data = data_tensor[train_idx]
    val_data = data_tensor[val_idx]
```

### Metrics Computation

```python
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

predictions = model(data).argmax(dim=1).cpu().numpy()
targets = labels.cpu().numpy()

accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, average='weighted')
recall = recall_score(targets, predictions, average='weighted')
f1 = f1_score(targets, predictions, average='weighted')

print(classification_report(targets, predictions, target_names=class_names))
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(targets, predictions)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()
```

### ROC and AUC

```python
from sklearn.metrics import roc_curve, auc

probs = F.softmax(model(data), dim=1).cpu().numpy()

fpr, tpr, _ = roc_curve(targets, probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

---

## Error Analysis

**Error analysis** identifies failure modes. Techniques include **misclassification analysis**, **per-class performance**, **confidence calibration**, and **failure mode identification**.

### Misclassification Analysis

```python
def analyze_misclassifications(model, data_loader):
    model.eval()
    misclassified = []
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            wrong = (preds != targets)
            for i in range(wrong.sum().item()):
                idx = wrong.nonzero()[i].item()
                misclassified.append({
                    'data': data[idx].cpu(),
                    'true': targets[idx].item(),
                    'pred': preds[idx].item(),
                    'confidence': probs[idx, preds[idx]].item()
                })
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return misclassified, np.array(all_preds), np.array(all_targets), np.array(all_probs)
```

### Per-Class Performance

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    targets, predictions, labels=range(num_classes)
)

for i in range(num_classes):
    print(f"Class {i}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, n={support[i]}")
```

### Confidence Calibration

Check if predicted confidence matches actual accuracy (reliability diagram):

```python
import numpy as np
import matplotlib.pyplot as plt

def reliability_diagram(confidences, correct, n_bins=10):
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(confidences[mask].mean())

    plt.plot(bin_confidences, bin_accuracies, 'o-')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
```

### Failure Mode Identification

Group errors by true class, predicted class, or confidence:

```python
def failure_modes(misclassified):
    by_true_class = {}
    by_pred_class = {}
    for m in misclassified:
        t, p = m['true'], m['pred']
        by_true_class.setdefault(t, []).append(m)
        by_pred_class.setdefault(p, []).append(m)

    print("Most confused pairs (true -> pred):")
    from collections import Counter
    pairs = Counter((m['true'], m['pred']) for m in misclassified)
    for (t, p), count in pairs.most_common(10):
        print(f"  {t} -> {p}: {count}")
```
