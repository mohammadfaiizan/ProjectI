import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss Function"""
    
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature  # Temperature for softmax
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, student_outputs, teacher_outputs, targets):
        """Compute distillation loss"""
        # Standard cross-entropy loss
        hard_loss = self.criterion(student_outputs, targets)
        
        # Soft targets from teacher
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss, soft_loss

# Attention Transfer
class AttentionTransfer(nn.Module):
    """Attention Transfer Loss"""
    
    def __init__(self, beta=1000):
        super().__init__()
        self.beta = beta
    
    def attention_map(self, feature_map):
        """Compute attention map from feature map"""
        # Sum over channel dimension and normalize
        attention = torch.sum(feature_map ** 2, dim=1, keepdim=True)
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        return attention.view(attention.size(0), 1, feature_map.size(2), feature_map.size(3))
    
    def forward(self, student_features, teacher_features):
        """Compute attention transfer loss"""
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Compute attention maps
            s_attention = self.attention_map(s_feat)
            t_attention = self.attention_map(t_feat)
            
            # L2 loss between attention maps
            loss += F.mse_loss(s_attention, t_attention)
        
        return self.beta * loss

# Teacher Model (Large)
class TeacherModel(nn.Module):
    """Large teacher model"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, return_features=False):
        features = []
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        features.append(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        features.append(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        features.append(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        features.append(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        if return_features:
            return x, features
        return x

# Student Model (Small)
class StudentModel(nn.Module):
    """Small student model"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, return_features=False):
        features = []
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        features.append(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        features.append(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        features.append(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if return_features:
            return x, features
        return x

# Knowledge Distillation Trainer
class KnowledgeDistillationTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(self, teacher_model, student_model, device='cuda'):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        # Loss functions
        self.distillation_loss = DistillationLoss(alpha=0.7, temperature=4.0)
        self.attention_transfer = AttentionTransfer(beta=1000)
        
        # Optimizer for student
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=0.001)
        
    def train_epoch(self, dataloader, use_attention_transfer=False):
        """Train student for one epoch"""
        self.student.train()
        
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        total_at_loss = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Get teacher predictions
            with torch.no_grad():
                if use_attention_transfer:
                    teacher_outputs, teacher_features = self.teacher(data, return_features=True)
                else:
                    teacher_outputs = self.teacher(data)
            
            # Get student predictions
            if use_attention_transfer:
                student_outputs, student_features = self.student(data, return_features=True)
            else:
                student_outputs = self.student(data)
            
            # Compute distillation loss
            dist_loss, hard_loss, soft_loss = self.distillation_loss(
                student_outputs, teacher_outputs, targets
            )
            
            total_loss_batch = dist_loss
            
            # Add attention transfer loss
            if use_attention_transfer:
                at_loss = self.attention_transfer(student_features, teacher_features)
                total_loss_batch += at_loss
                total_at_loss += at_loss.item()
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Total Loss = {total_loss_batch.item():.4f}')
        
        num_batches = len(dataloader)
        return (total_loss / num_batches, 
                total_hard_loss / num_batches,
                total_soft_loss / num_batches,
                total_at_loss / num_batches if use_attention_transfer else 0)
    
    def evaluate(self, dataloader):
        """Evaluate both teacher and student"""
        self.teacher.eval()
        self.student.eval()
        
        teacher_correct = 0
        student_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Teacher predictions
                teacher_outputs = self.teacher(data)
                _, teacher_pred = torch.max(teacher_outputs, 1)
                
                # Student predictions
                student_outputs = self.student(data)
                _, student_pred = torch.max(student_outputs, 1)
                
                total += targets.size(0)
                teacher_correct += (teacher_pred == targets).sum().item()
                student_correct += (student_pred == targets).sum().item()
        
        teacher_acc = 100. * teacher_correct / total
        student_acc = 100. * student_correct / total
        
        return teacher_acc, student_acc

# Sample Dataset
class SampleDataset(Dataset):
    """Sample dataset for distillation"""
    
    def __init__(self, size=1000, input_shape=(3, 32, 32), num_classes=10):
        self.data = torch.randn(size, *input_shape)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Comparison Functions
def compare_model_sizes(teacher, student):
    """Compare model sizes"""
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")

def train_student_baseline(student_model, dataloader, device, epochs=5):
    """Train student without distillation (baseline)"""
    model = student_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100. * correct / total

if __name__ == "__main__":
    print("Knowledge Distillation")
    print("=" * 25)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SampleDataset(size=800, num_classes=10)
    test_dataset = SampleDataset(size=200, num_classes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create models
    teacher = TeacherModel(num_classes=10)
    student = StudentModel(num_classes=10)
    
    print("\n1. Model Comparison")
    print("-" * 20)
    compare_model_sizes(teacher, student)
    
    # Pre-train teacher (simplified)
    print("\n2. Pre-training Teacher")
    print("-" * 25)
    
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
    teacher_criterion = nn.CrossEntropyLoss()
    
    teacher = teacher.to(device)
    teacher.train()
    
    for epoch in range(3):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            teacher_optimizer.zero_grad()
            outputs = teacher(data)
            loss = teacher_criterion(outputs, targets)
            loss.backward()
            teacher_optimizer.step()
    
    print("Teacher pre-training completed")
    
    # Train student with distillation
    print("\n3. Knowledge Distillation Training")
    print("-" * 35)
    
    # Create fresh student for distillation
    student_kd = StudentModel(num_classes=10)
    kd_trainer = KnowledgeDistillationTrainer(teacher, student_kd, device)
    
    for epoch in range(3):
        total_loss, hard_loss, soft_loss, at_loss = kd_trainer.train_epoch(
            train_loader, use_attention_transfer=False
        )
        
        print(f'Epoch {epoch + 1}:')
        print(f'  Total Loss: {total_loss:.4f}')
        print(f'  Hard Loss: {hard_loss:.4f}')
        print(f'  Soft Loss: {soft_loss:.4f}')
    
    # Evaluate models
    print("\n4. Model Evaluation")
    print("-" * 20)
    
    teacher_acc, student_kd_acc = kd_trainer.evaluate(test_loader)
    
    print(f"Teacher accuracy: {teacher_acc:.2f}%")
    print(f"Student (with KD) accuracy: {student_kd_acc:.2f}%")
    
    # Train student baseline
    print("\n5. Baseline Student Training")
    print("-" * 30)
    
    student_baseline = StudentModel(num_classes=10)
    baseline_acc = train_student_baseline(student_baseline, train_loader, device, epochs=3)
    
    print(f"Student (baseline) accuracy: {baseline_acc:.2f}%")
    
    # Test with attention transfer
    print("\n6. Knowledge Distillation + Attention Transfer")
    print("-" * 45)
    
    student_at = StudentModel(num_classes=10)
    at_trainer = KnowledgeDistillationTrainer(teacher, student_at, device)
    
    for epoch in range(2):
        total_loss, hard_loss, soft_loss, at_loss = at_trainer.train_epoch(
            train_loader, use_attention_transfer=True
        )
        
        print(f'Epoch {epoch + 1}:')
        print(f'  Total Loss: {total_loss:.4f}')
        print(f'  Hard Loss: {hard_loss:.4f}')
        print(f'  Soft Loss: {soft_loss:.4f}')
        print(f'  AT Loss: {at_loss:.4f}')
    
    _, student_at_acc = at_trainer.evaluate(test_loader)
    print(f"Student (KD + AT) accuracy: {student_at_acc:.2f}%")
    
    # Summary
    print("\n7. Summary")
    print("-" * 15)
    print(f"Teacher: {teacher_acc:.2f}%")
    print(f"Student (baseline): {baseline_acc:.2f}%")
    print(f"Student (KD): {student_kd_acc:.2f}%")
    print(f"Student (KD + AT): {student_at_acc:.2f}%")
    
    improvement_kd = student_kd_acc - baseline_acc
    improvement_at = student_at_acc - baseline_acc
    
    print(f"\nImprovement with KD: {improvement_kd:.2f}%")
    print(f"Improvement with KD + AT: {improvement_at:.2f}%")
    
    print("\nKnowledge distillation demonstrations completed!") 