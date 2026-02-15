import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# InfoNCE Loss
class InfoNCE(nn.Module):
    """InfoNCE (Information Noise Contrastive Estimation) Loss"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        """
        anchor: [batch_size, dim]
        positive: [batch_size, dim]  
        negatives: [batch_size, num_negatives, dim]
        """
        batch_size, dim = anchor.shape
        num_negatives = negatives.size(1)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        return F.cross_entropy(logits, labels)

# Contrastive Learning with Momentum
class MoCo(nn.Module):
    """Momentum Contrast (MoCo) implementation"""
    
    def __init__(self, encoder, dim=128, K=4096, m=0.999, T=0.07):
        super().__init__()
        self.K = K  # Queue size
        self.m = m  # Momentum coefficient
        self.T = T  # Temperature
        
        # Create query and key encoders
        self.encoder_q = encoder
        self.encoder_k = self._build_key_encoder(encoder)
        
        # Create projection heads
        self.head_q = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, dim)
        )
        self.head_k = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.ReLU(),
            nn.Linear(encoder.output_dim, dim)
        )
        
        # Copy query parameters to key
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _build_key_encoder(self, encoder):
        """Build key encoder (copy of query encoder)"""
        import copy
        return copy.deepcopy(encoder)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # Move pointer
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """Forward pass"""
        # Compute query features
        q = self.encoder_q(im_q)
        q = F.normalize(self.head_q(q), dim=1)
        
        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.encoder_k(im_k)
            k = F.normalize(self.head_k(k), dim=1)
        
        # Compute logits
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # Labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels

# Supervised Contrastive Learning
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss"""
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(self, features, labels=None, mask=None):
        """
        features: [batch_size, n_views, feature_dim] or [batch_size, feature_dim]
        labels: [batch_size] (for supervised contrastive learning)
        mask: [batch_size, batch_size] (for self-supervised contrastive learning)
        """
        device = features.device
        
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

# Simple Encoder
class SimpleEncoder(nn.Module):
    """Simple encoder for contrastive learning"""
    
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=64):
        super().__init__()
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.encoder(x)

# Contrastive Learning Trainer
class ContrastiveTrainer:
    """Trainer for contrastive learning methods"""
    
    def __init__(self, model, device='cuda', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_epoch_infonce(self, dataloader, num_negatives=16):
        """Train with InfoNCE loss"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, data in enumerate(dataloader):
            if isinstance(data, tuple):
                data = data[0]  # Take first element if tuple
            
            data = data.to(self.device)
            batch_size = data.size(0)
            
            # Create augmented versions (simplified)
            anchor = data + 0.1 * torch.randn_like(data)
            positive = data + 0.1 * torch.randn_like(data)
            
            # Sample negatives from the batch
            if batch_size > num_negatives:
                neg_indices = torch.randperm(batch_size)[:num_negatives]
                negatives = data[neg_indices].unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            else:
                negatives = data.unsqueeze(1).repeat(1, num_negatives, 1, 1, 1)
            
            self.optimizer.zero_grad()
            
            # Encode
            anchor_emb = self.model(anchor)
            positive_emb = self.model(positive)
            neg_emb = self.model(negatives.view(-1, *negatives.shape[2:]))
            neg_emb = neg_emb.view(batch_size, num_negatives, -1)
            
            # Compute InfoNCE loss
            infonce = InfoNCE(temperature=0.1)
            loss = infonce(anchor_emb, positive_emb, neg_emb)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}: InfoNCE Loss = {loss.item():.4f}')
        
        return total_loss / len(dataloader)
    
    def train_epoch_supcon(self, dataloader):
        """Train with Supervised Contrastive loss"""
        self.model.train()
        total_loss = 0
        
        supcon_loss = SupConLoss(temperature=0.1)
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Create two augmented views
            data1 = data + 0.1 * torch.randn_like(data)
            data2 = data + 0.1 * torch.randn_like(data)
            
            self.optimizer.zero_grad()
            
            # Encode both views
            features1 = F.normalize(self.model(data1), dim=1)
            features2 = F.normalize(self.model(data2), dim=1)
            
            # Stack features
            features = torch.stack([features1, features2], dim=1)
            
            # Compute supervised contrastive loss
            loss = supcon_loss(features, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}: SupCon Loss = {loss.item():.4f}')
        
        return total_loss / len(dataloader)

# Sample Datasets
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    """Simple dataset for contrastive learning"""
    
    def __init__(self, size=500, input_shape=(3, 32, 32), num_classes=5):
        self.data = torch.randn(size, *input_shape)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class UnsupervisedDataset(Dataset):
    """Unsupervised dataset (no labels)"""
    
    def __init__(self, size=500, input_shape=(3, 32, 32)):
        self.data = torch.randn(size, *input_shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Evaluation Function
def evaluate_representations(encoder, train_dataset, test_dataset, device='cuda'):
    """Evaluate learned representations with linear classification"""
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Extract features
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_features, train_labels = [], []
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            features = encoder(data)
            train_features.append(features.cpu())
            train_labels.append(labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    
    test_features, test_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            features = encoder(data)
            test_features.append(features.cpu())
            test_labels.append(labels)
    
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Train linear classifier
    num_classes = len(torch.unique(train_labels))
    classifier = nn.Linear(train_features.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    classifier.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = classifier(train_features.to(device))
        loss = criterion(logits, train_labels.to(device))
        loss.backward()
        optimizer.step()
    
    # Evaluation
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_features.to(device))
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == test_labels.to(device)).float().mean().item()
    
    return accuracy * 100

if __name__ == "__main__":
    print("Contrastive Learning Methods")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    supervised_dataset = SimpleDataset(size=300, num_classes=5)
    unsupervised_dataset = UnsupervisedDataset(size=300)
    test_dataset = SimpleDataset(size=100, num_classes=5)
    
    supervised_loader = DataLoader(supervised_dataset, batch_size=16, shuffle=True)
    unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=16, shuffle=True)
    
    # Test InfoNCE
    print("\n1. Testing InfoNCE Loss")
    print("-" * 25)
    
    encoder1 = SimpleEncoder(input_dim=3*32*32, output_dim=64)
    trainer1 = ContrastiveTrainer(encoder1, device)
    
    # Train with InfoNCE
    for epoch in range(3):
        avg_loss = trainer1.train_epoch_infonce(unsupervised_loader, num_negatives=8)
        print(f'Epoch {epoch + 1}: InfoNCE Loss = {avg_loss:.4f}')
    
    # Evaluate
    infonce_accuracy = evaluate_representations(encoder1, supervised_dataset, test_dataset, device)
    print(f"InfoNCE representation accuracy: {infonce_accuracy:.2f}%")
    
    # Test Supervised Contrastive Learning
    print("\n2. Testing Supervised Contrastive Learning")
    print("-" * 40)
    
    encoder2 = SimpleEncoder(input_dim=3*32*32, output_dim=64)
    trainer2 = ContrastiveTrainer(encoder2, device)
    
    # Train with SupCon
    for epoch in range(3):
        avg_loss = trainer2.train_epoch_supcon(supervised_loader)
        print(f'Epoch {epoch + 1}: SupCon Loss = {avg_loss:.4f}')
    
    # Evaluate
    supcon_accuracy = evaluate_representations(encoder2, supervised_dataset, test_dataset, device)
    print(f"SupCon representation accuracy: {supcon_accuracy:.2f}%")
    
    # Test MoCo (simplified)
    print("\n3. Testing MoCo (Momentum Contrast)")
    print("-" * 35)
    
    base_encoder = SimpleEncoder(input_dim=3*32*32, output_dim=128)
    moco = MoCo(base_encoder, dim=64, K=128, m=0.999, T=0.07)
    moco_trainer = ContrastiveTrainer(moco, device)
    
    # Simplified MoCo training
    moco.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, data in enumerate(unsupervised_loader):
            data = data.to(device)
            
            # Create two augmented views
            im_q = data + 0.1 * torch.randn_like(data)
            im_k = data + 0.1 * torch.randn_like(data)
            
            moco_trainer.optimizer.zero_grad()
            
            logits, labels = moco(im_q, im_k)
            loss = criterion(logits, labels)
            
            loss.backward()
            moco_trainer.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}: MoCo Loss = {loss.item():.4f}')
        
        avg_loss = total_loss / len(unsupervised_loader)
        print(f'Epoch {epoch + 1}: MoCo Average Loss = {avg_loss:.4f}')
    
    # Evaluate MoCo encoder
    moco_accuracy = evaluate_representations(moco.encoder_q, supervised_dataset, test_dataset, device)
    print(f"MoCo representation accuracy: {moco_accuracy:.2f}%")
    
    # Random baseline
    print("\n4. Random Baseline")
    print("-" * 20)
    
    random_encoder = SimpleEncoder(input_dim=3*32*32, output_dim=64)
    random_accuracy = evaluate_representations(random_encoder, supervised_dataset, test_dataset, device)
    print(f"Random baseline accuracy: {random_accuracy:.2f}%")
    
    # Summary
    print("\n5. Results Summary")
    print("-" * 20)
    print(f"Random baseline: {random_accuracy:.2f}%")
    print(f"InfoNCE: {infonce_accuracy:.2f}%")
    print(f"Supervised Contrastive: {supcon_accuracy:.2f}%")
    print(f"MoCo: {moco_accuracy:.2f}%")
    
    print("\nContrastive learning demonstrations completed!") 