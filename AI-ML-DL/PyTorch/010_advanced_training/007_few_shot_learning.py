import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import math

# Prototypical Networks
class PrototypicalNetwork(nn.Module):
    """Prototypical Networks for Few-Shot Learning"""
    
    def __init__(self, encoder, distance_metric='euclidean'):
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes from support set"""
        prototypes = torch.zeros(n_way, support_embeddings.size(1)).to(support_embeddings.device)
        
        for class_idx in range(n_way):
            class_mask = (support_labels == class_idx)
            if class_mask.sum() > 0:
                prototypes[class_idx] = support_embeddings[class_mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        """Compute distances between query embeddings and prototypes"""
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity (convert to distance)
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        elif self.distance_metric == 'squared_euclidean':
            # Squared Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2) ** 2
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, support_data, support_labels, query_data, n_way, k_shot):
        """Forward pass for prototypical networks"""
        # Encode support and query sets
        support_embeddings = self.encoder(support_data)
        query_embeddings = self.encoder(query_data)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)
        
        # Compute distances and convert to logits
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances  # Negative distance as logits
        
        return logits

# Relation Networks
class RelationModule(nn.Module):
    """Relation module for Relation Networks"""
    
    def __init__(self, input_size, hidden_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """Relation Networks for Few-Shot Learning"""
    
    def __init__(self, encoder, relation_dim=64):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)  # Assuming image input
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.size(1)
        
        self.relation_module = RelationModule(encoder_dim * 2, relation_dim)
    
    def forward(self, support_data, support_labels, query_data, n_way, k_shot):
        """Forward pass for relation networks"""
        # Encode support and query sets
        support_embeddings = self.encoder(support_data)  # [n_way*k_shot, embed_dim]
        query_embeddings = self.encoder(query_data)      # [n_query, embed_dim]
        
        # Reshape support embeddings by class
        support_embeddings = support_embeddings.view(n_way, k_shot, -1)
        
        # Compute class representations (mean of support examples)
        class_representations = support_embeddings.mean(dim=1)  # [n_way, embed_dim]
        
        # Compute relations
        n_query = query_embeddings.size(0)
        embed_dim = query_embeddings.size(1)
        
        # Expand dimensions for pairwise comparison
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_way, -1)  # [n_query, n_way, embed_dim]
        class_expanded = class_representations.unsqueeze(0).expand(n_query, -1, -1)  # [n_query, n_way, embed_dim]
        
        # Concatenate query and class representations
        relations_input = torch.cat([query_expanded, class_expanded], dim=2)  # [n_query, n_way, embed_dim*2]
        
        # Flatten for relation module
        relations_input = relations_input.view(-1, embed_dim * 2)
        
        # Compute relation scores
        relation_scores = self.relation_module(relations_input)  # [n_query*n_way, 1]
        relation_scores = relation_scores.view(n_query, n_way)   # [n_query, n_way]
        
        return relation_scores

# Matching Networks
class AttentionLSTM(nn.Module):
    """Attention LSTM for Matching Networks"""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x, context):
        """Forward pass with context attention"""
        # x: [batch_size, seq_len, input_size]
        # context: [batch_size, context_size]
        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state with context
        h_0 = context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        
        # Compute attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted average
        attended_output = (lstm_out * attention_weights).sum(dim=1)
        
        return attended_output

class MatchingNetwork(nn.Module):
    """Matching Networks for Few-Shot Learning"""
    
    def __init__(self, encoder, use_full_context_embeddings=True):
        super().__init__()
        self.encoder = encoder
        self.use_full_context_embeddings = use_full_context_embeddings
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            encoder_output = self.encoder(dummy_input)
            self.embed_dim = encoder_output.size(1)
        
        if use_full_context_embeddings:
            self.g_encoder = AttentionLSTM(self.embed_dim, self.embed_dim)
            self.f_encoder = AttentionLSTM(self.embed_dim, self.embed_dim)
    
    def cosine_similarity(self, x, y):
        """Compute cosine similarity between vectors"""
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return torch.mm(x_norm, y_norm.t())
    
    def forward(self, support_data, support_labels, query_data, n_way, k_shot):
        """Forward pass for matching networks"""
        # Encode support and query sets
        support_embeddings = self.encoder(support_data)
        query_embeddings = self.encoder(query_data)
        
        if self.use_full_context_embeddings:
            # Full Context Embeddings (FCE)
            # Enhance support embeddings with context
            support_context = support_embeddings.mean(dim=0, keepdim=True)
            support_embeddings_enhanced = self.g_encoder(
                support_embeddings.unsqueeze(0), 
                support_context
            ).squeeze(0)
            
            # Enhance query embeddings with support context
            query_embeddings_enhanced = self.f_encoder(
                query_embeddings.unsqueeze(0),
                support_context.expand(query_embeddings.size(0), -1)
            ).squeeze(0)
        else:
            support_embeddings_enhanced = support_embeddings
            query_embeddings_enhanced = query_embeddings
        
        # Compute attention weights (cosine similarities)
        similarities = self.cosine_similarity(query_embeddings_enhanced, support_embeddings_enhanced)
        
        # Convert similarities to attention weights
        attention_weights = F.softmax(similarities, dim=1)
        
        # Weighted sum over support labels (one-hot encoded)
        support_labels_one_hot = F.one_hot(support_labels, num_classes=n_way).float()
        predictions = torch.mm(attention_weights, support_labels_one_hot)
        
        return predictions

# Metric Learning Approaches
class TripletNetwork(nn.Module):
    """Triplet Network for Metric Learning"""
    
    def __init__(self, encoder, margin=1.0):
        super().__init__()
        self.encoder = encoder
        self.margin = margin
    
    def triplet_loss(self, anchor, positive, negative):
        """Compute triplet loss"""
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()
    
    def forward(self, anchor_data, positive_data, negative_data):
        """Forward pass for triplet network"""
        anchor_embeddings = self.encoder(anchor_data)
        positive_embeddings = self.encoder(positive_data)
        negative_embeddings = self.encoder(negative_data)
        
        loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        return loss, anchor_embeddings, positive_embeddings, negative_embeddings

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for Siamese Networks"""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, labels):
        """Compute contrastive loss"""
        # Euclidean distance
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        loss_positive = labels * distances.pow(2)
        loss_negative = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = 0.5 * (loss_positive + loss_negative)
        return loss.mean()

class SiameseNetwork(nn.Module):
    """Siamese Network for Few-Shot Learning"""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
    
    def forward(self, x1, x2, labels=None):
        """Forward pass for siamese network"""
        embedding1 = self.encoder(x1)
        embedding2 = self.encoder(x2)
        
        if labels is not None:
            loss = self.contrastive_loss(embedding1, embedding2, labels)
            return loss, embedding1, embedding2
        else:
            return embedding1, embedding2

# Encoders for Few-Shot Learning
class ConvEncoder(nn.Module):
    """Convolutional encoder for few-shot learning"""
    
    def __init__(self, input_channels=3, hidden_dim=64, output_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        self.bn4 = nn.BatchNorm2d(output_dim)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for ResNet encoder"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    """ResNet-based encoder for few-shot learning"""
    
    def __init__(self, input_channels=3, output_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, output_dim, 2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

# Few-Shot Learning Trainer
class FewShotTrainer:
    """Trainer for few-shot learning models"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_episode(self, support_data, support_labels, query_data, query_labels, n_way, k_shot):
        """Train on a single episode"""
        self.model.train()
        
        # Move data to device
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        query_data = query_data.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if isinstance(self.model, (PrototypicalNetwork, RelationNetwork, MatchingNetwork)):
            logits = self.model(support_data, support_labels, query_data, n_way, k_shot)
            loss = self.criterion(logits, query_labels)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == query_labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def evaluate_episode(self, support_data, support_labels, query_data, query_labels, n_way, k_shot):
        """Evaluate on a single episode"""
        self.model.eval()
        
        with torch.no_grad():
            # Move data to device
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Forward pass
            if isinstance(self.model, (PrototypicalNetwork, RelationNetwork, MatchingNetwork)):
                logits = self.model(support_data, support_labels, query_data, n_way, k_shot)
                loss = self.criterion(logits, query_labels)
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == query_labels).float().mean().item()
        
        return loss.item(), accuracy

# Sample Dataset
class SyntheticFewShotDataset(Dataset):
    """Synthetic dataset for few-shot learning"""
    
    def __init__(self, num_classes=20, samples_per_class=100, input_shape=(3, 32, 32)):
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_shape = input_shape
        
        # Generate class-specific patterns
        self.class_patterns = []
        for _ in range(num_classes):
            pattern = torch.randn(input_shape) * 0.3
            self.class_patterns.append(pattern)
        
        # Generate data
        self.data = []
        self.labels = []
        
        for class_id in range(num_classes):
            base_pattern = self.class_patterns[class_id]
            
            for _ in range(samples_per_class):
                # Add noise to base pattern
                sample = base_pattern + torch.randn(input_shape) * 0.2
                sample = torch.clamp(sample, 0, 1)
                
                self.data.append(sample)
                self.labels.append(class_id)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Task Generator
class FewShotTaskGenerator:
    """Generate few-shot learning tasks"""
    
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        # Organize by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.classes = list(self.class_to_indices.keys())
    
    def sample_task(self):
        """Sample a few-shot learning task"""
        # Sample classes
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for new_class_id, original_class_id in enumerate(selected_classes):
            class_indices = self.class_to_indices[original_class_id]
            
            # Sample support and query examples
            total_needed = self.k_shot + self.q_query
            if len(class_indices) >= total_needed:
                sampled_indices = random.sample(class_indices, total_needed)
            else:
                sampled_indices = random.choices(class_indices, k=total_needed)
            
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:self.k_shot + self.q_query]
            
            # Collect support examples
            for idx in support_indices:
                data, _ = self.dataset[idx]
                support_data.append(data)
                support_labels.append(new_class_id)
            
            # Collect query examples
            for idx in query_indices:
                data, _ = self.dataset[idx]
                query_data.append(data)
                query_labels.append(new_class_id)
        
        return (torch.stack(support_data), torch.tensor(support_labels),
                torch.stack(query_data), torch.tensor(query_labels))

if __name__ == "__main__":
    print("Few-Shot Learning Techniques")
    print("=" * 35)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    dataset = SyntheticFewShotDataset(num_classes=10, samples_per_class=50)
    task_generator = FewShotTaskGenerator(dataset, n_way=3, k_shot=2, q_query=5)
    
    print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")
    
    # Test task generation
    print("\n1. Testing Task Generation")
    print("-" * 30)
    
    support_data, support_labels, query_data, query_labels = task_generator.sample_task()
    print(f"Support: {support_data.shape}, Query: {query_data.shape}")
    print(f"Support labels: {support_labels}")
    print(f"Query labels: {query_labels}")
    
    # Test Prototypical Networks
    print("\n2. Testing Prototypical Networks")
    print("-" * 30)
    
    encoder = ConvEncoder(input_channels=3, output_dim=64)
    proto_net = PrototypicalNetwork(encoder, distance_metric='euclidean')
    trainer = FewShotTrainer(proto_net, device=device)
    
    # Train on a few episodes
    for episode in range(3):
        support_data, support_labels, query_data, query_labels = task_generator.sample_task()
        loss, accuracy = trainer.train_episode(support_data, support_labels, query_data, query_labels, 3, 2)
        print(f"Episode {episode + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Test Relation Networks
    print("\n3. Testing Relation Networks")
    print("-" * 30)
    
    encoder = ConvEncoder(input_channels=3, output_dim=64)
    relation_net = RelationNetwork(encoder, relation_dim=8)
    trainer = FewShotTrainer(relation_net, device=device)
    
    # Train on a few episodes
    for episode in range(3):
        support_data, support_labels, query_data, query_labels = task_generator.sample_task()
        loss, accuracy = trainer.train_episode(support_data, support_labels, query_data, query_labels, 3, 2)
        print(f"Episode {episode + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Test Matching Networks
    print("\n4. Testing Matching Networks")
    print("-" * 30)
    
    encoder = ConvEncoder(input_channels=3, output_dim=64)
    matching_net = MatchingNetwork(encoder, use_full_context_embeddings=False)
    trainer = FewShotTrainer(matching_net, device=device)
    
    # Train on a few episodes
    for episode in range(3):
        support_data, support_labels, query_data, query_labels = task_generator.sample_task()
        loss, accuracy = trainer.train_episode(support_data, support_labels, query_data, query_labels, 3, 2)
        print(f"Episode {episode + 1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    # Test Siamese Network
    print("\n5. Testing Siamese Network")
    print("-" * 30)
    
    encoder = ConvEncoder(input_channels=3, output_dim=64)
    siamese_net = SiameseNetwork(encoder)
    
    # Create pairs for siamese training
    data1, labels1 = [], []
    data2, labels2 = [], []
    pair_labels = []
    
    for _ in range(10):
        # Sample positive pair
        class_id = random.choice(list(task_generator.class_to_indices.keys()))
        class_indices = task_generator.class_to_indices[class_id]
        
        if len(class_indices) >= 2:
            idx1, idx2 = random.sample(class_indices, 2)
            data1.append(dataset[idx1][0])
            data2.append(dataset[idx2][0])
            pair_labels.append(1)  # Positive pair
        
        # Sample negative pair
        class1, class2 = random.sample(list(task_generator.class_to_indices.keys()), 2)
        idx1 = random.choice(task_generator.class_to_indices[class1])
        idx2 = random.choice(task_generator.class_to_indices[class2])
        
        data1.append(dataset[idx1][0])
        data2.append(dataset[idx2][0])
        pair_labels.append(0)  # Negative pair
    
    data1 = torch.stack(data1).to(device)
    data2 = torch.stack(data2).to(device)
    pair_labels = torch.tensor(pair_labels, dtype=torch.float).to(device)
    
    siamese_net = siamese_net.to(device)
    loss, emb1, emb2 = siamese_net(data1, data2, pair_labels)
    print(f"Siamese Network - Loss: {loss.item():.4f}")
    print(f"Embedding shapes: {emb1.shape}, {emb2.shape}")
    
    print("\nFew-shot learning demonstrations completed!") 