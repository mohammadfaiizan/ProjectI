#!/usr/bin/env python3
"""PyTorch Ranking Loss Functions - Margin, triplet, ranking losses"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

print("=== Ranking Loss Functions Overview ===")

print("Common ranking losses:")
print("1. Margin Ranking Loss")
print("2. Triplet Loss and variants")
print("3. Pairwise Ranking Loss")
print("4. ListNet and ListMLE")
print("5. RankNet Loss")
print("6. LambdaRank Loss")
print("7. Contrastive Loss")
print("8. Angular losses")

print("\n=== Margin Ranking Loss ===")

# Margin Ranking Loss - basic ranking loss
margin_loss = nn.MarginRankingLoss(margin=1.0)
margin_loss_no_reduction = nn.MarginRankingLoss(margin=1.0, reduction='none')

# Sample data for ranking
batch_size = 10
x1 = torch.randn(batch_size)  # Scores for first set
x2 = torch.randn(batch_size)  # Scores for second set
y = torch.randint(-1, 2, (batch_size,), dtype=torch.float)  # -1 or 1

print(f"x1 (first scores): {x1[:5]}")
print(f"x2 (second scores): {x2[:5]}")
print(f"y (rankings): {y[:5]} (-1: x2 > x1, 1: x1 > x2)")

# Margin ranking loss computation
margin_result = margin_loss(x1, x2, y)
margin_none = margin_loss_no_reduction(x1, x2, y)

print(f"Margin Ranking Loss: {margin_result.item():.6f}")
print(f"Per-sample losses: {margin_none[:5]}")

# Manual margin ranking computation
def margin_ranking_manual(x1, x2, y, margin=1.0):
    return torch.clamp(margin - y * (x1 - x2), min=0).mean()

margin_manual = margin_ranking_manual(x1, x2, y, margin=1.0)
print(f"Manual Margin Ranking: {margin_manual.item():.6f}")

# Test with different margins
for margin in [0.5, 1.0, 2.0]:
    loss_fn = nn.MarginRankingLoss(margin=margin)
    result = loss_fn(x1, x2, y)
    print(f"Margin {margin}: {result.item():.6f}")

print("\n=== Triplet Loss ===")

class TripletLoss(nn.Module):
    """Triplet Loss for metric learning"""
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, anchor, positive, negative):
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TripletLossWithHardMining(nn.Module):
    """Triplet Loss with hard negative mining"""
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings, labels):
        # embeddings: [batch_size, embedding_dim]
        # labels: [batch_size]
        
        distances = torch.cdist(embeddings, embeddings, p=2)
        batch_size = embeddings.size(0)
        
        losses = []
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Find positive samples (same label)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            if not positive_mask.any():
                continue
            
            # Find negative samples (different label)
            negative_mask = labels != anchor_label
            if not negative_mask.any():
                continue
            
            # Hard positive (farthest positive)
            pos_distances = distances[i][positive_mask]
            hard_pos_dist = pos_distances.max()
            
            # Hard negative (closest negative)
            neg_distances = distances[i][negative_mask]
            hard_neg_dist = neg_distances.min()
            
            # Compute triplet loss
            loss = F.relu(hard_pos_dist - hard_neg_dist + self.margin)
            losses.append(loss)
        
        if losses:
            total_loss = torch.stack(losses)
            if self.reduction == 'mean':
                return total_loss.mean()
            elif self.reduction == 'sum':
                return total_loss.sum()
            else:
                return total_loss
        else:
            return torch.tensor(0.0, requires_grad=True)

# Sample embedding data
embedding_dim = 64
batch_size = 20
embeddings = torch.randn(batch_size, embedding_dim)
embedding_labels = torch.randint(0, 5, (batch_size,))

# Create triplets manually
anchor = embeddings[:10]
positive = embeddings[10:20]  # Different samples but we'll treat as positive
negative = torch.randn(10, embedding_dim)

# Test triplet losses
triplet_loss = TripletLoss(margin=1.0)
triplet_hard_mining = TripletLossWithHardMining(margin=1.0)

triplet_result = triplet_loss(anchor, positive, negative)
hard_mining_result = triplet_hard_mining(embeddings, embedding_labels)

print(f"Basic Triplet Loss: {triplet_result.item():.6f}")
print(f"Triplet with Hard Mining: {hard_mining_result.item():.6f}")

# Built-in PyTorch triplet loss
builtin_triplet = nn.TripletMarginLoss(margin=1.0)
builtin_result = builtin_triplet(anchor, positive, negative)
print(f"Built-in Triplet Loss: {builtin_result.item():.6f}")

print("\n=== Contrastive Loss ===")

class ContrastiveLoss(nn.Module):
    """Contrastive Loss for siamese networks"""
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embedding1, embedding2, labels):
        # labels: 1 for similar pairs, 0 for dissimilar pairs
        distances = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Loss for similar pairs: distance^2
        pos_loss = labels * distances.pow(2)
        
        # Loss for dissimilar pairs: max(0, margin - distance)^2
        neg_loss = (1 - labels) * F.relu(self.margin - distances).pow(2)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Test contrastive loss
embedding1 = torch.randn(10, embedding_dim)
embedding2 = torch.randn(10, embedding_dim)
similarity_labels = torch.randint(0, 2, (10,)).float()

contrastive_loss = ContrastiveLoss(margin=2.0)
contrastive_result = contrastive_loss(embedding1, embedding2, similarity_labels)

print(f"Contrastive Loss: {contrastive_result.item():.6f}")
print(f"Similarity labels: {similarity_labels}")

print("\n=== Angular Losses ===")

class AngularLoss(nn.Module):
    """Angular Loss for metric learning"""
    def __init__(self, alpha=45, in_degree=True):
        super().__init__()
        if in_degree:
            self.alpha = torch.tensor(alpha * torch.pi / 180)  # Convert to radians
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, anchor, positive, negative):
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute angles
        ap_dot = (anchor * positive).sum(dim=1)
        an_dot = (anchor * negative).sum(dim=1)
        pn_dot = (positive * negative).sum(dim=1)
        
        # Angular constraint
        ap_angle = torch.acos(torch.clamp(ap_dot, -1 + 1e-7, 1 - 1e-7))
        
        # Target angle
        target_angle = ap_angle + self.alpha
        target_cosine = torch.cos(target_angle)
        
        # Loss: encourage an_dot < target_cosine
        loss = F.relu(an_dot - target_cosine)
        
        return loss.mean()

class ArcFaceLoss(nn.Module):
    """ArcFace Loss - additive angular margin"""
    def __init__(self, feature_dim, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(features, weight)
        
        # Get target cosine
        target_cosine = cosine[torch.arange(len(labels)), labels]
        
        # Add angular margin
        target_angle = torch.acos(torch.clamp(target_cosine, -1 + 1e-7, 1 - 1e-7))
        target_angle_with_margin = target_angle + self.margin
        target_cosine_with_margin = torch.cos(target_angle_with_margin)
        
        # Replace target class cosine with margin-adjusted cosine
        cosine_with_margin = cosine.clone()
        cosine_with_margin[torch.arange(len(labels)), labels] = target_cosine_with_margin
        
        # Scale and compute cross-entropy
        logits = cosine_with_margin * self.scale
        
        return F.cross_entropy(logits, labels)

# Test angular losses
num_classes = 5
angular_loss = AngularLoss(alpha=30)  # 30 degrees
arcface_loss = ArcFaceLoss(embedding_dim, num_classes, margin=0.3)

angular_result = angular_loss(anchor, positive, negative)
arcface_result = arcface_loss(embeddings[:15], embedding_labels[:15])

print(f"Angular Loss (30°): {angular_result.item():.6f}")
print(f"ArcFace Loss: {arcface_result.item():.6f}")

print("\n=== Learning-to-Rank Losses ===")

class RankNetLoss(nn.Module):
    """RankNet Loss for learning to rank"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, scores1, scores2, labels):
        # labels: 1 if scores1 > scores2, -1 if scores1 < scores2, 0 if equal
        prob = torch.sigmoid(scores1 - scores2)
        
        # Convert labels to probabilities
        target_prob = (labels + 1) / 2  # Convert {-1, 0, 1} to {0, 0.5, 1}
        
        # Cross-entropy loss
        loss = -(target_prob * torch.log(prob + 1e-8) + 
                (1 - target_prob) * torch.log(1 - prob + 1e-8))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ListNetLoss(nn.Module):
    """ListNet Loss for permutation-based ranking"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predicted_scores, target_scores):
        # Convert scores to probabilities using softmax
        pred_probs = F.softmax(predicted_scores, dim=-1)
        target_probs = F.softmax(target_scores, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(torch.log(pred_probs + 1e-8), target_probs, reduction='none')
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ListMLELoss(nn.Module):
    """ListMLE Loss for maximum likelihood ranking"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predicted_scores, target_ranking):
        # target_ranking: permutation indices (0 is best, 1 is second best, etc.)
        batch_size, list_size = predicted_scores.shape
        
        losses = []
        for b in range(batch_size):
            scores = predicted_scores[b]
            ranking = target_ranking[b]
            
            loss = 0
            for i in range(list_size):
                # Get items not yet selected
                remaining_items = ranking[i:]
                remaining_scores = scores[remaining_items]
                
                # Probability of selecting the correct next item
                if len(remaining_scores) > 1:
                    target_score = remaining_scores[0]
                    log_prob = target_score - torch.logsumexp(remaining_scores, dim=0)
                    loss -= log_prob
            
            losses.append(loss)
        
        total_loss = torch.stack(losses)
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

# Test learning-to-rank losses
list_size = 5
batch_size = 8

predicted_scores = torch.randn(batch_size, list_size)
target_scores = torch.randn(batch_size, list_size)

# Create rankings (indices sorted by score)
target_rankings = torch.argsort(target_scores, dim=1, descending=True)

# Create pairwise data for RankNet
scores1 = torch.randn(20)
scores2 = torch.randn(20)
pairwise_labels = torch.randint(-1, 2, (20,)).float()

ranknet_loss = RankNetLoss()
listnet_loss = ListNetLoss()
listmle_loss = ListMLELoss()

ranknet_result = ranknet_loss(scores1, scores2, pairwise_labels)
listnet_result = listnet_loss(predicted_scores, target_scores)
listmle_result = listmle_loss(predicted_scores, target_rankings)

print(f"RankNet Loss: {ranknet_result.item():.6f}")
print(f"ListNet Loss: {listnet_result.item():.6f}")
print(f"ListMLE Loss: {listmle_result.item():.6f}")

print("\n=== Advanced Ranking Losses ===")

class ApproxNDCGLoss(nn.Module):
    """Approximate NDCG Loss"""
    def __init__(self, temperature=1.0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, predicted_scores, relevance_scores):
        # predicted_scores: [batch_size, list_size]
        # relevance_scores: [batch_size, list_size] (ground truth relevance)
        
        batch_size, list_size = predicted_scores.shape
        
        # Compute ideal DCG
        ideal_sorted_relevance = torch.sort(relevance_scores, dim=1, descending=True)[0]
        ideal_dcg = self._compute_dcg(ideal_sorted_relevance)
        
        # Compute predicted DCG using differentiable ranking
        soft_ranking = F.softmax(predicted_scores / self.temperature, dim=1)
        
        # Weighted relevance by soft ranking positions
        dcg_values = []
        for i in range(list_size):
            position_weight = 1.0 / torch.log2(torch.tensor(i + 2.0))
            dcg_values.append(position_weight)
        dcg_weights = torch.tensor(dcg_values)
        
        # Approximate DCG
        predicted_dcg = torch.sum(soft_ranking * relevance_scores.unsqueeze(-1) * dcg_weights, dim=(1, 2))
        
        # NDCG
        ndcg = predicted_dcg / (ideal_dcg + 1e-8)
        
        # Loss is negative NDCG (we want to maximize NDCG)
        loss = -ndcg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _compute_dcg(self, relevance_scores):
        """Compute DCG for sorted relevance scores"""
        batch_size, list_size = relevance_scores.shape
        dcg = torch.zeros(batch_size)
        
        for i in range(list_size):
            position_weight = 1.0 / torch.log2(torch.tensor(i + 2.0))
            dcg += relevance_scores[:, i] * position_weight
        
        return dcg

class LambdaLoss(nn.Module):
    """Simplified Lambda Loss"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predicted_scores, relevance_scores):
        # Compute all pairwise differences
        batch_size, list_size = predicted_scores.shape
        
        losses = []
        for b in range(batch_size):
            pred = predicted_scores[b]
            rel = relevance_scores[b]
            
            total_loss = 0
            count = 0
            
            for i in range(list_size):
                for j in range(list_size):
                    if rel[i] > rel[j]:  # i should be ranked higher than j
                        # Cross-entropy style loss
                        diff = pred[i] - pred[j]
                        loss = torch.log(1 + torch.exp(-diff))
                        
                        # Weight by relevance difference (simplified lambda weight)
                        weight = abs(rel[i] - rel[j])
                        total_loss += weight * loss
                        count += 1
            
            if count > 0:
                losses.append(total_loss / count)
        
        if losses:
            total_loss = torch.stack(losses)
            if self.reduction == 'mean':
                return total_loss.mean()
            elif self.reduction == 'sum':
                return total_loss.sum()
            else:
                return total_loss
        else:
            return torch.tensor(0.0, requires_grad=True)

# Test advanced ranking losses
relevance_scores = torch.randint(0, 4, (batch_size, list_size)).float()  # 0-3 relevance

approx_ndcg_loss = ApproxNDCGLoss(temperature=1.0)
lambda_loss = LambdaLoss()

ndcg_result = approx_ndcg_loss(predicted_scores, relevance_scores)
lambda_result = lambda_loss(predicted_scores, relevance_scores)

print(f"Approximate NDCG Loss: {ndcg_result.item():.6f}")
print(f"Lambda Loss: {lambda_result.item():.6f}")

print("\n=== Multi-Task Ranking Loss ===")

class MultiTaskRankingLoss(nn.Module):
    """Multi-task ranking with shared embeddings"""
    def __init__(self, num_tasks, task_weights=None, margin=1.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.margin = margin
        
        if task_weights is None:
            self.task_weights = nn.Parameter(torch.ones(num_tasks))
        else:
            self.task_weights = nn.Parameter(torch.tensor(task_weights))
    
    def forward(self, embeddings, task_rankings):
        # embeddings: [batch_size, embedding_dim]
        # task_rankings: [batch_size, num_tasks] (rankings for each task)
        
        total_loss = 0
        
        for task_id in range(self.num_tasks):
            task_ranking = task_rankings[:, task_id]
            
            # Compute triplet loss for this task
            task_loss = 0
            count = 0
            
            batch_size = embeddings.size(0)
            for i in range(batch_size):
                for j in range(batch_size):
                    for k in range(batch_size):
                        if task_ranking[i] > task_ranking[j] and task_ranking[j] > task_ranking[k]:
                            # i > j > k in ranking for this task
                            anchor = embeddings[i]
                            positive = embeddings[j]  
                            negative = embeddings[k]
                            
                            pos_dist = F.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))
                            neg_dist = F.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))
                            
                            loss = F.relu(pos_dist - neg_dist + self.margin)
                            task_loss += loss
                            count += 1
            
            if count > 0:
                task_loss = task_loss / count
                total_loss += self.task_weights[task_id] * task_loss
        
        return total_loss

# Test multi-task ranking
num_tasks = 3
task_rankings = torch.randint(0, 5, (batch_size, num_tasks)).float()

multi_task_ranking = MultiTaskRankingLoss(num_tasks)
multi_task_result = multi_task_ranking(embeddings, task_rankings)

print(f"Multi-task Ranking Loss: {multi_task_result.item():.6f}")
print(f"Task weights: {multi_task_ranking.task_weights.data}")

print("\n=== Ranking Loss Best Practices ===")

print("Loss Selection Guidelines:")
print("1. Margin Ranking: Simple pairwise comparisons")
print("2. Triplet Loss: Metric learning, similarity search")
print("3. Contrastive Loss: Siamese networks, verification tasks")
print("4. RankNet: Pairwise learning-to-rank")
print("5. ListNet/ListMLE: Listwise learning-to-rank")
print("6. NDCG Loss: Information retrieval, recommendation")
print("7. Angular losses: When angular relationships matter")

print("\nImplementation Tips:")
print("1. Use hard mining for better triplet selection")
print("2. Balance positive and negative pairs")
print("3. Consider margin values based on embedding scale")
print("4. Normalize embeddings for angular losses")
print("5. Use appropriate sampling strategies")
print("6. Monitor distance distributions during training")

print("\nCommon Issues:")
print("1. Collapse of embeddings to a single point")
print("2. Imbalanced positive/negative pairs")
print("3. Poor triplet selection (too easy/too hard)")
print("4. Scale mismatch between different loss components")
print("5. Numerical instability with extreme distances")

print("\nDebugging Ranking Losses:")
print("1. Visualize embedding distributions")
print("2. Check distance histograms")
print("3. Monitor hard/semi-hard triplet ratios")
print("4. Validate ranking quality metrics")
print("5. Test with known good/bad pairs")

print("\n=== Ranking Losses Complete ===")

# Memory cleanup
del embeddings, embedding_labels, anchor, positive, negative
del embedding1, embedding2, similarity_labels
del predicted_scores, target_scores, target_rankings, relevance_scores