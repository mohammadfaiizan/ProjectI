import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Dict

# DARTS (Differentiable Architecture Search) Implementation
class MixedOp(nn.Module):
    """Mixed operation for DARTS"""
    
    def __init__(self, C, stride, ops_list):
        super().__init__()
        self.ops = nn.ModuleList()
        
        for op_name in ops_list:
            op = self._create_operation(op_name, C, stride)
            self.ops.append(op)
    
    def _create_operation(self, op_name, C, stride):
        """Create primitive operation"""
        if op_name == 'none':
            return Zero(stride)
        elif op_name == 'skip_connect':
            return Identity() if stride == 1 else FactorizedReduce(C, C)
        elif op_name == 'sep_conv_3x3':
            return SepConv(C, C, 3, stride, 1)
        elif op_name == 'sep_conv_5x5':
            return SepConv(C, C, 5, stride, 2)
        elif op_name == 'dil_conv_3x3':
            return DilConv(C, C, 3, stride, 2, 2)
        elif op_name == 'dil_conv_5x5':
            return DilConv(C, C, 5, stride, 4, 2)
        elif op_name == 'avg_pool_3x3':
            return nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif op_name == 'max_pool_3x3':
            return nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError(f"Unknown operation: {op_name}")
    
    def forward(self, x, weights):
        """Forward pass with weighted sum of operations"""
        return sum(w * op(x) for w, op in zip(weights, self.ops))

# Primitive Operations
class SepConv(nn.Module):
    """Separable convolution"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated convolution"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    """Identity operation"""
    
    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation"""
    
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
    
    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    """Factorized reduce operation"""
    
    def __init__(self, C_in, C_out):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
    
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

# DARTS Cell
class Cell(nn.Module):
    """DARTS cell with mixed operations"""
    
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier
        
        # Preprocess operations
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        # Define operations
        if reduction:
            ops_list = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
                       'avg_pool_3x3', 'max_pool_3x3', 'skip_connect']
        else:
            ops_list = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
                       'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'none']
        
        # Create mixed operations
        self._ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, ops_list)
                self._ops.append(op)
    
    def forward(self, s0, s1, weights):
        """Forward pass through cell"""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self.multiplier:], dim=1)

class ReLUConvBN(nn.Module):
    """ReLU + Conv + BatchNorm"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )
    
    def forward(self, x):
        return self.op(x)

# DARTS Network
class DARTSNetwork(nn.Module):
    """DARTS searchable network"""
    
    def __init__(self, C=16, num_classes=10, layers=8, steps=4, multiplier=4):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self.steps = steps
        self.multiplier = multiplier
        
        C_curr = multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        
        # Calculate reduction layers
        reduction_prev = False
        self.cells = nn.ModuleList()
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # Initialize architecture parameters
        self._initialize_alphas()
    
    def _initialize_alphas(self):
        """Initialize architecture parameters"""
        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_ops = 8  # Number of operations
        
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    
    def forward(self, input):
        """Forward pass"""
        s0 = s1 = self.stem(input)
        
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            
            s0, s1 = s1, cell(s0, s1, weights)
        
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
    
    def arch_parameters(self):
        """Return architecture parameters"""
        return [self.alphas_normal, self.alphas_reduce]
    
    def genotype(self):
        """Extract genotype from current architecture"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != 7))[:2]  # Exclude 'none'
                
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != 7:  # Exclude 'none'
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((ops_list[k_best], j))
                start = end
                n += 1
            return gene
        
        ops_list = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
                   'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'none']
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        return {'normal': gene_normal, 'reduce': gene_reduce}

# DARTS Trainer
class DARTSTrainer:
    """Trainer for DARTS"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Separate optimizers for model weights and architecture parameters
        self.model_optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.025, 
            momentum=0.9, 
            weight_decay=3e-4
        )
        
        self.arch_optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.model_optimizer, T_max=50, eta_min=0.001
        )
    
    def train_step(self, train_data, train_targets, valid_data, valid_targets):
        """Single training step with bilevel optimization"""
        # Step 1: Update architecture parameters
        self.arch_optimizer.zero_grad()
        
        # Forward pass on validation set
        logits = self.model(valid_data)
        arch_loss = self.criterion(logits, valid_targets)
        
        # Backward pass for architecture
        arch_loss.backward()
        self.arch_optimizer.step()
        
        # Step 2: Update model parameters
        self.model_optimizer.zero_grad()
        
        # Forward pass on training set
        logits = self.model(train_data)
        model_loss = self.criterion(logits, train_targets)
        
        # Backward pass for model
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.model_optimizer.step()
        
        return model_loss.item(), arch_loss.item()
    
    def train_epoch(self, train_loader, valid_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_model_loss = 0
        total_arch_loss = 0
        num_batches = 0
        
        train_iter = iter(train_loader)
        valid_iter = iter(valid_loader)
        
        for batch_idx in range(min(len(train_loader), len(valid_loader))):
            try:
                train_data, train_targets = next(train_iter)
                valid_data, valid_targets = next(valid_iter)
            except StopIteration:
                break
            
            train_data, train_targets = train_data.to(self.device), train_targets.to(self.device)
            valid_data, valid_targets = valid_data.to(self.device), valid_targets.to(self.device)
            
            model_loss, arch_loss = self.train_step(train_data, train_targets, valid_data, valid_targets)
            
            total_model_loss += model_loss
            total_arch_loss += arch_loss
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}: Model Loss = {model_loss:.4f}, Arch Loss = {arch_loss:.4f}')
        
        self.scheduler.step()
        
        return total_model_loss / num_batches, total_arch_loss / num_batches
    
    def evaluate(self, test_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

# Random Search Baseline
class RandomSearchNAS:
    """Random search baseline for NAS"""
    
    def __init__(self, search_space, num_samples=100):
        self.search_space = search_space
        self.num_samples = num_samples
        self.evaluated_archs = []
    
    def sample_architecture(self):
        """Sample random architecture"""
        arch = {}
        for layer_name, ops in self.search_space.items():
            arch[layer_name] = np.random.choice(ops)
        return arch
    
    def evaluate_architecture(self, arch, dataset, epochs=5):
        """Evaluate sampled architecture"""
        # Create model based on architecture
        # This is simplified - in practice would need to build actual model
        model = self._build_model_from_arch(arch)
        
        # Train and evaluate
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Simple training loop
        model.train()
        for epoch in range(epochs):
            for data, targets in dataloader:
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
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        self.evaluated_archs.append((arch, accuracy))
        
        return accuracy
    
    def _build_model_from_arch(self, arch):
        """Build model from architecture description"""
        # Simplified model builder
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
    
    def search(self, dataset):
        """Perform random search"""
        best_arch = None
        best_accuracy = 0
        
        for i in range(self.num_samples):
            arch = self.sample_architecture()
            accuracy = self.evaluate_architecture(arch, dataset)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_arch = arch
            
            print(f'Sample {i+1}/{self.num_samples}: Accuracy = {accuracy:.4f}')
        
        return best_arch, best_accuracy

# Sample Dataset
class SampleCIFARDataset(Dataset):
    """Sample CIFAR-like dataset"""
    
    def __init__(self, size=1000, num_classes=10):
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

if __name__ == "__main__":
    print("Neural Architecture Search (DARTS)")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample datasets
    train_dataset = SampleCIFARDataset(size=800, num_classes=10)
    valid_dataset = SampleCIFARDataset(size=200, num_classes=10)
    test_dataset = SampleCIFARDataset(size=200, num_classes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    # Test DARTS
    print("\n1. Testing DARTS")
    print("-" * 20)
    
    # Create DARTS model
    model = DARTSNetwork(C=16, num_classes=10, layers=4, steps=4)
    trainer = DARTSTrainer(model, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture parameters: {sum(p.numel() for p in model.arch_parameters()):,}")
    
    # Train for a few epochs
    for epoch in range(3):
        model_loss, arch_loss = trainer.train_epoch(train_loader, valid_loader)
        test_accuracy = trainer.evaluate(test_loader)
        
        print(f'Epoch {epoch + 1}:')
        print(f'  Model Loss: {model_loss:.4f}')
        print(f'  Arch Loss: {arch_loss:.4f}')
        print(f'  Test Accuracy: {test_accuracy:.2f}%')
        
        # Print current architecture weights
        print('  Architecture weights (normal):')
        normal_weights = F.softmax(model.alphas_normal, dim=-1)
        print(f'    Mean: {normal_weights.mean(dim=0)}')
        
        print()
    
    # Extract final architecture
    print("2. Final Architecture")
    print("-" * 20)
    
    genotype = model.genotype()
    print("Normal cell:")
    for op, node in genotype['normal']:
        print(f"  {op} -> node {node}")
    
    print("Reduction cell:")
    for op, node in genotype['reduce']:
        print(f"  {op} -> node {node}")
    
    # Test Random Search baseline
    print("\n3. Random Search Baseline")
    print("-" * 25)
    
    search_space = {
        'conv1': ['conv3x3', 'conv5x5', 'sepconv3x3'],
        'conv2': ['conv3x3', 'conv5x5', 'sepconv3x3'],
        'pool': ['maxpool', 'avgpool']
    }
    
    random_nas = RandomSearchNAS(search_space, num_samples=5)  # Small number for demo
    
    print("Starting random search...")
    best_arch, best_acc = random_nas.search(train_dataset)
    
    print(f"\nBest architecture: {best_arch}")
    print(f"Best accuracy: {best_acc:.4f}")
    
    # Architecture analysis
    print("\n4. Architecture Analysis")
    print("-" * 25)
    
    # Analyze operation importance
    ops_list = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
               'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'none']
    
    normal_weights = F.softmax(model.alphas_normal, dim=-1)
    reduce_weights = F.softmax(model.alphas_reduce, dim=-1)
    
    print("Operation importance (normal cell):")
    avg_weights = normal_weights.mean(dim=0)
    for i, op in enumerate(ops_list):
        print(f"  {op}: {avg_weights[i].item():.4f}")
    
    print("\nOperation importance (reduction cell):")
    avg_weights = reduce_weights.mean(dim=0)
    for i, op in enumerate(ops_list):
        print(f"  {op}: {avg_weights[i].item():.4f}")
    
    print("\nNeural Architecture Search demonstrations completed!") 