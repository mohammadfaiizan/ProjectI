import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

# Note: PyTorch Geometric operations require the torch_geometric package
# Install with: pip install torch-geometric

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
    from torch_geometric.nn import MessagePassing, aggr
    from torch_geometric.utils import to_networkx, from_networkx
    from torch_geometric.datasets import TUDataset, Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. Install with: pip install torch-geometric")

# Graph Data Creation and Manipulation
class GraphCreator:
    """Utilities for creating and manipulating graph data"""
    
    @staticmethod
    def create_simple_graph() -> 'Data':
        """Create a simple graph for demonstration"""
        # Node features (4 nodes, 3 features each)
        x = torch.tensor([
            [1.0, 2.0, 3.0],  # Node 0
            [2.0, 3.0, 1.0],  # Node 1
            [3.0, 1.0, 2.0],  # Node 2
            [1.0, 3.0, 2.0]   # Node 3
        ], dtype=torch.float)
        
        # Edge list (source, target)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
            [1, 0, 2, 1, 3, 2, 0, 3]   # Target nodes
        ], dtype=torch.long)
        
        # Edge features (optional)
        edge_attr = torch.tensor([
            [1.0], [1.0], [2.0], [2.0],
            [3.0], [3.0], [1.0], [1.0]
        ], dtype=torch.float)
        
        # Graph label (for graph classification)
        y = torch.tensor([1], dtype=torch.long)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        else:
            return {
                'x': x, 'edge_index': edge_index, 
                'edge_attr': edge_attr, 'y': y
            }
    
    @staticmethod
    def from_networkx_graph(G: nx.Graph, node_features: Optional[Dict] = None) -> 'Data':
        """Convert NetworkX graph to PyTorch Geometric format"""
        
        if TORCH_GEOMETRIC_AVAILABLE:
            # Add node features if provided
            if node_features:
                for node, features in node_features.items():
                    G.nodes[node]['x'] = features
            
            return from_networkx(G)
        else:
            # Fallback implementation
            num_nodes = len(G.nodes())
            edges = list(G.edges())
            edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
            
            # Create default node features if not provided
            if node_features is None:
                x = torch.randn(num_nodes, 3)
            else:
                x = torch.tensor([node_features.get(i, [0.0, 0.0, 0.0]) for i in range(num_nodes)])
            
            return {'x': x, 'edge_index': edge_index}
    
    @staticmethod
    def create_molecular_graph(atom_types: List[str], bonds: List[Tuple[int, int]]) -> 'Data':
        """Create a molecular graph from atoms and bonds"""
        
        # Simple atom type encoding
        atom_encoding = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'H': 4}
        
        # Create node features (one-hot encoding + additional features)
        x = torch.zeros(len(atom_types), 5 + 3)  # 5 atom types + 3 additional features
        
        for i, atom in enumerate(atom_types):
            if atom in atom_encoding:
                x[i, atom_encoding[atom]] = 1.0
            # Add some dummy additional features
            x[i, 5:] = torch.randn(3)
        
        # Create edge index
        if bonds:
            edge_index = torch.tensor([
                [bond[0] for bond in bonds] + [bond[1] for bond in bonds],
                [bond[1] for bond in bonds] + [bond[0] for bond in bonds]
            ], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            return Data(x=x, edge_index=edge_index)
        else:
            return {'x': x, 'edge_index': edge_index}

# Graph Neural Network Layers
class CustomGCNLayer(nn.Module):
    """Custom Graph Convolutional Network layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Simple message passing implementation
        row, col = edge_index
        
        # Linear transformation
        x = self.linear(x)
        
        # Aggregate messages from neighbors
        out = torch.zeros_like(x)
        for i in range(x.size(0)):
            neighbor_mask = (col == i)
            if neighbor_mask.any():
                neighbors = row[neighbor_mask]
                out[i] = torch.mean(x[neighbors], dim=0)
            else:
                out[i] = x[i]  # No neighbors, keep original
        
        return out

class GATLayer(nn.Module):
    """Graph Attention Network layer (simplified)"""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        self.linear = nn.Linear(in_channels, heads * out_channels)
        self.attention = nn.Linear(2 * heads * out_channels, heads)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        # Linear transformation
        x = self.linear(x)  # [N, heads * out_channels]
        
        # Compute attention coefficients
        x_i = x[row]  # Source nodes
        x_j = x[col]  # Target nodes
        
        # Concatenate source and target features
        alpha_input = torch.cat([x_i, x_j], dim=-1)  # [E, 2 * heads * out_channels]
        alpha = self.attention(alpha_input)  # [E, heads]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Apply softmax per node
        alpha = self._softmax(alpha, row, x.size(0))
        
        # Apply attention and aggregate
        out = torch.zeros(x.size(0), self.heads, self.out_channels, device=x.device)
        
        for head in range(self.heads):
            head_start = head * self.out_channels
            head_end = (head + 1) * self.out_channels
            
            x_head = x[:, head_start:head_end]
            alpha_head = alpha[:, head]
            
            # Weighted aggregation
            for i in range(x.size(0)):
                neighbor_mask = (col == i)
                if neighbor_mask.any():
                    neighbors = row[neighbor_mask]
                    weights = alpha_head[neighbor_mask]
                    out[i, head] = torch.sum(weights.unsqueeze(-1) * x_head[neighbors], dim=0)
                else:
                    out[i, head] = x_head[i]
        
        # Concatenate or average heads
        if self.heads > 1:
            out = out.view(x.size(0), -1)  # Concatenate
        else:
            out = out.squeeze(1)
        
        return out
    
    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Apply softmax per node"""
        alpha_max = torch.full((num_nodes, alpha.size(1)), float('-inf'), device=alpha.device)
        alpha_max.index_reduce_(0, index, alpha, 'amax')
        alpha_max = alpha_max[index]
        
        alpha = alpha - alpha_max
        alpha = torch.exp(alpha)
        
        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.index_add_(0, index, alpha)
        alpha_sum = alpha_sum[index]
        
        return alpha / (alpha_sum + 1e-16)

# Graph Neural Network Models
class GCNModel(nn.Module):
    """Graph Convolutional Network for node classification"""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, num_classes)
        else:
            self.conv1 = CustomGCNLayer(num_features, hidden_channels)
            self.conv2 = CustomGCNLayer(hidden_channels, hidden_channels)
            self.conv3 = CustomGCNLayer(hidden_channels, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    """GraphSAGE model for node classification"""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.convs = nn.ModuleList([
                SAGEConv(num_features, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels),
                SAGEConv(hidden_channels, num_classes)
            ])
        else:
            # Fallback to custom GCN layers
            self.convs = nn.ModuleList([
                CustomGCNLayer(num_features, hidden_channels),
                CustomGCNLayer(hidden_channels, hidden_channels),
                CustomGCNLayer(hidden_channels, num_classes)
            ])
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GATModel(nn.Module):
    """Graph Attention Network model"""
    
    def __init__(self, num_features: int, hidden_channels: int, 
                 num_classes: int, heads: int = 8):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)
            self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, dropout=0.6)
        else:
            self.conv1 = GATLayer(num_features, hidden_channels, heads=heads)
            self.conv2 = GATLayer(hidden_channels * heads, num_classes, heads=1)
        
        self.dropout = nn.Dropout(0.6)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphClassificationModel(nn.Module):
    """Model for graph-level classification"""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.convs = nn.ModuleList([
                GCNConv(num_features, hidden_channels),
                GCNConv(hidden_channels, hidden_channels),
                GCNConv(hidden_channels, hidden_channels)
            ])
        else:
            self.convs = nn.ModuleList([
                CustomGCNLayer(num_features, hidden_channels),
                CustomGCNLayer(hidden_channels, hidden_channels),
                CustomGCNLayer(hidden_channels, hidden_channels)
            ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Graph convolutions
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        # Graph-level pooling
        if TORCH_GEOMETRIC_AVAILABLE and batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # Simple global pooling for single graph
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Custom Message Passing Layer
if TORCH_GEOMETRIC_AVAILABLE:
    class CustomMessagePassing(MessagePassing):
        """Custom message passing layer"""
        
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__(aggr='add')  # "Add" aggregation
            self.linear = nn.Linear(in_channels, out_channels)
            self.reset_parameters()
        
        def reset_parameters(self):
            self.linear.reset_parameters()
        
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.propagate(edge_index, x=x)
        
        def message(self, x_j: torch.Tensor) -> torch.Tensor:
            # x_j has shape [E, in_channels]
            return self.linear(x_j)
        
        def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
            # aggr_out has shape [N, out_channels]
            return aggr_out

# Graph Utilities
class GraphUtils:
    """Utility functions for graph processing"""
    
    @staticmethod
    def visualize_graph(data: Union['Data', Dict], title: str = "Graph"):
        """Visualize graph structure"""
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'edge_index'):
            G = to_networkx(data, to_undirected=True)
        else:
            # Fallback for dictionary format
            G = nx.Graph()
            if 'x' in data:
                G.add_nodes_from(range(data['x'].size(0)))
            if 'edge_index' in data:
                edges = data['edge_index'].t().tolist()
                G.add_edges_from(edges)
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=16, font_weight='bold')
        plt.title(title)
        plt.show()
    
    @staticmethod
    def compute_graph_statistics(data: Union['Data', Dict]) -> Dict[str, Any]:
        """Compute basic graph statistics"""
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'edge_index'):
            edge_index = data.edge_index
            x = data.x
        else:
            edge_index = data['edge_index']
            x = data['x']
        
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        num_features = x.size(1)
        
        # Compute degree statistics
        degrees = torch.zeros(num_nodes, dtype=torch.long)
        degrees.index_add_(0, edge_index[1], torch.ones(num_edges, dtype=torch.long))
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'avg_degree': degrees.float().mean().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item(),
            'degree_std': degrees.float().std().item()
        }
        
        return stats
    
    @staticmethod
    def create_random_graph(num_nodes: int, num_edges: int, 
                           num_features: int) -> Union['Data', Dict]:
        """Create a random graph"""
        # Random node features
        x = torch.randn(num_nodes, num_features)
        
        # Random edges (ensuring no self-loops)
        edge_list = []
        while len(edge_list) < num_edges:
            src = torch.randint(0, num_nodes, (1,)).item()
            dst = torch.randint(0, num_nodes, (1,)).item()
            if src != dst and (src, dst) not in edge_list and (dst, src) not in edge_list:
                edge_list.append((src, dst))
        
        edge_index = torch.tensor([[e[0] for e in edge_list], [e[1] for e in edge_list]], 
                                 dtype=torch.long)
        
        if TORCH_GEOMETRIC_AVAILABLE:
            return Data(x=x, edge_index=edge_index)
        else:
            return {'x': x, 'edge_index': edge_index}

# Graph Dataset Creation
class GraphDataset:
    """Custom graph dataset"""
    
    def __init__(self, graphs: List[Union['Data', Dict]], labels: List[int]):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

# Training Pipeline
class GraphTrainer:
    """Training pipeline for graph neural networks"""
    
    def __init__(self, model: nn.Module, task_type: str = 'node_classification'):
        self.model = model
        self.task_type = task_type
        self.optimizer = None
        self.criterion = None
    
    def setup_training(self, learning_rate: float = 0.01):
        """Setup optimizer and loss function"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
    
    def train_node_classification(self, data: Union['Data', Dict], 
                                train_mask: torch.Tensor,
                                val_mask: torch.Tensor,
                                epochs: int = 100):
        """Train model for node classification"""
        
        if TORCH_GEOMETRIC_AVAILABLE and hasattr(data, 'x'):
            x, edge_index, y = data.x, data.edge_index, data.y
        else:
            x, edge_index, y = data['x'], data['edge_index'], data['y']
        
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(x, edge_index)
            
            # Compute loss only on training nodes
            loss = self.criterion(out[train_mask], y[train_mask])
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Validation
            if epoch % 20 == 0:
                val_acc = self._evaluate_node_classification(
                    x, edge_index, y, val_mask
                )
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    def train_graph_classification(self, data_loader, val_loader, epochs: int = 100):
        """Train model for graph classification"""
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data, batch_labels in data_loader:
                self.optimizer.zero_grad()
                
                if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
                    out = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                else:
                    # Handle single graphs
                    out = self.model(batch_data['x'], batch_data['edge_index'])
                
                loss = self.criterion(out, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Validation
            if epoch % 20 == 0:
                val_acc = self._evaluate_graph_classification(val_loader)
                print(f'Epoch {epoch:03d}, Loss: {epoch_loss/num_batches:.4f}, Val Acc: {val_acc:.4f}')
    
    def _evaluate_node_classification(self, x: torch.Tensor, edge_index: torch.Tensor,
                                    y: torch.Tensor, mask: torch.Tensor) -> float:
        """Evaluate node classification accuracy"""
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(x, edge_index)
            pred = out[mask].argmax(dim=1)
            correct = (pred == y[mask]).float().sum()
            accuracy = correct / mask.sum()
        
        self.model.train()
        return accuracy.item()
    
    def _evaluate_graph_classification(self, data_loader) -> float:
        """Evaluate graph classification accuracy"""
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                if TORCH_GEOMETRIC_AVAILABLE and hasattr(batch_data, 'x'):
                    out = self.model(batch_data.x, batch_data.edge_index, batch_data.batch)
                else:
                    out = self.model(batch_data['x'], batch_data['edge_index'])
                
                pred = out.argmax(dim=1)
                correct += (pred == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        self.model.train()
        return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    print("PyTorch Geometric Graph Processing")
    print("=" * 38)
    
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("PyTorch Geometric not available. Demonstrating with fallback implementations.")
    
    print("\n1. Graph Creation and Manipulation")
    print("-" * 37)
    
    graph_creator = GraphCreator()
    
    # Create simple graph
    simple_graph = graph_creator.create_simple_graph()
    
    if isinstance(simple_graph, dict):
        print(f"Simple graph - Nodes: {simple_graph['x'].shape[0]}, Edges: {simple_graph['edge_index'].shape[1]}")
    else:
        print(f"Simple graph - Nodes: {simple_graph.num_nodes}, Edges: {simple_graph.num_edges}")
    
    # Create molecular graph
    atoms = ['C', 'C', 'O', 'N']
    bonds = [(0, 1), (1, 2), (2, 3)]
    molecular_graph = graph_creator.create_molecular_graph(atoms, bonds)
    
    print(f"Molecular graph created with {len(atoms)} atoms and {len(bonds)} bonds")
    
    # Create random graph
    random_graph = GraphUtils.create_random_graph(num_nodes=10, num_edges=15, num_features=5)
    
    print("✓ Random graph created")
    
    print("\n2. Graph Statistics")
    print("-" * 21)
    
    # Compute statistics for different graphs
    graphs = [
        ("Simple", simple_graph),
        ("Molecular", molecular_graph),
        ("Random", random_graph)
    ]
    
    for name, graph in graphs:
        stats = GraphUtils.compute_graph_statistics(graph)
        print(f"\n{name} Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n3. Graph Neural Network Models")
    print("-" * 34)
    
    # Model parameters
    num_features = 5
    hidden_channels = 16
    num_classes = 3
    
    # Create different GNN models
    models = {
        'GCN': GCNModel(num_features, hidden_channels, num_classes),
        'GraphSAGE': GraphSAGE(num_features, hidden_channels, num_classes),
        'GAT': GATModel(num_features, hidden_channels, num_classes, heads=4)
    }
    
    # Test forward pass
    test_x = torch.randn(10, num_features)
    test_edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    
    print("Model forward pass results:")
    for name, model in models.items():
        try:
            output = model(test_x, test_edge_index)
            print(f"  {name}: Output shape {output.shape}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    print("\n4. Node Classification Task")
    print("-" * 29)
    
    # Create synthetic node classification data
    num_nodes = 100
    node_features = torch.randn(num_nodes, num_features)
    
    # Create edges (small-world network)
    edges = []
    for i in range(num_nodes):
        # Connect to next few nodes (circular)
        for j in range(1, 4):
            target = (i + j) % num_nodes
            edges.append([i, target])
        
        # Add some random long-range connections
        if torch.rand(1) < 0.1:
            target = torch.randint(0, num_nodes, (1,)).item()
            if target != i:
                edges.append([i, target])
    
    edge_index = torch.tensor(edges).t().contiguous()
    node_labels = torch.randint(0, num_classes, (num_nodes,))
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[:60] = True
    val_mask[60:80] = True
    test_mask[80:] = True
    
    # Create data object
    if TORCH_GEOMETRIC_AVAILABLE:
        node_data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    else:
        node_data = {'x': node_features, 'edge_index': edge_index, 'y': node_labels}
    
    print(f"Node classification dataset: {num_nodes} nodes, {edge_index.shape[1]} edges")
    
    # Train GCN model
    gcn_model = GCNModel(num_features, hidden_channels, num_classes)
    trainer = GraphTrainer(gcn_model, task_type='node_classification')
    trainer.setup_training(learning_rate=0.01)
    
    print("Training GCN for node classification...")
    trainer.train_node_classification(node_data, train_mask, val_mask, epochs=50)
    
    print("\n5. Graph Classification Task")
    print("-" * 30)
    
    # Create synthetic graph classification dataset
    graphs = []
    graph_labels = []
    
    for i in range(50):
        # Create random graphs with different properties
        if i < 25:
            # Type A: Small, dense graphs
            num_nodes = torch.randint(10, 20, (1,)).item()
            num_edges = torch.randint(20, 40, (1,)).item()
            label = 0
        else:
            # Type B: Larger, sparse graphs
            num_nodes = torch.randint(20, 30, (1,)).item()
            num_edges = torch.randint(30, 50, (1,)).item()
            label = 1
        
        graph = GraphUtils.create_random_graph(num_nodes, num_edges, num_features)
        graphs.append(graph)
        graph_labels.append(label)
    
    # Create dataset
    graph_dataset = GraphDataset(graphs, graph_labels)
    
    # Split into train/val
    train_size = int(0.8 * len(graph_dataset))
    val_size = len(graph_dataset) - train_size
    
    train_graphs = graphs[:train_size]
    train_labels = graph_labels[:train_size]
    val_graphs = graphs[train_size:]
    val_labels = graph_labels[train_size:]
    
    print(f"Graph classification dataset: {len(graphs)} graphs")
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Create simple data loaders (without PyG DataLoader)
    def simple_collate(batch):
        """Simple collate function for graph batches"""
        graphs, labels = zip(*batch)
        return graphs[0], torch.tensor(labels)  # For demo, just take first graph
    
    # Train graph classification model
    graph_model = GraphClassificationModel(num_features, hidden_channels, 2)
    graph_trainer = GraphTrainer(graph_model, task_type='graph_classification')
    graph_trainer.setup_training(learning_rate=0.01)
    
    print("Training model for graph classification...")
    
    # Simplified training for demo
    for epoch in range(20):
        total_loss = 0.0
        for i in range(0, len(train_graphs), 4):  # Batch size 4
            batch_graphs = train_graphs[i:i+4]
            batch_labels = torch.tensor(train_labels[i:i+4])
            
            graph_trainer.optimizer.zero_grad()
            
            # Process each graph in batch
            batch_outputs = []
            for graph in batch_graphs:
                if isinstance(graph, dict):
                    out = graph_model(graph['x'], graph['edge_index'])
                else:
                    out = graph_model(graph.x, graph.edge_index)
                batch_outputs.append(out)
            
            batch_output = torch.cat(batch_outputs, dim=0)
            loss = graph_trainer.criterion(batch_output, batch_labels)
            
            loss.backward()
            graph_trainer.optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Loss: {total_loss:.4f}")
    
    print("\n6. Custom Message Passing")
    print("-" * 27)
    
    if TORCH_GEOMETRIC_AVAILABLE:
        # Create custom message passing layer
        custom_mp = CustomMessagePassing(num_features, hidden_channels)
        
        # Test custom layer
        test_output = custom_mp(test_x, test_edge_index)
        print(f"Custom message passing output shape: {test_output.shape}")
    else:
        print("Custom message passing requires PyTorch Geometric")
    
    print("\n7. Graph Processing Best Practices")
    print("-" * 38)
    
    best_practices = [
        "Choose appropriate GNN architecture for your task",
        "Use proper graph normalization and feature scaling",
        "Handle graph size variations with appropriate pooling",
        "Apply regularization (dropout, weight decay) to prevent overfitting",
        "Use attention mechanisms for interpretability",
        "Consider graph augmentation for data scarce scenarios",
        "Monitor for over-smoothing in deep GNNs",
        "Use appropriate evaluation metrics for graph tasks",
        "Handle heterogeneous graphs with type-specific layers",
        "Consider computational complexity for large graphs",
        "Use mini-batching for scalability",
        "Apply proper train/val/test splits avoiding data leakage"
    ]
    
    print("Graph Processing Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n8. Common Graph Learning Tasks")
    print("-" * 33)
    
    graph_tasks = {
        "Node Classification": "Classify nodes in a graph (e.g., social network roles)",
        "Edge Prediction": "Predict missing or future edges",
        "Graph Classification": "Classify entire graphs (e.g., molecular property prediction)",
        "Node Regression": "Predict continuous values for nodes",
        "Graph Generation": "Generate new graphs with desired properties",
        "Community Detection": "Find clusters or communities in graphs",
        "Graph Matching": "Find correspondences between graphs",
        "Anomaly Detection": "Detect unusual patterns in graph structure"
    }
    
    print("Common Graph Learning Tasks:")
    for task, description in graph_tasks.items():
        print(f"  {task}: {description}")
    
    print("\n9. Applications and Domains")
    print("-" * 28)
    
    applications = [
        "Social Networks: Friend recommendation, influence analysis",
        "Molecular Chemistry: Drug discovery, property prediction",
        "Knowledge Graphs: Entity relationship modeling, reasoning",
        "Computer Networks: Routing optimization, anomaly detection",
        "Recommendation Systems: User-item interaction modeling",
        "Transportation: Route planning, traffic flow optimization",
        "Biology: Protein interaction networks, gene regulatory networks",
        "Finance: Fraud detection, risk assessment"
    ]
    
    print("Graph Learning Applications:")
    for application in applications:
        print(f"  - {application}")
    
    print("\nPyTorch Geometric graph processing demonstration completed!")
    print("Key components covered:")
    print("  - Graph data creation and manipulation")
    print("  - Multiple GNN architectures (GCN, GraphSAGE, GAT)")
    print("  - Node and graph classification tasks")
    print("  - Custom message passing layers")
    print("  - Training pipelines and evaluation")
    print("  - Graph statistics and visualization")
    
    print("\nPyTorch Geometric enables:")
    print("  - Efficient graph neural network implementations")
    print("  - Rich set of graph datasets and transforms")
    print("  - Scalable graph processing with batching")
    print("  - Integration with PyTorch ecosystem")
    print("  - State-of-the-art GNN architectures")