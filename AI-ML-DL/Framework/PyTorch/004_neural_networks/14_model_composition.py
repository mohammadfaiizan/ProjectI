#!/usr/bin/env python3
"""PyTorch Model Composition - Composing complex models"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Model Composition Overview ===")

print("Model composition enables:")
print("1. Complex architecture design")
print("2. Multi-branch and multi-task models")
print("3. Modular component reuse")
print("4. Hierarchical model building")
print("5. Model ensembling strategies")

print("\n=== Basic Model Composition ===")

# Simple composition with functional building blocks
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class SimpleComposedModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Compose with building blocks
        self.stem = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.res_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test basic composition
composed_model = SimpleComposedModel()
test_input = torch.randn(2, 3, 224, 224)
output = composed_model(test_input)

print(f"Composed model output: {output.shape}")
print(f"Model parameters: {sum(p.numel() for p in composed_model.parameters())}")

print("\n=== Multi-Branch Architecture ===")

class MultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Shared backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Multiple branches
        self.branch_a = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Classification
        )
        
        self.branch_b = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # Regression
        )
        
        self.branch_c = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()        # Segmentation-like output
        )
    
    def forward(self, x, branches=['a', 'b', 'c']):
        # Shared feature extraction
        features = self.backbone(x)
        
        outputs = {}
        
        if 'a' in branches:
            outputs['classification'] = self.branch_a(features)
        
        if 'b' in branches:
            outputs['regression'] = self.branch_b(features)
        
        if 'c' in branches:
            outputs['segmentation'] = self.branch_c(features)
        
        return outputs

# Test multi-branch model
multi_branch = MultiBranchModel()
multi_input = torch.randn(2, 3, 32, 32)

# All branches
all_outputs = multi_branch(multi_input)
print("Multi-branch outputs:")
for branch, output in all_outputs.items():
    print(f"  {branch}: {output.shape}")

# Selective branches
partial_outputs = multi_branch(multi_input, branches=['a', 'c'])
print(f"Partial branches: {list(partial_outputs.keys())}")

print("\n=== Multi-Scale Architecture ===")

class MultiScaleModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Multiple scale processors
        self.scale_1 = self._make_scale_branch(3, 32)    # Original scale
        self.scale_2 = self._make_scale_branch(3, 32)    # 1/2 scale
        self.scale_4 = self._make_scale_branch(3, 32)    # 1/4 scale
        
        # Feature fusion
        self.fusion_conv = nn.Conv2d(96, 64, 1)  # 32*3 = 96 channels
        self.fusion_bn = nn.BatchNorm2d(64)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
    
    def _make_scale_branch(self, in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Get input dimensions
        _, _, h, w = x.size()
        
        # Process at different scales
        feat_1 = self.scale_1(x)
        
        # Downsample for other scales
        x_2 = F.interpolate(x, size=(h//2, w//2), mode='bilinear', align_corners=False)
        feat_2 = self.scale_2(x_2)
        feat_2 = F.interpolate(feat_2, size=(h, w), mode='bilinear', align_corners=False)
        
        x_4 = F.interpolate(x, size=(h//4, w//4), mode='bilinear', align_corners=False)
        feat_4 = self.scale_4(x_4)
        feat_4 = F.interpolate(feat_4, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fuse features
        fused = torch.cat([feat_1, feat_2, feat_4], dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))
        
        # Classify
        x = self.global_pool(fused)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Test multi-scale model
multi_scale = MultiScaleModel()
scale_input = torch.randn(2, 3, 64, 64)
scale_output = multi_scale(scale_input)

print(f"Multi-scale output: {scale_output.shape}")

print("\n=== Attention-Based Composition ===")

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class AttentionComposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Feature extractors
        self.low_level = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32)
        )
        
        self.mid_level = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64)
        )
        
        self.high_level = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        
        # Attention gates
        self.low_attention = AttentionGate(32)
        self.mid_attention = AttentionGate(64)
        self.high_attention = AttentionGate(128)
        
        # Feature integration
        self.integrate = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # Extract features at different levels
        low_feat = self.low_level(x)
        mid_feat = self.mid_level(low_feat)
        high_feat = self.high_level(mid_feat)
        
        # Apply attention
        low_feat = self.low_attention(low_feat)
        mid_feat = self.mid_attention(mid_feat)
        high_feat = self.high_attention(high_feat)
        
        # Resize to same spatial dimensions
        h, w = low_feat.size(2), low_feat.size(3)
        mid_feat = F.interpolate(mid_feat, size=(h, w), mode='bilinear', align_corners=False)
        high_feat = F.interpolate(high_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate and integrate
        combined = torch.cat([low_feat, mid_feat, high_feat], dim=1)
        integrated = self.integrate(combined)
        
        # Classify
        output = self.classifier(integrated)
        
        return output

# Test attention model
attention_model = AttentionComposedModel()
attn_input = torch.randn(2, 3, 32, 32)
attn_output = attention_model(attn_input)

print(f"Attention model output: {attn_output.shape}")

print("\n=== Model Ensembling ===")

class ModelEnsemble(nn.Module):
    def __init__(self, models, ensemble_method='average'):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        elif ensemble_method == 'learned':
            self.fusion_layer = nn.Linear(len(models), 1)
    
    def forward(self, x):
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Combine predictions
        if self.ensemble_method == 'average':
            return torch.stack(predictions).mean(dim=0)
        
        elif self.ensemble_method == 'max':
            return torch.stack(predictions).max(dim=0)[0]
        
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            weighted_preds = []
            for pred, weight in zip(predictions, weights):
                weighted_preds.append(pred * weight)
            return sum(weighted_preds)
        
        elif self.ensemble_method == 'learned':
            # Stack predictions and learn combination
            stacked = torch.stack(predictions, dim=-1)  # [batch, classes, num_models]
            weights = F.softmax(self.fusion_layer(stacked), dim=-1)
            return (stacked * weights).sum(dim=-1)
        
        else:
            return torch.stack(predictions).mean(dim=0)

# Create ensemble
model1 = SimpleComposedModel(num_classes=5)
model2 = SimpleComposedModel(num_classes=5)
model3 = SimpleComposedModel(num_classes=5)

ensemble_avg = ModelEnsemble([model1, model2, model3], 'average')
ensemble_weighted = ModelEnsemble([model1, model2, model3], 'weighted')

ensemble_input = torch.randn(3, 3, 32, 32)

avg_output = ensemble_avg(ensemble_input)
weighted_output = ensemble_weighted(ensemble_input)

print(f"Ensemble average output: {avg_output.shape}")
print(f"Ensemble weighted output: {weighted_output.shape}")
print(f"Ensemble weights: {F.softmax(ensemble_weighted.weights, dim=0)}")

print("\n=== Dynamic Model Composition ===")

class DynamicComposer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Available building blocks
        self.blocks = nn.ModuleDict({
            'conv_3x3': ConvBlock(64, 64, 3),
            'conv_5x5': ConvBlock(64, 64, 5, padding=2),
            'conv_1x1': ConvBlock(64, 64, 1, padding=0),
            'residual': ResidualBlock(64),
            'identity': nn.Identity()
        })
        
        # Path selector (learnable)
        self.path_selector = nn.Parameter(torch.randn(len(self.blocks)))
        
        # Input/output adapters
        self.input_adapter = ConvBlock(3, 64)
        self.output_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x, use_gumbel=True, temperature=1.0):
        x = self.input_adapter(x)
        
        if use_gumbel:
            # Gumbel softmax for differentiable selection
            gumbel_weights = F.gumbel_softmax(self.path_selector, tau=temperature, hard=False)
        else:
            # Soft selection
            gumbel_weights = F.softmax(self.path_selector / temperature, dim=0)
        
        # Weighted combination of all paths
        output = 0
        for weight, (name, block) in zip(gumbel_weights, self.blocks.items()):
            output += weight * block(x)
        
        return self.output_adapter(output)
    
    def get_path_probabilities(self):
        """Get current path selection probabilities"""
        return F.softmax(self.path_selector, dim=0)

# Test dynamic composition
dynamic_composer = DynamicComposer()
dynamic_input = torch.randn(2, 3, 32, 32)

dynamic_output = dynamic_composer(dynamic_input)
path_probs = dynamic_composer.get_path_probabilities()

print(f"Dynamic composer output: {dynamic_output.shape}")
print("Path probabilities:")
for (name, _), prob in zip(dynamic_composer.blocks.items(), path_probs):
    print(f"  {name}: {prob.item():.4f}")

print("\n=== Hierarchical Model Composition ===")

class HierarchicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Level 1: Low-level features
        self.level1 = nn.ModuleDict({
            'edge_detector': nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU()
            ),
            'texture_detector': nn.Sequential(
                nn.Conv2d(3, 16, 5, padding=2),
                nn.ReLU()
            )
        })
        
        # Level 2: Mid-level features
        self.level2 = nn.ModuleDict({
            'shape_detector': nn.Sequential(
                ConvBlock(32, 32),  # 16+16 from level1
                ConvBlock(32, 32)
            ),
            'pattern_detector': nn.Sequential(
                ConvBlock(32, 32),
                ConvBlock(32, 32)
            )
        })
        
        # Level 3: High-level features
        self.level3 = nn.Sequential(
            ConvBlock(64, 64),  # 32+32 from level2
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Final classifier
        self.classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        # Level 1 processing
        edge_features = self.level1['edge_detector'](x)
        texture_features = self.level1['texture_detector'](x)
        level1_out = torch.cat([edge_features, texture_features], dim=1)
        
        # Level 2 processing
        shape_features = self.level2['shape_detector'](level1_out)
        pattern_features = self.level2['pattern_detector'](level1_out)
        level2_out = torch.cat([shape_features, pattern_features], dim=1)
        
        # Level 3 processing
        level3_out = self.level3(level2_out)
        
        # Classification
        output = self.classifier(level3_out.view(level3_out.size(0), -1))
        
        return output

# Test hierarchical model
hierarchical_model = HierarchicalModel()
hier_input = torch.randn(2, 3, 64, 64)
hier_output = hierarchical_model(hier_input)

print(f"Hierarchical model output: {hier_output.shape}")

print("\n=== Model Factory for Composition ===")

class CompositionFactory:
    """Factory for creating composed models"""
    
    @staticmethod
    def create_multi_task_model(shared_config, task_configs):
        """Create multi-task model with shared backbone"""
        
        class MultiTaskModel(nn.Module):
            def __init__(self, shared_config, task_configs):
                super().__init__()
                
                # Shared backbone
                self.shared_layers = CompositionFactory._build_layers(shared_config)
                
                # Task-specific heads
                self.task_heads = nn.ModuleDict()
                for task_name, task_config in task_configs.items():
                    self.task_heads[task_name] = CompositionFactory._build_layers(task_config)
            
            def forward(self, x, tasks=None):
                # Shared feature extraction
                shared_features = self.shared_layers(x)
                
                # Task-specific processing
                outputs = {}
                tasks_to_process = tasks or list(self.task_heads.keys())
                
                for task in tasks_to_process:
                    if task in self.task_heads:
                        outputs[task] = self.task_heads[task](shared_features)
                
                return outputs
        
        return MultiTaskModel(shared_config, task_configs)
    
    @staticmethod
    def _build_layers(config):
        """Build layers from configuration"""
        layers = []
        
        for layer_config in config:
            layer_type = layer_config['type']
            
            if layer_type == 'conv':
                layers.append(nn.Conv2d(**layer_config['params']))
            elif layer_type == 'linear':
                layers.append(nn.Linear(**layer_config['params']))
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'pool':
                layers.append(nn.AdaptiveAvgPool2d(**layer_config['params']))
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'dropout':
                layers.append(nn.Dropout(**layer_config['params']))
        
        return nn.Sequential(*layers)

# Test composition factory
shared_config = [
    {'type': 'conv', 'params': {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'padding': 1}},
    {'type': 'relu'},
    {'type': 'conv', 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
    {'type': 'relu'},
    {'type': 'pool', 'params': {'output_size': (4, 4)}},
    {'type': 'flatten'}
]

task_configs = {
    'classification': [
        {'type': 'linear', 'params': {'in_features': 1024, 'out_features': 128}},
        {'type': 'relu'},
        {'type': 'dropout', 'params': {'p': 0.5}},
        {'type': 'linear', 'params': {'in_features': 128, 'out_features': 10}}
    ],
    'regression': [
        {'type': 'linear', 'params': {'in_features': 1024, 'out_features': 64}},
        {'type': 'relu'},
        {'type': 'linear', 'params': {'in_features': 64, 'out_features': 1}}
    ]
}

factory_model = CompositionFactory.create_multi_task_model(shared_config, task_configs)
factory_input = torch.randn(2, 3, 32, 32)
factory_outputs = factory_model(factory_input)

print("Factory model outputs:")
for task, output in factory_outputs.items():
    print(f"  {task}: {output.shape}")

print("\n=== Model Composition Best Practices ===")

print("Model Composition Guidelines:")
print("1. Design reusable building blocks")
print("2. Use appropriate abstraction levels")
print("3. Consider computational efficiency")
print("4. Plan for feature sharing and fusion")
print("5. Design for modularity and testing")
print("6. Use factory patterns for complex compositions")
print("7. Consider memory and parameter sharing")

print("\nArchitectural Patterns:")
print("- Multi-branch: Parallel processing paths")
print("- Multi-scale: Different resolution processing")
print("- Hierarchical: Bottom-up feature building")
print("- Attention-based: Dynamic feature weighting")
print("- Ensemble: Multiple model combination")
print("- Dynamic: Learnable architecture selection")

print("\nComposition Strategies:")
print("- Early fusion: Combine features early")
print("- Late fusion: Combine final predictions")
print("- Intermediate fusion: Multiple fusion points")
print("- Attention fusion: Learned combination weights")
print("- Gating: Conditional feature routing")
print("- Residual connections: Skip pathways")

print("\nImplementation Tips:")
print("- Use ModuleDict for conditional execution")
print("- Use ModuleList for sequential processing")
print("- Consider parameter sharing for efficiency")
print("- Design for different input resolutions")
print("- Plan for feature map size compatibility")
print("- Use appropriate interpolation for resizing")

print("\n=== Model Composition Complete ===") 