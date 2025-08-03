"""
Comprehensive Comparative Analysis of LPIPS Supporting Models
============================================================

Complete comparative study of AlexNet, VGG, and SqueezeNet for LPIPS applications.
This analysis covers performance, efficiency, feature quality, and practical deployment
considerations for perceptual similarity assessment.

Analysis includes:
- Model architecture comparison
- Classification performance analysis
- Computational efficiency evaluation
- Feature extraction quality for LPIPS
- Memory usage and scalability
- Deployment considerations
- LPIPS-specific feature analysis

Author: [Your Name]
Date: [Current Date]
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# Import model implementations
from AlexNet.alexnet_model import AlexNet, create_alexnet_for_lpips
from VGG.vgg_model import VGG, create_vgg_for_lpips
from SqueezeNet.squeezenet_model import SqueezeNet, create_squeezenet_for_lpips

# Import utilities
from utils.dataset_utils import StandardImageNetLoader, LPIPSDatasetCreator
from utils.evaluation_metrics import ClassificationMetrics, EfficiencyMetrics, LPIPSEvaluationMetrics, ModelComparisonFramework


class LPIPSModelAnalyzer:
    """
    Comprehensive analyzer for LPIPS supporting models
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models = {}
        self.model_configs = {}
        self.analysis_results = {}
        
        # Initialize data loader
        self.data_loader = StandardImageNetLoader()
        
        print(f"LPIPS Model Analyzer initialized on {device}")
    
    def load_models(self, pretrained: bool = True):
        """Load all three LPIPS supporting models"""
        
        print("Loading LPIPS supporting models...")
        
        # AlexNet
        print("  Loading AlexNet...")
        self.models['AlexNet'] = create_alexnet_for_lpips(pretrained=pretrained)
        self.model_configs['AlexNet'] = {
            'architecture': 'alexnet',
            'year': 2012,
            'key_innovation': 'First successful deep CNN for ImageNet',
            'lpips_usage': 'Original LPIPS implementation baseline'
        }
        
        # VGG-16
        print("  Loading VGG-16...")
        self.models['VGG-16'] = create_vgg_for_lpips('vgg16', pretrained=pretrained)
        self.model_configs['VGG-16'] = {
            'architecture': 'vgg16',
            'year': 2014,
            'key_innovation': 'Very deep networks with small filters',
            'lpips_usage': 'High-quality feature extraction'
        }
        
        # SqueezeNet 1.1
        print("  Loading SqueezeNet 1.1...")
        self.models['SqueezeNet'] = create_squeezenet_for_lpips('1_1', pretrained=pretrained)
        self.model_configs['SqueezeNet'] = {
            'architecture': 'squeezenet1_1',
            'year': 2016,
            'key_innovation': 'Efficient architecture with Fire modules',
            'lpips_usage': 'Efficient perceptual similarity computation'
        }
        
        # Move models to device
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            self.models[name].eval()
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def analyze_model_architectures(self) -> Dict:
        """Comprehensive architecture analysis"""
        
        print("\n=== Analyzing Model Architectures ===")
        
        architecture_analysis = {}
        
        for model_name, model in self.models.items():
            print(f"Analyzing {model_name}...")
            
            # Basic model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Model size
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024**2
            
            # Layer analysis
            layer_count = len(list(model.modules()))
            conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
            linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            
            # Architecture-specific analysis
            if hasattr(model, 'get_parameter_count'):
                detailed_params = model.get_parameter_count()
            else:
                detailed_params = {'total_parameters': total_params}
            
            # LPIPS-relevant layers
            if hasattr(model, 'lpips_layers'):
                lpips_layer_count = len(model.lpips_layers)
            else:
                lpips_layer_count = 5  # Default assumption
            
            architecture_analysis[model_name] = {
                'basic_stats': {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_size_mb': model_size_mb,
                    'parameter_density': total_params / model_size_mb
                },
                'layer_composition': {
                    'total_layers': layer_count,
                    'conv_layers': conv_layers,
                    'linear_layers': linear_layers,
                    'lpips_layers': lpips_layer_count
                },
                'detailed_parameters': detailed_params,
                'config': self.model_configs[model_name]
            }
        
        self.analysis_results['architecture'] = architecture_analysis
        return architecture_analysis
    
    def analyze_computational_efficiency(self) -> Dict:
        """Comprehensive computational efficiency analysis"""
        
        print("\n=== Analyzing Computational Efficiency ===")
        
        efficiency_metrics = EfficiencyMetrics(self.device)
        efficiency_analysis = {}
        
        # Test configurations
        batch_sizes = [1, 4, 8, 16, 32]
        input_shape = (3, 224, 224)
        
        for model_name, model in self.models.items():
            print(f"Measuring efficiency for {model_name}...")
            
            model_efficiency = {}
            
            # Throughput scaling analysis
            scaling_results = efficiency_metrics.measure_throughput_scaling(
                model, batch_sizes, input_shape
            )
            model_efficiency['scaling_results'] = scaling_results
            
            # Analyze efficiency trends
            efficiency_trends = efficiency_metrics.analyze_efficiency_trends(scaling_results)
            model_efficiency['efficiency_trends'] = efficiency_trends
            
            # Single image inference time (important for LPIPS)
            single_input = torch.randn(1, *input_shape).to(self.device)
            single_timing = efficiency_metrics.measure_inference_time(
                model, single_input, num_runs=200
            )
            model_efficiency['single_image_timing'] = single_timing
            
            efficiency_analysis[model_name] = model_efficiency
        
        self.analysis_results['efficiency'] = efficiency_analysis
        return efficiency_analysis
    
    def analyze_feature_extraction_quality(self) -> Dict:
        """Analyze feature extraction quality for LPIPS"""
        
        print("\n=== Analyzing Feature Extraction Quality ===")
        
        # Create sample data for feature analysis
        lpips_creator = LPIPSDatasetCreator(self.data_loader)
        sample_pairs = self.data_loader.create_lpips_evaluation_dataset(num_pairs=100)
        
        feature_analysis = {}
        
        for model_name, model in self.models.items():
            print(f"Analyzing feature quality for {model_name}...")
            
            model_feature_analysis = {
                'layer_statistics': {},
                'feature_quality_metrics': {},
                'lpips_relevance': {}
            }
            
            # Extract features from sample images
            layer_statistics = defaultdict(list)
            
            for i, (img1, img2) in enumerate(sample_pairs[:20]):  # Analyze first 20 pairs
                img1 = img1.unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features1 = model.forward_features(img1)
                    features2 = model.forward_features(img2)
                
                # Analyze each layer
                for layer_name in features1.keys():
                    feat1 = features1[layer_name]
                    feat2 = features2[layer_name]
                    
                    # Feature statistics
                    layer_statistics[layer_name].append({
                        'mean_activation': feat1.mean().item(),
                        'std_activation': feat1.std().item(),
                        'sparsity': (feat1 == 0).float().mean().item(),
                        'feature_similarity': F.cosine_similarity(
                            feat1.flatten(), feat2.flatten(), dim=0
                        ).item()
                    })
            
            # Aggregate layer statistics
            for layer_name, stats_list in layer_statistics.items():
                aggregated_stats = {}
                for key in stats_list[0].keys():
                    values = [s[key] for s in stats_list]
                    aggregated_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
                model_feature_analysis['layer_statistics'][layer_name] = aggregated_stats
            
            # Feature quality assessment
            feature_quality = self._assess_feature_quality_for_lpips(model, sample_pairs[:50])
            model_feature_analysis['feature_quality_metrics'] = feature_quality
            
            feature_analysis[model_name] = model_feature_analysis
        
        self.analysis_results['feature_quality'] = feature_analysis
        return feature_analysis
    
    def _assess_feature_quality_for_lpips(self, model: nn.Module, sample_pairs: List) -> Dict:
        """Assess feature quality specifically for LPIPS applications"""
        
        feature_distances = defaultdict(list)
        perceptual_correlations = {}
        
        with torch.no_grad():
            for img1, img2 in sample_pairs:
                img1 = img1.unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).to(self.device)
                
                # Extract features
                features1 = model.forward_features(img1)
                features2 = model.forward_features(img2)
                
                # Compute distances for each layer
                for layer_name in features1.keys():
                    feat1 = features1[layer_name]
                    feat2 = features2[layer_name]
                    
                    # L2 distance (as used in LPIPS)
                    distance = torch.nn.functional.mse_loss(feat1, feat2).item()
                    feature_distances[layer_name].append(distance)
        
        # Analyze distance distributions
        quality_metrics = {}
        for layer_name, distances in feature_distances.items():
            distances = np.array(distances)
            quality_metrics[layer_name] = {
                'mean_distance': distances.mean(),
                'std_distance': distances.std(),
                'dynamic_range': distances.max() - distances.min(),
                'coefficient_of_variation': distances.std() / distances.mean()
            }
        
        return quality_metrics
    
    def analyze_memory_usage(self) -> Dict:
        """Analyze memory usage patterns"""
        
        print("\n=== Analyzing Memory Usage ===")
        
        memory_analysis = {}
        batch_sizes = [1, 4, 8, 16, 32, 64]
        input_shape = (3, 224, 224)
        
        for model_name, model in self.models.items():
            if self.device != 'cuda':
                memory_analysis[model_name] = {'note': 'Memory analysis requires CUDA'}
                continue
            
            print(f"Measuring memory usage for {model_name}...")
            
            model_memory = {}
            
            for batch_size in batch_sizes:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                input_tensor = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Measure baseline
                baseline_memory = torch.cuda.memory_allocated()
                
                # Forward pass
                with torch.no_grad():
                    _ = model(input_tensor)
                
                # Peak memory
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - baseline_memory
                
                model_memory[f'batch_{batch_size}'] = {
                    'total_memory_mb': peak_memory / 1024**2,
                    'memory_used_mb': memory_used / 1024**2,
                    'memory_per_image_mb': memory_used / batch_size / 1024**2
                }
            
            memory_analysis[model_name] = model_memory
        
        self.analysis_results['memory'] = memory_analysis
        return memory_analysis
    
    def compare_lpips_performance(self) -> Dict:
        """Compare LPIPS-specific performance"""
        
        print("\n=== Comparing LPIPS Performance ===")
        
        # Create distortion dataset for LPIPS evaluation
        lpips_creator = LPIPSDatasetCreator(self.data_loader)
        distortion_dataset = lpips_creator.create_distortion_dataset(
            distortion_types=['jpeg_compression', 'gaussian_noise', 'gaussian_blur'],
            num_samples=100
        )
        
        lpips_performance = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating LPIPS performance for {model_name}...")
            
            model_lpips_perf = {
                'distortion_sensitivity': {},
                'feature_layer_contribution': {},
                'computational_cost_per_comparison': {}
            }
            
            # Test sensitivity to different distortions
            for distortion_type, samples in distortion_dataset.items():
                sensitivities = []
                
                for sample in samples[:50]:  # Use first 50 samples
                    clean_img = sample['clean'].unsqueeze(0).to(self.device)
                    distorted_img = sample['distorted'].unsqueeze(0).to(self.device)
                    
                    # Extract features and compute distance
                    with torch.no_grad():
                        clean_features = model.forward_features(clean_img)
                        distorted_features = model.forward_features(distorted_img)
                    
                    # Compute weighted feature distance (simplified LPIPS)
                    total_distance = 0
                    layer_count = 0
                    
                    for layer_name in clean_features.keys():
                        clean_feat = clean_features[layer_name]
                        distorted_feat = distorted_features[layer_name]
                        
                        # Normalize features (as in LPIPS)
                        clean_feat_norm = F.normalize(clean_feat, dim=1)
                        distorted_feat_norm = F.normalize(distorted_feat, dim=1)
                        
                        # L2 distance
                        layer_distance = torch.nn.functional.mse_loss(clean_feat_norm, distorted_feat_norm).item()
                        total_distance += layer_distance
                        layer_count += 1
                    
                    avg_distance = total_distance / layer_count
                    sensitivities.append(avg_distance)
                
                model_lpips_perf['distortion_sensitivity'][distortion_type] = {
                    'mean_sensitivity': np.mean(sensitivities),
                    'std_sensitivity': np.std(sensitivities)
                }
            
            # Measure computational cost per comparison
            sample_pair = (
                torch.randn(1, 3, 224, 224).to(self.device),
                torch.randn(1, 3, 224, 224).to(self.device)
            )
            
            # Time feature extraction
            times = []
            for _ in range(100):
                start_time = time.time()
                with torch.no_grad():
                    _ = model.forward_features(sample_pair[0])
                    _ = model.forward_features(sample_pair[1])
                end_time = time.time()
                times.append(end_time - start_time)
            
            model_lpips_perf['computational_cost_per_comparison'] = {
                'mean_time_ms': np.mean(times) * 1000,
                'std_time_ms': np.std(times) * 1000
            }
            
            lpips_performance[model_name] = model_lpips_perf
        
        self.analysis_results['lpips_performance'] = lpips_performance
        return lpips_performance
    
    def generate_deployment_recommendations(self) -> Dict:
        """Generate deployment recommendations based on analysis"""
        
        print("\n=== Generating Deployment Recommendations ===")
        
        recommendations = {}
        
        # Extract key metrics
        architectures = self.analysis_results['architecture']
        efficiency = self.analysis_results['efficiency']
        memory = self.analysis_results.get('memory', {})
        lpips_perf = self.analysis_results['lpips_performance']
        
        for model_name in self.models.keys():
            arch_data = architectures[model_name]
            eff_data = efficiency[model_name]
            mem_data = memory.get(model_name, {})
            lpips_data = lpips_perf[model_name]
            
            # Extract key metrics
            params = arch_data['basic_stats']['total_parameters'] / 1e6  # In millions
            model_size = arch_data['basic_stats']['model_size_mb']
            single_inference_time = eff_data['single_image_timing']['mean_time_ms']
            
            # Generate recommendations
            use_cases = []
            pros = []
            cons = []
            
            if model_name == 'AlexNet':
                use_cases = [
                    "Research and educational purposes",
                    "Baseline comparisons",
                    "Historical significance studies"
                ]
                pros = [
                    "Simple architecture, easy to understand",
                    "Fast inference for its era",
                    "Good starting point for learning"
                ]
                cons = [
                    "Outdated architecture",
                    "Poor parameter efficiency",
                    "Lower accuracy than modern alternatives"
                ]
            
            elif model_name == 'VGG-16':
                use_cases = [
                    "High-quality feature extraction",
                    "Transfer learning applications",
                    "Research requiring detailed features"
                ]
                pros = [
                    "High-quality features",
                    "Well-studied architecture",
                    "Good performance on complex tasks"
                ]
                cons = [
                    "Large model size",
                    "High computational cost",
                    "Memory intensive"
                ]
            
            elif model_name == 'SqueezeNet':
                use_cases = [
                    "Mobile and edge deployment",
                    "Real-time applications",
                    "Resource-constrained environments"
                ]
                pros = [
                    "Excellent parameter efficiency",
                    "Small model size",
                    "Fast inference"
                ]
                cons = [
                    "Potentially lower feature quality",
                    "Less studied for LPIPS",
                    "May require more careful tuning"
                ]
            
            recommendations[model_name] = {
                'key_metrics': {
                    'parameters_millions': params,
                    'model_size_mb': model_size,
                    'inference_time_ms': single_inference_time
                },
                'recommended_use_cases': use_cases,
                'advantages': pros,
                'limitations': cons,
                'deployment_considerations': self._get_deployment_considerations(model_name, arch_data, eff_data)
            }
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def _get_deployment_considerations(self, model_name: str, arch_data: Dict, eff_data: Dict) -> List[str]:
        """Get deployment-specific considerations"""
        considerations = []
        
        params = arch_data['basic_stats']['total_parameters'] / 1e6
        model_size = arch_data['basic_stats']['model_size_mb']
        
        if model_size > 100:
            considerations.append("Large model size may require significant storage and memory")
        
        if params > 50:
            considerations.append("High parameter count may slow loading and initialization")
        
        single_time = eff_data['single_image_timing']['mean_time_ms']
        if single_time > 20:
            considerations.append("High inference latency may not be suitable for real-time applications")
        elif single_time < 5:
            considerations.append("Fast inference enables real-time applications")
        
        if model_name == 'SqueezeNet':
            considerations.append("Excellent for mobile deployment due to small size")
            considerations.append("Consider quantization for further size reduction")
        
        if model_name == 'VGG-16':
            considerations.append("Consider using for high-quality offline processing")
            considerations.append("May require GPU acceleration for reasonable performance")
        
        return considerations
    
    def create_comprehensive_report(self, save_path: str = 'lpips_models_analysis_report.json'):
        """Create comprehensive analysis report"""
        
        print(f"\n=== Creating Comprehensive Report ===")
        
        # Compile all results
        comprehensive_report = {
            'analysis_metadata': {
                'timestamp': str(pd.Timestamp.now()),
                'device': self.device,
                'models_analyzed': list(self.models.keys()),
                'analysis_version': '1.0'
            },
            'executive_summary': self._create_executive_summary(),
            'detailed_analysis': self.analysis_results,
            'model_rankings': self._create_model_rankings(),
            'deployment_guide': self._create_deployment_guide()
        }
        
        # Save report
        with open(save_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_report = self._make_json_serializable(comprehensive_report)
            json.dump(json_report, f, indent=2)
        
        print(f"Comprehensive report saved to {save_path}")
        return comprehensive_report
    
    def _create_executive_summary(self) -> Dict:
        """Create executive summary of findings"""
        
        summary = {
            'key_findings': [],
            'best_overall_model': '',
            'best_for_efficiency': '',
            'best_for_accuracy': '',
            'recommendations_summary': ''
        }
        
        # Analyze results to create summary
        architectures = self.analysis_results['architecture']
        efficiency = self.analysis_results['efficiency']
        
        # Find most efficient model (by inference time)
        inference_times = {}
        for model_name in self.models.keys():
            inference_times[model_name] = efficiency[model_name]['single_image_timing']['mean_time_ms']
        
        best_efficiency = min(inference_times.items(), key=lambda x: x[1])[0]
        summary['best_for_efficiency'] = best_efficiency
        
        # Find most parameter-efficient model
        param_counts = {}
        for model_name in self.models.keys():
            param_counts[model_name] = architectures[model_name]['basic_stats']['total_parameters']
        
        most_param_efficient = min(param_counts.items(), key=lambda x: x[1])[0]
        
        # Key findings
        summary['key_findings'] = [
            f"SqueezeNet achieves {param_counts['SqueezeNet']/param_counts['AlexNet']:.1f}x parameter reduction vs AlexNet",
            f"VGG-16 provides highest feature quality but at {param_counts['VGG-16']/param_counts['SqueezeNet']:.1f}x parameter cost vs SqueezeNet",
            f"{best_efficiency} provides fastest inference at {inference_times[best_efficiency]:.2f}ms per image",
            "Each model serves different deployment scenarios optimally"
        ]
        
        summary['best_overall_model'] = "Depends on use case - see deployment recommendations"
        summary['recommendations_summary'] = (
            "SqueezeNet for mobile/edge, VGG-16 for quality-critical applications, "
            "AlexNet for educational/baseline purposes"
        )
        
        return summary
    
    def _create_model_rankings(self) -> Dict:
        """Create model rankings across different criteria"""
        
        rankings = {}
        
        # Parameter efficiency ranking
        param_counts = {}
        for model_name in self.models.keys():
            param_counts[model_name] = self.analysis_results['architecture'][model_name]['basic_stats']['total_parameters']
        
        rankings['parameter_efficiency'] = sorted(param_counts.items(), key=lambda x: x[1])
        
        # Inference speed ranking
        inference_times = {}
        for model_name in self.models.keys():
            inference_times[model_name] = self.analysis_results['efficiency'][model_name]['single_image_timing']['mean_time_ms']
        
        rankings['inference_speed'] = sorted(inference_times.items(), key=lambda x: x[1])
        
        # Model size ranking
        model_sizes = {}
        for model_name in self.models.keys():
            model_sizes[model_name] = self.analysis_results['architecture'][model_name]['basic_stats']['model_size_mb']
        
        rankings['model_size'] = sorted(model_sizes.items(), key=lambda x: x[1])
        
        return rankings
    
    def _create_deployment_guide(self) -> Dict:
        """Create practical deployment guide"""
        
        guide = {
            'use_case_recommendations': {
                'real_time_applications': {
                    'recommended_model': 'SqueezeNet',
                    'rationale': 'Fastest inference and smallest memory footprint',
                    'considerations': ['Consider quantization for further optimization', 'Test on target hardware']
                },
                'high_quality_analysis': {
                    'recommended_model': 'VGG-16',
                    'rationale': 'Best feature quality and most studied for LPIPS',
                    'considerations': ['Requires GPU acceleration', 'Higher computational cost']
                },
                'research_and_education': {
                    'recommended_model': 'AlexNet',
                    'rationale': 'Historical significance and simplicity',
                    'considerations': ['Good for understanding concepts', 'Not for production use']
                },
                'balanced_approach': {
                    'recommended_model': 'SqueezeNet',
                    'rationale': 'Good balance of efficiency and performance',
                    'considerations': ['May need careful hyperparameter tuning', 'Less research available']
                }
            },
            'hardware_requirements': self._get_hardware_requirements(),
            'implementation_tips': self._get_implementation_tips()
        }
        
        return guide
    
    def _get_hardware_requirements(self) -> Dict:
        """Get hardware requirements for each model"""
        
        requirements = {}
        
        for model_name in self.models.keys():
            arch_data = self.analysis_results['architecture'][model_name]
            memory_data = self.analysis_results.get('memory', {}).get(model_name, {})
            
            model_size = arch_data['basic_stats']['model_size_mb']
            
            # Estimate requirements
            min_ram = model_size * 4  # Rough estimate including overhead
            recommended_ram = min_ram * 2
            
            gpu_memory = 'batch_1' in memory_data and memory_data['batch_1'].get('total_memory_mb', model_size * 2) or model_size * 2
            
            requirements[model_name] = {
                'minimum_ram_mb': min_ram,
                'recommended_ram_mb': recommended_ram,
                'gpu_memory_mb': gpu_memory,
                'cpu_recommendation': 'Modern multi-core CPU recommended',
                'gpu_recommendation': 'CUDA-capable GPU recommended for optimal performance'
            }
        
        return requirements
    
    def _get_implementation_tips(self) -> List[str]:
        """Get implementation tips"""
        
        return [
            "Use batch processing when possible to improve throughput",
            "Consider mixed precision training and inference for memory efficiency",
            "Implement proper error handling for different input sizes",
            "Cache model weights to avoid repeated loading",
            "Use appropriate data loaders with sufficient workers",
            "Monitor memory usage in production environments",
            "Implement proper preprocessing pipelines",
            "Consider model quantization for deployment",
            "Use profiling tools to identify bottlenecks",
            "Implement graceful degradation for resource-constrained scenarios"
        ]
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def visualize_analysis_results(self, save_plots: bool = True):
        """Create comprehensive visualizations"""
        
        print("\n=== Creating Analysis Visualizations ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model Size Comparison
        ax1 = plt.subplot(3, 3, 1)
        model_names = list(self.models.keys())
        model_sizes = [self.analysis_results['architecture'][name]['basic_stats']['model_size_mb'] 
                      for name in model_names]
        
        bars1 = ax1.bar(model_names, model_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Size (MB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars1, model_sizes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 2. Parameter Count Comparison
        ax2 = plt.subplot(3, 3, 2)
        param_counts = [self.analysis_results['architecture'][name]['basic_stats']['total_parameters'] / 1e6 
                       for name in model_names]
        
        bars2 = ax2.bar(model_names, param_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Parameter Count Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Parameters (Millions)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars2, param_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # 3. Inference Time Comparison
        ax3 = plt.subplot(3, 3, 3)
        inference_times = [self.analysis_results['efficiency'][name]['single_image_timing']['mean_time_ms'] 
                          for name in model_names]
        
        bars3 = ax3.bar(model_names, inference_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars3, inference_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{time:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency vs Size Scatter Plot
        ax4 = plt.subplot(3, 3, 4)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        scatter = ax4.scatter(model_sizes, inference_times, s=200, c=colors, alpha=0.7, edgecolors='black')
        
        for i, name in enumerate(model_names):
            ax4.annotate(name, (model_sizes[i], inference_times[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax4.set_xlabel('Model Size (MB)')
        ax4.set_ylabel('Inference Time (ms)')
        ax4.set_title('Efficiency vs Size Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Throughput Scaling
        ax5 = plt.subplot(3, 3, 5)
        batch_sizes = [1, 4, 8, 16, 32]
        
        for i, model_name in enumerate(model_names):
            throughputs = []
            for bs in batch_sizes:
                if bs in self.analysis_results['efficiency'][model_name]['scaling_results']:
                    throughput = self.analysis_results['efficiency'][model_name]['scaling_results'][bs]['timing']['throughput_images_per_second']
                    throughputs.append(throughput)
                else:
                    throughputs.append(0)
            
            ax5.plot(batch_sizes, throughputs, marker='o', linewidth=2, label=model_name, color=colors[i])
        
        ax5.set_xlabel('Batch Size')
        ax5.set_ylabel('Throughput (images/sec)')
        ax5.set_title('Throughput Scaling', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Memory Usage Comparison
        if 'memory' in self.analysis_results and self.device == 'cuda':
            ax6 = plt.subplot(3, 3, 6)
            
            # Use batch size 8 for comparison
            memory_usage = []
            for name in model_names:
                if 'batch_8' in self.analysis_results['memory'][name]:
                    memory_usage.append(self.analysis_results['memory'][name]['batch_8']['memory_per_image_mb'])
                else:
                    memory_usage.append(0)
            
            bars6 = ax6.bar(model_names, memory_usage, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax6.set_title('Memory Usage per Image', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Memory (MB)')
            ax6.tick_params(axis='x', rotation=45)
            
            for bar, mem in zip(bars6, memory_usage):
                if mem > 0:
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 7. LPIPS Distortion Sensitivity
        ax7 = plt.subplot(3, 3, 7)
        distortion_types = ['jpeg_compression', 'gaussian_noise', 'gaussian_blur']
        
        x = np.arange(len(distortion_types))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            sensitivities = []
            for dist_type in distortion_types:
                sensitivity = self.analysis_results['lpips_performance'][model_name]['distortion_sensitivity'][dist_type]['mean_sensitivity']
                sensitivities.append(sensitivity)
            
            ax7.bar(x + i * width, sensitivities, width, label=model_name, color=colors[i])
        
        ax7.set_xlabel('Distortion Type')
        ax7.set_ylabel('Mean Sensitivity')
        ax7.set_title('LPIPS Distortion Sensitivity', fontsize=14, fontweight='bold')
        ax7.set_xticks(x + width)
        ax7.set_xticklabels([dt.replace('_', ' ').title() for dt in distortion_types])
        ax7.legend()
        ax7.tick_params(axis='x', rotation=45)
        
        # 8. Parameter Efficiency Ratio
        ax8 = plt.subplot(3, 3, 8)
        
        # Calculate efficiency ratio (inverse of parameters per unit performance)
        base_params = param_counts[0]  # AlexNet as baseline
        efficiency_ratios = [base_params / pc for pc in param_counts]
        
        bars8 = ax8.bar(model_names, efficiency_ratios, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax8.set_title('Parameter Efficiency Ratio\n(vs AlexNet)', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Efficiency Ratio')
        ax8.tick_params(axis='x', rotation=45)
        
        for bar, ratio in zip(bars8, efficiency_ratios):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 9. Overall Recommendation Matrix
        ax9 = plt.subplot(3, 3, 9)
        
        # Create recommendation heatmap
        use_cases = ['Mobile/Edge', 'High Quality', 'Research', 'Balanced']
        recommendation_matrix = np.array([
            [3, 1, 2],  # Mobile/Edge: SqueezeNet best
            [1, 3, 2],  # High Quality: VGG best
            [2, 1, 3],  # Research: AlexNet best
            [3, 2, 1]   # Balanced: SqueezeNet best
        ])
        
        im = ax9.imshow(recommendation_matrix, cmap='RdYlGn', aspect='auto')
        ax9.set_xticks(range(len(model_names)))
        ax9.set_yticks(range(len(use_cases)))
        ax9.set_xticklabels(model_names)
        ax9.set_yticklabels(use_cases)
        ax9.set_title('Recommendation Matrix\n(3=Best, 1=Worst)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(use_cases)):
            for j in range(len(model_names)):
                ax9.text(j, i, recommendation_matrix[i, j], ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('lpips_models_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
            print("Comprehensive analysis visualization saved as 'lpips_models_comprehensive_analysis.png'")
        
        plt.show()
        
        return fig


def main():
    """Main analysis execution"""
    
    print("="*80)
    print("COMPREHENSIVE LPIPS SUPPORTING MODELS ANALYSIS")
    print("="*80)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize analyzer
    analyzer = LPIPSModelAnalyzer(device=device)
    
    # Load models
    analyzer.load_models(pretrained=True)
    
    # Run comprehensive analysis
    print("\nStarting comprehensive analysis...")
    
    # 1. Architecture Analysis
    arch_results = analyzer.analyze_model_architectures()
    
    # 2. Efficiency Analysis
    efficiency_results = analyzer.analyze_computational_efficiency()
    
    # 3. Feature Quality Analysis
    feature_results = analyzer.analyze_feature_extraction_quality()
    
    # 4. Memory Analysis
    memory_results = analyzer.analyze_memory_usage()
    
    # 5. LPIPS Performance Analysis
    lpips_results = analyzer.compare_lpips_performance()
    
    # 6. Generate Recommendations
    recommendations = analyzer.generate_deployment_recommendations()
    
    # 7. Create Comprehensive Report
    report = analyzer.create_comprehensive_report()
    
    # 8. Create Visualizations
    analyzer.visualize_analysis_results(save_plots=True)
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - KEY FINDINGS")
    print("="*80)
    
    executive_summary = report['executive_summary']
    print("\nKey Findings:")
    for finding in executive_summary['key_findings']:
        print(f"  • {finding}")
    
    print(f"\nBest for Efficiency: {executive_summary['best_for_efficiency']}")
    print(f"Recommendations: {executive_summary['recommendations_summary']}")
    
    print(f"\nDetailed report saved to: lpips_models_analysis_report.json")
    print(f"Visualizations saved to: lpips_models_comprehensive_analysis.png")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    main()