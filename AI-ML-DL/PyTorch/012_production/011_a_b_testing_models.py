import torch
import torch.nn as nn
import numpy as np
import random
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

# A/B Testing Framework Components
class ModelVariant(Enum):
    """Model variant types for A/B testing"""
    CONTROL = "control"
    TREATMENT = "treatment"
    CHAMPION = "champion"
    CHALLENGER = "challenger"

@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    control_model_id: str
    treatment_model_id: str
    traffic_split: float  # Percentage to treatment (0.0 to 1.0)
    success_metrics: List[str]
    minimum_sample_size: int
    test_duration_hours: int
    confidence_level: float = 0.95

@dataclass
class PredictionResult:
    """Result of a model prediction"""
    model_id: str
    variant: ModelVariant
    prediction: torch.Tensor
    confidence: float
    inference_time_ms: float
    timestamp: float
    request_id: str

@dataclass
class ABTestMetrics:
    """Metrics collected during A/B testing"""
    variant: ModelVariant
    total_requests: int
    avg_inference_time_ms: float
    avg_confidence: float
    success_rate: float
    error_rate: float
    conversion_rate: float = 0.0

# Sample Models for A/B Testing
class ControlModel(nn.Module):
    """Control model (baseline)"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class TreatmentModel(nn.Module):
    """Treatment model (new version)"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Improved architecture with batch normalization
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# Traffic Splitter
class TrafficSplitter:
    """Split traffic between model variants"""
    
    def __init__(self, split_ratio: float = 0.5, sticky_sessions: bool = True):
        self.split_ratio = split_ratio  # Percentage to treatment
        self.sticky_sessions = sticky_sessions
        self.user_assignments = {}  # For sticky sessions
    
    def get_variant(self, user_id: str = None) -> ModelVariant:
        """Determine which model variant to use"""
        
        if self.sticky_sessions and user_id:
            # Check if user already assigned
            if user_id in self.user_assignments:
                return self.user_assignments[user_id]
            
            # Assign user to variant
            variant = ModelVariant.TREATMENT if random.random() < self.split_ratio else ModelVariant.CONTROL
            self.user_assignments[user_id] = variant
            return variant
        else:
            # Random assignment for each request
            return ModelVariant.TREATMENT if random.random() < self.split_ratio else ModelVariant.CONTROL
    
    def update_split_ratio(self, new_ratio: float):
        """Update traffic split ratio"""
        self.split_ratio = max(0.0, min(1.0, new_ratio))
    
    def get_assignment_stats(self) -> Dict[str, int]:
        """Get assignment statistics"""
        if not self.sticky_sessions:
            return {"info": "Random assignment - no persistent stats"}
        
        stats = defaultdict(int)
        for variant in self.user_assignments.values():
            stats[variant.value] += 1
        
        return dict(stats)

# A/B Testing Manager
class ABTestManager:
    """Manage A/B testing experiments"""
    
    def __init__(self):
        self.models = {}
        self.active_tests = {}
        self.test_results = {}
        self.traffic_splitter = None
        
        # Metrics collection
        self.prediction_history = deque(maxlen=10000)
        self.variant_metrics = defaultdict(lambda: {
            'requests': 0,
            'inference_times': [],
            'confidences': [],
            'errors': 0,
            'successes': 0
        })
    
    def register_model(self, model_id: str, model: nn.Module, variant: ModelVariant):
        """Register a model for A/B testing"""
        
        model.eval()  # Set to evaluation mode
        self.models[model_id] = {
            'model': model,
            'variant': variant,
            'registered_at': time.time()
        }
        
        print(f"✓ Model registered: {model_id} ({variant.value})")
    
    def start_ab_test(self, config: ABTestConfig) -> str:
        """Start an A/B test"""
        
        # Validate configuration
        if config.control_model_id not in self.models:
            raise ValueError(f"Control model {config.control_model_id} not found")
        if config.treatment_model_id not in self.models:
            raise ValueError(f"Treatment model {config.treatment_model_id} not found")
        
        # Setup traffic splitter
        self.traffic_splitter = TrafficSplitter(config.traffic_split)
        
        # Store test configuration
        test_id = f"{config.test_name}_{int(time.time())}"
        self.active_tests[test_id] = {
            'config': config,
            'start_time': time.time(),
            'status': 'RUNNING'
        }
        
        print(f"✓ A/B test started: {test_id}")
        print(f"  Control: {config.control_model_id}")
        print(f"  Treatment: {config.treatment_model_id}")
        print(f"  Traffic split: {config.traffic_split*100:.1f}% to treatment")
        
        return test_id
    
    def predict(self, input_data: torch.Tensor, 
               user_id: str = None,
               request_id: str = None) -> PredictionResult:
        """Make prediction using A/B testing"""
        
        if not self.traffic_splitter:
            raise ValueError("No active A/B test. Start a test first.")
        
        # Get variant assignment
        variant = self.traffic_splitter.get_variant(user_id)
        
        # Find model for variant
        model_id = None
        for mid, model_info in self.models.items():
            if model_info['variant'] == variant:
                model_id = mid
                break
        
        if model_id is None:
            raise ValueError(f"No model found for variant {variant}")
        
        # Make prediction
        model = self.models[model_id]['model']
        
        start_time = time.time()
        try:
            with torch.no_grad():
                output = model(input_data)
                probabilities = torch.softmax(output, dim=1)
                confidence = torch.max(probabilities).item()
            
            inference_time = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                model_id=model_id,
                variant=variant,
                prediction=output,
                confidence=confidence,
                inference_time_ms=inference_time,
                timestamp=time.time(),
                request_id=request_id or f"req_{int(time.time()*1000)}"
            )
            
            # Record metrics
            self._record_prediction_metrics(result, success=True)
            
            return result
            
        except Exception as e:
            # Record error
            self._record_prediction_metrics(
                PredictionResult(
                    model_id=model_id,
                    variant=variant,
                    prediction=None,
                    confidence=0.0,
                    inference_time_ms=(time.time() - start_time) * 1000,
                    timestamp=time.time(),
                    request_id=request_id or f"req_{int(time.time()*1000)}"
                ),
                success=False
            )
            raise e
    
    def _record_prediction_metrics(self, result: PredictionResult, success: bool):
        """Record prediction metrics for analysis"""
        
        # Store in history
        self.prediction_history.append(result)
        
        # Update variant metrics
        variant_key = result.variant.value
        metrics = self.variant_metrics[variant_key]
        
        metrics['requests'] += 1
        metrics['inference_times'].append(result.inference_time_ms)
        
        if success:
            metrics['successes'] += 1
            if result.prediction is not None:
                metrics['confidences'].append(result.confidence)
        else:
            metrics['errors'] += 1
    
    def get_test_results(self, test_id: str = None) -> Dict[str, Any]:
        """Get current A/B test results"""
        
        if test_id and test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        # Calculate metrics for each variant
        results = {}
        
        for variant_key, metrics in self.variant_metrics.items():
            if metrics['requests'] == 0:
                continue
                
            avg_inference_time = np.mean(metrics['inference_times']) if metrics['inference_times'] else 0
            avg_confidence = np.mean(metrics['confidences']) if metrics['confidences'] else 0
            error_rate = metrics['errors'] / metrics['requests']
            success_rate = metrics['successes'] / metrics['requests']
            
            results[variant_key] = ABTestMetrics(
                variant=ModelVariant(variant_key),
                total_requests=metrics['requests'],
                avg_inference_time_ms=avg_inference_time,
                avg_confidence=avg_confidence,
                success_rate=success_rate,
                error_rate=error_rate
            )
        
        return results
    
    def analyze_test_significance(self, test_id: str = None) -> Dict[str, Any]:
        """Analyze statistical significance of A/B test"""
        
        results = self.get_test_results(test_id)
        
        if len(results) < 2:
            return {"error": "Need at least 2 variants to analyze"}
        
        # Get control and treatment metrics
        control_metrics = results.get('control') or results.get('champion')
        treatment_metrics = results.get('treatment') or results.get('challenger')
        
        if not control_metrics or not treatment_metrics:
            return {"error": "Missing control or treatment metrics"}
        
        # Statistical analysis (simplified)
        analysis = {
            'control_requests': control_metrics.total_requests,
            'treatment_requests': treatment_metrics.total_requests,
            'control_success_rate': control_metrics.success_rate,
            'treatment_success_rate': treatment_metrics.success_rate,
            'relative_improvement': (treatment_metrics.success_rate - control_metrics.success_rate) / control_metrics.success_rate if control_metrics.success_rate > 0 else 0,
            'inference_time_improvement': (control_metrics.avg_inference_time_ms - treatment_metrics.avg_inference_time_ms) / control_metrics.avg_inference_time_ms if control_metrics.avg_inference_time_ms > 0 else 0,
            'confidence_improvement': treatment_metrics.avg_confidence - control_metrics.avg_confidence
        }
        
        # Simple significance test (Chi-square test for success rates)
        total_control = control_metrics.total_requests
        total_treatment = treatment_metrics.total_requests
        success_control = int(control_metrics.success_rate * total_control)
        success_treatment = int(treatment_metrics.success_rate * total_treatment)
        
        # Chi-square statistic (simplified)
        expected_control = (success_control + success_treatment) * total_control / (total_control + total_treatment)
        expected_treatment = (success_control + success_treatment) * total_treatment / (total_control + total_treatment)
        
        if expected_control > 0 and expected_treatment > 0:
            chi_square = ((success_control - expected_control) ** 2 / expected_control +
                         (success_treatment - expected_treatment) ** 2 / expected_treatment)
            
            # Rough significance threshold (chi-square > 3.84 for p < 0.05)
            analysis['chi_square_statistic'] = chi_square
            analysis['is_significant'] = chi_square > 3.84
            analysis['p_value_estimate'] = "< 0.05" if chi_square > 3.84 else "> 0.05"
        else:
            analysis['is_significant'] = False
            analysis['p_value_estimate'] = "Cannot calculate"
        
        return analysis
    
    def stop_test(self, test_id: str, winner: ModelVariant = None) -> Dict[str, Any]:
        """Stop A/B test and declare winner"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        # Get final results
        final_results = self.get_test_results(test_id)
        significance_analysis = self.analyze_test_significance(test_id)
        
        # Update test status
        self.active_tests[test_id]['status'] = 'COMPLETED'
        self.active_tests[test_id]['end_time'] = time.time()
        self.active_tests[test_id]['winner'] = winner.value if winner else 'NO_WINNER'
        
        # Store results
        self.test_results[test_id] = {
            'final_results': final_results,
            'significance_analysis': significance_analysis,
            'winner': winner.value if winner else 'NO_WINNER',
            'completed_at': time.time()
        }
        
        print(f"✓ A/B test stopped: {test_id}")
        if winner:
            print(f"  Winner: {winner.value}")
        
        return self.test_results[test_id]

# Champion/Challenger Framework
class ChampionChallengerFramework:
    """Champion/Challenger model deployment framework"""
    
    def __init__(self):
        self.champion_model = None
        self.challenger_models = {}
        self.performance_monitor = None
        self.auto_promotion_enabled = False
        self.promotion_criteria = {
            'min_requests': 1000,
            'improvement_threshold': 0.05,  # 5% improvement
            'confidence_level': 0.95
        }
    
    def set_champion(self, model: nn.Module, model_id: str):
        """Set the current champion model"""
        
        self.champion_model = {
            'model': model,
            'model_id': model_id,
            'promoted_at': time.time()
        }
        
        print(f"✓ Champion model set: {model_id}")
    
    def add_challenger(self, model: nn.Module, model_id: str, traffic_percentage: float = 10.0):
        """Add a challenger model"""
        
        self.challenger_models[model_id] = {
            'model': model,
            'model_id': model_id,
            'traffic_percentage': traffic_percentage,
            'added_at': time.time()
        }
        
        print(f"✓ Challenger model added: {model_id} ({traffic_percentage}% traffic)")
    
    def route_prediction(self, input_data: torch.Tensor, user_id: str = None) -> Tuple[torch.Tensor, str]:
        """Route prediction request to champion or challenger"""
        
        if not self.champion_model:
            raise ValueError("No champion model set")
        
        # Determine routing
        total_challenger_traffic = sum(c['traffic_percentage'] for c in self.challenger_models.values())
        
        if random.random() * 100 < total_challenger_traffic:
            # Route to challenger
            challenger_weights = [c['traffic_percentage'] for c in self.challenger_models.values()]
            challenger_ids = list(self.challenger_models.keys())
            
            # Weighted random selection
            chosen_idx = np.random.choice(len(challenger_ids), p=np.array(challenger_weights)/sum(challenger_weights))
            chosen_challenger_id = challenger_ids[chosen_idx]
            
            model = self.challenger_models[chosen_challenger_id]['model']
            
            with torch.no_grad():
                output = model(input_data)
            
            return output, chosen_challenger_id
        else:
            # Route to champion
            with torch.no_grad():
                output = self.champion_model['model'](input_data)
            
            return output, self.champion_model['model_id']
    
    def check_auto_promotion(self, ab_manager: ABTestManager) -> Optional[str]:
        """Check if any challenger should be auto-promoted"""
        
        if not self.auto_promotion_enabled:
            return None
        
        results = ab_manager.get_test_results()
        significance = ab_manager.analyze_test_significance()
        
        # Check promotion criteria
        if (significance.get('treatment_requests', 0) >= self.promotion_criteria['min_requests'] and
            significance.get('is_significant', False) and
            significance.get('relative_improvement', 0) >= self.promotion_criteria['improvement_threshold']):
            
            # Promote challenger to champion
            treatment_model_id = None
            for model_id, model_info in ab_manager.models.items():
                if model_info['variant'] == ModelVariant.TREATMENT:
                    treatment_model_id = model_id
                    break
            
            if treatment_model_id:
                # Promote
                new_champion = ab_manager.models[treatment_model_id]['model']
                old_champion_id = self.champion_model['model_id'] if self.champion_model else None
                
                self.set_champion(new_champion, treatment_model_id)
                
                print(f"🏆 Auto-promotion: {treatment_model_id} promoted to champion")
                if old_champion_id:
                    print(f"   Previous champion: {old_champion_id}")
                
                return treatment_model_id
        
        return None

# Model Performance Comparison
class ModelComparator:
    """Compare model performance across different metrics"""
    
    @staticmethod
    def compare_models(results: Dict[str, ABTestMetrics]) -> Dict[str, Any]:
        """Compare models across multiple dimensions"""
        
        if len(results) < 2:
            return {"error": "Need at least 2 models to compare"}
        
        metrics_comparison = {}
        
        # Compare each metric
        for metric in ['avg_inference_time_ms', 'avg_confidence', 'success_rate', 'error_rate']:
            values = {variant: getattr(metrics, metric) for variant, metrics in results.items()}
            
            best_variant = min(values.keys(), key=lambda k: values[k]) if metric in ['avg_inference_time_ms', 'error_rate'] else max(values.keys(), key=lambda k: values[k])
            worst_variant = max(values.keys(), key=lambda k: values[k]) if metric in ['avg_inference_time_ms', 'error_rate'] else min(values.keys(), key=lambda k: values[k])
            
            metrics_comparison[metric] = {
                'best': {'variant': best_variant, 'value': values[best_variant]},
                'worst': {'variant': worst_variant, 'value': values[worst_variant]},
                'all_values': values
            }
        
        return metrics_comparison

if __name__ == "__main__":
    print("A/B Testing for Model Deployment")
    print("=" * 37)
    
    # Create sample models
    control_model = ControlModel(num_classes=10)
    treatment_model = TreatmentModel(num_classes=10)
    
    print("\n1. A/B Test Setup")
    print("-" * 20)
    
    # Initialize A/B test manager
    ab_manager = ABTestManager()
    
    # Register models
    ab_manager.register_model("control_v1", control_model, ModelVariant.CONTROL)
    ab_manager.register_model("treatment_v2", treatment_model, ModelVariant.TREATMENT)
    
    # Configure A/B test
    test_config = ABTestConfig(
        test_name="cnn_architecture_comparison",
        control_model_id="control_v1",
        treatment_model_id="treatment_v2",
        traffic_split=0.3,  # 30% to treatment
        success_metrics=["accuracy", "inference_time"],
        minimum_sample_size=100,
        test_duration_hours=24,
        confidence_level=0.95
    )
    
    # Start A/B test
    test_id = ab_manager.start_ab_test(test_config)
    
    print("\n2. Simulating Production Traffic")
    print("-" * 38)
    
    # Simulate requests
    print("Processing requests...")
    
    for i in range(200):
        try:
            # Create random input
            input_data = torch.randn(1, 3, 32, 32)
            user_id = f"user_{i % 50}"  # 50 different users
            
            # Make prediction
            result = ab_manager.predict(input_data, user_id=user_id)
            
            if i % 50 == 0:
                print(f"Request {i+1}: {result.variant.value} model, "
                      f"{result.inference_time_ms:.2f}ms, "
                      f"confidence: {result.confidence:.3f}")
            
        except Exception as e:
            print(f"Error in request {i+1}: {e}")
    
    print("\n3. A/B Test Results")
    print("-" * 22)
    
    # Get test results
    test_results = ab_manager.get_test_results(test_id)
    
    print("Performance Comparison:")
    print("-" * 25)
    print(f"{'Variant':<12} {'Requests':<10} {'Avg Time':<12} {'Success Rate':<12} {'Avg Confidence':<15}")
    print("-" * 70)
    
    for variant_name, metrics in test_results.items():
        print(f"{variant_name:<12} {metrics.total_requests:<10} "
              f"{metrics.avg_inference_time_ms:<12.2f} "
              f"{metrics.success_rate:<12.2%} "
              f"{metrics.avg_confidence:<15.3f}")
    
    print("\n4. Statistical Analysis")
    print("-" * 26)
    
    # Analyze significance
    significance = ab_manager.analyze_test_significance(test_id)
    
    if "error" not in significance:
        print("Statistical Significance Analysis:")
        print(f"  Control success rate: {significance['control_success_rate']:.2%}")
        print(f"  Treatment success rate: {significance['treatment_success_rate']:.2%}")
        print(f"  Relative improvement: {significance['relative_improvement']:.2%}")
        print(f"  Inference time improvement: {significance['inference_time_improvement']:.2%}")
        print(f"  Confidence improvement: {significance['confidence_improvement']:.3f}")
        print(f"  Statistical significance: {significance['is_significant']}")
        print(f"  P-value estimate: {significance['p_value_estimate']}")
    else:
        print(f"Analysis error: {significance['error']}")
    
    print("\n5. Champion/Challenger Framework")
    print("-" * 37)
    
    # Setup champion/challenger
    cc_framework = ChampionChallengerFramework()
    
    # Set champion
    cc_framework.set_champion(control_model, "control_v1")
    
    # Add challenger
    cc_framework.add_challenger(treatment_model, "treatment_v2", traffic_percentage=20.0)
    
    # Enable auto-promotion
    cc_framework.auto_promotion_enabled = True
    
    # Simulate routing
    print("\nSimulating champion/challenger routing:")
    
    routing_stats = defaultdict(int)
    
    for i in range(100):
        input_data = torch.randn(1, 3, 32, 32)
        output, chosen_model = cc_framework.route_prediction(input_data)
        routing_stats[chosen_model] += 1
    
    print("Routing statistics:")
    for model_id, count in routing_stats.items():
        print(f"  {model_id}: {count} requests ({count}%)")
    
    # Check for auto-promotion
    promoted_model = cc_framework.check_auto_promotion(ab_manager)
    if promoted_model:
        print(f"Model {promoted_model} was auto-promoted!")
    else:
        print("No auto-promotion triggered")
    
    print("\n6. Model Performance Comparison")
    print("-" * 37)
    
    # Compare models
    comparison = ModelComparator.compare_models(test_results)
    
    if "error" not in comparison:
        print("Performance Comparison by Metric:")
        for metric, data in comparison.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Best: {data['best']['variant']} ({data['best']['value']:.3f})")
            print(f"  Worst: {data['worst']['variant']} ({data['worst']['value']:.3f})")
    
    print("\n7. Test Conclusion")
    print("-" * 20)
    
    # Stop test and declare winner
    if significance.get('is_significant', False):
        if significance['relative_improvement'] > 0:
            winner = ModelVariant.TREATMENT
        else:
            winner = ModelVariant.CONTROL
    else:
        winner = None
    
    final_results = ab_manager.stop_test(test_id, winner)
    
    if winner:
        print(f"Test concluded with winner: {winner.value}")
    else:
        print("Test concluded with no significant winner")
    
    print("\n8. A/B Testing Best Practices")
    print("-" * 35)
    
    best_practices = [
        "Define clear success metrics before starting tests",
        "Ensure sufficient sample sizes for statistical power",
        "Use proper randomization to avoid bias",
        "Monitor both primary and secondary metrics",
        "Set appropriate test duration based on traffic",
        "Implement guardrails to prevent negative impact",
        "Use sticky sessions for consistent user experience",
        "Document test hypotheses and learnings",
        "Consider business impact beyond technical metrics",
        "Implement gradual rollout strategies",
        "Monitor for negative externalities",
        "Plan for rollback procedures"
    ]
    
    print("A/B Testing Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n9. Common A/B Testing Pitfalls")
    print("-" * 35)
    
    pitfalls = [
        "Stopping tests too early (insufficient sample size)",
        "Not accounting for multiple testing corrections",
        "Ignoring business context in favor of statistical significance",
        "Testing multiple variants without proper power analysis",
        "Not monitoring for implementation bugs",
        "Confusing correlation with causation",
        "Not considering seasonal or temporal effects",
        "Inadequate randomization leading to biased results"
    ]
    
    print("Common Pitfalls to Avoid:")
    for i, pitfall in enumerate(pitfalls, 1):
        print(f"{i}. {pitfall}")
    
    print("\n10. Deployment Strategies")
    print("-" * 28)
    
    deployment_strategies = {
        "Canary Deployment": "Gradual rollout starting with small percentage",
        "Blue-Green Deployment": "Complete environment switch after validation",
        "Feature Flags": "Toggle between models without deployment",
        "Shadow Mode": "Run new model alongside production without serving traffic",
        "Multi-Armed Bandit": "Dynamic traffic allocation based on performance"
    }
    
    print("Model Deployment Strategies:")
    for strategy, description in deployment_strategies.items():
        print(f"  {strategy}: {description}")
    
    print("\nA/B testing demonstration completed!")
    print("Key learnings:")
    print(f"  - Total requests processed: {sum(metrics.total_requests for metrics in test_results.values())}")
    print(f"  - Statistical significance: {significance.get('is_significant', False)}")
    print(f"  - Champion/challenger routing tested")
    print(f"  - Winner: {winner.value if winner else 'No winner'}")
    
    traffic_stats = ab_manager.traffic_splitter.get_assignment_stats()
    if isinstance(traffic_stats, dict) and "info" not in traffic_stats:
        print(f"  - Traffic split achieved: {traffic_stats}")
    
    print("\nRecommendations for production:")
    print("  - Implement proper logging and monitoring")
    print("  - Set up automated alerts for test anomalies")
    print("  - Use feature flags for quick rollback capability")
    print("  - Consider business metrics alongside technical metrics")