# Evaluation Metrics Guide: Measuring Agent Performance

## Metrics Categories Overview

| Category | Metrics | Purpose | Measurement Complexity |
|----------|---------|---------|----------------------|
| **Performance** | Accuracy, F1, Precision, Recall | Task effectiveness | Low-Medium |
| **Efficiency** | Response time, Throughput, Resource usage | System performance | Low |
| **Reliability** | Uptime, Error rate, Consistency | System stability | Medium |
| **User Experience** | Satisfaction, Usability, Engagement | User perception | High |
| **Safety** | Bias detection, Harm prevention, Compliance | Risk assessment | High |
| **Explainability** | Transparency, Interpretability, Trust | Understanding | High |

---

## Performance Metrics

### **1. Accuracy-Based Metrics**
```python
class AccuracyMetrics:
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
    
    def accuracy(self):
        """Overall accuracy percentage"""
        if not self.predictions:
            return 0.0
        
        correct = sum(1 for p, t in zip(self.predictions, self.ground_truth) 
                     if p == t)
        return correct / len(self.predictions)
    
    def precision(self, label=None):
        """Precision for specific label or macro average"""
        if label is not None:
            tp = sum(1 for p, t in zip(self.predictions, self.ground_truth) 
                    if p == label and t == label)
            fp = sum(1 for p, t in zip(self.predictions, self.ground_truth) 
                    if p == label and t != label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # Macro average precision
            labels = set(self.ground_truth)
            return sum(self.precision(l) for l in labels) / len(labels)
    
    def recall(self, label=None):
        """Recall for specific label or macro average"""
        if label is not None:
            tp = sum(1 for p, t in zip(self.predictions, self.ground_truth) 
                    if p == label and t == label)
            fn = sum(1 for p, t in zip(self.predictions, self.ground_truth) 
                    if p != label and t == label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            # Macro average recall
            labels = set(self.ground_truth)
            return sum(self.recall(l) for l in labels) / len(labels)
    
    def f1_score(self, label=None):
        """F1 score for specific label or macro average"""
        p = self.precision(label)
        r = self.recall(label)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

# Usage Example
class CustomerServiceEvaluator:
    def __init__(self):
        self.accuracy_metrics = AccuracyMetrics()
        self.response_quality_scores = []
        self.customer_satisfaction_scores = []
    
    def evaluate_interaction(self, agent_response, expected_response, customer_feedback):
        """Evaluate single customer interaction"""
        
        # Intent classification accuracy
        predicted_intent = self.extract_intent(agent_response)
        true_intent = self.extract_intent(expected_response)
        self.accuracy_metrics.predictions.append(predicted_intent)
        self.accuracy_metrics.ground_truth.append(true_intent)
        
        # Response quality (semantic similarity)
        quality_score = self.calculate_semantic_similarity(
            agent_response, expected_response
        )
        self.response_quality_scores.append(quality_score)
        
        # Customer satisfaction
        satisfaction = customer_feedback.get('satisfaction_rating', 0)
        self.customer_satisfaction_scores.append(satisfaction)
        
        return {
            'intent_accuracy': predicted_intent == true_intent,
            'response_quality': quality_score,
            'customer_satisfaction': satisfaction
        }
```

### **2. Task-Specific Metrics**
```python
class TaskSpecificMetrics:
    def __init__(self, task_type):
        self.task_type = task_type
        self.task_completion_times = []
        self.success_rates = {}
        self.error_patterns = {}
    
    def evaluate_rag_system(self, questions, generated_answers, reference_answers, contexts):
        """Evaluate RAG system performance"""
        metrics = {}
        
        # Answer relevancy
        relevancy_scores = [
            self.calculate_answer_relevancy(q, a) 
            for q, a in zip(questions, generated_answers)
        ]
        metrics['answer_relevancy'] = sum(relevancy_scores) / len(relevancy_scores)
        
        # Faithfulness (answer supported by context)
        faithfulness_scores = [
            self.calculate_faithfulness(a, c) 
            for a, c in zip(generated_answers, contexts)
        ]
        metrics['faithfulness'] = sum(faithfulness_scores) / len(faithfulness_scores)
        
        # Context relevancy
        context_relevancy_scores = [
            self.calculate_context_relevancy(q, c) 
            for q, c in zip(questions, contexts)
        ]
        metrics['context_relevancy'] = sum(context_relevancy_scores) / len(context_relevancy_scores)
        
        # BLEU score for reference comparison
        bleu_scores = [
            self.calculate_bleu_score(gen, ref) 
            for gen, ref in zip(generated_answers, reference_answers)
        ]
        metrics['bleu_score'] = sum(bleu_scores) / len(bleu_scores)
        
        return metrics
    
    def evaluate_trading_agent(self, trades, market_data, portfolio_performance):
        """Evaluate trading agent performance"""
        metrics = {}
        
        # Return metrics
        total_return = portfolio_performance['final_value'] / portfolio_performance['initial_value'] - 1
        metrics['total_return'] = total_return
        
        # Risk metrics
        returns = portfolio_performance['daily_returns']
        metrics['volatility'] = self.calculate_volatility(returns)
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns)
        metrics['max_drawdown'] = self.calculate_max_drawdown(portfolio_performance['values'])
        
        # Trade efficiency
        metrics['win_rate'] = sum(1 for trade in trades if trade['profit'] > 0) / len(trades)
        metrics['average_trade_duration'] = sum(trade['duration'] for trade in trades) / len(trades)
        
        return metrics
```

---

## Efficiency Metrics

### **3. Performance and Resource Metrics**
```python
import time
import psutil
import threading

class EfficiencyMetrics:
    def __init__(self):
        self.response_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.throughput_counter = 0
        self.start_time = time.time()
    
    def measure_response_time(self, func):
        """Decorator to measure function response time"""
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            response_time = end - start
            self.response_times.append(response_time)
            return result
        return wrapper
    
    def start_resource_monitoring(self):
        """Start monitoring system resources"""
        def monitor():
            while self.monitoring:
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # CPU usage
                cpu = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu)
                
                time.sleep(5)  # Monitor every 5 seconds
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def calculate_throughput(self):
        """Calculate requests per second"""
        elapsed_time = time.time() - self.start_time
        return self.throughput_counter / elapsed_time if elapsed_time > 0 else 0
    
    def get_efficiency_report(self):
        """Generate comprehensive efficiency report"""
        return {
            'response_time': {
                'mean': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'median': sorted(self.response_times)[len(self.response_times)//2] if self.response_times else 0,
                'p95': sorted(self.response_times)[int(len(self.response_times)*0.95)] if self.response_times else 0,
                'p99': sorted(self.response_times)[int(len(self.response_times)*0.99)] if self.response_times else 0
            },
            'resource_usage': {
                'avg_memory_percent': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                'avg_cpu_percent': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
                'peak_memory': max(self.memory_usage) if self.memory_usage else 0,
                'peak_cpu': max(self.cpu_usage) if self.cpu_usage else 0
            },
            'throughput': self.calculate_throughput()
        }

class AgentPerformanceMonitor:
    def __init__(self, agent):
        self.agent = agent
        self.efficiency_metrics = EfficiencyMetrics()
        self.agent.process = self.efficiency_metrics.measure_response_time(agent.process)
    
    def start_monitoring(self):
        """Start comprehensive performance monitoring"""
        self.efficiency_metrics.start_resource_monitoring()
    
    def process_request(self, request):
        """Process request with monitoring"""
        self.efficiency_metrics.throughput_counter += 1
        return self.agent.process(request)
```

---

## Reliability Metrics

### **4. System Reliability and Consistency**
```python
class ReliabilityMetrics:
    def __init__(self):
        self.uptime_start = time.time()
        self.downtime_periods = []
        self.error_counts = {}
        self.consistency_scores = []
        self.request_outcomes = []
    
    def track_uptime(self):
        """Calculate system uptime percentage"""
        current_time = time.time()
        total_time = current_time - self.uptime_start
        total_downtime = sum(period['duration'] for period in self.downtime_periods)
        uptime_percentage = ((total_time - total_downtime) / total_time) * 100
        return uptime_percentage
    
    def record_downtime(self, start_time, end_time, reason):
        """Record system downtime period"""
        self.downtime_periods.append({
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'reason': reason
        })
    
    def track_error(self, error_type, error_message):
        """Track error occurrences"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = []
        
        self.error_counts[error_type].append({
            'message': error_message,
            'timestamp': time.time()
        })
    
    def calculate_error_rate(self, time_window_hours=24):
        """Calculate error rate within time window"""
        current_time = time.time()
        window_start = current_time - (time_window_hours * 3600)
        
        total_errors = 0
        for error_type, errors in self.error_counts.items():
            recent_errors = [e for e in errors if e['timestamp'] >= window_start]
            total_errors += len(recent_errors)
        
        total_requests = len([r for r in self.request_outcomes 
                             if r['timestamp'] >= window_start])
        
        return (total_errors / total_requests) * 100 if total_requests > 0 else 0
    
    def measure_consistency(self, test_inputs, num_runs=5):
        """Measure output consistency for same inputs"""
        consistency_results = []
        
        for test_input in test_inputs:
            outputs = []
            for _ in range(num_runs):
                output = self.agent.process(test_input)
                outputs.append(output)
            
            # Calculate consistency score (similarity between outputs)
            consistency_score = self.calculate_output_consistency(outputs)
            consistency_results.append(consistency_score)
        
        return sum(consistency_results) / len(consistency_results)
```

---

## User Experience Metrics

### **5. User Satisfaction and Engagement**
```python
class UserExperienceMetrics:
    def __init__(self):
        self.satisfaction_scores = []
        self.engagement_metrics = {}
        self.usability_scores = []
        self.trust_indicators = []
    
    def collect_satisfaction_feedback(self, interaction_id, rating, feedback_text):
        """Collect user satisfaction feedback"""
        self.satisfaction_scores.append({
            'interaction_id': interaction_id,
            'rating': rating,  # 1-5 scale
            'feedback': feedback_text,
            'timestamp': time.time()
        })
    
    def track_engagement(self, user_id, session_data):
        """Track user engagement metrics"""
        if user_id not in self.engagement_metrics:
            self.engagement_metrics[user_id] = {
                'total_interactions': 0,
                'session_lengths': [],
                'return_visits': 0,
                'feature_usage': {}
            }
        
        user_metrics = self.engagement_metrics[user_id]
        user_metrics['total_interactions'] += session_data['interaction_count']
        user_metrics['session_lengths'].append(session_data['session_length'])
        
        for feature, usage_count in session_data['features_used'].items():
            if feature not in user_metrics['feature_usage']:
                user_metrics['feature_usage'][feature] = 0
            user_metrics['feature_usage'][feature] += usage_count
    
    def measure_usability(self, task_completion_data):
        """Measure usability through task completion"""
        usability_score = {
            'task_completion_rate': 
                sum(1 for task in task_completion_data if task['completed']) / len(task_completion_data),
            'average_completion_time': 
                sum(task['time'] for task in task_completion_data if task['completed']) / 
                len([task for task in task_completion_data if task['completed']]),
            'error_rate': 
                sum(task['errors'] for task in task_completion_data) / len(task_completion_data),
            'help_requests': 
                sum(1 for task in task_completion_data if task['help_requested']) / len(task_completion_data)
        }
        
        self.usability_scores.append(usability_score)
        return usability_score
    
    def assess_trust_indicators(self, user_interactions):
        """Assess indicators of user trust in the system"""
        trust_metrics = {
            'explanation_requests': 
                sum(1 for interaction in user_interactions if interaction.get('asked_for_explanation')),
            'override_frequency': 
                sum(1 for interaction in user_interactions if interaction.get('user_override')),
            'delegation_willingness': 
                sum(interaction.get('delegation_score', 0) for interaction in user_interactions) / len(user_interactions),
            'repeated_usage': 
                len(set(interaction['user_id'] for interaction in user_interactions if interaction.get('return_user')))
        }
        
        self.trust_indicators.append(trust_metrics)
        return trust_metrics
```

---

## Safety and Bias Metrics

### **6. Safety and Fairness Evaluation**
```python
class SafetyMetrics:
    def __init__(self):
        self.bias_detectors = {
            'gender': GenderBiasDetector(),
            'race': RaceBiasDetector(),
            'age': AgeBiasDetector()
        }
        self.safety_violations = []
        self.fairness_scores = {}
    
    def detect_bias(self, decisions, protected_attributes):
        """Detect bias in agent decisions"""
        bias_results = {}
        
        for attribute_type, detector in self.bias_detectors.items():
            if attribute_type in protected_attributes:
                bias_score = detector.detect_bias(
                    decisions, protected_attributes[attribute_type]
                )
                bias_results[attribute_type] = bias_score
        
        return bias_results
    
    def measure_fairness(self, decisions, groups):
        """Measure fairness across different groups"""
        fairness_metrics = {}
        
        # Demographic parity
        fairness_metrics['demographic_parity'] = self.calculate_demographic_parity(decisions, groups)
        
        # Equalized odds
        fairness_metrics['equalized_odds'] = self.calculate_equalized_odds(decisions, groups)
        
        # Individual fairness
        fairness_metrics['individual_fairness'] = self.calculate_individual_fairness(decisions, groups)
        
        return fairness_metrics
    
    def check_safety_violations(self, agent_outputs, safety_guidelines):
        """Check for safety guideline violations"""
        violations = []
        
        for output in agent_outputs:
            for guideline in safety_guidelines:
                if guideline.is_violated(output):
                    violations.append({
                        'output': output,
                        'violated_guideline': guideline.name,
                        'severity': guideline.severity,
                        'timestamp': time.time()
                    })
        
        self.safety_violations.extend(violations)
        return violations
    
    def calculate_harm_potential(self, agent_actions, context):
        """Assess potential harm from agent actions"""
        harm_score = 0
        
        for action in agent_actions:
            # Assess immediate harm potential
            immediate_harm = self.assess_immediate_harm(action, context)
            
            # Assess long-term consequences
            longterm_harm = self.assess_longterm_harm(action, context)
            
            # Assess scope of impact
            impact_scope = self.assess_impact_scope(action, context)
            
            action_harm_score = (immediate_harm + longterm_harm) * impact_scope
            harm_score += action_harm_score
        
        return harm_score / len(agent_actions) if agent_actions else 0
```

---

## Composite Evaluation Framework

### **7. Comprehensive Agent Evaluation**
```python
class ComprehensiveAgentEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self.accuracy_metrics = AccuracyMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.reliability_metrics = ReliabilityMetrics()
        self.ux_metrics = UserExperienceMetrics()
        self.safety_metrics = SafetyMetrics()
        
        self.evaluation_weights = {
            'accuracy': 0.25,
            'efficiency': 0.20,
            'reliability': 0.20,
            'user_experience': 0.20,
            'safety': 0.15
        }
    
    def run_comprehensive_evaluation(self, test_suite):
        """Run complete evaluation across all metrics"""
        results = {}
        
        # Performance evaluation
        results['accuracy'] = self.evaluate_accuracy(test_suite['accuracy_tests'])
        
        # Efficiency evaluation
        results['efficiency'] = self.evaluate_efficiency(test_suite['performance_tests'])
        
        # Reliability evaluation
        results['reliability'] = self.evaluate_reliability(test_suite['reliability_tests'])
        
        # User experience evaluation
        results['user_experience'] = self.evaluate_user_experience(test_suite['ux_tests'])
        
        # Safety evaluation
        results['safety'] = self.evaluate_safety(test_suite['safety_tests'])
        
        # Calculate composite score
        composite_score = sum(
            results[metric] * self.evaluation_weights[metric]
            for metric in self.evaluation_weights
        )
        
        results['composite_score'] = composite_score
        
        return self.generate_evaluation_report(results)
    
    def generate_evaluation_report(self, results):
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {
                'overall_score': results['composite_score'],
                'strengths': self.identify_strengths(results),
                'weaknesses': self.identify_weaknesses(results),
                'recommendations': self.generate_recommendations(results)
            },
            'detailed_metrics': results,
            'benchmarking': self.compare_to_benchmarks(results),
            'improvement_plan': self.create_improvement_plan(results)
        }
        
        return report
    
    def continuous_monitoring_setup(self):
        """Setup continuous monitoring of agent performance"""
        monitor_config = {
            'metrics_to_track': [
                'response_time', 'accuracy', 'error_rate', 
                'user_satisfaction', 'safety_violations'
            ],
            'alert_thresholds': {
                'response_time': 5.0,  # seconds
                'error_rate': 5.0,     # percentage
                'safety_violations': 1  # count per hour
            },
            'reporting_frequency': 'daily',
            'dashboard_updates': 'real_time'
        }
        
        return monitor_config
```

---

## Evaluation Best Practices

### **Metric Selection Guidelines**

1. **Task-Specific Metrics**: Choose metrics relevant to your specific use case
2. **Multi-Dimensional Evaluation**: Don't rely on a single metric
3. **Stakeholder Alignment**: Include metrics that matter to end users
4. **Baseline Comparisons**: Always compare against baseline performance
5. **Continuous Monitoring**: Set up ongoing evaluation, not just one-time testing

### **Common Pitfalls to Avoid**

- **Metric Gaming**: Optimizing for metrics instead of real performance
- **Evaluation Bias**: Using biased test sets or evaluation methods
- **Static Evaluation**: Not updating evaluation as the system evolves
- **Missing Edge Cases**: Failing to test unusual or adversarial inputs
- **Ignoring Context**: Not considering deployment environment differences

This guide provides a comprehensive framework for evaluating AI agents across multiple dimensions, ensuring robust and reliable performance assessment.
