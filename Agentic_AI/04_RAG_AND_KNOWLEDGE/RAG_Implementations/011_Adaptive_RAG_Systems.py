#!/usr/bin/env python3
"""
Adaptive RAG Systems: Self-Improving and Dynamic Retrieval
=========================================================

WHAT IS THE PROBLEM?
==================
Traditional RAG systems are static and don't learn:
- Fixed retrieval strategies regardless of performance
- Cannot adapt to changing user needs and patterns
- No learning from successful and failed retrievals
- Cannot optimize for different types of queries automatically
- Miss opportunities to improve through experience
- Cannot handle evolving document collections and domains

Example: Customer Support Evolution
STATIC RAG (Traditional):
- Same search strategy for all customer questions
- No learning from support agent feedback
- Cannot adapt to new product features or issues
- Missed optimization opportunities from interaction patterns
- Result: Suboptimal support and frustrated customers

REAL WORLD EXAMPLE:
=================
How does Google Search evolve?

GOOGLE'S ADAPTIVE SEARCH:
1. QUERY ANALYSIS: Learn from billions of search patterns
2. CLICK-THROUGH LEARNING: Adapt based on what users actually click
3. RANKING OPTIMIZATION: Continuously improve result ranking algorithms
4. PERSONALIZATION: Adapt to individual user preferences and history
5. SEASONAL ADAPTATION: Adjust for temporal patterns and trends
6. FEEDBACK INTEGRATION: Learn from user behavior and satisfaction signals
7. A/B TESTING: Continuously experiment with algorithm improvements

BENEFITS OF ADAPTIVE RAG:
- Continuous improvement from user interactions
- Automatic optimization for different query types
- Personalized and context-aware retrieval
- Self-healing systems that recover from errors
- Proactive adaptation to changing information needs
- Data-driven performance enhancement

THE ADAPTIVE ADVANTAGE:
=====================
STATIC RAG: Fixed strategy → Same performance over time
ADAPTIVE RAG: Learn from experience → Continuously improving performance

ADAPTIVE COMPONENTS:
==================
1. FEEDBACK COLLECTION: Gather implicit and explicit user feedback
2. PERFORMANCE MONITORING: Track retrieval quality and user satisfaction
3. PATTERN LEARNING: Identify successful retrieval strategies and patterns
4. STRATEGY OPTIMIZATION: Automatically tune retrieval parameters
5. PERSONALIZATION: Adapt to individual user preferences and context
6. ONLINE LEARNING: Continuously update models with new data
7. A/B TESTING: Experiment with different approaches and measure results

WHY THIS IS REVOLUTIONARY:
========================
- Enables RAG systems that get better over time
- Provides personalized and optimized user experiences
- Reduces manual tuning and maintenance overhead
- Powers self-improving AI systems that adapt to reality
- Critical for production systems serving diverse user needs
- Enables truly intelligent and responsive information systems
"""

import asyncio
import time
import json
import uuid
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import math
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FeedbackType(Enum):
    """Types of user feedback"""
    EXPLICIT_RATING = "explicit_rating"     # Direct user rating (1-5 stars)
    CLICK_THROUGH = "click_through"         # User clicked on result
    DWELL_TIME = "dwell_time"              # Time spent on result
    TASK_COMPLETION = "task_completion"     # Whether user completed their task
    QUERY_REFORMULATION = "query_reformulation"  # User refined their query
    RESULT_REJECTION = "result_rejection"   # User dismissed/ignored results

class AdaptationStrategy(Enum):
    """Strategies for system adaptation"""
    ONLINE_LEARNING = "online_learning"         # Continuous learning from interactions
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # RL-based optimization
    BANDIT_OPTIMIZATION = "bandit_optimization" # Multi-armed bandit for strategy selection
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"  # Bayesian parameter tuning
    EVOLUTIONARY = "evolutionary"              # Evolutionary algorithm optimization
    ENSEMBLE_LEARNING = "ensemble_learning"    # Adaptive ensemble methods

class QueryPattern(Enum):
    """Patterns of user queries"""
    EXPLORATORY = "exploratory"        # Open-ended exploration
    SPECIFIC_LOOKUP = "specific_lookup" # Looking for specific information
    COMPARISON = "comparison"           # Comparing options
    TROUBLESHOOTING = "troubleshooting" # Solving problems
    LEARNING = "learning"              # Educational/learning queries
    TASK_ORIENTED = "task_oriented"    # Completing specific tasks

@dataclass
class UserInteraction:
    """Record of user interaction with RAG system"""
    interaction_id: str
    user_id: str
    session_id: str
    
    # Query information
    query_text: str
    query_timestamp: datetime
    query_pattern: Optional[QueryPattern] = None
    
    # System response
    documents_returned: List[str] = field(default_factory=list)
    response_time: float = 0.0
    retrieval_strategy: str = ""
    
    # User feedback
    feedback_type: Optional[FeedbackType] = None
    feedback_value: Optional[float] = None
    clicked_documents: List[str] = field(default_factory=list)
    dwell_times: Dict[str, float] = field(default_factory=dict)
    
    # Contextual information
    user_context: Dict[str, Any] = field(default_factory=dict)
    task_completed: Optional[bool] = None
    follow_up_query: Optional[str] = None
    
    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())

@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptive optimization"""
    
    # Accuracy metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    
    # User satisfaction metrics
    click_through_rate: float = 0.0
    user_rating_average: float = 0.0
    task_completion_rate: float = 0.0
    session_success_rate: float = 0.0
    
    # Efficiency metrics
    average_response_time: float = 0.0
    query_success_rate: float = 0.0
    
    # Learning metrics
    improvement_rate: float = 0.0
    adaptation_speed: float = 0.0
    
    # Temporal tracking
    measurement_window: timedelta = timedelta(hours=24)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AdaptationParameters:
    """Parameters that can be adapted by the system"""
    
    # Retrieval parameters
    top_k: int = 10
    similarity_threshold: float = 0.7
    reranking_enabled: bool = True
    
    # Query processing parameters
    query_expansion_strength: float = 0.5
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4
    
    # Personalization parameters
    user_history_weight: float = 0.3
    context_weight: float = 0.2
    recency_bias: float = 0.1
    
    # Strategy selection parameters
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            'reranking_enabled': self.reranking_enabled,
            'query_expansion_strength': self.query_expansion_strength,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'user_history_weight': self.user_history_weight,
            'context_weight': self.context_weight,
            'recency_bias': self.recency_bias,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }

class FeedbackCollector:
    """Collects and processes user feedback"""
    
    def __init__(self):
        self.interactions: List[UserInteraction] = []
        self.feedback_buffer = deque(maxlen=1000)
        
        # Feedback analysis
        self.feedback_patterns = defaultdict(list)
        self.user_profiles = defaultdict(dict)
        
        self.logger = logging.getLogger("FeedbackCollector")
    
    async def record_interaction(self, interaction: UserInteraction) -> None:
        """Record user interaction"""
        
        try:
            self.interactions.append(interaction)
            self.feedback_buffer.append(interaction)
            
            # Update user profile
            await self._update_user_profile(interaction)
            
            # Analyze feedback patterns
            await self._analyze_feedback_patterns(interaction)
            
            self.logger.debug(f"Recorded interaction: {interaction.interaction_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record interaction: {e}")
    
    async def collect_implicit_feedback(self, interaction_id: str, 
                                      feedback_data: Dict[str, Any]) -> None:
        """Collect implicit feedback (clicks, dwell time, etc.)"""
        
        try:
            # Find interaction
            interaction = None
            for inter in self.interactions:
                if inter.interaction_id == interaction_id:
                    interaction = inter
                    break
            
            if not interaction:
                self.logger.warning(f"Interaction {interaction_id} not found")
                return
            
            # Update interaction with feedback
            if 'clicked_documents' in feedback_data:
                interaction.clicked_documents = feedback_data['clicked_documents']
                interaction.feedback_type = FeedbackType.CLICK_THROUGH
                
                # Calculate click-through rate
                if interaction.documents_returned:
                    ctr = len(interaction.clicked_documents) / len(interaction.documents_returned)
                    interaction.feedback_value = ctr
            
            if 'dwell_times' in feedback_data:
                interaction.dwell_times = feedback_data['dwell_times']
                
                # Calculate average dwell time as feedback signal
                if interaction.dwell_times:
                    avg_dwell = sum(interaction.dwell_times.values()) / len(interaction.dwell_times)
                    
                    # Convert dwell time to feedback score (0-1)
                    # Assume 30 seconds is perfect engagement
                    normalized_dwell = min(1.0, avg_dwell / 30.0)
                    interaction.feedback_value = normalized_dwell
            
            if 'task_completed' in feedback_data:
                interaction.task_completed = feedback_data['task_completed']
                interaction.feedback_type = FeedbackType.TASK_COMPLETION
                interaction.feedback_value = 1.0 if feedback_data['task_completed'] else 0.0
            
            self.logger.debug(f"Updated interaction {interaction_id} with implicit feedback")
            
        except Exception as e:
            self.logger.error(f"Failed to collect implicit feedback: {e}")
    
    async def collect_explicit_feedback(self, interaction_id: str, 
                                      rating: float, comments: str = "") -> None:
        """Collect explicit user feedback"""
        
        try:
            # Find interaction
            interaction = None
            for inter in self.interactions:
                if inter.interaction_id == interaction_id:
                    interaction = inter
                    break
            
            if not interaction:
                self.logger.warning(f"Interaction {interaction_id} not found")
                return
            
            # Update with explicit feedback
            interaction.feedback_type = FeedbackType.EXPLICIT_RATING
            interaction.feedback_value = rating / 5.0  # Normalize to 0-1
            
            if comments:
                interaction.user_context['feedback_comments'] = comments
            
            self.logger.debug(f"Recorded explicit feedback for {interaction_id}: {rating}/5")
            
        except Exception as e:
            self.logger.error(f"Failed to collect explicit feedback: {e}")
    
    async def _update_user_profile(self, interaction: UserInteraction) -> None:
        """Update user profile based on interaction"""
        
        user_id = interaction.user_id
        profile = self.user_profiles[user_id]
        
        # Track query patterns
        if 'query_patterns' not in profile:
            profile['query_patterns'] = defaultdict(int)
        
        if interaction.query_pattern:
            profile['query_patterns'][interaction.query_pattern.value] += 1
        
        # Track response time preferences
        if 'response_time_satisfaction' not in profile:
            profile['response_time_satisfaction'] = []
        
        if interaction.feedback_value is not None:
            profile['response_time_satisfaction'].append({
                'response_time': interaction.response_time,
                'satisfaction': interaction.feedback_value
            })
        
        # Track document preferences
        if 'document_preferences' not in profile:
            profile['document_preferences'] = defaultdict(float)
        
        for doc_id in interaction.clicked_documents:
            profile['document_preferences'][doc_id] += 0.1
        
        # Track successful strategies
        if 'successful_strategies' not in profile:
            profile['successful_strategies'] = defaultdict(float)
        
        if interaction.feedback_value and interaction.feedback_value > 0.7:
            profile['successful_strategies'][interaction.retrieval_strategy] += 0.1
    
    async def _analyze_feedback_patterns(self, interaction: UserInteraction) -> None:
        """Analyze feedback patterns for insights"""
        
        if interaction.feedback_value is None:
            return
        
        # Store pattern for analysis
        pattern_key = f"{interaction.query_pattern}_{interaction.retrieval_strategy}"
        self.feedback_patterns[pattern_key].append(interaction.feedback_value)
        
        # Maintain sliding window
        if len(self.feedback_patterns[pattern_key]) > 100:
            self.feedback_patterns[pattern_key] = self.feedback_patterns[pattern_key][-100:]
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences for personalization"""
        
        profile = self.user_profiles.get(user_id, {})
        
        preferences = {
            'preferred_query_patterns': {},
            'successful_strategies': {},
            'optimal_response_time': 1.0,
            'document_type_preferences': {}
        }
        
        # Query pattern preferences
        if 'query_patterns' in profile:
            total_queries = sum(profile['query_patterns'].values())
            for pattern, count in profile['query_patterns'].items():
                preferences['preferred_query_patterns'][pattern] = count / total_queries
        
        # Successful strategy preferences
        if 'successful_strategies' in profile:
            preferences['successful_strategies'] = dict(profile['successful_strategies'])
        
        # Optimal response time
        if 'response_time_satisfaction' in profile and profile['response_time_satisfaction']:
            # Find response time with highest satisfaction
            best_time = max(profile['response_time_satisfaction'], 
                          key=lambda x: x['satisfaction'])
            preferences['optimal_response_time'] = best_time['response_time']
        
        return preferences
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback collection statistics"""
        
        if not self.interactions:
            return {}
        
        # Calculate overall statistics
        total_interactions = len(self.interactions)
        interactions_with_feedback = len([i for i in self.interactions if i.feedback_value is not None])
        
        feedback_coverage = interactions_with_feedback / total_interactions if total_interactions > 0 else 0
        
        # Average feedback by type
        feedback_by_type = defaultdict(list)
        for interaction in self.interactions:
            if interaction.feedback_type and interaction.feedback_value is not None:
                feedback_by_type[interaction.feedback_type.value].append(interaction.feedback_value)
        
        avg_feedback_by_type = {}
        for feedback_type, values in feedback_by_type.items():
            avg_feedback_by_type[feedback_type] = sum(values) / len(values)
        
        # Pattern analysis
        pattern_performance = {}
        for pattern, feedback_values in self.feedback_patterns.items():
            if feedback_values:
                pattern_performance[pattern] = {
                    'average_feedback': sum(feedback_values) / len(feedback_values),
                    'sample_count': len(feedback_values),
                    'improvement_trend': self._calculate_trend(feedback_values)
                }
        
        return {
            'total_interactions': total_interactions,
            'feedback_coverage': feedback_coverage,
            'average_feedback_by_type': avg_feedback_by_type,
            'pattern_performance': pattern_performance,
            'unique_users': len(self.user_profiles)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate improvement trend in feedback values"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x2 = sum(a * a for a in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope

class PerformanceMonitor:
    """Monitors system performance and tracks improvements"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics = PerformanceMetrics()
        
        # Performance tracking
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.improvement_targets = {
            'click_through_rate': 0.05,  # 5% improvement target
            'user_rating_average': 0.1,  # 0.1 point improvement
            'task_completion_rate': 0.1,  # 10% improvement
            'average_response_time': -0.2  # 20% faster (negative = improvement)
        }
        
        self.logger = logging.getLogger("PerformanceMonitor")
    
    async def update_metrics(self, interactions: List[UserInteraction]) -> None:
        """Update performance metrics based on interactions"""
        
        try:
            if not interactions:
                return
            
            # Calculate metrics from recent interactions
            recent_interactions = [i for i in interactions 
                                 if i.query_timestamp >= datetime.now() - self.current_metrics.measurement_window]
            
            if not recent_interactions:
                return
            
            # Calculate accuracy metrics (simplified)
            await self._calculate_accuracy_metrics(recent_interactions)
            
            # Calculate user satisfaction metrics
            await self._calculate_satisfaction_metrics(recent_interactions)
            
            # Calculate efficiency metrics
            await self._calculate_efficiency_metrics(recent_interactions)
            
            # Calculate learning metrics
            await self._calculate_learning_metrics()
            
            # Update timestamp
            self.current_metrics.last_updated = datetime.now()
            
            # Store historical metrics
            self.metrics_history.append(self.current_metrics)
            
            # Set baseline if not set
            if self.baseline_metrics is None:
                self.baseline_metrics = self.current_metrics
            
            self.logger.debug("Updated performance metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    async def _calculate_accuracy_metrics(self, interactions: List[UserInteraction]) -> None:
        """Calculate precision, recall, and NDCG metrics"""
        
        # Simplified accuracy calculation based on feedback
        relevant_interactions = [i for i in interactions if i.feedback_value is not None]
        
        if not relevant_interactions:
            return
        
        # Calculate precision@k (fraction of retrieved docs that are relevant)
        for k in [1, 3, 5, 10]:
            precision_scores = []
            
            for interaction in relevant_interactions:
                if len(interaction.documents_returned) >= k:
                    # Use clicked documents as relevant
                    clicked_in_top_k = len([doc for doc in interaction.clicked_documents 
                                          if doc in interaction.documents_returned[:k]])
                    precision = clicked_in_top_k / k
                    precision_scores.append(precision)
            
            if precision_scores:
                self.current_metrics.precision_at_k[k] = sum(precision_scores) / len(precision_scores)
        
        # Calculate NDCG@k (simplified)
        for k in [1, 3, 5, 10]:
            ndcg_scores = []
            
            for interaction in relevant_interactions:
                if len(interaction.documents_returned) >= k and interaction.clicked_documents:
                    # Simple NDCG calculation
                    dcg = 0.0
                    for i, doc in enumerate(interaction.documents_returned[:k]):
                        if doc in interaction.clicked_documents:
                            dcg += 1.0 / math.log2(i + 2)
                    
                    # Ideal DCG (all relevant docs at top)
                    num_relevant = min(len(interaction.clicked_documents), k)
                    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
                    
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    ndcg_scores.append(ndcg)
            
            if ndcg_scores:
                self.current_metrics.ndcg_at_k[k] = sum(ndcg_scores) / len(ndcg_scores)
    
    async def _calculate_satisfaction_metrics(self, interactions: List[UserInteraction]) -> None:
        """Calculate user satisfaction metrics"""
        
        # Click-through rate
        interactions_with_results = [i for i in interactions if i.documents_returned]
        if interactions_with_results:
            clicked_interactions = [i for i in interactions_with_results if i.clicked_documents]
            self.current_metrics.click_through_rate = len(clicked_interactions) / len(interactions_with_results)
        
        # User rating average
        rated_interactions = [i for i in interactions 
                            if i.feedback_type == FeedbackType.EXPLICIT_RATING and i.feedback_value is not None]
        if rated_interactions:
            # Convert back to 1-5 scale
            ratings = [i.feedback_value * 5.0 for i in rated_interactions]
            self.current_metrics.user_rating_average = sum(ratings) / len(ratings)
        
        # Task completion rate
        task_interactions = [i for i in interactions if i.task_completed is not None]
        if task_interactions:
            completed_tasks = [i for i in task_interactions if i.task_completed]
            self.current_metrics.task_completion_rate = len(completed_tasks) / len(task_interactions)
        
        # Session success rate (sessions with positive feedback)
        sessions = defaultdict(list)
        for interaction in interactions:
            sessions[interaction.session_id].append(interaction)
        
        successful_sessions = 0
        for session_interactions in sessions.values():
            # Session is successful if any interaction has positive feedback
            session_feedback = [i.feedback_value for i in session_interactions 
                              if i.feedback_value is not None]
            if session_feedback and max(session_feedback) > 0.7:
                successful_sessions += 1
        
        if sessions:
            self.current_metrics.session_success_rate = successful_sessions / len(sessions)
    
    async def _calculate_efficiency_metrics(self, interactions: List[UserInteraction]) -> None:
        """Calculate efficiency metrics"""
        
        # Average response time
        response_times = [i.response_time for i in interactions if i.response_time > 0]
        if response_times:
            self.current_metrics.average_response_time = sum(response_times) / len(response_times)
        
        # Query success rate (queries that returned results)
        total_queries = len(interactions)
        successful_queries = len([i for i in interactions if i.documents_returned])
        if total_queries > 0:
            self.current_metrics.query_success_rate = successful_queries / total_queries
    
    async def _calculate_learning_metrics(self) -> None:
        """Calculate learning and improvement metrics"""
        
        if len(self.metrics_history) < 2:
            return
        
        # Calculate improvement rate over time
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        
        if len(recent_metrics) >= 2:
            # Calculate trend in click-through rate
            ctr_values = [m.click_through_rate for m in recent_metrics]
            ctr_trend = self._calculate_trend(ctr_values)
            
            # Calculate trend in user ratings
            rating_values = [m.user_rating_average for m in recent_metrics if m.user_rating_average > 0]
            rating_trend = self._calculate_trend(rating_values) if rating_values else 0.0
            
            # Combined improvement rate
            self.current_metrics.improvement_rate = (ctr_trend + rating_trend) / 2
        
        # Adaptation speed (how quickly system responds to changes)
        if self.baseline_metrics:
            baseline_ctr = self.baseline_metrics.click_through_rate
            current_ctr = self.current_metrics.click_through_rate
            
            if baseline_ctr > 0:
                improvement_percentage = (current_ctr - baseline_ctr) / baseline_ctr
                time_elapsed = (datetime.now() - self.baseline_metrics.last_updated).total_seconds() / 3600  # hours
                
                if time_elapsed > 0:
                    self.current_metrics.adaptation_speed = improvement_percentage / time_elapsed
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in metric values"""
        
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        sum_x2 = sum(a * a for a in x)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope
    
    def detect_performance_regression(self) -> Dict[str, Any]:
        """Detect performance regressions"""
        
        if not self.baseline_metrics:
            return {'regression_detected': False}
        
        regressions = {}
        
        # Check key metrics for regression
        metrics_to_check = {
            'click_through_rate': (self.baseline_metrics.click_through_rate, self.current_metrics.click_through_rate),
            'user_rating_average': (self.baseline_metrics.user_rating_average, self.current_metrics.user_rating_average),
            'task_completion_rate': (self.baseline_metrics.task_completion_rate, self.current_metrics.task_completion_rate)
        }
        
        for metric_name, (baseline, current) in metrics_to_check.items():
            if baseline > 0:
                change_percentage = (current - baseline) / baseline
                
                # Consider >10% decrease as regression
                if change_percentage < -0.1:
                    regressions[metric_name] = {
                        'baseline': baseline,
                        'current': current,
                        'change_percentage': change_percentage
                    }
        
        return {
            'regression_detected': len(regressions) > 0,
            'regressions': regressions,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        summary = {
            'current_metrics': {
                'click_through_rate': self.current_metrics.click_through_rate,
                'user_rating_average': self.current_metrics.user_rating_average,
                'task_completion_rate': self.current_metrics.task_completion_rate,
                'average_response_time': self.current_metrics.average_response_time,
                'query_success_rate': self.current_metrics.query_success_rate,
                'improvement_rate': self.current_metrics.improvement_rate,
                'adaptation_speed': self.current_metrics.adaptation_speed
            },
            'accuracy_metrics': {
                'precision_at_k': dict(self.current_metrics.precision_at_k),
                'ndcg_at_k': dict(self.current_metrics.ndcg_at_k)
            },
            'measurements_count': len(self.metrics_history),
            'last_updated': self.current_metrics.last_updated.isoformat()
        }
        
        # Add baseline comparison if available
        if self.baseline_metrics:
            summary['improvement_vs_baseline'] = {
                'click_through_rate': self.current_metrics.click_through_rate - self.baseline_metrics.click_through_rate,
                'user_rating_average': self.current_metrics.user_rating_average - self.baseline_metrics.user_rating_average,
                'task_completion_rate': self.current_metrics.task_completion_rate - self.baseline_metrics.task_completion_rate
            }
        
        return summary

class AdaptiveOptimizer:
    """Optimizes system parameters based on performance feedback"""
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ONLINE_LEARNING):
        self.adaptation_strategy = adaptation_strategy
        self.current_parameters = AdaptationParameters()
        self.parameter_history: List[Tuple[AdaptationParameters, float]] = []
        
        # Optimization state
        self.optimization_iteration = 0
        self.best_parameters: Optional[AdaptationParameters] = None
        self.best_performance: float = 0.0
        
        # Strategy-specific state
        self.bandit_arms: Dict[str, Dict[str, float]] = {}  # For bandit optimization
        self.exploration_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("AdaptiveOptimizer")
    
    async def optimize_parameters(self, performance_metrics: PerformanceMetrics, 
                                user_feedback: Dict[str, Any]) -> AdaptationParameters:
        """Optimize system parameters based on performance"""
        
        try:
            # Calculate overall performance score
            performance_score = await self._calculate_performance_score(performance_metrics)
            
            # Store current performance
            self.parameter_history.append((self.current_parameters, performance_score))
            
            # Update best parameters if this is the best so far
            if performance_score > self.best_performance:
                self.best_performance = performance_score
                self.best_parameters = self.current_parameters
                self.logger.info(f"New best performance: {performance_score:.3f}")
            
            # Apply optimization strategy
            if self.adaptation_strategy == AdaptationStrategy.ONLINE_LEARNING:
                new_parameters = await self._online_learning_optimization(performance_score)
            
            elif self.adaptation_strategy == AdaptationStrategy.BANDIT_OPTIMIZATION:
                new_parameters = await self._bandit_optimization(performance_score)
            
            elif self.adaptation_strategy == AdaptationStrategy.BAYESIAN_OPTIMIZATION:
                new_parameters = await self._bayesian_optimization(performance_score)
            
            else:  # Default to gradient-based optimization
                new_parameters = await self._gradient_optimization(performance_score)
            
            self.current_parameters = new_parameters
            self.optimization_iteration += 1
            
            self.logger.debug(f"Optimization iteration {self.optimization_iteration}, "
                            f"performance: {performance_score:.3f}")
            
            return new_parameters
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            return self.current_parameters
    
    async def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score"""
        
        # Weighted combination of different metrics
        weights = {
            'click_through_rate': 0.3,
            'user_rating_average': 0.2,
            'task_completion_rate': 0.2,
            'query_success_rate': 0.1,
            'response_time': 0.1,  # Inverted - faster is better
            'improvement_rate': 0.1
        }
        
        score = 0.0
        
        # Click-through rate (0-1)
        score += weights['click_through_rate'] * metrics.click_through_rate
        
        # User rating (normalize from 1-5 to 0-1)
        if metrics.user_rating_average > 0:
            normalized_rating = (metrics.user_rating_average - 1) / 4
            score += weights['user_rating_average'] * normalized_rating
        
        # Task completion rate (0-1)
        score += weights['task_completion_rate'] * metrics.task_completion_rate
        
        # Query success rate (0-1)
        score += weights['query_success_rate'] * metrics.query_success_rate
        
        # Response time (inverted and normalized)
        if metrics.average_response_time > 0:
            # Assume 5 seconds is the worst acceptable time
            normalized_time = max(0, 1 - metrics.average_response_time / 5.0)
            score += weights['response_time'] * normalized_time
        
        # Improvement rate (can be positive or negative)
        normalized_improvement = max(0, min(1, metrics.improvement_rate + 0.5))
        score += weights['improvement_rate'] * normalized_improvement
        
        return score
    
    async def _online_learning_optimization(self, performance_score: float) -> AdaptationParameters:
        """Online learning parameter optimization"""
        
        new_params = AdaptationParameters()
        
        # Copy current parameters
        for field_name, field_value in self.current_parameters.__dict__.items():
            setattr(new_params, field_name, field_value)
        
        # Learning rate
        lr = new_params.learning_rate
        
        # Gradient estimation through finite differences
        if len(self.parameter_history) >= 2:
            prev_params, prev_score = self.parameter_history[-2]
            
            # Estimate gradient for key parameters
            param_changes = {
                'top_k': self.current_parameters.top_k - prev_params.top_k,
                'similarity_threshold': self.current_parameters.similarity_threshold - prev_params.similarity_threshold,
                'query_expansion_strength': self.current_parameters.query_expansion_strength - prev_params.query_expansion_strength,
                'semantic_weight': self.current_parameters.semantic_weight - prev_params.semantic_weight
            }
            
            score_change = performance_score - prev_score
            
            # Update parameters based on estimated gradients
            for param_name, param_change in param_changes.items():
                if param_change != 0:
                    gradient = score_change / param_change
                    
                    # Update parameter
                    current_value = getattr(new_params, param_name)
                    new_value = current_value + lr * gradient
                    
                    # Apply constraints
                    if param_name == 'top_k':
                        new_value = max(1, min(20, int(new_value)))
                    elif param_name in ['similarity_threshold', 'query_expansion_strength', 'semantic_weight']:
                        new_value = max(0.0, min(1.0, new_value))
                    
                    setattr(new_params, param_name, new_value)
        
        # Add exploration noise
        if random.random() < new_params.exploration_rate:
            await self._add_exploration_noise(new_params)
        
        return new_params
    
    async def _bandit_optimization(self, performance_score: float) -> AdaptationParameters:
        """Multi-armed bandit optimization"""
        
        # Define parameter arms (discrete choices)
        parameter_arms = {
            'top_k': [5, 10, 15, 20],
            'similarity_threshold': [0.5, 0.6, 0.7, 0.8],
            'query_expansion_strength': [0.3, 0.5, 0.7],
            'semantic_weight': [0.4, 0.6, 0.8]
        }
        
        new_params = AdaptationParameters()
        
        # Initialize bandit arms if not done
        for param_name, arm_values in parameter_arms.items():
            if param_name not in self.bandit_arms:
                self.bandit_arms[param_name] = {
                    str(value): {'count': 0, 'total_reward': 0.0, 'avg_reward': 0.0}
                    for value in arm_values
                }
        
        # Update current arm rewards
        for param_name in parameter_arms.keys():
            current_value = str(getattr(self.current_parameters, param_name))
            if current_value in self.bandit_arms[param_name]:
                arm = self.bandit_arms[param_name][current_value]
                arm['count'] += 1
                arm['total_reward'] += performance_score
                arm['avg_reward'] = arm['total_reward'] / arm['count']
        
        # Select arms using Upper Confidence Bound (UCB)
        for param_name, arm_values in parameter_arms.items():
            total_counts = sum(arm['count'] for arm in self.bandit_arms[param_name].values())
            
            best_arm = None
            best_ucb = -float('inf')
            
            for value_str, arm in self.bandit_arms[param_name].items():
                if arm['count'] == 0:
                    ucb = float('inf')  # Explore unvisited arms
                else:
                    confidence = math.sqrt(2 * math.log(total_counts) / arm['count'])
                    ucb = arm['avg_reward'] + confidence
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_arm = value_str
            
            if best_arm:
                # Convert back to appropriate type
                if param_name == 'top_k':
                    setattr(new_params, param_name, int(best_arm))
                else:
                    setattr(new_params, param_name, float(best_arm))
        
        return new_params
    
    async def _bayesian_optimization(self, performance_score: float) -> AdaptationParameters:
        """Bayesian optimization (simplified)"""
        
        # For simplicity, use a grid search with Gaussian Process-like behavior
        new_params = AdaptationParameters()
        
        # If we have enough history, use best parameters as starting point
        if len(self.parameter_history) >= 5:
            # Find parameters with highest average performance
            param_performance = defaultdict(list)
            
            for params, score in self.parameter_history[-10:]:  # Last 10 iterations
                param_dict = params.to_dict()
                for param_name, param_value in param_dict.items():
                    param_performance[param_name].append((param_value, score))
            
            # For each parameter, find value with best performance
            for param_name, value_score_pairs in param_performance.items():
                if not value_score_pairs:
                    continue
                
                # Group by value and calculate average performance
                value_groups = defaultdict(list)
                for value, score in value_score_pairs:
                    # Discretize continuous values
                    if isinstance(value, float):
                        discretized = round(value, 1)
                    else:
                        discretized = value
                    value_groups[discretized].append(score)
                
                # Find best performing value
                best_value = None
                best_avg_score = -float('inf')
                
                for value, scores in value_groups.items():
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_value = value
                
                if best_value is not None:
                    setattr(new_params, param_name, best_value)
        
        # Add some exploration around best parameters
        await self._add_exploration_noise(new_params, noise_scale=0.1)
        
        return new_params
    
    async def _gradient_optimization(self, performance_score: float) -> AdaptationParameters:
        """Simple gradient-based optimization"""
        
        # Very basic gradient descent with finite differences
        new_params = AdaptationParameters()
        
        # Copy current parameters
        for field_name, field_value in self.current_parameters.__dict__.items():
            setattr(new_params, field_name, field_value)
        
        # If we have performance history, estimate gradients
        if len(self.parameter_history) >= 2:
            current_params = self.current_parameters
            prev_params, prev_score = self.parameter_history[-2]
            
            lr = 0.01  # Learning rate
            
            # Simple parameter updates
            param_updates = {
                'similarity_threshold': lr * (performance_score - prev_score) * 0.1,
                'query_expansion_strength': lr * (performance_score - prev_score) * 0.1,
                'semantic_weight': lr * (performance_score - prev_score) * 0.1
            }
            
            for param_name, update in param_updates.items():
                current_value = getattr(new_params, param_name)
                new_value = current_value + update
                
                # Apply constraints
                new_value = max(0.0, min(1.0, new_value))
                setattr(new_params, param_name, new_value)
        
        return new_params
    
    async def _add_exploration_noise(self, params: AdaptationParameters, 
                                   noise_scale: float = 0.05) -> None:
        """Add exploration noise to parameters"""
        
        # Add small random noise to continuous parameters
        continuous_params = [
            'similarity_threshold', 'query_expansion_strength', 
            'semantic_weight', 'keyword_weight', 'user_history_weight',
            'context_weight', 'recency_bias'
        ]
        
        for param_name in continuous_params:
            current_value = getattr(params, param_name)
            noise = random.gauss(0, noise_scale)
            new_value = current_value + noise
            
            # Apply constraints
            new_value = max(0.0, min(1.0, new_value))
            setattr(params, param_name, new_value)
        
        # Add noise to discrete parameters
        if random.random() < 0.1:  # 10% chance to modify top_k
            current_k = params.top_k
            new_k = current_k + random.choice([-1, 1])
            params.top_k = max(1, min(20, new_k))
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        
        summary = {
            'adaptation_strategy': self.adaptation_strategy.value,
            'optimization_iterations': self.optimization_iteration,
            'best_performance': self.best_performance,
            'current_performance': self.parameter_history[-1][1] if self.parameter_history else 0.0,
            'parameter_history_length': len(self.parameter_history)
        }
        
        if self.best_parameters:
            summary['best_parameters'] = self.best_parameters.to_dict()
        
        summary['current_parameters'] = self.current_parameters.to_dict()
        
        # Add strategy-specific information
        if self.adaptation_strategy == AdaptationStrategy.BANDIT_OPTIMIZATION and self.bandit_arms:
            summary['bandit_arm_performance'] = {}
            for param_name, arms in self.bandit_arms.items():
                summary['bandit_arm_performance'][param_name] = {
                    arm_value: arm_data['avg_reward']
                    for arm_value, arm_data in arms.items()
                }
        
        return summary

class AdaptiveRAGSystem:
    """
    Complete Adaptive RAG System that learns and improves over time
    
    EXAMPLE USAGE:
    =============
    # Create adaptive RAG system
    rag = AdaptiveRAGSystem()
    await rag.initialize()
    
    # Process queries and collect feedback
    result = await rag.adaptive_search("machine learning algorithms", user_id="user123")
    
    # Provide feedback
    await rag.record_feedback(
        interaction_id=result['interaction_id'],
        feedback_type=FeedbackType.EXPLICIT_RATING,
        feedback_value=4.0  # 4/5 stars
    )
    
    # System automatically adapts based on feedback
    await rag.trigger_adaptation()
    
    # Check system improvements
    summary = rag.get_adaptation_summary()
    print(f"Performance improvement: {summary['performance_improvement']:.2%}")
    """
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ONLINE_LEARNING):
        # Core components
        self.feedback_collector = FeedbackCollector()
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_optimizer = AdaptiveOptimizer(adaptation_strategy)
        
        # Mock document store (in real implementation, use actual retriever)
        self.documents = self._create_mock_documents()
        
        # System state
        self.adaptation_enabled = True
        self.adaptation_frequency = timedelta(hours=1)  # Adapt every hour
        self.last_adaptation = datetime.now()
        
        # Statistics
        self.system_stats = {
            'total_queries': 0,
            'adaptations_performed': 0,
            'performance_improvements': 0,
            'user_satisfaction_trend': [],
            'adaptation_effectiveness': 0.0
        }
        
        self.logger = logging.getLogger("AdaptiveRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize adaptive RAG system"""
        self.logger.info("Adaptive RAG system initialized")
    
    async def adaptive_search(self, query: str, user_id: str, 
                            session_id: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform adaptive search with personalization"""
        
        start_time = time.time()
        self.system_stats['total_queries'] += 1
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Get user preferences for personalization
            user_preferences = self.feedback_collector.get_user_preferences(user_id)
            
            # Get current system parameters
            params = self.adaptive_optimizer.current_parameters
            
            # Perform retrieval with current parameters
            documents = await self._retrieve_documents(query, params, user_preferences)
            
            # Determine query pattern
            query_pattern = self._classify_query_pattern(query)
            
            response_time = time.time() - start_time
            
            # Create interaction record
            interaction = UserInteraction(
                interaction_id="",
                user_id=user_id,
                session_id=session_id,
                query_text=query,
                query_timestamp=datetime.now(),
                query_pattern=query_pattern,
                documents_returned=[doc['id'] for doc in documents],
                response_time=response_time,
                retrieval_strategy=f"adaptive_{self.adaptive_optimizer.adaptation_strategy.value}",
                user_context=context or {}
            )
            
            # Record interaction
            await self.feedback_collector.record_interaction(interaction)
            
            result = {
                'success': True,
                'interaction_id': interaction.interaction_id,
                'query': query,
                'documents': documents,
                'response_time': response_time,
                'personalization_applied': len(user_preferences) > 0,
                'current_parameters': params.to_dict(),
                'query_pattern': query_pattern.value if query_pattern else 'unknown'
            }
            
            self.logger.info(f"Adaptive search completed for user {user_id}: "
                           f"{len(documents)} docs, {response_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptive search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'interaction_id': None,
                'response_time': time.time() - start_time
            }
    
    async def record_feedback(self, interaction_id: str, 
                            feedback_type: FeedbackType,
                            feedback_value: Optional[float] = None,
                            feedback_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record user feedback for adaptation"""
        
        try:
            if feedback_type == FeedbackType.EXPLICIT_RATING:
                if feedback_value is None:
                    return {'success': False, 'error': 'Rating value required'}
                
                await self.feedback_collector.collect_explicit_feedback(
                    interaction_id, feedback_value
                )
            
            elif feedback_type in [FeedbackType.CLICK_THROUGH, FeedbackType.DWELL_TIME, 
                                 FeedbackType.TASK_COMPLETION]:
                if feedback_data is None:
                    return {'success': False, 'error': 'Feedback data required'}
                
                await self.feedback_collector.collect_implicit_feedback(
                    interaction_id, feedback_data
                )
            
            # Check if adaptation should be triggered
            if self._should_trigger_adaptation():
                await self.trigger_adaptation()
            
            return {
                'success': True,
                'feedback_recorded': True,
                'adaptation_triggered': self._should_trigger_adaptation()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to record feedback: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def trigger_adaptation(self) -> Dict[str, Any]:
        """Trigger system adaptation based on feedback"""
        
        if not self.adaptation_enabled:
            return {'success': False, 'error': 'Adaptation disabled'}
        
        try:
            # Update performance metrics
            await self.performance_monitor.update_metrics(self.feedback_collector.interactions)
            
            # Get feedback statistics
            feedback_stats = self.feedback_collector.get_feedback_statistics()
            
            # Optimize parameters
            current_metrics = self.performance_monitor.current_metrics
            new_parameters = await self.adaptive_optimizer.optimize_parameters(
                current_metrics, feedback_stats
            )
            
            # Check for performance regression
            regression_check = self.performance_monitor.detect_performance_regression()
            
            # Update system statistics
            self.system_stats['adaptations_performed'] += 1
            self.last_adaptation = datetime.now()
            
            # Check if this adaptation improved performance
            if len(self.performance_monitor.metrics_history) >= 2:
                prev_metrics = self.performance_monitor.metrics_history[-2]
                current_metrics = self.performance_monitor.current_metrics
                
                # Compare key metrics
                ctr_improvement = current_metrics.click_through_rate - prev_metrics.click_through_rate
                rating_improvement = current_metrics.user_rating_average - prev_metrics.user_rating_average
                
                if ctr_improvement > 0 or rating_improvement > 0:
                    self.system_stats['performance_improvements'] += 1
            
            # Calculate adaptation effectiveness
            if self.system_stats['adaptations_performed'] > 0:
                self.system_stats['adaptation_effectiveness'] = (
                    self.system_stats['performance_improvements'] / 
                    self.system_stats['adaptations_performed']
                )
            
            result = {
                'success': True,
                'adaptation_performed': True,
                'new_parameters': new_parameters.to_dict(),
                'performance_summary': self.performance_monitor.get_performance_summary(),
                'regression_detected': regression_check['regression_detected'],
                'optimization_iteration': self.adaptive_optimizer.optimization_iteration
            }
            
            self.logger.info(f"Adaptation triggered: iteration {self.adaptive_optimizer.optimization_iteration}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Adaptation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _retrieve_documents(self, query: str, params: AdaptationParameters,
                                user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve documents using current parameters"""
        
        # Simulate document retrieval with current parameters
        query_words = set(query.lower().split())
        document_scores = []
        
        for doc in self.documents:
            # Calculate base relevance
            doc_words = set(doc['content'].lower().split())
            title_words = set(doc['title'].lower().split())
            
            content_overlap = len(query_words & doc_words) / max(len(query_words | doc_words), 1)
            title_overlap = len(query_words & title_words) / max(len(query_words | title_words), 1)
            
            base_score = content_overlap * params.semantic_weight + title_overlap * params.keyword_weight
            
            # Apply similarity threshold
            if base_score < params.similarity_threshold:
                continue
            
            # Apply personalization
            if user_preferences and 'document_type_preferences' in user_preferences:
                doc_type = doc.get('type', 'general')
                type_preference = user_preferences['document_type_preferences'].get(doc_type, 0.5)
                base_score = base_score * (1 + params.user_history_weight * type_preference)
            
            # Apply recency bias
            doc_age_days = (datetime.now() - datetime.fromisoformat(doc['created_date'])).days
            recency_factor = math.exp(-doc_age_days / 30.0)  # 30-day half-life
            final_score = base_score * (1 + params.recency_bias * recency_factor)
            
            document_scores.append((doc, final_score))
        
        # Sort by score and return top_k
        document_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in document_scores[:params.top_k]]
    
    def _classify_query_pattern(self, query: str) -> Optional[QueryPattern]:
        """Classify query pattern for adaptation"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'what', 'why', 'explain', 'learn']):
            return QueryPattern.LEARNING
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return QueryPattern.COMPARISON
        elif any(word in query_lower for word in ['problem', 'error', 'issue', 'fix', 'troubleshoot']):
            return QueryPattern.TROUBLESHOOTING
        elif any(word in query_lower for word in ['find', 'search', 'lookup', 'specific']):
            return QueryPattern.SPECIFIC_LOOKUP
        elif any(word in query_lower for word in ['explore', 'overview', 'general', 'about']):
            return QueryPattern.EXPLORATORY
        else:
            return QueryPattern.TASK_ORIENTED
    
    def _should_trigger_adaptation(self) -> bool:
        """Check if adaptation should be triggered"""
        
        if not self.adaptation_enabled:
            return False
        
        # Time-based trigger
        time_since_last = datetime.now() - self.last_adaptation
        if time_since_last >= self.adaptation_frequency:
            return True
        
        # Feedback-based trigger (enough new feedback)
        recent_feedback = [i for i in self.feedback_collector.interactions 
                          if i.feedback_value is not None and 
                          i.query_timestamp >= self.last_adaptation]
        
        if len(recent_feedback) >= 10:  # Minimum feedback threshold
            return True
        
        return False
    
    def _create_mock_documents(self) -> List[Dict[str, Any]]:
        """Create mock document collection"""
        
        documents = []
        topics = [
            ("machine learning", "technology"),
            ("data science", "technology"), 
            ("artificial intelligence", "technology"),
            ("business strategy", "business"),
            ("market analysis", "business"),
            ("customer service", "business"),
            ("software development", "technology"),
            ("project management", "business")
        ]
        
        for i in range(100):
            topic, doc_type = topics[i % len(topics)]
            
            doc = {
                'id': f'doc_{i:03d}',
                'title': f'{topic.title()} Guide {i // len(topics) + 1}',
                'content': f'Comprehensive guide about {topic} with detailed information and practical examples. '
                          f'This document covers key concepts, best practices, and real-world applications of {topic}.',
                'type': doc_type,
                'created_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'author': f'Expert {i % 10}',
                'relevance_score': random.uniform(0.5, 1.0)
            }
            documents.append(doc)
        
        return documents
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get comprehensive adaptation summary"""
        
        # Get component summaries
        feedback_stats = self.feedback_collector.get_feedback_statistics()
        performance_summary = self.performance_monitor.get_performance_summary()
        optimization_summary = self.adaptive_optimizer.get_optimization_summary()
        
        # Calculate overall performance improvement
        performance_improvement = 0.0
        if (self.performance_monitor.baseline_metrics and 
            self.performance_monitor.current_metrics):
            
            baseline_ctr = self.performance_monitor.baseline_metrics.click_through_rate
            current_ctr = self.performance_monitor.current_metrics.click_through_rate
            
            if baseline_ctr > 0:
                performance_improvement = (current_ctr - baseline_ctr) / baseline_ctr
        
        return {
            'system_statistics': self.system_stats,
            'performance_improvement': performance_improvement,
            'feedback_statistics': feedback_stats,
            'performance_summary': performance_summary,
            'optimization_summary': optimization_summary,
            'adaptation_enabled': self.adaptation_enabled,
            'last_adaptation': self.last_adaptation.isoformat(),
            'next_adaptation_due': (self.last_adaptation + self.adaptation_frequency).isoformat()
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            'adaptive_rag_stats': self.get_adaptation_summary(),
            'capabilities': {
                'continuous_learning': True,
                'user_personalization': True,
                'performance_optimization': True,
                'feedback_integration': True,
                'regression_detection': True,
                'a_b_testing': True
            }
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_feedback_collection():
    """Demo: Feedback collection and analysis"""
    print("\nDEMO 1: FEEDBACK COLLECTION AND ANALYSIS")
    print("=" * 50)
    
    collector = FeedbackCollector()
    
    # Simulate user interactions with different feedback types
    interactions = [
        {
            'user_id': 'user1',
            'query': 'machine learning algorithms',
            'docs_returned': ['doc_001', 'doc_002', 'doc_003'],
            'clicked_docs': ['doc_001'],
            'rating': 4.0,
            'dwell_time': {'doc_001': 45.0}
        },
        {
            'user_id': 'user1',
            'query': 'deep learning tutorials',
            'docs_returned': ['doc_004', 'doc_005', 'doc_006'],
            'clicked_docs': ['doc_004', 'doc_005'],
            'rating': 5.0,
            'dwell_time': {'doc_004': 60.0, 'doc_005': 30.0}
        },
        {
            'user_id': 'user2',
            'query': 'business strategy frameworks',
            'docs_returned': ['doc_007', 'doc_008'],
            'clicked_docs': [],
            'rating': 2.0,
            'dwell_time': {}
        }
    ]
    
    print("Simulating user interactions and feedback:")
    
    for i, interaction_data in enumerate(interactions, 1):
        # Create interaction
        interaction = UserInteraction(
            interaction_id=f"interaction_{i}",
            user_id=interaction_data['user_id'],
            session_id=f"session_{i}",
            query_text=interaction_data['query'],
            query_timestamp=datetime.now() - timedelta(minutes=i*10),
            documents_returned=interaction_data['docs_returned'],
            response_time=random.uniform(0.5, 2.0),
            retrieval_strategy="adaptive_system"
        )
        
        # Record interaction
        await collector.record_interaction(interaction)
        
        # Add implicit feedback
        await collector.collect_implicit_feedback(
            interaction.interaction_id,
            {
                'clicked_documents': interaction_data['clicked_docs'],
                'dwell_times': interaction_data['dwell_time']
            }
        )
        
        # Add explicit feedback
        await collector.collect_explicit_feedback(
            interaction.interaction_id,
            interaction_data['rating']
        )
        
        print(f"  ✓ Interaction {i}: {interaction_data['query'][:30]}...")
        print(f"    User: {interaction_data['user_id']}")
        print(f"    Clicks: {len(interaction_data['clicked_docs'])}")
        print(f"    Rating: {interaction_data['rating']}/5")
    
    # Show user preferences
    print(f"\nUser Preferences Analysis:")
    for user_id in ['user1', 'user2']:
        preferences = collector.get_user_preferences(user_id)
        print(f"\n  {user_id}:")
        print(f"    Successful strategies: {preferences['successful_strategies']}")
        print(f"    Optimal response time: {preferences['optimal_response_time']:.2f}s")
    
    # Show feedback statistics
    stats = collector.get_feedback_statistics()
    print(f"\nFeedback Statistics:")
    print(f"  Total interactions: {stats['total_interactions']}")
    print(f"  Feedback coverage: {stats['feedback_coverage']:.1%}")
    print(f"  Average feedback by type:")
    for feedback_type, avg_value in stats['average_feedback_by_type'].items():
        print(f"    {feedback_type}: {avg_value:.2f}")

async def demo_performance_monitoring():
    """Demo: Performance monitoring and regression detection"""
    print("\nDEMO 2: PERFORMANCE MONITORING")
    print("=" * 50)
    
    monitor = PerformanceMonitor()
    
    # Simulate interactions over time with changing performance
    time_periods = [
        {'name': 'Baseline', 'ctr': 0.3, 'rating': 3.5, 'completion': 0.6},
        {'name': 'Improvement 1', 'ctr': 0.35, 'rating': 3.8, 'completion': 0.65},
        {'name': 'Peak Performance', 'ctr': 0.42, 'rating': 4.2, 'completion': 0.75},
        {'name': 'Regression', 'ctr': 0.28, 'rating': 3.2, 'completion': 0.55},
        {'name': 'Recovery', 'ctr': 0.38, 'rating': 3.9, 'completion': 0.7}
    ]
    
    print("Simulating performance over time:")
    
    all_interactions = []
    
    for period_idx, period in enumerate(time_periods):
        print(f"\n--- {period['name']} Period ---")
        
        period_interactions = []
        
        # Generate interactions for this period
        for i in range(20):  # 20 interactions per period
            interaction = UserInteraction(
                interaction_id=f"perf_{period_idx}_{i}",
                user_id=f"user_{i%5}",
                session_id=f"session_{period_idx}_{i}",
                query_text=f"test query {i}",
                query_timestamp=datetime.now() - timedelta(days=(4-period_idx), hours=i),
                documents_returned=[f"doc_{j}" for j in range(5)],
                response_time=random.uniform(0.5, 3.0)
            )
            
            # Simulate feedback based on period performance
            if random.random() < period['ctr']:
                interaction.clicked_documents = [f"doc_{random.randint(0,2)}"]
                interaction.feedback_type = FeedbackType.CLICK_THROUGH
                interaction.feedback_value = period['ctr']
            
            if random.random() < 0.3:  # 30% explicit rating
                rating = random.gauss(period['rating'], 0.5)
                rating = max(1, min(5, rating))
                interaction.feedback_type = FeedbackType.EXPLICIT_RATING
                interaction.feedback_value = rating / 5.0
            
            if random.random() < period['completion']:
                interaction.task_completed = True
            
            period_interactions.append(interaction)
        
        all_interactions.extend(period_interactions)
        
        # Update metrics for this period
        await monitor.update_metrics(all_interactions[-20:])  # Last 20 interactions
        
        current_metrics = monitor.current_metrics
        print(f"  Click-through rate: {current_metrics.click_through_rate:.3f}")
        print(f"  User rating average: {current_metrics.user_rating_average:.2f}")
        print(f"  Task completion rate: {current_metrics.task_completion_rate:.3f}")
        
        # Check for regression
        regression = monitor.detect_performance_regression()
        if regression['regression_detected']:
            print(f"  ⚠️  Regression detected!")
            for metric, details in regression['regressions'].items():
                print(f"    {metric}: {details['change_percentage']:.1%} decrease")
        else:
            print(f"  ✅ Performance stable")
    
    # Show final performance summary
    print(f"\nFinal Performance Summary:")
    summary = monitor.get_performance_summary()
    current = summary['current_metrics']
    print(f"  Click-through rate: {current['click_through_rate']:.3f}")
    print(f"  User rating average: {current['user_rating_average']:.2f}")
    print(f"  Task completion rate: {current['task_completion_rate']:.3f}")
    print(f"  Improvement rate: {current['improvement_rate']:.3f}")
    print(f"  Adaptation speed: {current['adaptation_speed']:.3f}")
    
    if 'improvement_vs_baseline' in summary:
        print(f"\nImprovement vs Baseline:")
        for metric, improvement in summary['improvement_vs_baseline'].items():
            print(f"  {metric}: {improvement:+.3f}")

async def demo_adaptive_optimization():
    """Demo: Adaptive parameter optimization"""
    print("\nDEMO 3: ADAPTIVE PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Test different optimization strategies
    strategies = [
        AdaptationStrategy.ONLINE_LEARNING,
        AdaptationStrategy.BANDIT_OPTIMIZATION,
        AdaptationStrategy.BAYESIAN_OPTIMIZATION
    ]
    
    for strategy in strategies:
        print(f"\n--- {strategy.value.replace('_', ' ').title()} ---")
        
        optimizer = AdaptiveOptimizer(strategy)
        
        # Simulate optimization over multiple iterations
        performance_scores = []
        
        for iteration in range(10):
            # Simulate performance feedback
            # Add some noise and trend
            base_performance = 0.6 + 0.03 * iteration + random.gauss(0, 0.05)
            base_performance = max(0.1, min(1.0, base_performance))
            
            # Create mock performance metrics
            metrics = PerformanceMetrics()
            metrics.click_through_rate = base_performance * 0.8
            metrics.user_rating_average = base_performance * 4.0 + 1.0
            metrics.task_completion_rate = base_performance * 0.9
            metrics.improvement_rate = 0.01 if iteration > 0 else 0.0
            
            # Optimize parameters
            new_params = await optimizer.optimize_parameters(metrics, {})
            
            performance_scores.append(base_performance)
            
            if iteration % 3 == 0:  # Show every 3rd iteration
                print(f"  Iteration {iteration + 1}:")
                print(f"    Performance score: {base_performance:.3f}")
                print(f"    Top-k: {new_params.top_k}")
                print(f"    Similarity threshold: {new_params.similarity_threshold:.3f}")
                print(f"    Semantic weight: {new_params.semantic_weight:.3f}")
        
        # Show final optimization summary
        summary = optimizer.get_optimization_summary()
        print(f"\n  Final Results:")
        print(f"    Best performance: {summary['best_performance']:.3f}")
        print(f"    Current performance: {summary['current_performance']:.3f}")
        print(f"    Optimization iterations: {summary['optimization_iterations']}")
        
        # Show performance trend
        if len(performance_scores) > 1:
            initial = performance_scores[0]
            final = performance_scores[-1]
            improvement = (final - initial) / initial * 100
            print(f"    Overall improvement: {improvement:+.1f}%")

async def demo_complete_adaptive_system():
    """Demo: Complete adaptive RAG system"""
    print("\nDEMO 4: COMPLETE ADAPTIVE RAG SYSTEM")
    print("=" * 50)
    
    # Create adaptive RAG system
    rag_system = AdaptiveRAGSystem(AdaptationStrategy.ONLINE_LEARNING)
    await rag_system.initialize()
    
    # Simulate user sessions with learning
    users = ['alice', 'bob', 'charlie']
    
    queries_by_user = {
        'alice': [
            'machine learning basics',
            'neural network architectures', 
            'deep learning frameworks',
            'computer vision applications'
        ],
        'bob': [
            'business strategy planning',
            'market analysis techniques',
            'competitive intelligence',
            'growth strategy frameworks'
        ],
        'charlie': [
            'software development best practices',
            'agile project management',
            'code review processes',
            'team collaboration tools'
        ]
    }
    
    print("Simulating user sessions with adaptive learning:")
    
    session_results = []
    
    for session_round in range(3):  # 3 rounds of queries
        print(f"\n--- Session Round {session_round + 1} ---")
        
        for user in users:
            if session_round < len(queries_by_user[user]):
                query = queries_by_user[user][session_round]
                
                # Perform adaptive search
                result = await rag_system.adaptive_search(
                    query=query,
                    user_id=user,
                    context={'round': session_round + 1}
                )
                
                if result['success']:
                    print(f"  {user}: '{query}'")
                    print(f"    Documents: {len(result['documents'])}")
                    print(f"    Response time: {result['response_time']:.3f}s")
                    print(f"    Personalization: {result['personalization_applied']}")
                    
                    # Simulate user feedback
                    # Alice prefers technical content, Bob prefers business, Charlie prefers practical
                    satisfaction_by_user = {
                        'alice': 4.0 if 'machine' in query or 'neural' in query or 'deep' in query else 3.0,
                        'bob': 4.5 if 'business' in query or 'market' in query or 'strategy' in query else 2.5,
                        'charlie': 4.2 if 'software' in query or 'development' in query or 'agile' in query else 3.0
                    }
                    
                    # Add some randomness
                    satisfaction = satisfaction_by_user[user] + random.gauss(0, 0.5)
                    satisfaction = max(1.0, min(5.0, satisfaction))
                    
                    # Record feedback
                    await rag_system.record_feedback(
                        interaction_id=result['interaction_id'],
                        feedback_type=FeedbackType.EXPLICIT_RATING,
                        feedback_value=satisfaction
                    )
                    
                    # Simulate clicks (higher satisfaction = more clicks)
                    num_clicks = int(satisfaction / 5.0 * len(result['documents']))
                    clicked_docs = [doc['id'] for doc in result['documents'][:num_clicks]]
                    
                    if clicked_docs:
                        await rag_system.record_feedback(
                            interaction_id=result['interaction_id'],
                            feedback_type=FeedbackType.CLICK_THROUGH,
                            feedback_data={'clicked_documents': clicked_docs}
                        )
                    
                    session_results.append({
                        'user': user,
                        'query': query,
                        'satisfaction': satisfaction,
                        'round': session_round + 1
                    })
                    
                    print(f"    User satisfaction: {satisfaction:.1f}/5")
        
        # Trigger adaptation after each round
        if session_round < 2:  # Don't adapt after last round
            adaptation_result = await rag_system.trigger_adaptation()
            if adaptation_result['success']:
                print(f"\n  🔄 System adapted based on feedback")
                print(f"     Optimization iteration: {adaptation_result['optimization_iteration']}")
    
    # Show final adaptation summary
    print(f"\nADAPTIVE LEARNING RESULTS")
    print("=" * 30)
    
    summary = rag_system.get_adaptation_summary()
    
    print(f"System Performance:")
    print(f"  Total queries: {summary['system_statistics']['total_queries']}")
    print(f"  Adaptations performed: {summary['system_statistics']['adaptations_performed']}")
    print(f"  Performance improvements: {summary['system_statistics']['performance_improvements']}")
    print(f"  Adaptation effectiveness: {summary['system_statistics']['adaptation_effectiveness']:.1%}")
    print(f"  Overall performance improvement: {summary['performance_improvement']:+.1%}")
    
    # Show user satisfaction trends
    print(f"\nUser Satisfaction Trends:")
    for user in users:
        user_results = [r for r in session_results if r['user'] == user]
        if user_results:
            initial_satisfaction = user_results[0]['satisfaction']
            final_satisfaction = user_results[-1]['satisfaction']
            improvement = final_satisfaction - initial_satisfaction
            
            print(f"  {user}: {initial_satisfaction:.1f} → {final_satisfaction:.1f} ({improvement:+.1f})")

async def demo_system_analytics():
    """Demo: Comprehensive system analytics"""
    print("\nDEMO 5: SYSTEM ANALYTICS")
    print("=" * 50)
    
    rag_system = AdaptiveRAGSystem(AdaptationStrategy.BANDIT_OPTIMIZATION)
    await rag_system.initialize()
    
    # Generate comprehensive test data
    users = [f"user_{i}" for i in range(10)]
    query_templates = [
        "machine learning {topic}",
        "business strategy {topic}",
        "software development {topic}",
        "data science {topic}",
        "artificial intelligence {topic}"
    ]
    
    topics = ['basics', 'advanced', 'tools', 'best practices', 'frameworks']
    
    print("Generating comprehensive interaction data...")
    
    # Simulate 100 interactions across multiple users and adaptation cycles
    interaction_count = 0
    
    for cycle in range(5):  # 5 adaptation cycles
        print(f"  Cycle {cycle + 1}: ", end="")
        
        cycle_interactions = 0
        
        for _ in range(20):  # 20 interactions per cycle
            user = random.choice(users)
            template = random.choice(query_templates)
            topic = random.choice(topics)
            query = template.format(topic=topic)
            
            # Perform search
            result = await rag_system.adaptive_search(query, user)
            
            if result['success']:
                # Generate realistic feedback
                base_satisfaction = 3.5 + cycle * 0.2  # Improvement over cycles
                satisfaction = base_satisfaction + random.gauss(0, 1.0)
                satisfaction = max(1.0, min(5.0, satisfaction))
                
                # Record feedback
                await rag_system.record_feedback(
                    interaction_id=result['interaction_id'],
                    feedback_type=FeedbackType.EXPLICIT_RATING,
                    feedback_value=satisfaction
                )
                
                # Simulate clicks
                if satisfaction > 3.0:
                    num_clicks = random.randint(1, min(3, len(result['documents'])))
                    clicked_docs = [doc['id'] for doc in result['documents'][:num_clicks]]
                    
                    await rag_system.record_feedback(
                        interaction_id=result['interaction_id'],
                        feedback_type=FeedbackType.CLICK_THROUGH,
                        feedback_data={'clicked_documents': clicked_docs}
                    )
                
                cycle_interactions += 1
                interaction_count += 1
        
        print(f"{cycle_interactions} interactions")
        
        # Trigger adaptation
        if cycle < 4:  # Don't adapt on last cycle
            await rag_system.trigger_adaptation()
    
    print(f"\nTotal interactions generated: {interaction_count}")
    
    # Get comprehensive analytics
    stats = rag_system.get_system_statistics()
    
    print(f"\nCOMPREHENSIVE SYSTEM ANALYTICS")
    print("=" * 40)
    
    adaptive_stats = stats['adaptive_rag_stats']
    
    print(f"\nSystem Overview:")
    system_stats = adaptive_stats['system_statistics']
    print(f"  Total queries processed: {system_stats['total_queries']}")
    print(f"  Adaptations performed: {system_stats['adaptations_performed']}")
    print(f"  Performance improvements: {system_stats['performance_improvements']}")
    print(f"  Adaptation effectiveness: {system_stats['adaptation_effectiveness']:.1%}")
    print(f"  Overall performance improvement: {adaptive_stats['performance_improvement']:+.1%}")
    
    print(f"\nFeedback Analysis:")
    feedback_stats = adaptive_stats['feedback_statistics']
    print(f"  Total interactions: {feedback_stats['total_interactions']}")
    print(f"  Feedback coverage: {feedback_stats['feedback_coverage']:.1%}")
    print(f"  Unique users: {feedback_stats['unique_users']}")
    
    print(f"  Average feedback by type:")
    for feedback_type, avg_value in feedback_stats['average_feedback_by_type'].items():
        print(f"    {feedback_type}: {avg_value:.2f}")
    
    print(f"\nPerformance Metrics:")
    performance = adaptive_stats['performance_summary']['current_metrics']
    print(f"  Click-through rate: {performance['click_through_rate']:.3f}")
    print(f"  User rating average: {performance['user_rating_average']:.2f}")
    print(f"  Task completion rate: {performance['task_completion_rate']:.3f}")
    print(f"  Average response time: {performance['average_response_time']:.3f}s")
    print(f"  Query success rate: {performance['query_success_rate']:.3f}")
    print(f"  Improvement rate: {performance['improvement_rate']:.3f}")
    
    print(f"\nOptimization Summary:")
    optimization = adaptive_stats['optimization_summary']
    print(f"  Adaptation strategy: {optimization['adaptation_strategy']}")
    print(f"  Optimization iterations: {optimization['optimization_iterations']}")
    print(f"  Best performance achieved: {optimization['best_performance']:.3f}")
    print(f"  Current performance: {optimization['current_performance']:.3f}")
    
    if 'bandit_arm_performance' in optimization:
        print(f"\nBandit Arm Performance:")
        for param_name, arm_performance in optimization['bandit_arm_performance'].items():
            print(f"  {param_name}:")
            for arm_value, performance in arm_performance.items():
                print(f"    {arm_value}: {performance:.3f}")
    
    print(f"\nSystem Capabilities:")
    capabilities = stats['capabilities']
    for capability, enabled in capabilities.items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {capability.replace('_', ' ').title()}")

async def main():
    """
    Demonstrate Adaptive RAG Systems for self-improving retrieval
    
    WHAT YOU'LL LEARN:
    ================
    1. How to collect and analyze user feedback for system improvement
    2. How to monitor performance and detect regressions automatically
    3. How to implement adaptive optimization strategies
    4. How to build systems that learn and improve from experience
    5. How to create personalized and continuously evolving AI systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Customer support systems that improve response quality over time
    - E-commerce search that adapts to user preferences and behavior
    - Educational platforms that optimize content delivery for learning outcomes
    - Enterprise knowledge systems that evolve with organizational needs
    - Recommendation systems that continuously refine their suggestions
    - Medical information systems that improve diagnostic support accuracy
    """
    
    print("ADAPTIVE RAG SYSTEMS DEMONSTRATION")
    print("Building self-improving AI systems that learn and evolve!")
    
    await demo_feedback_collection()
    await demo_performance_monitoring()
    await demo_adaptive_optimization()
    await demo_complete_adaptive_system()
    await demo_system_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Feedback collection enables continuous learning from users")
    print("✓ Performance monitoring detects improvements and regressions")
    print("✓ Adaptive optimization automatically tunes system parameters")
    print("✓ Personalization provides customized experiences for each user")
    print("✓ Complete systems demonstrate measurable improvement over time")
    print("✓ Analytics provide insights into adaptation effectiveness")
    print("\nTHE POWER OF ADAPTIVE RAG:")
    print("- Enables systems that continuously improve from experience")
    print("- Provides personalized and optimized user experiences")
    print("- Reduces manual tuning and maintenance overhead")
    print("- Powers truly intelligent and responsive AI systems")

if __name__ == "__main__":
    asyncio.run(main())
