#!/usr/bin/env python3
"""
Feedback Loop Agents: Continuous Improvement Through Learning
============================================================

WHAT IS THE PROBLEM?
==================
Most systems produce outputs but never learn from the results, so they keep making the same mistakes forever.

Example: Bad Recommendation System
- Recommends products to customers
- Never checks if customers actually buy or like the products
- Keeps recommending the same poor products
- Customer satisfaction drops, but system doesn't know why
- No improvement over time

REAL WORLD EXAMPLE:
=================
How does Netflix's recommendation system actually work?

FEEDBACK LOOP CYCLE:
1. RECOMMEND: Show movies/shows to users
2. OBSERVE: Track what users actually watch, rate, skip
3. MEASURE: Calculate success metrics (completion rate, ratings)
4. LEARN: Identify what worked and what didn't
5. ADJUST: Update recommendation algorithms
6. REPEAT: Apply improved recommendations

Specific Example:
- Week 1: Recommend action movies to User A
- Observe: User A skips action movies, watches comedies instead
- Learn: User A prefers comedies despite profile suggesting action
- Adjust: Update User A's profile to prefer comedies
- Week 2: Recommend comedies to User A
- Observe: User A watches and rates comedies highly
- Result: Better recommendations, happier user

THE ALGORITHM:
=============
1. EXECUTE: Perform action or make decision
2. COLLECT: Gather feedback on results
3. MEASURE: Quantify success/failure metrics
4. ANALYZE: Identify patterns in feedback
5. LEARN: Extract insights and rules
6. ADAPT: Modify behavior based on learning
7. REPEAT: Apply improved behavior

PSEUDO CODE:
===========
class FeedbackLoopAgent:
    def __init__(self):
        self.behavior_model = initial_model
        self.feedback_history = []
        self.performance_metrics = {}
    
    def execute_action(self, input_data):
        # Use current model to make decision
        action = self.behavior_model.predict(input_data)
        
        # Execute action and get result
        result = perform_action(action)
        
        # Collect feedback (may be delayed)
        feedback = collect_feedback(result, action, input_data)
        self.feedback_history.append(feedback)
        
        # Learn from recent feedback
        if self.should_update_model():
            insights = self.analyze_feedback()
            self.behavior_model = self.update_model(insights)
        
        return result

WHY IS THIS CRUCIAL?
===================
- Enables continuous improvement without manual intervention
- Adapts to changing conditions and user preferences
- Identifies and corrects systematic biases and errors
- Optimizes performance over time through data-driven learning
- Makes systems more responsive and effective
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class FeedbackType(Enum):
    EXPLICIT = "explicit"      # Direct user ratings, reviews
    IMPLICIT = "implicit"      # Behavior signals, usage patterns
    SYSTEM = "system"          # Performance metrics, error rates
    ENVIRONMENTAL = "environmental"  # External conditions, context changes

class LearningMode(Enum):
    IMMEDIATE = "immediate"    # Learn from each feedback immediately
    BATCH = "batch"           # Learn from accumulated feedback
    SCHEDULED = "scheduled"   # Learn at regular intervals

@dataclass
class FeedbackData:
    """Represents feedback on an action or decision"""
    id: str
    action_id: str
    feedback_type: FeedbackType
    value: float  # Normalized feedback value (-1 to 1)
    raw_data: Any
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Importance weight for this feedback

@dataclass
class ActionRecord:
    """Record of an action taken by the agent"""
    id: str
    action_type: str
    input_data: Any
    output_data: Any
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    feedback_received: List[str] = field(default_factory=list)  # Feedback IDs
    success_score: Optional[float] = None

@dataclass
class LearningInsight:
    """Insight learned from feedback analysis"""
    insight_id: str
    description: str
    confidence: float
    supporting_evidence: List[str]  # Feedback IDs that support this insight
    suggested_changes: Dict[str, Any]
    learned_at: float = field(default_factory=time.time)

class FeedbackLoopAgent:
    """
    An agent that continuously improves through feedback loops
    
    EXAMPLE USAGE:
    =============
    # Create recommendation agent with feedback learning
    agent = FeedbackLoopAgent("recommendation_system")
    
    # Agent makes recommendations
    recommendation = agent.make_decision("recommend_movie", user_profile)
    
    # Collect feedback on recommendation
    agent.receive_feedback(recommendation_id, FeedbackType.EXPLICIT, 
                          rating=4.5, user_watched=True)
    
    # Agent learns and improves recommendations over time
    """
    
    def __init__(self, agent_id: str, learning_mode: LearningMode = LearningMode.BATCH,
                 feedback_window_size: int = 100):
        self.agent_id = agent_id
        self.learning_mode = learning_mode
        self.feedback_window_size = feedback_window_size
        
        # Core components
        self.behavior_model: Dict[str, Any] = self._initialize_behavior_model()
        self.action_history: Dict[str, ActionRecord] = {}
        self.feedback_buffer: deque = deque(maxlen=feedback_window_size)
        self.learned_insights: List[LearningInsight] = []
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            "success_rate": [],
            "average_feedback": [],
            "user_satisfaction": [],
            "system_efficiency": []
        }
        
        # Learning configuration
        self.learning_threshold = 10  # Minimum feedback items before learning
        self.confidence_threshold = 0.7  # Minimum confidence for applying insights
        self.adaptation_rate = 0.1  # How quickly to adapt (0.0 to 1.0)
        
        # Feedback analysis tools
        self.feedback_analyzers: Dict[str, Callable] = {
            "pattern_detector": self._detect_feedback_patterns,
            "trend_analyzer": self._analyze_feedback_trends,
            "correlation_finder": self._find_correlations,
            "outlier_detector": self._detect_outliers
        }
    
    def _initialize_behavior_model(self) -> Dict[str, Any]:
        """Initialize the agent's behavior model"""
        return {
            "decision_weights": {
                "user_history": 0.4,
                "content_similarity": 0.3,
                "popularity": 0.2,
                "diversity": 0.1
            },
            "success_patterns": {},
            "failure_patterns": {},
            "user_preferences": {},
            "adaptation_history": []
        }
    
    async def make_decision(self, decision_type: str, input_data: Any, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a decision using current behavior model
        
        Args:
            decision_type: Type of decision to make
            input_data: Input data for the decision
            context: Additional context information
            
        Returns:
            Decision result with action record
        """
        action_id = f"action_{len(self.action_history) + 1}_{int(time.time())}"
        
        print(f"\nMAKING DECISION: {decision_type}")
        print(f"Action ID: {action_id}")
        
        # Use behavior model to make decision
        if decision_type == "recommend_content":
            decision_result = await self._make_recommendation(input_data, context)
        elif decision_type == "classify_content":
            decision_result = await self._classify_content(input_data, context)
        elif decision_type == "prioritize_tasks":
            decision_result = await self._prioritize_tasks(input_data, context)
        else:
            decision_result = await self._make_generic_decision(input_data, context)
        
        # Record the action
        action_record = ActionRecord(
            id=action_id,
            action_type=decision_type,
            input_data=input_data,
            output_data=decision_result,
            context=context or {}
        )
        
        self.action_history[action_id] = action_record
        
        print(f"Decision made: {decision_result}")
        
        return {
            "action_id": action_id,
            "decision": decision_result,
            "model_version": len(self.learned_insights),
            "confidence": self._calculate_decision_confidence(decision_result)
        }
    
    async def receive_feedback(self, action_id: str, feedback_type: FeedbackType,
                             feedback_value: float = None, **feedback_data) -> None:
        """
        Receive feedback on a previous action
        
        Args:
            action_id: ID of the action being evaluated
            feedback_type: Type of feedback
            feedback_value: Normalized feedback value (-1 to 1)
            **feedback_data: Additional feedback data
        """
        if action_id not in self.action_history:
            print(f"Warning: Received feedback for unknown action {action_id}")
            return
        
        # Calculate feedback value if not provided
        if feedback_value is None:
            feedback_value = self._calculate_feedback_value(feedback_type, feedback_data)
        
        # Create feedback record
        feedback_id = f"feedback_{len(self.feedback_buffer) + 1}_{int(time.time())}"
        feedback = FeedbackData(
            id=feedback_id,
            action_id=action_id,
            feedback_type=feedback_type,
            value=feedback_value,
            raw_data=feedback_data,
            context=self.action_history[action_id].context.copy()
        )
        
        # Store feedback
        self.feedback_buffer.append(feedback)
        self.action_history[action_id].feedback_received.append(feedback_id)
        
        print(f"RECEIVED FEEDBACK: {feedback_type.value} = {feedback_value:.2f} for {action_id}")
        
        # Trigger learning based on mode
        if self.learning_mode == LearningMode.IMMEDIATE:
            await self._trigger_learning()
        elif (self.learning_mode == LearningMode.BATCH and 
              len(self.feedback_buffer) >= self.learning_threshold):
            await self._trigger_learning()
    
    def _calculate_feedback_value(self, feedback_type: FeedbackType, 
                                feedback_data: Dict[str, Any]) -> float:
        """Calculate normalized feedback value from raw feedback data"""
        
        if feedback_type == FeedbackType.EXPLICIT:
            # Direct ratings or scores
            if "rating" in feedback_data:
                # Assume rating is 1-5, normalize to -1 to 1
                rating = feedback_data["rating"]
                return (rating - 3.0) / 2.0  # 5->1, 3->0, 1->-1
            elif "satisfaction" in feedback_data:
                return feedback_data["satisfaction"]  # Assume already normalized
            
        elif feedback_type == FeedbackType.IMPLICIT:
            # Behavioral signals
            positive_signals = feedback_data.get("positive_signals", 0)
            negative_signals = feedback_data.get("negative_signals", 0)
            total_signals = positive_signals + negative_signals
            
            if total_signals > 0:
                return (positive_signals - negative_signals) / total_signals
            
        elif feedback_type == FeedbackType.SYSTEM:
            # Performance metrics
            if "success" in feedback_data:
                return 1.0 if feedback_data["success"] else -1.0
            elif "error_rate" in feedback_data:
                return 1.0 - (feedback_data["error_rate"] * 2.0)  # Convert error rate to success
            
        # Default neutral feedback
        return 0.0
    
    async def _trigger_learning(self) -> None:
        """Trigger the learning process to update behavior model"""
        
        if len(self.feedback_buffer) < self.learning_threshold:
            return
        
        print(f"\nTRIGGERING LEARNING (feedback items: {len(self.feedback_buffer)})")
        print("-" * 40)
        
        # Analyze feedback to extract insights
        insights = await self._analyze_feedback()
        
        # Apply high-confidence insights to behavior model
        applied_insights = []
        for insight in insights:
            if insight.confidence >= self.confidence_threshold:
                await self._apply_insight(insight)
                applied_insights.append(insight)
                self.learned_insights.append(insight)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        print(f"LEARNING COMPLETE: Applied {len(applied_insights)} insights")
        for insight in applied_insights:
            print(f"  - {insight.description} (confidence: {insight.confidence:.2f})")
    
    async def _analyze_feedback(self) -> List[LearningInsight]:
        """Analyze feedback to extract learning insights"""
        insights = []
        
        # Run different analysis methods
        for analyzer_name, analyzer_func in self.feedback_analyzers.items():
            try:
                analyzer_insights = await analyzer_func()
                insights.extend(analyzer_insights)
            except Exception as e:
                print(f"Error in {analyzer_name}: {e}")
        
        # Sort insights by confidence
        insights.sort(key=lambda x: x.confidence, reverse=True)
        
        return insights
    
    async def _detect_feedback_patterns(self) -> List[LearningInsight]:
        """Detect patterns in feedback data"""
        insights = []
        
        # Group feedback by action type
        feedback_by_action_type = {}
        for feedback in self.feedback_buffer:
            action = self.action_history[feedback.action_id]
            action_type = action.action_type
            
            if action_type not in feedback_by_action_type:
                feedback_by_action_type[action_type] = []
            feedback_by_action_type[action_type].append(feedback)
        
        # Analyze patterns within each action type
        for action_type, feedbacks in feedback_by_action_type.items():
            if len(feedbacks) >= 5:  # Need minimum samples
                avg_feedback = statistics.mean([f.value for f in feedbacks])
                
                if avg_feedback > 0.3:
                    insight = LearningInsight(
                        insight_id=f"pattern_{action_type}_positive",
                        description=f"Action type '{action_type}' receives consistently positive feedback",
                        confidence=min(0.9, abs(avg_feedback)),
                        supporting_evidence=[f.id for f in feedbacks],
                        suggested_changes={
                            "increase_weight": {action_type: 0.1},
                            "prioritize": action_type
                        }
                    )
                    insights.append(insight)
                
                elif avg_feedback < -0.3:
                    insight = LearningInsight(
                        insight_id=f"pattern_{action_type}_negative",
                        description=f"Action type '{action_type}' receives consistently negative feedback",
                        confidence=min(0.9, abs(avg_feedback)),
                        supporting_evidence=[f.id for f in feedbacks],
                        suggested_changes={
                            "decrease_weight": {action_type: 0.1},
                            "avoid": action_type
                        }
                    )
                    insights.append(insight)
        
        return insights
    
    async def _analyze_feedback_trends(self) -> List[LearningInsight]:
        """Analyze trends in feedback over time"""
        insights = []
        
        if len(self.feedback_buffer) < 10:
            return insights
        
        # Sort feedback by timestamp
        sorted_feedback = sorted(self.feedback_buffer, key=lambda x: x.timestamp)
        
        # Calculate moving average
        window_size = 5
        recent_avg = statistics.mean([f.value for f in sorted_feedback[-window_size:]])
        earlier_avg = statistics.mean([f.value for f in sorted_feedback[-window_size*2:-window_size]])
        
        trend = recent_avg - earlier_avg
        
        if abs(trend) > 0.2:  # Significant trend
            if trend > 0:
                insight = LearningInsight(
                    insight_id="trend_improving",
                    description=f"Feedback trend is improving (change: +{trend:.2f})",
                    confidence=min(0.8, abs(trend)),
                    supporting_evidence=[f.id for f in sorted_feedback[-window_size:]],
                    suggested_changes={"maintain_current_approach": True}
                )
            else:
                insight = LearningInsight(
                    insight_id="trend_declining",
                    description=f"Feedback trend is declining (change: {trend:.2f})",
                    confidence=min(0.8, abs(trend)),
                    supporting_evidence=[f.id for f in sorted_feedback[-window_size:]],
                    suggested_changes={"need_significant_changes": True}
                )
            
            insights.append(insight)
        
        return insights
    
    async def _find_correlations(self) -> List[LearningInsight]:
        """Find correlations between context and feedback"""
        insights = []
        
        # Simple correlation analysis
        context_feedback = {}
        for feedback in self.feedback_buffer:
            for key, value in feedback.context.items():
                if key not in context_feedback:
                    context_feedback[key] = {"values": [], "feedback": []}
                
                context_feedback[key]["values"].append(str(value))
                context_feedback[key]["feedback"].append(feedback.value)
        
        # Look for strong correlations
        for context_key, data in context_feedback.items():
            if len(set(data["values"])) > 1 and len(data["feedback"]) >= 5:
                # Group feedback by context value
                value_groups = {}
                for value, feedback_val in zip(data["values"], data["feedback"]):
                    if value not in value_groups:
                        value_groups[value] = []
                    value_groups[value].append(feedback_val)
                
                # Find best and worst performing values
                avg_by_value = {v: statistics.mean(feedbacks) for v, feedbacks in value_groups.items()}
                best_value = max(avg_by_value.items(), key=lambda x: x[1])
                worst_value = min(avg_by_value.items(), key=lambda x: x[1])
                
                if best_value[1] - worst_value[1] > 0.4:  # Significant difference
                    insight = LearningInsight(
                        insight_id=f"correlation_{context_key}",
                        description=f"Context '{context_key}' strongly correlates with feedback: '{best_value[0]}' performs best, '{worst_value[0]}' performs worst",
                        confidence=0.7,
                        supporting_evidence=[],
                        suggested_changes={
                            "prefer_context": {context_key: best_value[0]},
                            "avoid_context": {context_key: worst_value[0]}
                        }
                    )
                    insights.append(insight)
        
        return insights
    
    async def _detect_outliers(self) -> List[LearningInsight]:
        """Detect outlier feedback that might indicate special cases"""
        insights = []
        
        if len(self.feedback_buffer) < 10:
            return insights
        
        feedback_values = [f.value for f in self.feedback_buffer]
        mean_feedback = statistics.mean(feedback_values)
        std_feedback = statistics.stdev(feedback_values) if len(feedback_values) > 1 else 0
        
        if std_feedback > 0:
            outliers = [f for f in self.feedback_buffer 
                       if abs(f.value - mean_feedback) > 2 * std_feedback]
            
            if len(outliers) > 0:
                avg_outlier_value = statistics.mean([o.value for o in outliers])
                
                insight = LearningInsight(
                    insight_id="outliers_detected",
                    description=f"Detected {len(outliers)} outlier feedback instances (avg: {avg_outlier_value:.2f})",
                    confidence=0.6,
                    supporting_evidence=[o.id for o in outliers],
                    suggested_changes={"investigate_outliers": True}
                )
                insights.append(insight)
        
        return insights
    
    async def _apply_insight(self, insight: LearningInsight) -> None:
        """Apply a learning insight to update the behavior model"""
        
        changes = insight.suggested_changes
        
        # Apply weight adjustments
        if "increase_weight" in changes:
            for key, adjustment in changes["increase_weight"].items():
                if key in self.behavior_model["decision_weights"]:
                    old_weight = self.behavior_model["decision_weights"][key]
                    new_weight = min(1.0, old_weight + adjustment * self.adaptation_rate)
                    self.behavior_model["decision_weights"][key] = new_weight
                    print(f"  Increased weight for '{key}': {old_weight:.2f} -> {new_weight:.2f}")
        
        if "decrease_weight" in changes:
            for key, adjustment in changes["decrease_weight"].items():
                if key in self.behavior_model["decision_weights"]:
                    old_weight = self.behavior_model["decision_weights"][key]
                    new_weight = max(0.0, old_weight - adjustment * self.adaptation_rate)
                    self.behavior_model["decision_weights"][key] = new_weight
                    print(f"  Decreased weight for '{key}': {old_weight:.2f} -> {new_weight:.2f}")
        
        # Record successful patterns
        if "prioritize" in changes:
            pattern = changes["prioritize"]
            if pattern not in self.behavior_model["success_patterns"]:
                self.behavior_model["success_patterns"][pattern] = 0
            self.behavior_model["success_patterns"][pattern] += 1
        
        # Record patterns to avoid
        if "avoid" in changes:
            pattern = changes["avoid"]
            if pattern not in self.behavior_model["failure_patterns"]:
                self.behavior_model["failure_patterns"][pattern] = 0
            self.behavior_model["failure_patterns"][pattern] += 1
        
        # Record adaptation
        self.behavior_model["adaptation_history"].append({
            "insight_id": insight.insight_id,
            "description": insight.description,
            "applied_at": time.time(),
            "changes": changes
        })
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics based on recent feedback"""
        
        if not self.feedback_buffer:
            return
        
        recent_feedback = list(self.feedback_buffer)[-self.learning_threshold:]
        
        # Calculate success rate (feedback > 0)
        positive_feedback = [f for f in recent_feedback if f.value > 0]
        success_rate = len(positive_feedback) / len(recent_feedback)
        self.performance_metrics["success_rate"].append(success_rate)
        
        # Calculate average feedback
        avg_feedback = statistics.mean([f.value for f in recent_feedback])
        self.performance_metrics["average_feedback"].append(avg_feedback)
        
        # Simulate user satisfaction (based on recent feedback trend)
        satisfaction = max(0, min(1, (avg_feedback + 1) / 2))  # Convert -1,1 to 0,1
        self.performance_metrics["user_satisfaction"].append(satisfaction)
        
        # Simulate system efficiency (based on adaptation count)
        efficiency = min(1.0, len(self.learned_insights) * 0.1)
        self.performance_metrics["system_efficiency"].append(efficiency)
    
    # DECISION-MAKING METHODS
    # ======================
    
    async def _make_recommendation(self, user_data: Any, context: Dict) -> Dict[str, Any]:
        """Make content recommendation based on current model"""
        weights = self.behavior_model["decision_weights"]
        
        # Simulate recommendation logic
        recommendation_score = (
            weights["user_history"] * 0.8 +
            weights["content_similarity"] * 0.7 +
            weights["popularity"] * 0.6 +
            weights["diversity"] * 0.5
        )
        
        return {
            "recommended_item": f"Content_{int(recommendation_score * 100)}",
            "confidence": recommendation_score,
            "reasoning": "Based on learned preferences"
        }
    
    async def _classify_content(self, content_data: Any, context: Dict) -> Dict[str, Any]:
        """Classify content based on current model"""
        # Simulate classification
        return {
            "category": "educational",
            "confidence": 0.85,
            "tags": ["technology", "learning"]
        }
    
    async def _prioritize_tasks(self, tasks: Any, context: Dict) -> Dict[str, Any]:
        """Prioritize tasks based on current model"""
        # Simulate task prioritization
        return {
            "priority_order": ["task_1", "task_3", "task_2"],
            "priority_scores": [0.9, 0.7, 0.4]
        }
    
    async def _make_generic_decision(self, input_data: Any, context: Dict) -> Dict[str, Any]:
        """Make generic decision"""
        return {
            "decision": "proceed",
            "confidence": 0.7
        }
    
    def _calculate_decision_confidence(self, decision: Dict[str, Any]) -> float:
        """Calculate confidence in the decision"""
        return decision.get("confidence", 0.5)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status and performance"""
        
        recent_performance = {}
        for metric, values in self.performance_metrics.items():
            if values:
                recent_performance[metric] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "trend": "improving" if len(values) > 1 and values[-1] > values[-2] else "stable"
                }
        
        return {
            "agent_id": self.agent_id,
            "learning_mode": self.learning_mode.value,
            "total_actions": len(self.action_history),
            "total_feedback": len(self.feedback_buffer),
            "insights_learned": len(self.learned_insights),
            "model_adaptations": len(self.behavior_model["adaptation_history"]),
            "current_weights": self.behavior_model["decision_weights"].copy(),
            "performance_metrics": recent_performance,
            "learning_active": len(self.feedback_buffer) >= self.learning_threshold
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_recommendation_system():
    """Demo: Recommendation system that learns from user feedback"""
    print("\nDEMO 1: RECOMMENDATION SYSTEM WITH FEEDBACK LEARNING")
    print("=" * 60)
    
    # Create recommendation agent
    agent = FeedbackLoopAgent("movie_recommender", LearningMode.BATCH, feedback_window_size=20)
    
    # Simulate user interactions over time
    users = ["user_a", "user_b", "user_c"]
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1}: Making Recommendations ---")
        
        # Make recommendations for each user
        recommendations = []
        for user in users:
            result = await agent.make_decision("recommend_content", 
                                              {"user_id": user, "history": ["action", "comedy", "drama"]},
                                              {"time_of_day": "evening", "device": "tv"})
            recommendations.append(result)
            print(f"Recommended for {user}: {result['decision']['recommended_item']}")
        
        # Simulate user feedback (some positive, some negative)
        print(f"\n--- Round {round_num + 1}: Collecting Feedback ---")
        for i, rec in enumerate(recommendations):
            # Simulate different user satisfaction levels
            if round_num == 0:
                # Initial round - mixed feedback
                satisfaction = 0.3 if i == 0 else (0.8 if i == 1 else 0.5)
            elif round_num == 1:
                # Second round - improvement
                satisfaction = 0.6 if i == 0 else (0.9 if i == 1 else 0.7)
            else:
                # Third round - good performance
                satisfaction = 0.8 if i == 0 else (0.95 if i == 1 else 0.85)
            
            await agent.receive_feedback(rec["action_id"], FeedbackType.EXPLICIT,
                                       satisfaction, rating=satisfaction * 5)
            print(f"Feedback for {users[i]}: {satisfaction:.1f}")
        
        # Show learning progress
        if round_num > 0:
            status = agent.get_agent_status()
            print(f"\nLearning Progress:")
            print(f"Insights learned: {status['insights_learned']}")
            print(f"Model adaptations: {status['model_adaptations']}")
            if "user_satisfaction" in status["performance_metrics"]:
                satisfaction = status["performance_metrics"]["user_satisfaction"]["current"]
                print(f"Current satisfaction: {satisfaction:.2f}")
    
    # Final status
    final_status = agent.get_agent_status()
    print(f"\nFINAL RESULTS:")
    print(f"Total insights learned: {final_status['insights_learned']}")
    print(f"Performance trend: {final_status['performance_metrics'].get('user_satisfaction', {}).get('trend', 'unknown')}")

async def demo_task_prioritization():
    """Demo: Task prioritization system that learns from outcomes"""
    print("\nDEMO 2: TASK PRIORITIZATION WITH OUTCOME LEARNING")
    print("=" * 60)
    
    # Create task prioritization agent
    agent = FeedbackLoopAgent("task_prioritizer", LearningMode.IMMEDIATE, feedback_window_size=15)
    
    # Simulate task prioritization scenarios
    tasks = [
        {"id": "urgent_bug", "type": "bug_fix", "priority": "high"},
        {"id": "feature_request", "type": "feature", "priority": "medium"},
        {"id": "documentation", "type": "docs", "priority": "low"},
        {"id": "performance_optimization", "type": "optimization", "priority": "medium"}
    ]
    
    for iteration in range(4):
        print(f"\n--- Iteration {iteration + 1}: Prioritizing Tasks ---")
        
        # Agent prioritizes tasks
        result = await agent.make_decision("prioritize_tasks", tasks,
                                          {"sprint_capacity": 10, "team_size": 3})
        
        print(f"Priority order: {result['decision']['priority_order']}")
        
        # Simulate outcome feedback based on whether prioritization was good
        if iteration < 2:
            # Early iterations - suboptimal prioritization
            outcome_score = 0.4  # Poor outcomes
        else:
            # Later iterations - improved prioritization after learning
            outcome_score = 0.9  # Good outcomes
        
        await agent.receive_feedback(result["action_id"], FeedbackType.SYSTEM,
                                   outcome_score, success=outcome_score > 0.5,
                                   team_satisfaction=outcome_score)
        
        print(f"Outcome feedback: {outcome_score:.1f}")
        
        # Brief delay for demo
        await asyncio.sleep(0.1)
    
    # Show final learning results
    status = agent.get_agent_status()
    print(f"\nTask Prioritization Learning Results:")
    print(f"Insights learned: {status['insights_learned']}")
    print(f"Decision weights adapted: {len(status['current_weights'])} parameters")

async def main():
    """
    Demonstrate Feedback Loop Agents that learn and improve continuously
    
    WHAT YOU'LL LEARN:
    ================
    1. How to collect and process feedback from multiple sources
    2. How to analyze feedback patterns and extract insights
    3. How to adapt behavior models based on learned insights
    4. How to track performance improvements over time
    5. How feedback loops enable continuous optimization
    
    REAL WORLD APPLICATIONS:
    =======================
    - Recommendation systems (Netflix, Amazon, Spotify)
    - Search engines and ranking algorithms
    - Personalization and user experience optimization
    - Automated trading and financial decision systems
    - Customer service chatbots and virtual assistants
    - Content moderation and quality control systems
    """
    
    print("FEEDBACK LOOP AGENTS DEMONSTRATION")
    print("This shows how agents learn and improve from feedback over time!")
    
    await demo_recommendation_system()
    await demo_task_prioritization()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Feedback loops enable continuous improvement without manual intervention")
    print("✓ Pattern analysis in feedback reveals actionable insights")
    print("✓ Adaptive behavior models improve performance over time")
    print("✓ Multiple feedback types provide comprehensive learning signals")
    print("✓ Performance tracking validates the effectiveness of learning")
    print("\nTRY IT YOURSELF:")
    print("- Implement feedback collection for your own applications")
    print("- Add more sophisticated pattern analysis algorithms")
    print("- Create multi-objective optimization with competing feedback")
    print("- Add A/B testing capabilities for controlled learning")

if __name__ == "__main__":
    asyncio.run(main())
