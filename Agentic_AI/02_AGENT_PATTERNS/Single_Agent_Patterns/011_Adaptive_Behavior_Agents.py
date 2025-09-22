#!/usr/bin/env python3
"""
Adaptive Behavior Agents: Changing Behavior Based on Context
===========================================================

WHAT IS THE PROBLEM?
==================
Most systems use the same approach regardless of the situation, leading to poor performance when conditions change.

Example: GPS Navigation Without Adaptation
- Always uses "fastest route" algorithm
- Doesn't adapt to current traffic conditions
- Doesn't consider time of day, weather, or road conditions
- Doesn't learn from user preferences or past experiences
- Result: Poor routes during rush hour, construction, or bad weather

REAL WORLD EXAMPLE:
=================
How does Google Maps actually work?

ADAPTIVE BEHAVIOR:
- NORMAL CONDITIONS: Prioritize fastest route
- HEAVY TRAFFIC: Switch to alternate routes, avoid highways
- RAINY WEATHER: Avoid steep hills, prefer major roads
- NIGHT TIME: Prioritize well-lit, safer routes
- WORK COMMUTE: Learn user's preferred routes and timing
- WEEKEND: Different behavior - scenic routes, leisure destinations

CONTEXT AWARENESS:
- Time of day (rush hour vs off-peak)
- Weather conditions (rain, snow, heat)
- Traffic patterns (accidents, construction)
- User preferences (avoid tolls, prefer highways)
- Historical data (this route usually has problems)

THE ALGORITHM:
=============
1. SENSE: Continuously monitor environmental context
2. CLASSIFY: Determine current situation type
3. ADAPT: Switch behavior strategy based on context
4. EXECUTE: Apply context-appropriate behavior
5. MONITOR: Track effectiveness of adaptive behavior
6. LEARN: Update adaptation rules based on outcomes

PSEUDO CODE:
===========
class AdaptiveBehaviorAgent:
    def __init__(self):
        self.behavior_strategies = {
            'normal': NormalBehavior(),
            'high_stress': HighStressBehavior(),
            'resource_constrained': EfficientBehavior(),
            'exploration': ExplorativeBehavior()
        }
        self.context_classifier = ContextClassifier()
        self.adaptation_rules = AdaptationRules()
    
    def execute_action(self, input_data):
        # Sense current context
        current_context = self.sense_environment()
        
        # Classify situation
        situation_type = self.context_classifier.classify(current_context)
        
        # Select appropriate behavior strategy
        strategy = self.select_strategy(situation_type)
        
        # Execute with adapted behavior
        result = strategy.execute(input_data, current_context)
        
        # Learn from outcome
        self.update_adaptation_rules(situation_type, strategy, result)
        
        return result

WHY IS THIS POWERFUL?
===================
- Maximizes performance across diverse conditions
- Handles unexpected situations gracefully
- Learns optimal behavior for each context
- Reduces need for manual configuration
- Enables robust operation in dynamic environments
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class ContextType(Enum):
    NORMAL = "normal"
    HIGH_LOAD = "high_load"
    LOW_RESOURCES = "low_resources"
    ERROR_PRONE = "error_prone"
    LEARNING = "learning"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class AdaptationTrigger(Enum):
    PERFORMANCE_DROP = "performance_drop"
    CONTEXT_CHANGE = "context_change"
    ERROR_INCREASE = "error_increase"
    RESOURCE_CONSTRAINT = "resource_constraint"
    USER_FEEDBACK = "user_feedback"
    SCHEDULED = "scheduled"

@dataclass
class EnvironmentContext:
    """Current environment state and conditions"""
    timestamp: float = field(default_factory=time.time)
    system_load: float = 0.5  # 0.0 to 1.0
    available_resources: float = 0.8  # 0.0 to 1.0
    error_rate: float = 0.05  # 0.0 to 1.0
    user_activity: float = 0.6  # 0.0 to 1.0
    external_conditions: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class AdaptationRule:
    """Rule for when and how to adapt behavior"""
    rule_id: str
    trigger_condition: str
    context_requirements: Dict[str, Any]
    behavior_change: str
    confidence: float
    success_rate: float = 0.0
    usage_count: int = 0

class BehaviorStrategy(ABC):
    """Abstract base class for behavior strategies"""
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of this strategy"""
        pass
    
    @abstractmethod
    async def execute(self, task_data: Any, context: EnvironmentContext) -> Dict[str, Any]:
        """Execute task using this strategy"""
        pass
    
    @abstractmethod
    def get_resource_requirements(self) -> Dict[str, float]:
        """Get resource requirements for this strategy"""
        pass
    
    @abstractmethod
    def is_suitable_for_context(self, context: EnvironmentContext) -> float:
        """Return suitability score (0.0 to 1.0) for given context"""
        pass

class AdaptiveBehaviorAgent:
    """
    An agent that adapts its behavior based on environmental context
    
    EXAMPLE USAGE:
    =============
    # Create adaptive agent for content processing
    agent = AdaptiveBehaviorAgent("content_processor")
    
    # Add different behavior strategies
    agent.add_strategy(FastProcessingStrategy())
    agent.add_strategy(QualityFocusedStrategy())
    agent.add_strategy(ResourceEfficientStrategy())
    
    # Agent automatically adapts based on current conditions
    result = await agent.process_task(content_data)
    """
    
    def __init__(self, agent_id: str, adaptation_threshold: float = 0.3):
        self.agent_id = agent_id
        self.adaptation_threshold = adaptation_threshold
        
        # Behavior management
        self.strategies: Dict[str, BehaviorStrategy] = {}
        self.current_strategy: Optional[str] = None
        self.default_strategy: Optional[str] = None
        
        # Context monitoring
        self.context_history: List[EnvironmentContext] = []
        self.context_window_size = 10
        
        # Adaptation rules and learning
        self.adaptation_rules: List[AdaptationRule] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        self.context_performance: Dict[str, Dict[str, float]] = {}
        
        # Sensors for context monitoring
        self.context_sensors: Dict[str, Callable] = {
            "system_monitor": self._monitor_system_performance,
            "resource_monitor": self._monitor_resource_availability,
            "error_monitor": self._monitor_error_rates,
            "user_monitor": self._monitor_user_activity
        }
    
    def add_strategy(self, strategy: BehaviorStrategy) -> None:
        """Add a behavior strategy to the agent"""
        strategy_name = strategy.get_strategy_name()
        self.strategies[strategy_name] = strategy
        
        # Initialize performance tracking
        self.strategy_performance[strategy_name] = {
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "resource_efficiency": 0.0,
            "adaptation_count": 0
        }
        
        # Set as default if first strategy
        if self.default_strategy is None:
            self.default_strategy = strategy_name
            self.current_strategy = strategy_name
        
        print(f"Added behavior strategy: {strategy_name}")
    
    async def process_task(self, task_data: Any, 
                          force_context_update: bool = True) -> Dict[str, Any]:
        """
        Process a task using adaptive behavior
        
        Args:
            task_data: Data for the task to process
            force_context_update: Whether to force context sensing
            
        Returns:
            Task processing results with adaptation information
        """
        print(f"\nPROCESSING TASK with {self.agent_id}")
        print("-" * 40)
        
        # Sense current environment context
        if force_context_update or not self.context_history:
            current_context = await self.sense_environment()
            self.context_history.append(current_context)
            
            # Maintain context window
            if len(self.context_history) > self.context_window_size:
                self.context_history.pop(0)
        else:
            current_context = self.context_history[-1]
        
        print(f"Current context: Load={current_context.system_load:.2f}, "
              f"Resources={current_context.available_resources:.2f}, "
              f"Errors={current_context.error_rate:.3f}")
        
        # Check if adaptation is needed
        adaptation_needed = await self.check_adaptation_triggers(current_context)
        
        if adaptation_needed:
            print("Adaptation triggered - selecting optimal strategy...")
            await self.adapt_behavior(current_context)
        
        # Execute task with current strategy
        if self.current_strategy and self.current_strategy in self.strategies:
            strategy = self.strategies[self.current_strategy]
            print(f"Using strategy: {self.current_strategy}")
            
            start_time = time.time()
            result = await strategy.execute(task_data, current_context)
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self.update_strategy_performance(self.current_strategy, result, execution_time)
            
            # Record adaptation if occurred
            if adaptation_needed:
                self.record_adaptation(current_context, self.current_strategy, result)
            
            return {
                "task_result": result,
                "strategy_used": self.current_strategy,
                "execution_time": execution_time,
                "context": current_context,
                "adaptation_occurred": adaptation_needed,
                "performance_score": result.get("success_score", 0.0)
            }
        
        else:
            return {
                "error": "No suitable strategy available",
                "context": current_context
            }
    
    async def sense_environment(self) -> EnvironmentContext:
        """
        Sense current environment context using all available sensors
        """
        context = EnvironmentContext()
        
        # Run all context sensors
        for sensor_name, sensor_func in self.context_sensors.items():
            try:
                sensor_data = await sensor_func()
                
                # Update context with sensor data
                if sensor_name == "system_monitor":
                    context.system_load = sensor_data.get("load", 0.5)
                    context.performance_metrics.update(sensor_data.get("metrics", {}))
                elif sensor_name == "resource_monitor":
                    context.available_resources = sensor_data.get("availability", 0.8)
                elif sensor_name == "error_monitor":
                    context.error_rate = sensor_data.get("error_rate", 0.05)
                elif sensor_name == "user_monitor":
                    context.user_activity = sensor_data.get("activity", 0.6)
                
                # Add external conditions
                context.external_conditions.update(sensor_data.get("external", {}))
                
            except Exception as e:
                print(f"Error in sensor {sensor_name}: {e}")
        
        return context
    
    async def check_adaptation_triggers(self, context: EnvironmentContext) -> bool:
        """
        Check if any adaptation triggers are activated
        """
        if not self.context_history or len(self.context_history) < 2:
            return False
        
        previous_context = self.context_history[-2]
        
        # Check various adaptation triggers
        triggers_activated = []
        
        # Performance drop trigger
        if (context.performance_metrics.get("success_rate", 1.0) < 
            previous_context.performance_metrics.get("success_rate", 1.0) - self.adaptation_threshold):
            triggers_activated.append(AdaptationTrigger.PERFORMANCE_DROP)
        
        # Context change trigger
        context_change = (
            abs(context.system_load - previous_context.system_load) > 0.3 or
            abs(context.available_resources - previous_context.available_resources) > 0.3 or
            abs(context.error_rate - previous_context.error_rate) > 0.1
        )
        if context_change:
            triggers_activated.append(AdaptationTrigger.CONTEXT_CHANGE)
        
        # Error increase trigger
        if context.error_rate > previous_context.error_rate + 0.05:
            triggers_activated.append(AdaptationTrigger.ERROR_INCREASE)
        
        # Resource constraint trigger
        if context.available_resources < 0.3:
            triggers_activated.append(AdaptationTrigger.RESOURCE_CONSTRAINT)
        
        if triggers_activated:
            print(f"Adaptation triggers activated: {[t.value for t in triggers_activated]}")
            return True
        
        return False
    
    async def adapt_behavior(self, context: EnvironmentContext) -> None:
        """
        Adapt behavior by selecting the most suitable strategy for current context
        """
        # Evaluate all strategies for current context
        strategy_scores = {}
        
        for strategy_name, strategy in self.strategies.items():
            # Get suitability score from strategy
            suitability = strategy.is_suitable_for_context(context)
            
            # Adjust score based on past performance
            performance = self.strategy_performance[strategy_name]
            performance_factor = (performance["success_rate"] * 0.4 + 
                                performance["resource_efficiency"] * 0.3 + 
                                (1 - performance["average_execution_time"] / 10) * 0.3)
            
            # Combined score
            strategy_scores[strategy_name] = suitability * 0.7 + performance_factor * 0.3
        
        # Select best strategy
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            new_strategy = best_strategy[0]
            
            if new_strategy != self.current_strategy:
                old_strategy = self.current_strategy
                self.current_strategy = new_strategy
                
                print(f"Strategy adaptation: {old_strategy} -> {new_strategy} "
                      f"(score: {best_strategy[1]:.2f})")
                
                # Update adaptation count
                self.strategy_performance[new_strategy]["adaptation_count"] += 1
    
    def update_strategy_performance(self, strategy_name: str, result: Dict[str, Any], 
                                  execution_time: float) -> None:
        """Update performance metrics for a strategy"""
        if strategy_name not in self.strategy_performance:
            return
        
        perf = self.strategy_performance[strategy_name]
        
        # Update success rate
        success = result.get("success", True)
        current_rate = perf["success_rate"]
        perf["success_rate"] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        
        # Update execution time
        current_time = perf["average_execution_time"]
        perf["average_execution_time"] = current_time * 0.9 + execution_time * 0.1
        
        # Update resource efficiency (based on result data)
        resource_usage = result.get("resource_usage", 0.5)
        efficiency = 1.0 - resource_usage
        current_efficiency = perf["resource_efficiency"]
        perf["resource_efficiency"] = current_efficiency * 0.9 + efficiency * 0.1
    
    def record_adaptation(self, context: EnvironmentContext, strategy: str, result: Dict[str, Any]) -> None:
        """Record an adaptation event for learning"""
        adaptation_record = {
            "timestamp": time.time(),
            "context": {
                "system_load": context.system_load,
                "available_resources": context.available_resources,
                "error_rate": context.error_rate,
                "user_activity": context.user_activity
            },
            "strategy_selected": strategy,
            "result_success": result.get("success", True),
            "performance_score": result.get("success_score", 0.0)
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Learn new adaptation rules
        self.learn_adaptation_rules()
    
    def learn_adaptation_rules(self) -> None:
        """Learn new adaptation rules from adaptation history"""
        if len(self.adaptation_history) < 5:
            return
        
        # Analyze recent adaptations for patterns
        recent_adaptations = self.adaptation_history[-10:]
        
        # Group by strategy and context conditions
        strategy_contexts = {}
        for adaptation in recent_adaptations:
            strategy = adaptation["strategy_selected"]
            if strategy not in strategy_contexts:
                strategy_contexts[strategy] = []
            strategy_contexts[strategy].append(adaptation)
        
        # Create rules for successful patterns
        for strategy, adaptations in strategy_contexts.items():
            successful_adaptations = [a for a in adaptations if a["performance_score"] > 0.7]
            
            if len(successful_adaptations) >= 3:
                # Calculate average context for successful adaptations
                avg_context = {}
                for key in ["system_load", "available_resources", "error_rate", "user_activity"]:
                    values = [a["context"][key] for a in successful_adaptations]
                    avg_context[key] = sum(values) / len(values)
                
                # Create adaptation rule
                rule = AdaptationRule(
                    rule_id=f"rule_{strategy}_{len(self.adaptation_rules)}",
                    trigger_condition=f"context_match_{strategy}",
                    context_requirements=avg_context,
                    behavior_change=strategy,
                    confidence=len(successful_adaptations) / len(adaptations),
                    success_rate=sum(a["performance_score"] for a in successful_adaptations) / len(successful_adaptations)
                )
                
                self.adaptation_rules.append(rule)
                print(f"Learned new adaptation rule: Use {strategy} when context matches pattern")
    
    # CONTEXT SENSORS
    # ==============
    
    async def _monitor_system_performance(self) -> Dict[str, Any]:
        """Monitor system performance metrics"""
        await asyncio.sleep(0.01)  # Simulate monitoring delay
        
        # Simulate realistic system metrics with some randomness
        base_load = 0.4
        load_variation = random.uniform(-0.2, 0.4)
        system_load = max(0.0, min(1.0, base_load + load_variation))
        
        return {
            "load": system_load,
            "metrics": {
                "cpu_usage": system_load * 0.8,
                "memory_usage": system_load * 0.6,
                "success_rate": max(0.5, 1.0 - system_load * 0.5)
            }
        }
    
    async def _monitor_resource_availability(self) -> Dict[str, Any]:
        """Monitor available system resources"""
        await asyncio.sleep(0.01)
        
        # Simulate resource availability
        base_availability = 0.7
        availability_variation = random.uniform(-0.3, 0.2)
        availability = max(0.1, min(1.0, base_availability + availability_variation))
        
        return {
            "availability": availability,
            "external": {
                "disk_space": availability * 0.9,
                "network_bandwidth": availability * 0.8
            }
        }
    
    async def _monitor_error_rates(self) -> Dict[str, Any]:
        """Monitor system error rates"""
        await asyncio.sleep(0.01)
        
        # Simulate error rate (higher load = more errors)
        base_error_rate = 0.02
        load_factor = random.uniform(0.5, 1.5)
        error_rate = min(0.5, base_error_rate * load_factor)
        
        return {
            "error_rate": error_rate
        }
    
    async def _monitor_user_activity(self) -> Dict[str, Any]:
        """Monitor user activity levels"""
        await asyncio.sleep(0.01)
        
        # Simulate user activity with time-based patterns
        hour = time.localtime().tm_hour
        if 9 <= hour <= 17:  # Business hours
            base_activity = 0.8
        elif 18 <= hour <= 22:  # Evening
            base_activity = 0.6
        else:  # Night/early morning
            base_activity = 0.2
        
        activity_variation = random.uniform(-0.2, 0.2)
        activity = max(0.0, min(1.0, base_activity + activity_variation))
        
        return {
            "activity": activity
        }
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get comprehensive adaptation status and metrics"""
        
        # Calculate adaptation effectiveness
        if self.adaptation_history:
            recent_adaptations = self.adaptation_history[-5:]
            avg_performance = sum(a["performance_score"] for a in recent_adaptations) / len(recent_adaptations)
        else:
            avg_performance = 0.0
        
        return {
            "agent_id": self.agent_id,
            "current_strategy": self.current_strategy,
            "available_strategies": list(self.strategies.keys()),
            "adaptation_rules_learned": len(self.adaptation_rules),
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptation_performance": avg_performance,
            "strategy_performance": self.strategy_performance.copy(),
            "context_window_size": len(self.context_history),
            "current_context": self.context_history[-1] if self.context_history else None
        }

# EXAMPLE BEHAVIOR STRATEGIES
# ===========================

class FastProcessingStrategy(BehaviorStrategy):
    """Strategy optimized for speed"""
    
    def get_strategy_name(self) -> str:
        return "fast_processing"
    
    async def execute(self, task_data: Any, context: EnvironmentContext) -> Dict[str, Any]:
        # Simulate fast but potentially less accurate processing
        await asyncio.sleep(0.05)  # Very fast
        
        # Success rate depends on system load
        success_rate = max(0.6, 1.0 - context.system_load * 0.4)
        success = random.random() < success_rate
        
        return {
            "success": success,
            "result": f"Fast processing result for {task_data}",
            "processing_time": 0.05,
            "resource_usage": 0.3,
            "success_score": success_rate
        }
    
    def get_resource_requirements(self) -> Dict[str, float]:
        return {"cpu": 0.3, "memory": 0.2, "bandwidth": 0.1}
    
    def is_suitable_for_context(self, context: EnvironmentContext) -> float:
        # Good for high load, low resource situations
        suitability = 0.5
        
        if context.system_load > 0.7:
            suitability += 0.3  # Good for high load
        
        if context.available_resources < 0.4:
            suitability += 0.2  # Good for low resources
        
        return min(1.0, suitability)

class QualityProcessingStrategy(BehaviorStrategy):
    """Strategy optimized for quality and accuracy"""
    
    def get_strategy_name(self) -> str:
        return "quality_processing"
    
    async def execute(self, task_data: Any, context: EnvironmentContext) -> Dict[str, Any]:
        # Simulate thorough, high-quality processing
        await asyncio.sleep(0.3)  # Slower but more thorough
        
        # High success rate but requires more resources
        success_rate = min(0.95, 0.8 + context.available_resources * 0.15)
        success = random.random() < success_rate
        
        return {
            "success": success,
            "result": f"High quality processing result for {task_data}",
            "processing_time": 0.3,
            "resource_usage": 0.7,
            "quality_score": 0.9,
            "success_score": success_rate
        }
    
    def get_resource_requirements(self) -> Dict[str, float]:
        return {"cpu": 0.7, "memory": 0.6, "bandwidth": 0.3}
    
    def is_suitable_for_context(self, context: EnvironmentContext) -> float:
        # Good for low load, high resource situations
        suitability = 0.3
        
        if context.system_load < 0.3:
            suitability += 0.4  # Good for low load
        
        if context.available_resources > 0.7:
            suitability += 0.3  # Good for high resources
        
        if context.error_rate < 0.02:
            suitability += 0.2  # Good for stable conditions
        
        return min(1.0, suitability)

class AdaptiveProcessingStrategy(BehaviorStrategy):
    """Strategy that adapts its approach based on context"""
    
    def get_strategy_name(self) -> str:
        return "adaptive_processing"
    
    async def execute(self, task_data: Any, context: EnvironmentContext) -> Dict[str, Any]:
        # Adapt processing approach based on context
        
        if context.system_load > 0.8:
            # High load - use fast approach
            processing_time = 0.08
            resource_usage = 0.4
            quality_factor = 0.7
        elif context.available_resources < 0.3:
            # Low resources - use efficient approach
            processing_time = 0.15
            resource_usage = 0.3
            quality_factor = 0.8
        else:
            # Normal conditions - balanced approach
            processing_time = 0.2
            resource_usage = 0.5
            quality_factor = 0.85
        
        await asyncio.sleep(processing_time)
        
        success_rate = quality_factor * (1 - context.error_rate)
        success = random.random() < success_rate
        
        return {
            "success": success,
            "result": f"Adaptive processing result for {task_data}",
            "processing_time": processing_time,
            "resource_usage": resource_usage,
            "quality_score": quality_factor,
            "success_score": success_rate,
            "adaptation_applied": f"Adapted to load={context.system_load:.2f}, resources={context.available_resources:.2f}"
        }
    
    def get_resource_requirements(self) -> Dict[str, float]:
        return {"cpu": 0.5, "memory": 0.4, "bandwidth": 0.2}
    
    def is_suitable_for_context(self, context: EnvironmentContext) -> float:
        # Generally suitable, especially for varying conditions
        base_suitability = 0.6
        
        # Bonus for variable conditions (where adaptation is valuable)
        if hasattr(context, 'variability_score'):
            base_suitability += context.variability_score * 0.3
        
        return min(1.0, base_suitability)

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_content_processing_adaptation():
    """Demo: Content processing system that adapts to changing conditions"""
    print("\nDEMO 1: CONTENT PROCESSING WITH ADAPTIVE BEHAVIOR")
    print("=" * 60)
    
    # Create adaptive agent
    agent = AdaptiveBehaviorAgent("content_processor", adaptation_threshold=0.2)
    
    # Add different processing strategies
    agent.add_strategy(FastProcessingStrategy())
    agent.add_strategy(QualityProcessingStrategy())
    agent.add_strategy(AdaptiveProcessingStrategy())
    
    # Simulate processing under different conditions
    conditions = [
        ("Normal Load", {}),
        ("High Load", {"high_load": True}),
        ("Low Resources", {"low_resources": True}),
        ("Error Prone", {"high_errors": True}),
        ("Recovery", {})
    ]
    
    content_items = ["document_1", "document_2", "document_3", "document_4", "document_5"]
    
    for i, (condition_name, condition_override) in enumerate(conditions):
        print(f"\n--- {condition_name} Condition ---")
        
        # Override sensors for demo purposes
        if condition_override:
            if "high_load" in condition_override:
                agent.context_sensors["system_monitor"] = lambda: asyncio.create_task(
                    agent._create_demo_context(load=0.9, resources=0.4))
            elif "low_resources" in condition_override:
                agent.context_sensors["system_monitor"] = lambda: asyncio.create_task(
                    agent._create_demo_context(load=0.5, resources=0.2))
            elif "high_errors" in condition_override:
                agent.context_sensors["error_monitor"] = lambda: asyncio.create_task(
                    agent._create_demo_context(errors=0.15))
        else:
            # Reset to normal monitoring
            agent.context_sensors = {
                "system_monitor": agent._monitor_system_performance,
                "resource_monitor": agent._monitor_resource_availability,
                "error_monitor": agent._monitor_error_rates,
                "user_monitor": agent._monitor_user_activity
            }
        
        # Process content item
        result = await agent.process_task(content_items[i])
        
        print(f"Strategy used: {result['strategy_used']}")
        print(f"Success: {result['task_result']['success']}")
        print(f"Processing time: {result['execution_time']:.3f}s")
        if result['adaptation_occurred']:
            print("✓ Behavior adaptation occurred")
    
    # Show final adaptation status
    status = agent.get_adaptation_status()
    print(f"\nADAPTATION SUMMARY:")
    print(f"Total adaptations: {status['total_adaptations']}")
    print(f"Rules learned: {status['adaptation_rules_learned']}")
    print(f"Current strategy: {status['current_strategy']}")

# Helper method for demo
async def _create_demo_context(self, load=None, resources=None, errors=None):
    """Create demo context with specific values"""
    return {
        "load": load if load is not None else 0.5,
        "availability": resources if resources is not None else 0.8,
        "error_rate": errors if errors is not None else 0.05,
        "metrics": {"success_rate": 0.9}
    }

# Monkey patch the demo method
AdaptiveBehaviorAgent._create_demo_context = _create_demo_context

async def demo_dynamic_workload():
    """Demo: System adapting to dynamic workload patterns"""
    print("\nDEMO 2: DYNAMIC WORKLOAD ADAPTATION")
    print("=" * 60)
    
    agent = AdaptiveBehaviorAgent("workload_processor")
    agent.add_strategy(FastProcessingStrategy())
    agent.add_strategy(QualityProcessingStrategy())
    
    # Simulate workload throughout a day
    workload_scenarios = [
        ("Morning Low", 0.3, 0.9),
        ("Peak Hours", 0.9, 0.4),
        ("Lunch Break", 0.4, 0.8),
        ("Afternoon High", 0.8, 0.5),
        ("Evening Wind Down", 0.2, 0.9)
    ]
    
    for scenario_name, load, resources in workload_scenarios:
        print(f"\n--- {scenario_name} ---")
        
        # Override context for demo
        agent.context_sensors["system_monitor"] = lambda l=load: asyncio.create_task(
            agent._create_demo_context(load=l, resources=resources))
        
        # Process multiple tasks in this scenario
        for task_num in range(2):
            result = await agent.process_task(f"task_{scenario_name}_{task_num}")
            
            if task_num == 0:  # Only show details for first task in each scenario
                print(f"Strategy: {result['strategy_used']}")
                print(f"Load: {result['context'].system_load:.2f}, "
                      f"Resources: {result['context'].available_resources:.2f}")
                if result['adaptation_occurred']:
                    print("✓ Adapted to new conditions")
    
    # Show strategy performance comparison
    status = agent.get_adaptation_status()
    print(f"\nSTRATEGY PERFORMANCE COMPARISON:")
    for strategy_name, performance in status['strategy_performance'].items():
        print(f"{strategy_name}:")
        print(f"  Success rate: {performance['success_rate']:.2f}")
        print(f"  Adaptations: {performance['adaptation_count']}")

async def main():
    """
    Demonstrate Adaptive Behavior Agents that change behavior based on context
    
    WHAT YOU'LL LEARN:
    ================
    1. How to monitor environmental context continuously
    2. How to detect when adaptation is needed
    3. How to select optimal behavior strategies for different contexts
    4. How to learn adaptation rules from experience
    5. How adaptive behavior improves system performance
    
    REAL WORLD APPLICATIONS:
    =======================
    - Cloud computing auto-scaling and resource optimization
    - Network routing and traffic management systems
    - Mobile app performance optimization based on device conditions
    - Game AI that adapts to player skill and behavior patterns
    - Smart home systems that adapt to usage patterns and preferences
    - Trading algorithms that adapt to market conditions
    """
    
    print("ADAPTIVE BEHAVIOR AGENTS DEMONSTRATION")
    print("This shows how agents adapt their behavior based on changing environmental conditions!")
    
    await demo_content_processing_adaptation()
    await demo_dynamic_workload()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Context awareness enables optimal behavior selection")
    print("✓ Adaptive strategies outperform fixed approaches in dynamic environments")
    print("✓ Performance monitoring guides effective adaptation decisions")
    print("✓ Learning adaptation rules improves future responsiveness")
    print("✓ Multiple behavior strategies provide flexibility for diverse conditions")
    print("\nTRY IT YOURSELF:")
    print("- Add more sophisticated context sensors and metrics")
    print("- Implement predictive adaptation based on historical patterns")
    print("- Create domain-specific behavior strategies")
    print("- Add multi-objective optimization for competing goals")

if __name__ == "__main__":
    asyncio.run(main())
