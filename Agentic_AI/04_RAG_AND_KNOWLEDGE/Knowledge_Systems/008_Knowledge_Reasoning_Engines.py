#!/usr/bin/env python3
"""
Knowledge Reasoning Engines: Automated Logical Inference and Deduction
=====================================================================

WHAT IS THE PROBLEM?
==================
Raw knowledge is useless without the ability to reason with it:
- Knowledge graphs contain millions of facts but can't answer implicit questions
- Databases store information but lack logical inference capabilities
- AI systems can retrieve facts but struggle with complex reasoning chains
- Human-like reasoning requires connecting multiple pieces of information
- Traditional systems fail when explicit answers aren't directly stored
- Complex decision-making requires multi-step logical deduction

Example: Medical Diagnosis Reasoning Gap
KNOWLEDGE WITHOUT REASONING (Traditional):
- Database contains: "Patient has fever", "Patient has cough", "Patient has fatigue"
- Database contains: "Pneumonia symptoms include fever, cough, fatigue"
- System can retrieve individual facts but can't conclude "Patient likely has pneumonia"
- Cannot reason about symptom combinations or differential diagnosis
- Fails to connect symptom patterns with potential conditions
- Result: Missed diagnoses, inability to provide intelligent recommendations

REAL WORLD EXAMPLE:
=================
How does IBM Watson perform medical reasoning?

WATSON'S REASONING ENGINE:
1. KNOWLEDGE INGESTION: Absorbs medical literature, case studies, guidelines
2. FACT EXTRACTION: Identifies key medical facts and relationships
3. REASONING RULES: Applies medical reasoning patterns and protocols
4. HYPOTHESIS GENERATION: Creates multiple diagnostic hypotheses
5. EVIDENCE EVALUATION: Weighs supporting and contradicting evidence
6. CONFIDENCE SCORING: Assigns probability scores to each hypothesis
7. EXPLANATION GENERATION: Provides reasoning chain for recommendations

BENEFITS OF KNOWLEDGE REASONING:
- Answers questions not explicitly stored in knowledge base
- Discovers implicit relationships and hidden patterns
- Enables complex decision-making through logical chains
- Provides explainable reasoning paths for transparency
- Handles uncertainty and conflicting information intelligently
- Scales human expertise through automated logical inference

THE REASONING ADVANTAGE:
======================
RETRIEVAL: "What facts do we know?" → Limited to stored information
REASONING: "What can we infer?" → Unlimited intelligent deduction

KNOWLEDGE REASONING COMPONENTS:
=============================
1. RULE-BASED INFERENCE: Apply logical rules to derive new facts
2. PROBABILISTIC REASONING: Handle uncertainty with probability theory
3. TEMPORAL REASONING: Understand time-based relationships and causality
4. SPATIAL REASONING: Reason about location and geometric relationships
5. CAUSAL REASONING: Identify cause-and-effect relationships
6. ABDUCTIVE REASONING: Generate explanations for observed phenomena
7. ANALOGICAL REASONING: Apply knowledge from similar situations

WHY THIS IS REVOLUTIONARY:
========================
- Transforms static knowledge into dynamic intelligence
- Enables AI systems to "think" rather than just "remember"
- Critical for expert systems, decision support, and autonomous agents
- Powers next-generation AI that can reason like humans
- Creates foundation for artificial general intelligence
- Enables complex problem-solving across all domains
"""

import asyncio
import time
import json
import uuid
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import itertools
import heapq
from fractions import Fraction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ReasoningType(Enum):
    """Types of reasoning approaches"""
    DEDUCTIVE = "deductive"           # From general to specific
    INDUCTIVE = "inductive"           # From specific to general
    ABDUCTIVE = "abductive"           # Best explanation
    ANALOGICAL = "analogical"         # Similarity-based
    CAUSAL = "causal"                 # Cause-and-effect
    TEMPORAL = "temporal"             # Time-based
    SPATIAL = "spatial"               # Location-based
    PROBABILISTIC = "probabilistic"   # Uncertainty-based

class InferenceRule(Enum):
    """Types of inference rules"""
    MODUS_PONENS = "modus_ponens"           # If P then Q, P, therefore Q
    MODUS_TOLLENS = "modus_tollens"         # If P then Q, not Q, therefore not P
    SYLLOGISM = "syllogism"                 # All A are B, C is A, therefore C is B
    RESOLUTION = "resolution"               # Logical resolution
    FORWARD_CHAINING = "forward_chaining"   # Data-driven inference
    BACKWARD_CHAINING = "backward_chaining" # Goal-driven inference

class ConfidenceLevel(Enum):
    """Confidence levels for reasoning"""
    CERTAIN = 1.0
    VERY_HIGH = 0.9
    HIGH = 0.8
    MODERATE = 0.6
    LOW = 0.4
    VERY_LOW = 0.2
    UNCERTAIN = 0.1

@dataclass
class Fact:
    """Represents a fact in the knowledge base"""
    
    id: str
    subject: str
    predicate: str
    object: str
    
    # Metadata
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Truth value
    is_true: bool = True
    
    # Temporal information
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Convert to RDF-like triple"""
        return (self.subject, self.predicate, self.object)
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if fact is valid at given timestamp"""
        if self.valid_from and timestamp < self.valid_from:
            return False
        if self.valid_until and timestamp > self.valid_until:
            return False
        return True

@dataclass
class Rule:
    """Represents a reasoning rule"""
    
    id: str
    name: str
    rule_type: InferenceRule
    
    # Rule definition
    premises: List[str]  # Condition patterns
    conclusion: str      # Conclusion pattern
    
    # Rule metadata
    confidence: float = 1.0
    priority: int = 1
    
    # Applicability conditions
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    applications: int = 0
    success_rate: float = 1.0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class Hypothesis:
    """Represents a hypothesis generated during reasoning"""
    
    id: str
    statement: str
    reasoning_type: ReasoningType
    
    # Evidence
    supporting_facts: List[str] = field(default_factory=list)
    contradicting_facts: List[str] = field(default_factory=list)
    
    # Confidence assessment
    confidence: float = 0.5
    confidence_sources: Dict[str, float] = field(default_factory=dict)
    
    # Reasoning chain
    reasoning_steps: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

@dataclass
class ReasoningResult:
    """Represents the result of a reasoning process"""
    
    id: str
    query: str
    reasoning_type: ReasoningType
    
    # Results
    conclusions: List[Hypothesis] = field(default_factory=list)
    new_facts: List[Fact] = field(default_factory=list)
    
    # Process information
    reasoning_steps: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time: float = 0.0
    confidence: float = 0.0
    
    # Explanation
    explanation: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

class KnowledgeBase:
    """Manages facts and rules for reasoning"""
    
    def __init__(self):
        self.facts: Dict[str, Fact] = {}
        self.rules: Dict[str, Rule] = {}
        
        # Indexes for efficient querying
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)
        self.object_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Rule indexes
        self.rule_by_conclusion: Dict[str, Set[str]] = defaultdict(set)
        self.rule_by_premise: Dict[str, Set[str]] = defaultdict(set)
        
        self.logger = logging.getLogger("KnowledgeBase")
    
    def add_fact(self, fact: Fact) -> bool:
        """Add a fact to the knowledge base"""
        
        try:
            self.facts[fact.id] = fact
            
            # Update indexes
            self.subject_index[fact.subject].add(fact.id)
            self.predicate_index[fact.predicate].add(fact.id)
            self.object_index[fact.object].add(fact.id)
            
            self.logger.debug(f"Added fact: {fact.to_triple()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add fact: {e}")
            return False
    
    def add_rule(self, rule: Rule) -> bool:
        """Add a reasoning rule"""
        
        try:
            self.rules[rule.id] = rule
            
            # Update rule indexes
            self.rule_by_conclusion[rule.conclusion].add(rule.id)
            for premise in rule.premises:
                self.rule_by_premise[premise].add(rule.id)
            
            self.logger.debug(f"Added rule: {rule.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add rule: {e}")
            return False
    
    def query_facts(self, subject: str = None, predicate: str = None, 
                   object: str = None, timestamp: datetime = None) -> List[Fact]:
        """Query facts by subject, predicate, or object"""
        
        candidate_ids = None
        
        # Find candidate fact IDs
        if subject:
            candidate_ids = self.subject_index.get(subject, set())
        
        if predicate:
            predicate_ids = self.predicate_index.get(predicate, set())
            if candidate_ids is None:
                candidate_ids = predicate_ids
            else:
                candidate_ids = candidate_ids.intersection(predicate_ids)
        
        if object:
            object_ids = self.object_index.get(object, set())
            if candidate_ids is None:
                candidate_ids = object_ids
            else:
                candidate_ids = candidate_ids.intersection(object_ids)
        
        # If no specific criteria, return all facts
        if candidate_ids is None:
            candidate_ids = set(self.facts.keys())
        
        # Filter by timestamp if provided
        results = []
        for fact_id in candidate_ids:
            fact = self.facts[fact_id]
            if timestamp is None or fact.is_valid_at(timestamp):
                results.append(fact)
        
        return results
    
    def find_applicable_rules(self, available_facts: List[Fact]) -> List[Rule]:
        """Find rules that can be applied to available facts"""
        
        applicable_rules = []
        fact_triples = set(f.to_triple() for f in available_facts)
        
        for rule in self.rules.values():
            if self._can_apply_rule(rule, fact_triples):
                applicable_rules.append(rule)
        
        # Sort by priority
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable_rules
    
    def _can_apply_rule(self, rule: Rule, fact_triples: Set[Tuple[str, str, str]]) -> bool:
        """Check if a rule can be applied to given facts"""
        
        # Simple pattern matching (in practice, use more sophisticated matching)
        for premise in rule.premises:
            if not self._matches_pattern(premise, fact_triples):
                return False
        
        return True
    
    def _matches_pattern(self, pattern: str, fact_triples: Set[Tuple[str, str, str]]) -> bool:
        """Check if pattern matches any fact triple"""
        
        # Simple pattern matching implementation
        # In practice, use more sophisticated pattern matching with variables
        
        if pattern.startswith("?"):
            # Variable pattern - matches any
            return True
        
        # Split pattern into components
        parts = pattern.split(" ")
        if len(parts) != 3:
            return False
        
        pattern_triple = tuple(parts)
        
        # Check for exact match or wildcard match
        for fact_triple in fact_triples:
            if self._triple_matches_pattern(fact_triple, pattern_triple):
                return True
        
        return False
    
    def _triple_matches_pattern(self, fact_triple: Tuple[str, str, str], 
                               pattern_triple: Tuple[str, str, str]) -> bool:
        """Check if fact triple matches pattern triple"""
        
        for fact_part, pattern_part in zip(fact_triple, pattern_triple):
            if pattern_part != "?" and pattern_part != fact_part:
                return False
        
        return True

class DeductiveReasoner:
    """Performs deductive reasoning (general to specific)"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.logger = logging.getLogger("DeductiveReasoner")
    
    async def reason(self, query: str, max_steps: int = 10) -> ReasoningResult:
        """Perform deductive reasoning"""
        
        start_time = time.time()
        result = ReasoningResult(
            id="",
            query=query,
            reasoning_type=ReasoningType.DEDUCTIVE
        )
        
        try:
            # Parse query into goal pattern
            goal_pattern = self._parse_query(query)
            
            # Backward chaining from goal
            conclusions = await self._backward_chain(goal_pattern, max_steps)
            
            # Convert conclusions to hypotheses
            for conclusion in conclusions:
                hypothesis = Hypothesis(
                    id="",
                    statement=conclusion['statement'],
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    confidence=conclusion['confidence'],
                    reasoning_steps=conclusion['steps'],
                    applied_rules=conclusion['rules']
                )
                result.conclusions.append(hypothesis)
            
            # Calculate overall confidence
            if result.conclusions:
                result.confidence = max(h.confidence for h in result.conclusions)
            
            # Generate explanation
            result.explanation = self._generate_explanation(result.conclusions)
            
            result.processing_time = time.time() - start_time
            
            self.logger.debug(f"Deductive reasoning completed in {result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deductive reasoning failed: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    async def _backward_chain(self, goal_pattern: str, max_steps: int) -> List[Dict[str, Any]]:
        """Perform backward chaining to prove goal"""
        
        conclusions = []
        stack = [(goal_pattern, [], [], 1.0)]  # (goal, steps, rules, confidence)
        visited = set()
        
        step_count = 0
        
        while stack and step_count < max_steps:
            current_goal, steps, rules, confidence = stack.pop()
            
            if current_goal in visited:
                continue
            visited.add(current_goal)
            
            step_count += 1
            
            # Check if goal is directly satisfied by facts
            if self._is_goal_satisfied(current_goal):
                conclusions.append({
                    'statement': current_goal,
                    'confidence': confidence,
                    'steps': steps + [f"Goal satisfied: {current_goal}"],
                    'rules': rules
                })
                continue
            
            # Find rules that can prove this goal
            applicable_rules = self._find_rules_for_goal(current_goal)
            
            for rule in applicable_rules:
                # Add premises as new subgoals
                new_confidence = confidence * rule.confidence
                new_steps = steps + [f"Applied rule: {rule.name}"]
                new_rules = rules + [rule.id]
                
                for premise in rule.premises:
                    stack.append((premise, new_steps, new_rules, new_confidence))
        
        return conclusions
    
    def _parse_query(self, query: str) -> str:
        """Parse natural language query into goal pattern"""
        
        # Simple query parsing (in practice, use NLP)
        # Convert "Is X a Y?" to "X is_a Y"
        
        query = query.lower().strip()
        
        if query.startswith("is ") and " a " in query:
            parts = query[3:].split(" a ")
            if len(parts) == 2:
                subject = parts[0].strip()
                object = parts[1].strip().rstrip("?")
                return f"{subject} is_a {object}"
        
        # Default pattern
        return query.replace("?", "").strip()
    
    def _is_goal_satisfied(self, goal_pattern: str) -> bool:
        """Check if goal is satisfied by existing facts"""
        
        # Parse goal pattern
        parts = goal_pattern.split(" ")
        if len(parts) != 3:
            return False
        
        subject, predicate, object = parts
        
        # Query knowledge base
        matching_facts = self.kb.query_facts(
            subject=subject if subject != "?" else None,
            predicate=predicate if predicate != "?" else None,
            object=object if object != "?" else None
        )
        
        return len(matching_facts) > 0
    
    def _find_rules_for_goal(self, goal_pattern: str) -> List[Rule]:
        """Find rules that can prove the goal"""
        
        applicable_rules = []
        
        for rule in self.kb.rules.values():
            if self._rule_can_prove_goal(rule, goal_pattern):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _rule_can_prove_goal(self, rule: Rule, goal_pattern: str) -> bool:
        """Check if rule can prove the goal"""
        
        # Simple pattern matching
        return self._pattern_matches(rule.conclusion, goal_pattern)
    
    def _pattern_matches(self, pattern1: str, pattern2: str) -> bool:
        """Check if two patterns match (with variables)"""
        
        parts1 = pattern1.split(" ")
        parts2 = pattern2.split(" ")
        
        if len(parts1) != len(parts2):
            return False
        
        for p1, p2 in zip(parts1, parts2):
            if p1 != "?" and p2 != "?" and p1 != p2:
                return False
        
        return True
    
    def _generate_explanation(self, conclusions: List[Hypothesis]) -> str:
        """Generate explanation for conclusions"""
        
        if not conclusions:
            return "No conclusions could be reached through deductive reasoning."
        
        best_conclusion = max(conclusions, key=lambda h: h.confidence)
        
        explanation = f"Deductive reasoning concluded: {best_conclusion.statement} "
        explanation += f"(confidence: {best_conclusion.confidence:.3f})\n"
        
        if best_conclusion.reasoning_steps:
            explanation += "Reasoning steps:\n"
            for i, step in enumerate(best_conclusion.reasoning_steps, 1):
                explanation += f"  {i}. {step}\n"
        
        return explanation

class InductiveReasoner:
    """Performs inductive reasoning (specific to general)"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.logger = logging.getLogger("InductiveReasoner")
    
    async def reason(self, observations: List[str], confidence_threshold: float = 0.7) -> ReasoningResult:
        """Perform inductive reasoning from observations"""
        
        start_time = time.time()
        result = ReasoningResult(
            id="",
            query=f"Inductive reasoning from {len(observations)} observations",
            reasoning_type=ReasoningType.INDUCTIVE
        )
        
        try:
            # Analyze patterns in observations
            patterns = self._find_patterns(observations)
            
            # Generate generalizations
            generalizations = self._generate_generalizations(patterns, confidence_threshold)
            
            # Convert to hypotheses
            for generalization in generalizations:
                hypothesis = Hypothesis(
                    id="",
                    statement=generalization['statement'],
                    reasoning_type=ReasoningType.INDUCTIVE,
                    confidence=generalization['confidence'],
                    supporting_facts=generalization['supporting_observations'],
                    reasoning_steps=generalization['reasoning_steps']
                )
                result.conclusions.append(hypothesis)
            
            # Calculate overall confidence
            if result.conclusions:
                result.confidence = max(h.confidence for h in result.conclusions)
            
            # Generate explanation
            result.explanation = self._generate_explanation(result.conclusions, observations)
            
            result.processing_time = time.time() - start_time
            
            self.logger.debug(f"Inductive reasoning completed in {result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inductive reasoning failed: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    def _find_patterns(self, observations: List[str]) -> List[Dict[str, Any]]:
        """Find patterns in observations"""
        
        patterns = []
        
        # Group observations by predicate
        predicate_groups = defaultdict(list)
        
        for obs in observations:
            parts = obs.split(" ")
            if len(parts) >= 3:
                subject, predicate = parts[0], parts[1]
                object = " ".join(parts[2:])
                predicate_groups[predicate].append((subject, object))
        
        # Analyze each predicate group
        for predicate, instances in predicate_groups.items():
            if len(instances) >= 2:  # Need at least 2 instances for pattern
                
                # Find common object patterns
                objects = [obj for _, obj in instances]
                object_counts = defaultdict(int)
                
                for obj in objects:
                    object_counts[obj] += 1
                
                # Create patterns for frequent objects
                for obj, count in object_counts.items():
                    if count >= 2:  # Appears at least twice
                        confidence = count / len(instances)
                        
                        pattern = {
                            'type': 'predicate_object_pattern',
                            'predicate': predicate,
                            'object': obj,
                            'frequency': count,
                            'total_instances': len(instances),
                            'confidence': confidence,
                            'supporting_observations': [
                                f"{subj} {predicate} {obj}" 
                                for subj, o in instances if o == obj
                            ]
                        }
                        patterns.append(pattern)
        
        return patterns
    
    def _generate_generalizations(self, patterns: List[Dict[str, Any]], 
                                confidence_threshold: float) -> List[Dict[str, Any]]:
        """Generate generalizations from patterns"""
        
        generalizations = []
        
        for pattern in patterns:
            if pattern['confidence'] >= confidence_threshold:
                
                if pattern['type'] == 'predicate_object_pattern':
                    # Create generalization rule
                    generalization = {
                        'statement': f"Most entities with predicate '{pattern['predicate']}' have object '{pattern['object']}'",
                        'confidence': pattern['confidence'],
                        'supporting_observations': pattern['supporting_observations'],
                        'reasoning_steps': [
                            f"Observed {pattern['frequency']} instances of '{pattern['predicate']} {pattern['object']}'",
                            f"Out of {pattern['total_instances']} total instances with predicate '{pattern['predicate']}'",
                            f"Pattern frequency: {pattern['confidence']:.3f}",
                            f"Generalization: {pattern['predicate']} → {pattern['object']}"
                        ],
                        'rule_type': 'statistical_generalization'
                    }
                    generalizations.append(generalization)
        
        return generalizations
    
    def _generate_explanation(self, conclusions: List[Hypothesis], observations: List[str]) -> str:
        """Generate explanation for inductive conclusions"""
        
        explanation = f"Inductive reasoning from {len(observations)} observations:\n\n"
        
        if not conclusions:
            explanation += "No significant patterns found to generalize."
            return explanation
        
        explanation += "Discovered patterns:\n"
        for i, conclusion in enumerate(conclusions, 1):
            explanation += f"{i}. {conclusion.statement} (confidence: {conclusion.confidence:.3f})\n"
            
            if conclusion.supporting_facts:
                explanation += f"   Supporting observations: {len(conclusion.supporting_facts)}\n"
        
        return explanation

class AbductiveReasoner:
    """Performs abductive reasoning (inference to best explanation)"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.logger = logging.getLogger("AbductiveReasoner")
    
    async def reason(self, observations: List[str], max_explanations: int = 5) -> ReasoningResult:
        """Perform abductive reasoning to find best explanations"""
        
        start_time = time.time()
        result = ReasoningResult(
            id="",
            query=f"Best explanations for {len(observations)} observations",
            reasoning_type=ReasoningType.ABDUCTIVE
        )
        
        try:
            # Generate possible explanations
            candidate_explanations = self._generate_candidate_explanations(observations)
            
            # Evaluate and rank explanations
            ranked_explanations = self._rank_explanations(candidate_explanations, observations)
            
            # Take top explanations
            top_explanations = ranked_explanations[:max_explanations]
            
            # Convert to hypotheses
            for explanation in top_explanations:
                hypothesis = Hypothesis(
                    id="",
                    statement=explanation['explanation'],
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    confidence=explanation['score'],
                    supporting_facts=explanation['supporting_observations'],
                    reasoning_steps=explanation['reasoning_steps']
                )
                result.conclusions.append(hypothesis)
            
            # Calculate overall confidence
            if result.conclusions:
                result.confidence = result.conclusions[0].confidence  # Best explanation
            
            # Generate explanation
            result.explanation = self._generate_explanation(result.conclusions, observations)
            
            result.processing_time = time.time() - start_time
            
            self.logger.debug(f"Abductive reasoning completed in {result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Abductive reasoning failed: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    def _generate_candidate_explanations(self, observations: List[str]) -> List[Dict[str, Any]]:
        """Generate candidate explanations for observations"""
        
        explanations = []
        
        # Extract entities and relationships from observations
        entities = set()
        relationships = set()
        
        for obs in observations:
            parts = obs.split(" ")
            if len(parts) >= 3:
                subject, predicate = parts[0], parts[1]
                object = " ".join(parts[2:])
                
                entities.add(subject)
                entities.add(object)
                relationships.add((subject, predicate, object))
        
        # Generate explanations based on common causes
        common_cause_explanations = self._find_common_cause_explanations(relationships)
        explanations.extend(common_cause_explanations)
        
        # Generate explanations based on known patterns
        pattern_explanations = self._find_pattern_explanations(relationships)
        explanations.extend(pattern_explanations)
        
        # Generate simple direct explanations
        direct_explanations = self._find_direct_explanations(relationships)
        explanations.extend(direct_explanations)
        
        return explanations
    
    def _find_common_cause_explanations(self, relationships: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find explanations based on common causes"""
        
        explanations = []
        
        # Group relationships by object to find common effects
        effects_by_object = defaultdict(list)
        
        for subject, predicate, object in relationships:
            effects_by_object[object].append((subject, predicate))
        
        # Look for objects that are effects of multiple subjects
        for object, causes in effects_by_object.items():
            if len(causes) >= 2:
                explanation = {
                    'explanation': f"Common cause leading to multiple effects on '{object}'",
                    'type': 'common_cause',
                    'supporting_observations': [
                        f"{subj} {pred} {object}" for subj, pred in causes
                    ],
                    'reasoning_steps': [
                        f"Observed multiple entities affecting '{object}'",
                        f"Entities: {[subj for subj, _ in causes]}",
                        f"This suggests a common underlying cause or pattern"
                    ],
                    'complexity': len(causes),
                    'coverage': len(causes) / len(relationships)
                }
                explanations.append(explanation)
        
        return explanations
    
    def _find_pattern_explanations(self, relationships: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find explanations based on known patterns"""
        
        explanations = []
        
        # Look for transitivity patterns (A→B, B→C implies A→C)
        transitivity_explanations = self._find_transitivity_patterns(relationships)
        explanations.extend(transitivity_explanations)
        
        # Look for hierarchical patterns
        hierarchy_explanations = self._find_hierarchy_patterns(relationships)
        explanations.extend(hierarchy_explanations)
        
        return explanations
    
    def _find_transitivity_patterns(self, relationships: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find transitivity patterns in relationships"""
        
        explanations = []
        
        # Build adjacency for transitivity detection
        adjacency = defaultdict(list)
        
        for subject, predicate, object in relationships:
            if predicate in ['causes', 'leads_to', 'implies', 'follows']:
                adjacency[subject].append(object)
        
        # Find transitive chains
        for start in adjacency:
            for middle in adjacency[start]:
                if middle in adjacency:
                    for end in adjacency[middle]:
                        # Found transitive chain: start → middle → end
                        explanation = {
                            'explanation': f"Transitive chain: {start} → {middle} → {end}",
                            'type': 'transitivity',
                            'supporting_observations': [
                                f"{start} leads_to {middle}",
                                f"{middle} leads_to {end}"
                            ],
                            'reasoning_steps': [
                                f"Observed: {start} leads to {middle}",
                                f"Observed: {middle} leads to {end}",
                                f"By transitivity: {start} indirectly leads to {end}"
                            ],
                            'complexity': 2,
                            'coverage': 2 / len(relationships)
                        }
                        explanations.append(explanation)
        
        return explanations
    
    def _find_hierarchy_patterns(self, relationships: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find hierarchical patterns in relationships"""
        
        explanations = []
        
        # Look for 'is_a' relationships to build hierarchy
        hierarchy = defaultdict(set)
        
        for subject, predicate, object in relationships:
            if predicate == 'is_a':
                hierarchy[object].add(subject)
        
        # Find hierarchical explanations
        for parent, children in hierarchy.items():
            if len(children) >= 2:
                explanation = {
                    'explanation': f"Hierarchical relationship: multiple entities are types of '{parent}'",
                    'type': 'hierarchy',
                    'supporting_observations': [
                        f"{child} is_a {parent}" for child in children
                    ],
                    'reasoning_steps': [
                        f"Observed multiple 'is_a' relationships with '{parent}'",
                        f"Children: {list(children)}",
                        f"This suggests '{parent}' is a general category"
                    ],
                    'complexity': len(children),
                    'coverage': len(children) / len(relationships)
                }
                explanations.append(explanation)
        
        return explanations
    
    def _find_direct_explanations(self, relationships: Set[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Find simple direct explanations"""
        
        explanations = []
        
        # Each relationship is a direct explanation
        for subject, predicate, object in relationships:
            explanation = {
                'explanation': f"Direct relationship: {subject} {predicate} {object}",
                'type': 'direct',
                'supporting_observations': [f"{subject} {predicate} {object}"],
                'reasoning_steps': [
                    f"Directly observed: {subject} {predicate} {object}",
                    "No further explanation needed for direct observation"
                ],
                'complexity': 1,
                'coverage': 1 / len(relationships)
            }
            explanations.append(explanation)
        
        return explanations
    
    def _rank_explanations(self, explanations: List[Dict[str, Any]], 
                          observations: List[str]) -> List[Dict[str, Any]]:
        """Rank explanations by quality metrics"""
        
        # Calculate scores for each explanation
        for explanation in explanations:
            score = self._calculate_explanation_score(explanation, observations)
            explanation['score'] = score
        
        # Sort by score (descending)
        ranked = sorted(explanations, key=lambda e: e['score'], reverse=True)
        
        return ranked
    
    def _calculate_explanation_score(self, explanation: Dict[str, Any], 
                                   observations: List[str]) -> float:
        """Calculate quality score for an explanation"""
        
        # Scoring factors
        coverage = explanation.get('coverage', 0.0)      # How many observations explained
        simplicity = 1.0 / explanation.get('complexity', 1.0)  # Simpler is better
        support = len(explanation.get('supporting_observations', [])) / len(observations)
        
        # Weight factors
        coverage_weight = 0.4
        simplicity_weight = 0.3
        support_weight = 0.3
        
        score = (coverage * coverage_weight + 
                simplicity * simplicity_weight + 
                support * support_weight)
        
        return min(1.0, score)  # Cap at 1.0
    
    def _generate_explanation(self, conclusions: List[Hypothesis], observations: List[str]) -> str:
        """Generate explanation for abductive conclusions"""
        
        explanation = f"Abductive reasoning for {len(observations)} observations:\n\n"
        
        if not conclusions:
            explanation += "No satisfactory explanations found."
            return explanation
        
        explanation += "Best explanations (ranked by quality):\n"
        for i, conclusion in enumerate(conclusions, 1):
            explanation += f"{i}. {conclusion.statement} (score: {conclusion.confidence:.3f})\n"
            
            if conclusion.reasoning_steps:
                explanation += "   Reasoning:\n"
                for step in conclusion.reasoning_steps:
                    explanation += f"     - {step}\n"
            
            explanation += "\n"
        
        return explanation

class ProbabilisticReasoner:
    """Performs probabilistic reasoning with uncertainty"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.logger = logging.getLogger("ProbabilisticReasoner")
    
    async def reason(self, query: str, evidence: List[str] = None) -> ReasoningResult:
        """Perform probabilistic reasoning with uncertainty"""
        
        start_time = time.time()
        result = ReasoningResult(
            id="",
            query=query,
            reasoning_type=ReasoningType.PROBABILISTIC
        )
        
        try:
            if evidence is None:
                evidence = []
            
            # Build probabilistic model
            prob_model = self._build_probabilistic_model()
            
            # Calculate posterior probabilities
            posteriors = self._calculate_posteriors(query, evidence, prob_model)
            
            # Convert to hypotheses
            for outcome, probability in posteriors.items():
                hypothesis = Hypothesis(
                    id="",
                    statement=f"{query} = {outcome}",
                    reasoning_type=ReasoningType.PROBABILISTIC,
                    confidence=probability,
                    supporting_facts=evidence,
                    reasoning_steps=[
                        f"Built probabilistic model from knowledge base",
                        f"Applied evidence: {evidence}",
                        f"Calculated posterior probability: {probability:.3f}"
                    ]
                )
                result.conclusions.append(hypothesis)
            
            # Sort by probability
            result.conclusions.sort(key=lambda h: h.confidence, reverse=True)
            
            # Overall confidence is highest probability
            if result.conclusions:
                result.confidence = result.conclusions[0].confidence
            
            # Generate explanation
            result.explanation = self._generate_explanation(result.conclusions, query, evidence)
            
            result.processing_time = time.time() - start_time
            
            self.logger.debug(f"Probabilistic reasoning completed in {result.processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Probabilistic reasoning failed: {e}")
            result.processing_time = time.time() - start_time
            return result
    
    def _build_probabilistic_model(self) -> Dict[str, Any]:
        """Build probabilistic model from knowledge base"""
        
        model = {
            'variables': set(),
            'conditional_probs': {},
            'prior_probs': {},
            'dependencies': defaultdict(set)
        }
        
        # Extract variables and probabilities from facts
        for fact in self.kb.facts.values():
            model['variables'].add(fact.subject)
            model['variables'].add(fact.object)
            
            # Use fact confidence as probability
            key = f"{fact.subject}_{fact.predicate}_{fact.object}"
            model['prior_probs'][key] = fact.confidence
            
            # Build dependencies (simplified)
            if fact.predicate in ['causes', 'implies', 'leads_to']:
                model['dependencies'][fact.object].add(fact.subject)
        
        return model
    
    def _calculate_posteriors(self, query: str, evidence: List[str], 
                            model: Dict[str, Any]) -> Dict[str, float]:
        """Calculate posterior probabilities using Bayesian inference"""
        
        # Simplified Bayesian calculation
        # In practice, use proper probabilistic inference algorithms
        
        posteriors = {}
        
        # Define possible outcomes for query
        possible_outcomes = ['true', 'false']
        
        for outcome in possible_outcomes:
            # Calculate P(outcome | evidence)
            likelihood = self._calculate_likelihood(outcome, evidence, model)
            prior = self._calculate_prior(outcome, model)
            
            # Simplified Bayes: P(outcome|evidence) ∝ P(evidence|outcome) * P(outcome)
            posterior = likelihood * prior
            posteriors[outcome] = posterior
        
        # Normalize probabilities
        total = sum(posteriors.values())
        if total > 0:
            for outcome in posteriors:
                posteriors[outcome] /= total
        else:
            # Uniform distribution if no information
            for outcome in posteriors:
                posteriors[outcome] = 1.0 / len(posteriors)
        
        return posteriors
    
    def _calculate_likelihood(self, outcome: str, evidence: List[str], 
                            model: Dict[str, Any]) -> float:
        """Calculate likelihood of evidence given outcome"""
        
        if not evidence:
            return 1.0
        
        # Simplified likelihood calculation
        likelihood = 1.0
        
        for evidence_item in evidence:
            # Parse evidence
            parts = evidence_item.split(" ")
            if len(parts) >= 3:
                key = f"{parts[0]}_{parts[1]}_{' '.join(parts[2:])}"
                
                if outcome == 'true':
                    # Evidence supports the outcome
                    prob = model['prior_probs'].get(key, 0.5)
                else:
                    # Evidence contradicts the outcome
                    prob = 1.0 - model['prior_probs'].get(key, 0.5)
                
                likelihood *= prob
        
        return likelihood
    
    def _calculate_prior(self, outcome: str, model: Dict[str, Any]) -> float:
        """Calculate prior probability of outcome"""
        
        # Simplified prior calculation
        if outcome == 'true':
            return 0.5  # Default prior
        else:
            return 0.5
    
    def _generate_explanation(self, conclusions: List[Hypothesis], 
                            query: str, evidence: List[str]) -> str:
        """Generate explanation for probabilistic conclusions"""
        
        explanation = f"Probabilistic reasoning for query: {query}\n"
        
        if evidence:
            explanation += f"Given evidence: {evidence}\n"
        
        explanation += "\nPosterior probabilities:\n"
        
        for conclusion in conclusions:
            explanation += f"  {conclusion.statement}: {conclusion.confidence:.3f}\n"
        
        if conclusions:
            best = conclusions[0]
            explanation += f"\nMost likely outcome: {best.statement} "
            explanation += f"(probability: {best.confidence:.3f})"
        
        return explanation

class KnowledgeReasoningEngine:
    """Complete knowledge reasoning engine with multiple reasoning types"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        
        # Initialize reasoners
        self.deductive_reasoner = DeductiveReasoner(self.knowledge_base)
        self.inductive_reasoner = InductiveReasoner(self.knowledge_base)
        self.abductive_reasoner = AbductiveReasoner(self.knowledge_base)
        self.probabilistic_reasoner = ProbabilisticReasoner(self.knowledge_base)
        
        # Reasoning history
        self.reasoning_history: List[ReasoningResult] = []
        
        # Statistics
        self.stats = {
            'facts_added': 0,
            'rules_added': 0,
            'reasoning_requests': 0,
            'successful_reasoning': 0,
            'total_reasoning_time': 0.0
        }
        
        self.logger = logging.getLogger("KnowledgeReasoningEngine")
    
    async def initialize(self) -> None:
        """Initialize the reasoning engine"""
        
        # Add some basic reasoning rules
        await self._add_basic_rules()
        
        self.logger.info("Knowledge reasoning engine initialized")
    
    async def add_fact(self, subject: str, predicate: str, object: str,
                      confidence: float = 1.0, source: str = "") -> bool:
        """Add a fact to the knowledge base"""
        
        fact = Fact(
            id="",
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source=source
        )
        
        success = self.knowledge_base.add_fact(fact)
        if success:
            self.stats['facts_added'] += 1
        
        return success
    
    async def add_rule(self, name: str, premises: List[str], conclusion: str,
                      rule_type: InferenceRule, confidence: float = 1.0) -> bool:
        """Add a reasoning rule"""
        
        rule = Rule(
            id="",
            name=name,
            rule_type=rule_type,
            premises=premises,
            conclusion=conclusion,
            confidence=confidence
        )
        
        success = self.knowledge_base.add_rule(rule)
        if success:
            self.stats['rules_added'] += 1
        
        return success
    
    async def reason(self, query: str, reasoning_type: ReasoningType,
                    context: Dict[str, Any] = None) -> ReasoningResult:
        """Perform reasoning of specified type"""
        
        self.stats['reasoning_requests'] += 1
        start_time = time.time()
        
        try:
            if context is None:
                context = {}
            
            # Route to appropriate reasoner
            if reasoning_type == ReasoningType.DEDUCTIVE:
                result = await self.deductive_reasoner.reason(query)
                
            elif reasoning_type == ReasoningType.INDUCTIVE:
                observations = context.get('observations', [])
                result = await self.inductive_reasoner.reason(observations)
                
            elif reasoning_type == ReasoningType.ABDUCTIVE:
                observations = context.get('observations', [])
                result = await self.abductive_reasoner.reason(observations)
                
            elif reasoning_type == ReasoningType.PROBABILISTIC:
                evidence = context.get('evidence', [])
                result = await self.probabilistic_reasoner.reason(query, evidence)
                
            else:
                # Default to deductive reasoning
                result = await self.deductive_reasoner.reason(query)
            
            # Store result
            self.reasoning_history.append(result)
            
            processing_time = time.time() - start_time
            self.stats['total_reasoning_time'] += processing_time
            self.stats['successful_reasoning'] += 1
            
            self.logger.debug(f"Reasoning completed: {reasoning_type.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}")
            
            # Return empty result
            result = ReasoningResult(
                id="",
                query=query,
                reasoning_type=reasoning_type,
                explanation=f"Reasoning failed: {e}"
            )
            result.processing_time = time.time() - start_time
            
            return result
    
    async def multi_step_reasoning(self, query: str, max_steps: int = 5) -> ReasoningResult:
        """Perform multi-step reasoning combining different approaches"""
        
        start_time = time.time()
        combined_result = ReasoningResult(
            id="",
            query=query,
            reasoning_type=ReasoningType.DEDUCTIVE,  # Primary type
            explanation="Multi-step reasoning combining multiple approaches"
        )
        
        reasoning_chain = []
        
        # Step 1: Try deductive reasoning first
        step1_result = await self.reason(query, ReasoningType.DEDUCTIVE)
        reasoning_chain.append(f"Step 1 (Deductive): {step1_result.explanation}")
        
        if step1_result.conclusions:
            combined_result.conclusions.extend(step1_result.conclusions)
        
        # Step 2: If no strong conclusions, try abductive reasoning
        if not step1_result.conclusions or step1_result.confidence < 0.7:
            
            # Extract relevant facts as observations
            observations = []
            relevant_facts = self.knowledge_base.query_facts()[:10]  # Limit for demo
            
            for fact in relevant_facts:
                observations.append(f"{fact.subject} {fact.predicate} {fact.object}")
            
            step2_result = await self.reason(
                query, 
                ReasoningType.ABDUCTIVE,
                {'observations': observations}
            )
            reasoning_chain.append(f"Step 2 (Abductive): {step2_result.explanation}")
            
            if step2_result.conclusions:
                combined_result.conclusions.extend(step2_result.conclusions)
        
        # Step 3: Use probabilistic reasoning for uncertainty quantification
        step3_result = await self.reason(query, ReasoningType.PROBABILISTIC)
        reasoning_chain.append(f"Step 3 (Probabilistic): {step3_result.explanation}")
        
        if step3_result.conclusions:
            combined_result.conclusions.extend(step3_result.conclusions)
        
        # Combine and rank all conclusions
        if combined_result.conclusions:
            # Remove duplicates and rank by confidence
            unique_conclusions = {}
            for conclusion in combined_result.conclusions:
                key = conclusion.statement
                if key not in unique_conclusions or conclusion.confidence > unique_conclusions[key].confidence:
                    unique_conclusions[key] = conclusion
            
            combined_result.conclusions = list(unique_conclusions.values())
            combined_result.conclusions.sort(key=lambda h: h.confidence, reverse=True)
            
            # Overall confidence is best conclusion
            combined_result.confidence = combined_result.conclusions[0].confidence
        
        # Build comprehensive explanation
        combined_explanation = "Multi-step reasoning process:\n\n"
        for step in reasoning_chain:
            combined_explanation += step + "\n\n"
        
        combined_explanation += "Final conclusions:\n"
        for i, conclusion in enumerate(combined_result.conclusions[:3], 1):  # Top 3
            combined_explanation += f"{i}. {conclusion.statement} (confidence: {conclusion.confidence:.3f})\n"
        
        combined_result.explanation = combined_explanation
        combined_result.reasoning_chain = reasoning_chain
        combined_result.processing_time = time.time() - start_time
        
        return combined_result
    
    async def _add_basic_rules(self) -> None:
        """Add basic reasoning rules to the knowledge base"""
        
        # Transitivity rule
        await self.add_rule(
            name="Transitivity",
            premises=["? implies ?", "? implies ?"],
            conclusion="? implies ?",
            rule_type=InferenceRule.MODUS_PONENS,
            confidence=0.9
        )
        
        # Inheritance rule
        await self.add_rule(
            name="Inheritance",
            premises=["? is_a ?", "? has_property ?"],
            conclusion="? has_property ?",
            rule_type=InferenceRule.SYLLOGISM,
            confidence=0.8
        )
        
        # Causation rule
        await self.add_rule(
            name="Causation",
            premises=["? causes ?"],
            conclusion="? leads_to ?",
            rule_type=InferenceRule.MODUS_PONENS,
            confidence=0.7
        )
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base contents"""
        
        return {
            'total_facts': len(self.knowledge_base.facts),
            'total_rules': len(self.knowledge_base.rules),
            'reasoning_history': len(self.reasoning_history),
            'statistics': self.stats,
            'recent_reasoning': [
                {
                    'query': r.query,
                    'type': r.reasoning_type.value,
                    'confidence': r.confidence,
                    'conclusions': len(r.conclusions)
                }
                for r in self.reasoning_history[-5:]  # Last 5
            ]
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_deductive_reasoning():
    """Demo: Deductive reasoning from general to specific"""
    print("\nDEMO 1: DEDUCTIVE REASONING")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Add facts about animals
    print("Adding facts to knowledge base:")
    facts = [
        ("mammals", "have_property", "warm_blooded"),
        ("mammals", "have_property", "hair"),
        ("dog", "is_a", "mammal"),
        ("cat", "is_a", "mammal"),
        ("whale", "is_a", "mammal"),
        ("bird", "is_a", "animal"),
        ("sparrow", "is_a", "bird")
    ]
    
    for subject, predicate, object in facts:
        await engine.add_fact(subject, predicate, object)
        print(f"  {subject} {predicate} {object}")
    
    # Perform deductive reasoning
    print(f"\nPerforming deductive reasoning:")
    queries = [
        "Is dog warm_blooded?",
        "Is cat a mammal?",
        "Is whale warm_blooded?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await engine.reason(query, ReasoningType.DEDUCTIVE)
        
        print(f"Confidence: {result.confidence:.3f}")
        if result.conclusions:
            print("Conclusions:")
            for conclusion in result.conclusions:
                print(f"  - {conclusion.statement} (confidence: {conclusion.confidence:.3f})")
        
        print(f"Explanation: {result.explanation}")

async def demo_inductive_reasoning():
    """Demo: Inductive reasoning from specific to general"""
    print("\nDEMO 2: INDUCTIVE REASONING")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Provide specific observations
    observations = [
        "sparrow can fly",
        "eagle can fly",
        "robin can fly", 
        "hawk can fly",
        "sparrow is_a bird",
        "eagle is_a bird",
        "robin is_a bird",
        "hawk is_a bird",
        "penguin is_a bird",
        "penguin cannot fly"  # Exception to test pattern
    ]
    
    print("Observations:")
    for obs in observations:
        print(f"  {obs}")
    
    # Perform inductive reasoning
    print(f"\nPerforming inductive reasoning:")
    result = await engine.reason(
        "What patterns can we infer?",
        ReasoningType.INDUCTIVE,
        {'observations': observations}
    )
    
    print(f"Confidence: {result.confidence:.3f}")
    
    if result.conclusions:
        print("Discovered patterns:")
        for i, conclusion in enumerate(result.conclusions, 1):
            print(f"{i}. {conclusion.statement}")
            print(f"   Confidence: {conclusion.confidence:.3f}")
            print(f"   Supporting evidence: {len(conclusion.supporting_facts)} observations")
    
    print(f"\nExplanation:\n{result.explanation}")

async def demo_abductive_reasoning():
    """Demo: Abductive reasoning to find best explanations"""
    print("\nDEMO 3: ABDUCTIVE REASONING")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Provide mysterious observations that need explanation
    observations = [
        "patient has fever",
        "patient has cough",
        "patient has fatigue",
        "patient has chest_pain",
        "patient visited hospital",
        "doctor prescribed antibiotics",
        "patient symptoms improved"
    ]
    
    print("Mysterious observations:")
    for obs in observations:
        print(f"  {obs}")
    
    # Add medical knowledge
    medical_facts = [
        ("fever", "symptom_of", "infection"),
        ("cough", "symptom_of", "respiratory_illness"),
        ("fatigue", "symptom_of", "illness"),
        ("chest_pain", "symptom_of", "pneumonia"),
        ("antibiotics", "treats", "bacterial_infection"),
        ("pneumonia", "is_a", "bacterial_infection"),
        ("pneumonia", "causes", "fever"),
        ("pneumonia", "causes", "cough"),
        ("pneumonia", "causes", "fatigue")
    ]
    
    print(f"\nAdding medical knowledge:")
    for subject, predicate, object in medical_facts:
        await engine.add_fact(subject, predicate, object)
        print(f"  {subject} {predicate} {object}")
    
    # Perform abductive reasoning
    print(f"\nPerforming abductive reasoning:")
    result = await engine.reason(
        "What explains these symptoms?",
        ReasoningType.ABDUCTIVE,
        {'observations': observations}
    )
    
    print(f"Overall confidence: {result.confidence:.3f}")
    
    if result.conclusions:
        print("\nPossible explanations (ranked by quality):")
        for i, conclusion in enumerate(result.conclusions, 1):
            print(f"{i}. {conclusion.statement}")
            print(f"   Quality score: {conclusion.confidence:.3f}")
            
            if conclusion.reasoning_steps:
                print("   Reasoning:")
                for step in conclusion.reasoning_steps:
                    print(f"     - {step}")
            print()
    
    print(f"Detailed explanation:\n{result.explanation}")

async def demo_probabilistic_reasoning():
    """Demo: Probabilistic reasoning with uncertainty"""
    print("\nDEMO 4: PROBABILISTIC REASONING")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Add probabilistic facts with confidence levels
    print("Adding probabilistic facts:")
    prob_facts = [
        ("rain", "causes", "wet_ground", 0.9),
        ("sprinkler", "causes", "wet_ground", 0.8),
        ("wet_ground", "implies", "slippery", 0.7),
        ("cloudy", "implies", "rain", 0.6),
        ("morning", "implies", "sprinkler_on", 0.4)
    ]
    
    for subject, predicate, object, confidence in prob_facts:
        await engine.add_fact(subject, predicate, object, confidence)
        print(f"  {subject} {predicate} {object} (confidence: {confidence})")
    
    # Provide evidence and query
    evidence = ["wet_ground observed", "cloudy weather"]
    query = "What caused wet_ground?"
    
    print(f"\nEvidence: {evidence}")
    print(f"Query: {query}")
    
    # Perform probabilistic reasoning
    print(f"\nPerforming probabilistic reasoning:")
    result = await engine.reason(
        query,
        ReasoningType.PROBABILISTIC,
        {'evidence': evidence}
    )
    
    print(f"Overall confidence: {result.confidence:.3f}")
    
    if result.conclusions:
        print("\nProbabilistic conclusions:")
        for conclusion in result.conclusions:
            print(f"  {conclusion.statement}: {conclusion.confidence:.3f}")
    
    print(f"\nDetailed explanation:\n{result.explanation}")

async def demo_multi_step_reasoning():
    """Demo: Multi-step reasoning combining approaches"""
    print("\nDEMO 5: MULTI-STEP REASONING")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Build comprehensive knowledge base
    print("Building comprehensive knowledge base:")
    
    # Scientific facts
    science_facts = [
        ("water", "freezes_at", "0_celsius"),
        ("ice", "is_a", "solid_water"),
        ("temperature", "below", "0_celsius"),
        ("water", "becomes", "ice"),
        ("ice", "has_property", "slippery"),
        ("slippery_surface", "causes", "accidents"),
        ("winter", "has_property", "cold_temperature"),
        ("cold_temperature", "causes", "water_freezing")
    ]
    
    for subject, predicate, object in science_facts:
        await engine.add_fact(subject, predicate, object, confidence=0.9)
        print(f"  {subject} {predicate} {object}")
    
    # Query requiring multi-step reasoning
    complex_query = "Why are there more accidents in winter?"
    
    print(f"\nComplex query: {complex_query}")
    print("This requires connecting multiple reasoning steps...")
    
    # Perform multi-step reasoning
    print(f"\nPerforming multi-step reasoning:")
    result = await engine.multi_step_reasoning(complex_query, max_steps=5)
    
    print(f"Final confidence: {result.confidence:.3f}")
    print(f"Processing time: {result.processing_time:.3f}s")
    
    if result.conclusions:
        print("\nFinal conclusions:")
        for i, conclusion in enumerate(result.conclusions[:3], 1):
            print(f"{i}. {conclusion.statement}")
            print(f"   Confidence: {conclusion.confidence:.3f}")
            print(f"   Reasoning type: {conclusion.reasoning_type.value}")
    
    print(f"\nComplete reasoning process:\n{result.explanation}")

async def demo_knowledge_base_analysis():
    """Demo: Analyzing knowledge base contents and reasoning performance"""
    print("\nDEMO 6: KNOWLEDGE BASE ANALYSIS")
    print("=" * 50)
    
    engine = KnowledgeReasoningEngine()
    await engine.initialize()
    
    # Add diverse knowledge
    print("Adding diverse knowledge to the system:")
    
    knowledge_sets = [
        # Biology
        [("human", "is_a", "mammal"), ("mammal", "has_property", "vertebrate")],
        # Physics  
        [("force", "equals", "mass_times_acceleration"), ("gravity", "is_a", "force")],
        # Technology
        [("computer", "has_component", "processor"), ("processor", "executes", "instructions")],
        # Medicine
        [("virus", "causes", "infection"), ("infection", "causes", "symptoms")]
    ]
    
    for knowledge_set in knowledge_sets:
        for subject, predicate, object in knowledge_set:
            await engine.add_fact(subject, predicate, object)
            print(f"  {subject} {predicate} {object}")
    
    # Perform various reasoning tasks
    print(f"\nPerforming multiple reasoning tasks:")
    
    reasoning_tasks = [
        ("Is human a vertebrate?", ReasoningType.DEDUCTIVE),
        ("What causes symptoms?", ReasoningType.ABDUCTIVE),
        ("Will computer execute instructions?", ReasoningType.PROBABILISTIC)
    ]
    
    for query, reasoning_type in reasoning_tasks:
        print(f"\nTask: {query} ({reasoning_type.value})")
        result = await engine.reason(query, reasoning_type)
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Conclusions: {len(result.conclusions)}")
        print(f"  Processing time: {result.processing_time:.3f}s")
    
    # Analyze knowledge base
    print(f"\nKnowledge base analysis:")
    summary = engine.get_knowledge_summary()
    
    print(f"  Total facts: {summary['total_facts']}")
    print(f"  Total rules: {summary['total_rules']}")
    print(f"  Reasoning requests: {summary['statistics']['reasoning_requests']}")
    print(f"  Successful reasoning: {summary['statistics']['successful_reasoning']}")
    print(f"  Total reasoning time: {summary['statistics']['total_reasoning_time']:.3f}s")
    
    if summary['recent_reasoning']:
        print(f"\nRecent reasoning activities:")
        for activity in summary['recent_reasoning']:
            print(f"    {activity['type']}: {activity['query'][:50]}...")
            print(f"      Confidence: {activity['confidence']:.3f}, Conclusions: {activity['conclusions']}")

async def main():
    """
    Demonstrate Knowledge Reasoning Engines for automated logical inference
    
    WHAT YOU'LL LEARN:
    ================
    1. How to implement deductive reasoning (general to specific)
    2. How to perform inductive reasoning (specific to general patterns)
    3. How to use abductive reasoning (inference to best explanation)
    4. How to handle probabilistic reasoning with uncertainty
    5. How to combine multiple reasoning approaches in multi-step processes
    6. How to build and analyze comprehensive knowledge reasoning systems
    
    REAL WORLD APPLICATIONS:
    =======================
    - Medical diagnosis systems reasoning from symptoms to conditions
    - Legal expert systems analyzing cases and precedents
    - Financial analysis systems detecting patterns and risks
    - Scientific discovery systems generating and testing hypotheses
    - Autonomous systems making decisions under uncertainty
    - Educational systems providing explanations and tutoring
    """
    
    print("KNOWLEDGE REASONING ENGINES DEMONSTRATION")
    print("Automated logical inference and intelligent deduction!")
    
    await demo_deductive_reasoning()
    await demo_inductive_reasoning()
    await demo_abductive_reasoning()
    await demo_probabilistic_reasoning()
    await demo_multi_step_reasoning()
    await demo_knowledge_base_analysis()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Deductive reasoning applies general rules to specific cases")
    print("✓ Inductive reasoning discovers patterns from specific observations")
    print("✓ Abductive reasoning finds best explanations for observations")
    print("✓ Probabilistic reasoning handles uncertainty and incomplete information")
    print("✓ Multi-step reasoning combines approaches for complex problems")
    print("✓ Complete systems enable scalable automated logical inference")
    print("\nTHE POWER OF KNOWLEDGE REASONING:")
    print("- Transforms static knowledge into dynamic intelligence")
    print("- Enables AI systems to 'think' rather than just 'remember'")
    print("- Provides explainable reasoning paths for transparency")
    print("- Handles complex decision-making through logical chains")
    print("- Creates foundation for expert systems and artificial intelligence")

if __name__ == "__main__":
    asyncio.run(main())
