#!/usr/bin/env python3
"""
Multi-Step RAG Systems: Complex Reasoning Through Iterative Retrieval
====================================================================

WHAT IS THE PROBLEM?
==================
Complex questions require multiple information retrieval steps:
- Single retrieval can't answer multi-hop questions
- Complex reasoning needs progressive information gathering
- Cross-domain questions require different knowledge sources
- Verification and fact-checking need multiple evidence sources
- Strategic analysis requires iterative refinement

Example: Investment Analysis Complexity
SINGLE-STEP RAG (Insufficient):
- Question: "Should I invest in renewable energy companies in emerging markets considering current geopolitical tensions and climate policies?"
- Single retrieval gets: Generic renewable energy information
- Misses: Geopolitical analysis, emerging market dynamics, policy impacts
- Result: Shallow, incomplete investment advice

REAL WORLD EXAMPLE:
=================
How does McKinsey analyze complex business questions?

MCKINSEY'S MULTI-STEP ANALYSIS:
For market entry strategy:
1. MARKET SIZING: Retrieve market data, demographics, economic indicators
2. COMPETITIVE LANDSCAPE: Get competitor analysis, market share data
3. REGULATORY ENVIRONMENT: Research policies, compliance requirements  
4. CUSTOMER INSIGHTS: Gather consumer behavior, preferences data
5. FINANCIAL MODELING: Combine all data for projections
6. RISK ASSESSMENT: Analyze potential challenges and mitigation
7. SYNTHESIS: Integrate findings into strategic recommendations

BENEFITS:
- Comprehensive coverage of all relevant factors
- Progressive refinement of understanding
- Evidence-based decision making
- Risk mitigation through thorough analysis
- Professional-grade strategic insights

THE MULTI-STEP PROCESS:
=====================
1. DECOMPOSITION: Break complex questions into sub-questions
2. PLANNING: Design retrieval strategy and sequence
3. ITERATIVE RETRIEVAL: Gather information step by step
4. PROGRESSIVE SYNTHESIS: Build understanding incrementally
5. VALIDATION: Cross-check facts across multiple sources
6. INTEGRATION: Combine insights into comprehensive answer
7. REFINEMENT: Iterate based on gaps or contradictions

STEP TYPES:
- Factual Gathering: Collect basic facts and data
- Contextual Analysis: Understand broader context
- Comparative Research: Compare options or alternatives  
- Causal Investigation: Understand cause-effect relationships
- Temporal Analysis: Historical trends and future projections
- Validation Steps: Verify information across sources
- Synthesis Steps: Combine and integrate findings

WHY THIS IS REVOLUTIONARY:
========================
- Enables AI to handle human-expert level analysis
- Supports complex reasoning and decision-making
- Provides comprehensive coverage of complex topics
- Enables transparent, auditable reasoning chains
- Powers next-generation analytical AI systems
- Critical for professional and enterprise applications
"""

import asyncio
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class StepType(Enum):
    """Types of retrieval steps"""
    FACTUAL_GATHERING = "factual_gathering"       # Collect basic facts
    CONTEXTUAL_ANALYSIS = "contextual_analysis"   # Understand context
    COMPARATIVE_RESEARCH = "comparative_research" # Compare alternatives
    CAUSAL_INVESTIGATION = "causal_investigation" # Cause-effect analysis
    TEMPORAL_ANALYSIS = "temporal_analysis"       # Time-based analysis
    VALIDATION = "validation"                     # Verify information
    SYNTHESIS = "synthesis"                       # Combine insights
    REFINEMENT = "refinement"                     # Fill gaps, resolve conflicts

class StepStatus(Enum):
    """Status of execution steps"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ReasoningType(Enum):
    """Types of reasoning chains"""
    LINEAR = "linear"                 # Sequential step-by-step
    BRANCHING = "branching"          # Multiple parallel paths
    ITERATIVE = "iterative"          # Refining with feedback
    HIERARCHICAL = "hierarchical"    # Top-down decomposition
    EXPLORATORY = "exploratory"      # Discovery-driven

@dataclass
class RetrievalStep:
    """Single step in multi-step retrieval process"""
    step_id: str
    step_type: StepType
    step_name: str
    query: str
    
    # Step configuration
    priority: int = 1
    depends_on: List[str] = field(default_factory=list)
    context_from_steps: List[str] = field(default_factory=list)
    
    # Execution state
    status: StepStatus = StepStatus.PLANNED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_time: float = 0.0
    
    # Results
    retrieved_documents: List[Any] = field(default_factory=list)
    extracted_facts: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize step"""
        if not self.step_id:
            self.step_id = str(uuid.uuid4())
    
    def start_execution(self) -> None:
        """Mark step as started"""
        self.status = StepStatus.IN_PROGRESS
        self.start_time = time.time()
    
    def complete_execution(self, success: bool = True) -> None:
        """Mark step as completed"""
        self.status = StepStatus.COMPLETED if success else StepStatus.FAILED
        self.end_time = time.time()
        if self.start_time:
            self.execution_time = self.end_time - self.start_time
    
    def can_execute(self, completed_steps: Set[str]) -> bool:
        """Check if step dependencies are satisfied"""
        return all(dep_id in completed_steps for dep_id in self.depends_on)
    
    def add_insight(self, insight: str) -> None:
        """Add insight from step execution"""
        if insight not in self.insights:
            self.insights.append(insight)
    
    def add_fact(self, fact: str) -> None:
        """Add extracted fact"""
        if fact not in self.extracted_facts:
            self.extracted_facts.append(fact)

@dataclass
class ReasoningChain:
    """Chain of reasoning steps for complex question"""
    chain_id: str
    original_question: str
    reasoning_type: ReasoningType
    
    # Steps and execution
    steps: List[RetrievalStep] = field(default_factory=list)
    step_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution state
    current_step_index: int = 0
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    
    # Results
    final_answer: str = ""
    confidence_score: float = 0.0
    total_execution_time: float = 0.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        """Initialize reasoning chain"""
        if not self.chain_id:
            self.chain_id = str(uuid.uuid4())
    
    def add_step(self, step: RetrievalStep) -> None:
        """Add step to reasoning chain"""
        self.steps.append(step)
        self.step_dependencies[step.step_id] = step.depends_on.copy()
    
    def get_next_executable_steps(self) -> List[RetrievalStep]:
        """Get steps that can be executed now"""
        executable = []
        
        for step in self.steps:
            if (step.status == StepStatus.PLANNED and 
                step.can_execute(self.completed_steps)):
                executable.append(step)
        
        return executable
    
    def mark_step_completed(self, step_id: str) -> None:
        """Mark step as completed"""
        self.completed_steps.add(step_id)
    
    def mark_step_failed(self, step_id: str) -> None:
        """Mark step as failed"""
        self.failed_steps.add(step_id)
    
    def is_complete(self) -> bool:
        """Check if reasoning chain is complete"""
        return len(self.completed_steps) == len(self.steps)
    
    def has_failures(self) -> bool:
        """Check if any steps failed"""
        return len(self.failed_steps) > 0
    
    def get_step_by_id(self, step_id: str) -> Optional[RetrievalStep]:
        """Get step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

class QuestionDecomposer:
    """Decomposes complex questions into multi-step reasoning chains"""
    
    def __init__(self):
        # Pattern matching for different question types
        self.investment_patterns = [
            r'invest(?:ment)?\s+(?:in|opportunity)',
            r'should\s+(?:i|we)\s+(?:buy|invest)',
            r'(?:stock|market|financial)\s+analysis'
        ]
        
        self.comparison_patterns = [
            r'(?:compare|vs|versus|better)',
            r'(?:difference|similar)',
            r'(?:pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)'
        ]
        
        self.causal_patterns = [
            r'(?:why|what\s+causes?|reason)',
            r'(?:impact|effect|consequence)',
            r'(?:leads?\s+to|results?\s+in)'
        ]
        
        self.temporal_patterns = [
            r'(?:trend|historical|over\s+time)',
            r'(?:future|prediction|forecast)',
            r'(?:past|previous|recent)'
        ]
    
    async def decompose_question(self, question: str) -> ReasoningChain:
        """Decompose complex question into reasoning steps"""
        
        # Analyze question type and complexity
        question_type = self._analyze_question_type(question)
        entities = self._extract_entities(question)
        complexity = self._assess_complexity(question)
        
        # Create reasoning chain based on question type
        if self._matches_patterns(question, self.investment_patterns):
            return await self._create_investment_analysis_chain(question, entities)
        
        elif self._matches_patterns(question, self.comparison_patterns):
            return await self._create_comparison_chain(question, entities)
        
        elif self._matches_patterns(question, self.causal_patterns):
            return await self._create_causal_analysis_chain(question, entities)
        
        elif self._matches_patterns(question, self.temporal_patterns):
            return await self._create_temporal_analysis_chain(question, entities)
        
        else:
            return await self._create_general_analysis_chain(question, entities)
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['invest', 'buy', 'stock', 'market']):
            return 'investment'
        elif any(word in question_lower for word in ['compare', 'vs', 'versus', 'better']):
            return 'comparison'
        elif any(word in question_lower for word in ['why', 'cause', 'reason', 'impact']):
            return 'causal'
        elif any(word in question_lower for word in ['trend', 'historical', 'future', 'forecast']):
            return 'temporal'
        else:
            return 'general'
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from question"""
        # Simple entity extraction (capitalize words, companies, etc.)
        words = question.split()
        entities = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and 
                clean_word[0].isupper() and 
                len(clean_word) > 2 and
                clean_word.lower() not in ['what', 'when', 'where', 'how', 'why']):
                entities.append(clean_word)
        
        return entities
    
    def _assess_complexity(self, question: str) -> float:
        """Assess question complexity score"""
        words = question.split()
        complexity = 0.0
        
        # Length factor
        complexity += min(len(words) / 30.0, 0.3)
        
        # Question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        complexity += sum(0.1 for word in words if word.lower() in question_words)
        
        # Conjunctions (indicate multi-part questions)
        conjunctions = ['and', 'or', 'but', 'considering', 'given', 'while']
        complexity += sum(0.2 for word in words if word.lower() in conjunctions)
        
        return min(complexity, 1.0)
    
    def _matches_patterns(self, question: str, patterns: List[str]) -> bool:
        """Check if question matches any pattern"""
        question_lower = question.lower()
        return any(re.search(pattern, question_lower) for pattern in patterns)
    
    async def _create_investment_analysis_chain(self, question: str, 
                                              entities: List[str]) -> ReasoningChain:
        """Create reasoning chain for investment analysis"""
        
        chain = ReasoningChain(
            chain_id="",
            original_question=question,
            reasoning_type=ReasoningType.HIERARCHICAL
        )
        
        # Step 1: Market Research
        step1 = RetrievalStep(
            step_id="market_research",
            step_type=StepType.FACTUAL_GATHERING,
            step_name="Market Research",
            query=f"market analysis data for {' '.join(entities)} industry trends",
            priority=1
        )
        chain.add_step(step1)
        
        # Step 2: Financial Analysis
        step2 = RetrievalStep(
            step_id="financial_analysis",
            step_type=StepType.FACTUAL_GATHERING,
            step_name="Financial Analysis",
            query=f"financial performance metrics revenue profit {' '.join(entities)}",
            priority=1,
            depends_on=[step1.step_id]
        )
        chain.add_step(step2)
        
        # Step 3: Risk Assessment
        step3 = RetrievalStep(
            step_id="risk_assessment",
            step_type=StepType.CAUSAL_INVESTIGATION,
            step_name="Risk Assessment",
            query=f"investment risks market volatility {' '.join(entities)} sector risks",
            priority=2,
            depends_on=[step1.step_id, step2.step_id]
        )
        chain.add_step(step3)
        
        # Step 4: Competitive Analysis
        step4 = RetrievalStep(
            step_id="competitive_analysis",
            step_type=StepType.COMPARATIVE_RESEARCH,
            step_name="Competitive Analysis",
            query=f"competitors comparison market share {' '.join(entities)}",
            priority=2,
            depends_on=[step1.step_id]
        )
        chain.add_step(step4)
        
        # Step 5: Future Outlook
        step5 = RetrievalStep(
            step_id="future_outlook",
            step_type=StepType.TEMPORAL_ANALYSIS,
            step_name="Future Outlook",
            query=f"future projections growth forecast {' '.join(entities)} industry outlook",
            priority=3,
            depends_on=[step2.step_id, step4.step_id]
        )
        chain.add_step(step5)
        
        # Step 6: Investment Recommendation
        step6 = RetrievalStep(
            step_id="investment_recommendation",
            step_type=StepType.SYNTHESIS,
            step_name="Investment Recommendation",
            query=f"investment recommendation synthesis {' '.join(entities)}",
            priority=4,
            depends_on=[step3.step_id, step5.step_id],
            context_from_steps=[step1.step_id, step2.step_id, step3.step_id, step4.step_id, step5.step_id]
        )
        chain.add_step(step6)
        
        return chain
    
    async def _create_comparison_chain(self, question: str, 
                                     entities: List[str]) -> ReasoningChain:
        """Create reasoning chain for comparison analysis"""
        
        chain = ReasoningChain(
            chain_id="",
            original_question=question,
            reasoning_type=ReasoningType.BRANCHING
        )
        
        # Parallel analysis of each entity
        for i, entity in enumerate(entities[:3]):  # Limit to 3 entities
            # Factual analysis for each entity
            step = RetrievalStep(
                step_id=f"analysis_{entity.lower()}",
                step_type=StepType.FACTUAL_GATHERING,
                step_name=f"Analysis of {entity}",
                query=f"detailed information about {entity} features characteristics",
                priority=1
            )
            chain.add_step(step)
        
        # Comparative synthesis
        comparison_step = RetrievalStep(
            step_id="comparison_synthesis",
            step_type=StepType.COMPARATIVE_RESEARCH,
            step_name="Comparison Synthesis",
            query=f"comparison analysis {' '.join(entities)} differences similarities",
            priority=2,
            depends_on=[f"analysis_{entity.lower()}" for entity in entities[:3]],
            context_from_steps=[f"analysis_{entity.lower()}" for entity in entities[:3]]
        )
        chain.add_step(comparison_step)
        
        return chain
    
    async def _create_causal_analysis_chain(self, question: str, 
                                          entities: List[str]) -> ReasoningChain:
        """Create reasoning chain for causal analysis"""
        
        chain = ReasoningChain(
            chain_id="",
            original_question=question,
            reasoning_type=ReasoningType.LINEAR
        )
        
        # Step 1: Context Gathering
        step1 = RetrievalStep(
            step_id="context_gathering",
            step_type=StepType.CONTEXTUAL_ANALYSIS,
            step_name="Context Gathering",
            query=f"background context {' '.join(entities)} relevant factors",
            priority=1
        )
        chain.add_step(step1)
        
        # Step 2: Causal Factors
        step2 = RetrievalStep(
            step_id="causal_factors",
            step_type=StepType.CAUSAL_INVESTIGATION,
            step_name="Causal Factors Analysis",
            query=f"causes factors leading to {' '.join(entities)} reasons",
            priority=2,
            depends_on=[step1.step_id]
        )
        chain.add_step(step2)
        
        # Step 3: Effect Analysis
        step3 = RetrievalStep(
            step_id="effect_analysis",
            step_type=StepType.CAUSAL_INVESTIGATION,
            step_name="Effect Analysis",
            query=f"effects consequences impact of {' '.join(entities)}",
            priority=2,
            depends_on=[step1.step_id]
        )
        chain.add_step(step3)
        
        # Step 4: Causal Chain Synthesis
        step4 = RetrievalStep(
            step_id="causal_synthesis",
            step_type=StepType.SYNTHESIS,
            step_name="Causal Chain Synthesis",
            query=f"causal relationship chain {' '.join(entities)}",
            priority=3,
            depends_on=[step2.step_id, step3.step_id],
            context_from_steps=[step1.step_id, step2.step_id, step3.step_id]
        )
        chain.add_step(step4)
        
        return chain
    
    async def _create_temporal_analysis_chain(self, question: str, 
                                            entities: List[str]) -> ReasoningChain:
        """Create reasoning chain for temporal analysis"""
        
        chain = ReasoningChain(
            chain_id="",
            original_question=question,
            reasoning_type=ReasoningType.LINEAR
        )
        
        # Step 1: Historical Analysis
        step1 = RetrievalStep(
            step_id="historical_analysis",
            step_type=StepType.TEMPORAL_ANALYSIS,
            step_name="Historical Analysis",
            query=f"historical data trends {' '.join(entities)} past performance",
            priority=1
        )
        chain.add_step(step1)
        
        # Step 2: Current State
        step2 = RetrievalStep(
            step_id="current_state",
            step_type=StepType.FACTUAL_GATHERING,
            step_name="Current State Analysis",
            query=f"current status recent developments {' '.join(entities)}",
            priority=2,
            depends_on=[step1.step_id]
        )
        chain.add_step(step2)
        
        # Step 3: Trend Analysis
        step3 = RetrievalStep(
            step_id="trend_analysis",
            step_type=StepType.TEMPORAL_ANALYSIS,
            step_name="Trend Analysis",
            query=f"trend patterns analysis {' '.join(entities)} direction",
            priority=3,
            depends_on=[step1.step_id, step2.step_id]
        )
        chain.add_step(step3)
        
        # Step 4: Future Projections
        step4 = RetrievalStep(
            step_id="future_projections",
            step_type=StepType.TEMPORAL_ANALYSIS,
            step_name="Future Projections",
            query=f"future forecast predictions {' '.join(entities)} outlook",
            priority=4,
            depends_on=[step3.step_id],
            context_from_steps=[step1.step_id, step2.step_id, step3.step_id]
        )
        chain.add_step(step4)
        
        return chain
    
    async def _create_general_analysis_chain(self, question: str, 
                                           entities: List[str]) -> ReasoningChain:
        """Create general analysis chain for complex questions"""
        
        chain = ReasoningChain(
            chain_id="",
            original_question=question,
            reasoning_type=ReasoningType.ITERATIVE
        )
        
        # Step 1: Information Gathering
        step1 = RetrievalStep(
            step_id="information_gathering",
            step_type=StepType.FACTUAL_GATHERING,
            step_name="Information Gathering",
            query=f"comprehensive information about {' '.join(entities)}",
            priority=1
        )
        chain.add_step(step1)
        
        # Step 2: Context Analysis
        step2 = RetrievalStep(
            step_id="context_analysis",
            step_type=StepType.CONTEXTUAL_ANALYSIS,
            step_name="Context Analysis",
            query=f"context background {' '.join(entities)} related factors",
            priority=2,
            depends_on=[step1.step_id]
        )
        chain.add_step(step2)
        
        # Step 3: Synthesis
        step3 = RetrievalStep(
            step_id="synthesis",
            step_type=StepType.SYNTHESIS,
            step_name="Information Synthesis",
            query=f"synthesis analysis {' '.join(entities)} comprehensive overview",
            priority=3,
            depends_on=[step1.step_id, step2.step_id],
            context_from_steps=[step1.step_id, step2.step_id]
        )
        chain.add_step(step3)
        
        return chain

class MultiStepExecutor:
    """Executes multi-step reasoning chains"""
    
    def __init__(self, retriever, max_concurrent_steps: int = 3):
        self.retriever = retriever
        self.max_concurrent_steps = max_concurrent_steps
        
        # Execution state
        self.active_chains: Dict[str, ReasoningChain] = {}
        self.execution_stats = {
            'total_chains_executed': 0,
            'total_steps_executed': 0,
            'average_chain_time': 0.0,
            'average_step_time': 0.0,
            'success_rate': 0.0
        }
        
        self.logger = logging.getLogger("MultiStepExecutor")
    
    async def execute_reasoning_chain(self, chain: ReasoningChain) -> Dict[str, Any]:
        """Execute complete reasoning chain"""
        
        self.active_chains[chain.chain_id] = chain
        
        start_time = time.time()
        self.logger.info(f"Starting execution of reasoning chain: {chain.chain_id}")
        
        try:
            # Execute steps based on reasoning type
            if chain.reasoning_type == ReasoningType.LINEAR:
                await self._execute_linear_chain(chain)
            elif chain.reasoning_type == ReasoningType.BRANCHING:
                await self._execute_branching_chain(chain)
            elif chain.reasoning_type == ReasoningType.HIERARCHICAL:
                await self._execute_hierarchical_chain(chain)
            elif chain.reasoning_type == ReasoningType.ITERATIVE:
                await self._execute_iterative_chain(chain)
            else:
                await self._execute_sequential_chain(chain)
            
            # Generate final answer
            final_answer = await self._synthesize_final_answer(chain)
            chain.final_answer = final_answer
            
            # Calculate confidence
            chain.confidence_score = self._calculate_chain_confidence(chain)
            
            chain.completed_at = time.time()
            chain.total_execution_time = chain.completed_at - start_time
            
            # Update statistics
            self._update_execution_stats(chain)
            
            success = chain.is_complete() and not chain.has_failures()
            
            result = {
                'chain_id': chain.chain_id,
                'success': success,
                'final_answer': chain.final_answer,
                'confidence_score': chain.confidence_score,
                'execution_time': chain.total_execution_time,
                'steps_completed': len(chain.completed_steps),
                'steps_failed': len(chain.failed_steps),
                'reasoning_type': chain.reasoning_type.value,
                'step_details': [self._step_to_dict(step) for step in chain.steps]
            }
            
            self.logger.info(f"Reasoning chain completed: {chain.chain_id} "
                           f"(success: {success}, time: {chain.total_execution_time:.2f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing reasoning chain {chain.chain_id}: {e}")
            return {
                'chain_id': chain.chain_id,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
        
        finally:
            self.active_chains.pop(chain.chain_id, None)
    
    async def _execute_linear_chain(self, chain: ReasoningChain) -> None:
        """Execute steps sequentially in order"""
        
        for step in chain.steps:
            if step.status == StepStatus.PLANNED:
                # Wait for dependencies
                while not step.can_execute(chain.completed_steps):
                    await asyncio.sleep(0.1)
                
                # Execute step
                success = await self._execute_single_step(step, chain)
                
                if success:
                    chain.mark_step_completed(step.step_id)
                else:
                    chain.mark_step_failed(step.step_id)
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        step.status = StepStatus.PLANNED  # Retry
    
    async def _execute_branching_chain(self, chain: ReasoningChain) -> None:
        """Execute independent steps in parallel, then synthesis steps"""
        
        # Execute independent steps in parallel
        independent_steps = []
        synthesis_steps = []
        
        for step in chain.steps:
            if step.step_type == StepType.SYNTHESIS:
                synthesis_steps.append(step)
            else:
                independent_steps.append(step)
        
        # Run independent steps concurrently
        if independent_steps:
            tasks = []
            for step in independent_steps:
                task = asyncio.create_task(self._execute_single_step(step, chain))
                tasks.append((task, step))
            
            results = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
            
            for (task, step), success in zip(tasks, results):
                if isinstance(success, bool) and success:
                    chain.mark_step_completed(step.step_id)
                else:
                    chain.mark_step_failed(step.step_id)
        
        # Execute synthesis steps sequentially
        for step in synthesis_steps:
            if step.can_execute(chain.completed_steps):
                success = await self._execute_single_step(step, chain)
                if success:
                    chain.mark_step_completed(step.step_id)
                else:
                    chain.mark_step_failed(step.step_id)
    
    async def _execute_hierarchical_chain(self, chain: ReasoningChain) -> None:
        """Execute steps by priority levels"""
        
        # Group steps by priority
        priority_groups = defaultdict(list)
        for step in chain.steps:
            priority_groups[step.priority].append(step)
        
        # Execute each priority level
        for priority in sorted(priority_groups.keys()):
            steps = priority_groups[priority]
            
            # Execute steps in this priority level
            tasks = []
            for step in steps:
                if step.can_execute(chain.completed_steps):
                    task = asyncio.create_task(self._execute_single_step(step, chain))
                    tasks.append((task, step))
            
            if tasks:
                results = await asyncio.gather(*[task for task, _ in tasks], return_exceptions=True)
                
                for (task, step), success in zip(tasks, results):
                    if isinstance(success, bool) and success:
                        chain.mark_step_completed(step.step_id)
                    else:
                        chain.mark_step_failed(step.step_id)
    
    async def _execute_iterative_chain(self, chain: ReasoningChain) -> None:
        """Execute with iterative refinement"""
        
        max_iterations = 3
        iteration = 0
        
        while iteration < max_iterations and not chain.is_complete():
            iteration += 1
            
            # Get executable steps
            executable_steps = chain.get_next_executable_steps()
            
            if not executable_steps:
                break
            
            # Execute available steps
            for step in executable_steps:
                success = await self._execute_single_step(step, chain)
                
                if success:
                    chain.mark_step_completed(step.step_id)
                    
                    # Check if we need refinement
                    if step.confidence_score < 0.7:
                        # Add refinement step
                        refinement_step = RetrievalStep(
                            step_id=f"refinement_{step.step_id}",
                            step_type=StepType.REFINEMENT,
                            step_name=f"Refinement of {step.step_name}",
                            query=f"additional information clarification {step.query}",
                            depends_on=[step.step_id]
                        )
                        chain.add_step(refinement_step)
                else:
                    chain.mark_step_failed(step.step_id)
    
    async def _execute_sequential_chain(self, chain: ReasoningChain) -> None:
        """Default sequential execution"""
        await self._execute_linear_chain(chain)
    
    async def _execute_single_step(self, step: RetrievalStep, chain: ReasoningChain) -> bool:
        """Execute single retrieval step"""
        
        step.start_execution()
        
        try:
            # Build context from previous steps
            context = self._build_step_context(step, chain)
            
            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(step.query, context)
            
            # Simulate retrieval (in real implementation, use actual retriever)
            documents = await self._simulate_retrieval(enhanced_query, step.step_type)
            step.retrieved_documents = documents
            
            # Extract insights from retrieved documents
            insights = self._extract_step_insights(documents, step.step_type)
            step.insights = insights
            
            # Extract facts
            facts = self._extract_step_facts(documents)
            step.extracted_facts = facts
            
            # Calculate confidence
            step.confidence_score = self._calculate_step_confidence(step)
            
            step.complete_execution(success=True)
            
            self.logger.info(f"Step completed: {step.step_name} "
                           f"(confidence: {step.confidence_score:.2f})")
            
            return True
            
        except Exception as e:
            step.error_message = str(e)
            step.complete_execution(success=False)
            
            self.logger.error(f"Step failed: {step.step_name} - {e}")
            
            return False
    
    def _build_step_context(self, step: RetrievalStep, chain: ReasoningChain) -> str:
        """Build context from previous steps"""
        context_parts = []
        
        for context_step_id in step.context_from_steps:
            context_step = chain.get_step_by_id(context_step_id)
            if context_step and context_step.insights:
                context_parts.append(f"From {context_step.step_name}: {' '.join(context_step.insights[:2])}")
        
        return " ".join(context_parts)
    
    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """Enhance query with context from previous steps"""
        if context:
            return f"{query} considering {context}"
        return query
    
    async def _simulate_retrieval(self, query: str, step_type: StepType) -> List[str]:
        """Simulate document retrieval (replace with actual retriever)"""
        # Simulate different types of documents based on step type
        if step_type == StepType.FACTUAL_GATHERING:
            return [
                f"Factual document about {query[:30]}...",
                f"Data report on {query[:30]}...",
                f"Statistical analysis of {query[:30]}..."
            ]
        elif step_type == StepType.COMPARATIVE_RESEARCH:
            return [
                f"Comparison analysis: {query[:30]}...",
                f"Competitive landscape: {query[:30]}...",
                f"Benchmarking study: {query[:30]}..."
            ]
        elif step_type == StepType.CAUSAL_INVESTIGATION:
            return [
                f"Causal analysis: {query[:30]}...",
                f"Root cause investigation: {query[:30]}...",
                f"Impact assessment: {query[:30]}..."
            ]
        else:
            return [
                f"General document: {query[:30]}...",
                f"Research paper: {query[:30]}...",
                f"Analysis report: {query[:30]}..."
            ]
    
    def _extract_step_insights(self, documents: List[str], step_type: StepType) -> List[str]:
        """Extract insights from documents based on step type"""
        insights = []
        
        if step_type == StepType.FACTUAL_GATHERING:
            insights.append("Key facts and data points identified")
            insights.append("Statistical trends observed")
        elif step_type == StepType.COMPARATIVE_RESEARCH:
            insights.append("Significant differences identified between options")
            insights.append("Competitive advantages and disadvantages noted")
        elif step_type == StepType.CAUSAL_INVESTIGATION:
            insights.append("Primary causal factors identified")
            insights.append("Secondary effects and consequences mapped")
        elif step_type == StepType.SYNTHESIS:
            insights.append("Integrated analysis of all available information")
            insights.append("Comprehensive conclusions drawn from evidence")
        else:
            insights.append("Relevant information gathered and analyzed")
        
        return insights
    
    def _extract_step_facts(self, documents: List[str]) -> List[str]:
        """Extract factual information from documents"""
        facts = []
        
        for i, doc in enumerate(documents[:3]):  # Extract from top 3 documents
            facts.append(f"Fact {i+1}: Key information from {doc[:40]}...")
        
        return facts
    
    def _calculate_step_confidence(self, step: RetrievalStep) -> float:
        """Calculate confidence score for step"""
        base_confidence = 0.8
        
        # Adjust based on number of documents retrieved
        doc_factor = min(len(step.retrieved_documents) / 3.0, 1.0)
        
        # Adjust based on step type (some types are more reliable)
        type_factor = 1.0
        if step.step_type == StepType.FACTUAL_GATHERING:
            type_factor = 0.9
        elif step.step_type == StepType.SYNTHESIS:
            type_factor = 0.8  # Synthesis is inherently less certain
        
        confidence = base_confidence * doc_factor * type_factor
        return min(confidence, 1.0)
    
    def _calculate_chain_confidence(self, chain: ReasoningChain) -> float:
        """Calculate overall confidence for reasoning chain"""
        if not chain.steps:
            return 0.0
        
        step_confidences = [step.confidence_score for step in chain.steps 
                           if step.status == StepStatus.COMPLETED]
        
        if not step_confidences:
            return 0.0
        
        # Weighted average with penalty for failed steps
        avg_confidence = sum(step_confidences) / len(step_confidences)
        failure_penalty = len(chain.failed_steps) * 0.1
        
        return max(0.0, avg_confidence - failure_penalty)
    
    async def _synthesize_final_answer(self, chain: ReasoningChain) -> str:
        """Synthesize final answer from all step results"""
        
        if not chain.completed_steps:
            return "Unable to provide a comprehensive answer due to insufficient information."
        
        # Collect insights from all completed steps
        all_insights = []
        for step in chain.steps:
            if step.status == StepStatus.COMPLETED:
                all_insights.extend(step.insights)
        
        # Build comprehensive answer based on reasoning type
        if chain.reasoning_type == ReasoningType.HIERARCHICAL:
            return self._synthesize_hierarchical_answer(chain, all_insights)
        elif chain.reasoning_type == ReasoningType.BRANCHING:
            return self._synthesize_comparative_answer(chain, all_insights)
        elif chain.reasoning_type == ReasoningType.LINEAR:
            return self._synthesize_sequential_answer(chain, all_insights)
        else:
            return self._synthesize_general_answer(chain, all_insights)
    
    def _synthesize_hierarchical_answer(self, chain: ReasoningChain, 
                                      insights: List[str]) -> str:
        """Synthesize answer for hierarchical reasoning"""
        return f"""
Based on comprehensive multi-step analysis of your question: "{chain.original_question}"

Executive Summary:
{insights[0] if insights else 'Analysis completed successfully'}

Key Findings:
- Market and financial analysis completed
- Risk factors assessed and evaluated  
- Competitive landscape analyzed
- Future outlook considered

Recommendation:
Based on the evidence gathered across {len(chain.completed_steps)} analytical steps, 
the data suggests a nuanced approach considering multiple factors identified in the research.

Confidence Level: {chain.confidence_score:.1%}
"""
    
    def _synthesize_comparative_answer(self, chain: ReasoningChain, 
                                     insights: List[str]) -> str:
        """Synthesize answer for comparative analysis"""
        return f"""
Comparative Analysis Results for: "{chain.original_question}"

Analysis Summary:
The comparison reveals distinct characteristics and trade-offs between the options analyzed.

Key Insights:
{chr(10).join(f"- {insight}" for insight in insights[:5])}

Conclusion:
Each option has unique strengths and considerations. The choice depends on specific 
priorities and circumstances based on the comparative analysis conducted.

Analysis Confidence: {chain.confidence_score:.1%}
"""
    
    def _synthesize_sequential_answer(self, chain: ReasoningChain, 
                                    insights: List[str]) -> str:
        """Synthesize answer for sequential reasoning"""
        return f"""
Step-by-Step Analysis of: "{chain.original_question}"

Progressive Analysis Results:
{chr(10).join(f"Step {i+1}: {insight}" for i, insight in enumerate(insights[:4]))}

Final Conclusion:
The sequential analysis provides a comprehensive understanding of the topic, 
building from foundational concepts to specific conclusions.

Overall Confidence: {chain.confidence_score:.1%}
"""
    
    def _synthesize_general_answer(self, chain: ReasoningChain, 
                                 insights: List[str]) -> str:
        """Synthesize general answer"""
        return f"""
Comprehensive Analysis: "{chain.original_question}"

Analysis completed across {len(chain.completed_steps)} information gathering steps.

Key Findings:
{chr(10).join(f"- {insight}" for insight in insights[:3])}

The multi-step analysis provides a thorough examination of the topic with 
insights drawn from multiple sources and perspectives.

Confidence: {chain.confidence_score:.1%}
"""
    
    def _step_to_dict(self, step: RetrievalStep) -> Dict[str, Any]:
        """Convert step to dictionary for response"""
        return {
            'step_id': step.step_id,
            'step_name': step.step_name,
            'step_type': step.step_type.value,
            'status': step.status.value,
            'execution_time': step.execution_time,
            'confidence_score': step.confidence_score,
            'insights_count': len(step.insights),
            'facts_count': len(step.extracted_facts),
            'documents_retrieved': len(step.retrieved_documents)
        }
    
    def _update_execution_stats(self, chain: ReasoningChain) -> None:
        """Update execution statistics"""
        self.execution_stats['total_chains_executed'] += 1
        self.execution_stats['total_steps_executed'] += len(chain.steps)
        
        # Update averages
        chain_count = self.execution_stats['total_chains_executed']
        current_avg_time = self.execution_stats['average_chain_time']
        
        self.execution_stats['average_chain_time'] = (
            (current_avg_time * (chain_count - 1) + chain.total_execution_time) / chain_count
        )
        
        # Calculate success rate
        if chain.is_complete() and not chain.has_failures():
            current_success_rate = self.execution_stats['success_rate']
            self.execution_stats['success_rate'] = (
                (current_success_rate * (chain_count - 1) + 1.0) / chain_count
            )
        else:
            current_success_rate = self.execution_stats['success_rate']
            self.execution_stats['success_rate'] = (
                (current_success_rate * (chain_count - 1) + 0.0) / chain_count
            )
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()

class MultiStepRAGSystem:
    """
    Complete multi-step RAG system for complex reasoning
    
    EXAMPLE USAGE:
    =============
    # Create multi-step RAG system
    rag = MultiStepRAGSystem()
    await rag.initialize()
    
    # Ask complex question
    question = "Should I invest in renewable energy companies in emerging markets considering current geopolitical tensions?"
    
    result = await rag.process_complex_question(question)
    
    print(result['final_answer'])
    print(f"Analysis completed in {result['execution_time']:.2f}s")
    print(f"Confidence: {result['confidence_score']:.1%}")
    """
    
    def __init__(self):
        self.decomposer = QuestionDecomposer()
        self.executor = MultiStepExecutor(retriever="mock_retriever")
        
        # System state
        self.initialized = False
        
        # Statistics
        self.system_stats = {
            'complex_questions_processed': 0,
            'average_steps_per_question': 0.0,
            'average_processing_time': 0.0,
            'reasoning_type_distribution': defaultdict(int)
        }
        
        self.logger = logging.getLogger("MultiStepRAGSystem")
    
    async def initialize(self) -> None:
        """Initialize multi-step RAG system"""
        self.initialized = True
        self.logger.info("Multi-step RAG system initialized")
    
    async def process_complex_question(self, question: str) -> Dict[str, Any]:
        """Process complex question using multi-step reasoning"""
        
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        self.system_stats['complex_questions_processed'] += 1
        
        try:
            # Step 1: Decompose question into reasoning chain
            self.logger.info(f"Decomposing complex question: {question[:50]}...")
            
            reasoning_chain = await self.decomposer.decompose_question(question)
            
            self.system_stats['reasoning_type_distribution'][reasoning_chain.reasoning_type.value] += 1
            
            # Step 2: Execute reasoning chain
            self.logger.info(f"Executing reasoning chain with {len(reasoning_chain.steps)} steps")
            
            execution_result = await self.executor.execute_reasoning_chain(reasoning_chain)
            
            # Step 3: Update system statistics
            total_time = time.time() - start_time
            self._update_system_stats(reasoning_chain, total_time)
            
            # Step 4: Format final result
            result = {
                'original_question': question,
                'reasoning_chain_id': reasoning_chain.chain_id,
                'reasoning_type': reasoning_chain.reasoning_type.value,
                'total_steps': len(reasoning_chain.steps),
                'steps_completed': len(reasoning_chain.completed_steps),
                'steps_failed': len(reasoning_chain.failed_steps),
                'final_answer': execution_result.get('final_answer', ''),
                'confidence_score': execution_result.get('confidence_score', 0.0),
                'execution_time': total_time,
                'step_details': execution_result.get('step_details', []),
                'success': execution_result.get('success', False)
            }
            
            self.logger.info(f"Complex question processed successfully: "
                           f"{total_time:.2f}s, {len(reasoning_chain.completed_steps)} steps")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing complex question: {e}")
            return {
                'original_question': question,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _update_system_stats(self, chain: ReasoningChain, total_time: float) -> None:
        """Update system statistics"""
        question_count = self.system_stats['complex_questions_processed']
        
        # Update average steps per question
        current_avg_steps = self.system_stats['average_steps_per_question']
        self.system_stats['average_steps_per_question'] = (
            (current_avg_steps * (question_count - 1) + len(chain.steps)) / question_count
        )
        
        # Update average processing time
        current_avg_time = self.system_stats['average_processing_time']
        self.system_stats['average_processing_time'] = (
            (current_avg_time * (question_count - 1) + total_time) / question_count
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'system_stats': self.system_stats,
            'executor_stats': self.executor.get_execution_statistics(),
            'reasoning_types_supported': [t.value for t in ReasoningType],
            'step_types_supported': [t.value for t in StepType]
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_question_decomposition():
    """Demo: Complex question decomposition"""
    print("\nDEMO 1: COMPLEX QUESTION DECOMPOSITION")
    print("=" * 50)
    
    decomposer = QuestionDecomposer()
    
    complex_questions = [
        "Should I invest in Tesla stock considering the current EV market trends and Elon Musk's leadership?",
        "Compare the advantages and disadvantages of Python versus Java for enterprise web development",
        "Why did the 2008 financial crisis happen and what were its long-term effects on global markets?",
        "What are the historical trends in climate change and what do they predict for future global temperatures?",
        "How do neural networks work and what makes them effective for image recognition tasks?"
    ]
    
    print("Decomposing complex questions into reasoning chains:")
    
    for i, question in enumerate(complex_questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")
        
        chain = await decomposer.decompose_question(question)
        
        print(f"Reasoning Type: {chain.reasoning_type.value}")
        print(f"Steps Generated: {len(chain.steps)}")
        
        print("Step Breakdown:")
        for j, step in enumerate(chain.steps, 1):
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            print(f"  {j}. {step.step_name} [{step.step_type.value}]{deps}")
            print(f"     Query: {step.query}")

async def demo_step_execution():
    """Demo: Multi-step execution process"""
    print("\nDEMO 2: MULTI-STEP EXECUTION")
    print("=" * 50)
    
    # Create a sample reasoning chain
    decomposer = QuestionDecomposer()
    executor = MultiStepExecutor(retriever="mock")
    
    question = "Should I invest in renewable energy companies considering current market conditions?"
    
    print(f"Question: {question}")
    
    # Decompose question
    chain = await decomposer.decompose_question(question)
    
    print(f"\nGenerated reasoning chain:")
    print(f"Type: {chain.reasoning_type.value}")
    print(f"Steps: {len(chain.steps)}")
    
    # Execute chain
    print(f"\nExecuting reasoning chain...")
    result = await executor.execute_reasoning_chain(chain)
    
    print(f"\nExecution Results:")
    print(f"Success: {result['success']}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Steps Completed: {result['steps_completed']}")
    print(f"Steps Failed: {result['steps_failed']}")
    print(f"Confidence: {result['confidence_score']:.1%}")
    
    print(f"\nStep Details:")
    for step_detail in result['step_details']:
        print(f"  - {step_detail['step_name']}: {step_detail['status']}")
        print(f"    Time: {step_detail['execution_time']:.3f}s, "
              f"Confidence: {step_detail['confidence_score']:.2f}")
    
    print(f"\nFinal Answer:")
    print(result['final_answer'])

async def demo_reasoning_types():
    """Demo: Different reasoning types"""
    print("\nDEMO 3: DIFFERENT REASONING TYPES")
    print("=" * 50)
    
    decomposer = QuestionDecomposer()
    
    # Questions designed to trigger different reasoning types
    reasoning_examples = [
        ("Linear", "What causes inflation and how does it affect the economy?"),
        ("Branching", "Compare the features and performance of iPhone vs Samsung Galaxy"),
        ("Hierarchical", "Analyze the investment potential of Apple stock in the current market"),
        ("Iterative", "How has artificial intelligence evolved and what are future prospects?")
    ]
    
    for reasoning_type, question in reasoning_examples:
        print(f"\n--- {reasoning_type} Reasoning ---")
        print(f"Question: {question}")
        
        chain = await decomposer.decompose_question(question)
        
        print(f"Detected Type: {chain.reasoning_type.value}")
        print(f"Step Structure:")
        
        # Group steps by priority for hierarchical
        if chain.reasoning_type == ReasoningType.HIERARCHICAL:
            priority_groups = defaultdict(list)
            for step in chain.steps:
                priority_groups[step.priority].append(step)
            
            for priority in sorted(priority_groups.keys()):
                print(f"  Priority {priority}:")
                for step in priority_groups[priority]:
                    print(f"    - {step.step_name}")
        
        # Show dependencies for other types
        else:
            for step in chain.steps:
                deps = f" → depends on: {', '.join(step.depends_on)}" if step.depends_on else ""
                print(f"  - {step.step_name}{deps}")

async def demo_complete_multistep_rag():
    """Demo: Complete multi-step RAG system"""
    print("\nDEMO 4: COMPLETE MULTI-STEP RAG SYSTEM")
    print("=" * 50)
    
    rag_system = MultiStepRAGSystem()
    await rag_system.initialize()
    
    # Test complex questions
    complex_questions = [
        "Should I invest in electric vehicle companies considering the current supply chain issues and government policies?",
        "Compare the long-term career prospects of software engineering versus data science in the AI era",
        "What factors led to the rise of remote work and how will it affect urban real estate markets?"
    ]
    
    print("Processing complex questions through multi-step RAG:")
    
    for i, question in enumerate(complex_questions, 1):
        print(f"\n{'='*60}")
        print(f"COMPLEX QUESTION {i}")
        print(f"{'='*60}")
        print(f"Q: {question}")
        
        result = await rag_system.process_complex_question(question)
        
        if result['success']:
            print(f"\nReasoning Analysis:")
            print(f"  Type: {result['reasoning_type']}")
            print(f"  Total Steps: {result['total_steps']}")
            print(f"  Completed: {result['steps_completed']}")
            print(f"  Processing Time: {result['execution_time']:.2f}s")
            print(f"  Confidence: {result['confidence_score']:.1%}")
            
            print(f"\nStep Execution Summary:")
            for step in result['step_details'][:5]:  # Show first 5 steps
                status_icon = "✅" if step['status'] == 'completed' else "❌"
                print(f"  {status_icon} {step['step_name']}")
                print(f"     Time: {step['execution_time']:.3f}s, "
                      f"Confidence: {step['confidence_score']:.2f}")
            
            print(f"\nFinal Analysis:")
            print(result['final_answer'])
        
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")

async def demo_multistep_analytics():
    """Demo: Multi-step RAG analytics and insights"""
    print("\nDEMO 5: MULTI-STEP RAG ANALYTICS")
    print("=" * 50)
    
    rag_system = MultiStepRAGSystem()
    await rag_system.initialize()
    
    # Process multiple questions to generate analytics
    test_questions = [
        "Analyze the impact of AI on healthcare industry transformation",
        "Compare renewable energy investment opportunities across different technologies",
        "What are the economic implications of cryptocurrency adoption by institutions?",
        "How will autonomous vehicles affect transportation and urban planning?",
        "Evaluate the sustainability challenges facing global supply chains"
    ]
    
    print("Processing multiple complex questions for analytics...")
    
    results = []
    for question in test_questions:
        result = await rag_system.process_complex_question(question)
        results.append(result)
        print(f"  ✓ Processed: {question[:50]}...")
    
    # Get comprehensive statistics
    stats = rag_system.get_system_statistics()
    
    print(f"\nMULTI-STEP RAG SYSTEM ANALYTICS")
    print("=" * 40)
    
    print(f"\nQuestion Processing Statistics:")
    system_stats = stats['system_stats']
    print(f"  Questions processed: {system_stats['complex_questions_processed']}")
    print(f"  Average steps per question: {system_stats['average_steps_per_question']:.1f}")
    print(f"  Average processing time: {system_stats['average_processing_time']:.2f}s")
    
    print(f"\nReasoning Type Distribution:")
    for reasoning_type, count in system_stats['reasoning_type_distribution'].items():
        percentage = (count / system_stats['complex_questions_processed']) * 100
        print(f"  {reasoning_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nStep Execution Statistics:")
    executor_stats = stats['executor_stats']
    print(f"  Total chains executed: {executor_stats['total_chains_executed']}")
    print(f"  Total steps executed: {executor_stats['total_steps_executed']}")
    print(f"  Average chain time: {executor_stats['average_chain_time']:.2f}s")
    print(f"  Success rate: {executor_stats['success_rate']:.1%}")
    
    print(f"\nSystem Capabilities:")
    print(f"  Reasoning types: {len(stats['reasoning_types_supported'])}")
    print(f"  Step types: {len(stats['step_types_supported'])}")
    
    print(f"\nProcessing Results Summary:")
    successful_results = [r for r in results if r['success']]
    print(f"  Successful: {len(successful_results)}/{len(results)}")
    
    if successful_results:
        avg_confidence = sum(r['confidence_score'] for r in successful_results) / len(successful_results)
        avg_steps = sum(r['total_steps'] for r in successful_results) / len(successful_results)
        
        print(f"  Average confidence: {avg_confidence:.1%}")
        print(f"  Average steps: {avg_steps:.1f}")

async def main():
    """
    Demonstrate Multi-Step RAG Systems for complex reasoning through iterative retrieval
    
    WHAT YOU'LL LEARN:
    ================
    1. How to decompose complex questions into reasoning chains
    2. How to implement different reasoning types (linear, branching, hierarchical)
    3. How to execute multi-step retrieval with dependencies
    4. How to synthesize comprehensive answers from multiple steps
    5. How to build systems that handle human-expert level analysis
    
    REAL WORLD APPLICATIONS:
    =======================
    - Strategic business analysis and consulting
    - Investment research and financial analysis
    - Scientific literature review and research
    - Legal case analysis and precedent research
    - Market research and competitive intelligence
    - Policy analysis and regulatory impact assessment
    """
    
    print("MULTI-STEP RAG SYSTEMS DEMONSTRATION")
    print("Showing how to build systems that reason through complex questions!")
    
    await demo_question_decomposition()
    await demo_step_execution()
    await demo_reasoning_types()
    await demo_complete_multistep_rag()
    await demo_multistep_analytics()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Complex questions require multi-step reasoning chains")
    print("✓ Different reasoning types handle different question patterns")
    print("✓ Step dependencies ensure logical information gathering")
    print("✓ Progressive synthesis builds comprehensive understanding")
    print("✓ Multi-step systems enable human-expert level analysis")
    print("✓ Analytics help optimize reasoning chain performance")
    print("\nTHE POWER OF MULTI-STEP RAG:")
    print("- Handles complex analytical and strategic questions")
    print("- Provides comprehensive coverage of multi-faceted topics")
    print("- Enables transparent, auditable reasoning processes")
    print("- Powers professional-grade analysis and decision support")

if __name__ == "__main__":
    asyncio.run(main())
