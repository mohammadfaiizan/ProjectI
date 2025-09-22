#!/usr/bin/env python3
"""
Pipeline Agent Pattern: Sequential Processing with Quality Control
===============================================================

WHAT IS THE PROBLEM?
==================
Complex tasks require multiple steps in sequence, but doing everything at once leads to poor quality and errors.

Example: Making a Professional Video
BAD APPROACH:
- Try to film, edit, add music, add effects, and publish all at the same time
- Result: Chaos, poor quality, missed steps, unusable output

REAL WORLD EXAMPLE:
=================
How do professional video production companies work?

PIPELINE STAGES:
1. PRE-PRODUCTION: Script writing, planning, storyboarding
2. FILMING: Actual video recording with proper setup
3. ROUGH EDIT: Basic video editing and sequencing
4. QUALITY REVIEW: Check for issues, gather feedback
5. FINAL EDIT: Polish, effects, color correction
6. AUDIO POST: Music, sound effects, voice-over
7. FINAL REVIEW: Quality control and approval
8. PUBLISHING: Export and distribute

Each stage:
- Has specific inputs and outputs
- Has quality criteria
- Can reject work back to previous stage
- Has specialized tools and expertise
- Builds on the previous stage's output

THE ALGORITHM:
=============
1. DEFINE STAGES: Break process into sequential stages
2. DEFINE INTERFACES: Specify inputs/outputs for each stage
3. DEFINE QUALITY GATES: Set criteria for stage completion
4. PROCESS: Execute stages sequentially
5. QUALITY CHECK: Validate output before moving to next stage
6. FEEDBACK LOOP: Return to previous stage if quality fails

PSEUDO CODE:
===========
class Pipeline:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage = 0
        self.stage_outputs = []
    
    def process(self, initial_input):
        current_data = initial_input
        
        for stage_index, stage in enumerate(self.stages):
            # Process through current stage
            stage_output = stage.process(current_data)
            
            # Quality check
            if not stage.quality_check(stage_output):
                # Retry or return to previous stage
                if stage_index > 0:
                    current_data = self.stage_outputs[stage_index - 1]
                    continue
                else:
                    return {"error": "Initial stage failed quality check"}
            
            # Stage passed, save output and continue
            self.stage_outputs.append(stage_output)
            current_data = stage_output
        
        return {"final_output": current_data}

WHY IS THIS POWERFUL?
===================
- Breaks complex processes into manageable steps
- Enables quality control at each stage
- Allows specialization and optimization per stage
- Makes errors easier to identify and fix
- Enables parallel processing of multiple items
- Provides clear progress tracking
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class StageStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REWORK = "requires_rework"

class QualityLevel(Enum):
    POOR = 1
    ACCEPTABLE = 2
    GOOD = 3
    EXCELLENT = 4

@dataclass
class ProcessingItem:
    """Item moving through the pipeline"""
    id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    current_stage: int = 0
    stage_history: List[Dict] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    total_processing_time: float = 0.0

@dataclass
class StageResult:
    """Result of processing through one stage"""
    success: bool
    output_data: Any
    quality_score: float
    processing_time: float
    issues: List[str] = field(default_factory=list)
    stage_metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""
    
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this stage"""
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        """Process data through this stage"""
        pass
    
    @abstractmethod
    def quality_threshold(self) -> float:
        """Minimum quality score to pass this stage"""
        pass
    
    @abstractmethod
    def can_retry(self) -> bool:
        """Whether this stage supports retry on failure"""
        pass

class PipelineAgent:
    """
    An agent that processes items through a sequential pipeline with quality control
    
    EXAMPLE USAGE:
    =============
    # Create content creation pipeline
    pipeline = PipelineAgent("content_creation")
    
    pipeline.add_stage(IdeaGenerationStage())
    pipeline.add_stage(ContentWritingStage())
    pipeline.add_stage(EditingStage())
    pipeline.add_stage(QualityReviewStage())
    pipeline.add_stage(PublishingStage())
    
    # Process content through pipeline
    result = await pipeline.process_item("Create blog post about AI")
    """
    
    def __init__(self, pipeline_id: str, max_retries: int = 2):
        self.pipeline_id = pipeline_id
        self.max_retries = max_retries
        self.stages: List[PipelineStage] = []
        self.processing_items: Dict[str, ProcessingItem] = {}
        self.completed_items: List[ProcessingItem] = []
        self.failed_items: List[ProcessingItem] = []
        
        # Pipeline metrics
        self.total_items_processed = 0
        self.stage_performance: Dict[str, Dict[str, float]] = {}
        self.bottleneck_analysis: Dict[str, float] = {}
    
    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline"""
        self.stages.append(stage)
        stage_name = stage.get_stage_name()
        
        # Initialize performance tracking
        self.stage_performance[stage_name] = {
            "total_processing_time": 0.0,
            "items_processed": 0,
            "quality_scores": [],
            "failure_rate": 0.0
        }
        
        print(f"Added pipeline stage: {stage_name}")
    
    async def process_item(self, initial_data: Any, item_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single item through the entire pipeline
        
        Args:
            initial_data: The data to process
            item_id: Optional ID for the item
            
        Returns:
            Processing results including final output and pipeline statistics
        """
        if item_id is None:
            item_id = f"item_{self.total_items_processed + 1}"
        
        print(f"\nPROCESSING ITEM: {item_id}")
        print("=" * 50)
        
        # Create processing item
        item = ProcessingItem(id=item_id, data=initial_data)
        self.processing_items[item_id] = item
        
        start_time = time.time()
        
        # Process through each stage
        for stage_index, stage in enumerate(self.stages):
            stage_name = stage.get_stage_name()
            print(f"\nSTAGE {stage_index + 1}: {stage_name}")
            print("-" * 30)
            
            # Update item's current stage
            item.current_stage = stage_index
            
            # Process with retry logic
            stage_result = await self.process_stage_with_retry(item, stage, stage_index)
            
            # Record stage result in history
            item.stage_history.append({
                "stage_index": stage_index,
                "stage_name": stage_name,
                "result": stage_result,
                "timestamp": time.time()
            })
            
            # Update performance metrics
            self.update_stage_performance(stage_name, stage_result)
            
            if not stage_result.success:
                # Stage failed completely
                print(f"✗ STAGE FAILED: {stage_name}")
                print(f"  Issues: {stage_result.issues}")
                
                item.data = stage_result.output_data
                self.failed_items.append(item)
                del self.processing_items[item_id]
                
                return {
                    "item_id": item_id,
                    "success": False,
                    "failed_at_stage": stage_name,
                    "issues": stage_result.issues,
                    "pipeline_progress": f"{stage_index + 1}/{len(self.stages)} stages"
                }
            
            # Stage succeeded, update item data
            item.data = stage_result.output_data
            item.quality_scores.append(stage_result.quality_score)
            
            print(f"✓ STAGE COMPLETED: {stage_name}")
            print(f"  Quality Score: {stage_result.quality_score:.2f}")
            print(f"  Processing Time: {stage_result.processing_time:.2f}s")
        
        # All stages completed successfully
        total_time = time.time() - start_time
        item.total_processing_time = total_time
        
        self.completed_items.append(item)
        del self.processing_items[item_id]
        self.total_items_processed += 1
        
        # Calculate overall metrics
        average_quality = sum(item.quality_scores) / len(item.quality_scores)
        
        print(f"\n✓ PIPELINE COMPLETED: {item_id}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Quality: {average_quality:.2f}")
        print(f"  Stages Completed: {len(self.stages)}/{len(self.stages)}")
        
        return {
            "item_id": item_id,
            "success": True,
            "final_output": item.data,
            "total_processing_time": total_time,
            "average_quality_score": average_quality,
            "stages_completed": len(self.stages),
            "stage_history": item.stage_history
        }
    
    async def process_stage_with_retry(self, item: ProcessingItem, stage: PipelineStage, 
                                     stage_index: int) -> StageResult:
        """
        Process item through a stage with retry logic
        """
        stage_name = stage.get_stage_name()
        retries_attempted = 0
        
        while retries_attempted <= self.max_retries:
            try:
                # Process through stage
                stage_result = await stage.process(item.data, item.metadata)
                
                # Check quality threshold
                if stage_result.quality_score >= stage.quality_threshold():
                    print(f"  ✓ Quality check passed ({stage_result.quality_score:.2f} >= {stage.quality_threshold():.2f})")
                    return stage_result
                else:
                    print(f"  ⚠ Quality check failed ({stage_result.quality_score:.2f} < {stage.quality_threshold():.2f})")
                    
                    if stage.can_retry() and retries_attempted < self.max_retries:
                        retries_attempted += 1
                        print(f"  ↻ Retrying stage (attempt {retries_attempted + 1}/{self.max_retries + 1})")
                        continue
                    else:
                        # Quality failed and no more retries
                        stage_result.success = False
                        stage_result.issues.append(f"Quality score {stage_result.quality_score:.2f} below threshold {stage.quality_threshold():.2f}")
                        return stage_result
                        
            except Exception as e:
                if retries_attempted < self.max_retries and stage.can_retry():
                    retries_attempted += 1
                    print(f"  ✗ Stage error: {str(e)}")
                    print(f"  ↻ Retrying stage (attempt {retries_attempted + 1}/{self.max_retries + 1})")
                    continue
                else:
                    # Return failure result
                    return StageResult(
                        success=False,
                        output_data=item.data,
                        quality_score=0.0,
                        processing_time=0.0,
                        issues=[f"Stage error: {str(e)}"]
                    )
        
        # Should not reach here, but failsafe
        return StageResult(
            success=False,
            output_data=item.data,
            quality_score=0.0,
            processing_time=0.0,
            issues=["Maximum retries exceeded"]
        )
    
    def update_stage_performance(self, stage_name: str, result: StageResult) -> None:
        """Update performance metrics for a stage"""
        if stage_name not in self.stage_performance:
            return
        
        perf = self.stage_performance[stage_name]
        perf["total_processing_time"] += result.processing_time
        perf["items_processed"] += 1
        perf["quality_scores"].append(result.quality_score)
        
        # Update failure rate
        total_failures = sum(1 for item in self.failed_items 
                           if any(history["stage_name"] == stage_name and not history["result"].success 
                                 for history in item.stage_history))
        perf["failure_rate"] = total_failures / perf["items_processed"] if perf["items_processed"] > 0 else 0
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status and metrics"""
        # Calculate average processing time per stage
        stage_averages = {}
        for stage_name, perf in self.stage_performance.items():
            if perf["items_processed"] > 0:
                stage_averages[stage_name] = {
                    "avg_processing_time": perf["total_processing_time"] / perf["items_processed"],
                    "avg_quality_score": sum(perf["quality_scores"]) / len(perf["quality_scores"]) if perf["quality_scores"] else 0,
                    "failure_rate": perf["failure_rate"]
                }
        
        # Identify bottlenecks
        bottleneck_stage = max(stage_averages.items(), 
                              key=lambda x: x[1]["avg_processing_time"]) if stage_averages else None
        
        return {
            "pipeline_id": self.pipeline_id,
            "total_stages": len(self.stages),
            "items_in_progress": len(self.processing_items),
            "items_completed": len(self.completed_items),
            "items_failed": len(self.failed_items),
            "total_items_processed": self.total_items_processed,
            "success_rate": len(self.completed_items) / max(1, self.total_items_processed),
            "stage_performance": stage_averages,
            "bottleneck_stage": bottleneck_stage[0] if bottleneck_stage else None
        }

# EXAMPLE PIPELINE STAGES
# =======================

class ContentIdeaStage(PipelineStage):
    """Stage for generating content ideas"""
    
    def get_stage_name(self) -> str:
        return "Idea Generation"
    
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        start_time = time.time()
        
        # Simulate idea generation process
        await asyncio.sleep(0.1)
        
        topic = str(input_data)
        ideas = [
            f"Introduction to {topic}",
            f"Advanced concepts in {topic}",
            f"Real-world applications of {topic}",
            f"Common misconceptions about {topic}",
            f"Future trends in {topic}"
        ]
        
        # Select best ideas (simulated)
        selected_ideas = ideas[:3]
        
        # Quality scoring based on idea diversity and relevance
        quality_score = 0.8 if len(selected_ideas) >= 3 else 0.6
        
        processing_time = time.time() - start_time
        
        return StageResult(
            success=True,
            output_data={
                "topic": topic,
                "ideas": selected_ideas,
                "selected_count": len(selected_ideas)
            },
            quality_score=quality_score,
            processing_time=processing_time,
            stage_metadata={"ideas_generated": len(ideas)}
        )
    
    def quality_threshold(self) -> float:
        return 0.7
    
    def can_retry(self) -> bool:
        return True

class ContentWritingStage(PipelineStage):
    """Stage for writing content based on ideas"""
    
    def get_stage_name(self) -> str:
        return "Content Writing"
    
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        start_time = time.time()
        
        # Simulate writing process
        await asyncio.sleep(0.2)
        
        ideas_data = input_data
        topic = ideas_data["topic"]
        ideas = ideas_data["ideas"]
        
        # Generate content for each idea
        content_sections = []
        for idea in ideas:
            section = {
                "heading": idea,
                "content": f"Detailed explanation of {idea}. This section covers the fundamental concepts, provides practical examples, and explains the significance in the context of {topic}. Key points include implementation strategies, best practices, and common pitfalls to avoid.",
                "word_count": 150
            }
            content_sections.append(section)
        
        total_words = sum(section["word_count"] for section in content_sections)
        
        # Quality scoring based on content length and completeness
        quality_score = min(1.0, total_words / 400)  # Target 400+ words
        
        processing_time = time.time() - start_time
        
        return StageResult(
            success=True,
            output_data={
                "topic": topic,
                "sections": content_sections,
                "total_words": total_words,
                "completion_status": "draft"
            },
            quality_score=quality_score,
            processing_time=processing_time,
            stage_metadata={"sections_written": len(content_sections)}
        )
    
    def quality_threshold(self) -> float:
        return 0.6
    
    def can_retry(self) -> bool:
        return True

class ContentEditingStage(PipelineStage):
    """Stage for editing and improving content"""
    
    def get_stage_name(self) -> str:
        return "Content Editing"
    
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        start_time = time.time()
        
        # Simulate editing process
        await asyncio.sleep(0.15)
        
        content_data = input_data
        sections = content_data["sections"]
        
        # Improve each section
        edited_sections = []
        improvements_made = 0
        
        for section in sections:
            edited_section = section.copy()
            
            # Simulate editing improvements
            if len(section["content"]) < 200:
                edited_section["content"] += " Additional insights and detailed explanations have been added to provide more comprehensive coverage of this topic."
                edited_section["word_count"] += 50
                improvements_made += 1
            
            # Add formatting and structure
            edited_section["formatted"] = True
            edited_sections.append(edited_section)
        
        total_words = sum(section["word_count"] for section in edited_sections)
        
        # Quality scoring based on improvements and completeness
        quality_score = 0.7 + (improvements_made / len(sections)) * 0.3
        
        processing_time = time.time() - start_time
        
        return StageResult(
            success=True,
            output_data={
                "topic": content_data["topic"],
                "sections": edited_sections,
                "total_words": total_words,
                "completion_status": "edited",
                "improvements_made": improvements_made
            },
            quality_score=quality_score,
            processing_time=processing_time,
            stage_metadata={"improvements_made": improvements_made}
        )
    
    def quality_threshold(self) -> float:
        return 0.7
    
    def can_retry(self) -> bool:
        return True

class QualityReviewStage(PipelineStage):
    """Stage for final quality review"""
    
    def get_stage_name(self) -> str:
        return "Quality Review"
    
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        start_time = time.time()
        
        # Simulate quality review process
        await asyncio.sleep(0.1)
        
        content_data = input_data
        sections = content_data["sections"]
        total_words = content_data["total_words"]
        
        # Quality checks
        quality_issues = []
        quality_score = 1.0
        
        # Check word count
        if total_words < 500:
            quality_issues.append("Content length below recommended minimum")
            quality_score -= 0.2
        
        # Check section completeness
        incomplete_sections = [s for s in sections if s["word_count"] < 100]
        if incomplete_sections:
            quality_issues.append(f"{len(incomplete_sections)} sections need more content")
            quality_score -= 0.1 * len(incomplete_sections)
        
        # Check formatting
        unformatted_sections = [s for s in sections if not s.get("formatted", False)]
        if unformatted_sections:
            quality_issues.append("Some sections lack proper formatting")
            quality_score -= 0.1
        
        quality_score = max(0.0, quality_score)
        
        processing_time = time.time() - start_time
        
        # Add quality review metadata
        reviewed_content = content_data.copy()
        reviewed_content["quality_review"] = {
            "score": quality_score,
            "issues": quality_issues,
            "reviewed_at": time.time()
        }
        reviewed_content["completion_status"] = "reviewed"
        
        return StageResult(
            success=True,
            output_data=reviewed_content,
            quality_score=quality_score,
            processing_time=processing_time,
            issues=quality_issues,
            stage_metadata={"quality_issues_found": len(quality_issues)}
        )
    
    def quality_threshold(self) -> float:
        return 0.8
    
    def can_retry(self) -> bool:
        return False  # Quality review doesn't retry, sends back to editing

class PublishingStage(PipelineStage):
    """Stage for publishing the final content"""
    
    def get_stage_name(self) -> str:
        return "Publishing"
    
    async def process(self, input_data: Any, metadata: Dict[str, Any]) -> StageResult:
        start_time = time.time()
        
        # Simulate publishing process
        await asyncio.sleep(0.05)
        
        content_data = input_data
        
        # Prepare for publishing
        published_content = {
            "title": f"Complete Guide: {content_data['topic']}",
            "content": content_data,
            "published_at": time.time(),
            "status": "published",
            "url": f"https://example.com/articles/{content_data['topic'].lower().replace(' ', '-')}"
        }
        
        # Publishing always succeeds if we reach this stage
        quality_score = 1.0
        
        processing_time = time.time() - start_time
        
        return StageResult(
            success=True,
            output_data=published_content,
            quality_score=quality_score,
            processing_time=processing_time,
            stage_metadata={"published_url": published_content["url"]}
        )
    
    def quality_threshold(self) -> float:
        return 0.9
    
    def can_retry(self) -> bool:
        return True

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_content_creation_pipeline():
    """Demo: Content creation pipeline"""
    print("\nDEMO 1: CONTENT CREATION PIPELINE")
    print("=" * 50)
    
    # Create content creation pipeline
    pipeline = PipelineAgent("content_creation", max_retries=1)
    
    # Add stages in order
    pipeline.add_stage(ContentIdeaStage())
    pipeline.add_stage(ContentWritingStage())
    pipeline.add_stage(ContentEditingStage())
    pipeline.add_stage(QualityReviewStage())
    pipeline.add_stage(PublishingStage())
    
    # Process content
    result1 = await pipeline.process_item("Machine Learning")
    result2 = await pipeline.process_item("Web Development")
    
    # Show pipeline status
    status = pipeline.get_pipeline_status()
    print(f"\nPIPELINE STATUS:")
    print(f"Success Rate: {status['success_rate']:.1%}")
    print(f"Items Completed: {status['items_completed']}")
    print(f"Items Failed: {status['items_failed']}")
    
    if status['bottleneck_stage']:
        print(f"Bottleneck Stage: {status['bottleneck_stage']}")

async def demo_simple_pipeline():
    """Demo: Simple data processing pipeline"""
    print("\nDEMO 2: SIMPLE DATA PROCESSING PIPELINE")
    print("=" * 50)
    
    # Create simple stages for demo
    class DataValidationStage(PipelineStage):
        def get_stage_name(self): return "Data Validation"
        async def process(self, input_data, metadata):
            await asyncio.sleep(0.05)
            is_valid = len(str(input_data)) > 0
            return StageResult(is_valid, input_data, 0.9 if is_valid else 0.1, 0.05)
        def quality_threshold(self): return 0.8
        def can_retry(self): return False
    
    class DataTransformationStage(PipelineStage):
        def get_stage_name(self): return "Data Transformation"
        async def process(self, input_data, metadata):
            await asyncio.sleep(0.1)
            transformed = str(input_data).upper()
            return StageResult(True, transformed, 0.85, 0.1)
        def quality_threshold(self): return 0.7
        def can_retry(self): return True
    
    class DataOutputStage(PipelineStage):
        def get_stage_name(self): return "Data Output"
        async def process(self, input_data, metadata):
            await asyncio.sleep(0.03)
            output = f"PROCESSED: {input_data}"
            return StageResult(True, output, 1.0, 0.03)
        def quality_threshold(self): return 0.9
        def can_retry(self): return False
    
    # Create and run simple pipeline
    simple_pipeline = PipelineAgent("data_processing")
    simple_pipeline.add_stage(DataValidationStage())
    simple_pipeline.add_stage(DataTransformationStage())
    simple_pipeline.add_stage(DataOutputStage())
    
    result = await simple_pipeline.process_item("hello world")
    print(f"Final Output: {result['final_output']}")

async def main():
    """
    Demonstrate Pipeline Agent Pattern for sequential processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to break complex processes into sequential stages
    2. How to implement quality control at each stage
    3. How to handle failures and implement retry logic
    4. How to track performance and identify bottlenecks
    5. How pipelines enable scalable and reliable processing
    
    REAL WORLD APPLICATIONS:
    =======================
    - Content creation and publishing workflows
    - Data processing and ETL pipelines
    - Software build and deployment pipelines
    - Manufacturing and quality control processes
    - Document processing and approval workflows
    - Media production and post-processing
    """
    
    print("PIPELINE AGENT PATTERN DEMONSTRATION")
    print("This shows how to process complex tasks through sequential stages with quality control!")
    
    await demo_content_creation_pipeline()
    await demo_simple_pipeline()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Pipelines break complex processes into manageable stages")
    print("✓ Quality gates ensure output meets standards before proceeding")
    print("✓ Retry logic handles temporary failures gracefully")
    print("✓ Performance monitoring identifies bottlenecks and optimization opportunities")
    print("✓ Sequential processing enables specialization and quality control")
    print("\nTRY IT YOURSELF:")
    print("- Create pipelines for your own complex processes")
    print("- Add parallel processing for independent stages")
    print("- Implement dynamic quality thresholds based on context")
    print("- Add pipeline visualization and real-time monitoring")

if __name__ == "__main__":
    asyncio.run(main())
