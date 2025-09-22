#!/usr/bin/env python3
"""
Master-Worker Pattern: Coordinated Distributed Processing
========================================================

WHAT IS THE PROBLEM?
==================
When you have a big job, one person working alone is slow and inefficient. You need multiple people working together, but without coordination, you get chaos.

Example: Restaurant Kitchen Without Coordination
BAD APPROACH:
- 5 chefs all try to cook the same dish
- No one knows who's doing what
- Ingredients get wasted, orders get duplicated
- Kitchen becomes chaotic, customers wait forever
- No quality control or progress tracking

REAL WORLD EXAMPLE:
=================
How does a professional restaurant kitchen actually work?

MASTER-WORKER COORDINATION:
MASTER (Head Chef):
- Receives all orders
- Breaks down complex dishes into tasks
- Assigns tasks to specialized workers
- Monitors progress of all tasks
- Coordinates timing so everything finishes together
- Handles quality control and final plating

WORKERS (Line Cooks):
- Each specializes in different stations (grill, salad, dessert)
- Focus on their assigned tasks
- Report completion back to head chef
- Handle their part of the workflow efficiently

WORKFLOW:
1. Order comes in: "Table 5 wants: Caesar salad, grilled salmon, chocolate cake"
2. Master breaks down: Salad station -> Caesar, Grill station -> Salmon, Dessert station -> Cake
3. Workers execute their tasks in parallel
4. Master coordinates timing: "Salad ready in 2 minutes, salmon needs 5 more minutes"
5. Master does final assembly and quality check
6. Complete order delivered to table

THE ALGORITHM:
=============
1. MASTER receives large task
2. BREAK DOWN task into smaller, parallel subtasks
3. ASSIGN subtasks to available workers based on their capabilities
4. MONITOR progress of all workers
5. COLLECT results as workers complete their tasks
6. COMBINE results into final output
7. HANDLE any worker failures or bottlenecks

PSEUDO CODE:
===========
class MasterWorkerSystem:
    def __init__(self):
        self.workers = []
        self.task_queue = []
        self.active_tasks = {}
        self.results = []
    
    def process_large_task(self, big_task):
        # Master breaks down the task
        subtasks = self.decompose_task(big_task)
        
        # Assign tasks to workers
        for subtask in subtasks:
            available_worker = self.find_available_worker(subtask.requirements)
            self.assign_task(worker, subtask)
        
        # Monitor and collect results
        while not all_tasks_complete():
            for worker in self.workers:
                if worker.has_result():
                    result = worker.get_result()
                    self.collect_result(result)
        
        # Combine all results
        final_result = self.combine_results(self.results)
        return final_result

WHY IS THIS POWERFUL?
===================
- Enables parallel processing for much faster completion
- Scales to handle large workloads by adding more workers
- Provides centralized coordination and quality control
- Handles worker failures gracefully through reassignment
- Optimizes resource utilization across the team
- Makes complex tasks manageable through decomposition
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class Task:
    """A unit of work that can be assigned to a worker"""
    id: str
    task_type: str
    data: Any
    priority: int = 1
    estimated_duration: float = 1.0
    required_capabilities: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    result: Any = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

@dataclass
class WorkerInfo:
    """Information about a worker's capabilities and status"""
    id: str
    capabilities: List[str]
    status: WorkerStatus
    current_task: Optional[str] = None
    tasks_completed: int = 0
    average_task_time: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

class Worker(ABC):
    """Abstract base class for workers"""
    
    @abstractmethod
    def get_id(self) -> str:
        """Get worker's unique identifier"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this worker can handle"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task and return the result"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if worker is available to take new tasks"""
        pass

class MasterWorkerSystem:
    """
    Master-Worker system for coordinated distributed processing
    
    EXAMPLE USAGE:
    =============
    # Create master-worker system for content processing
    system = MasterWorkerSystem("content_processing")
    
    # Add specialized workers
    system.add_worker(TextProcessorWorker("worker_1"))
    system.add_worker(ImageProcessorWorker("worker_2"))
    system.add_worker(DataAnalysisWorker("worker_3"))
    
    # Process large job
    result = await system.process_large_job(big_content_dataset)
    """
    
    def __init__(self, system_id: str, max_concurrent_tasks: int = 10):
        self.system_id = system_id
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Worker management
        self.workers: Dict[str, Worker] = {}
        self.worker_info: Dict[str, WorkerInfo] = {}
        
        # Task management
        self.task_queue: List[Task] = []
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # System monitoring
        self.total_tasks_processed = 0
        self.system_start_time = time.time()
        self.task_assignment_strategy = "capability_match"  # or "round_robin", "load_balance"
    
    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the system"""
        worker_id = worker.get_id()
        capabilities = worker.get_capabilities()
        
        self.workers[worker_id] = worker
        self.worker_info[worker_id] = WorkerInfo(
            id=worker_id,
            capabilities=capabilities,
            status=WorkerStatus.IDLE
        )
        
        print(f"Added worker: {worker_id} with capabilities: {capabilities}")
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the system"""
        if worker_id in self.workers:
            # Reassign any active tasks
            worker_tasks = [task for task in self.active_tasks.values() 
                          if task.assigned_worker == worker_id]
            
            for task in worker_tasks:
                task.status = TaskStatus.PENDING
                task.assigned_worker = None
                self.task_queue.append(task)
                del self.active_tasks[task.id]
            
            del self.workers[worker_id]
            del self.worker_info[worker_id]
            print(f"Removed worker: {worker_id}")
    
    async def process_large_job(self, job_data: Any, job_type: str = "general") -> Dict[str, Any]:
        """
        Process a large job by breaking it down and distributing to workers
        
        Args:
            job_data: The large dataset or job to process
            job_type: Type of job to help with task decomposition
            
        Returns:
            Results from processing the job
        """
        print(f"\nPROCESSING LARGE JOB: {job_type}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Break down large job into smaller tasks
        print("Step 1: Breaking down large job into tasks...")
        tasks = await self.decompose_job(job_data, job_type)
        print(f"Created {len(tasks)} tasks")
        
        # Add tasks to queue
        for task in tasks:
            self.task_queue.append(task)
        
        # Step 2: Distribute and execute tasks
        print("\nStep 2: Distributing tasks to workers...")
        await self.execute_all_tasks()
        
        # Step 3: Combine results
        print("\nStep 3: Combining results...")
        final_result = await self.combine_results(job_type)
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            "job_type": job_type,
            "total_tasks": len(tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_processing_time": total_time,
            "final_result": final_result,
            "worker_utilization": self.get_worker_utilization(),
            "success_rate": len(self.completed_tasks) / len(tasks) if tasks else 0
        }
        
        print(f"\nJOB COMPLETED:")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Tasks completed: {summary['completed_tasks']}/{summary['total_tasks']}")
        
        return summary
    
    async def decompose_job(self, job_data: Any, job_type: str) -> List[Task]:
        """
        Break down a large job into smaller, manageable tasks
        """
        tasks = []
        
        if job_type == "data_processing":
            # Break data into chunks for parallel processing
            if isinstance(job_data, list):
                chunk_size = max(1, len(job_data) // (len(self.workers) * 2))  # 2 tasks per worker
                
                for i in range(0, len(job_data), chunk_size):
                    chunk = job_data[i:i + chunk_size]
                    task = Task(
                        id=f"data_chunk_{i // chunk_size}",
                        task_type="process_data_chunk",
                        data=chunk,
                        priority=1,
                        estimated_duration=len(chunk) * 0.1,
                        required_capabilities=["data_processing"]
                    )
                    tasks.append(task)
        
        elif job_type == "content_generation":
            # Break content generation into different types
            content_types = ["introduction", "main_content", "conclusion", "summary"]
            
            for i, content_type in enumerate(content_types):
                task = Task(
                    id=f"content_{content_type}",
                    task_type="generate_content",
                    data={"content_type": content_type, "source_data": job_data},
                    priority=2 if content_type == "main_content" else 1,
                    estimated_duration=2.0,
                    required_capabilities=["text_generation", "content_writing"]
                )
                tasks.append(task)
        
        elif job_type == "image_processing":
            # Break image processing into different operations
            if isinstance(job_data, list):  # List of images
                operations = ["resize", "enhance", "analyze", "compress"]
                
                for img_idx, image_data in enumerate(job_data):
                    for operation in operations:
                        task = Task(
                            id=f"image_{img_idx}_{operation}",
                            task_type=f"image_{operation}",
                            data={"image": image_data, "operation": operation},
                            priority=1,
                            estimated_duration=1.5,
                            required_capabilities=["image_processing", operation]
                        )
                        tasks.append(task)
        
        else:
            # Generic decomposition
            if isinstance(job_data, list):
                for i, item in enumerate(job_data):
                    task = Task(
                        id=f"generic_task_{i}",
                        task_type="process_item",
                        data=item,
                        priority=1,
                        estimated_duration=1.0,
                        required_capabilities=["general_processing"]
                    )
                    tasks.append(task)
            else:
                # Single large task - create subtasks
                subtask_count = min(len(self.workers) * 2, 8)
                for i in range(subtask_count):
                    task = Task(
                        id=f"subtask_{i}",
                        task_type="process_subtask",
                        data={"subtask_index": i, "total_subtasks": subtask_count, "data": job_data},
                        priority=1,
                        estimated_duration=1.0,
                        required_capabilities=["general_processing"]
                    )
                    tasks.append(task)
        
        return tasks
    
    async def execute_all_tasks(self) -> None:
        """Execute all tasks in the queue using available workers"""
        
        # Start task assignment and monitoring loop
        while self.task_queue or self.active_tasks:
            # Assign pending tasks to available workers
            await self.assign_pending_tasks()
            
            # Check for completed tasks
            await self.check_completed_tasks()
            
            # Handle failed workers
            await self.handle_worker_failures()
            
            # Brief pause to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def assign_pending_tasks(self) -> None:
        """Assign pending tasks to available workers"""
        
        if not self.task_queue:
            return
        
        # Don't exceed max concurrent tasks
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return
        
        # Find available workers
        available_workers = [
            worker_id for worker_id, info in self.worker_info.items()
            if info.status == WorkerStatus.IDLE and self.workers[worker_id].is_available()
        ]
        
        if not available_workers:
            return
        
        # Assign tasks based on strategy
        tasks_to_assign = min(len(self.task_queue), len(available_workers))
        
        for _ in range(tasks_to_assign):
            if not self.task_queue:
                break
            
            task = self.task_queue.pop(0)
            worker_id = self.select_worker_for_task(task, available_workers)
            
            if worker_id:
                await self.assign_task_to_worker(task, worker_id)
                available_workers.remove(worker_id)
    
    def select_worker_for_task(self, task: Task, available_workers: List[str]) -> Optional[str]:
        """Select the best worker for a given task"""
        
        if self.task_assignment_strategy == "capability_match":
            # Find workers with matching capabilities
            capable_workers = []
            for worker_id in available_workers:
                worker_capabilities = self.worker_info[worker_id].capabilities
                if any(cap in worker_capabilities for cap in task.required_capabilities):
                    capable_workers.append(worker_id)
            
            if capable_workers:
                # Among capable workers, choose least loaded
                return min(capable_workers, 
                          key=lambda w: self.worker_info[w].tasks_completed)
            else:
                # Fallback to any available worker
                return available_workers[0] if available_workers else None
        
        elif self.task_assignment_strategy == "load_balance":
            # Choose worker with least completed tasks
            return min(available_workers, 
                      key=lambda w: self.worker_info[w].tasks_completed)
        
        elif self.task_assignment_strategy == "round_robin":
            # Simple round-robin assignment
            return available_workers[0] if available_workers else None
        
        return available_workers[0] if available_workers else None
    
    async def assign_task_to_worker(self, task: Task, worker_id: str) -> None:
        """Assign a specific task to a specific worker"""
        
        task.assigned_worker = worker_id
        task.status = TaskStatus.ASSIGNED
        task.started_at = time.time()
        
        # Update worker status
        worker_info = self.worker_info[worker_id]
        worker_info.status = WorkerStatus.BUSY
        worker_info.current_task = task.id
        
        # Move task to active tasks
        self.active_tasks[task.id] = task
        
        print(f"Assigned task {task.id} to worker {worker_id}")
        
        # Start task execution asynchronously
        asyncio.create_task(self.execute_task_on_worker(task, worker_id))
    
    async def execute_task_on_worker(self, task: Task, worker_id: str) -> None:
        """Execute a task on a specific worker"""
        
        try:
            task.status = TaskStatus.IN_PROGRESS
            worker = self.workers[worker_id]
            
            # Execute the task
            result = await worker.execute_task(task)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            # Update worker info
            worker_info = self.worker_info[worker_id]
            worker_info.tasks_completed += 1
            
            # Calculate average task time
            task_duration = task.completed_at - task.started_at
            if worker_info.average_task_time == 0:
                worker_info.average_task_time = task_duration
            else:
                worker_info.average_task_time = (
                    worker_info.average_task_time * 0.8 + task_duration * 0.2
                )
            
            print(f"Task {task.id} completed by worker {worker_id} in {task_duration:.2f}s")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completed_at = time.time()
            
            print(f"Task {task.id} failed on worker {worker_id}: {str(e)}")
            
            # Mark worker as error state temporarily
            self.worker_info[worker_id].status = WorkerStatus.ERROR
        
        finally:
            # Clean up worker state
            worker_info = self.worker_info[worker_id]
            worker_info.status = WorkerStatus.IDLE
            worker_info.current_task = None
            worker_info.last_heartbeat = time.time()
    
    async def check_completed_tasks(self) -> None:
        """Check for completed tasks and move them to appropriate collections"""
        
        completed_task_ids = []
        
        for task_id, task in self.active_tasks.items():
            if task.status == TaskStatus.COMPLETED:
                self.completed_tasks[task_id] = task
                completed_task_ids.append(task_id)
                self.total_tasks_processed += 1
            elif task.status == TaskStatus.FAILED:
                self.failed_tasks[task_id] = task
                completed_task_ids.append(task_id)
                
                # Optionally retry failed tasks
                if task.error_message and "retryable" in task.error_message.lower():
                    retry_task = Task(
                        id=f"{task_id}_retry",
                        task_type=task.task_type,
                        data=task.data,
                        priority=task.priority + 1,  # Higher priority for retries
                        estimated_duration=task.estimated_duration,
                        required_capabilities=task.required_capabilities
                    )
                    self.task_queue.append(retry_task)
                    print(f"Retrying failed task {task_id}")
        
        # Remove completed/failed tasks from active tasks
        for task_id in completed_task_ids:
            del self.active_tasks[task_id]
    
    async def handle_worker_failures(self) -> None:
        """Handle workers that have failed or become unresponsive"""
        
        current_time = time.time()
        
        for worker_id, worker_info in self.worker_info.items():
            # Check for unresponsive workers (no heartbeat for too long)
            if current_time - worker_info.last_heartbeat > 30:  # 30 seconds timeout
                
                if worker_info.status != WorkerStatus.OFFLINE:
                    print(f"Worker {worker_id} appears to be offline")
                    worker_info.status = WorkerStatus.OFFLINE
                    
                    # Reassign current task if any
                    if worker_info.current_task:
                        task = self.active_tasks.get(worker_info.current_task)
                        if task:
                            task.status = TaskStatus.PENDING
                            task.assigned_worker = None
                            self.task_queue.append(task)
                            del self.active_tasks[task.id]
                            worker_info.current_task = None
            
            # Recovery for error state workers
            elif worker_info.status == WorkerStatus.ERROR:
                # After a brief timeout, mark worker as available again
                if current_time - worker_info.last_heartbeat > 5:  # 5 seconds recovery
                    worker_info.status = WorkerStatus.IDLE
                    print(f"Worker {worker_id} recovered from error state")
    
    async def combine_results(self, job_type: str) -> Any:
        """Combine results from all completed tasks into final result"""
        
        if not self.completed_tasks:
            return {"error": "No completed tasks to combine"}
        
        if job_type == "data_processing":
            # Combine processed data chunks
            all_results = []
            for task in self.completed_tasks.values():
                if task.result and isinstance(task.result, list):
                    all_results.extend(task.result)
                elif task.result:
                    all_results.append(task.result)
            
            return {
                "processed_data": all_results,
                "total_items": len(all_results),
                "processing_summary": "Data processing completed successfully"
            }
        
        elif job_type == "content_generation":
            # Combine content pieces in logical order
            content_pieces = {}
            for task in self.completed_tasks.values():
                if task.result and isinstance(task.result, dict):
                    content_type = task.data.get("content_type", "unknown")
                    content_pieces[content_type] = task.result.get("content", "")
            
            # Combine in logical order
            order = ["introduction", "main_content", "conclusion", "summary"]
            combined_content = []
            for section in order:
                if section in content_pieces:
                    combined_content.append(content_pieces[section])
            
            return {
                "final_content": "\n\n".join(combined_content),
                "sections_completed": len(content_pieces),
                "total_length": sum(len(content) for content in content_pieces.values())
            }
        
        elif job_type == "image_processing":
            # Organize image processing results
            image_results = {}
            for task in self.completed_tasks.values():
                if task.result:
                    # Extract image index and operation from task ID
                    task_parts = task.id.split("_")
                    if len(task_parts) >= 3:
                        img_idx = task_parts[1]
                        operation = task_parts[2]
                        
                        if img_idx not in image_results:
                            image_results[img_idx] = {}
                        image_results[img_idx][operation] = task.result
            
            return {
                "processed_images": image_results,
                "images_processed": len(image_results),
                "operations_completed": sum(len(ops) for ops in image_results.values())
            }
        
        else:
            # Generic result combination
            results = [task.result for task in self.completed_tasks.values() if task.result]
            return {
                "combined_results": results,
                "total_results": len(results),
                "summary": "All tasks completed successfully"
            }
    
    def get_worker_utilization(self) -> Dict[str, Any]:
        """Get worker utilization statistics"""
        
        utilization = {}
        
        for worker_id, worker_info in self.worker_info.items():
            total_time = time.time() - self.system_start_time
            estimated_work_time = worker_info.tasks_completed * worker_info.average_task_time
            
            utilization[worker_id] = {
                "tasks_completed": worker_info.tasks_completed,
                "average_task_time": worker_info.average_task_time,
                "estimated_utilization": min(1.0, estimated_work_time / total_time) if total_time > 0 else 0,
                "current_status": worker_info.status.value
            }
        
        return utilization
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "system_id": self.system_id,
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.worker_info.values() if w.status != WorkerStatus.OFFLINE]),
            "tasks_in_queue": len(self.task_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks_processed": self.total_tasks_processed,
            "system_uptime": time.time() - self.system_start_time,
            "worker_utilization": self.get_worker_utilization()
        }

# EXAMPLE WORKER IMPLEMENTATIONS
# =============================

class DataProcessorWorker(Worker):
    """Worker specialized in data processing tasks"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.capabilities = ["data_processing", "general_processing"]
        self.is_busy = False
    
    def get_id(self) -> str:
        return self.worker_id
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    async def execute_task(self, task: Task) -> Any:
        """Process data chunks"""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        self.is_busy = True
        
        try:
            if task.task_type == "process_data_chunk":
                data_chunk = task.data
                
                # Simulate data processing
                processed_chunk = []
                for item in data_chunk:
                    processed_item = {
                        "original": item,
                        "processed": str(item).upper() if isinstance(item, str) else item * 2,
                        "processed_by": self.worker_id,
                        "timestamp": time.time()
                    }
                    processed_chunk.append(processed_item)
                
                return processed_chunk
            
            elif task.task_type == "process_subtask":
                # Generic subtask processing
                subtask_data = task.data
                return {
                    "subtask_index": subtask_data["subtask_index"],
                    "result": f"Processed subtask {subtask_data['subtask_index']} by {self.worker_id}",
                    "processing_time": 0.2
                }
            
            else:
                return {"error": f"Unknown task type: {task.task_type}"}
        
        finally:
            self.is_busy = False
    
    def is_available(self) -> bool:
        return not self.is_busy

class ContentGeneratorWorker(Worker):
    """Worker specialized in content generation"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.capabilities = ["text_generation", "content_writing", "general_processing"]
        self.is_busy = False
    
    def get_id(self) -> str:
        return self.worker_id
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    async def execute_task(self, task: Task) -> Any:
        """Generate content based on task requirements"""
        await asyncio.sleep(0.5)  # Simulate content generation time
        
        self.is_busy = True
        
        try:
            if task.task_type == "generate_content":
                content_type = task.data.get("content_type", "general")
                source_data = task.data.get("source_data", "")
                
                # Generate content based on type
                content_templates = {
                    "introduction": f"Introduction: This document provides an overview of {source_data}. The following sections will explore key concepts and applications.",
                    "main_content": f"Main Content: Detailed analysis of {source_data}. This includes comprehensive examination of methodologies, implementations, and best practices.",
                    "conclusion": f"Conclusion: In summary, {source_data} represents an important area with significant implications for future development.",
                    "summary": f"Summary: Key points about {source_data} include fundamental concepts, practical applications, and strategic considerations."
                }
                
                content = content_templates.get(content_type, f"General content about {source_data}")
                
                return {
                    "content_type": content_type,
                    "content": content,
                    "word_count": len(content.split()),
                    "generated_by": self.worker_id
                }
            
            else:
                return {"error": f"Unknown task type: {task.task_type}"}
        
        finally:
            self.is_busy = False
    
    def is_available(self) -> bool:
        return not self.is_busy

class ImageProcessorWorker(Worker):
    """Worker specialized in image processing"""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.capabilities = ["image_processing", "resize", "enhance", "analyze", "compress", "general_processing"]
        self.is_busy = False
    
    def get_id(self) -> str:
        return self.worker_id
    
    def get_capabilities(self) -> List[str]:
        return self.capabilities
    
    async def execute_task(self, task: Task) -> Any:
        """Process images based on operation type"""
        await asyncio.sleep(0.3)  # Simulate image processing time
        
        self.is_busy = True
        
        try:
            if task.task_type.startswith("image_"):
                operation = task.data.get("operation", "unknown")
                image_data = task.data.get("image", "image_placeholder")
                
                # Simulate different image operations
                results = {
                    "resize": f"Resized {image_data} to optimal dimensions",
                    "enhance": f"Enhanced {image_data} with improved contrast and brightness",
                    "analyze": f"Analyzed {image_data}: detected objects, quality score: 0.85",
                    "compress": f"Compressed {image_data} with 80% quality, reduced size by 60%"
                }
                
                return {
                    "operation": operation,
                    "result": results.get(operation, f"Unknown operation: {operation}"),
                    "processed_by": self.worker_id,
                    "processing_time": 0.3
                }
            
            else:
                return {"error": f"Unknown task type: {task.task_type}"}
        
        finally:
            self.is_busy = False
    
    def is_available(self) -> bool:
        return not self.is_busy

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_data_processing_system():
    """Demo: Large dataset processing using master-worker pattern"""
    print("\nDEMO 1: LARGE DATASET PROCESSING")
    print("=" * 50)
    
    # Create master-worker system
    system = MasterWorkerSystem("data_processing_system", max_concurrent_tasks=8)
    
    # Add multiple data processing workers
    for i in range(4):
        worker = DataProcessorWorker(f"data_worker_{i+1}")
        system.add_worker(worker)
    
    # Create large dataset to process
    large_dataset = [f"data_item_{i}" for i in range(50)]
    
    print(f"Processing dataset with {len(large_dataset)} items using {len(system.workers)} workers")
    
    # Process the large dataset
    result = await system.process_large_job(large_dataset, "data_processing")
    
    print(f"\nProcessing completed:")
    print(f"- Success rate: {result['success_rate']:.1%}")
    print(f"- Total time: {result['total_processing_time']:.2f} seconds")
    print(f"- Items processed: {result['final_result']['total_items']}")
    
    # Show worker utilization
    utilization = result['worker_utilization']
    print(f"\nWorker utilization:")
    for worker_id, stats in utilization.items():
        print(f"- {worker_id}: {stats['tasks_completed']} tasks, {stats['estimated_utilization']:.1%} utilized")

async def demo_content_creation_system():
    """Demo: Content creation using specialized workers"""
    print("\nDEMO 2: COLLABORATIVE CONTENT CREATION")
    print("=" * 50)
    
    # Create system for content generation
    system = MasterWorkerSystem("content_creation_system")
    
    # Add content generation workers
    for i in range(3):
        worker = ContentGeneratorWorker(f"content_worker_{i+1}")
        system.add_worker(worker)
    
    # Content creation job
    content_topic = "Artificial Intelligence in Healthcare"
    
    print(f"Creating comprehensive content about: {content_topic}")
    
    # Process content creation job
    result = await system.process_large_job(content_topic, "content_generation")
    
    print(f"\nContent creation completed:")
    print(f"- Sections completed: {result['final_result']['sections_completed']}")
    print(f"- Total length: {result['final_result']['total_length']} characters")
    print(f"- Success rate: {result['success_rate']:.1%}")
    
    # Show sample of generated content
    if result['final_result']['final_content']:
        content_preview = result['final_result']['final_content'][:200] + "..."
        print(f"\nContent preview:\n{content_preview}")

async def demo_mixed_workload_system():
    """Demo: System handling different types of work simultaneously"""
    print("\nDEMO 3: MIXED WORKLOAD PROCESSING")
    print("=" * 50)
    
    # Create system with multiple worker types
    system = MasterWorkerSystem("mixed_workload_system", max_concurrent_tasks=6)
    
    # Add different types of workers
    system.add_worker(DataProcessorWorker("data_specialist_1"))
    system.add_worker(DataProcessorWorker("data_specialist_2"))
    system.add_worker(ContentGeneratorWorker("content_specialist_1"))
    system.add_worker(ImageProcessorWorker("image_specialist_1"))
    
    # Create mixed workload
    mixed_data = [
        {"type": "text", "content": "Document 1"},
        {"type": "text", "content": "Document 2"},
        {"type": "image", "content": "image_001.jpg"},
        {"type": "image", "content": "image_002.jpg"},
        {"type": "data", "content": [1, 2, 3, 4, 5]},
        {"type": "data", "content": [6, 7, 8, 9, 10]}
    ]
    
    print(f"Processing mixed workload with {len(mixed_data)} items")
    
    # Process mixed workload
    result = await system.process_large_job(mixed_data, "data_processing")
    
    print(f"\nMixed workload completed:")
    print(f"- Total items processed: {len(result['final_result']['processed_data'])}")
    print(f"- Worker efficiency: {result['success_rate']:.1%}")
    
    # Show system status
    status = system.get_system_status()
    print(f"\nFinal system status:")
    print(f"- Active workers: {status['active_workers']}/{status['total_workers']}")
    print(f"- Total tasks processed: {status['total_tasks_processed']}")

async def main():
    """
    Demonstrate Master-Worker Pattern for coordinated distributed processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to break large jobs into smaller, manageable tasks
    2. How to coordinate multiple workers efficiently
    3. How to handle worker failures and load balancing
    4. How to combine results from distributed processing
    5. How master-worker pattern scales to handle large workloads
    
    REAL WORLD APPLICATIONS:
    =======================
    - MapReduce and distributed computing frameworks (Hadoop, Spark)
    - Web crawling and data processing pipelines
    - Video/image processing and rendering farms
    - Scientific computing and simulations
    - Content generation and publishing systems
    - E-commerce order processing and fulfillment
    """
    
    print("MASTER-WORKER PATTERN DEMONSTRATION")
    print("This shows how to coordinate multiple workers for large-scale processing!")
    
    await demo_data_processing_system()
    await demo_content_creation_system()
    await demo_mixed_workload_system()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Master-worker pattern enables efficient distributed processing")
    print("✓ Task decomposition allows parallel execution for faster completion")
    print("✓ Worker specialization improves efficiency and quality")
    print("✓ Centralized coordination prevents chaos and ensures completion")
    print("✓ Load balancing and failure handling maintain system reliability")
    print("\nTRY IT YOURSELF:")
    print("- Add more sophisticated task decomposition strategies")
    print("- Implement dynamic worker scaling based on workload")
    print("- Add real-time monitoring and performance optimization")
    print("- Create domain-specific worker types for your use cases")

if __name__ == "__main__":
    asyncio.run(main())
