#!/usr/bin/env python3
"""
Distributed Task Execution: Coordinated Processing Across Multiple Agents
=========================================================================

WHAT IS THE PROBLEM?
==================
Large tasks are too big for single agents to handle efficiently. Breaking them into smaller pieces and coordinating execution across multiple agents dramatically improves performance.

Example: Video Rendering Disaster
BAD APPROACH (Single Computer):
- Render 2-hour movie on one computer
- Takes 48 hours to complete
- Computer overheats and crashes
- Have to restart from beginning
- Movie deadline missed

REAL WORLD EXAMPLE:
=================
How does Pixar render movies like "Toy Story"?

PIXAR'S RENDER FARM:
1. Movie broken into individual frames (144,000 frames for 2-hour movie)
2. Each frame divided into small tiles
3. Thousands of computers work simultaneously
4. Each computer renders small pieces
5. Results combined into final movie
6. 48-hour job completes in 2 hours with 1000 computers

DISTRIBUTED EXECUTION BENEFITS:
- Massive speedup through parallelization
- Fault tolerance - if one computer fails, others continue
- Resource efficiency - use all available computing power
- Scalability - add more computers for faster processing

THE ALGORITHM:
=============
1. DECOMPOSE: Break large task into smaller independent subtasks
2. DISTRIBUTE: Assign subtasks to available agents
3. COORDINATE: Track progress and manage dependencies
4. MONITOR: Watch for failures and bottlenecks
5. AGGREGATE: Combine results from all agents
6. DELIVER: Provide final result to user

WHY IS THIS REVOLUTIONARY?
========================
- Transforms impossible tasks into achievable ones
- Utilizes distributed computing resources efficiently
- Provides fault tolerance and resilience
- Scales performance with available resources
- Enables real-time processing of massive data
"""

import asyncio
import time
import random
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"

@dataclass
class DistributedTask:
    """A task that can be distributed across multiple agents"""
    id: str
    description: str
    data: Any
    estimated_duration: float
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['DistributedTask'] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Any = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class AgentCapacity:
    """Agent's current capacity and capabilities"""
    agent_id: str
    max_concurrent_tasks: int
    current_tasks: int
    processing_power: float  # Relative processing capability
    specializations: List[str]
    reliability: float  # 0.0 to 1.0
    average_task_time: float

class DistributedWorker:
    """Worker agent that executes distributed tasks"""
    
    def __init__(self, agent_id: str, processing_power: float = 1.0, specializations: List[str] = None):
        self.agent_id = agent_id
        self.processing_power = processing_power
        self.specializations = specializations or ["general"]
        
        # Capacity management
        self.max_concurrent_tasks = max(1, int(processing_power * 3))
        self.current_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: List[DistributedTask] = []
        
        # Performance tracking
        self.total_execution_time = 0.0
        self.task_success_rate = 1.0
        self.average_task_duration = 1.0
        
        # Agent state
        self.status = AgentStatus.IDLE
        self.last_heartbeat = time.time()
    
    async def execute_task(self, task: DistributedTask) -> Any:
        """Execute a distributed task"""
        
        print(f"  {self.agent_id} executing: {task.description}")
        
        # Update status
        self.status = AgentStatus.BUSY if len(self.current_tasks) < self.max_concurrent_tasks else AgentStatus.OVERLOADED
        self.current_tasks[task.id] = task
        
        try:
            # Start task execution
            task.status = TaskStatus.IN_PROGRESS
            task.start_time = time.time()
            
            # Simulate task execution based on type
            result = await self.process_task_data(task)
            
            # Complete task
            task.status = TaskStatus.COMPLETED
            task.completion_time = time.time()
            task.result = result
            
            # Update performance metrics
            execution_time = task.completion_time - task.start_time
            self.total_execution_time += execution_time
            self.update_performance_metrics(execution_time, True)
            
            print(f"    {self.agent_id} completed {task.id} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.completion_time = time.time()
            
            execution_time = task.completion_time - task.start_time if task.start_time else 0
            self.update_performance_metrics(execution_time, False)
            
            print(f"    {self.agent_id} failed {task.id}: {str(e)}")
            raise e
            
        finally:
            # Clean up
            if task.id in self.current_tasks:
                del self.current_tasks[task.id]
            self.completed_tasks.append(task)
            
            # Update status
            self.status = AgentStatus.IDLE if len(self.current_tasks) == 0 else AgentStatus.BUSY
            self.last_heartbeat = time.time()
    
    async def process_task_data(self, task: DistributedTask) -> Any:
        """Process task data based on task type"""
        
        # Simulate processing time based on task complexity and agent capability
        base_duration = task.estimated_duration
        adjusted_duration = base_duration / self.processing_power
        
        # Add some randomness to simulate real-world variability
        actual_duration = adjusted_duration * random.uniform(0.8, 1.2)
        
        # Check for specialization bonus
        task_type = task.description.split()[0].lower() if task.description else "general"
        if task_type in self.specializations:
            actual_duration *= 0.7  # 30% faster for specialized tasks
        
        await asyncio.sleep(actual_duration)
        
        # Simulate different types of processing
        if "compute" in task.description.lower():
            return await self.compute_processing(task)
        elif "data" in task.description.lower():
            return await self.data_processing(task)
        elif "render" in task.description.lower():
            return await self.render_processing(task)
        else:
            return await self.general_processing(task)
    
    async def compute_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Process computational tasks"""
        
        # Simulate mathematical computation
        data_size = len(str(task.data)) if task.data else 100
        operations = data_size * 1000
        
        return {
            "task_id": task.id,
            "operations_performed": operations,
            "computation_result": random.uniform(0, 1000),
            "processed_by": self.agent_id,
            "processing_time": time.time() - task.start_time if task.start_time else 0
        }
    
    async def data_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Process data analysis tasks"""
        
        # Simulate data analysis
        if isinstance(task.data, list):
            processed_data = [item * 2 if isinstance(item, (int, float)) else str(item).upper() 
                            for item in task.data]
        else:
            processed_data = f"Processed: {task.data}"
        
        return {
            "task_id": task.id,
            "original_data": task.data,
            "processed_data": processed_data,
            "data_points": len(task.data) if isinstance(task.data, list) else 1,
            "processed_by": self.agent_id
        }
    
    async def render_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Process rendering tasks"""
        
        # Simulate rendering (like video frames)
        frame_data = task.data if task.data else {"frame": 1, "quality": "HD"}
        
        return {
            "task_id": task.id,
            "rendered_frame": frame_data,
            "render_quality": "high",
            "pixels_processed": 1920 * 1080,  # HD resolution
            "render_time": time.time() - task.start_time if task.start_time else 0,
            "processed_by": self.agent_id
        }
    
    async def general_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Process general tasks"""
        
        return {
            "task_id": task.id,
            "description": task.description,
            "result": f"Task processed successfully by {self.agent_id}",
            "processing_method": "general",
            "data_processed": task.data
        }
    
    def update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """Update agent performance metrics"""
        
        # Update average task duration
        total_tasks = len(self.completed_tasks)
        if total_tasks > 0:
            self.average_task_duration = self.total_execution_time / total_tasks
        
        # Update success rate
        successful_tasks = len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED])
        self.task_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 1.0
    
    def can_accept_task(self, task: DistributedTask) -> bool:
        """Check if agent can accept a new task"""
        
        # Check capacity
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check if agent is online
        if self.status == AgentStatus.OFFLINE:
            return False
        
        return True
    
    def get_capacity_info(self) -> AgentCapacity:
        """Get current capacity information"""
        
        return AgentCapacity(
            agent_id=self.agent_id,
            max_concurrent_tasks=self.max_concurrent_tasks,
            current_tasks=len(self.current_tasks),
            processing_power=self.processing_power,
            specializations=self.specializations,
            reliability=self.task_success_rate,
            average_task_time=self.average_task_duration
        )

class DistributedExecutionSystem:
    """
    System for coordinating distributed task execution
    
    EXAMPLE USAGE:
    =============
    # Create distributed execution system
    system = DistributedExecutionSystem("video_rendering")
    
    # Add worker agents
    for i in range(10):
        worker = DistributedWorker(f"worker_{i}", processing_power=random.uniform(0.5, 2.0))
        system.add_worker(worker)
    
    # Execute large distributed task
    result = await system.execute_distributed_task("Render 1000-frame animation")
    """
    
    def __init__(self, system_id: str):
        self.system_id = system_id
        self.workers: Dict[str, DistributedWorker] = {}
        
        # Task management
        self.task_queue: List[DistributedTask] = []
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: List[DistributedTask] = []
        self.failed_tasks: List[DistributedTask] = []
        
        # System state
        self.total_tasks_processed = 0
        self.system_throughput = 0.0
        self.load_balancing_strategy = "least_loaded"  # or "round_robin", "capability_match"
    
    def add_worker(self, worker: DistributedWorker) -> None:
        """Add worker to the system"""
        self.workers[worker.agent_id] = worker
        print(f"Added worker: {worker.agent_id} (power: {worker.processing_power:.1f})")
    
    async def execute_distributed_task(self, task_description: str, 
                                     task_data: Any = None, 
                                     subtask_count: int = None) -> Dict[str, Any]:
        """Execute a large task by distributing it across workers"""
        
        print(f"\nEXECUTING DISTRIBUTED TASK: {task_description}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Create main task
        main_task = DistributedTask(
            id=f"main_{uuid.uuid4().hex[:8]}",
            description=task_description,
            data=task_data,
            estimated_duration=10.0  # Will be distributed
        )
        
        # Break down into subtasks
        subtasks = await self.decompose_task(main_task, subtask_count)
        
        print(f"Decomposed into {len(subtasks)} subtasks")
        
        # Execute subtasks in parallel
        results = await self.execute_subtasks_parallel(subtasks)
        
        # Combine results
        final_result = await self.combine_results(main_task, results)
        
        execution_time = time.time() - start_time
        
        # Update system metrics
        self.total_tasks_processed += len(subtasks)
        self.system_throughput = len(subtasks) / execution_time if execution_time > 0 else 0
        
        print(f"\nDistributed execution completed in {execution_time:.2f} seconds")
        print(f"System throughput: {self.system_throughput:.2f} tasks/second")
        
        return {
            "main_task": main_task.description,
            "execution_time": execution_time,
            "subtasks_completed": len(results),
            "system_throughput": self.system_throughput,
            "final_result": final_result,
            "worker_utilization": self.get_worker_utilization()
        }
    
    async def decompose_task(self, main_task: DistributedTask, 
                           subtask_count: int = None) -> List[DistributedTask]:
        """Decompose large task into smaller subtasks"""
        
        if subtask_count is None:
            # Automatically determine subtask count based on available workers
            subtask_count = len(self.workers) * 2  # 2 tasks per worker
        
        subtasks = []
        
        # Create subtasks based on task type
        if "render" in main_task.description.lower():
            # Video/image rendering - break into frames
            for i in range(subtask_count):
                subtask = DistributedTask(
                    id=f"render_frame_{i}",
                    description=f"Render frame {i}",
                    data={"frame_number": i, "total_frames": subtask_count},
                    estimated_duration=main_task.estimated_duration / subtask_count
                )
                subtasks.append(subtask)
        
        elif "compute" in main_task.description.lower():
            # Computational task - break into chunks
            for i in range(subtask_count):
                subtask = DistributedTask(
                    id=f"compute_chunk_{i}",
                    description=f"Compute chunk {i}",
                    data={"chunk_id": i, "total_chunks": subtask_count, "data_range": (i*100, (i+1)*100)},
                    estimated_duration=main_task.estimated_duration / subtask_count
                )
                subtasks.append(subtask)
        
        elif "data" in main_task.description.lower():
            # Data processing - break data into segments
            if isinstance(main_task.data, list):
                chunk_size = max(1, len(main_task.data) // subtask_count)
                for i in range(0, len(main_task.data), chunk_size):
                    chunk = main_task.data[i:i+chunk_size]
                    subtask = DistributedTask(
                        id=f"data_segment_{i // chunk_size}",
                        description=f"Process data segment {i // chunk_size}",
                        data=chunk,
                        estimated_duration=main_task.estimated_duration / subtask_count
                    )
                    subtasks.append(subtask)
            else:
                # Generic data breakdown
                for i in range(subtask_count):
                    subtask = DistributedTask(
                        id=f"data_part_{i}",
                        description=f"Process data part {i}",
                        data=f"Part {i} of {main_task.data}",
                        estimated_duration=main_task.estimated_duration / subtask_count
                    )
                    subtasks.append(subtask)
        
        else:
            # Generic task breakdown
            for i in range(subtask_count):
                subtask = DistributedTask(
                    id=f"subtask_{i}",
                    description=f"Execute subtask {i}",
                    data=f"Subtask {i} data",
                    estimated_duration=main_task.estimated_duration / subtask_count
                )
                subtasks.append(subtask)
        
        return subtasks
    
    async def execute_subtasks_parallel(self, subtasks: List[DistributedTask]) -> List[Dict[str, Any]]:
        """Execute subtasks in parallel across available workers"""
        
        # Assign tasks to workers
        task_assignments = self.assign_tasks_to_workers(subtasks)
        
        # Execute tasks in parallel
        execution_tasks = []
        for worker_id, assigned_tasks in task_assignments.items():
            worker = self.workers[worker_id]
            for task in assigned_tasks:
                execution_task = worker.execute_task(task)
                execution_tasks.append(execution_task)
        
        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results and handle failures
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Task failed: {result}")
                    # Retry failed tasks if possible
                    failed_task = subtasks[i]
                    if failed_task.retry_count < failed_task.max_retries:
                        await self.retry_failed_task(failed_task)
                else:
                    successful_results.append(result)
            
            return successful_results
            
        except Exception as e:
            print(f"Error in parallel execution: {e}")
            return []
    
    def assign_tasks_to_workers(self, tasks: List[DistributedTask]) -> Dict[str, List[DistributedTask]]:
        """Assign tasks to workers based on load balancing strategy"""
        
        assignments = {worker_id: [] for worker_id in self.workers.keys()}
        
        if self.load_balancing_strategy == "round_robin":
            # Simple round-robin assignment
            for i, task in enumerate(tasks):
                worker_ids = list(self.workers.keys())
                worker_id = worker_ids[i % len(worker_ids)]
                assignments[worker_id].append(task)
        
        elif self.load_balancing_strategy == "least_loaded":
            # Assign to workers with least current load
            for task in tasks:
                # Find worker with least current tasks
                best_worker = min(self.workers.values(), 
                                key=lambda w: len(w.current_tasks))
                assignments[best_worker.agent_id].append(task)
        
        elif self.load_balancing_strategy == "capability_match":
            # Assign based on worker capabilities
            for task in tasks:
                # Find best worker for this task type
                task_type = task.description.split()[0].lower()
                
                suitable_workers = [w for w in self.workers.values() 
                                  if task_type in w.specializations or "general" in w.specializations]
                
                if suitable_workers:
                    best_worker = min(suitable_workers, 
                                    key=lambda w: len(w.current_tasks))
                else:
                    best_worker = min(self.workers.values(), 
                                    key=lambda w: len(w.current_tasks))
                
                assignments[best_worker.agent_id].append(task)
        
        return assignments
    
    async def retry_failed_task(self, task: DistributedTask) -> None:
        """Retry a failed task on a different worker"""
        
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        
        print(f"Retrying task {task.id} (attempt {task.retry_count})")
        
        # Find different worker for retry
        available_workers = [w for w in self.workers.values() if w.can_accept_task(task)]
        
        if available_workers:
            retry_worker = random.choice(available_workers)
            try:
                await retry_worker.execute_task(task)
            except Exception as e:
                print(f"Retry failed for task {task.id}: {e}")
                if task.retry_count >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    self.failed_tasks.append(task)
    
    async def combine_results(self, main_task: DistributedTask, 
                            subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from all subtasks"""
        
        combined_result = {
            "main_task_id": main_task.id,
            "main_task_description": main_task.description,
            "total_subtasks": len(subtask_results),
            "successful_subtasks": len([r for r in subtask_results if r]),
            "combined_data": []
        }
        
        # Combine based on task type
        if "render" in main_task.description.lower():
            # Combine rendered frames
            frames = sorted([r for r in subtask_results if r and "rendered_frame" in r],
                          key=lambda x: x["rendered_frame"].get("frame_number", 0))
            combined_result["rendered_frames"] = frames
            combined_result["total_pixels"] = sum(r.get("pixels_processed", 0) for r in frames)
        
        elif "compute" in main_task.description.lower():
            # Combine computational results
            computations = [r for r in subtask_results if r and "computation_result" in r]
            combined_result["computation_results"] = computations
            combined_result["total_operations"] = sum(r.get("operations_performed", 0) for r in computations)
            combined_result["aggregate_result"] = sum(r.get("computation_result", 0) for r in computations)
        
        elif "data" in main_task.description.lower():
            # Combine processed data
            data_results = [r for r in subtask_results if r and "processed_data" in r]
            combined_result["processed_datasets"] = data_results
            combined_result["total_data_points"] = sum(r.get("data_points", 0) for r in data_results)
            
            # Flatten processed data
            all_processed_data = []
            for result in data_results:
                processed_data = result.get("processed_data", [])
                if isinstance(processed_data, list):
                    all_processed_data.extend(processed_data)
                else:
                    all_processed_data.append(processed_data)
            
            combined_result["combined_data"] = all_processed_data
        
        return combined_result
    
    def get_worker_utilization(self) -> Dict[str, Any]:
        """Get worker utilization statistics"""
        
        utilization = {}
        
        for worker_id, worker in self.workers.items():
            capacity_info = worker.get_capacity_info()
            utilization_rate = capacity_info.current_tasks / capacity_info.max_concurrent_tasks
            
            utilization[worker_id] = {
                "utilization_rate": utilization_rate,
                "current_tasks": capacity_info.current_tasks,
                "max_capacity": capacity_info.max_concurrent_tasks,
                "processing_power": capacity_info.processing_power,
                "reliability": capacity_info.reliability,
                "specializations": capacity_info.specializations
            }
        
        return utilization
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        active_workers = len([w for w in self.workers.values() if w.status != AgentStatus.OFFLINE])
        total_capacity = sum(w.max_concurrent_tasks for w in self.workers.values())
        current_load = sum(len(w.current_tasks) for w in self.workers.values())
        
        return {
            "system_id": self.system_id,
            "total_workers": len(self.workers),
            "active_workers": active_workers,
            "total_capacity": total_capacity,
            "current_load": current_load,
            "system_utilization": current_load / total_capacity if total_capacity > 0 else 0,
            "tasks_processed": self.total_tasks_processed,
            "system_throughput": self.system_throughput,
            "failed_tasks": len(self.failed_tasks)
        }

# EXAMPLE DEMONSTRATIONS
# =====================

async def demo_video_rendering_farm():
    """Demo: Distributed video rendering like Pixar"""
    print("\nDEMO 1: VIDEO RENDERING FARM")
    print("=" * 50)
    
    render_farm = DistributedExecutionSystem("pixar_render_farm")
    
    # Add rendering nodes with different capabilities
    render_nodes = [
        ("gpu_node_1", 3.0, ["render", "compute"]),
        ("gpu_node_2", 2.8, ["render", "compute"]),
        ("cpu_node_1", 1.5, ["render", "general"]),
        ("cpu_node_2", 1.2, ["render", "general"]),
        ("workstation_1", 2.0, ["render", "compute"]),
        ("workstation_2", 1.8, ["render", "compute"])
    ]
    
    for node_id, power, specializations in render_nodes:
        worker = DistributedWorker(node_id, power, specializations)
        render_farm.add_worker(worker)
    
    # Render a full animation sequence
    animation_data = [{"scene": i, "quality": "4K"} for i in range(100)]
    
    result = await render_farm.execute_distributed_task(
        "Render 100-frame 4K animation sequence",
        animation_data,
        subtask_count=50  # 50 render tasks
    )
    
    print(f"\nVideo Rendering Results:")
    print(f"- Frames rendered: {result['subtasks_completed']}")
    print(f"- Total render time: {result['execution_time']:.2f} seconds")
    print(f"- Rendering speed: {result['system_throughput']:.1f} frames/second")
    
    # Show worker performance
    utilization = result['worker_utilization']
    best_performer = max(utilization.items(), key=lambda x: x[1]['processing_power'])
    print(f"- Best performer: {best_performer[0]} ({best_performer[1]['processing_power']:.1f}x power)")

async def demo_scientific_computing():
    """Demo: Distributed scientific computation"""
    print("\nDEMO 2: SCIENTIFIC COMPUTING CLUSTER")
    print("=" * 50)
    
    compute_cluster = DistributedExecutionSystem("scientific_cluster")
    
    # Add compute nodes
    compute_nodes = [
        ("hpc_node_1", 4.0, ["compute", "analysis"]),
        ("hpc_node_2", 3.8, ["compute", "analysis"]),
        ("server_1", 2.5, ["compute", "general"]),
        ("server_2", 2.2, ["compute", "general"]),
        ("cloud_instance_1", 1.8, ["compute", "general"]),
        ("cloud_instance_2", 1.5, ["compute", "general"])
    ]
    
    for node_id, power, specializations in compute_nodes:
        worker = DistributedWorker(node_id, power, specializations)
        compute_cluster.add_worker(worker)
    
    # Execute large-scale computation
    result = await compute_cluster.execute_distributed_task(
        "Compute climate simulation with 1 million data points",
        {"simulation_type": "climate", "data_points": 1000000},
        subtask_count=20
    )
    
    print(f"\nScientific Computing Results:")
    print(f"- Computation chunks: {result['subtasks_completed']}")
    print(f"- Total computation time: {result['execution_time']:.2f} seconds")
    print(f"- Processing throughput: {result['system_throughput']:.1f} tasks/second")

async def demo_big_data_processing():
    """Demo: Distributed big data processing"""
    print("\nDEMO 3: BIG DATA PROCESSING PIPELINE")
    print("=" * 50)
    
    data_pipeline = DistributedExecutionSystem("big_data_cluster")
    
    # Add data processing workers
    data_workers = [
        ("spark_worker_1", 2.5, ["data", "analysis"]),
        ("spark_worker_2", 2.3, ["data", "analysis"]),
        ("hadoop_node_1", 1.8, ["data", "storage"]),
        ("hadoop_node_2", 1.6, ["data", "storage"]),
        ("analytics_server", 3.0, ["data", "analysis", "compute"])
    ]
    
    for worker_id, power, specializations in data_workers:
        worker = DistributedWorker(worker_id, power, specializations)
        data_pipeline.add_worker(worker)
    
    # Process large dataset
    large_dataset = list(range(10000))  # Simulate large dataset
    
    result = await data_pipeline.execute_distributed_task(
        "Process 10,000-record customer dataset for analytics",
        large_dataset,
        subtask_count=15
    )
    
    print(f"\nBig Data Processing Results:")
    print(f"- Data segments processed: {result['subtasks_completed']}")
    print(f"- Processing time: {result['execution_time']:.2f} seconds")
    print(f"- Data throughput: {len(large_dataset) / result['execution_time']:.0f} records/second")
    
    # Show system status
    status = data_pipeline.get_system_status()
    print(f"- System utilization: {status['system_utilization']:.1%}")

async def main():
    """
    Demonstrate Distributed Task Execution for scalable processing
    
    WHAT YOU'LL LEARN:
    ================
    1. How to decompose large tasks into distributable subtasks
    2. How to coordinate execution across multiple worker agents
    3. How to implement load balancing and fault tolerance
    4. How to combine results from distributed processing
    5. How distributed systems achieve massive performance gains
    
    REAL WORLD APPLICATIONS:
    =======================
    - Movie and animation rendering farms (Pixar, DreamWorks)
    - Scientific computing and simulations (weather, climate)
    - Big data processing platforms (Hadoop, Spark)
    - Distributed machine learning training
    - Cryptocurrency mining and blockchain processing
    - Content delivery networks and edge computing
    """
    
    print("DISTRIBUTED TASK EXECUTION DEMONSTRATION")
    print("This shows how to coordinate massive parallel processing across multiple agents!")
    
    await demo_video_rendering_farm()
    await demo_scientific_computing()
    await demo_big_data_processing()
    
    print("\nWHAT WE LEARNED:")
    print("=" * 40)
    print("✓ Task decomposition enables massive parallel processing")
    print("✓ Load balancing optimizes resource utilization")
    print("✓ Fault tolerance ensures system reliability")
    print("✓ Coordination mechanisms manage complex distributed workflows")
    print("✓ Distributed execution achieves dramatic performance improvements")

if __name__ == "__main__":
    asyncio.run(main())
