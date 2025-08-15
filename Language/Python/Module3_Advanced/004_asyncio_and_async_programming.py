"""
Python Asyncio and Async Programming: Event Loops, Coroutines, and Concurrent Execution
Implementation-focused with minimal comments, maximum functionality coverage
"""

import asyncio
import aiohttp
import aiofiles
import time
import random
import concurrent.futures
from typing import AsyncGenerator, Awaitable, List, Dict, Any, Optional, Callable
import json
import threading
from dataclasses import dataclass
from contextlib import asynccontextmanager
import weakref
import signal
import functools

# Basic async/await patterns
async def simple_async_function():
    """Basic async function"""
    await asyncio.sleep(0.1)
    return "Hello from async function"

async def async_generator():
    """Async generator function"""
    for i in range(5):
        await asyncio.sleep(0.01)
        yield f"Item {i}"

async def async_context_manager_example():
    """Using async context managers"""
    async with aiofiles.open(__file__, 'r') as f:
        first_line = await f.readline()
        return first_line.strip()

async def basic_async_demo():
    # Simple async function call
    result1 = await simple_async_function()
    
    # Async generator consumption
    generator_results = []
    async for item in async_generator():
        generator_results.append(item)
    
    # Async context manager
    try:
        file_result = await async_context_manager_example()
    except Exception as e:
        file_result = f"File error: {e}"
    
    return {
        "simple_function": result1,
        "async_generator": generator_results,
        "file_result": file_result[:50] + "..." if len(file_result) > 50 else file_result
    }

# Event loop management
def event_loop_demo():
    """Demonstrate event loop operations"""
    
    async def async_task(name, delay):
        print(f"Task {name} starting")
        await asyncio.sleep(delay)
        print(f"Task {name} completed")
        return f"Result from {name}"
    
    async def run_tasks():
        # Create tasks
        task1 = asyncio.create_task(async_task("A", 0.1))
        task2 = asyncio.create_task(async_task("B", 0.05))
        task3 = asyncio.create_task(async_task("C", 0.15))
        
        # Wait for all tasks
        results = await asyncio.gather(task1, task2, task3)
        return results
    
    # Run with different methods
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Method 1: loop.run_until_complete
        start_time = time.time()
        results1 = loop.run_until_complete(run_tasks())
        time1 = time.time() - start_time
        
        # Method 2: asyncio.run (resets loop)
        start_time = time.time()
        results2 = asyncio.run(run_tasks())
        time2 = time.time() - start_time
        
        return {
            "method1_results": results1,
            "method2_results": results2,
            "timing": {
                "run_until_complete": f"{time1:.4f}s",
                "asyncio_run": f"{time2:.4f}s"
            }
        }
    finally:
        loop.close()

# Concurrent execution patterns
async def concurrent_execution_demo():
    """Demonstrate different concurrent execution patterns"""
    
    async def fetch_data(url, delay):
        """Simulate API call"""
        await asyncio.sleep(delay)
        return {
            "url": url,
            "data": f"Data from {url}",
            "delay": delay
        }
    
    urls = [
        ("http://api1.com", 0.1),
        ("http://api2.com", 0.05),
        ("http://api3.com", 0.15),
        ("http://api4.com", 0.08),
        ("http://api5.com", 0.12)
    ]
    
    # Pattern 1: gather() - wait for all
    start_time = time.time()
    gather_results = await asyncio.gather(*[fetch_data(url, delay) for url, delay in urls])
    gather_time = time.time() - start_time
    
    # Pattern 2: as_completed() - process as they complete
    start_time = time.time()
    completed_results = []
    tasks = [fetch_data(url, delay) for url, delay in urls]
    
    for completed_task in asyncio.as_completed(tasks):
        result = await completed_task
        completed_results.append(result)
    
    completed_time = time.time() - start_time
    
    # Pattern 3: wait() with timeout
    start_time = time.time()
    tasks = [fetch_data(url, delay) for url, delay in urls]
    done, pending = await asyncio.wait(tasks, timeout=0.1)
    
    wait_results = [await task for task in done]
    for task in pending:
        task.cancel()
    
    wait_time = time.time() - start_time
    
    return {
        "gather_pattern": {
            "results": len(gather_results),
            "time": f"{gather_time:.4f}s"
        },
        "as_completed_pattern": {
            "results": len(completed_results),
            "time": f"{completed_time:.4f}s"
        },
        "wait_with_timeout": {
            "completed": len(wait_results),
            "pending_cancelled": len(pending),
            "time": f"{wait_time:.4f}s"
        }
    }

# Task management and cancellation
async def task_management_demo():
    """Demonstrate task creation, management, and cancellation"""
    
    async def long_running_task(name, duration):
        """Task that can be cancelled"""
        try:
            for i in range(int(duration * 10)):
                await asyncio.sleep(0.1)
                print(f"Task {name}: step {i+1}")
            return f"Task {name} completed normally"
        except asyncio.CancelledError:
            print(f"Task {name} was cancelled")
            return f"Task {name} cancelled"
    
    async def task_with_cleanup(name):
        """Task with proper cleanup"""
        try:
            await asyncio.sleep(0.5)
            return f"Task {name} finished"
        except asyncio.CancelledError:
            print(f"Cleaning up task {name}")
            await asyncio.sleep(0.1)  # Cleanup work
            raise
    
    # Create multiple tasks
    task1 = asyncio.create_task(long_running_task("T1", 1.0))
    task2 = asyncio.create_task(long_running_task("T2", 0.5))
    task3 = asyncio.create_task(task_with_cleanup("T3"))
    
    # Let tasks run for a bit
    await asyncio.sleep(0.2)
    
    # Cancel some tasks
    task1.cancel()
    task3.cancel()
    
    # Collect results
    results = []
    for task in [task1, task2, task3]:
        try:
            result = await task
            results.append({"status": "completed", "result": result})
        except asyncio.CancelledError:
            results.append({"status": "cancelled", "result": "Task was cancelled"})
    
    return {
        "task_results": results,
        "cancellation_demo": "Completed"
    }

# Async context managers and async iterators
class AsyncResource:
    """Async context manager example"""
    
    def __init__(self, name):
        self.name = name
        self.acquired = False
    
    async def __aenter__(self):
        print(f"Acquiring resource: {self.name}")
        await asyncio.sleep(0.01)  # Simulate async acquisition
        self.acquired = True
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        print(f"Releasing resource: {self.name}")
        await asyncio.sleep(0.01)  # Simulate async cleanup
        self.acquired = False
        return False

class AsyncRange:
    """Async iterator example"""
    
    def __init__(self, start, stop, delay=0.01):
        self.start = start
        self.stop = stop
        self.delay = delay
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.start >= self.stop:
            raise StopAsyncIteration
        
        await asyncio.sleep(self.delay)
        current = self.start
        self.start += 1
        return current

async def async_protocols_demo():
    """Demonstrate async context managers and iterators"""
    
    # Async context manager
    async with AsyncResource("Database") as db:
        resource_status = f"Resource acquired: {db.acquired}"
    
    # Async iterator
    async_range_results = []
    async for value in AsyncRange(0, 5, 0.01):
        async_range_results.append(value)
    
    # Async generator with context manager
    async def async_file_reader():
        async with AsyncResource("FileReader") as reader:
            for i in range(3):
                await asyncio.sleep(0.01)
                yield f"Line {i} from {reader.name}"
    
    file_lines = []
    async for line in async_file_reader():
        file_lines.append(line)
    
    return {
        "resource_status": resource_status,
        "async_range": async_range_results,
        "file_lines": file_lines
    }

# Exception handling in async code
async def exception_handling_demo():
    """Demonstrate exception handling in async code"""
    
    async def failing_task(fail_after=0.1):
        await asyncio.sleep(fail_after)
        raise ValueError("Task failed intentionally")
    
    async def timeout_task():
        await asyncio.sleep(1.0)
        return "This shouldn't complete"
    
    async def successful_task():
        await asyncio.sleep(0.05)
        return "Success!"
    
    results = {}
    
    # Handle individual task exception
    try:
        await failing_task(0.05)
    except ValueError as e:
        results["individual_exception"] = f"Caught: {e}"
    
    # Handle timeout
    try:
        result = await asyncio.wait_for(timeout_task(), timeout=0.1)
    except asyncio.TimeoutError:
        results["timeout_exception"] = "Task timed out"
    
    # Handle exceptions in gather
    try:
        await asyncio.gather(
            successful_task(),
            failing_task(0.1),
            return_exceptions=False
        )
    except ValueError as e:
        results["gather_exception"] = f"Gather failed: {e}"
    
    # Gather with return_exceptions=True
    gather_results = await asyncio.gather(
        successful_task(),
        failing_task(0.1),
        return_exceptions=True
    )
    
    results["gather_with_exceptions"] = [
        str(r) if isinstance(r, Exception) else r 
        for r in gather_results
    ]
    
    return results

# Producer-consumer patterns
async def producer_consumer_demo():
    """Demonstrate producer-consumer pattern with asyncio.Queue"""
    
    async def producer(queue, name, items):
        """Produce items and put them in queue"""
        for i in range(items):
            item = f"{name}-item-{i}"
            await queue.put(item)
            print(f"Produced: {item}")
            await asyncio.sleep(0.01)
        
        await queue.put(None)  # Sentinel to signal completion
        print(f"Producer {name} finished")
    
    async def consumer(queue, name):
        """Consume items from queue"""
        consumed = []
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            
            consumed.append(item)
            print(f"Consumer {name} got: {item}")
            await asyncio.sleep(0.02)  # Simulate processing
            queue.task_done()
        
        print(f"Consumer {name} finished")
        return consumed
    
    # Create queue and tasks
    queue = asyncio.Queue(maxsize=3)
    
    # Start producer and consumer
    producer_task = asyncio.create_task(producer(queue, "P1", 5))
    consumer_task = asyncio.create_task(consumer(queue, "C1"))
    
    # Wait for completion
    await producer_task
    consumed_items = await consumer_task
    
    return {
        "items_consumed": consumed_items,
        "pattern": "producer-consumer completed"
    }

# Async HTTP client example (simplified)
async def http_client_demo():
    """Simulate async HTTP client operations"""
    
    class MockHttpResponse:
        def __init__(self, url, status=200):
            self.url = url
            self.status = status
            self.data = f"Response from {url}"
    
    async def fetch_url(session, url):
        """Simulate HTTP request"""
        delay = random.uniform(0.05, 0.15)
        await asyncio.sleep(delay)
        
        # Simulate occasional failures
        if random.random() < 0.1:  # 10% failure rate
            return MockHttpResponse(url, 500)
        
        return MockHttpResponse(url, 200)
    
    async def fetch_multiple_urls(urls):
        """Fetch multiple URLs concurrently"""
        session = "mock_session"  # In real code, this would be aiohttp.ClientSession()
        
        tasks = [fetch_url(session, url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append({"error": str(response)})
            else:
                results.append({
                    "url": response.url,
                    "status": response.status,
                    "success": response.status == 200
                })
        
        return results
    
    urls = [
        "http://api1.example.com",
        "http://api2.example.com", 
        "http://api3.example.com",
        "http://api4.example.com",
        "http://api5.example.com"
    ]
    
    results = await fetch_multiple_urls(urls)
    
    return {
        "http_requests": results,
        "successful_requests": sum(1 for r in results if r.get("success", False)),
        "total_requests": len(results)
    }

# Async background tasks and services
class AsyncService:
    """Background async service example"""
    
    def __init__(self, name):
        self.name = name
        self.running = False
        self.task = None
        self.stats = {"processed": 0, "errors": 0}
    
    async def start(self):
        """Start the background service"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run())
        print(f"Service {self.name} started")
    
    async def stop(self):
        """Stop the background service"""
        if not self.running:
            return
        
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        print(f"Service {self.name} stopped")
    
    async def _run(self):
        """Main service loop"""
        try:
            while self.running:
                await self._process_work()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            print(f"Service {self.name} cancelled")
            raise
    
    async def _process_work(self):
        """Simulate work processing"""
        try:
            # Simulate work
            await asyncio.sleep(0.01)
            
            # Simulate occasional errors
            if random.random() < 0.1:
                raise ValueError("Processing error")
            
            self.stats["processed"] += 1
        except ValueError:
            self.stats["errors"] += 1

async def background_service_demo():
    """Demonstrate background async services"""
    
    # Create and start services
    service1 = AsyncService("DataProcessor")
    service2 = AsyncService("LogCollector")
    
    await service1.start()
    await service2.start()
    
    # Let services run for a while
    await asyncio.sleep(0.5)
    
    # Stop services
    await service1.stop()
    await service2.stop()
    
    return {
        "service1_stats": service1.stats,
        "service2_stats": service2.stats,
        "demo_completed": True
    }

# Async semaphores and locks
async def synchronization_demo():
    """Demonstrate async synchronization primitives"""
    
    # Semaphore example
    semaphore = asyncio.Semaphore(2)  # Allow 2 concurrent operations
    
    async def limited_resource_access(name, duration):
        async with semaphore:
            print(f"Task {name} acquired semaphore")
            await asyncio.sleep(duration)
            print(f"Task {name} releasing semaphore")
            return f"Task {name} completed"
    
    # Lock example
    lock = asyncio.Lock()
    shared_resource = {"value": 0}
    
    async def increment_shared_resource(name, increments):
        results = []
        for i in range(increments):
            async with lock:
                old_value = shared_resource["value"]
                await asyncio.sleep(0.01)  # Simulate work
                shared_resource["value"] = old_value + 1
                results.append(f"{name}: {old_value} -> {shared_resource['value']}")
        return results
    
    # Event example
    event = asyncio.Event()
    
    async def waiter(name):
        print(f"Waiter {name} waiting for event")
        await event.wait()
        print(f"Waiter {name} received event")
        return f"Waiter {name} completed"
    
    async def setter():
        await asyncio.sleep(0.1)
        print("Setting event")
        event.set()
        return "Event set"
    
    # Run semaphore demo
    semaphore_tasks = [
        limited_resource_access("A", 0.1),
        limited_resource_access("B", 0.05),
        limited_resource_access("C", 0.15),
        limited_resource_access("D", 0.08)
    ]
    
    semaphore_results = await asyncio.gather(*semaphore_tasks)
    
    # Run lock demo
    lock_tasks = [
        increment_shared_resource("Worker1", 3),
        increment_shared_resource("Worker2", 3)
    ]
    
    lock_results = await asyncio.gather(*lock_tasks)
    
    # Run event demo
    event_tasks = [
        waiter("W1"),
        waiter("W2"),
        setter()
    ]
    
    event_results = await asyncio.gather(*event_tasks)
    
    return {
        "semaphore_results": semaphore_results,
        "lock_results": lock_results,
        "shared_resource_final": shared_resource["value"],
        "event_results": event_results
    }

# Mixing sync and async code
async def sync_async_integration_demo():
    """Demonstrate integration between sync and async code"""
    
    def cpu_bound_task(n):
        """CPU-intensive synchronous task"""
        total = 0
        for i in range(n):
            total += i ** 2
        return total
    
    def io_blocking_task(duration):
        """Blocking I/O task"""
        time.sleep(duration)
        return f"Blocking task completed after {duration}s"
    
    # Run CPU-bound task in thread pool
    loop = asyncio.get_event_loop()
    
    # Method 1: run_in_executor with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        cpu_result = await loop.run_in_executor(executor, cpu_bound_task, 10000)
    
    # Method 2: run_in_executor with default executor
    blocking_result = await loop.run_in_executor(None, io_blocking_task, 0.1)
    
    # Method 3: asyncio.to_thread (Python 3.9+)
    try:
        thread_result = await asyncio.to_thread(cpu_bound_task, 5000)
    except AttributeError:
        # Fallback for older Python versions
        thread_result = await loop.run_in_executor(None, cpu_bound_task, 5000)
    
    return {
        "cpu_bound_result": cpu_result,
        "blocking_io_result": blocking_result,
        "thread_result": thread_result,
        "integration_successful": True
    }

# Async testing utilities
class AsyncTestUtils:
    """Utilities for testing async code"""
    
    @staticmethod
    async def async_timer(coro):
        """Time an async operation"""
        start = time.time()
        result = await coro
        duration = time.time() - start
        return result, duration
    
    @staticmethod
    async def retry_async(coro_func, max_attempts=3, delay=0.1):
        """Retry an async operation"""
        for attempt in range(max_attempts):
            try:
                return await coro_func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    @staticmethod
    def mock_async_function(return_value, delay=0.01):
        """Create a mock async function"""
        async def mock():
            await asyncio.sleep(delay)
            return return_value
        return mock

async def async_testing_demo():
    """Demonstrate async testing utilities"""
    
    # Test timing
    async def test_operation():
        await asyncio.sleep(0.1)
        return "Operation complete"
    
    result, duration = await AsyncTestUtils.async_timer(test_operation())
    
    # Test retry mechanism
    attempt_count = 0
    
    async def unreliable_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return f"Success on attempt {attempt_count}"
    
    retry_result = await AsyncTestUtils.retry_async(unreliable_operation, max_attempts=5, delay=0.01)
    
    # Test mock function
    mock_func = AsyncTestUtils.mock_async_function("Mocked result", 0.01)
    mock_result = await mock_func()
    
    return {
        "timed_operation": {
            "result": result,
            "duration": f"{duration:.4f}s"
        },
        "retry_mechanism": {
            "result": retry_result,
            "attempts_made": attempt_count
        },
        "mock_function": mock_result
    }

# Real-world async patterns
@dataclass
class AsyncJob:
    """Represents an async job"""
    id: str
    task: Callable
    priority: int = 0
    timeout: float = 30.0

class AsyncJobProcessor:
    """Process async jobs with priority and timeout"""
    
    def __init__(self, max_workers=3):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.jobs_processed = 0
        self.jobs_failed = 0
    
    async def process_job(self, job: AsyncJob):
        """Process a single job"""
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(job.task(), timeout=job.timeout)
                self.jobs_processed += 1
                return {"job_id": job.id, "status": "success", "result": result}
            except asyncio.TimeoutError:
                self.jobs_failed += 1
                return {"job_id": job.id, "status": "timeout", "error": "Job timed out"}
            except Exception as e:
                self.jobs_failed += 1
                return {"job_id": job.id, "status": "error", "error": str(e)}
    
    async def process_jobs(self, jobs: List[AsyncJob]):
        """Process multiple jobs concurrently"""
        # Sort by priority (higher priority first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        tasks = [self.process_job(job) for job in sorted_jobs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "results": results,
            "stats": {
                "processed": self.jobs_processed,
                "failed": self.jobs_failed,
                "total": len(jobs)
            }
        }

async def real_world_patterns_demo():
    """Demonstrate real-world async patterns"""
    
    # Create sample jobs
    async def quick_job():
        await asyncio.sleep(0.05)
        return "Quick job result"
    
    async def slow_job():
        await asyncio.sleep(0.2)
        return "Slow job result"
    
    async def failing_job():
        await asyncio.sleep(0.1)
        raise ValueError("Job failed")
    
    async def timeout_job():
        await asyncio.sleep(1.0)
        return "This will timeout"
    
    jobs = [
        AsyncJob("job1", quick_job, priority=1, timeout=1.0),
        AsyncJob("job2", slow_job, priority=2, timeout=1.0),
        AsyncJob("job3", failing_job, priority=3, timeout=1.0),
        AsyncJob("job4", timeout_job, priority=1, timeout=0.1),
        AsyncJob("job5", quick_job, priority=2, timeout=1.0)
    ]
    
    processor = AsyncJobProcessor(max_workers=2)
    job_results = await processor.process_jobs(jobs)
    
    return job_results

# Comprehensive demo runner
async def run_all_async_demos():
    """Execute all async programming demonstrations"""
    demo_functions = [
        ('basic_async', basic_async_demo),
        ('concurrent_execution', concurrent_execution_demo),
        ('task_management', task_management_demo),
        ('async_protocols', async_protocols_demo),
        ('exception_handling', exception_handling_demo),
        ('producer_consumer', producer_consumer_demo),
        ('http_client', http_client_demo),
        ('background_service', background_service_demo),
        ('synchronization', synchronization_demo),
        ('sync_async_integration', sync_async_integration_demo),
        ('async_testing', async_testing_demo),
        ('real_world_patterns', real_world_patterns_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = await func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Add event loop demo (non-async)
    results['event_loop'] = event_loop_demo()
    
    return results

if __name__ == "__main__":
    print("=== Python Asyncio and Async Programming Demo ===")
    
    # Run all demonstrations
    all_results = asyncio.run(run_all_async_demos())
    
    for category, data in all_results.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        
        if 'error' in data:
            print(f"  Error: {data['error']}")
            continue
        
        # Display results
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"  {key}: {dict(list(value.items())[:3])}... (truncated)")
                elif isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]}... (showing first 5)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Result: {data}")
    
    print("\n=== ASYNC PROGRAMMING CONCEPTS ===")
    
    concepts = {
        "Event Loop": "Central execution engine that manages and executes async tasks",
        "Coroutines": "Functions defined with async def that can be paused and resumed",
        "await": "Keyword that pauses execution until awaitable completes",
        "Tasks": "Wrapper around coroutines for concurrent execution",
        "asyncio.gather()": "Run multiple awaitables concurrently and collect results",
        "asyncio.create_task()": "Schedule coroutine for execution",
        "async with": "Async context manager protocol",
        "async for": "Async iterator protocol"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== ASYNCIO PATTERNS ===")
    
    patterns = {
        "Producer-Consumer": "Use asyncio.Queue for async producer-consumer patterns",
        "Rate Limiting": "Use asyncio.Semaphore to limit concurrent operations",
        "Timeout Handling": "Use asyncio.wait_for() to add timeouts to operations",
        "Exception Handling": "Use return_exceptions=True in gather() to handle failures",
        "Background Tasks": "Create long-running services with async task management",
        "Mixed Sync/Async": "Use run_in_executor() to integrate blocking code",
        "Graceful Shutdown": "Handle cancellation and cleanup properly",
        "Resource Pooling": "Manage limited resources with semaphores and context managers"
    }
    
    for pattern, description in patterns.items():
        print(f"  {pattern}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use asyncio.run() as the main entry point for async programs",
        "Always await coroutines - don't forget the await keyword",
        "Use asyncio.create_task() to run coroutines concurrently",
        "Handle CancelledError properly for graceful shutdowns",
        "Use async context managers for resource management",
        "Avoid blocking operations in async code - use async alternatives",
        "Use return_exceptions=True in gather() when appropriate",
        "Set reasonable timeouts for async operations",
        "Use semaphores to limit concurrent resource usage",
        "Test async code thoroughly including cancellation scenarios",
        "Use asyncio.to_thread() for CPU-bound tasks (Python 3.9+)",
        "Prefer async libraries (aiohttp, aiofiles) over sync equivalents"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Asyncio and Async Programming Complete! ===")
    print("  Advanced async patterns and concurrent execution mastered")
