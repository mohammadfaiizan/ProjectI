"""
Performance Benchmarking Suite - Comprehensive Testing Framework
Difficulty: Hard

Comprehensive benchmarking suite for queue and stack implementations.
Focus on performance analysis, profiling, and optimization recommendations.
"""

import time
import sys
import gc
import threading
import multiprocessing
import random
import statistics
from typing import Any, List, Dict, Callable, Tuple, Optional
from collections import deque, defaultdict
import heapq
import psutil
import tracemalloc
from contextlib import contextmanager

class PerformanceProfiler:
    """
    Approach 1: Comprehensive Performance Profiler
    
    Profile time, memory, and CPU usage of data structure operations.
    """
    
    def __init__(self):
        self.results = {}
        self.memory_snapshots = []
        self.cpu_usage = []
    
    @contextmanager
    def profile(self, test_name: str):
        """Context manager for profiling operations"""
        # Start memory tracing
        tracemalloc.start()
        
        # Record initial state
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu = process.cpu_percent()
        
        # Start timing
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Record final state
            final_memory = process.memory_info().rss
            final_cpu = process.cpu_percent()
            
            # Store results
            self.results[test_name] = {
                'wall_time': end_time - start_time,
                'cpu_time': end_cpu_time - start_cpu_time,
                'memory_current': current,
                'memory_peak': peak,
                'memory_delta': final_memory - initial_memory,
                'cpu_usage': final_cpu
            }
    
    def get_results(self) -> Dict:
        """Get all profiling results"""
        return self.results
    
    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Profiling Summary ===")
        print(f"{'Test Name':<30} | {'Wall Time':<12} | {'CPU Time':<12} | {'Memory Peak':<15} | {'CPU %':<8}")
        print("-" * 90)
        
        for test_name, metrics in self.results.items():
            wall_time = f"{metrics['wall_time']:.6f}s"
            cpu_time = f"{metrics['cpu_time']:.6f}s"
            memory_peak = f"{metrics['memory_peak'] / 1024 / 1024:.2f} MB"
            cpu_usage = f"{metrics['cpu_usage']:.1f}%"
            
            print(f"{test_name:<30} | {wall_time:<12} | {cpu_time:<12} | {memory_peak:<15} | {cpu_usage:<8}")


class BenchmarkSuite:
    """
    Approach 2: Comprehensive Benchmark Suite
    
    Test suite for comparing different data structure implementations.
    """
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.implementations = {}
        self.test_results = defaultdict(dict)
    
    def register_implementation(self, name: str, stack_class: type, queue_class: type):
        """Register a data structure implementation for testing"""
        self.implementations[name] = {
            'stack': stack_class,
            'queue': queue_class
        }
    
    def benchmark_stack_operations(self, n_operations: int = 100000):
        """Benchmark stack operations across implementations"""
        print(f"\n=== Stack Operations Benchmark ({n_operations:,} operations) ===")
        
        for impl_name, classes in self.implementations.items():
            stack_class = classes['stack']
            
            with self.profiler.profile(f"{impl_name}_stack_push"):
                stack = stack_class()
                for i in range(n_operations):
                    stack.push(i)
            
            with self.profiler.profile(f"{impl_name}_stack_pop"):
                for _ in range(n_operations):
                    stack.pop()
            
            # Mixed operations
            with self.profiler.profile(f"{impl_name}_stack_mixed"):
                stack = stack_class()
                for i in range(n_operations // 2):
                    stack.push(i)
                    if i % 3 == 0 and not stack.is_empty():
                        stack.pop()
    
    def benchmark_queue_operations(self, n_operations: int = 100000):
        """Benchmark queue operations across implementations"""
        print(f"\n=== Queue Operations Benchmark ({n_operations:,} operations) ===")
        
        for impl_name, classes in self.implementations.items():
            queue_class = classes['queue']
            
            with self.profiler.profile(f"{impl_name}_queue_enqueue"):
                queue = queue_class()
                for i in range(n_operations):
                    queue.enqueue(i)
            
            with self.profiler.profile(f"{impl_name}_queue_dequeue"):
                for _ in range(n_operations):
                    queue.dequeue()
            
            # Mixed operations
            with self.profiler.profile(f"{impl_name}_queue_mixed"):
                queue = queue_class()
                for i in range(n_operations // 2):
                    queue.enqueue(i)
                    if i % 3 == 0 and not queue.is_empty():
                        queue.dequeue()
    
    def benchmark_memory_usage(self, n_items: int = 50000):
        """Benchmark memory usage patterns"""
        print(f"\n=== Memory Usage Benchmark ({n_items:,} items) ===")
        
        for impl_name, classes in self.implementations.items():
            # Stack memory usage
            with self.profiler.profile(f"{impl_name}_stack_memory"):
                stack = classes['stack']()
                for i in range(n_items):
                    stack.push(i)
                
                # Keep reference to measure peak usage
                _ = stack
            
            # Queue memory usage
            with self.profiler.profile(f"{impl_name}_queue_memory"):
                queue = classes['queue']()
                for i in range(n_items):
                    queue.enqueue(i)
                
                # Keep reference to measure peak usage
                _ = queue
    
    def benchmark_concurrent_access(self, n_threads: int = 4, operations_per_thread: int = 10000):
        """Benchmark concurrent access patterns"""
        print(f"\n=== Concurrent Access Benchmark ({n_threads} threads, {operations_per_thread:,} ops/thread) ===")
        
        for impl_name, classes in self.implementations.items():
            if 'ThreadSafe' not in impl_name:
                continue  # Skip non-thread-safe implementations
            
            def worker(stack, queue, worker_id: int, results: List):
                """Worker thread function"""
                start_time = time.perf_counter()
                
                # Stack operations
                for i in range(operations_per_thread // 2):
                    stack.push(f"{worker_id}_{i}")
                
                for _ in range(operations_per_thread // 4):
                    try:
                        stack.pop()
                    except:
                        pass
                
                # Queue operations
                for i in range(operations_per_thread // 2):
                    queue.enqueue(f"{worker_id}_{i}")
                
                for _ in range(operations_per_thread // 4):
                    try:
                        queue.dequeue()
                    except:
                        pass
                
                end_time = time.perf_counter()
                results.append(end_time - start_time)
            
            with self.profiler.profile(f"{impl_name}_concurrent"):
                stack = classes['stack']()
                queue = classes['queue']()
                threads = []
                results = []
                
                # Start worker threads
                for i in range(n_threads):
                    t = threading.Thread(target=worker, args=(stack, queue, i, results))
                    threads.append(t)
                    t.start()
                
                # Wait for completion
                for t in threads:
                    t.join()
                
                # Store thread timing results
                self.test_results[impl_name]['concurrent_times'] = results
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PERFORMANCE REPORT")
        report.append("=" * 80)
        
        results = self.profiler.get_results()
        
        # Group results by implementation
        impl_results = defaultdict(dict)
        for test_name, metrics in results.items():
            impl_name = test_name.split('_')[0]
            operation = '_'.join(test_name.split('_')[1:])
            impl_results[impl_name][operation] = metrics
        
        # Generate per-implementation analysis
        for impl_name, operations in impl_results.items():
            report.append(f"\n{impl_name.upper()} IMPLEMENTATION:")
            report.append("-" * 40)
            
            for op_name, metrics in operations.items():
                report.append(f"  {op_name}:")
                report.append(f"    Wall Time: {metrics['wall_time']:.6f}s")
                report.append(f"    CPU Time:  {metrics['cpu_time']:.6f}s")
                report.append(f"    Memory Peak: {metrics['memory_peak'] / 1024 / 1024:.2f} MB")
                report.append(f"    CPU Usage: {metrics['cpu_usage']:.1f}%")
        
        # Performance rankings
        report.append(f"\nPERFORMANCE RANKINGS:")
        report.append("-" * 30)
        
        # Rank by wall time for push operations
        push_times = [(name, metrics['wall_time']) for name, metrics in results.items() 
                     if 'push' in name or 'enqueue' in name]
        push_times.sort(key=lambda x: x[1])
        
        report.append("Fastest Push/Enqueue Operations:")
        for i, (name, time_val) in enumerate(push_times[:5]):
            report.append(f"  {i+1}. {name}: {time_val:.6f}s")
        
        return '\n'.join(report)


class ScalabilityTester:
    """
    Approach 3: Scalability Testing Framework
    
    Test how implementations scale with increasing data sizes.
    """
    
    def __init__(self):
        self.scaling_results = defaultdict(list)
    
    def test_scaling(self, implementation_name: str, stack_class: type, queue_class: type,
                    sizes: List[int] = None):
        """Test scaling behavior across different data sizes"""
        if sizes is None:
            sizes = [1000, 5000, 10000, 50000, 100000]
        
        print(f"\n=== Scalability Test: {implementation_name} ===")
        
        for size in sizes:
            # Test stack scaling
            start_time = time.perf_counter()
            stack = stack_class()
            
            for i in range(size):
                stack.push(i)
            
            for _ in range(size):
                stack.pop()
            
            stack_time = time.perf_counter() - start_time
            
            # Test queue scaling
            start_time = time.perf_counter()
            queue = queue_class()
            
            for i in range(size):
                queue.enqueue(i)
            
            for _ in range(size):
                queue.dequeue()
            
            queue_time = time.perf_counter() - start_time
            
            self.scaling_results[implementation_name].append({
                'size': size,
                'stack_time': stack_time,
                'queue_time': queue_time,
                'stack_ops_per_sec': (2 * size) / stack_time,
                'queue_ops_per_sec': (2 * size) / queue_time
            })
            
            print(f"  Size {size:6,}: Stack {stack_time:.4f}s, Queue {queue_time:.4f}s")
    
    def analyze_complexity(self, implementation_name: str) -> Dict:
        """Analyze time complexity based on scaling results"""
        results = self.scaling_results[implementation_name]
        
        if len(results) < 3:
            return {"error": "Insufficient data points"}
        
        sizes = [r['size'] for r in results]
        stack_times = [r['stack_time'] for r in results]
        queue_times = [r['queue_time'] for r in results]
        
        # Simple complexity analysis (ratio of time increase vs size increase)
        stack_ratios = []
        queue_ratios = []
        
        for i in range(1, len(results)):
            size_ratio = sizes[i] / sizes[i-1]
            stack_time_ratio = stack_times[i] / stack_times[i-1]
            queue_time_ratio = queue_times[i] / queue_times[i-1]
            
            stack_ratios.append(stack_time_ratio / size_ratio)
            queue_ratios.append(queue_time_ratio / size_ratio)
        
        avg_stack_ratio = statistics.mean(stack_ratios)
        avg_queue_ratio = statistics.mean(queue_ratios)
        
        def classify_complexity(ratio: float) -> str:
            if ratio < 1.2:
                return "O(n) - Linear"
            elif ratio < 1.8:
                return "O(n log n) - Linearithmic"
            elif ratio < 3.0:
                return "O(n²) - Quadratic"
            else:
                return "O(n³+) - Cubic or worse"
        
        return {
            'stack_complexity': classify_complexity(avg_stack_ratio),
            'queue_complexity': classify_complexity(avg_queue_ratio),
            'stack_ratio': avg_stack_ratio,
            'queue_ratio': avg_queue_ratio
        }


class StressTester:
    """
    Approach 4: Stress Testing Framework
    
    Test implementations under extreme conditions.
    """
    
    def __init__(self):
        self.stress_results = {}
    
    def memory_stress_test(self, implementation_name: str, stack_class: type, 
                          max_items: int = 1000000):
        """Test memory limits and behavior"""
        print(f"\n=== Memory Stress Test: {implementation_name} ===")
        
        try:
            stack = stack_class()
            memory_usage = []
            
            process = psutil.Process()
            
            for i in range(0, max_items, max_items // 10):
                # Add items
                for j in range(max_items // 10):
                    stack.push(f"item_{i+j}")
                
                # Record memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_usage.append((i + max_items // 10, memory_mb))
                
                print(f"  Items: {i + max_items // 10:7,} | Memory: {memory_mb:.1f} MB")
                
                # Check if we're running out of memory
                if memory_mb > 1000:  # 1GB limit
                    print(f"  Memory limit reached at {i + max_items // 10:,} items")
                    break
            
            self.stress_results[f"{implementation_name}_memory"] = {
                'max_items_tested': i + max_items // 10,
                'memory_usage': memory_usage,
                'memory_per_item': memory_usage[-1][1] / memory_usage[-1][0] if memory_usage else 0
            }
            
        except MemoryError:
            print(f"  MemoryError at approximately {i:,} items")
            self.stress_results[f"{implementation_name}_memory"] = {
                'memory_error_at': i,
                'status': 'memory_error'
            }
    
    def concurrency_stress_test(self, implementation_name: str, stack_class: type,
                               n_threads: int = 50, operations_per_thread: int = 1000):
        """Test under high concurrency"""
        if 'ThreadSafe' not in implementation_name:
            print(f"Skipping concurrency test for non-thread-safe {implementation_name}")
            return
        
        print(f"\n=== Concurrency Stress Test: {implementation_name} ===")
        print(f"Testing with {n_threads} threads, {operations_per_thread} operations each")
        
        shared_stack = stack_class()
        errors = []
        completion_times = []
        
        def stress_worker(worker_id: int):
            """High-stress worker thread"""
            start_time = time.perf_counter()
            local_errors = []
            
            try:
                # Rapid push operations
                for i in range(operations_per_thread):
                    shared_stack.push(f"w{worker_id}_i{i}")
                    
                    # Occasional pop
                    if i % 10 == 0:
                        try:
                            shared_stack.pop()
                        except Exception as e:
                            local_errors.append(str(e))
                
            except Exception as e:
                local_errors.append(f"Worker {worker_id}: {str(e)}")
            
            end_time = time.perf_counter()
            completion_times.append(end_time - start_time)
            errors.extend(local_errors)
        
        # Start all threads simultaneously
        threads = []
        start_time = time.perf_counter()
        
        for i in range(n_threads):
            t = threading.Thread(target=stress_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        total_time = time.perf_counter() - start_time
        
        self.stress_results[f"{implementation_name}_concurrency"] = {
            'total_time': total_time,
            'avg_thread_time': statistics.mean(completion_times),
            'max_thread_time': max(completion_times),
            'min_thread_time': min(completion_times),
            'error_count': len(errors),
            'errors': errors[:10],  # First 10 errors
            'final_stack_size': shared_stack.size() if hasattr(shared_stack, 'size') else 'unknown'
        }
        
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Average thread time: {statistics.mean(completion_times):.4f}s")
        print(f"  Errors encountered: {len(errors)}")
        print(f"  Final stack size: {shared_stack.size() if hasattr(shared_stack, 'size') else 'unknown'}")


class OptimizationRecommender:
    """
    Approach 5: Optimization Recommendation Engine
    
    Analyze performance results and provide optimization recommendations.
    """
    
    def __init__(self, benchmark_suite: BenchmarkSuite, scalability_tester: ScalabilityTester):
        self.benchmark_suite = benchmark_suite
        self.scalability_tester = scalability_tester
    
    def analyze_and_recommend(self) -> List[str]:
        """Analyze results and provide optimization recommendations"""
        recommendations = []
        results = self.benchmark_suite.profiler.get_results()
        
        # Analyze memory usage patterns
        memory_results = [(name, metrics['memory_peak']) for name, metrics in results.items() 
                         if 'memory' in name]
        
        if memory_results:
            memory_results.sort(key=lambda x: x[1])
            best_memory = memory_results[0]
            worst_memory = memory_results[-1]
            
            recommendations.append(f"Memory Efficiency: {best_memory[0]} uses {best_memory[1]/1024/1024:.1f}MB "
                                 f"vs {worst_memory[0]} at {worst_memory[1]/1024/1024:.1f}MB")
            
            if worst_memory[1] > best_memory[1] * 2:
                recommendations.append(f"Consider using {best_memory[0].split('_')[0]} for memory-constrained environments")
        
        # Analyze time performance
        time_results = [(name, metrics['wall_time']) for name, metrics in results.items() 
                       if 'push' in name or 'enqueue' in name]
        
        if time_results:
            time_results.sort(key=lambda x: x[1])
            fastest = time_results[0]
            slowest = time_results[-1]
            
            recommendations.append(f"Speed: {fastest[0]} is {slowest[1]/fastest[1]:.1f}x faster than {slowest[0]}")
            
            if slowest[1] > fastest[1] * 1.5:
                recommendations.append(f"For high-performance applications, prefer {fastest[0].split('_')[0]}")
        
        # Analyze scalability
        for impl_name, results_list in self.scalability_tester.scaling_results.items():
            complexity = self.scalability_tester.analyze_complexity(impl_name)
            
            if 'Quadratic' in complexity.get('stack_complexity', ''):
                recommendations.append(f"Warning: {impl_name} stack shows quadratic scaling - avoid for large datasets")
            
            if 'Quadratic' in complexity.get('queue_complexity', ''):
                recommendations.append(f"Warning: {impl_name} queue shows quadratic scaling - avoid for large datasets")
        
        # General recommendations
        recommendations.extend([
            "For memory-critical applications: Use array-based or slotted implementations",
            "For high-concurrency: Use thread-safe implementations with proper locking",
            "For large datasets: Prefer implementations with O(1) or O(log n) complexity",
            "For embedded systems: Consider memory-mapped or compact implementations"
        ])
        
        return recommendations


# Sample implementations for testing
class SimpleStack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


class SimpleQueue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        return self.items.popleft()
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)


class ThreadSafeStack:
    def __init__(self):
        self.items = []
        self.lock = threading.RLock()
    
    def push(self, item):
        with self.lock:
            self.items.append(item)
    
    def pop(self):
        with self.lock:
            return self.items.pop()
    
    def is_empty(self):
        with self.lock:
            return len(self.items) == 0
    
    def size(self):
        with self.lock:
            return len(self.items)


class ThreadSafeQueue:
    def __init__(self):
        self.items = deque()
        self.lock = threading.RLock()
    
    def enqueue(self, item):
        with self.lock:
            self.items.append(item)
    
    def dequeue(self):
        with self.lock:
            return self.items.popleft()
    
    def is_empty(self):
        with self.lock:
            return len(self.items) == 0
    
    def size(self):
        with self.lock:
            return len(self.items)


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARKING SUITE")
    print("=" * 80)
    
    # Initialize test framework
    benchmark_suite = BenchmarkSuite()
    scalability_tester = ScalabilityTester()
    stress_tester = StressTester()
    
    # Register implementations
    benchmark_suite.register_implementation("Simple", SimpleStack, SimpleQueue)
    benchmark_suite.register_implementation("ThreadSafe", ThreadSafeStack, ThreadSafeQueue)
    
    # Run benchmarks
    print("\n1. Running basic operation benchmarks...")
    benchmark_suite.benchmark_stack_operations(50000)
    benchmark_suite.benchmark_queue_operations(50000)
    
    print("\n2. Running memory usage benchmarks...")
    benchmark_suite.benchmark_memory_usage(25000)
    
    print("\n3. Running concurrent access benchmarks...")
    benchmark_suite.benchmark_concurrent_access(4, 5000)
    
    print("\n4. Running scalability tests...")
    scalability_tester.test_scaling("Simple", SimpleStack, SimpleQueue, [1000, 5000, 10000, 25000])
    scalability_tester.test_scaling("ThreadSafe", ThreadSafeStack, ThreadSafeQueue, [1000, 5000, 10000, 25000])
    
    print("\n5. Running stress tests...")
    stress_tester.memory_stress_test("Simple", SimpleStack, 100000)
    stress_tester.concurrency_stress_test("ThreadSafe", ThreadSafeStack, 20, 500)
    
    # Generate analysis
    print("\n6. Generating analysis and recommendations...")
    recommender = OptimizationRecommender(benchmark_suite, scalability_tester)
    recommendations = recommender.analyze_and_recommend()
    
    # Print results
    benchmark_suite.profiler.print_summary()
    
    print("\n" + "=" * 80)
    print("SCALABILITY ANALYSIS")
    print("=" * 80)
    
    for impl_name in scalability_tester.scaling_results:
        complexity = scalability_tester.analyze_complexity(impl_name)
        print(f"\n{impl_name}:")
        print(f"  Stack Complexity: {complexity.get('stack_complexity', 'Unknown')}")
        print(f"  Queue Complexity: {complexity.get('queue_complexity', 'Unknown')}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")
    
    # Generate full report
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE REPORT")
    print("=" * 80)
    
    report = benchmark_suite.generate_report()
    print(report)


def demonstrate_profiling_capabilities():
    """Demonstrate advanced profiling capabilities"""
    print("\n=== Advanced Profiling Demonstration ===")
    
    profiler = PerformanceProfiler()
    
    # Profile different operations
    with profiler.profile("list_operations"):
        data = []
        for i in range(10000):
            data.append(i)
        for _ in range(10000):
            data.pop()
    
    with profiler.profile("deque_operations"):
        data = deque()
        for i in range(10000):
            data.append(i)
        for _ in range(10000):
            data.pop()
    
    with profiler.profile("memory_intensive"):
        # Create memory-intensive operation
        big_list = [list(range(1000)) for _ in range(100)]
        del big_list
    
    profiler.print_summary()


def analyze_real_world_scenarios():
    """Analyze performance in real-world scenarios"""
    print("\n=== Real-World Scenario Analysis ===")
    
    scenarios = [
        ("Web Server Request Queue", "High concurrency, mixed operations"),
        ("Undo/Redo System", "Sequential operations, memory sensitive"),
        ("Task Scheduler", "Priority-based, scalability important"),
        ("Cache System", "Fast access, memory efficient"),
        ("Game State Stack", "Real-time, low latency required")
    ]
    
    print("Scenario Analysis:")
    for scenario, characteristics in scenarios:
        print(f"\n{scenario}:")
        print(f"  Characteristics: {characteristics}")
        
        # Provide recommendations based on characteristics
        if "concurrency" in characteristics.lower():
            print("  Recommendation: Use thread-safe implementations with proper locking")
        if "memory" in characteristics.lower():
            print("  Recommendation: Consider memory-optimized structures")
        if "priority" in characteristics.lower():
            print("  Recommendation: Use priority queue or multi-level queue")
        if "real-time" in characteristics.lower():
            print("  Recommendation: Optimize for O(1) operations, avoid GC pressure")


if __name__ == "__main__":
    # Check if required packages are available
    try:
        import psutil
        run_comprehensive_benchmark()
        demonstrate_profiling_capabilities()
        analyze_real_world_scenarios()
    except ImportError:
        print("Note: psutil package not available. Running limited benchmark...")
        
        # Run basic benchmark without system monitoring
        benchmark_suite = BenchmarkSuite()
        benchmark_suite.register_implementation("Simple", SimpleStack, SimpleQueue)
        benchmark_suite.benchmark_stack_operations(10000)
        benchmark_suite.benchmark_queue_operations(10000)
        
        print("\nBasic benchmark completed. Install psutil for full functionality:")
        print("pip install psutil")

"""
Performance Benchmarking Suite provides comprehensive testing framework
for analyzing queue and stack implementations including performance profiling,
scalability testing, stress testing, and optimization recommendations.
"""
