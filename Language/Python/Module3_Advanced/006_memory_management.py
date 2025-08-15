"""
Python Memory Management: Reference Counting, Garbage Collection, and Memory Optimization
Implementation-focused with minimal comments, maximum functionality coverage
"""

import gc
import sys
import weakref
import tracemalloc
import psutil
import os
import time
from typing import Any, Dict, List, Optional, Set, Callable
import threading
from collections import defaultdict
import ctypes
from dataclasses import dataclass
import array
import mmap

# Reference counting demonstration
def reference_counting_demo():
    """Demonstrate Python's reference counting mechanism"""
    
    class RefCountedObject:
        def __init__(self, name):
            self.name = name
            print(f"Created {self.name}")
        
        def __del__(self):
            print(f"Deleted {self.name}")
    
    # Create object and track references
    obj = RefCountedObject("obj1")
    initial_refcount = sys.getrefcount(obj) - 1  # -1 for getrefcount itself
    
    # Add references
    ref1 = obj
    ref2 = obj
    list_ref = [obj]
    dict_ref = {'key': obj}
    
    max_refcount = sys.getrefcount(obj) - 1
    
    # Remove references
    del ref1
    after_del1 = sys.getrefcount(obj) - 1
    
    del ref2
    after_del2 = sys.getrefcount(obj) - 1
    
    list_ref.clear()
    after_list_clear = sys.getrefcount(obj) - 1
    
    del dict_ref
    after_dict_del = sys.getrefcount(obj) - 1
    
    # Final reference count before deletion
    final_refcount = sys.getrefcount(obj) - 1
    
    return {
        "initial_refcount": initial_refcount,
        "max_refcount": max_refcount,
        "after_del1": after_del1,
        "after_del2": after_del2,
        "after_list_clear": after_list_clear,
        "after_dict_del": after_dict_del,
        "final_refcount": final_refcount
    }

# Garbage collection cycles
def garbage_collection_demo():
    """Demonstrate garbage collection for circular references"""
    
    class Node:
        def __init__(self, name):
            self.name = name
            self.children = []
            self.parent = None
        
        def add_child(self, child):
            child.parent = self
            self.children.append(child)
        
        def __del__(self):
            print(f"Node {self.name} deleted")
    
    # Get initial garbage collection stats
    initial_stats = gc.get_stats()
    initial_count = len(gc.get_objects())
    
    # Disable automatic garbage collection
    gc.disable()
    
    # Create circular reference
    root = Node("root")
    child1 = Node("child1")
    child2 = Node("child2")
    
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(root)  # Circular reference
    
    # Check reference counts
    root_refs = sys.getrefcount(root) - 1
    child1_refs = sys.getrefcount(child1) - 1
    
    # Delete variables but objects should remain due to circular refs
    del root, child1, child2
    
    before_gc_count = len(gc.get_objects())
    
    # Force garbage collection
    collected = gc.collect()
    
    after_gc_count = len(gc.get_objects())
    
    # Re-enable automatic garbage collection
    gc.enable()
    
    final_stats = gc.get_stats()
    
    return {
        "initial_object_count": initial_count,
        "before_gc_count": before_gc_count,
        "after_gc_count": after_gc_count,
        "objects_collected": collected,
        "root_references": root_refs,
        "child1_references": child1_refs,
        "gc_stats_diff": len(final_stats) - len(initial_stats)
    }

# Weak references
def weak_references_demo():
    """Demonstrate weak references to avoid circular dependencies"""
    
    class Parent:
        def __init__(self, name):
            self.name = name
            self.children = []
        
        def add_child(self, child):
            self.children.append(child)
            child.parent = weakref.ref(self)  # Weak reference to parent
        
        def __del__(self):
            print(f"Parent {self.name} deleted")
    
    class Child:
        def __init__(self, name):
            self.name = name
            self.parent = None
        
        def get_parent(self):
            if self.parent is not None:
                return self.parent()  # Call weak reference
            return None
        
        def __del__(self):
            print(f"Child {self.name} deleted")
    
    # WeakSet and WeakKeyDictionary examples
    class Registry:
        def __init__(self):
            self.objects = weakref.WeakSet()
            self.metadata = weakref.WeakKeyDictionary()
        
        def register(self, obj, metadata=None):
            self.objects.add(obj)
            if metadata:
                self.metadata[obj] = metadata
        
        def get_count(self):
            return len(self.objects)
        
        def get_metadata(self, obj):
            return self.metadata.get(obj)
    
    # Test weak references
    parent = Parent("parent1")
    child1 = Child("child1")
    child2 = Child("child2")
    
    parent.add_child(child1)
    parent.add_child(child2)
    
    # Test parent retrieval
    retrieved_parent = child1.get_parent()
    parent_name = retrieved_parent.name if retrieved_parent else None
    
    # Test registry
    registry = Registry()
    
    # Create some objects
    obj1 = Parent("obj1")
    obj2 = Child("obj2")
    
    registry.register(obj1, {"type": "parent", "priority": 1})
    registry.register(obj2, {"type": "child", "priority": 2})
    
    count_before_deletion = registry.get_count()
    metadata_obj1 = registry.get_metadata(obj1)
    
    # Delete one object
    del obj1
    gc.collect()  # Force cleanup
    
    count_after_deletion = registry.get_count()
    
    # Delete parent and check if child can still access it
    del parent
    gc.collect()
    
    parent_after_deletion = child1.get_parent()
    
    return {
        "parent_name_retrieved": parent_name,
        "registry_before_deletion": count_before_deletion,
        "registry_after_deletion": count_after_deletion,
        "metadata_obj1": metadata_obj1,
        "parent_accessible_after_deletion": parent_after_deletion is not None,
        "weak_ref_demo_completed": True
    }

# Memory profiling
def memory_profiling_demo():
    """Demonstrate memory profiling techniques"""
    
    def memory_intensive_function():
        """Function that uses significant memory"""
        # Create large data structures
        large_list = list(range(100000))
        large_dict = {i: f"value_{i}" for i in range(50000)}
        large_set = set(range(75000))
        
        # Nested structures
        nested = [[i] * 100 for i in range(1000)]
        
        return len(large_list) + len(large_dict) + len(large_set) + len(nested)
    
    # Start tracing memory allocations
    tracemalloc.start()
    
    # Get current memory usage
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Take snapshot before function call
    snapshot_before = tracemalloc.take_snapshot()
    
    # Call memory-intensive function
    result = memory_intensive_function()
    
    # Take snapshot after function call
    snapshot_after = tracemalloc.take_snapshot()
    
    # Get memory usage after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Compare snapshots
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    
    # Get top memory allocations
    top_allocations = []
    for stat in top_stats[:5]:
        top_allocations.append({
            'filename': stat.traceback.format()[-1],
            'size_mb': stat.size / 1024 / 1024,
            'count': stat.count
        })
    
    # Get current memory statistics
    current_stats = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    return {
        "function_result": result,
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_increase_mb": round(memory_after - memory_before, 2),
        "traced_current_mb": round(current_stats[0] / 1024 / 1024, 2),
        "traced_peak_mb": round(current_stats[1] / 1024 / 1024, 2),
        "top_allocations": top_allocations
    }

# __slots__ optimization
def slots_optimization_demo():
    """Demonstrate memory optimization using __slots__"""
    
    class RegularClass:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    class SlottedClass:
        __slots__ = ['x', 'y', 'z']
        
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
    
    class SlottedWithDict:
        __slots__ = ['x', 'y', '__dict__']
        
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.dynamic_attr = z
    
    # Create instances and measure memory
    regular_instances = [RegularClass(i, i*2, i*3) for i in range(1000)]
    slotted_instances = [SlottedClass(i, i*2, i*3) for i in range(1000)]
    
    # Measure instance sizes
    regular_size = sys.getsizeof(regular_instances[0])
    regular_dict_size = sys.getsizeof(regular_instances[0].__dict__)
    slotted_size = sys.getsizeof(slotted_instances[0])
    
    # Test attribute access performance
    import timeit
    
    regular_access_time = timeit.timeit(
        lambda: regular_instances[0].x,
        number=100000
    )
    
    slotted_access_time = timeit.timeit(
        lambda: slotted_instances[0].x,
        number=100000
    )
    
    # Test dynamic attribute capabilities
    regular_obj = RegularClass(1, 2, 3)
    slotted_obj = SlottedClass(1, 2, 3)
    slotted_with_dict_obj = SlottedWithDict(1, 2, 3)
    
    # Test dynamic attribute assignment
    regular_obj.dynamic = "works"
    
    try:
        slotted_obj.dynamic = "fails"
        slotted_dynamic_error = None
    except AttributeError as e:
        slotted_dynamic_error = str(e)
    
    slotted_with_dict_obj.another_dynamic = "works"
    
    # Memory usage comparison
    process = psutil.Process(os.getpid())
    memory_with_objects = process.memory_info().rss / 1024 / 1024
    
    return {
        "instance_sizes": {
            "regular_instance": regular_size,
            "regular_dict": regular_dict_size,
            "regular_total": regular_size + regular_dict_size,
            "slotted_instance": slotted_size
        },
        "performance": {
            "regular_access_time": f"{regular_access_time:.6f}s",
            "slotted_access_time": f"{slotted_access_time:.6f}s",
            "speedup_factor": round(regular_access_time / slotted_access_time, 2)
        },
        "dynamic_attributes": {
            "regular_supports_dynamic": hasattr(regular_obj, 'dynamic'),
            "slotted_dynamic_error": slotted_dynamic_error,
            "slotted_with_dict_supports": hasattr(slotted_with_dict_obj, 'another_dynamic')
        },
        "memory_impact_mb": round(memory_with_objects, 2)
    }

# Memory leaks detection
def memory_leak_demo():
    """Demonstrate common memory leak patterns and detection"""
    
    class LeakyClass:
        _instances = []  # Class variable storing all instances
        
        def __init__(self, data):
            self.data = data
            self._instances.append(self)  # Memory leak: instances never removed
    
    class FixedClass:
        _instances = weakref.WeakSet()  # Use weak references
        
        def __init__(self, data):
            self.data = data
            self._instances.add(self)
    
    class CallbackRegistry:
        def __init__(self):
            self.callbacks = []  # Strong references to callbacks
            self.weak_callbacks = weakref.WeakSet()
        
        def register_callback(self, callback, use_weak=False):
            if use_weak:
                self.weak_callbacks.add(callback)
            else:
                self.callbacks.append(callback)
        
        def clear_callbacks(self):
            self.callbacks.clear()
    
    # Simulate memory leak
    initial_object_count = len(gc.get_objects())
    
    # Create leaky instances
    leaky_objects = []
    for i in range(100):
        obj = LeakyClass(f"data_{i}")
        leaky_objects.append(obj)
    
    leaky_instances_count = len(LeakyClass._instances)
    
    # Clear our references but instances remain in class variable
    del leaky_objects
    gc.collect()
    
    instances_after_del = len(LeakyClass._instances)
    
    # Create fixed instances
    fixed_objects = []
    for i in range(100):
        obj = FixedClass(f"data_{i}")
        fixed_objects.append(obj)
    
    fixed_instances_count = len(FixedClass._instances)
    
    # Clear references - should be cleaned up properly
    del fixed_objects
    gc.collect()
    
    fixed_instances_after_del = len(FixedClass._instances)
    
    # Callback leak demonstration
    registry = CallbackRegistry()
    
    def callback_function():
        return "callback result"
    
    class CallbackObject:
        def callback_method(self):
            return "method result"
    
    callback_obj = CallbackObject()
    
    # Register callbacks
    registry.register_callback(callback_function, use_weak=False)
    registry.register_callback(callback_obj.callback_method, use_weak=False)
    registry.register_callback(callback_obj.callback_method, use_weak=True)
    
    strong_callbacks_before = len(registry.callbacks)
    weak_callbacks_before = len(registry.weak_callbacks)
    
    # Delete callback object
    del callback_obj
    gc.collect()
    
    strong_callbacks_after = len(registry.callbacks)
    weak_callbacks_after = len(registry.weak_callbacks)
    
    final_object_count = len(gc.get_objects())
    
    return {
        "object_counts": {
            "initial": initial_object_count,
            "final": final_object_count,
            "increase": final_object_count - initial_object_count
        },
        "leaky_class": {
            "instances_created": leaky_instances_count,
            "instances_after_deletion": instances_after_del,
            "leak_detected": instances_after_del > 0
        },
        "fixed_class": {
            "instances_created": fixed_instances_count,
            "instances_after_deletion": fixed_instances_after_del,
            "properly_cleaned": fixed_instances_after_del == 0
        },
        "callback_registry": {
            "strong_before": strong_callbacks_before,
            "strong_after": strong_callbacks_after,
            "weak_before": weak_callbacks_before,
            "weak_after": weak_callbacks_after,
            "weak_refs_cleaned": weak_callbacks_after < weak_callbacks_before
        }
    }

# Object lifecycle management
def object_lifecycle_demo():
    """Demonstrate object lifecycle and finalization"""
    
    class ManagedResource:
        def __init__(self, name):
            self.name = name
            self.finalized = False
            print(f"Acquiring resource: {self.name}")
            
            # Register finalizer
            self._finalizer = weakref.finalize(self, self._cleanup, self.name)
        
        @staticmethod
        def _cleanup(name):
            print(f"Finalizing resource: {name}")
        
        def close(self):
            """Explicit cleanup"""
            if not self.finalized:
                print(f"Explicitly closing resource: {self.name}")
                self.finalized = True
                self._finalizer.detach()  # Prevent finalizer from running
        
        def __del__(self):
            print(f"__del__ called for {self.name}")
    
    class ContextManagedResource:
        def __init__(self, name):
            self.name = name
            self.active = False
        
        def __enter__(self):
            self.active = True
            print(f"Context entering: {self.name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.active = False
            print(f"Context exiting: {self.name}")
    
    lifecycle_events = []
    
    # Test explicit cleanup
    resource1 = ManagedResource("resource1")
    resource1.close()
    
    # Test automatic cleanup via finalizer
    resource2 = ManagedResource("resource2")
    del resource2
    gc.collect()  # Force finalization
    
    # Test context manager
    with ContextManagedResource("context_resource") as ctx_res:
        context_active = ctx_res.active
    
    context_active_after = ctx_res.active
    
    # Test weak reference callbacks
    callback_called = []
    
    def weak_callback(weak_ref):
        callback_called.append("Weak reference callback executed")
    
    obj = ManagedResource("callback_test")
    weak_ref = weakref.ref(obj, weak_callback)
    
    del obj
    gc.collect()
    
    return {
        "explicit_cleanup": "Resource closed explicitly",
        "automatic_cleanup": "Resource finalized automatically",
        "context_manager": {
            "active_inside_context": context_active,
            "active_after_context": context_active_after
        },
        "weak_callback": {
            "callback_executed": len(callback_called) > 0,
            "callback_message": callback_called[0] if callback_called else None
        }
    }

# Memory optimization patterns
def memory_optimization_patterns():
    """Demonstrate various memory optimization techniques"""
    
    # Pattern 1: Object pooling
    class ObjectPool:
        def __init__(self, factory, max_size=10):
            self._factory = factory
            self._pool = []
            self._max_size = max_size
            self._created = 0
            self._reused = 0
        
        def get(self):
            if self._pool:
                self._reused += 1
                return self._pool.pop()
            else:
                self._created += 1
                return self._factory()
        
        def put(self, obj):
            if len(self._pool) < self._max_size:
                # Reset object state before pooling
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
        
        def get_stats(self):
            return {
                'created': self._created,
                'reused': self._reused,
                'pool_size': len(self._pool)
            }
    
    class PooledObject:
        def __init__(self):
            self.data = None
            self.processed = False
        
        def reset(self):
            self.data = None
            self.processed = False
        
        def process(self, data):
            self.data = data
            self.processed = True
            return f"Processed: {data}"
    
    # Pattern 2: Flyweight pattern
    class FlyweightFactory:
        _flyweights = {}
        
        @classmethod
        def get_flyweight(cls, shared_state):
            if shared_state not in cls._flyweights:
                cls._flyweights[shared_state] = Flyweight(shared_state)
            return cls._flyweights[shared_state]
        
        @classmethod
        def get_created_count(cls):
            return len(cls._flyweights)
    
    class Flyweight:
        def __init__(self, shared_state):
            self.shared_state = shared_state
        
        def operation(self, unique_state):
            return f"Shared: {self.shared_state}, Unique: {unique_state}"
    
    # Pattern 3: Lazy initialization
    class LazyProperty:
        def __init__(self, func):
            self.func = func
            self.name = func.__name__
        
        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            
            value = self.func(obj)
            setattr(obj, self.name, value)
            return value
    
    class LazyObject:
        def __init__(self, size):
            self.size = size
        
        @LazyProperty
        def expensive_data(self):
            print(f"Computing expensive data for size {self.size}")
            return list(range(self.size * 1000))
    
    # Test object pooling
    pool = ObjectPool(PooledObject, max_size=3)
    
    # Use objects from pool
    obj1 = pool.get()
    obj2 = pool.get()
    obj3 = pool.get()
    
    result1 = obj1.process("data1")
    result2 = obj2.process("data2")
    
    # Return to pool
    pool.put(obj1)
    pool.put(obj2)
    
    # Reuse objects
    obj4 = pool.get()  # Should reuse obj2
    obj5 = pool.get()  # Should reuse obj1
    
    pool_stats = pool.get_stats()
    
    # Test flyweight pattern
    flyweight1 = FlyweightFactory.get_flyweight("type_A")
    flyweight2 = FlyweightFactory.get_flyweight("type_B")
    flyweight3 = FlyweightFactory.get_flyweight("type_A")  # Should reuse
    
    flyweight_same = flyweight1 is flyweight3
    flyweight_result = flyweight1.operation("unique_data")
    flyweight_count = FlyweightFactory.get_created_count()
    
    # Test lazy initialization
    lazy_obj = LazyObject(100)
    
    # Property not computed yet
    has_expensive_data_initially = hasattr(lazy_obj, 'expensive_data')
    
    # Access property - should trigger computation
    data_length = len(lazy_obj.expensive_data)
    
    # Property should now be cached
    has_expensive_data_after = hasattr(lazy_obj, 'expensive_data')
    
    return {
        "object_pooling": {
            "pool_stats": pool_stats,
            "reuse_efficiency": pool_stats['reused'] / (pool_stats['created'] + pool_stats['reused'])
        },
        "flyweight_pattern": {
            "flyweight_reused": flyweight_same,
            "flyweight_result": flyweight_result,
            "total_flyweights_created": flyweight_count
        },
        "lazy_initialization": {
            "initially_computed": has_expensive_data_initially,
            "data_length": data_length,
            "cached_after_access": has_expensive_data_after
        }
    }

# Memory debugging utilities
class MemoryDebugger:
    """Utility class for memory debugging"""
    
    def __init__(self):
        self.snapshots = []
        self.tracking = False
    
    def start_tracking(self):
        """Start memory tracking"""
        if not self.tracking:
            tracemalloc.start()
            self.tracking = True
            self.take_snapshot("start")
    
    def stop_tracking(self):
        """Stop memory tracking"""
        if self.tracking:
            tracemalloc.stop()
            self.tracking = False
    
    def take_snapshot(self, label):
        """Take a memory snapshot"""
        if self.tracking:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((label, snapshot))
    
    def get_top_allocations(self, count=10):
        """Get top memory allocations"""
        if not self.snapshots:
            return []
        
        _, latest_snapshot = self.snapshots[-1]
        top_stats = latest_snapshot.statistics('lineno')
        
        return [
            {
                'file': stat.traceback.format()[-1],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            }
            for stat in top_stats[:count]
        ]
    
    def compare_snapshots(self, label1, label2):
        """Compare two snapshots"""
        snapshot1 = None
        snapshot2 = None
        
        for label, snapshot in self.snapshots:
            if label == label1:
                snapshot1 = snapshot
            elif label == label2:
                snapshot2 = snapshot
        
        if snapshot1 and snapshot2:
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            return [
                {
                    'file': stat.traceback.format()[-1],
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'count_diff': stat.count_diff
                }
                for stat in top_stats[:10]
            ]
        
        return []

def memory_debugging_demo():
    """Demonstrate memory debugging utilities"""
    
    debugger = MemoryDebugger()
    debugger.start_tracking()
    
    # Initial state
    debugger.take_snapshot("initial")
    
    # Allocate some memory
    large_data = []
    for i in range(10000):
        large_data.append({'id': i, 'data': f'item_{i}' * 10})
    
    debugger.take_snapshot("after_allocation")
    
    # Modify data
    for item in large_data[:5000]:
        item['processed'] = True
        item['timestamp'] = time.time()
    
    debugger.take_snapshot("after_modification")
    
    # Free some memory
    del large_data[:5000]
    gc.collect()
    
    debugger.take_snapshot("after_partial_cleanup")
    
    # Get analysis
    top_allocations = debugger.get_top_allocations(5)
    
    # Compare snapshots
    allocation_diff = debugger.compare_snapshots("initial", "after_allocation")
    cleanup_diff = debugger.compare_snapshots("after_modification", "after_partial_cleanup")
    
    debugger.stop_tracking()
    
    return {
        "top_allocations": top_allocations,
        "allocation_differences": allocation_diff[:3],  # Top 3
        "cleanup_differences": cleanup_diff[:3],  # Top 3
        "snapshots_taken": len(debugger.snapshots)
    }

# Comprehensive testing
def run_all_memory_demos():
    """Execute all memory management demonstrations"""
    demo_functions = [
        ('reference_counting', reference_counting_demo),
        ('garbage_collection', garbage_collection_demo),
        ('weak_references', weak_references_demo),
        ('memory_profiling', memory_profiling_demo),
        ('slots_optimization', slots_optimization_demo),
        ('memory_leaks', memory_leak_demo),
        ('object_lifecycle', object_lifecycle_demo),
        ('optimization_patterns', memory_optimization_patterns),
        ('memory_debugging', memory_debugging_demo)
    ]
    
    results = {}
    for name, func in demo_functions:
        try:
            result = func()
            results[name] = result
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Add system memory info
    process = psutil.Process(os.getpid())
    results['system_info'] = {
        'memory_percent': process.memory_percent(),
        'memory_rss_mb': round(process.memory_info().rss / 1024 / 1024, 2),
        'memory_vms_mb': round(process.memory_info().vms / 1024 / 1024, 2)
    }
    
    return results

if __name__ == "__main__":
    print("=== Python Memory Management Demo ===")
    
    # Run all demonstrations
    all_results = run_all_memory_demos()
    
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
    
    print("\n=== MEMORY MANAGEMENT CONCEPTS ===")
    
    concepts = {
        "Reference Counting": "CPython tracks object references, deletes when count reaches zero",
        "Garbage Collection": "Handles circular references using cycle detection",
        "Weak References": "References that don't increase reference count",
        "__slots__": "Restricts instance attributes to save memory",
        "Object Pooling": "Reuse objects to reduce allocation overhead",
        "Flyweight Pattern": "Share common state between multiple objects",
        "Memory Profiling": "Tools to track memory usage and detect leaks",
        "Finalizers": "Cleanup code that runs when objects are garbage collected"
    }
    
    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    
    print("\n=== MEMORY OPTIMIZATION TECHNIQUES ===")
    
    techniques = {
        "Use __slots__": "Reduce memory overhead for classes with fixed attributes",
        "Weak References": "Break circular references and implement caches",
        "Object Pooling": "Reuse expensive-to-create objects",
        "Lazy Loading": "Load data only when needed",
        "Generator Expressions": "Process large datasets without loading all into memory",
        "Array Module": "Use array.array for homogeneous numeric data",
        "Memory Mapping": "Use mmap for large files",
        "Interning": "Reuse common strings and small integers",
        "del Statements": "Explicitly remove references to large objects",
        "gc.collect()": "Force garbage collection when needed"
    }
    
    for technique, description in techniques.items():
        print(f"  {technique}: {description}")
    
    print("\n=== MEMORY DEBUGGING TOOLS ===")
    
    tools = {
        "tracemalloc": "Built-in memory profiler for Python applications",
        "gc module": "Garbage collection interface and debugging",
        "sys.getsizeof()": "Get size of individual objects",
        "psutil": "System and process memory monitoring",
        "weakref module": "Weak reference utilities",
        "memory_profiler": "Line-by-line memory profiling (external)",
        "objgraph": "Visualize object references (external)",
        "pympler": "Advanced memory profiling (external)"
    }
    
    for tool, description in tools.items():
        print(f"  {tool}: {description}")
    
    print("\n=== BEST PRACTICES ===")
    
    best_practices = [
        "Use __slots__ for classes with many instances and fixed attributes",
        "Implement proper cleanup in __del__ and finalizers",
        "Use weak references to break circular dependencies",
        "Monitor memory usage in production applications",
        "Profile memory usage during development",
        "Prefer generators over lists for large datasets",
        "Use context managers for resource management",
        "Clear large data structures explicitly when done",
        "Avoid storing references in global variables unnecessarily",
        "Use object pooling for frequently created/destroyed objects",
        "Test for memory leaks in long-running applications",
        "Understand the trade-offs between memory and performance"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")
    
    print("\n=== Memory Management Complete! ===")
    print("  Advanced memory optimization and debugging techniques mastered")
