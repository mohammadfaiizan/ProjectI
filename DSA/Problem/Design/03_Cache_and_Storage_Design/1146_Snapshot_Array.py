"""
1146. Snapshot Array - Multiple Approaches
Difficulty: Medium

Implement a SnapshotArray that supports the following interface:
- SnapshotArray(int length) initializes an array-like data structure with the given length. Initially, each element equals 0.
- void set(index, val) sets the element at the given index to be equal to val.
- int snap() takes a snapshot of the array and returns the snap_id (the total number of times we called snap() minus 1).
- int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import bisect

class SnapshotArraySimple:
    """
    Approach 1: Store Complete Snapshots
    
    Store complete array copy for each snapshot.
    
    Time Complexity:
    - __init__: O(length)
    - set: O(1)
    - snap: O(length)
    - get: O(1)
    
    Space Complexity: O(length * snapshots)
    """
    
    def __init__(self, length: int):
        self.length = length
        self.current_array = [0] * length
        self.snapshots = []
        self.snap_id = 0
    
    def set(self, index: int, val: int) -> None:
        self.current_array[index] = val
    
    def snap(self) -> int:
        # Store complete snapshot
        self.snapshots.append(self.current_array[:])
        self.snap_id += 1
        return self.snap_id - 1
    
    def get(self, index: int, snap_id: int) -> int:
        if snap_id < len(self.snapshots):
            return self.snapshots[snap_id][index]
        return 0

class SnapshotArrayOptimal:
    """
    Approach 2: Store Only Changes (Optimal)
    
    Store only changed values with their snapshot IDs.
    
    Time Complexity:
    - __init__: O(length)
    - set: O(1)
    - snap: O(1)
    - get: O(log snapshots) using binary search
    
    Space Complexity: O(changes)
    """
    
    def __init__(self, length: int):
        self.length = length
        self.snap_id = 0
        # For each index, store list of (snap_id, value) pairs
        self.history = [[(0, 0)] for _ in range(length)]
    
    def set(self, index: int, val: int) -> None:
        # If this is the first change in current snapshot, add it
        if self.history[index][-1][0] == self.snap_id:
            # Update the current snapshot's value
            self.history[index][-1] = (self.snap_id, val)
        else:
            # Add new entry for current snapshot
            self.history[index].append((self.snap_id, val))
    
    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1
    
    def get(self, index: int, snap_id: int) -> int:
        # Binary search for the latest change <= snap_id
        history_list = self.history[index]
        
        left, right = 0, len(history_list) - 1
        result_value = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if history_list[mid][0] <= snap_id:
                result_value = history_list[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        return result_value

class SnapshotArrayHashMap:
    """
    Approach 3: HashMap-based Storage
    
    Use hash maps to store sparse changes.
    
    Time Complexity:
    - __init__: O(1)
    - set: O(1)
    - snap: O(1)
    - get: O(log snapshots)
    
    Space Complexity: O(changes)
    """
    
    def __init__(self, length: int):
        self.length = length
        self.snap_id = 0
        # Map: index -> list of (snap_id, value)
        self.data = defaultdict(list)
    
    def set(self, index: int, val: int) -> None:
        if index not in self.data:
            self.data[index] = [(0, 0)]  # Initialize with default
        
        # Check if we already have an entry for current snapshot
        if self.data[index] and self.data[index][-1][0] == self.snap_id:
            # Update existing entry
            self.data[index][-1] = (self.snap_id, val)
        else:
            # Add new entry
            self.data[index].append((self.snap_id, val))
    
    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1
    
    def get(self, index: int, snap_id: int) -> int:
        if index not in self.data:
            return 0
        
        # Binary search using bisect
        history = self.data[index]
        
        # Find rightmost entry with snap_id <= target snap_id
        pos = bisect.bisect_right(history, (snap_id, float('inf'))) - 1
        
        if pos >= 0:
            return history[pos][1]
        
        return 0

class SnapshotArrayAdvanced:
    """
    Approach 4: Advanced with Compression and Analytics
    
    Enhanced version with data compression and usage analytics.
    
    Time Complexity:
    - __init__: O(length)
    - set: O(1) amortized
    - snap: O(1)
    - get: O(log snapshots)
    
    Space Complexity: O(changes + metadata)
    """
    
    def __init__(self, length: int):
        self.length = length
        self.snap_id = 0
        self.history = [[(0, 0)] for _ in range(length)]
        
        # Analytics
        self.total_sets = 0
        self.total_snaps = 0
        self.total_gets = 0
        self.set_operations_per_index = [0] * length
        self.get_operations_per_index = [0] * length
        
        # Compression settings
        self.compression_threshold = 100
        self.compressed_snapshots = {}
    
    def set(self, index: int, val: int) -> None:
        self.total_sets += 1
        self.set_operations_per_index[index] += 1
        
        if self.history[index][-1][0] == self.snap_id:
            self.history[index][-1] = (self.snap_id, val)
        else:
            self.history[index].append((self.snap_id, val))
        
        # Periodic compression
        if self.total_sets % self.compression_threshold == 0:
            self._compress_old_data()
    
    def snap(self) -> int:
        self.total_snaps += 1
        self.snap_id += 1
        return self.snap_id - 1
    
    def get(self, index: int, snap_id: int) -> int:
        self.total_gets += 1
        self.get_operations_per_index[index] += 1
        
        # Check compressed snapshots first
        if snap_id in self.compressed_snapshots:
            if index in self.compressed_snapshots[snap_id]:
                return self.compressed_snapshots[snap_id][index]
        
        # Binary search in history
        history_list = self.history[index]
        
        left, right = 0, len(history_list) - 1
        result_value = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if history_list[mid][0] <= snap_id:
                result_value = history_list[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        return result_value
    
    def _compress_old_data(self) -> None:
        """Compress old snapshot data"""
        # Compress snapshots older than 50 snaps
        compress_threshold = max(0, self.snap_id - 50)
        
        for snap_to_compress in range(compress_threshold):
            if snap_to_compress not in self.compressed_snapshots:
                compressed_data = {}
                
                # For each index, find value at this snapshot
                for index in range(self.length):
                    if self.set_operations_per_index[index] > 0:  # Only compress if used
                        value = self._get_value_at_snapshot(index, snap_to_compress)
                        if value != 0:  # Only store non-zero values
                            compressed_data[index] = value
                
                if compressed_data:
                    self.compressed_snapshots[snap_to_compress] = compressed_data
    
    def _get_value_at_snapshot(self, index: int, snap_id: int) -> int:
        """Helper to get value at specific snapshot"""
        history_list = self.history[index]
        
        for snap, value in reversed(history_list):
            if snap <= snap_id:
                return value
        
        return 0
    
    def get_analytics(self) -> dict:
        """Get usage analytics"""
        hot_indices = [i for i, count in enumerate(self.set_operations_per_index) if count > 0]
        
        return {
            'total_operations': self.total_sets + self.total_snaps + self.total_gets,
            'set_operations': self.total_sets,
            'snap_operations': self.total_snaps,
            'get_operations': self.total_gets,
            'current_snap_id': self.snap_id,
            'hot_indices_count': len(hot_indices),
            'compressed_snapshots': len(self.compressed_snapshots),
            'average_sets_per_index': self.total_sets / max(1, len(hot_indices))
        }
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        total_history_entries = sum(len(hist) for hist in self.history)
        compressed_entries = sum(len(data) for data in self.compressed_snapshots.values())
        
        return {
            'history_entries': total_history_entries,
            'compressed_entries': compressed_entries,
            'total_storage_units': total_history_entries + compressed_entries,
            'compression_ratio': compressed_entries / max(1, total_history_entries)
        }

class SnapshotArrayDifferential:
    """
    Approach 5: Differential Storage
    
    Store differences between snapshots for better compression.
    
    Time Complexity:
    - __init__: O(length)
    - set: O(1)
    - snap: O(changed_indices)
    - get: O(snapshots) worst case
    
    Space Complexity: O(changes)
    """
    
    def __init__(self, length: int):
        self.length = length
        self.snap_id = 0
        self.current_state = [0] * length
        self.snapshots = []  # List of {index: value} diffs
        self.pending_changes = {}
    
    def set(self, index: int, val: int) -> None:
        self.current_state[index] = val
        self.pending_changes[index] = val
    
    def snap(self) -> int:
        # Store only the changes since last snapshot
        if self.pending_changes:
            self.snapshots.append(dict(self.pending_changes))
            self.pending_changes.clear()
        else:
            self.snapshots.append({})
        
        self.snap_id += 1
        return self.snap_id - 1
    
    def get(self, index: int, snap_id: int) -> int:
        if snap_id >= len(self.snapshots):
            return self.current_state[index]
        
        # Reconstruct value by applying diffs up to snap_id
        value = 0
        
        for i in range(snap_id + 1):
            if index in self.snapshots[i]:
                value = self.snapshots[i][index]
        
        return value


def test_snapshot_array_basic():
    """Test basic SnapshotArray functionality"""
    print("=== Testing Basic SnapshotArray Functionality ===")
    
    implementations = [
        ("Simple Complete", SnapshotArraySimple),
        ("Optimal Changes", SnapshotArrayOptimal),
        ("HashMap Based", SnapshotArrayHashMap),
        ("Advanced", SnapshotArrayAdvanced),
        ("Differential", SnapshotArrayDifferential)
    ]
    
    for name, ArrayClass in implementations:
        print(f"\n{name}:")
        
        arr = ArrayClass(3)
        
        # Test sequence from problem
        operations = [
            ("set", 0, 5), ("snap", None, None), ("set", 0, 6),
            ("get", 0, 0), ("snap", None, None), ("get", 0, 1)
        ]
        
        for op, index, val_or_snap in operations:
            if op == "set":
                arr.set(index, val_or_snap)
                print(f"  set({index}, {val_or_snap})")
            elif op == "snap":
                result = arr.snap()
                print(f"  snap(): {result}")
            elif op == "get":
                result = arr.get(index, val_or_snap)
                print(f"  get({index}, {val_or_snap}): {result}")

def test_snapshot_array_edge_cases():
    """Test edge cases"""
    print("\n=== Testing SnapshotArray Edge Cases ===")
    
    arr = SnapshotArrayOptimal(5)
    
    # Test getting before any changes
    print("Getting before any changes:")
    for i in range(5):
        result = arr.get(i, 0)
        print(f"  get({i}, 0): {result}")
    
    # Test multiple sets on same index
    print(f"\nMultiple sets on same index:")
    arr.set(0, 10)
    arr.set(0, 20)
    arr.set(0, 30)
    snap_id = arr.snap()
    print(f"  set(0, 10), set(0, 20), set(0, 30), snap(): {snap_id}")
    
    result = arr.get(0, snap_id)
    print(f"  get(0, {snap_id}): {result}")
    
    # Test multiple snapshots without changes
    print(f"\nMultiple snapshots without changes:")
    snap1 = arr.snap()
    snap2 = arr.snap()
    snap3 = arr.snap()
    
    print(f"  Three consecutive snaps: {snap1}, {snap2}, {snap3}")
    
    for snap in [snap1, snap2, snap3]:
        result = arr.get(0, snap)
        print(f"    get(0, {snap}): {result}")
    
    # Test large index values
    print(f"\nLarge array test:")
    large_arr = SnapshotArrayHashMap(10000)
    
    large_arr.set(9999, 42)
    large_snap = large_arr.snap()
    result = large_arr.get(9999, large_snap)
    
    print(f"  set(9999, 42), snap(), get(9999, snap): {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple Complete", SnapshotArraySimple),
        ("Optimal Changes", SnapshotArrayOptimal),
        ("HashMap Based", SnapshotArrayHashMap)
    ]
    
    array_size = 1000
    num_operations = 1000
    
    for name, ArrayClass in implementations:
        arr = ArrayClass(array_size)
        
        # Time set operations
        start_time = time.time()
        for i in range(num_operations):
            arr.set(i % array_size, i)
        set_time = (time.time() - start_time) * 1000
        
        # Time snap operations
        start_time = time.time()
        snaps = []
        for _ in range(10):
            snap_id = arr.snap()
            snaps.append(snap_id)
        snap_time = (time.time() - start_time) * 1000
        
        # Time get operations
        start_time = time.time()
        for i in range(num_operations):
            snap_id = snaps[i % len(snaps)]
            index = i % array_size
            arr.get(index, snap_id)
        get_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_operations} sets: {set_time:.2f}ms")
        print(f"    10 snaps: {snap_time:.2f}ms")
        print(f"    {num_operations} gets: {get_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    arr = SnapshotArrayAdvanced(100)
    
    # Create usage pattern
    operations = [
        (0, 10), (1, 20), (2, 30),  # Hot indices
        (50, 100), (99, 200)       # Sparse usage
    ]
    
    print("Creating usage pattern:")
    for index, value in operations:
        arr.set(index, value)
        print(f"  set({index}, {value})")
    
    # Create multiple snapshots
    snap_ids = []
    for i in range(5):
        snap_id = arr.snap()
        snap_ids.append(snap_id)
        print(f"  snap(): {snap_id}")
        
        # Make some more changes
        arr.set(i, i * 10)
    
    # Get analytics
    analytics = arr.get_analytics()
    print(f"\nAnalytics:")
    for key, value in analytics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Get memory usage
    memory = arr.get_memory_usage()
    print(f"\nMemory usage:")
    for key, value in memory.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Version control system
    print("Application 1: File Version Control")
    
    file_system = SnapshotArrayOptimal(1000)  # 1000 files
    
    # Simulate file edits
    edits = [
        (100, 1001),  # File 100, version 1001
        (101, 2001),  # File 101, version 2001
        (100, 1002),  # File 100, version 1002
    ]
    
    print("  File edits:")
    for file_id, version in edits:
        file_system.set(file_id, version)
        print(f"    Edit file {file_id} to version {version}")
    
    commit1 = file_system.snap()
    print(f"    Commit 1: {commit1}")
    
    # More edits
    file_system.set(102, 3001)
    file_system.set(100, 1003)
    
    commit2 = file_system.snap()
    print(f"    Commit 2: {commit2}")
    
    # Check file versions at different commits
    print("  File version history:")
    for commit in [commit1, commit2]:
        version = file_system.get(100, commit)
        print(f"    File 100 at commit {commit}: version {version}")
    
    # Application 2: Database backup system
    print(f"\nApplication 2: Database Table Snapshots")
    
    table = SnapshotArrayHashMap(10000)  # 10k rows
    
    # Simulate database updates
    updates = [
        (1000, 12345),  # Row 1000, new value
        (2000, 23456),  # Row 2000, new value
        (1000, 12346),  # Row 1000, updated again
    ]
    
    print("  Database updates:")
    for row, value in updates:
        table.set(row, value)
        print(f"    Update row {row} to {value}")
    
    backup1 = table.snap()
    print(f"    Backup 1: {backup1}")
    
    # Simulate more updates
    table.set(3000, 34567)
    table.set(2000, 23457)
    
    backup2 = table.snap()
    print(f"    Backup 2: {backup2}")
    
    # Query historical data
    print("  Historical data query:")
    for backup in [backup1, backup2]:
        value = table.get(1000, backup)
        print(f"    Row 1000 at backup {backup}: {value}")
    
    # Application 3: Configuration management
    print(f"\nApplication 3: Configuration Management")
    
    config = SnapshotArrayAdvanced(50)  # 50 config parameters
    
    # Initial configuration
    config.set(0, 8080)   # Port
    config.set(1, 1000)   # Max connections
    config.set(2, 300)    # Timeout
    
    baseline = config.snap()
    print(f"    Baseline config: {baseline}")
    
    # Performance tuning
    config.set(1, 2000)   # Increase max connections
    config.set(2, 600)    # Increase timeout
    
    perf_config = config.snap()
    print(f"    Performance config: {perf_config}")
    
    # Security hardening
    config.set(0, 8443)   # HTTPS port
    config.set(1, 500)    # Reduce connections for security
    
    secure_config = config.snap()
    print(f"    Secure config: {secure_config}")
    
    # Compare configurations
    print("  Configuration comparison:")
    configs = [("Baseline", baseline), ("Performance", perf_config), ("Secure", secure_config)]
    
    for name, config_id in configs:
        port = config.get(0, config_id)
        max_conn = config.get(1, config_id)
        timeout = config.get(2, config_id)
        print(f"    {name}: port={port}, max_conn={max_conn}, timeout={timeout}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple Complete", SnapshotArraySimple),
        ("Optimal Changes", SnapshotArrayOptimal),
        ("HashMap Based", SnapshotArrayHashMap)
    ]
    
    array_size = 1000
    num_snapshots = 50
    
    for name, ArrayClass in implementations:
        arr = ArrayClass(array_size)
        
        # Create sparse updates (only 10% of array changes)
        for snap in range(num_snapshots):
            # Update 10% of array
            for i in range(0, array_size, 10):
                arr.set(i, snap * 100 + i)
            
            arr.snap()
        
        # Estimate memory usage
        if hasattr(arr, 'snapshots') and isinstance(arr.snapshots[0], list):
            # Simple approach stores complete arrays
            memory_estimate = array_size * num_snapshots
        elif hasattr(arr, 'history'):
            # Count actual history entries
            memory_estimate = sum(len(hist) for hist in arr.history)
        elif hasattr(arr, 'data'):
            # Count entries in hash map
            memory_estimate = sum(len(hist) for hist in arr.data.values())
        else:
            memory_estimate = "Unknown"
        
        print(f"  {name}: ~{memory_estimate} storage units")

def test_sparse_vs_dense_patterns():
    """Test sparse vs dense update patterns"""
    print("\n=== Testing Sparse vs Dense Patterns ===")
    
    import time
    
    arr = SnapshotArrayOptimal(10000)
    
    # Test sparse pattern (few indices updated)
    print("Sparse pattern (1% of indices):")
    start_time = time.time()
    
    for snap in range(100):
        # Update only 1% of indices
        for i in range(0, 10000, 100):
            arr.set(i, snap)
        arr.snap()
    
    sparse_time = (time.time() - start_time) * 1000
    
    # Test dense pattern (many indices updated)
    arr2 = SnapshotArrayOptimal(1000)
    
    print("Dense pattern (50% of indices):")
    start_time = time.time()
    
    for snap in range(20):
        # Update 50% of indices
        for i in range(0, 1000, 2):
            arr2.set(i, snap)
        arr2.snap()
    
    dense_time = (time.time() - start_time) * 1000
    
    print(f"  Sparse: {sparse_time:.2f}ms")
    print(f"  Dense: {dense_time:.2f}ms")

def stress_test_snapshot_array():
    """Stress test snapshot array"""
    print("\n=== Stress Testing Snapshot Array ===")
    
    import time
    
    arr = SnapshotArrayOptimal(100000)
    
    print("Large scale test:")
    start_time = time.time()
    
    # Many operations
    snap_ids = []
    
    for i in range(10000):
        # Update random indices
        index = i % 100000
        arr.set(index, i)
        
        # Periodic snapshots
        if i % 1000 == 0:
            snap_id = arr.snap()
            snap_ids.append(snap_id)
    
    operation_time = (time.time() - start_time) * 1000
    
    # Test retrieval performance
    start_time = time.time()
    
    for i in range(1000):
        snap_id = snap_ids[i % len(snap_ids)]
        index = i % 100000
        arr.get(index, snap_id)
    
    retrieval_time = (time.time() - start_time) * 1000
    
    print(f"  10k operations: {operation_time:.2f}ms")
    print(f"  1k retrievals: {retrieval_time:.2f}ms")

def benchmark_get_performance():
    """Benchmark get operation performance"""
    print("\n=== Benchmarking Get Performance ===")
    
    import time
    
    arr = SnapshotArrayOptimal(1000)
    
    # Create many snapshots with changes
    snap_ids = []
    for i in range(100):
        arr.set(i % 1000, i)
        snap_id = arr.snap()
        snap_ids.append(snap_id)
    
    # Benchmark gets at different snapshot ages
    test_cases = [
        ("Recent snapshots", snap_ids[-10:]),
        ("Middle snapshots", snap_ids[40:50]),
        ("Old snapshots", snap_ids[:10])
    ]
    
    for test_name, test_snaps in test_cases:
        start_time = time.time()
        
        for _ in range(1000):
            snap_id = test_snaps[_ % len(test_snaps)]
            index = _ % 1000
            arr.get(index, snap_id)
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {test_name}: {elapsed:.2f}ms for 1000 gets")

if __name__ == "__main__":
    test_snapshot_array_basic()
    test_snapshot_array_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    test_sparse_vs_dense_patterns()
    stress_test_snapshot_array()
    benchmark_get_performance()

"""
Snapshot Array Design demonstrates key concepts:

Core Approaches:
1. Simple Complete - Store full array copy for each snapshot
2. Optimal Changes - Store only changed values with binary search
3. HashMap Based - Use sparse storage with hash maps
4. Advanced - Enhanced with compression and analytics
5. Differential - Store differences between snapshots

Key Design Principles:
- Time vs space trade-offs in snapshot storage
- Binary search for efficient historical lookups
- Sparse data structure optimization
- Lazy evaluation and compression strategies

Performance Characteristics:
- Simple: O(length) snap, O(1) get, O(length * snaps) space
- Optimal: O(1) snap, O(log snaps) get, O(changes) space
- HashMap: O(1) snap, O(log snaps) get, O(changes) space
- Advanced: Compressed storage with analytics

Real-world Applications:
- Version control systems (Git-like snapshots)
- Database backup and point-in-time recovery
- Configuration management with rollback capability
- Undo/redo systems in applications
- Time-series data with historical queries
- A/B testing with configuration snapshots

The optimal approach storing only changes with binary search
is most commonly used due to its excellent space efficiency
and logarithmic query time for historical data access.
"""
