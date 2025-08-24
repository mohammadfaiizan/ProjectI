"""
Memory Optimized Structures - Multiple Approaches
Difficulty: Medium

Implement memory-optimized versions of queue and stack data structures.
Focus on reducing memory overhead, cache efficiency, and space-time tradeoffs.
"""

import sys
import array
from typing import Any, Optional, List, Union
import struct
import mmap
import tempfile
import os

class SlottedStack:
    """
    Approach 1: Slotted Stack for Memory Efficiency
    
    Use __slots__ to reduce memory overhead per instance.
    
    Memory: Reduces overhead by ~40-50% compared to regular classes
    """
    
    __slots__ = ['_items', '_capacity', '_size', '_growth_factor']
    
    def __init__(self, initial_capacity: int = 16, growth_factor: float = 1.5):
        self._items = [None] * initial_capacity
        self._capacity = initial_capacity
        self._size = 0
        self._growth_factor = growth_factor
    
    def push(self, item: Any) -> None:
        """Push item with automatic resizing"""
        if self._size >= self._capacity:
            self._resize()
        
        self._items[self._size] = item
        self._size += 1
    
    def pop(self) -> Any:
        """Pop item and shrink if needed"""
        if self._size == 0:
            raise IndexError("Stack underflow")
        
        self._size -= 1
        item = self._items[self._size]
        self._items[self._size] = None  # Help GC
        
        # Shrink if using less than 25% of capacity
        if self._size < self._capacity // 4 and self._capacity > 16:
            self._shrink()
        
        return item
    
    def peek(self) -> Any:
        """Peek at top item"""
        if self._size == 0:
            raise IndexError("Stack is empty")
        return self._items[self._size - 1]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def _resize(self) -> None:
        """Resize array to larger capacity"""
        old_capacity = self._capacity
        self._capacity = int(old_capacity * self._growth_factor)
        
        new_items = [None] * self._capacity
        for i in range(self._size):
            new_items[i] = self._items[i]
        
        self._items = new_items
    
    def _shrink(self) -> None:
        """Shrink array to smaller capacity"""
        self._capacity = max(16, self._capacity // 2)
        
        new_items = [None] * self._capacity
        for i in range(self._size):
            new_items[i] = self._items[i]
        
        self._items = new_items
    
    def memory_usage(self) -> dict:
        """Get memory usage statistics"""
        return {
            'items_memory': sys.getsizeof(self._items),
            'capacity': self._capacity,
            'size': self._size,
            'utilization': self._size / self._capacity if self._capacity > 0 else 0,
            'overhead': self._capacity - self._size
        }


class ArrayBasedQueue:
    """
    Approach 2: Array-based Queue with Type Optimization
    
    Use Python's array module for memory-efficient storage of numeric types.
    
    Memory: ~75% less memory for numeric types compared to lists
    """
    
    def __init__(self, typecode: str = 'i', initial_capacity: int = 16):
        """
        Initialize with specific type code:
        'i' - signed int (4 bytes)
        'f' - float (4 bytes)
        'd' - double (8 bytes)
        'l' - long (8 bytes)
        """
        self._typecode = typecode
        self._items = array.array(typecode)
        self._capacity = initial_capacity
        self._front = 0
        self._rear = 0
        self._size = 0
        
        # Pre-allocate space
        for _ in range(initial_capacity):
            self._items.append(0)
    
    def enqueue(self, item: Union[int, float]) -> None:
        """Enqueue numeric item"""
        if self._size >= self._capacity:
            self._resize()
        
        self._items[self._rear] = item
        self._rear = (self._rear + 1) % self._capacity
        self._size += 1
    
    def dequeue(self) -> Union[int, float]:
        """Dequeue numeric item"""
        if self._size == 0:
            raise IndexError("Queue underflow")
        
        item = self._items[self._front]
        self._front = (self._front + 1) % self._capacity
        self._size -= 1
        
        return item
    
    def front(self) -> Union[int, float]:
        """Peek at front item"""
        if self._size == 0:
            raise IndexError("Queue is empty")
        return self._items[self._front]
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def _resize(self) -> None:
        """Resize circular array"""
        old_capacity = self._capacity
        self._capacity *= 2
        
        new_items = array.array(self._typecode)
        
        # Copy existing items in order
        for i in range(self._size):
            index = (self._front + i) % old_capacity
            new_items.append(self._items[index])
        
        # Fill remaining space with zeros
        for _ in range(self._capacity - self._size):
            new_items.append(0)
        
        self._items = new_items
        self._front = 0
        self._rear = self._size
    
    def memory_usage(self) -> dict:
        """Get memory usage statistics"""
        item_size = self._items.itemsize
        return {
            'array_memory': sys.getsizeof(self._items),
            'item_size': item_size,
            'total_items_size': self._capacity * item_size,
            'used_items_size': self._size * item_size,
            'utilization': self._size / self._capacity if self._capacity > 0 else 0
        }


class CompactStack:
    """
    Approach 3: Compact Stack using Struct Packing
    
    Use struct packing for memory-efficient storage of fixed-size data.
    
    Memory: Optimal packing for structured data
    """
    
    def __init__(self, item_format: str = 'i'):
        """
        Initialize with struct format:
        'i' - int (4 bytes)
        'f' - float (4 bytes)
        'd' - double (8 bytes)
        'ii' - two ints (8 bytes)
        """
        self._format = item_format
        self._item_size = struct.calcsize(item_format)
        self._data = bytearray()
        self._count = 0
    
    def push(self, *values) -> None:
        """Push structured data"""
        packed_data = struct.pack(self._format, *values)
        self._data.extend(packed_data)
        self._count += 1
    
    def pop(self) -> tuple:
        """Pop structured data"""
        if self._count == 0:
            raise IndexError("Stack underflow")
        
        start_pos = len(self._data) - self._item_size
        packed_data = self._data[start_pos:]
        
        # Remove from data
        self._data = self._data[:start_pos]
        self._count -= 1
        
        # Unpack and return
        return struct.unpack(self._format, packed_data)
    
    def peek(self) -> tuple:
        """Peek at top structured data"""
        if self._count == 0:
            raise IndexError("Stack is empty")
        
        start_pos = len(self._data) - self._item_size
        packed_data = self._data[start_pos:]
        
        return struct.unpack(self._format, packed_data)
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._count == 0
    
    def size(self) -> int:
        """Get size"""
        return self._count
    
    def memory_usage(self) -> dict:
        """Get memory usage statistics"""
        return {
            'data_size': len(self._data),
            'item_size': self._item_size,
            'count': self._count,
            'theoretical_size': self._count * self._item_size,
            'overhead': len(self._data) - (self._count * self._item_size)
        }


class MemoryMappedQueue:
    """
    Approach 4: Memory-Mapped Queue for Large Data
    
    Use memory mapping for very large queues that exceed RAM.
    
    Memory: Can handle datasets larger than available RAM
    """
    
    def __init__(self, max_items: int = 1000000, item_size: int = 8):
        self._max_items = max_items
        self._item_size = item_size
        self._total_size = max_items * item_size
        
        # Create temporary file
        self._temp_file = tempfile.NamedTemporaryFile(delete=False)
        self._temp_file.write(b'\x00' * self._total_size)
        self._temp_file.flush()
        
        # Memory map the file
        self._mmap = mmap.mmap(self._temp_file.fileno(), self._total_size)
        
        self._front = 0
        self._rear = 0
        self._size = 0
    
    def enqueue(self, data: bytes) -> None:
        """Enqueue binary data"""
        if self._size >= self._max_items:
            raise OverflowError("Queue overflow")
        
        if len(data) > self._item_size:
            raise ValueError(f"Data too large: {len(data)} > {self._item_size}")
        
        # Pad data to item size
        padded_data = data.ljust(self._item_size, b'\x00')
        
        # Write to memory-mapped location
        start_pos = self._rear * self._item_size
        self._mmap[start_pos:start_pos + self._item_size] = padded_data
        
        self._rear = (self._rear + 1) % self._max_items
        self._size += 1
    
    def dequeue(self) -> bytes:
        """Dequeue binary data"""
        if self._size == 0:
            raise IndexError("Queue underflow")
        
        # Read from memory-mapped location
        start_pos = self._front * self._item_size
        data = self._mmap[start_pos:start_pos + self._item_size]
        
        self._front = (self._front + 1) % self._max_items
        self._size -= 1
        
        # Remove padding
        return data.rstrip(b'\x00')
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def close(self) -> None:
        """Close memory map and clean up"""
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_temp_file'):
            self._temp_file.close()
            os.unlink(self._temp_file.name)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class PooledStack:
    """
    Approach 5: Object Pool Stack
    
    Use object pooling to reduce allocation/deallocation overhead.
    
    Memory: Reduces GC pressure and allocation overhead
    """
    
    class Node:
        __slots__ = ['data', 'next']
        
        def __init__(self, data=None, next_node=None):
            self.data = data
            self.next = next_node
    
    def __init__(self, pool_size: int = 100):
        self._top = None
        self._size = 0
        
        # Pre-allocate node pool
        self._node_pool = []
        for _ in range(pool_size):
            self._node_pool.append(self.Node())
    
    def push(self, item: Any) -> None:
        """Push using pooled nodes"""
        if self._node_pool:
            # Reuse node from pool
            node = self._node_pool.pop()
            node.data = item
            node.next = self._top
        else:
            # Create new node if pool is empty
            node = self.Node(item, self._top)
        
        self._top = node
        self._size += 1
    
    def pop(self) -> Any:
        """Pop and return node to pool"""
        if self._top is None:
            raise IndexError("Stack underflow")
        
        node = self._top
        data = node.data
        self._top = node.next
        self._size -= 1
        
        # Return node to pool
        node.data = None
        node.next = None
        self._node_pool.append(node)
        
        return data
    
    def peek(self) -> Any:
        """Peek at top item"""
        if self._top is None:
            raise IndexError("Stack is empty")
        return self._top.data
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._top is None
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def pool_stats(self) -> dict:
        """Get pool statistics"""
        return {
            'pool_size': len(self._node_pool),
            'active_nodes': self._size,
            'total_nodes': len(self._node_pool) + self._size
        }


class CacheOptimizedQueue:
    """
    Approach 6: Cache-Optimized Queue
    
    Optimize for CPU cache efficiency with data locality.
    
    Performance: Better cache utilization for sequential access
    """
    
    def __init__(self, block_size: int = 64):  # Typical cache line size
        self._block_size = block_size
        self._blocks = []
        self._current_block = []
        self._front_block_idx = 0
        self._front_item_idx = 0
        self._rear_block_idx = 0
        self._rear_item_idx = 0
        self._size = 0
    
    def enqueue(self, item: Any) -> None:
        """Enqueue with cache optimization"""
        # Create new block if current is full
        if len(self._current_block) >= self._block_size:
            self._blocks.append(self._current_block)
            self._current_block = []
            self._rear_block_idx += 1
            self._rear_item_idx = 0
        
        self._current_block.append(item)
        self._rear_item_idx += 1
        self._size += 1
    
    def dequeue(self) -> Any:
        """Dequeue with cache optimization"""
        if self._size == 0:
            raise IndexError("Queue underflow")
        
        # Get item from front block
        if self._front_block_idx < len(self._blocks):
            block = self._blocks[self._front_block_idx]
        else:
            block = self._current_block
        
        item = block[self._front_item_idx]
        self._front_item_idx += 1
        self._size -= 1
        
        # Move to next block if current is exhausted
        if (self._front_block_idx < len(self._blocks) and 
            self._front_item_idx >= len(self._blocks[self._front_block_idx])):
            self._front_block_idx += 1
            self._front_item_idx = 0
        
        return item
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return self._size == 0
    
    def size(self) -> int:
        """Get size"""
        return self._size
    
    def cache_stats(self) -> dict:
        """Get cache optimization statistics"""
        return {
            'total_blocks': len(self._blocks) + (1 if self._current_block else 0),
            'block_size': self._block_size,
            'front_block': self._front_block_idx,
            'rear_block': self._rear_block_idx,
            'items_per_cache_line': self._block_size
        }


def test_memory_optimized_structures():
    """Test memory-optimized structures"""
    print("=== Testing Memory-Optimized Structures ===")
    
    # Test SlottedStack
    print("\n--- Slotted Stack ---")
    slotted_stack = SlottedStack()
    
    for i in range(10):
        slotted_stack.push(i)
    
    print(f"Pushed 10 items, size: {slotted_stack.size()}")
    print(f"Memory usage: {slotted_stack.memory_usage()}")
    
    for _ in range(5):
        item = slotted_stack.pop()
        print(f"Popped: {item}")
    
    print(f"After popping 5 items: {slotted_stack.memory_usage()}")
    
    # Test ArrayBasedQueue
    print("\n--- Array-Based Queue ---")
    array_queue = ArrayBasedQueue('i')  # Integer array
    
    for i in range(8):
        array_queue.enqueue(i * 10)
    
    print(f"Enqueued 8 items, size: {array_queue.size()}")
    print(f"Memory usage: {array_queue.memory_usage()}")
    
    for _ in range(3):
        item = array_queue.dequeue()
        print(f"Dequeued: {item}")
    
    # Test CompactStack
    print("\n--- Compact Stack ---")
    compact_stack = CompactStack('ii')  # Two integers per item
    
    for i in range(5):
        compact_stack.push(i, i * 2)
    
    print(f"Pushed 5 items, size: {compact_stack.size()}")
    print(f"Memory usage: {compact_stack.memory_usage()}")
    
    for _ in range(2):
        item = compact_stack.pop()
        print(f"Popped: {item}")
    
    # Test PooledStack
    print("\n--- Pooled Stack ---")
    pooled_stack = PooledStack(pool_size=10)
    
    for i in range(7):
        pooled_stack.push(f"item_{i}")
    
    print(f"Pushed 7 items, size: {pooled_stack.size()}")
    print(f"Pool stats: {pooled_stack.pool_stats()}")
    
    for _ in range(3):
        item = pooled_stack.pop()
        print(f"Popped: {item}")
    
    print(f"After popping: {pooled_stack.pool_stats()}")


def benchmark_memory_efficiency():
    """Benchmark memory efficiency of different implementations"""
    print("\n=== Memory Efficiency Benchmark ===")
    
    import gc
    
    # Regular Python list stack
    class RegularStack:
        def __init__(self):
            self.items = []
        
        def push(self, item):
            self.items.append(item)
        
        def memory_size(self):
            return sys.getsizeof(self.items) + sum(sys.getsizeof(item) for item in self.items)
    
    implementations = [
        ("Regular Stack", RegularStack),
        ("Slotted Stack", SlottedStack),
    ]
    
    n_items = 10000
    
    for name, stack_class in implementations:
        gc.collect()  # Clean up before measurement
        
        if name == "Regular Stack":
            stack = stack_class()
        else:
            stack = stack_class(initial_capacity=n_items)
        
        # Add items
        for i in range(n_items):
            stack.push(i)
        
        if hasattr(stack, 'memory_usage'):
            memory_info = stack.memory_usage()
            memory_size = memory_info.get('items_memory', 0)
        else:
            memory_size = stack.memory_size()
        
        print(f"{name:15} | Memory: {memory_size:,} bytes | Per item: {memory_size/n_items:.2f} bytes")


def demonstrate_cache_optimization():
    """Demonstrate cache optimization benefits"""
    print("\n=== Cache Optimization Demonstration ===")
    
    import time
    
    # Compare regular queue vs cache-optimized queue
    from collections import deque
    
    regular_queue = deque()
    cache_queue = CacheOptimizedQueue(block_size=64)
    
    n_operations = 100000
    
    # Benchmark regular queue
    start_time = time.time()
    for i in range(n_operations):
        regular_queue.append(i)
    for _ in range(n_operations):
        regular_queue.popleft()
    regular_time = time.time() - start_time
    
    # Benchmark cache-optimized queue
    start_time = time.time()
    for i in range(n_operations):
        cache_queue.enqueue(i)
    for _ in range(n_operations):
        cache_queue.dequeue()
    cache_time = time.time() - start_time
    
    print(f"Regular queue time: {regular_time:.4f}s")
    print(f"Cache-optimized time: {cache_time:.4f}s")
    print(f"Cache optimization: {((regular_time - cache_time) / regular_time * 100):.1f}% improvement")
    print(f"Cache stats: {cache_queue.cache_stats()}")


def demonstrate_memory_mapping():
    """Demonstrate memory mapping for large datasets"""
    print("\n=== Memory Mapping Demonstration ===")
    
    # Create memory-mapped queue
    mmap_queue = MemoryMappedQueue(max_items=1000, item_size=16)
    
    try:
        # Add some data
        test_data = [
            b"Hello World",
            b"Memory Mapped",
            b"Queue Test",
            b"Large Dataset",
            b"Efficient"
        ]
        
        print("Adding data to memory-mapped queue:")
        for data in test_data:
            mmap_queue.enqueue(data)
            print(f"  Enqueued: {data}")
        
        print(f"\nQueue size: {mmap_queue.size()}")
        
        print("\nRetrieving data:")
        while not mmap_queue.is_empty():
            data = mmap_queue.dequeue()
            print(f"  Dequeued: {data}")
    
    finally:
        mmap_queue.close()


def analyze_space_time_tradeoffs():
    """Analyze space-time tradeoffs"""
    print("\n=== Space-Time Tradeoffs Analysis ===")
    
    tradeoffs = [
        ("Slotted Stack", "40-50% memory reduction", "Minimal time overhead", "Good for memory-constrained environments"),
        ("Array Queue", "75% memory reduction (numeric)", "Faster access for numeric data", "Best for numeric-only data"),
        ("Compact Stack", "Optimal packing", "Struct pack/unpack overhead", "Good for structured data"),
        ("Memory Mapped", "Handles > RAM datasets", "I/O overhead for disk access", "Best for very large datasets"),
        ("Pooled Stack", "Reduced GC pressure", "Pool management overhead", "Good for high-frequency operations"),
        ("Cache Optimized", "Better cache utilization", "Block management overhead", "Good for sequential access patterns"),
    ]
    
    print(f"{'Implementation':<20} | {'Space Benefit':<25} | {'Time Cost':<25} | {'Best Use Case'}")
    print("-" * 110)
    
    for impl, space, time, use_case in tradeoffs:
        print(f"{impl:<20} | {space:<25} | {time:<25} | {use_case}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: High-frequency trading system
    print("1. High-Frequency Trading Order Book:")
    
    class OrderBook:
        def __init__(self):
            # Use compact stack for price-quantity pairs
            self.buy_orders = CompactStack('dd')  # price, quantity
            self.sell_orders = CompactStack('dd')
        
        def add_buy_order(self, price: float, quantity: float):
            self.buy_orders.push(price, quantity)
            print(f"   Buy order: {quantity} @ ${price}")
        
        def add_sell_order(self, price: float, quantity: float):
            self.sell_orders.push(price, quantity)
            print(f"   Sell order: {quantity} @ ${price}")
        
        def get_memory_usage(self):
            buy_mem = self.buy_orders.memory_usage()
            sell_mem = self.sell_orders.memory_usage()
            return {
                'buy_orders_memory': buy_mem['data_size'],
                'sell_orders_memory': sell_mem['data_size'],
                'total_orders': buy_mem['count'] + sell_mem['count']
            }
    
    order_book = OrderBook()
    order_book.add_buy_order(100.50, 1000)
    order_book.add_buy_order(100.25, 500)
    order_book.add_sell_order(100.75, 750)
    
    print(f"   Memory usage: {order_book.get_memory_usage()}")
    
    # Application 2: IoT sensor data buffer
    print(f"\n2. IoT Sensor Data Buffer:")
    
    class SensorBuffer:
        def __init__(self):
            # Use array queue for numeric sensor readings
            self.temperature_buffer = ArrayBasedQueue('f')  # float values
            self.humidity_buffer = ArrayBasedQueue('f')
        
        def add_reading(self, temp: float, humidity: float):
            self.temperature_buffer.enqueue(temp)
            self.humidity_buffer.enqueue(humidity)
            print(f"   Sensor reading: {temp}°C, {humidity}% humidity")
        
        def get_latest_readings(self, count: int):
            readings = []
            temp_readings = []
            humidity_readings = []
            
            # Get recent readings (simplified)
            for _ in range(min(count, self.temperature_buffer.size())):
                temp = self.temperature_buffer.dequeue()
                humidity = self.humidity_buffer.dequeue()
                readings.append((temp, humidity))
                temp_readings.append(temp)
                humidity_readings.append(humidity)
            
            # Put them back (for demo)
            for temp, humidity in readings:
                self.temperature_buffer.enqueue(temp)
                self.humidity_buffer.enqueue(humidity)
            
            return readings
        
        def get_memory_efficiency(self):
            temp_mem = self.temperature_buffer.memory_usage()
            humidity_mem = self.humidity_buffer.memory_usage()
            return {
                'temperature_efficiency': temp_mem['used_items_size'] / temp_mem['array_memory'],
                'humidity_efficiency': humidity_mem['used_items_size'] / humidity_mem['array_memory'],
                'total_readings': temp_mem.get('size', 0)
            }
    
    sensor_buffer = SensorBuffer()
    sensor_buffer.add_reading(23.5, 65.2)
    sensor_buffer.add_reading(24.1, 63.8)
    sensor_buffer.add_reading(23.8, 64.5)
    
    latest = sensor_buffer.get_latest_readings(2)
    print(f"   Latest readings: {latest}")
    print(f"   Memory efficiency: {sensor_buffer.get_memory_efficiency()}")


if __name__ == "__main__":
    test_memory_optimized_structures()
    benchmark_memory_efficiency()
    demonstrate_cache_optimization()
    demonstrate_memory_mapping()
    analyze_space_time_tradeoffs()
    demonstrate_real_world_applications()

"""
Memory Optimized Structures demonstrates advanced memory optimization
techniques including slotted classes, array-based storage, struct packing,
memory mapping, object pooling, and cache optimization strategies.
"""
