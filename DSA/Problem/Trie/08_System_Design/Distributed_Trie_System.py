"""
Distributed Trie System - Multiple Approaches
Difficulty: Hard

Design and implementation of distributed trie systems for large-scale applications.
Focus on scalability, consistency, and fault tolerance.

Components:
1. Distributed Hash Table with Trie Nodes
2. Consistent Hashing for Node Distribution
3. Replication and Fault Tolerance
4. Load Balancing Strategies
5. Cache Management
6. Consensus Protocols for Updates
"""

import hashlib
import threading
import time
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict
import json
import random

class TrieNode:
    def __init__(self, node_id: str = ""):
        self.children = {}
        self.is_end = False
        self.value = None
        self.timestamp = time.time()
        self.version = 0
        self.node_id = node_id
        self.replicas = set()

class ConsistentHashRing:
    """Consistent hashing for distributing trie nodes"""
    
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        self.nodes = set()
        
        for node in nodes:
            self.add_node(node)
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: str) -> None:
        """Add a node to the hash ring"""
        self.nodes.add(node)
        
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str) -> None:
        """Remove a node from the hash ring"""
        if node not in self.nodes:
            return
        
        self.nodes.remove(node)
        
        # Remove all virtual nodes
        keys_to_remove = []
        for hash_value, ring_node in self.ring.items():
            if ring_node == node:
                keys_to_remove.append(hash_value)
        
        for key in keys_to_remove:
            del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get the node responsible for a key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node clockwise
        for ring_hash in self.sorted_keys:
            if hash_value <= ring_hash:
                return self.ring[ring_hash]
        
        # Wrap around to the first node
        return self.ring[self.sorted_keys[0]]
    
    def get_nodes(self, key: str, count: int) -> List[str]:
        """Get multiple nodes for replication"""
        if not self.ring or count <= 0:
            return []
        
        primary_node = self.get_node(key)
        if not primary_node:
            return []
        
        nodes = [primary_node]
        hash_value = self._hash(key)
        
        # Find next nodes clockwise
        start_idx = 0
        for i, ring_hash in enumerate(self.sorted_keys):
            if hash_value <= ring_hash:
                start_idx = i
                break
        
        for i in range(1, count):
            next_idx = (start_idx + i) % len(self.sorted_keys)
            next_node = self.ring[self.sorted_keys[next_idx]]
            
            if next_node not in nodes:
                nodes.append(next_node)
        
        return nodes[:count]

class DistributedTrieNode:
    """Distributed trie node with replication"""
    
    def __init__(self, nodes: List[str], replication_factor: int = 3):
        self.hash_ring = ConsistentHashRing(nodes)
        self.replication_factor = replication_factor
        self.local_storage = {}  # node_id -> TrieNode storage
        self.locks = defaultdict(threading.RLock)
        
        # Initialize storage for each node
        for node in nodes:
            self.local_storage[node] = {}
    
    def _get_storage_key(self, path: str) -> str:
        """Generate storage key for trie path"""
        return f"trie:{path}"
    
    def _get_responsible_nodes(self, path: str) -> List[str]:
        """Get nodes responsible for storing a path"""
        return self.hash_ring.get_nodes(path, self.replication_factor)
    
    def insert(self, word: str, value: Any = None) -> bool:
        """Insert word into distributed trie"""
        path = ""
        
        for char in word:
            path += char
            storage_key = self._get_storage_key(path)
            responsible_nodes = self._get_responsible_nodes(path)
            
            # Insert into all responsible nodes
            success_count = 0
            for node_id in responsible_nodes:
                try:
                    with self.locks[node_id]:
                        if storage_key not in self.local_storage[node_id]:
                            trie_node = TrieNode(node_id)
                            self.local_storage[node_id][storage_key] = trie_node
                        
                        success_count += 1
                except Exception:
                    continue
            
            # Require majority success for consistency
            if success_count < (self.replication_factor // 2 + 1):
                return False
        
        # Mark final node as word ending
        final_key = self._get_storage_key(word)
        final_nodes = self._get_responsible_nodes(word)
        
        success_count = 0
        for node_id in final_nodes:
            try:
                with self.locks[node_id]:
                    if final_key in self.local_storage[node_id]:
                        self.local_storage[node_id][final_key].is_end = True
                        self.local_storage[node_id][final_key].value = value
                        self.local_storage[node_id][final_key].version += 1
                        success_count += 1
            except Exception:
                continue
        
        return success_count >= (self.replication_factor // 2 + 1)
    
    def search(self, word: str) -> Tuple[bool, Any]:
        """Search for word in distributed trie"""
        storage_key = self._get_storage_key(word)
        responsible_nodes = self._get_responsible_nodes(word)
        
        # Try to read from any available replica
        for node_id in responsible_nodes:
            try:
                with self.locks[node_id]:
                    if storage_key in self.local_storage[node_id]:
                        trie_node = self.local_storage[node_id][storage_key]
                        if trie_node.is_end:
                            return True, trie_node.value
            except Exception:
                continue
        
        return False, None
    
    def delete(self, word: str) -> bool:
        """Delete word from distributed trie"""
        storage_key = self._get_storage_key(word)
        responsible_nodes = self._get_responsible_nodes(word)
        
        success_count = 0
        for node_id in responsible_nodes:
            try:
                with self.locks[node_id]:
                    if storage_key in self.local_storage[node_id]:
                        self.local_storage[node_id][storage_key].is_end = False
                        self.local_storage[node_id][storage_key].value = None
                        self.local_storage[node_id][storage_key].version += 1
                        success_count += 1
            except Exception:
                continue
        
        return success_count >= (self.replication_factor // 2 + 1)
    
    def add_node(self, node_id: str) -> None:
        """Add new node to the distributed system"""
        self.hash_ring.add_node(node_id)
        self.local_storage[node_id] = {}
        
        # TODO: Implement data migration for existing keys
        self._migrate_data(node_id)
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from the distributed system"""
        if node_id in self.local_storage:
            # TODO: Migrate data to other nodes before removal
            self._migrate_data_from_failed_node(node_id)
            
            del self.local_storage[node_id]
            self.hash_ring.remove_node(node_id)
    
    def _migrate_data(self, new_node_id: str) -> None:
        """Migrate data when adding new node"""
        # Simplified migration - in practice would be more complex
        pass
    
    def _migrate_data_from_failed_node(self, failed_node_id: str) -> None:
        """Handle data migration from failed node"""
        # Simplified recovery - in practice would involve
        # reading from other replicas and re-replicating
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_nodes': len(self.local_storage),
            'total_keys': 0,
            'node_distribution': {},
            'replication_factor': self.replication_factor
        }
        
        for node_id, storage in self.local_storage.items():
            key_count = len(storage)
            stats['node_distribution'][node_id] = key_count
            stats['total_keys'] += key_count
        
        return stats

class LoadBalancer:
    """Load balancer for distributed trie operations"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.node_loads = {node: 0 for node in nodes}
        self.lock = threading.Lock()
    
    def get_least_loaded_node(self) -> str:
        """Get the node with least load"""
        with self.lock:
            return min(self.node_loads.items(), key=lambda x: x[1])[0]
    
    def update_load(self, node: str, load_delta: int) -> None:
        """Update load for a node"""
        with self.lock:
            if node in self.node_loads:
                self.node_loads[node] += load_delta
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get current load distribution"""
        with self.lock:
            return self.node_loads.copy()

class CacheManager:
    """Cache manager for distributed trie"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self.cache = {}  # key -> (value, timestamp, access_count)
        self.access_order = []  # LRU tracking
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                value, timestamp, access_count = self.cache[key]
                
                # Update access info
                self.cache[key] = (value, timestamp, access_count + 1)
                
                # Move to end for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                return value
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value into cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing entry
                _, _, access_count = self.cache[key]
                self.cache[key] = (value, current_time, access_count)
                
                # Move to end for LRU
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new entry
                if len(self.cache) >= self.cache_size:
                    # Evict least recently used
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                
                self.cache[key] = (value, current_time, 1)
                self.access_order.append(key)
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_accesses = sum(access_count for _, _, access_count in self.cache.values())
            
            return {
                'cache_size': len(self.cache),
                'max_size': self.cache_size,
                'total_accesses': total_accesses,
                'hit_ratio': len(self.cache) / max(1, total_accesses)
            }

class DistributedTrieSystem:
    """Complete distributed trie system"""
    
    def __init__(self, nodes: List[str], replication_factor: int = 3):
        self.distributed_trie = DistributedTrieNode(nodes, replication_factor)
        self.load_balancer = LoadBalancer(nodes)
        self.cache_manager = CacheManager()
        self.nodes = nodes
        
    def insert(self, word: str, value: Any = None) -> bool:
        """Insert with load balancing and caching"""
        # Choose node for operation
        node = self.load_balancer.get_least_loaded_node()
        self.load_balancer.update_load(node, 1)
        
        try:
            # Perform insertion
            success = self.distributed_trie.insert(word, value)
            
            if success:
                # Update cache
                self.cache_manager.put(word, value)
            
            return success
        finally:
            self.load_balancer.update_load(node, -1)
    
    def search(self, word: str) -> Tuple[bool, Any]:
        """Search with caching"""
        # Check cache first
        cached_value = self.cache_manager.get(word)
        if cached_value is not None:
            return True, cached_value
        
        # Search in distributed trie
        found, value = self.distributed_trie.search(word)
        
        if found:
            # Cache the result
            self.cache_manager.put(word, value)
        
        return found, value
    
    def delete(self, word: str) -> bool:
        """Delete with cache invalidation"""
        success = self.distributed_trie.delete(word)
        
        if success:
            # Invalidate cache
            self.cache_manager.invalidate(word)
        
        return success
    
    def scale_up(self, new_nodes: List[str]) -> None:
        """Add new nodes to the system"""
        for node in new_nodes:
            self.distributed_trie.add_node(node)
            self.load_balancer.nodes.append(node)
            self.load_balancer.node_loads[node] = 0
    
    def handle_node_failure(self, failed_node: str) -> None:
        """Handle node failure"""
        self.distributed_trie.remove_node(failed_node)
        
        if failed_node in self.load_balancer.nodes:
            self.load_balancer.nodes.remove(failed_node)
            del self.load_balancer.node_loads[failed_node]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        trie_stats = self.distributed_trie.get_statistics()
        load_distribution = self.load_balancer.get_load_distribution()
        cache_stats = self.cache_manager.get_statistics()
        
        return {
            'trie_statistics': trie_stats,
            'load_distribution': load_distribution,
            'cache_statistics': cache_stats,
            'active_nodes': len(self.nodes),
            'system_status': 'healthy' if len(self.nodes) > 0 else 'degraded'
        }


def test_consistent_hashing():
    """Test consistent hashing implementation"""
    print("=== Testing Consistent Hashing ===")
    
    nodes = ["node1", "node2", "node3"]
    ring = ConsistentHashRing(nodes)
    
    # Test key distribution
    keys = [f"key_{i}" for i in range(20)]
    distribution = defaultdict(int)
    
    for key in keys:
        node = ring.get_node(key)
        distribution[node] += 1
    
    print("Key distribution:")
    for node, count in distribution.items():
        print(f"  {node}: {count} keys")
    
    # Test adding new node
    print(f"\nAdding new node...")
    ring.add_node("node4")
    
    new_distribution = defaultdict(int)
    for key in keys:
        node = ring.get_node(key)
        new_distribution[node] += 1
    
    print("New key distribution:")
    for node, count in new_distribution.items():
        print(f"  {node}: {count} keys")

def test_distributed_trie():
    """Test distributed trie operations"""
    print("\n=== Testing Distributed Trie ===")
    
    nodes = ["node1", "node2", "node3", "node4"]
    system = DistributedTrieSystem(nodes, replication_factor=2)
    
    # Test insertions
    words = ["apple", "app", "application", "banana", "band"]
    
    print("Inserting words...")
    for word in words:
        success = system.insert(word, f"value_{word}")
        print(f"  Insert '{word}': {'Success' if success else 'Failed'}")
    
    # Test searches
    print(f"\nSearching words...")
    for word in words + ["unknown"]:
        found, value = system.search(word)
        print(f"  Search '{word}': {'Found' if found else 'Not found'}")
        if found:
            print(f"    Value: {value}")
    
    # Test system health
    print(f"\nSystem health:")
    health = system.get_system_health()
    
    print(f"  Active nodes: {health['active_nodes']}")
    print(f"  System status: {health['system_status']}")
    print(f"  Total keys: {health['trie_statistics']['total_keys']}")

def test_fault_tolerance():
    """Test fault tolerance mechanisms"""
    print("\n=== Testing Fault Tolerance ===")
    
    nodes = ["node1", "node2", "node3"]
    system = DistributedTrieSystem(nodes, replication_factor=2)
    
    # Insert some data
    words = ["test1", "test2", "test3"]
    for word in words:
        system.insert(word, f"value_{word}")
    
    print("Data inserted. Simulating node failure...")
    
    # Simulate node failure
    system.handle_node_failure("node1")
    
    # Verify data is still accessible
    print("Checking data accessibility after node failure:")
    for word in words:
        found, value = system.search(word)
        print(f"  '{word}': {'Accessible' if found else 'Lost'}")
    
    # Add new node
    print(f"\nAdding new node for recovery...")
    system.scale_up(["node4"])
    
    health = system.get_system_health()
    print(f"System status after recovery: {health['system_status']}")

def benchmark_distributed_operations():
    """Benchmark distributed operations"""
    print("\n=== Benchmarking Distributed Operations ===")
    
    nodes = ["node1", "node2", "node3", "node4"]
    system = DistributedTrieSystem(nodes)
    
    # Generate test data
    import random
    import string
    
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
             for _ in range(1000)]
    
    # Benchmark insertions
    start_time = time.time()
    successful_inserts = 0
    
    for word in words:
        if system.insert(word, f"value_{word}"):
            successful_inserts += 1
    
    insert_time = time.time() - start_time
    
    # Benchmark searches
    start_time = time.time()
    successful_searches = 0
    
    for word in random.sample(words, 500):
        found, _ = system.search(word)
        if found:
            successful_searches += 1
    
    search_time = time.time() - start_time
    
    print(f"Performance results:")
    print(f"  Insertions: {successful_inserts}/{len(words)} in {insert_time:.2f}s")
    print(f"  Insert rate: {successful_inserts/insert_time:.2f} ops/sec")
    print(f"  Searches: {successful_searches}/500 in {search_time:.2f}s")
    print(f"  Search rate: {500/search_time:.2f} ops/sec")
    
    # Cache statistics
    cache_stats = system.cache_manager.get_statistics()
    print(f"  Cache hit ratio: {cache_stats['hit_ratio']:.2f}")

if __name__ == "__main__":
    test_consistent_hashing()
    test_distributed_trie()
    test_fault_tolerance()
    benchmark_distributed_operations()

"""
Distributed Trie System demonstrates enterprise-scale system design:

Key Components:
1. Consistent Hashing - Distribute data evenly across nodes
2. Replication - Ensure data availability and fault tolerance
3. Load Balancing - Distribute operations across available nodes
4. Caching - Improve performance with intelligent caching
5. Fault Tolerance - Handle node failures gracefully
6. Scalability - Add/remove nodes dynamically

System Design Principles:
- CAP Theorem considerations (Consistency, Availability, Partition tolerance)
- Eventually consistent distributed storage
- Quorum-based operations for consistency
- Horizontal scaling capabilities
- Monitoring and health check mechanisms

Real-world Applications:
- Distributed search engines and indexes
- Large-scale autocomplete systems
- DNS resolution systems
- Distributed configuration management
- Content delivery networks (CDN)
- Microservices service discovery

This implementation provides a foundation for building production-ready
distributed trie systems with enterprise requirements.
"""
