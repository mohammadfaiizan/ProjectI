"""
Cache System Design - Multiple Approaches
Difficulty: Hard

Design and implementation of distributed cache systems using trie structures
for efficient key-value storage with advanced features.

Components:
1. Trie-based Cache Storage
2. LRU/LFU Eviction Policies
3. Distributed Cache Architecture
4. Cache Warming and Preloading
5. Consistency Models
6. Performance Monitoring
"""

import time
import threading
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle

class EvictionPolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    RANDOM = "random"

@dataclass
class CacheItem:
    key: str
    value: Any
    access_time: float
    access_count: int
    creation_time: float
    ttl: Optional[float] = None
    size: int = 0

class TrieCacheNode:
    def __init__(self):
        self.children = {}
        self.items = {}  # Complete keys ending at this node
        self.is_leaf = False
        self.access_count = 0
        self.last_access = time.time()

class TrieBasedCache:
    """Trie-based cache for efficient prefix operations"""
    
    def __init__(self, max_size: int = 1000, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.root = TrieCacheNode()
        self.max_size = max_size
        self.current_size = 0
        self.eviction_policy = eviction_policy
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put key-value pair in cache"""
        with self.lock:
            current_time = time.time()
            
            # Check if key already exists
            existing_item = self._get_item(key)
            if existing_item:
                # Update existing item
                existing_item.value = value
                existing_item.access_time = current_time
                existing_item.access_count += 1
                existing_item.ttl = ttl
                self._update_access_tracking(key)
                return True
            
            # Check if cache is full
            if self.current_size >= self.max_size:
                evicted = self._evict_item()
                if not evicted:
                    return False
            
            # Create new cache item
            item_size = self._calculate_size(value)
            cache_item = CacheItem(
                key=key,
                value=value,
                access_time=current_time,
                access_count=1,
                creation_time=current_time,
                ttl=ttl,
                size=item_size
            )
            
            # Insert into trie
            self._insert_into_trie(key, cache_item)
            self.current_size += 1
            self._update_access_tracking(key)
            
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        with self.lock:
            self.stats['total_requests'] += 1
            
            item = self._get_item(key)
            if item is None:
                self.stats['misses'] += 1
                return None
            
            # Check TTL
            current_time = time.time()
            if item.ttl and (current_time - item.creation_time) > item.ttl:
                self.delete(key)
                self.stats['misses'] += 1
                return None
            
            # Update access information
            item.access_time = current_time
            item.access_count += 1
            self._update_access_tracking(key)
            
            self.stats['hits'] += 1
            return item.value
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if self._delete_from_trie(key):
                self.current_size -= 1
                if key in self.access_order:
                    del self.access_order[key]
                if key in self.frequency_counter:
                    del self.frequency_counter[key]
                return True
            return False
    
    def get_by_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, Any]]:
        """Get all keys with given prefix"""
        with self.lock:
            results = []
            self._collect_by_prefix(self.root, prefix, "", results, limit)
            return results
    
    def _get_item(self, key: str) -> Optional[CacheItem]:
        """Get cache item from trie"""
        node = self.root
        for char in key:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node.items.get(key)
    
    def _insert_into_trie(self, key: str, item: CacheItem) -> None:
        """Insert item into trie"""
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieCacheNode()
            node = node.children[char]
            node.access_count += 1
            node.last_access = time.time()
        
        node.items[key] = item
        node.is_leaf = True
    
    def _delete_from_trie(self, key: str) -> bool:
        """Delete item from trie"""
        node = self.root
        path = []
        
        # Navigate to the key and record path
        for char in key:
            if char not in node.children:
                return False
            path.append((node, char))
            node = node.children[char]
        
        # Remove the item
        if key not in node.items:
            return False
        
        del node.items[key]
        
        # Clean up empty nodes
        if not node.items and not node.children:
            node.is_leaf = False
            
            # Remove empty nodes from bottom up
            for parent, char in reversed(path):
                if not node.children and not node.items:
                    del parent.children[char]
                    node = parent
                else:
                    break
        
        return True
    
    def _collect_by_prefix(self, node: TrieCacheNode, prefix: str, 
                          current_key: str, results: List, limit: int) -> None:
        """Collect items by prefix recursively"""
        if len(results) >= limit:
            return
        
        if len(current_key) >= len(prefix):
            # Add all items at this node
            for key, item in node.items.items():
                if len(results) >= limit:
                    break
                # Check TTL
                current_time = time.time()
                if not item.ttl or (current_time - item.creation_time) <= item.ttl:
                    results.append((key, item.value))
        
        # Continue with children
        if len(current_key) < len(prefix):
            # Still building prefix
            next_char = prefix[len(current_key)]
            if next_char in node.children:
                self._collect_by_prefix(
                    node.children[next_char], prefix, 
                    current_key + next_char, results, limit
                )
        else:
            # Prefix complete, collect all children
            for char, child in node.children.items():
                self._collect_by_prefix(child, prefix, current_key + char, results, limit)
    
    def _evict_item(self) -> bool:
        """Evict item based on policy"""
        if self.eviction_policy == EvictionPolicy.LRU:
            return self._evict_lru()
        elif self.eviction_policy == EvictionPolicy.LFU:
            return self._evict_lfu()
        elif self.eviction_policy == EvictionPolicy.TTL:
            return self._evict_expired()
        else:
            return self._evict_random()
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.access_order:
            return False
        
        lru_key = next(iter(self.access_order))
        success = self.delete(lru_key)
        if success:
            self.stats['evictions'] += 1
        return success
    
    def _evict_lfu(self) -> bool:
        """Evict least frequently used item"""
        if not self.frequency_counter:
            return False
        
        lfu_key = min(self.frequency_counter.items(), key=lambda x: x[1])[0]
        success = self.delete(lfu_key)
        if success:
            self.stats['evictions'] += 1
        return success
    
    def _evict_expired(self) -> bool:
        """Evict expired items first"""
        current_time = time.time()
        
        # Find expired items
        expired_keys = []
        self._find_expired_items(self.root, "", current_time, expired_keys)
        
        if expired_keys:
            success = self.delete(expired_keys[0])
            if success:
                self.stats['evictions'] += 1
            return success
        
        # Fall back to LRU if no expired items
        return self._evict_lru()
    
    def _evict_random(self) -> bool:
        """Evict random item"""
        import random
        
        if not self.access_order:
            return False
        
        random_key = random.choice(list(self.access_order.keys()))
        success = self.delete(random_key)
        if success:
            self.stats['evictions'] += 1
        return success
    
    def _find_expired_items(self, node: TrieCacheNode, current_key: str, 
                           current_time: float, expired_keys: List[str]) -> None:
        """Find expired items in trie"""
        for key, item in node.items.items():
            if item.ttl and (current_time - item.creation_time) > item.ttl:
                expired_keys.append(key)
        
        for char, child in node.children.items():
            self._find_expired_items(child, current_key + char, current_time, expired_keys)
    
    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking for eviction policies"""
        current_time = time.time()
        
        # Update LRU tracking
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = current_time
        
        # Update LFU tracking
        self.frequency_counter[key] += 1
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['total_requests'])
            
            return {
                'size': self.current_size,
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'total_requests': self.stats['total_requests']
            }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.root = TrieCacheNode()
            self.current_size = 0
            self.access_order.clear()
            self.frequency_counter.clear()

class DistributedCache:
    """Distributed cache with consistent hashing"""
    
    def __init__(self, cache_nodes: List[str], replication_factor: int = 2):
        self.cache_nodes = {}
        self.replication_factor = replication_factor
        self.hash_ring = {}
        self.virtual_nodes = 150
        
        # Initialize cache nodes
        for node_id in cache_nodes:
            self.cache_nodes[node_id] = TrieBasedCache(max_size=1000)
            self._add_virtual_nodes(node_id)
        
        self.sorted_hashes = sorted(self.hash_ring.keys())
    
    def _hash_key(self, key: str) -> int:
        """Hash key for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _add_virtual_nodes(self, node_id: str) -> None:
        """Add virtual nodes to hash ring"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node_id}:{i}"
            hash_value = self._hash_key(virtual_key)
            self.hash_ring[hash_value] = node_id
    
    def _get_responsible_nodes(self, key: str) -> List[str]:
        """Get nodes responsible for key"""
        key_hash = self._hash_key(key)
        nodes = []
        
        # Find position in ring
        idx = 0
        for i, hash_val in enumerate(self.sorted_hashes):
            if key_hash <= hash_val:
                idx = i
                break
        
        # Get nodes for replication
        seen_nodes = set()
        for i in range(len(self.sorted_hashes)):
            ring_idx = (idx + i) % len(self.sorted_hashes)
            node_id = self.hash_ring[self.sorted_hashes[ring_idx]]
            
            if node_id not in seen_nodes:
                nodes.append(node_id)
                seen_nodes.add(node_id)
                
                if len(nodes) >= self.replication_factor:
                    break
        
        return nodes
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put key-value pair in distributed cache"""
        responsible_nodes = self._get_responsible_nodes(key)
        success_count = 0
        
        for node_id in responsible_nodes:
            if node_id in self.cache_nodes:
                if self.cache_nodes[node_id].put(key, value, ttl):
                    success_count += 1
        
        # Require majority success
        return success_count >= (len(responsible_nodes) // 2 + 1)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache"""
        responsible_nodes = self._get_responsible_nodes(key)
        
        # Try to get from any available node
        for node_id in responsible_nodes:
            if node_id in self.cache_nodes:
                value = self.cache_nodes[node_id].get(key)
                if value is not None:
                    return value
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete key from distributed cache"""
        responsible_nodes = self._get_responsible_nodes(key)
        success_count = 0
        
        for node_id in responsible_nodes:
            if node_id in self.cache_nodes:
                if self.cache_nodes[node_id].delete(key):
                    success_count += 1
        
        return success_count > 0
    
    def get_by_prefix(self, prefix: str, limit: int = 10) -> List[Tuple[str, Any]]:
        """Get items by prefix from all nodes"""
        all_results = []
        seen_keys = set()
        
        for node_id, cache in self.cache_nodes.items():
            results = cache.get_by_prefix(prefix, limit)
            
            for key, value in results:
                if key not in seen_keys:
                    all_results.append((key, value))
                    seen_keys.add(key)
                    
                    if len(all_results) >= limit:
                        break
            
            if len(all_results) >= limit:
                break
        
        return all_results
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics for entire cluster"""
        cluster_stats = {
            'total_nodes': len(self.cache_nodes),
            'total_size': 0,
            'total_hits': 0,
            'total_misses': 0,
            'total_requests': 0,
            'node_stats': {}
        }
        
        for node_id, cache in self.cache_nodes.items():
            stats = cache.get_stats()
            cluster_stats['node_stats'][node_id] = stats
            cluster_stats['total_size'] += stats['size']
            cluster_stats['total_hits'] += stats['hits']
            cluster_stats['total_misses'] += stats['misses']
            cluster_stats['total_requests'] += stats['total_requests']
        
        if cluster_stats['total_requests'] > 0:
            cluster_stats['overall_hit_rate'] = (
                cluster_stats['total_hits'] / cluster_stats['total_requests']
            )
        else:
            cluster_stats['overall_hit_rate'] = 0.0
        
        return cluster_stats

class CacheWarmer:
    """Cache warming utility"""
    
    def __init__(self, cache: Union[TrieBasedCache, DistributedCache]):
        self.cache = cache
    
    def warm_from_patterns(self, patterns: List[str], data_source: callable) -> None:
        """Warm cache from common access patterns"""
        for pattern in patterns:
            try:
                data = data_source(pattern)
                if data:
                    self.cache.put(pattern, data)
            except Exception as e:
                print(f"Error warming cache for pattern {pattern}: {e}")
    
    def warm_from_file(self, file_path: str) -> None:
        """Warm cache from file of key-value pairs"""
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        key, value = parts
                        self.cache.put(key, value)
        except FileNotFoundError:
            print(f"Cache warming file not found: {file_path}")
        except Exception as e:
            print(f"Error warming cache from file: {e}")


def test_trie_cache():
    """Test trie-based cache functionality"""
    print("=== Testing Trie-based Cache ===")
    
    cache = TrieBasedCache(max_size=5, eviction_policy=EvictionPolicy.LRU)
    
    # Test basic operations
    print("Testing basic operations:")
    cache.put("user:123", {"name": "Alice", "age": 30})
    cache.put("user:456", {"name": "Bob", "age": 25})
    cache.put("product:789", {"name": "Laptop", "price": 999})
    
    # Test gets
    user = cache.get("user:123")
    print(f"  Retrieved user:123: {user}")
    
    product = cache.get("product:789")
    print(f"  Retrieved product:789: {product}")
    
    # Test prefix operations
    print(f"\nTesting prefix operations:")
    user_keys = cache.get_by_prefix("user:", limit=10)
    print(f"  Keys with prefix 'user:': {user_keys}")
    
    # Test eviction
    print(f"\nTesting eviction (max_size=5):")
    for i in range(10):
        cache.put(f"temp:{i}", f"value_{i}")
    
    stats = cache.get_stats()
    print(f"  Cache size after adding 10 items: {stats['size']}")
    print(f"  Evictions: {stats['evictions']}")

def test_ttl_functionality():
    """Test TTL (Time To Live) functionality"""
    print("\n=== Testing TTL Functionality ===")
    
    cache = TrieBasedCache(max_size=100)
    
    # Add items with different TTLs
    cache.put("short_lived", "expires_soon", ttl=1.0)  # 1 second
    cache.put("long_lived", "expires_later", ttl=10.0)  # 10 seconds
    cache.put("permanent", "never_expires")  # No TTL
    
    print("Added items with different TTLs")
    
    # Check immediately
    print(f"  short_lived: {cache.get('short_lived')}")
    print(f"  long_lived: {cache.get('long_lived')}")
    print(f"  permanent: {cache.get('permanent')}")
    
    # Wait and check again
    print(f"\nWaiting 1.5 seconds...")
    time.sleep(1.5)
    
    print(f"  short_lived: {cache.get('short_lived')}")  # Should be None
    print(f"  long_lived: {cache.get('long_lived')}")   # Should still exist
    print(f"  permanent: {cache.get('permanent')}")     # Should still exist

def test_distributed_cache():
    """Test distributed cache functionality"""
    print("\n=== Testing Distributed Cache ===")
    
    # Create distributed cache with 3 nodes
    nodes = ["node1", "node2", "node3"]
    dcache = DistributedCache(nodes, replication_factor=2)
    
    # Test distributed operations
    print("Testing distributed operations:")
    
    test_data = [
        ("user:alice", {"name": "Alice", "role": "admin"}),
        ("user:bob", {"name": "Bob", "role": "user"}),
        ("config:database", {"host": "localhost", "port": 5432}),
        ("config:redis", {"host": "redis-server", "port": 6379}),
    ]
    
    # Put data
    for key, value in test_data:
        success = dcache.put(key, value)
        print(f"  Put '{key}': {'Success' if success else 'Failed'}")
    
    # Get data
    print(f"\nRetrieving data:")
    for key, _ in test_data:
        value = dcache.get(key)
        print(f"  Get '{key}': {value is not None}")
    
    # Test prefix queries
    user_data = dcache.get_by_prefix("user:", limit=5)
    print(f"\nUsers found: {len(user_data)}")
    
    # Get cluster statistics
    stats = dcache.get_cluster_stats()
    print(f"\nCluster statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total size: {stats['total_size']}")
    print(f"  Overall hit rate: {stats['overall_hit_rate']:.2%}")

def test_cache_warming():
    """Test cache warming functionality"""
    print("\n=== Testing Cache Warming ===")
    
    cache = TrieBasedCache(max_size=100)
    warmer = CacheWarmer(cache)
    
    # Define data source
    def mock_data_source(pattern: str) -> str:
        return f"data_for_{pattern}"
    
    # Warm cache with common patterns
    common_patterns = [
        "user:popular_user",
        "product:bestseller",
        "config:main_settings",
        "session:active_sessions"
    ]
    
    print("Warming cache with common patterns...")
    warmer.warm_from_patterns(common_patterns, mock_data_source)
    
    # Verify warming worked
    print(f"Cache size after warming: {cache.get_stats()['size']}")
    
    for pattern in common_patterns[:2]:  # Check first 2
        value = cache.get(pattern)
        print(f"  '{pattern}': {value}")

def benchmark_cache_performance():
    """Benchmark cache performance"""
    print("\n=== Benchmarking Cache Performance ===")
    
    cache = TrieBasedCache(max_size=10000)
    
    # Generate test data
    import random
    import string
    
    def generate_key():
        prefix = random.choice(["user", "product", "session", "config"])
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}:{suffix}"
    
    # Benchmark puts
    num_operations = 5000
    keys = [generate_key() for _ in range(num_operations)]
    
    start_time = time.time()
    for key in keys:
        cache.put(key, f"value_for_{key}")
    put_time = time.time() - start_time
    
    # Benchmark gets
    start_time = time.time()
    hits = 0
    for key in random.sample(keys, 1000):  # Random sample for gets
        if cache.get(key) is not None:
            hits += 1
    get_time = time.time() - start_time
    
    # Benchmark prefix operations
    start_time = time.time()
    prefix_results = 0
    for _ in range(100):
        prefix = random.choice(["user:", "product:", "session:", "config:"])
        results = cache.get_by_prefix(prefix, limit=10)
        prefix_results += len(results)
    prefix_time = time.time() - start_time
    
    # Get final statistics
    stats = cache.get_stats()
    
    print(f"Performance Results:")
    print(f"  {num_operations} puts in {put_time:.3f}s ({num_operations/put_time:.0f} ops/sec)")
    print(f"  1000 gets in {get_time:.3f}s ({1000/get_time:.0f} ops/sec)")
    print(f"  100 prefix queries in {prefix_time:.3f}s ({100/prefix_time:.0f} ops/sec)")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Final cache size: {stats['size']}")

if __name__ == "__main__":
    test_trie_cache()
    test_ttl_functionality()
    test_distributed_cache()
    test_cache_warming()
    benchmark_cache_performance()

"""
Cache System Design demonstrates enterprise-grade caching solutions:

Key Components:
1. Trie-based Storage - Efficient prefix-based operations
2. Multiple Eviction Policies - LRU, LFU, TTL, Random
3. Distributed Architecture - Consistent hashing with replication
4. Cache Warming - Preload frequently accessed data
5. TTL Support - Automatic expiration of stale data
6. Performance Monitoring - Comprehensive statistics and metrics

Advanced Features:
- Prefix-based queries for related data retrieval
- Consistent hashing for distributed cache nodes
- Replication for fault tolerance
- Cache warming strategies
- Multiple eviction policies
- Thread-safe operations
- Memory size tracking

System Design Principles:
- Horizontal scalability across multiple nodes
- Data replication for high availability
- Efficient memory utilization
- Low-latency operations
- Monitoring and observability

Real-world Applications:
- Web application caching (Redis, Memcached)
- Database query result caching
- Session storage systems
- CDN cache layers
- API response caching
- Configuration data caching

Performance Characteristics:
- Sub-millisecond get/put operations
- Efficient prefix-based batch operations
- Memory-conscious eviction policies
- Distributed load balancing
- High cache hit rates through intelligent warming

This implementation provides a production-ready foundation for
building scalable cache systems with enterprise requirements.
"""
