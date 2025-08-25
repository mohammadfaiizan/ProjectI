"""
677. Map Sum Pairs - Multiple Approaches
Difficulty: Medium

Design a map that allows you to do the following:
- Insert a (key, value) pair. If the key already exists, update the value.
- Return the sum of all values whose keys have the given prefix.

LeetCode Problem: https://leetcode.com/problems/map-sum-pairs/

Example:
mapSum = MapSum()
mapSum.insert("apple", 3)
mapSum.sum("ap")           # return 3
mapSum.insert("app", 2)
mapSum.sum("ap")           # return 5
"""

from typing import Dict, Optional
from collections import defaultdict

class TrieNode:
    """Trie node with sum tracking"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.value: int = 0  # Value if this is end of a key
        self.prefix_sum: int = 0  # Sum of all values with this prefix

class MapSum1:
    """
    Approach 1: Trie with Prefix Sum Storage
    
    Store prefix sums at each node to enable O(prefix_length) queries.
    
    Time Complexity:
    - insert: O(key_length)
    - sum: O(prefix_length)
    
    Space Complexity: O(ALPHABET_SIZE * total_key_length)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.key_values: Dict[str, int] = {}  # Track current values for updates
    
    def insert(self, key: str, val: int) -> None:
        """Insert or update a key-value pair"""
        # Calculate the difference in value
        old_val = self.key_values.get(key, 0)
        delta = val - old_val
        self.key_values[key] = val
        
        # Update trie with the delta
        node = self.root
        node.prefix_sum += delta
        
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_sum += delta
        
        node.value = val
    
    def sum(self, prefix: str) -> int:
        """Get sum of all values with the given prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        return node.prefix_sum

class MapSum2:
    """
    Approach 2: Simple Dictionary with Linear Search
    
    Store key-value pairs in dictionary and compute sum on demand.
    
    Time Complexity:
    - insert: O(1)
    - sum: O(n * prefix_length) where n is number of keys
    
    Space Complexity: O(total_key_length)
    """
    
    def __init__(self):
        self.data: Dict[str, int] = {}
    
    def insert(self, key: str, val: int) -> None:
        """Insert or update key-value pair"""
        self.data[key] = val
    
    def sum(self, prefix: str) -> int:
        """Compute sum by checking all keys"""
        total = 0
        for key, value in self.data.items():
            if key.startswith(prefix):
                total += value
        return total

class MapSum3:
    """
    Approach 3: Trie with Lazy Propagation
    
    Use lazy propagation to avoid updating all nodes on each insert.
    
    Time Complexity:
    - insert: O(key_length)
    - sum: O(prefix_length + nodes_in_subtree)
    
    Space Complexity: O(ALPHABET_SIZE * total_key_length)
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.key_values: Dict[str, int] = {}
    
    def insert(self, key: str, val: int) -> None:
        """Insert key-value pair without updating prefix sums"""
        self.key_values[key] = val
        
        # Just ensure the path exists in trie
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.value = val
    
    def sum(self, prefix: str) -> int:
        """Compute sum by traversing subtree"""
        node = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        
        # DFS to sum all values in subtree
        return self._dfs_sum(node, "")
    
    def _dfs_sum(self, node: TrieNode, current_suffix: str) -> int:
        """DFS to compute sum of all values in subtree"""
        total = node.value
        
        for char, child in node.children.items():
            total += self._dfs_sum(child, current_suffix + char)
        
        return total

class MapSum4:
    """
    Approach 4: Prefix Dictionary Optimization
    
    Maintain a separate dictionary for each prefix length.
    
    Time Complexity:
    - insert: O(key_length²)
    - sum: O(1)
    
    Space Complexity: O(key_length² * number_of_keys)
    """
    
    def __init__(self):
        self.key_values: Dict[str, int] = {}
        self.prefix_sums: Dict[str, int] = defaultdict(int)
    
    def insert(self, key: str, val: int) -> None:
        """Insert key and update all prefix sums"""
        old_val = self.key_values.get(key, 0)
        delta = val - old_val
        self.key_values[key] = val
        
        # Update all prefixes
        for i in range(1, len(key) + 1):
            prefix = key[:i]
            self.prefix_sums[prefix] += delta
    
    def sum(self, prefix: str) -> int:
        """O(1) prefix sum lookup"""
        return self.prefix_sums.get(prefix, 0)

class MapSum5:
    """
    Approach 5: Compressed Trie (Radix Tree)
    
    Use compressed trie to save space for sparse key sets.
    
    Time Complexity:
    - insert: O(key_length)
    - sum: O(prefix_length + compressed_nodes)
    
    Space Complexity: Optimized for sparse tries
    """
    
    def __init__(self):
        self.root = {"path": "", "children": {}, "value": 0, "prefix_sum": 0}
        self.key_values: Dict[str, int] = {}
    
    def insert(self, key: str, val: int) -> None:
        """Insert into compressed trie"""
        old_val = self.key_values.get(key, 0)
        delta = val - old_val
        self.key_values[key] = val
        
        self._insert_compressed(self.root, key, val, delta, 0)
    
    def _insert_compressed(self, node: dict, key: str, val: int, delta: int, start_idx: int) -> None:
        """Insert into compressed trie with prefix sum updates"""
        node["prefix_sum"] += delta
        
        if start_idx >= len(key):
            node["value"] = val
            return
        
        path = node["path"]
        remaining_key = key[start_idx:]
        
        # Handle compressed path
        if path:
            if remaining_key.startswith(path):
                # Path matches, continue with children
                if len(remaining_key) == len(path):
                    node["value"] = val
                    return
                else:
                    # Continue with children
                    char = remaining_key[len(path)]
                    if char not in node["children"]:
                        node["children"][char] = {
                            "path": remaining_key[len(path) + 1:],
                            "children": {},
                            "value": val,
                            "prefix_sum": delta
                        }
                    else:
                        self._insert_compressed(node["children"][char], key, val, delta, 
                                              start_idx + len(path) + 1)
            else:
                # Need to split the compressed path
                self._split_compressed_path(node, key, val, delta, start_idx)
        else:
            # No compressed path, navigate to children
            char = remaining_key[0]
            if char not in node["children"]:
                node["children"][char] = {
                    "path": remaining_key[1:],
                    "children": {},
                    "value": val if len(remaining_key) == 1 else 0,
                    "prefix_sum": delta
                }
                if len(remaining_key) > 1:
                    self._insert_compressed(node["children"][char], key, val, delta, start_idx + 1)
            else:
                self._insert_compressed(node["children"][char], key, val, delta, start_idx + 1)
    
    def _split_compressed_path(self, node: dict, key: str, val: int, delta: int, start_idx: int) -> None:
        """Split a compressed path when inserting"""
        path = node["path"]
        remaining_key = key[start_idx:]
        
        # Find common prefix
        common_len = 0
        while (common_len < len(path) and 
               common_len < len(remaining_key) and
               path[common_len] == remaining_key[common_len]):
            common_len += 1
        
        if common_len > 0:
            # Split the path
            old_path = path[common_len:]
            node["path"] = path[:common_len]
            
            # Save old state
            old_children = node["children"]
            old_value = node["value"]
            old_prefix_sum = node["prefix_sum"]
            
            # Reset current node
            node["children"] = {}
            node["value"] = 0
            
            # Create child for old path continuation
            if old_path:
                first_char = old_path[0]
                node["children"][first_char] = {
                    "path": old_path[1:],
                    "children": old_children,
                    "value": old_value,
                    "prefix_sum": old_prefix_sum - delta
                }
            
            # Handle new path continuation
            if common_len < len(remaining_key):
                new_path = remaining_key[common_len:]
                first_char = new_path[0]
                node["children"][first_char] = {
                    "path": new_path[1:],
                    "children": {},
                    "value": val if len(new_path) == 1 else 0,
                    "prefix_sum": delta
                }
                if len(new_path) > 1:
                    self._insert_compressed(node["children"][first_char], key, val, delta, 
                                          start_idx + common_len + 1)
            else:
                node["value"] = val
    
    def sum(self, prefix: str) -> int:
        """Get prefix sum from compressed trie"""
        return self._sum_compressed(self.root, prefix, 0)
    
    def _sum_compressed(self, node: dict, prefix: str, start_idx: int) -> int:
        """Get sum from compressed trie"""
        if start_idx >= len(prefix):
            return node["prefix_sum"]
        
        path = node["path"]
        remaining_prefix = prefix[start_idx:]
        
        # Handle compressed path
        if path:
            if len(remaining_prefix) <= len(path):
                # Prefix ends within compressed path
                if path.startswith(remaining_prefix):
                    return node["prefix_sum"]
                else:
                    return 0
            elif remaining_prefix.startswith(path):
                # Continue searching
                char = remaining_prefix[len(path)]
                if char in node["children"]:
                    return self._sum_compressed(node["children"][char], prefix, 
                                              start_idx + len(path) + 1)
                else:
                    return 0
            else:
                return 0
        else:
            # Navigate to children
            char = remaining_prefix[0]
            if char in node["children"]:
                return self._sum_compressed(node["children"][char], prefix, start_idx + 1)
            else:
                return 0

class MapSum6:
    """
    Approach 6: Segment Tree based Implementation
    
    Use segment tree for range sum queries on sorted keys.
    
    Time Complexity:
    - insert: O(log n)
    - sum: O(log n + k) where k is number of matching keys
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.keys: list = []  # Sorted list of keys
        self.values: Dict[str, int] = {}
        self.needs_rebuild = True
        self.segment_tree: list = []
    
    def insert(self, key: str, val: int) -> None:
        """Insert key-value pair"""
        self.values[key] = val
        self.needs_rebuild = True
    
    def _rebuild_segment_tree(self) -> None:
        """Rebuild segment tree when needed"""
        self.keys = sorted(self.values.keys())
        n = len(self.keys)
        
        if n == 0:
            self.segment_tree = []
            return
        
        # Build segment tree
        self.segment_tree = [0] * (4 * n)
        self._build_tree(0, 0, n - 1)
        self.needs_rebuild = False
    
    def _build_tree(self, node: int, start: int, end: int) -> None:
        """Build segment tree recursively"""
        if start == end:
            self.segment_tree[node] = self.values[self.keys[start]]
        else:
            mid = (start + end) // 2
            self._build_tree(2 * node + 1, start, mid)
            self._build_tree(2 * node + 2, mid + 1, end)
            self.segment_tree[node] = (self.segment_tree[2 * node + 1] + 
                                     self.segment_tree[2 * node + 2])
    
    def sum(self, prefix: str) -> int:
        """Get sum using segment tree"""
        if self.needs_rebuild:
            self._rebuild_segment_tree()
        
        if not self.keys:
            return 0
        
        # Find range of keys with the prefix
        start_idx = self._find_first_with_prefix(prefix)
        if start_idx == -1:
            return 0
        
        end_idx = self._find_last_with_prefix(prefix)
        
        return self._query_tree(0, 0, len(self.keys) - 1, start_idx, end_idx)
    
    def _find_first_with_prefix(self, prefix: str) -> int:
        """Find first key with given prefix"""
        left, right = 0, len(self.keys)
        while left < right:
            mid = (left + right) // 2
            if self.keys[mid] >= prefix:
                right = mid
            else:
                left = mid + 1
        
        if left < len(self.keys) and self.keys[left].startswith(prefix):
            return left
        return -1
    
    def _find_last_with_prefix(self, prefix: str) -> int:
        """Find last key with given prefix"""
        first = self._find_first_with_prefix(prefix)
        if first == -1:
            return -1
        
        # Find the end of the prefix range
        for i in range(first, len(self.keys)):
            if not self.keys[i].startswith(prefix):
                return i - 1
        
        return len(self.keys) - 1
    
    def _query_tree(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Query segment tree for range sum"""
        if r < start or end < l:
            return 0
        
        if l <= start and end <= r:
            return self.segment_tree[node]
        
        mid = (start + end) // 2
        return (self._query_tree(2 * node + 1, start, mid, l, r) +
                self._query_tree(2 * node + 2, mid + 1, end, l, r))


def test_basic_operations():
    """Test basic MapSum operations"""
    print("=== Testing Basic Operations ===")
    
    implementations = [
        ("Trie Prefix Sum", MapSum1),
        ("Linear Search", MapSum2),
        ("Lazy Propagation", MapSum3),
        ("Prefix Dictionary", MapSum4),
        ("Compressed Trie", MapSum5),
        ("Segment Tree", MapSum6),
    ]
    
    # Test operations from LeetCode example
    test_operations = [
        ("insert", "apple", 3),
        ("sum", "ap", None),      # Expected: 3
        ("insert", "app", 2),
        ("sum", "ap", None),      # Expected: 5
        ("insert", "apple", 5),   # Update existing key
        ("sum", "ap", None),      # Expected: 7
        ("sum", "app", None),     # Expected: 2
        ("sum", "apple", None),   # Expected: 5
    ]
    
    for name, MapSumClass in implementations:
        print(f"\n--- Testing {name} ---")
        
        mapSum = MapSumClass()
        
        for operation in test_operations:
            if operation[0] == "insert":
                mapSum.insert(operation[1], operation[2])
                print(f"  insert('{operation[1]}', {operation[2]})")
            elif operation[0] == "sum":
                result = mapSum.sum(operation[1])
                print(f"  sum('{operation[1]}') -> {result}")


def test_complex_scenarios():
    """Test complex scenarios"""
    print("\n=== Testing Complex Scenarios ===")
    
    mapSum = MapSum1()
    
    # Complex key-value pairs
    operations = [
        ("insert", "apple", 3),
        ("insert", "app", 2),
        ("insert", "application", 5),
        ("insert", "apply", 4),
        ("insert", "banana", 6),
        ("insert", "band", 3),
        ("insert", "bandana", 2),
    ]
    
    for op, key, val in operations:
        mapSum.insert(key, val)
        print(f"insert('{key}', {val})")
    
    # Test various prefixes
    prefixes = ["a", "ap", "app", "appl", "b", "ban", "band", "xyz"]
    
    print(f"\nPrefix sum queries:")
    for prefix in prefixes:
        result = mapSum.sum(prefix)
        print(f"  sum('{prefix}') -> {result}")


def test_update_operations():
    """Test key update operations"""
    print("\n=== Testing Update Operations ===")
    
    mapSum = MapSum1()
    
    # Initial inserts
    mapSum.insert("test", 10)
    mapSum.insert("testing", 20)
    mapSum.insert("tester", 15)
    
    print("Initial state:")
    print(f"  sum('test') -> {mapSum.sum('test')}")  # Should be 45
    
    # Update existing key
    print(f"\nUpdating 'test' from 10 to 5:")
    mapSum.insert("test", 5)
    print(f"  sum('test') -> {mapSum.sum('test')}")  # Should be 40
    
    # Update to 0 (effectively removing contribution)
    print(f"\nUpdating 'testing' to 0:")
    mapSum.insert("testing", 0)
    print(f"  sum('test') -> {mapSum.sum('test')}")  # Should be 20


def benchmark_implementations():
    """Benchmark different implementations"""
    print("\n=== Benchmarking Implementations ===")
    
    import time
    import random
    import string
    
    # Generate test data
    def generate_keys(n: int, avg_length: int) -> list:
        keys = []
        for _ in range(n):
            length = max(1, avg_length + random.randint(-2, 2))
            key = ''.join(random.choices(string.ascii_lowercase, k=length))
            keys.append(key)
        return keys
    
    test_keys = generate_keys(1000, 6)
    test_prefixes = [key[:random.randint(1, len(key))] for key in test_keys[:100]]
    
    implementations = [
        ("Trie Prefix", MapSum1),
        ("Linear", MapSum2),
        ("Lazy Trie", MapSum3),
        ("Prefix Dict", MapSum4),
    ]
    
    print(f"Testing with {len(test_keys)} keys, {len(test_prefixes)} queries")
    
    for name, MapSumClass in implementations:
        start_time = time.time()
        
        mapSum = MapSumClass()
        
        # Insert phase
        for i, key in enumerate(test_keys):
            mapSum.insert(key, i + 1)
        
        # Query phase
        total_sum = 0
        for prefix in test_prefixes:
            total_sum += mapSum.sum(prefix)
        
        end_time = time.time()
        
        print(f"{name:12}: {(end_time - start_time)*1000:.2f}ms (total_sum: {total_sum})")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Sales Analytics by Region
    print("1. Sales Analytics by Region:")
    sales = MapSum1()
    
    # Regional sales data
    sales_data = [
        ("north.usa.california", 1000),
        ("north.usa.nevada", 800),
        ("north.canada.ontario", 600),
        ("south.usa.texas", 1200),
        ("south.usa.florida", 900),
        ("south.mexico.df", 400),
    ]
    
    for region, amount in sales_data:
        sales.insert(region, amount)
    
    print("   Regional sales:")
    for region, amount in sales_data:
        print(f"     {region}: ${amount}")
    
    # Aggregate queries
    queries = ["north", "south", "north.usa", "south.usa"]
    print(f"   Aggregate sales:")
    for query in queries:
        total = sales.sum(query)
        print(f"     {query}: ${total}")
    
    # Application 2: Website Traffic by URL Path
    print("\n2. Website Traffic Analytics:")
    traffic = MapSum1()
    
    # URL path traffic data
    traffic_data = [
        ("/api/users", 500),
        ("/api/users/profile", 200),
        ("/api/posts", 300),
        ("/api/posts/comments", 150),
        ("/admin/dashboard", 50),
        ("/admin/users", 30),
    ]
    
    for path, hits in traffic_data:
        traffic.insert(path, hits)
    
    print("   URL path traffic:")
    for path, hits in traffic_data:
        print(f"     {path}: {hits} hits")
    
    # Path prefix analytics
    path_queries = ["/api", "/admin", "/api/users", "/api/posts"]
    print(f"   Path prefix analytics:")
    for query in path_queries:
        total = traffic.sum(query)
        print(f"     {query}: {total} total hits")
    
    # Application 3: Product Inventory by Category
    print("\n3. Product Inventory by Category:")
    inventory = MapSum1()
    
    # Product category inventory
    products = [
        ("electronics.phones.iphone", 100),
        ("electronics.phones.samsung", 80),
        ("electronics.laptops.macbook", 50),
        ("electronics.laptops.dell", 70),
        ("clothing.shirts.cotton", 200),
        ("clothing.pants.jeans", 150),
    ]
    
    for product, quantity in products:
        inventory.insert(product, quantity)
    
    print("   Product inventory:")
    for product, quantity in products:
        print(f"     {product}: {quantity} units")
    
    # Category queries
    category_queries = ["electronics", "clothing", "electronics.phones", "electronics.laptops"]
    print(f"   Category inventory:")
    for query in category_queries:
        total = inventory.sum(query)
        print(f"     {query}: {total} total units")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    mapSum = MapSum1()
    
    # Edge cases
    edge_cases = [
        ("Empty string key", "", 10),
        ("Single char key", "a", 5),
        ("Long key", "a" * 100, 15),
        ("Special chars", "key-with-dash", 20),
        ("Zero value", "zero", 0),
        ("Negative value", "negative", -5),
    ]
    
    print("Testing edge cases:")
    for description, key, value in edge_cases:
        try:
            mapSum.insert(key, value)
            result = mapSum.sum(key)
            print(f"  {description}: insert('{key[:10]}{'...' if len(key) > 10 else ''}', {value}) -> sum = {result}")
        except Exception as e:
            print(f"  {description}: Error - {e}")
    
    # Test empty prefix
    try:
        total = mapSum.sum("")
        print(f"  Empty prefix sum: {total}")
    except Exception as e:
        print(f"  Empty prefix sum: Error - {e}")


if __name__ == "__main__":
    test_basic_operations()
    test_complex_scenarios()
    test_update_operations()
    benchmark_implementations()
    demonstrate_real_world_applications()
    test_edge_cases()

"""
677. Map Sum Pairs demonstrates multiple approaches for prefix-based sum queries:

1. Trie with Prefix Sum - Efficient prefix sums stored at each node
2. Linear Search - Simple approach with O(n) queries
3. Lazy Propagation - Deferred computation for better insert performance
4. Prefix Dictionary - Pre-computed prefix sums for O(1) queries
5. Compressed Trie - Space-optimized trie for sparse key sets
6. Segment Tree - Tree-based range sum queries on sorted keys

Each approach shows different trade-offs between insert and query performance,
suitable for various use cases from real-time analytics to batch processing.
"""
