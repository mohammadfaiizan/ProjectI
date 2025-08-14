"""
B-Trees and Variants - Multi-way Search Trees
=============================================

Topics: B-Trees, B+ Trees, B* Trees, database indexing, external storage
Companies: Google, Amazon, Microsoft, Oracle, MongoDB, PostgreSQL
Difficulty: Hard to Expert
Time Complexity: O(log n) for all operations
Space Complexity: O(n) with optimal disk utilization
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from collections import deque
import math

class BTreesAndVariants:
    
    def __init__(self):
        """Initialize with B-tree tracking"""
        self.operation_count = 0
        self.disk_access_count = 0
    
    # ==========================================
    # 1. B-TREE FUNDAMENTALS
    # ==========================================
    
    def explain_btree_concept(self) -> None:
        """
        Explain B-tree concepts and properties
        
        B-trees are balanced multi-way search trees optimized for disk storage
        """
        print("=== B-TREE FUNDAMENTALS ===")
        print("B-Tree: Balanced multi-way search tree optimized for external storage")
        print()
        print("KEY PROPERTIES:")
        print("• All leaves are at the same level")
        print("• Internal nodes can have multiple keys (not just one)")
        print("• Minimum degree t: internal nodes have t-1 to 2t-1 keys")
        print("• Root can have 1 to 2t-1 keys")
        print("• All nodes except root have at least t-1 keys")
        print("• Keys in each node are sorted in ascending order")
        print()
        print("STRUCTURAL PROPERTIES:")
        print("• Height = O(log_t n) where t is minimum degree")
        print("• Each node corresponds to one disk page")
        print("• Minimizes disk I/O operations")
        print("• Self-balancing through splitting and merging")
        print()
        print("B-TREE vs BINARY TREES:")
        print("• Binary trees: 2 children max, height O(log₂ n)")
        print("• B-trees: 2t children max, height O(log_t n)")
        print("• Fewer levels → fewer disk accesses")
        print("• Optimized for block-based storage systems")
        print()
        print("DEGREE SELECTION:")
        print("• Typically chosen based on disk page size")
        print("• Common values: t = 100-1000 for database systems")
        print("• Higher t → shorter tree, fewer disk accesses")
        print("• Limited by page size and key/pointer sizes")
        print()
        print("OPERATIONS:")
        print("• Search: Navigate from root to leaf")
        print("• Insert: Add key, split full nodes")
        print("• Delete: Remove key, merge/redistribute if needed")
        print("• All operations maintain B-tree properties")
        print()
        print("APPLICATIONS:")
        print("• Database indexing (primary and secondary indexes)")
        print("• File systems (NTFS, ext4, Btrfs)")
        print("• Key-value stores and NoSQL databases")
        print("• Any system with large datasets on disk")
    
    def demonstrate_btree_operations(self) -> None:
        """
        Demonstrate B-tree operations with detailed explanation
        """
        print("=== B-TREE OPERATIONS DEMONSTRATION ===")
        print("Demonstrating B-tree with minimum degree t=3")
        print("(Each node can have 2-5 keys, 3-6 children)")
        print()
        
        btree = BTree(t=3)
        
        # Test sequence that demonstrates splitting
        values = [10, 20, 5, 6, 12, 30, 7, 17, 15, 25, 35, 40]
        
        print(f"Inserting values: {values}")
        print()
        
        for value in values:
            print(f"Inserting {value}:")
            btree.insert(value)
            print(f"  Tree structure after insertion:")
            btree.display_tree()
            print(f"  Height: {btree.get_height()}, Disk accesses: {btree.disk_accesses}")
            print()
        
        # Test search operations
        print("Search operations:")
        search_values = [15, 25, 8, 50]
        for value in search_values:
            found, accesses = btree.search(value)
            print(f"  Search {value}: {'Found' if found else 'Not found'} (disk accesses: {accesses})")
        
        print()
        
        # Test deletion
        print("Deletion operations:")
        delete_values = [6, 12, 20]
        for value in delete_values:
            print(f"Deleting {value}:")
            btree.delete(value)
            print(f"  Tree structure after deletion:")
            btree.display_tree()
            print()


class BTreeNode:
    """
    B-Tree Node implementation
    
    Contains keys, children pointers, and metadata
    """
    
    def __init__(self, t: int, is_leaf: bool = False):
        self.t = t  # Minimum degree
        self.keys = []  # List of keys
        self.children = []  # List of child pointers
        self.is_leaf = is_leaf  # True if leaf node
        self.n = 0  # Current number of keys
    
    def __str__(self):
        return f"BTreeNode(keys={self.keys}, leaf={self.is_leaf})"


class BTree:
    """
    Complete B-Tree implementation
    
    Features:
    - Insertion with node splitting
    - Deletion with merging and redistribution
    - Search with disk access counting
    - Tree visualization and validation
    """
    
    def __init__(self, t: int):
        self.t = t  # Minimum degree
        self.root = BTreeNode(t, True)
        self.disk_accesses = 0
    
    def search(self, key: int) -> Tuple[bool, int]:
        """
        Search for key in B-tree
        
        Returns: (found, disk_accesses)
        Time: O(log_t n), Disk I/O: O(log_t n)
        """
        accesses = 0
        return self._search_recursive(self.root, key, accesses)
    
    def _search_recursive(self, node: BTreeNode, key: int, accesses: int) -> Tuple[bool, int]:
        """Recursive search with disk access counting"""
        accesses += 1  # Disk access to read this node
        
        # Find the index of first key >= key
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
        
        # Check if key is found
        if i < node.n and key == node.keys[i]:
            return True, accesses
        
        # If leaf node and key not found
        if node.is_leaf:
            return False, accesses
        
        # Recurse to appropriate child
        return self._search_recursive(node.children[i], key, accesses)
    
    def insert(self, key: int) -> None:
        """
        Insert key into B-tree
        
        Time: O(log_t n), Disk I/O: O(log_t n)
        """
        root = self.root
        
        # If root is full, create new root
        if root.n == 2 * self.t - 1:
            print(f"  Root is full, creating new root")
            new_root = BTreeNode(self.t, False)
            new_root.children.append(root)
            self.root = new_root
            self._split_child(new_root, 0)
            root = new_root
        
        self._insert_non_full(root, key)
        self.disk_accesses += 1
    
    def _insert_non_full(self, node: BTreeNode, key: int) -> None:
        """Insert into non-full node"""
        i = node.n - 1
        
        if node.is_leaf:
            # Insert into leaf node
            node.keys.append(0)  # Make space
            
            # Shift keys to make room for new key
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            
            node.keys[i + 1] = key
            node.n += 1
            print(f"    Inserted {key} into leaf node")
        else:
            # Insert into internal node
            # Find child to insert into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            # Check if child is full
            if node.children[i].n == 2 * self.t - 1:
                print(f"    Child at index {i} is full, splitting")
                self._split_child(node, i)
                
                # After split, determine which child to insert into
                if key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key)
    
    def _split_child(self, parent: BTreeNode, index: int) -> None:
        """Split full child of parent at given index"""
        full_child = parent.children[index]
        new_child = BTreeNode(self.t, full_child.is_leaf)
        
        # Copy second half of keys to new child
        mid_index = self.t - 1
        new_child.keys = full_child.keys[mid_index + 1:]
        new_child.n = len(new_child.keys)
        
        # Copy second half of children if not leaf
        if not full_child.is_leaf:
            new_child.children = full_child.children[mid_index + 1:]
        
        # Truncate original child
        mid_key = full_child.keys[mid_index]
        full_child.keys = full_child.keys[:mid_index]
        full_child.n = len(full_child.keys)
        
        if not full_child.is_leaf:
            full_child.children = full_child.children[:mid_index + 1]
        
        # Insert new child into parent
        parent.children.insert(index + 1, new_child)
        parent.keys.insert(index, mid_key)
        parent.n += 1
        
        print(f"    Split node: promoted key {mid_key} to parent")
    
    def delete(self, key: int) -> None:
        """Delete key from B-tree"""
        self._delete_recursive(self.root, key)
        
        # If root becomes empty, make first child the new root
        if self.root.n == 0 and not self.root.is_leaf:
            self.root = self.root.children[0]
    
    def _delete_recursive(self, node: BTreeNode, key: int) -> None:
        """Recursive deletion with rebalancing"""
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
        
        if i < node.n and key == node.keys[i]:
            # Key found in current node
            if node.is_leaf:
                # Case 1: Delete from leaf
                print(f"    Deleting {key} from leaf")
                node.keys.pop(i)
                node.n -= 1
            else:
                # Case 2: Delete from internal node
                self._delete_internal_node(node, key, i)
        else:
            # Key not in current node
            if node.is_leaf:
                print(f"    Key {key} not found")
                return
            
            # Ensure child has enough keys
            flag = (i == node.n)  # True if going to last child
            
            if node.children[i].n < self.t:
                self._fix_child(node, i)
            
            # After fixing, the key might have moved
            if flag and i > node.n:
                self._delete_recursive(node.children[i - 1], key)
            else:
                self._delete_recursive(node.children[i], key)
    
    def _delete_internal_node(self, node: BTreeNode, key: int, index: int) -> None:
        """Delete key from internal node"""
        left_child = node.children[index]
        right_child = node.children[index + 1]
        
        if left_child.n >= self.t:
            # Case 2a: Left child has enough keys
            pred = self._get_predecessor(left_child)
            node.keys[index] = pred
            self._delete_recursive(left_child, pred)
            print(f"    Replaced {key} with predecessor {pred}")
        elif right_child.n >= self.t:
            # Case 2b: Right child has enough keys
            succ = self._get_successor(right_child)
            node.keys[index] = succ
            self._delete_recursive(right_child, succ)
            print(f"    Replaced {key} with successor {succ}")
        else:
            # Case 2c: Both children have minimum keys, merge
            print(f"    Merging children to delete {key}")
            self._merge_children(node, index)
            self._delete_recursive(left_child, key)
    
    def _get_predecessor(self, node: BTreeNode) -> int:
        """Get predecessor (rightmost key in subtree)"""
        while not node.is_leaf:
            node = node.children[node.n]
        return node.keys[node.n - 1]
    
    def _get_successor(self, node: BTreeNode) -> int:
        """Get successor (leftmost key in subtree)"""
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0]
    
    def _fix_child(self, parent: BTreeNode, index: int) -> None:
        """Fix child that has too few keys"""
        child = parent.children[index]
        
        # Try to borrow from left sibling
        if index > 0 and parent.children[index - 1].n >= self.t:
            self._borrow_from_left(parent, index)
        # Try to borrow from right sibling
        elif index < parent.n and parent.children[index + 1].n >= self.t:
            self._borrow_from_right(parent, index)
        # Merge with sibling
        else:
            if index < parent.n:
                self._merge_children(parent, index)
            else:
                self._merge_children(parent, index - 1)
    
    def _borrow_from_left(self, parent: BTreeNode, index: int) -> None:
        """Borrow key from left sibling"""
        child = parent.children[index]
        left_sibling = parent.children[index - 1]
        
        # Move parent key to child
        child.keys.insert(0, parent.keys[index - 1])
        child.n += 1
        
        # Move child pointer if not leaf
        if not child.is_leaf:
            child.children.insert(0, left_sibling.children.pop())
        
        # Move left sibling's last key to parent
        parent.keys[index - 1] = left_sibling.keys.pop()
        left_sibling.n -= 1
        
        print(f"    Borrowed key from left sibling")
    
    def _borrow_from_right(self, parent: BTreeNode, index: int) -> None:
        """Borrow key from right sibling"""
        child = parent.children[index]
        right_sibling = parent.children[index + 1]
        
        # Move parent key to child
        child.keys.append(parent.keys[index])
        child.n += 1
        
        # Move child pointer if not leaf
        if not child.is_leaf:
            child.children.append(right_sibling.children.pop(0))
        
        # Move right sibling's first key to parent
        parent.keys[index] = right_sibling.keys.pop(0)
        right_sibling.n -= 1
        
        print(f"    Borrowed key from right sibling")
    
    def _merge_children(self, parent: BTreeNode, index: int) -> None:
        """Merge child with its right sibling"""
        left_child = parent.children[index]
        right_child = parent.children[index + 1]
        
        # Move parent key to left child
        left_child.keys.append(parent.keys[index])
        
        # Move all keys from right child to left child
        left_child.keys.extend(right_child.keys)
        left_child.n = len(left_child.keys)
        
        # Move children if not leaf
        if not left_child.is_leaf:
            left_child.children.extend(right_child.children)
        
        # Remove key and child from parent
        parent.keys.pop(index)
        parent.children.pop(index + 1)
        parent.n -= 1
        
        print(f"    Merged children")
    
    def get_height(self) -> int:
        """Get height of B-tree"""
        def height_recursive(node):
            if node.is_leaf:
                return 1
            return 1 + height_recursive(node.children[0])
        
        return height_recursive(self.root)
    
    def display_tree(self) -> None:
        """Display B-tree structure"""
        if self.root.n == 0:
            print("    (empty tree)")
            return
        
        print("    B-tree structure:")
        self._display_level_order()
    
    def _display_level_order(self) -> None:
        """Display tree level by level"""
        if not self.root:
            return
        
        queue = deque([(self.root, 0)])
        current_level = 0
        
        while queue:
            node, level = queue.popleft()
            
            if level > current_level:
                print()
                current_level = level
                print(f"      Level {level}: ", end="")
            
            print(f"[{', '.join(map(str, node.keys))}] ", end="")
            
            if not node.is_leaf:
                for child in node.children:
                    queue.append((child, level + 1))
        
        print()


# ==========================================
# 2. B+ TREE IMPLEMENTATION
# ==========================================

class BPlusTreeNode:
    """
    B+ Tree Node implementation
    
    Separates internal nodes (keys only) from leaf nodes (keys + data)
    """
    
    def __init__(self, t: int, is_leaf: bool = False):
        self.t = t
        self.keys = []
        self.children = []  # For internal nodes
        self.data = []  # For leaf nodes
        self.is_leaf = is_leaf
        self.next = None  # Pointer to next leaf (for range queries)
        self.n = 0


class BPlusTree:
    """
    B+ Tree implementation optimized for range queries
    
    Key differences from B-tree:
    - All data stored in leaves
    - Internal nodes contain only keys for navigation
    - Leaves are linked for efficient range queries
    - Better for database applications
    """
    
    def __init__(self, t: int):
        self.t = t
        self.root = BPlusTreeNode(t, True)
        self.leftmost_leaf = self.root  # For range queries
    
    def explain_bplus_advantages(self) -> None:
        """Explain B+ tree advantages over B-tree"""
        print("=== B+ TREE ADVANTAGES ===")
        print()
        print("KEY DIFFERENCES FROM B-TREE:")
        print("• All data stored in leaf nodes only")
        print("• Internal nodes contain only keys (no data)")
        print("• Leaf nodes are linked in sorted order")
        print("• More keys can fit in internal nodes")
        print("• Consistent performance for all queries")
        print()
        print("ADVANTAGES:")
        print("• Excellent for range queries (linked leaves)")
        print("• Higher fanout in internal nodes")
        print("• All data accesses go to same level (leaves)")
        print("• Sequential access is very efficient")
        print("• Better cache performance")
        print()
        print("APPLICATIONS:")
        print("• Database management systems (primary indexes)")
        print("• File systems with range query support")
        print("• Time-series databases")
        print("• Any system requiring efficient range scans")
    
    def range_query(self, start_key: int, end_key: int) -> List[int]:
        """
        Efficient range query using linked leaves
        
        Time: O(log_t n + k) where k is result size
        """
        print(f"Performing range query: {start_key} to {end_key}")
        
        # Find starting leaf
        leaf = self._find_leaf(start_key)
        
        result = []
        current_leaf = leaf
        
        while current_leaf:
            # Find starting position in current leaf
            start_idx = 0
            while (start_idx < current_leaf.n and 
                   current_leaf.keys[start_idx] < start_key):
                start_idx += 1
            
            # Collect keys in range from current leaf
            for i in range(start_idx, current_leaf.n):
                if current_leaf.keys[i] <= end_key:
                    result.append(current_leaf.keys[i])
                else:
                    print(f"  Found {len(result)} keys in range")
                    return result
            
            # Move to next leaf
            current_leaf = current_leaf.next
        
        print(f"  Found {len(result)} keys in range")
        return result
    
    def _find_leaf(self, key: int) -> BPlusTreeNode:
        """Find leaf node that should contain the key"""
        current = self.root
        
        while not current.is_leaf:
            i = 0
            while i < current.n and key >= current.keys[i]:
                i += 1
            current = current.children[i]
        
        return current


# ==========================================
# 3. B* TREE AND OTHER VARIANTS
# ==========================================

class BTreeVariants:
    """Explanation and comparison of B-tree variants"""
    
    def explain_btree_variants(self) -> None:
        """Explain different B-tree variants and their use cases"""
        print("=== B-TREE VARIANTS ===")
        print()
        
        print("1. B-TREE (Original):")
        print("   • Keys and data in all nodes")
        print("   • Good general-purpose balanced tree")
        print("   • Used in: File systems, general indexing")
        print()
        
        print("2. B+ TREE:")
        print("   • Data only in leaves, keys in internal nodes")
        print("   • Leaves linked for range queries")
        print("   • Higher fanout in internal nodes")
        print("   • Used in: Database systems, primary indexes")
        print()
        
        print("3. B* TREE:")
        print("   • Delayed splitting (split when 2/3 full)")
        print("   • Better space utilization (67% vs 50%)")
        print("   • More complex insertion algorithm")
        print("   • Used in: Space-critical applications")
        print()
        
        print("4. B-LINK TREE:")
        print("   • Additional right-link pointers")
        print("   • Supports concurrent operations")
        print("   • Lock-free searching possible")
        print("   • Used in: Concurrent database systems")
        print()
        
        print("5. FRACTAL TREE:")
        print("   • Buffered updates at each node")
        print("   • Optimized for write-heavy workloads")
        print("   • Batch updates for better performance")
        print("   • Used in: Write-intensive databases")
        print()
        
        print("SELECTION CRITERIA:")
        print("┌─────────────────┬─────────────┬─────────────┬──────────────┐")
        print("│ Workload        │ Best Choice │ Reason      │ Trade-off    │")
        print("├─────────────────┼─────────────┼─────────────┼──────────────┤")
        print("│ Range queries   │ B+          │ Linked leaf │ Complexity   │")
        print("│ Point queries   │ B           │ Direct acc  │ Range perf   │")
        print("│ Space critical  │ B*          │ 67% util    │ Complex ops  │")
        print("│ Concurrent      │ B-link      │ Lock-free   │ Extra space  │")
        print("│ Write-heavy     │ Fractal     │ Buffering   │ Read latency │")
        print("└─────────────────┴─────────────┴─────────────┴──────────────┘")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_btrees_and_variants():
    """Demonstrate all B-tree concepts and variants"""
    print("=== B-TREES AND VARIANTS DEMONSTRATION ===\n")
    
    btrees = BTreesAndVariants()
    
    # 1. B-tree fundamentals
    btrees.explain_btree_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. B-tree operations
    btrees.demonstrate_btree_operations()
    print("\n" + "="*60 + "\n")
    
    # 3. B+ tree concepts
    bplus = BPlusTree(t=3)
    bplus.explain_bplus_advantages()
    print("\n" + "="*60 + "\n")
    
    # 4. B-tree variants comparison
    variants = BTreeVariants()
    variants.explain_btree_variants()
    print("\n" + "="*60 + "\n")
    
    # 5. Performance analysis
    print("=== PERFORMANCE ANALYSIS ===")
    
    print("Theoretical Performance (n = 1,000,000 keys):")
    print()
    
    degrees = [2, 10, 100, 1000]  # Different minimum degrees
    
    print("Impact of minimum degree on tree height:")
    print(f"{'Degree (t)':<10} {'Max Height':<12} {'Disk I/O':<10} {'Space Util'}")
    print("-" * 50)
    
    n = 1_000_000
    for t in degrees:
        max_height = math.ceil(math.log(n/2 + 1, t))
        space_util = "50-100%" if t == 2 else f"{100 * (t-1)/(2*t-1):.0f}-100%"
        print(f"{t:<10} {max_height:<12} {max_height:<10} {space_util}")
    
    print()
    print("Key Insights:")
    print("• Higher degree → lower height → fewer disk accesses")
    print("• Degree limited by disk page size")
    print("• Typical database B+ trees: t = 100-1000")
    print("• Memory B-trees can use smaller degrees")


if __name__ == "__main__":
    demonstrate_btrees_and_variants()
    
    print("\n=== B-TREES MASTERY GUIDE ===")
    
    print("\n🎯 KEY CONCEPTS TO MASTER:")
    print("• Multi-way search tree principles")
    print("• Node splitting and merging algorithms")
    print("• Disk I/O optimization strategies")
    print("• Different variants and their trade-offs")
    print("• Database indexing applications")
    
    print("\n📊 COMPLEXITY GUARANTEES:")
    print("• Height: O(log_t n) where t is minimum degree")
    print("• Search: O(log_t n) disk accesses")
    print("• Insert: O(log_t n) disk accesses")
    print("• Delete: O(log_t n) disk accesses")
    print("• Range query (B+): O(log_t n + k) where k is result size")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Choose degree based on disk page size")
    print("• Use B+ trees for range-heavy workloads")
    print("• Consider B* trees for space-critical applications")
    print("• Implement proper caching for frequently accessed nodes")
    print("• Use appropriate buffering strategies")
    
    print("\n🔧 IMPLEMENTATION CONSIDERATIONS:")
    print("• Handle disk I/O efficiently")
    print("• Implement proper node serialization")
    print("• Consider concurrent access patterns")
    print("• Plan for crash recovery")
    print("• Optimize for specific workload patterns")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Database Management Systems (MySQL, PostgreSQL)")
    print("• File Systems (NTFS, ext4, Btrfs)")
    print("• Key-Value Stores (MongoDB, CouchDB)")
    print("• Search Engines (index structures)")
    print("• Time-Series Databases")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master binary search trees and balancing")
    print("2. Understand disk storage and I/O patterns")
    print("3. Learn B-tree operations and invariants")
    print("4. Study variants and their specific advantages")
    print("5. Practice with real database implementations")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Understand the disk I/O motivation")
    print("• Practice node splitting/merging by hand")
    print("• Study real database implementations")
    print("• Consider workload-specific optimizations")
    print("• Master the relationship between degree and performance")
