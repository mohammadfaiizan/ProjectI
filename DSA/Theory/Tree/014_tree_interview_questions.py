"""
Tree Interview Questions - Common Tree Problems in Technical Interviews
This module implements solutions to frequently asked tree problems with multiple approaches.
"""

from typing import List, Optional, Dict, Set, Tuple, Deque
from collections import deque, defaultdict
import json

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeInterviewProblems:
    """Solutions to common tree interview problems"""
    
    def __init__(self):
        """Initialize tree interview problem solver"""
        pass
    
    # ==================== DISTANCE BETWEEN TWO NODES ====================
    
    def distance_between_nodes(self, root: Optional[TreeNode], p: int, q: int) -> int:
        """
        Find distance between two nodes in binary tree
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
            p, q: Values of two nodes
        
        Returns:
            int: Distance between nodes p and q
        """
        def find_lca(node):
            if not node or node.val == p or node.val == q:
                return node
            
            left = find_lca(node.left)
            right = find_lca(node.right)
            
            if left and right:
                return node
            return left or right
        
        def find_distance(node, target, distance):
            if not node:
                return -1
            
            if node.val == target:
                return distance
            
            left_dist = find_distance(node.left, target, distance + 1)
            if left_dist != -1:
                return left_dist
            
            return find_distance(node.right, target, distance + 1)
        
        lca = find_lca(root)
        if not lca:
            return -1
        
        dist_p = find_distance(lca, p, 0)
        dist_q = find_distance(lca, q, 0)
        
        if dist_p == -1 or dist_q == -1:
            return -1
        
        return dist_p + dist_q
    
    def distance_with_path(self, root: Optional[TreeNode], p: int, q: int) -> Tuple[int, List[int]]:
        """
        Find distance and actual path between two nodes
        
        Args:
            root: Root of binary tree
            p, q: Values of two nodes
        
        Returns:
            Tuple of (distance, path)
        """
        def find_path(node, target, path):
            if not node:
                return False
            
            path.append(node.val)
            
            if node.val == target:
                return True
            
            if (find_path(node.left, target, path) or 
                find_path(node.right, target, path)):
                return True
            
            path.pop()
            return False
        
        path_p = []
        path_q = []
        
        find_path(root, p, path_p)
        find_path(root, q, path_q)
        
        if not path_p or not path_q:
            return -1, []
        
        # Find LCA by comparing paths
        i = 0
        while (i < len(path_p) and i < len(path_q) and 
               path_p[i] == path_q[i]):
            i += 1
        
        # Build complete path
        complete_path = path_p[:i-1:-1] + path_p[i-1:] + path_q[i:]
        
        return len(complete_path) - 1, complete_path
    
    def distance_using_parent_pointers(self, node_p: TreeNode, node_q: TreeNode) -> int:
        """
        Find distance when nodes have parent pointers
        
        Args:
            node_p, node_q: Tree nodes with parent pointers
        
        Returns:
            int: Distance between nodes
        """
        # Find depth of both nodes
        def get_depth(node):
            depth = 0
            while hasattr(node, 'parent') and node.parent:
                depth += 1
                node = node.parent
            return depth
        
        depth_p = get_depth(node_p)
        depth_q = get_depth(node_q)
        
        # Make nodes at same level
        while depth_p > depth_q:
            node_p = node_p.parent
            depth_p -= 1
        
        while depth_q > depth_p:
            node_q = node_q.parent
            depth_q -= 1
        
        # Find LCA
        distance = 0
        while node_p != node_q:
            node_p = node_p.parent
            node_q = node_q.parent
            distance += 2
        
        return distance
    
    # ==================== COUNT NODES IN COMPLETE BINARY TREE ====================
    
    def count_nodes_complete_tree(self, root: Optional[TreeNode]) -> int:
        """
        Count nodes in complete binary tree (LeetCode 222)
        
        Time Complexity: O(log^2 n)
        Space Complexity: O(log n)
        
        Args:
            root: Root of complete binary tree
        
        Returns:
            int: Number of nodes
        """
        if not root:
            return 0
        
        def get_height(node, go_left=True):
            height = 0
            while node:
                height += 1
                node = node.left if go_left else node.right
            return height
        
        left_height = get_height(root, True)
        right_height = get_height(root, False)
        
        if left_height == right_height:
            # Perfect binary tree
            return (1 << left_height) - 1
        
        return 1 + self.count_nodes_complete_tree(root.left) + self.count_nodes_complete_tree(root.right)
    
    def count_nodes_binary_search(self, root: Optional[TreeNode]) -> int:
        """
        Count nodes using binary search approach
        
        Time Complexity: O(log^2 n)
        
        Args:
            root: Root of complete binary tree
        
        Returns:
            int: Number of nodes
        """
        if not root:
            return 0
        
        def get_depth(node):
            depth = 0
            while node:
                depth += 1
                node = node.left
            return depth
        
        def node_exists(index, depth, node):
            left, right = 0, (1 << (depth - 1)) - 1
            
            for _ in range(depth - 1):
                mid = (left + right) // 2
                if index <= mid:
                    node = node.left
                    right = mid
                else:
                    node = node.right
                    left = mid + 1
            
            return node is not None
        
        depth = get_depth(root)
        if depth == 0:
            return 0
        
        # Binary search on last level
        left, right = 1, (1 << (depth - 1))
        
        while left <= right:
            mid = (left + right) // 2
            if node_exists(mid, depth, root):
                left = mid + 1
            else:
                right = mid - 1
        
        return (1 << (depth - 1)) - 1 + right
    
    def count_nodes_iterative(self, root: Optional[TreeNode]) -> int:
        """
        Count nodes iteratively
        
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        
        Args:
            root: Root of tree
        
        Returns:
            int: Number of nodes
        """
        if not root:
            return 0
        
        count = 0
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            count += 1
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return count
    
    # ==================== RECOVER BST (LEETCODE 99) ====================
    
    def recover_bst(self, root: Optional[TreeNode]) -> None:
        """
        Recover BST where exactly two nodes were swapped
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of BST to recover
        """
        self.first_node = None
        self.second_node = None
        self.prev_node = None
        
        def inorder(node):
            if not node:
                return
            
            inorder(node.left)
            
            # Check if current node violates BST property
            if self.prev_node and self.prev_node.val > node.val:
                if not self.first_node:
                    # First violation
                    self.first_node = self.prev_node
                    self.second_node = node
                else:
                    # Second violation
                    self.second_node = node
            
            self.prev_node = node
            inorder(node.right)
        
        inorder(root)
        
        # Swap the values of the two nodes
        if self.first_node and self.second_node:
            self.first_node.val, self.second_node.val = self.second_node.val, self.first_node.val
    
    def recover_bst_morris(self, root: Optional[TreeNode]) -> None:
        """
        Recover BST using Morris traversal (O(1) space)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            root: Root of BST to recover
        """
        first_node = None
        second_node = None
        prev_node = None
        
        current = root
        
        while current:
            if not current.left:
                # Process current node
                if prev_node and prev_node.val > current.val:
                    if not first_node:
                        first_node = prev_node
                        second_node = current
                    else:
                        second_node = current
                
                prev_node = current
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make threading
                    predecessor.right = current
                    current = current.left
                else:
                    # Remove threading and process current
                    predecessor.right = None
                    
                    if prev_node and prev_node.val > current.val:
                        if not first_node:
                            first_node = prev_node
                            second_node = current
                        else:
                            second_node = current
                    
                    prev_node = current
                    current = current.right
        
        # Swap the values
        if first_node and second_node:
            first_node.val, second_node.val = second_node.val, first_node.val
    
    def find_swapped_nodes(self, root: Optional[TreeNode]) -> Tuple[Optional[TreeNode], Optional[TreeNode]]:
        """
        Find the two swapped nodes without recovering
        
        Args:
            root: Root of BST
        
        Returns:
            Tuple of (first_swapped_node, second_swapped_node)
        """
        inorder_nodes = []
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            inorder_nodes.append(node)
            inorder(node.right)
        
        inorder(root)
        
        first = None
        second = None
        
        # Find violations
        for i in range(len(inorder_nodes) - 1):
            if inorder_nodes[i].val > inorder_nodes[i + 1].val:
                if first is None:
                    first = inorder_nodes[i]
                    second = inorder_nodes[i + 1]
                else:
                    second = inorder_nodes[i + 1]
                    break
        
        return first, second
    
    # ==================== SERIALIZE AND DESERIALIZE BINARY TREE ====================
    
    def serialize_preorder(self, root: Optional[TreeNode]) -> str:
        """
        Serialize binary tree using preorder traversal (LeetCode 297)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: Serialized string
        """
        def preorder(node):
            if not node:
                vals.append("null")
                return
            
            vals.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
        
        vals = []
        preorder(root)
        return ",".join(vals)
    
    def deserialize_preorder(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize binary tree from preorder string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        def build():
            val = next(vals)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        
        vals = iter(data.split(","))
        return build()
    
    def serialize_level_order(self, root: Optional[TreeNode]) -> str:
        """
        Serialize using level order traversal
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: Serialized string
        """
        if not root:
            return ""
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")
        
        # Remove trailing nulls
        while result and result[-1] == "null":
            result.pop()
        
        return ",".join(result)
    
    def deserialize_level_order(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize from level order string
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        if not data:
            return None
        
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1
        
        while queue and i < len(vals):
            node = queue.popleft()
            
            # Left child
            if i < len(vals) and vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            
            # Right child
            if i < len(vals) and vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        
        return root
    
    def serialize_with_structure(self, root: Optional[TreeNode]) -> str:
        """
        Serialize with explicit structure information
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: JSON serialized string with structure
        """
        def tree_to_dict(node):
            if not node:
                return None
            
            return {
                'val': node.val,
                'left': tree_to_dict(node.left),
                'right': tree_to_dict(node.right)
            }
        
        tree_dict = tree_to_dict(root)
        return json.dumps(tree_dict)
    
    def deserialize_from_structure(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize from JSON structure
        
        Args:
            data: JSON serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        def dict_to_tree(tree_dict):
            if not tree_dict:
                return None
            
            node = TreeNode(tree_dict['val'])
            node.left = dict_to_tree(tree_dict['left'])
            node.right = dict_to_tree(tree_dict['right'])
            return node
        
        if not data or data == "null":
            return None
        
        tree_dict = json.loads(data)
        return dict_to_tree(tree_dict)
    
    # ==================== K-DISTANCE FROM NODE ====================
    
    def distance_k_from_node(self, root: Optional[TreeNode], target: TreeNode, k: int) -> List[int]:
        """
        Find all nodes at distance k from target node (LeetCode 863)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            root: Root of binary tree
            target: Target node
            k: Distance
        
        Returns:
            List of node values at distance k
        """
        # Build parent pointers
        parent_map = {}
        
        def build_parent_map(node, parent):
            if not node:
                return
            parent_map[node] = parent
            build_parent_map(node.left, node)
            build_parent_map(node.right, node)
        
        build_parent_map(root, None)
        
        # BFS from target node
        visited = set()
        queue = deque([(target, 0)])
        visited.add(target)
        result = []
        
        while queue:
            node, distance = queue.popleft()
            
            if distance == k:
                result.append(node.val)
                continue
            
            if distance < k:
                # Explore children and parent
                for neighbor in [node.left, node.right, parent_map.get(node)]:
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
        
        return result
    
    def distance_k_with_return_values(self, root: Optional[TreeNode], target: TreeNode, k: int) -> List[int]:
        """
        Find nodes at distance k using return values approach
        
        Args:
            root: Root of binary tree
            target: Target node
            k: Distance
        
        Returns:
            List of node values at distance k
        """
        result = []
        
        def find_nodes_at_distance(node, distance):
            """Find all nodes at given distance from node"""
            if not node or distance < 0:
                return
            
            if distance == 0:
                result.append(node.val)
                return
            
            find_nodes_at_distance(node.left, distance - 1)
            find_nodes_at_distance(node.right, distance - 1)
        
        def dfs(node):
            """DFS that returns distance to target if found, -1 otherwise"""
            if not node:
                return -1
            
            if node == target:
                # Found target, find nodes at distance k in subtree
                find_nodes_at_distance(node, k)
                return 0
            
            # Check left subtree
            left_dist = dfs(node.left)
            if left_dist != -1:
                # Target found in left subtree
                if left_dist + 1 == k:
                    result.append(node.val)
                else:
                    # Find nodes in right subtree
                    find_nodes_at_distance(node.right, k - left_dist - 2)
                return left_dist + 1
            
            # Check right subtree
            right_dist = dfs(node.right)
            if right_dist != -1:
                # Target found in right subtree
                if right_dist + 1 == k:
                    result.append(node.val)
                else:
                    # Find nodes in left subtree
                    find_nodes_at_distance(node.left, k - right_dist - 2)
                return right_dist + 1
            
            return -1
        
        dfs(root)
        return result
    
    def distance_k_bidirectional(self, root: Optional[TreeNode], target: TreeNode, k: int) -> List[int]:
        """
        Find nodes at distance k using bidirectional search
        
        Args:
            root: Root of binary tree
            target: Target node
            k: Distance
        
        Returns:
            List of node values at distance k
        """
        # Convert tree to graph
        graph = defaultdict(list)
        
        def build_graph(node):
            if not node:
                return
            
            if node.left:
                graph[node].append(node.left)
                graph[node.left].append(node)
                build_graph(node.left)
            
            if node.right:
                graph[node].append(node.right)
                graph[node.right].append(node)
                build_graph(node.right)
        
        build_graph(root)
        
        # BFS from target
        visited = set([target])
        current_level = [target]
        
        for distance in range(k):
            next_level = []
            for node in current_level:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.append(neighbor)
            current_level = next_level
        
        return [node.val for node in current_level]
    
    # ==================== BOUNDARY OF BINARY TREE ====================
    
    def boundary_of_binary_tree(self, root: Optional[TreeNode]) -> List[int]:
        """
        Find boundary of binary tree (anti-clockwise)
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            List of boundary node values
        """
        if not root:
            return []
        
        boundary = []
        
        def is_leaf(node):
            return node and not node.left and not node.right
        
        def add_left_boundary(node):
            """Add left boundary (excluding leaves)"""
            while node:
                if not is_leaf(node):
                    boundary.append(node.val)
                
                if node.left:
                    node = node.left
                else:
                    node = node.right
        
        def add_leaves(node):
            """Add all leaves"""
            if not node:
                return
            
            if is_leaf(node):
                boundary.append(node.val)
                return
            
            add_leaves(node.left)
            add_leaves(node.right)
        
        def add_right_boundary(node):
            """Add right boundary (excluding leaves, in reverse)"""
            temp = []
            
            while node:
                if not is_leaf(node):
                    temp.append(node.val)
                
                if node.right:
                    node = node.right
                else:
                    node = node.left
            
            # Add in reverse order
            boundary.extend(temp[::-1])
        
        # Root
        boundary.append(root.val)
        
        # Special case: single node
        if is_leaf(root):
            return boundary
        
        # Left boundary (excluding root and leaves)
        if root.left:
            add_left_boundary(root.left)
        
        # Leaves
        add_leaves(root)
        
        # Right boundary (excluding root and leaves)
        if root.right:
            add_right_boundary(root.right)
        
        return boundary
    
    def boundary_detailed(self, root: Optional[TreeNode]) -> Dict[str, List[int]]:
        """
        Get detailed boundary information
        
        Args:
            root: Root of binary tree
        
        Returns:
            Dict with separate boundary parts
        """
        if not root:
            return {'left_boundary': [], 'leaves': [], 'right_boundary': []}
        
        def is_leaf(node):
            return node and not node.left and not node.right
        
        left_boundary = []
        leaves = []
        right_boundary = []
        
        # Left boundary
        node = root.left
        while node:
            if not is_leaf(node):
                left_boundary.append(node.val)
            
            if node.left:
                node = node.left
            else:
                node = node.right
        
        # Leaves
        def collect_leaves(node):
            if not node:
                return
            
            if is_leaf(node):
                leaves.append(node.val)
                return
            
            collect_leaves(node.left)
            collect_leaves(node.right)
        
        collect_leaves(root)
        
        # Right boundary
        node = root.right
        temp = []
        while node:
            if not is_leaf(node):
                temp.append(node.val)
            
            if node.right:
                node = node.right
            else:
                node = node.left
        
        right_boundary = temp[::-1]
        
        return {
            'root': [root.val],
            'left_boundary': left_boundary,
            'leaves': leaves,
            'right_boundary': right_boundary
        }
    
    def boundary_traversal_types(self, root: Optional[TreeNode]) -> Dict[str, List[int]]:
        """
        Different types of boundary traversals
        
        Args:
            root: Root of binary tree
        
        Returns:
            Dict with different traversal types
        """
        result = {
            'clockwise': [],
            'anti_clockwise': [],
            'only_boundary_nodes': [],
            'with_internal_visible': []
        }
        
        if not root:
            return result
        
        # Anti-clockwise (standard)
        result['anti_clockwise'] = self.boundary_of_binary_tree(root)
        
        # Clockwise (reverse)
        def boundary_clockwise(node):
            if not node:
                return []
            
            boundary = []
            
            def is_leaf(n):
                return n and not n.left and not n.right
            
            # Root
            boundary.append(node.val)
            
            if is_leaf(node):
                return boundary
            
            # Right boundary (top to bottom)
            if node.right:
                current = node.right
                while current:
                    if not is_leaf(current):
                        boundary.append(current.val)
                    current = current.right if current.right else current.left
            
            # Leaves (right to left)
            def add_leaves_reverse(n):
                if not n:
                    return
                add_leaves_reverse(n.right)
                if is_leaf(n):
                    boundary.append(n.val)
                add_leaves_reverse(n.left)
            
            add_leaves_reverse(node)
            
            # Left boundary (bottom to top)
            if node.left:
                current = node.left
                temp = []
                while current:
                    if not is_leaf(current):
                        temp.append(current.val)
                    current = current.left if current.left else current.right
                boundary.extend(temp[::-1])
            
            return boundary
        
        result['clockwise'] = boundary_clockwise(root)
        
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """Build tree from array representation"""
        if not arr or arr[0] is None:
            return None
        
        root = TreeNode(arr[0])
        queue = deque([root])
        i = 1
        
        while queue and i < len(arr):
            node = queue.popleft()
            
            if i < len(arr) and arr[i] is not None:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            
            if i < len(arr) and arr[i] is not None:
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    def print_tree_structure(self, root: Optional[TreeNode], level: int = 0, prefix: str = "Root: "):
        """Print tree structure"""
        if root is not None:
            print(" " * (level * 4) + prefix + str(root.val))
            if root.left is not None or root.right is not None:
                if root.left:
                    self.print_tree_structure(root.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if root.right:
                    self.print_tree_structure(root.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")
    
    def find_node_by_value(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        """Find node by value"""
        if not root:
            return None
        
        if root.val == target:
            return root
        
        left_result = self.find_node_by_value(root.left, target)
        if left_result:
            return left_result
        
        return self.find_node_by_value(root.right, target)

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Tree Interview Questions Demo ===\n")
    
    solver = TreeInterviewProblems()
    
    # Example 1: Distance Between Two Nodes
    print("1. Distance Between Two Nodes:")
    
    # Build test tree
    distance_arr = [1, 2, 3, 4, 5, 6, 7, 8, None, None, None, None, None, None, 9]
    distance_tree = solver.build_tree_from_array(distance_arr)
    
    print("Tree structure:")
    solver.print_tree_structure(distance_tree)
    
    # Test distance queries
    test_pairs = [(8, 9), (4, 5), (2, 7), (8, 7)]
    for p, q in test_pairs:
        distance = solver.distance_between_nodes(distance_tree, p, q)
        distance_with_path, path = solver.distance_with_path(distance_tree, p, q)
        print(f"Distance between {p} and {q}: {distance}")
        print(f"  Path: {path}")
    print()
    
    # Example 2: Count Nodes in Complete Binary Tree
    print("2. Count Nodes in Complete Binary Tree:")
    
    # Perfect binary tree
    complete_arr = [1, 2, 3, 4, 5, 6, 7]
    complete_tree = solver.build_tree_from_array(complete_arr)
    
    print("Complete tree structure:")
    solver.print_tree_structure(complete_tree)
    
    count1 = solver.count_nodes_complete_tree(complete_tree)
    count2 = solver.count_nodes_binary_search(complete_tree)
    count3 = solver.count_nodes_iterative(complete_tree)
    
    print(f"Node count (optimized): {count1}")
    print(f"Node count (binary search): {count2}")
    print(f"Node count (iterative): {count3}")
    print()
    
    # Example 3: Recover BST
    print("3. Recover BST (Two nodes swapped):")
    
    # Create BST with two swapped nodes
    bst_arr = [1, 3, None, None, 2]  # Should be [3, 1, None, None, 2]
    bst_tree = solver.build_tree_from_array(bst_arr)
    
    print("BST before recovery:")
    solver.print_tree_structure(bst_tree)
    
    # Find swapped nodes
    first, second = solver.find_swapped_nodes(bst_tree)
    print(f"Swapped nodes: {first.val if first else None}, {second.val if second else None}")
    
    # Recover BST
    solver.recover_bst(bst_tree)
    print("BST after recovery:")
    solver.print_tree_structure(bst_tree)
    print()
    
    # Example 4: Serialize and Deserialize
    print("4. Serialize and Deserialize Binary Tree:")
    
    serialize_arr = [1, 2, 3, None, None, 4, 5]
    serialize_tree = solver.build_tree_from_array(serialize_arr)
    
    print("Original tree:")
    solver.print_tree_structure(serialize_tree)
    
    # Preorder serialization
    preorder_data = solver.serialize_preorder(serialize_tree)
    print(f"Preorder serialized: {preorder_data}")
    
    deserialized_preorder = solver.deserialize_preorder(preorder_data)
    print("Deserialized from preorder:")
    solver.print_tree_structure(deserialized_preorder)
    
    # Level order serialization
    level_data = solver.serialize_level_order(serialize_tree)
    print(f"Level order serialized: {level_data}")
    
    deserialized_level = solver.deserialize_level_order(level_data)
    print("Deserialized from level order:")
    solver.print_tree_structure(deserialized_level)
    
    # JSON serialization
    json_data = solver.serialize_with_structure(serialize_tree)
    print(f"JSON serialized: {json_data}")
    print()
    
    # Example 5: K-Distance from Node
    print("5. K-Distance from Node:")
    
    k_distance_arr = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    k_distance_tree = solver.build_tree_from_array(k_distance_arr)
    
    print("Tree structure:")
    solver.print_tree_structure(k_distance_tree)
    
    target_node = solver.find_node_by_value(k_distance_tree, 5)
    
    for k in [0, 1, 2, 3]:
        nodes_at_k = solver.distance_k_from_node(k_distance_tree, target_node, k)
        nodes_at_k_alt = solver.distance_k_with_return_values(k_distance_tree, target_node, k)
        print(f"Nodes at distance {k} from node 5: {sorted(nodes_at_k)}")
        print(f"  Alternative method: {sorted(nodes_at_k_alt)}")
    print()
    
    # Example 6: Boundary of Binary Tree
    print("6. Boundary of Binary Tree:")
    
    boundary_arr = [1, 2, 3, 4, 5, 6, None, None, None, 7, 8, 9, 10]
    boundary_tree = solver.build_tree_from_array(boundary_arr)
    
    print("Tree structure:")
    solver.print_tree_structure(boundary_tree)
    
    boundary = solver.boundary_of_binary_tree(boundary_tree)
    print(f"Boundary traversal (anti-clockwise): {boundary}")
    
    detailed_boundary = solver.boundary_detailed(boundary_tree)
    print("Detailed boundary:")
    for part, nodes in detailed_boundary.items():
        print(f"  {part}: {nodes}")
    
    different_traversals = solver.boundary_traversal_types(boundary_tree)
    print("Different boundary traversals:")
    for traversal_type, nodes in different_traversals.items():
        if nodes:  # Only print non-empty traversals
            print(f"  {traversal_type}: {nodes}")
    print()
    
    # Example 7: Complex Problem Combinations
    print("7. Complex Problem Combinations:")
    
    # Large tree for comprehensive testing
    complex_arr = list(range(1, 32))  # Complete binary tree with 31 nodes
    complex_tree = solver.build_tree_from_array(complex_arr)
    
    print(f"Complex tree with {len(complex_arr)} nodes:")
    
    # Count nodes
    node_count = solver.count_nodes_complete_tree(complex_tree)
    print(f"Node count: {node_count}")
    
    # Serialize and check
    serialized = solver.serialize_level_order(complex_tree)
    print(f"Serialization length: {len(serialized)}")
    
    # Distance queries
    sample_distances = [(8, 15), (16, 31), (1, 31)]
    for p, q in sample_distances:
        distance = solver.distance_between_nodes(complex_tree, p, q)
        print(f"Distance({p}, {q}): {distance}")
    
    # Boundary
    boundary = solver.boundary_of_binary_tree(complex_tree)
    print(f"Boundary length: {len(boundary)}")
    print(f"First 10 boundary nodes: {boundary[:10]}")
    
    # K-distance from root
    root_target = solver.find_node_by_value(complex_tree, 1)
    nodes_at_3 = solver.distance_k_from_node(complex_tree, root_target, 3)
    print(f"Nodes at distance 3 from root: {sorted(nodes_at_3)}")
    
    print("\n=== Demo Complete ===") 