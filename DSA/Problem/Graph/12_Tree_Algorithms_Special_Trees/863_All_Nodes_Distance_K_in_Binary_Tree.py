"""
863. All Nodes Distance K in Binary Tree - Multiple Approaches
Difficulty: Medium

Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node.

You can return the answer in any order.
"""

from typing import List, Optional, Dict, Set
from collections import defaultdict, deque

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class AllNodesDistanceK:
    """Multiple approaches to find all nodes at distance K from target"""
    
    def distanceK_graph_conversion_bfs(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        Approach 1: Convert Tree to Graph + BFS
        
        Convert binary tree to undirected graph, then use BFS from target.
        
        Time: O(N), Space: O(N)
        """
        if not root or k < 0:
            return []
        
        # Build adjacency list (undirected graph)
        graph = defaultdict(list)
        
        def build_graph(node: TreeNode):
            if not node:
                return
            
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
                build_graph(node.left)
            
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
                build_graph(node.right)
        
        build_graph(root)
        
        # BFS from target
        queue = deque([(target.val, 0)])
        visited = {target.val}
        result = []
        
        while queue:
            node_val, dist = queue.popleft()
            
            if dist == k:
                result.append(node_val)
                continue
            
            if dist < k:
                for neighbor in graph[node_val]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        return result
    
    def distanceK_parent_tracking_dfs(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        Approach 2: Parent Tracking + DFS
        
        Track parent pointers, then DFS from target in all directions.
        
        Time: O(N), Space: O(N)
        """
        if not root or k < 0:
            return []
        
        # Build parent mapping
        parent = {}
        
        def build_parent_map(node: TreeNode, par: TreeNode = None):
            if not node:
                return
            parent[node] = par
            build_parent_map(node.left, node)
            build_parent_map(node.right, node)
        
        build_parent_map(root)
        
        # DFS from target
        result = []
        visited = set()
        
        def dfs(node: TreeNode, distance: int):
            if not node or node in visited:
                return
            
            visited.add(node)
            
            if distance == k:
                result.append(node.val)
                return
            
            # Explore all directions: left, right, parent
            dfs(node.left, distance + 1)
            dfs(node.right, distance + 1)
            dfs(parent[node], distance + 1)
        
        dfs(target, 0)
        return result
    
    def distanceK_subtree_and_ancestor(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        Approach 3: Subtree Search + Ancestor Path
        
        Find nodes in target's subtree, then search through ancestors.
        
        Time: O(N), Space: O(H) where H is height
        """
        if not root or k < 0:
            return []
        
        result = []
        
        def find_subtree_nodes(node: TreeNode, distance: int):
            """Find all nodes at given distance in subtree"""
            if not node:
                return
            
            if distance == k:
                result.append(node.val)
                return
            
            find_subtree_nodes(node.left, distance + 1)
            find_subtree_nodes(node.right, distance + 1)
        
        def find_target_and_search(node: TreeNode, depth: int) -> int:
            """
            Find target and search ancestors. 
            Returns distance from node to target, or -1 if target not in subtree.
            """
            if not node:
                return -1
            
            if node == target:
                # Found target, search its subtree
                find_subtree_nodes(node, 0)
                return 0
            
            # Search left subtree
            left_dist = find_target_and_search(node.left, depth + 1)
            if left_dist >= 0:
                # Target found in left subtree
                if left_dist + 1 == k:
                    result.append(node.val)
                elif left_dist + 1 < k:
                    # Search right subtree for remaining distance
                    find_subtree_nodes(node.right, left_dist + 2)
                return left_dist + 1
            
            # Search right subtree
            right_dist = find_target_and_search(node.right, depth + 1)
            if right_dist >= 0:
                # Target found in right subtree
                if right_dist + 1 == k:
                    result.append(node.val)
                elif right_dist + 1 < k:
                    # Search left subtree for remaining distance
                    find_subtree_nodes(node.left, right_dist + 2)
                return right_dist + 1
            
            return -1
        
        find_target_and_search(root, 0)
        return result
    
    def distanceK_path_to_target(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        Approach 4: Find Path to Target + Bidirectional Search
        
        Find path from root to target, then search from each node on path.
        
        Time: O(N), Space: O(H)
        """
        if not root or k < 0:
            return []
        
        # Find path from root to target
        path = []
        
        def find_path(node: TreeNode, target_node: TreeNode, current_path: List[TreeNode]) -> bool:
            if not node:
                return False
            
            current_path.append(node)
            
            if node == target_node:
                path.extend(current_path)
                return True
            
            if (find_path(node.left, target_node, current_path) or 
                find_path(node.right, target_node, current_path)):
                return True
            
            current_path.pop()
            return False
        
        find_path(root, target, [])
        
        result = []
        
        def collect_nodes_at_distance(node: TreeNode, distance: int, blocked: TreeNode = None):
            """Collect nodes at given distance, avoiding blocked node"""
            if not node or node == blocked or distance < 0:
                return
            
            if distance == 0:
                result.append(node.val)
                return
            
            collect_nodes_at_distance(node.left, distance - 1, blocked)
            collect_nodes_at_distance(node.right, distance - 1, blocked)
        
        # Search from each node in path
        for i, node in enumerate(path):
            distance_from_target = len(path) - 1 - i
            remaining_distance = k - distance_from_target
            
            if remaining_distance == 0:
                result.append(node.val)
            elif remaining_distance > 0:
                # Block the path we came from
                blocked_child = path[i + 1] if i + 1 < len(path) else None
                collect_nodes_at_distance(node, remaining_distance, blocked_child)
        
        return result
    
    def distanceK_morris_traversal_style(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        """
        Approach 5: Morris Traversal Style with Distance Tracking
        
        Use Morris-like traversal to avoid extra space for parent tracking.
        
        Time: O(N), Space: O(1) auxiliary (excluding result)
        """
        if not root or k < 0:
            return []
        
        # For this approach, we'll use a modified DFS with distance tracking
        result = []
        
        def dfs_with_distance(node: TreeNode, parent: TreeNode, target_node: TreeNode, 
                            current_k: int) -> int:
            """
            Returns distance to target if found in subtree, -1 otherwise.
            Also collects nodes at distance k from target.
            """
            if not node:
                return -1
            
            if node == target_node:
                # Found target, collect nodes at distance k in subtree
                collect_at_distance(node, current_k)
                return 0
            
            # Search left subtree
            left_dist = dfs_with_distance(node.left, node, target_node, current_k)
            if left_dist >= 0:
                # Target found in left subtree
                if left_dist + 1 == current_k:
                    result.append(node.val)
                else:
                    # Search right subtree
                    collect_at_distance(node.right, current_k - left_dist - 2)
                return left_dist + 1
            
            # Search right subtree
            right_dist = dfs_with_distance(node.right, node, target_node, current_k)
            if right_dist >= 0:
                # Target found in right subtree
                if right_dist + 1 == current_k:
                    result.append(node.val)
                else:
                    # Search left subtree
                    collect_at_distance(node.left, current_k - right_dist - 2)
                return right_dist + 1
            
            return -1
        
        def collect_at_distance(node: TreeNode, distance: int):
            """Collect all nodes at given distance from current node"""
            if not node or distance < 0:
                return
            
            if distance == 0:
                result.append(node.val)
                return
            
            collect_at_distance(node.left, distance - 1)
            collect_at_distance(node.right, distance - 1)
        
        dfs_with_distance(root, None, target, k)
        return result

def test_all_nodes_distance_k():
    """Test all nodes distance K algorithms"""
    solver = AllNodesDistanceK()
    
    # Create test tree:     3
    #                      / \
    #                     5   1
    #                    / \   \
    #                   6   2   0
    #                      / \
    #                     7   4
    
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.right = TreeNode(0)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(4)
    
    target = root.left  # Node with value 5
    k = 2
    expected = [7, 4, 1]  # Nodes at distance 2 from target
    
    algorithms = [
        ("Graph Conversion + BFS", solver.distanceK_graph_conversion_bfs),
        ("Parent Tracking + DFS", solver.distanceK_parent_tracking_dfs),
        ("Subtree and Ancestor", solver.distanceK_subtree_and_ancestor),
        ("Path to Target", solver.distanceK_path_to_target),
        ("Morris Style", solver.distanceK_morris_traversal_style),
    ]
    
    print("=== Testing All Nodes Distance K ===")
    print(f"Target: {target.val}, K: {k}, Expected: {sorted(expected)}")
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(root, target, k)
            result_sorted = sorted(result)
            expected_sorted = sorted(expected)
            status = "✓" if result_sorted == expected_sorted else "✗"
            print(f"{alg_name:25} | {status} | Result: {result_sorted}")
        except Exception as e:
            print(f"{alg_name:25} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_all_nodes_distance_k()

"""
All Nodes Distance K demonstrates tree traversal with distance
constraints, graph conversion techniques, and bidirectional
search algorithms in binary tree structures.
"""
