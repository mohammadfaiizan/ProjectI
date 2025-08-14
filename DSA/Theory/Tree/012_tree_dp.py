"""
Tree Dynamic Programming - Advanced DP Algorithms on Trees
This module implements comprehensive tree DP algorithms for optimization problems on trees.
"""

from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict
import sys

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeDP:
    """Tree Dynamic Programming algorithms"""
    
    def __init__(self):
        """Initialize tree DP solver"""
        pass
    
    # ==================== TREE DIAMETER USING DP ====================
    
    def tree_diameter_dp(self, root: Optional[TreeNode]) -> int:
        """
        Find diameter of binary tree using tree DP
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Diameter of tree
        """
        self.diameter = 0
        
        def dfs(node: Optional[TreeNode]) -> int:
            """Returns height of subtree and updates diameter"""
            if not node:
                return 0
            
            left_height = dfs(node.left)
            right_height = dfs(node.right)
            
            # Update diameter passing through current node
            self.diameter = max(self.diameter, left_height + right_height)
            
            # Return height of current subtree
            return 1 + max(left_height, right_height)
        
        dfs(root)
        return self.diameter
    
    def tree_diameter_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Find diameter and the actual path
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (diameter, path_nodes)
        """
        self.max_diameter = 0
        self.diameter_path = []
        
        def dfs(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            """Returns (height, path_to_deepest_node)"""
            if not node:
                return 0, []
            
            left_height, left_path = dfs(node.left)
            right_height, right_path = dfs(node.right)
            
            # Current diameter through this node
            current_diameter = left_height + right_height
            
            if current_diameter > self.max_diameter:
                self.max_diameter = current_diameter
                # Build path: left_path + current + right_path_reversed
                self.diameter_path = (left_path[::-1] + [node.val] + right_path)
            
            # Return height and path to deepest node
            if left_height > right_height:
                return 1 + left_height, left_path + [node.val]
            else:
                return 1 + right_height, right_path + [node.val]
        
        dfs(root)
        return self.max_diameter, self.diameter_path
    
    def tree_diameter_general_tree(self, adj_list: Dict[int, List[int]], root: int = 0) -> int:
        """
        Find diameter of general tree (not binary)
        
        Args:
            adj_list: Adjacency list representation
            root: Root node (can be any node)
        
        Returns:
            int: Diameter of tree
        """
        diameter = 0
        
        def dfs(node: int, parent: int) -> int:
            nonlocal diameter
            
            # Find two largest heights among children
            heights = []
            
            for child in adj_list.get(node, []):
                if child != parent:
                    height = dfs(child, node)
                    heights.append(height)
            
            heights.sort(reverse=True)
            
            # Update diameter
            if len(heights) >= 2:
                diameter = max(diameter, heights[0] + heights[1])
            elif len(heights) == 1:
                diameter = max(diameter, heights[0])
            
            # Return height of current subtree
            return 1 + (heights[0] if heights else 0)
        
        dfs(root, -1)
        return diameter
    
    # ==================== MAXIMUM PATH SUM PROBLEMS ====================
    
    def max_path_sum(self, root: Optional[TreeNode]) -> int:
        """
        Maximum path sum in binary tree (LeetCode 124)
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Maximum path sum
        """
        self.max_sum = float('-inf')
        
        def dfs(node: Optional[TreeNode]) -> int:
            """Returns maximum sum path starting from node"""
            if not node:
                return 0
            
            # Maximum sum from left and right subtrees
            left_sum = max(0, dfs(node.left))   # Take 0 if negative
            right_sum = max(0, dfs(node.right))
            
            # Maximum path sum passing through current node
            path_sum = node.val + left_sum + right_sum
            self.max_sum = max(self.max_sum, path_sum)
            
            # Return maximum sum path starting from current node
            return node.val + max(left_sum, right_sum)
        
        dfs(root)
        return self.max_sum
    
    def max_path_sum_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Maximum path sum with actual path
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_sum, path)
        """
        self.max_sum = float('-inf')
        self.best_path = []
        
        def dfs(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            """Returns (max_sum_from_node, path_from_node)"""
            if not node:
                return 0, []
            
            left_sum, left_path = dfs(node.left)
            right_sum, right_path = dfs(node.right)
            
            # Take only positive contributions
            left_sum = max(0, left_sum)
            right_sum = max(0, right_sum)
            
            # Path sum through current node
            current_path_sum = node.val + left_sum + right_sum
            
            if current_path_sum > self.max_sum:
                self.max_sum = current_path_sum
                # Build path
                if left_sum > 0 and right_sum > 0:
                    self.best_path = left_path[::-1] + [node.val] + right_path
                elif left_sum > 0:
                    self.best_path = left_path[::-1] + [node.val]
                elif right_sum > 0:
                    self.best_path = [node.val] + right_path
                else:
                    self.best_path = [node.val]
            
            # Return best path starting from current node
            if left_sum > right_sum:
                return node.val + left_sum, left_path + [node.val]
            else:
                return node.val + right_sum, [node.val] + right_path
        
        dfs(root)
        return self.max_sum, self.best_path
    
    def max_subtree_sum(self, root: Optional[TreeNode]) -> int:
        """
        Maximum sum of any subtree
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Maximum subtree sum
        """
        self.max_sum = float('-inf')
        
        def dfs(node: Optional[TreeNode]) -> int:
            """Returns sum of subtree rooted at node"""
            if not node:
                return 0
            
            left_sum = dfs(node.left)
            right_sum = dfs(node.right)
            
            subtree_sum = node.val + left_sum + right_sum
            self.max_sum = max(self.max_sum, subtree_sum)
            
            return subtree_sum
        
        dfs(root)
        return self.max_sum
    
    def min_subtree_sum(self, root: Optional[TreeNode]) -> int:
        """
        Minimum sum of any subtree
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Minimum subtree sum
        """
        self.min_sum = float('inf')
        
        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            left_sum = dfs(node.left)
            right_sum = dfs(node.right)
            
            subtree_sum = node.val + left_sum + right_sum
            self.min_sum = min(self.min_sum, subtree_sum)
            
            return subtree_sum
        
        dfs(root)
        return self.min_sum
    
    # ==================== INDEPENDENT SETS ON TREES ====================
    
    def max_independent_set(self, root: Optional[TreeNode]) -> int:
        """
        Maximum size independent set in tree
        An independent set contains no two adjacent nodes
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Size of maximum independent set
        """
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            """Returns (max_size_including_node, max_size_excluding_node)"""
            if not node:
                return 0, 0
            
            left_incl, left_excl = dfs(node.left)
            right_incl, right_excl = dfs(node.right)
            
            # Include current node: can't include children
            incl = 1 + left_excl + right_excl
            
            # Exclude current node: can include or exclude children
            excl = max(left_incl, left_excl) + max(right_incl, right_excl)
            
            return incl, excl
        
        incl, excl = dfs(root)
        return max(incl, excl)
    
    def max_independent_set_with_nodes(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Maximum independent set with actual nodes
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_size, nodes_in_set)
        """
        def dfs(node: Optional[TreeNode]) -> Tuple[Tuple[int, List[int]], Tuple[int, List[int]]]:
            """Returns ((size_incl, nodes_incl), (size_excl, nodes_excl))"""
            if not node:
                return (0, []), (0, [])
            
            left_result = dfs(node.left)
            right_result = dfs(node.right)
            
            left_incl, left_excl = left_result
            right_incl, right_excl = right_result
            
            # Include current node
            incl_size = 1 + left_excl[0] + right_excl[0]
            incl_nodes = [node.val] + left_excl[1] + right_excl[1]
            
            # Exclude current node
            if left_incl[0] + right_incl[0] > left_excl[0] + right_excl[0]:
                excl_size = left_incl[0] + right_incl[0]
                excl_nodes = left_incl[1] + right_incl[1]
            else:
                excl_size = left_excl[0] + right_excl[0]
                excl_nodes = left_excl[1] + right_excl[1]
            
            return (incl_size, incl_nodes), (excl_size, excl_nodes)
        
        incl_result, excl_result = dfs(root)
        
        if incl_result[0] > excl_result[0]:
            return incl_result[0], incl_result[1]
        else:
            return excl_result[0], excl_result[1]
    
    def count_independent_sets(self, root: Optional[TreeNode]) -> int:
        """
        Count number of independent sets
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Number of independent sets
        """
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            """Returns (count_including_node, count_excluding_node)"""
            if not node:
                return 1, 1  # Empty set is valid
            
            left_incl, left_excl = dfs(node.left)
            right_incl, right_excl = dfs(node.right)
            
            # Include current node: children must be excluded
            incl = left_excl * right_excl
            
            # Exclude current node: children can be included or excluded
            excl = (left_incl + left_excl) * (right_incl + right_excl)
            
            return incl, excl
        
        incl, excl = dfs(root)
        return incl + excl
    
    # ==================== HOUSE ROBBER III (LEETCODE 337) ====================
    
    def rob_houses_tree(self, root: Optional[TreeNode]) -> int:
        """
        House Robber III - Rob houses arranged in binary tree
        Cannot rob two directly connected houses
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree (house values)
        
        Returns:
            int: Maximum money that can be robbed
        """
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            """Returns (max_money_if_rob_node, max_money_if_not_rob_node)"""
            if not node:
                return 0, 0
            
            left_rob, left_not_rob = dfs(node.left)
            right_rob, right_not_rob = dfs(node.right)
            
            # Rob current house: cannot rob children
            rob_current = node.val + left_not_rob + right_not_rob
            
            # Don't rob current house: can rob or not rob children
            not_rob_current = max(left_rob, left_not_rob) + max(right_rob, right_not_rob)
            
            return rob_current, not_rob_current
        
        rob, not_rob = dfs(root)
        return max(rob, not_rob)
    
    def rob_houses_with_plan(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        House Robber with actual houses to rob
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_money, houses_to_rob)
        """
        def dfs(node: Optional[TreeNode]) -> Tuple[Tuple[int, List[int]], Tuple[int, List[int]]]:
            """Returns ((money_rob, houses_rob), (money_not_rob, houses_not_rob))"""
            if not node:
                return (0, []), (0, [])
            
            left_result = dfs(node.left)
            right_result = dfs(node.right)
            
            left_rob, left_not_rob = left_result
            right_rob, right_not_rob = right_result
            
            # Rob current house
            rob_money = node.val + left_not_rob[0] + right_not_rob[0]
            rob_houses = [node.val] + left_not_rob[1] + right_not_rob[1]
            
            # Don't rob current house
            left_best = left_rob if left_rob[0] > left_not_rob[0] else left_not_rob
            right_best = right_rob if right_rob[0] > right_not_rob[0] else right_not_rob
            
            not_rob_money = left_best[0] + right_best[0]
            not_rob_houses = left_best[1] + right_best[1]
            
            return (rob_money, rob_houses), (not_rob_money, not_rob_houses)
        
        rob_result, not_rob_result = dfs(root)
        
        if rob_result[0] > not_rob_result[0]:
            return rob_result[0], rob_result[1]
        else:
            return not_rob_result[0], not_rob_result[1]
    
    def rob_houses_general_tree(self, adj_list: Dict[int, List[int]], values: Dict[int, int], root: int) -> int:
        """
        House robber on general tree
        
        Args:
            adj_list: Adjacency list representation
            values: Node values (house values)
            root: Root node
        
        Returns:
            int: Maximum money that can be robbed
        """
        def dfs(node: int, parent: int) -> Tuple[int, int]:
            """Returns (max_money_rob_node, max_money_not_rob_node)"""
            rob_current = values.get(node, 0)
            not_rob_current = 0
            
            for child in adj_list.get(node, []):
                if child != parent:
                    child_rob, child_not_rob = dfs(child, node)
                    
                    # If we rob current, we can't rob children
                    rob_current += child_not_rob
                    
                    # If we don't rob current, we can rob or not rob children
                    not_rob_current += max(child_rob, child_not_rob)
            
            return rob_current, not_rob_current
        
        rob, not_rob = dfs(root, -1)
        return max(rob, not_rob)
    
    # ==================== TREE COLORING PROBLEMS ====================
    
    def min_colors_tree_2coloring(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree can be 2-colored (bipartite)
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree can be 2-colored
        """
        def dfs(node: Optional[TreeNode], color: int, colors: Dict[TreeNode, int]) -> bool:
            if not node:
                return True
            
            if node in colors:
                return colors[node] == color
            
            colors[node] = color
            
            # Color children with opposite color
            return (dfs(node.left, 1 - color, colors) and 
                    dfs(node.right, 1 - color, colors))
        
        return dfs(root, 0, {})
    
    def min_colors_tree_3coloring(self, adj_list: Dict[int, List[int]], root: int = 0) -> int:
        """
        Find minimum colors needed for tree (always 2 or 3)
        
        Args:
            adj_list: Adjacency list representation
            root: Root node
        
        Returns:
            int: Minimum colors needed
        """
        def dfs(node: int, parent: int, parent_color: int) -> int:
            """Returns minimum colors needed for subtree"""
            if not adj_list.get(node):
                return 1  # Leaf node needs 1 color
            
            # Try to color current node
            current_color = 0 if parent_color != 0 else 1
            max_colors = 1
            
            for child in adj_list.get(node, []):
                if child != parent:
                    child_colors = dfs(child, node, current_color)
                    max_colors = max(max_colors, child_colors)
            
            return max_colors
        
        return dfs(root, -1, -1)
    
    def tree_coloring_dp(self, adj_list: Dict[int, List[int]], k: int, root: int = 0) -> int:
        """
        Count ways to color tree with k colors such that no adjacent nodes have same color
        
        Args:
            adj_list: Adjacency list representation
            k: Number of colors available
            root: Root node
        
        Returns:
            int: Number of ways to color tree
        """
        def dfs(node: int, parent: int) -> int:
            """Returns number of ways to color subtree rooted at node"""
            if not adj_list.get(node):
                return k  # Leaf can be colored in k ways
            
            ways = k if parent == -1 else k - 1  # Current node coloring options
            
            for child in adj_list.get(node, []):
                if child != parent:
                    child_ways = dfs(child, node)
                    ways *= child_ways
            
            return ways
        
        return dfs(root, -1)
    
    # ==================== ADVANCED TREE DP PROBLEMS ====================
    
    def tree_distance_sum(self, adj_list: Dict[int, List[int]], root: int = 0) -> Dict[int, int]:
        """
        For each node, calculate sum of distances to all other nodes
        
        Time Complexity: O(n)
        
        Args:
            adj_list: Adjacency list representation
            root: Root node
        
        Returns:
            Dict mapping node to sum of distances
        """
        n = len(adj_list)
        subtree_size = {}
        down_distance = {}
        total_distance = {}
        
        # First DFS: Calculate subtree sizes and down distances
        def dfs1(node: int, parent: int) -> int:
            size = 1
            dist_sum = 0
            
            for child in adj_list.get(node, []):
                if child != parent:
                    child_size = dfs1(child, node)
                    size += child_size
                    dist_sum += down_distance[child] + child_size
            
            subtree_size[node] = size
            down_distance[node] = dist_sum
            return size
        
        # Second DFS: Calculate total distances using re-rooting
        def dfs2(node: int, parent: int, parent_contribution: int):
            total_distance[node] = down_distance[node] + parent_contribution
            
            for child in adj_list.get(node, []):
                if child != parent:
                    # Calculate contribution from parent and other subtrees
                    other_contribution = (total_distance[node] - down_distance[child] - subtree_size[child] + 
                                        n - subtree_size[child])
                    dfs2(child, node, other_contribution)
        
        dfs1(root, -1)
        dfs2(root, -1, 0)
        
        return total_distance
    
    def tree_centroid_dp(self, adj_list: Dict[int, List[int]]) -> List[int]:
        """
        Find centroid(s) of tree using DP
        
        Args:
            adj_list: Adjacency list representation
        
        Returns:
            List of centroid nodes (1 or 2 nodes)
        """
        n = len(adj_list)
        subtree_size = {}
        
        def dfs(node: int, parent: int) -> int:
            size = 1
            for child in adj_list.get(node, []):
                if child != parent:
                    size += dfs(child, node)
            subtree_size[node] = size
            return size
        
        # Calculate subtree sizes from arbitrary root
        dfs(0, -1)
        
        centroids = []
        
        def find_centroids(node: int, parent: int):
            is_centroid = True
            
            # Check if any subtree has more than n/2 nodes
            for child in adj_list.get(node, []):
                if child != parent:
                    if subtree_size[child] > n // 2:
                        is_centroid = False
                    find_centroids(child, node)
            
            # Check parent subtree
            if parent != -1 and n - subtree_size[node] > n // 2:
                is_centroid = False
            
            if is_centroid:
                centroids.append(node)
        
        find_centroids(0, -1)
        return centroids
    
    def tree_matching_dp(self, adj_list: Dict[int, List[int]], root: int = 0) -> int:
        """
        Maximum matching in tree (maximum independent edge set)
        
        Args:
            adj_list: Adjacency list representation
            root: Root node
        
        Returns:
            int: Size of maximum matching
        """
        def dfs(node: int, parent: int) -> Tuple[int, int]:
            """Returns (max_matching_including_parent_edge, max_matching_excluding_parent_edge)"""
            incl = 0  # Include edge to parent
            excl = 0  # Exclude edge to parent
            
            children = [child for child in adj_list.get(node, []) if child != parent]
            
            if not children:
                return 0, 0
            
            # If we include edge to parent, we can't include edges to children
            for child in children:
                child_incl, child_excl = dfs(child, node)
                incl += child_excl
                excl += max(child_incl, child_excl)
            
            return incl, excl
        
        # For root, there's no parent edge
        total_matching = 0
        for child in adj_list.get(root, []):
            child_incl, child_excl = dfs(child, root)
            total_matching += max(child_incl, child_excl)
        
        return total_matching
    
    # ==================== UTILITY METHODS ====================
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """Build binary tree from array representation"""
        if not arr or arr[0] is None:
            return None
        
        from collections import deque
        
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

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Tree Dynamic Programming Demo ===\n")
    
    tree_dp = TreeDP()
    
    # Example 1: Tree Diameter using DP
    print("1. Tree Diameter using DP:")
    
    # Build test tree
    diameter_arr = [1, 2, 3, 4, 5, None, 6, None, None, 7, 8, None, 9]
    diameter_tree = tree_dp.build_tree_from_array(diameter_arr)
    
    print("Tree structure:")
    tree_dp.print_tree_structure(diameter_tree)
    
    diameter = tree_dp.tree_diameter_dp(diameter_tree)
    diameter_with_path, path = tree_dp.tree_diameter_with_path(diameter_tree)
    
    print(f"Tree diameter: {diameter}")
    print(f"Diameter with path: {diameter_with_path}, path: {path}")
    
    # Test on general tree
    general_tree = {
        0: [1, 2, 3],
        1: [0, 4, 5],
        2: [0],
        3: [0, 6],
        4: [1],
        5: [1, 7],
        6: [3],
        7: [5]
    }
    
    general_diameter = tree_dp.tree_diameter_general_tree(general_tree, 0)
    print(f"General tree diameter: {general_diameter}")
    print()
    
    # Example 2: Maximum Path Sum
    print("2. Maximum Path Sum:")
    
    # Tree with negative values
    path_sum_arr = [10, 2, 10, 20, 1, None, -25, None, None, None, None, 3, 4]
    path_sum_tree = tree_dp.build_tree_from_array(path_sum_arr)
    
    print("Tree structure (with negative values):")
    tree_dp.print_tree_structure(path_sum_tree)
    
    max_sum = tree_dp.max_path_sum(path_sum_tree)
    max_sum_with_path, path = tree_dp.max_path_sum_with_path(path_sum_tree)
    
    print(f"Maximum path sum: {max_sum}")
    print(f"Maximum path sum with path: {max_sum_with_path}, path: {path}")
    
    max_subtree = tree_dp.max_subtree_sum(path_sum_tree)
    min_subtree = tree_dp.min_subtree_sum(path_sum_tree)
    
    print(f"Maximum subtree sum: {max_subtree}")
    print(f"Minimum subtree sum: {min_subtree}")
    print()
    
    # Example 3: Independent Sets
    print("3. Maximum Independent Set:")
    
    # Build tree for independent set
    indep_arr = [5, 1, 4, None, None, 2, None, None, 3]
    indep_tree = tree_dp.build_tree_from_array(indep_arr)
    
    print("Tree structure:")
    tree_dp.print_tree_structure(indep_tree)
    
    max_indep_size = tree_dp.max_independent_set(indep_tree)
    max_indep_with_nodes, nodes = tree_dp.max_independent_set_with_nodes(indep_tree)
    count_indep = tree_dp.count_independent_sets(indep_tree)
    
    print(f"Maximum independent set size: {max_indep_size}")
    print(f"Maximum independent set with nodes: {max_indep_with_nodes}, nodes: {nodes}")
    print(f"Total number of independent sets: {count_indep}")
    print()
    
    # Example 4: House Robber III
    print("4. House Robber III:")
    
    # Houses with money values
    robber_arr = [3, 2, 3, None, 3, None, 1]
    robber_tree = tree_dp.build_tree_from_array(robber_arr)
    
    print("House arrangement (values = money):")
    tree_dp.print_tree_structure(robber_tree)
    
    max_money = tree_dp.rob_houses_tree(robber_tree)
    max_money_with_plan, houses = tree_dp.rob_houses_with_plan(robber_tree)
    
    print(f"Maximum money that can be robbed: {max_money}")
    print(f"Maximum money with plan: {max_money_with_plan}, houses: {houses}")
    
    # Test on general tree
    house_tree = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1],
        5: [2]
    }
    house_values = {0: 10, 1: 5, 2: 8, 3: 3, 4: 7, 5: 4}
    
    max_money_general = tree_dp.rob_houses_general_tree(house_tree, house_values, 0)
    print(f"Maximum money in general tree: {max_money_general}")
    print()
    
    # Example 5: Tree Coloring
    print("5. Tree Coloring:")
    
    # Test 2-coloring
    color_tree = tree_dp.build_tree_from_array([1, 2, 3, 4, 5, 6, 7])
    
    print("Tree for coloring:")
    tree_dp.print_tree_structure(color_tree)
    
    can_2color = tree_dp.min_colors_tree_2coloring(color_tree)
    print(f"Can be 2-colored (bipartite): {can_2color}")
    
    # Test on general tree
    color_adj_list = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 4, 5],
        3: [1],
        4: [2],
        5: [2]
    }
    
    min_colors = tree_dp.min_colors_tree_3coloring(color_adj_list, 0)
    print(f"Minimum colors needed: {min_colors}")
    
    # Count coloring ways
    for k in [2, 3, 4]:
        ways = tree_dp.tree_coloring_dp(color_adj_list, k, 0)
        print(f"Ways to color with {k} colors: {ways}")
    print()
    
    # Example 6: Advanced Tree DP
    print("6. Advanced Tree DP:")
    
    # Distance sum
    adv_tree = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1],
        5: [2]
    }
    
    distance_sums = tree_dp.tree_distance_sum(adv_tree, 0)
    print("Sum of distances from each node:")
    for node, dist_sum in sorted(distance_sums.items()):
        print(f"  Node {node}: {dist_sum}")
    
    # Find centroids
    centroids = tree_dp.tree_centroid_dp(adv_tree)
    print(f"Tree centroids: {centroids}")
    
    # Maximum matching
    max_matching = tree_dp.tree_matching_dp(adv_tree, 0)
    print(f"Maximum matching size: {max_matching}")
    print()
    
    # Example 7: Performance Analysis on Large Trees
    print("7. Performance Analysis:")
    
    # Create larger tree for testing
    large_tree_adj = {}
    large_values = {}
    
    # Create a balanced binary tree structure
    n = 127  # 2^7 - 1 nodes
    for i in range(n):
        large_tree_adj[i] = []
        large_values[i] = i % 10 + 1  # Values 1-10
        
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < n:
            large_tree_adj[i].append(left_child)
            large_tree_adj[left_child] = large_tree_adj.get(left_child, []) + [i]
        
        if right_child < n:
            large_tree_adj[i].append(right_child)
            large_tree_adj[right_child] = large_tree_adj.get(right_child, []) + [i]
    
    print(f"Testing on tree with {n} nodes:")
    
    # Test diameter
    large_diameter = tree_dp.tree_diameter_general_tree(large_tree_adj, 0)
    print(f"Large tree diameter: {large_diameter}")
    
    # Test house robber
    large_max_money = tree_dp.rob_houses_general_tree(large_tree_adj, large_values, 0)
    print(f"Maximum money from large tree: {large_max_money}")
    
    # Test distance sums (just for a few nodes)
    large_distance_sums = tree_dp.tree_distance_sum(large_tree_adj, 0)
    sample_nodes = [0, 1, 2, 50, 100]
    print("Sample distance sums:")
    for node in sample_nodes:
        if node in large_distance_sums:
            print(f"  Node {node}: {large_distance_sums[node]}")
    
    # Test coloring
    large_coloring_ways = tree_dp.tree_coloring_dp(large_tree_adj, 3, 0)
    print(f"Ways to color large tree with 3 colors: {large_coloring_ways}")
    
    # Test matching
    large_matching = tree_dp.tree_matching_dp(large_tree_adj, 0)
    print(f"Maximum matching in large tree: {large_matching}")
    
    print("\n=== Demo Complete ===") 