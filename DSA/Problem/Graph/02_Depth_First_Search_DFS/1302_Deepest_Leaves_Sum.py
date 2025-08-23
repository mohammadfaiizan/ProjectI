"""
1302. Deepest Leaves Sum - Multiple Approaches
Difficulty: Medium

Given the root of a binary tree, return the sum of values of its deepest leaves.
"""

from typing import Optional, List
from collections import deque, defaultdict

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DeepestLeavesSum:
    """Multiple approaches to find sum of deepest leaves"""
    
    def deepestLeavesSum_dfs_two_pass(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: Two-pass DFS (Find depth, then sum)
        
        First pass finds max depth, second pass sums leaves at max depth.
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        def find_max_depth(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            return 1 + max(find_max_depth(node.left), find_max_depth(node.right))
        
        max_depth = find_max_depth(root)
        
        def sum_at_depth(node: Optional[TreeNode], current_depth: int) -> int:
            if not node:
                return 0
            
            if current_depth == max_depth:
                return node.val
            
            return (sum_at_depth(node.left, current_depth + 1) + 
                   sum_at_depth(node.right, current_depth + 1))
        
        return sum_at_depth(root, 1)
    
    def deepestLeavesSum_dfs_single_pass(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Single-pass DFS with global state
        
        Track max depth and sum simultaneously.
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        self.max_depth = 0
        self.deepest_sum = 0
        
        def dfs(node: Optional[TreeNode], depth: int):
            if not node:
                return
            
            if depth > self.max_depth:
                self.max_depth = depth
                self.deepest_sum = node.val
            elif depth == self.max_depth:
                self.deepest_sum += node.val
            
            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)
        
        dfs(root, 1)
        return self.deepest_sum
    
    def deepestLeavesSum_bfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: BFS Level-by-level
        
        Process each level and keep sum of last level.
        
        Time: O(n), Space: O(w) where w is max width
        """
        if not root:
            return 0
        
        queue = deque([root])
        
        while queue:
            level_sum = 0
            level_size = len(queue)
            
            for _ in range(level_size):
                node = queue.popleft()
                level_sum += node.val
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            # If this is the last level, level_sum will be the answer
            if not queue:  # No more nodes to process
                return level_sum
        
        return 0
    
    def deepestLeavesSum_level_map(self, root: Optional[TreeNode]) -> int:
        """
        Approach 4: Level mapping with DFS
        
        Map each level to sum of nodes at that level.
        
        Time: O(n), Space: O(h + d) where d is depth
        """
        if not root:
            return 0
        
        level_sums = defaultdict(int)
        
        def dfs(node: Optional[TreeNode], level: int):
            if not node:
                return
            
            level_sums[level] += node.val
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)
        
        dfs(root, 0)
        
        # Return sum of deepest level
        max_level = max(level_sums.keys())
        return level_sums[max_level]
    
    def deepestLeavesSum_iterative_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 5: Iterative DFS with stack
        
        Use explicit stack for DFS traversal.
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        stack = [(root, 1)]  # (node, depth)
        max_depth = 0
        deepest_sum = 0
        
        while stack:
            node, depth = stack.pop()
            
            if depth > max_depth:
                max_depth = depth
                deepest_sum = node.val
            elif depth == max_depth:
                deepest_sum += node.val
            
            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))
        
        return deepest_sum

def test_deepest_leaves_sum():
    """Test deepest leaves sum algorithms"""
    solver = DeepestLeavesSum()
    
    # Create test trees
    # Tree 1: [1,2,3,4,5,null,6,7,null,null,null,null,8]
    root1 = TreeNode(1)
    root1.left = TreeNode(2)
    root1.right = TreeNode(3)
    root1.left.left = TreeNode(4)
    root1.left.right = TreeNode(5)
    root1.right.right = TreeNode(6)
    root1.left.left.left = TreeNode(7)
    root1.right.right.right = TreeNode(8)
    
    # Tree 2: [6,7,8,2,7,1,3,9,null,1,4,null,null,null,5]
    root2 = TreeNode(6)
    root2.left = TreeNode(7)
    root2.right = TreeNode(8)
    root2.left.left = TreeNode(2)
    root2.left.right = TreeNode(7)
    root2.right.left = TreeNode(1)
    root2.right.right = TreeNode(3)
    root2.left.left.left = TreeNode(9)
    root2.left.right.left = TreeNode(1)
    root2.left.right.right = TreeNode(4)
    root2.right.right.right = TreeNode(5)
    
    test_cases = [
        (root1, 15, "Tree with deepest leaves 7+8=15"),
        (root2, 19, "Tree with deepest leaves 9+1+4+5=19"),
        (TreeNode(1), 1, "Single node tree"),
    ]
    
    algorithms = [
        ("Two-pass DFS", solver.deepestLeavesSum_dfs_two_pass),
        ("Single-pass DFS", solver.deepestLeavesSum_dfs_single_pass),
        ("BFS", solver.deepestLeavesSum_bfs),
        ("Level Map", solver.deepestLeavesSum_level_map),
        ("Iterative DFS", solver.deepestLeavesSum_iterative_dfs),
    ]
    
    print("=== Testing Deepest Leaves Sum ===")
    
    for root, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(root)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Sum: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_deepest_leaves_sum()

"""
Deepest Leaves Sum demonstrates tree traversal techniques
for level-based processing and depth analysis.
"""
