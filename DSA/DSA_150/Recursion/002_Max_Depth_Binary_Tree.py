"""
Problem: Max Depth/Height of a Binary Tree
URL: https://leetcode.com/problems/maximum-depth-of-binary-tree/

Problem Statement:
Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Sample Input/Output:
Input: root = [3,9,20,null,null,15,7]
Output: 3
Explanation: The tree has depth 3

Input: root = [1,null,2]
Output: 2
Explanation: The tree has depth 2
"""

from typing import List, Optional
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def Max_Depth_Level_Order_Iterative(self, root: Optional[TreeNode]) -> int:
        """
        Level Order Traversal - Iterative approach using queue
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        """
        if not root:
            return 0
        
        queue = deque([root])
        depth = 0
        
        while queue:
            depth += 1
            level_size = len(queue)
            
            for _ in range(level_size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return depth
    
    def Max_Depth_DFS_Iterative(self, root: Optional[TreeNode]) -> int:
        """
        DFS Iterative - Using stack with depth tracking
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        if not root:
            return 0
        
        stack = [(root, 1)]
        max_depth = 0
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            
            if node.right:
                stack.append((node.right, depth + 1))
            if node.left:
                stack.append((node.left, depth + 1))
        
        return max_depth
    
    def Max_Depth_Recursive_Optimal(self, root: Optional[TreeNode]) -> int:
        """
        Recursive DFS - Optimal solution
        Time Complexity: O(n)
        Space Complexity: O(h) - recursion stack
        """
        if not root:
            return 0
        
        left_depth = self.Max_Depth_Recursive_Optimal(root.left)
        right_depth = self.Max_Depth_Recursive_Optimal(root.right)
        
        return 1 + max(left_depth, right_depth)
    
    def Max_Depth_Recursive_Simple(self, root: Optional[TreeNode]) -> int:
        """
        Simple Recursive - Direct approach
        Time Complexity: O(n)
        Space Complexity: O(h) - recursion stack
        """
        def Dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            return 1 + max(Dfs(node.left), Dfs(node.right))
        
        return Dfs(root)
    
    def Max_Depth_Recursive_Helper(self, root: Optional[TreeNode]) -> int:
        """
        Recursive Helper - Using helper function
        Time Complexity: O(n)
        Space Complexity: O(h) - recursion stack
        """
        def Calculate_Depth(node: Optional[TreeNode], current_depth: int) -> int:
            if not node:
                return current_depth
            
            left_depth = Calculate_Depth(node.left, current_depth + 1)
            right_depth = Calculate_Depth(node.right, current_depth + 1)
            
            return max(left_depth, right_depth)
        
        return Calculate_Depth(root, 0)
    
    def Max_Depth_Postorder_Traversal(self, root: Optional[TreeNode]) -> int:
        """
        Postorder Traversal - Bottom-up approach
        Time Complexity: O(n)
        Space Complexity: O(h) - recursion stack
        """
        def Postorder(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            left_height = Postorder(node.left)
            right_height = Postorder(node.right)
            
            return max(left_height, right_height) + 1
        
        return Postorder(root)

def Build_Tree_From_List(values: List) -> Optional[TreeNode]:
    """Helper function to build tree from list representation"""
    if not values:
        return None
    
    root = TreeNode(values[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(values):
        node = queue.popleft()
        
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    
    return root

def Test_Max_Depth():
    solution = Solution()
    
    test_cases = [
        ([3,9,20,None,None,15,7], 3),
        ([1,None,2], 2),
        ([1,2,3,4,5], 3),
        ([], 0),
        ([1], 1)
    ]
    
    for values, expected in test_cases:
        root = Build_Tree_From_List(values)
        
        result1 = solution.Max_Depth_Level_Order_Iterative(root)
        result2 = solution.Max_Depth_DFS_Iterative(root)
        result3 = solution.Max_Depth_Recursive_Optimal(root)
        result4 = solution.Max_Depth_Recursive_Simple(root)
        result5 = solution.Max_Depth_Recursive_Helper(root)
        result6 = solution.Max_Depth_Postorder_Traversal(root)
        
        print(f"Tree: {values}")
        print(f"Expected: {expected}")
        print(f"Level Order Iterative: {result1}")
        print(f"DFS Iterative: {result2}")
        print(f"Recursive Optimal: {result3}")
        print(f"Recursive Simple: {result4}")
        print(f"Recursive Helper: {result5}")
        print(f"Postorder Traversal: {result6}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Max_Depth()
