"""
Problem: Largest Independent Set in Binary Tree
URL: https://www.geeksforgeeks.org/largest-independent-set-problem-dp-26/

Problem Statement:
Given a binary tree, find the size of the largest independent set (LIS). An independent set is a set of nodes such that no two nodes in the set are adjacent (parent-child relationship).

Sample Input/Output:
Input: Binary tree
Output: Size of largest independent set
"""


class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def LIS_Tree_Recursive(self, root: TreeNode) -> int:
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if not root:
            return 0
        
        exclude = self.LIS_Tree_Recursive(root.left) + self.LIS_Tree_Recursive(root.right)
        
        include = 1
        if root.left:
            include += (self.LIS_Tree_Recursive(root.left.left) + 
                       self.LIS_Tree_Recursive(root.left.right))
        if root.right:
            include += (self.LIS_Tree_Recursive(root.right.left) + 
                       self.LIS_Tree_Recursive(root.right.right))
        
        return max(include, exclude)
    
    def LIS_Tree_Memo(self, root: TreeNode, dp: dict) -> int:
        """
        Memoization approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        if root in dp:
            return dp[root]
        
        exclude = self.LIS_Tree_Memo(root.left, dp) + self.LIS_Tree_Memo(root.right, dp)
        
        include = 1
        if root.left:
            include += (self.LIS_Tree_Memo(root.left.left, dp) + 
                       self.LIS_Tree_Memo(root.left.right, dp))
        if root.right:
            include += (self.LIS_Tree_Memo(root.right.left, dp) + 
                       self.LIS_Tree_Memo(root.right.right, dp))
        
        dp[root] = max(include, exclude)
        return dp[root]


def Test_LargestIndependentSet():
    solution = Solution()
    
    root = TreeNode(10)
    root.left = TreeNode(20)
    root.right = TreeNode(30)
    root.left.left = TreeNode(40)
    root.left.right = TreeNode(50)
    root.right.right = TreeNode(60)
    root.left.right.left = TreeNode(70)
    root.left.right.right = TreeNode(80)
    
    result1 = solution.LIS_Tree_Recursive(root)
    assert result1 > 0
    
    dp = {}
    result2 = solution.LIS_Tree_Memo(root, dp)
    assert result2 > 0


if __name__ == "__main__":
    Test_LargestIndependentSet()
