"""
Problem: Lowest Common Ancestor in Binary Tree
URL: https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-binary-tree/1

Problem Statement:
Find the Lowest Common Ancestor (LCA) of two nodes in a binary tree. LCA is the lowest node that has both nodes as descendants.

Sample Input/Output:
Input: Tree [1, 2, 3, 4, 5], nodes 4 and 5
Output: 2
Explanation: Node 2 is the LCA of nodes 4 and 5.
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def Build_Tree(vals):
    if not vals or vals[0] == -1:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] != -1:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != -1:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root


class Solution:
    def Find_Path(self, root, target, path):
        """
        Find path from root to target node.
        
        Approach: Recursively traverse the tree, adding nodes to path.
        If target found, return True. Otherwise backtrack by removing node.
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return False
        path.append(root)
        if root.val == target:
            return True
        if self.Find_Path(root.left, target, path) or self.Find_Path(root.right, target, path):
            return True
        path.pop()
        return False
    
    def LCA_Path_Storage(self, root, n1, n2):
        """
        Find LCA using path storage and comparison.
        
        Approach: Find paths from root to both nodes, then compare
        paths to find the last common node.
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        path1, path2 = [], []
        if not self.Find_Path(root, n1, path1) or not self.Find_Path(root, n2, path2):
            return None
        i = 0
        while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
            i += 1
        return path1[i - 1]
    
    def LCA_Single_Traversal(self, root, n1, n2):
        """
        Find LCA using single traversal recursion.
        
        Approach: Traverse tree once. If current node matches either n1 or n2,
        return it. Recursively find LCA in left and right subtrees.
        If both subtrees return non-None, current node is LCA.
        Otherwise return the non-None result from subtrees.
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return None
        if root.val == n1 or root.val == n2:
            return root
        left_lca = self.LCA_Single_Traversal(root.left, n1, n2)
        right_lca = self.LCA_Single_Traversal(root.right, n1, n2)
        if left_lca and right_lca:
            return root
        return left_lca if left_lca else right_lca
    
    def Find_LCA(self, root, n1, n2):
        """
        Find Lowest Common Ancestor of two nodes.
        
        Approach: Uses single traversal method for efficiency.
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height of tree
        """
        return self.LCA_Single_Traversal(root, n1, n2)


def Test_LCA_Binary_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5]
    root1 = Build_Tree(vals1)
    lca1 = solution.Find_LCA(root1, 4, 5)
    print(f"Test 1 - LCA of 4 and 5: {lca1.val if lca1 else -1}")
    
    lca2 = solution.Find_LCA(root1, 2, 3)
    print(f"Test 2 - LCA of 2 and 3: {lca2.val if lca2 else -1}")
    
    lca3 = solution.Find_LCA(root1, 4, 3)
    print(f"Test 3 - LCA of 4 and 3: {lca3.val if lca3 else -1}")


if __name__ == "__main__":
    Test_LCA_Binary_Tree()
