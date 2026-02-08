"""
Problem: Height Of Tree
URL: https://practice.geeksforgeeks.org/problems/height-of-binary-tree/1

Problem Statement:
Given a binary tree, find its height. Height of a tree is the number of edges in the longest path from root to a leaf node.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 2
Explanation: Longest path from root to leaf has 2 edges (e.g., 1->2->4).
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


def Print_Tree(root):
    if not root:
        return
    q = deque([root])
    result = []
    while q:
        node = q.popleft()
        result.append(str(node.val))
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    print(" ".join(result))


class Solution:
    def Height_Recursive(self, root):
        """
        Recursive approach: Height is max of left and right subtree heights + 1
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return 0
        return 1 + max(self.Height_Recursive(root.left), self.Height_Recursive(root.right))

    def Height_Iterative(self, root):
        """
        Iterative BFS approach: Count levels using queue
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        """
        if not root:
            return 0
        q = deque([root])
        height = 0
        while q:
            size = len(q)
            height += 1
            for _ in range(size):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return height


def Test_Height_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", solution.Height_Recursive(root1))
    print("Test 1 - Iterative:", solution.Height_Iterative(root1))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Recursive:", solution.Height_Recursive(root2))
    print("Test 2 - Iterative:", solution.Height_Iterative(root2))
    
    vals3 = [1]
    root3 = Build_Tree(vals3)
    print("Test 3 - Recursive:", solution.Height_Recursive(root3))
    print("Test 3 - Iterative:", solution.Height_Iterative(root3))


if __name__ == "__main__":
    Test_Height_Of_Tree()
