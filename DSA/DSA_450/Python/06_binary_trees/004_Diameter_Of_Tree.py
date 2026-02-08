"""
Problem: Diameter Of Tree
URL: https://practice.geeksforgeeks.org/problems/diameter-of-binary-tree/1

Problem Statement:
Given a binary tree, find its diameter. Diameter of a tree is the number of nodes on the longest path between any two nodes in the tree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 5
Explanation: Longest path is 4->2->1->3->7 with 5 nodes.
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
    def Diameter_Optimized(self, root):
        """
        Optimized single pass: Calculate height and diameter together
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        diameter = [0]
        self.Height_Helper(root, diameter)
        return diameter[0]

    def Diameter_Naive(self, root):
        """
        Naive approach: Calculate height at each node
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return 0
        left_height = self.Height(root.left)
        right_height = self.Height(root.right)
        diameter_through_root = left_height + right_height + 1
        left_diameter = self.Diameter_Naive(root.left)
        right_diameter = self.Diameter_Naive(root.right)
        return max(diameter_through_root, left_diameter, right_diameter)

    def Height_Helper(self, root, diameter):
        if not root:
            return 0
        left_height = self.Height_Helper(root.left, diameter)
        right_height = self.Height_Helper(root.right, diameter)
        diameter[0] = max(diameter[0], left_height + right_height + 1)
        return 1 + max(left_height, right_height)

    def Height(self, root):
        if not root:
            return 0
        return 1 + max(self.Height(root.left), self.Height(root.right))


def Test_Diameter_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Optimized:", solution.Diameter_Optimized(root1))
    print("Test 1 - Naive:", solution.Diameter_Naive(root1))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Optimized:", solution.Diameter_Optimized(root2))
    print("Test 2 - Naive:", solution.Diameter_Naive(root2))
    
    vals3 = [1, 2, -1, 3, -1, 4]
    root3 = Build_Tree(vals3)
    print("Test 3 - Optimized:", solution.Diameter_Optimized(root3))
    print("Test 3 - Naive:", solution.Diameter_Naive(root3))


if __name__ == "__main__":
    Test_Diameter_Of_Tree()
