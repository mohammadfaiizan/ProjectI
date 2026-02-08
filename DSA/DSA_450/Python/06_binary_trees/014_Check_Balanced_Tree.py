"""
Problem: Check Balanced Tree
URL: https://practice.geeksforgeeks.org/problems/check-for-balanced-tree/1

Problem Statement:
Given a binary tree, find if it is height balanced or not. A tree is height balanced if difference between heights of left and right subtrees is not more than one for all nodes of tree.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \
   4   5

Output: true
Explanation: Height difference at each node is at most 1.

Input:
        1
      /
     2
    /
   3

Output: false
Explanation: Height difference exceeds 1.
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


def Print_Inorder(root):
    if not root:
        return
    Print_Inorder(root.left)
    print(root.val, end=" ")
    Print_Inorder(root.right)


class Solution:
    def Is_Balanced_Optimized(self, root):
        """
        Optimized single pass approach
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        return self.Check_Balanced_Helper(root) != -1

    def Is_Balanced_Naive(self, root):
        """
        Naive height check approach
        Time Complexity: O(n^2)
        Space Complexity: O(h) where h is height
        """
        if not root:
            return True
        leftHeight = self.Height(root.left)
        rightHeight = self.Height(root.right)
        return abs(leftHeight - rightHeight) <= 1 and \
               self.Is_Balanced_Naive(root.left) and \
               self.Is_Balanced_Naive(root.right)

    def Check_Balanced_Helper(self, root):
        if not root:
            return 0
        leftHeight = self.Check_Balanced_Helper(root.left)
        if leftHeight == -1:
            return -1
        rightHeight = self.Check_Balanced_Helper(root.right)
        if rightHeight == -1:
            return -1
        if abs(leftHeight - rightHeight) > 1:
            return -1
        return 1 + max(leftHeight, rightHeight)

    def Height(self, root):
        if not root:
            return 0
        return 1 + max(self.Height(root.left), self.Height(root.right))


def Test_Check_Balanced_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5]
    root1 = Build_Tree(vals1)
    print("Test 1 - Optimized:", solution.Is_Balanced_Optimized(root1))
    print("Test 1 - Naive:", solution.Is_Balanced_Naive(root1))
    
    vals2 = [1, 2, -1, 3]
    root2 = Build_Tree(vals2)
    print("Test 2 - Optimized:", solution.Is_Balanced_Optimized(root2))
    print("Test 2 - Naive:", solution.Is_Balanced_Naive(root2))


if __name__ == "__main__":
    Test_Check_Balanced_Tree()
