"""
Problem: Check Sum Tree
URL: https://practice.geeksforgeeks.org/problems/sum-tree/1

Problem Statement:
Check if binary tree is a sum tree (each node = sum of left + right subtree).

Sample Input/Output:
Input: 
        26
      /    \
    10      3
   /  \    / \
  4    6  1   2

Output: true
Explanation: 26 = 10 + 3 + 13, 10 = 4 + 6, 3 = 1 + 2
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
    def Sum_Tree_Optimized(self, root, is_sum_tree):
        """
        Optimized single pass: Return sum and check condition simultaneously
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return 0
        if not root.left and not root.right:
            return root.val
        left_sum = self.Sum_Tree_Optimized(root.left, is_sum_tree)
        right_sum = self.Sum_Tree_Optimized(root.right, is_sum_tree)
        if root.val != left_sum + right_sum:
            is_sum_tree[0] = False
        return root.val + left_sum + right_sum

    def Is_Sum_Tree_Optimized(self, root):
        is_sum_tree = [True]
        self.Sum_Tree_Optimized(root, is_sum_tree)
        return is_sum_tree[0]

    def Get_Sum(self, root):
        if not root:
            return 0
        return root.val + self.Get_Sum(root.left) + self.Get_Sum(root.right)

    def Is_Sum_Tree_Naive(self, root):
        """
        Naive with separate sum function: Check each node separately
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        """
        if not root or (not root.left and not root.right):
            return True
        left_sum = self.Get_Sum(root.left)
        right_sum = self.Get_Sum(root.right)
        return (root.val == left_sum + right_sum) and \
               self.Is_Sum_Tree_Naive(root.left) and \
               self.Is_Sum_Tree_Naive(root.right)


def Test_Check_Sum_Tree():
    solution = Solution()
    
    vals1 = [26, 10, 3, 4, 6, 1, 2]
    root1 = Build_Tree(vals1)
    print("Tree 1 (optimized):", solution.Is_Sum_Tree_Optimized(root1))
    print("Tree 1 (naive):", solution.Is_Sum_Tree_Naive(root1))
    
    vals2 = [10, 4, 6]
    root2 = Build_Tree(vals2)
    print("Tree 2 (optimized):", solution.Is_Sum_Tree_Optimized(root2))
    print("Tree 2 (naive):", solution.Is_Sum_Tree_Naive(root2))
    
    vals3 = [10, 3, 5]
    root3 = Build_Tree(vals3)
    print("Tree 3 (optimized):", solution.Is_Sum_Tree_Optimized(root3))
    print("Tree 3 (naive):", solution.Is_Sum_Tree_Naive(root3))


if __name__ == "__main__":
    Test_Check_Sum_Tree()
