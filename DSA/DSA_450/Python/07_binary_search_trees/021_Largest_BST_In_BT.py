"""
Problem: Largest BST in Binary Tree
URL: https://practice.geeksforgeeks.org/problems/largest-bst/1

Problem Statement:
Find size of largest BST subtree in a binary tree.

Sample Input/Output:
Input: root = [10,5,15,1,8,null,7]
Output: 3
Explanation: Largest BST subtree has size 3 (rooted at node 5).
"""

import sys


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def Build_BST(keys):
        root = None
        for key in keys:
            root = TreeNode.Insert_BST(root, key)
        return root

    @staticmethod
    def Insert_BST(root, key):
        if root is None:
            return TreeNode(key)
        if key < root.val:
            root.left = TreeNode.Insert_BST(root.left, key)
        else:
            root.right = TreeNode.Insert_BST(root.right, key)
        return root

    @staticmethod
    def Print_Inorder(root):
        if root is None:
            return
        TreeNode.Print_Inorder(root.left)
        print(root.val, end=" ")
        TreeNode.Print_Inorder(root.right)


class Solution:
    class BSTInfo:
        def __init__(self, is_bst, size, min_val, max_val):
            self.is_bst = is_bst
            self.size = size
            self.min_val = min_val
            self.max_val = max_val

    def Largest_BST_Helper(self, root, max_size_ref):
        if root is None:
            return self.BSTInfo(True, 0, sys.maxsize, -sys.maxsize - 1)
        left_info = self.Largest_BST_Helper(root.left, max_size_ref)
        right_info = self.Largest_BST_Helper(root.right, max_size_ref)
        if (left_info.is_bst and right_info.is_bst and
                root.val > left_info.max_val and root.val < right_info.min_val):
            size = left_info.size + right_info.size + 1
            max_size_ref[0] = max(max_size_ref[0], size)
            min_val = root.val if left_info.size == 0 else left_info.min_val
            max_val = root.val if right_info.size == 0 else right_info.max_val
            return self.BSTInfo(True, size, min_val, max_val)
        return self.BSTInfo(False, 0, 0, 0)

    def Largest_BST_Size(self, root):
        """
        Bottom-up with min/max/size tracking approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        max_size_ref = [0]
        self.Largest_BST_Helper(root, max_size_ref)
        return max_size_ref[0]


def Test_Largest_BST_In_BT():
    solution = Solution()
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.left.left = TreeNode(1)
    root.left.right = TreeNode(8)
    root.right.right = TreeNode(7)
    print("Largest BST Size:", solution.Largest_BST_Size(root))


if __name__ == "__main__":
    Test_Largest_BST_In_BT()
