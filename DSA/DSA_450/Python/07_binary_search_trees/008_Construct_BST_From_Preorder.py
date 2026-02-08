"""
Problem: Construct BST from Given Preorder Traversal
URL: https://www.geeksforgeeks.org/construct-bst-from-given-preorder-traversa/

Problem Statement:
Given preorder traversal of a binary search tree, construct the BST. Preorder traversal is Root, Left, Right.

Sample Input/Output:
Input: [10, 5, 1, 7, 40, 50]
Output: BST with root 10, left subtree [5,1,7], right subtree [40,50]
Explanation: First element is root, then elements less than root form left subtree, greater form right subtree
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
        elif key > root.val:
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
    def Construct_BST_Array_Splitting(self, preorder, start, end):
        if start > end:
            return None
        root = TreeNode(preorder[start])
        right_start = start + 1
        while right_start <= end and preorder[right_start] < preorder[start]:
            right_start += 1
        root.left = self.Construct_BST_Array_Splitting(preorder, start + 1, right_start - 1)
        root.right = self.Construct_BST_Array_Splitting(preorder, right_start, end)
        return root

    def Build_BST_Array_Splitting(self, preorder):
        """
        Array splitting approach: find split point for left and right subtrees
        Time Complexity: O(n^2) worst case when tree is skewed
        Space Complexity: O(h) where h is height for recursion stack
        """
        return self.Construct_BST_Array_Splitting(preorder, 0, len(preorder) - 1)

    def Construct_BST_Range_Based(self, preorder, index_ref, min_value, max_value):
        if index_ref[0] >= len(preorder):
            return None
        value = preorder[index_ref[0]]
        if value < min_value or value > max_value:
            return None
        root = TreeNode(value)
        index_ref[0] += 1
        root.left = self.Construct_BST_Range_Based(preorder, index_ref, min_value, value - 1)
        root.right = self.Construct_BST_Range_Based(preorder, index_ref, value + 1, max_value)
        return root

    def Build_BST_Range_Based(self, preorder):
        """
        Range-based approach: use min-max range to validate each node
        Time Complexity: O(n) single pass through array
        Space Complexity: O(h) where h is height for recursion stack
        """
        index_ref = [0]
        return self.Construct_BST_Range_Based(preorder, index_ref, -sys.maxsize - 1, sys.maxsize)


def Test_Construct_BST_From_Preorder():
    solution = Solution()
    preorder = [10, 5, 1, 7, 40, 50]

    root_array_split = solution.Build_BST_Array_Splitting(preorder)
    print("BST from Array Splitting (Inorder):", end=" ")
    TreeNode.Print_Inorder(root_array_split)
    print()

    root_range_based = solution.Build_BST_Range_Based(preorder)
    print("BST from Range Based (Inorder):", end=" ")
    TreeNode.Print_Inorder(root_range_based)
    print()


if __name__ == "__main__":
    Test_Construct_BST_From_Preorder()
