"""
Problem: Check if Binary Tree is BST
URL: https://practice.geeksforgeeks.org/problems/check-for-bst/1

Problem Statement:
Given a binary tree, check whether it is a valid Binary Search Tree (BST). A valid BST is defined as follows: The left subtree of a node contains only nodes with keys less than the node's key. The right subtree of a node contains only nodes with keys greater than the node's key. Both the left and right subtrees must also be binary search trees.

Sample Input/Output:
Input: Root = [2,1,3]
Output: true
Explanation: All nodes satisfy BST property
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
    def Is_Valid_BST_Min_Max_Range(self, root, min_value, max_value):
        if root is None:
            return True
        if root.val <= min_value or root.val >= max_value:
            return False
        return (self.Is_Valid_BST_Min_Max_Range(root.left, min_value, root.val) and
                self.Is_Valid_BST_Min_Max_Range(root.right, root.val, max_value))

    def Validate_BST_Min_Max_Range(self, root):
        """
        Using min-max range approach: each node must be in valid range
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height for recursion stack
        """
        return self.Is_Valid_BST_Min_Max_Range(root, -sys.maxsize - 1, sys.maxsize)

    def Inorder_Traversal_Check(self, root, inorder_list):
        if root is None:
            return
        self.Inorder_Traversal_Check(root.left, inorder_list)
        inorder_list.append(root.val)
        self.Inorder_Traversal_Check(root.right, inorder_list)

    def Validate_BST_Inorder_Traversal(self, root):
        """
        Using inorder traversal: BST inorder should be strictly increasing
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) for recursion stack + O(n) for storing inorder
        """
        inorder_list = []
        self.Inorder_Traversal_Check(root, inorder_list)
        for i in range(1, len(inorder_list)):
            if inorder_list[i] <= inorder_list[i - 1]:
                return False
        return True


def Test_Validate_BST():
    solution = Solution()
    valid_bst = None
    valid_bst = TreeNode.Insert_BST(valid_bst, 50)
    valid_bst = TreeNode.Insert_BST(valid_bst, 30)
    valid_bst = TreeNode.Insert_BST(valid_bst, 70)
    valid_bst = TreeNode.Insert_BST(valid_bst, 20)
    valid_bst = TreeNode.Insert_BST(valid_bst, 40)

    print("Valid BST check (Min-Max):", solution.Validate_BST_Min_Max_Range(valid_bst))
    print("Valid BST check (Inorder):", solution.Validate_BST_Inorder_Traversal(valid_bst))

    invalid_bst = TreeNode(10)
    invalid_bst.left = TreeNode(5)
    invalid_bst.right = TreeNode(15)
    invalid_bst.right.left = TreeNode(6)
    invalid_bst.right.right = TreeNode(20)

    print("Invalid BST check (Min-Max):", solution.Validate_BST_Min_Max_Range(invalid_bst))
    print("Invalid BST check (Inorder):", solution.Validate_BST_Inorder_Traversal(invalid_bst))


if __name__ == "__main__":
    Test_Validate_BST()
