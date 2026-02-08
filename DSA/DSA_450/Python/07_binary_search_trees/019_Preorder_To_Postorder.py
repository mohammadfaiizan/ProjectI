"""
Problem: Preorder to Postorder
URL: https://practice.geeksforgeeks.org/problems/preorder-to-postorder4423/1

Problem Statement:
Given preorder of BST, find postorder without constructing tree.

Sample Input/Output:
Input: pre[] = {40, 30, 35, 80, 100}
Output: post[] = {35, 30, 100, 80, 40}
Explanation: BST preorder converted to postorder.
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
    def Preorder_To_Postorder_Range(self, pre, index_ref, min_val, max_val, post):
        if index_ref[0] >= len(pre):
            return
        if pre[index_ref[0]] < min_val or pre[index_ref[0]] > max_val:
            return
        val = pre[index_ref[0]]
        index_ref[0] += 1
        self.Preorder_To_Postorder_Range(pre, index_ref, min_val, val, post)
        self.Preorder_To_Postorder_Range(pre, index_ref, val, max_val, post)
        post.append(val)

    def Preorder_To_Postorder_Range_Based(self, pre):
        """
        Range-based recursion approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        post = []
        index_ref = [0]
        self.Preorder_To_Postorder_Range(pre, index_ref, -sys.maxsize - 1, sys.maxsize, post)
        return post

    def Postorder_Traversal(self, root, post):
        if root is None:
            return
        self.Postorder_Traversal(root.left, post)
        self.Postorder_Traversal(root.right, post)
        post.append(root.val)

    def Preorder_To_Postorder_Construct_BST(self, pre):
        """
        Construct BST then postorder approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        root = None
        for val in pre:
            root = TreeNode.Insert_BST(root, val)
        post = []
        self.Postorder_Traversal(root, post)
        return post


def Test_Preorder_To_Postorder():
    solution = Solution()
    pre = [40, 30, 35, 80, 100]
    post1 = solution.Preorder_To_Postorder_Range_Based(pre)
    post2 = solution.Preorder_To_Postorder_Construct_BST(pre)
    print("Postorder (Range):", end=" ")
    for val in post1:
        print(val, end=" ")
    print()
    print("Postorder (Construct BST):", end=" ")
    for val in post2:
        print(val, end=" ")
    print()


if __name__ == "__main__":
    Test_Preorder_To_Postorder()
