"""
Problem: Replace Every Element with the Least Greater Element on Its Right
URL: https://www.geeksforgeeks.org/replace-every-element-with-the-least-greater-element-on-its-right/

Problem Statement:
Replace every element with the least greater element on its right side. If none, use -1.

Sample Input/Output:
Input: [8, 58, 71, 18, 31, 32, 63, 92, 43, 3, 91, 93, 25, 80, 28]
Output: [18, 63, 80, 25, 32, 43, 80, 93, 80, 25, 93, -1, 28, -1, -1]
Explanation: For 8, least greater on right is 18.
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
    def Insert_BST_With_Successor(self, root, key, successor_ref):
        if root is None:
            return TreeNode(key)
        if key < root.val:
            successor_ref[0] = root
            root.left = self.Insert_BST_With_Successor(root.left, key, successor_ref)
        else:
            root.right = self.Insert_BST_With_Successor(root.right, key, successor_ref)
        return root

    def Replace_Least_Greater_BST(self, arr):
        """
        BST insertion from right with successor tracking approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        n = len(arr)
        result = [-1] * n
        root = None
        for i in range(n - 1, -1, -1):
            successor_ref = [None]
            root = self.Insert_BST_With_Successor(root, arr[i], successor_ref)
            if successor_ref[0] is not None:
                result[i] = successor_ref[0].val
        return result

    def Replace_Least_Greater_Brute(self, arr):
        """
        Brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        n = len(arr)
        result = [-1] * n
        for i in range(n):
            min_greater = sys.maxsize
            for j in range(i + 1, n):
                if arr[j] > arr[i] and arr[j] < min_greater:
                    min_greater = arr[j]
            if min_greater != sys.maxsize:
                result[i] = min_greater
        return result


def Test_Replace_With_Least_Greater_Right():
    solution = Solution()
    arr = [8, 58, 71, 18, 31, 32, 63, 92, 43, 3, 91, 93, 25, 80, 28]
    result1 = solution.Replace_Least_Greater_BST(arr)
    result2 = solution.Replace_Least_Greater_Brute(arr)
    print("Replace (BST):", end=" ")
    for val in result1:
        print(val, end=" ")
    print()
    print("Replace (Brute):", end=" ")
    for val in result2:
        print(val, end=" ")
    print()


if __name__ == "__main__":
    Test_Replace_With_Least_Greater_Right()
