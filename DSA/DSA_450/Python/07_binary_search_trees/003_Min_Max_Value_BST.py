"""
Problem: Find Minimum and Maximum Values in BST
URL: https://practice.geeksforgeeks.org/problems/minimum-element-in-bst/1

Problem Statement:
Given a Binary Search Tree, find the minimum and maximum values in the BST.

Sample Input/Output:
Input: BST with root 5, left 3, right 7, left of 3 is 2, right of 3 is 4
Output: Min = 2, Max = 7
Explanation: Leftmost node has minimum value, rightmost node has maximum value
"""


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
    def Min_Value_Iterative(self, root):
        """
        Iterative approach: traverse to leftmost node
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        """
        if root is None:
            return -1
        while root.left is not None:
            root = root.left
        return root.val

    def Max_Value_Iterative(self, root):
        """
        Iterative approach: traverse to rightmost node
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        """
        if root is None:
            return -1
        while root.right is not None:
            root = root.right
        return root.val

    def Min_Value_Recursive(self, root):
        """
        Recursive approach: base case and recurse left
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None:
            return -1
        if root.left is None:
            return root.val
        return self.Min_Value_Recursive(root.left)

    def Max_Value_Recursive(self, root):
        """
        Recursive approach: base case and recurse right
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None:
            return -1
        if root.right is None:
            return root.val
        return self.Max_Value_Recursive(root.right)


def Test_Min_Max_Value_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 50)
    root = TreeNode.Insert_BST(root, 30)
    root = TreeNode.Insert_BST(root, 70)
    root = TreeNode.Insert_BST(root, 20)
    root = TreeNode.Insert_BST(root, 40)
    root = TreeNode.Insert_BST(root, 60)
    root = TreeNode.Insert_BST(root, 80)

    print("BST Inorder:", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    print("Min (Iterative):", solution.Min_Value_Iterative(root))
    print("Max (Iterative):", solution.Max_Value_Iterative(root))
    print("Min (Recursive):", solution.Min_Value_Recursive(root))
    print("Max (Recursive):", solution.Max_Value_Recursive(root))


if __name__ == "__main__":
    Test_Min_Max_Value_BST()
