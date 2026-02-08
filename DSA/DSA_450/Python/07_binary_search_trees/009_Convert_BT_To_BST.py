"""
Problem: Convert Binary Tree to BST
URL: https://practice.geeksforgeeks.org/problems/binary-tree-to-bst/1

Problem Statement:
Given a Binary Tree, convert it to Binary Search Tree in such a way that keeps the original structure of Binary Tree intact. Store inorder traversal, sort it, then assign back using inorder traversal.

Sample Input/Output:
Input: BT with root 10, left 2, right 7, left of 2 is 8, right of 2 is 4
Output: BST with same structure but values rearranged: root 8, left 4, right 10, left of 4 is 2, right of 4 is 7
Explanation: Inorder of BT: [8,2,4,10,7], sorted: [2,4,7,8,10], reassigned maintaining structure
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
    def Store_Inorder(self, root, inorder_list):
        if root is None:
            return
        self.Store_Inorder(root.left, inorder_list)
        inorder_list.append(root.val)
        self.Store_Inorder(root.right, inorder_list)

    def Assign_Inorder(self, root, inorder_list, index_ref):
        if root is None:
            return
        self.Assign_Inorder(root.left, inorder_list, index_ref)
        root.val = inorder_list[index_ref[0]]
        index_ref[0] += 1
        self.Assign_Inorder(root.right, inorder_list, index_ref)

    def Convert_BT_To_BST(self, root):
        """
        Approach: Store inorder, sort, assign back maintaining structure
        Time Complexity: O(n log n) for sorting
        Space Complexity: O(n) for storing inorder list
        """
        if root is None:
            return root
        inorder_list = []
        self.Store_Inorder(root, inorder_list)
        inorder_list.sort()
        index_ref = [0]
        self.Assign_Inorder(root, inorder_list, index_ref)
        return root


def Test_Convert_BT_To_BST():
    solution = Solution()
    root = TreeNode(10)
    root.left = TreeNode(2)
    root.right = TreeNode(7)
    root.left.left = TreeNode(8)
    root.left.right = TreeNode(4)

    print("Binary Tree Inorder (before conversion):", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    root = solution.Convert_BT_To_BST(root)

    print("BST Inorder (after conversion):", end=" ")
    TreeNode.Print_Inorder(root)
    print()


if __name__ == "__main__":
    Test_Convert_BT_To_BST()
