"""
Problem: Delete Node in BST
URL: https://leetcode.com/problems/delete-node-in-a-bst/

Problem Statement:
Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST. Handle three cases: node is a leaf, node has one child, node has two children (replace with inorder successor).

Sample Input/Output:
Input: Root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
Explanation: Node 3 is deleted and replaced with its inorder successor 4
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
    def Find_Min(self, root):
        while root.left is not None:
            root = root.left
        return root

    def Delete_Node_Recursive(self, root, key):
        """
        Recursive deletion handling three cases: leaf, one child, two children
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None:
            return root
        if key < root.val:
            root.left = self.Delete_Node_Recursive(root.left, key)
        elif key > root.val:
            root.right = self.Delete_Node_Recursive(root.right, key)
        else:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            inorder_successor = self.Find_Min(root.right)
            root.val = inorder_successor.val
            root.right = self.Delete_Node_Recursive(root.right, inorder_successor.val)
        return root


def Test_Delete_Node_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 50)
    root = TreeNode.Insert_BST(root, 30)
    root = TreeNode.Insert_BST(root, 70)
    root = TreeNode.Insert_BST(root, 20)
    root = TreeNode.Insert_BST(root, 40)
    root = TreeNode.Insert_BST(root, 60)
    root = TreeNode.Insert_BST(root, 80)

    print("Before deletion:", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    root = solution.Delete_Node_Recursive(root, 20)
    print("After deleting 20 (leaf):", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    root = solution.Delete_Node_Recursive(root, 30)
    print("After deleting 30 (one child):", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    root = solution.Delete_Node_Recursive(root, 50)
    print("After deleting 50 (two children):", end=" ")
    TreeNode.Print_Inorder(root)
    print()


if __name__ == "__main__":
    Test_Delete_Node_BST()
