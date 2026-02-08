"""
Problem: Lowest Common Ancestor in BST
URL: https://practice.geeksforgeeks.org/problems/lowest-common-ancestor-in-a-bst/1

Problem Statement:
Given a Binary Search Tree (with all values unique) and two node values. Find the Lowest Common Ancestors (LCA) of the two nodes in the BST. LCA is the node which is the ancestor of both nodes.

Sample Input/Output:
Input: BST with root 20, left 8, right 22, left of 8 is 4, right of 8 is 12. Nodes: 4 and 12
Output: 8
Explanation: LCA of 4 and 12 is 8, which is their common ancestor
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
    def LCA_Iterative(self, root, node1, node2):
        """
        Iterative approach using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        """
        while root is not None:
            if root.val > node1 and root.val > node2:
                root = root.left
            elif root.val < node1 and root.val < node2:
                root = root.right
            else:
                break
        return root

    def LCA_Recursive(self, root, node1, node2):
        """
        Recursive approach using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None:
            return None
        if root.val > node1 and root.val > node2:
            return self.LCA_Recursive(root.left, node1, node2)
        if root.val < node1 and root.val < node2:
            return self.LCA_Recursive(root.right, node1, node2)
        return root


def Test_LCA_In_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 20)
    root = TreeNode.Insert_BST(root, 8)
    root = TreeNode.Insert_BST(root, 22)
    root = TreeNode.Insert_BST(root, 4)
    root = TreeNode.Insert_BST(root, 12)
    root = TreeNode.Insert_BST(root, 10)
    root = TreeNode.Insert_BST(root, 14)

    print("BST Inorder:", end=" ")
    TreeNode.Print_Inorder(root)
    print()

    lca_iter = solution.LCA_Iterative(root, 4, 12)
    print("LCA of 4 and 12 (Iterative):", lca_iter.val if lca_iter else -1)

    lca_rec = solution.LCA_Recursive(root, 10, 14)
    print("LCA of 10 and 14 (Recursive):", lca_rec.val if lca_rec else -1)

    lca_mixed = solution.LCA_Iterative(root, 4, 22)
    print("LCA of 4 and 22 (Iterative):", lca_mixed.val if lca_mixed else -1)


if __name__ == "__main__":
    Test_LCA_In_BST()
