"""
Problem: Check Whether BST Contains Dead End
URL: https://practice.geeksforgeeks.org/problems/check-whether-bst-contains-dead-end/1

Problem Statement:
Check if BST contains a dead end (leaf where no new node can be inserted).

Sample Input/Output:
Input: root = [8, 5, 9, 2, 7, null, null, null, null, null, 3]
Output: true
Explanation: Node 3 is a dead end (leaf with value 3, parent is 2, cannot insert 1 or 4).
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
    def Store_Nodes(self, root, all_nodes, leaf_nodes):
        if root is None:
            return
        all_nodes.add(root.val)
        if root.left is None and root.right is None:
            leaf_nodes.add(root.val)
        self.Store_Nodes(root.left, all_nodes, leaf_nodes)
        self.Store_Nodes(root.right, all_nodes, leaf_nodes)

    def Contains_Dead_End_Hash(self, root):
        """
        Hash set approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        all_nodes, leaf_nodes = set(), set()
        self.Store_Nodes(root, all_nodes, leaf_nodes)
        for leaf in leaf_nodes:
            if ((leaf == 1 or (leaf - 1) not in all_nodes) and
                    (leaf + 1) not in all_nodes):
                return True
        return False

    def Contains_Dead_End_Range(self, root, min_val, max_val):
        """
        Range-based recursion approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if root is None:
            return False
        if root.left is None and root.right is None:
            if (min_val == max_val or
                    (min_val == -sys.maxsize - 1 and max_val == root.val - 1) or
                    (max_val == sys.maxsize and min_val == root.val + 1) or
                    (min_val == root.val + 1 and max_val == root.val - 1)):
                return True
        left_dead = self.Contains_Dead_End_Range(root.left, min_val, root.val - 1)
        right_dead = self.Contains_Dead_End_Range(root.right, root.val + 1, max_val)
        return left_dead or right_dead


def Test_Dead_End_In_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 8)
    root = TreeNode.Insert_BST(root, 5)
    root = TreeNode.Insert_BST(root, 9)
    root = TreeNode.Insert_BST(root, 2)
    root = TreeNode.Insert_BST(root, 7)
    root = TreeNode.Insert_BST(root, 3)
    print("Dead End (Hash):", solution.Contains_Dead_End_Hash(root))
    print("Dead End (Range):", solution.Contains_Dead_End_Range(root, -sys.maxsize - 1, sys.maxsize))


if __name__ == "__main__":
    Test_Dead_End_In_BST()
