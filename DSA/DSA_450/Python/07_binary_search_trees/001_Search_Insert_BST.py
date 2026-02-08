"""
Problem: Search and Insert a Node in BST
URL: https://practice.geeksforgeeks.org/problems/insert-a-node-in-a-bst/1

Problem Statement:
Given a BST and a key K. If K is not present in the BST, Insert a new Node with a value equal to K into the BST. If K is already present in the BST, don't modify the BST.

Sample Input/Output:
Input: BST with root 2, left 1, right 3. Key = 4
Output: BST with 4 inserted as right child of 3
Explanation: 4 is not present, so insert it maintaining BST property
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
    def Search_Recursive(self, root, key):
        """
        Recursive search using BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None or root.val == key:
            return root
        if key < root.val:
            return self.Search_Recursive(root.left, key)
        return self.Search_Recursive(root.right, key)

    def Insert_Recursive(self, root, key):
        """
        Recursive insert maintaining BST property
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion stack
        """
        if root is None:
            return TreeNode(key)
        if key < root.val:
            root.left = self.Insert_Recursive(root.left, key)
        elif key > root.val:
            root.right = self.Insert_Recursive(root.right, key)
        return root

    def Insert_Iterative(self, root, key):
        """
        Iterative insert using while loop
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        """
        new_node = TreeNode(key)
        if root is None:
            return new_node
        current = root
        parent = None
        while current is not None:
            parent = current
            if key < current.val:
                current = current.left
            elif key > current.val:
                current = current.right
            else:
                return root
        if key < parent.val:
            parent.left = new_node
        else:
            parent.right = new_node
        return root


def Test_Search_Insert_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 50)
    root = TreeNode.Insert_BST(root, 30)
    root = TreeNode.Insert_BST(root, 70)
    root = TreeNode.Insert_BST(root, 20)
    root = TreeNode.Insert_BST(root, 40)
    root = TreeNode.Insert_BST(root, 60)
    root = TreeNode.Insert_BST(root, 80)

    found = solution.Search_Recursive(root, 40)
    print("Search 40:", "Found" if found else "Not Found")

    root = solution.Insert_Recursive(root, 35)
    root = solution.Insert_Iterative(root, 90)

    print("Inorder after inserts:", end=" ")
    TreeNode.Print_Inorder(root)
    print()


if __name__ == "__main__":
    Test_Search_Insert_BST()
