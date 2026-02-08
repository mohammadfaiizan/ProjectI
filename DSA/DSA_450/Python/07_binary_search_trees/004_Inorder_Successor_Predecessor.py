"""
Problem: Find Inorder Predecessor and Successor
URL: https://practice.geeksforgeeks.org/problems/predecessor-and-successor/1

Problem Statement:
Given a BST and a key, find the inorder predecessor and successor of the given key in the BST. If the key does not exist in BST, return the two values between which this key will lie.

Sample Input/Output:
Input: BST with root 50, left 30, right 70. Key = 65
Output: Predecessor = 50, Successor = 70
Explanation: 65 is not present, so predecessor is largest value less than 65, successor is smallest value greater than 65
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
    def Find_Predecessor_Successor_BST_Property(self, root, key):
        """
        Using BST property to find predecessor and successor
        Time Complexity: O(h) where h is height
        Space Complexity: O(1) constant space
        """
        predecessor = None
        successor = None
        while root is not None:
            if root.val == key:
                if root.left is not None:
                    temp = root.left
                    while temp.right is not None:
                        temp = temp.right
                    predecessor = temp
                if root.right is not None:
                    temp = root.right
                    while temp.left is not None:
                        temp = temp.left
                    successor = temp
                return predecessor, successor
            elif root.val > key:
                successor = root
                root = root.left
            else:
                predecessor = root
                root = root.right
        return predecessor, successor

    def Inorder_Traversal(self, root, inorder_list):
        if root is None:
            return
        self.Inorder_Traversal(root.left, inorder_list)
        inorder_list.append(root.val)
        self.Inorder_Traversal(root.right, inorder_list)

    def Find_Predecessor_Successor_Inorder(self, root, key):
        """
        Using inorder traversal to get sorted list, then find predecessor and successor
        Time Complexity: O(n) for traversal
        Space Complexity: O(n) for storing inorder list
        """
        inorder_list = []
        self.Inorder_Traversal(root, inorder_list)
        predecessor = -1
        successor = -1
        for i in range(len(inorder_list)):
            if inorder_list[i] == key:
                if i > 0:
                    predecessor = inorder_list[i - 1]
                if i < len(inorder_list) - 1:
                    successor = inorder_list[i + 1]
                break
            elif inorder_list[i] < key:
                predecessor = inorder_list[i]
            elif inorder_list[i] > key and successor == -1:
                successor = inorder_list[i]
                break
        return predecessor, successor


def Test_Inorder_Successor_Predecessor():
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

    predecessor, successor = solution.Find_Predecessor_Successor_BST_Property(root, 65)
    print("Key 65 - Predecessor:", predecessor.val if predecessor else -1, ", Successor:", successor.val if successor else -1)

    predecessor, successor = solution.Find_Predecessor_Successor_Inorder(root, 40)
    print("Key 40 - Predecessor:", predecessor, ", Successor:", successor)


if __name__ == "__main__":
    Test_Inorder_Successor_Predecessor()
