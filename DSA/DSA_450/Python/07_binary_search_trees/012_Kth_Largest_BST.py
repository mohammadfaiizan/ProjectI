"""
Problem: Kth Largest Element in BST
URL: https://practice.geeksforgeeks.org/problems/kth-largest-element-in-bst/1

Problem Statement:
Find kth largest element in BST.

Sample Input/Output:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 4
Explanation: The 3rd largest element is 4.
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
    def Kth_Largest_Reverse_Inorder(self, root, k_ref):
        """
        Reverse inorder traversal approach
        Time Complexity: O(h + k)
        Space Complexity: O(h)
        """
        if root is None:
            return -1
        right = self.Kth_Largest_Reverse_Inorder(root.right, k_ref)
        if right != -1:
            return right
        k_ref[0] -= 1
        if k_ref[0] == 0:
            return root.val
        return self.Kth_Largest_Reverse_Inorder(root.left, k_ref)

    def Kth_Largest_Morris(self, root, k):
        """
        Morris reverse inorder traversal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        curr = root
        count = 0
        result = -1
        while curr is not None:
            if curr.right is None:
                count += 1
                if count == k:
                    result = curr.val
                curr = curr.left
            else:
                prev = curr.right
                while prev.left is not None and prev.left != curr:
                    prev = prev.left
                if prev.left is None:
                    prev.left = curr
                    curr = curr.right
                else:
                    prev.left = None
                    count += 1
                    if count == k:
                        result = curr.val
                    curr = curr.left
        return result


def Test_Kth_Largest_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 5)
    root = TreeNode.Insert_BST(root, 3)
    root = TreeNode.Insert_BST(root, 6)
    root = TreeNode.Insert_BST(root, 2)
    root = TreeNode.Insert_BST(root, 4)
    root = TreeNode.Insert_BST(root, 1)
    k = 3
    k_ref = [k]
    print("Kth Largest (Reverse Inorder):", solution.Kth_Largest_Reverse_Inorder(root, k_ref))
    print("Kth Largest (Morris):", solution.Kth_Largest_Morris(root, k))


if __name__ == "__main__":
    Test_Kth_Largest_BST()
