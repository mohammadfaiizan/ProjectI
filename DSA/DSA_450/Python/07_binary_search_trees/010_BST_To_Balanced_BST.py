"""
Problem: Convert Normal BST to Balanced BST
URL: https://www.geeksforgeeks.org/convert-normal-bst-balanced-bst/

Problem Statement:
Given a BST (Binary Search Tree) that may be unbalanced, convert it into a balanced BST that has minimum possible height. Store inorder traversal to get sorted array, then build balanced BST from sorted array.

Sample Input/Output:
Input: Skewed BST: 1->2->3->4->5->6->7 (all right children)
Output: Balanced BST with root 4, left subtree [1,2,3], right subtree [5,6,7]
Explanation: Inorder gives sorted array, then build balanced tree from middle element
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
    def Store_Inorder_To_Array(self, root, inorder_array):
        if root is None:
            return
        self.Store_Inorder_To_Array(root.left, inorder_array)
        inorder_array.append(root.val)
        self.Store_Inorder_To_Array(root.right, inorder_array)

    def Build_Balanced_BST_From_Sorted_Array(self, sorted_array, start, end):
        if start > end:
            return None
        mid = start + (end - start) // 2
        root = TreeNode(sorted_array[mid])
        root.left = self.Build_Balanced_BST_From_Sorted_Array(sorted_array, start, mid - 1)
        root.right = self.Build_Balanced_BST_From_Sorted_Array(sorted_array, mid + 1, end)
        return root

    def Convert_To_Balanced_BST(self, root):
        """
        Approach: Inorder to sorted array + build balanced BST from sorted array
        Time Complexity: O(n) for traversal and building
        Space Complexity: O(n) for storing inorder array
        """
        inorder_array = []
        self.Store_Inorder_To_Array(root, inorder_array)
        return self.Build_Balanced_BST_From_Sorted_Array(inorder_array, 0, len(inorder_array) - 1)


def Test_BST_To_Balanced_BST():
    solution = Solution()
    skewed_bst = None
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 1)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 2)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 3)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 4)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 5)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 6)
    skewed_bst = TreeNode.Insert_BST(skewed_bst, 7)

    print("Skewed BST Inorder:", end=" ")
    TreeNode.Print_Inorder(skewed_bst)
    print()

    balanced_bst = solution.Convert_To_Balanced_BST(skewed_bst)

    print("Balanced BST Inorder:", end=" ")
    TreeNode.Print_Inorder(balanced_bst)
    print()


if __name__ == "__main__":
    Test_BST_To_Balanced_BST()
