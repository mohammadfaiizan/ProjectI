"""
Problem: Merge Two Balanced Binary Search Trees
URL: https://www.geeksforgeeks.org/merge-two-balanced-binary-search-trees/

Problem Statement:
Given two Balanced Binary Search Trees (BSTs), merge them into a single balanced BST. Get inorder traversal of both BSTs, merge the two sorted arrays, then construct balanced BST from merged sorted array.

Sample Input/Output:
Input: BST1: [1,2,3], BST2: [4,5,6]
Output: Balanced BST: root 3, left subtree [1,2], right subtree [4,5,6]
Explanation: Merged sorted array [1,2,3,4,5,6], then build balanced BST from middle
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

    def Merge_Sorted_Arrays(self, array1, array2):
        merged_array = []
        i, j = 0, 0
        while i < len(array1) and j < len(array2):
            if array1[i] < array2[j]:
                merged_array.append(array1[i])
                i += 1
            else:
                merged_array.append(array2[j])
                j += 1
        while i < len(array1):
            merged_array.append(array1[i])
            i += 1
        while j < len(array2):
            merged_array.append(array2[j])
            j += 1
        return merged_array

    def Build_Balanced_BST_From_Sorted_Array(self, sorted_array, start, end):
        if start > end:
            return None
        mid = start + (end - start) // 2
        root = TreeNode(sorted_array[mid])
        root.left = self.Build_Balanced_BST_From_Sorted_Array(sorted_array, start, mid - 1)
        root.right = self.Build_Balanced_BST_From_Sorted_Array(sorted_array, mid + 1, end)
        return root

    def Merge_Two_BST(self, root1, root2):
        """
        Approach: Inorder + merge sorted arrays + sorted array to BST
        Time Complexity: O(m+n) where m and n are sizes of two BSTs
        Space Complexity: O(m+n) for storing inorder lists and merged array
        """
        inorder_list1 = []
        inorder_list2 = []
        self.Store_Inorder(root1, inorder_list1)
        self.Store_Inorder(root2, inorder_list2)
        merged_array = self.Merge_Sorted_Arrays(inorder_list1, inorder_list2)
        return self.Build_Balanced_BST_From_Sorted_Array(merged_array, 0, len(merged_array) - 1)


def Test_Merge_Two_BST():
    solution = Solution()
    bst1 = None
    bst1 = TreeNode.Insert_BST(bst1, 1)
    bst1 = TreeNode.Insert_BST(bst1, 2)
    bst1 = TreeNode.Insert_BST(bst1, 3)

    bst2 = None
    bst2 = TreeNode.Insert_BST(bst2, 4)
    bst2 = TreeNode.Insert_BST(bst2, 5)
    bst2 = TreeNode.Insert_BST(bst2, 6)

    print("BST1 Inorder:", end=" ")
    TreeNode.Print_Inorder(bst1)
    print()

    print("BST2 Inorder:", end=" ")
    TreeNode.Print_Inorder(bst2)
    print()

    merged_bst = solution.Merge_Two_BST(bst1, bst2)

    print("Merged Balanced BST Inorder:", end=" ")
    TreeNode.Print_Inorder(merged_bst)
    print()


if __name__ == "__main__":
    Test_Merge_Two_BST()
