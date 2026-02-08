"""
Problem: Count Pairs from Two BSTs Whose Sum Equals X
URL: https://practice.geeksforgeeks.org/problems/brothers-from-different-root/1

Problem Statement:
Count pairs from two BSTs whose sum equals X.

Sample Input/Output:
Input: root1 = [5,3,7,2,4,6,8], root2 = [10,6,15,3,8,11,18], X = 16
Output: 3
Explanation: Pairs are (5,11), (6,10), (8,8)
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
    def Search_BST(self, root, key):
        if root is None:
            return False
        if root.val == key:
            return True
        if key < root.val:
            return self.Search_BST(root.left, key)
        return self.Search_BST(root.right, key)

    def Count_Pairs_BST_Search(self, root1, root2, X):
        """
        Inorder traversal + BST search approach
        Time Complexity: O(n log m)
        Space Complexity: O(h)
        """
        if root1 is None:
            return 0
        count = self.Count_Pairs_BST_Search(root1.left, root2, X)
        if self.Search_BST(root2, X - root1.val):
            count += 1
        count += self.Count_Pairs_BST_Search(root1.right, root2, X)
        return count

    def Inorder_To_Array(self, root, arr):
        if root is None:
            return
        self.Inorder_To_Array(root.left, arr)
        arr.append(root.val)
        self.Inorder_To_Array(root.right, arr)

    def Count_Pairs_Two_Pointer(self, root1, root2, X):
        """
        Inorder both trees + two pointer approach
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        arr1, arr2 = [], []
        self.Inorder_To_Array(root1, arr1)
        self.Inorder_To_Array(root2, arr2)
        i, j = 0, len(arr2) - 1
        count = 0
        while i < len(arr1) and j >= 0:
            sum_val = arr1[i] + arr2[j]
            if sum_val == X:
                count += 1
                i += 1
                j -= 1
            elif sum_val < X:
                i += 1
            else:
                j -= 1
        return count


def Test_Count_Pairs_Sum_X():
    solution = Solution()
    root1 = None
    root1 = TreeNode.Insert_BST(root1, 5)
    root1 = TreeNode.Insert_BST(root1, 3)
    root1 = TreeNode.Insert_BST(root1, 7)
    root1 = TreeNode.Insert_BST(root1, 2)
    root1 = TreeNode.Insert_BST(root1, 4)
    root1 = TreeNode.Insert_BST(root1, 6)
    root1 = TreeNode.Insert_BST(root1, 8)
    root2 = None
    root2 = TreeNode.Insert_BST(root2, 10)
    root2 = TreeNode.Insert_BST(root2, 6)
    root2 = TreeNode.Insert_BST(root2, 15)
    root2 = TreeNode.Insert_BST(root2, 3)
    root2 = TreeNode.Insert_BST(root2, 8)
    root2 = TreeNode.Insert_BST(root2, 11)
    root2 = TreeNode.Insert_BST(root2, 18)
    X = 16
    print("Count Pairs (BST Search):", solution.Count_Pairs_BST_Search(root1, root2, X))
    print("Count Pairs (Two Pointer):", solution.Count_Pairs_Two_Pointer(root1, root2, X))


if __name__ == "__main__":
    Test_Count_Pairs_Sum_X()
