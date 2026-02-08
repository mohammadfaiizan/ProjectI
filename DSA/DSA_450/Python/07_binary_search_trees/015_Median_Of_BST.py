"""
Problem: Find Median of BST
URL: https://www.geeksforgeeks.org/find-median-bst-time-o1-space/

Problem Statement:
Find median of BST in O(n) time and O(1) space.

Sample Input/Output:
Input: root = [6,3,8,1,5,7,9]
Output: 6
Explanation: Inorder: [1,3,5,6,7,8,9], median is 6.
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
    def Count_Nodes(self, root):
        count = 0
        curr = root
        while curr is not None:
            if curr.left is None:
                count += 1
                curr = curr.right
            else:
                prev = curr.left
                while prev.right is not None and prev.right != curr:
                    prev = prev.right
                if prev.right is None:
                    prev.right = curr
                    curr = curr.left
                else:
                    prev.right = None
                    count += 1
                    curr = curr.right
        return count

    def Find_Kth_Node_Morris(self, root, k):
        curr = root
        count = 0
        while curr is not None:
            if curr.left is None:
                count += 1
                if count == k:
                    return curr.val
                curr = curr.right
            else:
                prev = curr.left
                while prev.right is not None and prev.right != curr:
                    prev = prev.right
                if prev.right is None:
                    prev.right = curr
                    curr = curr.left
                else:
                    prev.right = None
                    count += 1
                    if count == k:
                        return curr.val
                    curr = curr.right
        return -1

    def Median_Morris(self, root):
        """
        Morris traversal with node counting approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = self.Count_Nodes(root)
        if n % 2 == 1:
            return self.Find_Kth_Node_Morris(root, (n + 1) // 2)
        else:
            first = self.Find_Kth_Node_Morris(root, n // 2)
            second = self.Find_Kth_Node_Morris(root, n // 2 + 1)
            return (first + second) / 2.0

    def Inorder_To_Array(self, root, arr):
        if root is None:
            return
        self.Inorder_To_Array(root.left, arr)
        arr.append(root.val)
        self.Inorder_To_Array(root.right, arr)

    def Median_Array(self, root):
        """
        Inorder to array approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        arr = []
        self.Inorder_To_Array(root, arr)
        n = len(arr)
        if n % 2 == 1:
            return arr[n // 2]
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0


def Test_Median_Of_BST():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 6)
    root = TreeNode.Insert_BST(root, 3)
    root = TreeNode.Insert_BST(root, 8)
    root = TreeNode.Insert_BST(root, 1)
    root = TreeNode.Insert_BST(root, 5)
    root = TreeNode.Insert_BST(root, 7)
    root = TreeNode.Insert_BST(root, 9)
    print("Median (Morris):", solution.Median_Morris(root))
    print("Median (Array):", solution.Median_Array(root))


if __name__ == "__main__":
    Test_Median_Of_BST()
