"""
Problem: Count BST Nodes That Lie in a Given Range
URL: https://practice.geeksforgeeks.org/problems/count-bst-nodes-that-lie-in-a-given-range/1

Problem Statement:
Count BST nodes that lie in a given range [low, high].

Sample Input/Output:
Input: root = [10,5,50,1,null,40,100], low = 5, high = 45
Output: 3
Explanation: Nodes in range are 5, 10, 40.
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
    def Count_Nodes_Pruned(self, root, low, high):
        """
        Pruned traversal approach
        Time Complexity: O(h + k) where k is nodes in range
        Space Complexity: O(h)
        """
        if root is None:
            return 0
        if root.val >= low and root.val <= high:
            return (1 + self.Count_Nodes_Pruned(root.left, low, high) +
                    self.Count_Nodes_Pruned(root.right, low, high))
        elif root.val < low:
            return self.Count_Nodes_Pruned(root.right, low, high)
        else:
            return self.Count_Nodes_Pruned(root.left, low, high)

    def Count_Nodes_Full(self, root, low, high):
        """
        Full traversal approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if root is None:
            return 0
        count = 0
        if root.val >= low and root.val <= high:
            count += 1
        count += self.Count_Nodes_Full(root.left, low, high)
        count += self.Count_Nodes_Full(root.right, low, high)
        return count


def Test_Count_Nodes_In_Range():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 10)
    root = TreeNode.Insert_BST(root, 5)
    root = TreeNode.Insert_BST(root, 50)
    root = TreeNode.Insert_BST(root, 1)
    root = TreeNode.Insert_BST(root, 40)
    root = TreeNode.Insert_BST(root, 100)
    low, high = 5, 45
    print("Count Nodes (Pruned):", solution.Count_Nodes_Pruned(root, low, high))
    print("Count Nodes (Full):", solution.Count_Nodes_Full(root, low, high))


if __name__ == "__main__":
    Test_Count_Nodes_In_Range()
