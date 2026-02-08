"""
Problem: Convert BST to Min Heap
URL: https://www.geeksforgeeks.org/convert-bst-min-heap/

Problem Statement:
Given a BST (with property that each node has either 0 or 2 children), convert it to a Min Heap such that all values in left subtree < all values in right subtree.

Sample Input/Output:
Input: BST structure
Output: Min Heap structure
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def __init__(self):
        self.inorder = []
        self.index = 0

    def Convert_BST_Heap_Inorder_Preorder(self, root):
        """
        Inorder-Preorder Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        self.inorder = []
        self.index = 0
        self.InorderTraversal(root)
        self.PreorderFill(root)
        return root

    def InorderTraversal(self, root):
        if not root:
            return
        self.InorderTraversal(root.left)
        self.inorder.append(root.val)
        self.InorderTraversal(root.right)

    def PreorderFill(self, root):
        if not root:
            return
        root.val = self.inorder[self.index]
        self.index += 1
        self.PreorderFill(root.left)
        self.PreorderFill(root.right)


def PrintLevelOrder(root):
    if not root:
        return
    q = deque([root])
    while q:
        node = q.popleft()
        print(node.val, end=" ")
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    print()


def Test_Convert_BST_Heap():
    solution = Solution()

    root1 = TreeNode(4)
    root1.left = TreeNode(2)
    root1.right = TreeNode(6)
    root1.left.left = TreeNode(1)
    root1.left.right = TreeNode(3)
    root1.right.left = TreeNode(5)
    root1.right.right = TreeNode(7)

    print("Original BST (Level Order):", end=" ")
    PrintLevelOrder(root1)

    result1 = solution.Convert_BST_Heap_Inorder_Preorder(root1)
    print("Converted Min Heap (Level Order):", end=" ")
    PrintLevelOrder(result1)

    root2 = TreeNode(8)
    root2.left = TreeNode(4)
    root2.right = TreeNode(12)
    root2.left.left = TreeNode(2)
    root2.left.right = TreeNode(6)
    root2.right.left = TreeNode(10)
    root2.right.right = TreeNode(14)

    print("Original BST 2 (Level Order):", end=" ")
    PrintLevelOrder(root2)

    solution2 = Solution()
    result2 = solution2.Convert_BST_Heap_Inorder_Preorder(root2)
    print("Converted Min Heap 2 (Level Order):", end=" ")
    PrintLevelOrder(result2)


if __name__ == "__main__":
    Test_Convert_BST_Heap()
