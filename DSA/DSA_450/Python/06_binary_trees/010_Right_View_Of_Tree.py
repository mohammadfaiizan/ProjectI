"""
Problem: Right View Of Tree
URL: https://practice.geeksforgeeks.org/problems/right-view-of-binary-tree/1

Problem Statement:
Given a Binary Tree, print Right view of it. Right view of a Binary Tree is set of nodes visible when tree is visited from Right side.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
      /
     8

Output: 1 3 7 8
Explanation: Right view shows the rightmost node at each level.
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def Build_Tree(vals):
    if not vals or vals[0] == -1:
        return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] != -1:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] != -1:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root


def Print_Tree(root):
    if not root:
        return
    q = deque([root])
    result = []
    while q:
        node = q.popleft()
        result.append(str(node.val))
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    print(" ".join(result))


def Print_Inorder(root):
    if not root:
        return
    Print_Inorder(root.left)
    print(root.val, end=" ")
    Print_Inorder(root.right)


class Solution:
    def Right_View_Recursive(self, root):
        """
        Recursive approach with level tracking
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        result = []
        self.Right_View_Helper(root, 0, result)
        return result

    def Right_View_BFS(self, root):
        """
        Queue BFS approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        q = deque([root])
        while q:
            size = len(q)
            for i in range(size):
                node = q.popleft()
                if i == size - 1:
                    result.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return result

    def Right_View_Helper(self, root, level, result):
        if not root:
            return
        if level == len(result):
            result.append(root.val)
        self.Right_View_Helper(root.right, level + 1, result)
        self.Right_View_Helper(root.left, level + 1, result)


def Test_Right_View_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7, -1, -1, 8]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Right_View_Recursive(root1))))
    
    print("Test 1 - BFS:", " ".join(map(str, solution.Right_View_BFS(root1))))


if __name__ == "__main__":
    Test_Right_View_Of_Tree()
