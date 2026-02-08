"""
Problem: Diagonal Traversal
URL: https://www.geeksforgeeks.org/diagonal-traversal-of-binary-tree/

Problem Statement:
Given a Binary Tree, print the diagonal traversal of the binary tree. Diagonal traversal means traversing nodes diagonally from top-left to bottom-right.

Sample Input/Output:
Input:
        8
      /   \
     3    10
    / \     \
   1   6    14
      / \   /
     4   7 13

Output: 8 10 14 3 6 7 13 1 4
Explanation: Nodes are printed diagonally from top-left to bottom-right.
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
    def Diagonal_Traversal_Map(self, root):
        """
        Map-based recursion approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        m = {}
        self.Diagonal_Traversal_Helper(root, 0, m)
        result = []
        for key in sorted(m.keys()):
            result.extend(m[key])
        return result

    def Diagonal_Traversal_Queue(self, root):
        """
        Queue-based BFS approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        q = deque([root])
        while q:
            node = q.popleft()
            while node:
                result.append(node.val)
                if node.left:
                    q.append(node.left)
                node = node.right
        return result

    def Diagonal_Traversal_Helper(self, root, diagonal, m):
        if not root:
            return
        if diagonal not in m:
            m[diagonal] = []
        m[diagonal].append(root.val)
        self.Diagonal_Traversal_Helper(root.left, diagonal + 1, m)
        self.Diagonal_Traversal_Helper(root.right, diagonal, m)


def Test_Diagonal_Traversal():
    solution = Solution()
    
    vals1 = [8, 3, 10, 1, 6, -1, 14, -1, -1, 4, 7, 13]
    root1 = Build_Tree(vals1)
    print("Test 1 - Map:", " ".join(map(str, solution.Diagonal_Traversal_Map(root1))))
    
    print("Test 1 - Queue:", " ".join(map(str, solution.Diagonal_Traversal_Queue(root1))))


if __name__ == "__main__":
    Test_Diagonal_Traversal()
