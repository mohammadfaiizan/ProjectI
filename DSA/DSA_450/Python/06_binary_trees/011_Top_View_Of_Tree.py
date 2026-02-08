"""
Problem: Top View Of Tree
URL: https://practice.geeksforgeeks.org/problems/top-view-of-binary-tree/1

Problem Statement:
Given a binary tree, print the top view of it. Top view means when you look the tree from the top, the nodes you will see will be called the top view of the tree.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 2 1 3 7
Explanation: Top view shows nodes at each horizontal distance from root.
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
    def Top_View_BFS(self, root):
        """
        BFS with horizontal distance and map
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        m = {}
        q = deque([(root, 0)])
        while q:
            node, hd = q.popleft()
            if hd not in m:
                m[hd] = node.val
            if node.left:
                q.append((node.left, hd - 1))
            if node.right:
                q.append((node.right, hd + 1))
        for key in sorted(m.keys()):
            result.append(m[key])
        return result

    def Top_View_Recursive(self, root):
        """
        Recursive with map
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        m = {}
        self.Top_View_Helper(root, 0, 0, m)
        result = []
        for key in sorted(m.keys()):
            result.append(m[key][1])
        return result

    def Top_View_Helper(self, root, hd, level, m):
        if not root:
            return
        if hd not in m or level < m[hd][0]:
            m[hd] = (level, root.val)
        self.Top_View_Helper(root.left, hd - 1, level + 1, m)
        self.Top_View_Helper(root.right, hd + 1, level + 1, m)


def Test_Top_View_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - BFS:", " ".join(map(str, solution.Top_View_BFS(root1))))
    
    print("Test 1 - Recursive:", " ".join(map(str, solution.Top_View_Recursive(root1))))


if __name__ == "__main__":
    Test_Top_View_Of_Tree()
