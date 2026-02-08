"""
Problem: Boundary Traversal
URL: https://practice.geeksforgeeks.org/problems/boundary-traversal-of-binary-tree/1

Problem Statement:
Given a Binary Tree, find its Boundary Traversal. The traversal should be in the following order: Left boundary nodes, Leaf nodes, Right boundary nodes in reverse order.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
      /
     8

Output: 1 2 4 8 5 6 7 3
Explanation: Left boundary: 1 2 4, Leaves: 8 5 6 7, Right boundary (reverse): 3
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
    def Boundary_Traversal_Recursive(self, root):
        """
        Recursive approach (left boundary + leaves + right boundary)
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        result = []
        if not root:
            return result
        if root.left or root.right:
            result.append(root.val)
        self.Left_Boundary(root.left, result)
        self.Leaves(root, result)
        self.Right_Boundary(root.right, result)
        return result

    def Boundary_Traversal_Iterative(self, root):
        """
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        if root.left or root.right:
            result.append(root.val)
        node = root.left
        while node and (node.left or node.right):
            result.append(node.val)
            node = node.left if node.left else node.right
        s = [root]
        while s:
            curr = s.pop()
            if curr.right:
                s.append(curr.right)
            if curr.left:
                s.append(curr.left)
            if not curr.left and not curr.right and curr != root:
                result.append(curr.val)
        rightBoundary = []
        node = root.right
        while node and (node.left or node.right):
            rightBoundary.append(node.val)
            node = node.right if node.right else node.left
        rightBoundary.reverse()
        result.extend(rightBoundary)
        return result

    def Left_Boundary(self, root, result):
        if not root or (not root.left and not root.right):
            return
        result.append(root.val)
        if root.left:
            self.Left_Boundary(root.left, result)
        else:
            self.Left_Boundary(root.right, result)

    def Right_Boundary(self, root, result):
        if not root or (not root.left and not root.right):
            return
        if root.right:
            self.Right_Boundary(root.right, result)
        else:
            self.Right_Boundary(root.left, result)
        result.append(root.val)

    def Leaves(self, root, result):
        if not root:
            return
        if not root.left and not root.right:
            result.append(root.val)
            return
        self.Leaves(root.left, result)
        self.Leaves(root.right, result)


def Test_Boundary_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7, -1, -1, 8]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Boundary_Traversal_Recursive(root1))))
    
    print("Test 1 - Iterative:", " ".join(map(str, solution.Boundary_Traversal_Iterative(root1))))


if __name__ == "__main__":
    Test_Boundary_Traversal()
