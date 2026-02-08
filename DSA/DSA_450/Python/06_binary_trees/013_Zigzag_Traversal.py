"""
Problem: Zigzag Traversal
URL: https://practice.geeksforgeeks.org/problems/zigzag-tree-traversal/1

Problem Statement:
Given a Binary Tree. Find the Zig-Zag Level Order Traversal of the Binary Tree. Zig-Zag traversal means starting from level 0 for the root node, for all the even levels we print the node's value from left to right and for all the odd levels we print the node's value from right to left.

Sample Input/Output:
Input:
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 3 2 4 5 6 7
Explanation: Level 0: 1 (left to right), Level 1: 3 2 (right to left), Level 2: 4 5 6 7 (left to right)
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
    def Zigzag_Queue_Stack(self, root):
        """
        Queue + Stack approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        q = deque([root])
        s = []
        leftToRight = True
        while q:
            size = len(q)
            for _ in range(size):
                node = q.popleft()
                if leftToRight:
                    result.append(node.val)
                else:
                    s.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            while s:
                result.append(s.pop())
            leftToRight = not leftToRight
        return result

    def Zigzag_Two_Stacks(self, root):
        """
        Two Stacks approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        s1 = [root]
        s2 = []
        while s1 or s2:
            while s1:
                node = s1.pop()
                result.append(node.val)
                if node.left:
                    s2.append(node.left)
                if node.right:
                    s2.append(node.right)
            while s2:
                node = s2.pop()
                result.append(node.val)
                if node.right:
                    s1.append(node.right)
                if node.left:
                    s1.append(node.left)
        return result

    def Zigzag_Deque(self, root):
        """
        Deque approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not root:
            return result
        dq = deque([root])
        leftToRight = True
        while dq:
            size = len(dq)
            for _ in range(size):
                if leftToRight:
                    node = dq.popleft()
                    result.append(node.val)
                    if node.left:
                        dq.append(node.left)
                    if node.right:
                        dq.append(node.right)
                else:
                    node = dq.pop()
                    result.append(node.val)
                    if node.right:
                        dq.appendleft(node.right)
                    if node.left:
                        dq.appendleft(node.left)
            leftToRight = not leftToRight
        return result


def Test_Zigzag_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Queue+Stack:", " ".join(map(str, solution.Zigzag_Queue_Stack(root1))))
    
    print("Test 1 - Two Stacks:", " ".join(map(str, solution.Zigzag_Two_Stacks(root1))))
    
    print("Test 1 - Deque:", " ".join(map(str, solution.Zigzag_Deque(root1))))


if __name__ == "__main__":
    Test_Zigzag_Traversal()
