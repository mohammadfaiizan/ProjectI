"""
Problem: Reverse Level Order Traversal
URL: https://practice.geeksforgeeks.org/problems/reverse-level-order-traversal/1

Problem Statement:
Given a binary tree, print its reverse level order traversal. Reverse level order means visiting nodes level by level from bottom to top, left to right within each level.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 5 6 7 2 3 1
Explanation: Nodes are printed level by level from bottom to top.
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


class Solution:
    def Reverse_Level_Order_Recursive(self, root):
        """
        Recursive approach: Print levels in reverse order
        Time Complexity: O(n^2) worst case for skewed tree
        Space Complexity: O(n) for recursion stack
        """
        result = []
        if not root:
            return result
        h = self.Height(root)
        for i in range(h, 0, -1):
            self.Print_Level_Helper(root, i, result)
        return result

    def Reverse_Level_Order_Queue_Stack(self, root):
        """
        Queue + Stack approach: Use queue for BFS and stack to reverse
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue and stack
        """
        result = []
        if not root:
            return result
        q = deque([root])
        st = []
        while q:
            node = q.popleft()
            st.append(node.val)
            if node.right:
                q.append(node.right)
            if node.left:
                q.append(node.left)
        while st:
            result.append(st.pop())
        return result

    def Height(self, root):
        if not root:
            return 0
        return 1 + max(self.Height(root.left), self.Height(root.right))

    def Print_Level_Helper(self, root, level, result):
        if not root:
            return
        if level == 1:
            result.append(root.val)
            return
        self.Print_Level_Helper(root.left, level - 1, result)
        self.Print_Level_Helper(root.right, level - 1, result)


def Test_Reverse_Level_Order_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Queue+Stack Approach:", " ".join(map(str, solution.Reverse_Level_Order_Queue_Stack(root1))))
    
    print("Test 1 - Recursive Approach:", " ".join(map(str, solution.Reverse_Level_Order_Recursive(root1))))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Queue+Stack Approach:", " ".join(map(str, solution.Reverse_Level_Order_Queue_Stack(root2))))


if __name__ == "__main__":
    Test_Reverse_Level_Order_Traversal()
