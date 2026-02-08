"""
Problem: Left View Of Tree
URL: https://practice.geeksforgeeks.org/problems/left-view-of-binary-tree/1

Problem Statement:
Given a binary tree, print its left view. Left view of a tree is the set of nodes visible when the tree is viewed from the left side.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 2 4
Explanation: When viewed from left, nodes 1, 2, and 4 are visible.
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
    def Left_View_Recursive(self, root):
        """
        Recursive level tracking: Track max level reached
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        max_level = [0]
        self.Left_View_Helper(root, 1, max_level, result)
        return result

    def Left_View_Queue(self, root):
        """
        Queue BFS approach: First node of each level
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        """
        result = []
        if not root:
            return result
        q = deque([root])
        while q:
            size = len(q)
            result.append(q[0].val)
            for _ in range(size):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return result

    def Left_View_Helper(self, root, level, max_level, result):
        if not root:
            return
        if level > max_level[0]:
            result.append(root.val)
            max_level[0] = level
        self.Left_View_Helper(root.left, level + 1, max_level, result)
        self.Left_View_Helper(root.right, level + 1, max_level, result)


def Test_Left_View_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Left_View_Recursive(root1))))
    
    print("Test 1 - Queue:", " ".join(map(str, solution.Left_View_Queue(root1))))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Recursive:", " ".join(map(str, solution.Left_View_Recursive(root2))))
    
    print("Test 2 - Queue:", " ".join(map(str, solution.Left_View_Queue(root2))))
    
    vals3 = [1, 2, -1, 3]
    root3 = Build_Tree(vals3)
    print("Test 3 - Recursive:", " ".join(map(str, solution.Left_View_Recursive(root3))))
    
    print("Test 3 - Queue:", " ".join(map(str, solution.Left_View_Queue(root3))))


if __name__ == "__main__":
    Test_Left_View_Of_Tree()
