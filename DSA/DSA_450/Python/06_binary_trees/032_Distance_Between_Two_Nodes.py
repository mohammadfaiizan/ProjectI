"""
Problem: Minimum Distance Between Two Nodes
URL: https://practice.geeksforgeeks.org/problems/min-distance-between-two-given-nodes-of-a-binary-tree/1

Problem Statement:
Find the minimum distance between two nodes in a binary tree. Distance is the number of edges between them.
Formula: dist(a,b) = dist(root,a) + dist(root,b) - 2*dist(root,lca)

Sample Input/Output:
Input: Tree [1, 2, 3, 4, 5], nodes 4 and 5
Output: 2
Explanation: Path from 4 to 5: 4->2->5, distance = 2 edges.
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
    def Find_LCA(self, root, n1, n2):
        if not root:
            return None
        if root.val == n1 or root.val == n2:
            return root
        left_lca = self.Find_LCA(root.left, n1, n2)
        right_lca = self.Find_LCA(root.right, n1, n2)
        if left_lca and right_lca:
            return root
        return left_lca if left_lca else right_lca
    
    def Find_Level(self, root, target, level):
        if not root:
            return -1
        if root.val == target:
            return level
        left_level = self.Find_Level(root.left, target, level + 1)
        if left_level != -1:
            return left_level
        return self.Find_Level(root.right, target, level + 1)
    
    def Distance_LCA_Level(self, root, a, b):
        """
        LCA + level finding
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        lca = self.Find_LCA(root, a, b)
        if not lca:
            return -1
        dist_a = self.Find_Level(root, a, 0)
        dist_b = self.Find_Level(root, b, 0)
        dist_lca = self.Find_Level(root, lca.val, 0)
        return dist_a + dist_b - 2 * dist_lca
    
    def Distance_Single_Traversal(self, root, a, b, dist):
        """
        Single traversal
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if not root:
            return 0
        left = self.Distance_Single_Traversal(root.left, a, b, dist)
        right = self.Distance_Single_Traversal(root.right, a, b, dist)
        if root.val == a or root.val == b:
            if left or right:
                dist[0] = max(left, right)
                return 0
            return 1
        if left and right:
            dist[0] = left + right
            return 0
        if left or right:
            return max(left, right) + 1
        return 0
    
    def Find_Distance_Between_Nodes(self, root, a, b):
        dist = [0]
        self.Distance_Single_Traversal(root, a, b, dist)
        return dist[0]


def Test_Distance_Between_Two_Nodes():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5]
    root1 = Build_Tree(vals1)
    print("Test 1 - Distance between 4 and 5:", solution.Find_Distance_Between_Nodes(root1, 4, 5))
    
    print("Test 2 - Distance between 2 and 3:", solution.Find_Distance_Between_Nodes(root1, 2, 3))
    
    print("Test 3 - Distance between 4 and 3:", solution.Find_Distance_Between_Nodes(root1, 4, 3))


if __name__ == "__main__":
    Test_Distance_Between_Two_Nodes()
