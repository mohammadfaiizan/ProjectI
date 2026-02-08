"""
Problem: Find Largest Subtree Sum
URL: https://www.geeksforgeeks.org/find-largest-subtree-sum-tree/

Problem Statement:
Given a binary tree, find the largest subtree sum. The subtree sum of a node is the sum of all node values in the subtree rooted at that node.

Sample Input/Output:
Input: [1, 2, 3, 4, 5, -6, 2]
Output: 7
Explanation: Subtree rooted at node with value 2 has sum 2+4+5-6+2 = 7, which is the maximum.
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
    def Largest_Subtree_Sum_Postorder(self, root, max_sum):
        """
        Post-order recursion tracking max sum
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if not root:
            return 0
        left_sum = self.Largest_Subtree_Sum_Postorder(root.left, max_sum)
        right_sum = self.Largest_Subtree_Sum_Postorder(root.right, max_sum)
        subtree_sum = root.val + left_sum + right_sum
        max_sum[0] = max(max_sum[0], subtree_sum)
        return subtree_sum
    
    def Find_Largest_Subtree_Sum(self, root):
        max_sum = [float('-inf')]
        self.Largest_Subtree_Sum_Postorder(root, max_sum)
        return max_sum[0]


def Test_Largest_Subtree_Sum():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, -6, 2]
    root1 = Build_Tree(vals1)
    print("Test 1:", solution.Find_Largest_Subtree_Sum(root1))
    
    vals2 = [1, -2, 3, 4, 5, -6, 2]
    root2 = Build_Tree(vals2)
    print("Test 2:", solution.Find_Largest_Subtree_Sum(root2))
    
    vals3 = [-5, 2, 3]
    root3 = Build_Tree(vals3)
    print("Test 3:", solution.Find_Largest_Subtree_Sum(root3))


if __name__ == "__main__":
    Test_Largest_Subtree_Sum()
