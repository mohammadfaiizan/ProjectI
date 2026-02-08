"""
Problem: Maximum Sum of Nodes with No Two Adjacent
URL: https://www.geeksforgeeks.org/maximum-sum-nodes-binary-tree-no-two-adjacent/

Problem Statement:
Find the maximum sum of nodes in a binary tree such that no two selected nodes are adjacent (parent-child relationship).

Sample Input/Output:
Input: [1, 2, 3]
Output: 5
Explanation: Select nodes with values 2 and 3 (sum = 5), avoiding adjacent nodes.
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
    def Max_Sum_No_Adjacent_Pair(self, root):
        """
        Pair-based recursion (include/exclude)
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        if not root:
            return (0, 0)
        left = self.Max_Sum_No_Adjacent_Pair(root.left)
        right = self.Max_Sum_No_Adjacent_Pair(root.right)
        include = root.val + left[1] + right[1]
        exclude = max(left[0], left[1]) + max(right[0], right[1])
        return (include, exclude)
    
    def Max_Sum_No_Adjacent_Recursion(self, root):
        result = self.Max_Sum_No_Adjacent_Pair(root)
        return max(result[0], result[1])
    
    def Max_Sum_No_Adjacent_Memoization(self, root, memo):
        """
        Memoization with map
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return 0
        if root in memo:
            return memo[root]
        include = root.val
        if root.left:
            include += self.Max_Sum_No_Adjacent_Memoization(root.left.left, memo)
            include += self.Max_Sum_No_Adjacent_Memoization(root.left.right, memo)
        if root.right:
            include += self.Max_Sum_No_Adjacent_Memoization(root.right.left, memo)
            include += self.Max_Sum_No_Adjacent_Memoization(root.right.right, memo)
        exclude = self.Max_Sum_No_Adjacent_Memoization(root.left, memo) + \
                  self.Max_Sum_No_Adjacent_Memoization(root.right, memo)
        memo[root] = max(include, exclude)
        return memo[root]
    
    def Find_Max_Sum_No_Adjacent(self, root):
        return self.Max_Sum_No_Adjacent_Recursion(root)


def Test_Max_Sum_No_Adjacent():
    solution = Solution()
    
    vals1 = [1, 2, 3]
    root1 = Build_Tree(vals1)
    print("Test 1:", solution.Find_Max_Sum_No_Adjacent(root1))
    
    vals2 = [10, 1, 2, 3, 4, 5, 6]
    root2 = Build_Tree(vals2)
    print("Test 2:", solution.Find_Max_Sum_No_Adjacent(root2))
    
    vals3 = [1, 2, 3, 1, 3, 5]
    root3 = Build_Tree(vals3)
    print("Test 3:", solution.Find_Max_Sum_No_Adjacent(root3))


if __name__ == "__main__":
    Test_Max_Sum_No_Adjacent()
