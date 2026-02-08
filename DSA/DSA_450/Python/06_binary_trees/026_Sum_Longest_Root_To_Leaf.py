"""
Problem: Sum Longest Root To Leaf
URL: https://practice.geeksforgeeks.org/problems/sum-of-the-longest-bloodline-of-a-tree/1

Problem Statement:
Find sum of nodes on the longest path from root to leaf.

Sample Input/Output:
Input: 
        4
      /   \
     2     5
    / \   / \
   7   1 2   3
      /
     6

Output: 13
Explanation: Longest path is 4->2->1->6 with sum 13.
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
    def Sum_Longest_Path_Recursive(self, root, level, sum_val, max_level, max_sum):
        """
        Recursive with level and sum tracking: Track level and sum simultaneously
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return
        sum_val += root.val
        if not root.left and not root.right:
            if level > max_level[0]:
                max_level[0] = level
                max_sum[0] = sum_val
            elif level == max_level[0]:
                max_sum[0] = max(max_sum[0], sum_val)
            return
        self.Sum_Longest_Path_Recursive(root.left, level + 1, sum_val, max_level, max_sum)
        self.Sum_Longest_Path_Recursive(root.right, level + 1, sum_val, max_level, max_sum)

    def Sum_Of_Longest_Bloodline_Recursive(self, root):
        max_level = [0]
        max_sum = [0]
        self.Sum_Longest_Path_Recursive(root, 1, 0, max_level, max_sum)
        return max_sum[0]

    def Sum_Of_Longest_Bloodline_BFS(self, root):
        """
        BFS with level tracking: Use level order traversal
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        """
        if not root:
            return 0
        q = deque([(root, 1, root.val)])
        max_level = 0
        max_sum = 0
        while q:
            node, level, sum_val = q.popleft()
            if not node.left and not node.right:
                if level > max_level:
                    max_level = level
                    max_sum = sum_val
                elif level == max_level:
                    max_sum = max(max_sum, sum_val)
            if node.left:
                q.append((node.left, level + 1, sum_val + node.left.val))
            if node.right:
                q.append((node.right, level + 1, sum_val + node.right.val))
        return max_sum


def Test_Sum_Longest_Root_To_Leaf():
    solution = Solution()
    
    vals1 = [4, 2, 5, 7, 1, 2, 3, -1, -1, 6]
    root1 = Build_Tree(vals1)
    print("Tree 1 (recursive):", solution.Sum_Of_Longest_Bloodline_Recursive(root1))
    print("Tree 1 (BFS):", solution.Sum_Of_Longest_Bloodline_BFS(root1))
    
    vals2 = [1, 2, 3]
    root2 = Build_Tree(vals2)
    print("Tree 2 (recursive):", solution.Sum_Of_Longest_Bloodline_Recursive(root2))
    print("Tree 2 (BFS):", solution.Sum_Of_Longest_Bloodline_BFS(root2))


if __name__ == "__main__":
    Test_Sum_Longest_Root_To_Leaf()
