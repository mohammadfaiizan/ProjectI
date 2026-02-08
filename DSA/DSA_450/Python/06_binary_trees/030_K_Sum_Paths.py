"""
Problem: Print K Sum Paths in Binary Tree
URL: https://www.geeksforgeeks.org/print-k-sum-paths-binary-tree/

Problem Statement:
Print all paths in a binary tree whose sum equals k. A path can start and end at any node but must be downward.

Sample Input/Output:
Input: k=5, tree [1, 3, -1, 2, 1, 4, 5]
Output: [3 2], [3 1 1], [4 1], [1 3 1], [5]
Explanation: Multiple paths sum to 5.
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
    def K_Sum_Paths_Backtracking(self, root, k, path, result):
        """
        Recursion with path vector and backtracking
        Time Complexity: O(n^2)
        Space Complexity: O(h)
        """
        if not root:
            return
        path.append(root.val)
        sum_val = 0
        for i in range(len(path) - 1, -1, -1):
            sum_val += path[i]
            if sum_val == k:
                valid_path = path[i:]
                result.append(valid_path[:])
        self.K_Sum_Paths_Backtracking(root.left, k, path, result)
        self.K_Sum_Paths_Backtracking(root.right, k, path, result)
        path.pop()
    
    def K_Sum_Paths_Prefix_Sum(self, root, k, current_sum, prefix_map, path, result):
        """
        Prefix sum with hashmap
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return
        current_sum += root.val
        path.append(root.val)
        if current_sum == k:
            result.append(path[:])
        if (current_sum - k) in prefix_map:
            start_idx = prefix_map[current_sum - k]
            valid_path = path[start_idx + 1:]
            result.append(valid_path[:])
        prefix_map[current_sum] = len(path) - 1
        self.K_Sum_Paths_Prefix_Sum(root.left, k, current_sum, prefix_map, path, result)
        self.K_Sum_Paths_Prefix_Sum(root.right, k, current_sum, prefix_map, path, result)
        del prefix_map[current_sum]
        path.pop()
    
    def Find_K_Sum_Paths(self, root, k):
        result = []
        path = []
        self.K_Sum_Paths_Backtracking(root, k, path, result)
        return result
    
    def Find_K_Sum_Paths_Optimized(self, root, k):
        result = []
        path = []
        prefix_map = {}
        self.K_Sum_Paths_Prefix_Sum(root, k, 0, prefix_map, path, result)
        return result


def Test_K_Sum_Paths():
    solution = Solution()
    
    vals1 = [1, 3, -1, 2, 1, 4, 5]
    root1 = Build_Tree(vals1)
    paths1 = solution.Find_K_Sum_Paths(root1, 5)
    print("Test 1 - Paths with sum 5:")
    for path in paths1:
        print(" ".join(map(str, path)))
    
    vals2 = [1, 2, 3, 4, 5]
    root2 = Build_Tree(vals2)
    paths2 = solution.Find_K_Sum_Paths(root2, 3)
    print("Test 2 - Paths with sum 3:")
    for path in paths2:
        print(" ".join(map(str, path)))


if __name__ == "__main__":
    Test_K_Sum_Paths()
