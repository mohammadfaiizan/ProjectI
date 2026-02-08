"""
Problem: Min Swaps BT To BST
URL: https://www.geeksforgeeks.org/minimum-swap-required-convert-binary-tree-binary-search-tree/

Problem Statement:
Find minimum swaps required to convert a binary tree to BST. Get inorder, then find min swaps to sort.

Sample Input/Output:
Input: 
        5
      /   \
     6     7
    / \   / \
   8   9 10  11

Output: 3
Explanation: Inorder: [8, 6, 9, 5, 10, 7, 11]. Need 3 swaps to sort.
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
    def Get_Inorder(self, root, inorder):
        if not root:
            return
        self.Get_Inorder(root.left, inorder)
        inorder.append(root.val)
        self.Get_Inorder(root.right, inorder)

    def Min_Swaps_Cycle_Detection(self, arr):
        """
        Inorder traversal + min swaps using cycle detection
        Time Complexity: O(n log n) for sorting
        Space Complexity: O(n) for storing pairs and visited array
        """
        n = len(arr)
        pairs = [(arr[i], i) for i in range(n)]
        pairs.sort()
        visited = [False] * n
        swaps = 0
        for i in range(n):
            if visited[i] or pairs[i][1] == i:
                continue
            cycle_size = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = pairs[j][1]
                cycle_size += 1
            if cycle_size > 0:
                swaps += (cycle_size - 1)
        return swaps

    def Min_Swaps_BT_To_BST(self, root):
        inorder = []
        self.Get_Inorder(root, inorder)
        return self.Min_Swaps_Cycle_Detection(inorder)


def Test_Min_Swaps_BT_To_BST():
    solution = Solution()
    
    vals1 = [5, 6, 7, 8, 9, 10, 11]
    root1 = Build_Tree(vals1)
    print("Inorder:", end=" ")
    Print_Inorder(root1)
    print()
    swaps1 = solution.Min_Swaps_BT_To_BST(root1)
    print("Minimum swaps needed:", swaps1)
    
    vals2 = [1, 2, 3]
    root2 = Build_Tree(vals2)
    print("Inorder:", end=" ")
    Print_Inorder(root2)
    print()
    swaps2 = solution.Min_Swaps_BT_To_BST(root2)
    print("Minimum swaps needed:", swaps2)


if __name__ == "__main__":
    Test_Min_Swaps_BT_To_BST()
