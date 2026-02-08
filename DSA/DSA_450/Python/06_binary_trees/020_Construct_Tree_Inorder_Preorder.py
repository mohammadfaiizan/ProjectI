"""
Problem: Construct Tree Inorder Preorder
URL: https://practice.geeksforgeeks.org/problems/construct-tree-1/1

Problem Statement:
Construct binary tree from inorder and preorder traversals.

Sample Input/Output:
Input: 
Inorder: [9, 3, 15, 20, 7]
Preorder: [3, 9, 20, 15, 7]

Output:
        3
      /   \
     9    20
         /  \
       15    7

Explanation: Root is first in preorder, then left and right subtrees are constructed recursively.
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
    def Build_Tree_Linear_Search(self, inorder, preorder, pre_idx, in_start, in_end):
        """
        Recursive with linear search: Find root in inorder array using linear search
        Time Complexity: O(n^2) worst case
        Space Complexity: O(h) where h is height of tree
        """
        if in_start > in_end or pre_idx[0] >= len(preorder):
            return None
        root = TreeNode(preorder[pre_idx[0]])
        pre_idx[0] += 1
        in_idx = in_start
        for i in range(in_start, in_end + 1):
            if inorder[i] == root.val:
                in_idx = i
                break
        root.left = self.Build_Tree_Linear_Search(inorder, preorder, pre_idx, in_start, in_idx - 1)
        root.right = self.Build_Tree_Linear_Search(inorder, preorder, pre_idx, in_idx + 1, in_end)
        return root

    def Build_Tree_Hashmap(self, inorder, preorder, pre_idx, in_start, in_end, in_map):
        """
        Recursive with hashmap: Use hashmap to find root index in O(1)
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for hashmap
        """
        if in_start > in_end or pre_idx[0] >= len(preorder):
            return None
        root = TreeNode(preorder[pre_idx[0]])
        pre_idx[0] += 1
        in_idx = in_map[root.val]
        root.left = self.Build_Tree_Hashmap(inorder, preorder, pre_idx, in_start, in_idx - 1, in_map)
        root.right = self.Build_Tree_Hashmap(inorder, preorder, pre_idx, in_idx + 1, in_end, in_map)
        return root

    def Construct_Tree_Linear_Search(self, inorder, preorder):
        pre_idx = [0]
        return self.Build_Tree_Linear_Search(inorder, preorder, pre_idx, 0, len(inorder) - 1)

    def Construct_Tree_Hashmap(self, inorder, preorder):
        in_map = {val: idx for idx, val in enumerate(inorder)}
        pre_idx = [0]
        return self.Build_Tree_Hashmap(inorder, preorder, pre_idx, 0, len(inorder) - 1, in_map)


def Test_Construct_Tree_Inorder_Preorder():
    solution = Solution()
    
    inorder1 = [9, 3, 15, 20, 7]
    preorder1 = [3, 9, 20, 15, 7]
    root1 = solution.Construct_Tree_Hashmap(inorder1, preorder1)
    print("Constructed tree (hashmap):", end=" ")
    Print_Inorder(root1)
    print()
    
    inorder2 = [4, 2, 5, 1, 6, 3, 7]
    preorder2 = [1, 2, 4, 5, 3, 6, 7]
    root2 = solution.Construct_Tree_Linear_Search(inorder2, preorder2)
    print("Constructed tree (linear search):", end=" ")
    Print_Inorder(root2)
    print()


if __name__ == "__main__":
    Test_Construct_Tree_Inorder_Preorder()
