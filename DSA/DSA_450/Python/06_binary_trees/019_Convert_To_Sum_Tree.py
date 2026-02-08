"""
Problem: Convert To Sum Tree
URL: https://practice.geeksforgeeks.org/problems/transform-to-sum-tree/1

Problem Statement:
Convert a binary tree such that each node contains sum of left and right subtree values. Leaf nodes become 0.

Sample Input/Output:
Input: 
        10
      /    \
     -2     6
    /  \   / \
   8   -4 7   5

Output:
        20
      /    \
     4     12
    /  \   / \
   0   0  0   0

Explanation: Each node is replaced with sum of its left and right subtree values.
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
    def Convert_To_Sum_Tree_Postorder(self, root):
        """
        Post-order recursion: Process children first, then update current node
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return 0
        left_sum = self.Convert_To_Sum_Tree_Postorder(root.left)
        right_sum = self.Convert_To_Sum_Tree_Postorder(root.right)
        old_data = root.val
        root.val = left_sum + right_sum
        return old_data + root.val


def Test_Convert_To_Sum_Tree():
    solution = Solution()
    
    vals1 = [10, -2, 6, 8, -4, 7, 5]
    root1 = Build_Tree(vals1)
    print("Before conversion:", end=" ")
    Print_Inorder(root1)
    print()
    solution.Convert_To_Sum_Tree_Postorder(root1)
    print("After conversion:", end=" ")
    Print_Inorder(root1)
    print()
    
    vals2 = [1, 2, 3]
    root2 = Build_Tree(vals2)
    print("Before conversion:", end=" ")
    Print_Inorder(root2)
    print()
    solution.Convert_To_Sum_Tree_Postorder(root2)
    print("After conversion:", end=" ")
    Print_Inorder(root2)
    print()


if __name__ == "__main__":
    Test_Convert_To_Sum_Tree()
