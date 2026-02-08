"""
Problem: Mirror Of Tree
URL: https://www.geeksforgeeks.org/create-a-mirror-tree-from-the-given-binary-tree/

Problem Statement:
Given a binary tree, create its mirror tree. Mirror of a tree is obtained by swapping left and right children of all nodes.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output (Mirror):
        1
      /   \
     3     2
    / \   / \
   7   6 5   4

Explanation: Left and right children of all nodes are swapped.
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
    def Mirror_In_Place(self, root):
        """
        In-place recursive swap: Swap left and right children recursively
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.Mirror_In_Place(root.left)
        self.Mirror_In_Place(root.right)
        return root

    def Mirror_Separate_Tree(self, root):
        """
        Create separate mirror tree: Build new tree with swapped children
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for new tree
        """
        if not root:
            return None
        mirror = TreeNode(root.val)
        mirror.left = self.Mirror_Separate_Tree(root.right)
        mirror.right = self.Mirror_Separate_Tree(root.left)
        return mirror


def Test_Mirror_Of_Tree():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Original Inorder:", end=" ")
    Print_Inorder(root1)
    print()
    
    root1_copy = Build_Tree(vals1)
    solution.Mirror_In_Place(root1_copy)
    print("Mirror In-Place Inorder:", end=" ")
    Print_Inorder(root1_copy)
    print()
    
    mirror1 = solution.Mirror_Separate_Tree(root1)
    print("Mirror Separate Tree Inorder:", end=" ")
    Print_Inorder(mirror1)
    print()
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("\nOriginal Inorder:", end=" ")
    Print_Inorder(root2)
    print()
    
    root2_copy = Build_Tree(vals2)
    solution.Mirror_In_Place(root2_copy)
    print("Mirror In-Place Inorder:", end=" ")
    Print_Inorder(root2_copy)
    print()


if __name__ == "__main__":
    Test_Mirror_Of_Tree()
