"""
Problem: Postorder Traversal
URL: https://www.techiedelight.com/postorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform postorder traversal. Postorder traversal visits left subtree, right subtree, then root.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 5 2 6 7 3 1
Explanation: Left subtree -> Right subtree -> Root
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
    def Postorder_Recursive(self, root):
        """
        Recursive approach: Visit left, right, root
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        self.Postorder_Helper(root, result)
        return result

    def Postorder_Iterative(self, root):
        """
        Iterative two stacks: Use two stacks for postorder traversal
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for stacks
        """
        result = []
        if not root:
            return result
        st1 = [root]
        st2 = []
        while st1:
            node = st1.pop()
            st2.append(node)
            if node.left:
                st1.append(node.left)
            if node.right:
                st1.append(node.right)
        while st2:
            result.append(st2.pop().val)
        return result

    def Postorder_Helper(self, root, result):
        if not root:
            return
        self.Postorder_Helper(root.left, result)
        self.Postorder_Helper(root.right, result)
        result.append(root.val)


def Test_Postorder_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Postorder_Recursive(root1))))
    
    print("Test 1 - Iterative:", " ".join(map(str, solution.Postorder_Iterative(root1))))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Recursive:", " ".join(map(str, solution.Postorder_Recursive(root2))))
    
    print("Test 2 - Iterative:", " ".join(map(str, solution.Postorder_Iterative(root2))))


if __name__ == "__main__":
    Test_Postorder_Traversal()
