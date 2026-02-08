"""
Problem: Preorder Traversal
URL: https://www.techiedelight.com/preorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform preorder traversal. Preorder traversal visits root, left subtree, then right subtree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 1 2 4 5 3 6 7
Explanation: Root -> Left subtree -> Right subtree
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
    def Preorder_Recursive(self, root):
        """
        Recursive approach: Visit root, left, right
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        self.Preorder_Helper(root, result)
        return result

    def Preorder_Iterative(self, root):
        """
        Iterative with stack: Use stack to simulate recursion
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        if not root:
            return result
        st = [root]
        while st:
            node = st.pop()
            result.append(node.val)
            if node.right:
                st.append(node.right)
            if node.left:
                st.append(node.left)
        return result

    def Preorder_Helper(self, root, result):
        if not root:
            return
        result.append(root.val)
        self.Preorder_Helper(root.left, result)
        self.Preorder_Helper(root.right, result)


def Test_Preorder_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Preorder_Recursive(root1))))
    
    print("Test 1 - Iterative:", " ".join(map(str, solution.Preorder_Iterative(root1))))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Recursive:", " ".join(map(str, solution.Preorder_Recursive(root2))))
    
    print("Test 2 - Iterative:", " ".join(map(str, solution.Preorder_Iterative(root2))))


if __name__ == "__main__":
    Test_Preorder_Traversal()
