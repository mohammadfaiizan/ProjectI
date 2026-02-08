"""
Problem: Inorder Traversal
URL: https://www.techiedelight.com/inorder-tree-traversal-iterative-recursive/

Problem Statement:
Given a binary tree, perform inorder traversal. Inorder traversal visits left subtree, root, then right subtree.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 6   7

Output: 4 2 5 1 6 3 7
Explanation: Left subtree -> Root -> Right subtree
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
    def Inorder_Recursive(self, root):
        """
        Recursive approach: Visit left, root, right
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        self.Inorder_Helper(root, result)
        return result

    def Inorder_Iterative(self, root):
        """
        Iterative with stack: Use stack to simulate recursion
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        result = []
        if not root:
            return result
        st = []
        current = root
        while current or st:
            while current:
                st.append(current)
                current = current.left
            current = st.pop()
            result.append(current.val)
            current = current.right
        return result

    def Inorder_Helper(self, root, result):
        if not root:
            return
        self.Inorder_Helper(root.left, result)
        result.append(root.val)
        self.Inorder_Helper(root.right, result)


def Test_Inorder_Traversal():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 6, 7]
    root1 = Build_Tree(vals1)
    print("Test 1 - Recursive:", " ".join(map(str, solution.Inorder_Recursive(root1))))
    
    print("Test 1 - Iterative:", " ".join(map(str, solution.Inorder_Iterative(root1))))
    
    vals2 = [1, 2, 3, -1, -1, 4, 5]
    root2 = Build_Tree(vals2)
    print("Test 2 - Recursive:", " ".join(map(str, solution.Inorder_Recursive(root2))))
    
    print("Test 2 - Iterative:", " ".join(map(str, solution.Inorder_Iterative(root2))))


if __name__ == "__main__":
    Test_Inorder_Traversal()
