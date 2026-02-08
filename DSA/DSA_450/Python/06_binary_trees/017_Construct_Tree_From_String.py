"""
Problem: Construct Tree From String
URL: https://www.geeksforgeeks.org/construct-binary-tree-string-bracket-representation/

Problem Statement:
Construct a binary tree from a string consisting of parenthesis and integers. The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

Sample Input/Output:
Input: "4(2(3)(1))(6(5))"
Output:
        4
      /   \
     2     6
    / \   /
   3   1 5

Explanation: Root is 4, left subtree is 2(3)(1), right subtree is 6(5).
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
    def Construct_Tree_Stack(self, s):
        """
        Recursive with stack for bracket matching
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        if not s:
            return None
        i = [0]
        return self.Construct_Helper_Stack(s, i)

    def Construct_Tree_Pointer(self, s):
        """
        Recursive with pointer index
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        if not s:
            return None
        index = [0]
        return self.Construct_Helper_Pointer(s, index)

    def Construct_Helper_Stack(self, s, i):
        if i[0] >= len(s):
            return None
        num = 0
        while i[0] < len(s) and s[i[0]].isdigit():
            num = num * 10 + int(s[i[0]])
            i[0] += 1
        root = TreeNode(num)
        if i[0] < len(s) and s[i[0]] == '(':
            i[0] += 1
            root.left = self.Construct_Helper_Stack(s, i)
            i[0] += 1
        if i[0] < len(s) and s[i[0]] == '(':
            i[0] += 1
            root.right = self.Construct_Helper_Stack(s, i)
            i[0] += 1
        return root

    def Construct_Helper_Pointer(self, s, index):
        if index[0] >= len(s):
            return None
        negative = False
        if s[index[0]] == '-':
            negative = True
            index[0] += 1
        num = 0
        while index[0] < len(s) and s[index[0]].isdigit():
            num = num * 10 + int(s[index[0]])
            index[0] += 1
        if negative:
            num = -num
        root = TreeNode(num)
        if index[0] < len(s) and s[index[0]] == '(':
            index[0] += 1
            root.left = self.Construct_Helper_Pointer(s, index)
            index[0] += 1
        if index[0] < len(s) and s[index[0]] == '(':
            index[0] += 1
            root.right = self.Construct_Helper_Pointer(s, index)
            index[0] += 1
        return root


def Test_Construct_Tree_From_String():
    solution = Solution()
    
    s1 = "4(2(3)(1))(6(5))"
    print("Test 1 - Stack:", end=" ")
    root1 = solution.Construct_Tree_Stack(s1)
    Print_Inorder(root1)
    print()
    
    print("Test 1 - Pointer:", end=" ")
    root2 = solution.Construct_Tree_Pointer(s1)
    Print_Inorder(root2)
    print()


if __name__ == "__main__":
    Test_Construct_Tree_From_String()
