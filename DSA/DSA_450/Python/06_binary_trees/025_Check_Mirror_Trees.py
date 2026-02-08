"""
Problem: Check Mirror Trees
URL: https://practice.geeksforgeeks.org/problems/check-mirror-in-n-ary-tree1528/1

Problem Statement:
Check if two N-ary trees are mirror of each other. Given as edge lists.

Sample Input/Output:
Input: 
Tree 1: (1,2), (1,3), (1,4)
Tree 2: (1,4), (1,3), (1,2)

Output: true
Explanation: Tree 2 is mirror of Tree 1.
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
    def Check_Mirror_Stack(self, tree1, tree2, n, e):
        """
        Stack-based comparison: Compare children order using stacks
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for stacks
        """
        from collections import defaultdict
        adj1 = defaultdict(list)
        adj2 = defaultdict(list)
        for i in range(0, 2 * e, 2):
            adj1[tree1[i]].append(tree1[i + 1])
        for i in range(0, 2 * e, 2):
            adj2[tree2[i]].append(tree2[i + 1])
        for i in range(1, n + 1):
            if len(adj1[i]) != len(adj2[i]):
                return False
            while adj1[i] and adj2[i]:
                if adj1[i].pop() != adj2[i].pop():
                    return False
        return True


def Test_Check_Mirror_Trees():
    solution = Solution()
    
    n1, e1 = 3, 3
    tree1_1 = [1, 2, 1, 3, 1, 4]
    tree1_2 = [1, 4, 1, 3, 1, 2]
    print("Test 1:", solution.Check_Mirror_Stack(tree1_1, tree1_2, n1, e1))
    
    n2, e2 = 3, 3
    tree2_1 = [1, 2, 1, 3, 1, 4]
    tree2_2 = [1, 2, 1, 3, 1, 4]
    print("Test 2:", solution.Check_Mirror_Stack(tree2_1, tree2_2, n2, e2))


if __name__ == "__main__":
    Test_Check_Mirror_Trees()
