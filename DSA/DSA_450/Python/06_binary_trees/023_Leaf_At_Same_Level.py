"""
Problem: Leaf At Same Level
URL: https://practice.geeksforgeeks.org/problems/leaf-at-same-level/1

Problem Statement:
Check if all leaf nodes are at the same level.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    /
   4

Output: false
Explanation: Leaf nodes 4 and 3 are at different levels.
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
    def Check_Leaf_Level_Recursive(self, root, level, leaf_level):
        """
        Recursive level tracking: Track level of first leaf, compare with others
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height of tree
        """
        if not root:
            return True
        if not root.left and not root.right:
            if leaf_level[0] == -1:
                leaf_level[0] = level
                return True
            return level == leaf_level[0]
        return self.Check_Leaf_Level_Recursive(root.left, level + 1, leaf_level) and \
               self.Check_Leaf_Level_Recursive(root.right, level + 1, leaf_level)

    def Check_Leaf_Same_Level_Recursive(self, root):
        leaf_level = [-1]
        return self.Check_Leaf_Level_Recursive(root, 0, leaf_level)

    def Check_Leaf_Same_Level_BFS(self, root):
        """
        BFS iterative: Use level order traversal to check leaf levels
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for queue
        """
        if not root:
            return True
        q = deque([root])
        leaf_level = -1
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                node = q.popleft()
                if not node.left and not node.right:
                    if leaf_level == -1:
                        leaf_level = level
                    elif leaf_level != level:
                        return False
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            level += 1
        return True


def Test_Leaf_At_Same_Level():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4]
    root1 = Build_Tree(vals1)
    print("Tree 1 (recursive):", solution.Check_Leaf_Same_Level_Recursive(root1))
    print("Tree 1 (BFS):", solution.Check_Leaf_Same_Level_BFS(root1))
    
    vals2 = [1, 2, 3]
    root2 = Build_Tree(vals2)
    print("Tree 2 (recursive):", solution.Check_Leaf_Same_Level_Recursive(root2))
    print("Tree 2 (BFS):", solution.Check_Leaf_Same_Level_BFS(root2))
    
    vals3 = [10, 20, 30, 40, 50]
    root3 = Build_Tree(vals3)
    print("Tree 3 (recursive):", solution.Check_Leaf_Same_Level_Recursive(root3))
    print("Tree 3 (BFS):", solution.Check_Leaf_Same_Level_BFS(root3))


if __name__ == "__main__":
    Test_Leaf_At_Same_Level()
