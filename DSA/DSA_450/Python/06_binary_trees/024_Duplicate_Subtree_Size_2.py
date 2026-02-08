"""
Problem: Duplicate Subtree Size 2
URL: https://practice.geeksforgeeks.org/problems/duplicate-subtree-in-binary-tree/1

Problem Statement:
Check if binary tree contains duplicate subtrees of size 2 or more.

Sample Input/Output:
Input: 
        1
      /   \
     2     3
    / \   / \
   4   5 2   4
      /
     4

Output: true
Explanation: Subtree with root 2 and children 4,5 appears twice.
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
    def Serialize_Subtree(self, root, subtree_map, found_duplicate):
        """
        Hashing with serialization using unordered_map: Serialize each subtree
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for storing serializations
        """
        if not root:
            return "#"
        serial = str(root.val) + "," + \
                 self.Serialize_Subtree(root.left, subtree_map, found_duplicate) + "," + \
                 self.Serialize_Subtree(root.right, subtree_map, found_duplicate)
        if root.left or root.right:
            subtree_map[serial] = subtree_map.get(serial, 0) + 1
            if subtree_map[serial] == 2:
                found_duplicate[0] = True
        return serial

    def Has_Duplicate_Subtree_Map(self, root):
        subtree_map = {}
        found_duplicate = [False]
        self.Serialize_Subtree(root, subtree_map, found_duplicate)
        return found_duplicate[0]

    def Serialize_Subtree_Set(self, root, subtree_set):
        """
        Using unordered_set: Track seen subtrees
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(n) for storing serializations
        """
        if not root:
            return "#"
        serial = str(root.val) + "," + \
                 self.Serialize_Subtree_Set(root.left, subtree_set) + "," + \
                 self.Serialize_Subtree_Set(root.right, subtree_set)
        if root.left or root.right:
            if serial in subtree_set:
                return "DUPLICATE"
            subtree_set.add(serial)
        return serial

    def Has_Duplicate_Subtree_Set(self, root):
        subtree_set = set()
        result = self.Serialize_Subtree_Set(root, subtree_set)
        return result == "DUPLICATE"


def Test_Duplicate_Subtree_Size_2():
    solution = Solution()
    
    vals1 = [1, 2, 3, 4, 5, 2, 4, -1, -1, -1, -1, 4]
    root1 = Build_Tree(vals1)
    print("Tree 1 (map):", solution.Has_Duplicate_Subtree_Map(root1))
    print("Tree 1 (set):", solution.Has_Duplicate_Subtree_Set(root1))
    
    vals2 = [1, 2, 3]
    root2 = Build_Tree(vals2)
    print("Tree 2 (map):", solution.Has_Duplicate_Subtree_Map(root2))
    print("Tree 2 (set):", solution.Has_Duplicate_Subtree_Set(root2))


if __name__ == "__main__":
    Test_Duplicate_Subtree_Size_2()
