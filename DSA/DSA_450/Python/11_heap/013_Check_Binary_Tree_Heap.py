"""
Problem: Check if a Binary Tree is a Heap
URL: https://practice.geeksforgeeks.org/problems/is-binary-tree-heap/1

Problem Statement:
Check if a given binary tree satisfies max heap properties: completeness + max heap ordering.

Sample Input/Output:
Input: Tree structure
Output: true/false
"""

from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def Check_Heap_Recursive(self, root):
        """
        Recursive Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        node_count = self.CountNodes(root)
        return self.IsComplete(root, 0, node_count) and self.IsMaxHeap(root)

    def Check_Heap_Level_Order(self, root):
        """
        Level Order Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not root:
            return True

        q = deque([root])
        found_null = False

        while q:
            node = q.popleft()

            if not node:
                found_null = True
            else:
                if found_null:
                    return False

                if node.left:
                    if node.left.val > node.val:
                        return False
                    q.append(node.left)
                else:
                    q.append(None)

                if node.right:
                    if node.right.val > node.val:
                        return False
                    q.append(node.right)
                else:
                    q.append(None)

        return True

    def CountNodes(self, root):
        if not root:
            return 0
        return 1 + self.CountNodes(root.left) + self.CountNodes(root.right)

    def IsComplete(self, root, index, node_count):
        if not root:
            return True
        if index >= node_count:
            return False
        return self.IsComplete(root.left, 2 * index + 1, node_count) and \
               self.IsComplete(root.right, 2 * index + 2, node_count)

    def IsMaxHeap(self, root):
        if not root:
            return True

        left_valid = True
        right_valid = True

        if root.left:
            if root.left.val > root.val:
                return False
            left_valid = self.IsMaxHeap(root.left)

        if root.right:
            if root.right.val > root.val:
                return False
            right_valid = self.IsMaxHeap(root.right)

        return left_valid and right_valid


def Test_Check_Heap():
    solution = Solution()

    root1 = TreeNode(10)
    root1.left = TreeNode(9)
    root1.right = TreeNode(8)
    root1.left.left = TreeNode(7)
    root1.left.right = TreeNode(6)
    root1.right.left = TreeNode(5)

    print("Test 1 (Valid Heap):", solution.Check_Heap_Recursive(root1))
    print("Test 1 Level Order:", solution.Check_Heap_Level_Order(root1))

    root2 = TreeNode(10)
    root2.left = TreeNode(15)
    root2.right = TreeNode(8)

    print("Test 2 (Invalid Heap):", solution.Check_Heap_Recursive(root2))
    print("Test 2 Level Order:", solution.Check_Heap_Level_Order(root2))

    root3 = TreeNode(10)
    root3.left = TreeNode(9)
    root3.right = TreeNode(8)
    root3.left.left = TreeNode(7)

    print("Test 3 (Valid Heap):", solution.Check_Heap_Recursive(root3))
    print("Test 3 Level Order:", solution.Check_Heap_Level_Order(root3))


if __name__ == "__main__":
    Test_Check_Heap()
