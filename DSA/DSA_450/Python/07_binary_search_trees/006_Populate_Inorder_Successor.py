"""
Problem: Populate Inorder Successor for All Nodes
URL: https://practice.geeksforgeeks.org/problems/populate-inorder-successor-for-all-nodes/1

Problem Statement:
Given a Binary Tree, write a function to populate next pointer for all nodes. The next pointer for every node should be set to point to inorder successor. Use reverse inorder traversal approach.

Sample Input/Output:
Input: Root with data 10, left 8, right 12
Output: Node 8's next points to 10, Node 10's next points to 12, Node 12's next is NULL
Explanation: Inorder is 8->10->12, so next pointers follow this order
"""


class TreeNode_With_Next:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:
    def Populate_Inorder_Successor_Reverse_Inorder(self, root, next_node_ref):
        """
        Reverse inorder traversal: process right, root, left
        Time Complexity: O(n) where n is number of nodes
        Space Complexity: O(h) where h is height for recursion stack
        """
        if root is None:
            return
        self.Populate_Inorder_Successor_Reverse_Inorder(root.right, next_node_ref)
        root.next = next_node_ref[0]
        next_node_ref[0] = root
        self.Populate_Inorder_Successor_Reverse_Inorder(root.left, next_node_ref)

    def Populate_Next_Pointers(self, root):
        next_node_ref = [None]
        self.Populate_Inorder_Successor_Reverse_Inorder(root, next_node_ref)


def Print_Inorder_With_Next(root):
    if root is None:
        return
    Print_Inorder_With_Next(root.left)
    print(root.val, "->", root.next.val if root.next else -1)
    Print_Inorder_With_Next(root.right)


def Test_Populate_Inorder_Successor():
    solution = Solution()
    root = TreeNode_With_Next(10)
    root.left = TreeNode_With_Next(8)
    root.right = TreeNode_With_Next(12)
    root.left.left = TreeNode_With_Next(3)

    solution.Populate_Next_Pointers(root)

    print("Inorder traversal with next pointers:")
    Print_Inorder_With_Next(root)


if __name__ == "__main__":
    Test_Populate_Inorder_Successor()
