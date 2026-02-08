"""
Problem: Binary Tree To DLL
URL: https://practice.geeksforgeeks.org/problems/binary-tree-to-dll/1

Problem Statement:
Given a Binary Tree (BT), convert it to a Doubly Linked List(DLL) In-Place. The left and right pointers in nodes are to be used as previous and next pointers respectively in converted DLL. The order of nodes in DLL must be same as Inorder of the given Binary Tree. The first node of Inorder traversal (leftmost node in BT) must be head node of the DLL.

Sample Input/Output:
Input:
        10
      /    \
     12    15
    / \    /
   25 30  36

Output: 25 12 30 10 36 15
Explanation: DLL is created using inorder traversal. Left pointer becomes prev, right pointer becomes next.
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


def Print_DLL(head):
    curr = head
    while curr:
        print(curr.val, end=" ")
        curr = curr.right
    print()


class Solution:
    def Binary_Tree_To_DLL_Inorder(self, root):
        """
        Inorder recursion with head/prev tracking
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        head = [None]
        prev = [None]
        self.Convert_To_DLL(root, head, prev)
        return head[0]

    def Binary_Tree_To_DLL_Inplace(self, root):
        """
        In-place conversion approach
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        if not root:
            return None
        head = [None]
        prev = [None]
        self.Convert_Inplace(root, head, prev)
        return head[0]

    def Convert_To_DLL(self, root, head, prev):
        if not root:
            return
        self.Convert_To_DLL(root.left, head, prev)
        if not prev[0]:
            head[0] = root
        else:
            root.left = prev[0]
            prev[0].right = root
        prev[0] = root
        self.Convert_To_DLL(root.right, head, prev)

    def Convert_Inplace(self, root, head, prev):
        if not root:
            return
        self.Convert_Inplace(root.left, head, prev)
        if not head[0]:
            head[0] = root
        else:
            prev[0].right = root
            root.left = prev[0]
        prev[0] = root
        self.Convert_Inplace(root.right, head, prev)


def Test_Binary_Tree_To_DLL():
    solution = Solution()
    
    vals1 = [10, 12, 15, 25, 30, 36]
    root1 = Build_Tree(vals1)
    print("Test 1 - Inorder:", end=" ")
    head1 = solution.Binary_Tree_To_DLL_Inorder(root1)
    Print_DLL(head1)
    
    vals2 = [10, 12, 15, 25, 30, 36]
    root2 = Build_Tree(vals2)
    print("Test 1 - Inplace:", end=" ")
    head2 = solution.Binary_Tree_To_DLL_Inplace(root2)
    Print_DLL(head2)


if __name__ == "__main__":
    Test_Binary_Tree_To_DLL()
