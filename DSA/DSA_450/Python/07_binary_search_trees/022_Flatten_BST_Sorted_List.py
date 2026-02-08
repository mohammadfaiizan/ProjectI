"""
Problem: Flatten BST to Sorted List
URL: https://www.geeksforgeeks.org/flatten-bst-to-sorted-list-increasing-order/

Problem Statement:
Flatten BST to sorted linked list using right pointers.

Sample Input/Output:
Input: root = [5,3,7,2,4,6,8]
Output: 2->3->4->5->6->7->8
Explanation: BST flattened to sorted linked list.
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def Build_BST(keys):
        root = None
        for key in keys:
            root = TreeNode.Insert_BST(root, key)
        return root

    @staticmethod
    def Insert_BST(root, key):
        if root is None:
            return TreeNode(key)
        if key < root.val:
            root.left = TreeNode.Insert_BST(root.left, key)
        else:
            root.right = TreeNode.Insert_BST(root.right, key)
        return root

    @staticmethod
    def Print_Inorder(root):
        if root is None:
            return
        TreeNode.Print_Inorder(root.left)
        print(root.val, end=" ")
        TreeNode.Print_Inorder(root.right)


class Solution:
    def Flatten_BST_Inorder(self, root, prev_ref):
        if root is None:
            return
        self.Flatten_BST_Inorder(root.left, prev_ref)
        if prev_ref[0] is not None:
            prev_ref[0].right = root
            prev_ref[0].left = None
        prev_ref[0] = root
        self.Flatten_BST_Inorder(root.right, prev_ref)

    def Flatten_BST_Inorder_Approach(self, root):
        """
        Inorder with prev pointer approach
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        dummy = TreeNode(0)
        prev_ref = [dummy]
        self.Flatten_BST_Inorder(root, prev_ref)
        prev_ref[0].left = None
        prev_ref[0].right = None
        return dummy.right

    def Flatten_BST_Morris(self, root):
        """
        Morris traversal approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        curr = root
        prev = None
        head = None
        while curr is not None:
            if curr.left is None:
                if head is None:
                    head = curr
                if prev is not None:
                    prev.right = curr
                    prev.left = None
                prev = curr
                curr = curr.right
            else:
                predecessor = curr.left
                while predecessor.right is not None and predecessor.right != curr:
                    predecessor = predecessor.right
                if predecessor.right is None:
                    predecessor.right = curr
                    curr = curr.left
                else:
                    predecessor.right = None
                    if head is None:
                        head = curr
                    if prev is not None:
                        prev.right = curr
                        prev.left = None
                    prev = curr
                    curr = curr.right
        if prev is not None:
            prev.left = None
            prev.right = None
        return head


def Test_Flatten_BST_Sorted_List():
    solution = Solution()
    root = None
    root = TreeNode.Insert_BST(root, 5)
    root = TreeNode.Insert_BST(root, 3)
    root = TreeNode.Insert_BST(root, 7)
    root = TreeNode.Insert_BST(root, 2)
    root = TreeNode.Insert_BST(root, 4)
    root = TreeNode.Insert_BST(root, 6)
    root = TreeNode.Insert_BST(root, 8)
    flattened1 = solution.Flatten_BST_Inorder_Approach(root)
    temp = flattened1
    print("Flattened (Inorder):", end=" ")
    while temp is not None:
        print(temp.val, end=" ")
        temp = temp.right
    print()


if __name__ == "__main__":
    Test_Flatten_BST_Sorted_List()
