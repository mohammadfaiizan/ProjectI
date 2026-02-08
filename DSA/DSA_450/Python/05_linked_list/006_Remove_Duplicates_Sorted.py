"""
Problem: Remove Duplicates from Sorted Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-duplicate-element-from-sorted-linked-list/1

Problem Statement:
Remove duplicate nodes from a sorted linked list.

Sample Input/Output:
Input: 1->1->2->3->3->4->NULL
Output: 1->2->3->4->NULL
Explanation: Duplicate nodes are removed, keeping only one occurrence.
"""

class ListNode:
    def __init__(self, x):
        self.data = x
        self.next = None

def Create_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = ListNode(arr[i])
        curr = curr.next
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result

def Print_List(head):
    arr = List_To_Array(head)
    print("->".join(map(str, arr)) + "->NULL")

class Solution:
    def Remove_Duplicates_Iterative(self, head):
        """
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head
        
        while curr and curr.next:
            if curr.data == curr.next.data:
                curr.next = curr.next.next
            else:
                curr = curr.next
        
        return head
    
    def Remove_Duplicates_Recursive(self, head):
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        head.next = self.Remove_Duplicates_Recursive(head.next)
        
        if head.data == head.next.data:
            return head.next
        
        return head

def Test_Remove_Duplicates_Sorted():
    solution = Solution()
    
    arr1 = [1, 1, 2, 3, 3, 4]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Remove_Duplicates_Iterative(head1)
    print("After removal (Iterative): ", end="")
    Print_List(head1)
    
    arr2 = [1, 1, 1]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Remove_Duplicates_Recursive(head2)
    print("After removal (Recursive): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2, 3, 4, 5]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Remove_Duplicates_Iterative(head3)
    print("After removal (No duplicates): ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Remove_Duplicates_Sorted()
