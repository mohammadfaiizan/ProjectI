"""
Problem: Remove Duplicates from Unsorted Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-duplicates-from-an-unsorted-linked-list/1

Problem Statement:
Remove duplicate nodes from an unsorted linked list.

Sample Input/Output:
Input: 5->2->2->4->NULL
Output: 5->2->4->NULL
Explanation: Duplicate node with value 2 is removed.
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
    def Remove_Duplicates_Hashing(self, head):
        """
        Hashing approach using set
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        seen = set()
        curr = head
        seen.add(curr.data)
        
        while curr and curr.next:
            if curr.next.data in seen:
                curr.next = curr.next.next
            else:
                seen.add(curr.next.data)
                curr = curr.next
        
        return head
    
    def Remove_Duplicates_Two_Loops(self, head):
        """
        Two loops brute force approach
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head
        
        while curr and curr.next:
            runner = curr
            
            while runner.next:
                if runner.next.data == curr.data:
                    runner.next = runner.next.next
                else:
                    runner = runner.next
            
            curr = curr.next
        
        return head

def Test_Remove_Duplicates_Unsorted():
    solution = Solution()
    
    arr1 = [5, 2, 2, 4]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Remove_Duplicates_Hashing(head1)
    print("After removal (Hashing): ", end="")
    Print_List(head1)
    
    arr2 = [2, 2, 2, 2]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Remove_Duplicates_Two_Loops(head2)
    print("After removal (Two loops): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2, 3, 4, 5]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Remove_Duplicates_Hashing(head3)
    print("After removal (No duplicates): ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Remove_Duplicates_Unsorted()
