"""
Problem: Reverse a Doubly Linked List
URL: https://practice.geeksforgeeks.org/problems/reverse-a-doubly-linked-list/1

Problem Statement:
Given a doubly linked list, reverse it.

Sample Input/Output:
Input: 1 <-> 2 <-> 3 <-> 4 <-> 5
Output: 5 <-> 4 <-> 3 <-> 2 <-> 1
Explanation: All pointers are reversed
"""

class DLLNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.prev = None

def Create_DLL(arr):
    if not arr:
        return None
    head = DLLNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = DLLNode(arr[i])
        curr.next.prev = curr
        curr = curr.next
    return head

def Print_DLL(head):
    curr = head
    result = []
    while curr:
        result.append(str(curr.data))
        curr = curr.next
    print(" ".join(result))

class Solution:
    def Reverse_DLL_Iterative(self, head):
        """
        Iterative pointer swap
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head
        temp = None
        
        while curr:
            temp = curr.prev
            curr.prev = curr.next
            curr.next = temp
            curr = curr.prev
        
        if temp:
            head = temp.prev
        return head
    
    def Reverse_DLL_Stack(self, head):
        """
        Stack-based reversal
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        st = []
        curr = head
        while curr:
            st.append(curr.data)
            curr = curr.next
        
        curr = head
        while st:
            curr.data = st.pop()
            curr = curr.next
        
        return head

def Test_Reverse_Doubly_Linked_List():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_DLL(arr1)
    print("Original: ", end="")
    Print_DLL(head1)
    head1 = solution.Reverse_DLL_Iterative(head1)
    print("Reversed (Iterative): ", end="")
    Print_DLL(head1)
    
    arr2 = [10, 20]
    head2 = Create_DLL(arr2)
    print("Original: ", end="")
    Print_DLL(head2)
    head2 = solution.Reverse_DLL_Stack(head2)
    print("Reversed (Stack): ", end="")
    Print_DLL(head2)
    
    arr3 = [5]
    head3 = Create_DLL(arr3)
    print("Original: ", end="")
    Print_DLL(head3)
    head3 = solution.Reverse_DLL_Iterative(head3)
    print("Reversed: ", end="")
    Print_DLL(head3)

if __name__ == "__main__":
    Test_Reverse_Doubly_Linked_List()
