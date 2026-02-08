"""
Problem: Reverse a Linked List
URL: https://www.geeksforgeeks.org/reverse-a-linked-list/

Problem Statement:
Given a linked list, reverse it.

Sample Input/Output:
Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
Explanation: The linked list is reversed completely.
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
    def Reverse_Iterative(self, head):
        """
        Iterative approach using three pointers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        prev = None
        curr = head
        next_node = None
        
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        return prev
    
    def Reverse_Recursive(self, head):
        """
        Recursive approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        rest = self.Reverse_Recursive(head.next)
        head.next.next = head
        head.next = None
        
        return rest
    
    def Reverse_Stack_Based(self, head):
        """
        Stack-based approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if not head or not head.next:
            return head
        
        st = []
        curr = head
        
        while curr:
            st.append(curr)
            curr = curr.next
        
        new_head = st.pop()
        curr = new_head
        
        while st:
            curr.next = st.pop()
            curr = curr.next
        
        curr.next = None
        return new_head

def Test_Reverse_Linked_List():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Reverse_Iterative(head1)
    print("Reversed (Iterative): ", end="")
    Print_List(head1)
    
    arr2 = [1, 2]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Reverse_Recursive(head2)
    print("Reversed (Recursive): ", end="")
    Print_List(head2)
    
    arr3 = [1]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Reverse_Stack_Based(head3)
    print("Reversed (Stack): ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Reverse_Linked_List()
