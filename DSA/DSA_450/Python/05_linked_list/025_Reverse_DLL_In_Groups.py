"""
Problem: Reverse a Doubly Linked List in Groups of Given Size
URL: https://www.geeksforgeeks.org/reverse-doubly-linked-list-groups-given-size/

Problem Statement:
Given a doubly linked list, reverse it in groups of given size K.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 3 <-> 4 <-> 5 <-> 6, K = 3
Output: List: 3 <-> 2 <-> 1 <-> 6 <-> 5 <-> 4
Explanation: Reversed in groups of 3
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
    def Reverse_DLL_Groups_Recursive(self, head, k):
        """
        Recursive group reversal
        Time Complexity: O(n)
        Space Complexity: O(n/k)
        """
        if not head:
            return None
        
        curr = head
        next_node = None
        new_head = None
        count = 0
        
        while curr and count < k:
            next_node = curr.next
            curr.next = new_head
            if new_head:
                new_head.prev = curr
            new_head = curr
            new_head.prev = None
            curr = next_node
            count += 1
        
        if next_node:
            head.next = self.Reverse_DLL_Groups_Recursive(next_node, k)
            if head.next:
                head.next.prev = head
        
        return new_head
    
    def Reverse_DLL_Groups_Iterative(self, head, k):
        """
        Iterative approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or k == 1:
            return head
        
        dummy = DLLNode(0)
        dummy.next = head
        head.prev = dummy
        
        group_prev = dummy
        
        while group_prev.next:
            group_start = group_prev.next
            group_end = group_start
            count = 1
            
            while group_end.next and count < k:
                group_end = group_end.next
                count += 1
            
            group_next = group_end.next
            
            prev_node = group_prev
            curr = group_start
            
            while curr != group_next:
                next_temp = curr.next
                curr.next = prev_node
                curr.prev = next_temp
                prev_node = curr
                curr = next_temp
            
            group_prev.next = group_end
            if group_next:
                group_next.prev = group_start
            group_start.next = group_next
            group_prev = group_start
        
        result = dummy.next
        result.prev = None
        return result

def Test_Reverse_DLL_In_Groups():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5, 6]
    head1 = Create_DLL(arr1)
    print("Original: ", end="")
    Print_DLL(head1)
    head1 = solution.Reverse_DLL_Groups_Recursive(head1, 3)
    print("Reversed in groups of 3 (Recursive): ", end="")
    Print_DLL(head1)
    
    arr2 = [1, 2, 3, 4, 5, 6, 7, 8]
    head2 = Create_DLL(arr2)
    print("Original: ", end="")
    Print_DLL(head2)
    head2 = solution.Reverse_DLL_Groups_Iterative(head2, 3)
    print("Reversed in groups of 3 (Iterative): ", end="")
    Print_DLL(head2)
    
    arr3 = [1, 2, 3, 4]
    head3 = Create_DLL(arr3)
    print("Original: ", end="")
    Print_DLL(head3)
    head3 = solution.Reverse_DLL_Groups_Recursive(head3, 2)
    print("Reversed in groups of 2: ", end="")
    Print_DLL(head3)

if __name__ == "__main__":
    Test_Reverse_DLL_In_Groups()
