"""
Problem: Split a Circular Linked List into Two Halves
URL: https://practice.geeksforgeeks.org/problems/split-a-circular-linked-list-into-two-halves/1

Problem Statement:
Given a Cirular Linked List of size N, split it into two halves circular lists. If there are odd number of nodes in the given circular linked list then out of the resulting two halved lists, first list should have one node more than the second list. The resultant lists should also be circular lists and not linear lists.

Sample Input/Output:
Input: Circular LinkedList: 1->5->7
Output: 1->5 and 7->1
Explanation: Your function will split the given circular linked list into two circular linked lists, one having 1->5 and another having 7->1.
"""

class ListNode:
    def __init__(self, x):
        self.data = x
        self.next = None

def Create_Circular_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = ListNode(arr[i])
        curr = curr.next
    curr.next = head
    return head

def Print_Circular_List(head):
    if not head:
        print("NULL")
        return
    curr = head
    result = []
    while True:
        result.append(str(curr.data))
        curr = curr.next
        if curr == head:
            break
    print("->".join(result))

class Solution:
    def Split_Circular_List_Size_Based(self, head):
        """
        Size-based approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None, None
        
        count = 1
        curr = head.next
        while curr != head:
            count += 1
            curr = curr.next
        
        mid = (count + 1) // 2
        curr = head
        for i in range(1, mid):
            curr = curr.next
        
        head1 = head
        head2 = curr.next
        curr.next = head1
        
        tail = head2
        while tail.next != head:
            tail = tail.next
        tail.next = head2
        
        return head1, head2
    
    def Split_Circular_List_Slow_Fast(self, head):
        """
        Slow-Fast pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None, None
        
        slow = head
        fast = head
        
        while fast.next != head and fast.next.next != head:
            slow = slow.next
            fast = fast.next.next
        
        if fast.next.next == head:
            fast = fast.next
        
        head1 = head
        if head.next != head:
            head2 = slow.next
        else:
            head2 = head
        
        fast.next = slow.next
        slow.next = head
        
        return head1, head2

def Test_Split_Circular_List_Two_Halves():
    solution = Solution()
    
    arr = [1, 5, 7]
    head = Create_Circular_List(arr)
    head1, head2 = solution.Split_Circular_List_Size_Based(head)
    print("Test 1 - Size-Based:")
    print("First half: ", end="")
    Print_Circular_List(head1)
    print("Second half: ", end="")
    Print_Circular_List(head2)
    
    arr = [1, 2, 3, 4, 5]
    head = Create_Circular_List(arr)
    head1, head2 = solution.Split_Circular_List_Slow_Fast(head)
    print("Test 2 - Slow-Fast:")
    print("First half: ", end="")
    Print_Circular_List(head1)
    print("Second half: ", end="")
    Print_Circular_List(head2)

if __name__ == "__main__":
    Test_Split_Circular_List_Two_Halves()
