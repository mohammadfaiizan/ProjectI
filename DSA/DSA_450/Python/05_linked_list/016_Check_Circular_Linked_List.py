"""
Problem: Check if Linked List is Circular
URL: https://practice.geeksforgeeks.org/problems/circular-linked-list/1

Problem Statement:
Given a singly linked list, find if the linked list is circular or not. A linked list is called circular if it not NULL terminated and all nodes are connected in the form of a cycle.

Sample Input/Output:
Input: LinkedList: 1->2->3->4->5->1 (5 is connected to 1)
Output: 1
Explanation: The given linked list is circular.
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

def Create_Linear_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = ListNode(arr[i])
        curr = curr.next
    return head

def Print_Circular_List(head, max_nodes=10):
    if not head:
        print("NULL")
        return
    curr = head
    count = 0
    result = []
    while count < max_nodes:
        result.append(str(curr.data))
        curr = curr.next
        count += 1
        if curr == head:
            break
    print("->".join(result))

class Solution:
    def Is_Circular_Traverse(self, head):
        """
        Traverse and check if last points to head
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        curr = head.next
        while curr and curr != head:
            curr = curr.next
        
        return curr == head
    
    def Is_Circular_Floyd(self, head):
        """
        Floyd's approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return slow == head or fast == head
        
        return False

def Test_Check_Circular_Linked_List():
    solution = Solution()
    
    arr = [1, 2, 3, 4, 5]
    circular_head = Create_Circular_List(arr)
    result1 = solution.Is_Circular_Traverse(circular_head)
    print("Test 1 - Traverse (Circular):", result1)
    
    result2 = solution.Is_Circular_Floyd(circular_head)
    print("Test 1 - Floyd (Circular):", result2)
    
    linear_head = Create_Linear_List(arr)
    result1 = solution.Is_Circular_Traverse(linear_head)
    print("Test 2 - Traverse (Linear):", result1)
    
    result2 = solution.Is_Circular_Floyd(linear_head)
    print("Test 2 - Floyd (Linear):", result2)

if __name__ == "__main__":
    Test_Check_Circular_Linked_List()
