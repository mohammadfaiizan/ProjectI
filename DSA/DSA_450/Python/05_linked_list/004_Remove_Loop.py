"""
Problem: Remove Loop in Linked List
URL: https://practice.geeksforgeeks.org/problems/remove-loop-in-linked-list/1

Problem Statement:
Remove the loop from a linked list if it exists.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: 1->2->3->4->5->NULL
Explanation: Loop is removed from the linked list.
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

def Create_Loop(head, pos):
    if pos < 0:
        return
    loop_node = None
    curr = head
    index = 0
    
    while curr.next:
        if index == pos:
            loop_node = curr
        curr = curr.next
        index += 1
    
    if loop_node:
        curr.next = loop_node

def List_To_Array(head, max_nodes=10):
    result = []
    count = 0
    while head and count < max_nodes:
        result.append(head.data)
        head = head.next
        count += 1
    return result

def Print_List(head, max_nodes=10):
    arr = List_To_Array(head, max_nodes)
    if arr:
        print("->".join(map(str, arr)), end="")
        if head and max_nodes == len(arr):
            print("->...", end="")
        print("->NULL")
    else:
        print("NULL")

class Solution:
    def Remove_Loop_Hashing(self, head):
        """
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = set()
        prev = None
        curr = head
        
        while curr:
            if curr in visited:
                prev.next = None
                return
            visited.add(curr)
            prev = curr
            curr = curr.next
    
    def Remove_Loop_Floyd_Detect_Remove(self, head):
        """
        Floyd's detect and remove with counting
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        
        if slow != fast:
            return
        
        loop_length = 1
        temp = slow.next
        while temp != slow:
            loop_length += 1
            temp = temp.next
        
        ptr1 = head
        ptr2 = head
        
        for i in range(loop_length):
            ptr2 = ptr2.next
        
        while ptr1 != ptr2:
            ptr1 = ptr1.next
            ptr2 = ptr2.next
        
        while ptr2.next != ptr1:
            ptr2 = ptr2.next
        
        ptr2.next = None
    
    def Remove_Loop_Floyd_Optimized(self, head):
        """
        Floyd's optimized without counting
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        
        if slow != fast:
            return
        
        if slow == head:
            while fast.next != head:
                fast = fast.next
            fast.next = None
            return
        
        slow = head
        while slow.next != fast.next:
            slow = slow.next
            fast = fast.next
        
        fast.next = None

def Test_Remove_Loop():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    Create_Loop(head1, 1)
    print("Before removal (with loop): ", end="")
    Print_List(head1)
    solution.Remove_Loop_Hashing(head1)
    print("After removal (Hashing): ", end="")
    Print_List(head1)
    
    arr2 = [1, 2, 3, 4, 5]
    head2 = Create_List(arr2)
    Create_Loop(head2, 0)
    print("\nBefore removal (loop at head): ", end="")
    Print_List(head2)
    solution.Remove_Loop_Floyd_Detect_Remove(head2)
    print("After removal (Floyd with counting): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2, 3, 4, 5, 6]
    head3 = Create_List(arr3)
    Create_Loop(head3, 2)
    print("\nBefore removal: ", end="")
    Print_List(head3)
    solution.Remove_Loop_Floyd_Optimized(head3)
    print("After removal (Floyd optimized): ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Remove_Loop()
