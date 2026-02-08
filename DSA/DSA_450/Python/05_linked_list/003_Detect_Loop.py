"""
Problem: Detect Loop in Linked List
URL: https://practice.geeksforgeeks.org/problems/detect-loop-in-linked-list/1

Problem Statement:
Detect if there is a loop in the linked list.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: true
Explanation: Loop exists in the linked list.
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

class Solution:
    def Detect_Loop_Hashing(self, head):
        """
        Hashing approach using set
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = set()
        curr = head
        
        while curr:
            if curr in visited:
                return True
            visited.add(curr)
            curr = curr.next
        
        return False
    
    def Detect_Loop_Floyd(self, head):
        """
        Floyd's Cycle Detection Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return False
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
        
        return False
    
    def Detect_Loop_Temp_Node(self, head):
        """
        Temp node marking approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return False
        
        temp = ListNode(0)
        curr = head
        
        while curr:
            if curr.next == temp:
                return True
            
            next_node = curr.next
            curr.next = temp
            curr = next_node
        
        return False

def Test_Detect_Loop():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    Create_Loop(head1, 1)
    print("Test 1 (Loop exists):", solution.Detect_Loop_Hashing(head1))
    print("Test 1 (Floyd):", solution.Detect_Loop_Floyd(head1))
    
    arr2 = [1, 2, 3, 4, 5]
    head2 = Create_List(arr2)
    print("\nTest 2 (No loop):", solution.Detect_Loop_Hashing(head2))
    print("Test 2 (Floyd):", solution.Detect_Loop_Floyd(head2))
    
    arr3 = [1]
    head3 = Create_List(arr3)
    print("\nTest 3 (Single node, no loop):", solution.Detect_Loop_Temp_Node(head3))

if __name__ == "__main__":
    Test_Detect_Loop()
