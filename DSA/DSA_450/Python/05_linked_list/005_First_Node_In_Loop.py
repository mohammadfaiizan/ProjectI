"""
Problem: Find First Node of Loop in a Linked List
URL: https://www.geeksforgeeks.org/find-first-node-of-loop-in-a-linked-list/

Problem Statement:
Find the first node of the loop in a linked list.

Sample Input/Output:
Input: 1->2->3->4->5->2 (loop at node 2)
Output: 2
Explanation: Node with value 2 is the first node of the loop.
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
    def First_Node_Floyd(self, head):
        """
        Floyd's Cycle Detection Algorithm
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return None
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                break
        
        if slow != fast:
            return None
        
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        return slow
    
    def First_Node_Hashing(self, head):
        """
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        visited = set()
        curr = head
        
        while curr:
            if curr in visited:
                return curr
            visited.add(curr)
            curr = curr.next
        
        return None

def Test_First_Node_In_Loop():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    Create_Loop(head1, 1)
    result1 = solution.First_Node_Floyd(head1)
    print("Test 1 (Floyd): First node value =", result1.data if result1 else -1)
    
    arr2 = [1, 2, 3, 4, 5]
    head2 = Create_List(arr2)
    Create_Loop(head2, 1)
    result2 = solution.First_Node_Hashing(head2)
    print("Test 1 (Hashing): First node value =", result2.data if result2 else -1)
    
    arr3 = [1, 2, 3]
    head3 = Create_List(arr3)
    Create_Loop(head3, 0)
    result3 = solution.First_Node_Floyd(head3)
    print("\nTest 2 (Loop at head): First node value =", result3.data if result3 else -1)
    
    arr4 = [1, 2, 3, 4, 5]
    head4 = Create_List(arr4)
    result4 = solution.First_Node_Hashing(head4)
    print("\nTest 3 (No loop): First node value =", result4.data if result4 else -1)

if __name__ == "__main__":
    Test_First_Node_In_Loop()
