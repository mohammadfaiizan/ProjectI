"""
Problem: Move Last Element to Front of Linked List
URL: https://www.geeksforgeeks.org/move-last-element-to-front-of-a-given-linked-list/

Problem Statement:
Move the last element of the linked list to the front.

Sample Input/Output:
Input: 1->2->3->4->5->NULL
Output: 5->1->2->3->4->NULL
Explanation: Last node (5) is moved to the front.
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
    def Move_Last_To_Front_Traverse(self, head):
        """
        Traverse to end approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        curr = head
        prev = None
        
        while curr.next:
            prev = curr
            curr = curr.next
        
        prev.next = None
        curr.next = head
        head = curr
        
        return head
    
    def Move_Last_To_Front_Two_Pointer(self, head):
        """
        Two pointer approach (fast pointer reaching end first)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return head
        
        slow = head
        fast = head.next
        
        while fast.next:
            slow = slow.next
            fast = fast.next
        
        fast.next = head
        head = fast
        slow.next = None
        
        return head

def Test_Move_Last_To_Front():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_List(arr1)
    print("Original: ", end="")
    Print_List(head1)
    head1 = solution.Move_Last_To_Front_Traverse(head1)
    print("After moving last to front (Traverse): ", end="")
    Print_List(head1)
    
    arr2 = [1, 2, 3, 4, 5]
    head2 = Create_List(arr2)
    print("\nOriginal: ", end="")
    Print_List(head2)
    head2 = solution.Move_Last_To_Front_Two_Pointer(head2)
    print("After moving last to front (Two pointer): ", end="")
    Print_List(head2)
    
    arr3 = [1, 2]
    head3 = Create_List(arr3)
    print("\nOriginal: ", end="")
    Print_List(head3)
    head3 = solution.Move_Last_To_Front_Traverse(head3)
    print("After moving last to front: ", end="")
    Print_List(head3)

if __name__ == "__main__":
    Test_Move_Last_To_Front()
