"""
Problem: Deletion from a Circular Linked List
URL: https://www.geeksforgeeks.org/deletion-circular-linked-list/

Problem Statement:
Given a circular linked list and a key, delete the node with the given key.

Sample Input/Output:
Input: List: 1->2->3->4->5 (circular), key = 3
Output: List: 1->2->4->5 (circular)
Explanation: Node with value 3 is removed from the circular list
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
        return
    curr = head
    result = []
    while True:
        result.append(str(curr.data))
        curr = curr.next
        if curr == head:
            break
    print(" ".join(result))

class Solution:
    def Delete_From_Circular_List_Search_Delete(self, head, key):
        """
        Search and delete with edge cases (head, middle, not found)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head:
            return None
        
        if head.data == key:
            if head.next == head:
                return None
            last = head
            while last.next != head:
                last = last.next
            last.next = head.next
            new_head = head.next
            return new_head
        
        curr = head
        while curr.next != head:
            if curr.next.data == key:
                curr.next = curr.next.next
                return head
            curr = curr.next
        
        return head

def Test_Delete_From_Circular_List():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 5]
    head1 = Create_Circular_List(arr1)
    print("Original: ", end="")
    Print_Circular_List(head1)
    head1 = solution.Delete_From_Circular_List_Search_Delete(head1, 3)
    print("After deleting 3: ", end="")
    Print_Circular_List(head1)
    
    arr2 = [10]
    head2 = Create_Circular_List(arr2)
    print("Original: ", end="")
    Print_Circular_List(head2)
    head2 = solution.Delete_From_Circular_List_Search_Delete(head2, 10)
    print("After deleting 10: ", end="")
    if head2:
        Print_Circular_List(head2)
    else:
        print("Empty list")
    
    arr3 = [5, 10, 15]
    head3 = Create_Circular_List(arr3)
    print("Original: ", end="")
    Print_Circular_List(head3)
    head3 = solution.Delete_From_Circular_List_Search_Delete(head3, 5)
    print("After deleting head (5): ", end="")
    Print_Circular_List(head3)

if __name__ == "__main__":
    Test_Delete_From_Circular_List()
