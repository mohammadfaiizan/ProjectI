"""
Problem: Intersection of Two Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/intersection-of-two-sorted-linked-lists/1

Problem Statement:
Given two lists sorted in increasing order, create a new list representing the intersection of the two lists. The new list should be made with its own memory — the original lists should not be changed.

Sample Input/Output:
Input: First linked list: 1->2->3->4->6, Second linked list: 2->4->6->8
Output: 2->4->6
Explanation: Nodes 2, 4 and 6 are common in both lists.
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
    if arr:
        print("->".join(map(str, arr)))
    else:
        print("NULL")

class Solution:
    def Intersection_Two_Pointer(self, head1, head2):
        """
        Two pointer on sorted lists
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        """
        dummy = ListNode(0)
        tail = dummy
        
        while head1 and head2:
            if head1.data == head2.data:
                tail.next = ListNode(head1.data)
                tail = tail.next
                head1 = head1.next
                head2 = head2.next
            elif head1.data < head2.data:
                head1 = head1.next
            else:
                head2 = head2.next
        
        return dummy.next
    
    def Intersection_Recursive(self, head1, head2):
        """
        Recursive approach
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        """
        if not head1 or not head2:
            return None
        
        if head1.data < head2.data:
            return self.Intersection_Recursive(head1.next, head2)
        
        if head1.data > head2.data:
            return self.Intersection_Recursive(head1, head2.next)
        
        node = ListNode(head1.data)
        node.next = self.Intersection_Recursive(head1.next, head2.next)
        return node

def Test_Intersection_Two_Sorted_Lists():
    solution = Solution()
    
    arr1 = [1, 2, 3, 4, 6]
    arr2 = [2, 4, 6, 8]
    head1 = Create_List(arr1)
    head2 = Create_List(arr2)
    result1 = solution.Intersection_Two_Pointer(head1, head2)
    print("Test 1 - Two Pointer: ", end="")
    Print_List(result1)
    
    head1 = Create_List(arr1)
    head2 = Create_List(arr2)
    result2 = solution.Intersection_Recursive(head1, head2)
    print("Test 1 - Recursive: ", end="")
    Print_List(result2)
    
    arr3 = [1, 3, 5]
    arr4 = [2, 4, 6]
    head1 = Create_List(arr3)
    head2 = Create_List(arr4)
    result1 = solution.Intersection_Two_Pointer(head1, head2)
    print("Test 2 - Two Pointer: ", end="")
    Print_List(result1)

if __name__ == "__main__":
    Test_Intersection_Two_Sorted_Lists()
