"""
Problem: Merge Two Sorted Linked Lists
URL: https://leetcode.com/problems/merge-two-sorted-lists/

Problem Statement:
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together 
the nodes of the first two lists.

Return the head of the merged linked list.

Sample Input/Output:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Input: list1 = [], list2 = []
Output: []

Input: list1 = [], list2 = [0]
Output: [0]
"""

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Merge_Two_Lists_Iterative(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Iterative Approach - Optimal solution
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        dummy = ListNode(0)
        current = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        current.next = list1 if list1 else list2
        
        return dummy.next
    
    def Merge_Two_Lists_Recursive(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Recursive Approach
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        if not list1:
            return list2
        if not list2:
            return list1
        
        if list1.val <= list2.val:
            list1.next = self.Merge_Two_Lists_Recursive(list1.next, list2)
            return list1
        else:
            list2.next = self.Merge_Two_Lists_Recursive(list1, list2.next)
            return list2
    
    def Merge_Two_Lists_In_Place(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        In-place Merge Approach
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not list1:
            return list2
        if not list2:
            return list1
        
        if list1.val > list2.val:
            list1, list2 = list2, list1
        
        head = list1
        
        while list1.next and list2:
            if list1.next.val <= list2.val:
                list1 = list1.next
            else:
                temp = list1.next
                list1.next = list2
                list2 = temp
        
        if not list1.next:
            list1.next = list2
        
        return head
    
    def Merge_Two_Lists_New_List(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Create New List Approach
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        dummy = ListNode(0)
        tail = dummy
        
        p1, p2 = list1, list2
        
        while p1 and p2:
            if p1.val <= p2.val:
                tail.next = ListNode(p1.val)
                p1 = p1.next
            else:
                tail.next = ListNode(p2.val)
                p2 = p2.next
            tail = tail.next
        
        while p1:
            tail.next = ListNode(p1.val)
            p1 = p1.next
            tail = tail.next
        
        while p2:
            tail.next = ListNode(p2.val)
            p2 = p2.next
            tail = tail.next
        
        return dummy.next
    
    def Merge_Two_Lists_Priority_Queue(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Priority Queue Approach
        Time Complexity: O((m + n) log(m + n))
        Space Complexity: O(m + n)
        """
        import heapq
        
        heap = []
        
        while list1:
            heapq.heappush(heap, list1.val)
            list1 = list1.next
        
        while list2:
            heapq.heappush(heap, list2.val)
            list2 = list2.next
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            current.next = ListNode(heapq.heappop(heap))
            current = current.next
        
        return dummy.next

def Create_List(values: List[int]) -> Optional[ListNode]:
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Print_List(head: Optional[ListNode]) -> List[int]:
    result = []
    current = head
    
    while current:
        result.append(current.val)
        current = current.next
    
    return result

def Test_Merge_Two_Lists():
    solution = Solution()
    
    test_cases = [
        ([1,2,4], [1,3,4], [1,1,2,3,4,4]),
        ([], [], []),
        ([], [0], [0]),
        ([1,3,5], [2,4,6], [1,2,3,4,5,6]),
        ([1], [2], [1,2])
    ]
    
    for list1_vals, list2_vals, expected in test_cases:
        result1 = Print_List(solution.Merge_Two_Lists_Iterative(Create_List(list1_vals), Create_List(list2_vals)))
        result2 = Print_List(solution.Merge_Two_Lists_Recursive(Create_List(list1_vals), Create_List(list2_vals)))
        result3 = Print_List(solution.Merge_Two_Lists_In_Place(Create_List(list1_vals), Create_List(list2_vals)))
        result4 = Print_List(solution.Merge_Two_Lists_New_List(Create_List(list1_vals), Create_List(list2_vals)))
        result5 = Print_List(solution.Merge_Two_Lists_Priority_Queue(Create_List(list1_vals), Create_List(list2_vals)))
        
        print(f"List1: {list1_vals}, List2: {list2_vals}")
        print(f"Expected: {expected}")
        print(f"Iterative: {result1}")
        print(f"Recursive: {result2}")
        print(f"In-place: {result3}")
        print(f"New List: {result4}")
        print(f"Priority Queue: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Merge_Two_Lists()

