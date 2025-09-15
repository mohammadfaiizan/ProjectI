"""
Problem: Merge Two Sorted Lists
URL: https://leetcode.com/problems/merge-two-sorted-lists/

Problem Statement:
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.

Sample Input/Output:
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Input: list1 = [], list2 = []
Output: []

Input: list1 = [], list2 = [0]
Output: [0]
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Merge_Two_Lists_Iterative_Optimal(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Iterative Optimal - Use dummy node for simplicity
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
        Recursive Approach - Merge using recursion
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
    
    def Merge_Two_Lists_No_Dummy(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        No Dummy Node - Direct merging without dummy
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not list1:
            return list2
        if not list2:
            return list1
        
        if list1.val <= list2.val:
            head = list1
            list1 = list1.next
        else:
            head = list2
            list2 = list2.next
        
        current = head
        
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        current.next = list1 if list1 else list2
        
        return head
    
    def Merge_Two_Lists_New_List(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        New List Creation - Create new nodes for result
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        dummy = ListNode(0)
        current = dummy
        
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = ListNode(list1.val)
                list1 = list1.next
            else:
                current.next = ListNode(list2.val)
                list2 = list2.next
            current = current.next
        
        while list1:
            current.next = ListNode(list1.val)
            list1 = list1.next
            current = current.next
        
        while list2:
            current.next = ListNode(list2.val)
            list2 = list2.next
            current = current.next
        
        return dummy.next
    
    def Merge_Two_Lists_Tail_Recursive(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Tail Recursive - Optimized recursive approach
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        def Merge_Helper(l1: Optional[ListNode], l2: Optional[ListNode], result: Optional[ListNode]) -> Optional[ListNode]:
            if not l1 and not l2:
                return result
            
            if not l1:
                result.next = l2
                return result
            
            if not l2:
                result.next = l1
                return result
            
            if l1.val <= l2.val:
                result.next = l1
                return Merge_Helper(l1.next, l2, l1)
            else:
                result.next = l2
                return Merge_Helper(l1, l2.next, l2)
        
        dummy = ListNode(0)
        Merge_Helper(list1, list2, dummy)
        return dummy.next

def Create_Linked_List(values):
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Linked_List_To_Array(head):
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
        ([1,2,3], [4,5,6], [1,2,3,4,5,6]),
        ([1], [2], [1,2])
    ]
    
    methods = [
        ("Iterative Optimal", solution.Merge_Two_Lists_Iterative_Optimal),
        ("Recursive", solution.Merge_Two_Lists_Recursive),
        ("No Dummy", solution.Merge_Two_Lists_No_Dummy),
        ("New List", solution.Merge_Two_Lists_New_List),
        ("Tail Recursive", solution.Merge_Two_Lists_Tail_Recursive)
    ]
    
    for list1_vals, list2_vals, expected in test_cases:
        print(f"List1: {list1_vals}, List2: {list2_vals}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            list1 = Create_Linked_List(list1_vals)
            list2 = Create_Linked_List(list2_vals)
            result_head = method(list1, list2)
            result = Linked_List_To_Array(result_head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Merge_Two_Lists()
