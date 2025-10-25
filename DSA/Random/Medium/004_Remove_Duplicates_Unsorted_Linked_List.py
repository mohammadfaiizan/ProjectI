"""
Problem: Remove Duplicates From an Unsorted Linked List
URL: https://leetcode.com/problems/remove-duplicates-from-an-unsorted-linked-list/

Problem Statement:
Given the head of a linked list, find all the values that appear more than once in the list 
and delete the nodes that have any of those values.

Return the linked list after the deletions.

Sample Input/Output:
Input: head = [1,2,3,2]
Output: [1,3]

Input: head = [2,1,1,2]
Output: []

Input: head = [3,2,2,1,3,2,4]
Output: [1,4]
"""

from typing import List, Optional
from collections import Counter

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Delete_Duplicates_Hash_Set(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Hash Set Approach - Two pass
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = {}
        current = head
        
        while current:
            freq[current.val] = freq.get(current.val, 0) + 1
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev, current = dummy, head
        
        while current:
            if freq[current.val] > 1:
                prev.next = current.next
            else:
                prev = current
            current = current.next
        
        return dummy.next
    
    def Delete_Duplicates_Counter(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Counter Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        freq = Counter(values)
        
        dummy = ListNode(0)
        dummy.next = head
        prev, current = dummy, head
        
        while current:
            if freq[current.val] > 1:
                prev.next = current.next
            else:
                prev = current
            current = current.next
        
        return dummy.next
    
    def Delete_Duplicates_Dictionary(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Dictionary Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        count_map = {}
        current = head
        
        while current:
            if current.val in count_map:
                count_map[current.val] += 1
            else:
                count_map[current.val] = 1
            current = current.next
        
        dummy = ListNode(0, head)
        current = dummy
        
        while current.next:
            if count_map[current.next.val] > 1:
                current.next = current.next.next
            else:
                current = current.next
        
        return dummy.next
    
    def Delete_Duplicates_Set_Two_Pass(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Set Two Pass Approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = set()
        duplicates = set()
        current = head
        
        while current:
            if current.val in seen:
                duplicates.add(current.val)
            else:
                seen.add(current.val)
            current = current.next
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head
        
        while current:
            if current.val in duplicates:
                prev.next = current.next
            else:
                prev = current
            current = current.next
        
        return dummy.next
    
    def Delete_Duplicates_Optimized(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        Optimized Single Dictionary
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = {}
        curr = head
        
        while curr:
            freq[curr.val] = freq.get(curr.val, 0) + 1
            curr = curr.next
        
        dummy = ListNode(0, head)
        curr = dummy
        
        while curr and curr.next:
            if freq.get(curr.next.val, 0) > 1:
                curr.next = curr.next.next
            else:
                curr = curr.next
        
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

def Test_Delete_Duplicates():
    solution = Solution()
    
    test_cases = [
        ([1,2,3,2], [1,3]),
        ([2,1,1,2], []),
        ([3,2,2,1,3,2,4], [1,4]),
        ([1,1,1], []),
        ([1,2,3,4,5], [1,2,3,4,5])
    ]
    
    for values, expected in test_cases:
        result1 = Print_List(solution.Delete_Duplicates_Hash_Set(Create_List(values)))
        result2 = Print_List(solution.Delete_Duplicates_Counter(Create_List(values)))
        result3 = Print_List(solution.Delete_Duplicates_Dictionary(Create_List(values)))
        result4 = Print_List(solution.Delete_Duplicates_Set_Two_Pass(Create_List(values)))
        result5 = Print_List(solution.Delete_Duplicates_Optimized(Create_List(values)))
        
        print(f"Input: {values}")
        print(f"Expected: {expected}")
        print(f"Hash Set: {result1}")
        print(f"Counter: {result2}")
        print(f"Dictionary: {result3}")
        print(f"Set Two Pass: {result4}")
        print(f"Optimized: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Delete_Duplicates()

