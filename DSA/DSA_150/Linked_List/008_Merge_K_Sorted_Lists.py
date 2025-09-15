"""
Problem: Merge K Sorted Lists
URL: https://leetcode.com/problems/merge-k-sorted-lists/

Problem Statement:
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
Merge all the linked-lists into one sorted linked-list and return it.

Sample Input/Output:
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6

Input: lists = []
Output: []

Input: lists = [[]]
Output: []
"""

from typing import List, Optional
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Merge_K_Lists_Brute_Force(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Brute Force - Collect all values, sort, and rebuild
        Time Complexity: O(N log N)
        Space Complexity: O(N)
        """
        values = []
        
        for head in lists:
            current = head
            while current:
                values.append(current.val)
                current = current.next
        
        values.sort()
        
        dummy = ListNode(0)
        current = dummy
        
        for val in values:
            current.next = ListNode(val)
            current = current.next
        
        return dummy.next
    
    def Merge_K_Lists_Priority_Queue_Optimal(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Priority Queue Optimal - Use min heap for efficient merging
        Time Complexity: O(N log k)
        Space Complexity: O(k)
        """
        class ListNodeWrapper:
            def __init__(self, node: ListNode, index: int):
                self.node = node
                self.index = index
            
            def __lt__(self, other):
                if self.node.val != other.node.val:
                    return self.node.val < other.node.val
                return self.index < other.index
        
        heap = []
        
        for i, head in enumerate(lists):
            if head:
                heapq.heappush(heap, ListNodeWrapper(head, i))
        
        dummy = ListNode(0)
        current = dummy
        
        while heap:
            wrapper = heapq.heappop(heap)
            node = wrapper.node
            
            current.next = node
            current = current.next
            
            if node.next:
                heapq.heappush(heap, ListNodeWrapper(node.next, wrapper.index))
        
        return dummy.next
    
    def Merge_K_Lists_Divide_Conquer(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Divide and Conquer - Merge pairs recursively
        Time Complexity: O(N log k)
        Space Complexity: O(log k)
        """
        def Merge_Two_Lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 if l1 else l2
            return dummy.next
        
        if not lists:
            return None
        
        while len(lists) > 1:
            merged_lists = []
            
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged_lists.append(Merge_Two_Lists(l1, l2))
            
            lists = merged_lists
        
        return lists[0]
    
    def Merge_K_Lists_Sequential_Merge(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Sequential Merge - Merge one by one
        Time Complexity: O(k² * N)
        Space Complexity: O(1)
        """
        def Merge_Two_Lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
            dummy = ListNode(0)
            current = dummy
            
            while l1 and l2:
                if l1.val <= l2.val:
                    current.next = l1
                    l1 = l1.next
                else:
                    current.next = l2
                    l2 = l2.next
                current = current.next
            
            current.next = l1 if l1 else l2
            return dummy.next
        
        if not lists:
            return None
        
        result = lists[0]
        for i in range(1, len(lists)):
            result = Merge_Two_Lists(result, lists[i])
        
        return result
    
    def Merge_K_Lists_Min_Selection(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        """
        Min Selection - Select minimum at each step
        Time Complexity: O(k * N)
        Space Complexity: O(1)
        """
        dummy = ListNode(0)
        current = dummy
        
        while True:
            min_index = -1
            min_val = float('inf')
            
            for i, head in enumerate(lists):
                if head and head.val < min_val:
                    min_val = head.val
                    min_index = i
            
            if min_index == -1:
                break
            
            current.next = lists[min_index]
            lists[min_index] = lists[min_index].next
            current = current.next
        
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

def Test_Merge_K_Lists():
    solution = Solution()
    
    test_cases = [
        ([[1,4,5],[1,3,4],[2,6]], [1,1,2,3,4,4,5,6]),
        ([], []),
        ([[]], []),
        ([[1],[2],[3]], [1,2,3]),
        ([[1,2,3],[4,5,6]], [1,2,3,4,5,6])
    ]
    
    methods = [
        ("Brute Force", solution.Merge_K_Lists_Brute_Force),
        ("Priority Queue Optimal", solution.Merge_K_Lists_Priority_Queue_Optimal),
        ("Divide Conquer", solution.Merge_K_Lists_Divide_Conquer),
        ("Sequential Merge", solution.Merge_K_Lists_Sequential_Merge),
        ("Min Selection", solution.Merge_K_Lists_Min_Selection)
    ]
    
    for lists_vals, expected in test_cases:
        print(f"Input: {lists_vals}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            lists = [Create_Linked_List(vals) for vals in lists_vals]
            result_head = method(lists)
            result = Linked_List_To_Array(result_head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Merge_K_Lists()
