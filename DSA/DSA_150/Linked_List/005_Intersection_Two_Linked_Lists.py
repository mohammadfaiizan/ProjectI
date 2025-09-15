"""
Problem: Intersection of Two Linked Lists
URL: https://leetcode.com/problems/intersection-of-two-linked-lists/

Problem Statement:
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. 
If the two linked lists have no intersection at all, return null.

Sample Input/Output:
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
Output: Intersected at '8'

Input: intersectVal = 2, listA = [1,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Intersected at '2'

Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: No intersection
"""

from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def Get_Intersection_Hash_Set(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Hash Set - Store nodes from one list, check in other
        Time Complexity: O(m + n)
        Space Complexity: O(m)
        """
        visited = set()
        current = headA
        
        while current:
            visited.add(current)
            current = current.next
        
        current = headB
        while current:
            if current in visited:
                return current
            current = current.next
        
        return None
    
    def Get_Intersection_Two_Pointers_Optimal(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Two Pointers Optimal - Switch heads when reaching end
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not headA or not headB:
            return None
        
        ptrA, ptrB = headA, headB
        
        while ptrA != ptrB:
            ptrA = ptrA.next if ptrA else headB
            ptrB = ptrB.next if ptrB else headA
        
        return ptrA
    
    def Get_Intersection_Length_Difference(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Length Difference - Align lists by length difference
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        def Get_Length(head: ListNode) -> int:
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            return length
        
        lenA = Get_Length(headA)
        lenB = Get_Length(headB)
        
        ptrA, ptrB = headA, headB
        
        if lenA > lenB:
            for _ in range(lenA - lenB):
                ptrA = ptrA.next
        else:
            for _ in range(lenB - lenA):
                ptrB = ptrB.next
        
        while ptrA and ptrB:
            if ptrA == ptrB:
                return ptrA
            ptrA = ptrA.next
            ptrB = ptrB.next
        
        return None
    
    def Get_Intersection_Stack_Based(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Stack Based - Use stacks to compare from end
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        stackA, stackB = [], []
        
        current = headA
        while current:
            stackA.append(current)
            current = current.next
        
        current = headB
        while current:
            stackB.append(current)
            current = current.next
        
        intersection = None
        
        while stackA and stackB and stackA[-1] == stackB[-1]:
            intersection = stackA.pop()
            stackB.pop()
        
        return intersection
    
    def Get_Intersection_Cycle_Detection(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        Cycle Detection - Create cycle and detect it
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not headA or not headB:
            return None
        
        tailA = headA
        while tailA.next:
            tailA = tailA.next
        
        tailA.next = headB
        
        slow = fast = headA
        has_cycle = False
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                has_cycle = True
                break
        
        if not has_cycle:
            tailA.next = None
            return None
        
        slow = headA
        while slow != fast:
            slow = slow.next
            fast = fast.next
        
        tailA.next = None
        return slow

def Create_Intersection_Lists(listA_vals, listB_vals, skipA, skipB, intersectVal):
    if intersectVal == 0:
        headA = Create_Simple_List(listA_vals) if listA_vals else None
        headB = Create_Simple_List(listB_vals) if listB_vals else None
        return headA, headB, None
    
    headA = Create_Simple_List(listA_vals[:skipA]) if skipA > 0 else None
    headB = Create_Simple_List(listB_vals[:skipB]) if skipB > 0 else None
    
    intersection = Create_Simple_List(listA_vals[skipA:])
    
    if headA:
        current = headA
        while current.next:
            current = current.next
        current.next = intersection
    else:
        headA = intersection
    
    if headB:
        current = headB
        while current.next:
            current = current.next
        current.next = intersection
    else:
        headB = intersection
    
    return headA, headB, intersection

def Create_Simple_List(values):
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Test_Get_Intersection():
    solution = Solution()
    
    test_cases = [
        ([4,1,8,4,5], [5,6,1,8,4,5], 2, 3, 8),
        ([1,9,1,2,4], [3,2,4], 3, 1, 2),
        ([2,6,4], [1,5], 3, 2, 0)
    ]
    
    methods = [
        ("Hash Set", solution.Get_Intersection_Hash_Set),
        ("Two Pointers Optimal", solution.Get_Intersection_Two_Pointers_Optimal),
        ("Length Difference", solution.Get_Intersection_Length_Difference),
        ("Stack Based", solution.Get_Intersection_Stack_Based),
        ("Cycle Detection", solution.Get_Intersection_Cycle_Detection)
    ]
    
    for listA_vals, listB_vals, skipA, skipB, intersectVal in test_cases:
        print(f"ListA: {listA_vals}, ListB: {listB_vals}")
        print(f"SkipA: {skipA}, SkipB: {skipB}, IntersectVal: {intersectVal}")
        
        for method_name, method in methods:
            headA, headB, expected_node = Create_Intersection_Lists(listA_vals, listB_vals, skipA, skipB, intersectVal)
            result = method(headA, headB)
            result_val = result.val if result else None
            print(f"{method_name}: {result_val}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Get_Intersection()
