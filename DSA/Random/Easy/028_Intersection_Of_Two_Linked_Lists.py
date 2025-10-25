"""
Problem: Intersection of Two Linked Lists
URL: https://leetcode.com/problems/intersection-of-two-linked-lists/

Problem Statement:
Given the heads of two singly linked-lists headA and headB, return the node at which the 
two lists intersect. If the two linked lists have no intersection at all, return null.

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
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Get_Intersection_Hash_Set(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Hash Set Approach
        Time Complexity: O(m + n)
        Space Complexity: O(m) or O(n)
        """
        seen = set()
        
        current = headA
        while current:
            seen.add(current)
            current = current.next
        
        current = headB
        while current:
            if current in seen:
                return current
            current = current.next
        
        return None
    
    def Get_Intersection_Two_Pointer(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pointer Approach - Optimal solution
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not headA or not headB:
            return None
        
        pA, pB = headA, headB
        
        while pA != pB:
            pA = pA.next if pA else headB
            pB = pB.next if pB else headA
        
        return pA
    
    def Get_Intersection_Length_Diff(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Length Difference Approach
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        def Get_Length(head):
            length = 0
            current = head
            while current:
                length += 1
                current = current.next
            return length
        
        lenA = Get_Length(headA)
        lenB = Get_Length(headB)
        
        while lenA > lenB:
            headA = headA.next
            lenA -= 1
        
        while lenB > lenA:
            headB = headB.next
            lenB -= 1
        
        while headA and headB:
            if headA == headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
    
    def Get_Intersection_Stack(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Stack Approach
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
    
    def Get_Intersection_Reverse(self, headA: Optional[ListNode], headB: Optional[ListNode]) -> Optional[ListNode]:
        """
        Two Pointer with Switching
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        if not headA or not headB:
            return None
        
        currA, currB = headA, headB
        
        while currA != currB:
            currA = headB if not currA else currA.next
            currB = headA if not currB else currB.next
        
        return currA

def Test_Get_Intersection():
    solution = Solution()
    
    intersection = ListNode(8)
    intersection.next = ListNode(4)
    intersection.next.next = ListNode(5)
    
    listA = ListNode(4)
    listA.next = ListNode(1)
    listA.next.next = intersection
    
    listB = ListNode(5)
    listB.next = ListNode(6)
    listB.next.next = ListNode(1)
    listB.next.next.next = intersection
    
    result1 = solution.Get_Intersection_Hash_Set(listA, listB)
    result2 = solution.Get_Intersection_Two_Pointer(listA, listB)
    result3 = solution.Get_Intersection_Length_Diff(listA, listB)
    result4 = solution.Get_Intersection_Stack(listA, listB)
    result5 = solution.Get_Intersection_Reverse(listA, listB)
    
    print(f"Expected intersection at value: 8")
    print(f"Hash Set: {result1.val if result1 else None}")
    print(f"Two Pointer: {result2.val if result2 else None}")
    print(f"Length Diff: {result3.val if result3 else None}")
    print(f"Stack: {result4.val if result4 else None}")
    print(f"Reverse: {result5.val if result5 else None}")
    print("-" * 50)

if __name__ == "__main__":
    Test_Get_Intersection()

