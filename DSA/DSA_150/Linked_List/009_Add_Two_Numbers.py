"""
Problem: Add Two Numbers
URL: https://leetcode.com/problems/add-two-numbers/

Problem Statement:
You are given two non-empty linked lists representing two non-negative integers. 
The digits are stored in reverse order, and each of their nodes contains a single digit. 
Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Sample Input/Output:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Input: l1 = [0], l2 = [0]
Output: [0]

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Add_Two_Numbers_Iterative_Optimal(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Iterative Optimal - Process digits with carry
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        dummy = ListNode(0)
        current = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            current.next = ListNode(digit)
            current = current.next
            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        
        return dummy.next
    
    def Add_Two_Numbers_Recursive(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Recursive - Add numbers using recursion
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        def Add_Helper(node1: Optional[ListNode], node2: Optional[ListNode], carry: int) -> Optional[ListNode]:
            if not node1 and not node2 and carry == 0:
                return None
            
            val1 = node1.val if node1 else 0
            val2 = node2.val if node2 else 0
            
            total = val1 + val2 + carry
            new_carry = total // 10
            digit = total % 10
            
            result = ListNode(digit)
            
            next1 = node1.next if node1 else None
            next2 = node2.next if node2 else None
            
            result.next = Add_Helper(next1, next2, new_carry)
            
            return result
        
        return Add_Helper(l1, l2, 0)
    
    def Add_Two_Numbers_String_Conversion(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        String Conversion - Convert to numbers, add, convert back
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        def List_To_Number(head: Optional[ListNode]) -> int:
            result = 0
            multiplier = 1
            
            while head:
                result += head.val * multiplier
                multiplier *= 10
                head = head.next
            
            return result
        
        def Number_To_List(num: int) -> Optional[ListNode]:
            if num == 0:
                return ListNode(0)
            
            dummy = ListNode(0)
            current = dummy
            
            while num > 0:
                digit = num % 10
                current.next = ListNode(digit)
                current = current.next
                num //= 10
            
            return dummy.next
        
        num1 = List_To_Number(l1)
        num2 = List_To_Number(l2)
        total = num1 + num2
        
        return Number_To_List(total)
    
    def Add_Two_Numbers_In_Place_Modification(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        In Place Modification - Modify l1 to store result
        Time Complexity: O(max(m, n))
        Space Complexity: O(1)
        """
        head = l1
        carry = 0
        prev = None
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            if l1:
                l1.val = digit
                prev = l1
                l1 = l1.next
            else:
                prev.next = ListNode(digit)
                prev = prev.next
            
            l2 = l2.next if l2 else None
        
        return head
    
    def Add_Two_Numbers_Stack_Based(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        """
        Stack Based - Use stacks to reverse process
        Time Complexity: O(max(m, n))
        Space Complexity: O(max(m, n))
        """
        stack1, stack2 = [], []
        
        while l1:
            stack1.append(l1.val)
            l1 = l1.next
        
        while l2:
            stack2.append(l2.val)
            l2 = l2.next
        
        result_stack = []
        carry = 0
        
        while stack1 or stack2 or carry:
            val1 = stack1.pop() if stack1 else 0
            val2 = stack2.pop() if stack2 else 0
            
            total = val1 + val2 + carry
            carry = total // 10
            digit = total % 10
            
            result_stack.append(digit)
        
        dummy = ListNode(0)
        current = dummy
        
        for digit in result_stack:
            current.next = ListNode(digit)
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

def Test_Add_Two_Numbers():
    solution = Solution()
    
    test_cases = [
        ([2,4,3], [5,6,4], [7,0,8]),
        ([0], [0], [0]),
        ([9,9,9,9,9,9,9], [9,9,9,9], [8,9,9,9,0,0,0,1]),
        ([1], [9,9], [0,0,1]),
        ([5], [5], [0,1])
    ]
    
    methods = [
        ("Iterative Optimal", solution.Add_Two_Numbers_Iterative_Optimal),
        ("Recursive", solution.Add_Two_Numbers_Recursive),
        ("String Conversion", solution.Add_Two_Numbers_String_Conversion),
        ("Stack Based", solution.Add_Two_Numbers_Stack_Based)
    ]
    
    for l1_vals, l2_vals, expected in test_cases:
        print(f"L1: {l1_vals}, L2: {l2_vals}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            l1 = Create_Linked_List(l1_vals)
            l2 = Create_Linked_List(l2_vals)
            result_head = method(l1, l2)
            result = Linked_List_To_Array(result_head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Add_Two_Numbers()
