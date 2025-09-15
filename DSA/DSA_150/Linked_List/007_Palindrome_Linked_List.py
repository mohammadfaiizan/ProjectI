"""
Problem: Palindrome Linked List
URL: https://leetcode.com/problems/palindrome-linked-list/

Problem Statement:
Given the head of a singly linked list, return true if it is a palindrome or false otherwise.

Sample Input/Output:
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
"""

from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def Is_Palindrome_Array_Storage(self, head: Optional[ListNode]) -> bool:
        """
        Array Storage - Convert to array and check palindrome
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        return values == values[::-1]
    
    def Is_Palindrome_Reverse_Half_Optimal(self, head: Optional[ListNode]) -> bool:
        """
        Reverse Half Optimal - Find middle and reverse second half
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if not head or not head.next:
            return True
        
        def Find_Middle(node: Optional[ListNode]) -> Optional[ListNode]:
            slow = fast = node
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        def Reverse_List(node: Optional[ListNode]) -> Optional[ListNode]:
            prev = None
            while node:
                next_temp = node.next
                node.next = prev
                prev = node
                node = next_temp
            return prev
        
        middle = Find_Middle(head)
        second_half = Reverse_List(middle.next)
        
        first_half = head
        while second_half:
            if first_half.val != second_half.val:
                return False
            first_half = first_half.next
            second_half = second_half.next
        
        return True
    
    def Is_Palindrome_Stack_Half(self, head: Optional[ListNode]) -> bool:
        """
        Stack Half - Use stack for first half comparison
        Time Complexity: O(n)
        Space Complexity: O(n/2)
        """
        if not head or not head.next:
            return True
        
        slow = fast = head
        stack = []
        
        while fast and fast.next:
            stack.append(slow.val)
            slow = slow.next
            fast = fast.next.next
        
        if fast:
            slow = slow.next
        
        while slow:
            if stack.pop() != slow.val:
                return False
            slow = slow.next
        
        return True
    
    def Is_Palindrome_Recursive(self, head: Optional[ListNode]) -> bool:
        """
        Recursive - Use recursion to compare from both ends
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        self.front_pointer = head
        
        def Recursively_Check(current_node: Optional[ListNode]) -> bool:
            if current_node:
                if not Recursively_Check(current_node.next):
                    return False
                if self.front_pointer.val != current_node.val:
                    return False
                self.front_pointer = self.front_pointer.next
            return True
        
        return Recursively_Check(head)
    
    def Is_Palindrome_Two_Pointers_Array(self, head: Optional[ListNode]) -> bool:
        """
        Two Pointers Array - Use two pointers on converted array
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        values = []
        current = head
        
        while current:
            values.append(current.val)
            current = current.next
        
        left, right = 0, len(values) - 1
        
        while left < right:
            if values[left] != values[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    def Is_Palindrome_Length_Based(self, head: Optional[ListNode]) -> bool:
        """
        Length Based - Calculate length and use indices
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        def Get_Length(node: Optional[ListNode]) -> int:
            length = 0
            while node:
                length += 1
                node = node.next
            return length
        
        def Get_Node_At_Index(node: Optional[ListNode], index: int) -> Optional[ListNode]:
            for _ in range(index):
                node = node.next
            return node
        
        length = Get_Length(head)
        
        for i in range(length // 2):
            left_node = Get_Node_At_Index(head, i)
            right_node = Get_Node_At_Index(head, length - 1 - i)
            
            if left_node.val != right_node.val:
                return False
        
        return True

def Create_Linked_List(values):
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    
    return head

def Test_Is_Palindrome():
    solution = Solution()
    
    test_cases = [
        ([1,2,2,1], True),
        ([1,2], False),
        ([1], True),
        ([1,2,3,2,1], True),
        ([1,2,3,4,5], False),
        ([1,0,1], True)
    ]
    
    methods = [
        ("Array Storage", solution.Is_Palindrome_Array_Storage),
        ("Reverse Half Optimal", solution.Is_Palindrome_Reverse_Half_Optimal),
        ("Stack Half", solution.Is_Palindrome_Stack_Half),
        ("Recursive", solution.Is_Palindrome_Recursive),
        ("Two Pointers Array", solution.Is_Palindrome_Two_Pointers_Array),
        ("Length Based", solution.Is_Palindrome_Length_Based)
    ]
    
    for values, expected in test_cases:
        print(f"Input: {values}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            head = Create_Linked_List(values)
            result = method(head)
            print(f"{method_name}: {result}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Palindrome()
