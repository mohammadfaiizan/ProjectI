"""
Problem: Add Two Numbers Represented by Linked Lists
URL: https://practice.geeksforgeeks.org/problems/add-two-numbers-represented-by-linked-lists/1

Problem Statement:
Given two numbers represented by two linked lists, write a function that returns the sum list. The sum list is linked list representation of the addition of two input numbers.

Sample Input/Output:
Input: First List: 5->6->3, Second List: 8->4->2
Output: Resultant list: 1->4->0->5
Explanation: 563 + 842 = 1405
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
    print("->".join(map(str, arr)))

def Reverse_List(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

class Solution:
    def Add_Two_Numbers_Reverse(self, first, second):
        """
        Reverse both lists, add with carry, reverse result
        Time Complexity: O(m + n)
        Space Complexity: O(max(m, n))
        """
        first = Reverse_List(first)
        second = Reverse_List(second)
        
        dummy = ListNode(0)
        curr = dummy
        carry = 0
        
        while first or second or carry:
            sum_val = carry
            if first:
                sum_val += first.data
                first = first.next
            if second:
                sum_val += second.data
                second = second.next
            carry = sum_val // 10
            curr.next = ListNode(sum_val % 10)
            curr = curr.next
        
        result = Reverse_List(dummy.next)
        return result
    
    def Add_Recursive_Helper(self, first, second, carry):
        if not first and not second:
            return None
        
        next_node = self.Add_Recursive_Helper(first.next if first else None, second.next if second else None, carry)
        sum_val = (first.data if first else 0) + (second.data if second else 0) + carry[0]
        carry[0] = sum_val // 10
        node = ListNode(sum_val % 10)
        node.next = next_node
        return node
    
    def Add_Two_Numbers_Recursive(self, first, second):
        """
        Recursive approach for same-size lists
        Time Complexity: O(m + n)
        Space Complexity: O(max(m, n))
        """
        len1 = 0
        len2 = 0
        temp1 = first
        temp2 = second
        while temp1:
            len1 += 1
            temp1 = temp1.next
        while temp2:
            len2 += 1
            temp2 = temp2.next
        
        if len1 < len2:
            while len1 < len2:
                node = ListNode(0)
                node.next = first
                first = node
                len1 += 1
        elif len2 < len1:
            while len2 < len1:
                node = ListNode(0)
                node.next = second
                second = node
                len2 += 1
        
        carry = [0]
        result = self.Add_Recursive_Helper(first, second, carry)
        if carry[0] > 0:
            node = ListNode(carry[0])
            node.next = result
            result = node
        return result

def Test_Add_Two_Numbers():
    solution = Solution()
    
    arr1 = [5, 6, 3]
    arr2 = [8, 4, 2]
    first = Create_List(arr1)
    second = Create_List(arr2)
    result1 = solution.Add_Two_Numbers_Reverse(first, second)
    print("Test 1 - Reverse Approach: ", end="")
    Print_List(result1)
    
    first = Create_List(arr1)
    second = Create_List(arr2)
    result2 = solution.Add_Two_Numbers_Recursive(first, second)
    print("Test 1 - Recursive Approach: ", end="")
    Print_List(result2)
    
    arr3 = [9, 9, 9]
    arr4 = [1]
    first = Create_List(arr3)
    second = Create_List(arr4)
    result1 = solution.Add_Two_Numbers_Reverse(first, second)
    print("Test 2 - Reverse Approach: ", end="")
    Print_List(result1)

if __name__ == "__main__":
    Test_Add_Two_Numbers()
