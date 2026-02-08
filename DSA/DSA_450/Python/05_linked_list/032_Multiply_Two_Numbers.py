"""
Problem: Multiply Two Numbers Represented by Linked Lists
URL: https://practice.geeksforgeeks.org/problems/multiply-two-linked-lists/1

Problem Statement:
Given two numbers represented by linked lists, multiply them and return the result as a number.

Sample Input/Output:
Input: First List: 3->2->1 (represents 123)
       Second List: 2->1 (represents 12)
Output: 1476
Explanation: 123 * 12 = 1476
"""

class ListNode:
    def __init__(self, x):
        self.data = x
        self.next = None

def Create_List(arr):
    if not arr:
        return None
    head = ListNode(arr[0])
    current = head
    for i in range(1, len(arr)):
        current.next = ListNode(arr[i])
        current = current.next
    return head

def List_To_Array(head):
    result = []
    while head:
        result.append(head.data)
        head = head.next
    return result

def Print_List(head):
    arr = List_To_Array(head)
    print(" -> ".join(map(str, arr)) + " -> NULL")

class Solution:
    def Multiply_Two_Numbers_Modular(self, first, second):
        """
        Convert to numbers using modular arithmetic to avoid overflow
        Time Complexity: O(m + n) where m and n are lengths
        Space Complexity: O(1)
        """
        MOD = 1000000007
        num1 = 0
        num2 = 0
        current = first
        while current:
            num1 = (num1 * 10 + current.data) % MOD
            current = current.next
        current = second
        while current:
            num2 = (num2 * 10 + current.data) % MOD
            current = current.next
        return (num1 * num2) % MOD

    def Multiply_Two_Numbers_Build_Then_Multiply(self, first, second):
        """
        Build numbers then multiply directly
        Time Complexity: O(m + n)
        Space Complexity: O(1)
        """
        num1 = 0
        num2 = 0
        current = first
        while current:
            num1 = num1 * 10 + current.data
            current = current.next
        current = second
        while current:
            num2 = num2 * 10 + current.data
            current = current.next
        return num1 * num2

def Test_Multiply_Two_Numbers():
    solution = Solution()
    
    list1 = Create_List([3, 2, 1])
    list2 = Create_List([2, 1])
    result1 = solution.Multiply_Two_Numbers_Modular(list1, list2)
    print(f"Test 1 Modular: {result1}")
    
    list3 = Create_List([9, 9, 9])
    list4 = Create_List([1, 1])
    result2 = solution.Multiply_Two_Numbers_Build_Then_Multiply(list3, list4)
    print(f"Test 2 Build Multiply: {result2}")
    
    list5 = Create_List([1])
    list6 = Create_List([5])
    result3 = solution.Multiply_Two_Numbers_Modular(list5, list6)
    print(f"Test 3 Single Digit: {result3}")

if __name__ == "__main__":
    Test_Multiply_Two_Numbers()
