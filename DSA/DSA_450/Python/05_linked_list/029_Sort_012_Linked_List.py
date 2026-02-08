"""
Problem: Sort a Linked List of 0s, 1s, and 2s
URL: https://practice.geeksforgeeks.org/problems/given-a-linked-list-of-0s-1s-and-2s-sort-it/1

Problem Statement:
Given a linked list of 0s, 1s and 2s, sort it.

Sample Input/Output:
Input: 1 -> 1 -> 2 -> 0 -> 2 -> 0 -> 1 -> NULL
Output: 0 -> 0 -> 1 -> 1 -> 1 -> 2 -> 2 -> NULL
Explanation: All 0s come first, then 1s, then 2s.
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
    def Sort_012_Count_Based(self, head):
        """
        Count occurrences then rebuild list
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        count = [0, 0, 0]
        current = head
        while current:
            count[current.data] += 1
            current = current.next
        current = head
        for i in range(3):
            while count[i] > 0:
                current.data = i
                current = current.next
                count[i] -= 1
        return head

    def Sort_012_Three_Dummy_Nodes(self, head):
        """
        Separate into three lists then merge
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        zero_head = ListNode(0)
        one_head = ListNode(0)
        two_head = ListNode(0)
        zero = zero_head
        one = one_head
        two = two_head
        current = head
        while current:
            if current.data == 0:
                zero.next = current
                zero = zero.next
            elif current.data == 1:
                one.next = current
                one = one.next
            else:
                two.next = current
                two = two.next
            current = current.next
        zero.next = one_head.next if one_head.next else two_head.next
        one.next = two_head.next
        two.next = None
        result = zero_head.next
        return result

def Test_Sort_012_Linked_List():
    solution = Solution()
    
    test1 = [1, 1, 2, 0, 2, 0, 1]
    list1 = Create_List(test1)
    result1 = solution.Sort_012_Count_Based(list1)
    print("Test 1 Count Based: ", end="")
    Print_List(result1)
    
    test2 = [2, 1, 2, 1, 1, 2, 0, 2, 0]
    list2 = Create_List(test2)
    result2 = solution.Sort_012_Three_Dummy_Nodes(list2)
    print("Test 2 Three Dummy: ", end="")
    Print_List(result2)
    
    test3 = [2, 2, 1, 1, 0]
    list3 = Create_List(test3)
    result3 = solution.Sort_012_Count_Based(list3)
    print("Test 3 Mixed: ", end="")
    Print_List(result3)

if __name__ == "__main__":
    Test_Sort_012_Linked_List()
