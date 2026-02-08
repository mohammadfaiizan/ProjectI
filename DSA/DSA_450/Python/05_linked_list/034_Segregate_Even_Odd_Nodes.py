"""
Problem: Segregate Even and Odd Nodes in a Linked List
URL: https://practice.geeksforgeeks.org/problems/segregate-even-and-odd-nodes-in-a-linked-list5035/1

Problem Statement:
Given a linked list, segregate even and odd nodes such that all even nodes come before all odd nodes.

Sample Input/Output:
Input: 17 -> 15 -> 8 -> 12 -> 10 -> 5 -> 4 -> NULL
Output: 8 -> 12 -> 10 -> 4 -> 17 -> 15 -> 5 -> NULL
Explanation: All even nodes (8, 12, 10, 4) come before odd nodes (17, 15, 5).
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
    def Segregate_Even_Odd_Separate_Merge(self, head):
        """
        Separate into even and odd lists then merge
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if head is None or head.next is None:
            return head
        even_head = ListNode(0)
        odd_head = ListNode(0)
        even = even_head
        odd = odd_head
        current = head
        while current:
            if current.data % 2 == 0:
                even.next = current
                even = even.next
            else:
                odd.next = current
                odd = odd.next
            current = current.next
        even.next = odd_head.next
        odd.next = None
        result = even_head.next
        return result

    def Segregate_Even_Odd_Move_Odd_To_End(self, head):
        """
        Move odd nodes to end while keeping even nodes at front
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if head is None or head.next is None:
            return head
        last = head
        count = 0
        while last.next:
            last = last.next
            count += 1
        current = head
        prev = None
        moved = 0
        while current and moved <= count:
            if current.data % 2 == 1:
                if prev is None:
                    head = current.next
                    last.next = current
                    last = last.next
                    last.next = None
                    current = head
                else:
                    prev.next = current.next
                    last.next = current
                    last = last.next
                    last.next = None
                    current = prev.next
                moved += 1
            else:
                prev = current
                current = current.next
        return head

def Test_Segregate_Even_Odd_Nodes():
    solution = Solution()
    
    list1 = Create_List([17, 15, 8, 12, 10, 5, 4])
    result1 = solution.Segregate_Even_Odd_Separate_Merge(list1)
    print("Test 1 Separate Merge: ", end="")
    Print_List(result1)
    
    list2 = Create_List([1, 3, 5, 7])
    result2 = solution.Segregate_Even_Odd_Move_Odd_To_End(list2)
    print("Test 2 Move Odd: ", end="")
    Print_List(result2)
    
    list3 = Create_List([2, 4, 6, 8])
    result3 = solution.Segregate_Even_Odd_Separate_Merge(list3)
    print("Test 3 All Even: ", end="")
    Print_List(result3)

if __name__ == "__main__":
    Test_Segregate_Even_Odd_Nodes()
