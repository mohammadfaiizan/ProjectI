"""
Problem: Find Pairs with Given Sum in Sorted Doubly Linked List
URL: https://www.geeksforgeeks.org/find-pairs-given-sum-doubly-linked-list/

Problem Statement:
Given a sorted doubly linked list and a target sum, find all pairs of nodes whose sum equals the target.

Sample Input/Output:
Input: List: 1 <-> 2 <-> 4 <-> 5 <-> 6 <-> 8 <-> 9, sum = 7
Output: (1, 6), (2, 5)
Explanation: Pairs that sum to 7
"""

class DLLNode:
    def __init__(self, x):
        self.data = x
        self.next = None
        self.prev = None

def Create_DLL(arr):
    if not arr:
        return None
    head = DLLNode(arr[0])
    curr = head
    for i in range(1, len(arr)):
        curr.next = DLLNode(arr[i])
        curr.next.prev = curr
        curr = curr.next
    return head

def Print_DLL(head):
    curr = head
    result = []
    while curr:
        result.append(str(curr.data))
        curr = curr.next
    print(" ".join(result))

class Solution:
    def Find_Pairs_Two_Pointer(self, head, target):
        """
        Two pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        result = []
        if not head or not head.next:
            return result
        
        left = head
        right = head
        while right.next:
            right = right.next
        
        while left != right and right.next != left:
            sum_val = left.data + right.data
            if sum_val == target:
                result.append((left.data, right.data))
                left = left.next
                right = right.prev
            elif sum_val < target:
                left = left.next
            else:
                right = right.prev
        
        return result
    
    def Find_Pairs_Hashing(self, head, target):
        """
        Hashing approach
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        result = []
        if not head:
            return result
        
        seen = set()
        curr = head
        
        while curr:
            complement = target - curr.data
            if complement in seen:
                result.append((complement, curr.data))
            seen.add(curr.data)
            curr = curr.next
        
        return result

def Test_Pair_With_Given_Sum_DLL():
    solution = Solution()
    
    arr1 = [1, 2, 4, 5, 6, 8, 9]
    head1 = Create_DLL(arr1)
    print("List: ", end="")
    Print_DLL(head1)
    pairs1 = solution.Find_Pairs_Two_Pointer(head1, 7)
    print("Pairs with sum 7 (Two Pointer):", end=" ")
    for p in pairs1:
        print(f"({p[0]}, {p[1]})", end=" ")
    print()
    
    pairs2 = solution.Find_Pairs_Hashing(head1, 7)
    print("Pairs with sum 7 (Hashing):", end=" ")
    for p in pairs2:
        print(f"({p[0]}, {p[1]})", end=" ")
    print()
    
    arr3 = [1, 3, 5, 7]
    head3 = Create_DLL(arr3)
    print("List: ", end="")
    Print_DLL(head3)
    pairs3 = solution.Find_Pairs_Two_Pointer(head3, 8)
    print("Pairs with sum 8:", end=" ")
    for p in pairs3:
        print(f"({p[0]}, {p[1]})", end=" ")
    print()

if __name__ == "__main__":
    Test_Pair_With_Given_Sum_DLL()
