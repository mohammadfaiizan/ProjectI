"""
Problem: Merge K Sorted Linked Lists
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-linked-lists/1

Problem Statement:
Merge K sorted linked lists into one sorted linked list.

Sample Input/Output:
Input: [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
"""

import heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def Merge_K_Lists_Min_Heap(self, lists):
        """
        Min Heap Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        pq = []
        for i, list_node in enumerate(lists):
            if list_node:
                heapq.heappush(pq, (list_node.val, i, list_node))

        dummy = ListNode(0)
        current = dummy

        while pq:
            val, idx, node = heapq.heappop(pq)
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(pq, (node.next.val, idx, node.next))

        return dummy.next

    def Merge_K_Lists_Divide_Conquer(self, lists):
        """
        Divide and Conquer Approach
        Time Complexity: O(n log k)
        Space Complexity: O(1)
        """
        if not lists:
            return None

        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(self.MergeTwoLists(l1, l2))
            lists = merged

        return lists[0]

    def MergeTwoLists(self, l1, l2):
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


def Test_Merge_K_Lists():
    solution = Solution()

    list1 = ListNode(1)
    list1.next = ListNode(4)
    list1.next.next = ListNode(5)

    list2 = ListNode(1)
    list2.next = ListNode(3)
    list2.next.next = ListNode(4)

    list3 = ListNode(2)
    list3.next = ListNode(6)

    lists1 = [list1, list2, list3]
    result1 = solution.Merge_K_Lists_Min_Heap(lists1)
    print("Min Heap Result:", end=" ")
    while result1:
        print(result1.val, end=" ")
        result1 = result1.next
    print()

    list4 = ListNode(1)
    list4.next = ListNode(4)
    list4.next.next = ListNode(5)

    list5 = ListNode(1)
    list5.next = ListNode(3)
    list5.next.next = ListNode(4)

    list6 = ListNode(2)
    list6.next = ListNode(6)

    lists2 = [list4, list5, list6]
    result2 = solution.Merge_K_Lists_Divide_Conquer(lists2)
    print("Divide Conquer Result:", end=" ")
    while result2:
        print(result2.val, end=" ")
        result2 = result2.next
    print()


if __name__ == "__main__":
    Test_Merge_K_Lists()
