"""
Problem: Find Median in a Stream of Integers
URL: https://practice.geeksforgeeks.org/problems/find-median-in-a-stream-1587115620/1

Problem Statement:
Design a data structure that supports adding integers and finding median efficiently.

Sample Input/Output:
Input: stream [5,15,1,3,2,8,7,9]
Output: medians after each insertion
"""

import heapq


class Solution:
    def __init__(self):
        self.max_heap = []
        self.min_heap = []

    def Insert_Median(self, num):
        """
        Two Heaps Approach
        Time Complexity: O(log n)
        Space Complexity: O(n)
        """
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def Find_Median(self):
        """
        Find Median
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        elif len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return self.min_heap[0]


def Test_Median_Stream():
    solution = Solution()

    stream1 = [5, 15, 1, 3, 2, 8, 7, 9]
    print("Stream 1:", end=" ")
    for num in stream1:
        solution.Insert_Median(num)
        print(num, "-> Median:", solution.Find_Median(), end=" ")
    print()

    solution2 = Solution()
    stream2 = [1, 2, 3, 4, 5]
    print("Stream 2:", end=" ")
    for num in stream2:
        solution2.Insert_Median(num)
        print(num, "-> Median:", solution2.Find_Median(), end=" ")
    print()

    solution3 = Solution()
    stream3 = [10, 20, 30]
    print("Stream 3:", end=" ")
    for num in stream3:
        solution3.Insert_Median(num)
        print(num, "-> Median:", solution3.Find_Median(), end=" ")
    print()


if __name__ == "__main__":
    Test_Median_Stream()
