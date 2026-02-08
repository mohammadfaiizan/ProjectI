"""
Problem: Kth Smallest Element
URL: https://practice.geeksforgeeks.org/problems/kth-smallest-element5635/1

Problem Statement:
Given an array arr[] and an integer K where K is smaller than size of array,
the task is to find the Kth smallest element in the given array.

Sample Input/Output:
Input: arr = [7, 10, 4, 3, 20, 15], K = 3
Output: 7
Explanation: 3rd smallest element is 7.

Input: arr = [7, 10, 4, 20, 15], K = 4
Output: 15
Explanation: 4th smallest element is 15.
"""

import heapq


class Solution:
    def Kth_Smallest_Sorting(self, arr, k):
        """
        Sorting Approach - Sort and return kth element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr_sorted = sorted(arr)
        return arr_sorted[k - 1]

    def Kth_Smallest_Min_Heap(self, arr, k):
        """
        Min Heap - Build min heap and extract k times
        Time Complexity: O(n + k log n)
        Space Complexity: O(n)
        """
        min_heap = arr.copy()
        heapq.heapify(min_heap)
        for _ in range(k - 1):
            heapq.heappop(min_heap)
        return min_heap[0]

    def Kth_Smallest_Max_Heap_Optimal(self, arr, k):
        """
        Max Heap of Size K - Maintain heap of k smallest seen so far
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        max_heap = []
        for x in arr:
            heapq.heappush(max_heap, -x)
            if len(max_heap) > k:
                heapq.heappop(max_heap)
        return -max_heap[0]

    def Kth_Smallest_Quickselect(self, arr, k):
        """
        Quickselect (Hoare's Selection) - Partition-based selection
        Time Complexity: O(n) average, O(n^2) worst
        Space Complexity: O(log n) - recursion stack
        """
        temp = arr.copy()
        return self.Quickselect_Helper(temp, 0, len(temp) - 1, k - 1)

    def Kth_Smallest_STL_Nth_Element(self, arr, k):
        """
        STL nth_element - Uses introselect algorithm
        Time Complexity: O(n) average
        Space Complexity: O(1)
        """
        temp = arr.copy()
        temp.sort()
        return temp[k - 1]

    def Quickselect_Helper(self, arr, low, high, k):
        if low <= high:
            pivot = self.Partition(arr, low, high)
            if pivot == k:
                return arr[pivot]
            if pivot > k:
                return self.Quickselect_Helper(arr, low, pivot - 1, k)
            return self.Quickselect_Helper(arr, pivot + 1, high, k)
        return -1

    def Partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1


def Test_Kth_Smallest_Element():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, k, expected):
            self.arr = arr
            self.k = k
            self.expected = expected

    test_cases = [
        TestCase([7, 10, 4, 3, 20, 15], 3, 7),
        TestCase([7, 10, 4, 20, 15], 4, 15),
        TestCase([1], 1, 1),
        TestCase([12, 3, 5, 7, 19], 2, 5)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, K={tc.k}, Expected={tc.expected}")

        print("Sorting:", solution.Kth_Smallest_Sorting(tc.arr, tc.k))
        print("Min Heap:", solution.Kth_Smallest_Min_Heap(tc.arr, tc.k))
        print("Max Heap:", solution.Kth_Smallest_Max_Heap_Optimal(tc.arr, tc.k))
        print("Quickselect:", solution.Kth_Smallest_Quickselect(tc.arr, tc.k))
        print("STL nth_element:", solution.Kth_Smallest_STL_Nth_Element(tc.arr, tc.k))

        print("-" * 50)


if __name__ == "__main__":
    Test_Kth_Smallest_Element()
