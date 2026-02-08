"""
Problem: K Largest Elements in Array
URL: https://practice.geeksforgeeks.org/problems/k-largest-elements3736/1

Problem Statement:
Find K largest elements from an unsorted array.

Sample Input/Output:
Input: [1,23,12,9,30,2,50], k=3
Output: [50,30,23]
"""

import heapq


class Solution:
    def K_Largest_Min_Heap(self, arr, k):
        """
        Min Heap of Size K
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        min_heap = []

        for num in arr:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            elif num > min_heap[0]:
                heapq.heapreplace(min_heap, num)

        result = []
        while min_heap:
            result.append(heapq.heappop(min_heap))

        result.reverse()
        return result

    def K_Largest_Sort(self, arr, k):
        """
        Sort Descending and Take First K
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        sorted_arr = sorted(arr, reverse=True)
        return sorted_arr[:k]


def Test_K_Largest():
    solution = Solution()

    arr1 = [1, 23, 12, 9, 30, 2, 50]
    k1 = 3

    print("Array:", arr1, ", k =", k1)

    res1 = solution.K_Largest_Min_Heap(arr1, k1)
    print("Min Heap Result:", res1)

    res2 = solution.K_Largest_Sort(arr1, k1)
    print("Sort Result:", res2)

    arr2 = [12, 5, 787, 1, 23]
    k2 = 2

    print("\nArray:", arr2, ", k =", k2)

    res3 = solution.K_Largest_Min_Heap(arr2, k2)
    print("Min Heap Result:", res3)

    arr3 = [7, 10, 4, 3, 20, 15]
    k3 = 3

    print("\nArray:", arr3, ", k =", k3)

    res4 = solution.K_Largest_Min_Heap(arr3, k3)
    print("Min Heap Result:", res4)


if __name__ == "__main__":
    Test_K_Largest()
