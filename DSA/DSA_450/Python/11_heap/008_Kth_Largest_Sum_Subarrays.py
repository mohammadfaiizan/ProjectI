"""
Problem: Kth Largest Sum of Contiguous Subarrays
URL: https://www.geeksforgeeks.org/k-th-largest-sum-contiguous-subarray/

Problem Statement:
Find the Kth largest sum among all contiguous subarrays.

Sample Input/Output:
Input: [20,-5,-1], k=3
Output: 14
"""

import heapq


class Solution:
    def Kth_Largest_Sum_Min_Heap(self, arr, k):
        """
        Min Heap of Size K
        Time Complexity: O(n^2 log k)
        Space Complexity: O(k)
        """
        min_heap = []

        for i in range(len(arr)):
            sum_val = 0
            for j in range(i, len(arr)):
                sum_val += arr[j]

                if len(min_heap) < k:
                    heapq.heappush(min_heap, sum_val)
                elif sum_val > min_heap[0]:
                    heapq.heapreplace(min_heap, sum_val)

        return min_heap[0]

    def Kth_Largest_Sum_Sort(self, arr, k):
        """
        Store All Sums and Sort
        Time Complexity: O(n^2 log n)
        Space Complexity: O(n^2)
        """
        sums = []

        for i in range(len(arr)):
            sum_val = 0
            for j in range(i, len(arr)):
                sum_val += arr[j]
                sums.append(sum_val)

        sums.sort(reverse=True)
        return sums[k - 1]


def Test_Kth_Largest_Sum():
    solution = Solution()

    arr1 = [20, -5, -1]
    k1 = 3

    print("Array:", arr1, ", k =", k1)

    result1 = solution.Kth_Largest_Sum_Min_Heap(arr1, k1)
    print("Min Heap Result:", result1)

    result2 = solution.Kth_Largest_Sum_Sort(arr1, k1)
    print("Sort Result:", result2)

    arr2 = [10, -10, 20, -40]
    k2 = 6

    print("\nArray:", arr2, ", k =", k2)

    result3 = solution.Kth_Largest_Sum_Min_Heap(arr2, k2)
    print("Min Heap Result:", result3)

    arr3 = [1, 2, 3, 4]
    k3 = 3

    print("\nArray:", arr3, ", k =", k3)

    result4 = solution.Kth_Largest_Sum_Min_Heap(arr3, k3)
    print("Min Heap Result:", result4)

    arr4 = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    k4 = 2

    print("\nArray:", arr4, ", k =", k4)

    result5 = solution.Kth_Largest_Sum_Min_Heap(arr4, k4)
    print("Min Heap Result:", result5)


if __name__ == "__main__":
    Test_Kth_Largest_Sum()
