"""
Problem: Kth Smallest and Kth Largest Element in Unsorted Array
URL: https://practice.geeksforgeeks.org/problems/kth-smallest-element5635/1

Problem Statement:
Find the Kth smallest and Kth largest element in an unsorted array.

Sample Input/Output:
Input: [7,10,4,3,20,15], k=3
Output: kth smallest=7, kth largest=10
"""

import heapq


class Solution:
    def Kth_Element_Heap(self, arr, k, smallest):
        """
        Heap Based Approach
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        if smallest:
            max_heap = []
            for num in arr:
                if len(max_heap) < k:
                    heapq.heappush(max_heap, -num)
                elif num < -max_heap[0]:
                    heapq.heapreplace(max_heap, -num)
            return -max_heap[0]
        else:
            min_heap = []
            for num in arr:
                if len(min_heap) < k:
                    heapq.heappush(min_heap, num)
                elif num > min_heap[0]:
                    heapq.heapreplace(min_heap, num)
            return min_heap[0]

    def Kth_Element_QuickSelect(self, arr, k, smallest):
        """
        Randomized QuickSelect
        Time Complexity: O(n) average, O(n^2) worst case
        Space Complexity: O(1)
        """
        arr_copy = arr.copy()
        if smallest:
            return self.QuickSelect_Smallest(arr_copy, 0, len(arr_copy) - 1, k - 1)
        else:
            return self.QuickSelect_Largest(arr_copy, 0, len(arr_copy) - 1, k - 1)

    def Kth_Element_Sort(self, arr, k, smallest):
        """
        Sort and Return Kth Element
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        sorted_arr = sorted(arr)
        if smallest:
            return sorted_arr[k - 1]
        else:
            return sorted_arr[len(sorted_arr) - k]

    def Partition(self, arr, low, high):
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def QuickSelect_Smallest(self, arr, low, high, k):
        if low == high:
            return arr[low]

        pivot_idx = self.Partition(arr, low, high)

        if pivot_idx == k:
            return arr[pivot_idx]
        elif pivot_idx > k:
            return self.QuickSelect_Smallest(arr, low, pivot_idx - 1, k)
        else:
            return self.QuickSelect_Smallest(arr, pivot_idx + 1, high, k)

    def QuickSelect_Largest(self, arr, low, high, k):
        target = len(arr) - 1 - k
        return self.QuickSelect_Smallest(arr, low, high, target)


def Test_Kth_Element():
    solution = Solution()

    arr1 = [7, 10, 4, 3, 20, 15]
    k1 = 3

    print("Array:", arr1, ", k =", k1)

    kth_smallest_heap = solution.Kth_Element_Heap(arr1, k1, True)
    kth_largest_heap = solution.Kth_Element_Heap(arr1, k1, False)
    print("Heap - Kth Smallest:", kth_smallest_heap, ", Kth Largest:", kth_largest_heap)

    kth_smallest_qs = solution.Kth_Element_QuickSelect(arr1, k1, True)
    kth_largest_qs = solution.Kth_Element_QuickSelect(arr1, k1, False)
    print("QuickSelect - Kth Smallest:", kth_smallest_qs, ", Kth Largest:", kth_largest_qs)

    kth_smallest_sort = solution.Kth_Element_Sort(arr1, k1, True)
    kth_largest_sort = solution.Kth_Element_Sort(arr1, k1, False)
    print("Sort - Kth Smallest:", kth_smallest_sort, ", Kth Largest:", kth_largest_sort)

    arr2 = [3, 2, 1, 5, 6, 4]
    k2 = 2

    print("\nArray:", arr2, ", k =", k2)

    kth_smallest2 = solution.Kth_Element_Heap(arr2, k2, True)
    kth_largest2 = solution.Kth_Element_Heap(arr2, k2, False)
    print("Kth Smallest:", kth_smallest2, ", Kth Largest:", kth_largest2)

    arr3 = [1, 5, 2, 8, 3, 9, 4]
    k3 = 4

    print("\nArray:", arr3, ", k =", k3)

    kth_smallest3 = solution.Kth_Element_Heap(arr3, k3, True)
    kth_largest3 = solution.Kth_Element_Heap(arr3, k3, False)
    print("Kth Smallest:", kth_smallest3, ", Kth Largest:", kth_largest3)


if __name__ == "__main__":
    Test_Kth_Element()
