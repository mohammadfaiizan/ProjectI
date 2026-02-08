"""
Problem: Merge K Sorted Arrays
URL: https://practice.geeksforgeeks.org/problems/merge-k-sorted-arrays/1

Problem Statement:
Merge K sorted arrays into a single sorted array.

Sample Input/Output:
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,4,5,6,7,8,9]
"""

import heapq


class Solution:
    def Merge_K_Sorted_Min_Heap(self, arrays):
        """
        Min Heap with {value, array_idx, element_idx}
        Time Complexity: O(n*k log k)
        Space Complexity: O(k)
        """
        result = []
        min_heap = []

        for i in range(len(arrays)):
            if arrays[i]:
                heapq.heappush(min_heap, (arrays[i][0], i, 0))

        while min_heap:
            value, array_idx, element_idx = heapq.heappop(min_heap)
            result.append(value)

            if element_idx + 1 < len(arrays[array_idx]):
                heapq.heappush(min_heap, (arrays[array_idx][element_idx + 1], array_idx, element_idx + 1))

        return result

    def Merge_K_Sorted_Divide_Conquer(self, arrays):
        """
        Divide and Conquer (Merge Two at a Time)
        Time Complexity: O(n*k log k)
        Space Complexity: O(n*k)
        """
        if not arrays:
            return []
        if len(arrays) == 1:
            return arrays[0]

        return self.Merge_Helper(arrays, 0, len(arrays) - 1)

    def Merge_Helper(self, arrays, left, right):
        if left == right:
            return arrays[left]

        mid = left + (right - left) // 2
        left_merged = self.Merge_Helper(arrays, left, mid)
        right_merged = self.Merge_Helper(arrays, mid + 1, right)

        return self.Merge_Two_Arrays(left_merged, right_merged)

    def Merge_Two_Arrays(self, arr1, arr2):
        result = []
        i, j = 0, 0

        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1

        while i < len(arr1):
            result.append(arr1[i])
            i += 1

        while j < len(arr2):
            result.append(arr2[j])
            j += 1

        return result


def Test_Merge_K_Sorted():
    solution = Solution()

    arrays1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    print("Input Arrays:")
    for i in range(len(arrays1)):
        print("Array", i, ":", arrays1[i])

    res1 = solution.Merge_K_Sorted_Min_Heap(arrays1)
    print("Min Heap Result:", res1)

    res2 = solution.Merge_K_Sorted_Divide_Conquer(arrays1)
    print("Divide Conquer Result:", res2)

    arrays2 = [[1, 3, 5, 7], [2, 4, 6, 8], [0, 9, 10, 11]]

    print("\nInput Arrays:")
    for i in range(len(arrays2)):
        print("Array", i, ":", arrays2[i])

    res3 = solution.Merge_K_Sorted_Min_Heap(arrays2)
    print("Min Heap Result:", res3)

    arrays3 = [[1, 4, 7], [2, 5, 8], [3, 6, 9], [10, 11, 12]]

    print("\nInput Arrays:")
    for i in range(len(arrays3)):
        print("Array", i, ":", arrays3[i])

    res4 = solution.Merge_K_Sorted_Min_Heap(arrays3)
    print("Min Heap Result:", res4)


if __name__ == "__main__":
    Test_Merge_K_Sorted()
