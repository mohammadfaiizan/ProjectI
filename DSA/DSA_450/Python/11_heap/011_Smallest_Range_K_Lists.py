"""
Problem: Smallest Range Covering Elements from K Lists
URL: https://practice.geeksforgeeks.org/problems/find-smallest-range-containing-elements-from-k-lists/1

Problem Statement:
Given K sorted lists, find the smallest range [a, b] such that at least one element from each list falls in the range.

Sample Input/Output:
Input: [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
Output: [20,24]
"""

import heapq


class Solution:
    def Smallest_Range_Min_Heap(self, nums):
        """
        Min Heap Approach
        Time Complexity: O(n*k log k)
        Space Complexity: O(k)
        """
        k = len(nums)
        pq = []

        max_val = float('-inf')
        for i in range(k):
            heapq.heappush(pq, (nums[i][0], i, 0))
            max_val = max(max_val, nums[i][0])

        result = [pq[0][0], max_val]
        min_range = max_val - pq[0][0]

        while True:
            val, row, col = heapq.heappop(pq)

            if col + 1 < len(nums[row]):
                next_val = nums[row][col + 1]
                heapq.heappush(pq, (next_val, row, col + 1))
                max_val = max(max_val, next_val)

                current_range = max_val - pq[0][0]
                if current_range < min_range:
                    min_range = current_range
                    result = [pq[0][0], max_val]
            else:
                break

        return result

    def Smallest_Range_Pointers(self, nums):
        """
        Pointer-based Approach
        Time Complexity: O(n*k^2)
        Space Complexity: O(k)
        """
        k = len(nums)
        pointers = [0] * k
        result = [0, float('inf')]
        min_range = float('inf')

        while True:
            min_val = float('inf')
            max_val = float('-inf')
            min_idx = -1

            for i in range(k):
                if pointers[i] < len(nums[i]):
                    if nums[i][pointers[i]] < min_val:
                        min_val = nums[i][pointers[i]]
                        min_idx = i
                    max_val = max(max_val, nums[i][pointers[i]])

            if min_idx == -1:
                break

            current_range = max_val - min_val
            if current_range < min_range:
                min_range = current_range
                result = [min_val, max_val]

            pointers[min_idx] += 1

        return result


def Test_Smallest_Range():
    solution = Solution()

    nums1 = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
    result1 = solution.Smallest_Range_Min_Heap(nums1)
    print("Min Heap Result:", result1)

    nums2 = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
    result2 = solution.Smallest_Range_Pointers(nums2)
    print("Pointers Result:", result2)

    nums3 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    result3 = solution.Smallest_Range_Min_Heap(nums3)
    print("Test 2 Result:", result3)


if __name__ == "__main__":
    Test_Smallest_Range()
