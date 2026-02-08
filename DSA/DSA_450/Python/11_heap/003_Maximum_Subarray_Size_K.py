"""
Problem: Maximum of All Subarrays of Size K (Sliding Window Maximum)
URL: https://practice.geeksforgeeks.org/problems/maximum-of-all-subarrays-of-size-k3101/1

Problem Statement:
Given an array and integer K, find the maximum for each contiguous subarray of size K.

Sample Input/Output:
Input: [1,2,3,1,4,5,2,3,6], k=3
Output: [3,3,4,5,5,5,6]
"""

from collections import deque


class Solution:
    def Max_Subarray_K_Deque(self, arr, k):
        """
        Deque Based Approach
        Time Complexity: O(n)
        Space Complexity: O(k)
        """
        result = []
        dq = deque()

        for i in range(len(arr)):
            while dq and dq[0] <= i - k:
                dq.popleft()

            while dq and arr[dq[-1]] <= arr[i]:
                dq.pop()

            dq.append(i)

            if i >= k - 1:
                result.append(arr[dq[0]])

        return result

    def Max_Subarray_K_Heap(self, arr, k):
        """
        Max Heap with Lazy Deletion
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        import heapq
        result = []
        pq = []

        for i in range(len(arr)):
            heapq.heappush(pq, (-arr[i], i))

            if i >= k - 1:
                while pq and pq[0][1] <= i - k:
                    heapq.heappop(pq)
                result.append(-pq[0][0])

        return result

    def Max_Subarray_K_Brute(self, arr, k):
        """
        Brute Force Approach
        Time Complexity: O(n*k)
        Space Complexity: O(1)
        """
        result = []

        for i in range(len(arr) - k + 1):
            max_val = arr[i]
            for j in range(i + 1, i + k):
                max_val = max(max_val, arr[j])
            result.append(max_val)

        return result


def Test_Max_Subarray_K():
    solution = Solution()

    arr1 = [1, 2, 3, 1, 4, 5, 2, 3, 6]
    k1 = 3

    print("Array:", arr1, ", k =", k1)

    res1 = solution.Max_Subarray_K_Deque(arr1, k1)
    print("Deque Result:", res1)

    res2 = solution.Max_Subarray_K_Heap(arr1, k1)
    print("Heap Result:", res2)

    res3 = solution.Max_Subarray_K_Brute(arr1, k1)
    print("Brute Result:", res3)

    arr2 = [8, 5, 10, 7, 9, 4, 15, 12, 90, 13]
    k2 = 4

    print("\nArray:", arr2, ", k =", k2)

    res4 = solution.Max_Subarray_K_Deque(arr2, k2)
    print("Deque Result:", res4)

    arr3 = [1, 3, -1, -3, 5, 3, 6, 7]
    k3 = 3

    print("\nArray:", arr3, ", k =", k3)

    res5 = solution.Max_Subarray_K_Deque(arr3, k3)
    print("Deque Result:", res5)


if __name__ == "__main__":
    Test_Max_Subarray_K()
