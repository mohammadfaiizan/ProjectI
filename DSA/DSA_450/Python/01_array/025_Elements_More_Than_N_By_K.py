"""
Problem: Find All Elements Appearing More Than N/K Times
URL: https://www.geeksforgeeks.org/given-an-array-of-of-size-n-finds-all-the-elements-that-appear-more-than-nk-times/

Problem Statement:
Given an array of size n, find all elements that appear more than n/k times.

Sample Input/Output:
Input: arr = [3, 1, 2, 2, 1, 2, 3, 3], K = 4
Output: [2, 3]
Explanation: Elements 2 and 3 appear more than 8/4 = 2 times.

Input: arr = [9, 8, 7, 9, 2, 9, 7], K = 3
Output: [9]
Explanation: Only 9 appears more than 7/3 = 2 times.
"""


class Solution:
    def Elements_N_By_K_Hashing_Optimal(self, arr, k):
        """
        Hashing Approach - Count frequency using map
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = {}
        for x in arr:
            freq[x] = freq.get(x, 0) + 1
        threshold = len(arr) // k
        result = []
        for val, count in freq.items():
            if count > threshold:
                result.append(val)
        return result

    def Elements_N_By_K_Sorting(self, arr, k):
        """
        Sorting Approach - Sort and check consecutive count
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        threshold = n // k
        result = []
        i = 0
        while i < n:
            count = 1
            while i + count < n and arr_sorted[i + count] == arr_sorted[i]:
                count += 1
            if count > threshold:
                result.append(arr_sorted[i])
            i += count
        return result


def Test_Elements_More_Than_N_By_K():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, k):
            self.arr = arr
            self.k = k

    test_cases = [
        TestCase([3, 1, 2, 2, 1, 2, 3, 3], 4),
        TestCase([9, 8, 7, 9, 2, 9, 7], 3),
        TestCase([1, 1, 2, 2, 3, 5, 4, 2, 2, 3, 1, 1, 1], 3)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, K={tc.k}")

        r1 = solution.Elements_N_By_K_Hashing_Optimal(tc.arr, tc.k)
        print("Hashing:", r1)

        r2 = solution.Elements_N_By_K_Sorting(tc.arr, tc.k)
        print("Sorting:", r2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Elements_More_Than_N_By_K()
