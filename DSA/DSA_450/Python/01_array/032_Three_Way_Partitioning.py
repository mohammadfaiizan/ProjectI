"""
Problem: Three Way Partitioning
URL: https://practice.geeksforgeeks.org/problems/three-way-partitioning/1

Problem Statement:
Given an array and a range [a, b], partition the array such that elements < a come first,
then elements in range [a, b], and finally elements > b.

Sample Input/Output:
Input: arr = [1, 14, 5, 20, 4, 2, 54, 20, 87, 98, 3, 1, 32], a = 14, b = 20
Output: [1, 5, 4, 2, 3, 1, 14, 20, 20, 54, 87, 98, 32] (one possible output)

Input: arr = [1, 2, 3, 3, 4], a = 1, b = 2
Output: [1, 2, 3, 3, 4]
"""


class Solution:
    def Three_Way_Partition_Dutch_Flag_Optimal(self, arr, a, b):
        """
        Dutch National Flag Variant - Three pointer approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        low = mid = 0
        high = len(arr) - 1
        while mid <= high:
            if arr[mid] < a:
                arr[mid], arr[low] = arr[low], arr[mid]
                mid += 1
                low += 1
            elif arr[mid] > b:
                arr[mid], arr[high] = arr[high], arr[mid]
                high -= 1
            else:
                mid += 1


def Test_Three_Way_Partitioning():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, a, b):
            self.arr = arr
            self.a = a
            self.b = b

    test_cases = [
        TestCase([1, 14, 5, 20, 4, 2, 54, 20, 87, 98, 3, 1, 32], 14, 20),
        TestCase([1, 2, 3, 3, 4], 1, 2),
        TestCase([87, 78, 16, 94], 16, 78)
    ]

    for tc in test_cases:
        print(f"Original: {tc.arr}, Range=[{tc.a}, {tc.b}]")

        arr1 = tc.arr.copy()
        solution.Three_Way_Partition_Dutch_Flag_Optimal(arr1, tc.a, tc.b)
        print("Dutch Flag:", arr1)

        print("-" * 50)


if __name__ == "__main__":
    Test_Three_Way_Partitioning()
