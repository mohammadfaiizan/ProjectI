"""
Problem: Count Pairs with Given Sum
URL: https://practice.geeksforgeeks.org/problems/count-pairs-with-given-sum5022/1

Problem Statement:
Given an array of N integers, and an integer K, find the number of pairs of elements
in the array whose sum is equal to K.

Sample Input/Output:
Input: arr = [1, 5, 7, -1, 5], K = 6
Output: 3
Explanation: Pairs are (1,5), (7,-1), (1,5).

Input: arr = [1, 1, 1, 1], K = 2
Output: 6
Explanation: All C(4,2) = 6 pairs give sum 2.
"""


class Solution:
    def Count_Pairs_Hashing_Optimal(self, arr, k):
        """
        Hashing Approach - Count complements as we iterate
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        freq = {}
        count = 0
        for x in arr:
            count += freq.get(k - x, 0)
            freq[x] = freq.get(x, 0) + 1
        return count

    def Count_Pairs_Brute_Force(self, arr, k):
        """
        Brute Force - Check all pairs
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] + arr[j] == k:
                    count += 1
        return count


def Test_Count_Pairs_With_Given_Sum():
    solution = Solution()

    class TestCase:
        def __init__(self, arr, k, expected):
            self.arr = arr
            self.k = k
            self.expected = expected

    test_cases = [
        TestCase([1, 5, 7, -1, 5], 6, 3),
        TestCase([1, 1, 1, 1], 2, 6),
        TestCase([10, 12, 10, 15, -1, 7, 6, 5, 4, 2, 1, 1, 1], 11, 9),
        TestCase([1, 2, 3, 4], 10, 0)
    ]

    for tc in test_cases:
        print(f"Array: {tc.arr}, K={tc.k}, Expected={tc.expected}")

        print("Hashing:", solution.Count_Pairs_Hashing_Optimal(tc.arr, tc.k))
        print("Brute Force:", solution.Count_Pairs_Brute_Force(tc.arr, tc.k))

        print("-" * 50)


if __name__ == "__main__":
    Test_Count_Pairs_With_Given_Sum()
