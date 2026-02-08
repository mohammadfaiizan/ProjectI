"""
Problem: Union and Intersection of Two Arrays
URL: https://practice.geeksforgeeks.org/problems/union-of-two-arrays3538/1

Problem Statement:
Given two arrays a[] and b[], find the number of elements in the union and intersection.
Union: Set of all distinct elements from both arrays.
Intersection: Set of all elements common to both arrays.

Sample Input/Output:
Input: a = [1, 2, 3, 4, 5], b = [1, 2, 3]
Output: Union = 5, Intersection = 3

Input: a = [85, 25, 1, 32, 54, 6], b = [85, 2]
Output: Union = 7, Intersection = 1
"""


class Solution:
    def Union_Set_Optimal(self, a, b):
        """
        Set Based Union - Insert all elements into set
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        """
        s = set()
        for x in a:
            s.add(x)
        for x in b:
            s.add(x)
        return len(s)

    def Intersection_Sorting(self, a, b):
        """
        Sorting + Two Pointers - Sort both and merge to find common
        Time Complexity: O(m log m + n log n)
        Space Complexity: O(1)
        """
        a_sorted = sorted(a)
        b_sorted = sorted(b)
        i = j = count = 0
        while i < len(a_sorted) and j < len(b_sorted):
            if a_sorted[i] == b_sorted[j]:
                count += 1
                i += 1
                j += 1
            elif a_sorted[i] < b_sorted[j]:
                i += 1
            else:
                j += 1
        return count

    def Intersection_Hashing_Optimal(self, a, b):
        """
        Hashing Approach - Use map to count and find common
        Time Complexity: O(m + n)
        Space Complexity: O(min(m, n))
        """
        freq = {}
        for x in a:
            freq[x] = freq.get(x, 0) + 1
        count = 0
        for x in b:
            if freq.get(x, 0) > 0:
                count += 1
                freq[x] -= 1
        return count


def Test_Union_And_Intersection():
    solution = Solution()

    class TestCase:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    test_cases = [
        TestCase([1, 2, 3, 4, 5], [1, 2, 3]),
        TestCase([85, 25, 1, 32, 54, 6], [85, 2]),
        TestCase([1, 2, 3], [4, 5, 6]),
        TestCase([1, 1, 1], [1, 1])
    ]

    for tc in test_cases:
        print(f"A: {tc.a}, B: {tc.b}")

        print("Union (Set):", solution.Union_Set_Optimal(tc.a, tc.b))
        print("Intersection (Sorting):", solution.Intersection_Sorting(tc.a, tc.b))
        print("Intersection (Hashing):", solution.Intersection_Hashing_Optimal(tc.a, tc.b))

        print("-" * 50)


if __name__ == "__main__":
    Test_Union_And_Intersection()
