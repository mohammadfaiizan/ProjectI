"""
Problem: Common Elements in Three Sorted Arrays
URL: https://practice.geeksforgeeks.org/problems/common-elements1132/1

Problem Statement:
Given three arrays sorted in increasing order, find the elements that are common in all three.

Sample Input/Output:
Input: A = [1, 5, 10, 20, 40, 80], B = [6, 7, 20, 80, 100], C = [3, 4, 15, 20, 30, 70, 80, 120]
Output: [20, 80]

Input: A = [1, 5, 5], B = [3, 4, 5, 5, 10], C = [5, 5, 10, 20]
Output: [5, 5]
"""


class Solution:
    def Common_Elements_Three_Pointers_Optimal(self, A, B, C):
        """
        Three Pointers - Advance pointer of smallest element
        Time Complexity: O(n1 + n2 + n3)
        Space Complexity: O(1) excluding result
        """
        result = []
        i = j = k = 0
        prev = float('-inf')
        while i < len(A) and j < len(B) and k < len(C):
            if A[i] == B[j] == C[k]:
                if A[i] != prev:
                    result.append(A[i])
                    prev = A[i]
                i += 1
                j += 1
                k += 1
            elif A[i] < B[j]:
                i += 1
            elif B[j] < C[k]:
                j += 1
            else:
                k += 1
        return result

    def Common_Elements_Hashing(self, A, B, C):
        """
        Hashing Approach - Use maps to count occurrences
        Time Complexity: O(n1 + n2 + n3)
        Space Complexity: O(n1 + n2 + n3)
        """
        freqA = {}
        freqB = {}
        for x in A:
            freqA[x] = freqA.get(x, 0) + 1
        for x in B:
            freqB[x] = freqB.get(x, 0) + 1
        result = []
        added = set()
        for x in C:
            if freqA.get(x, 0) > 0 and freqB.get(x, 0) > 0 and x not in added:
                result.append(x)
                added.add(x)
        return result


def Test_Common_Elements():
    solution = Solution()

    class TestCase:
        def __init__(self, A, B, C):
            self.A = A
            self.B = B
            self.C = C

    test_cases = [
        TestCase([1, 5, 10, 20, 40, 80], [6, 7, 20, 80, 100], [3, 4, 15, 20, 30, 70, 80, 120]),
        TestCase([1, 5, 5], [3, 4, 5, 5, 10], [5, 5, 10, 20]),
        TestCase([1, 2, 3], [4, 5, 6], [7, 8, 9])
    ]

    for tc in test_cases:
        print(f"A: {tc.A}, B: {tc.B}, C: {tc.C}")

        r1 = solution.Common_Elements_Three_Pointers_Optimal(tc.A, tc.B, tc.C)
        print("Three Pointers:", r1)

        r2 = solution.Common_Elements_Hashing(tc.A, tc.B, tc.C)
        print("Hashing:", r2)

        print("-" * 50)


if __name__ == "__main__":
    Test_Common_Elements()
