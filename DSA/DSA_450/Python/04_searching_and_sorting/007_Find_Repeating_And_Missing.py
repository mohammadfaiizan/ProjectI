"""
Problem: Find Missing and Repeating
URL: https://practice.geeksforgeeks.org/problems/find-missing-and-repeating2512/1

Problem Statement:
Given an unsorted array Arr of size N of positive integers. One number 'A' from set {1, 2, …N} is missing and one number 'B' occurs twice in array. Find these two numbers.

Sample Input/Output:
Input: N = 2, Arr[] = {2, 2}
Output: 2 1

Input: N = 3, Arr[] = {1, 3, 3}
Output: 3 2
"""


class Solution:
    def Find_Repeating_Missing_Count_Array(self, arr, n):
        """
        Using count array to track occurrences
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        count = [0] * (n + 1)
        repeating = -1
        missing = -1
        
        for i in range(n):
            count[arr[i]] += 1
        
        for i in range(1, n + 1):
            if count[i] == 0:
                missing = i
            elif count[i] == 2:
                repeating = i
        
        return (repeating, missing)

    def Find_Repeating_Missing_Sign_Marking(self, arr, n):
        """
        Using sign marking to identify repeating and missing
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        repeating = -1
        missing = -1
        
        for i in range(n):
            index = abs(arr[i]) - 1
            if arr[index] < 0:
                repeating = abs(arr[i])
            else:
                arr[index] = -arr[index]
        
        for i in range(n):
            if arr[i] > 0:
                missing = i + 1
                break
        
        return (repeating, missing)

    def Find_Repeating_Missing_Math(self, arr, n):
        """
        Using mathematical formulas (sum and sum of squares)
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        sum_val = 0
        sum_sq = 0
        expected_sum = n * (n + 1) // 2
        expected_sum_sq = n * (n + 1) * (2 * n + 1) // 6
        
        for i in range(n):
            sum_val += arr[i]
            sum_sq += arr[i] * arr[i]
        
        diff = sum_val - expected_sum
        diff_sq = sum_sq - expected_sum_sq
        
        sum_both = diff_sq // diff
        repeating = (diff + sum_both) // 2
        missing = sum_both - repeating
        
        return (repeating, missing)


def Test_Find_Repeating_And_Missing():
    sol = Solution()
    tests = [
        [2, 2],
        [1, 3, 3],
        [1, 2, 2, 4],
        [4, 3, 6, 2, 1, 1]
    ]

    for arr in tests:
        n = len(arr)
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        res1 = sol.Find_Repeating_Missing_Count_Array(arr[:], n)
        print(f"Count Array: Repeating = {res1[0]}, Missing = {res1[1]}")
        
        res2 = sol.Find_Repeating_Missing_Sign_Marking(arr[:], n)
        print(f"Sign Marking: Repeating = {res2[0]}, Missing = {res2[1]}")
        
        res3 = sol.Find_Repeating_Missing_Math(arr[:], n)
        print(f"Math: Repeating = {res3[0]}, Missing = {res3[1]}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Find_Repeating_And_Missing()
