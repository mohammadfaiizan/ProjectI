"""
Problem: Find Pair Given Difference
URL: https://practice.geeksforgeeks.org/problems/find-pair-given-difference1559/1

Problem Statement:
Given an unsorted array arr[] of size n and an integer diff, find if there exists a pair of elements in the array whose difference is diff.

Sample Input/Output:
Input: n = 6, diff = 78, arr[] = {5, 20, 3, 2, 5, 80}
Output: 1

Input: n = 5, diff = 45, arr[] = {90, 70, 20, 80, 50}
Output: -1
"""


class Solution:
    def Find_Pair_Sorting_Two_Pointer(self, arr, n, diff):
        """
        Sort array and use two pointers to find pair with given difference
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr_sorted = sorted(arr)
        i = 0
        j = 1
        while i < n and j < n:
            if i != j and arr_sorted[j] - arr_sorted[i] == diff:
                return True
            elif arr_sorted[j] - arr_sorted[i] < diff:
                j += 1
            else:
                i += 1
        return False

    def Find_Pair_HashSet(self, arr, n, diff):
        """
        Use hash set to store elements and check if arr[i] + diff or arr[i] - diff exists
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        seen = set()
        for i in range(n):
            if (arr[i] + diff) in seen or (arr[i] - diff) in seen:
                return True
            seen.add(arr[i])
        return False


def Test_Find_Pair_With_Given_Diff():
    sol = Solution()
    tests = [
        ([5, 20, 3, 2, 5, 80], 78),
        ([90, 70, 20, 80, 50], 45),
        ([1, 8, 30, 40, 100], 60),
        ([10, 20, 30], 10),
        ([1, 2, 3, 4, 5], 0)
    ]

    for test in tests:
        arr = test[0]
        diff = test[1]
        n = len(arr)
        
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print(f", diff = {diff}")
        
        arr1 = arr[:]
        arr2 = arr[:]
        res1 = sol.Find_Pair_Sorting_Two_Pointer(arr1, n, diff)
        res2 = sol.Find_Pair_HashSet(arr2, n, diff)
        
        print(f"Sorting + Two Pointer: {'Found' if res1 else 'Not Found'}")
        print(f"HashSet: {'Found' if res2 else 'Not Found'}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Find_Pair_With_Given_Diff()
