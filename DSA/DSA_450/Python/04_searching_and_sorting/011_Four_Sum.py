"""
Problem: Find All Four Sum Numbers
URL: https://practice.geeksforgeeks.org/problems/find-all-four-sum-numbers1732/1

Problem Statement:
Given an array of integers arr[] and a target X, find all unique quadruplets (a, b, c, d) such that a + b + c + d = X.

Sample Input/Output:
Input: arr[] = {0,0,2,1,1}, X = 3
Output: 0 0 1 2

Input: arr[] = {10,2,3,4,5,7,8}, X = 23
Output: 2 3 7 8 2 4 5 10 3 5 7 8
"""


class Solution:
    def Four_Sum_Sorting_Two_Pointer(self, arr, n, X):
        """
        Sort array and use nested loops with two pointers for last two elements
        Time Complexity: O(n^3)
        Space Complexity: O(1)
        """
        result = []
        arr_sorted = sorted(arr)
        
        for i in range(n - 3):
            if i > 0 and arr_sorted[i] == arr_sorted[i - 1]:
                continue
            
            for j in range(i + 1, n - 2):
                if j > i + 1 and arr_sorted[j] == arr_sorted[j - 1]:
                    continue
                
                left = j + 1
                right = n - 1
                while left < right:
                    sum_val = arr_sorted[i] + arr_sorted[j] + arr_sorted[left] + arr_sorted[right]
                    if sum_val == X:
                        result.append([arr_sorted[i], arr_sorted[j], arr_sorted[left], arr_sorted[right]])
                        while left < right and arr_sorted[left] == arr_sorted[left + 1]:
                            left += 1
                        while left < right and arr_sorted[right] == arr_sorted[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif sum_val < X:
                        left += 1
                    else:
                        right -= 1
        
        return result

    def Four_Sum_Hashing(self, arr, n, X):
        """
        Use hash map to store sum of pairs and find complement pairs
        Time Complexity: O(n^2)
        Space Complexity: O(n^2)
        """
        result = []
        pair_sum = {}
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                sum_val = arr[i] + arr[j]
                complement = X - sum_val
                
                if complement in pair_sum:
                    for p in pair_sum[complement]:
                        if p[0] != i and p[0] != j and p[1] != i and p[1] != j:
                            quad = sorted([arr[p[0]], arr[p[1]], arr[i], arr[j]])
                            if quad not in result:
                                result.append(quad)
                
                if sum_val not in pair_sum:
                    pair_sum[sum_val] = []
                pair_sum[sum_val].append((i, j))
        
        result.sort()
        return result


def Test_Four_Sum():
    sol = Solution()
    tests = [
        ([0, 0, 2, 1, 1], 3),
        ([10, 2, 3, 4, 5, 7, 8], 23),
        ([1, 0, -1, 0, -2, 2], 0),
        ([2, 2, 2, 2, 2], 8)
    ]

    for test in tests:
        arr = test[0]
        X = test[1]
        n = len(arr)
        
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print(f", X = {X}")
        
        arr1 = arr[:]
        arr2 = arr[:]
        res1 = sol.Four_Sum_Sorting_Two_Pointer(arr1, n, X)
        res2 = sol.Four_Sum_Hashing(arr2, n, X)
        
        print("Sorting + Two Pointer:", end=" ")
        for quad in res1:
            for num in quad:
                print(num, end=" ")
        print()
        
        print("Hashing:", end=" ")
        for quad in res2:
            for num in quad:
                print(num, end=" ")
        print()
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Four_Sum()
