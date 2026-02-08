"""
Problem: Zero Sum Subarrays
URL: https://practice.geeksforgeeks.org/problems/zero-sum-subarrays1825/1

Problem Statement:
You are given an array arr[] of size n. Find the total count of sub-arrays having their sum equal to 0.

Sample Input/Output:
Input: n = 6, arr[] = {0, 0, 5, 5, 0, 0}
Output: 6

Input: n = 10, arr[] = {6, -1, -3, 4, -2, 2, 4, 6, -12, -7}
Output: 4
"""


class Solution:
    def Subarray_Sum_Zero_Prefix_HashMap(self, arr, n):
        """
        Use prefix sum and hash map to count subarrays with zero sum
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        prefix_sum = {}
        sum_val = 0
        count = 0
        
        prefix_sum[0] = 1
        
        for i in range(n):
            sum_val += arr[i]
            if sum_val in prefix_sum:
                count += prefix_sum[sum_val]
            prefix_sum[sum_val] = prefix_sum.get(sum_val, 0) + 1
        
        return count

    def Subarray_Sum_Zero_Brute_Force(self, arr, n):
        """
        Check all possible subarrays and count those with zero sum
        Time Complexity: O(n^2)
        Space Complexity: O(1)
        """
        count = 0
        
        for i in range(n):
            sum_val = 0
            for j in range(i, n):
                sum_val += arr[j]
                if sum_val == 0:
                    count += 1
        
        return count


def Test_Subarray_With_Sum_Zero():
    sol = Solution()
    tests = [
        [0, 0, 5, 5, 0, 0],
        [6, -1, -3, 4, -2, 2, 4, 6, -12, -7],
        [1, -1, 1, -1],
        [0],
        [1, 2, 3]
    ]

    for arr in tests:
        n = len(arr)
        print("Array:", end=" ")
        for num in arr:
            print(num, end=" ")
        print()
        
        res1 = sol.Subarray_Sum_Zero_Prefix_HashMap(arr[:], n)
        res2 = sol.Subarray_Sum_Zero_Brute_Force(arr[:], n)
        
        print(f"Prefix Sum + HashMap: {res1}")
        print(f"Brute Force: {res2}")
        
        print("-" * 50)


if __name__ == "__main__":
    Test_Subarray_With_Sum_Zero()
