"""
Problem: Kth Permutation Sequence
URL: https://leetcode.com/problems/permutation-sequence/

Problem Statement:
Given N and K, find the Kth permutation sequence of numbers 1 to N.

Sample Input/Output:
Input: N = 3, K = 3
Output: "213"
Explanation: Permutations: "123", "132", "213", "231", "312", "321"
"""


class Solution:
    def Kth_Permutation_Math_Based(self, n, k):
        """
        Math-based factorial number system
        Time Complexity: O(N^2)
        Space Complexity: O(N)
        """
        factorial = [1] * (n + 1)
        for i in range(1, n + 1):
            factorial[i] = factorial[i - 1] * i
        
        numbers = list(range(1, n + 1))
        result = []
        k -= 1
        
        for i in range(n, 0, -1):
            idx = k // factorial[i - 1]
            result.append(str(numbers[idx]))
            numbers.pop(idx)
            k %= factorial[i - 1]
        
        return ''.join(result)
    
    def Kth_Permutation_Generate_All(self, n, k):
        """
        Generate all permutations
        Time Complexity: O(N!)
        Space Complexity: O(N!)
        """
        nums = list(range(1, n + 1))
        permutations = []
        
        def backtrack(arr, idx):
            if idx == len(arr):
                perm = ''.join(str(num) for num in arr)
                permutations.append(perm)
                return
            
            arr_list = arr[:]
            for i in range(idx, len(arr_list)):
                arr_list[idx], arr_list[i] = arr_list[i], arr_list[idx]
                backtrack(arr_list, idx + 1)
                arr_list[idx], arr_list[i] = arr_list[i], arr_list[idx]
                
                if len(permutations) >= k:
                    return
        
        backtrack(nums, 0)
        return permutations[k - 1]


def Test_Kth_Permutation_Sequence():
    solution = Solution()
    n = 3
    k = 3
    print("Math-Based Approach:", solution.Kth_Permutation_Math_Based(n, k))
    print("Generate All Approach:", solution.Kth_Permutation_Generate_All(n, k))


if __name__ == "__main__":
    Test_Kth_Permutation_Sequence()
