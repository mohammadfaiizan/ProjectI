"""
Problem: Maximize Sum Arr[i]*i
URL: https://practice.geeksforgeeks.org/problems/maximize-arrii-of-an-array0026/1

Problem Statement:
Maximize sum of arr[i]*i by rearranging the array.

Sample Input/Output:
Input: arr[] = {3, 5, 6, 1}
Output: 31
Explanation: Rearrange to {1, 3, 5, 6}. Sum = 0*1 + 1*3 + 2*5 + 3*6 = 31
"""


class Solution:
    def Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(self, arr):
        """
        Sort ascending greedy approach: Smallest element at smallest index
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        sum_val = 0
        mod = 1000000007
        
        for i in range(len(arr)):
            sum_val = (sum_val + arr[i] * i) % mod
        
        return sum_val


def Test_Maximize_Sum_Arr_I_Mul_I():
    solution = Solution()
    
    arr1 = [3, 5, 6, 1]
    print(f"Test 1: {solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr1)}")
    
    arr2 = [1, 2, 3]
    print(f"Test 2: {solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr2)}")
    
    arr3 = [5, 3, 2, 4, 1]
    print(f"Test 3: {solution.Maximize_Sum_Arr_I_Mul_I_Sort_Ascending(arr3)}")


if __name__ == "__main__":
    Test_Maximize_Sum_Arr_I_Mul_I()
