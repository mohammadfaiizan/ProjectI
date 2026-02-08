"""
Problem: Maximum Sum Absolute Difference
URL: https://www.geeksforgeeks.org/maximum-sum-absolute-difference-array/

Problem Statement:
Rearrange array to maximize sum of |arr[i]-arr[i+1]| (circular).

Sample Input/Output:
Input: arr[] = {1, 2, 4, 8}
Output: 18
Explanation: Rearrange to {1, 8, 2, 4}. Sum = |1-8| + |8-2| + |2-4| + |4-1| = 7 + 6 + 2 + 3 = 18
"""


class Solution:
    def Max_Sum_Absolute_Difference_Sort_Interleave(self, arr):
        """
        Sort + interleave small/large greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        """
        arr.sort()
        n = len(arr)
        result = [0] * n
        
        left, right = 0, n - 1
        for i in range(n):
            if i % 2 == 0:
                result[i] = arr[left]
                left += 1
            else:
                result[i] = arr[right]
                right -= 1
        
        sum_val = 0
        for i in range(n):
            sum_val += abs(result[i] - result[(i + 1) % n])
        
        return sum_val
    
    def Max_Sum_Absolute_Difference_Sort_Double_Difference(self, arr):
        """
        Sort + double difference greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        arr.sort()
        n = len(arr)
        sum_val = 0
        
        for i in range(n // 2):
            sum_val -= 2 * arr[i]
            sum_val += 2 * arr[n - 1 - i]
        
        return sum_val


def Test_Max_Sum_Absolute_Difference():
    solution = Solution()
    
    arr1 = [1, 2, 4, 8]
    print(f"Test 1 (Interleave): {solution.Max_Sum_Absolute_Difference_Sort_Interleave(arr1[:])}")
    
    arr2 = [1, 2, 4, 8]
    print(f"Test 1 (Double Diff): {solution.Max_Sum_Absolute_Difference_Sort_Double_Difference(arr2[:])}")
    
    arr3 = [4, 2, 1, 8]
    print(f"Test 2 (Interleave): {solution.Max_Sum_Absolute_Difference_Sort_Interleave(arr3[:])}")


if __name__ == "__main__":
    Test_Max_Sum_Absolute_Difference()
