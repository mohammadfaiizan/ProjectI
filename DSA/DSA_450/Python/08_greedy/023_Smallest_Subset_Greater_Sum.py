"""
Problem: Smallest Subset Greater Sum
URL: https://www.geeksforgeeks.org/smallest-subset-sum-greater-elements/

Problem Statement:
Find minimum number of elements such that their sum is greater than sum of rest.

Sample Input/Output:
Input: arr[] = {3, 1, 7, 1}
Output: 1
Explanation: Subset {7} has sum 7 > sum of rest (3+1+1=5). Minimum size is 1.
"""


class Solution:
    def Smallest_Subset_Greater_Sum_Sort_Descending(self, arr):
        """
        Sort descending + greedy pick greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        total_sum = sum(arr)
        
        arr.sort(reverse=True)
        
        subset_sum = 0
        count = 0
        
        for num in arr:
            subset_sum += num
            count += 1
            if subset_sum > total_sum - subset_sum:
                return count
        
        return count


def Test_Smallest_Subset_Greater_Sum():
    solution = Solution()
    
    arr1 = [3, 1, 7, 1]
    print(f"Test 1: {solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr1)}")
    
    arr2 = [2, 1, 2]
    print(f"Test 2: {solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr2)}")
    
    arr3 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print(f"Test 3: {solution.Smallest_Subset_Greater_Sum_Sort_Descending(arr3)}")


if __name__ == "__main__":
    Test_Smallest_Subset_Greater_Sum()
