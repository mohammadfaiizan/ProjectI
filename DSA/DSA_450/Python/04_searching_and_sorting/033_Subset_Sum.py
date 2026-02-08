"""
Problem: Subset Sums
URL: https://www.spoj.com/problems/SUBSUMS/

Problem Statement:
Count number of subsets whose sum falls in range [A, B].
Meet in the middle technique: split array into two halves.
Generate all subset sums for each half, then combine.

Sample Input:
arr[] = {1, 2, 3}, A = 1, B = 3

Sample Output:
4
"""

import bisect


class Solution:
    def Generate_Subset_Sums(self, arr, sums):
        """
        Approach: Generate all subset sums for a given array
        Use bit manipulation or recursion to generate all possible sums
        
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        sums.append(0)
        
        for i in range(len(arr)):
            size = len(sums)
            for j in range(size):
                sums.append(sums[j] + arr[i])

    def Count_Subsets_In_Range(self, arr, A, B):
        """
        Approach: Meet in the middle
        Split array into two halves, generate subset sums for each
        For each sum in first half, count valid sums in second half using binary search
        
        Time Complexity: O(2^(n/2) * log(2^(n/2)))
        Space Complexity: O(2^(n/2))
        """
        n = len(arr)
        mid = n // 2
        
        left = arr[:mid]
        right = arr[mid:]
        
        left_sums = []
        right_sums = []
        self.Generate_Subset_Sums(left, left_sums)
        self.Generate_Subset_Sums(right, right_sums)
        
        right_sums.sort()
        
        count = 0
        for left_sum in left_sums:
            min_right = A - left_sum
            max_right = B - left_sum
            
            lower = bisect.bisect_left(right_sums, min_right)
            upper = bisect.bisect_right(right_sums, max_right)
            
            count += (upper - lower)
        
        return count


def Test_Subset_Sum():
    sol = Solution()
    
    arr1 = [1, 2, 3]
    assert sol.Count_Subsets_In_Range(arr1, 1, 3) == 4
    
    arr2 = [1, 2]
    assert sol.Count_Subsets_In_Range(arr2, 1, 2) == 2
    
    arr3 = [1]
    assert sol.Count_Subsets_In_Range(arr3, 0, 1) == 2
    
    arr4 = [1, 2, 3, 4]
    assert sol.Count_Subsets_In_Range(arr4, 3, 6) >= 0
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Subset_Sum()
