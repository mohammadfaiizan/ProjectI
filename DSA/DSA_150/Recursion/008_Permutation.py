"""
Problem: Permutation
URL: https://leetcode.com/problems/permutations/description/

Problem Statement:
Given an array nums of distinct integers, return all the possible permutations.
You can return the answer in any order.

Sample Input/Output:
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Explanation: All permutations of [1,2,3]

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Explanation: All permutations of [0,1]
"""

from typing import List
import itertools

class Solution:
    def Permute_Built_In(self, nums: List[int]) -> List[List[int]]:
        """
        Built-in Permutations - Using itertools
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        return [list(p) for p in itertools.permutations(nums)]
    
    def Permute_Backtracking_Recursive(self, nums: List[int]) -> List[List[int]]:
        """
        Backtracking Recursive - Classic backtracking approach
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = []
        
        def Backtrack(current_permutation: List[int]) -> None:
            if len(current_permutation) == len(nums):
                result.append(current_permutation[:])
                return
            
            for num in nums:
                if num not in current_permutation:
                    current_permutation.append(num)
                    Backtrack(current_permutation)
                    current_permutation.pop()
        
        Backtrack([])
        return result
    
    def Permute_Swap_Based_Recursive(self, nums: List[int]) -> List[List[int]]:
        """
        Swap Based Recursive - Generate permutations by swapping
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = []
        
        def Generate_Permutations(start: int) -> None:
            if start == len(nums):
                result.append(nums[:])
                return
            
            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                Generate_Permutations(start + 1)
                nums[start], nums[i] = nums[i], nums[start]
        
        Generate_Permutations(0)
        return result
    
    def Permute_Used_Array_Recursive(self, nums: List[int]) -> List[List[int]]:
        """
        Used Array Recursive - Track used elements with boolean array
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = []
        used = [False] * len(nums)
        
        def Generate(current: List[int]) -> None:
            if len(current) == len(nums):
                result.append(current[:])
                return
            
            for i in range(len(nums)):
                if not used[i]:
                    used[i] = True
                    current.append(nums[i])
                    Generate(current)
                    current.pop()
                    used[i] = False
        
        Generate([])
        return result
    
    def Permute_Heap_Algorithm_Recursive(self, nums: List[int]) -> List[List[int]]:
        """
        Heap's Algorithm Recursive - Efficient permutation generation
        Time Complexity: O(n!)
        Space Complexity: O(n! * n)
        """
        result = []
        
        def Heap_Permute(n: int) -> None:
            if n == 1:
                result.append(nums[:])
                return
            
            for i in range(n):
                Heap_Permute(n - 1)
                
                if n % 2 == 1:
                    nums[0], nums[n - 1] = nums[n - 1], nums[0]
                else:
                    nums[i], nums[n - 1] = nums[n - 1], nums[i]
        
        Heap_Permute(len(nums))
        return result
    
    def Permute_Lexicographic_Recursive(self, nums: List[int]) -> List[List[int]]:
        """
        Lexicographic Recursive - Generate in lexicographic order
        Time Complexity: O(n! * n)
        Space Complexity: O(n! * n)
        """
        result = []
        nums.sort()
        
        def Next_Permutation() -> bool:
            i = len(nums) - 2
            while i >= 0 and nums[i] >= nums[i + 1]:
                i -= 1
            
            if i == -1:
                return False
            
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            
            nums[i], nums[j] = nums[j], nums[i]
            nums[i + 1:] = reversed(nums[i + 1:])
            return True
        
        result.append(nums[:])
        while Next_Permutation():
            result.append(nums[:])
        
        return result

def Test_Permute():
    solution = Solution()
    
    test_cases = [
        [1,2,3],
        [0,1],
        [1],
        [1,2,3,4]
    ]
    
    for nums in test_cases:
        result1 = solution.Permute_Built_In(nums.copy())
        result2 = solution.Permute_Backtracking_Recursive(nums.copy())
        result3 = solution.Permute_Swap_Based_Recursive(nums.copy())
        result4 = solution.Permute_Used_Array_Recursive(nums.copy())
        result5 = solution.Permute_Heap_Algorithm_Recursive(nums.copy())
        result6 = solution.Permute_Lexicographic_Recursive(nums.copy())
        
        print(f"Array: {nums}")
        print(f"Built-in count: {len(result1)}")
        print(f"Backtracking count: {len(result2)}")
        print(f"Swap-based count: {len(result3)}")
        print(f"Used array count: {len(result4)}")
        print(f"Heap algorithm count: {len(result5)}")
        print(f"Lexicographic count: {len(result6)}")
        
        if len(nums) <= 3:
            print(f"Backtracking result: {result2}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Permute()
