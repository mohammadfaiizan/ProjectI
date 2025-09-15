"""
Problem: Next Alphabetical Element
URL: No link found

Problem Statement:
Given a sorted array of letters and a target letter, find the smallest letter in the array that is greater than the target.

Sample Input/Output:
Input: letters = ['c', 'f', 'j'], target = 'a'
Output: 'c'
Explanation: 'c' is the smallest letter greater than 'a'

Input: letters = ['c', 'f', 'j'], target = 'd'
Output: 'f'
Explanation: 'f' is the smallest letter greater than 'd'
"""

from typing import List

class Solution:
    def Next_Greatest_Letter_Linear(self, letters: List[str], target: str) -> str:
        """
        Linear Search Approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for letter in letters:
            if letter > target:
                return letter
        return letters[0]
    
    def Next_Greatest_Letter_Binary_Search_Optimal(self, letters: List[str], target: str) -> str:
        """
        Binary Search Optimal Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 0, len(letters) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        
        return letters[left % len(letters)]

def Test_Next_Greatest_Letter():
    solution = Solution()
    
    test_cases = [
        (['c', 'f', 'j'], 'a', 'c'),
        (['c', 'f', 'j'], 'd', 'f'),
        (['c', 'f', 'j'], 'j', 'c'),
        (['c', 'f', 'j'], 'k', 'c')
    ]
    
    for letters, target, expected in test_cases:
        result1 = solution.Next_Greatest_Letter_Linear(letters.copy(), target)
        result2 = solution.Next_Greatest_Letter_Binary_Search_Optimal(letters.copy(), target)
        
        print(f"Letters: {letters}, Target: '{target}'")
        print(f"Expected: '{expected}'")
        print(f"Linear: '{result1}'")
        print(f"Binary Search: '{result2}'")
        print("-" * 50)

if __name__ == "__main__":
    Test_Next_Greatest_Letter()
