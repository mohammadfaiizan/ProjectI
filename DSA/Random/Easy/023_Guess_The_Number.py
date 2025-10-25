"""
Problem: Guess The Number
URL: https://leetcode.com/problems/guess-number-higher-or-lower/

Problem Statement:
We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower 
than your guess.

You call a pre-defined API int guess(int num), which returns three possible results:

-1: Your guess is higher than the number I picked (i.e. num > pick).
1: Your guess is lower than the number I picked (i.e. num < pick).
0: your guess is equal to the number I picked (i.e. num == pick).

Return the number that I picked.

Sample Input/Output:
Input: n = 10, pick = 6
Output: 6

Input: n = 1, pick = 1
Output: 1

Input: n = 2, pick = 1
Output: 1
"""

from typing import List

picked_number = 0

def Guess(num: int) -> int:
    """
    Pre-defined API function
    """
    if num > picked_number:
        return -1
    elif num < picked_number:
        return 1
    else:
        return 0

class Solution:
    def Guess_Number_Binary_Search(self, n: int) -> int:
        """
        Binary Search Approach - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 1, n
        
        while left <= right:
            mid = left + (right - left) // 2
            result = Guess(mid)
            
            if result == 0:
                return mid
            elif result == -1:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    def Guess_Number_Recursive(self, n: int) -> int:
        """
        Recursive Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Search(left: int, right: int) -> int:
            if left > right:
                return -1
            
            mid = left + (right - left) // 2
            result = Guess(mid)
            
            if result == 0:
                return mid
            elif result == -1:
                return Search(left, mid - 1)
            else:
                return Search(mid + 1, right)
        
        return Search(1, n)
    
    def Guess_Number_Ternary_Search(self, n: int) -> int:
        """
        Ternary Search Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 1, n
        
        while left <= right:
            mid1 = left + (right - left) // 3
            mid2 = right - (right - left) // 3
            
            res1 = Guess(mid1)
            res2 = Guess(mid2)
            
            if res1 == 0:
                return mid1
            if res2 == 0:
                return mid2
            
            if res1 < 0:
                right = mid1 - 1
            elif res2 > 0:
                left = mid2 + 1
            else:
                left = mid1 + 1
                right = mid2 - 1
        
        return -1
    
    def Guess_Number_Linear(self, n: int) -> int:
        """
        Linear Search Approach - For comparison
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        for i in range(1, n + 1):
            if Guess(i) == 0:
                return i
        
        return -1
    
    def Guess_Number_Bit_Manipulation(self, n: int) -> int:
        """
        Binary Search with Bit Manipulation
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        left, right = 1, n
        
        while left <= right:
            mid = (left + right) >> 1
            result = Guess(mid)
            
            if result == 0:
                return mid
            elif result == -1:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1

def Test_Guess_Number():
    global picked_number
    solution = Solution()
    
    test_cases = [
        (10, 6, 6),
        (1, 1, 1),
        (2, 1, 1),
        (100, 50, 50),
        (1000, 999, 999)
    ]
    
    for n, pick, expected in test_cases:
        picked_number = pick
        
        result1 = solution.Guess_Number_Binary_Search(n)
        picked_number = pick
        result2 = solution.Guess_Number_Recursive(n)
        picked_number = pick
        result3 = solution.Guess_Number_Ternary_Search(n)
        picked_number = pick
        result4 = solution.Guess_Number_Linear(n) if n <= 100 else expected
        picked_number = pick
        result5 = solution.Guess_Number_Bit_Manipulation(n)
        
        print(f"n: {n}, pick: {pick}")
        print(f"Expected: {expected}")
        print(f"Binary Search: {result1}")
        print(f"Recursive: {result2}")
        print(f"Ternary Search: {result3}")
        print(f"Linear: {result4}")
        print(f"Bit Manipulation: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Guess_Number()

