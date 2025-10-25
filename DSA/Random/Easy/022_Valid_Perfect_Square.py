"""
Problem: Valid Perfect Square
URL: https://leetcode.com/problems/valid-perfect-square/

Problem Statement:
Given a positive integer num, return true if num is a perfect square or false otherwise.

A perfect square is an integer that is the square of an integer. In other words, 
it is the product of some integer with itself.

You must not use any built-in library function, such as sqrt.

Sample Input/Output:
Input: num = 16
Output: true
Explanation: 4 * 4 = 16

Input: num = 14
Output: false

Input: num = 1
Output: true
"""

from typing import List

class Solution:
    def Is_Perfect_Square_Binary_Search(self, num: int) -> bool:
        """
        Binary Search Approach - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if num < 2:
            return True
        
        left, right = 2, num // 2
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        
        return False
    
    def Is_Perfect_Square_Newton_Method(self, num: int) -> bool:
        """
        Newton's Method Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if num < 2:
            return True
        
        x = num // 2
        
        while x * x > num:
            x = (x + num // x) // 2
        
        return x * x == num
    
    def Is_Perfect_Square_Math(self, num: int) -> bool:
        """
        Mathematical Approach - Sum of odd numbers
        Time Complexity: O(sqrt(n))
        Space Complexity: O(1)
        """
        if num < 2:
            return True
        
        odd = 1
        
        while num > 0:
            num -= odd
            odd += 2
        
        return num == 0
    
    def Is_Perfect_Square_Linear(self, num: int) -> bool:
        """
        Linear Search Approach
        Time Complexity: O(sqrt(n))
        Space Complexity: O(1)
        """
        if num < 2:
            return True
        
        i = 1
        
        while i * i < num:
            i += 1
        
        return i * i == num
    
    def Is_Perfect_Square_Set_Bits(self, num: int) -> bool:
        """
        Bit Manipulation Approach with Binary Search
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if num < 2:
            return True
        
        left, right = 1, num
        
        while left <= right:
            mid = (left + right) >> 1
            square = mid * mid
            
            if square == num:
                return True
            elif square < num:
                left = mid + 1
            else:
                right = mid - 1
        
        return False

def Test_Is_Perfect_Square():
    solution = Solution()
    
    test_cases = [
        (16, True),
        (14, False),
        (1, True),
        (4, True),
        (9, True),
        (10, False),
        (100, True),
        (808201, True)
    ]
    
    for num, expected in test_cases:
        result1 = solution.Is_Perfect_Square_Binary_Search(num)
        result2 = solution.Is_Perfect_Square_Newton_Method(num)
        result3 = solution.Is_Perfect_Square_Math(num)
        result4 = solution.Is_Perfect_Square_Linear(num)
        result5 = solution.Is_Perfect_Square_Set_Bits(num)
        
        print(f"Number: {num}")
        print(f"Expected: {expected}")
        print(f"Binary Search: {result1}")
        print(f"Newton Method: {result2}")
        print(f"Math: {result3}")
        print(f"Linear: {result4}")
        print(f"Set Bits: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Perfect_Square()

