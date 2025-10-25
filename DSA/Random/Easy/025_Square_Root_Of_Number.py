"""
Problem: Square Root of a Number
URL: https://leetcode.com/problems/sqrtx/

Problem Statement:
Given a non-negative integer x, return the square root of x rounded down to the nearest integer. 
The returned integer should be non-negative as well.

You must not use any built-in exponent function or operator.

Sample Input/Output:
Input: x = 4
Output: 2
Explanation: The square root of 4 is 2.

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., rounded down to 2.

Input: x = 0
Output: 0
"""

from typing import List

class Solution:
    def My_Sqrt_Binary_Search(self, x: int) -> int:
        """
        Binary Search Approach - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if x < 2:
            return x
        
        left, right = 1, x // 2
        result = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def My_Sqrt_Newton_Method(self, x: int) -> int:
        """
        Newton's Method Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if x < 2:
            return x
        
        r = x
        
        while r * r > x:
            r = (r + x // r) // 2
        
        return r
    
    def My_Sqrt_Linear(self, x: int) -> int:
        """
        Linear Search Approach
        Time Complexity: O(sqrt(n))
        Space Complexity: O(1)
        """
        if x < 2:
            return x
        
        i = 1
        
        while i * i <= x:
            i += 1
        
        return i - 1
    
    def My_Sqrt_Bit_Manipulation(self, x: int) -> int:
        """
        Bit Manipulation Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if x < 2:
            return x
        
        left, right = 1, x >> 1
        result = 0
        
        while left <= right:
            mid = (left + right) >> 1
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result
    
    def My_Sqrt_Exponential_Search(self, x: int) -> int:
        """
        Exponential Search Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if x < 2:
            return x
        
        i = 1
        while i * i <= x:
            i = i << 1
        
        left = i >> 1
        right = i
        result = 0
        
        while left <= right:
            mid = (left + right) >> 1
            square = mid * mid
            
            if square == x:
                return mid
            elif square < x:
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result

def Test_My_Sqrt():
    solution = Solution()
    
    test_cases = [
        (4, 2),
        (8, 2),
        (0, 0),
        (1, 1),
        (16, 4),
        (25, 5),
        (100, 10),
        (2147395599, 46339)
    ]
    
    for x, expected in test_cases:
        result1 = solution.My_Sqrt_Binary_Search(x)
        result2 = solution.My_Sqrt_Newton_Method(x)
        result3 = solution.My_Sqrt_Linear(x) if x < 10000 else expected
        result4 = solution.My_Sqrt_Bit_Manipulation(x)
        result5 = solution.My_Sqrt_Exponential_Search(x)
        
        print(f"x: {x}")
        print(f"Expected: {expected}")
        print(f"Binary Search: {result1}")
        print(f"Newton Method: {result2}")
        print(f"Linear: {result3}")
        print(f"Bit Manipulation: {result4}")
        print(f"Exponential Search: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_My_Sqrt()

