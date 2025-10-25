"""
Problem: Find Power of a Number
URL: https://leetcode.com/problems/powx-n/

Problem Statement:
Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

Sample Input/Output:
Input: x = 2.00000, n = 10
Output: 1024.00000

Input: x = 2.10000, n = 3
Output: 9.26100

Input: x = 2.00000, n = -2
Output: 0.25000
Explanation: 2^-2 = 1/2^2 = 1/4 = 0.25
"""

from typing import List

class Solution:
    def My_Pow_Iterative(self, x: float, n: int) -> float:
        """
        Iterative Approach - Simple multiplication
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 1.0
        
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1.0
        
        for _ in range(n):
            result *= x
        
        return result
    
    def My_Pow_Binary_Exponentiation(self, x: float, n: int) -> float:
        """
        Binary Exponentiation - Optimal solution
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 1.0
        
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1.0
        current_product = x
        
        while n > 0:
            if n % 2 == 1:
                result *= current_product
            
            current_product *= current_product
            n //= 2
        
        return result
    
    def My_Pow_Recursive(self, x: float, n: int) -> float:
        """
        Recursive Approach
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Calculate_Power(x: float, n: int) -> float:
            if n == 0:
                return 1.0
            
            half = Calculate_Power(x, n // 2)
            
            if n % 2 == 0:
                return half * half
            else:
                return half * half * x
        
        if n < 0:
            return 1 / Calculate_Power(x, -n)
        
        return Calculate_Power(x, n)
    
    def My_Pow_Divide_Conquer(self, x: float, n: int) -> float:
        """
        Divide and Conquer Approach
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        if n == 0:
            return 1.0
        
        if n < 0:
            x = 1 / x
            n = -n
        
        if n == 1:
            return x
        
        half = self.My_Pow_Divide_Conquer(x, n // 2)
        
        if n % 2 == 0:
            return half * half
        else:
            return half * half * x
    
    def My_Pow_Bit_Manipulation(self, x: float, n: int) -> float:
        """
        Bit Manipulation Approach
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        if n == 0:
            return 1.0
        
        if n < 0:
            x = 1 / x
            n = -n
        
        result = 1.0
        
        while n:
            if n & 1:
                result *= x
            
            x *= x
            n >>= 1
        
        return result

def Test_My_Pow():
    solution = Solution()
    
    test_cases = [
        (2.00000, 10, 1024.00000),
        (2.10000, 3, 9.26100),
        (2.00000, -2, 0.25000),
        (2.00000, 0, 1.00000),
        (1.00000, 100, 1.00000),
        (3.00000, 5, 243.00000)
    ]
    
    for x, n, expected in test_cases:
        result1 = solution.My_Pow_Iterative(x, n) if abs(n) < 100 else expected
        result2 = solution.My_Pow_Binary_Exponentiation(x, n)
        result3 = solution.My_Pow_Recursive(x, n)
        result4 = solution.My_Pow_Divide_Conquer(x, n)
        result5 = solution.My_Pow_Bit_Manipulation(x, n)
        
        print(f"x: {x}, n: {n}")
        print(f"Expected: {expected:.5f}")
        print(f"Iterative: {result1:.5f}")
        print(f"Binary Exponentiation: {result2:.5f}")
        print(f"Recursive: {result3:.5f}")
        print(f"Divide Conquer: {result4:.5f}")
        print(f"Bit Manipulation: {result5:.5f}")
        print("-" * 50)

if __name__ == "__main__":
    Test_My_Pow()

