"""
Problem: Happy Number
URL: https://leetcode.com/problems/happy-number/

Problem Statement:
Write an algorithm to determine if a number n is happy.
A happy number is a number defined by the following process:
- Starting with any positive integer, replace the number by the sum of the squares of its digits.
- Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
- Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.

Sample Input/Output:
Input: n = 19
Output: true
Explanation: 
1² + 9² = 82
8² + 2² = 68
6² + 8² = 100
1² + 0² + 0² = 1

Input: n = 2
Output: false
"""

class Solution:
    def Is_Happy_Hash_Set(self, n: int) -> bool:
        """
        Hash Set - Track seen numbers to detect cycles
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        seen = set()
        
        while n != 1 and n not in seen:
            seen.add(n)
            n = Get_Sum_Of_Squares(n)
        
        return n == 1
    
    def Is_Happy_Floyd_Optimal(self, n: int) -> bool:
        """
        Floyd's Cycle Detection Optimal - Detect cycles without extra space
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        slow = n
        fast = n
        
        while True:
            slow = Get_Sum_Of_Squares(slow)
            fast = Get_Sum_Of_Squares(Get_Sum_Of_Squares(fast))
            
            if fast == 1:
                return True
            
            if slow == fast:
                return False
    
    def Is_Happy_Hardcoded_Cycle(self, n: int) -> bool:
        """
        Hardcoded Cycle - Use known cycle detection
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        cycle_members = {4, 16, 37, 58, 89, 145, 42, 20}
        
        while n != 1 and n not in cycle_members:
            n = Get_Sum_Of_Squares(n)
        
        return n == 1
    
    def Is_Happy_Recursive(self, n: int) -> bool:
        """
        Recursive - Use recursion with memoization
        Time Complexity: O(log n)
        Space Complexity: O(log n)
        """
        memo = {}
        
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        def Is_Happy_Helper(num: int) -> bool:
            if num == 1:
                return True
            
            if num in memo:
                return memo[num]
            
            memo[num] = False
            next_num = Get_Sum_Of_Squares(num)
            
            if next_num not in memo:
                result = Is_Happy_Helper(next_num)
                memo[num] = result
                return result
            
            return False
        
        return Is_Happy_Helper(n)
    
    def Is_Happy_Limit_Iterations(self, n: int) -> bool:
        """
        Limit Iterations - Stop after reasonable number of iterations
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        for _ in range(1000):
            if n == 1:
                return True
            n = Get_Sum_Of_Squares(n)
        
        return False
    
    def Is_Happy_Mathematical_Approach(self, n: int) -> bool:
        """
        Mathematical Approach - Use mathematical properties
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        def Get_Sum_Of_Squares(num: int) -> int:
            total = 0
            while num > 0:
                digit = num % 10
                total += digit * digit
                num //= 10
            return total
        
        while n != 1:
            n = Get_Sum_Of_Squares(n)
            
            if n == 4:
                return False
        
        return True

def Test_Is_Happy():
    solution = Solution()
    
    test_cases = [
        (19, True),
        (2, False),
        (7, True),
        (10, True),
        (1, True),
        (23, True),
        (4, False),
        (0, False)
    ]
    
    methods = [
        ("Hash Set", solution.Is_Happy_Hash_Set),
        ("Floyd Optimal", solution.Is_Happy_Floyd_Optimal),
        ("Hardcoded Cycle", solution.Is_Happy_Hardcoded_Cycle),
        ("Recursive", solution.Is_Happy_Recursive),
        ("Limit Iterations", solution.Is_Happy_Limit_Iterations),
        ("Mathematical Approach", solution.Is_Happy_Mathematical_Approach)
    ]
    
    for n, expected in test_cases:
        print(f"Number: {n}")
        print(f"Expected: {expected}")
        
        for method_name, method in methods:
            try:
                result = method(n)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Is_Happy()
