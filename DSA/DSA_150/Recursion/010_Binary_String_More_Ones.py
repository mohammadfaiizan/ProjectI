"""
Problem: Binary String with number of 1s is greater than number of 0s
URL: https://www.geeksforgeeks.org/problems/count-the-substring--170645/1

Problem Statement:
Given a number N, find count of all binary strings of length N which have more 1s than 0s.

Sample Input/Output:
Input: n = 2
Output: 1
Explanation: "11" is the only string of length 2 with more 1s than 0s

Input: n = 3
Output: 4
Explanation: "110", "101", "011", "111" have more 1s than 0s
"""

from typing import List

class Solution:
    def Count_Binary_Strings_Brute_Force(self, n: int) -> int:
        """
        Brute Force - Generate all strings and count valid ones
        Time Complexity: O(2^n)
        Space Complexity: O(2^n)
        """
        def Generate_All_Strings(length: int) -> List[str]:
            if length == 0:
                return [""]
            
            shorter = Generate_All_Strings(length - 1)
            result = []
            for s in shorter:
                result.append(s + "0")
                result.append(s + "1")
            return result
        
        all_strings = Generate_All_Strings(n)
        count = 0
        
        for s in all_strings:
            ones = s.count('1')
            zeros = s.count('0')
            if ones > zeros:
                count += 1
        
        return count
    
    def Count_Binary_Strings_Recursive_Optimal(self, n: int) -> int:
        """
        Recursive Approach - Count valid strings directly
        Time Complexity: O(2^n)
        Space Complexity: O(n) - recursion stack
        """
        def Count_Recursive(remaining: int, ones: int, zeros: int) -> int:
            if remaining == 0:
                return 1 if ones > zeros else 0
            
            count = 0
            count += Count_Recursive(remaining - 1, ones + 1, zeros)
            count += Count_Recursive(remaining - 1, ones, zeros + 1)
            
            return count
        
        return Count_Recursive(n, 0, 0)
    
    def Count_Binary_Strings_Memoized(self, n: int) -> int:
        """
        Memoized Recursive - Cache results for efficiency
        Time Complexity: O(n³)
        Space Complexity: O(n³)
        """
        memo = {}
        
        def Count_With_Memo(remaining: int, ones: int, zeros: int) -> int:
            if remaining == 0:
                return 1 if ones > zeros else 0
            
            if (remaining, ones, zeros) in memo:
                return memo[(remaining, ones, zeros)]
            
            count = 0
            count += Count_With_Memo(remaining - 1, ones + 1, zeros)
            count += Count_With_Memo(remaining - 1, ones, zeros + 1)
            
            memo[(remaining, ones, zeros)] = count
            return count
        
        return Count_With_Memo(n, 0, 0)
    
    def Count_Binary_Strings_Dynamic_Programming(self, n: int) -> int:
        """
        Dynamic Programming - Bottom-up approach
        Time Complexity: O(n³)
        Space Complexity: O(n³)
        """
        if n == 1:
            return 1
        
        dp = {}
        
        for length in range(1, n + 1):
            for ones in range(length + 1):
                zeros = length - ones
                if length == 1:
                    dp[(length, ones, zeros)] = 1 if ones > zeros else 0
                else:
                    count = 0
                    if ones > 0:
                        count += dp.get((length - 1, ones - 1, zeros), 0)
                    if zeros > 0:
                        count += dp.get((length - 1, ones, zeros - 1), 0)
                    dp[(length, ones, zeros)] = count
        
        total = 0
        for ones in range(n + 1):
            zeros = n - ones
            if ones > zeros:
                total += dp.get((n, ones, zeros), 0)
        
        return total
    
    def Count_Binary_Strings_Mathematical(self, n: int) -> int:
        """
        Mathematical Formula - Using Catalan numbers
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        if n % 2 == 0:
            return 0
        
        from math import comb
        return comb(n, (n + 1) // 2)

def Test_Count_Binary_Strings():
    solution = Solution()
    
    test_cases = [1, 2, 3, 4, 5]
    
    for n in test_cases:
        result1 = solution.Count_Binary_Strings_Brute_Force(n)
        result2 = solution.Count_Binary_Strings_Recursive_Optimal(n)
        result3 = solution.Count_Binary_Strings_Memoized(n)
        result4 = solution.Count_Binary_Strings_Dynamic_Programming(n)
        result5 = solution.Count_Binary_Strings_Mathematical(n)
        
        print(f"n = {n}")
        print(f"Brute Force: {result1}")
        print(f"Recursive Optimal: {result2}")
        print(f"Memoized: {result3}")
        print(f"Dynamic Programming: {result4}")
        print(f"Mathematical: {result5}")
        print("-" * 50)

if __name__ == "__main__":
    Test_Count_Binary_Strings()
