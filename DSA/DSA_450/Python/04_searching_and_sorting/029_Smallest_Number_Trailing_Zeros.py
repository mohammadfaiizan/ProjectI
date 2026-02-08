"""
Problem: Smallest Factorial Number with N Trailing Zeros
URL: https://practice.geeksforgeeks.org/problems/smallest-factorial-number5929/1

Problem Statement:
Find smallest number whose factorial has at least n trailing zeros.
Trailing zeros in factorial = count of factor 5 (since 10 = 2*5, and 2s are more abundant)

Sample Input:
n = 1

Sample Output:
5
"""


class Solution:
    def Count_Trailing_Zeros(self, num):
        """
        Approach: Count trailing zeros in factorial of a number
        Count number of 5s in prime factorization: n/5 + n/25 + n/125 + ...
        
        Time Complexity: O(log n)
        Space Complexity: O(1)
        """
        count = 0
        divisor = 5
        while divisor <= num:
            count += num // divisor
            divisor *= 5
        return count

    def Find_Smallest_Number(self, n):
        """
        Approach: Binary search on answer
        Search for smallest number whose factorial has at least n trailing zeros
        Low = 0, High = 5*n (upper bound)
        
        Time Complexity: O(log^2(n))
        Space Complexity: O(1)
        """
        if n == 0:
            return 0
        
        low = 0
        high = 5 * n
        result = high
        
        while low <= high:
            mid = low + (high - low) // 2
            zeros = self.Count_Trailing_Zeros(mid)
            
            if zeros >= n:
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return result


def Test_Smallest_Number_Trailing_Zeros():
    sol = Solution()
    
    assert sol.Find_Smallest_Number(1) == 5
    assert sol.Find_Smallest_Number(5) == 25
    assert sol.Find_Smallest_Number(0) == 0
    assert sol.Find_Smallest_Number(6) == 25
    assert sol.Find_Smallest_Number(25) == 105
    
    assert sol.Count_Trailing_Zeros(5) == 1
    assert sol.Count_Trailing_Zeros(10) == 2
    assert sol.Count_Trailing_Zeros(25) == 6
    assert sol.Count_Trailing_Zeros(100) == 24
    
    print("All test cases passed!")


if __name__ == "__main__":
    Test_Smallest_Number_Trailing_Zeros()
