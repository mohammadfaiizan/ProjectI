"""
Problem: Palindrome Partitioning
URL: https://www.geeksforgeeks.org/given-a-string-print-all-possible-palindromic-partition/

Problem Statement:
Given a string, find all possible palindromic partitions.

Sample Input/Output:
Input: s="aab"
Output: [["a","a","b"],["aa","b"]]
Explanation: Two ways to partition into palindromes
"""


class Solution:
    def Palindrome_Partitioning_Backtracking(self, s):
        """
        Backtracking with palindrome check
        Time Complexity: O(n * 2^n)
        Space Complexity: O(n)
        """
        result = []
        current_partition = []
        
        def Is_Palindrome(str_val):
            left = 0
            right = len(str_val) - 1
            while left < right:
                if str_val[left] != str_val[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def backtrack(start):
            if start == len(s):
                result.append(current_partition[:])
                return
            
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if Is_Palindrome(substring):
                    current_partition.append(substring)
                    backtrack(end)
                    current_partition.pop()
        
        backtrack(0)
        return result


def Test_Palindrome_Partitioning():
    solution = Solution()
    
    s = "aab"
    partitions = solution.Palindrome_Partitioning_Backtracking(s)
    
    print("Palindromic partitions:")
    for partition in partitions:
        print(" ".join(partition))


if __name__ == "__main__":
    Test_Palindrome_Partitioning()
