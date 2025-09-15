"""
Problem: Palindrome Partitioning II
URL: https://leetcode.com/problems/palindrome-partitioning-ii

Problem Statement:
Given a string s, partition s such that every substring of the partition is a palindrome.
Return the minimum cuts needed for a palindrome partitioning of s.

Sample Input/Output:
Input: s = "aab"
Output: 1
Explanation: The palindrome partitioning ["aa","b"] could be produced using 1 cut.

Input: s = "aba"
Output: 0
Explanation: "aba" is already a palindrome, so no cuts are needed.

Input: s = "abcba"
Output: 0
Explanation: "abcba" is already a palindrome.
"""

from typing import List

class Solution:
    def Min_Cut_Brute_Force(self, s: str) -> int:
        """
        Brute Force - Try all possible partitions
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def Is_Palindrome(string: str) -> bool:
            return string == string[::-1]
        
        def Min_Cuts_Recursive(start: int) -> int:
            if start >= len(s):
                return 0
            
            if Is_Palindrome(s[start:]):
                return 0
            
            min_cuts = float('inf')
            
            for end in range(start + 1, len(s) + 1):
                if Is_Palindrome(s[start:end]):
                    cuts = 1 + Min_Cuts_Recursive(end)
                    min_cuts = min(min_cuts, cuts)
            
            return min_cuts
        
        return Min_Cuts_Recursive(0)
    
    def Min_Cut_MCM_Recursive(self, s: str) -> int:
        """
        MCM Recursive - Apply MCM pattern
        Time Complexity: O(2^n)
        Space Complexity: O(n)
        """
        def Is_Palindrome(string: str, start: int, end: int) -> bool:
            while start < end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def MCM_Palindrome(i: int, j: int) -> int:
            if i >= j or Is_Palindrome(s, i, j):
                return 0
            
            min_cuts = float('inf')
            
            for k in range(i, j):
                if Is_Palindrome(s, i, k):
                    cuts = 1 + MCM_Palindrome(k + 1, j)
                    min_cuts = min(min_cuts, cuts)
            
            return min_cuts
        
        return MCM_Palindrome(0, len(s) - 1)
    
    def Min_Cut_Memoized(self, s: str) -> int:
        """
        Memoized - Top-down DP with memoization
        Time Complexity: O(n³)
        Space Complexity: O(n²)
        """
        n = len(s)
        memo = {}
        palindrome_memo = {}
        
        def Is_Palindrome(start: int, end: int) -> bool:
            if (start, end) in palindrome_memo:
                return palindrome_memo[(start, end)]
            
            if start >= end:
                palindrome_memo[(start, end)] = True
                return True
            
            result = s[start] == s[end] and Is_Palindrome(start + 1, end - 1)
            palindrome_memo[(start, end)] = result
            return result
        
        def MCM_Memo(i: int, j: int) -> int:
            if i >= j or Is_Palindrome(i, j):
                return 0
            
            if (i, j) in memo:
                return memo[(i, j)]
            
            min_cuts = float('inf')
            
            for k in range(i, j):
                if Is_Palindrome(i, k):
                    cuts = 1 + MCM_Memo(k + 1, j)
                    min_cuts = min(min_cuts, cuts)
            
            memo[(i, j)] = min_cuts
            return min_cuts
        
        return MCM_Memo(0, n - 1)
    
    def Min_Cut_Tabulation_Optimal(self, s: str) -> int:
        """
        Tabulation Optimal - Bottom-up DP with palindrome precomputation
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(s)
        
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if length == 2:
                    is_palindrome[i][j] = (s[i] == s[j])
                else:
                    is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        dp = [0] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
            else:
                dp[i] = float('inf')
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n - 1]
    
    def Min_Cut_Optimized_Space(self, s: str) -> int:
        """
        Optimized Space - Space-optimized version
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(s)
        cuts = list(range(n))
        
        for center in range(n):
            left, right = center, center
            while left >= 0 and right < n and s[left] == s[right]:
                if left == 0:
                    cuts[right] = 0
                else:
                    cuts[right] = min(cuts[right], cuts[left - 1] + 1)
                left -= 1
                right += 1
            
            left, right = center, center + 1
            while left >= 0 and right < n and s[left] == s[right]:
                if left == 0:
                    cuts[right] = 0
                else:
                    cuts[right] = min(cuts[right], cuts[left - 1] + 1)
                left -= 1
                right += 1
        
        return cuts[n - 1]
    
    def Min_Cut_With_Partitions(self, s: str) -> tuple:
        """
        With Partitions - Return cuts and actual partition
        Time Complexity: O(n²)
        Space Complexity: O(n²)
        """
        n = len(s)
        
        is_palindrome = [[False] * n for _ in range(n)]
        
        for i in range(n):
            is_palindrome[i][i] = True
        
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                
                if length == 2:
                    is_palindrome[i][j] = (s[i] == s[j])
                else:
                    is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
        
        dp = [0] * n
        parent = [-1] * n
        
        for i in range(n):
            if is_palindrome[0][i]:
                dp[i] = 0
                parent[i] = -1
            else:
                dp[i] = float('inf')
                for j in range(i):
                    if is_palindrome[j + 1][i]:
                        if dp[j] + 1 < dp[i]:
                            dp[i] = dp[j] + 1
                            parent[i] = j
        
        partition = []
        i = n - 1
        while i >= 0:
            if parent[i] == -1:
                partition.append(s[0:i + 1])
                break
            else:
                partition.append(s[parent[i] + 1:i + 1])
                i = parent[i]
        
        return dp[n - 1], partition[::-1]
    
    def Min_Cut_Manacher_Approach(self, s: str) -> int:
        """
        Manacher Approach - Use Manacher's algorithm concept
        Time Complexity: O(n²)
        Space Complexity: O(n)
        """
        n = len(s)
        if n <= 1:
            return 0
        
        cuts = [i for i in range(n)]
        
        for i in range(n):
            radius = 0
            while i - radius >= 0 and i + radius < n and s[i - radius] == s[i + radius]:
                if i - radius == 0:
                    cuts[i + radius] = 0
                else:
                    cuts[i + radius] = min(cuts[i + radius], cuts[i - radius - 1] + 1)
                radius += 1
            
            radius = 0
            while i - radius >= 0 and i + radius + 1 < n and s[i - radius] == s[i + radius + 1]:
                if i - radius == 0:
                    cuts[i + radius + 1] = 0
                else:
                    cuts[i + radius + 1] = min(cuts[i + radius + 1], cuts[i - radius - 1] + 1)
                radius += 1
        
        return cuts[n - 1]

def Test_Min_Cut():
    solution = Solution()
    
    test_cases = [
        ("aab", 1),
        ("aba", 0),
        ("abcba", 0),
        ("raceacar", 1),
        ("abccba", 0),
        ("abcdef", 5),
        ("aaabaa", 1)
    ]
    
    methods = [
        ("MCM Recursive", solution.Min_Cut_MCM_Recursive),
        ("Memoized", solution.Min_Cut_Memoized),
        ("Tabulation Optimal", solution.Min_Cut_Tabulation_Optimal),
        ("Optimized Space", solution.Min_Cut_Optimized_Space),
        ("Manacher Approach", solution.Min_Cut_Manacher_Approach)
    ]
    
    for s, expected in test_cases:
        print(f"String: '{s}'")
        print(f"Expected: {expected}")
        
        if len(s) <= 8:
            result_bf = solution.Min_Cut_Brute_Force(s)
            print(f"Brute Force: {result_bf}")
        
        for method_name, method in methods:
            try:
                result = method(s)
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        cuts, partition = solution.Min_Cut_With_Partitions(s)
        print(f"With Partitions: Cuts={cuts}")
        print(f"Partition: {partition}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Min_Cut()
