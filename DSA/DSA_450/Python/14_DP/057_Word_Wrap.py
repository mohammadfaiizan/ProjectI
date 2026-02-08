"""
Problem: Word Wrap
URL: https://practice.geeksforgeeks.org/problems/word-wrap1646/1

Problem Statement:
Given a sequence of words and a line width, arrange the words in lines such that the total cost (penalty for extra spaces) is minimized.

Sample Input/Output:
Input: words = [3,2,2,5], line_width = 6
Output: Minimum cost arrangement
"""


class Solution:
    def Word_Wrap_DP(self, words: list[int], line_width: int) -> int:
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        n = len(words)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        for i in range(n - 1, -1, -1):
            current_length = 0
            for j in range(i, n):
                current_length += words[j]
                if j > i:
                    current_length += 1
                
                if current_length > line_width:
                    break
                
                cost = 0 if j == n - 1 else (line_width - current_length) ** 2
                dp[i] = min(dp[i], cost + dp[j + 1])
        
        return dp[0]
    
    def Word_Wrap_Greedy(self, words: list[int], line_width: int) -> int:
        """
        Greedy approach
        Time Complexity: O(n)
        Space Complexity: O(1)
        """
        n = len(words)
        cost = 0
        current_length = 0
        
        for i in range(n):
            if current_length + words[i] > line_width:
                if current_length > 0:
                    cost += (line_width - current_length) ** 2
                current_length = words[i]
            else:
                if current_length > 0:
                    current_length += 1
                current_length += words[i]
        
        if current_length > 0:
            cost += (line_width - current_length) ** 2
        
        return cost


def Test_WordWrap():
    solution = Solution()
    
    words = [3, 2, 2, 5]
    line_width = 6
    
    result1 = solution.Word_Wrap_DP(words, line_width)
    assert result1 >= 0
    
    result2 = solution.Word_Wrap_Greedy(words, line_width)
    assert result2 >= 0


if __name__ == "__main__":
    Test_WordWrap()
