"""
Problem: Edit Distance
URL: https://practice.geeksforgeeks.org/problems/edit-distance3702/1

Problem Statement:
Given two strings s and t. Find the minimum number of operations that need to be performed on str1 to convert it to str2. The possible operations are: Insert, Remove, Replace.

Sample Input/Output:
Input: "horse", "ros"
Output: 3
Explanation: horse -> rorse -> rose -> ros
"""

class Solution:
    def Edit_Distance_Edit_Dist_Recursive(self, s1, s2, m, n):
        """
        Recursive approach
        Time Complexity: O(3^(m+n))
        Space Complexity: O(m+n)
        """
        if m == 0:
            return n
        if n == 0:
            return m
        if s1[m-1] == s2[n-1]:
            return self.Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n-1)
        return 1 + min(self.Edit_Distance_Edit_Dist_Recursive(s1, s2, m, n-1),
                       self.Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n),
                       self.Edit_Distance_Edit_Dist_Recursive(s1, s2, m-1, n-1))

    def Edit_Distance_Edit_Dist_Memo(self, s1, s2, m, n):
        """
        Memoization approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        memo = [[-1] * (n+1) for _ in range(m+1)]
        return self.Edit_Dist_Memo_Helper(s1, s2, m, n, memo)

    def Edit_Dist_Memo_Helper(self, s1, s2, m, n, memo):
        if m == 0:
            return n
        if n == 0:
            return m
        if memo[m][n] != -1:
            return memo[m][n]
        if s1[m-1] == s2[n-1]:
            memo[m][n] = self.Edit_Dist_Memo_Helper(s1, s2, m-1, n-1, memo)
        else:
            memo[m][n] = 1 + min(self.Edit_Dist_Memo_Helper(s1, s2, m, n-1, memo),
                                 self.Edit_Dist_Memo_Helper(s1, s2, m-1, n, memo),
                                 self.Edit_Dist_Memo_Helper(s1, s2, m-1, n-1, memo))
        return memo[m][n]

    def Edit_Distance_Edit_Dist_Tab(self, s1, s2, m, n):
        """
        Tabulation approach
        Time Complexity: O(m*n)
        Space Complexity: O(m*n)
        """
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
        return dp[m][n]

    def Edit_Distance_Edit_Dist_Space(self, s1, s2, m, n):
        """
        Space optimized approach
        Time Complexity: O(m*n)
        Space Complexity: O(min(m,n))
        """
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m
        prev = [0] * (n+1)
        curr = [0] * (n+1)
        for j in range(n+1):
            prev[j] = j
        for i in range(1, m+1):
            curr[0] = i
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = 1 + min(curr[j-1], prev[j], prev[j-1])
            prev = curr[:]
        return curr[n]

def Test_Edit_Distance():
    solution = Solution()
    s1 = "horse"
    s2 = "ros"
    
    print("Recursive:", solution.Edit_Distance_Edit_Dist_Recursive(s1, s2, len(s1), len(s2)))
    print("Memoization:", solution.Edit_Distance_Edit_Dist_Memo(s1, s2, len(s1), len(s2)))
    print("Tabulation:", solution.Edit_Distance_Edit_Dist_Tab(s1, s2, len(s1), len(s2)))
    print("Space Optimized:", solution.Edit_Distance_Edit_Dist_Space(s1, s2, len(s1), len(s2)))

if __name__ == "__main__":
    Test_Edit_Distance()
