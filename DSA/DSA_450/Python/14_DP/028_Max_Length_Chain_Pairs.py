"""
Problem: Maximum Length Chain of Pairs
URL: https://practice.geeksforgeeks.org/problems/max-length-chain/1

Problem Statement:
You are given N pairs of numbers. In every pair, the first number is always smaller than the second number. A pair (c, d) can follow another pair (a, b) if b < c. Chain of pairs can be formed in this fashion. Find the longest chain which can be formed from a given set of pairs.

Sample Input/Output:
Input: [[5,24], [39,60], [15,28], [27,40], [50,90]]
Output: 3
"""

class Solution:
    def Chain_DP(self, pairs, n):
        """
        DP approach
        Time Complexity: O(n^2)
        Space Complexity: O(n)
        """
        pairs.sort()
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def Chain_Greedy(self, pairs, n):
        """
        Greedy approach
        Time Complexity: O(n log n)
        Space Complexity: O(1)
        """
        pairs.sort(key=lambda x: x[1])
        count = 1
        last = pairs[0][1]
        for i in range(1, n):
            if pairs[i][0] > last:
                count += 1
                last = pairs[i][1]
        return count

def Test_Chain():
    solution = Solution()
    pairs = [[5,24], [39,60], [15,28], [27,40], [50,90]]
    print("DP:", solution.Chain_DP(pairs, len(pairs)))
    print("Greedy:", solution.Chain_Greedy(pairs, len(pairs)))

if __name__ == "__main__":
    Test_Chain()
