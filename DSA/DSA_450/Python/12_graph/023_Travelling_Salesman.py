"""
Problem: Travelling Salesman Problem
URL: https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming-solution/

Problem Statement:
Find the shortest route that visits all cities exactly once and returns to the starting city. Given a distance matrix representing distances between cities.

Sample Input/Output:
Input: 4 cities, distance matrix
Output: Minimum cost to visit all cities and return
"""

import itertools


class Solution:
    def TSP_DP_Bitmask(self, dist):
        """
        DP with bitmask
        Time Complexity: O(2^n * n^2)
        Space Complexity: O(2^n * n)
        """
        n = len(dist)
        totalStates = 1 << n
        dp = [[float('inf')] * n for _ in range(totalStates)]
        
        dp[1][0] = 0
        
        for mask in range(1, totalStates):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                if dp[mask][u] == float('inf'):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    newMask = mask | (1 << v)
                    if dist[u][v] > 0:
                        dp[newMask][v] = min(dp[newMask][v], dp[mask][u] + dist[u][v])
        
        finalMask = totalStates - 1
        result = float('inf')
        for u in range(1, n):
            if dist[u][0] > 0 and dp[finalMask][u] != float('inf'):
                result = min(result, dp[finalMask][u] + dist[u][0])
        
        return int(result) if result != float('inf') else -1
    
    def TSP_Brute_Force(self, dist):
        """
        Permutation-based
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        n = len(dist)
        cities = list(range(1, n))
        
        minCost = float('inf')
        for perm in itertools.permutations(cities):
            cost = dist[0][perm[0]]
            for i in range(len(perm) - 1):
                cost += dist[perm[i]][perm[i + 1]]
            cost += dist[perm[-1]][0]
            minCost = min(minCost, cost)
        
        return int(minCost) if minCost != float('inf') else -1


def Test_TSP():
    solution = Solution()
    
    print("Test Case 1: 4 cities distance matrix")
    dist1 = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    print(f"DP Bitmask Result: {solution.TSP_DP_Bitmask(dist1)}")
    print(f"Brute Force Result: {solution.TSP_Brute_Force(dist1)}")
    
    print("\nTest Case 2: 3 cities")
    dist2 = [
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]
    print(f"DP Bitmask Result: {solution.TSP_DP_Bitmask(dist2)}")
    print(f"Brute Force Result: {solution.TSP_Brute_Force(dist2)}")
    
    print("\nTest Case 3: 5 cities")
    dist3 = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 1],
        [7, 3, 5, 1, 0]
    ]
    print(f"DP Bitmask Result: {solution.TSP_DP_Bitmask(dist3)}")


if __name__ == "__main__":
    Test_TSP()
