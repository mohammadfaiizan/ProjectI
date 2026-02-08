"""
Problem: Journey to the Moon
URL: https://www.hackerrank.com/challenges/journey-to-the-moon/

Problem Statement:
Given N astronauts and pairs of astronauts from the same country, count the number of ways to choose 2 astronauts from different countries.

Sample Input/Output:
Input: n=5, pairs = [[0,1],[2,3],[0,4]]
Output: 6
"""


class Solution:
    def Journey_Moon_DFS(self, n, pairs):
        """
        Find connected component sizes via DFS, compute pairs combinatorially
        Time Complexity: O(V+E)
        Space Complexity: O(V+E)
        """
        adj = [[] for _ in range(n)]
        for p in pairs:
            adj[p[0]].append(p[1])
            adj[p[1]].append(p[0])
        
        visited = [False] * n
        componentSizes = []
        
        def dfs(u):
            visited[u] = True
            size = 1
            for v in adj[u]:
                if not visited[v]:
                    size += dfs(v)
            return size
        
        for i in range(n):
            if not visited[i]:
                componentSizes.append(dfs(i))
        
        totalPairs = n * (n - 1) // 2
        sameCountryPairs = 0
        
        for size in componentSizes:
            sameCountryPairs += size * (size - 1) // 2
        
        return totalPairs - sameCountryPairs


def Test_Journey_Moon_DFS():
    solution = Solution()
    
    n1 = 5
    pairs1 = [[0, 1], [2, 3], [0, 4]]
    print(f"Test 1: {solution.Journey_Moon_DFS(n1, pairs1)}")
    
    n2 = 4
    pairs2 = [[0, 2]]
    print(f"Test 2: {solution.Journey_Moon_DFS(n2, pairs2)}")
    
    n3 = 6
    pairs3 = [[0, 1], [2, 3], [4, 5]]
    print(f"Test 3: {solution.Journey_Moon_DFS(n3, pairs3)}")
    
    n4 = 3
    pairs4 = []
    print(f"Test 4: {solution.Journey_Moon_DFS(n4, pairs4)}")


if __name__ == "__main__":
    Test_Journey_Moon_DFS()
