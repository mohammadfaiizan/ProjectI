"""
Problem: Oliver and the Game (Euler Tour / Ancestor Check)
URL: https://www.hackerearth.com/practice/algorithms/graphs/topological-sort/practice-problems/algorithm/oliver-and-the-game-3/

Problem Statement:
Given a rooted tree, answer queries: is node X an ancestor of node Y? Use Euler tour (in-time/out-time) to check subtree relationship.

Sample Input/Output:
Input: tree edges = [[0,1],[0,2],[1,3],[1,4]], queries = [[0,3],[2,4]]
Output: [true, false]
"""


class Solution:
    def Oliver_Game_Euler_Tour(self, n, edges, queries, root):
        """
        DFS to compute in/out times, check subtree relationship
        Time Complexity: O(N+Q)
        Space Complexity: O(N)
        """
        adj = [[] for _ in range(n)]
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        inTime = [0] * n
        outTime = [0] * n
        timer = [0]
        
        def dfs(u, parent):
            inTime[u] = timer[0]
            timer[0] += 1
            for v in adj[u]:
                if v != parent:
                    dfs(v, u)
            outTime[u] = timer[0]
            timer[0] += 1
        
        dfs(root, -1)
        
        results = []
        for q in queries:
            x, y = q[0], q[1]
            isAncestor = (inTime[x] <= inTime[y] and outTime[x] >= outTime[y])
            results.append(isAncestor)
        
        return results


def Test_Oliver_Game_Euler_Tour():
    solution = Solution()
    
    n1 = 5
    edges1 = [[0, 1], [0, 2], [1, 3], [1, 4]]
    queries1 = [[0, 3], [2, 4], [1, 4]]
    result1 = solution.Oliver_Game_Euler_Tour(n1, edges1, queries1, 0)
    print("Test 1:", end=" ")
    for r in result1:
        print("true" if r else "false", end=" ")
    print()
    
    n2 = 4
    edges2 = [[0, 1], [0, 2], [2, 3]]
    queries2 = [[0, 3], [1, 2]]
    result2 = solution.Oliver_Game_Euler_Tour(n2, edges2, queries2, 0)
    print("Test 2:", end=" ")
    for r in result2:
        print("true" if r else "false", end=" ")
    print()
    
    n3 = 3
    edges3 = [[0, 1], [0, 2]]
    queries3 = [[0, 1], [0, 2], [1, 2]]
    result3 = solution.Oliver_Game_Euler_Tour(n3, edges3, queries3, 0)
    print("Test 3:", end=" ")
    for r in result3:
        print("true" if r else "false", end=" ")
    print()


if __name__ == "__main__":
    Test_Oliver_Game_Euler_Tour()
