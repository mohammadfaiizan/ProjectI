"""
Problem: Path More Than K Length
URL: https://www.geeksforgeeks.org/find-if-there-is-a-path-of-more-than-k-length-from-a-source/

Problem Statement:
Given a weighted graph and a source vertex, determine if there exists a path of length >= K starting from the source. Graph is represented as adjacency list with weights.

Sample Input/Output:
Input: Graph edges: (0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7), (2,8,2), (2,5,4), (3,4,9), (3,5,14), (4,5,10), (5,6,2), (6,7,1), (6,8,6), (7,8,7)
       Source = 0, K = 58
Output: true
Explanation: Path exists with length >= 58
"""


class Solution:
    def Path_More_Than_K_Length_DFS(self, graph, source, k):
        """
        DFS backtracking with weight accumulation
        Time Complexity: O(V!)
        Space Complexity: O(V)
        """
        V = len(graph)
        visited = [False] * V
        
        def dfs(u, current_length):
            if current_length >= k:
                return True
            
            visited[u] = True
            
            for v, weight in graph[u]:
                if not visited[v]:
                    if dfs(v, current_length + weight):
                        return True
            
            visited[u] = False
            return False
        
        return dfs(source, 0)


def Test_Path_More_Than_K_Length():
    solution = Solution()
    V = 9
    graph = [[] for _ in range(V)]
    
    graph[0].append((1, 4))
    graph[0].append((7, 8))
    graph[1].append((0, 4))
    graph[1].append((2, 8))
    graph[1].append((7, 11))
    graph[2].append((1, 8))
    graph[2].append((3, 7))
    graph[2].append((8, 2))
    graph[2].append((5, 4))
    graph[3].append((2, 7))
    graph[3].append((4, 9))
    graph[3].append((5, 14))
    graph[4].append((3, 9))
    graph[4].append((5, 10))
    graph[5].append((2, 4))
    graph[5].append((3, 14))
    graph[5].append((4, 10))
    graph[5].append((6, 2))
    graph[6].append((5, 2))
    graph[6].append((7, 1))
    graph[6].append((8, 6))
    graph[7].append((0, 8))
    graph[7].append((1, 11))
    graph[7].append((6, 1))
    graph[7].append((8, 7))
    graph[8].append((2, 2))
    graph[8].append((6, 6))
    graph[8].append((7, 7))
    
    source = 0
    k = 58
    result = solution.Path_More_Than_K_Length_DFS(graph, source, k)
    print(f"Path with length >= {k} exists:", result)


if __name__ == "__main__":
    Test_Path_More_Than_K_Length()
