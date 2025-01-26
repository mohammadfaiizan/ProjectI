from collections import deque

class GraphTraversal:
    def __init__(self):
        self.graph = {}

    def create_from_adj_list(self, adj_list):
        self.graph = adj_list

    def bfs(self, start):
        res = []
        visited = [False] * len(self.graph)
        queue = deque()
        queue.append(start)
        visited[start] = True

        while queue:
            node = queue.popleft()
            res.append(node)
            for nbr in self.graph[node]:
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        return res
        
    def dfs(self, start):
        n = len(self.graph)
        visited = [False] * n
        stack = [start]
        res = []
        
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                res.append(node)
                for nbr in reversed(self.graph[node]):
                    if not visited[nbr]:
                        stack.append(nbr)
        return res
    
    def dfs_recursive(self, node, visited, res):
        visited[node] = True
        res.append(node)
        
        # Recursively visit all unvisited neighbors
        for nbr in self.graph[node]:
            if not visited[nbr]:
                self.dfs_recursive(nbr, visited, res)
        
        return res

if __name__ == "__main__":
    adj_list = [
        [1, 2, 3],    # Node 0 is connected to 1, 2, 3
        [0, 4],       # Node 1 is connected to 0, 4
        [0, 4, 5],    # Node 2 is connected to 0, 4, 5
        [0, 5],       # Node 3 is connected to 0, 5
        [1, 2, 6],    # Node 4 is connected to 1, 2, 6
        [2, 3, 6],    # Node 5 is connected to 2, 3, 6
        [4, 5, 7],    # Node 6 is connected to 4, 5, 7
        [6]           # Node 7 is connected to 6
    ]
    
    g = GraphTraversal()
    g.create_from_adj_list(adj_list)
    print("BFS:", g.bfs(0))
    print("DFS:", g.dfs(0))
