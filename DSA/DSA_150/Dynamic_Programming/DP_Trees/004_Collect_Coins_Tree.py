"""
Problem: Collect Coins in a Tree
URL: https://leetcode.com/problems/collect-coins-in-a-tree/description/

Problem Statement:
There exists an undirected and unrooted tree with n nodes indexed from 0 to n - 1. You are given an integer n and a 2D integer array edges of length n - 1, where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.
You are also given an array coins of length n where coins[i] can be either 0 or 1, where 1 indicates the presence of a coin at node i.
Initially, you choose to start at any node in this tree. Then, you can perform the following operations any number of times:
- Collect all the coins that are within a distance of at most 2 from your current node.
- Move to an adjacent node.
Return the minimum number of edges you need to travel to collect all the coins and go back to your starting node.

Sample Input/Output:
Input: coins = [1,0,0,0,1,1], edges = [[0,1],[1,2],[2,3],[3,4],[4,5]]
Output: 2
Explanation: Start at node 2, collect coins at nodes 0, 4, and 5, and return to node 2.

Input: coins = [0,0,0,1,1,1], edges = [[0,1],[1,2],[2,3],[3,4],[4,5]]
Output: 4
Explanation: Start at node 0, travel to node 3, collect all coins, and return to node 0.
"""

from typing import List, Dict, Set
from collections import defaultdict, deque

class Solution:
    def Collect_Coins_Brute_Force(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        Brute Force - Try all possible starting nodes and paths
        Time Complexity: O(n!)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def Get_Nodes_Within_Distance(start: int, distance: int) -> Set[int]:
            if distance < 0:
                return set()
            
            visited = set()
            queue = deque([(start, 0)])
            result = set()
            
            while queue:
                node, dist = queue.popleft()
                if node in visited:
                    continue
                
                visited.add(node)
                if dist <= distance:
                    result.add(node)
                
                if dist < distance:
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1))
            
            return result
        
        def Can_Collect_All_Coins(start: int, path: List[int]) -> bool:
            collected = set()
            
            for node in path:
                reachable = Get_Nodes_Within_Distance(node, 2)
                for r in reachable:
                    if coins[r] == 1:
                        collected.add(r)
            
            coin_nodes = {i for i in range(n) if coins[i] == 1}
            return collected >= coin_nodes
        
        def Generate_All_Paths(start: int, visited: Set[int], path: List[int]) -> List[List[int]]:
            if len(path) > n:
                return []
            
            coin_nodes = {i for i in range(n) if coins[i] == 1}
            if not coin_nodes:
                return [[start]]
            
            if Can_Collect_All_Coins(start, path):
                return [path + [start]]
            
            all_paths = []
            
            for neighbor in graph[path[-1]]:
                if neighbor not in visited or (neighbor == start and len(path) > 1):
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    paths = Generate_All_Paths(start, new_visited, path + [neighbor])
                    all_paths.extend(paths)
            
            return all_paths
        
        min_moves = float('inf')
        
        for start in range(n):
            paths = Generate_All_Paths(start, {start}, [start])
            for path in paths:
                if path[0] == path[-1]:
                    moves = len(path) - 1
                    min_moves = min(min_moves, moves)
        
        return min_moves if min_moves != float('inf') else 0
    
    def Collect_Coins_Tree_Trimming(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        Tree Trimming - Remove unnecessary leaf nodes
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        degree = [len(graph[i]) for i in range(n)]
        
        queue = deque()
        for i in range(n):
            if degree[i] <= 1 and coins[i] == 0:
                queue.append(i)
        
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                graph[neighbor].discard(node)
                degree[neighbor] -= 1
                
                if degree[neighbor] <= 1 and coins[neighbor] == 0:
                    queue.append(neighbor)
        
        queue = deque()
        for i in range(n):
            if degree[i] <= 1 and coins[i] == 1:
                queue.append(i)
        
        for _ in range(2):
            next_queue = deque()
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    graph[neighbor].discard(node)
                    degree[neighbor] -= 1
                    
                    if degree[neighbor] <= 1:
                        next_queue.append(neighbor)
            queue = next_queue
        
        remaining_edges = 0
        for i in range(n):
            remaining_edges += degree[i]
        
        return max(0, remaining_edges - 2) if remaining_edges > 0 else 0
    
    def Collect_Coins_DFS_Optimal(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        DFS Optimal - Tree DP with pruning
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        removed = [False] * n
        
        def Remove_Leaves_Without_Coins():
            changed = True
            while changed:
                changed = False
                leaves_to_remove = []
                
                for i in range(n):
                    if removed[i]:
                        continue
                    
                    neighbors = [j for j in graph[i] if not removed[j]]
                    
                    if len(neighbors) <= 1 and coins[i] == 0:
                        leaves_to_remove.append(i)
                
                for leaf in leaves_to_remove:
                    removed[leaf] = True
                    changed = True
        
        Remove_Leaves_Without_Coins()
        
        def Remove_Two_Layers_From_Coin_Leaves():
            for _ in range(2):
                leaves_to_remove = []
                
                for i in range(n):
                    if removed[i]:
                        continue
                    
                    neighbors = [j for j in graph[i] if not removed[j]]
                    
                    if len(neighbors) <= 1 and coins[i] == 1:
                        leaves_to_remove.append(i)
                
                for leaf in leaves_to_remove:
                    removed[leaf] = True
                    coins[leaf] = 0
                
                leaves_to_remove = []
                
                for i in range(n):
                    if removed[i]:
                        continue
                    
                    neighbors = [j for j in graph[i] if not removed[j]]
                    
                    if len(neighbors) <= 1:
                        leaves_to_remove.append(i)
                
                for leaf in leaves_to_remove:
                    removed[leaf] = True
        
        Remove_Two_Layers_From_Coin_Leaves()
        
        remaining_nodes = [i for i in range(n) if not removed[i]]
        
        if len(remaining_nodes) <= 1:
            return 0
        
        remaining_edges = 0
        for u, v in edges:
            if not removed[u] and not removed[v]:
                remaining_edges += 1
        
        return 2 * remaining_edges - 2 if remaining_edges > 0 else 0
    
    def Collect_Coins_Topological_Sort(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        Topological Sort - Layer by layer removal
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        degree = [len(graph[i]) for i in range(n)]
        removed = [False] * n
        
        def Remove_Layer(condition_func):
            queue = deque()
            for i in range(n):
                if not removed[i] and degree[i] <= 1 and condition_func(i):
                    queue.append(i)
            
            while queue:
                node = queue.popleft()
                if removed[node]:
                    continue
                
                removed[node] = True
                
                for neighbor in graph[node]:
                    if not removed[neighbor]:
                        degree[neighbor] -= 1
                        if degree[neighbor] <= 1 and condition_func(neighbor):
                            queue.append(neighbor)
        
        Remove_Layer(lambda x: coins[x] == 0)
        
        for _ in range(2):
            to_remove = []
            for i in range(n):
                if not removed[i] and degree[i] <= 1:
                    to_remove.append(i)
            
            for node in to_remove:
                removed[node] = True
                for neighbor in graph[node]:
                    if not removed[neighbor]:
                        degree[neighbor] -= 1
        
        remaining_edges = 0
        for u, v in edges:
            if not removed[u] and not removed[v]:
                remaining_edges += 1
        
        return max(0, 2 * remaining_edges - 2) if remaining_edges > 0 else 0
    
    def Collect_Coins_BFS_Distance(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        BFS Distance - Calculate minimum path using BFS
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        coin_nodes = {i for i in range(n) if coins[i] == 1}
        
        if not coin_nodes:
            return 0
        
        def Get_Min_Distance_To_Cover_All():
            min_distance = float('inf')
            
            for start in range(n):
                visited = set()
                queue = deque([(start, 0, set())])
                
                while queue:
                    node, dist, covered = queue.popleft()
                    
                    if node in visited:
                        continue
                    
                    visited.add(node)
                    
                    new_covered = covered.copy()
                    for i in range(n):
                        if coins[i] == 1:
                            if self.Get_Distance(node, i, graph) <= 2:
                                new_covered.add(i)
                    
                    if new_covered >= coin_nodes:
                        min_distance = min(min_distance, dist * 2)
                        continue
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            queue.append((neighbor, dist + 1, new_covered))
            
            return min_distance
        
        return Get_Min_Distance_To_Cover_All() if Get_Min_Distance_To_Cover_All() != float('inf') else 0
    
    def Get_Distance(self, start: int, end: int, graph: Dict) -> int:
        """Helper function to get distance between two nodes"""
        if start == end:
            return 0
        
        visited = set()
        queue = deque([(start, 0)])
        
        while queue:
            node, dist = queue.popleft()
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node == end:
                return dist
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        
        return float('inf')
    
    def Collect_Coins_Greedy_Pruning(self, coins: List[int], edges: List[List[int]]) -> int:
        """
        Greedy Pruning - Greedy approach with tree pruning
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        n = len(coins)
        if n <= 1:
            return 0
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def Prune_No_Coin_Leaves():
            removed = True
            while removed:
                removed = False
                to_remove = []
                
                for node in list(adj.keys()):
                    if len(adj[node]) == 1 and coins[node] == 0:
                        to_remove.append(node)
                
                for node in to_remove:
                    neighbor = list(adj[node])[0]
                    adj[neighbor].remove(node)
                    del adj[node]
                    removed = True
        
        Prune_No_Coin_Leaves()
        
        for _ in range(2):
            to_remove = []
            for node in list(adj.keys()):
                if len(adj[node]) == 1:
                    to_remove.append(node)
            
            for node in to_remove:
                if node in adj:
                    if adj[node]:
                        neighbor = list(adj[node])[0]
                        adj[neighbor].discard(node)
                    del adj[node]
        
        remaining_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
        
        return max(0, 2 * remaining_edges - 2) if remaining_edges > 0 else 0

def Test_Collect_Coins():
    solution = Solution()
    
    test_cases = [
        ([1,0,0,0,1,1], [[0,1],[1,2],[2,3],[3,4],[4,5]], 2),
        ([0,0,0,1,1,1], [[0,1],[1,2],[2,3],[3,4],[4,5]], 4),
        ([1,0,0,0,1], [[0,1],[1,2],[2,3],[3,4]], 2),
        ([0], [], 0),
        ([1], [], 0),
        ([1,1], [[0,1]], 0),
        ([1,0,1,0,1,1], [[0,1],[1,2],[2,3],[3,4],[4,5]], 4)
    ]
    
    methods = [
        ("Tree Trimming", solution.Collect_Coins_Tree_Trimming),
        ("DFS Optimal", solution.Collect_Coins_DFS_Optimal),
        ("Topological Sort", solution.Collect_Coins_Topological_Sort),
        ("Greedy Pruning", solution.Collect_Coins_Greedy_Pruning)
    ]
    
    for coins, edges, expected in test_cases:
        print(f"Coins: {coins}")
        print(f"Edges: {edges}")
        print(f"Expected: {expected}")
        
        if len(coins) <= 6:
            try:
                result_bf = solution.Collect_Coins_Brute_Force(coins.copy(), [edge[:] for edge in edges])
                print(f"Brute Force: {result_bf}")
            except Exception as e:
                print(f"Brute Force: Error - {e}")
        
        for method_name, method in methods:
            try:
                result = method(coins.copy(), [edge[:] for edge in edges])
                print(f"{method_name}: {result}")
            except Exception as e:
                print(f"{method_name}: Error - {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    Test_Collect_Coins()
