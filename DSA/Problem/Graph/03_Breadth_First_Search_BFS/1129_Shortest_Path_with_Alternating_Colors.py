"""
1129. Shortest Path with Alternating Colors - Multiple Approaches
Difficulty: Medium

You are given an integer n, the number of nodes in a directed graph where the nodes are labeled from 0 to n - 1. Each edge is red or blue in this graph, and there could be self-edges and parallel edges.

You are given two arrays redEdges and blueEdges where:
- redEdges[i] = [ai, bi] indicates that there is a directed red edge from node ai to node bi in the graph.
- blueEdges[i] = [ai, bi] indicates that there is a directed blue edge from node ai to node bi in the graph.

Return an array answer of length n, where each answer[i] is the length of the shortest path from node 0 to node i such that the edge colors alternate along the path, or -1 if no such path exists.
"""

from typing import List, Dict, Tuple
from collections import deque, defaultdict

class ShortestPathAlternatingColors:
    """Multiple approaches for shortest path with color constraints"""
    
    def shortestAlternatingPaths_bfs_state_tracking(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        """
        Approach 1: BFS with State Tracking
        
        Track both node and last edge color in BFS state.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency lists for red and blue edges
        red_graph = defaultdict(list)
        blue_graph = defaultdict(list)
        
        for u, v in redEdges:
            red_graph[u].append(v)
        
        for u, v in blueEdges:
            blue_graph[u].append(v)
        
        # BFS with state (node, last_color, distance)
        # last_color: 0 = red, 1 = blue, -1 = start (can use either)
        queue = deque([(0, -1, 0)])  # (node, last_color, distance)
        visited = set([(0, -1)])
        
        # Result array
        result = [-1] * n
        result[0] = 0
        
        while queue:
            node, last_color, dist = queue.popleft()
            
            # Try red edges (if last was not red)
            if last_color != 0:
                for neighbor in red_graph[node]:
                    state = (neighbor, 0)
                    if state not in visited:
                        visited.add(state)
                        if result[neighbor] == -1:
                            result[neighbor] = dist + 1
                        queue.append((neighbor, 0, dist + 1))
            
            # Try blue edges (if last was not blue)
            if last_color != 1:
                for neighbor in blue_graph[node]:
                    state = (neighbor, 1)
                    if state not in visited:
                        visited.add(state)
                        if result[neighbor] == -1:
                            result[neighbor] = dist + 1
                        queue.append((neighbor, 1, dist + 1))
        
        return result
    
    def shortestAlternatingPaths_dual_bfs(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        """
        Approach 2: Dual BFS (Red-first and Blue-first)
        
        Run two separate BFS: one starting with red, one with blue.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency lists
        red_graph = defaultdict(list)
        blue_graph = defaultdict(list)
        
        for u, v in redEdges:
            red_graph[u].append(v)
        
        for u, v in blueEdges:
            blue_graph[u].append(v)
        
        def bfs_with_first_color(first_color: int) -> List[int]:
            """BFS starting with specific color (0=red, 1=blue)"""
            distances = [-1] * n
            distances[0] = 0
            
            # Queue: (node, expected_color, distance)
            queue = deque([(0, first_color, 0)])
            visited = set([(0, first_color)])
            
            while queue:
                node, expected_color, dist = queue.popleft()
                
                if expected_color == 0:  # Expecting red edge
                    for neighbor in red_graph[node]:
                        state = (neighbor, 1)  # Next expect blue
                        if state not in visited:
                            visited.add(state)
                            if distances[neighbor] == -1:
                                distances[neighbor] = dist + 1
                            queue.append((neighbor, 1, dist + 1))
                
                else:  # Expecting blue edge
                    for neighbor in blue_graph[node]:
                        state = (neighbor, 0)  # Next expect red
                        if state not in visited:
                            visited.add(state)
                            if distances[neighbor] == -1:
                                distances[neighbor] = dist + 1
                            queue.append((neighbor, 0, dist + 1))
            
            return distances
        
        # Run BFS starting with red and blue
        red_first = bfs_with_first_color(0)
        blue_first = bfs_with_first_color(1)
        
        # Combine results
        result = []
        for i in range(n):
            if red_first[i] == -1 and blue_first[i] == -1:
                result.append(-1)
            elif red_first[i] == -1:
                result.append(blue_first[i])
            elif blue_first[i] == -1:
                result.append(red_first[i])
            else:
                result.append(min(red_first[i], blue_first[i]))
        
        return result
    
    def shortestAlternatingPaths_dijkstra_like(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        """
        Approach 3: Dijkstra-like Algorithm
        
        Use priority queue to ensure shortest paths are found first.
        
        Time: O((V + E) log V), Space: O(V + E)
        """
        import heapq
        
        # Build adjacency lists
        red_graph = defaultdict(list)
        blue_graph = defaultdict(list)
        
        for u, v in redEdges:
            red_graph[u].append(v)
        
        for u, v in blueEdges:
            blue_graph[u].append(v)
        
        # Priority queue: (distance, node, last_color)
        pq = [(0, 0, -1)]  # Start with distance 0, node 0, no last color
        distances = {}  # (node, last_color) -> distance
        result = [-1] * n
        
        while pq:
            dist, node, last_color = heapq.heappop(pq)
            
            # Skip if we've seen this state with better distance
            if (node, last_color) in distances:
                continue
            
            distances[(node, last_color)] = dist
            
            # Update result if this is the first time reaching this node
            if result[node] == -1:
                result[node] = dist
            
            # Try red edges (if last was not red)
            if last_color != 0:
                for neighbor in red_graph[node]:
                    if (neighbor, 0) not in distances:
                        heapq.heappush(pq, (dist + 1, neighbor, 0))
            
            # Try blue edges (if last was not blue)
            if last_color != 1:
                for neighbor in blue_graph[node]:
                    if (neighbor, 1) not in distances:
                        heapq.heappush(pq, (dist + 1, neighbor, 1))
        
        return result
    
    def shortestAlternatingPaths_level_bfs(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        """
        Approach 4: Level-by-Level BFS
        
        Process nodes level by level to ensure shortest paths.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build adjacency lists
        red_graph = defaultdict(list)
        blue_graph = defaultdict(list)
        
        for u, v in redEdges:
            red_graph[u].append(v)
        
        for u, v in blueEdges:
            blue_graph[u].append(v)
        
        # Initialize result
        result = [-1] * n
        result[0] = 0
        
        # Current level: set of (node, last_color) states
        current_level = {(0, 0), (0, 1)}  # Can start with either color
        visited = set(current_level)
        distance = 0
        
        while current_level:
            next_level = set()
            
            for node, last_color in current_level:
                # Update result if not set yet
                if result[node] == -1:
                    result[node] = distance
                
                # Try red edges (if last was not red)
                if last_color != 0:
                    for neighbor in red_graph[node]:
                        state = (neighbor, 0)
                        if state not in visited:
                            visited.add(state)
                            next_level.add(state)
                
                # Try blue edges (if last was not blue)
                if last_color != 1:
                    for neighbor in blue_graph[node]:
                        state = (neighbor, 1)
                        if state not in visited:
                            visited.add(state)
                            next_level.add(state)
            
            current_level = next_level
            distance += 1
        
        return result

def test_shortest_path_alternating_colors():
    """Test shortest path with alternating colors algorithms"""
    solver = ShortestPathAlternatingColors()
    
    test_cases = [
        (3, [[0,1],[1,2]], [], [0,1,-1], "Only red edges"),
        (3, [[0,1]], [[2,1]], [0,1,-1], "Mixed edges"),
        (3, [[1,0]], [[2,1]], [0,-1,-1], "No path from 0"),
        (3, [[0,1],[0,2]], [[1,0]], [0,1,1], "Multiple paths"),
        (5, [[0,1],[1,2],[2,3],[3,4]], [[1,2],[2,3],[3,1]], [0,1,2,3,7], "Complex case"),
    ]
    
    algorithms = [
        ("BFS State Tracking", solver.shortestAlternatingPaths_bfs_state_tracking),
        ("Dual BFS", solver.shortestAlternatingPaths_dual_bfs),
        ("Dijkstra-like", solver.shortestAlternatingPaths_dijkstra_like),
        ("Level BFS", solver.shortestAlternatingPaths_level_bfs),
    ]
    
    print("=== Testing Shortest Path with Alternating Colors ===")
    
    for n, redEdges, blueEdges, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, Red: {redEdges}, Blue: {blueEdges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, redEdges, blueEdges)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_shortest_path_alternating_colors()

"""
Shortest Path with Alternating Colors demonstrates
advanced BFS techniques with state constraints
and multi-dimensional graph traversal strategies.
"""
