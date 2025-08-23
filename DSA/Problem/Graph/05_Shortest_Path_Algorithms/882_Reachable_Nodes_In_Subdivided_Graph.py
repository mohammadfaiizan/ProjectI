"""
882. Reachable Nodes In Subdivided Graph - Multiple Approaches
Difficulty: Medium

You are given an undirected graph (the "original graph") with n nodes labeled from 0 to n - 1. You decide to subdivide each edge in the graph into a chain of nodes, with the number of new nodes varying for each edge.

The graph is given as a 2D array of edges where edges[i] = [ui, vi, cnti] means there is an original edge between nodes ui and vi in the original graph, and cnti is the total number of new nodes that you will subdivide the edge into. Note that cnti == 0 means you will not subdivide the edge.

To subdivide the edge [ui, vi], replace it with (cnti + 1) new edges and cnti new nodes. The new nodes are x1, x2, ..., xcnti, and the new edges are [ui, x1], [x1, x2], [x2, x3], ..., [xcnti, vi].

In this new graph, you want to know how many nodes are reachable from the node 0 where a node is reachable if the distance from node 0 to that node is maxMoves or less.

Given the original graph and maxMoves, return the number of reachable nodes in the new graph.
"""

from typing import List, Dict, Tuple
import heapq
from collections import defaultdict

class ReachableNodesSubdividedGraph:
    """Multiple approaches to count reachable nodes in subdivided graph"""
    
    def reachableNodes_dijkstra_simulation(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        """
        Approach 1: Dijkstra with Edge Subdivision Simulation
        
        Use Dijkstra to find shortest distances, then simulate subdivision.
        
        Time: O(E log V), Space: O(V + E)
        """
        # Build adjacency list with edge weights
        graph = defaultdict(list)
        edge_map = {}
        
        for u, v, cnt in edges:
            graph[u].append((v, cnt + 1))  # Cost is cnt + 1 to traverse
            graph[v].append((u, cnt + 1))
            edge_map[(min(u, v), max(u, v))] = cnt
        
        # Dijkstra to find shortest distances from node 0
        distances = [float('inf')] * n
        distances[0] = 0
        pq = [(0, 0)]  # (distance, node)
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if dist > distances[node]:
                continue
            
            for neighbor, weight in graph[node]:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Count reachable original nodes
        reachable = sum(1 for d in distances if d <= maxMoves)
        
        # Count reachable subdivided nodes on each edge
        for u, v, cnt in edges:
            if cnt == 0:
                continue
            
            edge_key = (min(u, v), max(u, v))
            
            # Calculate how many subdivided nodes we can reach from each end
            from_u = max(0, maxMoves - distances[u])
            from_v = max(0, maxMoves - distances[v])
            
            # Total reachable subdivided nodes on this edge
            reachable += min(cnt, from_u + from_v)
        
        return reachable
    
    def reachableNodes_modified_dijkstra(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        """
        Approach 2: Modified Dijkstra with Edge Tracking
        
        Track how many nodes we can reach on each edge during Dijkstra.
        
        Time: O(E log V), Space: O(V + E)
        """
        # Build graph
        graph = defaultdict(list)
        for u, v, cnt in edges:
            graph[u].append((v, cnt + 1))
            graph[v].append((u, cnt + 1))
        
        # Dijkstra with edge usage tracking
        distances = {}
        pq = [(0, 0)]  # (distance, node)
        edge_used = defaultdict(int)  # Track nodes used on each edge
        
        while pq:
            dist, node = heapq.heappop(pq)
            
            if node in distances:
                continue
            
            distances[node] = dist
            
            for neighbor, weight in graph[node]:
                if neighbor not in distances:
                    new_dist = dist + weight
                    if new_dist <= maxMoves:
                        heapq.heappush(pq, (new_dist, neighbor))
                    
                    # Calculate how many subdivided nodes we can use on this edge
                    remaining_moves = maxMoves - dist
                    edge_key = (min(node, neighbor), max(node, neighbor))
                    
                    # Find the original edge info
                    for u, v, cnt in edges:
                        if (min(u, v), max(u, v)) == edge_key:
                            can_use = min(cnt, remaining_moves)
                            edge_used[edge_key] = max(edge_used[edge_key], can_use)
                            break
        
        # Count reachable nodes
        reachable_original = len(distances)
        reachable_subdivided = sum(edge_used.values())
        
        return reachable_original + reachable_subdivided
    
    def reachableNodes_optimized_dijkstra(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        """
        Approach 3: Optimized Dijkstra with Careful Edge Counting
        
        Optimize edge counting to avoid double counting.
        
        Time: O(E log V), Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, cnt in edges:
            graph[u].append((v, cnt + 1))
            graph[v].append((u, cnt + 1))
        
        # Dijkstra
        dist = {}
        pq = [(0, 0)]
        
        while pq:
            d, node = heapq.heappop(pq)
            
            if node in dist:
                continue
            
            dist[node] = d
            
            for neighbor, weight in graph[node]:
                if neighbor not in dist and d + weight <= maxMoves:
                    heapq.heappush(pq, (d + weight, neighbor))
        
        # Count reachable nodes
        result = len(dist)
        
        # For each edge, count reachable subdivided nodes
        for u, v, cnt in edges:
            if cnt == 0:
                continue
            
            # Distance from 0 to u and v
            dist_u = dist.get(u, float('inf'))
            dist_v = dist.get(v, float('inf'))
            
            # How many subdivided nodes can we reach from each end
            from_u = max(0, maxMoves - dist_u) if dist_u <= maxMoves else 0
            from_v = max(0, maxMoves - dist_v) if dist_v <= maxMoves else 0
            
            # Total reachable on this edge (avoid double counting)
            result += min(cnt, from_u + from_v)
        
        return result
    
    def reachableNodes_bfs_approach(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        """
        Approach 4: BFS-based Approach
        
        Use BFS to explore reachable nodes with move counting.
        
        Time: O(V * maxMoves), Space: O(V + E)
        """
        from collections import deque
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, cnt in edges:
            graph[u].append((v, cnt + 1))
            graph[v].append((u, cnt + 1))
        
        # BFS with move tracking
        queue = deque([(0, maxMoves)])  # (node, remaining_moves)
        visited = {}  # node -> max_remaining_moves_when_visited
        
        while queue:
            node, moves_left = queue.popleft()
            
            if node in visited and visited[node] >= moves_left:
                continue
            
            visited[node] = moves_left
            
            for neighbor, cost in graph[node]:
                if moves_left >= cost:
                    new_moves = moves_left - cost
                    if neighbor not in visited or visited[neighbor] < new_moves:
                        queue.append((neighbor, new_moves))
        
        # Count reachable original nodes
        result = len(visited)
        
        # Count subdivided nodes on edges
        for u, v, cnt in edges:
            if cnt == 0:
                continue
            
            moves_u = visited.get(u, -1)
            moves_v = visited.get(v, -1)
            
            from_u = max(0, moves_u) if moves_u >= 0 else 0
            from_v = max(0, moves_v) if moves_v >= 0 else 0
            
            result += min(cnt, from_u + from_v)
        
        return result
    
    def reachableNodes_dp_approach(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        """
        Approach 5: Dynamic Programming Approach
        
        Use DP to track reachable nodes with different move counts.
        
        Time: O(V * maxMoves), Space: O(V * maxMoves)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, cnt in edges:
            graph[u].append((v, cnt + 1))
            graph[v].append((u, cnt + 1))
        
        # DP: dp[node][moves] = True if node is reachable with exactly 'moves' moves
        dp = defaultdict(lambda: defaultdict(bool))
        dp[0][0] = True
        
        reachable_nodes = set([0])
        
        for moves in range(maxMoves + 1):
            for node in range(n):
                if dp[node][moves]:
                    for neighbor, cost in graph[node]:
                        if moves + cost <= maxMoves:
                            dp[neighbor][moves + cost] = True
                            reachable_nodes.add(neighbor)
        
        # Count original reachable nodes
        result = len(reachable_nodes)
        
        # Count subdivided nodes
        max_moves_to_node = {}
        for node in reachable_nodes:
            max_moves_to_node[node] = max(moves for moves in range(maxMoves + 1) if dp[node][moves])
        
        for u, v, cnt in edges:
            if cnt == 0:
                continue
            
            moves_u = maxMoves - max_moves_to_node.get(u, maxMoves + 1)
            moves_v = maxMoves - max_moves_to_node.get(v, maxMoves + 1)
            
            from_u = max(0, moves_u) if u in reachable_nodes else 0
            from_v = max(0, moves_v) if v in reachable_nodes else 0
            
            result += min(cnt, from_u + from_v)
        
        return result

def test_reachable_nodes_subdivided():
    """Test reachable nodes in subdivided graph algorithms"""
    solver = ReachableNodesSubdividedGraph()
    
    test_cases = [
        ([[0,1,10],[0,2,1],[1,2,2]], 6, 3, 13, "Example 1"),
        ([[0,1,4],[1,2,6],[0,2,8],[1,3,1]], 10, 4, 23, "Example 2"),
        ([[1,2,4],[1,4,5],[1,3,1],[2,3,4],[3,4,5]], 17, 5, 1, "Complex graph"),
    ]
    
    algorithms = [
        ("Dijkstra Simulation", solver.reachableNodes_dijkstra_simulation),
        ("Optimized Dijkstra", solver.reachableNodes_optimized_dijkstra),
        ("BFS Approach", solver.reachableNodes_bfs_approach),
    ]
    
    print("=== Testing Reachable Nodes in Subdivided Graph ===")
    
    for edges, maxMoves, n, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Edges: {edges}, MaxMoves: {maxMoves}, N: {n}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(edges, maxMoves, n)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Reachable: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_reachable_nodes_subdivided()

"""
Reachable Nodes in Subdivided Graph demonstrates advanced
shortest path algorithms with graph modification and
complex reachability analysis in dynamic graph structures.
"""
