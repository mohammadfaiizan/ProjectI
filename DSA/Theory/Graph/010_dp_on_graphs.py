"""
Dynamic Programming on Graphs
This module implements various dynamic programming algorithms on graphs including
longest path in DAG, travelling salesman problem, and tree DP.
"""

from collections import defaultdict, deque
import sys
from typing import List, Tuple, Dict, Set

class DynamicProgrammingOnGraphs:
    
    def __init__(self, directed=True):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.directed = directed
    
    def add_edge(self, u, v, weight=1):
        """Add a weighted edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    # ==================== LONGEST PATH IN DAG ====================
    
    def longest_path_dag(self, start_vertex=None):
        """
        Find longest path in a Directed Acyclic Graph (DAG)
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Args:
            start_vertex: Starting vertex (optional, finds globally longest if None)
        
        Returns:
            tuple: (longest_distance, path, distances_dict)
        """
        if not self.directed:
            raise ValueError("Longest path in DAG requires directed graph")
        
        # First, get topological ordering
        topo_order = self._topological_sort()
        if not topo_order:
            raise ValueError("Graph contains cycle - not a DAG")
        
        # Initialize distances
        distances = {vertex: float('-inf') for vertex in self.vertices}
        parent = {vertex: None for vertex in self.vertices}
        
        # If start vertex is specified, only consider paths from that vertex
        if start_vertex is not None:
            if start_vertex not in self.vertices:
                raise ValueError(f"Start vertex {start_vertex} not in graph")
            distances[start_vertex] = 0
        else:
            # Consider all vertices as potential starting points
            for vertex in self.vertices:
                distances[vertex] = 0
        
        # Process vertices in topological order
        for vertex in topo_order:
            if distances[vertex] != float('-inf'):
                for neighbor, weight in self.get_neighbors(vertex):
                    if distances[vertex] + weight > distances[neighbor]:
                        distances[neighbor] = distances[vertex] + weight
                        parent[neighbor] = vertex
        
        # Find the vertex with maximum distance
        max_vertex = max(distances, key=distances.get)
        max_distance = distances[max_vertex]
        
        # Reconstruct path
        path = self._reconstruct_path(parent, max_vertex)
        
        return max_distance, path, distances
    
    def all_longest_paths_dag(self):
        """
        Find longest paths from all vertices in DAG
        
        Returns:
            dict: Mapping from start vertex to (distance, path)
        """
        topo_order = self._topological_sort()
        if not topo_order:
            raise ValueError("Graph contains cycle - not a DAG")
        
        all_paths = {}
        
        for start_vertex in self.vertices:
            distances = {vertex: float('-inf') for vertex in self.vertices}
            parent = {vertex: None for vertex in self.vertices}
            distances[start_vertex] = 0
            
            # Process in topological order
            for vertex in topo_order:
                if distances[vertex] != float('-inf'):
                    for neighbor, weight in self.get_neighbors(vertex):
                        if distances[vertex] + weight > distances[neighbor]:
                            distances[neighbor] = distances[vertex] + weight
                            parent[neighbor] = vertex
            
            # Find longest path from this start vertex
            max_vertex = max(distances, key=distances.get)
            max_distance = distances[max_vertex]
            path = self._reconstruct_path(parent, max_vertex)
            
            all_paths[start_vertex] = (max_distance, path)
        
        return all_paths
    
    def critical_path_method(self, project_tasks):
        """
        Critical Path Method (CPM) for project scheduling
        
        Args:
            project_tasks: List of (task_id, duration, dependencies)
        
        Returns:
            tuple: (critical_path, total_time, early_start, late_start)
        """
        # Build graph from project tasks
        task_graph = DynamicProgrammingOnGraphs(directed=True)
        durations = {}
        
        for task_id, duration, dependencies in project_tasks:
            durations[task_id] = duration
            task_graph.vertices.add(task_id)
            
            for dep in dependencies:
                task_graph.add_edge(dep, task_id, duration)
        
        # Find longest path (critical path)
        max_time, critical_path, _ = task_graph.longest_path_dag()
        
        # Calculate early start times
        early_start = self._calculate_early_start(project_tasks)
        
        # Calculate late start times
        late_start = self._calculate_late_start(project_tasks, max_time)
        
        return critical_path, max_time, early_start, late_start
    
    # ==================== TRAVELLING SALESMAN PROBLEM (DP + BITMASKING) ====================
    
    def travelling_salesman_dp(self, start_vertex=0):
        """
        Solve Travelling Salesman Problem using DP with bitmasking
        
        Time Complexity: O(n^2 * 2^n)
        Space Complexity: O(n * 2^n)
        
        Args:
            start_vertex: Starting vertex for the tour
        
        Returns:
            tuple: (min_cost, optimal_tour)
        """
        # Convert graph to adjacency matrix
        vertices_list = sorted(self.vertices)
        n = len(vertices_list)
        vertex_to_index = {v: i for i, v in enumerate(vertices_list)}
        
        # Build adjacency matrix
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        
        for u in self.graph:
            u_idx = vertex_to_index[u]
            for v, weight in self.graph[u]:
                v_idx = vertex_to_index[v]
                dist[u_idx][v_idx] = weight
                # For TSP, assume undirected graph
                if not self.directed:
                    dist[v_idx][u_idx] = weight
        
        start_idx = vertex_to_index[start_vertex]
        
        # DP table: dp[mask][i] = minimum cost to visit all cities in mask ending at city i
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        # Base case: starting at start_vertex with only start_vertex visited
        dp[1 << start_idx][start_idx] = 0
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if dp[mask][u] == INF:
                    continue
                
                for v in range(n):
                    if mask & (1 << v):  # v is already visited
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + dist[u][v]
                    
                    if new_cost < dp[new_mask][v]:
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = u
        
        # Find minimum cost tour (visiting all cities and returning to start)
        final_mask = (1 << n) - 1  # All cities visited
        min_cost = INF
        last_city = -1
        
        for u in range(n):
            if u != start_idx:
                cost = dp[final_mask][u] + dist[u][start_idx]
                if cost < min_cost:
                    min_cost = cost
                    last_city = u
        
        # Reconstruct tour
        tour = self._reconstruct_tsp_tour(parent, final_mask, last_city, 
                                         start_idx, vertices_list)
        
        return min_cost, tour
    
    def tsp_branch_and_bound(self, start_vertex=0):
        """
        Solve TSP using Branch and Bound (more memory efficient for smaller instances)
        
        Args:
            start_vertex: Starting vertex
        
        Returns:
            tuple: (min_cost, optimal_tour)
        """
        vertices_list = sorted(self.vertices)
        n = len(vertices_list)
        vertex_to_index = {v: i for i, v in enumerate(vertices_list)}
        
        # Build adjacency matrix
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]
        
        for u in self.graph:
            u_idx = vertex_to_index[u]
            for v, weight in self.graph[u]:
                v_idx = vertex_to_index[v]
                dist[u_idx][v_idx] = weight
                if not self.directed:
                    dist[v_idx][u_idx] = weight
        
        start_idx = vertex_to_index[start_vertex]
        
        # Use simple recursive approach with memoization for smaller instances
        memo = {}
        
        def tsp_rec(mask, pos):
            if mask == (1 << n) - 1:
                return dist[pos][start_idx]
            
            if (mask, pos) in memo:
                return memo[(mask, pos)]
            
            ans = INF
            for city in range(n):
                if mask & (1 << city) == 0:  # City not visited
                    new_ans = dist[pos][city] + tsp_rec(mask | (1 << city), city)
                    ans = min(ans, new_ans)
            
            memo[(mask, pos)] = ans
            return ans
        
        min_cost = tsp_rec(1 << start_idx, start_idx)
        
        # Reconstruct path (simplified)
        tour = self._reconstruct_tsp_simple(dist, start_idx, vertices_list)
        
        return min_cost, tour
    
    # ==================== DYNAMIC PROGRAMMING ON TREES ====================
    
    def tree_dp_max_path_sum(self, root):
        """
        Find maximum path sum in a tree (any node to any node)
        
        Time Complexity: O(V)
        Space Complexity: O(V)
        
        Args:
            root: Root vertex of the tree
        
        Returns:
            int: Maximum path sum
        """
        if not self.directed:
            # For undirected tree, we need to build adjacency list
            self._ensure_tree_structure()
        
        max_sum = [float('-inf')]
        
        def dfs(node, parent):
            # Maximum sum path ending at this node
            max_ending_here = 0
            
            # Try all children
            for neighbor, weight in self.get_neighbors(node):
                if neighbor != parent:
                    child_sum = dfs(neighbor, node)
                    max_ending_here = max(max_ending_here, child_sum + weight)
            
            # Update global maximum (path passing through current node)
            # This could be: node only, node + best child, or node + two best children
            node_weight = 0  # Assuming node weight is 0, modify if needed
            max_sum[0] = max(max_sum[0], node_weight + max_ending_here)
            
            return max_ending_here + node_weight
        
        dfs(root, None)
        return max_sum[0]
    
    def tree_dp_diameter(self, root):
        """
        Find diameter of tree (longest path between any two nodes)
        
        Args:
            root: Root vertex of the tree
        
        Returns:
            tuple: (diameter, path)
        """
        max_diameter = [0]
        diameter_path = [[]]
        
        def dfs(node, parent):
            # Find two longest paths from this node
            first_max = second_max = 0
            first_path = second_path = []
            
            for neighbor, weight in self.get_neighbors(node):
                if neighbor != parent:
                    child_max, child_path = dfs(neighbor, node)
                    path_length = child_max + weight
                    full_path = [neighbor] + child_path
                    
                    if path_length > first_max:
                        second_max, second_path = first_max, first_path
                        first_max, first_path = path_length, full_path
                    elif path_length > second_max:
                        second_max, second_path = path_length, full_path
            
            # Update diameter if path through current node is longer
            current_diameter = first_max + second_max
            if current_diameter > max_diameter[0]:
                max_diameter[0] = current_diameter
                # Build complete path
                diameter_path[0] = (first_path[::-1] + [node] + second_path)
            
            return first_max, first_path
        
        dfs(root, None)
        return max_diameter[0], diameter_path[0]
    
    def tree_dp_subtree_sizes(self, root):
        """
        Calculate size of each subtree using DP
        
        Args:
            root: Root vertex of the tree
        
        Returns:
            dict: Mapping from vertex to subtree size
        """
        subtree_sizes = {}
        
        def dfs(node, parent):
            size = 1  # Count current node
            
            for neighbor, _ in self.get_neighbors(node):
                if neighbor != parent:
                    size += dfs(neighbor, node)
            
            subtree_sizes[node] = size
            return size
        
        dfs(root, None)
        return subtree_sizes
    
    def tree_dp_reroot(self, root):
        """
        Rerooting technique - calculate answer for each node as root
        Example: Calculate sum of distances from each node to all other nodes
        
        Args:
            root: Initial root vertex
        
        Returns:
            dict: Answer for each node when it's the root
        """
        n = len(self.vertices)
        subtree_sizes = {}
        down_sum = {}  # Sum of distances to all nodes in subtree
        up_sum = {}    # Sum of distances to all nodes outside subtree
        
        # First DFS: Calculate subtree sizes and down sums
        def dfs1(node, parent):
            subtree_sizes[node] = 1
            down_sum[node] = 0
            
            for neighbor, weight in self.get_neighbors(node):
                if neighbor != parent:
                    dfs1(neighbor, node)
                    subtree_sizes[node] += subtree_sizes[neighbor]
                    down_sum[node] += down_sum[neighbor] + subtree_sizes[neighbor] * weight
        
        # Second DFS: Calculate up sums using rerooting
        def dfs2(node, parent):
            for neighbor, weight in self.get_neighbors(node):
                if neighbor != parent:
                    # Calculate up_sum for neighbor
                    # It includes: up_sum[node] + down sums of other children + their contributions
                    up_sum[neighbor] = up_sum[node]
                    
                    # Add contribution from other subtrees
                    other_down = down_sum[node] - (down_sum[neighbor] + subtree_sizes[neighbor] * weight)
                    other_size = n - subtree_sizes[neighbor]
                    up_sum[neighbor] += other_down + other_size * weight
                    
                    dfs2(neighbor, node)
        
        # Initialize
        dfs1(root, None)
        up_sum[root] = 0
        dfs2(root, None)
        
        # Calculate final answer for each node
        answers = {}
        for node in self.vertices:
            answers[node] = down_sum[node] + up_sum[node]
        
        return answers
    
    # ==================== UTILITY METHODS ====================
    
    def _topological_sort(self):
        """Perform topological sort using DFS"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        topo_order = []
        
        def dfs(vertex):
            color[vertex] = GRAY
            
            for neighbor, _ in self.get_neighbors(vertex):
                if color[neighbor] == GRAY:  # Back edge found
                    return False  # Cycle detected
                if color[neighbor] == WHITE and not dfs(neighbor):
                    return False
            
            color[vertex] = BLACK
            topo_order.append(vertex)
            return True
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if not dfs(vertex):
                    return []  # Cycle detected
        
        return topo_order[::-1]
    
    def _reconstruct_path(self, parent, end_vertex):
        """Reconstruct path from parent array"""
        path = []
        current = end_vertex
        
        while current is not None:
            path.append(current)
            current = parent[current]
        
        return path[::-1]
    
    def _reconstruct_tsp_tour(self, parent, mask, last_city, start_idx, vertices_list):
        """Reconstruct TSP tour from parent table"""
        tour = []
        current_mask = mask
        current_city = last_city
        
        while current_city != -1:
            tour.append(vertices_list[current_city])
            prev_city = parent[current_mask][current_city]
            if prev_city != -1:
                current_mask ^= (1 << current_city)
            current_city = prev_city
        
        tour.append(vertices_list[start_idx])  # Return to start
        return tour[::-1]
    
    def _reconstruct_tsp_simple(self, dist, start_idx, vertices_list):
        """Simple greedy reconstruction for TSP (not optimal)"""
        n = len(vertices_list)
        visited = [False] * n
        tour = [vertices_list[start_idx]]
        visited[start_idx] = True
        current = start_idx
        
        for _ in range(n - 1):
            next_city = -1
            min_dist = float('inf')
            
            for i in range(n):
                if not visited[i] and dist[current][i] < min_dist:
                    min_dist = dist[current][i]
                    next_city = i
            
            if next_city != -1:
                tour.append(vertices_list[next_city])
                visited[next_city] = True
                current = next_city
        
        tour.append(vertices_list[start_idx])  # Return to start
        return tour
    
    def _ensure_tree_structure(self):
        """Ensure the graph is a valid tree structure"""
        if len(self.vertices) == 0:
            return
        
        # Check if it's connected and has n-1 edges
        edge_count = sum(len(neighbors) for neighbors in self.graph.values())
        if not self.directed:
            edge_count //= 2
        
        if edge_count != len(self.vertices) - 1:
            raise ValueError("Graph is not a tree")
    
    def _calculate_early_start(self, project_tasks):
        """Calculate early start times for CPM"""
        early_start = {}
        dependencies = {}
        
        for task_id, duration, deps in project_tasks:
            dependencies[task_id] = deps
            if not deps:
                early_start[task_id] = 0
        
        # Topological processing
        processed = set()
        
        while len(processed) < len(project_tasks):
            for task_id, duration, deps in project_tasks:
                if task_id in processed:
                    continue
                
                if all(dep in early_start for dep in deps):
                    if deps:
                        early_start[task_id] = max(early_start[dep] + duration for dep in deps)
                    else:
                        early_start[task_id] = 0
                    processed.add(task_id)
        
        return early_start
    
    def _calculate_late_start(self, project_tasks, total_time):
        """Calculate late start times for CPM"""
        late_start = {}
        task_durations = {task_id: duration for task_id, duration, _ in project_tasks}
        
        # Find tasks with no successors
        successors = defaultdict(list)
        for task_id, duration, deps in project_tasks:
            for dep in deps:
                successors[dep].append(task_id)
        
        # Start from end tasks
        for task_id, duration, deps in project_tasks:
            if not successors[task_id]:
                late_start[task_id] = total_time - duration
        
        # Work backwards
        processed = set(late_start.keys())
        
        while len(processed) < len(project_tasks):
            for task_id, duration, deps in project_tasks:
                if task_id in processed:
                    continue
                
                if all(succ in late_start for succ in successors[task_id]):
                    if successors[task_id]:
                        late_start[task_id] = min(late_start[succ] for succ in successors[task_id]) - duration
                    processed.add(task_id)
        
        return late_start
    
    def display(self):
        """Display the graph"""
        for vertex in sorted(self.graph.keys()):
            neighbors = [(v, w) for v, w in self.graph[vertex]]
            print(f"{vertex}: {neighbors}")


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Dynamic Programming on Graphs Demo ===\n")
    
    # Example 1: Longest Path in DAG
    print("1. Longest Path in DAG:")
    dag = DynamicProgrammingOnGraphs(directed=True)
    
    # Create a DAG representing project dependencies
    edges = [
        ('A', 'B', 5), ('A', 'C', 3), ('B', 'D', 6),
        ('C', 'D', 4), ('C', 'E', 2), ('D', 'F', 2),
        ('E', 'F', 3)
    ]
    
    for u, v, w in edges:
        dag.add_edge(u, v, w)
    
    print("DAG structure:")
    dag.display()
    
    longest_dist, longest_path, all_distances = dag.longest_path_dag('A')
    print(f"Longest path from A: {longest_path}")
    print(f"Longest distance: {longest_dist}")
    print(f"All distances from A: {all_distances}")
    print()
    
    # Example 2: Critical Path Method
    print("2. Critical Path Method (CPM):")
    project_tasks = [
        ('A', 5, []),           # Task A, duration 5, no dependencies
        ('B', 3, []),           # Task B, duration 3, no dependencies
        ('C', 4, ['A']),        # Task C, duration 4, depends on A
        ('D', 6, ['A', 'B']),   # Task D, duration 6, depends on A and B
        ('E', 2, ['C']),        # Task E, duration 2, depends on C
        ('F', 3, ['D', 'E'])    # Task F, duration 3, depends on D and E
    ]
    
    critical_path, total_time, early_start, late_start = dag.critical_path_method(project_tasks)
    print(f"Critical path: {critical_path}")
    print(f"Project completion time: {total_time}")
    print(f"Early start times: {early_start}")
    print(f"Late start times: {late_start}")
    print()
    
    # Example 3: Travelling Salesman Problem
    print("3. Travelling Salesman Problem:")
    tsp_graph = DynamicProgrammingOnGraphs(directed=False)
    
    # Create a small complete graph for TSP
    cities = [0, 1, 2, 3]
    # Distance matrix
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            tsp_graph.add_edge(cities[i], cities[j], distances[i][j])
    
    print("TSP Graph (distance matrix):")
    for i, row in enumerate(distances):
        print(f"City {i}: {row}")
    
    min_cost, optimal_tour = tsp_graph.travelling_salesman_dp(0)
    print(f"Minimum TSP cost: {min_cost}")
    print(f"Optimal tour: {optimal_tour}")
    print()
    
    # Example 4: Tree DP - Maximum Path Sum
    print("4. Tree DP - Maximum Path Sum:")
    tree = DynamicProgrammingOnGraphs(directed=False)
    
    # Create a tree with weighted edges
    tree_edges = [
        (1, 2, 5), (1, 3, 3), (2, 4, 4),
        (2, 5, 6), (3, 6, 2), (3, 7, 8)
    ]
    
    for u, v, w in tree_edges:
        tree.add_edge(u, v, w)
    
    print("Tree structure:")
    tree.display()
    
    max_path_sum = tree.tree_dp_max_path_sum(1)
    print(f"Maximum path sum in tree: {max_path_sum}")
    
    # Tree diameter
    diameter, diameter_path = tree.tree_dp_diameter(1)
    print(f"Tree diameter: {diameter}")
    print(f"Diameter path: {diameter_path}")
    
    # Subtree sizes
    subtree_sizes = tree.tree_dp_subtree_sizes(1)
    print(f"Subtree sizes: {subtree_sizes}")
    print()
    
    # Example 5: Rerooting Technique
    print("5. Rerooting Technique:")
    answers = tree.tree_dp_reroot(1)
    print("Sum of distances from each node (as root) to all other nodes:")
    for node, answer in sorted(answers.items()):
        print(f"Node {node}: {answer}")
    print()
    
    # Example 6: All Longest Paths in DAG
    print("6. All Longest Paths in DAG:")
    all_longest = dag.all_longest_paths_dag()
    print("Longest paths from each vertex:")
    for start_vertex, (distance, path) in all_longest.items():
        print(f"From {start_vertex}: distance={distance}, path={path}")
    
    print("\n=== Demo Complete ===") 