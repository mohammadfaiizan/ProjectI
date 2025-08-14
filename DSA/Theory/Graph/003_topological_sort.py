"""
Topological Sorting Algorithms
This module implements various topological sorting algorithms and cycle detection in directed graphs.
"""

from collections import defaultdict, deque

class TopologicalSort:
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v):
        """Add a directed edge from u to v"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
    
    def get_neighbors(self, vertex):
        """Get neighbors (outgoing edges) of a vertex"""
        return self.graph[vertex]
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    # ==================== KAHN'S ALGORITHM (BFS-BASED) ====================
    
    def topological_sort_kahns(self):
        """
        Kahn's Algorithm for Topological Sorting (BFS-based)
        
        Algorithm:
        1. Calculate in-degree for all vertices
        2. Add all vertices with in-degree 0 to queue
        3. While queue is not empty:
           - Remove vertex from queue and add to result
           - For each neighbor, decrease in-degree by 1
           - If neighbor's in-degree becomes 0, add to queue
        4. If result contains all vertices, return it; else graph has cycle
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: Topologically sorted vertices, or None if cycle exists
        """
        # Step 1: Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        
        # Step 2: Initialize queue with vertices having in-degree 0
        queue = deque()
        for vertex in self.vertices:
            if in_degree[vertex] == 0:
                queue.append(vertex)
        
        result = []
        
        # Step 3: Process vertices
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree of neighbors
            for neighbor in self.get_neighbors(current):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Step 4: Check if topological sort is possible
        if len(result) != len(self.vertices):
            return None  # Cycle detected
        
        return result
    
    def has_cycle_kahns(self):
        """
        Detect cycle using Kahn's algorithm
        If topological sort is not possible, then cycle exists
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        return self.topological_sort_kahns() is None
    
    def topological_sort_kahns_detailed(self):
        """
        Kahn's Algorithm with detailed step-by-step information
        
        Returns:
            tuple: (result, steps, has_cycle)
        """
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        
        # Initialize queue with vertices having in-degree 0
        queue = deque()
        for vertex in self.vertices:
            if in_degree[vertex] == 0:
                queue.append(vertex)
        
        result = []
        steps = []
        step_count = 0
        
        # Process vertices
        while queue:
            step_count += 1
            current = queue.popleft()
            result.append(current)
            
            step_info = {
                'step': step_count,
                'processed': current,
                'queue_before': list(queue) + [current],
                'queue_after': list(queue),
                'in_degrees': in_degree.copy()
            }
            
            # Reduce in-degree of neighbors
            for neighbor in self.get_neighbors(current):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
            step_info['queue_after'] = list(queue)
            step_info['updated_in_degrees'] = in_degree.copy()
            steps.append(step_info)
        
        has_cycle = len(result) != len(self.vertices)
        
        return result, steps, has_cycle
    
    # ==================== DFS-BASED TOPOLOGICAL SORT ====================
    
    def topological_sort_dfs(self):
        """
        DFS-based Topological Sorting
        
        Algorithm:
        1. Perform DFS traversal
        2. After visiting all neighbors of a vertex, add it to stack
        3. Return reversed stack
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: Topologically sorted vertices, or None if cycle exists
        """
        # First check for cycle
        if self.has_cycle_dfs():
            return None
        
        visited = set()
        stack = []
        
        def dfs(vertex):
            visited.add(vertex)
            
            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs(neighbor)
            
            stack.append(vertex)  # Add to stack after visiting all neighbors
        
        # Visit all vertices
        for vertex in self.vertices:
            if vertex not in visited:
                dfs(vertex)
        
        return stack[::-1]  # Return reversed stack
    
    def has_cycle_dfs(self):
        """
        Detect cycle in directed graph using DFS (3-color method)
        
        Colors:
        - White (0): Unvisited
        - Gray (1): Currently being processed
        - Black (2): Completely processed
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        
        def dfs_cycle(vertex):
            color[vertex] = GRAY
            
            for neighbor in self.get_neighbors(vertex):
                if color[neighbor] == GRAY:  # Back edge found (cycle)
                    return True
                if color[neighbor] == WHITE and dfs_cycle(neighbor):
                    return True
            
            color[vertex] = BLACK
            return False
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if dfs_cycle(vertex):
                    return True
        
        return False
    
    def find_cycle_dfs(self):
        """
        Find and return a cycle in the directed graph using DFS
        
        Returns:
            list: Vertices forming a cycle, or None if no cycle exists
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        parent = {vertex: None for vertex in self.vertices}
        
        def dfs_find_cycle(vertex):
            color[vertex] = GRAY
            
            for neighbor in self.get_neighbors(vertex):
                if color[neighbor] == GRAY:  # Back edge found
                    # Reconstruct cycle
                    cycle = [neighbor]
                    current = vertex
                    while current != neighbor:
                        cycle.append(current)
                        current = parent[current]
                    return cycle[::-1]
                
                if color[neighbor] == WHITE:
                    parent[neighbor] = vertex
                    result = dfs_find_cycle(neighbor)
                    if result:
                        return result
            
            color[vertex] = BLACK
            return None
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                cycle = dfs_find_cycle(vertex)
                if cycle:
                    return cycle
        
        return None
    
    # ==================== ADVANCED TOPOLOGICAL SORTING ====================
    
    def all_topological_sorts(self):
        """
        Find all possible topological sorts of the DAG
        
        Returns:
            list: List of all possible topological orderings
        """
        if self.has_cycle_dfs():
            return []
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        
        result = []
        current_sort = []
        
        def backtrack():
            # Find vertices with in-degree 0
            available = [v for v in self.vertices if in_degree[v] == 0 and v not in current_sort]
            
            if not available:
                if len(current_sort) == len(self.vertices):
                    result.append(current_sort.copy())
                return
            
            for vertex in available:
                # Choose vertex
                current_sort.append(vertex)
                
                # Update in-degrees
                for neighbor in self.get_neighbors(vertex):
                    in_degree[neighbor] -= 1
                
                # Recurse
                backtrack()
                
                # Backtrack
                current_sort.pop()
                for neighbor in self.get_neighbors(vertex):
                    in_degree[neighbor] += 1
        
        backtrack()
        return result
    
    def lexicographically_smallest_topological_sort(self):
        """
        Find lexicographically smallest topological sort
        Uses priority queue (min-heap) instead of regular queue in Kahn's algorithm
        
        Returns:
            list: Lexicographically smallest topological ordering
        """
        import heapq
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        
        # Use min-heap instead of queue
        heap = []
        for vertex in self.vertices:
            if in_degree[vertex] == 0:
                heapq.heappush(heap, vertex)
        
        result = []
        
        while heap:
            current = heapq.heappop(heap)
            result.append(current)
            
            for neighbor in self.get_neighbors(current):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(heap, neighbor)
        
        return result if len(result) == len(self.vertices) else None
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in self.graph:
            print(f"{vertex} -> {self.graph[vertex]}")
    
    def get_in_degrees(self):
        """Get in-degree of all vertices"""
        in_degree = {vertex: 0 for vertex in self.vertices}
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        return in_degree
    
    def get_out_degrees(self):
        """Get out-degree of all vertices"""
        return {vertex: len(self.graph[vertex]) for vertex in self.vertices}
    
    def is_dag(self):
        """Check if the graph is a Directed Acyclic Graph (DAG)"""
        return not self.has_cycle_dfs()
    
    def get_sources(self):
        """Get all source vertices (in-degree = 0)"""
        in_degrees = self.get_in_degrees()
        return [vertex for vertex, degree in in_degrees.items() if degree == 0]
    
    def get_sinks(self):
        """Get all sink vertices (out-degree = 0)"""
        out_degrees = self.get_out_degrees()
        return [vertex for vertex, degree in out_degrees.items() if degree == 0]


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Topological Sorting Algorithms Demo ===\n")
    
    # Example 1: Simple DAG
    print("1. Simple DAG Example:")
    topo = TopologicalSort()
    
    # Add edges for a simple dependency graph
    # Course prerequisites: A->C, B->C, C->D, B->D
    edges = [('A', 'C'), ('B', 'C'), ('C', 'D'), ('B', 'D')]
    for u, v in edges:
        topo.add_edge(u, v)
    
    print("Graph:")
    topo.display()
    print(f"Is DAG: {topo.is_dag()}")
    print(f"In-degrees: {topo.get_in_degrees()}")
    print(f"Out-degrees: {topo.get_out_degrees()}")
    print(f"Sources: {topo.get_sources()}")
    print(f"Sinks: {topo.get_sinks()}")
    print()
    
    # Topological sorting using different methods
    print("Kahn's Algorithm:", topo.topological_sort_kahns())
    print("DFS-based:", topo.topological_sort_dfs())
    print("Lexicographically Smallest:", topo.lexicographically_smallest_topological_sort())
    print("All Topological Sorts:", topo.all_topological_sorts())
    print()
    
    # Example 2: Graph with Cycle
    print("2. Graph with Cycle Example:")
    cyclic_graph = TopologicalSort()
    cyclic_edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D')]
    for u, v in cyclic_edges:
        cyclic_graph.add_edge(u, v)
    
    print("Cyclic Graph:")
    cyclic_graph.display()
    print(f"Has Cycle (DFS): {cyclic_graph.has_cycle_dfs()}")
    print(f"Has Cycle (Kahn's): {cyclic_graph.has_cycle_kahns()}")
    print(f"Cycle Found: {cyclic_graph.find_cycle_dfs()}")
    print(f"Topological Sort (Kahn's): {cyclic_graph.topological_sort_kahns()}")
    print(f"Topological Sort (DFS): {cyclic_graph.topological_sort_dfs()}")
    print()
    
    # Example 3: Complex DAG
    print("3. Complex DAG Example:")
    complex_dag = TopologicalSort()
    
    # Task scheduling example
    tasks = [
        ('wake_up', 'shower'),
        ('wake_up', 'breakfast'),
        ('shower', 'dress'),
        ('breakfast', 'brush_teeth'),
        ('dress', 'leave_home'),
        ('brush_teeth', 'leave_home'),
        ('wake_up', 'check_weather'),
        ('check_weather', 'dress')
    ]
    
    for u, v in tasks:
        complex_dag.add_edge(u, v)
    
    print("Task Scheduling DAG:")
    complex_dag.display()
    print(f"Topological Order (Kahn's): {complex_dag.topological_sort_kahns()}")
    print(f"Topological Order (DFS): {complex_dag.topological_sort_dfs()}")
    print()
    
    # Detailed Kahn's algorithm execution
    print("4. Detailed Kahn's Algorithm Execution:")
    simple_dag = TopologicalSort()
    simple_edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for u, v in simple_edges:
        simple_dag.add_edge(u, v)
    
    result, steps, has_cycle = simple_dag.topological_sort_kahns_detailed()
    print(f"Result: {result}")
    print(f"Has Cycle: {has_cycle}")
    print("Step-by-step execution:")
    for step in steps:
        print(f"Step {step['step']}: Processed {step['processed']}")
        print(f"  Queue before: {step['queue_before']}")
        print(f"  Queue after: {step['queue_after']}")
        print(f"  Updated in-degrees: {step['updated_in_degrees']}")
        print() 