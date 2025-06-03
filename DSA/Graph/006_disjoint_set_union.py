"""
Disjoint Set Union (Union-Find) Data Structure
This module implements various optimizations of Union-Find and its applications.
"""

from collections import defaultdict

class DisjointSetUnion:
    """
    Disjoint Set Union (Union-Find) data structure with optimizations
    Supports Union by Rank and Path Compression
    """
    
    def __init__(self, elements=None):
        """
        Initialize DSU with given elements
        
        Args:
            elements: List or set of elements to initialize with
        """
        self.parent = {}
        self.rank = {}
        self.size = {}  # Size of each component
        self.num_components = 0
        
        if elements:
            for element in elements:
                self.make_set(element)
    
    # ==================== BASIC OPERATIONS ====================
    
    def make_set(self, x):
        """
        Create a new set containing only element x
        
        Time Complexity: O(1)
        
        Args:
            x: Element to create set for
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
            self.num_components += 1
    
    def find_basic(self, x):
        """
        Basic find operation without path compression
        
        Time Complexity: O(log n) average, O(n) worst case
        
        Args:
            x: Element to find root of
        
        Returns:
            Root of the set containing x
        """
        if x not in self.parent:
            self.make_set(x)
        
        if self.parent[x] != x:
            return self.find_basic(self.parent[x])
        return x
    
    def find(self, x):
        """
        Find operation with path compression
        
        Time Complexity: O(α(n)) amortized, where α is inverse Ackermann function
        
        Args:
            x: Element to find root of
        
        Returns:
            Root of the set containing x
        """
        if x not in self.parent:
            self.make_set(x)
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union_basic(self, x, y):
        """
        Basic union operation without rank optimization
        
        Time Complexity: O(log n) with path compression
        
        Args:
            x, y: Elements to union
        
        Returns:
            bool: True if union performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Always attach second tree to first
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        del self.size[root_y]
        self.num_components -= 1
        
        return True
    
    def union_by_rank(self, x, y):
        """
        Union operation with union by rank optimization
        
        Time Complexity: O(α(n)) amortized
        
        Args:
            x, y: Elements to union
        
        Returns:
            bool: True if union performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank: attach smaller tree to larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
            del self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            del self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            self.size[root_x] += self.size[root_y]
            del self.size[root_y]
        
        self.num_components -= 1
        return True
    
    def union_by_size(self, x, y):
        """
        Union operation with union by size optimization
        
        Time Complexity: O(α(n)) amortized
        
        Args:
            x, y: Elements to union
        
        Returns:
            bool: True if union performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by size: attach smaller tree to larger tree
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
            del self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            del self.size[root_y]
        
        self.num_components -= 1
        return True
    
    # Default union operation (uses union by rank)
    def union(self, x, y):
        """Default union operation using union by rank"""
        return self.union_by_rank(x, y)
    
    # ==================== QUERY OPERATIONS ====================
    
    def connected(self, x, y):
        """
        Check if two elements are in the same set
        
        Time Complexity: O(α(n))
        
        Args:
            x, y: Elements to check
        
        Returns:
            bool: True if in same set, False otherwise
        """
        return self.find(x) == self.find(y)
    
    def get_component_size(self, x):
        """
        Get size of the component containing x
        
        Time Complexity: O(α(n))
        
        Args:
            x: Element to check
        
        Returns:
            int: Size of component containing x
        """
        root = self.find(x)
        return self.size[root]
    
    def get_num_components(self):
        """
        Get number of connected components
        
        Time Complexity: O(1)
        
        Returns:
            int: Number of connected components
        """
        return self.num_components
    
    def get_all_components(self):
        """
        Get all connected components
        
        Time Complexity: O(n)
        
        Returns:
            dict: Mapping from root to list of elements in component
        """
        components = defaultdict(list)
        
        for element in self.parent:
            root = self.find(element)
            components[root].append(element)
        
        return dict(components)
    
    def get_component_roots(self):
        """
        Get all component roots
        
        Time Complexity: O(n)
        
        Returns:
            list: List of all component roots
        """
        roots = set()
        for element in self.parent:
            roots.add(self.find(element))
        return list(roots)
    
    # ==================== APPLICATIONS ====================
    
    def cycle_detection_undirected(self, edges):
        """
        Detect cycle in undirected graph using Union-Find
        
        Time Complexity: O(E * α(V))
        
        Args:
            edges: List of edges as tuples (u, v) or (u, v, weight)
        
        Returns:
            tuple: (has_cycle, cycle_edge)
        """
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
            else:
                u, v = edge[0], edge[1]
            
            if self.connected(u, v):
                return True, (u, v)  # Cycle detected
            
            self.union(u, v)
        
        return False, None
    
    def kruskals_mst(self, edges, num_vertices=None):
        """
        Find Minimum Spanning Tree using Kruskal's Algorithm
        
        Time Complexity: O(E log E)
        
        Args:
            edges: List of weighted edges as tuples (u, v, weight)
            num_vertices: Number of vertices (optional)
        
        Returns:
            tuple: (mst_edges, total_weight)
        """
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        mst_edges = []
        total_weight = 0
        
        for u, v, weight in sorted_edges:
            if not self.connected(u, v):
                self.union(u, v)
                mst_edges.append((u, v, weight))
                total_weight += weight
                
                # Stop when we have V-1 edges (if num_vertices is known)
                if num_vertices and len(mst_edges) == num_vertices - 1:
                    break
        
        return mst_edges, total_weight
    
    def count_islands(self, grid):
        """
        Count number of islands in a 2D grid using Union-Find
        
        Time Complexity: O(m * n * α(m * n))
        
        Args:
            grid: 2D list where 1 represents land and 0 represents water
        
        Returns:
            int: Number of islands
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        
        # Convert 2D coordinates to 1D
        def get_index(r, c):
            return r * cols + c
        
        # Add all land cells to DSU
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    self.make_set(get_index(r, c))
        
        # Connect adjacent land cells
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    current = get_index(r, c)
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            grid[nr][nc] == 1):
                            neighbor = get_index(nr, nc)
                            self.union(current, neighbor)
        
        # Count number of components (islands)
        return self.get_num_components()
    
    def redundant_connection(self, edges):
        """
        Find redundant connection that creates a cycle
        
        Time Complexity: O(E * α(V))
        
        Args:
            edges: List of edges in order they were added
        
        Returns:
            tuple: First edge that creates a cycle
        """
        for u, v in edges:
            if self.connected(u, v):
                return (u, v)  # This edge creates a cycle
            self.union(u, v)
        
        return None
    
    # ==================== UTILITY METHODS ====================
    
    def clear(self):
        """Clear all data structures"""
        self.parent.clear()
        self.rank.clear()
        self.size.clear()
        self.num_components = 0
    
    def display_structure(self):
        """Display the current structure of DSU"""
        print("DSU Structure:")
        print(f"Number of components: {self.num_components}")
        
        components = self.get_all_components()
        for i, (root, elements) in enumerate(components.items(), 1):
            print(f"Component {i} (root={root}, size={len(elements)}): {elements}")
    
    def display_detailed(self):
        """Display detailed information about DSU"""
        print("Detailed DSU Information:")
        print(f"Parent: {self.parent}")
        print(f"Rank: {self.rank}")
        print(f"Size: {self.size}")
        print(f"Components: {self.num_components}")


# ==================== OPTIMIZED IMPLEMENTATIONS ====================

class WeightedUnionFind(DisjointSetUnion):
    """
    Weighted Union-Find for applications requiring edge weights
    """
    
    def __init__(self, elements=None):
        super().__init__(elements)
        self.edge_weights = {}  # Store weights of edges in MST
    
    def union_with_weight(self, x, y, weight):
        """Union operation that also stores edge weight"""
        if self.union(x, y):
            # Store the edge weight
            root_x, root_y = self.find(x), self.find(y)
            # Since one of them is now parent, store weight appropriately
            edge_key = tuple(sorted([x, y]))
            self.edge_weights[edge_key] = weight
            return True
        return False


class PathTrackingUnionFind(DisjointSetUnion):
    """
    Union-Find that tracks paths for reconstruction
    """
    
    def __init__(self, elements=None):
        super().__init__(elements)
        self.path_parent = {}  # For path reconstruction
    
    def union_with_path(self, x, y):
        """Union operation that maintains path information"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Maintain path information
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.path_parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            self.path_parent[root_y] = root_x
            if self.rank[root_x] == self.rank[root_y]:
                self.rank[root_x] += 1
        
        self.num_components -= 1
        return True


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Disjoint Set Union (Union-Find) Demo ===\n")
    
    # Example 1: Basic Union-Find Operations
    print("1. Basic Union-Find Operations:")
    dsu = DisjointSetUnion()
    
    # Add elements
    elements = [1, 2, 3, 4, 5, 6]
    for elem in elements:
        dsu.make_set(elem)
    
    print(f"Initial components: {dsu.get_num_components()}")
    dsu.display_structure()
    print()
    
    # Perform unions
    print("Performing unions: (1,2), (3,4), (1,3)")
    dsu.union(1, 2)
    dsu.union(3, 4)
    dsu.union(1, 3)
    
    print(f"After unions: {dsu.get_num_components()} components")
    dsu.display_structure()
    print()
    
    # Check connectivity
    print("Connectivity tests:")
    print(f"Connected(1, 4): {dsu.connected(1, 4)}")
    print(f"Connected(1, 5): {dsu.connected(1, 5)}")
    print(f"Component size of 1: {dsu.get_component_size(1)}")
    print()
    
    # Example 2: Cycle Detection in Undirected Graph
    print("2. Cycle Detection in Undirected Graph:")
    cycle_dsu = DisjointSetUnion()
    
    # Edges that will form a cycle
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 5)]
    
    has_cycle, cycle_edge = cycle_dsu.cycle_detection_undirected(edges)
    print(f"Edges: {edges}")
    print(f"Has cycle: {has_cycle}")
    if has_cycle:
        print(f"Cycle detected at edge: {cycle_edge}")
    print()
    
    # Example 3: Kruskal's MST Algorithm
    print("3. Kruskal's Minimum Spanning Tree:")
    mst_dsu = DisjointSetUnion()
    
    # Weighted edges: (u, v, weight)
    weighted_edges = [
        (0, 1, 4), (0, 7, 8), (1, 2, 8), (1, 7, 11),
        (2, 3, 7), (2, 8, 2), (2, 5, 4), (3, 4, 9),
        (3, 5, 14), (4, 5, 10), (5, 6, 2), (6, 7, 1),
        (6, 8, 6), (7, 8, 7)
    ]
    
    mst_edges, total_weight = mst_dsu.kruskals_mst(weighted_edges, num_vertices=9)
    print(f"Original edges: {len(weighted_edges)}")
    print(f"MST edges: {mst_edges}")
    print(f"Total MST weight: {total_weight}")
    print()
    
    # Example 4: Count Islands
    print("4. Count Islands in 2D Grid:")
    island_dsu = DisjointSetUnion()
    
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    
    num_islands = island_dsu.count_islands(grid)
    print("Grid:")
    for row in grid:
        print(row)
    print(f"Number of islands: {num_islands}")
    print()
    
    # Example 5: Redundant Connection
    print("5. Find Redundant Connection:")
    redundant_dsu = DisjointSetUnion()
    
    graph_edges = [(1, 2), (1, 3), (2, 3)]
    redundant_edge = redundant_dsu.redundant_connection(graph_edges)
    print(f"Edges: {graph_edges}")
    print(f"Redundant edge: {redundant_edge}")
    print()
    
    # Example 6: Performance Comparison
    print("6. Performance Comparison:")
    print("Comparing basic union vs union by rank:")
    
    # Test basic union
    basic_dsu = DisjointSetUnion(range(100))
    for i in range(99):
        basic_dsu.union_basic(i, i + 1)
    
    # Test union by rank
    rank_dsu = DisjointSetUnion(range(100))
    for i in range(99):
        rank_dsu.union_by_rank(i, i + 1)
    
    print("Both should result in 1 component:")
    print(f"Basic union components: {basic_dsu.get_num_components()}")
    print(f"Union by rank components: {rank_dsu.get_num_components()}")
    
    # Display final structure
    print("\nFinal structure with union by rank:")
    rank_dsu.display_detailed() 