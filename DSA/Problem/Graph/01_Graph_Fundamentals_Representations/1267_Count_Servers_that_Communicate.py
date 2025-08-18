"""
1267. Count Servers that Communicate
Difficulty: Medium

Problem:
You are given a map of a server center, represented as a m x n integer matrix grid, 
where 1 means that there is a server at that position and 0 means that there is no server. 
Two servers are said to communicate if they are on the same row or on the same column.

Return the number of servers that communicate with at least one other server.

Examples:
Input: grid = [[1,0],[0,1]]
Output: 0
Explanation: No servers can communicate with others.

Input: grid = [[1,0],[1,1]]
Output: 3
Explanation: All three servers can communicate with at least one other server.

Input: grid = [[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
Output: 4
Explanation: The two servers in the first row can communicate with each other. 
The two servers in the third column can communicate with each other. 
The server at right bottom corner cannot communicate with any other server.

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m <= 250
- 1 <= n <= 250
- grid[i][j] == 0 or 1
"""

from typing import List
from collections import defaultdict

class Solution:
    def countServers_approach1_row_col_count(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Row and Column counting
        
        Count servers in each row and column. A server can communicate
        if its row has >1 servers OR its column has >1 servers.
        
        Time: O(M*N)
        Space: O(M+N)
        """
        m, n = len(grid), len(grid[0])
        
        # Count servers in each row and column
        row_count = [0] * m
        col_count = [0] * n
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    row_count[i] += 1
                    col_count[j] += 1
        
        # Count communicating servers
        communicating = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if row_count[i] > 1 or col_count[j] > 1:
                        communicating += 1
        
        return communicating
    
    def countServers_approach2_two_pass(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Two-pass solution
        
        First pass: collect server positions and count per row/col
        Second pass: check which servers can communicate
        
        Time: O(M*N)
        Space: O(K) where K = number of servers
        """
        m, n = len(grid), len(grid[0])
        
        # First pass: find all servers and count per row/column
        servers = []
        row_count = defaultdict(int)
        col_count = defaultdict(int)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    servers.append((i, j))
                    row_count[i] += 1
                    col_count[j] += 1
        
        # Second pass: count communicating servers
        communicating = 0
        for row, col in servers:
            if row_count[row] > 1 or col_count[col] > 1:
                communicating += 1
        
        return communicating
    
    def countServers_approach3_union_find(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Union-Find approach
        
        Connect all servers in same row/column using Union-Find.
        Count servers in components with size > 1.
        
        Time: O(M*N*α(K)) where K = number of servers
        Space: O(K)
        """
        m, n = len(grid), len(grid[0])
        
        # Find all servers
        servers = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    servers.append((i, j))
        
        if len(servers) <= 1:
            return 0
        
        # Union-Find implementation
        parent = list(range(len(servers)))
        size = [1] * len(servers)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if size[px] < size[py]:
                    px, py = py, px
                parent[py] = px
                size[px] += size[py]
        
        # Connect servers in same row or column
        for i in range(len(servers)):
            for j in range(i + 1, len(servers)):
                r1, c1 = servers[i]
                r2, c2 = servers[j]
                if r1 == r2 or c1 == c2:  # Same row or column
                    union(i, j)
        
        # Count servers in components with size > 1
        communicating = 0
        for i in range(len(servers)):
            if size[find(i)] > 1:
                communicating += 1
        
        return communicating
    
    def countServers_approach4_graph_components(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Graph-based connected components
        
        Build a graph where servers are connected if they share row/column.
        Find connected components and count servers in components > 1.
        
        Time: O(M*N + K²) where K = number of servers
        Space: O(K²)
        """
        m, n = len(grid), len(grid[0])
        
        # Find all servers
        servers = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    servers.append((i, j))
        
        if len(servers) <= 1:
            return 0
        
        # Build adjacency list
        adj = defaultdict(list)
        for i in range(len(servers)):
            for j in range(i + 1, len(servers)):
                r1, c1 = servers[i]
                r2, c2 = servers[j]
                if r1 == r2 or c1 == c2:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find connected components using DFS
        visited = [False] * len(servers)
        communicating = 0
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(len(servers)):
            if not visited[i]:
                component = []
                dfs(i, component)
                if len(component) > 1:
                    communicating += len(component)
        
        return communicating
    
    def countServers_approach5_optimized_single_pass(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Optimized single-pass with early termination
        
        Combine counting and checking in optimized way.
        
        Time: O(M*N)
        Space: O(M+N)
        """
        m, n = len(grid), len(grid[0])
        
        # Quick check: if total servers <= 1, return 0
        total_servers = sum(sum(row) for row in grid)
        if total_servers <= 1:
            return 0
        
        # Count servers per row and column
        row_servers = [sum(grid[i]) for i in range(m)]
        col_servers = [sum(grid[i][j] for i in range(m)) for j in range(n)]
        
        # Count communicating servers
        communicating = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    if row_servers[i] > 1 or col_servers[j] > 1:
                        communicating += 1
        
        return communicating

def test_count_servers():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,0],[0,1]], 0),
        ([[1,0],[1,1]], 3),
        ([[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]], 4),
        ([[1]], 0),  # Single server
        ([[0]], 0),  # No servers
        ([[1,1],[1,1]], 4),  # All servers communicate
        ([[1,0,0],[0,1,0],[0,0,1]], 0),  # Diagonal servers
        ([[1,1,1],[0,0,0],[1,1,1]], 6),  # Two communicating rows
    ]
    
    approaches = [
        ("Row/Col Count", solution.countServers_approach1_row_col_count),
        ("Two Pass", solution.countServers_approach2_two_pass),
        ("Union-Find", solution.countServers_approach3_union_find),
        ("Graph Components", solution.countServers_approach4_graph_components),
        ("Optimized Single Pass", solution.countServers_approach5_optimized_single_pass),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            result = func(grid)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Grid: {grid}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_server_communication():
    """Demonstrate server communication analysis"""
    print("\n=== Server Communication Analysis ===")
    
    grid = [[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
    
    m, n = len(grid), len(grid[0])
    print(f"Server grid ({m}x{n}):")
    for i, row in enumerate(grid):
        print(f"  Row {i}: {row}")
    
    # Find all servers
    servers = []
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                servers.append((i, j))
    
    print(f"\nServer positions: {servers}")
    
    # Analyze row and column counts
    row_count = [sum(grid[i]) for i in range(m)]
    col_count = [sum(grid[i][j] for i in range(m)) for j in range(n)]
    
    print(f"\nRow server counts: {row_count}")
    print(f"Column server counts: {col_count}")
    
    # Check each server
    print(f"\nServer communication analysis:")
    communicating_servers = []
    
    for i, (r, c) in enumerate(servers):
        can_communicate = row_count[r] > 1 or col_count[c] > 1
        status = "✓" if can_communicate else "✗"
        print(f"  Server {i+1} at ({r},{c}): {status}")
        print(f"    Row {r} has {row_count[r]} servers")
        print(f"    Column {c} has {col_count[c]} servers")
        
        if can_communicate:
            communicating_servers.append((r, c))
    
    print(f"\nCommunicating servers: {len(communicating_servers)}")
    print(f"Positions: {communicating_servers}")

def analyze_communication_patterns():
    """Analyze different communication patterns"""
    print("\n=== Communication Pattern Analysis ===")
    
    patterns = [
        ("Isolated servers", [[1,0,0],[0,1,0],[0,0,1]]),
        ("Row communication", [[1,1,0],[0,0,0],[0,0,1]]),
        ("Column communication", [[1,0,0],[1,0,0],[0,0,1]]),
        ("Mixed communication", [[1,1,0],[1,0,1],[0,0,0]]),
        ("Full grid", [[1,1,1],[1,1,1],[1,1,1]]),
    ]
    
    solution = Solution()
    
    for name, grid in patterns:
        result = solution.countServers_approach1_row_col_count(grid)
        total_servers = sum(sum(row) for row in grid)
        
        # Calculate communication efficiency
        efficiency = (result / total_servers * 100) if total_servers > 0 else 0
        
        print(f"{name}:")
        print(f"  Grid: {grid}")
        print(f"  Total servers: {total_servers}")
        print(f"  Communicating servers: {result}")
        print(f"  Communication efficiency: {efficiency:.1f}%")
        print()

def visualize_server_grid():
    """Create a visual representation of server communication"""
    print("\n=== Server Grid Visualization ===")
    
    grid = [[1,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]
    m, n = len(grid), len(grid[0])
    
    # Calculate which servers can communicate
    row_count = [sum(grid[i]) for i in range(m)]
    col_count = [sum(grid[i][j] for i in range(m)) for j in range(n)]
    
    print("Legend: 🟢 = Communicating server, 🔴 = Isolated server, ⬜ = No server")
    print()
    
    for i in range(m):
        row_display = []
        for j in range(n):
            if grid[i][j] == 1:
                if row_count[i] > 1 or col_count[j] > 1:
                    row_display.append("🟢")
                else:
                    row_display.append("🔴")
            else:
                row_display.append("⬜")
        print(f"Row {i}: {' '.join(row_display)}")
    
    print(f"\nRow communication potential:")
    for i in range(m):
        status = "Can communicate" if row_count[i] > 1 else "Isolated"
        print(f"  Row {i}: {row_count[i]} servers - {status}")
    
    print(f"\nColumn communication potential:")
    for j in range(n):
        status = "Can communicate" if col_count[j] > 1 else "Isolated"
        print(f"  Column {j}: {col_count[j]} servers - {status}")

if __name__ == "__main__":
    test_count_servers()
    demonstrate_server_communication()
    analyze_communication_patterns()
    visualize_server_grid()

"""
Graph Theory Concepts:
1. Grid-based Graph Representation
2. Connected Components in Implicit Graphs
3. Row/Column Connectivity Analysis
4. Communication Network Modeling

Key Insights:
- Servers communicate if they share the same row OR column
- A server is isolated if it's the only server in both its row AND column
- Problem can be solved by counting servers per row/column
- Alternative: model as graph connectivity problem

Algorithm Comparison:
┌─────────────────┬─────────────┬─────────────┬─────────────────────┐
│ Approach        │ Time        │ Space       │ Characteristics     │
├─────────────────┼─────────────┼─────────────┼─────────────────────┤
│ Row/Col Count   │ O(M*N)      │ O(M+N)      │ Simple, efficient   │
│ Two Pass        │ O(M*N)      │ O(K)        │ Server-focused      │
│ Union-Find      │ O(M*N*α(K)) │ O(K)        │ General connectivity│
│ Graph DFS       │ O(M*N+K²)   │ O(K²)       │ Explicit graph      │
└─────────────────┴─────────────┴─────────────┴─────────────────────┘

Real-world Applications:
- Data center communication analysis
- Network topology optimization
- Wireless communication coverage
- Grid-based sensor networks
- Distributed system connectivity
"""
