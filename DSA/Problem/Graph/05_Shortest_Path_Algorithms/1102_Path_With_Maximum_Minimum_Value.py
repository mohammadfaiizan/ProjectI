"""
1102. Path With Maximum Minimum Value
Difficulty: Medium

Problem:
Given an m x n integer matrix grid, return the maximum score of a path from the top-left 
cell to the bottom-right cell.

The score of a path is the minimum value in that path.

For example, the score of the path 8 → 4 → 5 → 9 is 4.

A path moves some number of times from one visited cell to any neighbouring cell in one 
of the four directions (north, east, south, west).

Examples:
Input: grid = [[5,4,5],[1,2,6],[7,7,8]]
Output: 4

Input: grid = [[2,2,1,2,2,2],[1,2,2,2,1,2]]
Output: 2

Input: grid = [[3,4,6,3,4],[0,2,1,1,7],[8,8,3,2,7],[3,2,4,9,8],[4,1,2,0,0],[4,6,5,4,3]]
Output: 3

Constraints:
- m == grid.length
- n == grid[i].length
- 1 <= m, n <= 100
- 0 <= grid[i][j] <= 10^9
"""

from typing import List
import heapq
from collections import deque

class Solution:
    def maximumMinimumPath_approach1_max_heap_dijkstra(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Max Heap Dijkstra (Optimal)
        
        Use Dijkstra with max heap to find path with maximum minimum value.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        m, n = len(grid), len(grid[0])
        
        # Max heap: use negative values for max heap behavior
        max_heap = [(-grid[0][0], 0, 0)]  # (-min_val, row, col)
        visited = [[False] * n for _ in range(m)]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while max_heap:
            neg_min_val, row, col = heapq.heappop(max_heap)
            min_val = -neg_min_val
            
            if visited[row][col]:
                continue
            
            visited[row][col] = True
            
            # Reached destination
            if row == m - 1 and col == n - 1:
                return min_val
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    not visited[new_row][new_col]):
                    
                    # New minimum value is min of current min and next cell
                    new_min_val = min(min_val, grid[new_row][new_col])
                    heapq.heappush(max_heap, (-new_min_val, new_row, new_col))
        
        return -1  # Should not reach here
    
    def maximumMinimumPath_approach2_binary_search_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Binary Search + BFS
        
        Binary search on answer, use BFS to check if path exists with min value >= target.
        
        Time: O(M*N*log(max_value))
        Space: O(M*N)
        """
        m, n = len(grid), len(grid[0])
        
        def can_reach_with_min_value(min_threshold):
            """Check if we can reach destination with minimum value >= min_threshold"""
            if grid[0][0] < min_threshold or grid[m-1][n-1] < min_threshold:
                return False
            
            visited = [[False] * n for _ in range(m)]
            queue = deque([(0, 0)])
            visited[0][0] = True
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            while queue:
                row, col = queue.popleft()
                
                if row == m - 1 and col == n - 1:
                    return True
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < m and 0 <= new_col < n and 
                        not visited[new_row][new_col] and 
                        grid[new_row][new_col] >= min_threshold):
                        
                        visited[new_row][new_col] = True
                        queue.append((new_row, new_col))
            
            return False
        
        # Binary search on the answer
        left, right = 0, min(grid[0][0], grid[m-1][n-1])
        
        # Find all possible values for more precise binary search
        all_values = set()
        for row in grid:
            for val in row:
                all_values.add(val)
        
        all_values = sorted(list(all_values), reverse=True)
        
        # Binary search on sorted values
        left, right = 0, len(all_values) - 1
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            if can_reach_with_min_value(all_values[mid]):
                result = all_values[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def maximumMinimumPath_approach3_union_find_kruskal_style(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Union-Find with Kruskal-style Processing
        
        Sort cells by value and process in descending order using Union-Find.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        m, n = len(grid), len(grid[0])
        
        # Union-Find
        parent = list(range(m * n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        def get_id(r, c):
            return r * n + c
        
        # Create list of cells sorted by value (descending)
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((grid[i][j], i, j))
        
        cells.sort(reverse=True)
        
        # Process cells in descending order of values
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        processed = [[False] * n for _ in range(m)]
        
        start_id = get_id(0, 0)
        end_id = get_id(m - 1, n - 1)
        
        for value, row, col in cells:
            processed[row][col] = True
            cell_id = get_id(row, col)
            
            # Union with processed neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    processed[new_row][new_col]):
                    
                    neighbor_id = get_id(new_row, new_col)
                    union(cell_id, neighbor_id)
            
            # Check if start and end are connected
            if find(start_id) == find(end_id):
                return value
        
        return grid[0][0]  # Fallback
    
    def maximumMinimumPath_approach4_dfs_with_memoization(self, grid: List[List[int]]) -> int:
        """
        Approach 4: DFS with Memoization
        
        Use DFS to explore all paths with memoization for efficiency.
        
        Time: O(M*N*V) where V is number of unique values
        Space: O(M*N*V)
        """
        m, n = len(grid), len(grid[0])
        
        # Memoization: memo[row][col][min_so_far] = can_reach_destination
        memo = {}
        
        def dfs(row, col, min_so_far, visited):
            """DFS to find if we can reach destination with current minimum"""
            if row == m - 1 and col == n - 1:
                return min_so_far
            
            if (row, col, min_so_far) in memo:
                return memo[(row, col, min_so_far)]
            
            max_min_path = 0
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < m and 0 <= new_col < n and 
                    (new_row, new_col) not in visited):
                    
                    new_min = min(min_so_far, grid[new_row][new_col])
                    visited.add((new_row, new_col))
                    
                    path_min = dfs(new_row, new_col, new_min, visited)
                    max_min_path = max(max_min_path, path_min)
                    
                    visited.remove((new_row, new_col))
            
            memo[(row, col, min_so_far)] = max_min_path
            return max_min_path
        
        visited = {(0, 0)}
        return dfs(0, 0, grid[0][0], visited)
    
    def maximumMinimumPath_approach5_modified_a_star(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Modified A* Algorithm
        
        Use A* with heuristic to guide search toward maximum minimum path.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        m, n = len(grid), len(grid[0])
        
        def heuristic(row, col):
            """Heuristic: minimum value on direct path to destination (optimistic)"""
            # Simple heuristic: minimum of current cell and destination
            return min(grid[row][col], grid[m-1][n-1])
        
        # Priority queue: (-min_val - heuristic, -min_val, row, col)
        pq = [(-grid[0][0] - heuristic(0, 0), -grid[0][0], 0, 0)]
        visited = {}  # (row, col) -> best_min_value_seen
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            _, neg_min_val, row, col = heapq.heappop(pq)
            min_val = -neg_min_val
            
            # Skip if we've seen this cell with better min value
            if (row, col) in visited and visited[(row, col)] >= min_val:
                continue
            
            visited[(row, col)] = min_val
            
            if row == m - 1 and col == n - 1:
                return min_val
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < m and 0 <= new_col < n:
                    new_min_val = min(min_val, grid[new_row][new_col])
                    
                    # Only proceed if this is better than what we've seen
                    if ((new_row, new_col) not in visited or 
                        visited[(new_row, new_col)] < new_min_val):
                        
                        h = heuristic(new_row, new_col)
                        priority = -(new_min_val + h)
                        heapq.heappush(pq, (priority, -new_min_val, new_row, new_col))
        
        return -1

def test_maximum_minimum_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[5,4,5],[1,2,6],[7,7,8]], 4),
        ([[2,2,1,2,2,2],[1,2,2,2,1,2]], 2),
        ([[3,4,6,3,4],[0,2,1,1,7],[8,8,3,2,7],[3,2,4,9,8],[4,1,2,0,0],[4,6,5,4,3]], 3),
        ([[1,2,3],[4,5,6],[7,8,9]], 5),
        ([[10]], 10),
    ]
    
    approaches = [
        ("Max Heap Dijkstra", solution.maximumMinimumPath_approach1_max_heap_dijkstra),
        ("Binary Search + BFS", solution.maximumMinimumPath_approach2_binary_search_bfs),
        ("Union-Find Kruskal", solution.maximumMinimumPath_approach3_union_find_kruskal_style),
        ("DFS Memoization", solution.maximumMinimumPath_approach4_dfs_with_memoization),
        ("Modified A*", solution.maximumMinimumPath_approach5_modified_a_star),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            result = func([row[:] for row in grid])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_max_min_path_concept():
    """Demonstrate the max-min path concept"""
    print("\n=== Max-Min Path Concept Demo ===")
    
    grid = [[5,4,5],[1,2,6],[7,7,8]]
    
    print(f"Grid:")
    for i, row in enumerate(grid):
        print(f"  {i}: {row}")
    
    print(f"\nGoal: Find path from (0,0) to (2,2) with maximum minimum value")
    print(f"Path score = minimum value along the path")
    
    # Analyze possible paths manually
    paths = [
        {
            "route": [(0,0), (0,1), (0,2), (1,2), (2,2)],
            "values": [5, 4, 5, 6, 8],
            "min_value": 4
        },
        {
            "route": [(0,0), (1,0), (1,1), (1,2), (2,2)],
            "values": [5, 1, 2, 6, 8],
            "min_value": 1
        },
        {
            "route": [(0,0), (1,0), (2,0), (2,1), (2,2)],
            "values": [5, 1, 7, 7, 8],
            "min_value": 1
        },
        {
            "route": [(0,0), (0,1), (1,1), (2,1), (2,2)],
            "values": [5, 4, 2, 7, 8],
            "min_value": 2
        }
    ]
    
    print(f"\nAnalyzing possible paths:")
    for i, path in enumerate(paths, 1):
        route_str = " → ".join([f"({r},{c})" for r, c in path["route"]])
        values_str = " → ".join(map(str, path["values"]))
        print(f"Path {i}: {route_str}")
        print(f"  Values: {values_str}")
        print(f"  Minimum: {path['min_value']}")
    
    optimal_paths = [p for p in paths if p["min_value"] == max(p["min_value"] for p in paths)]
    print(f"\nOptimal path(s) with maximum minimum value:")
    for path in optimal_paths:
        route_str = " → ".join([f"({r},{c})" for r, c in path["route"]])
        print(f"  {route_str} with min value {path['min_value']}")

def demonstrate_max_heap_dijkstra():
    """Demonstrate max heap Dijkstra algorithm"""
    print("\n=== Max Heap Dijkstra Demo ===")
    
    grid = [[5,4,5],[1,2,6]]
    m, n = len(grid), len(grid[0])
    
    print(f"Grid: {grid}")
    print(f"Finding maximum minimum path from (0,0) to ({m-1},{n-1})")
    
    # Simulate max heap Dijkstra
    import heapq
    
    max_heap = [(-grid[0][0], 0, 0)]
    visited = [[False] * n for _ in range(m)]
    
    print(f"\nMax Heap Dijkstra simulation:")
    print(f"Initial heap: {max_heap} (negative values for max heap)")
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    step = 0
    
    while max_heap:
        step += 1
        neg_min_val, row, col = heapq.heappop(max_heap)
        min_val = -neg_min_val
        
        print(f"\nStep {step}: Processing ({row},{col}) with min_value={min_val}")
        
        if visited[row][col]:
            print(f"  Already visited, skipping")
            continue
        
        visited[row][col] = True
        
        if row == m - 1 and col == n - 1:
            print(f"  Reached destination! Maximum minimum value: {min_val}")
            break
        
        print(f"  Exploring neighbors:")
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < m and 0 <= new_col < n and 
                not visited[new_row][new_col]):
                
                new_min_val = min(min_val, grid[new_row][new_col])
                heapq.heappush(max_heap, (-new_min_val, new_row, new_col))
                print(f"    ({new_row},{new_col}): cell_value={grid[new_row][new_col]}, new_min={new_min_val}")
        
        print(f"  Updated heap: {max_heap}")

def analyze_algorithmic_approaches():
    """Analyze different algorithmic approaches for max-min path"""
    print("\n=== Algorithmic Approaches Analysis ===")
    
    print("Problem Type: Bottleneck Path with Maximum Objective")
    print("• Maximize the minimum value along path")
    print("• Different from standard shortest path (sum minimization)")
    print("• Related to maximum capacity path problems")
    
    print("\nAlgorithm Comparison:")
    
    print("\n1. **Max Heap Dijkstra (Recommended):**")
    print("   • Time: O(M*N*log(M*N))")
    print("   • Space: O(M*N)")
    print("   • Pros: Optimal, natural fit for max-min objective")
    print("   • Cons: Requires priority queue with max heap")
    
    print("\n2. **Binary Search + BFS:**")
    print("   • Time: O(M*N*log(max_value))")
    print("   • Space: O(M*N)")
    print("   • Pros: Simple BFS for each threshold")
    print("   • Cons: Multiple graph traversals")
    
    print("\n3. **Union-Find + Kruskal Style:**")
    print("   • Time: O(M*N*log(M*N))")
    print("   • Space: O(M*N)")
    print("   • Pros: Elegant, similar to MST algorithms")
    print("   • Cons: More complex implementation")
    
    print("\n4. **DFS with Memoization:**")
    print("   • Time: O(M*N*V) where V = unique values")
    print("   • Space: O(M*N*V)")
    print("   • Pros: Explores all paths systematically")
    print("   • Cons: Exponential without good pruning")
    
    print("\n5. **Modified A*:**")
    print("   • Time: O(M*N*log(M*N))")
    print("   • Space: O(M*N)")
    print("   • Pros: Heuristic guidance toward target")
    print("   • Cons: Heuristic design challenging for this problem")

def compare_with_related_problems():
    """Compare with related pathfinding problems"""
    print("\n=== Related Problems Comparison ===")
    
    print("Path Optimization Objectives:")
    
    print("\n1. **Shortest Path (Sum Minimization):**")
    print("   • Minimize sum of edge weights")
    print("   • Algorithm: Dijkstra, Bellman-Ford")
    print("   • Applications: GPS navigation, network routing")
    
    print("\n2. **Maximum Minimum Path (This Problem):**")
    print("   • Maximize minimum value along path")
    print("   • Algorithm: Modified Dijkstra, Binary Search")
    print("   • Applications: Network capacity, robustness analysis")
    
    print("\n3. **Minimum Maximum Path:**")
    print("   • Minimize maximum value along path")
    print("   • Algorithm: Binary Search, Modified Dijkstra")
    print("   • Applications: Minimize worst-case scenarios")
    
    print("\n4. **Maximum Sum Path:**")
    print("   • Maximize sum of values along path")
    print("   • Algorithm: Modified Dijkstra with max heap")
    print("   • Applications: Profit maximization, reward collection")
    
    print("\n5. **Constrained Shortest Path:**")
    print("   • Minimize cost subject to constraints")
    print("   • Algorithm: State space extension")
    print("   • Applications: Resource-limited planning")
    
    print("\nKey Insights:")
    print("• **Objective function determines algorithm choice**")
    print("• **Bottleneck problems use modified priority ordering**")
    print("• **Binary search applicable when monotonic property exists**")
    print("• **Graph structure affects algorithm efficiency**")

def analyze_real_world_applications():
    """Analyze real-world applications of maximum minimum path"""
    print("\n=== Real-World Applications ===")
    
    print("1. **Network Capacity Planning:**")
    print("   • Find path with maximum minimum bandwidth")
    print("   • Ensure sufficient capacity for data transmission")
    print("   • Critical for quality of service guarantees")
    
    print("\n2. **Load Balancing:**")
    print("   • Distribute traffic to maximize minimum server capacity")
    print("   • Prevent bottlenecks in system components")
    print("   • Improve overall system robustness")
    
    print("\n3. **Supply Chain Resilience:**")
    print("   • Find supply routes with maximum minimum reliability")
    print("   • Ensure supply chain stability")
    print("   • Minimize impact of weakest link")
    
    print("\n4. **Infrastructure Planning:**")
    print("   • Design roads with maximum minimum capacity")
    print("   • Ensure emergency evacuation routes")
    print("   • Optimize for peak usage scenarios")
    
    print("\n5. **Financial Risk Management:**")
    print("   • Investment paths with maximum minimum return")
    print("   • Portfolio optimization for worst-case scenarios")
    print("   • Risk-adjusted decision making")
    
    print("\n6. **Game Design:**")
    print("   • Player progression paths with balanced difficulty")
    print("   • Ensure minimum skill requirements are manageable")
    print("   • Optimize player experience curves")
    
    print("\nDesign Principles:")
    print("• **Robustness:** Optimize for weakest component")
    print("• **Reliability:** Ensure minimum performance standards")
    print("• **Bottleneck analysis:** Identify and optimize constraints")
    print("• **Risk management:** Plan for worst-case scenarios")
    print("• **Quality assurance:** Maintain minimum service levels")

if __name__ == "__main__":
    test_maximum_minimum_path()
    demonstrate_max_min_path_concept()
    demonstrate_max_heap_dijkstra()
    analyze_algorithmic_approaches()
    compare_with_related_problems()
    analyze_real_world_applications()

"""
Shortest Path Concepts:
1. Maximum Minimum Path (Max-Min Optimization)
2. Modified Dijkstra for Alternative Objectives
3. Binary Search on Answer with Feasibility Check
4. Union-Find for Connectivity Analysis
5. Bottleneck Path Problems and Applications

Key Problem Insights:
- Maximize minimum value along path (not sum)
- Bottleneck optimization problem
- Path quality determined by weakest link
- Modified Dijkstra with max heap behavior

Algorithm Strategy:
1. Use Dijkstra with modified priority function
2. Priority: maximum minimum value seen so far
3. Update rule: min(current_min, next_cell_value)
4. Max heap to prioritize higher minimum values

Modified Dijkstra Adaptations:
- Use max heap (negative values in min heap)
- Track minimum value along path to each cell
- Update condition: new path has higher minimum value
- Early termination when destination reached

Alternative Approaches:
- Binary search on possible minimum values
- Union-Find with descending value processing
- DFS with memoization for all paths
- A* with appropriate heuristic function

Bottleneck vs Standard Optimization:
- Standard: minimize/maximize sum of values
- Bottleneck: optimize minimum/maximum value
- Different algorithms and data structures needed
- Applications in capacity and reliability analysis

Real-world Applications:
- Network capacity and bandwidth optimization
- Supply chain resilience planning
- Infrastructure robustness analysis
- Load balancing and resource allocation
- Financial risk management

This problem demonstrates bottleneck optimization
in pathfinding with practical engineering applications.
"""
