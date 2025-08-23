"""
1631. Path With Minimum Effort
Difficulty: Medium

Problem:
You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of 
size rows x columns, where heights[i][j] represents the height of cell (i, j).

You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right 
cell, (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, and 
you wish to find a route that requires the minimum effort.

A route's effort is the maximum absolute difference in heights between two consecutive 
cells of the route.

Return the minimum effort required to travel from the top-left cell to the bottom-right cell.

Examples:
Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
Output: 2

Input: heights = [[1,2,3],[3,8,4],[5,3,5]]
Output: 1

Input: heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
Output: 0

Constraints:
- rows == heights.length
- columns == heights[i].length
- 1 <= rows, columns <= 100
- 1 <= heights[i][j] <= 10^6
"""

from typing import List
import heapq
from collections import deque

class Solution:
    def minimumEffortPath_approach1_dijkstra_modified(self, heights: List[List[int]]) -> int:
        """
        Approach 1: Modified Dijkstra's Algorithm (Optimal)
        
        Use Dijkstra to find path with minimum maximum edge weight.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        
        # Dijkstra with effort tracking
        efforts = [[float('inf')] * cols for _ in range(rows)]
        efforts[0][0] = 0
        
        # Priority queue: (effort, row, col)
        pq = [(0, 0, 0)]
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while pq:
            effort, row, col = heapq.heappop(pq)
            
            # Reached destination
            if row == rows - 1 and col == cols - 1:
                return effort
            
            # Skip if we've found better path
            if effort > efforts[row][col]:
                continue
            
            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    # Calculate effort for this edge
                    edge_effort = abs(heights[new_row][new_col] - heights[row][col])
                    
                    # New path effort is max of current effort and this edge
                    new_effort = max(effort, edge_effort)
                    
                    if new_effort < efforts[new_row][new_col]:
                        efforts[new_row][new_col] = new_effort
                        heapq.heappush(pq, (new_effort, new_row, new_col))
        
        return efforts[rows - 1][cols - 1]
    
    def minimumEffortPath_approach2_binary_search_bfs(self, heights: List[List[int]]) -> int:
        """
        Approach 2: Binary Search + BFS
        
        Binary search on effort value, use BFS to check if path exists.
        
        Time: O(M*N*log(max_height))
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        
        def can_reach_with_effort(max_effort):
            """Check if we can reach destination with given max effort"""
            visited = [[False] * cols for _ in range(rows)]
            queue = deque([(0, 0)])
            visited[0][0] = True
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            while queue:
                row, col = queue.popleft()
                
                if row == rows - 1 and col == cols - 1:
                    return True
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < rows and 0 <= new_col < cols and 
                        not visited[new_row][new_col]):
                        
                        effort = abs(heights[new_row][new_col] - heights[row][col])
                        
                        if effort <= max_effort:
                            visited[new_row][new_col] = True
                            queue.append((new_row, new_col))
            
            return False
        
        # Binary search on effort
        left, right = 0, max(max(row) for row in heights)
        
        while left < right:
            mid = (left + right) // 2
            
            if can_reach_with_effort(mid):
                right = mid
            else:
                left = mid + 1
        
        return left
    
    def minimumEffortPath_approach3_union_find_kruskal(self, heights: List[List[int]]) -> int:
        """
        Approach 3: Union-Find with Kruskal-style Edge Processing
        
        Sort all edges by effort and use Union-Find to find minimum effort.
        
        Time: O(M*N*log(M*N))
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        
        if rows == 1 and cols == 1:
            return 0
        
        # Union-Find
        parent = list(range(rows * cols))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        def get_id(r, c):
            return r * cols + c
        
        # Collect all edges with their efforts
        edges = []
        directions = [(0, 1), (1, 0)]  # Only right and down to avoid duplicates
        
        for r in range(rows):
            for c in range(cols):
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if nr < rows and nc < cols:
                        effort = abs(heights[nr][nc] - heights[r][c])
                        edges.append((effort, get_id(r, c), get_id(nr, nc)))
        
        # Sort edges by effort
        edges.sort()
        
        start_id = get_id(0, 0)
        end_id = get_id(rows - 1, cols - 1)
        
        # Process edges in order
        for effort, u, v in edges:
            union(u, v)
            
            # Check if start and end are connected
            if find(start_id) == find(end_id):
                return effort
        
        return 0
    
    def minimumEffortPath_approach4_dfs_backtracking(self, heights: List[List[int]]) -> int:
        """
        Approach 4: DFS with Backtracking and Pruning
        
        Use DFS to explore all paths with pruning for efficiency.
        
        Time: O(4^(M*N)) worst case, much better with pruning
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        
        self.min_effort = float('inf')
        
        def dfs(row, col, max_effort, visited):
            """DFS with current maximum effort tracking"""
            if row == rows - 1 and col == cols - 1:
                self.min_effort = min(self.min_effort, max_effort)
                return
            
            # Pruning: if current effort already exceeds best found
            if max_effort >= self.min_effort:
                return
            
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < rows and 0 <= new_col < cols and 
                    (new_row, new_col) not in visited):
                    
                    effort = abs(heights[new_row][new_col] - heights[row][col])
                    new_max_effort = max(max_effort, effort)
                    
                    # More pruning
                    if new_max_effort < self.min_effort:
                        visited.add((new_row, new_col))
                        dfs(new_row, new_col, new_max_effort, visited)
                        visited.remove((new_row, new_col))
        
        visited = {(0, 0)}
        dfs(0, 0, 0, visited)
        
        return self.min_effort
    
    def minimumEffortPath_approach5_binary_search_dijkstra(self, heights: List[List[int]]) -> int:
        """
        Approach 5: Binary Search + Dijkstra
        
        Binary search on effort, use Dijkstra to verify reachability.
        
        Time: O(M*N*log(M*N)*log(max_height))
        Space: O(M*N)
        """
        rows, cols = len(heights), len(heights[0])
        
        def can_reach_with_dijkstra(max_effort):
            """Use Dijkstra to check reachability with effort limit"""
            efforts = [[float('inf')] * cols for _ in range(rows)]
            efforts[0][0] = 0
            
            pq = [(0, 0, 0)]  # (effort, row, col)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            while pq:
                effort, row, col = heapq.heappop(pq)
                
                if row == rows - 1 and col == cols - 1:
                    return True
                
                if effort > efforts[row][col]:
                    continue
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if 0 <= new_row < rows and 0 <= new_col < cols:
                        edge_effort = abs(heights[new_row][new_col] - heights[row][col])
                        
                        if edge_effort <= max_effort:
                            new_effort = max(effort, edge_effort)
                            
                            if new_effort < efforts[new_row][new_col]:
                                efforts[new_row][new_col] = new_effort
                                heapq.heappush(pq, (new_effort, new_row, new_col))
            
            return False
        
        # Binary search
        left, right = 0, max(max(row) for row in heights)
        
        while left < right:
            mid = (left + right) // 2
            
            if can_reach_with_dijkstra(mid):
                right = mid
            else:
                left = mid + 1
        
        return left

def test_minimum_effort_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (heights, expected)
        ([[1,2,2],[3,8,2],[5,3,5]], 2),
        ([[1,2,3],[3,8,4],[5,3,5]], 1),
        ([[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]], 0),
        ([[1,10,6,7,9,10,4,9]], 9),
        ([[4,3,4,10,5,5,9,2],[10,8,2,10,9,7,5,6],[5,8,10,10,10,7,4,2]], 5),
    ]
    
    approaches = [
        ("Modified Dijkstra", solution.minimumEffortPath_approach1_dijkstra_modified),
        ("Binary Search + BFS", solution.minimumEffortPath_approach2_binary_search_bfs),
        ("Union-Find Kruskal", solution.minimumEffortPath_approach3_union_find_kruskal),
        ("DFS Backtracking", solution.minimumEffortPath_approach4_dfs_backtracking),
        ("Binary Search + Dijkstra", solution.minimumEffortPath_approach5_binary_search_dijkstra),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (heights, expected) in enumerate(test_cases):
            result = func([row[:] for row in heights])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_minimum_effort_concept():
    """Demonstrate the minimum effort path concept"""
    print("\n=== Minimum Effort Path Demo ===")
    
    heights = [[1,2,2],[3,8,2],[5,3,5]]
    print(f"Height matrix:")
    for i, row in enumerate(heights):
        print(f"  {i}: {row}")
    
    print(f"\nGoal: Find path from (0,0) to (2,2) with minimum maximum effort")
    print(f"Effort = maximum absolute height difference along the path")
    
    # Manual path analysis
    print(f"\nPossible paths analysis:")
    
    paths = [
        {
            "name": "Path 1: Right → Right → Down → Down",
            "route": [(0,0), (0,1), (0,2), (1,2), (2,2)],
            "efforts": []
        },
        {
            "name": "Path 2: Down → Down → Right → Right", 
            "route": [(0,0), (1,0), (2,0), (2,1), (2,2)],
            "efforts": []
        },
        {
            "name": "Path 3: Right → Down → Right → Down",
            "route": [(0,0), (0,1), (1,1), (1,2), (2,2)],
            "efforts": []
        }
    ]
    
    for path in paths:
        route = path["route"]
        efforts = []
        
        for i in range(len(route) - 1):
            r1, c1 = route[i]
            r2, c2 = route[i + 1]
            effort = abs(heights[r2][c2] - heights[r1][c1])
            efforts.append(effort)
        
        path["efforts"] = efforts
        max_effort = max(efforts)
        
        print(f"\n{path['name']}:")
        print(f"  Route: {' → '.join([f'({r},{c})' for r, c in route])}")
        print(f"  Heights: {' → '.join([str(heights[r][c]) for r, c in route])}")
        print(f"  Efforts: {efforts}")
        print(f"  Maximum effort: {max_effort}")
    
    min_max_effort = min(max(path["efforts"]) for path in paths)
    print(f"\nMinimum maximum effort across all paths: {min_max_effort}")

def demonstrate_modified_dijkstra():
    """Demonstrate modified Dijkstra algorithm for min-max path"""
    print("\n=== Modified Dijkstra Demo ===")
    
    heights = [[1,2,2],[3,8,2]]
    rows, cols = len(heights), len(heights[0])
    
    print(f"Height matrix: {heights}")
    print(f"Find minimum effort path from (0,0) to ({rows-1},{cols-1})")
    
    # Modified Dijkstra step by step
    efforts = [[float('inf')] * cols for _ in range(rows)]
    efforts[0][0] = 0
    
    pq = [(0, 0, 0)]  # (effort, row, col)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    print(f"\nInitial state:")
    print(f"  efforts: {[['∞' if x == float('inf') else x for x in row] for row in efforts]}")
    print(f"  priority_queue: {pq}")
    
    step = 0
    while pq:
        step += 1
        effort, row, col = heapq.heappop(pq)
        
        print(f"\nStep {step}: Processing ({row},{col}) with effort {effort}")
        
        if row == rows - 1 and col == cols - 1:
            print(f"  Reached destination! Final effort: {effort}")
            break
        
        if effort > efforts[row][col]:
            print(f"  Skipping (already have better path)")
            continue
        
        # Explore neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                edge_effort = abs(heights[new_row][new_col] - heights[row][col])
                new_effort = max(effort, edge_effort)
                
                print(f"  Neighbor ({new_row},{new_col}): edge_effort={edge_effort}, new_effort={new_effort}")
                
                if new_effort < efforts[new_row][new_col]:
                    efforts[new_row][new_col] = new_effort
                    heapq.heappush(pq, (new_effort, new_row, new_col))
                    print(f"    Updated effort[{new_row}][{new_col}] = {new_effort}")
        
        print(f"  Current efforts: {[['∞' if x == float('inf') else x for x in row] for row in efforts]}")
        print(f"  Priority queue: {sorted(pq)}")

def analyze_min_max_path_algorithms():
    """Analyze different approaches for min-max path problems"""
    print("\n=== Min-Max Path Algorithms Analysis ===")
    
    print("Problem Type: Bottleneck Shortest Path")
    print("• Objective: Minimize the maximum edge weight along path")
    print("• Different from standard shortest path (minimize sum)")
    print("• Also known as: widest path, minimax path")
    
    print("\nAlgorithm Comparison:")
    
    print("\n1. **Modified Dijkstra (Recommended):**")
    print("   • Time: O(M*N*log(M*N))")
    print("   • Space: O(M*N)")
    print("   • Pros: Optimal, handles min-max objective naturally")
    print("   • Cons: Requires priority queue modification")
    
    print("\n2. **Binary Search + BFS:**")
    print("   • Time: O(M*N*log(H)) where H = max height")
    print("   • Space: O(M*N)")
    print("   • Pros: Simple BFS for each threshold")
    print("   • Cons: May do more work than necessary")
    
    print("\n3. **Union-Find + Kruskal Style:**")
    print("   • Time: O(M*N*log(M*N))")
    print("   • Space: O(M*N)")
    print("   • Pros: Elegant MST-based approach")
    print("   • Cons: More complex implementation")
    
    print("\n4. **DFS with Backtracking:**")
    print("   • Time: O(4^(M*N)) worst case")
    print("   • Space: O(M*N)")
    print("   • Pros: Explores all possible paths")
    print("   • Cons: Exponential time without good pruning")
    
    print("\n5. **Binary Search + Dijkstra:**")
    print("   • Time: O(M*N*log(M*N)*log(H))")
    print("   • Space: O(M*N)")
    print("   • Pros: Combines two well-known algorithms")
    print("   • Cons: Overkill - redundant work")
    
    print("\nOptimal Choice:")
    print("• **Modified Dijkstra** for best performance")
    print("• **Binary Search + BFS** for simplicity")
    print("• **Union-Find** for educational/MST perspective")

def compare_shortest_path_variations():
    """Compare different shortest path problem variations"""
    print("\n=== Shortest Path Problem Variations ===")
    
    print("1. **Standard Shortest Path:**")
    print("   • Objective: Minimize sum of edge weights")
    print("   • Algorithm: Dijkstra, Bellman-Ford")
    print("   • Example: Minimum travel time")
    
    print("\n2. **Bottleneck Shortest Path (This Problem):**")
    print("   • Objective: Minimize maximum edge weight")
    print("   • Algorithm: Modified Dijkstra, Binary Search")
    print("   • Example: Minimum effort, maximum capacity")
    
    print("\n3. **Constrained Shortest Path:**")
    print("   • Objective: Minimize cost with constraints")
    print("   • Algorithm: Modified Dijkstra with state")
    print("   • Example: K-stops flight problem")
    
    print("\n4. **Maximum Probability Path:**")
    print("   • Objective: Maximize path probability")
    print("   • Algorithm: Modified Dijkstra (max-heap)")
    print("   • Example: Most reliable network path")
    
    print("\n5. **Multi-Objective Shortest Path:**")
    print("   • Objective: Optimize multiple criteria")
    print("   • Algorithm: Pareto-optimal solutions")
    print("   • Example: Time vs cost trade-offs")
    
    print("\nKey Insights:")
    print("• **Objective function determines algorithm choice**")
    print("• **State space may need extension for constraints**")
    print("• **Priority queue behavior adapts to objective**")
    print("• **Problem structure guides optimization approach**")
    
    print("\nReal-world Applications:")
    print("• **Hiking/mountaineering:** Minimize maximum elevation gain")
    print("• **Network routing:** Maximize minimum bandwidth")
    print("• **Transportation:** Minimize maximum load")
    print("• **Manufacturing:** Minimize bottleneck capacity")
    print("• **Game design:** Difficulty progression optimization")

if __name__ == "__main__":
    test_minimum_effort_path()
    demonstrate_minimum_effort_concept()
    demonstrate_modified_dijkstra()
    analyze_min_max_path_algorithms()
    compare_shortest_path_variations()

"""
Shortest Path Concepts:
1. Bottleneck Shortest Path (Min-Max Path)
2. Modified Dijkstra for Alternative Objectives
3. Binary Search on Answer with Reachability Check
4. Union-Find for MST-style Path Finding
5. Grid-based Shortest Path Problems

Key Problem Insights:
- Minimize maximum edge weight along path (not sum)
- Different objective requires algorithm modification
- Path effort = maximum absolute height difference
- Grid connectivity with 4-directional movement

Algorithm Strategy:
1. Modify Dijkstra to track maximum effort instead of total cost
2. Use max(current_effort, edge_effort) for path extension
3. Priority queue orders by minimum maximum effort
4. Early termination when destination reached

Modified Dijkstra Adaptations:
- Track maximum effort along path instead of total distance
- Update condition: new_max_effort < current_best_effort
- Priority queue holds (max_effort, position) pairs
- Path effort = max of all edge efforts in path

Alternative Approaches:
- Binary search on effort value + BFS reachability
- Union-Find with edge sorting (Kruskal-style)
- DFS backtracking with pruning
- Multiple algorithm combinations

Bottleneck vs Standard Shortest Path:
- Standard: minimize sum of edge weights
- Bottleneck: minimize maximum edge weight
- Different objectives require different algorithms
- Applications vary based on problem constraints

Real-world Applications:
- Hiking trail difficulty optimization
- Network bandwidth maximization
- Transportation load balancing
- Manufacturing bottleneck analysis
- Game difficulty progression

This problem demonstrates adaptation of shortest path
algorithms for alternative optimization objectives.
"""
