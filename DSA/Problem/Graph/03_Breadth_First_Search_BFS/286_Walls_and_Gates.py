"""
286. Walls and Gates
Difficulty: Easy

Problem:
You are given an m x n grid rooms initialized with these three possible values:
- -1: A wall or an obstacle.
- 0: A gate.
- INF: Infinity means an empty room.

Fill each empty room with the distance to its nearest gate. If it is impossible to 
reach a gate, it should be filled with INF.

Examples:
Input: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
Output: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]

Input: rooms = [[-1]]
Output: [[-1]]

Constraints:
- m == rooms.length
- n == rooms[i].length
- 1 <= m, n <= 250
- rooms[i][j] is -1, 0, or 2^31 - 1
"""

from typing import List
from collections import deque

class Solution:
    def wallsAndGates_approach1_multi_source_bfs(self, rooms: List[List[int]]) -> None:
        """
        Approach 1: Multi-Source BFS (Optimal)
        
        Start BFS from all gates simultaneously. Each level represents distance + 1.
        Update rooms with minimum distance to any gate.
        
        Time: O(M*N) - visit each cell at most once
        Space: O(M*N) - queue size in worst case
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        queue = deque()
        
        # Find all gates and add to queue
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Multi-source BFS
        while queue:
            i, j = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                # Check bounds and if cell can be updated
                if (0 <= ni < m and 0 <= nj < n and rooms[ni][nj] == INF):
                    rooms[ni][nj] = rooms[i][j] + 1
                    queue.append((ni, nj))
    
    def wallsAndGates_approach2_level_by_level_bfs(self, rooms: List[List[int]]) -> None:
        """
        Approach 2: Level-by-Level BFS with Distance Tracking
        
        Process all cells at current distance before moving to next distance.
        
        Time: O(M*N)
        Space: O(M*N)
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        queue = deque()
        
        # Initialize with all gates
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        distance = 0
        
        while queue:
            distance += 1
            size = len(queue)
            
            # Process all cells at current distance
            for _ in range(size):
                i, j = queue.popleft()
                
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    
                    if (0 <= ni < m and 0 <= nj < n and rooms[ni][nj] == INF):
                        rooms[ni][nj] = distance
                        queue.append((ni, nj))
    
    def wallsAndGates_approach3_dfs_from_each_gate(self, rooms: List[List[int]]) -> None:
        """
        Approach 3: DFS from Each Gate (Alternative Approach)
        
        For each gate, use DFS to update reachable rooms with minimum distance.
        
        Time: O(M*N*G) where G is number of gates
        Space: O(M*N) - recursion stack
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        
        def dfs(i, j, distance):
            """DFS to update distances from current gate"""
            if (i < 0 or i >= m or j < 0 or j >= n or 
                rooms[i][j] < distance):  # Wall, or already has shorter distance
                return
            
            rooms[i][j] = distance
            
            # Explore 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                dfs(i + di, j + dj, distance + 1)
        
        # Start DFS from each gate
        for i in range(m):
            for j in range(n):
                if rooms[i][j] == 0:
                    dfs(i, j, 0)
    
    def wallsAndGates_approach4_iterative_relaxation(self, rooms: List[List[int]]) -> None:
        """
        Approach 4: Iterative Distance Relaxation
        
        Repeatedly relax distances until no more improvements can be made.
        
        Time: O(M*N*D) where D is maximum distance
        Space: O(1) - in-place updates
        """
        if not rooms or not rooms[0]:
            return
        
        m, n = len(rooms), len(rooms[0])
        INF = 2147483647
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Iteratively relax distances
        changed = True
        while changed:
            changed = False
            
            for i in range(m):
                for j in range(n):
                    if rooms[i][j] != -1 and rooms[i][j] != INF:
                        # Try to relax neighbors
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            
                            if (0 <= ni < m and 0 <= nj < n and 
                                rooms[ni][nj] == INF):
                                rooms[ni][nj] = rooms[i][j] + 1
                                changed = True

def test_walls_and_gates():
    """Test all approaches with various cases"""
    solution = Solution()
    
    INF = 2147483647
    
    test_cases = [
        # (rooms_input, expected_output)
        ([[INF,-1,0,INF],[INF,INF,INF,-1],[INF,-1,INF,-1],[0,-1,INF,INF]], 
         [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]),
        ([[-1]], [[-1]]),
        ([[0]], [[0]]),
        ([[INF]], [[INF]]),
        ([[0,-1],[INF,INF]], [[0,-1],[1,2]]),
        ([[INF,0,INF],[INF,INF,INF],[INF,INF,0]], [[2,0,1],[3,2,1],[2,1,0]]),
    ]
    
    approaches = [
        ("Multi-Source BFS", solution.wallsAndGates_approach1_multi_source_bfs),
        ("Level-by-Level BFS", solution.wallsAndGates_approach2_level_by_level_bfs),
        ("DFS from Each Gate", solution.wallsAndGates_approach3_dfs_from_each_gate),
        ("Iterative Relaxation", solution.wallsAndGates_approach4_iterative_relaxation),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (rooms_input, expected) in enumerate(test_cases):
            # Create deep copy for testing
            test_rooms = [row[:] for row in rooms_input]
            func(test_rooms)
            status = "✓" if test_rooms == expected else "✗"
            print(f"Test {i+1}: {status}")
            if test_rooms != expected:
                print(f"         Input: {rooms_input}")
                print(f"         Expected: {expected}")
                print(f"         Got: {test_rooms}")

def demonstrate_distance_propagation():
    """Demonstrate distance propagation from gates"""
    print("\n=== Distance Propagation Demo ===")
    
    INF = 2147483647
    rooms = [[INF,-1,0,INF],
             [INF,INF,INF,-1],
             [INF,-1,INF,-1],
             [0,-1,INF,INF]]
    
    print("Initial state:")
    print_rooms(rooms)
    
    # Show gates
    gates = []
    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            if rooms[i][j] == 0:
                gates.append((i, j))
    
    print(f"\nGates found at: {gates}")
    
    # Simulate BFS step by step
    m, n = len(rooms), len(rooms[0])
    queue = deque(gates)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    step = 0
    
    print(f"\nBFS propagation:")
    
    while queue:
        step += 1
        print(f"\nStep {step}:")
        size = len(queue)
        updated_cells = []
        
        for _ in range(size):
            i, j = queue.popleft()
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < m and 0 <= nj < n and rooms[ni][nj] == INF):
                    rooms[ni][nj] = rooms[i][j] + 1
                    queue.append((ni, nj))
                    updated_cells.append((ni, nj, rooms[ni][nj]))
        
        if updated_cells:
            print(f"  Updated cells: {updated_cells}")
            print_rooms(rooms)
        else:
            print("  No cells updated")
            break
    
    print(f"\nFinal result:")
    print_rooms(rooms)

def print_rooms(rooms):
    """Helper function to print rooms in a readable format"""
    INF = 2147483647
    
    for row in rooms:
        formatted_row = []
        for cell in row:
            if cell == -1:
                formatted_row.append("🧱")  # Wall
            elif cell == 0:
                formatted_row.append("🚪")  # Gate
            elif cell == INF:
                formatted_row.append("∞")   # Infinity
            else:
                formatted_row.append(f"{cell:2}")  # Distance
        print(f"  {' '.join(formatted_row)}")

def analyze_multi_source_bfs():
    """Analyze why multi-source BFS is optimal"""
    print("\n=== Multi-Source BFS Analysis ===")
    
    print("Why Multi-Source BFS is optimal for Walls and Gates:")
    print("1. 🎯 Simultaneous expansion: All gates expand simultaneously")
    print("2. 📊 Optimal distances: First time a cell is reached = shortest distance")
    print("3. ⚡ Single pass: Each cell visited at most once")
    print("4. 🔄 Natural parallelism: All gates work together")
    
    print("\nComparison with alternatives:")
    print("• Single-source BFS per gate: ❌ O(M*N*G) time")
    print("• DFS from each gate: ❌ May find suboptimal paths")
    print("• Dijkstra's algorithm: ❌ Overkill for unweighted graph")
    print("• Multi-source BFS: ✅ O(M*N) optimal solution")
    
    print("\nKey insights:")
    print("• Start BFS from all gates simultaneously")
    print("• Each level represents distance + 1 from nearest gate")
    print("• First visit to cell guarantees minimum distance")
    print("• Walls (-1) are never processed")
    print("• INF cells become targets for distance updates")
    
    print("\nReal-world applications:")
    print("• Fire station coverage analysis")
    print("• Hospital accessibility planning")
    print("• Network latency optimization")
    print("• Emergency response planning")
    print("• Retail location strategy")

if __name__ == "__main__":
    test_walls_and_gates()
    demonstrate_distance_propagation()
    analyze_multi_source_bfs()

"""
Graph Theory Concepts:
1. Multi-Source Shortest Path
2. Distance Propagation from Multiple Sources
3. BFS for Minimum Distance Calculation
4. Graph Relaxation Algorithms

Key Multi-Source BFS Insights:
- Start from all sources (gates) simultaneously
- BFS guarantees shortest path in unweighted graphs
- First visit to any cell provides optimal distance
- Level-order processing ensures distance optimality

Algorithm Strategy:
- Initialize queue with all gate positions
- Use BFS to expand from all gates simultaneously  
- Update empty rooms (INF) with distance from nearest gate
- Walls (-1) block movement and are never updated

Optimization Advantages:
- Single pass through grid: O(M*N) time
- Optimal distances guaranteed by BFS properties
- Memory efficient: in-place updates
- Natural parallelism: all gates expand together

Real-world Applications:
- Facility location optimization
- Emergency service coverage
- Network routing and latency
- Urban planning and accessibility
- Game AI for area control
- Logistics and distribution planning

This problem demonstrates the power of multi-source BFS
for distance calculation from multiple starting points.
"""
