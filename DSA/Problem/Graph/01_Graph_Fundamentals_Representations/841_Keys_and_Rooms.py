"""
841. Keys and Rooms
Difficulty: Medium

Problem:
There are n rooms labeled from 0 to n - 1 and initially, you can only access room 0.
Each room contains a set of keys, and each key has a number on it corresponding to 
the room it unlocks. You can walk back and forth between rooms freely once unlocked.

Given an array rooms where rooms[i] is the set of keys you can obtain if you visited room i, 
return true if you can visit all rooms, or false otherwise.

Examples:
Input: rooms = [[1],[2],[3],[]]
Output: true
Explanation: Start in room 0, pick up key 1. Go to room 1, pick up key 2. 
Go to room 2, pick up key 3. Go to room 3.

Input: rooms = [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: You can not enter room 2 since the only key that unlocks it is in room 2.

Constraints:
- n == rooms.length
- 2 <= n <= 1000
- 0 <= rooms[i].length <= 1000
- 0 <= rooms[i][j] < n
- All the values of rooms[i] are unique
"""

from typing import List
from collections import deque, defaultdict

class Solution:
    def canVisitAllRooms_approach1_dfs_recursive(self, rooms: List[List[int]]) -> bool:
        """
        Approach 1: DFS with Recursion
        
        Start from room 0 and recursively visit all reachable rooms.
        Track visited rooms to avoid cycles.
        
        Time: O(N + K) where N = number of rooms, K = total number of keys
        Space: O(N) for visited set and recursion stack
        """
        n = len(rooms)
        visited = set()
        
        def dfs(room):
            if room in visited:
                return
            
            visited.add(room)
            
            # Visit all rooms we can unlock from current room
            for key in rooms[room]:
                if key < n:  # Valid room number
                    dfs(key)
        
        dfs(0)
        return len(visited) == n
    
    def canVisitAllRooms_approach2_dfs_iterative(self, rooms: List[List[int]]) -> bool:
        """
        Approach 2: DFS with Iteration using Stack
        
        Use explicit stack to avoid recursion limits.
        
        Time: O(N + K)
        Space: O(N)
        """
        n = len(rooms)
        visited = set()
        stack = [0]  # Start from room 0
        
        while stack:
            room = stack.pop()
            
            if room in visited:
                continue
            
            visited.add(room)
            
            # Add all accessible rooms to stack
            for key in rooms[room]:
                if key < n and key not in visited:
                    stack.append(key)
        
        return len(visited) == n
    
    def canVisitAllRooms_approach3_bfs(self, rooms: List[List[int]]) -> bool:
        """
        Approach 3: BFS with Queue
        
        Use BFS to explore rooms level by level.
        
        Time: O(N + K)
        Space: O(N)
        """
        n = len(rooms)
        visited = set()
        queue = deque([0])
        
        while queue:
            room = queue.popleft()
            
            if room in visited:
                continue
            
            visited.add(room)
            
            # Add all rooms we can access from current room
            for key in rooms[room]:
                if key < n and key not in visited:
                    queue.append(key)
        
        return len(visited) == n
    
    def canVisitAllRooms_approach4_union_find(self, rooms: List[List[int]]) -> bool:
        """
        Approach 4: Union-Find approach
        
        Build connected components and check if all rooms are reachable from room 0.
        This is overkill for this problem but demonstrates the concept.
        
        Time: O(N + K * α(N)) where α is inverse Ackermann function
        Space: O(N)
        """
        n = len(rooms)
        
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.components = n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                self.components -= 1
        
        uf = UnionFind(n)
        
        # Union rooms that can access each other
        for room, keys in enumerate(rooms):
            for key in keys:
                if key < n:
                    uf.union(room, key)
        
        # Check if all rooms are in the same component as room 0
        root_0 = uf.find(0)
        for room in range(n):
            if uf.find(room) != root_0:
                return False
        
        return True
    
    def canVisitAllRooms_approach5_graph_analysis(self, rooms: List[List[int]]) -> bool:
        """
        Approach 5: Graph analysis with reachability matrix
        
        Build a graph and analyze reachability from room 0.
        
        Time: O(N + K)
        Space: O(N + K)
        """
        n = len(rooms)
        
        # Build adjacency list
        graph = defaultdict(list)
        for room, keys in enumerate(rooms):
            for key in keys:
                if key < n:
                    graph[room].append(key)
        
        # Find all reachable rooms from room 0
        reachable = set()
        
        def explore(room):
            if room in reachable:
                return
            
            reachable.add(room)
            for next_room in graph[room]:
                explore(next_room)
        
        explore(0)
        
        return len(reachable) == n
    
    def canVisitAllRooms_approach6_optimized_early_termination(self, rooms: List[List[int]]) -> bool:
        """
        Approach 6: Optimized with early termination
        
        Stop as soon as we've visited all rooms.
        
        Time: O(N + K) with better average case
        Space: O(N)
        """
        n = len(rooms)
        visited = [False] * n
        stack = [0]
        visited_count = 0
        
        while stack and visited_count < n:
            room = stack.pop()
            
            if visited[room]:
                continue
            
            visited[room] = True
            visited_count += 1
            
            # Early termination
            if visited_count == n:
                return True
            
            # Add unvisited accessible rooms
            for key in rooms[room]:
                if key < n and not visited[key]:
                    stack.append(key)
        
        return visited_count == n

def test_keys_and_rooms():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (rooms, expected)
        ([[1],[2],[3],[]], True),
        ([[1,3],[3,0,1],[2],[0]], False),
        ([[1],[0]], True),  # Simple cycle
        ([[1,2,3],[0],[0],[0]], True),  # All keys in first room
        ([[],[]], False),   # No keys in first room
        ([[]], False),      # Only one room, isolated others
        ([[1,2],[],[3],[]], False),  # Missing connection
        ([[1,2,3,4],[],[],[],[]], True),  # Linear unlocking
    ]
    
    approaches = [
        ("DFS Recursive", solution.canVisitAllRooms_approach1_dfs_recursive),
        ("DFS Iterative", solution.canVisitAllRooms_approach2_dfs_iterative),
        ("BFS", solution.canVisitAllRooms_approach3_bfs),
        ("Union-Find", solution.canVisitAllRooms_approach4_union_find),
        ("Graph Analysis", solution.canVisitAllRooms_approach5_graph_analysis),
        ("Early Termination", solution.canVisitAllRooms_approach6_optimized_early_termination),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (rooms, expected) in enumerate(test_cases):
            result = func(rooms)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} rooms={rooms}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_room_exploration():
    """Demonstrate the room exploration process"""
    print("\n=== Room Exploration Demonstration ===")
    
    rooms = [[1,3],[3,0,1],[2],[0]]
    print(f"Rooms and their keys: {rooms}")
    
    # Trace the exploration process
    n = len(rooms)
    visited = set()
    exploration_order = []
    queue = deque([0])
    
    print(f"\nExploration process:")
    step = 0
    
    while queue:
        room = queue.popleft()
        
        if room in visited:
            continue
        
        visited.add(room)
        exploration_order.append(room)
        step += 1
        
        print(f"Step {step}: Visit room {room}")
        print(f"         Keys found: {rooms[room]}")
        print(f"         Visited so far: {sorted(visited)}")
        
        # Add accessible rooms
        new_rooms = []
        for key in rooms[room]:
            if key < n and key not in visited:
                queue.append(key)
                new_rooms.append(key)
        
        if new_rooms:
            print(f"         New rooms accessible: {new_rooms}")
        
        print()
    
    print(f"Final result:")
    print(f"Visited rooms: {sorted(visited)}")
    print(f"Total rooms: {n}")
    print(f"Can visit all rooms: {len(visited) == n}")
    
    # Show unreachable rooms
    unreachable = set(range(n)) - visited
    if unreachable:
        print(f"Unreachable rooms: {sorted(unreachable)}")

def analyze_graph_connectivity():
    """Analyze different connectivity patterns"""
    print("\n=== Graph Connectivity Analysis ===")
    
    examples = [
        ("Fully Connected", [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]),
        ("Linear Chain", [[1],[2],[3],[]]),
        ("Star Pattern", [[1,2,3],[],[],[]]),
        ("Disconnected", [[1],[0],[3],[]]),
        ("Complex", [[1,3],[2],[3,4],[],[0]]),
    ]
    
    solution = Solution()
    
    for name, rooms in examples:
        result = solution.canVisitAllRooms_approach1_dfs_recursive(rooms)
        
        # Calculate connectivity metrics
        n = len(rooms)
        total_keys = sum(len(keys) for keys in rooms)
        avg_keys = total_keys / n if n > 0 else 0
        
        # Find strongly connected components
        visited = set()
        def dfs_component(room, component):
            if room in visited:
                return
            visited.add(room)
            component.append(room)
            for key in rooms[room]:
                if key < n:
                    dfs_component(key, component)
        
        components = []
        visited = set()
        for room in range(n):
            if room not in visited:
                component = []
                dfs_component(room, component)
                if component:
                    components.append(component)
        
        print(f"{name}:")
        print(f"  Rooms: {rooms}")
        print(f"  Can visit all: {result}")
        print(f"  Average keys per room: {avg_keys:.1f}")
        print(f"  Connected components: {len(components)}")
        if len(components) > 1:
            print(f"  Components: {components}")

if __name__ == "__main__":
    test_keys_and_rooms()
    demonstrate_room_exploration()
    analyze_graph_connectivity()

"""
Graph Theory Concepts:
1. Graph Reachability
2. Connected Components
3. Directed Graph Traversal
4. Accessibility Analysis

Key Insights:
- This is fundamentally a graph reachability problem
- Rooms are vertices, keys represent directed edges
- Starting from room 0, we need to reach all other rooms
- Problem equivalent to checking if all nodes are reachable from a source

Algorithm Comparison:
┌─────────────────┬─────────────┬─────────────┬─────────────────────┐
│ Approach        │ Time        │ Space       │ Characteristics     │
├─────────────────┼─────────────┼─────────────┼─────────────────────┤
│ DFS Recursive   │ O(N + K)    │ O(N)        │ Simple, natural     │
│ DFS Iterative   │ O(N + K)    │ O(N)        │ Avoids recursion    │
│ BFS             │ O(N + K)    │ O(N)        │ Level-by-level      │
│ Union-Find      │ O(N + K*α)  │ O(N)        │ Overkill but works  │
│ Early Term.     │ O(N + K)    │ O(N)        │ Best average case   │
└─────────────────┴─────────────┴─────────────┴─────────────────────┘

Real-world Applications:
- Security system design (access control)
- Dependency resolution (package managers)
- Network connectivity analysis
- Maze solving and pathfinding
- Resource accessibility in systems
"""
