"""
752. Open the Lock
Difficulty: Medium

Problem:
You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: 
'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and 
wrap around: for example we can turn '9' to '0' and '0' to '9'. Each move consists 
of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these 
codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return 
the minimum number of turns required to open the lock, or -1 if it is impossible.

Examples:
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6

Input: deadends = ["8888"], target = "0009"
Output: 1

Input: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
Output: -1

Constraints:
- 1 <= deadends.length <= 500
- deadends[i].length == 4
- target.length == 4
- target will not be in the list of deadends
- target and deadends[i] consist of digits only
"""

from typing import List
from collections import deque

class Solution:
    def openLock_approach1_standard_bfs(self, deadends: List[str], target: str) -> int:
        """
        Approach 1: Standard BFS (Optimal)
        
        Use BFS to find shortest path from "0000" to target.
        Each state has up to 8 neighbors (4 wheels × 2 directions).
        
        Time: O(10^4) = O(1) - bounded by 10000 possible states
        Space: O(10^4) - visited set + queue
        """
        start = "0000"
        dead_set = set(deadends)
        
        # Check if start or target is blocked
        if start in dead_set or target in dead_set:
            return -1
        
        if start == target:
            return 0
        
        queue = deque([(start, 0)])
        visited = {start}
        
        def get_neighbors(state):
            """Generate all possible next states"""
            neighbors = []
            for i in range(4):  # 4 wheels
                digit = int(state[i])
                
                # Turn wheel up
                new_digit = (digit + 1) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]
                neighbors.append(new_state)
                
                # Turn wheel down
                new_digit = (digit - 1) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]
                neighbors.append(new_state)
            
            return neighbors
        
        while queue:
            current_state, turns = queue.popleft()
            
            for neighbor in get_neighbors(current_state):
                if neighbor == target:
                    return turns + 1
                
                if neighbor not in visited and neighbor not in dead_set:
                    visited.add(neighbor)
                    queue.append((neighbor, turns + 1))
        
        return -1
    
    def openLock_approach2_bidirectional_bfs(self, deadends: List[str], target: str) -> int:
        """
        Approach 2: Bidirectional BFS
        
        Search from both start and target simultaneously.
        
        Time: O(10^4)
        Space: O(10^4)
        """
        start = "0000"
        dead_set = set(deadends)
        
        if start in dead_set or target in dead_set:
            return -1
        
        if start == target:
            return 0
        
        def get_neighbors(state):
            neighbors = []
            for i in range(4):
                digit = int(state[i])
                
                # Up
                new_digit = (digit + 1) % 10
                neighbors.append(state[:i] + str(new_digit) + state[i+1:])
                
                # Down  
                new_digit = (digit - 1) % 10
                neighbors.append(state[:i] + str(new_digit) + state[i+1:])
            
            return neighbors
        
        # Two BFS frontiers
        start_queue = deque([start])
        target_queue = deque([target])
        
        start_visited = {start: 0}
        target_visited = {target: 0}
        
        while start_queue or target_queue:
            # Expand from start
            if start_queue:
                for _ in range(len(start_queue)):
                    current = start_queue.popleft()
                    
                    for neighbor in get_neighbors(current):
                        if neighbor in target_visited:
                            return start_visited[current] + target_visited[neighbor] + 1
                        
                        if neighbor not in start_visited and neighbor not in dead_set:
                            start_visited[neighbor] = start_visited[current] + 1
                            start_queue.append(neighbor)
            
            # Expand from target
            if target_queue:
                for _ in range(len(target_queue)):
                    current = target_queue.popleft()
                    
                    for neighbor in get_neighbors(current):
                        if neighbor in start_visited:
                            return target_visited[current] + start_visited[neighbor] + 1
                        
                        if neighbor not in target_visited and neighbor not in dead_set:
                            target_visited[neighbor] = target_visited[current] + 1
                            target_queue.append(neighbor)
        
        return -1
    
    def openLock_approach3_a_star(self, deadends: List[str], target: str) -> int:
        """
        Approach 3: A* Algorithm
        
        Use A* with Manhattan distance heuristic.
        
        Time: O(10^4 * log(10^4))
        Space: O(10^4)
        """
        import heapq
        
        start = "0000"
        dead_set = set(deadends)
        
        if start in dead_set or target in dead_set:
            return -1
        
        if start == target:
            return 0
        
        def heuristic(state):
            """Manhattan distance in circular space"""
            distance = 0
            for i in range(4):
                diff = abs(int(state[i]) - int(target[i]))
                distance += min(diff, 10 - diff)  # Circular distance
            return distance
        
        def get_neighbors(state):
            neighbors = []
            for i in range(4):
                digit = int(state[i])
                
                # Up
                new_digit = (digit + 1) % 10
                neighbors.append(state[:i] + str(new_digit) + state[i+1:])
                
                # Down
                new_digit = (digit - 1) % 10
                neighbors.append(state[:i] + str(new_digit) + state[i+1:])
            
            return neighbors
        
        # Priority queue: (f_score, g_score, state)
        pq = [(heuristic(start), 0, start)]
        visited = {start: 0}
        
        while pq:
            f_score, g_score, current = heapq.heappop(pq)
            
            if current == target:
                return g_score
            
            if visited.get(current, float('inf')) < g_score:
                continue
            
            for neighbor in get_neighbors(current):
                if neighbor in dead_set:
                    continue
                
                new_g_score = g_score + 1
                
                if new_g_score < visited.get(neighbor, float('inf')):
                    visited[neighbor] = new_g_score
                    f_score = new_g_score + heuristic(neighbor)
                    heapq.heappush(pq, (f_score, new_g_score, neighbor))
        
        return -1
    
    def openLock_approach4_optimized_bfs(self, deadends: List[str], target: str) -> int:
        """
        Approach 4: Optimized BFS with Early Termination
        
        Add optimizations for better average case performance.
        
        Time: O(10^4)
        Space: O(10^4)
        """
        start = "0000"
        dead_set = set(deadends)
        
        if start in dead_set or target in dead_set:
            return -1
        
        if start == target:
            return 0
        
        # Pre-generate all neighbors for faster lookup
        def get_neighbors(state):
            neighbors = []
            state_list = list(state)
            
            for i in range(4):
                original = state_list[i]
                
                # Up
                state_list[i] = str((int(original) + 1) % 10)
                neighbors.append(''.join(state_list))
                
                # Down
                state_list[i] = str((int(original) - 1) % 10)
                neighbors.append(''.join(state_list))
                
                # Restore
                state_list[i] = original
            
            return neighbors
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current_state, turns = queue.popleft()
            
            # Generate and check neighbors
            for neighbor in get_neighbors(current_state):
                if neighbor == target:
                    return turns + 1
                
                if neighbor not in visited and neighbor not in dead_set:
                    visited.add(neighbor)
                    queue.append((neighbor, turns + 1))
        
        return -1

def test_open_lock():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (deadends, target, expected)
        (["0201","0101","0102","1212","2002"], "0202", 6),
        (["8888"], "0009", 1),
        (["8887","8889","8878","8898","8788","8988","7888","9888"], "8888", -1),
        (["0000"], "8888", -1),  # Start blocked
        ([], "0001", 1),  # No deadends
        ([], "1111", 4),  # Each wheel once
        (["0001","0010","0100","1000"], "1111", -1),  # Blocked path
    ]
    
    approaches = [
        ("Standard BFS", solution.openLock_approach1_standard_bfs),
        ("Bidirectional BFS", solution.openLock_approach2_bidirectional_bfs),
        ("A* Algorithm", solution.openLock_approach3_a_star),
        ("Optimized BFS", solution.openLock_approach4_optimized_bfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (deadends, target, expected) in enumerate(test_cases):
            result = func(deadends, target)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Target: {target}, Expected: {expected}, Got: {result}")

def demonstrate_lock_opening():
    """Demonstrate lock opening process"""
    print("\n=== Lock Opening Demo ===")
    
    deadends = ["0201","0101","0102","1212","2002"]
    target = "0202"
    start = "0000"
    
    print(f"Start: {start}")
    print(f"Target: {target}")
    print(f"Deadends: {deadends}")
    
    dead_set = set(deadends)
    
    def get_neighbors(state):
        neighbors = []
        for i in range(4):
            digit = int(state[i])
            
            # Up
            new_digit = (digit + 1) % 10
            neighbors.append(state[:i] + str(new_digit) + state[i+1:])
            
            # Down
            new_digit = (digit - 1) % 10
            neighbors.append(state[:i] + str(new_digit) + state[i+1:])
        
        return neighbors
    
    # BFS with path tracking
    queue = deque([(start, 0, [start])])
    visited = {start}
    
    print(f"\nBFS exploration:")
    
    step = 0
    while queue and step < 8:  # Limit for demo
        step += 1
        print(f"\nStep {step}:")
        
        level_size = len(queue)
        for _ in range(min(level_size, 3)):  # Show first few
            if not queue:
                break
                
            current, turns, path = queue.popleft()
            
            print(f"  State: {current}, Turns: {turns}")
            
            if current == target:
                print(f"  🎯 Target reached! Path: {' -> '.join(path)}")
                return
            
            valid_neighbors = []
            for neighbor in get_neighbors(current):
                if neighbor not in visited and neighbor not in dead_set:
                    visited.add(neighbor)
                    queue.append((neighbor, turns + 1, path + [neighbor]))
                    valid_neighbors.append(neighbor)
            
            if valid_neighbors:
                print(f"    Next states: {valid_neighbors[:3]}{'...' if len(valid_neighbors) > 3 else ''}")

def analyze_lock_state_space():
    """Analyze the lock's state space"""
    print("\n=== Lock State Space Analysis ===")
    
    print("State Space Properties:")
    print(f"  • Total states: 10^4 = 10,000")
    print(f"  • Each state has 8 neighbors (4 wheels × 2 directions)")
    print(f"  • Graph is undirected and unweighted")
    print(f"  • Circular wheel property: 9 + 1 = 0, 0 - 1 = 9")
    
    print("\nNeighbor Generation Example (state '1234'):")
    state = "1234"
    
    def get_neighbors_demo(state):
        neighbors = []
        for i in range(4):
            digit = int(state[i])
            
            # Up
            new_digit = (digit + 1) % 10
            neighbors.append((f"Wheel {i} up", state[:i] + str(new_digit) + state[i+1:]))
            
            # Down
            new_digit = (digit - 1) % 10
            neighbors.append((f"Wheel {i} down", state[:i] + str(new_digit) + state[i+1:]))
        
        return neighbors
    
    neighbors = get_neighbors_demo(state)
    for description, neighbor_state in neighbors:
        print(f"  {description}: {state} -> {neighbor_state}")
    
    print("\nDistance Analysis:")
    
    def circular_distance(a, b):
        """Distance between two digits in circular space"""
        diff = abs(a - b)
        return min(diff, 10 - diff)
    
    def state_distance(state1, state2):
        """Manhattan distance between two states"""
        return sum(circular_distance(int(a), int(b)) 
                  for a, b in zip(state1, state2))
    
    examples = [("0000", "0001"), ("0000", "9999"), ("0000", "5555"), ("1234", "5678")]
    
    for s1, s2 in examples:
        dist = state_distance(s1, s2)
        print(f"  Distance from {s1} to {s2}: {dist}")

def compare_search_algorithms():
    """Compare different search algorithms for lock problem"""
    print("\n=== Search Algorithm Comparison ===")
    
    print("1. Standard BFS:")
    print("   ✅ Guaranteed shortest path")
    print("   ✅ Simple implementation")
    print("   ✅ Explores level by level")
    print("   ❌ May explore many unnecessary states")
    
    print("\n2. Bidirectional BFS:")
    print("   ✅ Reduces search space significantly")
    print("   ✅ Meets in the middle")
    print("   ✅ Especially good for longer paths")
    print("   ❌ More complex implementation")
    
    print("\n3. A* Algorithm:")
    print("   ✅ Guided search with heuristic")
    print("   ✅ Often faster than BFS")
    print("   ✅ Optimal with admissible heuristic")
    print("   ❌ Priority queue overhead")
    
    print("\nHeuristic Function (Manhattan Distance):")
    print("• Sum of circular distances for each wheel")
    print("• Admissible: never overestimates actual distance")
    print("• Consistent: satisfies triangle inequality")
    
    print("\nReal-world Applications:")
    print("• Physical lock mechanisms")
    print("• Combination puzzle solving")
    print("• State machine navigation")
    print("• Configuration space search")
    print("• Game AI (puzzle games)")

if __name__ == "__main__":
    test_open_lock()
    demonstrate_lock_opening()
    analyze_lock_state_space()
    compare_search_algorithms()

"""
Graph Theory Concepts:
1. State Space Search in Combinatorial Problems
2. Circular Distance and Wraparound
3. BFS in Constraint Satisfaction Problems
4. Heuristic Search with A*

Key Lock Problem Insights:
- Each lock state is a graph node
- 8 neighbors per state (4 wheels × 2 directions)
- Circular arithmetic for wheel transitions
- Deadends are blocked nodes

Algorithm Strategy:
- Model as unweighted graph with 10^4 nodes
- Use BFS for shortest path guarantee
- Handle circular transitions (9→0, 0→9)
- Early termination when target reached

State Space Optimization:
- Bidirectional BFS reduces search space
- A* with Manhattan distance heuristic
- Deadend checking during exploration
- Efficient neighbor generation

Real-world Applications:
- Physical security systems
- Combination lock design
- Puzzle game algorithms
- Configuration optimization
- State machine design

This problem demonstrates graph search in
discrete combinatorial optimization problems.
"""
