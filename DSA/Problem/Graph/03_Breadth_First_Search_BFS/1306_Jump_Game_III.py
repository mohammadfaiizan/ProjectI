"""
1306. Jump Game III
Difficulty: Medium

Problem:
Given an array of non-negative integers arr, you are initially positioned at start index 
of the array. When you are at index i, you can jump to i + arr[i] or i - arr[i], check 
if you can reach to any index with value 0.

Notice that you can not jump outside of the array at any time.

Examples:
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true

Input: arr = [4,2,3,0,3,1,2], start = 0
Output: false

Input: arr = [3,0,2,1,2], start = 2
Output: false

Constraints:
- 1 <= arr.length <= 5 * 10^4
- 0 <= arr[i] < arr.length
- 0 <= start < arr.length
"""

from typing import List
from collections import deque

class Solution:
    def canReach_approach1_bfs(self, arr: List[int], start: int) -> bool:
        """
        Approach 1: BFS (Optimal)
        
        Use BFS to explore all reachable indices from start.
        
        Time: O(N) - visit each index at most once
        Space: O(N) - queue and visited set
        """
        if not arr or start >= len(arr):
            return False
        
        n = len(arr)
        
        # Check if start position has value 0
        if arr[start] == 0:
            return True
        
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            # Calculate next possible positions
            next_positions = [current + arr[current], current - arr[current]]
            
            for next_pos in next_positions:
                # Check bounds
                if 0 <= next_pos < n and next_pos not in visited:
                    # Check if we found a zero
                    if arr[next_pos] == 0:
                        return True
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def canReach_approach2_dfs(self, arr: List[int], start: int) -> bool:
        """
        Approach 2: DFS Recursive
        
        Use recursive DFS to explore reachable indices.
        
        Time: O(N)
        Space: O(N) - recursion stack + visited set
        """
        if not arr or start >= len(arr):
            return False
        
        n = len(arr)
        visited = set()
        
        def dfs(index):
            """DFS to check if we can reach zero from current index"""
            if index < 0 or index >= n or index in visited:
                return False
            
            if arr[index] == 0:
                return True
            
            visited.add(index)
            
            # Try both directions
            return (dfs(index + arr[index]) or 
                   dfs(index - arr[index]))
        
        return dfs(start)
    
    def canReach_approach3_iterative_dfs(self, arr: List[int], start: int) -> bool:
        """
        Approach 3: Iterative DFS
        
        Use stack-based DFS to avoid recursion.
        
        Time: O(N)
        Space: O(N)
        """
        if not arr or start >= len(arr):
            return False
        
        n = len(arr)
        
        if arr[start] == 0:
            return True
        
        stack = [start]
        visited = {start}
        
        while stack:
            current = stack.pop()
            
            # Try both jump directions
            for next_pos in [current + arr[current], current - arr[current]]:
                if 0 <= next_pos < n and next_pos not in visited:
                    if arr[next_pos] == 0:
                        return True
                    
                    visited.add(next_pos)
                    stack.append(next_pos)
        
        return False
    
    def canReach_approach4_optimized_early_termination(self, arr: List[int], start: int) -> bool:
        """
        Approach 4: Optimized with Early Termination
        
        Add optimizations for better average case performance.
        
        Time: O(N)
        Space: O(N)
        """
        if not arr or start >= len(arr):
            return False
        
        n = len(arr)
        
        # Early check
        if arr[start] == 0:
            return True
        
        # Quick scan for zeros
        zero_positions = set()
        for i, val in enumerate(arr):
            if val == 0:
                zero_positions.add(i)
        
        if not zero_positions:
            return False
        
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            
            # Try both directions
            for next_pos in [current + arr[current], current - arr[current]]:
                if 0 <= next_pos < n and next_pos not in visited:
                    if next_pos in zero_positions:
                        return True
                    
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False

def test_jump_game_iii():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (arr, start, expected)
        ([4,2,3,0,3,1,2], 5, True),
        ([4,2,3,0,3,1,2], 0, False),
        ([3,0,2,1,2], 2, False),
        ([0], 0, True),
        ([1,1,1,1,0], 0, False),
        ([2,1,1,0], 0, True),
        ([0,3,0,2,3], 2, True),
        ([1,0,3], 1, True),
    ]
    
    approaches = [
        ("BFS", solution.canReach_approach1_bfs),
        ("DFS Recursive", solution.canReach_approach2_dfs),
        ("Iterative DFS", solution.canReach_approach3_iterative_dfs),
        ("Optimized Early Term", solution.canReach_approach4_optimized_early_termination),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (arr, start, expected) in enumerate(test_cases):
            result = func(arr, start)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} arr={arr}, start={start}, expected={expected}, got={result}")

def demonstrate_jump_mechanics():
    """Demonstrate jump game mechanics"""
    print("\n=== Jump Game Mechanics Demo ===")
    
    arr = [4,2,3,0,3,1,2]
    start = 5
    
    print(f"Array: {arr}")
    print(f"Start: {start}")
    print(f"Goal: Reach any index with value 0")
    
    # Show positions with value 0
    zero_positions = [i for i, val in enumerate(arr) if val == 0]
    print(f"Zero positions: {zero_positions}")
    
    # BFS simulation
    n = len(arr)
    queue = deque([(start, 0, [start])])
    visited = {start}
    
    print(f"\nBFS exploration:")
    
    while queue:
        current, steps, path = queue.popleft()
        
        print(f"\nStep {steps}: At index {current} (value={arr[current]})")
        print(f"  Path: {path}")
        
        if arr[current] == 0:
            print(f"  🎯 Found zero! Success in {steps} steps")
            print(f"  Complete path: {' -> '.join(map(str, path))}")
            break
        
        # Show possible jumps
        left_jump = current - arr[current]
        right_jump = current + arr[current]
        
        print(f"  Possible jumps:")
        print(f"    Left: {current} - {arr[current]} = {left_jump}")
        print(f"    Right: {current} + {arr[current]} = {right_jump}")
        
        valid_moves = []
        for next_pos in [left_jump, right_jump]:
            if 0 <= next_pos < n and next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, steps + 1, path + [next_pos]))
                valid_moves.append(next_pos)
        
        if valid_moves:
            print(f"  Valid moves: {valid_moves}")
        else:
            print(f"  No valid moves")

def analyze_jump_patterns():
    """Analyze different jump patterns and reachability"""
    print("\n=== Jump Pattern Analysis ===")
    
    examples = [
        ("Linear Progression", [1,1,1,0], 0, "Can reach end"),
        ("Backwards Only", [3,0,2,1], 3, "Must go backwards"),
        ("Oscillation", [2,1,2,0], 0, "Oscillating pattern"),
        ("Blocked Path", [3,3,3,0], 0, "Cannot reach zero"),
        ("Immediate Zero", [0,1,2,3], 0, "Start at zero"),
    ]
    
    solution = Solution()
    
    for name, arr, start, description in examples:
        result = solution.canReach_approach1_bfs(arr, start)
        print(f"\n{name}:")
        print(f"  Array: {arr}")
        print(f"  Start: {start}")
        print(f"  Description: {description}")
        print(f"  Can reach zero: {result}")
        
        # Show reachable positions
        n = len(arr)
        queue = deque([start])
        reachable = {start}
        
        while queue:
            current = queue.popleft()
            
            for next_pos in [current + arr[current], current - arr[current]]:
                if 0 <= next_pos < n and next_pos not in reachable:
                    reachable.add(next_pos)
                    queue.append(next_pos)
        
        print(f"  Reachable positions: {sorted(reachable)}")
        zero_positions = {i for i, val in enumerate(arr) if val == 0}
        intersection = reachable & zero_positions
        print(f"  Reachable zeros: {sorted(intersection)}")

def compare_bfs_vs_dfs():
    """Compare BFS vs DFS for jump game"""
    print("\n=== BFS vs DFS Comparison ===")
    
    print("BFS Approach:")
    print("  ✅ Level-order exploration")
    print("  ✅ Shortest path to zero (minimum jumps)")
    print("  ✅ Systematic exploration")
    print("  ❌ Requires queue storage")
    
    print("\nDFS Approach:")
    print("  ✅ Memory efficient (recursion/stack)")
    print("  ✅ May find solution quickly with good path")
    print("  ✅ Simple recursive implementation")
    print("  ❌ May explore deeper paths first")
    
    print("\nKey Insights:")
    print("• Both have O(N) time complexity")
    print("• Both explore same reachable state space")
    print("• BFS guarantees minimum jumps if needed")
    print("• DFS may be faster on average for existence check")
    print("• Problem only asks for existence, not shortest path")
    
    print("\nReal-world Applications:")
    print("• Game AI: Character movement with special abilities")
    print("• Robot navigation: Fixed jump distances")
    print("• Puzzle solving: Constraint-based movement")
    print("• Network routing: Fixed hop distances")
    print("• Dynamic programming: State reachability")

if __name__ == "__main__":
    test_jump_game_iii()
    demonstrate_jump_mechanics()
    analyze_jump_patterns()
    compare_bfs_vs_dfs()

"""
Graph Theory Concepts:
1. Reachability Analysis in Constrained Graphs
2. BFS/DFS for Existence Queries
3. State Space Exploration with Rules
4. Jump-based Graph Traversal

Key Jump Game Insights:
- Each array index is a graph node
- Jump rules define directed edges
- Goal: Check if zero-value nodes are reachable
- BFS/DFS both work for existence queries

Algorithm Strategy:
- Model array as graph with jump-based edges
- Use BFS/DFS to explore reachable positions
- Early termination when zero found
- Track visited to avoid cycles

Real-world Applications:
- Game development (character abilities)
- Robot navigation (fixed movement patterns)
- Puzzle game AI
- Network analysis (hop-based routing)
- Dynamic programming state transitions

This problem demonstrates graph reachability analysis
with constraint-based movement rules.
"""
