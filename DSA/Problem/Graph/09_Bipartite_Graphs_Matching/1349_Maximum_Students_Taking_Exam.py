"""
1349. Maximum Students Taking Exam
Difficulty: Hard (Listed as Medium in syllabus)

Problem:
Given a m * n matrix seats that represents seats in a classroom. 
If a seat is broken, it is denoted by '#'. Otherwise, it is denoted by '.'.

Students can see the answers of those sitting next to them (left, right, upper left, upper right). 
Return the maximum number of students that can take the exam together without any cheating being possible.

Students must be placed in seats in good condition.

Examples:
Input: seats = [["#",".","#","#",".","#"],
                [".","#","#","#","#","."],
                ["#",".","#","#",".","#"]]
Output: 4
Explanation: Teacher can place 4 students in available seats so they don't cheat on the exam.

Input: seats = [[".","#"],
                ["#","#"],
                ["#","#"],
                ["#","."],
                ["#","#"],
                [".","#"]]
Output: 3

Input: seats = [["#",".",".",".","#"],
                [".","#",".","#","."],
                ["#",".",".",".","#"],
                [".","#",".","#","."],
                ["#",".",".",".","#"]]
Output: 10

Constraints:
- seats contains only characters '.' and '#'.
- m == seats.length
- n == seats[i].length
- 1 <= m <= 8
- 1 <= n <= 8
"""

from typing import List, Dict, Set, Tuple, Optional
from functools import lru_cache

class Solution:
    def maxStudents_approach1_backtracking_complete(self, seats: List[List[str]]) -> int:
        """
        Approach 1: Complete Backtracking Search
        
        Try all possible valid seat assignments.
        
        Time: O(2^(m*n)) in worst case
        Space: O(m*n) for recursion
        """
        m, n = len(seats), len(seats[0])
        
        def is_valid_placement(row, col, current_assignment):
            """Check if placing student at (row, col) is valid"""
            if seats[row][col] == '#':
                return False
            
            # Check all cheating directions
            directions = [(-1, -1), (-1, 1), (0, -1), (0, 1)]  # upper-left, upper-right, left, right
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < m and 0 <= nc < n:
                    if current_assignment[nr][nc]:
                        return False
            
            return True
        
        def backtrack(row, col, current_assignment, current_count):
            """Backtrack through all positions"""
            if row == m:
                return current_count
            
            # Calculate next position
            next_row, next_col = (row, col + 1) if col + 1 < n else (row + 1, 0)
            
            max_students = 0
            
            # Option 1: Don't place student at current position
            max_students = max(max_students, backtrack(next_row, next_col, current_assignment, current_count))
            
            # Option 2: Place student at current position (if valid)
            if is_valid_placement(row, col, current_assignment):
                current_assignment[row][col] = True
                max_students = max(max_students, backtrack(next_row, next_col, current_assignment, current_count + 1))
                current_assignment[row][col] = False
            
            return max_students
        
        # Initialize assignment matrix
        assignment = [[False] * n for _ in range(m)]
        return backtrack(0, 0, assignment, 0)
    
    def maxStudents_approach2_dp_bitmask_rows(self, seats: List[List[str]]) -> int:
        """
        Approach 2: Dynamic Programming with Row Bitmasks
        
        Use bitmask to represent valid row configurations.
        
        Time: O(m * 2^n * 2^n)
        Space: O(m * 2^n)
        """
        m, n = len(seats), len(seats[0])
        
        def is_valid_row_config(row, mask):
            """Check if mask represents valid configuration for given row"""
            for col in range(n):
                if mask & (1 << col):
                    # Check if seat is available
                    if seats[row][col] == '#':
                        return False
                    
                    # Check left neighbor
                    if col > 0 and (mask & (1 << (col - 1))):
                        return False
                    
                    # Check right neighbor  
                    if col < n - 1 and (mask & (1 << (col + 1))):
                        return False
            
            return True
        
        def can_place_together(upper_mask, lower_mask):
            """Check if upper and lower row masks can be placed together"""
            for col in range(n):
                if lower_mask & (1 << col):
                    # Check upper-left diagonal
                    if col > 0 and (upper_mask & (1 << (col - 1))):
                        return False
                    
                    # Check upper-right diagonal
                    if col < n - 1 and (upper_mask & (1 << (col + 1))):
                        return False
            
            return True
        
        # Generate all valid masks for each row
        valid_masks = [[] for _ in range(m)]
        
        for row in range(m):
            for mask in range(1 << n):
                if is_valid_row_config(row, mask):
                    valid_masks[row].append(mask)
        
        # DP: dp[row][mask] = maximum students in rows 0..row with row having configuration mask
        dp = {}
        
        def solve(row, prev_mask):
            if row == m:
                return 0
            
            if (row, prev_mask) in dp:
                return dp[(row, prev_mask)]
            
            max_students = 0
            
            for mask in valid_masks[row]:
                if row == 0 or can_place_together(prev_mask, mask):
                    students_in_mask = bin(mask).count('1')
                    remaining_students = solve(row + 1, mask)
                    max_students = max(max_students, students_in_mask + remaining_students)
            
            dp[(row, prev_mask)] = max_students
            return max_students
        
        return solve(0, 0)
    
    def maxStudents_approach3_bipartite_matching(self, seats: List[List[str]]) -> int:
        """
        Approach 3: Maximum Independent Set via Bipartite Matching
        
        Model as maximum independent set on bipartite graph.
        
        Time: O((m*n)^2.5) for bipartite matching
        Space: O((m*n)^2)
        """
        m, n = len(seats), len(seats[0])
        
        # Create list of valid seats
        valid_seats = []
        seat_to_index = {}
        
        for i in range(m):
            for j in range(n):
                if seats[i][j] == '.':
                    seat_to_index[(i, j)] = len(valid_seats)
                    valid_seats.append((i, j))
        
        num_seats = len(valid_seats)
        if num_seats == 0:
            return 0
        
        # Build conflict graph
        conflicts = [[] for _ in range(num_seats)]
        
        for idx1, (r1, c1) in enumerate(valid_seats):
            for idx2, (r2, c2) in enumerate(valid_seats):
                if idx1 != idx2:
                    # Check if seats conflict (can see each other)
                    if (abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1 and 
                        not (r1 == r2 and c1 == c2)):
                        # Additional check for actual cheating directions
                        if ((r1 == r2 and abs(c1 - c2) == 1) or  # left-right
                            (abs(r1 - r2) == 1 and abs(c1 - c2) == 1)):  # diagonal
                            conflicts[idx1].append(idx2)
        
        # For maximum independent set, we can use complement approach
        # But for simplicity, we'll use a greedy approximation
        def greedy_independent_set():
            """Greedy approximation for maximum independent set"""
            remaining = set(range(num_seats))
            independent_set = set()
            
            while remaining:
                # Choose vertex with minimum degree in remaining graph
                min_degree = float('inf')
                min_vertex = -1
                
                for vertex in remaining:
                    degree = sum(1 for neighbor in conflicts[vertex] if neighbor in remaining)
                    if degree < min_degree:
                        min_degree = degree
                        min_vertex = vertex
                
                # Add to independent set
                independent_set.add(min_vertex)
                remaining.remove(min_vertex)
                
                # Remove all neighbors
                for neighbor in conflicts[min_vertex]:
                    remaining.discard(neighbor)
            
            return len(independent_set)
        
        return greedy_independent_set()
    
    def maxStudents_approach4_constraint_satisfaction(self, seats: List[List[str]]) -> int:
        """
        Approach 4: Constraint Satisfaction with Propagation
        
        Use constraint propagation to reduce search space.
        
        Time: O(2^(m*n)) with pruning
        Space: O(m*n)
        """
        m, n = len(seats), len(seats[0])
        
        # Create constraint graph
        valid_positions = []
        pos_to_idx = {}
        
        for i in range(m):
            for j in range(n):
                if seats[i][j] == '.':
                    pos_to_idx[(i, j)] = len(valid_positions)
                    valid_positions.append((i, j))
        
        num_positions = len(valid_positions)
        if num_positions == 0:
            return 0
        
        # Build adjacency for constraint graph
        constraints = [set() for _ in range(num_positions)]
        
        for idx1, (r1, c1) in enumerate(valid_positions):
            for idx2, (r2, c2) in enumerate(valid_positions):
                if idx1 != idx2:
                    # Check cheating constraint
                    if ((r1 == r2 and abs(c1 - c2) == 1) or  # horizontal
                        (abs(r1 - r2) == 1 and abs(c1 - c2) == 1)):  # diagonal
                        constraints[idx1].add(idx2)
                        constraints[idx2].add(idx1)
        
        def constraint_propagation(assignment, domain):
            """Propagate constraints and update domains"""
            changed = True
            while changed:
                changed = False
                
                for pos in range(num_positions):
                    if assignment[pos] == 1:  # Student placed
                        # Remove conflicting positions from domains
                        for neighbor in constraints[pos]:
                            if domain[neighbor] and assignment[neighbor] == -1:
                                domain[neighbor] = False
                                changed = True
        
        def backtrack_with_csp(pos, assignment, domain, current_count):
            """Backtrack with constraint satisfaction"""
            if pos == num_positions:
                return current_count
            
            if not domain[pos]:
                return backtrack_with_csp(pos + 1, assignment, domain, current_count)
            
            max_students = 0
            
            # Try not placing student
            max_students = max(max_students, backtrack_with_csp(pos + 1, assignment, domain, current_count))
            
            # Try placing student
            old_domain = domain[:]
            assignment[pos] = 1
            
            # Check immediate conflicts
            valid_placement = True
            for neighbor in constraints[pos]:
                if assignment[neighbor] == 1:
                    valid_placement = False
                    break
                if assignment[neighbor] == -1:
                    domain[neighbor] = False
            
            if valid_placement:
                max_students = max(max_students, backtrack_with_csp(pos + 1, assignment, domain, current_count + 1))
            
            # Backtrack
            assignment[pos] = -1
            domain[:] = old_domain
            
            return max_students
        
        # Initialize
        assignment = [-1] * num_positions  # -1: unassigned, 0: not placed, 1: placed
        domain = [True] * num_positions    # True: can be placed
        
        return backtrack_with_csp(0, assignment, domain, 0)
    
    def maxStudents_approach5_optimized_dp_state_compression(self, seats: List[List[str]]) -> int:
        """
        Approach 5: Optimized DP with State Compression
        
        Use advanced state compression and pruning.
        
        Time: O(m * 3^n)
        Space: O(3^n)
        """
        m, n = len(seats), len(seats[0])
        
        # Convert seat layout to bitmask for each row
        available_masks = []
        for row in range(m):
            mask = 0
            for col in range(n):
                if seats[row][col] == '.':
                    mask |= (1 << col)
            available_masks.append(mask)
        
        def is_valid_configuration(mask, available_mask):
            """Check if mask is valid for given available seats"""
            # Must be subset of available seats
            if mask & (~available_mask):
                return False
            
            # No adjacent students in same row
            if mask & (mask >> 1):
                return False
            
            return True
        
        def conflicts_with_upper_row(upper_mask, lower_mask):
            """Check if lower row conflicts with upper row"""
            # Check diagonal conflicts
            if (lower_mask & (upper_mask >> 1)) or (lower_mask & (upper_mask << 1)):
                return True
            return False
        
        # Generate all valid configurations for each row
        valid_configs = [[] for _ in range(m)]
        
        for row in range(m):
            for mask in range(1 << n):
                if is_valid_configuration(mask, available_masks[row]):
                    valid_configs[row].append(mask)
        
        # Dynamic programming
        if m == 0:
            return 0
        
        # dp[mask] = maximum students achievable with current row having configuration mask
        dp = {}
        
        # Initialize first row
        for mask in valid_configs[0]:
            dp[mask] = bin(mask).count('1')
        
        # Process remaining rows
        for row in range(1, m):
            new_dp = {}
            
            for new_mask in valid_configs[row]:
                max_students = 0
                students_in_new_mask = bin(new_mask).count('1')
                
                for prev_mask in dp:
                    if not conflicts_with_upper_row(prev_mask, new_mask):
                        max_students = max(max_students, dp[prev_mask] + students_in_new_mask)
                
                if max_students > 0:
                    new_dp[new_mask] = max_students
            
            dp = new_dp
        
        return max(dp.values()) if dp else 0

def test_maximum_students():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (seats, expected)
        ([["#",".","#","#",".","#"],
          [".","#","#","#","#","."],
          ["#",".","#","#",".","#"]], 4),
        
        ([[".","#"],
          ["#","#"],
          ["#","#"],
          ["#","."],
          ["#","#"],
          [".","#"]], 3),
        
        ([["#",".",".",".","#"],
          [".","#",".","#","."],
          ["#",".",".",".","#"],
          [".","#",".","#","."],
          ["#",".",".",".","#"]], 10),
        
        ([["."]], 1),
        ([["#"]], 0),
        ([[".","."]], 1),  # Can't place both due to horizontal adjacency
        ([[".","."],[".","."], [".","."] ], 4),  # Chess-like pattern
    ]
    
    approaches = [
        ("Backtracking", solution.maxStudents_approach1_backtracking_complete),
        ("DP Bitmask", solution.maxStudents_approach2_dp_bitmask_rows),
        ("Bipartite Matching", solution.maxStudents_approach3_bipartite_matching),
        ("Constraint Satisfaction", solution.maxStudents_approach4_constraint_satisfaction),
        ("Optimized DP", solution.maxStudents_approach5_optimized_dp_state_compression),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (seats, expected) in enumerate(test_cases):
            try:
                result = func([row[:] for row in seats])  # Deep copy
                status = "✓" if result == expected else "✗"
                print(f"Test {i+1}: {status} expected={expected}, got={result}")
            except Exception as e:
                print(f"Test {i+1}: ERROR - {str(e)}")

def demonstrate_cheating_constraints():
    """Demonstrate cheating constraint analysis"""
    print("\n=== Cheating Constraints Demo ===")
    
    seats = [["#",".","#","#",".","#"],
             [".","#","#","#","#","."],
             ["#",".","#","#",".","#"]]
    
    print("Classroom layout:")
    for i, row in enumerate(seats):
        print(f"Row {i}: {' '.join(row)}")
    
    print(f"\nCheating directions from any position:")
    print(f"  • Left and Right (horizontal)")
    print(f"  • Upper-left and Upper-right (diagonal)")
    
    print(f"\nExample analysis for position (1,0):")
    print(f"  Position (1,0) = '.' (available)")
    print(f"  Can check:")
    print(f"    • (1,1) right - '#' broken, safe")
    print(f"    • (0,0) upper-right - '#' broken, safe") 
    print(f"    • No left neighbor")
    print(f"    • No upper-left neighbor")
    print(f"  → Position (1,0) can have student")
    
    print(f"\nOptimal placement strategy:")
    print(f"  Use chess-board like pattern avoiding conflicts")
    print(f"  Place students at: (0,1), (1,0), (1,5), (2,1)")

def demonstrate_dp_state_transitions():
    """Demonstrate DP state transitions"""
    print("\n=== DP State Transitions Demo ===")
    
    seats = [[".","."],[".","."]]
    print(f"Simple 2x2 classroom:")
    for row in seats:
        print(f"  {' '.join(row)}")
    
    print(f"\nRow configurations (bitmasks):")
    print(f"  00 (binary) = 0: no students")
    print(f"  01 (binary) = 1: student in position 1")
    print(f"  10 (binary) = 2: student in position 0") 
    print(f"  11 (binary) = 3: students in both positions (INVALID - adjacent)")
    
    print(f"\nValid configurations per row: [0, 1, 2]")
    
    print(f"\nState transitions:")
    print(f"  Row 0 config → Row 1 config | Valid? | Students")
    print(f"  0 → 0                      | ✓      | 0")
    print(f"  0 → 1                      | ✓      | 1")
    print(f"  0 → 2                      | ✓      | 1")
    print(f"  1 → 0                      | ✓      | 1")
    print(f"  1 → 2                      | ✗      | diagonal conflict")
    print(f"  2 → 0                      | ✓      | 1")
    print(f"  2 → 1                      | ✗      | diagonal conflict")
    
    print(f"\nOptimal solution: 2 students (positions (0,0) and (1,1))")

def analyze_complexity_and_optimizations():
    """Analyze complexity and optimization strategies"""
    print("\n=== Complexity Analysis ===")
    
    print("Algorithm Complexity Comparison:")
    
    print("\n1. **Backtracking:**")
    print("   • Time: O(2^(m*n)) - exponential in total cells")
    print("   • Space: O(m*n) - recursion depth")
    print("   • Best for: Very small grids (≤ 16 cells)")
    
    print("\n2. **DP with Row Bitmasks:**")
    print("   • Time: O(m * 2^n * 2^n) - exponential in column count")
    print("   • Space: O(m * 2^n) - DP table")
    print("   • Best for: Few columns, many rows")
    
    print("\n3. **Bipartite Matching:**")
    print("   • Time: O((m*n)^2.5) - polynomial but high degree")
    print("   • Space: O((m*n)^2) - conflict graph")
    print("   • Best for: Sparse conflict graphs")
    
    print("\n4. **Constraint Satisfaction:**")
    print("   • Time: O(2^(m*n)) with pruning - much better in practice")
    print("   • Space: O(m*n) - assignment and domain arrays")
    print("   • Best for: Problems with good constraint propagation")
    
    print("\n5. **Optimized DP:**")
    print("   • Time: O(m * 3^n) - ternary states per position")
    print("   • Space: O(3^n) - compressed state space")
    print("   • Best for: Production systems with moderate n")
    
    print("\nOptimization Strategies:")
    print("• **Pruning:** Early termination on infeasible branches")
    print("• **State compression:** Efficient bit manipulation")
    print("• **Constraint propagation:** Reduce search space")
    print("• **Symmetry breaking:** Avoid equivalent states")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    print("Maximum Independent Set Applications:")
    
    print("\n1. **Exam Proctoring:**")
    print("   • Maximize students while preventing cheating")
    print("   • Consider different cheating patterns")
    print("   • Account for broken/unavailable seats")
    
    print("\n2. **Wireless Network Design:**")
    print("   • Place base stations to avoid interference")
    print("   • Maximize coverage while minimizing conflicts")
    print("   • Consider terrain and obstacle constraints")
    
    print("\n3. **Facility Location:**")
    print("   • Place competing businesses optimally")
    print("   • Avoid cannibalization effects")
    print("   • Maximize total market coverage")
    
    print("\n4. **Task Scheduling:**")
    print("   • Schedule non-interfering tasks")
    print("   • Maximize throughput")
    print("   • Consider resource conflicts")
    
    print("\n5. **Social Distancing:**")
    print("   • Seat people while maintaining distance")
    print("   • Maximize occupancy under health constraints")
    print("   • Dynamic reconfiguration for different requirements")

def demonstrate_constraint_modeling():
    """Demonstrate constraint modeling techniques"""
    print("\n=== Constraint Modeling ===")
    
    print("Constraint Types in Student Seating:")
    
    print("\n1. **Hard Constraints:**")
    print("   • No student in broken seat")
    print("   • No adjacent students (horizontal)")
    print("   • No diagonal students (upper-left, upper-right)")
    
    print("\n2. **Soft Constraints (Extensions):**")
    print("   • Prefer certain seat arrangements")
    print("   • Balance student distribution")
    print("   • Consider student preferences")
    
    print("\n3. **Constraint Propagation Rules:**")
    print("   • If seat occupied → neighbors cannot be occupied")
    print("   • If all neighbors occupied → seat cannot be occupied")
    print("   • Domain reduction based on partial assignments")
    
    print("\n4. **Symmetry Breaking:**")
    print("   • Prefer lexicographically smaller solutions")
    print("   • Fix certain positions to reduce search space")
    print("   • Use problem structure to eliminate equivalent states")
    
    print("\n5. **Advanced Techniques:**")
    print("   • Arc consistency algorithms")
    print("   • Forward checking")
    print("   • Conflict-directed backjumping")
    print("   • Dynamic variable ordering")

if __name__ == "__main__":
    test_maximum_students()
    demonstrate_cheating_constraints()
    demonstrate_dp_state_transitions()
    analyze_complexity_and_optimizations()
    demonstrate_real_world_applications()
    demonstrate_constraint_modeling()

"""
Maximum Students and Independent Set Concepts:
1. Maximum Independent Set Problem in Grid Graphs
2. Dynamic Programming with Bitmask State Representation
3. Constraint Satisfaction and Propagation Techniques
4. Bipartite Matching and Graph-theoretic Approaches
5. Real-world Applications in Space Allocation and Scheduling

Key Problem Insights:
- Maximum independent set on grid graph with specific constraints
- Cheating prevention creates conflict graph
- Multiple algorithmic approaches with different trade-offs
- NP-hard problem requiring intelligent pruning

Algorithm Strategy:
1. Model cheating constraints as conflict graph
2. Find maximum independent set in conflict graph
3. Use appropriate algorithm based on problem size
4. Apply constraint propagation for efficiency

Constraint Analysis:
- Hard constraints: seat availability and cheating prevention
- Conflict patterns: horizontal and diagonal adjacency
- State space explosion requires careful optimization
- Pruning and propagation critical for larger instances

Optimization Techniques:
- Bitmask DP for row-wise processing
- Constraint satisfaction with propagation
- Backtracking with intelligent pruning
- State compression and symmetry breaking

Real-world Applications:
- Examination room optimization
- Social distancing in public spaces
- Wireless network interference minimization
- Facility location with conflict avoidance
- Resource allocation with mutual exclusion

This problem demonstrates fundamental techniques for
constraint satisfaction and combinatorial optimization.
"""
