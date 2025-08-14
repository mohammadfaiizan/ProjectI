"""
Constraint Satisfaction Problems with Backtracking
==================================================

Topics: N-Queens, Sudoku, Graph Coloring, Cryptarithmetic, CSP techniques
Companies: Google, Amazon, Microsoft, Facebook, Apple, Tesla, Airbnb
Difficulty: Medium to Hard
Time Complexity: Exponential, but pruning makes it practical
Space Complexity: O(depth) for recursion stack
"""

from typing import List, Set, Dict, Tuple, Optional, Callable
import copy

class ConstraintSatisfactionProblems:
    
    def __init__(self):
        """Initialize with solution tracking and constraint checking metrics"""
        self.solutions = []
        self.call_count = 0
        self.constraint_checks = 0
        self.pruned_branches = 0
    
    # ==========================================
    # 1. N-QUEENS PROBLEM
    # ==========================================
    
    def solve_n_queens(self, n: int) -> List[List[str]]:
        """
        Classic N-Queens Problem: Place N queens on N×N board
        
        Constraint: No two queens attack each other
        - Same row: handled by placing one queen per row
        - Same column: track occupied columns
        - Same diagonal: track diagonal sums and differences
        
        Company: Google, Amazon, Microsoft
        Difficulty: Hard
        Time: O(n!), Space: O(n)
        """
        solutions = []
        board = [-1] * n  # board[i] = column position of queen in row i
        
        def is_safe(row: int, col: int) -> bool:
            """Check if queen can be placed at (row, col)"""
            self.constraint_checks += 1
            
            for i in range(row):
                # Check column conflict
                if board[i] == col:
                    return False
                
                # Check diagonal conflicts
                # Main diagonal: row - col is constant
                # Anti-diagonal: row + col is constant
                if abs(board[i] - col) == abs(i - row):
                    return False
            
            return True
        
        def backtrack(row: int) -> None:
            self.call_count += 1
            
            print(f"{'  ' * row}Placing queen in row {row}, board state: {board[:row]}")
            
            # BASE CASE: All queens placed successfully
            if row == n:
                # Convert board to string representation
                solution = []
                for i in range(n):
                    row_str = ['.'] * n
                    row_str[board[i]] = 'Q'
                    solution.append(''.join(row_str))
                solutions.append(solution)
                print(f"{'  ' * row}✓ Found solution: {board}")
                return
            
            # TRY: Each column in current row
            for col in range(n):
                if is_safe(row, col):
                    print(f"{'  ' * row}Trying queen at ({row}, {col})")
                    
                    # MAKE CHOICE: Place queen
                    board[row] = col
                    
                    # RECURSE: Move to next row
                    backtrack(row + 1)
                    
                    # BACKTRACK: Remove queen (implicit - will be overwritten)
                    board[row] = -1
                    print(f"{'  ' * row}← Backtracking from ({row}, {col})")
                else:
                    self.pruned_branches += 1
                    print(f"{'  ' * row}✗ Cannot place queen at ({row}, {col}) - conflicts")
        
        print(f"Solving {n}-Queens problem:")
        self.call_count = 0
        self.constraint_checks = 0
        self.pruned_branches = 0
        
        backtrack(0)
        
        print(f"\nStatistics:")
        print(f"  Solutions found: {len(solutions)}")
        print(f"  Function calls: {self.call_count}")
        print(f"  Constraint checks: {self.constraint_checks}")
        print(f"  Pruned branches: {self.pruned_branches}")
        
        return solutions
    
    def n_queens_optimized(self, n: int) -> int:
        """
        Optimized N-Queens using bit manipulation
        
        Uses bitmasks to track attacked positions
        Time: O(n!), Space: O(n) - but much faster in practice
        """
        count = 0
        
        def backtrack(row: int, cols: int, diag1: int, diag2: int) -> None:
            nonlocal count
            
            # BASE CASE: All queens placed
            if row == n:
                count += 1
                return
            
            # Available positions = all positions - attacked positions
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            
            while available:
                # Get rightmost available position
                pos = available & -available
                available -= pos
                
                # Place queen and recurse
                backtrack(row + 1,
                         cols | pos,           # Add column
                         (diag1 | pos) << 1,  # Add main diagonal
                         (diag2 | pos) >> 1)  # Add anti-diagonal
        
        backtrack(0, 0, 0, 0)
        return count
    
    def print_n_queens_board(self, solution: List[str]) -> None:
        """Pretty print N-Queens solution"""
        print("N-Queens Solution:")
        for i, row in enumerate(solution):
            print(f"  {i}: {row}")
        print()
    
    # ==========================================
    # 2. SUDOKU SOLVER
    # ==========================================
    
    def solve_sudoku(self, board: List[List[str]]) -> bool:
        """
        Solve 9×9 Sudoku puzzle using backtracking
        
        Constraints:
        - Each row contains digits 1-9 exactly once
        - Each column contains digits 1-9 exactly once
        - Each 3×3 box contains digits 1-9 exactly once
        
        Company: Amazon, Microsoft, Google
        Difficulty: Medium
        Time: O(9^(n*n)), Space: O(n*n)
        """
        def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            """Check if placing num at (row, col) violates constraints"""
            self.constraint_checks += 1
            
            # Check row constraint
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check column constraint
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check 3×3 box constraint
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def find_empty_cell(board: List[List[str]]) -> Optional[Tuple[int, int]]:
            """Find next empty cell in the board"""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        return (i, j)
            return None
        
        def backtrack() -> bool:
            self.call_count += 1
            
            # Find next empty cell
            empty_cell = find_empty_cell(board)
            if not empty_cell:
                return True  # Puzzle solved
            
            row, col = empty_cell
            print(f"Trying to fill cell ({row}, {col})")
            
            # Try digits 1-9
            for num in '123456789':
                if is_valid(board, row, col, num):
                    print(f"  Placing {num} at ({row}, {col})")
                    
                    # MAKE CHOICE: Place number
                    board[row][col] = num
                    
                    # RECURSE: Try to solve rest of puzzle
                    if backtrack():
                        return True
                    
                    # BACKTRACK: Remove number
                    board[row][col] = '.'
                    print(f"  ← Backtracking from ({row}, {col})")
                else:
                    self.pruned_branches += 1
            
            return False  # No valid number found for this cell
        
        print("Solving Sudoku puzzle:")
        self.call_count = 0
        self.constraint_checks = 0
        self.pruned_branches = 0
        
        solved = backtrack()
        
        print(f"\nSudoku Statistics:")
        print(f"  Solved: {solved}")
        print(f"  Function calls: {self.call_count}")
        print(f"  Constraint checks: {self.constraint_checks}")
        print(f"  Pruned branches: {self.pruned_branches}")
        
        return solved
    
    def print_sudoku_board(self, board: List[List[str]]) -> None:
        """Pretty print Sudoku board"""
        print("Sudoku Board:")
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            row = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row += "| "
                row += board[i][j] + " "
            print(row)
        print()
    
    # ==========================================
    # 3. GRAPH COLORING
    # ==========================================
    
    def graph_coloring(self, graph: List[List[int]], num_colors: int) -> List[int]:
        """
        Graph Coloring Problem: Color vertices with minimum colors
        
        Constraint: No two adjacent vertices have same color
        
        Company: Google, Microsoft
        Difficulty: Hard (NP-Complete)
        Time: O(k^n), Space: O(n)
        """
        n = len(graph)
        colors = [0] * n  # color[i] = color of vertex i (0 = uncolored)
        
        def is_safe(vertex: int, color: int) -> bool:
            """Check if vertex can be colored with given color"""
            self.constraint_checks += 1
            
            # Check all adjacent vertices
            for neighbor in range(n):
                if graph[vertex][neighbor] == 1 and colors[neighbor] == color:
                    return False
            return True
        
        def backtrack(vertex: int) -> bool:
            self.call_count += 1
            
            print(f"{'  ' * vertex}Coloring vertex {vertex}, current colors: {colors}")
            
            # BASE CASE: All vertices colored
            if vertex == n:
                return True
            
            # TRY: Each color for current vertex
            for color in range(1, num_colors + 1):
                if is_safe(vertex, color):
                    print(f"{'  ' * vertex}Assigning color {color} to vertex {vertex}")
                    
                    # MAKE CHOICE: Assign color
                    colors[vertex] = color
                    
                    # RECURSE: Color next vertex
                    if backtrack(vertex + 1):
                        return True
                    
                    # BACKTRACK: Remove color
                    colors[vertex] = 0
                    print(f"{'  ' * vertex}← Backtracking from vertex {vertex}")
                else:
                    self.pruned_branches += 1
            
            return False
        
        print(f"Graph coloring with {num_colors} colors:")
        self.call_count = 0
        self.constraint_checks = 0
        self.pruned_branches = 0
        
        if backtrack(0):
            print(f"✓ Solution found: {colors}")
            return colors
        else:
            print("✗ No solution exists")
            return []
    
    # ==========================================
    # 4. CRYPTARITHMETIC PUZZLES
    # ==========================================
    
    def solve_cryptarithmetic(self, puzzle: str) -> Dict[str, int]:
        """
        Solve cryptarithmetic puzzles like SEND + MORE = MONEY
        
        Constraints:
        - Each letter represents a unique digit
        - Leading letters cannot be 0
        - Arithmetic equation must be satisfied
        
        Company: Google, Microsoft
        Difficulty: Hard
        Time: O(10!), Space: O(1)
        """
        # Parse puzzle (simplified for SEND + MORE = MONEY)
        if puzzle == "SEND + MORE = MONEY":
            letters = ['S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y']
            leading_letters = {'S', 'M'}  # Cannot be 0
        else:
            # Generic parsing would go here
            return {}
        
        assignment = {}
        used_digits = set()
        
        def is_valid_assignment() -> bool:
            """Check if current assignment satisfies the equation"""
            if len(assignment) != len(letters):
                return True  # Partial assignment, continue
            
            # Convert words to numbers
            send = (assignment['S'] * 1000 + assignment['E'] * 100 + 
                   assignment['N'] * 10 + assignment['D'])
            more = (assignment['M'] * 1000 + assignment['O'] * 100 + 
                   assignment['R'] * 10 + assignment['E'])
            money = (assignment['M'] * 10000 + assignment['O'] * 1000 + 
                    assignment['N'] * 100 + assignment['E'] * 10 + assignment['Y'])
            
            return send + more == money
        
        def backtrack(letter_index: int) -> bool:
            self.call_count += 1
            
            # BASE CASE: All letters assigned
            if letter_index >= len(letters):
                return is_valid_assignment()
            
            current_letter = letters[letter_index]
            print(f"{'  ' * letter_index}Assigning digit to '{current_letter}'")
            
            # TRY: Each available digit
            for digit in range(10):
                # CONSTRAINT: Leading letters cannot be 0
                if digit == 0 and current_letter in leading_letters:
                    continue
                
                # CONSTRAINT: Each digit used at most once
                if digit in used_digits:
                    continue
                
                print(f"{'  ' * letter_index}Trying {current_letter} = {digit}")
                
                # MAKE CHOICE: Assign digit to letter
                assignment[current_letter] = digit
                used_digits.add(digit)
                
                # RECURSE: Assign next letter
                if backtrack(letter_index + 1):
                    return True
                
                # BACKTRACK: Remove assignment
                del assignment[current_letter]
                used_digits.remove(digit)
                print(f"{'  ' * letter_index}← Backtracking {current_letter} = {digit}")
            
            return False
        
        print(f"Solving cryptarithmetic puzzle: {puzzle}")
        self.call_count = 0
        
        if backtrack(0):
            print(f"✓ Solution found: {assignment}")
            
            # Verify solution
            send = sum(assignment[c] * (10 ** (3-i)) for i, c in enumerate('SEND'))
            more = sum(assignment[c] * (10 ** (3-i)) for i, c in enumerate('MORE'))
            money = sum(assignment[c] * (10 ** (4-i)) for i, c in enumerate('MONEY'))
            
            print(f"Verification: {send} + {more} = {money}")
            return assignment
        else:
            print("✗ No solution exists")
            return {}
    
    # ==========================================
    # 5. GENERAL CSP FRAMEWORK
    # ==========================================
    
    def generic_csp_solver(self, variables: List[str], domains: Dict[str, List], 
                          constraints: List[Callable]) -> Dict[str, any]:
        """
        Generic CSP solver framework
        
        Args:
            variables: List of variable names
            domains: Dict mapping variables to possible values
            constraints: List of constraint functions
        
        Returns:
            Solution assignment or empty dict if no solution
        """
        assignment = {}
        
        def is_consistent(variable: str, value: any) -> bool:
            """Check if assigning value to variable is consistent"""
            temp_assignment = assignment.copy()
            temp_assignment[variable] = value
            
            # Check all constraints
            for constraint in constraints:
                if not constraint(temp_assignment):
                    return False
            return True
        
        def select_unassigned_variable() -> Optional[str]:
            """Select next variable to assign (MRV heuristic)"""
            unassigned = [v for v in variables if v not in assignment]
            if not unassigned:
                return None
            
            # Minimum Remaining Values heuristic
            return min(unassigned, key=lambda v: len(domains[v]))
        
        def backtrack() -> bool:
            # BASE CASE: All variables assigned
            if len(assignment) == len(variables):
                return True
            
            # SELECT: Next variable to assign
            variable = select_unassigned_variable()
            if variable is None:
                return True
            
            # TRY: Each value in domain
            for value in domains[variable]:
                if is_consistent(variable, value):
                    # MAKE CHOICE
                    assignment[variable] = value
                    
                    # RECURSE
                    if backtrack():
                        return True
                    
                    # BACKTRACK
                    del assignment[variable]
            
            return False
        
        print("Solving generic CSP:")
        if backtrack():
            print(f"✓ Solution: {assignment}")
            return assignment
        else:
            print("✗ No solution exists")
            return {}
    
    # ==========================================
    # 6. CSP OPTIMIZATION TECHNIQUES
    # ==========================================
    
    def demonstrate_csp_techniques(self) -> None:
        """
        Demonstrate various CSP optimization techniques
        """
        print("=== CSP OPTIMIZATION TECHNIQUES ===")
        print()
        
        print("1. VARIABLE ORDERING HEURISTICS:")
        print("   • MRV (Minimum Remaining Values): Choose variable with fewest legal values")
        print("   • Degree Heuristic: Choose variable involved in most constraints")
        print("   • Combination: MRV first, then degree as tiebreaker")
        print()
        
        print("2. VALUE ORDERING HEURISTICS:")
        print("   • Least Constraining Value: Choose value that rules out fewest choices")
        print("   • Random ordering: Sometimes effective for hard problems")
        print("   • Problem-specific ordering: Use domain knowledge")
        print()
        
        print("3. CONSTRAINT PROPAGATION:")
        print("   • Forward Checking: Check constraints when making assignment")
        print("   • Arc Consistency: Ensure binary constraints are consistent")
        print("   • Maintaining Arc Consistency (MAC): Propagate during search")
        print()
        
        print("4. INTELLIGENT BACKTRACKING:")
        print("   • Backjumping: Jump back to source of conflict")
        print("   • Conflict-directed backjumping: Use conflict analysis")
        print("   • Learning: Remember reasons for failures")
        print()
        
        print("5. PROBLEM DECOMPOSITION:")
        print("   • Tree decomposition: Break problem into tree structure")
        print("   • Cutset conditioning: Instantiate key variables first")
        print("   • Subproblem identification: Solve independent parts separately")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_constraint_satisfaction():
    """Demonstrate all constraint satisfaction problems"""
    print("=== CONSTRAINT SATISFACTION PROBLEMS DEMONSTRATION ===\n")
    
    csp = ConstraintSatisfactionProblems()
    
    # 1. N-Queens Problem
    print("=== N-QUEENS PROBLEM ===")
    print("1. Solving 4-Queens:")
    queens_solutions = csp.solve_n_queens(4)
    print(f"Found {len(queens_solutions)} solutions")
    
    if queens_solutions:
        print("First solution:")
        csp.print_n_queens_board(queens_solutions[0])
    
    print("2. N-Queens count comparison (optimized):")
    for n in range(4, 9):
        count = csp.n_queens_optimized(n)
        print(f"  {n}-Queens: {count} solutions")
    print()
    
    # 2. Sudoku Solver
    print("=== SUDOKU SOLVER ===")
    # Simple Sudoku puzzle for demonstration
    sudoku_board = [
        ['5', '3', '.', '.', '7', '.', '.', '.', '.'],
        ['6', '.', '.', '1', '9', '5', '.', '.', '.'],
        ['.', '9', '8', '.', '.', '.', '.', '6', '.'],
        ['8', '.', '.', '.', '6', '.', '.', '.', '3'],
        ['4', '.', '.', '8', '.', '3', '.', '.', '1'],
        ['7', '.', '.', '.', '2', '.', '.', '.', '6'],
        ['.', '6', '.', '.', '.', '.', '2', '8', '.'],
        ['.', '.', '.', '4', '1', '9', '.', '.', '5'],
        ['.', '.', '.', '.', '8', '.', '.', '7', '9']
    ]
    
    print("Original puzzle:")
    csp.print_sudoku_board(sudoku_board)
    
    # Make a copy for solving
    sudoku_copy = [row[:] for row in sudoku_board]
    solved = csp.solve_sudoku(sudoku_copy)
    
    if solved:
        print("Solved puzzle:")
        csp.print_sudoku_board(sudoku_copy)
    print()
    
    # 3. Graph Coloring
    print("=== GRAPH COLORING ===")
    # Simple graph: triangle (3 vertices, all connected)
    triangle_graph = [
        [0, 1, 1],  # Vertex 0 connected to 1, 2
        [1, 0, 1],  # Vertex 1 connected to 0, 2
        [1, 1, 0]   # Vertex 2 connected to 0, 1
    ]
    
    print("Coloring triangle graph with 3 colors:")
    colors = csp.graph_coloring(triangle_graph, 3)
    if colors:
        print(f"Color assignment: {colors}")
    print()
    
    # 4. Cryptarithmetic
    print("=== CRYPTARITHMETIC PUZZLE ===")
    print("Solving SEND + MORE = MONEY:")
    solution = csp.solve_cryptarithmetic("SEND + MORE = MONEY")
    print()
    
    # 5. CSP Techniques
    csp.demonstrate_csp_techniques()

if __name__ == "__main__":
    demonstrate_constraint_satisfaction()
    
    print("\n=== CONSTRAINT SATISFACTION MASTERY GUIDE ===")
    
    print("\n🎯 CSP PROBLEM CHARACTERISTICS:")
    print("• Variables: What needs to be assigned values")
    print("• Domains: Possible values for each variable")
    print("• Constraints: Rules that must be satisfied")
    print("• Goal: Find assignment satisfying all constraints")
    
    print("\n📋 CSP SOLVING STRATEGY:")
    print("1. Identify variables, domains, and constraints")
    print("2. Choose variable and value ordering heuristics")
    print("3. Implement constraint checking efficiently")
    print("4. Add pruning and propagation techniques")
    print("5. Use backjumping for intelligent backtracking")
    
    print("\n⚡ PERFORMANCE OPTIMIZATION:")
    print("• Constraint checking: Make it as fast as possible")
    print("• Variable ordering: MRV + degree heuristic")
    print("• Value ordering: Least constraining value")
    print("• Pruning: Eliminate invalid branches early")
    print("• Propagation: Reduce domain sizes during search")
    
    print("\n🔍 PROBLEM ANALYSIS:")
    print("• Constraint graph density affects difficulty")
    print("• Domain size vs number of variables trade-off")
    print("• Constraint tightness determines search space")
    print("• Symmetries can be exploited for optimization")
    
    print("\n📚 REAL-WORLD APPLICATIONS:")
    print("• Scheduling: Course scheduling, job shop scheduling")
    print("• Planning: Resource allocation, task assignment")
    print("• Configuration: Product configuration, network design")
    print("• Games: Puzzle solving, game AI")
    print("• Verification: Hardware verification, protocol checking")
