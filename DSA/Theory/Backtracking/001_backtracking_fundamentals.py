"""
Backtracking Fundamentals - Core Concepts and Templates
======================================================

Topics: Backtracking definition, templates, basic examples, decision trees
Companies: All major tech companies test backtracking fundamentals
Difficulty: Easy to Medium
Time Complexity: Often exponential O(2^n), O(n!)
Space Complexity: O(depth) for recursion stack
"""

from typing import List, Set, Dict, Tuple, Optional, Any
import copy

class BacktrackingFundamentals:
    
    def __init__(self):
        """Initialize with solution tracking and performance metrics"""
        self.solutions = []
        self.call_count = 0
        self.max_depth = 0
        self.current_depth = 0
        self.pruned_branches = 0
    
    # ==========================================
    # 1. WHAT IS BACKTRACKING?
    # ==========================================
    
    def explain_backtracking_concept(self) -> None:
        """
        Explain the fundamental concept of backtracking
        
        Backtracking is a systematic way to explore all possible solutions
        by making choices, exploring consequences, and undoing choices
        when they lead to dead ends.
        """
        print("=== WHAT IS BACKTRACKING? ===")
        print("Backtracking is an algorithmic approach that considers searching")
        print("every possible combination in order to solve computational problems.")
        print()
        print("Key Characteristics:")
        print("1. Incremental Solution Building: Build solution step by step")
        print("2. Constraint Checking: Validate partial solutions early")
        print("3. Systematic Exploration: Try all possibilities systematically")
        print("4. Backtrack on Failure: Undo choices that lead to dead ends")
        print("5. Pruning: Eliminate impossible branches early")
        print()
        print("Real-world Analogy: Maze Solving")
        print("- Walk through maze, mark path")
        print("- When you hit dead end, backtrack to last decision point")
        print("- Try different path from that point")
        print("- Continue until you find exit or exhaust all possibilities")
    
    def backtracking_vs_other_techniques(self) -> None:
        """Compare backtracking with other algorithmic techniques"""
        print("=== BACKTRACKING VS OTHER TECHNIQUES ===")
        print()
        print("Backtracking vs Brute Force:")
        print("  Brute Force: Generate all solutions, then check validity")
        print("  Backtracking: Check validity while building, prune early")
        print("  Example: Generate all permutations vs backtrack invalid ones")
        print()
        print("Backtracking vs Dynamic Programming:")
        print("  DP: Optimal substructure, overlapping subproblems")
        print("  Backtracking: Systematic enumeration, constraint satisfaction")
        print("  DP focuses on optimization, Backtracking on feasibility")
        print()
        print("Backtracking vs Greedy:")
        print("  Greedy: Make locally optimal choice at each step")
        print("  Backtracking: Explore all choices, backtrack if needed")
        print("  Greedy is faster but may miss optimal solutions")
    
    # ==========================================
    # 2. BACKTRACKING TEMPLATE
    # ==========================================
    
    def general_backtracking_template(self, problem_state, choices, constraints, goal_test):
        """
        General backtracking template - this is conceptual
        
        The actual implementation varies by problem, but this shows the pattern:
        
        def backtrack(state, path):
            if goal_reached(state):
                add_solution(path)
                return
            
            for choice in get_choices(state):
                if is_valid(choice, state):
                    make_choice(choice, state, path)
                    backtrack(new_state, path)
                    unmake_choice(choice, state, path)  # BACKTRACK
        """
        print("=== GENERAL BACKTRACKING TEMPLATE ===")
        print()
        print("def backtrack(current_state, partial_solution):")
        print("    # BASE CASE: Check if we have a complete solution")
        print("    if is_solution_complete(current_state):")
        print("        if is_valid_solution(partial_solution):")
        print("            solutions.append(copy(partial_solution))")
        print("        return")
        print("    ")
        print("    # RECURSIVE CASE: Try all possible choices")
        print("    for choice in get_available_choices(current_state):")
        print("        # CONSTRAINT CHECK: Is this choice valid?")
        print("        if is_valid_choice(choice, current_state):")
        print("            # MAKE CHOICE: Add choice to partial solution")
        print("            make_choice(choice, current_state, partial_solution)")
        print("            ")
        print("            # RECURSE: Explore with this choice")
        print("            backtrack(new_state, partial_solution)")
        print("            ")
        print("            # BACKTRACK: Undo the choice")
        print("            unmake_choice(choice, current_state, partial_solution)")
        print()
        print("Key Components:")
        print("1. State: Current problem state (what we know so far)")
        print("2. Choices: Available options at current state")
        print("3. Constraints: Rules that must be satisfied")
        print("4. Goal Test: Check if solution is complete")
        print("5. Backtrack: Undo choice and try next option")
    
    # ==========================================
    # 3. SIMPLE BACKTRACKING EXAMPLES
    # ==========================================
    
    def generate_binary_strings(self, n: int) -> List[str]:
        """
        Generate all binary strings of length n
        
        This is a simple example to demonstrate backtracking fundamentals
        
        Time: O(2^n), Space: O(n) for recursion depth
        """
        self.solutions = []
        self.call_count = 0
        
        def backtrack(current_string: str, remaining_length: int) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current_string)}Building: '{current_string}', remaining: {remaining_length}")
            
            # BASE CASE: String is complete
            if remaining_length == 0:
                self.solutions.append(current_string)
                print(f"{'  ' * len(current_string)}✓ Complete: '{current_string}'")
                return
            
            # RECURSIVE CASE: Try both '0' and '1'
            for digit in ['0', '1']:
                # MAKE CHOICE
                new_string = current_string + digit
                
                # RECURSE
                backtrack(new_string, remaining_length - 1)
                
                # BACKTRACK (automatic when function returns)
                print(f"{'  ' * len(current_string)}← Backtracking from '{new_string}'")
        
        print(f"Generating all binary strings of length {n}:")
        backtrack("", n)
        return self.solutions
    
    def find_all_paths_in_grid(self, rows: int, cols: int) -> List[List[Tuple[int, int]]]:
        """
        Find all paths from top-left to bottom-right in grid
        
        Can only move right or down
        Demonstrates path-finding with backtracking
        
        Time: O(2^(m+n)), Space: O(m+n)
        """
        self.solutions = []
        
        def backtrack(row: int, col: int, current_path: List[Tuple[int, int]]) -> None:
            print(f"{'  ' * len(current_path)}Visiting ({row}, {col}), path: {current_path}")
            
            # BASE CASE: Reached destination
            if row == rows - 1 and col == cols - 1:
                current_path.append((row, col))
                self.solutions.append(current_path[:])  # Make a copy
                print(f"{'  ' * len(current_path)}✓ Found path: {current_path}")
                current_path.pop()  # Backtrack
                return
            
            # CONSTRAINT CHECK: Within bounds
            if row >= rows or col >= cols:
                return
            
            # MAKE CHOICE: Add current position to path
            current_path.append((row, col))
            
            # RECURSIVE CASES: Try moving right and down
            # Move right
            if col + 1 < cols:
                backtrack(row, col + 1, current_path)
            
            # Move down
            if row + 1 < rows:
                backtrack(row + 1, col, current_path)
            
            # BACKTRACK: Remove current position
            current_path.pop()
            print(f"{'  ' * len(current_path)}← Backtracking from ({row}, {col})")
        
        print(f"Finding all paths in {rows}x{cols} grid:")
        backtrack(0, 0, [])
        return self.solutions
    
    def generate_valid_parentheses_simple(self, n: int) -> List[str]:
        """
        Generate all valid parentheses combinations
        
        Demonstrates constraint-based backtracking
        
        Time: O(4^n / sqrt(n)), Space: O(n)
        """
        self.solutions = []
        
        def backtrack(current: str, open_count: int, close_count: int) -> None:
            print(f"{'  ' * len(current)}Building: '{current}', open={open_count}, close={close_count}")
            
            # BASE CASE: Used all parentheses
            if len(current) == 2 * n:
                self.solutions.append(current)
                print(f"{'  ' * len(current)}✓ Valid: '{current}'")
                return
            
            # CHOICE 1: Add opening parenthesis
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)
            
            # CHOICE 2: Add closing parenthesis
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)
            
            # BACKTRACK: Automatic when function returns
            print(f"{'  ' * len(current)}← Backtracking from '{current}'")
        
        print(f"Generating valid parentheses for n={n}:")
        backtrack("", 0, 0)
        return self.solutions
    
    # ==========================================
    # 4. DECISION TREE VISUALIZATION
    # ==========================================
    
    def visualize_decision_tree(self, problem_type: str) -> None:
        """
        Visualize decision trees for different backtracking problems
        """
        print(f"=== DECISION TREE FOR {problem_type.upper()} ===")
        
        if problem_type == "binary_strings":
            print("Decision tree for binary strings (n=3):")
            print("                    ''")
            print("                 /      \\")
            print("               '0'      '1'")
            print("             /   \\    /   \\")
            print("          '00'  '01' '10' '11'")
            print("          / \\   / \\  / \\  / \\")
            print("       '000''001''010''011''100''101''110''111'")
            print()
            print("Each level represents a choice point")
            print("Leaves represent complete solutions")
            
        elif problem_type == "parentheses":
            print("Decision tree for parentheses (n=2):")
            print("                    ''")
            print("                    |")
            print("                   '('")
            print("                 /     \\")
            print("              '(('     '()'")
            print("               |        |")
            print("            '(()'     '()('")
            print("               |        |")
            print("           '(())'   '()()'")
            print()
            print("Invalid branches are pruned (e.g., starting with ')')")
            print("Constraints guide the search space pruning")
    
    def trace_backtracking_execution(self, example: str) -> None:
        """
        Trace through backtracking execution step by step
        """
        print(f"=== TRACING BACKTRACKING EXECUTION: {example.upper()} ===")
        
        if example == "simple":
            print("Problem: Find all 2-bit binary strings")
            print()
            print("Execution trace:")
            print("1. Start: backtrack('', 2)")
            print("2. Try '0': backtrack('0', 1)")
            print("3.   Try '0': backtrack('00', 0) → Solution: '00'")
            print("4.   Backtrack to '0'")
            print("5.   Try '1': backtrack('01', 0) → Solution: '01'")
            print("6.   Backtrack to ''")
            print("7. Try '1': backtrack('1', 1)")
            print("8.   Try '0': backtrack('10', 0) → Solution: '10'")
            print("9.   Backtrack to '1'")
            print("10.  Try '1': backtrack('11', 0) → Solution: '11'")
            print("11.  Backtrack to '' and finish")
            print()
            print("Key observations:")
            print("• Each recursive call makes a choice")
            print("• When base case reached, solution is recorded")
            print("• Function returns to try next choice")
            print("• All possibilities are systematically explored")
    
    # ==========================================
    # 5. BACKTRACKING PATTERNS AND VARIATIONS
    # ==========================================
    
    def demonstrate_backtracking_patterns(self) -> None:
        """
        Demonstrate common backtracking patterns
        """
        print("=== COMMON BACKTRACKING PATTERNS ===")
        print()
        
        print("1. CHOICE-BASED PATTERNS:")
        print("   • Include/Exclude Pattern (Subsets)")
        print("     - At each element: include it or exclude it")
        print("     - Example: Generate all subsets of [1,2,3]")
        print()
        print("   • Position-Based Pattern (Permutations)")
        print("     - At each position: try all available elements")
        print("     - Example: Arrange [1,2,3] in all possible orders")
        print()
        print("   • Value-Based Pattern (Assignments)")
        print("     - For each variable: try all possible values")
        print("     - Example: N-Queens, Sudoku")
        print()
        
        print("2. CONSTRAINT PATTERNS:")
        print("   • Global Constraints (affects entire solution)")
        print("     - Example: Sum must equal target")
        print("   • Local Constraints (affects current choice)")
        print("     - Example: No two queens attack each other")
        print("   • Progressive Constraints (build up gradually)")
        print("     - Example: Parentheses matching")
        print()
        
        print("3. SEARCH SPACE PATTERNS:")
        print("   • Tree-like Space (branching decisions)")
        print("     - Example: Decision trees, game trees")
        print("   • Grid-like Space (2D exploration)")
        print("     - Example: Maze solving, word search")
        print("   • Sequence Space (ordering problems)")
        print("     - Example: Permutations, combinations")
    
    # ==========================================
    # 6. PERFORMANCE ANALYSIS
    # ==========================================
    
    def analyze_backtracking_performance(self) -> None:
        """
        Analyze performance characteristics of backtracking
        """
        print("=== BACKTRACKING PERFORMANCE ANALYSIS ===")
        print()
        
        print("TIME COMPLEXITY:")
        print("• Worst case: Often exponential O(b^d)")
        print("  - b = branching factor (choices per level)")
        print("  - d = depth of search tree")
        print("• Examples:")
        print("  - Binary strings of length n: O(2^n)")
        print("  - Permutations of n elements: O(n!)")
        print("  - Subsets of n elements: O(2^n)")
        print("  - N-Queens: O(n!) but pruned significantly")
        print()
        
        print("SPACE COMPLEXITY:")
        print("• Recursion stack: O(depth)")
        print("• Solution storage: O(number of solutions)")
        print("• Working space: O(state representation)")
        print()
        
        print("OPTIMIZATION OPPORTUNITIES:")
        print("• Pruning: Eliminate invalid branches early")
        print("• Ordering: Try most promising choices first")
        print("• Memoization: Cache results if subproblems repeat")
        print("• Constraint propagation: Reduce search space")
        print("• Heuristics: Guide search toward solutions")
    
    def demonstrate_pruning_effectiveness(self, n: int = 4) -> None:
        """
        Demonstrate how pruning improves backtracking performance
        """
        print(f"=== PRUNING EFFECTIVENESS DEMO (n={n}) ===")
        
        # Generate all strings (no pruning)
        self.call_count = 0
        all_strings = []
        
        def generate_all_no_pruning(current: str, length: int) -> None:
            self.call_count += 1
            if length == 0:
                all_strings.append(current)
                return
            
            for char in ['(', ')']:
                generate_all_no_pruning(current + char, length - 1)
        
        generate_all_no_pruning("", 2 * n)
        no_pruning_calls = self.call_count
        
        # Generate valid parentheses (with pruning)
        self.call_count = 0
        valid_strings = []
        
        def generate_with_pruning(current: str, open_count: int, close_count: int) -> None:
            self.call_count += 1
            if len(current) == 2 * n:
                valid_strings.append(current)
                return
            
            # Pruning: only add '(' if we haven't used all n
            if open_count < n:
                generate_with_pruning(current + "(", open_count + 1, close_count)
            
            # Pruning: only add ')' if it doesn't exceed '('
            if close_count < open_count:
                generate_with_pruning(current + ")", open_count, close_count + 1)
        
        generate_with_pruning("", 0, 0)
        with_pruning_calls = self.call_count
        
        print(f"Results for parentheses generation (n={n}):")
        print(f"  Without pruning: {no_pruning_calls} function calls")
        print(f"  With pruning: {with_pruning_calls} function calls")
        print(f"  Pruning effectiveness: {no_pruning_calls / with_pruning_calls:.1f}x reduction")
        print(f"  Valid solutions found: {len(valid_strings)}")
        print(f"  Invalid solutions avoided: {len(all_strings) - len(valid_strings)}")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_backtracking_fundamentals():
    """Demonstrate all fundamental backtracking concepts"""
    print("=== BACKTRACKING FUNDAMENTALS DEMONSTRATION ===\n")
    
    bt = BacktrackingFundamentals()
    
    # 1. Concept explanation
    bt.explain_backtracking_concept()
    print("\n" + "="*60 + "\n")
    
    # 2. Comparison with other techniques
    bt.backtracking_vs_other_techniques()
    print("\n" + "="*60 + "\n")
    
    # 3. Template explanation
    bt.general_backtracking_template(None, None, None, None)
    print("\n" + "="*60 + "\n")
    
    # 4. Simple examples
    print("=== SIMPLE BACKTRACKING EXAMPLES ===")
    
    # Binary strings
    print("1. Binary Strings (n=3):")
    binary_strings = bt.generate_binary_strings(3)
    print(f"Generated {len(binary_strings)} binary strings: {binary_strings}")
    print(f"Total function calls: {bt.call_count}")
    print()
    
    # Grid paths
    print("2. Grid Paths (2x3):")
    grid_paths = bt.find_all_paths_in_grid(2, 3)
    print(f"Found {len(grid_paths)} paths:")
    for i, path in enumerate(grid_paths):
        print(f"  Path {i+1}: {path}")
    print()
    
    # Valid parentheses
    print("3. Valid Parentheses (n=2):")
    parentheses = bt.generate_valid_parentheses_simple(2)
    print(f"Generated {len(parentheses)} valid combinations: {parentheses}")
    print()
    
    # 5. Decision tree visualization
    bt.visualize_decision_tree("binary_strings")
    print("\n" + "="*60 + "\n")
    
    bt.visualize_decision_tree("parentheses")
    print("\n" + "="*60 + "\n")
    
    # 6. Execution tracing
    bt.trace_backtracking_execution("simple")
    print("\n" + "="*60 + "\n")
    
    # 7. Patterns demonstration
    bt.demonstrate_backtracking_patterns()
    print("\n" + "="*60 + "\n")
    
    # 8. Performance analysis
    bt.analyze_backtracking_performance()
    print("\n" + "="*60 + "\n")
    
    # 9. Pruning effectiveness
    bt.demonstrate_pruning_effectiveness(3)

if __name__ == "__main__":
    demonstrate_backtracking_fundamentals()
    
    print("\n" + "="*60)
    print("=== BACKTRACKING MASTERY GUIDE ===")
    print("="*60)
    
    print("\n🎯 WHEN TO USE BACKTRACKING:")
    print("✅ Need to find ALL solutions to a problem")
    print("✅ Problem has a finite search space")
    print("✅ Can define clear constraints")
    print("✅ Partial solutions can be validated early")
    print("✅ Solution can be built incrementally")
    
    print("\n📋 BACKTRACKING CHECKLIST:")
    print("1. Define the state representation")
    print("2. Identify available choices at each state")
    print("3. Define constraints for validity checking")
    print("4. Determine when solution is complete")
    print("5. Implement make/unmake choice operations")
    print("6. Add pruning for optimization")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Constraint checking: Fail fast on invalid choices")
    print("• Choice ordering: Try most promising choices first")
    print("• Symmetry breaking: Avoid redundant solutions")
    print("• Memoization: Cache results if subproblems repeat")
    print("• Iterative deepening: For memory-constrained problems")
    
    print("\n🚨 COMMON PITFALLS:")
    print("• Forgetting to backtrack (unmake choices)")
    print("• Incorrect constraint checking")
    print("• Missing base cases")
    print("• Inefficient state representation")
    print("• Not leveraging pruning opportunities")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master the basic template")
    print("2. Practice simple enumeration problems")
    print("3. Learn constraint satisfaction problems")
    print("4. Study optimization and pruning techniques")
    print("5. Tackle complex combinatorial problems")
    
    print("\n📚 PROBLEM CATEGORIES TO PRACTICE:")
    print("• Permutations and combinations")
    print("• Subset generation and partitioning")
    print("• Constraint satisfaction (N-Queens, Sudoku)")
    print("• Path finding and maze solving")
    print("• Game theory and decision making")
    print("• String and sequence problems")
