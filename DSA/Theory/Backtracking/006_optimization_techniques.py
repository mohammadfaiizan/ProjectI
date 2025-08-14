"""
Backtracking Optimization Techniques and Advanced Strategies
===========================================================

Topics: Pruning strategies, heuristics, constraint propagation, search ordering
Companies: All tech companies value optimization skills
Difficulty: Hard
Time Complexity: Varies by optimization technique
Space Complexity: Often improved through optimization
"""

from typing import List, Set, Dict, Tuple, Optional, Callable, Any
import time
import heapq
from functools import wraps

class BacktrackingOptimization:
    
    def __init__(self):
        """Initialize with comprehensive performance tracking"""
        self.total_calls = 0
        self.pruned_branches = 0
        self.constraint_checks = 0
        self.cache_hits = 0
        self.heuristic_applications = 0
        self.search_time = 0
    
    # ==========================================
    # 1. PRUNING STRATEGIES
    # ==========================================
    
    def demonstrate_pruning_strategies(self) -> None:
        """
        Demonstrate various pruning strategies in backtracking
        """
        print("=== BACKTRACKING PRUNING STRATEGIES ===")
        print()
        
        print("1. CONSTRAINT-BASED PRUNING:")
        print("   • Early constraint checking")
        print("   • Fail-fast on constraint violations")
        print("   • Propagate constraints to reduce search space")
        print("   • Example: N-Queens diagonal checking")
        print()
        
        print("2. BOUND-BASED PRUNING:")
        print("   • Upper/lower bound estimation")
        print("   • Prune if bound cannot improve current best")
        print("   • Example: Branch and bound for optimization")
        print()
        
        print("3. SYMMETRY BREAKING:")
        print("   • Eliminate symmetric solutions")
        print("   • Reduce search space by avoiding redundant paths")
        print("   • Example: Place first queen in first half of board")
        print()
        
        print("4. DOMINANCE PRUNING:")
        print("   • Eliminate dominated choices")
        print("   • If choice A is always better than B, skip B")
        print("   • Example: In knapsack, prefer higher value/weight ratio")
    
    def subset_sum_with_pruning(self, numbers: List[int], target: int) -> List[List[int]]:
        """
        Find all subsets that sum to target with aggressive pruning
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(2^n) worst case, much better with pruning
        """
        numbers.sort(reverse=True)  # Sort descending for better pruning
        solutions = []
        
        def backtrack(index: int, current_subset: List[int], current_sum: int, remaining_sum: int) -> None:
            self.total_calls += 1
            
            print(f"{'  ' * len(current_subset)}Index {index}, sum={current_sum}, target={target}, remaining={remaining_sum}")
            
            # BASE CASE: Found target sum
            if current_sum == target:
                solutions.append(current_subset[:])
                print(f"{'  ' * len(current_subset)}✓ Found solution: {current_subset}")
                return
            
            # PRUNING 1: Exceeded target
            if current_sum > target:
                self.pruned_branches += 1
                print(f"{'  ' * len(current_subset)}✗ Pruned: sum {current_sum} > target {target}")
                return
            
            # PRUNING 2: No more numbers
            if index >= len(numbers):
                self.pruned_branches += 1
                return
            
            # PRUNING 3: Remaining sum too small
            if current_sum + remaining_sum < target:
                self.pruned_branches += 1
                print(f"{'  ' * len(current_subset)}✗ Pruned: max possible {current_sum + remaining_sum} < target {target}")
                return
            
            # PRUNING 4: Current number alone exceeds remaining needed
            if numbers[index] > target - current_sum:
                # Skip this number and continue
                new_remaining = remaining_sum - numbers[index]
                backtrack(index + 1, current_subset, current_sum, new_remaining)
                return
            
            # CHOICE 1: Include current number
            current_subset.append(numbers[index])
            new_remaining = remaining_sum - numbers[index]
            backtrack(index + 1, current_subset, current_sum + numbers[index], new_remaining)
            current_subset.pop()
            
            # CHOICE 2: Exclude current number
            backtrack(index + 1, current_subset, current_sum, new_remaining)
        
        print(f"Finding subsets that sum to {target} from {numbers}:")
        self.total_calls = 0
        self.pruned_branches = 0
        
        total_sum = sum(numbers)
        backtrack(0, [], 0, total_sum)
        
        print(f"\nPruning Statistics:")
        print(f"  Total function calls: {self.total_calls}")
        print(f"  Pruned branches: {self.pruned_branches}")
        print(f"  Pruning efficiency: {self.pruned_branches / self.total_calls * 100:.1f}%")
        
        return solutions
    
    def n_queens_optimized_pruning(self, n: int) -> int:
        """
        N-Queens with multiple pruning strategies
        
        Demonstrates various optimization techniques
        """
        solutions_count = 0
        
        # Use sets for O(1) conflict checking
        cols = set()
        diag1 = set()  # r - c
        diag2 = set()  # r + c
        
        def backtrack(row: int) -> None:
            nonlocal solutions_count
            self.total_calls += 1
            
            # BASE CASE: All queens placed
            if row == n:
                solutions_count += 1
                return
            
            # SYMMETRY BREAKING: For first row, only try first half
            start_col = 0
            end_col = n
            if row == 0:
                end_col = (n + 1) // 2  # Only try first half for symmetry
            
            for col in range(start_col, end_col):
                # CONSTRAINT CHECKING: Fast conflict detection
                if col in cols or (row - col) in diag1 or (row + col) in diag2:
                    self.pruned_branches += 1
                    continue
                
                # MAKE CHOICE: Place queen
                cols.add(col)
                diag1.add(row - col)
                diag2.add(row + col)
                
                # RECURSE
                backtrack(row + 1)
                
                # BACKTRACK: Remove queen
                cols.remove(col)
                diag1.remove(row - col)
                diag2.remove(row + col)
            
            # SYMMETRY BREAKING: Double count for first half (except middle column)
            if row == 0 and n % 2 == 1:
                # Handle middle column separately for odd n
                col = n // 2
                if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                    cols.add(col)
                    diag1.add(row - col)
                    diag2.add(row + col)
                    
                    backtrack(row + 1)
                    
                    cols.remove(col)
                    diag1.remove(row - col)
                    diag2.remove(row + col)
        
        print(f"Solving {n}-Queens with optimizations:")
        self.total_calls = 0
        self.pruned_branches = 0
        
        backtrack(0)
        
        # Apply symmetry multiplication
        final_count = solutions_count * 2
        if n % 2 == 1:
            # Subtract over-counted middle column solutions
            # This is simplified; exact calculation is more complex
            pass
        
        print(f"Optimization Statistics:")
        print(f"  Function calls: {self.total_calls}")
        print(f"  Pruned branches: {self.pruned_branches}")
        print(f"  Solutions found: {solutions_count}")
        
        return solutions_count
    
    # ==========================================
    # 2. HEURISTIC ORDERING
    # ==========================================
    
    def demonstrate_heuristic_ordering(self) -> None:
        """
        Demonstrate various heuristic ordering strategies
        """
        print("=== HEURISTIC ORDERING STRATEGIES ===")
        print()
        
        print("1. MOST CONSTRAINED VARIABLE (MCV):")
        print("   • Choose variable with fewest remaining legal values")
        print("   • Fails fast if no solution exists")
        print("   • Example: In Sudoku, fill cells with fewest possibilities first")
        print()
        
        print("2. LEAST CONSTRAINING VALUE (LCV):")
        print("   • Choose value that eliminates fewest options for other variables")
        print("   • Preserves maximum flexibility")
        print("   • Example: Choose color that conflicts with fewest neighbors")
        print()
        
        print("3. DEGREE HEURISTIC:")
        print("   • Choose variable involved in most constraints")
        print("   • Break ties in MCV")
        print("   • Example: In graph coloring, choose node with most connections")
        print()
        
        print("4. FAIL-FIRST PRINCIPLE:")
        print("   • Try to fail as early as possible")
        print("   • Detect infeasibility quickly")
        print("   • Reduces overall search time")
    
    def sudoku_with_heuristics(self, board: List[List[str]]) -> bool:
        """
        Solve Sudoku using MCV and LCV heuristics
        
        Company: Google, Microsoft
        Difficulty: Hard
        Time: Significantly better than naive approach
        """
        def get_possible_values(board: List[List[str]], row: int, col: int) -> Set[str]:
            """Get possible values for cell (row, col)"""
            if board[row][col] != '.':
                return set()
            
            used = set()
            
            # Check row
            for c in range(9):
                if board[row][c] != '.':
                    used.add(board[row][c])
            
            # Check column
            for r in range(9):
                if board[r][col] != '.':
                    used.add(board[r][col])
            
            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    if board[r][c] != '.':
                        used.add(board[r][c])
            
            return set('123456789') - used
        
        def choose_cell_mcv(board: List[List[str]]) -> Optional[Tuple[int, int]]:
            """Choose cell with minimum remaining values (MCV heuristic)"""
            min_choices = 10
            best_cell = None
            
            for r in range(9):
                for c in range(9):
                    if board[r][c] == '.':
                        possible = get_possible_values(board, r, c)
                        if len(possible) < min_choices:
                            min_choices = len(possible)
                            best_cell = (r, c)
                            
                            # If no choices, fail immediately
                            if min_choices == 0:
                                return best_cell
            
            return best_cell
        
        def order_values_lcv(board: List[List[str]], row: int, col: int, values: Set[str]) -> List[str]:
            """Order values using Least Constraining Value heuristic"""
            value_scores = []
            
            for value in values:
                # Count how many cells this value would constrain
                constraints = 0
                
                # Check affected cells in same row
                for c in range(9):
                    if c != col and board[row][c] == '.':
                        possible = get_possible_values(board, row, c)
                        if value in possible:
                            constraints += 1
                
                # Check affected cells in same column
                for r in range(9):
                    if r != row and board[r][col] == '.':
                        possible = get_possible_values(board, r, col)
                        if value in possible:
                            constraints += 1
                
                # Check affected cells in same box
                box_row, box_col = 3 * (row // 3), 3 * (col // 3)
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        if (r != row or c != col) and board[r][c] == '.':
                            possible = get_possible_values(board, r, c)
                            if value in possible:
                                constraints += 1
                
                value_scores.append((constraints, value))
            
            # Sort by least constraining (fewest constraints first)
            value_scores.sort()
            return [value for _, value in value_scores]
        
        def backtrack() -> bool:
            self.total_calls += 1
            
            # MCV: Choose cell with fewest possibilities
            cell = choose_cell_mcv(board)
            if not cell:
                return True  # Puzzle solved
            
            row, col = cell
            possible_values = get_possible_values(board, row, col)
            
            # If no possible values, this path fails
            if not possible_values:
                self.pruned_branches += 1
                return False
            
            # LCV: Order values by least constraining
            ordered_values = order_values_lcv(board, row, col, possible_values)
            
            print(f"{'  ' * self.total_calls}Trying cell ({row},{col}) with values {ordered_values}")
            
            for value in ordered_values:
                self.heuristic_applications += 1
                
                # MAKE CHOICE
                board[row][col] = value
                
                # RECURSE
                if backtrack():
                    return True
                
                # BACKTRACK
                board[row][col] = '.'
            
            return False
        
        print("Solving Sudoku with MCV and LCV heuristics:")
        self.total_calls = 0
        self.pruned_branches = 0
        self.heuristic_applications = 0
        
        solved = backtrack()
        
        print(f"Heuristic Statistics:")
        print(f"  Function calls: {self.total_calls}")
        print(f"  Pruned branches: {self.pruned_branches}")
        print(f"  Heuristic applications: {self.heuristic_applications}")
        print(f"  Solved: {solved}")
        
        return solved
    
    # ==========================================
    # 3. CONSTRAINT PROPAGATION
    # ==========================================
    
    def demonstrate_constraint_propagation(self) -> None:
        """
        Demonstrate constraint propagation techniques
        """
        print("=== CONSTRAINT PROPAGATION TECHNIQUES ===")
        print()
        
        print("1. FORWARD CHECKING:")
        print("   • When variable assigned, remove inconsistent values from neighbors")
        print("   • Detect conflicts early")
        print("   • Example: In N-Queens, mark attacked squares as unavailable")
        print()
        
        print("2. ARC CONSISTENCY (AC-3):")
        print("   • Ensure every value in domain has supporting value in related variable")
        print("   • Iteratively remove inconsistent values")
        print("   • More powerful than forward checking")
        print()
        
        print("3. MAINTAINING ARC CONSISTENCY (MAC):")
        print("   • Apply arc consistency after each variable assignment")
        print("   • Combines search with constraint propagation")
        print("   • Often dramatically reduces search space")
        print()
        
        print("4. CONSTRAINT PROPAGATION ALGORITHMS:")
        print("   • AC-1, AC-3, AC-4: Different arc consistency algorithms")
        print("   • PC-2: Path consistency")
        print("   • Bucket elimination: Variable elimination ordering")
    
    def sudoku_with_constraint_propagation(self, board: List[List[str]]) -> bool:
        """
        Sudoku solver with constraint propagation
        
        Uses forward checking and constraint propagation
        """
        # Initialize domains for each cell
        domains = {}
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    domains[(r, c)] = set('123456789')
                else:
                    domains[(r, c)] = {board[r][c]}
        
        def propagate_constraints(row: int, col: int, value: str) -> bool:
            """Propagate constraints when (row, col) is assigned value"""
            # Remove value from same row
            for c in range(9):
                if c != col and value in domains.get((row, c), set()):
                    domains[(row, c)].remove(value)
                    if not domains[(row, c)]:
                        return False  # Domain became empty
            
            # Remove value from same column
            for r in range(9):
                if r != row and value in domains.get((r, col), set()):
                    domains[(r, col)].remove(value)
                    if not domains[(r, col)]:
                        return False
            
            # Remove value from same 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    if (r != row or c != col) and value in domains.get((r, c), set()):
                        domains[(r, c)].remove(value)
                        if not domains[(r, c)]:
                            return False
            
            return True
        
        def choose_variable() -> Optional[Tuple[int, int]]:
            """Choose unassigned variable with smallest domain"""
            min_domain_size = 10
            best_var = None
            
            for r in range(9):
                for c in range(9):
                    if board[r][c] == '.' and len(domains[(r, c)]) < min_domain_size:
                        min_domain_size = len(domains[(r, c)])
                        best_var = (r, c)
            
            return best_var
        
        def backtrack() -> bool:
            self.total_calls += 1
            
            # Choose variable with smallest domain
            var = choose_variable()
            if not var:
                return True  # All variables assigned
            
            row, col = var
            
            # Try each value in domain
            for value in list(domains[(row, col)]):  # Copy to avoid modification during iteration
                print(f"{'  ' * self.total_calls}Trying ({row},{col}) = {value}")
                
                # Save current domains
                old_domains = {k: v.copy() for k, v in domains.items()}
                
                # Assign value
                board[row][col] = value
                domains[(row, col)] = {value}
                
                # Propagate constraints
                if propagate_constraints(row, col, value):
                    # Recursively solve
                    if backtrack():
                        return True
                
                # Restore domains and board
                domains.clear()
                domains.update(old_domains)
                board[row][col] = '.'
            
            return False
        
        # Initial constraint propagation
        print("Applying initial constraint propagation...")
        for r in range(9):
            for c in range(9):
                if board[r][c] != '.':
                    if not propagate_constraints(r, c, board[r][c]):
                        return False
        
        print("Solving with constraint propagation:")
        self.total_calls = 0
        
        return backtrack()
    
    # ==========================================
    # 4. MEMOIZATION AND CACHING
    # ==========================================
    
    def memoized_backtracking_decorator(self):
        """
        Decorator for memoizing backtracking functions
        """
        def memoize(func):
            cache = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create hashable key from arguments
                key = str(args) + str(sorted(kwargs.items()))
                
                if key in cache:
                    self.cache_hits += 1
                    return cache[key]
                
                result = func(*args, **kwargs)
                cache[key] = result
                return result
            
            wrapper.cache = cache
            wrapper.cache_clear = lambda: cache.clear()
            return wrapper
        
        return memoize
    
    def word_break_memoized(self, s: str, word_dict: List[str]) -> bool:
        """
        Word Break with memoization
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n^2), Space: O(n)
        """
        memo = {}
        word_set = set(word_dict)
        
        def backtrack(start: int) -> bool:
            # Base case
            if start >= len(s):
                return True
            
            # Check memo
            if start in memo:
                self.cache_hits += 1
                return memo[start]
            
            self.total_calls += 1
            
            # Try all possible words starting at current position
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    if backtrack(end):
                        memo[start] = True
                        return True
            
            memo[start] = False
            return False
        
        print(f"Word break for '{s}' with dictionary {word_dict}:")
        self.total_calls = 0
        self.cache_hits = 0
        
        result = backtrack(0)
        
        print(f"Memoization Statistics:")
        print(f"  Function calls: {self.total_calls}")
        print(f"  Cache hits: {self.cache_hits}")
        print(f"  Cache efficiency: {self.cache_hits / (self.total_calls + self.cache_hits) * 100:.1f}%")
        
        return result
    
    # ==========================================
    # 5. PERFORMANCE COMPARISON
    # ==========================================
    
    def compare_optimization_techniques(self) -> None:
        """
        Compare different optimization techniques on same problem
        """
        print("=== OPTIMIZATION TECHNIQUES COMPARISON ===")
        
        # Test problem: subset sum
        numbers = [3, 34, 4, 12, 5, 2, 8, 15]
        target = 20
        
        print(f"Problem: Find subsets of {numbers} that sum to {target}")
        print()
        
        # 1. Naive backtracking
        print("1. Naive Backtracking:")
        start_time = time.time()
        solutions1 = self.subset_sum_naive(numbers, target)
        time1 = time.time() - start_time
        print(f"   Time: {time1:.6f}s")
        print(f"   Solutions: {len(solutions1)}")
        print()
        
        # 2. With pruning
        print("2. With Pruning:")
        start_time = time.time()
        solutions2 = self.subset_sum_with_pruning(numbers, target)
        time2 = time.time() - start_time
        print(f"   Time: {time2:.6f}s")
        print(f"   Solutions: {len(solutions2)}")
        if time1 > 0:
            print(f"   Speedup: {time1/time2:.1f}x")
        print()
        
        # 3. Summary
        print("Optimization Impact:")
        print(f"   Pruning reduced calls by {self.pruned_branches} branches")
        print(f"   Performance improvement: {time1/time2:.1f}x" if time1 > 0 and time2 > 0 else "")
    
    def subset_sum_naive(self, numbers: List[int], target: int) -> List[List[int]]:
        """Naive subset sum without optimizations"""
        solutions = []
        
        def backtrack(index: int, current_subset: List[int], current_sum: int) -> None:
            # Base case: processed all numbers
            if index >= len(numbers):
                if current_sum == target:
                    solutions.append(current_subset[:])
                return
            
            # Include current number
            current_subset.append(numbers[index])
            backtrack(index + 1, current_subset, current_sum + numbers[index])
            current_subset.pop()
            
            # Exclude current number
            backtrack(index + 1, current_subset, current_sum)
        
        backtrack(0, [], 0)
        return solutions

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_optimization_techniques():
    """Demonstrate all backtracking optimization techniques"""
    print("=== BACKTRACKING OPTIMIZATION TECHNIQUES DEMONSTRATION ===\n")
    
    opt = BacktrackingOptimization()
    
    # 1. Pruning strategies
    opt.demonstrate_pruning_strategies()
    print("\n" + "="*60 + "\n")
    
    # 2. Subset sum with pruning
    print("=== SUBSET SUM WITH PRUNING ===")
    solutions = opt.subset_sum_with_pruning([3, 34, 4, 12, 5, 2], 9)
    print(f"Found {len(solutions)} solutions: {solutions}")
    print("\n" + "="*60 + "\n")
    
    # 3. N-Queens optimization
    print("=== N-QUEENS OPTIMIZATION ===")
    for n in [4, 6, 8]:
        count = opt.n_queens_optimized_pruning(n)
        print(f"{n}-Queens: {count} solutions")
        print()
    print("="*60 + "\n")
    
    # 4. Heuristic ordering
    opt.demonstrate_heuristic_ordering()
    print("\n" + "="*60 + "\n")
    
    # 5. Sudoku with heuristics
    print("=== SUDOKU WITH HEURISTICS ===")
    sudoku_puzzle = [
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
    
    puzzle_copy = [row[:] for row in sudoku_puzzle]
    solved = opt.sudoku_with_heuristics(puzzle_copy)
    print()
    print("="*60 + "\n")
    
    # 6. Constraint propagation
    opt.demonstrate_constraint_propagation()
    print("\n" + "="*60 + "\n")
    
    # 7. Memoization example
    print("=== MEMOIZATION EXAMPLE ===")
    word_break_result = opt.word_break_memoized("leetcode", ["leet", "code"])
    print(f"Can break 'leetcode': {word_break_result}")
    print("\n" + "="*60 + "\n")
    
    # 8. Performance comparison
    opt.compare_optimization_techniques()

if __name__ == "__main__":
    demonstrate_optimization_techniques()
    
    print("\n=== OPTIMIZATION MASTERY GUIDE ===")
    
    print("\n🎯 OPTIMIZATION PRINCIPLES:")
    print("• Fail Fast: Detect impossible paths as early as possible")
    print("• Prune Aggressively: Eliminate branches that cannot lead to solutions")
    print("• Order Smartly: Try most promising choices first")
    print("• Cache Results: Avoid recomputing same subproblems")
    print("• Propagate Constraints: Reduce search space through inference")
    
    print("\n📋 OPTIMIZATION CHECKLIST:")
    print("1. ✅ Constraint checking: Are constraints checked as early as possible?")
    print("2. ✅ Pruning conditions: Can we eliminate branches based on bounds?")
    print("3. ✅ Variable ordering: Are we choosing variables optimally?")
    print("4. ✅ Value ordering: Are we trying values in best order?")
    print("5. ✅ Symmetry breaking: Can we avoid redundant explorations?")
    print("6. ✅ Memoization: Are there overlapping subproblems to cache?")
    
    print("\n⚡ PERFORMANCE TECHNIQUES:")
    print("• Bit manipulation: Use bitmasks for set operations")
    print("• Data structure optimization: Use appropriate data structures")
    print("• Precomputation: Calculate invariants once")
    print("• Iterative deepening: For memory-constrained problems")
    print("• Parallel processing: Explore branches in parallel")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Worst case often remains exponential")
    print("• Average case can improve dramatically with optimization")
    print("• Space complexity may increase with memoization")
    print("• Trade-off between time and space optimization")
    
    print("\n🏆 OPTIMIZATION IMPACT:")
    print("• Pruning: Can reduce branches by 90%+ in good cases")
    print("• Heuristics: Often reduce search time by orders of magnitude")
    print("• Constraint propagation: Dramatic reduction in CSP problems")
    print("• Memoization: Converts exponential to polynomial for some problems")
    print("• Combined techniques: Multiplicative benefits when combined properly")
