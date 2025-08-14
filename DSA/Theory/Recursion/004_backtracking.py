"""
Backtracking with Recursion
===========================

Topics: N-Queens, permutations, combinations, subset generation, maze solving
Companies: Google, Amazon, Microsoft, Facebook, Apple, Uber
Difficulty: Medium to Hard
Time Complexity: Often exponential O(2^n), O(n!)
Space Complexity: O(depth) for recursion stack
"""

from typing import List, Set, Tuple, Optional
import copy

class Backtracking:
    
    def __init__(self):
        """Initialize with solution tracking"""
        self.solutions = []
        self.call_count = 0
    
    # ==========================================
    # 1. BACKTRACKING FUNDAMENTALS
    # ==========================================
    
    def backtracking_template(self, problem_state, choices, constraints, goal_test):
        """
        General backtracking template
        
        Backtracking Pattern:
        1. Check if current state satisfies goal
        2. If yes, add to solutions
        3. For each possible choice:
           a. Make the choice (modify state)
           b. Check if choice is valid (constraints)
           c. If valid, recurse with new state
           d. Unmake the choice (backtrack)
        
        This is a conceptual template - actual implementation varies by problem
        """
        print("=== BACKTRACKING TEMPLATE ===")
        print("def backtrack(state, path):")
        print("    if goal_reached(state):")
        print("        solutions.append(path[:])")
        print("        return")
        print("    ")
        print("    for choice in get_choices(state):")
        print("        if is_valid(choice, state):")
        print("            make_choice(choice, state, path)")
        print("            backtrack(state, path)")
        print("            unmake_choice(choice, state, path)  # BACKTRACK")
    
    # ==========================================
    # 2. GENERATE ALL PERMUTATIONS
    # ==========================================
    
    def generate_permutations(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all permutations of given numbers
        
        Time: O(n! * n), Space: O(n! * n)
        """
        self.solutions = []
        self.call_count = 0
        
        def backtrack(current_permutation: List[int], remaining: List[int]) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current_permutation)}Trying: {current_permutation}, Remaining: {remaining}")
            
            # Base case: no more numbers to add
            if not remaining:
                self.solutions.append(current_permutation[:])  # Make a copy
                print(f"{'  ' * len(current_permutation)}✓ Found permutation: {current_permutation}")
                return
            
            # Try each remaining number
            for i, num in enumerate(remaining):
                # Make choice
                current_permutation.append(num)
                new_remaining = remaining[:i] + remaining[i+1:]
                
                # Recurse
                backtrack(current_permutation, new_remaining)
                
                # Backtrack (unmake choice)
                current_permutation.pop()
                print(f"{'  ' * len(current_permutation)}← Backtracking from {current_permutation + [num]}")
        
        backtrack([], nums)
        return self.solutions
    
    def generate_permutations_optimized(self, nums: List[int]) -> List[List[int]]:
        """
        Optimized permutation generation using swapping
        
        Time: O(n!), Space: O(n!) for output + O(n) for recursion
        """
        result = []
        
        def backtrack(start: int) -> None:
            # Base case: complete permutation
            if start == len(nums):
                result.append(nums[:])  # Make a copy
                return
            
            # Try each position from start to end
            for i in range(start, len(nums)):
                # Swap current element to start position
                nums[start], nums[i] = nums[i], nums[start]
                
                # Recurse for remaining positions
                backtrack(start + 1)
                
                # Backtrack: restore original order
                nums[start], nums[i] = nums[i], nums[start]
        
        backtrack(0)
        return result
    
    # ==========================================
    # 3. GENERATE ALL COMBINATIONS
    # ==========================================
    
    def generate_combinations(self, n: int, k: int) -> List[List[int]]:
        """
        Generate all combinations of k numbers from 1 to n
        
        Time: O(C(n,k) * k), Space: O(C(n,k) * k)
        """
        result = []
        
        def backtrack(start: int, current_combination: List[int]) -> None:
            print(f"{'  ' * len(current_combination)}Current: {current_combination}, Start: {start}")
            
            # Base case: combination is complete
            if len(current_combination) == k:
                result.append(current_combination[:])
                print(f"{'  ' * len(current_combination)}✓ Found combination: {current_combination}")
                return
            
            # Try numbers from start to n
            for i in range(start, n + 1):
                # Make choice
                current_combination.append(i)
                
                # Recurse with next start position
                backtrack(i + 1, current_combination)
                
                # Backtrack
                current_combination.pop()
                print(f"{'  ' * len(current_combination)}← Backtracking from {current_combination + [i]}")
        
        backtrack(1, [])
        return result
    
    def generate_subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Generate all subsets (power set) of given array
        
        Time: O(2^n * n), Space: O(2^n * n)
        """
        result = []
        
        def backtrack(start: int, current_subset: List[int]) -> None:
            # Add current subset to result (every state is a valid subset)
            result.append(current_subset[:])
            print(f"{'  ' * len(current_subset)}Added subset: {current_subset}")
            
            # Try adding each remaining element
            for i in range(start, len(nums)):
                # Include nums[i]
                current_subset.append(nums[i])
                backtrack(i + 1, current_subset)
                
                # Exclude nums[i] (backtrack)
                current_subset.pop()
        
        backtrack(0, [])
        return result
    
    # ==========================================
    # 4. N-QUEENS PROBLEM
    # ==========================================
    
    def solve_n_queens(self, n: int) -> List[List[str]]:
        """
        Solve N-Queens problem using backtracking
        
        Place n queens on n×n chessboard such that no two queens attack each other
        
        Time: O(n!), Space: O(n)
        """
        solutions = []
        board = [-1] * n  # board[i] = column position of queen in row i
        
        def is_safe(row: int, col: int) -> bool:
            """Check if queen can be placed at (row, col)"""
            for i in range(row):
                # Check column conflict
                if board[i] == col:
                    return False
                
                # Check diagonal conflicts
                if abs(board[i] - col) == abs(i - row):
                    return False
            
            return True
        
        def backtrack(row: int) -> None:
            print(f"{'  ' * row}Placing queen in row {row}")
            
            # Base case: all queens placed
            if row == n:
                # Convert board representation to string format
                solution = []
                for i in range(n):
                    row_str = ['.'] * n
                    row_str[board[i]] = 'Q'
                    solution.append(''.join(row_str))
                solutions.append(solution)
                print(f"{'  ' * row}✓ Found solution: {board}")
                return
            
            # Try placing queen in each column of current row
            for col in range(n):
                if is_safe(row, col):
                    print(f"{'  ' * row}Trying queen at ({row}, {col})")
                    
                    # Place queen
                    board[row] = col
                    
                    # Recurse to next row
                    backtrack(row + 1)
                    
                    # Backtrack
                    board[row] = -1
                    print(f"{'  ' * row}← Backtracking from ({row}, {col})")
                else:
                    print(f"{'  ' * row}✗ Cannot place queen at ({row}, {col}) - conflicts")
        
        backtrack(0)
        return solutions
    
    def print_n_queens_solution(self, solution: List[str]) -> None:
        """Pretty print N-Queens solution"""
        print("N-Queens Solution:")
        for row in solution:
            print(f"  {row}")
        print()
    
    # ==========================================
    # 5. SUDOKU SOLVER
    # ==========================================
    
    def solve_sudoku(self, board: List[List[str]]) -> bool:
        """
        Solve Sudoku puzzle using backtracking
        
        Time: O(9^(n*n)), Space: O(n*n) where n=9
        """
        def is_valid(board: List[List[str]], row: int, col: int, num: str) -> bool:
            """Check if placing num at (row, col) is valid"""
            
            # Check row
            for j in range(9):
                if board[row][j] == num:
                    return False
            
            # Check column
            for i in range(9):
                if board[i][col] == num:
                    return False
            
            # Check 3x3 box
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
            # Find next empty cell
            empty_cell = find_empty_cell(board)
            if not empty_cell:
                return True  # Puzzle solved
            
            row, col = empty_cell
            
            # Try numbers 1-9
            for num in '123456789':
                if is_valid(board, row, col, num):
                    # Make choice
                    board[row][col] = num
                    
                    # Recurse
                    if backtrack():
                        return True
                    
                    # Backtrack
                    board[row][col] = '.'
            
            return False  # No valid number found
        
        return backtrack()
    
    # ==========================================
    # 6. MAZE SOLVING
    # ==========================================
    
    def solve_maze(self, maze: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find path through maze using backtracking
        
        0 = open path, 1 = wall
        Returns path from start to end, or empty list if no path
        
        Time: O(4^(m*n)), Space: O(m*n)
        """
        rows, cols = len(maze), len(maze[0])
        path = []
        visited = set()
        
        def is_valid_move(row: int, col: int) -> bool:
            """Check if move to (row, col) is valid"""
            return (0 <= row < rows and 
                   0 <= col < cols and 
                   maze[row][col] == 0 and 
                   (row, col) not in visited)
        
        def backtrack(row: int, col: int) -> bool:
            print(f"{'  ' * len(path)}Visiting ({row}, {col})")
            
            # Base case: reached destination
            if (row, col) == end:
                path.append((row, col))
                print(f"{'  ' * len(path)}✓ Reached destination!")
                return True
            
            # Mark current cell as visited
            visited.add((row, col))
            path.append((row, col))
            
            # Try all four directions: up, right, down, left
            directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if is_valid_move(new_row, new_col):
                    if backtrack(new_row, new_col):
                        return True
            
            # Backtrack: remove current cell from path and visited set
            path.pop()
            visited.remove((row, col))
            print(f"{'  ' * len(path)}← Backtracking from ({row}, {col})")
            
            return False
        
        start_row, start_col = start
        if backtrack(start_row, start_col):
            return path
        else:
            return []  # No path found
    
    def print_maze_with_path(self, maze: List[List[int]], path: List[Tuple[int, int]]) -> None:
        """Print maze with path marked"""
        path_set = set(path)
        
        print("Maze with path (. = path, # = wall, space = open):")
        for i, row in enumerate(maze):
            for j, cell in enumerate(row):
                if (i, j) in path_set:
                    print('.', end=' ')
                elif cell == 1:
                    print('#', end=' ')
                else:
                    print(' ', end=' ')
            print()
    
    # ==========================================
    # 7. WORD SEARCH
    # ==========================================
    
    def word_search(self, board: List[List[str]], word: str) -> bool:
        """
        Search for word in 2D board using backtracking
        
        Word can be constructed from letters of adjacent cells (horizontal/vertical)
        Same cell cannot be used more than once
        
        Time: O(m*n*4^L), Space: O(L) where L is word length
        """
        rows, cols = len(board), len(board[0])
        
        def backtrack(row: int, col: int, index: int, path: List[Tuple[int, int]]) -> bool:
            print(f"{'  ' * index}Checking ({row}, {col}) for '{word[index]}', path: {path}")
            
            # Base case: found complete word
            if index == len(word):
                print(f"{'  ' * index}✓ Found word: {word}")
                return True
            
            # Check bounds and character match
            if (row < 0 or row >= rows or 
                col < 0 or col >= cols or 
                board[row][col] != word[index] or 
                (row, col) in path):
                return False
            
            # Add current cell to path
            path.append((row, col))
            
            # Try all four directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if backtrack(new_row, new_col, index + 1, path):
                    return True
            
            # Backtrack: remove current cell from path
            path.pop()
            print(f"{'  ' * index}← Backtracking from ({row}, {col})")
            
            return False
        
        # Try starting from each cell
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == word[0]:  # Potential starting point
                    print(f"\nTrying to start word search from ({i}, {j})")
                    if backtrack(i, j, 0, []):
                        return True
        
        return False

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_backtracking():
    """Demonstrate all backtracking algorithms"""
    print("=== BACKTRACKING DEMONSTRATION ===\n")
    
    bt = Backtracking()
    
    # Show template
    bt.backtracking_template(None, None, None, None)
    print()
    
    # 1. Permutations
    print("=== GENERATE PERMUTATIONS ===")
    nums = [1, 2, 3]
    print(f"Generating permutations of {nums}:")
    perms = bt.generate_permutations(nums)
    print(f"All permutations: {perms}")
    print(f"Total function calls: {bt.call_count}")
    print()
    
    # 2. Combinations
    print("=== GENERATE COMBINATIONS ===")
    print("Generating combinations C(4,2):")
    combs = bt.generate_combinations(4, 2)
    print(f"All combinations: {combs}")
    print()
    
    # 3. Subsets
    print("=== GENERATE SUBSETS ===")
    nums = [1, 2, 3]
    print(f"Generating subsets of {nums}:")
    subsets = bt.generate_subsets(nums)
    print(f"All subsets: {subsets}")
    print()
    
    # 4. N-Queens (4x4 board)
    print("=== N-QUEENS PROBLEM ===")
    print("Solving 4-Queens:")
    solutions = bt.solve_n_queens(4)
    print(f"Found {len(solutions)} solutions:")
    for i, solution in enumerate(solutions):
        print(f"Solution {i+1}:")
        bt.print_n_queens_solution(solution)
    
    # 5. Maze solving
    print("=== MAZE SOLVING ===")
    maze = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
    start = (0, 0)
    end = (4, 4)
    
    print("Solving maze:")
    path = bt.solve_maze(maze, start, end)
    
    if path:
        print(f"Path found: {path}")
        bt.print_maze_with_path(maze, path)
    else:
        print("No path found!")
    print()
    
    # 6. Word Search
    print("=== WORD SEARCH ===")
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    word = "ABCCED"
    
    print(f"Searching for word '{word}' in board:")
    for row in board:
        print(f"  {row}")
    
    found = bt.word_search(board, word)
    print(f"Word found: {found}")

if __name__ == "__main__":
    demonstrate_backtracking()
    
    print("\n=== BACKTRACKING PRINCIPLES ===")
    print("1. 🔄 Try all possibilities systematically")
    print("2. 🎯 Make choices and explore consequences")
    print("3. ↩️  Backtrack when dead end is reached")
    print("4. 🔍 Prune invalid branches early")
    
    print("\n=== BACKTRACKING TEMPLATE ===")
    print("✅ Choose: Make a choice and move forward")
    print("✅ Explore: Recursively explore with the choice")
    print("✅ Unchoose: Backtrack and try next possibility")
    
    print("\n=== WHEN TO USE BACKTRACKING ===")
    print("🎯 Finding all solutions to a problem")
    print("🎯 Constraint satisfaction problems")
    print("🎯 Combinatorial optimization")
    print("🎯 Puzzle solving (Sudoku, N-Queens)")
    print("🎯 Path finding with constraints")
    
    print("\n=== OPTIMIZATION TECHNIQUES ===")
    print("⚡ Early termination when constraints violated")
    print("⚡ Pruning branches that cannot lead to solution")
    print("⚡ Ordering choices by likelihood of success")
    print("⚡ Using heuristics to guide search")
    print("⚡ Memoization for overlapping subproblems")
