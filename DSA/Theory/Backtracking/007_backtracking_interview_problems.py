"""
Backtracking Interview Problems - Common Questions and Solutions
==============================================================

Topics: Most frequently asked backtracking problems in technical interviews
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber, LinkedIn
Difficulty: Medium to Hard
Time Complexity: Exponential but optimized with pruning
Space Complexity: O(depth) for recursion stack + solution storage
"""

from typing import List, Set, Dict, Tuple, Optional
import copy

class BacktrackingInterviewProblems:
    
    def __init__(self):
        """Initialize with solution tracking and interview performance metrics"""
        self.solutions = []
        self.call_count = 0
        self.time_complexity_analysis = {}
    
    # ==========================================
    # 1. CLASSIC INTERVIEW PROBLEMS
    # ==========================================
    
    def letter_combinations_phone_number(self, digits: str) -> List[str]:
        """
        Letter Combinations of a Phone Number
        
        Given string of digits 2-9, return all possible letter combinations
        that the number could represent (like old phone keypads)
        
        Company: Amazon, Facebook, Google, Microsoft
        Difficulty: Medium
        Time: O(4^n), Space: O(4^n)
        
        Example: digits = "23" → ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
        """
        if not digits:
            return []
        
        # Phone keypad mapping
        phone_map = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
        }
        
        result = []
        
        def backtrack(index: int, current_combination: str) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current_combination)}Building: '{current_combination}', digit index: {index}")
            
            # BASE CASE: All digits processed
            if index >= len(digits):
                result.append(current_combination)
                print(f"{'  ' * len(current_combination)}✓ Complete: '{current_combination}'")
                return
            
            # Get letters for current digit
            current_digit = digits[index]
            if current_digit in phone_map:
                letters = phone_map[current_digit]
                
                # TRY: Each letter for current digit
                for letter in letters:
                    backtrack(index + 1, current_combination + letter)
        
        print(f"Generating letter combinations for '{digits}':")
        self.call_count = 0
        backtrack(0, '')
        
        print(f"Generated {len(result)} combinations")
        print(f"Function calls: {self.call_count}")
        
        return result
    
    def generate_parentheses(self, n: int) -> List[str]:
        """
        Generate Parentheses
        
        Generate all combinations of well-formed parentheses for n pairs
        
        Company: Google, Facebook, Amazon, Uber
        Difficulty: Medium
        Time: O(4^n / √n) - Catalan number, Space: O(4^n / √n)
        
        Example: n = 3 → ["((()))", "(()())", "(())()", "()(())", "()()()"]
        """
        result = []
        
        def backtrack(current: str, open_count: int, close_count: int) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current)}Building: '{current}', open={open_count}, close={close_count}")
            
            # BASE CASE: Used all n pairs
            if len(current) == 2 * n:
                result.append(current)
                print(f"{'  ' * len(current)}✓ Valid: '{current}'")
                return
            
            # CHOICE 1: Add opening parenthesis (if we have remaining)
            if open_count < n:
                backtrack(current + "(", open_count + 1, close_count)
            
            # CHOICE 2: Add closing parenthesis (if it balances)
            if close_count < open_count:
                backtrack(current + ")", open_count, close_count + 1)
        
        print(f"Generating valid parentheses for n={n}:")
        self.call_count = 0
        backtrack("", 0, 0)
        
        print(f"Generated {len(result)} valid combinations")
        return result
    
    def combination_sum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Combination Sum
        
        Find all unique combinations where candidates sum to target
        Same number may be chosen unlimited times
        
        Company: Amazon, Facebook, Microsoft
        Difficulty: Medium
        Time: O(2^target), Space: O(target)
        
        Example: candidates = [2,3,6,7], target = 7 → [[2,2,3], [7]]
        """
        candidates.sort()  # Sort for optimization
        result = []
        
        def backtrack(start: int, current: List[int], remaining: int) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current)}Current: {current}, remaining: {remaining}, start: {start}")
            
            # BASE CASE: Found target sum
            if remaining == 0:
                result.append(current[:])
                print(f"{'  ' * len(current)}✓ Found combination: {current}")
                return
            
            # PRUNING: If remaining is negative, stop
            if remaining < 0:
                return
            
            # TRY: Each candidate from start index
            for i in range(start, len(candidates)):
                # PRUNING: If current candidate > remaining, break (sorted array)
                if candidates[i] > remaining:
                    break
                
                # MAKE CHOICE: Add candidate
                current.append(candidates[i])
                
                # RECURSE: Can reuse same candidate (start=i)
                backtrack(i, current, remaining - candidates[i])
                
                # BACKTRACK: Remove candidate
                current.pop()
        
        print(f"Finding combinations that sum to {target} from {candidates}:")
        self.call_count = 0
        backtrack(0, [], target)
        
        print(f"Found {len(result)} combinations")
        return result
    
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Subsets (Power Set)
        
        Generate all possible subsets of given array
        
        Company: Facebook, Google, Amazon
        Difficulty: Medium
        Time: O(2^n * n), Space: O(2^n * n)
        
        Example: [1,2,3] → [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
        """
        result = []
        
        def backtrack(start: int, current_subset: List[int]) -> None:
            # ADD: Current subset to result (every state is valid)
            result.append(current_subset[:])
            print(f"{'  ' * len(current_subset)}Added subset: {current_subset}")
            
            # TRY: Adding each remaining element
            for i in range(start, len(nums)):
                # MAKE CHOICE: Include nums[i]
                current_subset.append(nums[i])
                
                # RECURSE: Start from next index
                backtrack(i + 1, current_subset)
                
                # BACKTRACK: Exclude nums[i]
                current_subset.pop()
        
        print(f"Generating all subsets of {nums}:")
        backtrack(0, [])
        
        print(f"Generated {len(result)} subsets")
        return result
    
    def permutations(self, nums: List[int]) -> List[List[int]]:
        """
        Permutations
        
        Generate all possible permutations of given array
        
        Company: Google, Amazon, Microsoft
        Difficulty: Medium
        Time: O(n! * n), Space: O(n!)
        
        Example: [1,2,3] → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
        """
        result = []
        
        def backtrack(current_permutation: List[int]) -> None:
            self.call_count += 1
            
            print(f"{'  ' * len(current_permutation)}Building: {current_permutation}")
            
            # BASE CASE: Permutation is complete
            if len(current_permutation) == len(nums):
                result.append(current_permutation[:])
                print(f"{'  ' * len(current_permutation)}✓ Complete: {current_permutation}")
                return
            
            # TRY: Each unused number
            for num in nums:
                if num not in current_permutation:
                    # MAKE CHOICE: Add number
                    current_permutation.append(num)
                    
                    # RECURSE
                    backtrack(current_permutation)
                    
                    # BACKTRACK: Remove number
                    current_permutation.pop()
        
        print(f"Generating all permutations of {nums}:")
        self.call_count = 0
        backtrack([])
        
        print(f"Generated {len(result)} permutations")
        return result
    
    # ==========================================
    # 2. STRING AND ARRAY PROBLEMS
    # ==========================================
    
    def palindrome_partitioning(self, s: str) -> List[List[str]]:
        """
        Palindrome Partitioning
        
        Partition string such that every substring is a palindrome
        
        Company: Amazon, Microsoft, Google
        Difficulty: Medium
        Time: O(2^n * n), Space: O(n)
        
        Example: "aab" → [["a","a","b"], ["aa","b"]]
        """
        result = []
        
        def is_palindrome(string: str, start: int, end: int) -> bool:
            """Check if substring is palindrome"""
            while start < end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        def backtrack(start: int, current_partition: List[str]) -> None:
            print(f"{'  ' * len(current_partition)}Partitioning from index {start}: {current_partition}")
            
            # BASE CASE: Reached end of string
            if start >= len(s):
                result.append(current_partition[:])
                print(f"{'  ' * len(current_partition)}✓ Valid partition: {current_partition}")
                return
            
            # TRY: All possible substrings starting from start
            for end in range(start, len(s)):
                substring = s[start:end + 1]
                
                # CHECK: If current substring is palindrome
                if is_palindrome(s, start, end):
                    # MAKE CHOICE: Add to partition
                    current_partition.append(substring)
                    
                    # RECURSE: Continue from end + 1
                    backtrack(end + 1, current_partition)
                    
                    # BACKTRACK: Remove from partition
                    current_partition.pop()
        
        print(f"Finding palindromic partitions of '{s}':")
        backtrack(0, [])
        
        print(f"Found {len(result)} palindromic partitions")
        return result
    
    def word_search(self, board: List[List[str]], word: str) -> bool:
        """
        Word Search
        
        Find if word exists in 2D board of characters
        Word can be constructed from adjacent cells (not diagonal)
        
        Company: Microsoft, Amazon, Facebook
        Difficulty: Medium
        Time: O(m*n*4^L), Space: O(L) where L is word length
        
        Example: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
                 word = "ABCCED" → True
        """
        rows, cols = len(board), len(board[0])
        
        def backtrack(row: int, col: int, index: int, path: Set[Tuple[int, int]]) -> bool:
            print(f"{'  ' * index}Checking ({row}, {col}) for '{word[index]}', path: {path}")
            
            # BASE CASE: Found complete word
            if index == len(word):
                print(f"{'  ' * index}✓ Found word: {word}")
                return True
            
            # CONSTRAINT CHECKS
            if (row < 0 or row >= rows or 
                col < 0 or col >= cols or 
                board[row][col] != word[index] or 
                (row, col) in path):
                return False
            
            # MAKE CHOICE: Add current cell to path
            path.add((row, col))
            
            # TRY: All four directions
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if backtrack(new_row, new_col, index + 1, path):
                    return True
            
            # BACKTRACK: Remove current cell from path
            path.remove((row, col))
            print(f"{'  ' * index}← Backtracking from ({row}, {col})")
            
            return False
        
        # Try starting from each cell
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == word[0]:  # Potential starting point
                    print(f"\nTrying to start word search from ({i}, {j})")
                    if backtrack(i, j, 0, set()):
                        return True
        
        return False
    
    def restore_ip_addresses(self, s: str) -> List[str]:
        """
        Restore IP Addresses
        
        Generate all valid IP addresses from string of digits
        
        Company: Google, Facebook, Amazon
        Difficulty: Medium
        Time: O(1) - constant since max 12 digits, Space: O(1)
        
        Example: "25525511135" → ["255.255.11.135", "255.255.111.35"]
        """
        result = []
        
        def is_valid_part(part: str) -> bool:
            """Check if string is valid IP part (0-255, no leading zeros)"""
            if not part or len(part) > 3:
                return False
            if len(part) > 1 and part[0] == '0':  # No leading zeros except "0"
                return False
            return 0 <= int(part) <= 255
        
        def backtrack(start: int, parts: List[str]) -> None:
            print(f"{'  ' * len(parts)}Building IP: {'.'.join(parts)}, start: {start}")
            
            # BASE CASE: Have 4 parts
            if len(parts) == 4:
                if start == len(s):  # Used all characters
                    ip = '.'.join(parts)
                    result.append(ip)
                    print(f"{'  ' * len(parts)}✓ Valid IP: {ip}")
                return
            
            # TRY: Different lengths for next part (1, 2, or 3 digits)
            for length in range(1, 4):
                if start + length > len(s):
                    break
                
                part = s[start:start + length]
                
                if is_valid_part(part):
                    # MAKE CHOICE: Add part
                    parts.append(part)
                    
                    # RECURSE
                    backtrack(start + length, parts)
                    
                    # BACKTRACK: Remove part
                    parts.pop()
        
        print(f"Generating valid IP addresses from '{s}':")
        backtrack(0, [])
        
        print(f"Found {len(result)} valid IP addresses")
        return result
    
    # ==========================================
    # 3. ADVANCED INTERVIEW PROBLEMS
    # ==========================================
    
    def n_queens(self, n: int) -> List[List[str]]:
        """
        N-Queens
        
        Place n queens on n×n chessboard so that no two queens attack each other
        
        Company: Google, Amazon, Microsoft, Apple
        Difficulty: Hard
        Time: O(n!), Space: O(n)
        
        Example: n = 4 → [[".Q..","...Q","Q...","..Q."], ["..Q.","Q...","...Q",".Q.."]]
        """
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        
        def is_safe(row: int, col: int) -> bool:
            """Check if queen can be placed at (row, col)"""
            # Check column
            for i in range(row):
                if board[i][col] == 'Q':
                    return False
            
            # Check diagonal (top-left to bottom-right)
            i, j = row - 1, col - 1
            while i >= 0 and j >= 0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            
            # Check diagonal (top-right to bottom-left)
            i, j = row - 1, col + 1
            while i >= 0 and j < n:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            
            return True
        
        def backtrack(row: int) -> None:
            print(f"{'  ' * row}Placing queen in row {row}")
            
            # BASE CASE: All queens placed
            if row == n:
                # Convert board to string representation
                solution = [''.join(row) for row in board]
                solutions.append(solution)
                print(f"{'  ' * row}✓ Found solution")
                return
            
            # TRY: Each column in current row
            for col in range(n):
                if is_safe(row, col):
                    print(f"{'  ' * row}Trying queen at ({row}, {col})")
                    
                    # MAKE CHOICE: Place queen
                    board[row][col] = 'Q'
                    
                    # RECURSE: Move to next row
                    backtrack(row + 1)
                    
                    # BACKTRACK: Remove queen
                    board[row][col] = '.'
                    print(f"{'  ' * row}← Backtracking from ({row}, {col})")
        
        print(f"Solving {n}-Queens problem:")
        backtrack(0)
        
        print(f"Found {len(solutions)} solutions")
        return solutions
    
    def solve_sudoku(self, board: List[List[str]]) -> None:
        """
        Sudoku Solver
        
        Solve 9×9 Sudoku puzzle (modify board in-place)
        
        Company: Amazon, Microsoft, Google, Uber
        Difficulty: Hard
        Time: O(9^(n*n)), Space: O(1)
        
        Example: Modifies the input board to solve the Sudoku puzzle
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
            
            # Check 3×3 box
            start_row = (row // 3) * 3
            start_col = (col // 3) * 3
            
            for i in range(start_row, start_row + 3):
                for j in range(start_col, start_col + 3):
                    if board[i][j] == num:
                        return False
            
            return True
        
        def find_empty_cell() -> Optional[Tuple[int, int]]:
            """Find next empty cell"""
            for i in range(9):
                for j in range(9):
                    if board[i][j] == '.':
                        return (i, j)
            return None
        
        def backtrack() -> bool:
            self.call_count += 1
            
            # Find next empty cell
            empty_cell = find_empty_cell()
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
                    
                    # RECURSE: Try to solve rest
                    if backtrack():
                        return True
                    
                    # BACKTRACK: Remove number
                    board[row][col] = '.'
                    print(f"  ← Backtracking from ({row}, {col})")
            
            return False
        
        print("Solving Sudoku puzzle:")
        self.call_count = 0
        backtrack()
        print(f"Function calls: {self.call_count}")
    
    # ==========================================
    # 4. INTERVIEW STRATEGY AND TIPS
    # ==========================================
    
    def interview_approach_framework(self) -> None:
        """
        Framework for approaching backtracking problems in interviews
        """
        print("=== BACKTRACKING INTERVIEW APPROACH FRAMEWORK ===")
        print()
        
        print("🎯 STEP 1: PROBLEM ANALYSIS")
        print("• Identify if it's a backtracking problem:")
        print("  - Need to find ALL solutions")
        print("  - Explore multiple possibilities")
        print("  - Build solution incrementally")
        print("  - Can undo choices (backtrack)")
        print()
        
        print("🎯 STEP 2: DEFINE COMPONENTS")
        print("• State representation: How to encode current state")
        print("• Choices: What options are available at each step")
        print("• Constraints: What makes a choice valid/invalid")
        print("• Goal test: When is solution complete")
        print("• Backtrack condition: When to undo and try next option")
        print()
        
        print("🎯 STEP 3: IMPLEMENT TEMPLATE")
        print("def backtrack(state, partial_solution):")
        print("    if is_goal(state):")
        print("        add_solution(partial_solution)")
        print("        return")
        print("    ")
        print("    for choice in get_choices(state):")
        print("        if is_valid(choice, state):")
        print("            make_choice(choice, state, partial_solution)")
        print("            backtrack(new_state, partial_solution)")
        print("            unmake_choice(choice, state, partial_solution)")
        print()
        
        print("🎯 STEP 4: OPTIMIZATION")
        print("• Add pruning conditions")
        print("• Use efficient data structures")
        print("• Consider memoization if applicable")
        print("• Order choices by likelihood of success")
        print()
        
        print("🎯 STEP 5: TEST AND ANALYZE")
        print("• Test with given examples")
        print("• Consider edge cases")
        print("• Analyze time and space complexity")
        print("• Discuss potential optimizations")
    
    def common_interview_mistakes(self) -> None:
        """
        Common mistakes in backtracking interview problems
        """
        print("=== COMMON BACKTRACKING INTERVIEW MISTAKES ===")
        print()
        
        print("❌ MISTAKE 1: Forgetting to backtrack")
        print("• Not undoing choices after recursive call")
        print("• Solution: Always pair make_choice with unmake_choice")
        print()
        
        print("❌ MISTAKE 2: Incorrect base case")
        print("• Missing edge cases or wrong termination condition")
        print("• Solution: Carefully think through when to stop recursion")
        print()
        
        print("❌ MISTAKE 3: Inefficient constraint checking")
        print("• Checking constraints too late or redundantly")
        print("• Solution: Check constraints as early as possible")
        print()
        
        print("❌ MISTAKE 4: Not handling duplicates")
        print("• Generating duplicate solutions")
        print("• Solution: Sort input and skip duplicates systematically")
        print()
        
        print("❌ MISTAKE 5: Poor space management")
        print("• Creating unnecessary copies of data structures")
        print("• Solution: Modify in-place and backtrack properly")
        print()
        
        print("❌ MISTAKE 6: Missing optimizations")
        print("• Not adding obvious pruning conditions")
        print("• Solution: Think about when you can eliminate branches early")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_interview_problems():
    """Demonstrate key backtracking interview problems"""
    print("=== BACKTRACKING INTERVIEW PROBLEMS DEMONSTRATION ===\n")
    
    problems = BacktrackingInterviewProblems()
    
    # 1. Classic problems
    print("=== CLASSIC INTERVIEW PROBLEMS ===")
    
    # Letter combinations
    print("1. Letter Combinations of Phone Number ('23'):")
    combinations = problems.letter_combinations_phone_number('23')
    print(f"Result: {combinations}")
    print()
    
    # Generate parentheses
    print("2. Generate Parentheses (n=3):")
    parentheses = problems.generate_parentheses(3)
    print(f"Result: {parentheses}")
    print()
    
    # Combination sum
    print("3. Combination Sum ([2,3,6,7], target=7):")
    comb_sum = problems.combination_sum([2, 3, 6, 7], 7)
    print(f"Result: {comb_sum}")
    print()
    
    # Subsets
    print("4. Subsets ([1,2,3]):")
    subsets = problems.subsets([1, 2, 3])
    print(f"Result: {subsets}")
    print()
    
    # 2. String and array problems
    print("=== STRING AND ARRAY PROBLEMS ===")
    
    # Palindrome partitioning
    print("1. Palindrome Partitioning ('aab'):")
    palindrome_parts = problems.palindrome_partitioning('aab')
    print(f"Result: {palindrome_parts}")
    print()
    
    # Word search
    print("2. Word Search:")
    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]
    word_found = problems.word_search(board, 'ABCCED')
    print(f"Word 'ABCCED' found: {word_found}")
    print()
    
    # Restore IP addresses
    print("3. Restore IP Addresses ('25525511135'):")
    ip_addresses = problems.restore_ip_addresses('25525511135')
    print(f"Result: {ip_addresses}")
    print()
    
    # 3. Advanced problems
    print("=== ADVANCED INTERVIEW PROBLEMS ===")
    
    # N-Queens
    print("1. N-Queens (n=4):")
    queens_solutions = problems.n_queens(4)
    print(f"Found {len(queens_solutions)} solutions")
    if queens_solutions:
        print("First solution:")
        for row in queens_solutions[0]:
            print(f"  {row}")
    print()
    
    # 4. Interview approach
    problems.interview_approach_framework()
    print()
    
    # 5. Common mistakes
    problems.common_interview_mistakes()

if __name__ == "__main__":
    demonstrate_interview_problems()
    
    print("\n=== INTERVIEW SUCCESS STRATEGY ===")
    
    print("\n🎯 BEFORE THE INTERVIEW:")
    print("• Practice 20-30 backtracking problems")
    print("• Master the basic template")
    print("• Understand time/space complexity patterns")
    print("• Learn common optimization techniques")
    
    print("\n💡 DURING THE INTERVIEW:")
    print("1. Clarify requirements and constraints")
    print("2. Start with brute force approach")
    print("3. Implement basic backtracking solution")
    print("4. Add optimizations and pruning")
    print("5. Test with examples and edge cases")
    print("6. Analyze and discuss complexity")
    
    print("\n📚 KEY PROBLEM TYPES TO MASTER:")
    print("• Combinations and permutations")
    print("• Subset generation")
    print("• String partitioning and generation")
    print("• Grid and matrix problems")
    print("• Constraint satisfaction (N-Queens, Sudoku)")
    print("• Game theory and decision trees")
    
    print("\n⚡ OPTIMIZATION TECHNIQUES:")
    print("• Early constraint checking")
    print("• Pruning impossible branches")
    print("• Sorting for better pruning")
    print("• Using efficient data structures")
    print("• Memoization when applicable")
    
    print("\n🏆 INTERVIEW TIPS:")
    print("• Think out loud - explain your thought process")
    print("• Start simple, then optimize")
    print("• Draw examples and trace through execution")
    print("• Discuss trade-offs between time and space")
    print("• Be ready to implement optimizations")
    print("• Practice coding without IDE assistance")
