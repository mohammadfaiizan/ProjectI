"""
🎮 BIT MASKING FOR STATES & GAMES
=================================

This module covers bitmasking techniques for state-based and game problems.
Bitmasking is powerful for representing and manipulating game states efficiently.

Topics Covered:
1. N-Queens using Bitmask
2. Sudoku Solver Optimization
3. Knight Dialer
4. Lights On/Off Puzzles

Author: Interview Preparation Collection
LeetCode Problems: 51, 37, 935, 1349, 1434, 464
"""

class NQueensBitmask:
    """N-Queens problem using bitmask optimization."""
    
    @staticmethod
    def solve_n_queens(n: int) -> list:
        """
        Solve N-Queens problem using bitmask for efficient conflict checking.
        
        Args:
            n: Size of chessboard (n x n)
            
        Returns:
            List of all valid board configurations
            
        Time: O(n!), Space: O(n)
        LeetCode: 51
        """
        def backtrack(row, cols, diag1, diag2, board):
            if row == n:
                solutions.append([''.join(row) for row in board])
                return
            
            # Find available positions using bitmasking
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            
            while available:
                # Get rightmost available position
                col = available & (-available)
                col_idx = col.bit_length() - 1
                
                # Place queen
                board[row][col_idx] = 'Q'
                
                # Recursive call with updated bitmasks
                backtrack(
                    row + 1,
                    cols | col,                    # Block column
                    (diag1 | col) << 1,          # Block diagonal (\)
                    (diag2 | col) >> 1,          # Block diagonal (/)
                    board
                )
                
                # Backtrack
                board[row][col_idx] = '.'
                
                # Remove this position from available
                available &= available - 1
        
        solutions = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        backtrack(0, 0, 0, 0, board)
        return solutions
    
    @staticmethod
    def count_n_queens(n: int) -> int:
        """
        Count total number of N-Queens solutions.
        
        Args:
            n: Size of chessboard
            
        Returns:
            Total number of solutions
            
        Time: O(n!), Space: O(n)
        LeetCode: 52
        """
        def backtrack(row, cols, diag1, diag2):
            if row == n:
                return 1
            
            count = 0
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            
            while available:
                col = available & (-available)
                
                count += backtrack(
                    row + 1,
                    cols | col,
                    (diag1 | col) << 1,
                    (diag2 | col) >> 1
                )
                
                available &= available - 1
            
            return count
        
        return backtrack(0, 0, 0, 0)
    
    @staticmethod
    def n_queens_first_solution(n: int) -> list:
        """
        Find first valid N-Queens solution quickly.
        
        Args:
            n: Size of chessboard
            
        Returns:
            First valid solution or empty list if none exists
            
        Time: O(n!), Space: O(n)
        """
        def backtrack(row, cols, diag1, diag2, board):
            if row == n:
                return True
            
            available = ((1 << n) - 1) & ~(cols | diag1 | diag2)
            
            while available:
                col = available & (-available)
                col_idx = col.bit_length() - 1
                
                board[row][col_idx] = 'Q'
                
                if backtrack(
                    row + 1,
                    cols | col,
                    (diag1 | col) << 1,
                    (diag2 | col) >> 1,
                    board
                ):
                    return True
                
                board[row][col_idx] = '.'
                available &= available - 1
            
            return False
        
        board = [['.' for _ in range(n)] for _ in range(n)]
        if backtrack(0, 0, 0, 0, board):
            return [''.join(row) for row in board]
        return []


class SudokuBitmaskSolver:
    """Sudoku solver with bitmask optimization."""
    
    @staticmethod
    def solve_sudoku(board: list) -> bool:
        """
        Solve Sudoku using bitmask for efficient constraint checking.
        
        Args:
            board: 9x9 Sudoku board (0 or '.' for empty cells)
            
        Returns:
            True if solved, False if no solution
            
        Time: O(9^(empty_cells)), Space: O(1)
        LeetCode: 37
        """
        # Initialize bitmasks for constraints
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        empty_cells = []
        
        # Setup initial bitmasks and collect empty cells
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 or board[r][c] == '.':
                    empty_cells.append((r, c))
                else:
                    num = int(board[r][c])
                    bit = 1 << (num - 1)
                    rows[r] |= bit
                    cols[c] |= bit
                    boxes[3 * (r // 3) + (c // 3)] |= bit
        
        def backtrack(idx):
            if idx == len(empty_cells):
                return True
            
            r, c = empty_cells[idx]
            box_idx = 3 * (r // 3) + (c // 3)
            
            # Find available numbers using bitmask
            used = rows[r] | cols[c] | boxes[box_idx]
            available = ((1 << 9) - 1) & ~used
            
            while available:
                # Get next available number
                bit = available & (-available)
                num = bit.bit_length()
                
                # Place number
                board[r][c] = str(num)
                rows[r] |= bit
                cols[c] |= bit
                boxes[box_idx] |= bit
                
                if backtrack(idx + 1):
                    return True
                
                # Backtrack
                board[r][c] = 0
                rows[r] &= ~bit
                cols[c] &= ~bit
                boxes[box_idx] &= ~bit
                
                available &= available - 1
            
            return False
        
        return backtrack(0)
    
    @staticmethod
    def count_valid_sudoku_solutions(board: list) -> int:
        """
        Count number of valid Sudoku solutions.
        
        Args:
            board: Partially filled Sudoku board
            
        Returns:
            Number of valid solutions
            
        Time: O(9^(empty_cells)), Space: O(1)
        """
        # Similar to solve_sudoku but counts all solutions
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        empty_cells = []
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
                else:
                    num = board[r][c]
                    bit = 1 << (num - 1)
                    rows[r] |= bit
                    cols[c] |= bit
                    boxes[3 * (r // 3) + (c // 3)] |= bit
        
        def count_solutions(idx):
            if idx == len(empty_cells):
                return 1
            
            r, c = empty_cells[idx]
            box_idx = 3 * (r // 3) + (c // 3)
            
            used = rows[r] | cols[c] | boxes[box_idx]
            available = ((1 << 9) - 1) & ~used
            
            count = 0
            while available:
                bit = available & (-available)
                num = bit.bit_length()
                
                board[r][c] = num
                rows[r] |= bit
                cols[c] |= bit
                boxes[box_idx] |= bit
                
                count += count_solutions(idx + 1)
                
                board[r][c] = 0
                rows[r] &= ~bit
                cols[c] &= ~bit
                boxes[box_idx] &= ~bit
                
                available &= available - 1
            
            return count
        
        return count_solutions(0)


class KnightDialer:
    """Knight dialer problem using bitmask state representation."""
    
    @staticmethod
    def knight_dialer_moves(n: int) -> int:
        """
        Count distinct phone numbers of length n that can be dialed.
        Knight moves like in chess on a phone keypad.
        
        Args:
            n: Length of phone number
            
        Returns:
            Number of distinct phone numbers
            
        Time: O(n), Space: O(1)
        LeetCode: 935
        """
        MOD = 10**9 + 7
        
        # Define possible moves for each digit (knight moves)
        moves = {
            0: [4, 6],
            1: [6, 8],
            2: [7, 9],
            3: [4, 8],
            4: [0, 3, 9],
            5: [],  # No valid moves from 5
            6: [0, 1, 7],
            7: [2, 6],
            8: [1, 3],
            9: [2, 4]
        }
        
        # DP approach: dp[i] = count of sequences of length n starting at digit i
        prev_dp = [1] * 10
        
        for _ in range(n - 1):
            curr_dp = [0] * 10
            
            for digit in range(10):
                for next_digit in moves[digit]:
                    curr_dp[next_digit] = (curr_dp[next_digit] + prev_dp[digit]) % MOD
            
            prev_dp = curr_dp
        
        return sum(prev_dp) % MOD
    
    @staticmethod
    def knight_dialer_bitmask(n: int) -> int:
        """
        Knight dialer using bitmask representation for optimization.
        
        Args:
            n: Length of phone number
            
        Returns:
            Number of distinct phone numbers
            
        Time: O(n), Space: O(1)
        """
        MOD = 10**9 + 7
        
        # Represent transitions as bitmasks
        # transitions[i] = bitmask of digits reachable from digit i
        transitions = [
            0b1010000,   # 0 -> 4, 6
            0b101000000, # 1 -> 6, 8
            0b110000000, # 2 -> 7, 9
            0b100010000, # 3 -> 4, 8
            0b1000001001, # 4 -> 0, 3, 9
            0b0,         # 5 -> none
            0b1000011,   # 6 -> 0, 1, 7
            0b1000100,   # 7 -> 2, 6
            0b100001010, # 8 -> 1, 3
            0b10000100   # 9 -> 2, 4
        ]
        
        # Use bitmask DP
        dp = [1] * 10
        
        for _ in range(n - 1):
            new_dp = [0] * 10
            
            for digit in range(10):
                transition_mask = transitions[digit]
                
                # Add count to all reachable digits
                for next_digit in range(10):
                    if transition_mask & (1 << next_digit):
                        new_dp[next_digit] = (new_dp[next_digit] + dp[digit]) % MOD
            
            dp = new_dp
        
        return sum(dp) % MOD


class LightsPuzzle:
    """Lights on/off puzzle using bitmask state manipulation."""
    
    @staticmethod
    def minimum_flips_to_turn_off_all_lights(lights: str) -> int:
        """
        Minimum button presses to turn off all lights.
        Each button toggles specific lights pattern.
        
        Args:
            lights: String of '0' (off) and '1' (on)
            
        Returns:
            Minimum flips needed, -1 if impossible
            
        Time: O(2^b) where b is number of buttons, Space: O(2^n)
        """
        n = len(lights)
        target = 0  # All lights off
        initial = int(lights, 2)  # Convert binary string to int
        
        # Define button effects (example: each button toggles specific pattern)
        buttons = []
        
        # Button 0: toggles all lights
        buttons.append((1 << n) - 1)
        
        # Button 1: toggles odd positions
        odd_mask = 0
        for i in range(n):
            if i % 2 == 1:
                odd_mask |= (1 << i)
        buttons.append(odd_mask)
        
        # Button 2: toggles even positions
        even_mask = 0
        for i in range(n):
            if i % 2 == 0:
                even_mask |= (1 << i)
        buttons.append(even_mask)
        
        # BFS to find minimum flips
        from collections import deque
        
        queue = deque([(initial, 0)])
        visited = {initial}
        
        while queue:
            state, flips = queue.popleft()
            
            if state == target:
                return flips
            
            # Try each button
            for button_mask in buttons:
                new_state = state ^ button_mask
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, flips + 1))
        
        return -1  # Impossible
    
    @staticmethod
    def lights_puzzle_bulbs_and_switches(n: int, presses: int) -> int:
        """
        Count different light states after exactly 'presses' button presses.
        
        Args:
            n: Number of bulbs
            presses: Number of button presses
            
        Returns:
            Number of possible light states
            
        Time: O(1), Space: O(1)
        LeetCode: 672
        """
        # For n bulbs, there are limited patterns due to button effects
        if presses == 0:
            return 1
        
        if n == 1:
            return 2 if presses >= 1 else 1
        
        if n == 2:
            if presses == 1:
                return 4
            return 7 if presses >= 2 else 4
        
        # For n >= 3
        if presses == 1:
            return 4
        elif presses == 2:
            return 7
        else:
            return 7  # Maximum distinct states
    
    @staticmethod
    def flip_game_can_win(current_state: str) -> bool:
        """
        Determine if current player can guarantee a win in flip game.
        
        Args:
            current_state: String of '+' and '-'
            
        Returns:
            True if current player can win
            
        Time: O(2^n), Space: O(2^n)
        LeetCode: 294
        """
        def can_win(state):
            # Try all possible moves
            for i in range(len(state) - 1):
                if state[i] == '+' and state[i + 1] == '+':
                    # Make move: flip "++" to "--"
                    new_state = state[:i] + '--' + state[i + 2:]
                    
                    # If opponent cannot win from new state, current player wins
                    if not can_win(new_state):
                        return True
            
            return False  # No winning move found
        
        return can_win(current_state)


class GameStateBitmask:
    """General game state representation using bitmasks."""
    
    @staticmethod
    def tic_tac_toe_winner(moves: list) -> str:
        """
        Determine Tic-Tac-Toe winner using bitmask representation.
        
        Args:
            moves: List of [row, col] moves alternating between X and O
            
        Returns:
            'X', 'O', 'Draw', or 'Pending'
            
        Time: O(moves), Space: O(1)
        LeetCode: 1275
        """
        # Represent 3x3 board as 9-bit mask
        x_mask = 0
        o_mask = 0
        
        # Winning patterns (rows, columns, diagonals)
        win_patterns = [
            0b111000000,  # Top row
            0b000111000,  # Middle row
            0b000000111,  # Bottom row
            0b100100100,  # Left column
            0b010010010,  # Middle column
            0b001001001,  # Right column
            0b100010001,  # Main diagonal
            0b001010100   # Anti-diagonal
        ]
        
        for i, (row, col) in enumerate(moves):
            bit_pos = row * 3 + col
            
            if i % 2 == 0:  # X's turn
                x_mask |= (1 << bit_pos)
                
                # Check if X wins
                for pattern in win_patterns:
                    if (x_mask & pattern) == pattern:
                        return 'X'
            else:  # O's turn
                o_mask |= (1 << bit_pos)
                
                # Check if O wins
                for pattern in win_patterns:
                    if (o_mask & pattern) == pattern:
                        return 'O'
        
        # Check if board is full
        if len(moves) == 9:
            return 'Draw'
        
        return 'Pending'
    
    @staticmethod
    def connect_four_winner(grid: list) -> str:
        """
        Check Connect Four winner using bitmask techniques.
        
        Args:
            grid: 2D grid with player moves
            
        Returns:
            Winner ('X', 'O') or 'None'
            
        Time: O(rows * cols), Space: O(1)
        """
        rows, cols = len(grid), len(grid[0])
        
        def check_winner(player):
            # Convert player positions to bitmask (simplified for demo)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == player:
                        # Check 4 directions: horizontal, vertical, diagonal
                        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                        
                        for dr, dc in directions:
                            count = 1
                            
                            # Check forward direction
                            nr, nc = r + dr, c + dc
                            while (0 <= nr < rows and 0 <= nc < cols and 
                                   grid[nr][nc] == player):
                                count += 1
                                nr, nc = nr + dr, nc + dc
                            
                            # Check backward direction
                            nr, nc = r - dr, c - dc
                            while (0 <= nr < rows and 0 <= nc < cols and 
                                   grid[nr][nc] == player):
                                count += 1
                                nr, nc = nr - dr, nc - dc
                            
                            if count >= 4:
                                return True
            return False
        
        if check_winner('X'):
            return 'X'
        elif check_winner('O'):
            return 'O'
        else:
            return 'None'


class StateGameDemo:
    """Demonstration of state and game bitmask techniques."""
    
    @staticmethod
    def demonstrate_n_queens():
        """Demonstrate N-Queens bitmask optimization."""
        print("=== N-QUEENS BITMASK ===")
        
        # Solve 4-Queens
        n = 4
        solutions = NQueensBitmask.solve_n_queens(n)
        count = NQueensBitmask.count_n_queens(n)
        first_solution = NQueensBitmask.n_queens_first_solution(n)
        
        print(f"{n}-Queens problem:")
        print(f"Total solutions: {count}")
        print(f"First solution:")
        for row in first_solution:
            print(f"  {row}")
        
        # Compare with 8-Queens count
        n8_count = NQueensBitmask.count_n_queens(8)
        print(f"\n8-Queens total solutions: {n8_count}")
    
    @staticmethod
    def demonstrate_sudoku_solver():
        """Demonstrate Sudoku bitmask solver."""
        print("\n=== SUDOKU BITMASK SOLVER ===")
        
        # Example Sudoku puzzle
        puzzle = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
        
        print("Original puzzle:")
        for row in puzzle:
            print(row)
        
        solved = SudokuBitmaskSolver.solve_sudoku([row[:] for row in puzzle])
        
        if solved:
            print("\nSolved puzzle:")
            for row in puzzle:
                print(row)
        else:
            print("\nNo solution found")
    
    @staticmethod
    def demonstrate_knight_dialer():
        """Demonstrate Knight Dialer problem."""
        print("\n=== KNIGHT DIALER ===")
        
        for n in [1, 2, 3, 4]:
            count_regular = KnightDialer.knight_dialer_moves(n)
            count_bitmask = KnightDialer.knight_dialer_bitmask(n)
            
            print(f"Phone numbers of length {n}: {count_regular}")
            print(f"  (Bitmask method: {count_bitmask})")
    
    @staticmethod
    def demonstrate_lights_puzzle():
        """Demonstrate lights puzzle problems."""
        print("\n=== LIGHTS PUZZLE ===")
        
        # Example lights state
        lights = "1011"
        min_flips = LightsPuzzle.minimum_flips_to_turn_off_all_lights(lights)
        print(f"Lights state: {lights}")
        print(f"Minimum flips to turn off all: {min_flips}")
        
        # Bulbs and switches
        for n in [1, 2, 3]:
            for presses in [0, 1, 2, 3]:
                states = LightsPuzzle.lights_puzzle_bulbs_and_switches(n, presses)
                print(f"  {n} bulbs, {presses} presses: {states} states")
        
        # Flip game
        game_state = "++++"
        can_win = LightsPuzzle.flip_game_can_win(game_state)
        print(f"\nFlip game state: {game_state}")
        print(f"Current player can win: {can_win}")
    
    @staticmethod
    def demonstrate_game_states():
        """Demonstrate general game state bitmasks."""
        print("\n=== GAME STATE BITMASKS ===")
        
        # Tic-Tac-Toe
        moves = [[0,0],[2,0],[1,1],[1,0],[2,1]]
        winner = GameStateBitmask.tic_tac_toe_winner(moves)
        print(f"Tic-Tac-Toe moves: {moves}")
        print(f"Winner: {winner}")
        
        # Connect Four example (simplified)
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 'X', 0, 0, 0],
            [0, 0, 'X', 'X', 0, 0, 0],
            [0, 'X', 'O', 'X', 0, 0, 0],
            ['X', 'O', 'O', 'O', 0, 0, 0]
        ]
        
        connect_winner = GameStateBitmask.connect_four_winner(grid)
        print(f"\nConnect Four winner: {connect_winner}")


def performance_analysis():
    """Analyze performance benefits of bitmask techniques."""
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    print("Bitmask Optimization Benefits:")
    print("1. N-Queens: O(1) conflict checking vs O(n) traditional")
    print("2. Sudoku: O(1) constraint validation vs O(27) checking")
    print("3. Game states: Compact representation and fast operations")
    print("4. State transitions: Efficient using bitwise operations")
    
    print("\nSpace Complexity Improvements:")
    print("• Traditional board: O(n²) space")
    print("• Bitmask representation: O(log n) space per constraint")
    print("• State caching: More efficient with integer keys")


def practical_applications():
    """Discuss practical applications of state bitmasks."""
    print("\n=== PRACTICAL APPLICATIONS ===")
    
    print("Real-world applications:")
    print("1. Game AI: Efficient state evaluation and minimax")
    print("2. Puzzle solvers: Constraint satisfaction problems")
    print("3. Logic circuits: Boolean function optimization")
    print("4. Chess engines: Position evaluation and move generation")
    print("5. Cryptography: State machines and permutation tracking")
    print("6. Computer graphics: Collision detection optimization")


if __name__ == "__main__":
    # Run all demonstrations
    demo = StateGameDemo()
    
    demo.demonstrate_n_queens()
    demo.demonstrate_sudoku_solver()
    demo.demonstrate_knight_dialer()
    demo.demonstrate_lights_puzzle()
    demo.demonstrate_game_states()
    
    performance_analysis()
    practical_applications()
    
    print("\n🎯 Key State & Game Bitmask Patterns:")
    print("1. Constraint representation: Use bits for conflict tracking")
    print("2. State compression: Represent complex states as integers")
    print("3. Transition optimization: Bitwise operations for state changes")
    print("4. Pruning: Fast validity checking with bitmasks")
    print("5. Memoization: Integer states for efficient caching")
    print("6. Pattern matching: Bitwise AND/OR for game pattern detection") 